use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::*;

use super::Color3;

/// Matches the fixed-size `array<Light, MAX_LIGHTS>` declared in the mesh/solid
/// fragment shaders — keep these in sync.
///
/// SIXTEEN, not eight (2026-10-02). test_room's seven lamps, the sky's sun and
/// the player's flashlight are nine already: with eight, lighting the torch
/// dropped the brick wall wash's light from every surface, and the torch's
/// bounce never made the cut at all. Only the lights a frame actually has are
/// walked -- every loop runs to `count`, and each lamp that cannot reach a
/// pixel is skipped there -- and the array's size costs nothing of its own: the
/// driver reads these blocks through memory whatever their size (exp63).
pub const MAX_LIGHTS: usize = 16;

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
        shadow_near: None,
        source_radius: 0.0,
        in_level_bake: true,
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
    // THE LIT SURFACES' LIGHTS FIRST, each kind in its own order: the scene
    // readers shade those from the front of the list on their own
    // (`GpuLights::surface_lights`). Only the order of what fits changes,
    // never which lights fit.
    idx.sort_by_key(|&i| !lights[i].is_surface_light());
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
    /// Where a spot's shadow map begins, metres from the light. `None` is a
    /// fixture's [`super::shadow::SPOT_SHADOW_NEAR`], which leaves the lamp's
    /// own housing out of its map; a light with nothing round it -- the
    /// player's flashlight, whose glass is the front of the torch -- gives its
    /// own, so a hand a few centimetres in front of it still casts. See
    /// `shadow::spot_shadow_matrices`, which keeps the shadow's bias in step.
    /// Past the light's range, it casts no shadow at all: see
    /// [`Light::casts_shadow`].
    pub shadow_near: Option<f32>,
    /// HOW WIDE ITS SOURCE IS, metres: the radius of the patch a light that
    /// stands for a lit SURFACE gives its light off from -- a flashlight's
    /// bounce ([`SURFACE_LIGHT`]). Its light falls off as `1 / (d^2 + r^2)`,
    /// what a disc that wide gives on its axis, so nothing beside the patch is
    /// lit more than the patch could light it. A point standing for a metre of
    /// lit wall lit the stone a few centimetres off at hundreds of times what
    /// it gave a metre away (headset, 2026-10-02: "a bright spot that looked
    /// almost like what it looks like when you focus a light through a
    /// magnifying glass"). 0 for a bulb, whose near field the lights block
    /// holds at its `LAMP_RADIUS`. Only a light that casts no shadow carries
    /// it to the GPU (`pack_lights`): a lamp with a shadow is a bulb.
    pub source_radius: f32,
    /// WHETHER THE LEVEL'S BAKE SAW IT: its light is in the probe photographs
    /// and the models' cards already -- every light a level authors, whatever
    /// its mode. One the game adds while it runs -- the player's flashlight,
    /// its bounce -- is not, and a reflection of a model on cards takes its
    /// light there (`PROBE_CARD_RELIT`): `position.w` -1 on the GPU.
    pub in_level_bake: bool,
}

impl Light {
    /// Whether this light can cast a shadow: not when its shadow would begin
    /// past where its light ends. Such a light takes no spot shadow slot and
    /// no characters' tile -- a flashlight's BOUNCE, which stands for a lit
    /// patch of wall: a source that wide casts no shadow sharp enough to map,
    /// and its half-space cone no shadow map can hold.
    pub fn casts_shadow(&self) -> bool {
        self.shadow_near.is_none_or(|near| near < self.range)
    }

    /// Whether the scene readers shade this light ON ITS OWN, as a lit
    /// surface's (`surface_lights` in the lights block) rather than through
    /// their lamp loop: it casts no shadow, so the shader gives it no
    /// highlight either ([`SURFACE_LIGHT`]), and a spot's edge spans more
    /// cosine than [`SURFACE_LIGHT_MIN_BAND`]. A flashlight's bounce.
    /// [`rank_for_budget_indices`] puts these at the front of the list, and
    /// `pack_lights` counts them there.
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

    /// The cosines of a spot's outer and inner half-angles, as the shader's
    /// `spot_cone` takes them.
    pub fn cone_cosines(&self) -> (f32, f32) {
        let cos_outer = (self.cone_angle_deg.to_radians() * 0.5).cos();
        // Clamped above cos_outer so an inner angle authored wider than the
        // outer one cannot invert the gradient (or divide by ~zero) and
        // turn the beam inside out.
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
    /// x = range, y = cos(outer half-angle), z = kind (0 = point, 1 = spot),
    /// w = which spot shadow layer this light casts into, or -1 for none --
    /// [`SURFACE_LIGHT`] for a light that stands for a lit surface.
    ///
    /// `direction.w` carries cos(inner half-angle) -- it was padding, and the
    /// inner angle is measured against that very direction.
    ///
    /// On the LIGHT rather than in the camera uniform, because it is a property
    /// of the light. The camera used to carry a single "flashlight index",
    /// which by construction could only ever name one shadow-casting spot.
    params: [f32; 4],
}

/// A SURFACE A LIVE LAMP'S BEAM IS KNOWN TO LIGHT, in the PLAYER's frame like
/// every light: the plane `normal . p == offset`, `normal` facing the lamp, and
/// its albedo, linear RGB -- with the frame of its POOL MAP, the beam's light
/// on that plane as seen from the glass: where the glass is (`lens`), the
/// map's middle (`forward`, the beam's axis) and across (`right`), and how far
/// off the axis it reaches, as a tangent (`tan_half`; 0 for none). A
/// flashlight's rays find these where they land
/// (`quest_app::flashlight_bounce`). Each frame the lamps the bake never saw
/// light every such plane into its map (`pool_cards`, `pool_map_light` in the
/// shader); a reflection meeting the plane reads it there, so the torch's
/// pool on a wall shows in the polished floor (`probe_surface_relit`) -- the
/// photographs hold only the level's own light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LitSurface {
    pub normal: Vec3,
    pub offset: f32,
    pub albedo: Vec3,
    pub lens: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub tan_half: f32,
}

/// How many [`LitSurface`]s the shaders take: a pool map each, in the card
/// atlas's rows kept for them (`pool_cards`).
pub const MAX_LIT_SURFACES: usize = 6;

/// The vec4s a surface takes in the lights block. See [`pack_lit_surfaces`].
pub const LIT_SURFACE_VEC4S: usize = 5;

/// The surfaces as the shader reads them, five vec4s each: the plane; the
/// glass and the map's reach (0 ends the list); the beam's axis and the
/// card-atlas texel row the maps start at; the map's across; the albedo. NONE
/// without that row: then no map is made this frame, and none may be read.
/// See `lit_surface_at` and `pool_map_light` in the shader.
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
        out[at + 3] = [s.right.x, s.right.y, s.right.z, 0.0];
        out[at + 4] = [s.albedo.x, s.albedo.y, s.albedo.z, 0.0];
    }
    out
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuLights {
    /// x = active light count; y = how many of them, from the front, are LIVE
    /// (the rest are baked into the level's lightmaps and shaded only by
    /// surfaces without one -- see `receiver_skips_baked`); z = 1 turns the
    /// shader's light culling OFF, to measure it (see `light_culling` in the
    /// shader); w = 1 turns the lamps' footprint-filtered terminator OFF, to
    /// measure it (see `terminator_aa` in the shader).
    count: [u32; 4],
    /// x = how many lights, from the front, are lit surfaces' own
    /// ([`Light::is_surface_light`]), which the scene readers shade apart
    /// (`surface_lights` in the shader) and their lamp loop starts past; 0
    /// with the `surface_light_loop` lever off, and they are lamps there.
    surface_lights: [u32; 4],
    lights: [GpuLight; MAX_LIGHTS],
    /// The frame's [`LitSurface`]s: see `pack_lit_surfaces`.
    surfaces: [[f32; 4]; LIT_SURFACE_VEC4S * MAX_LIT_SURFACES],
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
    /// Whether each lamp's terminator is shaded over the pixel's footprint
    /// (`terminator_aa` in the shader). On as shipped; off only to measure it
    /// (the `terminator_aa` lever).
    terminator_aa: std::cell::Cell<bool>,
    /// Whether the scene readers shade the lit surfaces' lights apart from
    /// the lamps (`GpuLights::surface_lights`). On as shipped; off only to
    /// measure it (the `surface_light_loop` lever).
    surface_lights_apart: std::cell::Cell<bool>,
    /// The surfaces the live lamps' beams are known to light, in order. See
    /// [`LitSurface`]. Uploaded with the lights.
    surfaces: std::cell::Cell<[Option<LitSurface>; MAX_LIT_SURFACES]>,
    /// The card-atlas texel row their pool maps are made in this frame, or
    /// none where none are made -- and then no surface is uploaded. See
    /// `pool_cards`.
    pool_row: std::cell::Cell<Option<u32>>,
    /// Which lights hold the moving casters' tiles this frame: tile k's light
    /// index, or `usize::MAX` for none. Uploaded as each light's shadow layer,
    /// `MAX_SPOT_SHADOWS + k` (`character_shadow_tile` in the shader).
    tile_lamps: std::cell::Cell<[usize; super::shadow::MAX_CHARACTER_SHADOWS]>,
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
        }
    }

    /// The lights holding the moving casters' tiles, tile by tile: `lamps[k]`
    /// is the index in the uploaded list of tile k's light. Takes effect with
    /// the next upload; `&[]` for none.
    pub fn set_tile_lamps(&self, lamps: &[usize]) {
        let mut held = [usize::MAX; super::shadow::MAX_CHARACTER_SHADOWS];
        for (slot, &l) in held.iter_mut().zip(lamps) {
            *slot = l;
        }
        self.tile_lamps.set(held);
    }

    /// The surfaces the live lamps' beams light this frame, at most
    /// [`MAX_LIT_SURFACES`], brightest first; one with no map's reach is left
    /// out. Takes effect with the next upload.
    pub fn set_lit_surfaces(&self, surfaces: &[LitSurface]) {
        let mut held = [None; MAX_LIT_SURFACES];
        for (slot, s) in held.iter_mut().zip(surfaces.iter().filter(|s| s.tan_half > 0.0)) {
            *slot = Some(*s);
        }
        self.surfaces.set(held);
    }

    /// The surfaces the next upload sends, in the order their pool maps lie in
    /// the atlas (`pool_cards`).
    pub fn lit_surfaces(&self) -> Vec<LitSurface> {
        self.surfaces.get().iter().flatten().copied().collect()
    }

    /// The card-atlas texel row the pool maps are made in this frame, or None
    /// where they are not made -- then no surface is uploaded, so none is
    /// read. Takes effect with the next upload.
    pub fn set_pool_row(&self, row: Option<u32>) {
        self.pool_row.set(row);
    }

    /// The next upload sends a surface for the pool maps' lookup to read.
    /// Where it does not, every lookup adds nothing, and the reflection
    /// passes' poolless twins draw the same pixels (`without_pool_maps`).
    pub fn reads_pool_maps(&self) -> bool {
        self.pool_row.get().is_some() && self.surfaces.get().iter().any(Option::is_some)
    }

    /// See `culling`. Takes effect with the next upload.
    pub fn set_culling(&self, on: bool) {
        self.culling.set(on);
    }

    /// See `terminator_aa`. Takes effect with the next upload.
    pub fn set_terminator_aa(&self, on: bool) {
        self.terminator_aa.set(on);
    }

    /// See `surface_lights_apart`. Takes effect with the next upload.
    pub fn set_surface_lights_apart(&self, on: bool) {
        self.surface_lights_apart.set(on);
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
        let mut gpu = pack_lights(lights, live, spot_layers, sun_is_baked, self.culling.get());
        for (k, &l) in self.tile_lamps.get().iter().enumerate() {
            // A spot slot's light keeps its slot: a light reads one map.
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

/// The pool map's light at a reflection's hit, read in `probe_hit_colour` with
/// the photographs' four texels so all five reads wait together
/// (`probe_surface_relit`), and added to a hit off any model -- in the passes
/// that ship (the deferring probe pass and its fix-up) and nowhere else. The
/// `*_cut_relight` measurement cuts take both out.
pub(crate) const SURFACE_RELIT_READ: &str = "    let relit = probe_surface_relit(h, d, roughness, t);\n";
pub(crate) const SURFACE_RELIT_CALL: &str = "    if (!model) {\n        col = vec4<f32>(col.rgb + relit, col.a);\n    }\n";

/// `GpuLight::params.w` for a light that stands for a lit SURFACE -- one that
/// casts no shadow ([`Light::casts_shadow`]), a flashlight's bounce off a
/// wall: it lights as any lamp does but makes no highlight, a point's
/// highlight being the reflection of something a metre wide. Below the
/// shadow layers' -1, so every `layer >= 0` test reads it as no layer. Less
/// the patch's radius ([`Light::source_radius`]), which the lights block
/// reads back as `-2 - params.w`.
pub const SURFACE_LIGHT: f32 = -2.0;

/// A spot whose soft edge spans more cosine than this is shaded as a lit
/// surface's light ([`Light::is_surface_light`]): its edge ramped plainly, the
/// baker's smoothstep from the outer cosine to the inner. `spot_cone` would
/// not widen an edge this wide, and its average along the pixel's long step
/// would be no truer: that average stretches the ramp about its middle, a box
/// filter's width for an edge about a pixel wide, while across an edge this
/// wide a pixel's true mean IS the plain ramp to second order in what the
/// pixel spans (`surface_light_gpu_tests`). Offline renders of six torch
/// views moved by at most one level, on no pixel by more (2026-10-06). A
/// flashlight's bounce is a half space or wider
/// (`quest_app::flashlight_bounce`); every lamp's edge is far narrower -- the
/// torch's 0.084, a 16 degree hot spot in 50.
pub const SURFACE_LIGHT_MIN_BAND: f32 = 0.5;

/// The GPU's copy of a frame's lights. See `GpuLights::count`.
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
            // `stationary_visibility`; -1 a light the level's bake never saw --
            // see `PROBE_CARD_RELIT`.
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
    wgsl_lights_block_with(group_index, binding_index, LightsBlockOptions::default())
}

/// The lights block's switch for the spots' shadow maps, as generated: on.
pub const SPOT_SHADOWS_ON: &str = "const SPOT_SHADOWS: bool = true;";
const SPOT_SHADOWS_OFF: &str = "const SPOT_SHADOWS: bool = false;";
/// The lights block's switch for the scene readers' own loop over the lit
/// surfaces' lights (`surface_lights` in the shader), as generated: on.
pub const SURFACE_LIGHTS_ON: &str = "const SURFACE_LIGHTS: bool = true;";
const SURFACE_LIGHTS_OFF: &str = "const SURFACE_LIGHTS: bool = false;";

/// A scene shader's SPOTLESS TWIN: `src` with the spots' shadow maps never
/// read, so the compiler drops the tent and everything only it kept live. The
/// same pixels in any frame where no spot casts (`shadow_params.y` = 0), where
/// every spot's test is false anyway. 2026-10-02: the tent alone took the
/// scene readers from 19 registers to 22 and their occupancy from 62% to 50%,
/// in every view, for the flashlight's sake.
///
/// AND NO LIT SURFACE'S LIGHT SHADED APART: the readers' loop over them
/// (`surface_lights`) is compiled out with the spots' shadows, since a frame
/// with one draws with the full readers (`XrRenderer::spotless_frame`) -- a
/// flashlight's bounce comes with its beam. Any such light in a spotless
/// frame is shaded as a lamp, as before the loop.
pub fn without_spot_shadows(src: String) -> String {
    assert!(src.contains(SPOT_SHADOWS_ON), "the shader has no spot-shadow switch");
    assert!(src.contains(SURFACE_LIGHTS_ON), "the shader has no surface-light switch");
    src.replacen(SPOT_SHADOWS_ON, SPOT_SHADOWS_OFF, 1).replacen(SURFACE_LIGHTS_ON, SURFACE_LIGHTS_OFF, 1)
}

/// A reflection pass's POOLLESS TWIN: `src` without the torch pool maps'
/// lookup, for frames where no surface is lit (`LightsUniform::reads_pool_maps`),
/// where every lookup adds nothing anyway -- the same pixels. 2026-10-06
/// (deploy96, the lever off against on, two passes): the twins save 0.10-0.18
/// ms a frame in every torchless view, halls and outdoors.
pub fn without_pool_maps(src: String) -> String {
    assert!(
        src.contains(SURFACE_RELIT_READ) && src.contains(SURFACE_RELIT_CALL),
        "the shader reads no pool map"
    );
    src.replacen(SURFACE_RELIT_READ, "", 1).replacen(SURFACE_RELIT_CALL, "", 1)
}

/// What a shader asks of the lights block beyond the default. See
/// `wgsl_lights_block_with`.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct LightsBlockOptions {
    /// `shade_material_env` takes its probe reflection from `probe_env_given`
    /// -- the half-resolution probe pass's answer -- instead of tracing it.
    /// See `PROBE_ENV_FROM_PASS`.
    pub probe_from_pass: bool,
    /// Every caller of `probe_environment` sets `probe_face_given`. See
    /// `PROBE_FACE_ALWAYS_GIVEN`.
    pub probe_face_always: bool,
    /// `probe_environment` leaves a traced hit's secondary lookups to
    /// `probe_fixup`. See `PROBE_SECONDARY_DEFERRED`.
    pub defer_secondary: bool,
    /// The lamp pre-pass tests a lamp's range before its baked mask. See
    /// `CULL_RANGE_FIRST`.
    pub cull_range_first: bool,
    /// A model card's trust is tested at its four nearest texels and the
    /// verdicts blended, rather than tested once on blended texels. See
    /// `PROBE_CARD_TESTS_FILTERED`.
    pub card_tests_filtered: bool,
    /// A reflection of a model on cards takes the light of the lamps the
    /// level's bake never saw -- a player's flashlight -- on the card's albedo.
    /// See `PROBE_CARD_RELIT`.
    pub card_relit: bool,
    /// A spot's averaged edge takes the pixel's long step from the surface's
    /// own plane, which the fragment shader hands `set_pixel_long_step` once a
    /// pixel, rather than from the normal it shades with. See
    /// `spot_long_step`.
    pub long_step_from_plane: bool,
}

/// HOW MANY OF THE PROBE PASS'S OWN PIXELS A REFLECTED EDGE IS SOFTENED
/// OVER, on each side: a doorway's rim, a solid proxy's outline, a model's
/// (`probe_pixel_spread` in the lights block).
///
/// The pass runs at half resolution and the scene reads it back bilinearly,
/// and a bilinear read shows an edge's position snapped to the texel grid
/// unless the edge is at least two texels wide. At one footprint a side, the
/// reflected doorway's edge moved 0.45 px (RMS) off its line in the headset's
/// view -- the staircase that crawls as the head moves, worst at middle and
/// far distance where a texel covers the most (headset 22:04:15, measured
/// offline with the edge's per-row crossing, 2026-09-29).
pub const PROBE_EDGE_FOOTPRINTS: f32 = 1.0;

/// `wgsl_lights_block`, with `options`. See `LightsBlockOptions`.
pub fn wgsl_lights_block_with(group_index: u32, binding_index: u32, options: LightsBlockOptions) -> String {
    let LightsBlockOptions {
        probe_from_pass,
        probe_face_always,
        defer_secondary,
        cull_range_first,
        card_tests_filtered,
        card_relit,
        long_step_from_plane,
    } = options;
    // Written as asked, so every shader that sets no plane keeps its code
    // exactly. See `spot_long_step`.
    let spot_long_step_fn = if long_step_from_plane {
        "var<private> pixel_long_step: f32 = 0.0;\nfn set_pixel_long_step(n_plane: vec3<f32>, view_dir: vec3<f32>) {\n    pixel_long_step = pixel_footprint * inverseSqrt(max(abs(dot(view_dir, n_plane)), SPOT_LONG_MIN_COS));\n}\nfn spot_long_step(vn: f32) -> f32 {\n    return pixel_long_step;\n}"
    } else {
        "fn spot_long_step(vn: f32) -> f32 {\n    return pixel_footprint * inverseSqrt(max(abs(vn), SPOT_LONG_MIN_COS));\n}"
    };
    // Written into `probe_card_colour` only where asked for, not behind the
    // constant: a call in a constant-false branch still counts the lamps and
    // the shadow maps among the bindings a shader uses, and every pipeline
    // laid out from its shader would need them. See `PROBE_CARD_RELIT`.
    let card_relit_call = if card_relit {
        "    if (PROBE_CARD_RELIT && sum.best > 0.0) {\n        colour += probe_card_relit(row, sum.card, lod, h, q);\n    }\n"
    } else {
        ""
    };
    // A hit on a surface a live lamp's beam lights, lit by it: where the
    // reflections that ship are coloured -- the probe pass that defers and its
    // fix-up -- and nowhere else, for the same reason. See
    // `probe_surface_relit`.
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
    let probe_select_rows = 3 * crate::renderer::uniforms::MAX_PROBES;
    let proxy_field_rows = crate::renderer::proxy_field::MAX_PROXY_FIELDS * 3;
    let proxy_card_rows = crate::renderer::uniforms::MAX_PROXIES / 4;
    let building_rows = crate::renderer::uniforms::MAX_BUILDINGS * 2;
    let capsule_rows = crate::renderer::uniforms::MAX_CAPSULES * 2;
    let capsule_group_rows = crate::renderer::uniforms::MAX_CAPSULE_GROUPS * 2;
    let capsules_per_group = crate::renderer::uniforms::CAPSULES_PER_GROUP;
    let character_card_sets = super::proxy_cards::CHARACTER_CARD_SETS;
    let character_card_rows = 2 * character_card_sets;
    let character_card_max_lod = (super::character_cards::CARD_MIPS - 1) as f32;
    let pool_maps_across = super::pool_cards::POOL_MAPS_ACROSS;
    let reflection_contrast = format!("{:?}", crate::renderer::space_warp::REFLECTION_CONTRAST_RATIO);
    let probe_edge_footprints = PROBE_EDGE_FOOTPRINTS;
    let floor_mirror_bias = super::brush_pipeline::probe_pass::FLOOR_MIRROR_BIAS;
    // A carried glass's glow, which a shader tracing its own reflection adds
    // past the probe's normalisation (see `capsule_glow`). One reading the
    // probe pass has it in the pass's answer already, and keeps its text.
    let capsule_glow_term = if probe_from_pass { "" } else { " + capsule_glow * spec_occ" };
    // Three vec4 per probe -- centre, min, max -- so the WGSL array length is
    // three times the probe count. Derived rather than written twice: a shader
    // array shorter than the uniform reads garbage past its end.
    let probe_slots = crate::renderer::uniforms::MAX_PROBES * 3;
    let max_probes = crate::renderer::uniforms::MAX_PROBES;
    // `probe_nearest_two_unrolled`'s body, one block a slot: see there.
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
    // The same, reading `probe_select`: `probe_nearest_two_const_unrolled`.
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
// HALF-PRECISION ALIASES: `f32` unless `shader_precision::for_device` rewrites
// them to `f16` for a device that has it -- which it does not as shipped:
// `shader_precision::HALF_PRECISION` is off since B1 (2026-10-06).
{precision_aliases}
// THE LARGEST VALUE CARRIED AT HALF PRECISION: below f16's 65,504, so a light
// clamped to it stays finite whichever precision `hf` is.
const HF_MAX: f32 = 60000.0;
// THE SPOTS' SHADOW MAPS READ AT ALL: false in the scene shaders' spotless
// twins, drawn in frames where no spot casts (`without_spot_shadows`).
{spot_shadows_on}
// THE LIT SURFACES' LIGHTS SHADED APART FROM THE LAMPS (`surface_lights`):
// false in the same twins, whose frames have none.
{surface_lights_on}
struct Camera {{
    view_proj: array<mat4x4<f32>, 2>,
    inv_view_proj: array<mat4x4<f32>, 2>,
    sun_view_proj: mat4x4<f32>,
    // The moving-objects sun map's matrix [0], then one per spot tile and per
    // characters' tile [1 + i]: `sun_dynamic_view_proj` and `spot_view_proj`
    // in `uniforms::Uniforms`, which lie side by side, read as one array, so
    // the lamp loop carries a point into whichever map a lamp reads with one
    // copy of `shadow_coords`. Read through `spot_view_proj(i)`,
    // `SUN_MOVING_MAP` and `CHARACTER_MAPS`.
    moving_view_proj: array<mat4x4<f32>, {moving_maps}>,
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
    // The resident rooms' tables -- each room's first slot, doorway and proxy,
    // and each one's next in its room. See `uniforms::Uniforms::probe_rooms`.
    probe_rooms: array<vec4<f32>, {room_table_rows}>,
    // Where the ground map lies: [min.x, min.z, 1 / extent.x, 1 / extent.z].
    // Must match `uniforms::Uniforms::ground_params`.
    ground_params: vec4<f32>,
    // Where each model's distance field lies in the atlas: [origin, reach],
    // [size, stop], [albedo, 0] per field. Must match
    // `uniforms::Uniforms::proxy_fields`.
    proxy_fields: array<vec4<f32>, {proxy_field_rows}>,
    // Each proxy's row in the card atlas plus one, 0 for none: entry i at
    // [i >> 2][i & 3]. Must match `uniforms::Uniforms::proxy_cards`.
    proxy_cards: array<vec4<f32>, {proxy_card_rows}>,
    // The buildings' outsides: [min.xyz, cube layer], [max.xyz, 0] each; how
    // many in portal_params.w. Must match `uniforms::Uniforms::building_boxes`.
    building_boxes: array<vec4<f32>, {building_rows}>,
    // The characters as capsules, player frame: [a, radius], [b, 0] each,
    // CAPSULES_PER_GROUP slots a character; [centre, bound radius] and
    // [colour, capsule count] a character; how many in capsule_params.x.
    // Must match `uniforms::Uniforms::capsules`.
    capsules: array<vec4<f32>, {capsule_rows}>,
    capsule_groups: array<vec4<f32>, {capsule_group_rows}>,
    capsule_params: vec4<f32>,
    // The characters on cards, player frame: [box centre, atlas row + 1 (0
    // for none)], [box half size, capsule group] a set. Must match
    // `uniforms::Uniforms::character_cards`.
    character_cards: array<vec4<f32>, {character_card_rows}>,
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

// `to_world_direction` undone: a direction baked in the world -- a lightmap's
// bounce direction -- into the frame the normals and lights arrive in.
fn to_player_direction(d: vec3<f32>) -> vec3<f32> {{
    let yaw = camera.player_frame.w;
    let s = sin(yaw);
    let c = cos(yaw);
    return vec3<f32>(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
}}

// `to_world_space` undone.
fn to_player_space(w: vec3<f32>) -> vec3<f32> {{
    let yaw = camera.player_frame.w;
    let s = sin(yaw);
    let c = cos(yaw);
    let p = w - camera.player_frame.xyz;
    return vec3<f32>(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
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
    // x: how many lights, from the front, are lit surfaces' own. See
    // `surface_light_count`.
    surface_lights: vec4<u32>,
    lights: array<Light, {MAX_LIGHTS}>,
    // The surfaces the live lamps' beams are known to light. See
    // `lit_surface_at` and `lights::LitSurface`.
    surfaces: array<vec4<f32>, {lit_surface_vec4s}>,
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
// THE GROUND SEEN FROM ABOVE: RGB the light it returns, A its world height.
// See `ground_map` and `outdoor_radiance`.
@group({group_index}) @binding({ground_tex}) var ground_map: texture_2d<f32>;
// Standing models' distance fields. See `proxy_field` and `probe_proxy_field`.
@group({group_index}) @binding({proxy_field_tex}) var proxy_field: texture_3d<f32>;
// Standing models' cards: a model a row, six cards a row. See `proxy_cards`
// and `probe_card_colour`.
@group({group_index}) @binding({proxy_card_tex}) var proxy_cards: texture_2d<f32>;
// THE PHOTOGRAPHS' CAPTURE POINTS AND ROOMS AGAIN, as a uniform block of their
// own: the camera buffer's `probe_boxes`, bound a second time over just those
// bytes (`uniforms::PROBE_SELECT_OFFSET`). Read only at indices a loop counts,
// so the driver may keep all of it in constant memory -- which it cannot for
// the camera block, whose tables the trace indexes by what it read before.
// Read by `probe_nearest_two_const`, the `def_scan_const` cut.
@group({group_index}) @binding({probe_select_binding}) var<uniform> probe_select: array<vec4<f32>, {probe_select_rows}>;

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
    // Each term is coefficient * basis * cosine-lobe weight, the lobe's
    // weights already divided by pi (1, 2/3 three times, 1/4 five times) --
    // see SkyIrradiance::evaluate in sky.rs, which this mirrors exactly, in the
    // same order.
    //
    // WRITTEN OUT, NOT A LOOP OVER TWO LOCAL ARRAYS. That was the documented
    // Adreno cliff: a local array read at a loop index is kept in scratch
    // MEMORY unless the compiler unrolls the loop, and the loop counter wgpu
    // adds to every loop is exactly what stops it doing so. This runs twice
    // a pixel in the brush shader and once in the probe pass.
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
// A PIXEL'S LONGER STEP ON A SURFACE, for a spot's soft edge seen edge-on
// (`spot_cone_across`): the footprint over the square root of how squarely
// the eye sees the surface, `vn` the cosine between the view and the normal.
// On a plane a pixel's footprint is an ellipse whose axes are the footprint
// times and over that root: seen edge-on a pixel spans a few centimetres up a
// wall and decimetres along it, and the footprint -- the mean of the two --
// left a far wall spot's whole edge inside a pixel along the wall, flickering
// as the head moved (2026-10-05). From the footprint the lamps hold already
// and the cosine they work out anyway. The screen derivatives for it, taken
// beside the lamps, cost the spotless scene readers three registers and an
// occupancy step, 0.45 ms a frame (PIPESTATS bisection, 2026-10-06) -- and
// the longer of the screen's two steps is not even the ellipse's axis: it
// turns with the head. Capped at grazing, where the pool is a sliver inside
// one pixel whatever the cap.
//
// FROM THE SURFACE'S OWN PLANE where the shader knows it
// (`LightsBlockOptions::long_step_from_plane`: the brushes and the ground,
// once a pixel through `set_pixel_long_step`), not from the normal map's
// normal. A bump tilts the shading, not the surface: the pixel still covers a
// patch of the plane. Taken from a bump turned nearly edge-on to the eye, the
// step stretched up to 32 times, the cone's edge was averaged that far, and a
// pixel past a pool took about half its light -- white specks past the
// sconces' pools on the stone ceiling (headset, 2026-10-06). The bumps still
// shade every lamp's light through `dot(n, l)` and the highlight.
const SPOT_LONG_MIN_COS: f32 = 1e-3;
{spot_long_step_fn}
// HOW FAR `dot(n, l)` SWINGS ACROSS THIS PIXEL, set once at the top of a
// fragment shader whose normal is MAPPED, from that normal's derivatives (see
// `terminator_width_of`). The lamps' terminator is then shaded over the pixel's
// footprint rather than at its centre -- see `terminator_aa`. 0, as every
// shader that does not set it leaves it, is the hard clamp exactly as before.
var<private> terminator_width: f32 = 0.0;
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
// brushes from their atlas, the ground from its own map -- through
// `set_stationary_masks`; 1, unshadowed, everywhere else. See
// `stationary_visibility`.
var<private> stationary_vis_a: vec4<f32> = vec4<f32>(1.0);
var<private> stationary_vis_b: vec4<f32> = vec4<f32>(1.0);

// THE STATIONARY LAMPS' SHADOWS, rebuilt from their masks as the sun's is: per
// lamp a signed distance and the bulb's penumbra, two lamps a layer (red and
// green, blue and alpha), over +-`range` texels -- the edge put back at zero
// at any magnification and never narrower than a pixel. Away from a sharp
// edge the baker stores a lamp's visibility as a distance this smoothstep
// gives back exactly. A layer the receiver does not have is passed as 1s,
// which reads fully lit. Call from uniform control flow: it takes `fwidth`.
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
// A MODEL'S masks, read exactly as baked: no widening by how fast the code
// changes across the screen. A brush's mask is one continuous field, and
// widening it there antialiases a shadow's edge; a model gives every triangle
// a chart of its own, so its code jumps at every triangle -- and across a
// sliver's quad, whose other pixels read other triangles' charts, a texel
// baked fully hidden read up to 16% lit (`mesh_pipeline` GPU test
// `a_hidden_texel_stays_hidden_beside_lit_ones`; 2026-10-01). Found while
// chasing the sconce plate's white dashes, which were not this alone: see
// `MeshVertex::uv2_rect`. The model bake gives its masks a full-width
// penumbra already.
fn set_stationary_masks_exact(st_0: vec4<f32>, st_1: vec4<f32>, st_2: vec4<f32>, st_3: vec4<f32>, range: f32) {{
    let st_da = (vec4<f32>(st_0.r, st_0.b, st_1.r, st_1.b) - vec4<f32>(0.5)) * (2.0 * range);
    let st_db = (vec4<f32>(st_2.r, st_2.b, st_3.r, st_3.b) - vec4<f32>(0.5)) * (2.0 * range);
    let st_wa = max(vec4<f32>(st_0.g, st_0.a, st_1.g, st_1.a) * range, vec4<f32>(0.02));
    let st_wb = max(vec4<f32>(st_2.g, st_2.a, st_3.g, st_3.a) * range, vec4<f32>(0.02));
    stationary_vis_a = smoothstep(-st_wa, st_wa, st_da);
    stationary_vis_b = smoothstep(-st_wb, st_wb, st_db);
}}

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
// THE CHARACTERS AS CAPSULES: the darkening of what they stand over in
// indirect light, and their presence in reflections -- one loop a pixel each.
// Their shadows from lamps are the characters' shadow tiles (see the lamp
// loop). See `uniforms::CapsuleUpload`.
const CAPSULES_PER_GROUP: i32 = {capsules_per_group};
// HOW FAR A CHARACTER DARKENS WHAT IS ROUND IT, from each capsule's own
// surface: in full within the first distance, faded out smoothly by the
// second -- which is where the group is culled, so nothing is cut off. It
// used to be cut off there, at a sphere round the whole body where the
// capsules together still took 5-10% of the light: a disc with a hard rim
// on the marble pillar a metre away, which grew and shrank as the player
// raised an arm, because the arm moved the sphere (headset, 2026-10-01).
const CAPSULE_AMBIENT_FULL: f32 = 0.15;
const CAPSULE_AMBIENT_REACH: f32 = 0.6;
// HOW WRONG A CAPSULE BODY IS, in metres: no clothes, hands, shoulders or
// hair. Nothing it shows may be sharper than that -- a crisp capsule body
// reads as a mannequin, and the headset's pillar reflected the player as a
// stick figure waving its arms (2026-09-29). So its reflection and its shadow
// are blurred by at least this, and a limb thinner than the blur fades with
// it. The crisp shadow of the real body is the characters' shadow tiles'.
const CAPSULE_SHAPE_BLUR: f32 = 0.15;
// Whether this surface takes the characters' shadows and darkening: not the
// characters themselves, whose surfaces lie inside their own capsules.
var<private> capsule_receiver: bool = true;

// The point of capsule `a`..`b` nearest the line `o + d t` (`d` unit length).
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

// THE CHARACTERS' CONTACT DARKENING at `p`, facing `n`: the share of the light
// arriving from all around -- the lightmap's bounce, the sky -- they block.
// Each capsule as the sphere at its point nearest `p`, whose cosine-weighted
// share of the hemisphere is (r / d)^2 max(n.w, 0). What grounds a character
// where no lamp shines on it. Unreal's indirect capsule shadows.
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
            // Faded by this capsule's own distance, so moving an arm changes
            // the arm's darkening and nothing else's.
            let fade = 1.0 - smoothstep(CAPSULE_AMBIENT_FULL, CAPSULE_AMBIENT_REACH, d - ar.w);
            let k2 = (ar.w / d) * (ar.w / d) * fade;
            vis = vis * (1.0 - k2 * max(dot(n, w / d), 0.0));
        }}
    }}
    return vis;
}}

// Where a ray from `p` along `d` (player frame) leaves the room this surface's
// face belongs to; far away where no room is given -- outdoors, the ground.
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

// Surfaces rougher than this show no character in their reflection: the lobe
// is far wider than a body at any distance it could be seen, and the test is
// not worth its cost there.
const CAPSULE_REFLECT_MAX_ROUGHNESS: f32 = 0.7;

// THE PLAYER ON CARDS: six pictures of their body drawn this frame, one
// looking in through each face of a box round it (`character_cards`). The
// least blurred level a reflection can be read at sits under the footprint;
// the coarsest is this.
const CHARACTER_CARD_SETS: i32 = {character_card_sets};
const CHARACTER_CARD_MAX_LOD: f32 = {character_card_max_lod:?};

// One card of a character's set: card `card` (0..6), at `uv` across it, read
// at `lod` and never closer to its edge than half a texel of that level -- a
// bilinear read there would take its neighbour's.
fn character_card_at(card: f32, uv: vec2<f32>, row: f32, res: f32, lod: f32, dims: vec2<f32>) -> vec4<f32> {{
    let s = 0.5 * exp2(ceil(lod));
    let at = vec2<f32>(card * res, row) + clamp(uv * res, vec2<f32>(s), vec2<f32>(res - s));
    return textureSampleLevel(proxy_cards, probe_samp, at / dims, lod);
}}

// What a reflection leaving `p` along `d` shows of character `g`, `t` along
// it where it passes nearest the character's capsules: the body's albedo
// premultiplied by its coverage, from the three cards that face the ray, each
// weighted by how squarely (`d` squared, which sums to one), read at that
// point and as blurred as the footprint there -- and covered only as far as
// all three agree. A card is an orthographic view, so for a ray along its
// axis this is exactly what the ray meets first -- a hand in front of a chest
// included; a ray between two axes mixes their two views' colours. Read at
// the point inside the body rather than where the ray
// enters its capsule: a capsule is fatter than the body it stands for, and
// its surface seen from the side of the ray falls outside the body's outline
// on the other card. -1 where the character has no cards this frame. The
// cards are square to the WORLD (`character_cards::card_box`): `p` and `d`
// come in the player's frame, which turns with the rig, and are turned back
// to the world's axes here, so a snap or smooth turn changes nothing a
// reflection shows.
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
    let across_x = character_card_at(select(1.0, 0.0, w.x < 0.0), uvw.yz, row, res, lod, dims);
    let across_y = character_card_at(select(3.0, 2.0, w.y < 0.0), uvw.zx, row, res, lod, dims);
    let across_z = character_card_at(select(5.0, 4.0, w.z < 0.0), uvw.xy, row, res, lod, dims);
    // In the body only where EVERY card saw body (the visual hull), so the
    // least of the three; the colour the cards facing the ray see, by how
    // squarely. Summed by direction, a card the ray half faces vouched for a
    // point the other saw empty: looking down at the marble pillar, its
    // reflection of the player carried half the top view's shoulders beside
    // the legs and between them -- a faint shadow round the body that grew
    // and shrank as an arm rose (headset, 2026-10-01).
    let cover = min(across_x.a, min(across_y.a, across_z.a));
    let seen = k.x * across_x + k.y * across_y + k.z * across_z;
    return vec4<f32>(seen.rgb / max(seen.a, 1e-4) * cover, cover);
}}

// THE GLOW OF A CARRIED GLASS in the reflection `capsule_reflection` last
// answered, as straight colour to add to its answer's before that answer's
// alpha is applied. Light given off, not the probe's light: each caller adds
// it past the probe's brightness normalisation, which would put a torch out
// in the dark corner it is lighting. See `CapsuleGroup::surfaces`. Already
// dimmed by the glass's own beam's shadow: see `capsule_glass_beam`.
var<private> capsule_glow: vec3<f32> = vec3<f32>(0.0);

// THE CHARACTERS IN A REFLECTION leaving `p` along `d` (player frame), over
// `behind` -- what the probe answered, which cannot hold anything that moves:
// the capsules the ray passes through before it leaves the room, soft over
// the footprint (the lobe, or a pixel on a mirror), in each character's
// colour lit by `lit`, the irradiance arriving here -- the light the underside
// of someone standing on this floor would get. Where the capsule the ray
// passes closest is a character with cards, the cards decide what it shows:
// the body's own outline and colours, lit the same way.
//
// AND WHAT THEY CARRY (`CapsuleGroup::surfaces`): a torch as its own shape,
// unblurred, its glass glowing (user, 2026-10-02: the flashlight "does not
// show as a reflection on any surface ... nor does the front of the
// flashlight light up or show lit in the avatar's reflection"). Of two
// capsules covering a ray alike the nearer shows, so a torch held in front of
// a chest is not lost behind it. The glass's glow is light given off, so it
// is ADDED, in `capsule_glow`, dimmed only by what covers the ray nearer.
fn capsule_reflection(p: vec3<f32>, d: vec3<f32>, roughness: f32, lit: vec3<f32>, behind: vec4<f32>) -> vec4<f32> {{
    let groups = i32(camera.capsule_params.x);
    if (groups <= 0 || roughness > CAPSULE_REFLECT_MAX_ROUGHNESS) {{
        return behind;
    }}
    let lobe = probe_lobe_tan(roughness);
    let eye = max(distance(cam_pos(), p), 0.05);
    var t_max = -1.0;
    // The capsule that covers most, and how far along the ray it passes --
    // for its colour, and its character's cards.
    var cover = 0.0;
    var win_i = -1;
    var win_t = 0.0;
    // The brightest glass: the light it sends along the ray, and from where.
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
        // Only for a ray that passes a character: the room's wall.
        if (t_max < 0.0) {{
            t_max = capsule_room_exit(p, d);
        }}
        let count = i32(camera.capsule_groups[g * 2 + 1].w);
        for (var k = 0; k < count; k = k + 1) {{
            let i = g * CAPSULES_PER_GROUP + k;
            let ar = camera.capsules[i * 2];
            let bs = camera.capsules[i * 2 + 1];
            // Where the ray passes it: nearest its axis -- or, for a glass,
            // where the ray crosses the glass's plane from in front.
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
            // A body's capsule is as wrong as `CAPSULE_SHAPE_BLUR`; a carried
            // thing's capsules are its shape.
            let footprint =
                max(max(t * lobe, pixel_footprint * (1.0 + t / eye)), select(0.0, CAPSULE_SHAPE_BLUR, bs.w == 0.0));
            if (bs.w > 0.0) {{
                // The disc blurred over the footprint, keeping the light it
                // gives off: its edge softened by f, to the footprint's own
                // radius where that is wider than the disc, and its share
                // scaled so that summed over the plane it is the disc's own
                // area -- the softened edge adds 0.2 f^2 to the spread's.
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
            // Blurred by the footprint, a limb thinner than it covers only
            // part of any pixel: its share fades as radius over footprint.
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
        // The group's colour; a carried solid's scaled by its surface.
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
        // Through whatever covers the ray nearer than the glass.
        seen = glow * select(1.0, 1.0 - cover, win_t < glow_t);
        // The glass covers its own share of the ray, whatever lay behind.
        a = max(a, glow / camera.capsules[glow_i * 2 + 1].w);
    }}
    // The answer made BEFORE the glass's beam is asked about, so what it is
    // made from is let go first. See `capsule_glass_beam`.
    let answer = vec4<f32>(mix(behind.rgb, colour * lit * INV_PI, cover), a);
    if (glow_i >= 0) {{
        capsule_glow = camera.capsule_groups[(glow_i / CAPSULES_PER_GROUP) * 2 + 1].rgb * seen / a
            * capsule_glass_beam(p, camera.capsules[glow_i * 2 + 1].xyz);
    }}
    return answer;
}}

// HOW MUCH OF THE GLASS AT `glass` A MIRROR AT `p` SEES (player frame): as
// much as that glass's own beam reaches `p` past its spot shadow map -- which
// holds the hand raised in front of the torch, as the capsules cannot: the
// trace dims a glass only by the capsule covering most of the ray, and behind
// a carried glass that is the arm carrying it, so a hand held between the
// glass and a polished wall hid none of the glass's image in it (user,
// 2026-10-05: "I cast a shadow with the hand that would have blocked the
// light from being visible on the wall but it still showed it reflected as if
// nothing was blocking it"). The beam reaching `p` and the glass seen in a
// mirror at `p` are one line, so the shadow on the wall and the image in it
// now go together. Its beam is the shadowed spot standing at the glass
// (`flashlight::torch_capsules` puts the glass on the beam's origin); a point
// outside its map, or a glass with no shadowed beam, sees it whole.
//
// ASKED LAST, inside `capsule_reflection` once its answer is made: asked
// while that answer's colour, cover and light were still held -- at the end
// of the probe pass, or before the answer here -- it took the ground's
// reflection pass from 24 registers to 25, its occupancy from 50% to 37%,
// whichever of the search or the shadow lookup below was cut out: +0.3 to
// +0.6 ms a frame outdoors (headset A/B, 2026-10-06). Asked after, 24.
fn capsule_glass_beam(p: vec3<f32>, glass: vec3<f32>) -> f32 {{
    for (var i: u32 = 0u; i < live_light_count(); i = i + 1u) {{
        let apart = lights.lights[i].position.xyz - glass;
        if (dot(apart, apart) >= GLASS_ON_ITS_BEAM * GLASS_ON_ITS_BEAM) {{
            continue;
        }}
        let layer = i32(lights.lights[i].params.w);
        if (SPOT_SHADOWS && layer >= 0 && f32(layer) < camera.shadow_params.y) {{
            return pcf_layer_tap(spot_shadow_tex, layer, p, spot_view_proj(layer));
        }}
    }}
    return 1.0;
}}
// How near a lamp must stand to a glass to be its beam, metres.
const GLASS_ON_ITS_BEAM: f32 = 0.01;

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
// How many lights, from the front of the list, are lit surfaces' -- a
// flashlight's bounce -- which the scene readers shade apart from the lamps
// (`surface_lights`), their lamp loop starting past them. 0 in the spotless
// twins, which shade every light as a lamp (`SURFACE_LIGHTS`). Every other
// light loop takes them as lamps. See `GpuLights::surface_lights`.
fn surface_light_count() -> u32 {{
    return select(0u, lights.surface_lights.x, SURFACE_LIGHTS);
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
// THE PIXEL'S MEAN ALONG ITS LONG STEP (`spot_cone`). Measured 2026-10-05:
// from the hall's front doorways the far corner spot's pixels that jump more
// than 8 levels between 3 mm head steps fell from 3.2% to 1.0%
// (`offline_frame::measure_the_move_shimmer`), and `spot_edge_gpu_tests` holds
// the shimmer below a sensor's square pixels'. It first cost 0.45 ms a frame:
// the spotless scene readers went from 18-19 registers to 21-22, 62% -> 50%
// occupancy. Eight formulations of the lamps' maths -- the steps as vectors,
// packed, the gradient as scalar dots, the lamp read field by field, the
// stationary masks packed, the sum at half precision -- all landed at 22; a
// bisection (2026-10-06) found the cost was the pixel's long step taken from
// screen derivatives beside the lamps, not the lamps' maths at all. See
// `spot_long_step`.
const SPOT_EDGE_AVERAGE: bool = true;
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
//
// THE PIXEL'S AVERAGE ALONG ITS LONG STEP (2026-10-05). The footprint above
// is the mean of the pixel's two axes, and seen edge-on the axis along the
// surface is ten times the other: from the hall's front doorways the far
// corner spot's whole soft edge fitted inside one pixel along the wall, and
// as the head moved its pool flickered between two and three pixels wide.
// `across` is how far the cone's cosine really moves across this pixel
// (`spot_cone_across`; 0 where no fragment shader set the pixel's steps), and
// what of it the widening above has not already spanned (the two spreads add
// as squares, as blurs do) is averaged over: the smoothstep stretched by that
// range on both sides about its middle, which is the shape of the pixel's
// mean of it and keeps its light and its centre. One smoothstep, not the
// exact mean's two integrals -- the scene readers' register peak is in this
// loop -- and it shimmers less than a sensor's square pixels would
// (`spot_edge_gpu_tests`). Nothing spreads past the pixel that saw it, the
// trap a widening by the full footprint fell into (2026-09-23), so no cap. A
// pixel seen head-on spans no more than the footprint says, and is exactly
// as before.
fn spot_cone(cos_angle: f32, cos_outer: f32, cos_inner: f32, dist: f32, across: f32) -> f32 {{
    let authored = max(cos_inner - cos_outer, 0.0001);
    let sin_a = sqrt(max(1.0 - cos_angle * cos_angle, 0.0));
    let footprint = sin_a * pixel_footprint / max(dist, 0.001);
    let at_least = SPOT_EDGE_MIN_PIXELS * footprint;
    let width = max(authored, min(at_least, authored * SPOT_EDGE_MAX_WIDEN));
    // How much the edge grew, split evenly either side of the authored band.
    // ZERO in a bake, which has no pixels.
    let widen = width - authored;
    // The ramp stretched about its middle by `half` of itself each way: the
    // average along the pixel's long step, the span's excess over the
    // footprint taken in quadrature. ONE expression, not the plain ramp and
    // the stretched one chosen between -- which held both -- since at `half`
    // zero the stretched ramp IS the plain one. With `widen` and `half` at
    // zero this is exactly the baker's cone, (cos_angle - cos_outer) /
    // authored, which `renderer_and_baker_agree_on_the_formula` pins by text.
    let half = select(0.0, 0.5 * sqrt(max(across * across - footprint * footprint, 0.0)) / width, SPOT_EDGE_AVERAGE);
    let s = clamp(((cos_angle - cos_outer + 0.5 * widen) / width + half) / (1.0 + 2.0 * half), 0.0, 1.0);
    return s * s * (3.0 - 2.0 * s);
}}
// How far a spot's cone cosine moves across this pixel, for `spot_cone`: the
// cosine's gradient over the surface, (d - cos u) / dist with `u` the way from
// the lamp (`-l_dir`), along the pixel's longer screen step -- which runs
// along the view ray laid onto the surface (`n`, `view_dir` toward the eye),
// as long as `spot_long_step` says. 0 where no footprint was set.
fn spot_cone_across(spot_dir: vec3<f32>, l_dir: vec3<f32>, cos_angle: f32, dist: f32, n: vec3<f32>, view_dir: vec3<f32>) -> f32 {{
    // g . along, with g = (d + cos l) / dist and along = v - (v.n) n, as dot
    // products of what the loop holds anyway: no vector is built.
    let vn = dot(view_dir, n);
    let g_along = dot(view_dir, spot_dir) - vn * dot(n, spot_dir) + cos_angle * (dot(view_dir, l_dir) - vn * dot(n, l_dir));
    return abs(g_along) * inverseSqrt(max(1.0 - vn * vn, 1e-8)) * spot_long_step(vn) / max(dist, 0.001);
}}
// The average radiance the chosen probe photographed, written by
// `probe_environment` for `shade_material_env` to normalise against. Zero means
// unknown, and an unknown probe is used as it is. See `PROBE_NORMALISATION`.
var<private> probe_brightness: f32 = 0.0;
// HOW FAR THE REFLECTED RAY TRAVELLED to what the reflection shows, in metres,
// written by `probe_environment`: the traced hit, the ground or a building
// beyond a doorway, `PROBE_REACH_SKY` for the sky; for an untraced (rough)
// reflection, the room box it was projected onto. The half-resolution probe
// pass stores it beside the reflection for SpaceWarp, which moves a reflected
// image with the point it is an image of rather than with the surface showing
// it. See `space_warp::reflected_point`.
var<private> probe_reach: f32 = 0.0;
// How many levels coarser than `GROUND_TRACE_FINEST_LEVEL` the ground trace
// stops at. Only the water sets it: a rippled, moving surface blurs what it
// mirrors far past the finest cells, and its rays skim the ground for tens
// of metres, the trace's most expensive case.
var<private> ground_trace_coarsen: i32 = 0;
// THE ROUGHNESS FROM WHICH A SURFACE'S REFLECTION IS THE LIGHTMAP'S LIGHT
// ALONE: its lobe is the whole hemisphere the diffuse term already integrates,
// so the probe's sharper answer takes no share (`lobe_is_hemispherical` in
// `shade_material_env_part`), and the probe pass does not trace it.
const PROBE_LOBE_HEMISPHERICAL: f32 = 0.75;
// The reach of a reflection that met only sky: far enough that its image moves
// as the sky does, within a half float.
const PROBE_REACH_SKY: f32 = 10000.0;
// WHETHER THIS SHADER READS ITS PROBE REFLECTION FROM THE HALF-RESOLUTION
// PASS rather than tracing it per pixel. See `brush_pipeline::probe_pass`. A
// constant, so a shader that reads it carries none of the trace: the branch in
// `shade_material_env` folds away, and the trace's registers with it.
const PROBE_ENV_FROM_PASS: bool = {probe_from_pass};
// WHETHER EVERY CALLER OF `probe_environment` HANDS IN ITS FACE'S ROOM
// (`probe_face_given`), as the half-resolution probe pass does. A constant, so
// that shader carries none of the searches for callers that do not -- nor the
// registers their inputs held through the trace. See `probe_environment`.
const PROBE_FACE_ALWAYS_GIVEN: bool = {probe_face_always};
// WHETHER A TRACED HIT'S SECONDARY LOOKUPS -- across a doorway's rim, across a
// solid proxy's outline; see `probe_secondary` -- ARE LEFT TO `probe_fixup`.
// A constant, so the probe pass that defers them carries none of their code
// or registers: with them it held 26 registers a pixel and kept 37% of its
// waves in flight, without them 18 and 62% (`PIPESTATS`, 2026-09-28), for
// lookups that fewer than one pixel in ten ever makes. The pixel's primary
// reflection is returned as usual, and everything the lookups need is recorded
// for `probe_fixup` (`probe_fixup_begin` and `_finish`) -- the ray's part the
// moment the trace ends, so it is not carried through the colour lookup.
const PROBE_SECONDARY_DEFERRED: bool = {defer_secondary};
// A MODEL CARD'S TRUST, FILTERED: each of the four texels round the point
// tested on its own and the verdicts blended, as percentage-closer filtering
// blends a shadow map's -- where one test on blended texels averaged a facing
// normal with an opposite one, and a depth in with a depth out, and flipped
// between them as the point moved a hundredth of a texel: the bright inside
// of a lamp, flickering in its reflection in the polished walls as the head
// moved a millimetre (offline, 2026-10-01). Four reads a card instead of one,
// so only in `probe_fixup`, which every texel meeting a model on cards goes
// through; the probe pass keeps its one read.
const PROBE_CARD_TESTS_FILTERED: bool = {card_tests_filtered};
// A MODEL'S REFLECTION LIT BY WHAT THE BAKE NEVER SAW: the cards hold the
// level's own light, baked, so a player's flashlight on a hanging lamp lit the
// lamp and not its reflection in the polished floor (headset, 2026-10-02:
// "if I shine the flashlight on the hanging light fixture, the reflection of
// the light fixture appears to be lit from the flashlight in its
// reflection"). Each hit on cards takes, on the albedo of the card vouching
// most for it, the light of every lamp the bake never saw (`position.w` -1:
// `pack_lights`), shadowed as the lamp's own map has it. Only in
// `probe_fixup`, which every texel meeting a model on cards goes through.
const PROBE_CARD_RELIT: bool = {card_relit};
// How far off the card's surface its point is lifted before the lamp's
// shadow map is read there: a card texel's depth is a fraction of a texel of
// the lamp's map from the surface the map drew, either side.
const PROBE_CARD_RELIT_LIFT: f32 = 0.02;
// WHETHER THE LAMP PRE-PASS TESTS A LAMP'S RANGE BEFORE ITS BAKED MASK: three
// instructions before a dozen, the better order where most lamps are out of
// range -- the ground outdoors, far from the building's lamps (-0.3 to
// -0.6 ms a frame, 2026-09-28). The brushes keep the mask first, the order
// they were measured in. The same lamps are kept either way.
const CULL_RANGE_FIRST: bool = {cull_range_first};
// THE SKY'S SUN CANNOT REACH THIS SHADER'S SURFACES: the brushes whose baked
// sun mask is dark over every texel their pixels can read (see
// `brush_pipeline::SunFaces`), drawn with a reader that sets this. There
// every directional light already came out exactly 0 -- the mask's 0 culled it
// -- so dropping it in the lamp pre-pass changes no pixel, and its shadow
// lookups leave the shader: ~350 of the scene reader's ~3,400 instructions,
// which on the Quest decide whether it fits its instruction cache (headset,
// 2026-10-01: 3,387 instructions drew hall_front in 16.3 ms, 3,398 in 18.6).
// `brush_pipeline::sun_reader_shader` sets it; nothing else does.
const SKY_SUN_NEVER_REACHES: bool = false;
// THIS SHADER'S SURFACES ALWAYS HAVE A BAKED SUN MASK to read: the brushes
// whose every readable mask texel is baked (`brush_pipeline::SunFaces`), where
// `receiver_sun_mask` is never the -1 of "no bake", so `sun_visibility` never
// falls back to the level's static sun map -- and that lookup leaves the
// shader. The same instruction-cache reason as `SKY_SUN_NEVER_REACHES`, for
// the faces the sun does reach. `brush_pipeline::sun_reader_shader` sets it.
const SUN_MASK_EVERYWHERE: bool = false;
// The fragment whose reflection this is -- its position builtin, set by the
// probe pass before it shades -- for the record of a deferred lookup.
var<private> probe_fragment: vec4<f32> = vec4<f32>(0.0);
{probe_fixup_wgsl}
// The half-resolution pass's answer for this pixel -- the probe radiance,
// already normalised, and its coverage -- set by the brush shader before it
// shades. Only read when `PROBE_ENV_FROM_PASS`.
var<private> probe_env_given: vec4<f32> = vec4<f32>(0.0);
// THE CHARACTERS' FLOOR MIRROR (`brush_pipeline::probe_pass::MIRROR_FORMAT`):
// the plane it mirrors in, `FLOOR_MIRROR_BIAS` above its height in
// `capsule_params.w`, 0 when there is none this frame. The probe pass lays the
// mirrored characters over a floor texel's reflection; a pixel of that floor
// then has them already, and its capsules are left out.
const FLOOR_MIRROR_BIAS: f32 = {floor_mirror_bias:?};
// How close to the plane a surface must be to be the floor, in metres.
const FLOOR_MIRROR_SLAB: f32 = 0.02;
fn on_floor_mirror(pos: vec3<f32>, geom_n: vec3<f32>) -> bool {{
    let lane = camera.capsule_params.w;
    return lane != 0.0 && geom_n.y > 0.95 && abs(pos.y - (lane - FLOOR_MIRROR_BIAS)) < FLOOR_MIRROR_SLAB;
}}
// Whether the probe pass texel being shaded is on the floor mirror's plane,
// and the light its capsules are lit by: set before its trace, so a deferred
// lookup's record carries them too.
var<private> probe_floor_mirror_here: bool = false;
var<private> probe_capsule_lit: vec3<f32> = vec3<f32>(0.0);
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

// Rows of `Camera::moving_view_proj`: the sun's moving-objects map's, the
// first characters' tile's, and spot tile `i`'s (or, past the spots,
// characters' tile `i - {max_spot_shadows}`'s), which lie between.
const SUN_MOVING_MAP: i32 = 0;
const CHARACTER_MAPS: i32 = 1 + {max_spot_shadows};
fn spot_view_proj(i: i32) -> mat4x4<f32> {{ return camera.moving_view_proj[1 + i]; }}

fn shadow_coords(world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> vec4<f32> {{
    let lp = light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let bias = 0.0015;
    // THE MAP'S FAR END, TESTED ON THE DEPTH COMPARED. The flashlight's lookup
    // matrix puts this bias into its depth row for the line below to take back
    // out (`shadow::spot_shadow_matrices`); tested before that, a receiver
    // past about 8 m from its 2 cm near plane read as outside the map, and lit:
    // its shadows stopped at a set distance from the torch (headset,
    // 2026-10-02).
    let depth = ndc.z - bias;
    var valid = 1.0;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || depth > 1.0 || ndc.z < 0.0) {{
        valid = 0.0;
    }}
    return vec4<f32>(uv.x, uv.y, depth, valid);
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
    var vis = sun_level_visibility(world_pos);
    if (vis > 0.0 && l.position.w > 0.5 && camera.shadow_params.z > 0.5) {{
        vis = vis * sun_moving_visibility(world_pos);
    }}
    return vis;
}}

// `sun_visibility`'s first half: the LEVEL's shadow alone, baked or from the
// static map.
fn sun_level_visibility(world_pos: vec3<f32>) -> f32 {{
    var vis = 1.0;
    if (SUN_MASK_EVERYWHERE || receiver_sun_mask >= 0.0) {{
        vis = receiver_sun_mask;
    }} else if (camera.shadow_params.x > 0.5) {{
        vis = pcf(sun_shadow_tex, world_pos, camera.sun_view_proj);
    }}
    return vis;
}}

// THE MOVING-OBJECTS MAP AT `world_pos`: its near tile -- the middle half of
// the sun tile's box at twice the detail -- wherever that holds the point and
// its kernel, which round the player is everywhere their own shadow falls; the
// sun tile beyond. One kernel either way: the near coordinates are the sun
// tile's scaled about its middle, and the depth is the same. See
// `shadow::SUN_NEAR_ZOOM`.
fn sun_moving_visibility(world_pos: vec3<f32>) -> f32 {{
    let c = shadow_coords(world_pos, camera.moving_view_proj[SUN_MOVING_MAP]);
    if (c.w < 0.5) {{ return 1.0; }}
    let near = (c.xy - vec2<f32>(0.5)) * SUN_NEAR_ZOOM + vec2<f32>(0.5);
    // Two of its texels in from its edge: the kernel reaches one and a half.
    let edge = 2.0 * SUN_ATLAS_GRID.x / f32(textureDimensions(sun_dynamic_shadow_tex).x);
    let in_near = all(abs(near - vec2<f32>(0.5)) < vec2<f32>(0.5 - edge));
    return pcf_tile_at(
        sun_dynamic_shadow_tex,
        vec2<f32>(select(0.0, SUN_NEAR_TILE, in_near), 0.0),
        SUN_ATLAS_GRID,
        vec3<f32>(select(c.xy, near, in_near), c.z),
    );
}}

// Where `sun_moving_visibility` reads the map for a point at `c` in the sun
// tile: the point in the tile it reads -- the near tile wherever that holds
// the point and its kernel -- and its depth (xyz), and that tile's column
// (w). For the lamp loop, which carries the point into whichever map a lamp
// reads itself. The wrappers keep their own bodies: written over helpers like
// these, they cost the models' shaders 50 instructions (PIPESTATS, build 117).
fn sun_moving_tile_at(c: vec3<f32>) -> vec4<f32> {{
    let near = (c.xy - vec2<f32>(0.5)) * SUN_NEAR_ZOOM + vec2<f32>(0.5);
    // Two of its texels in from its edge: the kernel reaches one and a half.
    let edge = 2.0 * SUN_ATLAS_GRID.x / f32(textureDimensions(sun_dynamic_shadow_tex).x);
    let in_near = all(abs(near - vec2<f32>(0.5)) < vec2<f32>(0.5 - edge));
    return vec4<f32>(select(c.xy, near, in_near), c.z, select(0.0, SUN_NEAR_TILE, in_near));
}}

// One spot's depth, read out of its tile of the shared atlas.
//
// The atlas exists because a pass is the expensive unit on a tile GPU, not the
// triangles in it -- see `ShadowMap::spots`. The cost of that is here: every
// sample has to be mapped into its own tile AND CLAMPED to it. Without the
// clamp, the kernel at a tile's edge reaches into the neighbouring tile and
// reads another light's depth, which shows up as a shadow cast by a lamp that
// is nowhere near -- far more confusing than a missing shadow.
//
// THE LEAN TENT (`pcf_tile_tent_lean_at`) since B1 (2026-10-06): the same
// shadow to the hardware's own bilinear precision
// (`every_tent_form_gives_the_same_shadow`), with nothing per axis held across
// its loop: one register under the loop over picked taps in the full reader
// at either precision -- 21 against 22 at `f32` as shipped (the
// `scene_tent_loop` cut, deploy107), 19 against 20 at `f16` (deploy105).
fn pcf_layer(tex: texture_depth_2d, layer: i32, world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> f32 {{
    let c = shadow_coords(world_pos, light_view_proj);
    if (c.w < 0.5) {{ return 1.0; }}
    return pcf_layer_at(tex, layer, c.xyz);
}}
// `pcf_layer` from the point already in the tile's map: `c` as
// `shadow_coords` gives it, where it holds the point. For the lamp loop,
// which carries the point into whichever map a lamp reads itself.
fn pcf_layer_at(tex: texture_depth_2d, layer: i32, c: vec3<f32>) -> f32 {{
    return pcf_tile_tent_lean_at(
        tex,
        vec2<f32>(f32(layer % {atlas_cols}), f32(layer / {atlas_cols})),
        vec2<f32>(f32({atlas_cols}), f32({atlas_rows})),
        c,
    );
}}
// `pcf_layer` as ONE bilinear compare: the four texels round the point,
// blended by where it falls among them, so the answer still slides as the
// point moves rather than stepping. For the glass's gate in the reflection
// passes (`capsule_glass_beam`), which says only whether a highlight shows:
// 160 instructions fewer in the ground's reflection pass than the tent's
// nine taps (headset, 2026-10-06), for an edge a few millimetres sharper on
// a wall a few metres off.
fn pcf_layer_tap(tex: texture_depth_2d, layer: i32, world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> f32 {{
    let c = shadow_coords(world_pos, light_view_proj);
    if (c.w < 0.5) {{ return 1.0; }}
    let grid = vec2<f32>(f32({atlas_cols}), f32({atlas_rows}));
    let lo = grid / vec2<f32>(textureDimensions(tex)) * 0.5;
    let tile = vec2<f32>(f32(layer % {atlas_cols}), f32(layer / {atlas_cols}));
    return textureSampleCompareLevel(tex, shadow_samp, (clamp(c.xy, lo, vec2<f32>(1.0) - lo) + tile) / grid, c.z);
}}

// ONE AXIS OF A 5x5 TENT carried by three bilinear taps: each tap's offset in
// texels from the texel corner nearest the sample (`off`), and its weight
// (`w`, summing to one). `f` is the sample's offset from that corner,
// -0.5..0.5.
//
// The tent, 2.5 texels to each side, lies over six texels; each pair of them
// is one bilinear tap, placed between the pair's centres so it blends them in
// the ratio of the tent's area over each, and weighted by their sum. 0.08 is
// 1/12.5, twice the tent's area. See `pcf_tile_tent_at`.
struct PcfTentAxis {{
    off: vec3<f32>,
    w: vec3<f32>,
}}
fn pcf_tent_axis(f: f32) -> PcfTentAxis {{
    let w = vec3<f32>((1.5 - f) * (1.5 - f) * 0.08, 0.0, (1.5 + f) * (1.5 + f) * 0.08);
    let w_mid = 1.0 - w.x - w.z;
    // Each pair's right-hand texel's share.
    let fp = max(f, 0.0);
    let right = vec3<f32>((2.0 - 2.0 * f) * 0.08, (4.0 + 2.0 * f - 2.0 * fp * fp) * 0.08, (0.5 + f) * (0.5 + f) * 0.08);
    let weights = vec3<f32>(w.x, w_mid, w.z);
    return PcfTentAxis(vec3<f32>(-2.5, -0.5, 1.5) + right / weights, weights);
}}

// `pcf_tile_tent_at` written out, tap by tap: the same taps, weights and
// order of summing. As shipped for a day (2026-10-01): it put the scene
// shader's baked reader past the instruction-cache cliff, ~0.9 ms an eye.
// For the `scene_tent_unrolled` cut.
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
// `pcf_tile_at` with a 5x5 TENT in place of its box (Castano, *Shadow Mapping
// Summary*, 2013): the same nine bilinear compares, each moved within its pair
// of texels and weighted by the tent's area over them. The box's equal taps
// at whole-texel offsets made a kernel with corners a texel apart, and those
// corners drew steps along every spot shadow's edge, which shimmered as the
// head moved over them (jitter plan, row 3); the tent's slope has none.
//
// A LOOP, each axis's tap picked by its index out of what `pcf_tent_axis`
// gives -- not written out, and not indexing a local array, which goes to
// scratch memory on this GPU (`adreno-local-array-cliff`). Written out, it
// was the code that put the scene shader past the instruction-cache cliff
// (`pcf_tile_tent_unrolled_at`).
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
// `pcf_tent_axis`'s tap `k` (-1, 0 or 1) alone, in closed form: its position
// in texels from the corner (x) and its weight (y). An outer pair, with
// a = 1.5 + k f, weighs 0.08 a^2 and its outer texel 0.08 (a - 1)^2, so its
// tap lies (1 - 1/a)^2 out from the inner texel's centre, 1.5 texels from the
// corner; the middle pair's tap is f / (4 + 2|f|) from the corner. The shipped
// tent's taps (`pcf_tile_tent_lean_at`) since B1.
fn pcf_tent_tap_axis(k: f32, f: f32) -> vec2<f32> {{
    let a = 1.5 + k * f;
    let s = 1.0 - 1.0 / a;
    return select(
        vec2<f32>(k * (1.5 + s * s), 0.08 * a * a),
        vec2<f32>(f / (4.0 + 2.0 * abs(f)), 0.16 * (4.0 - f * f)),
        k == 0.0,
    );
}}
// `pcf_tile_tent_at` with each tap's position and weight worked out at the
// tap from the sample's offset: the same taps, weights and order of summing,
// and nothing per axis held across the loop. 2026-10-02: the tent's six
// positions and six weights, live through the loop, took the scene readers
// from 19 registers to 22, and occupancy from 62% to 50%.
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
// `pcf_tile_tent_at` with each axis's tap positions and weights held at half
// precision (`hf`): positions to a five-hundredth of a texel, weights to three
// figures -- finer than the bilinear compare's own weights -- in half the
// registers where the device has `f16` and `HALF_PRECISION` is on. For the
// `scene_tent_half` cut.
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
// One tap at `local` in tile space, clamped into the tile as `pcf_tile_at`'s.
fn pcf_tent_tap(tex: texture_depth_2d, tile: vec2<f32>, grid: vec2<f32>, local: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>, depth: f32) -> f32 {{
    return textureSampleCompareLevel(tex, shadow_samp, (clamp(local, lo, hi) + tile) / grid, depth);
}}

// Tile `tile` (column, row) of an atlas `grid` tiles across and down.
fn pcf_tile(tex: texture_depth_2d, tile: vec2<f32>, grid: vec2<f32>, world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> f32 {{
    let c = shadow_coords(world_pos, light_view_proj);
    if (c.w < 0.5) {{ return 1.0; }}
    return pcf_tile_at(tex, tile, grid, c.xyz);
}}

// `pcf_tile` at `c`: the point in the tile (xy, 0..1 across it) and its depth.
fn pcf_tile_at(tex: texture_depth_2d, tile: vec2<f32>, grid: vec2<f32>, c: vec3<f32>) -> f32 {{
    // One texel, in tile space.
    let tile_texel = grid / vec2<f32>(textureDimensions(tex));
    // Half a texel in from each edge of this tile, in tile space. Sampling
    // exactly ON the boundary already blends the neighbour under linear
    // filtering.
    let guard = tile_texel * 0.5;
    let lo = guard;
    let hi = vec2<f32>(1.0) - guard;

    var sum = 0.0;
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {{
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {{
            // Offset in TILE space, then clamped there, so the kernel never
            // walks out of the tile however close to its edge the sample is.
            let off = vec2<f32>(f32(dx), f32(dy)) * tile_texel;
            let local = clamp(c.xy + off, lo, hi);
            let uv = (local + tile) / grid;
            sum = sum + textureSampleCompareLevel(tex, shadow_samp, uv, c.z);
        }}
    }}
    return sum / 9.0;
}}

// The moving-objects map's row of tiles: the sun's, then the characters',
// then the sun's near tile. See `shadow::SUN_ATLAS_TILES`.
const SUN_ATLAS_GRID: vec2<f32> = vec2<f32>(f32({sun_atlas_tiles}), 1.0);
// The sun's near tile, and how much finer it is. See `shadow::SUN_NEAR_ZOOM`.
const SUN_NEAR_TILE: f32 = f32({sun_near_tile});
const SUN_NEAR_ZOOM: f32 = {sun_near_zoom:?};

// A light's shadow of the moving casters alone -- the characters and the
// doors -- where its shadow `layer` names one of their tiles; 1 elsewhere. See
// `character_shadow_tile`.
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

// WHICH OF THE MOVING CASTERS' TILES HOLDS A LIGHT'S SHADOW of them, or -1:
// the light's shadow `layer` (`params.w`) past the spot slots names one,
// `MAX_SPOT_SHADOWS + k` for tile k. See `shadow::MAX_CHARACTER_SHADOWS`.
// The light's own field rather than a list of lights to compare its index
// with: one subtraction for any number of tiles, where the list was a compare
// a tile (`capsule_params.y` and `.z`, 2026-10-07).
fn character_shadow_tile(layer: i32) -> i32 {{
    return select(-1, layer - {max_spot_shadows}, layer >= {max_spot_shadows});
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
// THE LIGHT A PIXEL'S FOOTPRINT RECEIVES AT A LAMP'S TERMINATOR (2026-09-30).
//
// `max(dot(n, l), 0)` at the pixel's centre is a HARD STEP wherever a mapped
// normal turns away from the lamp: every rock crevice lit at a grazing angle
// flipped whole pixels between lit and black as the head moved. The hallway's
// sconce-lit rock and the brick hall's ceiling over its lamp were the worst
// aliasing in the level against a 3x-supersampled reference
// (`offline_frame::aliasing`), and mipping cannot help: it averages the
// normal, and the clamp comes after.
//
// Box-filtered instead. With `dot(n, l)` spread evenly over `nl +- w` across
// the pixel, the mean of the clamp is `(nl + w)^2 / 4w` inside that band, and
// the share of the pixel facing the lamp -- which gates its highlight -- is
// `(nl + w) / 2w`. At `w = 0` both are the old step exactly. `lights.count.w`
// nonzero switches it off, to measure (the `terminator_aa` lever).
fn terminator_aa(nl: f32) -> vec2<f32> {{
    let w = select(terminator_width, 0.0, lights.count.w != 0u);
    let facing = clamp((nl + w) / max(2.0 * w, 1e-6), 0.0, 1.0);
    return vec2<f32>(select(max(nl, 0.0), 0.5 * (nl + w) * facing, abs(nl) < w), facing);
}}

// `terminator_width` from the NORMAL MAP'S OWN MEASURE of how much the normals
// under this pixel disagree: its filtered sample, averaged unnormalised by the
// mips and by the sampler, comes back shorter the more they spread (Toksvig) --
// `sigma^2 = (1 - |n|) / |n|`, less the 1% a quantised unit normal reads short.
// `dot(n, l)` then varies about `sigma^2 / 2`, and a box of that variance is
// `sqrt(3 sigma^2 / 2)` either side. Capped, so a map seen almost edge-on is
// softened rather than washed flat.
//
// NOT FROM THE SCREEN DERIVATIVES of the normal, which were tried first: a
// quad's derivatives of a minified normal map are themselves aliased, so the
// width jumped from quad to quad -- the hallway's aliasing fell 4%, against
// 11% for this; the brick hall's 5%, against 18% (`offline_frame::aliasing`).
const TERMINATOR_MAX_WIDTH: f32 = 0.5;
fn terminator_width_of(normal_length: f32) -> f32 {{
    let sigma2 = max(1.0 - normal_length - 0.01, 0.0) / max(normal_length, 1e-3);
    return min(sqrt(1.5 * sigma2), TERMINATOR_MAX_WIDTH);
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
        // x^4 as two multiplies, as the baker's `powi(4)` has it: `pow` is
        // exp2(4 log2 x), two transcendental instructions a lamp a pixel.
        let d2_over_r2 = d_over_r * d_over_r;
        let window = clamp(1.0 - d2_over_r2 * d2_over_r2, 0.0, 1.0);
        // A light standing for a lit surface carries its patch's radius below
        // `SURFACE_LIGHT` (`Light::source_radius`): the falloff a disc that
        // wide gives. 0 for every lamp, whose falloff is what it was.
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
    // No highlight where no light arrives: outside a spot's cone `atten` is
    // exactly 0, and the half-vector and its `pow` were computed to be
    // multiplied by it. Most of a room is outside most cones. And only on the
    // share of the pixel that faces the lamp -- see `terminator_aa`. None
    // from a light that stands for a lit surface (`SURFACE_LIGHT`).
    if (aa.y > 0.0 && atten > 0.0 && l.params.w > -1.5) {{
        let h = normalize(l_dir + view_dir);
        let spec = pow(max(dot(n, h), 0.0), shininess) * spec_strength;
        out.specular = radiance * spec * atten * aa.y;
    }}
    return out;
}}

// THE LIT SURFACES' LIGHTS, APART FROM THE LAMPS: the diffuse light
// `light_contribution_split` gives each, summed, and nothing it holds for a
// lamp. Such a light (`Light::is_surface_light`, a flashlight's bounce) casts
// no shadow and makes no highlight, and a spot's edge is a half space or
// wider -- so it is its range, its patch's falloff, a plain ramp across its
// edge (`SURFACE_LIGHT_MIN_BAND`) and the terminator. Through the lamp loop
// each paid the culling pre-pass, the zero test, the half-precision sums and
// the shadows' branches as well, which changed no pixel: the torch's two
// bounce lights were 260 ALU a fragment of the readers and 1.5M clocks of a
// torch view's scene passes, 9% of its frame (per-draw trace, 2026-10-06).
// The caller weights it by the albedo.
fn surface_lights(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {{
    var lit = vec3<f32>(0.0);
    // WALKED AS A MASK, as the lamp loop walks `reaching`: counted to a
    // uniform, the same loop compiled 42-47 instructions bigger in every
    // reader that has it (PIPESTATS, deploy110 against deploy111: 219 against
    // 177 in the baked reader) -- and past about 3,390 a reader's size is its
    // cost on this GPU, through its instruction cache
    // (docs/frame-budget-plan-2026-10-06.md §1.6).
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
        // Past its range the window is exactly zero.
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
        // x^4 as two multiplies, as the baker's `powi(4)` has it: `pow` is
        // exp2(4 log2 x), two transcendental instructions a lamp a pixel.
        let d2_over_r2 = d_over_r * d_over_r;
        let window = clamp(1.0 - d2_over_r2 * d2_over_r2, 0.0, 1.0);
        // A light standing for a lit surface carries its patch's radius below
        // `SURFACE_LIGHT` (`Light::source_radius`): the falloff a disc that
        // wide gives. 0 for every lamp, whose falloff is what it was.
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

    // Blinn-Phong specular, gated on ndotl so a surface facing away from the
    // light gets no highlight. Ungated, the half-vector still lines up on the
    // far side and rims every object with light coming from behind it. And on
    // light arriving at all: outside a spot's cone the highlight would be
    // multiplied by an `atten` of exactly 0. Nor from a light that stands for
    // a lit surface (`SURFACE_LIGHT`).
    if (ndotl > 0.0 && atten > 0.0 && l.params.w > -1.5) {{
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
        if (SPOT_SHADOWS && layer >= 0 && f32(layer) < camera.shadow_params.y) {{
            shadow = shadow * pcf_layer(spot_shadow_tex, layer, world_pos, spot_view_proj(layer));
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

// THE FACE'S ROOM, when its vertex stage chose it: (x) the room, (y) 1 when a
// resident box holds the face, (z) 1 when a doorway's carve does, (w) 1 for
// "given". All zero -- the default -- and `probe_environment` chooses per
// pixel as it always has. Set by the probe pass; see `probe_face_room`.
var<private> probe_face_given: vec4<f32> = vec4<f32>(0.0);

// WHICH ROOM A FACE REFLECTS, from its centre `face_pos` (player frame): the
// half of `probe_environment`'s choice that depends only on the face -- the
// tightest resident box holding the centre, and whether a doorway's carve
// does -- made once a VERTEX. Every vertex of a face carries the same centre,
// so the whole face agrees, as it does when each pixel works it out: the box
// tests are the same, in the same slot order, with the same margin and the
// same 0.1% tie rule. Pass the result to the fragment `flat`, and set
// `probe_face_given` from it before shading.
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

// WHICH PHOTOGRAPHS a surface at `select_world` takes where its reflection is
// not traced, and which ROOM it is in when its face does not say. Made only
// where something needs it; see `probe_environment`.
struct ProbeChoice {{
    // The nearest photograph and the runner-up it blends with, -1 for none,
    // and their SQUARED distances from the surface.
    best: i32,
    second: i32,
    best_dist: f32,
    second_dist: f32,
    // The room they photograph: the face's own when given, else the tightest
    // box holding `volume_world`.
    room: f32,
}}

fn probe_choose(select_world: vec3<f32>, volume_world: vec3<f32>) -> ProbeChoice {{
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
    // than a switch. See the mix at the end of `probe_environment`.
    var second = -1;
    var second_dist = 1e30;
    // THE ROOM IS A PROPERTY OF THE FACE, so a caller whose vertex stage has
    // already chosen it -- `probe_face_room`, once a vertex instead of once a
    // pixel -- hands it in through `probe_face_given`, and only the choice of
    // photograph WITHIN that room, which does depend on the pixel, is made
    // here. See `probe_choose_in_room`.
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

    if (!face_given) {{
        c.best = best;
        c.second = second;
        c.best_dist = best_dist;
        c.second_dist = second_dist;
        c.room = best_room;
    }}
    return c;
}}

// The nearest two of `room`'s photographs to `select_world`: `probe_choose`
// for a face whose room is given. The same comparisons as its full loop, over
// the same slots in the same order: the room's first slot resets exactly as
// `tighter` does there, and the rest compete as `same_room_but_nearer` does.
fn probe_choose_in_room(room: f32, select_world: vec3<f32>) -> ProbeChoice {{
    var c: ProbeChoice;
    c.best = -1;
    c.second = -1;
    c.best_dist = 1e30;
    c.second_dist = 1e30;
    c.room = room;
    // ONE ROOM'S CHAIN, WALKED: this room's few slots, where
    // `probe_nearest_two` looks at all sixteen -- the same two photographs.
    // Per pixel at full resolution in the scene pass's models; the probe
    // pass's hit, over two rooms, scans (exp61-62). Walk against scan, the
    // models' draws timed: exp63-64.
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

// THE FAR END OF A DOORWAY'S OPENING, for the part of a reflection's lobe that
// entered it: `through` is what that part shows as far as it went -- out the
// far end, or the opening's side it met first -- and this blends in the other.
//
// A ray passing a doorway's corner has TWO rims within its footprint: the
// wall's front edge (the wall, or into the opening) and, a wall's depth on,
// its back edge (the jamb's inside face, or out to the sky). Only the first
// was ever blended, so which way the second went decided a whole texel: a
// dark jamb beside the bright sky, a staircase down the reflected doorway
// that crawled as the head moved (headset 22:04:15, 2026-09-29). Across a
// jamb the lobe falls three ways -- the wall (1 - near), the jamb's face
// (near - far) and through (far) -- so inside the opening the far end's share
// is far / near.
fn probe_through_far_end(hit: ProbeHit, d: vec3<f32>, through: vec4<f32>, roughness: f32, dir: vec3<f32>, probe_lod: f32) -> vec4<f32> {{
    let p = (hit.rim_code >> 3u) & 31;
    let axis = (hit.rim_code >> 1u) & 3;
    // Parallel to the wall, the ray never reaches its far face.
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
    // This line out the far end: the other part met the side just before it,
    // a step back inside the wall where the room's photographs saw it. Else
    // this line met the side: the other part goes out, traced from there.
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

// A TRACED HIT'S SECONDARY LOOKUPS, from its colour `primary`: across a
// doorway's rim, and across a solid proxy's outline, each a colour at a point
// already known or a second trace. Made in `probe_environment`, or, where the
// probe pass defers them (`PROBE_SECONDARY_DEFERRED`), in `probe_fixup` from
// what the pass recorded -- the same function, so the same answer.
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
    // ACROSS A DOORWAY'S RIM, the lobe's two parts: what the ray found,
    // and the other side of the rim, weighted by how much of the lobe
    // passes through the opening. See `probe_rim_at`.
    if (hit.rim >= 0.0) {{
        // The wall beside the opening where the ray went through it: a
        // point already known. Else traced again from just inside the
        // opening: whatever the doorway shows there, the next room or
        // outdoors.
        let rim_pos = probe_hit_rim_point(hit, d);
        let went_through = probe_hit_rim_went_through(hit);
        var side = probe_point_hit(rim_pos, probe_hit_rim_room(hit), hit.rim_t);
        if (!went_through) {{
            side = probe_trace(rim_pos, d, -1.0, roughness);
            side.t = hit.rim_t + side.t;
        }}
        if (side.found) {{
            let x = probe_traced_colour(side, d, roughness, dir, probe_lod);
            // What passed into the opening, and what met the wall beside it:
            // one is this hit, the other the lookup.
            var through = select(x, col, went_through);
            let beside = select(col, x, went_through);
            // A NEAR rim's opening also has a far end, where the part of the
            // lobe inside it either leaves or meets the opening's side. See
            // `probe_through_far_end`. Not for a lobe as wide as
            // `PROBE_FAR_END_MAX_ROUGHNESS`'s -- see there.
            if ((hit.rim_code & PROBE_RIM_FAR) == 0 && roughness < PROBE_FAR_END_MAX_ROUGHNESS) {{
                through = probe_through_far_end(hit, d, through, roughness, dir, probe_lod);
            }}
            col = mix(beside, through, hit.rim);
        }}
    }}
    // ACROSS A SOLID PROXY'S OUTLINE, the footprint's two parts: the
    // proxy, and what lies past it, by how much of the footprint the
    // proxy covers. See `probe_proxy_hit`. A texel across a model's own
    // outline carries that cover negated (`PROBE_SUBSAMPLE`).
    let edge_cover = probe_edge_cover(hit.edge_cover);
    if (hit.edge_code >= 0 && edge_cover < 0.99) {{
        // What lies past the proxy, where the ray hit it: traced again as
        // though it were not there. Else the proxy itself, at its outline.
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
    let face_given = PROBE_FACE_ALWAYS_GIVEN || probe_face_given.w > 0.5;
    // Whether a doorway's carve holds the face: also the face's, so also given.
    var in_doorway = false;
    if (face_given) {{
        in_doorway = probe_face_given.z > 0.5;
    }} else {{
        in_doorway = probe_portal_holding(volume_world) >= 0;
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
    //
    // THE TRACE NEEDS ONLY THE ROOM, which a given face carries. The
    // photographs are chosen before it only where it cannot start without
    // them -- no room given, or a face in a doorway -- and then only to find
    // the room; they are chosen (again) after it, where it found nothing.
    // Chosen once, first, on every pixel, the choice was carried unused
    // through the whole trace in registers the probe pass is short of (36%
    // wave occupancy, 2026-09-28). The same choice either way: see
    // `probe_choose`.
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
    // ONE COLOUR LOOKUP A HIT, never two. A hit that left the rooms is
    // coloured from what lies out there, and only that: the photographs of
    // the doorway it left through were read as well, and thrown away, on
    // every such pixel -- two depth and two colour reads each time, three
    // times over with the rim and outline lookups below. See
    // `probe_traced_colour`.
    let hit = probe_trace(world_pos, d, trace_room, roughness);
    if (hit.found) {{
        // A MODEL ON CARDS is coloured by the fix-up in a pass that defers:
        // see `probe_hit_carded`.
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
// `ProbeHit::edge_cover`'s sign for a texel that straddles one of a model's
// own outlines -- a lampshade's rim against its lit mouth, both the model --
// where one ray decides the texel all rim or all mouth: the line between them
// crawled as the head moved (offline crawl, 2026-09-30). The cover's size is
// still the model's presence in a rough lobe, blended over what lies past it
// as any model outline is. `probe_fixup` traces the texel again, rays spread
// across its footprint or its lobe, and averages them; a pass that does not
// defer keeps its one ray.
const PROBE_SUBSAMPLE: f32 = -1.0;
// `ProbeHit::edge_cover` for a texel that meets a model on cards away from its
// outlines: the model's presence, less four. `probe_fixup` traces its one ray
// again with the cards' tests filtered (`PROBE_CARD_TESTS_FILTERED`) -- which
// is what held the lamps' insides still in the walls -- rather than four rays
// across it, which cost the headset 0.8 ms a frame for the last few percent
// (2026-10-01). Read the cover back with `probe_edge_cover`.
const PROBE_RETEST: f32 = -4.0;
// The cover a hit's `edge_cover` carries, whatever it marks.
fn probe_edge_cover(edge_cover: f32) -> f32 {{
    let c = abs(edge_cover);
    return select(c, c + PROBE_RETEST, c > 2.0);
}}
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
    // WHERE THE TRACE STARTED, held inside its first room or doorway: the rim
    // and outline points below lie on the ray from here, `rim_t` and `edge_t`
    // along it, and are worked out from that when they are needed.
    origin: vec3<f32>,
    // A DOORWAY'S RIM INSIDE THE LOBE, the first one the ray met: how much of
    // the lobe passes through the opening (0..1), or -1 for none; `rim_t`
    // along the ray. The rest is packed in `rim_code`: whether the ray went
    // through the opening (bit 0), the axis crossed (bits 1-2), the doorway
    // (bits 3-7) and the room the rim was met from, plus one (bits 8-). The
    // other side of the rim is `probe_hit_rim_point`: just inside the opening
    // when the ray hit the wall, on the wall just outside when it went
    // through. See `probe_rim_at` and `probe_secondary`.
    //
    // PACKED, and the points not carried, because this rides through every
    // hop of the trace in a pass whose occupancy is set by its register peak:
    // 15 values, now 6 (and the origin, which the trace holds anyway).
    rim: f32,
    rim_t: f32,
    rim_code: i32,
    // A SOLID PROXY'S OUTLINE INSIDE THE FOOTPRINT, the first one the ray
    // passed: how much of the footprint it covers there (0..1) and where it is
    // along the ray; `edge_code` -1 for none, else whether this ray hit it
    // (bit 0), which proxy (bits 1-5) and its room plus one (bits 6-). See
    // `probe_proxy_hit`, `probe_hit_edge` and `probe_secondary`.
    edge_cover: f32,
    edge_t: f32,
    edge_code: i32,
}}

// The packed fields of a `ProbeHit`, unpacked. See `rim_code` and `edge_code`.
fn probe_hit_rim_went_through(h: ProbeHit) -> bool {{
    return (h.rim_code & 1) != 0;
}}
fn probe_hit_rim_room(h: ProbeHit) -> f32 {{
    return f32(((h.rim_code >> 8u) & 31) - 1);
}}
// The other side of the rim: see `probe_rim_point` and `probe_rim_point_far`.
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

// A rim at the FAR end of a doorway's opening: see `probe_trace`. Bit 13 of
// `rim_code`, above the room (bits 8-12).
const PROBE_RIM_FAR: i32 = 8192;

// The point across a doorway's FAR rim from `e`, which lies on the plane where
// the opening ends -- the wall's far face, or the next room's box. Inside the
// opening when `into` (the ray met the opening's side before it: the other
// side is the way through); else ON the side it just missed -- the lintel's
// underside, a jamb -- a step back inside the wall, where the room's
// photographs saw it.
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
// The proxy whose outline the ray passed, or -1.
fn probe_hit_edge(h: ProbeHit) -> i32 {{
    return select(-1, (h.edge_code >> 1u) & 31, h.edge_code >= 0);
}}
fn probe_hit_edge_hit(h: ProbeHit) -> bool {{
    return h.edge_code >= 0 && (h.edge_code & 1) != 0;
}}
fn probe_hit_edge_room(h: ProbeHit) -> f32 {{
    return f32((h.edge_code >> 6u) - 1);
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
// THE ROUGHEST SURFACE WHOSE DOORWAY RIMS GET THEIR FAR END BLENDED TOO
// (`probe_through_far_end`). The far end was for the marble floor's saw-tooth,
// where a narrow lobe sees the two rims apart; a lobe this wide already spans
// both, and on brick its early-out almost never fires -- so every rim record
// (14% of the brick hall's pass texels) paid another trace for nothing:
// offline, dropping it changes 0.0035% of brick_hall_diagonal's pixels by more
// than 6 levels, none visibly (2026-09-30). The near rim's blend stays at every
// roughness: without it rough stone catches the doorway's light in hard patches.
const PROBE_FAR_END_MAX_ROUGHNESS: f32 = 0.45;

// HOW MANY OF THE PASS'S OWN PIXELS A REFLECTED EDGE IS SOFTENED OVER, each
// side of it: a doorway's rim, a solid proxy's outline, a model's. See
// `PROBE_EDGE_FOOTPRINTS` on the Rust side.
const PROBE_EDGE_FOOTPRINTS: f32 = {probe_edge_footprints:?};

// The half-width, in metres at `t` along a reflected ray, that an edge met
// there is softened over: `PROBE_EDGE_FOOTPRINTS` of this pixel's footprint,
// widened as the pixel's cone is by the path from the eye.
fn probe_pixel_spread(t: f32) -> f32 {{
    return PROBE_EDGE_FOOTPRINTS * pixel_footprint * (1.0 + t / max(probe_eye_distance, 0.05));
}}
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
    // This room's doorways only, in order. See `probe_room_portal`.
    for (var p = probe_room_portal(room); p >= 0; p = probe_portal_next(p, room)) {{
        if (i32(camera.probe_portals[p * 3].w) != axis) {{
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
    return probe_seen_distance_of(t, v);
}}
// `probe_seen_distance` from its depth texel `t`, read already.
fn probe_seen_distance_of(t: vec4<f32>, v: vec3<f32>) -> f32 {{
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
    let t = textureSampleLevel(probe_depth, probe_depth_samp, v, i32(camera.probe_boxes[slot * 3].w), 0.0);
    return probe_clearance_of(t, v);
}}
// `probe_clearance` from the depth texel `t` read along `v`, the point less
// the capture point.
fn probe_clearance_of(t: vec4<f32>, v: vec3<f32>) -> f32 {{
    let s = probe_seen_distance_of(t, v);
    return select(s - length(v), -3.4e38, s < 0.0);
}}

// A resident slot photographing `room`, or -1. Any one serves for the room's
// BOX: every cell of a room carries the room's box.
//
// LOOKED UP, NOT SEARCHED: rooms are numbered 0.. in order of their first slot
// (`ProbeUpload::dense_rooms`), and this is that first slot -- the one a scan
// of every slot would have stopped at. A room with no resident slot, -1 ("in
// no room") included, is outside the table.
fn probe_room_slot(room: f32) -> i32 {{
    if (!(room >= 0.0 && room < 16.0)) {{
        return -1;
    }}
    let r = i32(room);
    return i32(camera.probe_rooms[r >> 2u][r & 3]);
}}
// The next slot of the same room after `slot`, ascending, or -1. Walking from
// `probe_room_slot` visits a room's slots in the order a scan of every slot
// would, so every tie between two photographs is broken the same way.
fn probe_slot_next(slot: i32) -> i32 {{
    return i32(camera.probe_rooms[4 + (slot >> 2u)][slot & 3]);
}}

// THE NEAREST TWO PHOTOGRAPHS OF ROOM `a` OR ROOM `b` to `p`, -1 for none,
// and their SQUARED distances. Every slot the camera holds is looked at, in
// ascending order and with the same strict comparisons as walking the two
// rooms' chains merged (`probe_room_slot`, `probe_slot_next`): the same slots
// in the same order, so the same two, ties included.
//
// LOOKED AT, NOT WALKED: every slot in turn, at an index the loop counts,
// not one the previous read had to supply. Walking a chain read the table at
// places only the previous read could say, each read waiting out the one
// before -- and through the memory path: indices the compiler cannot see keep
// a table out of constant memory (Qualcomm, *Adreno GPU best practices*). The
// walk in `probe_hit_colour` was a quarter of the probe pass's time (stage
// counters, 2026-10-01: 478K clocks a tile with it, 349K with no choice made
// at all).
//
// A LOOP, NOT WRITTEN OUT, by measurement: per tile of the probe pass,
// hall_front, 2026-10-01 (exp61), the loop 514K clocks, the same scan written
// out one block a slot (`probe_nearest_two_unrolled`) 556K and 563K, the walk
// 576K. Slots past the live count keep stale rooms, so the loop stops there --
// the same count for every pixel, so the stop costs no divergence.
struct ProbeNearest {{
    s0: i32,
    s1: i32,
    d0: f32,
    d1: f32,
}}
const PROBE_MAX_SLOTS: i32 = {max_probes};
// `probe_nearest_two` written out, one block a slot, reading the table at
// fixed places: the same slots, order and result. Measured slower (above);
// kept for the `def_scan_unrolled` cut.
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
// `probe_nearest_two_unrolled` reading `probe_select`: the `def_scan_const_unrolled` cut.
fn probe_nearest_two_const_unrolled(p: vec3<f32>, a: f32, b: f32) -> ProbeNearest {{
    let count = i32(camera.probe_params.x);
    var n = ProbeNearest(-1, -1, 3.4e38, 3.4e38);
{probe_nearest_const_unrolled}    return n;
}}
// `probe_nearest_two` reading its own uniform block, `probe_select`, in place
// of the camera's: the same bytes, so the same slots, order and result.
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
// The first doorway of `room`, or -1; then each one's next in that room,
// ascending -- onward through whichever of its two sides IS that room. The
// same doorways, in the same order, as a scan of all of them for the room.
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
// The first proxy standing in `room`, or -1; then each one's next, ascending.
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

// -2 - `i` for a MODEL proxy -- a lamp, not a brush piece -- else -1. Carried
// in a hit's `other`, which a hit on a proxy has no use for, so the colour
// lookup knows to shade whatever no photograph saw as the model itself. See
// `probe_model_colour`.
fn probe_proxy_model(i: i32) -> f32 {{
    var m = -1.0;
    if (i >= 0 && camera.probe_proxies[i * 3 + 1].w > 0.5) {{
        m = -2.0 - f32(i);
    }}
    return m;
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
    // The nearest model box with a distance field that the ray enters:
    // walked once, after the loop. See the walk below.
    var field_i = -1;
    var field_near = 3.4e38;
    var field_far = 0.0;
    // What stands in this room only, in order. See `probe_room_proxy`.
    for (var i = probe_room_proxy(room); i >= 0; i = probe_proxy_next(i)) {{
        if (i == skip) {{
            continue;
        }}
        // Into the box's own frame. SKIPPED for an unrotated box -- the
        // pillar, and every model standing square to the room -- where the
        // rotation by the identity is exactly the input: two quaternion
        // rotations a proxy a hop a pixel, for nothing. The test is on the
        // uniform, so a whole wave takes the same branch.
        let q = camera.probe_proxies[i * 3 + 2];
        var lo = o - camera.probe_proxies[i * 3].xyz;
        var ld = d;
        if (any(q != vec4<f32>(0.0, 0.0, 0.0, 1.0))) {{
            let qi = vec4<f32>(-q.xyz, q.w);
            lo = probe_quat_rotate(qi, lo);
            ld = probe_quat_rotate(qi, ld);
        }}
        // Per proxy, NOT hoisted out of the loop for the unrotated ones: the
        // probe pass is register-bound (36% occupancy), and a reciprocal kept
        // live across the whole loop made it 0.4 ms slower on the headset
        // (2026-09-28) -- three divisions a proxy are cheaper than a register.
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
            let footprint = max(t_edge * lobe, probe_pixel_spread(t_edge));
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
            }} else if (camera.probe_proxies[i * 3 + 1].w > 1.5) {{
                // A MODEL WITH A DISTANCE FIELD: noted, and walked after the
                // loop -- the nearest such box the ray enters.
                if (t_in < field_near) {{
                    field_i = i;
                    field_near = t_in;
                    field_far = min(far, t1);
                }}
            }} else {{
                // A model without one: the photographs' guess.
                let t_surface = probe_proxy_surface(o, d, t_in, min(far, t1), room, camera.probe_proxies[i * 3].xyz);
                if (t_surface < best) {{
                    best = t_surface;
                    out.index = i;
                }}
            }}
        }}
    }}
    // THE NEAREST MODEL'S OWN SHAPE, walked out here rather than inside the
    // loop: there, every proxy's slab test stayed live beside the walk, and
    // the probe pass went from 22 registers to 25 -- 50% of its waves in
    // flight to 37%, for every texel, lamp or none (PIPESTATS, 2026-09-29).
    // A ray through one lamp's empty box to another lamp behind it sees only
    // the first: rare, and shown as the wall past it.
    if (field_i >= 0 && field_near < best) {{
        let walk = probe_proxy_field_at(o, d, field_i, field_near, min(field_far, best), lobe);
        let touched = walk.x < best;
        if (touched) {{
            best = walk.x;
            out.index = field_i;
        }}
        // HOW MUCH OF THE REFLECTION THE MODEL IS, blended over what lies past
        // it (`ProbeHit::edge`), as a solid proxy's outline is:
        //
        // - A NEAR MISS within a footprint -- a pixel's, on a mirror -- is its
        //   outline, faded out over the footprint. A field has no inside to
        //   measure how deep a hit went, so only misses are faded, outside the
        //   outline. One ray a texel decided each texel all or nothing, and
        //   the sconce's reflection in the polished doorway jamb came out in
        //   stair-steps (headset, 2026-09-29).
        // - ON A ROUGH SURFACE the lobe is as wide as the model or wider, and a
        //   model that fills only part of it is only part of the reflection:
        //   extent^2 / (extent^2 + spread^2), the share of the lobe's area it
        //   can cover. The rock ceiling over each sconce, 0.44 rough, showed the
        //   fixture as a hard shape half a metre off (offline, 2026-09-29).
        let t_at = select(walk.y, walk.x, touched);
        let extent = (2.0 / 3.0) * dot(camera.probe_proxies[field_i * 3 + 1].xyz, vec3<f32>(1.0));
        let spread = t_at * lobe;
        let presence = extent * extent / max(extent * extent + spread * spread, 1e-8);
        let cover = select(presence * (1.0 - smoothstep(0.0, 1.0, walk.z)), presence, touched);
        // A MODEL ON CARDS is coloured again wherever the ray meets it, not
        // only past an outline: each card vouches for a point by tests read
        // at full size (see `probe_card_vote`), and read once, unfiltered, they
        // let a lamp's bright inside come and go in its reflection in the
        // polished walls as the head moved a millimetre -- the shimmer left
        // once the walls and floors held still (headset and offline,
        // 2026-10-01). Only texels that meet a lamp pay for it. See
        // `PROBE_RETEST`.
        let carded = camera.proxy_cards[field_i >> 2u][field_i & 3] > 0.5;
        if (out.edge < 0 && touched && walk.z < 0.0) {{
            // Past one of its own outlines: the texel is sampled again,
            // several rays across it, and then blended over what lies past the
            // model by its presence, as below. See `PROBE_SUBSAMPLE`.
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

// `probe_proxy_field` for proxy `i`: the ray into its box's own frame first.
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

// Steps a ray may take through a model's distance field.
const PROXY_FIELD_STEPS: i32 = 32;

// WHERE INSIDE A MODEL'S BOX THE MODEL IS, by walking its distance field:
// `lo`, `ld` the ray in the box's own frame, `half` the box, `t_in`..`t_out`
// the stretch inside it. Each step goes as far as the field says is empty, at
// least the stop distance, until it is within that of the surface (a hit) or
// leaves the box (3.4e38). See `proxy_field` -- the surface itself, where
// `probe_proxy_surface` below could only guess it from 256-pixel photographs.
//
// Returns (hit, near_t, near): the hit, or 3.4e38; and the CLOSEST the ray
// came to the model on the way, `near` footprints past the surface at
// `near_t` -- the footprint as at a solid proxy's outline, the wider of the
// lobe and the pixel there. The steps shorten as the ray grazes the model, so
// they sample that minimum where it matters.
//
// A HIT PAST ONE OF THE MODEL'S OWN OUTLINES -- the ray passed within a
// footprint of it, drew a footprint away again, and met the model further
// on, as through a lampshade's mouth past its rim -- returns `near` -1
// instead: the texel straddles the outline, and one ray would show it all
// rim or all mouth. See `PROBE_SUBSAMPLE`.
fn probe_proxy_field(lo: vec3<f32>, ld: vec3<f32>, half: vec3<f32>, t_in: f32, t_out: f32, field: i32, lobe: f32) -> vec3<f32> {{
    let slot = camera.proxy_fields[field * 3];
    let size = camera.proxy_fields[field * 3 + 1];
    // The ray straight in the atlas's coordinates, `uvw = fo + fd * t`: two
    // vectors live through the walk instead of the box and the slot.
    let scale = size.xyz * (0.5 / max(half, vec3<f32>(1e-4)));
    let fo = slot.xyz + 0.5 * size.xyz + lo * scale;
    let fd = ld * scale;
    // Half a texel inside the field, so filtering never reaches the gap of
    // "far" packed between fields.
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
            // ON the stop surface, not wherever the step that crossed it
            // landed: anywhere in a band half a field sample deep (1.5 cm on
            // a hanging lamp), and which step lands there changes with the
            // ray -- so the hit, and the card texel read there, jumped as the
            // head moved a millimetre, and the bright inside a lamp's cards
            // show came and went in its reflection (headset, 2026-10-01).
            // The field is smooth there: two secant steps settle on it.
            var th = t + (dist - size.w);
            th = th + (textureSampleLevel(proxy_field, probe_samp, clamp(fo + fd * th, lo_uvw, hi_uvw), 0.0).r * slot.w - size.w);
            hit = clamp(th, t_in, t_out);
            break;
        }}
        // No wider than half the field's reach, where its distances are exact:
        // past that the field says only "at least this far", and a wider fade
        // ran out at the model's box -- a box-shaped shadow of every lamp in the
        // rough ceiling above it (offline, 2026-09-29).
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
    // The room's two photographs nearest the object: see `probe_nearest_two`.
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

// The doorway the exit point `e` of `room` lies in, crossed along `axis`; -1
// where `e` is on a wall. The portal's box is the carve through the wall, so
// `e` -- on the room's face -- lies inside it exactly when it is in the opening.
fn probe_portal_at(e: vec3<f32>, room: f32, axis: i32) -> i32 {{
    // This room's doorways only, in order. See `probe_room_portal`.
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
    hit.origin = world_pos;
    hit.rim = -1.0;
    hit.rim_t = 0.0;
    hit.rim_code = -1;
    hit.edge_cover = 0.0;
    hit.edge_t = 0.0;
    hit.edge_code = -1;
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
        // OUTDOORS THERE ARE NO WALLS. The outdoor volume's box only stands in
        // for the sky dome, and a ray run to it met a floor metres under the
        // grass. A reflection starting out here -- off an outside wall, the
        // roof -- is out already: `outdoor_radiance` takes it to the ground or
        // the sky. `portal_params.z` names the outdoor room plus one, 0 none.
        if (cur + 1.0 == camera.portal_params.z) {{
            hit.pos = hit.origin + d * t0;
            hit.room = cur;
            hit.found = true;
            hit.escaped = true;
            hit.t = t0;
            return hit;
        }}
        // Where the ray leaves the box, along the ray from its origin: the
        // ray is in the box from its entry on, so the slabs' far sides are
        // the exit.
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
        // From `t0`: past the first room, where the ray LEFT the room before
        // -- through the doorway's wall, where a door's leaf hangs (see
        // `space_soup_engine::reflection_proxy::door_proxies`).
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
        // The first doorway rim within the lobe, whichever side of it this
        // ray lands on. See `probe_rim_at`.
        //
        // AT LEAST A PIXEL WIDE, as a solid proxy's outline is: on polished
        // marble the lobe is millimetres across, one ray decided each texel
        // all or nothing, and the front door's lintel came out of the floor's
        // reflection as a staircase that crawled as the head moved (headset,
        // 2026-09-29).
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
            max((face[axis] - hit.origin[axis]) * inv[axis], t_exit),
            max((wall_far - hit.origin[axis]) * inv[axis], t_exit),
            escapes
        );
        // THE FAR END OF THE OPENING is a rim too: there the ray either leaves
        // the wall -- outdoors, into the next room -- or has met the opening's
        // side just before, the lintel's underside or a jamb. Only the near end
        // was softened, and there both sides are the same marble; out here one
        // is the sky, and the front door's lintel came out of the floor's
        // reflection as a staircase (headset, 2026-09-29). Softened over the
        // footprint as the near rim is. See `probe_rim_point_far`.
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
        // The next room's proxies are met from this room's exit on: a shut
        // door stands in the wall between the two boxes.
        t0 = t_exit;
    }}
    return hit;
}}

// The ground trace's finest level, the level it starts from, its readings of
// the ground across a finest cell and the most cells it visits. Emitted from `ground_map::GROUND_TRACE_*`, which its CPU
// twin `ground_map::trace` uses.
const GROUND_TRACE_FINEST_LEVEL: i32 = {ground_trace_finest};
const GROUND_TRACE_START_LEVEL: i32 = {ground_trace_start};
const GROUND_TRACE_READINGS: i32 = {ground_trace_readings};
const GROUND_TRACE_MAX_STEPS: i32 = {ground_trace_steps};

// THE SKY A REFLECTION SEES along WORLD direction `d`: the panorama without its
// sun, from its layer of the probe array, at the blur `lod` asks for -- the
// same prefiltered chain a probe has, so a rough surface blurs it exactly as
// much as a photograph. The sun reaches every surface as a light, and its
// highlight is that light's; reflected here too, it would be counted twice.
// Where no layer was built, the sky's harmonics, which want the direction in
// the player's frame: `dir`.
fn sky_reflection(d: vec3<f32>, dir: vec3<f32>, lod: f32) -> vec3<f32> {{
    // Stored plus one: 0 is none. See `Uniforms::sky_params`.
    let layer = i32(camera.sky_params.y) - 1;
    if (layer < 0) {{
        return environment_radiance(dir);
    }}
    return textureSampleLevel(probe_cube, probe_samp, d, layer, lod).rgb;
}}

// Where world point `p` lies on the ground map, 0..1 across the terrain.
fn ground_uv(p: vec3<f32>) -> vec2<f32> {{
    return (p.xz - camera.ground_params.xy) * camera.ground_params.zw;
}}

// THE GROUND A RAY MEETS, from WORLD point `e` along `d`: how far along it the
// ray first passes below the ground, or -1 where it meets none -- off the
// terrain, or above its highest ground and rising.
//
// A walk through the ground map's cells, coarse where the ray passes high and
// fine where it comes close: above level 0 each texel's A is the highest
// ground under it (`ground_map::levels`). A cell whose highest ground stays
// below the ray on both sides is passed over whole, and the walk climbs a
// level once it leaves its parent; a cell the ray may touch is entered a level
// finer, from where the ray has come down to its highest ground. At the finest
// level the ground itself is read evenly across the rest of the cell, and the
// crossing interpolated between the first reading below the ray and the one
// before. Exact to the ground's own shape, so the horizon it finds moves only
// when the ray does.
//
// `ground_map::trace` is this, step for step, and a GPU test holds the two to
// the same answers. It replaced a march in doubling steps that tested the
// ground at 32 m and next at 64 m, past test_room's edge: the hills between
// were never tested, a reflection's horizon was the ground's height 32 m out,
// it stepped wherever that reading changed hands -- a wall mirrored a
// building that was not there -- and it crawled as the head moved (headset,
// 2026-09-29).
fn ground_trace(e: vec3<f32>, d: vec3<f32>) -> f32 {{
    return ground_trace_until(e, d, 3.4e38);
}}

// `ground_trace`, giving up past `t_max` -- where the ray has already met a
// building, the ground beyond it cannot be seen.
fn ground_trace_until(e: vec3<f32>, d: vec3<f32>, t_max: f32) -> f32 {{
    let top = camera.sky_params.z;
    let size = f32(textureDimensions(ground_map, 0).x);
    let scale = camera.ground_params.zw * size;
    // The ray in level-0 texels across the map; an axis it barely moves along
    // never bounds a cell it is in.
    let q0 = (e.xz - camera.ground_params.xy) * scale;
    let raw = d.xz * scale;
    let dq = select(raw, vec2<f32>(1e-6), abs(raw) < vec2<f32>(1e-6));
    let inv = 1.0 / dq;
    // On the map, and below its highest ground.
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
    // A thousandth of a texel along the ray: a point on a border is in the
    // cell ahead.
    let nudge = sign(dq) * 1e-3;
    var hit = -1.0;
    for (var steps = 0; steps < GROUND_TRACE_MAX_STEPS && t < t_out; steps = steps + 1) {{
        let cell = f32(1u << u32(level));
        let last = floor(size / cell) - 1.0;
        let c = clamp(floor((q0 + dq * t + nudge) / cell), vec2<f32>(0.0), vec2<f32>(last));
        let exits = ((c + ahead) * cell - q0) * inv;
        let t_exit = min(min(exits.x, exits.y), t_out);
        let highest = textureLoad(ground_map, vec2<i32>(c), level).a;
        let y_in = e.y + d.y * t;
        if (min(y_in, e.y + d.y * t_exit) <= highest) {{
            // Nothing in this cell before the ray is down to its highest ground.
            let t_top = select(t, t + (y_in - highest) / -d.y, y_in > highest);
            if (level > finest) {{
                level = level - 1;
                t = t_top;
                continue;
            }}
            // The ground itself, read evenly from there to the cell's far
            // side; the crossing between the first reading below the ray and
            // the one before it. Every reading is taken -- no exit between
            // them -- so the reads go out together rather than one by one.
            let span = (t_exit - t_top) / f32(GROUND_TRACE_READINGS - 1);
            var f_prev = e.y + d.y * t_top - textureSampleLevel(ground_map, probe_samp, (q0 + dq * t_top) / size, 0.0).a;
            var crossing = select(-1.0, t_top, f_prev <= 0.0);
            for (var k = 1; k < GROUND_TRACE_READINGS; k = k + 1) {{
                let tk = t_top + span * f32(k);
                let f = e.y + d.y * tk - textureSampleLevel(ground_map, probe_samp, (q0 + dq * tk) / size, 0.0).a;
                if (crossing < 0.0 && f <= 0.0) {{
                    crossing = tk - span + span * f_prev / (f_prev - f);
                }}
                f_prev = f;
            }}
            if (crossing >= 0.0) {{
                hit = crossing;
                break;
            }}
        }}
        // Over this cell: on to the next, a level coarser once it is in
        // another parent.
        let next = floor((q0 + dq * t_exit + nudge) / cell);
        if (any(floor(next * 0.5) != floor(c * 0.5))) {{
            level = min(level + 1, top_level);
        }}
        t = t_exit;
    }}
    return hit;
}}

// THE OUTDOORS ALONG A RAY from WORLD point `e` heading `d`: the ground where
// the ray first passes below it (`ground_trace`), else the sky. The ground is
// read at the blur the lobe has spread to where it lands, against the size of
// a texel. See `ground_map`.
//
// Before, the ray was run into the outdoor volume's BOX -- whose floor lies
// metres under the grass -- and the outdoor photograph read toward that point:
// the shaded wall facing the lake reflected sunlit ground at its foot, lit as
// if from underneath, and a seam crossed it at eye height (headset, 2026-09-28).
//
// And the other BUILDINGS: a ray that meets one before the ground or the sky
// shows its outside (`building_hit`, `building_colour`). Until 2026-09-29 the
// marble hall's outer walls mirrored the hills standing where the brick hall is.
fn outdoor_radiance(e: vec3<f32>, d: vec3<f32>, dir: vec3<f32>, roughness: f32, lod: f32) -> vec3<f32> {{
    let b = building_hit(e, d);
    // From `e`, which is where the caller's own reach ends. See `probe_reach`.
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

// THE NEAREST BUILDING a ray from WORLD point `e` along `d` enters, as
// `[t, slot]`, or t < 0 for none. A box the ray starts in or on -- the building
// it is leaving -- is behind it. See `Uniforms::building_boxes`.
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

// A BUILDING'S OUTSIDE where a reflection meets it: the baked cube whose six
// faces are its six sides (the baker's `probe::capture_exterior`), read along
// `(p - centre) / half_size` -- that is the cube direction of the box point --
// at the blur the lobe has spread to over `t`, against the size of a texel on
// its faces. Alpha is what the photograph covered; the rest is sky.
fn building_colour(p: vec3<f32>, slot: i32, roughness: f32, t: f32) -> vec4<f32> {{
    let lo = camera.building_boxes[slot * 2];
    let hi = camera.building_boxes[slot * 2 + 1].xyz;
    let half = max((hi - lo.xyz) * 0.5, vec3<f32>(1e-3));
    let texel = 2.0 * max(max(half.x, half.y), half.z) / f32(textureDimensions(probe_cube).x);
    let spread = max(t * probe_lobe_tan(roughness), 1e-4);
    let lod = clamp(log2(spread / texel), 0.0, PROBE_MAX_LOD);
    return textureSampleLevel(probe_cube, probe_samp, (p - (lo.xyz + hi) * 0.5) / half, i32(lo.w), lod);
}}

// THE COLOUR OF A REFLECTION THAT IS OUTDOORS: one that left the rooms through
// a doorway at `e`, or that started on an outside wall, heading WORLD `d`. What
// is out there is the ground and the sky, so that is what it shows; `sky_dir`
// is `d` in the player's frame, for the harmonics.
//
// It used to be read from a photograph -- a room's own looking out through the
// same doorway, else the outdoor volume's -- and none of them sees the ground
// just past a door or beside a wall, the building hides it, nor the sky above
// a door from deep inside the room. The floor mirrored the front door as a
// flat pale patch where Cycles shows clouds, and the ceiling mirrored the
// sunlit grass outside as a blocky dim patchwork (headset, 2026-09-28).
fn probe_escape_colour(e: vec3<f32>, d: vec3<f32>, sky_dir: vec3<f32>, roughness: f32, lod: f32) -> vec4<f32> {{
    return vec4<f32>(outdoor_radiance(e, d, sky_dir, roughness, lod), 1.0);
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
// A MODEL WITH NO FIELD of its own, where no photograph saw it: a mid-grey
// surface. See `probe_model_colour`.
const PROBE_MODEL_ALBEDO: f32 = 0.2;

// A MODEL'S OWN SURFACE WHERE NO PHOTOGRAPH SHOWS IT: the model's mean colour,
// lit by the room -- the nearest photograph at its blurriest, the roughness-one
// lobe, towards `n`: the side facing the ray that met it.
//
// The photographs are taken at a person's height, below every lamp: nothing
// saw a lamp's top, and the photograph read in that direction instead showed
// the lamp's glowing mouth underneath -- a bright ghost of each sconce in the
// ceiling above it and on the wall beside it (headset, 2026-09-29).
fn probe_model_colour(proxy: i32, n: vec3<f32>, slot: i32) -> vec4<f32> {{
    let lit = textureSampleLevel(probe_cube, probe_samp, n, i32(camera.probe_boxes[slot * 3].w), PROBE_MAX_LOD);
    let kind = camera.probe_proxies[proxy * 3 + 1].w;
    let field = camera.proxy_fields[max(i32(kind) - 2, 0) * 3 + 2].xyz;
    let albedo = select(vec3<f32>(PROBE_MODEL_ALBEDO), field, kind > 1.5);
    return vec4<f32>(albedo * lit.rgb, 1.0);
}}

// HOW FAR A PHOTOGRAPH TAKEN AT `c` SHOWS A MODEL as a ray along `d` sees it
// at `h`: all of it within about 25 degrees of the ray's own direction, none
// past 60. A model is where a view's angle matters: a lampshade is a thin
// shell with a glowing inside and a dark outside, and a photograph taken from
// under it shows the glow wherever it looks -- read for a ray coming down onto
// the shade from the ceiling, blurred at the ceiling's roughness, the glow
// spilled over the dark outside, and each sconce's ghost glowed in the rock
// above it (offline, 2026-09-29). A depth test cannot see this: on a shell
// the two sides are millimetres apart.
fn probe_view_trust(d: vec3<f32>, c: vec3<f32>, h: vec3<f32>) -> f32 {{
    return smoothstep(0.5, 0.9, dot(d, normalize(h - c)));
}}

// ONE CARD'S SAY about the box-frame point it shows at `uv`, `t` of the way
// into the box along its axis `e`: its colour, read at mip `lod` -- the
// reflection's footprint -- and its trust, read at full size from the row
// below (`row + 1`): which way the surface there faces and the range of depths
// it lies in (see `proxy_cards`). Read coarse, the tests averaged a shade's
// inside into "straight down", facing every reflection looking up, and the
// glowing inside vouched for the dark outside all round the rim. The card is
// trusted as far as
// - the surface it saw FACES THE RAY `ld` -- strictly: a surface only
//   grazing it counts for nothing. A lampshade is a shell millimetres thick,
//   lit inside by its bulb and dark outside: at any point on it the card on
//   one side saw the outside and the card on the other side the inside, both
//   within any depth tolerance, and only their facing tells them apart, since
//   the two faces' normals are opposite. At the shade's silhouette the inside
//   is a hair past grazing, and half-trusted there -- where the wall plate
//   hid the outside from every card but the one below -- it drew white specks
//   down both edges of the reflected shade (offline, 2026-09-30);
// - the hit is not BEHIND the surface it saw there -- by more than a
//   millimetre along that surface's normal -- nor far in front of the nearest
//   surface round it (`depth` the box's depth along the axis; the field walk
//   stops up to `field_stop` short of a surface, plus 5 mm). Behind is the
//   test that counts: a wall two millimetres thick is thinner than a card's
//   texel, and at a bell-shaped shade's silhouette the card looking up saw the
//   inside a centimetre below the outside the ray met, where the flare turns
//   it toward the ray -- within any depth range, but the hit is behind it;
// -- how far, in `sure` -- and weighed among the cards by how squarely that
// surface faces this one, in `w`. Two measures, not one: the inside of a
// steep shade is seen only by the card below, and only at a slant, and a
// confidence read from its slant alone left the glowing mouth grey. Read
// half a texel inside the card, so the filter never reaches the next one.
struct ProbeCardVote {{
    rgb: vec3<f32>,
    w: f32,
    sure: f32,
    // Which card, and where on it: its number, u and v.
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
        // See `PROBE_CARD_TESTS_FILTERED`: the four texels' verdicts, blended.
        let p = clamp(uv * res, vec2<f32>(0.5), vec2<f32>(res - 0.5)) - 0.5;
        let i0 = floor(p);
        let f = p - i0;
        let lo = vec2<i32>(origin + vec2<f32>(0.0, res));
        let hi = lo + vec2<i32>(i32(res) - 1);
        let a = vec2<i32>(i0) + lo;
        let b = min(a + vec2<i32>(1), hi);
        // Metres from each texel's centre to the point, across the card: each
        // texel's depth is carried there along its own surface before it is
        // compared -- a sloped surface's neighbours lie centimetres deeper or
        // shallower than the point, and compared as they are, a sphere's own
        // texels failed the millimetre test round every hit.
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
    // Stored compressed, so every filtered read averages as a display would:
    // expanded back to radiance. See `proxy_cards`.
    vote.rgb = card.rgb / max(1.0 - dot(card.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-3);
    vote.w = trust.x;
    vote.sure = trust.y;
    vote.card = vec3<f32>(face, uv);
    return vote;
}}

// One test-row texel's verdict on the point (see `probe_card_vote`), `off`
// metres across the card from the texel's centre: its vote's weight -- how
// squarely its surface faces the card, squared, times how far it is trusted
// -- and how sure. The texel's depth is carried along its surface's plane to
// the point; 0 for a test read at the point itself. The slope carried is held
// to 3.5 to one (74 degrees): steeper, a texel is a rim turning away inside
// itself, and carried as a plane it reached a sphere's hits a texel away and
// voted for them with another card's colour. A surface steeper than ten to
// one is all but edge-on to the card, and its vote weighs nothing.
fn probe_card_trust(test: vec4<f32>, off: vec2<f32>, t: f32, depth: f32, field_stop: f32, cu: vec3<f32>, cv: vec3<f32>, cz: vec3<f32>, ld: vec3<f32>) -> vec2<f32> {{
    // The card's own frame: `cu`, `cv` its axes, `cz` the way it looks from.
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

// `proxy_cards::hemi_octahedral` undone: a unit direction, in a card's own
// frame, from two numbers.
fn probe_card_hemisphere(o: vec2<f32>) -> vec3<f32> {{
    return normalize(vec3<f32>(o.x, o.y, max(1.0 - abs(o.x) - abs(o.y), 0.0)));
}}

// Votes summed for `probe_card_colour`: colour weighted by the vote AND by
// 1 / (1 + luminance) -- Karis's weight, as the cards' mips are -- so a
// bright card among dark ones is one voice, not the loudest; and the surest
// card's `sure`.
struct ProbeCardSum {{
    colour: vec4<f32>,
    sure: f32,
    // The card that vouches most, for what one card's texel says alone: the
    // surface's albedo and facing (`probe_card_relit`).
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

// Axis `k` of a box's own frame.
fn probe_axis(k: u32) -> vec3<f32> {{
    return vec3<f32>(f32(k == 0u), f32(k == 1u), f32(k == 2u));
}}

// THE LIGHT OF THE LAMPS THE BAKE NEVER SAW on a surface their beams are
// known to light (`lights::LitSurface`), where a reflection meets it at `h`
// (world) after `t` from a surface of `roughness`: the torch's pool on a
// wall, as the polished floor should show it (user, 2026-10-05: "making the
// flashlight torch light on the wall be reflected on other surfaces"). The
// photographs hold the level's own light only. READ FROM THE SURFACE'S POOL
// MAP, made once a frame (`pool_map_light`): the map as seen from the glass,
// at `h`'s place in it, as blurred as the reflection's footprint there --
// so a hand's shadow in the pool softens with the floor's roughness as the
// pool's edge does. Nothing for a point on no such surface, or outside its
// map.
//
// A MAP, NOT THE LAMPS WORKED OUT AT EVERY HIT: worked out here -- the lamp
// loop, the cone and each lamp's nine-tap shadow at every reflected point on
// a lit plane -- the light cost the torch views 1.3-1.5 ms of the reflection
// pass (headset A/B, 2026-10-06), and the ground's pass a register whatever
// form it took. Nor deferred to the fix-up, whose list it filled to the cap
// (277,000 records, half the pass, offline 2026-10-06). `d` is unused: the
// map is the same light whichever way it is seen.
//
// READ WITH NO BRANCH but the frame's (no surface lit at all): every hit reads
// a texel -- surface 0's map where it is on no lit plane -- and keeps it only
// where it is on one, inside its map. So the read can be issued beside the
// photographs' (`probe_hit_colour`), and waits with them rather than after.
// Headset, synced frame, two passes: read here, the lookup costs the torch
// views 0.36-0.52 ms (deploy96); read last behind a branch it cost 0.22-0.60
// (deploy95) -- the same within noise, and this form is 53 instructions
// smaller. A frame with no surface lit draws without any of this: the passes'
// poolless twins (`without_pool_maps`).
fn probe_surface_relit(h: vec3<f32>, d: vec3<f32>, roughness: f32, t: f32) -> vec3<f32> {{
    if (lights.surfaces[1].w <= 0.0) {{
        return vec3<f32>(0.0);
    }}
    let p = to_player_space(h);
    let k = lit_surface_at(p);
    let at = max(k, 0) * LIT_SURFACE_VEC4S;
    let glass = lights.surfaces[at + 1];
    let axis = lights.surfaces[at + 2];
    let right = lights.surfaces[at + 3].xyz;
    let v = p - glass.xyz;
    let z = dot(v, axis.xyz);
    let xy = vec2<f32>(dot(v, right), dot(v, cross(right, axis.xyz))) / (max(z, 1e-4) * glass.w);
    let inside = k >= 0 && z > 0.0 && max(abs(xy.x), abs(xy.y)) < 1.0;
    // The maps lie three across the atlas in bands, each two cards square,
    // from the texel row the surfaces name (`pool_cards`); a texel of one
    // spans `2 z tan / block` where the hit is.
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

// THE POOL MAP's TEXEL `uv` OF SURFACE `k`: the lamps the bake never saw on
// the surface's plane, where the ray from the glass through that texel meets
// it -- each lamp's cone, falloff and own shadow map (lifted off the plane),
// on the surface's albedo; black where the ray meets the plane behind the
// glass or not at all. `texels`: the map's size, which sizes the cone's
// edge (`pixel_footprint`) to a texel there. Seen from the glass, the map's
// texels are as fine where the pool is near as where it is far, as the
// player -- holding the glass -- sees it. Made by `pool_cards` once a frame;
// read by `probe_surface_relit`.
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
        // The lamps the bake never saw -- but not a lit surface's own light,
        // which it sends away from itself (`SURFACE_LIGHT`).
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

// Which lit surface (`lights::LitSurface`) holds `p` (player frame): the
// first on whose plane it lies within `LIT_SURFACE_TOLERANCE`; -1 for none.
// One with no map (`w` 0: past the list's end) holds nothing. Where its light
// falls on the plane is its map's to say. Every surface tested, last first,
// with no early exit: a loop of fixed length over fixed places in the block,
// which the compiler can unroll into reads it need not index. 2026-10-06:
// finding the surface as a walk with a `break`, with nothing lit there, cost
// the torch views 0.25-0.32 ms of the reflection pass (headset, cut
// `def_cut_relight_found`).
fn lit_surface_at(p: vec3<f32>) -> i32 {{
    var k = -1;
    for (var j = MAX_LIT_SURFACES - 1; j >= 0; j = j - 1) {{
        let plane = lights.surfaces[j * LIT_SURFACE_VEC4S];
        let mapped = lights.surfaces[j * LIT_SURFACE_VEC4S + 1].w > 0.0;
        k = select(k, j, mapped && abs(dot(plane.xyz, p) - plane.w) < LIT_SURFACE_TOLERANCE);
    }}
    return k;
}}
const MAX_LIT_SURFACES: i32 = {MAX_LIT_SURFACES};
const LIT_SURFACE_VEC4S: i32 = {LIT_SURFACE_VEC4S};
const POOL_MAPS_ACROSS: i32 = {pool_maps_across};
// The pool maps' blur levels: the character cards' mip pass makes both.
const POOL_MAP_MAX_LOD: f32 = {character_card_max_lod:?};
// How far off a lit surface's plane a reflected point may lie and still be on
// it, metres: the rooms' boxes the trace meets lie on the walls they stand
// for, to the bake's centimetre.
const LIT_SURFACE_TOLERANCE: f32 = 0.05;

// THE LIGHT OF THE LAMPS THE BAKE NEVER SAW on a model's surface, where a
// reflection meets it: see `PROBE_CARD_RELIT`. `card` is the card vouching
// most for the hit and the point's u, v on it, in the colours' row `row`; `h`
// the hit, world; `q` the proxy's turn; `lod` the level its colour was read
// at, which the albedo is read at too. Nothing at all -- not a texture read --
// while no such lamp is lit.
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
    // Its facing from the tests' row, read where it is, and its albedo two
    // rows down, read as its colour was. See `proxy_cards`.
    let test = textureSampleLevel(proxy_cards, probe_samp, (origin + vec2<f32>(0.0, res) + clamp(uv * res, vec2<f32>(0.5), vec2<f32>(res - 0.5))) / dims, 0.0);
    let albedo = textureSampleLevel(proxy_cards, probe_samp, (origin + vec2<f32>(0.0, 2.0 * res) + clamp(uv * res, vec2<f32>(0.5 * s), vec2<f32>(res - 0.5 * s))) / dims, lod).rgb;
    // The card's frame, as `proxy_cards::card_frame` has it: it looks along
    // axis `a`, u along the next, v the one after.
    let a = u32(card.x) / 2u;
    let o = probe_card_hemisphere(test.xy);
    var n = o.x * probe_axis((a + 1u) % 3u) + o.y * probe_axis((a + 2u) % 3u)
        + o.z * probe_axis(a) * select(1.0, -1.0, (u32(card.x) & 1u) == 1u);
    if (any(q != vec4<f32>(0.0, 0.0, 0.0, 1.0))) {{
        n = probe_quat_rotate(q, n);
    }}
    // In the player's frame, as the lamps arrive.
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
    // As the lamps light a surface: its albedo times what arrives.
    return albedo * lit;
}}

// A MODEL'S OWN LOOK WHERE A REFLECTION MEETS IT, at world `h` along world
// `d`: all six of its cards (`proxy_cards`), each trusted where what it saw
// there is the hit and faces the ray (`probe_card_vote`), weighed by how
// squarely. The room photographs no longer hold the model, so they cannot
// colour it. `w` is how sure the cards are: 0 where the proxy has none, or
// none vouches for the hit.
//
// ALL SIX, BY WHAT THEY SAW. The three facing the ray, then the three facing
// the surface's normal from the model's distance field, each left a sconce's
// glowing inside to colour its dark outside: the field's normal beside a thin
// shell is barely a direction at all -- samples on both sides of it cancel;
// 74 degrees off in the tilted-plate test -- and read far off, one averaged
// depth needed a tolerance of three coarse texels, within which the card
// looking up into the shade vouched for the stem above it. Together: white
// specks round every sconce in the floor's reflection, coming and going as the
// head moved (headset, 2026-09-29 23:25 and the bench view
// hall_to_hallway_floor, 2026-09-30). Each card now says which way its own
// surface faces and what range of depths it saw.
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
    // How far short of the surface the field's walk stops: half a sample.
    let field_stop = select(0.0, camera.proxy_fields[max(i32(box.w) - 2, 0) * 3 + 1].w, box.w > 1.5);
    let res = f32(textureDimensions(proxy_cards).x) / 6.0;
    // READ AT THE FOOTPRINT: what one texel of this reflection covers where it
    // meets the model -- the pixel's cone, or the lobe on a rough surface --
    // in card texels. A sconce's glowing mouth, far off, is a fraction of a
    // pixel; read at full size it lit whole half-resolution texels, on or off
    // as the head moved (headset, 2026-09-29).
    let texel_m = 2.0 * max(max(half.x, half.y), half.z) / res;
    let footprint = max(t_hit * lobe, pixel_footprint * (1.0 + t_hit / max(probe_eye_distance, 0.05)));
    let lod = clamp(log2(max(footprint / texel_m, 1.0)), 0.0, log2(res));
    // Card 2a looks in through the +a face and shows the surfaces facing +a,
    // `t` from that face; 2a + 1 through the -a face, those facing -a. Axis
    // a's cards run u along a + 1 and v along a + 2. See
    // `space_soup_engine::reflection_cards`.
    let uvw = lo / (2.0 * half) + 0.5;
    let ax = vec3<f32>(1.0, 0.0, 0.0);
    let ay = vec3<f32>(0.0, 1.0, 0.0);
    let az = vec3<f32>(0.0, 0.0, 1.0);
    // Each card's frame: u, v, and the side it looks from. See
    // `proxy_cards::card_frame`.
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
    // HOW SURE, in `w`: a point no card vouches for -- the collar's underside,
    // hidden from the card below by the shade itself -- is left to the
    // model's own colour (`probe_model_colour`), not to a card's say at a
    // hundredth of a vote.
    var colour = sum.colour.rgb / max(sum.colour.w, 1e-9);
{card_relit_call}    return vec4<f32>(colour, sum.sure);
}}

// `mix(a, b, k)` weighed as a tone-mapped image would be (Karis's
// 1 / (1 + luminance)): a colour sixty times brighter than the other shows
// only as far as it is sure, not as far as a sixtieth of it saturates. A
// fixture's glowing inside, trusted at a tenth, was still six times the
// shade round it -- and white on the screen.
fn probe_mix_bright(a: vec4<f32>, b: vec4<f32>, k: f32) -> vec4<f32> {{
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    let wa = (1.0 - k) / (1.0 + dot(a.rgb, luma));
    let wb = k / (1.0 + dot(b.rgb, luma));
    return (a * wa + b * wb) / max(wa + wb, 1e-9);
}}

fn probe_hit_colour(h: vec3<f32>, room: f32, other: f32, roughness: f32, t: f32, d: vec3<f32>) -> vec4<f32> {{
    // The nearest two photographs of the hit's room and of `other`'s: see
    // `probe_nearest_two`.
    let near = probe_nearest_two(h, room, other);
    let s0 = near.s0;
    let s1 = near.s1;
    let d0 = near.d0;
    let d1 = near.d1;
    // A MODEL is shown by a photograph only from about the ray's direction.
    // See `probe_view_trust`.
    let model = other < -1.5;
    // BOTH PHOTOGRAPHS' TEXELS AT ONCE, depth and colour, before anything is
    // made of them: four reads that wait together. Read in turn -- the
    // second's depth after the first's had come back, the colours after the
    // depths had said which to keep -- each waited out the one before, and
    // these reads miss the cache, since every pixel's ray meets the
    // photographs somewhere else: this colouring was nearly half the probe
    // pass's time (stage counters, 2026-10-01). A photograph that turns out
    // not to vouch has its colour read for nothing; with no second
    // photograph, the first is read twice, which the cache serves.
    let has1 = s1 >= 0;
    let b0 = camera.probe_boxes[s0 * 3];
    let b1 = camera.probe_boxes[select(s0, s1, has1) * 3];
    let v0 = h - b0.xyz;
    let v1 = h - b1.xyz;
    // Each photograph read at the blur the hit's distance calls for, from its
    // own distance to the hit. See `probe_hit_lod`.
    let lod0 = probe_hit_lod(roughness, t, sqrt(d0));
    let lod1 = probe_hit_lod(roughness, t, sqrt(select(d0, d1, has1)));
    let depth0 = textureSampleLevel(probe_depth, probe_depth_samp, v0, i32(b0.w), 0.0);
    let depth1 = textureSampleLevel(probe_depth, probe_depth_samp, v1, i32(b1.w), 0.0);
    let col0 = textureSampleLevel(probe_cube, probe_samp, v0, i32(b0.w), lod0);
    let col1 = textureSampleLevel(probe_cube, probe_samp, v1, i32(b1.w), lod1);
{surface_relit_read}    let c0 = probe_clearance_of(depth0, v0);
    let tol0 = PROBE_SEEN_TOLERANCE + 0.01 * sqrt(d0);
    // How far each photograph vouches for the point, 0..1, and the most any does.
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
    // A MODEL, where no photograph saw it, is the model: see
    // `probe_model_colour`.
    if (model && seen < 1.0) {{
        col = mix(probe_model_colour(i32(-2.0 - other), -d, s0), col, seen);
    }}
    // A ROOM'S WALL NO PHOTOGRAPH SAW: the photograph shows whatever stood in
    // front of it there -- a sconce, whose silhouette, cut out along the
    // photograph's depth texels, read on the polished doorway jamb as a
    // stepped "shadow the wall shouldn't cast" (headset, 2026-09-29). Read
    // the same photograph soft enough that the thing in front is averaged into
    // the wall around it. Black holes in walls behind a pillar were worse, so
    // it is still the photograph. What this wants is photographs taken without
    // the props in them (tracker).
    if (!model && seen < 1.0) {{
        let wide = textureSampleLevel(probe_cube, probe_samp, v0, i32(b0.w), max(lod0, PROBE_UNSEEN_LOD));
        col = mix(wide, col, seen);
    }}
    // A MODEL ON CARDS IS ITS CARDS: a bake that pictures a model on its own
    // cards leaves it out of the photographs, which then show the wall behind
    // it. See `probe_card_colour`. The photographs' guess above stays for a
    // model without cards -- an older bake, or one past the level's shaped
    // models (`space_soup_engine::reflection_proxy::shaped_models`). Not in a
    // pass that defers, whose fix-up colours every hit on cards: see
    // `probe_hit_carded`.
    if (model && !PROBE_SECONDARY_DEFERRED) {{
        let card = probe_card_colour(i32(-2.0 - other), h, d, t, probe_lobe_tan(roughness));
        col = probe_mix_bright(col, vec4<f32>(card.rgb, 1.0), card.w);
    }}
{surface_relit_call}    return col;
}}

// WHETHER A HIT IS ON A MODEL'S CARDS (`probe_card_colour`). A pass that
// defers its secondary lookups leaves those hits' colour to `probe_fixup`,
// which reads the cards' tests filtered (`PROBE_CARD_TESTS_FILTERED`) and had
// traced nearly every such texel's ray again already (`PROBE_RETEST`): the
// six cards' votes were about a fifth of the probe pass's code, run for
// colours the fix-up then replaced -- in a pass that misses the GPU's
// instruction cache three times as often as the scene shader over its cliff
// (2026-10-01). Only where another outline already held the hit's edge did
// the pass's own card colour stand, read unfiltered.
fn probe_hit_carded(h: ProbeHit) -> bool {{
    let m = max(i32(-2.0 - h.other), 0);
    return h.other < -1.5 && camera.proxy_cards[m >> 2u][m & 3] > 0.5;
}}

// How soft a photograph is read where it did not see a room's wall: level 4 of
// a 256-texel face is about six degrees a texel, wider than a sconce seen from
// across a room. See `probe_hit_colour`.
const PROBE_UNSEEN_LOD: f32 = 4.0;

// THE COLOUR OF A TRACED HIT: where the ray left the rooms, what lies out
// there (`probe_escape_colour`); else the photographs of its room that saw it
// (`probe_hit_colour`). One or the other, never both. One exit: a helper with
// early returns, inlined into a loop, measured slower on the headset.
fn probe_traced_colour(h: ProbeHit, d: vec3<f32>, roughness: f32, sky_dir: vec3<f32>, lod: f32) -> vec4<f32> {{
    var col: vec4<f32>;
    if (h.escaped) {{
        col = probe_escape_colour(h.pos, d, sky_dir, roughness, lod);
        // Out through the doorway, then on to the ground, a building or sky.
        probe_reach = h.t + probe_reach;
    }} else {{
        col = probe_hit_colour(h.pos, h.room, h.other, roughness, h.t, d);
        probe_reach = h.t;
    }}
    return col;
}}

// A hit at a point already known -- the wall beside a doorway's rim, a solid
// proxy's outline -- in `room`, `t` along the ray, on no doorway.
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
        // The box is where this untraced reflection was projected. See
        // `probe_reach`.
        probe_reach = dist;
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
    // The room's nearest photograph, its chain walked: see `probe_choose_in_room`.
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
    return environment_radiance_of(dir, sky_irradiance(dir));
}}

// `environment_radiance` with the sky's sum along `dir` already taken.
fn environment_radiance_of(dir: vec3<f32>, sky: vec3<f32>) -> vec3<f32> {{
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
// THE PROBE PASS STORES ITS REFLECTIONS COMPRESSED, as Karis's
// `c / (1 + luminance)`, and the scene expands them after its bilinear read of
// the half-resolution texels (`probe_pass_expand`, beside the reader in
// `brush_pipeline::probe_pass`). A bulb's mouth reflected at a thousand times
// white beside the dark shade round it was averaged as light: a texel a
// hundredth bright still read as white, so the reflection's outline stood on
// the dark texels' centres and jumped a whole texel at a time as the head
// moved (offline crawl, 2026-09-30). Averaged compressed, each texel counts
// as far as it shows, as a display averages, and the outline moves smoothly
// between texels.
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
    // THE LIGHT AT THIS PIXEL FIRST, then the reflection: one number carried
    // through the trace instead of the normal, the baked light and the
    // occlusion it is made from -- eight registers, in a pass that is short
    // of them (see `probe_choose`).
    let occ = clamp(ao, 0.0, 1.0) * clamp(sky_vis, 0.0, 1.0);
    // The sky only where some reaches: `occ` is exactly 0 across most of an
    // interior, and there the nine-term sum was computed to be multiplied by it.
    var sky_here = vec3<f32>(0.0);
    if (occ > 0.0) {{
        sky_here = sky_irradiance(n) * occ;
    }}
    let ambient_here = dot(env + sky_here, vec3<f32>(0.2126, 0.7152, 0.0722));
    // THE CHARACTERS IN THE REFLECTION, here rather than in the scene shader,
    // at half the resolution each way: on the floor the player stands on,
    // themselves, mirrored (`probe_fixup::floor_mirror_blend_wgsl`); elsewhere
    // their capsules (`capsule_reflection`), lit by the light arriving here.
    // Decided before the trace, which records both for a deferred lookup.
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
    // A carried glass's glow past the normalisation: see `capsule_glow`.
    return vec4<f32>(probe_pass_compress(probe.rgb * probe_scale + capsule_glow) * a, a);
}}

// WHAT THE LAMP HALF OF `shade_material_env` NEEDS FROM ITS ENVIRONMENT HALF.
//
// Two halves so a caller can do something between them: the brush shader
// samples its stationary lamps' masks there. Eight visibilities computed
// before the environment work were carried, unused, through all of it, and
// that is where the scene shader held its register peak -- 21 registers, 50%
// occupancy, where 19 is the next step (`PIPESTATS`, 2026-09-28). The same
// arithmetic, in the same order, as the one function it was.
struct MaterialEnvPart {{
    diffuse: vec3<f32>,
    specular: vec3<f32>,
    // The bounce, shaped by its direction; and that direction and its
    // coherence for the gloss the lamp half adds -- `has_dir` false where the
    // baked light has no direction.
    bounce: vec3<f32>,
    bounce_dir: vec3<f32>,
    directionality: f32,
    has_dir: bool,
    fresnel: f32,
    view_dir: vec3<f32>,
    r: f32,
    // The luminance of the part of `specular` that is NOT an image of
    // anything: the lightmap's own light, which stays put on the surface.
    // See `reflected_image`.
    surface_spec: f32,
}}

// HOW MUCH OF THIS PIXEL IS A REFLECTED IMAGE, as luminance, written by
// `shade_material_lamps`: the probe's and the sky's reflection and every
// lamp's highlight -- what moves with the thing it shows, not with the
// surface. The brush shader stores its share of the pixel in alpha for
// SpaceWarp (`reflection_alpha`).
var<private> reflected_image: f32 = 0.0;
// WHICH REFLECTIONS SPACEWARP MOVES AS IMAGES, by the surface's roughness: all
// of one up to the first, none past the second. A step of the eye moves an
// image `step / distance` radians against its surface, and a reflection is
// blurred over about `roughness^2`; with steps of 1-4 cm a frame at 2-5 m the
// two meet around 0.1 (a 0.6 degree blur) and the blur is far the larger by
// 0.3. Marble is 0.05, the hallway rock 0.55 and up.
const REFLECTION_SHARP_ROUGHNESS: f32 = 0.1;
const REFLECTION_BLURRED_ROUGHNESS: f32 = 0.3;

// THE EYE IMAGE'S ALPHA FOR SPACEWARP: one minus how much the reflected image
// counts in this pixel's motion, from its share of `lit` (the pixel's linear
// colour) -- `space_warp::reflection_motion_weight`, which this is held to,
// and `space_warp::reflected_point`, which reads it back. The MSAA resolve and
// any blend over it average it as they average the colour.
const REFLECTION_CONTRAST_RATIO: f32 = {reflection_contrast};
fn reflection_alpha(lit: vec3<f32>) -> f32 {{
    // Only while SpaceWarp's motion pass reads it (`PostUpload::reflection_share`).
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
) -> MaterialEnvPart {{
    let view_dir = normalize(cam_pos() - world_pos);
    let r = clamp(roughness, 0.04, 1.0);
    // THE CHARACTERS' CONTACT DARKENING, on everything that arrives from all
    // around -- the baked bounce and the sky. See `capsule_ambient`.
    let contact = capsule_ambient(world_pos, n);
    let env = env_baked * contact;
    let occ = clamp(ao, 0.0, 1.0) * clamp(sky_vis, 0.0, 1.0) * contact;
    // Two accumulators from here on: what the surface's colour tints, and what
    // it does not. The first starts as the sky along the normal -- only where
    // the baked sky visibility is not exactly 0, most of an interior being
    // multiplied by that 0. Its reflection is taken below.
    var diffuse = vec3<f32>(0.0);
    if (occ > 0.0) {{
        diffuse = sky_irradiance(n) * occ;
    }}
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
    // x^5 as three multiplies, as the CPU mirror's `powi(5)` has it; `pow` is
    // exp2(5 log2 x), two transcendental instructions every pixel.
    let grazing = 1.0 - cos_v;
    let grazing2 = grazing * grazing;
    let fresnel = f0 + (f_max - f0) * (grazing2 * grazing2 * grazing);
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
    // THE CHARACTERS IN IT: on the floor the player stands on they are in the
    // probe pass's answer already, mirrored (see `on_floor_mirror`); elsewhere
    // their capsules, at this pixel's own resolution, lit by the light
    // arriving here. See `capsule_reflection`.
    // THE CHARACTERS IN IT, where this shader traces its own reflection: their
    // capsules, at this pixel's resolution, lit by the light arriving here.
    // A shader that reads the probe pass has them already -- the pass puts
    // them in, or mirrors them on the floor the player stands on (see
    // `probe_env_for_pass`) -- so here they are not even compiled. Any test
    // for the floor made here, however cheap to read, cost the scene shader
    // 2 ms a frame with the same registers (headset, 2026-09-30).
    if (!PROBE_ENV_FROM_PASS) {{
        probe = capsule_reflection(world_pos, refl, roughness, env, probe);
    }}
    // THE SKY ALONG THE MIRROR DIRECTION, for its reflection -- its nine-term
    // sum written out here and at the diffuse above. Builds 118-119 walked one
    // copy over a two-bit mask instead, the two copies being a reader's
    // largest repeated code (`shader_inlining`), and that form was slower in
    // every view priced against this one in one session -- outdoors 0.10-0.21
    // ms, halls 0.07, the torch doorway 0.1 (build 120, 2026-10-06; it is
    // `READER_EDITS`' `sky_walked`).
    var sky_reflection = vec3<f32>(0.0);
    if (occ > 0.0) {{
        sky_reflection = environment_radiance(refl) * occ;
    }}
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
    let sharp = mix(baseline, probe.rgb * spec_occ * probe_scale{capsule_glow_term}, clamp(probe.a, 0.0, 1.0));
    // Marble here is 0.048 and takes the sharp answer; brick is near 1 and
    // falls back to the baseline, because at that roughness the probe's extra
    // directional detail is not information, it is the artefact.
    let lobe_is_hemispherical = smoothstep(0.25, PROBE_LOBE_HEMISPHERICAL, r);
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
    // Of that, what is the lightmap's own light rather than an image: the
    // baseline's indirect part, where neither the probe nor the lobe replaced
    // it. See `MaterialEnvPart::surface_spec`.
    let image_weight = (1.0 - lobe_is_hemispherical) * clamp(probe.a, 0.0, 1.0);
    let surface_spec = dot(indirect_specular, vec3<f32>(0.2126, 0.7152, 0.0722)) * (1.0 - image_weight) * fresnel;

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
    var bounce_dir = vec3<f32>(0.0);
    let has_dir = bd_len > MIN_BOUNCE_DIR_LENGTH;
    if (has_dir) {{
        // Baked in the world; the normals and the view arrive in the player's
        // frame. Read as it was stored, the bounce turned with every snap and
        // stick turn: the hallway's rock 1% darker at 180 degrees, its gloss
        // pointing the wrong way (headset bench, `NAME-yQ`, 2026-10-01).
        bounce_dir = to_player_direction(bd / bd_len);
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

// The lamp half of `shade_material_env`: the lamps, and the bounce's gloss,
// which is sized by the nearest of them. Reads the stationary masks and the
// sun mask, which must be set by now. See `MaterialEnvPart`.
fn shade_material_lamps(p: MaterialEnvPart, world_pos: vec3<f32>, n: vec3<f32>, env: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {{
    let view_dir = p.view_dir;
    let r = p.r;
    // Blinn-Phong has an exponent where a PBR model has a roughness, so the two
    // are bridged by the usual mapping: alpha = r^2, exponent = 2/alpha^2 - 2.
    // Exact enough for a preview and monotonic, which is what matters -- a
    // rougher material must never come out shinier.
    // Distance to the nearest punctual light, for the sphere-light widening.
    // Computed before the loop because the roughness it feeds is per surface,
    // not per light: one lobe, sized by whatever is actually lighting this spot.
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
    //
    // By SQUARED distance, and one square root after the loop: the nearest
    // lamp is the same either way, and a root per lamp per pixel was paid to
    // compare numbers whose order the root does not change.
    //
    // AND WHICH LAMPS CAN REACH THIS PIXEL AT ALL, in the same pass: bit `i`
    // of `reaching` is lamp `i`, tested exactly as the lighting loop used to
    // test it on a second walk over every lamp -- its baked visibility, its
    // range, its cone, the sun's mask. The lighting loop below then visits
    // only those, in the same order. See the comment there.
    // The pixel's longer step on this surface, for the cone test below: how
    // far a spot's averaged edge can reach past its cone. Once a pixel, not
    // per lamp.
    let pixel_long = spot_long_step(dot(view_dir, n));
    var light_dist_sq = 1e18;
    var reaching = 0u;
    let culling = light_culling();
    // Past the lit surfaces' lights, which are shaded apart below.
    for (var i: u32 = surface_light_count(); i < live_light_count(); i = i + 1u) {{
        let kind = lights.lights[i].params.z;
        let to_lamp = lights.lights[i].position.xyz - world_pos;
        let dist_sq = dot(to_lamp, to_lamp);
        if (kind <= 1.5) {{
            // The nearest BULB. A light standing for a lit surface makes no
            // highlight (`SURFACE_LIGHT`), so it sizes none: a flashlight's
            // bounce 3 cm off a wall opened the torch's own highlight round
            // it into a bright blob, nowhere near the torch's mirror angle
            // (headset, 2026-10-02).
            light_dist_sq = select(light_dist_sq, min(light_dist_sq, dist_sq), lights.lights[i].params.w > -1.5);
        }} else if (SKY_SUN_NEVER_REACHES) {{
            // Whatever the culling lever says: the sun's lookups are not in
            // this shader. See `SKY_SUN_NEVER_REACHES`.
            continue;
        }}
        if (culling) {{
            // Past its range, where the window is exactly zero: here or below,
            // by `CULL_RANGE_FIRST`.
            if (CULL_RANGE_FIRST && kind < 1.5 && dist_sq >= lights.lights[i].params.x * lights.lights[i].params.x) {{
                continue;
            }}
            // Hidden from here by its baked mask: exactly zero light.
            if (stationary_visibility_of(lights.lights[i].position.w) <= 0.0) {{
                continue;
            }}
            if (kind < 1.5) {{
                let reach = lights.lights[i].params.x;
                if (!CULL_RANGE_FIRST && dist_sq >= reach * reach) {{
                    continue;
                }}
                // OUTSIDE A SPOT'S CONE BY MORE THAN ITS SOFT EDGE CAN REACH.
                // `spot_cone` widens the authored band for antialiasing to
                // `SPOT_EDGE_MIN_PIXELS` of this pixel's footprint as seen from
                // the lamp, never past SPOT_EDGE_MAX_WIDEN times the band --
                // so wherever the angle's cosine is below `cos_outer` less
                // half that widening (less a hair for rounding) the cone is
                // exactly 0, and so is all the maths it multiplies. The
                // widening is bounded HERE, at this distance, with the sine at
                // its largest: never less than `spot_cone` widens. Bounded by
                // the cap alone, a band as wide as the flashlight bounce's
                // half space was never culled at all, and the wall behind its
                // patch was shaded in full for nothing (2026-10-02). `cos =
                // along / dist`, compared squared, with the signs, so no root
                // is taken but the one reciprocal the footprint needs.
                if (kind > 0.5) {{
                    let cos_outer = lights.lights[i].params.y;
                    let authored = max(lights.lights[i].direction.w - cos_outer, 0.0001);
                    // And past that, the pixel's average along its long step
                    // reaches half of `spot_cone_across` further out, which is
                    // at most half the pixel's LONGER step over the distance
                    // (the cosine's gradient is sin / dist); 0 where no steps
                    // were set.
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
                // THE SKY'S SUN WHERE ITS BAKED MASK HIDES IT COMPLETELY --
                // indoors, most of the level. `sun_visibility` returns exactly
                // that 0 without sampling anything. A receiver with no mask
                // carries -1 and is shaded as before.
                continue;
            }}
        }}
        reaching = reaching | (1u << i);
    }}
    let light_dist = sqrt(light_dist_sq);
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

    var specular = p.specular;
    let fresnel = p.fresnel;
    if (p.has_dir) {{
        let bounce_dir = p.bounce_dir;
        let directionality = p.directionality;
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
    let bounce = p.bounce;
    // ENERGY CONSERVATION. The albedo lands on the diffuse half only.
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
    let kd = albedo * (1.0 - fresnel);
    // BAKED: the lightmap's bounce, before any runtime light is added.
    // Weighted as the picture weights diffuse light.
    dbg_baked = bounce * kd;
    // ONE SUM, NOT TWO. The lamps' diffuse and specular were summed apart and
    // weighted after the loop, which kept two colours, the albedo, the Fresnel
    // and the roughness live across every lamp: the loop is the scene readers'
    // register peak, and those were four registers of it (PIPESTATS,
    // 2026-10-05). Each lamp is weighted as it is added, which is the same sum.
    // The specular's luminance -- all `reflected_image` needs of it -- is kept
    // as one number, the lightmap's own share taken off at the start.
    //
    // AND WRITTEN IN `hf` THROUGH THE LOOP (B1, 2026-10-06): the sums, and
    // the albedo, tightness and strength each lamp is weighted by, are the
    // colours and factors `hf` is for -- live across every lamp's shadow
    // lookup, the readers' register peak. The lamp maths that needs `f32` --
    // positions, the normal, the view, the highlight's `pow` -- stays `f32`.
    // Every sum is clamped below f16's limit: a reflection of the sun can
    // carry more, and comes out white either way. `hf` is `f32` as shipped
    // (`shader_precision::HALF_PRECISION`), and these are the same sums.
    var colour = hf3(min((p.diffuse + bounce) * kd + specular, vec3<f32>(HF_MAX)));
    var spec_luma = hf(clamp(dot(specular, vec3<f32>(0.2126, 0.7152, 0.0722)) - p.surface_spec, -HF_MAX, HF_MAX));
    let kd_h = hf3(kd);
    let shininess_h = hf(shininess);
    let spec_strength_h = hf(spec_strength);
    // For SpaceWarp: everything specular but the lightmap's part is an image
    // -- counted only as far as it is sharp enough to show something. A rough
    // surface's reflection is a blur with nothing in it to judder, while its
    // own texture has plenty: counted whole, the hallway rock's dark crevices
    // came out as mostly reflection and would have swum with it (the motion
    // pass's inputs read off the headset, 2026-09-29). Between these two
    // roughnesses the blur outgrows what a step of the eye moves an image by.
    let sharp = hf(1.0 - smoothstep(REFLECTION_SHARP_ROUGHNESS, REFLECTION_BLURRED_ROUGHNESS, r));
    // THE LIT SURFACES' LIGHTS, apart from the lamps (`surface_lights`), and
    // weighted as each lamp's diffuse is below.
    if (SURFACE_LIGHTS) {{
        let surface_lit = hf3(min(surface_lights(world_pos, n), vec3<f32>(HF_MAX))) * kd_h;
        colour = min(colour + surface_lit, hf3(hf(HF_MAX)));
        dbg_direct = dbg_direct + vec3<f32>(surface_lit);
    }}
    // A LAMP THAT CANNOT REACH THIS PIXEL IS SKIPPED BEFORE ANY OF ITS MATHS:
    // past its range, outside its cone, or a stationary lamp its baked mask
    // says is hidden from here -- behind a wall, in another room. Each makes
    // the whole contribution exactly zero, so skipping changes no pixel. In a
    // level of several rooms it is the common case: most lamps are behind a
    // wall from most pixels. The tests were made in the nearest-lamp pass
    // above (`reaching`), which walks every lamp anyway; this loop visits only
    // the lamps that passed, lowest first -- the same lamps, summed in the
    // same order, as when it walked them all and tested each here.
    var todo = reaching;
    loop {{
        if (todo == 0u) {{
            break;
        }}
        let i = countTrailingZeros(todo);
        todo = todo & (todo - 1u);
        let l = lights.lights[i];
        let c = light_contribution_split(l, world_pos, n, view_dir, f32(shininess_h), f32(spec_strength_h));
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
        // WEIGHTED BEFORE THE SHADOW, AND AT HALF PRECISION (B1, 2026-10-06).
        // The lamp's two halves were six numbers held through its shadow
        // lookup, to be weighted after it -- and the lookups are the readers'
        // register peak: every cut of a shadow moved it, 22 -> 19 without the
        // spots' or the sun's (PIPESTATS, `ps96`). Weighted first, they are the
        // one colour and one luminance the shadow scales, as `hf`: half the
        // registers where the device has `f16`, and at `f32` the same sum as
        // before. A lamp's light here is a few thousand at most (intensities
        // up to 10 over `LAMP_RADIUS` squared), clamped below f16's limit all
        // the same. `hf` is `f32` as shipped: at `f16` the readers lost
        // registers and no time (`shader_precision::HALF_PRECISION`).
        let unshadowed = hf3(min(c.diffuse, vec3<f32>(HF_MAX))) * kd_h + hf3(min(c.specular, vec3<f32>(HF_MAX)));
        let unshadowed_luma = hf(min(dot(c.specular, vec3<f32>(0.2126, 0.7152, 0.0722)), HF_MAX));
        // ONE shadow factor for both halves: a surface in shadow receives no
        // light at all, and a highlight that survives its own shadow is the
        // classic tell of a renderer that shadows only the diffuse term. The
        // baked mask's part read here, past the lamp's own maths, so it is not
        // carried through them.
        var shadow = stationary_visibility_of(l.position.w);
        if (!SKY_SUN_NEVER_REACHES && l.params.z > 1.5) {{
            shadow = sun_level_visibility(world_pos);
        }}
        // Its spot slot, which draws everything; else, for a lamp lighting
        // the player most, its tile of the characters alone
        // (`character_shadow`); else, for the sky's sun, the moving things
        // round the player (`sun_visibility`).
        //
        // ONE TRANSFORM AND ONE KERNEL A MAP KIND, at one place. No lamp
        // reads two maps -- spot slots go to spots alone -- and every map is a
        // row of `camera.moving_view_proj`, so the branches only choose the
        // row: one copy of `shadow_coords` carries the point into it, where
        // each read at its own call was a copy of its own. A characters' tile
        // and the sun's moving things are the same kernel on the same atlas,
        // read by one copy too (2026-10-06).
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
        // NO CAPSULE SHADOWS HERE. They were a capsule loop inside this lamp
        // loop -- 1,100 of the scene shader's 3,600 instructions -- and cost
        // 1 ms an eye even where no character was near, the code's size alone
        // (headset trace, 2026-09-29). The player's shadows from the lamps
        // lighting them most are the characters' tiles above; the capsules
        // keep only what they do once a pixel: contact darkening, reflections.
        let lit = unshadowed * hf(shadow);
        colour = min(colour + lit, hf3(hf(HF_MAX)));
        spec_luma = min(spec_luma + unshadowed_luma * hf(shadow), hf(HF_MAX));
        // DIRECT: runtime lights, after their shadow test.
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
    let p = shade_material_env_part(
        world_pos, n, roughness, ao, sky_vis, env, env_dir, albedo, probe_select_pos, geom_n,
    );
    return shade_material_lamps(p, world_pos, n, env, albedo);
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
    // The characters' contact darkening on the sky's light: see `capsule_ambient`.
    //
    // THE SUM AND EACH LAMP'S LIGHT IN `hf` through the shadow lookups, as in
    // `shade_material_lamps` (B1, 2026-10-06): colours, held live across
    // every lamp's shadows -- the meshes' register peak. Clamped below f16's
    // limit; `hf` is `f32` as shipped, and these are the same sums.
    var lit = hf3(min(sky_irradiance(n) * clamp(sky_vis, 0.0, 1.0) * capsule_ambient(world_pos, n), vec3<f32>(HF_MAX)));
    for (var i: u32 = 0u; i < live_light_count(); i = i + 1u) {{
        let l = lights.lights[i];
        let arriving = light_contribution(l, world_pos, n, view_dir);
        // As in `shade_material_env`: no light arriving, no shadow test.
        if (max(max(arriving.r, arriving.g), arriving.b) <= 0.0) {{
            continue;
        }}
        var c = hf3(min(arriving, vec3<f32>(HF_MAX)));
        // Only the sun casts the orthographic map, and only the flashlight the
        // perspective one. Every other light is unshadowed, which is the whole
        // reason a scene may have eight of them.
        if (l.params.z > 1.5) {{
            c = c * hf(sun_visibility(l, world_pos));
        }}
        c = c * hf(stationary_visibility(l));
        // params.w is this light's own shadow layer, or -1 when it did not get
        // one. Asking the LIGHT beats the old "is this the flashlight index"
        // test, which by construction could only ever be true for one lamp.
        let layer = i32(l.params.w);
        if (SPOT_SHADOWS && layer >= 0 && f32(layer) < camera.shadow_params.y) {{
            c = c * hf(pcf_layer(spot_shadow_tex, layer, world_pos, spot_view_proj(layer)));
        }} else if (l.params.z < 1.5) {{
            // The characters' shadows: see the brushes' loop.
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

    /// A light whose shadow would begin past its range casts none -- a
    /// flashlight's bounce, a lit patch of wall -- and any other light can.
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
        // On the GPU it is marked to make no highlight, and holds no layer.
        let gpu = pack_lights(&[spot, surface], 2, &[], false, true);
        assert_eq!(gpu.lights[0].params[3], -1.0);
        assert_eq!(gpu.lights[1].params[3], SURFACE_LIGHT);
        assert!(SURFACE_LIGHT < -1.5, "below the shader's test");
        // Its patch's radius rides below the mark; a lamp with a shadow is a
        // bulb whatever it says.
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

    /// A LIT SURFACE'S LIGHT -- no shadow, a spot's edge a half space or
    /// wider -- ranks to the front of the list, each kind keeping its order,
    /// and the upload counts the run of them there for the scene readers to
    /// shade apart (`surface_lights`). A lamp, a narrow spot without a
    /// shadow, one behind a lamp and one in a baked tail are lamps.
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
        // Over budget the ranking chooses first: scores 0.5, 0.2, 1.0, 0.2.
        assert_eq!(rank_for_budget_indices(&ls, 3), vec![1, 2, 0]);
        assert_eq!(rank_for_budget_indices(&ls, 2), vec![2, 0]);
        let ranked = rank_for_budget(&ls, MAX_LIGHTS);
        assert_eq!(pack_lights(&ranked, 4, &[], false, true).surface_lights[0], 2);
        assert_eq!(pack_lights(&[lamp, bounce], 2, &[], false, true).surface_lights[0], 0, "only a leading run");
        assert_eq!(pack_lights(&[bounce, lamp], 0, &[], false, true).surface_lights[0], 0, "never a baked tail");
        // The spotless twins shade none apart; their frames have none.
        let src = wgsl_lights_block(0, 1);
        assert!(src.contains(SURFACE_LIGHTS_ON));
        assert!(without_spot_shadows(src).contains(SURFACE_LIGHTS_OFF));
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
        // Weighted as each lamp is added (one sum, for the registers) -- before
        // its shadow, at half precision -- and the lightmap's diffuse the same
        // way.
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
            code.contains("sky_reflection = environment_radiance(refl) * occ;"),
            "the specular environment is back to reading the raw sky harmonics",
        );
        // And the DIFFUSE ambient must not: it is evaluated along the surface
        // normal, and a ground term there would light undersides that see no
        // ground.
        assert!(
            code.contains("diffuse = sky_irradiance(n) * occ;"),
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
        // The nine terms -- coefficient, basis, cosine-lobe weight already
        // divided by pi -- exactly as the mirror spells them, in its order.
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
        // The traced hit's second photograph: its own box, its own direction
        // to the hit, its own blur (`probe_hit_colour` reads both at once).
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
        // The material path's lamps are in its lamp half. See `MaterialEnvPart`.
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
        // ...and not to a carried glass's glow, which is not the probe's light.
        assert!(
            code.contains(
                "let sharp = mix(baseline, probe.rgb * spec_occ * probe_scale + capsule_glow * spec_occ, clamp(probe.a, 0.0, 1.0));"
            ),
            "the scale is computed but not applied to the probe",
        );
        // A shader reading the probe pass has the glow in the pass's answer,
        // normalised already, and its line is as it was.
        let from_pass = super::wgsl_lights_block_with(
            0,
            1,
            super::LightsBlockOptions { probe_from_pass: true, ..Default::default() },
        );
        assert!(from_pass.contains("let sharp = mix(baseline, probe.rgb * spec_occ * probe_scale, clamp(probe.a, 0.0, 1.0));"));
        assert!(from_pass.contains("probe_pass_compress(probe.rgb * probe_scale + capsule_glow) * a"));
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
            // Both spellings: `var x: array<f32, 9>` and the inferred
            // `var x = array<f32, 9>(...)`, which is how the two in
            // `sky_irradiance` slipped past this guard until 2026-09-28.
            let local_array_decl = t.starts_with("var<function>")
                || (t.starts_with("var ") && (t.contains(": array<") || t.contains("= array<")));
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
        // Compared by squared distance: one root after the loop, none in it.
        assert!(
            !loop_body.contains("distance(") && !loop_body.contains("length(") && !loop_body.contains("sqrt("),
            "the nearest-light loop takes a square root per lamp again",
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

    /// `terminator_aa` on the CPU: the Lambert term and the share of the pixel
    /// facing the lamp, for `dot(n, l)` spread over `nl +- w`.
    fn terminator_aa(nl: f32, w: f32) -> (f32, f32) {
        let facing = ((nl + w) / (2.0 * w).max(1e-6)).clamp(0.0, 1.0);
        let lambert = if nl.abs() < w { 0.5 * (nl + w) * facing } else { nl.max(0.0) };
        (lambert, facing)
    }

    /// `terminator_width_of` on the CPU.
    fn terminator_width_of(normal_length: f32) -> f32 {
        let sigma2 = (1.0 - normal_length - 0.01).max(0.0) / normal_length.max(1e-3);
        (1.5 * sigma2).sqrt().min(0.5)
    }

    /// THE TERMINATOR IS THE MEAN OF THE CLAMP OVER THE PIXEL, not the clamp
    /// at its centre: integrated numerically, the pixel's light and the share
    /// of it facing the lamp are what the closed forms give -- and with no
    /// spread they are the hard step the shader always had.
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

    /// A normal the map's texels agree on -- unit length, give or take its
    /// 8-bit rounding -- keeps the hard terminator; the shorter the filtered
    /// normal, the wider it is spread, up to the cap.
    #[test]
    fn only_a_normal_its_texels_disagree_on_spreads_the_terminator() {
        assert_eq!(terminator_width_of(1.0), 0.0);
        assert_eq!(terminator_width_of(0.992), 0.0, "8-bit rounding is not spread");
        let (a, b) = (terminator_width_of(0.97), terminator_width_of(0.9));
        assert!(0.0 < a && a < b && b < 0.5, "{a} {b}");
        assert_eq!(terminator_width_of(0.2), 0.5);
    }

    /// And the WGSL computes what its twins above do.
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
            "out.specular = radiance * spec * atten * aa.y;",
        ] {
            assert!(code.contains(line), "the lights block no longer has `{line}`");
        }
    }
}

/// Baked lamps ride behind the live ones, and lightmapped surfaces stop there.
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
        // More baked lamps than fit beside the live ones, whatever the budget.
        let baked: Vec<Light> = (0..MAX_LIGHTS + 2).map(|i| point(i as f32, 0.5)).collect();
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
        // Seven walks bounded by the live count: two of them the lamps the
        // bake never saw lighting a model's cards (`probe_card_relit`), one
        // those lamps on a surface their beams light (`probe_surface_relit`),
        // one the beam of a glass a reflection shows (`capsule_glass_beam`).
        // The eighth, the lighting in `shade_material_env`, walks the
        // `reaching` bits that its nearest-lamp pass -- one of the seven --
        // set, so it honours the split too.
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
        trace_seen_from(rays, 0.0, 1.0)
    }

    /// `trace`, as a pixel `footprint` metres across on the reflecting
    /// surface, `eye` metres from the eye, would trace it -- what the probe
    /// pass's fragment stage sets before tracing (`pixel_footprint`,
    /// `probe_eye_distance`); a compute shader has no derivatives, so the
    /// other tests trace as a point.
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
        // As `Uniforms::update` fills it: rooms renumbered, with their tables.
        // (Volumes 0, 0, 1, 2 in slot order renumber to themselves, so the
        // rooms the rays name below and the hits report keep their numbers.)
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
        // The models' distance fields, and the sampler the trace reads them
        // with: none here, every proxy is a box. See `proxy_field`.
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

    /// A MIRROR'S REFLECTION OF A DOORWAY'S RIM, AS THE PROBE PASS TRACES IT:
    /// from the floor on the left of the hall toward the front door's left
    /// jamb (the headset's 22:04:15 staircase), with the half-resolution
    /// pixel's footprint there. Across the jamb the share that passes through
    /// must rise smoothly -- never jump from nothing to all of it between two
    /// neighbouring rays.
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
                // A rim: the share of the footprint that went through.
                (true, _, _) => h.rim,
                // No rim: all or nothing, by where the one ray went.
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

/// THE GROUND TRACE, RUN ON THE GPU: the real WGSL `ground_trace` from a
/// compute shader over test_room-shaped hills, against its CPU twin
/// `ground_map::trace`. The twin is what the ground map's tests hold to brute
/// force; this holds the shader to the twin.
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
        // Bindings as `wgsl_lights_block(0, 1)` numbers them: the probe
        // sampler at 1 + 5, the ground map at 1 + 9.
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

    /// Hit for hit and centimetre for centimetre, except where the ray only
    /// grazes the ground: there the GPU's filtering, a few bits coarser than
    /// the twin's arithmetic, may round a touch either way.
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

/// A MODEL'S DISTANCE FIELD, WALKED ON THE GPU: the real WGSL
/// `probe_proxy_field` over a sphere's field in its box, from a compute shader.
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

    /// A sphere of `RADIUS` in a cube of half-size `HALF`: unsigned distance,
    /// reaching four samples, as `model_field` builds a model's.
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

    /// Walk each box-local `(origin, direction)` through the field.
    fn walk(rays: &[(Vec3, Vec3)]) -> Option<Vec<f32>> {
        walk_full(sphere(), rays, 0.0).map(|v| v.into_iter().map(|h| h[0]).collect())
    }

    /// Walk each box-local `(origin, direction)` through `field` with a
    /// mirror's lobe and a pixel `pixel_footprint` wide on the surface it
    /// left, the eye a metre from that: everything the walk returns.
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
    // Where the ray is inside the box.
    let inv = 1.0 / d;
    let ta = (-half - o) * inv;
    let tb = (half - o) * inv;
    let t_in = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z));
    let t_out = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
    // A mirror's lobe (0).
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

    /// A bowl of `RADIUS`, open upward: the lower half of a sphere's shell,
    /// its rim the circle where it ends -- a lampshade's mouth turned over.
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

    /// Straight at the sphere, the walk stops on its surface; through its
    /// edge, where the chord says; five centimetres above it, through air
    /// the box's bounds would have called solid, it finds nothing.
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

    /// A HIT PAST THE MODEL'S OWN OUTLINE is marked for sampling again: a ray
    /// that passes a centimetre over the bowl's rim, inside a footprint of it,
    /// and meets its inside beyond, straddles the rim and its inside. Not one
    /// straight down into it, nor one meeting its outside at a slant -- which
    /// comes ever nearer, never away and back -- nor one skimming the rim and
    /// out again, a near miss the outline fade already softens.
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
        // Past the rim, down the chord to the bottom of the bowl's far side.
        assert!(h[0][0] > 1.1 && h[0][0] < 1.35, "over the rim, into the bowl: hit at {:?}", h[0]);
        assert_eq!(h[0][2], -1.0, "straddles the rim: {:?}", h[0]);
        assert!((h[1][0] - (0.9 + RADIUS)).abs() < 0.03 && h[1][2] >= 0.0, "straight down: {:?}", h[1]);
        assert!(h[2][0] < 1e30 && h[2][2] >= 0.0, "a slanting hit on its outside: {:?}", h[2]);
        assert!(h[3][0] > 1e30 && h[3][2] >= 0.0 && h[3][2] < 1.0, "skimming the rim, a near miss: {:?}", h[3]);
    }
}

/// A MODEL'S CARDS, READ ON THE GPU: the real WGSL `probe_card_colour` over a
/// sphere -- a real distance field, and six cards baked from it the way
/// `tools/bake` bakes them, each card its own colour -- from a compute shader.
/// See `proxy_cards` and `space_soup_engine::reflection_cards`.
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
        /// The pixel footprint and hit distance the next `colours_of` reads at.
        static FOOTPRINT: std::cell::Cell<f32> = const { std::cell::Cell::new(0.0) };
        static HIT_T: std::cell::Cell<f32> = const { std::cell::Cell::new(0.0) };
        /// Whether the next `colours_of` filters the cards' trust, as
        /// `probe_fixup` does -- the default, since every texel meeting a
        /// model on cards is finished there. See `PROBE_CARD_TESTS_FILTERED`.
        static FILTERED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
        /// The lamps the next `colours_of` lights the cards with, as
        /// `probe_fixup` does (`PROBE_CARD_RELIT`); `None` reads them unlit,
        /// as every other shader does.
        static RELIT: std::cell::RefCell<Option<Vec<Light>>> = const { std::cell::RefCell::new(None) };
    }
    const RADIUS: f32 = 0.2;
    const SAMPLES: u32 = 24;

    /// The sphere's unsigned distance over its box, reaching four samples.
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

    /// The sphere's six cards, card k coloured (k + 1, 0, 0): each texel's
    /// depth the sphere's surface along the card's axis, or nothing, and that
    /// surface's normal.
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

    /// `probe_card_colour` for each world `(hit, direction)` against the sphere
    /// turned by `rotation`; `with_cards` false leaves it without.
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
        // The lamps, and a spot atlas no lamp holds a layer of.
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
        // w: the pixel's footprint on the reflecting surface, and the hit's
        // distance along the ray -- 0 and 0 read the cards at full size.
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
        // No field texture: the cards alone colour a hit (the field's slot, in
        // the uniform, gives only its stop distance).
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

    /// Two cards' colours as `probe_card_colour` averages two equal votes:
    /// weighted by 1 / (1 + luminance), the fixtures' colour all red.
    fn karis(a: f32, b: f32) -> f32 {
        let (wa, wb) = (1.0 / (1.0 + 0.2126 * a), 1.0 / (1.0 + 0.2126 * b));
        (a * wa + b * wb) / (wa + wb)
    }

    /// Where a ray meets the sphere facing `n`: the field walk stops a hair
    /// short of the surface, on the ray's side.
    fn on_sphere(n: Vec3) -> Vec3 {
        CENTRE + n.normalize() * (RADIUS + 0.003)
    }

    /// The +z card seeing a step: the half with x < 0 a surface 0.3 of the
    /// box deep, the rest one 0.6 deep -- a cage wire in front of the lamp's
    /// lit inside, as the card looking up into it sees them. Every texel faces
    /// the card; the other cards see nothing.
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

    /// THE TRUST FILTERED, AS THE FIX-UP READS IT: a hit on the deeper side
    /// of `step_cards`' step, sliding toward it a tenth of a texel at a time.
    /// Tested once on blended texels, the verdict dropped from all to nothing
    /// the moment the blend took in the nearer surface: the hit lies behind
    /// that by centimetres. So a lamp's lit inside, seen past its cage wires,
    /// flickered in its reflection as the head moved a millimetre (offline,
    /// 2026-10-01). Each texel tested alone and the verdicts blended, the
    /// trust falls across the texel as the hit crosses it -- and a smooth
    /// surface, its texels' depths carried to the hit along their slopes, is
    /// trusted whole (every other test here).
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

    /// The card the SURFACE faces colours it, whichever way the ray came: a
    /// ray climbing to the sphere's side takes nothing from the card looking
    /// up at its underside -- the wall sconce's glowing inside, which turned
    /// every sconce's reflection white (headset, 2026-09-29).
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

    /// A SHELL'S TWO FACES, millimetres apart and at one depth from every card:
    /// a thin plate through the box's centre, tilted 45 degrees (normal +x+y),
    /// its upper face met by a ray CLIMBING toward it. The card looking up sees
    /// the plate's underside right there -- a lampshade's lit inside -- and
    /// only the normal says the hit is not on it: the upper face is the +x and
    /// +y cards' (1 and 3), never the one looking up (4).
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

    /// The unsigned field of a thin plate through the box's centre, facing `n`.
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

    /// The cards of the plate through the centre facing +x+y, card k coloured
    /// (k + 1, 0, 0). At (u, v): along x (k 0, 1) the plate is at x = -y;
    /// along y (k 2, 3) at y = -x; along z it is edge-on, and missed. The +x
    /// and +y cards see its upper face, the -x and -y ones its underside.
    fn plate_cards() -> ProxyCards {
        let n = Vec3::new(1.0, 1.0, 0.0).normalize();
        let mut texels = Vec::new();
        let mut normals = Vec::new();
        for face in 0..6u32 {
            for y in 0..RES {
                for x in 0..RES {
                    let v = ((y as f32 + 0.5) / RES as f32 * 2.0 - 1.0) * HALF;
                    let u = ((x as f32 + 0.5) / RES as f32 * 2.0 - 1.0) * HALF;
                    // Axis x runs u along y; axis y runs v along x.
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

    /// A RAY GRAZING A SHELL takes the face it passes, not the one behind it,
    /// whatever the model's field says: here the field's plate is tilted a few
    /// degrees off the cards' (a field's distances are samples, and a shade's
    /// silhouette is where they are least sure), so by the field the ray has
    /// just passed the upper face -- by what the cards saw it just meets it. A
    /// normal from the field, turned inside out there, took the underside's
    /// cards (2 and 4): the glowing inside of a sconce's shade, as white dots
    /// along its reflected outline that came and went as the head moved
    /// (headset, 2026-09-29 23:25).
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

    /// A CARD THAT SAW WHAT FACES AWAY FROM THE RAY does not colour it: a
    /// horizontal plate met from below -- the underside of a sconce's collar --
    /// where the card looking up saw, 5 mm lower and within its depth
    /// tolerance, a bright surface turned away from the ray: the shade's
    /// glowing inside, in front of the collar. No card saw the collar, and the
    /// cards say so (`w` 0) rather than pass off the inside as it.
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
                // The card looking up (3) sees the inside 5 mm below the
                // collar's underside, bright; the others see nothing there.
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

    /// THE INSIDE OF A STEEP SHADE, seen only by the card below it and only at
    /// a slant (its normal 18 degrees below the horizontal), is that card's
    /// colour, trusted fully: how squarely a card sees a surface weighs it
    /// among the cards, not whether it vouches. Taken from the slant alone, the
    /// trust was a third, and a sconce's glowing mouth reflected grey
    /// (offline, 2026-09-30).
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
                    // The card looking up sees the inside through y = 0 at the
                    // box's centre, sloping as its normal says: along the
                    // card's v, which is x, its depth changes by -x * n.x / n.y.
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
        // The field stops short of the surface on the ray's side: below it.
        let hit = CENTRE + Vec3::new(0.0, -0.003, 0.0);
        let Some(c) = colours_of(field(), cards, Quat::IDENTITY, true, &[(hit, d)]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!((c[0][0] - 100.0).abs() < 1.0, "the card below's colour: {:?}", c[0]);
        assert!(c[0][3] > 0.99, "trusted fully: {:?}", c[0]);
    }

    /// A CARD READ FROM AFAR IS TESTED WHERE THE HIT IS, not over its
    /// footprint: the card looking up saw a surface whose normal alternates
    /// column by column -- the inside of a shade all round its rim, facing in
    /// from every side -- so read coarse its normal averages to "straight
    /// down", facing a ray climbing toward it, while at the hit's own texel it
    /// faces away. The glowing inside vouched for the dark outside all round
    /// the rim of every sconce's reflection so, as white specks beside the
    /// mouth (offline, 2026-09-30).
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
        // At the centre of column 7 of the card looking up (u runs along z),
        // just below the surface.
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

    /// A HIT BEHIND WHAT A CARD SAW IS NOT WHAT IT SAW: at a bell-shaped
    /// shade's silhouette the card looking up saw, a centimetre BELOW the
    /// outside the ray met, the inside where the flare turns it just toward
    /// the ray -- within any depth range a card can afford, and facing the ray,
    /// but the hit lies behind it. White specks down both edges of every
    /// sconce's reflection (offline, 2026-09-30).
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
                    // The card looking up: the inside at y = -0.01, under the
                    // hit at y = 0.
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

    /// A CARD READ FROM AFAR VOUCHES ONLY FOR THE DEPTHS IT SAW: a wall facing
    /// +x -- the stem above a sconce's shade -- met by a ray climbing toward
    /// it, read at a coarse level, where the card looking up saw a bright
    /// surface turned toward the ray 5 cm below the hit: the shade's glowing
    /// inside under the stem. One averaged depth needed three coarse texels'
    /// tolerance -- half a metre here -- and the stem reflected as a line of
    /// white texels under the mouth (bench view hall_to_hallway_floor,
    /// 2026-09-30). The card that saw the wall (+x, colour 1) is the stem.
    #[test]
    fn a_card_read_from_afar_vouches_only_for_the_depths_it_saw() {
        let wall = 0.05;
        let mut texels = Vec::new();
        let mut normals = Vec::new();
        for face in 0..6u32 {
            for _ in 0..RES * RES {
                match face {
                    // The +x card: the wall, at x = 0.05.
                    0 => {
                        texels.push([1.0, 0.0, 0.0, 0.5 - 0.5 * wall / HALF]);
                        normals.push([1.0, 0.0, 0.0]);
                    }
                    // The card looking up: something bright and facing down
                    // at y = -0.05, below the hit at y = 0.
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
        // A footprint of 0.2 m at the hit: about ten card texels, level 3.
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

    /// A SMALL BRIGHT PART, FAR OFF, SHOWS AS ITS SHARE: the sphere's +x card
    /// carries a 2 x 2 texel glint of 100 on 1. Read near, the glint's centre
    /// is the glint; read from far off, where a pixel covers the whole card,
    /// it is the card's average -- a steady faint glint, not a texel that
    /// switches on and off as the head moves.
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

    /// The box's own frame: turned a quarter about y, the sphere's side facing
    /// world -z faces the box's +x, and takes the +x card.
    #[test]
    fn a_turned_box_is_read_in_its_own_frame() {
        let q = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let Some(c) = colours(q, true, &[(on_sphere(q * Vec3::X), q * -Vec3::X)]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!((c[0][0] - 1.0).abs() < 0.05, "{:?}", c[0]);
    }

    /// A proxy without cards says so, and the photographs' guess stands.
    #[test]
    fn a_proxy_without_cards_has_no_colour_from_them() {
        let Some(c) = colours(Quat::IDENTITY, false, &[(on_sphere(Vec3::X), -Vec3::X)]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert_eq!(c[0][3], 0.0, "{:?}", c[0]);
    }

    /// The sphere's cards with an albedo: `a` grey wherever a card saw it.
    fn albedo_cards(a: f32) -> ProxyCards {
        let mut c = cards();
        c.albedo = c.normals.iter().map(|n| if *n == [0.0; 3] { [0.0; 3] } else { [a; 3] }).collect();
        c
    }

    /// A point lamp at `at` the level's bake never saw -- a flashlight's kind.
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

    /// `colours_of` the sphere's `cards`, lit by `lamps` as `probe_fixup` lights
    /// them, or read as every other shader reads them for `None`.
    fn relit(lamps: Option<Vec<Light>>, cards: ProxyCards, rays: &[(Vec3, Vec3)]) -> Option<Vec<[f32; 4]>> {
        RELIT.with(|r| *r.borrow_mut() = lamps);
        let out = colours_of(field(), cards, Quat::IDENTITY, true, rays);
        RELIT.with(|r| *r.borrow_mut() = None);
        out
    }

    /// A FLASHLIGHT ON A LAMP LIGHTS ITS REFLECTION (`PROBE_CARD_RELIT`): a
    /// lamp the bake never saw, out along the +x side's normal, adds that
    /// side's albedo times the light arriving -- `a I / d^2` head on -- to its
    /// reflection, and nothing to the far side's, which faces away. The same
    /// lamp as one the bake saw adds nothing: its light is in the cards
    /// already. Nor does any lamp to cards baked without albedo.
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

/// A SPOT'S POOL SEEN EDGE-ON, as `spot_cone` draws it: test_room's corner
/// spot (34 degrees, a 16-degree core) 0.85 m from the wall it lights, the
/// wall seen from 15 m down the hall at about 5 degrees, where a pixel spans
/// 1.9 cm up the wall and 20 cm along it (2026-10-05).
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

    /// The cone at each `(point, toward the eye, the wall's normal, footprint)`:
    /// the lamp at the origin aimed along +x, run on the GPU through the
    /// shipped functions -- with the pixel's mean along its long step
    /// (`SPOT_EDGE_AVERAGE`), or without it, the cone as it was before. With
    /// `plane`, the long step from that plane (`long_step_from_plane`), as the
    /// brushes and the ground take it, whatever the normal shaded with.
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

    /// What a row of pixels `step` apart sees of `profile` (sampled every
    /// millimetre) as the head moves and the row slides `slide` at a time: per
    /// pixel, the RMS of the second difference of its value along the slide --
    /// `offline_frame`'s `measure_the_move_shimmer` in one dimension. A
    /// picture that moves smoothly scores nothing; what jumps scores.
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

    /// FAR, EDGE-ON, ALONG THE WALL the pool's edge was inside one pixel: a
    /// head moving 3 mm at a time (the wall 16 m off moves 0.16 of a pixel)
    /// saw pixels jump along the pool's edge, the far corner spot shimmering
    /// from the hall's front doorways. Averaged over what the pixel spans
    /// along the wall, the jumps go and the pool keeps its light. UP the wall,
    /// where the same pixel is short, and for a pixel seen head-on, the cone is
    /// exactly what it was.
    #[test]
    fn an_edge_on_pool_is_averaged_along_the_long_step_alone() {
        let footprint = (ALONG * UP).sqrt();
        // The wall faces the lamp (-x); the eye is 15.8 m down it along +z
        // and 1.5 m in from it, about 5 degrees off the wall's plane -- where
        // the footprint over the root of that cosine is the 20 cm a pixel
        // spans along the wall (`spot_long_step`).
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
            q.push((*p, -wall_n, wall_n, UP)); // head-on
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
        // The least any filter can do: each pixel the exact mean of the pool
        // as it was over the 20 cm it spans along the wall, a sensor's pixel.
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
        // A mean moves no light, to first order: the cosine's span is taken
        // as straight across the pixel, and over 20 cm a lamp 0.85 m off it
        // bends, which overstates the pool's light here by about 3%.
        assert!(
            (total(&along_now) / total(&along_was) - 1.0).abs() < 0.05,
            "averaging moved the pool's light along the wall: {} -> {}",
            total(&along_was),
            total(&along_now),
        );
        assert!(largest(&up_now, &up_was) < 1e-5, "the pool changed up the wall, where the pixel is short");
        assert!(largest(&head_now, &head_was) < 1e-5, "a pool seen head-on changed");
    }

    /// A BUMP TURNED EDGE-ON TO THE EYE PULLS NO LIGHT PAST THE POOL. Where the
    /// shader knows the surface's own plane (`long_step_from_plane`), the
    /// pixel's long step is the patch of that plane it covers, whatever the
    /// normal map says. Taken from a bump's normal nearly square to the eye's
    /// line, the step stretched up to 32 times, and pixels just past the
    /// pool's edge took a share of its light: the white specks past the
    /// sconces' pools on the stone ceiling (headset, 2026-10-06). A flat
    /// pixel's cone is exactly what it was, seen square or edge-on.
    #[test]
    fn a_bump_turned_edge_on_pulls_no_light_past_the_pool() {
        let wall_n = Vec3::NEG_X;
        // The wall seen square from 3 m, a pixel 2 cm across; the bump's
        // normal tipped all but square to the eye's line.
        let (square, footprint) = (Vec3::NEG_X, 0.02);
        let bump = Vec3::new(-0.0005, 1.0, 0.0).normalize();
        // Just past the pool's edge: 17.3 to 19.4 degrees off the beam,
        // outside its 17.
        let past: Vec<Vec3> = (0..=35).map(|i| Vec3::new(WALL, 0.0, 0.265 + i as f32 * 0.001)).collect();
        let bumpy: Vec<_> = past.iter().map(|&p| (p, square, bump, footprint)).collect();
        // Flat, square and edge-on (`an_edge_on_pool_is_averaged_along_the_long_step_alone`'s
        // eye), across the whole soft edge.
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

/// THE LIT SURFACES' OWN LOOP (`surface_lights`) against the lamp loop's
/// maths for the same lights (`light_contribution_split`), on the GPU through
/// the shipped functions and one uploaded list: a flashlight's bounce off a
/// wall, a second as a point, and a lamp behind them that is not one.
#[cfg(test)]
mod surface_light_gpu_tests {
    use super::{Light, LightKind, LightsUniform};
    use glam::Vec3;
    use wgpu::util::DeviceExt;
    use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages, ShaderModuleDescriptor, ShaderSource};

    /// Per `(point, normal)`: the surface loop's light, and the lamp maths'
    /// diffuse summed over the lights the upload counted as surfaces' -- with
    /// `spot_cone`'s average along the pixel's long step, or without it.
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
    // A pixel 5 mm across, and normal-mapped stone's terminator width.
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

    /// THE SAME LIGHT AS THE LAMP MATHS GIVE with no average along the
    /// pixel's long step -- the falloff, the patch's radius, the window, the
    /// cone's ramp, the terminator and the colour, to rounding -- and exactly
    /// none past the lights' range. With that average, the lamp maths
    /// stretch the ramp about its middle, a box filter's width for an edge a
    /// pixel wide (`spot_cone`); across an edge this wide a pixel's true
    /// mean is the plain ramp to second order, so the stretch is what moves:
    /// printed, by distance from the bounce's light.
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
            // A spiral out of the patch to past the bounce's range, every way.
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
        // Past the range of both: nothing, from either.
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

/// THE CHARACTERS' CONTACT DARKENING, as the shader computes it.
#[cfg(test)]
mod capsule_ambient_tests {
    use glam::Vec3;
    use wgpu::util::DeviceExt;
    use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages, ShaderModuleDescriptor, ShaderSource};

    use crate::renderer::uniforms::{CapsuleGroup, CapsuleUpload, Uniforms};

    /// The shader's reach, mirrored: past this from a capsule's surface it
    /// darkens nothing.
    const REACH: f32 = 0.6;

    /// A standing body as capsules, its right arm hanging or held out toward
    /// -z. Returned with the right arm's own capsule.
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

    /// How far `p` is from a capsule's surface.
    fn from_surface(p: Vec3, (a, b, r): (Vec3, Vec3, f32)) -> f32 {
        let ba = b - a;
        let s = ((p - a).dot(ba) / ba.length_squared().max(1e-8)).clamp(0.0, 1.0);
        (a + ba * s - p).length() - r
    }

    /// `capsule_ambient` at each `(point, normal)`, run on the GPU.
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

    /// A wall half a metre in front of the body, facing it, sampled a
    /// centimetre apart across and up; and the floor round it, 5 cm apart.
    fn wall_and_floor() -> (Vec<(Vec3, Vec3)>, Vec<(Vec3, Vec3)>, Vec<(Vec3, Vec3)>) {
        let across = (0..=400).map(|i| (Vec3::new(-2.0 + i as f32 * 0.01, 1.2, -0.5), Vec3::Z)).collect();
        let up = (0..=300).map(|i| (Vec3::new(0.05, i as f32 * 0.01, -0.5), Vec3::Z)).collect();
        let floor = (0..=80)
            .flat_map(|i| (0..=80).map(move |j| (Vec3::new(-2.0 + i as f32 * 0.05, 0.0, -2.0 + j as f32 * 0.05), Vec3::Y)))
            .collect();
        (across, up, floor)
    }

    /// NO RIM. Across a wall the darkening fades out; it never steps. It used
    /// to stop dead at a sphere round the whole body, where the capsules
    /// together still took several percent: a disc with a hard edge on the
    /// marble pillar (headset, 2026-10-01).
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
                // Still darker somewhere: the fade did not take it all away.
                let darkest = vis.iter().cloned().fold(1.0f32, f32::min);
                assert!(darkest < 0.97, "arm out {arm_out}: nothing darkened ({darkest})");
            }
        }
    }

    /// RAISING AN ARM CHANGES THE ARM'S DARKENING, AND NO OTHER. Everywhere
    /// farther than the reach from both of the arm's places reads the same
    /// with it hanging and held out. Faded by the bound round the whole body,
    /// the disc grew and shrank with the arm (headset, 2026-10-01).
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

    /// CONTACT STILL DARKENS: a hand held 5 cm off a wall shades the wall
    /// under it, and the feet the floor they stand on.
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

/// A CARRIED TORCH IN A REFLECTION, as the shader shows it: its body dark and
/// its own shape, its glass glowing toward what is in front of it, in front
/// of or behind the characters round it as it stands. See `capsule_reflection`
/// and `CapsuleGroup::surfaces`.
#[cfg(test)]
mod capsule_reflection_tests {
    use glam::Vec3;
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

    /// A torch 15 cm long, its glass at `at` facing `forward`.
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

    /// A character's trunk standing at `at`, wide enough that its middle
    /// covers a ray wholly through the shape blur.
    fn trunk(at: Vec3) -> CapsuleGroup {
        CapsuleGroup {
            capsules: vec![(at + Vec3::Y * 0.3, at - Vec3::Y * 0.3, 0.3)],
            colour: [0.5, 0.25, 0.1],
            surfaces: Vec::new(),
        }
    }

    /// What a ray's reflection shows: `capsule_reflection`'s answer over
    /// nothing, lit by pi so a body shows its albedo, and the glow it hands
    /// the caller, as it reaches the picture (times the answer's alpha).
    struct Seen {
        reflection: [f32; 4],
        glow: [f32; 3],
    }

    /// Each `(from, direction, pixel footprint)` ray, run on the GPU.
    fn reflect(groups: &[CapsuleGroup], rays: &[(Vec3, Vec3, f32)]) -> Option<Vec<Seen>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let caps = CapsuleUpload::from_groups(groups);
        let mut u: Uniforms = bytemuck::Zeroable::zeroed();
        u.capsules = caps.capsules;
        u.capsule_groups = caps.groups;
        u.capsule_params = [caps.group_count as f32, -1.0, -1.0, 0.0];
        // The eye far off: a mirror's footprint is then the pixel's alone.
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
        // No lamps, so no glass's beam to look past (`capsule_glass_beam`).
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

    /// THE GLASS GLOWS TOWARD WHAT IS IN FRONT OF IT: a mirror in front of a
    /// torch shows its glass as bright as the glass is drawn, across the
    /// glass and not beside it; from behind or beside the torch the glass is
    /// out of sight and the body shows, dark.
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
        // The glass alone shows nothing from behind: it faces one way, and
        // a body is not what hides it.
        let mut glass = torch(Vec3::new(0.0, 1.0, 0.0), Vec3::NEG_Z);
        glass.capsules.remove(0);
        glass.surfaces.remove(0);
        let Some(back) = reflect(&[glass], &rays[3..4]) else { return };
        assert_eq!(back[0].glow, [0.0; 3], "the glass's back");
    }

    /// A BLURRED GLASS KEEPS ITS LIGHT. However wide the footprint a rough
    /// surface or a far reflection spreads it over, the glow summed across
    /// the plane is the glass's own: its radiance times its area.
    #[test]
    fn a_blurred_glass_keeps_the_light_it_gives_off() {
        let at = Vec3::new(0.0, 1.0, 0.0);
        let glass_only = CapsuleGroup {
            capsules: vec![(at + Vec3::Z * 0.01, at, GLASS_RADIUS)],
            colour: [1.0; 3],
            surfaces: vec![DRIVE],
        };
        for f in [0.002, 0.01, 0.03, 0.1] {
            // Past the softened edge of the widest the glass spreads to.
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

    /// NEARER SHOWS. A torch held in front of a chest shows in front of it --
    /// of two capsules covering a ray alike, the first listed used to win,
    /// and the player's body is listed first -- and its glass behind a body is
    /// hidden by it.
    #[test]
    fn a_torch_in_front_of_a_body_shows_and_one_behind_it_is_hidden() {
        // Held across the chest, pointing to the side: the viewer sees the
        // torch's body with the trunk behind it.
        let across = [trunk(Vec3::new(0.0, 1.0, 0.4)), torch(Vec3::new(0.05, 1.0, 0.0), Vec3::X)];
        // Pointing at the viewer from behind someone.
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

    /// Each ray's glass glow as the reflecting point sees it past the glass's
    /// own beam (`capsule_glass_beam`): `lamps` uploaded with the first in
    /// spot layer 0 when `layered`, every tile of the spot atlas cleared to
    /// `depth` -- 1 hides nothing, 0 everything in a map's frustum.
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

    /// The player's flashlight as `flashlight::beam` makes it: a spot at the
    /// glass `at`, along `forward`, its map starting 2 cm out.
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

    /// THE GLASS IS HIDDEN WHERE ITS OWN BEAM IS: a mirror point the beam's
    /// shadow map says is shadowed -- a hand held up between the torch and the
    /// wall -- sees no glass in it, and one it lights sees it whole. A lamp
    /// that is not the glass's beam, a beam with no shadow layer, and a point
    /// outside the beam's map leave the glow as the capsules gave it.
    #[test]
    fn a_glass_is_hidden_where_its_own_beam_is_shadowed() {
        let glass = Vec3::new(0.0, 1.0, 0.0);
        let groups = [torch(glass, Vec3::NEG_Z)];
        // A wall 2 m down the beam, looking back at the glass: head on, and
        // 30 cm off the beam's axis, inside its cone. And a point well off to
        // the side of the glass, 80 degrees off the beam, outside its map.
        let rays = [
            (Vec3::new(0.0, 1.0, -2.0), Vec3::Z, 0.002),
            (Vec3::new(0.3, 1.0, -2.0), (glass - Vec3::new(0.3, 1.0, -2.0)).normalize(), 0.002),
            (Vec3::new(1.5, 1.0, -0.26), (glass - Vec3::new(1.5, 1.0, -0.26)).normalize(), 0.002),
        ];
        let lamp = beam(glass, Vec3::NEG_Z);
        // With no lamp at all, the glow as the capsules give it.
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
            assert!(whole[i][0] > 1.0, "{what}: the capsules give the glass a glow: {:?}", whole[i]);
            assert!(shadowed[i].iter().all(|g| *g < 1e-4), "{what}: the shadowed point still sees the glass: {:?}", shadowed[i]);
            for c in 0..3 {
                assert!(near(lit[i][c], whole[i][c], 1e-3 * whole[i][c]), "{what}: a lit point sees all of it: {:?}", lit[i]);
                assert_eq!(elsewhere[i][c], whole[i][c], "{what}: a lamp elsewhere is not its beam");
                assert_eq!(unlayered[i][c], whole[i][c], "{what}: a beam with no shadow layer");
            }
        }
        assert!(whole[2][0] > 1.0, "off to the side the glass still shows: {:?}", whole[2]);
        assert_eq!(shadowed[2], whole[2], "outside the beam's map nothing is known to hide it");
    }

    /// A CAPSULE KEEPS WHAT IT IS past one left out of the upload.
    #[test]
    fn a_capsule_keeps_what_it_is_past_one_left_out() {
        let mut g = torch(Vec3::ZERO, Vec3::NEG_Z);
        g.capsules.insert(0, (Vec3::splat(f32::NAN), Vec3::ZERO, 0.1));
        g.surfaces.insert(0, -0.5);
        let up = CapsuleUpload::from_groups(&[g]);
        assert_eq!(up.groups[1][3], 2.0, "two capsules kept");
        assert_eq!((up.capsules[1][3], up.capsules[3][3]), (-ALBEDO, DRIVE));
        // And a group with no surfaces given is all body.
        let t = CapsuleUpload::from_groups(&[trunk(Vec3::ZERO)]);
        assert_eq!(t.capsules[1][3], 0.0);
    }
}

/// A TORCH'S POOL IN A REFLECTION, as the fix-up lights it: a reflected point
/// on a surface the beam lights takes the beam's light on that surface's
/// albedo, as the surface itself does, shadowed by the beam's own map; a point
/// off the surface, past its reach or outside the cone takes none. See
/// `probe_surface_relit`.
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

    /// The player's flashlight as `flashlight::beam` makes it: a spot at
    /// `at` along `forward`, its map starting 2 cm out.
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

    /// The wall 3 m down the beam from a torch at (0, 1, 0) along -z, facing
    /// it, its pool map seen from the glass a little past the 25-degree cone.
    fn wall() -> LitSurface {
        LitSurface {
            normal: Vec3::Z,
            offset: -3.0,
            albedo: ALBEDO,
            lens: Vec3::new(0.0, 1.0, 0.0),
            forward: Vec3::NEG_Z,
            right: Vec3::X,
            tan_half: 26.25f32.to_radians().tan(),
        }
    }

    /// What `probe_surface_relit` gives each `(hit, ray)` once the pool maps
    /// are made (`pool_cards`) into a card atlas with one character's rows,
    /// the player standing at the world's origin: `lamps` with the first in
    /// spot layer 0 when `layered`, every tile of the spot atlas cleared to
    /// `depth`.
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
        // The maps' pipeline takes the scene's group 0; here, only what its
        // shader reads of it -- camera, lights, the shadow sampler and the
        // spot atlas.
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
        // No row, nothing read -- the poolless twins' condition.
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
        // The lookup's: the camera, the lights, the probes' sampler and the
        // card atlas.
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

    /// The wall's own light on the beam's axis, 3 m off, square on: I / d^2
    /// in the window `(1 - (d/r)^4)^2`. LAMP_RADIUS is far smaller.
    fn expected_scale() -> f32 {
        let window = (1.0 - (3.0f32 / 8.0).powi(4)).powi(2);
        40.0 / 9.0 * window
    }

    #[test]
    fn a_reflected_point_on_a_lit_wall_takes_the_beams_light_and_its_shadow() {
        let glass = Vec3::new(0.0, 1.0, 0.0);
        let lamp = beam(glass, Vec3::NEG_Z);
        // Seen from a polished floor in front of the wall, down and back.
        let ray = Vec3::new(0.0, 0.5, -1.0).normalize();
        let hits = [
            // On the beam's axis.
            (Vec3::new(0.0, 1.0, -3.0), ray),
            // 10 cm off the wall's plane: not on it.
            (Vec3::new(0.0, 1.0, -2.9), ray),
            // On it, outside the 25-degree cone and past the map's reach.
            (Vec3::new(1.9, 1.0, -3.0), ray),
            // On it, inside the cone, 50 cm off the axis.
            (Vec3::new(0.5, 1.0, -3.0), ray),
            // On it, just outside the cone (26 degrees), inside the map.
            (Vec3::new(1.463, 1.0, -3.0), ray),
        ];
        let Some(lit) = relit_at(&[lamp], true, 1.0, &[wall()], &hits) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let shadowed = relit_at(&[lamp], true, 0.0, &[wall()], &hits).unwrap();
        let no_surface = relit_at(&[lamp], true, 1.0, &[], &hits).unwrap();
        // The same wall with a map reaching only 30 cm round the axis.
        let narrow = relit_at(&[lamp], true, 1.0, &[LitSurface { tan_half: 0.1, ..wall() }], &hits).unwrap();
        let baked = relit_at(&[super::Light { in_level_bake: true, ..lamp }], true, 1.0, &[wall()], &hits).unwrap();
        // The wall as the fifth surface, behind four planes no hit lies on:
        // its map in the second band, second across -- made and read there.
        let decoy = |k: f32| LitSurface { normal: Vec3::Y, offset: 50.0 + k, ..wall() };
        let fifth = relit_at(&[lamp], true, 1.0, &[decoy(0.0), decoy(1.0), decoy(2.0), decoy(3.0), wall()], &hits).unwrap();
        assert!((fifth[0] - lit[0]).abs().max_element() < 1e-3 * expected_scale(), "the fifth map, read where it was made: {:?}", fifth[0]);
        eprintln!(
            "SURFACE RELIT: lit {lit:?}\n  shadowed {shadowed:?}\n  no surface {no_surface:?}\n  narrow {narrow:?}\n  a lamp the bake saw {baked:?}"
        );
        // The wall's own light on its axis, on its albedo, read from the map:
        // within its half-float texels.
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
}

#[cfg(test)]
mod tent_tests {
    use wgpu::util::DeviceExt;
    use wgpu::*;

    /// Each tent form at the same points of one random depth tile pair, on the
    /// GPU: the shipped loop, the written-out copy, the closed form per tap
    /// and the half-precision one (`f16` where the adapter has it). Columns:
    /// shipped, unrolled, lean, half.
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
        // Two 32x32 tiles side by side, every texel a random depth.
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
        // Points all over both tiles, edges included, at depths across the range.
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

    /// The largest and the mean difference of column `k` from the shipped one.
    fn spread(got: &[[f32; 4]], k: usize) -> (f32, f32) {
        let d: Vec<f32> = got.iter().map(|r| (r[k] - r[0]).abs()).collect();
        (d.iter().cloned().fold(0.0, f32::max), d.iter().sum::<f32>() / d.len() as f32)
    }

    /// THE LEAN TENT IS THE TENT: each tap's position and weight worked out at
    /// the tap give the shipped loop's shadow, over random depths at random
    /// points, to the precision of the hardware's own bilinear weights.
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

    /// At `f16` the half-precision tent moves a tap by at most a
    /// five-hundredth of a texel: within a few hundredths of a percent of the
    /// shipped shadow, and never darker than a bilinear weight's step.
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
