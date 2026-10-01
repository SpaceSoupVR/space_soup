//! Real-time shadow mapping for the desktop render path.
//!
//! Two shadow slots share one depth-only pipeline set and one comparison
//! sampler:
//!   * **Sun** — an orthographic shadow for the directional light.
//!   * **Spot** — a perspective shadow for the flashlight / shadow-casting
//!     spot light (hand-attached on the Quest; placed in a scene in the editor).
//!
//! Each slot renders every caster from its light's point of view into its own
//! depth texture; the main pass samples both (bound into the shared camera/
//! lights group — see `uniforms::UniformBuffer`) and darkens fragments that
//! fail the depth comparison.
//!
//! Casters: world-space cuboids and terrain (SolidVertex), level brushes
//! (BrushVertex), and non-skinned meshes -- which includes baked caves, because
//! a layered mesh keeps its ordinary vertex buffer alongside the weighted one
//! and this pass only ever reads position -- and SKINNED meshes, which are
//! posed here by the same joint matrices the lit pass uses. A character is the
//! one caster whose silhouette changes every frame, and it is also the one a
//! player can hold up in front of a light and check, so it is the first shadow
//! anybody notices missing.
//!
//! ONE PASS FOR BOTH EYES
//!
//! The sun's shadow map is built in LIGHT space and does not depend on where
//! the viewer is, so on the XR path it is rendered once per frame rather than
//! once per eye. That is not an optimisation to get to later -- doing it inside
//! the eye loop would double the cost of the most expensive thing here for an
//! identical result.

use glam::{Mat4, Vec3};
use wgpu::*;

use super::cuboid::SolidVertex;
use super::mesh::MeshVertex;

/// Side length of each (square) shadow depth texture on desktop.
///
/// A parameter rather than a constant because the headset cannot afford this
/// one: 2048 squared at Depth32Float is 16MB per slot, and the Quest is already
/// spending its bandwidth on two eyes at full resolution. See `QUEST_SHADOW_DIM`.
pub const SHADOW_DIM: u32 = 2048;

/// Side length used on the headset.
///
/// Half the desktop resolution in each axis, so a quarter of the memory and a
/// quarter of the fill. The honest cost is that a single map at this size,
/// stretched over an outdoor sightline, gives visibly chunky shadow edges far
/// from the viewer -- the real fix for that is cascades, which is a much larger
/// piece of work and is not this. Near shadows, which are the ones a player
/// actually reads cover from, hold up.
pub const QUEST_SHADOW_DIM: u32 = 1024;

/// Side of the moving-objects sun shadow map, in texels.
///
/// WHY A SECOND SUN MAP. The level's sun shadow never changes -- it is baked
/// into the brush mask and drawn once into the static map for everything else
/// -- but a player standing in sunlight must still cast one. Redrawing the
/// whole level into a head-following map every frame to get that was the old
/// arrangement, and it paid for every brush to shadow one avatar.
///
/// This map holds only what moves, over `SUN_DYNAMIC_RADIUS` around the
/// player, so it is small and nearly empty: 512 over a 6 m box is 1.2 cm a
/// texel, sharper than the level map, for 1 MB of depth and a pass with a
/// handful of draws in it.
pub const SUN_DYNAMIC_DIM: u32 = 512;

/// Half the side of the box the moving-objects sun map covers, in metres.
///
/// An avatar is under 2 m tall and this sun's shadow of it is about as long
/// again, so 3 m around the head holds the whole shadow with room to reach a
/// held object at arm's length. Past it, a moving object casts no sun shadow.
pub const SUN_DYNAMIC_RADIUS: f32 = 3.0;

/// How many spot lights can cast a real-time shadow at once.
///
/// A BUDGET, chosen rather than inherited. It was one, which is not a decision
/// anybody made -- the code took the first spot in the scene and the rest lit
/// without shadows, so a room with two matching lamps had one casting and one
/// not. Four covers a lit interior; each layer costs `dim * dim` of
/// Depth32Float, so at the Quest's 1024 that is 4MB apiece.
///
/// Lights beyond the budget still light the scene; they just do not occlude.
/// That is the honest failure -- a missing shadow rather than a missing light --
/// and the editor says which lights are affected.
pub const MAX_SPOT_SHADOWS: usize = 4;

/// Tiles per side of the spot shadow atlas.
///
/// Square rather than a strip: four tiles in a row makes an atlas four times
/// wider than tall, and a very wide render target wastes tiles on a tile GPU.
/// Derived from `MAX_SPOT_SHADOWS` so raising the budget does not silently
/// leave half the spots writing outside the texture.
pub const SPOT_ATLAS_COLS: u32 = 2;

/// THE CHARACTERS' OWN TILES: one each for the lamps lighting the player most
/// that hold no spot slot, fitted around the player's body and drawing only
/// the characters (`character_light_matrix`). Such a lamp otherwise has no
/// shadow of the characters at all -- and the user asked for "more defined
/// shadows from direct lights" (2026-09-29). Everything else the lamp's light
/// meets takes its shadow from the lamp's baked mask.
///
/// Tiles of the SUN'S MOVING-OBJECTS MAP, after its own (`SUN_ATLAS_TILES`):
/// that pass already draws the characters every frame, so a character tile
/// costs its draws and no pass of its own. In the spot atlas they took a
/// 2048x3072 pass, 0.7 ms a frame for two small tiles (trace, 2026-09-29).
pub const MAX_CHARACTER_SHADOWS: usize = 2;

/// Tiles of the moving-objects map, in one row: the sun's, then the
/// characters', then the sun's near tile. See `SUN_DYNAMIC_DIM`.
pub const SUN_ATLAS_TILES: u32 = 2 + MAX_CHARACTER_SHADOWS as u32;

/// THE PLAYER'S OWN SUN SHADOW AT TWICE THE DETAIL: the near tile holds the
/// middle half of the sun tile's box -- 1.5 m round the player's body -- at
/// 0.59 cm a texel where the sun tile has 1.17, and the shader reads it
/// wherever it holds the point (`sun_moving_visibility`). The sun tile still
/// holds everything out to 3 m.
///
/// WHY HALF THE BOX IS ENOUGH: the map is the sun's view, so a caster and its
/// shadow on the ground land on the SAME texel. The player's shadow can be as
/// long as they are tall, but the map only has to hold the player: their whole
/// body, arms raised, is inside 1.5 m of its middle seen along the sun.
///
/// WHY EXACTLY TWO: the near matrix is the sun tile's scaled by two about its
/// middle (`sun_near_matrix`), so a texel corner of the sun tile at `t` is at
/// `2t - SUN_DYNAMIC_DIM / 2` in the near tile -- a texel corner there too.
/// The world stays snapped to the near tile's grid as it is to the sun tile's
/// (`lights::dynamic_sun_matrix`), and the shader finds the near coordinates
/// from the sun tile's with no matrix of its own.
///
/// The user, 2026-09-30: the outdoor player shadows look pixelated -- "is there
/// anyway to make them look less pixellated while not requiring them to be
/// much higher quality?" A 1.17 cm texel is eight display pixels at two metres
/// on a Quest 3. The Quest filters the map linearly (logged at startup), so the
/// kernel was not the fault: the texel was.
pub const SUN_NEAR_ZOOM: f32 = 2.0;

/// The near tile's place in the row. See `SUN_NEAR_ZOOM`.
pub const SUN_NEAR_TILE: u32 = 1 + MAX_CHARACTER_SHADOWS as u32;

/// The near tile's matrix: the sun tile's, scaled by `SUN_NEAR_ZOOM` about the
/// middle of its box. The depth is the sun tile's, so one comparison value
/// serves both. See `SUN_NEAR_ZOOM`.
pub fn sun_near_matrix(sun_dynamic: Mat4) -> Mat4 {
    Mat4::from_scale(Vec3::new(SUN_NEAR_ZOOM, SUN_NEAR_ZOOM, 1.0)) * sun_dynamic
}

/// Light matrices in the uniform: the spots' in layer order, then the
/// characters' (`Uniforms::spot_view_proj`).
pub const SHADOW_MATRICES: usize = MAX_SPOT_SHADOWS + MAX_CHARACTER_SHADOWS;

/// Tile rows of the spot atlas.
pub const SPOT_ATLAS_ROWS: u32 = (MAX_SPOT_SHADOWS as u32).div_ceil(SPOT_ATLAS_COLS);

const _: () = assert!(
    (SPOT_ATLAS_COLS * SPOT_ATLAS_ROWS) as usize >= MAX_SPOT_SHADOWS,
    "the spot atlas must have a tile for every spot the budget allows",
);

/// Which tile of the spot atlas a layer occupies, as `(col, row)`.
pub fn spot_tile(layer: usize) -> (u32, u32) {
    let l = layer as u32;
    (l % SPOT_ATLAS_COLS, l / SPOT_ATLAS_COLS)
}

/// A lamp as [`character_shadow_lamps`] weighs it.
#[derive(Clone, Copy, Debug)]
pub struct CharacterLamp {
    pub position: Vec3,
    /// Where a spot points, and the cosine of its outer half-angle; -1 for a
    /// lamp that shines every way.
    pub direction: Vec3,
    pub cos_outer: f32,
    pub range: f32,
    pub intensity: f32,
    /// A point or spot lamp without a spot slot of its own.
    pub eligible: bool,
}

/// WHICH LAMPS CAST THE PLAYER'S CRISP SHADOW: of `lamps`, those lighting the
/// body -- the sphere at `centre` of `radius` -- most: by intensity over
/// squared distance, only within range, and for a spot only where some of the
/// body is inside its cone (the hall's pendants are downlights: a player a
/// step to the side of one is not lit by it at all). Strongest first, at most
/// `MAX_CHARACTER_SHADOWS`. A lamp `held` last frame (by position) keeps its
/// tile unless one left out lights the body a third more: two lamps lighting
/// it almost alike would otherwise trade the crisp shadow back and forth as
/// the player moves, and each trade swaps a sharp shadow for a soft one.
pub fn character_shadow_lamps(lamps: &[CharacterLamp], centre: Vec3, radius: f32, held: &[Vec3]) -> Vec<usize> {
    const KEEP: f32 = 1.0 / 1.33;
    let lights_body = |l: &CharacterLamp| {
        let to = centre - l.position;
        let d = to.length();
        if !l.eligible || l.intensity <= 0.0 || d >= l.range {
            return false;
        }
        if l.cos_outer <= -1.0 || d <= radius {
            return true;
        }
        // The body's own angular size widens the cone it may touch.
        let angle = l.direction.normalize_or_zero().dot(to / d).clamp(-1.0, 1.0).acos();
        angle <= l.cos_outer.clamp(-1.0, 1.0).acos() + (radius / d).min(1.0).asin()
    };
    let mut candidates: Vec<(usize, f32)> = lamps
        .iter()
        .enumerate()
        .filter(|(_, l)| lights_body(l))
        .map(|(i, l)| (i, l.intensity / ((l.position - centre).length_squared() + 0.25)))
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let was_held = |i: usize| held.iter().any(|h| (lamps[i].position - *h).length() < 0.01);
    let mut chosen: Vec<(usize, f32)> = candidates.iter().copied().filter(|c| was_held(c.0)).take(MAX_CHARACTER_SHADOWS).collect();
    for c in &candidates {
        if chosen.iter().any(|k| k.0 == c.0) {
            continue;
        }
        if chosen.len() < MAX_CHARACTER_SHADOWS {
            chosen.push(*c);
            continue;
        }
        // Full: the newcomer takes the weakest tile only by a clear margin.
        let (weakest, w) = chosen.iter().enumerate().min_by(|a, b| a.1 .1.total_cmp(&b.1 .1)).map(|(k, c)| (k, c.1)).unwrap();
        if w < c.1 * KEEP {
            chosen[weakest] = *c;
        }
    }
    chosen.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    chosen.into_iter().map(|c| c.0).collect()
}

/// A CHARACTER'S SHADOW FROM A LAMP: the lamp at `light` looking at the body
/// bounded by the sphere at `centre` of `radius`, just wide enough to hold it,
/// out to the lamp's `range` -- every receiver the body can shadow for this
/// lamp lies in that cone, and past its range the lamp lights nothing. `None`
/// when the lamp is inside the body's bound, which no frustum can hold.
pub fn character_light_matrix(light: Vec3, centre: Vec3, radius: f32, range: f32) -> Option<Mat4> {
    let to = centre - light;
    let dist = to.length();
    if !(dist > radius * 1.05 && radius > 0.0) {
        return None;
    }
    // The cone that holds the sphere, a tenth wider for the kernel.
    let half = (radius / dist).asin() * 1.1;
    let d = to / dist;
    let up = if d.dot(Vec3::Y).abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let view = Mat4::look_at_rh(light, centre, up);
    let near = (dist - radius).max(0.05);
    let far = range.max(dist + radius).max(near * 2.0);
    Some(Mat4::perspective_rh((2.0 * half).min(std::f32::consts::PI - 0.1), 1.0, near, far) * view)
}

/// Aim a pass at tile `tile` of the moving-objects map. See `SUN_ATLAS_TILES`.
fn sun_atlas_viewport(pass: &mut wgpu::RenderPass, tile: u32) {
    let d = SUN_DYNAMIC_DIM as f32;
    pass.set_viewport(tile as f32 * d, 0.0, d, d, 0.0, 1.0);
}

/// Which shadow slot a pass/upload targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowKind {
    Sun,
    /// The sun's shadow of MOVING things only, redrawn every frame over a
    /// small box around the player. See `SUN_DYNAMIC_DIM`.
    SunDynamic,
    /// The middle of that box at twice the detail. See `SUN_NEAR_ZOOM`.
    SunNear,
    /// One of the spot layers, by index.
    Spot(usize),
    /// One of the characters' tiles, by index. See `MAX_CHARACTER_SHADOWS`.
    Character(usize),
}

/// Builds the sun's light-space view-projection: an orthographic box aimed
/// along `dir`, centered on `center`, sized to enclose a sphere of `radius`.
pub fn directional_light_matrix(dir: Vec3, center: Vec3, radius: f32) -> Mat4 {
    let radius = radius.max(0.001);
    let d = {
        let n = dir.normalize_or_zero();
        if n == Vec3::ZERO {
            Vec3::NEG_Y
        } else {
            n
        }
    };
    let dist = radius * 2.0;
    let eye = center - d * dist;
    let up = if d.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(eye, center, up);
    // glam `orthographic_rh` maps z into [0,1] to match wgpu clip space.
    let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.0, dist + radius);
    proj * view
}

/// Builds a spot/flashlight light-space view-projection: a perspective frustum
/// from `pos` aimed along `dir`, with a vertical field of view covering the
/// spot's full cone angle (`cone_angle_deg`) and a far plane at `range`.
/// The six clip planes of a view-projection, as `(nx, ny, nz, d)` with the
/// interior on the positive side.
///
/// Gribb-Hartmann: each plane is a sum or difference of two rows of the matrix,
/// which works for an orthographic sun and a perspective spot alike because it
/// asks the matrix what it does rather than assuming how it was built.
pub fn frustum_planes(view_proj: Mat4) -> [glam::Vec4; 6] {
    let m = view_proj.to_cols_array_2d();
    // Row-major rows out of glam's column-major storage.
    let row = |r: usize| glam::Vec4::new(m[0][r], m[1][r], m[2][r], m[3][r]);
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    let norm = |p: glam::Vec4| {
        let len = p.truncate().length();
        if len > 1e-9 { p / len } else { p }
    };
    [
        norm(r3 + r0), // left
        norm(r3 - r0), // right
        norm(r3 + r1), // bottom
        norm(r3 - r1), // top
        norm(r3 + r2), // near
        norm(r3 - r2), // far
    ]
}

/// Whether an axis-aligned box is at all inside the frustum.
///
/// CONSERVATIVE: it may say yes for a box that is actually outside, near the
/// corners where several planes meet. That is the only safe direction to be
/// wrong in -- a false yes costs a few triangles, a false no deletes a shadow
/// and looks like the caster has vanished.
pub fn aabb_in_frustum(planes: &[glam::Vec4; 6], min: Vec3, max: Vec3) -> bool {
    for p in planes {
        // The box corner FURTHEST along the plane normal. If even that one is
        // behind the plane, every corner is, and the box is fully outside.
        let far = Vec3::new(
            if p.x >= 0.0 { max.x } else { min.x },
            if p.y >= 0.0 { max.y } else { min.y },
            if p.z >= 0.0 { max.z } else { min.z },
        );
        if p.truncate().dot(far) + p.w < 0.0 {
            return false;
        }
    }
    true
}

/// One spatially-coherent run of a caster's index buffer, with its bounds.
///
/// The unit frustum culling works on. A single 32k-triangle terrain draw cannot
/// be culled at all -- the light is standing on it, so its bounding box always
/// intersects -- and it was being redrawn in full into every shadow map: three
/// spots meant ~295k triangles a frame of depth-only work for a 4 m cone that
/// touches almost none of it.
#[derive(Debug, Clone, Copy)]
pub struct CasterChunk {
    pub first_index: u32,
    pub index_count: u32,
    pub min: Vec3,
    pub max: Vec3,
}

/// Where a spot light's shadow map begins, in metres from the bulb.
///
/// Far enough out to leave a hanging fixture's own housing in front of it, so
/// no lamp shadows the floor under itself. See `spot_light_matrix` for why this
/// is a constant rather than a fraction of the light's range.
pub const SPOT_SHADOW_NEAR: f32 = 0.30;

pub fn spot_light_matrix(pos: Vec3, dir: Vec3, cone_angle_deg: f32, range: f32) -> Mat4 {
    let d = {
        let n = dir.normalize_or_zero();
        if n == Vec3::ZERO {
            Vec3::NEG_Z
        } else {
            n
        }
    };
    let up = if d.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(pos, pos + d, up);
    // Pad the fov slightly beyond the full cone so the cone edge isn't clipped.
    let fov = (cone_angle_deg.to_radians() * 1.1).clamp(0.1, std::f32::consts::PI - 0.1);
    let far = range.max(0.2);
    // THE NEAR PLANE IS A PROPERTY OF THE FIXTURE, NOT OF THE LIGHT'S REACH.
    //
    // It used to be `far * 0.02`, which tied it to `range` -- a number chosen
    // for how far the light travels, which says nothing about how big its
    // housing is. In `test_room` that produced five different near planes
    // between 0.08 m and 0.32 m, so two hanging fixtures of the SAME design
    // behaved differently: the one on an 8 m light (near 0.16) had its shade
    // inside the shadow map and cast a dark dot in the middle of its own pool
    // of light, and the one on a 14 m light (near 0.28) had the shade in FRONT
    // of the near plane and cast nothing. Reported from the headset exactly
    // that way -- one fixture dots, the other does not (2026-09-21).
    //
    // A constant makes every lamp behave alike, and `a_lamp_does_not_shadow_
    // its_own_bulb` says what the intended behaviour is: a fixture must not
    // shadow the floor beneath itself. That test passed only BY ACCIDENT
    // before, for whichever lights happened to have a long enough range.
    //
    // 0.3 m clears a hanging lamp's housing. It is a real trade: a caster
    // genuinely within 30 cm of a bulb now casts nothing. For room fixtures
    // the only thing that close is the fixture, which is the thing being
    // excluded on purpose.
    //
    // Halved against `far` as well, so a deliberately short-range light -- a
    // muzzle flash, a small prop lamp -- cannot end up with its near plane
    // past its own far plane and produce an inside-out projection.
    let near = SPOT_SHADOW_NEAR.min(far * 0.5).max(0.02);
    let proj = Mat4::perspective_rh(fov, 1.0, near, far);
    proj * view
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightMatrix {
    view_proj: [[f32; 4]; 4],
}

/// One mesh caster for a shadow pass: vertex buffer, index buffer, index count,
/// and the mesh's model-matrix bind group (reused from the main mesh pass).
pub type ShadowMeshDraw<'a> = (&'a Buffer, &'a Buffer, u32, &'a BindGroup);

/// A skinned caster: vertices, indices, count, model uniform, joint matrices.
///
/// Separate from `ShadowMeshDraw` because it needs the pose. A character is the
/// one caster whose silhouette changes every frame, so it cannot share the
/// static path -- and without it a player watching their own hand pass through
/// a beam sees the light unbroken, which reads as the shadows being fake.
pub type ShadowSkinnedDraw<'a> = (&'a Buffer, &'a Buffer, u32, &'a BindGroup, &'a BindGroup);

/// One shadow map's own resources: depth texture and the per-frame light matrix.
struct ShadowSlot {
    /// `None` for a spot: its depth is a tile of the shared atlas, which the
    /// map owns, and only the light matrix below is per-spot.
    _depth_texture: Option<Texture>,
    depth_view: Option<TextureView>,
    light_buffer: Buffer,
    light_bind_group: BindGroup,
}

impl ShadowSlot {
    fn new(device: &Device, light_layout: &BindGroupLayout, label: &str, dim: u32) -> Self {
        Self::new_layered(device, light_layout, label, dim, 1, 0)
    }

    /// A slot that renders into one layer of an existing array texture.
    /// A slot with no attachment of its own: its depth lives in a tile of a
    /// shared atlas, and only its light matrix is per-spot.
    fn light_only(device: &Device, light_layout: &BindGroupLayout) -> Self {
        let (light_buffer, light_bind_group) = Self::light_uniform(device, light_layout);
        Self {
            _depth_texture: None,
            depth_view: None,
            light_buffer,
            light_bind_group,
        }
    }

    fn from_texture(
        device: &Device,
        light_layout: &BindGroupLayout,
        texture: &Texture,
        layer: u32,
    ) -> Self {
        let depth_view = texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });
        let (light_buffer, light_bind_group) = Self::light_uniform(device, light_layout);
        Self {
            _depth_texture: Some(texture.clone()),
            depth_view: Some(depth_view),
            light_buffer,
            light_bind_group,
        }
    }

    /// One layer of a shared array texture, or a standalone map when `layers`
    /// is 1.
    ///
    /// An ARRAY TEXTURE rather than an array of bindings. Binding several
    /// textures to one slot needs the TEXTURE_BINDING_ARRAY feature, which is
    /// not something to rely on for a headset; a depth_2d_array sampled by
    /// layer index is core WebGPU and works everywhere.
    fn new_layered(
        device: &Device,
        light_layout: &BindGroupLayout,
        label: &str,
        dim: u32,
        layers: u32,
        layer: u32,
    ) -> Self {
        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d { width: dim, height: dim, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        // A single-layer view, which is what a render pass attaches to. The
        // array view used for sampling is built separately, from the same
        // texture.
        let depth_view = depth_texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });
        let (light_buffer, light_bind_group) = Self::light_uniform(device, light_layout);
        Self {
            _depth_texture: Some(depth_texture),
            depth_view: Some(depth_view),
            light_buffer,
            light_bind_group,
        }
    }

    /// A standalone map `width` x `height`.
    fn new_sized(device: &Device, light_layout: &BindGroupLayout, label: &str, width: u32, height: u32) -> Self {
        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());
        let (light_buffer, light_bind_group) = Self::light_uniform(device, light_layout);
        Self { _depth_texture: Some(depth_texture), depth_view: Some(depth_view), light_buffer, light_bind_group }
    }

    /// The per-slot light matrix buffer and its bind group.
    fn light_uniform(device: &Device, light_layout: &BindGroupLayout) -> (Buffer, BindGroup) {
        let light_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("shadow_light_matrix"),
            size: std::mem::size_of::<LightMatrix>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("shadow_light_bg"),
            layout: light_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
        });
        (light_buffer, light_bind_group)
    }
}

pub struct ShadowMap {
    sun: ShadowSlot,
    /// See `SUN_DYNAMIC_DIM`.
    sun_dynamic: ShadowSlot,
    /// One slot per shadow-casting spot: its light matrix and bind group.
    ///
    /// AN ATLAS, NOT AN ARRAY, and the reason is measured rather than
    /// aesthetic. With one array layer per spot, each spot needed its own
    /// render pass -- and on a tile GPU a pass is a tile load/store cycle
    /// whatever is inside it. Culling 95.5% of the caster geometry (295,830
    /// indices down to 13,206) returned only 1.8 ms of the 3.2 ms that three
    /// spots cost, which says the cost is PASSES and not triangles.
    ///
    /// Tiles of one 2D texture can all be filled in a single pass by moving the
    /// viewport, so N spots cost one load/store instead of N.
    ///
    /// The price is that every sample must offset and clamp into its own tile,
    /// and a wrong clamp reads a neighbour's depth -- a shadow cast by a light
    /// that is not there. `each_spot_reads_its_own_shadow_map_and_not_a_
    /// neighbours` is the test that holds that line.
    spots: [ShadowSlot; MAX_SPOT_SHADOWS],
    /// The characters' tiles' light matrices; their depth is in the
    /// moving-objects map (`SUN_ATLAS_TILES`).
    characters: [ShadowSlot; MAX_CHARACTER_SHADOWS],
    /// The sun's near tile's light matrix; its depth is in the moving-objects
    /// map too. See `SUN_NEAR_ZOOM`.
    sun_near: ShadowSlot,
    _spot_texture: Texture,
    /// The whole atlas, as the shader samples it and as the pass renders to it.
    spot_array_view: TextureView,
    /// Side of one tile in texels; the atlas is `SPOT_ATLAS_COLS` of these.
    spot_tile_dim: u32,
    sampler: Sampler,
    solid_pipeline: RenderPipeline,
    mesh_pipeline: RenderPipeline,
    skinned_pipeline: RenderPipeline,
    /// Level geometry. A separate pipeline only because `BrushVertex` has a
    /// different stride -- the shader is `vs_solid`, unchanged, because a depth
    /// pass reads position and nothing else.
    brush_pipeline: RenderPipeline,
}

impl ShadowMap {
    /// Constructible from just the device — it owns a model bind-group layout
    /// with the same shape the mesh pipeline uses (wgpu treats identical
    /// descriptors as compatible), so the mesh pass's `ModelUniform` bind
    /// groups are reused here without rebinding, and there is no init-order
    /// cycle with `UniformBuffer`.
    pub fn new(device: &Device) -> Self {
        Self::with_dimension(device, SHADOW_DIM)
    }

    pub fn with_dimension(device: &Device, dim: u32) -> Self {
        let model_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("shadow_model_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                // MUST match the mesh pipeline's model layout exactly, flags
                // included. The bind groups drawn here are the ones the LIT
                // pass created, and they interoperate only because wgpu dedups
                // structurally identical layouts -- this file never sees the
                // mesh pipeline, so the identity is the whole contract.
                //
                // The shadow vertex stage does not read `params`, so FRAGMENT
                // is redundant to this shader and load-bearing anyway: when the
                // mesh layout gained FRAGMENT and this one did not, the two
                // stopped deduping and the avatar silently stopped casting. It
                // did not error -- the draw simply produced nothing.
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let light_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("shadow_light_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("shadow_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            compare: Some(CompareFunction::LessEqual),
            ..Default::default()
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("shadow_shader"),
            source: ShaderSource::Wgsl(
                SHADOW_SHADER
                    .replace(
                        "MAX_SKIN_JOINTS_PLACEHOLDER",
                        &super::mesh::MAX_SKIN_JOINTS.to_string(),
                    )
                    .into(),
            ),
        });

        let bias = DepthBiasState {
            constant: 2,
            slope_scale: 2.0,
            clamp: 0.0,
        };

        let solid_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("shadow_solid_layout"),
            bind_group_layouts: &[Some(&light_layout)],
            immediate_size: 0,
        });
        let solid_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("shadow_solid_pipeline"),
            layout: Some(&solid_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_solid"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(SolidVertex::layout())],
            },
            fragment: None,
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
                bias,
            }),
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Structurally identical to `SkinnedMeshPipeline`'s, because wgpu
        // matches bind group layouts by their descriptor rather than by
        // identity -- so the skin bind group a mesh already owns is accepted
        // here without building a second one. Declared locally because the
        // shadow maps are constructed before that pipeline exists.
        let skin_joint_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("skin_joint_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let mesh_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("shadow_mesh_layout"),
            bind_group_layouts: &[Some(&light_layout), Some(&model_layout)],
            immediate_size: 0,
        });
        let mesh_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("shadow_mesh_pipeline"),
            layout: Some(&mesh_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_mesh"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(MeshVertex::layout())],
            },
            fragment: None,
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
                bias,
            }),
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let skinned_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("shadow_skinned_layout"),
            bind_group_layouts: &[Some(&light_layout), Some(&model_layout), Some(&skin_joint_layout)],
            immediate_size: 0,
        });
        let skinned_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("shadow_skinned_pipeline"),
            layout: Some(&skinned_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_skinned"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(super::mesh::SkinnedMeshVertex::layout())],
            },
            fragment: None,
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                // No culling. A character's shadow should be cast by the whole
                // silhouette, and a skinned mesh can turn itself inside out at
                // an extreme pose -- dropping back faces would punch holes in
                // the shadow exactly when the limb is bent hardest.
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
                bias,
            }),
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let brush_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("shadow_brush_pipeline"),
            layout: Some(&solid_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_solid"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(super::brush_pipeline::BrushVertex::layout())],
            },
            fragment: None,
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
                bias,
            }),
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sun = ShadowSlot::new(device, &light_layout, "sun_shadow_depth", dim);
        // ONE 2D atlas, `SPOT_ATLAS_COLS` tiles a side, so every spot's depth
        // is filled by moving the viewport inside a single render pass. Same
        // total memory as the array this replaced -- four dim x dim tiles
        // either way -- and one tile load/store instead of four.
        let spot_texture = device.create_texture(&TextureDescriptor {
            label: Some("spot_shadow_atlas"),
            size: Extent3d { width: dim * SPOT_ATLAS_COLS, height: dim * SPOT_ATLAS_ROWS, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let spot_array_view = spot_texture.create_view(&TextureViewDescriptor::default());
        // Slots no longer own a view -- there is one attachment for all of them.
        // They keep their light matrix and its bind group, which is what still
        // differs per spot.
        let spots = std::array::from_fn(|_| ShadowSlot::light_only(device, &light_layout));
        let characters = std::array::from_fn(|_| ShadowSlot::light_only(device, &light_layout));
        let sun_near = ShadowSlot::light_only(device, &light_layout);
        // NOTE: a fifth `ShadowSlot` used to be allocated here and never read.
        // The four in `spots` are views into one array texture; this was a
        // whole separate depth target, created on every construction and used
        // by nothing. The compiler had been warning about the binding for some
        // time -- the wasted memory was the part nobody had noticed.

        // A row of tiles: the sun's, then the characters', then the sun's near.
        let sun_dynamic = ShadowSlot::new_sized(
            device,
            &light_layout,
            "sun_dynamic_shadow_depth",
            SUN_DYNAMIC_DIM * SUN_ATLAS_TILES,
            SUN_DYNAMIC_DIM,
        );

        Self {
            sun,
            sun_dynamic,
            spots,
            characters,
            sun_near,
            _spot_texture: spot_texture,
            spot_array_view,
            spot_tile_dim: dim,
            sampler,
            solid_pipeline,
            mesh_pipeline,
            skinned_pipeline,
            brush_pipeline,
        }
    }

    pub fn sun_depth_view(&self) -> &TextureView {
        self.sun.depth_view.as_ref().expect("the sun owns its own depth target")
    }

    /// The moving-objects sun map, as the shading pass samples it.
    pub fn sun_dynamic_depth_view(&self) -> &TextureView {
        self.sun_dynamic.depth_view.as_ref().expect("the dynamic sun owns its own depth target")
    }

    /// The spot shadow array, as the shading pass samples it.
    pub fn spot_depth_view(&self) -> &TextureView {
        &self.spot_array_view
    }

    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    fn slot(&self, kind: ShadowKind) -> &ShadowSlot {
        match kind {
            ShadowKind::Sun => &self.sun,
            ShadowKind::SunDynamic => &self.sun_dynamic,
            ShadowKind::SunNear => &self.sun_near,
            ShadowKind::Spot(i) => &self.spots[i.min(MAX_SPOT_SHADOWS - 1)],
            ShadowKind::Character(k) => &self.characters[k.min(MAX_CHARACTER_SHADOWS - 1)],
        }
    }

    /// Uploads a slot's light-space matrix for this frame.
    pub fn upload_light(&self, queue: &Queue, kind: ShadowKind, view_proj: Mat4) {
        let m = LightMatrix {
            view_proj: view_proj.to_cols_array_2d(),
        };
        queue.write_buffer(&self.slot(kind).light_buffer, 0, bytemuck::bytes_of(&m));
        // Its near tile's with it: always the same box, twice the detail.
        if kind == ShadowKind::SunDynamic {
            self.upload_light(queue, ShadowKind::SunNear, sun_near_matrix(view_proj));
        }
    }

    /// Every shadow-casting spot, in ONE render pass.
    ///
    /// THIS is why the spots share an atlas. On a tile GPU a render pass is a
    /// tile load/store cycle whatever is inside it, and the measurement that
    /// settled it was blunt: culling 95.5% of the caster geometry returned only
    /// 1.8 ms of the 3.2 ms that three spot passes cost. The cost was the
    /// passes.
    ///
    /// The atlas is cleared ONCE here, so tiles belonging to spots that are not
    /// casting this frame read as far depth -- unoccluded, which is the right
    /// answer for a light that is not there.
    ///
    /// Returns the indices actually drawn, for the frame diagnostic.
    #[allow(clippy::too_many_arguments)]
    pub fn record_spots(
        &self,
        encoder: &mut CommandEncoder,
        count: usize,
        view_proj: &[Mat4],
        solid: Option<(&Buffer, &Buffer, u32)>,
        brushes: Option<(&Buffer, &Buffer, u32)>,
        mesh_draws: &[ShadowMeshDraw],
        skinned_draws: &[ShadowSkinnedDraw],
        solid_chunks: &[CasterChunk],
    ) -> u32 {
        let mut drawn = 0u32;
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("spot_shadow_atlas_pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &self.spot_array_view,
                depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        for layer in 0..count.min(MAX_SPOT_SHADOWS) {
            let (col, row) = spot_tile(layer);
            let d = self.spot_tile_dim as f32;
            pass.set_viewport(col as f32 * d, row as f32 * d, d, d, 0.0, 1.0);
            // Per-spot frustum culling still applies -- it is worth less than
            // merging the passes was, but it is not worth nothing, and it is
            // already paid for.
            let planes = frustum_planes(view_proj[layer]);
            let light_bg = &self.spots[layer].light_bind_group;

            if let Some((vb, ib, total)) = solid {
                if total > 0 {
                    pass.set_pipeline(&self.solid_pipeline);
                    pass.set_bind_group(0, light_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    if solid_chunks.is_empty() {
                        pass.draw_indexed(0..total, 0, 0..1);
                        drawn += total;
                    } else {
                        for c in solid_chunks {
                            if !aabb_in_frustum(&planes, c.min, c.max) {
                                continue;
                            }
                            pass.draw_indexed(c.first_index..c.first_index + c.index_count, 0, 0..1);
                            drawn += c.index_count;
                        }
                    }
                }
            }
            if let Some((vb, ib, total)) = brushes {
                if total > 0 {
                    pass.set_pipeline(&self.brush_pipeline);
                    pass.set_bind_group(0, light_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..total, 0, 0..1);
                    drawn += total;
                }
            }
            if !mesh_draws.is_empty() {
                pass.set_pipeline(&self.mesh_pipeline);
                pass.set_bind_group(0, light_bg, &[]);
                for (vb, ib, count, model_bg) in mesh_draws {
                    pass.set_bind_group(1, *model_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
            }
            if !skinned_draws.is_empty() {
                pass.set_pipeline(&self.skinned_pipeline);
                pass.set_bind_group(0, light_bg, &[]);
                for (vb, ib, count, model_bg, skin_bg) in skinned_draws {
                    pass.set_bind_group(1, *model_bg, &[]);
                    pass.set_bind_group(2, *skin_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
            }
        }
        drop(pass);
        drawn
    }

    /// THE MOVING-OBJECTS MAP, in ONE pass: the sun's tile and its near tile
    /// when `sun` (every moving caster, from `ShadowKind::SunDynamic`'s matrix
    /// and `ShadowKind::SunNear`'s), then the first `characters` characters'
    /// tiles (the characters alone, from `ShadowKind::Character(k)`'s). Tiles
    /// not drawn read as far depth, unshadowed. Returns the indices drawn.
    pub fn record_moving(
        &self,
        encoder: &mut CommandEncoder,
        sun: bool,
        characters: usize,
        mesh_draws: &[ShadowMeshDraw],
        skinned_draws: &[ShadowSkinnedDraw],
    ) -> u32 {
        let mut drawn = 0u32;
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("moving_shadow_pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: self.sun_dynamic_depth_view(),
                depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        let mut skinned = |pass: &mut wgpu::RenderPass, light_bg: &BindGroup| {
            if skinned_draws.is_empty() {
                return;
            }
            pass.set_pipeline(&self.skinned_pipeline);
            pass.set_bind_group(0, light_bg, &[]);
            for (vb, ib, count, model_bg, skin_bg) in skinned_draws {
                pass.set_bind_group(1, *model_bg, &[]);
                pass.set_bind_group(2, *skin_bg, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..*count, 0, 0..1);
                drawn += *count;
            }
        };
        if sun {
            // The sun's tile, then the middle of its box at twice the detail
            // (`SUN_NEAR_ZOOM`): the same casters into both.
            for (tile, slot) in [(0, &self.sun_dynamic), (SUN_NEAR_TILE, &self.sun_near)] {
                sun_atlas_viewport(&mut pass, tile);
            if !mesh_draws.is_empty() {
                pass.set_pipeline(&self.mesh_pipeline);
                pass.set_bind_group(0, &slot.light_bind_group, &[]);
                for (vb, ib, count, model_bg) in mesh_draws {
                    pass.set_bind_group(1, *model_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
            }
            skinned(&mut pass, &slot.light_bind_group);
        }
        }
        for k in 0..characters.min(MAX_CHARACTER_SHADOWS) {
            sun_atlas_viewport(&mut pass, 1 + k as u32);
            skinned(&mut pass, &self.characters[k].light_bind_group);
        }
        drop(pass);
        drawn
    }

    /// Records a depth-only shadow pass for `kind` into `encoder`: clears that
    /// slot's depth texture, then draws the solid geometry and mesh casters
    /// from the light's viewpoint.
    pub fn record(
        &self,
        encoder: &mut CommandEncoder,
        kind: ShadowKind,
        solid: Option<(&Buffer, &Buffer, u32)>,
        brushes: Option<(&Buffer, &Buffer, u32)>,
        mesh_draws: &[ShadowMeshDraw],
        skinned_draws: &[ShadowSkinnedDraw],
        solid_chunks: &[CasterChunk],
        light_view_proj: Mat4,
    ) -> u32 {
        let slot = self.slot(kind);
        let planes = frustum_planes(light_view_proj);
        let mut drawn_indices = 0u32;
        let view = match kind {
            ShadowKind::Sun | ShadowKind::SunDynamic => {
                slot.depth_view.as_ref().expect("the sun owns its own depth target")
            }
            // A spot renders into a TILE of the shared atlas. Recording one spot
            // on its own still works and still clears the whole atlas, which is
            // why `record_spots` exists for the frame path -- one clear and one
            // store for all of them.
            ShadowKind::Spot(_) => &self.spot_array_view,
            // A tile of the moving-objects map. See `SUN_ATLAS_TILES`.
            ShadowKind::Character(_) | ShadowKind::SunNear => self.sun_dynamic.depth_view.as_ref().expect("the dynamic sun owns its own depth target"),
        };
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("shadow_pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        // The moving-objects sun fills its tile and then the middle of its box
        // at twice the detail, as `record_moving` does. See `SUN_NEAR_ZOOM`.
        let tiles = match kind {
            ShadowKind::SunDynamic => vec![(kind, slot), (ShadowKind::SunNear, &self.sun_near)],
            _ => vec![(kind, slot)],
        };
        for (kind, slot) in tiles {
            match kind {
                ShadowKind::Spot(i) => {
                    let (col, row) = spot_tile(i.min(MAX_SPOT_SHADOWS - 1));
                    let d = self.spot_tile_dim as f32;
                    pass.set_viewport(col as f32 * d, row as f32 * d, d, d, 0.0, 1.0);
                }
                ShadowKind::SunDynamic => sun_atlas_viewport(&mut pass, 0),
                ShadowKind::SunNear => sun_atlas_viewport(&mut pass, SUN_NEAR_TILE),
                ShadowKind::Character(k) => sun_atlas_viewport(&mut pass, 1 + k.min(MAX_CHARACTER_SHADOWS - 1) as u32),
                ShadowKind::Sun => {}
            }

            if let Some((vb, ib, count)) = solid {
                if count > 0 {
                    pass.set_pipeline(&self.solid_pipeline);
                    pass.set_bind_group(0, &slot.light_bind_group, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    if solid_chunks.is_empty() {
                        // No chunking supplied: draw it whole. A caster nobody has
                        // partitioned is still a caster, and silently dropping it
                        // would be a missing shadow rather than a slow one.
                        pass.draw_indexed(0..count, 0, 0..1);
                        drawn_indices += count;
                    } else {
                        for c in solid_chunks {
                            if !aabb_in_frustum(&planes, c.min, c.max) {
                                continue;
                            }
                            pass.draw_indexed(
                                c.first_index..c.first_index + c.index_count,
                                0,
                                0..1,
                            );
                            drawn_indices += c.index_count;
                        }
                    }
                }
            }

            if let Some((vb, ib, count)) = brushes {
                if count > 0 {
                    pass.set_pipeline(&self.brush_pipeline);
                    pass.set_bind_group(0, &slot.light_bind_group, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..count, 0, 0..1);
                    drawn_indices += count;
                }
            }

            if !mesh_draws.is_empty() {
                pass.set_pipeline(&self.mesh_pipeline);
                pass.set_bind_group(0, &slot.light_bind_group, &[]);
                for (vb, ib, count, model_bg) in mesh_draws {
                    pass.set_bind_group(1, *model_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
            }

            if !skinned_draws.is_empty() {
                pass.set_pipeline(&self.skinned_pipeline);
                pass.set_bind_group(0, &slot.light_bind_group, &[]);
                for (vb, ib, count, model_bg, skin_bg) in skinned_draws {
                    pass.set_bind_group(1, *model_bg, &[]);
                    pass.set_bind_group(2, *skin_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
            }
        }
        drop(pass);
        drawn_indices
    }
}

/// Depth-only shadow-caster shaders shared by both slots. `vs_solid` draws
/// world-space cuboid geometry; `vs_mesh` applies the mesh's model matrix.
const SHADOW_SHADER: &str = r#"
struct LightMatrix { view_proj: mat4x4<f32> }
@group(0) @binding(0) var<uniform> light: LightMatrix;

struct ModelUniform { model: mat4x4<f32> }
@group(1) @binding(0) var<uniform> model_u: ModelUniform;

@vertex
fn vs_solid(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return light.view_proj * vec4<f32>(pos, 1.0);
}

@vertex
fn vs_mesh(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return light.view_proj * model_u.model * vec4<f32>(pos, 1.0);
}

struct JointMatrices { mats: array<mat4x4<f32>, MAX_SKIN_JOINTS_PLACEHOLDER> }
@group(2) @binding(0) var<uniform> joints: JointMatrices;

// A skinned caster, posed exactly the way the lit pass poses it.
//
// The skinning has to be repeated here rather than shared, because a shadow map
// is rendered from the light and the lit pass from the eye -- the only thing
// they can share is the arithmetic, and it is four multiply-adds. Getting it
// subtly different is the failure to watch for: the shadow would be cast by a
// slightly differently posed body than the one you can see, which reads as the
// shadow lagging or sliding rather than as a skinning bug.
@vertex
fn vs_skinned(
    @location(0) pos: vec3<f32>,
    @location(1) _normal: vec3<f32>,
    @location(2) _uv: vec2<f32>,
    @location(3) joint_ids: vec4<u32>,
    @location(4) joint_weights: vec4<f32>,
) -> @builtin(position) vec4<f32> {
    let p = vec4<f32>(pos, 1.0);
    let skinned =
        (joints.mats[joint_ids.x] * p) * joint_weights.x +
        (joints.mats[joint_ids.y] * p) * joint_weights.y +
        (joints.mats[joint_ids.z] * p) * joint_weights.z +
        (joints.mats[joint_ids.w] * p) * joint_weights.w;
    return light.view_proj * model_u.model * skinned;
}
"#;

#[cfg(test)]
mod atlas_tests {
    use super::*;

    /// The atlas's grid, columns by rows, as the shader's `pcf_layer` has it.
    fn grid() -> glam::Vec2 {
        glam::Vec2::new(SPOT_ATLAS_COLS as f32, SPOT_ATLAS_ROWS as f32)
    }

    #[test]
    fn every_tile_is_its_own() {
        let tiles: Vec<(u32, u32)> = (0..MAX_SPOT_SHADOWS).map(spot_tile).collect();
        let unique: std::collections::HashSet<_> = tiles.iter().collect();
        assert_eq!(unique.len(), MAX_SPOT_SHADOWS, "two tiles coincide: {tiles:?}");
        for (c, r) in tiles {
            assert!(c < SPOT_ATLAS_COLS && r < SPOT_ATLAS_ROWS, "tile ({c},{r}) is off the atlas");
        }
    }

    /// The shader's clamp, in Rust, so the arithmetic can be checked without a GPU.
    ///
    /// NOT a substitute for rendering it. This proves the FORMULA keeps a kernel
    /// inside its tile; it cannot prove the shader uses the formula. Four
    /// attempts at a render test for this all passed with the clamp removed --
    /// most likely because `rank_for_budget` reorders lights, so which tile a
    /// given lamp occupies is not what the geometry assumed -- and a green test
    /// that cannot fail is worse than none, so they were deleted rather than
    /// kept. The gap is real and is recorded here on purpose.
    fn clamped_atlas_uv(layer: usize, tile_uv: glam::Vec2, step: (i32, i32), dim: u32) -> glam::Vec2 {
        let atlas_texel = glam::Vec2::ONE / (glam::Vec2::splat(dim as f32) * grid());
        let tile_texel = atlas_texel * grid();
        let guard = tile_texel * 0.5;
        let (c, r) = spot_tile(layer);
        let off = glam::Vec2::new(step.0 as f32, step.1 as f32) * tile_texel;
        let local = (tile_uv + off).clamp(guard, glam::Vec2::ONE - guard);
        (local + glam::Vec2::new(c as f32, r as f32)) / grid()
    }

    #[test]
    fn a_kernel_at_a_tiles_edge_stays_inside_that_tile() {
        // The atlas's own hazard: a 3x3 kernel taken at a tile's border reaches
        // into the tile beside it and reads another lamp's depth -- a shadow
        // cast by a light that is nowhere near.
        let dim = 4;
        for layer in 0..MAX_SPOT_SHADOWS {
            let (c, r) = spot_tile(layer);
            let lo = glam::Vec2::new(c as f32, r as f32) / grid();
            let hi = lo + glam::Vec2::ONE / grid();
            for corner in [
                glam::Vec2::new(0.0, 0.0),
                glam::Vec2::new(1.0, 0.0),
                glam::Vec2::new(0.0, 1.0),
                glam::Vec2::new(1.0, 1.0),
            ] {
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let uv = clamped_atlas_uv(layer, corner, (dx, dy), dim);
                        assert!(
                            uv.x >= lo.x && uv.x <= hi.x && uv.y >= lo.y && uv.y <= hi.y,
                            "layer {layer} corner {corner:?} step ({dx},{dy}) sampled {uv:?}, \
                             outside its tile {lo:?}..{hi:?}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn without_the_guard_the_kernel_would_leave_its_tile() {
        // The test above only means something if the situation it guards
        // against can actually arise. The same arithmetic without the clamp
        // must escape, or the guard is protecting nothing.
        let dim = 4;
        let tile_texel = glam::Vec2::ONE / glam::Vec2::splat(dim as f32);
        let (c, r) = spot_tile(MAX_SPOT_SHADOWS - 1);
        let lo = glam::Vec2::new(c as f32, r as f32) / grid();
        let unclamped = (glam::Vec2::new(-1.0, -1.0) * tile_texel + glam::Vec2::new(c as f32, r as f32)) / grid();
        assert!(
            unclamped.x < lo.x || unclamped.y < lo.y,
            "the unclamped kernel stayed inside its tile, so the guard guards nothing",
        );
    }

    /// The lamps lighting the player most take the tiles, strongest first;
    /// a slotted, out-of-range or directional lamp never does; and a held lamp
    /// keeps its tile against one only a little brighter, not against one
    /// clearly brighter.
    #[test]
    fn the_lamps_lighting_the_player_most_cast_their_crisp_shadows() {
        let body = Vec3::new(0.0, 0.9, 0.0);
        let r = 0.8;
        let lamp = |x: f32, intensity: f32, eligible: bool| CharacterLamp {
            position: Vec3::new(x, 2.4, 0.0),
            direction: Vec3::NEG_Y,
            cos_outer: -1.0,
            range: 8.0,
            intensity,
            eligible,
        };
        let lamps = [lamp(1.0, 4.0, true), lamp(-2.0, 4.0, true), lamp(0.5, 50.0, false), lamp(12.0, 99.0, true), lamp(3.0, 4.0, true)];
        assert_eq!(character_shadow_lamps(&lamps, body, r, &[]), vec![0, 1], "nearest two eligible, strongest first");
        // Lamp 4 is now a little brighter than lamp 1 at the body, but lamp 1
        // held a tile: it stays.
        let lamps = [lamp(1.0, 4.0, true), lamp(-2.0, 4.0, true), lamp(0.5, 50.0, false), lamp(12.0, 99.0, true), lamp(2.0, 4.6, true)];
        let s = |i: usize| lamps[i].intensity / ((lamps[i].position - body).length_squared() + 0.25);
        assert!(s(4) > s(1) && s(4) < s(1) / (1.0 / 1.33), "the test lamps are not placed as meant");
        assert_eq!(character_shadow_lamps(&lamps, body, r, &[lamps[0].position, lamps[1].position]), vec![0, 1]);
        // Clearly brighter: it takes the weakest held tile.
        let lamps = [lamp(1.0, 4.0, true), lamp(-2.0, 4.0, true), lamp(0.5, 50.0, false), lamp(12.0, 99.0, true), lamp(1.5, 9.0, true)];
        assert_eq!(character_shadow_lamps(&lamps, body, r, &[lamps[0].position, lamps[1].position]), vec![4, 0]);
        // A downlight a step and a half to the side, with a 40 degree cone,
        // lights none of the body: it gets no tile however near it hangs.
        let beside = CharacterLamp { position: Vec3::new(1.8, 2.4, 0.0), cos_outer: 20f32.to_radians().cos(), intensity: 40.0, ..lamp(0.0, 0.0, true) };
        assert_eq!(character_shadow_lamps(&[beside, lamp(-2.0, 4.0, true)], body, r, &[]), vec![1]);
        let over = CharacterLamp { position: Vec3::new(0.3, 2.4, 0.0), ..beside };
        assert_eq!(character_shadow_lamps(&[over, lamp(-2.0, 4.0, true)], body, r, &[]), vec![0, 1]);
    }

    /// A character's tile holds the body whole from the lamp -- its bounding
    /// sphere inside the frustum -- and reaches the floor beyond it; and a lamp
    /// inside the body's bound gets none.
    #[test]
    fn a_character_tile_holds_the_body_and_what_it_shadows() {
        let lamp = Vec3::new(1.5, 2.4, 0.0);
        let centre = Vec3::new(0.0, 0.9, 0.0);
        let m = character_light_matrix(lamp, centre, 1.0, 8.0).expect("a lamp outside the body");
        let inside = |p: Vec3| {
            let c = m * p.extend(1.0);
            let n = c.truncate() / c.w;
            c.w > 0.0 && n.x.abs() <= 1.0 && n.y.abs() <= 1.0 && (0.0..=1.0).contains(&n.z)
        };
        let to = (centre - lamp).normalize();
        let side = to.cross(Vec3::Y).normalize();
        for p in [centre, centre + Vec3::Y * 0.95, centre - Vec3::Y * 0.95, centre + side * 0.95] {
            assert!(inside(p), "{p} of the body is outside its tile");
        }
        // The floor where the body's shadow falls, past the body from the lamp.
        let floor = lamp + to * ((lamp.y - 0.0) / -to.y);
        assert!(inside(floor), "the floor behind the body, {floor}, is outside the tile");
        assert!(character_light_matrix(centre + Vec3::X * 0.2, centre, 1.0, 8.0).is_none());
    }
}

#[cfg(test)]
mod frustum_tests {
    use super::*;

    /// A spot at the origin looking down -Z, 60 degrees, 10 m range.
    fn spot() -> [glam::Vec4; 6] {
        frustum_planes(spot_light_matrix(Vec3::ZERO, Vec3::NEG_Z, 60.0, 10.0))
    }

    fn box_at(c: Vec3, half: f32) -> (Vec3, Vec3) {
        (c - Vec3::splat(half), c + Vec3::splat(half))
    }

    #[test]
    fn a_box_in_front_of_the_light_is_kept() {
        let (min, max) = box_at(Vec3::new(0.0, 0.0, -5.0), 0.5);
        assert!(aabb_in_frustum(&spot(), min, max), "a box 5m down the cone must be drawn");
    }

    #[test]
    fn a_box_behind_the_light_is_culled() {
        // The case that matters most for a spot pointing at a wall: everything
        // behind it is half the level.
        let (min, max) = box_at(Vec3::new(0.0, 0.0, 5.0), 0.5);
        assert!(!aabb_in_frustum(&spot(), min, max), "a box behind the light must be culled");
    }

    #[test]
    fn a_box_past_the_range_is_culled() {
        let (min, max) = box_at(Vec3::new(0.0, 0.0, -40.0), 0.5);
        assert!(!aabb_in_frustum(&spot(), min, max), "beyond the 10m range must be culled");
    }

    #[test]
    fn a_box_off_to_the_side_is_culled() {
        // 30 m sideways at 5 m depth is far outside a 60-degree cone.
        let (min, max) = box_at(Vec3::new(30.0, 0.0, -5.0), 0.5);
        assert!(!aabb_in_frustum(&spot(), min, max), "outside the cone must be culled");
    }

    #[test]
    fn a_box_containing_the_light_is_kept() {
        // The terrain case. The ground the lamp stands on straddles every
        // plane, and culling it would delete the shadow the lamp casts onto it.
        let (min, max) = box_at(Vec3::ZERO, 60.0);
        assert!(aabb_in_frustum(&spot(), min, max), "a box enclosing the light must be drawn");
    }

    #[test]
    fn culling_never_drops_a_box_that_straddles_a_plane() {
        // Conservative in the safe direction: crossing the far plane means part
        // of it is inside, and that part still casts.
        let (min, max) = box_at(Vec3::new(0.0, 0.0, -10.0), 2.0);
        assert!(aabb_in_frustum(&spot(), min, max), "a straddling box must be kept");
    }

    #[test]
    fn the_sun_frustum_culls_too() {
        // Same extraction on an ORTHOGRAPHIC matrix, which is the whole point
        // of doing it from the matrix rather than from the cone angle.
        let planes =
            frustum_planes(directional_light_matrix(Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO, 20.0));
        let (min, max) = box_at(Vec3::ZERO, 1.0);
        assert!(aabb_in_frustum(&planes, min, max), "under the sun box must be kept");
        let (fmin, fmax) = box_at(Vec3::new(500.0, 0.0, 0.0), 1.0);
        assert!(!aabb_in_frustum(&planes, fmin, fmax), "500m away must be culled");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_matrix_places_center_inside_the_clip_box() {
        let center = Vec3::new(1.0, 2.0, -3.0);
        let m = directional_light_matrix(Vec3::new(0.0, -1.0, 0.0), center, 10.0);
        let clip = m * center.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.x.abs() < 1e-4 && ndc.y.abs() < 1e-4, "center should project to the middle, got {ndc:?}");
        assert!(ndc.z > 0.0 && ndc.z < 1.0, "center depth should be inside [0,1], got {}", ndc.z);
    }

    #[test]
    fn points_farther_along_the_light_direction_have_greater_depth() {
        let center = Vec3::ZERO;
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let m = directional_light_matrix(dir, center, 10.0);
        let high = Vec3::new(0.0, 4.0, 0.0);
        let low = Vec3::new(0.0, -4.0, 0.0);
        let z = |p: Vec3| {
            let c = m * p.extend(1.0);
            c.z / c.w
        };
        assert!(z(high) < z(low), "point nearer the sun should have smaller depth: {} vs {}", z(high), z(low));
    }

    #[test]
    fn degenerate_direction_does_not_panic() {
        let m = directional_light_matrix(Vec3::ZERO, Vec3::ZERO, 5.0);
        assert!(m.is_finite());
    }

    #[test]
    fn spot_matrix_projects_target_ahead_into_view() {
        // Flashlight at origin aimed down -Z; a point 3m ahead should land near
        // the center of the shadow map with depth inside [0,1].
        let m = spot_light_matrix(Vec3::ZERO, Vec3::NEG_Z, 45.0, 10.0);
        let ahead = Vec3::new(0.0, 0.0, -3.0);
        let clip = m * ahead.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.x.abs() < 1e-3 && ndc.y.abs() < 1e-3, "aimed point should be centered, got {ndc:?}");
        assert!(ndc.z > 0.0 && ndc.z < 1.0, "aimed point depth should be inside [0,1], got {}", ndc.z);
    }

    #[test]
    fn spot_matrix_closer_point_has_smaller_depth() {
        let m = spot_light_matrix(Vec3::ZERO, Vec3::NEG_Z, 45.0, 10.0);
        let z = |p: Vec3| {
            let c = m * p.extend(1.0);
            c.z / c.w
        };
        assert!(z(Vec3::new(0.0, 0.0, -1.0)) < z(Vec3::new(0.0, 0.0, -5.0)), "closer point should have smaller depth");
    }
}

/// Render tests for the parts of shadowing that a matrix test cannot see.
///
/// The matrix tests above prove the projection is sane. They say nothing about
/// whether the depth pass runs, whether the main pass samples the map it wrote,
/// or whether a directional light is treated as directional -- all of which are
/// wiring, and all of which fail silently and look like a lighting bug.
#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::renderer::cuboid::SolidVertex;
    use crate::renderer::lights::{Light, LightKind, LightsUniform};
    use crate::renderer::pipeline::{lightmap_bind_group_layout, SolidPipeline};
    use crate::renderer::uniforms::{PlayerUpload, PostUpload, ShadowUpload, SkyUpload, UniformBuffer};
    use crate::renderer::Color3;
    use wgpu::util::DeviceExt;

    /// ODD, so the centre texel sits exactly at NDC (0, 0).
    ///
    /// With an even width the middle pixel is half a texel off centre, and the
    /// quad's world position there is not the position the test asked for. That
    /// is invisible for a test comparing two renders -- both are off by the
    /// same amount -- and it is fatal for one comparing against an absolute
    /// number, which is how `the_falloff_the_editor_predicts...` first came out
    /// 2% low and looked like a real disagreement with the editor.
    const SIZE: u32 = 9;

    fn gpu() -> Option<(Device, Queue)> {
        crate::renderer::terrain_pipeline::tests::headless_gpu()
    }

    fn vertex(pos: [f32; 3], normal: [f32; 3]) -> SolidVertex {
        SolidVertex {
            position: pos,
            normal,
            color: [1.0, 1.0, 1.0, 1.0],
            uv2: [0.0, 0.0],
            reflectivity: 0.0,
        }
    }

    /// A horizontal quad of the given size at height `y`, wound so the sun
    /// overhead sees its front.
    fn ground(y: f32, half: f32) -> (Vec<SolidVertex>, Vec<u32>) {
        let n = [0.0, 1.0, 0.0];
        let v = vec![
            vertex([-half, y, -half], n),
            vertex([half, y, -half], n),
            vertex([half, y, half], n),
            vertex([-half, y, half], n),
        ];
        (v, vec![0, 2, 1, 0, 3, 2])
    }

    /// What one render asks for.
    struct Scene {
        lights: Vec<Light>,
        /// Extra geometry drawn ONLY into the shadow map: the thing casting.
        caster: Option<(Vec<SolidVertex>, Vec<u32>)>,
        /// Where the receiving quad sits in the world.
        ///
        /// A full position and not just a height: the shadow frustum can be
        /// escaped two ways -- past the far plane, or off the side of the map --
        /// and only the second isolates the uv guard from everything else.
        receiver_at: glam::Vec3,
        /// Where the eye is RELATIVE TO THE RECEIVER.
        ///
        /// Relative and not absolute, because specular depends on the angle
        /// between the eye and the surface: an absolute eye left behind while
        /// the receiver moved 500m turned a test of distance falloff into a
        /// test of the highlight's geometry, and it failed for the wrong
        /// reason with a very convincing-looking number.
        eye_offset: glam::Vec3,
        shadows_on: bool,
        /// Side of ONE spot atlas tile, in texels.
        ///
        /// Tiny in the tile-bleed test and normal everywhere else: the guard
        /// that keeps a sample inside its own tile is half a texel wide, so at
        /// 256 it protects a strip 0.2% of the frustum across and no test could
        /// plausibly land in it. At 8 texels the same strip is 6% of the
        /// frustum, which a receiver can simply be placed in.
        shadow_dim: u32,
        /// The receiving quad's surface normal.
        ///
        /// Fixed pointing up until the sky arrived. The sky's ambient depends on
        /// which way a surface faces -- that being the whole difference between
        /// nine coefficients and one -- so it has to be a variable now.
        normal: [f32; 3],
        /// The sky's ambient. Defaults to the flat term the engine always had.
        sky: SkyUpload,
        /// Treat the directional light as the SKY's sun: the caster goes into
        /// the moving-objects map (`ShadowKind::SunDynamic`) and the static map
        /// is left off -- which is how a player shadows sunlit ground.
        sky_sun_dynamic: bool,
        /// The light (by index) whose characters' tile the caster is drawn
        /// into, as the player is: see `MAX_CHARACTER_SHADOWS`.
        character_lamp: Option<usize>,
    }

    /// Renders a lit quad filling the view and returns its centre pixel.
    ///
    /// The receiver is drawn with an identity view_proj so it fills the target
    /// whatever its world position -- which is what lets the same quad be moved
    /// hundreds of metres to test that a sun does not attenuate.
    fn shade_receiver(scene: Scene) -> Option<[u8; 4]> {
        let (device, queue) = gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let shadow_map = ShadowMap::with_dimension(&device, scene.shadow_dim);
        let lights_uniform = LightsUniform::new(&device);
        let uniforms = UniformBuffer::new(
            &device,
            &lights_uniform,
            shadow_map.sun_depth_view(),
            shadow_map.sun_dynamic_depth_view(),
            shadow_map.spot_depth_view(),
            shadow_map.sampler(),
        );
        // Spots claim shadow layers in scene order, exactly as both renderers
        // do -- so a test with two spots exercises two layers rather than one.
        let spot_indices: Vec<usize> = if scene.shadows_on {
            scene
                .lights
                .iter()
                .enumerate()
                .filter(|(_, l)| l.kind == LightKind::Spot)
                .map(|(i, _)| i)
                .take(MAX_SPOT_SHADOWS)
                .collect()
        } else {
            Vec::new()
        };
        lights_uniform.upload_frame(&queue, &scene.lights, &spot_indices, scene.sky_sun_dynamic);

        let sun = scene.lights.iter().find(|l| l.kind == LightKind::Directional);
        let sun_view_proj = sun
            .map(|l| directional_light_matrix(l.direction, Vec3::ZERO, 20.0))
            .unwrap_or(Mat4::IDENTITY);
        let mut spot_view_proj = [Mat4::IDENTITY; SHADOW_MATRICES];
        for (layer, &i) in spot_indices.iter().enumerate() {
            let l = &scene.lights[i];
            spot_view_proj[layer] =
                spot_light_matrix(l.position, l.direction, l.cone_angle_deg, l.range);
        }
        // The caster in a characters' tile, fitted round it from its lamp.
        let character_tile = scene.character_lamp.zip(scene.caster.as_ref()).and_then(|(i, (v, _))| {
            let centre = v.iter().fold(Vec3::ZERO, |s, p| s + Vec3::from(p.position)) / v.len().max(1) as f32;
            let radius = v.iter().map(|p| (Vec3::from(p.position) - centre).length()).fold(0.0f32, f32::max);
            let l = &scene.lights[i];
            character_light_matrix(l.position, centre, radius, l.range).map(|m| (i, m))
        });
        if let Some((_, m)) = character_tile {
            spot_view_proj[MAX_SPOT_SHADOWS] = m;
        }
        let upload = ShadowUpload {
            sun_view_proj,
            spot_view_proj,
            sun_enabled: scene.shadows_on && sun.is_some() && !scene.sky_sun_dynamic,
            spot_count: spot_indices.len() as u32,
            sun_dynamic_view_proj: sun_view_proj,
            sun_dynamic_enabled: scene.shadows_on && sun.is_some() && scene.sky_sun_dynamic,
        };

        // The receiver, as clip-space coordinates that fill the target. Its
        // WORLD position -- which is what the shadow lookup and the lights use
        // -- rides on the model translation the view_proj cancels out.
        let world = scene.receiver_at;
        let mut player = PlayerUpload::default();
        player.capsules.shadow_lights[0] = character_tile.map_or(-1.0, |(i, _)| i as f32);
        uniforms.upload_scene(
            &queue,
            glam::Mat4::from_translation(-world),
            world + scene.eye_offset,
            &upload,
            &scene.sky,
            &PostUpload::default(),
            &player,
        );

        let pipeline = SolidPipeline::new(&device, format, &uniforms.layout);
        let white = crate::renderer::mesh::create_lightmap_texture(
            &device,
            &queue,
            &lightmap_bind_group_layout(&device),
            &[255u8, 255, 255, 255],
            1,
            1,
            None,
        );

        let n = scene.normal;
        // z = 0 so the quad's world position is exactly `receiver_at`, for the
        // same reason SIZE is odd.
        let quad: Vec<SolidVertex> = [
            [-1.0, -1.0, 0.0],
            [3.0, -1.0, 0.0],
            [-1.0, 3.0, 0.0],
        ]
        .iter()
        .map(|p| {
            // Clip position, with the world position added back so world_pos in
            // the shader is the receiver's real place in the scene.
            vertex([p[0] + world.x, p[1] + world.y, p[2] + world.z], n)
        })
        .collect();

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("recv_vb"),
            contents: bytemuck::cast_slice(&quad),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("recv_ib"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        let caster_bufs = scene.caster.as_ref().map(|(v, i)| {
            (
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("caster_vb"),
                    contents: bytemuck::cast_slice(v),
                    usage: BufferUsages::VERTEX,
                }),
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("caster_ib"),
                    contents: bytemuck::cast_slice(i),
                    usage: BufferUsages::INDEX,
                }),
                i.len() as u32,
            )
        });

        let desc = |fmt, usage| TextureDescriptor {
            label: None,
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        };
        let target = device.create_texture(&desc(
            format,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let depth = device.create_texture(&desc(
            TextureFormat::Depth32Float,
            TextureUsages::RENDER_ATTACHMENT,
        ));
        let target_view = target.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("recv_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        if upload.sun_enabled {
            shadow_map.upload_light(&queue, ShadowKind::Sun, sun_view_proj);
            let solid = caster_bufs
                .as_ref()
                .map(|(vb, ib, count)| (vb, ib, *count));
            shadow_map.record(&mut encoder, ShadowKind::Sun, solid, None, &[], &[], &[], sun_view_proj);
        }
        if upload.sun_dynamic_enabled {
            shadow_map.upload_light(&queue, ShadowKind::SunDynamic, sun_view_proj);
            let solid = caster_bufs
                .as_ref()
                .map(|(vb, ib, count)| (vb, ib, *count));
            shadow_map.record(&mut encoder, ShadowKind::SunDynamic, solid, None, &[], &[], &[], sun_view_proj);
        }
        // One depth pass per spot layer, matching both renderers.
        for layer in 0..upload.spot_count as usize {
            shadow_map.upload_light(
                &queue,
                ShadowKind::Spot(layer),
                upload.spot_view_proj[layer],
            );
            let solid = caster_bufs
                .as_ref()
                .map(|(vb, ib, count)| (vb, ib, *count));
            shadow_map.record(
                &mut encoder, ShadowKind::Spot(layer), solid, None, &[], &[], &[],
                upload.spot_view_proj[layer],
            );
        }
        if let Some((_, m)) = character_tile {
            shadow_map.upload_light(&queue, ShadowKind::Character(0), m);
            let solid = caster_bufs.as_ref().map(|(vb, ib, count)| (vb, ib, *count));
            shadow_map.record(&mut encoder, ShadowKind::Character(0), solid, None, &[], &[], &[], m);
        }
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("recv_pass"),
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
            pass.set_bind_group(1, &white.bind_group, &[]);
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
        let at = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[at], data[at + 1], data[at + 2], data[at + 3]])
    }

    fn sun() -> Light {
        Light {
            mask_channel: None,
            position: Vec3::ZERO,
            // Straight down. `direction` is the way the light TRAVELS.
            direction: Vec3::NEG_Y,
            kind: LightKind::Directional,
            color: Color3(255, 255, 255, 255),
            // Dim on purpose. Ambient alone is already 0.6 of full white, so a
            // brighter sun saturates every channel and the specular difference
            // the eye test measures disappears into the clamp -- which it did,
            // at 0.5, with both eyes reading a flat 255.
            intensity: 0.2,
            // Ignored for a sun. Set small on purpose: if anything ever applied
            // range falloff to a directional light, these tests would go dark.
            range: 1.0,
            cone_angle_deg: 180.0,
            inner_cone_angle_deg: 0.0,
        }
    }

    macro_rules! shot {
        ($s:expr) => {
            match shade_receiver($s) {
                Some(px) => px,
                None => {
                    eprintln!("skipping: no GPU adapter available");
                    return;
                }
            }
        };
    }

    fn base() -> Scene {
        Scene {
            lights: vec![sun()],
            caster: None,
            receiver_at: Vec3::ZERO,
            eye_offset: Vec3::new(0.0, 5.0, 0.0),
            shadows_on: true,
            shadow_dim: 256,
            normal: [0.0, 1.0, 0.0],
            sky: SkyUpload::none(),
            sky_sun_dynamic: false,
            character_lamp: None,
        }
    }

    /// A POINT lamp without a spot slot shadows through its characters' tile
    /// -- the path the player's crisp shadow takes -- and nothing else it has
    /// holds the caster: without the tile the same caster casts nothing. Also
    /// the only test that reads a tile in the atlas's added rows.
    #[test]
    fn a_lamp_shadows_the_player_through_the_characters_tile() {
        let lamp = Light {
            mask_channel: None,
            position: Vec3::new(0.0, 4.0, 0.0),
            direction: Vec3::NEG_Y,
            kind: LightKind::Point,
            color: Color3(255, 255, 255, 255),
            intensity: 6.0,
            range: 12.0,
            cone_angle_deg: 90.0,
            inner_cone_angle_deg: 0.0,
        };
        let scene = |lights: Vec<Light>, tile: Option<usize>| Scene {
            lights,
            caster: Some(ground(2.0, 0.6)),
            character_lamp: tile,
            ..base()
        };
        let Some(shadowed) = shade_receiver(scene(vec![lamp.clone()], Some(0))) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let open = shade_receiver(scene(vec![lamp.clone()], None)).unwrap();
        let dark = shade_receiver(scene(Vec::new(), None)).unwrap();
        eprintln!("through the characters' tile {shadowed:?}, without it {open:?}, no lamp {dark:?}");
        assert!(open[0] as i32 - dark[0] as i32 > 40, "the lamp does not light the receiver measurably: {open:?} vs {dark:?}");
        assert!(
            (shadowed[0] as i32 - dark[0] as i32).abs() <= 2,
            "the characters' tile let the lamp's light through: {shadowed:?}, dark {dark:?}, open {open:?}",
        );
    }

    /// A spot above the receiver, aimed straight down.
    fn spot_above(x: f32, intensity: f32) -> Light {
        Light {
            mask_channel: None,
            position: Vec3::new(x, 6.0, 0.0),
            direction: Vec3::NEG_Y,
            kind: LightKind::Spot,
            color: Color3(255, 255, 255, 255),
            intensity,
            range: 30.0,
            cone_angle_deg: 120.0,
            inner_cone_angle_deg: 0.0,
        }
    }

    #[test]
    fn a_spot_light_casts_a_shadow() {
        // The baseline the second-spot test is measured against. If this did
        // not hold, the two-spot result would prove nothing.
        let lit = shot!(Scene { lights: vec![spot_above(0.0, 4.0)], ..base() });
        let shaded = shot!(Scene {
            lights: vec![spot_above(0.0, 4.0)],
            caster: Some(ground(3.0, 8.0)),
            ..base()
        });
        assert!(
            (lit[0] as i32) - (shaded[0] as i32) > 8,
            "a spot cast no shadow: {lit:?} lit vs {shaded:?} with a caster between",
        );
    }

    #[test]
    fn a_second_spot_light_also_casts_a_shadow() {
        // THE POINT OF THE SHADOW ARRAY. There used to be one spot depth map,
        // claimed by the first spot in the scene, so a room with two matching
        // lamps had one casting a shadow and one not -- which reads as a broken
        // light rather than an exhausted budget.
        //
        // The first light here is deliberately dark, so anything measured comes
        // from the SECOND one. With a single shared map the second light gets no
        // shadow at all and the caster makes no difference.
        let dim = spot_above(-4.0, 0.0);
        let lit = shot!(Scene { lights: vec![dim.clone(), spot_above(0.0, 4.0)], ..base() });
        let shaded = shot!(Scene {
            lights: vec![dim, spot_above(0.0, 4.0)],
            caster: Some(ground(3.0, 8.0)),
            ..base()
        });
        assert!(
            (lit[0] as i32) - (shaded[0] as i32) > 8,
            "the SECOND spot cast no shadow: {lit:?} lit vs {shaded:?} with a caster between",
        );
    }

    #[test]
    fn a_spot_beyond_the_budget_still_lights_without_shadowing() {
        // The honest failure past MAX_SPOT_SHADOWS: a missing shadow, never a
        // missing light. A level that silently dropped its overflow lamps would
        // go dark in patches, which is far worse than an unshadowed one.
        let mut lights: Vec<Light> = (0..MAX_SPOT_SHADOWS + 1)
            .map(|i| spot_above(-8.0 + i as f32, 0.0))
            .collect();
        *lights.last_mut().unwrap() = spot_above(0.0, 4.0);

        let with_extra = shot!(Scene { lights: lights.clone(), ..base() });
        let alone = shot!(Scene { lights: vec![spot_above(0.0, 4.0)], ..base() });
        assert!(
            (with_extra[0] as i32 - alone[0] as i32).abs() < 6,
            "the light past the shadow budget stopped lighting: {with_extra:?} vs {alone:?}",
        );
    }

    #[test]
    fn a_sun_lights_a_surface_facing_it() {
        let lit = shot!(base());
        let unlit = shot!(Scene { lights: vec![], ..base() });
        assert!(
            lit[0] as i32 - unlit[0] as i32 > 8,
            "a directional light contributed nothing: {lit:?} vs ambient {unlit:?}",
        );
    }

    #[test]
    fn a_sun_does_not_dim_with_distance() {
        // THE DEFINING PROPERTY. A sun's rays are parallel and infinitely far
        // away, so a surface 500m from the origin must be lit exactly as one at
        // the origin. Treating it as a point light -- the easiest mistake, since
        // it shares the struct -- makes distant terrain fade to ambient, which
        // reads as fog nobody asked for.
        // At the origin and 500m up. Both unshadowed, so the only thing that
        // could separate them is falloff.
        //
        // An earlier version compared 100m with 500m, to keep the shadow
        // frustum out of it. That made the test USELESS: a point-light falloff
        // has already collapsed to nothing by 100m, so both readings came back
        // at ambient and matched. It only passed for the wrong reason, and a
        // deliberate break proved it -- the failure landed on a different test
        // entirely.
        let near = shot!(base());
        let far = shot!(Scene { receiver_at: Vec3::new(0.0, 500.0, 0.0), ..base() });
        for c in 0..3 {
            assert!(
                (near[c] as i32 - far[c] as i32).abs() <= 2,
                "the sun attenuated with distance: {near:?} at the origin vs {far:?} at y=500",
            );
        }
    }

    #[test]
    fn an_occluder_darkens_what_is_under_it() {
        // The whole point, end to end: a depth pass that actually runs, a map
        // that is actually sampled, and a comparison that comes out the right
        // way round. Each half is invisible on its own -- a depth pass that
        // never runs leaves a cleared map that shadows nothing, which looks
        // exactly like a scene with no caster in it.
        let clear = shot!(base());
        let shaded = shot!(Scene {
            caster: Some(ground(3.0, 8.0)),
            ..base()
        });
        assert!(
            clear[0] as i32 - shaded[0] as i32 > 8,
            "a slab three metres overhead cast no shadow: {clear:?} lit vs {shaded:?} shadowed",
        );
    }

    #[test]
    fn a_spot_casts_a_shadow_through_its_own_map() {
        // The SPOT path end to end, rendered: matrix, depth pass, and the
        // sampling that reads it back. Written deliberately before the spot
        // shadow storage was reworked, as the thing that has to survive it --
        // the sun path had render coverage and the spot path had none, so a
        // change to how spot depth is stored could have broken every spot
        // shadow in the project with the whole suite still green.
        let unshadowed = shot!(Scene { lights: vec![spot_above(0.0, 6.0)], ..base() });
        let shadowed = shot!(Scene {
            lights: vec![spot_above(0.0, 6.0)],
            caster: Some(ground(3.0, 8.0)),
            ..base()
        });
        assert!(
            shadowed[0] < unshadowed[0],
            "a caster between a spot and the ground must darken it: \
             {shadowed:?} vs {unshadowed:?}",
        );
    }

    #[test]
    fn each_spot_reads_its_own_shadow_map_and_not_a_neighbours() {
        // Two spots, and only ONE of them occluded. Whatever holds their depth
        // -- array layers, or tiles of one atlas -- each light has to sample
        // the map that was rendered for it. Getting the indexing wrong makes a
        // lamp cast the shadow of a completely different lamp, which reads as
        // shadows appearing in impossible places rather than as an index bug.
        //
        // The occluder covers the FIRST spot only; the second is off to the
        // side with a clear path, so the receiver must stay partly lit.
        let both_clear = shot!(Scene {
            lights: vec![spot_above(0.0, 6.0), spot_above(2.5, 6.0)],
            ..base()
        });
        let one_blocked = shot!(Scene {
            lights: vec![spot_above(0.0, 6.0), spot_above(2.5, 6.0)],
            caster: Some(ground(3.0, 8.0)),
            ..base()
        });
        assert!(
            one_blocked[0] < both_clear[0],
            "blocking a spot must darken the ground: {one_blocked:?} vs {both_clear:?}",
        );
    }

    #[test]
    fn a_shadowed_surface_keeps_its_ambient_light() {
        // Shadow multiplies the LIGHT and not the surface, so a shadowed
        // fragment falls back to ambient rather than to black. A shadow that
        // reads as a hole in the world is the classic version of this bug.
        let shaded = shot!(Scene { caster: Some(ground(3.0, 8.0)), ..base() });
        let ambient_only = shot!(Scene { lights: vec![], ..base() });
        for c in 0..3 {
            assert!(
                (shaded[c] as i32 - ambient_only[c] as i32).abs() <= 3,
                "a shadowed fragment is not at ambient: {shaded:?} vs {ambient_only:?}",
            );
        }
    }

    #[test]
    fn geometry_outside_the_shadow_map_is_lit_rather_than_black() {
        // The `valid` guard in `shadow_coords`. Without it everything beyond the
        // map's reach reads as fully shadowed, and a level shows a hard line
        // across the ground at the edge of the shadow box -- far more
        // objectionable than a distant shadow simply going missing.
        // SIDEWAYS out of the map, not past its far plane: same height, same
        // distance from everything, 300m off the side of a box 20m across. That
        // is the `uv.x < 0 || uv.x > 1` branch specifically, which the distance
        // test above cannot reach.
        let inside = shot!(base());
        let beside = shot!(Scene { receiver_at: Vec3::new(300.0, 0.0, 0.0), ..base() });
        for c in 0..3 {
            assert!(
                (inside[c] as i32 - beside[c] as i32).abs() <= 2,
                "off the side of the shadow map went dark: {inside:?} vs {beside:?}",
            );
        }
    }

    #[test]
    fn the_falloff_the_editor_predicts_is_the_falloff_the_shader_applies() {
        // THE OTHER HALF OF A CROSS-LANGUAGE PIN. The editor warns about a
        // light's intensity using its own copy of this curve
        // (scene_editor_web/frontend/src/lib/lightFalloff.js), and a warning
        // computed from the wrong curve is worse than none -- it was authoring
        // against a mismatched preview that put a light of intensity 500 in the
        // lobby in the first place.
        //
        // Neither side transcribes the other's code. Both are asserted against
        // the same NUMBERS, so a change to the shader's attenuation fails here
        // and a change to the editor's fails there.
        //
        // The eye goes far off to the side so the specular lobe contributes
        // essentially nothing and this measures diffuse falloff alone, which is
        // what the editor models.
        let at = |intensity: f32, distance: f32| {
            shade_receiver(Scene {
                lights: vec![Light {
                    mask_channel: None,
                    position: Vec3::new(0.0, distance, 0.0),
                    direction: Vec3::NEG_Y,
                    kind: LightKind::Point,
                    color: Color3(255, 255, 255, 255),
                    intensity,
                    range: 25.0,
                    cone_angle_deg: 180.0,
                    inner_cone_angle_deg: 0.0,
                }],
                eye_offset: Vec3::new(400.0, 0.5, 0.0),
                ..base()
            })
        };

        // ambient 0.6 + intensity * window^2 / max(d^2, LAMP_RADIUS^2), range 25.
        //
        // The 0.85 m case is the one that tells the curves apart: at 5 m the
        // old `d^2 + 1` falloff differed from the inverse square by 4%, inside
        // this test's tolerance, and it was at a wall spot's own distance from
        // its wall that the two parted by 2.4x.
        for (intensity, distance, expected) in
            [(1.0f32, 5.0f32, 0.6399f32), (3.0, 5.0, 0.7196), (10.0, 5.0, 0.9987), (0.2, 0.85, 0.8768)]
        {
            let px = match at(intensity, distance) {
                Some(px) => px,
                None => {
                    eprintln!("skipping: no GPU adapter available");
                    return;
                }
            };
            let got = px[0] as f32 / 255.0;
            assert!(
                (got - expected).abs() < 0.01,
                "intensity {intensity} at {distance}m: shader gave {got:.4}, the editor \
                 predicts {expected:.4}",
            );
        }
    }

    /// A MOVING OBJECT SHADOWS THE SKY'S SUN through its own map.
    ///
    /// The same caster over the same ground, once with the sun as the sky's
    /// (caster in the moving-objects map, no static map) and once with the
    /// caster absent: the first must be darker. This is the player's shadow on
    /// sunlit ground, which the sun baked into the brushes could never show.
    #[test]
    fn a_moving_caster_shadows_the_sky_sun_through_the_dynamic_map() {
        let shadowed = shade_receiver(Scene {
            caster: Some(ground(3.0, 8.0)),
            sky_sun_dynamic: true,
            ..base()
        });
        let Some(shadowed) = shadowed else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let open = shade_receiver(Scene { sky_sun_dynamic: true, ..base() }).unwrap();
        eprintln!("under a moving caster {shadowed:?}, in the open {open:?}");
        assert!(
            (open[1] as i32) - (shadowed[1] as i32) > 20,
            "a moving object cast no sun shadow: {shadowed:?} vs {open:?}",
        );
    }

    /// THE NEAR TILE IS WHAT THE MIDDLE OF THE BOX IS READ FROM, at twice the
    /// detail (`SUN_NEAR_ZOOM`). One straight shadow edge, on a texel corner,
    /// and the ground one sun-tile texel outside it: in the middle of the box,
    /// where the near tile holds it, and 15 m out, where only the sun tile
    /// does. Through the sun tile the kernel still reaches past the edge and
    /// takes a sixth of the sun; through the near tile, half as wide, it
    /// reaches nothing. Lit and shadowed ground read the same in both places,
    /// so the difference is the tile and not the place.
    #[test]
    fn the_near_tile_holds_the_middle_of_the_box_at_twice_the_detail() {
        // Everything from `x0 - 8` to `x0`, three metres up.
        let plate = |x0: f32| {
            let n = [0.0, 1.0, 0.0];
            let v = vec![
                vertex([x0 - 8.0, 3.0, -8.0], n),
                vertex([x0, 3.0, -8.0], n),
                vertex([x0, 3.0, 8.0], n),
                vertex([x0 - 8.0, 3.0, 8.0], n),
            ];
            (v, vec![0, 2, 1, 0, 3, 2])
        };
        // `shade_receiver`'s sun box is 20 m round the origin; its near tile 10.
        let texel = 2.0 * 20.0 / SUN_DYNAMIC_DIM as f32;
        // A dim sky under a bright sun, for a wide range without clipping.
        let at = |x0: f32, dx: f32, caster: bool| {
            shade_receiver(Scene {
                lights: vec![Light {
                    intensity: 0.5,
                    ..sun()
                }],
                caster: caster.then(|| plate(x0)),
                receiver_at: Vec3::new(x0 + dx, 0.0, 0.0),
                sky: SkyUpload::from(&crate::renderer::sky::SkyIrradiance::flat(0.1)),
                sky_sun_dynamic: true,
                ..base()
            })
        };
        let Some(near) = at(0.0, texel, true) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let wide = at(15.0, texel, true).unwrap();
        let (lit_near, lit_wide) = (
            at(0.0, texel, false).unwrap(),
            at(15.0, texel, false).unwrap(),
        );
        let (dark_near, dark_wide) = (at(0.0, -1.0, true).unwrap(), at(15.0, -1.0, true).unwrap());
        eprintln!(
            "a texel outside the edge: near tile {near:?}, sun tile {wide:?}; lit {lit_near:?} / {lit_wide:?}; \
             shadowed {dark_near:?} / {dark_wide:?}",
        );
        assert!(
            (lit_near[1] as i32 - lit_wide[1] as i32).abs() <= 1 && (dark_near[1] as i32 - dark_wide[1] as i32).abs() <= 1,
            "the two places are not lit alike: lit {lit_near:?} / {lit_wide:?}, shadowed {dark_near:?} / {dark_wide:?}",
        );
        let range = lit_wide[1] as i32 - dark_wide[1] as i32;
        assert!(
            range > 40,
            "too little sun to measure a penumbra by: {range}"
        );
        // A sixth of the range lost through the sun tile; none through the near.
        let lost_wide = lit_wide[1] as i32 - wide[1] as i32;
        let lost_near = lit_near[1] as i32 - near[1] as i32;
        assert!(
            lost_wide * 10 > range && lost_near * 4 < lost_wide,
            "the middle of the box was not read from the near tile: lost {lost_near} there and {lost_wide} \
             through the sun tile, of {range}",
        );
    }

    /// THE NEAR TILE STAYS SNAPPED: the world's origin is on a texel corner of
    /// the near tile wherever the player stands, as on the sun tile's
    /// (`lights::dynamic_sun_matrix`) -- so a moving shadow's edge does not
    /// crawl in the near tile either -- and the near tile is the sun tile's
    /// middle at twice the size, depth unchanged.
    #[test]
    fn the_near_tile_is_the_middle_of_the_sun_tile_snapped_to_its_own_grid() {
        let dir = Vec3::new(0.4, -0.75, 0.3).normalize();
        let n = SUN_DYNAMIC_DIM as f32;
        for head in [
            Vec3::new(0.0, 1.6, 0.0),
            Vec3::new(3.137, 1.7, -2.71),
            Vec3::new(-41.3, 1.55, 12.06),
        ] {
            let wide = crate::renderer::lights::dynamic_sun_matrix(dir, head);
            let near = sun_near_matrix(wide);
            let texel_of = |m: Mat4, p: Vec3| {
                let c = m.project_point3(p);
                (glam::Vec2::new(c.x, c.y) * 0.5 + glam::Vec2::splat(0.5)) * n
            };
            let o = texel_of(near, Vec3::ZERO);
            assert!(
                (o - o.round()).abs().max_element() < 2e-3,
                "the world's origin is off the near tile's grid at {o:?}, player at {head}",
            );
            let p = head + Vec3::new(0.37, -0.81, 0.52);
            let (w, nr) = (wide.project_point3(p), near.project_point3(p));
            assert!(
                (nr.x - 2.0 * w.x).abs() < 1e-5
                    && (nr.y - 2.0 * w.y).abs() < 1e-5
                    && (nr.z - w.z).abs() < 1e-6
            );
        }
    }

    #[test]
    fn the_skys_ambient_follows_the_surface_normal() {
        // WHAT NINE COEFFICIENTS BUY OVER ONE. The old ambient was a constant:
        // every surface received the same grey whichever way it faced, which is
        // why an unlit blockout reads as flat. A sky bright on one side must now
        // light the surfaces facing it more than those facing away, with no
        // light in the scene at all.
        let bright_half = {
            let mut pano = crate::renderer::sky::Panorama::solid([0.0, 0.0, 0.0], 64, 32);
            for y in 0..32 {
                for x in 0..32 {
                    let i = ((y * 64 + x) * 3) as usize;
                    pano.rgb[i] = 3.0;
                    pano.rgb[i + 1] = 3.0;
                    pano.rgb[i + 2] = 3.0;
                }
            }
            SkyUpload::from(&crate::renderer::sky::project_irradiance(&pano, 0.0, 1.0))
        };

        let toward = shot!(Scene {
            lights: vec![],
            normal: [-1.0, 0.0, 0.0],
            sky: bright_half.clone(),
            ..base()
        });
        let away = shot!(Scene {
            lights: vec![],
            normal: [1.0, 0.0, 0.0],
            sky: bright_half,
            ..base()
        });
        assert!(
            toward[0] as i32 - away[0] as i32 > 20,
            "the ambient did not follow the normal: facing the bright half gave \
             {toward:?}, facing away gave {away:?}",
        );
    }

    #[test]
    fn the_shader_evaluates_the_same_irradiance_the_cpu_does() {
        // THE CROSS-IMPLEMENTATION PIN, and the reason it exists: the
        // coefficients are projected in Rust and evaluated in WGSL, and those
        // are two separate transcriptions of the same nine constants. Every
        // other test here predicts the shader's output using the Rust side, so
        // if the two drift, they all keep passing and the headset renders
        // something else.
        //
        // Deliberately breaking the shader's l=1 band from 2/3 to 1.0 changed
        // nothing in this file until this test existed -- the directional test
        // below only asks which side is brighter, which a scaled band survives.
        let pano = {
            let mut p = crate::renderer::sky::Panorama::solid([0.05, 0.05, 0.05], 64, 32);
            // A bright quadrant, so bands 1 and 2 both carry real weight rather
            // than the constant term dominating and hiding a drift in them.
            for y in 4..20 {
                for x in 8..28 {
                    let i = ((y * 64 + x) * 3) as usize;
                    p.rgb[i] = 0.9;
                    p.rgb[i + 1] = 0.7;
                    p.rgb[i + 2] = 0.4;
                }
            }
            p
        };
        let irr = crate::renderer::sky::project_irradiance(&pano, 0.0, 1.0);
        let sky = SkyUpload::from(&irr);

        for normal in [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, -0.7],
        ] {
            let px = shot!(Scene {
                lights: vec![],
                normal,
                sky: sky.clone(),
                ..base()
            });
            let want = irr.evaluate(normal);
            for c in 0..3 {
                // The target is linear Rgba8Unorm, so the byte IS the value.
                let got = px[c] as f32 / 255.0;
                assert!(
                    (got - want[c]).abs() < 0.012,
                    "normal {normal:?} channel {c}: the shader rendered {got:.4} \
                     and the CPU predicts {:.4}",
                    want[c],
                );
            }
        }
    }

    #[test]
    fn a_scene_with_no_sky_is_lit_exactly_as_before() {
        // The compatibility claim, at the pixel. `SkyUpload::none()` has to
        // evaluate to the old flat constant in every direction, or adding this
        // feature silently re-lights every level that does not use one.
        let up = shot!(Scene { lights: vec![], normal: [0.0, 1.0, 0.0], ..base() });
        let side = shot!(Scene { lights: vec![], normal: [1.0, 0.0, 0.0], ..base() });
        let down = shot!(Scene { lights: vec![], normal: [0.0, -1.0, 0.0], ..base() });
        for (a, b) in [(up, side), (up, down)] {
            for c in 0..3 {
                assert!(
                    (a[c] as i32 - b[c] as i32).abs() <= 1,
                    "flat ambient is no longer flat: {a:?} vs {b:?}",
                );
            }
        }
        let ambient = (crate::renderer::sky::AMBIENT * 255.0).round() as i32;
        assert!(
            (up[0] as i32 - ambient).abs() <= 2,
            "ambient came out {} rather than {ambient}",
            up[0],
        );
    }

    #[test]
    fn specular_depends_on_where_the_eye_is() {
        // Blinn-Phong's whole contribution. With a sun straight down on a
        // surface facing up, the highlight is directly overhead: an eye there
        // sees it and an eye far off to the side does not. If the camera
        // position never reaches the shader this is the test that notices.
        let overhead = shot!(Scene { eye_offset: Vec3::new(0.0, 5.0, 0.0), ..base() });
        let oblique = shot!(Scene { eye_offset: Vec3::new(50.0, 0.5, 0.0), ..base() });
        assert!(
            overhead[0] as i32 - oblique[0] as i32 > 4,
            "the specular highlight did not follow the eye: {overhead:?} vs {oblique:?}",
        );
    }

    // HANGING LAMP TESTS
    // `test_room`'s hanging lamp, reproduced through the shadow pass.
    //
    // Both rooms rendered black while the lamps visibly glowed. This puts the
    // REAL fixture geometry into the shadow map, at the real socket offset,
    // under the real light values, and reads the floor beneath it -- so the
    // question "does the fixture shadow its own bulb" is answered by a
    // rendered pixel rather than by argument.
    //
    // Skipped when the model is absent; the crate builds without `game/`.
    fn lamp_path() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../game/models/lights/hanging_industrial_lamp/hanging_industrial_lamp_1k.gltf");
        p.is_file().then_some(p)
    }

    /// The fixture as shadow-caster geometry, placed where the scene puts it.
    ///
    /// `include_transmissive` picks which rule is under test: false is what the
    /// renderer does now, true is what it did when the rooms were black.
    fn lamp_caster(include_transmissive: bool) -> Option<(Vec<SolidVertex>, Vec<u32>)> {
        let (doc, buffers, _) = gltf::import(lamp_path()?).ok()?;
        let origin = Vec3::new(0.0, 3.1, -4.5);
        let mut verts: Vec<SolidVertex> = Vec::new();
        let mut idx: Vec<u32> = Vec::new();
        for mesh in doc.meshes() {
            for prim in mesh.primitives() {
                let transmissive = prim
                    .material()
                    .transmission()
                    .map(|t| t.transmission_factor())
                    .unwrap_or(0.0)
                    > 0.5;
                if transmissive && !include_transmissive {
                    continue;
                }
                let reader = prim.reader(|b| Some(&buffers[b.index()]));
                let base = verts.len() as u32;
                let Some(pos) = reader.read_positions() else { continue };
                for p in pos {
                    verts.push(vertex(
                        [p[0] + origin.x, p[1] + origin.y, p[2] + origin.z],
                        [0.0, 1.0, 0.0],
                    ));
                }
                match reader.read_indices() {
                    Some(i) => idx.extend(i.into_u32().map(|i| i + base)),
                    None => idx.extend(base..verts.len() as u32),
                }
            }
        }
        (!idx.is_empty()).then_some((verts, idx))
    }

    /// `hall_spot_1`, at the socket the fixture declares: 1.18m below the
    /// object's origin, aimed straight down.
    fn hall_spot_1() -> Light {
        Light {
            mask_channel: None,
            position: Vec3::new(0.0, 3.1 - 1.18, -4.5),
            direction: Vec3::NEG_Y,
            kind: LightKind::Spot,
            color: Color3(255, 244, 214, 255),
            // The scene authors 9.0. Dialled down here purely for HEADROOM:
            // at 9 the floor 1.9m under the bulb saturates at 255 and a shadow
            // that halves the light is still 255, so every condition measures
            // identical and the test proves nothing. The shadow term is a
            // multiplier, so the ratio this measures is the same either way.
            intensity: 1.0,
            range: 14.0,
            cone_angle_deg: 64.0,
            inner_cone_angle_deg: 30.0,
        }
    }

    fn floor_under_the_lamp() -> Scene {
        Scene {
            lights: vec![hall_spot_1()],
            caster: None,
            receiver_at: Vec3::new(0.0, 0.0, -4.5),
            eye_offset: Vec3::new(0.0, 1.6, 1.0),
            shadows_on: true,
            shadow_dim: 256,
            normal: [0.0, 1.0, 0.0],
            sky: crate::renderer::uniforms::SkyUpload::none(),
            sky_sun_dynamic: false,
            character_lamp: None,
        }
    }

    /// A lamp must not shadow its own bulb -- and today it cannot, because the
    /// shadow frustum's near plane is farther away than the fixture is big.
    ///
    /// WHAT THIS TEST IS FOR
    ///
    /// It was written to confirm a diagnosis that turned out to be WRONG. Both
    /// rooms in `test_room` rendered black, and the fixture's own geometry
    /// blocks 100% of the bulb's downward rays (measured in
    /// `space_soup_engine::mesh_lightmap`), so the fixture appeared to be
    /// shadowing its own light. It is not: `spot_light_matrix` puts the near
    /// plane at `range * 0.02`, which for a 14m lamp is 0.28m, and every part
    /// of the envelope is within 0.25m of the bulb. The fixture is clipped out
    /// of its own shadow map before it can block anything.
    ///
    /// So this guards a real invariant that currently holds by accident. Drop
    /// the near plane -- a reasonable thing to want for close-range shadow
    /// detail -- and every lamp in the game seals itself into darkness.

    #[test]
    fn diag_self_shadow_acne() {
        // The receiver's OWN surface in the shadow map, which is what a brush
        // floor is on the device and what the rig never had.
        let open = shot!(Scene { ..floor_under_the_lamp() });
        let itself = shot!(Scene { caster: Some(ground(0.0, 8.0)), ..floor_under_the_lamp() });
        let slightly_below =
            shot!(Scene { caster: Some(ground(-0.01, 8.0)), ..floor_under_the_lamp() });
        eprintln!(
            "ACNE: no caster {open:?}  floor casts itself {itself:?}  floor 1cm below {slightly_below:?}"
        );
    }

    #[test]
    fn a_lamp_does_not_shadow_its_own_bulb() {
        let Some(with_glass) = lamp_caster(true) else { return };
        let Some(without_glass) = lamp_caster(false) else { return };

        let open = shot!(Scene { ..floor_under_the_lamp() });
        let full = shot!(Scene { caster: Some(with_glass), ..floor_under_the_lamp() });
        let exempt = shot!(Scene { caster: Some(without_glass), ..floor_under_the_lamp() });

        eprintln!("LAMP FLOOR: no caster {open:?}  whole fixture {full:?}  glass exempt {exempt:?}");
        assert!(
            open[0] > 40,
            "the rig itself does not light the floor ({open:?}); nothing below this means anything",
        );
        assert!(
            (open[0] as i32) - (full[0] as i32) < 16,
            "the fixture is shadowing its own bulb: floor at {full:?} against {open:?} with \
             nothing casting. Check the spot shadow near plane against the size of the \
             fixture -- if the near plane has come inside the housing, every lamp in the \
             game now blacks out the floor beneath it.",
        );
        assert_eq!(
            full, exempt,
            "the glass envelope changed the result, so it IS reaching the shadow map now \
             and the near-plane reasoning above no longer holds",
        );
    }

}


#[cfg(test)]
mod skinned_caster_tests {
    //! A character has to cast. It is the first missing shadow anyone notices,
    //! because it is the only caster a player can pick up and wave about.

    #[test]
    fn the_shadow_pass_poses_skinned_casters() {
        let src = super::SHADOW_SHADER;
        assert!(src.contains("fn vs_skinned"), "no skinned caster stage");
        assert!(
            src.contains("joints.mats[joint_ids.x]"),
            "a skinned caster must be posed, not drawn in bind pose",
        );
    }

    #[test]
    fn the_skinning_matches_the_lit_pass_exactly() {
        // Two copies of this arithmetic exist: one draws what you see, one
        // draws its shadow. If they differ the shadow is cast by a slightly
        // differently posed body, which reads as the shadow lagging or sliding
        // rather than as a skinning bug -- and nothing errors.
        let shadow = super::SHADOW_SHADER;
        let lit = include_str!("mesh_pipeline.rs");
        for term in [
            "(joints.mats[joint_ids.x] * p) * joint_weights.x",
            "(joints.mats[joint_ids.y] * p) * joint_weights.y",
            "(joints.mats[joint_ids.z] * p) * joint_weights.z",
            "(joints.mats[joint_ids.w] * p) * joint_weights.w",
        ] {
            assert!(shadow.contains(term), "shadow pass is missing: {term}");
        }
        // The lit pass writes the same sum with `v.` prefixes on its inputs.
        for term in [
            "(joints.mats[v.joint_ids.x] * p) * v.joint_weights.x",
            "(joints.mats[v.joint_ids.w] * p) * v.joint_weights.w",
        ] {
            assert!(lit.contains(term), "lit pass changed shape: {term}");
        }
    }

    #[test]
    fn the_joint_count_actually_reaches_the_shader() {
        // The source carries a placeholder; forgetting to substitute it leaves
        // an identifier WGSL cannot resolve, and the pipeline fails to build at
        // runtime rather than at compile time.
        assert!(
            super::SHADOW_SHADER.contains("MAX_SKIN_JOINTS_PLACEHOLDER"),
            "the placeholder is what the module substitutes",
        );
        let built = super::SHADOW_SHADER
            .replace("MAX_SKIN_JOINTS_PLACEHOLDER", &super::super::mesh::MAX_SKIN_JOINTS.to_string());
        assert!(!built.contains("PLACEHOLDER"), "substitution left a placeholder behind");
    }

}

/// Whether a chunk is worth drawing, given every frustum the pass covers.
///
/// EITHER, not both. A multiview scene pass draws both eyes together, so a
/// chunk visible to one eye and not the other must still be drawn -- culling
/// on a single frustum there produces a hole in ONE EYE, which is unpleasant
/// to look at, trivially easy to introduce, and invisible on a monitor or in
/// any screenshot taken from one eye.
///
/// A helper rather than an inline `any` because the XR renderer is
/// `#[cfg(target_os = "android")]`: a rule written there is a rule no test on
/// a development machine ever runs. Same reason `scene_pass_plan` exists.
pub fn chunk_seen(chunk: &CasterChunk, frusta: &[[glam::Vec4; 6]]) -> bool {
    frusta.iter().any(|planes| aabb_in_frustum(planes, chunk.min, chunk.max))
}

#[cfg(test)]
mod chunk_culling_tests {
    use super::{chunk_seen, frustum_planes, CasterChunk};
    use glam::{Mat4, Vec3};

    fn chunk(min: Vec3, max: Vec3) -> CasterChunk {
        CasterChunk { first_index: 0, index_count: 3, min, max }
    }

    /// Two eyes looking slightly apart, as a stereo pass has.
    fn eye(x_offset: f32) -> [glam::Vec4; 6] {
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let view = Mat4::look_at_rh(
            Vec3::new(x_offset, 0.0, 0.0),
            Vec3::new(x_offset, 0.0, -1.0),
            Vec3::Y,
        );
        frustum_planes(proj * view)
    }

    #[test]
    fn a_chunk_in_front_of_both_eyes_is_drawn() {
        let c = chunk(Vec3::new(-1.0, -1.0, -6.0), Vec3::new(1.0, 1.0, -4.0));
        assert!(chunk_seen(&c, &[eye(-0.03), eye(0.03)]));
    }

    #[test]
    fn a_chunk_far_behind_the_viewer_is_culled() {
        let c = chunk(Vec3::new(-1.0, -1.0, 40.0), Vec3::new(1.0, 1.0, 42.0));
        assert!(!chunk_seen(&c, &[eye(-0.03), eye(0.03)]));
    }

    /// THE STEREO BUG THIS EXISTS TO PREVENT. Build a case one eye can see and
    /// the other cannot, and require that the pass still draws it.
    #[test]
    fn a_chunk_only_one_eye_can_see_is_still_drawn() {
        // Eyes a long way apart so their frusta genuinely differ, and a chunk
        // placed out to one side.
        let (left, right) = (eye(-4.0), eye(4.0));
        let c = chunk(Vec3::new(-9.0, -1.0, -6.0), Vec3::new(-7.0, 1.0, -4.0));
        let in_left = super::aabb_in_frustum(&left, c.min, c.max);
        let in_right = super::aabb_in_frustum(&right, c.min, c.max);
        assert!(
            in_left != in_right,
            "the fixture no longer isolates one eye (left={in_left}, right={in_right}); \
             it cannot test the rule it exists for",
        );
        assert!(
            chunk_seen(&c, &[left, right]),
            "a chunk one eye can see was culled -- that is a hole in one eye",
        );
    }

    #[test]
    fn no_frusta_means_nothing_is_drawn_rather_than_everything() {
        // An empty list is a caller bug, and drawing the level twice over is a
        // worse response to it than drawing nothing and being noticed.
        let c = chunk(Vec3::new(-1.0, -1.0, -6.0), Vec3::new(1.0, 1.0, -4.0));
        assert!(!chunk_seen(&c, &[]));
    }
}

#[cfg(test)]
mod spot_near_plane_tests {
    use super::{spot_light_matrix, SPOT_SHADOW_NEAR};
    use glam::{Vec3, Vec4};

    /// Where the shadow map starts, recovered from the projection rather than
    /// recomputed -- so this tests the matrix that ships, not a copy of the
    /// arithmetic that built it.
    fn near_of(range: f32) -> f32 {
        let m = spot_light_matrix(Vec3::ZERO, Vec3::NEG_Z, 60.0, range);
        // A point exactly on the near plane maps to NDC z = 0 under this
        // projection; bisect for it along the light's axis.
        let (mut lo, mut hi) = (0.001f32, range.max(0.2));
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let c: Vec4 = m * Vec4::new(0.0, 0.0, -mid, 1.0);
            if c.w <= 0.0 || c.z / c.w < 0.0 { lo = mid } else { hi = mid }
        }
        0.5 * (lo + hi)
    }

    /// THE BUG THIS EXISTS FOR. The near plane was `far * 0.02`, so it tracked
    /// the light's RANGE -- a number about how far the light travels, which
    /// says nothing about the size of the fixture around the bulb. Five lights
    /// in test_room got five different near planes (0.08 m to 0.32 m), and two
    /// hanging fixtures of the same design behaved differently: one shadowed
    /// its own shade onto the floor below, the other did not.
    #[test]
    fn the_near_plane_does_not_depend_on_how_far_the_light_reaches() {
        // The actual ranges in test_room, which is where this was observed.
        let ranges = [4.0f32, 8.0, 12.0, 14.0, 16.0];
        let nears: Vec<f32> = ranges.iter().map(|r| near_of(*r)).collect();
        let first = nears[0];
        for (r, n) in ranges.iter().zip(nears.iter()) {
            assert!(
                (n - first).abs() < 0.01,
                "range {r} m put the near plane at {n:.3} m while range {} m put it at \
                 {first:.3} m -- two fixtures of the same design will disagree about \
                 whether they shadow themselves",
                ranges[0],
            );
        }
    }

    #[test]
    fn the_near_plane_clears_a_hanging_fixtures_housing() {
        assert!(
            SPOT_SHADOW_NEAR >= 0.25,
            "the near plane no longer clears a lamp housing; fixtures will start \
             casting a dark dot into the middle of their own pool of light",
        );
        assert!(
            (near_of(12.0) - SPOT_SHADOW_NEAR).abs() < 0.01,
            "an ordinary room light does not get the constant near plane",
        );
    }

    /// A short-range light must not end up with its near plane past its far
    /// plane -- that is an inside-out projection, not a small shadow.
    #[test]
    fn a_very_short_range_light_still_has_a_sane_projection() {
        for range in [0.05f32, 0.2, 0.4, 0.6] {
            let n = near_of(range);
            let far = range.max(0.2);
            assert!(
                n > 0.0 && n < far,
                "range {range} m produced near {n:.4} against far {far:.4}",
            );
        }
    }
}
