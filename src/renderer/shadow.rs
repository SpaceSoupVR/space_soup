use glam::{Mat4, Vec2, Vec3, Vec4};
use wgpu::*;

use super::cuboid::SolidVertex;
use super::lights::Light;
use super::mesh::MeshVertex;

pub const SHADOW_DIM: u32 = 2048;

pub const QUEST_SHADOW_DIM: u32 = 1024;

pub const SUN_DYNAMIC_DIM: u32 = 512;

pub const SUN_DYNAMIC_RADIUS: f32 = 9.0;

pub const MAX_SPOT_SHADOWS: usize = 4;

pub const SPOT_ATLAS_COLS: u32 = 2;

pub const MAX_CHARACTER_SHADOWS: usize = 4;

pub const MAX_BODY_TILES: usize = 2;

pub const SUN_ATLAS_TILES: u32 = 2 + MAX_CHARACTER_SHADOWS as u32;

pub const SUN_NEAR_ZOOM: f32 = 6.0;

pub const SUN_NEAR_TILE: u32 = 1 + MAX_CHARACTER_SHADOWS as u32;

pub fn sun_near_matrix(sun_dynamic: Mat4) -> Mat4 {
    Mat4::from_scale(Vec3::new(SUN_NEAR_ZOOM, SUN_NEAR_ZOOM, 1.0)) * sun_dynamic
}

pub const SHADOW_MATRICES: usize = MAX_SPOT_SHADOWS + MAX_CHARACTER_SHADOWS + 2;

pub const SPOT_ATLAS_ROWS: u32 = (MAX_SPOT_SHADOWS as u32).div_ceil(SPOT_ATLAS_COLS);

const _: () = assert!(
    (SPOT_ATLAS_COLS * SPOT_ATLAS_ROWS) as usize >= MAX_SPOT_SHADOWS,
    "the spot atlas must have a tile for every spot the budget allows",
);

pub fn spot_tile(layer: usize) -> (u32, u32) {
    let l = layer as u32;
    (l % SPOT_ATLAS_COLS, l / SPOT_ATLAS_COLS)
}

#[derive(Clone, Copy, Debug)]
pub struct CharacterLamp {
    pub position: Vec3,

    pub direction: Vec3,
    pub cos_outer: f32,
    pub range: f32,
    pub intensity: f32,

    pub eligible: bool,
}

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
    let mut chosen: Vec<(usize, f32)> = candidates.iter().copied().filter(|c| was_held(c.0)).take(MAX_BODY_TILES).collect();
    for c in &candidates {
        if chosen.iter().any(|k| k.0 == c.0) {
            continue;
        }
        if chosen.len() < MAX_BODY_TILES {
            chosen.push(*c);
            continue;
        }

        let (weakest, w) = chosen.iter().enumerate().min_by(|a, b| a.1 .1.total_cmp(&b.1 .1)).map(|(k, c)| (k, c.1)).unwrap();
        if w < c.1 * KEEP {
            chosen[weakest] = *c;
        }
    }
    chosen.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    chosen.into_iter().map(|c| c.0).collect()
}

#[derive(Clone, Copy, Debug)]
pub struct DoorCaster {
    pub corners: [Vec3; 8],
}

impl DoorCaster {
    fn centre(&self) -> Vec3 {
        self.corners.iter().fold(Vec3::ZERO, |s, c| s + *c) / 8.0
    }
    fn radius(&self) -> f32 {
        let c = self.centre();
        self.corners.iter().map(|p| (*p - c).length()).fold(0.0, f32::max)
    }
}

pub const DOOR_SHADOW_REACH: f32 = 12.0;

const DOOR_SHADOW_FALLOFF: f32 = 4.0;

const DOOR_CORNER_RADIUS: f32 = 0.02;

const MOVING_TILE_SPREAD_COS: f32 = 0.643;

pub fn moving_caster_tiles(
    lamps: &[CharacterLamp],
    body: Option<(Vec3, f32, &[(Vec3, f32)])>,
    doors: &[DoorCaster],
    eye: Vec3,
    held: &[Vec3],
) -> Vec<(usize, Vec<(Vec3, f32)>)> {
    const KEEP: f32 = 1.0 / 1.33;
    let reaches = |l: &CharacterLamp, centre: Vec3, radius: f32| {
        let to = centre - l.position;
        let d = to.length();
        if !l.eligible || l.intensity <= 0.0 || d - radius >= l.range {
            return false;
        }
        if l.cos_outer <= -1.0 || d <= radius {
            return true;
        }
        let angle = l.direction.normalize_or_zero().dot(to / d).clamp(-1.0, 1.0).acos();
        angle <= l.cos_outer.clamp(-1.0, 1.0).acos() + (radius / d).min(1.0).asin()
    };
    let body_lamps: Vec<usize> = match body {
        Some((centre, radius, _)) => character_shadow_lamps(lamps, centre, radius, held),
        None => Vec::new(),
    };
    let near: Vec<&DoorCaster> = doors.iter().filter(|d| (d.centre() - eye).length() < DOOR_SHADOW_REACH).collect();
    let door_claim = |i: usize| -> f32 {
        let l = &lamps[i];
        near.iter()
            .filter(|d| reaches(l, d.centre(), d.radius()))
            .map(|d| {
                let seen = (d.centre() - eye).length() / DOOR_SHADOW_FALLOFF;
                l.intensity / ((l.position - d.centre()).length_squared() + 0.25) / (1.0 + seen * seen)
            })
            .fold(0.0, f32::max)
    };
    let mut chosen: Vec<usize> = body_lamps.clone();
    let mut contenders: Vec<(usize, f32)> = (0..lamps.len())
        .filter(|i| !chosen.contains(i))
        .map(|i| (i, door_claim(i)))
        .filter(|c| c.1 > 0.0)
        .collect();
    contenders.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let was_held = |i: usize| held.iter().any(|h| (lamps[i].position - *h).length() < 0.01);
    let room = MAX_CHARACTER_SHADOWS - chosen.len().min(MAX_CHARACTER_SHADOWS);
    let mut doors_chosen: Vec<(usize, f32)> = contenders.iter().copied().filter(|c| was_held(c.0)).take(room).collect();
    for c in &contenders {
        if doors_chosen.iter().any(|k| k.0 == c.0) {
            continue;
        }
        if doors_chosen.len() < room {
            doors_chosen.push(*c);
            continue;
        }
        let Some((weakest, w)) = doors_chosen.iter().enumerate().min_by(|a, b| a.1 .1.total_cmp(&b.1 .1)).map(|(k, c)| (k, c.1)) else {
            break;
        };
        if w < c.1 * KEEP {
            doors_chosen[weakest] = *c;
        }
    }
    doors_chosen.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    chosen.extend(doors_chosen.into_iter().map(|c| c.0));
    chosen
        .into_iter()
        .map(|i| {
            let l = &lamps[i];
            let mut spheres: Vec<(Vec3, f32)> = Vec::new();
            let body_here = body.filter(|(centre, radius, _)| reaches(l, *centre, *radius));
            if let Some((_, _, parts)) = body_here {
                spheres.extend_from_slice(parts);
            }

            let mut doors_here: Vec<(&DoorCaster, f32)> = near
                .iter()
                .filter(|d| reaches(l, d.centre(), d.radius()))
                .map(|d| (*d, l.intensity / ((l.position - d.centre()).length_squared() + 0.25)))
                .collect();
            doors_here.sort_by(|a, b| b.1.total_cmp(&a.1));
            let aim = body_here.map(|(c, _, _)| c).or(doors_here.first().map(|(d, _)| d.centre()));
            let toward = |p: Vec3| (p - l.position).normalize_or_zero();
            for (d, _) in doors_here {
                if aim.is_some_and(|a| toward(a).dot(toward(d.centre())) >= MOVING_TILE_SPREAD_COS) {
                    spheres.extend(d.corners.iter().map(|c| (*c, DOOR_CORNER_RADIUS)));
                }
            }
            (i, spheres)
        })
        .filter(|(_, s)| !s.is_empty())
        .collect()
}

pub const CHARACTER_TILE_MARGIN: f32 = 0.06;

const CHARACTER_TILE_WIDEST_TAN: f32 = 3.73;

pub fn character_light_matrix(light: Vec3, spot: Option<(Vec3, f32)>, body: &[(Vec3, f32)], range: f32) -> Option<Mat4> {
    if body.is_empty() {
        return None;
    }

    let middle = body.iter().fold(Vec3::ZERO, |s, (c, _)| s + *c) / body.len() as f32;
    let axis = match spot {
        Some((d, _)) => d.normalize_or_zero(),
        None => (middle - light).normalize_or_zero(),
    };
    let axis = if axis == Vec3::ZERO { Vec3::NEG_Y } else { axis };
    let up = if axis.dot(Vec3::Y).abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let view = Mat4::look_at_rh(light, light + axis, up);

    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    let (mut open_lo, mut open_hi) = ([false; 2], [false; 2]);
    let mut near = f32::MAX;
    let mut deepest = 0.0f32;
    let mut held = 0;
    for &(c, r) in body {
        let r = r + CHARACTER_TILE_MARGIN;
        let p = view.transform_point3(c);
        let d = -p.z;

        if d + r <= 0.0 {
            continue;
        }
        held += 1;
        deepest = deepest.max(d + r);
        near = near.min(d - r);
        for (k, x) in [p.x, p.y].into_iter().enumerate() {

            let dist = (x * x + d * d).sqrt();
            if d <= 0.0 || dist <= r {
                open_lo[k] = true;
                open_hi[k] = true;
                continue;
            }
            let (centre, spread) = (x.atan2(d), (r / dist).asin());
            let (a, b) = (centre - spread, centre + spread);
            const SQUARE: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
            if a <= -SQUARE {
                open_lo[k] = true;
            } else {
                lo[k] = lo[k].min(a.tan());
            }
            if b >= SQUARE {
                open_hi[k] = true;
            } else {
                hi[k] = hi[k].max(b.tan());
            }
        }
    }
    if held == 0 {
        return None;
    }

    let widest = match spot {
        Some((_, cos_outer)) => {
            let c = cos_outer.clamp(-1.0, 1.0);
            if c <= 0.05 {
                CHARACTER_TILE_WIDEST_TAN
            } else {
                ((1.0 - c * c).sqrt() / c).min(CHARACTER_TILE_WIDEST_TAN)
            }
        }
        None => CHARACTER_TILE_WIDEST_TAN,
    };
    for k in 0..2 {
        lo[k] = if open_lo[k] { -widest } else { lo[k].max(-widest) };
        hi[k] = if open_hi[k] { widest } else { hi[k].min(widest) };
        if lo[k] >= hi[k] {

            return None;
        }

        let pad = (hi[k] - lo[k]) * (2.0 / SUN_DYNAMIC_DIM as f32);
        lo[k] -= pad;
        hi[k] += pad;
    }
    let near = near.clamp(0.02, 1.0);
    let far = range.max(deepest).max(near * 2.0);
    let depth = far / (near - far);
    let projection = Mat4::from_cols(
        Vec4::new(2.0 / (hi.x - lo.x), 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / (hi.y - lo.y), 0.0, 0.0),
        Vec4::new((hi.x + lo.x) / (hi.x - lo.x), (hi.y + lo.y) / (hi.y - lo.y), depth, -1.0),
        Vec4::new(0.0, 0.0, depth * near, 0.0),
    );
    Some(projection * view)
}

fn sun_atlas_viewport(pass: &mut wgpu::RenderPass, tile: u32) {
    let d = SUN_DYNAMIC_DIM as f32;
    pass.set_viewport(tile as f32 * d, 0.0, d, d, 0.0, 1.0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowKind {
    Sun,

    SunDynamic,

    SunNear,

    Spot(usize),

    Character(usize),
}

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

    let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.0, dist + radius);
    proj * view
}

pub fn frustum_planes(view_proj: Mat4) -> [glam::Vec4; 6] {
    let m = view_proj.to_cols_array_2d();

    let row = |r: usize| glam::Vec4::new(m[0][r], m[1][r], m[2][r], m[3][r]);
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    let norm = |p: glam::Vec4| {
        let len = p.truncate().length();
        if len > 1e-9 { p / len } else { p }
    };
    [
        norm(r3 + r0),
        norm(r3 - r0),
        norm(r3 + r1),
        norm(r3 - r1),
        norm(r3 + r2),
        norm(r3 - r2),
    ]
}

pub fn aabb_in_frustum(planes: &[glam::Vec4; 6], min: Vec3, max: Vec3) -> bool {
    for p in planes {

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

#[derive(Debug, Clone, Copy)]
pub struct CasterChunk {
    pub first_index: u32,
    pub index_count: u32,
    pub min: Vec3,
    pub max: Vec3,
}

pub const SPOT_SHADOW_NEAR: f32 = 0.30;

pub fn spot_light_matrix(pos: Vec3, dir: Vec3, cone_angle_deg: f32, range: f32) -> Mat4 {
    spot_light_matrix_near(pos, dir, cone_angle_deg, range, SPOT_SHADOW_NEAR)
}

pub const SHADER_DEPTH_BIAS: f32 = 0.0015;

pub const SPOT_BIAS_TEXELS: f32 = 1.5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpotShadowMatrices {
    pub pass: Mat4,
    pub lookup: Mat4,
}

pub fn spot_shadow_matrices(l: &Light, tile_dim: u32) -> SpotShadowMatrices {
    let Some(wanted) = l.shadow_near else {
        let m = spot_light_matrix(l.position, l.direction, l.cone_angle_deg, l.range);
        return SpotShadowMatrices { pass: m, lookup: m };
    };
    let pass = spot_light_matrix_near(l.position, l.direction, l.cone_angle_deg, l.range, wanted);
    let (fov, near, far) = spot_frustum(l.cone_angle_deg, l.range, wanted);
    let a = far / (far - near);
    let c = SPOT_BIAS_TEXELS * 2.0 * (fov * 0.5).tan() / tile_dim.max(1) as f32;
    let mut lookup = pass;
    for col in [&mut lookup.x_axis, &mut lookup.y_axis, &mut lookup.z_axis, &mut lookup.w_axis] {
        col.z = (1.0 + c) * col.z + (SHADER_DEPTH_BIAS - c * a) * col.w;
    }
    SpotShadowMatrices { pass, lookup }
}

fn spot_frustum(cone_angle_deg: f32, range: f32, near_wanted: f32) -> (f32, f32, f32) {

    let fov = (cone_angle_deg.to_radians() * 1.1).clamp(0.1, std::f32::consts::PI - 0.1);
    let far = range.max(0.2);

    let near = near_wanted.min(far * 0.5).max(0.02);
    (fov, near, far)
}

pub fn spot_light_matrix_near(pos: Vec3, dir: Vec3, cone_angle_deg: f32, range: f32, near_wanted: f32) -> Mat4 {
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

    let (fov, near, far) = spot_frustum(cone_angle_deg, range, near_wanted);
    let proj = Mat4::perspective_rh(fov, 1.0, near, far);
    proj * view
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightMatrix {
    view_proj: [[f32; 4]; 4],
}

pub type ShadowMeshDraw<'a> = (&'a Buffer, &'a Buffer, u32, &'a BindGroup);

pub type ShadowMeshBound = glam::Vec4;

pub fn sphere_in_frustum(planes: &[glam::Vec4; 6], sphere: ShadowMeshBound) -> bool {
    planes.iter().all(|p| p.truncate().dot(sphere.truncate()) + p.w >= -sphere.w)
}

fn mesh_caster_reaches(planes: &[glam::Vec4; 6], bounds: &[ShadowMeshBound], i: usize) -> bool {
    bounds.get(i).is_none_or(|&s| sphere_in_frustum(planes, s))
}

pub fn mesh_casters_reaching(
    planes: &[glam::Vec4; 6],
    draws: &[ShadowMeshDraw],
    bounds: &[ShadowMeshBound],
) -> (usize, u32) {
    draws
        .iter()
        .enumerate()
        .filter(|(i, _)| mesh_caster_reaches(planes, bounds, *i))
        .fold((0, 0), |(n, indices), (_, d)| (n + 1, indices + d.2))
}

pub fn mesh_caster_bound(mesh: &crate::renderer::mesh::GltfMesh) -> ShadowMeshBound {
    mesh.position.extend(mesh.bounding_radius * mesh.scale.abs().max_element())
}

pub type ShadowSkinnedDraw<'a> = (&'a Buffer, &'a Buffer, u32, &'a BindGroup, &'a BindGroup);

struct ShadowSlot {

    _depth_texture: Option<Texture>,
    depth_view: Option<TextureView>,
    light_buffer: Buffer,
    light_bind_group: BindGroup,
}

impl ShadowSlot {
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
    /// The static sun's second map (layer 1 of `sun`'s texture), drawn
    /// ahead of the first while the sun glides (`lights::SunPair`), and
    /// which of the two `ShadowKind::Sun` draws into.
    sun_next: ShadowSlot,
    sun_layer: std::cell::Cell<u32>,
    sun_array_view: TextureView,

    sun_dynamic: ShadowSlot,

    spots: [ShadowSlot; MAX_SPOT_SHADOWS],

    characters: [ShadowSlot; MAX_CHARACTER_SHADOWS],

    sun_near: ShadowSlot,
    _spot_texture: Texture,

    spot_array_view: TextureView,

    spot_tile_dim: u32,

    tile_scissor: std::cell::Cell<bool>,
    sampler: Sampler,
    solid_pipeline: RenderPipeline,

    solid_world_pipeline: RenderPipeline,

    solid_world: std::cell::RefCell<Option<(BindGroup, bool)>>,
    mesh_pipeline: RenderPipeline,
    skinned_pipeline: RenderPipeline,

    posed_pipeline: RenderPipeline,

    brush_pipeline: RenderPipeline,

    brush_world_pipeline: RenderPipeline,

    brush_world: std::cell::RefCell<Option<BindGroup>>,
}

impl ShadowMap {

    pub fn new(device: &Device) -> Self {
        Self::with_dimension(device, SHADOW_DIM)
    }

    pub fn with_dimension(device: &Device, dim: u32) -> Self {
        let model_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("shadow_model_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,

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

        let solid_world_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("shadow_solid_world_pipeline"),
            layout: Some(&mesh_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_mesh"),
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

        let posed_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("shadow_posed_pipeline"),
            layout: Some(&mesh_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_posed"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(super::skin_compute::posed_layout())],
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

        let brush_world_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("shadow_brush_world_pipeline"),
            layout: Some(&mesh_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_mesh"),
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

        let sun = ShadowSlot::new_layered(device, &light_layout, "sun_shadow_depth", dim, 2, 0);
        let sun_texture = sun._depth_texture.clone().expect("the sun owns its own depth target");
        let sun_next = ShadowSlot::from_texture(device, &light_layout, &sun_texture, 1);
        let sun_array_view = sun_texture.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() });

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

        let spots = std::array::from_fn(|_| ShadowSlot::light_only(device, &light_layout));
        let characters = std::array::from_fn(|_| ShadowSlot::light_only(device, &light_layout));
        let sun_near = ShadowSlot::light_only(device, &light_layout);

        let sun_dynamic = ShadowSlot::new_sized(
            device,
            &light_layout,
            "sun_dynamic_shadow_depth",
            SUN_DYNAMIC_DIM * SUN_ATLAS_TILES,
            SUN_DYNAMIC_DIM,
        );

        Self {
            sun,
            sun_next,
            sun_layer: std::cell::Cell::new(0),
            sun_array_view,
            sun_dynamic,
            spots,
            characters,
            sun_near,
            _spot_texture: spot_texture,
            spot_array_view,
            spot_tile_dim: dim,
            tile_scissor: Default::default(),
            sampler,
            solid_pipeline,
            solid_world_pipeline,
            solid_world: Default::default(),
            mesh_pipeline,
            skinned_pipeline,
            posed_pipeline,
            brush_pipeline,
            brush_world_pipeline,
            brush_world: Default::default(),
        }
    }

    fn draw_posed(&self, pass: &mut RenderPass, light_bg: &BindGroup, posed_draws: &[ShadowMeshDraw]) -> u32 {
        if posed_draws.is_empty() {
            return 0;
        }
        pass.set_pipeline(&self.posed_pipeline);
        pass.set_bind_group(0, light_bg, &[]);
        let mut drawn = 0;
        for (vb, ib, count, model_bg) in posed_draws {
            pass.set_bind_group(1, *model_bg, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..*count, 0, 0..1);
            drawn += *count;
        }
        drawn
    }

    /// Both static sun maps, as the lights block binds them (`sun_shadow_tex`).
    pub fn sun_depth_view(&self) -> &TextureView {
        &self.sun_array_view
    }

    /// Which static sun map (0 or 1) `ShadowKind::Sun` draws into and
    /// uploads the light of (`lights::SunPair`).
    pub fn set_sun_layer(&self, layer: u32) {
        self.sun_layer.set(layer.min(1));
    }

    pub fn sun_dynamic_depth_view(&self) -> &TextureView {
        self.sun_dynamic.depth_view.as_ref().expect("the dynamic sun owns its own depth target")
    }

    pub fn spot_depth_view(&self) -> &TextureView {
        &self.spot_array_view
    }

    pub fn set_tile_scissor(&self, on: bool) {
        self.tile_scissor.set(on);
    }

    pub fn spot_tile_dim(&self) -> u32 {
        self.spot_tile_dim
    }

    /// The static sun map's width in texels: the same `dim` as a spot tile.
    pub fn sun_dim(&self) -> u32 {
        self.spot_tile_dim
    }

    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    fn slot(&self, kind: ShadowKind) -> &ShadowSlot {
        match kind {
            ShadowKind::Sun => if self.sun_layer.get() == 1 { &self.sun_next } else { &self.sun },
            ShadowKind::SunDynamic => &self.sun_dynamic,
            ShadowKind::SunNear => &self.sun_near,
            ShadowKind::Spot(i) => &self.spots[i.min(MAX_SPOT_SHADOWS - 1)],
            ShadowKind::Character(k) => &self.characters[k.min(MAX_CHARACTER_SHADOWS - 1)],
        }
    }

    pub fn upload_light(&self, queue: &Queue, kind: ShadowKind, view_proj: Mat4) {
        let m = LightMatrix {
            view_proj: view_proj.to_cols_array_2d(),
        };
        queue.write_buffer(&self.slot(kind).light_buffer, 0, bytemuck::bytes_of(&m));

        if kind == ShadowKind::SunDynamic {
            self.upload_light(queue, ShadowKind::SunNear, sun_near_matrix(view_proj));
        }
    }

    pub fn set_solid_world(&self, model: Option<&BindGroup>, whole: bool) {
        *self.solid_world.borrow_mut() = model.map(|m| (m.clone(), whole));
    }

    pub fn set_brush_world(&self, model: Option<&BindGroup>) {
        *self.brush_world.borrow_mut() = model.cloned();
    }

    fn brush_pipeline_now(&self) -> (&RenderPipeline, Option<BindGroup>) {
        match self.brush_world.borrow().clone() {
            Some(model) => (&self.brush_world_pipeline, Some(model)),
            None => (&self.brush_pipeline, None),
        }
    }

    fn solid_pipeline_for(&self, chunks: bool) -> (&RenderPipeline, Option<BindGroup>) {
        match self.solid_world.borrow().clone().filter(|(_, whole)| chunks || *whole) {
            Some((model, _)) => (&self.solid_world_pipeline, Some(model)),
            None => (&self.solid_pipeline, None),
        }
    }

    #[allow(clippy::too_many_arguments, reason = "GPU setup passes each resource explicitly; bundling them is a separate refactor")]
    pub fn record_spots(
        &self,
        encoder: &mut CommandEncoder,
        count: usize,
        view_proj: &[Mat4],
        solid: Option<(&Buffer, &Buffer, u32)>,
        brushes: Option<(&Buffer, &Buffer, u32)>,
        mesh_draws: &[ShadowMeshDraw],
        mesh_bounds: &[ShadowMeshBound],
        skinned_draws: &[ShadowSkinnedDraw],
        posed_draws: &[ShadowMeshDraw],
        solid_chunks: &[CasterChunk],
    ) -> u32 {
        let mut drawn = 0u32;
        let (solid_pipeline, solid_model) = self.solid_pipeline_for(!solid_chunks.is_empty());
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
            if self.tile_scissor.get() {
                let t = self.spot_tile_dim;
                pass.set_scissor_rect(col * t, row * t, t, t);
            }

            let planes = frustum_planes(view_proj[layer]);
            let light_bg = &self.spots[layer].light_bind_group;

            if let Some((vb, ib, total)) = solid {
                if total > 0 {
                    pass.set_pipeline(solid_pipeline);
                    pass.set_bind_group(0, light_bg, &[]);
                    if let Some(model) = &solid_model {
                        pass.set_bind_group(1, model, &[]);
                    }
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    if solid_chunks.is_empty() {
                        pass.draw_indexed(0..total, 0, 0..1);
                        drawn += total;
                    } else {

                        let runs = crate::renderer::ground_lod::merged(
                            solid_chunks.iter().filter(|c| aabb_in_frustum(&planes, c.min, c.max)).map(|c| (c.first_index, c.index_count)),
                        );
                        for (first, count) in runs {
                            pass.draw_indexed(first..first + count, 0, 0..1);
                            drawn += count;
                        }
                    }
                }
            }
            if let Some((vb, ib, total)) = brushes {
                if total > 0 {
                    let (pipeline, model) = self.brush_pipeline_now();
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, light_bg, &[]);
                    if let Some(model) = &model {
                        pass.set_bind_group(1, model, &[]);
                    }
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..total, 0, 0..1);
                    drawn += total;
                }
            }
            if !mesh_draws.is_empty() {
                pass.set_pipeline(&self.mesh_pipeline);
                pass.set_bind_group(0, light_bg, &[]);
                for (i, (vb, ib, count, model_bg)) in mesh_draws.iter().enumerate() {
                    if !mesh_caster_reaches(&planes, mesh_bounds, i) {
                        continue;
                    }
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
            drawn += self.draw_posed(&mut pass, light_bg, posed_draws);
        }
        drop(pass);
        drawn
    }

    #[allow(clippy::too_many_arguments, reason = "GPU setup passes each resource explicitly; bundling them is a separate refactor")]
    pub fn record_moving(
        &self,
        encoder: &mut CommandEncoder,
        sun: bool,
        sun_view_proj: Mat4,
        characters: &[Mat4],
        mesh_draws: &[ShadowMeshDraw],
        mesh_bounds: &[ShadowMeshBound],
        skinned_draws: &[ShadowSkinnedDraw],
        posed_draws: &[ShadowMeshDraw],
        tile_meshes: &[usize],
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
            drawn += self.draw_posed(pass, light_bg, posed_draws);
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

            let tiles = [
                (0, &self.sun_dynamic, frustum_planes(sun_view_proj)),
                (SUN_NEAR_TILE, &self.sun_near, frustum_planes(sun_near_matrix(sun_view_proj))),
            ];
            for (tile, slot, planes) in tiles {
                sun_atlas_viewport(&mut pass, tile);
            if !mesh_draws.is_empty() {
                pass.set_pipeline(&self.mesh_pipeline);
                pass.set_bind_group(0, &slot.light_bind_group, &[]);
                for (i, (vb, ib, count, model_bg)) in mesh_draws.iter().enumerate() {
                    if !mesh_caster_reaches(&planes, mesh_bounds, i) {
                        continue;
                    }
                    pass.set_bind_group(1, *model_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
            }
            skinned(&mut pass, &slot.light_bind_group);
        }
        }
        let mut tile_drawn = 0u32;
        for (k, tile) in characters.iter().take(MAX_CHARACTER_SHADOWS).enumerate() {
            sun_atlas_viewport(&mut pass, 1 + k as u32);
            let bind = &self.characters[k].light_bind_group;
            if !tile_meshes.is_empty() {
                let planes = frustum_planes(*tile);
                pass.set_pipeline(&self.mesh_pipeline);
                pass.set_bind_group(0, bind, &[]);
                for &i in tile_meshes {
                    let Some((vb, ib, count, model_bg)) = mesh_draws.get(i) else { continue };
                    if !mesh_caster_reaches(&planes, mesh_bounds, i) {
                        continue;
                    }
                    pass.set_bind_group(1, *model_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                    tile_drawn += *count;
                }
            }
            skinned(&mut pass, bind);
        }
        drop(pass);
        drawn + tile_drawn
    }

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
        self.record_full(encoder, kind, solid, None, brushes, mesh_draws, skinned_draws, solid_chunks, light_view_proj)
    }

    #[allow(clippy::too_many_arguments, reason = "GPU setup passes each resource explicitly; bundling them is a separate refactor")]
    pub fn record_with_ground(
        &self,
        encoder: &mut CommandEncoder,
        kind: ShadowKind,
        solid: Option<(&Buffer, &Buffer, u32)>,
        ground: Option<(&Buffer, &Buffer, u32)>,
        mesh_draws: &[ShadowMeshDraw],
        skinned_draws: &[ShadowSkinnedDraw],
        light_view_proj: Mat4,
    ) -> u32 {
        self.record_full(encoder, kind, solid, ground, None, mesh_draws, skinned_draws, &[], light_view_proj)
    }

    #[allow(clippy::too_many_arguments, reason = "GPU setup passes each resource explicitly; bundling them is a separate refactor")]
    fn record_full(
        &self,
        encoder: &mut CommandEncoder,
        kind: ShadowKind,
        solid: Option<(&Buffer, &Buffer, u32)>,
        ground: Option<(&Buffer, &Buffer, u32)>,
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

            ShadowKind::Spot(_) => &self.spot_array_view,

            ShadowKind::Character(_) | ShadowKind::SunNear => self.sun_dynamic.depth_view.as_ref().expect("the dynamic sun owns its own depth target"),
        };
        let (solid_pipeline, solid_model) = self.solid_pipeline_for(!solid_chunks.is_empty());
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
                    pass.set_pipeline(solid_pipeline);
                    pass.set_bind_group(0, &slot.light_bind_group, &[]);
                    if let Some(model) = &solid_model {
                        pass.set_bind_group(1, model, &[]);
                    }
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    if solid_chunks.is_empty() {

                        pass.draw_indexed(0..count, 0, 0..1);
                        drawn_indices += count;
                    } else {
                        let runs = crate::renderer::ground_lod::merged(
                            solid_chunks.iter().filter(|c| aabb_in_frustum(&planes, c.min, c.max)).map(|c| (c.first_index, c.index_count)),
                        );
                        for (first, count) in runs {
                            pass.draw_indexed(first..first + count, 0, 0..1);
                            drawn_indices += count;
                        }
                    }
                }
            }

            if let Some((vb, ib, count)) = ground {
                if count > 0 {
                    pass.set_pipeline(&self.solid_pipeline);
                    pass.set_bind_group(0, &slot.light_bind_group, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..count, 0, 0..1);
                    drawn_indices += count;
                }
            }

            if let Some((vb, ib, count)) = brushes {
                if count > 0 {
                    let (pipeline, model) = self.brush_pipeline_now();
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &slot.light_bind_group, &[]);
                    if let Some(model) = &model {
                        pass.set_bind_group(1, model, &[]);
                    }
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

const SHADOW_SHADER: &str = r"
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

@vertex
fn vs_posed(@location(0) skinned: vec4<f32>) -> @builtin(position) vec4<f32> {
    return light.view_proj * model_u.model * skinned;
}
";

#[cfg(test)]
mod atlas_tests {
    use super::*;

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

        let lamps = [lamp(1.0, 4.0, true), lamp(-2.0, 4.0, true), lamp(0.5, 50.0, false), lamp(12.0, 99.0, true), lamp(2.0, 4.6, true)];
        let s = |i: usize| lamps[i].intensity / ((lamps[i].position - body).length_squared() + 0.25);
        assert!(s(4) > s(1) && s(4) < s(1) / (1.0 / 1.33), "the test lamps are not placed as meant");
        assert_eq!(character_shadow_lamps(&lamps, body, r, &[lamps[0].position, lamps[1].position]), vec![0, 1]);

        let lamps = [lamp(1.0, 4.0, true), lamp(-2.0, 4.0, true), lamp(0.5, 50.0, false), lamp(12.0, 99.0, true), lamp(1.5, 9.0, true)];
        assert_eq!(character_shadow_lamps(&lamps, body, r, &[lamps[0].position, lamps[1].position]), vec![4, 0]);

        let beside = CharacterLamp { position: Vec3::new(1.8, 2.4, 0.0), cos_outer: 20f32.to_radians().cos(), intensity: 40.0, ..lamp(0.0, 0.0, true) };
        assert_eq!(character_shadow_lamps(&[beside, lamp(-2.0, 4.0, true)], body, r, &[]), vec![1]);
        let over = CharacterLamp { position: Vec3::new(0.3, 2.4, 0.0), ..beside };
        assert_eq!(character_shadow_lamps(&[over, lamp(-2.0, 4.0, true)], body, r, &[]), vec![0, 1]);
    }

    #[test]
    fn the_lamps_reaching_a_door_shadow_it_in_their_tiles() {
        let point = |p: Vec3, intensity: f32| CharacterLamp {
            position: p,
            direction: Vec3::NEG_Y,
            cos_outer: -1.0,
            range: 6.0,
            intensity,
            eligible: true,
        };

        let leaf = DoorCaster {
            corners: std::array::from_fn(|i| {
                Vec3::new(
                    if i & 4 == 0 { 2.98 } else { 3.02 },
                    if i & 2 == 0 { 0.0 } else { 2.2 },
                    if i & 1 == 0 { -3.8 } else { -3.0 },
                )
            }),
        };
        let eye = Vec3::new(1.0, 1.6, -3.4);

        let lamps = [point(Vec3::new(5.0, 1.9, -4.2), 3.0), point(Vec3::new(0.0, 3.1, -4.5), 9.0), point(Vec3::new(11.0, 2.0, -3.0), 9.0)];
        let tiles = moving_caster_tiles(&lamps, None, &[leaf], eye, &[]);
        let chosen: Vec<usize> = tiles.iter().map(|t| t.0).collect();
        assert_eq!(chosen, vec![1, 0], "both lamps in reach, strongest at the door first");
        assert!(tiles.iter().all(|t| t.1.len() == 8), "a door-only tile holds the leaf's corners");

        let body_parts = [(Vec3::new(1.0, 0.4, -3.4), 0.2), (Vec3::new(1.0, 1.5, -3.4), 0.2)];
        let tiles = moving_caster_tiles(&lamps, Some((Vec3::new(1.0, 0.9, -3.4), 0.8, &body_parts)), &[leaf], eye, &[]);
        let lamp1 = tiles.iter().find(|t| t.0 == 1).expect("lamp 1 has a tile");
        assert_eq!(lamp1.1.len(), 2 + 8, "the body's spheres and the leaf's corners");
        assert!(tiles.len() <= MAX_CHARACTER_SHADOWS);

        assert!(moving_caster_tiles(&lamps, None, &[leaf], eye + Vec3::new(0.0, 0.0, 20.0), &[]).is_empty());

        let mut many = vec![point(Vec3::new(1.0, 2.4, -3.4), 50.0), point(Vec3::new(1.5, 2.4, -3.4), 50.0)];
        many.extend([point(Vec3::new(4.0, 1.9, -3.4), 3.0), point(Vec3::new(4.0, 1.9, -3.5), 3.2), point(Vec3::new(4.0, 2.0, -3.3), 3.1)]);
        let body = Some((Vec3::new(1.0, 0.9, -3.4), 0.8, &body_parts[..]));
        let fresh: Vec<usize> = moving_caster_tiles(&many, body, &[leaf], eye, &[]).iter().map(|t| t.0).collect();
        assert_eq!(fresh, vec![0, 1, 3, 2], "two for the player, then the two strongest at the door");

        let held = [many[0].position, many[1].position, many[3].position, many[4].position];
        let kept: Vec<usize> = moving_caster_tiles(&many, body, &[leaf], eye, &held).iter().map(|t| t.0).collect();
        assert!(kept.contains(&3) && kept.contains(&4) && !kept.contains(&2), "held door lamps keep their tiles: {kept:?}");

        let far_leaf = DoorCaster { corners: leaf.corners.map(|c| c + Vec3::new(7.3, 0.0, 0.0)) };
        let between = [point(Vec3::new(5.0, 1.9, -4.0), 3.0)];
        let tiles = moving_caster_tiles(&between, None, &[leaf, far_leaf], eye, &[]);
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].1.len(), 8, "one leaf's corners");
        assert!(tiles[0].1.iter().all(|(c, _)| c.x < 3.1), "the nearer leaf's");
        let m = character_light_matrix(between[0].position, None, &tiles[0].1, 6.0).unwrap();
        let behind = m * Vec3::new(1.5, 0.0, -3.2).extend(1.0);
        let ndc = behind.truncate() / behind.w;
        assert!(ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0 && ndc.z < 1.0, "the floor behind the leaf is in its tile: {ndc}");
    }

    #[test]
    fn a_character_tile_holds_the_body_and_what_it_shadows() {
        let inside = |m: Mat4, p: Vec3| {
            let c = m * p.extend(1.0);
            let n = c.truncate() / c.w;
            c.w > 0.0 && n.x.abs() <= 1.0 && n.y.abs() <= 1.0 && (0.0..=1.0).contains(&n.z)
        };

        let width = |m: Mat4, lamp: Vec3| {
            let inv = m.inverse();
            let edge = |x: f32| (inv.project_point3(Vec3::new(x, 0.0, 0.5)) - lamp).normalize();
            edge(-1.0).dot(edge(1.0)).clamp(-1.0, 1.0).acos().to_degrees()
        };
        let lamp = Vec3::new(1.5, 2.4, 0.0);
        let centre = Vec3::new(0.0, 0.9, 0.0);

        let (foot, head) = (Vec3::new(0.0, 0.2, 0.0), Vec3::new(0.0, 1.6, 0.0));
        let body = [(foot, 0.2), (head, 0.2)];
        let m = character_light_matrix(lamp, None, &body, 8.0).expect("a lamp outside the body");
        let to = (centre - lamp).normalize();
        let side = to.cross(Vec3::Y).normalize();
        for p in [foot, head, head + Vec3::Y * 0.25, foot - Vec3::Y * 0.25, centre + side * 0.25] {
            assert!(inside(m, p), "{p} of the body is outside its tile");
        }

        let ray = (head - lamp).normalize();
        let floor = lamp + ray * (lamp.y / -ray.y);
        assert!(inside(m, floor), "the floor behind the body, {floor}, is outside the tile");

        let sphere = character_light_matrix(lamp, None, &[(centre, 1.0)], 8.0).unwrap();
        assert!(
            width(m, lamp) < 0.6 * width(sphere, lamp),
            "a slim body's tile is {} degrees, a sphere's {}",
            width(m, lamp),
            width(sphere, lamp)
        );

        let spot_at = Vec3::new(0.0, 1.7, 0.0);
        let beam = (Vec3::X, 17f32.to_radians().cos());
        let player = [
            (Vec3::new(0.35, 0.2, 0.3), 0.15),
            (Vec3::new(0.35, 1.6, 0.3), 0.15),
            (Vec3::new(0.45, 1.5, 0.1), 0.05),
            (Vec3::new(0.7, 1.65, 0.05), 0.05),
        ];
        let m = character_light_matrix(spot_at, Some(beam), &player, 4.0).expect("a spot inside the player's reach has a tile");
        let hand = Vec3::new(0.7, 1.65, 0.05);
        let shadow = spot_at + (hand - spot_at) * (1.2 / (hand - spot_at).x);
        assert!(inside(m, hand) && inside(m, shadow), "the hand {hand} or its shadow on the wall {shadow} is outside the tile");
        assert!(width(m, spot_at) < 36.5, "the spot's tile is {} degrees across, wider than its cone", width(m, spot_at));

        let behind = [(Vec3::new(-1.0, 1.0, 0.0), 0.2), (Vec3::new(-1.0, 1.5, 0.0), 0.2)];
        assert!(character_light_matrix(spot_at, Some(beam), &behind, 4.0).is_none());

        let near = centre + Vec3::X * 0.1;
        let reached = character_light_matrix(near, None, &[(centre, 1.0)], 8.0).expect("a tile");
        assert!(width(reached, near) < 151.0, "{}", width(reached, near));
    }
}

#[cfg(test)]
mod frustum_tests {
    use super::*;

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

        let (min, max) = box_at(Vec3::new(30.0, 0.0, -5.0), 0.5);
        assert!(!aabb_in_frustum(&spot(), min, max), "outside the cone must be culled");
    }

    #[test]
    fn a_box_containing_the_light_is_kept() {

        let (min, max) = box_at(Vec3::ZERO, 60.0);
        assert!(aabb_in_frustum(&spot(), min, max), "a box enclosing the light must be drawn");
    }

    #[test]
    fn culling_never_drops_a_box_that_straddles_a_plane() {

        let (min, max) = box_at(Vec3::new(0.0, 0.0, -10.0), 2.0);
        assert!(aabb_in_frustum(&spot(), min, max), "a straddling box must be kept");
    }

    #[test]
    fn the_sun_frustum_culls_too() {

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

#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::renderer::cuboid::SolidVertex;
    use crate::renderer::lights::{Light, LightKind, LightsUniform};
    use crate::renderer::pipeline::{lightmap_bind_group_layout, SolidPipeline};
    use crate::renderer::uniforms::{PlayerUpload, PostUpload, ShadowUpload, SkyUpload, UniformBuffer};
    use crate::renderer::Color3;
    use wgpu::util::DeviceExt;

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

    struct Scene {
        lights: Vec<Light>,

        caster: Option<(Vec<SolidVertex>, Vec<u32>)>,

        receiver_at: glam::Vec3,

        eye_offset: glam::Vec3,
        shadows_on: bool,

        shadow_dim: u32,

        normal: [f32; 3],

        sky: SkyUpload,

        sky_sun_dynamic: bool,

        character_lamp: Option<usize>,
    }

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

        let spot_indices: Vec<usize> = if scene.shadows_on {
            scene
                .lights
                .iter()
                .enumerate()
                .filter(|(_, l)| l.kind == LightKind::Spot && l.casts_shadow())
                .map(|(i, _)| i)
                .take(MAX_SPOT_SHADOWS)
                .collect()
        } else {
            Vec::new()
        };
        lights_uniform.upload_frame(&queue, &scene.lights, &spot_indices, scene.sky_sun_dynamic);

        let sun = scene.lights.iter().find(|l| l.kind == LightKind::Directional);
        let sun_view_proj = sun
            .map_or(Mat4::IDENTITY, |l| directional_light_matrix(l.direction, Vec3::ZERO, 20.0));

        let spot_matrices: Vec<SpotShadowMatrices> = spot_indices
            .iter()
            .map(|&i| spot_shadow_matrices(&scene.lights[i], shadow_map.spot_tile_dim()))
            .collect();
        let mut spot_view_proj = [Mat4::IDENTITY; SHADOW_MATRICES];
        for (layer, mats) in spot_matrices.iter().enumerate() {
            spot_view_proj[layer] = mats.lookup;
        }

        let character_tile = scene.character_lamp.zip(scene.caster.as_ref()).and_then(|(i, (v, _))| {
            let centre = v.iter().fold(Vec3::ZERO, |s, p| s + Vec3::from(p.position)) / v.len().max(1) as f32;
            let radius = v.iter().map(|p| (Vec3::from(p.position) - centre).length()).fold(0.0f32, f32::max);
            let l = &scene.lights[i];
            let spot = (l.kind == LightKind::Spot).then(|| (l.direction, (l.cone_angle_deg.to_radians() * 0.5).cos()));
            character_light_matrix(l.position, spot, &[(centre, radius)], l.range).map(|m| (i, m))
        });
        if let Some((i, m)) = character_tile {
            spot_view_proj[MAX_SPOT_SHADOWS] = m;

            lights_uniform.set_tile_lamps(&[i]);
            lights_uniform.upload_frame(&queue, &scene.lights, &spot_indices, scene.sky_sun_dynamic);
        }
        let upload = ShadowUpload {
            sun_view_proj,
            spot_view_proj,
            sun_enabled: scene.shadows_on && sun.is_some() && !scene.sky_sun_dynamic,
            spot_count: spot_indices.len() as u32,
            sun_dynamic_view_proj: sun_view_proj,
            sun_dynamic_enabled: scene.shadows_on && sun.is_some() && scene.sky_sun_dynamic,
            sun_next_view_proj: Mat4::IDENTITY,
            sun_blend: [0.0; 4],
        };

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

        let quad: Vec<SolidVertex> = [
            [-1.0, -1.0, 0.0],
            [3.0, -1.0, 0.0],
            [-1.0, 3.0, 0.0],
        ]
        .iter()
        .map(|p| {

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
            size: u64::from(256 * SIZE),
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

        for layer in 0..upload.spot_count as usize {
            shadow_map.upload_light(&queue, ShadowKind::Spot(layer), spot_matrices[layer].pass);
            let solid = caster_bufs
                .as_ref()
                .map(|(vb, ib, count)| (vb, ib, *count));
            shadow_map.record(
                &mut encoder, ShadowKind::Spot(layer), solid, None, &[], &[], &[],
                spot_matrices[layer].pass,
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
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: true,
            position: Vec3::ZERO,

            direction: Vec3::NEG_Y,
            kind: LightKind::Directional,
            color: Color3(255, 255, 255, 255),

            intensity: 0.2,

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

    #[test]
    fn a_lamp_shadows_the_player_through_the_characters_tile() {
        let lamp = Light {
            mask_channel: None,
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: true,
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
        let Some(shadowed) = shade_receiver(scene(vec![lamp], Some(0))) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let open = shade_receiver(scene(vec![lamp], None)).unwrap();
        let dark = shade_receiver(scene(Vec::new(), None)).unwrap();
        eprintln!("through the characters' tile {shadowed:?}, without it {open:?}, no lamp {dark:?}");
        assert!(i32::from(open[0]) - i32::from(dark[0]) > 40, "the lamp does not light the receiver measurably: {open:?} vs {dark:?}");
        assert!(
            (i32::from(shadowed[0]) - i32::from(dark[0])).abs() <= 2,
            "the characters' tile let the lamp's light through: {shadowed:?}, dark {dark:?}, open {open:?}",
        );
    }

    fn spot_above(x: f32, intensity: f32) -> Light {
        Light {
            mask_channel: None,
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: true,
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

        let lit = shot!(Scene { lights: vec![spot_above(0.0, 4.0)], ..base() });
        let shaded = shot!(Scene {
            lights: vec![spot_above(0.0, 4.0)],
            caster: Some(ground(3.0, 8.0)),
            ..base()
        });
        assert!(
            i32::from(lit[0]) - i32::from(shaded[0]) > 8,
            "a spot cast no shadow: {lit:?} lit vs {shaded:?} with a caster between",
        );
    }

    #[test]
    fn a_second_spot_light_also_casts_a_shadow() {

        let dim = spot_above(-4.0, 0.0);
        let lit = shot!(Scene { lights: vec![dim, spot_above(0.0, 4.0)], ..base() });
        let shaded = shot!(Scene {
            lights: vec![dim, spot_above(0.0, 4.0)],
            caster: Some(ground(3.0, 8.0)),
            ..base()
        });
        assert!(
            i32::from(lit[0]) - i32::from(shaded[0]) > 8,
            "the SECOND spot cast no shadow: {lit:?} lit vs {shaded:?} with a caster between",
        );
    }

    #[test]
    fn a_spot_beyond_the_budget_still_lights_without_shadowing() {

        let mut lights: Vec<Light> = (0..=MAX_SPOT_SHADOWS)
            .map(|i| spot_above(-8.0 + i as f32, 0.0))
            .collect();
        *lights.last_mut().unwrap() = spot_above(0.0, 4.0);

        let with_extra = shot!(Scene { lights: lights.clone(), ..base() });
        let alone = shot!(Scene { lights: vec![spot_above(0.0, 4.0)], ..base() });
        assert!(
            (i32::from(with_extra[0]) - i32::from(alone[0])).abs() < 6,
            "the light past the shadow budget stopped lighting: {with_extra:?} vs {alone:?}",
        );
    }

    #[test]
    fn a_sun_lights_a_surface_facing_it() {
        let lit = shot!(base());
        let unlit = shot!(Scene { lights: vec![], ..base() });
        assert!(
            i32::from(lit[0]) - i32::from(unlit[0]) > 8,
            "a directional light contributed nothing: {lit:?} vs ambient {unlit:?}",
        );
    }

    #[test]
    fn a_sun_does_not_dim_with_distance() {

        let near = shot!(base());
        let far = shot!(Scene { receiver_at: Vec3::new(0.0, 500.0, 0.0), ..base() });
        for c in 0..3 {
            assert!(
                (i32::from(near[c]) - i32::from(far[c])).abs() <= 2,
                "the sun attenuated with distance: {near:?} at the origin vs {far:?} at y=500",
            );
        }
    }

    #[test]
    fn an_occluder_darkens_what_is_under_it() {

        let clear = shot!(base());
        let shaded = shot!(Scene {
            caster: Some(ground(3.0, 8.0)),
            ..base()
        });
        assert!(
            i32::from(clear[0]) - i32::from(shaded[0]) > 8,
            "a slab three metres overhead cast no shadow: {clear:?} lit vs {shaded:?} shadowed",
        );
    }

    #[test]
    fn a_spot_casts_a_shadow_through_its_own_map() {

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

        let shaded = shot!(Scene { caster: Some(ground(3.0, 8.0)), ..base() });
        let ambient_only = shot!(Scene { lights: vec![], ..base() });
        for c in 0..3 {
            assert!(
                (i32::from(shaded[c]) - i32::from(ambient_only[c])).abs() <= 3,
                "a shadowed fragment is not at ambient: {shaded:?} vs {ambient_only:?}",
            );
        }
    }

    #[test]
    fn geometry_outside_the_shadow_map_is_lit_rather_than_black() {

        let inside = shot!(base());
        let beside = shot!(Scene { receiver_at: Vec3::new(300.0, 0.0, 0.0), ..base() });
        for c in 0..3 {
            assert!(
                (i32::from(inside[c]) - i32::from(beside[c])).abs() <= 2,
                "off the side of the shadow map went dark: {inside:?} vs {beside:?}",
            );
        }
    }

    #[test]
    fn the_falloff_the_editor_predicts_is_the_falloff_the_shader_applies() {

        let at = |intensity: f32, distance: f32| {
            shade_receiver(Scene {
                lights: vec![Light {
                    mask_channel: None,
                    shadow_near: None,
                    source_radius: 0.0,
                    in_level_bake: true,
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

        for (intensity, distance, expected) in
            [(1.0f32, 5.0f32, 0.6399f32), (3.0, 5.0, 0.7196), (10.0, 5.0, 0.9987), (0.2, 0.85, 0.8768)]
        {
            let Some(px) = at(intensity, distance) else {
                eprintln!("skipping: no GPU adapter available");
                return;
            };
            let got = f32::from(px[0]) / 255.0;
            assert!(
                (got - expected).abs() < 0.01,
                "intensity {intensity} at {distance}m: shader gave {got:.4}, the editor \
                 predicts {expected:.4}",
            );
        }
    }

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
            i32::from(open[1]) - i32::from(shadowed[1]) > 20,
            "a moving object cast no sun shadow: {shadowed:?} vs {open:?}",
        );
    }

    #[test]
    fn the_near_tile_holds_the_middle_of_the_box_at_twice_the_detail() {

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

        let texel = 2.0 * 20.0 / SUN_DYNAMIC_DIM as f32;

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
            (i32::from(lit_near[1]) - i32::from(lit_wide[1])).abs() <= 1 && (i32::from(dark_near[1]) - i32::from(dark_wide[1])).abs() <= 1,
            "the two places are not lit alike: lit {lit_near:?} / {lit_wide:?}, shadowed {dark_near:?} / {dark_wide:?}",
        );
        let range = i32::from(lit_wide[1]) - i32::from(dark_wide[1]);
        assert!(
            range > 40,
            "too little sun to measure a penumbra by: {range}"
        );

        let lost_wide = i32::from(lit_wide[1]) - i32::from(wide[1]);
        let lost_near = i32::from(lit_near[1]) - i32::from(near[1]);
        assert!(
            lost_wide * 10 > range && lost_near * 4 < lost_wide,
            "the middle of the box was not read from the near tile: lost {lost_near} there and {lost_wide} \
             through the sun tile, of {range}",
        );
    }

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
            // The player's pivot sits on the grid (`lights::static_sun_pivot`).
            let o = texel_of(near, crate::renderer::lights::static_sun_pivot(head));
            assert!(
                (o - o.round()).abs().max_element() < 2e-3,
                "the player's pivot is off the near tile's grid at {o:?}, player at {head}",
            );
            let p = head + Vec3::new(0.37, -0.81, 0.52);
            let (w, nr) = (wide.project_point3(p), near.project_point3(p));
            assert!(
                (nr.x - SUN_NEAR_ZOOM * w.x).abs() < 1e-5
                    && (nr.y - SUN_NEAR_ZOOM * w.y).abs() < 1e-5
                    && (nr.z - w.z).abs() < 1e-6
            );
        }
    }

    #[test]
    fn a_culled_caster_had_nothing_inside_the_tile() {

        let head = Vec3::new(3.137, 1.7, -2.71);
        let wide = crate::renderer::lights::dynamic_sun_matrix(Vec3::new(0.4, -0.75, 0.3).normalize(), head);
        let tiles = [
            ("sun tile", wide),
            ("near tile", sun_near_matrix(wide)),
            ("spot", spot_light_matrix(head + Vec3::new(1.0, 1.2, 0.5), Vec3::new(-0.2, -1.0, 0.1), 70.0, 8.0)),
        ];
        let mut seed = 0x2545_f491_u32;
        let mut rand = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for (name, m) in tiles {
            let planes = frustum_planes(m);
            let (mut kept, mut culled) = (0, 0);
            for _ in 0..3000 {
                let centre = head + Vec3::new(rand() - 0.5, rand() - 0.5, rand() - 0.5) * 16.0;
                let sphere = centre.extend(0.05 + rand() * 1.5);
                let inside = sphere_in_frustum(&planes, sphere);
                if inside {
                    kept += 1;
                    continue;
                }
                culled += 1;
                for _ in 0..200 {
                    let d = Vec3::new(rand() - 0.5, rand() - 0.5, rand() - 0.5);
                    let p = sphere.truncate() + d.normalize_or_zero() * sphere.w * rand().sqrt();
                    let c = m * p.extend(1.0);
                    let clipped = c.x < -c.w || c.x > c.w || c.y < -c.w || c.y > c.w || c.z < 0.0 || c.z > c.w;
                    assert!(clipped, "{name}: culled {sphere:?} but {p} is inside the tile at {c:?}");
                }
            }
            assert!(culled > 100 && kept > 10, "{name}: kept {kept}, culled {culled}");
        }
    }

    #[test]
    fn the_skys_ambient_follows_the_surface_normal() {

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
            SkyUpload::from(&crate::renderer::sky::SkyIrradiance::from_panorama(&pano, 0.0, 1.0))
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
            i32::from(toward[0]) - i32::from(away[0]) > 20,
            "the ambient did not follow the normal: facing the bright half gave \
             {toward:?}, facing away gave {away:?}",
        );
    }

    #[test]
    fn the_shader_evaluates_the_same_irradiance_the_cpu_does() {

        let pano = {
            let mut p = crate::renderer::sky::Panorama::solid([0.05, 0.05, 0.05], 64, 32);

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
        let irr = crate::renderer::sky::SkyIrradiance::from_panorama(&pano, 0.0, 1.0);
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

                let got = f32::from(px[c]) / 255.0;
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

        let up = shot!(Scene { lights: vec![], normal: [0.0, 1.0, 0.0], ..base() });
        let side = shot!(Scene { lights: vec![], normal: [1.0, 0.0, 0.0], ..base() });
        let down = shot!(Scene { lights: vec![], normal: [0.0, -1.0, 0.0], ..base() });
        for (a, b) in [(up, side), (up, down)] {
            for c in 0..3 {
                assert!(
                    (i32::from(a[c]) - i32::from(b[c])).abs() <= 1,
                    "flat ambient is no longer flat: {a:?} vs {b:?}",
                );
            }
        }
        let ambient = (crate::renderer::sky::AMBIENT * 255.0).round() as i32;
        assert!(
            (i32::from(up[0]) - ambient).abs() <= 2,
            "ambient came out {} rather than {ambient}",
            up[0],
        );
    }

    #[test]
    fn specular_depends_on_where_the_eye_is() {

        let overhead = shot!(Scene { eye_offset: Vec3::new(0.0, 5.0, 0.0), ..base() });
        let oblique = shot!(Scene { eye_offset: Vec3::new(50.0, 0.5, 0.0), ..base() });
        assert!(
            i32::from(overhead[0]) - i32::from(oblique[0]) > 4,
            "the specular highlight did not follow the eye: {overhead:?} vs {oblique:?}",
        );
    }

    fn lamp_path() -> Option<std::path::PathBuf> {
        let p = crate::test_game_dir().join("models/lights/hanging_industrial_lamp/hanging_industrial_lamp_1k.gltf");
        p.is_file().then_some(p)
    }

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
                    .map_or(0.0, |t| t.transmission_factor())
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

    fn hall_spot_1() -> Light {
        Light {
            mask_channel: None,
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: true,
            position: Vec3::new(0.0, 3.1 - 1.18, -4.5),
            direction: Vec3::NEG_Y,
            kind: LightKind::Spot,
            color: Color3(255, 244, 214, 255),

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

    #[test]
    fn diag_self_shadow_acne() {

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
            i32::from(open[0]) - i32::from(full[0]) < 16,
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

    fn torch(height: f32, near: Option<f32>) -> Light {
        Light {
            mask_channel: None,
            shadow_near: near,
            source_radius: 0.0,
            in_level_bake: true,
            position: Vec3::new(0.0, height, -4.5),
            direction: Vec3::NEG_Y,
            kind: LightKind::Spot,
            color: Color3(255, 241, 228, 255),

            intensity: 0.12 * height * height,
            range: 20.0,
            cone_angle_deg: 50.0,
            inner_cone_angle_deg: 16.0,
        }
    }

    fn plate(y: f32, half: f32) -> (Vec<SolidVertex>, Vec<u32>) {
        let (mut v, i) = ground(y, half);
        for p in &mut v {
            p.position[2] -= 4.5;
        }
        (v, i)
    }

    #[test]
    fn a_hand_held_in_front_of_a_flashlight_shadows_its_beam() {
        let hand = || Some(plate(0.9, 0.03));
        let floor = |light: Light, caster| Scene { lights: vec![light], caster, ..floor_under_the_lamp() };

        let unlit = shot!(floor(Light { intensity: 0.0, ..torch(1.0, Some(0.02)) }, None));
        let open = shot!(floor(torch(1.0, Some(0.02)), None));
        let as_a_fixture = shot!(floor(torch(1.0, None), hand()));
        let as_a_torch = shot!(floor(torch(1.0, Some(0.02)), hand()));
        eprintln!(
            "TORCH HAND: unlit {unlit:?}  open {open:?}  fixture's near plane {as_a_fixture:?}  torch's {as_a_torch:?}"
        );
        assert!(
            i32::from(open[0]) > i32::from(unlit[0]) + 20,
            "the torch does not light the floor ({open:?} against {unlit:?}); nothing below means anything",
        );
        assert!(
            (i32::from(as_a_fixture[0]) - i32::from(open[0])).abs() <= 3,
            "with a fixture's near plane the hand 10 cm from the glass should be clipped out of \
             the map ({as_a_fixture:?} against {open:?}) -- the reason the torch has its own",
        );
        assert!(
            (i32::from(as_a_torch[0]) - i32::from(unlit[0])).abs() <= 3,
            "the hand before the torch does not shadow the floor: {as_a_torch:?}, unlit {unlit:?}, open {open:?}",
        );
    }

    #[test]
    fn a_flashlight_shadow_sits_on_its_caster_and_the_floor_does_not_shadow_itself() {
        let floor = |light: Light, caster| Scene { lights: vec![light], caster, ..floor_under_the_lamp() };
        let lamp = || torch(2.5, Some(0.02));
        let unlit = shot!(floor(Light { intensity: 0.0, ..lamp() }, None));
        let open = shot!(floor(lamp(), None));
        let low_plate = shot!(floor(lamp(), Some(plate(0.04, 0.2))));
        let itself = shot!(floor(lamp(), Some(ground(0.0, 8.0))));
        eprintln!(
            "TORCH BIAS: unlit {unlit:?}  open {open:?}  plate 4 cm up {low_plate:?}  floor in its own map {itself:?}"
        );
        assert!(i32::from(open[0]) > i32::from(unlit[0]) + 20, "the torch does not light the floor ({open:?} against {unlit:?})");
        assert!(
            (i32::from(low_plate[0]) - i32::from(unlit[0])).abs() <= 3,
            "a plate 4 cm off the floor casts nothing at 2.5 m: the lookup bias is not in texels \
             ({low_plate:?}, unlit {unlit:?}, open {open:?})",
        );
        assert!(
            (i32::from(open[0]) - i32::from(itself[0])) <= 3,
            "the floor shadows itself under the torch -- acne ({itself:?} against {open:?})",
        );
    }

    #[test]
    fn a_lit_patch_lights_the_floor_beside_it_as_a_disc_does() {
        let patch = |height: f32, radius: f32| Light {
            mask_channel: None,
            shadow_near: Some(f32::INFINITY),
            source_radius: radius,
            in_level_bake: true,
            position: Vec3::new(0.0, height, -4.5),
            direction: Vec3::NEG_Y,
            kind: LightKind::Point,
            color: Color3(255, 255, 255, 255),
            intensity: 0.12,
            range: 20.0,
            cone_angle_deg: 180.0,
            inner_cone_angle_deg: 0.0,
        };
        let floor = |light: Light| Scene { lights: vec![light], ..floor_under_the_lamp() };
        let unlit = shot!(floor(Light { intensity: 0.0, ..patch(1.0, 0.0) }));
        let bulb = shot!(floor(patch((1.0f32 + 0.05 * 0.05).sqrt(), 0.0)));
        let disc = shot!(floor(patch(0.05, 1.0)));
        let point = shot!(floor(patch(0.05, 0.0)));
        eprintln!("PATCH: unlit {unlit:?}  bulb 1 m up {bulb:?}  metre-wide patch 5 cm up {disc:?}  point 5 cm up {point:?}");
        assert!(i32::from(bulb[0]) > i32::from(unlit[0]) + 20, "the bulb does not light the floor ({bulb:?} against {unlit:?})");
        assert!(
            (i32::from(disc[0]) - i32::from(bulb[0])).abs() <= 2,
            "a metre-wide patch 5 cm up lights the floor to {disc:?}, a bulb a metre up to {bulb:?}",
        );
        assert!(
            i32::from(point[0]) > i32::from(bulb[0]) + 40,
            "a point 5 cm up lights the floor to {point:?}, no more than the bulb's {bulb:?}: \
             nothing above shows the radius doing anything",
        );
    }

    #[test]
    fn a_flashlights_shadow_reaches_as_far_as_its_light() {
        let floor = |light: Light, caster| Scene { lights: vec![light], caster, ..floor_under_the_lamp() };
        for height in [2.0f32, 5.0, 9.0, 12.0, 18.0] {
            let window = (1.0 - (height / 20.0).powi(4)).max(0.05).powi(2);
            let lamp = || Light { intensity: 0.12 * height * height / window, ..torch(height, Some(0.02)) };
            let hand = || Some(plate(height - 0.1, 0.03));
            let unlit = shot!(floor(Light { intensity: 0.0, ..lamp() }, None));
            let open = shot!(floor(lamp(), None));
            let shaded = shot!(floor(lamp(), hand()));
            eprintln!("TORCH REACH {height} m: unlit {unlit:?}  open {open:?}  hand {shaded:?}");
            assert!(
                i32::from(open[0]) > i32::from(unlit[0]) + 20,
                "the torch does not light the floor {height} m below ({open:?} against {unlit:?})",
            );
            assert!(
                (i32::from(shaded[0]) - i32::from(unlit[0])).abs() <= 3,
                "the hand 10 cm before the glass does not shadow the floor {height} m below: \
                 {shaded:?}, unlit {unlit:?}, open {open:?}",
            );
        }
    }

    #[test]
    fn the_lookup_bias_matches_the_shader() {
        let src = crate::renderer::lights::wgsl_lights_block(0, 1);
        assert!(
            src.contains(&format!("let bias = {SHADER_DEPTH_BIAS:?};")),
            "shadow_coords no longer subtracts SHADER_DEPTH_BIAS ({SHADER_DEPTH_BIAS:?})",
        );
    }

    #[test]
    fn a_fixtures_shadow_matrices_are_unchanged() {
        let l = hall_spot_1();
        let m = spot_shadow_matrices(&l, 1024);
        assert_eq!(m.pass, m.lookup);
        assert_eq!(m.pass, spot_light_matrix(l.position, l.direction, l.cone_angle_deg, l.range));
    }

    fn cuboid(lo: Vec3, hi: Vec3) -> (Vec<SolidVertex>, Vec<u32>) {
        let (mut v, mut idx) = (Vec::new(), Vec::new());
        for axis in 0..3 {
            for side in [0.0f32, 1.0] {
                let mut n = Vec3::ZERO;
                n[axis] = if side > 0.5 { 1.0 } else { -1.0 };
                let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
                let corner = |a: f32, b: f32| {
                    let mut p = Vec3::ZERO;
                    p[axis] = if side > 0.5 { hi[axis] } else { lo[axis] };
                    p[u] = if a > 0.5 { hi[u] } else { lo[u] };
                    p[w] = if b > 0.5 { hi[w] } else { lo[w] };
                    p
                };
                let q = [corner(0.0, 0.0), corner(1.0, 0.0), corner(1.0, 1.0), corner(0.0, 1.0)];
                let base = v.len() as u32;
                for p in q {
                    v.push(vertex(p.into(), n.into()));
                }
                let flip = (q[1] - q[0]).cross(q[2] - q[0]).dot(n) < 0.0;
                let tri: [u32; 6] = if flip { [0, 2, 1, 0, 3, 2] } else { [0, 1, 2, 0, 2, 3] };
                idx.extend(tri.iter().map(|t| base + t));
            }
        }
        (v, idx)
    }

    #[test]
    fn a_torchs_pillar_shadow_on_the_far_ceiling_is_where_the_rays_put_it() {
        let pos = Vec3::new(-0.8, 1.25, -4.7);
        let aim = Vec3::new(0.0, 1.9, -9.0);
        let lamp = |intensity: f32| Light {
            position: pos,
            direction: (aim - pos).normalize(),
            intensity,
            ..torch(1.0, Some(0.02))
        };
        let (pmin, pmax) = (Vec3::new(-0.45, 0.0, -7.45), Vec3::new(0.45, 3.1, -6.55));
        let mut caster = cuboid(pmin, pmax);

        for (lo, hi) in [(Vec3::new(-3.0, 3.4, -16.0), Vec3::new(3.0, 3.6, 4.0)), (Vec3::new(-3.0, 0.0, -16.2), Vec3::new(3.0, 3.4, -16.0))] {
            let (v, i) = cuboid(lo, hi);
            let base = caster.0.len() as u32;
            caster.0.extend(v);
            caster.1.extend(i.into_iter().map(|t| base + t));
        }
        let eye = Vec3::new(-1.0, 1.6, -4.6);
        let blocked = |p: Vec3| {

            let d = p - pos;
            let (mut t0, mut t1) = (0.0f32, 1.0f32);
            for k in 0..3 {
                if d[k].abs() < 1e-6 {
                    if pos[k] < pmin[k] || pos[k] > pmax[k] {
                        return false;
                    }
                    continue;
                }
                let (a, b) = ((pmin[k] - pos[k]) / d[k], (pmax[k] - pos[k]) / d[k]);
                t0 = t0.max(a.min(b));
                t1 = t1.min(a.max(b));
            }
            t0 <= t1
        };
        for (z, xs) in [(-9.0f32, [-0.6f32, -0.3, 0.3, 0.9]), (-12.0, [-0.6, -0.3, 0.6, 1.2]), (-15.0, [-0.6, 0.0, 1.0, 1.5])] {
            for x in xs {
                let p = Vec3::new(x, 3.399, z);
                let at = |light: Light, caster| Scene {
                    lights: vec![light],
                    caster,
                    receiver_at: p,
                    eye_offset: eye - p,
                    normal: [0.0, -1.0, 0.0],
                    shadow_dim: QUEST_SHADOW_DIM,
                    ..floor_under_the_lamp()
                };
                let k = 30.0;
                let unlit = shot!(at(lamp(0.0), None));
                let open = shot!(at(lamp(k), None));
                let lit = shot!(at(lamp(k), Some(caster.clone())));
                let shadowed = blocked(p);
                eprintln!(
                    "CEILING z {z:5.1} x {x:5.2}: {}  unlit {:3} open {:3} with shadows {:3}",
                    if shadowed { "SHADOW" } else { "lit   " },
                    unlit[0], open[0], lit[0]
                );
                assert!(i32::from(open[0]) > i32::from(unlit[0]) + 8, "the torch does not reach ({x}, {z}): {open:?} against {unlit:?}");
                let want = if shadowed { unlit } else { open };
                assert!(
                    (i32::from(lit[0]) - i32::from(want[0])).abs() <= 3,
                    "ceiling ({x}, {z}) is {} by the rays but reads {lit:?} (open {open:?}, unlit {unlit:?})",
                    if shadowed { "in the pillar's shadow" } else { "lit" },
                );
            }
        }
    }

}

#[cfg(test)]
mod skinned_caster_tests {

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

        for term in [
            "(joints.mats[v.joint_ids.x] * p) * v.joint_weights.x",
            "(joints.mats[v.joint_ids.w] * p) * v.joint_weights.w",
        ] {
            assert!(lit.contains(term), "lit pass changed shape: {term}");
        }
    }

    #[test]
    fn the_joint_count_actually_reaches_the_shader() {

        assert!(
            super::SHADOW_SHADER.contains("MAX_SKIN_JOINTS_PLACEHOLDER"),
            "the placeholder is what the module substitutes",
        );
        let built = super::SHADOW_SHADER
            .replace("MAX_SKIN_JOINTS_PLACEHOLDER", &super::super::mesh::MAX_SKIN_JOINTS.to_string());
        assert!(!built.contains("PLACEHOLDER"), "substitution left a placeholder behind");
    }

}

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

    #[test]
    fn a_chunk_only_one_eye_can_see_is_still_drawn() {

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

        let c = chunk(Vec3::new(-1.0, -1.0, -6.0), Vec3::new(1.0, 1.0, -4.0));
        assert!(!chunk_seen(&c, &[]));
    }
}

#[cfg(test)]
mod spot_near_plane_tests {
    use super::{spot_light_matrix, SPOT_SHADOW_NEAR};
    use glam::{Vec3, Vec4};

    fn near_of(range: f32) -> f32 {
        let m = spot_light_matrix(Vec3::ZERO, Vec3::NEG_Z, 60.0, range);

        let (mut lo, mut hi) = (0.001f32, range.max(0.2));
        for _ in 0..60 {
            let mid = f32::midpoint(lo, hi);
            let c: Vec4 = m * Vec4::new(0.0, 0.0, -mid, 1.0);
            if c.w <= 0.0 || c.z / c.w < 0.0 { lo = mid } else { hi = mid }
        }
        f32::midpoint(lo, hi)
    }

    #[test]
    fn the_near_plane_does_not_depend_on_how_far_the_light_reaches() {

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
