use bytemuck::{Pod, Zeroable};
use super::shadow::SHADOW_MATRICES;
use glam::{Mat4, Quat, Vec3};
use wgpu::*;

use super::lights::LightsUniform;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {

    pub view_proj: [[[f32; 4]; 4]; 2],

    pub inv_view_proj: [[[f32; 4]; 4]; 2],

    pub sun_view_proj: [[f32; 4]; 4],

    pub sun_dynamic_view_proj: [[f32; 4]; 4],

    pub spot_view_proj: [[[f32; 4]; 4]; SHADOW_MATRICES],

    pub camera_pos: [[f32; 4]; 2],

    pub shadow_params: [f32; 4],

    pub sky_params: [f32; 4],

    pub sky_sh: [[f32; 4]; 9],

    pub post_params: [f32; 4],

    pub player_frame: [f32; 4],

    pub probe_params: [f32; 4],

    pub probe_boxes: [[[f32; 4]; 3]; MAX_PROBES],

    pub portal_params: [f32; 4],

    pub probe_portals: [[[f32; 4]; 3]; MAX_PORTALS],

    pub proxy_params: [f32; 4],

    pub probe_proxies: [[[f32; 4]; 3]; MAX_PROXIES],

    pub probe_rooms: [[f32; 4]; ROOM_TABLE_ROWS],

    pub ground_params: [f32; 4],

    pub proxy_fields: [super::proxy_field::FieldSlot; super::proxy_field::MAX_PROXY_FIELDS],

    pub proxy_cards: [[f32; 4]; MAX_PROXIES / 4],

    pub building_boxes: [[f32; 4]; MAX_BUILDINGS * 2],

    pub capsules: [[f32; 4]; MAX_CAPSULES * 2],
    pub capsule_groups: [[f32; 4]; MAX_CAPSULE_GROUPS * 2],

    pub capsule_params: [f32; 4],

    pub character_cards: [[f32; 4]; 2 * super::proxy_cards::CHARACTER_CARD_SETS],

    pub room_exposure: RoomExposure,

    /// The clouds' shadow map for a player-space point p:
    /// uv = (a px + b pz, c px + d pz) + (u0, v0) from `[a, b, c, d]`,
    /// `[u0, v0, 0, 0]`, then `[fill rgb, on]`; read only when `on` > 0.
    /// `fill` is the light a cloud's base gives the ground under it where
    /// it hides the sun whole (`cloud_fill` in the lights block).
    pub cloud_map: [[f32; 4]; 3],
}

/// Texels a side of the clouds' shadow map (`UniformBuffer::write_cloud_shadow`).
pub const CLOUD_SHADOW_SIZE: u32 = 64;

/// Boxes -- closed rooms and their doorways -- drawn at the room's own local
/// adaptation (`room_local_exposure` in the lights block): at most this many,
/// a pair of rows each.
pub const ROOM_EXPOSURE_ROOMS: usize = 8;
/// Per room: `[min.x, min.y, min.z, factor]`, `[max.x, max.y, max.z, band]`,
/// world space; factor 0 is an empty row.
pub type RoomExposure = [[f32; 4]; 2 * ROOM_EXPOSURE_ROOMS];

pub const MAX_BUILDINGS: usize = 8;

pub const MAX_PROXIES: usize = 16;
const _: () = assert!(MAX_PROXIES % 4 == 0, "`proxy_cards` packs four proxies a vec4");

pub const MAX_PORTALS: usize = 8;

pub const MAX_PROBES: usize = 16;

pub const PROBE_SELECT_OFFSET: u64 = std::mem::offset_of!(Uniforms, probe_boxes) as u64;

pub const PROBE_PUSH_SIZE: u64 = (MAX_PROBES * 16) as u64;

pub const PROBE_SELECT_SIZE: u64 = std::mem::size_of::<[[[f32; 4]; 3]; MAX_PROBES]>() as u64;
const _: () = assert!(
    PROBE_SELECT_OFFSET % 256 == 0,
    "a uniform binding's offset must meet the strictest offset alignment a device may ask (256)",
);

pub const SCENE_SAMPLED_TEXTURES: u32 = 24;

pub fn scene_limits(base: wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_sampled_textures_per_shader_stage: base.max_sampled_textures_per_shader_stage.max(SCENE_SAMPLED_TEXTURES),
        ..base
    }
}

#[derive(Clone, Copy, Default)]
pub struct PlayerUpload {
    pub offset: Vec3,
    pub yaw: f32,
    pub capsules: CapsuleUpload,
}

pub const MAX_CAPSULE_GROUPS: usize = 8;

pub const CAPSULES_PER_GROUP: usize = 14;
pub const MAX_CAPSULES: usize = MAX_CAPSULE_GROUPS * CAPSULES_PER_GROUP;

#[derive(Clone, Debug, Default)]
pub struct CapsuleGroup {

    pub capsules: Vec<(Vec3, Vec3, f32)>,

    pub colour: [f32; 3],

    pub surfaces: Vec<f32>,
}

#[derive(Clone, Copy)]
pub struct CapsuleUpload {
    pub capsules: [[f32; 4]; MAX_CAPSULES * 2],
    pub groups: [[f32; 4]; MAX_CAPSULE_GROUPS * 2],
    pub group_count: u32,

    pub shadow_lights: [f32; 2],

    pub floor_mirror: bool,

    pub cards: [[f32; 4]; 2 * super::proxy_cards::CHARACTER_CARD_SETS],
}

impl Default for CapsuleUpload {
    fn default() -> Self {
        Self {
            capsules: [[0.0; 4]; MAX_CAPSULES * 2],
            groups: [[0.0; 4]; MAX_CAPSULE_GROUPS * 2],
            group_count: 0,
            shadow_lights: [-1.0; 2],
            floor_mirror: false,
            cards: [[0.0; 4]; 2 * super::proxy_cards::CHARACTER_CARD_SETS],
        }
    }
}

impl CapsuleUpload {

    pub fn from_groups(groups: &[CapsuleGroup]) -> Self {
        let mut out = Self::default();
        let mut g = 0usize;
        for group in groups {
            if g == MAX_CAPSULE_GROUPS {
                break;
            }

            let caps: Vec<(Vec3, Vec3, f32, f32)> = group
                .capsules
                .iter()
                .zip(group.surfaces.iter().copied().chain(std::iter::repeat(0.0)))
                .map(|(&(a, b, r), s)| (a, b, r, s))
                .filter(|(a, b, r, s)| a.is_finite() && b.is_finite() && *r > 0.0 && s.is_finite())
                .take(CAPSULES_PER_GROUP)
                .collect();
            if caps.is_empty() {
                continue;
            }

            let centre = caps.iter().fold(Vec3::ZERO, |s, (a, b, _, _)| s + *a + *b) / (2 * caps.len()) as f32;
            let reach = caps
                .iter()
                .map(|(a, b, r, _)| (*a - centre).length().max((*b - centre).length()) + r)
                .fold(0.0f32, f32::max);
            for (k, (a, b, r, s)) in caps.iter().enumerate() {
                let i = g * CAPSULES_PER_GROUP + k;
                out.capsules[i * 2] = [a.x, a.y, a.z, *r];
                out.capsules[i * 2 + 1] = [b.x, b.y, b.z, *s];
            }
            out.groups[g * 2] = [centre.x, centre.y, centre.z, reach];
            out.groups[g * 2 + 1] = [group.colour[0], group.colour[1], group.colour[2], caps.len() as f32];
            g += 1;
        }
        out.group_count = g as u32;
        out
    }
}

pub fn sky_in_player_frame(sh: &[[f32; 4]; 9], yaw: f32) -> [[f32; 4]; 9] {
    let light: super::room_light::RoomLight = std::array::from_fn(|i| [sh[i][0], sh[i][1], sh[i][2]]);
    let turned = super::room_light::turned_to_player(&light, yaw);
    std::array::from_fn(|i| [turned[i][0], turned[i][1], turned[i][2], sh[i][3]])
}

#[derive(Clone, Copy)]
pub struct PostUpload {
    pub exposure: f32,
    pub tonemap: super::tonemap::ToneMapping,

    pub terrain_detail_distance: f32,

    pub reflection_share: bool,

    pub night_vision: f32,

    /// The rooms seen at their own adaptation; see `RoomExposure`.
    pub rooms: RoomExposure,

    /// Where the clouds' shadow map lies; see `Uniforms::cloud_map`.
    pub cloud_map: [[f32; 4]; 3],
}

impl Default for PostUpload {

    fn default() -> Self {
        Self {
            exposure: 1.0,
            tonemap: super::tonemap::ToneMapping::default(),
            terrain_detail_distance: 0.0,
            reflection_share: false,
            night_vision: 0.0,
            rooms: [[0.0; 4]; 2 * ROOM_EXPOSURE_ROOMS],
            cloud_map: [[0.0; 4]; 3],
        }
    }
}

#[derive(Clone)]
pub struct SkyUpload {
    pub intensity: f32,
    pub sh: [[f32; 4]; 9],
}

impl SkyUpload {

    pub fn none() -> Self {
        Self::from(&crate::renderer::sky::SkyIrradiance::flat(crate::renderer::sky::AMBIENT))
    }

    pub fn from(irr: &crate::renderer::sky::SkyIrradiance) -> Self {
        let mut sh = [[0.0f32; 4]; 9];
        for i in 0..9 {
            sh[i] = [irr.sh[i][0], irr.sh[i][1], irr.sh[i][2], 0.0];
        }
        Self { intensity: 1.0, sh }
    }
}

pub struct ShadowUpload {
    pub sun_view_proj: Mat4,

    pub sun_dynamic_view_proj: Mat4,

    pub sun_dynamic_enabled: bool,

    pub spot_view_proj: [Mat4; SHADOW_MATRICES],
    pub sun_enabled: bool,

    pub spot_count: u32,
}

impl ShadowUpload {

    pub fn disabled() -> Self {
        Self {
            sun_view_proj: Mat4::IDENTITY,
            sun_dynamic_view_proj: Mat4::IDENTITY,
            sun_dynamic_enabled: false,
            spot_view_proj: [Mat4::IDENTITY; SHADOW_MATRICES],
            sun_enabled: false,
            spot_count: 0,
        }
    }
}

pub struct UniformBuffer {
    pub buffer: Buffer,
    pub layout: BindGroupLayout,
    pub bind_group: BindGroup,

    pub twin_buffer: Buffer,
    pub twin_bind_group: BindGroup,

    pub planar_buffer: Buffer,
    pub planar_bind_group: BindGroup,

    probes: ProbeUpload,

    probe_depth_view: TextureView,
    probe_depth_sampler: Sampler,

    ground_view: TextureView,

    proxy_field_view: TextureView,

    proxy_card_view: TextureView,

    cloud_shadow: Texture,
    cloud_shadow_view: TextureView,
}

impl UniformBuffer {

    pub fn set_probe_depth(&mut self, view: TextureView) {
        self.probe_depth_view = view;
    }

    pub fn set_ground_map(&mut self, view: TextureView) {
        self.ground_view = view;
    }

    pub fn set_proxy_field_atlas(&mut self, view: TextureView) {
        self.proxy_field_view = view;
    }

    pub fn set_proxy_card_atlas(&mut self, view: TextureView) {
        self.proxy_card_view = view;
    }

    pub fn set_probes(&mut self, probes: ProbeUpload) {
        self.probes = probes;
    }

    pub fn probe_push(&self, probes: Option<&ProbeUpload>) -> [[f32; 4]; MAX_PROBES] {
        push_slots(probes.unwrap_or(&self.probes))
    }

    pub fn probes_or_level<'a>(&'a self, probes: Option<&'a ProbeUpload>) -> &'a ProbeUpload {
        probes.unwrap_or(&self.probes)
    }

    pub fn probes(&self) -> ProbeUpload {
        self.probes
    }
}

#[allow(clippy::too_many_arguments)]
fn scene_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    label: &str,
    buffer: &Buffer,
    lights: &LightsUniform,
    sun_shadow_view: &TextureView,
    sun_dynamic_view: &TextureView,
    spot_shadow_view: &TextureView,
    shadow_sampler: &Sampler,
    probe_view: &TextureView,
    probe_sampler: &Sampler,
    probe_depth_view: &TextureView,
    probe_depth_sampler: &Sampler,
    ground_view: &TextureView,
    proxy_field_view: &TextureView,
    proxy_card_view: &TextureView,
    cloud_shadow_view: &TextureView,
) -> BindGroup {
    device.create_bind_group(&BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: lights.buffer().as_entire_binding() },
            BindGroupEntry { binding: 2, resource: BindingResource::TextureView(sun_shadow_view) },
            BindGroupEntry { binding: 3, resource: BindingResource::Sampler(shadow_sampler) },
            BindGroupEntry { binding: 4, resource: BindingResource::TextureView(spot_shadow_view) },
            BindGroupEntry { binding: 5, resource: BindingResource::TextureView(probe_view) },
            BindGroupEntry { binding: 6, resource: BindingResource::Sampler(probe_sampler) },
            BindGroupEntry { binding: 7, resource: BindingResource::TextureView(sun_dynamic_view) },
            BindGroupEntry { binding: 8, resource: BindingResource::TextureView(probe_depth_view) },
            BindGroupEntry { binding: 9, resource: BindingResource::Sampler(probe_depth_sampler) },
            BindGroupEntry { binding: 10, resource: BindingResource::TextureView(ground_view) },
            BindGroupEntry { binding: 11, resource: BindingResource::TextureView(proxy_field_view) },
            BindGroupEntry { binding: 12, resource: BindingResource::TextureView(proxy_card_view) },
            BindGroupEntry { binding: 14, resource: BindingResource::TextureView(cloud_shadow_view) },
            BindGroupEntry {
                binding: 13,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer,
                    offset: PROBE_SELECT_OFFSET,
                    size: std::num::NonZeroU64::new(PROBE_SELECT_SIZE),
                }),
            },
        ],
    })
}

impl UniformBuffer {
    pub fn new(
        device: &Device,
        lights: &LightsUniform,
        sun_shadow_view: &TextureView,
        sun_dynamic_view: &TextureView,
        spot_shadow_view: &TextureView,
        shadow_sampler: &Sampler,
    ) -> Self {

        let (probe_view, probe_sampler) = default_probe_cube(device);
        Self::new_with_probes(
            device, lights, sun_shadow_view, sun_dynamic_view, spot_shadow_view, shadow_sampler,
            &probe_view, &probe_sampler,
        )
    }

    pub fn new_with_probes(
        device: &Device,
        lights: &LightsUniform,
        sun_shadow_view: &TextureView,
        sun_dynamic_view: &TextureView,
        spot_shadow_view: &TextureView,
        shadow_sampler: &Sampler,
        probe_view: &TextureView,
        probe_sampler: &Sampler,
    ) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("uniform_buf"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("uniform_bgl"),

            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Comparison),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,

                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },

                        view_dimension: TextureViewDimension::CubeArray,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,

                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },

                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },

                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::CubeArray,
                        multisampled: false,
                    },
                    count: None,
                },

                BindGroupLayoutEntry {
                    binding: 9,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },

                BindGroupLayoutEntry {
                    binding: 10,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },

                BindGroupLayoutEntry {
                    binding: 11,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },

                BindGroupLayoutEntry {
                    binding: 12,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },

                BindGroupLayoutEntry {
                    binding: 13,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(PROBE_SELECT_SIZE),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 14,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let (probe_depth_view, probe_depth_sampler) = default_probe_depth(device);
        let ground_view = default_ground_map(device);
        let proxy_field_view = super::proxy_field::none(device);
        let proxy_card_view = super::proxy_cards::none(device);
        // Read only once written and switched on (`Uniforms::cloud_map`).
        let cloud_shadow = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud_shadow_map"),
            size: wgpu::Extent3d { width: CLOUD_SHADOW_SIZE, height: CLOUD_SHADOW_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let cloud_shadow_view = cloud_shadow.create_view(&wgpu::TextureViewDescriptor::default());

        let twin_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("uniform_buf_twin"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let planar_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("uniform_buf_planar"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = |buffer: &Buffer, label: &str| {
            scene_bind_group(
                device,
                &layout,
                label,
                buffer,
                lights,
                sun_shadow_view,
                sun_dynamic_view,
                spot_shadow_view,
                shadow_sampler,
                probe_view,
                probe_sampler,
                &probe_depth_view,
                &probe_depth_sampler,
                &ground_view,
                &proxy_field_view,
                &proxy_card_view,
                &cloud_shadow_view,
            )
        };
        let bind_group = group(&buffer, "uniform_bg");
        let twin_bind_group = group(&twin_buffer, "uniform_bg_twin");
        let planar_bind_group = group(&planar_buffer, "uniform_bg_planar");

        Self {
            buffer,
            layout,
            bind_group,
            twin_buffer,
            twin_bind_group,
            planar_buffer,
            planar_bind_group,
            probes: ProbeUpload::default(),
            probe_depth_view,
            probe_depth_sampler,
            ground_view,
            proxy_field_view,
            proxy_card_view,
            cloud_shadow,
            cloud_shadow_view,
        }
    }

    /// The clouds' shadow map, `CLOUD_SHADOW_SIZE` squared, row by row (z),
    /// 0 = no light through.
    pub fn write_cloud_shadow(&self, queue: &Queue, texels: &[u8]) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.cloud_shadow, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            texels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(CLOUD_SHADOW_SIZE), rows_per_image: Some(CLOUD_SHADOW_SIZE) },
            wgpu::Extent3d { width: CLOUD_SHADOW_SIZE, height: CLOUD_SHADOW_SIZE, depth_or_array_layers: 1 },
        );
    }

    pub fn upload(&self, queue: &Queue, view_proj: Mat4, camera_pos: Vec3, shadow: &ShadowUpload) {
        self.upload_with_sky(queue, view_proj, camera_pos, shadow, &SkyUpload::none());
    }

    pub fn upload_with_sky(
        &self,
        queue: &Queue,
        view_proj: Mat4,
        camera_pos: Vec3,
        shadow: &ShadowUpload,
        sky: &SkyUpload,
    ) {
        self.upload_scene(
            queue, view_proj, camera_pos, shadow, sky,
            &PostUpload::default(), &PlayerUpload::default(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload_scene(
        &self,
        queue: &Queue,
        view_proj: Mat4,
        camera_pos: Vec3,
        shadow: &ShadowUpload,
        sky: &SkyUpload,
        post: &PostUpload,
        player: &PlayerUpload,
    ) {
        self.upload_scene_with_probes(queue, view_proj, camera_pos, shadow, sky, post, player, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload_scene_with_probes(
        &self,
        queue: &Queue,
        view_proj: Mat4,
        camera_pos: Vec3,
        shadow: &ShadowUpload,
        sky: &SkyUpload,
        post: &PostUpload,
        player: &PlayerUpload,
        probes: Option<&ProbeUpload>,
    ) {
        self.upload_scene_stereo(
            queue,
            [view_proj; 2],
            [camera_pos; 2],
            shadow,
            sky,
            post,
            player,
            probes,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload_scene_stereo(
        &self,
        queue: &Queue,
        view_proj: [Mat4; 2],
        camera_pos: [Vec3; 2],
        shadow: &ShadowUpload,
        sky: &SkyUpload,
        post: &PostUpload,
        player: &PlayerUpload,
        probes: Option<&ProbeUpload>,
    ) {
        self.write_scene_stereo_to(queue, &self.buffer, view_proj, camera_pos, shadow, sky, post, player, probes);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_scene_stereo_to(
        &self,
        queue: &Queue,
        target: &Buffer,
        view_proj: [Mat4; 2],
        camera_pos: [Vec3; 2],
        shadow: &ShadowUpload,
        sky: &SkyUpload,
        post: &PostUpload,
        player: &PlayerUpload,
        probes: Option<&ProbeUpload>,
    ) {
        let probes = probes.unwrap_or(&self.probes);

        let (dense, room_tables) = probes.dense_rooms();
        let u = Uniforms {

            view_proj: [
                view_proj[0].to_cols_array_2d(),
                view_proj[1].to_cols_array_2d(),
            ],
            inv_view_proj: [
                view_proj[0].inverse().to_cols_array_2d(),
                view_proj[1].inverse().to_cols_array_2d(),
            ],
            sun_view_proj: shadow.sun_view_proj.to_cols_array_2d(),
            sun_dynamic_view_proj: shadow.sun_dynamic_view_proj.to_cols_array_2d(),
            spot_view_proj: std::array::from_fn(|i| shadow.spot_view_proj[i].to_cols_array_2d()),
            camera_pos: [
                [camera_pos[0].x, camera_pos[0].y, camera_pos[0].z, 1.0],
                [camera_pos[1].x, camera_pos[1].y, camera_pos[1].z, 1.0],
            ],
            shadow_params: [
                if shadow.sun_enabled { 1.0 } else { 0.0 },
                shadow.spot_count as f32,

                if shadow.sun_dynamic_enabled { 1.0 } else { 0.0 },
                0.0,
            ],

            sky_params: [
                sky.intensity,
                dense.sky_layer + 1.0,
                dense.ground_top,
                if dense.ground_top > NO_GROUND { 1.0 } else { 0.0 },
            ],

            sky_sh: sky_in_player_frame(&sky.sh, player.yaw),
            player_frame: [player.offset.x, player.offset.y, player.offset.z, player.yaw],

            probe_params: {

                let g = sky_ground_irradiance(&sky.sh);
                [probes.count as f32, g[0], g[1], g[2]]
            },
            probe_boxes: dense.boxes,

            portal_params: [
                probes.portal_count as f32,

                probes.trace_word(),
                dense.outdoor_volume + 1.0,
                probes.building_count as f32,
            ],
            probe_portals: dense.portals,

            proxy_params: [probes.proxy_count as f32, post.night_vision, 0.0, 0.0],
            probe_proxies: dense.proxies,
            probe_rooms: room_tables,
            ground_params: dense.ground,
            proxy_fields: dense.proxy_fields,
            proxy_cards: dense.proxy_cards,
            building_boxes: probes.buildings,
            capsules: player.capsules.capsules,
            capsule_groups: player.capsules.groups,
            capsule_params: [
                player.capsules.group_count as f32,
                player.capsules.shadow_lights[0],
                player.capsules.shadow_lights[1],
                if player.capsules.floor_mirror {
                    super::brush_pipeline::probe_pass::FLOOR_MIRROR_BIAS + super::brush_pipeline::probe_pass::FLOOR_MIRROR_PLANE
                } else {
                    0.0
                },
            ],
            character_cards: player.capsules.cards,
            room_exposure: post.rooms,
            cloud_map: post.cloud_map,
            post_params: [
                post.exposure,
                match post.tonemap {
                    super::tonemap::ToneMapping::Aces => 0.0,
                    super::tonemap::ToneMapping::None => 1.0,
                },
                post.terrain_detail_distance,
                if post.reflection_share { 1.0 } else { 0.0 },
            ],
        };
        queue.write_buffer(target, 0, bytemuck::bytes_of(&u));
    }

}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::renderer::shadow::ShadowMap;

    pub const TEST_EYE: glam::Vec3 = glam::Vec3::new(0.0, 0.0, 5.0);

    pub fn scene_uniforms(device: &Device, lights: &LightsUniform) -> (ShadowMap, UniformBuffer) {
        let shadows = ShadowMap::with_dimension(device, 64);
        let uniforms = UniformBuffer::new(
            device,
            lights,
            shadows.sun_depth_view(),
            shadows.sun_dynamic_depth_view(),
            shadows.spot_depth_view(),
            shadows.sampler(),
        );
        (shadows, uniforms)
    }

    fn test_f16(v: f32) -> u16 {
        let bits = v.to_bits();
        let exp = ((bits >> 23) & 0xff) as i32 - 127;
        let mant = bits & 0x007f_ffff;
        if v == 0.0 {
            return 0;
        }
        if exp < -14 {
            let shift = (-14 - exp) as u32;
            if shift > 24 {
                return 0;
            }
            return ((mant | 0x0080_0000) >> (shift + 13)) as u16;
        }
        (((exp + 15) as u16) << 10) | ((mant >> 13) as u16)
    }

    pub fn scene_uniforms_with_probe(
        device: &Device,
        queue: &Queue,
        lights: &LightsUniform,
        colour: [u8; 4],
        min: glam::Vec3,
        max: glam::Vec3,
    ) -> (ShadowMap, UniformBuffer, TextureView, Sampler) {
        const RES: u32 = 4;
        let shadows = ShadowMap::with_dimension(device, 64);
        let mut uniforms = UniformBuffer::new(
            device,
            lights,
            shadows.sun_depth_view(),
            shadows.sun_dynamic_depth_view(),
            shadows.spot_depth_view(),
            shadows.sampler(),
        );

        let texel: Vec<u8> = (0..4)
            .flat_map(|c| {
                let u = colour[c] as f32 / 255.0;

                let linear = if c == 3 {
                    u
                } else if u <= 0.04045 {
                    u / 12.92
                } else {
                    ((u + 0.055) / 1.055).powf(2.4)
                };
                test_f16(linear).to_le_bytes()
            })
            .collect();
        let face: Vec<u8> = texel
            .iter()
            .copied()
            .cycle()
            .take((RES * RES * 8 * 6) as usize)
            .collect();
        let view = super::upload_probe_cubes(device, queue, RES, &[&face]);
        let (_unused, sampler) = super::default_probe_cube(device);
        let centre = (min + max) * 0.5;
        let mut probes = ProbeUpload { count: 1, ..Default::default() };
        probes.boxes[0] = [
            [centre.x, centre.y, centre.z, 0.0],
            [min.x, min.y, min.z, 0.0],
            [max.x, max.y, max.z, 0.0],
        ];
        uniforms.rebind_probes(
            device,
            lights,
            shadows.sun_depth_view(),
            shadows.sun_dynamic_depth_view(),
            shadows.spot_depth_view(),
            shadows.sampler(),
            &view,
            &sampler,
            probes,
        );
        (shadows, uniforms, view, sampler)
    }
}

fn sky_ground_irradiance(sh: &[[f32; 4]; 9]) -> [f32; 3] {

    let terms = [
        (0usize, 0.282095_f32 * 1.0),
        (1, 0.488603 * 0.6666667),
        (6, -0.315392 * 0.25),
        (8, 0.546274 * (0.0 - 1.0) * 0.25),
    ];
    let mut out = [0.0f32; 3];
    for (i, w) in terms {
        for c in 0..3 {
            out[c] += sh[i][c] * w;
        }
    }

    [out[0].max(0.0), out[1].max(0.0), out[2].max(0.0)]
}

#[derive(Clone, Copy)]
pub struct ProbeUpload {

    pub count: u32,

    pub boxes: [[[f32; 4]; 3]; MAX_PROBES],

    pub portal_count: u32,

    pub portals: [[[f32; 4]; 3]; MAX_PORTALS],

    pub no_trace: bool,

    pub open_ground: bool,

    pub proxy_count: u32,

    pub proxies: [[[f32; 4]; 3]; MAX_PROXIES],

    pub outdoor_volume: f32,

    pub sky_layer: f32,

    pub ground: [f32; 4],
    pub ground_top: f32,

    pub proxy_fields: [super::proxy_field::FieldSlot; super::proxy_field::MAX_PROXY_FIELDS],

    pub proxy_cards: [[f32; 4]; MAX_PROXIES / 4],

    pub buildings: [[f32; 4]; MAX_BUILDINGS * 2],
    pub building_count: u32,
}

pub const NO_GROUND: f32 = -1.0e30;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeProxy {
    pub centre: Vec3,
    pub half_size: Vec3,
    pub rotation: Quat,

    pub volume: u32,

    pub solid: bool,

    pub field: Option<u32>,

    pub cards: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbePortal {

    pub min: Vec3,
    pub max: Vec3,

    pub axis: u32,

    pub low: u32,
    pub high: u32,

    pub wall: Option<(f32, f32)>,
}

impl ProbeUpload {

    pub fn set_portals(&mut self, portals: &[ProbePortal], player: Vec3, volumes: &[u32]) {
        let mut near: Vec<(f32, &ProbePortal)> = portals
            .iter()
            .filter(|p| volumes.contains(&p.low) || volumes.contains(&p.high))
            .map(|p| (p.min.max(player.min(p.max)).distance_squared(player), p))
            .collect();
        near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.portal_count = near.len().min(MAX_PORTALS) as u32;
        for (i, (_, p)) in near.iter().take(MAX_PORTALS).enumerate() {
            self.portals[i] = [
                [p.min.x, p.min.y, p.min.z, p.axis as f32],
                [p.max.x, p.max.y, p.max.z, p.low as f32],
                {
                    let a = p.axis as usize;
                    let (lo, hi) = p.wall.unwrap_or((p.min[a], p.max[a]));
                    [p.high as f32, lo, hi, 0.0]
                },
            ];
        }
    }

    pub fn set_buildings(&mut self, boxes: &[(glam::Vec3, glam::Vec3)], first_layer: Option<u32>) {
        self.buildings = [[0.0; 4]; MAX_BUILDINGS * 2];
        self.building_count = 0;
        let Some(first) = first_layer else { return };
        let mut order: Vec<usize> = (0..boxes.len()).collect();
        order.sort_by(|&a, &b| {
            let size = |i: usize| (boxes[i].1 - boxes[i].0).max(glam::Vec3::ZERO).element_product();
            size(b).total_cmp(&size(a))
        });
        for (slot, &i) in order.iter().take(MAX_BUILDINGS).enumerate() {
            let (lo, hi) = boxes[i];
            self.buildings[slot * 2] = [lo.x, lo.y, lo.z, (first + i as u32) as f32];
            self.buildings[slot * 2 + 1] = [hi.x, hi.y, hi.z, 0.0];
            self.building_count += 1;
        }
    }

    pub fn set_outdoors(&mut self, outdoor_volume: Option<u32>, sky_layer: Option<u32>, ground: Option<([f32; 4], f32)>) {
        self.outdoor_volume = outdoor_volume.map_or(-1.0, |v| v as f32);
        self.sky_layer = sky_layer.map_or(-1.0, |l| l as f32);
        let (g, top) = ground.unwrap_or(([0.0; 4], NO_GROUND));
        self.ground = g;
        self.ground_top = top;
    }

    pub fn set_proxy_fields(&mut self, slots: &[super::proxy_field::FieldSlot]) {
        for (i, s) in slots.iter().take(super::proxy_field::MAX_PROXY_FIELDS).enumerate() {
            self.proxy_fields[i] = *s;
        }
    }

    pub fn set_proxies(&mut self, proxies: &[ProbeProxy], player: Vec3, volumes: &[u32]) {
        let mut near: Vec<(f32, &ProbeProxy)> = proxies
            .iter()
            .filter(|p| volumes.contains(&p.volume))
            .map(|p| (p.centre.distance_squared(player), p))
            .collect();
        near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.proxy_count = near.len().min(MAX_PROXIES) as u32;
        self.proxy_cards = [[0.0; 4]; MAX_PROXIES / 4];
        for (i, (_, p)) in near.iter().take(MAX_PROXIES).enumerate() {
            self.proxy_cards[i / 4][i % 4] = p.cards.map_or(0.0, |row| row as f32 + 1.0);
            let q = p.rotation.normalize();
            self.proxies[i] = [
                [p.centre.x, p.centre.y, p.centre.z, p.volume as f32],

                [
                    p.half_size.x,
                    p.half_size.y,
                    p.half_size.z,
                    match (p.solid, p.field) {
                        (true, _) => 0.0,
                        (false, None) => 1.0,
                        (false, Some(f)) => 2.0 + f as f32,
                    },
                ],
                [q.x, q.y, q.z, q.w],
            ];
        }
    }

    pub fn set_volume(&mut self, slot: usize, volume: u32) {
        if let Some(b) = self.boxes.get_mut(slot) {
            b[2][3] = volume as f32;
        }
    }

    pub fn volume(&self, slot: usize) -> u32 {
        self.boxes.get(slot).map(|b| b[2][3] as u32).unwrap_or(0)
    }

    pub fn set(&mut self, slot: usize, layer: u32, centre: Vec3, min: Vec3, max: Vec3) {
        if slot >= MAX_PROBES {
            return;
        }
        self.boxes[slot] = [
            [centre.x, centre.y, centre.z, layer as f32],
            [min.x, min.y, min.z, 0.0],
            [max.x, max.y, max.z, 0.0],
        ];
    }

    pub fn set_brightness(&mut self, slot: usize, brightness: f32) {
        if let Some(b) = self.boxes.get_mut(slot) {
            b[1][3] = brightness;
        }
    }

    pub fn fill_brightness(&mut self, by_layer: &[f32]) {
        for slot in 0..(self.count as usize).min(MAX_PROBES) {
            let layer = self.layer(slot) as usize;
            self.set_brightness(slot, by_layer.get(layer).copied().unwrap_or(0.0));
        }
    }

    pub fn fill_volumes(&mut self, by_probe: &[u32]) {
        for slot in 0..(self.count as usize).min(MAX_PROBES) {
            let probe = self.layer(slot) as usize;
            self.set_volume(slot, by_probe.get(probe).copied().unwrap_or(u32::MAX));
        }
    }

    pub fn volumes(&self) -> Vec<u32> {
        (0..(self.count as usize).min(MAX_PROBES)).map(|s| self.volume(s)).collect()
    }

    pub fn layer(&self, slot: usize) -> u32 {
        self.boxes.get(slot).map(|b| b[0][3] as u32).unwrap_or(0)
    }

    pub fn dense_rooms(&self) -> (ProbeUpload, [[f32; 4]; ROOM_TABLE_ROWS]) {
        const _: () = assert!(
            MAX_PROBES <= 16 && MAX_PORTALS * 2 <= 16 && MAX_PROXIES <= 16,
            "each block of the room tables holds sixteen entries",
        );
        let mut out = *self;
        let mut tables = [[-1.0f32; 4]; ROOM_TABLE_ROWS];
        let put = |tables: &mut [[f32; 4]; ROOM_TABLE_ROWS], row: usize, k: usize, v: usize| {
            tables[row + k / 4][k % 4] = v as f32;
        };
        let mut rooms: Vec<f32> = Vec::new();
        let mut last: Vec<usize> = Vec::new();
        for slot in 0..(self.count as usize).min(MAX_PROBES) {
            let raw = self.boxes[slot][2][3];
            let room = match rooms.iter().position(|&r| r == raw) {
                Some(room) => {
                    put(&mut tables, 4, last[room], slot);
                    last[room] = slot;
                    room
                }
                None => {
                    rooms.push(raw);
                    last.push(slot);
                    let room = rooms.len() - 1;
                    put(&mut tables, 0, room, slot);
                    room
                }
            };
            out.boxes[slot][2][3] = room as f32;
        }
        let renumber = |raw: f32| rooms.iter().position(|&r| r == raw);

        let mut last_portal: Vec<Option<usize>> = vec![None; rooms.len()];
        for p in 0..(self.portal_count as usize).min(MAX_PORTALS) {
            let low = renumber(self.portals[p][1][3]);
            let high = renumber(self.portals[p][2][0]);
            out.portals[p][1][3] = low.map_or(NO_RESIDENT_ROOM, |r| r as f32);
            out.portals[p][2][0] = high.map_or(NO_RESIDENT_ROOM, |r| r as f32);
            for room in [low, high.filter(|&h| Some(h) != low)].into_iter().flatten() {
                match last_portal[room] {
                    None => put(&mut tables, 8, room, p),
                    Some(prev) => {
                        let side = if out.portals[prev][1][3] == room as f32 { 0 } else { 1 };
                        put(&mut tables, 12, 2 * prev + side, p);
                    }
                }
                last_portal[room] = Some(p);
            }
        }

        out.outdoor_volume = if self.outdoor_volume < 0.0 {
            -1.0
        } else {
            renumber(self.outdoor_volume).map_or(-1.0, |r| r as f32)
        };
        let mut last_proxy: Vec<Option<usize>> = vec![None; rooms.len()];
        for i in 0..(self.proxy_count as usize).min(MAX_PROXIES) {
            let room = renumber(self.proxies[i][0][3]);
            out.proxies[i][0][3] = room.map_or(NO_RESIDENT_ROOM, |r| r as f32);
            if let Some(room) = room {
                match last_proxy[room] {
                    None => put(&mut tables, 16, room, i),
                    Some(prev) => put(&mut tables, 20, prev, i),
                }
                last_proxy[room] = Some(i);
            }
        }
        (out, tables)
    }
}

pub const NO_RESIDENT_ROOM: f32 = 1_000_000.0;

pub const ROOM_TABLE_ROWS: usize = 24;

impl ProbeUpload {

    pub fn trace_word(&self) -> f32 {
        if self.no_trace {
            1.0
        } else if self.open_ground {
            -1.0
        } else {
            0.0
        }
    }
}

impl Default for ProbeUpload {
    fn default() -> Self {
        Self {
            count: 0,
            boxes: [[[0.0; 4]; 3]; MAX_PROBES],
            portal_count: 0,
            portals: [[[0.0; 4]; 3]; MAX_PORTALS],
            no_trace: false,
            open_ground: false,
            proxy_count: 0,
            proxies: [[[0.0; 4]; 3]; MAX_PROXIES],
            outdoor_volume: -1.0,
            sky_layer: -1.0,
            ground: [0.0; 4],
            ground_top: NO_GROUND,
            proxy_fields: [[[0.0; 4]; 3]; super::proxy_field::MAX_PROXY_FIELDS],
            proxy_cards: [[0.0; 4]; MAX_PROXIES / 4],
            buildings: [[0.0; 4]; MAX_BUILDINGS * 2],
            building_count: 0,
        }
    }
}

pub fn default_probe_cube(device: &Device) -> (TextureView, Sampler) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("default_probe_cube"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,

        format: TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("default_probe_cube_view"),
        dimension: Some(TextureViewDimension::CubeArray),
        ..Default::default()
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("probe_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    (view, sampler)
}

pub fn default_ground_map(device: &Device) -> TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("default_ground_map"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

pub fn default_probe_depth(device: &Device) -> (TextureView, Sampler) {
    let tex = device.create_texture(&probe_depth_descriptor(1, 1));
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("default_probe_depth_view"),
        dimension: Some(TextureViewDimension::CubeArray),
        ..Default::default()
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("probe_depth_sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    (view, sampler)
}

pub fn probe_depth_descriptor(res: u32, probes: u32) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("probe_depth_array"),
        size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 6 * probes.max(1) },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }
}

pub fn write_probe_depth_layer(queue: &Queue, tex: &wgpu::Texture, layer: u32, res: u32, depth: &[u16]) {
    let bytes: Vec<u8> = depth.iter().flat_map(|h| h.to_le_bytes()).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: layer * 6 },
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8 * res), rows_per_image: Some(res) },
        wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 6 },
    );
}

pub fn probe_cube_descriptor(res: u32, probes: u32) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("probe_cube_array"),
        size: wgpu::Extent3d {
            width: res,
            height: res,

            depth_or_array_layers: 6 * probes.max(1),
        },

        mip_level_count: probe_mip_levels(res),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,

        format: TextureFormat::Rgba16Float,

        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }
}

pub fn probe_mip_levels(resolution: u32) -> u32 {
    32 - resolution.max(1).leading_zeros()
}

pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = -14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((e + 127) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | (((exp - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub(crate) fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mant = bits & 0x007f_ffff;
    if exp > 15 {
        return sign | if exp == 128 && mant != 0 { 0x7e00 } else { 0x7c00 };
    }
    if exp < -14 {
        let shift = (-14 - exp) as u32;

        if shift > 10 {
            return sign;
        }
        return sign | ((mant | 0x0080_0000) >> (shift + 13)) as u16;
    }
    sign | (((exp + 15) as u16) << 10) | ((mant >> 13) as u16)
}

#[cfg(test)]
mod f16_tests {
    #[test]
    fn tiny_values_become_zero_or_a_subnormal_never_garbage() {
        for v in [1e-8f32, 1e-9, 1e-20, f32::MIN_POSITIVE] {
            assert_eq!(super::f32_to_f16(v), 0, "{v}");
        }

        assert_eq!(super::f32_to_f16(2f32.powi(-20)), 1 << 4);
        assert_eq!(super::f32_to_f16(2f32.powi(-24)), 1);
        assert_eq!(super::f32_to_f16(1.0), 0x3c00);
    }
}

pub fn probe_mean_radiance(faces: &[u8], res: u32) -> f32 {
    let res = res.max(1);
    if faces.len() < (res * res * 6 * 8) as usize {
        return 0.0;
    }
    let cube = decode_probe_faces(faces, res);
    let (mut sum, mut weight) = (0.0f64, 0.0f64);
    for [r, g, b, a] in cube.texels {
        let a = a.clamp(0.0, 1.0) as f64;
        sum += (0.2126 * r + 0.7152 * g + 0.0722 * b) as f64 * a;
        weight += a;
    }
    if weight > 0.0 {
        (sum / weight) as f32
    } else {
        0.0
    }
}

pub(crate) fn decode_probe_texels(faces: &[u8], res: u32) -> Option<Vec<[f32; 4]>> {
    let res = res.max(1);
    if faces.len() < (res * res * 6 * 8) as usize {
        return None;
    }
    Some(decode_probe_faces(faces, res).texels)
}

fn decode_probe_faces(src: &[u8], res: u32) -> super::probe_prefilter::CubeLevel {
    let n = (res * res * 6) as usize;
    let texels = (0..n)
        .map(|t| {
            let mut px = [0.0f32; 4];
            for (c, v) in px.iter_mut().enumerate() {
                let i = (t * 4 + c) * 2;
                *v = f16_to_f32(u16::from_le_bytes([src[i], src[i + 1]]));
            }
            px
        })
        .collect();
    super::probe_prefilter::CubeLevel { res, texels }
}

fn encode_probe_level(level: &super::probe_prefilter::CubeLevel) -> Vec<u8> {
    let mut out = Vec::with_capacity(level.texels.len() * 8);
    for px in &level.texels {
        for v in px {
            out.extend_from_slice(&f32_to_f16(*v).to_le_bytes());
        }
    }
    out
}

pub fn upload_probe_cubes(
    device: &Device,
    queue: &Queue,
    resolution: u32,
    faces: &[&[u8]],
) -> TextureView {
    let tex = upload_probe_cube_texture(device, queue, resolution, faces);
    tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("probe_cube_array_view"),
        dimension: Some(TextureViewDimension::CubeArray),
        ..Default::default()
    })
}

pub fn upload_probe_cube_texture(
    device: &Device,
    queue: &Queue,
    resolution: u32,
    faces: &[&[u8]],
) -> wgpu::Texture {
    let res = resolution.max(1);

    let fit = probe_layers_allowed(device);
    if faces.len() > fit as usize {
        log::warn!(
            "probe cube array: {} probes, room for {fit} on this device; the rest are dropped",
            faces.len(),
        );
    }
    let layers = (faces.len() as u32).min(fit).max(1);
    let tex = device.create_texture(&probe_cube_descriptor(res, layers));
    for (i, probe) in faces.iter().take(layers as usize).enumerate() {
        if let Some(chain) = prefilter_probe(probe, res) {
            write_probe_layer(queue, &tex, i as u32, res, &chain);
        }
    }
    tex
}

pub fn probe_layers_allowed(device: &Device) -> u32 {
    (device.limits().max_texture_array_layers / 6).max(1)
}

pub fn prefilter_probe(faces: &[u8], res: u32) -> Option<Vec<Vec<u8>>> {
    let per_face = (res * res * 8) as usize;
    if faces.len() < per_face * 6 {
        return None;
    }
    let chain = super::probe_prefilter::prefiltered_chain(decode_probe_faces(faces, res));
    Some(chain.iter().take(probe_mip_levels(res) as usize).map(encode_probe_level).collect())
}

pub fn write_probe_layer(queue: &Queue, tex: &wgpu::Texture, layer: u32, res: u32, chain: &[Vec<u8>]) {
    for (level, level_data) in chain.iter().enumerate() {
        let level_res = (res >> level).max(1);
        let per = (level_res * level_res * 8) as usize;
        for face in 0..6usize {
            let start = face * per;
            let Some(slice) = level_data.get(start..start + per) else { continue };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer * 6 + face as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                slice,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8 * level_res),
                    rows_per_image: Some(level_res),
                },
                wgpu::Extent3d { width: level_res, height: level_res, depth_or_array_layers: 1 },
            );
        }
    }
}

impl UniformBuffer {

    #[allow(clippy::too_many_arguments)]
    pub fn rebind_probes(
        &mut self,
        device: &Device,
        lights: &LightsUniform,
        sun_shadow_view: &TextureView,
        sun_dynamic_view: &TextureView,
        spot_shadow_view: &TextureView,
        shadow_sampler: &Sampler,
        probe_view: &TextureView,
        probe_sampler: &Sampler,
        probes: ProbeUpload,
    ) {
        let group = |buffer: &Buffer, label: &str| {
            scene_bind_group(
                device,
                &self.layout,
                label,
                buffer,
                lights,
                sun_shadow_view,
                sun_dynamic_view,
                spot_shadow_view,
                shadow_sampler,
                probe_view,
                probe_sampler,
                &self.probe_depth_view,
                &self.probe_depth_sampler,
                &self.ground_view,
                &self.proxy_field_view,
                &self.proxy_card_view,
                &self.cloud_shadow_view,
            )
        };
        self.bind_group = group(&self.buffer, "uniform_bg");
        self.twin_bind_group = group(&self.twin_buffer, "uniform_bg_twin");
        self.planar_bind_group = group(&self.planar_buffer, "uniform_bg_planar");
        self.probes = probes;
    }
}

#[cfg(test)]
mod ground_irradiance_tests {

    use super::*;

    fn shader_sky_irradiance(sh: &[[f32; 4]; 9], n: [f32; 3]) -> [f32; 3] {
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
        e
    }

    fn sample_sky() -> [[f32; 4]; 9] {

        let mut sh = [[0.0f32; 4]; 9];
        for (i, row) in sh.iter_mut().enumerate() {
            let f = i as f32;
            *row = [0.9 - 0.07 * f, 0.5 + 0.03 * f, 0.2 + 0.05 * f, 0.0];
        }
        sh
    }

    #[test]
    fn the_sky_light_stays_with_the_world_when_the_player_turns() {
        let sh = sample_sky();
        for yaw in [0.0f32, 0.4, std::f32::consts::FRAC_PI_2, 2.2, -1.3] {
            let turned = sky_in_player_frame(&sh, yaw);
            for n in [Vec3::X, Vec3::Z, Vec3::NEG_X, Vec3::new(0.6, 0.3, -0.74), Vec3::new(-0.2, -0.9, 0.39)] {
                let n = n.normalize();
                let world = glam::Quat::from_rotation_y(yaw) * n;
                let (got, want) = (shader_sky_irradiance(&turned, n.to_array()), shader_sky_irradiance(&sh, world.to_array()));
                for c in 0..3 {
                    assert!((got[c] - want[c]).abs() < 1e-4, "yaw {yaw}, normal {n}: {got:?} vs {want:?}");
                }
            }
        }

        assert_eq!(sky_ground_irradiance(&sky_in_player_frame(&sh, 1.1)).map(|v| (v * 1e4).round()), sky_ground_irradiance(&sh).map(|v| (v * 1e4).round()));
    }

    #[test]
    fn the_cpu_shortcut_matches_the_full_series_straight_up() {
        let sh = sample_sky();
        let fast = sky_ground_irradiance(&sh);
        let full = shader_sky_irradiance(&sh, [0.0, 1.0, 0.0]);
        for c in 0..3 {
            assert!(
                (fast[c] - full[c].max(0.0)).abs() < 1e-5,
                "channel {c}: shortcut {} vs full evaluation {}",
                fast[c],
                full[c],
            );
        }
    }

    #[test]
    fn a_ringing_sky_cannot_produce_negative_ground() {
        let mut sh = [[0.0f32; 4]; 9];
        sh[0] = [0.05, 0.05, 0.05, 0.0];
        sh[1] = [-3.0, -3.0, -3.0, 0.0];
        let g = sky_ground_irradiance(&sh);
        assert!(g.iter().all(|v| *v >= 0.0), "ground came out negative: {g:?}");
    }

    #[test]
    fn a_black_sky_has_no_ground() {
        assert_eq!(sky_ground_irradiance(&[[0.0; 4]; 9]), [0.0; 3]);
    }
}

#[cfg(test)]
mod probe_capacity_tests {
    use super::*;

    const FACE_BYTES: usize = 64 * 64 * 8;

    #[test]
    fn the_cube_array_stays_small_enough_to_be_free() {
        let bytes = FACE_BYTES * 6 * MAX_PROBES;
        assert!(
            bytes < 8 * 1024 * 1024,
            "the probe cube array is now {} MB, which is no longer a rounding \
             error on a headset; either drop the cap or drop the resolution",
            bytes / (1024 * 1024),
        );
    }

    #[test]
    fn the_uniform_still_fits_a_guaranteed_binding() {
        let size = std::mem::size_of::<Uniforms>();
        assert!(
            size <= 64 * 1024,
            "the scene uniform is {size} bytes; past 64 KiB it stops being \
             portable and starts depending on what the device reports",
        );
    }

    #[test]
    fn every_slot_the_shader_can_index_exists() {
        let probes = ProbeUpload { count: MAX_PROBES as u32, ..Default::default() };
        assert_eq!(probes.boxes.len(), MAX_PROBES);
        assert!(
            probes.count as usize <= probes.boxes.len(),
            "the shader would walk {} probes with {} boxes uploaded",
            probes.count,
            probes.boxes.len(),
        );
    }

    #[test]
    fn probes_past_the_cap_are_dropped_rather_than_aliased() {
        let mut probes = ProbeUpload { count: (MAX_PROBES + 5) as u32, ..Default::default() };
        probes.count = probes.count.min(MAX_PROBES as u32);
        assert_eq!(probes.count as usize, MAX_PROBES);
    }
}

pub fn select_resident_probes(
    volumes: &[(u32, Vec3, Vec3, Vec3)],
    player: Vec3,
    visible: impl Fn(Vec3, Vec3) -> bool,
) -> ProbeUpload {
    let mut scored: Vec<(u8, f32, usize)> = volumes
        .iter()
        .enumerate()
        .map(|(i, (_, centre, min, max))| {
            let inside = player.cmpge(*min).all() && player.cmple(*max).all();

            let rank = if inside {
                0
            } else if visible(*min, *max) {
                1
            } else {
                2
            };
            (rank, centre.distance_squared(player), i)
        })
        .collect();
    scored.sort_by(|a, b| {
        a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut upload = ProbeUpload {
        count: scored.len().min(MAX_PROBES) as u32,
        ..Default::default()
    };
    for (slot, &(_, _, i)) in scored.iter().take(MAX_PROBES).enumerate() {
        let (layer, centre, min, max) = volumes[i];
        upload.set(slot, layer, centre, min, max);
    }
    upload
}

pub fn push_slots(probes: &ProbeUpload) -> [[f32; 4]; MAX_PROBES] {
    let (dense, _) = probes.dense_rooms();
    std::array::from_fn(|i| {
        let b = dense.boxes[i];
        [b[0][0], b[0][1], b[0][2], b[2][3]]
    })
}

#[cfg(test)]
mod residency_tests {
    use super::*;

    #[test]
    fn the_push_slots_are_the_frames_resident_slots() {
        let vols: Vec<_> = (0..(MAX_PROBES as u32 + 4)).map(|i| cell(i, i as f32 * 3.0)).collect();
        let here = select_resident_probes(&vols, Vec3::new(1.0, 1.0, 1.0), |_, _| true);
        let there = select_resident_probes(&vols, Vec3::new(55.0, 1.0, 1.0), |_, _| true);
        for up in [&here, &there] {
            let push = push_slots(up);
            for slot in 0..up.count as usize {
                assert_eq!(&push[slot][..3], &up.boxes[slot][0][..3], "slot {slot}'s centre");
            }
        }
        assert_ne!(push_slots(&here), push_slots(&there), "two places, one table: the test proves nothing");
    }

    fn cell(layer: u32, x: f32) -> (u32, Vec3, Vec3, Vec3) {
        let min = Vec3::new(x, 0.0, 0.0);
        let max = min + Vec3::new(2.0, 2.0, 2.0);
        (layer, (min + max) * 0.5, min, max)
    }

    #[test]
    fn the_cell_around_the_player_is_never_dropped() {

        let player = Vec3::new(1.0, 1.0, 1.0);
        let big = (99u32, Vec3::new(50.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 0.0), Vec3::new(100.0, 2.0, 2.0));
        let mut vols: Vec<_> = (0..MAX_PROBES as u32).map(|i| cell(i, 3.0 + i as f32 * 3.0)).collect();
        vols.push(big);

        let up = select_resident_probes(&vols, player, |_, _| true);
        assert_eq!(
            up.layer(0), 99,
            "the cell the player is standing in lost its slot to nearer cells",
        );

        let up = select_resident_probes(&vols, player, |_, _| false);
        assert_eq!(up.layer(0), 99, "the player's own cell was frustum culled");
    }

    #[test]
    fn visible_and_near_are_preferred() {
        let vols = vec![
            cell(0, 40.0),
            cell(1, 10.0),
            cell(2, 20.0),
        ];

        let up = select_resident_probes(&vols, Vec3::ZERO, |min, _| min.x > 15.0);
        assert_eq!(up.layer(0), 2, "the nearer visible cell should come first");
        assert_eq!(up.layer(1), 0, "the further visible cell should come second");
        assert_eq!(up.layer(2), 1, "the invisible cell should come last");
    }

    #[test]
    fn the_layer_survives_into_the_slot() {
        let vols = vec![cell(7, 0.0), cell(23, 50.0)];
        let up = select_resident_probes(&vols, Vec3::new(1.0, 1.0, 1.0), |_, _| true);
        assert_eq!(up.layer(0), 7);
        assert_eq!(up.layer(1), 23);

        assert_eq!(up.boxes[0][1][0], 0.0);
        assert_eq!(up.boxes[1][1][0], 50.0);
    }

    #[test]
    fn the_slot_count_never_exceeds_the_loop() {
        let vols: Vec<_> = (0..(2 * MAX_PROBES) as u32).map(|i| cell(i, i as f32 * 3.0)).collect();
        let up = select_resident_probes(&vols, Vec3::ZERO, |_, _| true);
        assert_eq!(up.count as usize, MAX_PROBES);
        assert!(up.count as usize <= up.boxes.len());
    }

    #[test]
    fn no_probes_is_no_slots() {
        let up = select_resident_probes(&[], Vec3::ZERO, |_, _| true);
        assert_eq!(up.count, 0);
    }
}

#[cfg(test)]
mod probe_mip_tests {
    use super::*;

    #[test]
    fn a_64px_cube_has_levels_down_to_one_texel() {
        assert_eq!(probe_mip_levels(64), 7);
        assert_eq!(probe_mip_levels(1), 1);
        assert_eq!(probe_mip_levels(0), 1, "a zero-size probe must not ask for zero levels");
    }

    #[test]
    fn the_cube_is_declared_with_the_levels_the_shader_asks_for() {
        for res in [1u32, 8, 32, 64, 128] {
            let d = probe_cube_descriptor(res, 4);
            assert_eq!(
                d.mip_level_count,
                probe_mip_levels(res),
                "a {res}px probe cube was declared with {} level(s); a rough \
                 surface would clamp back to the sharpest texel and glint",
                d.mip_level_count,
            );
        }

        assert!(probe_mip_levels(128) as f32 > 7.0);
    }

    #[test]
    fn the_roughness_ramp_stays_inside_the_chain() {

        let levels = probe_mip_levels(128);
        let max_lod = (levels - 1) as f32;
        let code = super::super::lights::wgsl_lights_block(0, 1);
        assert!(
            code.contains(&format!("const PROBE_MAX_LOD: f32 = {max_lod:.1};")),
            "the shader caps probe LOD somewhere other than the top of a \
             {levels}-level chain",
        );
    }

    #[test]
    fn the_source_pyramid_averages_radiance() {
        let res = 2u32;
        let mut src = Vec::new();

        for _ in 0..6 {
            for v in [0.0f32, 1.0, 2.0, 3.0] {
                for c in 0..4 {
                    let x = if c == 3 { 1.0 } else if c == 0 { v } else { 0.0 };
                    src.extend_from_slice(&f32_to_f16(x).to_le_bytes());
                }
            }
        }
        let pyramid = super::super::probe_prefilter::box_pyramid(decode_probe_faces(&src, res));
        assert_eq!(pyramid.len(), 2, "a 2x2 cube did not halve to 1x1");
        assert_eq!(pyramid[1].texels.len(), 6);
        let [r, _, _, a] = pyramid[1].texels[0];
        assert!((r - 1.5).abs() < 1e-2, "mean of 0,1,2,3 came back as {r}, not 1.5");
        assert!((a - 1.0).abs() < 1e-2, "coverage was not preserved: {a}");

        let bytes = encode_probe_level(&pyramid[1]);
        assert_eq!(bytes.len(), 6 * 8);
        assert!((f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])) - 1.5).abs() < 1e-2);
    }

    #[test]
    fn half_floats_survive_the_chain() {
        for &v in &[0.0f32, 1e-4, 0.001, 0.5, 1.0, 3.7, 60.0] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() <= v * 1e-3 + 6e-8, "{v} became {back}");
        }
    }
}

#[cfg(test)]
mod probe_mip_upload_tests {
    use super::*;

    fn headless_gpu() -> Option<(Device, Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            apply_limit_buckets: false,
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: scene_limits(wgpu::Limits::default()),
            ..Default::default()
        }))
        .ok()
    }

    #[test]
    fn every_mip_level_receives_its_data() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        const RES: u32 = 8;
        let one = f32_to_f16(1.0).to_le_bytes();
        let texel: Vec<u8> = [one, one, one, one].concat();
        let faces: Vec<u8> =
            texel.iter().copied().cycle().take((RES * RES * 8 * 6) as usize).collect();
        let tex = upload_probe_cube_texture(&device, &queue, RES, &[&faces]);

        for level in 0..probe_mip_levels(RES) {
            let side = (RES >> level).max(1);

            let row = ((side * 8 + 255) / 256) * 256;
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("probe_mip_readback"),
                size: (row * side) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(side),
                    },
                },
                wgpu::Extent3d { width: side, height: side, depth_or_array_layers: 1 },
            );
            queue.submit(Some(enc.finish()));
            let slice = buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            let data = slice.get_mapped_range().unwrap();
            let first = u16::from_le_bytes([data[0], data[1]]);
            assert_ne!(
                first, 0,
                "probe mip level {level} ({side}x{side}) came back EMPTY. A cube \
                 of a constant averages to that constant at every level, so this \
                 level never received its upload -- and a rough surface, which \
                 samples exactly these levels, reflects nothing at all.",
            );
            drop(data);
            buf.unmap();
        }
    }
}

#[cfg(test)]
mod probe_brightness_tests {
    use super::*;

    fn cube(res: u32, texel: [f32; 4], sky_every: usize) -> Vec<u8> {
        let n = (res * res * 6) as usize;
        (0..n)
            .flat_map(|i| {
                let t = if sky_every > 0 && i % sky_every == 0 { [9.0, 9.0, 9.0, 0.0] } else { texel };
                t.into_iter().flat_map(|v| f32_to_f16(v).to_le_bytes())
            })
            .collect()
    }

    #[test]
    fn an_even_probe_averages_to_its_luminance() {
        let m = probe_mean_radiance(&cube(4, [0.2, 0.2, 0.2, 1.0], 0), 4);
        assert!((m - 0.2).abs() < 1e-3, "mean came back {m}");
    }

    #[test]
    fn sky_texels_do_not_count() {
        let m = probe_mean_radiance(&cube(4, [0.2, 0.2, 0.2, 1.0], 3), 4);
        assert!((m - 0.2).abs() < 1e-3, "sky behind alpha 0 moved the mean to {m}");
    }

    #[test]
    fn a_probe_of_only_sky_is_unknown() {
        assert_eq!(probe_mean_radiance(&cube(2, [0.0; 4], 1), 2), 0.0);
        assert_eq!(probe_mean_radiance(&[], 2), 0.0, "a short buffer is not a probe");
    }

    #[test]
    fn brightness_follows_the_layer_not_the_slot() {
        let mut u = ProbeUpload { count: 2, ..Default::default() };
        u.set(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::ONE);
        u.set(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::ONE);
        u.fill_brightness(&[0.25, 0.75]);
        assert_eq!(u.boxes[0][1][3], 0.75, "slot 0 holds layer 1");
        assert_eq!(u.boxes[1][1][3], 0.25, "slot 1 holds layer 0");

        assert_eq!(u.boxes[2][1][3], 0.0);
    }
}

#[cfg(test)]
mod dense_room_tests {
    use super::*;

    fn upload(volumes: &[u32]) -> ProbeUpload {
        let mut u = ProbeUpload { count: volumes.len() as u32, ..Default::default() };
        for (slot, &v) in volumes.iter().enumerate() {
            u.set(slot, slot as u32, Vec3::splat(slot as f32), Vec3::ZERO, Vec3::ONE);
            u.set_volume(slot, v);
        }
        u
    }

    fn walk(tables: &[[f32; 4]; ROOM_TABLE_ROWS], room: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut s = tables[room / 4][room % 4];
        while s >= 0.0 {
            out.push(s as usize);
            let i = s as usize;
            s = tables[4 + i / 4][i % 4];
        }
        out
    }

    #[test]
    fn a_room_walks_its_slots_in_scan_order() {
        let (dense, tables) = upload(&[7, 3, 7, 9, 3, 7]).dense_rooms();
        let rooms: Vec<f32> = (0..6).map(|s| dense.boxes[s][2][3]).collect();
        assert_eq!(rooms, vec![0.0, 1.0, 0.0, 2.0, 1.0, 0.0], "numbered by first appearance");
        assert_eq!(walk(&tables, 0), vec![0, 2, 5]);
        assert_eq!(walk(&tables, 1), vec![1, 4]);
        assert_eq!(walk(&tables, 2), vec![3]);
        assert!(walk(&tables, 3).is_empty(), "no fourth room");

        let raw = upload(&[7, 3, 7, 9, 3, 7]);
        for s in 0..6 {
            assert_eq!(dense.boxes[s][0], raw.boxes[s][0]);
            assert_eq!(dense.boxes[s][1], raw.boxes[s][1]);
            assert_eq!(dense.boxes[s][2][..3], raw.boxes[s][2][..3]);
        }
    }

    #[test]
    fn portals_and_proxies_are_renumbered_alike() {
        let mut u = upload(&[7, 3]);
        let portal = |low, high| ProbePortal { min: Vec3::ZERO, max: Vec3::ONE, axis: 0, low, high, wall: None };
        u.set_portals(&[portal(7, 3), portal(3, 42)], Vec3::ZERO, &[7, 3]);
        let proxy = |volume| ProbeProxy { centre: Vec3::ZERO, half_size: Vec3::ONE, rotation: Quat::IDENTITY, volume, solid: true, field: None, cards: None };
        u.set_proxies(&[proxy(3)], Vec3::ZERO, &[7, 3]);
        let (dense, _) = u.dense_rooms();
        let sides: Vec<(f32, f32)> = (0..2).map(|p| (dense.portals[p][1][3], dense.portals[p][2][0])).collect();
        assert!(sides.contains(&(0.0, 1.0)), "{sides:?}");
        assert!(sides.contains(&(1.0, NO_RESIDENT_ROOM)), "{sides:?}");
        assert_eq!(dense.proxies[0][0][3], 1.0);
        assert!(NO_RESIDENT_ROOM >= 16.0, "outside the tables, so the shader's lookup finds nothing");
    }

    #[test]
    fn doorways_and_proxies_chain_by_room() {
        let mut u = upload(&[10, 20, 30]);
        let portal = |low, high| ProbePortal { min: Vec3::ZERO, max: Vec3::ONE, axis: 0, low, high, wall: None };

        u.set_portals(&[portal(10, 20), portal(20, 30), portal(10, 30), portal(30, 99)], Vec3::ZERO, &[10, 20, 30]);
        let proxy = |volume| ProbeProxy { centre: Vec3::ZERO, half_size: Vec3::ONE, rotation: Quat::IDENTITY, volume, solid: true, field: None, cards: None };
        u.set_proxies(&[proxy(20), proxy(10), proxy(20), proxy(99)], Vec3::ZERO, &[10, 20, 30]);
        let (dense, t) = u.dense_rooms();
        let at = |row: usize, k: usize| t[row + k / 4][k % 4];
        let portals_of = |room: f32| {
            let mut out = Vec::new();
            let mut p = at(8, room as usize);
            while p >= 0.0 {
                let pi = p as usize;
                out.push(pi);
                let side = if dense.portals[pi][1][3] == room { 0 } else { 1 };
                p = at(12, 2 * pi + side);
            }
            out
        };
        assert_eq!(portals_of(0.0), vec![0, 2]);
        assert_eq!(portals_of(1.0), vec![0, 1]);
        assert_eq!(portals_of(2.0), vec![1, 2, 3]);
        let proxies_of = |room: usize| {
            let mut out = Vec::new();
            let mut i = at(16, room);
            while i >= 0.0 {
                out.push(i as usize);
                i = at(20, i as usize);
            }
            out
        };
        assert_eq!(proxies_of(1), vec![0, 2]);
        assert_eq!(proxies_of(0), vec![1]);
        assert!(proxies_of(2).is_empty());

        assert_eq!(u.proxy_count, 3);
        assert_eq!(dense.portals[3][2][0], NO_RESIDENT_ROOM);
    }

    #[test]
    fn an_empty_upload_has_empty_tables() {
        let (_, tables) = ProbeUpload::default().dense_rooms();
        assert!(tables.iter().flatten().all(|&v| v == -1.0));
    }
}

#[cfg(test)]
mod building_tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn buildings_keep_their_layers_and_the_largest_are_kept() {
        let mut up = ProbeUpload::default();
        let small = (Vec3::ZERO, Vec3::ONE);
        let big = (Vec3::splat(-5.0), Vec3::splat(5.0));
        up.set_buildings(&[small, big], Some(40));
        assert_eq!(up.building_count, 2);

        assert_eq!(up.buildings[0], [-5.0, -5.0, -5.0, 41.0]);
        assert_eq!(up.buildings[1], [5.0, 5.0, 5.0, 0.0]);
        assert_eq!(up.buildings[2], [0.0, 0.0, 0.0, 40.0]);

        up.set_buildings(&[small, big], None);
        assert_eq!(up.building_count, 0);

        let many: Vec<(Vec3, Vec3)> = (0..MAX_BUILDINGS + 1).map(|i| (Vec3::ZERO, Vec3::splat(1.0 + i as f32))).collect();
        up.set_buildings(&many, Some(0));
        assert_eq!(up.building_count as usize, MAX_BUILDINGS);
        assert!((0..MAX_BUILDINGS).all(|slot| up.buildings[slot * 2][3] != 0.0), "the smallest building was kept");
    }
}
