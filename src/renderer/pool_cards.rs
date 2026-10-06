//! THE TORCH'S POOL, LIT ONCE A FRAME FOR EVERY REFLECTION THAT SHOWS IT. A
//! reflection meeting a surface a live lamp's beam lights shows that light
//! there (user, 2026-10-05: "making the flashlight torch light on the wall be
//! reflected on other surfaces"), and the probe photographs hold only the
//! level's own. Worked out at every reflected point -- the lamp loop, the cone
//! and each lamp's nine-tap shadow -- it cost the torch views 1.2-1.45 ms of
//! the reflection pass on the headset, and the ground's pass a register in
//! every form tried (2026-10-06). So it is worked out here instead: each lit
//! surface (`lights::LitSurface`, at most `MAX_LIT_SURFACES`) gets a map of
//! that light on its plane, as seen from the glass (`pool_map_light` in the
//! lights block), a compute thread a texel; the floor mirror's mip pass blurs
//! it; and every level is copied into the card atlas, which every reflection
//! pass already samples -- the probe pass is at its sampled-texture limit, as
//! the character cards found. A reflection then reads one texel
//! (`probe_surface_relit`), as blurred as its footprint, so a hand's shadow in
//! the pool softens with the floor's roughness as the pool's edge does.
//!
//! WHERE: rows of the atlas kept for them after the characters'
//! (`proxy_cards::CardAtlas::pool_row`). Each map is two cards square, three
//! side by side across the atlas's six cards, in [`POOL_BANDS`] bands. Six
//! maps, not the three that fit under a character's own cards: with three,
//! the far wall a beam passes a pillar to reach lost its pool in every
//! reflection (offline torch_pillar, 2026-10-06: up to 10 levels on 3% of
//! the frame).

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, CommandEncoder, ComputePipeline, Device, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StorageTextureAccess, Texture, TextureFormat, TextureView, TextureViewDimension,
};

use super::brush_pipeline::probe_pass::{MirrorMips, MIRROR_MIPS};
use super::lights::MAX_LIT_SURFACES;
use super::proxy_cards::CARD_FACES;

/// Cards across a pool map, and up it.
pub const POOL_CARDS: u32 = 2;

/// Maps side by side across the atlas.
pub const POOL_MAPS_ACROSS: u32 = 3;

/// Bands of maps, one under the other.
pub const POOL_BANDS: u32 = 2;

/// The card rows the atlas keeps for the maps (`proxy_cards::CardAtlas::pool_row`).
pub const POOL_ATLAS_ROWS: u32 = POOL_BANDS * POOL_CARDS;

/// The pool maps: drawn into their own small target, blurred, copied into the
/// atlas.
pub struct PoolCards {
    /// The atlas's card size, in texels.
    resolution: u32,
    target: Texture,
    /// Level 0 drawn into and read by the blur; then the blur levels.
    levels: Vec<TextureView>,
    pipeline: ComputePipeline,
    target_group: BindGroup,
}

const FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// The maps' compute shader: the lights block, and a thread a texel of the
/// maps, [`POOL_MAPS_ACROSS`] side by side in each band, each map
/// `size.x / POOL_MAPS_ACROSS` square -- the order `probe_surface_relit`
/// reads them in.
pub(crate) fn shader() -> String {
    format!(
        r#"
{lights}
@group(1) @binding(0) var pool_out: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8)
fn pool_main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let size = textureDimensions(pool_out);
    if (id.x >= size.x || id.y >= size.y) {{
        return;
    }}
    let block = size.x / {across}u;
    let k = i32(id.y / block) * {across} + i32(id.x / block);
    let uv = (vec2<f32>(id.xy % vec2<u32>(block)) + vec2<f32>(0.5)) / f32(block);
    textureStore(pool_out, vec2<i32>(id.xy), vec4<f32>(pool_map_light(k, uv, f32(block)), 1.0));
}}
"#,
        lights = super::lights::wgsl_lights_block_with(0, 1, super::lights::LightsBlockOptions::default()),
        across = POOL_MAPS_ACROSS,
    )
}

impl PoolCards {
    /// Maps for an atlas of cards `resolution` texels across, lit with the
    /// scene's group 0 (`uniform_layout`: camera, lights, shadow maps).
    pub fn new(device: &Device, uniform_layout: &BindGroupLayout, resolution: u32) -> Self {
        let block = POOL_CARDS * resolution;
        let level_count = MIRROR_MIPS.min(block.max(1).ilog2() + 1);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pool_cards"),
            size: wgpu::Extent3d { width: block * POOL_MAPS_ACROSS, height: block * POOL_BANDS, depth_or_array_layers: 1 },
            mip_level_count: level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let levels: Vec<TextureView> = (0..level_count)
            .map(|l| {
                target.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: l,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let target_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("pool_cards_target"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: FORMAT,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            }],
        });
        let target_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("pool_cards_target"),
            layout: &target_layout,
            entries: &[BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&levels[0]) }],
        });
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("pool_cards"),
            source: ShaderSource::Wgsl(shader().into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pool_cards"),
            bind_group_layouts: &[Some(uniform_layout), Some(&target_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pool_cards"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("pool_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { resolution, target, levels, pipeline, target_group }
    }

    /// The atlas card size these maps were made for.
    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    /// The card-atlas texel row the maps go to, from the atlas's first pool
    /// row (`proxy_cards::CardAtlas::pool_row`).
    pub fn first_row(&self, pool_row: u32) -> u32 {
        pool_row * self.resolution
    }

    /// Lights every map from the lights uploaded this frame -- `uniforms` is
    /// the scene's group 0, after the frame's spot shadows are drawn -- blurs
    /// them, and copies every level into `atlas` from texel row `first_row`
    /// ([`Self::first_row`]). Before anything reads them. `timer`: a pass
    /// timer's slot for the lighting and the next for the blur levels.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        uniforms: &BindGroup,
        mips: &MirrorMips,
        atlas: &Texture,
        first_row: u32,
        timer: Option<(&crate::renderer::pass_timers::PassTimers, usize)>,
    ) {
        let block = POOL_CARDS * self.resolution;
        let (width, height) = (block * POOL_MAPS_ACROSS, block * POOL_BANDS);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pool_cards"),
                timestamp_writes: timer.and_then(|(timers, slot)| timers.compute_writes(slot)),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, uniforms, &[]);
            pass.set_bind_group(1, &self.target_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        mips.record(device, encoder, &self.levels, (width, height), timer.map(|(timers, slot)| (timers, slot + 1)));
        for l in 0..self.levels.len() as u32 {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.target,
                    mip_level: l,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: atlas,
                    mip_level: l,
                    origin: wgpu::Origin3d { x: 0, y: first_row >> l, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d { width: (width >> l).max(1), height: (height >> l).max(1), depth_or_array_layers: 1 },
            );
        }
    }
}

/// The maps fill the atlas's width exactly, so the shader finds each from the
/// atlas's size alone (`probe_surface_relit`), and there is a map for every
/// surface the lights block takes.
const _: () = assert!((POOL_CARDS * POOL_MAPS_ACROSS) as usize == CARD_FACES);
const _: () = assert!((POOL_MAPS_ACROSS * POOL_BANDS) as usize == MAX_LIT_SURFACES);
