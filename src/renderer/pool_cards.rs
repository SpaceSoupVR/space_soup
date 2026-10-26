use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, CommandEncoder, ComputePipeline, Device, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StorageTextureAccess, Texture, TextureFormat, TextureView, TextureViewDimension,
};

use super::brush_pipeline::probe_pass::{MirrorMips, MIRROR_MIPS};
use super::lights::MAX_LIT_SURFACES;
use super::proxy_cards::CARD_FACES;

pub const POOL_CARDS: u32 = 2;

pub const POOL_MAPS_ACROSS: u32 = 3;

pub const POOL_BANDS: u32 = 2;

pub const POOL_ATLAS_ROWS: u32 = POOL_BANDS * POOL_CARDS;

pub struct PoolCards {

    resolution: u32,
    target: Texture,

    levels: Vec<TextureView>,
    pipeline: ComputePipeline,
    target_group: BindGroup,
}

const FORMAT: TextureFormat = TextureFormat::Rgba16Float;

pub(crate) fn shader() -> String {
    format!(
        r"
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
",
        lights = super::lights::wgsl_lights_block_with(0, 1, super::lights::LightsBlockOptions::default()),
        across = POOL_MAPS_ACROSS,
    )
}

impl PoolCards {

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

    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    pub fn first_row(&self, pool_row: u32) -> u32 {
        pool_row * self.resolution
    }

    #[allow(clippy::too_many_arguments, reason = "GPU setup passes each resource explicitly; bundling them is a separate refactor")]
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

const _: () = assert!((POOL_CARDS * POOL_MAPS_ACROSS) as usize == CARD_FACES);
const _: () = assert!((POOL_MAPS_ACROSS * POOL_BANDS) as usize == MAX_LIT_SURFACES);
