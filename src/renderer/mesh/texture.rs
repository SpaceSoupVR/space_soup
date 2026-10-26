pub struct LoadedTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    _sampler: wgpu::Sampler,
    pub bind_group: wgpu::BindGroup,

    _direction: Option<wgpu::Texture>,

    _sun_mask: Option<(wgpu::Texture, wgpu::Sampler, wgpu::Texture)>,
}

pub(crate) fn load_primitive_texture(
    prim: &gltf::Primitive,
    images: &[gltf::image::Data],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> LoadedTexture {
    let material = prim.material();
    let pbr = material.pbr_metallic_roughness();

    let transmission = material
        .transmission()
        .map_or(0.0, |t| t.transmission_factor());
    let force_opaque =
        material.alpha_mode() != gltf::material::AlphaMode::Blend && transmission <= 0.0;

    let base = if let Some(info) = pbr.base_color_texture() { decode_image(&images[info.texture().source().index()], force_opaque) } else {
        let c = pbr.base_color_factor();
        (
            vec![
                (c[0] * 255.0) as u8,
                (c[1] * 255.0) as u8,
                (c[2] * 255.0) as u8,
                if force_opaque {
                    255
                } else {

                    (c[3] * (1.0 - transmission) * 255.0) as u8
                },
            ],
            1,
            1,
        )
    };

    let emissive = match material.emissive_texture() {
        Some(info) => {
            let (mut rgba, w, h) = decode_image(&images[info.texture().source().index()], true);
            clean_emissive_mask(&mut rgba);
            (rgba, w, h)
        }

        None => (vec![255u8, 255, 255, material_gloss_alpha(pbr.roughness_factor(), gltf_roughness())], 1, 1),
    };

    create_mesh_material_texture(device, queue, layout, &base, &emissive)
}

static GLTF_ROUGHNESS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_gltf_roughness(on: bool) {
    GLTF_ROUGHNESS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn gltf_roughness() -> bool {
    GLTF_ROUGHNESS.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn material_gloss_alpha(roughness: f32, honour: bool) -> u8 {
    if !honour || !roughness.is_finite() {
        return 255;
    }
    (roughness.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod gloss_tests {
    use super::material_gloss_alpha;

    #[test]
    fn off_or_matte_is_the_classic_sheen_and_on_is_the_roughness() {
        assert_eq!(material_gloss_alpha(0.5, false), 255);
        assert_eq!(material_gloss_alpha(1.0, true), 255);
        assert_eq!(material_gloss_alpha(0.5, true), 128);
        assert_eq!(material_gloss_alpha(0.3, true), 77);
        assert_eq!(material_gloss_alpha(0.0, true), 0);
        assert_eq!(material_gloss_alpha(f32::NAN, true), 255);
    }

    #[test]
    fn a_half_rough_material_takes_the_shaders_material_path() {

        assert!((f32::from(material_gloss_alpha(0.7, true)) / 255.0) < 0.996);
        assert!((f32::from(material_gloss_alpha(1.0, true)) / 255.0) >= 0.996);
    }
}

fn clean_emissive_mask(rgba: &mut [u8]) {
    let peak = rgba.chunks_exact(4).map(|p| p[0].max(p[1]).max(p[2])).max().unwrap_or(0);
    let floor = (peak / 8).max(2);
    for p in rgba.chunks_exact_mut(4) {
        if p[0].max(p[1]).max(p[2]) <= floor {
            p[..3].fill(0);
        }
    }
}

fn decode_image(image: &gltf::image::Data, force_opaque: bool) -> (Vec<u8>, u32, u32) {
    use gltf::image::Format;
    let (width, height) = (image.width, image.height);
    let rgba: Vec<u8> = match image.format {
        Format::R8G8B8A8 => {
            if force_opaque {
                let mut out = image.pixels.clone();
                out.chunks_exact_mut(4).for_each(|px| px[3] = 255);
                out
            } else {
                image.pixels.clone()
            }
        }
        Format::R8G8B8 => {
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for px in image.pixels.chunks_exact(3) {
                out.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            out
        }
        Format::R8 => {
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for &v in &image.pixels {
                out.extend_from_slice(&[v, v, v, 255]);
            }
            out
        }
        _ => {
            log::warn!("Unsupported glTF image format {:?}, using gray fallback", image.format);
            ([180u8, 180, 180, 255].repeat((width * height) as usize), width, height).0
        }
    };
    (rgba, width, height)
}

fn upload_mipped(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, px: &(Vec<u8>, u32, u32)) -> wgpu::Texture {
    let chain = crate::renderer::brush_pipeline::mip_chain(
        &crate::renderer::terrain_pipeline::TerrainImage { width: px.1, height: px.2, rgba: px.0.clone() },
        true,
    );
    let size = wgpu::Extent3d { width: px.1, height: px.2, depth_or_array_layers: 1 };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: chain.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,

        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    for (level, img) in chain.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &img.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * img.width),
                rows_per_image: Some(img.height),
            },
            wgpu::Extent3d { width: img.width, height: img.height, depth_or_array_layers: 1 },
        );
    }
    tex
}

pub fn create_mesh_material_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    base: &(Vec<u8>, u32, u32),
    emissive: &(Vec<u8>, u32, u32),
) -> LoadedTexture {
    let make = |label: &str, px: &(Vec<u8>, u32, u32)| upload_mipped(device, queue, label, px);

    let base_tex = make("gltf_base_color", base);
    let emissive_tex = make("gltf_emissive", emissive);
    let view = base_tex.create_view(&Default::default());
    let emissive_view = emissive_tex.create_view(&Default::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        anisotropy_clamp: 8,
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gltf_material_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&emissive_view),
            },
        ],
    });
    LoadedTexture { texture: base_tex, view, _sampler: sampler, bind_group, _direction: None, _sun_mask: None }
}

pub const LIGHTMAP_MIP_LEVELS: u32 = 3;

pub const SUN_MASK_MIP_LEVELS: u32 = 5;

pub const NEUTRAL_SUN_MASK: [u8; 4] = [0, 0, 0, 0];

pub const NEUTRAL_STATIONARY_MASK: [u8; 4] = [255, 255, 255, 255];

pub const STATIONARY_MASK_MIP_LEVELS: u32 = 4;

fn mip_levels_for(width: u32, height: u32, wanted: u32) -> u32 {
    let possible = 32 - width.max(height).max(1).leading_zeros();
    wanted.min(possible).max(1)
}

fn upload_rgba_texture_mipped(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba: &[u8],
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
    srgb: bool,
    wanted_levels: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    use crate::renderer::terrain_pipeline::{mip_chain, mip_chain_linear};

    let levels = mip_levels_for(width, height, wanted_levels);
    let chain = if srgb {
        mip_chain(rgba, width, height, levels)
    } else {
        mip_chain_linear(rgba, width, height, levels)
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, (data, lw, lh)) in chain.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * lw),
                rows_per_image: Some(*lh),
            },
            wgpu::Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub const NEUTRAL_BOUNCE_DIRECTION: [u8; 4] = [128, 128, 128, 0];

pub fn create_lightmap_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    rgba: &[u8],
    width: u32,
    height: u32,
    direction: Option<(&[u8], u32, u32)>,
) -> LoadedTexture {
    create_lightmap_texture_with_sun(
        device, queue, layout, LightmapLight::Srgb8(rgba), width, height, direction, None,
    )
}

#[derive(Clone, Copy)]
pub enum LightmapLight<'a> {

    Srgb8(&'a [u8]),

    Linear(&'a [f32]),
}

#[allow(clippy::too_many_arguments)]
pub fn create_lightmap_texture_with_sun(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    light: LightmapLight,
    width: u32,
    height: u32,
    direction: Option<(&[u8], u32, u32)>,
    sun_mask: Option<(&[u8], u32, u32)>,
) -> LoadedTexture {
    create_lightmap_texture_full(device, queue, layout, light, width, height, direction, sun_mask, None)
}

#[allow(clippy::too_many_arguments)]
pub fn create_lightmap_texture_full(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    light: LightmapLight,
    width: u32,
    height: u32,
    direction: Option<(&[u8], u32, u32)>,
    sun_mask: Option<(&[u8], u32, u32)>,
    stationary: Option<(&[&[u8]], u32, u32)>,
) -> LoadedTexture {

    let (base_texture, base_view) = match light {
        LightmapLight::Srgb8(rgba) => upload_rgba_texture_mipped(
            device,
            queue,
            rgba,
            width,
            height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            "lightmap",
            true,
            LIGHTMAP_MIP_LEVELS,
        ),
        LightmapLight::Linear(texels) => upload_linear_f16_mipped(device, queue, texels, width, height, "lightmap_hdr"),
    };
    let (dir_rgba, dir_w, dir_h) =
        direction.unwrap_or((&NEUTRAL_BOUNCE_DIRECTION, 1, 1));

    let (dir_tex, dir_view) = upload_rgba_texture_mipped(
        device,
        queue,
        dir_rgba,
        dir_w,
        dir_h,
        wgpu::TextureFormat::Rgba8Unorm,
        "lightmap_direction",
        false,
        LIGHTMAP_MIP_LEVELS,
    );
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("lightmap_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,

        lod_max_clamp: (LIGHTMAP_MIP_LEVELS - 1) as f32,
        ..Default::default()
    });
    let (sun_rgba, sun_w, sun_h) = sun_mask.unwrap_or((&NEUTRAL_SUN_MASK, 1, 1));
    let (sun_tex, sun_view) = upload_rg8_mipped(device, queue, sun_rgba, sun_w, sun_h, "lightmap_sun_mask");

    let sun_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("lightmap_sun_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        lod_max_clamp: (SUN_MASK_MIP_LEVELS - 1) as f32,
        ..Default::default()
    });
    let neutral: [&[u8]; 1] = [&NEUTRAL_STATIONARY_MASK];
    let (st_layers, st_w, st_h) = stationary.unwrap_or((&neutral, 1, 1));
    let (st_tex, st_view) = upload_rgba8_array_mipped(device, queue, st_layers, st_w, st_h, "lightmap_stationary");
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lightmap_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&base_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&dir_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&sun_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&sun_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&st_view),
            },
        ],
    });
    LoadedTexture {
        texture: base_texture,
        view: base_view,
        _sampler: sampler,
        bind_group,
        _direction: Some(dir_tex),
        _sun_mask: Some((sun_tex, sun_sampler, st_tex)),
    }
}

fn upload_linear_f16_mipped(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texels: &[f32],
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let levels = mip_levels_for(width, height, LIGHTMAP_MIP_LEVELS);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_lightmap_mips(queue, &texture, &lightmap_mips_f16(texels, width, height));
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub fn lightmap_mips_f16(texels: &[f32], width: u32, height: u32) -> Vec<(Vec<u8>, u32, u32)> {
    let levels = mip_levels_for(width, height, LIGHTMAP_MIP_LEVELS);
    let mut out = Vec::with_capacity(levels as usize);
    let mut level: Vec<f32> = texels.to_vec();
    let (mut lw, mut lh) = (width, height);
    for mip in 0..levels {
        let half: Vec<u8> = level
            .iter()
            .flat_map(|v| crate::renderer::sky::f32_to_f16(*v).to_le_bytes())
            .collect();
        out.push((half, lw, lh));
        if mip + 1 == levels {
            break;
        }

        let (dw, dh) = ((lw / 2).max(1), (lh / 2).max(1));
        let mut next = vec![0f32; (dw * dh * 4) as usize];
        for y in 0..dh {
            for x in 0..dw {
                for c in 0..4 {
                    let mut sum = 0.0;
                    for (sx, sy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (px, py) = ((2 * x + sx).min(lw - 1), (2 * y + sy).min(lh - 1));
                        sum += level[((py * lw + px) * 4 + c) as usize];
                    }
                    next[((y * dw + x) * 4 + c) as usize] = sum * 0.25;
                }
            }
        }
        level = next;
        lw = dw;
        lh = dh;
    }
    out
}

pub fn write_lightmap_mips(queue: &wgpu::Queue, texture: &wgpu::Texture, chain: &[(Vec<u8>, u32, u32)]) {
    for (mip, (half, lw, lh)) in chain.iter().enumerate() {
        if mip as u32 >= texture.mip_level_count() {
            break;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            half,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8 * lw), rows_per_image: Some(*lh) },
            wgpu::Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
        );
    }
}

impl LoadedTexture {

    pub fn from_texture(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        texture: wgpu::Texture,
    ) -> Self {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("panel_quad_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let emissive_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("panel_quad_emissive"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let emissive_view = emissive_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("panel_quad_texture_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&emissive_view),
                },
            ],
        });
        Self {
            texture,
            view,
            _sampler: sampler,
            bind_group,

            _direction: Some(emissive_tex),
            _sun_mask: None,
        }
    }

    pub fn sunless_bind_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> Option<(wgpu::BindGroup, wgpu::Texture)> {
        let dir = self._direction.as_ref()?;
        let (_, sun_sampler, st_tex) = self._sun_mask.as_ref()?;
        let (neutral, neutral_view) = upload_rg8_mipped(device, queue, &NEUTRAL_SUN_MASK, 1, 1, "lightmap_sun_mask_neutral");
        let dir_view = dir.create_view(&wgpu::TextureViewDescriptor::default());
        let st_view = st_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lightmap_bg_sunless"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self._sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dir_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&neutral_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(sun_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&st_view) },
            ],
        });
        Some((bind_group, neutral))
    }
}

fn upload_rgba8_array_mipped(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layers: &[&[u8]],
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    use crate::renderer::terrain_pipeline::mip_chain_linear;
    let levels = mip_levels_for(width, height, STATIONARY_MASK_MIP_LEVELS);
    let count = layers.len().max(1) as u32;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: count },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (layer, rgba) in layers.iter().enumerate() {
        for (level, (data, lw, lh)) in mip_chain_linear(rgba, width, height, levels).iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * lw),
                    rows_per_image: Some(*lh),
                },
                wgpu::Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
            );
        }
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

pub(crate) fn pack_sun_mask_texel(p: &[u8]) -> [u8; 2] {
    [p[0], if p[1] >= 128 { 128 + p[2] / 2 } else { 0 }]
}

fn upload_rg8_mipped(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba: &[u8],
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    use crate::renderer::terrain_pipeline::mip_chain_linear;
    let levels = mip_levels_for(width, height, SUN_MASK_MIP_LEVELS);
    let chain = mip_chain_linear(rgba, width, height, levels);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, (data, lw, lh)) in chain.iter().enumerate() {
        let rg: Vec<u8> = data.chunks_exact(4).flat_map(pack_sun_mask_texel).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rg,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(2 * lw),
                rows_per_image: Some(*lh),
            },
            wgpu::Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub fn create_texture_from_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> LoadedTexture {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gltf_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        size,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("gltf_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gltf_texture_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    LoadedTexture {
        texture,
        view,
        _sampler: sampler,
        bind_group,
        _direction: None,
        _sun_mask: None,
    }
}

#[cfg(test)]
mod transmission_tests {

    fn blocks_light(alpha_mode_is_blend: bool, transmission: f32, base_alpha: f32) -> f32 {
        let force_opaque = !alpha_mode_is_blend && transmission <= 0.0;
        if force_opaque { 1.0 } else { base_alpha * (1.0 - transmission) }
    }

    #[test]
    fn clear_glass_blocks_nothing() {

        assert_eq!(blocks_light(false, 1.0, 1.0), 0.0);
    }

    #[test]
    fn an_ordinary_opaque_material_is_untouched() {

        assert_eq!(blocks_light(false, 0.0, 1.0), 1.0);
        assert_eq!(blocks_light(false, 0.0, 0.25), 1.0, "alpha is ignored when opaque");
    }

    #[test]
    fn partial_transmission_partly_blocks() {
        assert!((blocks_light(false, 0.4, 1.0) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn a_blended_material_still_uses_its_own_alpha() {
        assert!((blocks_light(true, 0.0, 0.3) - 0.3).abs() < 1e-6);
    }
}

#[cfg(test)]
mod sun_mask_packing_tests {
    use super::*;

    #[test]
    fn the_sun_penumbra_survives_the_two_channel_upload() {
        let range = crate::renderer::brush_pipeline::SUN_MASK_DISTANCE_TEXELS;
        for blue in [0u8, 1, 64, 128, 200, 255] {
            let [r, g] = pack_sun_mask_texel(&[77, 255, blue, 255]);
            assert_eq!(r, 77, "the distance must pass through untouched");
            let g = f32::from(g) / 255.0;
            assert!(g > 0.25, "a baked texel read as unbaked");
            let decoded = (g - 0.5).max(0.0) * 2.0 * range;
            let written = f32::from(blue) / 255.0 * range;
            assert!((decoded - written).abs() < 0.04, "blue {blue}: baked {written}, the shader reads {decoded}");
        }
        let [_, g] = pack_sun_mask_texel(&NEUTRAL_SUN_MASK);
        assert!((f32::from(g) / 255.0) < 0.25, "the neutral mask read as baked");
    }
}

#[cfg(test)]
mod lightmap_mip_tests {
    use super::*;

    #[test]
    fn a_lightmap_is_uploaded_with_the_levels_its_sampler_asks_for() {
        let Some((device, queue)) =
            crate::renderer::terrain_pipeline::tests::headless_gpu()
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = crate::renderer::pipeline::lightmap_bind_group_layout(&device);
        let rgba = vec![200u8; 64 * 64 * 4];
        let lm = create_lightmap_texture(&device, &queue, &layout, &rgba, 64, 64, None);
        assert_eq!(
            lm.texture.mip_level_count(),
            LIGHTMAP_MIP_LEVELS,
            "the lightmap is back to a single level while `lightmap_sampler` \
             still asks for Linear mipmap filtering -- which is a silent no-op \
             and is exactly the bug the material textures had",
        );
    }

    #[test]
    fn an_image_is_never_asked_for_more_levels_than_it_has() {
        assert_eq!(mip_levels_for(1, 1, LIGHTMAP_MIP_LEVELS), 1);
        assert_eq!(mip_levels_for(2, 1, LIGHTMAP_MIP_LEVELS), 2);
        assert_eq!(mip_levels_for(4, 4, LIGHTMAP_MIP_LEVELS), 3);
        assert_eq!(mip_levels_for(1024, 1024, LIGHTMAP_MIP_LEVELS), 3);
    }

    #[test]
    fn the_chain_averages_colour_in_light_and_alpha_as_stored() {
        use crate::renderer::terrain_pipeline::mip_chain;

        let rgba: Vec<u8> = vec![
            0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
        ];
        let chain = mip_chain(&rgba, 2, 2, 2);
        let (level1, w, h) = &chain[1];
        assert_eq!((*w, *h), (1, 1));
        let grey = level1[0];
        assert!(
            (186..=190).contains(&grey),
            "half black and half white should average to mid GREY (sRGB ~188), \
             not to byte 128 which is 22% of the light: got {grey}",
        );
        assert_eq!(
            level1[3], 128,
            "alpha is a linear scalar and must average to 128, not through the \
             sRGB transfer",
        );
    }
}

#[cfg(test)]
mod model_texture_tests {
    use super::*;

    #[test]
    fn an_emissive_mask_keeps_its_bulb_and_drops_its_noise() {
        let mut rgba = vec![
            255, 250, 240, 255,
            60, 60, 60, 255,
            0, 0, 2, 255,
            20, 5, 30, 255,
            0, 0, 0, 255,
        ];
        clean_emissive_mask(&mut rgba);
        assert_eq!(&rgba[0..4], &[255, 250, 240, 255]);
        assert_eq!(&rgba[4..8], &[60, 60, 60, 255]);
        assert_eq!(&rgba[8..12], &[0, 0, 0, 255]);
        assert_eq!(&rgba[12..16], &[0, 0, 0, 255]);

        let mut dim = vec![64, 64, 64, 255, 9, 9, 9, 255, 8, 8, 8, 255];
        clean_emissive_mask(&mut dim);
        assert_eq!(dim, vec![64, 64, 64, 255, 9, 9, 9, 255, 0, 0, 0, 255]);
    }

    #[test]
    fn a_model_texture_carries_its_whole_mip_chain_and_it_averages() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut rgba = Vec::new();
        for y in 0..4u32 {
            for x in 0..4u32 {
                let v = if (x + y) % 2 == 0 { 255 } else { 0 };
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let tex = upload_mipped(&device, &queue, "test", &(rgba, 4, 4));
        assert_eq!(tex.mip_level_count(), 3);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256 * 2,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 1, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(2) },
            },
            wgpu::Extent3d { width: 2, height: 2, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let bytes = buffer.slice(..).get_mapped_range().unwrap();
        for (x, y) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
            let r = bytes[y * 256 + x * 4];
            assert!((187..=189).contains(&r), "level 1 texel {x},{y}: {r}");
        }
    }
}
