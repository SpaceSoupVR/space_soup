//! STANDING MODELS' CARDS on the GPU: six small pictures of each placed model,
//! one looking in through each face of its reflection proxy's box, which
//! colour the model wherever a reflection meets it. Baked by `tools/bake`
//! (`probe::capture_cards`); layout, encoding and why they exist in
//! `space_soup_engine::reflection_cards`.
//!
//! The room photographs no longer contain these models, so the wall behind a
//! lamp is never coloured from a picture with the lamp in front of it, and the
//! lamp is coloured from pictures of itself rather than from a capture point
//! across the room. See `probe_card_colour` in the lights block.
//!
//! ONE ATLAS, ONE ROW A MODEL: row `s` holds model `s`'s six cards side by
//! side, card `k` at columns `k * R .. (k + 1) * R`, so the shader finds a card
//! from its row, its number and the atlas's width alone. RGBA16Float: linear
//! radiance, and in alpha how deep in the box the surface is (2 where the card
//! saw nothing).
//!
//! Plain data here: `space_soup` cannot depend on the engine, so the app hands
//! the cards over as [`ProxyCards`].

use wgpu::{Device, Queue, TextureView};

/// Six cards a model. Must equal `space_soup_engine::reflection_cards::CARD_FACES`.
pub const CARD_FACES: usize = 6;

/// How many models' cards a level may carry at once: one per proxy the trace
/// can hold, and a row each.
pub const MAX_CARD_SETS: usize = 64;

/// One placed model's cards: see `space_soup_engine::reflection_cards::LoadedCards`,
/// which this mirrors.
#[derive(Clone, Debug, PartialEq)]
pub struct ProxyCards {
    /// Texels across a card.
    pub resolution: u32,
    /// `CARD_FACES * resolution^2` texels, card after card, row after row:
    /// linear RGB and the depth `t`.
    pub texels: Vec<[f32; 4]>,
}

/// The cards packed into one atlas, a model a row, and which row each input
/// landed in (`None` for one skipped: a different card size than the first,
/// or past [`MAX_CARD_SETS`]). `None` for no cards at all.
pub fn atlas(device: &Device, queue: &Queue, sets: &[ProxyCards]) -> Option<(TextureView, Vec<Option<u32>>)> {
    let res = sets.iter().map(|c| c.resolution).find(|&r| r > 0)?;
    let mut rows: Vec<Option<u32>> = Vec::with_capacity(sets.len());
    let mut used = 0u32;
    for c in sets {
        let fits = c.resolution == res && c.texels.len() == CARD_FACES * (res * res) as usize && (used as usize) < MAX_CARD_SETS;
        if !fits {
            log::warn!("reflection cards: a set of {} texels at {} across does not fit the atlas; skipped", c.texels.len(), c.resolution);
        }
        rows.push(fits.then(|| {
            used += 1;
            used - 1
        }));
    }
    if used == 0 {
        return None;
    }
    let width = res * CARD_FACES as u32;
    let height = res * used;
    let mut data: Vec<u16> = vec![0; (width * height * 4) as usize];
    for (c, row) in sets.iter().zip(&rows) {
        let Some(row) = row else { continue };
        for face in 0..CARD_FACES as u32 {
            for y in 0..res {
                for x in 0..res {
                    let src = c.texels[((face * res + y) * res + x) as usize];
                    let dst = (((row * res + y) * width + face * res + x) * 4) as usize;
                    for k in 0..4 {
                        data[dst + k] = crate::renderer::sky::f32_to_f16(src[k]);
                    }
                }
            }
        }
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("proxy_card_atlas"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width * 8), rows_per_image: Some(height) },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    Some((tex.create_view(&wgpu::TextureViewDescriptor::default()), rows))
}

/// What a level without cards binds: one texel, never read.
pub fn none(device: &Device) -> TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("default_proxy_card_atlas"),
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
