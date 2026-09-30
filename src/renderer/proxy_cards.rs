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
//! ONE ATLAS, TWO ROWS A MODEL: row `2s` holds model `s`'s six cards side by
//! side, card `k` at columns `k * R .. (k + 1) * R`, so the shader finds a card
//! from its row, its number and the atlas's width alone. RGBA16Float: linear
//! radiance, and in alpha how deep in the box the surface is (2 where the card
//! saw nothing). Row `2s + 1` holds the same cards' normals -- which way each
//! texel's surface faces, in the box's frame -- with the same depth in alpha.
//! One texture rather than a second binding: the brush shader is at its
//! sampled-texture limit.
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
    /// Which way each texel's surface faces, in the box's frame, laid out as
    /// `texels`; zero where the card saw nothing.
    pub normals: Vec<[f32; 3]>,
}

/// The cards packed into one atlas, two rows a model, and the row of each
/// input's colours -- its normals are the row after -- (`None` for one
/// skipped: a different card size than the first, or past
/// [`MAX_CARD_SETS`]). `None` for no cards at all.
pub fn atlas(device: &Device, queue: &Queue, sets: &[ProxyCards]) -> Option<(TextureView, Vec<Option<u32>>)> {
    let res = sets.iter().map(|c| c.resolution).find(|&r| r > 0)?;
    let mut rows: Vec<Option<u32>> = Vec::with_capacity(sets.len());
    let mut used = 0u32;
    for c in sets {
        let texels = CARD_FACES * (res * res) as usize;
        let fits = c.resolution == res && c.texels.len() == texels && c.normals.len() == texels && (used as usize) < MAX_CARD_SETS;
        if !fits {
            log::warn!("reflection cards: a set of {} texels at {} across does not fit the atlas; skipped", c.texels.len(), c.resolution);
        }
        rows.push(fits.then(|| {
            used += 1;
            2 * (used - 1)
        }));
    }
    if used == 0 {
        return None;
    }
    // Level 0, the atlas as the file has it: linear RGB and the depth `t`,
    // and below, the normal and the same `t`.
    let width = res * CARD_FACES as u32;
    let height = 2 * res * used;
    let mut level: Vec<[f32; 4]> = vec![[0.0, 0.0, 0.0, MISS]; (width * height) as usize];
    for (c, row) in sets.iter().zip(&rows) {
        let Some(row) = row else { continue };
        for face in 0..CARD_FACES as u32 {
            for y in 0..res {
                for x in 0..res {
                    let i = ((face * res + y) * res + x) as usize;
                    let t = c.texels[i];
                    let [nx, ny, nz] = c.normals[i];
                    level[((row * res + y) * width + face * res + x) as usize] = t;
                    level[(((row + 1) * res + y) * width + face * res + x) as usize] = [nx, ny, nz, t[3]];
                }
            }
        }
    }
    // THE MIP CHAIN, down to a texel a card. See `probe_card_colour`: a
    // reflection far off reads a card at its footprint's size, so a small
    // bright part -- a sconce's glowing mouth -- shows as its share of the
    // footprint rather than as whole half-resolution texels switching on and
    // off as the head moves. Each level averages only the texels that SAW the
    // model; a texel none of whose four saw it saw nothing. A card's 2 x 2 blocks never straddle two cards, since cards sit
    // at multiples of their own power-of-two size.
    let levels = if res.is_power_of_two() { res.trailing_zeros() + 1 } else { 1 };
    let mut chain: Vec<(u32, u32, Vec<[f32; 4]>)> = vec![(width, height, level)];
    for l in 1..levels {
        let (pw, ph, prev) = chain.last().unwrap();
        let (w, h) = (pw / 2, ph / 2);
        let mut next = vec![[0.0, 0.0, 0.0, MISS]; (w * h) as usize];
        for y in 0..h {
            // A normals row: averaged plainly, left unnormalised, so a texel
            // over a curve or an edge says how little one direction speaks for it.
            let normals = (y / (res >> l)) % 2 == 1;
            for x in 0..w {
                // Colour weighted by 1 / (1 + luminance) -- Karis's firefly
                // weight -- depth plainly. A glowing mouth is sixty times
                // brighter than the shade round it: averaged plainly, a texel a
                // quarter mouth was still fifteen times white, and every far
                // sconce reflected as a white blob the size of the footprint.
                let (mut rgb, mut weight, mut depth, mut n) = ([0.0f32; 3], 0.0f32, 0.0f32, 0.0f32);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let t = prev[((2 * y + dy) * pw + 2 * x + dx) as usize];
                    if t[3] < MISS * 0.75 {
                        let w = if normals { 1.0 } else { 1.0 / (1.0 + 0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2]) };
                        for k in 0..3 {
                            rgb[k] += t[k] * w;
                        }
                        weight += w;
                        depth += t[3];
                        n += 1.0;
                    }
                }
                if n > 0.0 {
                    next[(y * w + x) as usize] = [rgb[0] / weight, rgb[1] / weight, rgb[2] / weight, depth / n];
                }
            }
        }
        chain.push((w, h, next));
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("proxy_card_atlas"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (mip, (w, h, texels)) in chain.iter().enumerate() {
        let data: Vec<u16> = texels.iter().flat_map(|t| t.map(crate::renderer::sky::f32_to_f16)).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: mip as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(*h) },
            wgpu::Extent3d { width: *w, height: *h, depth_or_array_layers: 1 },
        );
    }
    Some((tex.create_view(&wgpu::TextureViewDescriptor::default()), rows))
}

/// `t` for a texel whose card saw nothing. Must equal
/// `space_soup_engine::reflection_cards::CARD_MISS`.
pub const MISS: f32 = 2.0;

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
