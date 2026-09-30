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
//! saw nothing). Row `2s + 1` holds what the shader TESTS a card by: which
//! way each texel's surface faces, in red and green -- in the CARD's frame,
//! its u, v and the way it looks from, on a hemispherical octahedral map
//! ([`to_card_octahedral`]): every surface a card saw faces it, so the map
//! never folds and a filtered read between two texels is a direction between
//! theirs -- in blue the texel's own depth, and in alpha the nearest
//! depth round it, widened to its neighbours' ([`widen_ranges`]) so a
//! filtered read is never nearer than a surface the texels around it saw. One
//! texture rather than a second binding: the brush shader is at its
//! sampled-texture limit.
//!
//! COLOUR BY THE FOOTPRINT, TRUST BY THE POINT: the shader reads the colours
//! at the reflection's footprint and the tests at full size. Read coarse, a
//! texel's normal is an average over a curve -- the inside of a sconce's shade
//! averages to "straight down", which faces every reflection looking up at it
//! -- and its depth range covers centimetres of the shade: the glowing inside
//! vouched for the outside all round the rim, and the sconce reflected with
//! white specks beside its mouth (offline, 2026-09-30).
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
    // and below, the normal and `t` as a range of one depth, until widened.
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
                    let [ox, oy] = to_card_octahedral(face as usize, [nx, ny, nz]);
                    level[((row * res + y) * width + face * res + x) as usize] = t;
                    level[(((row + 1) * res + y) * width + face * res + x) as usize] = [ox, oy, t[3], t[3]];
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
            let tests = (y / (res >> l)) % 2 == 1;
            for x in 0..w {
                let four = [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dy)| prev[((2 * y + dy) * pw + 2 * x + dx) as usize]);
                next[(y * w + x) as usize] = if tests { merge_tests(&four) } else { merge_colours(&four) };
            }
        }
        chain.push((w, h, next));
    }
    for (l, (w, h, texels)) in chain.iter_mut().enumerate() {
        widen_ranges(texels, *w, *h, (res >> l).max(1));
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

/// Four colour texels as one: the colour weighted by 1 / (1 + luminance) --
/// Karis's firefly weight -- over those that saw something, and the nearest of
/// their depths. A glowing mouth is sixty times brighter than the shade round
/// it: averaged plainly, a texel a quarter mouth was still fifteen times
/// white, and every far sconce reflected as a white blob the size of the
/// footprint.
fn merge_colours(four: &[[f32; 4]; 4]) -> [f32; 4] {
    let (mut rgb, mut weight, mut near) = ([0.0f32; 3], 0.0f32, MISS);
    for t in four.iter().filter(|t| t[3] < MISS * 0.75) {
        let w = 1.0 / (1.0 + 0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2]);
        for k in 0..3 {
            rgb[k] += t[k] * w;
        }
        weight += w;
        near = near.min(t[3]);
    }
    if weight > 0.0 {
        [rgb[0] / weight, rgb[1] / weight, rgb[2] / weight, near]
    } else {
        [0.0, 0.0, 0.0, MISS]
    }
}

/// Four test texels as one: their normals' mean direction, their depths' mean
/// and the nearest of their nearest. All four are one card's, so their
/// directions are in one frame.
fn merge_tests(four: &[[f32; 4]; 4]) -> [f32; 4] {
    let (mut n, mut own, mut near, mut count) = ([0.0f32; 3], 0.0f32, f32::MAX, 0.0f32);
    for t in four.iter().filter(|t| t[3] < MISS * 0.75) {
        let d = from_hemi_octahedral([t[0], t[1]]);
        for k in 0..3 {
            n[k] += d[k];
        }
        own += t[2];
        near = near.min(t[3]);
        count += 1.0;
    }
    if count == 0.0 {
        return [0.0, 0.0, MISS, MISS];
    }
    let [ox, oy] = hemi_octahedral(n);
    [ox, oy, own / count, near]
}

/// Card `face`'s frame in the box's: its u axis, its v axis, and the way it
/// looks FROM -- the side the surfaces it sees face. See
/// `space_soup_engine::reflection_cards` (card 2a looks in through the +a
/// face; u runs along a + 1, v along a + 2).
pub fn card_frame(face: usize) -> [[f32; 3]; 3] {
    let a = (face / 2).min(2);
    let axis = |k: usize, s: f32| {
        let mut v = [0.0f32; 3];
        v[k] = s;
        v
    };
    [axis((a + 1) % 3, 1.0), axis((a + 2) % 3, 1.0), axis(a, if face % 2 == 0 { 1.0 } else { -1.0 })]
}

/// A box-frame direction seen by card `face` as two numbers in -1..1: its
/// components along the card's u and v, on the hemispherical octahedral map
/// ([`hemi_octahedral`]). What the lights block's `probe_card_vote` undoes.
pub fn to_card_octahedral(face: usize, n: [f32; 3]) -> [f32; 2] {
    let [u, v, z] = card_frame(face);
    let dot = |a: [f32; 3]| a[0] * n[0] + a[1] * n[1] + a[2] * n[2];
    hemi_octahedral([dot(u), dot(v), dot(z)])
}

/// A direction on the +z side as two numbers in -1..1 (the octahedral map's
/// upper half; Meyer et al., "On floating-point normal vectors", 2010). A
/// direction a hair past the equator -- a texel's own rounding -- is taken on
/// it. Zero maps to +z.
pub fn hemi_octahedral(n: [f32; 3]) -> [f32; 2] {
    let z = n[2].max(0.0);
    let l1 = n[0].abs() + n[1].abs() + z;
    if l1 < 1e-9 {
        return [0.0, 0.0];
    }
    [n[0] / l1, n[1] / l1]
}

/// [`hemi_octahedral`] undone, as a unit vector.
pub fn from_hemi_octahedral(o: [f32; 2]) -> [f32; 3] {
    let z = (1.0 - o[0].abs() - o[1].abs()).max(0.0);
    let l = (o[0] * o[0] + o[1] * o[1] + z * z).sqrt().max(1e-9);
    [o[0] / l, o[1] / l, z / l]
}

/// Every test texel that saw something takes, as its NEAREST depth, the
/// nearest of the test texels round it that did too, within its card. A
/// bilinear read of it is then never nearer than a surface the four texels it
/// blends saw -- a hit on the stem before the plate behind it is not "far in
/// front" of what the card saw there. The texel's own depth, in blue, is left
/// as it is: it says which side of the card's surface a hit lies. Texels that
/// saw nothing stay so, and the colour rows are left alone. `card` is a card's
/// size at this level; the atlas's rows alternate colours, tests.
pub fn widen_ranges(texels: &mut [[f32; 4]], width: u32, height: u32, card: u32) {
    let before: Vec<f32> = texels.iter().map(|t| t[3]).collect();
    for y in 0..height {
        if (y / card) % 2 == 0 {
            continue;
        }
        let (cy0, cy1) = (y / card * card, y / card * card + card - 1);
        for x in 0..width {
            let i = (y * width + x) as usize;
            if before[i] >= MISS * 0.75 {
                continue;
            }
            let (cx0, cx1) = (x / card * card, x / card * card + card - 1);
            let mut near = before[i];
            for ny in y.saturating_sub(1).max(cy0)..=(y + 1).min(cy1) {
                for nx in x.saturating_sub(1).max(cx0)..=(x + 1).min(cx1) {
                    near = near.min(before[(ny * width + nx) as usize]);
                }
            }
            texels[i][3] = near;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every direction a card can see -- facing it -- survives the card's map,
    /// on all six cards, the one looking along -z included (whose directions
    /// sat on the fold of a whole-sphere map, where two texels either side of
    /// the pole blended into the opposite pole).
    #[test]
    fn a_direction_a_card_sees_survives_its_map() {
        let dirs: [[f32; 3]; 6] = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.48, -0.6, 0.64], [-0.3, 0.5, 0.81], [0.577, 0.577, 0.577], [0.1, -0.1, 0.99]];
        for face in 0..CARD_FACES {
            let [u, v, z] = card_frame(face);
            for d in dirs {
                // `d` in the card's own frame, taken into the box's.
                let n = [0, 1, 2].map(|k| d[0] * u[k] + d[1] * v[k] + d[2] * z[k]);
                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                let n = n.map(|c| c / l);
                let c = from_hemi_octahedral(to_card_octahedral(face, n));
                let back = [0, 1, 2].map(|k| c[0] * u[k] + c[1] * v[k] + c[2] * z[k]);
                let err = (0..3).map(|k| (back[k] - n[k]).abs()).fold(0.0f32, f32::max);
                assert!(err < 1e-5, "card {face}: {n:?} came back as {back:?}");
            }
        }
        // Two directions either side of the -z card's pole, averaged as the
        // filter would their encodings: still facing that card.
        let (a, b) = (to_card_octahedral(5, [0.1, 0.0, -0.995]), to_card_octahedral(5, [-0.1, 0.0, -0.995]));
        let mid = from_hemi_octahedral([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]);
        assert!(mid[2] > 0.99, "between two directions facing the card: {mid:?}");
    }

    /// A test texel's nearest depth after widening is its neighbours' nearest
    /// inside its card, and nothing across the card's edge; its own depth
    /// stays its own.
    #[test]
    fn ranges_widen_to_the_neighbours_within_a_card() {
        // Two 2 x 2 cards side by side, one colour row above one test row.
        let (w, h, card) = (4u32, 4u32, 2u32);
        let mut t = vec![[0.0, 0.0, 0.0, MISS]; (w * h) as usize];
        let depths = [[0.1, 0.2, 0.7, 0.8], [0.3, 0.4, 0.9, 1.0]];
        for y in 2..4u32 {
            for x in 0..4u32 {
                let d = depths[(y - 2) as usize][x as usize];
                t[(y * w + x) as usize] = [0.0, 0.0, d, d];
            }
        }
        widen_ranges(&mut t, w, h, card);
        let at = |x: u32, y: u32| t[(y * w + x) as usize];
        assert_eq!([at(1, 3)[2], at(1, 3)[3]], [0.4, 0.1], "its own depth, and the first card's nearest");
        assert_eq!([at(3, 3)[2], at(3, 3)[3]], [1.0, 0.7], "the second card's nearest, not the first's");
        assert_eq!(at(0, 0)[3], MISS, "colour rows untouched");
    }
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
