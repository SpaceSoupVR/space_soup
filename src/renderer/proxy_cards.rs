use wgpu::{Device, Queue, TextureView};

pub const CARD_FACES: usize = 6;

pub const ROWS_PER_SET: u32 = 3;

pub const MAX_CARD_SETS: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct ProxyCards {

    pub resolution: u32,

    pub texels: Vec<[f32; 4]>,

    pub normals: Vec<[f32; 3]>,

    pub albedo: Vec<[f32; 3]>,
}

pub fn atlas(device: &Device, queue: &Queue, sets: &[ProxyCards]) -> Option<(TextureView, Vec<Option<u32>>)> {
    if !sets.iter().any(|c| c.resolution > 0) {
        return None;
    }
    let a = atlas_with_characters(device, queue, sets, 0);
    (a.rows.iter().any(Option::is_some)).then_some((a.view, a.rows))
}

pub const CHARACTER_CARD_SETS: usize = 1;

pub const DEFAULT_RESOLUTION: u32 = 64;

pub struct CardAtlas {

    pub texture: wgpu::Texture,
    pub view: TextureView,

    pub resolution: u32,
    pub levels: u32,

    pub rows: Vec<Option<u32>>,

    pub character_rows: Vec<u32>,

    pub pool_row: u32,
}

pub fn atlas_with_characters(
    device: &Device,
    queue: &Queue,
    sets: &[ProxyCards],
    characters: usize,
) -> CardAtlas {
    let res = sets.iter().map(|c| c.resolution).find(|&r| r > 0)
        .unwrap_or(DEFAULT_RESOLUTION);
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
            ROWS_PER_SET * (used - 1)
        }));
    }
    let character_rows: Vec<u32> = (0..characters as u32).map(|k| ROWS_PER_SET * (used + k)).collect();
    let pool_row = ROWS_PER_SET * (used + characters as u32).max(1);

    let width = res * CARD_FACES as u32;
    let height = (pool_row + super::pool_cards::POOL_ATLAS_ROWS) * res;
    let mut level: Vec<[f32; 4]> = vec![[0.0, 0.0, 0.0, MISS]; (width * height) as usize];
    for &row in &character_rows {
        let start = (row * res * width) as usize;
        level[start..start + (ROWS_PER_SET * res * width) as usize].fill([0.0; 4]);
    }
    let pools = (pool_row * res * width) as usize;
    level[pools..].fill([0.0; 4]);
    for (c, row) in sets.iter().zip(&rows) {
        let Some(row) = row else { continue };
        for face in 0..CARD_FACES as u32 {
            for y in 0..res {
                for x in 0..res {
                    let i = ((face * res + y) * res + x) as usize;
                    let t = c.texels[i];
                    let [nx, ny, nz] = c.normals[i];
                    let [ox, oy] = to_card_octahedral(face as usize, [nx, ny, nz]);
                    level[((row * res + y) * width + face * res + x) as usize] = compress(t);
                    level[(((row + 1) * res + y) * width + face * res + x) as usize] = [ox, oy, t[3], t[3]];

                    let [ar, ag, ab] = c.albedo.get(i).copied().unwrap_or([0.0; 3]);
                    level[(((row + 2) * res + y) * width + face * res + x) as usize] = [ar, ag, ab, t[3]];
                }
            }
        }
    }

    let levels = if res.is_power_of_two() { res.trailing_zeros() + 1 } else { 1 };
    let mut chain: Vec<(u32, u32, Vec<[f32; 4]>)> = vec![(width, height, level)];
    for l in 1..levels {
        let (pw, ph, prev) = chain.last().unwrap();
        let (w, h) = (pw / 2, ph / 2);
        let mut next = vec![[0.0, 0.0, 0.0, MISS]; (w * h) as usize];
        for y in 0..h {

            let tests = (y / (res >> l)) % ROWS_PER_SET == 1;
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
    CardAtlas { view: tex.create_view(&wgpu::TextureViewDescriptor::default()), texture: tex, resolution: res, levels, rows, character_rows, pool_row }
}

pub const MISS: f32 = 2.0;

pub fn compress(t: [f32; 4]) -> [f32; 4] {
    let shown = 1.0 + 0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2];
    [t[0] / shown, t[1] / shown, t[2] / shown, t[3]]
}

pub fn expand(t: [f32; 4]) -> [f32; 4] {
    let left = (1.0 - (0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2])).max(1e-3);
    [t[0] / left, t[1] / left, t[2] / left, t[3]]
}

fn merge_colours(four: &[[f32; 4]; 4]) -> [f32; 4] {
    let (mut rgb, mut count, mut near) = ([0.0f32; 3], 0.0f32, MISS);
    for t in four.iter().filter(|t| t[3] < MISS * 0.75) {
        for k in 0..3 {
            rgb[k] += t[k];
        }
        count += 1.0;
        near = near.min(t[3]);
    }
    if count > 0.0 {
        [rgb[0] / count, rgb[1] / count, rgb[2] / count, near]
    } else {
        [0.0, 0.0, 0.0, MISS]
    }
}

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

pub fn card_frame(face: usize) -> [[f32; 3]; 3] {
    let a = (face / 2).min(2);
    let axis = |k: usize, s: f32| {
        let mut v = [0.0f32; 3];
        v[k] = s;
        v
    };
    [axis((a + 1) % 3, 1.0), axis((a + 2) % 3, 1.0), axis(a, if face % 2 == 0 { 1.0 } else { -1.0 })]
}

pub fn to_card_octahedral(face: usize, n: [f32; 3]) -> [f32; 2] {
    let [u, v, z] = card_frame(face);
    let dot = |a: [f32; 3]| a[0] * n[0] + a[1] * n[1] + a[2] * n[2];
    hemi_octahedral([dot(u), dot(v), dot(z)])
}

pub fn hemi_octahedral(n: [f32; 3]) -> [f32; 2] {
    let z = n[2].max(0.0);
    let l1 = n[0].abs() + n[1].abs() + z;
    if l1 < 1e-9 {
        return [0.0, 0.0];
    }
    [n[0] / l1, n[1] / l1]
}

pub fn from_hemi_octahedral(o: [f32; 2]) -> [f32; 3] {
    let z = (1.0 - o[0].abs() - o[1].abs()).max(0.0);
    let l = (o[0] * o[0] + o[1] * o[1] + z * z).sqrt().max(1e-9);
    [o[0] / l, o[1] / l, z / l]
}

pub fn widen_ranges(texels: &mut [[f32; 4]], width: u32, height: u32, card: u32) {
    let before: Vec<f32> = texels.iter().map(|t| t[3]).collect();
    for y in 0..height {
        if (y / card) % ROWS_PER_SET != 1 {
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

    #[test]
    fn a_compressed_average_is_the_karis_weighted_one() {

        let mouth = [600.0, 520.0, 380.0, 0.4];
        let shade = [0.05, 0.04, 0.03, 0.5];
        let missed = [0.0, 0.0, 0.0, MISS];
        let merged = expand(merge_colours(&[compress(mouth), compress(shade), missed, compress(shade)]));
        let luma = |c: [f32; 4]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let (wm, ws) = (1.0 / (1.0 + luma(mouth)), 1.0 / (1.0 + luma(shade)));
        for k in 0..3 {
            let want = (mouth[k] * wm + 2.0 * shade[k] * ws) / (wm + 2.0 * ws);
            assert!((merged[k] - want).abs() < 1e-3 * want.max(1e-3), "channel {k}: {} vs {want}", merged[k]);
        }

        assert!(luma(merged) < 1.0, "{merged:?}");
        assert_eq!(merged[3], 0.4, "the nearest depth");
        for c in [mouth, shade] {
            let back = expand(compress(c));
            assert!((0..3).all(|k| (back[k] - c[k]).abs() < 1e-3 * c[k].max(1e-3)), "{c:?} -> {back:?}");
        }
    }

    #[test]
    fn a_direction_a_card_sees_survives_its_map() {
        let dirs: [[f32; 3]; 6] = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.48, -0.6, 0.64], [-0.3, 0.5, 0.81], [0.577, 0.577, 0.577], [0.1, -0.1, 0.99]];
        for face in 0..CARD_FACES {
            let [u, v, z] = card_frame(face);
            for d in dirs {

                let n = [0, 1, 2].map(|k| d[0] * u[k] + d[1] * v[k] + d[2] * z[k]);
                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                let n = n.map(|c| c / l);
                let c = from_hemi_octahedral(to_card_octahedral(face, n));
                let back = [0, 1, 2].map(|k| c[0] * u[k] + c[1] * v[k] + c[2] * z[k]);
                let err = (0..3).map(|k| (back[k] - n[k]).abs()).fold(0.0f32, f32::max);
                assert!(err < 1e-5, "card {face}: {n:?} came back as {back:?}");
            }
        }

        let (a, b) = (to_card_octahedral(5, [0.1, 0.0, -0.995]), to_card_octahedral(5, [-0.1, 0.0, -0.995]));
        let mid = from_hemi_octahedral([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]);
        assert!(mid[2] > 0.99, "between two directions facing the card: {mid:?}");
    }

    #[test]
    fn ranges_widen_to_the_neighbours_within_a_card() {

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
