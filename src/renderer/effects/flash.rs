use super::{fbm, smoothstep, ATLAS_SIZE};
use glam::Vec3;

pub(super) fn lobe_frame(variant: u32, form: u32) -> Vec<u8> {
    let n = ATLAS_SIZE as usize;
    let seed = 0xf1a5_0000u64 + u64::from(variant) * 97 + u64::from(form) * 13;
    let z = 0.37 + variant as f32 * 1.9 + form as f32 * 0.7;
    let reach = if form == 2 { 0.7 } else { 1.0 };
    let mut out = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let px = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let h = 1.0 - (y as f32 + 0.5) / n as f32;

            let along = (h / reach).min(1.0);
            let width = 0.36 * along.powf(0.45) * (1.0 - along).powf(0.7) / 0.46;

            let middle = 0.22 * h * fbm(Vec3::new(h * 2.3, z, 1.3), 2, seed);
            let apart = if form == 1 { 0.17 * smoothstep(0.3, 0.85, h) } else { 0.0 };
            let off = ((px - middle).abs() - apart).abs();

            let torn = 1.0 + (0.55 + 0.9 * h) * fbm(Vec3::new(px * 3.2, h * 5.5, z), 4, seed ^ 0x55);
            let w = (width * torn * if form == 1 { 1.0 - 0.35 * smoothstep(0.3, 0.85, h) } else { 1.0 }).max(0.0);
            let body = if w > 1e-4 { 1.0 - smoothstep(0.3 * w, w, off) } else { 0.0 };

            let streak = (0.8 + 0.6 * fbm(Vec3::new(px * 9.0, h * 1.7, z + 4.2), 3, seed ^ 0x77)).clamp(0.3, 1.3);
            let edge = smoothstep(1.0, 0.9, px.abs()) * smoothstep(0.0, 0.03, h) * smoothstep(1.0, 0.97, h);
            let heat = (body.powf(0.8) * (1.0 - 0.5 * along) * streak).clamp(0.0, 1.0);
            let cover = (body * (1.3 - 0.5 * along) * streak).clamp(0.0, 1.0) * edge;
            let o = (y * n + x) * 4;
            out[o] = (heat * 255.0).round() as u8;
            out[o + 1] = (body * 255.0).round() as u8;
            out[o + 2] = 0;
            out[o + 3] = (cover * 255.0).round() as u8;
        }
    }
    out
}

pub(super) fn core_frame(variant: u32) -> Vec<u8> {
    let n = ATLAS_SIZE as usize;
    let seed = 0xc0e0_0000u64 + u64::from(variant) * 131;
    let mut out = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let px = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let py = 1.0 - (y as f32 + 0.5) / n as f32 * 2.0;
            let r = (px * px + py * py).sqrt();
            let (c, s) = if r > 1e-5 { (px / r, py / r) } else { (1.0, 0.0) };
            let rim = (0.6 + 0.5 * fbm(Vec3::new(c * 1.7, s * 1.7, 0.5 + variant as f32 * 2.3), 3, seed)).clamp(0.3, 0.9);
            let body = 1.0 - smoothstep(0.2 * rim, rim, r);
            let grain = (0.85 + 0.4 * fbm(Vec3::new(px * 5.0, py * 5.0, 2.1), 3, seed ^ 0x31)).clamp(0.5, 1.2);
            let heat = (body.powf(0.7) * grain).clamp(0.0, 1.0);
            let cover = (body.powf(0.6) * grain).clamp(0.0, 1.0) * smoothstep(1.0, 0.9, r);
            let o = (y * n + x) * 4;
            out[o] = (heat * 255.0).round() as u8;
            out[o + 1] = (body * 255.0).round() as u8;
            out[o + 2] = 0;
            out[o + 3] = (cover * 255.0).round() as u8;
        }
    }
    out
}
