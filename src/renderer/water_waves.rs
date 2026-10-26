use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use wgpu::*;

pub const N: u32 = 64;

pub const CASCADES: usize = 3;

pub const MIPS: u32 = 7;
const G: f32 = 9.81;

const PAIRS: u32 = 6;

const SPLIT_CELLS: f32 = 4.0;
const GROUP: u32 = 8;

const FOAM_SECONDS: f64 = 1.6;

const FOAM_JACOBIAN: [f32; 2] = [0.72, 0.3];

pub const FOAM_SIZE: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveParams {

    pub wind_speed: f32,

    pub wind_dir: f32,

    pub fetch: f32,

    pub depth: f32,

    pub choppiness: f32,

    pub tile: [f32; CASCADES],

    pub loop_seconds: f32,
    pub seed: u64,
}

impl Default for WaveParams {

    fn default() -> Self {
        Self {
            wind_speed: 7.0,
            wind_dir: 0.6,
            fetch: 800.0,
            depth: 3.0,
            choppiness: 0.8,
            tile: [24.0, 5.3, 1.17],
            loop_seconds: 240.0,
            seed: 0x5eed_0a7e,
        }
    }
}

impl WaveParams {

    fn band(&self, c: usize) -> (f32, f32) {
        let split = |c: usize| SPLIT_CELLS * 2.0 * PI / self.tile[c + 1];
        let lo = if c == 0 { 0.0 } else { split(c - 1) };
        let hi = if c + 1 == CASCADES { f32::INFINITY } else { split(c) };
        (lo, hi)
    }

    fn wave_vector(&self, c: usize, xi: usize, zi: usize) -> Option<(f32, f32)> {
        if xi == 0 || zi == 0 {
            return None;
        }
        let dk = 2.0 * PI / self.tile[c];
        let half = N as f32 / 2.0;
        let (kx, kz) = ((xi as f32 - half) * dk, (zi as f32 - half) * dk);
        let k = (kx * kx + kz * kz).sqrt();
        let (lo, hi) = self.band(c);
        (k >= lo && k < hi && k > 0.0).then_some((kx, kz))
    }

    pub fn omega(&self, k: f32) -> f32 {
        (G * k * (k * self.depth).tanh()).sqrt()
    }

    pub fn omega_looped(&self, k: f32) -> f32 {
        let w0 = 2.0 * PI / self.loop_seconds;
        ((self.omega(k) / w0).round() * w0).max(w0)
    }

    pub fn peak_omega(&self) -> f32 {
        22.0 * (G * G / (self.wind_speed.max(0.5) * self.fetch.max(1.0))).powf(1.0 / 3.0)
    }

    pub fn jonswap(&self, omega: f32) -> f32 {
        let u = self.wind_speed.max(0.5);
        let alpha = 0.076 * (u * u / (self.fetch.max(1.0) * G)).powf(0.22);
        let wp = self.peak_omega();
        let sigma = if omega <= wp { 0.07 } else { 0.09 };
        let r = (-(omega - wp).powi(2) / (2.0 * sigma * sigma * wp * wp)).exp();
        alpha * G * G / omega.powi(5) * (-1.25 * (wp / omega).powi(4)).exp() * 3.3f32.powf(r)
    }

    fn spread_s(&self, omega: f32) -> f32 {
        let wp = self.peak_omega();
        let x = omega / wp;
        if x < 1.0 {
            6.97 * x.powf(4.06)
        } else {
            let mu = -2.33 - 1.45 * (self.wind_speed * wp / G - 1.17);
            9.77 * x.powf(mu)
        }
        .clamp(0.5, 40.0)
    }

    pub fn spectrum(&self, kx: f32, kz: f32) -> f32 {
        let k = (kx * kx + kz * kz).sqrt();
        if k < 1e-6 {
            return 0.0;
        }
        let omega = self.omega(k);
        let th = (k * self.depth).tanh();
        let domega_dk = G * (th + k * self.depth * (1.0 - th * th)) / (2.0 * omega);
        let s = self.spread_s(omega);
        let theta = kz.atan2(kx) - self.wind_dir;
        let spread = (0.5 * theta).cos().abs().powf(2.0 * s) / spread_norm(s);
        self.jonswap(omega) * spread * domega_dk / k
    }

    pub fn waves(&self) -> Vec<[[f32; 4]; 2]> {
        let n = N as usize;
        let mut rng = SplitMix(self.seed);
        let mut out = Vec::with_capacity(CASCADES * n * n);
        for c in 0..CASCADES {
            let dk = 2.0 * PI / self.tile[c];

            let mut h0 = vec![[0.0f32; 2]; n * n];
            for zi in 0..n {
                for xi in 0..n {
                    let (g1, g2) = rng.gaussian_pair();
                    if let Some((kx, kz)) = self.wave_vector(c, xi, zi) {
                        let a = (self.spectrum(kx, kz) * dk * dk / 4.0).sqrt();
                        h0[zi * n + xi] = [g1 * a, g2 * a];
                    }
                }
            }
            for zi in 0..n {
                for xi in 0..n {
                    let here = h0[zi * n + xi];

                    let there = h0[((n - zi) % n) * n + (n - xi) % n];
                    let kw = match self.wave_vector(c, xi, zi) {
                        Some((kx, kz)) => [kx, kz, self.omega_looped((kx * kx + kz * kz).sqrt()), 0.0],
                        None => [0.0; 4],
                    };
                    out.push([[here[0], here[1], there[0], -there[1]], kw]);
                }
            }
        }
        out
    }

    fn sum_over_waves(&self, f: impl Fn(usize, f32, f32, f32) -> f64) -> f32 {
        let n = N as usize;
        let mut total = 0.0f64;
        for c in 0..CASCADES {
            let dk = 2.0 * PI / self.tile[c];
            for zi in 0..n {
                for xi in 0..n {
                    if let Some((kx, kz)) = self.wave_vector(c, xi, zi) {
                        total += f(c, kx, kz, self.spectrum(kx, kz) * dk * dk);
                    }
                }
            }
        }
        total as f32
    }

    pub fn sums(&self) -> WaveSums {
        let key = WaveSums::key(self);
        if let Some(found) = WAVE_SUMS.lock().ok().and_then(|kept| kept.iter().find(|(k, _)| *k == key).map(|(_, s)| *s)) {
            return found;
        }
        let made = WaveSums::of(self);
        if let Ok(mut kept) = WAVE_SUMS.lock() {

            if kept.len() >= 256 {
                kept.clear();
            }
            kept.push((key, made));
        }
        made
    }

    pub fn height_variance(&self) -> f32 {
        self.sums().height_variance
    }

    pub fn significant_height(&self) -> f32 {
        4.0 * self.height_variance().sqrt()
    }

    pub fn slope_variance_beyond(&self, k_cut: f32) -> f32 {
        if k_cut == 0.0 {
            return self.sums().slope_variance;
        }
        self.sum_over_waves(|_, kx, kz, e| {
            let k2 = kx * kx + kz * kz;
            if k2 >= k_cut * k_cut { f64::from(k2 * e) } else { 0.0 }
        })
    }

    pub fn cascade_slope_variance(&self, c: usize) -> f32 {
        self.sums().cascade_slope_variance[c]
    }

    pub fn lost_slope_variance(&self) -> [[f32; 4]; MIPS as usize] {
        self.sums().lost
    }

    pub fn foam_gains(&self) -> [f32; CASCADES] {
        let total: f32 = (0..CASCADES).map(|c| self.cascade_slope_variance(c)).sum();
        let mut out = [0.0; CASCADES];
        for (c, g) in out.iter_mut().enumerate().take(CASCADES - 1) {
            *g = (total / self.cascade_slope_variance(c).max(1e-12)).sqrt();
        }
        out
    }

    pub fn unresolved_slope_variance(&self) -> f32 {
        let measured = 0.003 + 0.005_12 * self.wind_speed.max(0.0);
        (measured - self.slope_variance_beyond(0.0)).max(0.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveSums {

    pub height_variance: f32,

    pub slope_variance: f32,

    pub cascade_slope_variance: [f32; CASCADES],

    pub lost: [[f32; 4]; MIPS as usize],
}

type WaveSumsKey = [u32; 4 + CASCADES];

static WAVE_SUMS: std::sync::Mutex<Vec<(WaveSumsKey, WaveSums)>> = std::sync::Mutex::new(Vec::new());

#[cfg(test)]
thread_local! {

    static SPECTRUM_PASSES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

impl WaveSums {
    fn key(p: &WaveParams) -> WaveSumsKey {
        let mut key = [p.wind_speed.to_bits(), p.wind_dir.to_bits(), p.fetch.to_bits(), p.depth.to_bits(), 0, 0, 0];
        for (k, t) in key[4..].iter_mut().zip(p.tile) {
            *k = t.to_bits();
        }
        key
    }

    pub fn of(p: &WaveParams) -> Self {
        #[cfg(test)]
        SPECTRUM_PASSES.with(|n| n.set(n.get() + 1));
        let n = N as usize;
        let mut height = 0.0f64;
        let mut slope = 0.0f64;
        let mut cascade = [0.0f64; CASCADES];
        let mut lost = [[0.0f64; CASCADES]; MIPS as usize];
        for c in 0..CASCADES {
            let dk = 2.0 * PI / p.tile[c];

            let nyquist: [f32; MIPS as usize] = std::array::from_fn(|level| {
                let texels = (N >> level).max(1) as f32;
                if level + 1 == MIPS as usize { 0.0 } else { PI * texels / p.tile[c] }
            });
            for zi in 0..n {
                for xi in 0..n {
                    let Some((kx, kz)) = p.wave_vector(c, xi, zi) else { continue };
                    let e = p.spectrum(kx, kz) * dk * dk;
                    let k2e = f64::from((kx * kx + kz * kz) * e);
                    height += f64::from(e);
                    slope += k2e;
                    cascade[c] += k2e;
                    for (level, lane) in lost.iter_mut().enumerate() {
                        if kx.abs() > nyquist[level] || kz.abs() > nyquist[level] {
                            lane[c] += k2e;
                        }
                    }
                }
            }
        }
        let mut out_lost = [[0.0f32; 4]; MIPS as usize];
        for (row, sums) in out_lost.iter_mut().zip(&lost) {
            for (lane, sum) in row.iter_mut().zip(sums) {
                *lane = *sum as f32;
            }
        }
        Self { height_variance: height as f32, slope_variance: slope as f32, cascade_slope_variance: cascade.map(|v| v as f32), lost: out_lost }
    }
}

fn spread_norm(s: f32) -> f32 {
    let m = 512;
    let h = 2.0 * PI / m as f32;
    let f = |i: usize| f32::midpoint(-PI, i as f32 * h).cos().abs().powf(2.0 * s);
    let mut sum = f(0) + f(m);
    for i in 1..m {
        sum += if i % 2 == 1 { 4.0 } else { 2.0 } * f(i);
    }
    sum * h / 3.0
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        ((self.next() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    fn gaussian_pair(&mut self) -> (f32, f32) {
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        (r * (2.0 * PI * u2).cos(), r * (2.0 * PI * u2).sin())
    }
}

pub fn foam_texture() -> Vec<u8> {
    let size = FOAM_SIZE as usize;
    let mut rng = SplitMix(0xf0a3_b0b0);
    let mut layer = |cells: usize| {
        let points: Vec<(f32, f32)> = (0..cells * cells)
            .map(|i| ((i % cells) as f32 + rng.uniform(), (i / cells) as f32 + rng.uniform()))
            .collect();
        move |x: f32, y: f32| {
            let (cx, cy) = (x.floor() as i32, y.floor() as i32);
            let (mut f1, mut f2) = (f32::MAX, f32::MAX);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (gx, gy) = (cx + dx, cy + dy);
                    let (wx, wy) = (gx.rem_euclid(cells as i32) as usize, gy.rem_euclid(cells as i32) as usize);
                    let p = points[wy * cells + wx];

                    let (px, py) = (p.0 - wx as f32 + gx as f32, p.1 - wy as f32 + gy as f32);
                    let d = ((px - x).powi(2) + (py - y).powi(2)).sqrt();
                    if d < f1 {
                        f2 = f1;
                        f1 = d;
                    } else if d < f2 {
                        f2 = d;
                    }
                }
            }
            (f1, f2)
        }
    };
    let (big, small) = (layer(14), layer(41));
    let mut out = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let (u, v) = (x as f32 / size as f32, y as f32 / size as f32);
            let (b1, b2) = big(u * 14.0, v * 14.0);
            let (s1, s2) = small(u * 41.0, v * 41.0);

            let wall = |f1: f32, f2: f32, width: f32| (1.0 - ((f2 - f1) / width).min(1.0)).powf(1.5) + 0.25 * f1.min(1.0);
            let v = (0.65 * wall(b1, b2, 0.22) + 0.55 * wall(s1, s2, 0.3)).min(1.0);
            out.push((v * 255.0).round() as u8);
        }
    }
    out
}

fn shader() -> String {
    let (n, half, stages, cascades, pairs) = (N, N / 2, N.trailing_zeros(), CASCADES, PAIRS);
    format!(
        r"
struct Waves {{ time: vec4<f32>, foam: vec4<f32>, gain: vec4<f32> }}

struct Wave {{ amp: vec4<f32>, kw: vec4<f32> }}
@group(0) @binding(0) var<uniform> waves: Waves;
@group(0) @binding(1) var<storage, read> spectrum: array<Wave>;

@group(0) @binding(2) var<storage, read_write> spec: array<vec2<f32>>;
@group(0) @binding(3) var displacement: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var derivatives: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var curvature: texture_storage_2d_array<rgba16float, write>;

@group(0) @binding(6) var previous: texture_2d_array<f32>;

const N: u32 = {n}u;
const PI: f32 = 3.14159265358979;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}}

fn times_i(a: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(-a.y, a.x);
}}
fn spec_index(c: u32, pair: u32, row: u32, col: u32) -> u32 {{
    return ((c * {pairs}u + pair) * N + row) * N + col;
}}

@compute @workgroup_size({GROUP}, {GROUP}, 1)
fn evolve(@builtin(global_invocation_id) id: vec3<u32>) {{
    let c = id.z;
    if (id.x >= N || id.y >= N || c >= {cascades}u) {{
        return;
    }}
    let w = spectrum[(c * N + id.y) * N + id.x];
    let phase = w.kw.z * waves.time.x;
    let e = vec2<f32>(cos(phase), sin(phase));

    let h = cmul(w.amp.xy, vec2<f32>(e.x, -e.y)) + cmul(w.amp.zw, e);
    let kv = w.kw.xy;
    let k = length(kv);
    var u = vec2<f32>(0.0);
    if (k > 1e-6) {{
        u = kv / k;
    }}
    let lambda = waves.time.y;

    let ih = times_i(h);
    let dx = u.x * lambda * ih;
    let dz = u.y * lambda * ih;
    let hx = kv.x * ih;
    let hz = kv.y * ih;
    let dxx = -kv.x * u.x * lambda * h;
    let dzz = -kv.y * u.y * lambda * h;
    let dxz = -kv.y * u.x * lambda * h;
    let hxx = -kv.x * kv.x * h;
    let hzz = -kv.y * kv.y * h;
    let hxz = -kv.x * kv.y * h;
    spec[spec_index(c, 0u, id.y, id.x)] = dx + times_i(h);
    spec[spec_index(c, 1u, id.y, id.x)] = dz + times_i(dxz);
    spec[spec_index(c, 2u, id.y, id.x)] = hx + times_i(hz);
    spec[spec_index(c, 3u, id.y, id.x)] = dxx + times_i(dzz);
    spec[spec_index(c, 4u, id.y, id.x)] = hxx + times_i(hzz);
    spec[spec_index(c, 5u, id.y, id.x)] = hxz;
}}

var<workgroup> buf: array<array<vec2<f32>, {n}>, 4>;

var<workgroup> twiddle: array<vec2<f32>, {half}>;

fn inverse_fft(line: u32, t: u32) -> u32 {{
    var src = 0u;
    for (var s = 0u; s < {stages}u; s = s + 1u) {{
        let ns = 1u << s;
        let v0 = buf[line * 2u + src][t];
        let v1 = cmul(buf[line * 2u + src][t + N / 2u], twiddle[(t % ns) * ({half}u / ns)]);
        let d = (t / ns) * ns * 2u + t % ns;
        buf[line * 2u + 1u - src][d] = v0 + v1;
        buf[line * 2u + 1u - src][d + ns] = v0 - v1;
        src = 1u - src;
        workgroupBarrier();
    }}
    return src;
}}

fn turns(i: u32) {{
    if (i < {half}u) {{
        let a = PI * f32(i) / f32({half}u);
        twiddle[i] = vec2<f32>(cos(a), sin(a));
    }}
}}

@compute @workgroup_size({n}, 1, 1)
fn rows(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let line = lid.x / {half}u;
    let t = lid.x % {half}u;
    let row = wid.x * 2u + line;
    let pair = wid.y;
    let c = wid.z;
    turns(lid.x);
    buf[line * 2u][t] = spec[spec_index(c, pair, row, t)];
    buf[line * 2u][t + N / 2u] = spec[spec_index(c, pair, row, t + N / 2u)];
    workgroupBarrier();
    let src = inverse_fft(line, t);
    spec[spec_index(c, pair, row, t)] = buf[line * 2u + src][t];
    spec[spec_index(c, pair, row, t + N / 2u)] = buf[line * 2u + src][t + N / 2u];
}}

@compute @workgroup_size({n}, 1, 1)
fn columns(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let line = lid.x / {half}u;
    let t = lid.x % {half}u;
    let col = wid.x * 2u + line;
    let pair = wid.y;
    let c = wid.z;
    turns(lid.x);
    buf[line * 2u][t] = spec[spec_index(c, pair, t, col)];
    buf[line * 2u][t + N / 2u] = spec[spec_index(c, pair, t + N / 2u, col)];
    workgroupBarrier();
    let src = inverse_fft(line, t);
    spec[spec_index(c, pair, t, col)] = buf[line * 2u + src][t];
    spec[spec_index(c, pair, t + N / 2u, col)] = buf[line * 2u + src][t + N / 2u];
}}

@compute @workgroup_size({GROUP}, {GROUP}, 1)
fn assemble(@builtin(global_invocation_id) id: vec3<u32>) {{
    let c = id.z;
    if (id.x >= N || id.y >= N || c >= {cascades}u) {{
        return;
    }}
    let sign = select(1.0, -1.0, ((id.x + id.y) & 1u) == 1u);
    let p0 = spec[spec_index(c, 0u, id.y, id.x)] * sign;
    let p1 = spec[spec_index(c, 1u, id.y, id.x)] * sign;
    let p2 = spec[spec_index(c, 2u, id.y, id.x)] * sign;
    let p3 = spec[spec_index(c, 3u, id.y, id.x)] * sign;
    let p4 = spec[spec_index(c, 4u, id.y, id.x)] * sign;
    let p5 = spec[spec_index(c, 5u, id.y, id.x)] * sign;
    let jacobian = (1.0 + p3.x) * (1.0 + p3.y) - p1.y * p1.y;
    let gain = waves.gain[c];
    let made = select(0.0, smoothstep(waves.foam.x, waves.foam.y, 1.0 + (jacobian - 1.0) * gain), gain > 0.0);
    let kept = textureLoad(previous, vec2<i32>(id.xy), i32(c), 0).w * waves.time.z;
    let foam = max(kept, made);
    textureStore(displacement, vec2<i32>(id.xy), i32(c), vec4<f32>(p0.x, p0.y, p1.x, p1.y));
    textureStore(derivatives, vec2<i32>(id.xy), i32(c), vec4<f32>(p2.x, p2.y, p3.x, p3.y));
    textureStore(curvature, vec2<i32>(id.xy), i32(c), vec4<f32>(p4.x, p4.y, p5.x, foam));
}}
"
    )
}

fn mip_shader() -> String {
    format!(
        r"
@group(0) @binding(0) var src_a: texture_2d_array<f32>;
@group(0) @binding(1) var src_b: texture_2d_array<f32>;
@group(0) @binding(2) var src_c: texture_2d_array<f32>;
@group(0) @binding(3) var dst_a: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var dst_b: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var dst_c: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size({GROUP}, {GROUP}, 1)
fn halve(@builtin(global_invocation_id) id: vec3<u32>) {{
    let size = textureDimensions(dst_a);
    if (id.x >= size.x || id.y >= size.y) {{
        return;
    }}
    let p = vec2<i32>(id.xy) * 2;
    let l = i32(id.z);
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    var c = vec4<f32>(0.0);
    for (var k = 0; k < 4; k = k + 1) {{
        let q = p + vec2<i32>(k & 1, k >> 1u);
        a += textureLoad(src_a, q, l, 0);
        b += textureLoad(src_b, q, l, 0);
        c += textureLoad(src_c, q, l, 0);
    }}
    textureStore(dst_a, vec2<i32>(id.xy), l, a * 0.25);
    textureStore(dst_b, vec2<i32>(id.xy), l, b * 0.25);
    textureStore(dst_c, vec2<i32>(id.xy), l, c * 0.25);
}}
"
    )
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveUniform {

    time: [f32; 4],

    foam: [f32; 4],

    gain: [f32; 4],
}

pub struct WaveField {
    pub params: WaveParams,

    pub significant_height: f32,
    uniform: Buffer,

    pub displacement: [Texture; 2],
    pub displacement_views: [TextureView; 2],

    pub derivatives: Texture,
    pub derivatives_view: TextureView,

    pub curvature: [Texture; 2],
    pub curvature_views: [TextureView; 2],

    pub foam: Texture,
    pub foam_view: TextureView,

    foam_gains: [f32; CASCADES],

    current: AtomicUsize,

    primed: AtomicBool,

    last_seconds: AtomicU64,
    evolve: ComputePipeline,
    rows: ComputePipeline,
    columns: ComputePipeline,
    assemble: ComputePipeline,
    mip: ComputePipeline,

    groups: [BindGroup; 2],
    mip_groups: [Vec<BindGroup>; 2],
}

impl WaveField {
    pub fn new(device: &Device, queue: &Queue, params: WaveParams) -> Self {
        let waves = params.waves();
        let spectrum = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("water_waves_spectrum"),
                contents: bytemuck::cast_slice(&waves),
                usage: BufferUsages::STORAGE,
            },
        );
        let spec = device.create_buffer(&BufferDescriptor {
            label: Some("water_waves_fields"),
            size: CASCADES as u64 * u64::from(PAIRS) * u64::from(N * N) * 8,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let uniform = device.create_buffer(&BufferDescriptor {
            label: Some("water_waves_uniform"),
            size: std::mem::size_of::<WaveUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let surface = |label| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: N, height: N, depth_or_array_layers: CASCADES as u32 },
                mip_level_count: MIPS,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let displacement = [surface("water_waves_displacement_a"), surface("water_waves_displacement_b")];
        let derivatives = surface("water_waves_derivatives");
        let curvature = [surface("water_waves_curvature_a"), surface("water_waves_curvature_b")];
        let level = |t: &Texture, l: u32| {
            t.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                base_mip_level: l,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let whole = |t: &Texture| t.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() });

        let foam_levels = FOAM_SIZE.trailing_zeros() + 1;
        let foam = device.create_texture(&TextureDescriptor {
            label: Some("water_foam"),
            size: Extent3d { width: FOAM_SIZE, height: FOAM_SIZE, depth_or_array_layers: 1 },
            mip_level_count: foam_levels,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut data = foam_texture();
        let mut side = FOAM_SIZE;
        for l in 0..foam_levels {
            queue.write_texture(
                TexelCopyTextureInfo { texture: &foam, mip_level: l, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                &data,
                TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(side), rows_per_image: Some(side) },
                Extent3d { width: side, height: side, depth_or_array_layers: 1 },
            );
            if side > 1 {
                let half = (side / 2) as usize;
                let s = side as usize;
                data = (0..half * half)
                    .map(|i| {
                        let (x, y) = (i % half * 2, i / half * 2);
                        let sum: u32 = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)].iter().map(|&(a, b)| u32::from(data[b * s + a])).sum();
                        ((sum + 2) / 4) as u8
                    })
                    .collect();
                side /= 2;
            }
        }
        let foam_view = foam.create_view(&TextureViewDescriptor::default());

        let entry = |binding, ty| BindGroupLayoutEntry { binding, visibility: ShaderStages::COMPUTE, ty, count: None };
        let storage = |read_only| BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let written = BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba16Float,
            view_dimension: TextureViewDimension::D2Array,
        };
        let read = BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2Array,
            multisampled: false,
        };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("water_waves_layout"),
            entries: &[
                entry(0, BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                entry(1, storage(true)),
                entry(2, storage(false)),
                entry(3, written),
                entry(4, written),
                entry(5, written),
                entry(6, read),
            ],
        });
        let mip_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("water_waves_mip_layout"),
            entries: &[entry(0, read), entry(1, read), entry(2, read), entry(3, written), entry(4, written), entry(5, written)],
        });
        let pipeline = |module: &ShaderModule, layout: &BindGroupLayout, entry: &str| {
            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("water_waves_pipeline_layout"),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("water_waves"),
            source: ShaderSource::Wgsl(shader().into()),
        });
        let mip_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("water_waves_mips"),
            source: ShaderSource::Wgsl(mip_shader().into()),
        });
        let group = |t: usize| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("water_waves"),
                layout: &layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: spectrum.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: spec.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&level(&displacement[t], 0)) },
                    BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&level(&derivatives, 0)) },
                    BindGroupEntry { binding: 5, resource: BindingResource::TextureView(&level(&curvature[t], 0)) },
                    BindGroupEntry { binding: 6, resource: BindingResource::TextureView(&level(&curvature[1 - t], 0)) },
                ],
            })
        };
        let mip_groups = |t: usize| {
            (1..MIPS)
                .map(|l| {
                    let pair = [&displacement[t], &derivatives, &curvature[t]];
                    device.create_bind_group(&BindGroupDescriptor {
                        label: Some("water_waves_mip"),
                        layout: &mip_layout,
                        entries: &[
                            BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&level(pair[0], l - 1)) },
                            BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&level(pair[1], l - 1)) },
                            BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&level(pair[2], l - 1)) },
                            BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&level(pair[0], l)) },
                            BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&level(pair[1], l)) },
                            BindGroupEntry { binding: 5, resource: BindingResource::TextureView(&level(pair[2], l)) },
                        ],
                    })
                })
                .collect::<Vec<_>>()
        };
        let groups = [group(0), group(1)];
        let mip_groups = [mip_groups(0), mip_groups(1)];
        Self {
            params,
            uniform,
            displacement_views: [whole(&displacement[0]), whole(&displacement[1])],
            derivatives_view: whole(&derivatives),
            curvature_views: [whole(&curvature[0]), whole(&curvature[1])],
            foam_view,
            groups,
            mip_groups,
            displacement,
            derivatives,
            curvature,
            foam,
            foam_gains: params.foam_gains(),
            significant_height: params.significant_height(),
            current: AtomicUsize::new(1),
            primed: AtomicBool::new(false),
            last_seconds: AtomicU64::new(0f64.to_bits()),
            evolve: pipeline(&module, &layout, "evolve"),
            rows: pipeline(&module, &layout, "rows"),
            columns: pipeline(&module, &layout, "columns"),
            assemble: pipeline(&module, &layout, "assemble"),
            mip: pipeline(&mip_module, &mip_layout, "halve"),
        }
    }

    pub fn reprime(&self) {
        self.primed.store(false, Ordering::Relaxed);
    }

    pub fn current(&self) -> usize {
        self.current.load(Ordering::Relaxed)
    }

    pub fn update(&self, queue: &Queue, encoder: &mut CommandEncoder, seconds: f64) {
        let p = &self.params;
        let wrapped = seconds.rem_euclid(f64::from(p.loop_seconds)) as f32;
        let first = !self.primed.swap(true, Ordering::Relaxed);
        let last = f64::from_bits(self.last_seconds.swap(seconds.to_bits(), Ordering::Relaxed));

        let dt = if first { 0.0 } else { (seconds - last).clamp(0.0, 0.1) };
        let keep = (-dt / FOAM_SECONDS).exp() as f32;
        let u = WaveUniform {
            time: [wrapped, p.choppiness, if first { 0.0 } else { keep }, 0.0],
            foam: [FOAM_JACOBIAN[0], FOAM_JACOBIAN[1], 0.0, 0.0],
            gain: [self.foam_gains[0], self.foam_gains[1], self.foam_gains[2], 0.0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        let next = 1 - self.current();
        let both = [1 - next, next];
        let targets: &[usize] = if first { &both } else { &both[1..] };
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("water_waves"), timestamp_writes: None });
        let cells = N.div_ceil(GROUP);
        for &t in targets {
            pass.set_bind_group(0, &self.groups[t], &[]);
            pass.set_pipeline(&self.evolve);
            pass.dispatch_workgroups(cells, cells, CASCADES as u32);

            pass.set_pipeline(&self.rows);
            pass.dispatch_workgroups(N / 2, PAIRS, CASCADES as u32);
            pass.set_pipeline(&self.columns);
            pass.dispatch_workgroups(N / 2, PAIRS, CASCADES as u32);
            pass.set_pipeline(&self.assemble);
            pass.dispatch_workgroups(cells, cells, CASCADES as u32);
            pass.set_pipeline(&self.mip);
            for (l, group) in self.mip_groups[t].iter().enumerate() {
                let size = (N >> (l + 1)).max(1);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(size.div_ceil(GROUP), size.div_ceil(GROUP), CASCADES as u32);
            }
        }
        self.current.store(next, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct C(f64, f64);
    impl std::ops::Mul for C {
        type Output = C;
        fn mul(self, o: C) -> C {
            C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
        }
    }
    impl std::ops::Add for C {
        type Output = C;
        fn add(self, o: C) -> C {
            C(self.0 + o.0, self.1 + o.1)
        }
    }

    const FIELDS: usize = 11;

    fn direct(p: &WaveParams, waves: &[[[f32; 4]; 2]], c: usize, x: u32, z: u32, t: f64) -> [C; FIELDS] {
        let n = N as usize;
        let cell = f64::from(p.tile[c]) / n as f64;
        let (px, pz) = (f64::from(x) * cell, f64::from(z) * cell);
        let lam = f64::from(p.choppiness);
        let mut out = [C(0.0, 0.0); FIELDS];
        for w in &waves[c * n * n..(c + 1) * n * n] {
            let [a, kw] = *w;
            let (kx, kz, omega) = (f64::from(kw[0]), f64::from(kw[1]), f64::from(kw[2]));
            let k = (kx * kx + kz * kz).sqrt();
            if k == 0.0 {
                continue;
            }
            let turn = |phase: f64| C(phase.cos(), phase.sin());
            let h = C(f64::from(a[0]), f64::from(a[1])) * turn(-omega * t) + C(f64::from(a[2]), f64::from(a[3])) * turn(omega * t);
            let i = C(0.0, 1.0);
            let dx = i * C(kx / k * lam, 0.0) * h;
            let dz = i * C(kz / k * lam, 0.0) * h;
            let (ikx, ikz) = (i * C(kx, 0.0), i * C(kz, 0.0));
            let e = turn(kx * px + kz * pz);
            let fields = [dx, h, dz, ikz * dx, ikx * h, ikz * h, ikx * dx, ikz * dz, ikx * ikx * h, ikz * ikz * h, ikx * ikz * h];
            for (o, f) in out.iter_mut().zip(fields) {
                *o = *o + f * e;
            }
        }
        out
    }

    fn half_to_f32(h: u16) -> f32 {
        let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let e = i32::from((h >> 10) & 0x1f);
        let m = f32::from(h & 0x3ff);
        s * match e {
            0 => m * 2f32.powi(-24),
            31 => f32::INFINITY,
            _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
        }
    }

    fn gpu_surface(p: WaveParams, times: &[f64]) -> Option<Vec<[f32; FIELDS + 1]>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let field = WaveField::new(&device, &queue, p);
        for (i, &t) in times.iter().enumerate() {
            let mut encoder = device.create_command_encoder(&Default::default());
            field.update(&queue, &mut encoder, t);
            if i + 1 < times.len() {
                queue.submit(Some(encoder.finish()));
                continue;
            }
            let row = (N * 8).div_ceil(256) * 256;
            let mut read = |tex: &Texture| {
                let buf = device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: u64::from(row * N * CASCADES as u32),
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_texture_to_buffer(
                    TexelCopyTextureInfo { texture: tex, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                    TexelCopyBufferInfo { buffer: &buf, layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(N) } },
                    Extent3d { width: N, height: N, depth_or_array_layers: CASCADES as u32 },
                );
                buf
            };
            let bufs = [read(&field.displacement[field.current()]), read(&field.derivatives), read(&field.curvature[field.current()])];
            queue.submit(Some(encoder.finish()));
            for buf in &bufs {
                buf.slice(..).map_async(MapMode::Read, |_| {});
            }
            let _ = device.poll(PollType::Wait { submission_index: None, timeout: None });
            let texels: Vec<Vec<u16>> = bufs.iter().map(|b| bytemuck::cast_slice(&b.slice(..).get_mapped_range().unwrap()).to_vec()).collect();
            let mut out = Vec::with_capacity(CASCADES * (N * N) as usize);
            for c in 0..CASCADES as u32 {
                for z in 0..N {
                    for x in 0..N {
                        let i = (c * N * row / 2 + z * row / 2 + x * 4) as usize;
                        let mut f = [0.0f32; FIELDS + 1];
                        for (j, v) in f.iter_mut().enumerate() {
                            *v = half_to_f32(texels[j / 4][i + j % 4]);
                        }
                        out.push(f);
                    }
                }
            }
            return Some(out);
        }
        None
    }

    #[test]
    fn the_transformed_surface_is_the_sum_of_its_waves() {
        let p = WaveParams::default();
        let t = 37.25;
        let Some(gpu) = gpu_surface(p, &[t]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let waves = p.waves();
        for c in 0..CASCADES {
            let mut worst = [0.0f64; FIELDS];
            let mut scale = [0.0f64; FIELDS];
            let mut imaginary = 0.0f64;
            for (x, z) in [(0, 0), (5, 9), (31, 2), (47, 63), (63, 17), (12, 40), (33, 33)] {
                let want = direct(&p, &waves, c, x, z, t);
                let got = gpu[(c as u32 * N * N + z * N + x) as usize];
                for f in 0..FIELDS {
                    worst[f] = worst[f].max((f64::from(got[f]) - want[f].0).abs());
                    scale[f] = scale[f].max(want[f].0.abs());
                    imaginary = imaginary.max(want[f].1.abs() / want[f].0.abs().max(1e-3));
                }
            }
            eprintln!(
                "cascade {c}: worst errors {:?} against magnitudes {:?}",
                worst.map(|v| format!("{v:.1e}")),
                scale.map(|v| format!("{v:.1e}"))
            );
            assert!(imaginary < 1e-5, "cascade {c}: the direct sum has an imaginary part ({imaginary})");
            for f in 0..FIELDS {
                assert!(scale[f] > 1e-5, "cascade {c} field {f} is empty: {scale:?}");

                assert!(worst[f] <= 2e-3 * scale[f], "cascade {c} field {f}: error {} at magnitude {}", worst[f], scale[f]);
            }
        }
    }

    #[test]
    fn the_push_gathers_the_surface_at_the_crests() {
        let Some(gpu) = gpu_surface(WaveParams::default(), &[11.0]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        for c in 0..CASCADES {
            let cells = &gpu[c * (N * N) as usize..(c + 1) * (N * N) as usize];
            let jacobian = |f: &[f32; FIELDS + 1]| f64::from((1.0 + f[6]) * (1.0 + f[7]) - f[3] * f[3]);
            let n = cells.len() as f64;
            let (mh, mj) = (cells.iter().map(|f| f64::from(f[1])).sum::<f64>() / n, cells.iter().map(jacobian).sum::<f64>() / n);
            let (mut shj, mut shh, mut sjj) = (0.0, 0.0, 0.0);
            for f in cells {
                let (dh, dj) = (f64::from(f[1]) - mh, jacobian(f) - mj);
                shj += dh * dj;
                shh += dh * dh;
                sjj += dj * dj;
            }
            let corr = shj / (shh * sjj).sqrt();
            eprintln!("cascade {c}: correlation of height with the Jacobian {corr:.3}");
            assert!(corr < -0.5, "cascade {c}: the push spreads the crests (correlation {corr})");
        }
    }

    #[test]
    fn whitecaps_come_from_folds_and_fade_after_them() {
        let gale = WaveParams { wind_speed: 16.0, fetch: 20_000.0, tile: [90.0, 20.0, 4.4], choppiness: 1.0, ..WaveParams::default() };
        let gains = gale.foam_gains();
        let Some(gpu) = gpu_surface(gale, &[20.0, 20.014, 20.028]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let cells = &gpu[..(N * N) as usize];
        let foamy = cells.iter().filter(|f| f[FIELDS] > 0.5).count();
        let mut below = 0;
        for f in cells {
            let j = 1.0 + ((1.0 + f[6]) * (1.0 + f[7]) - f[3] * f[3] - 1.0) * gains[0];
            let t = ((j - FOAM_JACOBIAN[0]) / (FOAM_JACOBIAN[1] - FOAM_JACOBIAN[0])).clamp(0.0, 1.0);
            let made = t * t * (3.0 - 2.0 * t);
            if f[FIELDS] + 2e-3 < made {
                below += 1;
            }
        }
        eprintln!("gale: {foamy} of {} texels foamy", cells.len());
        assert!(foamy > 0, "a gale makes no whitecaps");
        assert!(foamy < cells.len() / 2, "half the sea is foam ({foamy})");
        assert_eq!(below, 0, "{below} texels carry less foam than their fold makes");
    }

    #[test]
    fn the_spectrum_makes_the_sea_it_describes() {
        let p = WaveParams::default();

        let hs = p.significant_height();
        let u2 = p.wind_speed * p.wind_speed;
        let law = 0.0016 * (G * p.fetch / u2).sqrt() * u2 / G;
        eprintln!("Hs {hs:.3} m, fetch law {law:.3} m; peak wavelength {:.2} m", 2.0 * PI * G / p.peak_omega().powi(2));
        assert!(hs > 0.7 * law && hs < 1.4 * law, "Hs {hs} against the fetch law's {law}");

        let sampled: f64 = p.waves().iter().map(|w| f64::from(w[0][0] * w[0][0] + w[0][1] * w[0][1]) * 2.0).sum();
        let ratio = sampled / f64::from(p.height_variance());
        eprintln!("sampled / spectrum variance {ratio:.3}");
        assert!(ratio > 0.65 && ratio < 1.5, "sampled variance / spectrum variance = {ratio}");
    }

    #[test]
    fn the_sums_from_one_pass_are_the_sums_each_pass_made() {
        for p in [
            WaveParams::default(),
            WaveParams { wind_speed: 2.0, fetch: 20.0, depth: 0.7, ..WaveParams::default() },
            WaveParams { wind_speed: 16.0, fetch: 20_000.0, tile: [90.0, 20.0, 4.4], depth: 30.0, wind_dir: -2.1, ..WaveParams::default() },
            WaveParams { wind_speed: 0.0, fetch: 1.0, ..WaveParams::default() },
        ] {
            let sums = WaveSums::of(&p);
            assert!(sums.height_variance > 0.0 && sums.slope_variance > 0.0, "{p:?}");
            assert_eq!(sums.height_variance, p.sum_over_waves(|_, _, _, e| f64::from(e)));
            assert_eq!(sums.slope_variance, p.sum_over_waves(|_, kx, kz, e| f64::from((kx * kx + kz * kz) * e)));
            for c in 0..CASCADES {
                assert_eq!(sums.cascade_slope_variance[c], p.sum_over_waves(|cc, kx, kz, e| if cc == c { f64::from((kx * kx + kz * kz) * e) } else { 0.0 }));
            }
            for level in 0..MIPS as usize {
                for c in 0..CASCADES {
                    let texels = (N >> level).max(1) as f32;
                    let nyquist = if level + 1 == MIPS as usize { 0.0 } else { PI * texels / p.tile[c] };
                    let was = p.sum_over_waves(|cc, kx, kz, e| {
                        if cc == c && (kx.abs() > nyquist || kz.abs() > nyquist) { f64::from((kx * kx + kz * kz) * e) } else { 0.0 }
                    });
                    assert_eq!(sums.lost[level][c], was, "level {level} cascade {c}");
                }
                assert_eq!(sums.lost[level][3], 0.0);
            }

            assert_eq!(p.sums(), sums);
            assert_eq!((p.height_variance(), p.slope_variance_beyond(0.0), p.lost_slope_variance()), (sums.height_variance, sums.slope_variance, sums.lost));
        }
    }

    #[test]
    fn a_levels_bodies_under_one_wind_pass_over_the_spectrum_once() {

        let wind = WaveParams { wind_speed: 6.125, fetch: 437.5, depth: 2.25, ..WaveParams::default() };
        let passes = || SPECTRUM_PASSES.with(std::cell::Cell::get);
        let before = passes();
        let load = |p: &WaveParams| (p.lost_slope_variance(), p.unresolved_slope_variance(), p.foam_gains(), p.significant_height());
        let first = load(&wind);
        for body in 1..65u64 {
            let p = WaveParams { seed: wind.seed ^ body.wrapping_mul(0x9e37_79b9_7f4a_7c15), choppiness: 0.1 + body as f32 * 0.01, ..wind };
            assert_eq!(load(&p), first, "the sums do not follow the seed or the chop");
        }
        assert_eq!(passes() - before, 1, "65 bodies under one wind");
        load(&WaveParams { wind_speed: 6.25, ..wind });
        load(&WaveParams { depth: 2.5, ..wind });
        assert_eq!(passes() - before, 3);
    }

    #[test]
    fn roughness_accounts_for_every_wave() {
        let p = WaveParams::default();
        let lost = p.lost_slope_variance();
        let whole = |c: usize| p.cascade_slope_variance(c);
        for c in 0..CASCADES {
            assert!(lost[0][c] < 1e-3 * whole(c), "cascade {c} loses {} at level 0", lost[0][c]);
            for l in 1..MIPS as usize {
                assert!(lost[l][c] >= lost[l - 1][c], "cascade {c}: level {l} loses less than level {}", l - 1);
            }
            let last = lost[MIPS as usize - 1][c];
            assert!((last - whole(c)).abs() <= 1e-4 * whole(c), "cascade {c}: the last level loses {last} of {}", whole(c));
        }
        let held: f32 = (0..CASCADES).map(whole).sum();
        let measured = 0.003 + 0.005_12 * p.wind_speed;
        eprintln!(
            "slope variance: held {held:.4} ({:?} by cascade), unresolved {:.4}, Cox-Munk {measured:.4}",
            (0..CASCADES).map(|c| format!("{:.4}", whole(c))).collect::<Vec<_>>(),
            p.unresolved_slope_variance()
        );
        assert!((held + p.unresolved_slope_variance() - measured.max(held)).abs() < 1e-5);
    }

    #[test]
    fn every_wave_repeats_after_the_loop() {
        let p = WaveParams::default();
        for w in p.waves() {
            let turns = f64::from(w[1][2]) * f64::from(p.loop_seconds) / (2.0 * std::f64::consts::PI);
            assert!((turns - turns.round()).abs() < 1e-3, "omega {} makes {turns} turns a loop", w[1][2]);
        }
    }

    #[test]
    fn the_foam_texture_tiles() {
        let t = foam_texture();
        let s = FOAM_SIZE as usize;
        let at = |x: usize, y: usize| f32::from(t[y * s + x]);
        let step = |a: &dyn Fn(usize) -> f32, b: &dyn Fn(usize) -> f32| (0..s).map(|i| (a(i) - b(i)).abs()).sum::<f32>() / s as f32;
        let seam = step(&|y| at(s - 1, y), &|y| at(0, y)).max(step(&|x| at(x, s - 1), &|x| at(x, 0)));
        let inside = step(&|y| at(s / 2, y), &|y| at(s / 2 + 1, y)).max(step(&|x| at(x, s / 2), &|x| at(x, s / 2 + 1)));
        let mean = t.iter().map(|&v| f32::from(v)).sum::<f32>() / t.len() as f32;
        eprintln!("foam texture: mean {mean:.0}, seam step {seam:.1}, inside step {inside:.1}");
        assert!(seam < 2.0 * inside + 4.0, "the foam texture has a seam ({seam} against {inside} inside)");
        assert!(mean > 30.0 && mean < 200.0, "the foam texture is all one shade (mean {mean})");
    }
}
