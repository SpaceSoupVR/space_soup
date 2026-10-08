//! THE WATER'S WAVES: an FFT wave field (Tessendorf, *Simulating Ocean Water*,
//! SIGGRAPH course notes 2001), computed once a frame and read by the water.
//!
//! The sea state is a JONSWAP spectrum (Hasselmann et al. 1973) for a wind of
//! `wind_speed` blowing over `fetch` metres of open water, spread about the
//! wind by Hasselmann's cos^2s law (Hasselmann, Dunckel and Ewing 1980), with
//! finite-depth dispersion -- the recipe of Horvath, *Empirical Directional
//! Wave Spectra for Computer Graphics* (2015). Three cascades of 64^2 cover
//! it: the largest tile the long waves, the smallest the ripples, each
//! handing over to the next at a wavenumber both resolve, so no wave is
//! counted twice. The tiles' sides are not multiples of each other, so their
//! repeats never line up.
//!
//! Each frame a compute pass advances every wave's phase; six inverse FFTs a
//! cascade turn twelve spectral fields into the surface -- the sideways push
//! that sharpens the crests (Tessendorf's lambda), the height, its slopes and
//! curvatures, and the push's own derivatives, from which the water knows
//! where the surface folds over (the Jacobian below 1: whitecaps, kept and
//! left to fade, so foam trails behind the crest that made it) -- and the
//! results are mipmapped, so a distant patch reads its waves filtered rather
//! than aliased. What the filtering takes away is not lost:
//! [`WaveParams::lost_slope_variance`] is the slope variance of every wave a
//! mip level averages away, read straight off the spectrum, for the water's
//! roughness (Bruneton, Neyret and Holzschuch, *Real-time Realistic Ocean
//! Lighting using Seamless Transitions from Geometry to BRDF*, 2010), and
//! [`WaveParams::unresolved_slope_variance`] is the ripples too short for any
//! cascade, from Cox and Munk's measured sea-surface slopes (1954).
//!
//! Frequencies are rounded to multiples of 2 pi / `loop_seconds`, so the waves
//! repeat exactly and time is wrapped before it reaches `f32`: a phase taken
//! from hours of `f32` seconds would jitter.
//!
//! The displacement and the curvature are kept for this frame and the last:
//! the water's motion for SpaceWarp, and the foam each frame fades from.

use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use wgpu::*;

/// Cells along each side of a cascade's tile.
pub const N: u32 = 64;
/// Tiles, largest first.
pub const CASCADES: usize = 3;
/// Mip levels of each output: 64 down to 1.
pub const MIPS: u32 = 7;
const G: f32 = 9.81;
/// Spectral fields a cascade transforms: six complex pairs of real fields.
const PAIRS: u32 = 6;
/// A cascade hands over to the next at this many of the next one's
/// fundamental wavenumber.
const SPLIT_CELLS: f32 = 4.0;
const GROUP: u32 = 8;
/// Seconds a whitecap takes to fade to a third.
const FOAM_SECONDS: f64 = 1.6;
/// Where foam starts and where it is solid, as the surface's Jacobian falls:
/// 1 is a calm surface, 0 a fold.
const FOAM_JACOBIAN: [f32; 2] = [0.72, 0.3];
/// The foam texture's side, texels.
pub const FOAM_SIZE: u32 = 256;

/// The sea state and how it is sampled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveParams {
    /// Wind speed ten metres up, m/s.
    pub wind_speed: f32,
    /// The way the wind blows, radians from +x toward +z.
    pub wind_dir: f32,
    /// How far the wind has blown over open water, metres: with the speed,
    /// how grown the sea is.
    pub fetch: f32,
    /// Water depth for the dispersion, metres.
    pub depth: f32,
    /// Tessendorf's lambda: 0 rolls, ~1 sharpens the crests.
    pub choppiness: f32,
    /// Each cascade's tile side, metres, largest first.
    pub tile: [f32; CASCADES],
    /// The waves repeat exactly after this many seconds.
    pub loop_seconds: f32,
    pub seed: u64,
}

impl Default for WaveParams {
    /// A lake in a fresh breeze: 7 m/s over 800 m, peak waves ~2 m long.
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
    /// The band of wavenumbers cascade `c` keeps: from where the larger tile
    /// hands over to it to where it hands over to the smaller.
    fn band(&self, c: usize) -> (f32, f32) {
        let split = |c: usize| SPLIT_CELLS * 2.0 * PI / self.tile[c + 1];
        let lo = if c == 0 { 0.0 } else { split(c - 1) };
        let hi = if c + 1 == CASCADES { f32::INFINITY } else { split(c) };
        (lo, hi)
    }

    /// The wave vector cascade `c` holds at cell (`xi`, `zi`), the zero
    /// wavenumber in the middle of the grid -- or `None` if the cascade
    /// leaves it out: outside its band, or on the first row or column, the
    /// Nyquist wavenumber, which is its own mirror and so cannot carry a
    /// sideways push that stays real.
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

    /// Angular frequency of a wave of wavenumber `k`, before the loop's rounding.
    pub fn omega(&self, k: f32) -> f32 {
        (G * k * (k * self.depth).tanh()).sqrt()
    }

    /// `omega` rounded to the loop: every wave repeats after `loop_seconds`.
    pub fn omega_looped(&self, k: f32) -> f32 {
        let w0 = 2.0 * PI / self.loop_seconds;
        ((self.omega(k) / w0).round() * w0).max(w0)
    }

    /// JONSWAP's peak angular frequency.
    pub fn peak_omega(&self) -> f32 {
        22.0 * (G * G / (self.wind_speed.max(0.5) * self.fetch.max(1.0))).powf(1.0 / 3.0)
    }

    /// The JONSWAP spectrum at angular frequency `omega`, m^2 s.
    pub fn jonswap(&self, omega: f32) -> f32 {
        let u = self.wind_speed.max(0.5);
        let alpha = 0.076 * (u * u / (self.fetch.max(1.0) * G)).powf(0.22);
        let wp = self.peak_omega();
        let sigma = if omega <= wp { 0.07 } else { 0.09 };
        let r = (-(omega - wp).powi(2) / (2.0 * sigma * sigma * wp * wp)).exp();
        alpha * G * G / omega.powi(5) * (-1.25 * (wp / omega).powi(4)).exp() * 3.3f32.powf(r)
    }

    /// Hasselmann's spreading exponent at `omega`: narrow at the peak, wider
    /// away from it.
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

    /// The spectrum over the plane of wave vectors at (`kx`, `kz`), m^4: a
    /// wave's share of the height variance per unit of wave-vector area.
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

    /// Every cascade's waves as the GPU reads them: for each wave vector,
    /// `[h0.re, h0.im, conj(h0(-k)).re, conj(h0(-k)).im]` and
    /// `[kx, kz, omega, 0]` -- cascade by cascade, row (z) by row, the zero
    /// wavenumber in the middle. The frequencies are worked out here, once,
    /// so the GPU and the CPU never disagree about which way one rounds.
    pub fn waves(&self) -> Vec<[[f32; 4]; 2]> {
        let n = N as usize;
        let mut rng = SplitMix(self.seed);
        let mut out = Vec::with_capacity(CASCADES * n * n);
        for c in 0..CASCADES {
            let dk = 2.0 * PI / self.tile[c];
            // h0 for every wave vector first, so h0(-k) can be read back.
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
                    // -k: the cell mirrored about the middle. Cells without a
                    // wave mirror cells without one.
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

    /// Sum `f(c, kx, kz, spectrum * dk^2)` over every wave vector the field
    /// holds.
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

    /// The height variance the field holds, m^2: a sixteenth of the
    /// significant wave height squared.
    pub fn height_variance(&self) -> f32 {
        self.sum_over_waves(|_, _, _, e| e as f64)
    }

    /// The significant wave height the field holds, m: the mean height of the
    /// highest third of the waves, 4 sigma.
    pub fn significant_height(&self) -> f32 {
        4.0 * self.height_variance().sqrt()
    }

    /// The slope variance (both directions together) of every wave the field
    /// holds shorter than wavenumber `k_cut` allows.
    pub fn slope_variance_beyond(&self, k_cut: f32) -> f32 {
        self.sum_over_waves(|_, kx, kz, e| {
            let k2 = kx * kx + kz * kz;
            if k2 >= k_cut * k_cut { (k2 * e) as f64 } else { 0.0 }
        })
    }

    /// The slope variance of cascade `c`'s waves, all of them.
    pub fn cascade_slope_variance(&self, c: usize) -> f32 {
        self.sum_over_waves(|cc, kx, kz, e| if cc == c { ((kx * kx + kz * kz) * e) as f64 } else { 0.0 })
    }

    /// PER MIP LEVEL, the slope variance of each cascade's waves that level
    /// averages away -- those past its Nyquist wavenumber, pi over its texel,
    /// along either axis: a level is a square grid, and holds a diagonal wave
    /// longer than its axes allow. Level 0 loses nothing; the last level, a
    /// single texel, loses the whole cascade. `[level][cascade]`, the fourth
    /// lane unused.
    pub fn lost_slope_variance(&self) -> [[f32; 4]; MIPS as usize] {
        let mut out = [[0.0f32; 4]; MIPS as usize];
        for (level, row) in out.iter_mut().enumerate() {
            for (c, lane) in row.iter_mut().take(CASCADES).enumerate() {
                let texels = (N >> level).max(1) as f32;
                let nyquist = if level + 1 == MIPS as usize { 0.0 } else { PI * texels / self.tile[c] };
                *lane = self.sum_over_waves(|cc, kx, kz, e| {
                    if cc == c && (kx.abs() > nyquist || kz.abs() > nyquist) { ((kx * kx + kz * kz) * e) as f64 } else { 0.0 }
                });
            }
        }
        out
    }

    /// HOW HARD EACH CASCADE'S FOLDS ARE READ for foam. The surface folds
    /// where the Jacobian of its WHOLE sideways push falls, and each cascade
    /// holds only part of that push: alone, even a gale's largest waves
    /// rarely fold. The Jacobian's spread about 1 is lambda times the slope's
    /// (both come to k h), so a cascade's departure from 1 scaled by the
    /// whole surface's spread over its own folds as often as the whole
    /// surface does -- along that cascade's crests. The ripples make none:
    /// foam a hand's width across, repeating every metre, reads as a pattern.
    pub fn foam_gains(&self) -> [f32; CASCADES] {
        let total: f32 = (0..CASCADES).map(|c| self.cascade_slope_variance(c)).sum();
        let mut out = [0.0; CASCADES];
        for (c, g) in out.iter_mut().enumerate().take(CASCADES - 1) {
            *g = (total / self.cascade_slope_variance(c).max(1e-12)).sqrt();
        }
        out
    }

    /// The slope variance of the ripples no cascade holds: what Cox and Munk
    /// measured on a real sea under this wind (a clean surface, 0.003 +
    /// 0.00512 U), less what the cascades already make.
    pub fn unresolved_slope_variance(&self) -> f32 {
        let measured = 0.003 + 0.005_12 * self.wind_speed.max(0.0);
        (measured - self.slope_variance_beyond(0.0)).max(0.0)
    }
}

/// The integral of |cos(theta / 2)|^(2s) over a turn, so the spreading
/// integrates to one: by Simpson's rule, which needs no Gamma function.
fn spread_norm(s: f32) -> f32 {
    let m = 512;
    let h = 2.0 * PI / m as f32;
    let f = |i: usize| (0.5 * (-PI + i as f32 * h)).cos().abs().powf(2.0 * s);
    let mut sum = f(0) + f(m);
    for i in 1..m {
        sum += if i % 2 == 1 { 4.0 } else { 2.0 } * f(i);
    }
    sum * h / 3.0
}

/// A small seeded generator, so the same seed always makes the same sea.
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
    /// Two independent standard normals (Box-Muller).
    fn gaussian_pair(&mut self) -> (f32, f32) {
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        (r * (2.0 * PI * u2).cos(), r * (2.0 * PI * u2).sin())
    }
}

/// FOAM'S TEXTURE: bubbles, tileable, `FOAM_SIZE` square, one byte a texel.
///
/// Two scales of Worley cells (Worley, *A Cellular Texture Basis Function*,
/// 1996): the film between neighbouring bubbles is where the distance to the
/// nearest cell centre and to the next nearest are equal, so `F2 - F1` small
/// is a bubble's wall. The water thresholds it by how much foam a texel
/// carries: a little shows only the brightest walls -- lace -- and a lot
/// fills the cells too.
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
                    // The point's copy in the cell next to this one, across the wrap.
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
            // Walls bright, thinning with distance from them; the bubble's own
            // film a little bright toward its rim.
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
        r#"
// time: x the time in seconds, wrapped to the loop; y lambda; z the share of
// last frame's foam this frame keeps. foam: x, y the Jacobian where foam
// starts and where it is solid. gain: each cascade's folds' weight -- see
// `WaveParams::foam_gains`.
struct Waves {{ time: vec4<f32>, foam: vec4<f32>, gain: vec4<f32> }}
// amp: h0(k), conj(h0(-k)). kw: kx, kz, omega.
struct Wave {{ amp: vec4<f32>, kw: vec4<f32> }}
@group(0) @binding(0) var<uniform> waves: Waves;
@group(0) @binding(1) var<storage, read> spectrum: array<Wave>;
// Per cascade, pair and cell: the spectral fields, transformed in place.
@group(0) @binding(2) var<storage, read_write> spec: array<vec2<f32>>;
@group(0) @binding(3) var displacement: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var derivatives: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var curvature: texture_storage_2d_array<rgba16float, write>;
// Last frame's curvature, for the foam it carried.
@group(0) @binding(6) var previous: texture_2d_array<f32>;

const N: u32 = {n}u;
const PI: f32 = 3.14159265358979;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}}
// i a: a quarter turn.
fn times_i(a: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(-a.y, a.x);
}}
fn spec_index(c: u32, pair: u32, row: u32, col: u32) -> u32 {{
    return ((c * {pairs}u + pair) * N + row) * N + col;
}}

// Every wave at this time, and the twelve fields it makes, as six complex
// pairs of real fields, a + i b: (Dx, h), (Dz, dDx/dz), (dh/dx, dh/dz),
// (dDx/dx, dDz/dz), (d2h/dx2, d2h/dz2), (d2h/dxdz, 0).
@compute @workgroup_size({GROUP}, {GROUP}, 1)
fn evolve(@builtin(global_invocation_id) id: vec3<u32>) {{
    let c = id.z;
    if (id.x >= N || id.y >= N || c >= {cascades}u) {{
        return;
    }}
    let w = spectrum[(c * N + id.y) * N + id.x];
    let phase = w.kw.z * waves.time.x;
    let e = vec2<f32>(cos(phase), sin(phase));
    // h0(k) e^-iwt runs along k; conj(h0(-k)) e^iwt keeps the surface real.
    let h = cmul(w.amp.xy, vec2<f32>(e.x, -e.y)) + cmul(w.amp.zw, e);
    let kv = w.kw.xy;
    let k = length(kv);
    var u = vec2<f32>(0.0);
    if (k > 1e-6) {{
        u = kv / k;
    }}
    let lambda = waves.time.y;
    // The sideways push D = i (k / |k|) h lambda: Gerstner's sign, so a
    // positive lambda draws the surface in toward the crests. Each
    // derivative is one more factor of i k.
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

// THE INVERSE FFT of one row or column of one pair, in workgroup memory:
// Stockham's autosort radix-2 (Govindaraju et al. 2008), {stages} stages, no
// bit reversal, each thread one butterfly a stage. TWO lines a workgroup, so
// one fills a whole 64-wide wave of the Quest's GPU instead of half of one.
// Returns which of each line's two buffers in `buf` holds the result.
var<workgroup> buf: array<array<vec2<f32>, {n}>, 4>;
// e^(i pi j / {half}): every turn a butterfly takes, worked out once a
// workgroup rather than once a butterfly a stage.
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

// The turns, one a thread; the barrier after the caller's loads covers them.
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

// The transformed pairs into the three surfaces. The spectrum's zero sits in
// the middle of its grid, which multiplies every cell by (-1)^(x+z). The
// foam: last frame's, faded, or what a fold makes now, whichever is more.
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
"#
    )
}

fn mip_shader() -> String {
    format!(
        r#"
@group(0) @binding(0) var src_a: texture_2d_array<f32>;
@group(0) @binding(1) var src_b: texture_2d_array<f32>;
@group(0) @binding(2) var src_c: texture_2d_array<f32>;
@group(0) @binding(3) var dst_a: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var dst_b: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var dst_c: texture_storage_2d_array<rgba16float, write>;

// One level down: each texel the mean of the four above it.
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
"#
    )
}

/// The per-frame uniform.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveUniform {
    /// x time (s, wrapped to the loop), y choppiness, z the foam kept.
    time: [f32; 4],
    /// x, y the Jacobian where foam starts and where it is solid.
    foam: [f32; 4],
    /// Each cascade's folds' weight.
    gain: [f32; 4],
}

/// The wave field on the GPU: its spectrum, the passes that turn it into a
/// surface each frame, and the surface.
pub struct WaveField {
    pub params: WaveParams,
    /// [`WaveParams::significant_height`], worked out once: it sums the
    /// spectrum over every wave vector of every cascade, and read per eye per
    /// frame (`underwater::wave_reach`) it cost the Quest's render thread
    /// ~130 ms a frame (2026-10-08).
    pub significant_height: f32,
    uniform: Buffer,
    /// Displacement -- x, height, z, and d(Dx)/dz -- written in turns, so the
    /// one the last update did not write is last frame's: the water's motion
    /// for SpaceWarp. A 3-layer array, a cascade a layer, mipmapped. Read by
    /// the water's vertex stage.
    pub displacement: [Texture; 2],
    pub displacement_views: [TextureView; 2],
    /// Slopes and the push's own derivatives: dh/dx, dh/dz, d(Dx)/dx,
    /// d(Dz)/dz. Read by the water's fragment stage.
    pub derivatives: Texture,
    pub derivatives_view: TextureView,
    /// Curvatures and foam: d2h/dx2, d2h/dz2, d2h/dxdz, and the whitecaps
    /// still on the water, 0..1. In turns like the displacement, as each
    /// frame's foam is last frame's faded.
    pub curvature: [Texture; 2],
    pub curvature_views: [TextureView; 2],
    /// Foam's bubbles, `FOAM_SIZE` square, mipmapped, R8. See [`foam_texture`].
    pub foam: Texture,
    pub foam_view: TextureView,
    /// Each cascade's folds' weight, worked out once. See
    /// [`WaveParams::foam_gains`].
    foam_gains: [f32; CASCADES],
    /// Which of the pairs the last update wrote.
    current: AtomicUsize,
    /// Whether an update has run: the first writes both sets, so the first
    /// frame's motion is none rather than everything.
    primed: AtomicBool,
    /// The last update's time, as f64 bits, for the foam's fading.
    last_seconds: AtomicU64,
    evolve: ComputePipeline,
    rows: ComputePipeline,
    columns: ComputePipeline,
    assemble: ComputePipeline,
    mip: ComputePipeline,
    /// Per set written: the passes' group, and the mip chain's groups.
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
            size: CASCADES as u64 * PAIRS as u64 * (N * N) as u64 * 8,
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

        // Foam's bubbles, with a box-filtered chain so distant foam is grey
        // rather than sparkling.
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
                        let sum: u32 = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)].iter().map(|&(a, b)| data[b * s + a] as u32).sum();
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

    /// Have the next update write both sets, as the first does: for a field
    /// left unupdated for a while -- out of sight -- whose other set holds a
    /// surface long gone, and would tell SpaceWarp the water leapt.
    pub fn reprime(&self) {
        self.primed.store(false, Ordering::Relaxed);
    }

    /// Which of the pairs holds this frame's surface; the other holds last
    /// frame's.
    pub fn current(&self) -> usize {
        self.current.load(Ordering::Relaxed)
    }

    /// The surface at `seconds`: records every pass into `encoder`. Time is
    /// taken as f64 and wrapped to the loop here, so hours of play stay exact.
    pub fn update(&self, queue: &Queue, encoder: &mut CommandEncoder, seconds: f64) {
        let p = &self.params;
        let wrapped = seconds.rem_euclid(p.loop_seconds as f64) as f32;
        let first = !self.primed.swap(true, Ordering::Relaxed);
        let last = f64::from_bits(self.last_seconds.swap(seconds.to_bits(), Ordering::Relaxed));
        // A frame's fade from how long it was; none across a jump in time,
        // which is a restart or a scene change rather than a long frame.
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
            // Two lines a workgroup: see the shader.
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

    /// A complex number, for writing the sums out from first principles.
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

    /// The fields the GPU writes, in the order `direct` and `gpu_surface`
    /// give them: displacement, derivatives, curvature's first three.
    const FIELDS: usize = 11;

    /// The surface's fields at cell (`x`, `z`) of cascade `c`, summed
    /// directly over every wave -- no FFT, no simplified coefficients: the
    /// height is sum h(k) e^{ik.x}; the push is i (k/|k|) h lambda; every
    /// derivative is one more factor of i k. As `[Dx, h, Dz, dDx/dz, dh/dx,
    /// dh/dz, dDx/dx, dDz/dz, d2h/dx2, d2h/dz2, d2h/dxdz]`, each with the
    /// imaginary part left over -- which must be nothing, or the field is not
    /// real.
    fn direct(p: &WaveParams, waves: &[[[f32; 4]; 2]], c: usize, x: u32, z: u32, t: f64) -> [C; FIELDS] {
        let n = N as usize;
        let cell = p.tile[c] as f64 / n as f64;
        let (px, pz) = (x as f64 * cell, z as f64 * cell);
        let lam = p.choppiness as f64;
        let mut out = [C(0.0, 0.0); FIELDS];
        for w in &waves[c * n * n..(c + 1) * n * n] {
            let [a, kw] = *w;
            let (kx, kz, omega) = (kw[0] as f64, kw[1] as f64, kw[2] as f64);
            let k = (kx * kx + kz * kz).sqrt();
            if k == 0.0 {
                continue;
            }
            let turn = |phase: f64| C(phase.cos(), phase.sin());
            let h = C(a[0] as f64, a[1] as f64) * turn(-omega * t) + C(a[2] as f64, a[3] as f64) * turn(omega * t);
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
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as f32;
        s * match e {
            0 => m * 2f32.powi(-24),
            31 => f32::INFINITY,
            _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
        }
    }

    /// The GPU's surface after updates at each of `times`, level 0: per
    /// cascade and cell, the eleven fields in the order `direct` gives them,
    /// then the foam. `None` without a GPU.
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
                    size: (row * N * CASCADES as u32) as u64,
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

    /// THE GPU'S SURFACE IS THE SUM OF ITS WAVES: every field at a spread of
    /// cells in every cascade, after the FFTs, against the direct sum over
    /// all 4,096 wave vectors, at a time well into the loop -- and the direct
    /// sum is real, so the spectrum was built mirror-symmetric.
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
                    worst[f] = worst[f].max((got[f] as f64 - want[f].0).abs());
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
                // f16 keeps 11 bits; 1/500 of the field's size is ~4 ulps.
                assert!(worst[f] <= 2e-3 * scale[f], "cascade {c} field {f}: error {} at magnitude {}", worst[f], scale[f]);
            }
        }
    }

    /// THE CRESTS ARE SHARPENED, NOT THE TROUGHS: the surface is drawn in
    /// toward its high points -- the Jacobian of the sideways push falls
    /// where the water is high -- so a positive lambda makes peaked crests
    /// and rounded troughs, as real waves have.
    #[test]
    fn the_push_gathers_the_surface_at_the_crests() {
        let Some(gpu) = gpu_surface(WaveParams::default(), &[11.0]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        for c in 0..CASCADES {
            let cells = &gpu[c * (N * N) as usize..(c + 1) * (N * N) as usize];
            let jacobian = |f: &[f32; FIELDS + 1]| ((1.0 + f[6]) * (1.0 + f[7]) - f[3] * f[3]) as f64;
            let n = cells.len() as f64;
            let (mh, mj) = (cells.iter().map(|f| f[1] as f64).sum::<f64>() / n, cells.iter().map(jacobian).sum::<f64>() / n);
            let (mut shj, mut shh, mut sjj) = (0.0, 0.0, 0.0);
            for f in cells {
                let (dh, dj) = (f[1] as f64 - mh, jacobian(f) - mj);
                shj += dh * dj;
                shh += dh * dh;
                sjj += dj * dj;
            }
            let corr = shj / (shh * sjj).sqrt();
            eprintln!("cascade {c}: correlation of height with the Jacobian {corr:.3}");
            assert!(corr < -0.5, "cascade {c}: the push spreads the crests (correlation {corr})");
        }
    }

    /// FOAM IS MADE WHERE THE SURFACE FOLDS, AND OUTLASTS THE FOLD: in a gale
    /// the largest cascade folds somewhere, and every texel's foam is at
    /// least what its own Jacobian makes now -- a fold's foam is never
    /// thrown away -- while a calm frame's foam is only what was kept.
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

    /// THE SEA IS THE ONE ASKED FOR: the field's height is the spectrum's,
    /// and a 7 m/s breeze over 800 m makes waves as high as JONSWAP's fetch
    /// law says.
    #[test]
    fn the_spectrum_makes_the_sea_it_describes() {
        let p = WaveParams::default();
        // Significant height, 4 sigma, against the fetch law: g Hs / U^2 = 0.0016 (g F / U^2)^0.5.
        let hs = p.significant_height();
        let u2 = p.wind_speed * p.wind_speed;
        let law = 0.0016 * (G * p.fetch / u2).sqrt() * u2 / G;
        eprintln!("Hs {hs:.3} m, fetch law {law:.3} m; peak wavelength {:.2} m", 2.0 * PI * G / p.peak_omega().powi(2));
        assert!(hs > 0.7 * law && hs < 1.4 * law, "Hs {hs} against the fetch law's {law}");
        // The random amplitudes carry the spectrum's variance: the sum of
        // |h0(k)|^2 + |h0(-k)|^2, to the sampling error of a few hundred waves.
        let sampled: f64 = p.waves().iter().map(|w| (w[0][0] * w[0][0] + w[0][1] * w[0][1]) as f64 * 2.0).sum();
        let ratio = sampled / p.height_variance() as f64;
        eprintln!("sampled / spectrum variance {ratio:.3}");
        assert!(ratio > 0.65 && ratio < 1.5, "sampled variance / spectrum variance = {ratio}");
    }

    /// WHAT THE MIPS LOSE IS PUT BACK, AND NOTHING TWICE: level 0 loses
    /// nothing; each level loses more; the last loses the cascade whole; and
    /// the cascades' wholes plus the ripples none of them hold come to Cox
    /// and Munk's measured slope variance for the wind.
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

    /// THE WAVES LOOP: every frequency the field uses is a whole number of
    /// turns over the loop, so wrapping time loses nothing.
    #[test]
    fn every_wave_repeats_after_the_loop() {
        let p = WaveParams::default();
        for w in p.waves() {
            let turns = w[1][2] as f64 * p.loop_seconds as f64 / (2.0 * std::f64::consts::PI);
            assert!((turns - turns.round()).abs() < 1e-3, "omega {} makes {turns} turns a loop", w[1][2]);
        }
    }

    /// FOAM'S TEXTURE TILES: its left column continues its right, and its top
    /// row its bottom, as closely as any two neighbouring columns inside it.
    #[test]
    fn the_foam_texture_tiles() {
        let t = foam_texture();
        let s = FOAM_SIZE as usize;
        let at = |x: usize, y: usize| t[y * s + x] as f32;
        let step = |a: &dyn Fn(usize) -> f32, b: &dyn Fn(usize) -> f32| (0..s).map(|i| (a(i) - b(i)).abs()).sum::<f32>() / s as f32;
        let seam = step(&|y| at(s - 1, y), &|y| at(0, y)).max(step(&|x| at(x, s - 1), &|x| at(x, 0)));
        let inside = step(&|y| at(s / 2, y), &|y| at(s / 2 + 1, y)).max(step(&|x| at(x, s / 2), &|x| at(x, s / 2 + 1)));
        let mean = t.iter().map(|&v| v as f32).sum::<f32>() / t.len() as f32;
        eprintln!("foam texture: mean {mean:.0}, seam step {seam:.1}, inside step {inside:.1}");
        assert!(seam < 2.0 * inside + 4.0, "the foam texture has a seam ({seam} against {inside} inside)");
        assert!(mean > 30.0 && mean < 200.0, "the foam texture is all one shade (mean {mean})");
    }
}
