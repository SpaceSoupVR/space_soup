//! THE WAKE: the water a hand, a leg or a floating thing pushes aside as it
//! moves, as one continuous surface -- a bow wave ahead of it, a V of waves
//! spreading behind a fast one, ripples trailing -- not a chain of splashes.
//!
//! A small interactive HEIGHTFIELD centred on the player (`WAKE_N` texels of
//! `WAKE_DX`: 4.8 m square), stepped on the CPU each frame and read by the
//! water's shader (`water_pipeline`, binding 6) for its slope and the
//! caustics' curvature, on top of the waves and the splashes' rings: one
//! surface. PhysX has no water surface to give us this (its PBD fluids are
//! volumes of particles, not in our physx-rs fork's build, and far too costly
//! on a Quest); every engine that does this does it this way.
//!
//! THE EQUATION. The linear wave equation with Boussinesq's dispersion,
//! `(1 - alpha lap) h_tt = g D lap h`, `alpha = D^2 / 3`: long waves run at
//! `sqrt(g D)` (0.7 m/s for `WAKE_DEPTH` 5 cm, as a wave 30 cm long does in
//! deep water), shorter ones slower -- down to about 0.3 m/s at the grid's
//! finest, near the 0.23 m/s that is the slowest any ripple on real water
//! runs. So a thing moving slower than its waves makes only a swell round
//! it, and a faster one a V that narrows with its speed: the wake follows
//! speed with no rule of its own. The `(1 - alpha lap)` is solved by
//! `WAKE_JACOBI` Jacobi sweeps (spectral radius 0.57: converged to 6 %).
//! Ripples die by a film viscosity on the vertical speed, everything by a
//! slow drag, and a sponge round the edge drinks what leaves: nothing comes
//! back off the grid's edge (a pond's bank does not reflect either).
//!
//! THE SOURCES. A thing at the waterline displaces its submerged volume
//! `V`; moving across, it carries that volume with it -- `h_t += -(v . grad)
//! (V G)`, `G` a Gaussian its own width: water rises ahead, falls behind.
//! Moving down or up through the surface it pushes water out round it or
//! draws it in (a ring of zero net volume). Still, it does nothing.
//!
//! COST is fixed: `WAKE_N^2` (9,216) cells, two steps a frame at most, a
//! 74 KB upload -- whatever the level's size -- and nothing at all once no
//! source has touched the water for `WAKE_REST` and the last wave is under a
//! hundredth of a millimetre (`WakeField::step` returns false; the shader's
//! `water.wake.w` is 0).
//!
//! The grid moves with the player a whole texel at a time, its waves left
//! where they are in the world. It is shading, not geometry: the water's
//! vertices are half a metre apart and could not carry it, and SpaceWarp's
//! motion vectors are the surface's; the wake's crests are moved only by
//! being redrawn each app frame (as the rings' are).

use glam::{IVec2, Vec2, Vec3};

/// Texels a side, and metres a texel: 4.8 m round the player.
pub const WAKE_N: usize = 96;
pub const WAKE_DX: f32 = 0.05;
/// The depth whose shallow-water speed is the longest waves': `sqrt(g D)`.
pub const WAKE_DEPTH: f32 = 0.05;
const GRAVITY: f32 = 9.81;
const WAKE_JACOBI: usize = 6;
/// The film's viscosity on the vertical speed, m2/s: a 5 cm ripple lives
/// about a second, a 30 cm wave many.
const WAKE_FILM: f32 = 1.5e-4;
/// Drag on everything, per second, and the sponge's at the grid's edge
/// over its outer `WAKE_SPONGE` texels.
const WAKE_DRAG: f32 = 0.25;
const WAKE_SPONGE: usize = 10;
const WAKE_SPONGE_DRAG: f32 = 6.0;
/// The longest step, seconds: two half steps at 36 Hz.
const WAKE_STEP: f32 = 1.0 / 60.0;
/// Seconds with no source in the water before the field may rest.
pub const WAKE_REST: f32 = 4.0;
/// The most a source raises or lowers the water in one second, metres: a
/// cap on a body thrown through the field.
const WAKE_MOST: f32 = 0.6;

/// Something in the water, this frame: where it crosses the surface (world;
/// y is the surface), how fast it is moving, how wide it is there (its
/// radius), and how deep it reaches below the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WakeSource {
    pub position: Vec3,
    pub velocity: Vec3,
    pub radius: f32,
    pub depth: f32,
}

pub struct WakeField {
    h: Vec<f32>,
    u: Vec<f32>,
    a: Vec<f32>,
    rhs: Vec<f32>,
    /// The world cell of texel (0, 0).
    origin: IVec2,
    quiet: f32,
    resting: bool,
}

impl Default for WakeField {
    fn default() -> Self {
        let n = WAKE_N * WAKE_N;
        Self { h: vec![0.0; n], u: vec![0.0; n], a: vec![0.0; n], rhs: vec![0.0; n], origin: IVec2::ZERO, quiet: WAKE_REST, resting: true }
    }
}

impl WakeField {
    /// The cell a world x, z lies in.
    fn cell_of(p: Vec2) -> IVec2 {
        (p / WAKE_DX).floor().as_ivec2()
    }

    /// For the water's shader (`WaterUniform::wake`): the world x, z of the
    /// grid's corner, one over its side, and 1 while it holds anything.
    pub fn uniform(&self) -> [f32; 4] {
        let corner = self.origin.as_vec2() * WAKE_DX;
        [corner.x, corner.y, 1.0 / (WAKE_N as f32 * WAKE_DX), if self.resting { 0.0 } else { 1.0 }]
    }

    /// The height at a world x, z (nearest texel), metres: for tests.
    pub fn height_at(&self, p: Vec2) -> f32 {
        let c = Self::cell_of(p) - self.origin;
        if c.x < 0 || c.y < 0 || c.x >= WAKE_N as i32 || c.y >= WAKE_N as i32 {
            return 0.0;
        }
        self.h[c.y as usize * WAKE_N + c.x as usize]
    }

    /// Whether it holds nothing and is not stepped.
    pub fn is_resting(&self) -> bool {
        self.resting
    }

    /// Move the grid so `centre` (world x, z) is in its middle, a whole
    /// texel at a time, keeping the waves where they are in the world.
    fn follow(&mut self, centre: Vec2) {
        let want = Self::cell_of(centre) - IVec2::splat(WAKE_N as i32 / 2);
        let d = want - self.origin;
        if d == IVec2::ZERO {
            return;
        }
        self.origin = want;
        if self.resting || d.x.unsigned_abs() as usize >= WAKE_N || d.y.unsigned_abs() as usize >= WAKE_N {
            self.h.iter_mut().for_each(|x| *x = 0.0);
            self.u.iter_mut().for_each(|x| *x = 0.0);
            return;
        }
        let n = WAKE_N as i32;
        let shift = |v: &mut Vec<f32>| {
            let old = v.clone();
            for j in 0..n {
                for i in 0..n {
                    let (si, sj) = (i + d.x, j + d.y);
                    v[(j * n + i) as usize] = if si >= 0 && sj >= 0 && si < n && sj < n { old[(sj * n + si) as usize] } else { 0.0 };
                }
            }
        };
        shift(&mut self.h);
        shift(&mut self.u);
    }

    /// One frame: follow `centre`, push the water by `sources`, and step
    /// `dt` seconds. False when it rests (nothing to upload or draw).
    pub fn step(&mut self, dt: f32, centre: Vec2, sources: &[WakeSource]) -> bool {
        self.follow(centre);
        let dt = dt.clamp(0.0, 0.1);
        let touching = sources.iter().any(|s| s.depth > 0.0 && s.radius > 0.0);
        self.quiet = if touching { 0.0 } else { self.quiet + dt };
        if self.resting && !touching {
            return false;
        }
        self.resting = false;
        let steps = (dt / WAKE_STEP).ceil().max(1.0) as usize;
        let h_dt = dt / steps as f32;
        for _ in 0..steps {
            self.push(h_dt, sources);
            self.wave(h_dt);
        }
        if self.quiet >= WAKE_REST && self.h.iter().all(|x| x.abs() < 1e-5) {
            self.h.iter_mut().for_each(|x| *x = 0.0);
            self.u.iter_mut().for_each(|x| *x = 0.0);
            self.resting = true;
            return false;
        }
        true
    }

    /// THE SOURCES, for `dt`: each moves its submerged volume with it, and
    /// pushes water out round it going down (draws it in coming up).
    fn push(&mut self, dt: f32, sources: &[WakeSource]) {
        let n = WAKE_N as i32;
        for s in sources {
            if s.depth <= 0.0 || s.radius <= 0.0 {
                continue;
            }
            let sigma = s.radius.max(WAKE_DX);
            let volume = std::f32::consts::PI * s.radius * s.radius * s.depth.min(2.0 * s.radius);
            let across = Vec2::new(s.velocity.x, s.velocity.z);
            let down = -s.velocity.y;
            if across.length_squared() < 1e-6 && down.abs() < 1e-3 {
                continue;
            }
            let centre = Vec2::new(s.position.x, s.position.z);
            let reach = (3.0 * 2.0 * sigma / WAKE_DX).ceil() as i32;
            let c = Self::cell_of(centre) - self.origin;
            let (s2, w2) = (sigma * sigma, 4.0 * sigma * sigma);
            let norm = 1.0 / (2.0 * std::f32::consts::PI * s2);
            for j in (c.y - reach).max(0)..(c.y + reach + 1).min(n) {
                for i in (c.x - reach).max(0)..(c.x + reach + 1).min(n) {
                    let x = (self.origin + IVec2::new(i, j)).as_vec2() * WAKE_DX + Vec2::splat(0.5 * WAKE_DX) - centre;
                    let r2 = x.length_squared();
                    let g = norm * (-0.5 * r2 / s2).exp();
                    // Carried across: -(v . grad)(V G) = V (v . x) / sigma^2 G.
                    let mut rate = volume * across.dot(x) / s2 * g;
                    // Down through the surface: out of the middle, round it
                    // (zero net volume: narrow minus wide).
                    let wide = 0.25 * norm * (-0.5 * r2 / w2).exp();
                    rate -= down * std::f32::consts::PI * s.radius * s.radius * (g - wide);
                    let k = (j * n + i) as usize;
                    self.h[k] += rate.clamp(-WAKE_MOST, WAKE_MOST) * dt;
                }
            }
        }
    }

    /// THE WAVES, for `dt`.
    fn wave(&mut self, dt: f32) {
        let n = WAKE_N;
        let inv = 1.0 / (WAKE_DX * WAKE_DX);
        let alpha = WAKE_DEPTH * WAKE_DEPTH / 3.0;
        let c2 = GRAVITY * WAKE_DEPTH;
        let lap = |f: &[f32], i: usize, j: usize| -> f32 {
            let at = |ii: usize, jj: usize| f[jj * n + ii];
            let l = if i > 0 { at(i - 1, j) } else { 0.0 };
            let r = if i + 1 < n { at(i + 1, j) } else { 0.0 };
            let d = if j > 0 { at(i, j - 1) } else { 0.0 };
            let u = if j + 1 < n { at(i, j + 1) } else { 0.0 };
            (l + r + d + u - 4.0 * at(i, j)) * inv
        };
        for j in 0..n {
            for i in 0..n {
                self.rhs[j * n + i] = c2 * lap(&self.h, i, j) + WAKE_FILM * lap(&self.u, i, j);
            }
        }
        // (1 - alpha lap) a = rhs, by Jacobi sweeps from last step's a.
        let diag = 1.0 + 4.0 * alpha * inv;
        for _ in 0..WAKE_JACOBI {
            let prev = self.a.clone();
            for j in 0..n {
                for i in 0..n {
                    let at = |ii: usize, jj: usize| prev[jj * n + ii];
                    let mut s = 0.0;
                    if i > 0 {
                        s += at(i - 1, j);
                    }
                    if i + 1 < n {
                        s += at(i + 1, j);
                    }
                    if j > 0 {
                        s += at(i, j - 1);
                    }
                    if j + 1 < n {
                        s += at(i, j + 1);
                    }
                    self.a[j * n + i] = (self.rhs[j * n + i] + alpha * inv * s) / diag;
                }
            }
        }
        for j in 0..n {
            let ej = j.min(n - 1 - j);
            for i in 0..n {
                let e = i.min(n - 1 - i).min(ej);
                let sponge = if e < WAKE_SPONGE { WAKE_SPONGE_DRAG * (1.0 - e as f32 / WAKE_SPONGE as f32).powi(2) } else { 0.0 };
                let k = j * n + i;
                self.u[k] = (self.u[k] + self.a[k] * dt) * (-(WAKE_DRAG + sponge) * dt).exp();
                self.h[k] = (self.h[k] + self.u[k] * dt) * (-sponge * dt).exp();
            }
        }
    }

    /// What the shader reads, `WAKE_N^2` texels of four half floats: the
    /// slope along x and z, and the negated curvature along x and z (the
    /// caustics' `curve`), central differences.
    pub fn texels(&self) -> Vec<u16> {
        let n = WAKE_N;
        let mut out = vec![0u16; n * n * 4];
        let at = |i: usize, j: usize| self.h[j * n + i];
        for j in 0..n {
            for i in 0..n {
                let (l, r) = (at(i.saturating_sub(1), j), at((i + 1).min(n - 1), j));
                let (d, u) = (at(i, j.saturating_sub(1)), at(i, (j + 1).min(n - 1)));
                let c = at(i, j);
                let v = [(r - l) / (2.0 * WAKE_DX), (u - d) / (2.0 * WAKE_DX), -(l + r - 2.0 * c) / (WAKE_DX * WAKE_DX), -(d + u - 2.0 * c) / (WAKE_DX * WAKE_DX)];
                for (k, x) in v.iter().enumerate() {
                    out[(j * n + i) * 4 + k] = f16_bits(*x);
                }
            }
        }
        out
    }
}

/// An f32 as IEEE half-precision bits (round to nearest, clamped to the
/// largest finite half).
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = x.abs();
    if !(a >= 6.1035156e-5) {
        // Subnormal or zero.
        return sign | ((a / 5.9604645e-8).round() as u16).min(0x3ff);
    }
    if a >= 65504.0 {
        return sign | 0x7bff;
    }
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    let mut h = ((e as u32) << 10) | (m >> 13);
    // Round half up on the dropped bits.
    if m & 0x1000 != 0 {
        h += 1;
    }
    sign | (h.min(0x7bff) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand(at: Vec2, v: Vec3) -> WakeSource {
        WakeSource { position: Vec3::new(at.x, 0.0, at.y), velocity: v, radius: 0.045, depth: 0.06 }
    }

    /// Still in the water, nothing; moving across it, a bow wave rises
    /// ahead and the water falls behind; then it all spreads and dies, and
    /// the field rests.
    #[test]
    fn a_hand_drawn_through_the_water_pushes_a_bow_and_leaves_a_wake_that_dies() {
        let mut w = WakeField::default();
        let dt = 1.0 / 36.0;
        for _ in 0..36 {
            assert!(w.step(dt, Vec2::ZERO, &[hand(Vec2::ZERO, Vec3::ZERO)]));
        }
        assert!(w.h.iter().all(|x| x.abs() < 1e-9), "a still hand makes no waves");
        // Drawn along +x at 0.8 m/s for a second.
        let mut p = Vec2::new(-0.4, 0.0);
        for _ in 0..36 {
            p.x += 0.8 * dt;
            w.step(dt, Vec2::ZERO, &[hand(p, Vec3::new(0.8, 0.0, 0.0))]);
        }
        let ahead = w.height_at(p + Vec2::new(0.06, 0.0));
        let behind = w.height_at(p - Vec2::new(0.06, 0.0));
        assert!(ahead > 1e-4 && behind < ahead, "bow {ahead} behind {behind}");
        // A V: well behind it, the water is disturbed off the line more than
        // it is ahead of the bow.
        let wide = (0..20).map(|k| w.height_at(p + Vec2::new(-0.5, 0.05 * k as f32 - 0.5)).abs()).fold(0.0, f32::max);
        let far_ahead = (0..20).map(|k| w.height_at(p + Vec2::new(0.6, 0.05 * k as f32 - 0.5)).abs()).fold(0.0, f32::max);
        assert!(wide > 3.0 * far_ahead && wide > 1e-4, "wake {wide} ahead {far_ahead}");
        // Out of the water: it spreads, dies, and the field rests.
        let mut steps = 0;
        while w.step(dt, Vec2::ZERO, &[]) {
            steps += 1;
            assert!(steps < 36 * 30, "never rests");
        }
        assert!(w.is_resting() && w.uniform()[3] == 0.0);
    }

    /// A faster hand's waves trail in a narrower V (and a slow one's spread
    /// round it): the wake follows speed with no rule of its own.
    #[test]
    fn a_faster_hand_leaves_a_narrower_v() {
        let spread = |speed: f32| {
            let mut w = WakeField::default();
            let dt = 1.0 / 72.0;
            let mut p = Vec2::new(-1.2, 0.0);
            let steps = (1.6 / speed / dt) as usize;
            for _ in 0..steps {
                p.x += speed * dt;
                w.step(dt, Vec2::ZERO, &[hand(p, Vec3::new(speed, 0.0, 0.0))]);
            }
            // Behind it, how far off its line the disturbance lies for how
            // far behind: the wedge's slope, weighted by the waves' height.
            let (mut off, mut back) = (0.0f32, 0.0f32);
            for j in 0..WAKE_N {
                for i in 0..WAKE_N {
                    let x = (w.origin + IVec2::new(i as i32, j as i32)).as_vec2() * WAKE_DX - p;
                    if x.x < -0.15 && x.x > -1.0 {
                        let a = w.h[j * WAKE_N + i].abs();
                        off += a * x.y.abs();
                        back += a * -x.x;
                    }
                }
            }
            off / back
        };
        let (slow, fast) = (spread(0.5), spread(1.5));
        assert!(fast < slow, "fast {fast} slow {slow}");
    }

    /// The grid follows the player a texel at a time, the waves staying put
    /// in the world, and the edge drinks what reaches it.
    #[test]
    fn the_grid_follows_the_player_and_its_waves_stay_put() {
        let mut w = WakeField::default();
        let dt = 1.0 / 72.0;
        for _ in 0..10 {
            w.step(dt, Vec2::ZERO, &[hand(Vec2::new(0.3, 0.3), Vec3::new(0.0, -1.0, 0.0))]);
        }
        let here = w.height_at(Vec2::new(0.3, 0.3));
        assert!(here.abs() > 1e-5);
        let mut moved = WakeField { h: w.h.clone(), u: w.u.clone(), a: w.a.clone(), rhs: w.rhs.clone(), origin: w.origin, quiet: w.quiet, resting: w.resting };
        moved.follow(Vec2::new(0.52, -0.31));
        assert_eq!(moved.height_at(Vec2::new(0.3, 0.3)), here);
        assert!(w.uniform()[3] == 1.0 && (w.uniform()[2] - 1.0 / 4.8).abs() < 1e-6);
    }

    #[test]
    fn half_floats_round_trip() {
        for (x, bits) in [(0.0f32, 0u16), (1.0, 0x3c00), (-2.0, 0xc000), (0.5, 0x3800), (65504.0, 0x7bff), (1e9, 0x7bff)] {
            assert_eq!(f16_bits(x), bits, "{x}");
        }
        assert_eq!(f16_bits(1.0e-6) & 0x7c00, 0);
    }
}
