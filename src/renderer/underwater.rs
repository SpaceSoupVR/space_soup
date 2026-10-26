use std::collections::HashMap;

use glam::{Vec2, Vec3};
use wgpu::*;

use crate::renderer::lights::wgsl_lights_block;
use crate::renderer::water_pipeline::{surface_fn_wgsl, surface_wgsl, swash_lift, WaterUniform, WaterVertex, LIGHT_PATH};
use crate::renderer::water_waves::{WaveField, MIPS, N as WAVE_N};

pub const IOR_RGB: [f32; 3] = [1.331, 1.334, 1.338];

pub const PHASE_G: f32 = 0.5;

pub const SHAFT_STEPS: u32 = 5;
pub const SHAFT_REACH: f32 = 14.0;

pub const SHAFT_CONTRAST: f32 = 0.75;

pub const PROBE_MARGIN: f32 = 0.92;

pub const MENISCUS_PX: f32 = 2.5;

pub const FILM_SECONDS: f32 = 1.2;

pub const MAX_SPHERES: usize = 8;

pub const EYE_REACH: f32 = 0.06;

const OPEN_DEPTH: f32 = 30.0;

const TRACE_COARSEN: i32 = 2;

const FOAM_TILES: f32 = 1.0 / 3.0;

const BED_ALBEDO: f32 = 0.45;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UnderUniform {

    pub view: [f32; 4],

    pub sun_dir: [f32; 4],

    pub sun: [f32; 4],

    pub sky: [f32; 4],

    pub sun_air: [f32; 4],

    pub sun_rgb: [f32; 4],

    pub counts: [f32; 4],

    pub spheres: [[f32; 4]; MAX_SPHERES],
}

pub struct UnderFrame<'a> {

    pub size: (u32, u32),

    pub fov_y: f32,

    pub sun: Option<(Vec3, Vec3)>,

    pub sky_down: Vec3,
    pub waterline: bool,

    pub film_age: Option<f32>,

    pub spheres: &'a [(Vec3, f32)],
}

pub fn fresnel(cos_i: f32, n1: f32, n2: f32) -> f32 {
    let cos_i = cos_i.clamp(0.0, 1.0);
    let sin_t2 = (n1 / n2).powi(2) * (1.0 - cos_i * cos_i);
    if sin_t2 >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin_t2).sqrt();
    let rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    let rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);
    f32::midpoint(rs * rs, rp * rp)
}

pub fn window_half_angle_deg(ior: f32) -> f32 {
    (1.0 / ior).asin().to_degrees()
}

pub fn refracted_sun(to_sun: Vec3) -> Option<(Vec3, f32)> {
    let cos_i = to_sun.y;
    if cos_i <= 0.0 {
        return None;
    }
    let n = IOR_RGB[1];
    let sin_t = (1.0 - cos_i * cos_i).max(0.0).sqrt() / n;
    let cos_t = (1.0 - sin_t * sin_t).sqrt();
    let level = Vec2::new(to_sun.x, to_sun.z);
    let level = if level.length() > 1e-6 { level.normalize() } else { Vec2::ZERO };
    Some((Vec3::new(-level.x * sin_t, -cos_t, -level.y * sin_t), 1.0 - fresnel(cos_i, 1.0, n)))
}

impl UnderUniform {
    pub fn new(f: &UnderFrame) -> Self {
        let mut u = Self::default();
        u.view = [
            f.size.0 as f32,
            f.size.1 as f32,
            f.fov_y / f.size.1.max(1) as f32,
            f.film_age.unwrap_or(-1.0),
        ];

        let sky = f.sky_down * 0.934;
        u.sky = [sky.x, sky.y, sky.z, if f.waterline { 1.0 } else { 0.0 }];
        u.sun_dir = [0.0, -1.0, 0.0, 1.0];
        if let Some((to_sun, rgb)) = f.sun {
            let to_sun = to_sun.normalize_or_zero();
            if let Some((beam, gets_in)) = refracted_sun(to_sun) {
                let on_level = rgb * to_sun.y * gets_in;
                u.sun_dir = [beam.x, beam.y, beam.z, 1.0 / (-beam.y).max(0.2)];
                u.sun = [on_level.x, on_level.y, on_level.z, 0.0];
                u.sun_air = [to_sun.x, to_sun.y, to_sun.z, 1.0];
                u.sun_rgb = [rgb.x, rgb.y, rgb.z, 0.0];
            }
        }
        let n = f.spheres.len().min(MAX_SPHERES);
        u.counts = [n as f32, 0.0, 0.0, 0.0];
        for (slot, (c, r)) in u.spheres.iter_mut().zip(f.spheres.iter()) {
            *slot = [c.x, c.y, c.z, *r];
        }
        u
    }
}

pub fn sun_from_lights(lights: &[crate::renderer::lights::Light], player_to_world: glam::Quat) -> Option<(Vec3, Vec3)> {
    let l = lights.iter().find(|l| l.kind == crate::renderer::lights::LightKind::Directional)?;
    let c = l.color.to_linear();
    Some(((player_to_world * -l.direction).normalize_or_zero(), Vec3::new(c[0], c[1], c[2]) * l.intensity))
}

pub fn scatter(w: &WaterUniform, u: &UnderUniform, z0: f32, d: Vec3, t: f32, shafts: f32) -> Vec3 {
    let k = Vec3::new(w.extinction[0], w.extinction[1], w.extinction[2]);
    let c = Vec3::new(w.scatter[0], w.scatter[1], w.scatter[2]) * (1.0 + LIGHT_PATH) * k;
    let column = |kk: Vec3| -> Vec3 {
        let a = k - kk * d.y;
        let x = a * t;
        let phi = Vec3::new(phi(x.x), phi(x.y), phi(x.z));
        Vec3::new((-kk.x * z0).exp(), (-kk.y * z0).exp(), (-kk.z * z0).exp()) * t * phi
    };
    let sky = column(k * LIGHT_PATH) * Vec3::from_slice(&u.sky[..3]);
    let beam = Vec3::from_slice(&u.sun_dir[..3]);
    let g = PHASE_G;
    let cos_theta = beam.dot(-d);
    let phase = (1.0 - g * g) / (1.0 + g * g - 2.0 * g * cos_theta).powf(1.5);
    let sun = column(k * u.sun_dir[3]) * Vec3::from_slice(&u.sun[..3]) * phase * shafts;
    c * (sky + sun)
}

fn phi(x: f32) -> f32 {
    if x.abs() < 1e-3 {
        1.0 - 0.5 * x
    } else {
        (1.0 - (-x).exp()) / x
    }
}

pub fn in_water_luminance(w: &WaterUniform, u: &UnderUniform, depth: f32) -> f32 {
    let s = scatter(w, u, depth.max(0.0), Vec3::X, 1.0e4, 1.0);
    0.2126 * s.x + 0.7152 * s.y + 0.0722 * s.z
}

pub fn meter_in_water(air: f32, water: f32, share: f32) -> f32 {
    let share = share.clamp(0.0, 1.0);
    if share <= 0.0 {
        return air;
    }
    (air.max(1e-6).ln() * (1.0 - share) + water.max(1e-6).ln() * share).exp()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EyeWater {

    Above,

    Waterline,

    Under,
}

pub struct StillDepth {
    cell: f32,
    cells: HashMap<(i32, i32), Vec<(Vec2, f32)>>,
    lo: Vec2,
    hi: Vec2,
}

impl StillDepth {
    pub fn new(verts: &[WaterVertex]) -> Self {
        let cell = 2.0;
        let mut cells: HashMap<(i32, i32), Vec<(Vec2, f32)>> = HashMap::new();
        let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
        for v in verts {
            let p = Vec2::new(v.position[0], v.position[2]);
            lo = lo.min(p);
            hi = hi.max(p);
            let key = ((p.x / cell).floor() as i32, (p.y / cell).floor() as i32);
            cells.entry(key).or_default().push((p, v.depth));
        }
        Self { cell, cells, lo, hi }
    }

    pub fn at(&self, xz: Vec2) -> Option<f32> {
        if xz.cmplt(self.lo).any() || xz.cmpgt(self.hi).any() {
            return None;
        }
        let (cx, cz) = ((xz.x / self.cell).floor() as i32, (xz.y / self.cell).floor() as i32);
        let mut best: Option<(f32, f32)> = None;
        for dz in -1..=1 {
            for dx in -1..=1 {
                for &(p, d) in self.cells.get(&(cx + dx, cz + dz)).into_iter().flatten() {
                    let r = p.distance_squared(xz);
                    if best.is_none_or(|(b, _)| r < b) {
                        best = Some((r, d));
                    }
                }
            }
        }
        Some(best.map_or(OPEN_DEPTH, |(_, d)| d))
    }
}

pub fn wave_reach(significant_height: f32, still_depth: f32) -> f32 {
    let t = (still_depth / 1.2).clamp(0.0, 1.0);
    0.8 * significant_height * t * t * (3.0 - 2.0 * t) + 0.02
}

pub fn eye_water(u: &WaterUniform, still_depth: f32, reach: f32, eye: Vec3) -> EyeWater {
    if still_depth <= 0.0 {
        return EyeWater::Above;
    }
    let surface = u.extinction[3] + swash_lift(u, still_depth, Vec2::new(eye.x, eye.z));
    if eye.y - EYE_REACH > surface + reach {
        EyeWater::Above
    } else if eye.y + EYE_REACH < surface - reach {
        EyeWater::Under
    } else {
        EyeWater::Waterline
    }
}

pub fn eye_share(state: EyeWater) -> f32 {
    match state {
        EyeWater::Above => 0.0,
        EyeWater::Waterline => 0.5,
        EyeWater::Under => 1.0,
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Surfacing {
    was_in: bool,
    left_at: Option<f64>,
}

impl Surfacing {

    pub fn update(&mut self, state: EyeWater, now: f64) -> Option<f32> {
        let is_in = state != EyeWater::Above;
        if self.was_in && !is_in {
            self.left_at = Some(now);
        }
        if is_in {
            self.left_at = None;
        }
        self.was_in = is_in;
        let age = (now - self.left_at?) as f32;
        (age < FILM_SECONDS).then_some(age.max(0.0))
    }
}

fn header(under_group: u32, waterline: bool, dual: bool) -> String {
    let f = |v: f32| format!("{v:?}");
    format!(
        "{enable}\n{lights}\n{surface}\n{consts}\n{bindings}\n{surface_fn}\n",
        enable = if dual { "enable dual_source_blending;" } else { "" },
        lights = wgsl_lights_block(0, 1),
        surface = surface_wgsl(),
        consts = format!(
            "const UW_PI: f32 = 3.14159265;\n\
             const UW_INF: f32 = 1.0e6;\n\
             const UW_LIGHT_PATH: f32 = {lp};\n\
             const UW_IOR: vec3<f32> = vec3<f32>({r}, {g}, {b});\n\
             const UW_PHASE_G: f32 = {pg};\n\
             const UW_SHAFT_STEPS: i32 = {steps};\n\
             const UW_SHAFT_REACH: f32 = {reach};\n\
             const UW_SHAFT_CONTRAST: f32 = {sc};\n\
             const UW_SHAFT_LOD: f32 = 2.0;\n\
             const UW_PROBE_MARGIN: f32 = {margin};\n\
             const UW_MENISCUS_PX: f32 = {men};\n\
             const UW_FILM_SECONDS: f32 = {film};\n\
             const UW_WATERLINE: bool = {waterline};\n\
             const UW_MIPS: i32 = {mips};\n\
             const UW_WAVE_N: f32 = {wave_n};\n\
             const UW_SHELF_SLOPE: f32 = 0.15;\n\
             const UW_OPEN_DEPTH: f32 = {open};\n\
             const UW_CAUSTIC_CONTRAST: f32 = 0.6;\n\
             const UW_BED_ALBEDO: f32 = {bed};\n\
             const UW_FOAM_TILES: f32 = {foam};\n\
             const UW_TRACE_COARSEN: i32 = {coarsen};\n\
             const UW_PAWS_RATE: vec2<f32> = vec2<f32>(0.1308997, 0.0785398);\n",
            lp = f(LIGHT_PATH),
            r = f(IOR_RGB[0]),
            g = f(IOR_RGB[1]),
            b = f(IOR_RGB[2]),
            pg = f(PHASE_G),
            steps = SHAFT_STEPS,
            reach = f(SHAFT_REACH),
            sc = f(SHAFT_CONTRAST),
            margin = f(PROBE_MARGIN),
            men = f(MENISCUS_PX),
            film = f(FILM_SECONDS),
            mips = MIPS,
            wave_n = f(WAVE_N as f32),
            open = f(OPEN_DEPTH),
            bed = f(BED_ALBEDO),
            foam = f(FOAM_TILES),
            coarsen = TRACE_COARSEN,
        ),
        bindings = format!(
            "@group(1) @binding(0) var<uniform> water: WaterMat;\n\
             @group(1) @binding(1) var wave_disp: texture_2d_array<f32>;\n\
             @group(1) @binding(2) var wave_slope: texture_2d_array<f32>;\n\
             @group(1) @binding(3) var wave_curve: texture_2d_array<f32>;\n\
             @group(1) @binding(4) var foam_tex: texture_2d<f32>;\n\
             @group(1) @binding(5) var wave_samp: sampler;\n\
             struct Under {{\n    view: vec4<f32>,\n    sun_dir: vec4<f32>,\n    sun: vec4<f32>,\n    sky: vec4<f32>,\n    sun_air: vec4<f32>,\n    sun_rgb: vec4<f32>,\n    counts: vec4<f32>,\n    spheres: array<vec4<f32>, {MAX_SPHERES}>,\n}}\n\
             @group({under_group}) @binding(0) var<uniform> under: Under;\n"
        ),
        surface_fn = surface_fn_wgsl("water_surface", "wave_disp"),
    ) + COMMON_WGSL
}

const COMMON_WGSL: &str = r"
fn uw_bed(at: vec3<f32>) -> vec4<f32> {
    let uv = ground_uv(at);
    let half_texel = 0.5 / vec2<f32>(textureDimensions(ground_map));
    let edge = clamp(uv, half_texel, vec2<f32>(1.0) - half_texel);
    let g = textureSampleLevel(ground_map, probe_samp, edge, 0.0);
    return vec4<f32>(g.rgb * exp2(camera.proxy_params.w), g.a - UW_SHELF_SLOPE * length((uv - edge) / camera.ground_params.zw));
}

fn uw_has_map() -> bool {
    return camera.sky_params.w > 0.5;
}

fn uw_still_depth(xz: vec2<f32>) -> f32 {
    if (!uw_has_map()) {
        return UW_OPEN_DEPTH;
    }
    return water.extinction.w - uw_bed(vec3<f32>(xz.x, 0.0, xz.y)).w;
}

fn uw_surface_y(xz: vec2<f32>, eye: vec3<f32>) -> f32 {
    let h = water.extinction.w;
    let depth = uw_still_depth(xz);
    let t = water.tiles.w;
    let first = water_surface(vec3<f32>(xz.x, h, xz.y), depth, eye, t);
    let rest = xz - (first.xz - xz);
    return water_surface(vec3<f32>(rest.x, h, rest.y), depth, eye, t).y;
}

fn uw_point(ndc: vec2<f32>, z: f32) -> vec3<f32> {
    let h = cam_inv_view_proj() * vec4<f32>(ndc, z, 1.0);
    return h.xyz / h.w;
}

fn uw_line_plane(eye: vec3<f32>) -> vec3<f32> {
    let h0 = uw_surface_y(eye.xz, eye);
    let hx = uw_surface_y(eye.xz + vec2<f32>(0.1, 0.0), eye);
    let hz = uw_surface_y(eye.xz + vec2<f32>(0.0, 0.1), eye);
    return vec3<f32>(h0, (hx - h0) * 10.0, (hz - h0) * 10.0);
}

fn uw_line_below(start: vec3<f32>, eye: vec3<f32>, plane: vec3<f32>) -> f32 {
    return plane.x + dot(start.xz - eye.xz, plane.yz) - start.y;
}

fn uw_starts_under(player: vec3<f32>, eye: vec3<f32>) -> bool {
    let c = cam_view_proj() * vec4<f32>(player, 1.0);
    let start = to_world_space(uw_point(c.xy / c.w, 0.0));
    return uw_line_below(start, eye, uw_line_plane(eye)) > 0.0;
}

fn uw_phi(x: vec3<f32>) -> vec3<f32> {
    return select((vec3<f32>(1.0) - exp(-x)) / x, vec3<f32>(1.0) - 0.5 * x, abs(x) < vec3<f32>(1e-3));
}

fn uw_column(z0: f32, dy: f32, t: f32, k: vec3<f32>, kk: vec3<f32>) -> vec3<f32> {
    return exp(-kk * z0) * t * uw_phi((k - kk * dy) * t);
}

fn uw_scatter(z0: f32, d: vec3<f32>, t: f32, shafts: f32) -> vec3<f32> {
    let k = water.extinction.xyz;
    let c = water.scatter.rgb * (1.0 + UW_LIGHT_PATH) * k;
    let sky = uw_column(z0, d.y, t, k, k * UW_LIGHT_PATH) * under.sky.rgb;
    let g = UW_PHASE_G;
    let cos_theta = dot(under.sun_dir.xyz, -d);
    let phase = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5);
    let sun = uw_column(z0, d.y, t, k, k * under.sun_dir.w) * under.sun.rgb * (phase * shafts);
    return c * (sky + sun);
}

fn uw_gather(curve: vec3<f32>, focus: f32) -> f32 {
    let jx = 1.0 - focus * curve.x;
    let jz = 1.0 - focus * curve.y;
    let jxz = focus * curve.z;
    let det = jx * jz - jxz * jxz;
    return inverseSqrt(det * det + 0.0625);
}

fn uw_shafts(e: vec3<f32>, d: vec3<f32>, t: f32) -> f32 {
    if (under.sun_air.w < 0.5) {
        return 1.0;
    }
    let surface = water.extinction.w;
    let reach = min(t, UW_SHAFT_REACH);
    let s = under.sun_dir.xyz;
    let up = under.sun_dir.w;
    let kg = water.extinction.y;
    var sum = 0.0;
    var weights = 0.0;
    for (var i = 0; i < UW_SHAFT_STEPS; i = i + 1) {
        let ti = reach * (f32(i) + 0.5) / f32(UW_SHAFT_STEPS);
        let x = e + d * ti;
        let z = max(surface - x.y, 0.0);
        let q = x.xz - s.xz * (z * up);
        let curve = textureSampleLevel(wave_curve, wave_samp, q / water.tiles.x, 0, UW_SHAFT_LOD).xyz * 0.5
            + textureSampleLevel(wave_curve, wave_samp, q / water.tiles.y, 1, UW_SHAFT_LOD).xyz;
        let focus = z * up * (1.0 - 1.0 / UW_IOR.y);

        let keeps = UW_SHAFT_CONTRAST * exp(-kg * z * up * 0.5);
        let w = exp(-kg * (ti + z * up));
        sum = sum + mix(1.0, uw_gather(curve, focus), keeps) * w;
        weights = weights + w;
    }
    let mean = sum / max(weights, 1e-6);

    let a = kg * max(1.0 - up * d.y, 0.05);
    let near_share = (1.0 - exp(-a * reach)) / max(1.0 - exp(-a * t), 1e-6);
    return mix(1.0, mean, clamp(near_share, 0.0, 1.0));
}

fn uw_caustic(p: vec3<f32>, z: f32, span: f32) -> f32 {
    let up = under.sun_dir.w;
    let q = p.xz - under.sun_dir.xz * (z * up);
    let l0 = clamp(log2(max(span * UW_WAVE_N / water.tiles.x, 1e-6)), 0.0, f32(UW_MIPS - 1));
    let l1 = clamp(log2(max(span * UW_WAVE_N / water.tiles.y, 1e-6)), 0.0, f32(UW_MIPS - 1));

    let l2 = clamp(log2(max(span * UW_WAVE_N / water.tiles.z, 1e-6)) + 2.0, 2.0, f32(UW_MIPS - 1));
    let c0 = textureSampleLevel(wave_curve, wave_samp, q / water.tiles.x, 0, l0).xyz;
    let c1 = textureSampleLevel(wave_curve, wave_samp, q / water.tiles.y, 1, l1).xyz;
    let c2 = textureSampleLevel(wave_curve, wave_samp, q / water.tiles.z, 2, l2).xyz;
    let shoal = smoothstep(0.0, 1.2, uw_still_depth(p.xz));
    let fine = 0.5 * (1.0 - smoothstep(0.08, 0.3, span * UW_WAVE_N * 0.25 / water.tiles.z));
    let mid = max(shoal, 0.6) * (1.0 - smoothstep(0.08, 0.3, span * UW_WAVE_N / water.tiles.y));
    let curve = c0 * (0.5 * shoal) + c1 * mid + c2 * fine;
    let gather = uw_gather(curve, z * up * (1.0 - 1.0 / UW_IOR.y));
    let k = water.extinction.xyz;
    let murk = exp(-(k.x + k.y + k.z) * (1.0 / 3.0) * z);
    return mix(1.0, gather, UW_CAUSTIC_CONTRAST * murk * (1.0 - smoothstep(1.5, 5.0, z)));
}
";

fn veil_wgsl(waterline: bool, dual: bool) -> String {
    let output = if dual {
        "struct VeilOut {\n    @location(0) @blend_src(0) colour: vec4<f32>,\n    @location(0) @blend_src(1) through: vec4<f32>,\n    @builtin(frag_depth) depth: f32,\n}\n\
         struct FilmOut {\n    @location(0) @blend_src(0) colour: vec4<f32>,\n    @location(0) @blend_src(1) through: vec4<f32>,\n}\n\
         fn uw_out(colour: vec3<f32>, through: vec3<f32>, depth: f32) -> VeilOut { return VeilOut(vec4<f32>(colour, 1.0), vec4<f32>(through, 1.0), depth); }\n\
         fn uw_film_out(colour: vec3<f32>, through: vec3<f32>) -> FilmOut { return FilmOut(vec4<f32>(colour, 1.0), vec4<f32>(through, 1.0)); }\n"
    } else {
        "struct VeilOut {\n    @location(0) colour: vec4<f32>,\n    @builtin(frag_depth) depth: f32,\n}\n\
         struct FilmOut {\n    @location(0) colour: vec4<f32>,\n}\n\
         fn uw_out(colour: vec3<f32>, through: vec3<f32>, depth: f32) -> VeilOut { return VeilOut(vec4<f32>(colour, 1.0 - dot(through, vec3<f32>(1.0 / 3.0))), depth); }\n\
         fn uw_film_out(colour: vec3<f32>, through: vec3<f32>) -> FilmOut { return FilmOut(vec4<f32>(colour, 1.0 - dot(through, vec3<f32>(1.0 / 3.0)))); }\n"
    };
    format!("{}{}{}", header(3, waterline, dual), output, VEIL_WGSL)
}

const VEIL_WGSL: &str = r"
@group(2) @binding(1) var probe_pass_depth: texture_depth_2d_array;

struct FullOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex fn vs_full(@builtin(vertex_index) i: u32) -> FullOut {
    var o: FullOut;
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    o.ndc = uv * 2.0 - vec2<f32>(1.0);
    o.clip = vec4<f32>(o.ndc, 0.0, 1.0);
    return o;
}

fn uw_distance(ndc: vec2<f32>, z: f32, eye: vec3<f32>) -> f32 {
    if (z >= 1.0) {
        return UW_INF;
    }
    return distance(uw_point(ndc, z), eye);
}

fn uw_veil(in: FullOut, near: bool) -> VeilOut {
    let eye_p = cam_pos();
    let start_p = uw_point(in.ndc, 0.0);
    let dir_p = normalize(uw_point(in.ndc, 0.5) - start_p);
    let eye = to_world_space(eye_p);
    let d = to_world_direction(dir_p);
    let start = to_world_space(start_p);
    let surface = water.extinction.w;

    var wet = 1.0;
    var band = 0.0;
    var px = 0.0;
    if (UW_WATERLINE) {
        let below = uw_line_below(start, eye, uw_line_plane(eye));
        px = below / max(fwidth(below), 1e-7);
        if (px < -UW_MENISCUS_PX * 2.0) {
            discard;
        }
        wet = smoothstep(-0.5, 0.5, px);
        band = 1.0 - smoothstep(0.0, UW_MENISCUS_PX, abs(px));
    }

    let size = vec2<f32>(textureDimensions(probe_pass_depth));
    let f = in.clip.xy * size / under.view.xy - vec2<f32>(0.5);
    let base = floor(f);
    let w = f - base;
    let top = vec2<i32>(size) - vec2<i32>(1);
    let b = vec2<i32>(base);
    let d00 = uw_distance(in.ndc, textureLoad(probe_pass_depth, clamp(b, vec2<i32>(0), top), view_slot, 0), eye_p);
    let d10 = uw_distance(in.ndc, textureLoad(probe_pass_depth, clamp(b + vec2<i32>(1, 0), vec2<i32>(0), top), view_slot, 0), eye_p);
    let d01 = uw_distance(in.ndc, textureLoad(probe_pass_depth, clamp(b + vec2<i32>(0, 1), vec2<i32>(0), top), view_slot, 0), eye_p);
    let d11 = uw_distance(in.ndc, textureLoad(probe_pass_depth, clamp(b + vec2<i32>(1, 1), vec2<i32>(0), top), view_slot, 0), eye_p);
    let lo = min(min(d00, d10), min(d01, d11));
    let hi = max(max(d00, d10), max(d01, d11));

    var reach = select(hi, mix(mix(d00, d10, w.x), mix(d01, d11, w.x), w.y), hi < lo * 1.12);

    if (UW_WATERLINE && lo >= UW_INF * 0.5) {
        wet = step(0.0, px);
    }

    var z_test = 0.9999999;
    if (lo < UW_INF * 0.5) {
        let c = cam_view_proj() * vec4<f32>(eye_p + dir_p * (lo * UW_PROBE_MARGIN), 1.0);
        z_test = clamp(c.z / c.w, 0.0, 0.9999999);
    } else if (wet < 0.5 || (UW_WATERLINE && d.y > 0.0)) {

        z_test = 1.0;
    }

    if (near) {
        var est = select(0.5 * lo, 1.0, lo >= UW_INF * 0.5);
        var best = UW_INF;
        for (var i = 0; i < i32(under.counts.x); i = i + 1) {
            let s = under.spheres[i];
            let oc = s.xyz - eye_p;
            let along = dot(oc, dir_p);
            let disc = along * along - (dot(oc, oc) - s.w * s.w);
            if (disc > 0.0 && along > 0.0) {
                let t_in = max(along - sqrt(disc), 0.05);
                best = min(best, t_in + 0.3 * (along - t_in));
            }
        }
        if (best < UW_INF) {
            est = best;
        }
        reach = min(est, select(lo * UW_PROBE_MARGIN, est, lo >= UW_INF * 0.5));
    }

    let z0 = max(surface - eye.y, 0.0);
    var t = reach;
    if (d.y > 1e-4) {
        t = min(t, z0 / d.y);
    }
    let k = water.extinction.xyz;
    let t_view = exp(-k * t);

    var shafts = 1.0;
    if (!near) {
        shafts = uw_shafts(eye, d, t);
    }
    let light = uw_scatter(z0, d, t, shafts);
    var through = t_view;
    var extra = vec3<f32>(0.0);
    if (near) {
        let p = eye + d * reach;
        through = t_view * exp(-k * (max(surface - p.y, 0.0) * UW_LIGHT_PATH));
    } else if (reach < UW_INF * 0.5 && reach <= t + 1e-3) {

        let p = eye + d * reach;
        let zp = max(surface - p.y, 0.0);
        let sun_on = under.sun.rgb * exp(-k * (zp * under.sun_dir.w));
        let sky_on = under.sky.rgb * exp(-k * (zp * UW_LIGHT_PATH));
        let lit_dry = max(under.sun.rgb + under.sky.rgb, vec3<f32>(1e-4));
        let sun_share = dot(sun_on, vec3<f32>(1.0)) / max(dot(sun_on + sky_on, vec3<f32>(1.0)), 1e-4);
        let caustic = uw_caustic(p, zp, reach * under.view.z);
        let t_light = (sun_on + sky_on) / lit_dry;
        through = t_view * t_light * (1.0 + min(caustic - 1.0, 0.0) * sun_share);
        extra = max(caustic - 1.0, 0.0) * UW_BED_ALBEDO * sun_on * t_view;
    }
    var colour = tonemap(light + extra);
    through = mix(vec3<f32>(1.0), through, wet);
    colour = colour * wet;
    if (UW_WATERLINE) {

        through = through * (1.0 - 0.5 * band);
        let rim = exp(-(px - 1.0) * (px - 1.0) / 0.8) * wet;
        colour = colour * (1.0 - 0.5 * band) + rim * 0.35 * tonemap(under.sky.rgb + under.sun.rgb);
    }
    return uw_out(colour, through, z_test);
}

@fragment fn fs_veil(in: FullOut) -> VeilOut {
    return uw_veil(in, false);
}

@fragment fn fs_near(in: FullOut) -> VeilOut {
    return uw_veil(in, true);
}

@fragment fn fs_film(in: FullOut) -> FilmOut {
    let age = max(under.view.w, 0.0);
    let fade = 1.0 - smoothstep(0.0, UW_FILM_SECONDS, age);
    let v = 0.5 - 0.5 * in.ndc.y;
    let edge = -0.2 + 1.5 * age / UW_FILM_SECONDS;

    let wavy = edge + 0.03 * sin(in.ndc.x * 5.3 + 1.3) + 0.02 * sin(in.ndc.x * 13.1 + 0.4) + 0.012 * sin(in.ndc.x * 29.7);
    let mask = smoothstep(wavy - 0.18, wavy + 0.18, v);
    let rim = exp(-(v - wavy) * (v - wavy) / 0.002) * 0.3;
    let veil = 0.28 * mask * fade;

    let glow = tonemap(under.sky.rgb * 1.5);
    return uw_film_out(glow * veil + glow * (0.08 * rim * fade), vec3<f32>(1.0 - veil));
}
";

fn underside_wgsl() -> String {
    format!("{}{}", header(2, false, false), UNDERSIDE_WGSL)
}

const UNDERSIDE_WGSL: &str = r"
struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) depth: f32,
}
struct UOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) depth: f32,
    @location(1) tex_pos: vec3<f32>,
    @location(2) rest: vec2<f32>,
}

@vertex fn vs_under(v: VIn) -> UOut {
    var out: UOut;
    let p = water_surface(v.pos, v.depth, to_world_space(cam_pos()), water.tiles.w);
    out.clip = cam_view_proj() * vec4<f32>(to_player_space(p), 1.0);
    if (out.clip.w > 0.0) {
        out.clip.z = min(out.clip.z, out.clip.w * 0.999999);
    }
    out.depth = v.depth;
    out.tex_pos = p;
    out.rest = v.pos.xz;
    return out;
}

fn uw_lost(lod: f32) -> vec4<f32> {
    let l = clamp(lod, 0.0, f32(UW_MIPS - 1));
    let i = i32(floor(l));
    let j = min(i + 1, UW_MIPS - 1);
    return mix(water.lost[i], water.lost[j], l - f32(i));
}

fn uw_ggx(nh: f32, a2: f32) -> f32 {
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (UW_PI * dd * dd);
}

fn uw_smith(c: f32, a2: f32) -> f32 {
    return 2.0 * c / (c + sqrt(a2 + (1.0 - a2) * c * c));
}

@fragment fn fs_under(in: UOut) -> @location(0) vec4<f32> {

    let q = in.rest;
    let span = max(length(dpdx(q)), length(dpdy(q)));
    let p = in.tex_pos;
    let geometric = cross(dpdx(p), dpdy(p));
    let s0 = textureSample(wave_slope, wave_samp, q / water.tiles.x, 0);
    let s1 = textureSample(wave_slope, wave_samp, q / water.tiles.y, 1);
    let lull = sin(dot(q, vec2<f32>(0.47, 0.31)) + water.tiles.w * UW_PAWS_RATE.x + 1.6 * sin(dot(q, vec2<f32>(-0.27, 0.58)) - water.tiles.w * UW_PAWS_RATE.y));
    let paws = mix(water.air.x, 1.0, smoothstep(-0.6, 0.7, lull));
    let s2 = textureSample(wave_slope, wave_samp, q / water.tiles.z, 2) * paws;
    let c0 = textureSample(wave_curve, wave_samp, q / water.tiles.x, 0);
    let c1 = textureSample(wave_curve, wave_samp, q / water.tiles.y, 1);
    let bubbles = textureSample(foam_tex, wave_samp, q * UW_FOAM_TILES).r;

    let eye = to_world_space(cam_pos());
    let to_p = p - eye;
    let dist = length(to_p);
    let d = to_p / max(dist, 1e-5);

    let ng = normalize(geometric) * select(1.0, -1.0, geometric.y < 0.0);
    if (under.sky.w > 0.5 && (dot(d, ng) < 0.0 || !uw_starts_under(to_player_space(p), eye))) {
        discard;
    }

    let surface = water.extinction.w;
    let still_depth = select(in.depth, surface - uw_bed(p).w, uw_has_map());
    let shoal = smoothstep(0.0, 1.2, still_depth);
    let slope = (s0.xy + s1.xy) * shoal + s2.xy;
    let push = vec2<f32>(1.0) + (s0.zw + s1.zw) * shoal + s2.zw;
    var tilt = slope / max(push, vec2<f32>(0.25));
    if (water.air.y > 0.0) {
        tilt = tilt + water.air.y * smoothstep(0.15, 0.6, still_depth) * swell_at(q, water.tiles.w).yz;
    }

    let n = normalize(vec3<f32>(-tilt.x, 1.0, -tilt.y));
    let lost0 = uw_lost(log2(max(span * UW_WAVE_N / water.tiles.x, 1e-6))).x;
    let lost1 = uw_lost(log2(max(span * UW_WAVE_N / water.tiles.y, 1e-6))).y;
    let lost2 = uw_lost(log2(max(span * UW_WAVE_N / water.tiles.z, 1e-6))).z;
    let a2 = clamp((lost0 + lost1) * shoal * shoal + (lost2 + water.waves.x) * paws * paws, 4e-4, 1.0);
    let rough = sqrt(sqrt(a2));
    let k = water.extinction.xyz;

    let cos_i = max(dot(d, n), 1e-3);
    let sin2 = 1.0 - cos_i * cos_i;
    let st2 = UW_IOR * UW_IOR * sin2;
    let ct = sqrt(max(vec3<f32>(1.0) - st2, vec3<f32>(0.0)));
    let rs = (UW_IOR * cos_i - ct) / (UW_IOR * cos_i + ct);
    let rp = (vec3<f32>(cos_i) - UW_IOR * ct) / (vec3<f32>(cos_i) + UW_IOR * ct);
    let sharp = select(0.5 * (rs * rs + rp * rp), vec3<f32>(1.0), st2 >= vec3<f32>(1.0));
    let soft = clamp(1.5 * sqrt(a2), 0.01, 0.35);
    let f = max(min(sharp, vec3<f32>(1.0)), smoothstep(vec3<f32>(1.0 - soft), vec3<f32>(1.0 + soft), st2));

    var out_dir = refract(d, -n, UW_IOR.y);
    out_dir = normalize(vec3<f32>(out_dir.x, max(out_dir.y, 0.002), out_dir.z));
    ground_trace_coarsen = UW_TRACE_COARSEN;
    var outside = vec3<f32>(0.0);
    if (min(f.x, min(f.y, f.z)) < 0.999) {
        outside = outdoor_radiance(p, out_dir, to_player_direction(out_dir), rough, clamp(rough * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD));
    }
    outside = outside * (UW_IOR.y * UW_IOR.y);

    var sun_through = vec3<f32>(0.0);
    if (under.sun_air.w > 0.5) {
        let l = under.sun_air.xyz;
        let o = -d;
        var h = -(l + UW_IOR.y * o);
        h = normalize(h) * select(1.0, -1.0, h.y < 0.0);
        let ih = dot(l, h);
        let oh = dot(o, h);
        let ln = max(dot(l, n), 0.0);
        let nh = max(dot(n, h), 0.0);
        if (ih > 0.0 && oh < 0.0 && ln > 0.0) {
            let fh = 0.02 + 0.98 * pow(1.0 - ih, 5.0);
            let denom = ih + UW_IOR.y * oh;
            let bt = abs(ih) * abs(oh) / max(ln * cos_i, 1e-4) * UW_IOR.y * UW_IOR.y * (1.0 - fh)
                * uw_smith(ln, a2) * uw_smith(cos_i, a2) * uw_ggx(nh, a2) / max(denom * denom, 1e-4);
            sun_through = under.sun_rgb.rgb * (UW_PI * bt * ln);
        }
    }

    var r = reflect(d, n);
    r = normalize(vec3<f32>(r.x, min(r.y, -0.02), r.z));
    var t_bed = UW_INF;
    var bed_light = vec3<f32>(0.0);
    if (uw_has_map()) {
        let first = max(p.y - uw_bed(p).w, 0.0) / -r.y;
        let x1 = p + r * first;
        t_bed = max(first + (x1.y - uw_bed(x1).w) / -r.y, 0.0);
        let xb = p + r * t_bed;
        bed_light = uw_bed(xb).rgb * exp(-k * (max(surface - xb.y, 0.0) * UW_LIGHT_PATH));
    }
    let mirrored = bed_light * exp(-k * t_bed) + uw_scatter(max(surface - p.y, 0.0), r, t_bed, 1.0);

    var light = f * mirrored + (vec3<f32>(1.0) - f) * outside + sun_through;

    let caps = max(c0.w, c1.w) * shoal;
    let cover = clamp((bubbles - (1.0 - caps)) * 4.0, 0.0, 1.0) * smoothstep(0.0, 0.05, caps);
    light = mix(light, 0.12 * (under.sky.rgb + under.sun.rgb), 0.8 * cover);

    let z0 = max(surface - eye.y, 0.0);
    let seen = light * exp(-k * dist) + uw_scatter(z0, d, dist, uw_shafts(eye, d, dist));
    return vec4<f32>(tonemap(seen), 1.0);
}
";

pub struct UnderwaterPipelines {

    pub water_layout: BindGroupLayout,

    pub under_layout: BindGroupLayout,
    pub sampler: Sampler,

    pub veil: RenderPipeline,

    pub veil_near: RenderPipeline,

    pub veil_line: RenderPipeline,
    pub veil_line_near: RenderPipeline,
    pub underside: RenderPipeline,
    pub film: RenderPipeline,
    pub dual_source: bool,
}

pub fn veil_pair(p: &UnderwaterPipelines, state: EyeWater) -> Option<(&RenderPipeline, &RenderPipeline)> {
    match state {
        EyeWater::Above => None,
        EyeWater::Waterline => Some((&p.veil_line, &p.veil_line_near)),
        EyeWater::Under => Some((&p.veil, &p.veil_near)),
    }
}

impl UnderwaterPipelines {

    pub fn new(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        probe_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        let dual = device.features().contains(Features::DUAL_SOURCE_BLENDING);
        Self::new_with(device, format, camera_layout, probe_layout, samples, dual)
    }

    pub fn new_with(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        probe_layout: &BindGroupLayout,
        samples: u32,
        dual: bool,
    ) -> Self {
        let both = ShaderStages::VERTEX_FRAGMENT;
        let texture = |binding, view_dimension| BindGroupLayoutEntry {
            binding,
            visibility: both,
            ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension, multisampled: false },
            count: None,
        };
        let uniform = |binding| BindGroupLayoutEntry {
            binding,
            visibility: both,
            ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let water_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("underwater_water_bgl"),
            entries: &[
                uniform(0),
                texture(1, TextureViewDimension::D2Array),
                texture(2, TextureViewDimension::D2Array),
                texture(3, TextureViewDimension::D2Array),
                texture(4, TextureViewDimension::D2),
                BindGroupLayoutEntry { binding: 5, visibility: both, ty: BindingType::Sampler(SamplerBindingType::Filtering), count: None },
            ],
        });
        let under_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("underwater_under_bgl"),
            entries: &[uniform(0)],
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("underwater_waves_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });

        let veil_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("underwater_veil_layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&water_layout), Some(probe_layout), Some(&under_layout)],
            immediate_size: 0,
        });
        let under_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("underwater_underside_layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&water_layout), Some(&under_layout)],
            immediate_size: 0,
        });

        let blend = if dual {
            BlendState {
                color: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::Src1, operation: BlendOperation::Add },
                alpha: BlendComponent { src_factor: BlendFactor::Zero, dst_factor: BlendFactor::One, operation: BlendOperation::Add },
            }
        } else {
            BlendState {
                color: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::OneMinusSrcAlpha, operation: BlendOperation::Add },
                alpha: BlendComponent { src_factor: BlendFactor::Zero, dst_factor: BlendFactor::One, operation: BlendOperation::Add },
            }
        };
        let module = |label: &str, src: String| device.create_shader_module(ShaderModuleDescriptor { label: Some(label), source: ShaderSource::Wgsl(src.into()) });
        let open_sea = module("underwater_veil", veil_wgsl(false, dual));
        let line = module("underwater_veil_waterline", veil_wgsl(true, dual));
        let full = |label: &str, shader: &ShaderModule, entry: &str, compare: CompareFunction, write: bool| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&veil_layout),
                vertex: VertexState { module: shader, entry_point: Some("vs_full"), buffers: &[], compilation_options: Default::default() },
                fragment: Some(FragmentState {
                    module: shader,
                    entry_point: Some(entry),
                    targets: &[Some(ColorTargetState { format, blend: Some(blend), write_mask: ColorWrites::ALL })],
                    compilation_options: Default::default(),
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: Some(write),
                    depth_compare: Some(compare),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: MultisampleState { count: samples, ..Default::default() },
                multiview_mask: None,
                cache: None,
            })
        };

        let veil = full("underwater_veil", &open_sea, "fs_veil", CompareFunction::Less, true);
        let veil_near = full("underwater_veil_near", &open_sea, "fs_near", CompareFunction::Greater, false);
        let veil_line = full("underwater_veil_line", &line, "fs_veil", CompareFunction::Less, true);
        let veil_line_near = full("underwater_veil_line_near", &line, "fs_near", CompareFunction::Greater, false);
        let film = full("underwater_film", &open_sea, "fs_film", CompareFunction::Always, false);

        let under_module = module("underwater_underside", underside_wgsl());
        let underside = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("underwater_underside"),
            layout: Some(&under_pipeline_layout),
            vertex: VertexState {
                module: &under_module,
                entry_point: Some("vs_under"),
                buffers: &[Some(VertexBufferLayout {
                    array_stride: std::mem::size_of::<WaterVertex>() as BufferAddress,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &under_module,
                entry_point: Some("fs_under"),
                targets: &[Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),

            primitive: PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: Default::default(),

                bias: DepthBiasState { constant: -16, slope_scale: -2.0, clamp: 0.0 },
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: None,
            cache: None,
        });
        Self { water_layout, under_layout, sampler, veil, veil_near, veil_line, veil_line_near, underside, film, dual_source: dual }
    }

    pub fn water_groups(&self, device: &Device, uniform: &Buffer, waves: &WaveField) -> [BindGroup; 2] {
        std::array::from_fn(|set| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("underwater_water"),
                layout: &self.water_layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&waves.displacement_views[set]) },
                    BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&waves.derivatives_view) },
                    BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&waves.curvature_views[set]) },
                    BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&waves.foam_view) },
                    BindGroupEntry { binding: 5, resource: BindingResource::Sampler(&self.sampler) },
                ],
            })
        })
    }

    pub fn under_buffer(&self, device: &Device) -> (Buffer, BindGroup) {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("underwater_uniform"),
            size: std::mem::size_of::<UnderUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("underwater_under"),
            layout: &self.under_layout,
            entries: &[BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        (buffer, group)
    }

    pub fn draw_veil(&self, pass: &mut RenderPass<'_>, state: EyeWater, groups: [&BindGroup; 4]) {
        let Some((far, near)) = veil_pair(self, state) else { return };
        for pipeline in [far, near] {
            pass.set_pipeline(pipeline);
            for (i, g) in groups.iter().enumerate() {
                pass.set_bind_group(i as u32, *g, &[]);
            }
            pass.draw(0..3, 0..1);
        }
    }

    pub fn draw_underside(&self, pass: &mut RenderPass<'_>, groups: [&BindGroup; 3], vertices: &Buffer, indices: &Buffer, count: u32) {
        pass.set_pipeline(&self.underside);
        for (i, g) in groups.iter().enumerate() {
            pass.set_bind_group(i as u32, *g, &[]);
        }
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..count, 0, 0..1);
    }

    pub fn draw_film(&self, pass: &mut RenderPass<'_>, groups: [&BindGroup; 4]) {
        pass.set_pipeline(&self.film);
        for (i, g) in groups.iter().enumerate() {
            pass.set_bind_group(i as u32, *g, &[]);
        }
        pass.draw(0..3, 0..1);
    }
}

pub struct UnderwaterGpu {
    pub pipes: UnderwaterPipelines,
    pub buffer: Buffer,
    pub group: BindGroup,
}

impl UnderwaterGpu {
    pub fn new(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout, probe_layout: &BindGroupLayout, samples: u32) -> Self {
        let pipes = UnderwaterPipelines::new(device, format, camera_layout, probe_layout, samples);
        let (buffer, group) = pipes.under_buffer(device);
        Self { pipes, buffer, group }
    }
}

pub fn shader_sources() -> Vec<(&'static str, String)> {
    vec![
        ("underwater_veil", veil_wgsl(false, true)),
        ("underwater_veil_waterline", veil_wgsl(true, true)),
        ("underwater_veil_one_alpha", veil_wgsl(false, false)),
        ("underwater_underside", underside_wgsl()),
    ]
}

#[cfg(test)]
mod tests;
