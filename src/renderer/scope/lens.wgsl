// The ocular lens of a magnified optic. See lens.rs.
//
// Samples the scope render by ray angle (collimated light carries the image in
// angle), cuts the scope shadow as a crescent, and draws the reticle in its
// focal plane. Never discards: the texture read and fwidth stay in uniform flow.

var<private> view_slot: i32 = 0;
struct Camera { view_proj: array<mat4x4<f32>, 2> }
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var scope_tex: texture_2d<f32>;
@group(1) @binding(1) var scope_samp: sampler;

struct Lens {
    eyes: array<vec4<f32>, 2>,
    offsets: array<vec4<f32>, 2>,
    frame: array<vec4<f32>, 4>,
    optic: vec4<f32>,
    reticle: vec4<f32>,
    extra: vec4<f32>,
};
@group(2) @binding(0) var<uniform> lens: Lens;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.clip = camera.view_proj[view_slot] * vec4<f32>(pos, 1.0);
    out.world = pos;
    return out;
}

// Coverage of a line or disc edge `d` away, at least ~1.2 px wide so thin
// reticle lines never break up into dashes.
fn cover(d: f32, half_width: f32, aa: f32) -> f32 {
    let hw = max(half_width, aa * 0.6);
    return clamp((hw - d) / max(aa, 1e-6) + 0.5, 0.0, 1.0);
}

fn segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let h = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
    return length(p - a - ab * h);
}

// Fine crosshair out to `reach` mrad from the centre.
fn crosshair(q: vec2<f32>, hw: f32, aa: f32, reach: f32) -> f32 {
    let h = select(0.0, cover(abs(q.y), hw, aa), abs(q.x) < reach);
    let v = select(0.0, cover(abs(q.x), hw, aa), abs(q.y) < reach);
    return max(h, v);
}

// Heavy posts from `start` mrad outward, as on a duplex.
fn posts(q: vec2<f32>, hw: f32, aa: f32, start: f32) -> f32 {
    let h = select(0.0, cover(abs(q.y), hw, aa), abs(q.x) > start);
    let v = select(0.0, cover(abs(q.x), hw, aa), abs(q.y) > start);
    return max(h, v);
}

// Holdover ticks below the centre: one every 2 mrad from `first`, narrowing.
fn drop_ticks(q: vec2<f32>, aa: f32, first: f32, half_len: f32) -> f32 {
    var c = 0.0;
    for (var i = 0; i < 4; i++) {
        let y = -(first + 2.0 * f32(i));
        let len = half_len * (1.0 - 0.15 * f32(i));
        c = max(c, cover(segment(q, vec2<f32>(-len, y), vec2<f32>(len, y)), 0.06, aa));
    }
    return c;
}

fn mil_dots(q: vec2<f32>, aa: f32) -> f32 {
    let along_x = vec2<f32>(clamp(round(q.x), -5.0, 5.0), 0.0);
    let along_y = vec2<f32>(0.0, clamp(round(q.y), -5.0, 5.0));
    let dx = select(1e3, length(q - along_x), abs(along_x.x) >= 1.0);
    let dy = select(1e3, length(q - along_y), abs(along_y.y) >= 1.0);
    return cover(min(dx, dy), 0.125, aa);
}

// x: line work, y: the part a lit reticle always lights (dots, rings).
// Units are the reticle's mrad, y up. Style order is ReticleStyle::ALL.
fn reticle(style: u32, q: vec2<f32>, aa: f32) -> vec2<f32> {
    let r = length(q);
    switch style {
        case 1u: { return vec2<f32>(0.0, cover(r, 0.29, aa)); }
        case 2u: { return vec2<f32>(0.0, max(cover(r, 0.15, aa), cover(abs(r - 9.45), 0.3, aa))); }
        case 3u: { return vec2<f32>(crosshair(q, 0.05, aa, 1e4), 0.0); }
        case 4u: { return vec2<f32>(max(crosshair(q, 0.05, aa, 10.0), posts(q, 0.4, aa, 10.0)), 0.0); }
        case 5u: { return vec2<f32>(max(crosshair(q, 0.05, aa, 10.0), posts(q, 0.4, aa, 10.0)), cover(r, 0.2, aa)); }
        case 6u: { return vec2<f32>(max(max(crosshair(q, 0.04, aa, 5.0), posts(q, 0.3, aa, 5.0)), mil_dots(q, aa)), 0.0); }
        case 7u: {
            let arms = min(segment(q, vec2<f32>(0.0), vec2<f32>(-1.5, -2.0)), segment(q, vec2<f32>(0.0), vec2<f32>(1.5, -2.0)));
            let stadia = cover(segment(q, vec2<f32>(0.0, -2.5), vec2<f32>(0.0, -9.0)), 0.06, aa);
            return vec2<f32>(max(stadia, drop_ticks(q, aa, 3.0, 1.6)), cover(arms, 0.12, aa));
        }
        case 8u: { return vec2<f32>(max(crosshair(q, 0.05, aa, 1e4), drop_ticks(q, aa, 2.0, 0.8)), 0.0); }
        default: { return vec2<f32>(0.0); }
    }
}

// Scope shadow: misalignment eats a crescent from the side away from where
// the eye moved; it never blurs and never changes power.
fn shadow(occupancy: f32, moved: vec2<f32>, lens_xy: vec2<f32>) -> f32 {
    let occ = clamp(occupancy, 0.0, 1.0);
    let len = length(moved);
    let cut = 1.0 - 2.0 * occ;
    let crescent = clamp((dot(lens_xy, moved / max(len, 1e-6)) - cut) / 0.35 + 0.5, 0.0, 1.0);
    return select(select(crescent, occ, len <= 1e-6), 1.0, occ >= 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let e = min(u32(view_slot) + u32(lens.extra.z + 0.5), 1u);
    let right = lens.frame[0].xyz;
    let up = lens.frame[1].xyz;
    let fwd = lens.frame[2].xyz;
    let radius = max(lens.frame[3].w, 1e-4);

    // Disc mask on the lens plane, feathered over a couple of pixels.
    let rel = in.world - lens.frame[3].xyz;
    let lens_xy = vec2<f32>(dot(rel, right), dot(rel, up)) / radius;
    let r = length(lens_xy);
    let edge = clamp((1.0 - r) / max(lens.optic.z * fwidth(r), 1e-4), 0.0, 1.0);

    // The ray's angle off the axis, demagnified into a world angle.
    let ray = normalize(in.world - lens.eyes[e].xyz);
    let z = max(dot(ray, fwd), 1e-5);
    let mag = max(lens.optic.x, 1e-3);
    let world = atan(vec2<f32>(dot(ray, right), dot(ray, up)) / z) / mag;
    let st = tan(world) / max(lens.optic.y, 1e-5);
    let field = select(0.0, 1.0, abs(st.x) <= 1.0 && abs(st.y) <= 1.0);
    let picture = textureSample(scope_tex, scope_samp, vec2<f32>(st.x * 0.5 + 0.5, 0.5 - st.y * 0.5)).rgb;

    let shade = shadow(lens.eyes[e].w, lens.offsets[e].xy, lens_xy);
    let is_window = lens.optic.w > 0.5;
    let darkens = lens.extra.y > 0.5;
    // A scope shows its picture, black past the field stop; a window shows the
    // eye's own view and, for a tube, darkens it as the shadow sweeps in.
    let base_alpha = select(1.0, select(0.0, 1.0 - shade, darkens), is_window);
    let base = select(picture * field * shade, vec3<f32>(0.0), is_window);

    let q = world * 1000.0 * lens.extra.x;
    let aa = max(fwidth(q.x), fwidth(q.y));
    let marks = reticle(u32(lens.reticle.w + 0.5), q, aa) * select(lens.eyes[e].w, shade, darkens) * field;
    let emission = min(lens.reticle.rgb, vec3<f32>(1.0));
    let is_lit = max(emission.r, max(emission.g, emission.b)) > 0.0;
    let duplex_dot = u32(lens.reticle.w + 0.5) == 5u;
    let etched = select(max(marks.x, marks.y), select(0.0, marks.x, duplex_dot), is_lit);
    let lit = select(0.0, select(max(marks.x, marks.y), marks.y, duplex_dot), is_lit);

    var rgb = base * base_alpha * (1.0 - etched);
    var alpha = etched + base_alpha * (1.0 - etched);
    rgb = emission * lit + rgb * (1.0 - lit);
    alpha = lit + alpha * (1.0 - lit);
    return vec4<f32>(rgb, alpha) * edge;
}
