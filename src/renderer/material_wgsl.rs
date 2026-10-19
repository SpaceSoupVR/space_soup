pub fn wgsl_biplanar_block() -> &'static str {
    r#"
struct Biplanar {
    uv_major: vec2<f32>,
    uv_minor: vec2<f32>,
    axis_major: u32,
    axis_minor: u32,

    w: f32,
}

fn biplanar_uv(axis: u32, world: vec3<f32>, r: f32) -> vec2<f32> {
    if (axis == 0u) { return world.zy / r; }
    if (axis == 1u) { return world.xz / r; }
    return world.xy / r;
}

fn biplanar_axes(world: vec3<f32>, n: vec3<f32>, repeat: f32) -> Biplanar {
    let r = max(repeat, 0.001);
    let a = abs(n);

    let ma = max(a.x, max(a.y, a.z));
    let mi = min(a.x, min(a.y, a.z));
    let me = a.x + a.y + a.z - ma - mi;

    var major: u32;
    var minor: u32;
    if (a.y == ma) {
        major = 1u;
        minor = select(2u, 0u, a.x >= a.z);
    } else if (a.x == ma) {
        major = 0u;
        minor = select(2u, 1u, a.z < a.y);
    } else {
        major = 2u;
        minor = select(0u, 1u, a.x < a.y);
    }

    var out: Biplanar;
    out.axis_major = major;
    out.axis_minor = minor;
    out.uv_major = biplanar_uv(major, world, r);
    out.uv_minor = biplanar_uv(minor, world, r);
    out.w = ma / max(ma + me, 0.001);
    return out;
}
"#
}

pub fn wgsl_whiteout_block() -> &'static str {
    r#"
fn whiteout(axis: u32, tn: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    if (axis == 0u) { return vec3<f32>(abs(tn.z) * n.x, n.y - tn.y, tn.x + n.z); }
    if (axis == 1u) { return vec3<f32>(tn.x + n.x, abs(tn.z) * n.y, n.z - tn.y); }
    return vec3<f32>(tn.x + n.x, n.y - tn.y, abs(tn.z) * n.z);
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uv(axis: u32, w: [f32; 3]) -> [f32; 2] {
        match axis {
            0 => [w[2], w[1]],
            1 => [w[0], w[2]],
            _ => [w[0], w[1]],
        }
    }
    fn whiteout(axis: u32, tn: [f32; 3], n: [f32; 3]) -> [f32; 3] {
        match axis {
            0 => [tn[2].abs() * n[0], n[1] - tn[1], tn[0] + n[2]],
            1 => [tn[0] + n[0], tn[2].abs() * n[1], n[2] - tn[1]],
            _ => [tn[0] + n[0], n[1] - tn[1], tn[2].abs() * n[2]],
        }
    }

    #[test]
    fn the_transcription_is_the_shader_text() {
        let b = wgsl_biplanar_block();
        assert!(b.contains("if (axis == 0u) { return world.zy / r; }"));
        assert!(b.contains("if (axis == 1u) { return world.xz / r; }"));
        assert!(b.contains("return world.xy / r;"));
        let w = wgsl_whiteout_block();
        assert!(w.contains("if (axis == 0u) { return vec3<f32>(abs(tn.z) * n.x, n.y - tn.y, tn.x + n.z); }"));
        assert!(w.contains("if (axis == 1u) { return vec3<f32>(tn.x + n.x, abs(tn.z) * n.y, n.z - tn.y); }"));
        assert!(w.contains("return vec3<f32>(tn.x + n.x, n.y - tn.y, abs(tn.z) * n.z);"));
    }

    #[test]
    fn green_leans_toward_the_top_of_the_picture_on_every_axis() {
        for axis in 0..3u32 {
            let mut n = [0.0f32; 3];
            n[axis as usize] = 1.0;
            let lean = |tn: [f32; 3]| {
                let p = whiteout(axis, tn, n);
                let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                [p[0] / len - n[0], p[1] / len - n[1], p[2] / len - n[2]]
            };
            let g = lean([0.0, 0.6, 0.8]);
            let r = lean([0.6, 0.0, 0.8]);

            let dv = |d: [f32; 3]| uv(axis, d)[1];
            let du = |d: [f32; 3]| uv(axis, d)[0];
            assert!(dv(g) < -0.1, "axis {axis}: green must lean toward decreasing v (the picture's top), leaned {g:?}");
            assert!(du(g).abs() < 1e-6, "axis {axis}: green must not move u: {g:?}");
            assert!(du(r) > 0.1, "axis {axis}: red must lean toward increasing u (the picture's right), leaned {r:?}");
            assert!(dv(r).abs() < 1e-6, "axis {axis}: red must not move v: {r:?}");
        }
    }
}
