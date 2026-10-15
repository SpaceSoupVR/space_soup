use glam::Vec3;

pub const PROBE_ROUGHNESS_MIPS: f32 = 7.0;

const SAMPLES: u32 = 64;

pub struct CubeLevel {
    pub res: u32,
    pub texels: Vec<[f32; 4]>,
}

impl CubeLevel {
    fn at(&self, face: usize, x: u32, y: u32) -> [f32; 4] {
        self.texels[(face * (self.res * self.res) as usize) + (y * self.res + x) as usize]
    }
}

pub fn texel_direction(face: usize, u: f32, v: f32) -> Vec3 {
    let a = 2.0 * u - 1.0;
    let b = 1.0 - 2.0 * v;
    let d = match face {
        0 => Vec3::new(1.0, b, -a),
        1 => Vec3::new(-1.0, b, a),
        2 => Vec3::new(a, 1.0, -b),
        3 => Vec3::new(a, -1.0, b),
        4 => Vec3::new(a, b, 1.0),
        _ => Vec3::new(-a, b, -1.0),
    };
    d.normalize()
}

pub fn direction_to_face(d: Vec3) -> (usize, f32, f32) {
    let (ax, ay, az) = (d.x.abs(), d.y.abs(), d.z.abs());
    let (face, a, b) = if ax >= ay && ax >= az {
        if d.x > 0.0 {
            (0, -d.z / ax, d.y / ax)
        } else {
            (1, d.z / ax, d.y / ax)
        }
    } else if ay >= az {
        if d.y > 0.0 {
            (2, d.x / ay, -d.z / ay)
        } else {
            (3, d.x / ay, d.z / ay)
        }
    } else if d.z > 0.0 {
        (4, d.x / az, d.y / az)
    } else {
        (5, -d.x / az, d.y / az)
    };
    (face, 0.5 * (a + 1.0), 0.5 * (1.0 - b))
}

fn sample_level(level: &CubeLevel, d: Vec3) -> [f32; 4] {
    let (face, u, v) = direction_to_face(d);
    let r = level.res;
    let px = (u * r as f32 - 0.5).clamp(0.0, (r - 1) as f32);
    let py = (v * r as f32 - 0.5).clamp(0.0, (r - 1) as f32);
    let (x0, y0) = (px.floor() as u32, py.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(r - 1), (y0 + 1).min(r - 1));
    let (fx, fy) = (px - x0 as f32, py - y0 as f32);
    let (t00, t10, t01, t11) =
        (level.at(face, x0, y0), level.at(face, x1, y0), level.at(face, x0, y1), level.at(face, x1, y1));
    let mut out = [0.0; 4];
    for c in 0..4 {
        let top = t00[c] + (t10[c] - t00[c]) * fx;
        let bottom = t01[c] + (t11[c] - t01[c]) * fx;
        out[c] = top + (bottom - top) * fy;
    }
    out
}

fn sample_pyramid(pyramid: &[CubeLevel], d: Vec3, lod: f32) -> [f32; 4] {
    let top = (pyramid.len() - 1) as f32;
    let lod = lod.clamp(0.0, top);
    let l0 = lod.floor() as usize;
    let l1 = (l0 + 1).min(pyramid.len() - 1);
    let f = lod - l0 as f32;
    let a = sample_level(&pyramid[l0], d);
    if f <= 0.0 || l1 == l0 {
        return a;
    }
    let b = sample_level(&pyramid[l1], d);
    let mut out = [0.0; 4];
    for c in 0..4 {
        out[c] = a[c] + (b[c] - a[c]) * f;
    }
    out
}

pub fn box_pyramid(base: CubeLevel) -> Vec<CubeLevel> {
    let mut levels = vec![base];
    while levels.last().unwrap().res > 1 {
        let src = levels.last().unwrap();
        let half = src.res / 2;
        let mut texels = Vec::with_capacity((half * half * 6) as usize);
        for face in 0..6 {
            for y in 0..half {
                for x in 0..half {
                    let (x0, y0) = (x * 2, y * 2);
                    let (x1, y1) = ((x0 + 1).min(src.res - 1), (y0 + 1).min(src.res - 1));
                    let mut t = [0.0; 4];
                    for (xx, yy) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
                        let s = src.at(face, xx, yy);
                        for c in 0..4 {
                            t[c] += 0.25 * s[c];
                        }
                    }
                    texels.push(t);
                }
            }
        }
        levels.push(CubeLevel { res: half, texels });
    }
    levels
}

struct LobeSample {
    dir: Vec3,
    weight: f32,
    lod: f32,
}

fn radical_inverse(bits: u32) -> f32 {
    bits.reverse_bits() as f32 * 2.328_306_4e-10
}

fn lobe_samples(roughness: f32, base_res: u32) -> Vec<LobeSample> {

    let alpha = (roughness * roughness).max(1e-4);
    let a2 = alpha * alpha;
    let texel_solid_angle = 4.0 * std::f32::consts::PI / (6.0 * (base_res * base_res) as f32);
    let mut out = Vec::with_capacity(SAMPLES as usize);
    for i in 0..SAMPLES {
        let xi1 = (i as f32 + 0.5) / SAMPLES as f32;
        let xi2 = radical_inverse(i);
        let phi = 2.0 * std::f32::consts::PI * xi1;
        let cos_h = ((1.0 - xi2) / (1.0 + (a2 - 1.0) * xi2)).sqrt();
        let sin_h = (1.0 - cos_h * cos_h).max(0.0).sqrt();
        let h = Vec3::new(sin_h * phi.cos(), sin_h * phi.sin(), cos_h);

        let l = 2.0 * cos_h * h - Vec3::Z;
        let n_dot_l = l.z;
        if n_dot_l <= 0.0 {
            continue;
        }

        let denom = cos_h * cos_h * (a2 - 1.0) + 1.0;
        let d = a2 / (std::f32::consts::PI * denom * denom);
        let pdf = d * 0.25;
        let sample_solid_angle = 1.0 / (SAMPLES as f32 * pdf + 1e-6);
        let lod = (0.5 * (sample_solid_angle / texel_solid_angle).log2() + 1.0).max(0.0);
        out.push(LobeSample { dir: l, weight: n_dot_l, lod });
    }
    out
}

fn filter_direction(pyramid: &[CubeLevel], lobe: &[LobeSample], n: Vec3) -> [f32; 4] {
    let up = if n.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
    let t = up.cross(n).normalize();
    let b = n.cross(t);
    let mut sum = [0.0f32; 4];
    let mut total = 0.0;
    for s in lobe {
        let l = t * s.dir.x + b * s.dir.y + n * s.dir.z;
        let v = sample_pyramid(pyramid, l, s.lod);
        for c in 0..4 {
            sum[c] += v[c] * s.weight;
        }
        total += s.weight;
    }
    if total > 0.0 {
        for c in &mut sum {
            *c /= total;
        }
    }
    sum
}

pub fn prefiltered_chain(base: CubeLevel) -> Vec<CubeLevel> {
    let base_res = base.res;
    let pyramid = box_pyramid(base);
    let mut out = Vec::with_capacity(pyramid.len());
    out.push(CubeLevel { res: base_res, texels: pyramid[0].texels.clone() });
    for level in 1..pyramid.len() {
        let res = pyramid[level].res;
        let roughness = (level as f32 / PROBE_ROUGHNESS_MIPS).min(1.0);
        let lobe = lobe_samples(roughness, base_res);
        let face_texels = |face: usize| -> Vec<[f32; 4]> {
            let mut v = Vec::with_capacity((res * res) as usize);
            for y in 0..res {
                for x in 0..res {
                    let n = texel_direction(
                        face,
                        (x as f32 + 0.5) / res as f32,
                        (y as f32 + 0.5) / res as f32,
                    );
                    v.push(filter_direction(&pyramid, &lobe, n));
                }
            }
            v
        };
        #[cfg(not(target_arch = "wasm32"))]
        let faces: Vec<Vec<[f32; 4]>> = std::thread::scope(|scope| {
            let handles: Vec<_> =
                (0..6).map(|face| scope.spawn(move || face_texels(face))).collect();
            handles.into_iter().map(|h| h.join().expect("probe prefilter thread")).collect()
        });
        #[cfg(target_arch = "wasm32")]
        let faces: Vec<Vec<[f32; 4]>> = (0..6).map(face_texels).collect();
        out.push(CubeLevel { res, texels: faces.concat() });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(res: u32, f: impl Fn(Vec3) -> f32) -> CubeLevel {
        let mut texels = Vec::with_capacity((res * res * 6) as usize);
        for face in 0..6 {
            for y in 0..res {
                for x in 0..res {
                    let d = texel_direction(
                        face,
                        (x as f32 + 0.5) / res as f32,
                        (y as f32 + 0.5) / res as f32,
                    );
                    let v = f(d);
                    texels.push([v, v, v, 1.0]);
                }
            }
        }
        CubeLevel { res, texels }
    }

    fn spot(res: u32, centre: Vec3, radius_deg: f32) -> CubeLevel {
        let c = centre.normalize();
        let cos_r = radius_deg.to_radians().cos();
        cube(res, move |d| if d.dot(c) >= cos_r { 50.0 } else { 0.0 })
    }

    fn rotated(v: Vec3, axis: Vec3, deg: f32) -> Vec3 {
        glam::Quat::from_axis_angle(axis.normalize(), deg.to_radians()) * v
    }

    #[test]
    fn a_texel_direction_maps_back_to_its_own_texel() {
        let res = 8u32;
        for face in 0..6 {
            for y in 0..res {
                for x in 0..res {
                    let (u, v) = ((x as f32 + 0.5) / res as f32, (y as f32 + 0.5) / res as f32);
                    let (f, u2, v2) = direction_to_face(texel_direction(face, u, v));
                    assert_eq!(f, face, "texel ({x},{y}) of face {face} came back on face {f}");
                    assert!(
                        (u2 - u).abs() < 1e-4 && (v2 - v).abs() < 1e-4,
                        "face {face} texel ({x},{y}): ({u},{v}) came back as ({u2},{v2})",
                    );
                }
            }
        }
    }

    #[test]
    fn an_even_environment_stays_even_at_every_level() {
        let chain = prefiltered_chain(cube(16, |_| 2.0));
        assert_eq!(chain.len(), 5);
        for (i, level) in chain.iter().enumerate() {
            for t in &level.texels {
                assert!((t[0] - 2.0).abs() < 1e-3, "level {i} turned 2.0 into {}", t[0]);
                assert!((t[3] - 1.0).abs() < 1e-3, "level {i} lost coverage: {}", t[3]);
            }
        }
    }

    #[test]
    fn a_bright_spot_blurs_round_not_square() {
        let pyramid = box_pyramid(spot(64, Vec3::Z, 4.0));
        let lobe = lobe_samples(4.0 / PROBE_ROUGHNESS_MIPS, 64);
        let ggx = |d: Vec3| filter_direction(&pyramid, &lobe, d)[0];

        let boxed = |d: Vec3| sample_pyramid(&pyramid, d, 3.0)[0];
        for deg in [8.0f32, 14.0] {
            let along_axis = rotated(Vec3::Z, Vec3::Y, deg);
            let along_diagonal = rotated(Vec3::Z, Vec3::new(1.0, 1.0, 0.0), deg);
            let (a, d) = (ggx(along_axis), ggx(along_diagonal));
            assert!(a > 0.0, "the lobe at {deg} degrees saw nothing of the spot");
            assert!(
                (a / d - 1.0).abs() < 0.12,
                "GGX lobe at {deg} degrees: {a} along the axis, {d} along the diagonal",
            );
        }
        let (a, d) = (
            boxed(rotated(Vec3::Z, Vec3::Y, 20.0)),
            boxed(rotated(Vec3::Z, Vec3::new(1.0, 1.0, 0.0), 20.0)),
        );
        assert!(
            (a / d.max(1e-6) - 1.0).abs() > 0.25,
            "the box chain came out round too ({a} vs {d}), so this test cannot \
             tell the filters apart",
        );
    }

    #[test]
    fn a_rougher_level_is_a_wider_lobe() {
        let pyramid = box_pyramid(spot(64, Vec3::Z, 4.0));
        let off = rotated(Vec3::Z, Vec3::Y, 25.0);
        let mut last_centre = f32::INFINITY;
        let mut last_ratio = 0.0;
        for level in [2.0f32, 4.0, 6.0] {
            let lobe = lobe_samples(level / PROBE_ROUGHNESS_MIPS, 64);
            let centre = filter_direction(&pyramid, &lobe, Vec3::Z)[0];
            let side = filter_direction(&pyramid, &lobe, off)[0];
            let ratio = side / centre;
            assert!(centre < last_centre, "level {level} peak {centre} did not dim");
            assert!(ratio > last_ratio, "level {level} did not widen: {ratio}");
            last_centre = centre;
            last_ratio = ratio;
        }
    }

    #[test]
    fn a_spot_on_a_seam_blurs_evenly_onto_both_faces() {
        let seam = Vec3::new(1.0, 0.0, 1.0).normalize();
        let pyramid = box_pyramid(spot(64, seam, 4.0));
        let lobe = lobe_samples(4.0 / PROBE_ROUGHNESS_MIPS, 64);
        for deg in [8.0f32, 14.0] {
            let toward_z = filter_direction(&pyramid, &lobe, rotated(seam, Vec3::Y, -deg))[0];
            let toward_x = filter_direction(&pyramid, &lobe, rotated(seam, Vec3::Y, deg))[0];
            assert!(
                (toward_z / toward_x - 1.0).abs() < 0.12,
                "{deg} degrees either side of the seam: {toward_z} on +Z, {toward_x} on +X",
            );
        }
    }

    #[test]
    #[ignore]
    fn prefiltering_a_128px_probe() {
        let base = spot(128, Vec3::new(0.3, 0.8, 0.5), 6.0);
        let t = std::time::Instant::now();
        let chain = prefiltered_chain(base);
        eprintln!("prefiltered {} levels of a 128px cube in {} ms", chain.len(), t.elapsed().as_millis());
        assert_eq!(chain.len(), 8);
    }

    #[test]
    fn the_chain_is_filtered_for_the_ramp_the_shader_reads() {
        let code = super::super::lights::wgsl_lights_block(0, 1);
        assert!(
            code.contains(&format!("const PROBE_ROUGHNESS_MIPS: f32 = {PROBE_ROUGHNESS_MIPS:.1};")),
            "the shader maps roughness onto the probe chain with a different \
             constant than the chain was prefiltered with",
        );
        assert!(code.contains("clamp(roughness * PROBE_ROUGHNESS_MIPS"));
    }
}
