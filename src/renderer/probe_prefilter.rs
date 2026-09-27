//! The mip chain of a reflection-probe cube, prefiltered with the GGX lobe.
//!
//! WHY NOT A BOX FILTER. The chain used to be built by averaging 2x2 texels,
//! face by face. That makes level N a picture of the room at 1/2^N resolution
//! -- which is not what a surface of the roughness that picks level N reflects.
//! A bright feature in the photograph (a spot-lit pillar, a lamp's floor pool)
//! stays a small bright SQUARE at every level instead of spreading into the
//! broad round lobe a rough wall actually shows, and each face is filtered
//! alone so the blur stops dead at the cube seams. On the headset that was the
//! soft light rectangles on the hall walls and ceiling that come and go with
//! position: the lighting-sources view traced them to the probe, and they
//! survived both the one-probe-per-room fix and the 128 px faces (2026-09-11).
//!
//! WHAT THIS DOES INSTEAD is the prefiltered environment map of Karis, "Real
//! Shading in Unreal Engine 4" (2013) and Lagarde & de Rousiers, "Moving
//! Frostbite to PBR" (2014): every texel of level N is the GGX-weighted
//! integral of the environment around its direction, for the roughness the
//! shader maps onto level N, under the usual N = V = R assumption. Samples are
//! importance-sampled from the lobe and read from a box pyramid at a level
//! matched to each sample's solid angle -- filtered importance sampling,
//! Krivanek & Colbert, GPU Gems 3 ch. 20 -- so a few dozen samples give a
//! smooth lobe with no sparkle. Directions are resolved on the whole sphere,
//! so the lobe crosses a cube seam as if it were not there.
//!
//! It runs ONCE, when a level's probes load, on the CPU. It costs nothing per
//! frame: the shader reads exactly the same texture it did before.

use glam::Vec3;

/// Roughness spread over the chain: level `n` is prefiltered for roughness
/// `n / PROBE_ROUGHNESS_MIPS`.
///
/// MUST MATCH the WGSL constant of the same name in `lights.rs`, which picks
/// the level as `roughness * PROBE_ROUGHNESS_MIPS`. If the two disagree, every
/// surface reflects a lobe for a different roughness than its own.
/// `the_chain_is_filtered_for_the_ramp_the_shader_reads` pins them together.
pub const PROBE_ROUGHNESS_MIPS: f32 = 7.0;

/// GGX samples per output texel.
///
/// Filtered importance sampling is what makes this few enough: each sample
/// reads an area of the source the size of its own share of the lobe, so the
/// samples overlap rather than point-sampling bright texels between them.
const SAMPLES: u32 = 64;

/// One cube level: six `res`x`res` faces of linear RGBA, in cube order.
pub struct CubeLevel {
    pub res: u32,
    pub texels: Vec<[f32; 4]>,
}

impl CubeLevel {
    fn at(&self, face: usize, x: u32, y: u32) -> [f32; 4] {
        self.texels[(face * (self.res * self.res) as usize) + (y * self.res + x) as usize]
    }
}

/// The direction a cube texel looks along.
///
/// A COPY of `space_soup_engine::reflection_probe::face_direction`, which the
/// baker photographs each face with. `space_soup` ships to crates.io and cannot
/// depend on the engine, so the bake crate's tests compare the two texel for
/// texel (`the_renderer_reads_faces_the_way_the_baker_writes_them`).
///
/// Layer 0 is +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z; `u`, `v` run 0..1 across the
/// face with image rows downward.
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

/// The face and 0..1 face coordinates a direction lands on. The inverse of
/// [`texel_direction`].
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

/// Bilinear read of one level along a direction.
///
/// Clamped at the face edge rather than wrapped onto the neighbour face. The
/// error is half a texel at a seam, and every read here is already one of
/// dozens overlapping across it, so it does not survive into the result.
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

/// Trilinear read of a box pyramid at a fractional level.
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

/// A plain 2x2 box pyramid, all the way to one texel a face.
///
/// Not what the shader samples -- the SOURCE the GGX samples read from, so a
/// sample covering a wide solid angle reads an average over it instead of one
/// texel. Averages in linear light, which is correct because probe texels are
/// linear radiance.
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

/// One precomputed GGX sample: its direction in the lobe's tangent space
/// (normal along +Z), its cosine weight, and the pyramid level to read it at.
struct LobeSample {
    dir: Vec3,
    weight: f32,
    lod: f32,
}

/// Van der Corput sequence: the second coordinate of a Hammersley point set.
fn radical_inverse(bits: u32) -> f32 {
    bits.reverse_bits() as f32 * 2.328_306_4e-10
}

/// The GGX lobe for one roughness, as importance samples.
///
/// `base_res` is the resolution of pyramid level 0, which sets the solid angle
/// one source texel covers.
fn lobe_samples(roughness: f32, base_res: u32) -> Vec<LobeSample> {
    // Perceptual roughness squared, the same alpha the direct-light BRDF uses.
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
        // With N = V, the reflected direction is L = 2(N.H)H - N.
        let l = 2.0 * cos_h * h - Vec3::Z;
        let n_dot_l = l.z;
        if n_dot_l <= 0.0 {
            continue;
        }
        // pdf(L) = D(H) (N.H) / (4 V.H), and N.H == V.H under N = V.
        let denom = cos_h * cos_h * (a2 - 1.0) + 1.0;
        let d = a2 / (std::f32::consts::PI * denom * denom);
        let pdf = d * 0.25;
        let sample_solid_angle = 1.0 / (SAMPLES as f32 * pdf + 1e-6);
        let lod = (0.5 * (sample_solid_angle / texel_solid_angle).log2() + 1.0).max(0.0);
        out.push(LobeSample { dir: l, weight: n_dot_l, lod });
    }
    out
}

/// The GGX-filtered radiance around `n`, for a precomputed lobe.
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

/// The whole chain, largest first: level 0 is the photograph itself, every
/// later level the GGX lobe for its roughness.
///
/// Faces are filtered on separate threads where the platform has them; the
/// work is embarrassingly parallel and it happens while a level loads.
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

    /// A small bright disc around `centre`, `radius_deg` wide, on black.
    fn spot(res: u32, centre: Vec3, radius_deg: f32) -> CubeLevel {
        let c = centre.normalize();
        let cos_r = radius_deg.to_radians().cos();
        cube(res, move |d| if d.dot(c) >= cos_r { 50.0 } else { 0.0 })
    }

    /// Rotate `v` by `deg` about `axis`.
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

    /// Normalised weights: an evenly lit room reflects the same light at every
    /// roughness. A level that came out darker would dim every rough surface.
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

    /// THE SQUARES. A bright spot must blur ROUND.
    ///
    /// Read the rough lobe at the same angle from the spot along a face axis
    /// and along the face diagonal: a lobe is rotationally symmetric, so the
    /// two must agree. The box chain it replaced does not -- which is asserted
    /// too, so this test is known to tell the two apart.
    #[test]
    fn a_bright_spot_blurs_round_not_square() {
        let pyramid = box_pyramid(spot(64, Vec3::Z, 4.0));
        let lobe = lobe_samples(4.0 / PROBE_ROUGHNESS_MIPS, 64);
        let ggx = |d: Vec3| filter_direction(&pyramid, &lobe, d)[0];
        // Level 3 at 20 degrees: the reads straddle a texel boundary, where a
        // box level shows its squareness. (At level 4 and 14 degrees both land
        // inside the same four texels and agree by construction.)
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

    /// A rougher level spreads the same light wider and dimmer.
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

    /// A spot ON a cube seam blurs onto both faces alike. Filtering each face
    /// alone stopped the blur dead at the edge.
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

    /// What one 128 px probe costs to prefilter, for judging load time.
    ///
    /// Ignored because a timing is not a pass/fail fact on a shared machine.
    /// `cargo test --release --lib prefiltering_a_128px_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn prefiltering_a_128px_probe() {
        let base = spot(128, Vec3::new(0.3, 0.8, 0.5), 6.0);
        let t = std::time::Instant::now();
        let chain = prefiltered_chain(base);
        eprintln!("prefiltered {} levels of a 128px cube in {} ms", chain.len(), t.elapsed().as_millis());
        assert_eq!(chain.len(), 8);
    }

    /// The level the shader picks for a roughness is the level filtered for it.
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
