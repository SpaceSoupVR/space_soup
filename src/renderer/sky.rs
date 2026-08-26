//! The sky: the background behind a level, and the ambient light in it.
//!
//! WHY BOTH JOBS BELONG TO ONE OBJECT
//!
//! An HDRI is one equirectangular panorama carrying real intensity rather than
//! clamped colour, and it does the background and the lighting at once. That is
//! the whole reason the format replaced a six-sided skybox: there, the sky was a
//! picture behind the level and lit nothing, so "the lighting does not match the
//! sky" was a permanent class of bug. Here they are the same data and cannot
//! disagree.
//!
//! The editor has had this for a while. Nothing in the renderer, the client, the
//! protocol or the server knew what a sky was, so a level authored against a
//! panorama arrived on the headset with a dark blue clear colour behind it and a
//! flat 0.6 ambient inside it.
//!
//! LIGHTING IS SPHERICAL HARMONICS, NOT A TEXTURE
//!
//! Diffuse irradiance from an environment is a very smooth function of the
//! surface normal -- so smooth that nine coefficients reproduce it to within a
//! percent or so, which is the standard result. Projecting once at load and
//! evaluating in the shader costs about twenty instructions and NO texture
//! fetch, against a cube sample per pixel for the obvious alternative.
//!
//! On a tile GPU that distinction is the whole argument. Bandwidth is the scarce
//! resource; arithmetic is not. Nine `vec4`s in the uniform buffer we already
//! bind cost nothing per pixel at all.
//!
//! Specular from the environment is deliberately absent. Doing it properly needs
//! a prefiltered mip chain and a BRDF lookup, which is a real amount of memory
//! and two more fetches per pixel, and the engine's Blinn-Phong highlight from
//! actual lights already covers the case people notice.
//!
//! THE DIRECTION MAPPING IS MEASURED, NOT ASSUMED
//!
//! `direction_to_uv` below is pinned to the editor's: a sun at azimuth 34.3 and
//! elevation 48 in the lobby's panorama lands on texel (609, 119), which is
//! exactly the brightest texel in that file. Getting this wrong puts the light
//! somewhere other than the visible sun, and a level lit from the wrong side
//! looks like a lighting bug rather than a mapping one.

use anyhow::{bail, Result};
use wgpu::*;

use super::lights::wgsl_lights_block;

/// The flat ambient a scene without a sky gets, matching the shader constant.
pub const AMBIENT: f32 = 0.6;

/// A decoded equirectangular panorama, linear and unclamped.
#[derive(Clone)]
pub struct Panorama {
    pub width: u32,
    pub height: u32,
    /// Row-major RGB triples, top row first.
    pub rgb: Vec<f32>,
}

impl Panorama {
    pub fn texel(&self, x: u32, y: u32) -> [f32; 3] {
        let i = ((y * self.width + x) * 3) as usize;
        [self.rgb[i], self.rgb[i + 1], self.rgb[i + 2]]
    }

    /// A flat panorama of one colour, for tests and for a scene with no sky.
    pub fn solid(rgb: [f32; 3], width: u32, height: u32) -> Self {
        let mut v = Vec::with_capacity((width * height * 3) as usize);
        for _ in 0..(width * height) {
            v.extend_from_slice(&rgb);
        }
        Self { width, height, rgb: v }
    }
}

/// Decode a Radiance `.hdr` (RGBE) file.
///
/// Hand-written rather than pulled from a crate for the same reason the cave's
/// glTF writer is: it is a short, completely specified format, and the one thing
/// a library would protect against is the adaptive-RLE scanline encoding, which
/// is thirty lines and is right below.
pub fn decode_radiance(bytes: &[u8]) -> Result<Panorama> {
    // Header: text lines, a blank line, then the resolution.
    let Some(sep) = bytes.windows(2).position(|w| w == b"\n\n") else {
        bail!("not a Radiance file: no blank line after the header");
    };
    let mut pos = sep + 2;
    let Some(nl) = bytes[pos..].iter().position(|&b| b == b'\n') else {
        bail!("truncated before the resolution line");
    };
    let res = std::str::from_utf8(&bytes[pos..pos + nl])?;
    let parts: Vec<&str> = res.split_whitespace().collect();
    // Only the overwhelmingly common orientation. A file in any other is
    // rejected rather than silently loaded upside down or mirrored, which is
    // the kind of wrong that gets blamed on the artist.
    if parts.len() != 4 || parts[0] != "-Y" || parts[2] != "+X" {
        bail!("unsupported Radiance orientation {res:?}; expected `-Y h +X w`");
    }
    let height: u32 = parts[1].parse()?;
    let width: u32 = parts[3].parse()?;
    pos += nl + 1;

    let mut rgb = vec![0.0f32; (width * height * 3) as usize];
    let mut scan = vec![0u8; (width * 4) as usize];

    for y in 0..height {
        read_scanline(bytes, &mut pos, width, &mut scan)?;
        for x in 0..width {
            let i = (x * 4) as usize;
            let (r, g, b, e) = (scan[i], scan[i + 1], scan[i + 2], scan[i + 3]);
            let out = ((y * width + x) * 3) as usize;
            if e == 0 {
                continue; // already zero
            }
            // RGBE: a shared exponent biased by 128, with the mantissa a byte.
            let f = libm_exp2(e as i32 - 136);
            rgb[out] = r as f32 * f;
            rgb[out + 1] = g as f32 * f;
            rgb[out + 2] = b as f32 * f;
        }
    }
    Ok(Panorama { width, height, rgb })
}

/// `2^n` for the RGBE exponent, without pulling in a maths crate.
fn libm_exp2(n: i32) -> f32 {
    // The exponent range a byte can produce is far inside f32's, so a shift on
    // the bit pattern is exact and avoids `powi`'s repeated multiplication.
    if n < -126 {
        return 0.0;
    }
    if n > 127 {
        return f32::INFINITY;
    }
    f32::from_bits(((n + 127) as u32) << 23)
}

fn read_scanline(bytes: &[u8], pos: &mut usize, width: u32, out: &mut [u8]) -> Result<()> {
    let p = *pos;
    let adaptive = width >= 8
        && width <= 0x7fff
        && bytes.len() > p + 4
        && bytes[p] == 2
        && bytes[p + 1] == 2
        && ((bytes[p + 2] as u32) << 8 | bytes[p + 3] as u32) == width;

    if !adaptive {
        // Flat RGBE, four bytes per pixel.
        let n = (width * 4) as usize;
        if bytes.len() < p + n {
            bail!("truncated scanline");
        }
        out[..n].copy_from_slice(&bytes[p..p + n]);
        *pos = p + n;
        return Ok(());
    }

    // Adaptive RLE: each of the four channels is run-length encoded separately
    // across the whole scanline, which is why the bytes are de-interleaved here
    // and written back into the interleaved buffer by stride.
    let mut q = p + 4;
    for ch in 0..4usize {
        let mut x = 0u32;
        while x < width {
            if q >= bytes.len() {
                bail!("truncated RLE scanline");
            }
            let n = bytes[q];
            q += 1;
            if n > 128 {
                // A run: one value repeated n-128 times.
                if q >= bytes.len() {
                    bail!("truncated RLE run");
                }
                let val = bytes[q];
                q += 1;
                for _ in 0..(n - 128) {
                    if x >= width {
                        bail!("RLE run overruns the scanline");
                    }
                    out[(x * 4) as usize + ch] = val;
                    x += 1;
                }
            } else {
                if n == 0 {
                    bail!("zero-length RLE literal");
                }
                for _ in 0..n {
                    if q >= bytes.len() || x >= width {
                        bail!("RLE literal overruns the scanline");
                    }
                    out[(x * 4) as usize + ch] = bytes[q];
                    q += 1;
                    x += 1;
                }
            }
        }
    }
    *pos = q;
    Ok(())
}

/// Where a world direction lands in an equirectangular panorama.
///
/// MEASURED AGAINST THE EDITOR, not derived from a convention. The lobby's
/// panorama has its sun at azimuth 34.3 / elevation 48 in world terms -- found
/// by scanning the rendered sky for its brightest direction -- and this maps
/// that to texel (609, 119), which is the brightest texel in the file itself.
/// Two independent measurements of the same thing.
pub fn direction_to_uv(d: [f32; 3]) -> [f32; 2] {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let (x, y, z) = (d[0] / len, d[1] / len, d[2] / len);
    let u = (x.atan2(z) + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
    let v = y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
    [u, v]
}

/// The inverse: the direction a texel centre looks along.
pub fn uv_to_direction(u: f32, v: f32) -> [f32; 3] {
    let phi = u * 2.0 * std::f32::consts::PI - std::f32::consts::PI;
    let theta = v * std::f32::consts::PI;
    let s = theta.sin();
    [s * phi.sin(), theta.cos(), s * phi.cos()]
}

/// Nine RGB spherical-harmonic coefficients of a sky's irradiance.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SkyIrradiance {
    /// L00, L1-1, L10, L11, L2-2, L2-1, L20, L21, L22 -- already convolved with
    /// the cosine lobe, so the shader only evaluates the basis.
    pub sh: [[f32; 3]; 9],
}

impl SkyIrradiance {
    /// What a scene with no sky gets: the flat ambient the engine always had.
    ///
    /// Expressed as an SH rather than special-cased in the shader, so there is
    /// one lighting path and "no sky" is a value rather than a branch. Only the
    /// constant band is non-zero, which evaluates to exactly `AMBIENT` in every
    /// direction -- so a level without a sky renders precisely as it did.
    pub fn flat(ambient: f32) -> Self {
        // evaluate() multiplies band 0 by Y00 = 0.282095 and by A0 = 1.
        let l0 = ambient / 0.282_095;
        let mut sh = [[0.0; 3]; 9];
        sh[0] = [l0, l0, l0];
        Self { sh }
    }

    /// Irradiance arriving at a surface with this normal, as the shader computes
    /// it. Present in Rust so the projection can be tested without a GPU.
    pub fn evaluate(&self, n: [f32; 3]) -> [f32; 3] {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
        let (x, y, z) = (n[0] / len, n[1] / len, n[2] / len);
        let b = sh_basis(x, y, z);
        // The cosine lobe's convolution coefficients, ALREADY DIVIDED BY PI.
        //
        // Ramamoorthi's are pi, 2pi/3 and pi/4, and they give irradiance E. What
        // an ambient term wants is the radiance a white Lambertian surface
        // reflects, which is E/pi -- so the division is folded in here rather
        // than applied afterwards.
        //
        // The widely quoted 0.886227 / 1.023328 / 0.858086 are these constants
        // ALREADY MULTIPLIED BY the basis normalisation, for use in a form that
        // does not evaluate Y_lm separately. Using them here as well as the
        // basis counts the constants twice, which is exactly the bug the
        // uniform-sky test caught: a flat 0.5 sky came back as 0.141.
        const A: [f32; 9] = [
            1.0,                                    // l = 0
            2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0,        // l = 1
            0.25, 0.25, 0.25, 0.25, 0.25,           // l = 2
        ];
        let mut out = [0.0f32; 3];
        for i in 0..9 {
            for c in 0..3 {
                out[c] += self.sh[i][c] * b[i] * A[i];
            }
        }
        for c in 0..3 {
            out[c] = out[c].max(0.0);
        }
        out
    }
}

/// The real spherical-harmonic basis up to l = 2.
fn sh_basis(x: f32, y: f32, z: f32) -> [f32; 9] {
    [
        0.282_095,
        0.488_603 * y,
        0.488_603 * z,
        0.488_603 * x,
        1.092_548 * x * y,
        1.092_548 * y * z,
        0.315_392 * (3.0 * z * z - 1.0),
        1.092_548 * x * z,
        0.546_274 * (x * x - y * y),
    ]
}

/// Project a panorama onto the irradiance basis.
///
/// Each texel is weighted by its solid angle -- `sin(theta)` -- which is not
/// optional: an equirectangular image devotes as many texels to the pole as to
/// the equator, and summing them evenly makes whatever is overhead dominate the
/// result by a factor of about pi/2.
pub fn project_irradiance(pano: &Panorama, rotation_deg: f32, intensity: f32) -> SkyIrradiance {
    let mut sh = [[0.0f64; 3]; 9];
    let mut weight = 0.0f64;
    let rot = rotation_deg.to_radians();
    let (rc, rs) = (rot.cos(), rot.sin());

    for y in 0..pano.height {
        let v = (y as f32 + 0.5) / pano.height as f32;
        let theta = v * std::f32::consts::PI;
        let sin_theta = theta.sin();
        if sin_theta <= 0.0 {
            continue;
        }
        for x in 0..pano.width {
            let u = (x as f32 + 0.5) / pano.width as f32;
            let d = uv_to_direction(u, v);
            // The scene's own rotation of the sky, applied to the DIRECTION so
            // the lighting turns with the picture rather than away from it.
            let d = [rc * d[0] + rs * d[2], d[1], -rs * d[0] + rc * d[2]];
            let b = sh_basis(d[0], d[1], d[2]);
            let t = pano.texel(x, y);
            let w = sin_theta as f64;
            weight += w;
            for i in 0..9 {
                for c in 0..3 {
                    sh[i][c] += (t[c] * intensity) as f64 * b[i] as f64 * w;
                }
            }
        }
    }

    // Normalise to the sphere's solid angle.
    let scale = if weight > 0.0 { 4.0 * std::f64::consts::PI / weight } else { 0.0 };
    let mut out = SkyIrradiance::default();
    for i in 0..9 {
        for c in 0..3 {
            out.sh[i][c] = (sh[i][c] * scale) as f32;
        }
    }
    out
}

/// Half-precision, for the panorama texture.
///
/// `Rgba16Float` rather than `Rgba8UnormSrgb`: a panorama carries values well
/// above 1 and that is the entire point of it. Storing the background as sRGB
/// bytes would clamp the sun to white and make the buffer disagree with the
/// coefficients projected from the same file.
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00; // infinity, or a NaN flushed to one
    }
    if exp <= 0 {
        return sign; // subnormal, flushed to zero
    }
    if exp > 0x1e {
        exp = 0x1e;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

/// The panorama on the GPU, plus the coefficients projected from it.
pub struct Sky {
    pub bind_group: BindGroup,
    pub irradiance: SkyIrradiance,
    _texture: Texture,
    _sampler: Sampler,
}

pub fn sky_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("sky_bgl"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

impl Sky {
    pub fn new(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        pano: &Panorama,
        rotation_deg: f32,
        intensity: f32,
    ) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("sky_panorama"),
            size: Extent3d {
                width: pano.width,
                height: pano.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let mut half = Vec::with_capacity((pano.width * pano.height * 4) as usize);
        for i in 0..(pano.width * pano.height) as usize {
            half.push(f32_to_f16(pano.rgb[i * 3]));
            half.push(f32_to_f16(pano.rgb[i * 3 + 1]));
            half.push(f32_to_f16(pano.rgb[i * 3 + 2]));
            half.push(f32_to_f16(1.0));
        }
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(&half),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(pano.width * 8),
                rows_per_image: Some(pano.height),
            },
            Extent3d {
                width: pano.width,
                height: pano.height,
                depth_or_array_layers: 1,
            },
        );

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("sky_sampler"),
            // REPEAT across u, clamp down v. A panorama wraps in longitude and
            // emphatically does not in latitude -- wrapping there samples the
            // sky when looking at the ground.
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        let view = texture.create_view(&TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("sky_bg"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&view) },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&sampler) },
            ],
        });

        Self {
            bind_group,
            irradiance: project_irradiance(pano, rotation_deg, intensity),
            _texture: texture,
            _sampler: sampler,
        }
    }

    /// A scene with no sky: one black texel, and the flat ambient as before.
    ///
    /// Always bound, like terrain's placeholder layers. An optional binding
    /// would mean two bind group layouts and therefore two pipelines, and this
    /// costs one texel.
    pub fn none(device: &Device, queue: &Queue, layout: &BindGroupLayout, ambient: f32) -> Self {
        let mut s = Self::new(
            device,
            queue,
            layout,
            &Panorama::solid([0.0, 0.0, 0.0], 1, 1),
            0.0,
            1.0,
        );
        s.irradiance = SkyIrradiance::flat(ambient);
        s
    }
}

pub struct SkyPipeline {
    pub pipeline: RenderPipeline,
    pub layout: BindGroupLayout,
}

impl SkyPipeline {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sky_shader"),
            source: ShaderSource::Wgsl(sky_shader().into()),
        });
        let layout = sky_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sky_layout"),
            bind_group_layouts: &[uniform_layout, &layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("sky_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                // No vertex buffer: three vertices generated from their index.
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::COLOR,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: None,
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                // DRAWN LAST, AND WRITES NO DEPTH.
                //
                // The sky sits at the far plane and is only visible where
                // nothing else was drawn, so it goes after the opaques with
                // `LessEqual` and early-Z rejects every covered pixel. Drawing
                // it FIRST would shade every one of those pixels and then throw
                // the work away, which on a fill-limited tile GPU is the whole
                // cost of the pass for nothing.
                depth_write_enabled: false,
                depth_compare: CompareFunction::LessEqual,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview: None,
            cache: None,
        });

        Self { pipeline, layout }
    }
}

fn sky_shader() -> String {
    format!(
        r#"
{lights_block}

// Group 1: the panorama. Group 0 comes from the lights block above, which is
// where `sky_uv` and the irradiance coefficients live.
@group(1) @binding(0) var sky_tex: texture_2d<f32>;
@group(1) @binding(1) var sky_samp: sampler;

struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}}

// One oversized triangle covering the target, from the vertex index alone.
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {{
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VOut;
    // z = 1: the far plane. With depth_compare LessEqual and no depth write,
    // this passes only where nothing nearer was drawn.
    out.clip = vec4<f32>(p[vi], 1.0, 1.0);
    out.ndc = p[vi];
    return out;
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    // The view ray, recovered by putting the pixel back through the inverse of
    // view_proj. Two points on the ray rather than one, because the near point
    // is where the eye is and the difference is the direction -- which is what
    // makes this correct for an off-centre projection, and every headset's is.
    let near = camera.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let far  = camera.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - near.xyz / near.w);

    let uv = sky_uv(dir);
    let radiance = textureSample(sky_tex, sky_samp, uv).rgb * camera.sky_params.x;

    // Reinhard, then the sRGB target does the rest. A panorama carries values
    // well above one and this pass has no tonemapping stage in front of it, so
    // without this a bright sky clips to flat white and a dim one reads black.
    let mapped = radiance / (radiance + vec3<f32>(1.0));
    return vec4<f32>(mapped, 1.0);
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
    }

    fn lobby_sky() -> Option<Panorama> {
        let p = workspace_root()
            .join("game/skies/kloofendal_48d_partly_cloudy_puresky/sky.hdr");
        let bytes = std::fs::read(p).ok()?;
        decode_radiance(&bytes).ok()
    }

    #[test]
    fn the_shipped_panorama_decodes() {
        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        assert_eq!((pano.width, pano.height), (1024, 512));
        assert_eq!(pano.rgb.len(), (1024 * 512 * 3) as usize);
        assert!(pano.rgb.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn a_panorama_carries_values_above_one() {
        // The entire reason for the format. If a decode clamped -- or if this
        // were loaded as 8-bit -- the sun would be the same brightness as a
        // white cloud and the projection below would be meaningless.
        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        let peak = pano.rgb.iter().cloned().fold(0.0f32, f32::max);
        assert!(peak > 100.0, "peak radiance {peak} is too low to be HDR");
    }

    #[test]
    fn the_suns_texel_is_where_the_direction_mapping_says_it_is() {
        // THE PIN. The lobby's sun was measured in the running editor at
        // azimuth 34.3, elevation 48 -- by scanning the rendered sky for its
        // brightest direction. This asserts the mapping agrees, so the light
        // the engine computes comes from where the sky visibly shows it.
        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        let mut best = (f32::MIN, 0u32, 0u32);
        for y in 0..pano.height {
            for x in 0..pano.width {
                let t = pano.texel(x, y);
                let l = 0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2];
                if l > best.0 {
                    best = (l, x, y);
                }
            }
        }

        let (elev, azim) = (48.0f32.to_radians(), 34.3f32.to_radians());
        let d = [
            elev.cos() * azim.sin(),
            elev.sin(),
            elev.cos() * azim.cos(),
        ];
        let uv = direction_to_uv(d);
        let (px, py) = (
            (uv[0] * pano.width as f32) as u32,
            (uv[1] * pano.height as f32) as u32,
        );
        assert!(
            px.abs_diff(best.1) <= 2 && py.abs_diff(best.2) <= 2,
            "the sun's direction maps to ({px}, {py}) but the brightest texel is at ({}, {})",
            best.1,
            best.2,
        );
    }

    #[test]
    fn the_shipped_sky_lights_the_ground_from_above_and_from_its_sun() {
        // END TO END on the file a level actually uses. The synthetic tests
        // above prove the maths; this proves the maths applied to real data
        // gives a sane answer, which is the part that would otherwise only be
        // discovered by putting a headset on.
        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        let sh = project_irradiance(&pano, 0.0, 1.0);

        // An outdoor panorama is far brighter above than below: a surface
        // facing up must receive more than one facing down. If the latitude
        // mapping were flipped this is the assertion that notices.
        let up = sh.evaluate([0.0, 1.0, 0.0]);
        let down = sh.evaluate([0.0, -1.0, 0.0]);
        assert!(
            up[0] > down[0] * 1.5,
            "the sky is not brighter above than below: up {up:?}, down {down:?}",
        );

        // And it is brighter toward the sun than away from it. The lobby's sun
        // is at azimuth 34.3, so a surface facing that way gets more.
        let a = 34.3f32.to_radians();
        let toward = sh.evaluate([a.sin(), 0.0, a.cos()]);
        let away = sh.evaluate([-a.sin(), 0.0, -a.cos()]);
        assert!(
            toward[0] > away[0],
            "facing the sun ({toward:?}) is not brighter than facing away ({away:?})",
        );

        // Sane magnitudes. A daylight HDRI should land the ambient somewhere
        // usable rather than at 0.001 or at 40 -- both of which are what a
        // missing or doubled normalisation looks like, and both of which read
        // as "the sky does nothing" or "everything is white".
        assert!(
            up[0] > 0.05 && up[0] < 20.0,
            "upward irradiance {up:?} is not a plausible daylight value",
        );
    }

    #[test]
    fn the_uv_mapping_round_trips() {
        for &d in &[
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.3, 0.5, -0.8],
        ] {
            let uv = direction_to_uv(d);
            let back = uv_to_direction(uv[0], uv[1]);
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            for c in 0..3 {
                assert!(
                    (back[c] - d[c] / n).abs() < 1e-3,
                    "{d:?} -> {uv:?} -> {back:?}",
                );
            }
        }
    }

    #[test]
    fn a_uniform_sky_gives_uniform_irradiance() {
        // A sphere of constant radiance L lights every normal identically, and
        // the value is L. Anything else means the solid-angle weighting or the
        // normalisation is wrong -- both of which are easy to get subtly wrong
        // and impossible to see in a screenshot.
        let pano = Panorama::solid([0.5, 0.5, 0.5], 64, 32);
        let sh = project_irradiance(&pano, 0.0, 1.0);
        for n in [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.3, -0.4, 0.9]] {
            let e = sh.evaluate(n);
            for c in 0..3 {
                assert!(
                    (e[c] - 0.5).abs() < 0.02,
                    "normal {n:?} got {e:?}, expected a flat 0.5",
                );
            }
        }
    }

    #[test]
    fn the_solid_angle_weighting_is_not_optional() {
        // An equirectangular image gives the pole as many texels as the
        // equator. Without the sin(theta) weight, a small bright cap overhead
        // dominates the result; with it, it contributes in proportion to the
        // solid angle it actually covers. So: a sky black everywhere except a
        // narrow band at the very top must contribute only a little.
        let mut pano = Panorama::solid([0.0, 0.0, 0.0], 64, 32);
        for y in 0..2 {
            for x in 0..64 {
                let i = ((y * 64 + x) * 3) as usize;
                pano.rgb[i] = 10.0;
                pano.rgb[i + 1] = 10.0;
                pano.rgb[i + 2] = 10.0;
            }
        }
        let up = project_irradiance(&pano, 0.0, 1.0).evaluate([0.0, 1.0, 0.0]);
        assert!(
            up[0] < 1.0,
            "a thin cap at the pole contributed {up:?}, so it is not being \
             weighted by its solid angle",
        );
        assert!(up[0] > 0.0, "the cap contributed nothing at all: {up:?}");
    }

    #[test]
    fn a_sky_bright_on_one_side_lights_that_side() {
        // The whole point of using nine coefficients rather than one: the
        // ambient term becomes DIRECTIONAL. A normal facing the bright half
        // must receive more than one facing away, which a flat ambient cannot
        // express at all.
        let mut pano = Panorama::solid([0.0, 0.0, 0.0], 64, 32);
        for y in 0..32 {
            for x in 0..32 {
                let i = ((y * 64 + x) * 3) as usize;
                pano.rgb[i] = 4.0;
                pano.rgb[i + 1] = 4.0;
                pano.rgb[i + 2] = 4.0;
            }
        }
        let sh = project_irradiance(&pano, 0.0, 1.0);
        // u < 0.5 is the lit half, which by direction_to_uv is -x.
        let lit = sh.evaluate([-1.0, 0.0, 0.0]);
        let dark = sh.evaluate([1.0, 0.0, 0.0]);
        assert!(
            lit[0] > dark[0] * 2.0,
            "the lit side got {lit:?} and the dark side {dark:?}",
        );
    }

    #[test]
    fn rotating_the_sky_moves_the_light_with_it() {
        // `rotation_deg` exists so an author can put the sun where the level
        // needs it. If the coefficients did not turn with the picture, the
        // control would move the background and leave the lighting behind --
        // exactly the mismatch an HDRI exists to prevent.
        let mut pano = Panorama::solid([0.0, 0.0, 0.0], 64, 32);
        for y in 0..32 {
            for x in 0..16 {
                let i = ((y * 64 + x) * 3) as usize;
                pano.rgb[i] = 8.0;
                pano.rgb[i + 1] = 8.0;
                pano.rgb[i + 2] = 8.0;
            }
        }
        let probe = [-1.0, 0.0, 0.0];
        let straight = project_irradiance(&pano, 0.0, 1.0).evaluate(probe);
        let turned = project_irradiance(&pano, 180.0, 1.0).evaluate(probe);
        assert!(
            (straight[0] - turned[0]).abs() > straight[0] * 0.3,
            "turning the sky by 180 degrees barely changed the light: \
             {straight:?} vs {turned:?}",
        );
    }

    #[test]
    fn intensity_scales_the_light_it_casts() {
        let pano = Panorama::solid([0.4, 0.4, 0.4], 32, 16);
        let one = project_irradiance(&pano, 0.0, 1.0).evaluate([0.0, 1.0, 0.0]);
        let two = project_irradiance(&pano, 0.0, 2.0).evaluate([0.0, 1.0, 0.0]);
        assert!((two[0] - one[0] * 2.0).abs() < 0.01, "{one:?} vs {two:?}");
    }

    #[test]
    fn no_sky_evaluates_to_exactly_the_old_flat_ambient() {
        // The compatibility claim. A level with no sky must render EXACTLY as
        // it did before this existed, in every direction -- otherwise adding
        // the feature silently re-lights every scene that does not use it.
        let flat = SkyIrradiance::flat(0.6);
        for n in [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [-0.5, 0.2, 0.8]] {
            let e = flat.evaluate(n);
            for c in 0..3 {
                assert!((e[c] - 0.6).abs() < 1e-4, "normal {n:?} got {e:?}, wanted 0.6");
            }
        }
    }

    #[test]
    fn half_precision_survives_the_range_a_sky_uses() {
        for v in [0.0f32, 0.5, 1.0, 12.5, 250.0, 6000.0] {
            let h = f32_to_f16(v);
            // Reconstruct, the way the GPU will.
            let sign = (h >> 15) as u32;
            let exp = ((h >> 10) & 0x1f) as i32;
            let mant = (h & 0x3ff) as u32;
            let back = if exp == 0 {
                0.0
            } else {
                f32::from_bits((sign << 31) | (((exp - 15 + 127) as u32) << 23) | (mant << 13))
            };
            let err = if v == 0.0 { back } else { (back - v).abs() / v };
            assert!(err < 0.01, "{v} became {back}");
        }
    }

    #[test]
    fn a_malformed_file_is_refused_rather_than_guessed_at() {
        assert!(decode_radiance(b"not an hdr at all").is_err());
        assert!(decode_radiance(b"#?RADIANCE\n\n+Y 4 +X 4\n").is_err());
    }
}

/// Rendering tests for the background pass.
///
/// The projection and the coefficients above are pure and tested without a GPU.
/// What those cannot see is the shader: whether the view ray is reconstructed
/// correctly, whether `sky_uv` in WGSL agrees with `direction_to_uv` in Rust,
/// and whether the depth setup keeps the sky behind the level instead of over
/// it. All three fail silently and look like art problems.
#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::renderer::lights::LightsUniform;
    use crate::renderer::uniforms::{ShadowUpload, SkyUpload, UniformBuffer};

    const SIZE: u32 = 9; // odd, so the centre texel is exactly at NDC (0, 0)

    /// Red over one half of the longitude, blue over the other.
    ///
    /// A split rather than a gradient: it turns "is the shader looking the right
    /// way" into a question with a categorical answer, which a smooth panorama
    /// would blur into a judgement call.
    fn split_panorama() -> Panorama {
        let (w, h) = (64u32, 32u32);
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for _y in 0..h {
            for x in 0..w {
                // Values above 1 so the Reinhard curve in the shader has
                // something to do and the test notices if it is removed.
                if x < w / 2 {
                    rgb.extend_from_slice(&[3.0, 0.0, 0.0]);
                } else {
                    rgb.extend_from_slice(&[0.0, 0.0, 3.0]);
                }
            }
        }
        Panorama { width: w, height: h, rgb }
    }

    /// Renders the sky looking along `dir` and returns the centre pixel.
    fn look(dir: [f32; 3], pano: &Panorama) -> Option<[u8; 4]> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        lights.upload(&queue, &[]);

        let eye = glam::Vec3::ZERO;
        let d = glam::Vec3::from(dir).normalize();
        let up = if d.y.abs() > 0.99 { glam::Vec3::Z } else { glam::Vec3::Y };
        let view_proj = glam::Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0)
            * glam::Mat4::look_at_rh(eye, eye + d, up);
        uniforms.upload_with_sky(
            &queue,
            view_proj,
            eye,
            &ShadowUpload::disabled(),
            &SkyUpload { intensity: 1.0, sh: [[0.0; 4]; 9] },
        );

        let pipeline = SkyPipeline::new(&device, format, &uniforms.layout, 1);
        let sky = Sky::new(&device, &queue, &pipeline.layout, pano, 0.0, 1.0);

        let desc = |fmt, usage| TextureDescriptor {
            label: None,
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        };
        let target = device.create_texture(&desc(
            format,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let depth = device.create_texture(&desc(
            TextureFormat::Depth32Float,
            TextureUsages::RENDER_ATTACHMENT,
        ));
        let target_view = target.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: None,
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("sky_test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::GREEN), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &sky.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(SIZE),
                },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait).ok();
        let data = slice.get_mapped_range();
        let at = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[at], data[at + 1], data[at + 2], data[at + 3]])
    }

    macro_rules! shot {
        ($e:expr) => {
            match $e {
                Some(px) => px,
                None => {
                    eprintln!("skipping: no GPU adapter available");
                    return;
                }
            }
        };
    }

    #[test]
    fn the_shader_samples_the_direction_the_cpu_mapping_says_it_should() {
        // THE PIN BETWEEN THE TWO MAPPINGS. `sky_uv` in WGSL and
        // `direction_to_uv` in Rust are separate implementations of one
        // convention, and the coefficients are projected with the Rust one
        // while the background is drawn with the WGSL one. If they disagree,
        // a level is lit from somewhere other than where its sun is drawn --
        // which reads as a lighting bug and is a mapping bug.
        let pano = split_panorama();
        for dir in [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]] {
            let px = shot!(look(dir, &pano));
            let uv = direction_to_uv(dir);
            let expected = pano.texel(
                ((uv[0] * pano.width as f32) as u32).min(pano.width - 1),
                ((uv[1] * pano.height as f32) as u32).min(pano.height - 1),
            );
            let want_red = expected[0] > expected[2];
            assert!(
                (px[0] > px[2]) == want_red,
                "looking {dir:?}: the CPU mapping says {expected:?} but the \
                 shader drew {px:?}",
            );
        }
    }

    #[test]
    fn the_sky_is_tonemapped_rather_than_clipped() {
        // The panorama carries 3.0. Reinhard puts that at 0.75, so the channel
        // should land near 191 -- not at 255, which is what a missing tonemap
        // (or an 8-bit upload) would give, and which would look identical on a
        // bright sky while destroying every cloud in a dim one.
        let px = shot!(look([-1.0, 0.0, 0.0], &split_panorama()));
        assert!(
            px[0] > 170 && px[0] < 215,
            "expected the Reinhard value for radiance 3 (~191), got {px:?}",
        );
    }

    #[test]
    fn the_sky_only_fills_where_nothing_was_drawn() {
        // It is drawn LAST at the far plane with no depth write, so early-Z
        // rejects covered pixels rather than shading them and throwing the work
        // away. Asserted by clearing depth to 0 -- as though the whole frame
        // were already covered by nearer geometry -- and checking the clear
        // colour survives.
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let format = TextureFormat::Rgba8Unorm;
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        lights.upload(&queue, &[]);
        uniforms.upload_with_sky(
            &queue,
            glam::Mat4::IDENTITY,
            glam::Vec3::ZERO,
            &ShadowUpload::disabled(),
            &SkyUpload { intensity: 1.0, sh: [[0.0; 4]; 9] },
        );
        let pipeline = SkyPipeline::new(&device, format, &uniforms.layout, 1);
        let pano = split_panorama();
        let sky = Sky::new(&device, &queue, &pipeline.layout, &pano, 0.0, 1.0);

        let desc = |fmt, usage| TextureDescriptor {
            label: None,
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        };
        let target = device.create_texture(&desc(
            format,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let depth = device.create_texture(&desc(
            TextureFormat::Depth32Float,
            TextureUsages::RENDER_ATTACHMENT,
        ));
        let tv = target.create_view(&Default::default());
        let dv = depth.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: None,
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("occluded_sky"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &tv,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::GREEN), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &dv,
                    // Everything already at the near plane: nothing of the sky
                    // may pass the LessEqual test.
                    depth_ops: Some(Operations { load: LoadOp::Clear(0.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &sky.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &target, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0, bytes_per_row: Some(256), rows_per_image: Some(SIZE),
                },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait).ok();
        let data = slice.get_mapped_range();
        let at = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        let px = [data[at], data[at + 1], data[at + 2], data[at + 3]];
        assert!(
            px[1] > 200 && px[0] < 40 && px[2] < 40,
            "the sky drew over geometry that was already in front of it: {px:?}",
        );
    }

    #[test]
    fn the_pipeline_validates_multisampled_too() {
        // It runs in the scene pass, which is 4x on the headset.
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let _ = &queue;
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        device.push_error_scope(ErrorFilter::Validation);
        let _one = SkyPipeline::new(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 1);
        let _four = SkyPipeline::new(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4);
        if let Some(e) = pollster::block_on(device.pop_error_scope()) {
            panic!("the sky pipeline failed validation: {e}");
        }
    }
}
