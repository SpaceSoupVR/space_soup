//! STANDING MODELS' SHAPES for the reflection trace, on the GPU: each model's
//! distance field in one 3D atlas, traced inside its proxy box.
//!
//! A model standing in a room -- a hanging lamp, a wall sconce -- is a proxy
//! box to the reflection trace, and a box is only its bounds. Where inside it
//! the model was used to be asked of the room's photographs, four samples
//! against 256-pixel depth, and the lamps came out ragged in every reflection:
//! the dark "L" the ceiling showed in the front door's reflection (offline,
//! 2026-09-28). A field built from the model's own triangles
//! (`space_soup_engine::reflection_proxy::model_field`) lets the trace walk to
//! the surface itself, as Lumen and distance-field shadows walk mesh fields.
//!
//! Plain data here: `space_soup` cannot depend on the engine, so the app hands
//! the fields over as [`ProxyField`].

use wgpu::{Device, Queue, TextureView};

/// How many distinct fields a level may carry at once. A field is per MODEL
/// (and scale), not per object: test_room's five lamps are two models.
pub const MAX_PROXY_FIELDS: usize = 8;

/// One model's distances over its proxy box: see
/// `space_soup_engine::reflection_proxy::ProxyField`, which this mirrors.
#[derive(Clone, Debug, PartialEq)]
pub struct ProxyField {
    /// Samples along x, y, z of the proxy's own box, x fastest; sample
    /// (i, j, k) at the centre of its cell.
    pub dims: [u32; 3],
    /// Metres a byte of 255 stands for.
    pub max_distance: f32,
    pub distances: Vec<u8>,
    /// The model's mean surface colour, linear -- how a reflection shades a
    /// part of it no photograph saw. See `probe_model_colour` in the lights
    /// block and `space_soup_engine::mesh_lightmap::model_albedo`.
    pub albedo: [f32; 3],
}

/// Where each field lies in the atlas, as the shader reads it (the uniform's
/// `proxy_fields`): `[origin_uvw.xyz, max_distance]`, `[size_uvw.xyz,
/// stop_distance]`, `[albedo.rgb, 0]`. A box-local point `p` samples at
/// `origin + (p / (2 * half) + 0.5) * size`, clamped half a texel inside the
/// field so filtering never reaches the gap beside it.
pub type FieldSlot = [[f32; 4]; 3];

/// The fields packed into one R8 3D texture, stacked along z with a one-sample
/// gap of "far" between them so filtering never blends two models, and their
/// slots. `None` for no fields.
pub fn atlas(device: &Device, queue: &Queue, fields: &[ProxyField]) -> Option<(TextureView, Vec<FieldSlot>)> {
    let fields: Vec<&ProxyField> = fields
        .iter()
        .take(MAX_PROXY_FIELDS)
        .filter(|f| f.dims.iter().all(|&d| d > 0) && f.distances.len() as u32 == f.dims[0] * f.dims[1] * f.dims[2])
        .collect();
    if fields.is_empty() {
        return None;
    }
    let w = fields.iter().map(|f| f.dims[0]).max().unwrap_or(1);
    let h = fields.iter().map(|f| f.dims[1]).max().unwrap_or(1);
    let d: u32 = fields.iter().map(|f| f.dims[2] + 1).sum::<u32>() + 1;
    // "Far" everywhere a field does not cover.
    let mut data = vec![255u8; (w * h * d) as usize];
    let mut slots = Vec::with_capacity(fields.len());
    let mut z0 = 1u32;
    for f in &fields {
        for k in 0..f.dims[2] {
            for j in 0..f.dims[1] {
                let src = ((k * f.dims[1] + j) * f.dims[0]) as usize;
                let dst = (((z0 + k) * h + j) * w) as usize;
                data[dst..dst + f.dims[0] as usize].copy_from_slice(&f.distances[src..src + f.dims[0] as usize]);
            }
        }
        // The field reaches four samples of its longest side, so half a
        // sample -- where the trace calls it a hit -- is an eighth of that.
        slots.push([
            [0.0, 0.0, z0 as f32 / d as f32, f.max_distance],
            [
                f.dims[0] as f32 / w as f32,
                f.dims[1] as f32 / h as f32,
                f.dims[2] as f32 / d as f32,
                f.max_distance / 8.0,
            ],
            [f.albedo[0], f.albedo[1], f.albedo[2], 0.0],
        ]);
        z0 += f.dims[2] + 1;
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("proxy_field_atlas"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &data,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d },
    );
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    Some((view, slots))
}

/// What a level without fields binds: one far sample, never read.
pub fn none(device: &Device) -> TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("default_proxy_field_atlas"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}
