use wgpu::{Device, Queue, TextureView};

pub const MAX_PROXY_FIELDS: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub struct ProxyField {

    pub dims: [u32; 3],

    pub max_distance: f32,
    pub distances: Vec<u8>,

    pub albedo: [f32; 3],
}

pub type FieldSlot = [[f32; 4]; 3];

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
