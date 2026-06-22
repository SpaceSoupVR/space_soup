
use wgpu::{BindGroup, TextureViewDescriptor, TexelCopyBufferLayout, TextureAspect, Origin3d, TextureUsages, TexelCopyTextureInfo, Extent3d, TextureDimension, TextureDescriptor, TextureFormat, BindGroupLayout, Device, Queue, Sampler};

use crate::ui2d::{Text, Font, Color, RgbaImage, ShapeType};

use std::collections::HashMap;
use std::sync::Arc;

#[derive(Default, Debug)]
pub struct Atlas {
    pub(crate) image: ImageAtlas,
    pub(crate) text:  TextAtlas,
}

impl Atlas {
    pub(crate) fn trim(&mut self) {
        self.image.trim();
        self.text.trim();
    }
}

#[derive(Default, Debug)]
pub struct ImageAtlas(Vec<(Arc<RgbaImage>, Arc<BindGroup>)>);

impl ImageAtlas {
    pub fn trim(&mut self) {
        self.0 = self.0.drain(..).filter(|(i, _)| Arc::strong_count(i) > 1).collect();
    }

    pub fn get(
        &mut self,
        queue:  &Queue,
        device: &Device,
        layout: &BindGroupLayout,
        sampler: &Sampler,
        image:  &Arc<RgbaImage>,
    ) -> Arc<BindGroup> {
        match self.0.iter().find_map(|(i, b)| Arc::ptr_eq(i, image).then_some(b)) {
            Some(bind_group) => bind_group.clone(),
            None => {
                let size = Extent3d {
                    width:  image.width(),
                    height: image.height(),
                    depth_or_array_layers: 1,
                };

                let texture = device.create_texture(&TextureDescriptor {
                    size,
                    mip_level_count: 1,
                    sample_count:    1,
                    dimension:       TextureDimension::D2,
                    format:          TextureFormat::Rgba8UnormSrgb,
                    usage:           TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    label:           None,
                    view_formats:    &[],
                });

                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    image,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * image.width()),
                        rows_per_image: Some(image.height()),
                    },
                    size,
                );

                let texture_view = texture.create_view(&TextureViewDescriptor::default());

                let bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                    label: None,
                }));

                self.0.push((image.clone(), bind_group.clone()));
                bind_group
            }
        }
    }
}

type ImageMap = HashMap<(char, u32, u32), Option<Arc<RgbaImage>>>;
type Offset = (f32, f32);

const GLYPH_SUPERSAMPLE: usize = 3;

#[derive(Default, Debug)]
pub struct TextAtlas {
    fonts: Vec<(Arc<Font>, ImageMap)>,
}

impl TextAtlas {
    pub fn trim(&mut self) {
        self.fonts = self.fonts.drain(..).filter(|(k, _)| Arc::strong_count(k) > 1).collect();
    }

    pub fn clear(&mut self) {
        self.fonts.clear();
    }

    fn get_font(&mut self, font: &Arc<Font>) -> &mut ImageMap {
        let position = self.fonts.iter().position(|(f, _)| Arc::ptr_eq(f, font)).unwrap_or_else(|| {
            self.fonts.push((font.clone(), HashMap::new()));
            self.fonts.len() - 1
        });
        &mut self.fonts[position].1
    }

    fn get_image(&mut self, font: &Arc<Font>, c: char, font_size: f32, scale_factor: f32) -> Option<Arc<RgbaImage>> {
        let map = self.get_font(font);
        map.entry((c, font_size.to_bits(), scale_factor.to_bits())).or_insert_with(|| {
            rasterize_glyph_supersampled(font, c, font_size * scale_factor)
        }).clone()
    }

    pub fn get(&mut self, text: Text, scale_factor: f32) -> Vec<(Offset, ShapeType, Arc<RgbaImage>, Color)> {
        text.lines().iter().flat_map(|line| line.2.iter().flat_map(|ch| {
            self.get_image(&ch.2, ch.0, ch.6, scale_factor).map(|img| {
                let w = img.width()  as f32 / scale_factor;
                let h = img.height() as f32 / scale_factor;
                let shape = ShapeType::Rectangle(0.0, (w, h), 0.0);
                let offset = snap_to_physical_pixel((ch.1.0, ch.1.1), scale_factor);
                (offset, shape, img, ch.3)
            })
        }).collect::<Vec<_>>()).collect::<Vec<_>>()
    }
}

fn snap_to_physical_pixel(offset: (f32, f32), scale_factor: f32) -> (f32, f32) {
    let (x, y) = offset;
    (
        (x * scale_factor).round() / scale_factor,
        (y * scale_factor).round() / scale_factor,
    )
}

fn rasterize_glyph_supersampled(font: &Font, c: char, px: f32) -> Option<Arc<RgbaImage>> {
    let super_px = px * GLYPH_SUPERSAMPLE as f32;
    let (m_hi, bitmap_hi) = font.rasterize(c, super_px);

    if m_hi.width == 0 || m_hi.height == 0 {
        return None;
    }

    let out_w = (m_hi.width as f32 / GLYPH_SUPERSAMPLE as f32).round().max(1.0) as usize;
    let out_h = (m_hi.height as f32 / GLYPH_SUPERSAMPLE as f32).round().max(1.0) as usize;

    let downsampled = box_downsample(&bitmap_hi, m_hi.width, m_hi.height, out_w, out_h);

    let coverage: Vec<u8> = downsampled.iter().map(|a| {
        let t = *a as f32 / 255.0;
        (t.powf(1.4) * 255.0).round() as u8
    }).collect();

    let rgba: Vec<u8> = coverage.iter().flat_map(|a| [0, 0, 0, *a]).collect();

    rgba.iter().any(|a| *a != 0).then(|| {
        Arc::new(RgbaImage::from_raw(out_w as u32, out_h as u32, rgba).unwrap())
    })
}

fn box_downsample(src: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    if dst_w == 0 || dst_h == 0 {
        return Vec::new();
    }
    let mut out = vec![0u8; dst_w * dst_h];
    let x_ratio = src_w as f32 / dst_w as f32;
    let y_ratio = src_h as f32 / dst_h as f32;

    for dy in 0..dst_h {
        let sy0 = (dy as f32 * y_ratio).floor() as usize;
        let sy1 = (((dy + 1) as f32 * y_ratio).ceil() as usize).max(sy0 + 1).min(src_h);
        for dx in 0..dst_w {
            let sx0 = (dx as f32 * x_ratio).floor() as usize;
            let sx1 = (((dx + 1) as f32 * x_ratio).ceil() as usize).max(sx0 + 1).min(src_w);

            let mut sum: u32 = 0;
            let mut count: u32 = 0;
            for sy in sy0..sy1 {
                let row = sy * src_w;
                for sx in sx0..sx1 {
                    sum += src[row + sx] as u32;
                    count += 1;
                }
            }
            out[dy * dst_w + dx] = if count > 0 { (sum / count) as u8 } else { 0 };
        }
    }
    out
}