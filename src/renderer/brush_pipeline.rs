use bytemuck::{Pod, Zeroable};
use wgpu::*;

use super::lights::wgsl_lights_block;
use super::terrain_pipeline::{resample, solid_image, TerrainImage};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BrushVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],

    pub tangent: [f32; 4],

    pub uv: [f32; 2],

    pub material: u32,

    pub tint: [f32; 4],

    pub uv2: [f32; 2],

    pub face_centre: [f32; 3],

    pub uv2_rect: [f32; 4],

    pub face_half_extent: [f32; 2],
}

impl BrushVertex {
    pub const ATTRIBS: [VertexAttribute; 10] = vertex_attr_array![
        0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x2, 4 => Uint32, 5 => Float32x4,
        6 => Float32x2, 7 => Float32x3, 8 => Float32x4, 9 => Float32x2
    ];

    pub fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub const MAX_BRUSH_MATERIALS: usize = 24;

pub static WORLD_BRUSHES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

const POSED_BRUSH_VERTEX: &str = "@vertex fn vs_main(v: VIn) -> VOut {\n    var out: VOut;\n";
const WORLD_BRUSH_VERTEX: &str = "@vertex fn vs_main(w: VIn) -> VOut {\n    var out: VOut;\n    var v = w;\n    v.pos = to_player_space(w.pos);\n    v.norm = to_player_direction(w.norm);\n    v.tangent = vec4<f32>(to_player_direction(w.tangent.xyz), w.tangent.w);\n    v.face_centre = to_player_space(w.face_centre);\n";

const POSED_SEAL_VERTEX: &str = "return cam_view_proj() * vec4<f32>(pos, 1.0);";
const WORLD_SEAL_VERTEX: &str = "return cam_view_proj() * vec4<f32>(to_player_space(pos), 1.0);";

pub fn world_brush_vertices(src: &str) -> Option<String> {
    (src.matches(POSED_BRUSH_VERTEX).count() == 1 && src.contains("@location(7) face_centre: vec3<f32>"))
        .then(|| src.replacen(POSED_BRUSH_VERTEX, WORLD_BRUSH_VERTEX, 1))
}

fn in_the_brushes_frame(src: String) -> String {
    if !WORLD_BRUSHES.load(std::sync::atomic::Ordering::Relaxed) {
        return src;
    }
    match world_brush_vertices(&src) {
        Some(world) => world,
        None => {

            log::error!("WORLD BRUSHES: a brush shader does not open as expected; it reads its vertices as posed");
            src
        }
    }
}

pub struct BrushPipeline {
    pub pipeline: RenderPipeline,
    pub material_layout: BindGroupLayout,

    pub lightmap_layout: BindGroupLayout,

    pub reads_immediates: bool,
}

pub const DEFAULT_ROUGHNESS: u8 = 140;

pub fn default_brush_lightmap(
    device: &Device,
    queue: &Queue,
    layout: &BindGroupLayout,
) -> crate::renderer::mesh::LoadedTexture {
    crate::renderer::mesh::create_lightmap_texture(
        device, queue, layout, &[0u8, 0, 0, 255], 1, 1, None,
    )
}

pub struct BrushMaterials {
    pub bind_group: BindGroup,

    pub reflective: bool,

    _uv_scales: wgpu::Buffer,
}

const NORMAL_VARIANCE_CLAMP: f32 = 0.18;

fn roughness_chain_with_normal_variance(
    rough: &TerrainImage,
    normal: Option<&TerrainImage>,
) -> Vec<TerrainImage> {
    let mut rough_levels = mip_chain(rough, false);
    let Some(normal) = normal else {
        return rough_levels;
    };
    if normal.width != rough.width || normal.height != rough.height {

        return rough_levels;
    }

    let mut w = normal.width;
    let mut h = normal.height;
    let mut vecs: Vec<[f32; 3]> = (0..(w * h) as usize)
        .map(|i| {
            let px = &normal.rgba[i * 4..i * 4 + 3];
            [
                px[0] as f32 / 255.0 * 2.0 - 1.0,
                px[1] as f32 / 255.0 * 2.0 - 1.0,
                px[2] as f32 / 255.0 * 2.0 - 1.0,
            ]
        })
        .collect();

    for level in 1..rough_levels.len() {
        let (nw, nh) = (w.max(2) / 2, h.max(2) / 2);
        let mut next = vec![[0.0f32; 3]; (nw * nh) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 3];
                for (dx, dy) in [(0u32, 0u32), (1, 0), (0, 1), (1, 1)] {
                    let sx = (x * 2 + dx).min(w - 1);
                    let sy = (y * 2 + dy).min(h - 1);
                    let v = vecs[(sy * w + sx) as usize];
                    acc[0] += v[0];
                    acc[1] += v[1];
                    acc[2] += v[2];
                }
                next[(y * nw + x) as usize] = [acc[0] / 4.0, acc[1] / 4.0, acc[2] / 4.0];
            }
        }
        vecs = next;
        w = nw;
        h = nh;

        let lvl = &mut rough_levels[level];
        if lvl.width != w || lvl.height != h {

            break;
        }
        for i in 0..(w * h) as usize {
            let v = vecs[i];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();

            let sigma2 = if len > 1e-4 { (1.0 - len) / len } else { 1.0 };
            let kernel = (2.0 * sigma2).min(NORMAL_VARIANCE_CLAMP);

            let base = lvl.rgba[i * 4] as f32 / 255.0;
            let filtered = (base * base + kernel).clamp(0.0, 1.0).sqrt();
            let byte = (filtered * 255.0).round().clamp(0.0, 255.0) as u8;
            lvl.rgba[i * 4] = byte;
            lvl.rgba[i * 4 + 1] = byte;
            lvl.rgba[i * 4 + 2] = byte;
        }
    }
    rough_levels
}

pub(crate) fn mip_chain(img: &TerrainImage, srgb: bool) -> Vec<TerrainImage> {
    let to_linear = |b: u8| {
        let c = b as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let to_srgb = |v: f32| {
        let c = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (c.clamp(0.0, 1.0) * 255.0).round() as u8
    };

    let mut levels = vec![TerrainImage { width: img.width, height: img.height, rgba: img.rgba.clone() }];
    loop {
        let prev = levels.last().unwrap();
        if prev.width <= 1 && prev.height <= 1 {
            break;
        }
        let (w, h) = (prev.width.max(2) / 2, prev.height.max(2) / 2);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {

                    let linear = srgb && c < 3;
                    let mut sum = 0.0f32;
                    for (dx, dy) in [(0u32, 0u32), (1, 0), (0, 1), (1, 1)] {
                        let sx = (x * 2 + dx).min(prev.width - 1);
                        let sy = (y * 2 + dy).min(prev.height - 1);
                        let b = prev.rgba[((sy * prev.width + sx) * 4 + c as u32) as usize];
                        sum += if linear { to_linear(b) } else { b as f32 / 255.0 };
                    }
                    let avg = sum / 4.0;
                    rgba[((y * w + x) * 4 + c as u32) as usize] =
                        if linear { to_srgb(avg) } else { (avg * 255.0).round() as u8 };
                }
            }
        }
        levels.push(TerrainImage { width: w, height: h, rgba });
    }
    levels
}

pub const BRUSH_LIGHT_DEBUG: bool = false;

pub const BRUSH_SOURCE_DEBUG: bool = false;

pub const BRUSH_EDGE_DEBUG: bool = false;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourcesView {

    Ratios,

    ProbeFactors,

    Absolute,
}

pub const SOURCES_VIEW: SourcesView = SourcesView::Absolute;

pub const BRUSH_CENTROID_VARYINGS: bool = false;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DebugView {
    #[default]
    Off,
    Sources,
    Ssr,
}

impl DebugView {

    pub fn next(self) -> Self {
        match self {
            DebugView::Off => DebugView::Sources,
            DebugView::Sources => DebugView::Ssr,
            DebugView::Ssr => DebugView::Off,
        }
    }

    pub fn toggle_sources(self) -> Self {
        match self {
            DebugView::Off => DebugView::Sources,
            DebugView::Sources | DebugView::Ssr => DebugView::Off,
        }
    }
}

pub const SSR_ROUGHNESS_THRESHOLD: f32 = 0.35;

pub fn roughness_is_reflective(rough: Option<&crate::renderer::terrain_pipeline::TerrainImage>) -> bool {
    let Some(img) = rough else { return false };
    if img.rgba.is_empty() {
        return false;
    }

    let sum: u64 = img.rgba.chunks_exact(4).map(|p| p[0] as u64).sum();
    let n = (img.rgba.len() / 4).max(1) as f64;
    let mean = (sum as f64 / n) / 255.0;
    (mean as f32) < SSR_ROUGHNESS_THRESHOLD
}

pub fn brush_material_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("brush_material_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },

            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },

            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },

            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

pub fn material_uv_scales(colours: &[Option<TerrainImage>]) -> [[f32; 4]; MAX_BRUSH_MATERIALS] {
    let mut out = [[1.0, 1.0, 0.0, 0.0]; MAX_BRUSH_MATERIALS];
    for (slot, img) in out.iter_mut().zip(colours) {
        if let Some(img) = img.as_ref().filter(|i| i.width > 0 && i.height > 0) {
            slot[1] = img.width as f32 / img.height as f32;
        }
    }
    out
}

/// One of the four maps a brush material binds, in the bind group's order.
/// Its name is the source file's stem in a material's folder
/// (`materials/<id>/color.jpg`) and the encoded file's (`bin/brush_astc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushMap {
    Colour,
    Normal,
    Rough,
    Ao,
}

impl BrushMap {
    pub const ALL: [Self; 4] = [Self::Colour, Self::Normal, Self::Rough, Self::Ao];

    pub fn name(self) -> &'static str {
        match self {
            Self::Colour => "color",
            Self::Normal => "normal",
            Self::Rough => "rough",
            Self::Ao => "ao",
        }
    }

    /// Colour is stored in sRGB; everything else is data.
    pub fn srgb(self) -> bool {
        self == Self::Colour
    }

    /// What a material with no map of this kind holds: white, a flat normal,
    /// the default roughness, no occlusion.
    pub fn fallback(self) -> [u8; 3] {
        match self {
            Self::Colour | Self::Ao => [255, 255, 255],
            Self::Normal => [128, 128, 255],
            Self::Rough => [DEFAULT_ROUGHNESS; 3],
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Colour => "brush_colour_array",
            Self::Normal => "brush_normal_array",
            Self::Rough => "brush_rough_array",
            Self::Ao => "brush_ao_array",
        }
    }
}

/// The size every layer of a scene's material arrays has: its first colour
/// map's. The others are resampled to it.
pub fn brush_array_size(colours: &[Option<TerrainImage>]) -> (u32, u32) {
    colours.iter().flatten().map(|i| (i.width, i.height)).next().unwrap_or((1, 1))
}

/// Every mip of one material's map as its array layer holds it at `w`x`h`:
/// the map fitted to that size (resampled, or the map's fallback where the
/// material has none), then filtered -- sRGB-correct for colour, roughness
/// widened by its normal map's variance. The one filtering both the RGBA
/// upload and the ASTC encoder (`bin/brush_astc`) use, so an encoded
/// material is that upload, compressed. `normal` is read for roughness only.
pub fn brush_layer_mips(map: BrushMap, image: Option<&TerrainImage>, normal: Option<&TerrainImage>, w: u32, h: u32) -> Vec<TerrainImage> {
    let fit = |img: &TerrainImage| if img.width == w && img.height == h { img.clone() } else { resample(img, w, h) };
    let filled = image.map(fit).unwrap_or_else(|| solid_image(map.fallback(), w, h));
    match map {
        BrushMap::Rough => roughness_chain_with_normal_variance(&filled, normal.map(fit).as_ref()),
        _ => mip_chain(&filled, map.srgb()),
    }
}

impl BrushMaterials {

    pub fn new(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
    ) -> Self {
        let (w, h) = brush_array_size(colours);
        let textures = BrushMap::ALL.map(|map| {
            let images = match map {
                BrushMap::Colour => colours,
                BrushMap::Normal => normals,
                BrushMap::Rough => roughs,
                BrushMap::Ao => aos,
            };
            Self::array_texture(device, queue, map, images, normals, w, h)
        });
        Self::with_arrays(device, layout, &textures, colours, roughs)
    }

    /// The bind group over finished arrays (colour, normal, roughness,
    /// occlusion), from images ([`Self::new`]) or encoded blocks
    /// (`brush_astc::BrushMaterials::from_astc`). `colours` and `roughs` are
    /// the images either way: each material's tiling aspect and whether any
    /// is polished enough to reflect are read from them.
    pub(crate) fn with_arrays(device: &Device, layout: &BindGroupLayout, textures: &[Texture; 4], colours: &[Option<TerrainImage>], roughs: &[Option<TerrainImage>]) -> Self {
        let [colour_tex, normal_tex, rough_tex, ao_tex] = textures;

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("brush_sampler"),

            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,

            anisotropy_clamp: 8,
            ..Default::default()
        });

        let rough_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("brush_rough_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            anisotropy_clamp: 1,
            ..Default::default()
        });

        let uv_scales = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("brush_material_uv_scales"),
                contents: bytemuck::cast_slice(&material_uv_scales(colours)),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("brush_materials"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&colour_tex.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&normal_tex.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&rough_tex.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&ao_tex.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Sampler(&rough_sampler),
                },
                BindGroupEntry { binding: 6, resource: uv_scales.as_entire_binding() },
            ],
        });

        let reflective = roughs.iter().any(|r| roughness_is_reflective(r.as_ref()));
        Self { bind_group, reflective, _uv_scales: uv_scales }
    }

    pub fn fallback(device: &Device, queue: &Queue, layout: &BindGroupLayout) -> Self {
        Self::new(device, queue, layout, &[], &[], &[], &[])
    }

    /// One map's array, every layer's mips made by [`brush_layer_mips`].
    /// `normals` widen the roughness map's mips.
    pub(crate) fn array_texture(
        device: &Device,
        queue: &Queue,
        map: BrushMap,
        images: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        w: u32,
        h: u32,
    ) -> Texture {
        let chains: Vec<Vec<TerrainImage>> = (0..MAX_BRUSH_MATERIALS)
            .map(|i| brush_layer_mips(map, images.get(i).and_then(|x| x.as_ref()), normals.get(i).and_then(|x| x.as_ref()), w, h))
            .collect();
        let mip_level_count = chains.first().map(|c| c.len() as u32).unwrap_or(1);
        let format = if map.srgb() { TextureFormat::Rgba8UnormSrgb } else { TextureFormat::Rgba8Unorm };

        let tex = device.create_texture(&TextureDescriptor {
            label: Some(map.label()),
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: MAX_BRUSH_MATERIALS as u32,
            },
            mip_level_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (layer, chain) in chains.iter().enumerate() {
            for (level, img) in chain.iter().enumerate() {
                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: level as u32,
                        origin: Origin3d { x: 0, y: 0, z: layer as u32 },
                        aspect: TextureAspect::All,
                    },
                    &img.rgba,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * img.width),
                        rows_per_image: Some(img.height),
                    },
                    Extent3d { width: img.width, height: img.height, depth_or_array_layers: 1 },
                );
            }
        }
        tex
    }
}

impl BrushPipeline {
    pub fn new(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Ccw, 1, BRUSH_SOURCE_DEBUG, crate::renderer::multiview::ViewMode::Mono,
        )
    }

    pub fn new_multisampled_opaque(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_blended(
            device, format, uniform_layout, FrontFace::Ccw, samples, BRUSH_SOURCE_DEBUG, None, crate::renderer::multiview::ViewMode::Mono,
        )
    }

    pub fn new_multisampled_opaque_stereo(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_blended(
            device, format, uniform_layout, FrontFace::Ccw, samples, BRUSH_SOURCE_DEBUG, None,
            crate::renderer::multiview::ViewMode::Stereo,
        )
    }

    pub fn new_multisampled_sources_stereo(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(
            device, format, uniform_layout, FrontFace::Ccw, samples, true, crate::renderer::multiview::ViewMode::Stereo,
        )
    }

    pub fn new_multisampled_stereo(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(
            device, format, uniform_layout, FrontFace::Ccw, samples, BRUSH_SOURCE_DEBUG,
            crate::renderer::multiview::ViewMode::Stereo,
        )
    }

    pub fn new_multisampled_sources(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Ccw, samples, true, crate::renderer::multiview::ViewMode::Mono,
        )
    }

    pub fn new_multisampled(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Ccw, samples, BRUSH_SOURCE_DEBUG, crate::renderer::multiview::ViewMode::Mono,
        )
    }

    pub fn new_mirror(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
    ) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Cw, 1, BRUSH_SOURCE_DEBUG, crate::renderer::multiview::ViewMode::Mono,
        )
    }

    pub fn new_ssr(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,

    ) -> Self {
        Self::new_ssr_variant(
            device, format, uniform_layout, ssr_scene_layout,
            BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG,
        )
    }

    pub fn new_ssr_debug(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,
        view: DebugView,
    ) -> Self {
        Self::new_ssr_variant(
            device, format, uniform_layout, ssr_scene_layout,
            view == DebugView::Sources, view == DebugView::Ssr,
        )
    }

    pub fn new_ssr_trace(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,
    ) -> Self {
        Self::new_ssr_modes(
            device, format, uniform_layout, ssr_scene_layout, false, false, SsrPath::Trace,
        )
    }

    pub fn new_ssr_composite(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,
    ) -> Self {
        Self::new_ssr_modes(
            device, format, uniform_layout, ssr_scene_layout, false, false, SsrPath::Composite,
        )
    }

    fn new_ssr_variant(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,
        sources: bool,
        ssr_debug: bool,
    ) -> Self {
        Self::new_ssr_modes(
            device, format, uniform_layout, ssr_scene_layout, sources, ssr_debug, SsrPath::Inline,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_ssr_modes(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,
        sources: bool,
        ssr_debug: bool,
        path: SsrPath,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("brush_ssr_shader"),
            source: ShaderSource::Wgsl(
                brush_shader_modes(true, sources, ssr_debug, path).into(),
            ),
        });
        let material_layout = brush_material_bind_group_layout(device);
        let lightmap_layout = super::pipeline::lightmap_bind_group_layout(device);
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("brush_ssr_layout"),
            bind_group_layouts: &[
                Some(uniform_layout), Some(&material_layout), Some(&lightmap_layout), Some(ssr_scene_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("brush_ssr_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(BrushVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::LessEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),

            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, material_layout, lightmap_layout, reads_immediates: false }
    }

    fn new_with_front_face(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
        sources: bool,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        Self::new_blended(
            device, format, uniform_layout, front_face, samples, sources,
            Some(BlendState::ALPHA_BLENDING), view,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_blended(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
        sources: bool,
        blend: Option<BlendState>,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        Self::new_variant(
            device, format, uniform_layout, front_face, samples, sources, blend, view, BrushProbe::Trace, None,
        )
    }

    pub fn new_probe_pass(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        Self::new_variant(device, probe_pass::FORMAT, uniform_layout, FrontFace::Ccw, 1, false, None, view, BrushProbe::Pass, None)
    }

    pub fn new_probe_pass_deferred(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> Self {
        Self::probe_pass_deferred_from(device, uniform_layout, fixups, "brush_probe_pass_deferred", |source| source)
    }

    pub fn new_probe_pass_deferred_poolless(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> Self {
        Self::probe_pass_deferred_from(
            device,
            uniform_layout,
            fixups,
            "brush_probe_pass_deferred_poolless",
            crate::renderer::lights::without_pool_maps,
        )
    }

    fn probe_pass_deferred_from(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        label: &str,
        edit: fn(String) -> String,
    ) -> Self {
        let mut source = edit(brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::PassDeferred));
        if device.features().contains(wgpu::Features::SHADER_EARLY_DEPTH_TEST) {
            assert!(source.contains(FS_MAIN), "the fragment entry point moved");
            source = source.replacen(FS_MAIN, "@fragment @early_depth_test(force) fn fs_main(", 1);
        }
        Self::from_source(
            device,
            probe_pass::FORMAT,
            uniform_layout,
            FrontFace::Ccw,
            1,
            None,
            crate::renderer::multiview::ViewMode::Mono,
            Some(fixups.pass_layout()),
            label,
            with_push_scan(device, source),
        )
    }

    pub fn new_probe_pass_deferred_with_cut(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        cut: &str,
    ) -> Option<Self> {
        let (label, edits) = DEFERRED_REGISTER_CUTS.iter().find(|(label, _)| *label == cut)?;
        let mut src = brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::PassDeferred);
        for (from, to) in edits.iter() {
            if !src.contains(from) {
                return None;
            }
            src = src.replacen(from, to, 1);
        }
        if device.features().contains(wgpu::Features::SHADER_EARLY_DEPTH_TEST) {
            src = src.replacen(FS_MAIN, "@fragment @early_depth_test(force) fn fs_main(", 1);
        }

        if *label != SCAN_FROM_UNIFORMS {
            src = with_push_scan(device, src);
        }
        Some(Self::from_source(
            device,
            probe_pass::FORMAT,
            uniform_layout,
            FrontFace::Ccw,
            1,
            None,
            crate::renderer::multiview::ViewMode::Mono,
            Some(fixups.pass_layout()),
            label,
            src,
        ))
    }

    pub fn new_multisampled_probe_reader(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        Self::new_variant(
            device,
            format,
            uniform_layout,
            FrontFace::Ccw,
            samples,
            BRUSH_SOURCE_DEBUG,
            None,
            view,
            BrushProbe::Read,
            Some(probe_layout),
        )
    }

    pub fn new_spotless_probe_readers(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
    ) -> [Self; 3] {
        let read = brush_shader_probe(false, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Read);
        let spotless = crate::renderer::lights::without_spot_shadows;
        let build = |label: &str, source: String| {
            Self::from_source(
                device,
                format,
                uniform_layout,
                FrontFace::Ccw,
                samples,
                None,
                crate::renderer::multiview::ViewMode::Mono,
                Some(probe_layout),
                label,
                spotless(source),
            )
        };
        [
            build("brush_pipeline_read_spotless", read.clone()),
            build("brush_pipeline_read_sunless_spotless", sun_reader_shader(read.clone(), FaceSun::Never)),
            build("brush_pipeline_read_baked_spotless", sun_reader_shader(read, FaceSun::Baked)),
        ]
    }

    pub fn new_multisampled_probe_reader_for(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        view: crate::renderer::multiview::ViewMode,
        class: FaceSun,
    ) -> Self {
        let source = sun_reader_shader(
            brush_shader_probe(false, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Read),
            class,
        );
        let label = match class {
            FaceSun::Never => "brush_pipeline_read_sunless",
            FaceSun::Baked => "brush_pipeline_read_baked",
            FaceSun::Unbaked => "brush_pipeline_read",
        };
        Self::from_source(device, format, uniform_layout, FrontFace::Ccw, samples, None, view, Some(probe_layout), label, source)
    }

    pub fn new_edited_probe_readers(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        edit: &str,
    ) -> Option<[Self; 6]> {
        let (edit, only) = edit.split_once('@').map_or((edit, None), |(e, c)| (e, Some(c)));
        let shipped = brush_shader_probe(false, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Read);
        let edited = with_reader_edit(shipped.clone(), edit)?;
        let class = |name: &str| if only.map_or(true, |c| c == name) { edited.clone() } else { shipped.clone() };
        let spotless = crate::renderer::lights::without_spot_shadows;
        let build = |label: &str, source: String| {
            Self::from_source(
                device,
                format,
                uniform_layout,
                FrontFace::Ccw,
                samples,
                None,
                crate::renderer::multiview::ViewMode::Mono,
                Some(probe_layout),
                label,
                source,
            )
        };
        Some([
            build("brush_pipeline_read_edited", class("full")),
            build("brush_pipeline_read_sunless_edited", sun_reader_shader(class("sunless"), FaceSun::Never)),
            build("brush_pipeline_read_baked_edited", sun_reader_shader(class("baked"), FaceSun::Baked)),
            build("brush_pipeline_read_spotless_edited", spotless(class("full"))),
            build("brush_pipeline_read_sunless_spotless_edited", spotless(sun_reader_shader(class("sunless"), FaceSun::Never))),
            build("brush_pipeline_read_baked_spotless_edited", spotless(sun_reader_shader(class("baked"), FaceSun::Baked))),
        ])
    }

    #[allow(clippy::too_many_arguments)]
    fn new_variant(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
        sources: bool,
        blend: Option<BlendState>,
        view: crate::renderer::multiview::ViewMode,
        probe: BrushProbe,
        probe_layout: Option<&BindGroupLayout>,
    ) -> Self {
        let mut source = brush_shader_probe(false, sources, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, probe);
        if probe == BrushProbe::PassDeferred && device.features().contains(wgpu::Features::SHADER_EARLY_DEPTH_TEST) {
            assert!(source.contains(FS_MAIN), "the fragment entry point moved");
            source = source.replacen(FS_MAIN, "@fragment @early_depth_test(force) fn fs_main(", 1);
        }

        let label = match probe {
            BrushProbe::Trace => "brush_pipeline",
            BrushProbe::Pass => "brush_probe_pass",
            BrushProbe::PassDeferred => "brush_probe_pass_deferred",
            BrushProbe::Read => "brush_pipeline_read",
        };
        Self::from_source(device, format, uniform_layout, front_face, samples, blend, view, probe_layout, label, source)
    }

    pub fn log_probe_pass_register_cuts(device: &Device, uniform_layout: &BindGroupLayout) {
        let base = brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Pass);
        for (label, edits) in PROBE_PASS_REGISTER_CUTS {
            let mut src = base.clone();
            let mut missing = None;
            for (from, to) in edits.iter() {
                if !src.contains(from) {
                    missing = Some(*from);
                    break;
                }
                src = src.replacen(from, to, 1);
            }
            match missing {
                Some(from) => log::warn!("register cut {label}: `{from}` is not in the shader"),
                None => {
                    let _ = Self::from_source(
                        device,
                        probe_pass::FORMAT,
                        uniform_layout,
                        FrontFace::Ccw,
                        1,
                        None,
                        crate::renderer::multiview::ViewMode::Mono,
                        None,
                        label,
                        src,
                    );
                }
            }
        }
    }

    pub fn new_multisampled_probe_reader_with_cut(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        cut: &str,
    ) -> Option<Self> {
        let (label, edits) = SCENE_REGISTER_CUTS.iter().find(|(label, _)| *label == cut)?;
        let mut src = brush_shader_probe(false, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Read);
        for (from, to) in edits.iter() {
            if !src.contains(from) {
                return None;
            }
            src = src.replacen(from, to, 1);
        }
        Some(Self::from_source(
            device,
            format,
            uniform_layout,
            FrontFace::Ccw,
            samples,
            None,
            crate::renderer::multiview::ViewMode::Mono,
            Some(probe_layout),
            label,
            src,
        ))
    }

    pub fn log_scene_register_cuts(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
    ) {
        let base = brush_shader_probe(false, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Read);
        for (label, edits) in SCENE_REGISTER_CUTS {
            let mut src = base.clone();
            let mut missing = None;
            for (from, to) in edits.iter() {
                if !src.contains(from) {
                    missing = Some(*from);
                    break;
                }
                src = src.replacen(from, to, 1);
            }
            match missing {
                Some(from) => log::warn!("register cut {label}: `{from}` is not in the shader"),
                None => {
                    let _ = Self::from_source(
                        device,
                        format,
                        uniform_layout,
                        FrontFace::Ccw,
                        samples,
                        None,
                        crate::renderer::multiview::ViewMode::Mono,
                        Some(probe_layout),
                        label,
                        src,
                    );
                }
            }
        }
    }

    pub fn log_deferred_register_cuts(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) {
        let base = brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::PassDeferred);
        for (label, edits) in DEFERRED_REGISTER_CUTS {
            let mut src = base.clone();
            let mut missing = None;
            for (from, to) in edits.iter() {
                if !src.contains(from) {
                    missing = Some(*from);
                    break;
                }
                src = src.replacen(from, to, 1);
            }
            if *label != SCAN_FROM_UNIFORMS {
                src = with_push_scan(device, src);
            }
            match missing {
                Some(from) => log::warn!("register cut {label}: `{from}` is not in the shader"),
                None => {
                    let _ = Self::from_source(
                        device,
                        probe_pass::FORMAT,
                        uniform_layout,
                        FrontFace::Ccw,
                        1,
                        None,
                        crate::renderer::multiview::ViewMode::Mono,
                        Some(fixups.pass_layout()),
                        label,
                        src,
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]

    pub fn log_opts(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) {
        for (base, pass, src) in opt_bases() {
            for opt in crate::renderer::shader_opt::OPTS {
                let Some(mut source) = (opt.edit)(&src) else {
                    continue;
                };
                if pass {
                    source = with_push_scan(device, source);
                }
                let label = format!("opt_{base}_{}", opt.label);
                let _ = Self::from_source(
                    device,
                    if pass { probe_pass::FORMAT } else { format },
                    uniform_layout,
                    FrontFace::Ccw,
                    if pass { 1 } else { samples },
                    None,
                    crate::renderer::multiview::ViewMode::Mono,
                    Some(if pass { fixups.pass_layout() } else { probe_layout }),
                    &label,
                    source,
                );
            }
        }
    }

    fn from_source(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
        blend: Option<BlendState>,
        view: crate::renderer::multiview::ViewMode,
        probe_layout: Option<&BindGroupLayout>,
        label: &str,
        source: String,
    ) -> Self {

        let reaches = source.contains(probe_pass::OUTPUT_SIGNATURE);

        let reads_immediates = source.contains(IMMEDIATE_DECLARATION);
        let immediate_size = if reads_immediates { crate::renderer::uniforms::PROBE_PUSH_SIZE as u32 } else { 0 };

        let shader = crate::renderer::shader_checks::audited_shader_module(device, ShaderModuleDescriptor {
            label: Some("brush_shader"),
            source: ShaderSource::Wgsl(crate::renderer::shader_precision::for_device(device, view.shader(source)).into()),
        });
        let material_layout = brush_material_bind_group_layout(device);

        let lightmap_layout = super::pipeline::lightmap_bind_group_layout(device);
        let mut layouts = vec![Some(uniform_layout), Some(&material_layout), Some(&lightmap_layout)];
        if let Some(l) = probe_layout {
            layouts.push(Some(l));
        }
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("brush_layout"),
            bind_group_layouts: &layouts,
            immediate_size,
        });
        let probe_pass_targets = probe_pass::targets(true);
        let plain_target = [Some(ColorTargetState { format, blend, write_mask: ColorWrites::ALL })];
        let targets: &[Option<ColorTargetState>] =
            if reaches { &probe_pass_targets } else { &plain_target };
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(BrushVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets,
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                front_face,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),

                depth_compare: Some(CompareFunction::LessEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });
        Self { pipeline, material_layout, lightmap_layout, reads_immediates }
    }

    pub fn new_depth_prepass(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        let targets = [Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::empty() })];
        Self::depth_prepass_into(device, uniform_layout, samples, view, &targets, DepthBiasState::default(), "brush_depth_prepass")
    }

    pub fn new_probe_depth_prepass(device: &Device, uniform_layout: &BindGroupLayout) -> Self {
        let targets = probe_pass::targets(true).map(|t| t.map(|t| ColorTargetState { write_mask: ColorWrites::empty(), ..t }));
        Self::depth_prepass_into(
            device,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            &targets,
            PROBE_PREPASS_BIAS,
            "brush_probe_depth_prepass",
        )
    }

    fn depth_prepass_into(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        targets: &[Option<ColorTargetState>],
        bias: DepthBiasState,
        label: &str,
    ) -> Self {

        let shader = crate::renderer::shader_checks::audited_shader_module(device, ShaderModuleDescriptor {
            label: Some("brush_depth_prepass"),
            source: ShaderSource::Wgsl(
                view.shader(brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Trace))
                    .into(),
            ),
        });
        let material_layout = brush_material_bind_group_layout(device);
        let lightmap_layout = super::pipeline::lightmap_bind_group_layout(device);
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("brush_depth_prepass_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(&material_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(BrushVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_depth"),
                compilation_options: PipelineCompilationOptions::default(),
                targets,
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: StencilState::default(),
                bias,
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });
        Self { pipeline, material_layout, lightmap_layout, reads_immediates: false }
    }
}

pub const PROBE_PREPASS_BIAS: DepthBiasState = DepthBiasState { constant: 16, slope_scale: 2.0, clamp: 0.0 };

pub const SEAL_BRUSH_CRACKS: bool = true;

pub const CRACK_SEAL_COLOUR: [f32; 3] = [0.02, 0.02, 0.05];

/// THE SEAL NEVER TIES A FACE. Behind a convex edge whose other face is turned
/// away, the seal draws that face's back over the same pixels at all but the
/// front face's depth, and rounding let it win in dots along the edge: grey
/// specks down door jambs and building corners, more on the headset than on a
/// desktop GPU (2026-10-20). Pushed back, it still fills a real crack, where
/// only open space lies in front of it.
pub const SEAL_DEPTH_BIAS: DepthBiasState = DepthBiasState { constant: 16, slope_scale: 2.0, clamp: 0.0 };

pub struct BrushSealPipeline {
    pub pipeline: RenderPipeline,
}

impl BrushSealPipeline {
    pub fn new(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    pub fn new_stereo(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Stereo)
    }

    /// The seal in `colour` instead of `CRACK_SEAL_COLOUR`: one no shading
    /// can produce (magenta) makes every crack in the brushes countable in a
    /// frame, since only a hole in the surface shows the seal (2026-10-20).
    pub fn new_coloured(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout, samples: u32, colour: [f32; 3]) -> Self {
        Self::build(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Mono, brush_seal_shader_in(colour))
    }

    fn new_with_view(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        Self::build(device, format, uniform_layout, samples, view, brush_seal_shader())
    }

    fn build(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        source: String,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("brush_seal_shader"),
            source: ShaderSource::Wgsl(view.shader(source).into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("brush_seal_layout"),
            bind_group_layouts: &[Some(uniform_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("brush_seal_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(BrushVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),

                targets: &[Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::ALL })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,

                cull_mode: Some(Face::Front),
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: StencilState::default(),
                bias: SEAL_DEPTH_BIAS,
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });
        Self { pipeline }
    }
}

fn brush_seal_shader() -> String {
    brush_seal_shader_in(CRACK_SEAL_COLOUR)
}

fn brush_seal_shader_in([r, g, b]: [f32; 3]) -> String {
    let src = brush_seal_shader_posed(r, g, b);
    if WORLD_BRUSHES.load(std::sync::atomic::Ordering::Relaxed) {
        debug_assert_eq!(src.matches(POSED_SEAL_VERTEX).count(), 1);
        return src.replacen(POSED_SEAL_VERTEX, WORLD_SEAL_VERTEX, 1);
    }
    src
}

fn brush_seal_shader_posed(r: f32, g: f32, b: f32) -> String {
    format!(
        r#"
{lights_block}

@vertex fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) @invariant vec4<f32> {{
    return cam_view_proj() * vec4<f32>(pos, 1.0);
}}

@fragment fn fs_main() -> @location(0) vec4<f32> {{
    return vec4<f32>({r:?}, {g:?}, {b:?}, 1.0);
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
    )
}

fn brush_shader() -> String {
    brush_shader_with(false)
}

pub const SUN_MASK_DISTANCE_TEXELS: f32 = 4.0;

pub const STATIONARY_MASK_DISTANCE_TEXELS: f32 = 4.0;

pub struct SunFaces {

    reachable: Vec<u32>,
    unbaked: Vec<u32>,
    width: u32,
    height: u32,

    serial: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceSun {

    Never,

    Baked,

    Unbaked,
}

pub const SUN_MASK_READ_REACH: u32 = 32;

impl SunFaces {

    pub fn from_mask(rgba: &[u8], width: u32, height: u32) -> Option<Self> {
        let (w, h) = (width as usize, height as usize);
        if w == 0 || h == 0 || rgba.len() < w * h * 4 {
            return None;
        }
        let mut reachable = vec![0u32; (w + 1) * (h + 1)];
        let mut unbaked = vec![0u32; (w + 1) * (h + 1)];
        for y in 0..h {
            let (mut reach_row, mut unbaked_row) = (0u32, 0u32);
            for x in 0..w {
                let t = &rgba[(y * w + x) * 4..(y * w + x) * 4 + 2];
                let baked = t[1] >= 128;
                reach_row += u32::from(!(baked && t[0] == 0));
                unbaked_row += u32::from(!baked);
                let (above, here) = (y * (w + 1) + x + 1, (y + 1) * (w + 1) + x + 1);
                reachable[here] = reachable[above] + reach_row;
                unbaked[here] = unbaked[above] + unbaked_row;
            }
        }
        static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Some(Self { reachable, unbaked, width, height, serial: SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed) })
    }

    pub fn serial(&self) -> u64 {
        self.serial
    }

    pub fn class_of(&self, uv2_rect: [f32; 4]) -> FaceSun {
        if !uv2_rect.iter().all(|c| c.is_finite()) {
            return FaceSun::Unbaked;
        }
        let (w, h) = (i64::from(self.width), i64::from(self.height));
        let border = i64::from(SUN_MASK_READ_REACH);
        let x0 = ((uv2_rect[0] * w as f32).floor() as i64 - border).clamp(0, w);
        let y0 = ((uv2_rect[1] * h as f32).floor() as i64 - border).clamp(0, h);
        let x1 = ((uv2_rect[2] * w as f32).ceil() as i64 + border).clamp(0, w);
        let y1 = ((uv2_rect[3] * h as f32).ceil() as i64 + border).clamp(0, h);
        if x1 <= x0 || y1 <= y0 {
            return FaceSun::Unbaked;
        }
        let sum = |t: &[u32]| {
            let at = |x: i64, y: i64| i64::from(t[(y * (w + 1) + x) as usize]);
            at(x1, y1) - at(x0, y1) - at(x1, y0) + at(x0, y0)
        };
        if sum(&self.reachable) == 0 {
            FaceSun::Never
        } else if sum(&self.unbaked) == 0 {
            FaceSun::Baked
        } else {
            FaceSun::Unbaked
        }
    }

    pub fn partition(&self, vertices: &[BrushVertex], indices: &[u32]) -> (Vec<u32>, [u32; 2]) {
        let mut parts: [Vec<u32>; 3] = [Vec::with_capacity(indices.len()), Vec::new(), Vec::new()];

        let mut last: Option<([f32; 4], FaceSun)> = None;
        for tri in indices.chunks_exact(3) {
            let rect = vertices.get(tri[0] as usize).map_or([f32::NAN; 4], |v| v.uv2_rect);
            let class = match last {
                Some((r, c)) if r == rect => c,
                _ => {
                    let c = self.class_of(rect);
                    last = Some((rect, c));
                    c
                }
            };
            parts[class as usize].extend_from_slice(tri);
        }
        let [mut out, baked, rest] = parts;
        let never_end = out.len() as u32;
        out.extend_from_slice(&baked);
        let baked_end = out.len() as u32;
        out.extend_from_slice(&rest);
        (out, [never_end, baked_end])
    }
}

pub fn dilate_sun_mask(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let mut px = rgba.to_vec();
    if w == 0 || h == 0 || px.len() < w * h * 4 {
        return px;
    }
    let reach = SUN_MASK_READ_REACH as usize;
    let filled = |px: &[u8], i: usize| px[i * 4 + 1] >= 128;

    let mut left = vec![usize::MAX; w];
    let mut sources = Vec::new();
    for y in 0..h {
        let row = y * w;
        let mut last = usize::MAX;
        for x in 0..w {
            if filled(&px, row + x) {
                last = x;
            }
            left[x] = last;
        }
        let mut next = usize::MAX;
        sources.clear();
        for x in (0..w).rev() {
            if filled(&px, row + x) {
                next = x;
                continue;
            }
            let from_left = (left[x] != usize::MAX).then(|| x - left[x]);
            let from_right = (next != usize::MAX).then(|| next - x);
            let source = match (from_left, from_right) {
                (Some(l), Some(r)) if l <= r => Some((l, left[x])),
                (_, Some(r)) => Some((r, next)),
                (Some(l), None) => Some((l, left[x])),
                (None, None) => None,
            };
            if let Some((_, s)) = source.filter(|(d, _)| *d <= reach) {
                sources.push((x, s));
            }
        }
        for &(x, s) in &sources {
            px.copy_within((row + s) * 4..(row + s) * 4 + 4, (row + x) * 4);
        }
    }

    let mut above = vec![u8::MAX; w * h];
    let mut last = vec![usize::MAX; w];
    for y in 0..h {
        for x in 0..w {
            if filled(&px, y * w + x) {
                last[x] = y;
            }
            if last[x] != usize::MAX {
                above[y * w + x] = (y - last[x]).min(usize::from(u8::MAX)) as u8;
            }
        }
    }
    let mut below = vec![usize::MAX; w];
    for y in (0..h).rev() {
        for x in 0..w {
            let i = y * w + x;
            if filled(&px, i) {
                below[x] = y;
                continue;
            }
            let up = usize::from(above[i]);
            let down = if below[x] == usize::MAX { usize::MAX } else { below[x] - y };
            let source_row = if up <= down && up <= reach {
                y - up
            } else if down <= reach {
                y + down
            } else {
                continue;
            };
            let s = source_row * w + x;
            px.copy_within(s * 4..s * 4 + 4, i * 4);
        }
    }
    px
}

#[cfg(test)]
pub(crate) fn audit_sources() -> Vec<(&'static str, String)> {
    let read = brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Read);
    vec![
        ("brush_read_baked_spotless", crate::renderer::lights::without_spot_shadows(sun_reader_shader(read, FaceSun::Baked))),
        ("brush_probe_pass_deferred", brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::PassDeferred)),
    ]
}

pub(crate) fn opt_bases() -> Vec<(&'static str, bool, String)> {
    let read = brush_shader_probe(false, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Read);
    vec![
        ("bread", false, crate::renderer::lights::without_spot_shadows(sun_reader_shader(read.clone(), FaceSun::Baked))),
        ("bfull", false, read),
        ("bpass", true, brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::PassDeferred)),
    ]
}

pub(crate) fn sun_reader_shader(source: String, class: FaceSun) -> String {
    let switch = match class {
        FaceSun::Never => "SKY_SUN_NEVER_REACHES",
        FaceSun::Baked => "SUN_MASK_EVERYWHERE",
        FaceSun::Unbaked => return source,
    };
    let off = format!("const {switch}: bool = false;");
    assert_eq!(source.matches(off.as_str()).count(), 1, "the lights block's {switch} switch moved");
    source.replacen(&off, &format!("const {switch}: bool = true;"), 1)
}

fn brush_shader_with(ssr: bool) -> String {
    brush_shader_variant(ssr, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG)
}

const IMMEDIATE_DECLARATION: &str =
    "const PROBE_MAX_SLOTS: i32 = 16;\nvar<immediate> probe_push: array<vec4<f32>, 16>;\n";

const PUSH_SCAN: [(&str, &str); 2] = [
    ("const PROBE_MAX_SLOTS: i32 = 16;\n", IMMEDIATE_DECLARATION),
    (
        "        let room = camera.probe_boxes[i * 3 + 2].w;\n        let v = camera.probe_boxes[i * 3].xyz - p;",
        "        let room = probe_push[i].w;\n        let v = probe_push[i].xyz - p;",
    ),
];

fn push_scan(src: &str) -> Option<String> {
    if !PUSH_SCAN.iter().all(|(from, _)| src.contains(from)) {
        return None;
    }
    Some(PUSH_SCAN.iter().fold(src.to_string(), |s, (from, to)| s.replacen(from, to, 1)))
}

fn with_push_scan(device: &Device, src: String) -> String {
    if !device.features().contains(wgpu::Features::IMMEDIATES) {
        return src;
    }
    push_scan(&src).unwrap_or(src)
}

const SCAN_FROM_UNIFORMS: &str = "def_scan_uniform";

const PROBE_TRACE_CALL: &str = "    return probe_trace_skipping(world_pos, d, room, roughness, -1);";
const FOVEA_30: &str = "    let fwd = normalize(vec3<f32>(cam_view_proj()[0].w, cam_view_proj()[1].w, cam_view_proj()[2].w));\n    let fovea = dot(normalize(to_player_space(world_pos) - cam_pos()), fwd) > 0.866;\n    return probe_trace_skipping(world_pos, d, room, select(2.0, roughness, fovea), -1);";
const FOVEA_45: &str = "    let fwd = normalize(vec3<f32>(cam_view_proj()[0].w, cam_view_proj()[1].w, cam_view_proj()[2].w));\n    let fovea = dot(normalize(to_player_space(world_pos) - cam_pos()), fwd) > 0.707;\n    return probe_trace_skipping(world_pos, d, room, select(2.0, roughness, fovea), -1);";

const DEFERRED_REGISTER_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("def_cut_none", &[]),

    (
        "def_glass_any_way",
        &[("let into = select(1.0, s * s * (3.0 - 2.0 * s), lights.lights[i].params.z > 0.5);", "let into = 1.0;")],
    ),
    ("def_cut_rim_detect", &[("        if (hit.rim < 0.0 && spread > PROBE_RIM_MIN_SPREAD) {", "        if (false) {")]),
    ("def_cut_edge_detect", &[("        if (hit.edge_code < 0 && proxy.edge >= 0) {", "        if (false) {")]),
    (
        "def_cut_both_detect",
        &[
            ("        if (hit.rim < 0.0 && spread > PROBE_RIM_MIN_SPREAD) {", "        if (false) {"),
            ("        if (hit.edge_code < 0 && proxy.edge >= 0) {", "        if (false) {"),
        ],
    ),
    (
        "def_cut_record",
        &[("            slot = probe_fixup_begin(hit, world_pos, d, dir, roughness, probe_lod, trace_room, recolour);", "            slot = -1;")],
    ),

    (
        "def_cut_relight",
        &[(crate::renderer::lights::SURFACE_RELIT_READ, ""), (crate::renderer::lights::SURFACE_RELIT_CALL, "")],
    ),

    ("def_cards_in_pass", &[("    if (model && !PROBE_SECONDARY_DEFERRED) {", "    if (model) {")]),

    ("def_rolled_trace", &[(ROLL_TRACE_FROM, ROLL_TRACE_TO)]),
    ("def_rolled_proxy", &[(ROLL_SAMPLES_FROM, ROLL_SAMPLES_TO), (ROLL_FIELD_FROM, ROLL_FIELD_TO)]),
    ("def_rolled_ground", &[(ROLL_GROUND_FROM, ROLL_GROUND_TO), (ROLL_READINGS_FROM, ROLL_READINGS_TO)]),
    (
        "def_rolled_all",
        &[
            (ROLL_TRACE_FROM, ROLL_TRACE_TO),
            (ROLL_SAMPLES_FROM, ROLL_SAMPLES_TO),
            (ROLL_FIELD_FROM, ROLL_FIELD_TO),
            (ROLL_GROUND_FROM, ROLL_GROUND_TO),
            (ROLL_READINGS_FROM, ROLL_READINGS_TO),
        ],
    ),

    (
        "def_cards_before",
        &[
            ("    if (model && !PROBE_SECONDARY_DEFERRED) {", "    if (model) {"),
            ("        let recolour = PROBE_SECONDARY_DEFERRED && probe_hit_carded(hit);", "        let recolour = false;"),
        ],
    ),
    ("def_cut_proxy_surface", &[("    if (s0 < 0) {\n        return 3.4e38;\n    }", "    if (true) {\n        return 3.4e38;\n    }")]),
    ("def_cut_escape_colour", &[("    if (h.escaped) {\n        col = probe_escape_colour(", "    if (false) {\n        col = probe_escape_colour(")]),
    ("def_cut_untraced", &[("    return probe_through_portals(own, own_room, select_world, world_pos, d, probe_lod);", "    return own;")]),

    (
        "def_cut_floor_mirror",
        &[(
            "    if (probe_floor_mirror_here) {\n        probe = probe_floor_mirror_pass(probe, roughness);\n    } else {\n        probe = capsule_reflection(world_pos, refl, roughness, env, probe);\n    }",
            "    probe = capsule_reflection(world_pos, refl, roughness, env, probe);",
        )],
    ),
    (
        "def_cut_characters",
        &[(
            "    if (probe_floor_mirror_here) {\n        probe = probe_floor_mirror_pass(probe, roughness);\n    } else {\n        probe = capsule_reflection(world_pos, refl, roughness, env, probe);\n    }",
            "",
        )],
    ),
    (
        "def_cut_reach",
        &[(
            "    return ProbePassOut(reflection, vec2<f32>(probe_reach, probe_face_code(n_geom)));",
            "    return ProbePassOut(reflection, vec2<f32>(0.0, probe_face_code(n_geom)));",
        )],
    ),

    ("def_cut_trace", &[("    let hit = probe_trace(world_pos, d, trace_room, roughness);", STAGE_CUT_HIT)]),
    // FOVEATED TRACE LOD (the AAA doc's rank 5): no trace past 30 or 45
    // degrees off the eye's axis, where the density map already shades
    // coarsely; those pixels take the probe cube as rough ones do. The
    // trace's position is in the world; the eye's in the player's frame.
    ("def_cut_fovea30", &[(PROBE_TRACE_CALL, FOVEA_30)]),
    ("def_cut_fovea45", &[(PROBE_TRACE_CALL, FOVEA_45)]),
    // THE TRACE ONLY ON SMOOTH SURFACES (the AAA doc's rank 2, from
    // Frostbite's < 0.2): past these roughnesses a surface reads the probe
    // cube through its box instead of walking the rooms.
    ("def_cut_trace_max05", &[("const PROBE_TRACE_MAX_ROUGHNESS: f32 = 0.75;", "const PROBE_TRACE_MAX_ROUGHNESS: f32 = 0.5;")]),
    ("def_cut_trace_max03", &[("const PROBE_TRACE_MAX_ROUGHNESS: f32 = 0.75;", "const PROBE_TRACE_MAX_ROUGHNESS: f32 = 0.3;")]),
    (
        "def_cut_colour",
        &[(
            "        let col = probe_traced_colour(hit, d, roughness, dir, probe_lod);",
            "        let col = vec4<f32>(hit.pos * hit.t, hit.room + hit.other);",
        )],
    ),
    (
        "def_cut_trace_colour",
        &[
            ("    let hit = probe_trace(world_pos, d, trace_room, roughness);", STAGE_CUT_HIT),
            (
                "        let col = probe_traced_colour(hit, d, roughness, dir, probe_lod);",
                "        let col = vec4<f32>(hit.pos * hit.t, hit.room + hit.other);",
            ),
        ],
    ),
    (
        "def_cut_all_but_preamble",
        &[(
            "    let reflection = probe_env_for_pass(face_pos, n, rough_aa, ao, baked.a, baked.rgb, face_pos, n_geom);",
            "    let reflection = vec4<f32>(face_pos * rough_aa, ao + baked.a + dot(baked.rgb, n));",
        )],
    ),

    (
        "def_hit_serial",
        &[
            ("    let depth1 = textureSampleLevel(probe_depth, probe_depth_samp, v1, i32(b1.w), 0.0);\n", ""),
            ("        c1 = probe_clearance_of(depth1, v1);", "        c1 = probe_clearance(s1, h);"),
            (
                "    let col0 = textureSampleLevel(probe_cube, probe_samp, v0, i32(b0.w), lod0);\n    let col1 = textureSampleLevel(probe_cube, probe_samp, v1, i32(b1.w), lod1);\n",
                "",
            ),
            (
                "    var col = col0 * w0;\n    if (w1 > 0.0) {\n        col += col1 * w1;\n    }",
                "    var col = textureSampleLevel(probe_cube, probe_samp, v0, i32(b0.w), lod0) * w0;\n    if (w1 > 0.0) {\n        col += textureSampleLevel(probe_cube, probe_samp, v1, i32(b1.w), lod1) * w1;\n    }",
            ),
        ],
    ),

    (
        "def_hit_walk",
        &[(
            "    let near = probe_nearest_two(h, room, other);
    let s0 = near.s0;
    let s1 = near.s1;
    let d0 = near.d0;
    let d1 = near.d1;",
            "    var s0 = -1;
    var s1 = -1;
    var d0 = 3.4e38;
    var d1 = 3.4e38;
    var next_a = probe_room_slot(room);
    var next_b = select(-1, probe_room_slot(other), other != room);
    loop {
        if (next_a < 0 && next_b < 0) {
            break;
        }
        var i = next_a;
        if (next_b >= 0 && (next_a < 0 || next_b < next_a)) {
            i = next_b;
            next_b = probe_slot_next(next_b);
        } else {
            next_a = probe_slot_next(next_a);
        }
        let v = camera.probe_boxes[i * 3].xyz - h;
        let dd = dot(v, v);
        if (dd < d0) {
            s1 = s0;
            d1 = d0;
            s0 = i;
            d0 = dd;
        } else if (dd < d1) {
            s1 = i;
            d1 = dd;
        }
    }",
        )],
    ),

    (
        "def_scan_unrolled",
        &[("    let near = probe_nearest_two(h, room, other);", "    let near = probe_nearest_two_unrolled(h, room, other);")],
    ),

    (
        "def_scan_const",
        &[("    let near = probe_nearest_two(h, room, other);", "    let near = probe_nearest_two_const(h, room, other);")],
    ),

    (
        "def_scan_const_unrolled",
        &[("    let near = probe_nearest_two(h, room, other);", "    let near = probe_nearest_two_const_unrolled(h, room, other);")],
    ),

    (SCAN_FROM_UNIFORMS, &[]),

    (
        "def_scan_noload",
        &[(
            "        let room = camera.probe_boxes[i * 3 + 2].w;\n        let v = camera.probe_boxes[i * 3].xyz - p;",
            "        let room = a;\n        let v = vec3<f32>(f32(i)) - p;",
        )],
    ),

    (
        "def_scan_flat",
        &[(
            "        if (i >= count) {\n            break;\n        }\n        let room = camera.probe_boxes[i * 3 + 2].w;",
            "        let room = select(-2.0, camera.probe_boxes[i * 3 + 2].w, i < count);",
        )],
    ),

    (
        "def_rough_shading_n",
        &[(
            "    let rough_aa = specular_aa_roughness(rough, dpdx(n_geom), dpdy(n_geom));",
            "    let rough_aa = specular_aa_roughness(rough, dpdx(n), dpdy(n));",
        )],
    ),

    (
        "def_cut_slot_walk",
        &[(
            "    let near = probe_nearest_two(h, room, other);",
            "    var near: ProbeNearest;
    near.s0 = probe_room_slot(room);
    near.s1 = select(-1, probe_slot_next(near.s0), near.s0 >= 0);
    let cut_v0 = camera.probe_boxes[near.s0 * 3].xyz - h;
    let cut_v1 = camera.probe_boxes[max(near.s1, 0) * 3].xyz - h;
    near.d0 = dot(cut_v0, cut_v0);
    near.d1 = select(3.4e38, dot(cut_v1, cut_v1), near.s1 >= 0);",
        )],
    ),

    (
        "def_cut_hit_depth",
        &[(
            "    let depth0 = textureSampleLevel(probe_depth, probe_depth_samp, v0, i32(b0.w), 0.0);\n    let depth1 = textureSampleLevel(probe_depth, probe_depth_samp, v1, i32(b1.w), 0.0);",
            "    let depth0 = vec4<f32>(-normalize(v0), -length(v0));\n    let depth1 = vec4<f32>(-normalize(v1), -length(v1));",
        )],
    ),

    ("def_face_code_early", &[B2_FACE_CODE]),

    ("def_dir_from_d", &[B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR]),
    ("def_dir_and_code", &[B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR]),

    ("def_cut_characters_record", &[B2_NO_CHARACTERS, B2_NO_RECORD]),

    ("def_lit_grey", &[B2_LIT_GREY, B2_CAPSULE_GREY]),
    ("def_dir_code_grey", &[B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR_GREY, B2_LIT_GREY]),

    ("def_lod_late", &[B2_LOD_EARLY, B2_LOD_LATE]),
    ("def_dir_code_lod", &[B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE]),

    ("def_fragment_packed", &[B2_FRAG_DECL, B2_FRAG_SET, B2_FRAG_MIRROR, B2_FRAG_RECORD]),
    (
        "def_b2_lossless",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE,
            B2_FRAG_DECL, B2_FRAG_SET, B2_FRAG_MIRROR, B2_FRAG_RECORD,
        ],
    ),

    (
        "def_dlc_volume_late",
        &[B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE, B2_VOLUME_LATE],
    ),

    (
        "def_dlc_miss_at_hit",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE,
            B2_MISS_CHOICE_AT_HIT, B2_MISS_PORTALS_AT_HIT,
        ],
    ),

    (
        "def_dlc_grey",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR_GREY, B2_LIT_GREY, B2_LOD_EARLY,
            B2_LOD_LATE,
        ],
    ),

    (
        "def_dlc_pos_late",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE,
            B2_POS_LATE_SET, B2_POS_LATE_RECORD, B2_POS_LATE_CHOICE, B2_POS_LATE_NEAR, B2_POS_LATE_FAR, B2_POS_LATE_PORTALS,
        ],
    ),

    (
        "def_dlc_lit_half",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE,
            B2_LIT_HALF_DECL, B2_LIT_HALF_SET, B2_LIT_HALF_RECORD, B2_LIT_HALF_CAPSULES,
        ],
    ),
    (
        "def_dlc_pos_lit_half",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE,
            B2_POS_LATE_SET, B2_POS_LATE_RECORD, B2_POS_LATE_CHOICE, B2_POS_LATE_NEAR, B2_POS_LATE_FAR, B2_POS_LATE_PORTALS,
            B2_LIT_HALF_DECL, B2_LIT_HALF_SET, B2_LIT_HALF_RECORD, B2_LIT_HALF_CAPSULES,
        ],
    ),

    (
        "def_b2_all_exact",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE,
            B2_FRAG_DECL, B2_FRAG_SET, B2_FRAG_MIRROR, B2_FRAG_RECORD, B2_POS_LATE_SET, B2_POS_LATE_RECORD, B2_POS_LATE_CHOICE,
            B2_POS_LATE_NEAR, B2_POS_LATE_FAR, B2_POS_LATE_PORTALS, B2_LIT_HALF_DECL, B2_LIT_HALF_SET, B2_LIT_HALF_RECORD,
            B2_LIT_HALF_CAPSULES,
        ],
    ),
    (
        "def_dlc_pos_late_grey",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR_GREY, B2_LIT_GREY, B2_LOD_EARLY,
            B2_LOD_LATE, B2_POS_LATE_SET, B2_POS_LATE_RECORD, B2_POS_LATE_CHOICE, B2_POS_LATE_NEAR, B2_POS_LATE_FAR,
            B2_POS_LATE_PORTALS,
        ],
    ),

    (
        "def_dlc_no_far_rim",
        &[B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE, B2_NO_FAR_RIM],
    ),
    (
        "def_dlc_no_rims",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE,
            B2_NO_FAR_RIM, B2_NO_NEAR_RIM,
        ],
    ),
    (
        "def_dlc_no_edges",
        &[B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE, B2_NO_EDGES],
    ),
    (
        "def_dlc_no_field_walk",
        &[B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE, B2_NO_FIELD_WALK],
    ),
    (
        "def_dlc_no_proxy_surface",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE,
            B2_NO_PROXY_SURFACE,
        ],
    ),
    (
        "def_dlc_no_proxies",
        &[B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE, B2_NO_PROXIES],
    ),
    (
        "def_dlc_no_doorway_start",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE,
            B2_NO_DOORWAY_START,
        ],
    ),
    (
        "def_dlc_no_portals",
        &[B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE, B2_NO_PORTALS],
    ),
    (
        "def_dlc_one_hop",
        &[B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE, B2_ONE_HOP],
    ),

    ("def_dedup_sky", &[DEDUP_OUTDOOR_SKY]),

    ("def_dedup_portals", &[DEDUP_PORTAL_SAMPLES]),

    ("def_dedup_near_far", &[DEDUP_NEAR_FAR]),
    ("def_dedup_all", DEDUP_EDITS),

    ("def_inv_per_use", &[B2_INV_FAR, B2_INV_SIDE, B2_INV_ENTER]),
    (
        "def_b2_inv_per_use",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE,
            B2_FRAG_DECL, B2_FRAG_SET, B2_FRAG_MIRROR, B2_FRAG_RECORD, B2_POS_LATE_SET, B2_POS_LATE_RECORD, B2_POS_LATE_CHOICE,
            B2_POS_LATE_NEAR, B2_POS_LATE_FAR, B2_POS_LATE_PORTALS, B2_LIT_HALF_DECL, B2_LIT_HALF_SET, B2_LIT_HALF_RECORD,
            B2_LIT_HALF_CAPSULES, B2_INV_FAR, B2_INV_SIDE, B2_INV_ENTER,
        ],
    ),
    (
        "def_b2_rim_reread",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE,
            B2_FRAG_DECL, B2_FRAG_SET, B2_FRAG_MIRROR, B2_FRAG_RECORD, B2_POS_LATE_SET, B2_POS_LATE_RECORD, B2_POS_LATE_CHOICE,
            B2_POS_LATE_NEAR, B2_POS_LATE_FAR, B2_POS_LATE_PORTALS, B2_LIT_HALF_DECL, B2_LIT_HALF_SET, B2_LIT_HALF_RECORD,
            B2_LIT_HALF_CAPSULES, B2_RIM_REREAD,
        ],
    ),
    (
        "def_b2_round3",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE,
            B2_FRAG_DECL, B2_FRAG_SET, B2_FRAG_MIRROR, B2_FRAG_RECORD, B2_POS_LATE_SET, B2_POS_LATE_RECORD, B2_POS_LATE_CHOICE,
            B2_POS_LATE_NEAR, B2_POS_LATE_FAR, B2_POS_LATE_PORTALS, B2_LIT_HALF_DECL, B2_LIT_HALF_SET, B2_LIT_HALF_RECORD,
            B2_LIT_HALF_CAPSULES, B2_INV_FAR, B2_INV_SIDE, B2_INV_ENTER, B2_RIM_REREAD,
        ],
    ),

    ("def_seen_texel_only", &[B2_SEEN_TEXEL_ONLY]),
    (
        "def_b2_seen_texel_only",
        &[
            B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE,
            B2_FRAG_DECL, B2_FRAG_SET, B2_FRAG_MIRROR, B2_FRAG_RECORD, B2_POS_LATE_SET, B2_POS_LATE_RECORD, B2_POS_LATE_CHOICE,
            B2_POS_LATE_NEAR, B2_POS_LATE_FAR, B2_POS_LATE_PORTALS, B2_LIT_HALF_DECL, B2_LIT_HALF_SET, B2_LIT_HALF_RECORD,
            B2_LIT_HALF_CAPSULES, B2_SEEN_TEXEL_ONLY,
        ],
    ),

    ("def_dlc_seen_uniform", &[B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE, B2_SEEN_UNIFORM]),
    ("def_dlc_no_side", &[B2_FACE_CODE, B2_RAY_DECL, B2_RAY_SET, B2_RECORD_DIR, B2_COLOUR_DIR, B2_CAPSULE_DIR, B2_LOD_EARLY, B2_LOD_LATE, B2_NO_SIDE]),

    ("def_dedup_sky_near_far", &[DEDUP_OUTDOOR_SKY, DEDUP_NEAR_FAR]),
];

pub(crate) const DEDUP_EDITS: &[(&str, &str)] = &[DEDUP_OUTDOOR_SKY, DEDUP_PORTAL_SAMPLES, DEDUP_NEAR_FAR];
pub(crate) const DEDUP_OUTDOOR_SKY: (&str, &str) = (
    "    if (b.x >= 0.0) {\n        let c = building_colour(e + d * b.x, i32(b.y), roughness, b.x);\n        probe_reach = b.x;\n        return c.rgb * c.a + sky_reflection(d, dir, lod) * (1.0 - c.a);\n    }\n    return sky_reflection(d, dir, lod);\n",
    "    var c = vec4<f32>(0.0);\n    if (b.x >= 0.0) {\n        c = building_colour(e + d * b.x, i32(b.y), roughness, b.x);\n        probe_reach = b.x;\n    }\n    return c.rgb * c.a + sky_reflection(d, dir, lod) * (1.0 - c.a);\n",
);
pub(crate) const DEDUP_PORTAL_SAMPLES: (&str, &str) = (
    "        if (own_room == low_room || own_room == high_room) {

            let other_room = select(low_room, high_room, own_room == low_room);
            let w_other = reach * select(1.0 - w_high, w_high, own_room == low_room);
            let other = probe_volume_sample(other_room, select_world, world_pos, d, lod);
            if (other.a < 0.0) {
                return own;
            }
            probe_brightness = mix(probe_brightness, volume_sample_brightness, w_other);
            return mix(own, other, w_other);
        }

        let a = probe_volume_sample(low_room, select_world, world_pos, d, lod);
        let a_bright = volume_sample_brightness;
        let b = probe_volume_sample(high_room, select_world, world_pos, d, lod);
        let b_bright = volume_sample_brightness;
",
    "        let own_side = own_room == low_room || own_room == high_room;
        var a = vec4<f32>(-1.0);
        var a_bright = 0.0;
        var b = vec4<f32>(-1.0);
        var b_bright = 0.0;
        for (var k = 0; k < select(2, 1, own_side); k = k + 1) {
            let room = select(select(low_room, high_room, k == 1), select(low_room, high_room, own_room == low_room), own_side);
            let s = probe_volume_sample(room, select_world, world_pos, d, lod);
            if (k == 0) {
                a = s;
                a_bright = volume_sample_brightness;
            } else {
                b = s;
                b_bright = volume_sample_brightness;
            }
        }
        if (own_side) {
            let w_other = reach * select(1.0 - w_high, w_high, own_room == low_room);
            if (a.a < 0.0) {
                return own;
            }
            probe_brightness = mix(probe_brightness, a_bright, w_other);
            return mix(own, a, w_other);
        }
",
);
pub(crate) const DEDUP_NEAR_FAR: (&str, &str) = (
    "        let sample_dir = probe_parallax_direction(world_pos, d, best);
        let layer = i32(camera.probe_boxes[best * 3].w);
        let near = textureSampleLevel(probe_cube, probe_samp, sample_dir, layer, probe_lod);
        probe_brightness = camera.probe_boxes[best * 3 + 1].w;
        own = near;
        if (second >= 0) {
            let far_dir = probe_parallax_direction(world_pos, d, second);
            let far = textureSampleLevel(
                probe_cube, probe_samp, far_dir, i32(camera.probe_boxes[second * 3].w), probe_lod
            );
",
    "        var near = vec4<f32>(0.0);
        var far = vec4<f32>(0.0);
        for (var k = 0; k < select(1, 2, second >= 0); k = k + 1) {
            let slot = select(best, second, k == 1);
            let s = textureSampleLevel(
                probe_cube, probe_samp, probe_parallax_direction(world_pos, d, slot), i32(camera.probe_boxes[slot * 3].w), probe_lod
            );
            if (k == 0) {
                near = s;
            } else {
                far = s;
            }
        }
        probe_brightness = camera.probe_boxes[best * 3 + 1].w;
        own = near;
        if (second >= 0) {
",
);

const B2_FACE_CODE: (&str, &str) = (
    "    let reflection = probe_env_for_pass(face_pos, n, rough_aa, ao, baked.a, baked.rgb, face_pos, n_geom);\n    return ProbePassOut(reflection, vec2<f32>(probe_reach, probe_face_code(n_geom)));",
    "    let face_code = probe_face_code(n_geom);\n    let reflection = probe_env_for_pass(face_pos, n, rough_aa, ao, baked.a, baked.rgb, face_pos, n_geom);\n    return ProbePassOut(reflection, vec2<f32>(probe_reach, face_code));",
);
const B2_RAY_DECL: (&str, &str) = (
    "var<private> probe_capsule_lit: vec3<f32> = vec3<f32>(0.0);\n",
    "var<private> probe_capsule_lit: vec3<f32> = vec3<f32>(0.0);\nvar<private> probe_ray_world: vec3<f32> = vec3<f32>(0.0);\n",
);
const B2_RAY_SET: (&str, &str) = ("    let d = normalize(world_dir);\n", "    let d = normalize(world_dir);\n    probe_ray_world = d;\n");
const B2_RECORD_DIR: (&str, &str) = (
    "            slot = probe_fixup_begin(hit, world_pos, d, dir, roughness, probe_lod, trace_room, recolour);",
    "            slot = probe_fixup_begin(hit, world_pos, d, to_player_direction(d), roughness, probe_lod, trace_room, recolour);",
);
const B2_COLOUR_DIR: (&str, &str) = (
    "        let col = probe_traced_colour(hit, d, roughness, dir, probe_lod);",
    "        let col = probe_traced_colour(hit, d, roughness, to_player_direction(d), probe_lod);",
);
const B2_CAPSULE_DIR: (&str, &str) = (
    "    if (probe_floor_mirror_here) {\n        probe = probe_floor_mirror_pass(probe, roughness);\n    } else {\n        probe = capsule_reflection(world_pos, refl, roughness, env, probe);\n    }",
    "    if (probe_floor_mirror_here) {\n        probe = probe_floor_mirror_pass(probe, roughness);\n    } else {\n        probe = capsule_reflection(world_pos, to_player_direction(probe_ray_world), roughness, env, probe);\n    }",
);
const B2_NO_CHARACTERS: (&str, &str) = (
    "    if (probe_floor_mirror_here) {\n        probe = probe_floor_mirror_pass(probe, roughness);\n    } else {\n        probe = capsule_reflection(world_pos, refl, roughness, env, probe);\n    }",
    "",
);
const B2_NO_RECORD: (&str, &str) = (
    "            slot = probe_fixup_begin(hit, world_pos, d, dir, roughness, probe_lod, trace_room, recolour);",
    "            slot = -1;",
);
const B2_LIT_GREY: (&str, &str) = ("    probe_capsule_lit = env;\n", "    probe_capsule_lit = vec3<f32>(ambient_here);\n");
const B2_FRAG_DECL: (&str, &str) = (
    "var<private> probe_fragment: vec4<f32> = vec4<f32>(0.0);",
    "var<private> probe_fragment_texel: u32 = 0u;\nvar<private> probe_fragment_z: f32 = 0.0;",
);
const B2_FRAG_SET: (&str, &str) = (
    "    probe_fragment = in.clip;\n",
    "    probe_fragment_texel = u32(in.clip.x) | (u32(in.clip.y) << 16u);\n    probe_fragment_z = in.clip.z;\n",
);
const B2_FRAG_MIRROR: (&str, &str) = (
    "vec2<i32>(probe_fragment.xy), probe_fragment.z,",
    "vec2<i32>(vec2<u32>(probe_fragment_texel & 0xFFFFu, probe_fragment_texel >> 16u)), probe_fragment_z,",
);
const B2_FRAG_RECORD: (&str, &str) = (
    "vec4<u32>(vec2<u32>(probe_fragment.xy), bitcast<u32>(probe_fragment.z), select(0u, 1u, recolour))",
    "vec4<u32>(probe_fragment_texel & 0xFFFFu, probe_fragment_texel >> 16u, bitcast<u32>(probe_fragment_z), select(0u, 1u, recolour))",
);
const B2_LOD_EARLY: (&str, &str) = ("    let probe_lod = clamp(roughness * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD);\n", "");
const B2_VOLUME_LATE: (&str, &str) = (
    "    let choice = probe_choose(select_world, volume_world);\n",
    "    let choice = probe_choose(select_world, select(select_world, to_world_space(probe_volume_pos.xyz), probe_volume_pos.w > 0.5));\n",
);
const B2_MISS_CHOICE_AT_HIT: (&str, &str) =
    ("    let choice = probe_choose(select_world, volume_world);\n", "    let choice = probe_choose(world_pos, world_pos);\n");
const B2_MISS_PORTALS_AT_HIT: (&str, &str) = (
    "    return probe_through_portals(own, own_room, select_world, world_pos, d, probe_lod);",
    "    return probe_through_portals(own, own_room, world_pos, world_pos, d, probe_lod);",
);
const B2_LOD_LATE: (&str, &str) = (
    "    let hit = probe_trace(world_pos, d, trace_room, roughness);\n",
    "    let hit = probe_trace(world_pos, d, trace_room, roughness);\n    let probe_lod = clamp(roughness * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD);\n",
);
const B2_CAPSULE_GREY: (&str, &str) = (
    "    if (probe_floor_mirror_here) {\n        probe = probe_floor_mirror_pass(probe, roughness);\n    } else {\n        probe = capsule_reflection(world_pos, refl, roughness, env, probe);\n    }",
    "    if (probe_floor_mirror_here) {\n        probe = probe_floor_mirror_pass(probe, roughness);\n    } else {\n        probe = capsule_reflection(world_pos, refl, roughness, probe_capsule_lit, probe);\n    }",
);
const B2_CAPSULE_DIR_GREY: (&str, &str) = (
    "    if (probe_floor_mirror_here) {\n        probe = probe_floor_mirror_pass(probe, roughness);\n    } else {\n        probe = capsule_reflection(world_pos, refl, roughness, env, probe);\n    }",
    "    if (probe_floor_mirror_here) {\n        probe = probe_floor_mirror_pass(probe, roughness);\n    } else {\n        probe = capsule_reflection(world_pos, to_player_direction(probe_ray_world), roughness, probe_capsule_lit, probe);\n    }",
);

const B2_POS_LATE_SET: (&str, &str) = (
    "    let hit = probe_trace(world_pos, d, trace_room, roughness);\n",
    "    let hit = probe_trace(world_pos, d, trace_room, roughness);\n    let late_zero = vec3<f32>(f32(camera.post_params.z > 3.0e38));\n    let world_late = to_world_space(frag_pos + late_zero);\n    let select_late = to_world_space(select_pos + late_zero);\n    let volume_late = select(select_late, to_world_space(probe_volume_pos.xyz + late_zero), probe_volume_pos.w > 0.5);\n",
);
const B2_POS_LATE_RECORD: (&str, &str) = (
    "            slot = probe_fixup_begin(hit, world_pos, d, to_player_direction(d), roughness, probe_lod, trace_room, recolour);",
    "            slot = probe_fixup_begin(hit, world_late, d, to_player_direction(d), roughness, probe_lod, trace_room, recolour);",
);
const B2_POS_LATE_CHOICE: (&str, &str) =
    ("    let choice = probe_choose(select_world, volume_world);\n", "    let choice = probe_choose(select_late, volume_late);\n");
const B2_POS_LATE_NEAR: (&str, &str) = (
    "        let sample_dir = probe_parallax_direction(world_pos, d, best);",
    "        let sample_dir = probe_parallax_direction(world_late, d, best);",
);
const B2_POS_LATE_FAR: (&str, &str) = (
    "            let far_dir = probe_parallax_direction(world_pos, d, second);",
    "            let far_dir = probe_parallax_direction(world_late, d, second);",
);
const B2_POS_LATE_PORTALS: (&str, &str) = (
    "    return probe_through_portals(own, own_room, select_world, world_pos, d, probe_lod);",
    "    return probe_through_portals(own, own_room, select_late, world_late, d, probe_lod);",
);

const B2_LIT_HALF_DECL: (&str, &str) = (
    "var<private> probe_capsule_lit: vec3<f32> = vec3<f32>(0.0);\n",
    "var<private> probe_capsule_lit: vec3<f32> = vec3<f32>(0.0);\nvar<private> probe_capsule_lit_half: vec2<u32> = vec2<u32>(0u);\nfn probe_capsule_lit_read() -> vec3<f32> {\n    return vec3<f32>(unpack2x16float(probe_capsule_lit_half.x), unpack2x16float(probe_capsule_lit_half.y).x);\n}\n",
);
const B2_LIT_HALF_SET: (&str, &str) = (
    "    probe_capsule_lit = env;\n",
    "    probe_capsule_lit_half = vec2<u32>(pack2x16float(env.rg), pack2x16float(vec2<f32>(env.b, 0.0)));\n",
);
const B2_LIT_HALF_RECORD: (&str, &str) = (
    "let lit = dot(probe_capsule_lit, vec3<f32>(0.2126, 0.7152, 0.0722));",
    "let lit = dot(probe_capsule_lit_read(), vec3<f32>(0.2126, 0.7152, 0.0722));",
);
const B2_LIT_HALF_CAPSULES: (&str, &str) = (
    "        probe = capsule_reflection(world_pos, to_player_direction(probe_ray_world), roughness, env, probe);",
    "        probe = capsule_reflection(world_pos, to_player_direction(probe_ray_world), roughness, probe_capsule_lit_read(), probe);",
);

const B2_NO_FAR_RIM: (&str, &str) =
    ("        if (hit.rim < 0.0) {\n            let spread_far", "        if (false) {\n            let spread_far");
const B2_NO_NEAR_RIM: (&str, &str) = ("        if (hit.rim < 0.0 && spread > PROBE_RIM_MIN_SPREAD) {", "        if (false) {");
const B2_NO_EDGES: (&str, &str) = ("        if (hit.edge_code < 0 && proxy.edge >= 0) {", "        if (false) {");
const B2_NO_FIELD_WALK: (&str, &str) = (
    "        let walk = probe_proxy_field_at(o, d, field_i, field_near, min(field_far, best), lobe);",
    "        let walk = vec3<f32>(camera.post_params.z, camera.post_params.w, camera.post_params.z - 1.0);",
);
const B2_NO_PROXY_SURFACE: (&str, &str) =
    ("    if (s0 < 0) {\n        return 3.4e38;\n    }", "    if (true) {\n        return 3.4e38;\n    }");
const B2_NO_PROXIES: (&str, &str) = (
    "        let proxy = probe_proxy_hit(hit.origin, d, cur, t0, t_exit, skip, lobe);",
    "        var proxy: ProbeProxyHit;\n        proxy.t = t0 + camera.post_params.z;\n        proxy.index = i32(camera.post_params.w);\n        proxy.edge = i32(camera.post_params.w) - 1;\n        proxy.edge_cover = camera.post_params.w;\n        proxy.edge_t = t0 + camera.post_params.w;",
);
const B2_NO_DOORWAY_START: (&str, &str) = ("    if (!in_room) {\n", "    if (false) {\n");
const B2_NO_PORTALS: (&str, &str) = ("        let p = probe_portal_at(e, cur, axis);", "        let p = -1;");
const B2_ONE_HOP: (&str, &str) = (ROLL_TRACE_FROM, "    for (var hop = 0; hop < 1; hop = hop + 1) {");

const B2_INV_FAR: (&str, &str) = (
    "        let far = select(vec3<f32>(3.4e38), max((hi - hit.origin) * inv, (lo - hit.origin) * inv), moving);",
    "        let inv_far = select(vec3<f32>(3.4e38), 1.0 / (d + vec3<f32>(f32(camera.post_params.z > 3.0e38))), abs(d) > vec3<f32>(1e-6));\n        let far = select(vec3<f32>(3.4e38), max((hi - hit.origin) * inv_far, (lo - hit.origin) * inv_far), abs(d) > vec3<f32>(1e-6));",
);
const B2_INV_SIDE: (&str, &str) = (
    "        var side = select(vec3<f32>(3.4e38), max((phi - e) * inv, (plo - e) * inv), moving);",
    "        let inv_side = select(vec3<f32>(3.4e38), 1.0 / (d + vec3<f32>(f32(camera.post_params.z > 3.1e38))), abs(d) > vec3<f32>(1e-6));\n        var side = select(vec3<f32>(3.4e38), max((phi - e) * inv_side, (plo - e) * inv_side), abs(d) > vec3<f32>(1e-6));",
);
const B2_INV_ENTER: (&str, &str) = (
    "            max((face[axis] - hit.origin[axis]) * inv[axis], t_exit),\n            max((wall_far - hit.origin[axis]) * inv[axis], t_exit),",
    "            max((face[axis] - hit.origin[axis]) * select(3.4e38, 1.0 / (d[axis] + f32(camera.post_params.z > 3.2e38)), abs(d[axis]) > 1e-6), t_exit),\n            max((wall_far - hit.origin[axis]) * select(3.4e38, 1.0 / (d[axis] + f32(camera.post_params.z > 3.2e38)), abs(d[axis]) > 1e-6), t_exit),",
);
const B2_RIM_REREAD: (&str, &str) = (
    "            let in_a = min(e2[a] - plo[a], phi[a] - e2[a]);\n            let in_b = min(e2[b] - plo[b], phi[b] - e2[b]);",
    "            let plo_far = camera.probe_portals[p * 3 + i32(camera.post_params.z > 3.3e38)].xyz;\n            let phi_far = camera.probe_portals[p * 3 + 1 + i32(camera.post_params.z > 3.3e38)].xyz;\n            let in_a = min(e2[a] - plo_far[a], phi_far[a] - e2[a]);\n            let in_b = min(e2[b] - plo_far[b], phi_far[b] - e2[b]);",
);

const B2_SEEN_TEXEL_ONLY: (&str, &str) = (
    "        let escapes = probe_seen_distance(oslot, d) < 0.0;",
    "        let seen_texel = textureSampleLevel(probe_depth, probe_depth_samp, d, i32(camera.probe_boxes[oslot * 3].w), 0.0);\n        let escapes = dot(seen_texel.xyz, seen_texel.xyz) < 0.25 && !(seen_texel.w > 0.5);",
);

const B2_SEEN_UNIFORM: (&str, &str) = (
    "        let escapes = probe_seen_distance(oslot, d) < 0.0;",
    "        let escapes = other + 1.0 == camera.portal_params.z || camera.post_params.z > 3.0e38;",
);
const B2_NO_SIDE: (&str, &str) =
    ("        let t_side = t_exit + min(min(side.x, side.y), side.z);", "        let t_side = 3.4e38 + camera.post_params.z;");

const ROLL_TRACE_FROM: &str = "    for (var hop = 0; hop < PROBE_TRACE_ROOMS; hop = hop + 1) {";
const ROLL_TRACE_TO: &str =
    "    for (var hop = 0; hop < PROBE_TRACE_ROOMS + i32(camera.post_params.z > 3.0e38); hop = hop + 1) {";
const ROLL_SAMPLES_FROM: &str = "    for (var k = 0; k <= PROBE_PROXY_SAMPLES; k = k + 1) {";
const ROLL_SAMPLES_TO: &str =
    "    for (var k = 0; k <= PROBE_PROXY_SAMPLES + i32(camera.post_params.z > 3.0e38); k = k + 1) {";
const ROLL_FIELD_FROM: &str = "    for (var k = 0; k < PROXY_FIELD_STEPS && t <= t_out; k = k + 1) {";
const ROLL_FIELD_TO: &str =
    "    for (var k = 0; k < PROXY_FIELD_STEPS + i32(camera.post_params.z > 3.0e38) && t <= t_out; k = k + 1) {";
const ROLL_GROUND_FROM: &str = "    for (var steps = 0; steps < GROUND_TRACE_MAX_STEPS && t < t_out; steps = steps + 1) {";
const ROLL_GROUND_TO: &str = "    for (var steps = 0; steps < GROUND_TRACE_MAX_STEPS + i32(camera.post_params.z > 3.0e38) && t < t_out; steps = steps + 1) {";
const ROLL_READINGS_FROM: &str = "            for (var k = 1; k < GROUND_TRACE_READINGS; k = k + 1) {";
const ROLL_READINGS_TO: &str =
    "            for (var k = 1; k < GROUND_TRACE_READINGS + i32(camera.post_params.z > 3.0e38); k = k + 1) {";

const STAGE_CUT_HIT: &str = "    var hit: ProbeHit;
    hit.pos = world_pos + d * camera.post_params.z;
    hit.room = trace_room;
    hit.other = camera.post_params.w - 2.0;
    hit.found = camera.post_params.z > 0.5;
    hit.escaped = camera.post_params.w > 0.5;
    hit.portal = i32(camera.post_params.w);
    hit.t = camera.post_params.z;
    hit.origin = world_pos;
    hit.rim = camera.post_params.z - 1.0;
    hit.rim_t = camera.post_params.z;
    hit.rim_code = i32(camera.post_params.z * 100.0);
    hit.edge_cover = camera.post_params.w;
    hit.edge_t = camera.post_params.z;
    hit.edge_code = i32(camera.post_params.w * 100.0) - 1;";

const SHADOW_TENT_OPAQUE: (&str, &str) = (
    "    for (var t = 0; t < 9; t = t + 1) {\n        let x = pcf_tent_tap_axis(",
    "    for (var t = 0; t < select(9, i32(camera.shadow_params.w), camera.shadow_params.w > 0.5); t = t + 1) {\n        let x = pcf_tent_tap_axis(",
);

const SHADOW_BOX_OPAQUE: (&str, &str) = (
    "    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {",
    "    for (var dx: i32 = -1; dx <= select(1, i32(camera.shadow_params.w), camera.shadow_params.w > 0.5); dx = dx + 1) {",
);

/// PER-FACE LAMP LISTS (the AAA doc's rank 3, without splitting the draw):
/// the vertex stage marks, per face, the lamps whose range reaches the
/// face's rectangle (its centre plus the length of its half extent, which
/// bounds every shading point: `face_pos` is clamped to the rectangle), and
/// the lamp loop walks only those. Per-pixel range culling still runs on
/// the ones walked, so nothing a pixel receives changes, save the highlight
/// widening's nearest-lamp distance, which now ignores lamps out of reach.
pub(crate) const FACE_LAMPS: &[(&str, &str)] = &[
    (
        "struct VOut {\n",
        "var<private> face_lamp_mask: u32 = 0xffffffffu;\nstruct VOut {\n",
    ),
    (
        "    @location(11) room_near: f32,\n",
        "    @location(11) room_near: f32,\n    @location(12) @interpolate(flat) face_lamps: u32,\n",
    ),
    (
        "    out.room_near = room_near(to_world_space(v.pos));\n",
        "    out.room_near = room_near(to_world_space(v.pos));\n    var face_lamps = 0u;\n    let face_r = length(v.face_half_extent);\n    for (var k: u32 = 0u; k < lights.count.x; k = k + 1u) {\n        let reach = lights.lights[k].params.x + face_r;\n        let off = lights.lights[k].position.xyz - v.face_centre;\n        if (lights.lights[k].params.z > 1.5 || dot(off, off) < reach * reach) {\n            face_lamps = face_lamps | (1u << k);\n        }\n    }\n    out.face_lamps = face_lamps;\n",
    ),
    (
        "    probe_volume_pos = vec4<f32>(in.face_centre, 1.0);\n    let env_part = shade_material_env_part(",
        "    probe_volume_pos = vec4<f32>(in.face_centre, 1.0);\n    face_lamp_mask = in.face_lamps;\n    let env_part = shade_material_env_part(",
    ),
    (
        "    for (var i: u32 = surface_light_count(); i < live_light_count(); i = i + 1u) {\n        let kind = lights.lights[i].params.z;\n        let to_lamp",
        "    var face_todo = face_lamp_mask & ((1u << live_light_count()) - 1u) & ~((1u << surface_light_count()) - 1u);\n    loop {\n        if (face_todo == 0u) {\n            break;\n        }\n        let i = countTrailingZeros(face_todo);\n        face_todo = face_todo & (face_todo - 1u);\n        let kind = lights.lights[i].params.z;\n        let to_lamp",
    ),
];

/// The scene readers walk each face's lamps (`FACE_LAMPS`). Measured on the
/// Quest 3 offline (2026-10-23, hall_back, one eye): the scene pass fell
/// 8.04 -> 7.83 ms, with the image the same as the full walk's to within
/// run-to-run noise.
pub const FACE_LAMP_LISTS: bool = true;

fn with_face_lamps(mut src: String) -> String {
    for (from, to) in FACE_LAMPS {
        assert!(src.contains(from), "the face lamp lists no longer find `{from}`");
        src = src.replacen(from, to, 1);
    }
    src
}

pub(crate) const SCENE_REGISTER_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("scene_cut_none", &[]),
    ("scene_cut_lamp_loop", &[("    var todo = reaching;\n", "    var todo = 0u;\n")]),
    (
        "scene_cut_lamps",
        &[(
            "    var face_todo = face_lamp_mask & ",
            "    var face_todo = 0u & face_lamp_mask & ",
        )],
    ),
    ("scene_cut_stationary", &[("    set_stationary_masks(st_0, st_1, st_2, st_3, STATIONARY_MASK_DISTANCE_TEXELS);\n", "")]),
    ("scene_cut_bounce", &[("    if (has_dir) {", "    if (false) {")]),
    (
        "scene_cut_probe_read",
        &[(
            "    probe_env_given = probe_pass_upsample(in.clip.xy, in.clip.z, probe_pass_tolerance, probe_face_code(n_geom));",
            "    probe_env_given = vec4<f32>(0.0);",
        )],
    ),
    (
        "scene_cut_lamp_spec",
        &[(
            "    if (aa.y > 0.0 && atten > 0.0 && l.params.w > -1.5) {\n        let h = normalize(l_dir + view_dir);\n        let spec = pow(max(dot(n, h), 0.0), shininess) * spec_strength;\n        out.specular",
            "    if (false) {\n        let h = normalize(l_dir + view_dir);\n        let spec = pow(max(dot(n, h), 0.0), shininess) * spec_strength;\n        out.specular",
        )],
    ),

    ("scene_cut_spot_shadow", &[("const SPOT_SHADOWS: bool = true;", "const SPOT_SHADOWS: bool = false;")]),

    ("scene_tent_opaque", &[SHADOW_TENT_OPAQUE]),
    ("scene_box_opaque", &[SHADOW_BOX_OPAQUE, SHADOW_BOX_OPAQUE]),
    ("scene_shadow_loops_opaque", &[SHADOW_TENT_OPAQUE, SHADOW_BOX_OPAQUE, SHADOW_BOX_OPAQUE]),

    (
        "scene_cut_sun_shadow",
        &[
            (
                "        if (!SKY_SUN_NEVER_REACHES && l.params.z > 1.5) {\n            shadow = sun_level_visibility(world_pos);",
                "        if (false) {\n            shadow = sun_level_visibility(world_pos);",
            ),
            ("        } else if (!SKY_SUN_NEVER_REACHES && shadow > 0.0 && l.position.w > 0.5", "        } else if (false && shadow > 0.0 && l.position.w > 0.5"),
            ("                    if (!SKY_SUN_NEVER_REACHES && map == SUN_MOVING_MAP) {", "                    if (false) {"),
        ],
    ),

    (
        "scene_moving_two_reads",
        &[
            ("            shadow = sun_level_visibility(world_pos);\n", "            shadow = sun_visibility(l, world_pos);\n"),
            (
                "        var map = -1;\n        var tile = 0.0;\n        if (SPOT_SHADOWS && layer >= 0 && f32(layer) < camera.shadow_params.y) {\n            map = 1 + layer;\n        } else if (l.params.z < 1.5) {\n            let k = character_shadow_tile(layer);\n            if (k >= 0) {\n                map = CHARACTER_MAPS + k;\n                tile = f32(1 + k);\n            }\n        } else if (!SKY_SUN_NEVER_REACHES && shadow > 0.0 && l.position.w > 0.5 && camera.shadow_params.z > 0.5) {\n            map = SUN_MOVING_MAP;\n        }\n        if (map >= 0) {\n            let c = shadow_coords(world_pos, camera.moving_view_proj[map]);\n            if (c.w >= 0.5) {\n                if (SPOT_SHADOWS && map > SUN_MOVING_MAP && map < CHARACTER_MAPS) {\n                    shadow = shadow * pcf_layer_at(spot_shadow_tex, layer, c.xyz);\n                } else {\n                    var at = vec4<f32>(c.xyz, tile);\n                    if (!SKY_SUN_NEVER_REACHES && map == SUN_MOVING_MAP) {\n                        at = sun_moving_tile_at(c.xyz);\n                    }\n                    shadow = shadow * pcf_tile_at(sun_dynamic_shadow_tex, vec2<f32>(at.w, 0.0), SUN_ATLAS_GRID, at.xyz);\n                }\n            }\n        }\n",
                "        if (SPOT_SHADOWS && layer >= 0 && f32(layer) < camera.shadow_params.y) {\n            shadow = shadow * pcf_layer(spot_shadow_tex, layer, world_pos, spot_view_proj(layer));\n        } else if (l.params.z < 1.5) {\n            shadow = shadow * character_shadow(layer, world_pos);\n        }\n",
            ),
        ],
    ),
    (
        "scene_cut_spot_cone",
        &[(
            "            let cos_angle = dot(-l_dir, l.direction.xyz);\n            atten = atten * spot_cone(cos_angle, cos_outer, cos_inner, dist, spot_cone_across(l.direction.xyz, l_dir, cos_angle, dist, n, view_dir));\n        }\n    }\n\n    let aa = terminator_aa(dot(n, l_dir));\n    let ndotl = aa.x;\n    let radiance = l.color_intensity.rgb * l.color_intensity.a;\n    out.diffuse",
            "            let cos_angle = dot(-l_dir, l.direction.xyz);\n        }\n    }\n\n    let aa = terminator_aa(dot(n, l_dir));\n    let ndotl = aa.x;\n    let radiance = l.color_intensity.rgb * l.color_intensity.a;\n    out.diffuse",
        )],
    ),

    ("scene_cut_spot_average", &[("const SPOT_EDGE_AVERAGE: bool = true;", "const SPOT_EDGE_AVERAGE: bool = false;")]),

    ("scene_cut_face_same_loop", &[FACE_SAME_LOOP]),
    ("scene_cut_face_same_calls", &[FACE_SAME_CALLS]),

    ("scene_cut_lamp_loop_uniform", &[LAMP_LOOP_UNIFORM]),
    ("scene_cut_lamp_loop_merged", &[LAMP_REACHES_FN, LAMP_WALK_NO_MASK, LAMP_LOOP_MERGED]),

    ("scene_cut_spot_long_mapped", &[SPOT_LONG_MAPPED]),
    (
        "scene_cut_sky",
        &[
            ("        diffuse = sky_irradiance(n) * occ;\n", ""),
            ("        sky_reflection = environment_radiance(refl) * occ;\n", ""),
        ],
    ),
    (
        "scene_cut_sun_mask",
        &[("    receiver_sun_mask = select(-1.0, smoothstep(-sun_w, sun_w, sun_d), sun_mask.g > 0.25);", "    receiver_sun_mask = -1.0;")],
    ),
    ("scene_cut_terminator_aa", &[("    terminator_width = terminator_width_of(length(tn));", "")]),
    ("scene_cut_contact", &[("    let contact = capsule_ambient(world_pos, n);", "    let contact = 1.0;")]),

    ("scene_box_spots", &[("    return pcf_tile_tent_lean_at(\n        tex,", "    return pcf_tile_at(\n        tex,")]),

    ("scene_tent_unrolled", &[("    return pcf_tile_tent_lean_at(\n        tex,", "    return pcf_tile_tent_unrolled_at(\n        tex,")]),

    ("scene_tent_loop", &[("    return pcf_tile_tent_lean_at(\n        tex,", "    return pcf_tile_tent_at(\n        tex,")]),
    ("scene_tent_half", &[("    return pcf_tile_tent_lean_at(\n        tex,", "    return pcf_tile_tent_half_at(\n        tex,")]),

    ("scene_all_f32", &[("alias hf = f32;\n", "alias hf = f32; // scene_all_f32\n")]),

    (
        "scene_highlight_uncapped",
        &[("    out.specular = radiance * spec * atten * lamp_highlight_share(view_dir, h, ndotl, aa.y, shininess, spec_strength);", "    out.specular = radiance * spec * atten * aa.y;")],
    ),
    (
        "scene_census_trace",
        &[(
            "    return vec4<f32>(tonemap(c), reflection_alpha(lit));",
            "    let census = clamp(textureSampleBias(mat_rough, mat_rough_samp, in.uv, i32(in.material), 1.0).r, 0.04, 1.0) < PROBE_LOBE_HEMISPHERICAL;\n    return vec4<f32>(select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), census), 1.0);",
        )],
    ),
];

pub(crate) const READER_EDITS: &[(&str, &[(&str, &str)])] = &[
    ("reader_edit_none", &[]),

    ("sky_walked", &[SKY_WALKED_DIFFUSE, SKY_WALKED_REFLECTION]),

    ("sun_matrix_fixed", &[SUN_MATRIX_FIXED]),

    ("face_same_calls", &[FACE_SAME_CALLS]),
    ("face_same_loop", &[FACE_SAME_LOOP]),

    ("spot_long_mapped", &[SPOT_LONG_MAPPED]),

    ("lamp_loop_uniform", &[LAMP_LOOP_UNIFORM]),

    ("lamp_loop_merged", &[LAMP_REACHES_FN, LAMP_WALK_NO_MASK, LAMP_LOOP_MERGED]),
];

const LAMP_REACHES_FN: (&str, &str) = (
    "fn shade_material_lamps(",
    "fn lamp_reaches(i: u32, world_pos: vec3<f32>, pixel_long: f32, culling: bool) -> bool {\n    let kind = lights.lights[i].params.z;\n    let to_lamp = lights.lights[i].position.xyz - world_pos;\n    let dist_sq = dot(to_lamp, to_lamp);\n    if (kind > 1.5 && SKY_SUN_NEVER_REACHES) {\n        return false;\n    }\n    if (culling) {\n        if (CULL_RANGE_FIRST && kind < 1.5 && dist_sq >= lights.lights[i].params.x * lights.lights[i].params.x) {\n            return false;\n        }\n        if (stationary_visibility_of(lights.lights[i].position.w) <= 0.0) {\n            return false;\n        }\n        if (kind < 1.5) {\n            let reach = lights.lights[i].params.x;\n            if (!CULL_RANGE_FIRST && dist_sq >= reach * reach) {\n                return false;\n            }\n            if (kind > 0.5) {\n                let cos_outer = lights.lights[i].params.y;\n                let authored = max(lights.lights[i].direction.w - cos_outer, 0.0001);\n                let inv_dist = inverseSqrt(max(dist_sq, 1e-6));\n                let at_most = SPOT_EDGE_MIN_PIXELS * pixel_footprint * inv_dist;\n                let widen = clamp(at_most - authored, 0.0, authored * (SPOT_EDGE_MAX_WIDEN - 1.0));\n                let zero_below = cos_outer - 0.5 * widen - select(0.0, 0.5 * pixel_long * inv_dist, SPOT_EDGE_AVERAGE) - 1e-4;\n                let along = -dot(to_lamp, lights.lights[i].direction.xyz);\n                let bound_sq = zero_below * zero_below * max(dist_sq, 1e-8);\n                let outside = select(\n                    along < 0.0 && along * along >= bound_sq,\n                    along <= 0.0 || along * along <= bound_sq,\n                    zero_below >= 0.0,\n                );\n                if (outside) {\n                    return false;\n                }\n            }\n        } else if (receiver_sun_mask == 0.0) {\n            return false;\n        }\n    }\n    return true;\n}\n\nfn shade_material_lamps(",
);

const LAMP_WALK_NO_MASK: (&str, &str) = ("        reaching = reaching | (1u << i);\n", "");

const LAMP_LOOP_MERGED: (&str, &str) = (
    LAMP_LOOP_UNIFORM.0,
    "    for (var i: u32 = surface_light_count(); i < live_light_count(); i = i + 1u) {\n        if (!lamp_reaches(i, world_pos, pixel_long, culling)) {\n            continue;\n        }\n        let l = lights.lights[i];\n",
);

const LAMP_LOOP_UNIFORM: (&str, &str) = (
    "    var todo = reaching;\n    loop {\n        if (todo == 0u) {\n            break;\n        }\n        let i = countTrailingZeros(todo);\n        todo = todo & (todo - 1u);\n        let l = lights.lights[i];\n",
    "    for (var i: u32 = surface_light_count(); i < live_light_count(); i = i + 1u) {\n        if ((reaching & (1u << i)) == 0u) {\n            continue;\n        }\n        let l = lights.lights[i];\n",
);

const SPOT_LONG_MAPPED: (&str, &str) = (
    "fn spot_long_step(vn: f32) -> f32 {\n    return pixel_long_step;\n}",
    "fn spot_long_step(vn: f32) -> f32 {\n    return pixel_footprint * inverseSqrt(max(abs(vn), SPOT_LONG_MIN_COS));\n}",
);

const FACE_SAME_FROM: &str = "    let code_i = i32(code) - 1;\n    let codes_i = vec4<i32>(codes) - vec4<i32>(1);\n    let face_dx = abs(codes_i / vec4<i32>(32) - vec4<i32>(code_i / 32));\n    let face_dy = abs(codes_i % vec4<i32>(32) - vec4<i32>(code_i % 32));\n    let facing = (codes < vec4<f32>(0.5)) | vec4<bool>(code < 0.5) | ((face_dx <= vec4<i32>(2)) & (face_dy <= vec4<i32>(2)));\n";
const FACE_SAME_LOOP: (&str, &str) = (
    FACE_SAME_FROM,
    "    var facing = vec4<bool>(false);\n    for (var k = 0; k < 4 + i32(camera.post_params.z > 3.0e38); k = k + 1) {\n        facing[k] = probe_face_same(codes[k], code);\n    }\n",
);
const FACE_SAME_CALLS: (&str, &str) = (
    FACE_SAME_FROM,
    "    let facing = vec4<bool>(\n        probe_face_same(codes.x, code), probe_face_same(codes.y, code),\n        probe_face_same(codes.z, code), probe_face_same(codes.w, code),\n    );\n",
);
const SKY_WALKED_DIFFUSE: (&str, &str) = (
    "    var diffuse = vec3<f32>(0.0);\n    if (occ > 0.0) {\n        diffuse = sky_irradiance(n) * occ;\n    }\n",
    "    var diffuse = vec3<f32>(0.0);\n",
);
const SKY_WALKED_REFLECTION: (&str, &str) = (
    "    if (occ > 0.0) {\n        sky_reflection = environment_radiance(refl) * occ;\n    }\n",
    "    var sky_dirs = select(0u, 3u, occ > 0.0);\n    loop {\n        if (sky_dirs == 0u) {\n            break;\n        }\n        let k = countTrailingZeros(sky_dirs);\n        sky_dirs = sky_dirs & (sky_dirs - 1u);\n        let dir = select(n, refl, k == 1u);\n        let sky = sky_irradiance(dir);\n        if (k == 0u) {\n            diffuse = sky * occ;\n        } else {\n            sky_reflection = environment_radiance_of(dir, sky) * occ;\n        }\n    }\n",
);
const SUN_MATRIX_FIXED: (&str, &str) = (
    "            let c = shadow_coords(world_pos, camera.moving_view_proj[map]);\n",
    "            var c = vec4<f32>(0.0);\n            if (map == SUN_MOVING_MAP) {\n                c = shadow_coords(world_pos, camera.moving_view_proj[SUN_MOVING_MAP]);\n            } else {\n                c = shadow_coords(world_pos, camera.moving_view_proj[map]);\n            }\n",
);

pub(crate) fn with_reader_edit(src: String, edit: &str) -> Option<String> {

    if edit.starts_with("opt_") {
        return crate::renderer::shader_opt::edit_named(&src, edit);
    }
    let (_, edits) = READER_EDITS.iter().find(|(name, _)| *name == edit)?;
    let mut src = src;
    for (from, to) in edits.iter() {
        if src.matches(from).count() != 1 {
            return None;
        }
        src = src.replacen(from, to, 1);
    }
    Some(src)
}

const PROBE_PASS_REGISTER_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("cut_none", &[]),
    ("cut_edge_lookup", &[("    if (hit.edge_code >= 0 && edge_cover < 0.99) {\n", "    if (false) {\n")]),
    ("cut_rim_lookup", &[("    if (hit.rim >= 0.0) {\n", "    if (false) {\n")]),
    (
        "cut_both_lookups",
        &[("    if (hit.edge_code >= 0 && edge_cover < 0.99) {\n", "    if (false) {\n"), ("    if (hit.rim >= 0.0) {\n", "    if (false) {\n")],
    ),
    ("cut_edge_detect", &[("        if (camera.probe_proxies[i * 3 + 1].w < 0.5 && out.edge < 0) {", "        if (false) {")]),
    ("cut_rim_detect", &[("        if (hit.rim < 0.0 && spread > PROBE_RIM_MIN_SPREAD) {", "        if (false) {")]),
    ("cut_proxies", &[("    for (var i = probe_room_proxy(room); i >= 0; i = probe_proxy_next(i)) {", "    for (var i = -1; i >= 0; i = probe_proxy_next(i)) {")]),
    ("cut_proxy_surface", &[("    if (s0 < 0) {\n        return 3.4e38;\n    }", "    if (true) {\n        return 3.4e38;\n    }")]),
    ("cut_escape_colour", &[("    if (h.escaped) {\n        col = probe_escape_colour(", "    if (false) {\n        col = probe_escape_colour(")]),
    ("cut_untraced", &[("    return probe_through_portals(own, own_room, select_world, world_pos, d, probe_lod);", "    return own;")]),
    ("cut_one_hop", &[("const PROBE_TRACE_ROOMS: i32 = 3;", "const PROBE_TRACE_ROOMS: i32 = 1;")]),
    ("cut_trace", &[("    if (roughness > PROBE_TRACE_MAX_ROUGHNESS || camera.portal_params.y > 0.5) {", "    if (true) {")]),
];

pub mod probe_pass {
    use wgpu::{
        AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingResource, BindingType, Device, Extent3d, FilterMode, Sampler,
        SamplerBindingType, SamplerDescriptor, ShaderStages, Texture, TextureDescriptor, TextureDimension,
        TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDimension,
    };

    pub const FORMAT: TextureFormat = TextureFormat::Rgba16Float;

    pub const REACH_FORMAT: TextureFormat = TextureFormat::Rg16Float;

    pub(crate) const FACE_CODE_WGSL: &str = r#"
fn probe_face_code(n: vec3<f32>) -> f32 {
    let a = n / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-6);
    var o = a.xz;
    if (a.y < 0.0) {
        o = (vec2<f32>(1.0) - abs(a.zx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), a.xz >= vec2<f32>(0.0));
    }
    let q = clamp(round((o * 0.5 + vec2<f32>(0.5)) * 31.0), vec2<f32>(0.0), vec2<f32>(31.0));
    return 1.0 + q.x * 32.0 + q.y;
}

fn probe_face_same(a: f32, b: f32) -> bool {
    if (a < 0.5 || b < 0.5) {
        return true;
    }
    let ia = i32(a) - 1;
    let ib = i32(b) - 1;
    let d = abs(vec2<i32>(ia / 32 - ib / 32, ia % 32 - ib % 32));
    return d.x <= 2 && d.y <= 2;
}
"#;

    pub const MIRROR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

    pub const MIRROR_MIPS: u32 = 5;

    pub const MIRROR_EXPOSURE: f32 = 1.0 / 16.0;

    pub const FLOOR_MIRROR_PLANE: f32 = 0.0;

    pub const FLOOR_MIRROR_BIAS: f32 = 4096.0;

    pub const EYE_NEAR: f32 = 0.03;
    pub const EYE_FAR: f32 = 1000.0;

    pub const OUTPUT_SIGNATURE: &str = "-> ProbePassOut";
    pub(crate) const OUTPUT_WGSL: &str = r#"
struct ProbePassOut {
    @location(0) reflection: vec4<f32>,

    @location(1) reach: vec2<f32>,
}
"#;

    pub fn targets(reach: bool) -> [Option<wgpu::ColorTargetState>; 2] {
        [
            Some(wgpu::ColorTargetState { format: FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL }),
            Some(wgpu::ColorTargetState {
                format: REACH_FORMAT,
                blend: None,
                write_mask: if reach { wgpu::ColorWrites::ALL } else { wgpu::ColorWrites::empty() },
            }),
        ]
    }

    pub fn bind_group_layout(device: &Device) -> BindGroupLayout {
        let texture = |binding: u32, sample_type: TextureSampleType| BindGroupLayoutEntry {
            binding,
            visibility: if binding == 1 { ShaderStages::VERTEX_FRAGMENT } else { ShaderStages::FRAGMENT },

            ty: BindingType::Texture { sample_type, view_dimension: TextureViewDimension::D2Array, multisampled: false },
            count: None,
        };
        let sampler = |binding: u32, kind: SamplerBindingType| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(kind),
            count: None,
        };
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("probe_pass_layout"),
            entries: &[
                texture(0, TextureSampleType::Float { filterable: true }),
                texture(1, TextureSampleType::Depth),
                sampler(2, SamplerBindingType::Filtering),
                sampler(3, SamplerBindingType::NonFiltering),
                texture(4, TextureSampleType::Float { filterable: true }),
            ],
        })
    }

    pub struct Target {
        _color: Texture,

        pub color_view: TextureView,
        _reach: Texture,

        pub reach_view: TextureView,
        _depth: Texture,
        pub depth_view: TextureView,
        _mirror: Texture,

        pub mirror_levels: Vec<Vec<TextureView>>,
        _mirror_depth: Texture,

        pub mirror_depth_views: Vec<TextureView>,

        pub mirror_view: TextureView,
        pub reach_single: TextureView,
        _samplers: [Sampler; 2],
        pub bind_group: BindGroup,

        pub soft: Option<Soft>,
        pub width: u32,
        pub height: u32,
    }

    pub struct Soft {
        _texture: Texture,
        pub write_view: TextureView,
        pub bind_group: BindGroup,
    }

    impl Target {

        pub fn new(device: &Device, layout: &BindGroupLayout, eye_width: u32, eye_height: u32, layers: u32) -> Self {
            let (width, height) = (eye_width.div_ceil(2).max(1), eye_height.div_ceil(2).max(1));
            let layers = layers.max(1);
            let make = |label: &str, format: TextureFormat, extra: TextureUsages| {
                device.create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: Extent3d { width, height, depth_or_array_layers: layers },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING | extra,
                    view_formats: &[],
                })
            };

            let color = make(
                "probe_pass_color",
                FORMAT,
                if layers == 1 { TextureUsages::STORAGE_BINDING } else { TextureUsages::empty() },
            );
            let depth = make("probe_pass_depth", TextureFormat::Depth32Float, TextureUsages::empty());
            let reach = make("probe_pass_reach", REACH_FORMAT, TextureUsages::empty());

            let attachment = |t: &Texture| {
                t.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(if layers > 1 { TextureViewDimension::D2Array } else { TextureViewDimension::D2 }),
                    ..Default::default()
                })
            };
            let array = |t: &Texture| {
                t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() })
            };
            let (color_view, depth_view) = (attachment(&color), attachment(&depth));
            let reach_view = attachment(&reach);
            let (color_array, depth_array, reach_array) = (array(&color), array(&depth), array(&reach));

            let mirror_mips = MIRROR_MIPS.min(width.min(height).max(1).ilog2() + 1);
            let mirror = device.create_texture(&TextureDescriptor {
                label: Some("probe_pass_mirror"),
                size: Extent3d { width, height, depth_or_array_layers: layers },
                mip_level_count: mirror_mips,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: MIRROR_FORMAT,

                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING
                    | TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let mirror_depth = make("probe_pass_mirror_depth", TextureFormat::Depth32Float, TextureUsages::empty());
            let single = |t: &Texture, layer: u32, level: u32| {
                t.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            };
            let mirror_levels = (0..layers).map(|l| (0..mirror_mips).map(|k| single(&mirror, l, k)).collect()).collect();
            let mirror_depth_views = (0..layers).map(|l| single(&mirror_depth, l, 0)).collect();

            let mirror_view = mirror.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2),
                base_array_layer: 0,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let reach_single = single(&reach, 0, 0);

            let sampler = |label: &str, filter: FilterMode| {
                device.create_sampler(&SamplerDescriptor {
                    label: Some(label),
                    address_mode_u: AddressMode::ClampToEdge,
                    address_mode_v: AddressMode::ClampToEdge,
                    mag_filter: filter,
                    min_filter: filter,
                    ..Default::default()
                })
            };
            let samplers = [sampler("probe_pass_linear", FilterMode::Linear), sampler("probe_pass_point", FilterMode::Nearest)];
            let reading = |label: &str, colour: &TextureView| {
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some(label),
                    layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: BindingResource::TextureView(colour) },
                        BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&depth_array) },
                        BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&samplers[0]) },
                        BindGroupEntry { binding: 3, resource: BindingResource::Sampler(&samplers[1]) },
                        BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&reach_array) },
                    ],
                })
            };
            let bind_group = reading("probe_pass_bg", &color_array);

            let soft = (layers == 1).then(|| {
                let texture = make("probe_pass_soft", FORMAT, TextureUsages::STORAGE_BINDING);
                let write_view = attachment(&texture);
                let bind_group = reading("probe_pass_soft_bg", &array(&texture));
                Soft { _texture: texture, write_view, bind_group }
            });
            Self {
                _color: color,
                color_view,
                _reach: reach,
                reach_view,
                _depth: depth,
                depth_view,
                _mirror: mirror,
                mirror_levels,
                _mirror_depth: mirror_depth,
                mirror_depth_views,
                mirror_view,
                reach_single,
                _samplers: samplers,
                bind_group,
                soft,
                width,
                height,
            }
        }
    }

    impl Target {

        pub fn mirror_texture(&self) -> &Texture {
            &self._mirror
        }
    }

    pub struct MirrorMips {

        variants: Vec<(wgpu::ComputePipeline, BindGroupLayout)>,
    }

    impl MirrorMips {
        pub fn new(device: &Device) -> Self {
            let variants = (1..MIRROR_MIPS as usize)
                .map(|n| {
                    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("floor_mirror_mips"),
                        source: wgpu::ShaderSource::Wgsl(mirror_mips_wgsl(n).into()),
                    });
                    let mut entries = vec![BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    }];
                    entries.extend((1..=n as u32).map(|k| BindGroupLayoutEntry {
                        binding: k,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: MIRROR_FORMAT,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    }));
                    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                        label: Some("floor_mirror_mips_layout"),
                        entries: &entries,
                    });
                    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("floor_mirror_mips"),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    });
                    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("floor_mirror_mips"),
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });
                    (pipeline, layout)
                })
                .collect();
            Self { variants }
        }

        pub fn record(
            &self,
            device: &Device,
            encoder: &mut wgpu::CommandEncoder,
            levels: &[TextureView],
            size: (u32, u32),
            timer: Option<(&crate::renderer::pass_timers::PassTimers, usize)>,
        ) {
            let n = levels.len().saturating_sub(1);
            let Some((pipeline, layout)) = n.checked_sub(1).and_then(|i| self.variants.get(i)) else {
                return;
            };
            let entries: Vec<BindGroupEntry> =
                levels.iter().enumerate().map(|(k, view)| BindGroupEntry { binding: k as u32, resource: BindingResource::TextureView(view) }).collect();
            let bind = device.create_bind_group(&BindGroupDescriptor { label: Some("floor_mirror_mips"), layout, entries: &entries });
            let (w, h) = (size.0 / 2, size.1 / 2);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("floor_mirror_mips"),
                timestamp_writes: timer.and_then(|(timers, slot)| timers.compute_writes(slot)),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(w.div_ceil(16).max(1), h.div_ceil(16).max(1), 1);
        }
    }

    fn mirror_mips_wgsl(n: usize) -> String {
        let mut s = String::from("@group(0) @binding(0) var level0: texture_2d<f32>;\n");
        for k in 1..=n {
            s += &format!("@group(0) @binding({k}) var level{k}: texture_storage_2d<rgba16float, write>;\n");
        }
        s += r#"
var<workgroup> tile_a: array<vec4<f32>, 256>;
var<workgroup> tile_b: array<vec4<f32>, 64>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) t: vec3<u32>) {
    let d0 = vec2<i32>(textureDimensions(level0));
    let last = d0 - vec2<i32>(1);
    let d1 = d0 / 2;
    let p1 = vec2<i32>(g.xy * 16u + t.xy);
    let q = p1 * 2;
    let v1 = 0.25 * (textureLoad(level0, min(q, last), 0) + textureLoad(level0, min(q + vec2<i32>(1, 0), last), 0)
        + textureLoad(level0, min(q + vec2<i32>(0, 1), last), 0) + textureLoad(level0, min(q + vec2<i32>(1, 1), last), 0));
    if (all(p1 < d1)) {
        textureStore(level1, p1, v1);
    }
    tile_a[t.y * 16u + t.x] = v1;
"#;
        if n >= 2 {
            s += r#"
    workgroupBarrier();
    if (t.x < 8u && t.y < 8u) {
        let i = t.y * 32u + t.x * 2u;
        let v2 = 0.25 * (tile_a[i] + tile_a[i + 1u] + tile_a[i + 16u] + tile_a[i + 17u]);
        let p2 = vec2<i32>(g.xy * 8u + t.xy);
        if (all(p2 < d1 / 2)) {
            textureStore(level2, p2, v2);
        }
        tile_b[t.y * 8u + t.x] = v2;
    }
"#;
        }
        if n >= 3 {
            s += r#"
    workgroupBarrier();
    if (t.x < 4u && t.y < 4u) {
        let i = t.y * 16u + t.x * 2u;
        let v3 = 0.25 * (tile_b[i] + tile_b[i + 1u] + tile_b[i + 8u] + tile_b[i + 9u]);
        let p3 = vec2<i32>(g.xy * 4u + t.xy);
        if (all(p3 < d1 / 4)) {
            textureStore(level3, p3, v3);
        }
        tile_a[t.y * 4u + t.x] = v3;
    }
"#;
        }
        if n >= 4 {
            s += r#"
    workgroupBarrier();
    if (t.x < 2u && t.y < 2u) {
        let i = t.y * 8u + t.x * 2u;
        let v4 = 0.25 * (tile_a[i] + tile_a[i + 1u] + tile_a[i + 4u] + tile_a[i + 5u]);
        let p4 = vec2<i32>(g.xy * 2u + t.xy);
        if (all(p4 < d1 / 8)) {
            textureStore(level4, p4, v4);
        }
    }
"#;
        }
        s += "}\n";
        s
    }

    pub(super) const PASS_LIGHTING: &str = r#"
    let rough = textureSample(mat_rough, mat_rough_samp, in.uv, i32(in.material)).r;

    let rough_aa = specular_aa_roughness(rough, dpdx(n_geom), dpdy(n_geom));
    let ao = textureSample(mat_ao, mat_samp, in.uv, i32(in.material)).r;
    let lm_uv = clamp(in.uv2, in.uv2_rect.xy, in.uv2_rect.zw);
    let baked = textureSample(lm_tex, lm_samp, lm_uv) * vec4<f32>(vec3<f32>(exp2(-camera.proxy_params.z)), 1.0);
    let face_t = normalize(in.tangent.xyz - n_geom * dot(n_geom, in.tangent.xyz));
    let face_b = cross(n_geom, face_t) * in.tangent.w;
    let face_d = in.world_pos - in.face_centre;
    let face_pos = in.face_centre
        + face_t * clamp(dot(face_d, face_t), -in.face_half_extent.x, in.face_half_extent.x)
        + face_b * clamp(dot(face_d, face_b), -in.face_half_extent.y, in.face_half_extent.y);
    probe_volume_pos = vec4<f32>(in.face_centre, 1.0);
    probe_face_given = in.probe_face;

    if (clamp(rough, 0.04, 1.0) >= PROBE_LOBE_HEMISPHERICAL) {
        return ProbePassOut(vec4<f32>(0.0), vec2<f32>(0.0, probe_face_code(n_geom)));
    }
    let reflection = probe_env_for_pass(face_pos, n, rough_aa, ao, baked.a, baked.rgb, face_pos, n_geom);
    return ProbePassOut(reflection, vec2<f32>(probe_reach, probe_face_code(n_geom)));"#;

    pub(crate) const READER_WGSL: &str = r#"
@group(3) @binding(0) var probe_pass_tex: texture_2d_array<f32>;
@group(3) @binding(1) var probe_pass_depth: texture_depth_2d_array;
@group(3) @binding(2) var probe_pass_linear: sampler;
@group(3) @binding(3) var probe_pass_point: sampler;

@group(3) @binding(4) var probe_pass_code: texture_2d_array<f32>;

fn probe_pass_expand(c: vec3<f32>) -> vec3<f32> {
    return c / max(1.0 - dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-3);
}

fn probe_pass_upsample(pixel: vec2<f32>, depth: f32, tolerance: f32, code: f32) -> vec4<f32> {
    let dims = textureDimensions(probe_pass_tex);

    let uv = pixel * 0.5 / vec2<f32>(dims);

    let gaps = abs(textureGather(probe_pass_depth, probe_pass_point, uv, view_slot) - vec4<f32>(depth)).wzxy;
    let codes = textureGather(1, probe_pass_code, probe_pass_point, uv, view_slot).wzxy;

    let code_i = i32(code) - 1;
    let codes_i = vec4<i32>(codes) - vec4<i32>(1);
    let face_dx = abs(codes_i / vec4<i32>(32) - vec4<i32>(code_i / 32));
    let face_dy = abs(codes_i % vec4<i32>(32) - vec4<i32>(code_i % 32));
    let facing = (codes < vec4<f32>(0.5)) | vec4<bool>(code < 0.5) | ((face_dx <= vec4<i32>(2)) & (face_dy <= vec4<i32>(2)));
    let same = facing & (gaps <= vec4<f32>(tolerance));

    var pre: vec4<f32>;
    if (all(same)) {
        pre = textureSampleLevel(probe_pass_tex, probe_pass_linear, uv, view_slot, 0.0);
    } else {
        let size = vec2<i32>(dims);
        let h = pixel * 0.5 - vec2<f32>(0.5);
        let base = vec2<i32>(floor(h));
        let f = h - floor(h);
        let top = size - vec2<i32>(1);
        let c00 = textureLoad(probe_pass_tex, clamp(base, vec2<i32>(0), top), view_slot, 0);
        let c10 = textureLoad(probe_pass_tex, clamp(base + vec2<i32>(1, 0), vec2<i32>(0), top), view_slot, 0);
        let c01 = textureLoad(probe_pass_tex, clamp(base + vec2<i32>(0, 1), vec2<i32>(0), top), view_slot, 0);
        let c11 = textureLoad(probe_pass_tex, clamp(base + vec2<i32>(1, 1), vec2<i32>(0), top), view_slot, 0);
        let bilinear = vec4<f32>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
        let w = select(vec4<f32>(0.0), bilinear, same);
        let weight = w.x + w.y + w.z + w.w;

        let far = vec4<f32>(3.4e38);
        let near = select(far, gaps, facing);
        var nearest = vec4<f32>(0.0);
        var nearest_gap = far.x;
        if (near.x < nearest_gap) { nearest_gap = near.x; nearest = c00; }
        if (near.y < nearest_gap) { nearest_gap = near.y; nearest = c10; }
        if (near.z < nearest_gap) { nearest_gap = near.z; nearest = c01; }
        if (near.w < nearest_gap) { nearest_gap = near.w; nearest = c11; }
        let sum = c00 * w.x + c10 * w.y + c01 * w.z + c11 * w.w;
        pre = select(nearest, sum / max(weight, 1e-6), weight > 1e-4);
    }
    return vec4<f32>(probe_pass_expand(pre.rgb / max(pre.a, 1e-4)), pre.a);
}
"#;

    pub(crate) fn eye_linear_depth_wgsl() -> String {
        format!(
            r#"
const FLOOR_MIRROR_NEAR: f32 = {near:?};
const FLOOR_MIRROR_FAR: f32 = {far:?};
fn floor_mirror_linear_depth(d: f32) -> f32 {{
    let z = 2.0 * d - 1.0;
    return 2.0 * FLOOR_MIRROR_FAR * FLOOR_MIRROR_NEAR / ((FLOOR_MIRROR_FAR + FLOOR_MIRROR_NEAR) - z * (FLOOR_MIRROR_FAR - FLOOR_MIRROR_NEAR));
}}
"#,
            near = EYE_NEAR,
            far = EYE_FAR,
        )
    }
}

fn brush_shader_variant(ssr: bool, sources: bool, ssr_debug: bool) -> String {
    brush_shader_modes(ssr, sources, ssr_debug, SsrPath::Inline)
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SsrPath {
    Inline,
    Trace,
    Composite,
}

fn brush_shader_modes(ssr: bool, sources: bool, ssr_debug: bool, path: SsrPath) -> String {
    brush_shader_probe(ssr, sources, ssr_debug, path, BrushProbe::Trace)
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BrushProbe {

    Trace,

    Pass,

    PassDeferred,

    Read,
}

const FS_MAIN: &str = "@fragment fn fs_main(";

fn brush_shader_probe(ssr: bool, sources: bool, ssr_debug: bool, path: SsrPath, probe: BrushProbe) -> String {

    let pass_like = matches!(probe, BrushProbe::Pass | BrushProbe::PassDeferred);
    let trace = path == SsrPath::Trace;
    let composite = path == SsrPath::Composite;
    let ssr_block = match (ssr, path) {
        (true, SsrPath::Trace) => crate::renderer::ssr::wgsl_ssr_block_shared_camera_radiance(3),

        (true, SsrPath::Composite) => crate::renderer::ssr::wgsl_ssr_composite_block(3),
        (true, SsrPath::Inline) => {
            crate::renderer::ssr::wgsl_ssr_block_shared_camera_debug(3, ssr_debug)
        }
        (false, _) => String::new(),
    };

    if BRUSH_LIGHT_DEBUG {
        return format!(
            r#"
{lights_block}

struct VIn {{
    @location(0) pos: vec3<f32>,
    @location(1) norm: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) material: u32,
    @location(5) tint: vec4<f32>,
    @location(6) uv2: vec2<f32>,
}}
struct VOut {{
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
}}
@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    out.clip = cam_view_proj() * vec4<f32>(v.pos, 1.0);
    out.normal = v.norm;
    out.world_pos = v.pos;
    return out;
}}
@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let d = light_debug(in.world_pos, normalize(in.normal));
    return vec4<f32>(clamp(d.x, 0.0, 1.0), clamp(d.y, 0.0, 1.0), 0.0, 1.0);
}}
"#,
            lights_block = wgsl_lights_block(0, 1),
        );
    }

    let (edge_varying, edge_vs, edge_debug) = if BRUSH_EDGE_DEBUG {
        (

            "    @location(9) @interpolate(perspective, centroid) world_pos_centroid: vec3<f32>,\n",
            "    out.world_pos_centroid = v.pos;\n",
            r#"
    let edge_delta = length(in.world_pos - in.world_pos_centroid);
    let edge_pixel = length(fwidth(in.world_pos));
    let edge_out = edge_delta / max(edge_pixel, 1e-6);
    return vec4<f32>(clamp(edge_out, 0.0, 1.0), 0.02, 0.02, 1.0);"#,
        )
    } else {
        ("", "", "")
    };
    let source_debug = if sources {
        r#"
    let dbg_sum = dbg_direct + dbg_baked + dbg_probe;
    let dbg_total = max(dbg_sum.r + dbg_sum.g + dbg_sum.b, 1e-6);
    return vec4<f32>(
        (dbg_direct.r + dbg_direct.g + dbg_direct.b) / dbg_total,
        (dbg_baked.r + dbg_baked.g + dbg_baked.b) / dbg_total,
        (dbg_probe.r + dbg_probe.g + dbg_probe.b) / dbg_total,
        1.0,
    );"#
    } else {
        ""
    };

    let ssr_apply = if ssr {
        r#"
    let ssr_rough = textureSampleLevel(mat_rough, mat_rough_samp, in.uv, i32(in.material), 4.0).r;
    let ssr_n = normalize(mix(n_geom, n, 1.0 - smoothstep(0.1, 0.4, ssr_rough)));
    let ssr_view = normalize(cam_pos() - in.world_pos);
    let ssr_cos = clamp(dot(ssr_n, ssr_view), 0.0, 1.0);
    let ssr_fresnel = 0.04 + 0.96 * pow(1.0 - ssr_cos, 5.0);

    let reflectivity = clamp((1.0 - ssr_rough) * ssr_fresnel, 0.0, MAX_SSR_REFLECTIVITY);
"#
    } else {
        r#"
    return vec4<f32>(tonemap(c), albedo.a * in.tint.a);"#
    };

    let ssr_tail = if !ssr {
        ""
    } else if composite {

        r#"
    let ssr_fallback = textureLoad(ssr_scene_color, vec2<i32>(in.clip.xy), 0).rgb;
    let refl = ssr_reflection_at(in.clip.xy);
    return vec4<f32>(
        mix(ssr_fallback, refl.rgb, clamp(reflectivity, 0.0, 1.0) * refl.a),
        albedo.a * in.tint.a,
    );"#
    } else if trace {

        r#"
    return ssr_reflect(in.world_pos, ssr_n, vec3<f32>(0.0), reflectivity, ssr_rough);"#
    } else {
        r#"
    let ssr_fallback = textureLoad(ssr_scene_color, vec2<i32>(in.clip.xy), 0).rgb;
    return vec4<f32>(
        ssr_reflect(in.world_pos, ssr_n, ssr_fallback, reflectivity, ssr_rough),
        albedo.a * in.tint.a,
    );"#
    };
    let ssr_apply = format!("{ssr_apply}{ssr_tail}");

    let ssr_apply = if probe == BrushProbe::Read && !ssr {
        let plain = "    return vec4<f32>(tonemap(c), albedo.a * in.tint.a);";
        assert!(ssr_apply.contains(plain), "the brush shader's plain return moved");
        ssr_apply.replacen(plain, "    return vec4<f32>(tonemap(c), reflection_alpha(lit));", 1)
    } else {
        ssr_apply
    };

    let lighting = if ssr && !sources {
        ""
    } else {
        r#"
    let rough = textureSample(mat_rough, mat_rough_samp, in.uv, i32(in.material)).r;

    let rough_aa = rough;

    terminator_width = terminator_width_of(length(tn));
    let ao = textureSample(mat_ao, mat_samp, in.uv, i32(in.material)).r;

    let lm_uv = clamp(in.uv2, in.uv2_rect.xy, in.uv2_rect.zw);
    let baked = textureSample(lm_tex, lm_samp, lm_uv) * vec4<f32>(vec3<f32>(exp2(-camera.proxy_params.z)), 1.0);

    let bdir = textureSample(lm_dir_tex, lm_samp, lm_uv);

    let sun_mask = textureSample(lm_sun_tex, lm_sun_samp, lm_uv);

    let sun_d = (sun_mask.r - 0.5) * (2.0 * SUN_MASK_DISTANCE_TEXELS);
    let sun_pen = max(sun_mask.g - 0.5, 0.0) * (2.0 * SUN_MASK_DISTANCE_TEXELS);
    let sun_w = max(max(sun_pen, 0.5 * fwidth(sun_d)), 0.02);
    receiver_sun_mask = select(-1.0, smoothstep(-sun_w, sun_w, sun_d), sun_mask.g > 0.25);

    receiver_skips_baked = true;

    let face_t = normalize(in.tangent.xyz - n_geom * dot(n_geom, in.tangent.xyz));
    let face_b = cross(n_geom, face_t) * in.tangent.w;
    let face_d = in.world_pos - in.face_centre;
    let face_pos = in.face_centre
        + face_t * clamp(dot(face_d, face_t), -in.face_half_extent.x, in.face_half_extent.x)
        + face_b * clamp(dot(face_d, face_b), -in.face_half_extent.y, in.face_half_extent.y);

    probe_volume_pos = vec4<f32>(in.face_centre, 1.0);
    let env_part = shade_material_env_part(
        face_pos, n, rough_aa, ao, baked.a, baked.rgb, bdir, albedo.rgb, face_pos,
        n_geom,
    );

    let st_layers = textureNumLayers(lm_stationary);
    let st_0 = textureSample(lm_stationary, lm_sun_samp, lm_uv, 0);
    var st_1 = vec4<f32>(1.0);
    var st_2 = vec4<f32>(1.0);
    var st_3 = vec4<f32>(1.0);
    if (st_layers > 1u) {
        st_1 = textureSample(lm_stationary, lm_sun_samp, lm_uv, 1);
    }
    if (st_layers > 2u) {
        st_2 = textureSample(lm_stationary, lm_sun_samp, lm_uv, 2);
    }
    if (st_layers > 3u) {
        st_3 = textureSample(lm_stationary, lm_sun_samp, lm_uv, 3);
    }
    set_stationary_masks(st_0, st_1, st_2, st_3, STATIONARY_MASK_DISTANCE_TEXELS);
    let lit = shade_material_lamps(env_part, face_pos, n, baked.rgb, albedo.rgb);
    let c = in.tint.rgb * lit;"#
    };

    let lighting: String = match probe {
        BrushProbe::Trace => lighting.to_string(),
        BrushProbe::Pass => probe_pass::PASS_LIGHTING.to_string(),

        BrushProbe::PassDeferred => {
            let given = "    probe_face_given = in.probe_face;\n";
            assert!(probe_pass::PASS_LIGHTING.contains(given), "the probe pass no longer hands in its face");
            probe_pass::PASS_LIGHTING.replacen(given, &format!("{given}    probe_fragment = in.clip;\n"), 1)
        }
        BrushProbe::Read => {
            let marker = "    probe_volume_pos = vec4<f32>(in.face_centre, 1.0);\n    let env_part = shade_material_env_part(";
            assert!(
                lighting.contains(marker),
                "the brush lighting no longer sets the probe position just before it shades; the pass read goes there",
            );
            format!(
                "    let probe_pass_tolerance = max(4.0 * fwidth(in.clip.z), 1e-6);\n{}",
                lighting.replacen(
                    marker,
                    &format!(
                        "    probe_env_given = probe_pass_upsample(in.clip.xy, in.clip.z, probe_pass_tolerance, probe_face_code(n_geom));\n{marker}"
                    ),
                    1,
                )
            )
        }
    };
    let src = format!(
        r#"
@group(2) @binding(0) var lm_tex: texture_2d<f32>;
@group(2) @binding(2) var lm_dir_tex: texture_2d<f32>;
@group(2) @binding(1) var lm_samp: sampler;
@group(2) @binding(3) var lm_sun_tex: texture_2d<f32>;
@group(2) @binding(4) var lm_sun_samp: sampler;

const SUN_MASK_DISTANCE_TEXELS: f32 = {sun_mask_range:?};

@group(2) @binding(5) var lm_stationary: texture_2d_array<f32>;
const STATIONARY_MASK_DISTANCE_TEXELS: f32 = {stationary_range:?};

@group(1) @binding(0) var mat_color: texture_2d_array<f32>;
@group(1) @binding(1) var mat_normal: texture_2d_array<f32>;
@group(1) @binding(2) var mat_samp: sampler;
@group(1) @binding(3) var mat_rough: texture_2d_array<f32>;
@group(1) @binding(4) var mat_ao: texture_2d_array<f32>;

@group(1) @binding(5) var mat_rough_samp: sampler;

@group(1) @binding(6) var<uniform> mat_uv_scale: array<vec4<f32>, {max_materials}>;

{lights_block}
{ssr_block}

struct VIn {{
    @location(0) pos: vec3<f32>,
    @location(1) norm: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) material: u32,
    @location(5) tint: vec4<f32>,
    @location(6) uv2: vec2<f32>,
    @location(7) face_centre: vec3<f32>,
    @location(8) uv2_rect: vec4<f32>,
    @location(9) face_half_extent: vec2<f32>,
}}
struct VOut {{

    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) tangent: vec4<f32>,

    @location(2) {centroid}world_pos: vec3<f32>,
    @location(3) uv: vec2<f32>,

    @location(4) @interpolate(flat) material: u32,
    @location(5) tint: vec4<f32>,
    @location(6) {centroid}uv2: vec2<f32>,

    @location(7) @interpolate(flat) face_centre: vec3<f32>,

    @location(8) @interpolate(flat) uv2_rect: vec4<f32>,

    @location(9) @interpolate(flat) face_half_extent: vec2<f32>,

    @location(11) room_near: f32,
{probe_face_varying}{edge_varying}}}

@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    out.clip      = cam_view_proj() * vec4<f32>(v.pos, 1.0);
    out.normal    = v.norm;
    out.tangent   = v.tangent;
    out.world_pos = v.pos;
    out.face_centre = v.face_centre;
    out.uv2_rect  = v.uv2_rect;
    out.face_half_extent = v.face_half_extent;

    out.uv        = v.uv * mat_uv_scale[min(v.material, {max_materials}u - 1u)].xy;
    out.uv2       = v.uv2;
    out.material  = v.material;
    out.tint      = v.tint;
    out.room_near = room_near(to_world_space(v.pos));
{probe_face_vs}{edge_vs}    return out;
}}

@fragment fn fs_depth() -> @location(0) vec4<f32> {{
    return vec4<f32>(0.0);
}}

@fragment fn fs_main(in: VOut) -> {fs_return} {{

    let room_reach = ROOM_REACH_PIXELS * max(length(dpdx(in.world_pos)), length(dpdy(in.world_pos)));
    if (in.room_near > 0.0) {{
        local_exposure = room_local_exposure(to_world_space(in.world_pos), room_reach);
    }}
    pixel_footprint = sqrt(length(dpdx(in.world_pos)) * length(dpdy(in.world_pos)));
    let albedo = textureSample(mat_color, mat_samp, in.uv, i32(in.material));

    let n_geom = normalize(in.normal);

    set_pixel_long_step(n_geom, normalize(cam_pos() - in.world_pos));
    let t = normalize(in.tangent.xyz - n_geom * dot(n_geom, in.tangent.xyz));
    let b = cross(n_geom, t) * in.tangent.w;
    let tn = textureSample(mat_normal, mat_samp, in.uv, i32(in.material)).xyz * 2.0 - 1.0;

    let n = normalize(t * tn.x - b * tn.y + n_geom * tn.z);

{lighting}{tail}
}}
"#,
        lights_block = crate::renderer::lights::wgsl_lights_block_with(
            0,
            1,
            crate::renderer::lights::LightsBlockOptions {
                probe_from_pass: probe == BrushProbe::Read,
                probe_face_always: pass_like,
                defer_secondary: probe == BrushProbe::PassDeferred,
                cull_range_first: false,
                card_tests_filtered: false,
                card_relit: false,
                long_step_from_plane: true,
                open_fallback: false,
            },
        ),
        ssr_block = ssr_block,
        sun_mask_range = SUN_MASK_DISTANCE_TEXELS,
        stationary_range = STATIONARY_MASK_DISTANCE_TEXELS,
        max_materials = MAX_BRUSH_MATERIALS,
        lighting = lighting,

        fs_return = if pass_like { "ProbePassOut" } else { "@location(0) vec4<f32>" },
        centroid = if BRUSH_CENTROID_VARYINGS {
            "@interpolate(perspective, centroid) "
        } else {
            ""
        },
        edge_varying = edge_varying,
        edge_vs = edge_vs,

        probe_face_varying = if pass_like {
            "    @location(10) @interpolate(flat) probe_face: vec4<f32>,\n"
        } else {
            ""
        },
        probe_face_vs = if pass_like {
            "    out.probe_face = probe_face_room(v.face_centre);\n"
        } else {
            ""
        },
        tail = if pass_like {
            String::new()
        } else if BRUSH_EDGE_DEBUG {
            edge_debug.to_string()
        } else if sources {
            match SOURCES_VIEW {
                SourcesView::Ratios => source_debug.to_string(),
                SourcesView::ProbeFactors => {
                    "\n    return vec4<f32>(dbg_probe_factors, 1.0);".to_string()
                }
                SourcesView::Absolute => r#"
    let abs_lum = vec3<f32>(
        dot(dbg_baked, vec3<f32>(0.2126, 0.7152, 0.0722)),
        dot(dbg_probe, vec3<f32>(0.2126, 0.7152, 0.0722)),
        dot(dbg_direct, vec3<f32>(0.2126, 0.7152, 0.0722)),
    );
    return vec4<f32>(abs_lum / (abs_lum + vec3<f32>(0.05)), 1.0);"#
                    .to_string(),
            }
        } else {
            ssr_apply
        }
    );

    let src = in_the_brushes_frame(src);
    match probe {
        BrushProbe::Read => {
            let src = if FACE_LAMP_LISTS { with_face_lamps(src) } else { src };
            format!("{src}{}{}", probe_pass::READER_WGSL, probe_pass::FACE_CODE_WGSL)
        }
        BrushProbe::Pass | BrushProbe::PassDeferred => {
            format!("{src}{}{}", probe_pass::OUTPUT_WGSL, probe_pass::FACE_CODE_WGSL)
        }
        BrushProbe::Trace => src,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::lights::{Light, LightKind, LightsUniform};
    use crate::renderer::Color3;
    use crate::renderer::terrain_pipeline::tests::headless_gpu;
    use crate::renderer::uniforms::test_support::{scene_uniforms, TEST_EYE};
    use crate::renderer::uniforms::ShadowUpload;
    use wgpu::util::DeviceExt;

    fn flat(rgb: [u8; 3], size: u32) -> TerrainImage {
        TerrainImage {
            width: size,
            height: size,
            rgba: (0..size * size)
                .flat_map(|_| [rgb[0], rgb[1], rgb[2], 255])
                .collect(),
        }
    }

    fn tilted_normal_up(size: u32) -> TerrainImage {
        TerrainImage {
            width: size,
            height: size,
            rgba: (0..size * size).flat_map(|_| [128, 250, 140, 255]).collect(),
        }
    }

    fn tilted_normal(size: u32) -> TerrainImage {
        TerrainImage {
            width: size,
            height: size,
            rgba: (0..size * size).flat_map(|_| [250, 128, 140, 255]).collect(),
        }
    }

    fn render_brush(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
    ) -> Option<[u8; 4]> {
        render_brush_material(
            material, colours, normals, &[], &[], tint, uv_scale, lit, SIDE_LIGHT,
        )
    }

    #[test]
    fn a_two_to_one_material_is_not_stretched_to_its_tile() {

        let halves = |w: u32, h: u32| TerrainImage {
            width: w,
            height: h,
            rgba: (0..h)
                .flat_map(|y| (0..w).flat_map(move |_| if y < h / 2 { [255, 0, 0, 255] } else { [0, 0, 255, 255] }))
                .collect(),
        };
        let Some(square) = render_brush(0, &[Some(halves(4, 4))], &[None], [1.0; 4], 0.8, false) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let wide = render_brush(0, &[Some(halves(4, 2))], &[None], [1.0; 4], 0.8, false).unwrap();
        assert!(square[0] > square[2] + 20, "a square map at v = 0.4 is in its red top half: {square:?}");
        assert!(wide[2] > wide[0] + 20, "a 2:1 map at v = 0.4 of a tile is in its blue bottom half: {wide:?}");
    }

    #[test]
    fn a_materials_uv_scale_is_its_width_over_its_height() {
        let img = |w, h| Some(TerrainImage { width: w, height: h, rgba: vec![0; (w * h * 4) as usize] });
        let scales = material_uv_scales(&[img(1024, 512), img(1024, 1024), None, img(512, 1024)]);
        assert_eq!(scales[0][..2], [1.0, 2.0]);
        assert_eq!(scales[1][..2], [1.0, 1.0]);
        assert_eq!(scales[2][..2], [1.0, 1.0], "a missing map is square");
        assert_eq!(scales[3][..2], [1.0, 0.5]);
        assert_eq!(scales[MAX_BRUSH_MATERIALS - 1][..2], [1.0, 1.0], "past the list, square");
    }

    const SIDE_LIGHT: glam::Vec3 = glam::Vec3::new(4.0, 0.0, 2.0);

    const HEAD_ON_LIGHT: glam::Vec3 = glam::Vec3::new(0.0, 0.0, 4.0);

    const NEUTRAL_LM: [u8; 4] = [0, 0, 0, 255];

    #[allow(clippy::too_many_arguments)]
    fn render_brush_material(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
        light_pos: glam::Vec3,
    ) -> Option<[u8; 4]> {
        render_brush_baked(
            material, colours, normals, roughs, aos, tint, uv_scale, lit, light_pos, NEUTRAL_LM,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_brush_baked(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
        light_pos: glam::Vec3,
        lightmap_texel: [u8; 4],
    ) -> Option<[u8; 4]> {
        render_brush_baked_dir(
            material, colours, normals, roughs, aos, tint, uv_scale, lit, light_pos,
            lightmap_texel,
            crate::renderer::mesh::NEUTRAL_BOUNCE_DIRECTION,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_brush_baked_dir(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
        light_pos: glam::Vec3,
        lightmap_texel: [u8; 4],
        direction_texel: [u8; 4],
    ) -> Option<[u8; 4]> {
        render_brush_probed(
            material, colours, normals, roughs, aos, tint, uv_scale, lit, light_pos,
            lightmap_texel, direction_texel, None,
        )
    }

    #[test]
    fn half_float_and_srgb_lightmaps_of_the_same_light_draw_the_same() {
        let at = |lm: crate::renderer::mesh::LightmapLight| {
            render_brush_lit_by(
                0, &[], &[], &[], &[], [1.0; 4], 1.0, false, glam::Vec3::ZERO,
                lm, crate::renderer::mesh::NEUTRAL_BOUNCE_DIRECTION, None, None,
            )
        };

        let Some(bytes) = at(crate::renderer::mesh::LightmapLight::Srgb8(&[188, 188, 188, 255])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let v = 0.5029f32;
        let floats = at(crate::renderer::mesh::LightmapLight::Linear(&[v, v, v, 1.0])).unwrap();
        for c in 0..3 {
            assert!((bytes[c] as i32 - floats[c] as i32).abs() <= 1, "{bytes:?} vs {floats:?}");
        }
        let bright = at(crate::renderer::mesh::LightmapLight::Linear(&[3.0, 3.0, 3.0, 1.0])).unwrap();
        let one = at(crate::renderer::mesh::LightmapLight::Linear(&[1.0, 1.0, 1.0, 1.0])).unwrap();
        assert!(bright[1] > one[1], "light past 1.0 was clipped: {bright:?} vs {one:?}");
    }

    #[test]
    fn the_sky_sun_on_a_brush_is_scaled_by_its_baked_mask() {
        let at = |mask: [u8; 4]| {
            render_brush_full(
                0, &[], &[], &[], &[], [1.0; 4], 1.0, false, glam::Vec3::ZERO,
                NEUTRAL_LM, crate::renderer::mesh::NEUTRAL_BOUNCE_DIRECTION, None, Some(mask),
            )
        };
        let Some(dark) = at([0, 255, 0, 255]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lit = at([255, 255, 0, 255]).unwrap();
        let half = at([128, 255, 0, 255]).unwrap();
        let unbaked = at([0, 0, 0, 0]).unwrap();
        eprintln!("mask 0: {dark:?}  128: {half:?}  255: {lit:?}  unbaked: {unbaked:?}");
        assert!(lit[1] > dark[1] + 40, "the sun did not light an unshadowed wall: {dark:?} vs {lit:?}");
        assert!(half[1] > dark[1] && half[1] < lit[1], "half a sun is not between none and all");
        assert!(
            (unbaked[1] as i32 - lit[1] as i32).abs() <= 2,
            "an unbaked mask read as shadow: {unbaked:?} vs {lit:?}",
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn render_brush_probed(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
        light_pos: glam::Vec3,
        lightmap_texel: [u8; 4],
        direction_texel: [u8; 4],
        probe: Option<([u8; 4], glam::Vec3, glam::Vec3)>,
    ) -> Option<[u8; 4]> {
        render_brush_full(
            material, colours, normals, roughs, aos, tint, uv_scale, lit, light_pos,
            lightmap_texel, direction_texel, probe, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_brush_full(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
        light_pos: glam::Vec3,
        lightmap_texel: [u8; 4],
        direction_texel: [u8; 4],
        probe: Option<([u8; 4], glam::Vec3, glam::Vec3)>,
        sky_sun: Option<[u8; 4]>,
    ) -> Option<[u8; 4]> {
        render_brush_lit_by(
            material, colours, normals, roughs, aos, tint, uv_scale, lit, light_pos,
            crate::renderer::mesh::LightmapLight::Srgb8(&lightmap_texel), direction_texel, probe, sky_sun,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_brush_lit_by(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
        light_pos: glam::Vec3,
        lightmap: crate::renderer::mesh::LightmapLight,
        direction_texel: [u8; 4],
        probe: Option<([u8; 4], glam::Vec3, glam::Vec3)>,
        sky_sun: Option<[u8; 4]>,
    ) -> Option<[u8; 4]> {
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        if sky_sun.is_some() {

            lights.upload_frame(
                &queue,
                &[Light {
                    mask_channel: None,
                    shadow_near: None,
                    source_radius: 0.0,
                    in_level_bake: true,
                    position: glam::Vec3::ZERO,
                    direction: glam::Vec3::new(0.0, 0.0, -1.0),
                    kind: LightKind::Directional,
                    color: Color3(255, 255, 255, 255),
                    intensity: 0.5,
                    range: 0.0,
                    cone_angle_deg: 0.0,
                    inner_cone_angle_deg: 0.0,
                }],
                &[],
                true,
            );
        } else if lit {

            lights.upload(
                &queue,
                &[Light {
                    mask_channel: None,
                    shadow_near: None,
                    source_radius: 0.0,
                    in_level_bake: true,
                    position: light_pos,
                    direction: glam::Vec3::new(-1.0, 0.0, 0.0),
                    kind: LightKind::Point,
                    color: Color3(255, 255, 255, 255),
                    intensity: 4.0,
                    range: 20.0,
                    cone_angle_deg: 90.0,
                    inner_cone_angle_deg: 0.0,
                }],
            );
        } else {
            lights.upload(&queue, &[]);
        }

        let (_shadows, uniforms, _probe_keep) = match probe {
            Some((colour, lo, hi)) => {
                let (sh, u, v, samp) = crate::renderer::uniforms::test_support::scene_uniforms_with_probe(
                    &device, &queue, &lights, colour, lo, hi,
                );
                (sh, u, Some((v, samp)))
            }
            None => {
                let (sh, u) = scene_uniforms(&device, &lights);
                (sh, u, None)
            }
        };
        uniforms.upload(&queue, glam::Mat4::IDENTITY, TEST_EYE, &ShadowUpload::disabled());

        let pipeline = BrushPipeline::new(&device, format, &uniforms.layout);
        let materials =

            BrushMaterials::new(
                &device, &queue, &pipeline.material_layout, colours, normals, roughs, aos,
            );

        let v = |p: [f32; 3], uv: [f32; 2]| BrushVertex {
            position: p,
            normal: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, -1.0],
            uv,
            material,
            tint,

            uv2: [0.5, 0.5],

            face_centre: [1.0 / 3.0, 1.0 / 3.0, 0.0],

            uv2_rect: [0.0, 0.0, 1.0, 1.0],

            face_half_extent: [f32::INFINITY; 2],
        };

        let verts = [
            v([-1.0, -1.0, 0.0], [0.0, uv_scale]),
            v([3.0, -1.0, 0.0], [2.0 * uv_scale, uv_scale]),
            v([-1.0, 3.0, 0.0], [0.0, -uv_scale]),
        ];

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brush_test_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brush_test_ib"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        const SIZE: u32 = 8;
        let target = device.create_texture(&TextureDescriptor {
            label: Some("brush_test_target"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&TextureDescriptor {
            label: Some("brush_test_depth"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());

        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("brush_test_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
        let sun_texel = sky_sun.unwrap_or(crate::renderer::mesh::NEUTRAL_SUN_MASK);
        let lightmap = crate::renderer::mesh::create_lightmap_texture_with_sun(
            &device,
            &queue,
            &pipeline.lightmap_layout,
            lightmap,
            1,
            1,
            Some((&direction_texel, 1, 1)),
            Some((&sun_texel, 1, 1)),
        );

            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("brush_test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &materials.bind_group, &[]);

            pass.set_bind_group(2, &lightmap.bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..3, 0, 0..1);
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
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        let c = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[c], data[c + 1], data[c + 2], data[c + 3]])
    }

    fn palette() -> Vec<Option<TerrainImage>> {
        vec![
            Some(flat([200, 20, 20], 4)),
            Some(flat([20, 200, 20], 4)),
            Some(flat([20, 20, 200], 4)),
        ]
    }

    #[test]
    fn a_face_samples_the_material_its_index_names() {

        for (index, expected) in [(0u32, 0usize), (1, 1), (2, 2)] {
            let Some(px) = render_brush(index, &palette(), &[], [1.0; 4], 1.0, false) else {
                eprintln!("skipping: no GPU adapter available");
                return;
            };
            let brightest = (0..3).max_by_key(|i| px[*i]).unwrap();
            assert_eq!(
                brightest, expected,
                "material {index} should sample layer {expected}, got {px:?}"
            );
        }
    }

    #[test]
    fn the_tint_multiplies_the_sampled_colour() {

        let Some(white) = render_brush(0, &[], &[], [1.0, 1.0, 1.0, 1.0], 1.0, false) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let red = render_brush(0, &[], &[], [1.0, 0.0, 0.0, 1.0], 1.0, false).unwrap();
        assert!(white[1] > 40, "an absent material is white, not black: {white:?}");
        assert!(red[0] > red[1], "the tint must reach the output: {red:?}");
        assert!(red[1] < 20, "and must actually remove the channels it zeroes");
    }

    #[test]
    fn a_missing_normal_map_leaves_the_face_flat() {

        let Some(with_fallback) = render_brush(0, &palette(), &[], [1.0; 4], 1.0, true) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let flat_map = render_brush(
            0,
            &palette(),
            &[Some(flat([128, 128, 255], 4))],
            [1.0; 4],
            1.0,
            true,
        )
        .unwrap();
        assert_eq!(
            with_fallback, flat_map,
            "no normal map must shade identically to an explicitly flat one"
        );
    }

    #[test]
    fn roughness_changes_the_specular_highlight() {

        let Some(matte) = render_brush_material(
            0, &palette(), &[None], &[Some(flat([255, 255, 255], 4))], &[], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let glossy = render_brush_material(
            0, &palette(), &[None], &[Some(flat([20, 20, 20], 4))], &[], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        )
        .unwrap();
        assert_ne!(
            matte, glossy,
            "roughness is bound but never reaches the shading: {matte:?} vs {glossy:?}",
        );
        assert!(
            glossy[0] > matte[0],
            "a smooth surface should out-shine a matte one head on: {glossy:?} vs {matte:?}",
        );
    }

    #[test]
    fn a_missing_roughness_map_takes_the_documented_default() {

        let d = DEFAULT_ROUGHNESS;
        let Some(absent) = render_brush_material(
            0, &palette(), &[None], &[], &[], [1.0; 4], 1.0, true, HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let explicit_default = render_brush_material(
            0, &palette(), &[None], &[Some(flat([d, d, d], 4))], &[], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        )
        .unwrap();
        assert_eq!(
            absent, explicit_default,
            "no roughness map must shade as DEFAULT_ROUGHNESS: {absent:?} vs {explicit_default:?}",
        );
    }

    #[test]
    fn ambient_occlusion_darkens_without_wiping_out_direct_light() {

        let Some(open) = render_brush_material(
            0, &palette(), &[None], &[], &[Some(flat([255, 255, 255], 4))], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let occluded = render_brush_material(
            0, &palette(), &[None], &[], &[Some(flat([0, 0, 0], 4))], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        )
        .unwrap();
        assert!(
            occluded[0] < open[0],
            "ambient occlusion darkened nothing: {occluded:?} vs {open:?}",
        );
        assert!(
            occluded[0] > 0,
            "AO wiped out the direct light too, not just the ambient: {occluded:?}",
        );
    }

    #[test]
    fn baked_sky_visibility_darkens_the_ambient_term() {

        let Some(open) = render_brush_baked(
            0, &palette(), &[None], &[], &[], [1.0; 4], 1.0, false, SIDE_LIGHT, [0, 0, 0, 255],
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let sealed = render_brush_baked(
            0, &palette(), &[None], &[], &[], [1.0; 4], 1.0, false, SIDE_LIGHT, [0, 0, 0, 0],
        )
        .unwrap();
        assert!(
            sealed[0] < open[0],
            "sky visibility darkened nothing: sealed {sealed:?} vs open {open:?}",
        );
    }

    #[test]
    fn baked_rgb_still_adds_with_sky_visibility_at_zero() {

        let Some(dark) = render_brush_baked(
            0, &palette(), &[None], &[], &[], [1.0; 4], 1.0, false, SIDE_LIGHT, [0, 0, 0, 0],
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let baked = render_brush_baked(
            0, &palette(), &[None], &[], &[], [1.0; 4], 1.0, false, SIDE_LIGHT, [200, 200, 200, 0],
        )
        .unwrap();
        assert!(
            baked[0] > dark[0],
            "baked light was scaled away by zero sky visibility: {baked:?} vs {dark:?}",
        );
    }

    #[test]
    #[ignore = "measurement, not an assertion: prints the specular response"]
    fn measure_specular_response() {
        let Some(_) = headless_gpu() else { return };
        for (label, rough) in [
            ("marble (0.048)", 12u8),
            ("default (0.55)", DEFAULT_ROUGHNESS),
            ("fully rough", 255u8),
        ] {
            let lit = render_brush_material(
                0, &palette(), &[None], &[Some(flat([rough, rough, rough], 4))], &[],
                [1.0; 4], 1.0, true, HEAD_ON_LIGHT,
            )
            .unwrap();
            let unlit = render_brush_material(
                0, &palette(), &[None], &[Some(flat([rough, rough, rough], 4))], &[],
                [1.0; 4], 1.0, false, HEAD_ON_LIGHT,
            )
            .unwrap();
            eprintln!("{label:>16}: lit {lit:?}  unlit {unlit:?}  delta {}", lit[0] as i32 - unlit[0] as i32);
        }
    }

    #[test]
    fn a_material_with_no_roughness_map_can_still_shine() {

        let Some(matte) = render_brush_material(
            0, &palette(), &[None], &[Some(flat([255, 255, 255], 4))], &[], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let absent = render_brush_material(
            0, &palette(), &[None], &[], &[], [1.0; 4], 1.0, true, HEAD_ON_LIGHT,
        )
        .unwrap();
        assert!(
            absent[0] > matte[0],
            "no roughness map must NOT mean a perfect diffuser: absent {absent:?} vs \
             explicitly-fully-rough {matte:?}",
        );
    }

    #[test]
    fn a_polished_surface_reflects_more_of_the_sky_than_a_rough_one() {

        let Some(smooth) = render_brush_material(
            0, &palette(), &[Some(tilted_normal(4))], &[Some(flat([12, 12, 12], 4))], &[],
            [1.0; 4], 1.0, false, HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let rough = render_brush_material(
            0, &palette(), &[Some(tilted_normal(4))], &[Some(flat([255, 255, 255], 4))], &[],
            [1.0; 4], 1.0, false, HEAD_ON_LIGHT,
        )
        .unwrap();
        assert!(
            smooth[0] > rough[0],
            "a near-mirror must reflect more environment than a diffuser: \
             {smooth:?} vs {rough:?}",
        );
    }

    #[test]
    fn a_sealed_room_does_not_reflect_a_sky_it_cannot_see() {

        let sealed = |rough: u8| {
            render_brush_baked(
                0, &palette(), &[Some(tilted_normal(4))],
                &[Some(flat([rough, rough, rough], 4))], &[],
                [1.0; 4], 1.0, false, HEAD_ON_LIGHT, [0, 0, 0, 0],
            )
        };
        let Some(smooth) = sealed(12) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let rough = sealed(255).unwrap();
        assert_eq!(
            smooth, rough,
            "with no sky visible, a polished surface reflected something a matte \
             one did not: {smooth:?} vs {rough:?}",
        );
    }

    #[test]
    fn a_polished_surface_in_a_sealed_room_reflects_the_room_itself() {

        let room_lit = |rough: u8| {
            render_brush_baked(
                0, &palette(), &[Some(tilted_normal(4))],
                &[Some(flat([rough, rough, rough], 4))], &[],
                [1.0; 4], 1.0, false, HEAD_ON_LIGHT,

                [180, 180, 180, 0],
            )
        };
        let Some(smooth) = room_lit(12) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let rough = room_lit(255).unwrap();

        let room = smooth[1] as i32 - rough[1] as i32;
        assert!(
            room > 2,
            "a polished surface in a lit but sealed room reflected no more of \
             that room than a matte one: {smooth:?} vs {rough:?}",
        );

        let total = |p: [u8; 4]| p[0] as i32 + p[1] as i32 + p[2] as i32;
        assert!(
            total(smooth) > total(rough),
            "the polished surface is not showing more light in total: \
             {smooth:?} vs {rough:?}",
        );
    }

    #[test]
    fn bounced_light_falls_on_the_side_of_a_surface_it_came_from() {

        let lit_from = |dir: [u8; 4]| {
            render_brush_baked_dir(

                0, &palette(), &[Some(tilted_normal_up(4))],
                &[Some(flat([200, 200, 200], 4))], &[],
                [1.0; 4], 1.0, false, HEAD_ON_LIGHT,

                [170, 170, 170, 0],
                dir,
            )
        };

        let from_above = lit_from([128, 255, 128, 255]);
        let Some(above) = from_above else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let below = lit_from([128, 0, 128, 255]).unwrap();

        let gap = (above[0] as i32 - below[0] as i32).abs();
        assert!(
            gap > 4,
            "bounce arriving from above ({above:?}) and from below ({below:?}) \
             differ by only {gap}; the direction is barely reaching the surface",
        );
        assert!(
            above[0] > below[0],
            "a surface tilted UP should catch more of a bounce arriving from \
             above than from below: {above:?} vs {below:?}",
        );
    }

    #[test]
    fn a_bounce_with_no_direction_shades_exactly_as_it_always_did() {

        let baked = [170u8, 170, 170, 0];
        let flat_render = render_brush_baked(
            0, &palette(), &[Some(tilted_normal(4))],
            &[Some(flat([200, 200, 200], 4))], &[],
            [1.0; 4], 1.0, false, HEAD_ON_LIGHT, baked,
        );
        let Some(flat_px) = flat_render else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };

        let weightless = render_brush_baked_dir(
            0, &palette(), &[Some(tilted_normal(4))],
            &[Some(flat([200, 200, 200], 4))], &[],
            [1.0; 4], 1.0, false, HEAD_ON_LIGHT, baked,
            [128, 255, 128, 0],
        ).unwrap();
        assert_eq!(
            flat_px, weightless,
            "a direction with zero directionality changed the shading",
        );
    }

    #[test]
    fn an_unlit_sealed_room_still_reflects_nothing() {

        let dark = |rough: u8| {
            render_brush_baked(
                0, &palette(), &[Some(tilted_normal(4))],
                &[Some(flat([rough, rough, rough], 4))], &[],
                [1.0; 4], 1.0, false, HEAD_ON_LIGHT, [0, 0, 0, 0],
            )
        };
        let Some(smooth) = dark(12) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(smooth, dark(255).unwrap());
    }

    #[test]
    fn a_highlight_is_not_tinted_by_the_surface_colour() {

        let shine = |rough: u8| {
            render_brush_material(
                0, &palette(), &[None],
                &[Some(flat([rough, rough, rough], 4))], &[],
                [1.0; 4], 1.0, true, HEAD_ON_LIGHT,
            )
        };

        let Some(px) = shine(12) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };

        let matte = shine(255).unwrap();
        assert!(
            px[1] as i32 > matte[1] as i32 && px[2] as i32 > matte[2] as i32,
            "a polished red surface ({px:?}) carries no more green or blue than a \
             matte one ({matte:?}), so it has no untinted highlight at all",
        );

        let dg = px[1] as i32 - matte[1] as i32;
        let db = px[2] as i32 - matte[2] as i32;
        assert!(
            (dg - db).abs() <= 2,
            "the highlight added {dg} green and {db} blue -- it is carrying the \
             albedo's hue, so specular is still being multiplied by the diffuse \
             colour ({px:?} against {matte:?})",
        );
    }

    #[test]
    fn a_matte_surface_of_the_same_colour_has_no_such_highlight() {

        let Some(matte) = render_brush_material(
            0, &palette(), &[None],
            &[Some(flat([255, 255, 255], 4))], &[],
            [1.0; 4], 1.0, true, HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let shiny = render_brush_material(
            0, &palette(), &[None], &[Some(flat([12, 12, 12], 4))], &[],
            [1.0; 4], 1.0, true, HEAD_ON_LIGHT,
        ).unwrap();
        assert!(
            shiny[1] > matte[1],
            "polish added no untinted light: matte {matte:?} vs shiny {shiny:?}",
        );
    }

    #[test]
    fn a_near_mirror_still_produces_a_visible_highlight() {

        let Some(mirror) = render_brush_material(
            0, &palette(), &[None], &[Some(flat([12, 12, 12], 4))], &[], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let rough = render_brush_material(
            0, &palette(), &[None], &[Some(flat([255, 255, 255], 4))], &[], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        )
        .unwrap();
        assert!(
            mirror[0] > rough[0],
            "a near-mirror must be brighter at the mirror angle than a diffuser: \
             {mirror:?} vs {rough:?}",
        );
    }

    #[test]
    fn a_normal_map_changes_the_shading() {

        let Some(flat_px) = render_brush(
            0,
            &palette(),
            &[Some(flat([128, 128, 255], 4))],
            [1.0; 4],
            1.0,
            true,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let tilted =
            render_brush(0, &palette(), &[Some(tilted_normal(4))], [1.0; 4], 1.0, true).unwrap();
        assert_ne!(
            flat_px, tilted,
            "the tangent frame is discarding the normal map: {flat_px:?} vs {tilted:?}"
        );
        assert!(
            tilted[0] > flat_px[0],
            "tilted toward the light should be brighter: {flat_px:?} vs {tilted:?}"
        );
    }

    #[test]
    fn normal_maps_light_bumps_from_the_side_the_light_is_on() {
        let tilted = |green: u8| TerrainImage {
            width: 4,
            height: 4,
            rgba: (0..16).flat_map(|_| [128, green, 200, 255]).collect(),
        };

        let above = glam::Vec3::new(0.3, 1.5, 0.8);
        let grey = [Some(flat([90, 90, 90], 4))];
        let Some(faces_up) = render_brush_material(
            0, &grey, &[Some(tilted(230))], &[], &[], [1.0; 4], 1.0, true, above,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let faces_down =
            render_brush_material(0, &grey, &[Some(tilted(26))], &[], &[], [1.0; 4], 1.0, true, above).unwrap();
        let flat_px =
            render_brush_material(0, &grey, &[Some(flat([128, 128, 255], 4))], &[], &[], [1.0; 4], 1.0, true, above)
                .unwrap();
        let lum = |p: [u8; 4]| p[0] as i32 + p[1] as i32 + p[2] as i32;
        eprintln!("faces up {faces_up:?}  flat {flat_px:?}  faces down {faces_down:?}");
        assert!(faces_up[0] < 250 && faces_up[1] < 250, "clipped, so the comparison means nothing: {faces_up:?}");
        assert!(
            lum(faces_up) > lum(flat_px) && lum(flat_px) > lum(faces_down),
            "a texel facing the top of the picture must be lit best by a lamp above: \
             up {faces_up:?}, flat {flat_px:?}, down {faces_down:?}",
        );
        assert!(
            lum(faces_up) > lum(faces_down) + 30,
            "barely separated -- the tilt is hardly reaching the shading: {faces_up:?} vs {faces_down:?}",
        );
    }

    #[test]
    fn a_material_beyond_the_array_does_not_read_out_of_bounds() {

        let Some(px) = render_brush(
            (MAX_BRUSH_MATERIALS - 1) as u32,
            &palette(),
            &[],
            [1.0; 4],
            1.0,
            false,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(px[3] > 0, "the last layer must be bound and drawable: {px:?}");
    }

    fn striped() -> TerrainImage {
        let cols = [[0u8, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 255]];
        let mut rgba = Vec::with_capacity(4 * 4 * 4);
        for _row in 0..4 {
            for c in cols {
                rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        TerrainImage { width: 4, height: 4, rgba }
    }

    #[test]
    fn uv_beyond_one_repeats_rather_than_smearing() {

        let scales = [4.0f32, 5.0, 6.0, 7.0];
        let mut seen = std::collections::HashSet::new();
        for s in scales {
            let Some(px) = render_brush(0, &[Some(striped())], &[], [1.0; 4], s, false) else {
                eprintln!("skipping: no GPU adapter available");
                return;
            };
            seen.insert([px[0], px[1], px[2]]);
        }
        assert!(
            seen.len() > 1,
            "every scale past one tile sampled the same colour, which is what \
             clamping does: {seen:?}"
        );
    }

    mod probe_render {
        use super::*;

        use super::tests::*;
        use super::*;

        const SEALED: [u8; 4] = [0, 0, 0, 0];

        fn polished_wall(probe: Option<([u8; 4], glam::Vec3, glam::Vec3)>) -> Option<[u8; 4]> {
            render_brush_probed(
                0,
                &palette(),
                &[Some(tilted_normal(4))],

                &[Some(flat([12, 12, 12], 4))],
                &[],
                [1.0; 4],
                1.0,
                false,
                HEAD_ON_LIGHT,
                SEALED,
                crate::renderer::mesh::NEUTRAL_BOUNCE_DIRECTION,
                probe,
            )
        }

        #[test]
        fn a_probe_lights_a_polished_surface_that_has_nothing_else_to_reflect() {
            let big = (glam::Vec3::splat(-50.0), glam::Vec3::splat(50.0));
            let Some(without) = polished_wall(None) else {
                eprintln!("skipping: no GPU adapter available");
                return;
            };

            let with = polished_wall(Some(([0, 255, 0, 255], big.0, big.1))).unwrap();
            assert!(
                with[1] as i32 - without[1] as i32 > 4,
                "a probe covering the surface changed it from {without:?} to {with:?}; \
                 the cube is not reaching the shader",
            );
        }

        #[test]
        fn a_probe_elsewhere_in_the_level_does_not_reach_this_surface() {
            let far = (glam::Vec3::new(500.0, 500.0, 500.0), glam::Vec3::new(600.0, 600.0, 600.0));
            let Some(without) = polished_wall(None) else {
                eprintln!("skipping: no GPU adapter available");
                return;
            };
            let with = polished_wall(Some(([0, 255, 0, 255], far.0, far.1))).unwrap();
            assert_eq!(
                without, with,
                "a probe whose box is 500m away still lit this surface: {without:?} \
                 vs {with:?}",
            );
        }

        #[test]
        fn roughness_still_governs_how_much_of_the_probe_shows() {
            let big = (glam::Vec3::splat(-50.0), glam::Vec3::splat(50.0));
            let probe = Some(([0u8, 255, 0, 255], big.0, big.1));
            let rough = render_brush_probed(
                0, &palette(), &[Some(tilted_normal(4))],
                &[Some(flat([255, 255, 255], 4))], &[],
                [1.0; 4], 1.0, false, HEAD_ON_LIGHT, SEALED, crate::renderer::mesh::NEUTRAL_BOUNCE_DIRECTION, probe,
            );
            let Some(rough) = rough else {
                eprintln!("skipping: no GPU adapter available");
                return;
            };
            let smooth = polished_wall(probe).unwrap();
            assert!(
                smooth[1] > rough[1],
                "a polished surface ({smooth:?}) reflected no more of the probe than \
                 a matte one ({rough:?})",
            );
        }

        #[test]
        fn a_matte_surface_takes_nothing_from_the_probe() {
            let big = (glam::Vec3::splat(-50.0), glam::Vec3::splat(50.0));
            let lit_matte = |probe| {
                render_brush_probed(
                    0, &palette(), &[Some(tilted_normal(4))],
                    &[Some(flat([255, 255, 255], 4))], &[],
                    [1.0; 4], 1.0, false, HEAD_ON_LIGHT,

                    [180, 180, 180, 0],
                    crate::renderer::mesh::NEUTRAL_BOUNCE_DIRECTION, probe,
                )
            };
            let Some(without) = lit_matte(None) else {
                eprintln!("skipping: no GPU adapter available");
                return;
            };

            let with = lit_matte(Some(([0, 255, 0, 255], big.0, big.1))).unwrap();
            assert_eq!(
                without, with,
                "a matte surface changed from {without:?} to {with:?} when a probe \
                 covered it; the probe's fixtures are reaching rough geometry",
            );
        }
    }
}

#[cfg(test)]
mod ssr_pipeline_tests {
    use super::*;
    use crate::renderer::terrain_pipeline::tests::headless_gpu;
    use crate::renderer::lights::LightsUniform;
    use crate::renderer::uniforms::test_support::scene_uniforms;

    #[test]
    fn the_reflective_brush_pipeline_builds_within_the_bind_group_limit() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let _ = &queue;
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);

        for scene_samples in [1u32, 4] {
            let ssr = crate::renderer::ssr::SsrPipelines::new_with_depth_samples(
                &device,
                TextureFormat::Rgba8UnormSrgb,
                scene_samples,
            );
            let err_scope_1 = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let _pipeline = BrushPipeline::new_ssr(
                &device,
                TextureFormat::Rgba8UnormSrgb,
                &uniforms.layout,
                ssr.scene_texture_layout(),
            );
            for view in [DebugView::Sources, DebugView::Ssr] {
                let _p = BrushPipeline::new_ssr_debug(
                    &device,
                    TextureFormat::Rgba8UnormSrgb,
                    &uniforms.layout,
                    ssr.scene_texture_layout(),
                    view,
                );
            }

            let _trace = BrushPipeline::new_ssr_trace(
                &device,
                crate::renderer::ssr::REFLECTION_FORMAT,
                &uniforms.layout,
                ssr.scene_texture_layout(),
            );

            let _composite = BrushPipeline::new_ssr_composite(
                &device,
                TextureFormat::Rgba8UnormSrgb,
                &uniforms.layout,
                ssr.scene_texture_layout(),
            );
            let _solid = crate::renderer::pipeline::SolidPipeline::new_ssr(
                &device,
                TextureFormat::Rgba8UnormSrgb,
                &uniforms.layout,
                ssr.camera_layout(),
                ssr.scene_texture_layout(),
            );
            let err = pollster::block_on(err_scope_1.pop());
            assert!(
                err.is_none(),
                "reflective brush pipeline failed to build \
                 (scene_samples={scene_samples}): {err:?}",
            );
        }
    }

    #[test]
    fn a_debug_view_changes_only_the_shader_it_is_built_for() {
        assert!(!BRUSH_SOURCE_DEBUG && !crate::renderer::ssr::SSR_DEBUG, "a build switch is on");
        for ssr in [false, true] {
            assert_eq!(brush_shader_variant(ssr, false, false), brush_shader_with(ssr));
        }
        let sources = brush_shader_variant(false, true, false);

        let marker = match SOURCES_VIEW {
            SourcesView::Ratios => "dbg_total",
            SourcesView::ProbeFactors => "dbg_probe_factors",
            SourcesView::Absolute => "abs_lum",
        };
        assert!(
            sources.contains(marker),
            "the sources view does not paint the {SOURCES_VIEW:?} picture it is set to",
        );
        let ssr_debug = brush_shader_variant(true, false, true);
        assert!(ssr_debug.contains("MAGENTA: faces the viewer"), "the SSR view does not paint the SSR paths");
        assert!(!brush_shader_with(true).contains("MAGENTA"), "the shipped shader paints SSR paths");
        assert_eq!(DebugView::Off.next().next().next(), DebugView::Off, "the cycle does not return to Off");
        assert_eq!(DebugView::Off.toggle_sources(), DebugView::Sources);
        assert_eq!(DebugView::Sources.toggle_sources(), DebugView::Off);
        assert_eq!(DebugView::Ssr.toggle_sources(), DebugView::Off, "the tester's toggle must never reach the SSR view");
    }

    #[test]
    fn the_stereo_probe_pass_shaders_validate_as_multiview() {
        for (probe, sources) in [(BrushProbe::Pass, false), (BrushProbe::Read, BRUSH_SOURCE_DEBUG)] {
            let src = brush_shader_probe(false, sources, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, probe);
            assert_eq!(crate::renderer::multiview::multiview_validation_error_of(&src, false), None, "{probe:?} before the transform");
            assert_eq!(crate::renderer::multiview::multiview_validation_error(&src), None, "{probe:?} as multiview");
            if probe == BrushProbe::Read {
                assert!(crate::renderer::multiview::as_multiview(&src).contains("view_slot = i32(view_index_in);"));
            }
        }
    }

    #[test]
    fn the_floor_mirrors_depth_turns_back_into_a_distance() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };

        let proj = crate::renderer::camera::Camera::gl_to_wgpu_ndc(glam::Mat4::perspective_rh_gl(
            1.6,
            0.95,
            probe_pass::EYE_NEAR,
            probe_pass::EYE_FAR,
        ));
        let distances = [0.05f32, 0.5, 1.7, 3.0, 12.0, 80.0];
        let depths: Vec<f32> = distances
            .iter()
            .map(|&z| {
                let c = proj * glam::Vec4::new(0.3, -0.2, -z, 1.0);
                c.z / c.w
            })
            .collect();
        let code = format!(
            "{}\n@group(0) @binding(0) var<storage, read_write> io: array<f32>;\n@compute @workgroup_size(1)\nfn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n    io[id.x] = floor_mirror_linear_depth(io[id.x]);\n}}\n",
            probe_pass::eye_linear_depth_wgsl()
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let io = wgpu::util::DeviceExt::create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&depths),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        );
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (depths.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: io.as_entire_binding() }],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(depths.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&io, 0, &read, 0, (depths.len() * 4) as u64);
        queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let back: Vec<f32> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
        for (z, got) in distances.iter().zip(&back) {

            assert!((got - z).abs() < 0.002 * z + 1e-4, "{z} m came back {got}");
        }
    }

    #[test]
    fn the_floor_mirrors_blur_levels_average_the_level_below() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = probe_pass::bind_group_layout(&device);

        let target = probe_pass::Target::new(&device, &layout, 16, 16, 1);
        let levels = &target.mirror_levels[0];
        assert_eq!(levels.len(), 4);

        let mut texels = vec![0u16; 8 * 8 * 4];
        let one = crate::renderer::sky::f32_to_f16(1.0);
        for y in (0..8).step_by(2) {
            for x in (0..8).step_by(2) {
                let i = (y * 8 + x) * 4;
                texels[i..i + 4].copy_from_slice(&[one, one, one, one]);
            }
        }
        let texture = target.mirror_texture();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8 * 8), rows_per_image: Some(8) },
            wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
        );
        let mips = probe_pass::MirrorMips::new(&device);
        let mut enc = device.create_command_encoder(&Default::default());
        mips.record(&device, &mut enc, levels, (target.width, target.height), None);

        let read = |enc: &mut wgpu::CommandEncoder, level: u32, size: u32| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256 * size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture, mip_level: level, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(size) },
                },
                wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            );
            buffer
        };
        let (l1, l3) = (read(&mut enc, 1, 4), read(&mut enc, 3, 1));
        queue.submit([enc.finish()]);
        for b in [&l1, &l3] {
            b.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        let half_to_f32 = |h: u16| -> f32 {
            let exp = ((h >> 10) & 0x1f) as i32;
            let mant = (h & 0x3ff) as f32 / 1024.0;
            if exp == 0 { 0.0 } else { (1.0 + mant) * 2f32.powi(exp - 15) }
        };
        let half = |b: &wgpu::Buffer| -> Vec<f32> {
            let bytes = b.slice(..).get_mapped_range().unwrap();
            bytemuck::cast_slice::<u8, u16>(&bytes).iter().map(|&h| half_to_f32(h)).collect()
        };
        let (a, c) = (half(&l1), half(&l3));
        for k in 0..4 {
            assert!((a[k] - 0.25).abs() < 1e-3, "level 1, first texel, channel {k}: {}", a[k]);
            assert!((c[k] - 0.25).abs() < 1e-3, "level 3, channel {k}: {}", c[k]);
        }
    }

    #[test]
    fn the_floor_mirrors_blur_levels_at_odd_sizes_match_their_means() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = probe_pass::bind_group_layout(&device);
        let target = probe_pass::Target::new(&device, &layout, 2 * 45, 2 * 49, 1);
        let (w0, h0) = (target.width, target.height);
        let levels = &target.mirror_levels[0];
        assert_eq!(levels.len(), 5, "{w0}x{h0}");
        let value = |x: u32, y: u32, c: u32| ((x * 7 + y * 13 + c * 5) % 17) as f32 / 16.0;
        let mut texels = vec![0u16; (w0 * h0 * 4) as usize];
        for y in 0..h0 {
            for x in 0..w0 {
                for c in 0..4 {
                    texels[((y * w0 + x) * 4 + c) as usize] = crate::renderer::sky::f32_to_f16(value(x, y, c));
                }
            }
        }
        let texture = target.mirror_texture();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w0 * 8), rows_per_image: Some(h0) },
            wgpu::Extent3d { width: w0, height: h0, depth_or_array_layers: 1 },
        );
        let mut enc = device.create_command_encoder(&Default::default());
        probe_pass::MirrorMips::new(&device).record(&device, &mut enc, levels, (w0, h0), None);
        let dims: Vec<(u32, u32)> = (0..5).map(|k| ((w0 >> k).max(1), (h0 >> k).max(1))).collect();
        let reads: Vec<(wgpu::Buffer, u32)> = (1..5)
            .map(|k| {
                let (w, h) = dims[k];
                let row = (w * 8).div_ceil(256) * 256;
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (row * h) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                enc.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo { texture, mip_level: k as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &buffer,
                        layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
                    },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
                (buffer, row)
            })
            .collect();
        queue.submit([enc.finish()]);
        for (b, _) in &reads {
            b.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let half_to_f32 = |h: u16| -> f32 {
            let exp = ((h >> 10) & 0x1f) as i32;
            let mant = (h & 0x3ff) as f32 / 1024.0;
            if exp == 0 { mant * 2f32.powi(-14) } else { (1.0 + mant) * 2f32.powi(exp - 15) }
        };

        let mut below: Vec<f32> = texels.iter().map(|&h| half_to_f32(h)).collect();
        for k in 1..5 {
            let ((wb, _), (w, h)) = (dims[k - 1], dims[k]);
            let (buffer, row) = &reads[k - 1];
            let bytes = buffer.slice(..).get_mapped_range().unwrap();
            let mut got = vec![0f32; (w * h * 4) as usize];
            for y in 0..h {
                for x in 0..w {
                    for c in 0..4 {
                        let at = (y * row + x * 8 + c * 2) as usize;
                        let v = half_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
                        let tap = |dx: u32, dy: u32| below[(((2 * y + dy) * wb + 2 * x + dx) * 4 + c) as usize];
                        let want = 0.25 * (tap(0, 0) + tap(1, 0) + tap(0, 1) + tap(1, 1));
                        assert!((v - want).abs() < 2e-3, "level {k} texel {x},{y} channel {c}: {v} vs {want}");
                        got[((y * w + x) * 4 + c) as usize] = v;
                    }
                }
            }
            below = got;
        }
    }

    #[test]
    fn every_brush_shader_takes_its_vertices_in_the_world() {
        let variants: Vec<(&str, String)> = vec![
            ("trace", brush_shader_probe(false, false, false, SsrPath::Inline, BrushProbe::Trace)),
            ("sources", brush_shader_probe(false, true, false, SsrPath::Inline, BrushProbe::Trace)),
            ("ssr", brush_shader_probe(true, false, false, SsrPath::Inline, BrushProbe::Trace)),
            ("ssr trace", brush_shader_probe(true, false, false, SsrPath::Trace, BrushProbe::Trace)),
            ("ssr composite", brush_shader_probe(true, false, false, SsrPath::Composite, BrushProbe::Trace)),
            ("probe pass", brush_shader_probe(false, false, false, SsrPath::Inline, BrushProbe::Pass)),
            ("probe pass deferred", brush_shader_probe(false, false, false, SsrPath::Inline, BrushProbe::PassDeferred)),
            ("probe reader", brush_shader_probe(false, false, false, SsrPath::Inline, BrushProbe::Read)),
        ];
        for (name, posed) in variants {
            if BRUSH_LIGHT_DEBUG {
                assert!(world_brush_vertices(&posed).is_none());
                continue;
            }
            let world = world_brush_vertices(&posed).unwrap_or_else(|| panic!("{name}: does not open as a brush shader"));
            for line in ["v.pos = to_player_space(w.pos);", "v.norm = to_player_direction(w.norm);", "to_player_direction(w.tangent.xyz), w.tangent.w", "v.face_centre = to_player_space(w.face_centre);"] {
                assert_eq!(world.matches(line).count(), 1, "{name}: {line}");
            }

            assert_eq!(world.replacen(WORLD_BRUSH_VERTEX, POSED_BRUSH_VERTEX, 1), posed, "{name}");
            let from = posed.find("@fragment fn fs_main(").expect("a fragment stage");
            assert!(world.ends_with(&posed[from..]), "{name}: the fragment stage changed");
            naga::front::wgsl::parse_str(&world).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&world)));
        }

        let [r, g, b] = CRACK_SEAL_COLOUR;
        let seal = brush_seal_shader_posed(r, g, b);
        assert_eq!(seal.matches(POSED_SEAL_VERTEX).count(), 1);
        let world = seal.replacen(POSED_SEAL_VERTEX, WORLD_SEAL_VERTEX, 1);
        naga::front::wgsl::parse_str(&world).unwrap_or_else(|e| panic!("seal: {}", e.emit_to_string(&world)));
    }

    #[test]
    fn the_probe_upsample_is_the_depth_aware_bilinear_filter() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        const W: u32 = 64;
        const H: u32 = 32;
        const TOLERANCE: f32 = 0.0015;

        const PATTERN: &str = r#"
fn pass_colour(t: vec2<f32>) -> vec4<f32> {
    let a = 0.25 + 0.5 * fract(t.x * 0.37 + t.y * 0.11);
    return vec4<f32>(vec3<f32>(t.x / 32.0, t.y / 16.0, 0.5) * a, a);
}
fn pass_depth(t: vec2<f32>) -> f32 {
    return 0.3 + 0.4 * step(16.0, t.x) + 0.15 * step(8.0, t.y) + 0.001 * t.y;
}

fn pass_code(t: vec2<f32>) -> f32 {
    if (t.x < 8.0) { return 529.0; }
    if (t.x < 16.0) { return 533.0; }
    return 0.0;
}

fn pixel_code(p: vec2<f32>) -> f32 {
    if (p.x < 8.0 && p.y >= 28.0) { return 1.0; }
    if (p.x < 16.0) { return 530.0; }
    if (p.x < 32.0) { return 533.0; }
    return 0.0;
}

fn pixel_depth(p: vec2<f32>) -> f32 {
    return 0.3 + 0.4 * step(32.0, p.x) + 0.15 * step(16.0, p.y) + 0.001 * (p.y * 0.5 - 0.5);
}
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    return vec4<f32>(xy, 0.0, 1.0);
}
struct PassOut {
    @location(0) colour: vec4<f32>,
    @location(1) code: vec2<f32>,
    @builtin(frag_depth) depth: f32,
}
@fragment
fn fill(@builtin(position) pos: vec4<f32>) -> PassOut {
    let t = floor(pos.xy);
    return PassOut(pass_colour(t), vec2<f32>(0.0, pass_code(t)), pass_depth(t));
}
"#;
        let layout = probe_pass::bind_group_layout(&device);
        let target = probe_pass::Target::new(&device, &layout, W, H, 1);
        let fill_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("upsample_fill"),
            source: wgpu::ShaderSource::Wgsl(PATTERN.into()),
        });
        let fill = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("upsample_fill"),
            layout: None,
            vertex: wgpu::VertexState { module: &fill_module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &fill_module,
                entry_point: Some("fill"),
                targets: &[Some(probe_pass::FORMAT.into()), Some(probe_pass::REACH_FORMAT.into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let read_src = format!(
            "{PATTERN}var<private> view_slot: i32 = 0;\n{}{}\n@fragment\nfn read(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {{\n    return probe_pass_upsample(pos.xy, pixel_depth(pos.xy), {TOLERANCE:?}, pixel_code(pos.xy));\n}}\n",
            probe_pass::READER_WGSL,
            probe_pass::FACE_CODE_WGSL
        );
        let read_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("upsample_read"),
            source: wgpu::ShaderSource::Wgsl(read_src.into()),
        });
        let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[] });
        let empty_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &empty, entries: &[] });
        let read_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("upsample_read"),
            bind_group_layouts: &[Some(&empty), Some(&empty), Some(&empty), Some(&layout)],
            immediate_size: 0,
        });
        let read = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("upsample_read"),
            layout: Some(&read_layout),
            vertex: wgpu::VertexState { module: &read_module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &read_module,
                entry_point: Some("read"),
                targets: &[Some(TextureFormat::Rgba32Float.into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let out = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("upsample_out"),
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let out_view = out.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("upsample_fill"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &target.color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &target.reach_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&fill);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("upsample_read"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&read);
            for g in 0..3 {
                pass.set_bind_group(g, &empty_bg, &[]);
            }
            pass.set_bind_group(3, &target.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        let row = W * 16;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upsample_readback"),
            size: u64::from(row * H),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &out, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(H) },
            },
            wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = readback.slice(..).get_mapped_range().unwrap();
        let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);

        let step = |edge: f32, x: f32| if x >= edge { 1.0 } else { 0.0 };
        let colour = |t: [f32; 2]| {
            let a = 0.25 + 0.5 * (t[0] * 0.37 + t[1] * 0.11).fract();
            [t[0] / 32.0 * a, t[1] / 16.0 * a, 0.5 * a, a]
        };
        let texel_depth = |t: [f32; 2]| 0.3 + 0.4 * step(16.0, t[0]) + 0.15 * step(8.0, t[1]) + 0.001 * t[1];
        let texel_code = |t: [f32; 2]| if t[0] < 8.0 { 529.0 } else if t[0] < 16.0 { 533.0 } else { 0.0 };
        let pixel_code = |p: [f32; 2]| match p {
            [x, y] if x < 8.0 && y >= 28.0 => 1.0,
            [x, _] if x < 16.0 => 530.0,
            [x, _] if x < 32.0 => 533.0,
            _ => 0.0f32,
        };

        let same_face = |a: f32, b: f32| {
            if a < 0.5 || b < 0.5 {
                return true;
            }
            let (ia, ib) = (a as i32 - 1, b as i32 - 1);
            (ia / 32 - ib / 32).abs() <= 2 && (ia % 32 - ib % 32).abs() <= 2
        };
        let (hw, hh) = (W / 2, H / 2);
        let (mut worst, mut edges, mut corner, mut facing_none) = (0.0f32, 0usize, 0usize, 0usize);
        for y in 0..H {
            for x in 0..W {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                let depth = 0.3 + 0.4 * step(32.0, p[0]) + 0.15 * step(16.0, p[1]) + 0.001 * (p[1] * 0.5 - 0.5);
                let h = [p[0] * 0.5 - 0.5, p[1] * 0.5 - 0.5];
                let base = [h[0].floor(), h[1].floor()];
                let f = [h[0] - base[0], h[1] - base[1]];
                let (mut sum, mut weight) = ([0.0f32; 4], 0.0f32);
                let (mut nearest, mut nearest_gap) = ([0.0f32; 4], f32::MAX);
                let (mut other_face, mut facing) = (false, 0);
                for (ox, oy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                    let t = [(base[0] + ox).clamp(0.0, (hw - 1) as f32), (base[1] + oy).clamp(0.0, (hh - 1) as f32)];
                    let c = colour(t);
                    let gap = (texel_depth(t) - depth).abs();
                    let faces = same_face(texel_code(t), pixel_code(p));
                    other_face |= !faces && gap <= TOLERANCE;
                    facing += usize::from(faces);
                    let bw = (if ox == 1.0 { f[0] } else { 1.0 - f[0] }) * (if oy == 1.0 { f[1] } else { 1.0 - f[1] });
                    let w = if gap <= TOLERANCE && faces { bw } else { 0.0 };
                    for k in 0..4 {
                        sum[k] += c[k] * w;
                    }
                    weight += w;
                    if faces && gap < nearest_gap {
                        nearest_gap = gap;
                        nearest = c;
                    }
                }
                edges += usize::from(weight < 0.999);
                corner += usize::from(other_face && facing > 0);
                facing_none += usize::from(facing == 0);
                let pre = if weight > 1e-4 { sum.map(|v| v / weight) } else { nearest };
                let want = [pre[0] / pre[3].max(1e-4), pre[1] / pre[3].max(1e-4), pre[2] / pre[3].max(1e-4), pre[3]];
                let i = ((y * W + x) * 4) as usize;

                let luma = |c: &[f32]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                let shown = 1.0 + luma(&got[i..i + 3]);
                let filtered = [got[i] / shown, got[i + 1] / shown, got[i + 2] / shown, got[i + 3]];
                for k in 0..4 {
                    let d = (filtered[k] - want[k]).abs();
                    worst = worst.max(d);
                    assert!(d < 4e-3, "pixel ({x}, {y}) channel {k}: {} against {}", filtered[k], want[k]);
                }
            }
        }

        assert!(edges > 60 && edges < (W * H / 4) as usize, "{edges} edge pixels");
        assert!(corner >= 2 * H as usize, "{corner} pixels by the corner");
        assert!(facing_none >= 16, "{facing_none} pixels facing no texel");
        eprintln!(
            "upsample: worst difference {worst:.5} over {} pixels, {edges} on an edge, {corner} by the corner, {facing_none} facing none",
            W * H
        );
    }

    #[test]
    fn every_deferred_register_cut_applies_and_validates() {
        use wgpu::naga;
        let base = brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::PassDeferred);
        for (label, edits) in DEFERRED_REGISTER_CUTS {
            let mut src = base.clone();
            for (from, to) in edits.iter() {
                assert!(src.contains(from), "{label}: `{from}` is not in the deferring probe pass");

                assert_eq!(src.matches(from).count(), 1, "{label}: `{from}` is in the deferring probe pass more than once");
                src = src.replacen(from, to, 1);
            }
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
    }

    #[test]
    fn the_probe_pass_reads_its_slots_from_push_constants() {
        use wgpu::naga;
        let base = brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::PassDeferred);
        let src = push_scan(&base).expect("the deferring probe pass holds the scan's reads");
        assert!(!src.contains("camera.probe_boxes[i * 3 + 2].w;\n        let v = camera.probe_boxes[i * 3].xyz - p;"));
        assert_eq!(crate::renderer::uniforms::PROBE_PUSH_SIZE, 16 * 16);
        assert_eq!(crate::renderer::uniforms::MAX_PROBES, 16, "the declaration is written for sixteen slots");
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
    }

    #[test]
    fn every_scene_register_cut_applies_and_validates() {
        use wgpu::naga;
        let base = brush_shader_probe(false, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Read);
        for (label, edits) in SCENE_REGISTER_CUTS {
            let mut src = base.clone();
            for (from, to) in edits.iter() {
                assert!(src.contains(from), "{label}: `{from}` is not in the scene shader");
                src = src.replacen(from, to, 1);
            }
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
    }

    #[test]
    fn every_reader_edit_applies_and_validates() {
        use crate::renderer::terrain_pipeline::{terrain_shader_for, TerrainRole};
        use wgpu::naga;
        let readers = [
            ("brush", brush_shader_probe(false, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Read)),
            ("ground", terrain_shader_for(TerrainRole::Read)),
        ];
        for (label, _) in READER_EDITS {
            for (reader, base) in &readers {
                let src = with_reader_edit(base.clone(), label).unwrap_or_else(|| panic!("{label} does not apply to the {reader} reader"));
                assert_eq!(src == *base, *label == "reader_edit_none", "{label}: {reader}");
                for class in [FaceSun::Unbaked, FaceSun::Never, FaceSun::Baked] {
                    for twin in [sun_reader_shader(src.clone(), class), crate::renderer::lights::without_spot_shadows(sun_reader_shader(src.clone(), class))] {
                        let module = naga::front::wgsl::parse_str(&twin)
                            .unwrap_or_else(|e| panic!("{label} {reader}: {}", e.emit_to_string(&twin)));
                        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                            .validate(&module)
                            .unwrap_or_else(|e| panic!("{label} {reader} {class:?}: {e:?}"));
                    }
                }
            }
        }
        assert!(with_reader_edit(readers[0].1.clone(), "no_such_edit").is_none());
    }

    #[test]
    fn the_scene_shader_spreads_the_terminator_and_takes_no_derivative_specular_aa() {
        for probe in [BrushProbe::Read, BrushProbe::Trace] {
            let src = brush_shader_probe(false, false, false, SsrPath::Inline, probe);
            assert!(src.contains("    terminator_width = terminator_width_of(length(tn));"), "{probe:?}");
            assert!(src.contains("    let rough_aa = rough;"), "{probe:?}");
            assert!(!src.contains("specular_aa_roughness(rough, dpdx(n), dpdy(n))"), "{probe:?}");
        }
    }

    #[test]
    fn every_probe_pass_register_cut_applies_and_validates() {
        use wgpu::naga;
        let base = brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Pass);
        for (label, edits) in PROBE_PASS_REGISTER_CUTS {
            let mut src = base.clone();
            for (from, to) in edits.iter() {
                assert!(src.contains(from), "{label}: `{from}` is not in the probe pass shader");
                src = src.replacen(from, to, 1);
            }
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
    }

    #[test]
    fn the_probe_pass_pipelines_build_on_a_real_device() {
        let reader_src = brush_shader_probe(false, false, false, SsrPath::Inline, BrushProbe::Read);
        assert!(reader_src.contains("const PROBE_ENV_FROM_PASS: bool = true;"), "the reader still traces per pixel");
        assert!(brush_shader_variant(false, false, false).contains("const PROBE_ENV_FROM_PASS: bool = false;"));
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _pass = BrushPipeline::new_probe_pass(&device, &uniforms.layout, crate::renderer::multiview::ViewMode::Mono);
        let layout = probe_pass::bind_group_layout(&device);
        let _reader = BrushPipeline::new_multisampled_probe_reader(
            &device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &layout, crate::renderer::multiview::ViewMode::Mono,
        );
        let target = probe_pass::Target::new(&device, &layout, 1445, 1546, 1);

        let _stereo = probe_pass::Target::new(&device, &layout, 1445, 1546, 2);
        assert_eq!((target.width, target.height), (723, 773));

        let fixups = crate::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, target.width * target.height);
        let _deferred = BrushPipeline::new_probe_pass_deferred(&device, &uniforms.layout, &fixups);
        let _poolless = BrushPipeline::new_probe_pass_deferred_poolless(&device, &uniforms.layout, &fixups);
        let _fixup_target = fixups.target_bind_group(&device, &target);
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the probe pass pipelines failed to build: {err:?}");
        let deferred_src = brush_shader_probe(false, false, false, SsrPath::Inline, BrushProbe::PassDeferred);
        assert!(deferred_src.contains("const PROBE_SECONDARY_DEFERRED: bool = true;"));
        assert!(deferred_src.contains("probe_fragment = in.clip;") && deferred_src.contains("atomicAdd(&probe_fixups.count"));

        let poolless = crate::renderer::lights::without_pool_maps(deferred_src.clone());
        assert!(deferred_src.contains("= probe_surface_relit(") && !poolless.contains("= probe_surface_relit("));
        assert_eq!(
            poolless.len(),
            deferred_src.len()
                - crate::renderer::lights::SURFACE_RELIT_READ.len()
                - crate::renderer::lights::SURFACE_RELIT_CALL.len()
        );
        assert!(brush_shader_probe(false, false, false, SsrPath::Inline, BrushProbe::Pass)
            .contains("const PROBE_SECONDARY_DEFERRED: bool = false;"));
    }

    #[test]
    fn the_debug_view_pipelines_build_on_a_real_device() {
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);
        let err_scope_2 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _sources = BrushPipeline::new_multisampled_sources(
            &device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 1,
        );
        let err = pollster::block_on(err_scope_2.pop());
        assert!(err.is_none(), "the sources scene pipeline failed to build: {err:?}");
        for view in [DebugView::Sources, DebugView::Ssr] {
            let ssr = crate::renderer::ssr::SsrPipelines::new_with_depth_samples(
                &device, TextureFormat::Rgba8UnormSrgb, 1,
            );
            let err_scope_3 = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let _p = BrushPipeline::new_ssr_debug(
                &device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout,
                ssr.scene_texture_layout(), view,
            );
            let err = pollster::block_on(err_scope_3.pop());
            assert!(err.is_none(), "the {view:?} reflective pipeline failed to build: {err:?}");
        }
    }

    #[test]
    fn screen_space_reflection_follows_the_neighbourhood_not_the_texel() {
        let src = brush_shader_with(true);
        assert!(
            src.contains("let ssr_rough = textureSampleLevel(mat_rough, mat_rough_samp, in.uv, i32(in.material), 4.0).r;"),
            "SSR reads roughness per texel again; stone will sparkle",
        );
        assert!(src.contains("let reflectivity = clamp((1.0 - ssr_rough) * ssr_fresnel"));
        assert!(
            src.contains("ssr_reflect(in.world_pos, ssr_n, ssr_fallback, reflectivity, ssr_rough)"),
            "the march is steered by the texel's normal again",
        );
    }

    #[test]
    fn the_reflective_pass_does_not_light_the_surface_a_second_time() {
        const LIGHTS: &str = "let lit = shade_material_lamps(";
        const READ_BACK: &str =
            "let ssr_fallback = textureLoad(ssr_scene_color, vec2<i32>(in.clip.xy), 0).rgb;";
        for _ms in [false] {
            let shipped = brush_shader_variant(true, false, false);
            assert!(!shipped.contains(LIGHTS), "the reflective pass shades the surface again");
            assert!(shipped.contains(READ_BACK), "the reflective pass no longer reads the scene's colour");
            let ssr_view = brush_shader_variant(true, false, true);
            assert!(!ssr_view.contains(LIGHTS), "the SSR view shades the surface again");
        }

        assert!(brush_shader_variant(false, false, false).contains(LIGHTS), "the scene pass stopped lighting");
        assert!(
            brush_shader_variant(true, true, false).contains(LIGHTS),
            "the reflective sources view has nothing to paint",
        );
    }

    #[test]
    fn the_reflective_brush_shader_binds_exactly_four_groups() {
        let src = brush_shader_with(true);
        for g in 0..4 {
            assert!(
                src.contains(&format!("@group({g})")),
                "group {g} is unused -- the layout and the shader disagree",
            );
        }
        assert!(
            !src.contains("@group(4)"),
            "a fifth bind group would exceed the limit this hardware guarantees",
        );
    }

    #[test]
    fn the_ordinary_brush_shader_has_no_reflection_in_it() {
        let src = brush_shader();
        assert!(!src.contains("ssr_reflect"), "the non-reflective brush shader marches rays");
        assert!(!src.contains("@group(3)"), "the non-reflective brush shader binds a scene texture");
    }
}

#[cfg(test)]
mod reflectivity_tests {
    use super::*;
    use crate::renderer::terrain_pipeline::TerrainImage;

    fn rough_map(byte: u8) -> TerrainImage {
        TerrainImage { width: 4, height: 4, rgba: (0..16).flat_map(|_| [byte, byte, byte, 255]).collect() }
    }

    #[test]
    fn polished_marble_is_worth_reflecting() {

        assert!(roughness_is_reflective(Some(&rough_map(12))));
    }

    #[test]
    fn rock_and_gravel_are_not() {

        assert!(!roughness_is_reflective(Some(&rough_map(230))));
        assert!(!roughness_is_reflective(Some(&rough_map(255))));
    }

    #[test]
    fn a_material_with_no_roughness_map_is_treated_as_fully_rough() {

        assert!(!roughness_is_reflective(None));
    }

    #[test]
    fn an_empty_map_does_not_divide_by_zero() {
        let empty = TerrainImage { width: 0, height: 0, rgba: Vec::new() };
        assert!(!roughness_is_reflective(Some(&empty)));
    }

    #[test]
    fn the_mean_decides_not_the_shiniest_speck() {

        let mut mostly_rough = rough_map(240);
        mostly_rough.rgba[0] = 0;
        mostly_rough.rgba[4] = 0;
        assert!(!roughness_is_reflective(Some(&mostly_rough)));
    }
}

#[cfg(test)]
mod bounce_softening_tests {
    use super::*;

    fn shaped(directionality: f32, n_dot_dir: f32) -> f32 {
        let d = directionality.clamp(0.0, 0.5);
        ((1.0 - d) + d * (n_dot_dir + 1.0)).max(0.0)
    }

    #[test]
    fn a_confident_direction_cannot_swing_the_bounce_a_hundredfold() {

        let lo = shaped(1.0, -1.0);
        let hi = shaped(1.0, 1.0);
        assert!(hi / lo.max(1e-6) <= 3.1, "shaping still swings {:.1}x", hi / lo);
    }

    #[test]
    fn it_is_still_directional() {

        assert!(shaped(0.79, 1.0) > shaped(0.79, -1.0) * 2.0);
    }

    #[test]
    fn no_directionality_is_still_exactly_flat() {

        for cos in [-1.0f32, -0.4, 0.0, 0.6, 1.0] {
            assert_eq!(shaped(0.0, cos), 1.0);
        }
    }

    #[test]
    fn the_reflection_is_not_a_mirror() {

        let src = brush_shader_with(true);
        assert!(
            src.contains("MAX_SSR_REFLECTIVITY"),
            "the reflection strength is uncapped",
        );
    }
}

#[cfg(test)]
mod mip_chain_tests {

    use super::*;

    fn solid(w: u32, h: u32, px: [u8; 4]) -> TerrainImage {
        TerrainImage { width: w, height: h, rgba: px.iter().copied().cycle().take((w * h * 4) as usize).collect() }
    }

    #[test]
    fn a_chain_runs_all_the_way_down_to_one_texel() {
        let c = mip_chain(&solid(1024, 1024, [10, 20, 30, 255]), true);
        assert_eq!(c.len(), 11, "1024 needs 11 levels, got {}", c.len());
        assert_eq!((c[0].width, c[0].height), (1024, 1024));
        assert_eq!((c.last().unwrap().width, c.last().unwrap().height), (1, 1));
        for lv in &c {
            assert_eq!(lv.rgba.len(), (lv.width * lv.height * 4) as usize, "level is not its own size");
        }
    }

    #[test]
    fn a_flat_colour_survives_every_level() {
        for srgb in [true, false] {
            let c = mip_chain(&solid(64, 64, [180, 90, 45, 255]), srgb);
            for (i, lv) in c.iter().enumerate() {
                let px = &lv.rgba[0..4];
                assert!(
                    px.iter().zip([180u8, 90, 45, 255]).all(|(a, b)| (*a as i32 - b as i32).abs() <= 1),
                    "srgb={srgb} level {i} drifted to {px:?}",
                );
            }
        }
    }

    #[test]
    fn a_checkerboard_keeps_its_brightness() {
        let (w, h) = (64u32, 64u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 255 } else { 0 };
                let i = ((y * w + x) * 4) as usize;
                rgba[i..i + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        let img = TerrainImage { width: w, height: h, rgba };
        let level1 = &mip_chain(&img, true)[1];
        let got = level1.rgba[0];
        assert!(
            (got as i32 - 188).abs() <= 3,
            "half-lit sRGB averaged to {got}; 128 means it was averaged as raw bytes \
             and every mip is too dark",
        );
    }

    #[test]
    fn alpha_is_never_gamma_encoded() {
        let (w, h) = (2u32, 2u32);
        let rgba = vec![
            0, 0, 0, 255, 0, 0, 0, 0,
            0, 0, 0, 255, 0, 0, 0, 0,
        ];
        let img = TerrainImage { width: w, height: h, rgba };
        let a = mip_chain(&img, true)[1].rgba[3];
        assert!((a as i32 - 128).abs() <= 1, "alpha averaged to {a}, not 128");
    }
}

#[cfg(test)]
mod source_debug_tests {
    use super::*;

    #[test]
    fn the_source_diagnostic_is_off() {
        assert!(
            !BRUSH_SOURCE_DEBUG,
            "BRUSH_SOURCE_DEBUG is on: every brush renders as red/green/blue by \
             which term lit it, not as the scene. Set it back to false.",
        );
    }

    #[test]
    fn the_switch_reaches_the_shader() {
        let src = brush_shader_with(true);
        let painted = src.contains("let dbg_sum = dbg_direct + dbg_baked + dbg_probe;");
        assert_eq!(
            painted, BRUSH_SOURCE_DEBUG,
            "BRUSH_SOURCE_DEBUG is {BRUSH_SOURCE_DEBUG} but the shader {} the \
             diagnostic",
            if painted { "has" } else { "lacks" },
        );
    }

    #[test]
    fn every_source_is_recorded_at_its_own_term() {
        let lights = super::super::lights::wgsl_lights_block(0, 1);
        assert!(

            lights.contains(
                "dbg_probe = probe.rgb * clamp(probe.a, 0.0, 1.0) * spec_occ * probe_scale \
                 * (1.0 - lobe_is_hemispherical) * fresnel;",
            ),
            "the probe term is not recorded as the picture weights it",
        );

        assert!(
            lights.contains("let kd = albedo * (1.0 - fresnel);") && lights.contains("dbg_baked = bounce * kd;"),
            "the baked term is not recorded as the picture weights it",
        );
        assert!(
            lights.contains("let lit = unshadowed * hf(shadow);")
                && lights.contains("dbg_direct = dbg_direct + vec3<f32>(lit);"),
            "the direct term is not recorded, or is recorded before its shadow test",
        );
    }
}

#[cfg(test)]
mod crack_seal_tests {
    use super::*;
    use crate::renderer::lights::LightsUniform;
    use crate::renderer::terrain_pipeline::tests::headless_gpu;
    use crate::renderer::uniforms::test_support::{scene_uniforms, TEST_EYE};
    use crate::renderer::uniforms::ShadowUpload;
    use wgpu::util::DeviceExt;

    fn render_seal(facing_away: bool, z: f32, depth_clear: f32) -> Option<[u8; 4]> {
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;
        let lights = LightsUniform::new(&device);
        lights.upload(&queue, &[]);
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);

        uniforms.upload(&queue, glam::Mat4::IDENTITY, TEST_EYE, &ShadowUpload::disabled());
        let pipeline = BrushSealPipeline::new(&device, format, &uniforms.layout, 1);

        let v = |x: f32, y: f32| BrushVertex {
            position: [x, y, z],
            normal: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            material: 0,
            tint: [1.0; 4],
            uv2: [0.0, 0.0],
            face_centre: [1.0 / 3.0, 1.0 / 3.0, z],
            uv2_rect: [0.0, 0.0, 1.0, 1.0],

            face_half_extent: [f32::INFINITY; 2],
        };

        let mut verts = vec![v(-1.0, -1.0), v(3.0, -1.0), v(-1.0, 3.0)];
        if facing_away {
            verts.swap(1, 2);
        }
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seal_test_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seal_test_ib"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        const SIZE: u32 = 8;
        let tex = |label, format, usage| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let target = tex("seal_test_target", format, TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC);
        let depth = tex("seal_test_depth", TextureFormat::Depth32Float, TextureUsages::RENDER_ATTACHMENT);
        let target_view = target.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("seal_test_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("seal_test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::WHITE), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(depth_clear), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..3, 0, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(SIZE) },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        let c = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[c], data[c + 1], data[c + 2], data[c + 3]])
    }

    fn is_seal(px: [u8; 4]) -> bool {
        let want = |c: f32| (c * 255.0).round() as i32;
        let [r, g, b] = CRACK_SEAL_COLOUR;
        (px[0] as i32 - want(r)).abs() <= 2
            && (px[1] as i32 - want(g)).abs() <= 2
            && (px[2] as i32 - want(b)).abs() <= 2
    }

    #[test]
    fn a_face_pointing_away_is_sealed() {
        let Some(px) = render_seal(true, 0.5, 1.0) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(is_seal(px), "a back face drew {px:?}, not the seal colour");
    }

    #[test]
    fn a_face_pointing_at_the_viewer_is_not_drawn() {
        let Some(px) = render_seal(false, 0.5, 1.0) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(px, [255, 255, 255, 255], "the seal painted a front face: {px:?}");
    }

    #[test]
    fn a_back_face_behind_the_wall_stays_hidden() {
        let Some(px) = render_seal(true, 0.6, 0.3) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(px, [255, 255, 255, 255], "a back face behind the wall showed through: {px:?}");
    }

    /// A BACK FACE A HAIR IN FRONT OF THE WALL STILL LOSES: where a face meets
    /// its turned-away neighbour, the two depths differ only by rounding, and
    /// the unbiased seal won in dots down every such edge (`SEAL_DEPTH_BIAS`,
    /// 2026-10-20). Fails with the bias at zero.
    #[test]
    fn a_back_face_within_rounding_of_the_wall_stays_hidden() {
        let Some(px) = render_seal(true, 0.5, 0.5 + 4e-7) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(px, [255, 255, 255, 255], "the seal won a tie with the wall: {px:?}");
    }
}

#[cfg(test)]
mod centroid_tests {
    #[test]
    fn the_brush_does_not_pay_for_centroid_interpolation() {

        if super::BRUSH_EDGE_DEBUG {
            return;
        }
        let src = super::brush_shader_with(false);

        let allowed: &[&str] = if super::BRUSH_CENTROID_VARYINGS {
            &["world_pos", "uv2"]
        } else {
            &[]
        };
        let carriers: Vec<&str> = src
            .lines()
            .filter(|l| l.contains("@interpolate(perspective, centroid)"))
            .filter_map(|l| l.rsplit_once(") ").map(|(_, rest)| rest))
            .filter_map(|rest| rest.split(':').next())
            .map(str::trim)
            .collect();
        for name in &carriers {
            assert!(
                allowed.contains(name),
                "centroid interpolation has spread to `{name}`. It is allowed on \
                 {allowed:?} and nowhere else: on every varying it measured ~16 ms \
                 a frame on the headset, which does not fit in 13.9 ms. Add one \
                 only with a headset frame time next to it.",
            );
        }
        assert_eq!(
            carriers.len(),
            allowed.len(),
            "expected centroid on exactly {allowed:?}, found {carriers:?}",
        );
    }

    #[test]
    fn the_edge_diagnostic_is_off() {
        assert!(
            !super::BRUSH_EDGE_DEBUG,
            "BRUSH_EDGE_DEBUG is on: this build paints every brush by whether \
             MSAA shaded it from outside its polygon, and is a measurement, \
             not a build to ship",
        );
    }
}

#[cfg(test)]
mod trace_shader_tests {
    use super::*;

    fn fragment_of(trace: bool) -> String {
        let path = if trace { SsrPath::Trace } else { SsrPath::Inline };
        let src = brush_shader_modes(true, false, false, path);
        let i = src.find("fn fs_main").expect("no fragment entry point");
        src[i..].to_string()
    }

    #[test]
    fn the_trace_and_the_shipped_pass_share_the_ray_prep() {
        let (shipped, trace) = (fragment_of(false), fragment_of(true));
        for line in [
            "let ssr_rough = textureSampleLevel(mat_rough, mat_rough_samp, in.uv, i32(in.material), 4.0).r;",
            "let ssr_n = normalize(mix(n_geom, n, 1.0 - smoothstep(0.1, 0.4, ssr_rough)));",
            "let ssr_fresnel = 0.04 + 0.96 * pow(1.0 - ssr_cos, 5.0);",
            "let reflectivity = clamp((1.0 - ssr_rough) * ssr_fresnel, 0.0, MAX_SSR_REFLECTIVITY);",
        ] {
            assert!(shipped.contains(line), "the shipped reflective pass lost: {line}");
            assert!(
                trace.contains(line),
                "the trace prepares its ray differently from the pass that \
                 composites it, so the two will disagree about where the ray \
                 went. Missing: {line}",
            );
        }
    }

    #[test]
    fn reflectivity_is_applied_exactly_once() {
        let trace = fragment_of(true);
        assert!(
            trace.contains("return vec4<f32>(0.0);") || trace.contains("ssr_reflect("),
            "the trace no longer calls the march",
        );
        let block = crate::renderer::ssr::wgsl_ssr_block_shared_camera_radiance(3);
        assert!(
            block.contains("return vec4<f32>(hit_color, clamp(fade, 0.0, 1.0));"),
            "the trace folds something other than `fade` alone into confidence; \
             if that something is `weight`, the composite squares it",
        );
        let composite = {
            let src = brush_shader_modes(true, false, false, SsrPath::Composite);
            let i = src.find("fn fs_main").expect("no fragment entry point");
            src[i..].to_string()
        };
        assert!(
            composite.contains("clamp(reflectivity, 0.0, 1.0) * refl.a"),
            "the composite does not weight the reflection by this pixel's own \
             reflectivity:\n{composite}",
        );
        assert!(
            !composite.contains("ssr_scene_depth"),
            "the composite still declares the depth pyramid; it does not march \
             and binding 1 is the filtered reflection now",
        );
    }

    #[test]
    fn the_trace_writes_confidence_and_never_reads_the_fallback() {
        let trace = fragment_of(true);
        assert!(
            trace.contains(
                "return ssr_reflect(in.world_pos, ssr_n, vec3<f32>(0.0), reflectivity, ssr_rough);"
            ),
            "the trace no longer returns the march's result directly:\n{trace}",
        );

        assert!(
            !trace.contains("ssr_fallback"),
            "the trace reads the environment fallback; it must write only what \
             the march found, and leave the fallback to the composite",
        );

        let shipped = fragment_of(false);
        assert!(
            shipped.contains("let ssr_fallback = textureLoad(ssr_scene_color, vec2<i32>(in.clip.xy), 0).rgb;")
                && shipped.contains("albedo.a * in.tint.a"),
            "the shipped reflective pass changed while the trace was added",
        );
    }
}

#[cfg(test)]
pub fn brush_shader_src() -> String {
    brush_shader()
}
#[cfg(test)]
pub fn brush_seal_shader_src() -> String {
    brush_seal_shader()
}

#[cfg(test)]
mod normal_variance_roughness_tests {
    use super::{roughness_chain_with_normal_variance, NORMAL_VARIANCE_CLAMP};
    use crate::renderer::terrain_pipeline::TerrainImage;

    fn solid(w: u32, h: u32, px: [u8; 4]) -> TerrainImage {
        TerrainImage { width: w, height: h, rgba: px.iter().copied().cycle().take((w * h * 4) as usize).collect() }
    }

    fn opposing(w: u32, h: u32) -> TerrainImage {
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {

                let lean = if (x + y) % 2 == 0 { 255u8 } else { 0u8 };
                rgba.extend_from_slice(&[lean, 128, 128, 255]);
            }
        }
        TerrainImage { width: w, height: h, rgba }
    }

    #[test]
    fn a_flat_normal_map_changes_the_roughness_not_at_all() {

        let rough = solid(8, 8, [100, 100, 100, 255]);
        let flat = solid(8, 8, [128, 128, 255, 255]);
        let with = roughness_chain_with_normal_variance(&rough, Some(&flat));
        let without = roughness_chain_with_normal_variance(&rough, None);
        assert_eq!(with.len(), without.len());
        for (lvl, (a, b)) in with.iter().zip(without.iter()).enumerate() {

            for (i, (x, y)) in a.rgba.iter().zip(b.rgba.iter()).enumerate() {
                assert!(
                    (*x as i32 - *y as i32).abs() <= 2,
                    "level {lvl} byte {i}: flat normals moved roughness {} -> {}",
                    y, x,
                );
            }
        }
    }

    #[test]
    fn level_zero_is_never_touched() {

        let rough = solid(8, 8, [60, 60, 60, 255]);
        let with = roughness_chain_with_normal_variance(&rough, Some(&opposing(8, 8)));
        assert_eq!(with[0].rgba, rough.rgba, "level 0 was modified");
    }

    #[test]
    fn opposing_normals_roughen_the_coarser_mips() {
        let rough = solid(8, 8, [60, 60, 60, 255]);
        let with = roughness_chain_with_normal_variance(&rough, Some(&opposing(8, 8)));
        let without = roughness_chain_with_normal_variance(&rough, None);
        assert!(with.len() > 2);
        assert!(
            with[1].rgba[0] > without[1].rgba[0],
            "mip 1 did not get rougher ({} vs {}) even though its normals cancel",
            with[1].rgba[0],
            without[1].rgba[0],
        );
    }

    #[test]
    fn roughness_combines_in_variance_space_not_by_addition() {

        let base_byte = 60u8;
        let rough = solid(4, 4, [base_byte, base_byte, base_byte, 255]);
        let with = roughness_chain_with_normal_variance(&rough, Some(&opposing(4, 4)));
        let r = base_byte as f32 / 255.0;

        let expected = ((r * r + NORMAL_VARIANCE_CLAMP).min(1.0)).sqrt();
        let expected_byte = (expected * 255.0).round() as u8;
        let got = with[1].rgba[0];
        assert!(
            (got as i32 - expected_byte as i32).abs() <= 2,
            "variance-space combine wrong: got {got}, expected ~{expected_byte}",
        );

        let naive = ((r + NORMAL_VARIANCE_CLAMP).min(1.0) * 255.0).round() as u8;
        assert!(
            (got as i32 - naive as i32).abs() > 2,
            "the result matches naive addition ({naive}); the squares are not being used",
        );
    }

    #[test]
    fn a_normal_map_of_a_different_size_is_refused_rather_than_guessed() {

        let rough = solid(8, 8, [90, 90, 90, 255]);
        let wrong_size = solid(4, 4, [255, 128, 128, 255]);
        let with = roughness_chain_with_normal_variance(&rough, Some(&wrong_size));
        let without = roughness_chain_with_normal_variance(&rough, None);
        for (a, b) in with.iter().zip(without.iter()) {
            assert_eq!(a.rgba, b.rgba, "a mismatched normal map still altered roughness");
        }
    }

    #[test]
    fn the_added_roughness_is_capped() {

        assert!(NORMAL_VARIANCE_CLAMP < 0.25, "the cap is high enough to flatten highlights");
        let rough = solid(4, 4, [10, 10, 10, 255]);
        let with = roughness_chain_with_normal_variance(&rough, Some(&opposing(4, 4)));
        let r = 10.0f32 / 255.0;
        let ceiling = ((r * r + NORMAL_VARIANCE_CLAMP).sqrt() * 255.0).round() as u8;
        assert!(with[1].rgba[0] <= ceiling + 2, "added roughness exceeded the cap");
    }
}

#[cfg(test)]
mod roughness_sampler_tests {

    fn material_bindings(src: &str) -> Vec<u32> {
        let mut out: Vec<u32> = src
            .match_indices("@group(1) @binding(")
            .filter_map(|(i, _)| {
                let rest = &src[i + "@group(1) @binding(".len()..];
                rest.split(')').next()?.trim().parse().ok()
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    #[test]
    fn roughness_is_read_through_the_trilinear_sampler() {
        let src = super::brush_shader_src();
        assert!(
            src.contains("@group(1) @binding(5) var mat_rough_samp: sampler;"),
            "the roughness sampler is not declared",
        );
        for bad in [
            "textureSample(mat_rough, mat_samp",
            "textureSampleLevel(mat_rough, mat_samp",
        ] {
            assert!(
                !src.contains(bad),
                "roughness is still being read through the anisotropic sampler: {bad}",
            );
        }

        for keep in ["textureSample(mat_color, mat_samp", "textureSample(mat_normal, mat_samp"] {
            assert!(
                src.contains(keep),
                "{keep} lost the anisotropic sampler; walls will go soft at grazing angles",
            );
        }
    }

    #[test]
    fn the_material_bindings_have_no_holes() {
        let b = material_bindings(&super::brush_shader_src());
        assert!(!b.is_empty(), "the brush shader declares no group-1 bindings at all");
        let expected: Vec<u32> = (0..b.len() as u32).collect();
        assert_eq!(
            b, expected,
            "group(1) bindings are {b:?}; the layout in \
             `brush_material_bind_group_layout` must declare exactly these",
        );
    }
}

#[cfg(test)]
mod sun_faces_tests {
    use super::*;

    const W: u32 = 256;
    const H: u32 = 128;
    const LIT: [u8; 4] = [9, 255, 40, 255];
    const UNBAKED: [u8; 4] = [0, 0, 0, 0];

    fn mask(texels: &[(u32, u32)], texel: [u8; 4]) -> Vec<u8> {
        let mut m: Vec<u8> = (0..W * H).flat_map(|_| [0u8, 255, 40, 255]).collect();
        for &(x, y) in texels {
            let i = ((y * W + x) * 4) as usize;
            m[i..i + 4].copy_from_slice(&texel);
        }
        m
    }

    fn faces(m: &[u8]) -> SunFaces {
        SunFaces::from_mask(m, W, H).unwrap()
    }

    fn patch() -> [f32; 4] {
        [100.0 / W as f32, 50.0 / H as f32, 120.0 / W as f32, 60.0 / H as f32]
    }

    fn texel(m: &[u8], x: u32, y: u32) -> [u8; 4] {
        let i = ((y * W + x) * 4) as usize;
        m[i..i + 4].try_into().unwrap()
    }

    #[test]
    fn a_patch_hidden_out_to_its_border_is_never_reached() {
        assert_eq!(faces(&mask(&[], UNBAKED)).class_of(patch()), FaceSun::Never);
    }

    #[test]
    fn a_texel_the_sun_reaches_within_the_border_makes_it_baked() {

        assert_eq!(faces(&mask(&[(140, 55)], LIT)).class_of(patch()), FaceSun::Baked);

        let far = mask(&[(120 + SUN_MASK_READ_REACH + 2, 55)], LIT);
        assert_eq!(faces(&far).class_of(patch()), FaceSun::Never);
    }

    #[test]
    fn an_unbaked_texel_within_reach_keeps_the_full_reader() {

        assert_eq!(faces(&mask(&[(110, 55)], [0, 100, 0, 255])).class_of(patch()), FaceSun::Unbaked);
        assert_eq!(faces(&mask(&[(140, 55)], UNBAKED)).class_of(patch()), FaceSun::Unbaked);

        assert_eq!(faces(&mask(&[], UNBAKED)).class_of([f32::NAN; 4]), FaceSun::Unbaked);
    }

    #[test]
    fn the_border_covers_the_deepest_mip_the_sampler_reads() {

        let deepest = crate::renderer::mesh::SUN_MASK_MIP_LEVELS - 1;
        assert!(SUN_MASK_READ_REACH as f32 >= 1.5 * (1u32 << deepest) as f32 + 1.0);
    }

    #[test]
    fn the_partition_keeps_each_parts_order() {

        let mut m = mask(&[(10, 10)], [200, 255, 40, 255]);
        let i = ((100 * W + 200) * 4) as usize;
        m[i..i + 4].copy_from_slice(&UNBAKED);
        let f = faces(&m);
        let vertex = |rect: [f32; 4]| BrushVertex { uv2_rect: rect, ..bytemuck::Zeroable::zeroed() };
        let lit = [8.0 / W as f32, 8.0 / H as f32, 12.0 / W as f32, 12.0 / H as f32];
        let bare = [198.0 / W as f32, 98.0 / H as f32, 202.0 / W as f32, 102.0 / H as f32];
        let v = [vertex(patch()), vertex(lit), vertex(patch()), vertex(lit), vertex(bare)];
        let idx = [4, 4, 4, 1, 1, 1, 0, 0, 0, 3, 3, 3, 2, 2, 2];
        let (out, ends) = f.partition(&v, &idx);
        assert_eq!(ends, [6, 12]);
        assert_eq!(out, vec![0, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3, 4, 4, 4]);
    }

    #[test]
    fn dilation_fills_the_empty_atlas_out_to_the_reach_from_the_nearest_chart_texel() {

        let mut m: Vec<u8> = (0..W * H).flat_map(|_| UNBAKED).collect();
        for y in 40..60 {
            for x in 40..60 {
                let t = if x == 59 { LIT } else { [0, 255, 40, 255] };
                let i = ((y * W + x) * 4) as usize;
                m[i..i + 4].copy_from_slice(&t);
            }
        }
        let d = dilate_sun_mask(&m, W, H);

        assert_eq!(texel(&d, 59 + SUN_MASK_READ_REACH, 50), LIT);
        assert_eq!(texel(&d, 40 - SUN_MASK_READ_REACH, 50), [0, 255, 40, 255]);

        assert_eq!(texel(&d, 59 + 10, 40 - 10), LIT);
        assert_eq!(texel(&d, 40 - SUN_MASK_READ_REACH, 40 - SUN_MASK_READ_REACH), [0, 255, 40, 255]);

        assert_eq!(texel(&d, 59 + SUN_MASK_READ_REACH + 1, 50), UNBAKED);
        assert_eq!(texel(&d, 45, 40 - SUN_MASK_READ_REACH - 1), UNBAKED);

        for i in 0..(W * H) as usize {
            if m[i * 4 + 1] >= 128 {
                assert_eq!(d[i * 4..i * 4 + 4], m[i * 4..i * 4 + 4]);
            }
        }

        let inner = [41.0 / W as f32, 41.0 / H as f32, 50.0 / W as f32, 59.0 / H as f32];
        assert_eq!(faces(&m).class_of(inner), FaceSun::Unbaked);
        assert_eq!(faces(&d).class_of(inner), FaceSun::Baked);
    }

    #[test]
    fn each_sun_reader_differs_only_in_its_switch_and_validates() {
        use wgpu::naga;
        let base = brush_shader_probe(false, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Read);
        assert_eq!(sun_reader_shader(base.clone(), FaceSun::Unbaked), base);
        for (class, switch) in [(FaceSun::Never, "SKY_SUN_NEVER_REACHES"), (FaceSun::Baked, "SUN_MASK_EVERYWHERE")] {
            let source = sun_reader_shader(base.clone(), class);
            assert_eq!(
                source.replace(&format!("const {switch}: bool = true;"), &format!("const {switch}: bool = false;")),
                base
            );
            let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap();
        }
    }

    #[test]
    #[ignore]
    fn where_the_readers_instructions_come_from() {
        use crate::renderer::terrain_pipeline::{terrain_shader_for, TerrainRole};
        let read = brush_shader_probe(false, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Read);
        let ground = terrain_shader_for(TerrainRole::Read);
        let readers = [
            ("brush_pipeline_read", read.clone()),
            ("brush_pipeline_read_baked", sun_reader_shader(read, FaceSun::Baked)),
            ("terrain_pipeline_read", ground.clone()),
            ("terrain_pipeline_read_baked", sun_reader_shader(ground, FaceSun::Baked)),
        ];
        for (label, source) in readers {
            println!("{}", crate::renderer::shader_inlining::report(label, &source, "fs_main", 30));
        }
    }

    #[test]
    #[ignore]
    fn where_the_probe_passes_instructions_come_from() {
        use crate::renderer::terrain_pipeline::{terrain_shader_for, TerrainRole};
        let passes = [
            ("brush_probe_pass_deferred", brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::PassDeferred)),
            ("terrain_probe_pass", terrain_shader_for(TerrainRole::ProbePass)),
        ];
        for (label, source) in passes {
            println!("{}", crate::renderer::shader_inlining::report(label, &source, "fs_main", 30));
        }

        let (_, edits) = DEFERRED_REGISTER_CUTS.iter().find(|(label, _)| *label == "def_dedup_all").unwrap();
        let mut source = brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::PassDeferred);
        for (from, to) in edits.iter() {
            source = source.replacen(from, to, 1);
        }
        println!("{}", crate::renderer::shader_inlining::report("def_dedup_all", &source, "fs_main", 12));
    }
}

#[cfg(test)]
mod lightmap_clamp_tests {

    #[test]
    fn both_lightmap_reads_use_the_clamped_uv() {
        let src = super::brush_shader_src();
        assert!(
            src.contains("let lm_uv = clamp(in.uv2, in.uv2_rect.xy, in.uv2_rect.zw);"),
            "the lightmap uv is no longer clamped into the face's own patch",
        );
        for read in ["textureSample(lm_tex, lm_samp, lm_uv)", "textureSample(lm_dir_tex, lm_samp, lm_uv)"] {
            assert!(src.contains(read), "a lightmap read bypasses the clamp: {read}");
        }
        assert!(
            !src.contains("textureSample(lm_tex, lm_samp, in.uv2)")
                && !src.contains("textureSample(lm_dir_tex, lm_samp, in.uv2)"),
            "a lightmap read still samples the raw interpolated uv2",
        );
    }

    #[test]
    fn the_patch_rectangle_is_flat_interpolated() {
        let src = super::brush_shader_src();
        assert!(
            src.contains("@location(8) @interpolate(flat) uv2_rect: vec4<f32>;")
                || src.contains("@location(8) @interpolate(flat) uv2_rect: vec4<f32>,"),
            "uv2_rect lost its flat interpolation",
        );
    }
}

#[cfg(test)]
mod face_clamp_tests {

    #[test]
    fn the_shading_position_is_the_clamped_one() {
        let src = super::brush_shader_with(false);
        assert!(
            src.contains("let face_pos = in.face_centre"),
            "the per-face position clamp is gone from the brush shader",
        );
        assert!(
            src.contains("shade_material_env_part(\n        face_pos,"),
            "the clamp is computed but the RAW `in.world_pos` is still what \
             gets shaded, so it corrects nothing",
        );
    }

    #[test]
    fn the_room_is_chosen_from_the_face_centre() {
        let src = super::brush_shader_with(false);
        assert!(src.contains("probe_volume_pos = vec4<f32>(in.face_centre, 1.0);"));
        assert!(src.contains("bdir, albedo.rgb, face_pos,"));
    }

    #[test]
    fn the_extent_is_flat_interpolated() {
        let src = super::brush_shader_with(false);
        assert!(
            src.contains("@location(9) @interpolate(flat) face_half_extent: vec2<f32>,"),
            "face_half_extent is being smoothly interpolated, which extrapolates \
             it at exactly the edge pixels it is meant to correct",
        );
    }
}
