//! Textured level geometry: walls, floors and stairs built from brushes.
//!
//! WHY BRUSHES DO NOT RIDE THE SOLID PIPELINE
//!
//! They did, and it was the right first step -- correct geometry, correct
//! shading, one flat colour per object. What it cannot express is the thing a
//! brush is FOR. A face carries a material id and a tile scale in metres, and a
//! wall that is concrete where the author said concrete is most of what makes
//! level geometry read as a place rather than as blocking volumes.
//!
//! ONE DRAW CALL, NOT ONE PER MATERIAL
//!
//! The material index is per VERTEX and every colour map lives in one texture
//! array, so a level of twenty materials is a single draw. Sorting into a draw
//! per material would be the obvious alternative and is the wrong trade on a
//! Quest: draw calls are the scarce thing, and level geometry is exactly the
//! case with many materials and no per-object state to change between them.
//!
//! TANGENTS COME FROM THE BRUSH, NOT FROM THE TRIANGLES
//!
//! Normal mapping needs a tangent frame, and the usual way to get one is to
//! derive it from the triangles' uvs -- which is fiddly, degenerate on thin
//! triangles, and an approximation of something this data already knows
//! exactly. A brush face carries the u and v axes its uvs were generated from.
//! Taking the tangent straight from those is both cheaper and exact.

use bytemuck::{Pod, Zeroable};
use wgpu::*;

use super::lights::wgsl_lights_block;
use super::terrain_pipeline::{resample, solid_image, TerrainImage};

/// A brush vertex: geometry, its material, and the frame to light it in.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BrushVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// The face's u axis. `w` carries the bitangent's handedness.
    pub tangent: [f32; 4],
    /// Texture coordinate in TILES, not in 0..1 -- the face's scale is metres
    /// per tile, so a 4m wall at scale 2 arrives here spanning 0..2 and the
    /// sampler's repeat wrapping does the rest.
    pub uv: [f32; 2],
    /// Which layer of the material array to sample.
    pub material: u32,
    /// Multiplied over the sampled colour. Carries the object's authored colour,
    /// which is the whole appearance for a face whose material is missing or
    /// unassigned -- that case binds a white layer, so the tint is all there is.
    pub tint: [f32; 4],
    /// Lightmap coordinate, 0..1 over this brush object's own baked atlas.
    ///
    /// A SECOND uv set. `uv` above is in tiles and deliberately shared between
    /// faces so brickwork lines up across a corner; lighting needs one unshared
    /// patch per face, or two walls sample each other's shadows. The layout is
    /// decided by space_soup_engine::brush_lightmap, which the baker uses too.
    pub uv2: [f32; 2],
}

impl BrushVertex {
    pub const ATTRIBS: [VertexAttribute; 7] = vertex_attr_array![
        0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x2, 4 => Uint32, 5 => Float32x4,
        6 => Float32x2
    ];

    pub fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// The most materials one level may use at once.
///
/// A limit rather than a growing array because the array is one GPU allocation
/// sized at build time, and a level that quietly exceeded it would drop the
/// materials at the end -- silently, and only for whoever authored the
/// twenty-fifth. Exceeding it logs and clamps, so it is diagnosable.
pub const MAX_BRUSH_MATERIALS: usize = 24;

pub struct BrushPipeline {
    pub pipeline: RenderPipeline,
    pub material_layout: BindGroupLayout,
    /// Layout for group 2, a brush object's baked lighting.
    pub lightmap_layout: BindGroupLayout,
}

/// The neutral lightmap for a brush that has not been baked.
///
/// BLACK, and that is not a detail. A brush lightmap is ADDED to the shaded
/// result rather than multiplied into it, so zero is the value that changes
/// nothing -- the way white is for the mesh pipeline, which multiplies. Binding
/// the wrong neutral is a full stop of extra brightness on every surface of
/// every unbaked brush, which looks like a broken shader rather than a wrong
/// constant.
pub fn default_brush_lightmap(
    device: &Device,
    queue: &Queue,
    layout: &BindGroupLayout,
) -> crate::renderer::mesh::LoadedTexture {
    crate::renderer::mesh::create_texture_from_rgba(device, queue, layout, &[0u8, 0, 0, 255], 1, 1)
}

/// Colour and normal maps for every material a scene's brushes use.
pub struct BrushMaterials {
    pub bind_group: BindGroup,
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
            // Roughness, then ambient occlusion. Separate arrays rather than
            // channels of one packed map: the material library stores them as
            // the separate greyscale files ambientCG ships, and packing them
            // here would mean a second representation to keep in step with the
            // editor, which samples the same two files.
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
        ],
    })
}

impl BrushMaterials {
    /// Build the material arrays.
    ///
    /// `colours` and `normals` are parallel and both are padded to
    /// MAX_BRUSH_MATERIALS: every layer of a texture array shares one
    /// allocation and must match in size, so a missing or differently-sized map
    /// cannot simply be skipped. A missing colour becomes white (the vertex
    /// tint then decides the look) and a missing normal becomes flat.
    pub fn new(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
    ) -> Self {
        // Sized from the first real map rather than from a constant, so a
        // project of 2K materials is not downsampled to someone's guess.
        let (w, h) = colours
            .iter()
            .flatten()
            .map(|i| (i.width, i.height))
            .next()
            .unwrap_or((1, 1));

        let colour_tex = Self::array_texture(
            device,
            queue,
            "brush_colour_array",
            TextureFormat::Rgba8UnormSrgb,
            colours,
            [255, 255, 255],
            w,
            h,
        );
        // NOT sRGB. A normal map is a direction, not a colour, and decoding it
        // through the sRGB curve bends every normal toward the surface -- which
        // looks like weak lighting rather than like a format mistake.
        let normal_tex = Self::array_texture(
            device,
            queue,
            "brush_normal_array",
            TextureFormat::Rgba8Unorm,
            normals,
            [128, 128, 255],
            w,
            h,
        );

        // Fully rough and fully unoccluded where a material has no map, which
        // is exactly what the surface looked like before either existed. Not
        // sRGB: both are measurements, not colours, and decoding them through
        // the sRGB curve would skew every value toward the dark end.
        let rough_tex = Self::array_texture(
            device, queue, "brush_rough_array", TextureFormat::Rgba8Unorm,
            roughs, [255, 255, 255], w, h,
        );
        let ao_tex = Self::array_texture(
            device, queue, "brush_ao_array", TextureFormat::Rgba8Unorm,
            aos, [255, 255, 255], w, h,
        );

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("brush_sampler"),
            // Repeat, because the uv is in tiles: a wall covering six tiles of
            // its material arrives with uv spanning 0..6 and clamping would
            // smear the last pixel across five of them.
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

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
            ],
        });

        Self { bind_group }
    }

    /// A material array with nothing in it, for a scene that has no brushes.
    ///
    /// Bound anyway rather than left absent: an optional binding would mean two
    /// pipeline layouts and two shaders that have to agree, which is a bigger
    /// thing to keep right than one white texture.
    pub fn fallback(device: &Device, queue: &Queue, layout: &BindGroupLayout) -> Self {
        Self::new(device, queue, layout, &[], &[], &[], &[])
    }

    #[allow(clippy::too_many_arguments)]
    fn array_texture(
        device: &Device,
        queue: &Queue,
        label: &str,
        format: TextureFormat,
        images: &[Option<TerrainImage>],
        fallback: [u8; 3],
        w: u32,
        h: u32,
    ) -> Texture {
        let filled: Vec<TerrainImage> = (0..MAX_BRUSH_MATERIALS)
            .map(|i| match images.get(i).and_then(|x| x.as_ref()) {
                Some(img) if img.width == w && img.height == h => TerrainImage {
                    width: img.width,
                    height: img.height,
                    rgba: img.rgba.clone(),
                },
                Some(img) => resample(img, w, h),
                None => solid_image(fallback, w, h),
            })
            .collect();

        let tex = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: MAX_BRUSH_MATERIALS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (layer, img) in filled.iter().enumerate() {
            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: layer as u32 },
                    aspect: TextureAspect::All,
                },
                &img.rgba,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * w),
                    rows_per_image: Some(h),
                },
                Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
        tex
    }
}

impl BrushPipeline {
    pub fn new(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Ccw, 1)
    }

    /// See `pipeline::SolidPipeline::new_multisampled` -- a pipeline's sample
    /// count must match the pass it runs in, so a 4x eye pass needs its own.
    pub fn new_multisampled(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Ccw, samples)
    }

    /// The mirror pass draws a reflected world, which reverses every winding.
    pub fn new_mirror(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
    ) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Cw, 1)
    }

    fn new_with_front_face(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("brush_shader"),
            source: ShaderSource::Wgsl(brush_shader().into()),
        });
        let material_layout = brush_material_bind_group_layout(device);
        // Shared with the mesh and cuboid pipelines: a lightmap is a lightmap,
        // and three layouts that must stay identical is three chances to drift.
        let lightmap_layout = super::pipeline::lightmap_bind_group_layout(device);
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("brush_layout"),
            bind_group_layouts: &[uniform_layout, &material_layout, &lightmap_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("brush_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[BrushVertex::layout()],
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
                front_face,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview: None,
            cache: None,
        });
        Self { pipeline, material_layout, lightmap_layout }
    }
}

fn brush_shader() -> String {
    format!(
        r#"
// Group 0 -- the camera, the lights and both shadow maps -- is declared by
// `wgsl_lights_block` below, so there is one description of that layout rather
// than one per shader.

@group(2) @binding(0) var lm_tex: texture_2d<f32>;
@group(2) @binding(1) var lm_samp: sampler;

@group(1) @binding(0) var mat_color: texture_2d_array<f32>;
@group(1) @binding(1) var mat_normal: texture_2d_array<f32>;
@group(1) @binding(2) var mat_samp: sampler;
@group(1) @binding(3) var mat_rough: texture_2d_array<f32>;
@group(1) @binding(4) var mat_ao: texture_2d_array<f32>;

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
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) tangent: vec4<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) uv: vec2<f32>,
    // Flat: a material index is a choice, not a quantity, and interpolating it
    // across a triangle that spans two materials would sample layer 1.5.
    @location(4) @interpolate(flat) material: u32,
    @location(5) tint: vec4<f32>,
    @location(6) uv2: vec2<f32>,
}}

@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    out.clip      = camera.view_proj * vec4<f32>(v.pos, 1.0);
    out.normal    = v.norm;
    out.tangent   = v.tangent;
    out.world_pos = v.pos;
    out.uv        = v.uv;
    out.uv2       = v.uv2;
    out.material  = v.material;
    out.tint      = v.tint;
    return out;
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let albedo = textureSample(mat_color, mat_samp, in.uv, i32(in.material));

    // Tangent frame from the brush face's own axes, re-orthogonalised against
    // the interpolated normal so the two cannot drift apart.
    let n_geom = normalize(in.normal);
    let t = normalize(in.tangent.xyz - n_geom * dot(n_geom, in.tangent.xyz));
    let b = cross(n_geom, t) * in.tangent.w;
    let tn = textureSample(mat_normal, mat_samp, in.uv, i32(in.material)).xyz * 2.0 - 1.0;
    let n = normalize(t * tn.x + b * tn.y + n_geom * tn.z);

    // ADDED, not multiplied.
    //
    // The mesh pipeline multiplies its lightmap in, and multiplying cannot ADD
    // light: a wall the lamp never reaches directly sits at the ambient floor
    // and no amount of bounce can lift it, which is most of a real interior.
    // Bounced light is light and belongs in the sum.
    //
    // Adding is only correct because a light is either baked or realtime and
    // never both -- see LightMode. `shade` sees only the realtime ones, the
    // texture carries only the baked ones, and the ambient term belongs solely
    // to `shade`, which varies it by the direction the surface faces.
    //
    // An unbaked brush binds a BLACK texture, so this collapses to exactly the
    // old behaviour. Black is the neutral value for a sum the way white is for
    // a product, and binding the wrong one is a full stop of extra brightness
    // on every surface.
    // The material's own roughness and occlusion, so a polished surface and a
    // matte one do not light identically. A material with no map gets white in
    // both arrays, which is fully rough and fully unoccluded -- exactly how
    // every brush looked before these existed.
    let rough = textureSample(mat_rough, mat_samp, in.uv, i32(in.material)).r;
    let ao = textureSample(mat_ao, mat_samp, in.uv, i32(in.material)).r;

    let baked = textureSample(lm_tex, lm_samp, in.uv2).rgb;
    let lit = shade_material(in.world_pos, n, rough, ao) + baked;
    let c = albedo.rgb * in.tint.rgb * lit;
    return vec4<f32>(c, albedo.a * in.tint.a);
}}
"#,
        lights_block = wgsl_lights_block(0, 1)
    )
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

    /// A flat image of one colour, standing in for a material's colour map.
    fn flat(rgb: [u8; 3], size: u32) -> TerrainImage {
        TerrainImage {
            width: size,
            height: size,
            rgba: (0..size * size)
                .flat_map(|_| [rgb[0], rgb[1], rgb[2], 255])
                .collect(),
        }
    }

    /// A normal map that tilts every texel hard along +u.
    ///
    /// Encoded the way a normal map is: 0..255 mapping to -1..1, so 255 in red
    /// is a full tilt toward the face's u axis and 128 is no tilt at all.
    fn tilted_normal(size: u32) -> TerrainImage {
        TerrainImage {
            width: size,
            height: size,
            rgba: (0..size * size).flat_map(|_| [250, 128, 140, 255]).collect(),
        }
    }

    /// Draw one triangle of a brush and read the centre pixel back.
    ///
    /// Renders rather than merely building the pipeline. Creating the pipeline
    /// proves the WGSL parses and nothing more -- it cannot tell whether the
    /// right array layer was sampled, whether the tangent frame is the right way
    /// round, or whether the normal map moves the shading at all.
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

    /// Off to one side, so a normal tilted along u faces it differently from a
    /// flat one. What the normal-map tests need.
    const SIDE_LIGHT: glam::Vec3 = glam::Vec3::new(4.0, 0.0, 2.0);

    /// Straight down the view axis, so the half-vector lands on the normal and
    /// the specular highlight is at its peak.
    ///
    /// What the ROUGHNESS tests need. With the side light both a mirror and a
    /// matte surface come out with no highlight at all -- the matte one because
    /// its strength is zero, the mirror because its lobe is a pinpoint the
    /// sample misses -- so the two are identical and the test proves nothing.
    const HEAD_ON_LIGHT: glam::Vec3 = glam::Vec3::new(0.0, 0.0, 4.0);

    /// The same, with roughness and ambient-occlusion arrays supplied.
    ///
    /// Empty slices fill with white -- fully rough, fully unoccluded -- which is
    /// what every surface looked like before either map existed, so the older
    /// tests measure exactly what they always did.
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
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        if lit {
            // Off to one side, so a normal tilted along u faces it differently
            // from a flat one. With an empty list `shade` returns the ambient
            // constant whatever the normal, and a normal-map test on that rig
            // would pass without the mapping doing anything.
            lights.upload(
                &queue,
                &[Light {
                    position: light_pos,
                    direction: glam::Vec3::new(-1.0, 0.0, 0.0),
                    kind: LightKind::Point,
                    color: Color3(255, 255, 255, 255),
                    intensity: 4.0,
                    range: 20.0,
                    cone_angle_deg: 90.0,
                }],
            );
        } else {
            lights.upload(&queue, &[]);
        }
        let (_shadows, uniforms) =
            scene_uniforms(&device, &lights);
        uniforms.upload(&queue, glam::Mat4::IDENTITY, TEST_EYE, &ShadowUpload::disabled());

        let pipeline = BrushPipeline::new(&device, format, &uniforms.layout);
        let materials =
            // These tests are about the COLOUR and NORMAL path, so both new
            // arrays are left empty: they fill with white, which is fully rough
            // and fully unoccluded -- the shading every one of them was written
            // against.
            BrushMaterials::new(
                &device, &queue, &pipeline.material_layout, colours, normals, roughs, aos,
            );

        // A triangle covering the viewport in clip space, facing +z, with the
        // face's u axis along +x -- the frame a wall brush actually produces.
        let v = |p: [f32; 3], uv: [f32; 2]| BrushVertex {
            position: p,
            normal: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv,
            material,
            tint,
            // These tests are about the MATERIAL path, so every vertex samples
            // the same lightmap texel and the harness binds a black one --
            // additively neutral, so it changes none of their measurements.
            uv2: [0.5, 0.5],
        };
        let verts = [
            v([-1.0, -1.0, 0.0], [0.0, 0.0]),
            v([3.0, -1.0, 0.0], [2.0 * uv_scale, 0.0]),
            v([-1.0, 3.0, 0.0], [0.0, 2.0 * uv_scale]),
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
        let lightmap = default_brush_lightmap(&device, &queue, &pipeline.lightmap_layout);

            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("brush_test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store },
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
            pass.set_bind_group(1, &materials.bind_group, &[]);
            // Black: additively neutral, so these material tests measure the
            // material path and nothing else.
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
        device.poll(PollType::Wait).ok();
        let data = slice.get_mapped_range();
        let c = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[c], data[c + 1], data[c + 2], data[c + 3]])
    }

    /// Three materials, distinguishable by which channel dominates.
    fn palette() -> Vec<Option<TerrainImage>> {
        vec![
            Some(flat([200, 20, 20], 4)),
            Some(flat([20, 200, 20], 4)),
            Some(flat([20, 20, 200], 4)),
        ]
    }

    #[test]
    fn a_face_samples_the_material_its_index_names() {
        // The whole reason the index is per vertex: one draw call, many
        // materials. If the array layer were ignored every wall in a level
        // would come out the same colour, which reads as a texture-loading
        // failure rather than as an indexing one.
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
        // A face whose material is missing binds a white layer, so the tint is
        // the entire appearance -- which is what keeps an unassigned brush
        // looking like the colour its object was authored in.
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
        // The fallback layer is 128,128,255 -- straight out of the surface. A
        // fallback of zeros would decode to (-1,-1,-1) and light every
        // untextured wall as though it faced away from everything.
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
        // THE POINT of the roughness map. Without it every surface in a level
        // shades identically -- polished concrete lights exactly like rough
        // brick -- and no work on the lighting can tell them apart.
        //
        // A smooth surface concentrates the same energy into a tighter,
        // brighter highlight, so head on it must out-shine a matte one.
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
    fn a_missing_roughness_map_shades_fully_matte() {
        // The fallback has to be the OLD behaviour exactly, or adding this
        // feature relights every level already built. An absent array fills
        // with white, which is roughness 1.
        let Some(absent) = render_brush_material(
            0, &palette(), &[None], &[], &[], [1.0; 4], 1.0, true, HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let explicit_matte = render_brush_material(
            0, &palette(), &[None], &[Some(flat([255, 255, 255], 4))], &[], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        )
        .unwrap();
        assert_eq!(
            absent, explicit_matte,
            "no roughness map must shade as fully rough: {absent:?} vs {explicit_matte:?}",
        );
    }

    #[test]
    fn ambient_occlusion_darkens_without_wiping_out_direct_light() {
        // AO says how much of the SKY a crevice can see. Applying it to direct
        // light as well would darken a surface a lamp is shining straight onto,
        // which is a different effect and a wrong one.
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
    fn a_normal_map_changes_the_shading() {
        // Lit from one side, a surface tilted toward the light must come out
        // brighter than a flat one. Without this the normal map could be bound,
        // sampled, and multiplied by a tangent frame that discards it, and
        // every other test here would still pass.
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
    fn a_material_beyond_the_array_does_not_read_out_of_bounds() {
        // A level that outgrew the array must draw something rather than
        // sampling whatever is past the end. The loader clamps; this pins that
        // the last layer is a real, bound layer rather than undefined.
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

    /// Four columns: black, red, black, blue. Needs spatial variation, because
    /// a one-colour map samples identically whether the address mode repeats or
    /// clamps -- which is how a vacuous version of the test below passed.
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
        // The uv is in TILES: a 6m wall of a material tiling every metre spans
        // uv 0..6. Clamping would stretch the last column across five of them,
        // which looks like a broken texture rather than a wrong address mode.
        //
        // Asserted as a PROPERTY rather than by predicting where a pixel lands.
        // Working out the exact uv of the centre pixel means re-deriving the
        // rasteriser's sample position, and a first attempt at that produced a
        // test that failed against a correct sampler. What is true without any
        // of that arithmetic: past uv 1 a repeating sampler keeps moving through
        // the columns, so different scales give different colours, while a
        // clamping one is pinned to the edge column and gives one colour for
        // every scale.
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
}
