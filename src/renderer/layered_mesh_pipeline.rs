//! Meshes textured by per-vertex layer weights: caves, and anything else carved.
//!
//! WHAT THIS IS FOR
//!
//! The editor bakes a voxel-sculpted cave to a glTF mesh and paints each vertex
//! with four blend weights over the project's terrain layers. Nothing rendered
//! those weights: the mesh path samples ONE texture at a uv, a surface-nets mesh
//! has no uv, and a cave with no baseColorTexture therefore drew as a flat grey
//! blob however carefully it had been painted.
//!
//! THE SAME MATERIALS THE GROUND USES, DELIBERATELY
//!
//! This binds `terrain_pipeline`'s material bind group unchanged -- the same
//! four layers, the same normal maps, the same metres-per-tile. A cave is a hole
//! in the terrain, and the mouth of one is a place where cave rock and ground
//! rock meet within a metre of each other. Giving caves their own material set
//! would make matching them a permanent art chore that is impossible to get
//! right, because the two would be sampled and tiled differently. Sharing the
//! set makes them match by construction, and it means the layer indices the
//! editor's picker offers are the same indices everywhere.
//!
//! WHY NOT JUST EXTEND THE MESH PIPELINE
//!
//! Because the mesh pipeline already uses all four bind groups (camera+lights,
//! model, texture, lightmap) and Quest-class limits give us no fifth. This path
//! needs the material array and does not need a lightmap -- a cave is lit
//! dynamically, and baking one would defeat the point of being able to break it
//! -- so it fits in three groups by dropping the binding it does not want.
//!
//! THE TWO HEAVIEST LAYERS, NOT ALL FOUR
//!
//! Terrain samples all four layers unconditionally because it must: the weights
//! come from a texture and every fragment has to take the same path. Here the
//! weights arrive per VERTEX, and a surface-nets vertex takes its weights from a
//! vote among the solid voxels around it -- so they are one-hot almost
//! everywhere and pairwise on a boundary. A third layer with any weight at all
//! is rare and a fourth is essentially unheard of.
//!
//! So this ranks the weights and samples the top two, renormalised. Four layers
//! biplanar with normal maps is sixteen texture fetches per pixel, which is not
//! a thing a Quest 3 can afford on geometry that fills the view when you are
//! standing inside it. Two is eight. The cost is that at a point where three
//! layers genuinely meet, the third is dropped rather than faded -- a boundary
//! one voxel wide, against a saving of half the shader.

use bytemuck::{Pod, Zeroable};
use wgpu::*;

use super::lights::wgsl_lights_block;
use super::material_wgsl::{wgsl_biplanar_block, wgsl_whiteout_block};

/// A vertex of a layer-blended mesh.
///
/// No uv, on purpose: this geometry has none, and carrying a dead one would
/// invite someone to author into it. See `material_wgsl` for what replaces it.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LayeredVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// Blend weights over the four material layers, from the glTF `COLOR_0`.
    pub weights: [f32; 4],
}

impl LayeredVertex {
    pub const ATTRIBS: [VertexAttribute; 3] =
        vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4];

    pub fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct LayeredMeshPipeline {
    pub pipeline: RenderPipeline,
    pub model_layout: BindGroupLayout,
}

impl LayeredMeshPipeline {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        material_layout: &BindGroupLayout,
    ) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, material_layout, FrontFace::Ccw, 1)
    }

    /// See `pipeline::SolidPipeline::new_multisampled` -- a pipeline's sample
    /// count must match the pass it runs in, so a 4x eye pass needs its own.
    pub fn new_multisampled(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        material_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(
            device, format, uniform_layout, material_layout, FrontFace::Ccw, samples,
        )
    }

    pub fn new_mirror(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        material_layout: &BindGroupLayout,
    ) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, material_layout, FrontFace::Cw, 1)
    }

    fn new_with_front_face(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        material_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("layered_mesh_shader"),
            source: ShaderSource::Wgsl(layered_mesh_shader().into()),
        });

        // Structurally identical to the mesh pipeline's model layout, which is
        // what lets a cave reuse a `mesh_pipeline::ModelUniform` bind group:
        // wgpu treats identical layout descriptors as compatible, so the two
        // paths do not need to agree about who owns the layout.
        let model_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("layered_mesh_model_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("layered_mesh_layout"),
            bind_group_layouts: &[uniform_layout, material_layout, &model_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("layered_mesh_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[LayeredVertex::layout()],
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
                // NO CULLING. A cave is seen from the inside, and the surface
                // between a chamber and the rock is one sheet of triangles --
                // cull the back and a player standing in the chamber is looking
                // at the void through the wall they came in by.
                cull_mode: None,
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

        Self { pipeline, model_layout }
    }

    pub fn create_model_uniform(&self, device: &Device) -> super::mesh_pipeline::ModelUniform {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("layered_mesh_model_uniform"),
            size: 64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("layered_mesh_model_bg"),
            layout: &self.model_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        super::mesh_pipeline::ModelUniform { buffer, bind_group }
    }
}

fn layered_mesh_shader() -> String {
    format!(
        r#"
// Group 0 -- the camera, the lights and both shadow maps -- is declared by
// `wgsl_lights_block` below, so there is one description of that layout rather
// than one per shader.

// Group 1 is terrain's material bind group, bound here unchanged. `splat_tex`
// is part of that layout and is deliberately not declared: a cave's weights come
// from its vertices, and a binding a shader does not read costs nothing.
@group(1) @binding(0) var layer_tex: texture_2d_array<f32>;
@group(1) @binding(1) var layer_samp: sampler;
@group(1) @binding(2) var macro_tex: texture_2d<f32>;

struct Material {{
    repeat: vec4<f32>,
    slope_start_deg: f32,
    slope_end_deg: f32,
    height_start: f32,
    height_end: f32,
    macro_repeat: f32,
    macro_strength: f32,
    biplanar_start_deg: f32,
    use_splat: f32,
    normal_strength: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}}
@group(1) @binding(3) var<uniform> mat: Material;
@group(1) @binding(5) var normal_tex: texture_2d_array<f32>;

struct ModelUniform {{ model: mat4x4<f32> }}
@group(2) @binding(0) var<uniform> model_u: ModelUniform;

{lights_block}
{biplanar_block}
{whiteout_block}

struct VIn {{
    @location(0) pos: vec3<f32>,
    @location(1) norm: vec3<f32>,
    @location(2) weights: vec4<f32>,
}}

struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) weights: vec4<f32>,
}}

@vertex fn vs_main(v: VIn) -> VOut {{
    let world = model_u.model * vec4<f32>(v.pos, 1.0);
    var out: VOut;
    out.clip      = camera.view_proj * world;
    out.normal    = (model_u.model * vec4<f32>(v.norm, 0.0)).xyz;
    out.world_pos = world.xyz;
    out.weights   = v.weights;
    return out;
}}

struct TopTwo {{ a: i32, b: i32, wa: f32, wb: f32 }}

// The two heaviest layers, renormalised so they sum to one.
//
// All-zero weights become layer 0 at full strength rather than black. That is
// not defensive noise: a mesh with no COLOR_0 at all interpolates to zero here,
// and the failure it would otherwise produce -- a cave rendering pure black --
// looks exactly like a lighting bug and would be chased in the wrong place.
fn top_two(w: vec4<f32>) -> TopTwo {{
    let total = w.x + w.y + w.z + w.w;
    let safe = select(vec4<f32>(1.0, 0.0, 0.0, 0.0), w, total > 0.0001);
    var v = array<f32, 4>(safe.x, safe.y, safe.z, safe.w);

    var a: i32 = 0;
    for (var i: i32 = 1; i < 4; i = i + 1) {{
        if (v[i] > v[a]) {{ a = i; }}
    }}
    var b: i32 = -1;
    for (var i: i32 = 0; i < 4; i = i + 1) {{
        if (i != a && (b < 0 || v[i] > v[b])) {{ b = i; }}
    }}

    var out: TopTwo;
    out.a = a;
    out.b = b;
    let s = max(v[a] + v[b], 0.0001);
    out.wa = v[a] / s;
    out.wb = v[b] / s;
    return out;
}}

fn layer_colour(layer: i32, b: Biplanar) -> vec3<f32> {{
    let c_major = textureSample(layer_tex, layer_samp, b.uv_major, layer).rgb;
    let c_minor = textureSample(layer_tex, layer_samp, b.uv_minor, layer).rgb;
    return mix(c_minor, c_major, b.w);
}}

// Biplanar normal mapping, which terrain deliberately does not do.
//
// Terrain samples its normal maps top-down only, because most of a map is flat
// ground and the stretch on a cliff is a fair trade for four fewer fetches. A
// cave has no flat majority -- it is ceiling and wall and floor in equal measure
// -- so a top-down normal map would be stretched over most of what a player is
// looking at, and the shading would read as smeared rather than as rock.
fn layer_normal(layer: i32, b: Biplanar, n: vec3<f32>) -> vec3<f32> {{
    let p_major = textureSample(normal_tex, layer_samp, b.uv_major, layer).rgb;
    let p_minor = textureSample(normal_tex, layer_samp, b.uv_minor, layer).rgb;

    // OpenGL convention: green is +Y in tangent space, matching the NormalGL
    // maps the installer fetches. A DX map here lights every bump from the
    // wrong side.
    var t_major = p_major * 2.0 - 1.0;
    var t_minor = p_minor * 2.0 - 1.0;
    t_major = vec3<f32>(t_major.xy * mat.normal_strength, t_major.z);
    t_minor = vec3<f32>(t_minor.xy * mat.normal_strength, t_minor.z);

    let n_major = whiteout(b.axis_major, t_major, n);
    let n_minor = whiteout(b.axis_minor, t_minor, n);
    return normalize(mix(n_minor, n_major, b.w));
}}

// Metres per tile for one layer.
//
// Through a local array rather than indexing `mat.repeat` directly: dynamic
// indexing of a vector in the uniform address space is the kind of thing
// backends disagree about, and this costs four moves.
fn repeat_of(layer: i32) -> f32 {{
    var r = array<f32, 4>(mat.repeat.x, mat.repeat.y, mat.repeat.z, mat.repeat.w);
    return r[layer];
}}

@fragment fn fs_main(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {{
    // Renormalised because interpolation across a triangle shortens a normal
    // wherever its corners disagree, and a short normal darkens the surface.
    var n = normalize(in.normal);

    // FACE THE VIEWER. Culling is off, so the far side of the sheet arrives with
    // its normal pointing away from the eye; `shade` then finds every light
    // behind the surface, returns ambient alone, and the chamber renders
    // near-black from the inside. That is precisely the bug the editor preview
    // hit, and it was only found by looking at a screenshot -- so the fix goes
    // in the shader rather than in the mesher's choice of winding, where the
    // next mesh to arrive wound the other way would reintroduce it.
    //
    // `front_facing` and not a dot against the camera: it is the rasteriser's
    // own answer, needs no camera position in the uniform, and is right even
    // where the interpolated normal has been bent away from the geometry.
    n = select(-n, n, front);

    let t = top_two(in.weights);
    let ba = biplanar_axes(in.world_pos, n, repeat_of(t.a));
    let bb = biplanar_axes(in.world_pos, n, repeat_of(t.b));

    let albedo_raw = layer_colour(t.a, ba) * t.wa + layer_colour(t.b, bb) * t.wb;

    var shaded_n = layer_normal(t.a, ba, n) * t.wa + layer_normal(t.b, bb, n) * t.wb;
    shaded_n = normalize(select(n, shaded_n, length(shaded_n) > 0.0001));

    // Macro variation, sampled in the plane the surface faces rather than from
    // above. A cave wall sampled top-down gets one macro value down its whole
    // height, which is the one place the variation would do nothing at all.
    let mb = biplanar_axes(in.world_pos, n, mat.macro_repeat);
    let m = textureSample(macro_tex, layer_samp, mb.uv_major).r;
    let albedo = albedo_raw * (1.0 + (m - 0.5) * 2.0 * mat.macro_strength);

    let lit = shade(in.world_pos, shaded_n);
    return vec4<f32>(albedo * lit, 1.0);
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
        biplanar_block = wgsl_biplanar_block(),
        whiteout_block = wgsl_whiteout_block(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::lights::{Light, LightKind, LightsUniform};
    use crate::renderer::terrain_pipeline::{
        self, material_bind_group_layout, TerrainImage, TerrainMaterial, TerrainMaterialUniform,
    };
    use crate::renderer::uniforms::test_support::{scene_uniforms, TEST_EYE};
    use crate::renderer::uniforms::ShadowUpload;
    use crate::renderer::Color3;
    use wgpu::util::DeviceExt;

    /// A one-texel image of one colour.
    fn solid(rgba: [u8; 4]) -> TerrainImage {
        TerrainImage { width: 1, height: 1, rgba: rgba.to_vec() }
    }

    /// Two vertical bands, so a change of PROJECTION PLANE changes the sample.
    ///
    /// A solid layer cannot detect that at all -- it reads the same from every
    /// plane -- so the biplanar test needs something with structure in it.
    fn banded(left: [u8; 4], right: [u8; 4]) -> TerrainImage {
        let mut rgba = Vec::with_capacity(4 * 4 * 4);
        for _y in 0..4 {
            for x in 0..4 {
                rgba.extend_from_slice(if x < 2 { &left } else { &right });
            }
        }
        TerrainImage { width: 4, height: 4, rgba }
    }

    /// A normal map tilted hard in +u, for asserting that it reaches the shading.
    fn tilted_normal() -> TerrainImage {
        let mut rgba = Vec::new();
        for _ in 0..16 {
            rgba.extend_from_slice(&[250, 128, 140, 255]);
        }
        TerrainImage { width: 4, height: 4, rgba }
    }

    /// What a render asks for. Grouped rather than passed as eight positional
    /// arguments, which is how a harness ends up with call sites nobody can read.
    struct Shot {
        normal: [f32; 3],
        weights: [f32; 4],
        /// World position of the whole quad. Biplanar uvs come from world space,
        /// so moving the surface is how a test changes which texel it lands on.
        world: [f32; 3],
        lit: bool,
        /// Where the light sits, when `lit`. Off to one side by default so that
        /// tilting the surface changes how much light reaches it; a test of the
        /// normal's SIGN needs it head-on instead.
        light_at: [f32; 3],
        /// Wind the triangle the other way, reversing which side faces the eye.
        flipped: bool,
        layers: [TerrainImage; 4],
        normals: [Option<TerrainImage>; 4],
        settings: TerrainMaterialUniform,
    }

    impl Default for Shot {
        fn default() -> Self {
            Self {
                normal: [0.0, 1.0, 0.0],
                weights: [1.0, 0.0, 0.0, 0.0],
                world: [0.0, 0.0, 0.0],
                lit: false,
                light_at: [3.0, 2.0, 0.0],
                flipped: false,
                // Primaries, so a readback names the winning layer with no
                // arithmetic: red, blue, green, white.
                layers: [
                    solid([255, 0, 0, 255]),
                    solid([0, 0, 255, 255]),
                    solid([0, 255, 0, 255]),
                    solid([255, 255, 255, 255]),
                ],
                normals: [None, None, None, None],
                settings: TerrainMaterialUniform {
                    // Off unless a test asks for it: macro variation multiplies
                    // the result by a value read from a texture, which would
                    // make every colour assertion here depend on a second
                    // texture's contents.
                    macro_strength: 0.0,
                    ..Default::default()
                },
            }
        }
    }

    /// Renders one full-screen triangle through the real pipeline and returns
    /// its centre pixel, or None when the machine has no GPU.
    ///
    /// `view_proj` is the identity, so the vertex positions ARE clip
    /// coordinates and the triangle cannot land off-screen. The world position
    /// the shader textures from is carried separately, which is what lets a
    /// test move the surface through the tiling without moving it out of view.
    fn render(shot: Shot) -> Option<[u8; 4]> {
        let (device, queue) = terrain_pipeline::tests::headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            scene_uniforms(&device, &lights);
        uniforms.upload(&queue, glam::Mat4::IDENTITY, TEST_EYE, &ShadowUpload::disabled());
        if shot.lit {
            lights.upload(&queue, &[Light {
                position: glam::Vec3::from(shot.light_at),
                direction: glam::Vec3::new(0.0, -1.0, 0.0),
                kind: LightKind::Point,
                color: Color3(255, 255, 255, 255),
                // Dim on purpose. Ambient alone already puts a pure-red layer at
                // 153 of 255, and a bright light clips the channel so that the
                // difference under test vanishes into saturation.
                intensity: 3.0,
                range: 50.0,
                cone_angle_deg: 180.0,
            }]);
        } else {
            lights.upload(&queue, &[]);
        }

        let material_layout = material_bind_group_layout(&device);
        let pipeline = LayeredMeshPipeline::new(&device, format, &uniforms.layout, &material_layout);

        let material = TerrainMaterial::new(
            &device,
            &queue,
            &material_layout,
            &shot.layers,
            &solid([128, 128, 128, 255]),
            None,
            &shot.normals,
            shot.settings,
        );

        // Where the quad SITS in world space, without moving it on screen.
        //
        // The shader takes its texture coordinates from the world position, so a
        // test that wants a different texel has to move the surface -- but a
        // surface moved in clip space leaves the 8x8 target and the readback is
        // then just clear colour. So the model matrix translates by `world` and
        // view_proj translates back: `clip` is unchanged, `world_pos` is not.
        let world = glam::Vec3::from(shot.world);
        let model = pipeline.create_model_uniform(&device);
        model.upload(&queue, glam::Mat4::from_translation(world));
        uniforms.upload(&queue, glam::Mat4::from_translation(-world), TEST_EYE, &ShadowUpload::disabled());

        let v = |p: [f32; 3]| LayeredVertex {
            position: p,
            normal: shot.normal,
            weights: shot.weights,
        };
        let verts = [
            v([-1.0, -1.0, 0.5]),
            v([3.0, -1.0, 0.5]),
            v([-1.0, 3.0, 0.5]),
        ];
        let idx: [u32; 3] = if shot.flipped { [0, 2, 1] } else { [0, 1, 2] };

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layered_test_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layered_test_ib"),
            contents: bytemuck::cast_slice(&idx),
            usage: BufferUsages::INDEX,
        });

        const SIZE: u32 = 8;
        let target = device.create_texture(&TextureDescriptor {
            label: Some("layered_test_target"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let depth = device.create_texture(&TextureDescriptor {
            label: Some("layered_test_depth"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("layered_test_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("layered_test_pass"),
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
            pass.set_bind_group(1, &material.bind_group, &[]);
            pass.set_bind_group(2, &model.bind_group, &[]);
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
        let at = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[at], data[at + 1], data[at + 2], data[at + 3]])
    }

    /// Skip cleanly on a machine with no adapter, rather than failing.
    macro_rules! shot {
        ($s:expr) => {
            match render($s) {
                Some(px) => px,
                None => {
                    eprintln!("skipping: no GPU adapter available");
                    return;
                }
            }
        };
    }

    #[test]
    fn a_vertex_weighted_to_one_layer_renders_that_layer() {
        // THE CLAIM THE WHOLE PIPELINE EXISTS FOR: the weights the editor
        // painted onto the mesh choose the material. Layer 1 is blue.
        let px = shot!(Shot { weights: [0.0, 1.0, 0.0, 0.0], ..Default::default() });
        assert!(px[2] > px[0], "layer 1 (blue) should have won, got {px:?}");

        let px0 = shot!(Shot { weights: [1.0, 0.0, 0.0, 0.0], ..Default::default() });
        assert!(px0[0] > px0[2], "layer 0 (red) should have won, got {px0:?}");
    }

    #[test]
    fn two_layers_blend_rather_than_switching() {
        // A hard switch between materials reads as a painted-on seam. Red and
        // green at equal weight must produce both channels.
        let px = shot!(Shot { weights: [0.5, 0.0, 0.5, 0.0], ..Default::default() });
        assert!(px[0] > 8, "layer 0 vanished from the blend: {px:?}");
        assert!(px[1] > 8, "layer 2 vanished from the blend: {px:?}");
        assert!(px[2] < 8, "layer 1 was never weighted, but is present: {px:?}");
    }

    #[test]
    fn weights_that_do_not_sum_to_one_still_blend_evenly() {
        // The mesher normalises, but a hand-made file has no such guarantee and
        // unnormalised weights must not blow out or black out the surface.
        let even = shot!(Shot { weights: [0.5, 0.0, 0.5, 0.0], ..Default::default() });
        let tiny = shot!(Shot { weights: [0.01, 0.0, 0.01, 0.0], ..Default::default() });
        for c in 0..3 {
            assert!(
                (even[c] as i32 - tiny[c] as i32).abs() <= 2,
                "renormalisation failed: {even:?} vs {tiny:?}",
            );
        }
    }

    #[test]
    fn a_mesh_with_no_weights_at_all_renders_layer_zero() {
        // A glTF with no COLOR_0 interpolates to zero here. Falling through to
        // black would look exactly like a lighting bug and would be chased in
        // the wrong place for an afternoon.
        let px = shot!(Shot { weights: [0.0, 0.0, 0.0, 0.0], ..Default::default() });
        assert!(px[0] > 8, "all-zero weights rendered black instead of layer 0: {px:?}");
    }

    #[test]
    fn only_the_two_heaviest_layers_are_sampled() {
        // The documented trade, asserted rather than left as a comment: a third
        // layer at low weight is dropped, not faded. Layer 2 is the only green
        // one, so its absence is readable in a single channel.
        let px = shot!(Shot { weights: [0.5, 0.4, 0.1, 0.0], ..Default::default() });
        assert!(px[0] > 8, "layer 0 should be present: {px:?}");
        assert!(px[2] > 8, "layer 1 should be present: {px:?}");
        assert_eq!(px[1], 0, "the third layer was sampled after all: {px:?}");
    }

    #[test]
    fn the_projection_plane_follows_the_surface_normal() {
        // Biplanar's whole purpose. The same world position, textured on a
        // floor and on a wall, must land on different texels of a banded
        // layer -- otherwise a cave wall is getting the ceiling's projection
        // and the texture is smeared down it.
        // Every layer 4x4: they share one D2Array, which cannot hold two sizes.
        let layers = || {
            [
                banded([255, 0, 0, 255], [0, 0, 255, 255]),
                banded([0, 255, 0, 255], [0, 255, 0, 255]),
                banded([0, 255, 0, 255], [0, 255, 0, 255]),
                banded([0, 255, 0, 255], [0, 255, 0, 255]),
            ]
        };
        // A point whose x and z fall in different bands of the 4-texel tile.
        let world = [1.0, 0.0, 3.0];
        let floor = shot!(Shot {
            normal: [0.0, 1.0, 0.0],
            world,
            layers: layers(),
            ..Default::default()
        });
        let wall = shot!(Shot {
            normal: [1.0, 0.0, 0.0],
            world,
            layers: layers(),
            ..Default::default()
        });
        assert_ne!(
            floor, wall,
            "floor and wall sampled the same texel, so the projection is not following the normal",
        );
    }

    #[test]
    fn a_fragment_seen_from_behind_is_shaded_as_though_it_faced_the_viewer() {
        // Standing INSIDE a cave. Culling is off, so the far side of the sheet
        // arrives with its normal pointing away; without the front-facing flip
        // `shade` finds every light behind the surface and returns ambient
        // alone. That is the near-black chamber the editor preview rendered,
        // and it was only ever found by looking at a screenshot.
        //
        // Stated so that it does not depend on knowing WHICH winding the
        // rasteriser calls front, because that answer involves wgpu's clip-space
        // y and is exactly the sort of thing to get backwards in a comment:
        // reversing the winding AND the authored normal must cancel out, and of
        // the two windings exactly one must be lit.
        let shot = |normal: [f32; 3], flipped: bool| Shot {
            lit: true,
            // Head-on, so the SIGN of the normal is the whole difference
            // between lit and ambient. The default side-on light gives a normal
            // of (0,0,1) a cosine of 0.14 either way, and the eight levels that
            // separates would not survive a rounding change.
            light_at: [0.0, 0.0, 5.0],
            normal,
            flipped,
            ..Default::default()
        };
        let toward = shot!(shot([0.0, 0.0, 1.0], false));
        let away_flipped = shot!(shot([0.0, 0.0, -1.0], true));
        for c in 0..3 {
            assert!(
                (toward[c] as i32 - away_flipped[c] as i32).abs() <= 2,
                "reversing both the winding and the normal changed the shading: \
                 {toward:?} vs {away_flipped:?}",
            );
        }

        // And the flip is doing something. Which of the two windings comes out
        // lit is the rasteriser's convention and not this shader's claim, so
        // the assertion is that they DIFFER -- which, with the equality above,
        // pins the behaviour exactly.
        let away = shot!(shot([0.0, 0.0, 1.0], true));
        assert!(
            (toward[0] as i32 - away[0] as i32).abs() > 8,
            "both windings shade the same, so the normal is never flipped: \
             {toward:?} vs {away:?}",
        );
    }

    #[test]
    fn a_normal_map_changes_the_shading() {
        // Without a light in the list `shade` returns the ambient constant
        // whatever the normal, so this test would pass with the normal mapping
        // doing nothing at all. `lit: true` is load-bearing.
        let flat = shot!(Shot {
            lit: true,
            normals: [Some(solid([128, 128, 255, 255])), None, None, None],
            ..Default::default()
        });
        let bumpy = shot!(Shot {
            lit: true,
            normals: [Some(tilted_normal()), None, None, None],
            ..Default::default()
        });
        assert_ne!(flat, bumpy, "the normal map did not reach the shading");
    }

    #[test]
    fn a_normal_map_on_a_wall_tilts_along_the_wall_s_own_axes() {
        // THE CLASSIC TRIPLANAR NORMAL BUG, which the other normal tests cannot
        // see. Every projection needs its own swizzle -- on the y plane the
        // map's u runs along world x, on the x plane it runs along world z --
        // and using one swizzle for all three still passes a flat-map no-op
        // test, still passes a "the map changed something" test on a floor, and
        // lights every wall in the cave from the wrong side.
        //
        // So: a wall facing +x, with a map tilted hard in +u. Swizzled
        // correctly, u is world z and the surface leans into z, which a light
        // on +z and a light on -z disagree about. Swizzled as though it were the
        // floor, the surface stays pointing along x and the two lights give the
        // same answer.
        let wall = |light_at: [f32; 3]| Shot {
            lit: true,
            light_at,
            normal: [1.0, 0.0, 0.0],
            normals: [Some(tilted_normal()), None, None, None],
            ..Default::default()
        };
        let front = shot!(wall([0.0, 0.0, 5.0]));
        let behind = shot!(wall([0.0, 0.0, -5.0]));
        assert!(
            (front[0] as i32 - behind[0] as i32).abs() > 8,
            "a wall's normal map is not tilting along z, so it is using the \
             floor's swizzle: {front:?} vs {behind:?}",
        );
    }

    #[test]
    fn a_flat_normal_map_is_a_true_no_op() {
        // The fallback texel (128,128,255) unpacks to (0,0,1), so an unauthored
        // normal set must cost its samples and change nothing. If it does not,
        // every material without a normal map is shaded subtly wrong and there
        // is nothing to compare it against.
        let none = shot!(Shot { lit: true, ..Default::default() });
        let flat = shot!(Shot {
            lit: true,
            normals: [Some(solid([128, 128, 255, 255])), None, None, None],
            ..Default::default()
        });
        for c in 0..3 {
            assert!(
                (none[c] as i32 - flat[c] as i32).abs() <= 2,
                "a flat normal map changed the shading: {none:?} vs {flat:?}",
            );
        }
    }

    /// Renders arbitrary geometry and returns the whole image, for the
    /// end-to-end test below. Separate from `render` because that harness owns
    /// its triangle on purpose -- every other test here wants a surface that
    /// fills the target and never has to think about where a pixel landed.
    fn render_geometry(
        verts: &[LayeredVertex],
        indices: &[u32],
        view_proj: glam::Mat4,
        size: u32,
    ) -> Option<Vec<[u8; 4]>> {
        let (device, queue) = terrain_pipeline::tests::headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            scene_uniforms(&device, &lights);
        uniforms.upload(&queue, view_proj, TEST_EYE, &ShadowUpload::disabled());
        lights.upload(&queue, &[]);

        let material_layout = material_bind_group_layout(&device);
        let pipeline = LayeredMeshPipeline::new(&device, format, &uniforms.layout, &material_layout);
        let material = TerrainMaterial::new(
            &device,
            &queue,
            &material_layout,
            &[
                solid([255, 0, 0, 255]),
                solid([0, 0, 255, 255]),
                solid([0, 255, 0, 255]),
                solid([255, 255, 255, 255]),
            ],
            &solid([128, 128, 128, 255]),
            None,
            &[None, None, None, None],
            TerrainMaterialUniform { macro_strength: 0.0, ..Default::default() },
        );
        let model = pipeline.create_model_uniform(&device);
        model.upload(&queue, glam::Mat4::IDENTITY);

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("e2e_vb"),
            contents: bytemuck::cast_slice(verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("e2e_ib"),
            contents: bytemuck::cast_slice(indices),
            usage: BufferUsages::INDEX,
        });

        let desc = |fmt, usage| TextureDescriptor {
            label: None,
            size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
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

        // Rows are padded to 256 bytes, which is the copy alignment wgpu
        // requires -- a target wider than 64 pixels would need more than one
        // row's worth and the readback indexing below would be wrong.
        assert!(size * 4 <= 256, "row padding assumption broken");
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("e2e_readback"),
            size: (256 * size) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("e2e_pass"),
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
            pass.set_bind_group(1, &material.bind_group, &[]);
            pass.set_bind_group(2, &model.bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
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
                    rows_per_image: Some(size),
                },
            },
            Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait).ok();
        let data = slice.get_mapped_range();
        let mut out = Vec::with_capacity((size * size) as usize);
        for y in 0..size as usize {
            for x in 0..size as usize {
                let at = y * 256 + x * 4;
                out.push([data[at], data[at + 1], data[at + 2], data[at + 3]]);
            }
        }
        Some(out)
    }

    #[test]
    fn the_editors_own_bake_renders_the_layers_it_painted() {
        // END TO END, across both languages: the editor's glb writer, the
        // engine's glTF loader, and this pipeline. The two halves are each
        // tested on their own above, and neither can see the seam between them
        // -- an attribute location that disagreed with the shader, or weights
        // read into the wrong vertex, would pass everything else here.
        let Some((device, queue, tex_layout)) = crate::renderer::mesh::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/layered_cave.glb");
        let mesh = crate::renderer::mesh::GltfMesh::load(&device, &queue, &tex_layout, &path)
            .expect("the editor's cave glb should load");
        let prim = &mesh.primitives[0];
        let layered = prim.layered.as_ref().expect("the fixture is marked layered");

        // The fixture's triangle spans x,y in 0..1, so this maps it across the
        // target: clip = (2x-1, 2y-1).
        const SIZE: u32 = 32;
        let view_proj = glam::Mat4::from_translation(glam::Vec3::new(-1.0, -1.0, 0.0))
            * glam::Mat4::from_scale(glam::Vec3::new(2.0, 2.0, 1.0));
        let img = render_geometry(&layered.vertices, &prim.indices, view_proj, SIZE)
            .expect("an adapter was already found");
        let px = |x: usize, y: usize| img[y * SIZE as usize + x];

        // Corner 0 carries [1,0,0,0] -> layer 0, which the harness paints red.
        // Corner 1 carries [0,1,0,0] -> layer 1, blue. Sampled a few pixels in
        // from each so the assertion does not depend on whether the rasteriser
        // covers the exact corner texel.
        let near_v0 = px(3, 28);
        let near_v1 = px(27, 29);
        assert!(
            near_v0[0] > near_v0[2],
            "the corner painted layer 0 did not come out red: {near_v0:?}",
        );
        assert!(
            near_v1[2] > near_v1[0],
            "the corner painted layer 1 did not come out blue: {near_v1:?}",
        );
    }

    #[test]
    fn both_winding_variants_of_the_pipeline_validate() {
        // WGSL is compiled by naga at PIPELINE CREATION, not by cargo build, so
        // a clean build says nothing about whether this shader is valid. The
        // mirror variant is only ever built on the XR path, which does not
        // compile on this machine at all -- so without this it would first be
        // validated on a headset.
        let Some((device, queue)) = terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let _ = &queue;
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            scene_uniforms(&device, &lights);
        let material_layout = material_bind_group_layout(&device);

        device.push_error_scope(ErrorFilter::Validation);
        let _a = LayeredMeshPipeline::new(
            &device, TextureFormat::Rgba8Unorm, &uniforms.layout, &material_layout,
        );
        let _b = LayeredMeshPipeline::new_mirror(
            &device, TextureFormat::Rgba8Unorm, &uniforms.layout, &material_layout,
        );
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            panic!("layered mesh pipeline failed validation: {err}");
        }
    }
}
