use super::cuboid::{SolidVertex, WireVertex};
use super::lights::wgsl_lights_block;
use super::ssr::wgsl_ssr_block;
use wgpu::*;

pub fn lightmap_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("lightmap_bgl"),
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
            // The bounce DIRECTION map, sharing the lightmap's layout texel for
            // texel. Declared for every lightmap consumer, not only brushes:
            // wgpu deduplicates structurally identical bind group layouts, and
            // this renderer has already been bitten once by two layouts that
            // looked interchangeable and quietly stopped being so -- the avatar
            // stopped casting shadows and nothing errored. One layout, one
            // shape, and shaders that do not want the map simply never sample
            // it.
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // The brush SUN MASK and its sampler, on every lightmap for the
            // same one-layout reason as the direction map above. Only the brush
            // shader samples them; see `create_lightmap_texture_with_sun`.
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            // The STATIONARY LAMPS' baked shadows: one RGBA layer per four
            // lamps, a signed distance per channel, on the brush charts at
            // `STATIONARY_MASK_SCALE` times the lightmap's density. Declared on
            // every lightmap for the one-layout reason above; only the brush
            // shader samples it, through the sun mask's sampler.
            BindGroupLayoutEntry {
                binding: 5,
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

pub struct SolidPipeline {
    pub pipeline: RenderPipeline,
    pub lightmap_layout: BindGroupLayout,
}
pub struct WirePipeline {
    pub pipeline: RenderPipeline,
}

impl SolidPipeline {
    pub fn new(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        Self::new_with_front_face(
            device, format, uniform_layout, FrontFace::Ccw, 1,
            crate::renderer::multiview::ViewMode::Mono,
        )
    }

    /// Same pipeline, built for a multisampled target.
    ///
    /// A pipeline's sample count must equal that of the pass it runs in, and
    /// wgpu will not let you mix them -- so a pipeline used in both the eye pass
    /// and an offscreen 1x pass has to exist twice. `new` is the 1x form and
    /// stays the default, which keeps every existing call site honest about
    /// what it is drawing into.
    pub fn new_multisampled(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(
            device, format, uniform_layout, FrontFace::Ccw, samples,
            crate::renderer::multiview::ViewMode::Mono,
        )
    }

    /// The same, drawing BOTH EYES in one pass. See `multiview::ViewMode`.
    pub fn new_multisampled_stereo(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(
            device, format, uniform_layout, FrontFace::Ccw, samples,
            crate::renderer::multiview::ViewMode::Stereo,
        )
    }

    pub fn new_mirror(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        Self::new_with_front_face(
            device, format, uniform_layout, FrontFace::Cw, 1,
            crate::renderer::multiview::ViewMode::Mono,
        )
    }

    fn new_with_front_face(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("solid_shader"),
            source: ShaderSource::Wgsl(view.shader(solid_shader()).into()),
        });
        let lightmap_layout = lightmap_bind_group_layout(device);
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("solid_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(&lightmap_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("solid_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(SolidVertex::layout())],
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
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });
        Self {
            pipeline,
            lightmap_layout,
        }
    }

    /// The reflective variant, drawn in the final 1x eye pass -- it SAMPLES the
    /// scene texture, so it cannot run in the pass that produces it.
    pub fn new_ssr(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_camera_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,
        // Must match the scene target's depth sample count -- see
        // `ssr::SsrPipelines::new_with_depth_samples`.
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("solid_ssr_shader"),
            source: ShaderSource::Wgsl(solid_ssr_shader().into()),
        });
        let lightmap_layout = lightmap_bind_group_layout(device);
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("solid_ssr_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(&lightmap_layout), Some(ssr_camera_layout), Some(ssr_scene_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("solid_ssr_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(SolidVertex::layout())],
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
        Self {
            pipeline,
            lightmap_layout,
        }
    }
}

impl WirePipeline {
    pub fn new(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        Self::new_multisampled(device, format, uniform_layout, 1)
    }

    /// See `SolidPipeline::new_multisampled`.
    pub fn new_multisampled(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    /// The same, drawing BOTH EYES in one pass. See `multiview::ViewMode`.
    pub fn new_multisampled_stereo(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Stereo)
    }

    fn new_with_view(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("wire_shader"),
            source: ShaderSource::Wgsl(view.shader(WIRE_SHADER.to_string()).into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("wire_layout"),
            bind_group_layouts: &[Some(uniform_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("wire_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(WireVertex::layout())],
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
                topology: PrimitiveTopology::LineList,
                cull_mode: None,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::LessEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });
        Self { pipeline }
    }
}

fn solid_shader() -> String {
    format!(
        r#"
// Group 0 -- the camera, the lights and both shadow maps -- is declared by
// `wgsl_lights_block` below, so there is one description of that layout rather
// than one per shader.

@group(1) @binding(0) var lm_tex: texture_2d<f32>;
@group(1) @binding(1) var lm_samp: sampler;

{lights_block}

struct VIn  {{ @location(0) pos: vec3<f32>, @location(1) norm: vec3<f32>, @location(2) col: vec4<f32>, @location(3) uv2: vec2<f32> }}
struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) col: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) uv2: vec2<f32>,
}}

@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    out.clip      = cam_view_proj() * vec4<f32>(v.pos, 1.0);
    out.col       = v.col;
    out.normal    = v.norm;
    out.world_pos = v.pos;
    out.uv2       = v.uv2;
    return out;
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let n = normalize(in.normal);
    // The baked lamps are in this lightmap. See `receiver_skips_baked`.
    receiver_skips_baked = true;
    let lit = shade(in.world_pos, n);
    let lightmap = textureSample(lm_tex, lm_samp, in.uv2).rgb;
    return vec4<f32>(in.col.rgb * lit * lightmap, in.col.a);
}}
"#,
        lights_block = wgsl_lights_block(0, 1)
    )
}

fn solid_ssr_shader() -> String {
    format!(
        r#"
// Group 0 -- the camera, the lights and both shadow maps -- is declared by
// `wgsl_lights_block` below, so there is one description of that layout rather
// than one per shader.

@group(1) @binding(0) var lm_tex: texture_2d<f32>;
@group(1) @binding(1) var lm_samp: sampler;

{lights_block}

{ssr_block}

struct VIn  {{
    @location(0) pos: vec3<f32>,
    @location(1) norm: vec3<f32>,
    @location(2) col: vec4<f32>,
    @location(3) uv2: vec2<f32>,
    @location(4) reflectivity: f32,
}}
struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) col: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) uv2: vec2<f32>,
    @location(4) reflectivity: f32,
}}

@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    out.clip         = cam_view_proj() * vec4<f32>(v.pos, 1.0);
    out.col          = v.col;
    out.normal       = v.norm;
    out.world_pos    = v.pos;
    out.uv2          = v.uv2;
    out.reflectivity = v.reflectivity;
    return out;
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let n = normalize(in.normal);
    // The baked lamps are in this lightmap. See `receiver_skips_baked`.
    receiver_skips_baked = true;
    let lit = shade(in.world_pos, n);
    let lightmap = textureSample(lm_tex, lm_samp, in.uv2).rgb;
    let base = in.col.rgb * lit * lightmap;
    // A cuboid has no roughness map, so it reflects sharply and falls back to
    // its own colour -- the behaviour it had before the chain existed.
    let reflected = ssr_reflect(in.world_pos, n, base, in.reflectivity, 0.0);
    return vec4<f32>(reflected, in.col.a);
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
        ssr_block = wgsl_ssr_block(2, 3)
    )
}

// This one keeps its own two-field struct rather than taking the whole of group
// 0 from `wgsl_lights_block`: it never calls `shade`, and a uniform struct that
// is a PREFIX of the buffer bound to it is legal. So it is unaffected by
// anything the lighting block adds -- which is the point of the arrangement, and
// also why a rename that swept through the shading shaders must not touch this
// one. (It did, once. Nothing on this machine built the wire pipeline on a real
// device, so it went unnoticed until a test did.)
const WIRE_SHADER: &str = r#"
// ONE MATRIX PER EYE, like every other scene shader, even though this one
// carries no lights block. It used to declare a single `mat4x4` and read the
// front of the buffer, which is the left eye's -- correct while both slots hold
// the same matrix, and the RIGHT EYE DRAWN WITH THE LEFT EYE'S PROJECTION the
// moment a multiview pass gives them different ones. In std140 an array of
// mat4x4 is laid out as consecutive mat4x4, so slot 0 is the same bytes this
// read before and the single-view path is unchanged.
struct Uniforms { view_proj: array<mat4x4<f32>, 2> }
@group(0) @binding(0) var<uniform> u: Uniforms;
// DECLARED HERE because this shader carries no lights block, which is where
// every other scene shader gets it. 0 is the left eye, which is what a
// single-view pass wants; `as_multiview` assigns the real one per view and
// leaves this declaration alone.
var<private> view_slot: i32 = 0;

struct VIn  { @location(0) pos: vec3<f32>, @location(1) col: vec4<f32> }
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) col: vec4<f32> }

@vertex fn vs_main(v: VIn) -> VOut {
    var p = u.view_proj[view_slot] * vec4<f32>(v.pos, 1.0);
    p.z -= 0.0001 * p.w;
    return VOut(p, v.col);
}
@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> { return in.col; }
"#;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::renderer::lights::LightsUniform;
    use crate::renderer::ssr::SsrPipelines;

    pub(crate) fn headless_gpu() -> Option<(Device, Queue)> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            apply_limit_buckets: false,
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: Features::empty(),
            required_limits: crate::renderer::uniforms::scene_limits(Limits::default()),
            ..Default::default()
        }))
        .ok()
    }

    #[test]
    fn solid_ssr_pipeline_builds_on_a_real_device() {
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let format = TextureFormat::Rgba8UnormSrgb;

        let lights_uniform = LightsUniform::new(&device);
        let (_shadows, uniform_buf) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights_uniform);
        let ssr_pipelines = SsrPipelines::new(&device, format);

        let _solid_ssr = SolidPipeline::new_ssr(
            &device,
            format,
            &uniform_buf.layout,
            ssr_pipelines.camera_layout(),
            ssr_pipelines.scene_texture_layout(),
        );
    }

    /// Everything the multisampled scene pass draws with, built at 4x.
    ///
    /// WGSL is compiled by naga at PIPELINE CREATION and a sample count is
    /// validated there too, so a pipeline built for the wrong count fails here
    /// rather than on a headset. This is the only place that happens before the
    /// device does it.
    #[test]
    fn every_scene_pass_pipeline_builds_multisampled() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let format = TextureFormat::Rgba8UnormSrgb;
        let lights_uniform = LightsUniform::new(&device);
        let (_shadows, u) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights_uniform);

        // 4 only. WebGPU guarantees [1, 4] for a colour format and nothing
        // else -- 2x needs an optional adapter feature, which is why
        // `MsaaLevel` does not offer it. This test is where that was found.
        let err_scope_1 = device.push_error_scope(ErrorFilter::Validation);
        for samples in [4u32] {
            let _solid = SolidPipeline::new_multisampled(&device, format, &u.layout, samples);
            let _wire = WirePipeline::new_multisampled(&device, format, &u.layout, samples);
            let _mesh = crate::renderer::mesh_pipeline::MeshPipeline::new_multisampled(
                &device, format, &u.layout, samples,
            );
            let _skinned = crate::renderer::mesh_pipeline::SkinnedMeshPipeline::new_multisampled(
                &device, format, &u.layout, samples,
            );
            let terrain = crate::renderer::terrain_pipeline::TerrainPipeline::new_multisampled(
                &device, format, &u.layout, samples,
            );
            let _brush = crate::renderer::brush_pipeline::BrushPipeline::new_multisampled(
                &device, format, &u.layout, samples,
            );
            let _layered =
                crate::renderer::layered_mesh_pipeline::LayeredMeshPipeline::new_multisampled(
                    &device, format, &u.layout, &terrain.material_layout, samples,
                );
            let _particle = crate::renderer::particle::ParticlePipeline::new_multisampled(
                &device, format, &u.layout, samples,
            );

            // The scene target and the shaders that read its depth have to
            // agree about the sample count, and a disagreement is a validation
            // error rather than a wrong image -- so it belongs here.
            let ssr = SsrPipelines::new_with_depth_samples(&device, format, samples);
            let _target =
                ssr.create_scene_target_multisampled(&device, format, 16, 16, samples);
            // Single-sampled whatever the scene pass runs at: everything that
            // reads depth now reads the resolved copy.
            let _ssr_solid = SolidPipeline::new_ssr(
                &device,
                format,
                &u.layout,
                ssr.camera_layout(),
                ssr.scene_texture_layout(),
            );
        }
        let _ = &queue;
        if let Some(err) = pollster::block_on(err_scope_1.pop()) {
            panic!("a multisampled pipeline failed validation: {err}");
        }
    }
}

/// Does multisampling actually antialias?
///
/// Building the pipelines proves they are legal. It says nothing about whether
/// the resolve is wired up -- a resolve target left as `None`, or a target
/// created at one sample count and a pipeline at another, produces a perfectly
/// valid hard-edged image. So this renders a SLANTED edge and looks at it.
#[cfg(test)]
mod msaa_tests {
    use super::tests::headless_gpu;
    use super::*;
    use crate::renderer::cuboid::SolidVertex;
    use crate::renderer::lights::LightsUniform;
    use wgpu::util::DeviceExt;

    const SIZE: u32 = 32;

    /// Draws one slanted triangle and returns the resolved image.
    fn edge(samples: u32) -> Option<Vec<[u8; 4]>> {
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let (_shadows, u) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        u.upload(
            &queue,
            glam::Mat4::IDENTITY,
            glam::Vec3::new(0.0, 0.0, 5.0),
            &crate::renderer::uniforms::ShadowUpload::disabled(),
        );
        lights.upload(&queue, &[]);

        let pipeline = SolidPipeline::new_multisampled(&device, format, &u.layout, samples);
        let white = crate::renderer::mesh::create_lightmap_texture(
            &device,
            &queue,
            &lightmap_bind_group_layout(&device),
            &[255u8, 255, 255, 255],
            1,
            1,
            None,
        );

        // A triangle with one edge at a shallow angle -- the case where a
        // staircase is most obvious and where averaging has the most to do.
        let v = |p: [f32; 3]| SolidVertex {
            position: p,
            normal: [0.0, 0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv2: [0.0, 0.0],
            reflectivity: 0.0,
        };
        // A right triangle filling half the target, so the hypotenuse is a long
        // slanted edge with plenty of interior either side of it. An earlier
        // version used a thin sliver, where MSAA made almost every pixel
        // partial -- so "how much is covered" moved for a reason that had
        // nothing to do with the resolve, and both assertions below misfired.
        let verts = [
            v([-0.9, -0.9, 0.5]),
            v([0.9, -0.9, 0.5]),
            v([-0.9, 0.9, 0.5]),
        ];
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("msaa_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("msaa_ib"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        let desc = |fmt, count, usage| TextureDescriptor {
            label: None,
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: count,
            dimension: TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        };
        // The resolved image, which is what anything downstream would sample.
        let resolved = device.create_texture(&desc(
            format,
            1,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let resolved_view = resolved.create_view(&Default::default());
        let msaa = (samples > 1).then(|| {
            device
                .create_texture(&desc(format, samples, TextureUsages::RENDER_ATTACHMENT))
                .create_view(&Default::default())
        });
        let depth = device.create_texture(&desc(
            TextureFormat::Depth32Float,
            samples,
            TextureUsages::RENDER_ATTACHMENT,
        ));
        let depth_view = depth.create_view(&Default::default());

        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("msaa_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let (view, resolve_target, store) = match msaa.as_ref() {
                Some(m) => (m, Some(&resolved_view), StoreOp::Discard),
                None => (&resolved_view, None, StoreOp::Store),
            };
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("msaa_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &u.bind_group, &[]);
            pass.set_bind_group(1, &white.bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..3, 0, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &resolved,
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
        let mut out = Vec::with_capacity((SIZE * SIZE) as usize);
        for y in 0..SIZE as usize {
            for x in 0..SIZE as usize {
                let at = y * 256 + x * 4;
                out.push([data[at], data[at + 1], data[at + 2], data[at + 3]]);
            }
        }
        Some(out)
    }

    /// The value a fully-covered pixel takes.
    ///
    /// Measured, not assumed: the surface is shaded, and with no lights in the
    /// list that is the shader's ambient constant rather than white. Assuming
    /// 255 made every interior pixel look "partial" and the first version of
    /// this test asserted nothing at all.
    fn interior(img: &[[u8; 4]]) -> u8 {
        *img.iter().map(|p| &p[0]).max().unwrap_or(&0)
    }

    /// Pixels neither background nor fully covered -- the edge itself.
    fn partial(img: &[[u8; 4]]) -> usize {
        let full = interior(img);
        img.iter()
            .filter(|p| p[0] > 4 && p[0] + 4 < full)
            .count()
    }

    /// Total coverage, as the sum of the channel over the whole image.
    fn mass(img: &[[u8; 4]]) -> u64 {
        img.iter().map(|p| p[0] as u64).sum()
    }

    /// Where the coverage sits, weighted by how covered each pixel is.
    ///
    /// This is what the test below actually asserts on, and mass is not.
    /// Coverage mass is only preserved by antialiasing IN EXPECTATION over
    /// where the edges fall -- on a 32x32 target whose triangle has about a
    /// hundred edge pixels against four hundred interior ones, a perfectly
    /// correct resolve moves it by around a tenth, and an earlier version of
    /// this test called that a failure. The centroid does not care: it pins
    /// that the resolve produced the same shape in the same place, which is the
    /// thing that actually goes wrong when a resolve target is misconfigured.
    fn centroid(img: &[[u8; 4]]) -> (f64, f64) {
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut w = 0.0;
        for (i, p) in img.iter().enumerate() {
            let v = p[0] as f64;
            sx += (i % SIZE as usize) as f64 * v;
            sy += (i / SIZE as usize) as f64 * v;
            w += v;
        }
        let w = w.max(1.0);
        (sx / w, sy / w)
    }

    #[test]
    fn multisampling_softens_a_slanted_edge_and_1x_does_not() {
        let Some(one) = edge(1) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let four = edge(4).expect("an adapter was already found");

        // Without MSAA every pixel is in or out: the edge is a staircase of
        // hard steps, and nothing lands in between.
        assert_eq!(
            partial(&one),
            0,
            "the 1x render produced blended pixels, so this test is not measuring what it thinks",
        );
        assert!(
            partial(&four) > 8,
            "4x produced {} partially-covered pixels along a slanted edge -- \
             the resolve is not happening",
            partial(&four),
        );
    }

    #[test]
    fn the_resolved_image_still_covers_the_same_area() {
        // MSAA must soften the EDGE, not shrink or move the triangle. A resolve
        // wired to the wrong attachment can produce a plausible-looking blurry
        // image of the wrong thing.
        let Some(one) = edge(1) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let four = edge(4).expect("an adapter was already found");
        let (ax, ay) = centroid(&one);
        let (bx, by) = centroid(&four);
        assert!(
            (ax - bx).abs() < 0.5 && (ay - by).abs() < 0.5,
            "the multisampled triangle is in a different place: ({ax:.2}, {ay:.2}) vs ({bx:.2}, {by:.2})",
        );

        // And it is still there at all. A resolve pointed at the wrong
        // attachment gives an empty image, which the centroid alone would not
        // notice -- an empty image has a centroid too.
        let (a, b) = (mass(&one) as f64, mass(&four) as f64);
        assert!(
            b > a * 0.75 && b < a * 1.25,
            "the multisampled image is not the same triangle: mass {a} vs {b}",
        );
    }
}

// Scene shader sources, for `multiview::every_scene_shader_survives_the_multiview_transform`.
// Test-only: the gate has to see exactly the text each pipeline is built from,
// and nothing on a development machine can build a multiview pipeline to check.
#[cfg(test)]
pub fn solid_shader_src() -> String {
    solid_shader()
}
#[cfg(test)]
pub fn solid_ssr_shader_src() -> String {
    solid_ssr_shader()
}
#[cfg(test)]
pub fn wire_shader_src() -> String {
    WIRE_SHADER.to_string()
}
