use wgpu::*;

pub struct SceneTarget {
    _color_texture: Texture,
    /// The RESOLVED colour, single-sampled and sampleable. What the blit reads.
    pub color_view: TextureView,
    _depth_texture: Texture,
    /// Depth at the pass's own sample count. Read by SSR with `textureLoad`,
    /// which is why it cannot simply be discarded when multisampling.
    pub depth_view: TextureView,
    pub bind_group: BindGroup,
    /// The multisampled colour the scene pass actually draws into, when
    /// `samples > 1`. It resolves into `color_view` and is never stored, so on
    /// a tile GPU it lives and dies in tile memory.
    pub msaa_color_view: Option<TextureView>,
    _msaa_color_texture: Option<Texture>,
    pub samples: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsrCameraUniformData {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
}

pub struct SsrCameraUniform {
    buffer: Buffer,
    pub bind_group: BindGroup,
}

impl SsrCameraUniform {
    pub fn upload(&self, queue: &Queue, view_proj: glam::Mat4, camera_pos: glam::Vec3) {
        let data = SsrCameraUniformData {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.extend(0.0).into(),
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&data));
    }
}

pub struct SsrPipelines {
    pub blit_pipeline: RenderPipeline,
    scene_texture_layout: BindGroupLayout,
    camera_layout: BindGroupLayout,
    sampler: Sampler,
}

impl SsrPipelines {
    pub fn new(device: &Device, format: TextureFormat) -> Self {
        Self::new_with_depth_samples(device, format, 1)
    }

    /// `depth_samples` must match the scene target's depth texture.
    ///
    /// It exists because wgpu can resolve a colour attachment and cannot
    /// resolve a depth one: multisampling the scene pass leaves the depth
    /// multisampled, and every shader that reads it has to say so -- in the
    /// bind group layout AND in the WGSL type. A mismatch here fails pipeline
    /// creation rather than rendering wrongly, which is the good outcome.
    pub fn new_with_depth_samples(device: &Device, format: TextureFormat, depth_samples: u32) -> Self {
        let ms_depth = depth_samples > 1;
        let scene_texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssr_scene_texture_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: ms_depth,
                    },
                    count: None,
                },
            ],
        });

        let camera_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssr_camera_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("ssr_nearest_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..Default::default()
        });

        let blit_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ssr_blit_shader"),
            source: ShaderSource::Wgsl(blit_shader(ms_depth).into()),
        });
        let blit_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("ssr_blit_layout"),
            bind_group_layouts: &[&scene_texture_layout],
            push_constant_ranges: &[],
        });
        let blit_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("ssr_blit_pipeline"),
            layout: Some(&blit_layout),
            vertex: VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
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
                depth_write_enabled: true,
                depth_compare: CompareFunction::Always,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            blit_pipeline,
            scene_texture_layout,
            camera_layout,
            sampler,
        }
    }

    pub fn create_scene_target(&self, device: &Device, format: TextureFormat, width: u32, height: u32) -> SceneTarget {
        self.create_scene_target_multisampled(device, format, width, height, 1)
    }

    /// The scene target, optionally multisampled.
    ///
    /// Colour exists twice when `samples > 1`: the multisampled attachment the
    /// pass draws into, and the resolved single-sampled texture everything
    /// downstream reads. Depth exists once, at the pass's sample count, because
    /// wgpu can resolve colour and cannot resolve depth -- and SSR reads it.
    pub fn create_scene_target_multisampled(
        &self,
        device: &Device,
        format: TextureFormat,
        width: u32,
        height: u32,
        samples: u32,
    ) -> SceneTarget {
        let color_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssr_scene_color"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssr_scene_depth"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: samples,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssr_scene_texture_bg"),
            layout: &self.scene_texture_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&color_view) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&depth_view) },
            ],
        });

        let msaa_color_texture = (samples > 1).then(|| {
            device.create_texture(&TextureDescriptor {
                label: Some("ssr_scene_color_msaa"),
                size: Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: TextureDimension::D2,
                format,
                // NO texture binding: nothing samples it, which is what lets a
                // tile GPU keep it on chip and never write it out.
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        });
        let msaa_color_view = msaa_color_texture
            .as_ref()
            .map(|t| t.create_view(&TextureViewDescriptor::default()));

        SceneTarget {
            _color_texture: color_texture,
            color_view,
            _depth_texture: depth_texture,
            depth_view,
            bind_group,
            msaa_color_view,
            _msaa_color_texture: msaa_color_texture,
            samples,
        }
    }

    pub fn create_camera_uniform(&self, device: &Device) -> SsrCameraUniform {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ssr_camera_uniform"),
            size: std::mem::size_of::<SsrCameraUniformData>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssr_camera_bg"),
            layout: &self.camera_layout,
            entries: &[BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        SsrCameraUniform { buffer, bind_group }
    }

    pub fn scene_texture_layout(&self) -> &BindGroupLayout {
        &self.scene_texture_layout
    }

    pub fn camera_layout(&self) -> &BindGroupLayout {
        &self.camera_layout
    }

    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }
}

/// The blit that copies the resolved scene into the eye buffer.
///
/// It also writes `frag_depth` from the scene's depth, so the reflective solids
/// and the mirror quad drawn after it in the same pass depth-test against the
/// world. That read is the only reason a multisampled scene pass has to STORE
/// its depth rather than discard it -- see `XrRenderer`'s note on the cost.
///
/// Sample 0 of the multisampled depth, not an average: depth is not a quantity
/// you average. Averaging across a silhouette would place the composited
/// geometry at a distance where nothing is.
fn blit_shader(ms_depth: bool) -> String {
    let depth_ty = if ms_depth {
        "texture_depth_multisampled_2d"
    } else {
        "texture_depth_2d"
    };
    format!(
        r#"
@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var scene_depth: {depth_ty};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}}

struct FOut {{
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}}

@fragment
fn fs_main(@builtin(position) coord: vec4<f32>) -> FOut {{
    let px = vec2<i32>(coord.xy);
    var out: FOut;
    out.color = textureLoad(scene_color, px, 0);
    out.depth = textureLoad(scene_depth, px, 0);
    return out;
}}
"#
    )
}

pub fn wgsl_ssr_block(camera_group: u32, scene_group: u32, ms_depth: bool) -> String {
    let depth_ty = if ms_depth {
        "texture_depth_multisampled_2d"
    } else {
        "texture_depth_2d"
    };
    format!(
        r#"
struct SsrCamera {{
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
}}
@group({camera_group}) @binding(0) var<uniform> ssr_camera: SsrCamera;

@group({scene_group}) @binding(0) var ssr_scene_color: texture_2d<f32>;
@group({scene_group}) @binding(1) var ssr_scene_depth: {depth_ty};

const SSR_STEPS: u32 = 20u;
const SSR_STEP_SIZE: f32 = 0.12;
const SSR_STEP_GROWTH: f32 = 1.18;

fn ssr_reflect(world_pos: vec3<f32>, world_normal: vec3<f32>, base_color: vec3<f32>, reflectivity: f32) -> vec3<f32> {{
    if (reflectivity <= 0.0) {{
        return base_color;
    }}
    let n = normalize(world_normal);
    let view_dir = normalize(world_pos - ssr_camera.camera_pos.xyz);
    let refl_dir = reflect(view_dir, n);

    let scene_size = vec2<f32>(textureDimensions(ssr_scene_color));
    var step = SSR_STEP_SIZE;
    // Small bias along the reflection ray so the first sample isn't the
    // reflective surface's own texel (which would otherwise immediately
    // register as a "hit" on itself).
    var p = world_pos + refl_dir * step * 0.5;
    var hit = false;
    var hit_uv = vec2<f32>(0.0);

    for (var i = 0u; i < SSR_STEPS; i = i + 1u) {{
        p = p + refl_dir * step;
        step = step * SSR_STEP_GROWTH;

        let clip = ssr_camera.view_proj * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) {{
            break;
        }}
        let ndc = clip.xy / clip.w;
        if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) {{
            break;
        }}
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        let px = vec2<i32>(uv * scene_size);
        let scene_ndc_z = textureLoad(ssr_scene_depth, px, 0);
        let sample_ndc_z = clip.z / clip.w;
        if (scene_ndc_z < sample_ndc_z) {{
            hit = true;
            hit_uv = uv;
            break;
        }}
    }}

    if (!hit) {{
        return base_color;
    }}

    let hit_px = vec2<i32>(hit_uv * scene_size);
    let hit_color = textureLoad(ssr_scene_color, hit_px, 0).rgb;

    // Fade near the screen edge (the march ran off the sampled buffer just
    // outside the frame, not because there's really nothing there) and at
    // grazing angles (screen-space reflections are least reliable there).
    let edge = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
    let edge_fade = smoothstep(0.0, 0.08, edge);
    let grazing_fade = smoothstep(0.0, 0.2, abs(dot(refl_dir, -view_dir)));
    let fade = edge_fade * grazing_fade;

    return mix(base_color, hit_color, reflectivity * fade);
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headless_gpu() -> Option<(Device, Queue)> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: Features::empty(),
            required_limits: Limits::default(),
            ..Default::default()
        }))
        .ok()
    }

    #[test]
    fn ssr_pipelines_and_scene_target_build_on_a_real_device() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let format = TextureFormat::Rgba8UnormSrgb;

        let pipelines = SsrPipelines::new(&device, format);
        let _target = pipelines.create_scene_target(&device, format, 64, 64);
        let camera_uniform = pipelines.create_camera_uniform(&device);
        camera_uniform.upload(&queue, glam::Mat4::IDENTITY, glam::Vec3::ZERO);
    }
}
