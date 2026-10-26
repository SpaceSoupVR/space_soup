use wgpu::*;

pub const SSR_MIPS: u32 = 5;

pub const HI_Z_LEVELS: u32 = 8;

pub const REFLECTION_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

pub const SSR_DEBUG: bool = false;

pub struct SceneTarget {

    _color_texture: Option<Texture>,

    pub color_view: TextureView,

    pub sample_view: TextureView,

    pub mip_targets: Vec<TextureView>,

    pub mip_sources: Vec<BindGroup>,
    _depth_texture: Option<Texture>,

    pub depth_view: TextureView,

    pub resolved_depth_view: TextureView,

    pub hi_z_view: TextureView,

    hi_z_targets: Vec<TextureView>,

    hi_z_sources: Vec<BindGroup>,
    resolved_depth_texture: Texture,

    resolve_src: Option<BindGroup>,

    pub blit_bind_group: BindGroup,
    pub bind_group: BindGroup,

    pub msaa_color_view: Option<TextureView>,
    _msaa_color_texture: Option<Texture>,
    pub samples: u32,
}

pub fn msaa_colour_usage(layers: u32) -> TextureUsages {
    if layers == 1 {
        TextureUsages::RENDER_ATTACHMENT | TextureUsages::TRANSIENT_ATTACHMENT
    } else {
        TextureUsages::RENDER_ATTACHMENT
    }
}

pub struct StereoSceneTextures {
    _color: Texture,
    _depth: Texture,
    _msaa_color: Option<Texture>,

    pub array_color_view: TextureView,
    pub array_depth_view: TextureView,

    pub array_msaa_color_view: Option<TextureView>,
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

    resolve_pipeline: Option<RenderPipeline>,
    resolve_layout: BindGroupLayout,

    hi_z_pipeline: RenderPipeline,

    reflection_layout: BindGroupLayout,
    reflection_resolve_pipeline: RenderPipeline,

    reflection_resolve_wide_pipeline: RenderPipeline,
    hi_z_layout: BindGroupLayout,
    blit_layout: BindGroupLayout,

    downsample_pipeline: RenderPipeline,
    downsample_layout: BindGroupLayout,
    linear_sampler: Sampler,
    scene_texture_layout: BindGroupLayout,
    camera_layout: BindGroupLayout,
    sampler: Sampler,
}

pub const REFLECTION_SCALE: u32 = 2;

pub const BUFFERED_REFLECTIONS: bool = false;

pub const REFLECTION_FILTER_RADIUS: i32 = 2;

pub const REFLECTION_RESOLVE_STRIDES: [i32; 2] = [1, 2];

pub struct ReflectionTarget {
    _trace_texture: Texture,
    _filtered_texture: Texture,

    _scratch_texture: Texture,
    _depth_texture: Texture,

    pub trace_view: TextureView,

    pub depth_view: TextureView,

    pub filtered_view: TextureView,

    pub scratch_view: TextureView,

    pub scratch_src: BindGroup,

    pub resolve_src: BindGroup,

    pub composite_src: BindGroup,
    pub width: u32,
    pub height: u32,
}

impl StereoSceneTextures {

    fn new(
        device: &Device,
        format: TextureFormat,
        width: u32,
        height: u32,
        samples: u32,
    ) -> Self {
        let mips = crate::renderer::terrain_pipeline::mip_levels_for(width, height)
            .min(SSR_MIPS)
            .max(1);
        let layers = crate::renderer::multiview::STEREO_VIEWS;
        let color = device.create_texture(&TextureDescriptor {
            label: Some("ssr_scene_color_stereo"),
            size: Extent3d { width, height, depth_or_array_layers: layers },
            mip_level_count: mips,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth = device.create_texture(&TextureDescriptor {
            label: Some("ssr_scene_depth_stereo"),
            size: Extent3d { width, height, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: samples,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let msaa_color = (samples > 1).then(|| {
            device.create_texture(&TextureDescriptor {
                label: Some("ssr_scene_color_msaa_stereo"),
                size: Extent3d { width, height, depth_or_array_layers: layers },
                mip_level_count: 1,
                sample_count: samples,
                dimension: TextureDimension::D2,
                format,

                usage: msaa_colour_usage(layers),
                view_formats: &[],
            })
        });

        let array_view = |t: &Texture| {
            t.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(layers),
                ..Default::default()
            })
        };
        Self {
            array_color_view: array_view(&color),
            array_depth_view: array_view(&depth),
            array_msaa_color_view: msaa_color.as_ref().map(array_view),
            _color: color,
            _depth: depth,
            _msaa_color: msaa_color,
        }
    }
}

impl SceneTarget {

    #[cfg(test)]
    pub(crate) fn resolved_depth_texture(&self) -> &Texture {
        &self.resolved_depth_texture
    }

    #[cfg(test)]
    pub(crate) fn hi_z_level_count(&self) -> u32 {
        self.hi_z_targets.len() as u32 + 1
    }
}

impl SsrPipelines {

    pub fn create_reflection_target(
        &self,
        device: &Device,
        width: u32,
        height: u32,
    ) -> ReflectionTarget {
        let w = (width / REFLECTION_SCALE).max(1);
        let h = (height / REFLECTION_SCALE).max(1);
        let colour = |label| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: REFLECTION_FORMAT,

                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let trace_texture = colour("ssr_reflection_trace");
        let filtered_texture = colour("ssr_reflection_filtered");
        let scratch_texture = colour("ssr_reflection_scratch");

        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssr_reflection_depth"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let trace_view = trace_texture.create_view(&TextureViewDescriptor::default());
        let filtered_view = filtered_texture.create_view(&TextureViewDescriptor::default());
        let scratch_view = scratch_texture.create_view(&TextureViewDescriptor::default());
        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());
        let src_bg = |label, view: &TextureView| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some(label),
                layout: &self.reflection_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(view),
                }],
            })
        };
        ReflectionTarget {
            resolve_src: src_bg("ssr_reflection_resolve_src", &trace_view),
            composite_src: src_bg("ssr_reflection_composite_src", &filtered_view),
            scratch_src: src_bg("ssr_reflection_scratch_src", &scratch_view),
            trace_view,
            filtered_view,
            scratch_view,
            depth_view,
            _trace_texture: trace_texture,
            _filtered_texture: filtered_texture,
            _scratch_texture: scratch_texture,
            _depth_texture: depth_texture,
            width: w,
            height: h,
        }
    }

    pub fn create_composite_bind_group(
        &self,
        device: &Device,
        scene: &SceneTarget,
        reflection: &ReflectionTarget,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssr_composite_bg"),
            layout: &self.scene_texture_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&scene.sample_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&reflection.filtered_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&self.linear_sampler),
                },
            ],
        })
    }

    pub fn reflection_layout(&self) -> &BindGroupLayout {
        &self.reflection_layout
    }

    pub fn resolve_reflections(
        &self,
        encoder: &mut CommandEncoder,
        target: &ReflectionTarget,
        timestamps: Option<RenderPassTimestampWrites<'_>>,
    ) {

        self.resolve_pass(
            encoder,
            &self.reflection_resolve_pipeline,
            &target.resolve_src,
            &target.scratch_view,
            &target.depth_view,
            timestamps,
        );
        self.resolve_pass(
            encoder,
            &self.reflection_resolve_wide_pipeline,
            &target.scratch_src,
            &target.filtered_view,
            &target.depth_view,
            None,
        );
    }

    fn resolve_pass(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: &RenderPipeline,
        src: &BindGroup,
        dst: &TextureView,
        depth: &TextureView,
        timestamps: Option<RenderPassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("ssr_reflection_resolve"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: Operations { load: LoadOp::Clear(Color::TRANSPARENT), store: StoreOp::Store },
            })],

            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: timestamps,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, src, &[]);
        pass.draw(0..3, 0..1);
    }

    pub fn new(device: &Device, format: TextureFormat) -> Self {
        Self::new_with_depth_samples(device, format, 1)
    }

    pub fn new_with_depth_samples(device: &Device, format: TextureFormat, depth_samples: u32) -> Self {

        let scene_texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssr_scene_texture_bgl"),
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
                    ty: BindingType::Texture {

                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
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

        let linear_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("ssr_linear_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,

            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });
        let blit_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssr_blit_layout"),
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
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: depth_samples > 1,
                    },
                    count: None,
                },
            ],
        });

        let resolve_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssr_depth_resolve_layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Depth,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: depth_samples > 1,
                },
                count: None,
            }],
        });
        let resolve_pipeline = Some({
            let shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("ssr_depth_resolve_shader"),
                source: ShaderSource::Wgsl(depth_resolve_shader(depth_samples).into()),
            });
            let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssr_depth_resolve_pipeline_layout"),
                bind_group_layouts: &[Some(&resolve_layout)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("ssr_depth_resolve_pipeline"),
                layout: Some(&layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::R32Float,
                        blend: None,
                        write_mask: ColorWrites::RED,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        });

        let hi_z_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssr_hi_z_layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let reflection_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssr_reflection_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {

                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let mut reflection_resolve_pipelines = REFLECTION_RESOLVE_STRIDES.iter().map(|&stride| {
            let shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("ssr_reflection_resolve_shader"),
                source: ShaderSource::Wgsl(reflection_resolve_shader(stride).into()),
            });
            let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssr_reflection_resolve_layout"),
                bind_group_layouts: &[Some(&reflection_layout)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("ssr_reflection_resolve_pipeline"),
                layout: Some(&layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: REFLECTION_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: PrimitiveState::default(),

                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(CompareFunction::Greater),
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        });
        let reflection_resolve_pipeline = reflection_resolve_pipelines
            .next()
            .expect("REFLECTION_RESOLVE_STRIDES is empty");
        let reflection_resolve_wide_pipeline = reflection_resolve_pipelines
            .next()
            .expect("REFLECTION_RESOLVE_STRIDES needs a second pass");
        let hi_z_pipeline = {
            let shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("ssr_hi_z_shader"),
                source: ShaderSource::Wgsl(hi_z_reduce_shader().into()),
            });
            let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ssr_hi_z_pipeline_layout"),
                bind_group_layouts: &[Some(&hi_z_layout)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("ssr_hi_z_pipeline"),
                layout: Some(&layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::R32Float,
                        blend: None,
                        write_mask: ColorWrites::RED,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let downsample_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssr_downsample_layout"),
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
            source: ShaderSource::Wgsl(blit_shader(depth_samples > 1).into()),
        });
        let blit_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("ssr_blit_pipeline_layout"),
            bind_group_layouts: &[Some(&blit_layout)],
            immediate_size: 0,
        });
        let downsample_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ssr_downsample_shader"),
            source: ShaderSource::Wgsl(downsample_shader().into()),
        });
        let downsample_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("ssr_downsample_pipeline_layout"),
            bind_group_layouts: &[Some(&downsample_layout)],
            immediate_size: 0,
        });
        let downsample_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("ssr_downsample_pipeline"),
            layout: Some(&downsample_pipeline_layout),
            vertex: VertexState {
                module: &downsample_shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &downsample_shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let blit_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("ssr_blit_pipeline"),
            layout: Some(&blit_pipeline_layout),
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
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            blit_pipeline,
            resolve_pipeline,
            resolve_layout,
            hi_z_pipeline,
            reflection_layout,
            reflection_resolve_pipeline,
            reflection_resolve_wide_pipeline,
            hi_z_layout,
            blit_layout,
            downsample_pipeline,
            downsample_layout,
            linear_sampler,
            scene_texture_layout,
            camera_layout,
            sampler,
        }
    }

    pub fn create_scene_target(&self, device: &Device, format: TextureFormat, width: u32, height: u32) -> SceneTarget {
        self.create_scene_target_multisampled(device, format, width, height, 1)
    }

    pub fn create_scene_target_multisampled(
        &self,
        device: &Device,
        format: TextureFormat,
        width: u32,
        height: u32,
        samples: u32,
    ) -> SceneTarget {
        self.create_scene_target_layer(device, format, width, height, samples, None)
    }

    pub fn create_scene_targets_stereo(
        &self,
        device: &Device,
        format: TextureFormat,
        width: u32,
        height: u32,
        samples: u32,
    ) -> (StereoSceneTextures, [SceneTarget; 2]) {
        let stereo = StereoSceneTextures::new(device, format, width, height, samples);
        let targets = std::array::from_fn(|eye| {
            self.create_scene_target_layer(
                device,
                format,
                width,
                height,
                samples,
                Some((&stereo, eye as u32)),
            )
        });
        (stereo, targets)
    }

    fn create_scene_target_layer(
        &self,
        device: &Device,
        format: TextureFormat,
        width: u32,
        height: u32,
        samples: u32,
        shared: Option<(&StereoSceneTextures, u32)>,
    ) -> SceneTarget {

        let layer = shared.map_or(0, |(_, eye)| eye);

        let mips = crate::renderer::terrain_pipeline::mip_levels_for(width, height)
            .min(SSR_MIPS)
            .max(1);
        let owned_color = shared.is_none().then(|| {
            device.create_texture(&TextureDescriptor {
                label: Some("ssr_scene_color"),
                size: Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: mips,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        });
        let color_texture = owned_color
            .as_ref()
            .unwrap_or_else(|| &shared.unwrap().0._color);
        let level_view = |base: u32| {
            color_texture.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2),
                base_mip_level: base,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        };
        let color_view = level_view(0);
        let sample_view = color_texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });
        let mip_targets: Vec<TextureView> = (1..mips).map(level_view).collect();
        let mip_sources: Vec<BindGroup> = (1..mips)
            .map(|level| {
                let src = color_texture.create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2),
                    base_mip_level: level - 1,
                    mip_level_count: Some(1),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("ssr_downsample_src"),
                    layout: &self.downsample_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&src) },
                        BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&self.linear_sampler) },
                    ],
                })
            })
            .collect();

        let owned_depth = shared.is_none().then(|| {
            device.create_texture(&TextureDescriptor {
                label: Some("ssr_scene_depth"),
                size: Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        });
        let depth_texture = owned_depth
            .as_ref()
            .unwrap_or_else(|| &shared.unwrap().0._depth);
        let depth_view = depth_texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });

        let hi_z_levels = crate::renderer::terrain_pipeline::mip_levels_for(width, height)
            .min(HI_Z_LEVELS)
            .max(1);
        let resolved_depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("ssr_scene_depth_resolved"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: hi_z_levels,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,

            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let depth_level_view = |base: u32| {
            resolved_depth_texture.create_view(&TextureViewDescriptor {
                base_mip_level: base,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let resolved_depth_view = depth_level_view(0);
        let hi_z_view = resolved_depth_texture.create_view(&TextureViewDescriptor::default());
        let hi_z_targets: Vec<TextureView> = (1..hi_z_levels).map(depth_level_view).collect();
        let hi_z_sources: Vec<BindGroup> = (1..hi_z_levels)
            .map(|level| {
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("ssr_hi_z_src"),
                    layout: &self.hi_z_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&depth_level_view(level - 1)),
                    }],
                })
            })
            .collect();

        let resolve_src = Some(device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssr_depth_resolve_src"),
            layout: &self.resolve_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&depth_view),
            }],
        }));

        let blit_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssr_blit_bg"),
            layout: &self.blit_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&color_view) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&depth_view) },
            ],
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("ssr_scene_texture_bg"),
            layout: &self.scene_texture_layout,
            entries: &[

                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&sample_view) },

                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&hi_z_view) },
                BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&self.linear_sampler) },
            ],
        });

        let owned_msaa = (shared.is_none() && samples > 1).then(|| {
            device.create_texture(&TextureDescriptor {
                label: Some("ssr_scene_color_msaa"),
                size: Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: TextureDimension::D2,
                format,

                usage: msaa_colour_usage(1),
                view_formats: &[],
            })
        });
        let msaa_color_view = owned_msaa
            .as_ref()
            .or(shared.and_then(|(st, _)| st._msaa_color.as_ref()))
            .map(|t| {
                t.create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            });

        SceneTarget {
            _color_texture: owned_color,
            color_view,
            sample_view,
            mip_targets,
            mip_sources,
            _depth_texture: owned_depth,
            depth_view,
            resolved_depth_view,
            hi_z_view,
            hi_z_targets,
            hi_z_sources,
            resolved_depth_texture,
            resolve_src,
            blit_bind_group,
            bind_group,
            msaa_color_view,
            _msaa_color_texture: owned_msaa,
            samples,
        }
    }

    pub fn resolve_depth(
        &self,
        encoder: &mut CommandEncoder,
        target: &SceneTarget,
        timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let (Some(pipeline), Some(src)) = (&self.resolve_pipeline, &target.resolve_src) else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("ssr_depth_resolve"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &target.resolved_depth_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {

                    load: LoadOp::Clear(Color::WHITE),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: timestamps,
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, src, &[]);
        pass.draw(0..3, 0..1);
    }

    pub fn build_hi_z(
        &self,
        encoder: &mut CommandEncoder,
        target: &SceneTarget,
        last_timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let last = target.hi_z_targets.len().saturating_sub(1);
        let mut last_timestamps = last_timestamps;
        for (i, view) in target.hi_z_targets.iter().enumerate() {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("ssr_hi_z"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {

                        load: LoadOp::Clear(Color::WHITE),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,

                timestamp_writes: if i == last { last_timestamps.take() } else { None },
                ..Default::default()
            });
            pass.set_pipeline(&self.hi_z_pipeline);
            pass.set_bind_group(0, &target.hi_z_sources[i], &[]);
            pass.draw(0..3, 0..1);
        }
    }

    pub fn generate_mips(&self, encoder: &mut CommandEncoder, target: &SceneTarget) {
        for (i, view) in target.mip_targets.iter().enumerate() {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("ssr_downsample"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {

                        load: LoadOp::Clear(Color::TRANSPARENT),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, &target.mip_sources[i], &[]);
            pass.draw(0..3, 0..1);
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

fn reflection_resolve_shader(stride: i32) -> String {
    format!(
        r"
@group(0) @binding(0) var refl_src: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );

    return vec4<f32>(pos[vi], 1.0, 1.0);
}}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {{
    let size = vec2<i32>(textureDimensions(refl_src));
    let here = vec2<i32>(frag.xy);
    let centre = textureLoad(refl_src, here, 0);

    var sum = centre.rgb * centre.a * {centre_weight:.1};
    var weight = centre.a * {centre_weight:.1};

    for (var dy = -{radius}; dy <= {radius}; dy = dy + 1) {{
        for (var dx = -{radius}; dx <= {radius}; dx = dx + 1) {{
            if (dx == 0 && dy == 0) {{
                continue;
            }}

            let at = here + vec2<i32>(dx, dy) * {stride};

            let p = clamp(at, vec2<i32>(0), size - vec2<i32>(1));
            let n = textureLoad(refl_src, p, 0);

            let d = f32(dx * dx + dy * dy);
            let falloff = 1.0 / (1.0 + d);
            let w = n.a * falloff;
            sum = sum + n.rgb * w;
            weight = weight + w;
        }}
    }}

    if (weight <= 1e-5) {{

        return vec4<f32>(0.0);
    }}

    let taps = {taps:.1};
    return vec4<f32>(sum / weight, clamp(max(centre.a, weight / taps), 0.0, 1.0));
}}
",
        radius = REFLECTION_FILTER_RADIUS,
        stride = stride,
        centre_weight = REFLECTION_CENTRE_WEIGHT,

        taps = reflection_filter_total_weight(),
    )
}

const REFLECTION_CENTRE_WEIGHT: f32 = 4.0;

fn reflection_filter_total_weight() -> f32 {
    let r = REFLECTION_FILTER_RADIUS;
    let mut total = REFLECTION_CENTRE_WEIGHT;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx == 0 && dy == 0 {
                continue;
            }
            total += 1.0 / (1.0 + (dx * dx + dy * dy) as f32);
        }
    }
    total
}

fn hi_z_reduce_shader() -> String {{
    r"
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}}

fn tap(px: vec2<i32>, limit: vec2<i32>) -> f32 {{
    return textureLoad(src, clamp(px, vec2<i32>(0), limit), 0).x;
}}

@fragment
fn fs_main(@builtin(position) coord: vec4<f32>) -> @location(0) f32 {{
    let size = vec2<i32>(textureDimensions(src));
    let limit = size - vec2<i32>(1);
    let base = vec2<i32>(coord.xy) * 2;
    var m = tap(base, limit);
    m = min(m, tap(base + vec2<i32>(1, 0), limit));
    m = min(m, tap(base + vec2<i32>(0, 1), limit));
    m = min(m, tap(base + vec2<i32>(1, 1), limit));
    let odd = size % vec2<i32>(2);
    if (odd.x == 1) {{
        m = min(m, tap(base + vec2<i32>(2, 0), limit));
        m = min(m, tap(base + vec2<i32>(2, 1), limit));
    }}
    if (odd.y == 1) {{
        m = min(m, tap(base + vec2<i32>(0, 2), limit));
        m = min(m, tap(base + vec2<i32>(1, 2), limit));
    }}
    if (odd.x == 1 && odd.y == 1) {{
        m = min(m, tap(base + vec2<i32>(2, 2), limit));
    }}
    return m;
}}
"
    .to_string()
}}

fn depth_resolve_shader(samples: u32) -> String {{
    let src_ty = if samples > 1 {{
        "texture_depth_multisampled_2d"
    }} else {{
        "texture_depth_2d"
    }};
    format!(
        r"
@group(0) @binding(0) var ms_depth: {src_ty};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{{{
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}}}}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) f32 {{{{
    let px = vec2<i32>(frag.xy);
    var nearest = textureLoad(ms_depth, px, 0);
    for (var i = 1; i < {samples}; i = i + 1) {{{{
        nearest = min(nearest, textureLoad(ms_depth, px, i));
    }}}}
    return nearest;
}}}}
"
    )
}}

#[cfg(test)]
mod composite_block_tests {
    use super::*;

    #[test]
    fn the_two_blocks_agree_on_the_reflectivity_cap() {
        let march = wgsl_ssr_block_shared_camera_debug(3, false);
        let composite = wgsl_ssr_composite_block(3);
        let line = format!("const MAX_SSR_REFLECTIVITY: f32 = {MAX_SSR_REFLECTIVITY_VALUE:?};");
        assert!(
            march.contains(&line),
            "the march block's reflectivity cap is no longer \
             {MAX_SSR_REFLECTIVITY_VALUE:?}; the composite would silently keep \
             the old value and the two passes would disagree about how strong a \
             reflection is",
        );
        assert!(composite.contains(&line), "the composite lost its cap declaration");
    }

    #[test]
    fn the_composite_upsamples_by_confidence_and_not_by_a_sampler() {
        let composite = wgsl_ssr_composite_block(3);
        assert!(
            composite.contains("textureLoad(ssr_reflection, p, 0)"),
            "the composite is no longer loading exact texels:\n{composite}",
        );
        assert!(
            !composite.contains("textureSample(ssr_reflection")
                && !composite.contains("textureSampleLevel(ssr_reflection"),
            "the composite samples the reflection with a filter, which blends \
             confidence and radiance independently and invents reflections at \
             the hit/miss boundary",
        );
        assert!(
            composite.contains("let w = t.a * bw;"),
            "each tap is no longer weighted by its own confidence, so an empty \
             neighbour darkens the reflection instead of contributing nothing",
        );

        assert!(
            composite.contains(&format!("clip_xy / {REFLECTION_SCALE}.0")),
            "the composite does not divide by REFLECTION_SCALE ({REFLECTION_SCALE}), \
             so it reads the reflection buffer at the wrong scale",
        );
    }

    #[test]
    fn the_composite_replaces_the_depth_binding_and_not_the_colour() {
        let composite = wgsl_ssr_composite_block(3);
        assert!(
            composite.contains("@group(3) @binding(0) var ssr_scene_color: texture_2d<f32>;"),
            "the composite must still read the scene colour for the fallback",
        );
        assert!(
            composite.contains("@group(3) @binding(1) var ssr_reflection: texture_2d<f32>;"),
            "the filtered reflection must sit in binding 1, where the march's \
             depth pyramid was, or the bind group layout changes and the \
             four-group limit is exceeded",
        );
        assert!(
            !composite.contains("ssr_scene_depth"),
            "the composite does not march and must not declare the pyramid",
        );
    }
}

#[cfg(test)]
mod reflection_resolve_tests {
    use super::tests::headless_gpu;
    use super::*;

    fn prime_depth(
        device: &Device,
        encoder: &mut CommandEncoder,
        target: &ReflectionTarget,
        value: f32,
    ) {
        let _ = device;
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("prime_depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(value),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    fn f16_bits(v: f32) -> u16 {

        match v {
            x if x == 0.0 => 0x0000,
            x if x == 1.0 => 0x3C00,
            other => panic!("f16_bits only encodes 0.0 and 1.0, got {other}"),
        }
    }

    fn f16_to_f32(bits: u16) -> f32 {
        let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
        let exp = i32::from((bits >> 10) & 0x1f);
        let mant = f32::from(bits & 0x3ff);
        let mag = if exp == 0 {
            mant * 2f32.powi(-24)
        } else if exp == 31 {
            f32::INFINITY
        } else {
            (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
        };
        sign * mag
    }

    #[test]
    fn the_resolve_divides_by_the_weight_it_can_actually_reach() {
        let r = REFLECTION_FILTER_RADIUS;
        let mut by_hand = REFLECTION_CENTRE_WEIGHT;
        for dy in -r..=r {
            for dx in -r..=r {
                if (dx, dy) != (0, 0) {
                    by_hand += 1.0 / (1.0 + f64::from(dx * dx + dy * dy) as f32);
                }
            }
        }
        assert!(
            (reflection_filter_total_weight() - by_hand).abs() < 1e-5,
            "total weight {} does not match the ring the shader walks ({by_hand})",
            reflection_filter_total_weight(),
        );
        let src = reflection_resolve_shader(REFLECTION_RESOLVE_STRIDES[0]);
        assert!(
            src.contains(&format!("let taps = {by_hand:.1};")),
            "the shader divides by something other than the weight it sums:\n{src}",
        );
    }

    #[test]
    fn the_trace_pass_shape_records_and_submits() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let ssr = SsrPipelines::new(&device, TextureFormat::Rgba8UnormSrgb);
        let target = ssr.create_reflection_target(&device, 64, 64);
        let err_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder =
            device.create_command_encoder(&CommandEncoderDescriptor { label: Some("trace_shape") });
        {
            let _pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("ssr_reflection_trace_probe"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target.trace_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::TRANSPARENT),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        queue.submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        if let Some(err) = pollster::block_on(err_scope.pop()) {
            panic!("the reflection trace pass does not record: {err}");
        }
        assert_eq!(
            (target.width, target.height),
            (64 / REFLECTION_SCALE, 64 / REFLECTION_SCALE),
            "the reflection buffer is not at REFLECTION_SCALE of the frame",
        );
    }

    #[test]
    fn a_confident_pixel_next_to_a_miss_keeps_its_full_strength() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        const W: u32 = 16;
        const H: u32 = 4;
        let ssr = SsrPipelines::new(&device, TextureFormat::Rgba8UnormSrgb);
        let target =
            ssr.create_reflection_target(&device, W * REFLECTION_SCALE, H * REFLECTION_SCALE);
        let half = W / 2;
        let v = |f: f32| f16_bits(f);
        let mut texels: Vec<[u16; 4]> = Vec::new();
        for _ in 0..H {
            for x in 0..W {
                texels.push(if x < half {
                    [v(1.0), v(0.0), v(0.0), v(1.0)]
                } else {
                    [v(0.0), v(0.0), v(0.0), v(0.0)]
                });
            }
        }
        let bytes: Vec<u8> = texels.iter().flatten().flat_map(|b| b.to_le_bytes()).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &target._trace_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(W * 8),
                rows_per_image: Some(H),
            },
            Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );
        let mut encoder =
            device.create_command_encoder(&CommandEncoderDescriptor { label: Some("mute_test") });
        prime_depth(&device, &mut encoder, &target, 0.0);
        ssr.resolve_reflections(&mut encoder, &target, None);
        let row = (W * 8).next_multiple_of(256);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mute_readback"),
            size: u64::from(row * H),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target._filtered_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(H),
                },
            },
            Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = readback.slice(..).get_mapped_range().unwrap().to_vec();
        let at = |x: u32, y: u32| -> [f32; 4] {
            let o = (y * row + x * 8) as usize;
            std::array::from_fn(|i| {
                f16_to_f32(u16::from_le_bytes([data[o + i * 2], data[o + i * 2 + 1]]))
            })
        };

        let last_hit = at(half - 1, H / 2);
        println!("last confident pixel before the gap: {last_hit:?}");
        assert!(
            last_hit[3] > 0.99,
            "a pixel whose own ray hit came back at confidence {:.3} because \
             its neighbours missed -- that is the whole reflection being muted, \
             not a boundary being smoothed",
            last_hit[3],
        );
    }

    #[test]
    fn the_resolve_skips_pixels_the_trace_never_drew() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        const W: u32 = 16;
        const H: u32 = 4;
        let ssr = SsrPipelines::new(&device, TextureFormat::Rgba8UnormSrgb);
        let target =
            ssr.create_reflection_target(&device, W * REFLECTION_SCALE, H * REFLECTION_SCALE);

        let v = |f: f32| f16_bits(f);
        let texels: Vec<[u16; 4]> =
            (0..W * H).map(|_| [v(1.0), v(0.0), v(0.0), v(1.0)]).collect();
        let bytes: Vec<u8> = texels.iter().flatten().flat_map(|b| b.to_le_bytes()).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &target._trace_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(W * 8),
                rows_per_image: Some(H),
            },
            Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );

        let mut encoder = device
            .create_command_encoder(&CommandEncoderDescriptor { label: Some("gating_test") });

        prime_depth(&device, &mut encoder, &target, 1.0);
        ssr.resolve_reflections(&mut encoder, &target, None);
        let row = (W * 8).next_multiple_of(256);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gating_readback"),
            size: u64::from(row * H),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target._filtered_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(H),
                },
            },
            Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = readback.slice(..).get_mapped_range().unwrap().to_vec();
        for y in 0..H {
            for x in 0..W {
                let o = (y * row + x * 8) as usize;
                let px: [f32; 4] = std::array::from_fn(|i| {
                    f16_to_f32(u16::from_le_bytes([data[o + i * 2], data[o + i * 2 + 1]]))
                });
                assert!(
                    px.iter().all(|c| *c == 0.0),
                    "({x},{y}) came back {px:?} although the trace drew no \
                     geometry anywhere. The depth gate is not rejecting, so the \
                     resolve is still filtering the whole frame -- which is the \
                     7 to 17 ms an eye this exists to remove",
                );
            }
        }
    }

    #[test]
    fn the_resolve_fills_a_miss_from_its_confident_neighbours() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        const W: u32 = 16;
        const H: u32 = 4;
        let ssr = SsrPipelines::new(&device, TextureFormat::Rgba8UnormSrgb);
        let target = ssr.create_reflection_target(&device, W * REFLECTION_SCALE, H * REFLECTION_SCALE);

        let half = W / 2;
        let mut texels: Vec<[u16; 4]> = Vec::with_capacity((W * H) as usize);
        for _ in 0..H {
            for x in 0..W {
                let v = |f: f32| f16_bits(f);
                texels.push(if x < half {
                    [v(1.0), v(0.0), v(0.0), v(1.0)]
                } else {
                    [v(0.0), v(0.0), v(0.0), v(0.0)]
                });
            }
        }
        let bytes: Vec<u8> = texels.iter().flatten().flat_map(|b| b.to_le_bytes()).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &target._trace_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(W * 8),
                rows_per_image: Some(H),
            },
            Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );

        let err_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder =
            device.create_command_encoder(&CommandEncoderDescriptor { label: Some("resolve_test") });

        prime_depth(&device, &mut encoder, &target, 0.0);
        ssr.resolve_reflections(&mut encoder, &target, None);

        let row = (W * 8).next_multiple_of(256);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resolve_readback"),
            size: u64::from(row * H),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target._filtered_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(H),
                },
            },
            Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        if let Some(err) = pollster::block_on(err_scope.pop()) {
            panic!("the reflection resolve does not validate: {err}");
        }
        let data = readback.slice(..).get_mapped_range().unwrap().to_vec();
        let at = |x: u32, y: u32| -> [f32; 4] {
            let o = (y * row + x * 8) as usize;
            std::array::from_fn(|i| {
                f16_to_f32(u16::from_le_bytes([data[o + i * 2], data[o + i * 2 + 1]]))
            })
        };

        let deep_inside = at(1, H / 2);
        let just_outside = at(half, H / 2);
        let far_outside = at(W - 1, H / 2);

        assert!(
            deep_inside[3] > 0.9 && deep_inside[0] > 0.9,
            "a pixel surrounded by confident hits lost its reflection: {deep_inside:?}",
        );
        assert!(
            just_outside[3] > 0.05 && just_outside[0] > 0.05,
            "THE CLIFF IS STILL THERE: the pixel just past the boundary got \
             {just_outside:?} when its neighbours are fully confident red. \
             This is the comb, and filling it is the entire purpose of this pass",
        );
        assert!(
            just_outside[3] < deep_inside[3],
            "the boundary pixel is as confident as one surrounded by hits \
             ({just_outside:?} vs {deep_inside:?}), so the reflection will \
              spread past the surface that produced it",
        );
        assert!(
            far_outside[3] < 0.05,
            "a pixel far from any hit invented a reflection: {far_outside:?}",
        );

        let four_in = at(half + 3, H / 2);
        println!("four texels into the gap: {four_in:?}");
        assert!(
            four_in[3] > 0.02,
            "a pixel four texels from the nearest hit came back {four_in:?}. \
             One radius-2 pass cannot reach it; if this is zero the second, \
             wider pass is not running or its stride is 1",
        );
        assert!(
            four_in[3] < just_outside[3],
            "the fill is not falling off with distance ({four_in:?} at four \
             texels against {just_outside:?} at the boundary), so the \
             reflection will spread flat across the whole gap",
        );
        println!(
            "resolve: inside a={:.3}, boundary a={:.3} r={:.3}, far a={:.3}",
            deep_inside[3], just_outside[3], just_outside[0], far_outside[3],
        );
    }
}

#[cfg(test)]
mod stereo_target_tests {
    use super::tests::headless_gpu;
    use super::*;

    #[test]
    fn the_two_eyes_are_layers_of_one_texture_and_each_sees_its_own() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let format = TextureFormat::Rgba8UnormSrgb;
        let ssr = SsrPipelines::new(&device, format);
        let err_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let (stereo, targets) = ssr.create_scene_targets_stereo(&device, format, 64, 64, 1);

        let mut encoder =
            device.create_command_encoder(&CommandEncoderDescriptor { label: Some("stereo_test") });
        for (eye, target) in targets.iter().enumerate() {
            encoder
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("stereo_layer_clear"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &target.color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color {
                                r: if eye == 0 { 1.0 } else { 0.0 },
                                g: 0.0,
                                b: if eye == 0 { 0.0 } else { 1.0 },
                                a: 1.0,
                            }),
                            store: StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                    occlusion_query_set: None,
                })
                .set_viewport(0.0, 0.0, 64.0, 64.0, 0.0, 1.0);
        }
        queue.submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        if let Some(err) = pollster::block_on(err_scope.pop()) {
            panic!("the stereo scene target does not validate: {err}");
        }

        assert!(
            stereo.array_msaa_color_view.is_none(),
            "there is no multisampled colour at one sample",
        );
        assert_eq!(targets.len(), 2, "one view bundle per eye");

        for (eye, target) in targets.iter().enumerate() {
            assert!(
                target._color_texture.is_none() && target._depth_texture.is_none(),
                "eye {eye} owns its own colour or depth, so the two eyes are \
                 not layers of ONE texture and a multiview pass cannot draw \
                 both at once",
            );
        }
    }

    #[test]
    fn a_multisampled_array_is_refused_without_the_feature() {
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        if device.features().contains(Features::MULTISAMPLE_ARRAY) {
            eprintln!("skipping: this adapter has MULTISAMPLE_ARRAY");
            return;
        }
        let err_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _ = device.create_texture(&TextureDescriptor {
            label: Some("stereo_msaa_probe"),
            size: Extent3d { width: 64, height: 64, depth_or_array_layers: 2 },
            mip_level_count: 1,
            sample_count: 4,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let err = pollster::block_on(err_scope.pop());
        let msg = err.map(|e| e.to_string()).unwrap_or_default();
        assert!(
            msg.contains("Multisampled texture depth or array layers must be 1"),
            "a 2-layer 4x texture was accepted without MULTISAMPLE_ARRAY, or \
             refused for a different reason: {msg:?}. This is the restriction \
             the whole wgpu 25 -> 28 move was for; if it has gone, the feature \
             claim in `vulkan_interop` may no longer be needed.",
        );
    }
}

#[cfg(test)]
mod depth_resolve_tests {
    
    use super::*;
    

    #[test]
    fn the_resolve_takes_the_nearest_of_every_sample() {
        for samples in [2u32, 4] {
            let code = depth_resolve_shader(samples);
            assert!(
                code.contains(&format!("for (var i = 1; i < {samples}; i = i + 1)")),
                "the resolve at {samples}x does not walk all of its samples",
            );
            assert!(
                code.contains("nearest = min(nearest, textureLoad(ms_depth, px, i));"),
                "the resolve is no longer taking the NEAREST sample; an edge \
                 pixel will report a surface that covers a quarter of it",
            );
            assert!(
                code.contains("-> @location(0) f32"),
                "the resolve is writing somewhere other than a single-channel \
                 colour target; a depth-only pass validates, records and \
                 submits on Metal and crashed the Quest's VR driver on startup",
            );
        }
    }

    #[test]
    fn the_resolve_pass_records_and_submits() {
        let Some((device, queue)) = super::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let ssr = SsrPipelines::new_with_depth_samples(&device, TextureFormat::Rgba8UnormSrgb, 4);
        let target =
            ssr.create_scene_target_multisampled(&device, TextureFormat::Rgba8UnormSrgb, 64, 64, 4);
        let err_scope_1 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("ssr_depth_resolve_test"),
        });
        ssr.resolve_depth(&mut encoder, &target, None);
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let err = pollster::block_on(err_scope_1.pop());
        assert!(err.is_none(), "the depth resolve pass failed to record: {err:?}");
    }

    #[test]
    fn the_resolve_actually_writes_the_depth() {
        let Some((device, queue)) = super::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        const SIZE: u32 = 64;
        const DEPTH: f32 = 0.25;
        let ssr = SsrPipelines::new_with_depth_samples(&device, TextureFormat::Rgba8UnormSrgb, 4);
        let target =
            ssr.create_scene_target_multisampled(&device, TextureFormat::Rgba8UnormSrgb, SIZE, SIZE, 4);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        drop(encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("prime_depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(Operations { load: LoadOp::Clear(DEPTH), store: StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        }));
        ssr.resolve_depth(&mut encoder, &target, None);

        let bytes = SIZE * 4;
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("resolve_readback"),
            size: u64::from(bytes * SIZE),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: target.resolved_depth_texture(),
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes),
                    rows_per_image: Some(SIZE),
                },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        readback.slice(..).map_async(MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = readback.slice(..).get_mapped_range().unwrap();
        let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        readback.unmap();

        let middle = got[(SIZE / 2 * SIZE + SIZE / 2) as usize];
        assert!(
            (middle - DEPTH).abs() < 1e-4,
            "the resolved depth holds {middle} where the multisampled depth was \
             {DEPTH}; the copy is not copying, so every ray measures the scene \
             as infinitely far away and nothing is ever hit",
        );
    }

    #[test]
    fn only_a_one_layer_msaa_colour_is_transient() {
        assert!(super::msaa_colour_usage(1).contains(TextureUsages::TRANSIENT_ATTACHMENT));
        for layers in [2, crate::renderer::multiview::STEREO_VIEWS] {
            assert_eq!(super::msaa_colour_usage(layers), TextureUsages::RENDER_ATTACHMENT, "{layers} layers");
        }
    }

    #[test]
    fn the_multisampled_scene_colour_resolves_without_ever_being_stored() {
        let Some((device, queue)) = crate::renderer::pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };

        const SIZE: u32 = 64;
        let ssr = SsrPipelines::new_with_depth_samples(&device, TextureFormat::Rgba8UnormSrgb, 4);
        let target = ssr.create_scene_target_multisampled(&device, TextureFormat::Rgba8UnormSrgb, SIZE, SIZE, 4);
        let msaa = target.msaa_color_view.as_ref().expect("a 4x target has multisampled colour");

        let resolved = device.create_texture(&TextureDescriptor {
            label: Some("transient_resolve"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let resolved_view = resolved.create_view(&TextureViewDescriptor::default());
        let pass = |store: StoreOp| {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            drop(encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("transient_scene_colour"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: msaa,
                    depth_slice: None,
                    resolve_target: Some(&resolved_view),
                    ops: Operations { load: LoadOp::Clear(Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 }), store },
                })],
                ..Default::default()
            }));
            encoder
        };

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = pass(StoreOp::Discard);
        let bytes = SIZE * 4;
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("transient_readback"),
            size: u64::from(bytes * SIZE),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &resolved,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bytes), rows_per_image: Some(SIZE) },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "clear, resolve and discard was refused: {err:?}");
        readback.slice(..).map_async(MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = readback.slice(..).get_mapped_range().unwrap();
        let middle = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
        assert_eq!(&data[middle..middle + 4], &[255, 0, 255, 255], "the resolve did not carry the picture out");
        drop(data);
        readback.unmap();

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        queue.submit(Some(pass(StoreOp::Store).finish()));
        let err = pollster::block_on(scope.pop());
        assert!(err.is_some(), "storing a transient attachment was accepted");
    }

    #[test]
    fn the_march_reads_a_single_sampled_depth_texture() {
        let code = wgsl_ssr_block_shared_camera(3);
        assert!(
            code.contains("var ssr_scene_depth: texture_2d<f32>;"),
            "the march is not typed against a single-sampled depth texture",
        );
        assert!(
            !code.contains("texture_depth_multisampled_2d"),
            "the march still reads multisampled depth",
        );
    }
}

#[cfg(test)]
mod hi_z_tests {

    use super::*;

    fn pattern(w: u32, h: u32) -> Vec<f32> {

        let mut seed = 0x1234_5678u32;
        (0..w * h)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect()
    }

    fn build_and_read(w: u32, h: u32, level0: &[f32]) -> Vec<Vec<f32>> {
        let Some((device, queue)) = super::tests::headless_gpu() else {
            return Vec::new();
        };
        let ssr = SsrPipelines::new_with_depth_samples(&device, TextureFormat::Rgba8UnormSrgb, 4);
        let target =
            ssr.create_scene_target_multisampled(&device, TextureFormat::Rgba8UnormSrgb, w, h, 4);
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: target.resolved_depth_texture(),
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(level0),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        ssr.build_hi_z(&mut encoder, &target, None);

        let levels = target.hi_z_level_count();
        let mut buffers = Vec::new();
        for level in 0..levels {
            let (lw, lh) = ((w >> level).max(1), (h >> level).max(1));

            let row = (lw * 4).div_ceil(256) * 256;
            let buf = device.create_buffer(&BufferDescriptor {
                label: None,
                size: u64::from(row * lh),
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                TexelCopyTextureInfo {
                    texture: target.resolved_depth_texture(),
                    mip_level: level,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                TexelCopyBufferInfo {
                    buffer: &buf,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(lh),
                    },
                },
                Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
            );
            buffers.push((buf, lw, lh, row));
        }
        queue.submit(Some(encoder.finish()));
        for (buf, ..) in &buffers {
            buf.slice(..).map_async(MapMode::Read, |_| {});
        }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        buffers
            .iter()
            .map(|(buf, lw, lh, row)| {
                let data = buf.slice(..).get_mapped_range().unwrap();
                let mut out = Vec::with_capacity((lw * lh) as usize);
                for y in 0..*lh {
                    let start = (y * row) as usize;
                    let bytes = &data[start..start + (*lw as usize) * 4];
                    out.extend_from_slice(bytemuck::cast_slice::<u8, f32>(bytes));
                }
                drop(data);
                buf.unmap();
                out
            })
            .collect()
    }

    #[test]
    fn a_power_of_two_pyramid_is_the_exact_minimum() {
        const W: u32 = 64;
        const H: u32 = 64;
        let level0 = pattern(W, H);
        let levels = build_and_read(W, H, &level0);
        if levels.is_empty() {
            eprintln!("skipping: no GPU adapter available");
            return;
        }
        for (l, data) in levels.iter().enumerate().skip(1) {
            let (lw, lh) = ((W >> l).max(1), (H >> l).max(1));
            let block = 1u32 << l;
            for y in 0..lh {
                for x in 0..lw {
                    let mut want = f32::INFINITY;
                    for by in 0..block {
                        for bx in 0..block {
                            let (sx, sy) = (x * block + bx, y * block + by);
                            want = want.min(level0[(sy * W + sx) as usize]);
                        }
                    }
                    let got = data[(y * lw + x) as usize];
                    assert!(
                        (got - want).abs() < 1e-6,
                        "level {l} cell ({x},{y}) holds {got}, the nearest thing \
                         under it is {want}",
                    );
                }
            }
        }
    }

    #[test]
    fn an_odd_sized_pyramid_still_bounds_what_is_under_it() {
        const W: u32 = 105;
        const H: u32 = 61;
        let level0 = pattern(W, H);
        let levels = build_and_read(W, H, &level0);
        if levels.is_empty() {
            eprintln!("skipping: no GPU adapter available");
            return;
        }
        let mut ever_nearer = false;
        for (l, data) in levels.iter().enumerate().skip(1) {
            let (lw, lh) = ((W >> l).max(1), (H >> l).max(1));
            let block = 1u32 << l;
            for y in 0..lh {
                for x in 0..lw {
                    let mut want = f32::INFINITY;
                    for by in 0..block {
                        for bx in 0..block {
                            let (sx, sy) = ((x * block + bx).min(W - 1), (y * block + by).min(H - 1));
                            want = want.min(level0[(sy * W + sx) as usize]);
                        }
                    }
                    let got = data[(y * lw + x) as usize];
                    assert!(
                        got <= want + 1e-6,
                        "level {l} cell ({x},{y}) reports {got}, FURTHER than the \
                         {want} actually under it -- a ray will skip straight \
                         through whatever is in that cell",
                    );
                    ever_nearer |= got < want - 1e-6;
                }
            }
        }
        assert!(
            ever_nearer,
            "no cell in an odd-sized pyramid reached past its own block, so the \
             3x3 reduction is not running and the last column is being dropped",
        );
    }
}

fn downsample_shader() -> String {
    r"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VOut;
    let p = pos[vi];
    out.clip = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, samp, in.uv, 0.0);
}
"
    .to_string()
}

fn blit_shader(ms_depth: bool) -> String {
    let depth_ty = if ms_depth {
        "texture_depth_multisampled_2d"
    } else {
        "texture_depth_2d"
    };
    format!(
        r"
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
"
    )
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SsrOutput {

    Blended,

    FalseColour,

    Radiance,
}

impl SsrOutput {
    fn from_debug(debug: bool) -> Self {
        if debug { Self::FalseColour } else { Self::Blended }
    }
}

pub fn wgsl_ssr_block(camera_group: u32, scene_group: u32) -> String {
    wgsl_ssr_block_inner(Some(camera_group), scene_group)
}

pub fn wgsl_ssr_block_shared_camera(scene_group: u32) -> String {
    wgsl_ssr_block_inner(None, scene_group)
}

pub fn wgsl_ssr_block_shared_camera_debug(scene_group: u32, debug: bool) -> String {
    wgsl_ssr_block_debug(None, scene_group, SsrOutput::from_debug(debug))
}

pub fn wgsl_ssr_composite_block(scene_group: u32) -> String {
    format!(
        r"
const MAX_SSR_REFLECTIVITY: f32 = {MAX_SSR_REFLECTIVITY_VALUE:?};

@group({scene_group}) @binding(0) var ssr_scene_color: texture_2d<f32>;
@group({scene_group}) @binding(1) var ssr_reflection: texture_2d<f32>;
@group({scene_group}) @binding(2) var ssr_scene_samp: sampler;

fn ssr_reflection_at(clip_xy: vec2<f32>) -> vec4<f32> {{
    let limit = vec2<i32>(textureDimensions(ssr_reflection)) - vec2<i32>(1);

    let uv = clip_xy / {REFLECTION_SCALE}.0 - vec2<f32>(0.5);
    let base = floor(uv);
    let f = uv - base;
    let b = vec2<i32>(base);
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var j = 0; j < 2; j = j + 1) {{
        for (var i = 0; i < 2; i = i + 1) {{
            let p = clamp(b + vec2<i32>(i, j), vec2<i32>(0), limit);
            let t = textureLoad(ssr_reflection, p, 0);
            let bw = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
            let w = t.a * bw;
            sum = sum + t.rgb * w;
            weight = weight + w;
        }}
    }}
    if (weight <= 1e-5) {{
        return vec4<f32>(0.0);
    }}

    return vec4<f32>(sum / weight, weight);
}}
",
    )
}

const MAX_SSR_REFLECTIVITY_VALUE: f32 = 0.4;

pub fn wgsl_ssr_block_shared_camera_radiance(scene_group: u32) -> String {
    wgsl_ssr_block_debug(None, scene_group, SsrOutput::Radiance)
}

fn wgsl_ssr_block_inner(camera_group: Option<u32>, scene_group: u32) -> String {
    wgsl_ssr_block_debug(camera_group, scene_group, SsrOutput::from_debug(SSR_DEBUG))
}

fn wgsl_ssr_block_debug(
    camera_group: Option<u32>,
    scene_group: u32,
    output: SsrOutput,
) -> String {
    let _debug = output == SsrOutput::FalseColour;

    let depth_ty = "texture_2d<f32>";

    let max_mip = (SSR_MIPS - 1) as f32;

    let (camera_decl, vp, pos) = match camera_group {
        Some(g) => (
            format!(
                "struct SsrCamera {{\n    view_proj: mat4x4<f32>,\n    camera_pos: vec4<f32>,\n}}\n\
                 @group({g}) @binding(0) var<uniform> ssr_camera: SsrCamera;"
            ),
            "ssr_camera.view_proj",
            "ssr_camera.camera_pos.xyz",
        ),
        None => (String::new(), "cam_view_proj()", "cam_pos()"),
    };

    let (dbg_rough, dbg_miss, dbg_hit, dbg_facing) = match output {
        SsrOutput::FalseColour => (
            "return vec3<f32>(0.0, 0.0, 1.0); // BLUE: too rough to march",
            "return ssr_miss_colour(exit_reason); // see `ssr_miss_colour`",
            "return vec3<f32>(fade); // GREY: a hit, shaded by what survived the fades",
            "return vec3<f32>(1.0, 0.0, 1.0); // MAGENTA: faces the viewer, not marched",
        ),
        SsrOutput::Blended => (
            "return fallback;",
            "return fallback;",
            "return mix(fallback, hit_color, clamp(weight, 0.0, 1.0) * fade);",
            "return fallback;",
        ),

        SsrOutput::Radiance => (
            "return vec4<f32>(0.0);",
            "return vec4<f32>(0.0);",

            "return vec4<f32>(hit_color, clamp(fade, 0.0, 1.0));",
            "return vec4<f32>(0.0);",
        ),
    };

    let ret_ty = match output {
        SsrOutput::Radiance => "vec4<f32>",
        _ => "vec3<f32>",
    };
    format!(
        r"
{camera_decl}

@group({scene_group}) @binding(0) var ssr_scene_color: texture_2d<f32>;
@group({scene_group}) @binding(1) var ssr_scene_depth: {depth_ty};
@group({scene_group}) @binding(2) var ssr_scene_samp: sampler;

const SSR_STEPS: u32 = 32u;

const SSR_HI_Z_LEVELS: i32 = 8;

const SSR_HI_Z_START_LEVEL: i32 = 2;

const SSR_HI_Z_MAX_ITERATIONS: u32 = 64u;

const SSR_CELL_NUDGE_TEXELS: f32 = 0.05;

const SSR_MIN_RAY_TEXELS: f32 = 2.0;
const SSR_STRIDE_PIXELS: f32 = 4.0;
const SSR_STRIDE_MAX: f32 = 24.0;

const SSR_BUDGET_FADE_FROM: f32 = 0.6;

const SSR_MAX_DISTANCE: f32 = 5.07;

const SSR_NEAR_W: f32 = 0.06;

const MAX_SSR_REFLECTIVITY: f32 = 0.4;

const SSR_ROUGHNESS_MIPS: f32 = 8.0;
const SSR_THICKNESS_METRES: f32 = 0.25;

const SSR_REFINE_STEPS: u32 = 6u;

const SSR_MIN_WEIGHT: f32 = 0.025;

const SSR_TRUST_NEAR: f32 = 0.18;
const SSR_TRUST_FAR: f32 = 0.87;

const SSR_NORMAL_BIAS: f32 = 0.05;

fn ssr_miss_colour(reason: u32) -> vec3<f32> {{
    if (reason == 1u) {{
        return vec3<f32>(1.0, 0.0, 0.0);
    }}
    if (reason == 2u) {{
        return vec3<f32>(1.0, 1.0, 0.0);
    }}
    if (reason == 3u) {{
        return vec3<f32>(0.0, 1.0, 1.0);
    }}

    if (reason == 4u) {{
        return vec3<f32>(1.0, 0.45, 0.0);
    }}
    return vec3<f32>(0.0, 1.0, 0.0);
}}

fn ssr_frame_edge(c: vec4<f32>) -> f32 {{
    if (c.w <= 0.0) {{
        return 0.0;
    }}
    let ndc = c.xy / c.w;
    return 0.5 * min(min(1.0 - ndc.x, 1.0 + ndc.x), min(1.0 - ndc.y, 1.0 + ndc.y));
}}

fn ssr_reflect(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    fallback: vec3<f32>,
    weight: f32,
    roughness: f32,
) -> {ret_ty} {{

    if (weight < SSR_MIN_WEIGHT) {{
        {dbg_rough}
    }}

    let threshold_fade = smoothstep(SSR_MIN_WEIGHT, SSR_MIN_WEIGHT * 2.0, weight);
    let n = normalize(world_normal);
    let view_dir = normalize(world_pos - {pos});
    let refl_dir = reflect(view_dir, n);

    let facing_fade = smoothstep(-0.7, -0.2, dot(refl_dir, view_dir));
    if (facing_fade <= 0.0) {{
        {dbg_facing}
    }}

    let scene_size = vec2<f32>(textureDimensions(ssr_scene_color));

    let march_origin = world_pos + n * SSR_NORMAL_BIAS;
    var far_end = march_origin + refl_dir * SSR_MAX_DISTANCE;
    let h0 = {vp} * vec4<f32>(march_origin, 1.0);
    var h1 = {vp} * vec4<f32>(far_end, 1.0);

    var hit = false;
    var hit_uv = vec2<f32>(0.0);
    var hit_pos = march_origin;

    var budget_used = 0.0;

    var exit_reason: u32 = 0u;

    if (h0.w > SSR_NEAR_W) {{

        if (h1.w <= SSR_NEAR_W) {{
            far_end = mix(march_origin, far_end, (h0.w - SSR_NEAR_W) / (h0.w - h1.w));
            h1 = {vp} * vec4<f32>(far_end, 1.0);
        }}
        let k0 = 1.0 / h0.w;
        let k1 = 1.0 / h1.w;
        let s0 = vec2<f32>(h0.x * k0 * 0.5 + 0.5, 0.5 - h0.y * k0 * 0.5) * scene_size;
        let s1 = vec2<f32>(h1.x * k1 * 0.5 + 0.5, 0.5 - h1.y * k1 * 0.5) * scene_size;

        let dk = k1 - k0;
        let bz = select(0.0, (h1.z * k1 - h0.z * k0) / dk, abs(dk) > 1e-9);
        if (abs(bz) <= 1e-12) {{
            exit_reason = 3u;
        }}
        if (abs(bz) > 1e-12) {{
            let az = h0.z * k0 - bz * k0;

            let ds = s1 - s0;
            let dz = h1.z * k1 - h0.z * k0;

            if (length(ds) <= SSR_MIN_RAY_TEXELS) {{
                exit_reason = 2u;
            }}
            if (length(ds) > SSR_MIN_RAY_TEXELS) {{

                var level = SSR_HI_Z_START_LEVEL;

                var t = min(1.0 / length(ds), 1.0);
                for (var i = 0u; i < SSR_HI_Z_MAX_ITERATIONS; i = i + 1u) {{
                    let p = s0 + ds * t;
                    if (t > 1.0 || p.x < 0.0 || p.y < 0.0 || p.x >= scene_size.x || p.y >= scene_size.y) {{
                        exit_reason = 1u;
                        break;
                    }}
                    let cell_size = f32(1 << u32(level));
                    let cell = floor(p / cell_size);

                    let next_x = select(cell.x, cell.x + 1.0, ds.x > 0.0) * cell_size;
                    let next_y = select(cell.y, cell.y + 1.0, ds.y > 0.0) * cell_size;
                    let tx = select(1e30, (next_x - s0.x) / ds.x, abs(ds.x) > 1e-6);
                    let ty = select(1e30, (next_y - s0.y) / ds.y, abs(ds.y) > 1e-6);

                    let t_exit = select(
                        ty + SSR_CELL_NUDGE_TEXELS / max(abs(ds.y), 1e-6),
                        tx + SSR_CELL_NUDGE_TEXELS / max(abs(ds.x), 1e-6),
                        tx < ty,
                    );

                    let level_max = vec2<i32>(textureDimensions(ssr_scene_depth, level)) - vec2<i32>(1);
                    let cell_min = textureLoad(
                        ssr_scene_depth,
                        clamp(vec2<i32>(cell), vec2<i32>(0), level_max),
                        level,
                    ).x;

                    let t_depth = select(1e30, (cell_min - h0.z * k0) / dz, abs(dz) > 1e-12);

                    let t_next = max(t, t_depth);
                    let stayed = all(floor((s0 + ds * t_next) / cell_size) == cell);
                    if (!stayed) {{
                        t = t_exit;
                        level = min(level + 1, SSR_HI_Z_LEVELS - 1);
                    }} else if (level == 0) {{
                        let hit_t = clamp(t_next, 0.0, 1.0);
                        let hs = s0 + ds * hit_t;
                        let hpx = vec2<i32>(clamp(hs, vec2<f32>(0.0), scene_size - vec2<f32>(1.0)));
                        let kh = k0 + dk * hit_t;
                        let scene_w = bz / (textureLoad(ssr_scene_depth, hpx, 0).x - az);
                        if (1.0 / kh <= scene_w + SSR_THICKNESS_METRES) {{
                            hit = true;
                            hit_uv = hs / scene_size;
                            hit_pos = mix(march_origin * k0, far_end * k1, hit_t) / kh;
                            budget_used = 0.0;
                            break;
                        }}

                        exit_reason = 4u;
                        break;
                    }} else {{
                        t = t_next;
                        level = level - 1;
                    }}
                }}
            }}
        }}
    }}

    if (!hit) {{
        {dbg_miss}
    }}

    let mip = clamp(roughness * SSR_ROUGHNESS_MIPS, 0.0, {max_mip:.1});
    let hit_color = textureSampleLevel(ssr_scene_color, ssr_scene_samp, hit_uv, mip).rgb;

    let edge = min(
        ssr_frame_edge({vp} * vec4<f32>(march_origin, 1.0)),
        ssr_frame_edge({vp} * vec4<f32>(hit_pos, 1.0)),
    );

    let edge_fade = smoothstep(0.0, 0.25, edge);

    let grazing_fade = smoothstep(0.15, 0.45, dot(n, -view_dir));

    let reach = SSR_MAX_DISTANCE;

    let travelled = length(hit_pos - march_origin);
    let distance_fade =
        1.0 - smoothstep(SSR_TRUST_NEAR, SSR_TRUST_FAR, travelled / max(reach, 1e-4));

    let budget_fade = 1.0 - smoothstep(SSR_BUDGET_FADE_FROM, 1.0, budget_used);
    let fade = edge_fade * grazing_fade * distance_fade * facing_fade * threshold_fade * budget_fade;

    {dbg_hit}
}}
"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn headless_gpu() -> Option<(Device, Queue)> {
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

#[cfg(test)]
mod mip_and_fallback_tests {
    use super::*;
    use crate::renderer::terrain_pipeline::tests::headless_gpu;

    #[test]
    fn the_scene_colour_really_has_a_chain_to_blur_with() {

        let Some((device, _q)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let ssr = SsrPipelines::new(&device, TextureFormat::Rgba8UnormSrgb);
        let target = ssr.create_scene_target_multisampled(
            &device, TextureFormat::Rgba8UnormSrgb, 512, 512, 1,
        );
        assert_eq!(
            target.mip_targets.len() as u32,
            SSR_MIPS - 1,
            "the chain has no levels to downsample into",
        );
        assert_eq!(
            target.mip_sources.len(),
            target.mip_targets.len(),
            "every level to write needs a level to read",
        );
    }

    #[test]
    fn a_target_smaller_than_the_chain_does_not_ask_for_levels_it_lacks() {

        let Some((device, _q)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let ssr = SsrPipelines::new(&device, TextureFormat::Rgba8UnormSrgb);
        let target = ssr.create_scene_target_multisampled(
            &device, TextureFormat::Rgba8UnormSrgb, 4, 4, 1,
        );
        assert!(target.mip_targets.len() < (SSR_MIPS - 1) as usize);
    }

    #[test]
    fn the_reflection_falls_back_rather_than_vanishing() {

        let src = wgsl_ssr_block_debug(None, 3, SsrOutput::Blended);
        assert_eq!(
            src.matches("return fallback;").count(),
            3,
            "a miss path still returns something other than the fallback",
        );
        assert!(
            !src.contains("base_color"),
            "the old self-coloured fallback is still in the shader",
        );
    }

    #[test]
    fn roughness_chooses_the_mip() {
        let src = wgsl_ssr_block_shared_camera(3);
        assert!(src.contains("roughness * SSR_ROUGHNESS_MIPS"), "roughness does not select a level");
        assert!(
            src.contains("textureSampleLevel(ssr_scene_color"),
            "the reflection is still point-loaded, so it cannot be blurred",
        );
    }
}

#[cfg(test)]
mod reflection_blend_tests {

    use super::*;

    fn blend(fallback: f32, hit: f32, weight: f32, fade: f32) -> f32 {
        let t = weight.clamp(0.0, 1.0) * fade;
        fallback * (1.0 - t) + hit * t
    }

    #[test]
    fn a_weak_reflection_barely_disturbs_the_surface() {
        let wall = 0.8;
        let grass = 0.3;
        let got = blend(wall, grass, 0.04, 1.0);
        assert!(
            (got - wall).abs() <= 0.04,
            "a 4% reflective surface moved from {wall} to {got}; it is being \
             replaced by its reflection rather than tinted by it",
        );
        assert!(got > 0.5, "the wall stopped looking like a wall: {got}");
    }

    #[test]
    fn the_edge_of_a_reflection_is_not_a_cliff() {
        let (wall, grass, w) = (0.8, 0.3, 0.04);
        let hit = blend(wall, grass, w, 1.0);
        let miss = wall;
        assert!(
            (hit - miss).abs() < 0.05,
            "a marched hit ({hit}) and a miss ({miss}) differ by more than the \
             surface's own reflectance, so every step of the march is visible",
        );
    }

    #[test]
    fn a_fully_reflective_surface_still_shows_the_reflection() {
        assert!((blend(0.8, 0.3, 1.0, 1.0) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn no_confidence_means_no_reflection() {
        assert!((blend(0.8, 0.3, 1.0, 0.0) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn the_shader_blends_by_weight_rather_than_scaling_the_hit() {

        let src = wgsl_ssr_block_debug(None, 3, SsrOutput::Blended);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code.contains("mix(fallback, hit_color, clamp(weight, 0.0, 1.0) * fade)"),
            "the reflection blend is not the form `reflection_blend_tests` pins",
        );
        assert!(
            !code.contains("hit_color * weight"),
            "the reflection is scaling the hit and discarding the surface again",
        );
    }
}

#[cfg(test)]
mod grazing_fade_tests {

    use super::*;

    fn fade(view_deg: f32) -> f32 {
        let cos_v = view_deg.to_radians().cos();
        let t = ((cos_v - 0.15) / (0.45 - 0.15)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    fn old_fade(view_deg: f32) -> f32 {
        let a = view_deg.to_radians();
        let v = [a.sin(), 0.0, -a.cos()];
        let n = [0.0f32, 0.0, 1.0];
        let d = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
        let refl = [v[0] - 2.0 * d * n[0], v[1] - 2.0 * d * n[1], v[2] - 2.0 * d * n[2]];
        let dot_rv = -(refl[0] * v[0] + refl[1] * v[1] + refl[2] * v[2]);
        let t = (dot_rv.abs() / 0.2).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    #[test]
    fn a_surface_seen_edge_on_gets_no_screen_space_reflection() {
        assert!(fade(85.0) < 0.01, "at 85 degrees the fade is {}", fade(85.0));
        assert!(fade(89.0) < 0.01);
    }

    #[test]
    fn a_surface_seen_head_on_keeps_its_reflection() {
        assert!(fade(0.0) > 0.99);
        assert!(fade(30.0) > 0.99);
    }

    #[test]
    fn the_fade_only_ever_decreases_as_the_view_turns_away() {
        let mut last = f32::INFINITY;
        for deg in 0..90 {
            let f = fade(deg as f32);
            assert!(f <= last + 1e-6, "fade rose from {last} to {f} at {deg} degrees");
            last = f;
        }
    }

    #[test]
    fn the_old_fade_never_faded_anything() {
        for deg in [0.0f32, 30.0, 60.0, 80.0, 89.0] {
            assert!(
                old_fade(deg) > 0.99,
                "the old fade was {} at {deg} degrees; if it ever fell, the \
                 justification for replacing it is wrong",
                old_fade(deg),
            );
        }
    }

    #[test]
    fn the_shader_fades_on_the_view_angle() {
        let src = wgsl_ssr_block_shared_camera(3);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code.contains("smoothstep(0.15, 0.45, dot(n, -view_dir))"),
            "the grazing fade is not the form `grazing_fade_tests` pins",
        );
        assert!(
            !code.contains("abs(dot(refl_dir, -view_dir))"),
            "the fade that never faded is back",
        );
    }
}

#[cfg(test)]
mod march_precision_tests {

    use super::*;

    const HI_Z_LEVELS: i32 = 8;
    const HI_Z_START_LEVEL: i32 = 2;
    const HI_Z_MAX_ITERATIONS: u32 = 64;

    #[test]
    fn the_top_of_the_pyramid_skips_something_worth_skipping() {
        let cell = 1i32 << (HI_Z_LEVELS - 1);
        assert!(
            cell >= 64,
            "the coarsest cell is {cell} px, which is close enough to a texel \
             that the pyramid is not earning the passes that build it",
        );
    }

    #[test]
    fn the_iteration_cap_leaves_room_to_descend_and_climb() {
        assert!(
            HI_Z_MAX_ITERATIONS as i32 > HI_Z_LEVELS * 2,
            "{HI_Z_MAX_ITERATIONS} iterations is not enough for a ray to \
             descend {HI_Z_LEVELS} levels and climb back more than once",
        );
        assert!(
            HI_Z_START_LEVEL > 0 && HI_Z_START_LEVEL < HI_Z_LEVELS,
            "starting at {HI_Z_START_LEVEL} is either a linear march or above \
             the pyramid",
        );
    }

    #[test]
    fn the_shader_walks_the_pyramid_down_to_a_single_texel() {
        let src = wgsl_ssr_block_shared_camera(3);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code.contains("if (level == 0) {"),
            "the traversal never bottoms out at level 0, so a hit is only \
             located to whatever cell it stopped in",
        );
        assert!(
            code.contains("level = level - 1;") && code.contains("level = min(level + 1, SSR_HI_Z_LEVELS - 1);"),
            "the traversal no longer descends AND climbs; one direction alone \
             is a linear march wearing a pyramid",
        );
        assert!(
            code.contains("tx + SSR_CELL_NUDGE_TEXELS / max(abs(ds.x), 1e-6),")
                && code.contains("ty + SSR_CELL_NUDGE_TEXELS / max(abs(ds.y), 1e-6),")
                && code.contains("if (length(ds) > SSR_MIN_RAY_TEXELS) {"),
            "the nudge past a cell boundary is not measured in TEXELS; as a \
             constant in the ray parameter it vanishes on a long ray and the \
             walk stalls on its own boundary until the iteration cap gives up; \
             uncapped it swallows a short ray whole, which drew a starburst of \
             failed rays around the point where the reflection turns to face \
             the viewer",
        );

        assert!(code.contains(&format!("const SSR_HI_Z_LEVELS: i32 = {HI_Z_LEVELS};")));
        assert!(code.contains(&format!("const SSR_HI_Z_START_LEVEL: i32 = {HI_Z_START_LEVEL};")));
        assert!(code.contains(&format!(
            "const SSR_HI_Z_MAX_ITERATIONS: u32 = {HI_Z_MAX_ITERATIONS}u;"
        )));
    }

}

#[cfg(test)]
mod self_reflection_tests {

    use super::*;

    fn old_lift(view_deg: f32) -> f32 {
        let a = view_deg.to_radians();

        let v = [a.sin(), 0.0, -a.cos()];
        let n = [0.0f32, 0.0, 1.0];
        let d = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
        let refl = [v[0] - 2.0 * d * n[0], v[1] - 2.0 * d * n[1], v[2] - 2.0 * d * n[2]];

        (refl[0] * n[0] + refl[1] * n[1] + refl[2] * n[2]) * 0.12 * 0.5
    }

    #[test]
    fn the_old_bias_barely_left_the_surface_at_grazing() {
        assert!(old_lift(0.0) > 0.05, "head-on it lifted {}", old_lift(0.0));
        assert!(
            old_lift(85.0) < 0.006,
            "at 85 degrees it lifted {}m, which would have been enough",
            old_lift(85.0),
        );
    }

    #[test]
    fn the_normal_bias_is_the_same_at_every_angle() {

        for deg in [0.0f32, 45.0, 85.0, 89.0] {
            let _ = deg;
            assert!(SSR_NORMAL_BIAS_F >= 0.02, "the bias is too small to clear a surface");
        }
        assert!(
            SSR_NORMAL_BIAS_F > old_lift(85.0) * 5.0,
            "the new bias is not meaningfully larger than the old one at grazing",
        );
    }

    const SSR_NORMAL_BIAS_F: f32 = 0.05;

    const THICKNESS: f32 = 0.25;

    fn is_hit(w_prev: f32, w: f32, scene_w: f32) -> bool {
        w_prev.max(w) >= scene_w && w_prev.min(w) <= scene_w + THICKNESS
    }

    #[test]
    fn a_ray_that_crosses_a_surface_registers() {

        assert!(is_hit(2.0, 2.2, 2.1), "a genuine crossing was rejected");
    }

    #[test]
    fn a_surface_the_step_jumped_over_still_registers() {
        assert!(
            is_hit(2.0, 3.0, 2.4),
            "a surface in the middle of the step was missed; small things will \
             be stamped once per step instead of reflected once",
        );
    }

    #[test]
    fn a_ray_already_far_behind_the_surface_does_not_register() {

        assert!(
            !is_hit(3.0, 3.2, 2.0),
            "a ray long past the surface still reports a hit, so a ray skimming \
             a wall reflects the wall it is skimming",
        );
        assert!(2.0 < 3.2, "sanity: `behind the surface` alone is satisfied here");
    }

    #[test]
    fn a_crossing_that_overshot_is_still_a_crossing() {
        assert!(
            is_hit(2.0, 5.0, 2.05),
            "a crossing that overshot by metres was rejected; the reflection \
             will break into bars wherever it meets a depth discontinuity",
        );
    }

    #[test]
    fn the_slab_is_a_quarter_of_a_metre_wherever_the_ray_is() {

        assert!(is_hit(2.2, 2.2, 2.0));
        assert!(is_hit(40.2, 40.2, 40.0));

        assert!(!is_hit(2.3, 2.3, 2.0));
        assert!(!is_hit(40.3, 40.3, 40.0));
    }

    #[test]
    fn the_shader_carries_both_guards() {
        let src = wgsl_ssr_block_shared_camera(3);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code.contains("world_pos + n * SSR_NORMAL_BIAS"),
            "the ray no longer starts off its own surface along the normal",
        );

        assert!(
            code.contains("if (1.0 / kh <= scene_w + SSR_THICKNESS_METRES) {"),
            "the slab test is gone from the traversal's level 0, so a ray that \
             slid behind a wall reports hitting it",
        );
        assert!(
            code.contains("let cell_min = textureLoad(")
                && code.contains("clamp(vec2<i32>(cell), vec2<i32>(0), level_max),"),
            "the traversal no longer reads the pyramid, so it is a linear march \
             again -- see the note on `SSR_HI_Z_LEVELS`",
        );
        assert!(
            code.contains(&format!("const SSR_THICKNESS_METRES: f32 = {THICKNESS};")),
            "the shader's slab is not the {THICKNESS} m these tests measure",
        );
        assert!(
            !code.contains("SSR_THICKNESS_STEPS"),
            "the step-sized thickness tolerance is back; see \
             `fade_continuity_twin` for what it does to the fade",
        );
        assert!(code.contains(&format!("const SSR_NORMAL_BIAS: f32 = {SSR_NORMAL_BIAS_F};")));
    }
}

#[cfg(test)]
mod confidence_tests {

    use super::*;

    const NEAR: f32 = 0.18;
    const FAR: f32 = 0.87;

    fn reach() -> f32 {
        let mut total = 0.0;
        let mut s = 0.12f32;
        for _ in 0..13 {
            total += s;
            s *= 1.18;
        }
        total
    }

    fn trust(travelled_fraction: f32) -> f32 {
        let t = ((travelled_fraction - NEAR) / (FAR - NEAR)).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    }

    #[test]
    fn a_short_ray_keeps_its_reflection() {
        assert!(trust(0.0) > 0.99);

        assert!(
            trust(0.5 / reach()) > 0.9,
            "a reflection half a metre from its surface was already doubted; that \
             is the one case the screen answers better than the probe",
        );
    }

    #[test]
    fn a_long_ray_hands_over_to_the_probe() {

        assert!(trust(1.0) < 0.01);
        assert!(
            trust(4.5 / reach()) < 0.01,
            "a ray four and a half metres long still contributes {}; that is the \
             range where hits and misses alternate across a depth discontinuity",
            trust(4.5 / reach()),
        );
    }

    #[test]
    fn the_handover_is_gradual_and_never_reverses() {
        let n = 2000;
        let mut last = f32::INFINITY;
        for i in 0..=n {
            let f = trust(i as f32 / n as f32);
            assert!(f <= last + 1e-6, "trust rose from {last} to {f}");
            if i > 0 {
                let prev = trust((i - 1) as f32 / n as f32);
                assert!((prev - f).abs() < 0.02, "trust jumps {prev} -> {f}");
            }
            last = f;
        }
    }

    #[test]
    fn the_step_length_criterion_is_deliberately_not_tested_here() {
        let ratio = |k: u32| {
            let mut d = 0.0;
            let mut s = 0.12f32;
            for _ in 0..k {
                d += s;
                s *= 1.18;
            }
            (s / 1.18) / d
        };

        assert!(
            ratio(5) > ratio(19),
            "the march's relative coarseness no longer falls with distance ({} -> {}); \
             the note above is out of date",
            ratio(5),
            ratio(19),
        );
        assert!(
            ratio(19) > 0.15 && ratio(5) < 0.35,
            "relative coarseness left the range this note quotes: {} -> {}",
            ratio(5),
            ratio(19),
        );
    }

    #[test]
    fn the_shader_fades_by_distance_travelled() {
        let src = wgsl_ssr_block_shared_camera(3);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code.contains("let travelled = length(hit_pos - march_origin);"),
            "the ray no longer measures how far it went to the REFINED hit; if this \
             went back to accumulating coarse steps, the fade quantises into one \
             band per march step and every reflection ripples",
        );
        assert!(
            code.contains("hit_pos = mix(march_origin * k0, far_end * k1, hit_t) / kh;"),
            "the refinement no longer records WHERE it localised the hit -- \
             perspective-correct, so the world position matches the pixel the \
             colour is read from -- and the distance measured from it is the \
             coarse stride's again",
        );
        assert!(
            code.contains("smoothstep(SSR_TRUST_NEAR, SSR_TRUST_FAR, travelled / max(reach, 1e-4))"),
            "the distance fade is not the form `confidence_tests` pins",
        );

        assert!(
            code.contains("let fade = edge_fade * grazing_fade * distance_fade"),
            "the distance fade is computed but not applied",
        );
        assert!(
            code.contains("* facing_fade * threshold_fade * budget_fade;"),
            "the roughness threshold went back to a hard step; brick's roughness \
             map straddles it and neighbouring texels take different code paths, \
             which is the speckle on the stone",
        );
        assert!(
            code.contains(&format!("const SSR_TRUST_NEAR: f32 = {NEAR};")),
            "the shader's near trust bound is not the {NEAR} this module measures",
        );
        assert!(
            code.contains(&format!("const SSR_TRUST_FAR: f32 = {FAR};")),
            "the shader's far trust bound is not the {FAR} this module measures",
        );
    }
}

#[cfg(test)]
mod screen_edge_tests {

    use super::*;

    fn edge_fade(closest: f32) -> f32 {
        let t = (closest / 0.12).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    #[test]
    fn a_ray_that_nearly_left_the_frame_is_nearly_faded() {
        assert!(
            edge_fade(0.005) < 0.02,
            "a ray half a percent from the edge still contributes {}",
            edge_fade(0.005),
        );
    }

    #[test]
    fn a_ray_that_stayed_inside_keeps_its_reflection() {
        assert!(edge_fade(0.3) > 0.99);
        assert!(edge_fade(0.12) > 0.99);
    }

    #[test]
    fn the_fade_is_gradual_across_the_band() {
        let mut last = 0.0;
        for i in 0..=60 {
            let f = edge_fade(i as f32 * 0.003);
            assert!(f >= last - 1e-6, "the fade reversed at {i}");
            assert!(f - last < 0.09, "the fade jumps {last} -> {f}");
            last = f;
        }
    }

    #[test]
    fn the_shader_measures_the_whole_march_not_just_the_hit() {
        let src = wgsl_ssr_block_shared_camera(3);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code.contains("ssr_frame_edge(cam_view_proj() * vec4<f32>(march_origin, 1.0)),")
                && code.contains("ssr_frame_edge(cam_view_proj() * vec4<f32>(hit_pos, 1.0)),"),
            "the edge is no longer the exact minimum over the travelled \
             segment, taken at its two ends; anything sampled along the march \
             carries the step phase into the fade and bands it",
        );

        assert!(
            code.contains(&format!("smoothstep(0.0, {EDGE_FADE_BAND}, edge)")),
            "the screen-edge fade band moved; it is what hides the wedge where \
             rays leave the frame",
        );
        assert!(
            EDGE_FADE_BAND >= 0.2,
            "a band of {EDGE_FADE_BAND} is too narrow to hide the handover",
        );

        assert!(
            code.contains("exit_reason = 1u;"),
            "the march no longer records that it left the frame, so the \
             false-colour view cannot tell that case from running out of steps",
        );
        assert!(
            code.contains("p.x < 0.0 || p.y < 0.0 || p.x >= scene_size.x || p.y >= scene_size.y"),
            "the traversal no longer checks it is still on screen before \
             reading the pyramid",
        );
    }

    const EDGE_FADE_BAND: f32 = 0.25;
}

#[cfg(test)]
mod march_budget_tests {

    use super::*;

    const MIN: f32 = 0.025;

    fn weight(roughness: f32, view_deg: f32) -> f32 {
        let cv = view_deg.to_radians().cos();
        let f = 0.04 + 0.96 * (1.0 - cv).powi(5);
        ((1.0 - roughness) * f).min(0.4)
    }

    #[test]
    fn rough_stone_seen_head_on_skips_the_march() {
        for deg in [0.0f32, 30.0, 45.0] {
            let w = weight(0.596, deg);
            assert!(
                w < MIN,
                "brick at {deg} degrees asks for {w}, which still pays for a \
                 26-tap search to move the pixel by {} of 255",
                w * 255.0,
            );
        }
    }

    #[test]
    fn polished_marble_still_marches() {
        for deg in [0.0f32, 45.0, 70.0] {
            let w = weight(0.048, deg);
            assert!(w > MIN, "marble at {deg} degrees would skip the march ({w})");
        }
    }

    #[test]
    fn rough_stone_at_a_grazing_angle_marches_again() {
        assert!(weight(0.596, 70.0) > MIN);
    }

    #[test]
    fn the_threshold_cannot_cost_more_than_a_few_levels() {
        assert!(MIN * 255.0 < 7.0, "the skip can change a pixel by {}", MIN * 255.0);
    }

    #[test]
    fn the_shader_skips_below_the_threshold() {
        let src = wgsl_ssr_block_shared_camera(3);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(code.contains("if (weight < SSR_MIN_WEIGHT)"), "the march runs at any weight again");
        assert!(code.contains(&format!("const SSR_MIN_WEIGHT: f32 = {MIN};")));
    }
}

#[cfg(test)]
mod travel_continuity_tests {
    use super::*;

    const SSR_STEPS_COUNT: u32 = 32;
    const SSR_REFINE_COUNT: u32 = 6;
    const SSR_MAX_DISTANCE: f32 = 5.07;
    const SSR_STRIDE_MAX: f32 = 24.0;
    const SSR_TRUST_NEAR: f32 = 0.18;
    const SSR_TRUST_FAR: f32 = 0.87;

    const OLD_STEP_SIZE: f32 = 0.12;
    const OLD_STEP_GROWTH: f32 = 1.18;
    const OLD_STEPS: u32 = 13;

    fn ssr_wgsl_for_test() -> String {
        wgsl_ssr_block_shared_camera(3)
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn reach() -> f32 {
        SSR_MAX_DISTANCE
    }

    fn coarse_distance(k: u32) -> f32 {
        let mut d = 0.0;
        let mut s = OLD_STEP_SIZE;
        for _ in 0..k {
            d += s;
            s *= OLD_STEP_GROWTH;
        }
        d
    }

    fn distance_fade(travelled: f32) -> f32 {
        let t = (travelled / reach().max(1e-4)).clamp(0.0, 1.0);
        let x = ((t - SSR_TRUST_NEAR) / (SSR_TRUST_FAR - SSR_TRUST_NEAR)).clamp(0.0, 1.0);
        1.0 - x * x * (3.0 - 2.0 * x)
    }

    #[test]
    fn quantising_the_distance_to_coarse_steps_makes_a_visible_step_in_the_fade() {
        let worst = (1..OLD_STEPS)
            .map(|k| (distance_fade(coarse_distance(k)) - distance_fade(coarse_distance(k + 1))).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst > 0.1,
            "adjacent coarse steps differed by only {worst} of full strength, so \
             this test is no longer describing the banding it was written for",
        );
    }

    #[test]
    fn measuring_to_the_refined_hit_leaves_no_step_in_the_fade() {
        let r = reach();
        let n = 4000;
        let worst = (0..n)
            .map(|i| {
                let a = r * i as f32 / n as f32;
                let b = r * (i + 1) as f32 / n as f32;
                (distance_fade(a) - distance_fade(b)).abs()
            })
            .fold(0.0f32, f32::max);

        assert!(
            worst < 0.005,
            "a continuously varying distance still moved the fade by {worst} between \
             neighbouring samples",
        );
    }

    #[test]
    fn refinement_narrows_the_hit_to_far_less_than_a_stride() {
        let residual = SSR_STRIDE_MAX / 2f32.powi(SSR_REFINE_COUNT as i32);
        assert!(
            residual < 1.0,
            "six halvings leave {residual} px of a {SSR_STRIDE_MAX} px stride, \
             which is coarser than the buffer the hit is read from",
        );
    }

    #[test]
    fn the_shader_uses_the_constants_this_module_models() {
        let code = ssr_wgsl_for_test();
        assert!(code.contains(&format!("const SSR_STEPS: u32 = {SSR_STEPS_COUNT}u;")));
        assert!(code.contains(&format!("const SSR_REFINE_STEPS: u32 = {SSR_REFINE_COUNT}u;")));
        assert!(code.contains(&format!("const SSR_MAX_DISTANCE: f32 = {SSR_MAX_DISTANCE};")));
        assert!(code.contains(&format!("const SSR_STRIDE_MAX: f32 = {SSR_STRIDE_MAX:?};")));
        assert!(code.contains(&format!("const SSR_TRUST_NEAR: f32 = {SSR_TRUST_NEAR};")));
        assert!(code.contains(&format!("const SSR_TRUST_FAR: f32 = {SSR_TRUST_FAR};")));
    }
}

#[cfg(test)]
mod handover_range_tests {
    use super::*;

    const STEP: f32 = 0.12;
    const GROWTH: f32 = 1.18;
    const STEPS: u32 = 13;

    fn reach() -> f32 {
        let mut r = 0.0;
        let mut s = STEP;
        for _ in 0..STEPS {
            r += s;
            s *= GROWTH;
        }
        r
    }

    #[test]
    fn the_screen_answers_at_arms_length() {
        let full = SSR_TRUST_NEAR_F * reach();
        assert!(
            (0.5..1.5).contains(&full),
            "screen reflections hold full strength out to {full} m; they are meant \
             to be a contact effect, and beyond about a metre the probe is the \
             better answer",
        );
    }

    #[test]
    fn long_rays_cannot_contribute_at_all() {
        let gone = SSR_TRUST_FAR_F * reach();
        assert!(
            gone < 5.0,
            "screen reflections still contribute at {gone} m. Every remaining \
             artefact is a long-ray artefact: at that distance the ray straddles \
             depth discontinuities and its hits alternate with misses, which is \
             the sliced doorway reflection. The probe has no such failure.",
        );
        assert!(
            gone > SSR_TRUST_NEAR_F * reach(),
            "the fade band is inverted or empty",
        );
    }

    #[test]
    fn the_shader_uses_this_handover() {
        let code = wgsl_ssr_block_shared_camera(3);
        assert!(code.contains(&format!("const SSR_TRUST_NEAR: f32 = {SSR_TRUST_NEAR_F};")));
        assert!(code.contains(&format!("const SSR_TRUST_FAR: f32 = {SSR_TRUST_FAR_F};")));
    }

    const SSR_TRUST_NEAR_F: f32 = 0.18;
    const SSR_TRUST_FAR_F: f32 = 0.87;
}

#[cfg(test)]
mod ssr_debug_switch_tests {
    use super::*;

    #[test]
    fn every_miss_colour_is_off_the_grey_ramp() {
        let code = ssr_miss_colour_source();
        let colours: Vec<[f32; 3]> = regex_lite_vec3s(&code);
        assert!(
            colours.len() >= 5,
            "only {} miss colours found in `ssr_miss_colour`; the parse below \
             has stopped matching the source and this test is no longer \
             checking anything",
            colours.len(),
        );
        for c in &colours {
            let grey = (c[0] - c[1]).abs() < 1e-6 && (c[1] - c[2]).abs() < 1e-6;
            assert!(
                !grey,
                "miss colour {c:?} has three equal channels, so it is a shade \
                 of grey -- and a HIT is drawn as `vec3(fade)`, which is every \
                 shade of grey. Pick a colour off the diagonal.",
            );
        }

        for (i, a) in colours.iter().enumerate() {
            for b in &colours[i + 1..] {
                assert!(
                    a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-6),
                    "two miss reasons share the colour {a:?}",
                );
            }
        }
    }

    fn ssr_miss_colour_source() -> String {
        let code = wgsl_ssr_block_shared_camera_debug(1, true);
        let start = code
            .find("fn ssr_miss_colour(")
            .expect("`ssr_miss_colour` is gone from the debug shader");
        let rest = &code[start..];
        let end = rest.find("\n}").expect("`ssr_miss_colour` has no closing brace");
        rest[..end].to_string()
    }

    fn regex_lite_vec3s(src: &str) -> Vec<[f32; 3]> {
        let mut out = Vec::new();
        let mut rest = src;
        while let Some(i) = rest.find("vec3<f32>(") {
            rest = &rest[i + "vec3<f32>(".len()..];
            let Some(close) = rest.find(')') else { break };
            let parts: Vec<&str> = rest[..close].split(',').map(str::trim).collect();
            if parts.len() == 3 {
                if let (Ok(a), Ok(b), Ok(c)) = (
                    parts[0].parse::<f32>(),
                    parts[1].parse::<f32>(),
                    parts[2].parse::<f32>(),
                ) {
                    out.push([a, b, c]);
                }
            }
            rest = &rest[close..];
        }
        out
    }

    #[test]
    fn the_diagnostic_is_off() {
        assert!(
            !SSR_DEBUG,
            "SSR_DEBUG is on: reflective surfaces render as flat blue/red/green \
             instead of reflections. Set it back to false.",
        );
    }

    #[test]
    fn the_switch_reaches_the_shader() {
        let code = wgsl_ssr_block_shared_camera(3);
        let painted = code.contains("BLUE: too rough to march")
            && code.contains("ssr_miss_colour(exit_reason)")
            && code.contains("GREY: a hit");
        assert_eq!(
            painted, SSR_DEBUG,
            "SSR_DEBUG is {SSR_DEBUG} but the shader {} the diagnostic returns",
            if painted { "has" } else { "does not have" },
        );
    }
}

#[cfg(test)]
mod facing_fade_tests {

    use super::*;

    fn fade(incidence_deg: f32) -> f32 {
        let d = -(2.0 * incidence_deg.to_radians()).cos();
        let t = ((d + 0.7) / 0.5).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    #[test]
    fn the_reflection_and_view_meet_at_twice_the_incidence() {
        for deg in [0.0f32, 15.0, 30.0, 45.0, 60.0, 80.0] {
            let a = deg.to_radians();
            let v = glam::Vec3::new(a.sin(), 0.0, -a.cos());
            let n = glam::Vec3::Z;
            let r = v - 2.0 * v.dot(n) * n;
            let expected = -(2.0 * a).cos();
            assert!((r.dot(v) - expected).abs() < 1e-5, "at {deg} degrees dot is {}", r.dot(v));
        }
    }

    #[test]
    fn a_surface_seen_head_on_takes_no_screen_space_reflection() {
        assert!(fade(0.0) < 1e-6);
        assert!(fade(20.0) < 0.05, "at 20 degrees the fade is {}", fade(20.0));
    }

    #[test]
    fn a_floor_seen_at_a_walking_angle_keeps_its_reflection() {
        for deg in [45.0f32, 55.0, 65.0] {
            assert!(fade(deg) > 0.99, "at {deg} degrees the fade is {}", fade(deg));
        }
    }

    #[test]
    fn the_fade_never_reverses() {
        let mut last = -1.0f32;
        for deg in 0..90 {
            let f = fade(deg as f32);
            assert!(f >= last - 1e-6, "the fade fell from {last} to {f} at {deg} degrees");
            last = f;
        }
    }

    #[test]
    fn the_shader_stands_down_for_rays_toward_the_viewer() {
        let src = wgsl_ssr_block_shared_camera(3);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(code.contains("let facing_fade = smoothstep(-0.7, -0.2, dot(refl_dir, view_dir));"));
        assert!(code.contains("if (facing_fade <= 0.0) {"), "the fully faded case still marches");
        assert!(
            code.contains("edge_fade * grazing_fade * distance_fade * facing_fade * threshold_fade"),
            "the facing fade is computed but not applied",
        );
    }
}

#[cfg(test)]
mod silhouette_tests {

    fn square_fade(dr: f32, ds: f32) -> f32 {
        let squareness = ((dr - ds) / dr.abs().max(1e-9)).clamp(0.0, 1.0);
        let t = ((squareness - 0.05) / (0.35 - 0.05)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    #[test]
    fn it_reads_correctly_while_both_samples_share_a_surface() {
        assert_eq!(square_fade(0.02, 0.0), 1.0, "a square hit lost strength");
        assert!(
            square_fade(0.02, 0.0198) < 0.01,
            "a ray running parallel to what it hit kept {}",
            square_fade(0.02, 0.0198),
        );
    }

    #[test]
    fn it_collapses_wherever_the_second_read_crosses_an_edge() {

        assert_eq!(
            square_fade(0.02, 0.9),
            0.0,
            "a hit whose neighbouring read fell past a silhouette kept its \
             strength; if this ever passes again the black outlines are back",
        );

        assert_eq!(square_fade(0.02, -0.9), 1.0);
    }

    #[test]
    fn the_shader_reads_depth_only_along_the_ray_it_marched() {

        let src = super::wgsl_ssr_block_shared_camera(3);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in ["square_fade", "SSR_SQUARENESS_SPAN", "SSR_THICKNESS_STEPS"] {
            assert!(
                !code.contains(banned),
                "`{banned}` is back in the shader; a fade that reads the depth \
                 buffer away from its own hit draws an outline round every \
                 reflected edge in the room",
            );
        }
        assert!(
            code.contains(
                "let fade = edge_fade * grazing_fade * distance_fade * facing_fade * threshold_fade * budget_fade;"
            ),
            "the fade product changed; every term in it must depend on the \
             surface being shaded, on the REFINED hit, or on how much of the \
             search budget reaching it took -- and on nothing read at a march \
             sample",
        );
    }
}

#[cfg(test)]
mod fade_continuity_twin {

    use glam::{Mat4, Vec2, Vec3, Vec4};

    const STEPS: u32 = 13;
    const STEP_SIZE: f32 = 0.12;
    const STEP_GROWTH: f32 = 1.18;
    const REFINE: u32 = 6;
    const THICKNESS_STEPS: f32 = 2.5;
    const NORMAL_BIAS: f32 = 0.05;
    const TRUST_NEAR: f32 = 0.18;
    const TRUST_FAR: f32 = 0.87;

    const EDGE_FADE_WIDTH: f32 = 0.25;

    const SQUARENESS_SPAN: f32 = 0.08;

    const STRIDE_PIXELS: f32 = 4.0;
    const STRIDE_MAX: f32 = 24.0;
    const MAX_SCREEN_STEPS: u32 = 32;

    const BUDGET_FADE_FROM: f32 = 0.6;

    const MAX_DISTANCE: f32 = 5.07;
    const NEAR_W: f32 = 0.06;

    const THICKNESS_METRES: f32 = 0.25;

    const THICKNESS_SOFT_METRES: f32 = 0.75;

    const PANEL_STANDOFF: f32 = 0.35;
    const PANEL_HALF_WIDTH: f32 = 0.11;
    const PANEL_BOTTOM: f32 = 0.70;
    const PANEL_HEIGHT: f32 = 0.26;

    const VIEWPOINTS: [(f32, f32, f32, f32); 9] = [
        (1.30, 0.6, 0.9, 0.0),
        (1.30, 1.2, 0.7, 0.0),
        (1.60, 1.0, 0.8, 0.0),
        (1.60, 2.0, 0.6, 0.0),
        (1.10, 0.8, 1.0, 0.0),
        (1.60, 4.0, 0.45, 0.0),
        (1.60, 1.5, 0.25, 1.15),
        (1.40, 2.5, 0.15, 1.30),
        (1.60, 1.0, 0.35, 1.40),
    ];

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    enum Design {

        Shipped,

        CrossingExcluded,

        Proposed,

        ProposedNoConfidence,

        ScreenStride,

        ScreenStrideJittered,

        HiZ,
    }

    fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
        let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    struct Cam {
        vp: Mat4,
        inv: Mat4,
        eye: Vec3,

        wall_z: f32,
    }

    fn cam(height: f32, wall_ahead: f32, pitch: f32, yaw: f32) -> Cam {
        let eye = Vec3::new(0.0, height, 0.0);
        let look = Vec3::new(
            yaw.sin() * pitch.cos(),
            -pitch.sin(),
            -yaw.cos() * pitch.cos(),
        );
        let view = Mat4::look_at_rh(eye, eye + look, Vec3::Y);
        let proj = Mat4::perspective_rh(1.2, 1.0, 0.05, 50.0);
        let vp = proj * view;
        Cam { vp, inv: vp.inverse(), eye, wall_z: -wall_ahead }
    }

    impl Cam {
        fn project(&self, p: Vec3) -> Vec4 {
            self.vp * Vec4::new(p.x, p.y, p.z, 1.0)
        }

        fn ray(&self, uv: Vec2) -> Vec3 {
            let p = self.inv * Vec4::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
            (Vec3::new(p.x, p.y, p.z) / p.w - self.eye).normalize()
        }

        fn first_hit(&self, uv: Vec2) -> Option<(Vec3, Vec3)> {
            let d = self.ray(uv);
            let mut best: Option<(f32, Vec3, Vec3)> = None;
            if d.y < -1e-6 {
                let t = -self.eye.y / d.y;
                let p = self.eye + d * t;
                if t > 0.0 && p.z > self.wall_z {
                    best = Some((t, p, Vec3::Y));
                }
            }
            if d.z < -1e-6 {
                let t = (self.wall_z - self.eye.z) / d.z;
                let p = self.eye + d * t;
                if t > 0.0 && p.y > 0.0 && best.is_none_or(|(bt, _, _)| t < bt) {
                    best = Some((t, p, Vec3::Z));
                }
                let t = (self.wall_z + PANEL_STANDOFF - self.eye.z) / d.z;
                let p = self.eye + d * t;
                if t > 0.0
                    && p.x.abs() < PANEL_HALF_WIDTH
                    && p.y > PANEL_BOTTOM
                    && p.y < PANEL_BOTTOM + PANEL_HEIGHT
                    && best.is_none_or(|(bt, _, _)| t < bt)
                {
                    best = Some((t, p, Vec3::Z));
                }
            }
            best.map(|(_, p, n)| (p, n))
        }

        fn is_floor(&self, uv: Vec2) -> bool {
            self.first_hit(uv).is_some_and(|(_, n)| n == Vec3::Y)
        }

        fn scene_ndc_z(&self, uv: Vec2) -> f32 {
            match self.first_hit(uv) {
                Some((p, _)) => {
                    let c = self.project(p);
                    c.z / c.w
                }
                None => 1.0,
            }
        }

        fn ray_and_scene(&self, p: Vec3) -> Option<(f32, f32)> {
            let c = self.project(p);
            if c.w <= 0.0 {
                return None;
            }

            let uv = uv_of(c).clamp(Vec2::ZERO, Vec2::ONE);
            Some((c.z / c.w, self.scene_ndc_z(uv)))
        }

        fn squareness(&self, hit: Vec3, refl: Vec3, metres: f32) -> Option<f32> {
            let (a_ray, a_scene) = self.ray_and_scene(hit - refl * metres)?;
            let (b_ray, b_scene) = self.ray_and_scene(hit + refl * metres)?;
            let dr = b_ray - a_ray;
            let ds = b_scene - a_scene;
            Some(((dr - ds) / dr.abs().max(1e-9)).clamp(0.0, 1.0))
        }
    }

    fn edge_of(uv: Vec2) -> f32 {
        uv.x.min(1.0 - uv.x).min(uv.y).min(1.0 - uv.y)
    }

    fn clip_to_frame(a: Vec4, b: Vec4) -> f32 {
        let e = |c: Vec4| [c.w - c.x, c.w + c.x, c.w - c.y, c.w + c.y, c.w - 1e-4];
        let (ea, eb) = (e(a), e(b));
        let mut t = 1.0f32;
        for j in 0..5 {
            if eb[j] < 0.0 && ea[j] > 0.0 {
                t = t.min(ea[j] / (ea[j] - eb[j]));
            }
        }
        t * 0.999
    }

    fn uv_of(c: Vec4) -> Vec2 {
        Vec2::new(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5)
    }

    struct Marched {

        surface: u8,

        hit_uv: Vec2,
        fade: f32,
        travelled: f32,
        edge: f32,
        grazing: f32,
        distance: f32,
        confidence: f32,
        facing: f32,
    }

    struct Crossing {
        hit_uv: Vec2,
        hit_pos: Vec3,
        origin: Vec3,

        budget_used: f32,

        slab: f32,
    }

    thread_local! {
        static TRACE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    thread_local! {
        static SOFT_SLAB: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    fn soft_slab() -> bool {
        SOFT_SLAB.with(std::cell::Cell::get)
    }

    fn trace(msg: impl FnOnce() -> String) {
        TRACE.with(|t| {
            if t.get() {
                println!("    trace: {}", msg());
            }
        });
    }

    struct Pyramid {
        levels: Vec<Vec<f32>>,
        sizes: Vec<(usize, usize)>,
    }

    impl Pyramid {
        fn build(c: &Cam) -> Self {
            let (w, h) = (EYE_W as usize, EYE_H as usize);
            let mut level0 = vec![1.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let uv = Vec2::new(
                        (x as f32 + 0.5) / EYE_W,
                        (y as f32 + 0.5) / EYE_H,
                    );
                    level0[y * w + x] = c.scene_ndc_z(uv);
                }
            }
            let mut levels = vec![level0];
            let mut sizes = vec![(w, h)];
            while levels.len() < HI_Z_LEVELS {
                let (pw, ph) = *sizes.last().unwrap();
                if pw <= 1 && ph <= 1 {
                    break;
                }
                let (nw, nh) = ((pw / 2).max(1), (ph / 2).max(1));
                let parent = levels.last().unwrap();
                let mut next = vec![1.0f32; nw * nh];
                let (ox, oy) = (pw % 2, ph % 2);
                for y in 0..nh {
                    for x in 0..nw {
                        let mut m = f32::INFINITY;
                        for dy in 0..(2 + oy) {
                            for dx in 0..(2 + ox) {
                                let sx = (x * 2 + dx).min(pw - 1);
                                let sy = (y * 2 + dy).min(ph - 1);
                                m = m.min(parent[sy * pw + sx]);
                            }
                        }
                        next[y * nw + x] = m;
                    }
                }
                levels.push(next);
                sizes.push((nw, nh));
            }
            Self { levels, sizes }
        }

        fn at(&self, level: usize, x: i32, y: i32) -> f32 {
            let (w, h) = self.sizes[level];
            let x = x.clamp(0, w as i32 - 1) as usize;
            let y = y.clamp(0, h as i32 - 1) as usize;
            self.levels[level][y * w + x]
        }
    }

    const HI_Z_LEVELS: usize = 8;

    const HI_Z_START_LEVEL: usize = 2;

    const HI_Z_MAX_ITERATIONS: u32 = 64;

    const CELL_NUDGE_TEXELS: f32 = 0.05;
    const MIN_RAY_TEXELS: f32 = 2.0;

    thread_local! {

        static STALLS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        static WALKS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    fn march_hi_z(c: &Cam, pyramid: &Pyramid, world_pos: Vec3, n: Vec3, refl: Vec3) -> Option<Crossing> {
        let res = Vec2::new(EYE_W, EYE_H);
        let p0 = world_pos + n * NORMAL_BIAS;
        let mut p1 = p0 + refl * MAX_DISTANCE;
        let (h0, mut h1) = (c.project(p0), c.project(p1));
        if h0.w <= NEAR_W {
            return None;
        }
        if h1.w <= NEAR_W {
            let t = (h0.w - NEAR_W) / (h0.w - h1.w);
            p1 = p0.lerp(p1, t);
            h1 = c.project(p1);
        }
        let (k0, k1) = (1.0 / h0.w, 1.0 / h1.w);
        let (z0, z1) = (h0.z * k0, h1.z * k1);
        let (s0, s1) = (uv_of(h0) * res, uv_of(h1) * res);
        let ds = s1 - s0;
        let dz = z1 - z0;

        if ds.length() < MIN_RAY_TEXELS {
            return None;
        }
        let b = if (k1 - k0).abs() < 1e-9 { 0.0 } else { dz / (k1 - k0) };
        let a = z0 - b * k0;
        if b == 0.0 {
            return None;
        }

        let at = |t: f32| s0 + ds * t;

        let mut t = (1.0 / ds.length()).min(1.0);
        let mut level = HI_Z_START_LEVEL.min(pyramid.levels.len() - 1);

        WALKS.with(|w| w.set(w.get() + 1));
        for _ in 0..HI_Z_MAX_ITERATIONS {
            let p = at(t);
            if p.x < 0.0 || p.y < 0.0 || p.x >= EYE_W || p.y >= EYE_H || t > 1.0 {
                return None;
            }
            let cell_size = (1 << level) as f32;
            let cell = (p / cell_size).floor();

            let next_x = if ds.x > 0.0 { (cell.x + 1.0) * cell_size } else { cell.x * cell_size };
            let next_y = if ds.y > 0.0 { (cell.y + 1.0) * cell_size } else { cell.y * cell_size };
            let tx = if ds.x.abs() < 1e-6 { f32::INFINITY } else { (next_x - s0.x) / ds.x };
            let ty = if ds.y.abs() < 1e-6 { f32::INFINITY } else { (next_y - s0.y) / ds.y };

            let t_exit = if tx < ty {
                tx + CELL_NUDGE_TEXELS / ds.x.abs().max(1e-6)
            } else {
                ty + CELL_NUDGE_TEXELS / ds.y.abs().max(1e-6)
            };

            let cell_min = pyramid.at(level, cell.x as i32, cell.y as i32);

            let t_depth = if dz.abs() < 1e-12 { f32::INFINITY } else { (cell_min - z0) / dz };

            let t_next = t.max(t_depth);
            let stayed = (at(t_next) / cell_size).floor() == cell;
            if !stayed {
                t = t_exit;
                level = (level + 1).min(pyramid.levels.len() - 1);
            } else if level == 0 {
                let hit_t = t_next.clamp(0.0, 1.0);
                let hs = at(hit_t);
                let kh = k0 + (k1 - k0) * hit_t;
                let scene_w = b / (pyramid.at(0, hs.x as i32, hs.y as i32) - a);
                let overshoot = 1.0 / kh - scene_w;
                let slab = if soft_slab() {
                    1.0 - smoothstep(THICKNESS_METRES, THICKNESS_SOFT_METRES, overshoot)
                } else {
                    f32::from(overshoot <= THICKNESS_METRES)
                };
                if slab > 0.0 {
                    return Some(Crossing {
                        hit_uv: hs / res,
                        hit_pos: (p0 * k0).lerp(p1 * k1, hit_t) / kh,
                        origin: p0,
                        budget_used: 0.0,
                        slab,
                    });
                }

                t = t_exit;
                level = (level + 1).min(pyramid.levels.len() - 1);
            } else {
                t = t_next;
                level -= 1;
            }
        }
        STALLS.with(|st| st.set(st.get() + 1));
        None
    }

    fn march_screen_stride(c: &Cam, world_pos: Vec3, n: Vec3, refl: Vec3, jitter: f32) -> Option<Crossing> {
        let res = Vec2::new(EYE_W, EYE_H);
        let p0 = world_pos + n * NORMAL_BIAS;
        let mut p1 = p0 + refl * MAX_DISTANCE;
        let (h0, mut h1) = (c.project(p0), c.project(p1));
        if h0.w <= NEAR_W {
            trace(|| "origin behind the near plane".into());
            return None;
        }

        if h1.w <= NEAR_W {
            let t = (h0.w - NEAR_W) / (h0.w - h1.w);
            p1 = p0.lerp(p1, t);
            h1 = c.project(p1);
        }
        let (k0, k1) = (1.0 / h0.w, 1.0 / h1.w);
        let (z0, z1) = (h0.z * k0, h1.z * k1);
        let (s0, s1) = (uv_of(h0) * res, uv_of(h1) * res);

        let b = if (k1 - k0).abs() < 1e-9 { 0.0 } else { (z1 - z0) / (k1 - k0) };
        let a = z0 - b * k0;
        if b == 0.0 {
            trace(|| "the ray lies at a constant depth".into());
            return None;
        }
        let scene_w_at = |uv: Vec2| {
            let z = c.scene_ndc_z(uv);
            let d = z - a;
            if d.abs() < 1e-9 { f32::INFINITY } else { b / d }
        };

        let pixels = (s1 - s0).abs().max_element().max(1e-3);
        let stride = STRIDE_PIXELS.max(pixels / MAX_SCREEN_STEPS as f32).min(STRIDE_MAX);
        let dt = (stride / pixels).min(1.0);
        let mut t_prev = dt * (0.5 + jitter);
        let mut w_prev = 1.0 / (k0 + (k1 - k0) * t_prev);
        let mut t = t_prev + dt;
        for _ in 0..MAX_SCREEN_STEPS {
            if t > 1.0 {
                trace(|| format!("ran to the end of the ray at t {t:.4}, dt {dt:.5}, pixels {pixels:.0}, stride {stride:.1}"));
                break;
            }
            let uv = (s0.lerp(s1, t)) / res;
            if edge_of(uv) <= 0.0 {
                trace(|| format!("left the frame at t {t:.4}, uv ({:.3},{:.3})", uv.x, uv.y));
                return None;
            }
            let w = 1.0 / (k0 + (k1 - k0) * t);
            let scene_w = scene_w_at(uv);

            let (near, far) = (w_prev.min(w), w_prev.max(w));
            if far >= scene_w && near <= scene_w + THICKNESS_METRES {

                let (mut lo, mut hi) = (t_prev, t);
                for _ in 0..REFINE {
                    let mid = f32::midpoint(lo, hi);
                    let muv = (s0.lerp(s1, mid)) / res;
                    let mw = 1.0 / (k0 + (k1 - k0) * mid);
                    if mw >= scene_w_at(muv) { hi = mid } else { lo = mid }
                }
                let hit_uv = (s0.lerp(s1, hi)) / res;
                let kh = k0 + (k1 - k0) * hi;
                let q0 = p0 * k0;
                let q1 = p1 * k1;
                let reachable = (dt * MAX_SCREEN_STEPS as f32).min(1.0);
                return Some(Crossing {
                    hit_uv,
                    hit_pos: q0.lerp(q1, hi) / kh,
                    origin: p0,
                    budget_used: (hi / reachable).clamp(0.0, 1.0),

                    slab: 1.0,
                });
            }
            trace(|| format!(
                "  step t {t:.4} w {w:.3} scene_w {scene_w:.3} (interval {:.3}..{:.3})",
                w_prev.min(w), w_prev.max(w)
            ));
            t_prev = t;
            w_prev = w;
            t += dt;
        }
        trace(|| format!("out of steps: dt {dt:.5}, pixels {pixels:.0}, stride {stride:.1}"));
        None
    }

    fn finish(c: &Cam, n: Vec3, view_dir: Vec3, facing_fade: f32, x: Crossing) -> Marched {
        let edge = edge_of(uv_of(c.project(x.origin))).min(edge_of(uv_of(c.project(x.hit_pos))));
        let grazing_fade = smoothstep(0.15, 0.45, n.dot(-view_dir));
        let travelled = (x.hit_pos - x.origin).length();
        let distance_fade = 1.0 - smoothstep(TRUST_NEAR, TRUST_FAR, travelled / MAX_DISTANCE);
        let budget_fade = 1.0 - smoothstep(BUDGET_FADE_FROM, 1.0, x.budget_used);
        let surface = if (x.hit_pos.z - (c.wall_z + PANEL_STANDOFF)).abs() < 0.02 {
            2
        } else { u8::from((x.hit_pos.z - c.wall_z).abs() < 0.02) };
        Marched {
            surface,
            hit_uv: x.hit_uv,
            fade: smoothstep(0.0, EDGE_FADE_WIDTH, edge)
                * grazing_fade
                * distance_fade
                * facing_fade
                * budget_fade
                * x.slab,
            travelled,
            edge,
            grazing: grazing_fade,
            distance: distance_fade,
            confidence: x.slab,
            facing: facing_fade,
        }
    }

    fn march(c: &Cam, pixel: Vec2, design: Design) -> Option<Marched> {
        march_with(c, pixel, design, None)
    }

    fn march_with(
        c: &Cam,
        pixel: Vec2,
        design: Design,
        pyramid: Option<&Pyramid>,
    ) -> Option<Marched> {
        let (world_pos, n) = c.first_hit(pixel)?;
        let view_dir = (world_pos - c.eye).normalize();
        let refl = view_dir - n * (2.0 * view_dir.dot(n));

        let facing_fade = smoothstep(-0.7, -0.2, refl.dot(view_dir));
        if facing_fade <= 0.0 {
            return None;
        }
        if design == Design::HiZ {
            let x = march_hi_z(c, pyramid?, world_pos, n, refl)?;
            return Some(finish(c, n, view_dir, facing_fade, x));
        }
        if design == Design::ScreenStride || design == Design::ScreenStrideJittered {

            let px = pixel * Vec2::new(EYE_W, EYE_H);

            let jitter = if design == Design::ScreenStrideJittered {
                (52.982_918 * (0.06711056 * px.x + 0.00583715 * px.y).fract()).fract()
            } else {
                0.0
            };
            let x = march_screen_stride(c, world_pos, n, refl, jitter)?;
            return Some(finish(c, n, view_dir, facing_fade, x));
        }

        let mut step = STEP_SIZE;
        #[allow(unused_assignments)]
        let mut p = world_pos + n * NORMAL_BIAS + refl * step * 0.5;
        let mut prev = p;
        let origin = p;
        let mut min_edge = 1.0f32;
        let mut prev_ndc_z = {
            let c0 = c.project(p);
            if c0.w > 0.0 { c0.z / c0.w } else { 1.0 }
        };
        let mut prev_behind = -1.0f32;
        let mut hit = false;
        let mut hit_uv = Vec2::ZERO;
        let mut coarse_thickness = 1.0f32;

        for _ in 0..STEPS {
            prev = p;
            p += refl * step;
            step *= STEP_GROWTH;
            let clip = c.project(p);
            let left_frame = clip.w <= 0.0
                || (clip.x / clip.w).abs() > 1.0
                || (clip.y / clip.w).abs() > 1.0;
            if left_frame {
                if design == Design::Shipped || design == Design::CrossingExcluded {
                    return None;
                }

                let t = clip_to_frame(c.project(prev), clip);
                let q = prev + (p - prev) * t;
                let Some((q_ray, q_scene)) = c.ray_and_scene(q) else {
                    return None;
                };
                if !(q_ray - q_scene > 0.0 && prev_behind <= 0.0) {
                    return None;
                }
                hit = true;
                hit_uv = uv_of(c.project(q));
                p = q;
                break;
            }
            let ndc = Vec2::new(clip.x / clip.w, clip.y / clip.w);
            let uv = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            let sample_ndc_z = clip.z / clip.w;
            let span = (sample_ndc_z - prev_ndc_z).max(1e-7);
            let behind = sample_ndc_z - c.scene_ndc_z(uv);
            if design == Design::Shipped {
                min_edge = min_edge.min(edge_of(uv));
            }
            let crossed = match design {
                Design::Shipped | Design::CrossingExcluded => {
                    behind > 0.0 && behind < span * THICKNESS_STEPS
                }
                Design::Proposed
                | Design::ProposedNoConfidence
                | Design::ScreenStride
                | Design::ScreenStrideJittered
                | Design::HiZ => behind > 0.0 && prev_behind <= 0.0,
            };
            if crossed {
                hit = true;
                hit_uv = uv;
                coarse_thickness =
                    1.0 - smoothstep(0.5, 1.0, behind / (span * THICKNESS_STEPS));
                break;
            }
            if design != Design::Shipped {
                min_edge = min_edge.min(edge_of(uv));
            }
            prev_behind = behind;
            prev_ndc_z = sample_ndc_z;
        }
        if !hit {
            return None;
        }

        let mut lo = prev;
        let mut hi = p;
        let mut hit_pos = p;
        for _ in 0..REFINE {
            let mid = (lo + hi) * 0.5;
            let mc = c.project(mid);
            if mc.w <= 0.0 {
                break;
            }
            let muv = uv_of(mc);
            if c.scene_ndc_z(muv) < mc.z / mc.w {
                hi = mid;
                hit_uv = muv;
                hit_pos = mid;
            } else {
                lo = mid;
            }
        }

        let edge = match design {
            Design::Shipped | Design::CrossingExcluded => min_edge.min(edge_of(hit_uv)),

            Design::Proposed
            | Design::ProposedNoConfidence
            | Design::ScreenStride
            | Design::ScreenStrideJittered
            | Design::HiZ => {
                edge_of(uv_of(c.project(origin))).min(edge_of(uv_of(c.project(hit_pos))))
            }
        };
        let confidence = match design {
            Design::Shipped | Design::CrossingExcluded => coarse_thickness,

            Design::Proposed => match c.squareness(hit_pos, refl, SQUARENESS_SPAN) {
                Some(sq) => smoothstep(0.05, 0.35, sq),
                None => 1.0,
            },
            Design::ProposedNoConfidence
            | Design::ScreenStride
            | Design::ScreenStrideJittered
            | Design::HiZ => 1.0,
        };

        let mut reach = 0.0;
        let mut s = STEP_SIZE;
        for _ in 0..STEPS {
            reach += s;
            s *= STEP_GROWTH;
        }
        let travelled = (hit_pos - origin).length();
        let distance_fade = 1.0 - smoothstep(TRUST_NEAR, TRUST_FAR, travelled / reach);
        let grazing_fade = smoothstep(0.15, 0.45, n.dot(-view_dir));
        let surface = if (hit_pos.z - (c.wall_z + PANEL_STANDOFF)).abs() < 0.02 {
            2
        } else { u8::from((hit_pos.z - c.wall_z).abs() < 0.02) };
        Some(Marched {
            surface,
            hit_uv,
            fade: smoothstep(0.0, EDGE_FADE_WIDTH, edge)
                * grazing_fade
                * distance_fade
                * facing_fade
                * confidence,
            travelled,
            edge,
            grazing: grazing_fade,
            distance: distance_fade,
            confidence,
            facing: facing_fade,
        })
    }

    const EYE_W: f32 = 1680.0;
    const EYE_H: f32 = 1760.0;

    struct Worst {
        jump: f32,
        at: String,
        probes: usize,
    }

    fn measure(design: Design) -> Worst {
        const COLS: usize = 48;
        const ROWS: usize = 48;
        let mut worst = Worst { jump: 0.0, at: String::from("nowhere"), probes: 0 };
        for (height, wall, pitch, yaw) in VIEWPOINTS {
            let c = cam(height, wall, pitch, yaw);
            for r in 0..ROWS {

                let v = 0.50 + 0.49 * (r as f32) / (ROWS as f32);
                for k in 0..COLS {
                    let u = 0.02 + 0.96 * (k as f32) / (COLS as f32);
                    let here = Vec2::new(u, v);
                    let right = Vec2::new(u + 1.0 / EYE_W, v);
                    let down = Vec2::new(v.mul_add(0.0, u), v + 1.0 / EYE_H);
                    if !c.is_floor(here) || !c.is_floor(right) || !c.is_floor(down) {
                        continue;
                    }
                    let Some(m) = march(&c, here, design) else {
                        continue;
                    };
                    worst.probes += 1;

                    let neighbour = |at: Vec2| match march(&c, at, design) {
                        Some(o) if o.surface == m.surface => Some(o.fade),
                        Some(_) => None,
                        None => Some(0.0),
                    };
                    for (label, other) in [("across", neighbour(right)), ("down", neighbour(down))] {
                        let Some(other) = other else {
                            continue;
                        };
                        let jump = (m.fade - other).abs();
                        if jump > worst.jump {
                            worst.jump = jump;
                            worst.at = format!(
                                "eye {height}m, wall {wall}m, pitch {pitch}, yaw {yaw}, \
                                 uv ({u:.3}, {v:.3}), {label}"
                            );
                        }
                    }
                }
            }
        }
        worst
    }

    fn measure_read_jump(design: Design) -> Worst {

        const VISIBLE: f32 = 0.1;
        const COLS: usize = 48;
        const ROWS: usize = 48;
        let mut worst = Worst { jump: 0.0, at: String::from("nowhere"), probes: 0 };
        for (height, wall, pitch, yaw) in VIEWPOINTS {
            let c = cam(height, wall, pitch, yaw);
            for r in 0..ROWS {
                let v = 0.50 + 0.49 * (r as f32) / (ROWS as f32);
                for k in 0..COLS {
                    let u = 0.02 + 0.96 * (k as f32) / (COLS as f32);
                    let here = Vec2::new(u, v);
                    let Some(m) = march(&c, here, design) else {
                        continue;
                    };
                    if m.fade < VISIBLE {
                        continue;
                    }
                    worst.probes += 1;
                    for (label, at) in [
                        ("across", Vec2::new(u + 1.0 / EYE_W, v)),
                        ("down", Vec2::new(u, v + 1.0 / EYE_H)),
                    ] {
                        let Some(o) = march(&c, at, design) else {
                            continue;
                        };

                        if o.surface != m.surface {
                            continue;
                        }
                        let d = (o.hit_uv - m.hit_uv) * Vec2::new(EYE_W, EYE_H);
                        if d.length() > worst.jump {
                            worst.jump = d.length();
                            worst.at = format!(
                                "eye {height}m, wall {wall}m, pitch {pitch}, yaw {yaw}, \
                                 uv ({u:.3}, {v:.3}), {label}"
                            );
                        }
                    }
                }
            }
        }
        worst
    }

    #[test]
    #[ignore = "diagnostic: print the components either side of the worst fade jump"]
    fn dump_the_worst_screen_stride_pair() {
        let w = measure(Design::ScreenStride);
        println!("worst {:.3} at {}", w.jump, w.at);

        for (height, wall, pitch, yaw) in VIEWPOINTS {
            let c = cam(height, wall, pitch, yaw);
            for r in 0..48 {
                let v = 0.50 + 0.49 * (r as f32) / 48.0;
                for k in 0..48 {
                    let u = 0.02 + 0.96 * (k as f32) / 48.0;
                    let here = Vec2::new(u, v);
                    let Some(m) = march(&c, here, Design::ScreenStride) else { continue };
                    for at in [Vec2::new(u + 1.0 / EYE_W, v), Vec2::new(u, v + 1.0 / EYE_H)] {
                        let o = march(&c, at, Design::ScreenStride);
                        let same = o.as_ref().is_none_or(|o| o.surface == m.surface);
                        let of = o.as_ref().map_or(0.0, |o| o.fade);
                        if same && (m.fade - of).abs() > 0.9 {
                            TRACE.with(|t| t.set(true));
                            println!("--- HERE ---");
                            let _ = march(&c, here, Design::ScreenStride);
                            println!("--- NEXT ---");
                            let _ = march(&c, at, Design::ScreenStride);
                            TRACE.with(|t| t.set(false));
                            println!(
                                "eye {height} wall {wall} pitch {pitch} yaw {yaw} uv ({u:.3},{v:.3})\n  \
                                 HERE fade {:.3} edge {:.3} graze {:.3} dist {:.3} face {:.3} travelled {:.3} surf {}\n  \
                                 NEXT {}",
                                m.fade, m.edge, m.grazing, m.distance, m.facing, m.travelled, m.surface,
                                match &o {
                                    Some(o) => format!(
                                        "fade {:.3} edge {:.3} graze {:.3} dist {:.3} face {:.3} travelled {:.3} surf {}",
                                        o.fade, o.edge, o.grazing, o.distance, o.facing, o.travelled, o.surface),
                                    None => "MISS".to_string(),
                                },
                            );
                            return;
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "diagnostic: does the per-pixel jitter show up as a pattern?"]
    fn how_much_does_the_jitter_cost_in_smoothness() {
        for d in [Design::ScreenStride, Design::ScreenStrideJittered, Design::ProposedNoConfidence] {
            let fade = measure(d);
            let read = measure_read_jump(d);
            println!(
                "{d:?}: worst fade jump {:.3} ({} probes), worst read jump {:.0} px",
                fade.jump, fade.probes, read.jump,
            );
        }
    }

    #[test]
    #[ignore = "slow: rasterises a full 1680x1760 depth buffer per viewpoint"]
    fn a_soft_slab_takes_the_cliff_out_of_the_hit_miss_boundary() {

        const CLIFF: f32 = 0.5;

        const RUN: usize = 600;
        let measure_comb = |soft: bool| {
            SOFT_SLAB.with(|c| c.set(soft));
            let (mut cliffs, mut pairs, mut worst, mut reflected) = (0usize, 0usize, 0.0f32, 0usize);
            for (height, wall, pitch, yaw) in VIEWPOINTS {
                let c = cam(height, wall, pitch, yaw);
                let pyramid = Pyramid::build(&c);
                let pyr = Some(&pyramid);

                for r in 0..24 {
                    let v = 0.02 + 0.96 * (r as f32) / 24.0;
                    for k in 0..RUN {
                        let u = 0.05 + (k as f32 - RUN as f32 * 0.5) / EYE_W + 0.45;
                        let here = Vec2::new(u, v);
                        let Some(m) = march_with(&c, here, Design::HiZ, pyr) else {
                            continue;
                        };
                        if m.fade >= 0.1 {
                            reflected += 1;
                        }
                        for at in [
                            Vec2::new(u + 1.0 / EYE_W, v),
                            Vec2::new(u, v + 1.0 / EYE_H),
                        ] {

                            let other = match march_with(&c, at, Design::HiZ, pyr) {
                                Some(o) if o.surface == m.surface => o.fade,
                                Some(_) => continue,
                                None => 0.0,
                            };
                            pairs += 1;
                            let jump = (m.fade - other).abs();
                            worst = worst.max(jump);
                            if jump > CLIFF {
                                cliffs += 1;
                            }
                        }
                    }
                }
            }
            SOFT_SLAB.with(|c| c.set(true));
            (cliffs, pairs, worst, reflected)
        };
        let (hard_cliffs, hard_pairs, hard_worst, hard_hits) = measure_comb(false);
        let (soft_cliffs, soft_pairs, soft_worst, soft_hits) = measure_comb(true);
        println!(
            "hard slab: {hard_cliffs} cliffs in {hard_pairs} neighbouring pairs \
             ({:.2}%), worst jump {hard_worst:.3}, {hard_hits} pixels reflected\n\
             soft slab: {soft_cliffs} cliffs in {soft_pairs} neighbouring pairs \
             ({:.2}%), worst jump {soft_worst:.3}, {soft_hits} pixels reflected",
            100.0 * hard_cliffs as f32 / hard_pairs.max(1) as f32,
            100.0 * soft_cliffs as f32 / soft_pairs.max(1) as f32,
        );
        assert!(
            soft_hits >= hard_hits,
            "the soft slab accepts a SUPERSET of the hard slab's hits, so it \
             cannot reflect fewer pixels: {soft_hits} against {hard_hits}",
        );

        assert!(
            soft_cliffs >= hard_cliffs,
            "the soft slab now REMOVES cliffs ({soft_cliffs} against \
             {hard_cliffs}); it did not when it was measured, so re-read the \
             plan before acting on this",
        );

        assert!(
            hard_cliffs * 200 < hard_pairs,
            "the twin now reproduces the comb at {hard_cliffs} cliffs in \
             {hard_pairs} pairs; it did not before, so it has become able to \
             measure the headset's artefact and the note above is stale",
        );
    }

    #[test]
    #[ignore = "slow: rasterises a full 1680x1760 depth buffer per viewpoint"]
    fn hi_z_reaches_further_than_a_fixed_stride_without_reading_further_apart() {
        let mut stride_hits = 0usize;
        let mut hi_z_hits = 0usize;
        let mut stride_jump = 0.0f32;
        let mut hi_z_jump = 0.0f32;
        for (height, wall, pitch, yaw) in VIEWPOINTS {
            let c = cam(height, wall, pitch, yaw);
            let pyramid = Pyramid::build(&c);
            for r in 0..32 {
                let v = 0.02 + 0.96 * (r as f32) / 32.0;
                for k in 0..32 {
                    let u = 0.02 + 0.96 * (k as f32) / 32.0;
                    let here = Vec2::new(u, v);
                    let right = Vec2::new(u + 1.0 / EYE_W, v);
                    for (design, hits, jump) in [
                        (Design::ScreenStride, &mut stride_hits, &mut stride_jump),
                        (Design::HiZ, &mut hi_z_hits, &mut hi_z_jump),
                    ] {
                        let pyr = Some(&pyramid);
                        let Some(m) = march_with(&c, here, design, pyr) else {
                            continue;
                        };
                        if m.fade < 0.1 {
                            continue;
                        }
                        *hits += 1;
                        if let Some(o) = march_with(&c, right, design, pyr) {
                            if o.surface == m.surface {
                                let d = ((o.hit_uv - m.hit_uv) * Vec2::new(EYE_W, EYE_H)).length();
                                *jump = jump.max(d);
                            }
                        }
                    }
                }
            }
        }
        let (stalls, walks) = (STALLS.with(std::cell::Cell::get), WALKS.with(std::cell::Cell::get));
        println!(
            "stride: {stride_hits} probes reflected, worst read {stride_jump:.0} px\n\
             hi-z:   {hi_z_hits} probes reflected, worst read {hi_z_jump:.0} px\n\
             hi-z walks: {walks}, of which {stalls} used every iteration ({:.1}%)",
            100.0 * stalls as f32 / walks.max(1) as f32,
        );
        assert!(
            stalls * 20 < walks,
            "{stalls} of {walks} rays used every one of their \
             {HI_Z_MAX_ITERATIONS} iterations; the walk is stalling on its own \
             cell boundaries rather than searching",
        );
        assert!(
            hi_z_hits > stride_hits,
            "hi-z found {hi_z_hits} reflections against the stride's \
             {stride_hits}: reach is the whole reason for it",
        );
        assert!(
            hi_z_jump <= stride_jump.max(32.0),
            "hi-z reads {hi_z_jump:.0} px apart against the stride's \
             {stride_jump:.0}: it must not buy its reach back with ghosting",
        );
    }

    #[test]
    fn a_world_space_march_reads_its_reflection_from_all_over_the_place() {
        let world = measure_read_jump(Design::ProposedNoConfidence);
        let stride = measure_read_jump(Design::ScreenStride);
        println!(
            "worst distance between what two NEIGHBOURING pixels read, in eye-buffer pixels:\n  \
             world-space march  {:.0} px  at {}\n  \
             screen-space stride {:.0} px  at {}\n  \
             probes that reflected: world {}, stride {}",
            world.jump, world.at, stride.jump, stride.at, world.probes, stride.probes,
        );
        assert!(
            world.jump > 100.0,
            "the world-space march reads only {:.0} px apart at its worst, so \
             this twin no longer reproduces the ghosting it was written for",
            world.jump,
        );
        assert!(
            stride.jump < world.jump / 4.0,
            "a fixed stride of {STRIDE_PIXELS} px reads {:.0} px apart against \
             the world march's {:.0}: it is meant to be unable to skip \
             anything wider than its own stride",
            stride.jump,
            world.jump,
        );

        assert!(
            stride.probes * 4 >= world.probes * 3,
            "the stride march reflected {} probes against the world march's \
             {}: the search budget is falling even further short than it was",
            stride.probes,
            world.probes,
        );
    }

    #[test]
    fn the_edge_distance_is_monotone_along_a_segment() {
        let c = cam(1.6, 2.0, 0.6, 0.0);
        let mut seed = 0x5eed_u32;
        let mut rand = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let mut cases = 0;
        for _ in 0..600 {
            let a = Vec3::new(rand() * 6.0 - 3.0, rand() * 2.5 + 0.1, -rand() * 3.5 - 0.2);
            let b = Vec3::new(rand() * 6.0 - 3.0, rand() * 2.5 + 0.1, -rand() * 3.5 - 0.2);
            let (ca, cb) = (c.project(a), c.project(b));
            if ca.w <= 0.05 || cb.w <= 0.05 {
                continue;
            }
            cases += 1;
            let ends = edge_of(uv_of(ca)).min(edge_of(uv_of(cb)));
            let mut sampled = f32::INFINITY;
            for k in 0..=256 {
                sampled = sampled.min(edge_of(uv_of(c.project(a.lerp(b, k as f32 / 256.0)))));
            }
            assert!(
                (sampled - ends).abs() < 1e-4,
                "257 samples along the segment found {sampled}, its two ends say \
                 {ends}; if those disagree the endpoint form is not the minimum \
                 and the fade needs the samples after all",
            );
        }
        assert!(cases > 200, "only {cases} segments were in front of the camera");
    }

    #[test]
    fn the_sampled_fades_step_between_neighbouring_pixels() {
        let shipped = measure(Design::Shipped);
        let excluded = measure(Design::CrossingExcluded);
        let proposed = measure(Design::Proposed);
        let no_conf = measure(Design::ProposedNoConfidence);
        println!(
            "  without the confidence term {:.3}  at {}",
            no_conf.jump, no_conf.at,
        );
        println!(
            "worst jump in the final fade between neighbouring pixels:\n  \
             shipped           {:.3}  at {}\n  \
             crossing excluded {:.3}  at {}\n  \
             proposed          {:.3}  at {}\n  \
             probes that reflected: shipped {}, proposed {}",
            shipped.jump,
            shipped.at,
            excluded.jump,
            excluded.at,
            proposed.jump,
            proposed.at,
            shipped.probes,
            proposed.probes,
        );
        assert!(
            shipped.jump > 0.4,
            "the shipped fade jumped only {} between neighbouring pixels, so \
             this twin no longer reproduces the ribbons it was written to \
             explain",
            shipped.jump,
        );
        assert!(
            proposed.jump < shipped.jump * 0.5,
            "the proposed fade jumps {} against the shipped {}: not the \
             improvement this change is for",
            proposed.jump,
            shipped.jump,
        );
        assert!(
            proposed.jump < excluded.jump,
            "dropping the crossing sample ({}) is already as smooth as \
             measuring the segment exactly ({}), which would make the exact \
             form pointless",
            excluded.jump,
            proposed.jump,
        );
        assert!(
            proposed.probes >= shipped.probes,
            "the proposed design reflected {} of the grid's pixels against {}: \
             a smoother fade bought by dropping reflections is not a fix",
            proposed.probes,
            shipped.probes,
        );
    }

    #[test]
    #[ignore = "diagnostic: prints the fade's parts around the worst neighbour pair"]
    fn dump_the_worst_pair() {
        let c = cam(1.1, 0.8, 1.0, 0.0);
        for k in -6..7 {
            let u = 0.200 + (k as f32) / EYE_W;
            let v = 0.541;
            let pixel = Vec2::new(u, v);
            match march(&c, pixel, Design::Proposed) {
                Some(m) => println!(
                    "u {u:.5} fade {:.3}  edge {:.3} graze {:.3} dist {:.3} conf {:.3} face {:.3} travelled {:.3}",
                    m.fade, m.edge, m.grazing, m.distance, m.confidence, m.facing, m.travelled
                ),
                None => println!("u {u:.5} MISS (floor {})", c.is_floor(pixel)),
            }
        }
    }

    #[test]
    fn the_twin_uses_the_shipped_constants() {
        let code = super::wgsl_ssr_block_shared_camera_debug(3, false);
        for (name, value) in [
            ("SSR_STEPS", format!("{MAX_SCREEN_STEPS}u")),
            ("SSR_STRIDE_PIXELS", format!("{STRIDE_PIXELS:?}")),
            ("SSR_STRIDE_MAX", format!("{STRIDE_MAX:?}")),
            ("SSR_BUDGET_FADE_FROM", format!("{BUDGET_FADE_FROM:?}")),
            ("SSR_MAX_DISTANCE", format!("{MAX_DISTANCE:?}")),
            ("SSR_NEAR_W", format!("{NEAR_W:?}")),
            ("SSR_THICKNESS_METRES", format!("{THICKNESS_METRES:?}")),
            ("SSR_REFINE_STEPS", format!("{REFINE}u")),
            ("SSR_NORMAL_BIAS", format!("{NORMAL_BIAS:?}")),
            ("SSR_TRUST_NEAR", format!("{TRUST_NEAR:?}")),
            ("SSR_TRUST_FAR", format!("{TRUST_FAR:?}")),
        ] {
            let decl = format!("const {name}: ");
            let line = code
                .lines()
                .find(|l| l.trim_start().starts_with(&decl))
                .unwrap_or_else(|| panic!("{name} is gone from the shader"));
            assert!(
                line.contains(&value),
                "the twin mirrors {name} = {value} but the shader says `{line}`",
            );
        }
        assert!(
            code.contains(&format!("smoothstep(0.0, {EDGE_FADE_WIDTH:?}, edge)")),
            "the edge fade is no longer smoothstep(0.0, {EDGE_FADE_WIDTH:?}, edge), so \
             the twin is measuring a curve the shader does not use",
        );
    }
}
