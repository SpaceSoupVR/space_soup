use super::lights::wgsl_lights_block;
use super::mesh::{MeshVertex, SkinnedMeshVertex, MAX_SKIN_JOINTS};
use super::pipeline::lightmap_bind_group_layout;
use wgpu::*;

pub struct MeshPipeline {
    pub pipeline: RenderPipeline,

    pub thin_pipeline: RenderPipeline,

    pub opaque: Option<RenderPipeline>,
    pub texture_layout: BindGroupLayout,
    pub model_layout: BindGroupLayout,
    pub lightmap_layout: BindGroupLayout,
}

pub fn opaque_target(format: TextureFormat) -> [Option<ColorTargetState>; 1] {
    [Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::ALL })]
}

impl MeshPipeline {
    pub fn new(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout) -> Self {
        Self::new_with_front_face(device, format, camera_layout, FrontFace::Ccw, 1, crate::renderer::multiview::ViewMode::Mono)
    }

    pub fn new_multisampled_stereo(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(device, format, camera_layout, FrontFace::Ccw, samples, crate::renderer::multiview::ViewMode::Stereo)
    }

    pub fn new_multisampled(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {

        Self::new_edited(
            device, format, camera_layout, FrontFace::Ccw, samples, crate::renderer::multiview::ViewMode::Mono, true,
            &|src| src, ("mesh_pipeline", "mesh_thin_pipeline"), true,
        )
    }

    pub fn new_mirror(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout) -> Self {
        Self::new_with_front_face(device, format, camera_layout, FrontFace::Cw, 1, crate::renderer::multiview::ViewMode::Mono)
    }

    pub fn new_multisampled_thin_sampled(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_full(device, format, camera_layout, FrontFace::Ccw, samples, crate::renderer::multiview::ViewMode::Mono, false)
    }

    fn new_with_front_face(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        Self::new_full(device, format, camera_layout, front_face, samples, view, true)
    }

    fn new_full(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        thin_mean: bool,
) -> Self {
        Self::new_edited(device, format, camera_layout, front_face, samples, view, thin_mean, &|src| src, ("mesh_pipeline", "mesh_thin_pipeline"), false)
    }

    pub fn new_multisampled_spotless(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout, samples: u32) -> Self {
        Self::new_edited(
            device,
            format,
            camera_layout,
            FrontFace::Ccw,
            samples,
            crate::renderer::multiview::ViewMode::Mono,
            true,
            &crate::renderer::lights::without_spot_shadows,
            ("mesh_pipeline_spotless", "mesh_thin_pipeline_spotless"),
            true,
        )
    }

    pub fn log_register_cuts(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout, samples: u32) {
        let base = mesh_shader();
        for (label, edits) in MESH_REGISTER_CUTS {
            if let Some((from, _)) = edits.iter().find(|(from, _)| !base.contains(from)) {
                log::warn!("register cut {label}: `{from}` is not in the shader");
                continue;
            }
            let edit = |mut src: String| {
                for (from, to) in edits.iter() {
                    src = src.replacen(from, to, 1);
                }
                src
            };
            let thin_label = format!("{label}_thin");
            let _ = Self::new_edited(
                device,
                format,
                camera_layout,
                FrontFace::Ccw,
                samples,
                crate::renderer::multiview::ViewMode::Mono,
                true,
                &edit,
                (label, &thin_label),
                false,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn new_edited(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        thin_mean: bool,
        edit: &dyn Fn(String) -> String,
        (label, thin_label): (&str, &str),
        opaque_twin: bool,
    ) -> Self {

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mesh_shader"),
            source: ShaderSource::Wgsl(crate::renderer::shader_precision::for_device(device, view.shader(edit(mesh_shader()))).into()),
        });

        let texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mesh_texture_bgl"),
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
            ],
        });

        let model_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mesh_model_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,

                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let lightmap_layout = lightmap_bind_group_layout(device);

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("mesh_layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&model_layout), Some(&texture_layout), Some(&lightmap_layout)],
            immediate_size: 0,
        });

        let build_with = |label: &str, shader: &ShaderModule, buffers: &[Option<VertexBufferLayout>], blend: Option<BlendState>| device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers,
            },
            fragment: Some(FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend,
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
        let build = |label: &str, shader: &ShaderModule, buffers: &[Option<VertexBufferLayout>]| {
            build_with(label, shader, buffers, Some(BlendState::ALPHA_BLENDING))
        };
        let pipeline = build(label, &shader, &[Some(MeshVertex::layout())]);
        let opaque = opaque_twin.then(|| build_with(&format!("{label}_opaque"), &shader, &[Some(MeshVertex::layout())], None));
        let thin_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mesh_thin_shader"),
            source: ShaderSource::Wgsl(
                crate::renderer::shader_precision::for_device(device, view.shader(edit(mesh_shader_variant_with(true, thin_mean)))).into(),
            ),
        });
        let thin_pipeline = build(
            thin_label,
            &thin_shader,
            &[Some(MeshVertex::layout()), Some(super::mesh::ThinVertex::layout())],
        );

        Self {
            pipeline,
            thin_pipeline,
            opaque,
            texture_layout,
            model_layout,
            lightmap_layout,
        }
    }

    pub fn create_model_uniform(&self, device: &Device) -> ModelUniform {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("mesh_model_uniform"),

            size: MODEL_UNIFORM_SIZE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("mesh_model_bg"),
            layout: &self.model_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        ModelUniform { buffer, bind_group, last_plain: std::sync::Mutex::new(None) }
    }
}

pub const MODEL_UNIFORM_SIZE: u64 = 288;

pub const MODEL_WEATHER_OFFSET: u64 = 272;

pub struct ModelUniform {
    pub buffer: Buffer,
    pub bind_group: BindGroup,

    pub last_plain: std::sync::Mutex<Option<[f32; 16]>>,
}

impl ModelUniform {

    pub fn set_wetness(&self, queue: &Queue, wet: f32) {
        queue.write_buffer(&self.buffer, MODEL_WEATHER_OFFSET, bytemuck::cast_slice(&[wet.clamp(0.0, 1.0)]));
    }

    pub fn set_highlight(&self, queue: &Queue, level: f32) {
        queue.write_buffer(&self.buffer, MODEL_WEATHER_OFFSET + 4, bytemuck::cast_slice(&[level.clamp(0.0, 1.0)]));
    }

    pub fn upload(&self, queue: &Queue, model: glam::Mat4) {
        let cols = model.to_cols_array();
        if let Ok(last) = self.last_plain.lock() {
            if *last == Some(cols) {
                return;
            }
        }
        self.upload_with_sky(queue, model, 1.0);
        if let Ok(mut last) = self.last_plain.lock() {
            *last = Some(cols);
        }
    }

    pub fn upload_with_sky(&self, queue: &Queue, model: glam::Mat4, sky_vis: f32) {
        self.upload_full(queue, model, sky_vis, 0.0);
    }

    pub fn upload_full(
        &self,
        queue: &Queue,
        model: glam::Mat4,
        sky_vis: f32,
        emissive_drive: f32,
    ) {
        self.upload_lit(queue, model, sky_vis, emissive_drive, &[[0.0; 3]; 9]);
    }

    pub fn upload_lit(
        &self,
        queue: &Queue,
        model: glam::Mat4,
        sky_vis: f32,
        emissive_drive: f32,
        room: &crate::renderer::room_light::RoomLight,
    ) {
        self.upload_lit_bulb(queue, model, sky_vis, emissive_drive, room, None, 0.0);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload_lit_bulb(
        &self,
        queue: &Queue,
        model: glam::Mat4,
        sky_vis: f32,
        emissive_drive: f32,
        room: &crate::renderer::room_light::RoomLight,
        own: Option<&crate::renderer::Light>,
        thin_width: f32,
    ) {
        self.upload_lit_bulb_scaled(queue, model, sky_vis, emissive_drive, room, own, thin_width, 1.0);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upload_lit_bulb_scaled(
        &self,
        queue: &Queue,
        model: glam::Mat4,
        sky_vis: f32,
        emissive_drive: f32,
        room: &crate::renderer::room_light::RoomLight,
        own: Option<&crate::renderer::Light>,
        thin_width: f32,
        own_scale: f32,
    ) {
        if let Ok(mut last) = self.last_plain.lock() {
            *last = None;
        }
        let mut data = [0f32; (MODEL_WEATHER_OFFSET / 4) as usize];
        data[..16].copy_from_slice(&model.to_cols_array());
        data[16] = sky_vis.clamp(0.0, 1.0);
        data[17] = emissive_drive.max(0.0);
        data[19] = thin_width.max(0.0);
        for (i, c) in room.iter().enumerate() {
            data[20 + 4 * i..23 + 4 * i].copy_from_slice(c);
        }
        let spot = own.filter(|l| l.kind == crate::renderer::LightKind::Spot);
        if let Some((l, channel)) = spot.and_then(|l| Some((l, l.mask_channel?))) {
            let (cos_outer, cos_inner) = l.cone_cosines();
            let c = l.color.to_linear();
            let k = own_scale.clamp(0.0, 1.0);
            data[17] *= k;

            data[18] = 2.0 + channel as f32;
            data[56..60].copy_from_slice(&[l.position.x, l.position.y, l.position.z, l.range]);
            data[60..64].copy_from_slice(&[l.direction.x, l.direction.y, l.direction.z, cos_outer]);
            let i = l.intensity * k;
            data[64..68].copy_from_slice(&[c[0] * i, c[1] * i, c[2] * i, cos_inner]);
        }
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&data));
    }
}

const TENT_CALL: &str = "    return pcf_tile_tent_lean_at(\n        tex,";

const CHOOSE_SCAN: &str = "    let near = probe_nearest_two(select_world, room, room);
    c.best = near.s0;
    c.second = near.s1;
    c.best_dist = select(1e30, near.d0, near.s0 >= 0);
    c.second_dist = select(1e30, near.d1, near.s1 >= 0);";

const CHOOSE_WALK: &str = "    for (var i = probe_room_slot(room); i >= 0; i = probe_slot_next(i)) {
        let to_centre = camera.probe_boxes[i * 3].xyz - select_world;
        let dist = dot(to_centre, to_centre);
        if (dist < c.best_dist) {
            c.second = c.best;
            c.second_dist = c.best_dist;
            c.best_dist = dist;
            c.best = i;
        } else if (dist < c.second_dist) {
            c.second = i;
            c.second_dist = dist;
        }
    }";

const VOLUME_SCAN: &str = "    let pick = probe_nearest_two(select_world, room, room).s0;";

const VOLUME_WALK: &str = "    var pick = -1;
    var pick_dist = 1e30;
    for (var i = probe_room_slot(room); i >= 0; i = probe_slot_next(i)) {
        let to_centre = camera.probe_boxes[i * 3].xyz - select_world;
        let dist = dot(to_centre, to_centre);
        if (dist < pick_dist) {
            pick = i;
            pick_dist = dist;
        }
    }";

const GRAB_GLOW_BRANCH: &str = "    if (grab_glow > 0.0) {
        out = grab_highlight(out, n, in.world_pos, grab_glow);
    }";

const MESH_REGISTER_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("mesh_cut_none", &[]),

    ("mesh_no_grab_glow", &[(GRAB_GLOW_BRANCH, "")]),
    ("mesh_box_spots", &[(TENT_CALL, "    return pcf_tile_at(\n        tex,")]),

    ("mesh_tent_loop", &[(TENT_CALL, "    return pcf_tile_tent_at(\n        tex,")]),
    ("mesh_spotless", &[("const SPOT_SHADOWS: bool = true;", "const SPOT_SHADOWS: bool = false;")]),
    ("mesh_choose_scan", &[(CHOOSE_WALK, CHOOSE_SCAN), (VOLUME_WALK, VOLUME_SCAN)]),
    (
        "mesh_scan_loop",
        &[(TENT_CALL, "    return pcf_tile_tent_at(\n        tex,"), (CHOOSE_WALK, CHOOSE_SCAN), (VOLUME_WALK, VOLUME_SCAN)],
    ),
    (
        "mesh_scan_box",
        &[(TENT_CALL, "    return pcf_tile_at(\n        tex,"), (CHOOSE_WALK, CHOOSE_SCAN), (VOLUME_WALK, VOLUME_SCAN)],
    ),
];

pub struct SkinnedMeshPipeline {
    pub pipeline: RenderPipeline,
    pub texture_layout: BindGroupLayout,
    pub model_layout: BindGroupLayout,
    pub skin_joint_layout: BindGroupLayout,

    layout: PipelineLayout,
    format: TextureFormat,
    samples: u32,
    view: crate::renderer::multiview::ViewMode,

    pub body_twins: Option<[RenderPipeline; 2]>,

    pub opaque_twins: Option<[RenderPipeline; 2]>,

    pub prepass: Option<[RenderPipeline; 2]>,
}

pub(crate) fn skinned_body_twin(src: &str, wet: bool) -> Option<String> {
    let edit = if wet {
        ("const PROBE_NOWHERE: bool = false;", "const PROBE_NOWHERE: bool = true;")
    } else {
        ("    if (wet > 0.0) {", "    if (false) {")
    };
    let classic = ("    if (rough >= 0.996) {", "    if (true) {");
    (src.matches(edit.0).count() == 1 && src.matches(classic.0).count() == 1)
        .then(|| src.replacen(edit.0, edit.1, 1).replacen(classic.0, classic.1, 1))
}

pub static SKINNED_FILM_TRACED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl SkinnedMeshPipeline {
    pub fn new(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout) -> Self {
        Self::new_multisampled(device, format, camera_layout, 1)
    }

    pub fn new_multisampled(

        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, camera_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    pub fn new_multisampled_stereo(

        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, camera_layout, samples, crate::renderer::multiview::ViewMode::Stereo)
    }

    fn new_with_view(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("skinned_mesh_shader"),
            source: ShaderSource::Wgsl(crate::renderer::shader_precision::for_device(device, view.shader(skinned_mesh_shader())).into()),
        });

        let texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("skinned_mesh_texture_bgl"),
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
            ],
        });

        let model_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("skinned_mesh_model_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,

                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let skin_joint_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("skin_joint_bgl"),
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
            label: Some("skinned_mesh_layout"),
            bind_group_layouts: &[
                Some(camera_layout),
                Some(&model_layout),
                Some(&texture_layout),
                Some(&skin_joint_layout)],
            immediate_size: 0,
        });

        let pipeline = Self::build(device, &layout, &shader, "skinned_mesh_pipeline", format, samples, view, Some(BlendState::ALPHA_BLENDING));

        Self {
            pipeline,
            texture_layout,
            model_layout,
            skin_joint_layout,
            layout,
            format,
            samples,
            view,
            body_twins: None,
            opaque_twins: None,
            prepass: None,
        }
    }

    pub fn with_body_twins(mut self, device: &Device) -> Self {
        let src = skinned_mesh_shader();

        let twin = |wet: bool, label: &str| {
            let edited = skinned_body_twin(&src, wet)?;
            let shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some(label),
                source: ShaderSource::Wgsl(crate::renderer::shader_precision::for_device(device, self.view.shader(edited)).into()),
            });
            let blended = Self::build(device, &self.layout, &shader, label, self.format, self.samples, self.view, Some(BlendState::ALPHA_BLENDING));
            let opaque = Self::build(device, &self.layout, &shader, &format!("{label}_opaque"), self.format, self.samples, self.view, None);
            Some((blended, opaque))
        };
        match (twin(false, "skinned_mesh_dry"), twin(true, "skinned_mesh_wet_lean")) {
            (Some((dry, dry_opaque)), Some((wet, wet_opaque))) => {
                self.body_twins = Some([dry, wet]);
                self.opaque_twins = Some([dry_opaque, wet_opaque]);
            }
            _ => log::warn!("skinned body twins: the shader's edits no longer match; the body draws with the one shader"),
        }

        if let Some(dry) = skinned_body_twin(&src, false) {
            let shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("skinned_body_prepass"),
                source: ShaderSource::Wgsl(crate::renderer::shader_precision::for_device(device, self.view.shader(dry)).into()),
            });
            let depth = |label: &str, entry: &str| {
                Self::build_entry(device, &self.layout, &shader, label, self.format, self.samples, self.view, None, entry, ColorWrites::empty())
            };
            self.prepass = Some([
                depth("skinned_body_prepass", "fs_depth"),
                depth("skinned_body_prepass_texels", "fs_depth_opaque_texels"),
            ]);
        }
        self
    }

    pub fn for_opaque_body(&self, wet: bool) -> Option<&RenderPipeline> {
        match &self.opaque_twins {
            Some([dry, _]) if !wet => Some(dry),
            Some([_, lean]) if !SKINNED_FILM_TRACED.load(std::sync::atomic::Ordering::Relaxed) => Some(lean),
            _ => None,
        }
    }

    pub fn for_body(&self, wet: bool) -> &RenderPipeline {
        match &self.body_twins {
            Some([dry, _]) if !wet => dry,
            Some([_, lean]) if !SKINNED_FILM_TRACED.load(std::sync::atomic::Ordering::Relaxed) => lean,
            _ => &self.pipeline,
        }
    }

    fn build(
        device: &Device,
        layout: &PipelineLayout,
        shader: &ShaderModule,
        label: &str,
        format: TextureFormat,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        blend: Option<BlendState>,
    ) -> RenderPipeline {
        Self::build_entry(device, layout, shader, label, format, samples, view, blend, "fs_main", ColorWrites::ALL)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_entry(
        device: &Device,
        layout: &PipelineLayout,
        shader: &ShaderModule,
        label: &str,
        format: TextureFormat,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        blend: Option<BlendState>,
        fragment_entry: &str,
        write_mask: ColorWrites,
    ) -> RenderPipeline {
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(SkinnedMeshVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: shader,
                entry_point: Some(fragment_entry),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend,
                    write_mask,
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

                depth_compare: Some(CompareFunction::LessEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        })
    }

    pub fn create_model_uniform(&self, device: &Device) -> ModelUniform {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("skinned_mesh_model_uniform"),

            size: MODEL_UNIFORM_SIZE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("skinned_mesh_model_bg"),
            layout: &self.model_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        ModelUniform { buffer, bind_group, last_plain: std::sync::Mutex::new(None) }
    }
}

fn skinned_mesh_shader() -> String {
    format!(
        r#"
struct ModelUniform {{ model: mat4x4<f32>, params: vec4<f32>, room: array<vec4<f32>, 9>, own: array<vec4<f32>, 3>, weather: vec4<f32> }}
@group(1) @binding(0) var<uniform> model_u: ModelUniform;
{room_light}

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;
@group(2) @binding(2) var emissive_tex: texture_2d<f32>;

struct JointMatrices {{ mats: array<mat4x4<f32>, {max_skin_joints}> }}
@group(3) @binding(0) var<uniform> joints: JointMatrices;

{lights_block}

fn unpack_emissive(packed: u32) -> vec3<f32> {{
    let r = f32((packed >> 0u) & 255u) / 255.0;
    let g = f32((packed >> 8u) & 255u) / 255.0;
    let b = f32((packed >> 16u) & 255u) / 255.0;
    let c = vec3<f32>(r, g, b);
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    let hue = select(hi, lo, c <= vec3<f32>(0.04045));

    let scale = (f32((packed >> 24u) & 255u) / 255.0) * {max_strength:?};
    return hue * scale;
}}

struct VIn {{
    @location(0) position:      vec3<f32>,
    @location(1) normal:        vec3<f32>,
    @location(2) uv:            vec2<f32>,
    @location(3) joint_ids:     vec4<u32>,
    @location(4) joint_weights: vec4<f32>,
}}

struct VOut {{

    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv:     vec2<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) room_near: f32,
}}

@vertex
fn vs_main(v: VIn) -> VOut {{
    let p = vec4<f32>(v.position, 1.0);
    let n = vec4<f32>(v.normal, 0.0);

    let skinned_p =
        (joints.mats[v.joint_ids.x] * p) * v.joint_weights.x +
        (joints.mats[v.joint_ids.y] * p) * v.joint_weights.y +
        (joints.mats[v.joint_ids.z] * p) * v.joint_weights.z +
        (joints.mats[v.joint_ids.w] * p) * v.joint_weights.w;

    let skinned_n =
        (joints.mats[v.joint_ids.x] * n) * v.joint_weights.x +
        (joints.mats[v.joint_ids.y] * n) * v.joint_weights.y +
        (joints.mats[v.joint_ids.z] * n) * v.joint_weights.z +
        (joints.mats[v.joint_ids.w] * n) * v.joint_weights.w;

    let world_pos = model_u.model * skinned_p;
    var out: VOut;
    out.clip      = cam_view_proj() * world_pos;
    out.normal    = (model_u.model * skinned_n).xyz;
    out.uv        = v.uv;
    out.world_pos = world_pos.xyz;
    out.room_near = room_near(to_world_space(world_pos.xyz));
    return out;
}}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let n = normalize(in.normal);
    if (in.room_near > 0.0) {{
        local_exposure = room_local_exposure(to_world_space(in.world_pos));
    }}

    capsule_receiver = false;

    let rough = textureSample(emissive_tex, samp, in.uv).a;
    let tex_color = textureSample(tex, samp, in.uv);

    if (tex_color.a < 0.35) {{
        discard;
    }}

    var shaded: vec3<f32>;
    if (rough >= 0.996) {{
        shaded = tex_color.rgb * (shade_with_sky(in.world_pos, n, model_u.params.x) + room_irradiance(n));
    }} else {{
        shaded = shade_material_env(
            in.world_pos, n, rough, 1.0, model_u.params.x, room_irradiance(n), vec4<f32>(0.5, 0.5, 0.5, 0.0),
            tex_color.rgb, in.world_pos, n,
        );
    }}

    let wet = model_u.weather.x;
    var rgb = shaded * (1.0 - 0.4 * wet);
    if (wet > 0.0) {{
        rgb = rgb + wet * shade_material_env(
            in.world_pos, n, 0.3, 1.0, model_u.params.x, vec3<f32>(0.0), vec4<f32>(0.5, 0.5, 0.5, 0.0),
            vec3<f32>(0.0), in.world_pos, n,
        );
    }}
    return vec4<f32>(tonemap(rgb), tex_color.a);
}}

@fragment
fn fs_depth() -> @location(0) vec4<f32> {{
    return vec4<f32>(0.0);
}}

@fragment
fn fs_depth_opaque_texels(in: VOut) -> @location(0) vec4<f32> {{
    if (textureSample(tex, samp, in.uv).a < 0.999) {{
        discard;
    }}
    return vec4<f32>(0.0);
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
        max_strength = super::mesh::MeshVertex::MAX_EMISSIVE_STRENGTH,
        max_skin_joints = MAX_SKIN_JOINTS,
        room_light = wgsl_room_irradiance(),
    )
}

fn wgsl_room_irradiance() -> &'static str {
    r#"
fn room_irradiance(n: vec3<f32>) -> vec3<f32> {
    let x = n.x; let y = n.y; let z = n.z;
    var e = model_u.room[0].rgb * 0.282095;
    e = e + model_u.room[1].rgb * (0.488603 * y) * 0.6666667;
    e = e + model_u.room[2].rgb * (0.488603 * z) * 0.6666667;
    e = e + model_u.room[3].rgb * (0.488603 * x) * 0.6666667;
    e = e + model_u.room[4].rgb * (1.092548 * x * y) * 0.25;
    e = e + model_u.room[5].rgb * (1.092548 * y * z) * 0.25;
    e = e + model_u.room[6].rgb * (0.315392 * (3.0 * z * z - 1.0)) * 0.25;
    e = e + model_u.room[7].rgb * (1.092548 * x * z) * 0.25;
    e = e + model_u.room[8].rgb * (0.546274 * (x * x - y * y)) * 0.25;
    return max(e, vec3<f32>(0.0));
}
"#
}

fn mesh_shader() -> String {
    mesh_shader_variant(false)
}

const THIN_VS: &str = "
    out.fade = 1.0;
    let thin_axis = (model_u.model * vec4<f32>(v.thin.xyz, 0.0)).xyz;
    let thin_scale = length(thin_axis);
    let thin_r = v.thin.w * thin_scale;
    out.thin = vec4<f32>((model_u.model * vec4<f32>(v.normal, 0.0)).xyz, 0.0);
    if (thin_r > 0.0 && model_u.params.w > 0.0) {
        let depth = (cam_view_proj() * world_pos).w;
        let half_drawn = thin_r + 1.7071068 * model_u.params.w * depth;
        world_pos = vec4<f32>(world_pos.xyz + thin_axis / thin_scale * (half_drawn - thin_r), 1.0);
        out.fade = thin_r / half_drawn;
        out.thin = vec4<f32>(thin_axis / thin_scale, thin_r);
    }";

const THIN_SHARE: &str = "
fn thin_share(fade: f32, thin: vec4<f32>, world_pos: vec3<f32>) -> f32 {
    let v = normalize(cam_pos() - world_pos);
    let dnx = dpdx(thin.xyz);
    let dny = dpdy(thin.xyz);
    let turn = select(dny, dnx, dot(dnx, dnx) > dot(dny, dny));
    let way = turn - v * dot(v, turn);
    let way_len = length(way);
    let flat_across = length(thin.xyz - v * dot(v, thin.xyz));
    let across = select(flat_across, abs(dot(thin.xyz, way)) / max(way_len, 1e-9), way_len > 1e-6);
    let sx = dpdx(world_pos);
    let sy = dpdy(world_pos);
    let fragment = max(0.5 * (length(sx - v * dot(sx, v)) + length(sy - v * dot(sy, v))), 1e-9);
    let d = across * thin.w / max(fade, 1e-4) / fragment;
    let r = thin.w / fragment;
    return clamp(thin_kernel_below(d + r) - thin_kernel_below(d - r), 0.0, 1.0);
}

fn thin_kernel_below(x: f32) -> f32 {
    let a = abs(x);
    let s = max(1.7071068 - a, 0.0);
    let beyond = select(0.5 - 0.5 * a, 0.17677670 * s * s, a >= 0.29289322);
    return select(beyond, 1.0 - beyond, x > 0.0);
}
";

const THIN_SHADE: &str = "
struct ThinSpan {
    c: vec3<f32>,
    b: vec3<f32>,
    t: vec3<f32>,

    ua: f32,
    ub: f32,
    ta: f32,
    tb: f32,
}

fn thin_span_at(fade: f32, thin: vec4<f32>, world_pos: vec3<f32>) -> ThinSpan {
    let v = normalize(cam_pos() - world_pos);
    let dnx = dpdx(thin.xyz);
    let dny = dpdy(thin.xyz);
    let turn = select(dny, dnx, dot(dnx, dnx) > dot(dny, dny));
    let sx = dpdx(world_pos);
    let sy = dpdy(world_pos);
    let fragment = max(0.5 * (length(sx - v * dot(sx, v)) + length(sy - v * dot(sy, v))), 1e-9);
    var axis = cross(thin.xyz, turn);
    if (dot(axis, axis) < 1e-12) {
        axis = cross(thin.xyz, select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(thin.x) > 0.7));
    }
    var s: ThinSpan;
    s.t = normalize(axis);
    let v_square = v - s.t * dot(v, s.t);
    let v_square_len = length(v_square);
    s.c = select(normalize(thin.xyz), v_square / max(v_square_len, 1e-9), v_square_len > 1e-4);
    s.b = cross(s.t, s.c);

    let u = dot(thin.xyz, s.b) / max(fade, 1e-4);
    let reach = 1.7071068 * fragment / max(thin.w, 1e-9);
    var ua = clamp(u - reach, -1.0, 1.0);
    var ub = clamp(u + reach, -1.0, 1.0);

    if (ub - ua < 2e-3) {
        let m = clamp(0.5 * (ua + ub), -0.999, 0.999);
        ua = m - 1e-3;
        ub = m + 1e-3;
    }
    s.ua = ua;
    s.ub = ub;
    s.ta = asin(ua);
    s.tb = asin(ub);
    return s;
}

fn thin_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return sign(x) * (1.0 - p * exp(-a * a));
}

fn thin_mean_light(s: ThinSpan, l: vec3<f32>, h: vec3<f32>, shininess: f32) -> vec2<f32> {
    let lc = dot(l, s.c);
    let lb = dot(l, s.b);
    let phi = atan2(lb, lc);
    let a = max(s.ta, phi - 1.5707963);
    let z = min(s.tb, phi + 1.5707963);
    if (z <= a) {
        return vec2<f32>(0.0);
    }
    let w = s.ub - s.ua;
    let diffuse = sqrt(lc * lc + lb * lb) * 0.5 * ((z - a) * cos(phi) + cos(a + z - phi) * sin(z - a)) / w;
    let hc = dot(h, s.c);
    let hb = dot(h, s.b);
    let th = atan2(hb, hc);
    let k = sqrt(0.5 * shininess);
    let mass = 0.5 * (thin_erf((z - th) * k) - thin_erf((a - th) * k));
    let highlight = pow(max(hc * hc + hb * hb, 0.0), 0.5 * shininess) * sqrt(6.2831853 / shininess) * mass * cos(clamp(th, a, z)) / w;
    return vec2<f32>(diffuse, highlight);
}

struct ThinMoments {
    n: vec3<f32>,

    diag: vec3<f32>,

    off: vec3<f32>,
}
fn thin_moments(s: ThinSpan) -> ThinMoments {
    let w = s.ub - s.ua;
    let ca = sqrt(max(1.0 - s.ua * s.ua, 0.0));
    let cb = sqrt(max(1.0 - s.ub * s.ub, 0.0));

    let e_c = 0.5 * ((s.ub * cb + s.tb) - (s.ua * ca + s.ta)) / w;
    let e_s = 0.5 * (s.ua + s.ub);
    let e_ss = (s.ua * s.ua + s.ua * s.ub + s.ub * s.ub) / 3.0;
    let e_cs = (ca * ca * ca - cb * cb * cb) / (3.0 * w);
    let e_cc = 1.0 - e_ss;
    let c = s.c;
    let b = s.b;
    var m: ThinMoments;
    m.n = c * e_c + b * e_s;
    m.diag = c * c * e_cc + 2.0 * c * b * e_cs + b * b * e_ss;
    m.off = vec3<f32>(c.x * c.y, c.y * c.z, c.x * c.z) * e_cc
        + vec3<f32>(c.x * b.y + b.x * c.y, c.y * b.z + b.y * c.z, c.x * b.z + b.x * c.z) * e_cs
        + vec3<f32>(b.x * b.y, b.y * b.z, b.x * b.z) * e_ss;
    return m;
}

fn thin_sky_irradiance(m: ThinMoments) -> vec3<f32> {
    var e = vec3<f32>(0.0);
    e = e + camera.sky_sh[0].rgb * 0.282095 * 1.0;
    e = e + camera.sky_sh[1].rgb * (0.488603 * m.n.y) * 0.6666667;
    e = e + camera.sky_sh[2].rgb * (0.488603 * m.n.z) * 0.6666667;
    e = e + camera.sky_sh[3].rgb * (0.488603 * m.n.x) * 0.6666667;
    e = e + camera.sky_sh[4].rgb * (1.092548 * m.off.x) * 0.25;
    e = e + camera.sky_sh[5].rgb * (1.092548 * m.off.y) * 0.25;
    e = e + camera.sky_sh[6].rgb * (0.315392 * (3.0 * m.diag.z - 1.0)) * 0.25;
    e = e + camera.sky_sh[7].rgb * (1.092548 * m.off.z) * 0.25;
    e = e + camera.sky_sh[8].rgb * (0.546274 * (m.diag.x - m.diag.y)) * 0.25;
    return max(e, vec3<f32>(0.0));
}

fn thin_room_irradiance(m: ThinMoments) -> vec3<f32> {
    var e = model_u.room[0].rgb * 0.282095;
    e = e + model_u.room[1].rgb * (0.488603 * m.n.y) * 0.6666667;
    e = e + model_u.room[2].rgb * (0.488603 * m.n.z) * 0.6666667;
    e = e + model_u.room[3].rgb * (0.488603 * m.n.x) * 0.6666667;
    e = e + model_u.room[4].rgb * (1.092548 * m.off.x) * 0.25;
    e = e + model_u.room[5].rgb * (1.092548 * m.off.y) * 0.25;
    e = e + model_u.room[6].rgb * (0.315392 * (3.0 * m.diag.z - 1.0)) * 0.25;
    e = e + model_u.room[7].rgb * (1.092548 * m.off.z) * 0.25;
    e = e + model_u.room[8].rgb * (0.546274 * (m.diag.x - m.diag.y)) * 0.25;
    return max(e, vec3<f32>(0.0));
}

fn thin_shade_with_sky(world_pos: vec3<f32>, s: ThinSpan, m: ThinMoments, sky_vis: f32) -> vec3<f32> {
    let view_dir = normalize(cam_pos() - world_pos);

    var lit = hf3(min(thin_sky_irradiance(m) * clamp(sky_vis, 0.0, 1.0) * capsule_ambient(world_pos, normalize(m.n)), vec3<f32>(HF_MAX)));
    for (var i: u32 = 0u; i < live_light_count(); i = i + 1u) {
        let l = lights.lights[i];

        let l_dir = select(normalize(l.position.xyz - world_pos), normalize(-l.direction.xyz), l.params.z > 1.5);
        let arriving = light_contribution_rough(l, world_pos, l_dir, view_dir, 1.0, 0.0);

        if (max(max(arriving.r, arriving.g), arriving.b) <= 0.0) {
            continue;
        }
        let means = thin_mean_light(s, l_dir, normalize(l_dir + view_dir), SHININESS);
        let spread = arriving * (means.x + SPEC_STRENGTH * means.y);

        if (max(max(spread.r, spread.g), spread.b) <= 0.0) {
            continue;
        }
        var c = hf3(min(spread, vec3<f32>(HF_MAX)));
        if (l.params.z > 1.5) {
            c = c * hf(sun_visibility(l, world_pos));
        }
        c = c * hf(stationary_visibility(l));
        let layer = i32(l.params.w);
        if (SPOT_SHADOWS && layer >= 0 && f32(layer) < camera.shadow_params.y) {
            c = c * hf(pcf_layer(spot_shadow_tex, layer, world_pos, spot_view_proj(layer)));
        } else if (l.params.z < 1.5) {
            c = c * hf(character_shadow(layer, world_pos));
        }
        lit = min(lit + c, hf3(hf(HF_MAX)));
    }
    return vec3<f32>(lit);
}

fn thin_own_bulb_fill(world_pos: vec3<f32>, s: ThinSpan) -> vec3<f32> {
    let l_dir = normalize(model_u.bulb[0].xyz - world_pos);
    return own_bulb_fill(world_pos, l_dir) * thin_mean_light(s, l_dir, l_dir, SHININESS).x;
}
";

fn mesh_shader_variant(thin: bool) -> String {
    mesh_shader_variant_with(thin, thin)
}

const MATERIAL_SHADE: &str = r#"var colour: vec3<f32>;
    if (mask.a >= 0.996) {
        colour = tex_color.rgb * (shade_with_sky(in.world_pos, n, model_u.params.x * baked.a) + own_bulb_fill(in.world_pos, n) + baked.rgb + room_irradiance(n));
    } else {
        let env = baked.rgb + room_irradiance(n);
        colour = shade_material_env(
            in.world_pos, n, mask.a, 1.0, model_u.params.x * baked.a, env, vec4<f32>(0.5, 0.5, 0.5, 0.0),
            tex_color.rgb, in.world_pos, n,
        ) + tex_color.rgb * own_bulb_fill(in.world_pos, n);
    }"#;

fn mesh_shader_variant_with(thin: bool, thin_mean: bool) -> String {
    let (thin_in, thin_out, thin_vs, thin_fn, thin_fade) = if thin {
        (
            "@location(5) thin: vec4<f32>,",
            "@location(5) fade: f32,\n    @location(7) thin: vec4<f32>,",
            THIN_VS,
            THIN_SHARE,
            " * thin_share(in.fade, in.thin, in.world_pos)",
        )
    } else {
        ("", "", "", "", "")
    };

    let (thin_shade, thin_span, shaded) = if thin && thin_mean {
        (
            THIN_SHADE,
            "\n    let thin_span = thin_span_at(in.fade, in.thin, in.world_pos);\n    let thin_m = thin_moments(thin_span);",
            "let colour = tex_color.rgb * (thin_shade_with_sky(in.world_pos, thin_span, thin_m, model_u.params.x * baked.a) + thin_own_bulb_fill(in.world_pos, thin_span) + baked.rgb + thin_room_irradiance(thin_m));",
        )
    } else {
        ("", "", MATERIAL_SHADE)
    };
    format!(
        r#"
struct ModelUniform {{ model: mat4x4<f32>, params: vec4<f32>, room: array<vec4<f32>, 9>, bulb: array<vec4<f32>, 3>, weather: vec4<f32> }}
@group(1) @binding(0) var<uniform> model_u: ModelUniform;
{room_light}

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;
@group(2) @binding(2) var emissive_tex: texture_2d<f32>;

@group(3) @binding(0) var lm_tex: texture_2d<f32>;
@group(3) @binding(1) var lm_samp: sampler;

@group(3) @binding(4) var lm_sun_samp: sampler;
@group(3) @binding(5) var lm_stationary: texture_2d_array<f32>;
const STATIONARY_MASK_DISTANCE_TEXELS: f32 = {stationary_range:?};

{lights_block}

fn own_bulb_fill(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {{
    let marker = model_u.params.z;
    if (marker < 1.5) {{
        return vec3<f32>(0.0);
    }}
    let to_light = model_u.bulb[0].xyz - world_pos;
    let dist = length(to_light);
    let l_dir = to_light / max(dist, 0.0001);
    let d_over_r = dist / max(model_u.bulb[0].w, 0.0001);
    let d2_over_r2 = d_over_r * d_over_r;
    let window = clamp(1.0 - d2_over_r2 * d2_over_r2, 0.0, 1.0);
    let atten = (window * window) / max(dist * dist, LAMP_RADIUS * LAMP_RADIUS);
    let beam = spot_cone(dot(-l_dir, model_u.bulb[1].xyz), model_u.bulb[1].w, model_u.bulb[2].w, dist, 0.0);
    return model_u.bulb[2].rgb * max(dot(n, l_dir), 0.0) * atten * (1.0 - beam) * stationary_visibility_of(marker);
}}

fn unpack_emissive(packed: u32) -> vec3<f32> {{
    let r = f32((packed >> 0u) & 255u) / 255.0;
    let g = f32((packed >> 8u) & 255u) / 255.0;
    let b = f32((packed >> 16u) & 255u) / 255.0;
    let c = vec3<f32>(r, g, b);
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    let hue = select(hi, lo, c <= vec3<f32>(0.04045));

    let scale = (f32((packed >> 24u) & 255u) / 255.0) * {max_strength:?};
    return hue * scale;
}}

struct VIn {{
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
    @location(3) uv2:      vec2<f32>,
    @location(4) emissive: u32,
    {thin_in}
    @location(6) uv2_rect: vec4<f32>,
}}

struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv:     vec2<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) uv2:    vec2<f32>,

    @location(4) @interpolate(flat) emissive: vec3<f32>,
    {thin_out}
    @location(6) @interpolate(flat) uv2_rect: vec4<f32>,
    @location(8) room_near: f32,
}}

@vertex
fn vs_main(v: VIn) -> VOut {{
    var world_pos = model_u.model * vec4<f32>(v.position, 1.0);
    var out: VOut;
    {thin_vs}
    out.clip      = cam_view_proj() * world_pos;
    out.normal    = (model_u.model * vec4<f32>(v.normal, 0.0)).xyz;
    out.uv        = v.uv;
    out.world_pos = world_pos.xyz;
    out.uv2       = v.uv2;
    out.emissive  = unpack_emissive(v.emissive);
    out.uv2_rect  = v.uv2_rect;
    out.room_near = room_near(to_world_space(world_pos.xyz));
    return out;
}}

{thin_fn}{thin_shade}

const GRAB_GLOW_COLOUR: vec3<f32> = vec3<f32>(0.62, 0.86, 1.0);
fn grab_highlight(c: vec3<f32>, n: vec3<f32>, world_pos: vec3<f32>, k: f32) -> vec3<f32> {{
    let f = 1.0 - abs(dot(n, normalize(cam_pos() - world_pos)));
    let rim = f * f * f;
    return mix(c, GRAB_GLOW_COLOUR, k * (0.16 + 0.64 * rim));
}}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    if (in.room_near > 0.0) {{
        local_exposure = room_local_exposure(to_world_space(in.world_pos));
    }}
    let n = normalize(in.normal);{thin_span}

    let uv2 = clamp(in.uv2, in.uv2_rect.xy, in.uv2_rect.zw);

    let baked = textureSample(lm_tex, lm_samp, uv2);

    receiver_skips_baked = true;

    let st_layers = textureNumLayers(lm_stationary);
    let st_0 = textureSample(lm_stationary, lm_sun_samp, uv2, 0);
    var st_1 = vec4<f32>(1.0);
    var st_2 = vec4<f32>(1.0);
    var st_3 = vec4<f32>(1.0);
    if (st_layers > 1u) {{
        st_1 = textureSample(lm_stationary, lm_sun_samp, uv2, 1);
    }}
    if (st_layers > 2u) {{
        st_2 = textureSample(lm_stationary, lm_sun_samp, uv2, 2);
    }}
    if (st_layers > 3u) {{
        st_3 = textureSample(lm_stationary, lm_sun_samp, uv2, 3);
    }}
    set_stationary_masks_exact(st_0, st_1, st_2, st_3, STATIONARY_MASK_DISTANCE_TEXELS);
    let tex_color = textureSample(tex, samp, in.uv);

    if (tex_color.a < 0.35) {{
        discard;
    }}

    let mask = textureSample(emissive_tex, samp, in.uv);
    let glow = in.emissive * mask.rgb * model_u.params.y;

    {shaded}
    var out = tonemap(colour + glow);

    let grab_glow = model_u.weather.y;
    if (grab_glow > 0.0) {{
        out = grab_highlight(out, n, in.world_pos, grab_glow);
    }}
    return vec4<f32>(out, tex_color.a{thin_fade});
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
        max_strength = super::mesh::MeshVertex::MAX_EMISSIVE_STRENGTH,
        stationary_range = super::brush_pipeline::STATIONARY_MASK_DISTANCE_TEXELS,
        room_light = wgsl_room_irradiance(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::lights::{Light, LightKind, LightsUniform};
    use crate::renderer::mesh::{create_lightmap_texture, create_texture_from_rgba, MeshVertex};
    use crate::renderer::terrain_pipeline::tests::headless_gpu;
    use crate::renderer::uniforms::test_support::{scene_uniforms, TEST_EYE};
    use crate::renderer::uniforms::ShadowUpload;
    use crate::renderer::Color3;
    use wgpu::util::DeviceExt;

    #[test]
    fn every_spot_shadow_read_is_behind_the_switch() {
        const LIVE_SPOT: &str = "if (SPOT_SHADOWS && layer >= 0 && f32(layer) < camera.shadow_params.y) {";
        const SPOT_ROW: &str = "if (SPOT_SHADOWS && map > SUN_MOVING_MAP && map < CHARACTER_MAPS) {";
        let lights = crate::renderer::lights::wgsl_lights_block(0, 1);
        for (name, src) in [("mesh", mesh_shader()), ("thin", mesh_shader_variant_with(true, true)), ("lights", lights.clone())] {
            let lines: Vec<&str> = src.lines().collect();
            let mut reads = 0;
            for (k, line) in lines.iter().enumerate() {
                if line.contains("(spot_shadow_tex,") {
                    reads += 1;
                    assert!(
                        lines[k - 1].contains(LIVE_SPOT) || lines[k - 1].contains(SPOT_ROW),
                        "{name}: an ungated spot shadow read: {line}"
                    );
                }
            }
            assert!(reads > 0, "{name}: no spot shadow read found");
            let spotless = crate::renderer::lights::without_spot_shadows(src.clone());
            assert_eq!(spotless.lines().count(), lines.len(), "{name}: the twin changes one line");
        }

        let picks: Vec<&str> = lights.lines().map(str::trim).filter(|l| l.starts_with("map = ") || l.starts_with("var map = ")).collect();
        assert_eq!(picks, ["var map = -1;", "map = 1 + layer;", "map = CHARACTER_MAPS + k;", "map = SUN_MOVING_MAP;"]);
        assert!(lights.contains(&format!("{LIVE_SPOT}\n            map = 1 + layer;")), "a spot's row is handed out past the live-spot test");
    }

    #[test]
    fn every_mesh_register_cut_applies_and_validates() {
        use wgpu::naga;
        for base in [mesh_shader(), mesh_shader_variant_with(true, true)] {
            for (label, edits) in MESH_REGISTER_CUTS {
                let mut src = base.clone();
                for (from, to) in edits.iter() {
                    assert!(src.contains(from), "{label}: `{from}` is not in the mesh shader");
                    src = src.replacen(from, to, 1);
                }
                let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
                naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                    .validate(&module)
                    .unwrap_or_else(|e| panic!("{label}: {e:?}"));
            }
        }
    }

    fn render_mesh(emissive: [f32; 3], drive: f32, lit: bool) -> Option<[u8; 4]> {
        render_mesh_sky(emissive, drive, lit, 1.0)
    }

    fn render_mesh_masked(
        emissive: [f32; 3],
        drive: f32,
        mask: [u8; 4],
    ) -> Option<[u8; 4]> {
        render_mesh_full(emissive, drive, false, 0.0, mask)
    }

    fn render_mesh_sky(
        emissive: [f32; 3],
        drive: f32,
        lit: bool,
        sky_vis: f32,
    ) -> Option<[u8; 4]> {
        render_mesh_full(emissive, drive, lit, sky_vis, [255, 255, 255, 255])
    }

    fn render_mesh_full(
        emissive: [f32; 3],
        drive: f32,
        lit: bool,
        sky_vis: f32,
        mask: [u8; 4],
    ) -> Option<[u8; 4]> {

        render_mesh_baked(emissive, drive, lit, sky_vis, mask, [0, 0, 0, 255])
    }

    #[test]
    fn a_hidden_texel_stays_hidden_beside_lit_ones() {
        let grey = [128, 128, 128, 255];

        let hidden = [0u8, 255, 255, 255];
        let lit = [255u8, 255, 255, 255];
        let render = |texels: &[u8], w: u32, h: u32, uv2s: [[f32; 2]; 3]| {
            render_mesh_room_with(
                [0.0; 3], 0.0, true, 0.0, [255; 4], [0, 0, 0, 255], grey, Some((texels, w, h)), 40.0, &[[0.0; 3]; 9], None, uv2s, MeshVertex::WHOLE_ATLAS,
            )
        };
        let flat = [[0.5, 0.5]; 3];
        let Some(dark) = render(&hidden, 1, 1, flat) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let bright = render(&lit, 1, 1, flat).unwrap();

        let quad = [hidden, lit, lit, lit].concat();
        let mixed = render(&quad, 2, 2, [[-2.0, 2.0], [6.0, 2.0], [-2.0, -6.0]]).unwrap();
        assert!(bright[0] > dark[0] + 60, "the lamp lights the model: {bright:?} vs {dark:?}");
        assert!(mixed[0] <= dark[0] + 2, "a hidden texel read lit beside lit ones: {mixed:?} vs hidden {dark:?}, lit {bright:?}");
    }

    #[test]
    fn an_extrapolated_lightmap_read_stays_in_its_triangles_chart() {
        let grey = [128, 128, 128, 255];
        let hidden = [0u8, 255, 255, 255];
        let lit = [255u8, 255, 255, 255];
        let mask: Vec<u8> = (0..16).flat_map(|i| if i % 4 < 3 && i / 4 < 3 { hidden } else { lit }).collect();

        let uv2s = [[-1.375, 2.625], [6.625, 2.625], [-1.375, -5.375]];
        let render = |rect: [u16; 4]| {
            render_mesh_room_with(
                [0.0; 3], 0.0, true, 0.0, [255; 4], [0, 0, 0, 255], grey, Some((&mask, 4, 4)), 40.0, &[[0.0; 3]; 9], None, uv2s, rect,
            )
        };
        let Some(unbounded) = render(MeshVertex::WHOLE_ATLAS) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let chart = MeshVertex::uv2_rect_of([[0.25, 0.25], [0.5, 0.25], [0.25, 0.5]]);
        let bounded = render(chart).unwrap();
        let dark = render_mesh_room_with(
            [0.0; 3], 0.0, true, 0.0, [255; 4], [0, 0, 0, 255], grey, Some((&hidden, 1, 1)), 40.0, &[[0.0; 3]; 9], None, [[0.5, 0.5]; 3], MeshVertex::WHOLE_ATLAS,
        )
        .unwrap();
        assert!(unbounded[0] > dark[0] + 60, "the control reads the lit texel: {unbounded:?} vs hidden {dark:?}");
        assert!(bounded[0] <= dark[0] + 2, "the read left its chart: {bounded:?} vs hidden {dark:?}");
    }

    #[test]
    fn a_stationary_lamp_is_shadowed_on_a_mesh_by_its_mask() {

        let neutral_lm = [0, 0, 0, 255];
        let grey = [128, 128, 128, 255];
        let render = |stationary| render_mesh_stationary([0.0; 3], 0.0, true, 0.0, [255; 4], neutral_lm, grey, stationary, 40.0);
        let Some(plain) = render(None) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let lit = render(Some([255, 255, 255, 255])).unwrap();
        let hidden = render(Some([0, 0, 255, 255])).unwrap();
        assert!(plain[0] > 60, "the lamp does not light the mesh at all: {plain:?}");
        assert!((lit[0] as i32 - plain[0] as i32).abs() <= 1, "a fully lit mask changed the light: {lit:?} vs {plain:?}");
        assert!(hidden[0] <= 2, "a mask that hides the bulb let its light through: {hidden:?}");
    }

    #[test]
    fn a_spot_fixtures_bulb_lights_its_own_housing_outside_the_beam() {
        let grey = [128, 128, 128, 255];
        let lamp = |direction: glam::Vec3, mask_channel: Option<u8>| Light {
            mask_channel,
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: true,
            position: glam::Vec3::new(0.0, 0.0, 0.5),
            direction,
            kind: LightKind::Spot,
            color: Color3(255, 255, 255, 255),
            intensity: 0.4,
            range: 20.0,
            cone_angle_deg: 64.0,
            inner_cone_angle_deg: 30.0,
        };
        let (seen, hid) = ([255, 255, 255, 255], [0, 0, 255, 255]);
        let render = |own: Option<&Light>, mask: [u8; 4]| {
            render_mesh_room([0.0; 3], 0.0, false, 0.0, [255; 4], [0, 0, 0, 255], grey, Some(mask), 4.0, &[[0.0; 3]; 9], own)
        };
        let Some(none) = render(None, seen) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let away = lamp(glam::Vec3::Z, Some(0));
        let filled = render(Some(&away), seen).unwrap();
        let hidden = render(Some(&away), hid).unwrap();
        let aimed = render(Some(&lamp(glam::Vec3::NEG_Z, Some(0))), seen).unwrap();
        let unmasked = render(Some(&lamp(glam::Vec3::Z, None)), seen).unwrap();
        assert!(none[0] <= 2, "something else lights the test surface: {none:?}");
        assert!(filled[0] > 40, "the bulb did not light its housing outside the beam: {filled:?}");
        assert!(hidden[0] <= 2, "the mask let the bulb's light through the housing: {hidden:?}");
        assert!(aimed[0] <= 2, "the fill lit the beam a second time: {aimed:?}");
        assert!(unmasked[0] <= 2, "a lamp with no mask filled its housing: {unmasked:?}");
    }

    #[test]
    fn the_lit_room_lights_a_model_as_the_sky_would() {
        use crate::renderer::room_light::RoomLight;
        let grey = [128, 128, 128, 255];
        let render = |sky_vis: f32, room: &RoomLight| {
            render_mesh_room(
                [0.0; 3],
                0.0,
                false,
                sky_vis,
                [255; 4],
                [0, 0, 0, 255],
                grey,
                None,
                4.0,
                room,
                None,
            )
        };
        let Some(dark) = render(0.0, &[[0.0; 3]; 9]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!(dark[0] <= 1, "no lamp, no sky, no room: {dark:?}");

        let mut even = [[0.0f32; 3]; 9];
        even[0] = [crate::renderer::sky::AMBIENT / 0.282_095; 3];
        let by_room = render(0.0, &even).unwrap();
        let by_sky = render(1.0, &[[0.0; 3]; 9]).unwrap();
        assert!(by_room[0] > 8, "the room lit nothing: {by_room:?}");
        assert!(
            (by_room[0] as i32 - by_sky[0] as i32).abs() <= 1,
            "the room {by_room:?} vs the sky {by_sky:?}"
        );

        let mut toward = [[0.0f32; 3]; 9];
        toward[0] = [0.3 / 0.282_095; 3];
        toward[2] = [0.4; 3];
        let mut away = toward;
        away[2] = [-0.4; 3];
        let (facing, behind) = (render(0.0, &toward).unwrap(), render(0.0, &away).unwrap());
        assert!(
            facing[0] > 3 * behind[0].max(1),
            "the bright side {facing:?} vs the dark side {behind:?}"
        );
    }

    fn render_mesh_albedo(
        base: [u8; 4],
        sky_vis: f32,
        lightmap_texel: [u8; 4],
    ) -> Option<[u8; 4]> {
        render_mesh_inner([0.0; 3], 0.0, false, sky_vis, [255; 4], lightmap_texel, base)
    }

    fn render_mesh_baked(
        emissive: [f32; 3],
        drive: f32,
        lit: bool,
        sky_vis: f32,
        mask: [u8; 4],
        lightmap_texel: [u8; 4],
    ) -> Option<[u8; 4]> {
        render_mesh_inner(emissive, drive, lit, sky_vis, mask, lightmap_texel, [128, 128, 128, 255])
    }

    #[allow(clippy::too_many_arguments)]
    fn render_mesh_inner(
        emissive: [f32; 3],
        drive: f32,
        lit: bool,
        sky_vis: f32,
        mask: [u8; 4],
        lightmap_texel: [u8; 4],
        base: [u8; 4],
    ) -> Option<[u8; 4]> {
        render_mesh_stationary(emissive, drive, lit, sky_vis, mask, lightmap_texel, base, None, 4.0)
    }

    #[allow(clippy::too_many_arguments)]
    fn render_mesh_stationary(
        emissive: [f32; 3],
        drive: f32,
        lit: bool,
        sky_vis: f32,
        mask: [u8; 4],
        lightmap_texel: [u8; 4],
        base: [u8; 4],
        stationary: Option<[u8; 4]>,
        intensity: f32,
    ) -> Option<[u8; 4]> {
        render_mesh_room(
            emissive,
            drive,
            lit,
            sky_vis,
            mask,
            lightmap_texel,
            base,
            stationary,
            intensity,
            &[[0.0; 3]; 9],
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_mesh_room(
        emissive: [f32; 3],
        drive: f32,
        lit: bool,
        sky_vis: f32,
        mask: [u8; 4],
        lightmap_texel: [u8; 4],
        base: [u8; 4],
        stationary: Option<[u8; 4]>,
        intensity: f32,
        room: &crate::renderer::room_light::RoomLight,
        own: Option<&Light>,
    ) -> Option<[u8; 4]> {
        render_mesh_room_with(
            emissive,
            drive,
            lit,
            sky_vis,
            mask,
            lightmap_texel,
            base,
            stationary.as_ref().map(|t| (&t[..], 1, 1)),
            intensity,
            room,
            own,
            [[0.5, 0.5]; 3],
            MeshVertex::WHOLE_ATLAS,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_mesh_room_with(
        emissive: [f32; 3],
        drive: f32,
        lit: bool,
        sky_vis: f32,
        mask: [u8; 4],
        lightmap_texel: [u8; 4],
        base: [u8; 4],
        stationary: Option<(&[u8], u32, u32)>,
        intensity: f32,
        room: &crate::renderer::room_light::RoomLight,
        own: Option<&Light>,
        uv2s: [[f32; 2]; 3],
        uv2_rect: [u16; 4],
    ) -> Option<[u8; 4]> {
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        if lit {
            lights.upload(
                &queue,
                &[Light {
                    mask_channel: stationary.map(|_| 0),
                    shadow_near: None,
                    source_radius: 0.0,
                    in_level_bake: true,
                    position: glam::Vec3::new(0.0, 0.0, 4.0),
                    direction: glam::Vec3::NEG_Z,
                    kind: LightKind::Point,
                    color: Color3(255, 255, 255, 255),
                    intensity,
                    range: 20.0,
                    cone_angle_deg: 90.0,
                    inner_cone_angle_deg: 0.0,
                }],
            );
        } else {
            lights.upload(&queue, &[]);
        }
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);
        uniforms.upload(&queue, glam::Mat4::IDENTITY, TEST_EYE, &ShadowUpload::disabled());

        let pipeline = MeshPipeline::new(&device, format, &uniforms.layout);
        let model = pipeline.create_model_uniform(&device);
        model.upload_lit_bulb(&queue, glam::Mat4::IDENTITY, sky_vis, drive, room, own, 0.0);
        model.set_highlight(&queue, TEST_HIGHLIGHT.with(|h| h.get()));

        let tex = crate::renderer::mesh::create_mesh_material_texture(
            &device,
            &queue,
            &pipeline.texture_layout,
            &(base.to_vec(), 1, 1),
            &(mask.to_vec(), 1, 1),
        );
        let lm = match stationary {
            None => create_lightmap_texture(
                &device, &queue, &pipeline.lightmap_layout, &lightmap_texel, 1, 1, None,
            ),
            Some((texels, w, h)) => crate::renderer::mesh::create_lightmap_texture_full(
                &device,
                &queue,
                &pipeline.lightmap_layout,
                crate::renderer::mesh::LightmapLight::Srgb8(&lightmap_texel),
                1,
                1,
                None,
                None,
                Some((&[texels], w, h)),
            ),
        };

        let v = |p: [f32; 3], uv2: [f32; 2]| MeshVertex {
            position: p,
            normal: [0.0, 0.0, 1.0],
            uv: [0.5, 0.5],
            uv2,
            emissive: MeshVertex::pack_emissive(emissive),
            uv2_rect,
        };
        let verts = [v([-1.0, -1.0, 0.0], uv2s[0]), v([3.0, -1.0, 0.0], uv2s[1]), v([-1.0, 3.0, 0.0], uv2s[2])];
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_test_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_test_ib"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        const SIZE: u32 = 8;
        let target = device.create_texture(&TextureDescriptor {
            label: Some("mesh_test_target"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&TextureDescriptor {
            label: Some("mesh_test_depth"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let tv = target.create_view(&Default::default());
        let dv = depth.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("mesh_test_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("mesh_test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &tv,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &dv,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &model.bind_group, &[]);
            pass.set_bind_group(2, &tex.bind_group, &[]);
            pass.set_bind_group(3, &lm.bind_group, &[]);
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

    fn render_thin_wire(radius_px: f32, shift_px: f32, thin_px: Option<f32>) -> Option<Vec<Vec<u8>>> {
        render_thin_wire_at(radius_px, shift_px, thin_px, 0.0)
    }

    fn render_thin_wire_at(radius_px: f32, shift_px: f32, thin_px: Option<f32>, tilt: f32) -> Option<Vec<Vec<u8>>> {
        render_wire(radius_px, shift_px, thin_px, tilt, WireLight::Glow, 1.0, false)
    }

    #[derive(Clone, Copy, PartialEq)]
    enum WireLight {
        Glow,
        Side,
        SideOnWall,
    }

    const WALL: f32 = 51.0;

    const SIDE_LAMP_DISTANCE: f32 = 60.0;
    const SIDE_LAMP_LIGHT: f32 = 6.0;

    #[allow(clippy::too_many_arguments)]
    fn render_wire(
        radius_px: f32,
        shift_px: f32,
        thin_px: Option<f32>,
        tilt: f32,
        light: WireLight,
        zoom: f32,
        sampled: bool,
    ) -> Option<Vec<Vec<u8>>> {
        let lit = light != WireLight::Glow;
        const SIZE: u32 = 64;
        let px = 2.0 / SIZE as f32 / zoom;

        let (sides, r) = (8usize, radius_px * px);
        let (sin, cos) = tilt.sin_cos();
        let (axis, across) = (glam::Vec3::new(sin, cos, 0.0), glam::Vec3::new(cos, -sin, 0.0));
        let middle = glam::Vec3::new(-0.3 * cos / zoom, 0.0, 0.5) + across * (shift_px * px);
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;
        let lights = LightsUniform::new(&device);
        if lit {
            lights.upload(
                &queue,
                &[Light {
                    mask_channel: None,
                    shadow_near: None,
                    source_radius: 0.0,
                    in_level_bake: true,
                    position: middle + across * SIDE_LAMP_DISTANCE,
                    direction: glam::Vec3::NEG_Z,
                    kind: LightKind::Point,
                    color: Color3(255, 255, 255, 255),
                    intensity: SIDE_LAMP_LIGHT * SIDE_LAMP_DISTANCE * SIDE_LAMP_DISTANCE,
                    range: 1000.0,
                    cone_angle_deg: 90.0,
                    inner_cone_angle_deg: 0.0,
                }],
            );
        } else {
            lights.upload(&queue, &[]);
        }
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);
        let eye = if lit { glam::Vec3::new(0.0, 0.0, -1.0e4) } else { TEST_EYE };
        uniforms.upload(&queue, glam::Mat4::from_scale(glam::Vec3::new(zoom, zoom, 1.0)), eye, &ShadowUpload::disabled());
        let pipeline = if sampled {
            MeshPipeline::new_multisampled_thin_sampled(&device, format, &uniforms.layout, 4)
        } else {
            MeshPipeline::new_multisampled(&device, format, &uniforms.layout, 4)
        };
        let model = pipeline.create_model_uniform(&device);

        let glow = if lit { 0.0 } else { 1.0 };
        model.upload_lit_bulb(&queue, glam::Mat4::IDENTITY, 0.0, glow, &[[0.0; 3]; 9], None, thin_px.map_or(0.0, |p| p * px));
        let tex = crate::renderer::mesh::create_mesh_material_texture(
            &device, &queue, &pipeline.texture_layout, &(vec![128, 128, 128, 255], 1, 1), &(vec![255; 4], 1, 1),
        );
        let lm = create_lightmap_texture(&device, &queue, &pipeline.lightmap_layout, &[0, 0, 0, 255], 1, 1, None);

        let mut verts = Vec::new();
        for (y, _) in [(-0.9 / zoom, 0), (0.9 / zoom, 1)] {
            for k in 0..sides {
                let a = k as f32 / sides as f32 * std::f32::consts::TAU;
                let n = across * a.cos() + glam::Vec3::Z * a.sin();
                let n = [n.x, n.y, n.z];
                let at = middle + axis * y + glam::Vec3::from(n) * r;
                verts.push(MeshVertex {
                    position: at.to_array(),
                    normal: n,
                    uv: [0.5, 0.5],
                    uv2: [0.5, 0.5],
                    emissive: MeshVertex::pack_emissive([0.5, 0.5, 0.5]),
                    uv2_rect: MeshVertex::WHOLE_ATLAS,
                });
            }
        }
        let mut indices = Vec::new();
        for k in 0..sides as u32 {
            let (a, b) = (k, (k + 1) % sides as u32);
            let (c, d) = (a + sides as u32, b + sides as u32);

            indices.extend([a, b, c, b, d, c]);
        }
        let split = crate::renderer::mesh::split_thin_parts(&verts, &indices);
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("thin_test_vb"), contents: bytemuck::cast_slice(&verts), usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("thin_test_ib"), contents: bytemuck::cast_slice(&indices), usage: BufferUsages::INDEX,
        });
        let parts = split.as_ref().map(|s| crate::renderer::mesh::ThinParts::upload(&device, s));

        let tex_desc = |label, samples, format, usage| TextureDescriptor {
            label: Some(label),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: samples,
            dimension: TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        };
        let msaa = device.create_texture(&tex_desc("thin_test_msaa", 4, format, TextureUsages::RENDER_ATTACHMENT));
        let resolve = device.create_texture(&tex_desc("thin_test_resolve", 1, format, TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC));
        let depth = device.create_texture(&tex_desc("thin_test_depth", 4, TextureFormat::Depth32Float, TextureUsages::RENDER_ATTACHMENT));
        let (mv, rv, dv) = (msaa.create_view(&Default::default()), resolve.create_view(&Default::default()), depth.create_view(&Default::default()));
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("thin_test_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("thin_test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &mv,
                    depth_slice: None,
                    resolve_target: Some(&rv),
                    ops: Operations {
                        load: LoadOp::Clear(if light == WireLight::SideOnWall {
                            let w = WALL as f64 / 255.0;
                            Color { r: w, g: w, b: w, a: 1.0 }
                        } else {
                            Color::BLACK
                        }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &dv,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &model.bind_group, &[]);
            pass.set_bind_group(2, &tex.bind_group, &[]);
            pass.set_bind_group(3, &lm.bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            match (&parts, thin_px) {
                (Some(parts), Some(_)) => {
                    if parts.solid_count > 0 {
                        pass.set_pipeline(&pipeline.pipeline);
                        pass.set_index_buffer(parts.solid_index_buffer.slice(..), IndexFormat::Uint32);
                        pass.draw_indexed(0..parts.solid_count, 0, 0..1);
                    }
                    pass.set_pipeline(&pipeline.thin_pipeline);
                    pass.set_vertex_buffer(1, parts.vertex_buffer.slice(..));
                    pass.set_index_buffer(parts.thin_index_buffer.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..parts.thin_count, 0, 0..1);
                }
                _ => {
                    pass.set_pipeline(&pipeline.pipeline);
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                }
            }
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo { texture: &resolve, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
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
        Some((0..SIZE as usize).map(|y| (0..SIZE as usize).map(|x| data[y * 256 + x * 4]).collect()).collect())
    }

    fn wire_light_across_shifts(radius_px: f32, thin_px: Option<f32>) -> Option<(f32, f32)> {
        wire_light_across_shifts_at(radius_px, thin_px, 0.0)
    }

    fn wire_light_across_shifts_at(radius_px: f32, thin_px: Option<f32>, tilt: f32) -> Option<(f32, f32)> {
        wire_light_across_shifts_lit(radius_px, thin_px, tilt, WireLight::Glow, 1.0, false)
    }

    fn wire_light_across_shifts_lit(
        radius_px: f32,
        thin_px: Option<f32>,
        tilt: f32,
        light: WireLight,
        zoom: f32,
        sampled: bool,
    ) -> Option<(f32, f32)> {
        let rows: Vec<Vec<f32>> = (0..8)
            .map(|k| render_wire(radius_px, k as f32 / 8.0, thin_px, tilt, light, zoom, sampled))
            .map(|image| Some(image?.iter().map(|row| row.iter().map(|&v| v as f32).sum()).collect()))
            .collect::<Option<_>>()?;
        let (mut mean_all, mut cv_all, mut n) = (0.0, 0.0, 0.0);
        for y in 16..48 {
            let v: Vec<f32> = rows.iter().map(|r| r[y]).collect();
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
            mean_all += mean;
            cv_all += sd / mean.max(1e-6);
            n += 1.0;
        }
        Some((mean_all / n, cv_all / n))
    }

    #[test]
    fn a_thin_wire_holds_its_light_as_it_slides() {
        let Some((plain_mean, plain_cv)) = wire_light_across_shifts(0.16, None) else { return };
        let (thin_mean, thin_cv) = wire_light_across_shifts(0.16, Some(1.0)).unwrap();

        let full = *render_thin_wire(3.0, 0.0, None).unwrap()[32].iter().max().unwrap() as f32;
        let truth = 0.32 * full;
        eprintln!("THIN wire 0.32 px (truth {truth:.1}): plain mean {plain_mean:.1} varies {:.1}%, thin pass mean {thin_mean:.1} varies {:.1}%", plain_cv * 100.0, thin_cv * 100.0);
        assert!(plain_cv > 0.3, "the plain wire shimmers ({:.1}%), or this measures nothing", plain_cv * 100.0);
        assert!(thin_cv < 0.1, "widened, it holds its light: varies {:.1}%", thin_cv * 100.0);
        assert!((thin_mean / truth - 1.0).abs() < 0.1, "and it is the wire's own light: {thin_mean:.1} against {truth:.1}");
    }

    fn wire_steps(radius_px: f32, thin_px: Option<f32>) -> Option<f32> {
        wire_steps_at(radius_px, thin_px, 0.0)
    }

    fn wire_steps_at(radius_px: f32, thin_px: Option<f32>, tilt: f32) -> Option<f32> {
        wire_steps_lit(radius_px, thin_px, tilt, WireLight::Glow, 1.0)
    }

    fn wire_crawl(radius_px: f32, thin_px: Option<f32>, tilt: f32, light: WireLight, zoom: f32, sampled: bool) -> Option<f32> {
        let images: Vec<Vec<Vec<u8>>> = (0..8)
            .map(|k| render_wire(radius_px, k as f32 / 8.0, thin_px, tilt, light, zoom, sampled))
            .collect::<Option<_>>()?;
        let mut sum = 0.0f32;
        for y in 16..48 {
            for x in 0..64 {
                let v = |k: usize| images[k][y][x] as f32;
                for k in 1..7 {
                    let d = v(k + 1) - 2.0 * v(k) + v(k - 1);
                    sum += d * d;
                }
            }
        }
        Some((sum / (32.0 * 64.0 * 6.0)).sqrt())
    }

    fn wire_steps_lit(radius_px: f32, thin_px: Option<f32>, tilt: f32, light: WireLight, zoom: f32) -> Option<f32> {
        let images: Vec<Vec<Vec<u8>>> =
            (0..8).map(|k| render_wire(radius_px, k as f32 / 8.0, thin_px, tilt, light, zoom, false)).collect::<Option<_>>()?;
        let behind = if light == WireLight::SideOnWall { WALL } else { 0.0 };
        let (mut sum, mut n, mut peak) = (0.0f32, 0.0f32, 0.0f32);
        for y in 16..48 {
            for x in 0..64 {
                let v = |k: usize| images[k][y][x] as f32 - behind;
                for k in 1..7 {
                    if v(k - 1).abs() + v(k).abs() + v(k + 1).abs() > 0.0 {
                        let d = v(k + 1) - 2.0 * v(k) + v(k - 1);
                        sum += d * d;
                        n += 1.0;
                    }
                    peak = peak.max(v(k).abs());
                }
            }
        }
        Some((sum / n.max(1.0)).sqrt() / peak.max(1.0))
    }

    #[test]
    fn a_widened_wires_pixels_move_with_it_instead_of_stepping() {
        let Some(steps) = wire_steps(0.16, Some(1.0)) else { return };
        let plain = wire_steps(0.16, None).unwrap();
        let (mean, cv) = wire_light_across_shifts(0.16, Some(1.0)).unwrap();
        let full = *render_thin_wire(3.0, 0.0, None).unwrap()[32].iter().max().unwrap() as f32;
        let truth = 0.32 * full;
        eprintln!(
            "THIN wire 0.32 px: steps {steps:.4} (drawn as it is {plain:.4}); light {mean:.1} against {truth:.1}, varies {:.1}%",
            cv * 100.0
        );

        assert!(plain > 0.5, "the plain wire steps ({plain:.3}), or this measures nothing");
        assert!(steps < 0.09, "its pixels step {steps:.3} of its brightest as it slides");
        assert!(cv < 0.05, "and its light holds: varies {:.1}%", cv * 100.0);
        assert!((mean / truth - 1.0).abs() < 0.08, "and is the wire's own: {mean:.1} against {truth:.1}");
    }

    fn wire_beads(radius_px: f32, thin_px: Option<f32>, tilt: f32) -> Option<f32> {
        let mut total = 0.0;
        for k in 0..8 {
            let image = render_thin_wire_at(radius_px, k as f32 / 8.0, thin_px, tilt)?;
            let peaks: Vec<f32> = (16..48).map(|y| *image[y].iter().max().unwrap() as f32).collect();
            let mean = peaks.iter().sum::<f32>() / peaks.len() as f32;
            let sd = (peaks.iter().map(|p| (p - mean).powi(2)).sum::<f32>() / peaks.len() as f32).sqrt();
            total += sd / mean.max(1e-6);
        }
        Some(total / 8.0)
    }

    #[test]
    fn a_slanted_wire_neither_beads_nor_swings() {
        let Some(image) = render_thin_wire(3.0, 0.0, None) else { return };
        let full = *image[32].iter().max().unwrap() as f32;

        for degrees in [0.0f32, 22.5, 26.57, 45.0] {
            let tilt = degrees.to_radians();
            let (mean, cv) = wire_light_across_shifts_at(0.16, Some(1.0), tilt).unwrap();
            let beads = wire_beads(0.16, Some(1.0), tilt).unwrap();
            let steps = wire_steps_at(0.16, Some(1.0), tilt).unwrap();

            let truth = 0.32 * full / tilt.cos();
            eprintln!(
                "THIN wire 0.32 px at {degrees} degrees: light {mean:.1} against {truth:.1}, varies {:.1}%; beads {:.1}%; steps {steps:.4}",
                cv * 100.0,
                beads * 100.0
            );
            assert!(cv < 0.04, "at {degrees} degrees a row's light holds as it slides: varies {:.1}%", cv * 100.0);
            assert!(beads < 0.08, "at {degrees} degrees it does not bead: {:.1}%", beads * 100.0);
            assert!(steps < 0.09, "at {degrees} degrees its pixels move with it: steps {steps:.3}");
            assert!((mean / truth - 1.0).abs() < 0.08, "at {degrees} degrees it is the wire's own light: {mean:.1} against {truth:.1}");
        }
    }

    #[test]
    fn a_wide_part_keeps_its_middle_and_its_light() {
        for shift in [0.0, 0.375] {
            let Some(plain) = render_thin_wire(3.0, shift, None) else { return };
            let thin = render_thin_wire(3.0, shift, Some(1.0)).unwrap();
            let light = |image: &Vec<Vec<u8>>| (16..48).map(|y| image[y].iter().map(|&v| v as f32).sum::<f32>()).sum::<f32>();
            let (a, b) = (light(&plain), light(&thin));
            assert!((b / a - 1.0).abs() < 0.03, "a 6 px part, shifted {shift} px: light {b} against {a}");
            let middle = |image: &Vec<Vec<u8>>| *image[32].iter().max().unwrap() as i32;
            assert!((middle(&plain) - middle(&thin)).abs() <= 2, "its middle, shifted {shift} px");
        }
    }

    #[test]
    fn the_thin_pass_takes_the_same_terms_and_shadows_as_every_mesh() {
        let code = |text: &str| -> Vec<String> {
            text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with("//")).map(String::from).collect()
        };
        let moments = |line: &str| {
            line.replace("(0.488603 * y)", "(0.488603 * m.n.y)")
                .replace("(0.488603 * z)", "(0.488603 * m.n.z)")
                .replace("(0.488603 * x)", "(0.488603 * m.n.x)")
                .replace("(1.092548 * x * y)", "(1.092548 * m.off.x)")
                .replace("(1.092548 * y * z)", "(1.092548 * m.off.y)")
                .replace("(3.0 * z * z - 1.0)", "(3.0 * m.diag.z - 1.0)")
                .replace("(1.092548 * x * z)", "(1.092548 * m.off.z)")
                .replace("(x * x - y * y)", "(m.diag.x - m.diag.y)")
        };
        let body = |text: &str, from: &str, to: &str| -> Vec<String> {
            let start = text.find(from).unwrap_or_else(|| panic!("no `{from}`"));
            let end = start + text[start..].find(to).unwrap_or_else(|| panic!("no `{to}` after `{from}`")) + to.len();
            code(&text[start..end])
        };
        let thin = code(THIN_SHADE);
        let lights = crate::renderer::lights::wgsl_lights_block(0, 1);
        let sky = body(&lights, "fn sky_irradiance(", "return max(e, vec3<f32>(0.0));");
        let room = body(wgsl_room_irradiance(), "fn room_irradiance(", "return max(e, vec3<f32>(0.0));");
        for line in sky.iter().chain(&room).filter(|l| l.contains("_sh[") || l.contains("room[")) {
            let want = moments(line);
            assert!(thin.contains(&want), "THIN_SHADE lacks `{want}`, its mean of `{line}`");
        }
        let shade_with_sky = &lights[lights.find("fn shade_with_sky(").expect("the lights block shades meshes")..];
        let shadows = body(shade_with_sky, "if (l.params.z > 1.5) {\n            c = c * hf(sun_visibility", "c = c * hf(character_shadow(layer, world_pos));\n        }");
        let found = thin.windows(shadows.len()).any(|w| w == shadows.as_slice());
        assert!(found, "THIN_SHADE's shadows are not `shade_with_sky`'s:\n{}", shadows.join("\n"));
    }

    fn side_lit_wire_mean() -> f32 {
        let (eye, lamp) = (glam::Vec3::NEG_Z, glam::Vec3::X);
        let half = (lamp + eye).normalize();
        let steps = 20_000;
        let sum: f32 = (0..steps)
            .map(|k| {
                let u = -1.0 + (k as f32 + 0.5) / steps as f32 * 2.0;
                let n = eye * (1.0 - u * u).max(0.0).sqrt() + lamp * u;
                let facing = n.dot(lamp).max(0.0);
                let highlight = if facing > 0.0 { n.dot(half).max(0.0).powf(32.0) * 0.35 } else { 0.0 };
                facing + highlight
            })
            .sum();
        SIDE_LAMP_LIGHT * sum / steps as f32
    }

    #[test]
    fn a_side_lit_thin_wire_keeps_its_light_and_its_shading_as_it_slides() {
        let albedo = ((128.0f32 / 255.0 + 0.055) / 1.055).powf(2.4);
        let colour = crate::renderer::tonemap::tonemap(
            glam::Vec3::splat(albedo * side_lit_wire_mean()),
            1.0,
            crate::renderer::tonemap::ToneMapping::Aces,
        )
        .x;

        for (radius, zoom) in [(0.16f32, 1.0f32), (0.8, 10.0)] {
            for degrees in [0.0f32, 22.5, 26.57, 45.0] {
                let tilt = degrees.to_radians();
                let Some((mean, cv)) = wire_light_across_shifts_lit(radius, Some(1.0), tilt, WireLight::Side, zoom, false) else {
                    return;
                };
                let (sampled, sampled_cv) = wire_light_across_shifts_lit(radius, Some(1.0), tilt, WireLight::Side, zoom, true).unwrap();
                let crawl = wire_crawl(radius, Some(1.0), tilt, WireLight::SideOnWall, zoom, false).unwrap();
                let sampled_crawl = wire_crawl(radius, Some(1.0), tilt, WireLight::SideOnWall, zoom, true).unwrap();
                let truth = 2.0 * radius / tilt.cos() * 255.0 * colour;
                eprintln!(
                    "SIDE-LIT wire {:.2} px at {degrees} degrees: light {mean:.1} against {truth:.1}, varies {:.1}%, crawl {crawl:.3} levels \
                     (per fragment: {sampled:.1}, {:.1}%, {sampled_crawl:.3})",
                    2.0 * radius,
                    cv * 100.0,
                    sampled_cv * 100.0
                );
                assert!(sampled < 0.7 * truth, "per fragment it came out dark ({sampled:.1} against {truth:.1}), or this measures nothing");
                assert!(cv < 0.05, "a row's light holds as it slides: varies {:.1}%", cv * 100.0);
                assert!((mean / truth - 1.0).abs() < 0.06, "it is the wire's own light: {mean:.1} against {truth:.1}");
                assert!(crawl < sampled_crawl * 1.05, "and steps no more than it did: {crawl:.3} against {sampled_crawl:.3}");
            }
        }
    }

    const NONE: [f32; 3] = [0.0, 0.0, 0.0];

    thread_local! {

        static TEST_HIGHLIGHT: std::cell::Cell<f32> = const { std::cell::Cell::new(0.0) };
    }

    #[test]
    fn the_grab_glow_reads_in_the_dark_and_in_daylight() {
        let bright = |glow: f32| {
            TEST_HIGHLIGHT.with(|h| h.set(glow));
            let px = render_mesh_stationary(NONE, 0.0, true, 1.0, [255; 4], [0, 0, 0, 255], [200, 200, 200, 255], None, 60.0);
            TEST_HIGHLIGHT.with(|h| h.set(0.0));
            px
        };
        let dark = |glow: f32| {
            TEST_HIGHLIGHT.with(|h| h.set(glow));
            let px = render_mesh_sky(NONE, 0.0, false, 0.0);
            TEST_HIGHLIGHT.with(|h| h.set(0.0));
            px
        };
        let Some(dark_plain) = dark(0.0) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let (dark_lit, day_plain, day_lit) = (dark(1.0).unwrap(), bright(0.0).unwrap(), bright(1.0).unwrap());
        eprintln!("dark {dark_plain:?} -> {dark_lit:?}; daylight {day_plain:?} -> {day_lit:?}");

        assert!(dark_lit[2] >= dark_plain[2] + 24 && dark_lit[2] > dark_lit[0], "dark: {dark_plain:?} -> {dark_lit:?}");

        assert!(day_plain[0] > 200, "the daylight case is not bright: {day_plain:?}");
        assert!(day_lit[0] + 6 <= day_plain[0] && day_lit[2] >= day_lit[0], "daylight: {day_plain:?} -> {day_lit:?}");

        assert_eq!(dark(0.0).unwrap(), dark_plain);
    }

    #[test]
    fn the_bodys_dry_and_wet_twins_validate_and_are_lean() {
        let total = |s: &str| -> usize { crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|r| r.own * r.copies).sum() };
        let src = skinned_mesh_shader();
        let (dry, wet) = (skinned_body_twin(&src, false).expect("dry"), skinned_body_twin(&src, true).expect("wet"));
        for (label, s) in [("dry", &dry), ("wet", &wet)] {
            let module = wgpu::naga::front::wgsl::parse_str(s).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(s)));
            wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
        let (d, w, full) = (total(&dry), total(&wet), total(&src));
        eprintln!("skinned fs_main by the proxy: dry {d}, wet {w}, traced {full}");
        assert!(d < 1200 && w * 5 < full, "dry {d}, wet {w}, traced {full}");
    }

    #[test]
    fn the_scene_passs_opaque_twins_build_beside_the_blended_pipelines() {
        assert!(mesh_shader().contains("return vec4<f32>(out, tex_color.a);"), "the mesh shader's alpha moved");
        assert!(crate::renderer::pipeline::solid_shader_src().contains("return vec4<f32>(in.col.rgb * lit * lightmap, in.col.a);"));
        let Some((device, _queue)) = crate::renderer::pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = crate::renderer::lights::LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let format = TextureFormat::Rgba8UnormSrgb;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mesh = MeshPipeline::new_multisampled(&device, format, &uniforms.layout, 4);
        let spotless = MeshPipeline::new_multisampled_spotless(&device, format, &uniforms.layout, 4);
        let mirror = MeshPipeline::new_mirror(&device, format, &uniforms.layout);
        let body = SkinnedMeshPipeline::new_multisampled(&device, format, &uniforms.layout, 4).with_body_twins(&device);
        let solid = crate::renderer::pipeline::SolidPipeline::new_multisampled(&device, format, &uniforms.layout, 4);
        let mirror_solid = crate::renderer::pipeline::SolidPipeline::new_mirror(&device, format, &uniforms.layout);
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "an opaque twin failed to build: {err:?}");
        assert!(mesh.opaque.is_some() && spotless.opaque.is_some() && solid.opaque.is_some());
        assert!(mirror.opaque.is_none() && mirror_solid.opaque.is_none());
        assert!(body.for_opaque_body(false).is_some() && body.for_opaque_body(true).is_some());

        assert!(body.prepass.is_some(), "the body's depth prepass was not built");
        let src = skinned_mesh_shader();
        assert!(src.contains("@builtin(position) @invariant clip"));
        for twin in [skinned_body_twin(&src, false).unwrap(), skinned_body_twin(&src, true).unwrap()] {
            let vs = |s: &str| s[s.find("fn vs_main").unwrap()..s.find("@fragment").unwrap()].to_string();
            assert_eq!(vs(&twin), vs(&src), "a twin's vertex stage differs from the prepass's");
        }
    }

    #[test]
    #[ignore]
    fn skinned_shader_sizes() {
        let total = |s: &str| -> usize { crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|r| r.own * r.copies).sum() };
        let skinned = skinned_mesh_shader();
        let wet_from = "    if (wet > 0.0) {";
        let i = skinned.find(wet_from).expect("the wet branch");
        let j = i + skinned[i..].find("\n    }}\n").map(|k| k + 6).unwrap_or(0);
        let _ = j;
        let dry = skinned.replacen("    if (wet > 0.0) {", "    if (false) {", 1);
        eprintln!("skinned fs_main: dry path {} (wet branch compiled out), with the wet branch {}; static mesh {}", total(&dry), total(&skinned), total(&mesh_shader()));
        let lean = skinned.replacen("const PROBE_NOWHERE: bool = false;", "const PROBE_NOWHERE: bool = true;", 1);
        eprintln!("skinned fs_main, the wet film with no rooms' reflection machinery (PROBE_NOWHERE): {}", total(&lean));
        eprintln!("{}", crate::renderer::shader_inlining::report("skinned lean", &lean, "fs_main", 12));
    }

    #[test]
    fn the_grab_glow_is_behind_a_uniform_branch_and_small() {
        for (label, src) in [("mesh", mesh_shader()), ("thin", mesh_shader_variant_with(true, true))] {
            assert!(src.contains(GRAB_GLOW_BRANCH), "{label}: the glow is not behind its branch");
            assert!(src.contains("let grab_glow = model_u.weather.y;"), "{label}");
            let total = |s: &str| -> usize {
                crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|r| r.own * r.copies).sum()
            };
            let (with, without) = (total(&src), total(&src.replacen(GRAB_GLOW_BRANCH, "", 1)));
            eprintln!("{label} fs_main by the proxy: {without} without the grab glow, {with} with it (+{})", with - without);
            assert!(with - without <= 40, "{label}: the glow grew the shader by {}", with - without);
        }
    }
    const RED_BULB: [f32; 3] = [0.9, 0.1, 0.05];

    #[test]
    fn the_emissive_mask_decides_which_parts_glow() {

        let Some(masked_off) = render_mesh_masked([1.0, 1.0, 1.0], 2.0, [0, 0, 0, 255]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let masked_on = render_mesh_masked([1.0, 1.0, 1.0], 2.0, [255, 255, 255, 255]).unwrap();
        assert!(
            masked_on[0] > masked_off[0] + 32,
            "a black mask must silence a fully-emissive material: \
             on {masked_on:?} vs masked-off {masked_off:?}",
        );
    }

    #[test]
    fn a_material_with_no_emissive_texture_still_emits() {

        let Some(dark) = render_mesh_masked([0.0, 0.0, 0.0], 2.0, [255, 255, 255, 255]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let glowing = render_mesh_masked([0.8, 0.8, 0.8], 2.0, [255, 255, 255, 255]).unwrap();
        assert!(glowing[0] > dark[0] + 32, "factor-only emission must work: {glowing:?} vs {dark:?}");
    }

    #[test]
    fn a_non_emissive_material_is_unchanged_by_the_drive() {

        let Some(off) = render_mesh(NONE, 0.0, true) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let driven = render_mesh(NONE, 5.0, true).unwrap();
        assert_eq!(
            off, driven,
            "a material with no emissive must ignore the drive entirely: {off:?} vs {driven:?}",
        );
    }

    #[test]
    fn driving_an_emissive_material_makes_it_brighter() {
        let Some(dark) = render_mesh(RED_BULB, 0.0, true) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let on = render_mesh(RED_BULB, 1.0, true).unwrap();
        assert!(on[0] > dark[0], "the bulb did not light up: {on:?} vs {dark:?}");
    }

    #[test]
    fn a_switched_off_bulb_is_exactly_a_plain_material() {

        let Some(off_bulb) = render_mesh(RED_BULB, 0.0, true) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let plain = render_mesh(NONE, 0.0, true).unwrap();
        assert_eq!(off_bulb, plain, "a bulb at zero drive must render as plain material");
    }

    #[test]
    fn a_lit_bulb_still_glows_in_a_completely_dark_room() {

        let Some(unlit_room) = render_mesh_sky(RED_BULB, 1.0, false, 0.0) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let dark_plain = render_mesh_sky(NONE, 1.0, false, 0.0).unwrap();
        assert!(
            unlit_room[0] > dark_plain[0] + 16,
            "a driven bulb must be clearly visible with no lights at all: \
             {unlit_room:?} vs {dark_plain:?}",
        );
    }

    #[test]
    fn the_emissive_colour_is_the_authored_one() {

        let Some(dark) = render_mesh(NONE, 0.0, false) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let red = render_mesh(RED_BULB, 2.0, false).unwrap();
        let dr = red[0] as i32 - dark[0] as i32;
        let dg = red[1] as i32 - dark[1] as i32;
        let db = red[2] as i32 - dark[2] as i32;
        assert!(dr > dg && dr > db, "a red bulb must glow RED: deltas r{dr} g{dg} b{db}");
    }

    #[test]
    fn baked_bounce_lights_a_mesh_that_no_lamp_and_no_sky_reaches() {

        let level = |v: u8| render_mesh_baked([0.0; 3], 0.0, false, 0.0, [255; 4], [v, v, v, 255]);
        let (Some(dark), Some(some), Some(most)) = (level(0), level(128), level(255)) else {
            return;
        };
        assert_eq!(
            dark[0], 0,
            "no lamp, no sky and no bounce should be exactly black, got {dark:?}",
        );
        assert!(
            some[0] > dark[0] && most[0] > some[0],
            "bounce {} -> {} -> {} is not monotonic; a multiplied lightmap cannot \
             lift an unlit surface at all, and this is the test that says so",
            dark[0],
            some[0],
            most[0],
        );
        assert!(
            most[0] > 12,
            "a full-strength bounce on an unlit surface rendered only {most:?}",
        );
    }

    #[test]
    fn baked_bounce_is_tinted_by_the_material_it_lands_on() {

        let bounce = [255u8, 255, 255, 255];
        let (Some(pale), Some(dark)) = (
            render_mesh_albedo([230, 230, 230, 255], 0.0, bounce),
            render_mesh_albedo([20, 20, 20, 255], 0.0, bounce),
        ) else {
            return;
        };
        assert!(
            pale[0] as i32 - dark[0] as i32 > 24,
            "the same bounce rendered {dark:?} on a near-black material and {pale:?} \
             on a near-white one; it is being added outside the albedo multiply, so \
             indirect light ignores what it lands on",
        );
    }

    #[test]
    fn the_lightmap_alpha_is_baked_sky_visibility() {
        let Some(open) = render_mesh_baked([0.0; 3], 0.0, false, 1.0, [255; 4], [0, 0, 0, 255])
        else {
            return;
        };
        let Some(sealed) = render_mesh_baked([0.0; 3], 0.0, false, 1.0, [255; 4], [0, 0, 0, 0])
        else {
            return;
        };
        assert!(
            open[0] > sealed[0],
            "a texel that can see the sky ({open:?}) should be brighter than a \
             sealed one ({sealed:?})",
        );
        let Some(no_sky) = render_mesh_baked([0.0; 3], 0.0, false, 0.0, [255; 4], [0, 0, 0, 255])
        else {
            return;
        };
        assert_eq!(
            sealed, no_sky,
            "baked alpha 0 and object sky_vis 0 must mean the same thing; they \
             multiply into the same term",
        );
    }

}

#[cfg(test)]
pub fn mesh_shader_src() -> String {
    mesh_shader()
}
#[cfg(test)]
pub fn skinned_mesh_shader_src() -> String {
    skinned_mesh_shader()
}
