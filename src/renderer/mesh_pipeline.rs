use super::lights::wgsl_lights_block;
use super::mesh::{MeshVertex, SkinnedMeshVertex, MAX_SKIN_JOINTS};
use super::pipeline::lightmap_bind_group_layout;
use wgpu::*;

pub struct MeshPipeline {
    pub pipeline: RenderPipeline,
    /// The same shading for a primitive's thin parts, widened and faded: see
    /// `thin_parts`. Drawn after everything opaque and the sky, from the
    /// primitive's `ThinParts`, with its `ThinVertex` buffer in slot 1.
    pub thin_pipeline: RenderPipeline,
    pub texture_layout: BindGroupLayout,
    pub model_layout: BindGroupLayout,
    pub lightmap_layout: BindGroupLayout,
}

impl MeshPipeline {
    pub fn new(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout) -> Self {
        Self::new_with_front_face(device, format, camera_layout, FrontFace::Ccw, 1, crate::renderer::multiview::ViewMode::Mono)
    }

    /// See `pipeline::SolidPipeline::new_multisampled`.
    /// The same, drawing BOTH EYES in one pass. See `multiview::ViewMode`.
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
        Self::new_with_front_face(device, format, camera_layout, FrontFace::Ccw, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    pub fn new_mirror(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout) -> Self {
        Self::new_with_front_face(device, format, camera_layout, FrontFace::Cw, 1, crate::renderer::multiview::ViewMode::Mono)
    }

    /// MEASUREMENT: as [`Self::new_multisampled`], its thin pass shaded per
    /// fragment as before 2026-10-01. See `Levers::thin_shading_sampled`.
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
        Self::new_edited(device, format, camera_layout, front_face, samples, view, thin_mean, &|src| src, ("mesh_pipeline", "mesh_thin_pipeline"))
    }

    /// [`Self::new_multisampled`]'s SPOTLESS TWIN: the spots' shadow maps
    /// never read (`lights::without_spot_shadows`), for frames where no spot
    /// casts. `Levers::spotless_shaders`.
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
        )
    }

    /// MEASUREMENT ONLY: every [`MESH_REGISTER_CUTS`] variant of the scene
    /// pass's mesh shader and its thin pass, built under the cut's name so
    /// PIPESTATS logs their registers and occupancy. Nothing draws with them.
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
    ) -> Self {
        // Through `shader_precision`, as the scene readers: at half precision
        // where the device has it and `HALF_PRECISION` is on (off as shipped).
        // The lamp loop's colours (`shade_with_sky`, `thin_shade_with_sky`)
        // are the only `hf` maths a mesh shader reaches (B1, 2026-10-06).
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
                // The emissive MASK. glTF emission is factor x texture, and a
                // real fixture uses the texture to say which part glows -- one
                // material can cover a whole lamp with only the bulb emitting.
                // Materials without one bind a white 1x1, so the factor alone
                // still works.
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
                // VERTEX AND FRAGMENT: the vertex stage reads the matrix,
                // the fragment stage reads params.x for sky visibility. A
                // vertex-only layout fails pipeline creation with "visibility
                // flags don't include the shader stage", which is a runtime
                // validation error rather than a build one.
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

        let build = |label: &str, shader: &ShaderModule, buffers: &[Option<VertexBufferLayout>]| device.create_render_pipeline(&RenderPipelineDescriptor {
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
        let pipeline = build(label, &shader, &[Some(MeshVertex::layout())]);
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
            texture_layout,
            model_layout,
            lightmap_layout,
        }
    }

    pub fn create_model_uniform(&self, device: &Device) -> ModelUniform {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("mesh_model_uniform"),
            // See `MODEL_UNIFORM_SIZE`.
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

        ModelUniform { buffer, bind_group }
    }
}

/// A model's uniform, in bytes: its matrix (64), its params (16; x sky
/// visibility, y emissive drive, z its own bulb's mask marker, w the thin
/// parts' least drawn width as a share of depth, 0 as they are), the room's
/// light on it, nine vec4s of harmonics (144; see `room_light`), and a
/// fixture's own bulb, three vec4s (48; see `ModelUniform::upload_lit_bulb`).
/// EVERY buffer that is ever bound as a `ModelUniform` is this size -- the
/// mirror's and the caves' too, whose shaders read only the matrix -- so no
/// upload can overrun one left behind.
pub const MODEL_UNIFORM_SIZE: u64 = 272;

pub struct ModelUniform {
    pub buffer: Buffer,
    pub bind_group: BindGroup,
}

impl ModelUniform {
    /// Upload the model matrix with FULL sky visibility.
    ///
    /// The neutral, and what every caller that has no occlusion data wants: 1.0
    /// reproduces the shading meshes had before the term existed.
    pub fn upload(&self, queue: &Queue, model: glam::Mat4) {
        self.upload_with_sky(queue, model, 1.0);
    }

    /// Upload the model matrix and how much sky reaches this object.
    ///
    /// WHY A PER-OBJECT SCALAR AND NOT A MAP
    ///
    /// A brush has a lightmap and terrain has a footprint texture, but a moving
    /// character has neither -- it is somewhere different every frame, so there
    /// is nothing to bake against it. Without this it took the full open-sky
    /// ambient wherever it stood, so an avatar indoors lit exactly as brightly
    /// as one on a lawn, which reads as the character being unaffected by the
    /// building it is standing in.
    ///
    /// One value for the whole object is coarse and deliberately so: it is the
    /// ambient term, which varies slowly, and the alternative is a probe volume
    /// that costs memory and a bake for a difference nobody sees on a body.
    pub fn upload_with_sky(&self, queue: &Queue, model: glam::Mat4, sky_vis: f32) {
        self.upload_full(queue, model, sky_vis, 0.0);
    }

    /// Upload the model matrix, sky visibility, and how hard this object glows.
    ///
    /// WHY THE GLOW IS PER OBJECT AND THE COLOUR IS PER VERTEX
    ///
    /// They answer different questions and change at different times. The mesh
    /// says WHICH parts of a lamp emit and in what colour -- authored once, in
    /// the asset. This says HOW BRIGHTLY, right now -- zero when the light is
    /// off, the light's intensity when it is on, and anything in between for a
    /// dim or a flicker. Putting both in the same place would mean rewriting a
    /// vertex buffer to switch a lamp off.
    ///
    /// 0.0 is the default and means "does not glow", so every object that is
    /// not a fixture is unaffected.
    pub fn upload_full(
        &self,
        queue: &Queue,
        model: glam::Mat4,
        sky_vis: f32,
        emissive_drive: f32,
    ) {
        self.upload_lit(queue, model, sky_vis, emissive_drive, &[[0.0; 3]; 9]);
    }

    /// [`Self::upload_full`], with the light of the room the model stands in,
    /// turned into the frame its normals are in: see `room_light`. Zero is no
    /// room light, as before it existed.
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

    /// [`Self::upload_lit`], for a fixture: `own` is the fixture's own lamp
    /// as this frame lights the room with it, in the same frame. A SPOT with
    /// a stationary mask channel also lights the fixture's own surfaces
    /// OUTSIDE its beam -- the bulb shines every way, and the beam is what
    /// its reflector sends into the room; see `own_bulb_fill` in the shader.
    /// Anything else, or `None`, adds nothing.
    ///
    /// `thin_width`: the thin pass's kernel unit as a share of depth -- an eye
    /// pixel's size there times `levers.thin_parts` -- which sets how far past
    /// a thin part it draws (see `thin_parts`). 0 draws thin parts as they
    /// are, and only the thin pass reads it.
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

    /// [`Self::upload_lit_bulb`], the fixture's OWN light -- its bulb's glow
    /// and its lamp on its own surfaces -- scaled by `own_scale`, as an eye
    /// adapted to the lamp sees it (`tonemap::own_light_scale`). Only a
    /// fixture whose lamp lights its own housing takes it (see
    /// `own_bulb_fill`): scaling the glow of one whose shade the light loop
    /// lights at full strength -- a sconce -- would put a grey bulb in a white
    /// mouth, as before 2026-09-30.
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
        let mut data = [0f32; (MODEL_UNIFORM_SIZE / 4) as usize];
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
            // The marker the light list gives a stationary lamp: 2 + channel.
            data[18] = 2.0 + channel as f32;
            data[56..60].copy_from_slice(&[l.position.x, l.position.y, l.position.z, l.range]);
            data[60..64].copy_from_slice(&[l.direction.x, l.direction.y, l.direction.z, cos_outer]);
            let i = l.intensity * k;
            data[64..68].copy_from_slice(&[c[0] * i, c[1] * i, c[2] * i, cos_inner]);
        }
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&data));
    }
}

/// The spot shadows' tent, as `lights::pcf_layer` calls it: the lean form
/// since B1 (2026-10-06).
const TENT_CALL: &str = "    return pcf_tile_tent_lean_at(\n        tex,";
/// The models' per-pixel choice of photograph in their room: every slot
/// looked at (`lights::probe_nearest_two`), as from 2026-10-01 to exp64 ...
const CHOOSE_SCAN: &str = "    let near = probe_nearest_two(select_world, room, room);
    c.best = near.s0;
    c.second = near.s1;
    c.best_dist = select(1e30, near.d0, near.s0 >= 0);
    c.second_dist = select(1e30, near.d1, near.s1 >= 0);";
/// ... and the room's own chain walked, as before that and in exp64.
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
/// `lights::probe_volume_sample`'s one nearest photograph, scanned ...
const VOLUME_SCAN: &str = "    let pick = probe_nearest_two(select_world, room, room).s0;";
/// ... and walked.
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

/// MEASUREMENT ONLY: the scene pass's mesh shader with one part changed at a
/// time -- see [`MeshPipeline::log_register_cuts`]. 2026-10-02: the spots'
/// tent and the per-pixel scan together took `mesh_pipeline` from 19
/// registers to 22 and its occupancy from 62% to 50%; these say which. The
/// shader walks its room's chain (exp64); `scan` is every slot looked at.
const MESH_REGISTER_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("mesh_cut_none", &[]),
    ("mesh_box_spots", &[(TENT_CALL, "    return pcf_tile_at(\n        tex,")]),
    // The tent's loop over picked taps, as it shipped until B1.
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
}

impl SkinnedMeshPipeline {
    pub fn new(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout) -> Self {
        Self::new_multisampled(device, format, camera_layout, 1)
    }

    /// See `pipeline::SolidPipeline::new_multisampled`.
    pub fn new_multisampled(

        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, camera_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    /// The same, drawing BOTH EYES in one pass. See `multiview::ViewMode`.
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
                // The emissive MASK. glTF emission is factor x texture, and a
                // real fixture uses the texture to say which part glows -- one
                // material can cover a whole lamp with only the bulb emitting.
                // Materials without one bind a white 1x1, so the factor alone
                // still works.
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
                // VERTEX AND FRAGMENT: the vertex stage reads the matrix,
                // the fragment stage reads params.x for sky visibility. A
                // vertex-only layout fails pipeline creation with "visibility
                // flags don't include the shader stage", which is a runtime
                // validation error rather than a build one.
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

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("skinned_mesh_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(SkinnedMeshVertex::layout())],
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
                cull_mode: None,
                front_face: FrontFace::Ccw,
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
            texture_layout,
            model_layout,
            skin_joint_layout,
        }
    }

    pub fn create_model_uniform(&self, device: &Device) -> ModelUniform {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("skinned_mesh_model_uniform"),
            // See `MODEL_UNIFORM_SIZE`.
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
        ModelUniform { buffer, bind_group }
    }
}

fn skinned_mesh_shader() -> String {
    format!(
        r#"
// Group 0 -- the camera, the lights and both shadow maps -- is declared by
// `wgsl_lights_block` below, so there is one description of that layout rather
// than one per shader.

struct ModelUniform {{ model: mat4x4<f32>, params: vec4<f32>, room: array<vec4<f32>, 9> }}
@group(1) @binding(0) var<uniform> model_u: ModelUniform;
{room_light}

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;
@group(2) @binding(2) var emissive_tex: texture_2d<f32>;

struct JointMatrices {{ mats: array<mat4x4<f32>, {max_skin_joints}> }}
@group(3) @binding(0) var<uniform> joints: JointMatrices;

{lights_block}

// Unpack the vertex's authored emissive colour.
//
// sRGB-decoded, matching how it was packed -- see MeshVertex::pack_emissive.
// Skipping the decode makes every glowing part far too bright, in the same way
// and for the same reason a lightmap written linearly into an sRGB texture came
// out too dark.
fn unpack_emissive(packed: u32) -> vec3<f32> {{
    let r = f32((packed >> 0u) & 255u) / 255.0;
    let g = f32((packed >> 8u) & 255u) / 255.0;
    let b = f32((packed >> 16u) & 255u) / 255.0;
    let c = vec3<f32>(r, g, b);
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    let hue = select(hi, lo, c <= vec3<f32>(0.04045));
    // The alpha byte is the emissive SCALE, not an opacity: a bulb is many
    // times brighter than white, and clamping it to white is what made the
    // lamp's glow look switched off. See MeshVertex::pack_emissive.
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
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv:     vec2<f32>,
    @location(2) world_pos: vec3<f32>,
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
    return out;
}}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let n = normalize(in.normal);
    // A character's surface lies inside its own capsules: it takes no capsule
    // darkening, which would black it out. See `capsule_ambient`.
    capsule_receiver = false;
    // The lamps and the sky, and the lit room round it: see `room_light`.
    let lit = shade_with_sky(in.world_pos, n, model_u.params.x) + room_irradiance(n);
    let tex_color = textureSample(tex, samp, in.uv);
    return vec4<f32>(tonemap(tex_color.rgb * lit), tex_color.a);
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
        max_strength = super::mesh::MeshVertex::MAX_EMISSIVE_STRENGTH,
        max_skin_joints = MAX_SKIN_JOINTS,
        room_light = wgsl_room_irradiance(),
    )
}

/// THE ROOM'S LIGHT ON A MODEL, as WGSL: the harmonics `model_u.room` holds
/// (see `room_light`), evaluated as `sky_irradiance` evaluates the sky's --
/// the same basis, the cosine lobe's weights already over pi -- along a
/// normal in the frame the model is drawn in. Written out term by term, not a
/// loop over an array, for the reason `sky_irradiance` gives.
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

/// THE THIN PASS's widening (see `thin_parts`): a part is pushed out along
/// its welded normal past its true radius by the reach of the kernel that
/// spreads its light (see `THIN_SHARE`) -- 1.71 kernel units of `params.w`
/// times its depth -- so every fragment the kernel gives light to is drawn,
/// and `fade` is the share of the drawn radius the part really fills. The
/// depth is the clip w, which a pixel's size is proportional to.
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

/// ACROSS A WIDENED PART, each fragment's share of its light (see
/// `thin_parts`): the true part's width under a kernel round the fragment's
/// middle -- a box two fragments wide blurred by a box root-two wide, a
/// trapezoid reaching 1.71 fragments either side. Fragments sample the
/// distance across an upright wire a whole fragment apart and across a
/// diagonal one 0.71 apart; this kernel's shares add up to the same light at
/// both spacings wherever the wire falls (and within a percent at every other
/// slant), and its flat top keeps a diagonal wire from beading. A tent one
/// fragment either side added up only upright: along a diagonal its brightest
/// pixel swung by a quarter, beads that crawled along a lamp's cage as the
/// head moved -- 14% more shimmer on the headset than the round profile
/// before it, whose light swung by a quarter as an upright wire slid
/// (2026-10-01). How far across the part: the widened corners lie its drawn
/// radius out along their normals, so the normal as interpolated,
/// unnormalised, is the place across a facet, in radii, along the way it
/// turns (screen-parallel; the normal's change over the fragment). The
/// fragment's size: a step to the next one, less its part along the view --
/// a foveated block's own. A part many fragments wide is solid in its middle
/// and keeps the kernel's soft edge. Derivatives: called at the top level of
/// `fs_main`.
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
// The share of the kernel that lies below `x` (in fragments from its middle):
// half of it beyond 0, falling straight across the flat top (height a half)
// and as a square over the sloped sides out to 1 + root a half.
fn thin_kernel_below(x: f32) -> f32 {
    let a = abs(x);
    let s = max(1.7071068 - a, 0.0);
    let beyond = select(0.5 - 0.5 * a, 0.17677670 * s * s, a >= 0.29289322);
    return select(beyond, 1.0 - beyond, x > 0.0);
}
";

/// THE THIN PASS's LIGHT (see `thin_parts`). A part under a pixel wide has its
/// lit side, its shadowed side and its highlight inside one pixel, which shows
/// their MEAN. Shaded instead from the normal interpolated at each fragment of
/// the widened band, the band sampled that profile wherever the fragment
/// centres fell -- the lit half of a wire on one side of it, the dark half on
/// the other -- and which fragments took which half changed with every step
/// of the head: the lit side of a cage wire beside its bulb crawled, the
/// most left in the level (headset census, 2026-10-01: J 0.17 on hall_front's
/// near pendant), and a wire lit from its side came out half as bright as it
/// is, the band's middle -- most of its light -- facing the eye and not the
/// lamp (`a_side_lit_thin_wire_keeps_its_light_and_its_shading_as_it_slides`).
///
/// So each fragment takes the mean of the lighting over the stretch of the
/// part its kernel reaches (`THIN_SHARE`): all of a part under a pixel --
/// every fragment the same colour, which nothing can make crawl -- and a
/// fragment's own few pixels of one many pixels wide. Round the visible half,
/// with `c` the normal facing the eye, `b` across the part on screen and theta
/// the angle from `c`, the normal is c cos(theta) + b sin(theta) and the
/// position across the part, in radii, is u = sin(theta): a fragment's reach
/// is an interval of u, and a mean over the screen is a mean over u. In closed
/// form: the lamps' diffuse exactly, their highlights as Gaussians (erf), the
/// sky's and the room's harmonics from the normal's first and second moments.
/// The axis is square to the normal and to the way it turns across the
/// screen. The lightmap is still read where the fragment is.
const THIN_SHADE: &str = "
struct ThinSpan {
    c: vec3<f32>,
    b: vec3<f32>,
    t: vec3<f32>,
    // The stretch of the visible half a fragment's kernel reaches, as u and
    // as theta.
    ua: f32,
    ub: f32,
    ta: f32,
    tb: f32,
}
// Derivatives: called at the top level of `fs_main`.
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
    // Where this fragment is across the part, in its own radii: the band
    // holds the part scaled by `fade`. Its kernel reaches 1.71 fragments.
    let u = dot(thin.xyz, s.b) / max(fade, 1e-4);
    let reach = 1.7071068 * fragment / max(thin.w, 1e-9);
    var ua = clamp(u - reach, -1.0, 1.0);
    var ub = clamp(u + reach, -1.0, 1.0);
    // A fragment the part only grazes: a sliver at its edge.
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
// erf, to 1.5e-7 (Abramowitz and Stegun 7.1.26).
fn thin_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return sign(x) * (1.0 - p * exp(-a * a));
}
// A lamp along `l` over the stretch, as `light_contribution` shades it: x the
// mean of max(n.l, 0), y of pow(max(n.h, 0), shininess) where n.l > 0. Round
// the circle n.l is |l square to the part| cos(theta - phi) and a step across
// the screen is cos(theta) d theta, so the diffuse integral is closed-form;
// n.h is |h square| cos(theta - theta_h), and cos^s is close to a Gaussian of
// width 1/sqrt(s), whose integral over the lit stretch is an erf.
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
// The normal's mean over the stretch, and the means of the products of its
// components -- which a sum of harmonics to order two is linear in. Over u:
// cos(theta) = sqrt(1 - u^2), sin(theta) = u.
struct ThinMoments {
    n: vec3<f32>,
    // xx, yy, zz
    diag: vec3<f32>,
    // xy, yz, xz
    off: vec3<f32>,
}
fn thin_moments(s: ThinSpan) -> ThinMoments {
    let w = s.ub - s.ua;
    let ca = sqrt(max(1.0 - s.ua * s.ua, 0.0));
    let cb = sqrt(max(1.0 - s.ub * s.ub, 0.0));
    // From the antiderivatives (u sqrt(1 - u^2) + asin u) / 2 and
    // -(1 - u^2)^(3/2) / 3.
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
// `sky_irradiance`, term for term, of the moments.
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
// `room_irradiance`, term for term, of the moments.
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
// `shade_with_sky` over the stretch.
fn thin_shade_with_sky(world_pos: vec3<f32>, s: ThinSpan, m: ThinMoments, sky_vis: f32) -> vec3<f32> {
    let view_dir = normalize(cam_pos() - world_pos);
    // In `hf` through the shadows, as `shade_with_sky`.
    var lit = hf3(min(thin_sky_irradiance(m) * clamp(sky_vis, 0.0, 1.0) * capsule_ambient(world_pos, normalize(m.n)), vec3<f32>(HF_MAX)));
    for (var i: u32 = 0u; i < live_light_count(); i = i + 1u) {
        let l = lights.lights[i];
        // The light arriving here as `light_contribution` has it -- colour,
        // falloff, cone -- on a surface square to it; spread round the part.
        let l_dir = select(normalize(l.position.xyz - world_pos), normalize(-l.direction.xyz), l.params.z > 1.5);
        let arriving = light_contribution_rough(l, world_pos, l_dir, view_dir, 1.0, 0.0);
        // Out of its range or its cone: nothing to spread, as `shade_with_sky`
        // skips it -- the means cost two atan2s and two erfs a lamp.
        if (max(max(arriving.r, arriving.g), arriving.b) <= 0.0) {
            continue;
        }
        let means = thin_mean_light(s, l_dir, normalize(l_dir + view_dir), SHININESS);
        let spread = arriving * (means.x + SPEC_STRENGTH * means.y);
        // THE SHADOWS, AS `shade_with_sky` TAKES THEM.
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
            c = c * hf(character_shadow(i, world_pos));
        }
        lit = min(lit + c, hf3(hf(HF_MAX)));
    }
    return vec3<f32>(lit);
}
// `own_bulb_fill` over the stretch: the bulb shines every way, so only its
// diffuse.
fn thin_own_bulb_fill(world_pos: vec3<f32>, s: ThinSpan) -> vec3<f32> {
    let l_dir = normalize(model_u.bulb[0].xyz - world_pos);
    return own_bulb_fill(world_pos, l_dir) * thin_mean_light(s, l_dir, l_dir, SHININESS).x;
}
";

/// The mesh shader; `thin` for the thin pass, which widens a thin part by its
/// kernel's reach (a unit of `params.w` of its depth) and gives each fragment
/// its share of the part's light (see `thin_parts`), reading each vertex's
/// welded normal and radius from a second vertex buffer.
fn mesh_shader_variant(thin: bool) -> String {
    mesh_shader_variant_with(thin, thin)
}

/// `mesh_shader_variant`, the thin pass shaded with the mean over each
/// fragment's reach (`thin_mean`, as shipped) or, to measure, from the normal
/// interpolated at the fragment as before 2026-10-01.
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
    // The thin pass shades with the mean over each fragment's reach: see
    // `THIN_SHADE`. The other variant's text is unchanged.
    let (thin_shade, thin_span, lit, room) = if thin && thin_mean {
        (
            THIN_SHADE,
            "\n    let thin_span = thin_span_at(in.fade, in.thin, in.world_pos);\n    let thin_m = thin_moments(thin_span);",
            "thin_shade_with_sky(in.world_pos, thin_span, thin_m, model_u.params.x * baked.a) + thin_own_bulb_fill(in.world_pos, thin_span)",
            "thin_room_irradiance(thin_m)",
        )
    } else {
        (
            "",
            "",
            "shade_with_sky(in.world_pos, n, model_u.params.x * baked.a) + own_bulb_fill(in.world_pos, n)",
            "room_irradiance(n)",
        )
    };
    format!(
        r#"
// Group 0 -- the camera, the lights and both shadow maps -- is declared by
// `wgsl_lights_block` below, so there is one description of that layout rather
// than one per shader.

// `bulb`: a fixture's own lamp -- [position, range], [beam direction, cos of
// the outer half-angle], [radiance, cos of the inner] -- with its mask marker
// in `params.z`. See `own_bulb_fill`.
struct ModelUniform {{ model: mat4x4<f32>, params: vec4<f32>, room: array<vec4<f32>, 9>, bulb: array<vec4<f32>, 3> }}
@group(1) @binding(0) var<uniform> model_u: ModelUniform;
{room_light}

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;
@group(2) @binding(2) var emissive_tex: texture_2d<f32>;

@group(3) @binding(0) var lm_tex: texture_2d<f32>;
@group(3) @binding(1) var lm_samp: sampler;
// The stationary lamps' shadows on this mesh, on its own atlas, and the
// sampler the brushes read theirs with. See `lights::set_stationary_masks`.
@group(3) @binding(4) var lm_sun_samp: sampler;
@group(3) @binding(5) var lm_stationary: texture_2d_array<f32>;
const STATIONARY_MASK_DISTANCE_TEXELS: f32 = {stationary_range:?};

{lights_block}

// THE FIXTURE'S BULB ON ITS OWN HOUSING: a spot fixture's lamp, outside its
// beam, on the fixture's own surfaces. The inside of a hanging lamp's bell --
// what its bulb lights most brightly, 5-10 cm off -- lies outside the cone, so
// the spot left it grey, and from across the room the lamp looked switched
// off beside a path-traced reference of it (2026-10-01). The bulb shines every
// way; the beam is what its reflector sends into the room. So a fixture takes
// its own lamp as a point -- the light loop's beam, plus this, its complement
// at the same intensity -- and the room is not lit twice. Shadowed by the
// lamp's baked mask on this mesh, which keeps a housing's outside dark; with no
// mask there is no fill (`params.z` 0). The bake's `own_bulb_fill`, term for
// term.
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

// Unpack the vertex's authored emissive colour.
//
// sRGB-decoded, matching how it was packed -- see MeshVertex::pack_emissive.
// Skipping the decode makes every glowing part far too bright, in the same way
// and for the same reason a lightmap written linearly into an sRGB texture came
// out too dark.
fn unpack_emissive(packed: u32) -> vec3<f32> {{
    let r = f32((packed >> 0u) & 255u) / 255.0;
    let g = f32((packed >> 8u) & 255u) / 255.0;
    let b = f32((packed >> 16u) & 255u) / 255.0;
    let c = vec3<f32>(r, g, b);
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    let hue = select(hi, lo, c <= vec3<f32>(0.04045));
    // The alpha byte is the emissive SCALE, not an opacity: a bulb is many
    // times brighter than white, and clamping it to white is what made the
    // lamp's glow look switched off. See MeshVertex::pack_emissive.
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

// At the pixel's centre, NOT the centroid, measured (headset, 2026-10-01):
// centroid on `normal`, `world_pos` and `uv2` -- so an MSAA edge pixel of a
// sliver does not extrapolate past its triangle into a neighbour's lightmap
// chart -- took 13% off the shimmer of a near lamp cage's edges and cost the
// scene pass 0.75 ms where that lamp filled the view. Its specks on the sconce
// plate stayed only because the masks' fwidth widening leaked as well; `uv2`
// is now held inside the triangle's chart by the flat `uv2_rect`, for nothing.
// See `MeshVertex::uv2_rect` and `BRUSH_CENTROID_VARYINGS`.
struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv:     vec2<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) uv2:    vec2<f32>,
    // FLAT, not interpolated: this is a per-primitive constant that happens to
    // travel on the vertex. Interpolating it would be arithmetic on three
    // identical values, and would blend across a seam between two materials
    // rather than switching cleanly at it.
    @location(4) @interpolate(flat) emissive: vec3<f32>,
    {thin_out}
    @location(6) @interpolate(flat) uv2_rect: vec4<f32>,
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
    return out;
}}

{thin_fn}{thin_shade}
@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let n = normalize(in.normal);{thin_span}
    // Inside the triangle's own chart: an MSAA edge pixel's `uv2` is
    // extrapolated past it. See `MeshVertex::uv2_rect`.
    let uv2 = clamp(in.uv2, in.uv2_rect.xy, in.uv2_rect.zw);
    // RGB is baked direct+bounce and is ADDED; ALPHA is baked sky visibility
    // and is MULTIPLIED into the sky term. The two neutrals are at opposite
    // ends of the range -- black adds nothing, 255 scales by one -- which is
    // exactly what an unbaked mesh's default texture carries.
    let baked = textureSample(lm_tex, lm_samp, uv2);
    // Per-texel sky visibility, narrowing the per-OBJECT value the model
    // uniform carries. One number for a whole lamp cannot say that the inside
    // of its shade sees less sky than the top of it.
    // The baked lamps are in this atlas. See `receiver_skips_baked`.
    receiver_skips_baked = true;
    // THE STATIONARY LAMPS' SHADOWS, as the brushes take theirs. Without them
    // every live lamp lit a mesh with no visibility at all: a sconce's plate
    // glowed with its own bulb through its shade (headset, 2026-09-29). A mesh
    // baked before these existed binds one neutral layer, fully lit. Read
    // exactly as baked, unwidened -- see `set_stationary_masks_exact`.
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
    let lit = {lit};
    let tex_color = textureSample(tex, samp, in.uv);
    // ADDED, NOT MULTIPLIED.
    //
    // This used to be a product, and a product cannot ADD light: a surface no
    // lamp reaches directly sits at the ambient floor and no amount of bounce
    // can lift it -- which is the inside of every lamp shade, and most of a
    // real interior. Bounced light is light and belongs in the sum, exactly as
    // it does for brushes.
    //
    // Inside the albedo multiply rather than outside it: indirect light
    // reflects off the surface's own colour the same way direct light does.
    //
    // Adding is only correct because a light is either baked or realtime and
    // never both -- see LightMode. `shade_with_sky` sees only the realtime
    // ones, the texture carries only the baked ones, and the sky term belongs
    // solely to `shade_with_sky`, which varies it by the direction the surface
    // faces.
    //
    // params.y is how hard this object is driven right now: 0 when the light is
    // off, its intensity when on. So the asset decides WHAT glows and the light
    // entity decides HOW MUCH. The glow is unaffected by the lighting: a bulb
    // is bright because it is a source, not because something shines on it, and
    // scaling it by `lit` would make a lamp go dark exactly when the room does.
    //
    // factor x MASK x drive. Dropping the mask makes an entire lamp housing
    // glow because its one material declares emissiveFactor [1,1,1] and relies
    // on the texture to pick out the bulb.
    let mask = textureSample(emissive_tex, samp, in.uv).rgb;
    let glow = in.emissive * mask * model_u.params.y;
    // The lit room round it, which the baked bounce -- an old per-object
    // estimate -- all but left out: see `room_light`.
    return vec4<f32>(tonemap(tex_color.rgb * (lit + baked.rgb + {room}) + glow), tex_color.a{thin_fade});
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

    /// Draw one mesh triangle and read the centre pixel back.
    ///
    /// `lit` puts a lamp in the scene; `emissive` is the material's authored
    /// glow colour and `drive` is how hard the light entity is driving it right
    /// now. Rendering rather than merely building the pipeline is the point:
    /// creating it proves only that the WGSL parses, which is exactly the check
    /// that passed twice this session while the shader was wrong.
    /// THE SPOTLESS TWINS DRAW THE SAME PIXELS where no spot casts: every read
    /// of a spot's shadow map, in the models' shaders (the thin pass's own
    /// copy included) and in the lights block every scene shader shares, sits
    /// directly under the `SPOT_SHADOWS` switch and the live-spot test -- so
    /// with no spot casting, switching it off removes code no pixel ran. The
    /// scene readers' lamp loop, which carries the point into every kind of
    /// map with one transform, reads a spot's map under the switch and a
    /// spot's row of `camera.moving_view_proj` -- which only the live-spot
    /// test hands out.
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
        // Every row the lamp loop picks: none, a spot's under the live-spot
        // test, a characters' tile past the spots', the sun's moving map.
        let picks: Vec<&str> = lights.lines().map(str::trim).filter(|l| l.starts_with("map = ") || l.starts_with("var map = ")).collect();
        assert_eq!(picks, ["var map = -1;", "map = 1 + layer;", "map = CHARACTER_MAPS + k;", "map = SUN_MOVING_MAP;"]);
        assert!(lights.contains(&format!("{LIVE_SPOT}\n            map = 1 + layer;")), "a spot's row is handed out past the live-spot test");
    }

    /// Every measurement cut of the mesh shader still finds its text, in the
    /// shader and in its thin pass -- a cut that no longer applies would
    /// report the uncut shader under its name -- and both results are valid
    /// WGSL. See `MeshPipeline::log_register_cuts`.
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

    /// The same, with an explicit emissive MASK texel.
    fn render_mesh_masked(
        emissive: [f32; 3],
        drive: f32,
        mask: [u8; 4],
    ) -> Option<[u8; 4]> {
        render_mesh_full(emissive, drive, false, 0.0, mask)
    }

    /// The same, with baked sky visibility given explicitly.
    ///
    /// Needed because "no lights" is NOT the same as "no light": `shade_with_sky`
    /// still returns sky ambient, so a room with the lamp list empty is dim, not
    /// dark. Only sky_vis = 0 makes the lighting term genuinely zero, which is
    /// the one condition under which multiplying the glow by it can be told
    /// apart from adding it.
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
        // The neutral lightmap: adds nothing, narrows no sky.
        render_mesh_baked(emissive, drive, lit, sky_vis, mask, [0, 0, 0, 255])
    }

    /// A MODEL'S MASK IS READ AS BAKED, not widened by how fast it changes
    /// across the screen as a brush's is: a model gives each triangle a chart
    /// of its own, so its mask jumps at every sliver and between the pixels of
    /// one quad, and widened by that, a texel baked fully hidden read up to
    /// 16% lit -- enough to show white 15 cm from a bulb (2026-10-01; the
    /// sconce plate's dashes, which led here, needed `uv2_rect` as well).
    /// Here the centre pixel reads a hidden texel and both its quad
    /// neighbours read lit ones.
    #[test]
    fn a_hidden_texel_stays_hidden_beside_lit_ones() {
        let grey = [128, 128, 128, 255];
        // Channel 0: the distance code, then the full-width penumbra the
        // model bake writes. Channel 1 neutral.
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
        // u = 2x and v = -2y in NDC: the 8x8 target's centre pixel (4, 4)
        // reads texel (0, 0)'s centre, its right and lower neighbours texels
        // (1, 0) and (0, 1).
        let quad = [hidden, lit, lit, lit].concat();
        let mixed = render(&quad, 2, 2, [[-2.0, 2.0], [6.0, 2.0], [-2.0, -6.0]]).unwrap();
        assert!(bright[0] > dark[0] + 60, "the lamp lights the model: {bright:?} vs {dark:?}");
        assert!(mixed[0] <= dark[0] + 2, "a hidden texel read lit beside lit ones: {mixed:?} vs hidden {dark:?}, lit {bright:?}");
    }

    /// A PIXEL'S `uv2` STAYS IN ITS TRIANGLE'S CHART, however far past the
    /// triangle it was extrapolated -- as an MSAA edge pixel's is. A 4x4 mask:
    /// the triangle's chart is texel (1, 1), hidden, with its gutter ring
    /// hidden too, as the bake dilates it; everything past is lit, as another
    /// triangle's chart or the atlas's unused neutral is. The centre pixel's
    /// `uv2` lands on lit texel (3, 3): clamped to the chart it reads hidden,
    /// and with no bound it reads lit (the control).
    #[test]
    fn an_extrapolated_lightmap_read_stays_in_its_triangles_chart() {
        let grey = [128, 128, 128, 255];
        let hidden = [0u8, 255, 255, 255];
        let lit = [255u8, 255, 255, 255];
        let mask: Vec<u8> = (0..16).flat_map(|i| if i % 4 < 3 && i / 4 < 3 { hidden } else { lit }).collect();
        // u = -1.375 + 2 (x + 1) and v = 2.625 - 2 (y + 1) in NDC: the 8x8
        // target's centre pixel reads (0.875, 0.875), texel (3, 3)'s centre.
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

    /// A stationary lamp takes its shadow on a mesh from the mesh's own baked
    /// mask, as on a brush: fully shadowed, none of its light arrives; fully
    /// lit, exactly what the lamp gives unmasked. On the headset a sconce's
    /// plate glowed with its own bulb because meshes had no mask at all
    /// (2026-09-29).
    #[test]
    fn a_stationary_lamp_is_shadowed_on_a_mesh_by_its_mask() {
        // No sky, no bake: the lamp is the only light.
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

    /// THE FIXTURE'S BULB ON ITS OWN HOUSING (`own_bulb_fill`). A stationary
    /// spot aimed AWAY from the surface lights it through its bulb, outside
    /// the beam; the lamp's mask shadows that light; aimed AT the surface it
    /// adds nothing, its beam having it already; and a lamp with no mask adds
    /// nothing. The light list is empty, so only the fill can light anything.
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

    /// THE LIT ROOM LIGHTS A MODEL that no lamp and no sky reaches: black
    /// without it -- a hand indoors out of a lamp's reach, a hanging lamp's
    /// shade under a bright ceiling (headset, 2026-09-30) -- and with it,
    /// EXACTLY what the sky's harmonics would give for the same light: a room
    /// glowing evenly at the flat sky's level, on a model that sees no sky,
    /// draws the same pixel as that sky on a model that sees all of it. And
    /// the side the room is brighter on is the side that lights a face
    /// turned toward it.
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
        // The test uniforms' sky is the flat AMBIENT; the same, as a room.
        let mut even = [[0.0f32; 3]; 9];
        even[0] = [crate::renderer::sky::AMBIENT / 0.282_095; 3];
        let by_room = render(0.0, &even).unwrap();
        let by_sky = render(1.0, &[[0.0; 3]; 9]).unwrap();
        assert!(by_room[0] > 8, "the room lit nothing: {by_room:?}");
        assert!(
            (by_room[0] as i32 - by_sky[0] as i32).abs() <= 1,
            "the room {by_room:?} vs the sky {by_sky:?}"
        );
        // Brighter toward +z, which the model faces; then toward -z.
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

    /// The same, with the material's own base colour given explicitly.
    fn render_mesh_albedo(
        base: [u8; 4],
        sky_vis: f32,
        lightmap_texel: [u8; 4],
    ) -> Option<[u8; 4]> {
        render_mesh_inner([0.0; 3], 0.0, false, sky_vis, [255; 4], lightmap_texel, base)
    }

    /// The same, with the baked lightmap texel given explicitly.
    ///
    /// RGB is added and ALPHA multiplies the sky term, so the neutral texel is
    /// `[0, 0, 0, 255]` -- the two channels are neutral at opposite ends of the
    /// range. Every test above predates the lightmap and must keep measuring
    /// what it was written to measure, which is why they all go through the
    /// neutral wrapper rather than naming a texel.
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

    /// The same, the lamp a STATIONARY one on mask channel 0 when `stationary`
    /// is given, and the mesh's own mask layer that one texel; `intensity` the
    /// lamp's.
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

    /// The same, the model standing in a room whose light is `room` (see
    /// `room_light`; the model's normal is +z).
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

    /// The same, with a stationary mask of any size and the triangle's three
    /// lightmap coordinates given -- the centre pixel's neighbours can read
    /// other texels of the mask.
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

        // Mid grey base, with a WHITE emissive mask so this measures the
        // factor and the drive. The mask itself is covered separately.
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

    /// A glowing wire `radius_px` eye pixels thick, upright across a 64 px
    /// target with 4x MSAA, `shift_px` right of a fixed place; with `thin_px`
    /// drawn by the thin pass with that kernel unit (see `thin_parts`).
    /// Returns the red channel, a row at a time.
    fn render_thin_wire(radius_px: f32, shift_px: f32, thin_px: Option<f32>) -> Option<Vec<Vec<u8>>> {
        render_thin_wire_at(radius_px, shift_px, thin_px, 0.0)
    }

    /// As `render_thin_wire`, the wire leaning `tilt` radians from upright
    /// toward +x, and slid `shift_px` across itself.
    fn render_thin_wire_at(radius_px: f32, shift_px: f32, thin_px: Option<f32>, tilt: f32) -> Option<Vec<Vec<u8>>> {
        render_wire(radius_px, shift_px, thin_px, tilt, WireLight::Glow, 1.0, false)
    }

    /// How `render_wire` lights its wire: by its own glow over black, or by a
    /// lamp at its side -- over black, or over a grey wall, which shows a
    /// wire pixel darker than what is behind it as well as one brighter.
    #[derive(Clone, Copy, PartialEq)]
    enum WireLight {
        Glow,
        Side,
        SideOnWall,
    }

    /// What `WireLight::SideOnWall` clears to, as the target stores it.
    const WALL: f32 = 51.0;

    /// How far off to the wire's side `render_wire`'s lamp stands, and the
    /// light it lands there with (radiance times the falloff).
    const SIDE_LAMP_DISTANCE: f32 = 60.0;
    const SIDE_LAMP_LIGHT: f32 = 6.0;

    /// The wire `render_thin_wire_at` draws: glowing, or with `lit` dark
    /// itself and lit by a lamp far off to its side, square to it and to the
    /// eye -- the half of its visible side that faces the lamp lit, the other
    /// half in the wire's own shadow, as a cage wire beside its bulb is. Then
    /// the eye is far down -z, the side the identity camera shows. `zoom`
    /// magnifies the camera, the scene shrunk to match: a wire past a third
    /// of a pixel thick is thicker than a thin part can be (`THIN_RADIUS_MAX`)
    /// at 1, and a thin part's pixels at 10. `sampled`: the thin pass shaded
    /// per fragment, as before 2026-10-01 (`Levers::thin_shading_sampled`).
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
        // The wire: 8 facets round, along `axis` at depth 0.5. Facing the
        // camera (which looks along +z) means a normal with -z in it.
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
        // No sky, no room, and either the glow alone, so every pixel of the
        // wire is the same light, or the lamp alone. With an identity camera
        // the clip w is 1, so `params.w` is the width itself, in clip units.
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
            // Counter-clockwise on screen where the facet faces the camera
            // (its normal has -z): the far facets are culled, as on any tube.
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

    /// How a wire's light in the middle rows varies as it slides across a
    /// pixel in eighths: (mean light per row, its variation across shifts,
    /// averaged over rows).
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

    /// THE THIN PASS (`thin_parts`): a wire a third of a pixel thick, drawn
    /// as a mesh with 4x MSAA, gains and loses whole samples as it slides --
    /// the shimmer of every lamp's cage on the headset (2026-10-01). Drawn by
    /// the thin pass, its light per row holds steady, and is the same light
    /// it had.
    #[test]
    fn a_thin_wire_holds_its_light_as_it_slides() {
        let Some((plain_mean, plain_cv)) = wire_light_across_shifts(0.16, None) else { return };
        let (thin_mean, thin_cv) = wire_light_across_shifts(0.16, Some(1.0)).unwrap();
        // The light a 0.32 px wire really carries: that share of a pixel the
        // glow covers whole, read from the middle of a wide one.
        let full = *render_thin_wire(3.0, 0.0, None).unwrap()[32].iter().max().unwrap() as f32;
        let truth = 0.32 * full;
        eprintln!("THIN wire 0.32 px (truth {truth:.1}): plain mean {plain_mean:.1} varies {:.1}%, thin pass mean {thin_mean:.1} varies {:.1}%", plain_cv * 100.0, thin_cv * 100.0);
        assert!(plain_cv > 0.3, "the plain wire shimmers ({:.1}%), or this measures nothing", plain_cv * 100.0);
        assert!(thin_cv < 0.1, "widened, it holds its light: varies {:.1}%", thin_cv * 100.0);
        assert!((thin_mean / truth - 1.0).abs() < 0.1, "and it is the wire's own light: {thin_mean:.1} against {truth:.1}");
    }

    /// How much a wire's pixels STEP as it slides across a pixel in eighths,
    /// rather than move: each pixel's second difference across the shifts,
    /// RMS over the pixels it touches in the middle rows, over its brightest
    /// -- the headset's J, measured on one wire.
    fn wire_steps(radius_px: f32, thin_px: Option<f32>) -> Option<f32> {
        wire_steps_at(radius_px, thin_px, 0.0)
    }

    fn wire_steps_at(radius_px: f32, thin_px: Option<f32>, tilt: f32) -> Option<f32> {
        wire_steps_lit(radius_px, thin_px, tilt, WireLight::Glow, 1.0)
    }

    /// The headset's J on one wire, in 8-bit levels: each pixel's second
    /// difference across eighth-pixel shifts, RMS over every pixel of the
    /// middle rows -- the same pixels whatever is drawn, so a wire that steps
    /// less scores less however faint it is against what is behind it.
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

    /// Over a wall, each pixel as it differs from the wall: a wire pixel
    /// darker than what is behind it steps as much as a brighter one.
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

    /// ACROSS A WIDENED WIRE, ITS OWN PROFILE. Widened and faded, a wire
    /// still had hard edges, and where they fell among the four samples
    /// stepped the pixels they crossed by a quarter of its fade at a time --
    /// a far cage's wires redrawn at every millimetre of head movement
    /// (headset `lamp_by_pillar-mQ`, 2026-10-01: J 0.029 on the cage).
    /// Weighted across by the tube's own thickness profile, the edges carry
    /// none of it: the pixels move with the wire, and its light is the same.
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
        // A band of one faded value stepped 0.180 of its brightest at every
        // eighth of a pixel; a tent's shares 0.055, this kernel's 0.060 --
        // both near the rounding of 8-bit pixels this faint.
        assert!(plain > 0.5, "the plain wire steps ({plain:.3}), or this measures nothing");
        assert!(steps < 0.09, "its pixels step {steps:.3} of its brightest as it slides");
        assert!(cv < 0.05, "and its light holds: varies {:.1}%", cv * 100.0);
        assert!((mean / truth - 1.0).abs() < 0.08, "and is the wire's own: {mean:.1} against {truth:.1}");
    }

    /// How much a wire BEADS: along it, how its brightest pixel in each row
    /// varies (over their mean), averaged over the shifts -- a diagonal wire
    /// whose light lands on one pixel in a row and on two in the next.
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

    /// A WIRE AT ANY SLANT. Fragments sample the distance across an upright
    /// wire a pixel apart and across a diagonal one 0.71 apart: a kernel that
    /// adds up only at the first lets a diagonal wire's light land on one
    /// pixel of a row and on two in the next -- beads, which crawled along a
    /// lamp's cage as the head moved (headset `lamp_by_pillar-mQ`,
    /// 2026-10-01: the cage 14% worse with a tent one pixel either side). The
    /// thin pass's kernel adds up at both: upright, diagonal and between,
    /// each row of the wire keeps its light as it slides, its brightest pixel
    /// hardly changes along it, and its pixels move with it. Measured here
    /// (2026-10-01), beads at 22.5 and 26.6 degrees: the tent 16.2% and
    /// 14.0%, this kernel 4.7% and 4.6%; light at 45 degrees varied 4.4%, now
    /// 1.5%. Rows of a few 8-bit pixels: a percent or two, and steps of about
    /// 0.05, are rounding.
    #[test]
    fn a_slanted_wire_neither_beads_nor_swings() {
        let Some(image) = render_thin_wire(3.0, 0.0, None) else { return };
        let full = *image[32].iter().max().unwrap() as f32;
        // Steeper than 45 degrees is the same wire turned a right angle, which
        // the kernel (as the pixels) cannot tell apart; and its rows would
        // run off the wire's ends here.
        for degrees in [0.0f32, 22.5, 26.57, 45.0] {
            let tilt = degrees.to_radians();
            let (mean, cv) = wire_light_across_shifts_at(0.16, Some(1.0), tilt).unwrap();
            let beads = wire_beads(0.16, Some(1.0), tilt).unwrap();
            let steps = wire_steps_at(0.16, Some(1.0), tilt).unwrap();
            // A row crosses a slanted wire along 1/cos of a pixel of it.
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

    /// A part many pixels wide keeps its light, and its middle is drawn as
    /// it was: only its edges take the kernel's soft fall.
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

    /// THE THIN PASS's MEANS ARE EVERY MESH's TERMS. `THIN_SHADE` writes the
    /// sky's and the room's harmonics of the normal's moments, and repeats the
    /// light loop's shadows: held here to the functions they mirror, term for
    /// term, so a change to one that misses the other fails.
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
        let shadows = body(shade_with_sky, "if (l.params.z > 1.5) {\n            c = c * hf(sun_visibility", "c = c * hf(character_shadow(i, world_pos));\n        }");
        let found = thin.windows(shadows.len()).any(|w| w == shadows.as_slice());
        assert!(found, "THIN_SHADE's shadows are not `shade_with_sky`'s:\n{}", shadows.join("\n"));
    }

    /// What a pixel wider than `render_wire`'s side-lit wire shows of it: the
    /// light round its visible side -- `light_contribution`'s diffuse and
    /// highlight, the eye down -z and the lamp square to both -- averaged
    /// across the wire as the screen sees it (evenly in u = sin of the angle
    /// round from the eye), summed finely. Times the lamp's light: what the
    /// albedo multiplies.
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

    /// A WIRE LIT FROM ITS SIDE: its lit half and its shadowed half inside
    /// one pixel, which shows their average. The thin pass shaded each
    /// fragment of the widened band from the normal interpolated there -- the
    /// lit half of the wire on one side of the band, the dark half on the
    /// other -- so as the wire slid, which fragments sampled which half
    /// changed: the lit side of a cage wire beside its bulb crawled at every
    /// millimetre of head movement (headset census, 2026-10-01: J 0.17 on
    /// hall_front's near pendant, the most left in the level). Shaded with
    /// the mean over the stretch of the wire each fragment's kernel reaches
    /// (`THIN_SHADE`), at any slant every row keeps its light as it slides,
    /// its pixels move with it, and the light is the wire's own.
    #[test]
    fn a_side_lit_thin_wire_keeps_its_light_and_its_shading_as_it_slides() {
        let albedo = ((128.0f32 / 255.0 + 0.055) / 1.055).powf(2.4);
        let colour = crate::renderer::tonemap::tonemap(
            glam::Vec3::splat(albedo * side_lit_wire_mean()),
            1.0,
            crate::renderer::tonemap::ToneMapping::Aces,
        )
        .x;
        // A third of a pixel thick, and (zoomed, still a thin part) 1.6.
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
    const RED_BULB: [f32; 3] = [0.9, 0.1, 0.05];

    #[test]
    fn the_emissive_mask_decides_which_parts_glow() {
        // The case a REAL fixture exposed. Poly Haven's hanging_industrial_lamp
        // is one material covering the whole lamp, with emissiveFactor [1,1,1]
        // and an emissive texture that is black everywhere except the bulb.
        //
        // Honouring only the factor lights the entire housing white. The factor
        // is not "how much this material emits" -- it is a TINT on a mask that
        // has to be sampled. glTF says emission is factor x texture.
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
        // The other direction, and why the mask defaults to WHITE rather than
        // black: a material that declares only an emissiveFactor and no texture
        // emits uniformly, and a black default would silence it -- making the
        // factor do nothing at all.
        let Some(dark) = render_mesh_masked([0.0, 0.0, 0.0], 2.0, [255, 255, 255, 255]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let glowing = render_mesh_masked([0.8, 0.8, 0.8], 2.0, [255, 255, 255, 255]).unwrap();
        assert!(glowing[0] > dark[0] + 32, "factor-only emission must work: {glowing:?} vs {dark:?}");
    }

    #[test]
    fn a_non_emissive_material_is_unchanged_by_the_drive() {
        // Every object in every level that is not a fixture goes through this
        // path. Adding emissive must not relight any of them.
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
        // Off has to mean off. If the drive at zero still leaked any emissive,
        // every lamp in a dark level would glow faintly with no way to stop it.
        let Some(off_bulb) = render_mesh(RED_BULB, 0.0, true) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let plain = render_mesh(NONE, 0.0, true).unwrap();
        assert_eq!(off_bulb, plain, "a bulb at zero drive must render as plain material");
    }

    #[test]
    fn a_lit_bulb_still_glows_in_a_completely_dark_room() {
        // THE case the whole feature exists for, and the one a naive
        // implementation gets backwards: a bulb is bright because it is a
        // SOURCE, not because something is shining on it. Multiplying the glow
        // by the lighting makes a lamp go dark exactly when the room does --
        // the single moment it has to be visible.
        // GENUINELY dark: no lamps AND no sky. An earlier version of this used
        // only "no lamps", which still leaves sky ambient -- so the lighting
        // term was non-zero and multiplying the glow by it passed the test just
        // as well as adding it. The test named the right property and could not
        // observe it.
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
        // A red bulb must come out red. Packing or unpacking the channels in
        // the wrong order still lights up, and still passes every test above.
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

    /// THE reason the mesh lightmap stopped being a multiplier.
    ///
    /// A surface no realtime lamp reaches, in a room with no sky, is BLACK --
    /// and a product of black and anything is black, so every bounce the baker
    /// computed for the inside of a lamp shade, a corridor or an alcove was
    /// discarded at the last multiply. This is that exact configuration.
    #[test]
    fn baked_bounce_lights_a_mesh_that_no_lamp_and_no_sky_reaches() {
        // Three levels rather than one threshold: what matters is that the
        // baked value drives the result at all and keeps driving it, not that
        // it clears some particular byte. The absolute numbers are small
        // because they survive an sRGB decode, a mid-grey albedo and a
        // tonemapper -- all of which are correct, and none of which a
        // hand-picked constant would describe.
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

    /// Indirect light reflects off the surface's own colour.
    ///
    /// Added INSIDE the albedo multiply, not outside it: bounce arriving on a
    /// dark material must stay dark, the same way direct light does. Outside
    /// the multiply, a black material lit only by bounce would render as bright
    /// as a white one.
    #[test]
    fn baked_bounce_is_tinted_by_the_material_it_lands_on() {
        // The SAME bounce on two different materials. Added inside the albedo
        // multiply the dark one stays dark; added outside it, the material
        // makes no difference at all and both render the bounce itself.
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

    /// The alpha channel narrows the sky term rather than adding to it.
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

// Scene shader sources, for `multiview::every_scene_shader_survives_the_multiview_transform`.
// Test-only: the gate has to see exactly the text each pipeline is built from,
// and nothing on a development machine can build a multiview pipeline to check.
#[cfg(test)]
pub fn mesh_shader_src() -> String {
    mesh_shader()
}
#[cfg(test)]
pub fn skinned_mesh_shader_src() -> String {
    skinned_mesh_shader()
}

