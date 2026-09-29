use super::lights::wgsl_lights_block;
use super::mesh::{MeshVertex, SkinnedMeshVertex, MAX_SKIN_JOINTS};
use super::pipeline::lightmap_bind_group_layout;
use wgpu::*;

pub struct MeshPipeline {
    pub pipeline: RenderPipeline,
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

    fn new_with_front_face(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mesh_shader"),
            source: ShaderSource::Wgsl(view.shader(mesh_shader()).into()),
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

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("mesh_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(MeshVertex::layout())],
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
            texture_layout,
            model_layout,
            lightmap_layout,
        }
    }

    pub fn create_model_uniform(&self, device: &Device) -> ModelUniform {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("mesh_model_uniform"),
            // mat4 model + vec4 params (params.x is sky visibility).
            size: 80,
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
        let mut data = [0f32; 20];
        data[..16].copy_from_slice(&model.to_cols_array());
        data[16] = sky_vis.clamp(0.0, 1.0);
        data[17] = emissive_drive.max(0.0);
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&data));
    }
}

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
            source: ShaderSource::Wgsl(view.shader(skinned_mesh_shader()).into()),
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
            // mat4 model + vec4 params (params.x is sky visibility).
            size: 80,
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

struct ModelUniform {{ model: mat4x4<f32>, params: vec4<f32> }}
@group(1) @binding(0) var<uniform> model_u: ModelUniform;

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
    // shadow or darkening, which would black it out. See `capsule_visibility`.
    capsule_receiver = false;
    let lit = shade_with_sky(in.world_pos, n, model_u.params.x);
    let tex_color = textureSample(tex, samp, in.uv);
    return vec4<f32>(tonemap(tex_color.rgb * lit), tex_color.a);
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
        max_strength = super::mesh::MeshVertex::MAX_EMISSIVE_STRENGTH,
        max_skin_joints = MAX_SKIN_JOINTS
    )
}

fn mesh_shader() -> String {
    format!(
        r#"
// Group 0 -- the camera, the lights and both shadow maps -- is declared by
// `wgsl_lights_block` below, so there is one description of that layout rather
// than one per shader.

struct ModelUniform {{ model: mat4x4<f32>, params: vec4<f32> }}
@group(1) @binding(0) var<uniform> model_u: ModelUniform;

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;
@group(2) @binding(2) var emissive_tex: texture_2d<f32>;

@group(3) @binding(0) var lm_tex: texture_2d<f32>;
@group(3) @binding(1) var lm_samp: sampler;

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
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
    @location(3) uv2:      vec2<f32>,
    @location(4) emissive: u32,
}}

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
}}

@vertex
fn vs_main(v: VIn) -> VOut {{
    let world_pos = model_u.model * vec4<f32>(v.position, 1.0);
    var out: VOut;
    out.clip      = cam_view_proj() * world_pos;
    out.normal    = (model_u.model * vec4<f32>(v.normal, 0.0)).xyz;
    out.uv        = v.uv;
    out.world_pos = world_pos.xyz;
    out.uv2       = v.uv2;
    out.emissive  = unpack_emissive(v.emissive);
    return out;
}}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let n = normalize(in.normal);
    // RGB is baked direct+bounce and is ADDED; ALPHA is baked sky visibility
    // and is MULTIPLIED into the sky term. The two neutrals are at opposite
    // ends of the range -- black adds nothing, 255 scales by one -- which is
    // exactly what an unbaked mesh's default texture carries.
    let baked = textureSample(lm_tex, lm_samp, in.uv2);
    // Per-texel sky visibility, narrowing the per-OBJECT value the model
    // uniform carries. One number for a whole lamp cannot say that the inside
    // of its shade sees less sky than the top of it.
    // The baked lamps are in this atlas. See `receiver_skips_baked`.
    receiver_skips_baked = true;
    let lit = shade_with_sky(in.world_pos, n, model_u.params.x * baked.a);
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
    return vec4<f32>(tonemap(tex_color.rgb * (lit + baked.rgb) + glow), tex_color.a);
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
        max_strength = super::mesh::MeshVertex::MAX_EMISSIVE_STRENGTH,
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
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        if lit {
            lights.upload(
                &queue,
                &[Light {
                    mask_channel: None,
                    position: glam::Vec3::new(0.0, 0.0, 4.0),
                    direction: glam::Vec3::NEG_Z,
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
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);
        uniforms.upload(&queue, glam::Mat4::IDENTITY, TEST_EYE, &ShadowUpload::disabled());

        let pipeline = MeshPipeline::new(&device, format, &uniforms.layout);
        let model = pipeline.create_model_uniform(&device);
        model.upload_full(&queue, glam::Mat4::IDENTITY, sky_vis, drive);

        // Mid grey base, with a WHITE emissive mask so this measures the
        // factor and the drive. The mask itself is covered separately.
        let tex = crate::renderer::mesh::create_mesh_material_texture(
            &device,
            &queue,
            &pipeline.texture_layout,
            &(base.to_vec(), 1, 1),
            &(mask.to_vec(), 1, 1),
        );
        let lm = create_lightmap_texture(
            &device, &queue, &pipeline.lightmap_layout, &lightmap_texel, 1, 1, None,
        );

        let v = |p: [f32; 3]| MeshVertex {
            position: p,
            normal: [0.0, 0.0, 1.0],
            uv: [0.5, 0.5],
            uv2: [0.5, 0.5],
            emissive: MeshVertex::pack_emissive(emissive),
        };
        let verts = [v([-1.0, -1.0, 0.0]), v([3.0, -1.0, 0.0]), v([-1.0, 3.0, 0.0])];
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
