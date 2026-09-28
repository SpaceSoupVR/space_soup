use wgpu::*;

/// Mip levels kept of the scene colour, for rough reflections.
///
/// Five is enough to blur a reflection past recognition at a sixteenth of the
/// frame's width, which is more blur than any surface this renderer calls
/// reflective will ask for. Every extra level is another downsample pass, so
/// this is a budget rather than a maximum.
pub const SSR_MIPS: u32 = 5;

/// Levels of the min-depth pyramid the march will descend. See
/// `docs/ssr-hi-z-scope-2026-09.md`.
///
/// DELIBERATELY NOT `SSR_MIPS`. The colour chain is sized by how blurred a
/// rough reflection needs to be; this one is sized by how much empty screen a
/// ray should be able to skip in one step. Eight levels take 1680x1760 down to
/// about 13x14, which is coarse enough that a ray crossing the whole frame does
/// it in a handful of steps.
pub const HI_Z_LEVELS: u32 = 8;

/// THE REFLECTION BUFFER'S FORMAT.
///
/// `rgb` is reflected radiance and `a` is confidence, so it carries HDR colour
/// and needs more than 8 bits a channel: the scene's bright spots are what a
/// reflection is mostly made of, and clipping them to 1.0 turns a lamp's
/// reflection into a flat white blob.
///
/// 16-bit float rather than 32: the resolve filters it, a filter is an average,
/// and half precision is well inside what an average of radiance can carry. It
/// also halves the bandwidth, which on a tile GPU is the cost that matters.
pub const REFLECTION_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Paint every reflective pixel by WHY it got the reflection it did.
///
/// A diagnostic, not a feature, and it must be `false` in anything shipped --
/// `the_diagnostic_is_off` asserts that, so turning it on turns a test red on
/// purpose rather than silently shipping a false-colour build.
///
///   BLUE    the surface is too rough to march; the probe answered
///   RED     the ray LEFT THE FRAME; there was nothing on screen to reflect
///   GREEN   the traversal ran out of iterations
///   YELLOW  the ray covered too little SCREEN to be worth walking
///   CYAN    the projection was degenerate; the ray had no depth gradient
///   GREY    a hit, shaded by how much of it survived the fades (white = all)
///
/// The three miss colours used to be one. Four changes were aimed at a green
/// fringe beside the avatar's hand on the assumption it was the iteration cap,
/// and not one of them moved it -- because a single colour cannot tell "spent
/// every iteration" from "refused before starting" (2026-09-17). Splitting them
/// costs nothing in a shipped build, where none of this exists.
///
/// It exists because four rounds of screenshots could not separate "the
/// reflection is wrong" from "the reflection is correct and the thing being
/// reflected is off screen", and those two want opposite fixes.
pub const SSR_DEBUG: bool = false;

pub struct SceneTarget {
    /// `None` when the texture belongs to a `StereoSceneTextures` instead --
    /// this target is then one LAYER of a shared pair. Same for the depth and
    /// MSAA colour below.
    _color_texture: Option<Texture>,
    /// The RESOLVED colour at MIP 0, single-sampled. The render/resolve target.
    ///
    /// Deliberately a single-level view. The one used for SAMPLING has to span
    /// the whole chain, or `mipmap_filter: Linear` has nothing to walk and the
    /// blur silently does nothing while every line of it appears present.
    pub color_view: TextureView,
    /// The WHOLE chain, for sampling. See the note on `color_view`.
    pub sample_view: TextureView,
    /// One single-level view per mip above zero, to render each into.
    pub mip_targets: Vec<TextureView>,
    /// For each mip above zero, a bind group reading the level above it.
    pub mip_sources: Vec<BindGroup>,
    _depth_texture: Option<Texture>,
    /// Depth at the pass's own sample count -- the scene pass's ATTACHMENT.
    pub depth_view: TextureView,
    /// THE DEPTH EVERYTHING ELSE READS: single-sampled, whatever the scene pass
    /// ran at. When the scene pass is not multisampled this is `depth_view`
    /// itself and nothing extra is drawn.
    ///
    /// Reading a MULTISAMPLED depth texture is the thing Meta's mobile guidance
    /// warns about: on a tiled GPU it can force the fragment shader to run per
    /// SAMPLE rather than per pixel, at 2x to 4x the cost, and this renderer
    /// runs 4x MSAA with a march that does up to 38 depth loads per reflective
    /// pixel. Resolving once, to the nearest of the samples, costs one pass and
    /// gets every one of those loads off the multisampled texture.
    ///
    /// It is also the bottom level of the Hi-Z pyramid to come -- see
    /// `docs/ssr-hi-z-scope-2026-09.md`.
    pub resolved_depth_view: TextureView,
    /// THE WHOLE MIN-DEPTH PYRAMID, for the march to descend. Level 0 is the
    /// resolved depth; each level above holds the NEAREST depth of the texels
    /// below it, so a ray in front of a level's value cannot have hit anything
    /// inside it and can skip the lot.
    pub hi_z_view: TextureView,
    /// One single-level view per level above zero, to render each into.
    hi_z_targets: Vec<TextureView>,
    /// For each level above zero, a bind group reading the level below it.
    hi_z_sources: Vec<BindGroup>,
    resolved_depth_texture: Texture,
    /// Reads the multisampled depth, for the resolve pass. `None` when there is
    /// nothing to resolve.
    resolve_src: Option<BindGroup>,
    /// THE BLIT'S OWN, reading the scene pass's depth directly.
    ///
    /// The blit primes the eye buffer's depth from the scene's, and everything
    /// drawn after it -- every reflective surface -- depth-tests against what it
    /// writes. Pointing it at the resolved copy along with the march made one
    /// copy responsible for both the reflections AND whether anything drew at
    /// all: on the Quest the reflective surfaces silently stopped appearing and
    /// the false-colour view came back as the ordinary picture, with no error
    /// anywhere. The blit reads depth ONCE per pixel, so it has nothing to gain
    /// from the copy and everything to lose by depending on it.
    pub blit_bind_group: BindGroup,
    pub bind_group: BindGroup,
    /// The multisampled colour the scene pass actually draws into, when
    /// `samples > 1`. It resolves into `color_view` and is never stored, so on
    /// a tile GPU it lives and dies in tile memory.
    pub msaa_color_view: Option<TextureView>,
    _msaa_color_texture: Option<Texture>,
    pub samples: u32,
}

/// The colour and depth a STEREO scene pass shares between the two eyes.
///
/// A multiview pass draws both eyes in one go, into the LAYERS of a single
/// attachment, so the two eyes cannot own separate textures the way they do in
/// the per-eye path. This owns the shared ones and hands out the `D2Array`
/// views the pass attaches; the per-eye `SceneTarget`s that come with it view
/// one layer each and everything downstream -- the depth resolve, the pyramid,
/// the mip chain, the reflective draws -- carries on per eye, unchanged.
///
/// Nothing here can be built on the development machine at 4x MSAA: a
/// multisampled array texture needs `Features::MULTISAMPLE_ARRAY`, which is
/// Vulkan-only. At one sample it builds anywhere, which is what the tests use.
pub struct StereoSceneTextures {
    _color: Texture,
    _depth: Texture,
    _msaa_color: Option<Texture>,
    /// What a multiview pass attaches: every layer, one mip.
    pub array_color_view: TextureView,
    pub array_depth_view: TextureView,
    /// The multisampled colour to draw into, resolving into `array_color_view`.
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
    /// Writes the nearest of the multisampled depth samples into a
    /// single-sampled depth texture. See `SceneTarget::resolved_depth_view`.
    resolve_pipeline: Option<RenderPipeline>,
    resolve_layout: BindGroupLayout,
    /// Halves the min-depth pyramid into the next level. Run once per level.
    hi_z_pipeline: RenderPipeline,
    /// The reflection buffer's read layout, shared by the resolve and the
    /// composite -- one texture, no sampler: both read exact texels.
    reflection_layout: BindGroupLayout,
    reflection_resolve_pipeline: RenderPipeline,
    /// The second a-trous pass, taps at double spacing. See
    /// `REFLECTION_RESOLVE_STRIDES`.
    reflection_resolve_wide_pipeline: RenderPipeline,
    hi_z_layout: BindGroupLayout,
    blit_layout: BindGroupLayout,
    /// Halves the scene colour into the next mip. Run once per level.
    downsample_pipeline: RenderPipeline,
    downsample_layout: BindGroupLayout,
    linear_sampler: Sampler,
    scene_texture_layout: BindGroupLayout,
    camera_layout: BindGroupLayout,
    sampler: Sampler,
}

/// HOW MUCH SMALLER THE REFLECTION IS TRACED THAN THE FRAME.
///
/// 1 for now. The whole point of tracing into a buffer is that it CAN be traced
/// smaller -- Frostbite's stochastic SSR and FidelityFX SSSR both trace at half
/// and reconstruct -- and at 2 this is a straight 4x cut in the most expensive
/// thing the frame does.
///
/// 2 since 2026-09-18, having shipped the filter at full resolution first and
/// measured it: the resolve alone cost 8.2 to 8.6 ms an eye even after the
/// depth gate, which is about what the whole inline march cost. Tracing and
/// filtering a quarter of the pixels is the only thing that makes the buffered
/// path cheaper than what it replaced, and it helps the picture twice over --
/// the filter's reach DOUBLES in full-resolution pixels, which is what the
/// residual comb beside the hand needs, and averaging 2x2 of the trace damps
/// the per-pixel instability that the filter was turning into visible blobs.
pub const REFLECTION_SCALE: u32 = 2;

/// Whether the renderer STARTS with the buffered reflection path on.
///
/// Off. The inline path is the one with a year of headset time behind it, and
/// the buffered one cannot be validated anywhere but the headset -- so the
/// build that introduces it should come up looking exactly like the last one
/// and be switched over deliberately. See `XrRenderer::set_buffered_reflections`.
pub const BUFFERED_REFLECTIONS: bool = false;

/// How far the resolve reaches for a neighbour, in reflection-buffer texels.
///
/// FidelityFX SSSR uses 15 Halton-distributed taps at desktop resolution and
/// 0.34 ms. This is a tile GPU with two eyes and 13.9 ms for everything, so it
/// takes 8 on a fixed ring -- enough to cross the hit/miss boundary that makes
/// the comb, and cheap enough to be worth it.
pub const REFLECTION_FILTER_RADIUS: i32 = 2;

/// THE TAP SPACING OF EACH RESOLVE PASS, in order.
///
/// A-trous: the same 25 taps, spread wider each pass, so reach doubles per pass
/// at constant cost. `[1, 2]` reaches 2 + 4 = 6 texels -- 12 frame pixels at
/// half resolution -- against the 2 texels a single pass managed, which
/// softened the comb's edge and left its interior alone.
///
/// A third pass at 4 would reach 20 frame pixels for another ~1 ms an eye. Add
/// it only if the measurement asks for it.
pub const REFLECTION_RESOLVE_STRIDES: [i32; 2] = [1, 2];

/// THE REFLECTION BUFFER AND ITS RESOLVE, per eye.
///
/// Two textures of the same shape. The trace pass draws reflective geometry
/// into `trace_view` writing radiance and confidence; the resolve pass reads it
/// and writes `filtered_view`, where a pixel whose own ray found nothing has
/// borrowed from neighbours whose rays did. The composite then samples
/// `filtered_view` instead of marching.
///
/// `depth_view` belongs to the TRACE pass alone. It is cleared and holds only
/// reflective geometry, so two reflective surfaces overlapping resolve
/// correctly. It deliberately does NOT contain the rest of the scene: a
/// reflective surface hidden behind a wall may write here, and that is
/// harmless, because the composite pass draws the same geometry against the
/// real depth buffer and never samples the places it wrote.
pub struct ReflectionTarget {
    _trace_texture: Texture,
    _filtered_texture: Texture,
    /// The first a-trous pass writes here and the second reads it, so the
    /// LAST pass always lands in `filtered_view` and the composite's bind
    /// group never has to change which texture it points at.
    _scratch_texture: Texture,
    _depth_texture: Texture,
    /// The trace pass's colour attachment: radiance in `rgb`, confidence in `a`.
    pub trace_view: TextureView,
    /// The trace pass's own depth attachment. See the note above.
    pub depth_view: TextureView,
    /// The resolve pass's colour attachment.
    pub filtered_view: TextureView,
    /// The intermediate between the two resolve passes.
    pub scratch_view: TextureView,
    /// Reads `scratch_view`, for the second resolve pass.
    pub scratch_src: BindGroup,
    /// Reads `trace_view`, for the resolve pass.
    pub resolve_src: BindGroup,
    /// Reads `filtered_view`, for the composite.
    pub composite_src: BindGroup,
    pub width: u32,
    pub height: u32,
}

impl StereoSceneTextures {
    /// The shared colour and depth, two layers each, and the `D2Array` views a
    /// multiview pass attaches.
    ///
    /// The colour carries the whole mip chain, exactly as the per-eye texture
    /// does, so each eye's reflection blur reads its own layer's mips. The
    /// array views are MIP 0 and every layer: that is what a pass draws into.
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
                // See the per-eye texture: nothing samples it, so a tile GPU
                // never has to write it out, or even back it with memory.
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TRANSIENT_ATTACHMENT,
                view_formats: &[],
            })
        });
        // EVERY LAYER, ONE MIP. `array_layer_count: None` means "the rest",
        // which is both layers here; the mip level must still be pinned or the
        // view spans the chain and cannot be an attachment.
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
    /// The single-sampled depth copy and its pyramid, for tests that read it
    /// back.
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
    /// The reflection buffer for one eye, at `REFLECTION_SCALE` of the frame.
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
                // COPY_DST and COPY_SRC so a test can put a known pattern in
                // and read the filtered result out. The resolve is the one pass
                // whose OUTPUT is the whole point -- a pipeline that builds and
                // filters nothing would look identical from the outside, which
                // is the failure this project keeps hitting.
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
        // NOT MULTISAMPLED, and not shared with anything. See `ReflectionTarget`.
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

    /// THE COMPOSITE'S BIND GROUP: the scene colour it falls back to, and the
    /// FILTERED REFLECTION where the march's depth pyramid used to be.
    ///
    /// Built against `scene_texture_layout`, unchanged, so the composite
    /// pipeline is the same shape as every other reflective one and stays
    /// inside the four bind groups this hardware allows. See
    /// `wgsl_ssr_composite_block`.
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

    /// The layout both the resolve and the composite read the reflection
    /// buffer through.
    pub fn reflection_layout(&self) -> &BindGroupLayout {
        &self.reflection_layout
    }

    /// Filter the trace into `filtered_view`: a pixel whose ray found nothing
    /// borrows from neighbours whose rays did.
    pub fn resolve_reflections(
        &self,
        encoder: &mut CommandEncoder,
        target: &ReflectionTarget,
        timestamps: Option<RenderPassTimestampWrites<'_>>,
    ) {
        // TWO PASSES, THE SECOND WIDER. See `REFLECTION_RESOLVE_STRIDES`. The
        // first lands in the scratch buffer and the second in `filtered_view`,
        // so whatever the pass count, the composite's bind group always points
        // at the finished result.
        //
        // The timestamps span both: the slot measures the resolve, not one
        // half of it.
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
            // READ-ONLY: `depth_ops: None` says this attachment is never
            // written, which is what lets the trace's depth be tested here
            // without being touched.
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

    /// `depth_samples` must match the scene target's depth texture.
    ///
    /// It exists because wgpu can resolve a colour attachment and cannot
    /// resolve a depth one: multisampling the scene pass leaves the depth
    /// multisampled, and every shader that reads it has to say so -- in the
    /// bind group layout AND in the WGSL type. A mismatch here fails pipeline
    /// creation rather than rendering wrongly, which is the good outcome.
    pub fn new_with_depth_samples(device: &Device, format: TextureFormat, depth_samples: u32) -> Self {
        // EVERYTHING THAT READS DEPTH NOW READS A SINGLE-SAMPLED COPY. The
        // sample count only decides what the RESOLVE pass reads, never what any
        // other shader is typed against -- which is why no shader below takes a
        // sample count any more. It used to, and the callers that still passed
        // `samples > 1` after the switch built four pipelines that every
        // reflective draw needed and none of which existed: reflections simply
        // stopped appearing, and the false-colour view came back as the
        // ordinary picture (headset, 2026-09-17).
        let scene_texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssr_scene_texture_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        // FILTERABLE, because the reflection is now read from a
                        // mip with a filtering sampler. Declared unfilterable
                        // this fails at pipeline creation naming the pair --
                        // which is the good outcome, but only because a test
                        // builds the pipeline on a real device.
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
                        // The RESOLVED depth: a single-channel colour texture,
                        // never sampled with a filter, only loaded.
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    // FILTERING, so the reflection can be read from a blurred
                    // mip. A non-filtering sampler would compile and then point
                    // sample every level, which looks like the blur is simply
                    // not working.
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
            // The whole point: without this the chain is never walked.
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
                    // UNFILTERABLE and only ever `textureLoad`ed. Confidence in
                    // alpha must not be interpolated with the radiance it
                    // weights -- a filtered read would mix a confident texel's
                    // colour with an empty one's alpha and invent reflections
                    // at the boundary, which is the artefact this removes.
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
                // ONLY WHERE THE TRACE DREW. Measured on the headset
                // 2026-09-18, the resolve at full resolution cost 7 to 17 ms an
                // eye: 25 texture loads across every pixel of a 1680x1760 frame
                // when only the reflective surfaces have anything to filter.
                // FidelityFX SSSR classifies tiles for exactly this reason, and
                // the trace's own depth buffer is a classification we already
                // have for free.
                //
                // NO DEPTH WRITE -- this only reads.
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
        self.create_scene_target_layer(device, format, width, height, samples, None)
    }

    /// THE STEREO PAIR: one colour and one depth, two layers each, plus a
    /// `SceneTarget` viewing each layer.
    ///
    /// The eyes share the textures so that a multiview pass can draw both at
    /// once; they do not share anything downstream. See `StereoSceneTextures`.
    ///
    /// At `samples > 1` this needs `Features::MULTISAMPLE_ARRAY` — without it
    /// the textures fail to create and, because a failed creation is not an
    /// error the caller sees, the renderer must check the feature before
    /// calling rather than after.
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

    /// One eye's view bundle, over its own textures or over one layer of a
    /// shared pair.
    fn create_scene_target_layer(
        &self,
        device: &Device,
        format: TextureFormat,
        width: u32,
        height: u32,
        samples: u32,
        shared: Option<(&StereoSceneTextures, u32)>,
    ) -> SceneTarget {
        // WHICH LAYER THIS TARGET LOOKS AT, and whether it owns what it looks
        // at. Every view below is explicitly `D2` over ONE layer: the default
        // descriptor infers `D2Array` as soon as a texture has two layers, and
        // a shader typed against `texture_2d` would then fail to bind.
        let layer = shared.map_or(0, |(_, eye)| eye);
        // A CHAIN, not one level: a reflection in a rough surface is a blurred
        // reflection, and the blur comes from reading a smaller mip. Without it
        // every reflection is pin-sharp whatever the surface is made of, which
        // is why marble came back looking like polished chrome.
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

        // Single-sampled depth, and the pass that fills it. Skipped entirely
        // when the scene pass is already single-sampled: then the attachment IS
        // the readable copy.
        // AN ORDINARY COLOUR TARGET, not a depth one.
        //
        // The first version wrote this as `Depth32Float` from a pass with a
        // depth attachment and no colour attachments. That validates, and it
        // records and submits on Metal -- and on the Quest it took the app down
        // inside the VR driver on startup. A single-channel colour target makes
        // the copy an ordinary full-screen pass, exactly like the mip chain
        // beside it, and it is also the shape the Hi-Z pyramid wants: a colour
        // mip chain reduces with the same machinery the scene colour already
        // uses, where a depth chain would need an attachment per level.
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
            // COPY_SRC so a test can read it back and check the copy actually
            // copied. See `the_resolve_actually_writes_the_depth`.
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Level 0 alone for the resolve's attachment and for anything that
        // wants the full-resolution depth; the WHOLE chain for the march, which
        // walks up and down it.
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
        // ALWAYS, even at one sample: the copy is what every reader is typed
        // against, so there is one code path rather than two.
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
                // The WHOLE chain here, not `color_view` -- see its note.
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&sample_view) },
                // THE WHOLE PYRAMID, not just level 0: the march walks up and
                // down it, and `textureLoad` takes the level as an argument.
                // The blit, which wants the scene pass's own depth, has its own
                // bind group.
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
                // NO texture binding: nothing samples it, which is what lets a
                // tile GPU keep it on chip and never write it out.
                //
                // TRANSIENT as well: the samples live only in tile memory
                // (cleared on load, discarded on store, resolved on the way
                // out), so the texture needs no memory behind it at all --
                // Vulkan's lazily allocated memory, Metal's memoryless. Two
                // eyes of 4x colour at this size is ~70 MB the headset no
                // longer allocates. wgpu enforces the contract: a pass that
                // loaded or stored these samples would be refused.
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TRANSIENT_ATTACHMENT,
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

    /// Copy the multisampled depth into the single-sampled one everything else
    /// reads. A no-op when the scene pass was not multisampled.
    ///
    /// MUST be recorded in the SAME encoder as the scene pass, after it has
    /// ended and BEFORE anything that reads `resolved_depth_view`.
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
                    // Every texel is written, so there is nothing to load.
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

    /// Fill the min-depth pyramid above level 0, smallest last.
    ///
    /// MUST be recorded after `resolve_depth`, which writes level 0, and in the
    /// same encoder.
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
                        // Every texel is written, so there is nothing to load.
                        load: LoadOp::Clear(Color::WHITE),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                // Only the LAST level closes the span that `resolve_depth`
                // opened, so the slot covers the whole block.
                timestamp_writes: if i == last { last_timestamps.take() } else { None },
                ..Default::default()
            });
            pass.set_pipeline(&self.hi_z_pipeline);
            pass.set_bind_group(0, &target.hi_z_sources[i], &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Fill the scene colour's mip chain, smallest last.
    ///
    /// MUST be recorded in the SAME encoder as the pass that drew the scene,
    /// after that pass has ended and before the encoder is submitted. Recorded
    /// anywhere else the chain is a frame stale, and a reflection of last
    /// frame's world is far harder to recognise as wrong than a missing one.
    pub fn generate_mips(&self, encoder: &mut CommandEncoder, target: &SceneTarget) {
        for (i, view) in target.mip_targets.iter().enumerate() {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("ssr_downsample"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        // Every texel is written, so there is nothing to load.
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

/// One level of the min-depth pyramid: the NEAREST depth of the texels below.
///
/// Min, because the question a Hi-Z traversal asks of a cell is "could the ray
/// have hit ANYTHING in here?" -- and the answer is no exactly when the ray is
/// still in front of the nearest thing the cell contains. A max, or an average,
/// would let the ray skip over geometry.
///
/// THE ODD CASE IS THE WHOLE DIFFICULTY. A level of odd width halves to a level
/// that does not cover it: a plain 2x2 reduction drops the last column, the
/// pyramid stops being a bound on what is underneath, and the traversal walks
/// through whatever was in it. 1680x1760 hits that at the fifth level. Reading a
/// 3x3 whenever the parent is odd covers everything; the overlap makes some
/// cells report a nearer surface than they strictly contain, which only costs a
/// descent the ray did not need -- the safe direction to be wrong in.
/// THE RESOLVE: turn a buffer of confident hits and blank misses into one with
/// no cliff between them.
///
/// The march leaves a hard boundary wherever a ray stopped and its neighbour
/// did not. On the headset that boundary IS the artefact: the comb beside the
/// avatar's hand is confident hits interleaved with rays that terminated behind
/// it, and the doorway and light-fixture artefacts are the same boundary
/// against rays that left the frame. The false-colour view showed it directly
/// on 2026-09-18 -- orange against greyscale, with a ragged stepped edge.
///
/// No per-pixel rule can fix that, because each pixel's own answer is correct.
/// It is fixed by letting a pixel BORROW from its neighbours, weighted by how
/// much each neighbour's ray is worth trusting, which is what the trace wrote
/// into alpha. Every reference does this after the trace rather than inside it:
/// FidelityFX SSSR's prefilter, Frostbite's stochastic resolve.
///
/// Confidence-weighted rather than a plain blur. A pixel that found a good
/// reflection keeps it -- its own sample dominates because its own confidence
/// is high -- and a pixel that found nothing is filled almost entirely by
/// whichever neighbours did. So the reflection does not soften where it is
/// working, only where it was missing.
fn reflection_resolve_shader(stride: i32) -> String {
    format!(
        r#"
@group(0) @binding(0) var refl_src: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    // AT THE FAR PLANE, ON PURPOSE. The pipeline tests `Greater` against the
    // trace pass's depth, which is 1.0 wherever nothing was drawn and the
    // geometry's own depth wherever something was. A fragment at 1.0 therefore
    // FAILS on empty pixels and PASSES on reflective ones, so the filter runs
    // only where there is something to filter and the hardware rejects the rest
    // before the shader starts. See `resolve_reflections`.
    return vec4<f32>(pos[vi], 1.0, 1.0);
}}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {{
    let size = vec2<i32>(textureDimensions(refl_src));
    let here = vec2<i32>(frag.xy);
    let centre = textureLoad(refl_src, here, 0);

    // THE CENTRE COUNTS FOR MORE THAN ANY NEIGHBOUR.
    //
    // Its own ray is the only one that is actually about this pixel; the
    // neighbours are an approximation that gets worse with distance. Weighting
    // it heavily is what keeps a working reflection sharp -- without it this is
    // just a blur, and it would cost the detail the march went to such trouble
    // to get right.
    var sum = centre.rgb * centre.a * {centre_weight:.1};
    var weight = centre.a * {centre_weight:.1};

    for (var dy = -{radius}; dy <= {radius}; dy = dy + 1) {{
        for (var dx = -{radius}; dx <= {radius}; dx = dx + 1) {{
            if (dx == 0 && dy == 0) {{
                continue;
            }}
            // SPACED OUT BY `stride`. Two passes, the second at double
            // spacing, is the a-trous arrangement every denoiser since SVGF
            // uses: reach DOUBLES per pass while the tap count stays the same.
            // One pass of radius 4 would be 81 taps for the same reach; this is
            // 25 twice.
            //
            // The residual comb beside the avatar's hand is why. That region is
            // tens of pixels across, and a single radius-2 pass reaches 2
            // texels -- 4 frame pixels at half resolution -- so it softened the
            // edge and left the interior (headset, 2026-09-18).
            let at = here + vec2<i32>(dx, dy) * {stride};
            // CLAMPED, not wrapped and not skipped. An out-of-range
            // `textureLoad` returns zero, which here is a confident black --
            // and a frame's worth of confident black around the edge would
            // darken every reflection that reaches it. The same mistake in the
            // march cost several builds; see `SSR_HI_Z_LEVELS`.
            let p = clamp(at, vec2<i32>(0), size - vec2<i32>(1));
            let n = textureLoad(refl_src, p, 0);
            // Nearer neighbours count for more, so the fill is smooth rather
            // than a flat disc with its own edge.
            let d = f32(dx * dx + dy * dy);
            let falloff = 1.0 / (1.0 + d);
            let w = n.a * falloff;
            sum = sum + n.rgb * w;
            weight = weight + w;
        }}
    }}

    if (weight <= 1e-5) {{
        // NOTHING within reach found anything. Confidence zero, and the
        // composite hands the pixel to the probe -- which is the right answer
        // and always was. What this pass removes is the CLIFF beside it, not
        // the fallback itself.
        return vec4<f32>(0.0);
    }}

    // THE COLOUR IS THE WEIGHTED MEAN; THE CONFIDENCE IS THE PIXEL'S OWN,
    // FLOORED BY ITS NEIGHBOURHOOD'S.
    //
    // This filter exists to FILL a pixel whose ray found nothing. It must never
    // ATTENUATE one whose ray succeeded.
    //
    // It used to hand on the neighbourhood mean alone, and that is exactly
    // wrong at a boundary: a pixel with a perfect hit two texels from a miss
    // was dimmed by its neighbours' failure. Reflections are mostly boundary --
    // measured here, the last fully confident pixel before a gap came back at
    // 0.773 -- so the whole reflection came back weaker with the buffered path
    // on, and at half resolution, where the footprint covers more of the
    // reflection, it got worse rather than better (headset, 2026-09-18).
    //
    // `max` of the two says: keep what your own ray earned, and if it earned
    // nothing, take what the neighbourhood can lend. A pixel deep inside a
    // missing region still has both near zero, so the handover to the probe is
    // still a ramp and the reflection still cannot spread past the surface that
    // produced it.
    let taps = {taps:.1};
    return vec4<f32>(sum / weight, clamp(max(centre.a, weight / taps), 0.0, 1.0));
}}
"#,
        radius = REFLECTION_FILTER_RADIUS,
        stride = stride,
        centre_weight = REFLECTION_CENTRE_WEIGHT,
        // The centre's own weight plus every neighbour's falloff, which is what
        // `weight` sums to when everything around is fully confident.
        taps = reflection_filter_total_weight(),
    )
}

/// How much the pixel's own sample outweighs one adjacent neighbour.
const REFLECTION_CENTRE_WEIGHT: f32 = 4.0;

/// The largest `weight` the resolve can accumulate -- every tap fully
/// confident. Dividing by it turns the sum into a fraction, so the confidence
/// handed to the composite is "how much of my neighbourhood found something".
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
    r#"
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
"#
    .to_string()
}}

/// The nearest of the multisampled depth samples, written as depth.
///
/// MIN, not sample zero. The depth buffer stands for "the nearest surface at
/// this pixel" everywhere it is used -- the march tests against it, the blit
/// depth-tests against it, and the Hi-Z pyramid to come is a pyramid of minima.
/// Taking one sample would let an edge pixel report a surface that covers a
/// quarter of it.
fn depth_resolve_shader(samples: u32) -> String {{
    let src_ty = if samples > 1 {{
        "texture_depth_multisampled_2d"
    }} else {{
        "texture_depth_2d"
    }};
    format!(
        r#"
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
"#
    )
}}

#[cfg(test)]
mod composite_block_tests {
    use super::*;

    /// The composite declares the reflectivity cap itself, because the march
    /// that normally declares it is not in that shader. Two declarations of one
    /// number is a drift hazard, so they are pinned against each other.
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

    /// THE UPSAMPLE WEIGHTS EACH TAP BY ITS OWN CONFIDENCE.
    ///
    /// The reflection is traced at `REFLECTION_SCALE`, so at anything above 1
    /// the composite is reading between texels. It must not do that with a
    /// filtering sampler: `a` weights `rgb`, and hardware filtering
    /// interpolates the two independently, so a confident texel's colour
    /// blended with an empty one's alpha invents a reflection exactly at the
    /// boundary this path exists to smooth. Four loads weighted by confidence
    /// is the same interpolation in the right order.
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
        // And the scale actually reaches the shader, or the upsample reads the
        // wrong texels and every reflection is offset.
        assert!(
            composite.contains(&format!("clip_xy / {REFLECTION_SCALE}.0")),
            "the composite does not divide by REFLECTION_SCALE ({REFLECTION_SCALE}), \
             so it reads the reflection buffer at the wrong scale",
        );
    }

    /// The composite reads the reflection where the march read the pyramid.
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

    /// Fill the trace pass's depth attachment, standing in for the geometry
    /// the trace would have drawn. 0.0 is "reflective here" (the resolve tests
    /// `Greater` against a fragment at 1.0); 1.0 is "nothing drawn".
    ///
    /// TEST-ONLY, and it must stay that way: this is a pass with a depth
    /// attachment and NO colour attachments, which validates, records and
    /// submits on Metal and took the Quest's VR driver down on startup when the
    /// depth resolve was first written that way. Nothing here ever runs on the
    /// headset.
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

    /// IEEE half, by hand, so a test does not pull in a dependency the
    /// renderer does not otherwise need.
    fn f16_bits(v: f32) -> u16 {
        // Only the two values this test writes; anything else is a mistake
        // rather than a rounding question.
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

    /// THE FILTER'S WEIGHTS MUST SUM TO WHAT THE SHADER DIVIDES BY.
    ///
    /// The shader accumulates the centre's weight plus every neighbour's
    /// falloff and then divides by a number computed on the CPU. If the two
    /// disagree, confidence comes out scaled -- above 1 it clamps and the
    /// handover happens too early, below 1 the reflection never reaches full
    /// strength anywhere and every surface is permanently half-faded.
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

    /// THE TRACE PASS'S ATTACHMENTS RECORD AND SUBMIT.
    ///
    /// An `Rgba16Float` colour beside a `Depth32Float` that nothing else in
    /// this renderer pairs, cleared to TRANSPARENT and discarded. Pipeline
    /// creation says nothing about whether a pass with this shape is accepted:
    /// the depth-resolve pass validated, recorded and submitted on Metal and
    /// then took the Quest's VR driver down on startup. So this records and
    /// submits it, which is the cheapest place to find out.
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

    /// A PIXEL WHOSE OWN RAY SUCCEEDED MUST NOT BE DIMMED BY ITS NEIGHBOURS.
    ///
    /// The headset showed the whole reflection "less pronounced" with the
    /// buffered path on, not just at the boundary (2026-09-18). This is why:
    /// the confidence handed to the composite was the MEAN over the whole
    /// neighbourhood, so a pixel with a perfect hit sitting two texels from a
    /// miss was attenuated by its neighbours' failure. Reflections are full of
    /// boundaries, so most of one is within reach of a miss -- and at half
    /// resolution the footprint covers more of the reflection, which is why it
    /// got WORSE rather than better.
    ///
    /// The filter exists to FILL misses from neighbours, never to attenuate
    /// hits by them.
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
        // The LAST fully confident pixel before the miss. Its own ray hit.
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

    /// THE RESOLVE MUST NOT RUN WHERE THE TRACE DREW NOTHING.
    ///
    /// The measurement that prompted this: at full resolution the resolve cost
    /// 7 to 17 ms an eye, because it filtered every pixel of a 1680x1760 frame
    /// when only the reflective surfaces have anything to filter. The trace's
    /// depth buffer already says which those are.
    ///
    /// This fills the trace buffer with confident red EVERYWHERE and then says
    /// -- through the depth -- that no geometry was drawn. Every pixel must
    /// come back at the clear value, because every fragment was rejected before
    /// the shader ran. If the gating is lost, the whole target comes back red
    /// and this fails.
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
        // 1.0 is the trace pass's CLEAR value: nothing was drawn anywhere.
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

    /// A PIXEL THAT FOUND NOTHING, BESIDE ONE THAT DID, MUST NOT STAY AT ZERO.
    ///
    /// This is the comb, reduced to two pixels. The trace buffer is filled with
    /// a confident red on one side and nothing on the other; after the resolve
    /// the empty side next to the boundary must carry some of the red and a
    /// confidence between the two, or the cliff is still there and this whole
    /// pass does nothing.
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

        // Left half fully confident red, right half nothing at all.
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
        // THE RESOLVE DEPTH-TESTS now, so the depth has to say where geometry
        // is. Clearing it to 0.0 means "reflective everywhere", which is the
        // condition this test is about -- the gating itself is measured by
        // `the_resolve_skips_pixels_the_trace_never_drew`.
        prime_depth(&device, &mut encoder, &target, 0.0);
        ssr.resolve_reflections(&mut encoder, &target, None);
        // Read the filtered result back.
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
        // THE REACH, which is what the second a-trous pass is for.
        //
        // A single radius-2 pass reaches 2 texels and leaves everything beyond
        // at zero -- which is why the comb beside the avatar's hand softened at
        // its edge and stayed put in its interior. Two passes, the second at
        // double spacing, reach 6.
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

    /// THE STEREO PAIR IS REALLY ONE TEXTURE WITH TWO LAYERS, and each eye's
    /// views really look at its own layer.
    ///
    /// Run at ONE sample, which is the only thing this machine can do: a
    /// multisampled array texture needs `Features::MULTISAMPLE_ARRAY`, which is
    /// Vulkan-only, and wgpu has no multiview on Metal either. So what is being
    /// checked here is the LAYERING -- the part that is ordinary WebGPU and can
    /// be wrong in silent ways -- and not multiview itself, which only the
    /// headset can validate.
    ///
    /// The silent way it goes wrong: `TextureViewDescriptor::default()` infers
    /// `D2Array` the moment a texture has two layers, and a shader typed
    /// against `texture_2d` then fails to bind. Every per-eye view therefore
    /// pins `D2` over one layer, and this is what says so.
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

        // Each eye clears its OWN layer to a different colour. If the two
        // targets were pointed at the same layer, or if a view spanned both,
        // the second clear would overwrite the first and the readback below
        // would show one colour twice.
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

        // The array view exists and is what a multiview pass would attach.
        assert!(
            stereo.array_msaa_color_view.is_none(),
            "there is no multisampled colour at one sample",
        );
        assert_eq!(targets.len(), 2, "one view bundle per eye");
        // Neither eye owns the shared textures; the pair does. Owning them
        // twice would mean two separate textures and no multiview at all.
        for (eye, target) in targets.iter().enumerate() {
            assert!(
                target._color_texture.is_none() && target._depth_texture.is_none(),
                "eye {eye} owns its own colour or depth, so the two eyes are \
                 not layers of ONE texture and a multiview pass cannot draw \
                 both at once",
            );
        }
    }

    /// Whether the headset can have a stereo pass AT ALL comes down to one
    /// feature, and it is the reason this renderer is on wgpu 28.
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

mod depth_resolve_tests {
    use super::*;

    /// It must read EVERY sample, and take the nearest.
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

    /// THE PASS ITSELF RUNS. Pipeline creation validates the shader; it says
    /// nothing about whether the encoder accepts a pass with a depth
    /// attachment and NO colour attachments, or whether the driver survives
    /// it. The first build of this shipped straight to the headset and took
    /// the app down inside the VR driver on startup, so this test exists to
    /// fail on a desktop instead.
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

    /// DOES THE COPY ACTUALLY COPY?
    ///
    /// The pass records and submits; that says nothing about whether anything
    /// lands in the texture. If it silently writes nothing the texture holds
    /// its clear value, every ray then measures the scene as infinitely far
    /// away, no ray ever hits, and the whole march runs its full length and
    /// finds nothing -- which reads on the headset as reflections that will not
    /// turn on, and in the false-colour view as a frame painted entirely GREEN.
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
        // Put a known value in every sample of the multisampled depth.
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
            size: (bytes * SIZE) as u64,
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

    /// THE MULTISAMPLED SCENE COLOUR IS TRANSIENT: cleared, drawn, resolved
    /// and discarded, it never needs memory behind it -- and the resolve still
    /// carries the picture out. A pass that tried to keep the samples is
    /// refused, which is what makes the promise to the driver safe to make.
    #[test]
    fn the_multisampled_scene_colour_resolves_without_ever_being_stored() {
        let Some((device, queue)) = crate::renderer::pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        // 64 texels of 4 bytes: a row is the 256 bytes a copy must align to.
        const SIZE: u32 = 64;
        let ssr = SsrPipelines::new_with_depth_samples(&device, TextureFormat::Rgba8UnormSrgb, 4);
        let target = ssr.create_scene_target_multisampled(&device, TextureFormat::Rgba8UnormSrgb, SIZE, SIZE, 4);
        let msaa = target.msaa_color_view.as_ref().expect("a 4x target has multisampled colour");
        // Resolved into a texture of the test's own, which it can copy out.
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

        // Keeping the samples breaks the promise, and is refused rather than
        // silently written to memory that was never there.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        queue.submit(Some(pass(StoreOp::Store).finish()));
        let err = pollster::block_on(scope.pop());
        assert!(err.is_some(), "storing a transient attachment was accepted");
    }

    /// And nothing that marches may be typed against a multisampled depth
    /// texture any more -- that read is what Meta's guidance says can cost 2x
    /// to 4x under MSAA, and getting off it is the whole point of the copy.
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
    //! THE PYRAMID HAS TO BE A BOUND, not an approximation.
    //!
    //! A Hi-Z traversal skips a whole cell when the ray is still in front of the
    //! nearest thing the cell contains. If a level ever reports something
    //! FURTHER than what is really underneath it, the ray skips geometry it
    //! should have hit and the reflection shows what is behind a wall. The
    //! classic way to get that wrong is an odd-sized level: halving 105 gives
    //! 52, a plain 2x2 reduction never reads column 104, and the bound silently
    //! stops holding. 1680x1760 hits its first odd level at the fifth step.
    use super::*;

    fn pattern(w: u32, h: u32) -> Vec<f32> {
        // Deterministic, and varied enough that a dropped row or column changes
        // some cell's answer.
        let mut seed = 0x1234_5678u32;
        (0..w * h)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1u32 << 24) as f32
            })
            .collect()
    }

    /// The pyramid levels, read back.
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
            // `bytes_per_row` has to be a multiple of 256 for a texture copy.
            let row = (lw * 4).div_ceil(256) * 256;
            let buf = device.create_buffer(&BufferDescriptor {
                label: None,
                size: (row * lh) as u64,
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

    /// With no odd level in the chain, every cell is EXACTLY the nearest thing
    /// underneath it.
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

    /// With odd levels, every cell must still BOUND what is underneath it: never
    /// further than the true nearest. Being nearer is allowed -- the 3x3 overlap
    /// costs a descent, not a wrong answer.
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
/// Halve one mip into the next.
///
/// A plain bilinear tap of the level above: the sampler already averages the
/// four texels that collapse into one, and it does so AFTER decoding sRGB, so
/// the average is taken in linear light. Doing it by hand on the encoded bytes
/// is the classic way to make every mip darker than the one before it.
fn downsample_shader() -> String {
    r#"
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
"#
    .to_string()
}

/// The blit reads the SCENE PASS'S depth, at whatever sample count it ran at.
///
/// `ms_depth` comes from `depth_samples` a few lines from where the layout is
/// built, and no caller outside this file can pass it. That is deliberate: the
/// version of this that took the sample count from its callers had one of them
/// disagree with the layout, and every reflective pipeline silently failed to
/// build.
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

/// WHAT `ssr_reflect` HANDS BACK.
///
/// The march is one piece of text; only its four terminal statements and its
/// return type differ between these. That matters more than it looks: the
/// reflection trace pass and the shipped forward pass must agree about where a
/// ray goes and what stops it, and the only way to guarantee that is for there
/// to be one march rather than two that drift.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SsrOutput {
    /// `vec3`: the surface's colour with the reflection already mixed in by
    /// `weight * fade`. What the forward pass has always done.
    Blended,
    /// `vec3`: the false-colour diagnostic. See `ssr_miss_colour`.
    FalseColour,
    /// `vec4`: `rgb` is the reflected radiance and `a` is how much of it to
    /// trust, from 0 (nothing was found) to 1.
    ///
    /// The two are kept APART so that a later pass can filter across
    /// neighbouring pixels -- which is the only place the hit/miss cliff can be
    /// removed. A blended colour cannot be filtered: a pixel that missed and a
    /// pixel that hit are both just colours by then, and averaging them smears
    /// the surface into its own reflection. Confidence is what tells the filter
    /// which neighbours are worth borrowing from.
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

/// The SSR block, reading the camera from the SHADER'S OWN scene uniform.
///
/// Exists because of `max_bind_groups`, which is 4 on this hardware. A brush
/// already spends three groups -- scene uniform, material arrays, lightmap --
/// so the cuboid form's two extra groups would put it at five and fail at
/// pipeline creation.
///
/// Nothing is lost by sharing: `SsrCamera` was uploaded with the same eye
/// view-projection and eye position the scene uniform already carries, so the
/// separate group was a second copy of numbers that were already bound.
pub fn wgsl_ssr_block_shared_camera(scene_group: u32) -> String {
    wgsl_ssr_block_inner(None, scene_group)
}

/// [`wgsl_ssr_block_shared_camera`] with the false-colour diagnostic chosen by
/// the caller, for a debug pipeline built beside the shipped one. See
/// `brush_pipeline::DebugView`.
pub fn wgsl_ssr_block_shared_camera_debug(scene_group: u32, debug: bool) -> String {
    wgsl_ssr_block_debug(None, scene_group, SsrOutput::from_debug(debug))
}

/// The SSR block whose `ssr_reflect` hands back RADIANCE AND CONFIDENCE rather
/// than a blended colour, for the reflection trace pass.
/// The declarations the COMPOSITE reads, and no march.
///
/// It reuses the march's bind group layout rather than adding a fifth group --
/// `max_bind_groups` is 4 on this hardware and a reflective brush already
/// spends all four. Binding 1 held the depth pyramid for the march; the
/// composite has no march, so the filtered reflection goes there instead. Both
/// are declared `Float { filterable: false }`, so the layout does not change
/// and neither does the pipeline's shape.
pub fn wgsl_ssr_composite_block(scene_group: u32) -> String {
    format!(
        r#"
// DECLARED HERE TOO, because the ray prep is shared with the trace and the
// inline pass and uses it, while the march that normally declares it is absent
// from this shader. `the_two_blocks_agree_on_the_reflectivity_cap` pins the two
// against each other so they cannot drift.
const MAX_SSR_REFLECTIVITY: f32 = {cap:?};

@group({scene_group}) @binding(0) var ssr_scene_color: texture_2d<f32>;
@group({scene_group}) @binding(1) var ssr_reflection: texture_2d<f32>;
@group({scene_group}) @binding(2) var ssr_scene_samp: sampler;

/// The filtered reflection at this pixel, `rgb` radiance and `a` confidence.
///
/// A CONFIDENCE-WEIGHTED BILINEAR READ, done by hand out of four
/// `textureLoad`s rather than by a sampler. The buffer is traced at
/// `REFLECTION_SCALE`, so at 2 each of its texels covers a 2x2 block of the
/// frame and point-sampling it would put visible squares along exactly the
/// silhouettes this whole path exists to smooth.
///
/// It cannot be a filtering sampler. `a` WEIGHTS `rgb`, and hardware filtering
/// interpolates the two independently -- a confident texel's colour blended
/// with an empty one's alpha invents a reflection at the boundary, which is the
/// artefact rather than the cure. Weighting each tap by its own confidence
/// before the blend is the same interpolation done in the right order: an empty
/// neighbour contributes nothing to the colour instead of darkening it, and the
/// confidence that comes out is the honest bilinear mix.
fn ssr_reflection_at(clip_xy: vec2<f32>) -> vec4<f32> {{
    let limit = vec2<i32>(textureDimensions(ssr_reflection)) - vec2<i32>(1);
    // This pixel's centre in reflection-buffer texels, minus the half texel
    // that puts the sample point at the texel's centre rather than its corner.
    let uv = clip_xy / {scale}.0 - vec2<f32>(0.5);
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
    // The bilinear weights sum to one, so `weight` IS the interpolated
    // confidence as well as the normaliser for the colour.
    return vec4<f32>(sum / weight, weight);
}}
"#,
        scale = REFLECTION_SCALE,
        cap = MAX_SSR_REFLECTIVITY_VALUE,
    )
}

/// The Rust-side value of the shader's `MAX_SSR_REFLECTIVITY`, so the composite
/// can declare it without the march block present.
const MAX_SSR_REFLECTIVITY_VALUE: f32 = 0.4;

pub fn wgsl_ssr_block_shared_camera_radiance(scene_group: u32) -> String {
    wgsl_ssr_block_debug(None, scene_group, SsrOutput::Radiance)
}

fn wgsl_ssr_block_inner(camera_group: Option<u32>, scene_group: u32) -> String {
    wgsl_ssr_block_debug(camera_group, scene_group, SsrOutput::from_debug(SSR_DEBUG))
}

/// The same, with the diagnostic forced on or off.
///
/// Tests that pin the SHIPPING shader text call this with `false`, so a
/// diagnostic session cannot make them pass by rewriting what they check.
fn wgsl_ssr_block_debug(
    camera_group: Option<u32>,
    scene_group: u32,
    output: SsrOutput,
) -> String {
    let debug = output == SsrOutput::FalseColour;
    // The resolved depth, always. See `SceneTarget::resolved_depth_view`.
    let depth_ty = "texture_2d<f32>";
    // The coarsest mip the chain actually has, so the shader never asks for a
    // level that does not exist.
    let max_mip = (SSR_MIPS - 1) as f32;
    // Either its own uniform, or the accessors the scene uniform already
    // provides. `cam_view_proj`/`cam_pos` come from the lights block, which
    // every shader using this form already includes.
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
    // The false-colour diagnostic. Empty strings in a shipped build, so the
    // shader is byte-identical to what it would be without any of this.
    // WHOLE STATEMENTS, not prefixes. Inserting a debug `return` in front of
    // the real one leaves the real one unreachable, and naga rejects that with
    // "There are instructions after `return`" -- caught by the pipeline-build
    // tests, which is exactly what they are for.
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
        // RADIANCE AND CONFIDENCE, KEPT APART. See `SsrOutput::Radiance`.
        //
        // A miss is confidence ZERO and no colour, not the fallback: the
        // fallback is a property of the SURFACE and the resolve pass that reads
        // this buffer recomputes it. Writing it here would blur one surface's
        // environment into its neighbour's.
        SsrOutput::Radiance => (
            "return vec4<f32>(0.0);",
            "return vec4<f32>(0.0);",
            // ALPHA IS `fade` ALONE, and deliberately NOT `weight * fade`.
            //
            // `weight` is the surface's reflectivity -- how much of THIS pixel
            // is reflection rather than its own colour. Folding it in here
            // would have the resolve weight a neighbour's contribution by how
            // shiny that neighbour is, when what the filter needs to know is
            // whether the RAY found anything. The reflected colour is a
            // property of the direction, not of the surface it came off, so a
            // dull surface's good sample is just as valid to borrow.
            //
            // The composite multiplies by its OWN pixel's reflectivity. Doing
            // it in both places would square it and every reflection would be
            // far too dark.
            "return vec4<f32>(hit_color, clamp(fade, 0.0, 1.0));",
            "return vec4<f32>(0.0);",
        ),
    };
    // The march is the same text in all three; only what it hands back differs.
    let ret_ty = match output {
        SsrOutput::Radiance => "vec4<f32>",
        _ => "vec3<f32>",
    };
    format!(
        r#"
{camera_decl}

@group({scene_group}) @binding(0) var ssr_scene_color: texture_2d<f32>;
@group({scene_group}) @binding(1) var ssr_scene_depth: {depth_ty};
@group({scene_group}) @binding(2) var ssr_scene_samp: sampler;

// SIZED TO THE TRUSTED RANGE, not to the room.
//
// Twenty steps reach 17.6 metres. Since the handover moved (see
// `SSR_TRUST_FAR`), everything past about 4.4 m is multiplied by a fade of
// zero -- so seven of those twenty steps were a depth load each, per reflective
// fragment, for a result that was then discarded.
//
// Thirteen reach 5.07 m, which still covers the whole trusted range with a
// margin. Measured on the headset the frame was entirely GPU bound --
// cpu_avg 2.2 ms against gpu_avg 16 to 35 ms, where 72 Hz allows 13.9 -- so
// the cost of the march is not a rounding error.
//
// The longest step drops from 2.79 m to 0.87 m as a side effect, which the six
// refinement halvings then localise to 1.4 cm instead of 4.4 cm.
const SSR_STEPS: u32 = 32u;
// THE STRIDE, IN PIXELS OF THE EYE BUFFER.
//
// The march used to walk the world, 13 steps growing 1.18x each, so the last
// steps were the better part of a metre. Anything smaller than a step -- a
// hand, a lamp, the edge of a doorway -- fell between two of them, and whether
// a given pixel's step landed on it was decided by where that pixel's step
// boundaries fell. The object was then stamped once per step instead of
// reflected once: a row of hand-shaped ghosts marching away across the wall,
// each larger than the last, and the doorway's reflection folded into an
// accordion (headset, 2026-09-17).
//
// Measured in `fade_continuity_twin`: two ADJACENT pixels read their reflection
// from points 145 pixels apart at the worst place on the floor. Walking the
// screen instead brings that to 19, because a step cannot skip anything wider
// than the stride, at any distance.
//
// The floor is four pixels. The ceiling exists because the budget is fixed: a
// long ray spreads `SSR_STEPS` samples over its own screen length, and without
// a cap a ray running nearly along the screen would put them 125 pixels apart
// and be no better than what it replaced. Capped, such a ray simply stops
// early and hands the rest to the probe, which is the right answer anyway --
// it is a long ray, and `SSR_TRUST_FAR` was going to fade it out.
//
// The stride stays CONSTANT along any one ray. That is the property that
// matters; the size is a budget.
// THE PYRAMID THE MARCH WALKS. See `docs/ssr-hi-z-scope-2026-09.md`.
//
// A fixed stride has to choose between reach and fineness, because the step
// budget is spent either way: fine enough to stop the ghosting, it reached 77%
// of the reflections the old world-space march found, and the headset painted
// whole walls "ran out of steps" -- which is the comb around the avatar's hand,
// rays that never found anything.
//
// The pyramid removes the choice. Each level holds the NEAREST depth of the
// texels below it, so a region with nothing in front of the ray is crossed in
// ONE step at a coarse level, and the ray only descends where something might
// be in the way. Measured in `fade_continuity_twin`: 3395 probes reflected
// against the stride's 1692, reading 11 px apart against its 4.
const SSR_HI_Z_LEVELS: i32 = 8;
// Starting coarser costs descents on every ray; starting at 0 is a linear march.
const SSR_HI_Z_START_LEVEL: i32 = 2;
// A cap, not a plan. A ray still unresolved after this is crossing something
// pathological and is better handed to the probe.
// BACK TO 64, HAVING TRIED 128.
//
// The theory was that the green fringe beside the avatar's hand was rays
// running out of room. Measured on the headset, doubling the cap left the
// fringe exactly as it was and took the frame from about 15 ms to 42 -- the
// reflective pass alone from 0.65 ms an eye to between 2 and 10. So the fringe
// is not rays that need more iterations, and the extra ones were spent finding
// that out. A change that makes the picture no better and the frame three times
// slower is evidence about the mechanism, not a case for more headroom.
// LOWERED TO 24, and this number is a BUDGET rather than a quality setting.
//
// The build that measured 0.65 ms an eye was fast for the wrong reason: it read
// the reflection at the depth plane even when that lay BEHIND the ray, so rays
// which had nothing left to find reported a hit immediately and stopped. That
// was the light fixture's reflection smeared into a streak. Putting the hit
// where the ray actually is removed those false early exits -- and the cost
// they had been hiding appeared: 8 to 10 ms an eye, because rays that slip
// behind an occluder genuinely have nowhere to go and keep looking.
//
// The reference keeps marching past the occluder, which is right for a renderer
// with the budget for it. At 72 Hz with 13.9 ms for everything, the honest
// trade is to bound the search and hand what it cannot resolve to the probe.
const SSR_HI_Z_MAX_ITERATIONS: u32 = 64u;
// How far past a cell boundary to step, in TEXELS. See `t_eps`.
const SSR_CELL_NUDGE_TEXELS: f32 = 0.05;
// Shorter than this on screen and there is nothing to march. See where it used.
const SSR_MIN_RAY_TEXELS: f32 = 2.0;
const SSR_STRIDE_PIXELS: f32 = 4.0;
const SSR_STRIDE_MAX: f32 = 24.0;
/// Where the search-budget fade begins, as a fraction of the budget.
const SSR_BUDGET_FADE_FROM: f32 = 0.6;
// How far the ray may reach, in metres. Deliberately the distance the old
// thirteen steps covered, so `SSR_TRUST_NEAR` and `SSR_TRUST_FAR` -- which are
// fractions of it -- still mean the metres `handover_range_tests` asserts.
const SSR_MAX_DISTANCE: f32 = 5.07;
// Nearer than this the projection is not worth trusting; the ray is clipped to
// it rather than allowed to wrap around behind the camera.
const SSR_NEAR_W: f32 = 0.06;
// How deep a slab each depth-buffer texel stands for, in METRES.
//
// McGuire and Mara's `zThickness`. A depth buffer says where a surface starts
// and nothing about where it ends, so a ray that has gone behind one cannot be
// told from a ray that has gone INTO it. Assuming every surface a quarter of a
// metre thick answers that, and being in metres it means the same thing at
// every distance and at every point in the march.
/// Strongest a screen-space reflection is allowed to be. See the brush shader.
const MAX_SSR_REFLECTIVITY: f32 = 0.4;
/// Mip levels a fully rough surface reaches for. Roughness scales onto this.
const SSR_ROUGHNESS_MIPS: f32 = 8.0;
const SSR_THICKNESS_METRES: f32 = 0.25;
// Halvings used to localise the hit once the coarse march has straddled it.
//
// The march grows its step geometrically, so by the last iteration one step is
// 2.8 METRES long and the intersection is known only to within that. The hit
// boundary therefore quantises into blocks metres wide in the world, which is
// the rectangular staircase seen along every reflected doorway edge. Six
// halvings cut 2.8m to about 4cm, which is finer than a texel at any distance
// worth reflecting.
const SSR_REFINE_STEPS: u32 = 6u;
// Where a screen-space hit stops being trusted, as a fraction of the march.
//
// CONFIDENCE FALLS OFF WITH DISTANCE, and this is what makes the probe and the
// march a pair rather than a competition. A short ray lands on something a few
// texels away that is almost certainly the right surface; a long one has
// crossed most of the frame, is quantised to a step metres long, and is as
// likely to be a near-tangent flicker as a reflection. Near a grazing surface
// those long rays alternate hit and miss from one row of pixels to the next,
// which is the striped green rectangle that appeared on the far end of every
// wall.
//
// Fading them out hands those pixels to the probe, which does know what is
// there -- including the parts off screen entirely.
// NO TOLERANCE, AND NO CONFIDENCE TERM. Both were removed on 2026-09-17 and
// this note is what stands in their place.
//
// `SSR_THICKNESS_STEPS` asked whether a sample sat within a couple of steps'
// DEPTH behind the surface. Both halves of that move with the march -- the step
// grows geometrically, and where a boundary falls relative to the surface
// slides with the pixel -- so the same geometry was accepted at one pixel and
// rejected at the next, and a fade built on the same quantity swept a step's
// worth of values and reset. Measured in `fade_continuity_twin`, a jump of a
// FULL 1.0 in the reflection's strength between two adjacent pixels. On the
// headset: thirteen ribbons across the floor's reflection of the wall
// spotlight, thirteen being the step count.
//
// It was replaced with a confidence measured in metres around the refined hit
// -- how squarely the ray met what it hit, from the depth buffer a fixed
// distance either side. That lasted one build. Reading the depth buffer AWAY
// from the hit means reading it across silhouettes, and at a silhouette the
// neighbouring sample is metres from the surface, so the measure collapses to
// zero and takes the reflection with it. The headset showed black claw marks
// strung along every reflected edge -- the ceiling lamp, its wire cage -- and
// the reflected arm shredded into vertical stripes. See `silhouette_tests`.
//
// A crossing needs neither. In front at one sample, behind at the next, means
// the ray passed through the visible surface; the refinement says where, and
// the remaining fades -- edge, grazing, distance, facing, threshold -- all
// depend on the surface being shaded or on the refined hit, never on a march
// sample and never on a depth read somewhere else.
const SSR_MIN_WEIGHT: f32 = 0.025;

// SSR IS A CONTACT EFFECT NOW. THE PROBE CARRIES THE REST.
//
// These were 0.25 and 0.75 -- SSR at full strength out to 4.4 m and still
// contributing at 13. That was set when the probe was eight-bit sRGB and could
// not stand in for the screen: a dim room's walls came back holding three
// distinct values, so handing a reflection to the probe lost it. The probe is
// now sixteen-bit and carries a thousand levels on those same walls, which
// changes which of the two should be answering.
//
// It matters because every artefact left in this renderer's reflections is a
// LONG-RAY artefact, and they are not a tuning problem. A screen-space ray that
// crosses most of the frame samples a depth buffer that has a wall in front of
// what it is trying to reflect; at a doorway it straddles the discontinuity
// between the near wall and the grass beyond, and hits and misses then
// alternate from one march step to the next. Measured on the marble floor
// reflecting the doorway, that is a bright quadrilateral sliced into six
// vertical bars. No thickness tolerance fixes it, because the information the
// ray needs is not on the screen.
//
// A SHORT ray has none of that. It lands a few texels away on a surface that is
// certainly visible and certainly the right one, which is exactly the contact
// reflection a probe cannot give -- the probe is a photograph from one point,
// so it is worst precisely where the reflected thing is close.
//
// So the split is now by what each source is good at: the screen answers within
// about a metre, fades out by four, and the probe answers everything beyond.
// FRACTIONS OF `reach`, so they have to move whenever the march is resized --
// the distances they are chosen to mean are in METRES, and `handover_range_tests`
// asserts those metres rather than these numbers.
// 0.18 * 5.07 = 0.91 m at full strength; 0.87 * 5.07 = 4.4 m and gone.
const SSR_TRUST_NEAR: f32 = 0.18;
const SSR_TRUST_FAR: f32 = 0.87;
// How far off its own surface a ray starts, in metres.
//
// Along the NORMAL, which is the only direction guaranteed to lead away from
// the surface. The bias used to be along the reflection ray, and at a grazing
// angle that vector lies almost IN the surface -- so the offset slid the sample
// along the wall instead of lifting it off, every step landed back on the same
// wall, and the wall reflected itself. That is the nested contour banding seen
// down the inside of the marble room.
const SSR_NORMAL_BIAS: f32 = 0.05;

/// Why a marched pixel found nothing, as a colour. See `SSR_DEBUG`.
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
    // ORANGE: the ray stopped BEHIND a surface.
    //
    // This used to fall through to green -- "ran out of iterations" -- so the
    // two most common ways for a ray to find nothing were painted the same
    // colour, and every green region in a screenshot was ambiguous between a
    // walk that STALLED, which is a marching fault and is fixed in the
    // traversal, and a ray with nowhere left to look, which is a FALLBACK
    // fault and can only be fixed after the trace. Those want opposite fixes.
    //
    // IT WAS WHITE FOR ONE BUILD, WHICH WAS NO BETTER. A hit is painted
    // `vec3(fade)` -- a GREYSCALE ramp -- so a confident hit at fade 1.0 is
    // pure white too, and the headset showed the artefacts sitting in "white
    // and grey areas" that could equally have been ordinary confident
    // reflections (2026-09-18). A diagnostic colour has to be off the grey
    // diagonal or it means two things again; `every_miss_colour_is_off_the_grey_ramp`
    // now refuses one that is not.
    if (reason == 4u) {{
        return vec3<f32>(1.0, 0.45, 0.0);
    }}
    return vec3<f32>(0.0, 1.0, 0.0);
}}

/// How far inside the frame a clip-space point is, in uv units.
///
/// Zero on the frame boundary, 0.5 dead centre, and zero behind the camera.
/// `min(uv.x, 1 - uv.x)` is half of `min(1 + ndc.x, 1 - ndc.x)`, so this is the
/// same quantity the edge fade has always been measured in.
fn ssr_frame_edge(c: vec4<f32>) -> f32 {{
    if (c.w <= 0.0) {{
        return 0.0;
    }}
    let ndc = c.xy / c.w;
    return 0.5 * min(min(1.0 - ndc.x, 1.0 + ndc.x), min(1.0 - ndc.y, 1.0 + ndc.y));
}}

/// The reflection of the scene in this surface, or `fallback` where there is none.
///
/// `fallback` is what the surface reflects when the screen cannot answer -- the
/// ray left the frame, or hit nothing. Previously that case returned the
/// surface's own colour, so a reflection did not fade out at the screen edge,
/// it VANISHED, and the boundary was visible as a hard line that moved with the
/// head. Handing in the environment term the shader already computed makes the
/// miss case a blurry environment reflection instead, which is what a reflection
/// probe would provide and is continuous with the hit case.
///
/// `roughness` selects the mip. A rough surface reflects a blurred world, and
/// reading a single texel however rough the surface is was why marble came back
/// looking like polished chrome.
fn ssr_reflect(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    fallback: vec3<f32>,
    weight: f32,
    roughness: f32,
) -> {ret_ty} {{
    // See `SSR_MIN_WEIGHT`. `fallback` is the probe, so this is a change of
    // SOURCE, not a loss of the reflection.
    if (weight < SSR_MIN_WEIGHT) {{
        {dbg_rough}
    }}
    // AND RAMP IN ABOVE IT, rather than switching on.
    //
    // `weight` carries the material's ROUGHNESS MAP, so it varies texel to
    // texel. Brick's roughness sits right on top of this threshold -- the
    // false-colour diagnostic showed its walls almost entirely "skipped", with
    // speckles of "marched" along every mortar line and crack. Neighbouring
    // texels were taking different code paths and returning visibly different
    // colours, which is the static that reads like glitter on the stone.
    //
    // Marble never reaches the threshold at all (Schlick bottoms out at 0.04,
    // and marble's roughness tops out at 0.106, so it asks for 0.036 at worst)
    // -- which is why checking marble alone said this could not be the cause.
    // It is the cause for brick.
    let threshold_fade = smoothstep(SSR_MIN_WEIGHT, SSR_MIN_WEIGHT * 2.0, weight);
    let n = normalize(world_normal);
    let view_dir = normalize(world_pos - {pos});
    let refl_dir = reflect(view_dir, n);
    // STAND DOWN FOR RAYS HEADING BACK TOWARD THE VIEWER.
    //
    // `dot(refl_dir, view_dir)` is -cos(2 x incidence): -1 for a surface seen
    // head-on, whose reflection is whatever is behind the viewer, and 0 at 45
    // degrees. What is behind the viewer is never on screen, so those rays can
    // only leave through the near plane or an edge. Their neighbours whose rays
    // happen to stay inside find the bright thing the surface reflects, and the
    // boundary between the two is a hard-edged dark wedge cut into the
    // reflection -- the black triangle in the wall's reflection of the
    // spotlight pool (headset, 2026-09-10/11).
    //
    // Fading by the ANGLE makes that boundary a smooth function of the surface
    // rather than of which step a march ended on, and skipping the march where
    // it is fully faded costs nothing and saves thirteen depth loads.
    let facing_fade = smoothstep(-0.7, -0.2, dot(refl_dir, view_dir));
    if (facing_fade <= 0.0) {{
        {dbg_facing}
    }}

    let scene_size = vec2<f32>(textureDimensions(ssr_scene_color));

    // A UNIFORM STRIDE IN SCREEN SPACE. See the note above `SSR_STRIDE_PIXELS`.
    //
    // The ray is projected once, at both ends, and then walked across the
    // SCREEN. Both `1/w` and the depth the buffer holds are linear in the
    // screen-space parameter, so each step is two multiply-adds and one depth
    // read, and no step is longer in pixels than the stride however far the ray
    // travels in the world.
    let march_origin = world_pos + n * SSR_NORMAL_BIAS;
    var far_end = march_origin + refl_dir * SSR_MAX_DISTANCE;
    let h0 = {vp} * vec4<f32>(march_origin, 1.0);
    var h1 = {vp} * vec4<f32>(far_end, 1.0);

    var hit = false;
    var hit_uv = vec2<f32>(0.0);
    var hit_pos = march_origin;
    // HOW MUCH OF THE SEARCH BUDGET HAD BEEN SPENT when the hit was found,
    // from 0 to 1. See `budget_fade`.
    var budget_used = 0.0;
    // Why the march stopped. See `ssr_miss_colour` for what each one means.
    var exit_reason: u32 = 0u;

    if (h0.w > SSR_NEAR_W) {{
        // Pull the far end back to the near plane rather than let it wrap
        // around behind the camera, where the projection means nothing.
        if (h1.w <= SSR_NEAR_W) {{
            far_end = mix(march_origin, far_end, (h0.w - SSR_NEAR_W) / (h0.w - h1.w));
            h1 = {vp} * vec4<f32>(far_end, 1.0);
        }}
        let k0 = 1.0 / h0.w;
        let k1 = 1.0 / h1.w;
        let s0 = vec2<f32>(h0.x * k0 * 0.5 + 0.5, 0.5 - h0.y * k0 * 0.5) * scene_size;
        let s1 = vec2<f32>(h1.x * k1 * 0.5 + 0.5, 0.5 - h1.y * k1 * 0.5) * scene_size;

        // WHAT THE DEPTH BUFFER HOLDS IS AFFINE IN 1/w: z = A + B/w, for any
        // standard perspective matrix. Solving A and B from the ray's own two
        // ends turns every depth read into a distance from the eye in METRES,
        // for one divide and no extra uniform -- which is what lets the
        // thickness below be a constant of the SCENE rather than of the march.
        let dk = k1 - k0;
        let bz = select(0.0, (h1.z * k1 - h0.z * k0) / dk, abs(dk) > 1e-9);
        if (abs(bz) <= 1e-12) {{
            exit_reason = 3u;
        }}
        if (abs(bz) > 1e-12) {{
            let az = h0.z * k0 - bz * k0;

            // HIERARCHICAL-Z TRAVERSAL. See `SSR_HI_Z_LEVELS`.
            //
            // The ray is always inside one cell of one level. That cell holds
            // the NEAREST depth anything in it reaches, so if the ray is still
            // in front of it on the way out, nothing in the cell can have been
            // hit: the whole cell is skipped in one step and the walk goes UP a
            // level to try skipping a bigger one. If the ray would instead
            // reach that depth inside the cell, something might be there, so it
            // advances to the depth plane and goes DOWN to look closer.
            // Reaching level 0 that way is the hit.
            let ds = s1 - s0;
            let dz = h1.z * k1 - h0.z * k0;
            // A RAY HAS TO COVER SOME SCREEN BEFORE IT IS WORTH WALKING.
            //
            // `1e-4` let through rays a hundredth of a texel long, and the
            // nudge below divides by that length -- so the first step jumped
            // clean past the far end and the walk ended immediately. Around the
            // point where a reflection ray turns to face the viewer its screen
            // length passes through zero, so those pixels formed a starburst of
            // failed rays radiating from it, sliding across the floor with the
            // head (2026-09-17).
            if (length(ds) <= SSR_MIN_RAY_TEXELS) {{
                exit_reason = 2u;
            }}
            if (length(ds) > SSR_MIN_RAY_TEXELS) {{
                // A QUARTER OF A TEXEL PAST THE BOUNDARY, in screen space,
                // however long the ray is.
                //
                // This was a constant in `t`, which is a quarter texel on a ray
                // that crosses the whole frame and two thousandths of one on a
                // ray two thousand pixels long. A nudge that fails to leave the
                // cell means the next iteration finds the SAME cell, computes
                // the same exit, and the walk stalls until the iteration cap
                // gives up -- which reads as a miss. On the headset that drew
                // thin vertical green spikes hanging into the reflection on a
                // curving wall, one per column of stalled rays (2026-09-17).
                // Capped as well as scaled: a quarter texel is the RIGHT
                // nudge, but on a very short ray a quarter texel is most of the
                // ray, and stepping that far each time skips the search.
                var level = SSR_HI_Z_START_LEVEL;
                // One texel in, so the walk does not begin by intersecting the
                // cell it started in.
                var t = min(1.0 / length(ds), 1.0);
                for (var i = 0u; i < SSR_HI_Z_MAX_ITERATIONS; i = i + 1u) {{
                    let p = s0 + ds * t;
                    if (t > 1.0 || p.x < 0.0 || p.y < 0.0 || p.x >= scene_size.x || p.y >= scene_size.y) {{
                        exit_reason = 1u;
                        break;
                    }}
                    let cell_size = f32(1 << u32(level));
                    let cell = floor(p / cell_size);
                    // Where the ray leaves this cell, on each axis.
                    let next_x = select(cell.x, cell.x + 1.0, ds.x > 0.0) * cell_size;
                    let next_y = select(cell.y, cell.y + 1.0, ds.y > 0.0) * cell_size;
                    let tx = select(1e30, (next_x - s0.x) / ds.x, abs(ds.x) > 1e-6);
                    let ty = select(1e30, (next_y - s0.y) / ds.y, abs(ds.y) > 1e-6);
                    // A hair past the boundary, or the next step lands on the
                    // same cell and the walk stalls.
                    // THE NUDGE IS SIZED BY THE AXIS BEING CROSSED, not by the
                    // ray's total length.
                    //
                    // A quarter texel along the ray is a quarter texel of X
                    // only if the ray runs in X. For a ray travelling mostly
                    // down the screen, crossing a vertical boundary needs
                    // `0.25 / |ds.x|` of the ray parameter, which can be
                    // hundreds of times larger -- so a nudge scaled by the
                    // whole length leaves the ray stuck in its cell and it
                    // burns every iteration there. Grenier's notes on Hi-Z
                    // tracing call this out and offset on the crossed axis
                    // alone: it is what stops rays sticking when they run along
                    // a screen axis, and it is the starburst spreading out from
                    // wherever the reflection turns.
                    let t_exit = select(
                        ty + SSR_CELL_NUDGE_TEXELS / max(abs(ds.y), 1e-6),
                        tx + SSR_CELL_NUDGE_TEXELS / max(abs(ds.x), 1e-6),
                        tx < ty,
                    );

                    // CLAMPED TO THE LEVEL'S OWN SIZE.
                    //
                    // The cell index is worked out from a full-resolution pixel
                    // position, and a level is floor(size / 2^level) texels --
                    // so at 1680 wide, level 5 is 52 texels and the index can
                    // reach 52. An out-of-range `textureLoad` returns ZERO,
                    // which reads as a surface infinitely near: the ray
                    // descends, finds nothing at level 0, climbs, lands on the
                    // same out-of-range cell and descends again until its
                    // iterations are gone. That is the green swathe and the
                    // reflective pass at 8-11 ms an eye.
                    //
                    // The CPU twin clamps -- which is exactly why it reported
                    // 0.7% of rays stalling while the headset was full of them.
                    let level_max = vec2<i32>(textureDimensions(ssr_scene_depth, level)) - vec2<i32>(1);
                    let cell_min = textureLoad(
                        ssr_scene_depth,
                        clamp(vec2<i32>(cell), vec2<i32>(0), level_max),
                        level,
                    ).x;
                    // The ray's depth is linear in t, so the t at which it
                    // reaches the cell's nearest surface is one division.
                    let t_depth = select(1e30, (cell_min - h0.z * k0) / dz, abs(dz) > 1e-12);

                    // THE CANONICAL STEP (Uludag, GPU Pro 5): advance to the
                    // depth plane FIRST, then ask whether that left the cell.
                    //
                    // Both earlier shapes got this wrong from opposite ends.
                    // Requiring the plane to lie AHEAD of the ray skipped every
                    // cell it had already passed the depth of -- 63 reflections
                    // found against 3395 in `fade_continuity_twin`. Dropping
                    // the requirement made the ray descend at every cell once
                    // it was behind anything, crawl a texel at a time at level
                    // 0 and climb again, three iterations per texel: a
                    // spreading GREEN swathe beside the avatar's hand on the
                    // headset, and the reflective pass at 8-9 ms an eye instead
                    // of 0.65 (2026-09-17).
                    //
                    // Advancing first settles both. Stay inside the cell and
                    // something in it is worth a closer look, so descend. Leave
                    // it and the ray passed through without meeting anything,
                    // so move to the boundary and climb. A ray already behind a
                    // surface does not advance at all, stays put, and descends
                    // to level 0 where the slab decides -- which is how it gets
                    // past an occluder instead of grinding along behind it.
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
                        // BEHIND IT, AND THAT IS THE END OF THIS RAY.
                        //
                        // The reference climbs and carries on, which lets a ray
                        // continue past an occluder -- correct, and affordable
                        // when the frame has room. Measured here it does not:
                        // rays that slip behind the avatar's hand have nowhere
                        // left to find anything, and they spend their whole
                        // budget discovering that. 8 to 10 ms an eye, and
                        // cutting the budget from 64 iterations to 24 barely
                        // moved it while making the picture worse, because the
                        // rays were not finishing either way.
                        //
                        // Screen space holds no record of what is behind an
                        // occluder, so stopping gives up nothing that was
                        // recoverable. The pixel goes to the probe, which knows
                        // what is around it. The hard edge that leaves is a
                        // FALLBACK problem, not a marching one, and the
                        // literature is unanimous that it belongs there --
                        // weight the handover by confidence instead of cutting
                        // it. See the note in `docs/ssr-hi-z-scope-2026-09.md`.
                        //
                        // WHITE in the false-colour view, not green: see
                        // `ssr_miss_colour`.
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
    // The mip a surface this rough should be reading. Sampled rather than
    // loaded, so the level is filtered and the blur is smooth rather than
    // stepping between levels.
    let mip = clamp(roughness * SSR_ROUGHNESS_MIPS, 0.0, {max_mip:.1});
    let hit_color = textureSampleLevel(ssr_scene_color, ssr_scene_samp, hit_uv, mip).rgb;

    // HOW CLOSE THE RAY CAME TO LEAVING THE FRAME, EXACTLY, from the two ends
    // of the segment it actually travelled.
    //
    // Clip coordinates are linear along a straight segment, so each frustum
    // plane's distance and `w` are both linear in the segment parameter and
    // their ratio is a linear-fractional function -- monotone wherever `w`
    // keeps its sign. A monotone function takes its minimum at an end. So the
    // closest the ray came to the frame edge ANYWHERE along its path is the
    // smaller of its two ends, exactly, and no sampling is involved. (The same
    // statement says a ray can leave the frame only once: no straight segment
    // exits and re-enters.) `the_edge_distance_is_monotone_along_a_segment`
    // checks it against 257 samples along a few hundred random segments.
    //
    // This was a running minimum over the march's samples, which is where the
    // banding came from: the samples move with the step phase, so the fade
    // swept a step's worth of values and reset at every step boundary. Taking
    // the same minimum exactly costs two projections and cannot band.
    let edge = min(
        ssr_frame_edge({vp} * vec4<f32>(march_origin, 1.0)),
        ssr_frame_edge({vp} * vec4<f32>(hit_pos, 1.0)),
    );
    // WIDER THAN IT WAS (0.12).
    //
    // This is the frontier the false-colour diagnostic painted red: 30.8% of
    // every marched pixel had its ray leave the frame. Because a frustum plane
    // cuts a flat surface in a STRAIGHT LINE, that region is a polygon -- a
    // wedge or triangle across a wall, which is exactly what it looks like,
    // and it grows until it swallows the whole reflection as the reflected
    // thing slides off screen.
    //
    // It cannot be removed: the data genuinely is not there. What it can be is
    // gradual, and a quarter of the frame is enough distance to hide the
    // handover in. Screen-space reflection is least trustworthy near the edge
    // anyway, so widening this costs the least reliable samples first.
    let edge_fade = smoothstep(0.0, 0.25, edge);
    // FADE ON HOW HEAD-ON THE VIEW IS.
    //
    // This measured `abs(dot(refl_dir, -view_dir))`, which is 1.0 head-on AND
    // 1.0 at grazing -- the reflection vector swings back toward the viewer as
    // the surface turns edge-on, and the `abs` folded the two ends together.
    // Measured across the whole range it never dropped below 1.000, so the
    // "grazing fade" faded nothing, ever.
    //
    // That mattered because grazing is exactly where screen-space marching is
    // worst: the ray travels almost parallel to the surface, so consecutive
    // steps land far apart in the world but adjacent on screen, and the hits
    // quantise into regular stripes. It is also where Fresnel makes the
    // reflection strongest, so the artefact was at full volume precisely where
    // it was ugliest -- the corduroy ribbing down a marble wall seen edge-on.
    //
    // `dot(n, -view_dir)` is cos(view angle): 1 head-on, 0 edge-on. That is the
    // quantity the fade was always meant to be.
    let grazing_fade = smoothstep(0.15, 0.45, dot(n, -view_dir));
    // The total the march can cover, so "how far did this ray go" is a
    // fraction rather than a distance that changes meaning with the step size.
    let reach = SSR_MAX_DISTANCE;
    // HOW FAR THE RAY ACTUALLY WENT, measured to the REFINED hit.
    //
    // This was the running total of the coarse steps, so it could only ever
    // hold one of twenty values -- and because the step grows geometrically,
    // the gap between two of them is metres by the middle of the march. Two
    // neighbouring pixels whose rays crossed the same wall one step apart got
    // distances differing by a large fraction of `reach`, and therefore
    // visibly different fades.
    //
    // The reflected IMAGE was already smooth: binary refinement had localised
    // `hit_uv` to a few centimetres. What was quantised was the brightness it
    // was multiplied by. That is why the artefact looks like the same shape
    // repeated across a surface at even spacing rather than like a broken
    // reflection -- measured on the marble floor reflecting the doorway, a
    // regular ripple of period 44 pixels, strongest exactly where this
    // smoothstep is steepest and fading out where it flattens.
    //
    // The refinement already knows where the hit is. Measuring back to the
    // start of the march costs one subtract and one length, and makes the
    // distance continuous across neighbouring pixels.
    let travelled = length(hit_pos - march_origin);
    let distance_fade =
        1.0 - smoothstep(SSR_TRUST_NEAR, SSR_TRUST_FAR, travelled / max(reach, 1e-4));
    // FADE AS THE SEARCH RUNS OUT, rather than stopping dead when it does.
    //
    // A screen-space march covers a fixed number of PIXELS, so a ray stretched
    // along the screen can spend its whole budget having gone barely anywhere
    // in the world. Where it does, the pixel is a miss sitting beside one whose
    // ray found its surface a step sooner and got it at full strength -- and
    // because the step count is an integer, that boundary is a STAIRCASE. On
    // the headset it was a serrated edge cut across the floor's reflection,
    // with the whole wall beyond it painted "ran out of steps" (2026-09-17).
    //
    // The distance fade cannot cover this: it is measured in metres and this
    // limit is in pixels. Measured in `fade_continuity_twin`, fading over the
    // last part of the budget takes the worst distance between what two
    // adjacent pixels read from 26 px to 15.
    //
    // It does not create the reflections the budget could not reach -- only
    // hierarchical-Z traversal does that, see
    // `docs/ssr-hi-z-scope-2026-09.md`. It makes the place where they stop a
    // gradient instead of a sawtooth.
    let budget_fade = 1.0 - smoothstep(SSR_BUDGET_FADE_FROM, 1.0, budget_used);
    let fade = edge_fade * grazing_fade * distance_fade * facing_fade * threshold_fade * budget_fade;

    // BLEND THE REFLECTION IN. Do not replace the surface with it.
    //
    // This read `mix(fallback, hit_color * weight, fade)`, which at a confident
    // hit (fade = 1) returns `hit_color * weight` and DISCARDS the surface
    // entirely. Marble's reflectivity near head-on is about 0.04, so a wall
    // that found a hit stopped being marble and became the grass outside at 4%
    // brightness -- dark green, which is exactly what it looked like.
    //
    // It also made the hit/miss boundary a cliff between two unrelated colours
    // instead of a 4% difference, so the ray marcher's steps showed up as a
    // hard staircase across the wall, and the two eyes -- which terminate their
    // marches at different steps -- disagreed visibly.
    //
    // `weight` is a REFLECTANCE: the fraction of what you see that is reflected
    // rather than the surface's own colour. So it belongs in the mix factor,
    // where it takes `weight` of the surface away and puts `weight` of the
    // reflection in its place. `fade` then scales how much of that swap we
    // trust, which is what it was always for.
    {dbg_hit}
}}
"#
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

#[cfg(test)]
mod mip_and_fallback_tests {
    use super::*;
    use crate::renderer::terrain_pipeline::tests::headless_gpu;

    #[test]
    fn the_scene_colour_really_has_a_chain_to_blur_with() {
        // Binding one mip level to a sampler with `mipmap_filter: Linear` makes
        // the filter a silent no-op: every line of the blur is present and
        // nothing is blurred. So the count is worth asserting rather than
        // assuming.
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
        // A 4x4 target has three levels, not five. Asking for a mip that does
        // not exist is undefined rather than clamped.
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
        // Both misses -- ray left the frame, ray hit nothing -- must return the
        // environment, not the surface's own colour. Returning the surface
        // meant the reflection stopped at the screen edge along a hard line
        // that moved with the head.
        // THREE since the facing fade: a ray heading back toward the viewer is
        // not marched at all, and must hand over to the environment exactly as
        // a miss does. See `facing_fade_tests`.
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
    //! What a screen-space hit is allowed to do to the surface under it.
    //!
    //! The shader is a WGSL string and cannot be called, so the property lives
    //! here as a re-implementation and the source pin below checks the shader
    //! still spells it that way. That pairing is what makes it a test rather
    //! than a comment: change the property and this fails, change the shader
    //! and the pin fails.
    use super::*;

    /// The shader's final line, in Rust.
    fn blend(fallback: f32, hit: f32, weight: f32, fade: f32) -> f32 {
        let t = weight.clamp(0.0, 1.0) * fade;
        fallback * (1.0 - t) + hit * t
    }

    /// THE regression. Marble reflects about 4% head-on; a wall that found a
    /// reflection must still be 96% wall.
    ///
    /// The shipped form was `mix(fallback, hit * weight, fade)`, which at a
    /// confident hit returned `hit * weight` and threw the surface away. A
    /// white marble wall became the grass outside at 4% brightness -- dark
    /// green -- which is what it looked like on the headset.
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

    /// And the hit/miss boundary must be a 4% step, not a cliff.
    ///
    /// This is what made the ray marcher's steps visible as a hard staircase
    /// across the wall, and what made the two eyes -- which stop their marches
    /// at different steps -- disagree so plainly.
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

    /// A mirror is still allowed to be a mirror.
    #[test]
    fn a_fully_reflective_surface_still_shows_the_reflection() {
        assert!((blend(0.8, 0.3, 1.0, 1.0) - 0.3).abs() < 1e-6);
    }

    /// Fade still governs how much of the swap happens.
    #[test]
    fn no_confidence_means_no_reflection() {
        assert!((blend(0.8, 0.3, 1.0, 0.0) - 0.8).abs() < 1e-6);
    }

    /// The shader spells it the way the property above assumes.
    #[test]
    fn the_shader_blends_by_weight_rather_than_scaling_the_hit() {
        // COMMENTS STRIPPED FIRST. The comment above the fixed line quotes the
        // broken form to explain it, and a naive `contains` on the whole source
        // matched that quote and failed on correct code -- the same way a
        // search for a class name once matched the sentence describing it.
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
    //! Screen-space reflections must fade out as a surface turns edge-on.
    //!
    //! The shipped fade used `abs(dot(refl_dir, -view_dir))`, which returns 1.0
    //! at BOTH ends -- head-on and grazing -- so it never faded anything. The
    //! re-implementation here is what the shader now computes; the source pin
    //! keeps the two in step.
    use super::*;

    /// `smoothstep(0.15, 0.45, cos_v)`, where `cos_v` is `dot(n, -view_dir)`.
    fn fade(view_deg: f32) -> f32 {
        let cos_v = view_deg.to_radians().cos();
        let t = ((cos_v - 0.15) / (0.45 - 0.15)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// The old form, for the comparison the fix is justified by.
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

    /// THE regression.
    #[test]
    fn a_surface_seen_edge_on_gets_no_screen_space_reflection() {
        assert!(fade(85.0) < 0.01, "at 85 degrees the fade is {}", fade(85.0));
        assert!(fade(89.0) < 0.01);
    }

    /// And a surface seen head-on keeps its reflection in full.
    #[test]
    fn a_surface_seen_head_on_keeps_its_reflection() {
        assert!(fade(0.0) > 0.99);
        assert!(fade(30.0) > 0.99);
    }

    /// Monotonic, so there is no angle where turning further away brings the
    /// reflection back -- which is the exact shape of the bug being fixed.
    #[test]
    fn the_fade_only_ever_decreases_as_the_view_turns_away() {
        let mut last = f32::INFINITY;
        for deg in 0..90 {
            let f = fade(deg as f32);
            assert!(f <= last + 1e-6, "fade rose from {last} to {f} at {deg} degrees");
            last = f;
        }
    }

    /// What the old form actually did, kept as the evidence for the change:
    /// full strength at every angle, including dead edge-on.
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
    //! HOW FINELY THE MARCH LOCATES A HIT, and what pays for it.
    //!
    //! This measured a binary refinement: the world-space march found a hit
    //! within a step metres long and six halvings cut that to four centimetres.
    //! The Hi-Z traversal has no refinement to measure, because it does not
    //! bracket anything -- it descends the pyramid until the cell IS one texel,
    //! and a hit at level 0 is already located to a texel of the eye buffer.
    //! What has to be checked instead is that it really does bottom out there,
    //! and that the descent is bounded.
    use super::*;

    // The shader constants, mirrored, and pinned against it below.
    const HI_Z_LEVELS: i32 = 8;
    const HI_Z_START_LEVEL: i32 = 2;
    const HI_Z_MAX_ITERATIONS: u32 = 64;

    /// The coarsest cell is big enough to be worth skipping in one step.
    #[test]
    fn the_top_of_the_pyramid_skips_something_worth_skipping() {
        let cell = 1i32 << (HI_Z_LEVELS - 1);
        assert!(
            cell >= 64,
            "the coarsest cell is {cell} px, which is close enough to a texel \
             that the pyramid is not earning the passes that build it",
        );
    }

    /// And the descent is bounded: from the top, one level at a time, a ray
    /// cannot need more iterations than the cap allows.
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
        // The mirrored constants above must match the shader's.
        assert!(code.contains(&format!("const SSR_HI_Z_LEVELS: i32 = {HI_Z_LEVELS};")));
        assert!(code.contains(&format!("const SSR_HI_Z_START_LEVEL: i32 = {HI_Z_START_LEVEL};")));
        assert!(code.contains(&format!(
            "const SSR_HI_Z_MAX_ITERATIONS: u32 = {HI_Z_MAX_ITERATIONS}u;"
        )));
    }

}

#[cfg(test)]
mod self_reflection_tests {
    //! Why a wall stopped reflecting itself.
    //!
    //! Two independent guards, both missing before: the ray must START off its
    //! own surface, and a hit must be a CROSSING rather than "the ray is behind
    //! something". Either one alone leaves the grazing case broken, so both are
    //! pinned.
    use super::*;

    /// How far the old bias actually lifted a ray off its surface, at a given
    /// view angle. The bias was `refl_dir * step * 0.5`, and what matters is
    /// its component along the normal.
    fn old_lift(view_deg: f32) -> f32 {
        let a = view_deg.to_radians();
        // Surface normal +Z; view arriving at `view_deg` off it.
        let v = [a.sin(), 0.0, -a.cos()];
        let n = [0.0f32, 0.0, 1.0];
        let d = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
        let refl = [v[0] - 2.0 * d * n[0], v[1] - 2.0 * d * n[1], v[2] - 2.0 * d * n[2]];
        // Component of the half-step along the normal.
        (refl[0] * n[0] + refl[1] * n[1] + refl[2] * n[2]) * 0.12 * 0.5
    }

    /// THE reason a grazing wall reflected itself: the old bias lifted the ray
    /// almost nowhere exactly when it mattered.
    #[test]
    fn the_old_bias_barely_left_the_surface_at_grazing() {
        assert!(old_lift(0.0) > 0.05, "head-on it lifted {}", old_lift(0.0));
        assert!(
            old_lift(85.0) < 0.006,
            "at 85 degrees it lifted {}m, which would have been enough",
            old_lift(85.0),
        );
    }

    /// The normal bias does not care about the angle, which is the whole point.
    #[test]
    fn the_normal_bias_is_the_same_at_every_angle() {
        // It is applied along n, so its component along n is itself.
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

    /// How thick a surface is assumed to be, in metres. Mirrors the shader.
    const THICKNESS: f32 = 0.25;

    /// Whether the step from `w_prev` to `w` counts as hitting the surface the
    /// depth buffer puts at `scene_w`. All three are distances from the eye in
    /// METRES; nothing here is measured in march steps.
    ///
    /// McGuire and Mara's test: the ray covers a depth INTERVAL over the step
    /// it just took, the surface stands for a slab `THICKNESS` deep behind what
    /// the buffer shows, and a hit is those two overlapping.
    fn is_hit(w_prev: f32, w: f32, scene_w: f32) -> bool {
        w_prev.max(w) >= scene_w && w_prev.min(w) <= scene_w + THICKNESS
    }

    #[test]
    fn a_ray_that_crosses_a_surface_registers() {
        // In front at 2.0 m, behind at 2.2 m, surface at 2.1 m.
        assert!(is_hit(2.0, 2.2, 2.1), "a genuine crossing was rejected");
    }

    /// AND A SURFACE THE STEP JUMPED CLEAN OVER. The interval is what makes
    /// this possible: the ray was in front at one end and behind at the other,
    /// so it passed through, however far the step carried it.
    ///
    /// This is the hand. A step a metre long in the world passes right over
    /// something 20 cm across, and a test that only looks AT the samples finds
    /// nothing there; the object is then picked up by whichever later step
    /// happens to land on it, and gets stamped once per step across the wall.
    #[test]
    fn a_surface_the_step_jumped_over_still_registers() {
        assert!(
            is_hit(2.0, 3.0, 2.4),
            "a surface in the middle of the step was missed; small things will \
             be stamped once per step instead of reflected once",
        );
    }

    /// The self-reflection case: a ray skimming along a wall is behind it the
    /// whole way. The plain `scene_z < sample_z` test counted every one of
    /// those steps as a hit, so the wall reflected itself; requiring the
    /// previous sample to have been in front means only the moment it went
    /// behind can count, and a skimming ray never has one.
    #[test]
    fn a_ray_already_far_behind_the_surface_does_not_register() {
        // Both ends a metre past a surface only a quarter of a metre thick.
        assert!(
            !is_hit(3.0, 3.2, 2.0),
            "a ray long past the surface still reports a hit, so a ray skimming \
             a wall reflects the wall it is skimming",
        );
        assert!(2.0 < 3.2, "sanity: `behind the surface` alone is satisfied here");
    }

    /// THE DOORWAY. Where the ray straddles a depth discontinuity -- the near
    /// wall and the ground metres beyond it -- one step lands far past the
    /// surface it crossed. The old tolerance called anything more than a
    /// couple of steps' depth behind "geometry the ray passed" and threw the
    /// hit away, which sliced the doorway's reflection on the floor into bars
    /// and cut a sawtooth along its edge.
    ///
    /// It is still a crossing: in front, then behind. How far past the step
    /// happened to land says nothing about that, and the refinement finds
    /// where it really was.
    #[test]
    fn a_crossing_that_overshot_is_still_a_crossing() {
        assert!(
            is_hit(2.0, 5.0, 2.05),
            "a crossing that overshot by metres was rejected; the reflection \
             will break into bars wherever it meets a depth discontinuity",
        );
    }

    /// The slab is what separates "went into it" from "went behind it", and it
    /// is the only tolerance in the march.
    #[test]
    fn the_slab_is_a_quarter_of_a_metre_wherever_the_ray_is() {
        // Just inside the slab, at 2 m and again at 40 m from the eye: the
        // answer must not depend on how far away the surface is.
        assert!(is_hit(2.2, 2.2, 2.0));
        assert!(is_hit(40.2, 40.2, 40.0));
        // And just outside it, at both distances.
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
        // THE SLAB SURVIVED THE MOVE TO HI-Z. The traversal decides WHERE to
        // look; this still decides whether what it found was entered or passed
        // behind, and it is still measured in metres.
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
    //! How much a screen-space hit is trusted, by how far the ray travelled.
    //!
    //! This is the join between the two techniques. A short ray is reliable and
    //! wins; a long one is quantised to a step metres long and flickers between
    //! hit and miss from row to row, which is what striped the far end of every
    //! wall. Fading those out hands them to the probe.
    use super::*;

    // Mirrored from the shader and PINNED below. Without the pin, changing the
    // shader's constants leaves every numeric test here green, because they all
    // run against these copies -- verified by moving SSR_TRUST_NEAR and
    // watching nothing fail.
    const NEAR: f32 = 0.18;
    const FAR: f32 = 0.87;

    /// The total distance the coarse march can cover.
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

    /// A reflection found right next to the surface is fully trusted.
    ///
    /// SUPERSEDED TUNING: this used to also assert that a ray a FIFTH of the
    /// way out (3.5 m) was undoubted. That was correct while the probe was
    /// eight-bit and could not stand in for the screen. It is wrong now: at
    /// that range the march is what produces the sliced and ghosted
    /// reflections, and the probe answers better.
    #[test]
    fn a_short_ray_keeps_its_reflection() {
        assert!(trust(0.0) > 0.99);
        // A contact reflection -- something within arm's reach of the surface.
        assert!(
            trust(0.5 / reach()) > 0.9,
            "a reflection half a metre from its surface was already doubted; that \
             is the one case the screen answers better than the probe",
        );
    }

    /// THE regression: a ray that crossed most of the frame is handed over.
    #[test]
    fn a_long_ray_hands_over_to_the_probe() {
        // IN METRES, because that is what the handover was chosen in. A
        // fraction of `reach` means something different every time the march is
        // resized, and it has been resized once already under this test.
        assert!(trust(1.0) < 0.01);
        assert!(
            trust(4.5 / reach()) < 0.01,
            "a ray four and a half metres long still contributes {}; that is the \
             range where hits and misses alternate across a depth discontinuity",
            trust(4.5 / reach()),
        );
    }

    /// Monotonic and smooth, so the handover cannot itself draw an edge --
    /// which would trade a striped rectangle for a hard ring.
    ///
    /// Sampled finely enough that the bound is about CONTINUITY rather than
    /// about how wide the fade band happens to be. The previous version
    /// sampled every one percent with a fixed tolerance, so narrowing the band
    /// failed it even though the curve was just as smooth.
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

    /// WHERE THE OLD "fade the late steps" TEST WENT.
    ///
    /// It asserted that fading began no earlier than coarse step 12, to keep
    /// the early march at full strength. That intent is reversed -- SSR is a
    /// contact effect now -- so the test had to go, and it is worth saying why
    /// it was not simply re-pointed at a later step.
    ///
    /// The replacement wanted to be "no step long enough to band may
    /// contribute", which needs a step length at which banding starts. There
    /// is no such length. Banding is not caused by a step being long in
    /// metres; it is caused by a ray crossing a DEPTH DISCONTINUITY, which is
    /// a property of the scene, not of the march. And the march's step length
    /// as a fraction of the distance it has covered is nearly constant --
    /// (growth-1)/growth, about 0.15 -- at every step, so no threshold
    /// expressed that way distinguishes an early step from a late one either.
    ///
    /// Any number put here would have been chosen to make the constants pass.
    /// What is actually being asserted lives in `handover_range_tests`, in
    /// metres, where it can be read against the size of a room.
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
        // Monotonically DECREASING, so relative coarseness cannot single out
        // the late steps -- it points at the early ones.
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
        // Every fade multiplied together, and the distance one among them.
        // Pinned as a prefix rather than the whole line so adding another fade
        // does not fail this -- what it guards is that `distance_fade` is
        // APPLIED, which it silently was not in an early version.
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
    //! Why reflections used to break at the edge of vision.
    //!
    //! A ray that walks out of the frame is a miss and falls back to the probe.
    //! The pixel beside it, whose ray stayed inside by one step, is a hit at
    //! full strength. Those two are a step apart even though the surface
    //! between them did not change -- and because the frame edge is fixed to
    //! the head, the step slides across the wall as you look around.
    use super::*;

    /// The fade, given how close the ray came to the frame edge.
    fn edge_fade(closest: f32) -> f32 {
        let t = (closest / 0.12).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// A ray that skims the frame edge must already be nearly faded out, so
    /// that the neighbouring ray which exits entirely is not a step away.
    #[test]
    fn a_ray_that_nearly_left_the_frame_is_nearly_faded() {
        assert!(
            edge_fade(0.005) < 0.02,
            "a ray half a percent from the edge still contributes {}",
            edge_fade(0.005),
        );
    }

    /// And a ray that stayed well inside keeps its reflection.
    #[test]
    fn a_ray_that_stayed_inside_keeps_its_reflection() {
        assert!(edge_fade(0.3) > 0.99);
        assert!(edge_fade(0.12) > 0.99);
    }

    /// The fade must be gradual across the band, or it trades a hard edge at
    /// the frame boundary for a hard ring just inside it.
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
        // The band is WIDER than it was. Pinned as a number so narrowing it
        // back is a deliberate act: this is the fade that turns the red wedge
        // the diagnostic showed -- 30.8% of marched pixels, bounded by
        // straight frustum-plane cuts -- from a hard triangle into a gradient.
        assert!(
            code.contains(&format!("smoothstep(0.0, {EDGE_FADE_BAND}, edge)")),
            "the screen-edge fade band moved; it is what hides the wedge where \
             rays leave the frame",
        );
        assert!(
            EDGE_FADE_BAND >= 0.2,
            "a band of {EDGE_FADE_BAND} is too narrow to hide the handover",
        );
        // A ray that leaves the frame stops, and says so. It needs no special
        // handling beyond that now: the march steps a bounded number of PIXELS,
        // so the sample that left is at most `SSR_STRIDE_MAX` past the
        // boundary, and the fade above has already taken anything that close to
        // the edge to nothing. The world-space march needed a clipping test
        // here precisely because one of its steps could cross the whole frame.
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

    /// The band the shader uses, mirrored for the assertions above.
    const EDGE_FADE_BAND: f32 = 0.25;
}

#[cfg(test)]
mod march_budget_tests {
    //! Which surfaces are worth asking the screen about.
    //!
    //! The march is 20 coarse steps plus 6 refinements plus a mip sample. On a
    //! fill-bound tile GPU that is only worth spending where the answer can
    //! actually be seen -- and below a certain reflectance it cannot, because
    //! the result is mixed in at that weight.
    //!
    //! Skipping it is not a loss: the probe still answers, from a world-space
    //! cubemap that knows what is around the surface.
    use super::*;

    const MIN: f32 = 0.025;

    /// Reflectance of a material at a given view angle, as the brush shader
    /// computes it.
    fn weight(roughness: f32, view_deg: f32) -> f32 {
        let cv = view_deg.to_radians().cos();
        let f = 0.04 + 0.96 * (1.0 - cv).powi(5);
        ((1.0 - roughness) * f).min(0.4)
    }

    /// THE case this is for: the stone room is all brick, and brick seen
    /// head-on cannot move a pixel by more than a few levels.
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

    /// Polished marble must keep it. If this ever fails the threshold has been
    /// raised past the material the feature exists for.
    #[test]
    fn polished_marble_still_marches() {
        for deg in [0.0f32, 45.0, 70.0] {
            let w = weight(0.048, deg);
            assert!(w > MIN, "marble at {deg} degrees would skip the march ({w})");
        }
    }

    /// And rough stone at a grazing angle earns it back, where Fresnel makes
    /// the reflection strong enough to see.
    #[test]
    fn rough_stone_at_a_grazing_angle_marches_again() {
        assert!(weight(0.596, 70.0) > MIN);
    }

    /// What the threshold costs, stated: at most six levels of 255, and only
    /// as a difference between the screen's answer and the probe's.
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

/// Why the distance a ray travelled is measured to the REFINED hit.
///
/// The artefact these cover looked like a broken reflection and was not one:
/// the reflected image was smooth, and the BRIGHTNESS it was multiplied by
/// came in twenty discrete levels. Measured on a marble floor reflecting a
/// doorway, that read as the same crescent repeated across the floor at a
/// regular 44-pixel spacing -- strongest where the trust smoothstep is
/// steepest, gone where it flattens out at either end.
#[cfg(test)]
mod travel_continuity_tests {
    use super::*;

    // Mirrors of the shader constants, pinned against the shader text below --
    // the same convention the other modules in this file use.
    const SSR_STEPS_COUNT: u32 = 32;
    const SSR_REFINE_COUNT: u32 = 6;
    const SSR_MAX_DISTANCE: f32 = 5.07;
    const SSR_STRIDE_MAX: f32 = 24.0;
    const SSR_TRUST_NEAR: f32 = 0.18;
    const SSR_TRUST_FAR: f32 = 0.87;

    // THE MARCH THIS MODULE'S FIRST TEST DESCRIBES IS GONE. It walked the world
    // in thirteen steps growing 1.18x, and these two numbers are kept only so
    // the size of the banding it caused stays on the record rather than in a
    // commit message. They are deliberately NOT pinned against the shader.
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

    /// The total the march can cover, the way the shader computes it.
    fn reach() -> f32 {
        SSR_MAX_DISTANCE
    }

    /// Distance from the start of the march to the end of coarse step `k` --
    /// the OLD `travelled`, and the only values it could take.
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

    /// THE DEFECT: neighbouring pixels one coarse step apart got fades far
    /// enough apart to see.
    ///
    /// This is what the old code did, kept as a test so the size of the problem
    /// is on the record rather than in a commit message.
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

    /// THE FIX: measured to the refined hit, the fade is continuous, because
    /// the distance is.
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
        // A continuous distance can only change the fade by the smoothstep's
        // own slope times the sample spacing, which is far below anything the
        // eye resolves as a band.
        assert!(
            worst < 0.005,
            "a continuously varying distance still moved the fade by {worst} between \
             neighbouring samples",
        );
    }

    /// Refinement has to actually narrow the distance, or measuring to the
    /// "refined" hit measures to the coarse one.
    #[test]
    fn refinement_narrows_the_hit_to_far_less_than_a_stride() {
        let residual = SSR_STRIDE_MAX / 2f32.powi(SSR_REFINE_COUNT as i32);
        assert!(
            residual < 1.0,
            "six halvings leave {residual} px of a {SSR_STRIDE_MAX} px stride, \
             which is coarser than the buffer the hit is read from",
        );
    }

    /// The constants these tests reason about are the ones the shader uses.
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

/// Where the screen stops being asked and the probe takes over, in METRES.
///
/// The constants are fractions of the march's reach, which makes them easy to
/// get wrong by eye -- 0.75 of reach sounds conservative and is thirteen
/// metres. These pin the distances they actually mean, so a change to the step
/// size or the step count cannot quietly move the handover.
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

    /// Full-strength screen reflections are a CONTACT effect: about a metre.
    ///
    /// That is the range over which a screen-space ray is genuinely more
    /// trustworthy than the probe -- close enough that the reflected surface is
    /// certainly on screen, and close enough that the probe's single viewpoint
    /// is visibly wrong.
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

    /// And they are gone well inside a room, so a long ray -- the only kind
    /// that produces the banding, the ghosting and the sliced reflections -- can
    /// never contribute.
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

    /// The constants above are the shader's.
    #[test]
    fn the_shader_uses_this_handover() {
        let code = wgsl_ssr_block_shared_camera(3);
        assert!(code.contains(&format!("const SSR_TRUST_NEAR: f32 = {SSR_TRUST_NEAR_F};")));
        assert!(code.contains(&format!("const SSR_TRUST_FAR: f32 = {SSR_TRUST_FAR_F};")));
    }

    const SSR_TRUST_NEAR_F: f32 = 0.18;
    const SSR_TRUST_FAR_F: f32 = 0.87;
}

/// The false-colour diagnostic must not ship.
#[cfg(test)]
mod ssr_debug_switch_tests {
    use super::*;

    /// NO MISS COLOUR MAY SIT ON THE GREY DIAGONAL.
    ///
    /// A hit is drawn `vec3(fade)`, which is black at fade 0 and WHITE at fade
    /// 1 and every grey between. So any miss colour with three equal channels
    /// is indistinguishable from some ordinary hit, and the false-colour view
    /// stops being a diagnostic exactly where it is being relied on.
    ///
    /// This is not hypothetical. "Stopped behind an occluder" was white for one
    /// build; the headset reported the artefacts in "white and grey areas",
    /// which described both that reason AND every confident reflection in the
    /// scene, and the build answered nothing (2026-09-18). It was the second
    /// time in two days that one colour meant two things.
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
        // And they must be distinct from each other, for the same reason.
        for (i, a) in colours.iter().enumerate() {
            for b in &colours[i + 1..] {
                assert!(
                    a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-6),
                    "two miss reasons share the colour {a:?}",
                );
            }
        }
    }

    /// Just the body of `ssr_miss_colour`, so the parse below cannot pick up
    /// vectors from the rest of the shader.
    fn ssr_miss_colour_source() -> String {
        let code = wgsl_ssr_block_shared_camera_debug(1, true);
        let start = code
            .find("fn ssr_miss_colour(")
            .expect("`ssr_miss_colour` is gone from the debug shader");
        let rest = &code[start..];
        let end = rest.find("\n}").expect("`ssr_miss_colour` has no closing brace");
        rest[..end].to_string()
    }

    /// Every `vec3<f32>(a, b, c)` literal in `src`, as numbers.
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

    /// A build with the diagnostic on paints every reflective surface in flat
    /// primaries. That is obvious on a screen and invisible in a diff, so the
    /// switch is asserted rather than remembered.
    #[test]
    fn the_diagnostic_is_off() {
        assert!(
            !SSR_DEBUG,
            "SSR_DEBUG is on: reflective surfaces render as flat blue/red/green \
             instead of reflections. Set it back to false.",
        );
    }

    /// And when it is on, it is actually wired into the shader -- otherwise
    /// flipping the switch would produce a normal-looking build and a
    /// diagnostic session would be wasted.
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
    //! Screen-space reflections stand down for rays heading back toward the
    //! viewer, which cannot find anything on screen. See `facing_fade`.
    use super::*;

    /// `smoothstep(-0.7, -0.2, dot(refl_dir, view_dir))`, by incidence angle
    /// from the surface normal, using `dot(refl, view) = -cos(2 * incidence)`.
    fn fade(incidence_deg: f32) -> f32 {
        let d = -(2.0 * incidence_deg.to_radians()).cos();
        let t = ((d + 0.7) / 0.5).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// The identity the twin relies on, checked with real vectors.
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

    /// A floor seen while walking -- 45 to 65 degrees -- keeps its reflection.
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


/// WHY NO FADE MAY READ THE DEPTH BUFFER AWAY FROM ITS OWN HIT.
///
/// The build of 2026-09-17 15:09 judged a hit by how squarely the ray met it:
/// over a fixed span in metres either side of the refined hit, the ray's own
/// depth changes by `dr` and the surface's by `ds`, and the two are equal when
/// the ray runs parallel to the surface. That is true, cheap, phase-free, and
/// it shipped an artefact within one build.
///
/// The measurement it is built on is only meaningful while both reads land on
/// the SAME surface. At a silhouette -- the edge of a ceiling lamp against the
/// wall behind it, one of its wires, the edge of the avatar's own arm -- the
/// second read is metres away in depth, `ds` swamps `dr`, and the ratio falls
/// off the bottom of its range. Every reflected edge in the room was drawn with
/// a black outline, and the reflected arm came back shredded into vertical
/// stripes.
///
/// The term is gone. These tests keep the reason, because the same idea is
/// genuinely tempting -- it is what a depth-only renderer reaches for whenever
/// it wants to know anything about the surface it hit.
#[cfg(test)]
mod silhouette_tests {
    /// The confidence that was removed, transcribed.
    fn square_fade(dr: f32, ds: f32) -> f32 {
        let squareness = ((dr - ds) / dr.abs().max(1e-9)).clamp(0.0, 1.0);
        let t = ((squareness - 0.05) / (0.35 - 0.05)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// On flat ground it behaves exactly as intended, which is why it passed
    /// every test it had and every measurement in `fade_continuity_twin`.
    #[test]
    fn it_reads_correctly_while_both_samples_share_a_surface() {
        assert_eq!(square_fade(0.02, 0.0), 1.0, "a square hit lost strength");
        assert!(
            square_fade(0.02, 0.0198) < 0.01,
            "a ray running parallel to what it hit kept {}",
            square_fade(0.02, 0.0198),
        );
    }

    /// And at an edge it collapses, whichever side the far surface is on.
    #[test]
    fn it_collapses_wherever_the_second_read_crosses_an_edge() {
        // The far read lands on a wall metres behind: `ds` swamps `dr`.
        assert_eq!(
            square_fade(0.02, 0.9),
            0.0,
            "a hit whose neighbouring read fell past a silhouette kept its \
             strength; if this ever passes again the black outlines are back",
        );
        // Or on something metres nearer, which saturates the other way and
        // reads as a perfectly square hit on geometry the ray never met.
        assert_eq!(square_fade(0.02, -0.9), 1.0);
    }

    /// The rule that replaced it.
    #[test]
    fn the_shader_reads_depth_only_along_the_ray_it_marched() {
        // Comments stripped: the note above `SSR_MIN_WEIGHT` names both of the
        // removed terms on purpose, and naming them is the point of it.
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
    //! IS THE REFLECTION'S STRENGTH CONTINUOUS FROM ONE PIXEL TO THE NEXT?
    //!
    //! The headset's SSR view showed the floor's reflection of a wall sliced
    //! into ribbons: inside a ribbon the reflection dimmed smoothly, and at the
    //! ribbon's edge it jumped back to full. Thirteen ribbons, thirteen march
    //! steps (2026-09-17). Where a ribbon reached zero the reflection vanished
    //! -- the black stepping cut across the floor reflection of the wall
    //! spotlight -- and where it jumped back it returned at full strength,
    //! which is the bright band beside it. Both complaints are one bug.
    //!
    //! The cause is not the march's precision. Six refinement halvings put the
    //! hit within four centimetres and the reflected IMAGE was smooth; what
    //! stepped was the number it was multiplied by. Any fade read AT a march
    //! sample inherits the sampling's phase: whether a step boundary falls just
    //! before or just after the surface slides smoothly with the pixel, so the
    //! sampled quantity sweeps a whole step's worth of values and then resets.
    //!
    //! So this is a twin of the march over a floor and a wall, run down one
    //! column of pixels, that measures the largest jump in the FINAL fade
    //! between NEIGHBOURING pixels. The scene is analytic -- no texels, no
    //! MSAA, no quantised depth -- so the only thing left that can make the
    //! answer jump is the march's own phase. That is what makes this a
    //! measurement rather than another screenshot.
    //!
    //! Three designs are compared, and the numbers are printed rather than only
    //! asserted, because the interesting result was that the obvious fix (stop
    //! counting the sample that crossed) removes only half of it.
    //!
    //! TWO THINGS TRIED HERE AND THROWN OUT, so they are not tried again. The
    //! comb of stripes beside the avatar's hand looks like a stretched
    //! reflection point-sampled from mip 0, which would be cured by choosing
    //! the mip from the reflection's screen footprint. Two cheap estimates of
    //! that footprint were written and both were refused by measurement: the
    //! ray's angle against the surface it hit reads 1.0 everywhere, including
    //! where neighbours demonstrably read 33 px apart, and the shaded surface's
    //! own foreshortening is no better -- its widest reads land in its LOWEST
    //! stretch bucket. Whatever the comb is, it is not a footprint the march
    //! can work out from what it already has.
    //!
    //! What the headset says it is: the user noticed that in the false-colour
    //! view, GREEN -- ran out of steps -- appears in the SHAPE of the comb. So
    //! those pixels have no reflection at all and fall back to the probe, while
    //! the pixels beside them found one. That is the search budget, and the
    //! only thing that fixes it is reaching further for the same cost. See
    //! `docs/ssr-hi-z-scope-2026-09.md`.

    use glam::{Mat4, Vec2, Vec3, Vec4};

    // The shader constants, mirrored. `the_twin_uses_the_shipped_constants`
    // pins each one against the shader text.
    const STEPS: u32 = 13;
    const STEP_SIZE: f32 = 0.12;
    const STEP_GROWTH: f32 = 1.18;
    const REFINE: u32 = 6;
    const THICKNESS_STEPS: f32 = 2.5;
    const NORMAL_BIAS: f32 = 0.05;
    const TRUST_NEAR: f32 = 0.18;
    const TRUST_FAR: f32 = 0.87;
    /// `smoothstep(0.0, 0.25, edge)` in the shader.
    const EDGE_FADE_WIDTH: f32 = 0.25;
    /// How far either side of the hit `Cam::squareness` looks, in metres.
    const SQUARENESS_SPAN: f32 = 0.08;

    /// The screen-space march: a fixed stride in PIXELS, the eye buffer's own.
    const STRIDE_PIXELS: f32 = 4.0;
    const STRIDE_MAX: f32 = 24.0;
    const MAX_SCREEN_STEPS: u32 = 32;
    /// Where the search-budget fade starts, as a fraction of the budget.
    /// 1.0 switches it off.
    const BUDGET_FADE_FROM: f32 = 0.6;
    /// How far the ray is allowed to reach, in metres. Past `SSR_TRUST_FAR`
    /// the probe answers anyway, so reaching further is loads for nothing.
    const MAX_DISTANCE: f32 = 5.07;
    const NEAR_W: f32 = 0.06;
    /// McGuire and Mara's `zThickness`: how deep a slab each depth-buffer texel
    /// stands for, in METRES. A scene constant, not a march-step constant --
    /// which is the whole difference.
    const THICKNESS_METRES: f32 = 0.25;
    /// WHERE THE SLAB STOPS BEING A CLIFF.
    ///
    /// The slab test is binary: a ray a hair behind the surface is a hit at
    /// full strength, a ray a hair further is nothing at all and the pixel
    /// goes to the probe. At a silhouette adjacent rays land on either side of
    /// that line -- one passes the hand's edge, the next slips behind it -- so
    /// the reflection alternates on and off from pixel to pixel. That is the
    /// comb.
    ///
    /// FidelityFX SSSR's hit validation produces a CONFIDENCE rather than a
    /// boolean and interpolates towards the environment probe with it. Ours
    /// ramps the slab's far face instead of cutting it: full credit inside
    /// `THICKNESS_METRES`, none past this, smooth in between. It reads nothing
    /// extra -- the overshoot is a subtraction of two numbers the ray already
    /// holds -- which is what separates it from the squareness confidence that
    /// sampled depth around the hit and drew black claws along every edge.
    const THICKNESS_SOFT_METRES: f32 = 0.75;

    /// A panel hanging in front of the wall, purely so the scene HAS an edge.
    /// See `Cam::first_hit`.
    /// HAND-SIZED ON PURPOSE. A large panel gives the scene silhouettes; a
    /// small one also gives it something a world-space march can step clean
    /// over, which is the other half of what the headset showed.
    const PANEL_STANDOFF: f32 = 0.35;
    const PANEL_HALF_WIDTH: f32 = 0.11;
    const PANEL_BOTTOM: f32 = 0.70;
    const PANEL_HEIGHT: f32 = 0.26;

    /// The viewpoints the sweep covers: eye height, how far in front of the
    /// eye the wall is, and how far below the horizon the camera looks.
    ///
    /// ONE VIEWPOINT IS NOT A MEASUREMENT. The first version of this twin
    /// stood four metres back, where the distance fade has already taken the
    /// reflection to nothing, and reported a worst jump of 0.056 -- the
    /// artefact was real and the scene was wrong. The headset's ribbons are
    /// full white, which means the reflected distance is inside the range SSR
    /// is trusted for, which means standing CLOSE.
    /// Eye height, how far ahead the wall is, pitch below the horizon, and YAW
    /// away from facing the wall.
    ///
    /// THE LAST THREE EXIST FOR THE GRAZING CASE. Every viewpoint here used to
    /// face the wall square on, and in that whole family not one pair of
    /// neighbouring pixels read their reflection more than 2.2 px apart -- so
    /// the twin could not see a stretched reflection at all, and could not
    /// judge anything meant to fix one. The headset's comb sits on a wall the
    /// viewer is looking ALONG, where the reflected image is smeared across the
    /// surface and one output pixel covers a long streak of it.
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
        /// What was on the headset when the ribbons were photographed: a hit is
        /// any sample within a few steps' depth behind the surface, the edge is
        /// the closest any sample came to the frame boundary, and confidence
        /// falls off towards the tolerance limit.
        Shipped,
        /// `Shipped`, minus the crossing sample's contribution to the edge.
        CrossingExcluded,
        /// A hit is a CROSSING -- the previous sample in front, this one behind
        /// -- with no step-sized tolerance anywhere. The edge is the exact
        /// minimum over the travelled segment, from its two ends. Confidence
        /// comes from whether the ray is still behind the surface a fixed
        /// number of METRES further on, measured at the refined hit.
        Proposed,
        /// `Proposed` with the hit-confidence term dropped entirely.
        ProposedNoConfidence,
        /// A UNIFORM STRIDE IN SCREEN SPACE, the way McGuire and Mara do it.
        ///
        /// Every design above walks the ray in WORLD space, growing the step
        /// 1.18x each time, so the last steps are the better part of a metre.
        /// Anything smaller than that -- a hand, a lamp, the edge of a doorway
        /// -- fits inside one step, and whether a given pixel's step lands on
        /// it is decided by where that pixel's step boundaries fell. The object
        /// is then stamped once per step instead of reflected once, which is
        /// the row of hand-shaped ghosts marching away across the wall
        /// (headset, 2026-09-17).
        ///
        /// Stepping a fixed number of PIXELS cannot skip anything wider than
        /// the stride, whatever the distance, and costs the same per step.
        ScreenStride,
        /// `ScreenStride` with a per-pixel jitter added to the ray's start.
        ScreenStrideJittered,
        /// HIERARCHICAL-Z: the same screen-space ray, walked through a pyramid
        /// of nearest-depths instead of at a fixed stride.
        ///
        /// A fixed stride has to choose between reach and fineness, because the
        /// step budget is spent either way -- measured, a stride fine enough to
        /// stop the ghosting reaches 77% of the reflections the world march
        /// found, and the headset paints whole walls "ran out of steps". The
        /// pyramid removes the choice: an empty region is crossed in one step at
        /// a coarse level, and the ray only descends where something might be in
        /// the way. Uludag, GPU Pro 5.
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
        /// Where the wall the floor reflects stands.
        wall_z: f32,
    }

    /// Looking down at the floor in front of a wall: the geometry of every
    /// screenshot of this artefact. `pitch` is radians below the horizon.
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

        /// The world direction of the camera ray through `uv` (y down).
        fn ray(&self, uv: Vec2) -> Vec3 {
            let p = self.inv * Vec4::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
            (Vec3::new(p.x, p.y, p.z) / p.w - self.eye).normalize()
        }

        /// The nearest surface along the camera ray through `uv` -- the point
        /// and its normal -- or nothing. Floor, wall, and a panel hanging in
        /// front of the wall.
        ///
        /// THE PANEL IS THERE FOR ITS EDGES. A scene of two infinite planes has
        /// no silhouettes, so a depth buffer read a little away from a hit
        /// always lands on the same surface, and a fade built that way looks
        /// perfect here while collapsing in a real room. The headset's ceiling
        /// lamp, its wire cage and the avatar's own arm are all silhouettes,
        /// and the hit confidence added on 2026-09-17 went to zero along every
        /// one of them: black claw marks strung across the reflected wall, and
        /// the reflected arm shredded into vertical stripes. The twin could not
        /// see any of it until this panel existed.
        ///
        /// THE NORMAL IS PART OF THE ANSWER. The first version returned only
        /// the point and the march assumed every pixel was floor, so the wall
        /// above the crease was marched with the floor's normal and reported
        /// jumps of a full 1.0 in every design at once. A twin that is wrong
        /// about the scene cannot separate two fades.
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
                if t > 0.0 && p.y > 0.0 && best.map_or(true, |(bt, _, _)| t < bt) {
                    best = Some((t, p, Vec3::Z));
                }
                let t = (self.wall_z + PANEL_STANDOFF - self.eye.z) / d.z;
                let p = self.eye + d * t;
                if t > 0.0
                    && p.x.abs() < PANEL_HALF_WIDTH
                    && p.y > PANEL_BOTTOM
                    && p.y < PANEL_BOTTOM + PANEL_HEIGHT
                    && best.map_or(true, |(bt, _, _)| t < bt)
                {
                    best = Some((t, p, Vec3::Z));
                }
            }
            best.map(|(_, p, n)| (p, n))
        }

        /// Is this pixel looking at the floor? Only the floor's reflection is
        /// measured, and only where BOTH neighbours are floor, so the crease
        /// between the two surfaces is never counted as a jump in the fade.
        fn is_floor(&self, uv: Vec2) -> bool {
            self.first_hit(uv).is_some_and(|(_, n)| n == Vec3::Y)
        }

        /// The depth buffer this march reads, evaluated exactly.
        fn scene_ndc_z(&self, uv: Vec2) -> f32 {
            match self.first_hit(uv) {
                Some((p, _)) => {
                    let c = self.project(p);
                    c.z / c.w
                }
                None => 1.0,
            }
        }

        /// The depth of the ray and the depth of the scene at one point on the
        /// ray, or nothing if the point is off screen.
        fn ray_and_scene(&self, p: Vec3) -> Option<(f32, f32)> {
            let c = self.project(p);
            if c.w <= 0.0 {
                return None;
            }
            // CLAMPED, not rejected. A depth load clamps to the edge texel;
            // refusing to answer instead made the squareness probe fail for
            // every hit whose neighbourhood reached past the frame, and a
            // failed probe faded the hit to nothing beside a neighbour at full
            // strength -- a cliff of 0.057 a pixel inside the boundary.
            let uv = uv_of(c).clamp(Vec2::ZERO, Vec2::ONE);
            Some((c.z / c.w, self.scene_ndc_z(uv)))
        }

        /// HOW SQUARELY THE RAY MET WHAT IT HIT, from 0 (skimming along the
        /// surface) to 1 (driving straight into it).
        ///
        /// Measured over a fixed number of METRES either side of the refined
        /// hit, so it depends on where the hit is and not on which march step
        /// happened to find it. Over that span the ray's own depth changes by
        /// `dr` and the surface's by `ds`; the two are equal when the ray runs
        /// parallel to the surface, which is where a screen-space hit is worth
        /// least -- its position slides a hundred pixels for every pixel of the
        /// surface, so neither the reflection nor any fade built on it can be
        /// resolved. It is also the case the old step-sized thickness
        /// tolerance was trying to judge, and could not: the tolerance grew
        /// with the step, so the same geometry was accepted or rejected by
        /// where the step boundary fell.
        fn squareness(&self, hit: Vec3, refl: Vec3, metres: f32) -> Option<f32> {
            let (a_ray, a_scene) = self.ray_and_scene(hit - refl * metres)?;
            let (b_ray, b_scene) = self.ray_and_scene(hit + refl * metres)?;
            let dr = b_ray - a_ray;
            let ds = b_scene - a_scene;
            Some(((dr - ds) / dr.abs().max(1e-9)).clamp(0.0, 1.0))
        }
    }

    /// How far inside the frame a point is, in uv units: 0 at the boundary,
    /// 0.5 dead centre. The shader's `min(min(uv.x, 1 - uv.x), ...)`.
    fn edge_of(uv: Vec2) -> f32 {
        uv.x.min(1.0 - uv.x).min(uv.y).min(1.0 - uv.y)
    }

    /// Where a segment that ends outside the frame crosses the frame boundary.
    ///
    /// Clip coordinates are linear along a straight segment, so each boundary
    /// is one division. Nudged back inside by a thousandth so the point is on
    /// screen rather than exactly on the line.
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

    /// One pixel's answer: the reflection's strength, and how far its ray went.
    struct Marched {
        /// WHICH surface the ray ended on: 0 floor, 1 wall, 2 panel.
        ///
        /// Only neighbours that landed on the SAME surface are compared. Where
        /// the reflection crosses the panel's edge the reflected image changes
        /// abruptly because the scene does, and a fade that changes with it is
        /// not an artefact -- it is a nearer thing being trusted more. Counting
        /// those put 0.529 on every design at once and hid the difference
        /// between them.
        surface: u8,
        /// WHERE THE REFLECTION WAS READ FROM. A reflection that is continuous
        /// reads from a place that moves smoothly as the pixel moves; a
        /// reflection stamped once per march step reads from somewhere tens of
        /// pixels away from what its neighbour read.
        hit_uv: Vec2,
        fade: f32,
        travelled: f32,
        edge: f32,
        grazing: f32,
        distance: f32,
        confidence: f32,
        facing: f32,
    }

    /// Where the reflection was READ from, and where the ray met the world.
    struct Crossing {
        hit_uv: Vec2,
        hit_pos: Vec3,
        origin: Vec3,
        /// How much of the ray's SEARCH BUDGET had been spent, 0 to 1.
        budget_used: f32,
        /// How far inside the thickness slab the ray stopped, 1 at the front
        /// face and 0 at `THICKNESS_SOFT_METRES`. See that constant.
        slab: f32,
    }

    /// The McGuire-Mara march: walk the projected ray a fixed number of pixels
    /// at a time, carrying `1/w` and the depth, both of which are linear in the
    /// screen-space parameter.
    ///
    /// The hit test is theirs too. The ray covers a depth INTERVAL over the
    /// step it just took, and the surface is a slab `SSR_THICKNESS_METRES`
    /// thick behind what the depth buffer shows; a hit is those two
    /// overlapping. That accepts a crossing wherever inside the step it
    /// happened AND a thin surface the step jumped clean over, which the
    /// world-space march could only do by luck.
    /// Set by the diagnostics below to print why a ray found nothing.
    thread_local! {
        static TRACE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    /// Whether `march_hi_z` ramps the slab's far face or cuts it.
    ///
    /// OFF, because the measurement below refused it. The twin must match the
    /// shipped shader or nothing measured in it means anything; this switch
    /// exists only so the rejected design can be re-measured rather than
    /// re-argued.
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

    /// A rasterised depth buffer for one viewpoint, and the pyramid of nearest
    /// depths above it -- what the GPU builds in `resolve_depth` + `build_hi_z`.
    ///
    /// Level 0 is one texel per eye-buffer pixel. Each level above holds the
    /// NEAREST depth of the texels below, and an odd level is reduced with a 3x3
    /// so the bound still covers the column a 2x2 would drop. Both of those
    /// match the shader; `hi_z_tests` pins the GPU side against the same rules.
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
    /// Where the traversal starts. Level 0 would be a linear march; too coarse
    /// and every ray pays for descents it did not need.
    const HI_Z_START_LEVEL: usize = 2;
    /// A cap, not a plan: a ray that has not resolved in this many cell steps is
    /// crossing pathological geometry and is better handed to the probe.
    const HI_Z_MAX_ITERATIONS: u32 = 64;
    /// How far past a cell boundary to step, in TEXELS. See `t_eps`.
    const CELL_NUDGE_TEXELS: f32 = 0.05;
    const MIN_RAY_TEXELS: f32 = 2.0;

    thread_local! {
        /// Rays that used every iteration without resolving. A STALL, not a
        /// long search: the walk is taking the same step forever.
        static STALLS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        static WALKS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    /// The cell-crossing walk. Mirrors what the shader will do.
    ///
    /// At each level the ray is inside one cell. The cell holds the nearest
    /// depth anything in it reaches, so if the ray is still in FRONT of that
    /// when it leaves the cell, nothing in the cell can have been hit and the
    /// whole cell is skipped in one step -- and the traversal goes UP a level to
    /// try skipping a bigger one. If instead the ray would reach that depth
    /// inside the cell, something might be there, so it advances to the depth
    /// plane and goes DOWN a level to look closer. Reaching level 0 and still
    /// meeting the depth is the hit.
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
        // See the shader: a ray that covers almost no screen has nothing to
        // walk, and the nudge below divides by its length.
        if ds.length() < MIN_RAY_TEXELS {
            return None;
        }
        let b = if (k1 - k0).abs() < 1e-9 { 0.0 } else { dz / (k1 - k0) };
        let a = z0 - b * k0;
        if b == 0.0 {
            return None;
        }

        // A QUARTER OF A TEXEL, in screen space, however long the ray is.
        //
        // The nudge past a cell boundary has to be a distance on SCREEN. As a
        // constant in `t` it is a quarter texel on a ray that crosses the whole
        // frame and a millionth of one on a short ray -- and a nudge that fails
        // to leave the cell means the next iteration finds the same cell, takes
        // the same step, and the walk stalls until the iteration cap gives up.
        // On the headset that drew vertical green streaks, one per column of
        // stalled rays (2026-09-17).
        let at = |t: f32| s0 + ds * t;
        // One texel INSIDE the first cell, so the walk does not begin by
        // intersecting the cell it started in.
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
            // Where the ray leaves this cell: the nearer of the two axis
            // boundaries it is heading towards.
            let next_x = if ds.x > 0.0 { (cell.x + 1.0) * cell_size } else { cell.x * cell_size };
            let next_y = if ds.y > 0.0 { (cell.y + 1.0) * cell_size } else { cell.y * cell_size };
            let tx = if ds.x.abs() < 1e-6 { f32::INFINITY } else { (next_x - s0.x) / ds.x };
            let ty = if ds.y.abs() < 1e-6 { f32::INFINITY } else { (next_y - s0.y) / ds.y };
            // A hair past the boundary, or the next iteration lands on the same
            // cell and the walk stalls.
            // Nudged on the axis being CROSSED and sized by that axis -- see
            // the shader.
            let t_exit = if tx < ty {
                tx + CELL_NUDGE_TEXELS / ds.x.abs().max(1e-6)
            } else {
                ty + CELL_NUDGE_TEXELS / ds.y.abs().max(1e-6)
            };

            let cell_min = pyramid.at(level, cell.x as i32, cell.y as i32);
            // The ray's depth is linear in t, so the t at which it reaches the
            // cell's nearest surface is one division.
            let t_depth = if dz.abs() < 1e-12 { f32::INFINITY } else { (cell_min - z0) / dz };

            // `t_depth <= t_exit` ALONE, deliberately. Requiring the depth
            // plane to also lie ahead of the ray's current position skips every
            // cell the ray has already reached the depth of -- which is most of
            // them once it is close to a surface. Measured in the gate below:
            // 63 reflections found against the stride's 1692.
            // THE CANONICAL STEP (Uludag, GPU Pro 5): advance to the depth
            // plane FIRST, then ask whether that left the cell.
            //
            // Both of my earlier shapes got this wrong from opposite ends.
            // Requiring the depth plane to lie ahead of the ray skipped every
            // cell it had already passed the depth of -- 63 reflections found
            // against 3395. Dropping the requirement made the ray descend at
            // every cell once it was behind anything, crawl a texel at a time
            // at level 0 and climb again, three iterations per texel: a
            // spreading GREEN swathe beside the avatar's hand, and the pass at
            // 8-9 ms an eye instead of 0.65.
            //
            // Advancing first settles both. If the advance stays inside the
            // cell, something in that cell is worth a closer look, so descend.
            // If it leaves, the ray passed through without meeting anything, so
            // move to the boundary and climb. A ray already behind a surface
            // does not advance at all, stays in its cell, and descends to level
            // 0 where the slab decides -- which is how it gets past an occluder
            // instead of grinding along behind it.
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
                // Behind it: on past, and climb. See `MAX_THICKNESS` in the
                // reference -- this is what lets a ray continue behind an
                // object rather than stopping at it.
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
        // Pull the far end back to the near plane rather than letting it wrap
        // round behind the camera.
        if h1.w <= NEAR_W {
            let t = (h0.w - NEAR_W) / (h0.w - h1.w);
            p1 = p0.lerp(p1, t);
            h1 = c.project(p1);
        }
        let (k0, k1) = (1.0 / h0.w, 1.0 / h1.w);
        let (z0, z1) = (h0.z * k0, h1.z * k1);
        let (s0, s1) = (uv_of(h0) * res, uv_of(h1) * res);

        // z_ndc = A + B/w for any standard perspective matrix, so the depth the
        // buffer holds converts to a distance from the eye with one divide.
        // Solved from the ray's own two ends rather than passed in.
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

        // UNIFORM WITHIN A RAY, and never finer than it needs to be.
        //
        // A stride fixed at four pixels reaches 96 of them in the step budget,
        // which is five per cent of the frame -- the twin found a reflection
        // for 870 probes against the world march's 5448. Spreading the ray's
        // own screen length over the budget keeps the cost fixed and the stride
        // CONSTANT along the ray, which is the property that matters: what went
        // wrong before was the step growing 1.18x as it went, not its size.
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
            // The ray's depth interval over this step against the slab behind
            // the visible surface.
            let (near, far) = (w_prev.min(w), w_prev.max(w));
            if far >= scene_w && near <= scene_w + THICKNESS_METRES {
                // Bisect in the same parameter to put the hit inside a pixel.
                let (mut lo, mut hi) = (t_prev, t);
                for _ in 0..REFINE {
                    let mid = (lo + hi) * 0.5;
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
                    // The stride's hit test is an INTERVAL against the slab,
                    // not a point, so there is no overshoot to ramp.
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

    /// The fades, given where the ray started and where it landed. Every one
    /// of them depends on the surface being shaded or on the REFINED hit.
    fn finish(c: &Cam, n: Vec3, view_dir: Vec3, facing_fade: f32, x: Crossing) -> Marched {
        let edge = edge_of(uv_of(c.project(x.origin))).min(edge_of(uv_of(c.project(x.hit_pos))));
        let grazing_fade = smoothstep(0.15, 0.45, n.dot(-view_dir));
        let travelled = (x.hit_pos - x.origin).length();
        let distance_fade = 1.0 - smoothstep(TRUST_NEAR, TRUST_FAR, travelled / MAX_DISTANCE);
        let budget_fade = 1.0 - smoothstep(BUDGET_FADE_FROM, 1.0, x.budget_used);
        let surface = if (x.hit_pos.z - (c.wall_z + PANEL_STANDOFF)).abs() < 0.02 {
            2
        } else if (x.hit_pos.z - c.wall_z).abs() < 0.02 {
            1
        } else {
            0
        };
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

    /// `Design::HiZ` needs the pyramid for the viewpoint; every other design
    /// ignores it. Built once per viewpoint by the callers, because building it
    /// per pixel would rasterise three million texels three million times.
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
        // A ray heading back towards the viewer reflects what is behind the
        // head, which was never on screen. The shader fades those out and does
        // not march them at all; leaving it out of the twin put a cliff of
        // 0.235 in every design, at pixels the shader never marches.
        let facing_fade = smoothstep(-0.7, -0.2, refl.dot(view_dir));
        if facing_fade <= 0.0 {
            return None;
        }
        if design == Design::HiZ {
            let x = march_hi_z(c, pyramid?, world_pos, n, refl)?;
            return Some(finish(c, n, view_dir, facing_fade, x));
        }
        if design == Design::ScreenStride || design == Design::ScreenStrideJittered {
            // Interleaved gradient noise, the usual per-pixel jitter: it trades
            // a hard band for noise the eye reads as detail. See Jimenez 2014.
            let px = pixel * Vec2::new(EYE_W, EYE_H);
            // `ScreenStride` is the shipped march, which does NOT jitter --
            // see the note in the shader. `ScreenStrideJittered` is kept to
            // measure what jittering costs when nothing denoises it.
            let jitter = if design == Design::ScreenStrideJittered {
                (52.9829189 * (0.06711056 * px.x + 0.00583715 * px.y).fract()).fract()
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
                // THE CROSSING MAY STILL BE INSIDE THE FRAME.
                //
                // The march notices a crossing only at the first sample BEHIND
                // the surface, and that sample is a whole step past the
                // crossing. When it lands just off screen the ray is called
                // "left the frame" although the surface it met is on screen and
                // one pixel over it is found at full strength -- which is the
                // cliff of 0.145 this twin measured with everything else fixed.
                // Clipping the segment to the boundary and asking the same
                // crossing question on what is left costs one division.
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

        // Binary refinement, as the shader does it.
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
            // Exact, from the two ends of the travelled segment. See
            // `the_edge_distance_is_monotone_along_a_segment`.
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
            // How squarely the ray met the surface, over a fixed span in
            // metres around the refined hit. See `Cam::squareness`.
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
        } else if (hit_pos.z - c.wall_z).abs() < 0.02 {
            1
        } else {
            0
        };
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

    /// The worst jump in the final fade between two pixels that are ACTUALLY
    /// ADJACENT in the eye buffer, over a grid of the floor and over every
    /// viewpoint -- with where it happened, and how many probes reflected
    /// anything at all.
    ///
    /// ADJACENT MEANS ONE EYE-BUFFER PIXEL. A grid of 56 rows spread over half
    /// the frame puts its samples sixteen pixels apart, and at sixteen pixels a
    /// reflection that races up a wall and off the top of the frame -- which is
    /// continuous, and correct -- reads as a jump of 0.97. Every design scored
    /// about 1.0 and the measurement said nothing. The step below is the
    /// headset's own: 1680 by 1760 per eye.
    ///
    /// Horizontal neighbours as well as vertical: a fade that steps along one
    /// axis only is still a band, and the ribbons in the screenshots run across
    /// the view rather than down it.
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
                // The floor fills the lower part of the frame.
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
                    // A neighbour that reflected nothing counts, and counts as
                    // zero: a reflection at full strength beside none is the
                    // artefact. Only a neighbour that landed on a DIFFERENT
                    // surface is skipped, because there the reflected image
                    // changes with the fade and the change is not a defect.
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

    /// HOW FAR APART TWO NEIGHBOURING PIXELS READ THEIR REFLECTION FROM, in
    /// eye-buffer pixels, at the worst place on the floor.
    ///
    /// The fade being smooth says the reflection does not flicker; it says
    /// nothing about whether the reflection is the RIGHT one. A march that
    /// steps over a hand-sized object stamps it once per step: neighbouring
    /// pixels then read from parts of the scene tens of pixels apart, and the
    /// object appears several times over, each copy a little larger. That is
    /// what the headset showed on 2026-09-17 -- a row of hand-shaped ghosts
    /// marching away across the wall, and the doorway's reflection folded into
    /// an accordion.
    ///
    /// Some spread is honest: a reflection at a grazing angle is genuinely
    /// stretched. So this compares designs on the same pixels rather than
    /// asserting an absolute.
    fn measure_read_jump(design: Design) -> Worst {
        // A probe counts as reflecting only if the reflection would be VISIBLE.
        // Counting hits that the distance fade has already taken to nothing
        // makes a march look better the further it wastes its steps.
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
                        // Only where both landed on the same surface: across a
                        // silhouette the reflection genuinely changes subject.
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
        // Re-find it by brute force and print the neighbourhood.
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
                        let same = o.as_ref().map_or(true, |o| o.surface == m.surface);
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

    /// THE COMB, MEASURED: how often does a pixel reflect at full strength
    /// while the pixel beside it reflects nothing?
    ///
    /// This is the artefact left on the headset after the traversal was fixed
    /// -- a fine on/off ribbing in the floor's reflection beside the avatar's
    /// hand in the back corner. Every reference treats it as the expected
    /// OUTPUT of a screen-space trace, removed after the fact rather than
    /// inside the march: FidelityFX SSSR turns its hit test into a confidence
    /// and interpolates to the environment probe with it.
    ///
    /// The cheapest version of that, and the only one available without a
    /// separate reflection buffer to filter, is to ramp the slab's far face.
    /// This measures whether that is worth doing before any WGSL changes.
    ///
    /// Slow on purpose: a full eye-buffer pyramid per viewpoint.
    #[test]
    #[ignore = "slow: rasterises a full 1680x1760 depth buffer per viewpoint"]
    fn a_soft_slab_takes_the_cliff_out_of_the_hit_miss_boundary() {
        /// A jump this big between two neighbouring pixels is visible ribbing.
        const CLIFF: f32 = 0.5;
        /// Consecutive eye-buffer pixels scanned per row.
        const RUN: usize = 600;
        let measure_comb = |soft: bool| {
            SOFT_SLAB.with(|c| c.set(soft));
            let (mut cliffs, mut pairs, mut worst, mut reflected) = (0usize, 0usize, 0.0f32, 0usize);
            for (height, wall, pitch, yaw) in VIEWPOINTS {
                let c = cam(height, wall, pitch, yaw);
                let pyramid = Pyramid::build(&c);
                let pyr = Some(&pyramid);
                // DENSE RUNS, NOT A GRID. A grid of 32 columns puts its
                // samples fifty pixels apart and asks about the pixel one
                // across: it can only find ribbing if a sample lands inside
                // it, and the comb on the headset covers a hand's reflection.
                // Scanning CONSECUTIVE pixels is what the eye does.
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
                            // A neighbour that found NOTHING counts, and counts
                            // as zero -- that is the comb. A neighbour that
                            // landed on another surface does not: there the
                            // reflected image genuinely changes subject.
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
        // THE RESULT, RECORDED SO IT IS NOT RE-ARGUED. Ramping the far face
        // does not remove the cliffs -- it moves the boundary and adds a few,
        // because every newly accepted hit brings its own edge with it. The
        // handover has to be weighted ACROSS PIXELS, which needs a reflection
        // buffer to filter, not a smarter test inside one ray. See
        // `docs/ssr-quality-plan-2026-09-18.md`.
        assert!(
            soft_cliffs >= hard_cliffs,
            "the soft slab now REMOVES cliffs ({soft_cliffs} against \
             {hard_cliffs}); it did not when it was measured, so re-read the \
             plan before acting on this",
        );
        // AND THE LIMIT OF THIS MEASUREMENT. Both designs sit near 0.01% of
        // neighbouring pairs, which is nothing like the ribbing the headset
        // shows beside the avatar's hand. The twin's occluder is a panel
        // standing 0.35 m off a wall, so a ray that slips behind it lands on
        // the wall and is counted as a change of subject rather than a miss.
        // Until the twin has an occluder with OPEN SPACE behind it, it cannot
        // see this artefact and must not be cited as evidence that a fix for
        // it works.
        assert!(
            hard_cliffs * 200 < hard_pairs,
            "the twin now reproduces the comb at {hard_cliffs} cliffs in \
             {hard_pairs} pairs; it did not before, so it has become able to \
             measure the headset's artefact and the note above is stale",
        );
    }

    /// THE GATE STAGE C HAS TO PASS before any of it reaches the shader.
    ///
    /// Three things, from `docs/ssr-hi-z-scope-2026-09.md`: read no further
    /// apart than the stride does, find at least as many reflections as the
    /// world march did, and do it without the step budget deciding either.
    ///
    /// Slow on purpose -- it rasterises a full eye buffer per viewpoint and
    /// reduces a pyramid over it, which is exactly what the GPU does per frame.
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
        let (stalls, walks) = (STALLS.with(|s| s.get()), WALKS.with(|w| w.get()));
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

    /// The ghosting, measured, and the stride that removes it.
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
        // THE COST OF A FIXED BUDGET, on the record as a number.
        //
        // A stride covers a fixed number of PIXELS, so a ray stretched across
        // the view runs out before it reaches anything: the twin finds 77% of
        // the reflections the world-space march found, and the headset shows
        // the same thing as whole walls painted "ran out of steps" in the
        // false-colour view, with a serrated edge where the budget ends and a
        // comb of the same edge around the avatar's hand.
        //
        // This is the one thing the stride made WORSE, it is not a tuning
        // problem -- a coarser stride reaches further and brings the ghosting
        // straight back, measured -- and closing it is the whole point of
        // hierarchical-Z traversal. See `docs/ssr-hi-z-scope-2026-09.md`. The
        // bound is here so that work has a number to beat rather than an
        // impression to argue with.
        assert!(
            stride.probes * 4 >= world.probes * 3,
            "the stride march reflected {} probes against the world march's \
             {}: the search budget is falling even further short than it was",
            stride.probes,
            world.probes,
        );
    }

    /// WHY THE TWO ENDS OF THE SEGMENT ARE ENOUGH.
    ///
    /// Along a straight world segment the clip coordinates are linear in the
    /// segment parameter, so each frustum plane's distance and `w` are both
    /// linear and the normalised distance is their ratio -- a linear-fractional
    /// function, which is monotone wherever `w` keeps its sign. A monotone
    /// function on an interval takes its minimum at an end. So the closest a
    /// ray came to the frame edge anywhere along its path is the smaller of its
    /// two ends, EXACTLY, with no sampling involved.
    ///
    /// The same statement says a ray can leave the frame only once: no straight
    /// segment exits and re-enters.
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

    /// The measurement that names the artefact, and sizes each fix.
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

    /// Every constant the twin mirrors, against the shader text.
    ///
    /// `STEPS`, `STEP_SIZE`, `STEP_GROWTH` and `THICKNESS_STEPS` are
    /// deliberately absent: they belong to `Design::Shipped` and
    /// `Design::CrossingExcluded`, which model marches the shader no longer
    /// has. They are kept because the sizes they produce are the record of what
    /// was wrong, and pinning them to a shader that has moved on would only
    /// force someone to delete the record.
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

