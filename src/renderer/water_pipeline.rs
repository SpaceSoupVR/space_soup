//! Standing water and the open sea.
//!
//! WHAT MAKES WATER READ AS WATER
//!
//! In order of how much it matters on a headset:
//!
//! 1. **Fresnel.** Water is nearly clear looking straight down and nearly a
//!    mirror at a glancing angle -- why a lake shows you its bed at your feet
//!    and the sky at the far bank. What it mirrors is the world: the sky, the
//!    hills round it and the buildings on its shore (`outdoor_radiance` in the
//!    lights block), not a tint.
//! 2. **Real waves.** The surface is the FFT wave field of `water_waves`: the
//!    sea a wind of so many metres a second raises over so many metres of
//!    open water. Near the eye its long waves move the surface itself, so in
//!    stereo the swell has depth; everywhere its slopes are the normals, and
//!    what its mip levels average away comes back as roughness, so the far
//!    water is a soft sheen rather than aliased glitter.
//! 3. **Light inside the water.** Each colour of light is lost at its own rate
//!    on the way down to the bed and back up -- red first -- so a sandy bed
//!    goes green-blue and then disappears into the water's own colour, the
//!    light it scatters back. `shallow` is authored as a colour, and turned
//!    into those rates here.
//!
//! WHY THE BED IS TINTED BY A SECOND BLEND SOURCE
//!
//! Seeing the bed through the water means attenuating what is already in the
//! colour buffer, by a different amount per channel. One alpha cannot do that:
//! it greys the sand instead of turning it blue-green. Reading the scene
//! colour back would, but on a tile GPU that means ending the pass, resolving
//! and copying -- the most expensive thing a transparent surface can ask for.
//! Dual-source blending does it in the blend unit: the shader's second output
//! is the factor the bed is multiplied by. Where the device lacks it, the same
//! shader blends with one alpha, the mean of the three.
//!
//! WHY THE DEPTH COMES FROM THE GROUND MAP
//!
//! The bed's height under each pixel is read from the ground map the lights
//! block already binds -- not from the depth buffer, which a tile GPU would
//! have to resolve. Past the map's edge -- past the terrain, under a sea's
//! skirt -- the sea floor runs on down from the edge, and nothing is drawn
//! under the water there: what the view meets out there is lit by the shader
//! itself, as light as the map's edge, rather than shown through by the blend.
//! Taking it to be bottomless instead drew the line where a view first left
//! the map as a hard edge between turquoise and navy. The vertices carry depth
//! for a level with no map.
//!
//! WHY IT WRITES DEPTH
//!
//! The sky is drawn after the world, wherever depth is still clear, and a sea
//! runs past the terrain to the horizon over nothing: without depth the sky
//! would paint over it. SpaceWarp reprojects by depth too, and the surface is
//! what is seen. What lies under the water is drawn before it, so nothing
//! that should show through is lost.

use wgpu::*;

use crate::renderer::lights::wgsl_lights_block;
use crate::renderer::water_waves::{WaveField, WaveParams, MIPS, N as WAVE_N};

/// One vertex of the water surface, as the GPU sees it: in the WORLD, posed
/// into the player's frame by the vertex shader. A surface the player walks
/// round is never re-uploaded.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterVertex {
    pub position: [f32; 3],
    /// Metres from the still surface down to the ground: negative on a beach.
    pub depth: f32,
}

/// How many times the depth, on average, the bed's light crosses on its way
/// DOWN to the bed: light from the sky arrives at every angle, not only from
/// overhead. Its way back up to the eye is the view's own path.
pub const LIGHT_PATH: f32 = 1.15;
/// Seconds between breakers on a beach. Divides the waves' 240 s loop, so the
/// whole sea repeats as one.
pub const SWASH_PERIOD: f32 = 8.0;
/// How much later in its period a breaker reaches water a metre shallower:
/// the speed they roll in at.
const SWASH_LEAD: f32 = 0.45;
/// THE BREAKERS' SHAPE. A wave rolling in grows as the water shoals until it
/// is as high as the depth can hold (`BREAK_RATIO` of it, McCowan's 0.78),
/// then breaks and runs on as a bore no higher than that. At its tallest it is
/// `BREAKER_PER_SWASH` times the beach's swash. Its face toward the shore is
/// steep (a Gaussian `BREAKER_FRONT` of a period wide), its back long and
/// gentle (`BREAKER_BACK`); on the cove's 1-in-9 sand a period is about 20 m.
/// Before this the shore's waves only lifted the water as a sheet, and a
/// breaker was a white line on a flat surface.
const BREAKER_PER_SWASH: f32 = 1.8;
const BREAK_RATIO: f32 = 0.78;
const BREAKER_FRONT: f32 = 0.035;
const BREAKER_BACK: f32 = 0.07;
/// Metres from the eye over which a sea's skirt is raised to the eye's own
/// height, so its far edge meets the sky AT the horizon. Left level, the edge
/// lay a few ten-thousandths of a radian below it, where the sky draws its
/// dark ground: a sliver under a pixel, which MSAA showed as a dashed line
/// along the horizon that crawled as the head moved (headset, 2026-10-07).
const HORIZON_LIFT: (f32, f32) = (4000.0, 15000.0);
/// CAT'S PAWS: how far the ripples fall in the lulls between gusts, by the
/// wind. In light airs over a pond or a sheltered cove gusts roughen drifting
/// patches while the lulls lie glassy; by a fresh breeze (6 m/s and more) the
/// whole surface stays rippled and there are none.
fn paws_calm(wind_speed: f32) -> f32 {
    let t = ((wind_speed - 2.0) / 4.0).clamp(0.0, 1.0);
    0.15 + 0.85 * t * t * (3.0 - 2.0 * t)
}
/// THE SWELL'S WAVES (`WaterDef::swell`): wavelength, metres, way off the
/// wind, radians, and share of the height. Each runs at deep water's own
/// speed for its length, rounded to whole turns in the waves' loop.
const SWELL: [(f32, f32, f32); 3] = [(4.3, 0.0, 0.5), (2.9, 0.55, 0.3), (2.1, -0.7, 0.2)];
/// Swell wave `i` as WGSL: its `k` vector in the wind's frame, `omega`, and
/// share.
fn swell_wgsl(i: usize) -> String {
    let w0 = 2.0 * std::f32::consts::PI / crate::renderer::water_waves::WaveParams::default().loop_seconds;
    let (length, turn, share) = SWELL[i];
    let k = 2.0 * std::f32::consts::PI / length;
    let omega = ((9.81 * k).sqrt() / w0).round() * w0;
    format!("vec4<f32>({:?}, {:?}, {:?}, {:?})", k * turn.cos(), k * turn.sin(), omega, share)
}
/// Foam's bubbles repeat this many times a metre.
const FOAM_TILES: f32 = 1.0 / 3.0;
/// What the bed gives back of the sunlight the waves gather onto it: sand.
const BED_ALBEDO: f32 = 0.45;
/// Metres from the eye within which the waves move the surface itself; past
/// it they are normals alone. A wave a hand high is a few pixels at forty
/// metres.
pub const WAVE_REACH: f32 = 40.0;

/// SPLASH RINGS: the newest splashes' ripples spreading over the surface
/// (`effects::splash_rings`), each a short train of waves -- a few
/// centimetres long, as a foot or a hand raises -- running out at about the
/// slowest speed water waves have, widening and dying away.
pub const RINGS: usize = 4;
pub const RING_SECONDS: f32 = 3.0;
const RING_SPEED: f32 = 0.3;
const RING_WAVELENGTH: f32 = 0.08;
const RING_HEIGHT: f32 = 0.006;

/// Per-body optics and the waves' state, as the shader takes them.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterUniform {
    /// xyz: each channel's light lost per metre of water, as exp(-k d);
    /// w: the still surface's height, world y.
    pub extinction: [f32; 4],
    /// xyz: deep water's own colour, linear; w: the shore fade, metres.
    pub scatter: [f32; 4],
    /// xyz: the wave cascades' tile sides, metres; w: seconds, wrapped to the
    /// waves' loop.
    pub tiles: [f32; 4],
    /// x: the slope variance of the ripples no cascade holds; y: how high the
    /// shore's waves run, metres; z: `WAVE_REACH`, 0 for a surface the waves
    /// never move; w: `SWASH_PERIOD`.
    pub waves: [f32; 4],
    /// Per mip level of the wave textures, the slope variance each cascade's
    /// level averages away. See `WaveParams::lost_slope_variance`.
    pub lost: [[f32; 4]; MIPS as usize],
    /// Splashes' rings: world x, z, seconds since, strength (0: none, and
    /// none after it). See `RINGS`.
    pub rings: [[f32; 4]; RINGS],
    /// x: how far the ripples fall in the lulls between gusts (1: no lulls;
    /// see `paws_calm`); y: the swell, metres; zw: the wind's way, x and z.
    pub air: [f32; 4],
}

/// A body's optics as authored, in linear light.
#[derive(Copy, Clone, Debug)]
pub struct WaterOptics {
    /// The still surface, world y.
    pub height: f32,
    /// A white bed seen straight down through `depth_scale` metres of water.
    pub shallow: [f32; 3],
    /// The light the water itself scatters back: its colour when too deep to
    /// see through.
    pub deep: [f32; 3],
    pub depth_scale: f32,
    pub shore_fade: f32,
    /// How high the shore's waves run up a beach, metres.
    pub swash: f32,
    /// Metres the surface heaves in long slow undulations (`WaterDef::swell`).
    pub swell: f32,
}

impl WaterUniform {
    pub fn new(o: &WaterOptics, waves: &WaveParams) -> Self {
        // Straight down, the bed's light crosses the depth 1 + LIGHT_PATH
        // times; this is the loss per metre that makes white look `shallow`.
        let k = |c: f32| -c.clamp(1e-4, 1.0).ln() / (o.depth_scale.max(0.01) * (1.0 + LIGHT_PATH));
        Self {
            extinction: [k(o.shallow[0]), k(o.shallow[1]), k(o.shallow[2]), o.height],
            scatter: [o.deep[0], o.deep[1], o.deep[2], o.shore_fade.max(1e-3)],
            tiles: [waves.tile[0], waves.tile[1], waves.tile[2], 0.0],
            waves: [waves.unresolved_slope_variance(), o.swash.max(0.0), WAVE_REACH, SWASH_PERIOD],
            lost: waves.lost_slope_variance(),
            rings: [[0.0; 4]; RINGS],
            air: [paws_calm(waves.wind_speed), o.swell.max(0.0), waves.wind_dir.cos(), waves.wind_dir.sin()],
        }
    }

    /// The shore's waves' time, wrapped to the wave field's loop so they
    /// repeat with it.
    pub fn set_time(&mut self, seconds: f64, loop_seconds: f32) {
        self.tiles[3] = seconds.rem_euclid(loop_seconds.max(1.0) as f64) as f32;
    }
}

pub struct WaterPipeline {
    pub pipeline: RenderPipeline,
    /// The same without the splashes' rings: see `RING_BLOCK`.
    pub ringless: RenderPipeline,
    /// THE WATERLINE TWIN, drawn only while an eye straddles the surface:
    /// the same surface and shading, leaving out every pixel whose view
    /// starts under the water -- the underwater view's, split from this one
    /// along the same line (`underwater`). See `water_line_wgsl`.
    pub waterline: RenderPipeline,
    pub material_layout: BindGroupLayout,
    /// The waves' and the foam's: repeating, filtered between mip levels.
    pub sampler: Sampler,
    /// Whether the bed is tinted channel by channel (dual-source blending) or
    /// by one alpha. See the module notes.
    pub dual_source: bool,
    /// Its layout, target and blending, for a cut built later.
    inputs: WaterInputs,
}

impl WaterPipeline {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        let dual = Self::device_blends_two(device);
        Self::new_with(device, format, camera_layout, samples, crate::renderer::multiview::ViewMode::Mono, dual)
    }

    /// The same, drawing BOTH EYES in one pass. See `multiview::ViewMode`.
    pub fn new_stereo(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        let dual = Self::device_blends_two(device);
        Self::new_with(device, format, camera_layout, samples, crate::renderer::multiview::ViewMode::Stereo, dual)
    }

    /// Whether this device can blend by a second source.
    pub fn device_blends_two(device: &Device) -> bool {
        device.features().contains(Features::DUAL_SOURCE_BLENDING)
    }

    pub fn new_with(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        dual_source: bool,
    ) -> Self {
        let texture = |binding, visibility, view_dimension| BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let material_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("water_material_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // The fragments' too, for the waterline twin's test (`water_line_wgsl`).
                texture(1, ShaderStages::VERTEX_FRAGMENT, TextureViewDimension::D2Array),
                texture(2, ShaderStages::FRAGMENT, TextureViewDimension::D2Array),
                texture(3, ShaderStages::FRAGMENT, TextureViewDimension::D2Array),
                texture(4, ShaderStages::FRAGMENT, TextureViewDimension::D2),
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("water_waves_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("water_pipeline_layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&material_layout)],
            immediate_size: 0,
        });

        // dst * second source + first: the bed tinted, then the light the
        // water adds. Without the feature, premultiplied alpha.
        let blend = if dual_source {
            BlendState {
                color: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::Src1, operation: BlendOperation::Add },
                alpha: BlendComponent { src_factor: BlendFactor::Zero, dst_factor: BlendFactor::One, operation: BlendOperation::Add },
            }
        } else {
            BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent { src_factor: BlendFactor::Zero, dst_factor: BlendFactor::One, operation: BlendOperation::Add },
            }
        };

        let pipeline = build_water(device, &layout, format, blend, samples, view, water_wgsl_with(dual_source, true), "water_pipeline");
        let ringless = build_water(device, &layout, format, blend, samples, view, water_wgsl_with(dual_source, false), "water_pipeline_ringless");
        let waterline = build_water(device, &layout, format, blend, samples, view, water_line_wgsl(dual_source), "water_pipeline_waterline");
        let inputs = (layout, format, blend, samples, view);
        if crate::renderer::shader_checks::PIPELINE_STATISTICS.load(std::sync::atomic::Ordering::Relaxed) {
            // MEASUREMENT: each cut built once, for the PIPESTATS log.
            for (cut, _) in WATER_CUTS {
                let _ = water_with_cut(device, &inputs, dual_source, cut);
            }
        }

        Self { pipeline, ringless, waterline, material_layout, sampler, dual_source, inputs }
    }

    /// MEASUREMENT: the ringless water with one of [`WATER_CUTS`], to draw in
    /// its place (`Levers::water_cut`). `None` for a cut it does not have, or
    /// one whose text no longer matches.
    pub fn with_cut(&self, device: &Device, cut: &str) -> Option<RenderPipeline> {
        water_with_cut(device, &self.inputs, self.dual_source, cut)
    }

    /// A body's material, once for each of the wave field's two sets: draw
    /// with `[waves.current()]`.
    pub fn bind_groups(&self, device: &Device, uniform: &Buffer, waves: &WaveField) -> [BindGroup; 2] {
        std::array::from_fn(|set| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("water_material"),
                layout: &self.material_layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&waves.displacement_views[set]) },
                    BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&waves.derivatives_view) },
                    BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&waves.curvature_views[set]) },
                    BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&waves.foam_view) },
                    BindGroupEntry { binding: 5, resource: BindingResource::Sampler(&self.sampler) },
                ],
            })
        })
    }
}

/// What a water pipeline is built from besides its source.
type WaterInputs = (PipelineLayout, TextureFormat, BlendState, u32, crate::renderer::multiview::ViewMode);

/// MEASUREMENT ONLY: parts of the water's fragment stage, each cut out by a
/// text edit of the ringless shader -- what each costs by its presence and
/// its work (`Levers::water_cut`), and in instructions and registers
/// (PIPESTATS `water_cut_<name>`). Not the same picture.
pub(crate) const WATER_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("none", &[]),
    // The breaker's face, its slope from the phase over the bed map.
    (
        "breaker_face",
        &[("    if (water.waves.y > 0.0 && still_depth > 0.0 && still_depth < 2.8) {\n        breaker_phase", "    if (false) {\n        breaker_phase")],
    ),
    // The swell's tilt.
    ("swell", &[("        tilt = tilt + water.air.y * smoothstep(0.15, 0.6, still_depth) * swell_at(q, water.tiles.w).yz;\n", "")]),
    // Cat's paws: even ripples everywhere.
    ("paws", &[("    let paws = mix(water.air.x, 1.0, smoothstep(-0.6, 0.7, lull));", "    let paws = 1.0;")]),
    // The third cascade's curvature in the caustics.
    ("c2", &[("    let c2 = textureSampleBias(wave_curve, wave_samp, q / water.tiles.z, 2, 2.0) * paws;", "    let c2 = vec4<f32>(0.0);")]),
    // The caustics altogether.
    ("caustics", &[("    let caustic = mix(1.0, gather, CAUSTIC_CONTRAST * murk * (1.0 - smoothstep(1.5, 5.0, thickness)));", "    let caustic = 1.0;")]),
    // The breakers' and the wash's foam.
    ("surf", &[("    if (water.waves.y > 0.0) {\n        let phase = swash_phase(still_depth, q, water.tiles.w);", "    if (false) {\n        let phase = swash_phase(still_depth, q, water.tiles.w);")]),
    // The walk to the drawn bed: the map's picture everywhere.
    ("walk", &[("    if (on_map) {\n        let flat", "    if (false) {\n        let flat")]),
    // The map's picture of the bed.
    ("beyond", &[("    if (drawn < 1.0) {\n        beyond", "    if (false) {\n        beyond")]),
    // The refracted path's second read of the bed.
    ("refract", &[("    if (has_map) {\n        let closing", "    if (false) {\n        let closing")]),
    // The reflection: the ground's trace and the probes.
    ("env", &[("    let env = outdoor_radiance(", "    let env = vec3<f32>(0.0);\n    let env_unused = outdoor_radiance(")]),
    // The sun's glint (its lights loop's GGX).
    ("glint", &[("        glint = glint + radiance * WATER_PI * water_ggx(n, v, to_l, a2);\n", "")]),
];

fn water_with_cut(device: &Device, inputs: &WaterInputs, dual_source: bool, cut: &str) -> Option<RenderPipeline> {
    let (_, edits) = WATER_CUTS.iter().find(|(name, _)| *name == cut)?;
    let mut src = water_wgsl_with(dual_source, false);
    for (from, to) in edits.iter() {
        if src.matches(from).count() != 1 {
            log::warn!("water cut {cut}: `{from}` is not in the shader once");
            return None;
        }
        src = src.replacen(from, to, 1);
    }
    let (layout, format, blend, samples, view) = inputs;
    Some(build_water(device, layout, *format, *blend, *samples, *view, src, &format!("water_cut_{cut}")))
}

#[allow(clippy::too_many_arguments)]
fn build_water(
    device: &Device,
    layout: &PipelineLayout,
    format: TextureFormat,
    blend: BlendState,
    samples: u32,
    view: crate::renderer::multiview::ViewMode,
    source: String,
    label: &str,
) -> RenderPipeline {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(view.shader(source).into()),
        });
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(VertexBufferLayout {
                    array_stride: std::mem::size_of::<WaterVertex>() as BufferAddress,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState { format, blend: Some(blend), write_mask: ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                // NO BACK-FACE CULLING. A crest folding over, or a camera that
                // has waded in, sees the surface's back -- and a culled surface
                // simply vanishes there.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                // WRITES depth: see the module notes. Everything meant to be
                // seen through the water is drawn before it.
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        })
}

/// Metres a body's waves may carry its surface from its still vertices,
/// beyond twice their significant height: the long waves' sideways push and
/// the shore's swash. Generous, because it only widens a box that culls.
const WAVE_ALLOWANCE: f32 = 2.0;

/// The world box a body's surface can reach: its vertices, as far as its waves
/// can move them. See [`water_seen`].
pub fn water_bounds(verts: &[WaterVertex], waves: &WaveParams) -> (glam::Vec3, glam::Vec3) {
    let (lo, hi) = verts.iter().fold(
        (glam::Vec3::splat(f32::INFINITY), glam::Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), v| {
            let p = glam::Vec3::from(v.position);
            (lo.min(p), hi.max(p))
        },
    );
    let reach = glam::Vec3::splat(WAVE_ALLOWANCE + 2.0 * waves.significant_height());
    (lo - reach, hi + reach)
}

/// WHETHER A BODY OF WATER IS SEEN this frame -- whether its waves are worth
/// moving and its surface worth drawing: its `bounds` ([`water_bounds`]) in
/// one of `views`, each eye's frustum, and from inside a closed room in one of
/// `doorways` (`portal_cull::outdoor_frusta`; `None` when nothing can be
/// culled that way). All in the world, where the bounds are.
///
/// The far planes are no limit: a sea's skirt runs on past them, drawn held
/// just inside (`vs_main`'s horizon). Out of sight, the waves' compute alone
/// cost a millisecond and more a frame (Quest 3, 2026-10-07).
///
/// A helper rather than a test written into the frame, which only the headset
/// builds: see `shadow::chunk_seen`.
pub fn water_seen(
    bounds: (glam::Vec3, glam::Vec3),
    views: &[[glam::Vec4; 6]],
    doorways: Option<&[[glam::Vec4; 6]]>,
) -> bool {
    let meets = |planes: &[glam::Vec4; 6]| {
        let mut unbounded = *planes;
        unbounded[5] = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        crate::renderer::shadow::aabb_in_frustum(&unbounded, bounds.0, bounds.1)
    };
    views.iter().any(meets) && doorways.map_or(true, |d| d.iter().any(meets))
}

/// How far the shore's swash lifts the surface at a world `xz` over still
/// water `still_depth` deep, at the uniform's time: `swash_phase` and
/// `swash_rise` in [`surface_wgsl`], on the CPU, so a splash sits on the
/// water drawn rather than under it -- in a breaker's wash the drawn surface
/// stands a hand or more above the still one.
pub fn swash_lift(u: &WaterUniform, still_depth: f32, xz: glam::Vec2) -> f32 {
    let swash = u.waves[1];
    if swash <= 0.0 {
        return swell_lift(u, still_depth, xz);
    }
    let smooth = |e0: f32, e1: f32, x: f32| {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let along = 0.06 * xz.dot(glam::Vec2::new(0.071, 0.113)).sin() + 0.04 * (xz.dot(glam::Vec2::new(-0.193, 0.051)) + 1.7).sin();
    let phase = (u.tiles[3] / u.waves[3] + still_depth * SWASH_LEAD + along).rem_euclid(1.0);
    let rise = smooth(0.0, 0.16, phase) * (1.0 - smooth(0.16, 0.92, phase));
    // The breaker (`breaker_height`, `breaker_profile`).
    let grown = BREAKER_PER_SWASH * swash * (1.0 - smooth(1.2, 2.8, still_depth));
    let height = grown.min(BREAK_RATIO * still_depth.max(0.0));
    let s = phase - phase.round();
    let b = s / BREAKER_BACK;
    let profile = if s > 0.0 { (1.0 + b) * (-b).exp() } else { (-(s * s) / (BREAKER_FRONT * BREAKER_FRONT)).exp() };
    swash * (1.0 - smooth(0.3, 1.5, still_depth)) * rise + height * profile + swell_lift(u, still_depth, xz)
}

/// The swell's lift (`swell_at` in [`surface_wgsl`]) on the CPU.
fn swell_lift(u: &WaterUniform, still_depth: f32, xz: glam::Vec2) -> f32 {
    if u.air[1] <= 0.0 {
        return 0.0;
    }
    let w0 = 2.0 * std::f32::consts::PI / crate::renderer::water_waves::WaveParams::default().loop_seconds;
    let along = glam::Vec2::new(u.air[2], u.air[3]);
    let across = glam::Vec2::new(-along.y, along.x);
    let height: f32 = SWELL
        .iter()
        .enumerate()
        .map(|(i, &(length, turn, share))| {
            let k = 2.0 * std::f32::consts::PI / length;
            let omega = ((9.81 * k).sqrt() / w0).round() * w0;
            let kv = along * (k * turn.cos()) + across * (k * turn.sin());
            share * (kv.dot(xz) - omega * u.tiles[3] + i as f32 * 2.1).sin()
        })
        .sum();
    // Still at the shoreline: lifted there, the edge stood out of the bank.
    let t = ((still_depth - 0.15) / 0.45).clamp(0.0, 1.0);
    u.air[1] * height * t * t * (3.0 - 2.0 * t)
}

/// THE SURFACE THE WATER DRAWS, as WGSL for every shader that places it:
/// [`WaterUniform`] as `WaterMat`, and the shore's swash. The including shader
/// declares `water: WaterMat` and a filtering `wave_samp`, and makes its
/// surface functions with [`surface_fn_wgsl`].
///
/// Shared rather than copied because SpaceWarp's motion pass (`space_warp`)
/// places this same surface at this frame and the last: the motion the
/// compositor is told has to be the motion drawn.
pub fn surface_wgsl() -> String {
    format!(
        r#"
const SWASH_LEAD: f32 = {swash_lead};
const BREAKER_PER_SWASH: f32 = {breaker_per_swash};
const BREAK_RATIO: f32 = {break_ratio};
const BREAKER_FRONT: f32 = {breaker_front};
const BREAKER_BACK: f32 = {breaker_back};

struct WaterMat {{
    extinction: vec4<f32>,
    scatter: vec4<f32>,
    tiles: vec4<f32>,
    waves: vec4<f32>,
    lost: array<vec4<f32>, {mips}>,
    rings: array<vec4<f32>, {rings}>,
    air: vec4<f32>,
}}

// THE SHORE'S WAVES. Each breaker rolls in toward the beach -- a point a
// metre deeper meets it `SWASH_LEAD` of a period sooner -- and arrives a
// little unevenly along the shore. 0 is its crest.
fn swash_phase(still_depth: f32, xz: vec2<f32>, t: f32) -> f32 {{
    let along = 0.06 * sin(dot(xz, vec2<f32>(0.071, 0.113))) + 0.04 * sin(dot(xz, vec2<f32>(-0.193, 0.051)) + 1.7);
    return fract(t / water.waves.w + still_depth * SWASH_LEAD + along);
}}

// Its wash: quickly up the sand, slowly back down it.
fn swash_rise(phase: f32) -> f32 {{
    return smoothstep(0.0, 0.16, phase) * (1.0 - smoothstep(0.16, 0.92, phase));
}}

// THE BREAKER over still water `still_depth` deep: as high as it has grown
// shoaling in, no higher than the water there can hold.
fn breaker_height(still_depth: f32) -> f32 {{
    let grown = BREAKER_PER_SWASH * water.waves.y * (1.0 - smoothstep(1.2, 2.8, still_depth));
    return min(grown, BREAK_RATIO * max(still_depth, 0.0));
}}

// Its shape about its crest, 1 there, by the swash's phase: steep ahead of it
// (toward the shore, where the phase is still to come), long behind it, and
// level at the crest itself so the light does not crease along it.
fn breaker_profile(phase: f32) -> f32 {{
    let s = phase - round(phase);
    let b = s / BREAKER_BACK;
    return select(exp(-(s * s) / (BREAKER_FRONT * BREAKER_FRONT)), (1.0 + b) * exp(-b), s > 0.0);
}}

// Its rate of change with the phase.
fn breaker_profile_slope(phase: f32) -> f32 {{
    let s = phase - round(phase);
    let b = s / BREAKER_BACK;
    return select(-2.0 * s / (BREAKER_FRONT * BREAKER_FRONT) * exp(-(s * s) / (BREAKER_FRONT * BREAKER_FRONT)), -b / BREAKER_BACK * exp(-b), s > 0.0);
}}

// THE SWELL at a world xz and time: its height, per metre of `air.y`, and its
// slope along x and z.
// Written out wave by wave: a loop over a local array spills on Adreno.
fn swell_wave(w: vec4<f32>, xz: vec2<f32>, t: f32, offset: f32) -> vec3<f32> {{
    let along = water.air.zw;
    let k = along * w.x + vec2<f32>(-along.y, along.x) * w.y;
    let phase = dot(k, xz) - w.z * t + offset;
    return w.w * vec3<f32>(sin(phase), k * cos(phase));
}}
fn swell_at(xz: vec2<f32>, t: f32) -> vec3<f32> {{
    return swell_wave({swell0}, xz, t, 0.0) + swell_wave({swell1}, xz, t, 2.1) + swell_wave({swell2}, xz, t, 4.2);
}}
"#,
        swash_lead = format!("{SWASH_LEAD:?}"),
        breaker_per_swash = format!("{BREAKER_PER_SWASH:?}"),
        break_ratio = format!("{BREAK_RATIO:?}"),
        breaker_front = format!("{BREAKER_FRONT:?}"),
        breaker_back = format!("{BREAKER_BACK:?}"),
        swell0 = swell_wgsl(0),
        swell1 = swell_wgsl(1),
        swell2 = swell_wgsl(2),
        mips = MIPS,
        rings = RINGS,
    )
}

/// A WGSL function `name(pos, depth, eye, t) -> vec3<f32>`: where the water
/// draws the still-surface point `pos`, `depth` metres above the ground, at
/// time `t` (`WaterMat.tiles.w`'s clock), seen from `eye` in the world --
/// moved by the long waves in the texture `disp` and by the shore's swash.
///
/// One function per texture rather than a texture argument: nothing else in
/// the renderer passes a texture to a function, and the headset's shader
/// compiler is where a failure would be silent.
pub fn surface_fn_wgsl(name: &str, disp: &str) -> String {
    format!(
        r#"
fn {name}(pos: vec3<f32>, depth: f32, eye: vec3<f32>, t: f32) -> vec3<f32> {{
    var p = pos;
    // THE LONG WAVES MOVE THE SURFACE, near the eye, and die away into the
    // shallows. The two longer cascades: the third's are a hand's width,
    // which the normals carry.
    let reach = water.waves.z;
    let near = select(0.0, 1.0 - smoothstep(reach * 0.6, reach, distance(pos.xz, eye.xz)), reach > 0.0);
    let open = near * smoothstep(0.0, 1.2, depth);
    if (open > 0.0) {{
        let d0 = textureSampleLevel({disp}, wave_samp, pos.xz / water.tiles.x, 0, 1.0).xyz;
        let d1 = textureSampleLevel({disp}, wave_samp, pos.xz / water.tiles.y, 1, 3.0).xyz;
        p = p + (d0 + d1) * open;
    }}
    if (water.waves.y > 0.0) {{
        let shore = 1.0 - smoothstep(0.3, 1.5, depth);
        let phase = swash_phase(depth, pos.xz, t);
        p.y = p.y + water.waves.y * shore * swash_rise(phase) + breaker_height(depth) * breaker_profile(phase);
    }}
    if (water.air.y > 0.0) {{
        p.y = p.y + water.air.y * swell_at(pos.xz, t).x * near * smoothstep(0.15, 0.6, depth);
    }}
    // THE HORIZON: a sea's far skirt raised to the eye's height (`HORIZON_LIFT`).
    let far = distance(pos.xz, eye.xz);
    p.y = mix(p.y, max(p.y, eye.y), smoothstep({lift0}, {lift1}, far));
    return p;
}}
"#,
        lift0 = format!("{:?}", HORIZON_LIFT.0),
        lift1 = format!("{:?}", HORIZON_LIFT.1),
    )
}

/// The rings' loop (`RINGS`), left out of the RINGLESS twin: drawn while no
/// splash's rings are spreading -- nearly always -- the water does not pay
/// for code it skips, which on the headset cost about a millisecond at the
/// shore (bench 2026-10-07_2155).
const RING_BLOCK: &str = r#"    if (water.rings[0].w > 0.0) {
        let resolved = 1.0 - smoothstep(0.25, 0.5, span / RING_WAVELENGTH);
        for (var i = 0; i < RINGS_N; i = i + 1) {
            let r = water.rings[i];
            if (r.w <= 0.0) {
                break;
            }
            let d = q - r.xy;
            let dist = length(d);
            let age = r.z;
            let spread = 0.03 + 0.08 * age;
            let x = dist - RING_SPEED * age;
            let k = 2.0 * WATER_PI / RING_WAVELENGTH;
            let amp = RING_HEIGHT * sqrt(r.w) * exp(-age / 1.1) / sqrt(1.0 + dist * 10.0);
            let train = exp(-x * x / (2.0 * spread * spread));
            let rise = amp * resolved * train * -k * sin(k * x);
            tilt = tilt + d / max(dist, 1e-4) * rise;
            // Too fine for the pixel, its slopes roughen the water instead:
            // far off, a ring is a band of duller or sparklier water.
            ring_rough = ring_rough + 0.5 * (amp * k) * (amp * k) * train * (1.0 - resolved);
            let foam_reach = 0.12 + 0.1 * sqrt(r.w) + 0.15 * age;
            ring_foam = ring_foam + min(0.35 * sqrt(r.w), 0.6) * exp(-dist * dist / (foam_reach * foam_reach)) * (1.0 - smoothstep(0.4, RING_SECONDS, age));
        }
    }
"#;

fn water_wgsl(dual: bool) -> String {
    water_wgsl_with(dual, true)
}

/// THE WATERLINE TWIN's shader: the water's, its rings and all, with one test
/// before it returns -- whether the pixel's view starts under the surface
/// where it crosses the near plane, by the same surface and in the same steps
/// as `underwater`'s veil (`uw_surface_y`). The two then split the view along
/// one line. Tested against the drawn triangles instead, the two disagreed by
/// a few millimetres at the near plane -- the grid's flat triangles against
/// the waves' own surface -- and a band under the line showed the water's top
/// over an already fogged bed (offline, 2026-10-08).
const WATER_LINE_VIEW: &str = "    let t_view = exp(-k * path);";
const WATER_LINE_RISING: &str = "    let t_view = select(exp(-k * path), vec3<f32>(0.0), down.y > 0.0);";

fn water_line_wgsl(dual: bool) -> String {
    const TEST: &str = r#"
// Whether the view through this pixel starts under the surface: see
// `water_line_wgsl`. The surface across the near plane as a plane through
// its height at the eye, as `underwater`'s `uw_line_plane` finds it.
fn water_line_y(xz: vec2<f32>, eye: vec3<f32>) -> f32 {
    let surface = water.extinction.w;
    var depth = 30.0;
    if (camera.sky_params.w > 0.5) {
        depth = surface - bed_at(vec3<f32>(xz.x, 0.0, xz.y)).w;
    }
    let t = water.tiles.w;
    let first = water_surface(vec3<f32>(xz.x, surface, xz.y), depth, eye, t);
    let rest = xz - (first.xz - xz);
    return water_surface(vec3<f32>(rest.x, surface, rest.y), depth, eye, t).y;
}
fn water_line_under(player: vec3<f32>) -> bool {
    let c = cam_view_proj() * vec4<f32>(player, 1.0);
    let h = cam_inv_view_proj() * vec4<f32>(c.xy / c.w, 0.0, 1.0);
    let start = to_world_space(h.xyz / h.w);
    let eye = to_world_space(cam_pos());
    let h0 = water_line_y(eye.xz, eye);
    let hx = water_line_y(eye.xz + vec2<f32>(0.1, 0.0), eye);
    let hz = water_line_y(eye.xz + vec2<f32>(0.0, 0.1), eye);
    let plane = vec3<f32>(h0, (hx - h0) * 10.0, (hz - h0) * 10.0);
    return plane.x + dot(start.xz - eye.xz, plane.yz) - start.y > 0.0;
}

"#;
    let src = water_wgsl_with(dual, true);
    assert_eq!(src.matches("@fragment fn fs_main").count(), 1);
    assert_eq!(src.matches("    return FsOut(").count(), 1);
    assert_eq!(src.matches(WATER_LINE_VIEW).count(), 1);
    // And a view that RISES into the water -- from the air in a trough into
    // the face of the wave ahead, which only an eye inside the waves' height
    // has -- meets nothing drawn behind it, and showed the clear colour
    // through the surface: it sees the water's own light alone.
    src.replacen("@fragment fn fs_main", &format!("{TEST}@fragment fn fs_main"), 1)
        .replacen(WATER_LINE_VIEW, WATER_LINE_RISING, 1)
        .replacen("    return FsOut(", "    if (water_line_under(in.world_pos)) {\n        discard;\n    }\n    return FsOut(", 1)
}

fn water_wgsl_with(dual: bool, rings: bool) -> String {
    let ring_block = if rings { RING_BLOCK.replace("RINGS_N", &RINGS.to_string()) } else { String::new() };
    let output = if dual {
        "struct FsOut {\n    @location(0) @blend_src(0) colour: vec4<f32>,\n    @location(0) @blend_src(1) through: vec4<f32>,\n}"
    } else {
        "struct FsOut {\n    @location(0) colour: vec4<f32>,\n}"
    };
    let ret = if dual {
        "return FsOut(vec4<f32>(tonemap(colour) + beyond, 1.0), vec4<f32>(through, 1.0));"
    } else {
        // One alpha for every channel: the bed greys rather than tinting.
        "return FsOut(vec4<f32>(tonemap(colour) + beyond, 1.0 - dot(through, vec3<f32>(1.0 / 3.0))));"
    };
    format!(
        r#"{enable}
{lights_block}

const WATER_PI: f32 = 3.14159265;
const WATER_MIPS: i32 = {mips};
const WAVE_N: f32 = {wave_n};
const LIGHT_PATH: f32 = {light_path};
const IOR_WATER: f32 = 1.333;
// A view that never meets the floor -- over a drop it cannot follow -- is
// taken to cross this many times the water a flat bed would have put in its
// way: enough that nothing shows through.
const OPEN_PATH: f32 = 12.0;
// The eye's straight line walked to the bed drawn behind a pixel: its first
// step, how much longer each is than the last, and how many -- in lengths of
// the line's way to a flat bed at this depth.
const WALK_FIRST: f32 = 0.6;
const WALK_GROWTH: f32 = 1.55;
const WALK_STEPS: i32 = 6;
// How steeply the sea floor falls away past the ground map's edge.
const SHELF_SLOPE: f32 = 0.15;
const FOAM_TILES: f32 = {foam_tiles};
const BED_ALBEDO: f32 = {bed_albedo};
const RING_SPEED: f32 = {ring_speed};
const WATER_TRACE_COARSEN: i32 = 2;
const CAUSTIC_CONTRAST: f32 = 0.6;
const CAUSTIC_MURK: f32 = 1.0;
const PAWS_RATE: vec2<f32> = vec2<f32>(0.1308997, 0.0785398);
const RING_WAVELENGTH: f32 = {ring_wavelength};
const RING_HEIGHT: f32 = {ring_height};
const RING_SECONDS: f32 = {ring_seconds};
{surface}
@group(1) @binding(0) var<uniform> water: WaterMat;
@group(1) @binding(1) var wave_disp: texture_2d_array<f32>;
@group(1) @binding(2) var wave_slope: texture_2d_array<f32>;
@group(1) @binding(3) var wave_curve: texture_2d_array<f32>;
@group(1) @binding(4) var foam_tex: texture_2d<f32>;
@group(1) @binding(5) var wave_samp: sampler;

struct VIn {{
    @location(0) pos: vec3<f32>,
    @location(1) depth: f32,
}}
struct VOut {{
    @builtin(position) clip: vec4<f32>,
    // The surface point in the PLAYER's frame, for the eye and the lights.
    @location(0) world_pos: vec3<f32>,
    // Metres from the still surface down to the ground, at the vertex.
    @location(1) depth: f32,
    // The same point in the WORLD, waves and all: what the bed and the
    // reflection are found from.
    @location(2) tex_pos: vec3<f32>,
    // Where on the still surface the point started, world x/z: the waves' own
    // coordinates. Sampled where it moved to, a crest would read the slope of
    // the water it was pushed over.
    @location(3) rest: vec2<f32>,
}}

{surface_fn}
@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    let p = water_surface(v.pos, v.depth, to_world_space(cam_pos()), water.tiles.w);
    let player = to_player_space(p);
    out.clip = cam_view_proj() * vec4<f32>(player, 1.0);
    // THE HORIZON. A sea's skirt runs on past the far plane; held just inside
    // it, the water meets the sky instead of stopping a sliver short of it.
    if (out.clip.w > 0.0) {{
        out.clip.z = min(out.clip.z, out.clip.w * 0.999999);
    }}
    out.world_pos = player;
    out.depth = v.depth;
    out.tex_pos = p;
    out.rest = v.pos.xz;
    return out;
}}

// THE BED at a point: xyz the light it sends back, w its height. The ground
// map's, and past its edge the edge's own, running on down at `SHELF_SLOPE`.
fn bed_at(at: vec3<f32>) -> vec4<f32> {{
    let uv = ground_uv(at);
    let half_texel = 0.5 / vec2<f32>(textureDimensions(ground_map));
    let edge = clamp(uv, half_texel, vec2<f32>(1.0) - half_texel);
    let g = textureSampleLevel(ground_map, probe_samp, edge, 0.0);
    return vec4<f32>(g.rgb, g.a - SHELF_SLOPE * length((uv - edge) / camera.ground_params.zw));
}}

// The slope variance a level of the wave textures has averaged away, between
// levels as the filtering blends them.
fn water_lost(lod: f32) -> vec4<f32> {{
    let l = clamp(lod, 0.0, f32(WATER_MIPS - 1));
    let i = i32(floor(l));
    let j = min(i + 1, WATER_MIPS - 1);
    return mix(water.lost[i], water.lost[j], l - f32(i));
}}

// GGX, its Smith visibility and Schlick's Fresnel for water, times n.l.
fn water_ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, a2: f32) -> f32 {{
    let nl = max(dot(n, l), 0.0);
    let h = normalize(l + v);
    let nh = max(dot(n, h), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (WATER_PI * dd * dd);
    let vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2) + 1e-5);
    let f = 0.02 + 0.98 * pow(1.0 - max(dot(v, h), 0.0), 5.0);
    return d * vis * f * nl;
}}

{output}

@fragment fn fs_main(in: VOut) -> FsOut {{
    // The waves' coordinates, and how many metres of them a pixel spans:
    // taken first, while every fragment of the quad is still running.
    let q = in.rest;
    let span = max(length(dpdx(q)), length(dpdy(q)));
    let s0 = textureSample(wave_slope, wave_samp, q / water.tiles.x, 0);
    let s1 = textureSample(wave_slope, wave_samp, q / water.tiles.y, 1);
    // CAT'S PAWS: the wind never blows evenly over water. Gusts roughen
    // drifting patches of it while the lulls between lie glassy, and the
    // ripples (the third cascade, and the ones too short for any) come and go
    // with them -- the look of a real pond or bay, where an even field of
    // ripples everywhere read as busy. Rates whole turns in the waves' loop.
    let lull = sin(dot(q, vec2<f32>(0.47, 0.31)) + water.tiles.w * PAWS_RATE.x + 1.6 * sin(dot(q, vec2<f32>(-0.27, 0.58)) - water.tiles.w * PAWS_RATE.y));
    let paws = mix(water.air.x, 1.0, smoothstep(-0.6, 0.7, lull));
    let s2 = textureSample(wave_slope, wave_samp, q / water.tiles.z, 2) * paws;
    let c0 = textureSample(wave_curve, wave_samp, q / water.tiles.x, 0);
    let c1 = textureSample(wave_curve, wave_samp, q / water.tiles.y, 1);
    // The third cascade's curvature two levels coarse: its waves of 15-29 cm
    // net the bed as a beach's caustics do; the finger-width ripples under
    // that only made a crawling speckle (`curve` below).
    let c2 = textureSampleBias(wave_curve, wave_samp, q / water.tiles.z, 2, 2.0) * paws;
    let bubbles = textureSample(foam_tex, wave_samp, q * FOAM_TILES).r;

    let p = in.tex_pos;
    let surface = water.extinction.w;
    // THE BED under this pixel, from the ground map -- the shore is as fine
    // as the map, not the grid -- and on past its edge (`bed_at`); from the
    // vertices where there is no map.
    let has_map = camera.sky_params.w > 0.5;
    let uv = ground_uv(p);
    let on_map = has_map && all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0));
    let bed = select(surface - in.depth, bed_at(p).w, has_map);
    let still_depth = surface - bed;
    let thickness = max(p.y - bed, 0.0);

    // THE SURFACE'S SLOPE, all three cascades. The long waves die away into
    // the shallows as they break; the ripples stay. Tessendorf's sideways
    // push sharpens the crests: the slope over the push's own stretch.
    let shoal = smoothstep(0.0, 1.2, still_depth);
    let slope = (s0.xy + s1.xy) * shoal + s2.xy;
    let push = vec2<f32>(1.0) + (s0.zw + s1.zw) * shoal + s2.zw;
    var tilt = slope / max(push, vec2<f32>(0.25));
    // THE BREAKER'S FACE, here rather than from the vertices: the grid is
    // most of a metre, the steep face less. Its slope is the profile's along
    // the phase, times how fast the phase changes across the ground -- taken
    // over half a metre of the bed map, which is smooth where the grid's own
    // depth would crease at every triangle.
    if (water.air.y > 0.0) {{
        tilt = tilt + water.air.y * smoothstep(0.15, 0.6, still_depth) * swell_at(q, water.tiles.w).yz;
    }}
    var breaker_phase = 0.0;
    if (water.waves.y > 0.0 && still_depth > 0.0 && still_depth < 2.8) {{
        breaker_phase = swash_phase(still_depth, q, water.tiles.w);
        let ax = p + vec3<f32>(0.5, 0.0, 0.0);
        let az = p + vec3<f32>(0.0, 0.0, 0.5);
        let px = swash_phase(surface - select(bed, bed_at(ax).w, has_map), q + vec2<f32>(0.5, 0.0), water.tiles.w);
        let pz = swash_phase(surface - select(bed, bed_at(az).w, has_map), q + vec2<f32>(0.0, 0.5), water.tiles.w);
        let dx = px - breaker_phase;
        let dz = pz - breaker_phase;
        let across = vec2<f32>(dx - round(dx), dz - round(dz)) / 0.5;
        tilt = tilt + breaker_height(still_depth) * breaker_profile_slope(breaker_phase) * across;
    }}
    // SPLASHES' RINGS, and the foam each leaves where it struck. A ring's
    // train is gone where a pixel spans half its wavelength: further off it
    // would only alias.
    var ring_foam = 0.0;
    var ring_rough = 0.0;
{ring_block}    let n = normalize(to_player_direction(vec3<f32>(-tilt.x, 1.0, -tilt.y)));
    // ROUGHNESS: every wave the filtering averaged away at this pixel's span,
    // and the ripples too short for any cascade. Slope variance is GGX's
    // alpha squared.
    let lost0 = water_lost(log2(max(span * WAVE_N / water.tiles.x, 1e-6))).x;
    let lost1 = water_lost(log2(max(span * WAVE_N / water.tiles.y, 1e-6))).y;
    let lost2 = water_lost(log2(max(span * WAVE_N / water.tiles.z, 1e-6))).z;
    let a2 = clamp((lost0 + lost1) * shoal * shoal + (lost2 + water.waves.x) * paws * paws + ring_rough, 4e-4, 1.0);
    let rough = sqrt(sqrt(a2));

    let v = normalize(cam_pos() - in.world_pos);
    let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);

    // WHAT IT MIRRORS: the ground, the buildings and the sky, as every
    // outdoor reflection sees them. A ray the waves tilt below the horizon is
    // held just above it.
    var r = reflect(-v, n);
    r.y = max(r.y, 0.002);
    r = normalize(r);
    ground_trace_coarsen = WATER_TRACE_COARSEN;
    let env = outdoor_radiance(p, normalize(to_world_direction(r)), r, rough, clamp(rough * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD));

    // THE SUN'S GLINT, and the lights the game adds as it runs -- the torch.
    // A level's own lamps are in its bake; what they shed on water is left to
    // the reflections.
    var glint = vec3<f32>(0.0);
    var sunlight = vec3<f32>(0.0);
    for (var i = 0u; i < lights.count.x; i = i + 1u) {{
        let l = lights.lights[i];
        let kind = l.params.z;
        if (kind < 1.5 && l.position.w > -0.5) {{
            continue;
        }}
        var to_l = normalize(-l.direction.xyz);
        var atten = 1.0;
        if (kind > 1.5) {{
            atten = sun_visibility(l, in.world_pos);
        }} else {{
            let dl = l.position.xyz - in.world_pos;
            let dist = length(dl);
            to_l = dl / max(dist, 1e-4);
            let x2 = dist * dist / max(l.params.x * l.params.x, 1e-8);
            let fade = clamp(1.0 - x2 * x2, 0.0, 1.0);
            atten = fade * fade / max(dist * dist, LAMP_RADIUS * LAMP_RADIUS);
            if (kind > 0.5) {{
                atten = atten * smoothstep(l.params.y, l.direction.w, dot(-to_l, l.direction.xyz));
            }}
        }}
        let radiance = l.color_intensity.rgb * l.color_intensity.a * atten;
        glint = glint + radiance * WATER_PI * water_ggx(n, v, to_l, a2);
        sunlight = sunlight + radiance * max(to_l.y, 0.0);
    }}

    // THE VIEW'S WAY THROUGH THE WATER to the bed: REFRACTED at the surface,
    // the way the bed's light really comes -- however low the eye looks, down
    // within 49 degrees of straight down, so it crosses about as much water
    // as the depth there and the colour follows the depth's contours. Along
    // the eye's STRAIGHT line instead (the line the bed behind this pixel was
    // drawn along), a view skimming a floor that falls away stopped meeting it
    // all at once, and the water broke from turquoise to navy in one pixel.
    // Bent by the still surface, not the ripples: theirs would make the
    // colour crawl. Met where a flat bed would have been, then along the line
    // through the gap the floor's fall there leaves.
    let down = normalize(to_world_direction(-v));
    let cos_i = max(-down.y, 0.0);
    let sin_t = sqrt(max(1.0 - cos_i * cos_i, 0.0)) / IOR_WATER;
    let cos_t = sqrt(1.0 - sin_t * sin_t);
    let level = vec2<f32>(down.x, down.z);
    let across = level * (sin_t / max(length(level), 1e-6));
    let refracted = vec3<f32>(across.x, -cos_t, across.y);
    let first = thickness / cos_t;
    var path = first;
    if (has_map) {{
        let closing = thickness - (bed - bed_at(p + refracted * first).w);
        path = select(first * OPEN_PATH, first * thickness / max(closing, 1e-4), closing * OPEN_PATH > thickness);
    }}
    let bed_depth = max(surface - p.y + cos_t * path, 0.0);
    // HOW MUCH OF THE BED IS THE ONE DRAWN BEHIND THIS PIXEL, for the blend to
    // show through, and how much the ground map's picture of it, lit here
    // (`beyond`). The drawn one where the eye's straight line -- walked over
    // the map in lengthening steps -- meets the ground squarely, on the map.
    // At a LIP -- the line skimming a floor that falls away, or meeting the
    // near face of a ridge and rising clear of the floor again past it -- the
    // samples of the pixel just above the line pass over to nothing drawn at
    // all, and the blend showed whatever the eye image was cleared to: a
    // dotted line along the floor's horizon. Near the map's edge the same,
    // and past it nothing is drawn -- the terrain stops a few metres inside
    // the map, so the hand-over starts 6 m in (from 2 m, a dotted line of the
    // clear colour ran across the sea along the edge, 2026-10-07). All of
    // them hand over to the map's picture, which the drawn bed only adds its
    // grain to.
    var drawn = select(0.0, 1.0, !has_map);
    if (on_map) {{
        let flat = thickness / max(cos_i, 0.01);
        var seen = flat * OPEN_PATH;
        var walked = 0.0;
        var gap = thickness;
        var met = false;
        var clears = 0.0;
        var step = WALK_FIRST;
        for (var i = 0; i < WALK_STEPS; i = i + 1) {{
            let t = flat * step;
            let at = p + down * t;
            let next_gap = at.y - bed_at(at).w;
            if (met) {{
                clears = max(clears, next_gap);
            }}
            if (!met && next_gap <= 0.0) {{
                met = true;
                seen = walked + (t - walked) * gap / max(gap - next_gap, 1e-5);
            }}
            walked = t;
            gap = next_gap;
            step = step * WALK_GROWTH;
        }}
        let seen_uv = ground_uv(p + down * seen);
        let from_edge = min(min(seen_uv.x, 1.0 - seen_uv.x) / camera.ground_params.z, min(seen_uv.y, 1.0 - seen_uv.y) / camera.ground_params.w);
        drawn = (1.0 - smoothstep(0.0, 0.5, clears)) * (1.0 - smoothstep(2.0, 4.0, seen / max(flat, 1e-4))) * smoothstep(3.0, 6.0, from_edge);
    }}
    // Each colour of light is lost at its own rate, red first, on the light's
    // way down to the bed and the view's way back. What the water scatters
    // back is its own colour, lit by the sky and the sun.
    let sky_down = sky_irradiance(vec3<f32>(0.0, 1.0, 0.0));
    let k = water.extinction.xyz;
    let t_view = exp(-k * path);
    let t_light = exp(-k * (bed_depth * LIGHT_PATH));
    let body = water.scatter.rgb * (sky_down + sunlight) * (vec3<f32>(1.0) - t_view);

    // CAUSTICS: the waves gather the sun's light on the bed under their
    // troughs and spread it under their crests -- the refracted rays' spread,
    // from the surface's curvature, at this depth. Blurred away in deep water.
    // Only waves long enough to focus at a beach's depths pattern the bed:
    // ripples a finger wide focus within centimetres, and deeper their
    // overlapping folds and the sun's half-degree disc even their light out.
    // Gathered from them too, the bed was a dense crawling speckle in every
    // shallow (headset, 2026-10-07); the third cascade is read two levels
    // coarse (`c2`). And a net finer than a few pixels only crawls: each
    // cascade fades as a pixel spans its cells.
    let fine = 0.5 * (1.0 - smoothstep(0.08, 0.3, span * WAVE_N * 0.25 / water.tiles.z));
    let mid = max(shoal, 0.6) * (1.0 - smoothstep(0.08, 0.3, span * WAVE_N / water.tiles.y));
    let curve = c0.xyz * (0.5 * shoal) + c1.xyz * mid + c2.xyz * fine;
    let focus = thickness * (1.0 - 1.0 / IOR_WATER);
    let jx = 1.0 - focus * curve.x;
    let jz = 1.0 - focus * curve.y;
    let jxz = focus * curve.z;
    // Smoothly capped where the surface focuses it: a hard cap drew flat
    // plateaus with sharp rims where the long waves come to a focus.
    let det = jx * jz - jxz * jxz;
    let gather = inverseSqrt(det * det + 0.0625);
    // Past the waves' first focus the bed is lit by several folds of the
    // surface at once, so it never falls as dark between the bright lines as
    // one fold's spread says: a real net is bright threads over sand lit
    // nearly evenly (`CAUSTIC_CONTRAST`).
    // Murky water scatters the light on its way down and blurs the net away.
    let murk = exp(-(water.extinction.x + water.extinction.y + water.extinction.z) * (1.0 / 3.0) * thickness * CAUSTIC_MURK);
    let caustic = mix(1.0, gather, CAUSTIC_CONTRAST * murk * (1.0 - smoothstep(1.5, 5.0, thickness)));
    let sun_on_bed = BED_ALBEDO * sunlight * t_light;
    let sky_on_bed = BED_ALBEDO * sky_down * t_light;
    let sun_share = dot(sun_on_bed, vec3<f32>(1.0)) / max(dot(sun_on_bed + sky_on_bed, vec3<f32>(1.0)), 1e-4);
    let brighter = max(caustic - 1.0, 0.0) * sun_on_bed * t_view;
    let dimmer = 1.0 + min(caustic - 1.0, 0.0) * sun_share;

    // FOAM: whitecaps where the waves fold (kept as they fade, in the wave
    // field), and on a beach the breakers' white line rolling in, the wash's
    // leading edge up the sand and a lace where the water thins out. A little
    // foam is the brightest bubble walls; a lot fills the cells.
    let caps = max(c0.w, c1.w) * shoal;
    var surf = 0.0;
    if (water.waves.y > 0.0) {{
        let phase = swash_phase(still_depth, q, water.tiles.w);
        // A breaker does not break all along its length at once: patches of
        // it, drifting.
        let patchy = 0.5 + 0.5 * sin(dot(q, vec2<f32>(0.41, 0.17)) + 1.3 * sin(dot(q, vec2<f32>(-0.23, 0.37)) + water.tiles.w * 0.0785398));
        // WHITE WATER where the breaker has grown as high as the depth can
        // hold: its crest spilling first, then, broken, its whole face and a
        // trail of foam left behind it. Clear green face before that.
        let s = phase - round(phase);
        let grown = BREAKER_PER_SWASH * water.waves.y * (1.0 - smoothstep(1.2, 2.8, still_depth));
        let holds = grown / max(BREAK_RATIO * still_depth, 1e-3);
        let spilling = smoothstep(0.75, 1.0, holds) * exp(-(s - 0.008) * (s - 0.008) / (0.012 * 0.012));
        let broken = smoothstep(1.0, 1.4, holds) * smoothstep(-0.02, 0.1, still_depth);
        let face = exp(-(s + 0.012) * (s + 0.012) / (0.03 * 0.03));
        let trail = select(0.0, exp(-s / 0.05), s > 0.0);
        let breaker = max(spilling, broken * max(face, 0.65 * trail)) * mix(0.3, 1.0, patchy);
        let wash = smoothstep(0.0, 0.04, phase) * (1.0 - smoothstep(0.08, 0.4, phase))
            * (1.0 - smoothstep(0.0, 0.25, thickness)) * (1.0 - smoothstep(0.0, 0.6, still_depth));
        surf = max(breaker, wash) + 0.18 * patchy * (1.0 - smoothstep(0.0, 0.2, thickness));
    }}
    let froth = clamp(caps + surf + ring_foam, 0.0, 1.0);
    let cover = clamp((bubbles - (1.0 - froth)) * 4.0, 0.0, 1.0) * smoothstep(0.0, 0.05, froth);
    let foam_light = 0.8 * (sky_irradiance(n) + sunlight);

    // THE SHORELINE fades in over `shore_fade` of depth rather than ending
    // in the line where the surface meets the ground. Foam keeps to the lip.
    let edge = smoothstep(0.0, water.scatter.w, thickness);
    let lip = smoothstep(0.0, 0.015, thickness);
    let shade = (env * fresnel + glint + (body + brighter) * (1.0 - fresnel)) * edge;
    let under = mix(vec3<f32>(1.0), t_view * t_light * dimmer * (1.0 - fresnel), edge);
    let froth_cover = cover * lip;
    let colour = mix(shade, foam_light, froth_cover);
    // The map's picture of the bed where the view meets it, shown as the blend
    // shows the drawn one: its light, as the display has it, through the
    // water.
    var beyond = vec3<f32>(0.0);
    if (drawn < 1.0) {{
        beyond = under * tonemap(bed_at(p + refracted * path).xyz) * ((1.0 - drawn) * (1.0 - froth_cover));
    }}
    let through = under * (drawn * (1.0 - froth_cover));
    {ret}
}}
"#,
        enable = if dual { "enable dual_source_blending;" } else { "" },
        lights_block = wgsl_lights_block(0, 1),
        mips = MIPS,
        wave_n = format!("{:.1}", WAVE_N as f32),
        light_path = format!("{LIGHT_PATH:?}"),
        foam_tiles = format!("{FOAM_TILES:?}"),
        bed_albedo = format!("{BED_ALBEDO:?}"),
        ring_block = ring_block,
        ring_speed = format!("{RING_SPEED:?}"),
        ring_wavelength = format!("{RING_WAVELENGTH:?}"),
        ring_height = format!("{RING_HEIGHT:?}"),
        ring_seconds = format!("{RING_SECONDS:?}"),
        output = output,
        ret = ret,
        surface = surface_wgsl(),
        surface_fn = surface_fn_wgsl("water_surface", "wave_disp"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::lights::{Light, LightKind, LightsUniform};
    use crate::renderer::uniforms::test_support::scene_uniforms;
    use crate::renderer::uniforms::{PlayerUpload, ShadowUpload};
    use crate::renderer::Color3;
    use glam::Vec3;
    use wgpu::util::DeviceExt;

    /// A device, asking for dual-source blending when `dual` -- `None` when
    /// there is no GPU, or it cannot.
    fn gpu(dual: bool) -> Option<(Device, Queue)> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            apply_limit_buckets: false,
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        if dual && !adapter.features().contains(Features::DUAL_SOURCE_BLENDING) {
            return None;
        }
        pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: if dual { Features::DUAL_SOURCE_BLENDING } else { Features::empty() },
            required_limits: crate::renderer::uniforms::scene_limits(Limits::default()),
            ..Default::default()
        }))
        .ok()
    }

    /// One frame of water over a flat-coloured bed.
    #[derive(Clone, Copy)]
    struct Case {
        dual: bool,
        /// Metres of water over the bed.
        depth: f32,
        eye: Vec3,
        /// Where the player stands. The vertices move with them, so the
        /// screen is the same and only the patch of world under it changes.
        walked: Vec3,
        wind: f32,
        /// The colour already in the buffer, 0..1: what lies under the water.
        bed: [f64; 3],
        /// The sun's direction of travel, when there is one.
        sun: Option<Vec3>,
        shallow: [f32; 3],
        deep: [f32; 3],
    }

    fn case(dual: bool) -> Case {
        Case {
            dual,
            depth: 1.0,
            eye: OVERHEAD,
            walked: Vec3::ZERO,
            wind: 0.0,
            bed: [0.0; 3],
            sun: None,
            shallow: [0.30, 0.70, 0.75],
            deep: [0.01, 0.04, 0.07],
        }
    }

    /// Straight down onto the surface.
    const OVERHEAD: Vec3 = Vec3::new(0.0, 8.0, 0.5);
    /// Almost along it, which is where Fresnel takes over.
    const GRAZING: Vec3 = Vec3::new(0.0, 0.05, 40.0);
    /// Thirty degrees down onto it: where a wave's tilt moves the reflectance
    /// most, and so what lies under the water shows its slopes.
    const DOWN_THE_SLOPE: Vec3 = Vec3::new(0.0, 3.0, 5.7);

    /// Draw the water and read the centre pixel.
    fn render(c: &Case) -> Option<[u8; 4]> {
        let image = render_image(c)?;
        Some(image[(SIZE / 2 * SIZE + SIZE / 2) as usize])
    }

    const SIZE: u32 = 8;

    /// Draw the water and read every pixel, row by row.
    ///
    /// view_proj is identity, so a vertex posed into the player's frame IS
    /// its clip position: the triangle covers the screen, at clip depth 0.5,
    /// and its centre pixel is the world point (0, 0, 0.5) plus the walk.
    /// The water is shaded as a still surface at y = 0 over a bed `depth`
    /// below it -- from the vertices, as there is no ground map.
    fn render_image(c: &Case) -> Option<Vec<[u8; 4]>> {
        let (device, queue) = gpu(c.dual)?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let sun = c.sun.map(|d| Light {
            mask_channel: None,
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: true,
            position: Vec3::ZERO,
            direction: d.normalize(),
            kind: LightKind::Directional,
            color: Color3(255, 255, 255, 255),
            intensity: 2.0,
            range: 0.0,
            cone_angle_deg: 0.0,
            inner_cone_angle_deg: 0.0,
        });
        lights.upload(&queue, &sun.into_iter().collect::<Vec<_>>());
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);
        // The eye in the PLAYER's frame, as the headset gives it: it walks
        // with them.
        uniforms.upload_scene(
            &queue,
            glam::Mat4::IDENTITY,
            c.eye,
            &ShadowUpload::disabled(),
            &crate::renderer::uniforms::SkyUpload::none(),
            &crate::renderer::uniforms::PostUpload::default(),
            &PlayerUpload { offset: c.walked, yaw: 0.0, ..Default::default() },
        );

        let pipeline = WaterPipeline::new_with(
            &device,
            format,
            &uniforms.layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            c.dual,
        );
        let params = WaveParams { wind_speed: c.wind, fetch: 2000.0, ..Default::default() };
        let waves = WaveField::new(&device, &queue, params);
        let mut u = WaterUniform::new(
            &WaterOptics { height: 0.0, shallow: c.shallow, deep: c.deep, depth_scale: 1.0, shore_fade: 0.25, swash: 0.0, swell: 0.0 },
            &params,
        );
        u.waves[2] = 0.0; // the surface stays put: the pixel tests read the shading
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water_test_uniform"),
            contents: bytemuck::bytes_of(&u),
            usage: BufferUsages::UNIFORM,
        });
        let groups = pipeline.bind_groups(&device, &ubuf, &waves);

        let o = c.walked;
        let v = |x: f32, y: f32| WaterVertex { position: [x + o.x, y + o.y, 0.5 + o.z], depth: c.depth };
        let verts = [v(-1.0, -1.0), v(3.0, -1.0), v(-1.0, 3.0)];
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water_test_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });

        let target = device.create_texture(&TextureDescriptor {
            label: Some("water_test_target"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_tex = device.create_texture(&TextureDescriptor {
            label: Some("water_test_depth"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let tv = target.create_view(&Default::default());
        let dv = depth_tex.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("water_test_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        waves.update(&queue, &mut encoder, 12.0);
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("water_test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &tv,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color { r: c.bed[0], g: c.bed[1], b: c.bed[2], a: 1.0 }),
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
            pass.set_bind_group(1, &groups[waves.current()], &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.draw(0..3, 0..1);
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
        Some(
            (0..SIZE * SIZE)
                .map(|i| {
                    let o = (i / SIZE * 256 + i % SIZE * 4) as usize;
                    [data[o], data[o + 1], data[o + 2], data[o + 3]]
                })
                .collect(),
        )
    }

    /// How much of a white bed shows through, per channel: the pixel over
    /// white less the pixel over black, so the water's own light cancels.
    fn shows_through(c: Case) -> Option<[i32; 3]> {
        let white = render(&Case { bed: [1.0; 3], ..c })?;
        let black = render(&Case { bed: [0.0; 3], ..c })?;
        Some([0, 1, 2].map(|i| white[i] as i32 - black[i] as i32))
    }

    fn sum(c: [i32; 3]) -> i32 {
        c[0] + c[1] + c[2]
    }

    /// Both blend paths, where the GPU has them.
    fn each_path(test: impl Fn(bool)) {
        for dual in [false, true] {
            if gpu(dual).is_none() {
                eprintln!("skipping the {} path: no GPU for it", if dual { "dual-source" } else { "one-alpha" });
                continue;
            }
            test(dual);
        }
    }

    #[test]
    fn the_water_shader_compiles_and_draws() {
        // WGSL is validated at PIPELINE CREATION, not at cargo build: a bad
        // field accessor compiles clean and dies on the device.
        each_path(|dual| {
            let px = render(&Case { eye: GRAZING, ..case(dual) }).unwrap();
            assert!(px[0] as u32 + px[1] as u32 + px[2] as u32 > 0, "water drew nothing over a black bed ({dual}): {px:?}");
        });
    }

    #[test]
    fn deep_water_hides_the_bed() {
        // The bed's light is lost on its way through: at 20 cm the sand is
        // plain, at 8 m it is gone into the water's own colour.
        each_path(|dual| {
            let shallow = shows_through(Case { depth: 0.2, ..case(dual) }).unwrap();
            let deep = shows_through(Case { depth: 8.0, ..case(dual) }).unwrap();
            assert!(sum(shallow) > 3 * sum(deep) + 30, "deep water must hide the bed ({dual}): {deep:?} vs {shallow:?}");
        });
    }

    #[test]
    fn the_bed_takes_the_waters_colour_channel_by_channel() {
        // What one alpha cannot do: red is lost first, so white sand under a
        // metre of sea water goes blue-green rather than grey.
        let Some(through) = gpu(true).and_then(|_| shows_through(case(true))) else {
            eprintln!("skipping: no GPU with dual-source blending");
            return;
        };
        assert!(
            through[0] + 20 < through[1] && through[0] + 20 < through[2],
            "the bed must lose its red before its green and blue: {through:?}",
        );
    }

    #[test]
    fn a_grazing_view_shows_less_of_the_bed() {
        // Fresnel: clear at your feet, a mirror at the far bank.
        each_path(|dual| {
            let down = shows_through(Case { depth: 0.3, ..case(dual) }).unwrap();
            let across = shows_through(Case { depth: 0.3, eye: GRAZING, ..case(dual) }).unwrap();
            assert!(sum(down) > sum(across) + 60, "a grazing view must mirror rather than show the bed ({dual}): {across:?} vs {down:?}");
        });
    }

    #[test]
    fn the_shoreline_fades_out_rather_than_ending_in_a_line() {
        // Water millimetres deep leaves the bed as it was: otherwise it meets
        // the ground along a hard edge that flickers, being a geometric
        // intersection sampled per pixel.
        each_path(|dual| {
            let brink = render(&Case { depth: 0.001, bed: [0.6; 3], ..case(dual) }).unwrap();
            let body = render(&Case { depth: 2.0, bed: [0.6; 3], ..case(dual) }).unwrap();
            let off = |px: [u8; 4]| (0..3).map(|i| (px[i] as i32 - 153).abs()).max().unwrap();
            assert!(off(brink) <= 3, "the shoreline must leave the bed untouched ({dual}): {brink:?}");
            assert!(off(body) > 20, "two metres of water must change what is under it ({dual}): {body:?}");
        });
    }

    #[test]
    fn the_waves_belong_to_the_world_not_the_player() {
        // The level is drawn in the PLAYER's frame. Waves read from that frame
        // would follow the player about. Here the player and the vertices walk
        // together, so the screen is unchanged and only the patch of world
        // under the pixel differs: world waves must change it, player-frame
        // ones could not.
        each_path(|dual| {
            let c = Case { wind: 9.0, eye: DOWN_THE_SLOPE, bed: [1.0; 3], ..case(dual) };
            let here = render_image(&c).unwrap();
            let walked = render_image(&Case { walked: Vec3::new(7.3, 0.0, 3.1), ..c }).unwrap();
            assert_ne!(here, walked, "the waves did not change with the world under the pixels ({dual})");
        });
    }

    #[test]
    fn waves_change_the_surface() {
        // The wave field has to reach the shading: with it dropped this still
        // draws, and reads as ice.
        each_path(|dual| {
            let c = Case { eye: DOWN_THE_SLOPE, bed: [1.0; 3], ..case(dual) };
            let calm = render_image(&c).unwrap();
            let windy = render_image(&Case { wind: 9.0, ..c }).unwrap();
            assert_ne!(calm, windy, "a 9 m/s wind changed nothing ({dual})");
        });
    }

    #[test]
    fn the_sun_glints_where_the_water_mirrors_it() {
        // Looking straight down at still water, a sun overhead is mirrored
        // into the eye; one low in the side is not. Deep, dark water, so the
        // glint is what differs.
        each_path(|dual| {
            let c = Case { depth: 20.0, ..case(dual) };
            let mirrored = render(&Case { sun: Some(Vec3::NEG_Y), ..c }).unwrap();
            let aside = render(&Case { sun: Some(Vec3::new(1.0, -0.3, 0.0)), ..c }).unwrap();
            let lum = |p: [u8; 4]| p[0] as u32 + p[1] as u32 + p[2] as u32;
            assert!(lum(mirrored) > lum(aside) + 60, "no glint from a mirrored sun ({dual}): {mirrored:?} vs {aside:?}");
        });
    }

    /// The waterline twin is the water's own shader with one test added
    /// before it returns: the surface drawn above the line is the surface
    /// drawn everywhere else, and the ordinary pipelines carry none of it.
    #[test]
    fn every_water_cut_matches_and_validates() {
        for (cut, edits) in WATER_CUTS {
            for dual in [false, true] {
                let mut src = water_wgsl_with(dual, false);
                for (from, to) in edits.iter() {
                    assert_eq!(src.matches(from).count(), 1, "water cut {cut}: {from}");
                    src = src.replacen(from, to, 1);
                }
                let module = wgpu::naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{cut}: {}", e.emit_to_string(&src)));
                wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
                    .validate(&module)
                    .unwrap_or_else(|e| panic!("{cut}: {}", e.emit_to_string(&src)));
            }
        }
    }

    #[test]
    fn the_waterline_twin_is_the_water_plus_its_test() {
        for dual in [true, false] {
            let plain = water_wgsl_with(dual, true);
            let line = water_line_wgsl(dual);
            assert!(!plain.contains("water_line_under"));
            let without = line
                .replacen("    if (water_line_under(in.world_pos)) {\n        discard;\n    }\n", "", 1)
                .replacen(WATER_LINE_RISING, WATER_LINE_VIEW, 1);
            let start = without.find("\n// Whether the view through this pixel starts under").unwrap();
            let end = without.find("@fragment fn fs_main").unwrap();
            assert_eq!(format!("{}{}", &without[..start], &without[end..]), plain);
        }
    }

    #[test]
    fn white_seen_through_depth_scale_comes_out_shallow() {
        // The authored colour is what a white bed looks like straight down
        // through `depth_scale` metres: down and back up, as the shader takes
        // the light's two paths.
        let o = WaterOptics { height: 0.0, shallow: [0.3, 0.7, 0.75], deep: [0.0; 3], depth_scale: 2.0, shore_fade: 0.3, swash: 0.0, swell: 0.0 };
        let u = WaterUniform::new(&o, &WaveParams::default());
        for i in 0..3 {
            let seen = (-u.extinction[i] * o.depth_scale * (1.0 + LIGHT_PATH)).exp();
            assert!((seen - o.shallow[i]).abs() < 1e-5, "channel {i}: {seen} vs {}", o.shallow[i]);
        }
    }
}

// Scene shader sources, for `multiview::every_scene_shader_survives_the_multiview_transform`.
// Test-only: the gate has to see exactly the text each pipeline is built from,
// and nothing on a development machine can build a multiview pipeline to check.
#[cfg(test)]
pub fn water_shader_src() -> String {
    water_wgsl(false)
}

#[cfg(test)]
mod seen_tests {
    use super::*;
    use crate::renderer::shadow::frustum_planes;
    use glam::{Mat4, Vec3};

    /// An eye at the origin looking down -z, 90 degrees wide, out to 100 m.
    fn view() -> [glam::Vec4; 6] {
        frustum_planes(
            Mat4::perspective_rh_gl(90f32.to_radians(), 1.0, 0.1, 100.0)
                * Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y),
        )
    }

    fn pond(centre: Vec3) -> (Vec3, Vec3) {
        (centre - Vec3::new(5.0, 0.5, 5.0), centre + Vec3::new(5.0, 0.5, 5.0))
    }

    #[test]
    fn water_ahead_is_seen_and_water_behind_or_aside_is_not() {
        let v = [view()];
        assert!(water_seen(pond(Vec3::new(0.0, -1.0, -20.0)), &v, None), "ahead");
        assert!(!water_seen(pond(Vec3::new(0.0, -1.0, 20.0)), &v, None), "behind");
        assert!(!water_seen(pond(Vec3::new(60.0, -1.0, -20.0)), &v, None), "far to the side");
    }

    #[test]
    fn water_past_the_far_plane_is_still_seen() {
        // A lake two kilometres out, held inside the far plane as the sea's
        // skirt is.
        assert!(water_seen(pond(Vec3::new(0.0, -1.0, -2000.0)), &[view()], None));
    }

    #[test]
    fn either_eye_is_enough() {
        let other = frustum_planes(
            Mat4::perspective_rh_gl(90f32.to_radians(), 1.0, 0.1, 100.0)
                * Mat4::look_at_rh(Vec3::ZERO, Vec3::Z, Vec3::Y),
        );
        assert!(water_seen(pond(Vec3::new(0.0, -1.0, 20.0)), &[view(), other], None));
    }

    #[test]
    fn from_a_closed_room_only_a_doorway_shows_it() {
        let v = [view()];
        let ahead = pond(Vec3::new(0.0, -1.0, -20.0));
        assert!(!water_seen(ahead, &v, Some(&[])), "no doorway in view: hidden by the walls");
        assert!(water_seen(ahead, &v, Some(&[view()])), "a doorway looking at it");
        assert!(water_seen(ahead, &v, None), "not in a closed room: no culling");
    }

    #[test]
    fn the_bounds_reach_as_far_as_the_waves_can() {
        let verts = [
            WaterVertex { position: [0.0, 1.0, 0.0], depth: 1.0 },
            WaterVertex { position: [10.0, 1.0, 4.0], depth: 2.0 },
        ];
        let calm = WaveParams { wind_speed: 1.0, fetch: 10.0, ..WaveParams::default() };
        let storm = WaveParams { wind_speed: 20.0, fetch: 50_000.0, ..WaveParams::default() };
        let (lo, hi) = water_bounds(&verts, &calm);
        assert!(lo.x <= -WAVE_ALLOWANCE && hi.z >= 4.0 + WAVE_ALLOWANCE && lo.y < 1.0 && hi.y > 1.0);
        let (slo, shi) = water_bounds(&verts, &storm);
        assert!(shi.y - slo.y > hi.y - lo.y + 1.0, "a storm's waves reach further: {slo} {shi} vs {lo} {hi}");
    }
}
