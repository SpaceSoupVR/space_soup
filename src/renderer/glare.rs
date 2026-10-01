//! GLARE: WHAT A LAMP'S LIGHT DOES IN THE EYE, drawn where an HDR framebuffer
//! would have bloomed it.
//!
//! The renderer's lighting is high dynamic range -- a lamp's bulb is drawn at
//! the radiance of the light it gives off, hundreds of times white
//! (`space_soup_engine::scene_light::emissive_drive`) -- but it tone maps in
//! the forward pass and keeps no HDR image, so nothing downstream knows that a
//! white bulb is brighter than a white wall. An eye does: light scattered in
//! its lens and fluid veils the view round a bright source, and that veil is
//! what reads as "that is a light". Bloom on desktop blurs an HDR buffer to
//! get it; here the sources are known, so each draws its own veil as one
//! camera-facing quad at the end of the scene pass -- a few hundred thousand
//! cheap fragments where a full-screen bloom chain would be several passes over
//! the whole eye (user, 2026-09-30: HDR "simulated in whatever efficient ways
//! we can", a bloom pass "as long as it won't cost significant performance").
//!
//! HOW BRIGHT: the CIE's general disability glare equation (CIE 146:2002,
//! Vos and van den Berg) for a young eye -- the veil `theta` degrees from a
//! source is `E (10/theta^3 + 5/theta^2 + 0.1 p/theta)(1 + (A/62.5)^4)` cd/m^2
//! for an illuminance `E` in lux at the eye, age `A` 25 and pigmentation `p`
//! 0.5 -- times [`VEIL_SHARE`]. It replaced Stiles and Holladay's older
//! `10 E / theta^2`, whose long flat tail drew a grey disc round every lamp
//! (headset eye capture, 2026-09-30): most of a young eye's veil is within a
//! few degrees of the source. `E` is the light the lamp throws at the eye,
//! `intensity x visible share / d^2`, in the renderer's units, where a white
//! card at the eye would be lit to exactly `E`; so the veil is `pi` times the
//! bracket of that card, exposed with the frame and shown through the SAME tone
//! curve as the scene it lies over (`tonemap::aces_fitted`) -- a curve of its
//! own lifted the faint outer veil threefold over light the scene would show
//! near black. Near the source the veil stops growing at the lamp's own
//! angular size ([`LAMP_RADIUS`]); at the quad's edge it is exactly zero, the
//! veil there taken off all of it, so no quad ever shows its outline.
//!
//! HOW MUCH OF THE BULB SHOWS: `GlareSource::sides`, per side of the fixture
//! -- measured from its reflection cards where it has them (a sconce's bulb
//! shows from below, not from above or through its shade), the author's
//! "glare visible from" faces where it does not -- and a spot's cone.
//!
//! IN THE EYE, NOT IN THE ROOM. The veil is light scattered inside the eye, so
//! it lies over everything in view -- a door frame or a pillar nearer than the
//! lamp included -- and goes only when the BULB is hidden. So the quads are
//! not depth-tested pixel by pixel, which cut the veil along every nearer
//! outline and let it spill round a corner whose far side hides the bulb.
//! Instead each quad's vertices look at the bulb itself, in this frame's
//! half-resolution probe pass depth (every brush, no models: a fixture's own
//! shade is `sides`' business), and the veil takes the share of taps across
//! the bulb that no wall stands in front of.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use wgpu::*;

/// The lamp's radius as the lighting clamps its inverse square: the lights
/// block's `LAMP_RADIUS`, pinned to it by a test here.
pub const LAMP_RADIUS: f32 = 0.05;

/// The share of the CIE young eye's veil drawn: a bright bloom within a couple
/// of degrees of a bare bulb, a halo fading by fifteen, the room round it left
/// its own. The lever `glare_strength` scales it on the headset.
pub const VEIL_SHARE: f32 = 0.25;

/// The CIE bracket's terms for age 25 and pigmentation 0.5 (brown eyes; 1.0 is
/// blue): `10/theta^3 + CIE_SQUARE/theta^2 + CIE_LINEAR/theta`. Its constant
/// term, straylight spread evenly over the whole view, is left out.
const CIE_CUBE: f32 = 10.0;
const CIE_AGE: f32 = 1.0 + 0.4 * 0.4 * 0.4 * 0.4;
const CIE_SQUARE: f32 = 5.0 * CIE_AGE;
const CIE_LINEAR: f32 = 0.1 * 0.5 * CIE_AGE;

/// The faintest veil drawn, in exposed units -- a fiftieth of white, which the
/// tone curve's toe shows a thousandth as bright: where a quad ends.
const VEIL_FLOOR: f32 = 0.02;

/// The widest veil drawn, in degrees from its source: a quad's size, and so its
/// cost, is capped here.
pub const MAX_GLARE_DEGREES: f32 = 25.0;

/// The narrowest core, in degrees: a lamp far off is still a point to the eye.
const MIN_CORE_DEGREES: f32 = 0.5;

/// How far in front of its bulb a wall must stand to hide it: a brush within a
/// lamp's own radius of its bulb is what the lamp is mounted on, and depth at
/// half resolution is no finer than that.
const WALL_MARGIN: f32 = LAMP_RADIUS;

/// One source of glare this frame, in the player's frame (as lights are
/// uploaded).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlareSource {
    /// The bulb.
    pub position: Vec3,
    /// What it gives off: the light's linear colour times its intensity.
    pub radiance: Vec3,
    /// How much of the bright source shows from each side of the fixture, in
    /// its own frame -- +x, -x, +y, -y, +z, -z -- as a share of a bare lamp's
    /// light: 1 a bulb in plain view, more where a shade's lit inside adds to
    /// it.
    pub sides: [f32; 6],
    /// The fixture's frame: `sides` are along its axes.
    pub rotation: Quat,
    /// A spot's beam -- its axis and the cosines of its outer and inner half
    /// angles -- whose bulb shows only from where its light goes. `None` for a
    /// lamp that shows from every side `sides` allows.
    pub cone: Option<(Vec3, f32, f32)>,
    /// WHERE the light shows from each side, in the same order as `sides` --
    /// the middle of what a side's card saw glowing: a sconce's open mouth from
    /// below, not its bulb up inside the shade. `None` puts every side's at
    /// `position`. See `visible_centre`.
    pub centres: Option<[Vec3; 6]>,
}

/// Where the light of `s` shows from `eye`: each side's centre weighed as
/// `visible_share` weighs its share. The veil grows from here -- from the
/// bulb itself, a sconce seen from below glowed on its dark shade, above the
/// mouth the light actually leaves by (headset, 2026-09-30).
pub fn visible_centre(s: &GlareSource, eye: Vec3) -> Vec3 {
    let Some(centres) = s.centres else { return s.position };
    let Some(to_eye) = (eye - s.position).try_normalize() else { return s.position };
    let local = s.rotation.inverse() * to_eye;
    let (mut sum, mut weight) = (Vec3::ZERO, 0.0f32);
    for a in 0..3 {
        let c = local[a];
        let k = if c >= 0.0 { 2 * a } else { 2 * a + 1 };
        let w = c * c * s.sides[k].max(0.0);
        sum += centres[k] * w;
        weight += w;
    }
    if weight > 1e-6 {
        sum / weight
    } else {
        s.position
    }
}

/// How much of `s` shows toward an eye at `eye`, as a share of a bare lamp's
/// light: its sides weighed by how squarely the eye lies along each axis of the
/// fixture, and a spot's cone.
pub fn visible_share(s: &GlareSource, eye: Vec3) -> f32 {
    let Some(to_eye) = (eye - s.position).try_normalize() else { return 0.0 };
    let local = s.rotation.inverse() * to_eye;
    let mut share = 0.0;
    for a in 0..3 {
        let c = local[a];
        share += c * c * s.sides[if c >= 0.0 { 2 * a } else { 2 * a + 1 }];
    }
    if let Some((axis, cos_outer, cos_inner)) = s.cone {
        let c = axis.normalize_or_zero().dot(to_eye);
        share *= smoothstep(cos_outer, cos_inner.max(cos_outer + 1e-3), c);
    }
    share.max(0.0)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One corner of a glare quad.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GlareVertex {
    pub position: [f32; 3],
    /// The source's colour, its brightest channel 1.
    pub colour: [f32; 3],
    /// -1..1 across the quad; its length is the angle from the source over the
    /// quad's.
    pub uv: [f32; 2],
    /// x: the veil's scale, exposed (see [`GlareQuad::a`]); y: the core's
    /// angle, squared; z: the quad's reach, squared; w: the veil at the reach,
    /// taken off it all. Angles in degrees. See the shader.
    pub shape: [f32; 4],
    /// xyz: the quad's centre -- the bulb, pulled toward the eye by
    /// [`WALL_MARGIN`] -- where the walls are tested; w: the bulb's radius
    /// over the quad's half size, which sizes the taps, or -1 for no test
    /// (no probe pass this frame to test against).
    pub test: [f32; 4],
}

impl GlareVertex {
    pub const ATTRIBS: [VertexAttribute; 5] =
        vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4, 4 => Float32x4];

    pub fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// How far outside a capsule an eye must be for the capsule to count as
/// standing between it and a lamp: the eyes are inside their own head's.
const EYE_CLEARANCE: f32 = 0.02;

/// How much of `bulb` the characters leave in view of an eye at `eye`, 0..1:
/// none behind a capsule -- a hand raised against the light -- softened over
/// the bulb's own width where the capsule crosses the line of sight. Their
/// meshes are not in the probe pass, whose depth finds the walls. A capsule
/// the eye is inside hides nothing.
pub fn capsule_visibility(bulb: Vec3, eye: Vec3, capsules: &[(Vec3, Vec3, f32)]) -> f32 {
    let mut open = 1.0;
    for &(a, b, radius) in capsules {
        let (_, eye_gap) = closest_on_segment(eye, a, b);
        if eye_gap < radius + EYE_CLEARANCE {
            continue;
        }
        let (t, gap) = segment_gap(eye, bulb, a, b);
        // The bulb's disc, as wide where the line of sight passes the capsule
        // as it looks from the eye.
        let soft = (LAMP_RADIUS * t).max(1e-4);
        open *= smoothstep(radius - soft, radius + soft, gap);
    }
    open
}

/// The point of segment `a..b` nearest `p`, as its parameter, and how far.
fn closest_on_segment(p: Vec3, a: Vec3, b: Vec3) -> (f32, f32) {
    let ab = b - a;
    let t = if ab.length_squared() > 0.0 { ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
    (t, (a + ab * t - p).length())
}

/// How near segment `p..q` passes segment `a..b`: the parameter along `p..q`
/// of its nearest point, and the distance. Ericson, "Real-Time Collision
/// Detection", 5.1.9.
fn segment_gap(p: Vec3, q: Vec3, a: Vec3, b: Vec3) -> (f32, f32) {
    let (d1, d2, r) = (q - p, b - a, p - a);
    let (aa, ee, f) = (d1.length_squared(), d2.length_squared(), d2.dot(r));
    let (s, t) = if aa <= 1e-12 {
        (0.0, if ee > 1e-12 { (f / ee).clamp(0.0, 1.0) } else { 0.0 })
    } else {
        let c = d1.dot(r);
        if ee <= 1e-12 {
            ((-c / aa).clamp(0.0, 1.0), 0.0)
        } else {
            let bb = d1.dot(d2);
            let denom = aa * ee - bb * bb;
            let mut s = if denom > 1e-12 { ((bb * f - c * ee) / denom).clamp(0.0, 1.0) } else { 0.0 };
            let mut t = (bb * s + f) / ee;
            if t < 0.0 {
                t = 0.0;
                s = (-c / aa).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((bb - c) / aa).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };
    (s, ((p + d1 * s) - (a + d2 * t)).length())
}

/// The quads this frame, seen from the two `eyes` with the view's `right` and
/// `up` axes, at `exposure`, the veil scaled by `strength` (the lever). A
/// source whose veil would not reach [`VEIL_FLOOR`] anywhere draws nothing.
/// `test_walls`: whether this frame's probe pass depth is there to hide a
/// bulb behind a wall (see the module notes); without it every veil shows.
/// `capsules`: the characters, whose hands can shield an eye from a lamp --
/// each eye tested, the veil the share the two see.
pub fn build_glare(
    sources: &[GlareSource],
    eyes: [Vec3; 2],
    right: Vec3,
    up: Vec3,
    exposure: f32,
    strength: f32,
    test_walls: bool,
    capsules: &[(Vec3, Vec3, f32)],
) -> (Vec<GlareVertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut idx = Vec::new();
    let eye = 0.5 * (eyes[0] + eyes[1]);
    for s in sources {
        // From where the light SHOWS, not where the bulb hangs: see
        // `visible_centre`.
        let at = visible_centre(s, eye);
        let shielded =
            0.5 * (capsule_visibility(at, eyes[0], capsules) + capsule_visibility(at, eyes[1], capsules));
        let Some(q) = glare_quad(s, eye, exposure, strength * shielded) else { continue };
        // Where the walls are tested, and the quad's centre: in front of the
        // light by the wall margin, never past half way to the eye. The quad
        // is sized for the angle it subtends from there.
        let to_eye = eye - at;
        let d = to_eye.length();
        let centre = at + to_eye.normalize_or_zero() * WALL_MARGIN.min(0.5 * d);
        let half = q.half * (eye - centre).length() / d.max(1e-6);
        let test = if test_walls { LAMP_RADIUS / half.max(1e-6) } else { -1.0 };
        let base = verts.len() as u32;
        for (du, dv) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            verts.push(GlareVertex {
                position: (centre + right * (du * half) + up * (dv * half)).to_array(),
                colour: q.colour.to_array(),
                uv: [du, dv],
                shape: [q.a, q.core2, q.degrees * q.degrees, q.edge],
                test: [centre.x, centre.y, centre.z, test],
            });
        }
        idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, idx)
}

/// The CIE bracket at `theta2` -- the angle from the source squared, plus the
/// core's, in degrees squared: the veil a unit of light at the eye puts there,
/// over the light itself. See the module notes.
pub fn cie_veil(theta2: f32) -> f32 {
    let inv = theta2.max(1e-6).sqrt().recip();
    inv * (CIE_LINEAR + inv * (CIE_SQUARE + inv * CIE_CUBE))
}

/// One source's quad: its half size in metres, colour, its veil's scale and
/// the core's size, how far it reaches and the veil there -- or `None` for a
/// source too faint or hidden to glare.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlareQuad {
    pub half: f32,
    pub colour: Vec3,
    /// The veil at `theta` is `a x cie_veil(theta^2 + core2) - edge`, in
    /// exposed units: `a` is the light at the eye, exposed, times
    /// `VEIL_SHARE x pi` and the lever.
    pub a: f32,
    /// The core's angle, squared, in degrees squared.
    pub core2: f32,
    /// The quad's reach from its source, in degrees.
    pub degrees: f32,
    /// The veil at the reach, taken off all of it so it ends at zero there.
    pub edge: f32,
    /// The veil at the source, after that.
    pub peak: f32,
}

pub fn glare_quad(s: &GlareSource, eye: Vec3, exposure: f32, strength: f32) -> Option<GlareQuad> {
    let d = (eye - s.position).length();
    let luminance = s.radiance.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    let share = visible_share(s, eye);
    if !(d > 0.0) || luminance <= 0.0 || share <= 0.0 || strength <= 0.0 {
        return None;
    }
    // The light at the eye, as a white card there would be lit.
    let e = luminance * share / (d * d).max(LAMP_RADIUS * LAMP_RADIUS);
    let a = exposure * strength * VEIL_SHARE * std::f32::consts::PI * e;
    let core = (LAMP_RADIUS / d).atan().to_degrees().max(MIN_CORE_DEGREES);
    let core2 = core * core;
    let veil = |theta: f32| a * cie_veil(theta * theta + core2);
    if veil(0.0) < VEIL_FLOOR {
        return None;
    }
    // Out to where the veil falls to the floor, within the cap: the bracket
    // falls all the way, so halving the interval finds it.
    let degrees = if veil(MAX_GLARE_DEGREES) >= VEIL_FLOOR {
        MAX_GLARE_DEGREES
    } else {
        let (mut lo, mut hi) = (0.0, MAX_GLARE_DEGREES);
        for _ in 0..20 {
            let mid = 0.5 * (lo + hi);
            if veil(mid) >= VEIL_FLOOR {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi.max(2.0 * core).min(MAX_GLARE_DEGREES)
    };
    let edge = veil(degrees);
    let colour = s.radiance / s.radiance.max_element().max(1e-6);
    Some(GlareQuad {
        half: d * degrees.to_radians().tan(),
        colour,
        a,
        core2,
        degrees,
        edge,
        peak: veil(0.0) - edge,
    })
}

/// The glare's pipeline: additive, over everything the scene pass drew (no
/// depth test: see the module notes), drawn last in it. Group 1 is the probe
/// pass's (`brush_pipeline::probe_pass::bind_group_layout`), for its depth.
/// Mono and stereo twins, like every scene-pass pipeline. See `multiview`.
pub struct GlarePipeline {
    pub pipeline: RenderPipeline,
}

impl GlarePipeline {
    pub fn new_multisampled(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        probe_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, probe_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    pub fn new_multisampled_stereo(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        probe_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, probe_layout, samples, crate::renderer::multiview::ViewMode::Stereo)
    }

    fn new_with_view(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        probe_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("glare_shader"),
            source: ShaderSource::Wgsl(view.shader(glare_shader()).into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("glare_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(probe_layout)],
            immediate_size: 0,
        });
        let add = BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::One, operation: BlendOperation::Add };
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("glare_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(GlareVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState { color: add, alpha: BlendComponent::OVER }),
                    write_mask: ColorWrites::COLOR,
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
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Always),
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

/// The veil at `r` (the angle from the source over the quad's): the CIE
/// bracket at that angle, the edge's value off it so it ends at zero, shown
/// through the scene's own tone curve. Scaled by how much of the bulb no wall
/// hides, which every vertex of a quad works out alike from the probe pass's
/// depth: seven taps across the bulb's disc, each hidden where a brush stands
/// nearer than the quad's centre. This eye's camera and depth layer by
/// `view_slot`, which a stereo pass sets per view.
pub fn glare_shader() -> String {
    format!(
        "{}{}{}",
        crate::renderer::tonemap::wgsl_aces_block(),
        format!(
            "const CIE_CUBE: f32 = {CIE_CUBE:?};\nconst CIE_SQUARE: f32 = {CIE_SQUARE:?};\nconst CIE_LINEAR: f32 = {CIE_LINEAR:?};\n"
        ),
        r#"
var<private> view_slot: i32 = 0;
struct Camera { view_proj: array<mat4x4<f32>, 2> }
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(1) var probe_pass_depth: texture_depth_2d_array;
struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) colour: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) shape: vec4<f32>,
    @location(4) test: vec4<f32>,
}
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) colour: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) shape: vec4<f32>,
}
// The share of the bulb no wall hides: `centre` is the quad's centre, in
// front of the bulb by the wall margin; `corner` this vertex, whose distance
// from it on screen sizes the bulb's disc by `ratio`.
fn glare_bulb_visible(centre: vec3<f32>, corner: vec4<f32>, ratio: f32) -> f32 {
    let c = camera.view_proj[view_slot] * vec4<f32>(centre, 1.0);
    if (ratio < 0.0 || c.w <= 0.0 || corner.w <= 0.0) {
        return 1.0;
    }
    let ndc = c.xyz / c.w;
    let radius = length(corner.xy / corner.w - ndc.xy) * 0.70710678 * ratio;
    let size = vec2<f32>(textureDimensions(probe_pass_depth));
    let hi = vec2<i32>(size) - vec2<i32>(1);
    var open = 0.0;
    for (var i = 0; i < 7; i = i + 1) {
        var offset = vec2<f32>(0.0);
        if (i > 0) {
            let a = f32(i) * 1.0471976;
            offset = vec2<f32>(cos(a), sin(a)) * radius * 0.66;
        }
        let p = ndc.xy + offset;
        let texel = clamp(vec2<i32>(vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5) * size), vec2<i32>(0), hi);
        let wall = textureLoad(probe_pass_depth, texel, view_slot, 0);
        open = open + select(0.0, 1.0, wall >= ndc.z);
    }
    return open / 7.0;
}
@vertex fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    out.clip = camera.view_proj[view_slot] * vec4<f32>(v.pos, 1.0);
    out.colour = v.colour;
    out.uv = v.uv;
    // The share of the bulb no wall hides scales the veil and its edge alike,
    // so it still ends at zero.
    let open = glare_bulb_visible(v.test.xyz, out.clip, v.test.w);
    out.shape = vec4<f32>(v.shape.x * open, v.shape.y, v.shape.z, v.shape.w * open);
    return out;
}
@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let theta2 = dot(in.uv, in.uv) * in.shape.z + in.shape.y;
    let inv = inverseSqrt(max(theta2, 1e-6));
    let veil = max(in.shape.x * inv * (CIE_LINEAR + inv * (CIE_SQUARE + inv * CIE_CUBE)) - in.shape.w, 0.0);
    return vec4<f32>(aces_fitted(in.colour * veil), 0.0);
}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sconce(sides: [f32; 6]) -> GlareSource {
        GlareSource {
            position: Vec3::new(0.0, 2.0, 0.0),
            radiance: Vec3::new(1.0, 0.9, 0.8) * 3.0,
            sides,
            rotation: Quat::IDENTITY,
            cone: None,
            centres: None,
        }
    }

    /// A SCONCE'S LIGHT SHOWS FROM ITS MOUTH: seen from below, the veil grows
    /// from the middle of what the bottom card saw glowing, not from the bulb
    /// up in the shade; seen square from a side that shows nothing, nothing
    /// pulls it, and it stays at the bulb; between two sides that both show,
    /// it lies between their centres by how squarely each faces the eye.
    #[test]
    fn a_veil_grows_from_where_the_light_shows() {
        let bulb = Vec3::new(0.0, 2.0, 0.0);
        let mouth = Vec3::new(0.0, 1.85, 0.0);
        let mut centres = [bulb; 6];
        centres[3] = mouth; // -y: from below
        let s = GlareSource { centres: Some(centres), ..sconce([0.0, 0.0, 0.0, 0.5, 0.0, 0.0]) };
        assert!((visible_centre(&s, Vec3::new(0.0, 0.0, 0.0)) - mouth).length() < 1e-5);
        assert!((visible_centre(&s, Vec3::new(3.0, 2.0, 0.0)) - bulb).length() < 1e-5, "a dark side pulls nothing");
        let mut both = centres;
        both[0] = Vec3::new(0.1, 2.0, 0.0); // +x shows its light a little out
        let s2 = GlareSource { centres: Some(both), ..sconce([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) };
        let c = visible_centre(&s2, bulb + Vec3::new(1.0, -1.0, 0.0));
        assert!((c - (both[0] + mouth) * 0.5).length() < 1e-5, "{c}");
        // No centres: the bulb, as before.
        assert_eq!(visible_centre(&sconce([1.0; 6]), Vec3::ZERO), bulb);
    }

    /// The lamp radius is the lighting's: the veil's core and the light at the
    /// eye stop growing where the lighting stops growing.
    #[test]
    fn the_lamp_radius_is_the_lightings() {
        let wgsl = include_str!("lights.rs");
        assert!(wgsl.contains(&format!("const LAMP_RADIUS: f32 = {LAMP_RADIUS:?};")), "lights block's LAMP_RADIUS moved");
    }

    /// A bulb that shows only from below glares from below and not from above;
    /// a fixture turned upside down the other way round.
    #[test]
    fn a_bulb_glares_only_from_the_sides_it_shows_from() {
        let below_only = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let s = sconce(below_only);
        assert!(visible_share(&s, Vec3::new(0.0, 0.0, 0.0)) > 0.99, "straight below");
        assert!(visible_share(&s, Vec3::new(0.0, 4.0, 0.0)) < 0.01, "straight above");
        let slant = visible_share(&s, Vec3::new(1.0, 1.0, 0.0));
        assert!(slant > 0.4 && slant < 0.6, "45 degrees below: half the axis weight: {slant}");
        let upside = GlareSource { rotation: Quat::from_rotation_x(std::f32::consts::PI), ..s };
        assert!(visible_share(&upside, Vec3::new(0.0, 4.0, 0.0)) > 0.99, "turned over, it shows upward");
    }

    /// A spot's bulb shows only from inside its beam.
    #[test]
    fn a_spots_bulb_shows_from_inside_its_beam() {
        let cos = |deg: f32| deg.to_radians().cos();
        let s = GlareSource { cone: Some((Vec3::NEG_Y, cos(30.0), cos(20.0))), ..sconce([1.0; 6]) };
        assert!(visible_share(&s, Vec3::new(0.0, 0.0, 0.0)) > 0.99, "on the axis");
        assert_eq!(visible_share(&s, Vec3::new(3.0, 2.0, 0.0)), 0.0, "beside the beam");
    }

    /// The veil falls with the square of the distance and scales with
    /// exposure; far and dim enough, nothing is drawn.
    #[test]
    fn the_veil_follows_the_light_at_the_eye() {
        let s = sconce([1.0; 6]);
        let near = glare_quad(&s, Vec3::new(0.0, 0.0, 0.0), 3.6, 1.0).unwrap();
        let far = glare_quad(&s, Vec3::new(0.0, -2.0, 0.0), 3.6, 1.0).unwrap();
        // 2 m and 4 m: a quarter of the light; the core also halves in angle.
        assert!((far.a / near.a - 0.25).abs() < 1e-4, "{} vs {}", far.a, near.a);
        assert!((far.core2 / near.core2 - 0.25).abs() < 0.01);
        let brighter = glare_quad(&s, Vec3::new(0.0, 0.0, 0.0), 7.2, 1.0).unwrap();
        assert!((brighter.a / near.a - 2.0).abs() < 1e-4);
        // Below the cap, a brighter veil reaches farther.
        let (dim, less_dim) = (glare_quad(&s, Vec3::ZERO, 0.5, 1.0).unwrap(), glare_quad(&s, Vec3::ZERO, 1.0, 1.0).unwrap());
        assert!(dim.degrees < less_dim.degrees && less_dim.degrees < MAX_GLARE_DEGREES, "{dim:?} {less_dim:?}");
        assert!(glare_quad(&s, Vec3::new(0.0, -200.0, 0.0), 0.5, 1.0).is_none(), "a faint lamp far off");
        assert!(glare_quad(&sconce([0.0; 6]), Vec3::ZERO, 3.6, 1.0).is_none(), "a bulb that shows from nowhere");
    }

    /// The quad reaches as far as the veil is worth drawing and no farther
    /// than the cap, and its half size is that reach at its distance.
    #[test]
    fn the_quad_reaches_to_the_faintest_veil_drawn() {
        let s = sconce([1.0; 6]);
        let q = glare_quad(&s, Vec3::ZERO, 3.6, 1.0).unwrap();
        assert!(q.degrees <= MAX_GLARE_DEGREES && q.degrees > 2.0, "{q:?}");
        assert!((q.half - 2.0 * q.degrees.to_radians().tan()).abs() < 1e-4);
        // At the quad's edge the veil was at the floor (or the cap), and is
        // taken off everywhere, so it ends there at zero.
        assert!((q.edge - VEIL_FLOOR).abs() < 0.01 * VEIL_FLOOR || q.degrees == MAX_GLARE_DEGREES, "{q:?}");
        assert!((q.a * cie_veil(q.degrees * q.degrees + q.core2) - q.edge).abs() < 1e-6);
        assert!((q.peak - (q.a * cie_veil(q.core2) - q.edge)).abs() < 1e-3 * q.peak);
        // Bright enough, the cap: a brighter lamp's quad grows no wider.
        let blinding = glare_quad(&s, Vec3::ZERO, 3600.0, 1.0).unwrap();
        assert_eq!(blinding.degrees, MAX_GLARE_DEGREES);
        let (v, i) = build_glare(&[s], [Vec3::ZERO; 2], Vec3::X, Vec3::Z, 3.6, 1.0, true, &[]);
        assert_eq!((v.len(), i.len()), (4, 6));
        // Centred in front of the bulb by the wall margin, and as wide from
        // there as the veil is from the bulb; the taps sized to the bulb.
        let centre = Vec3::from_slice(&v[0].test[..3]);
        assert!((centre - Vec3::new(0.0, 2.0 - WALL_MARGIN, 0.0)).length() < 1e-5, "{centre}");
        let half = (Vec3::from(v[0].position) - centre).x.abs();
        assert!((half - (2.0 - WALL_MARGIN) * q.degrees.to_radians().tan()).abs() < 1e-4);
        assert!((v[0].test[3] - LAMP_RADIUS / half).abs() < 1e-5);
        let (untested, _) = build_glare(&[s], [Vec3::ZERO; 2], Vec3::X, Vec3::Z, 3.6, 1.0, false, &[]);
        assert!(untested.iter().all(|v| v.test[3] < 0.0), "no probe pass, no wall test");
    }

    /// A hand held between the eyes and a bulb takes its veil; beside the line
    /// of sight it takes none, and the eyes' own head never counts.
    #[test]
    fn a_hand_raised_against_the_light_shields_the_eye() {
        let bulb = Vec3::new(0.0, 2.0, -3.0);
        let eye = Vec3::new(0.0, 1.6, 0.0);
        let to_bulb = (bulb - eye).normalize();
        let across = to_bulb.cross(Vec3::Y).normalize();
        // A hand 30 cm out, across the line of sight.
        let hand = |off: f32| {
            let c = eye + to_bulb * 0.3 + across * off;
            (c - Vec3::Y * 0.08, c + Vec3::Y * 0.08, 0.045)
        };
        assert!(capsule_visibility(bulb, eye, &[hand(0.0)]) < 1e-3, "over the bulb");
        assert!(capsule_visibility(bulb, eye, &[hand(0.2)]) > 0.999, "beside it");
        let edge = capsule_visibility(bulb, eye, &[hand(0.045)]);
        assert!(edge > 0.2 && edge < 0.8, "its edge across the bulb: {edge}");
        let head = (eye - Vec3::Y * 0.05, eye + Vec3::Y * 0.05, 0.1);
        assert_eq!(capsule_visibility(bulb, eye, &[head]), 1.0, "the eye's own head");
        // One eye shielded, the other not: half the veil.
        let s = GlareSource { position: bulb, ..sconce([1.0; 6]) };
        let eyes = [eye - across * 0.032, eye + across * 0.032];
        let peak = |caps: &[(Vec3, Vec3, f32)]| build_glare(&[s], eyes, across, Vec3::Y, 3.6, 1.0, false, caps).0[0].shape[0];
        // `shape[0]` is the veil's scale, which carries the shielding.
        let one_eye = (eyes[0] + to_bulb * 0.3, eyes[0] + to_bulb * 0.3 + Vec3::Y * 0.01, 0.02);
        assert!((peak(&[one_eye]) / peak(&[]) - 0.5).abs() < 0.02, "{} vs {}", peak(&[one_eye]), peak(&[]));
    }

    /// RENDERED, into a 4x multisampled target as the scene pass draws it,
    /// with a probe pass depth beside it: bright at the source and faded out
    /// before the quad's edge; laid OVER a surface nearer than the lamp, since
    /// the veil is in the eye; gone when a wall hides the bulb, and part gone
    /// when a wall hides part of it.
    #[test]
    fn a_veil_lies_over_everything_and_goes_only_when_a_wall_hides_its_bulb() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        const W: u32 = 64;
        let format = TextureFormat::Rgba8Unorm;
        let uniform_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let probe_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(&device);
        let glare = GlarePipeline::new_multisampled(&device, format, &uniform_layout, &probe_layout, 4);
        // Clip space is the world here: the source at the origin at depth 0.5.
        let one: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0];
        let view_proj = [one, one];
        let camera = wgpu::util::DeviceExt::create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&view_proj), usage: BufferUsages::UNIFORM },
        );
        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &uniform_layout,
            entries: &[BindGroupEntry { binding: 0, resource: camera.as_entire_binding() }],
        });
        // A wall's depth in the probe pass, over `x < wall_x` in clip space.
        let wall = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: VertexState {
                module: &device.create_shader_module(ShaderModuleDescriptor {
                    label: None,
                    source: ShaderSource::Wgsl(
                        "struct W { x: f32 }
                        @group(0) @binding(0) var<uniform> w: W;
                        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                            let c = array<vec2<f32>, 6>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
                            let x = mix(-1.0, w.x, c[i].x * 0.5 + 0.5);
                            return vec4<f32>(x, c[i].y, 0.25, 1.0);
                        }"
                        .into(),
                    ),
                }),
                entry_point: Some("vs"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: None,
            primitive: PrimitiveState::default(),
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
        // `scene_depth`: what the scene pass drew in front of or behind the
        // lamp (at 0.5); `wall_x`: how far across the probe depth a wall
        // stands in front of it, from the left (-1: none, 1: all of it).
        // A veil reaching 10 degrees from a half-degree core, its scale `a`.
        const CORE2: f32 = 0.25;
        const REACH2: f32 = 100.0;
        let edge = |a: f32| a * cie_veil(REACH2 + CORE2);
        let render = |scene_depth: f32, wall_x: f32, a: f32| -> Vec<u8> {
            let quad = [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(u, v)| GlareVertex {
                position: [u, v, 0.0],
                colour: [1.0, 1.0, 1.0],
                uv: [u, v],
                shape: [a, CORE2, REACH2, edge(a)],
                // The bulb half the quad's size: taps a third of the way out.
                test: [0.0, 0.0, 0.0, 0.5],
            });
            let vb = wgpu::util::DeviceExt::create_buffer_init(
                &device,
                &wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&quad), usage: BufferUsages::VERTEX },
            );
            let ib = wgpu::util::DeviceExt::create_buffer_init(
                &device,
                &wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[0u32, 1, 2, 0, 2, 3]), usage: BufferUsages::INDEX },
            );
            let probe = crate::renderer::brush_pipeline::probe_pass::Target::new(&device, &probe_layout, W, W, 1);
            let wall_x_buf = wgpu::util::DeviceExt::create_buffer_init(
                &device,
                &wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[wall_x, 0.0, 0.0, 0.0]), usage: BufferUsages::UNIFORM },
            );
            let wall_bind = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &wall.get_bind_group_layout(0),
                entries: &[BindGroupEntry { binding: 0, resource: wall_x_buf.as_entire_binding() }],
            });
            let target = |fmt: TextureFormat, samples: u32, usage: TextureUsages| {
                device.create_texture(&TextureDescriptor {
                    label: None,
                    size: Extent3d { width: W, height: W, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: TextureDimension::D2,
                    format: fmt,
                    usage,
                    view_formats: &[],
                })
            };
            let msaa = target(format, 4, TextureUsages::RENDER_ATTACHMENT);
            let resolved = target(format, 1, TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC);
            let depth = target(TextureFormat::Depth32Float, 4, TextureUsages::RENDER_ATTACHMENT);
            let (msaa_v, resolved_v, depth_v) = (
                msaa.create_view(&Default::default()),
                resolved.create_view(&Default::default()),
                depth.create_view(&Default::default()),
            );
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &probe.depth_view,
                        depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if wall_x > -1.0 {
                    pass.set_pipeline(&wall);
                    pass.set_bind_group(0, &wall_bind, &[]);
                    pass.draw(0..6, 0..1);
                }
            }
            {
                let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &msaa_v,
                        depth_slice: None,
                        resolve_target: Some(&resolved_v),
                        ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Discard },
                    })],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &depth_v,
                        depth_ops: Some(Operations { load: LoadOp::Clear(scene_depth), store: StoreOp::Discard }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&glare.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.set_bind_group(1, &probe.bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..6, 0, 0..1);
            }
            let read = device.create_buffer(&BufferDescriptor {
                label: None,
                size: (W * W * 4) as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.copy_texture_to_buffer(
                TexelCopyTextureInfo { texture: &resolved, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                TexelCopyBufferInfo {
                    buffer: &read,
                    layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(W) },
                },
                Extent3d { width: W, height: W, depth_or_array_layers: 1 },
            );
            queue.submit([enc.finish()]);
            read.slice(..).map_async(MapMode::Read, |_| {});
            let _ = device.poll(PollType::Wait { submission_index: None, timeout: None });
            let data = read.slice(..).get_mapped_range().expect("mapped").to_vec();
            data
        };
        let px = |img: &[u8], x: u32, y: u32| img[((y * W + x) * 4) as usize];
        // The pixel `k` pixels right of the centre one, as the shader shades it
        // -- the veil, a `share` of it, through the scene's tone curve.
        let expected = |a: f32, share: f32, k: u32| {
            let u = ((W / 2 + k) as f32 + 0.5) / W as f32 * 2.0 - 1.0;
            let v = (W / 2) as f32 + 0.5;
            let v = v / W as f32 * 2.0 - 1.0;
            let veil = share * (a * cie_veil((u * u + v * v) * REACH2 + CORE2) - edge(a)).max(0.0);
            255.0 * crate::renderer::tonemap::aces_fitted(Vec3::splat(veil)).x
        };
        let open = render(1.0, -1.0, 1.0);
        let (centre, mid, rim) = (px(&open, W / 2, W / 2), px(&open, W / 2 + W / 8, W / 2), px(&open, W - 1, W / 2));
        assert!(centre > 240, "the core is white: {centre}");
        assert!((mid as f32 - expected(1.0, 1.0, W / 8)).abs() <= 2.0, "the veil round it: {mid} vs {}", expected(1.0, 1.0, W / 8));
        assert!(mid > 20 && mid < centre);
        assert_eq!(rim, 0, "nothing at the quad's edge");
        let nearer = render(0.25, -1.0, 1.0);
        assert_eq!(px(&nearer, W / 2, W / 2), centre, "over a surface nearer than the lamp, as the eye sees it");
        let walled = render(1.0, 1.0, 1.0);
        assert_eq!(px(&walled, W / 2, W / 2), 0, "a wall in front of the bulb takes its veil");
        // Three of the seven taps behind a wall standing over the left: the
        // veil at four sevenths of the light. Dim enough that the tone curve
        // does not flatten the difference.
        let dim = 0.01;
        let half = px(&render(1.0, -0.1, dim), W / 2, W / 2) as f32;
        let whole = px(&render(1.0, -1.0, dim), W / 2, W / 2) as f32;
        assert!((whole - expected(dim, 1.0, 0)).abs() <= 2.0, "{whole} vs {}", expected(dim, 1.0, 0));
        assert!(
            (half - expected(dim, 4.0 / 7.0, 0)).abs() <= 2.0,
            "part of the bulb, part of the veil: {half} vs {}",
            expected(dim, 4.0 / 7.0, 0)
        );
        assert!(whole - half > 20.0);
    }
}
