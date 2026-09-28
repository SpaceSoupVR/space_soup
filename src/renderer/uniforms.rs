use bytemuck::{Pod, Zeroable};
use super::shadow::MAX_SPOT_SHADOWS;
use glam::{Mat4, Quat, Vec3};
use wgpu::*;

use super::lights::LightsUniform;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    /// One view-projection PER EYE, always, whether or not the frame is drawn
    /// with multiview.
    ///
    /// Two slots even for a single-view pipeline, because the uniform's SHAPE
    /// decides the bind group layout, and a layout that changes with the render
    /// mode would split every pipeline in the renderer into two incompatible
    /// families. With one shape, a single-view shader reads slot 0 and a
    /// multiview shader indexes by `@builtin(view_index)`, and both are served
    /// by the same bind group -- which is what lets multiview arrive pipeline
    /// by pipeline instead of as one irreversible switch.
    pub view_proj: [[[f32; 4]; 4]; 2],
    /// The inverse, so the sky pass can turn a pixel back into a view ray.
    ///
    /// Inverted on the CPU once per eye rather than in the shader: a 4x4
    /// inverse per fragment is absurd, and the sky is a full-screen pass.
    pub inv_view_proj: [[[f32; 4]; 4]; 2],
    /// Sun (directional) light-space view-projection, for the sun shadow map.
    pub sun_view_proj: [[f32; 4]; 4],
    /// The same for the moving-objects sun map. MUST stay in step with the
    /// `Camera` struct in `wgsl_lights_block`.
    pub sun_dynamic_view_proj: [[f32; 4]; 4],
    /// One light-space view-projection per shadow-casting spot.
    ///
    /// An ARRAY because a level has more than one lamp. It used to be a single
    /// matrix and the first spot in the scene silently claimed it, so a room
    /// with two identical fixtures had one casting a shadow and one not -- which
    /// reads as a broken light rather than an exhausted budget.
    pub spot_view_proj: [[[f32; 4]; 4]; MAX_SPOT_SHADOWS],
    /// World-space camera position (xyz) PER EYE; w unused. Drives specular.
    ///
    /// Per eye for the same reason the matrices are: the two eyes are a few
    /// centimetres apart, and a specular highlight computed from the wrong one
    /// sits in the wrong place -- which in a stereo display is not a dimmer
    /// highlight but a highlight at the wrong DEPTH.
    pub camera_pos: [[f32; 4]; 2],
    /// x = sun shadow enabled (1/0), y = how many spot shadow layers are live,
    /// z = the moving-objects sun map is live (1/0), w reserved.
    ///
    /// Which light uses which layer is carried on the LIGHT (`params.w`) rather
    /// than here, because it is a property of the light and not of the camera --
    /// and the previous single "flashlight index" could only ever name one.
    pub shadow_params: [f32; 4],
    /// x = sky intensity. yzw reserved.
    pub sky_params: [f32; 4],
    /// Nine RGB spherical-harmonic coefficients of the sky's irradiance.
    ///
    /// `vec4` per coefficient because std140 rounds an array element up to 16
    /// bytes anyway -- packing them as vec3 would be the same size and would
    /// need the shader to index a padded array by hand.
    ///
    /// MUST stay in step with the `Camera` struct in `wgsl_lights_block`:
    /// bytemuck checks size and alignment, not names, and a mismatch surfaces
    /// as `invalid field accessor` pointing at the WGSL line that reads it.
    pub sky_sh: [[f32; 4]; 9],
    /// x = exposure, y = tone mapping mode (0 = ACES, 1 = none). zw reserved.
    ///
    /// On the camera because that is what they are -- exposure is a property of
    /// the eye looking at the scene, not of the sky or of any light. MUST stay
    /// in step with the `Camera` struct in `wgsl_lights_block`.
    pub post_params: [f32; 4],
    /// xyz = the player's world position, w = their yaw.
    ///
    /// Geometry reaches the renderer already rotated and translated into the
    /// player's frame, which is what makes the world move past them. Anything
    /// that must stay pinned to the WORLD -- terrain's texture projection, its
    /// height-based layer blend -- has to undo that, and this is what it undoes
    /// it with. Without it the ground's texture travels with the player and the
    /// scene looks static however far you walk.
    pub player_frame: [f32; 4],
    /// x = how many reflection probes are live. yzw reserved.
    pub probe_params: [f32; 4],
    /// Per probe: centre, box minimum, box maximum. See
    /// `space_soup_engine::reflection_probe::box_project`.
    ///
    /// Three `vec4` rather than a struct because std140 pads anything smaller
    /// to 16 bytes regardless, so a tighter layout would cost the same and
    /// would need the shader to unpack it by hand.
    pub probe_boxes: [[[f32; 4]; 3]; MAX_PROBES],
    /// x = how many doorway portals are live. yzw reserved.
    ///
    /// MUST stay in step with the `Camera` struct in `wgsl_lights_block`.
    pub portal_params: [f32; 4],
    /// Per portal: `[min.xyz, axis]`, `[max.xyz, low-side volume]`,
    /// `[high-side volume, wall min, wall max, _]`. See [`ProbeUpload::portals`].
    pub probe_portals: [[[f32; 4]; 3]; MAX_PORTALS],
    /// x = how many reflection proxies are live. yzw reserved.
    ///
    /// MUST stay in step with the `Camera` struct in `wgsl_lights_block`.
    pub proxy_params: [f32; 4],
    /// Per proxy: `[centre.xyz, volume]`, `[half_size.xyz, bounds-only]`,
    /// `[rotation xyzw]`. See [`ProbeUpload::proxies`].
    pub probe_proxies: [[[f32; 4]; 3]; MAX_PROXIES],
    /// THE RESIDENT ROOMS AS TABLES, so the reflection trace LOOKS a room's
    /// photographs, doorways and proxies up instead of searching all of them.
    /// Sixteen entries a block of four rows, entry `k` at `[row + k / 4][k %
    /// 4]`, -1 for none: rows 0-3 each room's first slot, 4-7 each slot's next
    /// slot of its room; 8-11 each room's first doorway, 12-15 each doorway's
    /// next in its low room (entry `2p`) and in its high room (`2p + 1`);
    /// 16-19 each room's first proxy, 20-23 each proxy's next in its room.
    /// Every chain ascends. The room numbers in `probe_boxes`,
    /// `probe_portals` and `probe_proxies` are THIS table's, not the level's
    /// volume ids. See [`ProbeUpload::dense_rooms`].
    ///
    /// MUST stay in step with the `Camera` struct in `wgsl_lights_block`.
    pub probe_rooms: [[f32; 4]; ROOM_TABLE_ROWS],
}

/// How many boxes standing inside rooms -- a pillar, a hanging lamp -- the
/// reflection trace tests. See [`ProbeProxy`].
///
/// Only those of the room a ray is crossing are tested at all, so the cap is
/// on the uniform (48 bytes each), not on the per-pixel cost. Residency picks
/// the nearest to the player each frame.
pub const MAX_PROXIES: usize = 16;

/// How many doorway portals the shader walks per fragment.
///
/// A portal is a box test on every reflective pixel, and only the pixels near
/// an opening go on to sample anything. Eight covers every doorway around the
/// player in any plausible room layout; residency picks the nearest each frame
/// from however many the level has.
pub const MAX_PORTALS: usize = 8;

/// How many reflection probes one scene may have live at once.
///
/// A cap on the uniform and on the cube array, not on how many a level may
/// contain: probes are selected per fragment by which box contains it, so the
/// cost of the cap is that a level with more rooms than this leaves some of
/// them reflecting the sky.
///
/// WHY THIS WENT FROM FOUR TO SIXTEEN
///
/// Four was set as "a level of a few rooms", which framed a probe as a
/// per-ROOM thing. It is not: a probe is a photograph from one point with a
/// box around it, and it is only true for a volume that can actually SEE what
/// the photograph saw. A room with a column in it is not such a volume -- the
/// probe reports the far wall's light to surfaces the column is standing in
/// front of, because nothing in the probe path tests visibility. That is the
/// reflection that appears and disappears as the player moves: two sources,
/// the screen and the probe, disagreeing about whether something is occluded.
///
/// The fix is smaller boxes, which means more of them, and the selection rule
/// already supports it -- the SMALLEST box containing a fragment wins, so a
/// tight volume nested inside a room's volume takes precedence with no extra
/// authoring rules.
///
/// The cap is not a budget. Each probe costs 196 KB of cube array (six 64x64
/// RGBA16F faces) and 48 bytes of uniform, and the per-fragment loop does box
/// tests only -- ONE `textureSampleLevel` happens, on the winner, however many
/// probes there are. Sixteen is 3.1 MB and sixteen box tests.
pub const MAX_PROBES: usize = 16;

// The cube ARRAY's size is no longer a constant: a level's probes stream
// through a pool sized from a memory budget and the device's own array-layer
// limit. See `probe_stream::pool_size`. (It was 32 here, justified as a
// hardware ceiling of 256 layers; that was wgpu's default limit, and Adreno 740
// reports 2048.)

/// Where the player is, for geometry that must not move with them.
#[derive(Clone, Copy, Default)]
pub struct PlayerUpload {
    pub offset: Vec3,
    pub yaw: f32,
}

/// Per-frame display settings: how radiance becomes pixels.
#[derive(Clone, Copy)]
pub struct PostUpload {
    pub exposure: f32,
    pub tonemap: super::tonemap::ToneMapping,
}

impl Default for PostUpload {
    /// ACES at exposure 1. A contemporary renderer tone maps; defaulting to
    /// the hard clamp would leave every scene blowing out its highlights,
    /// which is the thing the curve exists to fix.
    fn default() -> Self {
        Self { exposure: 1.0, tonemap: super::tonemap::ToneMapping::default() }
    }
}

/// Per-frame sky inputs.
#[derive(Clone)]
pub struct SkyUpload {
    pub intensity: f32,
    pub sh: [[f32; 4]; 9],
}

impl SkyUpload {
    /// A scene with no sky: the flat ambient the engine always had, expressed
    /// as a constant-band SH so there is one lighting path rather than a branch.
    pub fn none() -> Self {
        Self::from(&crate::renderer::sky::SkyIrradiance::flat(crate::renderer::sky::AMBIENT))
    }

    pub fn from(irr: &crate::renderer::sky::SkyIrradiance) -> Self {
        let mut sh = [[0.0f32; 4]; 9];
        for i in 0..9 {
            sh[i] = [irr.sh[i][0], irr.sh[i][1], irr.sh[i][2], 0.0];
        }
        Self { intensity: 1.0, sh }
    }
}

/// Per-frame shadow inputs, bundled to keep `upload` call sites readable.
pub struct ShadowUpload {
    pub sun_view_proj: Mat4,
    /// The moving-objects sun map's matrix. See `shadow::SUN_DYNAMIC_DIM`.
    pub sun_dynamic_view_proj: Mat4,
    /// Whether that map was drawn this frame. When it was not, the shader
    /// treats every point as unshadowed by moving things.
    pub sun_dynamic_enabled: bool,
    /// One per shadow-casting spot, in shadow-layer order.
    pub spot_view_proj: [Mat4; MAX_SPOT_SHADOWS],
    pub sun_enabled: bool,
    /// How many spot shadow layers this frame actually filled.
    ///
    /// Replaces a bool plus a single "flashlight index": with an array of
    /// layers, WHICH light uses WHICH layer belongs on the light, and the
    /// camera only needs to know how many are live.
    pub spot_count: u32,
}

impl ShadowUpload {
    /// No shadows (identity matrices, both disabled) — used by paths that don't
    /// render a shadow pass yet (e.g. the XR renderer).
    pub fn disabled() -> Self {
        Self {
            sun_view_proj: Mat4::IDENTITY,
            sun_dynamic_view_proj: Mat4::IDENTITY,
            sun_dynamic_enabled: false,
            spot_view_proj: [Mat4::IDENTITY; MAX_SPOT_SHADOWS],
            sun_enabled: false,
            spot_count: 0,
        }
    }
}

/// Camera, lights, and both shadow maps share one bind group — wgpu's default
/// `max_bind_groups` limit of 4 leaves no room for extra groups once the
/// skinned mesh pipeline's model/texture/joint groups are accounted for.
/// Bindings: 0 = camera uniforms (vertex + fragment), 1 = lights (fragment),
/// 2 = sun shadow depth texture, 3 = shadow comparison sampler,
/// 4 = spot shadow depth texture.
pub struct UniformBuffer {
    pub buffer: Buffer,
    pub layout: BindGroupLayout,
    pub bind_group: BindGroup,
    /// Which reflection probes this scene has. See [`ProbeUpload`].
    probes: ProbeUpload,
    /// The probes' per-texel distances, bound at 8. Zero until a level with
    /// depth loads, and zero tells the shader to project onto the box as it
    /// always did. Kept here because the bind group must outlive nothing it
    /// references; see [`Self::set_probe_depth`].
    probe_depth_view: TextureView,
    probe_depth_sampler: Sampler,
}

impl UniformBuffer {
    /// Tell the shader where this level's probes are.
    ///
    /// Call once when a bake loads. Takes effect on the next `upload_scene`,
    /// which every frame does anyway.
    /// Bind these probe distances from the next [`Self::rebind_probes`] on.
    pub fn set_probe_depth(&mut self, view: TextureView) {
        self.probe_depth_view = view;
    }

    pub fn set_probes(&mut self, probes: ProbeUpload) {
        self.probes = probes;
    }
}

impl UniformBuffer {
    pub fn new(
        device: &Device,
        lights: &LightsUniform,
        sun_shadow_view: &TextureView,
        sun_dynamic_view: &TextureView,
        spot_shadow_view: &TextureView,
        shadow_sampler: &Sampler,
    ) -> Self {
        // Every caller that has no probes yet passes the neutral one, so the
        // layout never varies. A bind group layout that changed with the
        // presence of baked data would split every pipeline in two.
        let (probe_view, probe_sampler) = default_probe_cube(device);
        Self::new_with_probes(
            device, lights, sun_shadow_view, sun_dynamic_view, spot_shadow_view, shadow_sampler,
            &probe_view, &probe_sampler,
        )
    }

    /// As [`Self::new`], with a baked reflection cube array bound.
    pub fn new_with_probes(
        device: &Device,
        lights: &LightsUniform,
        sun_shadow_view: &TextureView,
        sun_dynamic_view: &TextureView,
        spot_shadow_view: &TextureView,
        shadow_sampler: &Sampler,
        probe_view: &TextureView,
        probe_sampler: &Sampler,
    ) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("uniform_buf"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("uniform_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Comparison),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        // D2, not D2Array: the spots share ONE atlas texture and
                        // occupy tiles of it, because a render pass is the
                        // expensive unit on a tile GPU and an array layer needs
                        // a pass each. See `ShadowMap::spots`.
                        //
                        // Must match the `texture_depth_2d` the shader declares,
                        // or bind group creation fails -- which is the good
                        // outcome, since the alternative is reading the wrong
                        // lamp's depth and drawing a shadow from a light that is
                        // not there.
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        // A CUBE ARRAY, so a level's rooms each get their own
                        // probe and a fragment picks by which box contains it.
                        // One cube would mean one reflection for the whole
                        // level, which is exactly the artefact a probe exists
                        // to remove.
                        view_dimension: TextureViewDimension::CubeArray,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::FRAGMENT,
                    // Filtering, not comparison: this is colour, not depth.
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // The sun's shadow of moving things only. See
                // `shadow::SUN_DYNAMIC_DIM`.
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Each probe texel's distance, same layers as binding 5. See
                // `probe_depth_descriptor`.
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::CubeArray,
                        multisampled: false,
                    },
                    count: None,
                },
                // NEAREST: a distance filtered across a pillar's edge is a
                // surface halfway between the pillar and the wall behind it,
                // which is nowhere.
                BindGroupLayoutEntry {
                    binding: 9,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let (probe_depth_view, probe_depth_sampler) = default_probe_depth(device);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("uniform_bg"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: lights.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(sun_shadow_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(shadow_sampler),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(spot_shadow_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(probe_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Sampler(probe_sampler),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::TextureView(sun_dynamic_view),
                },
                BindGroupEntry { binding: 8, resource: BindingResource::TextureView(&probe_depth_view) },
                BindGroupEntry { binding: 9, resource: BindingResource::Sampler(&probe_depth_sampler) },
            ],
        });

        Self {
            buffer,
            layout,
            bind_group,
            probes: ProbeUpload::default(),
            probe_depth_view,
            probe_depth_sampler,
        }
    }

    pub fn upload(&self, queue: &Queue, view_proj: Mat4, camera_pos: Vec3, shadow: &ShadowUpload) {
        self.upload_with_sky(queue, view_proj, camera_pos, shadow, &SkyUpload::none());
    }

    pub fn upload_with_sky(
        &self,
        queue: &Queue,
        view_proj: Mat4,
        camera_pos: Vec3,
        shadow: &ShadowUpload,
        sky: &SkyUpload,
    ) {
        self.upload_scene(
            queue, view_proj, camera_pos, shadow, sky,
            &PostUpload::default(), &PlayerUpload::default(),
        );
    }

    /// The full per-frame upload, including how radiance is mapped to pixels.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_scene(
        &self,
        queue: &Queue,
        view_proj: Mat4,
        camera_pos: Vec3,
        shadow: &ShadowUpload,
        sky: &SkyUpload,
        post: &PostUpload,
        player: &PlayerUpload,
    ) {
        self.upload_scene_with_probes(queue, view_proj, camera_pos, shadow, sky, post, player, None)
    }

    /// The same, with this frame's probe residency.
    ///
    /// Passed in rather than stored, because residency is decided in the middle
    /// of the frame -- after the draw lists have borrowed the renderer -- and a
    /// setter would need `&mut self` there. It is also honest about the
    /// lifetime: which probes are resident is a property of THIS frame's
    /// camera, not of the level.
    ///
    /// `None` keeps whatever was bound at load, which is what every caller that
    /// does not run residency wants.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_scene_with_probes(
        &self,
        queue: &Queue,
        view_proj: Mat4,
        camera_pos: Vec3,
        shadow: &ShadowUpload,
        sky: &SkyUpload,
        post: &PostUpload,
        player: &PlayerUpload,
        probes: Option<&ProbeUpload>,
    ) {
        self.upload_scene_stereo(
            queue,
            [view_proj; 2],
            [camera_pos; 2],
            shadow,
            sky,
            post,
            player,
            probes,
        );
    }

    /// The same, with a DIFFERENT camera per eye -- what a multiview pass needs.
    ///
    /// A multiview pass draws both eyes in one go, so it cannot re-upload the
    /// uniform between them: both eyes' matrices have to be resident at once,
    /// and the shader picks its own with `@builtin(view_index)`. That is what
    /// the two slots in `Uniforms` have always been for, and until now nothing
    /// ever wrote anything different into them.
    ///
    /// Index 0 is the left eye and 1 the right, matching the OpenXR view array
    /// and the swapchain's layer order -- which is also the order
    /// `@builtin(view_index)` counts in, so there is no mapping to get wrong.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_scene_stereo(
        &self,
        queue: &Queue,
        view_proj: [Mat4; 2],
        camera_pos: [Vec3; 2],
        shadow: &ShadowUpload,
        sky: &SkyUpload,
        post: &PostUpload,
        player: &PlayerUpload,
        probes: Option<&ProbeUpload>,
    ) {
        let probes = probes.unwrap_or(&self.probes);
        // Rooms renumbered 0.. with their lookup tables; see `dense_rooms`.
        let (dense, room_tables) = probes.dense_rooms();
        let u = Uniforms {
            // ONE PER EYE. A single-view pass reads slot 0 and gets the same
            // matrix in both, because `upload_scene_with_probes` duplicates it;
            // a multiview pass gets the two it was given. Writing slot 1 even
            // in the single-view case is deliberate -- a multiview shader that
            // reads it must never see whatever the previous frame left there.
            view_proj: [
                view_proj[0].to_cols_array_2d(),
                view_proj[1].to_cols_array_2d(),
            ],
            inv_view_proj: [
                view_proj[0].inverse().to_cols_array_2d(),
                view_proj[1].inverse().to_cols_array_2d(),
            ],
            sun_view_proj: shadow.sun_view_proj.to_cols_array_2d(),
            sun_dynamic_view_proj: shadow.sun_dynamic_view_proj.to_cols_array_2d(),
            spot_view_proj: std::array::from_fn(|i| shadow.spot_view_proj[i].to_cols_array_2d()),
            camera_pos: [
                [camera_pos[0].x, camera_pos[0].y, camera_pos[0].z, 1.0],
                [camera_pos[1].x, camera_pos[1].y, camera_pos[1].z, 1.0],
            ],
            shadow_params: [
                if shadow.sun_enabled { 1.0 } else { 0.0 },
                shadow.spot_count as f32,
                // z = the moving-objects sun map is live this frame.
                if shadow.sun_dynamic_enabled { 1.0 } else { 0.0 },
                0.0,
            ],
            sky_params: [sky.intensity, 0.0, 0.0, 0.0],
            sky_sh: sky.sh,
            player_frame: [player.offset.x, player.offset.y, player.offset.z, player.yaw],
            // From the buffer's own state rather than an argument: probe boxes
            // are level data set once when a scene loads, and threading them
            // through every per-frame upload call would put a constant in the
            // hot path and touch a dozen call sites that have no opinion on it.
            probe_params: {
                // What a horizontal ground plane receives from this sky, used
                // by `environment_radiance` for reflections pointing down.
                // Computed once here rather than nine harmonics per fragment.
                let g = sky_ground_irradiance(&sky.sh);
                [probes.count as f32, g[0], g[1], g[2]]
            },
            probe_boxes: dense.boxes,
            portal_params: [probes.portal_count as f32, if probes.no_trace { 1.0 } else { 0.0 }, 0.0, 0.0],
            probe_portals: dense.portals,
            proxy_params: [probes.proxy_count as f32, 0.0, 0.0, 0.0],
            probe_proxies: dense.proxies,
            probe_rooms: room_tables,
            post_params: [
                post.exposure,
                match post.tonemap {
                    super::tonemap::ToneMapping::Aces => 0.0,
                    super::tonemap::ToneMapping::None => 1.0,
                },
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&u));
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::renderer::shadow::ShadowMap;

    /// Where the eye sits in a render test.
    ///
    /// Straight out in front of the quad every harness here draws, so a
    /// specular highlight lands symmetrically and cannot be mistaken for one
    /// material winning over another. Only specular reads it.
    pub const TEST_EYE: glam::Vec3 = glam::Vec3::new(0.0, 0.0, 5.0);

    /// Group 0 for a render test, wired exactly as the renderer wires it.
    ///
    /// A real `ShadowMap`, only tiny: the shadow textures have to exist and be
    /// bound with the right sample types or nothing validates, and building a
    /// stand-in here would mean the tests stopped noticing when the real layout
    /// changed. 64 squared costs nothing and proves the same thing.
    ///
    /// The `ShadowMap` comes back with the buffer because it owns the textures
    /// the bind group points at; dropping it would leave the group dangling.
    pub fn scene_uniforms(device: &Device, lights: &LightsUniform) -> (ShadowMap, UniformBuffer) {
        let shadows = ShadowMap::with_dimension(device, 64);
        let uniforms = UniformBuffer::new(
            device,
            lights,
            shadows.sun_depth_view(),
            shadows.sun_dynamic_depth_view(),
            shadows.spot_depth_view(),
            shadows.sampler(),
        );
        (shadows, uniforms)
    }

    /// IEEE binary16 bits for a small non-negative value. Test-only, and
    /// deliberately a separate copy from the engine's: `space_soup` ships to
    /// crates.io and must not depend on `space_soup_engine` to build a
    /// four-texel test cube.
    fn test_f16(v: f32) -> u16 {
        let bits = v.to_bits();
        let exp = ((bits >> 23) & 0xff) as i32 - 127;
        let mant = bits & 0x007f_ffff;
        if v == 0.0 {
            return 0;
        }
        if exp < -14 {
            let shift = (-14 - exp) as u32;
            if shift > 24 {
                return 0;
            }
            return ((mant | 0x0080_0000) >> (shift + 13)) as u16;
        }
        (((exp + 15) as u16) << 10) | ((mant >> 13) as u16)
    }

    /// The same, with a reflection probe of one flat colour covering a box.
    ///
    /// Every face of the cube is that colour with alpha 255, so a test can ask
    /// "did the probe reach this fragment" without also having to reason about
    /// which face box projection chose. Which face it chose is covered by the
    /// pure tests in `space_soup_engine::reflection_probe`.
    pub fn scene_uniforms_with_probe(
        device: &Device,
        queue: &Queue,
        lights: &LightsUniform,
        colour: [u8; 4],
        min: glam::Vec3,
        max: glam::Vec3,
    ) -> (ShadowMap, UniformBuffer, TextureView, Sampler) {
        const RES: u32 = 4;
        let shadows = ShadowMap::with_dimension(device, 64);
        let mut uniforms = UniformBuffer::new(
            device,
            lights,
            shadows.sun_depth_view(),
            shadows.sun_dynamic_depth_view(),
            shadows.spot_depth_view(),
            shadows.sampler(),
        );
        // The cube is LINEAR HALF FLOAT now, so the sRGB decode the GPU used
        // to do on the way in happens here instead. Callers still name a
        // colour the way they always did.
        let texel: Vec<u8> = (0..4)
            .flat_map(|c| {
                let u = colour[c] as f32 / 255.0;
                // Alpha is coverage and was never sRGB.
                let linear = if c == 3 {
                    u
                } else if u <= 0.04045 {
                    u / 12.92
                } else {
                    ((u + 0.055) / 1.055).powf(2.4)
                };
                test_f16(linear).to_le_bytes()
            })
            .collect();
        let face: Vec<u8> = texel
            .iter()
            .copied()
            .cycle()
            .take((RES * RES * 8 * 6) as usize)
            .collect();
        let view = super::upload_probe_cubes(device, queue, RES, &[&face]);
        let (_unused, sampler) = super::default_probe_cube(device);
        let centre = (min + max) * 0.5;
        let mut probes = ProbeUpload { count: 1, ..Default::default() };
        probes.boxes[0] = [
            [centre.x, centre.y, centre.z, 0.0],
            [min.x, min.y, min.z, 0.0],
            [max.x, max.y, max.z, 0.0],
        ];
        uniforms.rebind_probes(
            device,
            lights,
            shadows.sun_depth_view(),
            shadows.sun_dynamic_depth_view(),
            shadows.spot_depth_view(),
            shadows.sampler(),
            &view,
            &sampler,
            probes,
        );
        (shadows, uniforms, view, sampler)
    }
}

/// What an upward-facing surface receives from the sky.
///
/// The shader's `sky_irradiance(vec3(0, 1, 0))`, evaluated on the CPU: with the
/// direction fixed, only the first two harmonics survive, so this is the whole
/// series rather than a truncation of it.
fn sky_ground_irradiance(sh: &[[f32; 4]; 9]) -> [f32; 3] {
    // Basis at (0, 1, 0): Y00 = 0.282095, Y1-1 = 0.488603 * y = 0.488603, and
    // Y20 = 0.315392 * (3z^2 - 1) = -0.315392. Every other term carries an x or
    // a z and vanishes.
    let terms = [
        (0usize, 0.282095_f32 * 1.0),
        (1, 0.488603 * 0.6666667),
        (6, -0.315392 * 0.25),
        (8, 0.546274 * (0.0 - 1.0) * 0.25),
    ];
    let mut out = [0.0f32; 3];
    for (i, w) in terms {
        for c in 0..3 {
            out[c] += sh[i][c] * w;
        }
    }
    // Never negative: a truncated harmonic series can ring below zero, and a
    // negative ground would subtract light from every reflection facing down.
    [out[0].max(0.0), out[1].max(0.0), out[2].max(0.0)]
}

/// The reflection probes a scene has, as the shader indexes them.
///
/// Set once when a level's bake loads. `count` is how many of `boxes` are live;
/// the rest are ignored rather than cleared, so a scene that loses a probe
/// cannot leave a stale box selecting a cube layer that no longer means
/// anything.
#[derive(Clone, Copy)]
pub struct ProbeUpload {
    /// How many SLOTS are live. Never more than [`MAX_PROBES`].
    pub count: u32,
    /// Per slot: `[centre.xyz, layer]`, `[min.xyz, _]`, `[max.xyz, _]`.
    ///
    /// THE LAYER RIDES IN THE CENTRE'S UNUSED `w`.
    ///
    /// A slot is a place in the shader's per-fragment loop; a layer is a cube
    /// in the texture array. They used to be the same number, which meant the
    /// loop had to walk every probe the level owned. Splitting them lets the
    /// array hold every baked probe -- they are 196 KB each and nothing on a
    /// headset notices -- while the loop only ever walks the ones that can be
    /// seen from where the player is standing.
    ///
    /// Packed into an existing `w` rather than added as a field because all
    /// three vec4s already carry a wasted component, and a new array would
    /// grow the uniform for nothing.
    pub boxes: [[[f32; 4]; 3]; MAX_PROBES],
    /// How many of `portals` are live. Never more than [`MAX_PORTALS`].
    pub portal_count: u32,
    /// THE DOORWAYS NEAR THE PLAYER: `[min.xyz, axis]`, `[max.xyz, low-side
    /// volume]`, `[high-side volume, _, _, _]`.
    ///
    /// A portal names VOLUMES, not slots, so it means the same thing whichever
    /// cells of either room happen to be resident this frame; the shader finds
    /// each side's nearest resident photograph itself. See
    /// `probe_through_portals` in the lights block.
    pub portals: [[[f32; 4]; 3]; MAX_PORTALS],
    /// Skip the probe depth trace this frame: `perf_ab` measures its cost by
    /// turning it off. Rides in `portal_params.y`, which was unused.
    pub no_trace: bool,
    /// How many of `proxies` are live. Never more than [`MAX_PROXIES`].
    pub proxy_count: u32,
    /// WHAT STANDS IN THE ROOMS NEAR THE PLAYER: `[centre.xyz, volume]`,
    /// `[half_size.xyz, bounds-only]`, `[rotation xyzw]`. Named by VOLUME like the
    /// portals, so it means the same thing whichever cells are resident.
    pub proxies: [[[f32; 4]; 3]; MAX_PROXIES],
}

/// One box standing inside a room, for the reflection trace: the rooms' own
/// boxes are its walls, floors and ceilings, and a proxy is everything else
/// a reflected ray can hit on the way -- a pillar, a lamp. Built by the app
/// from the level (`space_soup_engine::reflection_proxy`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeProxy {
    pub centre: Vec3,
    pub half_size: Vec3,
    pub rotation: Quat,
    /// The probe VOLUME (room) it stands in. See [`ProbeUpload::set_volume`].
    pub volume: u32,
    /// A brush piece IS its box; a model's box is only its bounds, and the
    /// shader asks the room's photographs where inside it the object is.
    pub solid: bool,
}

/// One doorway between two probe volumes, as the level's bake found it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbePortal {
    /// The opening's box -- the carve through the wall.
    pub min: Vec3,
    pub max: Vec3,
    /// The axis the opening is thin along: 0 = x, 1 = y, 2 = z.
    pub axis: u32,
    /// The volume on the LOW side of the opening along `axis`, and the one on
    /// the high side.
    pub low: u32,
    pub high: u32,
    /// The WALL the opening goes through, along `axis`: how deep its jambs,
    /// lintel and threshold are. The carve reaches past it on both sides.
    /// `None` takes the carve's own extent. See
    /// `space_soup_engine::reflection_proxy::portal_wall_extent`.
    pub wall: Option<(f32, f32)>,
}

impl ProbeUpload {
    /// Fill the portal list with at most [`MAX_PORTALS`] of `portals`, nearest
    /// to `player` first, keeping only those with a side among `volumes` --
    /// a doorway between two rooms neither of which is resident reflects
    /// nothing the shader could sample.
    pub fn set_portals(&mut self, portals: &[ProbePortal], player: Vec3, volumes: &[u32]) {
        let mut near: Vec<(f32, &ProbePortal)> = portals
            .iter()
            .filter(|p| volumes.contains(&p.low) || volumes.contains(&p.high))
            .map(|p| (p.min.max(player.min(p.max)).distance_squared(player), p))
            .collect();
        near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.portal_count = near.len().min(MAX_PORTALS) as u32;
        for (i, (_, p)) in near.iter().take(MAX_PORTALS).enumerate() {
            self.portals[i] = [
                [p.min.x, p.min.y, p.min.z, p.axis as f32],
                [p.max.x, p.max.y, p.max.z, p.low as f32],
                {
                    let a = p.axis as usize;
                    let (lo, hi) = p.wall.unwrap_or((p.min[a], p.max[a]));
                    [p.high as f32, lo, hi, 0.0]
                },
            ];
        }
    }

    /// Fill the proxy list with at most [`MAX_PROXIES`] of `proxies`, nearest
    /// to `player` first, keeping only those standing in one of `volumes`.
    pub fn set_proxies(&mut self, proxies: &[ProbeProxy], player: Vec3, volumes: &[u32]) {
        let mut near: Vec<(f32, &ProbeProxy)> = proxies
            .iter()
            .filter(|p| volumes.contains(&p.volume))
            .map(|p| (p.centre.distance_squared(player), p))
            .collect();
        near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.proxy_count = near.len().min(MAX_PROXIES) as u32;
        for (i, (_, p)) in near.iter().take(MAX_PROXIES).enumerate() {
            let q = p.rotation.normalize();
            self.proxies[i] = [
                [p.centre.x, p.centre.y, p.centre.z, p.volume as f32],
                [p.half_size.x, p.half_size.y, p.half_size.z, if p.solid { 0.0 } else { 1.0 }],
                [q.x, q.y, q.z, q.w],
            ];
        }
    }

    /// The VOLUME slot `slot` belongs to -- which room it photographs.
    ///
    /// Rides in the MAX corner's unused `w`. Two cells of one room share it,
    /// and that is what lets the shader blend them and nothing else.
    pub fn set_volume(&mut self, slot: usize, volume: u32) {
        if let Some(b) = self.boxes.get_mut(slot) {
            b[2][3] = volume as f32;
        }
    }

    /// Which volume slot `slot` belongs to. See [`ProbeUpload::set_volume`].
    pub fn volume(&self, slot: usize) -> u32 {
        self.boxes.get(slot).map(|b| b[2][3] as u32).unwrap_or(0)
    }

    /// Fill slot `slot` with the box of the probe living at `layer`.
    pub fn set(&mut self, slot: usize, layer: u32, centre: Vec3, min: Vec3, max: Vec3) {
        if slot >= MAX_PROBES {
            return;
        }
        self.boxes[slot] = [
            [centre.x, centre.y, centre.z, layer as f32],
            [min.x, min.y, min.z, 0.0],
            [max.x, max.y, max.z, 0.0],
        ];
    }

    /// Record the average radiance the probe in `slot` photographed.
    ///
    /// Rides in the MIN corner's unused `w`, beside the layer in the centre's.
    /// Zero -- what [`ProbeUpload::set`] leaves -- means unknown, and the shader
    /// then uses that probe unnormalised. See `PROBE_NORMALISATION` in the
    /// lights block.
    pub fn set_brightness(&mut self, slot: usize, brightness: f32) {
        if let Some(b) = self.boxes.get_mut(slot) {
            b[1][3] = brightness;
        }
    }

    /// Fill every live slot's brightness from a per-LAYER table.
    pub fn fill_brightness(&mut self, by_layer: &[f32]) {
        for slot in 0..(self.count as usize).min(MAX_PROBES) {
            let layer = self.layer(slot) as usize;
            self.set_brightness(slot, by_layer.get(layer).copied().unwrap_or(0.0));
        }
    }

    /// Fill every live slot's room from a per-PROBE table, before the stream
    /// turns probes into layers. See [`ProbeUpload::set_volume`].
    pub fn fill_volumes(&mut self, by_probe: &[u32]) {
        for slot in 0..(self.count as usize).min(MAX_PROBES) {
            let probe = self.layer(slot) as usize;
            self.set_volume(slot, by_probe.get(probe).copied().unwrap_or(u32::MAX));
        }
    }

    /// The rooms of the live slots, for choosing which doorways matter.
    pub fn volumes(&self) -> Vec<u32> {
        (0..(self.count as usize).min(MAX_PROBES)).map(|s| self.volume(s)).collect()
    }

    /// Which cube-array layer slot `slot` reads. See [`ProbeUpload::boxes`].
    pub fn layer(&self, slot: usize) -> u32 {
        self.boxes.get(slot).map(|b| b[0][3] as u32).unwrap_or(0)
    }

    /// THE UPLOAD WITH ITS ROOMS RENUMBERED 0.., and the tables that let the
    /// shader find a room's photographs without searching.
    ///
    /// Rooms are numbered in order of their first slot, so room `r`'s slots
    /// are found from `tables[r / 4][r % 4]` and chained, ascending, through
    /// `tables[4 + s / 4][s % 4]` -- the same slots, in the same order, as a
    /// scan of every slot for that room visits them, which is what keeps
    /// every tie between two photographs broken the same way. A portal side
    /// or a proxy standing in a room with no resident slot is given
    /// [`NO_RESIDENT_ROOM`], which, like its old volume id, matches no slot.
    ///
    /// The shader only ever compares rooms for equality, so renumbering them
    /// changes no result: it is what makes them usable as indices.
    pub fn dense_rooms(&self) -> (ProbeUpload, [[f32; 4]; ROOM_TABLE_ROWS]) {
        const _: () = assert!(
            MAX_PROBES <= 16 && MAX_PORTALS * 2 <= 16 && MAX_PROXIES <= 16,
            "each block of the room tables holds sixteen entries",
        );
        let mut out = *self;
        let mut tables = [[-1.0f32; 4]; ROOM_TABLE_ROWS];
        let put = |tables: &mut [[f32; 4]; ROOM_TABLE_ROWS], row: usize, k: usize, v: usize| {
            tables[row + k / 4][k % 4] = v as f32;
        };
        let mut rooms: Vec<f32> = Vec::new();
        let mut last: Vec<usize> = Vec::new();
        for slot in 0..(self.count as usize).min(MAX_PROBES) {
            let raw = self.boxes[slot][2][3];
            let room = match rooms.iter().position(|&r| r == raw) {
                Some(room) => {
                    put(&mut tables, 4, last[room], slot);
                    last[room] = slot;
                    room
                }
                None => {
                    rooms.push(raw);
                    last.push(slot);
                    let room = rooms.len() - 1;
                    put(&mut tables, 0, room, slot);
                    room
                }
            };
            out.boxes[slot][2][3] = room as f32;
        }
        let renumber = |raw: f32| rooms.iter().position(|&r| r == raw);
        // DOORWAYS: each is in the chain of both its rooms. Its link onward in
        // its LOW room's chain is entry `2p`, in its HIGH room's `2p + 1`, so
        // the shader follows whichever side names the room it is walking.
        let mut last_portal: Vec<Option<usize>> = vec![None; rooms.len()];
        for p in 0..(self.portal_count as usize).min(MAX_PORTALS) {
            let low = renumber(self.portals[p][1][3]);
            let high = renumber(self.portals[p][2][0]);
            out.portals[p][1][3] = low.map_or(NO_RESIDENT_ROOM, |r| r as f32);
            out.portals[p][2][0] = high.map_or(NO_RESIDENT_ROOM, |r| r as f32);
            for room in [low, high.filter(|&h| Some(h) != low)].into_iter().flatten() {
                match last_portal[room] {
                    None => put(&mut tables, 8, room, p),
                    Some(prev) => {
                        let side = if out.portals[prev][1][3] == room as f32 { 0 } else { 1 };
                        put(&mut tables, 12, 2 * prev + side, p);
                    }
                }
                last_portal[room] = Some(p);
            }
        }
        let mut last_proxy: Vec<Option<usize>> = vec![None; rooms.len()];
        for i in 0..(self.proxy_count as usize).min(MAX_PROXIES) {
            let room = renumber(self.proxies[i][0][3]);
            out.proxies[i][0][3] = room.map_or(NO_RESIDENT_ROOM, |r| r as f32);
            if let Some(room) = room {
                match last_proxy[room] {
                    None => put(&mut tables, 16, room, i),
                    Some(prev) => put(&mut tables, 20, prev, i),
                }
                last_proxy[room] = Some(i);
            }
        }
        (out, tables)
    }
}

/// The room number [`ProbeUpload::dense_rooms`] gives a portal side or a
/// proxy standing in a room with no resident photograph: matches no slot, and
/// is outside the tables, so a lookup of it finds nothing.
pub const NO_RESIDENT_ROOM: f32 = 1_000_000.0;

/// Rows of [`Uniforms::probe_rooms`]: six blocks of sixteen entries.
pub const ROOM_TABLE_ROWS: usize = 24;

impl Default for ProbeUpload {
    fn default() -> Self {
        Self {
            count: 0,
            boxes: [[[0.0; 4]; 3]; MAX_PROBES],
            portal_count: 0,
            portals: [[[0.0; 4]; 3]; MAX_PORTALS],
            no_trace: false,
            proxy_count: 0,
            proxies: [[[0.0; 4]; 3]; MAX_PROXIES],
        }
    }
}

/// A single black cube, for a scene with no baked probes.
///
/// Black with alpha 0 -- the same "nothing here, use the sky" the baker writes
/// for a ray that hit nothing. So an unbaked level is not a special case in the
/// shader: it is a level where every probe direction is sky, which is true.
pub fn default_probe_cube(device: &Device) -> (TextureView, Sampler) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("default_probe_cube"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // HALF FLOAT, LINEAR. Not sRGB, and the difference is the whole reason
        // this format changed: eight-bit sRGB gave a dim room's walls three
        // distinct values across a whole cube face, so every reflection that
        // fell back to a probe banded into contour rings. The baker now writes
        // sixteen bits through a square root and the loader hands the decoded
        // linear radiance straight in, so there is no transfer curve on this
        // texture for the GPU to apply.
        format: TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("default_probe_cube_view"),
        dimension: Some(TextureViewDimension::CubeArray),
        ..Default::default()
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("probe_sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    (view, sampler)
}

/// A one-texel, all-zero distance cube array and the NEAREST sampler every
/// probe-distance read uses. Zero is "no distance baked", which the shader
/// answers with the box projection it has always used.
pub fn default_probe_depth(device: &Device) -> (TextureView, Sampler) {
    let tex = device.create_texture(&probe_depth_descriptor(1, 1));
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("default_probe_depth_view"),
        dimension: Some(TextureViewDimension::CubeArray),
        ..Default::default()
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("probe_depth_sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    (view, sampler)
}

/// The probe depth cube array: one level, per texel the plane of the surface
/// it saw as four half floats (`reflection_probe::decode_probe_depth`), a
/// layer per cube of the radiance array it rides beside.
pub fn probe_depth_descriptor(res: u32, probes: u32) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("probe_depth_array"),
        size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 6 * probes.max(1) },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }
}

/// Upload one probe's depth planes -- six stacked faces of four half floats
/// a texel, as `reflection_probe::decode_probe_depth` returns them -- into
/// `layer`.
pub fn write_probe_depth_layer(queue: &Queue, tex: &wgpu::Texture, layer: u32, res: u32, depth: &[u16]) {
    let bytes: Vec<u8> = depth.iter().flat_map(|h| h.to_le_bytes()).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: layer * 6 },
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8 * res), rows_per_image: Some(res) },
        wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 6 },
    );
}

/// The probe cube array's texture descriptor.
///
/// Pulled out so `the_cube_is_declared_with_the_levels_the_shader_asks_for`
/// can check it. That matters more than it looks: a cube declared with ONE
/// level does not fail when the shader asks for level 6 -- the sample is
/// CLAMPED back to level 0, which is silently the exact bug the mip chain was
/// added to fix. Nothing about the picture says the chain is missing.
pub fn probe_cube_descriptor(res: u32, probes: u32) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("probe_cube_array"),
        size: wgpu::Extent3d {
            width: res,
            height: res,
            // Six faces per probe. The depth may follow the level: a cube
            // ARRAY binding says nothing about how many cubes it holds, so the
            // bind group layout is the same whatever this is.
            depth_or_array_layers: 6 * probes.max(1),
        },
        // THE WHOLE CHAIN. A rough surface reads a blurred level; without
        // these the cube has only the sharpest one and every roughness reads
        // the same razor-sharp texel.
        mip_level_count: probe_mip_levels(res),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // HALF FLOAT, LINEAR. Not sRGB, and the difference is the whole reason
        // this format changed: eight-bit sRGB gave a dim room's walls three
        // distinct values across a whole cube face, so every reflection that
        // fell back to a probe banded into contour rings. The baker now writes
        // sixteen bits through a square root and the loader hands the decoded
        // linear radiance straight in, so there is no transfer curve on this
        // texture for the GPU to apply.
        format: TextureFormat::Rgba16Float,
        // COPY_SRC so a test can read a mip level back and prove the chain
        // actually landed. Sampling through a shader cannot tell "the level is
        // black" apart from "the shader asked for the wrong level".
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }
}

/// How many mip levels a probe cube of this resolution carries.
///
/// Down to a single texel, because that level is the average of everything the
/// probe saw and a fully rough surface should reflect exactly that.
pub fn probe_mip_levels(resolution: u32) -> u32 {
    32 - resolution.max(1).leading_zeros()
}

/// IEEE binary16 to f32. See `f32_to_f16` for why these live here.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = -14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (((e + 127) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | (((exp - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// f32 to IEEE binary16.
///
/// A SECOND COPY of the engine's converter, deliberately. `space_soup` ships to
/// crates.io and must not depend on `space_soup_engine`; the alternative to
/// twenty duplicated lines is a dependency that would drag the whole engine
/// into a published renderer.
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mant = bits & 0x007f_ffff;
    if exp > 15 {
        return sign | if exp == 128 && mant != 0 { 0x7e00 } else { 0x7c00 };
    }
    if exp < -14 {
        let shift = (-14 - exp) as u32;
        // A half float's smallest subnormal is 2^-24: ten steps below the
        // normal range. Past that the value is zero -- and the old bound of 24
        // let the shift below reach 37, which PANICS in a debug build and in
        // release silently wraps, turning a near-black probe texel into
        // garbage (found 2026-09-23, rendering nine probes off the headset).
        if shift > 10 {
            return sign;
        }
        return sign | ((mant | 0x0080_0000) >> (shift + 13)) as u16;
    }
    sign | (((exp + 15) as u16) << 10) | ((mant >> 13) as u16)
}

#[cfg(test)]
mod f16_tests {
    #[test]
    fn tiny_values_become_zero_or_a_subnormal_never_garbage() {
        for v in [1e-8f32, 1e-9, 1e-20, f32::MIN_POSITIVE] {
            assert_eq!(super::f32_to_f16(v), 0, "{v}");
        }
        // 2^-20 is a half-float subnormal: mantissa 1 << 4.
        assert_eq!(super::f32_to_f16(2f32.powi(-20)), 1 << 4);
        assert_eq!(super::f32_to_f16(2f32.powi(-24)), 1);
        assert_eq!(super::f32_to_f16(1.0), 0x3c00);
    }
}

/// The average radiance a probe photographed, as luminance.
///
/// Over the texels it actually SAW, weighted by coverage: sky texels carry
/// alpha 0 and the renderer fills them with its own sky term, so they are not
/// part of the photograph's brightness. A probe that saw nothing returns 0,
/// which the shader reads as "unknown" and leaves unnormalised.
pub fn probe_mean_radiance(faces: &[u8], res: u32) -> f32 {
    let res = res.max(1);
    if faces.len() < (res * res * 6 * 8) as usize {
        return 0.0;
    }
    let cube = decode_probe_faces(faces, res);
    let (mut sum, mut weight) = (0.0f64, 0.0f64);
    for [r, g, b, a] in cube.texels {
        let a = a.clamp(0.0, 1.0) as f64;
        sum += (0.2126 * r + 0.7152 * g + 0.0722 * b) as f64 * a;
        weight += a;
    }
    if weight > 0.0 {
        (sum / weight) as f32
    } else {
        0.0
    }
}

/// A probe's texels as linear RGBA, or `None` if the buffer is short -- for
/// metering (`exposure::EyeAdaptation`), which wants them without the cube.
pub(crate) fn decode_probe_texels(faces: &[u8], res: u32) -> Option<Vec<[f32; 4]>> {
    let res = res.max(1);
    if faces.len() < (res * res * 6 * 8) as usize {
        return None;
    }
    Some(decode_probe_faces(faces, res).texels)
}

/// One probe's six faces of RGBA half floats, decoded to linear f32.
fn decode_probe_faces(src: &[u8], res: u32) -> super::probe_prefilter::CubeLevel {
    let n = (res * res * 6) as usize;
    let texels = (0..n)
        .map(|t| {
            let mut px = [0.0f32; 4];
            for (c, v) in px.iter_mut().enumerate() {
                let i = (t * 4 + c) * 2;
                *v = f16_to_f32(u16::from_le_bytes([src[i], src[i + 1]]));
            }
            px
        })
        .collect();
    super::probe_prefilter::CubeLevel { res, texels }
}

/// A cube level back to RGBA half-float bytes, faces in cube order.
fn encode_probe_level(level: &super::probe_prefilter::CubeLevel) -> Vec<u8> {
    let mut out = Vec::with_capacity(level.texels.len() * 8);
    for px in &level.texels {
        for v in px {
            out.extend_from_slice(&f32_to_f16(*v).to_le_bytes());
        }
    }
    out
}

/// Upload a scene's baked probes as one cube-array texture.
///
/// `faces` is one entry per probe: six square RGBA faces concatenated in cube
/// order, all the same resolution. Probes past [`MAX_PROBES`] are dropped --
/// the uniform has nowhere to describe them and the fragment loop would never
/// reach them, so uploading their pixels would only cost memory.
///
/// The array is always `MAX_PROBES` layers deep whatever the scene has, so the
/// bind group layout never depends on level content.
pub fn upload_probe_cubes(
    device: &Device,
    queue: &Queue,
    resolution: u32,
    faces: &[&[u8]],
) -> TextureView {
    let tex = upload_probe_cube_texture(device, queue, resolution, faces);
    tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("probe_cube_array_view"),
        dimension: Some(TextureViewDimension::CubeArray),
        ..Default::default()
    })
}

/// The same upload, returning the TEXTURE.
///
/// Split out so a test can copy a mip level back off the GPU. Everything
/// shipping goes through `upload_probe_cubes` and gets the view.
pub fn upload_probe_cube_texture(
    device: &Device,
    queue: &Queue,
    resolution: u32,
    faces: &[&[u8]],
) -> wgpu::Texture {
    let res = resolution.max(1);
    // AS MANY LAYERS AS THE LEVEL HAS, up to what the device allows. The
    // renderer itself streams through a fixed pool (`probe_stream`); this
    // all-at-once form serves tests and the offline harness, and says so when
    // it has to drop a probe rather than doing it quietly.
    let fit = probe_layers_allowed(device);
    if faces.len() > fit as usize {
        log::warn!(
            "probe cube array: {} probes, room for {fit} on this device; the rest are dropped",
            faces.len(),
        );
    }
    let layers = (faces.len() as u32).min(fit).max(1);
    let tex = device.create_texture(&probe_cube_descriptor(res, layers));
    for (i, probe) in faces.iter().take(layers as usize).enumerate() {
        if let Some(chain) = prefilter_probe(probe, res) {
            write_probe_layer(queue, &tex, i as u32, res, &chain);
        }
    }
    tex
}

/// How many cubes one array may hold on `device`: six array layers each,
/// against the limit the device was CREATED with -- not the hardware's own,
/// which is only reachable if it was asked for. See `vulkan_interop`.
pub fn probe_layers_allowed(device: &Device) -> u32 {
    (device.limits().max_texture_array_layers / 6).max(1)
}

/// One probe's whole mip chain, GGX-prefiltered and encoded as the texture
/// wants it: one byte vector per level, six faces each. `None` for faces of the
/// wrong size.
///
/// THE WHOLE CHAIN, generated here on the CPU.
///
/// wgpu does not build mips for you. A texture declared with levels it was
/// never given reads as EMPTY at those levels -- and since roughness selects
/// the level, that means rough surfaces reflect nothing while smooth ones look
/// fine. It is invisible in a diff and looks like a material problem on screen.
/// `every_mip_level_receives_its_data` reads a level back off the GPU so that
/// cannot happen quietly again.
///
/// PREFILTERED WITH THE GGX LOBE, not box-averaged. See `probe_prefilter`: a
/// box chain kept bright features as squares at every level and was the source
/// of the light rectangles on the walls.
pub fn prefilter_probe(faces: &[u8], res: u32) -> Option<Vec<Vec<u8>>> {
    let per_face = (res * res * 8) as usize;
    if faces.len() < per_face * 6 {
        return None;
    }
    let chain = super::probe_prefilter::prefiltered_chain(decode_probe_faces(faces, res));
    Some(chain.iter().take(probe_mip_levels(res) as usize).map(encode_probe_level).collect())
}

/// Write a prefiltered chain into cube `layer` of `tex`.
pub fn write_probe_layer(queue: &Queue, tex: &wgpu::Texture, layer: u32, res: u32, chain: &[Vec<u8>]) {
    for (level, level_data) in chain.iter().enumerate() {
        let level_res = (res >> level).max(1);
        let per = (level_res * level_res * 8) as usize;
        for face in 0..6usize {
            let start = face * per;
            let Some(slice) = level_data.get(start..start + per) else { continue };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer * 6 + face as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                slice,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8 * level_res),
                    rows_per_image: Some(level_res),
                },
                wgpu::Extent3d { width: level_res, height: level_res, depth_or_array_layers: 1 },
            );
        }
    }
}

impl UniformBuffer {
    /// Point the scene bind group at a new probe cube array.
    ///
    /// Rebuilds the bind group rather than the layout: the layout is baked into
    /// every pipeline in the renderer and must not depend on whether a level
    /// happens to have probes. Only the resource behind binding 5 changes.
    #[allow(clippy::too_many_arguments)]
    pub fn rebind_probes(
        &mut self,
        device: &Device,
        lights: &LightsUniform,
        sun_shadow_view: &TextureView,
        sun_dynamic_view: &TextureView,
        spot_shadow_view: &TextureView,
        shadow_sampler: &Sampler,
        probe_view: &TextureView,
        probe_sampler: &Sampler,
        probes: ProbeUpload,
    ) {
        self.bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("uniform_bg"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: self.buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: lights.buffer().as_entire_binding() },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(sun_shadow_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(shadow_sampler),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(spot_shadow_view),
                },
                BindGroupEntry { binding: 5, resource: BindingResource::TextureView(probe_view) },
                BindGroupEntry { binding: 6, resource: BindingResource::Sampler(probe_sampler) },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::TextureView(sun_dynamic_view),
                },
                BindGroupEntry { binding: 8, resource: BindingResource::TextureView(&self.probe_depth_view) },
                BindGroupEntry { binding: 9, resource: BindingResource::Sampler(&self.probe_depth_sampler) },
            ],
        });
        self.probes = probes;
    }
}

#[cfg(test)]
mod ground_irradiance_tests {
    //! The CPU's shortcut must equal the shader's full evaluation.
    //!
    //! `sky_ground_irradiance` drops six of the nine harmonics because they
    //! vanish straight up. That is a real saving and an easy place for a
    //! typo -- and the symptom would be a reflected ground of the wrong colour,
    //! which looks like an art problem rather than an arithmetic one.
    use super::*;

    /// The shader's `sky_irradiance`, in Rust, for an arbitrary direction.
    fn shader_sky_irradiance(sh: &[[f32; 4]; 9], n: [f32; 3]) -> [f32; 3] {
        let (x, y, z) = (n[0], n[1], n[2]);
        let b = [
            0.282095,
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3.0 * z * z - 1.0),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ];
        let a = [1.0, 0.6666667, 0.6666667, 0.6666667, 0.25, 0.25, 0.25, 0.25, 0.25];
        let mut e = [0.0f32; 3];
        for i in 0..9 {
            for c in 0..3 {
                e[c] += sh[i][c] * b[i] * a[i];
            }
        }
        e
    }

    fn sample_sky() -> [[f32; 4]; 9] {
        // Deliberately asymmetric, so a dropped term or a sign error shows.
        let mut sh = [[0.0f32; 4]; 9];
        for (i, row) in sh.iter_mut().enumerate() {
            let f = i as f32;
            *row = [0.9 - 0.07 * f, 0.5 + 0.03 * f, 0.2 + 0.05 * f, 0.0];
        }
        sh
    }

    #[test]
    fn the_cpu_shortcut_matches_the_full_series_straight_up() {
        let sh = sample_sky();
        let fast = sky_ground_irradiance(&sh);
        let full = shader_sky_irradiance(&sh, [0.0, 1.0, 0.0]);
        for c in 0..3 {
            assert!(
                (fast[c] - full[c].max(0.0)).abs() < 1e-5,
                "channel {c}: shortcut {} vs full evaluation {}",
                fast[c],
                full[c],
            );
        }
    }

    /// The clamp is not cosmetic: a truncated harmonic series rings, and a
    /// negative ground would SUBTRACT light from every downward reflection.
    #[test]
    fn a_ringing_sky_cannot_produce_negative_ground() {
        let mut sh = [[0.0f32; 4]; 9];
        sh[0] = [0.05, 0.05, 0.05, 0.0];
        sh[1] = [-3.0, -3.0, -3.0, 0.0];
        let g = sky_ground_irradiance(&sh);
        assert!(g.iter().all(|v| *v >= 0.0), "ground came out negative: {g:?}");
    }

    #[test]
    fn a_black_sky_has_no_ground() {
        assert_eq!(sky_ground_irradiance(&[[0.0; 4]; 9]), [0.0; 3]);
    }
}

/// What the probe cap actually costs, so raising it again is a decision with
/// numbers rather than a guess.
#[cfg(test)]
mod probe_capacity_tests {
    use super::*;

    /// One 64x64 RGBA16F cube face.
    const FACE_BYTES: usize = 64 * 64 * 8;

    #[test]
    fn the_cube_array_stays_small_enough_to_be_free() {
        let bytes = FACE_BYTES * 6 * MAX_PROBES;
        assert!(
            bytes < 8 * 1024 * 1024,
            "the probe cube array is now {} MB, which is no longer a rounding \
             error on a headset; either drop the cap or drop the resolution",
            bytes / (1024 * 1024),
        );
    }

    /// The uniform has to stay inside the guaranteed binding size. wgpu's floor
    /// for `max_uniform_buffer_binding_size` is 64 KiB.
    #[test]
    fn the_uniform_still_fits_a_guaranteed_binding() {
        let size = std::mem::size_of::<Uniforms>();
        assert!(
            size <= 64 * 1024,
            "the scene uniform is {size} bytes; past 64 KiB it stops being \
             portable and starts depending on what the device reports",
        );
    }

    /// Every box slot the shader may index has storage behind it. A cap raised
    /// in one place and not the other reads uninitialised boxes.
    #[test]
    fn every_slot_the_shader_can_index_exists() {
        let probes = ProbeUpload { count: MAX_PROBES as u32, ..Default::default() };
        assert_eq!(probes.boxes.len(), MAX_PROBES);
        assert!(
            probes.count as usize <= probes.boxes.len(),
            "the shader would walk {} probes with {} boxes uploaded",
            probes.count,
            probes.boxes.len(),
        );
    }

    /// More probes than the cap are DROPPED, not wrapped into slot zero.
    #[test]
    fn probes_past_the_cap_are_dropped_rather_than_aliased() {
        let mut probes = ProbeUpload { count: (MAX_PROBES + 5) as u32, ..Default::default() };
        probes.count = probes.count.min(MAX_PROBES as u32);
        assert_eq!(probes.count as usize, MAX_PROBES);
    }
}

/// Choose which baked probes occupy the shader's slots this frame.
///
/// WHY THIS IS NOT "the first sixteen".
///
/// A level bakes a probe every couple of metres, because that is what makes a
/// probe's photograph true for the volume it claims -- see `MAX_CELL_EDGE` in
/// the baker. A level of any size therefore has more probes than the
/// per-fragment loop can afford to walk, and which sixteen matter depends
/// entirely on where the player is standing.
///
/// The rule is: the cell the player is INSIDE always gets a slot, then the
/// nearest cells that are in view, then the nearest cells that are not. The
/// last group is not wasted -- a probe behind the player still reflects in a
/// wall they are facing.
///
/// `volumes` is `(layer, centre, min, max)` for every baked probe.
pub fn select_resident_probes(
    volumes: &[(u32, Vec3, Vec3, Vec3)],
    player: Vec3,
    visible: impl Fn(Vec3, Vec3) -> bool,
) -> ProbeUpload {
    let mut scored: Vec<(u8, f32, usize)> = volumes
        .iter()
        .enumerate()
        .map(|(i, (_, centre, min, max))| {
            let inside = player.cmpge(*min).all() && player.cmple(*max).all();
            // Rank, then distance within a rank. Lower is better.
            let rank = if inside {
                0
            } else if visible(*min, *max) {
                1
            } else {
                2
            };
            (rank, centre.distance_squared(player), i)
        })
        .collect();
    scored.sort_by(|a, b| {
        a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut upload = ProbeUpload {
        count: scored.len().min(MAX_PROBES) as u32,
        ..Default::default()
    };
    for (slot, &(_, _, i)) in scored.iter().take(MAX_PROBES).enumerate() {
        let (layer, centre, min, max) = volumes[i];
        upload.set(slot, layer, centre, min, max);
    }
    upload
}

#[cfg(test)]
mod residency_tests {
    use super::*;

    fn cell(layer: u32, x: f32) -> (u32, Vec3, Vec3, Vec3) {
        let min = Vec3::new(x, 0.0, 0.0);
        let max = min + Vec3::new(2.0, 2.0, 2.0);
        (layer, (min + max) * 0.5, min, max)
    }

    /// THE CELL THE PLAYER IS STANDING IN ALWAYS GETS A SLOT.
    ///
    /// It is the one whose photograph is most nearly true for them, and it is
    /// also the one a frustum test is most likely to reject -- the player is
    /// inside it, so it surrounds the camera.
    #[test]
    fn the_cell_around_the_player_is_never_dropped() {
        // The player's cell is LARGE, so its centre is further from them than
        // a dozen small cells nearby. Distance alone would rank it last; only
        // containment puts it first. An earlier version of this test made the
        // player's cell the nearest one too, so it passed with containment
        // disabled and proved nothing.
        let player = Vec3::new(1.0, 1.0, 1.0);
        let big = (99u32, Vec3::new(50.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 0.0), Vec3::new(100.0, 2.0, 2.0));
        let mut vols: Vec<_> = (0..MAX_PROBES as u32).map(|i| cell(i, 3.0 + i as f32 * 3.0)).collect();
        vols.push(big);

        let up = select_resident_probes(&vols, player, |_, _| true);
        assert_eq!(
            up.layer(0), 99,
            "the cell the player is standing in lost its slot to nearer cells",
        );
        // And it survives a frustum that rejects everything, which is the case
        // that matters: the player is inside it, so it surrounds the camera.
        let up = select_resident_probes(&vols, player, |_, _| false);
        assert_eq!(up.layer(0), 99, "the player's own cell was frustum culled");
    }

    /// Visible cells beat invisible ones; among equals, nearer wins.
    #[test]
    fn visible_and_near_are_preferred() {
        let vols = vec![
            cell(0, 40.0),  // far, and visible
            cell(1, 10.0),  // near, not visible
            cell(2, 20.0),  // mid, visible
        ];
        // Only cells past x = 15 are in view.
        let up = select_resident_probes(&vols, Vec3::ZERO, |min, _| min.x > 15.0);
        assert_eq!(up.layer(0), 2, "the nearer visible cell should come first");
        assert_eq!(up.layer(1), 0, "the further visible cell should come second");
        assert_eq!(up.layer(2), 1, "the invisible cell should come last");
    }

    /// LAYERS ARE NOT SLOTS. A probe in slot 0 may live anywhere in the array,
    /// and the shader has to read the layer rather than the loop index.
    #[test]
    fn the_layer_survives_into_the_slot() {
        let vols = vec![cell(7, 0.0), cell(23, 50.0)];
        let up = select_resident_probes(&vols, Vec3::new(1.0, 1.0, 1.0), |_, _| true);
        assert_eq!(up.layer(0), 7);
        assert_eq!(up.layer(1), 23);
        // And the box travelled with it.
        assert_eq!(up.boxes[0][1][0], 0.0);
        assert_eq!(up.boxes[1][1][0], 50.0);
    }

    /// Never more slots than the shader walks.
    #[test]
    fn the_slot_count_never_exceeds_the_loop() {
        let vols: Vec<_> = (0..(2 * MAX_PROBES) as u32).map(|i| cell(i, i as f32 * 3.0)).collect();
        let up = select_resident_probes(&vols, Vec3::ZERO, |_, _| true);
        assert_eq!(up.count as usize, MAX_PROBES);
        assert!(up.count as usize <= up.boxes.len());
    }

    /// A level with no probes is not a panic and claims no slots.
    #[test]
    fn no_probes_is_no_slots() {
        let up = select_resident_probes(&[], Vec3::ZERO, |_, _| true);
        assert_eq!(up.count, 0);
    }
}

/// The probe cube's mip chain: what makes a rough surface reflect a blurred
/// world instead of one razor-sharp texel.
#[cfg(test)]
mod probe_mip_tests {
    use super::*;

    #[test]
    fn a_64px_cube_has_levels_down_to_one_texel() {
        assert_eq!(probe_mip_levels(64), 7);
        assert_eq!(probe_mip_levels(1), 1);
        assert_eq!(probe_mip_levels(0), 1, "a zero-size probe must not ask for zero levels");
    }

    /// THE CUBE IS DECLARED WITH THE LEVELS THE SHADER ASKS FOR.
    ///
    /// A cube with one level does not error when the shader samples level 6:
    /// the sample is clamped back to level 0, which is exactly the sharp-texel
    /// bug the chain exists to remove -- silently, with nothing in the picture
    /// to say the chain is absent. Deliberately reverting the descriptor to
    /// `1` turns this red and nothing else.
    #[test]
    fn the_cube_is_declared_with_the_levels_the_shader_asks_for() {
        for res in [1u32, 8, 32, 64, 128] {
            let d = probe_cube_descriptor(res, 4);
            assert_eq!(
                d.mip_level_count,
                probe_mip_levels(res),
                "a {res}px probe cube was declared with {} level(s); a rough \
                 surface would clamp back to the sharpest texel and glint",
                d.mip_level_count,
            );
        }
        // And the level the shader caps at exists in the default 128px cube.
        assert!(probe_mip_levels(128) as f32 > 7.0);
    }

    /// The shader's roughness ramp must not ask for a level the cube lacks --
    /// sampling past the last level is undefined and reads as nothing.
    #[test]
    fn the_roughness_ramp_stays_inside_the_chain() {
        // 128: the engine's DEFAULT_PROBE_RESOLUTION, which this crate cannot name.
        let levels = probe_mip_levels(128);
        let max_lod = (levels - 1) as f32;
        let code = super::super::lights::wgsl_lights_block(0, 1);
        assert!(
            code.contains(&format!("const PROBE_MAX_LOD: f32 = {max_lod:.1};")),
            "the shader caps probe LOD somewhere other than the top of a \
             {levels}-level chain",
        );
    }

    /// AVERAGING IS IN LINEAR LIGHT, which is only true because the probe
    /// stopped being sRGB. The source pyramid the GGX samples read from is a
    /// linear mean of the decoded half floats; a naive average of sRGB bytes
    /// would not be.
    #[test]
    fn the_source_pyramid_averages_radiance() {
        let res = 2u32;
        let mut src = Vec::new();
        // Six faces, 2x2, values 0, 1, 2, 3 in red; alpha 1.
        for _ in 0..6 {
            for v in [0.0f32, 1.0, 2.0, 3.0] {
                for c in 0..4 {
                    let x = if c == 3 { 1.0 } else if c == 0 { v } else { 0.0 };
                    src.extend_from_slice(&f32_to_f16(x).to_le_bytes());
                }
            }
        }
        let pyramid = super::super::probe_prefilter::box_pyramid(decode_probe_faces(&src, res));
        assert_eq!(pyramid.len(), 2, "a 2x2 cube did not halve to 1x1");
        assert_eq!(pyramid[1].texels.len(), 6);
        let [r, _, _, a] = pyramid[1].texels[0];
        assert!((r - 1.5).abs() < 1e-2, "mean of 0,1,2,3 came back as {r}, not 1.5");
        assert!((a - 1.0).abs() < 1e-2, "coverage was not preserved: {a}");
        // And the bytes survive the trip back out.
        let bytes = encode_probe_level(&pyramid[1]);
        assert_eq!(bytes.len(), 6 * 8);
        assert!((f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])) - 1.5).abs() < 1e-2);
    }

    /// Half-float round trip, since the mip chain runs every texel through it
    /// once per level.
    #[test]
    fn half_floats_survive_the_chain() {
        for &v in &[0.0f32, 1e-4, 0.001, 0.5, 1.0, 3.7, 60.0] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() <= v * 1e-3 + 6e-8, "{v} became {back}");
        }
    }
}

/// Does the mip chain actually reach the GPU?
///
/// Sampling it through a shader could not answer this: a black result is
/// equally consistent with "the level is empty" and "the shader asked for the
/// wrong level". Copying a level straight back off the texture separates them.
#[cfg(test)]
mod probe_mip_upload_tests {
    use super::*;

    fn headless_gpu() -> Option<(Device, Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            apply_limit_buckets: false,
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()
    }

    /// A uniform cube must read back as that same value at EVERY level.
    ///
    /// Averaging a constant gives the constant, so any level that comes back
    /// dark is a level that never received its data.
    #[test]
    fn every_mip_level_receives_its_data() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        const RES: u32 = 8;
        let one = f32_to_f16(1.0).to_le_bytes();
        let texel: Vec<u8> = [one, one, one, one].concat();
        let faces: Vec<u8> =
            texel.iter().copied().cycle().take((RES * RES * 8 * 6) as usize).collect();
        let tex = upload_probe_cube_texture(&device, &queue, RES, &[&faces]);

        for level in 0..probe_mip_levels(RES) {
            let side = (RES >> level).max(1);
            // 256-byte row alignment is required for texture-to-BUFFER copies,
            // unlike the writes going the other way.
            let row = ((side * 8 + 255) / 256) * 256;
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("probe_mip_readback"),
                size: (row * side) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(side),
                    },
                },
                wgpu::Extent3d { width: side, height: side, depth_or_array_layers: 1 },
            );
            queue.submit(Some(enc.finish()));
            let slice = buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            let data = slice.get_mapped_range().unwrap();
            let first = u16::from_le_bytes([data[0], data[1]]);
            assert_ne!(
                first, 0,
                "probe mip level {level} ({side}x{side}) came back EMPTY. A cube \
                 of a constant averages to that constant at every level, so this \
                 level never received its upload -- and a rough surface, which \
                 samples exactly these levels, reflects nothing at all.",
            );
            drop(data);
            buf.unmap();
        }
    }
}

/// A probe's average brightness, and how it reaches its shader slot.
#[cfg(test)]
mod probe_brightness_tests {
    use super::*;

    fn cube(res: u32, texel: [f32; 4], sky_every: usize) -> Vec<u8> {
        let n = (res * res * 6) as usize;
        (0..n)
            .flat_map(|i| {
                let t = if sky_every > 0 && i % sky_every == 0 { [9.0, 9.0, 9.0, 0.0] } else { texel };
                t.into_iter().flat_map(|v| f32_to_f16(v).to_le_bytes())
            })
            .collect()
    }

    /// An evenly lit room averages to its own luminance.
    #[test]
    fn an_even_probe_averages_to_its_luminance() {
        let m = probe_mean_radiance(&cube(4, [0.2, 0.2, 0.2, 1.0], 0), 4);
        assert!((m - 0.2).abs() < 1e-3, "mean came back {m}");
    }

    /// SKY texels are not part of the photograph -- the renderer fills them with
    /// its own sky term -- so a bright value behind alpha 0 must not raise it.
    #[test]
    fn sky_texels_do_not_count() {
        let m = probe_mean_radiance(&cube(4, [0.2, 0.2, 0.2, 1.0], 3), 4);
        assert!((m - 0.2).abs() < 1e-3, "sky behind alpha 0 moved the mean to {m}");
    }

    /// A probe that saw nothing reports UNKNOWN, which leaves it unnormalised.
    #[test]
    fn a_probe_of_only_sky_is_unknown() {
        assert_eq!(probe_mean_radiance(&cube(2, [0.0; 4], 1), 2), 0.0);
        assert_eq!(probe_mean_radiance(&[], 2), 0.0, "a short buffer is not a probe");
    }

    /// Residency reorders slots; the brightness must follow the LAYER.
    #[test]
    fn brightness_follows_the_layer_not_the_slot() {
        let mut u = ProbeUpload { count: 2, ..Default::default() };
        u.set(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::ONE);
        u.set(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::ONE);
        u.fill_brightness(&[0.25, 0.75]);
        assert_eq!(u.boxes[0][1][3], 0.75, "slot 0 holds layer 1");
        assert_eq!(u.boxes[1][1][3], 0.25, "slot 1 holds layer 0");
        // And a slot past `count` is left alone.
        assert_eq!(u.boxes[2][1][3], 0.0);
    }
}

#[cfg(test)]
mod dense_room_tests {
    use super::*;

    fn upload(volumes: &[u32]) -> ProbeUpload {
        let mut u = ProbeUpload { count: volumes.len() as u32, ..Default::default() };
        for (slot, &v) in volumes.iter().enumerate() {
            u.set(slot, slot as u32, Vec3::splat(slot as f32), Vec3::ZERO, Vec3::ONE);
            u.set_volume(slot, v);
        }
        u
    }

    /// What the shader does with the tables: every slot of `room`, in order.
    fn walk(tables: &[[f32; 4]; ROOM_TABLE_ROWS], room: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut s = tables[room / 4][room % 4];
        while s >= 0.0 {
            out.push(s as usize);
            let i = s as usize;
            s = tables[4 + i / 4][i % 4];
        }
        out
    }

    /// Each room's slots, in the order a scan of every slot visits them --
    /// which is what keeps the shader's ties broken as they were.
    #[test]
    fn a_room_walks_its_slots_in_scan_order() {
        let (dense, tables) = upload(&[7, 3, 7, 9, 3, 7]).dense_rooms();
        let rooms: Vec<f32> = (0..6).map(|s| dense.boxes[s][2][3]).collect();
        assert_eq!(rooms, vec![0.0, 1.0, 0.0, 2.0, 1.0, 0.0], "numbered by first appearance");
        assert_eq!(walk(&tables, 0), vec![0, 2, 5]);
        assert_eq!(walk(&tables, 1), vec![1, 4]);
        assert_eq!(walk(&tables, 2), vec![3]);
        assert!(walk(&tables, 3).is_empty(), "no fourth room");
        // Everything but the room numbers is untouched.
        let raw = upload(&[7, 3, 7, 9, 3, 7]);
        for s in 0..6 {
            assert_eq!(dense.boxes[s][0], raw.boxes[s][0]);
            assert_eq!(dense.boxes[s][1], raw.boxes[s][1]);
            assert_eq!(dense.boxes[s][2][..3], raw.boxes[s][2][..3]);
        }
    }

    /// Doorways and proxies take the same numbers, and a room with no
    /// resident photograph takes one that matches no slot and no table entry.
    #[test]
    fn portals_and_proxies_are_renumbered_alike() {
        let mut u = upload(&[7, 3]);
        let portal = |low, high| ProbePortal { min: Vec3::ZERO, max: Vec3::ONE, axis: 0, low, high, wall: None };
        u.set_portals(&[portal(7, 3), portal(3, 42)], Vec3::ZERO, &[7, 3]);
        let proxy = |volume| ProbeProxy { centre: Vec3::ZERO, half_size: Vec3::ONE, rotation: Quat::IDENTITY, volume, solid: true };
        u.set_proxies(&[proxy(3)], Vec3::ZERO, &[7, 3]);
        let (dense, _) = u.dense_rooms();
        let sides: Vec<(f32, f32)> = (0..2).map(|p| (dense.portals[p][1][3], dense.portals[p][2][0])).collect();
        assert!(sides.contains(&(0.0, 1.0)), "{sides:?}");
        assert!(sides.contains(&(1.0, NO_RESIDENT_ROOM)), "{sides:?}");
        assert_eq!(dense.proxies[0][0][3], 1.0);
        assert!(NO_RESIDENT_ROOM >= 16.0, "outside the tables, so the shader's lookup finds nothing");
    }

    /// Each room's doorways and proxies, ascending, as the shader walks them:
    /// a doorway onward through whichever of its sides is the walked room.
    #[test]
    fn doorways_and_proxies_chain_by_room() {
        let mut u = upload(&[10, 20, 30]);
        let portal = |low, high| ProbePortal { min: Vec3::ZERO, max: Vec3::ONE, axis: 0, low, high, wall: None };
        // Nearest first: all at the origin, so they keep this order.
        u.set_portals(&[portal(10, 20), portal(20, 30), portal(10, 30), portal(30, 99)], Vec3::ZERO, &[10, 20, 30]);
        let proxy = |volume| ProbeProxy { centre: Vec3::ZERO, half_size: Vec3::ONE, rotation: Quat::IDENTITY, volume, solid: true };
        u.set_proxies(&[proxy(20), proxy(10), proxy(20), proxy(99)], Vec3::ZERO, &[10, 20, 30]);
        let (dense, t) = u.dense_rooms();
        let at = |row: usize, k: usize| t[row + k / 4][k % 4];
        let portals_of = |room: f32| {
            let mut out = Vec::new();
            let mut p = at(8, room as usize);
            while p >= 0.0 {
                let pi = p as usize;
                out.push(pi);
                let side = if dense.portals[pi][1][3] == room { 0 } else { 1 };
                p = at(12, 2 * pi + side);
            }
            out
        };
        assert_eq!(portals_of(0.0), vec![0, 2]);
        assert_eq!(portals_of(1.0), vec![0, 1]);
        assert_eq!(portals_of(2.0), vec![1, 2, 3]);
        let proxies_of = |room: usize| {
            let mut out = Vec::new();
            let mut i = at(16, room);
            while i >= 0.0 {
                out.push(i as usize);
                i = at(20, i as usize);
            }
            out
        };
        assert_eq!(proxies_of(1), vec![0, 2]);
        assert_eq!(proxies_of(0), vec![1]);
        assert!(proxies_of(2).is_empty());
        // A proxy in no resident room never reaches the upload (`set_proxies`
        // keeps only resident rooms'); a doorway with ONE resident side does,
        // and its other side is in no chain.
        assert_eq!(u.proxy_count, 3);
        assert_eq!(dense.portals[3][2][0], NO_RESIDENT_ROOM);
    }

    /// No probes, no rooms: every lookup finds nothing.
    #[test]
    fn an_empty_upload_has_empty_tables() {
        let (_, tables) = ProbeUpload::default().dense_rooms();
        assert!(tables.iter().flatten().all(|&v| v == -1.0));
    }
}
