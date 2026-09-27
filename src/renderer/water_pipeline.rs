//! Standing water.
//!
//! WHAT MAKES WATER READ AS WATER
//!
//! Not transparency, and not a blue tint. Three things, in order of how much
//! they matter on a headset:
//!
//! 1. **Fresnel.** Water is nearly clear looking straight down and nearly a
//!    mirror at a glancing angle. That single behaviour is most of the read: it
//!    is why a lake shows you its bed at your feet and the sky at the far bank,
//!    and a surface with constant opacity looks like tinted glass instead.
//! 2. **Moving normals.** A still plane reads as ice. The waves here are
//!    procedural rather than a scrolling normal map -- two sine ridges crossing
//!    at an angle -- because it costs no texture fetch and no memory, and at
//!    the scale water is usually seen from, the difference is invisible.
//! 3. **Depth colour.** Shallow water takes the colour of what is under it,
//!    deep water takes its own. Interpolating between two authored colours by
//!    depth gets the hue shift real water has and a single colour cannot.
//!
//! WHY THE DEPTH COMES FROM THE VERTEX
//!
//! The usual way to find how deep the water is at a pixel is to read the scene
//! depth buffer. On a tile GPU that means resolving it and reading it back,
//! which is the most expensive thing a transparent pass can ask for -- and it
//! buys nothing, because the ground under standing water does not move. The
//! depth is measured once when the surface is tessellated and interpolated
//! across the triangle. See `space_soup_engine::water::build_surface`.

use wgpu::*;

use crate::renderer::lights::wgsl_lights_block;

/// One vertex of the water surface, as the GPU sees it.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterVertex {
    pub position: [f32; 3],
    pub depth: f32,
}

/// Per-body optical properties.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterUniform {
    pub shallow: [f32; 4],
    pub deep: [f32; 4],
    /// x depth_scale, y shore_fade, z wave_scale, w wave_strength
    pub params: [f32; 4],
    /// x time seconds, y wave_speed, z opacity, w unused
    pub anim: [f32; 4],
}

pub struct WaterPipeline {
    pub pipeline: RenderPipeline,
    pub material_layout: BindGroupLayout,
}

impl WaterPipeline {
    pub fn new(

        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, camera_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    /// The same, drawing BOTH EYES in one pass. See `multiview::ViewMode`.
    pub fn new_stereo(

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
        let material_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("water_material_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("water_shader"),
            source: ShaderSource::Wgsl(view.shader(water_wgsl()).into()),
        });

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("water_pipeline_layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&material_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("water_pipeline"),
            layout: Some(&layout),
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
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                // NO BACK-FACE CULLING. A swimmer, or a camera that has waded
                // in, is under the surface looking up at its back -- and a
                // culled surface simply vanishes at the waterline, which reads
                // as the water having been deleted rather than entered.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                // TESTS but does not WRITE. Water is transparent: the ground
                // under it has to remain visible through it, and writing depth
                // would let one wave hide the wave behind it.
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });

        Self { pipeline, material_layout }
    }
}

fn water_wgsl() -> String {
    format!(
        r#"
{lights_block}

struct WaterMat {{
    shallow: vec4<f32>,
    deep: vec4<f32>,
    params: vec4<f32>,
    anim: vec4<f32>,
}}
@group(1) @binding(0) var<uniform> water: WaterMat;

struct VIn {{
    @location(0) pos: vec3<f32>,
    @location(1) depth: f32,
}}
struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) depth: f32,
    // TRUE world position, for the wave pattern only.
    //
    // `world_pos` is in the PLAYER's frame -- every bit of level geometry is,
    // so the tracked head pose can be used directly. Feeding that to the wave
    // function pins the ripples to the player: they slide across the lake as
    // you walk and rotate around you as you turn, which reads as the water
    // being a texture stuck to the camera. Exactly the bug the terrain layers
    // had, and for the same reason -- see `tex_pos` in terrain_pipeline.
    @location(2) tex_pos: vec3<f32>,
}}

@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    out.clip = cam_view_proj() * vec4<f32>(v.pos, 1.0);
    out.world_pos = v.pos;
    out.tex_pos = to_world_space(v.pos);
    out.depth = v.depth;
    return out;
}}

// Two crossing sine ridges, and their analytic derivatives as the normal.
//
// Analytic rather than sampled: the slope of a sine is a cosine, so the normal
// is exact and costs two more multiplies. Finite-differencing the height would
// need three evaluations and would alias at grazing angles -- which is the
// angle almost all water is seen at.
fn wave_normal(p: vec2<f32>, t: f32, scale: f32, strength: f32) -> vec3<f32> {{
    let k = 6.2831853 / max(scale, 0.05);
    // Crossing at roughly 60 degrees, with different periods, so the pattern
    // does not visibly repeat along either axis.
    let d1 = normalize(vec2<f32>(1.0, 0.35));
    let d2 = normalize(vec2<f32>(-0.4, 1.0));
    let p1 = dot(p, d1) * k + t;
    let p2 = dot(p, d2) * k * 0.73 - t * 0.85;
    let s1 = cos(p1) * strength;
    let s2 = cos(p2) * strength * 0.7;
    // Gradient of the height field, turned into a normal about +Y.
    let g = d1 * s1 + d2 * s2;
    return normalize(vec3<f32>(-g.x, 1.0, -g.y));
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let depth_scale = max(water.params.x, 0.01);
    let shore_fade = max(water.params.y, 0.001);
    let t = water.anim.x * water.anim.y;

    let n = wave_normal(in.tex_pos.xz, t, water.params.z, water.params.w);
    let view_dir = normalize(cam_pos() - in.world_pos);

    // FRESNEL, Schlick, with water's 0.02 normal-incidence reflectance. This is
    // the term that makes it read as water rather than as tinted glass.
    let cos_v = clamp(dot(n, view_dir), 0.0, 1.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - cos_v, 5.0);

    // What the surface reflects. `sky_irradiance` in the MIRROR direction is a
    // coarse stand-in for a real reflection probe -- it carries the sky's colour
    // and its bright side, which is most of what a lake shows back.
    let refl_dir = reflect(-view_dir, n);
    let reflected = sky_irradiance(refl_dir);

    // Body colour by depth.
    let k = clamp(in.depth / depth_scale, 0.0, 1.0);
    let body = mix(water.shallow.rgb, water.deep.rgb, k);

    // Direct light on the surface, so a lamp over a pond lights it.
    var lit = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < lights.count.x; i = i + 1u) {{
        lit = lit + light_contribution(lights.lights[i], in.world_pos, n, view_dir);
    }}

    let colour = mix(body * (sky_irradiance(n) + lit), reflected, fresnel);

    // Alpha: the authored head-on opacity, taken to fully opaque at grazing
    // angles by the same Fresnel term, and faded out where the water is only
    // millimetres deep so the shoreline is a gradient rather than a cut edge.
    let shore = clamp(in.depth / shore_fade, 0.0, 1.0);
    let alpha = clamp(mix(water.anim.z, 1.0, fresnel), 0.0, 1.0) * shore;
    return vec4<f32>(tonemap(colour), alpha);
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::lights::{Light, LightKind, LightsUniform};
    use crate::renderer::terrain_pipeline::tests::headless_gpu;
    use crate::renderer::uniforms::test_support::{scene_uniforms, TEST_EYE};
    use crate::renderer::uniforms::ShadowUpload;
    use crate::renderer::Color3;
    use wgpu::util::DeviceExt;

    fn uniform_for(depth_scale: f32, opacity: f32, wave_strength: f32) -> WaterUniform {
        WaterUniform {
            shallow: [0.35, 0.6, 0.55, 1.0],
            deep: [0.04, 0.16, 0.28, 1.0],
            params: [depth_scale, 0.35, 2.5, wave_strength],
            anim: [0.0, 0.35, opacity, 0.0],
        }
    }

    /// Draw one water triangle at a given depth and read the centre pixel.
    ///
    /// Renders rather than merely building the pipeline. Creating the pipeline
    /// proves the WGSL parses and nothing more -- it cannot tell whether the
    /// Fresnel term is the right way round, whether depth reaches the fragment
    /// stage, or whether alpha is being written at all.
    fn render_water(depth: f32, u: WaterUniform, eye: glam::Vec3) -> Option<[u8; 4]> {
        render_water_at(depth, u, eye, crate::renderer::uniforms::PlayerUpload::default())
    }

    /// The same, with the player standing somewhere other than the origin.
    ///
    /// The whole level is drawn in the PLAYER's frame, so anything the shader
    /// derives from a vertex position has to say which frame it means. Without
    /// this parameter no test here could move, and the wave pattern being
    /// pinned to the player was invisible to all of them.
    fn render_water_at(
        depth: f32,
        u: WaterUniform,
        eye: glam::Vec3,
        player: crate::renderer::uniforms::PlayerUpload,
    ) -> Option<[u8; 4]> {
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        lights.upload(
            &queue,
            &[Light {
                mask_channel: None,
                position: glam::Vec3::new(0.0, 6.0, 0.0),
                direction: glam::Vec3::NEG_Y,
                kind: LightKind::Point,
                color: Color3(255, 255, 255, 255),
                intensity: 4.0,
                range: 40.0,
                cone_angle_deg: 90.0,
                inner_cone_angle_deg: 0.0,
            }],
        );
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);
        uniforms.upload_scene(
            &queue,
            glam::Mat4::IDENTITY,
            eye,
            &ShadowUpload::disabled(),
            &crate::renderer::uniforms::SkyUpload::none(),
            &crate::renderer::uniforms::PostUpload::default(),
            &player,
        );

        let pipeline = WaterPipeline::new(&device, format, &uniforms.layout, 1);
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water_test_uniform"),
            contents: bytemuck::bytes_of(&u),
            usage: BufferUsages::UNIFORM,
        });
        let mat_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("water_test_mat"),
            layout: &pipeline.material_layout,
            entries: &[BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }],
        });

        // A triangle covering clip space. view_proj is identity here, so the
        // positions ARE clip coordinates -- but world_pos is what the shading
        // reads, so the plane still sits at a sensible world height.
        let v = |p: [f32; 3]| WaterVertex { position: p, depth };
        let verts = [v([-1.0, -1.0, 0.0]), v([3.0, -1.0, 0.0]), v([-1.0, 3.0, 0.0])];
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water_test_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water_test_ib"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        const SIZE: u32 = 8;
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
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("water_test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &tv,
                    depth_slice: None,
                    resolve_target: None,
                    // Black AND FULLY TRANSPARENT. `Color::BLACK` has a = 1.0,
                    // and with alpha blending the stored alpha is then
                    // `src + (1 - src) * 1`, which is 1 whatever the shader
                    // returned -- so every alpha assertion below passed or
                    // failed on the clear colour rather than on the water. A
                    // test that cannot observe the thing it names is worse than
                    // no test.
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
            pass.set_bind_group(1, &mat_bg, &[]);
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

    /// Straight down onto the surface.
    const OVERHEAD: glam::Vec3 = glam::Vec3::new(0.0, 8.0, 0.0);
    /// Almost along it, which is where Fresnel should take over.
    const GRAZING: glam::Vec3 = glam::Vec3::new(0.0, 0.05, 40.0);

    #[test]
    fn the_water_shader_compiles_and_draws() {
        // WGSL is validated at PIPELINE CREATION, not at cargo build: a bad
        // field accessor compiles clean and dies on the device. This test is
        // the only thing standing between that and the headset.
        let Some(px) = render_water(2.0, uniform_for(4.0, 0.72, 0.35), OVERHEAD) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(px[3] > 0, "water drew nothing at all: {px:?}");
    }

    #[test]
    fn deep_water_is_darker_than_shallow() {
        // The depth term. Without it every lake is one flat colour, which is
        // the single most obvious way water can look wrong.
        let u = uniform_for(4.0, 0.72, 0.0);
        let Some(shallow) = render_water(0.2, u, OVERHEAD) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let deep = render_water(8.0, u, OVERHEAD).unwrap();
        let lum = |p: [u8; 4]| p[0] as u32 + p[1] as u32 + p[2] as u32;
        assert!(
            lum(deep) < lum(shallow),
            "deep water must be darker than shallow: {deep:?} vs {shallow:?}",
        );
    }

    #[test]
    fn a_grazing_view_is_more_opaque_than_looking_straight_down() {
        // Fresnel, and the property that most makes water read as water: clear
        // at your feet, mirror at the far bank. Waves off, so this measures the
        // angle and not the surface noise.
        let u = uniform_for(4.0, 0.5, 0.0);
        let Some(down) = render_water(3.0, u, OVERHEAD) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let across = render_water(3.0, u, GRAZING).unwrap();
        assert!(
            across[3] > down[3],
            "a grazing view must be more opaque than an overhead one: \
             {across:?} vs {down:?}",
        );
    }

    #[test]
    fn the_shoreline_fades_out_rather_than_ending_in_a_line() {
        // Water millimetres deep must approach invisible, or it meets the
        // ground along a hard edge no real shore has -- and one that flickers,
        // because it is a geometric intersection sampled per pixel.
        let u = uniform_for(4.0, 0.72, 0.0);
        let Some(brink) = render_water(0.001, u, OVERHEAD) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let body = render_water(2.0, u, OVERHEAD).unwrap();
        assert!(brink[3] < 8, "the shoreline must fade to nothing, got alpha {}", brink[3]);
        assert!(body[3] > 64, "open water must be solidly visible, got alpha {}", body[3]);
    }

    #[test]
    fn the_wave_pattern_stays_put_when_the_player_walks() {
        // Level geometry is drawn in the PLAYER's frame, so a vertex position
        // is not a world position. Feeding it to the wave function pins the
        // ripples to the player -- they slide across the lake as you walk and
        // swing around you as you turn, which reads as the water being a
        // texture stuck to the camera rather than a surface in the world.
        //
        // The same mistake the terrain layers had. Every other test here stands
        // at the origin, so none of them could see it.
        use crate::renderer::uniforms::PlayerUpload;
        // Waves at full strength and a long walk, deliberately. At a modest
        // strength a 7 m step moved the sampled pixel by less than one byte --
        // the mechanism worked and the test could not see it.
        let mut u = uniform_for(4.0, 0.72, 1.0);
        u.params[2] = 0.6; // short wavelength, so a step spans several waves
        // OVERHEAD, not grazing: with a constant-band sky the reflection is the
        // same in every direction, so at a grazing view the wave normal barely
        // moves the pixel. Looking down, the direct light term dominates and
        // the normal is what decides it.
        let Some(here) = render_water_at(3.0, u, OVERHEAD, PlayerUpload::default()) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let walked = render_water_at(
            3.0,
            u,
            OVERHEAD,
            PlayerUpload { offset: glam::Vec3::new(101.3, 0.0, 57.7), yaw: 0.0 },
        )
        .unwrap();

        // NOTE THE DIRECTION, which is inverted from what the player sees.
        //
        // In the real renderer the geometry is rebuilt in the player's frame
        // every time they move, so a world-anchored wave holds still on screen.
        // This harness does not rebuild anything: its vertices are fixed clip
        // coordinates, pinned to the screen. So the same screen pixel now
        // stands over a DIFFERENT patch of world, and a world-anchored wave must
        // therefore change.
        //
        // Which makes this the discriminating form: a wave computed from the
        // player-frame position -- the bug -- cannot change here, because the
        // vertex it reads never moved. Asserting equality passed either way and
        // proved nothing.
        assert_ne!(
            here, walked,
            "the wave did not follow the world under a moving player, so it is \
             computed in the player's frame and will slide as they walk",
        );
    }

    #[test]
    fn waves_change_the_surface() {
        // The normal has to reach the shading. With the wave term dropped this
        // still compiles, still draws, and reads as ice.
        let flat = uniform_for(4.0, 0.72, 0.0);
        let mut wavy = flat;
        wavy.params[3] = 0.9;
        let Some(a) = render_water(3.0, flat, GRAZING) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let b = render_water(3.0, wavy, GRAZING).unwrap();
        assert_ne!(a, b, "wave strength changed nothing: the normal is not reaching the shading");
    }
}

// Scene shader sources, for `multiview::every_scene_shader_survives_the_multiview_transform`.
// Test-only: the gate has to see exactly the text each pipeline is built from,
// and nothing on a development machine can build a multiview pipeline to check.
#[cfg(test)]
pub fn water_shader_src() -> String {
    water_wgsl()
}
