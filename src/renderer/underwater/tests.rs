use super::*;
use crate::renderer::lights::LightsUniform;
use crate::renderer::uniforms::test_support::scene_uniforms;
use crate::renderer::uniforms::{PlayerUpload, PostUpload, ShadowUpload, SkyUpload};
use crate::renderer::water_pipeline::WaterOptics;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// The test_room sea and pond, as `water_render::build` makes them.
fn sea() -> WaterUniform {
    let lin = |c: [u8; 3]| c.map(|v| {
        let s = v as f32 / 255.0;
        if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    });
    WaterUniform::new(
        &WaterOptics { height: -1.4, shallow: lin([130, 205, 210]), deep: lin([12, 52, 72]), depth_scale: 2.0, shore_fade: 0.2, swash: 0.22, swell: 0.0 },
        &WaveParams::default(),
    )
}

fn pond() -> WaterUniform {
    let lin = |c: [u8; 3]| c.map(|v| {
        let s = v as f32 / 255.0;
        if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    });
    WaterUniform::new(
        &WaterOptics { height: -0.368, shallow: lin([150, 180, 130]), deep: lin([20, 34, 22]), depth_scale: 0.4, shore_fade: 0.12, swash: 0.0, swell: 0.015 },
        &WaveParams::default(),
    )
}

fn light(sun: Option<Vec3>) -> UnderUniform {
    UnderUniform::new(&UnderFrame {
        size: (64, 64),
        fov_y: 1.6,
        sun: sun.map(|s| (s.normalize(), Vec3::splat(3.0))),
        sky_down: Vec3::new(0.5, 0.6, 0.8),
        waterline: false,
        film_age: None,
        spheres: &[],
    })
}

#[test]
fn snells_window_is_ninety_seven_degrees_across() {
    let half = window_half_angle_deg(IOR_RGB[1]);
    assert!((half - 48.6).abs() < 0.2, "{half}");
    // Red sees out a little wider than blue: the fringe.
    assert!(window_half_angle_deg(IOR_RGB[0]) > window_half_angle_deg(IOR_RGB[2]));
}

#[test]
fn fresnel_is_two_percent_head_on_and_total_past_the_critical_angle() {
    assert!((fresnel(1.0, 1.0, 1.333) - 0.0204).abs() < 0.001);
    assert!((fresnel(1.0, 1.333, 1.0) - 0.0204).abs() < 0.001);
    let critical = (1.0f32 / 1.333).asin();
    assert_eq!(fresnel((critical + 0.01).cos(), 1.333, 1.0), 1.0);
    assert!(fresnel((critical - 0.05).cos(), 1.333, 1.0) < 1.0);
}

#[test]
fn a_low_sun_comes_into_the_water_within_the_window() {
    let (beam, gets_in) = refracted_sun(Vec3::new(1.0, 0.05, 0.0).normalize()).unwrap();
    let from_down = beam.angle_between(Vec3::NEG_Y).to_degrees();
    assert!(from_down < 48.7 && from_down > 45.0, "{from_down}");
    assert!(beam.x < 0.0, "travels away from the sun");
    assert!(gets_in < 0.5, "a grazing sun is mostly mirrored: {gets_in}");
    let (beam, _) = refracted_sun(Vec3::Y).unwrap();
    assert!((beam - Vec3::NEG_Y).length() < 1e-5);
    assert!(refracted_sun(Vec3::new(1.0, -0.1, 0.0)).is_none());
}

/// Looking straight down into deep water from just under the surface, the
/// sky's share is the water's own colour under the sky -- what the surface
/// shows from above -- so the line between the two does not jump.
#[test]
fn deep_water_under_the_sky_is_the_waters_colour_from_above() {
    let w = sea();
    let u = light(None);
    let s = scatter(&w, &u, 0.0, Vec3::NEG_Y, 1.0e6, 1.0);
    let want = Vec3::new(w.scatter[0], w.scatter[1], w.scatter[2]) * Vec3::from_slice(&u.sky[..3]);
    assert!((s - want).length() < 1e-4 * want.length().max(1e-3), "{s} vs {want}");
}

/// The closed form against a plain sum along the ray.
#[test]
fn the_closed_form_matches_the_integral() {
    let w = pond();
    let u = light(Some(Vec3::new(0.3, 0.8, 0.2)));
    for (z0, d, t) in [(0.5, Vec3::new(1.0, -0.3, 0.0), 3.0), (2.0, Vec3::new(0.2, 0.6, 0.5), 2.5), (1.0, Vec3::X, 8.0)] {
        let d = d.normalize();
        let closed = scatter(&w, &u, z0, d, t, 1.0);
        let n = 20000;
        let dt = t / n as f32;
        let k = Vec3::from_slice(&w.extinction[..3]);
        let mut sum = Vec3::ZERO;
        for i in 0..n {
            let ti = (i as f32 + 0.5) * dt;
            let z = z0 - d.y * ti;
            let step = scatter(&w, &u, z, d, dt, 1.0);
            sum += step * Vec3::new((-k.x * ti).exp(), (-k.y * ti).exp(), (-k.z * ti).exp());
        }
        assert!((closed - sum).length() < 1e-3 * closed.length().max(1e-4), "{closed} vs {sum}");
    }
}

/// Red goes first: through ten metres of the sea a white thing is blue-green.
#[test]
fn red_is_lost_first() {
    let w = sea();
    let k = Vec3::from_slice(&w.extinction[..3]);
    let t = Vec3::new((-k.x * 10.0).exp(), (-k.y * 10.0).exp(), (-k.z * 10.0).exp());
    assert!(t.x < 0.1 * t.y && t.x < 0.1 * t.z, "{t}");
    // And the pond hides a metre of itself far more than the sea does.
    let kp = Vec3::from_slice(&pond().extinction[..3]);
    assert!(kp.y > 5.0 * k.y);
}

#[test]
fn the_water_darkens_with_depth() {
    let w = sea();
    let u = light(Some(Vec3::new(0.3, 0.9, 0.1)));
    let at = |d| in_water_luminance(&w, &u, d);
    assert!(at(0.2) > at(2.0) && at(2.0) > at(8.0), "{} {} {}", at(0.2), at(2.0), at(8.0));
    assert!(at(8.0) > 0.0);
    // The meter goes between the air's and the water's in stops.
    assert_eq!(meter_in_water(0.4, 0.1, 0.0), 0.4);
    assert!((meter_in_water(0.4, 0.1, 1.0) - 0.1).abs() < 1e-6);
    assert!((meter_in_water(0.4, 0.1, 0.5) - 0.2).abs() < 1e-5);
}

#[test]
fn an_eye_is_above_on_or_under_the_surface() {
    let mut w = pond();
    w.set_time(12.0, 240.0);
    let reach = wave_reach(WaveParams { wind_speed: 2.0, fetch: 20.0, ..Default::default() }.significant_height(), 0.7);
    assert!(reach < 0.1, "a pond's ripples: {reach}");
    let at = |y| eye_water(&w, 0.7, reach, Vec3::new(-26.0, y, -26.0));
    assert_eq!(at(1.2), EyeWater::Above);
    assert_eq!(at(-0.368), EyeWater::Waterline);
    assert_eq!(at(-0.65), EyeWater::Under);
    // Over dry ground there is no water to be under.
    assert_eq!(eye_water(&w, -0.2, reach, Vec3::new(-26.0, -2.0, -26.0)), EyeWater::Above);
}

#[test]
fn the_still_depth_is_the_nearest_vertexs_and_none_outside() {
    let verts: Vec<WaterVertex> = (0..10)
        .flat_map(|z| (0..10).map(move |x| WaterVertex { position: [x as f32, 0.0, z as f32], depth: x as f32 * 0.1 }))
        .collect();
    let s = StillDepth::new(&verts);
    assert!((s.at(Vec2::new(3.1, 4.0)).unwrap() - 0.3).abs() < 1e-6);
    assert!((s.at(Vec2::new(8.9, 0.2)).unwrap() - 0.9).abs() < 1e-6);
    assert!(s.at(Vec2::new(-3.0, 4.0)).is_none());
}

#[test]
fn the_film_shows_only_for_a_moment_after_surfacing() {
    let mut s = Surfacing::default();
    assert_eq!(s.update(EyeWater::Above, 0.0), None);
    assert_eq!(s.update(EyeWater::Under, 1.0), None);
    assert_eq!(s.update(EyeWater::Above, 2.0), Some(0.0));
    assert!(s.update(EyeWater::Above, 2.5).is_some_and(|a| (a - 0.5).abs() < 1e-6));
    assert_eq!(s.update(EyeWater::Above, 2.0 + FILM_SECONDS as f64 + 0.01), None);
    // Back under and out again: a fresh film.
    s.update(EyeWater::Waterline, 5.0);
    assert_eq!(s.update(EyeWater::Above, 6.0), Some(0.0));
}

// ---------------------------------------------------------------------------
// On the GPU
// ---------------------------------------------------------------------------

fn gpu(dual: bool) -> Option<(Device, Queue)> {
    let instance = Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
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

const SIZE: u32 = 16;

/// One frame from an eye at `eye` looking along `look`: the buffer cleared
/// to `behind`, the probe pass's depth and the scene's at `geometry_z` (1:
/// nothing), the veil drawn -- or, with `surface_at`, the underside of a
/// still surface at that height instead. Every pixel, row by row.
fn frame(dual: bool, w: WaterUniform, eye: Vec3, look: Vec3, behind: [f64; 3], geometry_z: f32, surface_at: Option<f32>) -> Option<Vec<[u8; 4]>> {
    let (device, queue) = gpu(dual)?;
    let format = TextureFormat::Rgba8Unorm;
    let lights = LightsUniform::new(&device);
    let (_shadows, uniforms) = scene_uniforms(&device, &lights);
    let up = if look.normalize().y.abs() > 0.9 { Vec3::Z } else { Vec3::Y };
    let view_proj = Mat4::perspective_rh(1.6, 1.0, 0.03, 100.0) * Mat4::look_at_rh(eye, eye + look, up);
    uniforms.upload_scene(
        &queue,
        view_proj,
        eye,
        &ShadowUpload::disabled(),
        &SkyUpload::none(),
        &PostUpload::default(),
        &PlayerUpload::default(),
    );
    let probe_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(&device);
    let probe = crate::renderer::brush_pipeline::probe_pass::Target::new(&device, &probe_layout, SIZE, SIZE, 1);

    let scope = device.push_error_scope(ErrorFilter::Validation);
    let pipes = UnderwaterPipelines::new_with(&device, format, &uniforms.layout, &probe_layout, 1, dual);
    let params = WaveParams { wind_speed: 0.5, fetch: 50.0, ..Default::default() };
    let waves = WaveField::new(&device, &queue, params);
    let mut w = w;
    w.waves[2] = 0.0;
    let ubuf = device.create_buffer_init(&util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&w), usage: BufferUsages::UNIFORM });
    let groups = pipes.water_groups(&device, &ubuf, &waves);
    let (under_buf, under_group) = pipes.under_buffer(&device);
    let mut u = light(Some(Vec3::new(0.2, 0.9, 0.1)));
    u.view = [SIZE as f32, SIZE as f32, 1.6 / SIZE as f32, -1.0];
    queue.write_buffer(&under_buf, 0, bytemuck::bytes_of(&u));

    let tex = |format, usage| {
        device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let target = tex(format, TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC);
    let depth = tex(TextureFormat::Depth32Float, TextureUsages::RENDER_ATTACHMENT);
    let (tv, dv) = (target.create_view(&Default::default()), depth.create_view(&Default::default()));
    let quad = surface_at.map(|y| {
        let v = |x: f32, z: f32| WaterVertex { position: [x, y, z], depth: 5.0 };
        let verts = [v(-200.0, -200.0), v(200.0, -200.0), v(-200.0, 200.0), v(200.0, 200.0)];
        let vb = device.create_buffer_init(&util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&verts), usage: BufferUsages::VERTEX });
        let ib = device.create_buffer_init(&util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[0u32, 2, 1, 1, 2, 3]), usage: BufferUsages::INDEX });
        (vb, ib)
    });
    let readback = device.create_buffer(&BufferDescriptor { label: None, size: (256 * SIZE) as u64, usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false });

    let mut encoder = device.create_command_encoder(&Default::default());
    waves.update(&queue, &mut encoder, 12.0);
    {
        // The probe pass's depth: the geometry, everywhere.
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &probe.depth_view,
                depth_ops: Some(Operations { load: LoadOp::Clear(geometry_z), store: StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
    }
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &tv,
                depth_slice: None,
                resolve_target: None,
                ops: Operations { load: LoadOp::Clear(Color { r: behind[0], g: behind[1], b: behind[2], a: 1.0 }), store: StoreOp::Store },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &dv,
                // The scene's geometry a hair behind the probe pass's.
                depth_ops: Some(Operations { load: LoadOp::Clear((geometry_z + 1e-5).min(1.0)), store: StoreOp::Discard }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        match &quad {
            None => pipes.draw_veil(&mut pass, EyeWater::Under, [&uniforms.bind_group, &groups[waves.current()], &probe.bind_group, &under_group]),
            Some((vb, ib)) => pipes.draw_underside(&mut pass, [&uniforms.bind_group, &groups[waves.current()], &under_group], vb, ib, 6),
        }
    }
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
        TexelCopyBufferInfo { buffer: &readback, layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(SIZE) } },
        Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
    );
    queue.submit(Some(encoder.finish()));
    let error = pollster::block_on(scope.pop());
    assert!(error.is_none(), "validation: {error:?}");
    let slice = readback.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
    let data = slice.get_mapped_range().unwrap();
    Some((0..SIZE * SIZE).map(|i| {
        let o = (i / SIZE * 256 + i % SIZE * 4) as usize;
        [data[o], data[o + 1], data[o + 2], data[o + 3]]
    }).collect())
}

fn centre(image: &[[u8; 4]]) -> [u8; 4] {
    image[(SIZE / 2 * SIZE + SIZE / 2) as usize]
}

/// Depth of a point `metres` straight ahead, for the tests' camera.
fn depth_at(metres: f32) -> f32 {
    let c = Mat4::perspective_rh(1.6, 1.0, 0.03, 100.0) * glam::Vec4::new(0.0, 0.0, -metres, 1.0);
    c.z / c.w
}

#[test]
fn every_pipeline_builds_and_draws() {
    for dual in [true, false] {
        let Some(image) = frame(dual, sea(), Vec3::new(0.0, -4.0, 0.0), Vec3::X, [1.0; 3], depth_at(3.0), None) else {
            eprintln!("skipping: no GPU (dual {dual})");
            continue;
        };
        assert_eq!(image.len(), (SIZE * SIZE) as usize);
        let _ = frame(dual, sea(), Vec3::new(0.0, -4.0, 0.0), Vec3::Y, [0.0; 3], 1.0, Some(-1.4));
    }
}

/// White three metres off in the sea comes through blue-green; ten metres
/// off, less of it and bluer; with nothing there, the water's own colour.
#[test]
fn white_through_the_sea_goes_blue_green_with_distance() {
    let eye = Vec3::new(0.0, -4.0, 0.0);
    let Some(near) = frame(true, sea(), eye, Vec3::X, [1.0; 3], depth_at(3.0), None) else { return };
    let far = frame(true, sea(), eye, Vec3::X, [1.0; 3], depth_at(10.0), None).unwrap();
    let open = frame(true, sea(), eye, Vec3::X, [1.0; 3], 1.0, None).unwrap();
    let (n, f, o) = (centre(&near), centre(&far), centre(&open));
    assert!(n[0] < n[1] && n[0] < n[2], "red first: {n:?}");
    assert!(f[0] < n[0] && f[1] < n[1], "further is dimmer: {f:?} vs {n:?}");
    assert!(o[0] < 40 && o[2] > o[0], "nothing there: the water: {o:?}");
}

/// The pond hides the same white in a metre or two.
#[test]
fn the_pond_is_murky() {
    let eye = Vec3::new(-26.0, -0.7, -26.0);
    let Some(sea_px) = frame(true, sea(), Vec3::new(0.0, -4.0, 0.0), Vec3::X, [1.0; 3], depth_at(2.0), None) else { return };
    let pond_px = frame(true, pond(), eye, Vec3::X, [1.0; 3], depth_at(2.0), None).unwrap();
    let sum = |p: [u8; 4]| p[0] as u32 + p[1] as u32 + p[2] as u32;
    assert!(sum(centre(&pond_px)) + 150 < sum(centre(&sea_px)), "{:?} vs {:?}", centre(&pond_px), centre(&sea_px));
}

/// Looking straight up through a calm surface the sky shows (Snell's window);
/// at the edge of the view, past the critical angle, the dark water mirrored.
#[test]
fn looking_up_shows_the_window_and_round_it_the_mirror() {
    let eye = Vec3::new(0.0, -3.0, 0.0);
    let Some(image) = frame(true, sea(), eye, Vec3::Y, [0.0; 3], 1.0, Some(-1.4)) else { return };
    let up = centre(&image);
    // 80 degrees field each way: the corners look ~66 degrees off vertical.
    let corner = image[0];
    let sum = |p: [u8; 4]| p[0] as u32 + p[1] as u32 + p[2] as u32;
    assert!(sum(up) > sum(corner) + 60, "window {up:?} vs mirror {corner:?}");
}
