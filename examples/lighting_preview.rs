//! Headless lighting preview: renders a small scene (floor + boxes) lit by a
//! directional sun and a spot "flashlight", both casting real-time shadows,
//! and writes a PNG. Lets us confirm the lighting/shadow pipeline visually
//! without a window or a GPU with ray tracing (runs on any adapter).
//!
//! Run: cargo run --example lighting_preview --features renderer -- out.png

use glam::{Quat, Vec3};
use space_soup::renderer::offscreen::OffscreenTarget;
use space_soup::renderer::{Camera, Color3, Cuboid, CuboidStyle, Light, LightKind, Renderer};
use wgpu::*;

const W: u32 = 960; // 960*4 = 3840 bytes/row = 256*15, so no row padding on readback
const H: u32 = 640;

fn grey(id: u64, pos: Vec3, half: Vec3, c: Color3) -> Cuboid {
    Cuboid {
        position: pos,
        half_size: half,
        rotation: Quat::IDENTITY,
        color: c,
        wire_color: Color3(0, 0, 0, 255),
        style: CuboidStyle::Solid,
        id,
    }
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "lighting_preview.png".into());

    let instance = Instance::new(&InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no GPU adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        required_features: Features::empty(),
        required_limits: Limits::default(),
        ..Default::default()
    }))
    .expect("request_device");

    let mut renderer = Renderer::from_device(device, queue, TextureFormat::Rgba8UnormSrgb, W, H);

    // Scene: a floor and a few boxes/pillars to catch shadows.
    let cubes = vec![
        grey(1, Vec3::new(0.0, 0.0, 0.0), Vec3::new(8.0, 0.1, 8.0), Color3(180, 180, 185, 255)),
        grey(2, Vec3::new(-2.5, 1.0, -0.5), Vec3::new(1.0, 1.0, 1.0), Color3(200, 120, 90, 255)),
        grey(3, Vec3::new(1.8, 0.75, 1.5), Vec3::new(0.75, 0.75, 0.75), Color3(90, 150, 200, 255)),
        grey(4, Vec3::new(2.5, 1.5, -2.0), Vec3::new(0.4, 1.5, 0.4), Color3(120, 190, 120, 255)),
    ];

    // A warm directional sun from the upper-left (casts the long shadows) plus
    // a spot "flashlight" from above aimed down at the red box (sharper shadow).
    let lights = vec![
        Light {
            position: Vec3::ZERO,
            direction: Vec3::new(-0.5, -1.0, -0.35).normalize(),
            kind: LightKind::Directional,
            color: Color3(255, 240, 210, 255),
            intensity: 1.6,
            range: 0.0,
            cone_angle_deg: 0.0,
        },
        Light {
            position: Vec3::new(-2.5, 5.5, -0.5),
            direction: Vec3::new(0.0, -1.0, 0.0),
            kind: LightKind::Spot,
            color: Color3(210, 225, 255, 255),
            intensity: 120.0,
            range: 30.0,
            cone_angle_deg: 45.0,
        },
    ];

    let mut cam = Camera::new(W as f32 / H as f32);
    cam.position = Vec3::new(9.0, 7.0, 9.0);
    cam.rotation = Quat::from_rotation_arc(Vec3::NEG_Z, (Vec3::new(0.0, 1.0, 0.0) - cam.position).normalize());
    cam.far = 100.0;

    // Render to a windowless target and read the frame back — the same
    // OffscreenTarget the CloudXR server render loop will use.
    let target = OffscreenTarget::new(&renderer.device, W, H);
    renderer.render_with_lights(target.view(), &cam, &cubes, &[], &lights);
    let data = target.read_rgba(&renderer.device, &renderer.queue);

    image::save_buffer(&out, &data, W, H, image::ColorType::Rgba8).expect("save png");
    println!("wrote {out} ({W}x{H})");
}
