use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};
use wgpu::*;
use glam::{Vec3, Quat};
use std::sync::{Arc, Mutex};
use std::io::Read;
use std::net::TcpListener;
use std::time::Instant;

use crate::renderer::{Renderer, Camera, Cuboid, Color3};

#[derive(Clone, Copy)]
struct HeadPose {
    position: Vec3,
    rotation: Quat,
}

impl Default for HeadPose {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 1.6, 0.0),
            rotation: Quat::IDENTITY,
        }
    }
}

enum SceneSource {
    Static(Vec<Cuboid>),
    Engine {
        last_time: Instant,
        frame_fn:  Box<dyn FnMut(f32, Vec3, Quat) -> Vec<Cuboid> + Send>,
    },
}

pub struct Canvas {
    renderer:  Option<Renderer>,
    surface:   Option<Surface<'static>>,
    config:    Option<SurfaceConfiguration>,
    instance:  Instance,
    camera:    Camera,
    scene:     SceneSource,
    window:    Option<Arc<Window>>,
    live_pose: Arc<Mutex<HeadPose>>,
}

impl Canvas {
    pub fn new() -> Self {
        Self::new_with_scene(default_scene())
    }

    pub fn new_with_scene(cuboids: Vec<Cuboid>) -> Self {
        Self::new_internal(SceneSource::Static(cuboids))
    }
    
    pub fn new_with_engine(
        frame_fn: impl FnMut(f32, Vec3, Quat) -> Vec<Cuboid> + Send + 'static,
    ) -> Self {
        Self::new_internal(SceneSource::Engine {
            last_time: Instant::now(),
            frame_fn:  Box::new(frame_fn),
        })
    }

    fn new_internal(scene: SceneSource) -> Self {
        let instance  = Instance::new(&InstanceDescriptor::default());
        let live_pose = Arc::new(Mutex::new(HeadPose::default()));

        let pose_recv = Arc::clone(&live_pose);
        std::thread::spawn(move || {
            let listener = match TcpListener::bind("127.0.0.1:7777") {
                Ok(l)  => l,
                Err(e) => {
                    log::warn!("Mirror: could not bind :7777 — head tracking disabled ({e})");
                    return;
                }
            };
            log::info!("Mirror: listening for head pose on TCP :7777");

            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                log::info!("Mirror: Quest connected");

                let mut buf = [0u8; 28];
                loop {
                    match stream.read_exact(&mut buf) {
                        Ok(()) => {
                            let px = f32::from_le_bytes(buf[ 0.. 4].try_into().unwrap());
                            let py = f32::from_le_bytes(buf[ 4.. 8].try_into().unwrap());
                            let pz = f32::from_le_bytes(buf[ 8..12].try_into().unwrap());
                            let qx = f32::from_le_bytes(buf[12..16].try_into().unwrap());
                            let qy = f32::from_le_bytes(buf[16..20].try_into().unwrap());
                            let qz = f32::from_le_bytes(buf[20..24].try_into().unwrap());
                            let qw = f32::from_le_bytes(buf[24..28].try_into().unwrap());
                            *pose_recv.lock().unwrap() = HeadPose {
                                position: Vec3::new(px, py, pz),
                                rotation: Quat::from_xyzw(qx, qy, qz, qw),
                            };
                        }
                        Err(e) => {
                            log::info!("Mirror: Quest disconnected ({e})");
                            break;
                        }
                    }
                }
            }
        });

        Self {
            renderer: None,
            surface:  None,
            config:   None,
            instance,
            camera:   Camera::new(1.0),
            scene,
            window:   None,
            live_pose,
        }
    }

    pub fn run(mut self) {
        let event_loop = EventLoop::new().unwrap();
        event_loop.run_app(&mut self).unwrap();
    }

    fn redraw(&mut self) {
        let Some(renderer) = self.renderer.as_mut() else { return; };
        let Some(surface)  = self.surface.as_ref()  else { return; };

        let pose = *self.live_pose.lock().unwrap();
        self.camera.position = pose.position;
        self.camera.rotation = pose.rotation;

        let cuboids: Vec<Cuboid> = match &mut self.scene {
            SceneSource::Static(c) => c.clone(),
            SceneSource::Engine { last_time, frame_fn } => {
                let now = Instant::now();
                let dt  = now.duration_since(*last_time).as_secs_f32();
                *last_time = now;
                frame_fn(dt, pose.position, pose.rotation)
            }
        };

        let frame = match surface.get_current_texture() {
            Ok(f)  => f,
            Err(e) => {
                log::warn!("Mirror: failed to get current texture ({e})");
                return;
            }
        };
        let view = frame.texture.create_view(&TextureViewDescriptor::default());
        renderer.render(&view, &self.camera, &cuboids);
        frame.present();
    }
}

impl ApplicationHandler for Canvas {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("space_soup — mirror")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32)),
            ).unwrap(),
        );
        self.window = Some(window.clone());

        let surface: Surface<'static> = unsafe {
            std::mem::transmute(self.instance.create_surface(window.clone()).unwrap())
        };

        let adapter = pollster::block_on(self.instance.request_adapter(
            &RequestAdapterOptions {
                power_preference:       PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            },
        )).unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                required_features: Features::empty(),
                required_limits:   Limits::default(),
                ..Default::default()
            },
        )).unwrap();

        let size   = window.inner_size();
        let caps   = surface.get_capabilities(&adapter);
        let format = caps.formats[0];

        let config = SurfaceConfiguration {
            usage:                         TextureUsages::RENDER_ATTACHMENT,
            format,
            width:                         size.width,
            height:                        size.height,
            present_mode:                  PresentMode::Fifo,
            alpha_mode:                    caps.alpha_modes[0],
            view_formats:                  vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        self.camera  = Camera::new(size.width as f32 / size.height as f32);
        let renderer = Renderer::from_device(device, queue, format, size.width, size.height);

        self.renderer = Some(renderer);
        self.surface  = Some(surface);
        self.config   = Some(config);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let (Some(surface), Some(config), Some(renderer)) =
                    (self.surface.as_ref(), self.config.as_mut(), self.renderer.as_mut())
                {
                    config.width  = size.width;
                    config.height = size.height;
                    surface.configure(&renderer.device, config);
                    renderer.resize(size.width, size.height);
                    self.camera.aspect = size.width as f32 / size.height as f32;
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn default_scene() -> Vec<Cuboid> {
    vec![
        Cuboid::solid(
            Vec3::new(0.0, 0.5, 0.0), Vec3::splat(0.5), Color3(220, 60, 60, 255),
        ),
        Cuboid::wireframe(
            Vec3::new(2.0, 0.5, 0.0), Vec3::splat(0.5), Color3(60, 220, 60, 255),
        ),
        Cuboid::solid_and_wire(
            Vec3::new(-2.0, 0.5, 0.0),
            Vec3::new(0.4, 0.8, 0.4),
            Color3(60, 100, 220, 200),
            Color3(200, 200, 255, 255),
        ),
        Cuboid::solid(
            Vec3::new(0.0, -0.05, 0.0), Vec3::new(10.0, 0.05, 10.0), Color3(80, 80, 80, 255),
        ),
    ]
}
