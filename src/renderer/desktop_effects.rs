//! Scene effects in the desktop view: the headset's simulation, governor and
//! shader, drawn after the sky against the scene's depth for soft edges, so
//! an editor shows what the headset will.
//!
//! The headset works in the player's frame; here the world is the frame
//! (offset zero, no yaw) and there is one mono view with no rooms or portals.

use glam::{Quat, Vec3};
use wgpu::*;

use super::brush_pipeline::probe_pass;
use super::effects::{self, stack, EffectEmitter, EffectsGpu, EffectsPipeline, EffectsUniform};
use super::lights::Light;
use super::Camera;

pub(super) struct DesktopEffects {
    probe_layout: BindGroupLayout,
    pipeline: EffectsPipeline,
    gpu: EffectsGpu,
    emitters: Vec<EffectEmitter>,
    seen_until: Vec<f64>,
    library: stack::EffectLibrary,
    bursts: stack::BurstSet,
    blank: TextureView,
    samplers: (Sampler, Sampler),
    depth_bind: Option<BindGroup>,
    pub seconds: f64,
    pub hour: Option<f32>,
    drawn: bool,
}

impl DesktopEffects {
    pub fn new(device: &Device, queue: &Queue, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        let started = std::time::Instant::now();
        let layout = effects::bind_group_layout(device);
        let probe_layout = probe_pass::bind_group_layout(device);
        let pipeline = EffectsPipeline::new_multisampled(device, format, uniform_layout, &probe_layout, &layout, 1);
        let gpu = EffectsGpu::new(device, queue, &layout);
        let blank = device
            .create_texture(&TextureDescriptor {
                label: Some("effects_blank"),
                size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() });
        let sampler = |filter| {
            device.create_sampler(&SamplerDescriptor { mag_filter: filter, min_filter: filter, ..Default::default() })
        };
        let samplers = (sampler(FilterMode::Linear), sampler(FilterMode::Nearest));
        log::info!("EFFECTS desktop pipelines made in {:.0} ms", started.elapsed().as_secs_f32() * 1000.0);
        Self {
            probe_layout,
            pipeline,
            gpu,
            emitters: Vec::new(),
            seen_until: Vec::new(),
            library: Default::default(),
            bursts: Default::default(),
            blank,
            samplers,
            depth_bind: None,
            seconds: 0.0,
            hour: None,
            drawn: false,
        }
    }

    pub fn set_emitters(&mut self, emitters: Vec<EffectEmitter>) {
        self.emitters = emitters;
        self.seen_until.clear();
    }

    pub fn set_library(&mut self, library: stack::EffectLibrary) {
        self.library = library;
        self.bursts.live.clear();
    }

    pub fn place(&mut self, placed: &[stack::PlacedEffect]) {
        self.bursts.clear_placed();
        for p in placed {
            match self.library.id(&p.asset) {
                Some(id) => self.bursts.place(&self.library, id, p.position, p.direction, p.source, p.placed),
                None => log::error!("EFFECTS placed `{}` names the asset `{}`, which the library does not hold", p.object, p.asset),
            }
        }
    }

    /// Adds `asset` to the library, or replaces the one of its name: its
    /// placements and bursts take the new version on their next frame.
    pub fn put(&mut self, asset: &super::effect_asset::EffectAsset) -> Result<stack::EffectId, Vec<super::effect_asset::Issue>> {
        self.library.add(asset)
    }

    pub fn id(&self, name: &str) -> Option<stack::EffectId> {
        self.library.id(name)
    }

    pub fn add_burst(&mut self, burst: stack::EffectBurst) {
        self.bursts.add(&self.library, burst);
    }

    /// The particles drawn last frame: over, and screened.
    pub fn counts(&self) -> (u32, u32) {
        if self.drawn { self.gpu.counts() } else { (0, 0) }
    }

    /// Draws every particle as one additive layer: overdraw as heat.
    pub fn set_overdraw(&mut self, on: bool) {
        self.gpu.overdraw = on;
    }

    pub fn is_empty(&self) -> bool {
        self.emitters.is_empty() && self.bursts.live.is_empty()
    }

    /// The scene's depth changed (a resize): its binding is made again.
    pub fn depth_moved(&mut self) {
        self.depth_bind = None;
    }

    /// Simulates the frame and uploads it, as the headset's frame does
    /// before its scene pass. Returns whether anything is to be drawn.
    #[allow(clippy::too_many_arguments)]
    pub fn simulate(
        &mut self,
        device: &Device,
        queue: &Queue,
        camera: &Camera,
        height: u32,
        lights: &[Light],
        ambient: Vec3,
        exposure: f32,
    ) -> bool {
        self.drawn = false;
        if self.is_empty() {
            return false;
        }
        let now = self.seconds;
        let head = camera.position;
        let views = [super::shadow::frustum_planes(camera.projection() * camera.view())];
        let every = |_: &EffectEmitter, _: usize| true;
        let ambient = |_: Vec3| ambient;
        let at = effects::Surroundings {
            head,
            offset: Vec3::ZERO,
            yaw_inv: Quat::IDENTITY,
            lights,
            reaches: &every,
            ambient: &ambient,
            exposure,
            room: None,
        };

        self.seen_until.resize(self.emitters.len(), f64::NEG_INFINITY);
        let mut sight: Vec<Vec<effects::seen::Eye>> = Vec::with_capacity(self.emitters.len());
        for (e, until) in self.emitters.iter().zip(self.seen_until.iter_mut()) {
            let eyes = effects::emitter_sight(e, &views, head, &[], &[]);
            if !eyes.is_empty() {
                *until = now + effects::EMITTER_HOLD;
            }
            sight.push(eyes);
        }
        let held = &self.seen_until;
        let seen = |i: usize| held.get(i).is_none_or(|&t| now <= t);
        let wants = effects::governor_wants(&self.emitters, &seen, &[], &self.bursts.splash_measured, head, &sight);

        self.bursts.mist_hour = self.hour;
        let bursts_drawn = self.bursts.frame_with(&self.library, now, &views, head, &wants);
        if (0..self.emitters.len()).any(seen) || bursts_drawn.contains(&true) {
            let bursts = stack::BurstFrame { lib: &self.library, set: &self.bursts, drawn: &bursts_drawn };
            let set = &self.bursts;
            let shares = effects::Shares {
                emitter: &|e| set.share_of(effects::emitter_key(e)),
                splash: &|s| set.share_of(effects::splash_key(s)),
                sight: &sight,
            };
            let frame = effects::simulate_governed(&self.emitters, &seen, &[], Some(&bursts), Some(&shares), now, &at);
            self.bursts.measured(&frame.splash_costs);
            self.gpu.upload(device, queue, &frame);
            self.drawn = self.gpu.any();
        }
        self.bursts.end_frame();

        if self.drawn {
            let v = camera.view();
            let right = Vec3::new(v.x_axis.x, v.y_axis.x, v.z_axis.x).normalize_or_zero();
            let up = Vec3::new(v.x_axis.y, v.y_axis.y, v.z_axis.y).normalize_or_zero();
            let pixel = 2.0 * (0.5 * camera.fov_y).tan() / height.max(1) as f32;
            self.gpu.set_view(
                queue,
                &EffectsUniform {
                    right: right.extend(0.0).to_array(),
                    up: up.extend(0.0).to_array(),
                    head: head.extend(1.0).to_array(),
                    depth: [camera.near, camera.far, 1.0, pixel],
                },
            );
        }
        self.drawn
    }

    /// Draws into a pass whose depth attachment is `depth`, held read-only,
    /// which the shader also reads for soft edges and the fog's far reach.
    pub fn draw(&mut self, device: &Device, pass: &mut RenderPass<'_>, uniforms: &BindGroup, depth: &Texture) {
        if !self.drawn {
            return;
        }
        let probe = self.depth_bind.get_or_insert_with(|| {
            let depth = depth.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                aspect: TextureAspect::DepthOnly,
                ..Default::default()
            });
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("effects_depth"),
                layout: &self.probe_layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&self.blank) },
                    BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&depth) },
                    BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&self.samplers.0) },
                    BindGroupEntry { binding: 3, resource: BindingResource::Sampler(&self.samplers.1) },
                    BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&self.blank) },
                ],
            })
        });
        self.gpu.draw(pass, &self.pipeline, uniforms, probe);
    }
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};
    use wgpu::*;

    use crate::renderer::effects::{EffectEmitter, EffectKind, Variation};
    use crate::renderer::{Camera, Renderer};

    /// Draws a frame of a fire in front of the camera and returns its
    /// pixels, or None without a GPU. Panics on a validation error.
    fn frame(fire: bool) -> Option<Vec<u8>> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor::default())).unwrap();
        let (w, h) = (256u32, 256u32);
        let format = TextureFormat::Rgba8UnormSrgb;
        let scope = device.push_error_scope(ErrorFilter::Validation);
        let mut renderer = Renderer::from_device(device.clone(), queue.clone(), format, w, h);
        renderer.clear_color = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
        if fire {
            renderer.set_effects(vec![EffectEmitter {
                id: "fire".into(),
                kind: EffectKind::Fire,
                position: Vec3::ZERO,
                direction: Vec3::Y,
                extent: [Vec3::X, Vec3::Y, Vec3::Z],
                scale: 1.5,
                rate: 1.0,
                tint: [1.0; 3],
                ceiling: None,
                variation: Variation::default(),
            }]);
        }
        let target = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let camera = Camera { position: Vec3::new(0.0, 0.6, 2.0), rotation: Quat::IDENTITY, ..Camera::new(1.0) };
        // A second of fire, so the flames have risen.
        for step in 0..30 {
            renderer.set_effects_time(f64::from(step) / 30.0, None);
            renderer.render(&view, &camera, &[]);
        }
        let row = (w * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&BufferDescriptor {
            label: None,
            size: u64::from(row * h),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            TexelCopyBufferInfo { buffer: &readback, layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
            Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        if let Some(error) = pollster::block_on(scope.pop()) {
            panic!("the desktop effects frame did not validate: {error}");
        }
        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        let _ = device.poll(PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().expect("map the frame");
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let start = (y * row) as usize;
            rgba.extend_from_slice(&data[start..start + (w * 4) as usize]);
        }
        Some(rgba)
    }

    #[test]
    fn the_desktop_view_draws_a_scenes_fire_as_the_headset_does() {
        let (Some(dark), Some(lit)) = (frame(false), frame(true)) else {
            eprintln!("skipping: no GPU");
            return;
        };
        let bright = |px: &[u8]| px.chunks(4).filter(|p| p[0] > 40).count();
        assert_eq!(bright(&dark), 0, "nothing in the scene but the fire");
        let flame = bright(&lit);
        assert!(flame > 200, "the fire is drawn: {flame} bright pixels");
        let warm = lit.chunks(4).filter(|p| p[0] > 40).all(|p| p[0] >= p[2]);
        assert!(warm, "flames are warm, not blue");
    }
}
