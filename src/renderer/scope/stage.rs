//! Everything the renderer keeps for the frame's optics: one slot per optical
//! path, each with a render target sized to its lens and its lens uniforms.

use glam::{Mat4, Vec3};
use wgpu::{BindGroup, BindGroupLayout, Buffer, CommandEncoder, Device, Queue, RenderPass, TextureFormat};

use super::{LensParams, LensPipeline, MipBlit, ScopeRender, ScopeTarget};
use crate::renderer::multiview::ViewMode;

/// Most optical paths drawn in one frame: two held binoculars.
pub const MAX_SCOPE_PATHS: usize = 4;

/// Where an eye is this frame and how tall its field is, for sizing targets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeSample {
    pub position: Vec3,
    pub vertical_fov_rad: f32,
}

struct ScopeSlot {
    target: Option<(ScopeTarget, BindGroup)>,
    quad: Buffer,
    /// Lens uniforms whose eye 0 stands for the left and the right eye.
    params: [(Buffer, BindGroup); 2],
}

/// The renderer's optics: lens pipelines, targets, and this frame's paths.
pub struct ScopeStage {
    mono: LensPipeline,
    stereo: Option<LensPipeline>,
    blit: MipBlit,
    /// Bound for window lenses, which render no picture.
    placeholder: (ScopeTarget, BindGroup),
    slots: Vec<ScopeSlot>,
    renders: Vec<ScopeRender>,
    format: TextureFormat,
    samples: u32,
}

impl ScopeStage {
    /// Lens pipelines for a scene pass of `format` and `samples`; the stereo
    /// one only when `can_stereo`, and dropped if the device refuses it.
    pub fn new(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout, samples: u32, can_stereo: bool) -> Self {
        let mono = LensPipeline::new(device, format, camera_layout, samples, ViewMode::Mono);
        let stereo = can_stereo.then(|| {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let lens = LensPipeline::new(device, format, camera_layout, samples, ViewMode::Stereo);
            match pollster::block_on(scope.pop()) {
                Some(e) => {
                    log::error!("scope: stereo lens refused, optics hidden in stereo frames -- {e}");
                    None
                }
                None => Some(lens),
            }
        });
        let blit = MipBlit::new(device, format);
        let target = ScopeTarget::new(device, format, 16, 1, &blit);
        let group = mono.target_group(device, &target.sample_view);
        Self { stereo: stereo.flatten(), placeholder: (target, group), mono, blit, slots: Vec::new(), renders: Vec::new(), format, samples }
    }

    /// The paths to draw this frame; past [`MAX_SCOPE_PATHS`] the rest are dropped.
    pub fn set_renders(&mut self, mut renders: Vec<ScopeRender>) {
        renders.truncate(MAX_SCOPE_PATHS);
        self.renders = renders;
    }

    pub fn is_empty(&self) -> bool {
        self.renders.is_empty()
    }

    /// Size each path's target to its lens on screen and write its lens
    /// uniforms. Runs once a frame, before any pass that reads them.
    pub fn prepare(&mut self, device: &Device, queue: &Queue, eyes: [EyeSample; 2], viewport_height: u32) {
        while self.slots.len() < self.renders.len() {
            let slot = self.new_slot(device);
            self.slots.push(slot);
        }
        let positions = eyes.map(|e| e.position);
        for (render, slot) in self.renders.iter().zip(self.slots.iter_mut()) {
            for (base, (buffer, _)) in slot.params.iter().enumerate() {
                queue.write_buffer(buffer, 0, bytemuck::bytes_of(&LensParams::for_scope(render, positions, base as u32)));
            }
            queue.write_buffer(&slot.quad, 0, bytemuck::cast_slice(&render.lens_quad()));
            if render.is_window() {
                continue;
            }
            let distance = eyes[0].position.distance(render.ocular);
            let size = render.target_size(distance, viewport_height, eyes[0].vertical_fov_rad);
            if slot.target.as_ref().is_none_or(|(t, _)| t.size != size) {
                let target = ScopeTarget::new(device, self.format, size, self.samples, &self.blit);
                let group = self.mono.target_group(device, &target.sample_view);
                slot.target = Some((target, group));
            }
        }
    }

    /// The paths that need a world view rendered: slot, camera, camera position.
    pub fn world_views(&self, far: f32) -> Vec<(usize, Mat4, Vec3)> {
        self.renders
            .iter()
            .enumerate()
            .filter(|(i, r)| !r.is_window() && self.slots.get(*i).is_some_and(|s| s.target.is_some()))
            .map(|(i, r)| (i, r.view_proj(far), r.objective))
            .collect()
    }

    pub fn target(&self, slot: usize) -> Option<&ScopeTarget> {
        self.slots.get(slot)?.target.as_ref().map(|(t, _)| t)
    }

    /// Build a world view's mip chain, in the encoder that drew it.
    pub fn finish_world(&self, encoder: &mut CommandEncoder, slot: usize) {
        if let Some(target) = self.target(slot) {
            target.generate_mips(encoder, &self.blit);
        }
    }

    /// Draw every lens into the scene pass for `eye` (both eyes when `stereo`).
    /// `camera` is the scene's group 0.
    pub fn draw_lenses<'a>(&'a self, pass: &mut RenderPass<'a>, camera: &'a BindGroup, stereo: bool, eye: usize) {
        let lens = if stereo { self.stereo.as_ref() } else { Some(&self.mono) };
        let Some(lens) = lens else { return };
        pass.set_pipeline(&lens.pipeline);
        pass.set_bind_group(0, camera, &[]);
        for (render, slot) in self.renders.iter().zip(&self.slots) {
            let texture = match (&slot.target, render.is_window()) {
                (Some((_, group)), false) => group,
                _ => &self.placeholder.1,
            };
            pass.set_bind_group(1, texture, &[]);
            pass.set_bind_group(2, &slot.params[if stereo { 0 } else { eye.min(1) }].1, &[]);
            pass.set_vertex_buffer(0, slot.quad.slice(..));
            pass.draw(0..6, 0..1);
        }
    }

    fn new_slot(&self, device: &Device) -> ScopeSlot {
        let quad = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scope_lens_quad"),
            size: std::mem::size_of::<[[f32; 3]; 6]>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ScopeSlot { target: None, quad, params: [self.mono.params_group(device), self.mono.params_group(device)] }
    }
}
