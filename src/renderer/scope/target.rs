//! The square texture a scope's world view is rendered into, and the blit that
//! builds its mip chain.
//!
//! The lens minifies this target onto an ocular disc a few times smaller, so it
//! must be sampled through a full mip chain: without one, minification point-
//! samples a few texels out of dozens and the picture crawls under head motion.

use wgpu::{
    BindGroup, BindGroupLayout, CommandEncoder, Device, Extent3d, Sampler, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

/// Fills a texture's mip levels by successive halving: a fullscreen triangle
/// sampling level N-1 bilinearly into level N, an exact 2x2 box on a 2:1 step.
pub struct MipBlit {
    pipeline: wgpu::RenderPipeline,
    layout: BindGroupLayout,
    sampler: Sampler,
}

/// A scope's colour (with its mip chain), depth, and the multisampled
/// attachments the world pass draws into when the scene pipelines use MSAA.
pub struct ScopeTarget {
    pub color: Texture,
    /// Mip 0 alone: what the world pass writes or resolves into.
    pub color_view: TextureView,
    /// The whole chain: what the lens samples. Binding one level would make the
    /// sampler's mip filter a silent no-op.
    pub sample_view: TextureView,
    msaa_view: Option<TextureView>,
    pub depth_view: TextureView,
    pub size: u32,
    mip_views: Vec<TextureView>,
    mip_sources: Vec<BindGroup>,
}

const BLIT_WGSL: &str = r"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var out: VsOut;
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    out.uv = vec2<f32>(x, y);
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, samp, in.uv, 0.0);
}
";

impl MipBlit {
    pub fn new(device: &Device, format: TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scope_mip_blit_shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let layout = super::lens::LensPipeline::texture_layout(device, "scope_mip_blit_bgl");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scope_mip_blit_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scope_mip_blit"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, layout, sampler: ScopeTarget::clamped_sampler(device, false) }
    }
}

impl ScopeTarget {
    /// A `size` square target in `format`, whose world pass draws with
    /// `samples` (the scene pipelines' MSAA count).
    pub fn new(device: &Device, format: TextureFormat, size: u32, samples: u32, blit: &MipBlit) -> Self {
        let size = size.max(16);
        let mip_levels = 32 - size.leading_zeros();
        let color = Self::texture(device, format, size, mip_levels, 1);
        let depth = Self::texture(device, TextureFormat::Depth32Float, size, 1, samples);
        let msaa_view = (samples > 1).then(|| {
            Self::texture(device, format, size, 1, samples).create_view(&TextureViewDescriptor::default())
        });
        let mip_views: Vec<TextureView> = (0..mip_levels).map(|level| Self::level_view(&color, level)).collect();
        let mip_sources = mip_views
            .iter()
            .map(|view| super::lens::LensPipeline::texture_group(device, &blit.layout, view, &blit.sampler))
            .collect();
        Self {
            color_view: Self::level_view(&color, 0),
            sample_view: color.create_view(&TextureViewDescriptor::default()),
            msaa_view,
            depth_view: depth.create_view(&TextureViewDescriptor::default()),
            color,
            size,
            mip_views,
            mip_sources,
        }
    }

    /// The colour attachment for the world pass: multisampled and resolved
    /// into mip 0 when the scene uses MSAA, mip 0 directly otherwise.
    pub fn color_attachment(&self) -> wgpu::RenderPassColorAttachment<'_> {
        let (view, resolve_target, store) = match &self.msaa_view {
            Some(msaa) => (msaa, Some(&self.color_view), wgpu::StoreOp::Discard),
            None => (&self.color_view, None, wgpu::StoreOp::Store),
        };
        wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.05, a: 1.0 }), store },
        }
    }

    pub fn depth_attachment(&self) -> wgpu::RenderPassDepthStencilAttachment<'_> {
        wgpu::RenderPassDepthStencilAttachment {
            view: &self.depth_view,
            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
            stencil_ops: None,
        }
    }

    /// Fill mips 1..n from mip 0. Record it in the world pass's own encoder,
    /// after the pass ends and before submit, or the chain is a frame stale.
    pub fn generate_mips(&self, encoder: &mut CommandEncoder, blit: &MipBlit) {
        for level in 1..self.mip_views.len() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scope_mip_blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mip_views[level],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&blit.pipeline);
            pass.set_bind_group(0, &self.mip_sources[level - 1], &[]);
            pass.draw(0..3, 0..1);
        }
    }

    pub fn mip_levels(&self) -> u32 {
        self.color.mip_level_count()
    }

    /// Clamped so sampling past the field never wraps the far side into view.
    pub(super) fn clamped_sampler(device: &Device, trilinear: bool) -> Sampler {
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scope_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: if trilinear { wgpu::MipmapFilterMode::Linear } else { wgpu::MipmapFilterMode::Nearest },
            ..Default::default()
        })
    }

    fn texture(device: &Device, format: TextureFormat, size: u32, mip_level_count: u32, sample_count: u32) -> Texture {
        let sampled = if sample_count == 1 { TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC } else { TextureUsages::empty() };
        device.create_texture(&TextureDescriptor {
            label: Some("scope_target"),
            size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count,
            sample_count,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | sampled,
            view_formats: &[],
        })
    }

    fn level_view(color: &Texture, level: u32) -> TextureView {
        color.create_view(&TextureViewDescriptor {
            label: Some("scope_mip"),
            base_mip_level: level,
            mip_level_count: Some(1),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

    #[test]
    fn a_target_is_square_with_a_full_chain_and_msaa_when_asked() {
        let Some((device, _queue)) = crate::renderer::pipeline::tests::headless_gpu() else {
            log::warn!("skipping: no GPU adapter");
            return;
        };
        let blit = MipBlit::new(&device, FORMAT);
        let t = ScopeTarget::new(&device, FORMAT, 256, 1, &blit);
        assert_eq!((t.color.width(), t.color.height(), t.mip_levels()), (256, 256, 9));
        assert_eq!(t.mip_views.len(), 9);
        assert!(t.msaa_view.is_none());
        assert!(ScopeTarget::new(&device, FORMAT, 256, 4, &blit).msaa_view.is_some());
    }

    /// Paints a 1-texel checkerboard into mip 0, builds the chain, and returns
    /// the variance of levels 0 and 1 and the mean of level 1.
    fn checkerboard_stats(device: &Device, queue: &wgpu::Queue) -> (f64, f64, f64) {
        let blit = MipBlit::new(device, FORMAT);
        let target = ScopeTarget::new(device, FORMAT, 128, 1, &blit);
        let pipeline = crate::renderer::scope::lens::tests::checker_pipeline(device, FORMAT);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(target.color_attachment())],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..3, 0..1);
        }
        target.generate_mips(&mut encoder, &blit);
        queue.submit(Some(encoder.finish()));
        let stats = |level: u32| {
            let reds = crate::renderer::scope::lens::tests::read_level(device, queue, &target.color, level);
            let mean = reds.iter().sum::<f64>() / reds.len() as f64;
            (mean, reds.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / reds.len() as f64)
        };
        let ((_, v0), (m1, v1)) = (stats(0), stats(1));
        (v0, v1, m1)
    }

    /// The shimmer test: level 1 of a worst-case checkerboard must average to
    /// mid grey, which is what a minifying sample now reads instead of edges.
    #[test]
    fn the_mip_chain_averages_away_the_minification_aliasing() {
        let Some((device, queue)) = crate::renderer::pipeline::tests::headless_gpu() else {
            log::warn!("skipping: no GPU adapter");
            return;
        };
        let (v0, v1, m1) = checkerboard_stats(&device, &queue);
        assert!(v0 > 1000.0, "level 0 should be a full-contrast checkerboard, variance {v0:.1}");
        assert!(v1 < v0 / 50.0, "level 1 must average it away: {v1:.1} vs {v0:.1}");
        assert!((m1 - 127.5).abs() < 20.0, "each 2x2 block should be mid grey, got {m1:.1}");
    }
}
