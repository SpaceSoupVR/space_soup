use super::lights::wgsl_lights_block;
use wgpu::util::DeviceExt;
use wgpu::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleInstance {
    pub pos: [f32; 3],
    pub size: f32,
    pub color: [f32; 4],
    pub stretch: [f32; 3],
    pub soft: f32,
    pub glow: f32,
    pub spin: f32,
    pub pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleFrame {
    right: [f32; 4],
    up: [f32; 4],
}

pub struct ParticlePipeline {
    pub pipeline: RenderPipeline,
    frame_buf: Buffer,
    frame_group: BindGroup,
    instances: Buffer,
    capacity: usize,
    count: u32,
}

impl ParticlePipeline {
    pub fn new(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        let frame_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("particle_frame_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let frame_buf = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("particle_frame"),
            contents: bytemuck::bytes_of(&ParticleFrame::default()),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let frame_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("particle_frame"),
            layout: &frame_layout,
            entries: &[BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() }],
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("particle_pipeline_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(&frame_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("particle_shader"),
            source: ShaderSource::Wgsl(particle_shader().into()),
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("particle_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(VertexBufferLayout {
                    array_stride: std::mem::size_of::<ParticleInstance>() as BufferAddress,
                    step_mode: VertexStepMode::Instance,
                    attributes: &vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4],
                })],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState { format, blend: Some(BlendState::ALPHA_BLENDING), write_mask: ColorWrites::ALL })],
            }),
            primitive: PrimitiveState { topology: PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::LessEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let capacity = 4096;
        let instances = device.create_buffer(&BufferDescriptor {
            label: Some("particle_instances"),
            size: (capacity * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, frame_buf, frame_group, instances, capacity, count: 0 }
    }

    pub fn upload(&mut self, device: &Device, queue: &Queue, right: glam::Vec3, up: glam::Vec3, items: &[ParticleInstance]) {
        queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&ParticleFrame { right: [right.x, right.y, right.z, 0.0], up: [up.x, up.y, up.z, 0.0] }));
        if items.len() > self.capacity {
            self.capacity = items.len().next_power_of_two();
            self.instances = device.create_buffer(&BufferDescriptor {
                label: Some("particle_instances"),
                size: (self.capacity * std::mem::size_of::<ParticleInstance>()) as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !items.is_empty() {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(items));
        }
        self.count = items.len() as u32;
    }

    pub fn draw<'a>(&'a self, pass: &mut RenderPass<'a>, scene_group: &'a BindGroup) {
        if self.count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, scene_group, &[]);
        pass.set_bind_group(1, &self.frame_group, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        pass.draw(0..6, 0..self.count);
    }
}

fn particle_shader() -> String {
    format!(
        r#"
struct ParticleFrame {{
    right: vec4<f32>,
    up: vec4<f32>,
}}
@group(1) @binding(0) var<uniform> frame: ParticleFrame;

{lights_block}

struct VIn {{
    @builtin(vertex_index) corner: u32,
    @location(0) pos_size: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) stretch_soft: vec4<f32>,
    @location(3) glow_spin: vec4<f32>,
}}
struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) soft: f32,
    @location(3) glow: f32,
    @location(4) world_pos: vec3<f32>,
}}

@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corners[v.corner];
    let size = v.pos_size.w;
    let right = frame.right.xyz;
    let up = frame.up.xyz;
    let stretch = v.stretch_soft.xyz;
    let spin = v.glow_spin.y;
    var axis_x = right;
    var axis_y = up;
    var len_x = size;
    var len_y = size;
    let s_len = length(stretch);
    if (s_len > 1e-4) {{
        let along = vec2<f32>(dot(stretch, right), dot(stretch, up));
        let a_len = length(along);
        if (a_len > 1e-5) {{
            let dir = along / a_len;
            axis_x = right * dir.x + up * dir.y;
            axis_y = right * (-dir.y) + up * dir.x;
            len_x = size + a_len;
        }}
    }} else if (abs(spin) > 1e-4) {{
        let cs = cos(spin);
        let sn = sin(spin);
        axis_x = right * cs + up * sn;
        axis_y = right * (-sn) + up * cs;
    }}
    let world = v.pos_size.xyz + axis_x * (c.x * len_x) + axis_y * (c.y * len_y);
    out.clip = cam_view_proj() * vec4<f32>(world, 1.0);
    out.uv = c;
    out.color = v.color;
    out.soft = v.stretch_soft.w;
    out.glow = v.glow_spin.x;
    out.world_pos = world;
    return out;
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let d = length(in.uv);
    if (d > 1.0) {{
        discard;
    }}
    let edge = 1.0 - smoothstep(in.soft, 1.0, d);
    let core = 1.0 - smoothstep(0.0, max(in.soft, 0.05), d);
    receiver_skips_baked = true;
    let lit = shade(in.world_pos, vec3<f32>(0.0, 1.0, 0.0));
    let rgb = mix(in.color.rgb * lit, in.color.rgb, in.glow) + in.color.rgb * in.glow * core * 0.8;
    return vec4<f32>(rgb, in.color.a * edge);
}}
"#,
        lights_block = wgsl_lights_block(0, 1)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_particle_shader_validates() {
        let src = particle_shader();
        let module = wgpu::naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
    }

    #[test]
    fn an_instance_is_sixteen_floats() {
        assert_eq!(std::mem::size_of::<ParticleInstance>(), 64);
    }
}
