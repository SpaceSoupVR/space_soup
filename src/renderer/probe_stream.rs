use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc};

use glam::Vec3;

use super::uniforms::{prefilter_probe, write_probe_depth_layer, write_probe_layer, ProbeUpload, MAX_PROBES};

pub type ProbeSource = Arc<dyn Fn(usize) -> Option<Vec<u8>> + Send + Sync>;

pub type ProbeDepthSource = Arc<dyn Fn(usize) -> Option<Vec<u16>> + Send + Sync>;

/// A probe's prefiltered chain: lit as it is (`Lit`), or, when the stream
/// relights on the GPU (`ProbeStream::new_smooth`), its lamps' light and the
/// daylight's share apart (`Split`), mixed into the layer every frame.
enum Chain {
    Lit(Vec<Vec<u8>>),
    Split(Vec<Vec<u8>>, Vec<Vec<u8>>),
}

type Prepared = (Chain, Option<Vec<u16>>);

pub type FixedFaces = Box<dyn FnOnce() -> Option<Vec<u8>> + Send>;

enum Request {

    Probe(usize),

    Fixed(u32, FixedFaces),
}

enum Done {
    Probe(usize, Option<Prepared>),
    Fixed(u32, Option<Vec<Vec<u8>>>),
}

/// A photograph as its lamps' light and the daylight's share above them: the
/// lamps' colour (alpha the photograph's) and max(full - lamps, 0) (alpha the
/// photograph's), so that `lamps + daylight * sun` is
/// `time_of_day::relight_probe_faces` to f16's rounding. With no lamps
/// photograph the lamps' light is black. With `outside` (a building's, as
/// `time_of_day::building_daylight_faces`) it is black anyway: the outside
/// takes only the daylight.
pub fn split_probe_faces(full: &[u8], lamps: Option<&[u8]>, outside: bool) -> (Vec<u8>, Vec<u8>) {
    use super::uniforms::{f16_to_f32, f32_to_f16};
    let lamps = lamps.filter(|l| l.len() == full.len());
    let (mut lit, mut sun) = (Vec::with_capacity(full.len()), Vec::with_capacity(full.len()));
    for (t, f) in full.chunks_exact(8).enumerate() {
        for c in 0..4 {
            let a = f16_to_f32(u16::from_le_bytes([f[2 * c], f[2 * c + 1]]));
            let b = lamps.map_or(0.0, |l| f16_to_f32(u16::from_le_bytes([l[t * 8 + 2 * c], l[t * 8 + 2 * c + 1]])));
            let (l, s) = if c == 3 { (a, a) } else { (if outside { 0.0 } else { b }, (a - b).max(0.0)) };
            lit.extend_from_slice(&f32_to_f16(l).to_le_bytes());
            sun.extend_from_slice(&f32_to_f16(s).to_le_bytes());
        }
    }
    (lit, sun)
}

/// SMOOTH REFLECTIONS (2026-10-22): the probes relit on the GPU from their
/// lamps' light and the daylight's share, at the daylight of the frame,
/// instead of rebuilt on the CPU each time the daylight moved by
/// `PROBE_RELIGHT_RATIO` (25% steps in every reflection; user and colleague:
/// "the lighting changes in steps"). The relight is linear, and so is the GGX
/// prefilter, so each mip of `lamps + k * sun` is the relit probe's mip.
struct Relight {
    lamps: wgpu::Texture,
    sun: wgpu::Texture,
    pipeline: wgpu::ComputePipeline,
    uniform: wgpu::Buffer,
    /// Per layer, per mip: the bind group writing it.
    groups: Vec<Vec<wgpu::BindGroup>>,
    /// The daylight each layer holds; None until written once.
    held: Vec<Option<f32>>,
    /// Layers with no split chain (the sky's): never relit here.
    fixed_lit: HashSet<u32>,
    resolution: u32,
}

/// A layer is relit once its daylight is this far off the frame's: a 2%
/// step in a reflection, each layer on its own frame, is under what the eye
/// tells apart, and halves the relights a dusk takes against 1%.
pub const RELIGHT_OFF_BY: f32 = 0.02;
/// And at most this many a frame: a layer is 4 MB written and 8 read at
/// 256 px, 0.56 ms on the Quest 3 (`relight_cost`, 2026-10-22; 0.15 on an
/// M-series Mac), paid only while the daylight moves 2% faster than the
/// layers go round (dusk and dawn).
pub const RELIGHTS_PER_FRAME: usize = 1;

const RELIGHT_WGSL: &str = r"
@group(0) @binding(0) var lamps: texture_2d_array<f32>;
@group(0) @binding(1) var sun: texture_2d_array<f32>;
@group(0) @binding(2) var out_tex: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var<uniform> k: vec4<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(out_tex);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let p = vec2<i32>(id.xy);
    let f = i32(id.z);
    let l = textureLoad(lamps, p, f, 0);
    let s = textureLoad(sun, p, f, 0);
    textureStore(out_tex, p, f, vec4<f32>(l.rgb + k.x * s.rgb, s.a));
}
";

impl Relight {
    fn new(device: &wgpu::Device, out: &wgpu::Texture, res: u32, layers: u32) -> Self {
        let source = |label| device.create_texture(&wgpu::TextureDescriptor { label: Some(label), ..super::uniforms::probe_cube_descriptor(res, layers) });
        let (lamps, sun) = (source("probe_relight_lamps"), source("probe_relight_sun"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe_relight"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2Array },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("probe_relight"), source: wgpu::ShaderSource::Wgsl(RELIGHT_WGSL.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe_relight"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("probe_relight"), bind_group_layouts: &[Some(&layout)], ..Default::default() })),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor { label: Some("probe_relight_k"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mips = super::uniforms::probe_mip_levels(res);
        let view = |t: &wgpu::Texture, layer: u32, mip: u32| {
            t.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: layer * 6,
                array_layer_count: Some(6),
                ..Default::default()
            })
        };
        let groups = (0..layers)
            .map(|layer| {
                (0..mips)
                    .map(|mip| {
                        let (l, s, o) = (view(&lamps, layer, mip), view(&sun, layer, mip), view(out, layer, mip));
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("probe_relight"),
                            layout: &layout,
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&l) },
                                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&s) },
                                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&o) },
                                wgpu::BindGroupEntry { binding: 3, resource: uniform.as_entire_binding() },
                            ],
                        })
                    })
                    .collect()
            })
            .collect();
        Self { lamps, sun, pipeline, uniform, groups, held: vec![None; layers as usize], fixed_lit: HashSet::new(), resolution: res }
    }

    fn write(&mut self, queue: &wgpu::Queue, layer: u32, lamps: &[Vec<u8>], sun: &[Vec<u8>]) {
        write_probe_layer(queue, &self.lamps, layer, self.resolution, lamps);
        write_probe_layer(queue, &self.sun, layer, self.resolution, sun);
        if let Some(h) = self.held.get_mut(layer as usize) {
            *h = None;
        }
    }

    /// The layers furthest off `daylight` (any never written first), up to
    /// `most`, relit at it.
    fn relight(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, daylight: f32, live: &HashSet<u32>, most: usize) -> usize {
        let off = |h: Option<f32>| h.map_or(f32::INFINITY, |h| (daylight.max(1e-12) / h.max(1e-12)).ln().abs());
        let mut due: Vec<(u32, f32)> = live
            .iter()
            .filter(|l| !self.fixed_lit.contains(l))
            .filter_map(|&l| self.held.get(l as usize).map(|&h| (l, off(h))))
            .filter(|&(_, e)| e > (1.0 + RELIGHT_OFF_BY).ln())
            .collect();
        if due.is_empty() {
            return 0;
        }
        due.sort_by(|a, b| b.1.total_cmp(&a.1));
        due.truncate(most);
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&[daylight, 0.0, 0.0, 0.0]));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("probe_relight") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("probe_relight"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            for &(layer, _) in &due {
                for (mip, group) in self.groups[layer as usize].iter().enumerate() {
                    let r = (self.resolution >> mip).max(1);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(r.div_ceil(8), r.div_ceil(8), 6);
                }
                self.held[layer as usize] = Some(daylight);
            }
        }
        queue.submit(Some(encoder.finish()));
        due.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ProbeDesc {

    pub centre: Vec3,

    pub min: Vec3,
    pub max: Vec3,

    pub volume: u32,

    pub has_depth: bool,

    pub room_light: Option<[[f32; 3]; 9]>,
}

pub fn outdoor_volume(descs: &[ProbeDesc]) -> Option<u32> {
    if !descs.iter().any(|d| d.has_depth) {
        return None;
    }
    let mut without: Vec<u32> = descs
        .iter()
        .filter(|d| !descs.iter().any(|o| o.volume == d.volume && o.has_depth))
        .map(|d| d.volume)
        .collect();
    without.sort_unstable();
    without.dedup();
    if without.len() > 1 {
        log::warn!("reflection probes: {} volumes have no distances; only the first is treated as outdoors", without.len());
    }
    without.first().copied()
}

pub const PROBE_POOL_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

pub const PROBE_UPLOADS_PER_FRAME: usize = 2;

pub fn probe_bytes(res: u32) -> u64 {
    let mut total = 0u64;
    for level in 0..super::uniforms::probe_mip_levels(res) {
        let r = u64::from((res >> level).max(1));
        total += r * r * 8 * 6;
    }
    total
}

pub fn pool_size(count: usize, res: u32, device_cubes: u32) -> u32 {
    let budget = (PROBE_POOL_BUDGET_BYTES / probe_bytes(res).max(1)) as u32;
    let floor = (2 * MAX_PROBES) as u32;
    (count as u32).min(budget.max(floor)).min(device_cubes).max(1)
}

pub struct LayerPool {
    owner: Vec<Option<usize>>,
    layer_of: HashMap<usize, u32>,
    last_used: Vec<u64>,
    frame: u64,
}

impl LayerPool {
    pub fn new(layers: u32) -> Self {
        Self {
            owner: vec![None; layers as usize],
            layer_of: HashMap::new(),
            last_used: vec![0; layers as usize],
            frame: 1,
        }
    }

    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    pub fn touch(&mut self, probe: usize) -> Option<u32> {
        let layer = *self.layer_of.get(&probe)?;
        self.last_used[layer as usize] = self.frame;
        Some(layer)
    }

    pub fn claim(&mut self, probe: usize) -> Option<u32> {
        if let Some(layer) = self.touch(probe) {
            return Some(layer);
        }
        let layer = match self.owner.iter().position(Option::is_none) {
            Some(free) => free,
            None => self
                .last_used
                .iter()
                .enumerate()
                .filter(|(_, &used)| used < self.frame)
                .min_by_key(|(_, &used)| used)
                .map(|(layer, _)| layer)?,
        };
        if let Some(old) = self.owner[layer].take() {
            self.layer_of.remove(&old);
        }
        self.owner[layer] = Some(probe);
        self.layer_of.insert(probe, layer as u32);
        self.last_used[layer] = self.frame;
        Some(layer as u32)
    }

    pub fn layers(&self) -> u32 {
        self.owner.len() as u32
    }

    pub fn is_resident(&self, probe: usize) -> bool {
        self.layer_of.contains_key(&probe)
    }
}

pub struct ProbeStream {
    pool: LayerPool,
    texture: wgpu::Texture,

    depth_texture: wgpu::Texture,
    resolution: u32,
    count: usize,
    requests: Option<mpsc::Sender<Request>>,
    results: mpsc::Receiver<Done>,
    ready: HashMap<usize, Prepared>,
    requested: HashSet<usize>,
    failed: HashSet<usize>,

    refreshing: HashMap<usize, u32>,
    refreshed: HashMap<usize, Chain>,

    fixed_ready: Vec<(u32, Vec<Vec<u8>>)>,

    sky_layer: Option<u32>,

    building_layer: Option<u32>,
    building_cubes: u32,

    /// The GPU relight (`new_smooth`); None relights on the CPU.
    relight: Option<Relight>,
}

/// What `ProbeStream::new_smooth` relights from.
pub struct SmoothSources {
    /// Each probe's lamps-only photograph (`TodState::probe_lamps`).
    pub lamps: Option<ProbeSource>,
    /// Each building's photograph and its lamps-only one.
    pub buildings: Vec<(Vec<u8>, Option<Vec<u8>>)>,
    /// The daylight to relight at first.
    pub daylight: f32,
}

impl ProbeStream {

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resolution: u32,
        count: usize,
        source: ProbeSource,
    ) -> Self {
        Self::new_with_depth(device, queue, resolution, count, source, None, None)
    }

    pub fn new_with_depth(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resolution: u32,
        count: usize,
        source: ProbeSource,
        depth: Option<ProbeDepthSource>,
        sky: Option<Vec<u8>>,
    ) -> Self {
        Self::new_with_extras(device, queue, resolution, count, source, depth, sky, Vec::new())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_extras(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resolution: u32,
        count: usize,
        source: ProbeSource,
        depth: Option<ProbeDepthSource>,
        sky: Option<Vec<u8>>,
        buildings: Vec<Vec<u8>>,
    ) -> Self {
        let buildings = buildings.into_iter().map(|f| (f, None)).collect();
        Self::build(device, queue, resolution, count, source, depth, sky, buildings, None)
    }

    /// As `new_with_extras`, the probes and buildings relit on the GPU at
    /// every frame's daylight (`relight`) from `source`'s full photographs
    /// and `smooth`'s lamps-only ones. The sky's layer is written lit.
    #[allow(clippy::too_many_arguments)]
    pub fn new_smooth(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resolution: u32,
        count: usize,
        source: ProbeSource,
        depth: Option<ProbeDepthSource>,
        sky: Option<Vec<u8>>,
        smooth: SmoothSources,
    ) -> Self {
        let SmoothSources { lamps, buildings, daylight } = smooth;
        let mut stream = Self::build(device, queue, resolution, count, source, depth, sky, buildings, Some(lamps));
        let n = stream.texture.depth_or_array_layers() / 6;
        stream.relight_at(device, queue, daylight, n as usize);
        stream
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resolution: u32,
        count: usize,
        source: ProbeSource,
        depth: Option<ProbeDepthSource>,
        sky: Option<Vec<u8>>,
        buildings: Vec<(Vec<u8>, Option<Vec<u8>>)>,
        smooth: Option<Option<ProbeSource>>,
    ) -> Self {
        let res = resolution.max(1);
        let sky_cubes = u32::from(sky.is_some());
        let building_cubes = buildings.len() as u32;
        let fixed = sky_cubes + building_cubes;
        let allowed = super::uniforms::probe_layers_allowed(device).saturating_sub(fixed).max(1);
        let layers = pool_size(count, res, allowed);
        let mut desc = super::uniforms::probe_cube_descriptor(res, layers + fixed);
        if smooth.is_some() {
            desc.usage |= wgpu::TextureUsages::STORAGE_BINDING;
        }
        let texture = device.create_texture(&desc);
        let mut relight = smooth.is_some().then(|| Relight::new(device, &texture, res, layers + fixed));
        let depth_texture = device.create_texture(&super::uniforms::probe_depth_descriptor(res, layers + fixed));

        let sky_layer = sky.and_then(|faces| super::uniforms::prefilter_probe(&faces, res)).map(|chain| {
            write_probe_layer(queue, &texture, layers, res, &chain);
            layers
        });
        if let (Some(r), Some(l)) = (relight.as_mut(), sky_layer) {
            r.fixed_lit.insert(l);
        }

        let first_building = layers + sky_cubes;
        let mut written = 0u32;
        for (faces, lamps) in &buildings {
            let layer = first_building + written;
            match relight.as_mut() {
                Some(r) => {
                    let (l, s) = split_probe_faces(faces, lamps.as_deref(), true);
                    if let (Some(l), Some(s)) = (prefilter_probe(&l, res), prefilter_probe(&s, res)) {
                        r.write(queue, layer, &l, &s);
                    }
                }
                None => {
                    if let Some(chain) = prefilter_probe(faces, res) {
                        write_probe_layer(queue, &texture, layer, res, &chain);
                    }
                }
            }
            written += 1;
        }
        let building_layer = (building_cubes > 0).then_some(first_building);
        let mut pool = LayerPool::new(layers);

        let prepare = {
            let source = source.clone();
            let depth = depth.clone();
            let smooth = smooth.clone();
            move |i: usize| -> Option<Prepared> {
                let faces = source(i)?;
                let chain = match &smooth {
                    None => Chain::Lit(prefilter_probe(&faces, res)?),
                    Some(lamps) => {
                        let lamp_faces = lamps.as_ref().and_then(|l| l(i));
                        let (l, s) = split_probe_faces(&faces, lamp_faces.as_deref(), false);
                        Chain::Split(prefilter_probe(&l, res)?, prefilter_probe(&s, res)?)
                    }
                };
                Some((chain, depth.as_ref().and_then(|d| d(i))))
            }
        };

        let everything_fits = count <= layers as usize;
        if everything_fits {
            for i in 0..count {
                let Some((chain, dist)) = prepare(i) else { continue };
                if let Some(layer) = pool.claim(i) {
                    write_chain(queue, &texture, relight.as_mut(), layer, res, chain);

                    let dist = dist.unwrap_or_else(|| vec![0; (res * res * 6 * 4) as usize]);
                    write_probe_depth_layer(queue, &depth_texture, layer, res, &dist);
                }
            }
        }
        log::info!(
            "reflection probes: {count} in the level, pool of {layers} ({} MB){}",
            probe_bytes(res) * u64::from(layers) / (1024 * 1024),
            if everything_fits { ", all resident" } else { ", streaming" },
        );

        let (req_tx, req_rx) = mpsc::channel::<Request>();
        let (res_tx, res_rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("probe_stream".into())
            .spawn(move || {
                while let Ok(request) = req_rx.recv() {
                    let done = match request {
                        Request::Probe(i) => Done::Probe(i, prepare(i)),
                        Request::Fixed(layer, faces) => Done::Fixed(layer, faces().and_then(|f| prefilter_probe(&f, res))),
                    };
                    if res_tx.send(done).is_err() {
                        break;
                    }
                }
            })
            .ok();
        Self {
            pool,
            texture,
            depth_texture,
            resolution: res,
            count,
            requests: Some(req_tx),
            results: res_rx,
            ready: HashMap::new(),
            requested: HashSet::new(),
            failed: HashSet::new(),
            refreshing: HashMap::new(),
            refreshed: HashMap::new(),
            fixed_ready: Vec::new(),
            sky_layer,
            building_layer,
            building_cubes,
            relight,
        }
    }

    /// Whether the probes are relit on the GPU (`new_smooth`).
    pub fn smooth(&self) -> bool {
        self.relight.is_some()
    }

    /// The probes and buildings furthest off `daylight`, relit at it: up to
    /// `RELIGHTS_PER_FRAME` a frame, any just streamed in first. Call after
    /// `resolve`, before the frame's passes are submitted.
    pub fn relight(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, daylight: f32) -> usize {
        self.relight_at(device, queue, daylight, RELIGHTS_PER_FRAME)
    }

    fn relight_at(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, daylight: f32, most: usize) -> usize {
        let Some(r) = self.relight.as_mut() else { return 0 };
        let mut live: HashSet<u32> = self.pool.layer_of.values().copied().collect();
        if let Some(first) = self.building_layer {
            live.extend(first..first + self.building_cubes);
        }
        r.relight(device, queue, daylight, &live, most)
    }

    pub fn sky_layer(&self) -> Option<u32> {
        self.sky_layer
    }

    pub fn building_layer(&self) -> Option<u32> {
        self.building_layer
    }

    pub fn view(&self) -> wgpu::TextureView {
        self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("probe_cube_array_view"),
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        })
    }

    pub fn begin_frame(&mut self) {
        self.pool.begin_frame();
        while let Ok(done) = self.results.try_recv() {
            let (i, chain) = match done {
                Done::Probe(i, chain) => (i, chain),
                Done::Fixed(layer, chain) => {
                    if let Some(chain) = chain {
                        self.fixed_ready.retain(|(l, _)| *l != layer);
                        self.fixed_ready.push((layer, chain));
                    }
                    continue;
                }
            };

            if let Some(n) = self.refreshing.get_mut(&i) {
                *n -= 1;
                if *n == 0 {
                    self.refreshing.remove(&i);
                }
                if let Some((chain, _)) = chain {
                    self.refreshed.insert(i, chain);
                }
                continue;
            }
            if let Some(chain) = chain {
                self.ready.insert(i, chain);
            } else {

                log::warn!("reflection probe {i} could not be decoded; it is skipped");
                self.failed.insert(i);
            }
            self.requested.remove(&i);
        }
    }

    pub fn resolve(&mut self, queue: &wgpu::Queue, upload: &mut ProbeUpload) {
        let mut uploads = 0;

        let mut kept = ProbeUpload { count: 0, boxes: [[[0.0; 4]; 3]; MAX_PROBES], ..*upload };
        let mut n = 0usize;
        for slot in 0..(upload.count as usize).min(MAX_PROBES) {
            let probe = upload.layer(slot) as usize;
            if probe >= self.count || self.failed.contains(&probe) {
                continue;
            }
            let layer = match self.pool.touch(probe) {
                Some(layer) => Some(layer),
                None if uploads < PROBE_UPLOADS_PER_FRAME && self.ready.contains_key(&probe) => {
                    let layer = self.pool.claim(probe);
                    if let (Some(layer), Some((chain, dist))) = (layer, self.ready.remove(&probe)) {
                        write_chain(queue, &self.texture, self.relight.as_mut(), layer, self.resolution, chain);

                        let res = self.resolution;
                        let dist = dist.unwrap_or_else(|| vec![0; (res * res * 6 * 4) as usize]);
                        write_probe_depth_layer(queue, &self.depth_texture, layer, res, &dist);
                        uploads += 1;
                    }
                    layer
                }
                None => {
                    if !self.ready.contains_key(&probe) && self.requested.insert(probe) {
                        if let Some(tx) = &self.requests {
                            let _ = tx.send(Request::Probe(probe));
                        }
                    }
                    None
                }
            };
            if let Some(layer) = layer {
                kept.boxes[n] = upload.boxes[slot];
                kept.boxes[n][0][3] = layer as f32;
                n += 1;
            }
        }
        kept.count = n as u32;
        *upload = kept;

        while uploads < PROBE_UPLOADS_PER_FRAME {
            if let Some((layer, chain)) = self.fixed_ready.pop() {
                write_probe_layer(queue, &self.texture, layer, self.resolution, &chain);
                uploads += 1;
                continue;
            }
            let Some(&i) = self.refreshed.keys().next() else { break };
            let chain = self.refreshed.remove(&i).expect("just listed");
            if let Some(&layer) = self.pool.layer_of.get(&i) {
                write_chain(queue, &self.texture, self.relight.as_mut(), layer, self.resolution, chain);
                uploads += 1;
            }
        }
    }

    /// Every resident probe, the sky and the buildings prepared again (the
    /// probes from the source as it is now). Relit on the GPU (`smooth`),
    /// the probes and buildings follow the daylight without it: only the
    /// sky is taken.
    pub fn refresh(&mut self, sky: Option<FixedFaces>, buildings: Vec<FixedFaces>) {
        let Some(tx) = &self.requests else { return };
        if self.relight.is_some() {
            if let (Some(layer), Some(faces)) = (self.sky_layer, sky) {
                let _ = tx.send(Request::Fixed(layer, faces));
            }
            return;
        }
        self.ready.clear();
        self.requested.clear();
        let resident: Vec<usize> = self.pool.layer_of.keys().copied().collect();
        for i in resident {
            if tx.send(Request::Probe(i)).is_ok() {
                *self.refreshing.entry(i).or_insert(0) += 1;
            }
        }
        if let (Some(layer), Some(faces)) = (self.sky_layer, sky) {
            let _ = tx.send(Request::Fixed(layer, faces));
        }
        if let Some(first) = self.building_layer {
            for (k, faces) in buildings.into_iter().enumerate() {
                let _ = tx.send(Request::Fixed(first + k as u32, faces));
            }
        }
    }

    pub fn refreshing(&self) -> bool {
        !self.refreshing.is_empty() || !self.refreshed.is_empty() || !self.fixed_ready.is_empty()
    }

    pub fn depth_view(&self) -> wgpu::TextureView {
        self.depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("probe_depth_array_view"),
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        })
    }

    pub fn pool_layers(&self) -> u32 {
        self.pool.layers()
    }
}

/// `chain` into `layer`: as it is when lit, into the relight's sources
/// (relit at the next `relight`, ahead of every other layer) when split.
fn write_chain(queue: &wgpu::Queue, texture: &wgpu::Texture, relight: Option<&mut Relight>, layer: u32, res: u32, chain: Chain) {
    match (chain, relight) {
        (Chain::Lit(c), relight) => {
            write_probe_layer(queue, texture, layer, res, &c);
            if let Some(r) = relight {
                r.fixed_lit.insert(layer);
            }
        }
        (Chain::Split(l, s), Some(r)) => {
            r.fixed_lit.remove(&layer);
            r.write(queue, layer, &l, &s);
        }
        (Chain::Split(l, s), None) => {
            // Never made: a split chain comes only from a smooth stream.
            let lit: Vec<Vec<u8>> = l.iter().zip(&s).map(|(l, s)| relight_bytes(l, s, 1.0)).collect();
            write_probe_layer(queue, texture, layer, res, &lit);
        }
    }
}

/// `lamps + k * sun` on the CPU, alpha `sun`'s: the relight shader's twin.
pub fn relight_bytes(lamps: &[u8], sun: &[u8], k: f32) -> Vec<u8> {
    use super::uniforms::{f16_to_f32, f32_to_f16};
    let mut out = Vec::with_capacity(sun.len());
    for (t, (l, s)) in lamps.chunks_exact(2).zip(sun.chunks_exact(2)).enumerate() {
        let (l, s) = (f16_to_f32(u16::from_le_bytes([l[0], l[1]])), f16_to_f32(u16::from_le_bytes([s[0], s[1]])));
        let v = if t % 4 == 3 { s } else { l + k * s };
        out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|&x| super::super::uniforms::f32_to_f16(x).to_le_bytes()).collect()
    }

    #[test]
    fn a_split_probe_relit_matches_the_cpu_relight_at_every_daylight() {
        let full = half(&[0.9, 0.5, 0.2, 0.7, 0.1, 0.4, 0.05, 1.0, 2.0, 0.0, 3.0, 0.25]);
        let lamps = half(&[0.3, 0.6, 0.1, 0.2, 0.1, 0.1, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5]);
        let (l, s) = split_probe_faces(&full, Some(&lamps), false);
        for k in [0.0, 1e-4, 0.37, 1.0] {
            let cpu = crate::renderer::time_of_day::relight_probe_faces(&full, &lamps, k);
            let gpu = relight_bytes(&l, &s, k);
            let read = |b: &[u8]| -> Vec<f32> { b.chunks_exact(2).map(|c| super::super::uniforms::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect() };
            for (a, b) in read(&cpu).iter().zip(read(&gpu)) {
                assert!((a - b).abs() <= a.abs() * 2e-3 + 1e-6, "daylight {k}: {a} vs {b}");
            }
        }
    }

    fn read_layer(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, layer: u32, mip: u32, side: u32) -> Vec<u8> {
        let row = (side * 8).div_ceil(256) * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row * side * 6),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: tex, mip_level: mip, origin: wgpu::Origin3d { x: 0, y: 0, z: layer * 6 }, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(side) } },
            wgpu::Extent3d { width: side, height: side, depth_or_array_layers: 6 },
        );
        queue.submit(Some(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().unwrap();
        let mut out = Vec::new();
        for r in 0..(side * 6) as usize {
            out.extend_from_slice(&data[r * row as usize..r * row as usize + (side * 8) as usize]);
        }
        out
    }

    /// The GPU relight against the CPU one, read back from the probe array:
    /// what a reflection samples, at two daylights in turn.
    #[test]
    fn the_gpu_relight_writes_what_the_cpu_relight_would() {
        use super::super::uniforms::{f16_to_f32, scene_limits};
        let instance = wgpu::Instance::default();
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let Ok((device, queue)) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: scene_limits(wgpu::Limits::default()),
            ..Default::default()
        })) else {
            return;
        };
        const RES: u32 = 16;
        let n = (RES * RES * 6) as usize;
        let cube = |f: &dyn Fn(usize) -> [f32; 4]| half(&(0..n).flat_map(f).collect::<Vec<f32>>());
        let full = Arc::new(cube(&|i| [0.2 + (i % 7) as f32 * 0.3, 0.5, 1.0 + (i % 3) as f32, if i % 5 == 0 { 0.0 } else { 1.0 }]));
        let lamps = Arc::new(cube(&|i| [0.1 * (i % 4) as f32, 0.6, 0.3, 1.0]));
        let (f2, l2) = (full.clone(), lamps.clone());
        let source: ProbeSource = Arc::new(move |_| Some(f2.to_vec()));
        let lamp_source: ProbeSource = Arc::new(move |_| Some(l2.to_vec()));
        let smooth = SmoothSources { lamps: Some(lamp_source), buildings: Vec::new(), daylight: 0.37 };
        let mut stream = ProbeStream::new_smooth(&device, &queue, RES, 1, source, None, None, smooth);
        let read = |b: &[u8]| -> Vec<f32> { b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect() };
        for (k, step) in [(0.37f32, false), (0.9, true)] {
            if step {
                assert_eq!(stream.relight(&device, &queue, k), 1, "a 2.4x daylight relights the layer");
                assert_eq!(stream.relight(&device, &queue, k * 1.01), 0, "one percent does not");
            }
            let want = prefilter_probe(&crate::renderer::time_of_day::relight_probe_faces(&full, &lamps, k), RES).unwrap();
            for mip in [0u32, 2] {
                let side = RES >> mip;
                let got = read(&read_layer(&device, &queue, &stream.texture, 0, mip, side));
                let exp = read(&want[mip as usize]);
                assert_eq!(got.len(), exp.len());
                for (a, b) in got.iter().zip(&exp) {
                    assert!((a - b).abs() <= b.abs() * 4e-3 + 1e-4, "daylight {k} mip {mip}: gpu {a} cpu {b}");
                }
            }
        }
    }

    /// The relight's cost on this GPU at 256 px: `RELIGHTS_PER_FRAME`
    /// layers a frame, 60 frames, wall clock to the GPU's finish.
    #[test]
    #[ignore]
    fn relight_cost() {
        use super::super::uniforms::scene_limits;
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: scene_limits(adapter.limits()),
            ..Default::default()
        }))
        .unwrap();
        const RES: u32 = 256;
        let n = (RES * RES * 6) as usize;
        let full = half(&(0..n).flat_map(|i| [0.5 + (i % 7) as f32, 0.5, 1.0, 1.0]).collect::<Vec<f32>>());
        let source: ProbeSource = Arc::new(move |_| Some(full.clone()));
        let smooth = SmoothSources { lamps: None, buildings: Vec::new(), daylight: 1.0 };
        let mut stream = ProbeStream::new_smooth(&device, &queue, RES, 13, source, None, None, smooth);
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let mut k = 1.0f32;
        let t = std::time::Instant::now();
        let mut layers = 0;
        for _ in 0..60 {
            k *= 0.9;
            layers += stream.relight(&device, &queue, k);
        }
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("relight {RES} px: {layers} layers in 60 frames, {:.3} ms a frame, {:.3} ms a layer", ms / 60.0, ms / layers as f64);
    }

    #[test]
    fn a_split_buildings_outside_takes_only_daylight() {
        let full = half(&[0.9, 0.5, 0.2, 0.7]);
        let lamps = half(&[0.3, 0.6, 0.1, 0.2]);
        let (l, s) = split_probe_faces(&full, Some(&lamps), true);
        let cpu = crate::renderer::time_of_day::building_daylight_faces(&full, &lamps, 0.5);
        assert_eq!(relight_bytes(&l, &s, 0.5), cpu);
    }

    #[test]
    #[ignore]
    fn prefilter_cost() {
        for res in [128u32, 256] {
            let faces: Vec<u8> = (0..res * res * 6).flat_map(|i| {
                let v = super::super::uniforms::f32_to_f16(0.1 + (i % 97) as f32 * 0.01).to_le_bytes();
                let one = super::super::uniforms::f32_to_f16(1.0).to_le_bytes();
                [v[0], v[1], v[0], v[1], v[0], v[1], one[0], one[1]]
            }).collect();
            let t = std::time::Instant::now();
            let chain = prefilter_probe(&faces, res).unwrap();
            let bytes: usize = chain.iter().map(Vec::len).sum();
            eprintln!("prefilter {res} px: {:.1} ms, {:.1} MB chain", t.elapsed().as_secs_f64() * 1e3, bytes as f64 / 1048576.0);
        }
    }

    fn desc(volume: u32, has_depth: bool) -> ProbeDesc {
        ProbeDesc { centre: Vec3::ZERO, min: Vec3::ZERO, max: Vec3::ONE, volume, has_depth,
            room_light: None
        }
    }

    #[test]
    fn the_outdoors_is_the_volume_with_no_distances() {
        let level = [desc(0, true), desc(0, true), desc(1, true), desc(2, false)];
        assert_eq!(outdoor_volume(&level), Some(2));

        assert_eq!(outdoor_volume(&[desc(0, true), desc(0, false), desc(1, false)]), Some(1));

        assert_eq!(outdoor_volume(&[desc(0, false), desc(1, false)]), None);
        assert_eq!(outdoor_volume(&[]), None);
    }

    #[test]
    fn free_layers_are_used_before_anything_is_evicted() {
        let mut pool = LayerPool::new(3);
        assert_eq!(pool.claim(10), Some(0));
        assert_eq!(pool.claim(11), Some(1));
        assert_eq!(pool.claim(12), Some(2));
        assert!(pool.is_resident(10) && pool.is_resident(11) && pool.is_resident(12));
    }

    #[test]
    fn eviction_takes_the_stalest_layer_the_frame_is_not_using() {
        let mut pool = LayerPool::new(2);
        pool.claim(1);
        pool.claim(2);
        pool.begin_frame();
        pool.touch(1);
        pool.begin_frame();
        pool.touch(2);

        assert_eq!(pool.claim(3), Some(0));
        assert!(!pool.is_resident(1));
        assert!(pool.is_resident(2) && pool.is_resident(3));
    }

    #[test]
    fn a_layer_in_use_this_frame_is_never_evicted() {
        let mut pool = LayerPool::new(2);
        pool.begin_frame();
        pool.claim(1);
        pool.claim(2);
        assert_eq!(pool.claim(3), None);
        assert!(pool.is_resident(1) && pool.is_resident(2));
    }

    #[test]
    fn a_resident_probe_keeps_its_layer() {
        let mut pool = LayerPool::new(4);
        let a = pool.claim(7);
        assert_eq!(pool.claim(7), a);
        assert_eq!(pool.layer_of.len(), 1);
    }

    #[test]
    fn the_pool_is_sized_by_memory_and_the_device_not_a_constant() {
        assert_eq!(pool_size(9, 128, 341), 9);
        let budget = (PROBE_POOL_BUDGET_BYTES / probe_bytes(128)) as u32;
        assert_eq!(pool_size(5000, 128, 341), budget);
        assert_eq!(pool_size(5000, 128, 42), 42, "the device limit was ignored");

        assert_eq!(pool_size(5000, 16, 341), 341);
        assert!(budget >= (2 * MAX_PROBES) as u32, "the budget cannot hold one frame's slots");
    }

    #[test]
    fn resolve_keeps_everything_but_the_slots() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU");
            return;
        };
        const RES: u32 = 4;
        let faces = vec![0u8; (RES * RES * 8 * 6) as usize];
        let source: ProbeSource = Arc::new(move |_| Some(faces.clone()));
        let mut stream = ProbeStream::new(&device, &queue, RES, 1, source);
        let mut up = ProbeUpload { count: 1, portal_count: 2, proxy_count: 3, no_trace: true, open_ground: true, ..Default::default() };
        up.portals[1][0][0] = 7.0;
        up.proxies[2][1][2] = 9.0;
        up.set(0, 0, Vec3::ZERO, Vec3::splat(-1.0), Vec3::splat(1.0));
        stream.begin_frame();
        stream.resolve(&queue, &mut up);
        assert_eq!((up.portal_count, up.proxy_count, up.no_trace, up.open_ground), (2, 3, true, true), "resolve dropped a field");
        assert_eq!(up.portals[1][0][0], 7.0, "the doorways were lost");
        assert_eq!(up.proxies[2][1][2], 9.0, "the reflection proxies were lost");
    }

    #[test]
    fn a_probe_past_the_pool_streams_in() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU");
            return;
        };
        const RES: u32 = 4;
        let device_cubes = super::super::uniforms::probe_layers_allowed(&device);
        let count = device_cubes as usize + 8;
        let faces = vec![0u8; (RES * RES * 8 * 6) as usize];
        let source: ProbeSource = Arc::new(move |_| Some(faces.clone()));
        let mut stream = ProbeStream::new(&device, &queue, RES, count, source);
        assert!((stream.pool_layers() as usize) < count, "the level was supposed not to fit");

        let want = count - 1;
        let ask = |stream: &mut ProbeStream| {
            let mut up = ProbeUpload { count: 1, ..Default::default() };
            up.set(0, want as u32, Vec3::ZERO, Vec3::splat(-1.0), Vec3::splat(1.0));
            up.set_volume(0, 3);
            stream.resolve(&queue, &mut up);
            up
        };
        stream.begin_frame();
        assert_eq!(ask(&mut stream).count, 0, "a probe not in the pool was pointed at a layer");
        let mut got = None;
        for _ in 0..500 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            stream.begin_frame();
            let up = ask(&mut stream);
            if up.count == 1 {
                got = Some(up);
                break;
            }
        }
        let up = got.expect("the requested probe never streamed in");
        assert!(up.layer(0) < stream.pool_layers(), "streamed into a layer the pool does not have");
        assert_eq!(up.volume(0), 3, "the slot lost its room on the way through");
    }

    #[test]
    fn a_probe_costs_about_a_megabyte_at_128() {
        let mb = probe_bytes(128) as f64 / (1024.0 * 1024.0);
        assert!((0.95..1.1).contains(&mb), "{mb} MB");
    }
}
