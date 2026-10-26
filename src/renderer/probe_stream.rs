use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc};

use glam::Vec3;

use super::uniforms::{prefilter_probe, write_probe_depth_layer, write_probe_layer, ProbeUpload, MAX_PROBES};

pub type ProbeSource = Arc<dyn Fn(usize) -> Option<Vec<u8>> + Send + Sync>;

pub type ProbeDepthSource = Arc<dyn Fn(usize) -> Option<Vec<u16>> + Send + Sync>;

type Prepared = (Vec<Vec<u8>>, Option<Vec<u16>>);

pub type FixedFaces = Box<dyn FnOnce() -> Option<Vec<u8>> + Send>;

enum Request {

    Probe(usize),

    Fixed(u32, FixedFaces),
}

enum Done {
    Probe(usize, Option<Prepared>),
    Fixed(u32, Option<Vec<Vec<u8>>>),
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
    refreshed: HashMap<usize, Vec<Vec<u8>>>,

    fixed_ready: Vec<(u32, Vec<Vec<u8>>)>,

    sky_layer: Option<u32>,

    building_layer: Option<u32>,
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

    #[allow(clippy::too_many_arguments, reason = "GPU setup passes each resource explicitly; bundling them is a separate refactor")]
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
        let res = resolution.max(1);
        let sky_cubes = u32::from(sky.is_some());
        let building_cubes = buildings.len() as u32;
        let fixed = sky_cubes + building_cubes;
        let allowed = super::uniforms::probe_layers_allowed(device).saturating_sub(fixed).max(1);
        let layers = pool_size(count, res, allowed);
        let texture = device.create_texture(&super::uniforms::probe_cube_descriptor(res, layers + fixed));
        let depth_texture = device.create_texture(&super::uniforms::probe_depth_descriptor(res, layers + fixed));

        let sky_layer = sky.and_then(|faces| super::uniforms::prefilter_probe(&faces, res)).map(|chain| {
            write_probe_layer(queue, &texture, layers, res, &chain);
            layers
        });

        let first_building = layers + sky_cubes;
        for (written, faces) in (0u32..).zip(&buildings) {
            if let Some(chain) = super::uniforms::prefilter_probe(faces, res) {
                write_probe_layer(queue, &texture, first_building + written, res, &chain);
            }
        }
        let building_layer = (building_cubes > 0).then_some(first_building);
        let mut pool = LayerPool::new(layers);

        let prepare = {
            let source = source.clone();
            let depth = depth.clone();
            move |i: usize| -> Option<Prepared> {
                let chain = source(i).and_then(|faces| prefilter_probe(&faces, res))?;
                Some((chain, depth.as_ref().and_then(|d| d(i))))
            }
        };

        let everything_fits = count <= layers as usize;
        if everything_fits {
            for i in 0..count {
                let Some((chain, dist)) = prepare(i) else { continue };
                if let Some(layer) = pool.claim(i) {
                    write_probe_layer(queue, &texture, layer, res, &chain);

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
        }
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
                        write_probe_layer(queue, &self.texture, layer, self.resolution, &chain);

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
                write_probe_layer(queue, &self.texture, layer, self.resolution, &chain);
                uploads += 1;
            }
        }
    }

    pub fn refresh(&mut self, sky: Option<FixedFaces>, buildings: Vec<FixedFaces>) {
        let Some(tx) = &self.requests else { return };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "diagnostic that prints sizes or timings or writes files; run by hand with --ignored"]
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
        let mut up = ProbeUpload { count: 1, portal_count: 2, proxy_count: 3, no_trace: true, ..Default::default() };
        up.portals[1][0][0] = 7.0;
        up.proxies[2][1][2] = 9.0;
        up.set(0, 0, Vec3::ZERO, Vec3::splat(-1.0), Vec3::splat(1.0));
        stream.begin_frame();
        stream.resolve(&queue, &mut up);
        assert_eq!((up.portal_count, up.proxy_count, up.no_trace), (2, 3, true), "resolve dropped a field");
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
