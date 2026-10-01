//! Reflection probes for a level of any size.
//!
//! A level used to be capped at the probes its cube array could hold at once
//! (32), and anything past that was dropped at load with no message -- a room
//! reflecting the sky indoors because the level was too big. The cap looked
//! like hardware and was not: it came from the 256 array layers of wgpu's
//! conservative default limits, while the Quest 3's Adreno 740 reports 2048.
//!
//! The real budget is MEMORY. A 128 px half-float cube with its mip chain is
//! about 1 MB, so a large level's worth held at once is hundreds of megabytes
//! of GPU memory for photographs of rooms nobody is standing in. So:
//!
//! - the level's probes are DESCRIBED up front (box, capture point, room) and
//!   their pixels come from a [`ProbeSource`] on demand -- on the headset, a
//!   decode from disk, so nothing holds every probe;
//! - the GPU holds a fixed POOL of cube layers, sized from a memory budget and
//!   the device's real limit, and residency (the slots a frame's shader walks)
//!   picks from every probe in the level;
//! - a probe residency wants that is not in the pool is prefiltered on a
//!   background thread and uploaded a few per frame, into a free layer or the
//!   least recently used one that the current frame does not need.
//!
//! A level whose probes all fit is uploaded completely at load, exactly as
//! before, so a small level behaves identically and deterministically.

use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc};

use glam::Vec3;

use super::uniforms::{prefilter_probe, write_probe_depth_layer, write_probe_layer, ProbeUpload, MAX_PROBES};

/// Decodes probe `i`'s six faces as linear RGBA half floats, in cube order --
/// `resolution^2 * 8` bytes a face, as `reflection_probe::LoadedProbe::faces`.
///
/// A function rather than the bytes, so the caller decides where they live: a
/// test hands over what it already has, the headset reads the file again.
pub type ProbeSource = Arc<dyn Fn(usize) -> Option<Vec<u8>> + Send + Sync>;

/// Decodes probe `i`'s per-texel depth planes: four half floats a texel, six
/// stacked faces, as `reflection_probe::decode_probe_depth` returns them.
pub type ProbeDepthSource = Arc<dyn Fn(usize) -> Option<Vec<u16>> + Send + Sync>;

/// A probe ready to upload: its radiance mip chain and, when baked, its
/// distances.
type Prepared = (Vec<Vec<u8>>, Option<Vec<u16>>);

/// Everything about a probe except its pixels.
#[derive(Clone, Copy, Debug)]
pub struct ProbeDesc {
    /// Where the photograph was taken.
    pub centre: Vec3,
    /// The room's box, which reflections are projected onto.
    pub min: Vec3,
    pub max: Vec3,
    /// Which room this probe photographs. Cells of one room share it.
    pub volume: u32,
    /// Whether the bake gave it distances. A volume none of whose photographs
    /// has them is the OUTDOOR volume -- its box stands in for the sky dome --
    /// which reflections trace against the ground and the sky instead.
    pub has_depth: bool,
    /// The room's light round the capture point, for the models standing in
    /// it: see `room_light`. `None` from an older bake.
    pub room_light: Option<[[f32; 3]; 9]>,
}

/// THE OUTDOOR VOLUME of a level's probes: the one volume none of whose
/// photographs has distances, when others do. Its box stands in for the sky
/// dome, so reflections starting in it are traced against the ground and the
/// sky instead (`outdoor_radiance` in the shader). `None` when no probe has
/// distances -- an older bake -- since then nothing tells the outdoors apart.
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

/// The most GPU memory the probe pool may take, in bytes.
///
/// 64 MB is ~60 probes at 128 px: every room and doorway within sight of the
/// player several times over, and a small fraction of what a Quest app has.
pub const PROBE_POOL_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

/// How many probes may be uploaded in one frame.
///
/// A 128 px probe's chain is ~1 MB of `write_texture`; two keep a streaming
/// burst -- walking into a new wing -- well inside a frame's slack.
pub const PROBE_UPLOADS_PER_FRAME: usize = 2;

/// Bytes one probe's cube and mip chain take on the GPU.
pub fn probe_bytes(res: u32) -> u64 {
    let mut total = 0u64;
    for level in 0..super::uniforms::probe_mip_levels(res) {
        let r = (res >> level).max(1) as u64;
        total += r * r * 8 * 6;
    }
    total
}

/// How many cubes the pool holds for a level of `count` probes.
///
/// Never more than the level has, never more than the device allows or the
/// budget pays for -- but at least what one frame's two eyes can ask for,
/// where the device allows it, or eviction would fight itself inside a frame.
pub fn pool_size(count: usize, res: u32, device_cubes: u32) -> u32 {
    let budget = (PROBE_POOL_BUDGET_BYTES / probe_bytes(res).max(1)) as u32;
    let floor = (2 * MAX_PROBES) as u32;
    (count as u32).min(budget.max(floor)).min(device_cubes).max(1)
}

/// Which probe lives in which cube layer, and which layer to give up next.
///
/// No GPU in here, so the policy is tested directly.
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

    /// Start a frame. Layers used from here on are protected from eviction
    /// until the next call.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// The layer holding `probe`, marking it used this frame.
    pub fn touch(&mut self, probe: usize) -> Option<u32> {
        let layer = *self.layer_of.get(&probe)?;
        self.last_used[layer as usize] = self.frame;
        Some(layer)
    }

    /// A layer for `probe`: a free one if there is one, otherwise the least
    /// recently used one that THIS frame has not used. `None` when every layer
    /// is in use this frame -- the frame wants more probes than the pool holds,
    /// and evicting one it is drawing with would show the wrong photograph.
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

/// A level's probes: the pool on the GPU, and the thread that fills it.
pub struct ProbeStream {
    pool: LayerPool,
    texture: wgpu::Texture,
    /// Distances, a layer beside each of `texture`'s. See
    /// `uniforms::probe_depth_descriptor`. All zero -- "none baked" -- when
    /// the level has none.
    depth_texture: wgpu::Texture,
    resolution: u32,
    count: usize,
    requests: Option<mpsc::Sender<usize>>,
    results: mpsc::Receiver<(usize, Option<Prepared>)>,
    ready: HashMap<usize, Prepared>,
    requested: HashSet<usize>,
    failed: HashSet<usize>,
    /// The cube layer holding the sky reflections see, past the pool's own
    /// layers so eviction never touches it. See `sky::ReflectionSky`.
    sky_layer: Option<u32>,
    /// The first of the buildings' outside cubes, one layer each after the
    /// sky's, never evicted. See `outdoor_radiance` in the shader.
    building_layer: Option<u32>,
}

impl ProbeStream {
    /// The pool for `count` probes at `resolution`, filled completely now when
    /// they all fit and on demand otherwise.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resolution: u32,
        count: usize,
        source: ProbeSource,
    ) -> Self {
        Self::new_with_depth(device, queue, resolution, count, source, None, None)
    }

    /// As [`Self::new`], with each probe's distances streamed beside it.
    /// `sky`, when given, is the sky reflections see as six cube faces at
    /// `resolution` (`sky::ReflectionSky::cube_faces`): it gets a layer of its
    /// own after the pool's, prefiltered like a probe. See [`Self::sky_layer`].
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

    /// As [`Self::new_with_depth`], with the buildings' outside cubes (six
    /// faces each at `resolution`) in layers of their own after the sky's.
    /// See [`Self::building_layer`].
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
        let res = resolution.max(1);
        let sky_cubes = u32::from(sky.is_some());
        let building_cubes = buildings.len() as u32;
        let fixed = sky_cubes + building_cubes;
        let allowed = super::uniforms::probe_layers_allowed(device).saturating_sub(fixed).max(1);
        let layers = pool_size(count, res, allowed);
        let texture = device.create_texture(&super::uniforms::probe_cube_descriptor(res, layers + fixed));
        let depth_texture = device.create_texture(&super::uniforms::probe_depth_descriptor(res, layers + fixed));
        // The sky, prefiltered once, in the layer after the pool's. Its depth
        // layer stays zero: "none baked", which nothing reads for it anyway.
        let sky_layer = sky.and_then(|faces| super::uniforms::prefilter_probe(&faces, res)).map(|chain| {
            write_probe_layer(queue, &texture, layers, res, &chain);
            layers
        });
        // The buildings, after the sky, prefiltered like it: a rough wall
        // reads a building at the blur its lobe has spread to.
        let first_building = layers + sky_cubes;
        let mut written = 0u32;
        for faces in &buildings {
            if let Some(chain) = super::uniforms::prefilter_probe(faces, res) {
                write_probe_layer(queue, &texture, first_building + written, res, &chain);
            }
            written += 1;
        }
        let building_layer = (building_cubes > 0).then_some(first_building);
        let mut pool = LayerPool::new(layers);
        // Radiance and distance in one step, so a layer never holds one
        // probe's picture beside another probe's depth.
        let prepare = {
            let source = source.clone();
            let depth = depth.clone();
            move |i: usize| -> Option<Prepared> {
                let chain = source(i).and_then(|faces| prefilter_probe(&faces, res))?;
                Some((chain, depth.as_ref().and_then(|d| d(i))))
            }
        };

        // THE WHOLE LEVEL, NOW, WHEN IT FITS. Same cost and same result as
        // the old all-at-once upload, and no frame ever lacks a probe.
        let everything_fits = count <= layers as usize;
        if everything_fits {
            for i in 0..count {
                let Some((chain, dist)) = prepare(i) else { continue };
                if let Some(layer) = pool.claim(i) {
                    write_probe_layer(queue, &texture, layer, res, &chain);
                    // Zeros where none was baked, so the layer can never hold
                    // another probe's distances.
                    let dist = dist.unwrap_or_else(|| vec![0; (res * res * 6 * 4) as usize]);
                    write_probe_depth_layer(queue, &depth_texture, layer, res, &dist);
                }
            }
        }
        log::info!(
            "reflection probes: {count} in the level, pool of {layers} ({} MB){}",
            probe_bytes(res) * layers as u64 / (1024 * 1024),
            if everything_fits { ", all resident" } else { ", streaming" },
        );

        // THE WORKER, for everything that was not preloaded. It holds only the
        // source; each probe it finishes is handed back and uploaded on the
        // render thread, which owns the queue.
        let (req_tx, req_rx) = mpsc::channel::<usize>();
        let (res_tx, res_rx) = mpsc::channel();
        if !everything_fits {
            std::thread::Builder::new()
                .name("probe_stream".into())
                .spawn(move || {
                    while let Ok(i) = req_rx.recv() {
                        if res_tx.send((i, prepare(i))).is_err() {
                            break;
                        }
                    }
                })
                .ok();
        }
        Self {
            pool,
            texture,
            depth_texture,
            resolution: res,
            count,
            requests: (!everything_fits).then_some(req_tx),
            results: res_rx,
            ready: HashMap::new(),
            requested: HashSet::new(),
            failed: HashSet::new(),
            sky_layer,
            building_layer,
        }
    }

    /// The cube layer holding the reflections' sky, if one was given.
    pub fn sky_layer(&self) -> Option<u32> {
        self.sky_layer
    }

    /// The first building's outside cube, the rest following in order, if any
    /// were given.
    pub fn building_layer(&self) -> Option<u32> {
        self.building_layer
    }

    /// The cube array view to bind.
    pub fn view(&self) -> wgpu::TextureView {
        self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("probe_cube_array_view"),
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        })
    }

    pub fn begin_frame(&mut self) {
        self.pool.begin_frame();
        while let Ok((i, chain)) = self.results.try_recv() {
            match chain {
                Some(chain) => {
                    self.ready.insert(i, chain);
                }
                None => {
                    // Undecodable: say so once, and stop asking.
                    log::warn!("reflection probe {i} could not be decoded; it is skipped");
                    self.failed.insert(i);
                }
            }
            self.requested.remove(&i);
        }
    }

    /// Turn an upload whose slots name PROBES (as residency fills them) into
    /// one whose slots name cube LAYERS, uploading what it can this frame.
    ///
    /// A slot whose probe is not in the pool yet is DROPPED from this frame
    /// rather than pointed at a stale layer, and its probe is requested. The
    /// room's other cells, or the room it sits in, answer meanwhile -- a
    /// slightly less exact photograph for a few frames, never a wrong one.
    pub fn resolve(&mut self, queue: &wgpu::Queue, upload: &mut ProbeUpload) {
        let mut uploads = 0;
        // EVERYTHING BUT THE SLOTS carries over: the doorways, the reflection
        // proxies, the trace switch. Listing the fields to keep dropped each
        // one added after the list was written -- `no_trace` once, then every
        // proxy, which made the pillar vanish from reflections on the headset
        // while the offline harness, which does not stream, drew it
        // (2026-09-27). Only the slot mapping is rebuilt here.
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
                        // Zeros where none was baked: a reused layer must not
                        // keep the evicted probe's distances.
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
                            let _ = tx.send(probe);
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
    }

    /// The distance cube array to bind beside [`Self::view`].
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

    fn desc(volume: u32, has_depth: bool) -> ProbeDesc {
        ProbeDesc { centre: Vec3::ZERO, min: Vec3::ZERO, max: Vec3::ONE, volume, has_depth,
            room_light: None
        }
    }

    #[test]
    fn the_outdoors_is_the_volume_with_no_distances() {
        let level = [desc(0, true), desc(0, true), desc(1, true), desc(2, false)];
        assert_eq!(outdoor_volume(&level), Some(2));
        // A room with one cell lacking distances is still a room.
        assert_eq!(outdoor_volume(&[desc(0, true), desc(0, false), desc(1, false)]), Some(1));
        // An older bake with no distances anywhere names no outdoors.
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

    /// The least recently used layer goes -- and only one this frame has not
    /// drawn with.
    #[test]
    fn eviction_takes_the_stalest_layer_the_frame_is_not_using() {
        let mut pool = LayerPool::new(2);
        pool.claim(1);
        pool.claim(2);
        pool.begin_frame();
        pool.touch(1);
        pool.begin_frame();
        pool.touch(2);
        // 1 was last used a frame ago, 2 this frame: 1 goes.
        assert_eq!(pool.claim(3), Some(0));
        assert!(!pool.is_resident(1));
        assert!(pool.is_resident(2) && pool.is_resident(3));
    }

    /// A frame that wants more probes than the pool holds does not evict one it
    /// is drawing with -- that would put another room's photograph in a slot.
    #[test]
    fn a_layer_in_use_this_frame_is_never_evicted() {
        let mut pool = LayerPool::new(2);
        pool.begin_frame();
        pool.claim(1);
        pool.claim(2);
        assert_eq!(pool.claim(3), None);
        assert!(pool.is_resident(1) && pool.is_resident(2));
    }

    /// Claiming a probe already resident is a touch, not a second layer.
    #[test]
    fn a_resident_probe_keeps_its_layer() {
        let mut pool = LayerPool::new(4);
        let a = pool.claim(7);
        assert_eq!(pool.claim(7), a);
        assert_eq!(pool.layer_of.len(), 1);
    }

    /// A small level gets exactly its own probe count; a huge one is capped by
    /// the memory budget and by the device, never by a fixed number.
    #[test]
    fn the_pool_is_sized_by_memory_and_the_device_not_a_constant() {
        assert_eq!(pool_size(9, 128, 341), 9);
        let budget = (PROBE_POOL_BUDGET_BYTES / probe_bytes(128)) as u32;
        assert_eq!(pool_size(5000, 128, 341), budget);
        assert_eq!(pool_size(5000, 128, 42), 42, "the device limit was ignored");
        // At a small face size the budget allows more than the device.
        assert_eq!(pool_size(5000, 16, 341), 341);
        assert!(budget >= (2 * MAX_PROBES) as u32, "the budget cannot hold one frame's slots");
    }

    /// EVERY FIELD BUT THE SLOTS survives `resolve`. It used to list the
    /// fields to keep, and silently dropped each one added after the list was
    /// written: the reflection proxies vanished on the headset -- the pillar
    /// missing from every reflection -- while the offline harness, which does
    /// not stream, drew them (2026-09-27).
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

    /// END TO END ON A REAL DEVICE: a level with more probes than the pool
    /// holds does not lose the ones past it. A probe residency asks for that
    /// is not in the pool is left out of that frame, prefiltered on the
    /// worker, and in a layer within a few frames.
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
