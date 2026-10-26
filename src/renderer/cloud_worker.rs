//! THE CLOUD POPULATION OFF THE RENDER THREAD (2026-10-21). The headset
//! logged the time of day's tick at 5-55 ms about once a second and at
//! 120-160 ms whenever a cloud epoch began (TODTIME): the population's
//! update, the splats packed for the GPU and the twin's tiles rebuilt for
//! the CPU's rays toward the sun, all on the render thread -- a dropped
//! frame every second while the clock ran (measured on the Mac: 120 frames
//! over 2 ms in 120 s, the worst 20 ms, `time_the_clouds_on_the_render_thread`).
//!
//! The population is a pure function of the parameters, the weather field
//! and the second, so a worker can draw the next second's before it comes:
//! it keeps the births (`CloudPopulation::update`), hands back what the sky
//! reads (`CloudPopulation::frame`) with the twin's tiles already warmed by
//! the rays the render thread will cast, and the render thread only swaps it
//! in when its second arrives. A worker running late leaves the last frame
//! up a second longer: the clouds still drift on it (`place`), and nothing
//! is born or dies a second late that anyone could see.
//!
//! The headset turns it on (`enable`); tests and the offline harness keep the
//! synchronous update, so a still frame is the same every time.

use space_soup_sky::cloud_layers::{transmittance_toward, CloudModel, CloudNoise, CloudWeather};
use space_soup_sky::cloud_population::{CloudPopulation, UPDATE_S};
use space_soup_sky::clouds::{CloudField, CloudParams};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Draw the clouds' populations on a worker from now on (the headset app).
pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// One second's population asked of the worker, and the rays (from, toward)
/// whose tiles to warm: the ones the render thread casts at the level's
/// middle, round the eye and over the shadow's grid.
pub struct Request {
    pub params: CloudParams,
    pub field: Arc<CloudField>,
    pub model: Arc<CloudModel>,
    pub seconds: f64,
    pub warm: Vec<([f32; 2], f32, [f32; 3])>,
}

pub struct CloudWorker {
    tx: mpsc::Sender<Request>,
    rx: mpsc::Receiver<CloudPopulation>,
    asked: Option<f64>,
    /// Handed back, not yet due.
    ready: Vec<CloudPopulation>,
}

impl CloudWorker {
    pub fn new() -> Self {
        let (tx, req_rx) = mpsc::channel::<Request>();
        let (res_tx, rx) = mpsc::channel::<CloudPopulation>();
        std::thread::Builder::new()
            .name("cloud_population".into())
            .spawn(move || {
                let mut master = CloudPopulation::new();
                let noise = CloudNoise::shared();
                while let Ok(mut r) = req_rx.recv() {
                    // Only the newest second matters.
                    while let Ok(next) = req_rx.try_recv() {
                        r = next;
                    }
                    master.update(&r.params, &r.field, r.seconds);
                    let frame = master.frame();
                    let mut layers = r.model.layers(r.seconds);
                    frame.place(&mut layers, r.seconds);
                    let w = CloudWeather::new(&r.field, Some(&frame));
                    for (xz, alt, dir) in &r.warm {
                        transmittance_toward(&layers, &noise, w, *xz, *alt, *dir);
                    }
                    if res_tx.send(frame).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn the cloud population worker");
        Self { tx, rx, asked: None, ready: Vec::new() }
    }

    /// The population to show at `now`, when a newer one's second has come;
    /// and the next second asked for (`request` builds it) if it is not yet.
    pub fn poll(&mut self, now: f64, request: impl FnOnce(f64) -> Request) -> Option<CloudPopulation> {
        while let Ok(f) = self.rx.try_recv() {
            self.ready.push(f);
        }
        let next = ((now / UPDATE_S).floor() + 1.0) * UPDATE_S;
        if self.asked != Some(next) {
            self.asked = Some(next);
            let _ = self.tx.send(request(next));
        }
        // The newest whose second has come; later ones wait for theirs, and a
        // clock gone back past them drops them.
        let due = self.ready.iter().rposition(|f| f.seconds() <= now)?;
        let f = self.ready.remove(due);
        self.ready.retain(|g| g.seconds() > now && g.seconds() <= now + 4.0 * UPDATE_S);
        Some(f)
    }
}

impl Default for CloudWorker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The worker's population for a second is the one the render thread
    /// would have drawn itself, and it is handed over only once that second
    /// has come.
    #[test]
    fn the_worker_draws_what_the_render_thread_would() {
        let field = Arc::new(CloudField::build(7));
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.5, ..Default::default() };
        let model = Arc::new(CloudModel::new(&params, &noise));
        let mut worker = CloudWorker::new();
        let ask = |s: f64| Request { params, field: field.clone(), model: model.clone(), seconds: s, warm: Vec::new() };
        let t = 20_000.5;
        assert!(worker.poll(t, ask).is_none(), "nothing is ready before it is asked for");
        let mut got = None;
        for _ in 0..400 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            if let Some(f) = worker.poll(t + 0.6, ask) {
                got = Some(f);
                break;
            }
        }
        let got = got.expect("the worker handed back the next second");
        assert_eq!(got.seconds(), 20_001.0);
        let mut own = CloudPopulation::new();
        own.update(&params, &field, 20_001.0);
        for k in 0..space_soup_sky::cloud_population::POPULATED {
            assert_eq!(got.splats(k).0, own.splats(k).0, "layer {k}");
        }
    }
}
