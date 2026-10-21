use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct WindowRecord {

    pub window: u64,

    pub t: f64,

    pub phase: String,

    pub cycle_pass: u64,
    pub cycle_len: u64,

    pub warmup: bool,

    pub levers: String,

    pub bench: Option<String>,
    pub ssr: bool,
    pub multiview: bool,

    pub frames: u64,
    pub cpu_avg: f64,
    pub cpu_max: f64,

    pub gpu_avg: f64,
    pub gpu_max: f64,

    pub frame_ms: f64,
    pub fps: f64,

    pub pass: BTreeMap<String, f32>,

    pub xr: BTreeMap<String, f64>,

    #[serde(default)]
    pub cpu: BTreeMap<String, f64>,

    /// The time of day's clock at the window's end (hours), when it runs: a
    /// sweep through sunset is read window by window against it.
    #[serde(default)]
    pub hour: Option<f32>,
}

impl WindowRecord {
    pub fn to_line(&self) -> String {

        serde_json::to_string(self).unwrap_or_default()
    }
}

pub struct PerfLog {
    tx: Option<std::sync::mpsc::Sender<String>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl PerfLog {
    pub fn new(path: std::path::PathBuf) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let thread = std::thread::Builder::new()
            .name("perf_log".into())
            .spawn(move || {
                let mut warned = false;
                for line in rx {
                    use std::io::Write;
                    let wrote = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&path)
                        .and_then(|mut f| writeln!(f, "{line}"));

                    if let Err(e) = wrote {
                        if !warned {
                            log::warn!("perf log: {} cannot be written: {e}", path.display());
                            warned = true;
                        }
                    }
                }
            })
            .ok();
        Self { tx: Some(tx), thread }
    }

    pub fn write(&self, line: String) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(line);
        }
    }
}

impl Drop for PerfLog {

    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(window: u64) -> WindowRecord {
        WindowRecord {
            window,
            t: 12.5,
            phase: "no_portals".into(),
            cycle_pass: 1,
            cycle_len: 14,
            warmup: false,
            levers: "ab_cycle,bench=hall_back".into(),
            bench: Some("hall_back".into()),
            ssr: false,
            multiview: false,
            frames: 112,
            cpu_avg: 4.1,
            cpu_max: 6.0,
            gpu_avg: 33.4,
            gpu_max: 35.9,
            frame_ms: 41.7,
            fps: 24.0,
            pass: [("scene0".to_string(), 12.0f32)].into_iter().collect(),
            xr: [("app/gpu_frametime".to_string(), 31.2)].into_iter().collect(),
            cpu: [("pose_ms".to_string(), 1.5)].into_iter().collect(),
            hour: Some(18.5),
        }
    }

    #[test]
    fn a_record_is_one_json_line_with_the_fields_the_host_reads() {
        let line = record(7).to_line();
        assert!(!line.contains('\n'));
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        for key in ["window", "phase", "cycle_pass", "cycle_len", "warmup", "bench", "levers", "gpu_avg", "frame_ms", "pass", "xr"] {
            assert!(v.get(key).is_some(), "{key} missing from {line}");
        }
        assert_eq!(v["xr"]["app/gpu_frametime"], 31.2);
        assert_eq!(v["bench"], "hall_back");
    }

    #[test]
    fn every_line_handed_over_is_on_disk_when_the_log_goes_away() {
        let dir = std::env::temp_dir().join(format!("perf_log_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("perf.jsonl");
        let _ = std::fs::remove_file(&path);
        {
            let log = PerfLog::new(path.clone());
            for w in 0..5 {
                log.write(record(w).to_line());
            }
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let windows: Vec<u64> = text
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["window"].as_u64().unwrap())
            .collect();
        assert_eq!(windows, vec![0, 1, 2, 3, 4]);

        std::fs::remove_file(&path).unwrap();
        {
            let log = PerfLog::new(path.clone());
            log.write(record(9).to_line());
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
