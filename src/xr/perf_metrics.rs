use log::info;
use openxr as xr;
use std::ptr;

struct Counter {

    short: String,
    path: xr::sys::Path,
}

pub use crate::perf_metrics_log::{format_line, short_name, worth_logging, Sample, Unit};

pub struct PerfMetrics {
    fp: xr::raw::PerformanceMetricsMETA,
    session: xr::sys::Session,
    counters: Vec<Counter>,
}

impl PerfMetrics {

    pub fn new(instance: &xr::Instance, session: &xr::Session<xr::Vulkan>) -> Option<Self> {
        let fp = instance.exts().meta_performance_metrics?;
        let raw_instance = instance.as_raw();
        let raw_session = session.as_raw();

        let mut count = 0u32;
        let ok = unsafe {
            (fp.enumerate_performance_metrics_counter_paths)(raw_instance, 0, &mut count, ptr::null_mut())
        };
        if ok != xr::sys::Result::SUCCESS {
            info!("xr perf metrics: enumerating counters failed ({ok:?}); none logged");
            return None;
        }
        let mut paths = vec![xr::sys::Path::NULL; count as usize];
        let ok = unsafe {
            (fp.enumerate_performance_metrics_counter_paths)(raw_instance, count, &mut count, paths.as_mut_ptr())
        };
        if ok != xr::sys::Result::SUCCESS {
            info!("xr perf metrics: enumerating counters failed ({ok:?}); none logged");
            return None;
        }
        let counters: Vec<Counter> = paths
            .iter()
            .take(count as usize)
            .filter_map(|&path| {
                let name = instance.path_to_string(path).ok()?;
                Some(Counter { short: short_name(&name), path })
            })
            .filter(|c| worth_logging(&c.short))
            .collect();

        let state = xr::sys::PerformanceMetricsStateMETA {
            ty: xr::sys::PerformanceMetricsStateMETA::TYPE,
            next: ptr::null(),
            enabled: xr::sys::TRUE,
        };
        let ok = unsafe { (fp.set_performance_metrics_state)(raw_session, &state) };
        if ok != xr::sys::Result::SUCCESS {
            info!("xr perf metrics: enabling failed ({ok:?}); none logged");
            return None;
        }
        info!(
            "xr perf metrics: enabled, {} of {count} counters logged: {}",
            counters.len(),
            counters.iter().map(|c| c.short.as_str()).collect::<Vec<_>>().join(" "),
        );
        Some(Self { fp, session: raw_session, counters })
    }

    /// One counter's value by its short name (`app/gpu_frametime`).
    pub fn value(&self, short: &str) -> Option<f64> {
        self.counters.iter().find(|c| c.short == short).and_then(|c| self.read(c)).map(|s| s.value)
    }

    pub fn sample(&self) -> Vec<Sample> {
        self.counters.iter().filter_map(|c| self.read(c)).collect()
    }

    fn read(&self, c: &Counter) -> Option<Sample> {
        {
            {
                let mut counter = xr::sys::PerformanceMetricsCounterMETA {
                    ty: xr::sys::PerformanceMetricsCounterMETA::TYPE,
                    next: ptr::null(),
                    counter_flags: xr::sys::PerformanceMetricsCounterFlagsMETA::EMPTY,
                    counter_unit: xr::sys::PerformanceMetricsCounterUnitMETA::GENERIC,
                    uint_value: 0,
                    float_value: 0.0,
                };
                let ok = unsafe { (self.fp.query_performance_metrics_counter)(self.session, c.path, &mut counter) };
                if ok != xr::sys::Result::SUCCESS
                    || !counter.counter_flags.contains(xr::sys::PerformanceMetricsCounterFlagsMETA::ANY_VALUE_VALID)
                {
                    return None;
                }
                let value = if counter
                    .counter_flags
                    .contains(xr::sys::PerformanceMetricsCounterFlagsMETA::FLOAT_VALUE_VALID)
                {
                    counter.float_value as f64
                } else {
                    counter.uint_value as f64
                };
                let unit = match counter.counter_unit {
                    xr::sys::PerformanceMetricsCounterUnitMETA::MILLISECONDS => Unit::Milliseconds,
                    xr::sys::PerformanceMetricsCounterUnitMETA::PERCENTAGE => Unit::Percentage,
                    xr::sys::PerformanceMetricsCounterUnitMETA::HERTZ => Unit::Hertz,
                    xr::sys::PerformanceMetricsCounterUnitMETA::BYTES => Unit::Bytes,
                    _ => Unit::Generic,
                };
                Some(Sample { short: c.short.clone(), value, unit })
            }
        }
    }
}
