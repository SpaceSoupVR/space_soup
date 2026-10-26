//! HEAT, as the headset reports it (the colleague's plan part 2, item 3):
//! the Quest throttles its CPU and GPU when hot, and a frame drop with no
//! recorded cause cannot be fixed. Two sources, both kept per perf window:
//! the `OpenXR` runtime's `XR_EXT_performance_settings` events (per domain and
//! sub-domain, normal / warning / impaired), which Meta does not document
//! sending, and Android's thermal status (`AThermal_getCurrentThermalStatus`,
//! API 30), looked up at run time so a system without it loads the app the
//! same and reports nothing.

use std::collections::BTreeMap;

/// Android's thermal status names, by `AThermalStatus` value.
pub const STATUS_NAMES: [&str; 7] = ["none", "light", "moderate", "severe", "critical", "emergency", "shutdown"];

pub fn status_name(status: i32) -> &'static str {
    usize::try_from(status).ok().and_then(|s| STATUS_NAMES.get(s)).copied().unwrap_or("unknown")
}

/// The latest heat state by source: `xr/<domain>/<sub-domain>` from the
/// runtime's events, `android` from the thermal status.
#[derive(Clone, Debug, Default)]
pub struct Thermal {
    state: BTreeMap<String, String>,
    #[cfg(target_os = "android")]
    manager: Option<android::Manager>,
}

impl Thermal {
    /// A runtime event: `domain` and `sub_domain` as the runtime names them,
    /// the level it went to.
    pub fn note_xr(&mut self, domain: &str, sub_domain: &str, level: &str) {
        self.state.insert(format!("xr/{domain}/{sub_domain}"), level.to_string());
    }

    /// Android's status now (cheap; once a window is plenty).
    pub fn poll(&mut self) {
        #[cfg(target_os = "android")]
        {
            let manager = self.manager.get_or_insert_with(android::Manager::acquire);
            if let Some(status) = manager.status() {
                self.state.insert("android".into(), status_name(status).into());
            }
        }
    }

    pub fn state(&self) -> &BTreeMap<String, String> {
        &self.state
    }
}

#[cfg(target_os = "android")]
mod android {
    use std::ffi::{c_char, c_int, c_void};

    extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    const RTLD_NOW: c_int = 2;

    type Acquire = unsafe extern "C" fn() -> *mut c_void;
    type Status = unsafe extern "C" fn(*mut c_void) -> c_int;

    #[derive(Clone, Debug)]
    pub struct Manager {
        manager: usize,
        status: Option<Status>,
    }

    impl Manager {
        /// `AThermal_acquireManager` from libandroid.so, if this system has it.
        pub fn acquire() -> Self {
            unsafe {
                let lib = dlopen(c"libandroid.so".as_ptr(), RTLD_NOW);
                if lib.is_null() {
                    return Self { manager: 0, status: None };
                }
                let acquire = dlsym(lib, c"AThermal_acquireManager".as_ptr());
                let status = dlsym(lib, c"AThermal_getCurrentThermalStatus".as_ptr());
                if acquire.is_null() || status.is_null() {
                    log::info!("THERMAL: this system has no AThermal (API 30): Android heat not read");
                    return Self { manager: 0, status: None };
                }
                let acquire: Acquire = std::mem::transmute(acquire);
                let manager = acquire();
                Self { manager: manager as usize, status: (!manager.is_null()).then(|| std::mem::transmute::<*mut c_void, Status>(status)) }
            }
        }

        pub fn status(&self) -> Option<i32> {
            self.status.map(|f| unsafe { f(self.manager as *mut c_void) })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_are_named_and_runtime_events_kept_per_domain() {
        assert_eq!(status_name(0), "none");
        assert_eq!(status_name(3), "severe");
        assert_eq!(status_name(9), "unknown");
        let mut t = Thermal::default();
        t.note_xr("gpu", "thermal", "warning");
        t.note_xr("gpu", "thermal", "impaired");
        t.note_xr("cpu", "rendering", "normal");
        assert_eq!(t.state().get("xr/gpu/thermal").map(String::as_str), Some("impaired"));
        assert_eq!(t.state().len(), 2);
    }
}
