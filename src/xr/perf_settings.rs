//! The performance levels asked of the runtime, from
//! `XR_EXT_performance_settings`. What to ask, and when, is decided in
//! `renderer::performance_level` (host-tested); this only makes the call.
//!
//! Gated on the runtime advertising the extension, as the other optional
//! extensions are. With it enabled, the runtime also sends
//! `XrEventDataPerfSettingsEXT` when compositing, rendering or heat cross a
//! warning level -- `quest_app` logs those.

use crate::renderer::performance_level::{Domain, PerformanceLevel};
use log::{info, warn};
use openxr as xr;

pub struct PerfSettings {
    fp: xr::raw::PerformanceSettingsEXT,
    session: xr::sys::Session,
}

impl PerfSettings {
    /// `None` when the instance was created without the extension.
    pub fn new(instance: &xr::Instance, session: &xr::Session<xr::Vulkan>) -> Option<Self> {
        let fp = instance.exts().ext_performance_settings?;
        Some(Self { fp, session: session.as_raw() })
    }

    /// Ask for `level` in `domain`, and log what the runtime said.
    pub fn request(&self, domain: Domain, level: PerformanceLevel) {
        let result = unsafe {
            (self.fp.perf_settings_set_performance_level)(
                self.session,
                xr::sys::PerfSettingsDomainEXT::from_raw(domain.raw()),
                xr::sys::PerfSettingsLevelEXT::from_raw(level.raw()),
            )
        };
        if result == xr::sys::Result::SUCCESS {
            info!("xr performance level: {} -> {}", domain.label(), level.label());
        } else {
            warn!("xr performance level: {} -> {} refused ({result:?})", domain.label(), level.label());
        }
    }
}
