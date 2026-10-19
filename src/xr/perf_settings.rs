use crate::renderer::performance_level::{Domain, PerformanceLevel};
use log::{info, warn};
use openxr as xr;

pub struct PerfSettings {
    fp: xr::raw::PerformanceSettingsEXT,
    session: xr::sys::Session,
}

impl PerfSettings {

    pub fn new(instance: &xr::Instance, session: &xr::Session<xr::Vulkan>) -> Option<Self> {
        let fp = instance.exts().ext_performance_settings?;
        Some(Self { fp, session: session.as_raw() })
    }

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
