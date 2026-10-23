#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AuditedChecks {

    All,

    NoLoopBounding,

    None,
}

pub const AUDITED_CHECKS: AuditedChecks = AuditedChecks::NoLoopBounding;

pub fn runtime_checks(checks: AuditedChecks) -> wgpu::ShaderRuntimeChecks {
    match checks {
        AuditedChecks::All => wgpu::ShaderRuntimeChecks::checked(),
        AuditedChecks::NoLoopBounding => {
            let mut c = wgpu::ShaderRuntimeChecks::checked();
            c.force_loop_bounding = false;
            c
        }

        AuditedChecks::None => unsafe { wgpu::ShaderRuntimeChecks::all(false) },
    }
}

/// THE AUDITED SHADERS' CHECKS FOR THIS RUN, over `AUDITED_CHECKS`: the
/// Quest's `debug.spacesoup.shaderchecks` property (`all` / `none`), read
/// when the device is made, to A/B naga's injected bounds checks in the hot
/// loops (the AAA doc's M4 / rank 9). `none` trusts every index in the
/// brush, terrain and fix-up shaders, and robust access is off on the
/// headset, so an out-of-range read is undefined: an A/B only.
pub static CHECKS_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// `all` or `none` from a property's text; anything else keeps the audit.
pub fn set_checks_override(text: &str) -> Option<AuditedChecks> {
    let (code, checks) = match text.trim() {
        "all" => (1, Some(AuditedChecks::All)),
        "none" => (2, Some(AuditedChecks::None)),
        _ => (0, None),
    };
    CHECKS_OVERRIDE.store(code, std::sync::atomic::Ordering::Relaxed);
    checks
}

pub fn audited_checks() -> AuditedChecks {
    match CHECKS_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => AuditedChecks::All,
        2 => AuditedChecks::None,
        _ => AUDITED_CHECKS,
    }
}

pub fn audited_shader_module(device: &wgpu::Device, desc: wgpu::ShaderModuleDescriptor<'_>) -> wgpu::ShaderModule {
    match audited_checks() {
        AuditedChecks::All => device.create_shader_module(desc),

        checks => unsafe { device.create_shader_module_trusted(desc, runtime_checks(checks)) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_setting_keeps_bounds_checks_and_drops_loop_counters() {
        let c = runtime_checks(AUDITED_CHECKS);
        assert!(c.bounds_checks, "bounds checks guard the room tables and light lists against a future bug");
        assert!(!c.force_loop_bounding);
        assert!(c.int_div_checks);
    }
}

pub static PIPELINE_STATISTICS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
