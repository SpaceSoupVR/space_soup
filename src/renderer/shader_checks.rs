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

pub fn audited_shader_module(device: &wgpu::Device, desc: wgpu::ShaderModuleDescriptor<'_>) -> wgpu::ShaderModule {
    match AUDITED_CHECKS {
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
