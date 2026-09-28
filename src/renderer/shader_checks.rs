//! THE RUNTIME CHECKS wgpu COMPILES INTO THE AUDITED HOT SHADERS.
//!
//! By default wgpu compiles every WGSL module defensively, as if it came from
//! an untrusted web page:
//!
//! - **bounds checks**: every dynamic array, matrix or vector index is clamped
//!   (`min(i, len - 1)`), and every `textureLoad` coordinate too;
//! - **loop bounding**: every loop carries a 64-bit iteration counter that is
//!   tested and decremented each pass, so the driver can never assume a loop
//!   terminates when it might not;
//! - **integer division guards**: `/` and `%` on integers go through a helper
//!   that replaces a zero divisor.
//!
//! Measured on naga's output for the brush shader (2026-09-28): 176 index
//! clamps and 24 loop counters, about 6-8% more code. Most of it sits inside
//! the light loop and the probe trace's loops, which run many times a pixel on
//! a frame that is fill-bound. Each counter also holds two registers for the
//! whole loop, and register pressure is occupancy: the probe pass ran at 36%.
//!
//! WHAT EACH CHECK IS FOR, IN OUR SHADERS:
//!
//! - Bounds checks turn an out-of-range index into a clamp instead of a read
//!   of whatever lies past the array. None of ours should ever go out of range:
//!   every uniform array is walked up to a count the CPU clamps to its length
//!   (`pack_lights`, `select_resident_probes`, `ProbeUpload::set_portals` /
//!   `set_proxies`), the room tables are bounds-tested before indexing
//!   (`probe_room_slot`), a doorway's axis is 0-2 from the baker, the terrain's
//!   layers are the constants 0-3, and the one `textureLoad` with computed
//!   coordinates (the probe-pass upsample) clamps them itself. But they are
//!   what stands between a FUTURE bug in that bookkeeping and a GPU fault, so
//!   they are the check with a purpose here, and they are KEPT.
//! - Loop bounding exists so a driver cannot miscompile a loop it cannot prove
//!   ends. Every loop here ends by construction: counts from the CPU,
//!   constants, and room chains whose links strictly ascend. And the counter
//!   is 64 bits, so a loop that did run away would hang the GPU either way; it
//!   protects nothing a game of ours can hit.
//! - Integer division: only by the constants 2, 3 and 4 here, which a
//!   compiler folds; kept, it costs nothing.
//!
//! So the audited shaders drop LOOP BOUNDING and keep the rest. The other
//! settings are here to measure against (`AuditedChecks`), and the shaders
//! compiled through `audited_shader_module` are listed where it is called:
//! the brush variants (scene, probe pass, per-pixel trace), the brush depth
//! prepass and the terrain. SSR, the crack seal and everything else keep
//! wgpu's defaults: they have not had this audit.
//!
//! `quest_app`'s offline renders of the benchmark views are the regression
//! check: a change here must alter no pixel.

/// Which of wgpu's runtime checks the audited shaders are compiled with.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AuditedChecks {
    /// wgpu's defaults: every check.
    All,
    /// Every check but the loop counters. See the module comment.
    NoLoopBounding,
    /// No runtime checks at all. For measurement only.
    None,
}

/// What the audited shaders ship with. See the module comment.
pub const AUDITED_CHECKS: AuditedChecks = AuditedChecks::NoLoopBounding;

/// The runtime checks `checks` stands for.
pub fn runtime_checks(checks: AuditedChecks) -> wgpu::ShaderRuntimeChecks {
    match checks {
        AuditedChecks::All => wgpu::ShaderRuntimeChecks::checked(),
        AuditedChecks::NoLoopBounding => {
            let mut c = wgpu::ShaderRuntimeChecks::checked();
            c.force_loop_bounding = false;
            c
        }
        // SAFETY (for the callers of `audited_shader_module`): see the module
        // comment -- indices in range, loops that end, divisors that are not
        // zero.
        AuditedChecks::None => unsafe { wgpu::ShaderRuntimeChecks::all(false) },
    }
}

/// `device.create_shader_module(desc)` with [`AUDITED_CHECKS`]. Only for the
/// shaders the module comment lists as audited.
pub fn audited_shader_module(device: &wgpu::Device, desc: wgpu::ShaderModuleDescriptor<'_>) -> wgpu::ShaderModule {
    match AUDITED_CHECKS {
        AuditedChecks::All => device.create_shader_module(desc),
        // SAFETY: the shaders passed here were audited for exactly the checks
        // `runtime_checks` drops -- see the module comment.
        checks => unsafe { device.create_shader_module_trusted(desc, runtime_checks(checks)) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What ships keeps the bounds checks, which have a purpose here, and
    /// drops only the loop counters, which do not.
    #[test]
    fn the_shipped_setting_keeps_bounds_checks_and_drops_loop_counters() {
        let c = runtime_checks(AUDITED_CHECKS);
        assert!(c.bounds_checks, "bounds checks guard the room tables and light lists against a future bug");
        assert!(!c.force_loop_bounding);
        assert!(c.int_div_checks);
    }
}

/// Whether the device reports each pipeline's shader statistics (the
/// `PIPESTATS` log lines): set by `xr::vulkan::VkContext::new` when
/// `debug.spacesoup.pipestats` is `1` and the driver has
/// `VK_KHR_pipeline_executable_properties`. Measurement builds of extra
/// pipelines -- `BrushPipeline::log_probe_pass_register_cuts` -- run only then.
pub static PIPELINE_STATISTICS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

