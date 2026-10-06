//! RUNTIME-DRIVEN DYNAMIC RESOLUTION, from `XR_META_recommended_layer_resolution`:
//! each frame the runtime recommends a size for the projection layer it is
//! about to receive -- smaller under GPU load or heat, larger with headroom,
//! never larger than the layer's swapchain.
//!
//! On Quest 3 an app that enables it is granted GPU level 5 (Meta, "CPU and
//! GPU levels", 2026-09-02). Measured (deploy98, 2026-10-06), a level is the
//! clock's FLOOR -- kgsl's devfreq `min_freq`, 599 MHz at level 5 -- and the
//! governor runs up to the runtime's 640 MHz cap at any level under load, so
//! the GPU-bound 72 Hz frame gains nothing. Asking each frame hands the level
//! to the runtime's controller: lowered to 4 where the frame has room (under
//! SpaceWarp), 5 under load. See cortex wall #16.
//!
//! openxr 0.18 / openxr-sys 0.10 predate the extension (registry number 255),
//! so its two structs and one function are declared here as `xr.xml` has them,
//! enabled through `ExtensionSet::other` and loaded by name. Asked for whether
//! or not the runtime lists it: runtime v209.91 does not list it to this app,
//! yet accepts it at instance creation, gives out the function and answers
//! every call (deploy98, 2026-10-06); `XrContext::new` makes the instance
//! without it where a runtime refuses.

use log::{info, warn};
use openxr as xr;
use std::ffi::c_void;

/// The extension's name, as the runtime lists it.
pub const EXTENSION_NAME: &str = "XR_META_recommended_layer_resolution";

// 1000000000 + (255 - 1) * 1000 + offset, the registry's rule for an
// extension's enum values.
const TYPE_RECOMMENDED_LAYER_RESOLUTION_META: i32 = 1_000_254_000;
const TYPE_RECOMMENDED_LAYER_RESOLUTION_GET_INFO_META: i32 = 1_000_254_001;

/// `XrRecommendedLayerResolutionMETA`.
#[repr(C)]
struct RecommendedLayerResolution {
    ty: xr::sys::StructureType,
    next: *mut c_void,
    recommended_image_dimensions: xr::sys::Extent2Di,
    is_valid: xr::sys::Bool32,
}

/// `XrRecommendedLayerResolutionGetInfoMETA`.
#[repr(C)]
struct RecommendedLayerResolutionGetInfo {
    ty: xr::sys::StructureType,
    next: *const c_void,
    layer: *const xr::sys::CompositionLayerBaseHeader,
    predicted_display_time: xr::sys::Time,
}

/// `xrGetRecommendedLayerResolutionMETA`.
type GetRecommendedLayerResolution = unsafe extern "system" fn(
    xr::sys::Session,
    *const RecommendedLayerResolutionGetInfo,
    *mut RecommendedLayerResolution,
) -> xr::sys::Result;

/// Whether the runtime lists the extension, among the names openxr 0.18 does
/// not know.
pub fn available(exts: &xr::ExtensionSet) -> bool {
    exts.other.iter().any(|name| name == EXTENSION_NAME)
}

/// The loaded function, for one session.
pub struct RecommendedResolution {
    fp: GetRecommendedLayerResolution,
    session: xr::sys::Session,
}

impl RecommendedResolution {
    /// `None` when the instance was created without the extension, or the
    /// runtime does not give out the function.
    pub fn new(instance: &xr::Instance, session: &xr::Session<xr::Vulkan>, enabled: bool) -> Option<Self> {
        if !enabled {
            return None;
        }
        let mut fp: Option<xr::sys::pfn::VoidFunction> = None;
        let result = unsafe {
            (instance.entry().fp().get_instance_proc_addr)(
                instance.as_raw(),
                c"xrGetRecommendedLayerResolutionMETA".as_ptr(),
                &mut fp,
            )
        };
        match fp {
            Some(f) if result == xr::sys::Result::SUCCESS => {
                info!("xr: {EXTENSION_NAME} loaded -- dynamic resolution available");
                Some(Self {
                    fp: unsafe { std::mem::transmute::<xr::sys::pfn::VoidFunction, GetRecommendedLayerResolution>(f) },
                    session: session.as_raw(),
                })
            }
            _ => {
                warn!("xr: {EXTENSION_NAME} enabled but xrGetRecommendedLayerResolutionMETA did not load ({result:?})");
                None
            }
        }
    }

    /// The runtime's recommended size for each view of `layer`, the layer as
    /// it will be submitted for `display_time` (no older than the latest
    /// `xrWaitFrame`'s): `Ok(None)` when it has no recommendation.
    ///
    /// # Safety
    /// `layer` must point to a valid composition layer whose chain and
    /// swapchains are alive for the call.
    pub unsafe fn recommend(
        &self,
        layer: *const xr::sys::CompositionLayerBaseHeader,
        display_time: xr::Time,
    ) -> Result<Option<(u32, u32)>, xr::sys::Result> {
        let info = RecommendedLayerResolutionGetInfo {
            ty: xr::sys::StructureType::from_raw(TYPE_RECOMMENDED_LAYER_RESOLUTION_GET_INFO_META),
            next: std::ptr::null(),
            layer,
            predicted_display_time: display_time,
        };
        let mut out = RecommendedLayerResolution {
            ty: xr::sys::StructureType::from_raw(TYPE_RECOMMENDED_LAYER_RESOLUTION_META),
            next: std::ptr::null_mut(),
            recommended_image_dimensions: xr::sys::Extent2Di { width: 0, height: 0 },
            is_valid: xr::sys::FALSE,
        };
        let result = unsafe { (self.fp)(self.session, &info, &mut out) };
        if result.into_raw() < 0 {
            return Err(result);
        }
        let xr::sys::Extent2Di { width, height } = out.recommended_image_dimensions;
        Ok((out.is_valid != xr::sys::FALSE && width > 0 && height > 0).then_some((width as u32, height as u32)))
    }
}

// The declared layouts are the C ones in `xr.xml` on a 64-bit target: a
// mismatch would not fail to compile, it would be the runtime reading the wrong
// bytes. Checked here at compile time, since this module builds for Android
// only and its tests would never run.
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(offset_of!(RecommendedLayerResolution, recommended_image_dimensions) == 16);
    assert!(offset_of!(RecommendedLayerResolution, is_valid) == 24);
    assert!(size_of::<RecommendedLayerResolution>() == 32);
    assert!(offset_of!(RecommendedLayerResolutionGetInfo, layer) == 16);
    assert!(offset_of!(RecommendedLayerResolutionGetInfo, predicted_display_time) == 24);
    assert!(size_of::<RecommendedLayerResolutionGetInfo>() == 32);
};
