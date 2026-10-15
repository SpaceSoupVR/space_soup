use log::{info, warn};
use openxr as xr;
use std::ffi::c_void;

pub const EXTENSION_NAME: &str = "XR_META_recommended_layer_resolution";

const TYPE_RECOMMENDED_LAYER_RESOLUTION_META: i32 = 1_000_254_000;
const TYPE_RECOMMENDED_LAYER_RESOLUTION_GET_INFO_META: i32 = 1_000_254_001;

#[repr(C)]
struct RecommendedLayerResolution {
    ty: xr::sys::StructureType,
    next: *mut c_void,
    recommended_image_dimensions: xr::sys::Extent2Di,
    is_valid: xr::sys::Bool32,
}

#[repr(C)]
struct RecommendedLayerResolutionGetInfo {
    ty: xr::sys::StructureType,
    next: *const c_void,
    layer: *const xr::sys::CompositionLayerBaseHeader,
    predicted_display_time: xr::sys::Time,
}

type GetRecommendedLayerResolution = unsafe extern "system" fn(
    xr::sys::Session,
    *const RecommendedLayerResolutionGetInfo,
    *mut RecommendedLayerResolution,
) -> xr::sys::Result;

pub fn available(exts: &xr::ExtensionSet) -> bool {
    exts.other.iter().any(|name| name == EXTENSION_NAME)
}

pub struct RecommendedResolution {
    fp: GetRecommendedLayerResolution,
    session: xr::sys::Session,
}

impl RecommendedResolution {

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

const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(offset_of!(RecommendedLayerResolution, recommended_image_dimensions) == 16);
    assert!(offset_of!(RecommendedLayerResolution, is_valid) == 24);
    assert!(size_of::<RecommendedLayerResolution>() == 32);
    assert!(offset_of!(RecommendedLayerResolutionGetInfo, layer) == 16);
    assert!(offset_of!(RecommendedLayerResolutionGetInfo, predicted_display_time) == 24);
    assert!(size_of::<RecommendedLayerResolutionGetInfo>() == 32);
};
