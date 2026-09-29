use log::info;
use openxr as xr;

pub struct XrContext {
    pub instance: xr::Instance,
    pub system: xr::SystemId,
    pub has_hand_tracking: bool,
    /// Whether `XR_FB_composition_layer_settings` is available AND was asked
    /// for. Both halves, because chaining the struct onto a projection layer
    /// when the runtime does not know the extension is not a no-op -- the
    /// runtime is entitled to reject the whole layer over an unrecognised
    /// `next`, which loses the frame rather than the sharpening.
    pub has_layer_settings: bool,
    /// Whether `XR_META_performance_metrics` was available and enabled. See
    /// `perf_metrics`.
    pub has_performance_metrics: bool,
    /// `XR_FB_space_warp`, available and enabled: the motion-vector image
    /// size the runtime recommends. See `renderer::space_warp`.
    pub space_warp: Option<(u32, u32)>,
}

impl XrContext {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let entry = unsafe { xr::Entry::load()? };

        #[cfg(target_os = "android")]
        {
            match entry.initialize_android_loader() {
                Ok(()) => info!("xr: android loader initialized"),
                Err(e) if e.to_string().contains("initialization of object") => {
                    info!("xr: android loader already initialized (hot-restart) — continuing");
                }
                Err(e) => return Err(Box::new(e)),
            }
        }

        let available_exts = entry.enumerate_extensions()?;

        let mut exts = xr::ExtensionSet::default();
        exts.khr_vulkan_enable2 = true;
        #[cfg(target_os = "android")]
        {
            exts.khr_android_create_instance = true;
        }

        let has_hand_tracking = available_exts.ext_hand_tracking;
        if has_hand_tracking {
            exts.ext_hand_tracking = true;
            info!("Hand tracking extension available");
        }

        // MQSR. Only asked for when the policy actually wants it, so the
        // extension is not enabled on a build that would chain nothing --
        // see `renderer::layer_settings`.
        let wants_sharpening =
            crate::renderer::layer_settings::XR_SHARPENING.wants_layer_settings();
        let has_layer_settings = available_exts.fb_composition_layer_settings && wants_sharpening;
        if has_layer_settings {
            exts.fb_composition_layer_settings = true;
            info!(
                "Composition layer settings available -- requesting {:?} sharpening",
                crate::renderer::layer_settings::XR_SHARPENING
            );
        } else if wants_sharpening {
            info!("Composition layer settings NOT available -- no compositor sharpening");
        }

        // The runtime's own frame counters, for the `XRPERF` log line. See
        // `perf_metrics`. Asked for only where the runtime lists it.
        let has_performance_metrics = available_exts.meta_performance_metrics;
        if has_performance_metrics {
            exts.meta_performance_metrics = true;
        }

        // APPLICATION SPACEWARP, where the runtime has it. Enabling it changes
        // nothing until a frame carries motion vectors; see
        // `renderer::space_warp`.
        let has_space_warp = available_exts.fb_space_warp;
        if has_space_warp {
            exts.fb_space_warp = true;
        }

        let instance = entry.create_instance(
            &xr::ApplicationInfo {
                application_name: "space_soup",
                application_version: 1,
                engine_name: "space_soup",
                engine_version: 1,
            },
            &exts,
            &[],
        )?;

        let props = instance.properties()?;
        info!("Runtime: {} v{}", props.runtime_name, props.runtime_version);

        let system = instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
        let _reqs = instance.graphics_requirements::<xr::Vulkan>(system)?;

        // The motion vectors' size, from the system's properties with the
        // space warp struct chained on (the safe wrapper has no `next`).
        let space_warp = has_space_warp.then(|| {
            let mut sw = xr::sys::SystemSpaceWarpPropertiesFB {
                ty: xr::sys::SystemSpaceWarpPropertiesFB::TYPE,
                next: std::ptr::null_mut(),
                recommended_motion_vector_image_rect_width: 0,
                recommended_motion_vector_image_rect_height: 0,
            };
            let mut props: xr::sys::SystemProperties = unsafe { std::mem::zeroed() };
            props.ty = xr::sys::SystemProperties::TYPE;
            props.next = &mut sw as *mut _ as *mut std::ffi::c_void;
            let result = unsafe { (instance.fp().get_system_properties)(instance.as_raw(), system, &mut props) };
            info!(
                "Space warp: {:?}, motion vectors {}x{}",
                result,
                sw.recommended_motion_vector_image_rect_width,
                sw.recommended_motion_vector_image_rect_height,
            );
            (sw.recommended_motion_vector_image_rect_width.max(1), sw.recommended_motion_vector_image_rect_height.max(1))
        });

        Ok(Self {
            instance,
            system,
            has_hand_tracking,
            has_layer_settings,
            has_performance_metrics,
            space_warp,
        })
    }
}

