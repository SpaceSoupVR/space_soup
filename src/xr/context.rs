use log::info;
use openxr as xr;

pub struct XrContext {
    pub instance: xr::Instance,
    pub system: xr::SystemId,
    pub has_hand_tracking: bool,

    pub has_layer_settings: bool,

    pub has_performance_metrics: bool,

    pub space_warp: Option<(u32, u32)>,

    pub has_recommended_resolution: bool,

    pub has_visibility_mask: bool,
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

        let has_performance_metrics = available_exts.meta_performance_metrics;
        if has_performance_metrics {
            exts.meta_performance_metrics = true;
        }

        if available_exts.ext_performance_settings {
            exts.ext_performance_settings = true;
        }

        let has_space_warp = available_exts.fb_space_warp;
        if has_space_warp {
            exts.fb_space_warp = true;
        }

        let has_visibility_mask = available_exts.khr_visibility_mask;
        if has_visibility_mask {
            exts.khr_visibility_mask = true;
        }

        info!("xr: runtime extensions unknown to openxr 0.18: {}", available_exts.other.join(", "));

        let app_info = xr::ApplicationInfo {
            application_name: "space_soup",
            application_version: 1,
            engine_name: "space_soup",
            engine_version: 1,
        };

        let listed = super::recommended_resolution::available(&available_exts);
        let mut with_it = exts.clone();
        with_it.other.push(super::recommended_resolution::EXTENSION_NAME.to_string());
        let (instance, has_recommended_resolution) = match entry.create_instance(&app_info, &with_it, &[]) {
            Ok(instance) => (instance, true),
            Err(e) => {
                info!("{}: refused at instance creation ({e})", super::recommended_resolution::EXTENSION_NAME);
                (entry.create_instance(&app_info, &exts, &[])?, false)
            }
        };
        info!(
            "{}: {} by the runtime, {}",
            super::recommended_resolution::EXTENSION_NAME,
            if listed { "listed" } else { "NOT listed" },
            if has_recommended_resolution { "enabled" } else { "not enabled" },
        );

        let props = instance.properties()?;
        info!("Runtime: {} v{}", props.runtime_name, props.runtime_version);

        let system = instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
        let _reqs = instance.graphics_requirements::<xr::Vulkan>(system)?;

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
            has_recommended_resolution,
            has_visibility_mask,
        })
    }
}
