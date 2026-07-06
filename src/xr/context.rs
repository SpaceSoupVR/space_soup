use log::info;
use openxr as xr;

pub struct XrContext {
    pub instance: xr::Instance,
    pub system: xr::SystemId,
    pub has_hand_tracking: bool,
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

        Ok(Self {
            instance,
            system,
            has_hand_tracking,
        })
    }
}
