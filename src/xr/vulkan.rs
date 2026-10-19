use ash::vk::{self, Handle};
use log::info;

use crate::xr::context::XrContext;

pub struct VkContext {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,

    pub multiview: bool,

    pub timestamp_period_ns: Option<f32>,

    pub shader_f16: bool,

    pub robust_access: bool,

    pub enabled_extensions: Vec<&'static std::ffi::CStr>,

    pub fragment_density_map: bool,

    pub dual_src_blend: bool,
}

const ENABLE_ROBUST_ACCESS: bool = false;

const PIPELINE_STATISTICS_PROPERTY: &std::ffi::CStr = c"debug.spacesoup.pipestats";

const EYE_CAPTURE_PROPERTY: &std::ffi::CStr = c"debug.spacesoup.eyecapture";

pub(crate) fn eye_capture_enabled() -> bool {
    system_property(EYE_CAPTURE_PROPERTY).trim() == "1"
}

fn system_property(name: &std::ffi::CStr) -> String {
    extern "C" {
        fn __system_property_get(name: *const std::ffi::c_char, value: *mut std::ffi::c_char) -> std::ffi::c_int;
    }

    let mut value = [0 as std::ffi::c_char; 92];
    let n = unsafe { __system_property_get(name.as_ptr(), value.as_mut_ptr()) };
    if n <= 0 {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(value.as_ptr()) }.to_string_lossy().into_owned()
}

impl VkContext {
    pub fn new(xr: &XrContext) -> Result<Self, Box<dyn std::error::Error>> {
        let vk_entry = ash::Entry::linked();

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 1, 0),
            ..Default::default()
        };

        let vk_instance = unsafe {
            let ci = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                ..Default::default()
            };
            let raw = xr
                .instance
                .create_vulkan_instance(
                    xr.system,
                    std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                    &ci as *const _ as *const _,
                )?
                .map_err(vk::Result::from_raw)?;
            ash::Instance::load(vk_entry.static_fn(), vk::Instance::from_raw(raw as _))
        };

        let physical_device = vk::PhysicalDevice::from_raw(unsafe {
            xr.instance
                .vulkan_graphics_device(xr.system, vk_instance.handle().as_raw() as _)?
                as _
        });

        let queue_family_index = unsafe {
            vk_instance
                .get_physical_device_queue_family_properties(physical_device)
                .into_iter()
                .enumerate()
                .find_map(|(i, p)| {
                    p.queue_flags
                        .contains(vk::QueueFlags::GRAPHICS)
                        .then_some(i as u32)
                })
                .ok_or("No graphics queue family")?
        };

        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };

        let mut multiview_query = vk::PhysicalDeviceMultiviewFeatures::default();
        let mut f16_query = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut robustness2_query = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
        let mut image_robustness_query = vk::PhysicalDeviceImageRobustnessFeatures::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut multiview_query)
            .push_next(&mut f16_query)
            .push_next(&mut robustness2_query)
            .push_next(&mut image_robustness_query);
        unsafe {
            vk_instance.get_physical_device_features2(physical_device, &mut features2);
        }
        let robust_buffer_access_supported = features2.features.robust_buffer_access == vk::TRUE;
        let dual_src_blend = features2.features.dual_src_blend == vk::TRUE;
        let robustness2_extension_available = unsafe {
            vk_instance
                .enumerate_device_extension_properties(physical_device)
                .map(|exts| {
                    exts.iter().any(|e| {
                        std::ffi::CStr::from_ptr(e.extension_name.as_ptr()).to_bytes() == b"VK_EXT_robustness2"
                    })
                })
                .unwrap_or(false)
        };
        let robust_access = ENABLE_ROBUST_ACCESS
            && robust_buffer_access_supported
            && robustness2_extension_available
            && robustness2_query.robust_buffer_access2 == vk::TRUE
            && robustness2_query.robust_image_access2 == vk::TRUE;
        let multiview_supported = multiview_query.multiview == vk::TRUE;
        info!("vulkan: dualSrcBlend {dual_src_blend}");

        info!(
            "vulkan: robustBufferAccess {}, robustBufferAccess2 {}, robustImageAccess2 {}, robustImageAccess {} -- {}",
            robust_buffer_access_supported,
            robustness2_query.robust_buffer_access2 == vk::TRUE,
            robustness2_query.robust_image_access2 == vk::TRUE,
            image_robustness_query.robust_image_access == vk::TRUE,
            if robust_access {
                "ENABLING robust access"
            } else {
                "robust access off (the shaders' own bounds checks are the guard)"
            },
        );
        info!(
            "vulkan: multiview {}",
            if multiview_supported { "supported -- enabling" } else { "NOT supported" },
        );

        let mesh_shader_supported = unsafe {
            vk_instance
                .enumerate_device_extension_properties(physical_device)
                .map(|exts| {
                    exts.iter().any(|e| {
                        std::ffi::CStr::from_ptr(e.extension_name.as_ptr())
                            .to_bytes()
                            == b"VK_EXT_mesh_shader"
                    })
                })
                .unwrap_or(false)
        };
        info!(
            "vulkan: VK_EXT_mesh_shader {} (nothing depends on this; \
             compute culling + indirect draw is the path either way)",
            if mesh_shader_supported { "SUPPORTED" } else { "not supported" },
        );

        let f16_extension_available = unsafe {
            vk_instance
                .enumerate_device_extension_properties(physical_device)
                .map(|exts| {
                    exts.iter().any(|e| {
                        std::ffi::CStr::from_ptr(e.extension_name.as_ptr()).to_bytes()
                            == b"VK_KHR_shader_float16_int8"
                    })
                })
                .unwrap_or(false)
        };
        let f16_supported = f16_extension_available
            && f16_query.shader_float16 == vk::TRUE

            && system_property(c"debug.spacesoup.nof16") != "1";

        match system_property(c"debug.spacesoup.painted").as_str() {
            "0" => crate::renderer::terrain_pipeline::pin_ground_build(false),
            "1" => crate::renderer::terrain_pipeline::pin_ground_build(true),
            "expect0" => crate::renderer::terrain_pipeline::expect_painted_ground(false),
            _ => {}
        }

        if let Ok(n) = system_property(c"debug.spacesoup.chunks").parse::<u32>() {
            crate::renderer::ground_horizon::CHUNKS_A_SIDE.store(n.min(64), std::sync::atomic::Ordering::Relaxed);
        }

        if let Ok(mrad) = system_property(c"debug.spacesoup.groundlodmrad").parse::<f32>() {
            if mrad > 0.0 {
                crate::renderer::ground_lod::TOLERANCE.store((mrad * 1e-3).to_bits(), std::sync::atomic::Ordering::Relaxed);
                log::info!("GROUNDLOD: a coarser ground may stand {mrad} mrad from step 1");
            }
        }

        if system_property(c"debug.spacesoup.groundshape") == "0" {
            crate::renderer::ground_shape::ENABLED.store(false, std::sync::atomic::Ordering::Relaxed);
            log::info!("GROUNDSHAPE: off -- the ground shaded by its vertices");
        }

        if system_property(c"debug.spacesoup.groundblocks") == "0" {
            crate::renderer::ground_lod::BLOCK_ORDER.store(false, std::sync::atomic::Ordering::Relaxed);
            log::info!("GROUNDBLOCKS: off -- each chunk its own runs");
        }
        if system_property(c"debug.spacesoup.groundlod") == "0" {
            crate::renderer::ground_lod::LISTS.store(false, std::sync::atomic::Ordering::Relaxed);
            log::info!("GROUNDLOD: off -- no coarser lists built; the ground is step 1 everywhere");
        }

        if system_property(c"debug.spacesoup.opaque") == "0" {
            crate::renderer::terrain_pipeline::OPAQUE_GROUND.store(false, std::sync::atomic::Ordering::Relaxed);
            log::info!("OPAQUE: off -- the scene's ground is drawn with blending");
        }

        let world_ground = system_property(c"debug.spacesoup.worldground") != "0";

        if system_property(c"debug.spacesoup.bodyfilmtraced") == "1" {
            crate::renderer::mesh_pipeline::SKINNED_FILM_TRACED.store(true, std::sync::atomic::Ordering::Relaxed);
            log::info!("BODYFILMTRACED: on -- the wet body traces the rooms' reflections");
        }
        if system_property(c"debug.spacesoup.torchskywalk") == "0" {
            crate::renderer::ground_twins::TORCH_WEATHER_SKY_WALK.store(false, std::sync::atomic::Ordering::Relaxed);
            log::info!("TORCHSKYWALK: off -- the torch-lit weather twins read the sky twice");
        }
        crate::renderer::terrain_pipeline::WORLD_GROUND.store(world_ground, std::sync::atomic::Ordering::Relaxed);

        let world_brushes = match system_property(c"debug.spacesoup.worldbrushes").as_str() {
            "0" => false,
            "1" => true,
            _ => crate::renderer::static_brush::WORLD_BRUSHES_BY_DEFAULT,
        };
        crate::renderer::brush_pipeline::WORLD_BRUSHES.store(world_brushes, std::sync::atomic::Ordering::Relaxed);

        let statistics_extension_available = unsafe {
            vk_instance
                .enumerate_device_extension_properties(physical_device)
                .map(|exts| {
                    exts.iter().any(|e| {
                        std::ffi::CStr::from_ptr(e.extension_name.as_ptr()).to_bytes()
                            == b"VK_KHR_pipeline_executable_properties"
                    })
                })
                .unwrap_or(false)
        };
        let capture_statistics =
            statistics_extension_available && system_property(PIPELINE_STATISTICS_PROPERTY) == "1";

        let fdm_extension_available = unsafe {
            vk_instance
                .enumerate_device_extension_properties(physical_device)
                .map(|exts| {
                    exts.iter().any(|e| {
                        std::ffi::CStr::from_ptr(e.extension_name.as_ptr()).to_bytes()
                            == b"VK_EXT_fragment_density_map"
                    })
                })
                .unwrap_or(false)
        };
        let mut fdm_query = vk::PhysicalDeviceFragmentDensityMapFeaturesEXT::default();
        let mut fdm_props = vk::PhysicalDeviceFragmentDensityMapPropertiesEXT::default();
        if fdm_extension_available {
            unsafe {
                let mut q = vk::PhysicalDeviceFeatures2::default().push_next(&mut fdm_query);
                vk_instance.get_physical_device_features2(physical_device, &mut q);
                let mut p = vk::PhysicalDeviceProperties2::default().push_next(&mut fdm_props);
                vk_instance.get_physical_device_properties2(physical_device, &mut p);
            }
        }
        let fdm_supported = fdm_extension_available
            && fdm_query.fragment_density_map == vk::TRUE
            && fdm_query.fragment_density_map_non_subsampled_images == vk::TRUE

            && system_property(c"debug.spacesoup.nofdm") != "1";
        info!(
            "vulkan: fragment density map: extension {}, map {}, dynamic {}, non-subsampled images {}, texel {}x{}..{}x{} -- {}",
            fdm_extension_available,
            fdm_query.fragment_density_map == vk::TRUE,
            fdm_query.fragment_density_map_dynamic == vk::TRUE,
            fdm_query.fragment_density_map_non_subsampled_images == vk::TRUE,
            fdm_props.min_fragment_density_texel_size.width,
            fdm_props.min_fragment_density_texel_size.height,
            fdm_props.max_fragment_density_texel_size.width,
            fdm_props.max_fragment_density_texel_size.height,
            if fdm_supported { "ENABLING (fixed foveated rendering)" } else { "no foveation" },
        );
        if capture_statistics {
            info!("vulkan: pipeline statistics ON -- every pipeline logs PIPESTATS");
        }
        crate::renderer::shader_checks::PIPELINE_STATISTICS.store(capture_statistics, std::sync::atomic::Ordering::Relaxed);
        info!(
            "vulkan: shaderFloat16 {}",
            if f16_supported { "supported -- enabling" } else { "NOT supported" },
        );
        let mut enabled_extensions: Vec<&'static std::ffi::CStr> = Vec::new();
        if f16_supported {
            enabled_extensions.push(ash::khr::shader_float16_int8::NAME);
        }
        if robust_access {
            enabled_extensions.push(ash::ext::robustness2::NAME);
        }
        if capture_statistics {
            enabled_extensions.push(ash::khr::pipeline_executable_properties::NAME);
        }
        if fdm_supported {
            enabled_extensions.push(ash::ext::fragment_density_map::NAME);
        }
        let extensions: Vec<*const std::ffi::c_char> = enabled_extensions.iter().map(|e| e.as_ptr()).collect();

        let mut robustness2_enable = vk::PhysicalDeviceRobustness2FeaturesEXT::default()
            .robust_buffer_access2(true)
            .robust_image_access2(true);
        let mut f16_enable = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(true);
        let mut multiview_enable = vk::PhysicalDeviceMultiviewFeatures::default()
            .multiview(multiview_supported);
        if robust_access {
            f16_enable.p_next = &mut robustness2_enable as *mut _ as *mut std::ffi::c_void;
        }
        if f16_supported || robust_access {
            multiview_enable.p_next = &mut f16_enable as *mut _ as *mut std::ffi::c_void;
            if !f16_supported {

                multiview_enable.p_next = &mut robustness2_enable as *mut _ as *mut std::ffi::c_void;
            }
        }

        let mut statistics_enable = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
            .pipeline_executable_info(true);
        statistics_enable.p_next = &mut multiview_enable as *mut _ as *mut std::ffi::c_void;
        let chain: *mut std::ffi::c_void = if capture_statistics {
            &mut statistics_enable as *mut _ as *mut std::ffi::c_void
        } else {
            &mut multiview_enable as *mut _ as *mut std::ffi::c_void
        };

        let mut fdm_enable = vk::PhysicalDeviceFragmentDensityMapFeaturesEXT::default()
            .fragment_density_map(true)
            .fragment_density_map_non_subsampled_images(true);
        fdm_enable.p_next = chain;
        let chain: *mut std::ffi::c_void = if fdm_supported {
            &mut fdm_enable as *mut _ as *mut std::ffi::c_void
        } else {
            chain
        };

        let core_features = vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(robust_access)
            .dual_src_blend(dual_src_blend);
        let device_ci = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            p_next: chain,
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: if extensions.is_empty() { std::ptr::null() } else { extensions.as_ptr() },
            p_enabled_features: &core_features,
            ..Default::default()
        };

        let device = unsafe {
            let raw = xr
                .instance
                .create_vulkan_device(
                    xr.system,
                    std::mem::transmute(vk_entry.static_fn().get_instance_proc_addr),
                    physical_device.as_raw() as _,
                    &device_ci as *const _ as *const _,
                )?
                .map_err(vk::Result::from_raw)?;
            ash::Device::load(vk_instance.fp_v1_0(), vk::Device::from_raw(raw as _))
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let timestamp_period_ns = unsafe {
            let families = vk_instance
                .get_physical_device_queue_family_properties(physical_device);
            let valid_bits = families
                .get(queue_family_index as usize)
                .map(|f| f.timestamp_valid_bits)
                .unwrap_or(0);
            let period = vk_instance
                .get_physical_device_properties(physical_device)
                .limits
                .timestamp_period;
            if valid_bits > 0 && period > 0.0 {
                Some(period)
            } else {
                None
            }
        };
        info!(
            "vulkan: timestamp queries {}",
            match timestamp_period_ns {
                Some(p) => format!("available ({p} ns/tick)"),
                None => "NOT available on this queue family".to_string(),
            },
        );
        info!("Vulkan ready");

        Ok(Self {
            multiview: multiview_supported,
            shader_f16: f16_supported,
            robust_access,
            enabled_extensions,
            fragment_density_map: fdm_supported,
            dual_src_blend,
            timestamp_period_ns,
            instance: vk_instance,
            physical_device,
            device,
            queue,
            queue_family_index,
        })
    }
}
