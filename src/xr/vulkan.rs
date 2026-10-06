use ash::vk::{self, Handle};
use log::info;

use crate::xr::context::XrContext;

pub struct VkContext {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    /// Whether `VkPhysicalDeviceMultiviewFeatures::multiview` was ENABLED on
    /// this device -- not merely supported by the hardware.
    ///
    /// The distinction matters because this renderer builds its own `VkDevice`
    /// and hands it to wgpu through `device_from_raw`, which TRUSTS the feature
    /// list it is given. Claiming `Features::MULTIVIEW` for a device that never
    /// enabled it produces pipelines that validate and then misbehave on the
    /// headset, with nothing on this side reporting a problem.
    pub multiview: bool,
    /// Nanoseconds per timestamp tick, or `None` when this queue cannot
    /// timestamp at all.
    ///
    /// Same rule as `multiview` above, for the same reason: `device_from_raw`
    /// trusts whatever feature list it is handed, so `Features::TIMESTAMP_QUERY`
    /// must be claimed only when the device really can do it. Timestamps are
    /// core Vulkan rather than an extension, so what decides it is not a feature
    /// bit but `timestampValidBits` on the QUEUE FAMILY -- a family can be a
    /// perfectly good graphics queue and still report zero, and then every
    /// timestamp written to it is silently meaningless rather than an error.
    ///
    /// `timestamp_period` is the scale: raw ticks mean nothing without it, and
    /// it is not 1.0 on Adreno.
    pub timestamp_period_ns: Option<f32>,
    /// Whether `shaderFloat16` was ENABLED (`VK_KHR_shader_float16_int8`):
    /// `f16` ARITHMETIC in shaders. Same contract as `multiview` -- claimed to
    /// wgpu as `Features::SHADER_F16` only when this is true.
    ///
    /// Arithmetic only. Adreno has no 16-bit uniform access
    /// (`uniformAndStorageBuffer16BitAccess`, gpuweb#5006), so every buffer
    /// stays 32-bit, and the SpaceSoupVR naga fork declares the 16-bit storage
    /// capabilities only for buffers that really hold a 16-bit type.
    pub shader_f16: bool,
    /// Whether ROBUST ACCESS is enabled: `robustBufferAccess`, and
    /// `robustBufferAccess2` + `robustImageAccess2` (`VK_EXT_robustness2`).
    /// Off: see `ENABLE_ROBUST_ACCESS`.
    pub robust_access: bool,
    /// The device extensions enabled here -- handed to wgpu's
    /// `device_from_raw`, which (in the SpaceSoupVR fork) trusts robust access
    /// only when its extension is among them.
    pub enabled_extensions: Vec<&'static std::ffi::CStr>,
    /// Whether FIXED FOVEATED RENDERING can run: `VK_EXT_fragment_density_map`
    /// enabled with `fragmentDensityMap` and
    /// `fragmentDensityMapNonSubsampledImages` -- the second is what lets a
    /// foveated pass draw straight into the OpenXR swapchain's ordinary
    /// images. See `renderer::foveation`.
    pub fragment_density_map: bool,
}

/// Whether to switch on the device's robust buffer and image access.
///
/// OFF, MEASURED (Quest 3, 2026-09-28): on, it cost 0.2-1.7 ms of GPU a
/// frame across the six benchmark views -- the hallway 1.73 ms. The shaders'
/// own index clamps protect against the same out-of-range reads far more
/// cheaply, and they are only compiled in when wgpu knows robust access is
/// OFF: stock wgpu-hal assumed it was on for any device that SUPPORTS it,
/// which is why `device_from_raw` is told, through `enabled_extensions`,
/// exactly what this device has (the fork's patch to wgpu-hal).
const ENABLE_ROBUST_ACCESS: bool = false;

/// The Android system property that makes every pipeline report the driver's
/// statistics for its shaders -- registers, instruction counts -- to the log,
/// as `PIPESTATS` lines (the SpaceSoupVR wgpu fork's `wgpu-hal`). Off unless
/// set; read once, when the device is created:
///
/// ```text
/// adb -s <quest> shell setprop debug.spacesoup.pipestats 1
/// ```
///
/// Measurement only: how many registers a shader holds decides how many waves
/// of it a GPU core keeps in flight, and no other tool on this headset says.
const PIPELINE_STATISTICS_PROPERTY: &std::ffi::CStr = c"debug.spacesoup.pipestats";

/// The Android system property that lets the renderer copy its eye images
/// out -- `Levers::eye_capture`. The colour swapchain is created able to be
/// copied from only when it is set, so the shipped build never pays for it.
/// Read once, when the renderer is created:
///
/// ```text
/// adb -s <quest> shell setprop debug.spacesoup.eyecapture 1
/// ```
const EYE_CAPTURE_PROPERTY: &std::ffi::CStr = c"debug.spacesoup.eyecapture";

/// Whether this run may copy its eye images out. See [`EYE_CAPTURE_PROPERTY`].
pub(crate) fn eye_capture_enabled() -> bool {
    system_property(EYE_CAPTURE_PROPERTY).trim() == "1"
}

/// An Android system property's value; empty when unset.
fn system_property(name: &std::ffi::CStr) -> String {
    extern "C" {
        fn __system_property_get(name: *const std::ffi::c_char, value: *mut std::ffi::c_char) -> std::ffi::c_int;
    }
    // PROP_VALUE_MAX.
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
        // MULTIVIEW: ask the physical device, then enable it if it says yes.
        //
        // Core in Vulkan 1.1, which the instance above requests, but a core
        // FEATURE still has to be switched on explicitly at device creation --
        // available and enabled are different things, and the failure mode for
        // confusing them is a pipeline that builds and then renders one eye.
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
        // ROBUST ACCESS, recorded: wgpu-hal decides from what the PHYSICAL
        // device supports whether naga's shaders clamp their own buffer and
        // image-load indices, on the assumption that it enabled the matching
        // robustness features itself -- which it does for a device it creates,
        // and which this device (built here, handed over by `device_from_raw`)
        // does not. Logged so the headset says which of them it has.
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

        // MESH SHADERS: asked once, answered permanently.
        //
        // Every plan that reaches for Nanite-style cluster rendering runs into
        // the same unresolved question -- whether the Adreno 740 in this
        // headset exposes `VK_EXT_mesh_shader` at all. The open-source Turnip
        // driver described A7XX mesh shader support as a "distant future...
        // maybe" item, Mesa's docs list no such extension, and no confirmation
        // was found for Qualcomm's proprietary driver either. That is an
        // absence of evidence, which is not the same as an answer.
        //
        // This makes it an answer. The device is right here and it knows; one
        // enumeration at startup costs nothing and ends the speculation in the
        // log rather than in another research round.
        //
        // NOTHING IS BUILT ON THE RESULT. The compute-culling + indirect-draw
        // path works on any Vulkan 1.x hardware and is the plan either way, so
        // this is a fact being recorded, not a branch being taken.
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

        // HALF-PRECISION ARITHMETIC: asked, then enabled, like multiview. The
        // extension is core in Vulkan 1.2, but this device is created at 1.1,
        // so it is named explicitly. The OpenXR runtime adds its own extensions
        // to the list when it creates the device.
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
            // MEASUREMENT: every shader at f32, from the next launch -- the
            // half-precision maths' control in one build (`shader_precision`).
            // `adb shell setprop debug.spacesoup.nof16 1`.
            && system_property(c"debug.spacesoup.nof16") != "1";
        // PIPELINE STATISTICS, when asked for. See `PIPELINE_STATISTICS_PROPERTY`.
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
        // FIXED FOVEATED RENDERING: the density map extension and the two
        // features it needs here, asked for only where the extension exists
        // (a feature struct of an extension the device lacks is not valid to
        // query). See `VkContext::fragment_density_map`.
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
            // DIAGNOSIS: no density map on any pass, from the next launch --
            // `adb shell setprop debug.spacesoup.nofdm 1`.
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

        // The feature chain: multiview, then f16, then robustness2, each
        // enabled only where supported. (A multiview struct with multiview
        // off is harmless.)
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
                // f16 off: skip its struct, keep robustness2 in the chain.
                multiview_enable.p_next = &mut robustness2_enable as *mut _ as *mut std::ffi::c_void;
            }
        }
        // Pipeline statistics, when asked for, at the head of the chain.
        let mut statistics_enable = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
            .pipeline_executable_info(true);
        statistics_enable.p_next = &mut multiview_enable as *mut _ as *mut std::ffi::c_void;
        let chain: *mut std::ffi::c_void = if capture_statistics {
            &mut statistics_enable as *mut _ as *mut std::ffi::c_void
        } else {
            &mut multiview_enable as *mut _ as *mut std::ffi::c_void
        };
        // The density map, when there is one, at the head of the chain.
        let mut fdm_enable = vk::PhysicalDeviceFragmentDensityMapFeaturesEXT::default()
            .fragment_density_map(true)
            .fragment_density_map_non_subsampled_images(true);
        fdm_enable.p_next = chain;
        let chain: *mut std::ffi::c_void = if fdm_supported {
            &mut fdm_enable as *mut _ as *mut std::ffi::c_void
        } else {
            chain
        };
        // `robustBufferAccess2` requires the core `robustBufferAccess` too.
        let core_features = vk::PhysicalDeviceFeatures::default().robust_buffer_access(robust_access);
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

        // TIMESTAMPS: ask the queue family, not the feature list.
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
            timestamp_period_ns,
            instance: vk_instance,
            physical_device,
            device,
            queue,
            queue_family_index,
        })
    }
}

