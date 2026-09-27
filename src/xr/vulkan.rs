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
        let mut features2 =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut multiview_query);
        unsafe {
            vk_instance.get_physical_device_features2(physical_device, &mut features2);
        }
        let multiview_supported = multiview_query.multiview == vk::TRUE;
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

        let mut multiview_enable = vk::PhysicalDeviceMultiviewFeatures::default()
            .multiview(multiview_supported);
        let device_ci = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            p_next: if multiview_supported {
                &mut multiview_enable as *mut _ as *mut std::ffi::c_void
            } else {
                std::ptr::null_mut()
            },
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
            timestamp_period_ns,
            instance: vk_instance,
            physical_device,
            device,
            queue,
            queue_family_index,
        })
    }
}

