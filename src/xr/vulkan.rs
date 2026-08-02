use ash::vk::{self, Handle};
use log::info;

use crate::xr::context::XrContext;

pub struct VkContext {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
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
        let queue_infos = [queue_info];

        // Multiview is core in Vulkan 1.1 (requested above), so it needs no
        // extension — but it is off unless the feature is enabled here. Without
        // it, wgpu cannot expose Features::MULTIVIEW and every single-pass
        // stereo pipeline fails at creation.
        //
        // `multiview_features` must outlive the create call: DeviceCreateInfo
        // holds a raw p_next pointer into it.
        let mut multiview_features =
            vk::PhysicalDeviceMultiviewFeatures::default().multiview(true);
        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .push_next(&mut multiview_features);

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
        info!("Vulkan ready");

        Ok(Self {
            instance: vk_instance,
            physical_device,
            device,
            queue,
            queue_family_index,
        })
    }
}

