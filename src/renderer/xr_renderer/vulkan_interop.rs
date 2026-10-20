use ash::vk;

use crate::xr::VkContext;

pub(super) unsafe fn build_wgpu_from_vulkan(
    vk: &VkContext,
) -> Result<(wgpu::Device, wgpu::Queue), Box<dyn std::error::Error>> {
    use wgpu::hal::vulkan as hvk;

    let shared_instance = hvk::Instance::from_raw(
        ash::Entry::linked(),
        vk.instance.clone(),
        vk::make_api_version(0, 1, 1, 0),
        0,
        None,
        vec![],
        wgpu::InstanceFlags::empty(),

        wgpu::MemoryBudgetThresholds::default(),
        false,
        None,
    )?;

    let exposed = shared_instance
        .expose_adapter(vk.physical_device)
        .ok_or("wgpu: failed to expose physical device")?;

    let mut features = if vk.multiview {
        wgpu::Features::MULTIVIEW
    } else {
        wgpu::Features::empty()
    };

    if vk.timestamp_period_ns.is_some() {
        features |= wgpu::Features::TIMESTAMP_QUERY;
    }

    if vk.shader_f16 {
        features |= wgpu::Features::SHADER_F16;
    }

    if vk.dual_src_blend {
        features |= wgpu::Features::DUAL_SOURCE_BLENDING;
    }

    let multisample_array = exposed
        .features
        .contains(wgpu::Features::MULTISAMPLE_ARRAY);
    if multisample_array {
        features |= wgpu::Features::MULTISAMPLE_ARRAY;
    }

    if exposed.features.contains(wgpu::Features::SHADER_EARLY_DEPTH_TEST) {
        features |= wgpu::Features::SHADER_EARLY_DEPTH_TEST;
    }

    if exposed.features.contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM) {
        features |= wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    }

    if exposed.features.contains(wgpu::Features::IMMEDIATES) {
        features |= wgpu::Features::IMMEDIATES;
    }
    // The ground's layers are encoded ahead as ASTC where the game ships them
    // so (`TerrainLayerArrays::from_astc`): a quarter of the memory.
    if exposed.features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
        features |= wgpu::Features::TEXTURE_COMPRESSION_ASTC;
    }
    log::info!("wgpu: ASTC {}", if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) { "yes" } else { "NO" });
    {
        use wgpu::hal::Adapter as _;
        let rgba16 = exposed.adapter.texture_format_capabilities(wgpu::TextureFormat::Rgba16Unorm);
        log::info!(
            "wgpu: 16-bit unorm {}; Rgba16Unorm {:?}",
            if features.contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM) { "yes" } else { "NO" },
            rgba16,
        );
    }
    log::info!(
        "wgpu: shader f16 {}",
        if vk.shader_f16 { "yes (arithmetic only)" } else { "NO" },
    );
    log_driver_offer(vk);
    log::info!(
        "wgpu: multiview {}, multisampled arrays {} -- stereo scene pass {}",
        if vk.multiview { "yes" } else { "NO" },
        if multisample_array { "yes" } else { "NO" },
        if vk.multiview && multisample_array { "AVAILABLE" } else { "unavailable" },
    );

    let limits = wgpu::Limits {
        max_texture_dimension_2d: exposed.capabilities.limits.max_texture_dimension_2d,

        max_multiview_view_count: exposed
            .capabilities
            .limits
            .max_multiview_view_count
            .max(crate::renderer::multiview::STEREO_VIEWS),

        max_texture_array_layers: exposed
            .capabilities
            .limits
            .max_texture_array_layers
            .max(wgpu::Limits::downlevel_defaults().max_texture_array_layers),

        max_sampled_textures_per_shader_stage: exposed
            .capabilities
            .limits
            .max_sampled_textures_per_shader_stage
            .min(crate::renderer::uniforms::SCENE_SAMPLED_TEXTURES)
            .max(wgpu::Limits::downlevel_defaults().max_sampled_textures_per_shader_stage),

        max_immediate_size: if features.contains(wgpu::Features::IMMEDIATES) {
            exposed.capabilities.limits.max_immediate_size
        } else {
            0
        },
        ..wgpu::Limits::downlevel_defaults()
    };
    log::info!(
        "wgpu: max_multiview_view_count = {}, max_texture_array_layers = {}, max_sampled_textures_per_shader_stage = {}, max_immediate_size = {}",
        limits.max_multiview_view_count,
        limits.max_texture_array_layers,
        limits.max_sampled_textures_per_shader_stage,
        limits.max_immediate_size,
    );
    let open_device = exposed.adapter.device_from_raw(
        vk.device.clone(),
        None,

        &vk.enabled_extensions,
        features,
        &limits,
        &wgpu::MemoryHints::default(),
        vk.queue_family_index,
        0,
    )?;

    let wgpu_instance = wgpu::Instance::from_hal::<hvk::Api>(shared_instance);
    let wgpu_adapter = wgpu_instance.create_adapter_from_hal(exposed);

    let (device, queue) = wgpu_adapter.create_device_from_hal(
        open_device,
        &wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: limits,
            ..Default::default()
        },
    )?;

    Ok((device, queue))
}

unsafe fn log_driver_offer(vk: &VkContext) {
    let props = unsafe { vk.instance.get_physical_device_properties(vk.physical_device) };
    let l = &props.limits;
    let mut inline = vk::PhysicalDeviceInlineUniformBlockProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut inline);
    unsafe { vk.instance.get_physical_device_properties2(vk.physical_device, &mut props2) };
    log::info!(
        "vulkan: api {}.{}.{}, driver {:#x}; push constants {} B; uniform range {} B, offset alignment {} B; \
         inline uniform block {} B, {} a stage; bound descriptor sets {}",
        vk::api_version_major(props.api_version),
        vk::api_version_minor(props.api_version),
        vk::api_version_patch(props.api_version),
        props.driver_version,
        l.max_push_constants_size,
        l.max_uniform_buffer_range,
        l.min_uniform_buffer_offset_alignment,
        inline.max_inline_uniform_block_size,
        inline.max_per_stage_descriptor_inline_uniform_blocks,
        l.max_bound_descriptor_sets,
    );
    let exts = unsafe { vk.instance.enumerate_device_extension_properties(vk.physical_device) }.unwrap_or_default();
    let names: Vec<String> = exts
        .iter()
        .filter_map(|e| e.extension_name_as_c_str().ok().map(|c| c.to_string_lossy().into_owned()))
        .collect();
    log::info!("vulkan: {} device extensions", names.len());
    for chunk in names.chunks(8) {
        log::info!("vulkan ext: {}", chunk.join(" "));
    }
}

pub(super) unsafe fn import_vk_image_as_wgpu(
    device: &wgpu::Device,
    image: vk::Image,
    wgpu_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    array_layers: u32,
) -> wgpu::Texture {
    unsafe {
        import_vk_image_as_wgpu_with(
            device,
            image,
            wgpu_format,
            (width, height, array_layers),
            wgpu::TextureUses::COLOR_TARGET | wgpu::TextureUses::RESOURCE,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        )
    }
}

pub(super) fn supports_depth_attachment(device: &wgpu::Device, format: vk::Format) -> bool {
    use wgpu::hal::vulkan as hvk;
    let Some(hal_device) = (unsafe { device.as_hal::<hvk::Api>() }) else {
        return false;
    };
    let props = unsafe {
        hal_device
            .shared_instance()
            .raw_instance()
            .get_physical_device_format_properties(hal_device.raw_physical_device(), format)
    };
    props.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
}

pub(super) fn filters_linearly(device: &wgpu::Device, format: vk::Format) -> bool {
    use wgpu::hal::vulkan as hvk;
    let Some(hal_device) = (unsafe { device.as_hal::<hvk::Api>() }) else {
        return false;
    };
    let props = unsafe {
        hal_device
            .shared_instance()
            .raw_instance()
            .get_physical_device_format_properties(hal_device.raw_physical_device(), format)
    };
    props
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
}

pub(super) unsafe fn import_vk_image_as_wgpu_with(
    device: &wgpu::Device,
    image: vk::Image,
    wgpu_format: wgpu::TextureFormat,
    (width, height, array_layers): (u32, u32, u32),
    hal_usage: wgpu::TextureUses,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    use wgpu::hal::vulkan as hvk;

    let hal_device = device
        .as_hal::<hvk::Api>()
        .expect("wgpu: the XR renderer requires the Vulkan backend");
    let hal_texture = hal_device.texture_from_raw(
        image,
        &wgpu::hal::TextureDescriptor {
            label: Some("xr_swapchain_image"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: array_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: hal_usage,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        },
        None,

        hvk::TextureMemory::External,
    );

    drop(hal_device);

    device.create_texture_from_hal::<hvk::Api>(
        hal_texture,
        &wgpu::TextureDescriptor {
            label: Some("xr_swapchain_tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: array_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage,
            view_formats: &[],
        },

        wgpu::TextureUses::UNINITIALIZED,
    )
}

const PIPELINE_CACHE_SAVES_S: [u64; 3] = [45, 180, 600];

pub(super) unsafe fn start_pipeline_cache(vk: &VkContext, device: &wgpu::Device) {
    use crate::renderer::pipeline_cache_file as file;
    use wgpu::hal::vulkan as hvk;

    let Some(hal) = (unsafe { device.as_hal::<hvk::Api>() }) else { return };
    let props = unsafe { vk.instance.get_physical_device_properties(vk.physical_device) };
    let key = file::CacheKey {
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        driver_version: props.driver_version,
        cache_uuid: props.pipeline_cache_uuid,
        build: file::build_identity(),
    };
    let path = ndk_glue::native_activity().external_data_path().join("pipelines.cache");
    let saved = std::fs::read(&path).ok();
    let initial: &[u8] = match saved.as_deref().map(|bytes| file::decode(&key, bytes)) {
        Some(Ok(data)) => data,
        Some(Err(why)) => {
            log::info!("pipeline cache: {} not used ({why:?}); every pipeline compiles", path.display());
            &[]
        }
        None => {
            log::info!("pipeline cache: none yet; every pipeline compiles");
            &[]
        }
    };
    let create = |data: &[u8]| unsafe {
        vk.device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default().initial_data(data), None)
    };

    let cache = match create(initial).or_else(|_| create(&[])) {
        Ok(cache) => cache,
        Err(e) => {
            log::warn!("pipeline cache: could not create one ({e}); every pipeline compiles");
            return;
        }
    };
    unsafe { hal.set_default_pipeline_cache(cache) };
    log::info!("pipeline cache: started from {} bytes (build {:016x})", initial.len(), key.build);

    let raw = vk.device.clone();
    let mut written = initial.len();
    let started = std::time::Instant::now();
    let saver = std::thread::Builder::new().name("pipeline-cache".into()).spawn(move || {
        for at in PIPELINE_CACHE_SAVES_S {
            let due = std::time::Duration::from_secs(at);
            std::thread::sleep(due.saturating_sub(started.elapsed()));
            let data = match unsafe { raw.get_pipeline_cache_data(cache) } {
                Ok(data) => data,
                Err(e) => {
                    log::warn!("pipeline cache: could not read it back ({e})");
                    return;
                }
            };
            if data.len() == written {
                continue;
            }
            match file::write_whole(&path, &file::encode(&key, &data)) {
                Ok(()) => {
                    log::info!("pipeline cache: wrote {} bytes at {at} s", data.len());
                    written = data.len();
                }
                Err(e) => log::warn!("pipeline cache: could not write {} ({e})", path.display()),
            }
        }
    });
    if let Err(e) = saver {
        log::warn!("pipeline cache: no thread to save it ({e}); it lasts this launch only");
    }
}

pub(super) unsafe fn enable_foveation(device: &wgpu::Device) -> bool {
    use wgpu::hal::vulkan as hvk;
    let Some(hal) = (unsafe { device.as_hal::<hvk::Api>() }) else { return false };
    match unsafe { hal.enable_foveation() } {
        Ok(()) => true,
        Err(e) => {
            log::warn!("foveation: could not create the density maps ({e}); every pixel is shaded");
            false
        }
    }
}

pub(super) unsafe fn add_foveation_maps(device: &wgpu::Device, patterns: &[(u32, u32, Vec<u8>)]) -> Option<Vec<usize>> {
    use wgpu::hal::vulkan as hvk;
    let hal = unsafe { device.as_hal::<hvk::Api>() }?;
    let patterns: Vec<hvk::DensityPattern> = patterns
        .iter()
        .map(|(width, height, texels)| hvk::DensityPattern { width: *width, height: *height, texels })
        .collect();
    match unsafe { hal.add_foveation_maps(&patterns) } {
        Ok(ids) => Some(ids),
        Err(e) => {
            log::warn!("foveation: could not add density maps ({e})");
            None
        }
    }
}

pub(super) unsafe fn set_foveation_target(device: &wgpu::Device, view: &wgpu::TextureView, map: Option<usize>) {
    use wgpu::hal::vulkan as hvk;
    let Some(hal) = (unsafe { device.as_hal::<hvk::Api>() }) else { return };
    let Some(raw) = (unsafe { view.as_hal::<hvk::Api>() }).map(|v| unsafe { v.raw_handle() }) else { return };
    hal.set_foveation_target(raw, map);
}
