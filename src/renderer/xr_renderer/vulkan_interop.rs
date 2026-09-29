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
        // NEW IN wgpu 28: how close to the memory budget the allocator may run
        // before it starts evicting. The default is what wgpu uses for a device
        // it creates itself, and this device is an ordinary one -- it is shared
        // with OpenXR rather than constrained differently.
        wgpu::MemoryBudgetThresholds::default(),
        false,
        None,
    )?;

    let exposed = shared_instance
        .expose_adapter(vk.physical_device)
        .ok_or("wgpu: failed to expose physical device")?;

    // Claim MULTIVIEW only when the device actually enabled it.
    //
    // `device_from_raw` believes whatever it is told: this renderer builds its
    // own `VkDevice`, so wgpu cannot check. Claiming a feature that was never
    // switched on gets pipelines that pass validation here and go wrong on the
    // headset -- exactly the class of failure that is invisible on a
    // development machine, where wgpu runs on Metal and has no multiview at all.
    let mut features = if vk.multiview {
        wgpu::Features::MULTIVIEW
    } else {
        wgpu::Features::empty()
    };
    // Same contract as MULTIVIEW: claimed only because `VkContext` asked the
    // queue family whether it can actually timestamp. Without this the renderer
    // can measure the whole frame and nothing inside it, which is how three
    // successive guesses about where GPU time goes all turned out wrong.
    if vk.timestamp_period_ns.is_some() {
        features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    // HALF-PRECISION ARITHMETIC, on the same contract: `VkContext` enabled
    // `shaderFloat16` itself, or this is not claimed. Stock wgpu would also
    // want 16-bit uniform access for it, which Adreno lacks; our naga fork
    // asks for that only where a buffer holds a 16-bit type, and no buffer
    // here does. See `VkContext::shader_f16`.
    if vk.shader_f16 {
        features |= wgpu::Features::SHADER_F16;
    }
    // MULTISAMPLE_ARRAY: the feature that makes MULTIVIEW USABLE HERE AT ALL.
    //
    // WebGPU forbids a texture that is both multisampled and layered -- wgpu
    // refuses one with "Multisampled texture depth or array layers must be 1".
    // A multiview pass draws into the LAYERS of one attachment, and this
    // renderer runs the scene pass at 4x MSAA, so before wgpu 28 the two were
    // mutually exclusive and multiview could not be had without giving up
    // antialiasing. Vulkan itself has always allowed it; the restriction was
    // WebGPU's.
    //
    // wgpu 28 added this feature to lift it, on Vulkan. That is the whole
    // reason this renderer moved from wgpu 25 to 28.
    // See https://github.com/gfx-rs/wgpu/pull/8571.
    //
    // Asked of the ADAPTER rather than assumed, on the same contract as
    // MULTIVIEW above: `device_from_raw` believes whatever it is told, and a
    // claimed feature that the driver does not have is a texture that fails to
    // create at startup rather than a warning.
    let multisample_array = exposed
        .features
        .contains(wgpu::Features::MULTISAMPLE_ARRAY);
    if multisample_array {
        features |= wgpu::Features::MULTISAMPLE_ARRAY;
    }
    // EARLY FRAGMENT TESTS on demand: SPIR-V's `EarlyFragmentTests`, core
    // Vulkan, needs nothing enabled. The probe pass that defers its lookups
    // writes a storage buffer, and a fragment shader with side effects is
    // otherwise depth-tested AFTER it runs -- every hidden fragment shaded in
    // full. See `BrushPipeline::new_probe_pass_deferred`. Asked of the adapter,
    // on the same contract as the rest.
    if exposed.features.contains(wgpu::Features::SHADER_EARLY_DEPTH_TEST) {
        features |= wgpu::Features::SHADER_EARLY_DEPTH_TEST;
    }
    log::info!(
        "wgpu: shader f16 {}",
        if vk.shader_f16 { "yes (arithmetic only)" } else { "NO" },
    );
    log::info!(
        "wgpu: multiview {}, multisampled arrays {} -- stereo scene pass {}",
        if vk.multiview { "yes" } else { "NO" },
        if multisample_array { "yes" } else { "NO" },
        if vk.multiview && multisample_array { "AVAILABLE" } else { "unavailable" },
    );

    // THE LIMITS ARE DECIDED ONCE, here, because since wgpu 30 they are passed
    // to `device_from_raw` AS WELL AS to the device descriptor below, and the
    // two disagreeing is a device that validates against one set and runs
    // against another. Read off `exposed` rather than the wgpu adapter, which
    // does not exist yet -- it is the same physical device either way, and
    // `exposed` is moved into `create_adapter_from_hal` further down.
    let limits = wgpu::Limits {
        max_texture_dimension_2d: exposed.capabilities.limits.max_texture_dimension_2d,
        // A FEATURE IS NOT ENOUGH; THE LIMIT HAS TO ALLOW IT TOO.
        //
        // `max_multiview_view_count` defaults to ZERO in every stock limit set,
        // including `downlevel_defaults`. So a device can report MULTIVIEW,
        // build every stereo pipeline without complaint, and then reject the
        // pass itself with "Multiview view count limit violated" -- which is a
        // scene that never draws, over which the eye pass keeps compositing
        // live, so one eye looks frozen under moving content (2026-09-19).
        //
        // AT LEAST 2, because the ADAPTER REPORTS 0 and is wrong about this
        // device.
        //
        // wgpu's adapter fills that limit from its own feature detection, and
        // this renderer builds its own `VkDevice` -- so wgpu never saw the
        // multiview feature get enabled and left the limit at its default of
        // zero. Measured on the headset: "multiview yes ... AVAILABLE" and
        // "max_multiview_view_count = 0" in consecutive log lines.
        //
        // Two is not a guess. `VkContext` asked the physical device directly
        // (`VkPhysicalDeviceMultiviewFeatures::multiview`) and enabled it at
        // device creation; the Vulkan spec's floor for `maxMultiviewViewCount`
        // once multiview is supported is 6. The adapter's own number is kept
        // when it is larger, in case a future wgpu learns to see it.
        max_multiview_view_count: exposed
            .capabilities
            .limits
            .max_multiview_view_count
            .max(crate::renderer::multiview::STEREO_VIEWS),
        // THE HARDWARE'S ARRAY-LAYER LIMIT, not wgpu's default of 256.
        //
        // 256 layers is 42 cubes, and that number sat in the probe code as a
        // "hardware" ceiling on how many reflection probes a level could have.
        // It was never the hardware: Adreno 740 reports maxImageArrayLayers =
        // 2048 (vulkan.gpuinfo.org), and it is the most common value on
        // Android. The probe pool is sized from this at load
        // (`probe_stream::pool_size`), so the log line below is the device
        // itself confirming what it allows.
        max_texture_array_layers: exposed
            .capabilities
            .limits
            .max_texture_array_layers
            .max(wgpu::Limits::downlevel_defaults().max_texture_array_layers),
        // THE SCENE SHADERS' TEXTURES: 17 in the brush shader that reads the
        // probe pass, one over WebGPU's portable 16. As many as the scene asks
        // for (`uniforms::scene_limits`), as far as the device allows -- and
        // the Adreno's Vulkan allows far more.
        max_sampled_textures_per_shader_stage: exposed
            .capabilities
            .limits
            .max_sampled_textures_per_shader_stage
            .min(crate::renderer::uniforms::SCENE_SAMPLED_TEXTURES)
            .max(wgpu::Limits::downlevel_defaults().max_sampled_textures_per_shader_stage),
        ..wgpu::Limits::downlevel_defaults()
    };
    log::info!(
        "wgpu: max_multiview_view_count = {}, max_texture_array_layers = {}, max_sampled_textures_per_shader_stage = {}",
        limits.max_multiview_view_count,
        limits.max_texture_array_layers,
        limits.max_sampled_textures_per_shader_stage,
    );
    let open_device = exposed.adapter.device_from_raw(
        vk.device.clone(),
        None,
        // WHAT THIS DEVICE REALLY HAS. The SpaceSoupVR fork's wgpu-hal trusts
        // robust access only when its extension is listed here; stock wgpu-hal
        // trusted the physical device's support, and compiled every shader
        // without its bounds checks on a device with robust access off.
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

/// [`import_vk_image_as_wgpu`] with its usages given: a depth swapchain's
/// image is a depth target, not a colour one.
pub(super) unsafe fn import_vk_image_as_wgpu_with(
    device: &wgpu::Device,
    image: vk::Image,
    wgpu_format: wgpu::TextureFormat,
    (width, height, array_layers): (u32, u32, u32),
    hal_usage: wgpu::TextureUses,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    use wgpu::hal::vulkan as hvk;

    // A METHOD ON THE HAL DEVICE since wgpu 28, not a free function. Getting
    // at it is `as_hal`, which returns `None` on any backend but Vulkan -- and
    // this whole module only exists on the headset, where the backend is
    // Vulkan by construction.
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
        // EXTERNAL: this image belongs to the OpenXR swapchain. Telling wgpu it
        // owns the memory would have it freed underneath the runtime.
        hvk::TextureMemory::External,
    );
    // Released before the texture is handed back, so nothing holds a borrow of
    // the device across the call below.
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
        // UNINITIALIZED: a freshly acquired swapchain image holds nothing we
        // are entitled to read, and the scene pass clears it. Claiming it
        // already held colour would let wgpu skip a transition the runtime
        // expects.
        wgpu::TextureUses::UNINITIALIZED,
    )
}

/// FIXED FOVEATED RENDERING from here on: every render pass carries a density
/// map, full density unless it draws into a registered eye target (the
/// SpaceSoupVR wgpu fork; see `foveation`). Must run before ANY pipeline is
/// created -- a pipeline made earlier is incompatible with every pass after.
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

/// Density maps for `patterns` (`(width, height, texels)`), written and ready;
/// their indices, or `None` if the device refused them.
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

/// A pass drawing into `view` gets density map `map`, or full density.
pub(super) unsafe fn set_foveation_target(device: &wgpu::Device, view: &wgpu::TextureView, map: Option<usize>) {
    use wgpu::hal::vulkan as hvk;
    let Some(hal) = (unsafe { device.as_hal::<hvk::Api>() }) else { return };
    let Some(raw) = (unsafe { view.as_hal::<hvk::Api>() }).map(|v| unsafe { v.raw_handle() }) else { return };
    hal.set_foveation_target(raw, map);
}
