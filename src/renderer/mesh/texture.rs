pub struct LoadedTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    _sampler: wgpu::Sampler,
    pub bind_group: wgpu::BindGroup,
    /// Held only so the bind group's view stays valid. Never sampled from Rust.
    _direction: Option<wgpu::Texture>,
    /// The brush sun mask, its sampler and the stationary lamps' masks, held
    /// for the same reason.
    _sun_mask: Option<(wgpu::Texture, wgpu::Sampler, wgpu::Texture)>,
}

pub(crate) fn load_primitive_texture(
    prim: &gltf::Primitive,
    images: &[gltf::image::Data],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> LoadedTexture {
    let material = prim.material();
    let pbr = material.pbr_metallic_roughness();
    // GLASS IS NOT OPAQUE, even when it forgets to say so.
    //
    // glTF's `KHR_materials_transmission` is how a modern asset describes clear
    // glass, and it does NOT set `alphaMode: BLEND` -- transmission is a
    // separate mechanism from alpha coverage, so a transmissive material is
    // formally opaque and this test called it opaque.
    //
    // The result was a lamp whose bulb never appeared to glow. The hanging
    // fixture in this project is two primitives: a housing that carries the
    // emissive, and a glass envelope around the bulb with a transmission factor
    // of 1.0. Drawn opaque, that envelope is a grey dome sitting directly in
    // front of the only part that glows -- so the emissive was rendering
    // correctly the whole time and nothing could see it.
    //
    // Approximated as ALPHA rather than as refraction: this renderer has no
    // transmission path, and clear glass that is simply see-through is far
    // closer to right than clear glass painted grey.
    let transmission = material
        .transmission()
        .map(|t| t.transmission_factor())
        .unwrap_or(0.0);
    let force_opaque =
        material.alpha_mode() != gltf::material::AlphaMode::Blend && transmission <= 0.0;

    let base = match pbr.base_color_texture() {
        Some(info) => decode_image(&images[info.texture().source().index()], force_opaque),
        None => {
            let c = pbr.base_color_factor();
            (
                vec![
                    (c[0] * 255.0) as u8,
                    (c[1] * 255.0) as u8,
                    (c[2] * 255.0) as u8,
                    if force_opaque {
                        255
                    } else {
                        // What the surface still BLOCKS. A transmission of 1.0
                        // is clear glass and blocks nothing.
                        (c[3] * (1.0 - transmission) * 255.0) as u8
                    },
                ],
                1,
                1,
            )
        }
    };

    // THE EMISSIVE MASK, and it is not optional in practice.
    //
    // glTF defines emission as `emissiveFactor * emissiveTexture`, and real
    // fixtures use the texture to say WHICH PART glows. Poly Haven's
    // hanging_industrial_lamp is one material covering the whole lamp, with
    // emissiveFactor [1,1,1] and a texture that is black everywhere except the
    // bulb. Honouring only the factor makes the entire housing glow white --
    // the factor alone is not "how much this material emits", it is a tint on a
    // mask that has to be sampled.
    //
    // A material with no emissive texture gets a WHITE 1x1, so the factor is
    // multiplied by one and a material that emits uniformly still works. Black
    // would silence every such material, and the factor would do nothing.
    let emissive = match material.emissive_texture() {
        Some(info) => {
            let (mut rgba, w, h) = decode_image(&images[info.texture().source().index()], true);
            clean_emissive_mask(&mut rgba);
            (rgba, w, h)
        }
        None => (vec![255u8, 255, 255, 255], 1, 1),
    };

    create_mesh_material_texture(device, queue, layout, &base, &emissive)
}

/// THE EMISSIVE MASK'S COMPRESSION NOISE, ZEROED (2026-09-30).
///
/// Poly Haven ships the hanging lamp's mask as a JPEG: black but for the
/// bulb, plus ~3,400 texels of 1-32 levels of ringing and chroma noise round
/// the bulb's islands -- 89% of the faintest of them a single colour channel.
/// Times the bulb's drive, a stray 2/255 of blue is a blue dot on the shade
/// (headset, 2026-09-30: "colored dots on the hanging light fixtures"). Every
/// texel above the noise -- the bulb and its anti-aliased edge, from 33 up --
/// is grey, and kept. So anything at or below an eighth of the mask's
/// brightest texel is not glow.
fn clean_emissive_mask(rgba: &mut [u8]) {
    let peak = rgba.chunks_exact(4).map(|p| p[0].max(p[1]).max(p[2])).max().unwrap_or(0);
    let floor = (peak / 8).max(2);
    for p in rgba.chunks_exact_mut(4) {
        if p[0].max(p[1]).max(p[2]) <= floor {
            p[..3].fill(0);
        }
    }
}

/// Decode a glTF image to RGBA8, returning `(pixels, width, height)`.
fn decode_image(image: &gltf::image::Data, force_opaque: bool) -> (Vec<u8>, u32, u32) {
    use gltf::image::Format;
    let (width, height) = (image.width, image.height);
    let rgba: Vec<u8> = match image.format {
        Format::R8G8B8A8 => {
            if force_opaque {
                let mut out = image.pixels.clone();
                out.chunks_exact_mut(4).for_each(|px| px[3] = 255);
                out
            } else {
                image.pixels.clone()
            }
        }
        Format::R8G8B8 => {
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for px in image.pixels.chunks_exact(3) {
                out.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            out
        }
        Format::R8 => {
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for &v in &image.pixels {
                out.extend_from_slice(&[v, v, v, 255]);
            }
            out
        }
        _ => {
            log::warn!("Unsupported glTF image format {:?}, using gray fallback", image.format);
            ([180u8, 180, 180, 255].repeat((width * height) as usize), width, height).0
        }
    };
    (rgba, width, height)
}


// THE WHOLE MIP CHAIN, averaged in linear light (both are sRGB). These
// were one level under a sampler asking for mipmaps -- the defect the
// brushes' materials and the lightmap each had first: a model seen from
// past arm's length sampled one texel of dozens, so fixtures sparkled,
// DIFFERENTLY IN EACH EYE, and an emissive mask's noise came and went as
// the head moved (headset, 2026-09-30). See `brush_pipeline::mip_chain`.
fn upload_mipped(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, px: &(Vec<u8>, u32, u32)) -> wgpu::Texture {
    let chain = crate::renderer::brush_pipeline::mip_chain(
        &crate::renderer::terrain_pipeline::TerrainImage { width: px.1, height: px.2, rgba: px.0.clone() },
        true,
    );
    let size = wgpu::Extent3d { width: px.1, height: px.2, depth_or_array_layers: 1 };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: chain.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        // COPY_SRC so a test can read a level back:
        // `a_model_texture_carries_its_whole_mip_chain_and_it_averages`.
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    for (level, img) in chain.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &img.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * img.width),
                rows_per_image: Some(img.height),
            },
            wgpu::Extent3d { width: img.width, height: img.height, depth_or_array_layers: 1 },
        );
    }
    tex
}

/// A mesh material's bind group: base colour, sampler, and the emissive mask.
///
/// Both textures live in ONE bind group because both belong to one material and
/// are created together, so the `Arc<LoadedTexture>` primitives share stays
/// consistent. It also has to be one group: mobile GPUs guarantee only four,
/// and the mesh pipeline already uses all four (camera, model, texture,
/// lightmap).
pub fn create_mesh_material_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    base: &(Vec<u8>, u32, u32),
    emissive: &(Vec<u8>, u32, u32),
) -> LoadedTexture {
    let make = |label: &str, px: &(Vec<u8>, u32, u32)| upload_mipped(device, queue, label, px);

    let base_tex = make("gltf_base_color", base);
    let emissive_tex = make("gltf_emissive", emissive);
    let view = base_tex.create_view(&Default::default());
    let emissive_view = emissive_tex.create_view(&Default::default());
    // Trilinear and anisotropic, as the brushes' materials are: the chain is
    // only used if the sampler asks for it.
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        anisotropy_clamp: 8,
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gltf_material_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&emissive_view),
            },
        ],
    });
    LoadedTexture { texture: base_tex, view, _sampler: sampler, bind_group, _direction: None, _sun_mask: None }
}

/// Upload RGBA bytes as a 2D texture and return it with a default view.
///
/// Split out of `create_texture_from_rgba` so a caller can build its own bind
/// group. The two are not interchangeable: that function also creates a
/// TWO-binding group, and passing it a layout with more entries fails
/// validation at bind group creation.
/// How many mip levels a lightmap gets.
///
/// # The lightmap had the bug the material textures already had
///
/// It was uploaded with `mip_level_count: 1` while the shared sampler asked for
/// `mipmap_filter: Linear` -- a filter with one level to choose from, which is
/// a silent no-op. That is the same defect, in the same renderer, that the
/// brush material textures were found to have: see `MipChain`'s note, which
/// calls it "ordinary minification aliasing" and records that it reads as a
/// reflection artefact and is not one.
///
/// On the lightmap it reads as a LIGHTING artefact and is not one either. A
/// distant pixel covers dozens of atlas texels and samples one, so the baked
/// term flickers with head motion. It is invisible across most of a room,
/// because minification aliasing can only be seen where the signal it is
/// sampling has a gradient -- and a baked lightmap is nearly flat over a plain
/// wall. Where it is steep is around an opening to the sky, which is why the
/// artefact was reported on "the side the sky light would be hitting from"
/// (user, 2026-09-22) and nowhere else.
///
/// # WHY THREE AND NOT THE WHOLE CHAIN
///
/// The atlas packs many charts, and a coarse mip averages across whatever sits
/// next to a chart in the atlas rather than what sits next to it on the wall.
/// The gutter is what buys the room to filter: bilinear at level L reaches
/// about 2^L base texels past an edge, so L is safe while `2^L <= GUTTER`.
/// With `brush_lightmap::GUTTER` at 4 that is L <= 2, so three levels. The two
/// constants must move together and are pinned to each other by
/// `quest_app::brush_render::the_lightmap_mip_depth_fits_the_gutter`.
///
/// Three levels is not a compromise here -- it covers a 4x linear
/// minification, and past that `lod_max_clamp` holds the sample at level 2
/// rather than letting it walk into a neighbouring chart.
pub const LIGHTMAP_MIP_LEVELS: u32 = 3;

/// Mip levels of the brush SUN MASK.
///
/// The same rule as `LIGHTMAP_MIP_LEVELS` -- level L is safe while
/// `2^L <= gutter` -- applied to the mask's own gutter, which is
/// `GUTTER * SUN_MASK_SCALE` = 16 texels, so L <= 4: five levels. Two more than
/// the lightmap because the mask is four times finer: the same surface at the
/// same distance asks for a level two deeper, and a mask held at the
/// lightmap's level 2 would alias exactly as the lightmap did before it had
/// mips. Pinned against the engine constants by
/// `quest_app::brush_render::the_sun_mask_mip_depth_fits_its_gutter`.
pub const SUN_MASK_MIP_LEVELS: u32 = 5;

/// The neutral sun-mask texel: green 0 says "not baked", so the shader shades
/// the sun from the level's static map instead -- how a brush from a bake with
/// no mask, or a mesh, which never has one, is lit.
pub const NEUTRAL_SUN_MASK: [u8; 4] = [0, 0, 0, 0];

/// The neutral stationary-lamp mask texel: every channel at the far LIT end of
/// its distance range, so a level baked before stationary lamps -- or a lamp
/// with no channel -- is simply unshadowed by it. One shading path, no branch.
pub const NEUTRAL_STATIONARY_MASK: [u8; 4] = [255, 255, 255, 255];

/// Mip levels on the stationary masks, as on the sun mask: distances average
/// into distances, so a far surface reads a smaller, still-straight edge
/// rather than aliasing across a texel grid finer than its pixels.
pub const STATIONARY_MASK_MIP_LEVELS: u32 = 4;

/// The levels an image this size can actually supply, never more than asked.
///
/// A 1x1 neutral direction map cannot produce three levels, and asking wgpu
/// for more than `floor(log2(max(w, h))) + 1` is a validation error rather than
/// a silently smaller texture.
fn mip_levels_for(width: u32, height: u32, wanted: u32) -> u32 {
    let possible = 32 - width.max(height).max(1).leading_zeros();
    wanted.min(possible).max(1)
}

/// Upload an image and its mip chain.
///
/// `srgb` picks how the chain is averaged, and it is not cosmetic: averaging
/// sRGB bytes directly darkens every level (a black-and-white checkerboard
/// averages to byte 128, which is 22% grey rather than 50%), so a wall would
/// visibly dim with distance. Alpha never goes through the transfer either
/// way -- on this lightmap it carries SKY VISIBILITY, which is a linear
/// scalar, not colour.
fn upload_rgba_texture_mipped(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba: &[u8],
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
    srgb: bool,
    wanted_levels: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    use crate::renderer::terrain_pipeline::{mip_chain, mip_chain_linear};

    let levels = mip_levels_for(width, height, wanted_levels);
    let chain = if srgb {
        mip_chain(rgba, width, height, levels)
    } else {
        mip_chain_linear(rgba, width, height, levels)
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, (data, lw, lh)) in chain.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * lw),
                rows_per_image: Some(*lh),
            },
            wgpu::Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn upload_rgba_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba: &[u8],
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// The neutral bounce-direction texel.
///
/// 128 decodes through `v * 2 - 1` to zero and the alpha says "no
/// directionality", so a lightmap with no direction map shades exactly as it
/// did before directional bounce existed. A NEUTRAL VALUE rather than a branch,
/// for the same reason the sky uploads a flat constant-band SH when a scene has
/// no panorama: one shading path, and every pre-existing test keeps its
/// original pixel values.
pub const NEUTRAL_BOUNCE_DIRECTION: [u8; 4] = [128, 128, 128, 0];

/// A baked lightmap and its companion bounce-direction map, as one bind group.
///
/// The direction map is LINEAR (`Rgba8Unorm`) while the lightmap is sRGB. They
/// are different kinds of data in the same bind group: one is colour and wants
/// the transfer curve, the other is a unit vector and an unsigned scalar, and
/// putting a vector through an sRGB decode bends every direction towards the
/// surface -- the same mistake as an sRGB normal map, which this renderer has
/// made once already.
pub fn create_lightmap_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    rgba: &[u8],
    width: u32,
    height: u32,
    direction: Option<(&[u8], u32, u32)>,
) -> LoadedTexture {
    create_lightmap_texture_with_sun(
        device, queue, layout, LightmapLight::Srgb8(rgba), width, height, direction, None,
    )
}

/// The light a lightmap carries, as the bake stored it.
#[derive(Clone, Copy)]
pub enum LightmapLight<'a> {
    /// Tightly packed RGBA8, RGB sRGB-encoded -- the 8-bit maps, and every
    /// bake from before the 16-bit format.
    Srgb8(&'a [u8]),
    /// Tightly packed RGBA as f32: RGB LINEAR light in the engine's units,
    /// unclipped, A a 0..1 fraction. Uploaded as half floats. See
    /// `space_soup_engine::lightmaps::BRUSH_LIGHTMAP_RANGE` for why light
    /// needs more than 8 bits.
    Linear(&'a [f32]),
}

/// `create_lightmap_texture`, with the brush sun-visibility mask as RGBA
/// (red = visibility, green = baked). Stored as two channels, `Rg8Unorm`: at
/// four times the lightmap's density a mask is the largest image in a level's
/// bake, and the two bytes it does not use would double it.
#[allow(clippy::too_many_arguments)]
pub fn create_lightmap_texture_with_sun(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    light: LightmapLight,
    width: u32,
    height: u32,
    direction: Option<(&[u8], u32, u32)>,
    sun_mask: Option<(&[u8], u32, u32)>,
) -> LoadedTexture {
    create_lightmap_texture_full(device, queue, layout, light, width, height, direction, sun_mask, None)
}

/// `create_lightmap_texture_with_sun`, with the STATIONARY lamps' shadow masks:
/// one RGBA8 image per two lamps, all the same size, at `STATIONARY_MASK_SCALE`
/// times the lightmap's density on its charts. `None` binds the neutral mask,
/// under which every stationary lamp is unshadowed.
#[allow(clippy::too_many_arguments)]
pub fn create_lightmap_texture_full(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    light: LightmapLight,
    width: u32,
    height: u32,
    direction: Option<(&[u8], u32, u32)>,
    sun_mask: Option<(&[u8], u32, u32)>,
    stationary: Option<(&[&[u8]], u32, u32)>,
) -> LoadedTexture {
    // The texture ONLY -- deliberately not `create_texture_from_rgba`, which
    // also builds a two-binding bind group. Handing it this three-binding
    // layout makes wgpu reject the group for having the wrong number of
    // entries, which is how this was caught rather than shipped.
    let (base_texture, base_view) = match light {
        LightmapLight::Srgb8(rgba) => upload_rgba_texture_mipped(
            device,
            queue,
            rgba,
            width,
            height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            "lightmap",
            true,
            LIGHTMAP_MIP_LEVELS,
        ),
        LightmapLight::Linear(texels) => upload_linear_f16_mipped(device, queue, texels, width, height, "lightmap_hdr"),
    };
    let (dir_rgba, dir_w, dir_h) =
        direction.unwrap_or((&NEUTRAL_BOUNCE_DIRECTION, 1, 1));
    // MIPPED TOO, and LINEAR rather than sRGB: this is a unit vector and an
    // unsigned scalar, not colour, so it must be averaged exactly as stored.
    // It shares `lightmap_sampler` with the base map, so leaving it at one
    // level would put the two halves of the same lookup on different filters.
    let (dir_tex, dir_view) = upload_rgba_texture_mipped(
        device,
        queue,
        dir_rgba,
        dir_w,
        dir_h,
        wgpu::TextureFormat::Rgba8Unorm,
        "lightmap_direction",
        false,
        LIGHTMAP_MIP_LEVELS,
    );
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("lightmap_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        // HOLD THE SAMPLE AT THE COARSEST LEVEL THE GUTTER PAYS FOR.
        //
        // Without this, a surface far enough away asks for a level that does
        // not exist, the sample clamps to the smallest one there IS, and that
        // level's texels average across whatever the atlas packed next to this
        // chart -- which is a neighbouring wall, not a neighbouring pixel.
        // `LIGHTMAP_MIP_LEVELS` says why the ceiling is where it is.
        lod_max_clamp: (LIGHTMAP_MIP_LEVELS - 1) as f32,
        ..Default::default()
    });
    let (sun_rgba, sun_w, sun_h) = sun_mask.unwrap_or((&NEUTRAL_SUN_MASK, 1, 1));
    let (sun_tex, sun_view) = upload_rg8_mipped(device, queue, sun_rgba, sun_w, sun_h, "lightmap_sun_mask");
    // Its own sampler only for the deeper mip clamp -- see `SUN_MASK_MIP_LEVELS`.
    let sun_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("lightmap_sun_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        lod_max_clamp: (SUN_MASK_MIP_LEVELS - 1) as f32,
        ..Default::default()
    });
    let neutral: [&[u8]; 1] = [&NEUTRAL_STATIONARY_MASK];
    let (st_layers, st_w, st_h) = stationary.unwrap_or((&neutral, 1, 1));
    let (st_tex, st_view) = upload_rgba8_array_mipped(device, queue, st_layers, st_w, st_h, "lightmap_stationary");
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lightmap_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&base_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&dir_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&sun_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&sun_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&st_view),
            },
        ],
    });
    LoadedTexture {
        texture: base_texture,
        view: base_view,
        _sampler: sampler,
        bind_group,
        _direction: Some(dir_tex),
        _sun_mask: Some((sun_tex, sun_sampler, st_tex)),
    }
}

/// Upload linear RGBA f32 light as a mipped `Rgba16Float`.
///
/// Half floats hold the engine's light units from a dark room's 0.001 to far
/// past anything a lamp bakes, at a tenth of a percent -- the precision the
/// 8-bit sRGB form lost exactly where eye adaptation lifts a room back up.
/// The chain is averaged in linear light, as light must be. Filterable on
/// every GPU this renderer targets; a texel costs 8 bytes against 4.
fn upload_linear_f16_mipped(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texels: &[f32],
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let levels = mip_levels_for(width, height, LIGHTMAP_MIP_LEVELS);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_lightmap_mips(queue, &texture, &lightmap_mips_f16(texels, width, height));
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// A linear light map's half-float mip chain, as [`upload_linear_f16_mipped`]
/// uploads it: each level's bytes and size. Pure CPU, so a worker thread can
/// make it (the time of day relights the brush atlas off the render thread).
pub fn lightmap_mips_f16(texels: &[f32], width: u32, height: u32) -> Vec<(Vec<u8>, u32, u32)> {
    let levels = mip_levels_for(width, height, LIGHTMAP_MIP_LEVELS);
    let mut out = Vec::with_capacity(levels as usize);
    let mut level: Vec<f32> = texels.to_vec();
    let (mut lw, mut lh) = (width, height);
    for mip in 0..levels {
        let half: Vec<u8> = level
            .iter()
            .flat_map(|v| crate::renderer::sky::f32_to_f16(*v).to_le_bytes())
            .collect();
        out.push((half, lw, lh));
        if mip + 1 == levels {
            break;
        }
        // 2x2 box, clamped at an odd edge.
        let (dw, dh) = ((lw / 2).max(1), (lh / 2).max(1));
        let mut next = vec![0f32; (dw * dh * 4) as usize];
        for y in 0..dh {
            for x in 0..dw {
                for c in 0..4 {
                    let mut sum = 0.0;
                    for (sx, sy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (px, py) = ((2 * x + sx).min(lw - 1), (2 * y + sy).min(lh - 1));
                        sum += level[((py * lw + px) * 4 + c) as usize];
                    }
                    next[((y * dw + x) * 4 + c) as usize] = sum * 0.25;
                }
            }
        }
        level = next;
        lw = dw;
        lh = dh;
    }
    out
}

/// Write a chain [`lightmap_mips_f16`] made into a light map's texture, in
/// place: the bind groups that hold it keep working.
pub fn write_lightmap_mips(queue: &wgpu::Queue, texture: &wgpu::Texture, chain: &[(Vec<u8>, u32, u32)]) {
    for (mip, (half, lw, lh)) in chain.iter().enumerate() {
        if mip as u32 >= texture.mip_level_count() {
            break;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            half,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8 * lw), rows_per_image: Some(*lh) },
            wgpu::Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
        );
    }
}

impl LoadedTexture {
    /// Wraps an existing texture (e.g. a UI panel render target) in the
    /// sampler + bind group the mesh pipeline expects.
    pub fn from_texture(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        texture: wgpu::Texture,
    ) -> Self {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("panel_quad_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("panel_quad_texture_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        Self {
            texture,
            view,
            _sampler: sampler,
            bind_group,
            _direction: None,
            _sun_mask: None,
        }
    }

    /// This light map's bind group again, with the NEUTRAL sun mask in place
    /// of its baked one: every brush then reads the sun's level shadow from
    /// the static map -- what a sun that has moved off the baked direction
    /// needs. Shares every other texture. `None` for a map built without the
    /// lightmap layout's parts.
    pub fn sunless_bind_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> Option<(wgpu::BindGroup, wgpu::Texture)> {
        let dir = self._direction.as_ref()?;
        let (_, sun_sampler, st_tex) = self._sun_mask.as_ref()?;
        let (neutral, neutral_view) = upload_rg8_mipped(device, queue, &NEUTRAL_SUN_MASK, 1, 1, "lightmap_sun_mask_neutral");
        let dir_view = dir.create_view(&wgpu::TextureViewDescriptor::default());
        let st_view = st_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lightmap_bg_sunless"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self._sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&dir_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&neutral_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(sun_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&st_view) },
            ],
        });
        Some((bind_group, neutral))
    }
}

/// Upload an RGBA image's red and green channels as a mipped `Rg8Unorm`.
///
/// The chain is averaged linearly: both channels are fractions, not colour.
/// RGBA8 images as the layers of one mipped 2D array. Every layer must be
/// `width` x `height`; mips are generated per layer on the CPU.
fn upload_rgba8_array_mipped(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layers: &[&[u8]],
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    use crate::renderer::terrain_pipeline::mip_chain_linear;
    let levels = mip_levels_for(width, height, STATIONARY_MASK_MIP_LEVELS);
    let count = layers.len().max(1) as u32;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: count },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (layer, rgba) in layers.iter().enumerate() {
        for (level, (data, lw, lh)) in mip_chain_linear(rgba, width, height, levels).iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * lw),
                    rows_per_image: Some(*lh),
                },
                wgpu::Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
            );
        }
    }
    // ALWAYS an array view, even of one layer: the layout says so.
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

/// A sun-mask texel as the GPU holds it: two channels, to halve the mask's
/// memory. Red is the baker's signed distance as it stands. Green folds the
/// baker's other two channels into one: under a half, not baked (the neutral
/// mask); from a half up, baked, with the sun's penumbra half-width -- the
/// baker's blue -- across the upper half. The brush shader reads it back that
/// way. The upload used to keep red and green and drop blue, so the penumbra
/// the baker measured never reached the shader and every sun shadow was drawn
/// razor sharp.
pub(crate) fn pack_sun_mask_texel(p: &[u8]) -> [u8; 2] {
    [p[0], if p[1] >= 128 { 128 + p[2] / 2 } else { 0 }]
}

fn upload_rg8_mipped(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba: &[u8],
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    use crate::renderer::terrain_pipeline::mip_chain_linear;
    let levels = mip_levels_for(width, height, SUN_MASK_MIP_LEVELS);
    let chain = mip_chain_linear(rgba, width, height, levels);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, (data, lw, lh)) in chain.iter().enumerate() {
        let rg: Vec<u8> = data.chunks_exact(4).flat_map(pack_sun_mask_texel).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rg,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(2 * lw),
                rows_per_image: Some(*lh),
            },
            wgpu::Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub fn create_texture_from_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> LoadedTexture {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gltf_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        size,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("gltf_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gltf_texture_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    LoadedTexture {
        texture,
        view,
        _sampler: sampler,
        bind_group,
        _direction: None,
        _sun_mask: None,
    }
}

#[cfg(test)]
mod transmission_tests {
    /// What `load_primitive_texture` decides about opacity, in isolation.
    ///
    /// The decision is the whole bug: a transmissive material is formally
    /// OPAQUE in glTF -- transmission and alpha coverage are separate
    /// mechanisms -- so a test of `alpha_mode` alone says the glass is opaque
    /// and is perfectly correct while the lamp's bulb stays invisible behind it.
    fn blocks_light(alpha_mode_is_blend: bool, transmission: f32, base_alpha: f32) -> f32 {
        let force_opaque = !alpha_mode_is_blend && transmission <= 0.0;
        if force_opaque { 1.0 } else { base_alpha * (1.0 - transmission) }
    }

    #[test]
    fn clear_glass_blocks_nothing() {
        // The lamp's envelope: transmission 1.0, no alphaMode. It must not
        // stand in front of the bulb.
        assert_eq!(blocks_light(false, 1.0, 1.0), 0.0);
    }

    #[test]
    fn an_ordinary_opaque_material_is_untouched() {
        // Everything without transmission keeps the old behaviour exactly --
        // this must not quietly make the whole project translucent.
        assert_eq!(blocks_light(false, 0.0, 1.0), 1.0);
        assert_eq!(blocks_light(false, 0.0, 0.25), 1.0, "alpha is ignored when opaque");
    }

    #[test]
    fn partial_transmission_partly_blocks() {
        assert!((blocks_light(false, 0.4, 1.0) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn a_blended_material_still_uses_its_own_alpha() {
        assert!((blocks_light(true, 0.0, 0.3) - 0.3).abs() < 1e-6);
    }
}

#[cfg(test)]
mod sun_mask_packing_tests {
    use super::*;

    /// The baker writes the sun's penumbra in BLUE, and the GPU holds two
    /// channels. It must arrive -- it used to be dropped -- and decode, as
    /// the brush shader decodes green, to what the baker wrote.
    #[test]
    fn the_sun_penumbra_survives_the_two_channel_upload() {
        let range = crate::renderer::brush_pipeline::SUN_MASK_DISTANCE_TEXELS;
        for blue in [0u8, 1, 64, 128, 200, 255] {
            let [r, g] = pack_sun_mask_texel(&[77, 255, blue, 255]);
            assert_eq!(r, 77, "the distance must pass through untouched");
            let g = g as f32 / 255.0;
            assert!(g > 0.25, "a baked texel read as unbaked");
            let decoded = (g - 0.5).max(0.0) * 2.0 * range;
            let written = blue as f32 / 255.0 * range;
            assert!((decoded - written).abs() < 0.04, "blue {blue}: baked {written}, the shader reads {decoded}");
        }
        let [_, g] = pack_sun_mask_texel(&NEUTRAL_SUN_MASK);
        assert!((g as f32 / 255.0) < 0.25, "the neutral mask read as baked");
    }
}

#[cfg(test)]
mod lightmap_mip_tests {
    use super::*;

    /// The defect this whole change exists to remove: a texture with one level
    /// under a sampler asking for `Linear` mipmap filtering, which silently
    /// does nothing and leaves the baked term aliasing at distance.
    #[test]
    fn a_lightmap_is_uploaded_with_the_levels_its_sampler_asks_for() {
        let Some((device, queue)) =
            crate::renderer::terrain_pipeline::tests::headless_gpu()
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = crate::renderer::pipeline::lightmap_bind_group_layout(&device);
        let rgba = vec![200u8; 64 * 64 * 4];
        let lm = create_lightmap_texture(&device, &queue, &layout, &rgba, 64, 64, None);
        assert_eq!(
            lm.texture.mip_level_count(),
            LIGHTMAP_MIP_LEVELS,
            "the lightmap is back to a single level while `lightmap_sampler` \
             still asks for Linear mipmap filtering -- which is a silent no-op \
             and is exactly the bug the material textures had",
        );
    }

    /// A 1x1 neutral direction map cannot supply three levels, and asking wgpu
    /// for more than an image can give is a validation error, not a smaller
    /// texture.
    #[test]
    fn an_image_is_never_asked_for_more_levels_than_it_has() {
        assert_eq!(mip_levels_for(1, 1, LIGHTMAP_MIP_LEVELS), 1);
        assert_eq!(mip_levels_for(2, 1, LIGHTMAP_MIP_LEVELS), 2);
        assert_eq!(mip_levels_for(4, 4, LIGHTMAP_MIP_LEVELS), 3);
        assert_eq!(mip_levels_for(1024, 1024, LIGHTMAP_MIP_LEVELS), 3);
    }

    /// THE LIGHTMAP IS sRGB AND ITS ALPHA IS NOT.
    ///
    /// Averaging sRGB bytes directly darkens every level -- a black-and-white
    /// checkerboard averages to byte 128, which is 22% grey rather than 50% --
    /// so a wall would visibly dim with distance. Alpha carries SKY VISIBILITY
    /// here, a linear scalar, and must be averaged as stored. One chain, two
    /// rules, and getting either backwards is invisible until it ships.
    #[test]
    fn the_chain_averages_colour_in_light_and_alpha_as_stored() {
        use crate::renderer::terrain_pipeline::mip_chain;
        // 2x2: two black texels and two white ones, alpha 0 and 255 likewise.
        let rgba: Vec<u8> = vec![
            0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
        ];
        let chain = mip_chain(&rgba, 2, 2, 2);
        let (level1, w, h) = &chain[1];
        assert_eq!((*w, *h), (1, 1));
        let grey = level1[0];
        assert!(
            (186..=190).contains(&grey),
            "half black and half white should average to mid GREY (sRGB ~188), \
             not to byte 128 which is 22% of the light: got {grey}",
        );
        assert_eq!(
            level1[3], 128,
            "alpha is a linear scalar and must average to 128, not through the \
             sRGB transfer",
        );
    }
}

#[cfg(test)]
mod model_texture_tests {
    use super::*;

    /// The bulb and its anti-aliased edge stay; compression noise below an
    /// eighth of the brightest texel -- coloured or not -- goes, relative to
    /// the mask's own peak.
    #[test]
    fn an_emissive_mask_keeps_its_bulb_and_drops_its_noise() {
        let mut rgba = vec![
            255, 250, 240, 255, // bulb
            60, 60, 60, 255, // its soft edge
            0, 0, 2, 255, // JPEG chroma noise
            20, 5, 30, 255, // ringing by the bulb
            0, 0, 0, 255,
        ];
        clean_emissive_mask(&mut rgba);
        assert_eq!(&rgba[0..4], &[255, 250, 240, 255]);
        assert_eq!(&rgba[4..8], &[60, 60, 60, 255]);
        assert_eq!(&rgba[8..12], &[0, 0, 0, 255]);
        assert_eq!(&rgba[12..16], &[0, 0, 0, 255]);
        // A dimmer mask's floor is an eighth of ITS peak.
        let mut dim = vec![64, 64, 64, 255, 9, 9, 9, 255, 8, 8, 8, 255];
        clean_emissive_mask(&mut dim);
        assert_eq!(dim, vec![64, 64, 64, 255, 9, 9, 9, 255, 0, 0, 0, 255]);
    }

    /// A model texture is uploaded with every level, and level 1 is the mean
    /// of the four below in LIGHT: a black/white checker comes back as the
    /// grey that is half the light (188), not half the bytes (128).
    #[test]
    fn a_model_texture_carries_its_whole_mip_chain_and_it_averages() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut rgba = Vec::new();
        for y in 0..4u32 {
            for x in 0..4u32 {
                let v = if (x + y) % 2 == 0 { 255 } else { 0 };
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let tex = upload_mipped(&device, &queue, "test", &(rgba, 4, 4));
        assert_eq!(tex.mip_level_count(), 3);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256 * 2,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 1, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(2) },
            },
            wgpu::Extent3d { width: 2, height: 2, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let bytes = buffer.slice(..).get_mapped_range().unwrap();
        for (x, y) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
            let r = bytes[y * 256 + x * 4];
            assert!((187..=189).contains(&r), "level 1 texel {x},{y}: {r}");
        }
    }
}
