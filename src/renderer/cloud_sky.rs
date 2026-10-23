//! THE VOLUMETRIC CLOUDS ON THE GPU (2026-10-20): the layers of
//! `space_soup_sky::cloud_layers` marched through, a slice of a sky map a
//! frame, and their shadow on the ground.
//!
//! The sky is at infinity: a cloud 1.5 km up shifts 0.002 deg between the
//! eyes. So its light is worked out once per direction for both eyes, into
//! a `MAP_WIDTH` x `MAP_HEIGHT` map over the upper hemisphere (azimuth by
//! elevation, rows packed toward the horizon), `MAP_SLICES` frames to a whole
//! map, and the sky shader reads it with two fetches: the last whole map
//! blending into the newest as the next is built, so nothing ever steps.
//! Its cost is fixed by the map, not by the eyes or the screen
//! (`docs/engine/sky-realism-research.md` 4.5).
//!
//! THE CLOUDS DRIFT EVERY FRAME (2026-10-21). A map is drawn of the sky at
//! one moment and from one eye (`CloudSky::encode_at`'s `map_time`), and
//! keeps beside its light the height of what it shows (`alts`). The sky
//! shader carries each map from then and there to here and now: every part
//! of a cloud on its own height's wind (`advected_uv`), so tops and bases
//! shear as real ones do, and walking under them moves them as parallax.
//! Before, the sky moved only by a dissolve from one map to the next. A map
//! now has to follow only what the wind cannot: shapes growing, merging and
//! thinning, and the light. That changes slowly, so maps are drawn half as
//! often as before, and the two blended are both where the clouds are now,
//! so the change reads as a morph rather than a double image. SpaceWarp's
//! sky pixels move the same way (`sky_motion`).
//!
//! The ground's shadow map (`UniformBuffer::cloud_shadow`, read per pixel by
//! `cloud_sun_shadow`) is drawn from the same density every frame, so the
//! shadows sweeping over the ground are the clouds overhead.
//!
//! Where each cloud is comes from its layer's population
//! (`space_soup_sky::cloud_population`): the live clouds are splatted into
//! an envelope map per layer once a second of cloud time (`set_population`,
//! `fs_splat`), and the march reads it where it read the weather.

pub use space_soup_sky::{cloud_layers, cloud_population, clouds};
use space_soup_sky::cloud_layers::{self as cl, CloudLayer, CloudNoise, LAYERS, NOISE_SIZE};
use space_soup_sky::cloud_population::{self as cp, CloudPopulation, ENVELOPE_SIZE, POPULATED};
use space_soup_sky::clouds::{self as c, CloudField, CLOUD_TILE};
use wgpu::*;

/// The sky map: 0.18 deg a texel round the horizon, about 4.5 of the
/// Quest 3's pixels. At 0.35 deg (1024 x 384) the headset showed the
/// texels' squares at cloud edges (2026-10-20); the sky shader reads it
/// through a cubic B-spline, so no texel's square is drawn.
pub const MAP_WIDTH: u32 = 2048;
pub const MAP_HEIGHT: u32 = 768;
/// Frames to a whole map: 8 rows, 16 k rays a frame; a map every 1.3 s at
/// 72 Hz (2.7 s under SpaceWarp's 36). Half the rays of the 48 slices
/// before the maps were carried by the wind: the drift is drawn every frame
/// anyway, and a cumulus changes shape over minutes.
pub const MAP_SLICES: u32 = 96;
/// A texel with no cloud keeps the height of the lowest layer that is up,
/// at this weight against what it holds: a filtered read across a cloud's
/// edge then blends heights, never a height with nothing (which would carry
/// the edge on the wind of half its height).
pub const ALT_FILL_WEIGHT: f32 = 0.02;
/// Rows are packed toward the horizon: elevation = 90 deg * v^MAP_PACK. At
/// 1.5 the rows near the horizon were three times finer than the map's
/// columns there and those near the zenith three times coarser; 1.15 keeps
/// both within about 1.5 (2026-10-21; it did not cause the overhead smear,
/// the shear's lean did -- `SHEAR_LEAN_MAX`). Not yet seen on the headset.
pub const MAP_PACK: f32 = 1.15;
/// THE MAPS ACCUMULATE (2026-10-21). Each map's samples sit at their own
/// fraction of the step (`HISTORY_JITTER` apart, every texel alike, so
/// neighbouring rays still read neighbouring noise), and a map built from a
/// slice keeps this share of its own march and the rest of the newest map,
/// carried to its moment and eye on the winds (`advected_uv`). The fixed
/// sample planes of one march drew a wood grain through every cloud; across
/// maps the planes move, so the grain averages out over about 1 / share maps
/// (a few seconds), long before a shape changes.
///
/// 0.3 AGAIN (2026-10-22 evening). At 1 (no history) the ghost copies of
/// the edges went (`maps_accumulated_under_a_moving_sky_hold_no_ghost_edges`:
/// 8.9% -> 0.8%), but with nothing averaging the sample planes, and the
/// jitter then held at one fraction for every map, the clouds came out in
/// flat stepped bands on the headset (user, video 21:29: "horrible"). Until
/// the ghosts are fixed another way, the map accumulates as before.
pub const HISTORY_NEW: f32 = 0.3;
/// Where the old and new maps disagree by this much of the sky showing (a
/// cloud born, or uncovered by the eye moving), the new march is taken
/// alone (from the first to the second): history is for the grain, not for
/// keeping a cloud. Before 2026-10-22 this was [0.2, 0.5]: a cloud edge
/// whose shape had changed between the maps kept 70% of the old edge, a
/// ghost fringe (`maps_accumulated_under_a_moving_sky_hold_no_ghost_edges`:
/// 8.9% of the edge texels; [0.03, 0.1]: 0.94%), and the grain the history
/// is for is under 0.008 of alpha since the refined march, so it still
/// averages (`maps_built_one_after_another_settle_on_a_finer_march`).
/// `HISTORY_REJECT_WIDE` is the old pair, for comparing
/// (`Levers::cloud_history_tight`).
pub const HISTORY_REJECT: [f32; 2] = [0.03, 0.1];
pub const HISTORY_REJECT_WIDE: [f32; 2] = [0.2, 0.5];
/// BUT NOT AT THE HORIZON: there a step of the march spans kilometres, one
/// map's grain is larger than the tight pair, and letting it go there
/// shows each map's own sample fraction, the horizon's flicker line (a
/// still sky's horizon, map to map: 0.0045 wide, 0.0085 tight; 0.0113 with
/// no history, the line the user saw). Far clouds move a pixel or less
/// between maps, so they leave no ghost to cut. Below `HISTORY_HORIZON_UP`
/// .0 (the sine of the elevation) the wide pair holds, above .1 the tight.
pub const HISTORY_HORIZON_UP: [f32; 2] = [0.06, 0.15];
/// AND AS A SHARE OF THE CLOUD THERE: a ghost's soft rim over clear sky
/// differs from the new march by under the tight pair (0.05 of the sky
/// showing), so each older map left its cloud's outline in the clear sky,
/// stacked like contour lines (headset video 2026-10-23 00:55). Taken over
/// the more cloud of the two (at least `HISTORY_SHARE_FLOOR`), that rim
/// differs by all of itself, while the grain inside a cloud (0.008 of 0.5)
/// is under 2%. From the first share to the second the new march is taken
/// alone; above the horizon only, as the tight pair. Faint rims (over 2% of
/// cloud where a fresh map has none, `maps_accumulated_..._no_ghost_edges`)
/// 6.9% -> 1.6% of the clear sky; grain and the horizon unchanged.
pub const HISTORY_REJECT_SHARE: [f32; 2] = [0.15, 0.35];
pub const HISTORY_SHARE_FLOOR: f32 = 0.02;
/// The history is carried a second time by the altitude the old map holds
/// (`hist_carry`), and that carry kept where the old map agrees with itself
/// there to this share of the altitude.
pub const HISTORY_REFETCH: bool = true;
pub const HISTORY_ALT_AGREE: f32 = 0.1;
/// One map's sample fraction to the next: the golden ratio's, so any run of
/// maps spreads its samples evenly through the step.
pub const HISTORY_JITTER: f32 = 0.618_034;
/// THE MAP IS PREFILTERED so the display's cubic B-spline (`sky_cloud_map`,
/// four bilinear taps) passes through its samples instead of blurring them:
/// the B-spline's inverse, sqrt(3) (sqrt(3) - 2)^|k|, cut at two texels and
/// renormalised, per axis. The B-spline alone took a cumulus edge's
/// response at half Nyquist to 0.66 (sigma 0.58 texel, about 2.6 headset
/// pixels; research `realistic-sky-research-2026-10-21` §3.1); with this
/// 0.98. Run once per map row, a band behind the march.
pub const PREFILTER: [f32; 3] = [1.645, -0.441, 0.118];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudSkyUniform {
    /// World x, altitude, world z of the eye the map is seen from; w the
    /// scale the map's light is stored at (`CloudSky::encode` sets it).
    pub eye: [f32; 4],
    /// Toward the light (the sun, or the moon after it), unit; w the
    /// aerosol's extinction at the ground per metre (`with_aerosol`).
    pub light_dir: [f32; 4],
    /// The light's irradiance; w the map's rows.
    pub light: [f32; 4],
    /// The sky's radiance on a cloud from above; w the map's width.
    pub amb_sky: [f32; 4],
    /// The ground's from below; w the shadow map's mean.
    pub amb_ground: [f32; 4],
    /// The shadow map's centre x, z, extent, and the ground's altitude.
    pub shadow: [f32; 4],
    pub layers: [[[f32; 4]; 12]; LAYERS],
}

impl CloudSkyUniform {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layers: &[CloudLayer; LAYERS],
        eye: [f32; 3],
        light_dir: [f32; 3],
        light: [f32; 3],
        amb_sky: [f32; 3],
        amb_ground: [f32; 3],
        shadow_centre: [f32; 2],
        shadow_extent: f32,
        shadow_mean: f32,
    ) -> Self {
        let len = (light_dir[0] * light_dir[0] + light_dir[1] * light_dir[1] + light_dir[2] * light_dir[2]).sqrt().max(1e-6);
        Self {
            eye: [eye[0], eye[1], eye[2], 0.0],
            light_dir: [light_dir[0] / len, light_dir[1] / len, light_dir[2] / len, space_soup_sky::time_of_day::AEROSOL_PER_M],
            light: [light[0], light[1], light[2], MAP_HEIGHT as f32],
            amb_sky: [amb_sky[0], amb_sky[1], amb_sky[2], MAP_WIDTH as f32],
            amb_ground: [amb_ground[0], amb_ground[1], amb_ground[2], shadow_mean],
            shadow: [shadow_centre[0], shadow_centre[1], shadow_extent, eye[1]],
            layers: std::array::from_fn(|k| layers[k].gpu()),
        }
    }

    /// The haze the clouds are seen through: the aerosol's extinction at the
    /// ground per metre (`SkySnapshot::aerosol_per_m`; turbidity 1's without).
    pub fn with_aerosol(mut self, per_m: f32) -> Self {
        self.light_dir[3] = per_m;
        self
    }
}

/// The direction a map coordinate looks along (`cloud_map_dir`'s twin).
pub fn map_dir(uv: [f32; 2]) -> [f32; 3] {
    let el = std::f32::consts::FRAC_PI_2 * uv[1].clamp(0.0, 1.0).powf(MAP_PACK);
    let az = (uv[0] - 0.5) * std::f32::consts::TAU;
    [az.sin() * el.cos(), el.sin(), az.cos() * el.cos()]
}

/// The map coordinate a direction is held at (the sky shader's lookup).
pub fn map_uv(d: [f32; 3]) -> [f32; 2] {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let el = (d[1] / len).clamp(0.0, 1.0).asin();
    [d[0].atan2(d[2]) / std::f32::consts::TAU + 0.5, (el / std::f32::consts::FRAC_PI_2).powf(1.0 / MAP_PACK)]
}

/// Metres from an eye at `eye_alt` along a direction rising `mu` to the
/// altitude `alt`, over the curved earth (`cloud_flow_distance`'s twin).
pub fn flow_distance(eye_alt: f32, alt: f32, mu: f32) -> f32 {
    let earth = c::EARTH_RADIUS_M;
    let r = earth + eye_alt;
    let k = (alt - eye_alt).max(0.0) * (2.0 * earth + alt + eye_alt);
    let rm = r * mu.max(0.0);
    k / ((rm * rm + k).sqrt() + rm).max(1e-3)
}

/// The wind at altitude `alt` (x, z, m/s): between the layers' own
/// (`flow_winds`, by height), held past the lowest and the highest
/// (`cloud_flow_wind`'s twin).
pub fn flow_wind(alt: f32, winds: &[[f32; 4]; LAYERS]) -> [f32; 2] {
    let mut v = [winds[0][1], winds[0][2]];
    for k in 1..LAYERS {
        let t = ((alt - winds[k - 1][0]) / (winds[k][0] - winds[k - 1][0]).max(1.0)).clamp(0.0, 1.0);
        v = [v[0] + (winds[k][1] - v[0]) * t, v[1] + (winds[k][2] - v[1]) * t];
    }
    v
}

/// Each layer's middle height, its wind (x, z, m/s) and its index, lowest
/// first. The map stores the wind of what each texel shows (`fs_map`, alts
/// .ba, by `layer_wind` on the index); the height's wind between these is
/// only the CPU twin's fallback.
pub fn flow_winds(p: &space_soup_sky::clouds::CloudParams, layers: &[CloudLayer; LAYERS]) -> [[f32; 4]; LAYERS] {
    let mut w: [[f32; 4]; LAYERS] = std::array::from_fn(|k| {
        let (dir, v) = cl::layer_wind(p, k);
        [0.5 * (layers[k].base_m + layers[k].top_m), (dir[0] * v) as f32, (dir[1] * v) as f32, k as f32]
    });
    w.sort_by(|a, b| a[0].total_cmp(&b[0]));
    w
}

/// WHERE A MAP HOLDS WHAT IS SEEN ALONG `dir` NOW: the cloud there, at
/// `alt` (the map's height), as it was `age` seconds ago on its wind `v`
/// (the map's alts .ba) and seen from where the map's eye was (`eye_delta`
/// = the eye now less the map's). The sky shader's lookup (`sky_clouds`).
pub fn advected_uv(dir: [f32; 3], alt: f32, eye_alt: f32, v: [f32; 2], eye_delta: [f32; 3], age: f32) -> [f32; 2] {
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt().max(1e-6);
    let d = dir.map(|v| v / len);
    if alt <= 0.0 || d[1] <= 0.0 {
        return map_uv(d);
    }
    let t = flow_distance(eye_alt, alt, d[1]);
    map_uv([d[0] * t + eye_delta[0] - v[0] * age, d[1] * t + eye_delta[1], d[2] * t + eye_delta[2] - v[1] * age])
}

/// The lookup and the wind the sky shader and SpaceWarp's sky share
/// (`map_uv`, `flow_distance`, `flow_wind`, line for line).
pub fn cloud_flow_wgsl() -> String {
    CLOUD_FLOW_WGSL.replace("{EARTH}", &format!("{:.1}", c::EARTH_RADIUS_M)).replace("{PACK}", &format!("{MAP_PACK:?}"))
}

const CLOUD_FLOW_WGSL: &str = r#"
const CLOUD_FLOW_EARTH: f32 = {EARTH};
const CLOUD_FLOW_PACK: f32 = {PACK};

fn cloud_map_uv(d: vec3<f32>) -> vec2<f32> {
    let n = normalize(d);
    let el = asin(clamp(n.y, 0.0, 1.0));
    return vec2<f32>(atan2(n.x, n.z) / 6.28318531 + 0.5, pow(el / 1.57079633, 1.0 / CLOUD_FLOW_PACK));
}

fn cloud_flow_distance(eye_alt: f32, alt: f32, mu: f32) -> f32 {
    let r = CLOUD_FLOW_EARTH + eye_alt;
    let k = max(alt - eye_alt, 0.0) * (2.0 * CLOUD_FLOW_EARTH + alt + eye_alt);
    let rm = r * max(mu, 0.0);
    return k / max(sqrt(rm * rm + k) + rm, 1e-3);
}

fn cloud_flow_wind(alt: f32, w0: vec4<f32>, w1: vec4<f32>, w2: vec4<f32>, w3: vec4<f32>) -> vec2<f32> {
    var v = w0.yz;
    v = mix(v, w1.yz, clamp((alt - w0.x) / max(w1.x - w0.x, 1.0), 0.0, 1.0));
    v = mix(v, w2.yz, clamp((alt - w1.x) / max(w2.x - w1.x, 1.0), 0.0, 1.0));
    v = mix(v, w3.yz, clamp((alt - w2.x) / max(w3.x - w2.x, 1.0), 0.0, 1.0));
    return v;
}
"#;

/// Mips of each envelope, down to 1 texel: the march reads the one its
/// sample's footprint spans. Read at the top level everywhere, the 2048 x 768
/// map's slice cost 4.1 ms on the Quest (bench 2026-10-21, against well under
/// 0.4 ms before the populations): rays toward the horizon stride across an
/// 8 MB map, a cache miss a sample.
pub const ENVELOPE_MIPS: u32 = ENVELOPE_SIZE.ilog2() + 1;

pub struct CloudSky {
    noise: Texture,
    weather: Texture,
    /// The populated layers' envelopes (`cloud_population`), one array
    /// layer each, and what draws them.
    envelope: Texture,
    envelope_layers: [TextureView; POPULATED],
    envelope_sampler: Sampler,
    splat_pipeline: RenderPipeline,
    /// [layer][mip] views and the passes that average each mip from the one
    /// above it.
    envelope_mip_views: Vec<Vec<TextureView>>,
    mip_pipeline: RenderPipeline,
    mip_bind_groups: Vec<Vec<BindGroup>>,
    splat_uniform: Buffer,
    splat_bind_group: BindGroup,
    splats: Option<Buffer>,
    splat_ranges: [std::ops::Range<u32>; POPULATED],
    /// The population generation the envelopes hold, and whether a newer
    /// one is waiting to be drawn.
    envelope_generation: u64,
    envelope_pending: bool,
    /// Whether the last `encode_at` splatted the envelopes (`CLOUDSUBMIT`).
    splatted: bool,
    maps: Texture,
    map_layers: [TextureView; 3],
    array_view: TextureView,
    /// The march's own texels, before the prefilter (`PREFILTER`).
    raw: Texture,
    raw_view: TextureView,
    prefilter_pipeline: RenderPipeline,
    prefilter_group: BindGroup,
    /// Beside each map, the height of what it shows (R16F, metres): what
    /// the wind it is carried on is read by.
    alts: Texture,
    alt_layers: [TextureView; 3],
    alt_view: TextureView,
    sampler: Sampler,
    uniform: Buffer,
    layout: BindGroupLayout,
    bind_group: BindGroup,
    /// The ground's shadow is of the clouds NOW, while a map holds them at
    /// its own moment: a uniform and a bind group of its own.
    shadow_uniform: Buffer,
    shadow_bind_group: BindGroup,
    map_pipeline: RenderPipeline,
    shadow_pipeline: RenderPipeline,
    /// What a map slice accumulates over (`HISTORY_NEW`): the carry from the
    /// newest whole map to the one being built, and a bind group per map
    /// to read (each map's own layer views, so the layer written is never
    /// the one read).
    history_uniform: Buffer,
    history_groups: [BindGroup; 3],
    /// Where history is let go above the horizon and at it (`HISTORY_REJECT`,
    /// `HISTORY_REJECT_WIDE`), or what `set_history_reject` gave.
    history_reject: [f32; 4],
    /// `HISTORY_REJECT_SHARE` and its floor, or zeros (off).
    history_share: [f32; 4],
    /// The winds the maps are carried on (`flow_winds`), and how many maps
    /// have been begun (the jitter's index).
    winds: [[f32; 4]; LAYERS],
    maps_begun: u32,
    /// The map shown last, the one shown now, and the one being built.
    prev: u32,
    cur: u32,
    build: u32,
    next_row: u32,
    /// No map drawn yet: the first is drawn whole.
    fresh: bool,
    /// Each map's light is stored over the light it was drawn under: a
    /// night's radiances (1e-8) fell among half floats' subnormals and the
    /// clouds came out banded and fringed red.
    scales: [f32; 3],
    /// The cloud time each map holds the sky at, and the eye it is seen from.
    times: [f64; 3],
    eyes: [[f32; 3]; 3],
    /// Each map's uniform as its first slice took it: every slice of one map
    /// draws the same light, ambient and eye. A light refreshed mid-build
    /// (the time-of-day snapshot, about once a second) split a map into
    /// bands of rows lit two ways (research 2026-10-21, cause (c)).
    map_uniforms: [CloudSkyUniform; 3],
    /// Seconds a map takes to build, as measured, and when the last began.
    period: f64,
    last_start: Option<f64>,
}

fn weather_texture(device: &Device, queue: &Queue, field: &CloudField) -> Texture {
    let t = device.create_texture(&TextureDescriptor {
        label: Some("cloud_weather"),
        size: Extent3d { width: CLOUD_TILE, height: CLOUD_TILE, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
        &field.rgba_bytes(),
        TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(CLOUD_TILE * 4), rows_per_image: Some(CLOUD_TILE) },
        Extent3d { width: CLOUD_TILE, height: CLOUD_TILE, depth_or_array_layers: 1 },
    );
    t
}

impl CloudSky {
    pub fn new(device: &Device, queue: &Queue, field: &CloudField, noise: &CloudNoise) -> Self {
        Self::with_shader(device, queue, field, noise, cloud_sky_shader())
    }

    /// `new` with the marcher's source given, for measuring a variant.
    pub(crate) fn with_shader(device: &Device, queue: &Queue, field: &CloudField, noise: &CloudNoise, source: String) -> Self {
        let n = NOISE_SIZE;
        let noise_tex = device.create_texture(&TextureDescriptor {
            label: Some("cloud_noise"),
            size: Extent3d { width: n, height: n, depth_or_array_layers: n },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo { texture: &noise_tex, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            &noise.rgba_bytes(),
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(n * 4), rows_per_image: Some(n) },
            Extent3d { width: n, height: n, depth_or_array_layers: n },
        );
        let weather = weather_texture(device, queue, field);
        let maps = device.create_texture(&TextureDescriptor {
            label: Some("cloud_sky_maps"),
            size: Extent3d { width: MAP_WIDTH, height: MAP_HEIGHT, depth_or_array_layers: 3 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let map_layers = std::array::from_fn(|i| {
            maps.create_view(&TextureViewDescriptor {
                label: Some("cloud_sky_map"),
                dimension: Some(TextureViewDimension::D2),
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });
        // What the march writes, before the prefilter turns it into a map
        // (`PREFILTER`): one layer, each row read again while the next
        // slice is marched.
        let raw = device.create_texture(&TextureDescriptor {
            label: Some("cloud_sky_raw"),
            size: Extent3d { width: MAP_WIDTH, height: MAP_HEIGHT, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let raw_view = raw.create_view(&TextureViewDescriptor::default());
        let prefilter_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("cloud_prefilter_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false },
                count: None,
            }],
        });
        let prefilter_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("cloud_prefilter_bg"),
            layout: &prefilter_layout,
            entries: &[BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&raw_view) }],
        });
        let prefilter_shader = device.create_shader_module(ShaderModuleDescriptor { label: Some("cloud_prefilter"), source: ShaderSource::Wgsl(prefilter_shader().into()) });
        let prefilter_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("cloud_prefilter_layout"),
            bind_group_layouts: &[Some(&prefilter_layout)],
            immediate_size: 0,
        });
        let prefilter_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("cloud_prefilter"),
            layout: Some(&prefilter_pipeline_layout),
            vertex: VertexState { module: &prefilter_shader, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(FragmentState {
                module: &prefilter_shader,
                entry_point: Some("fs_prefilter"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState { format: TextureFormat::Rgba16Float, blend: None, write_mask: ColorWrites::ALL })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let array_view = maps.create_view(&TextureViewDescriptor {
            label: Some("cloud_sky_maps"),
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });
        let alts = device.create_texture(&TextureDescriptor {
            label: Some("cloud_sky_alts"),
            size: Extent3d { width: MAP_WIDTH, height: MAP_HEIGHT, depth_or_array_layers: 3 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let alt_layers: [TextureView; 3] = std::array::from_fn(|i| {
            alts.create_view(&TextureViewDescriptor {
                label: Some("cloud_sky_alt"),
                dimension: Some(TextureViewDimension::D2),
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });
        let alt_view = alts.create_view(&TextureViewDescriptor {
            label: Some("cloud_sky_alts"),
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_noise_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let uniform_of = |label| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<CloudSkyUniform>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let uniform = uniform_of("cloud_sky_uniform");
        let shadow_uniform = uniform_of("cloud_shadow_uniform");
        let tex = |binding, dim| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: dim, multisampled: false },
            count: None,
        };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("cloud_sky_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex(1, TextureViewDimension::D3),
                tex(2, TextureViewDimension::D2),
                BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::FRAGMENT, ty: BindingType::Sampler(SamplerBindingType::Filtering), count: None },
                tex(4, TextureViewDimension::D2Array),
                BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::FRAGMENT, ty: BindingType::Sampler(SamplerBindingType::Filtering), count: None },
            ],
        });
        let envelope = device.create_texture(&TextureDescriptor {
            label: Some("cloud_envelope"),
            size: Extent3d { width: ENVELOPE_SIZE, height: ENVELOPE_SIZE, depth_or_array_layers: POPULATED as u32 },
            mip_level_count: ENVELOPE_MIPS,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let envelope_mip_views: Vec<Vec<TextureView>> = (0..POPULATED as u32)
            .map(|i| {
                (0..ENVELOPE_MIPS)
                    .map(|m| {
                        envelope.create_view(&TextureViewDescriptor {
                            label: Some("cloud_envelope_mip"),
                            dimension: Some(TextureViewDimension::D2),
                            base_array_layer: i,
                            array_layer_count: Some(1),
                            base_mip_level: m,
                            mip_level_count: Some(1),
                            ..Default::default()
                        })
                    })
                    .collect()
            })
            .collect();
        let envelope_layers: [TextureView; POPULATED] = std::array::from_fn(|i| envelope_mip_views[i][0].clone());
        let envelope_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_envelope_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });
        // Each mip the average of the one above: a linear read at the shared
        // corner of four texels.
        let mip_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("cloud_envelope_mip_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::FRAGMENT, ty: BindingType::Sampler(SamplerBindingType::Filtering), count: None },
            ],
        });
        let mip_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_envelope_mip_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let mip_bind_groups: Vec<Vec<BindGroup>> = envelope_mip_views
            .iter()
            .map(|views| {
                views[..views.len() - 1]
                    .iter()
                    .map(|v| {
                        device.create_bind_group(&BindGroupDescriptor {
                            label: Some("cloud_envelope_mip_bg"),
                            layout: &mip_layout,
                            entries: &[BindGroupEntry { binding: 0, resource: BindingResource::TextureView(v) }, BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&mip_sampler) }],
                        })
                    })
                    .collect()
            })
            .collect();
        let mip_shader = device.create_shader_module(ShaderModuleDescriptor { label: Some("cloud_envelope_mip"), source: ShaderSource::Wgsl(ENVELOPE_MIP_WGSL.into()) });
        let mip_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("cloud_envelope_mip_layout"),
            bind_group_layouts: &[Some(&mip_layout)],
            immediate_size: 0,
        });
        let mip_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("cloud_envelope_mip"),
            layout: Some(&mip_pipeline_layout),
            vertex: VertexState { module: &mip_shader, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(FragmentState {
                module: &mip_shader,
                entry_point: Some("fs_mip"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState { format: TextureFormat::Rgba16Float, blend: None, write_mask: ColorWrites::ALL })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let splat_uniform = device.create_buffer(&BufferDescriptor {
            label: Some("cloud_splat_uniform"),
            size: (16 * (POPULATED + 1)) as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let splat_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("cloud_splat_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let splat_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("cloud_splat_bg"),
            layout: &splat_layout,
            entries: &[BindGroupEntry { binding: 0, resource: splat_uniform.as_entire_binding() }],
        });
        let splat_shader = device.create_shader_module(ShaderModuleDescriptor { label: Some("cloud_splat"), source: ShaderSource::Wgsl(cloud_splat_shader().into()) });
        let splat_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("cloud_splat_layout"),
            bind_group_layouts: &[Some(&splat_layout)],
            immediate_size: 0,
        });
        let add = BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::One, operation: BlendOperation::Add };
        let splat_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("cloud_splat"),
            layout: Some(&splat_pipeline_layout),
            vertex: VertexState {
                module: &splat_shader,
                entry_point: Some("vs_splat"),
                compilation_options: Default::default(),
                buffers: &[Some(VertexBufferLayout {
                    array_stride: 32,
                    step_mode: VertexStepMode::Instance,
                    attributes: &vertex_attr_array![0 => Float32x4, 1 => Float32x4],
                })],
            },
            fragment: Some(FragmentState {
                module: &splat_shader,
                entry_point: Some("fs_splat"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState { format: TextureFormat::Rgba16Float, blend: Some(BlendState { color: add, alpha: add }), write_mask: ColorWrites::ALL })],
            }),
            primitive: PrimitiveState { topology: PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        // Clear sky until a map is drawn: a map never drawn reads black. Its
        // heights read 0: nothing to carry.
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor { label: Some("cloud_sky_clear") });
        for (view, clear) in map_layers.iter().map(|v| (v, Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 })).chain(alt_layers.iter().map(|v| (v, Color::TRANSPARENT))) {
            enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("cloud_sky_clear"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(clear), store: StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
        }
        queue.submit(Some(enc.finish()));
        let bind_group = Self::bind(device, &layout, &uniform, &noise_tex, &weather, &sampler, &envelope, &envelope_sampler);
        let shadow_bind_group = Self::bind(device, &layout, &shadow_uniform, &noise_tex, &weather, &sampler, &envelope, &envelope_sampler);
        let history_uniform = device.create_buffer(&BufferDescriptor {
            label: Some("cloud_history_uniform"),
            size: std::mem::size_of::<[[f32; 4]; 8]>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let history_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("cloud_history_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex(1, TextureViewDimension::D2),
                tex(2, TextureViewDimension::D2),
            ],
        });
        let history_groups: [BindGroup; 3] = std::array::from_fn(|i| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("cloud_history_bg"),
                layout: &history_layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: history_uniform.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&map_layers[i]) },
                    BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&alt_layers[i]) },
                ],
            })
        });
        let shader = device.create_shader_module(ShaderModuleDescriptor { label: Some("cloud_sky"), source: ShaderSource::Wgsl(source.into()) });
        let map_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("cloud_sky_map_layout"),
            bind_group_layouts: &[Some(&layout), Some(&history_layout)],
            immediate_size: 0,
        });
        let shadow_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("cloud_sky_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |label: &str, entry: &str, formats: &[TextureFormat], pipeline_layout: &PipelineLayout| {
            let targets: Vec<Option<ColorTargetState>> = formats.iter().map(|&format| Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::ALL })).collect();
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(pipeline_layout),
                vertex: VertexState { module: &shader, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let map_pipeline = pipeline("cloud_sky_map", "fs_map", &[TextureFormat::Rgba16Float, TextureFormat::Rgba16Float], &map_layout);
        let shadow_pipeline = pipeline("cloud_sky_shadow", "fs_shadow", &[TextureFormat::R8Unorm], &shadow_layout);
        Self {
            noise: noise_tex,
            weather,
            envelope,
            envelope_layers,
            envelope_sampler,
            splat_pipeline,
            envelope_mip_views,
            mip_pipeline,
            mip_bind_groups,
            splat_uniform,
            splat_bind_group,
            splats: None,
            splat_ranges: std::array::from_fn(|_| 0..0),
            envelope_generation: 0,
            envelope_pending: false,
            splatted: false,
            maps,
            map_layers,
            array_view,
            raw,
            raw_view,
            prefilter_pipeline,
            prefilter_group,
            alts,
            alt_layers,
            alt_view,
            sampler,
            uniform,
            layout,
            bind_group,
            shadow_uniform,
            shadow_bind_group,
            map_pipeline,
            shadow_pipeline,
            history_uniform,
            history_groups,
            history_reject: [HISTORY_REJECT[0], HISTORY_REJECT[1], HISTORY_REJECT_WIDE[0], HISTORY_REJECT_WIDE[1]],
            history_share: [HISTORY_REJECT_SHARE[0], HISTORY_REJECT_SHARE[1], HISTORY_SHARE_FLOOR, 1.0],
            winds: [[0.0; 4]; LAYERS],
            maps_begun: 0,
            prev: 0,
            cur: 0,
            build: 1,
            next_row: 0,
            fresh: true,
            scales: [1.0; 3],
            times: [0.0; 3],
            eyes: [[0.0; 3]; 3],
            map_uniforms: [CloudSkyUniform::default(); 3],
            period: MAP_SLICES as f64 / 72.0,
            last_start: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn bind(device: &Device, layout: &BindGroupLayout, uniform: &Buffer, noise: &Texture, weather: &Texture, sampler: &Sampler, envelope: &Texture, envelope_sampler: &Sampler) -> BindGroup {
        let nv = noise.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D3), ..Default::default() });
        let wv = weather.create_view(&Default::default());
        let ev = envelope.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() });
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("cloud_sky_bg"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&nv) },
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&wv) },
                BindGroupEntry { binding: 3, resource: BindingResource::Sampler(sampler) },
                BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&ev) },
                BindGroupEntry { binding: 5, resource: BindingResource::Sampler(envelope_sampler) },
            ],
        })
    }

    /// A new weather field (a new sky's seed).
    pub fn set_weather(&mut self, device: &Device, queue: &Queue, field: &CloudField) {
        self.weather = weather_texture(device, queue, field);
        self.bind_group = Self::bind(device, &self.layout, &self.uniform, &self.noise, &self.weather, &self.sampler, &self.envelope, &self.envelope_sampler);
        self.shadow_bind_group = Self::bind(device, &self.layout, &self.shadow_uniform, &self.noise, &self.weather, &self.sampler, &self.envelope, &self.envelope_sampler);
        self.fresh = true;
    }

    /// THE POPULATION'S LIVE CLOUDS, to be splatted into the envelopes at the
    /// next `encode` -- once per generation, so once a second of cloud time
    /// (`cloud_population::UPDATE_S`).
    pub fn set_population(&mut self, device: &Device, queue: &Queue, population: &CloudPopulation) {
        if population.generation() == self.envelope_generation {
            return;
        }
        self.envelope_generation = population.generation();
        let mut consts = [[0.0f32; 4]; POPULATED + 1];
        for (k, c) in consts.iter_mut().take(POPULATED).enumerate() {
            *c = population.splats(k).1.gpu();
        }
        // Packed on the worker (`CloudPopulation::frame`) when it drew them;
        // here otherwise.
        let own;
        let data: &[[f32; 8]] = match population.packed() {
            Some((data, ranges)) => {
                self.splat_ranges = ranges.clone();
                data
            }
            None => {
                let mut v: Vec<[f32; 8]> = Vec::new();
                for k in 0..POPULATED {
                    let start = v.len() as u32;
                    v.extend(population.splats(k).0.iter().map(|s| s.gpu()));
                    self.splat_ranges[k] = start..v.len() as u32;
                }
                own = v;
                &own
            }
        };
        consts[POPULATED] = [cp::ENVELOPE_TEXEL_M, cp::ENVELOPE_EXTENT_M as f32, 0.0, 0.0];
        queue.write_buffer(&self.splat_uniform, 0, bytemuck::cast_slice(&consts));
        let bytes = (data.len() * 32) as u64;
        if bytes > 0 {
            if self.splats.as_ref().is_none_or(|b| b.size() < bytes) {
                self.splats = Some(device.create_buffer(&BufferDescriptor {
                    label: Some("cloud_splats"),
                    size: (bytes + bytes / 2).max(64 * 1024),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            queue.write_buffer(self.splats.as_ref().unwrap(), 0, bytemuck::cast_slice(data));
        }
        self.envelope_pending = true;
    }

    /// The maps the sky shader reads (`texture_2d_array`).
    pub fn maps_view(&self) -> &TextureView {
        &self.array_view
    }

    /// The maps and their heights, for tests that read them back.
    #[cfg(test)]
    pub(crate) fn textures(&self) -> (&Texture, &Texture) {
        (&self.maps, &self.alts)
    }

    /// The march's texels before the prefilter: the last map's, row for
    /// row what the CPU twin computes. For tests.
    #[cfg(test)]
    pub(crate) fn raw(&self) -> &Texture {
        &self.raw
    }

    /// The heights beside the maps (`texture_2d_array`, R16F metres).
    pub fn alts_view(&self) -> &TextureView {
        &self.alt_view
    }

    /// The winds the maps are carried on (`flow_winds`): what a map being
    /// built carries the newest one to its own moment by. Taken as a map
    /// begins.
    pub fn set_winds(&mut self, winds: [[f32; 4]; LAYERS]) {
        self.winds = winds;
    }

    /// Where a map's history is let go above the horizon (`HISTORY_REJECT`)
    /// and at it (`HISTORY_HORIZON_UP`); taken as a map begins.
    pub fn set_history_reject(&mut self, above: [f32; 2], horizon: [f32; 2]) {
        self.history_reject = [above[0], above[1], horizon[0], horizon[1]];
    }

    /// Whether history is also let go by its share of the cloud
    /// (`HISTORY_REJECT_SHARE`); taken as a map begins.
    pub fn set_history_share(&mut self, on: bool) {
        self.history_share = if on { [HISTORY_REJECT_SHARE[0], HISTORY_REJECT_SHARE[1], HISTORY_SHARE_FLOOR, 1.0] } else { [0.0; 4] };
    }

    /// Whether the next `encode_at` begins a map (and so takes its moment).
    pub fn starts_map(&self, whole: bool) -> bool {
        self.fresh || whole || self.next_row == 0
    }

    /// Whether the last `encode_at` splatted the envelopes.
    pub fn splatted(&self) -> bool {
        self.splatted
    }

    /// No map is drawn yet: the first is drawn whole, of the sky now.
    pub fn is_fresh(&self) -> bool {
        self.fresh
    }

    /// How far ahead of the clock a map begun now should hold the sky: two
    /// builds, when it is half way through being shown (it rises over the
    /// build after its own and falls over the next), so the shapes on
    /// screen are neither behind nor ahead of the clock on average.
    pub fn map_lead(&self) -> f64 {
        2.0 * self.period
    }

    /// What the sky shader carries the two shown maps by (last whole,
    /// newest): the eye now less each map's (xyz), and the seconds from each
    /// map's moment to `now` (w), held within a few builds of it so a clock
    /// that jumps carries nothing off the sky.
    pub fn display_flow(&self, now: f64, eye: [f32; 3]) -> [[f32; 4]; 2] {
        let reach = 4.0 * self.period + 1.0;
        [self.prev, self.cur].map(|i| {
            let e = self.eyes[i as usize];
            [eye[0] - e[0], eye[1] - e[1], eye[2] - e[2], (now - self.times[i as usize]).clamp(-reach, reach) as f32]
        })
    }

    /// The layers the sky shader blends between and how far: the last whole
    /// map, the newest, and the share of the next one built.
    pub fn display(&self) -> [f32; 3] {
        [self.prev as f32, self.cur as f32, self.next_row as f32 / MAP_HEIGHT as f32]
    }

    /// The radiance one stored unit of the shown maps stands for: the last
    /// whole map's, the newest's.
    pub fn display_scales(&self) -> [f32; 2] {
        [1.0 / self.scales[self.prev as usize], 1.0 / self.scales[self.cur as usize]]
    }

    /// `encode_at` with one uniform for the map and the shadow, at time 0.
    pub fn encode(&mut self, encoder: &mut CommandEncoder, queue: &Queue, u: &CloudSkyUniform, shadow: Option<&TextureView>, whole: bool) {
        self.encode_at(encoder, queue, u, u, shadow, whole, 0.0, 0.0);
    }

    /// ONE FRAME'S CLOUDS: the next slice of the sky map (the whole map when
    /// `whole`, or when none is drawn yet) from `map` -- the sky at
    /// `map_time`, seen from its eye, the same for every slice of one map --
    /// and the ground's shadow into `shadow` when given, from `now_u`: the
    /// clouds at `now`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_at(&mut self, encoder: &mut CommandEncoder, queue: &Queue, map: &CloudSkyUniform, now_u: &CloudSkyUniform, shadow: Option<&TextureView>, whole: bool, now: f64, map_time: f64) {
        // A new population is splatted only as a map begins, so one map is
        // built from one population top to bottom; re-splatted mid-build it
        // held bands of rows from two (research 2026-10-21, cause (c)). The
        // ground's shadow reads the same envelopes, so it changes with it.
        self.splatted = self.envelope_pending && self.starts_map(whole);
        if self.splatted {
            self.envelope_pending = false;
            for k in 0..POPULATED {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("cloud_envelope"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &self.envelope_layers[k],
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations { load: LoadOp::Clear(Color::TRANSPARENT), store: StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                    occlusion_query_set: None,
                });
                let range = self.splat_ranges[k].clone();
                if let (Some(buf), false) = (self.splats.as_ref(), range.is_empty()) {
                    pass.set_pipeline(&self.splat_pipeline);
                    pass.set_bind_group(0, &self.splat_bind_group, &[]);
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.draw(0..4, range);
                }
            }
            for k in 0..POPULATED {
                for m in 1..ENVELOPE_MIPS as usize {
                    let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                        label: Some("cloud_envelope_mip"),
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &self.envelope_mip_views[k][m],
                            depth_slice: None,
                            resolve_target: None,
                            ops: Operations { load: LoadOp::Clear(Color::TRANSPARENT), store: StoreOp::Store },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        multiview_mask: None,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(&self.mip_pipeline);
                    pass.set_bind_group(0, &self.mip_bind_groups[k][m - 1], &[]);
                    pass.draw(0..3, 0..1);
                }
            }
        }
        let rows = MAP_HEIGHT / MAP_SLICES;
        let (target, from, to) = if self.fresh || whole {
            (self.cur, 0, MAP_HEIGHT)
        } else {
            (self.build, self.next_row, (self.next_row + rows).min(MAP_HEIGHT))
        };
        // A map keeps the scale its first slice was drawn at, and its moment.
        if from == 0 {
            let peak = (0..3).map(|c| map.light[c] + map.amb_sky[c]).fold(0.0f32, f32::max);
            self.scales[target as usize] = 1.0 / peak.max(1e-20);
            self.times[target as usize] = map_time;
            self.eyes[target as usize] = [map.eye[0], map.eye[1], map.eye[2]];
            self.map_uniforms[target as usize] = *map;
            if !(self.fresh || whole) {
                if let Some(s) = self.last_start {
                    self.period = (0.5 * self.period + 0.5 * (now - s)).clamp(0.05, 30.0);
                }
                self.last_start = Some(now);
            }
        }
        // What this map accumulates over: the newest whole map, unless it is
        // drawn whole (nothing is), or the clock has jumped between the two.
        let carry = !(self.fresh || whole);
        let history = if carry { self.cur } else { (target + 1) % 3 };
        if from == 0 {
            let (t, h) = (target as usize, history as usize);
            let age = self.times[t] - self.times[h];
            let take = carry && age.abs() <= 4.0 * self.period + 1.0;
            // The sample fraction moves map to map only while the maps
            // accumulate: unaveraged, a new fraction each map changed the far
            // clouds at every swap, a flicker along the horizon (user,
            // 2026-10-22; `a_still_sky_holds_still_at_the_horizon_from_map_to_map`).
            let jitter = if carry && HISTORY_NEW < 1.0 { (0.5 + self.maps_begun as f32 * HISTORY_JITTER).fract() } else { 0.5 };
            self.maps_begun = self.maps_begun.wrapping_add(1);
            let (et, eh) = (self.eyes[t], self.eyes[h]);
            let w = &self.winds;
            let carried: [[f32; 4]; 8] = [
                [et[0] - eh[0], et[1] - eh[1], et[2] - eh[2], age as f32],
                w[0],
                w[1],
                w[2],
                w[3],
                // The history carried as it is, a share of its own map's
                // light: a cloud's light follows the light it is lit by, so
                // that is what this map's would be. Converted to the old
                // map's absolute light (`scales[t] / scales[h]`), each map
                // kept 70% of the last map's radiance while the dusk dimmed
                // the sky 1e5 times; past 1.43 times a map the sum grew, and
                // the night's exposure drew the day's clouds blown white
                // through the night (headset, 2026-10-23).
                [if take { HISTORY_NEW } else { 1.0 }, 1.0, jitter, et[1]],
                self.history_reject,
                self.history_share,
            ];
            queue.write_buffer(&self.history_uniform, 0, bytemuck::cast_slice(&carried));
        }
        let mut u = self.map_uniforms[target as usize];
        u.eye[3] = self.scales[target as usize];
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        queue.write_buffer(&self.shadow_uniform, 0, bytemuck::bytes_of(now_u));
        {
            let ops = Operations { load: LoadOp::Load, store: StoreOp::Store };
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("cloud_sky_map"),
                color_attachments: &[
                    Some(RenderPassColorAttachment { view: &self.raw_view, depth_slice: None, resolve_target: None, ops }),
                    Some(RenderPassColorAttachment { view: &self.alt_layers[target as usize], depth_slice: None, resolve_target: None, ops }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.map_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, &self.history_groups[history as usize], &[]);
            pass.set_scissor_rect(0, from, MAP_WIDTH, to - from);
            pass.draw(0..3, 0..1);
        }
        // The prefilter reads two rows either side: the band before this
        // slice is whole now, and the last slice takes its own band too.
        let whole_map = self.fresh || whole;
        let (f0, f1) = if whole_map {
            (0, MAP_HEIGHT)
        } else {
            (from.saturating_sub(rows), if to >= MAP_HEIGHT { MAP_HEIGHT } else { from })
        };
        if f1 > f0 {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("cloud_sky_prefilter"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.map_layers[target as usize],
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Load, store: StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.prefilter_pipeline);
            pass.set_bind_group(0, &self.prefilter_group, &[]);
            pass.set_scissor_rect(0, f0, MAP_WIDTH, f1 - f0);
            pass.draw(0..3, 0..1);
        }
        if self.fresh || whole {
            self.fresh = false;
            self.prev = self.cur;
            self.next_row = 0;
        } else {
            self.next_row = to;
            if to >= MAP_HEIGHT {
                let (p, c) = (self.cur, self.build);
                self.prev = p;
                self.cur = c;
                self.build = 3 - p - c;
                self.next_row = 0;
            }
        }
        if let Some(view) = shadow {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("cloud_sky_shadow"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::WHITE), store: StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &self.shadow_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// The marcher: `cloud_layers`' density, height profiles, light and cirrus
/// twinned line for line, then the map and shadow entry points.
pub fn cloud_sky_shader() -> String {
    let f = |v: f32| format!("{v:?}");
    let means = CloudNoise::shared().channel_means();
    let steps = |s: [u32; 3]| format!("vec3<u32>({}u, {}u, {}u)", s[0], s[1], s[2]);
    CLOUD_SKY_WGSL
        .replace("{EARTH}", &format!("{:.1}", c::EARTH_RADIUS_M))
        .replace("{DETAIL_RATIO}", &f(cl::DETAIL_RATIO))
        .replace("{DETAIL_EROSION}", &f(cl::DETAIL_EROSION))
        .replace("{CU_SOLID0}", &f(cl::CU_SOLID[0]))
        .replace("{CU_SOLID1}", &f(cl::CU_SOLID[1]))
        .replace("{CU_CORE_FULL}", &f(cl::CU_CORE_FULL))
        .replace("{CU_BASE_CURL}", &f(cl::CU_BASE_CURL))
        .replace("{CU_LOBE}", &f(cl::CU_LOBE))
        .replace("{CU_TURRET}", &f(cl::CU_TURRET))
        .replace("{CU_BASE_LUMP}", &f(cl::CU_BASE_LUMP))
        .replace("{CU_LOBE_MEAN}", &f(cl::CU_LOBE_MEAN))
        .replace("{CU_TOP_FADE}", &f(cl::CU_TOP_FADE))
        .replace("{ERODE_DEPTH}", &f(cl::ERODE_DEPTH))
        .replace("{CHURN_WARP}", &f(cl::CHURN_WARP))
        .replace("{SHAPE_OCTAVES}", &cl::SHAPE_OCTAVES.map(f).join(", "))
        .replace("{CHURN_PULSE}", &f(cl::CHURN_PULSE))
        .replace("{CHURN_FAST}", &f(cl::CHURN_FAST))
        .replace("{CHURN_FAST_SHARE}", &f(cl::CHURN_FAST_SHARE))
        .replace("{THERMAL_S}", &f(cl::THERMAL_S))
        .replace("{TURRET_S}", &f(cl::TURRET_S))
        .replace("{RISE_EDGE}", &f(cl::RISE_EDGE))
        .replace("{RISE_CORE}", &f(cl::RISE_CORE))
        .replace("{WARP_GAIN0}", &f(cl::WARP_GAIN[0]))
        .replace("{WARP_GAIN1}", &f(cl::WARP_GAIN[1]))
        .replace("{SHEAR_MOMENTUM}", &f(cl::SHEAR_MOMENTUM))
        .replace("{SHEAR_LEAN_MAX}", &f(cl::SHEAR_LEAN_MAX))
        .replace("{DETILE_REACH}", &f(cl::DETILE_REACH))
        .replace("{DETILE_WAVES}", &cl::DETILE_WAVES.map(f).join(", "))
        .replace("{DETILE_DIRS}", &cl::DETILE_DIRS.map(|d| format!("vec2<f32>({}, {})", f(d[0]), f(d[1]))).join(", "))
        .replace("{EDGE_FRINGE}", &f(cl::EDGE_FRINGE))
        .replace("{EDGE_FRINGE_D}", &f(cl::EDGE_FRINGE_D))
        .replace("{EDGE_AGEING}", &f(cl::EDGE_AGEING))
        .replace("{EDGE_GAIN}", &f(cl::EDGE_GAIN))
        .replace("{EDGE_AGEING_DEPTH}", &f(cl::EDGE_AGEING_DEPTH))
        .replace("{MAX_MARCH}", &f(cl::MAX_MARCH_M))
        .replace("{CIRRUS_TAU}", &f(cl::CIRRUS_TAU))
        .replace("{SUN_RADIUS}", &f(cl::SHADOW_DISC_RAD))
        .replace("{CIRRUS_STRETCH}", &f(cl::CIRRUS_STRETCH))
        .replace("{EXTINCTION}", &f(c::EXTINCTION_PER_M))
        .replace("{VIEW_STEPS}", &steps(cl::VIEW_STEPS))
        .replace("{VIEW_STEPS_MOST}", &format!("{}u", cl::VIEW_STEPS_MOST))
        .replace("{VIEW_STEP_TEXELS}", &f(cl::VIEW_STEP_TEXELS))
        .replace("{SHADOW_STEPS}", &steps(cl::SHADOW_STEPS))
        .replace("{GLOW_GAIN}", &f(c::CLOUD_GLOW[0]))
        .replace("{GLOW_G}", &f(c::CLOUD_GLOW[1]))
        .replace("{GLOW_DECAY}", &f(c::CLOUD_GLOW[2]))
        .replace("{GLOW_NEAR}", &f(c::CLOUD_GLOW_NEAR))
        .replace("{FORWARD_G}", &f(c::PHASE_FORWARD_G))
        .replace("{BACK_G}", &f(c::PHASE_BACK_G))
        .replace("{BACK_SHARE}", &f(c::PHASE_BACK_SHARE))
        .replace("{HAZE}", &format!("{:e}", c::HAZE_PER_M))
        .replace("{AEROSOL_HEIGHT}", &f(space_soup_sky::time_of_day::AEROSOL_HEIGHT_M))
        .replace("{AIR_HAZE}", &format!("{:e}", space_soup_sky::time_of_day::AIR_HAZE_PER_M))
        .replace("{AIR_HAZE_HEIGHT}", &f(space_soup_sky::time_of_day::AIR_HAZE_HEIGHT_M))
        .replace("{POWDER_K}", &f(c::POWDER_K))
        .replace("{PACK}", &f(MAP_PACK))
        .replace("{KIND_CUMULUS}", &f(cl::KIND_CUMULUS))
        .replace("{KIND_STRATO}", &f(cl::KIND_STRATOCUMULUS))
        .replace("{KIND_CIRRUS}", &f(cl::KIND_CIRRUS))
        .replace("{KIND_ALTO}", &f(cl::KIND_ALTOCUMULUS))
        .replace("{HIGH_SHADE_FILL}", &f(cl::HIGH_SHADE_FILL))
        .replace("{SHADOW_SIZE}", &f(crate::renderer::uniforms::CLOUD_SHADOW_SIZE as f32))
        .replace("{NOISE_TEXELS}", &f(NOISE_SIZE as f32))
        .replace("{MEAN_A}", &f(means[3]))
        .replace("{MEAN_B}", &f(means[2]))
        .replace("{MEAN_FINE}", &f(0.625 * means[1] + 0.25 * means[2] + 0.125 * means[3]))
        .replace("{MERGE_LIFT}", &f(cp::MERGE_LIFT))
        .replace("{COVER_CEIL}", &f(cp::COVER_CEIL))
        .replace("{CUMULUS_COVER_CEIL}", &f(cp::CUMULUS_COVER_CEIL))
        .replace("{OVERCAST0}", &f(cp::OVERCAST[0]))
        .replace("{OVERCAST1}", &f(cp::OVERCAST[1]))
        .replace("{FAR_PHASE}", &f(cp::FAR_PHASE))
        .replace("{FAR_BLEND0}", &f(cp::FAR_BLEND[0]))
        .replace("{FAR_BLEND1}", &f(cp::FAR_BLEND[1]))
        .replace("{POPULATED}", &format!("{}u", POPULATED))
        .replace("{ENVELOPE_TEXEL}", &f(cp::ENVELOPE_TEXEL_M))
        .replace("{ENVELOPE_TOP_MIP}", &f((ENVELOPE_MIPS - 1) as f32))
        .replace("{ALT_FILL_WEIGHT}", &f(ALT_FILL_WEIGHT))
        .replace("{HISTORY_HORIZON_UP0}", &f(HISTORY_HORIZON_UP[0]))
        .replace("{HISTORY_HORIZON_UP1}", &f(HISTORY_HORIZON_UP[1]))
        .replace("{HISTORY_REFETCH}", if HISTORY_REFETCH { "true" } else { "false" })
        .replace("{HISTORY_ALT_AGREE}", &f(HISTORY_ALT_AGREE))
        + &cloud_flow_wgsl()
}

/// The envelope pass: each live cloud a square around its profile, added
/// into its layer's map (`cloud_population::splat_value`, line for line).
/// The prefilter (`PREFILTER`): the march's texels, 5 x 5, into a map row
/// band, wrapping round the sky and clamped at its rows' ends.
pub fn prefilter_shader() -> String {
    let w = PREFILTER;
    let sum = w[0] + 2.0 * (w[1] + w[2]);
    format!(
        r#"@group(0) @binding(0) var raw: texture_2d<f32>;
const W0: f32 = {:.6};
const W1: f32 = {:.6};
const W2: f32 = {:.6};
@vertex fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}}
@fragment fn fs_prefilter(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {{
    let size = vec2<i32>(textureDimensions(raw));
    let c = vec2<i32>(pos.xy);
    var w = array<f32, 5>(W2, W1, W0, W1, W2);
    var sum = vec4<f32>(0.0);
    for (var j = 0; j < 5; j = j + 1) {{
        let y = clamp(c.y + j - 2, 0, size.y - 1);
        var row = vec4<f32>(0.0);
        for (var i = 0; i < 5; i = i + 1) {{
            let x = (c.x + i - 2 + size.x) % size.x;
            row = row + w[i] * textureLoad(raw, vec2<i32>(x, y), 0);
        }}
        sum = sum + w[j] * row;
    }}
    return vec4<f32>(max(sum.rgb, vec3<f32>(0.0)), clamp(sum.a, 0.0, 1.0));
}}
"#,
        w[0] / sum,
        w[1] / sum,
        w[2] / sum
    )
}

pub fn cloud_splat_shader() -> String {
    CLOUD_SPLAT_WGSL
        .replace("{LUMP0}", &format!("{:?}", cp::LUMP[0]))
        .replace("{LUMP1}", &format!("{:?}", cp::LUMP[1]))
        .replace("{POPULATED}", &POPULATED.to_string())
}

const ENVELOPE_MIP_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@vertex fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn fs_mip(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(src));
    // This texel's centre is the corner its four sources share.
    return textureSampleLevel(src, samp, 2.0 * floor(pos.xy) / size + 1.0 / size, 0.0);
}
"#;

const CLOUD_SPLAT_WGSL: &str = r#"
struct SplatUniform {
    // Per layer: wind x, z, sqrt(stretch), gain; then texel m, extent m.
    layers: array<vec4<f32>, {POPULATED}>,
    misc: vec4<f32>,
}
@group(0) @binding(0) var<uniform> su: SplatUniform;

const LUMP0: f32 = {LUMP0};
const LUMP1: f32 = {LUMP1};

struct SplatOut {
    @builtin(position) pos: vec4<f32>,
    // Centre x, z (map m), radius, strength.
    @location(0) @interpolate(flat) a: vec4<f32>,
    // Top, phase, sin and cos of the outline's turn.
    @location(1) @interpolate(flat) b: vec4<f32>,
    // Wind x, z, sqrt(stretch), gain.
    @location(2) @interpolate(flat) c: vec4<f32>,
}

@vertex fn vs_splat(@builtin(vertex_index) vi: u32, @location(0) a: vec4<f32>, @location(1) b: vec4<f32>) -> SplatOut {
    let c = su.layers[u32(b.w)];
    let bound = a.z * max(c.z, 1.0 / c.z) * (1.0 + LUMP0 + LUMP1);
    let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u)) * 2.0 - 1.0;
    let uv = (a.xy + corner * bound) / su.misc.y;
    var out: SplatOut;
    out.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    out.a = a;
    out.b = vec4<f32>(b.x, b.y, sin(b.z), cos(b.z));
    out.c = c;
    return out;
}

@fragment fn fs_splat(in: SplatOut) -> @location(0) vec4<f32> {
    // The texel's centre, in map metres.
    let p = in.pos.xy * su.misc.x;
    let d = p - in.a.xy;
    let wind = in.c.xy;
    let al = d.x * wind.x + d.y * wind.y;
    let ac = -d.x * wind.y + d.y * wind.x;
    let qx = al / (in.a.z * in.c.z);
    let qz = ac * in.c.z / in.a.z;
    let r2 = qx * qx + qz * qz;
    let reach = 1.0 + LUMP0 + LUMP1;
    if (r2 >= reach * reach) {
        return vec4<f32>(0.0);
    }
    let len = max(sqrt(r2), 1e-6);
    let sn = in.b.z;
    let cs = in.b.w;
    let c1 = (qx * cs - qz * sn) / len;
    let s1 = (qx * sn + qz * cs) / len;
    let lump = 1.0 + LUMP0 * (4.0 * c1 * c1 * c1 - 3.0 * c1) + LUMP1 * (2.0 * s1 * c1);
    let x2 = r2 / (lump * lump);
    if (x2 >= 1.0) {
        return vec4<f32>(0.0);
    }
    let bb = 1.0 - x2;
    let w = in.a.w * in.c.w * bb * bb;
    return vec4<f32>(w, w * in.b.x, w * in.b.y, w * w);
}
"#;

const CLOUD_SKY_WGSL: &str = r#"
struct Layer {
    v: array<vec4<f32>, 12>,
}
struct CloudSkyUniform {
    eye: vec4<f32>,
    light_dir: vec4<f32>,
    light: vec4<f32>,
    amb_sky: vec4<f32>,
    amb_ground: vec4<f32>,
    shadow: vec4<f32>,
    layers: array<Layer, 4>,
}
@group(0) @binding(0) var<uniform> u: CloudSkyUniform;
@group(0) @binding(1) var noise_tex: texture_3d<f32>;
@group(0) @binding(2) var weather_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var env_tex: texture_2d_array<f32>;
@group(0) @binding(5) var env_samp: sampler;
// The map slices accumulate over (`HISTORY_NEW`), and how it is carried.
struct History {
    // The built map's eye less the history's (xyz); seconds between them (w).
    flow: vec4<f32>,
    winds: array<vec4<f32>, 4>,
    // The new march's share (1: no history), what the history's light is
    // multiplied by (1: carried as a share of its own map's light), the
    // samples' fraction of their step, the built eye's altitude.
    misc: vec4<f32>,
    // Where old and new disagree from (x) and to (y), the new march is
    // taken alone (`HISTORY_REJECT`); at the horizon from z to w.
    reject: vec4<f32>,
    // And by the share of the cloud it differs by: from (x) to (y), over at
    // least (z) of cloud; on (w).
    share: vec4<f32>,
}
@group(1) @binding(0) var<uniform> hist: History;
@group(1) @binding(1) var hist_light: texture_2d<f32>;
@group(1) @binding(2) var hist_alt: texture_2d<f32>;

const PI: f32 = 3.14159265;
const HISTORY_REFETCH: bool = {HISTORY_REFETCH};
const HISTORY_HORIZON_UP0: f32 = {HISTORY_HORIZON_UP0};
const HISTORY_HORIZON_UP1: f32 = {HISTORY_HORIZON_UP1};
const HISTORY_ALT_AGREE: f32 = {HISTORY_ALT_AGREE};
const EARTH: f32 = {EARTH};
const DETAIL_RATIO: f32 = {DETAIL_RATIO};
const DETAIL_EROSION: f32 = {DETAIL_EROSION};
const CU_SOLID0: f32 = {CU_SOLID0};
const CU_SOLID1: f32 = {CU_SOLID1};
const CU_CORE_FULL: f32 = {CU_CORE_FULL};
const CU_BASE_CURL: f32 = {CU_BASE_CURL};
const CU_LOBE: f32 = {CU_LOBE};
const CU_TURRET: f32 = {CU_TURRET};
const CU_BASE_LUMP: f32 = {CU_BASE_LUMP};
const CU_LOBE_MEAN: f32 = {CU_LOBE_MEAN};
const CU_TOP_FADE: f32 = {CU_TOP_FADE};
const ERODE_DEPTH: f32 = {ERODE_DEPTH};
const CHURN_WARP: f32 = {CHURN_WARP};
const SHAPE_OCTAVES: vec3<f32> = vec3<f32>({SHAPE_OCTAVES});
const CHURN_PULSE: f32 = {CHURN_PULSE};
const CHURN_FAST: f32 = {CHURN_FAST};
const CHURN_FAST_SHARE: f32 = {CHURN_FAST_SHARE};
const THERMAL_S: f32 = {THERMAL_S};
const TURRET_S: f32 = {TURRET_S};
const RISE_EDGE: f32 = {RISE_EDGE};
const RISE_CORE: f32 = {RISE_CORE};
const WARP_GAIN0: f32 = {WARP_GAIN0};
const WARP_GAIN1: f32 = {WARP_GAIN1};
const SHEAR_MOMENTUM: f32 = {SHEAR_MOMENTUM};
const SHEAR_LEAN_MAX: f32 = {SHEAR_LEAN_MAX};
const DETILE_REACH: f32 = {DETILE_REACH};
const DETILE_WAVES: vec4<f32> = vec4<f32>({DETILE_WAVES});
const DETILE_DIRS: array<vec2<f32>, 4> = array<vec2<f32>, 4>({DETILE_DIRS});
const EDGE_FRINGE: f32 = {EDGE_FRINGE};
const EDGE_FRINGE_D: f32 = {EDGE_FRINGE_D};
const EDGE_AGEING: f32 = {EDGE_AGEING};
const EDGE_GAIN: f32 = {EDGE_GAIN};
const EDGE_AGEING_DEPTH: f32 = {EDGE_AGEING_DEPTH};
const MAX_MARCH: f32 = {MAX_MARCH};
const CIRRUS_TAU: f32 = {CIRRUS_TAU};
const SUN_RADIUS: f32 = {SUN_RADIUS};
const CIRRUS_STRETCH: f32 = {CIRRUS_STRETCH};
const EXTINCTION: f32 = {EXTINCTION};
const FORWARD_G: f32 = {FORWARD_G};
const BACK_G: f32 = {BACK_G};
const BACK_SHARE: f32 = {BACK_SHARE};
const HAZE: f32 = {HAZE};
const AEROSOL_HEIGHT: f32 = {AEROSOL_HEIGHT};
const AIR_HAZE: f32 = {AIR_HAZE};
const AIR_HAZE_HEIGHT: f32 = {AIR_HAZE_HEIGHT};

// THE HAZE BETWEEN THE EYE AND A CLOUD `t` metres along a ray rising at `mu`
// (2026-10-22): the aerosol (`u.light_dir.w` at the ground) and the air
// thinning with height, through a flat atmosphere -- the depth of
// sigma0 exp(-s mu / H) from 0 to t. One haze for every distance had near
// clouds and those at the horizon equally clear, so the cumulus 30 km off
// stood crisp and white in a carpet down to the sea where the photographed
// sky's fade into a pale horizon.
fn haze_through(t: f32, mu: f32) -> f32 {
    let m = max(mu, 1e-3);
    let depth = u.light_dir.w * AEROSOL_HEIGHT / m * (1.0 - exp(-t * m / AEROSOL_HEIGHT))
        + AIR_HAZE * AIR_HAZE_HEIGHT / m * (1.0 - exp(-t * m / AIR_HAZE_HEIGHT));
    return exp(-depth);
}
const POWDER_K: f32 = {POWDER_K};
const POWDER_SHARE: f32 = 0.35;
const GLOW_GAIN: f32 = {GLOW_GAIN};
const GLOW_G: f32 = {GLOW_G};
const GLOW_DECAY: f32 = {GLOW_DECAY};
const GLOW_NEAR: f32 = {GLOW_NEAR};
const LIGHT_EVERY: u32 = 2u;
const PACK: f32 = {PACK};
const KIND_CUMULUS: f32 = {KIND_CUMULUS};
const KIND_STRATO: f32 = {KIND_STRATO};
const KIND_CIRRUS: f32 = {KIND_CIRRUS};
const KIND_ALTO: f32 = {KIND_ALTO};
const HIGH_SHADE_FILL: f32 = {HIGH_SHADE_FILL};
const NOISE_TEXELS: f32 = {NOISE_TEXELS};
const VIEW_STEPS: vec3<u32> = {VIEW_STEPS};
const VIEW_STEPS_MOST: u32 = {VIEW_STEPS_MOST};
const VIEW_STEP_TEXELS: f32 = {VIEW_STEP_TEXELS};
const REFINE: u32 = 4u;
const REFINE_HOLD: u32 = 2u;
// Halvings that find a cloud's surface between the empty sample and the
// first in cloud (`fs_map`). Off with the history back (2026-10-22 evening),
// as the map was when the user last liked it; 2 measured +0.27 ms a slice.
const ENTRY_BISECT: u32 = 0u;
const MARCH_CAP: u32 = 512u;
const SHADOW_STEPS: vec3<u32> = {SHADOW_STEPS};
const MEAN_A: f32 = {MEAN_A};
const MEAN_B: f32 = {MEAN_B};
const MEAN_FINE: f32 = {MEAN_FINE};
const MERGE_LIFT: f32 = {MERGE_LIFT};
const COVER_CEIL: f32 = {COVER_CEIL};
const CUMULUS_COVER_CEIL: f32 = {CUMULUS_COVER_CEIL};
const OVERCAST0: f32 = {OVERCAST0};
const OVERCAST1: f32 = {OVERCAST1};
const FAR_PHASE: f32 = {FAR_PHASE};
const FAR_BLEND0: f32 = {FAR_BLEND0};
const FAR_BLEND1: f32 = {FAR_BLEND1};
const POPULATED: u32 = {POPULATED};
const ENVELOPE_TEXEL: f32 = {ENVELOPE_TEXEL};
const ENVELOPE_TOP_MIP: f32 = {ENVELOPE_TOP_MIP};
const ALT_FILL_WEIGHT: f32 = {ALT_FILL_WEIGHT};
// The clouds' transmittance over which a disc behind them fades in
// (`fs_map`'s `through`): none below 25%, all of it from 85%. From 3% the
// whole disc showed through a cloud's edge (headset, 2026-10-22): the sun
// is ~1e5 the cloud, so even a third of it reads as the bare disc. Now an
// edge that thins over the disc shows only the part of it in clear sky.
const DISC_THROUGH0: f32 = 0.25;
const DISC_THROUGH1: f32 = 0.85;

@vertex fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn lv(k: u32, i: u32) -> vec4<f32> {
    return u.layers[k].v[i];
}

fn threshold(k: u32, c: f32) -> f32 {
    let x = clamp(c, 0.0, 1.0) * 15.0;
    let i = min(u32(floor(x)), 14u);
    let f = x - f32(i);
    let a = lv(k, 4u + i / 4u)[i % 4u];
    let j = i + 1u;
    let b = lv(k, 4u + j / 4u)[j % 4u];
    return a + (b - a) * f;
}

fn local_coverage(k: u32, xz: vec2<f32>) -> f32 {
    let w = textureSampleLevel(weather_tex, samp, xz * lv(k, 1u).z + lv(k, 3u).xy, 0.0).r;
    return clamp(lv(k, 0u).z + (w - 0.5) * lv(k, 1u).w, 0.0, 1.0);
}

// The weather's mean where there is no population map: coverage, the top
// from it, the fallback's phase, unmerged (`EnvelopeSample::far`).
fn envelope_far(k: u32, xz: vec2<f32>) -> vec4<f32> {
    let c = local_coverage(k, xz);
    return vec4<f32>(c, 0.3 + 0.7 * c, FAR_PHASE, c);
}

// One envelope texel's sums as coverage, column top, phase and merge
// (`EnvelopeSample::of`).
fn envelope_of(e: vec4<f32>) -> vec4<f32> {
    if (e.x <= 1e-4) {
        return vec4<f32>(0.0, 0.0, FAR_PHASE, 0.0);
    }
    let cover = clamp(e.x, 0.0, 1.0);
    let merge = clamp(1.0 - e.w / (e.x * e.x), 0.0, 1.0);
    let top = min(e.y / e.x * (0.35 + 0.65 * sqrt(cover)) * (1.0 + MERGE_LIFT * merge), 1.0);
    // The fourth is the summed strength before the clamp (`EnvelopeSample::core`).
    return vec4<f32>(cover, top, clamp(e.z / e.x, 0.0, 1.0), e.x);
}

// THE LAYER'S ENVELOPE at xz: its population's map, giving way to the
// weather's mean toward the map's edge (`cloud_layers::envelope`).
fn envelope(k: u32, xz: vec2<f32>, foot: f32) -> vec4<f32> {
    let m = lv(k, 8u);
    if (k >= POPULATED || m.z <= 0.0) {
        return envelope_far(k, xz);
    }
    let uv = xz * m.z + m.xy;
    let t = smoothstep(FAR_BLEND0, FAR_BLEND1, 2.0 * max(abs(uv.x - 0.5), abs(uv.y - 0.5)));
    if (t >= 1.0) {
        return envelope_far(k, xz);
    }
    // The mip the sample's footprint spans (`ENVELOPE_MIPS`).
    let lod = clamp(log2(max(foot, ENVELOPE_TEXEL) / ENVELOPE_TEXEL), 0.0, ENVELOPE_TOP_MIP);
    var near = envelope_of(textureSampleLevel(env_tex, env_samp, uv, k, lod));
    // Cores below the ceiling, both giving way to a sheet toward an overcast
    // (`EnvelopeSample::under`).
    let c = lv(k, 0u).z;
    let o = smoothstep(OVERCAST0, OVERCAST1, c);
    let ceil = select(COVER_CEIL, CUMULUS_COVER_CEIL, lv(k, 1u).y == KIND_CUMULUS);
    near.x = max(min(near.x, ceil + (1.0 - ceil) * o), o * c);
    if (t <= 0.0) {
        return near;
    }
    return mix(near, envelope_far(k, xz), t);
}

fn height_profile(kind: f32, h: f32, top: f32) -> f32 {
    if (kind == KIND_CUMULUS) {
        return smoothstep(0.0, 0.06, h) * (1.0 - smoothstep(0.5 * top, top, h));
    } else if (kind == KIND_STRATO) {
        return smoothstep(0.0, 0.15, h) * (1.0 - smoothstep(0.5, 0.95, h));
    }
    return smoothstep(0.0, 0.2, h) * (1.0 - smoothstep(0.45, 0.9, h));
}

// The noise at `s`. For the sky map (`eased`), between texels on a
// smoothstep rather than a straight line: a shape cut at a threshold from a
// trilinear field keeps a corner at every texel face, and an altocumulus
// puff a few texels wide came out as a square (2026-10-20). One tap; the
// light and the ground's shadow keep the trilinear field their CPU twin
// reads.
fn noise_at(s: vec3<f32>, eased: bool) -> vec4<f32> {
    if (!eased) {
        return textureSampleLevel(noise_tex, samp, s, 0.0);
    }
    let p = s * NOISE_TEXELS + 0.5;
    let i = floor(p);
    let f = p - i;
    // Eased across (x, z) only: up the layer (y) a smoothstep's zero slope
    // at every texel face terraced the density into flat strata 62.5 m
    // apart, which the boil dragged up through every cloud as lines
    // (research 2026-10-21, cause (a)). Straight in y, the shapes keep
    // their round puffs and lose the terraces.
    let e = f * f * (3.0 - 2.0 * f);
    let g = vec3<f32>(e.x, f.y, e.z);
    return textureSampleLevel(noise_tex, samp, (i + g - 0.5) / NOISE_TEXELS, 0.0);
}

// The shape noise's lookup shifted off its tile's lattice, `s` world x, z
// in tiles, `ph` the layer's phases (`cloud_layers::detile`, line for line).
fn detile(s: vec2<f32>, ph: vec4<f32>) -> vec2<f32> {
    let w = 6.28318531 / DETILE_WAVES;
    let a = sin(dot(s, DETILE_DIRS[0]) * w.x + ph.x);
    let b = sin(dot(s, DETILE_DIRS[1]) * w.y + ph.y);
    let c = sin(dot(s, DETILE_DIRS[2]) * w.z + ph.z);
    let d = sin(dot(s, DETILE_DIRS[3]) * w.w + ph.w);
    return 0.5 * DETILE_REACH * vec2<f32>(a + b, c + d);
}

// `foot`: the metres one sky-map texel spans where this sample is (0 for
// the light and the ground's shadow, which keep every octave). Detail finer
// than that cannot be drawn, only aliased -- near the horizon it came out
// as rows of specks and stripes (headset, 2026-10-20) -- so the octaves it
// outgrows give way to their means, which keeps the coverage as it was.
fn density(k: u32, xz_in: vec2<f32>, alt: f32, detail: bool, foot: f32) -> f32 {
    let a = lv(k, 0u);
    let h = (alt - a.x) / max(a.y - a.x, 1.0);
    if (h < 0.0 || h > 1.0) {
        return 0.0;
    }
    // The column leans downwind with height (`cloud_layers::shear_tilt`).
    let sh = lv(k, 9u);
    var xz = xz_in;
    if (sh.w > 0.0 && sh.z > 0.0) {
        let z = h * (a.y - a.x);
        xz = xz - sh.xy * min(SHEAR_MOMENTUM * sh.z * z * z / (2.0 * sh.w), SHEAR_LEAN_MAX * z);
    }
    let env = envelope(k, xz, foot);
    let cover = env.x;
    if (cover <= 0.0) {
        return 0.0;
    }
    let si = lv(k, 1u).x;
    let s0 = vec3<f32>(xz.x * si, alt * si, xz.y * si) + lv(k, 2u).xyz;
    let u = detile(xz * si, lv(k, 10u));
    let s = vec3<f32>(s0.x + u.x, s0.y, s0.z + u.y);
    let n = noise_at(s, detail);
    // The footprint in the shape noise's texels.
    let r = foot * NOISE_TEXELS * si;
    let fine_a = smoothstep(0.25, 0.5, r);
    let fine_b = smoothstep(0.5, 1.0, r);
    let low = SHAPE_OCTAVES.x * n.g + SHAPE_OCTAVES.y * mix(n.b, MEAN_B, fine_b) + SHAPE_OCTAVES.z * mix(n.a, MEAN_A, fine_a);
    let base = clamp((n.r - (low - 1.0)) / (2.0 - low), 0.0, 1.0);
    var shape = base;
    var tc = cover;
    if (lv(k, 1u).y == KIND_CUMULUS) {
        // A dome on the clouds' summed strength, lobed (`cloud_layers::cumulus_column`).
        let lob = 0.5 * n.g + 0.3 * n.b + 0.2 * n.a - CU_LOBE_MEAN;
        let q = clamp(env.w * (1.0 + CU_LOBE * lob) / CU_CORE_FULL, 0.0, 1.0);
        let t = env.y * sqrt(q) * (1.0 + CU_TURRET * lob);
        let b = CU_BASE_CURL * t * (1.0 - q) + CU_BASE_LUMP * lob;
        shape = shape * smoothstep(b, b + 0.04, h) * (1.0 - smoothstep(t - CU_TOP_FADE, t, h));
        // Filled in as the core rises (`cloud_layers::cumulus_cover`).
        let lo = min(cover, CU_SOLID0);
        let solid = lo + (cover - lo) * smoothstep(CU_SOLID0, CU_SOLID1, env.w);
        tc = solid + (cover - solid) * clamp((cover - CUMULUS_COVER_CEIL) / max(1.0 - CUMULUS_COVER_CEIL, 1e-3), 0.0, 1.0);
    } else {
        shape = shape * height_profile(lv(k, 1u).y, h, env.y);
    }
    let th = threshold(k, tc);
    var d = clamp((shape - th) / max(1.0 - th, 1e-3), 0.0, 1.0);
    if (d <= 0.0) {
        return 0.0;
    }
    if (detail) {
        // The erosion noise is DETAIL_RATIO times finer still.
        let gone = smoothstep(0.25, 0.5, r * DETAIL_RATIO);
        var fine = MEAN_FINE;
        // Each cloud's own thermals (`cloud_layers::convection`): its age
        // from its phase, the detail rising through one thermal and starting
        // again where its weight is 0.
        let age = env.z * lv(k, 8u).w;
        let p = clamp(env.x, 0.0, 1.0);
        let cyc = age / THERMAL_S + 0.5 * p;
        let ci = floor(cyc);
        let cf = cyc - ci;
        let sw = sin(3.14159265 * cf);
        let window = sw * sw;
        if (gone < 1.0) {
            // The churn (`cloud_layers::churn_warp`): the shape noise's lower
            // octaves as a vector field, turned with the layer's churn, its
            // reach growing as the cloud decays.
            let turn = 6.28318531 * lv(k, 2u).w;
            let c = cos(turn);
            let sn = sin(turn);
            let w = n.gba - vec3<f32>(0.5);
            let gain = WARP_GAIN0 + (WARP_GAIN1 - WARP_GAIN0) * smoothstep(0.6, 1.0, env.z);
            // And the fast churn's (`CHURN_FAST`): the octaves in another
            // order, turning CHURN_FAST times as fast.
            let fturn = turn * CHURN_FAST;
            let fc = cos(fturn);
            let fs = sin(fturn);
            let v = w.zxy * CHURN_FAST_SHARE;
            let warp = vec3<f32>(w.x * c - w.z * sn + v.x * fc - v.z * fs, w.y + v.y, w.x * sn + w.z * c + v.x * fs + v.z * fc) * (CHURN_WARP * gain);
            let wr = sh.w * (0.5 + env.y) * (RISE_EDGE + RISE_CORE * p);
            let up = wr * THERMAL_S * cf * si * DETAIL_RATIO;
            let rise = vec3<f32>(fract(ci * 0.618034), -up, fract(ci * 0.414214));
            let q = noise_at(s * DETAIL_RATIO + warp + rise, true);
            let raw = 0.625 * q.g + 0.25 * q.b + 0.125 * q.a;
            fine = mix(MEAN_FINE + (raw - MEAN_FINE) * window, MEAN_FINE, gone);
        }
        let e = fine + (1.0 - 2.0 * fine) * smoothstep(0.1, 0.6, h);
        // The turrets swell and ebb on each cloud's own age.
        let pulse = 1.0 + CHURN_PULSE * sin((age / TURRET_S + 2.0 * n.g) * 6.28318531);
        let cut = min(DETAIL_EROSION * (0.8 + 0.5 * env.z) * e * pulse, 0.95) * (1.0 - smoothstep(0.0, ERODE_DEPTH, d));
        d = clamp((d - cut) / (1.0 - cut), 0.0, 1.0);
    }
    // Frayed at the fringe, and at its margins with age (`cloud_layers::edge_gain`).
    let fringe = EDGE_FRINGE + (1.0 - EDGE_FRINGE) * smoothstep(0.0, EDGE_FRINGE_D, d);
    let ageing = 1.0 - EDGE_AGEING * smoothstep(0.6, 1.0, env.z) * (1.0 - smoothstep(0.0, EDGE_AGEING_DEPTH, d));
    return min(d * EDGE_GAIN * fringe * ageing, 1.0) * a.w;
}

fn cirrus_tau(k: u32, xz: vec2<f32>) -> f32 {
    let wd = lv(k, 3u).zw;
    let along = xz.x * wd.x + xz.y * wd.y;
    let across = -xz.x * wd.y + xz.y * wd.x;
    let s = lv(k, 1u).z * 10.0;
    let o = lv(k, 11u);
    let uv = vec2<f32>(along * s / CIRRUS_STRETCH, across * s) + o.xy;
    let f = textureSampleLevel(weather_tex, samp, uv, 0.0);
    let g = textureSampleLevel(weather_tex, samp, uv * 3.0 + vec2<f32>(0.37, 0.71), 0.0);
    let n = 0.45 * f.g + 0.3 * f.b + 0.25 * g.b;
    let tuft = textureSampleLevel(weather_tex, samp, vec2<f32>(along, across) * lv(k, 1u).z * 2.0 + o.zw, 0.0).r;
    let c = lv(k, 0u).z;
    let d = clamp((n - (1.0 - c)) / max(c, 1e-3), 0.0, 1.0) * smoothstep(0.35, 0.7, tuft);
    return d * d * CIRRUS_TAU * min(lv(k, 0u).w / EXTINCTION, 2.0);
}

// From altitude eye_alt along mu to the shell at alt (above it); < 0 when
// the ray never meets it.
fn shell_distance(eye_alt: f32, alt: f32, mu: f32) -> f32 {
    if (alt <= eye_alt) {
        return 0.0;
    }
    let r = EARTH + eye_alt;
    let c = (alt - eye_alt) * (2.0 * EARTH + alt + eye_alt);
    let rm = r * mu;
    let disc = rm * rm + c;
    if (disc < 0.0 || mu <= -0.2) {
        return -1.0;
    }
    return c / max(sqrt(disc) + rm, 1e-3);
}

fn altitude_along(eye_alt: f32, mu: f32, t: f32) -> f32 {
    let r = EARTH + eye_alt;
    return sqrt(r * r + t * t + 2.0 * r * t * mu) - EARTH;
}

fn layer_on(k: u32) -> bool {
    let a = lv(k, 0u);
    return a.z > 0.0 && a.w > 0.0;
}

// A high thin layer's depth toward the light as the ground feels it: its
// coverage, not its cells (`cloud_layers::high_layer_tau`).
fn high_layer_tau(k: u32, xz: vec2<f32>, slant: f32) -> f32 {
    let a = lv(k, 0u);
    if (lv(k, 1u).y == KIND_CIRRUS) {
        return a.z * HIGH_SHADE_FILL * CIRRUS_TAU * min(a.w / EXTINCTION, 2.0) * slant;
    }
    return envelope(k, xz, 0.0).x * HIGH_SHADE_FILL * a.w * (a.y - a.x) * slant;
}

// The column's direct transmittance toward dir from xz at alt, every layer
// (`cloud_layers::transmittance_toward`).
fn transmittance_toward(xz: vec2<f32>, alt: f32, dir: vec3<f32>) -> f32 {
    let mu = dir.y;
    if (mu <= 0.02) {
        return 1.0;
    }
    var tau = 0.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
        if (!layer_on(k)) {
            continue;
        }
        let a = lv(k, 0u);
        let kind = lv(k, 1u).y;
        if (kind == KIND_CIRRUS || kind == KIND_ALTO) {
            let t = shell_distance(alt, a.x, mu);
            if (t < 0.0) {
                continue;
            }
            tau = tau + high_layer_tau(k, xz + dir.xz * t, min(1.0 / mu, 8.0));
            continue;
        }
        let t0 = shell_distance(alt, a.x, mu);
        let t1 = shell_distance(alt, a.y, mu);
        if (t0 < 0.0 || t1 < 0.0) {
            continue;
        }
        let n = pick_steps(k, SHADOW_STEPS);
        let dt = min(t1 - t0, MAX_MARCH) / f32(n);
        for (var i = 0u; i < n; i = i + 1u) {
            let t = t0 + (f32(i) + 0.5) * dt;
            tau = tau + density(k, xz + dir.xz * t, altitude_along(alt, mu, t), false, 0.0) * dt;
        }
    }
    return exp(-tau);
}

// Steps for layer k from a vec3 of counts, by select: an array indexed at
// run time is a local array on Adreno, in scratch memory (memory
// `adreno-local-array-cliff`: 0.9-1.6 ms in sky_irradiance).
fn pick_steps(k: u32, s: vec3<u32>) -> u32 {
    return select(select(s.z, s.y, k == 1u), s.x, k == 0u);
}

fn hg(g: f32, c: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-6), 1.5));
}

fn phase(s: f32, c: f32) -> f32 {
    return (1.0 - BACK_SHARE) * hg(FORWARD_G * s, c) + BACK_SHARE * hg(BACK_G * s, c);
}

// The light's optical depth from a point in layer k toward it: taps out
// through the layer, nearer ones closer together. THE TAPS READ THE CLOUD
// AS IT IS DRAWN, eroded edges and all: read without the detail they met
// the fatter body the erosion carves, and against a path-traced reference
// a cumulus's sunlight came out 3-10 times too dark (2026-10-21,
// `tools/reference/cloud_reference.py`).
// The optical depth of the first ~200 m toward the light, from the last
// `light_tau` (a cumulus's first two taps; 0 for the two-tap layers): how
// deep in its own lobe a sample sits, which the glow dims with (`GLOW_NEAR`).
var<private> light_near: f32 = 0.0;
fn light_tau(k: u32, xz: vec2<f32>, alt: f32, foot: f32) -> f32 {
    let l = u.light_dir.xyz;
    light_near = 0.0;
    let a = lv(k, 0u);
    let thick = max(a.y - a.x, 100.0);
    var tau = 0.0;
    var last = 0.0;
    let taps = select(2u, 4u, lv(k, 1u).y == KIND_CUMULUS);
    for (var i = 0u; i < taps; i = i + 1u) {
        let j = i + 4u - taps;
        let r = select(select(select(0.8, 0.35, j == 2u), 0.15, j == 1u), 0.05, j == 0u) * thick;
        let p = vec3<f32>(xz.x, alt, xz.y) + l * r;
        tau = tau + density(k, p.xz, p.y, true, foot) * (r - last);
        last = r;
        if (j == 1u) {
            light_near = tau;
        }
    }
    return tau;
}

// The light a sample in layer k sends toward the eye per unit extinction:
// the sun's light scattered once (`ph`, the ray's phase toward it, worked
// out once per ray), and the glow of all the higher orders (`glow`, the
// same).
//
// THE GLOW (2026-10-21, against Cycles with every order of scattering, Mie
// droplets): the light scattered many times is a smooth glow over a
// cloud's body, thrown somewhat forward and hardly dimmed by the depth
// toward the sun -- 91-95% of what is seen of a cumulus from the side or
// with the sun behind. The octave approximation it replaces (Wrenninge)
// kept a fraction of it and made every cumulus grey: 0.25-0.35 of the
// reference's light where this is 0.82-0.93 (`CLOUD_GLOW`).
fn in_scatter(k: u32, alt: f32, tau_l: f32, cv: f32, ph: f32, glow: f32, d: f32) -> vec3<f32> {
    let a = lv(k, 0u);
    let h = clamp((alt - a.x) / max(a.y - a.x, 1.0), 0.0, 1.0);
    var lit = ph * exp(-tau_l) + glow * exp(-GLOW_DECAY * tau_l - GLOW_NEAR * light_near);
    // Beer-powder, mild and only away from the sun: thin outer parts have
    // built up less scattered light.
    let powder = 1.0 - exp(-POWDER_K * (2.0 * tau_l + 40.0 * d));
    lit = lit * mix(1.0, powder, POWDER_SHARE * (1.0 - cv));
    let amb = u.amb_sky.rgb * (0.45 + 0.55 * h) + u.amb_ground.rgb * (0.35 * (1.0 - h));
    return u.light.rgb * lit + amb;
}

fn map_dir(uv: vec2<f32>) -> vec3<f32> {
    let el = 0.5 * PI * pow(clamp(uv.y, 0.0, 1.0), PACK);
    let az = (uv.x - 0.5) * 2.0 * PI;
    return vec3<f32>(sin(az) * cos(el), sin(el), cos(az) * cos(el));
}

// THE SKY MAP: the clouds' light toward the eye along this texel's
// direction (rgb) and how much of the sky beyond shows (a), haze included.
struct MapOut {
    @location(0) light: vec4<f32>,
    @location(1) alt: vec4<f32>,
}

// The newest map read as the display reads it (`sky_cloud_map`): a cubic
// B-spline over a prefiltered map (`PREFILTER`) passes through its
// samples, where one bilinear tap per carry blurred the history a little
// more each map it was carried through.
fn hist_bspline(uv: vec2<f32>, size: vec2<f32>) -> vec4<f32> {
    let p = uv * size - 0.5;
    let i = floor(p);
    let f = p - i;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    let c0 = (i + 0.5 - 1.0 + w1 / g0) / size;
    var c1 = (i + 0.5 + 1.0 + w3 / g1) / size;
    let y0 = clamp(c0.y, 0.5 / size.y, 1.0 - 0.5 / size.y);
    let y1 = clamp(c1.y, 0.5 / size.y, 1.0 - 0.5 / size.y);
    let s00 = textureSampleLevel(hist_light, samp, vec2<f32>(c0.x, y0), 0.0);
    let s10 = textureSampleLevel(hist_light, samp, vec2<f32>(c1.x, y0), 0.0);
    let s01 = textureSampleLevel(hist_light, samp, vec2<f32>(c0.x, y1), 0.0);
    let s11 = textureSampleLevel(hist_light, samp, vec2<f32>(c1.x, y1), 0.0);
    let v = g0.y * (g0.x * s00 + g1.x * s10) + g1.y * (g0.x * s01 + g1.x * s11);
    return vec4<f32>(max(v.rgb, vec3<f32>(0.0)), clamp(v.a, 0.0, 1.0));
}

// Layer k's own wind (x, z): `flow_winds` holds each layer's index in .w.
fn layer_wind(k: u32) -> vec2<f32> {
    var v = hist.winds[0].yz;
    for (var i = 1u; i < 4u; i = i + 1u) {
        if (u32(hist.winds[i].w + 0.5) == k) {
            v = hist.winds[i].yz;
        }
    }
    if (u32(hist.winds[0].w + 0.5) == k) {
        v = hist.winds[0].yz;
    }
    return v;
}

// Where the newest map held what this texel shows, if it lies at altitude
// `h` and rides the wind `wv`: carried back along it (`advected_uv`).
fn hist_carry(dir: vec3<f32>, h: f32, wv: vec2<f32>) -> vec2<f32> {
    let size = vec2<f32>(u.amb_sky.w, u.light.w);
    let p = dir * cloud_flow_distance(hist.misc.w, h, max(dir.y, 0.0));
    var huv = cloud_map_uv(p + hist.flow.xyz - vec3<f32>(wv.x, 0.0, wv.y) * hist.flow.w);
    huv.y = clamp(huv.y, 0.5 / size.y, 1.0 - 0.5 / size.y);
    return huv;
}

@fragment fn fs_map(@builtin(position) pos: vec4<f32>) -> MapOut {
    let size = vec2<f32>(u.amb_sky.w, u.light.w);
    let dir = map_dir(pos.xy / size);
    let mu = dir.y;
    let alt = u.eye.y;
    let eye = u.eye.xz;
    let cv = dot(dir, u.light_dir.xyz);
    // Every sample at one fraction of its step, the same for every texel of
    // a map (`HISTORY_JITTER`): a jitter per texel, with nothing averaging
    // it, was drawn as a fine hatching through every cloud on the headset
    // (2026-10-20). One per map moves the sample planes as a whole, and the
    // accumulation below averages the planes' grain away.
    let jitter = hist.misc.z;
    let ph = phase(1.0, cv);
    let glow = GLOW_GAIN * hg(GLOW_G, cv);
    // The angle one texel spans: across (azimuth, narrowing up the sky)
    // or along (elevation, packed toward the horizon), whichever is wider.
    let v = pos.y / size.y;
    let texel = max(2.0 * PI * sqrt(max(1.0 - mu * mu, 0.0)) / size.x, 0.5 * PI * PACK * pow(max(v, 1e-3), PACK - 1.0) / size.y);
    var trans = 1.0;
    var seen = 0.0;
    var rgb = vec3<f32>(0.0);
    // The height of what is seen and the wind it rides, by its share of
    // the light (`alts`).
    var height = 0.0;
    var wind = vec2<f32>(0.0);
    var fill = 0.0;
    var fill_wind = vec2<f32>(0.0);
    for (var k = 0u; k < 4u; k = k + 1u) {
        if (fill == 0.0 && layer_on(k)) {
            fill = 0.5 * (lv(k, 0u).x + lv(k, 0u).y);
            fill_wind = layer_wind(k);
        }
    }
    for (var k = 0u; k < 3u; k = k + 1u) {
        if (!layer_on(k) || trans < 0.01) {
            continue;
        }
        let a = lv(k, 0u);
        let t0 = shell_distance(alt, a.x, mu);
        let t1 = shell_distance(alt, a.y, mu);
        if (t0 < 0.0 || t1 < 0.0) {
            continue;
        }
        let len = min(t1 - t0, MAX_MARCH);
        let lw = layer_wind(k);
        let base = pick_steps(k, VIEW_STEPS);
        // STEPS OF ONE STRIDE FROM THE LAYER'S ENTRY, AND A SHORT LAST ONE.
        // Split evenly into ceil(len / stride), every ray with the same count
        // sampled the same heights of the layer, and where the count rose by
        // one between rows every sample plane jumped: lines along the horizon
        // every few rows (research 2026-10-21, cause (b)). Now the samples
        // lie where the ray's own distance puts them, and a longer ray only
        // grows its last step.
        let stride = clamp(VIEW_STEP_TEXELS / (NOISE_TEXELS * lv(k, 1u).x), len / f32(max(base, VIEW_STEPS_MOST)), len / f32(base));
        // COARSE THROUGH THE AIR, FINE THROUGH THE CLOUD (2026-10-21): the
        // stride's sample planes showed in every cloud as ribbed stripes
        // (offline against a path-traced reference: gone at a quarter of
        // the stride). A coarse sample that finds cloud steps back to the
        // last empty one and comes in again at a REFINE-th of the stride,
        // and the march stays fine while it is in cloud and for
        // REFINE_HOLD strides after, so the cloud is sampled four times as
        // finely and the empty sky as before.
        let fine_dt = stride / f32(REFINE);
        let t_end = t0 + len;
        var t = t0 + jitter * stride;
        var hold = 0u;
        var steps = 0u;
        // The light toward the sun, marched every LIGHT_EVERY-th fine step in
        // cloud and kept between (its depth changes over hundreds of metres,
        // a fine step is tens): a third of the map's cost was these taps.
        var lit_for = 0u;
        var tau_kept = 0.0;
        var near_kept = 0.0;
        // The last empty sample: where a fine march into cloud came from;
        // and the halvings left of a search for the surface (`ENTRY_BISECT`),
        // between `lo` (empty) and `hi` (in cloud, density `d_hi`).
        var empty_t = t0;
        var was_empty = true;
        var bis = 0u;
        var lo = 0.0;
        var hi = 0.0;
        var d_hi = 0.0;
        loop {
            if (t >= t_end || steps >= MARCH_CAP) {
                break;
            }
            steps = steps + 1u;
            let fine = hold > 0u;
            let dt = min(select(stride, fine_dt, fine), t_end - t);
            let foot = max(t * texel, 0.25 * dt);
            // One density a pass round the loop, the search's included: a
            // second call site doubled the shader and tripled the map's
            // cost on the Quest (1.33 -> 4.28 ms a slice, 2026-10-22).
            let at = select(t, 0.5 * (lo + hi), bis > 0u);
            var d = density(k, eye + dir.xz * at, altitude_along(alt, mu, at), true, foot);
            // THE MARCH STEPS FROM THE CLOUD'S OWN SURFACE (2026-10-22). Its
            // first sample in cloud lay anywhere in a fine step past the
            // surface, set by the ray's distance alone, and at a cumulus's
            // extinction (an optical depth near 1 a step) that sample is
            // most of what the eye sees: the surfaces came out hatched with
            // the sample shells (headset, user: "striation"). Found to a
            // quarter of a step between the empty sample and this one, the
            // march goes on from there, so neighbouring rays sample a cloud
            // alike wherever it stands.
            if (bis > 0u) {
                if (d > 0.0) {
                    hi = at;
                    d_hi = d;
                } else {
                    lo = at;
                }
                bis = bis - 1u;
                if (bis > 0u) {
                    continue;
                }
                t = hi;
                d = d_hi;
            } else if (d > 0.0 && !fine) {
                empty_t = max(t - stride, t0);
                was_empty = true;
                t = empty_t + jitter * fine_dt;
                hold = REFINE_HOLD * REFINE;
                continue;
            } else if (d > 0.0 && was_empty && ENTRY_BISECT > 0u) {
                lo = empty_t;
                hi = t;
                d_hi = d;
                bis = ENTRY_BISECT;
                continue;
            }
            let p_alt = altitude_along(alt, mu, t);
            let xz = eye + dir.xz * t;
            was_empty = d <= 0.0;
            if (was_empty) {
                empty_t = t;
            }
            if (d > 0.0) {
                hold = REFINE_HOLD * REFINE;
                if (lit_for == 0u) {
                    tau_kept = light_tau(k, xz, p_alt, foot);
                    near_kept = light_near;
                    lit_for = LIGHT_EVERY;
                }
                lit_for = lit_for - 1u;
                light_near = near_kept;
                let s = in_scatter(k, p_alt, tau_kept, cv, ph, glow, d);
                let ts = exp(-d * dt);
                let w = trans * (1.0 - ts) * haze_through(t, mu);
                rgb = rgb + w * s;
                seen = seen + w;
                height = height + w * p_alt;
                wind = wind + w * lw;
                trans = trans * ts;
                if (trans < 0.01) {
                    break;
                }
            } else if (fine) {
                hold = hold - 1u;
                lit_for = 0u;
            }
            t = t + dt;
        }
    }
    // Cirrus: ice, a thin sheet, its light thrown forward.
    if (layer_on(3u) && trans > 0.01) {
        let t = shell_distance(alt, lv(3u, 0u).x, mu);
        if (t >= 0.0) {
            let tau = cirrus_tau(3u, eye + dir.xz * t) * min(1.0 / max(mu, 1e-3), 8.0);
            let alpha = 1.0 - exp(-tau);
            let s = u.light.rgb * (0.7 * hg(0.85, cv) + 0.3 * hg(0.2, cv)) + u.amb_sky.rgb * 0.8;
            let w = trans * alpha * haze_through(t, mu);
            rgb = rgb + w * s;
            seen = seen + w;
            height = height + w * lv(3u, 0u).x;
            wind = wind + w * layer_wind(3u);
            trans = trans * (1.0 - alpha);
        }
    }
    var light = vec4<f32>(rgb * u.eye.w, 1.0 - seen);
    // .g: the clouds' own transmittance, no haze -- what a disc behind them
    // gets (`cloud_through` in the sky shader).
    let h = (height + ALT_FILL_WEIGHT * fill) / (seen + ALT_FILL_WEIGHT);
    // THE WIND WHAT IS SEEN RIDES, not the height's (2026-10-23). Carried on
    // a wind blended between the layers' middles by height, a cumulus seen
    // low in its layer rode half the stratocumulus's wind, ran ahead of or
    // behind itself through a map and was pulled back by the next: the
    // clouds stepped "3 forward, 1-2 back" (user, headset). Each layer's
    // own wind, by its share of the light, is how what this texel shows moves.
    let hv = (wind + ALT_FILL_WEIGHT * fill_wind) / (seen + ALT_FILL_WEIGHT);
    // The march stops once less than 1% gets through, and the 1% it left
    // let the sun's disc -- some 1e5 times the cloud -- shine through thick
    // cumulus as a white dot (user: the sun and moon shone "as if they are in
    // front of the cloud", 2026-10-22). Past that point the cloud only goes
    // on, so a disc behind it gets nothing. Nor does it through a cumulus
    // edge of optical depth 3-4: a few per cent of the sun is still 1e3
    // times the cloud, a hard white dot, where a real edge reads as a
    // silver lining and no disc -- the forward scatter that swamps it is
    // finer than this map. The disc fades in through thinner cloud.
    let through = trans * smoothstep(DISC_THROUGH0, DISC_THROUGH1, trans);
    var alts = vec4<f32>(h, through, hv);
    // THE NEWEST MAP, CARRIED HERE: what this texel shows, where that map
    // held it (the sky shader's own carry, `sky_clouds`), mixed in.
    if (hist.misc.x < 1.0) {
        var huv = hist_carry(dir, h, hv);
        var old_alts = textureSampleLevel(hist_alt, samp, huv, 0.0);
        // CARRIED BY ITS OWN ALTITUDE (R1, cloud research 2026-10-22): the
        // first carry moves the old map by this march's altitude, which at an
        // edge is the wrong cloud's (or the empty sky's fill), so the old
        // cloud was fetched from where it no longer is and left a ghost
        // copy beside the new edge. Carried again by the altitude the old map
        // holds there, it lands where that cloud has moved to.
        if (HISTORY_REFETCH) {
            let again = hist_carry(dir, old_alts.r, old_alts.ba);
            let again_alts = textureSampleLevel(hist_alt, samp, again, 0.0);
            if (abs(again_alts.r - old_alts.r) < HISTORY_ALT_AGREE * max(old_alts.r, 1.0)) {
                huv = again;
                old_alts = again_alts;
            }
        }
        let old = hist_bspline(huv, size);
        let above = smoothstep(HISTORY_HORIZON_UP0, HISTORY_HORIZON_UP1, dir.y);
        let reject = mix(hist.reject.zw, hist.reject.xy, above);
        let apart = abs(old.a - light.a);
        let cloud = max(1.0 - min(old.a, light.a), hist.share.z);
        let let_go = max(smoothstep(reject.x, reject.y, apart), hist.share.w * above * smoothstep(hist.share.x, hist.share.y, apart / cloud));
        let w = mix(hist.misc.x, 1.0, let_go);
        light = mix(vec4<f32>(old.rgb * hist.misc.y, old.a), light, w);
        alts = mix(old_alts, alts, w);
    }
    var out: MapOut;
    out.light = light;
    out.alt = alts;
    return out;
}

// THE GROUND'S SHADOW: the light every layer lets through toward the light
// from this texel's patch of ground. The outer ring holds the map's mean,
// for the ground past it (`sky::cloud_map_border`).
@fragment fn fs_shadow(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = {SHADOW_SIZE};
    if (pos.x < 1.0 || pos.y < 1.0 || pos.x > size - 1.0 || pos.y > size - 1.0) {
        return vec4<f32>(u.amb_ground.w);
    }
    let step = u.shadow.z / size;
    let xz = u.shadow.xy - 0.5 * u.shadow.z + pos.xy * step;
    // Four rays across the light's disc: the penumbra
    // (`cloud_layers::transmittance_penumbra`).
    let l = u.light_dir.xyz;
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(l.y) < 0.9);
    let a = normalize(cross(l, up));
    let b = cross(l, a);
    var sum = 0.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let ang = (f32(k) + 0.5) * 0.5 * PI;
        let d = normalize(l + SUN_RADIUS * 0.7 * (cos(ang) * a + sin(ang) * b));
        sum = sum + transmittance_toward(xz, u.shadow.w, d);
    }
    return vec4<f32>(0.25 * sum);
}
"#;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use space_soup_sky::clouds::CloudParams;

    pub(crate) fn read_back(device: &Device, queue: &Queue, texture: &Texture, layer: u32, width: u32, height: u32, bytes_per_texel: u32) -> Vec<u8> {
        let row = (width * bytes_per_texel).div_ceil(256) * 256;
        let buf = device.create_buffer(&BufferDescriptor { label: None, size: (row * height) as u64, usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            TexelCopyTextureInfo { texture, mip_level: 0, origin: Origin3d { x: 0, y: 0, z: layer }, aspect: TextureAspect::All },
            TexelCopyBufferInfo { buffer: &buf, layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(height) } },
            Extent3d { width, height, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        (0..height).flat_map(|y| data[(y * row) as usize..(y * row + width * bytes_per_texel) as usize].to_vec()).collect()
    }

    pub(crate) fn half(b: &[u8]) -> f32 {
        let h = u16::from_le_bytes([b[0], b[1]]);
        let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
        let v = if e == 0 { m / 1024.0 * 2f32.powi(-14) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) };
        if s == 1 { -v } else { v }
    }

    /// THE GPU'S CLOUDS ARE THE CPU'S: the shadow map's texels against the
    /// twin's columns, and a whole sky map with clouds in it, sky between
    /// them, and none past the horizon.
    #[test]
    fn the_gpu_clouds_match_their_twin() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let scope = device.push_error_scope(ErrorFilter::Validation);
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.5, ..Default::default() };
        let layers = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise).layers(300.0);
        let mut sky = CloudSky::new(&device, &queue, &field, &noise);
        let size = crate::renderer::uniforms::CLOUD_SHADOW_SIZE;
        let shadow = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let sun = [0.35f32, 0.85, 0.4];
        let (centre, extent) = ([120.0f32, -40.0], 4096.0);
        let u = CloudSkyUniform::new(&layers, [10.0, 2.0, 5.0], sun, [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], centre, extent, 0.5);
        let mut enc = device.create_command_encoder(&Default::default());
        sky.encode(&mut enc, &queue, &u, Some(&shadow.create_view(&Default::default())), true);
        queue.submit(Some(enc.finish()));
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("the cloud passes do not validate: {e}");
        }
        let texels = read_back(&device, &queue, &shadow, 0, size, size, 1);
        let dir = { let l = (sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]).sqrt(); sun.map(|v| v / l) };
        let step = extent / size as f32;
        let (mut worst, mut sum, mut n, mut shaded) = (0.0f32, 0.0f32, 0, 0);
        for j in 1..size - 1 {
            for i in 1..size - 1 {
                let xz = [centre[0] - 0.5 * extent + (i as f32 + 0.5) * step, centre[1] - 0.5 * extent + (j as f32 + 0.5) * step];
                let cpu = space_soup_sky::cloud_layers::transmittance_penumbra(&layers, &noise, &field, xz, 2.0, dir);
                let gpu = texels[(j * size + i) as usize] as f32 / 255.0;
                worst = worst.max((cpu - gpu).abs());
                sum += (cpu - gpu).abs();
                n += 1;
                shaded += (gpu < 0.5) as u32;
            }
        }
        eprintln!("shadow vs twin: mean |diff| {:.4}, worst {worst:.3}; {shaded} of {n} shaded", sum / n as f32);
        assert!(sum / (n as f32) < 0.02 && worst < 0.25, "the GPU shadow is not the twin's: mean {}, worst {worst}", sum / n as f32);
        assert!(shaded > n / 20 && shaded < n * 19 / 20, "{shaded} of {n} shaded");
        assert_eq!(texels[0], 128, "the border holds the mean");

        let map = read_back(&device, &queue, &sky.maps, sky.display()[1] as u32, MAP_WIDTH, MAP_HEIGHT, 8);
        let alpha: Vec<f32> = (0..(MAP_WIDTH * MAP_HEIGHT) as usize).map(|i| half(&map[i * 8 + 6..i * 8 + 8])).collect();
        let at = |x: u32, y: u32| alpha[(y * MAP_WIDTH + x) as usize];
        let rows = |y0: u32, y1: u32| {
            let v: Vec<f32> = (y0..y1).flat_map(|y| (0..MAP_WIDTH).step_by(4).map(move |x| at(x, y))).collect();
            (v.iter().filter(|&&a| a < 0.5).count() as f32 / v.len() as f32, v.iter().filter(|&&a| a > 0.95).count() as f32 / v.len() as f32)
        };
        let (cloudy, clear) = rows(MAP_HEIGHT / 3, MAP_HEIGHT);
        eprintln!("sky map above 20 deg: {:.0}% cloud, {:.0}% clear", cloudy * 100.0, clear * 100.0);
        assert!(cloudy > 0.05 && clear > 0.05, "the sky map is {cloudy} cloud, {clear} clear");
        let rgb_ok = (0..(MAP_WIDTH * MAP_HEIGHT) as usize).step_by(97).all(|i| (0..3).all(|c| { let v = half(&map[i * 8 + 2 * c..i * 8 + 2 * c + 2]); v.is_finite() && v >= 0.0 }));
        assert!(rgb_ok, "the sky map holds a negative or non-finite light");
        // The map's direction and lookup are each other's inverse.
        for &d in &[[0.3f32, 0.2, -0.9], [0.0, 0.99, 0.1], [-0.7, 0.05, 0.7]] {
            let back = map_dir(map_uv(d));
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!((0..3).all(|k| (back[k] - d[k] / l).abs() < 1e-4), "{d:?} -> {back:?}");
        }
    }

    /// ONE CUMULUS LAYER FOR A PATH-TRACED REFERENCE (`--ignored`): the sky
    /// map as the headset builds it, lit by the sun alone (irradiance 1, no
    /// sky or ground light), and the same layer's extinction as a grid of
    /// voxels read from the CPU twin, for `tools/reference/cloud_reference.py`
    /// to render in Cycles under the same sun. `$OUT/cloud_map.f32` (rgb as
    /// radiance, a = transmittance; rows from the horizon up),
    /// `$OUT/cloud_grid.f32` ([x][y][z], per metre) and `$OUT/cloud.json`.
    /// `SUN_EL` / `SUN_AZ` degrees (35, 0: toward +z), `VOXEL` metres (20),
    /// `REACH` metres either side of the eye (7000).
    #[test]
    #[ignore]
    fn export_a_cumulus_for_a_path_traced_reference() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let (el, az) = (env("SUN_EL", 35.0).to_radians(), env("SUN_AZ", 0.0).to_radians());
        let sun = [az.sin() * el.cos(), el.sin(), az.cos() * el.cos()];
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.45, altitude_m: 1200.0, wind_speed: 16.0, time_scale: 2.5, ..Default::default() };
        let t = 20_000.0;
        let mut population = CloudPopulation::new();
        let mut layers = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise).layers(t);
        population.update(&params, &field, t);
        population.place(&mut layers, t);
        // One layer alone: `LAYER=k`, the cumulus by default.
        let k = std::env::var("LAYER").ok().and_then(|v| v.parse().ok())
            .unwrap_or_else(|| layers.iter().position(|l| l.kind == space_soup_sky::cloud_layers::KIND_CUMULUS).unwrap_or(0));
        for (i, l) in layers.iter_mut().enumerate() {
            if i != k {
                l.sigma = 0.0;
            }
        }
        let eye = [0.0f32, 2.0, 0.0];
        let u = CloudSkyUniform::new(&layers, eye, sun, [1.0; 3], [0.0; 3], [0.0; 3], [0.0, 0.0], 4096.0, 0.5);
        let maps: u32 = std::env::var("MAPS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let build_map = |src: String, file: &str| {
            let scope = device.push_error_scope(ErrorFilter::Validation);
            let mut sky = CloudSky::with_shader(&device, &queue, &field, &noise, src);
            sky.set_population(&device, &queue, &population);
            let mut enc = device.create_command_encoder(&Default::default());
            sky.encode(&mut enc, &queue, &u, None, true);
            queue.submit(Some(enc.finish()));
            // `MAPS=n`: then n maps more, a slice a frame, each carrying the
            // last as the headset's do (`HISTORY_NEW`).
            for _ in 0..maps * MAP_SLICES {
                let mut enc = device.create_command_encoder(&Default::default());
                sky.encode(&mut enc, &queue, &u, None, false);
                queue.submit(Some(enc.finish()));
            }
            if let Some(e) = pollster::block_on(scope.pop()) {
                panic!("{file}: the cloud passes do not validate: {e}");
            }
            let map = read_back(&device, &queue, &sky.maps, sky.display()[1] as u32, MAP_WIDTH, MAP_HEIGHT, 8);
            let scale = sky.display_scales()[1];
            let mut texels = Vec::with_capacity((MAP_WIDTH * MAP_HEIGHT * 4) as usize);
            for i in 0..(MAP_WIDTH * MAP_HEIGHT) as usize {
                for c in 0..4 {
                    let v = half(&map[i * 8 + 2 * c..i * 8 + 2 * c + 2]);
                    texels.push(if c < 3 { v * scale } else { v });
                }
            }
            std::fs::write(out.join(file), bytemuck::cast_slice(&texels)).unwrap();
        };
        // `FITS=file.json`: {"name": {"CONST": value, ...}, ...} -- the map
        // again with those shader constants, as `cloud_map_<name>.f32`, and
        // no grid (`cloud_reference.py --fits`).
        if let Ok(fits) = std::env::var("FITS") {
            let fits: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(fits).unwrap()).unwrap();
            for (name, consts) in &fits {
                let mut src = cloud_sky_shader();
                for (k, v) in consts.as_object().unwrap() {
                    // "__replace": [[from, to], ...] -- source edits.
                    if k == "__replace" {
                        for pair in v.as_array().unwrap() {
                            let (from, to) = (pair[0].as_str().unwrap(), pair[1].as_str().unwrap());
                            assert!(src.contains(from), "{name}: no {from:?} in the shader");
                            src = src.replace(from, to);
                        }
                        continue;
                    }
                    let at = src.find(&format!("const {k}: ")).unwrap_or_else(|| panic!("no constant {k}"));
                    let eq = at + src[at..].find("= ").unwrap() + 2;
                    let end = at + src[at..].find(';').unwrap();
                    let value = match v {
                        serde_json::Value::String(s) => s.clone(),
                        v => v.as_f64().unwrap().to_string(),
                    };
                    src.replace_range(eq..end, &value);
                }
                build_map(src, &format!("cloud_map_{name}.f32"));
            }
            eprintln!("{} fitted maps", fits.len());
            return;
        }
        build_map(cloud_sky_shader(), "cloud_map.f32");

        let l = &layers[k];
        let (voxel, reach) = (env("VOXEL", 20.0), env("REACH", 7000.0));
        let (nx, ny) = ((2.0 * reach / voxel) as usize, ((l.top_m - l.base_m) / voxel).ceil() as usize + 1);
        let weather = space_soup_sky::cloud_layers::CloudWeather::new(&field, Some(&population));
        // `DETAIL=0`: the cloud bodies without their eroded edges, which a
        // grid of voxels resolves (and `"__replace"` the map's to match).
        let detail = std::env::var("DETAIL").as_deref() != Ok("0");
        let mut grid = vec![0.0f32; nx * ny * nx];
        let started = std::time::Instant::now();
        std::thread::scope(|s| {
            for (ix, slab) in grid.chunks_mut(ny * nx).enumerate() {
                let weather = weather;
                let noise = &noise;
                s.spawn(move || {
                    let x = -reach + (ix as f32 + 0.5) * voxel;
                    for iy in 0..ny {
                        let y = l.base_m + iy as f32 * voxel;
                        for iz in 0..nx {
                            let z = -reach + (iz as f32 + 0.5) * voxel;
                            slab[iy * nx + iz] = space_soup_sky::cloud_layers::density(l, noise, weather, [x, z], y, detail);
                        }
                    }
                });
            }
        });
        std::fs::write(out.join("cloud_grid.f32"), bytemuck::cast_slice(&grid)).unwrap();
        let filled = grid.iter().filter(|&&d| d > 0.0).count();
        let meta = serde_json::json!({
            "eye": eye, "sun": sun, "sun_irradiance": 1.0,
            "map": { "width": MAP_WIDTH, "height": MAP_HEIGHT, "pack": MAP_PACK },
            "grid": { "origin": [-reach, l.base_m, -reach], "voxel": voxel, "dims": [nx, ny, nx] },
            "haze_per_m": space_soup_sky::clouds::HAZE_PER_M,
            "phase": { "forward_g": space_soup_sky::clouds::PHASE_FORWARD_G, "back_g": space_soup_sky::clouds::PHASE_BACK_G, "back_share": space_soup_sky::clouds::PHASE_BACK_SHARE },
            "layer": { "index": k, "kind": l.kind, "coverage": l.coverage, "base_m": l.base_m, "top_m": l.top_m, "sigma": l.sigma },
        });
        std::fs::write(out.join("cloud.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
        eprintln!("cumulus {nx}x{ny}x{nx} voxels of {voxel} m, {filled} in cloud, built in {:.1} s; base {} top {} sigma {}", started.elapsed().as_secs_f32(), l.base_m, l.top_m, l.sigma);
    }

    /// What a whole sky map costs on this machine's GPU, with the layers'
    /// populations and with the weather's mean alone (`--ignored`).
    #[test]
    #[ignore]
    fn time_the_sky_map() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.45, ..Default::default() };
        let t = 20_000.0;
        let base = cloud_sky_shader();
        let variant = |name: &str| -> String {
            match name {
                "no light taps" => base.replace("fn light_tau(k: u32, xz: vec2<f32>, alt: f32, foot: f32) -> f32 {", "fn light_tau(k: u32, xz: vec2<f32>, alt: f32, foot: f32) -> f32 {\n    if (true) { return 0.0; }"),
                "no detail" => base.replace("altitude_along(alt, mu, at), true, foot);", "altitude_along(alt, mu, at), false, foot);"),
                "entry bisection" => base.replace("const ENTRY_BISECT: u32 = 0u;", "const ENTRY_BISECT: u32 = 2u;"),
                "three halvings" => base.replace("const ENTRY_BISECT: u32 = 0u;", "const ENTRY_BISECT: u32 = 3u;"),
                "no refine" => base.replace("const REFINE: u32 = 4u;", "const REFINE: u32 = 1u;"),
                "light every step" => base.replace("const LIGHT_EVERY: u32 = 2u;", "const LIGHT_EVERY: u32 = 1u;"),
                "half steps" => base.replace(&format!("const VIEW_STEPS_MOST: u32 = {}u;", cl::VIEW_STEPS_MOST), &format!("const VIEW_STEPS_MOST: u32 = {}u;", cl::VIEW_STEPS_MOST / 2)),
                _ => base.clone(),
            }
        };
        for (populated, name) in [(false, "as built"), (true, "as built"), (true, "entry bisection"), (true, "three halvings"), (true, "no refine"), (true, "light every step"), (true, "no light taps"), (true, "no detail"), (true, "half steps")] {
            let src = variant(name);
            assert!(name == "as built" || src != base, "variant {name} changed nothing");
            let mut population = CloudPopulation::new();
            let mut layers = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise).layers(t);
            let mut sky = CloudSky::with_shader(&device, &queue, &field, &noise, src);
            if populated {
                population.update(&params, &field, t);
                population.place(&mut layers, t);
                sky.set_population(&device, &queue, &population);
            }
            let u = CloudSkyUniform::new(&layers, [10.0, 2.0, 5.0], [0.35, 0.85, 0.4], [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], [0.0, 0.0], 4096.0, 0.5);
            let run = |sky: &mut CloudSky, n: u32| {
                let start = std::time::Instant::now();
                for _ in 0..n {
                    let mut enc = device.create_command_encoder(&Default::default());
                    sky.encode(&mut enc, &queue, &u, None, true);
                    queue.submit(Some(enc.finish()));
                }
                device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
                start.elapsed().as_secs_f64() * 1000.0 / n as f64
            };
            run(&mut sky, 2);
            let ms = run(&mut sky, 10);
            eprintln!("{name}, populated {populated}: a whole map {ms:.2} ms, a slice {:.3} ms", ms / MAP_SLICES as f64);
        }
    }

    /// What one headset frame's clouds cost: the next slice as `step_clouds`
    /// draws it, alone and with the ground's shadow (`--ignored`).
    #[test]
    #[ignore]
    fn time_a_frames_clouds() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.45, ..Default::default() };
        let t = 20_000.0;
        let mut population = CloudPopulation::new();
        let mut layers = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise).layers(t);
        population.update(&params, &field, t);
        population.place(&mut layers, t);
        let mut sky = CloudSky::new(&device, &queue, &field, &noise);
        sky.set_population(&device, &queue, &population);
        let size = crate::renderer::uniforms::CLOUD_SHADOW_SIZE;
        let shadow = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = shadow.create_view(&Default::default());
        let u = CloudSkyUniform::new(&layers, [10.0, 2.0, 5.0], [0.35, 0.85, 0.4], [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], [0.0, 0.0], 4096.0, 0.5);
        let mut frame = 0u32;
        let mut run = |sky: &mut CloudSky, shadow: Option<&TextureView>, n: u32| {
            let start = std::time::Instant::now();
            for _ in 0..n {
                frame += 1;
                let mut enc = device.create_command_encoder(&Default::default());
                sky.encode_at(&mut enc, &queue, &u, &u, shadow, false, frame as f64 / 36.0, frame as f64 / 36.0);
                queue.submit(Some(enc.finish()));
                device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
            }
            start.elapsed().as_secs_f64() * 1000.0 / n as f64
        };
        run(&mut sky, None, MAP_SLICES);
        let n = 4 * MAP_SLICES;
        let alone = run(&mut sky, None, n);
        let with = run(&mut sky, Some(&view), n);
        // The same frame with no march (the passes, their targets, the
        // prefilter and the submit): what a slice costs before its rays.
        let bare_src = cloud_sky_shader().replace("for (var k = 0u; k < 3u; k = k + 1u) {\n        if (!layer_on(k) || trans < 0.01) {", "for (var k = 0u; k < 0u; k = k + 1u) {\n        if (!layer_on(k) || trans < 0.01) {");
        assert_ne!(bare_src, cloud_sky_shader(), "the bare variant changed nothing");
        let mut bare = CloudSky::with_shader(&device, &queue, &field, &noise, bare_src);
        bare.set_population(&device, &queue, &population);
        run(&mut bare, None, MAP_SLICES);
        let bare_ms = run(&mut bare, None, n);
        eprintln!("a frame's clouds with no march: {bare_ms:.3} ms");
        let empty = {
            let start = std::time::Instant::now();
            for _ in 0..n {
                queue.submit(Some(device.create_command_encoder(&Default::default()).finish()));
                device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
            }
            start.elapsed().as_secs_f64() * 1000.0 / n as f64
        };
        eprintln!("a frame's clouds: slice {alone:.3} ms, slice + ground shadow {with:.3} ms, an empty submit {empty:.3} ms");
    }

    /// A map's texel read between its four nearest, as the GPU's linear
    /// sampler does: wrapping round in azimuth, held at the horizon and the
    /// zenith.
    fn bilinear(texels: &[f32], uv: [f32; 2]) -> f32 {
        let (w, h) = (MAP_WIDTH as i32, MAP_HEIGHT as i32);
        let x = uv[0] * w as f32 - 0.5;
        let y = (uv[1] * h as f32 - 0.5).clamp(0.0, (h - 1) as f32);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |i: i32, j: i32| texels[(j.clamp(0, h - 1) * w + i.rem_euclid(w)) as usize];
        let (i, j) = (x0 as i32, y0 as i32);
        let top = at(i, j) + (at(i + 1, j) - at(i, j)) * fx;
        let bottom = at(i, j + 1) + (at(i + 1, j + 1) - at(i, j + 1)) * fx;
        top + (bottom - top) * fy
    }

    /// A map drawn of the sky one moment and from one eye, then each frame
    /// moved from there to here and now (wind, walking), shows the clouds
    /// where a map drawn here and now has them -- far nearer than the old map
    /// left where it was -- and the heights beside it are the layers'.
    #[test]
    fn a_map_carried_on_its_winds_shows_the_clouds_where_a_new_map_has_them() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let scope = device.push_error_scope(ErrorFilter::Validation);
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.5, ..Default::default() };
        let model = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise);
        let mut population = CloudPopulation::new();
        let mut draw = |t: f64, eye: [f32; 3]| {
            population.update(&params, &field, t);
            let mut layers = model.layers(t);
            population.place(&mut layers, t);
            // The churn held at the first map's turn: this measures the
            // carrying, not the shapes' own change (`churn_warp`).
            for (l, held) in layers.iter_mut().zip(model.layers(20_000.0)) {
                l.churn = held.churn;
            }
            let mut sky = CloudSky::new(&device, &queue, &field, &noise);
            sky.set_population(&device, &queue, &population);
            sky.set_winds(flow_winds(&params, &layers));
            let u = CloudSkyUniform::new(&layers, eye, [0.35, 0.85, 0.4], [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], [0.0, 0.0], 4096.0, 0.5);
            let mut enc = device.create_command_encoder(&Default::default());
            sky.encode_at(&mut enc, &queue, &u, &u, None, true, t, t);
            queue.submit(Some(enc.finish()));
            let layer = sky.display()[1] as u32;
            let map = read_back(&device, &queue, &sky.maps, layer, MAP_WIDTH, MAP_HEIGHT, 8);
            let alts = read_back(&device, &queue, &sky.alts, layer, MAP_WIDTH, MAP_HEIGHT, 8);
            let n = (MAP_WIDTH * MAP_HEIGHT) as usize;
            let alpha: Vec<f32> = (0..n).map(|i| half(&map[i * 8 + 6..i * 8 + 8])).collect();
            let alt: Vec<[f32; 3]> = (0..n).map(|i| [0, 4, 6].map(|o| half(&alts[i * 8 + o..i * 8 + o + 2]))).collect();
            (alpha, alt, layers)
        };
        let (t0, t1) = (20_000.0, 20_015.0);
        let (e0, e1) = ([10.0f32, 2.0, 5.0], [130.0f32, 2.0, -80.0]);
        let (old, _, _) = draw(t0, e0);
        let (new, new_alt, layers) = draw(t1, e1);
        let new_alt: [Vec<f32>; 3] = std::array::from_fn(|c| new_alt.iter().map(|t| t[c]).collect());
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("the cloud passes do not validate: {e}");
        }
        let winds = flow_winds(&params, &layers);
        assert!(winds.windows(2).all(|w| w[0][0] <= w[1][0]), "winds not by height: {winds:?}");
        let delta = [e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2]];
        let (mut carried, mut left, mut n) = (0.0f32, 0.0f32, 0);
        let lo = layers.iter().map(|l| l.base_m).fold(f32::MAX, f32::min);
        let hi = layers.iter().map(|l| l.top_m).fold(0.0, f32::max);
        let mut heights_ok = true;
        for j in (MAP_HEIGHT / 4..MAP_HEIGHT * 15 / 16).step_by(7) {
            for i in (0..MAP_WIDTH).step_by(11) {
                let uv = [(i as f32 + 0.5) / MAP_WIDTH as f32, (j as f32 + 0.5) / MAP_HEIGHT as f32];
                let want = bilinear(&new, uv);
                let [alt, vx, vz] = [0, 1, 2].map(|c| bilinear(&new_alt[c], uv));
                heights_ok &= alt == 0.0 || (alt >= lo - 1.0 && alt <= hi + 1.0);
                let moved = advected_uv(map_dir(uv), alt, e1[1], [vx, vz], delta, (t1 - t0) as f32);
                let still = bilinear(&old, uv);
                if (want - still).abs() < 0.02 && want > 0.98 {
                    continue;
                }
                carried += (bilinear(&old, moved) - want).abs();
                left += (still - want).abs();
                n += 1;
            }
        }
        let (carried, left) = (carried / n as f32, left / n as f32);
        eprintln!("{n} directions: the old map carried is off by {carried:.4}, left where it was {left:.4}");
        assert!(heights_ok, "a map height lies outside the layers' {lo}..{hi}");
        assert!(n > 500, "too few clouded directions: {n}");
        assert!(carried < 0.35 * left, "carrying the map does not follow the clouds: {carried} vs {left}");
    }

    /// HOW FAR THE SHOWN CLOUDS STEP BACK at a map swap: a map one build
    /// old, carried to the next map's moment as the sky shader carries it,
    /// against that next map, patch by patch. The shift that best lays one
    /// on the other is how far a cloud jumps while the display blends from
    /// the old map to the new (user, 2026-10-23: "3 steps forward, 1-2 steps
    /// back"). `STEP_DT` the build (cloud s, 6.7: a slice a frame at 36 Hz,
    /// 2.5x clock), `STEP_WIND=height` carries on the height's wind as before,
    /// `STEP_POP=hold` keeps one population (no drift, no growth),
    /// `STEP_CHURN=hold` holds the churn, `STEP_KINDS=0,0,0,1` the kinds.
    /// Carried on the height's wind with the cirrus creeping on the
    /// weather's: median 0.71 texels, p90 4.03, the cirrus alone 2.5 (and
    /// its best shift took 66% of the misfit). On each texel's own wind with
    /// the cirrus on its wind: median 0.50, p90 1.50, cirrus 0; what is left
    /// is the clouds' own drift and growth (0 with the population held).
    /// `--ignored`.
    #[test]
    #[ignore]
    fn carried_maps_step_back_by() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let mut params = CloudParams { coverage: 0.35, wind_speed: 16.0, time_scale: 2.5, ..Default::default() };
        params.kinds = [1.0, 1.0, 0.0, 1.0];
        if let Some(k) = std::env::var("STEP_KINDS").ok() {
            let v: Vec<f32> = k.split(',').map(|x| x.parse().unwrap()).collect();
            params.kinds = [v[0], v[1], v[2], v[3]];
        }
        let model = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise);
        let env = |k: &str| std::env::var(k).ok();
        let dt: f64 = env("STEP_DT").and_then(|v| v.parse().ok()).unwrap_or(6.7);
        let hold_pop = env("STEP_POP").is_some();
        let hold_churn = env("STEP_CHURN").is_some();
        let by_height = env("STEP_WIND").is_some();
        let mut population = CloudPopulation::new();
        let eye = [10.0f32, 2.0, 5.0];
        let t0 = 20_000.0;
        let mut draw = |t: f64| {
            if !hold_pop || population.generation() == 0 {
                population.update(&params, &field, t);
            }
            let mut layers = model.layers(t);
            population.place(&mut layers, t);
            if hold_churn {
                for (l, held) in layers.iter_mut().zip(model.layers(t0)) {
                    l.churn = held.churn;
                }
            }
            let mut sky = CloudSky::new(&device, &queue, &field, &noise);
            sky.set_population(&device, &queue, &population);
            sky.set_winds(flow_winds(&params, &layers));
            let u = CloudSkyUniform::new(&layers, eye, [0.35, 0.85, 0.4], [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], [0.0, 0.0], 4096.0, 0.5);
            let mut enc = device.create_command_encoder(&Default::default());
            sky.encode_at(&mut enc, &queue, &u, &u, None, true, t, t);
            queue.submit(Some(enc.finish()));
            let layer = sky.display()[1] as u32;
            let map = read_back(&device, &queue, &sky.maps, layer, MAP_WIDTH, MAP_HEIGHT, 8);
            let alts = read_back(&device, &queue, &sky.alts, layer, MAP_WIDTH, MAP_HEIGHT, 8);
            let n = (MAP_WIDTH * MAP_HEIGHT) as usize;
            let alpha: Vec<f32> = (0..n).map(|i| half(&map[i * 8 + 6..i * 8 + 8])).collect();
            let ch: [Vec<f32>; 3] = [0, 4, 6].map(|o| (0..n).map(|i| half(&alts[i * 8 + o..i * 8 + o + 2])).collect());
            (alpha, ch, flow_winds(&params, &layers))
        };
        let (old, _, _) = draw(t0);
        let (new, alt, winds) = draw(t0 + dt);
        const P: i32 = 32;
        let (mut shifts, mut gains) = (Vec::new(), Vec::new());
        let mut mean = [0.0f64; 2];
        for pj in (MAP_HEIGHT as i32 / 8..MAP_HEIGHT as i32 * 7 / 8).step_by(P as usize) {
            for pi in (0..MAP_WIDTH as i32).step_by(P as usize) {
                let mut texels = Vec::new();
                let (mut lo, mut hi) = (1.0f32, 0.0f32);
                for j in pj..pj + P {
                    for i in pi..pi + P {
                        let uv = [(i as f32 + 0.5) / MAP_WIDTH as f32, (j as f32 + 0.5) / MAP_HEIGHT as f32];
                        let want = bilinear(&new, uv);
                        let h = bilinear(&alt[0], uv);
                        let v = if by_height { flow_wind(h, &winds) } else { [bilinear(&alt[1], uv), bilinear(&alt[2], uv)] };
                        let from = advected_uv(map_dir(uv), h, eye[1], v, [0.0; 3], dt as f32);
                        lo = lo.min(want);
                        hi = hi.max(want);
                        texels.push((want, from));
                    }
                }
                // Only patches holding a cloud's edge say where it is.
                if hi - lo < 0.3 {
                    continue;
                }
                let cost = |sx: f32, sy: f32| -> f32 {
                    texels.iter().map(|&(w, f)| (bilinear(&old, [f[0] + sx / MAP_WIDTH as f32, f[1] + sy / MAP_HEIGHT as f32]) - w).abs()).sum()
                };
                let zero = cost(0.0, 0.0);
                let mut best = (zero, 0.0f32, 0.0f32);
                for a in -16..=16 {
                    for b in -16..=16 {
                        let (sx, sy) = (a as f32 * 0.5, b as f32 * 0.5);
                        let c = cost(sx, sy);
                        if c < best.0 {
                            best = (c, sx, sy);
                        }
                    }
                }
                shifts.push((best.1 * best.1 + best.2 * best.2).sqrt());
                gains.push(1.0 - best.0 / zero.max(1e-6));
                mean[0] += best.1 as f64;
                mean[1] += best.2 as f64;
            }
        }
        let n = shifts.len();
        shifts.sort_by(f32::total_cmp);
        gains.sort_by(f32::total_cmp);
        let q = |v: &[f32], f: f32| v[((v.len() - 1) as f32 * f) as usize];
        eprintln!(
            "STEPBACK dt {dt} s{}{}{}: {n} edge patches; shift (texels, 0.18 deg) median {:.2} p75 {:.2} p90 {:.2} max {:.2}; mean ({:.2}, {:.2}); the best shift cuts the misfit by median {:.0}%",
            if by_height { " height-wind" } else { "" },
            if hold_pop { " pop-held" } else { "" },
            if hold_churn { " churn-held" } else { "" },
            q(&shifts, 0.5), q(&shifts, 0.75), q(&shifts, 0.9), shifts[n - 1], mean[0] / n as f64, mean[1] / n as f64, 100.0 * q(&gains, 0.5)
        );
    }

    /// MAPS ACCUMULATED UNDER A MOVING, CHANGING SKY ARE THE SKY OF THEIR
    /// MOMENT: built slice by slice as `Sky::step_clouds` builds them (each
    /// map of a moment two builds ahead, the clouds churning, the winds
    /// carrying them, test_room's 2.5x clock at SpaceWarp's 36 Hz), the
    /// newest map against one drawn fresh at its moment. A history carried
    /// to the wrong place, or kept where a cloud has gone, leaves copies of
    /// the edges beside them (user, 2026-10-22: edges "duplicated in repeats
    /// that look wispy"). `GHOST_DUMP=dir` writes both maps' alpha as PGM.
    #[test]
    fn maps_accumulated_under_a_moving_sky_hold_no_ghost_edges() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.45, time_scale: 2.5, ..Default::default() };
        let model = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise);
        let mut population = CloudPopulation::new();
        let eye = [10.0f32, 2.0, 5.0];
        let uniform_at = |population: &mut CloudPopulation, at: f64| {
            population.update(&params, &field, at);
            let mut layers = model.layers(at);
            population.place(&mut layers, at);
            let u = CloudSkyUniform::new(&layers, eye, [0.35, 0.85, 0.4], [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], [0.0, 0.0], 4096.0, 0.5);
            (u, flow_winds(&params, &layers))
        };
        let t0 = 20_000.0;
        let dt = 2.5 / 36.0;
        let mut sky = CloudSky::new(&device, &queue, &field, &noise);
        let mut now = t0;
        let mut at = t0;
        let mut m = uniform_at(&mut population, at);
        let maps = 10;
        let mut frame = 0;
        // A slice every `GHOST_EVERY` frames (1), as the headset's pacing
        // does when the map runs over its budget (`CLOUDRATE`).
        let every: u32 = std::env::var("GHOST_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        while frame < maps * MAP_SLICES * every + 1 || sky.display()[2] != 0.0 {
            if frame % every != 0 {
                now += dt;
                frame += 1;
                continue;
            }
            if sky.starts_map(false) {
                at = if sky.is_fresh() { now } else { now + sky.map_lead() };
                m = uniform_at(&mut population, at);
            }
            sky.set_population(&device, &queue, &population);
            sky.set_winds(m.1);
            let mut enc = device.create_command_encoder(&Default::default());
            sky.encode_at(&mut enc, &queue, &m.0, &m.0, None, false, now, at);
            queue.submit(Some(enc.finish()));
            now += dt;
            frame += 1;
        }
        let newest = sky.display()[1] as u32;
        let moment = sky.times[newest as usize];
        let alpha_of = |sky: &CloudSky, layer: u32| -> Vec<f32> {
            let map = read_back(&device, &queue, &sky.maps, layer, MAP_WIDTH, MAP_HEIGHT, 8);
            (0..(MAP_WIDTH * MAP_HEIGHT) as usize).map(|i| half(&map[i * 8 + 6..i * 8 + 8])).collect()
        };
        let acc = alpha_of(&sky, newest);
        let mut fresh = CloudSky::new(&device, &queue, &field, &noise);
        // The population the newest map was drawn from, as it stands.
        let (u, winds) = uniform_at(&mut population, moment);
        fresh.set_population(&device, &queue, &population);
        fresh.set_winds(winds);
        let mut enc = device.create_command_encoder(&Default::default());
        fresh.encode_at(&mut enc, &queue, &u, &u, None, true, moment, moment);
        queue.submit(Some(enc.finish()));
        let want = alpha_of(&fresh, fresh.display()[1] as u32);
        let (mut off, mut n, mut ghosts, mut clear) = (0.0f64, 0u32, 0u32, 0u32);
        // Faint: a rim the ghost count above does not see, as in the
        // headset video (2026-10-23): over 2% of cloud where there is none.
        let (mut faint, mut clearest) = (0u32, 0u32);
        for j in 0..MAP_HEIGHT {
            let up = map_dir([0.5, (j as f32 + 0.5) / MAP_HEIGHT as f32])[1];
            if !(0.035..0.85).contains(&up) {
                continue;
            }
            for i in 0..MAP_WIDTH {
                let k = (j * MAP_WIDTH + i) as usize;
                if want[k] < 0.98 || acc[k] < 0.98 {
                    off += (acc[k] - want[k]).abs() as f64;
                    n += 1;
                }
                if want[k] > 0.99 {
                    clear += 1;
                    if acc[k] < 0.93 {
                        ghosts += 1;
                    }
                }
                if want[k] > 0.995 {
                    clearest += 1;
                    if acc[k] < 0.98 {
                        faint += 1;
                    }
                }
            }
        }
        let off = off / n.max(1) as f64;
        let ghost_share = ghosts as f64 / clear.max(1) as f64;
        let faint_share = faint as f64 / clearest.max(1) as f64;
        eprintln!("MOVING SKY faint rims: {faint} of {clearest} clear ({:.4}%)", 100.0 * faint_share);
        eprintln!("MOVING SKY after {frame} frames (moment {:.1} s on): alpha off by {off:.5} over {n} texels; ghosts {ghosts} of {clear} clear ({:.4}%)", moment - t0, 100.0 * ghost_share);
        if let Ok(dir) = std::env::var("GHOST_DUMP") {
            for (name, a) in [("acc", &acc), ("fresh", &want)] {
                let mut out = format!("P5\n{MAP_WIDTH} {MAP_HEIGHT}\n255\n").into_bytes();
                out.extend(a.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8));
                std::fs::write(format!("{dir}/ghost_{name}.pgm"), out).unwrap();
            }
        }
        // With the history (`HISTORY_NEW` 0.3) let go at 0.2 to 0.5: off by
        // 0.0399, 8.9% ghosts; at `HISTORY_REJECT` 0.03 to 0.1, 0.0188 and
        // 0.94%, and 0.0200 and 1.07% with the horizon kept wide. Without history, 0.0109 and 0.80% while each map sampled at
        // its own fraction (but bands on the headset); at one fraction for
        // every map, 0 and 0.
        assert!(off < 0.025, "the accumulated map is off the sky of its moment by {off:.5}");
        assert!(ghost_share < 0.015, "{ghosts} clear texels hold a cloud the sky has not");
    }

    /// How near one map of each march variant comes to a march four times
    /// finer, and how much of its error is fine-grained (the hatching): the
    /// error's own detail, |e - blur(e)|, a texel's 3x3 mean (`--ignored`).
    #[test]
    #[ignore]
    fn march_variants_against_a_finer_march() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.45, ..Default::default() };
        let model = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise);
        let mut population = CloudPopulation::new();
        let t = 20_000.0;
        population.update(&params, &field, t);
        let mut layers = model.layers(t);
        population.place(&mut layers, t);
        let u = CloudSkyUniform::new(&layers, [0.0, 2.0, 0.0], [0.35, 0.85, 0.4], [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], [0.0, 0.0], 4096.0, 0.5);
        let base = cloud_sky_shader();
        let draw = |src: String| -> Vec<[f32; 2]> {
            let mut sky = CloudSky::with_shader(&device, &queue, &field, &noise, src);
            sky.set_population(&device, &queue, &population);
            let mut enc = device.create_command_encoder(&Default::default());
            sky.encode_at(&mut enc, &queue, &u, &u, None, true, 0.0, t);
            queue.submit(Some(enc.finish()));
            let map = read_back(&device, &queue, &sky.raw, 0, MAP_WIDTH, MAP_HEIGHT, 8);
            (0..(MAP_WIDTH * MAP_HEIGHT) as usize).map(|i| [half(&map[i * 8 + 6..i * 8 + 8]), half(&map[i * 8..i * 8 + 2])]).collect()
        };
        let with = |refine: u32, bisect: u32, finer: u32| {
            base.replace("const REFINE: u32 = 4u;", &format!("const REFINE: u32 = {refine}u;"))
                .replace("const ENTRY_BISECT: u32 = 0u;", &format!("const ENTRY_BISECT: u32 = {bisect}u;"))
                .replace(&format!("const VIEW_STEPS_MOST: u32 = {}u;", cl::VIEW_STEPS_MOST), &format!("const VIEW_STEPS_MOST: u32 = {}u;", cl::VIEW_STEPS_MOST * finer))
                .replace(&format!("const VIEW_STEP_TEXELS: f32 = {:?};", cl::VIEW_STEP_TEXELS), &format!("const VIEW_STEP_TEXELS: f32 = {:?};", cl::VIEW_STEP_TEXELS / finer as f32))
        };
        let reference = draw(with(4, 3, 4));
        let (w, h) = (MAP_WIDTH as usize, MAP_HEIGHT as usize);
        let score = |m: &[[f32; 2]], c: usize| {
            let e = |i: usize, j: usize| m[j * w + i][c] - reference[j * w + i][c];
            let (mut off, mut grain, mut n) = (0.0f64, 0.0f64, 0u32);
            for j in 1..h - 1 {
                let up = map_dir([0.5, (j as f32 + 0.5) / h as f32])[1];
                if !(0.035..0.85).contains(&up) {
                    continue;
                }
                for i in 1..w - 1 {
                    if reference[j * w + i][0] >= 0.98 {
                        continue;
                    }
                    let mut mean = 0.0;
                    for dj in 0..3 {
                        for di in 0..3 {
                            mean += e(i + di - 1, j + dj - 1);
                        }
                    }
                    off += e(i, j).abs() as f64;
                    grain += (e(i, j) - mean / 9.0).abs() as f64;
                    n += 1;
                }
            }
            (off / n as f64, grain / n as f64)
        };
        for (refine, bisect) in [(4, 0), (4, 2), (4, 3), (3, 3), (2, 3), (2, 2), (1, 3)] {
            let m = draw(with(refine, bisect, 1));
            let (a, ag) = score(&m, 0);
            let (r, rg) = score(&m, 1);
            eprintln!("VARIANT refine {refine} bisect {bisect}: alpha off {a:.5} grain {ag:.5}; red off {r:.5} grain {rg:.5}");
        }
    }

    /// A STILL SKY BUILT MAP AFTER MAP STAYS STILL near the horizon, where a
    /// ray crosses kilometres of cloud in strides of hundreds of metres: the
    /// display cross-fades from each map to the next, so any change between
    /// two maps of one sky is a flicker (user, 2026-10-22: a line flickering
    /// just above the sea's horizon).
    #[test]
    #[ignore = "open: with the golden jitter a still sky's horizon moves 0.0045 a map (0.0113 with no history, the line the user saw); 0 needs one fraction, which banded the clouds (2026-10-22)"]
    fn a_still_sky_holds_still_at_the_horizon_from_map_to_map() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.45, ..Default::default() };
        let model = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise);
        let mut population = CloudPopulation::new();
        let t = 20_000.0;
        population.update(&params, &field, t);
        let mut layers = model.layers(t);
        population.place(&mut layers, t);
        let u = CloudSkyUniform::new(&layers, [0.0, 2.0, 0.0], [0.35, 0.85, 0.4], [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], [0.0, 0.0], 4096.0, 0.5);
        let mut sky = CloudSky::new(&device, &queue, &field, &noise);
        sky.set_population(&device, &queue, &population);
        let mut now = 0.0;
        let mut maps: Vec<Vec<f32>> = Vec::new();
        let mut enc = device.create_command_encoder(&Default::default());
        sky.encode_at(&mut enc, &queue, &u, &u, None, true, now, t);
        queue.submit(Some(enc.finish()));
        for _ in 0..3 {
            for _ in 0..MAP_SLICES {
                now += 1.0 / 72.0;
                let mut enc = device.create_command_encoder(&Default::default());
                sky.encode_at(&mut enc, &queue, &u, &u, None, false, now, t);
                queue.submit(Some(enc.finish()));
            }
            let map = read_back(&device, &queue, &sky.maps, sky.display()[1] as u32, MAP_WIDTH, MAP_HEIGHT, 8);
            maps.push((0..(MAP_WIDTH * MAP_HEIGHT) as usize).map(|i| half(&map[i * 8 + 6..i * 8 + 8])).collect());
        }
        let (mut change, mut n) = (0.0f64, 0u32);
        for j in 0..MAP_HEIGHT {
            let up = map_dir([0.5, (j as f32 + 0.5) / MAP_HEIGHT as f32])[1];
            if !(0.0..0.06).contains(&up) {
                continue;
            }
            for i in 0..MAP_WIDTH {
                let k = (j * MAP_WIDTH + i) as usize;
                for w in maps.windows(2) {
                    change += (w[1][k] - w[0][k]).abs() as f64;
                    n += 1;
                }
            }
        }
        let change = change / n.max(1) as f64;
        eprintln!("HORIZON a still sky's map to the next, within 3.4 deg of the horizon: alpha moves {change:.5}");
        assert!(change < 1e-4, "a still sky's horizon changes by {change:.5} from map to map");
    }

    /// A DIMMING SKY LEAVES NO DAY IN THE HISTORY: maps accumulated while the
    /// light falls 4 times a map (a dusk sped up) come out as a fresh map
    /// under the last light does, each read as a share of its own map's
    /// light. Carried as absolute light, the day's clouds outlived the dusk
    /// and the night's exposure blew them white (headset, 2026-10-23).
    #[test]
    fn a_dimming_sky_keeps_no_day_light_in_the_history() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.6, ..Default::default() };
        let model = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise);
        let mut population = CloudPopulation::new();
        let t = 20_000.0;
        population.update(&params, &field, t);
        let mut layers = model.layers(t);
        population.place(&mut layers, t);
        let lit = |f: f32| {
            let s = |v: [f32; 3]| v.map(|c| c * f);
            CloudSkyUniform::new(&layers, [0.0, 2.0, 0.0], [0.35, 0.85, 0.4], s([3.0, 2.9, 2.7]), s([0.4, 0.5, 0.7]), s([0.2, 0.2, 0.15]), [0.0, 0.0], 4096.0, 0.5)
        };
        const MAPS: u32 = 8;
        let last = 0.25f32.powi(MAPS as i32);
        let draw = |dim: bool| -> Vec<[f32; 2]> {
            let mut sky = CloudSky::with_shader(&device, &queue, &field, &noise, cloud_sky_shader());
            sky.set_population(&device, &queue, &population);
            let mut now = 0.0;
            let first = if dim { lit(1.0) } else { lit(last) };
            let mut enc = device.create_command_encoder(&Default::default());
            sky.encode_at(&mut enc, &queue, &first, &first, None, true, now, t);
            queue.submit(Some(enc.finish()));
            if dim {
                for i in 0..MAPS * MAP_SLICES {
                    let u = lit(0.25f32.powi((i / MAP_SLICES + 1) as i32));
                    now += 1.0 / 72.0;
                    let mut enc = device.create_command_encoder(&Default::default());
                    sky.encode_at(&mut enc, &queue, &u, &u, None, false, now, t);
                    queue.submit(Some(enc.finish()));
                }
            }
            let map = read_back(&device, &queue, &sky.maps, sky.display()[1] as u32, MAP_WIDTH, MAP_HEIGHT, 8);
            (0..(MAP_WIDTH * MAP_HEIGHT) as usize).map(|i| [half(&map[i * 8 + 6..i * 8 + 8]), half(&map[i * 8..i * 8 + 2])]).collect()
        };
        let (dimmed, fresh) = (draw(true), draw(false));
        let (mut a, mut b, mut n) = (0.0f64, 0.0f64, 0u32);
        for (d, f) in dimmed.iter().zip(&fresh) {
            if f[0] < 0.9 {
                a += d[1] as f64;
                b += f[1] as f64;
                n += 1;
            }
        }
        let ratio = a / b.max(1e-12);
        eprintln!("DIMMING: {n} cloud texels, the dimmed history's light {ratio:.3} of a fresh map's under the last light");
        assert!(n > 1000, "too little cloud: {n}");
        assert!((ratio - 1.0).abs() < 0.15, "the history kept the day's light: {ratio:.3} of a fresh map's");
    }

    /// ONE MAP IS ALREADY NEAR A MARCH FOUR TIMES FINER, AND THE MAPS
    /// ACCUMULATED KEEP IT: the stride's sample planes were the wood grain
    /// of one map (alpha off by 0.0236, 0.0123 after eight maps of
    /// `HISTORY_NEW`); since the march goes fine through the cloud
    /// (`REFINE`, 2026-10-21) one map is off by 0.0076 at the old extinction
    /// and 0.0136 at a cumulus's real one (`EXTINCTION_PER_M` 0.06, three
    /// times the optical depth a step), under 0.016 either way, and a still
    /// sky built map after map stays there.
    #[test]
    fn maps_built_one_after_another_settle_on_a_finer_march() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.6, ..Default::default() };
        let model = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise);
        let mut population = CloudPopulation::new();
        let t = 20_000.0;
        population.update(&params, &field, t);
        let mut layers = model.layers(t);
        population.place(&mut layers, t);
        let u = CloudSkyUniform::new(&layers, [0.0, 2.0, 0.0], [0.35, 0.85, 0.4], [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], [0.0, 0.0], 4096.0, 0.5);
        let base = cloud_sky_shader();
        let fine = base
            .replace(&format!("const VIEW_STEPS_MOST: u32 = {}u;", cl::VIEW_STEPS_MOST), &format!("const VIEW_STEPS_MOST: u32 = {}u;", cl::VIEW_STEPS_MOST * 4))
            .replace(&format!("const VIEW_STEP_TEXELS: f32 = {:?};", cl::VIEW_STEP_TEXELS), &format!("const VIEW_STEP_TEXELS: f32 = {:?};", cl::VIEW_STEP_TEXELS / 4.0));
        assert_ne!(fine, base, "the finer march changed nothing");
        // Each map's alpha and red, read back.
        let draw = |src: String, maps: u32| -> Vec<[f32; 2]> {
            let mut sky = CloudSky::with_shader(&device, &queue, &field, &noise, src);
            sky.set_population(&device, &queue, &population);
            let mut now = 0.0;
            let mut enc = device.create_command_encoder(&Default::default());
            sky.encode_at(&mut enc, &queue, &u, &u, None, true, now, t);
            queue.submit(Some(enc.finish()));
            for _ in 0..maps * MAP_SLICES {
                now += 1.0 / 72.0;
                let mut enc = device.create_command_encoder(&Default::default());
                sky.encode_at(&mut enc, &queue, &u, &u, None, false, now, t);
                queue.submit(Some(enc.finish()));
            }
            let map = read_back(&device, &queue, &sky.maps, sky.display()[1] as u32, MAP_WIDTH, MAP_HEIGHT, 8);
            (0..(MAP_WIDTH * MAP_HEIGHT) as usize).map(|i| [half(&map[i * 8 + 6..i * 8 + 8]), half(&map[i * 8..i * 8 + 2])]).collect()
        };
        let coarse = |k: u32| {
            base.replace(&format!("const VIEW_STEPS_MOST: u32 = {}u;", cl::VIEW_STEPS_MOST), &format!("const VIEW_STEPS_MOST: u32 = {}u;", cl::VIEW_STEPS_MOST / k))
                .replace(&format!("const VIEW_STEP_TEXELS: f32 = {:?};", cl::VIEW_STEP_TEXELS), &format!("const VIEW_STEP_TEXELS: f32 = {:?};", cl::VIEW_STEP_TEXELS * k as f32))
        };
        let reference = draw(fine, 0);
        let once = draw(base.clone(), 0);
        let half_settled = draw(coarse(2), 8);
        let third_settled = draw(coarse(3), 8);
        let settled = draw(base, 8);
        let off = |m: &[[f32; 2]], c: usize| {
            let (mut sum, mut n) = (0.0f64, 0u32);
            for j in 0..MAP_HEIGHT {
                let up = map_dir([0.5, (j as f32 + 0.5) / MAP_HEIGHT as f32])[1];
                if !(0.035..0.85).contains(&up) {
                    continue;
                }
                for i in 0..MAP_WIDTH {
                    let k = (j * MAP_WIDTH + i) as usize;
                    if reference[k][0] < 0.98 {
                        sum += (m[k][c] - reference[k][c]).abs() as f64;
                        n += 1;
                    }
                }
            }
            sum / n.max(1) as f64
        };
        let (once_a, settled_a, once_r, settled_r) = (off(&once, 0), off(&settled, 0), off(&once, 1), off(&settled, 1));
        eprintln!("ACCUMULATION off a march 4x finer: alpha {once_a:.5} once, {settled_a:.5} settled; red {once_r:.5} once, {settled_r:.5} settled");
        eprintln!(
            "ACCUMULATION with fewer steps, settled: half alpha {:.5} red {:.5}; a third alpha {:.5} red {:.5}",
            off(&half_settled, 0),
            off(&half_settled, 1),
            off(&third_settled, 0),
            off(&third_settled, 1)
        );
        // (0.0157 settled when set; 0.0163 once the shape noise was untiled,
        // `cloud_layers::detile`, its stretch a little finer than the march.)
        assert!(once_a < 0.018 && settled_a < 0.018, "alpha: once {once_a:.5}, settled {settled_a:.5}");
        // Or under half the settled error this was set at: vapour-soft edges
        // (`cloud_layers::EDGE_GAIN`, 2026-10-23) shrank the single map's error
        // to 0.0047, and 1.5 of that is stricter than the settled sky ever was.
        assert!(
            (settled_a < 1.5 * once_a || settled_a < 0.008) && (settled_r < 1.5 * once_r || settled_r < 0.0025),
            "accumulating moved off the finer march: alpha {once_a:.5} -> {settled_a:.5}, light {once_r:.5} -> {settled_r:.5}"
        );
    }

    /// NO LINES ACROSS THE SKY: a line parallel to the horizon moves every
    /// column of a map row together, where cloud edges do not line up round
    /// the whole sky. Each low row's change from the row below, averaged
    /// signed round the azimuth, is printed against the mean unsigned change
    /// (`BANDING` the worst ratio): the strata and step-plane jumps of the
    /// march stood out as rows far above it (research 2026-10-21).
    #[test]
    fn the_sky_map_has_no_rows_lined_across_the_sky() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let mut worst = 0.0f32;
        for coverage in [0.4f32, 0.6, 0.8] {
            let params = CloudParams { coverage, ..Default::default() };
            let model = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise);
            let mut population = CloudPopulation::new();
            let t = 20_000.0;
            population.update(&params, &field, t);
            let mut layers = model.layers(t);
            population.place(&mut layers, t);
            let mut sky = CloudSky::new(&device, &queue, &field, &noise);
            sky.set_population(&device, &queue, &population);
            let u = CloudSkyUniform::new(&layers, [0.0, 2.0, 0.0], [0.35, 0.85, 0.4], [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], [0.0, 0.0], 4096.0, 0.5);
            let mut enc = device.create_command_encoder(&Default::default());
            sky.encode_at(&mut enc, &queue, &u, &u, None, true, t, t);
            queue.submit(Some(enc.finish()));
            let map = read_back(&device, &queue, &sky.maps, sky.display()[1] as u32, MAP_WIDTH, MAP_HEIGHT, 8);
            let alpha = |i: u32, j: u32| half(&map[((j * MAP_WIDTH + i) * 8 + 6) as usize..((j * MAP_WIDTH + i) * 8 + 8) as usize]);
            let rows: Vec<u32> = (0..MAP_HEIGHT - 1)
                .filter(|&j| {
                    let up = map_dir([0.5, (j as f32 + 0.5) / MAP_HEIGHT as f32])[1];
                    (0.035..0.5).contains(&up)
                })
                .collect();
            let signed: Vec<f32> = rows.iter().map(|&j| (0..MAP_WIDTH).map(|i| alpha(i, j + 1) - alpha(i, j)).sum::<f32>() / MAP_WIDTH as f32).collect();
            let unsigned: f32 = rows.iter().map(|&j| (0..MAP_WIDTH).map(|i| (alpha(i, j + 1) - alpha(i, j)).abs()).sum::<f32>() / MAP_WIDTH as f32).sum::<f32>() / rows.len() as f32;
            let peak = signed.iter().map(|v| v.abs()).fold(0.0, f32::max);
            // Inside cloud edges, the curvature up the sky against across it:
            // stripes along the horizon raise the first.
            let (mut up_c, mut across_c, mut m) = (0.0f32, 0.0f32, 0u32);
            for &j in rows.iter().filter(|&&j| j > 0) {
                for i in 1..MAP_WIDTH - 1 {
                    let a = alpha(i, j);
                    if !(0.1..0.9).contains(&a) {
                        continue;
                    }
                    up_c += (alpha(i, j + 1) - 2.0 * a + alpha(i, j - 1)).abs();
                    across_c += (alpha(i + 1, j) - 2.0 * a + alpha(i - 1, j)).abs();
                    m += 1;
                }
            }
            eprintln!(
                "BANDING coverage {coverage}: worst row {:.4} against a mean change {:.4} ({:.2}x) over {} rows; STRIPES curvature up {:.5} across {:.5} ({:.3}) at {m} edge texels",
                peak, unsigned, peak / unsigned.max(1e-6), rows.len(), up_c / m.max(1) as f32, across_c / m.max(1) as f32, up_c / across_c.max(1e-6)
            );
            worst = worst.max(peak / unsigned.max(1e-6));
        }
        eprintln!("BANDING worst {worst:.2}");
    }

    /// A cloud far down the sky fades into the haze in the map's alpha, but
    /// still blocks a disc behind it: the heights' .g is the clouds' own
    /// transmittance, never more than the alpha, and low in the sky it is
    /// far less where the alpha reads nearly clear (the sun shone through
    /// clouds on the horizon, headset 2026-10-21).
    #[test]
    fn a_far_cloud_the_haze_hides_still_blocks_the_sun() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let scope = device.push_error_scope(ErrorFilter::Validation);
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.8, ..Default::default() };
        let model = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise);
        let mut population = CloudPopulation::new();
        let t = 20_000.0;
        population.update(&params, &field, t);
        let mut layers = model.layers(t);
        population.place(&mut layers, t);
        let mut sky = CloudSky::new(&device, &queue, &field, &noise);
        sky.set_population(&device, &queue, &population);
        let u = CloudSkyUniform::new(&layers, [0.0, 2.0, 0.0], [0.35, 0.85, 0.4], [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], [0.0, 0.0], 4096.0, 0.5);
        let mut enc = device.create_command_encoder(&Default::default());
        sky.encode_at(&mut enc, &queue, &u, &u, None, true, t, t);
        queue.submit(Some(enc.finish()));
        let layer = sky.display()[1] as u32;
        // The march's own alpha: the prefilter's lobes may lift a map's a
        // hair past the heights' .g beside a hard edge, which the sky shader
        // cancels (it weighs a disc by .g / a times a).
        let map = read_back(&device, &queue, sky.raw(), 0, MAP_WIDTH, MAP_HEIGHT, 8);
        let alts = read_back(&device, &queue, &sky.alts, layer, MAP_WIDTH, MAP_HEIGHT, 8);
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("the cloud passes do not validate: {e}");
        }
        let (mut above, mut hidden_low, mut hidden_high) = (0, 0, 0);
        for j in 0..MAP_HEIGHT {
            for i in 0..MAP_WIDTH {
                let k = (j * MAP_WIDTH + i) as usize;
                let alpha = half(&map[k * 8 + 6..k * 8 + 8]);
                let through = half(&alts[k * 8 + 2..k * 8 + 4]);
                if through > alpha + 2e-3 {
                    above += 1;
                }
                if alpha > 0.9 && through < 0.3 {
                    let up = map_dir([(i as f32 + 0.5) / MAP_WIDTH as f32, (j as f32 + 0.5) / MAP_HEIGHT as f32])[1];
                    if up < 0.1 { hidden_low += 1 } else { hidden_high += 1 }
                }
            }
        }
        eprintln!("haze-cleared but cloud-blocked texels: {hidden_low} low, {hidden_high} high; {above} with more through than the alpha");
        assert_eq!(above, 0, "{above} texels let more through a disc than the sky");
        assert!(hidden_low > 100, "no far cloud the haze clears: {hidden_low}");
        assert!(hidden_high < hidden_low / 10, "haze clears cloud high in the sky: {hidden_high} vs {hidden_low}");
    }

    /// THE ENVELOPES ARE THE POPULATION'S: each layer's splatted map against
    /// the CPU twin's texels, then the shadow the populated clouds cast
    /// against the twin's columns.
    #[test]
    fn the_gpu_envelopes_match_their_twin() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let scope = device.push_error_scope(ErrorFilter::Validation);
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.5, ..Default::default() };
        let t = 20_000.0;
        let mut population = CloudPopulation::new();
        population.update(&params, &field, t);
        let mut layers = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise).layers(t);
        population.place(&mut layers, t);
        let mut sky = CloudSky::new(&device, &queue, &field, &noise);
        sky.set_population(&device, &queue, &population);
        let size = crate::renderer::uniforms::CLOUD_SHADOW_SIZE;
        let shadow = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let sun = [0.35f32, 0.85, 0.4];
        let (centre, extent) = ([120.0f32, -40.0], 4096.0);
        let u = CloudSkyUniform::new(&layers, [10.0, 2.0, 5.0], sun, [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], centre, extent, 0.5);
        let mut enc = device.create_command_encoder(&Default::default());
        sky.encode(&mut enc, &queue, &u, Some(&shadow.create_view(&Default::default())), true);
        queue.submit(Some(enc.finish()));
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("the envelope passes do not validate: {e}");
        }
        let n = ENVELOPE_SIZE;
        for k in 0..POPULATED {
            let map = read_back(&device, &queue, &sky.envelope, k as u32, n, n, 8);
            let (mut worst, mut covered, mut seen) = (0.0f32, 0, 0);
            let mut worst_at = (0usize, 0.0f32, 0.0f32);
            let mut worst_saturated = 0.0f32;
            for j in (0..n).step_by(5) {
                for i in (0..n).step_by(3) {
                    let at = ((j * n + i) * 8) as usize;
                    let gpu = [0, 1, 2, 3].map(|c| half(&map[at + 2 * c..at + 2 * c + 2]));
                    let cpu = population.texel(k, i as i32, j as i32);
                    // Where several clouds' cores stack (a clustered field's
                    // sums reach 7 and more) the GPU adds them in half floats,
                    // 0.004 a step at 4-8, and the march clamps the cover to 1
                    // there anyway: those texels are held to the twin loosely.
                    let saturated = cpu[0] > 2.0;
                    for c in 0..4 {
                        let d = (gpu[c] - cpu[c]).abs() / (1.0 + cpu[c].abs());
                        if saturated {
                            worst_saturated = worst_saturated.max(d);
                        } else if d > worst {
                            worst = d;
                            worst_at = (c, cpu[c], gpu[c]);
                        }
                    }
                    covered += (cpu[0] > 0.05) as u32;
                    seen += 1;
                }
            }
            eprintln!("layer {k}: {} splats, worst relative difference {worst:.5} (channel {}, twin {:.4}, gpu {:.4}), {worst_saturated:.5} where the cover is saturated, {covered} of {seen} texels clouded", population.splats(k).0.len(), worst_at.0, worst_at.1, worst_at.2);
            assert!(worst_saturated < 1e-2, "layer {k}'s saturated cores are off the twin by {worst_saturated}");
            assert!(worst < 4e-3, "layer {k}'s envelope is not the twin's: {worst}");
            assert!(covered > seen / 20, "layer {k} has almost no clouds: {covered} of {seen}");
        }
        let texels = read_back(&device, &queue, &shadow, 0, size, size, 1);
        let dir = { let l = (sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]).sqrt(); sun.map(|v| v / l) };
        let weather = space_soup_sky::cloud_layers::CloudWeather::new(&field, Some(&population));
        let step = extent / size as f32;
        let (mut worst, mut sum, mut count, mut shaded) = (0.0f32, 0.0f32, 0, 0);
        for j in (1..size - 1).step_by(2) {
            for i in (1..size - 1).step_by(2) {
                let xz = [centre[0] - 0.5 * extent + (i as f32 + 0.5) * step, centre[1] - 0.5 * extent + (j as f32 + 0.5) * step];
                let cpu = space_soup_sky::cloud_layers::transmittance_penumbra(&layers, &noise, weather, xz, 2.0, dir);
                let gpu = texels[(j * size + i) as usize] as f32 / 255.0;
                worst = worst.max((cpu - gpu).abs());
                sum += (cpu - gpu).abs();
                count += 1;
                shaded += (gpu < 0.5) as u32;
            }
        }
        eprintln!("populated shadow vs twin: mean |diff| {:.4}, worst {worst:.3}; {shaded} of {count} shaded", sum / count as f32);
        assert!(sum / (count as f32) < 0.02 && worst < 0.25, "the populated shadow is not the twin's: mean {}, worst {worst}", sum / count as f32);
    }
}
