# Sky, day and night, and clouds

The time-of-day sky is simulated, not painted: `space_soup_sky::time_of_day` computes the
sun, moon, stars and atmosphere for any hour, latitude and day of the year. The renderer
draws it and lights the level from it.

## The cloud layer (`space_soup_sky::clouds`, 2026-10-19)

**What a cloud is.** A real layer at a real altitude above a curved Earth, met once per sky
pixel. There is no ray-march.

- What that buys:
  - perspective: large overhead, packed toward the horizon;
  - motion parallax as the player moves (the eye's world position is used);
  - wind that crosses the sky faster overhead;
  - lighting from the sun of the hour.
- Binocular disparity is invisible at cloud distance: 0.064 m at 2 km is 1/25 of a pixel.
  So none is faked, which would make the clouds look like a model a few metres away.

**Parts:**

| Part | Where | Note |
|---|---|---|
| `CloudParams` | sky crate, and `TimeOfDayParams::clouds` | coverage, altitude, thickness, scale, density, wind, evolution, seed. Coverage 0 means clear |
| `CloudsDef` | `space_soup_engine::scene`, as `time_of_day.clouds` | optional; a scene without it saves byte-identically |
| `CloudField` | sky crate | one 256² RGBA8 tile: two equalised Perlin-Worley layers (R, G) and Worley detail (B), plus `mips()`. 58 ms to build |
| `CloudFrame` | sky crate | the CPU twin of the shader: `coverage_density`, `hit`, `shade`, `light_transmittance`, `at(seconds)` |
| `Sky::clouds`, `SkyExtra::cloud_*` | `renderer::sky` | texture binding 3 and repeat sampler binding 4; seven uniform rows |
| `sky_cloud_shader()` | `renderer::sky` | a checked text twin of `sky_shader()`. `SkyPipeline::pick` uses it only when the sky has clouds |

**Rules:**

- **Coverage is exact.** The two layers are equalised, and their mean goes through the CDF
  of a sum of two uniforms. So coverage 0.4 covers 40% of the sky
  (`coverage_is_the_share_of_the_sky_covered`). Detail erosion thins edges without moving
  the outline.
- **Wind offsets are wrapped in f64 and uploaded as fractions of a tile.** Noise fed a
  growing clock turns into a grid (`wind_offsets_stay_small_after_days_of_play`).
- **The drawn panorama stays clear.** The shader draws the clouds over it. The snapshot
  also makes a clouded copy (`SkySnapshot::reflection`, `reflection_engine()`) for
  reflections and the SH, so overcast light is flat and grey
  (`overcast_dims_the_sun_and_flattens_the_sky_light`).
- **The sun is dimmed every frame.** `Sky::set_clock` advances the offsets and multiplies
  `light_clear` by the cloud transmittance toward the light. Direct light and shadows then
  fade as a cloud crosses the sun, with no renderer change. A level is about 100 m and
  clouds are kilometres across, so one number per frame is correct.
- **Lighting.**
  - The light at the cloud's altitude is the top-of-atmosphere light times
    `transmittance_at(R + h)`, so undersides stay red after the ground's sunset.
  - Three Wrenninge octaves of a two-lobe Henyey-Greenstein phase give the silver lining.
  - Two-stream diffuse transmission gives grey bases.
  - Sky and ground ambient, plus haze with distance.
- **Mips.** The tile is sampled with mips in uniform control flow, because distant
  clouds are heavily minified.

**Guards:**

- `the_cloud_twin_with_no_clouds_draws_the_clear_skys_pixels`: byte-identical.
- `the_cloud_shader_draws_what_the_cpu_twin_computes`: GPU against CPU within 6/255 over
  24 directions, through cloud and clear.
- `an_overcast_noon_zenith_is_grey_not_blue`.
- `the_cloud_twin_stays_small`: by the proxy the clear sky is 606 and with clouds 889.
  That is about 1,660 estimated PIPESTATS, against the 3,400 cliff, on sky pixels only.

**Not yet:**

- the editor's cloud controls;
- weather driving the clouds;
- cloud shadows moving across large terrain (a 128² shadow texture folded into the sun
  map's lookup, priced first);
- the headset's measured cost.
