# A realistic simulated sky on Quest 3: research and plan (2026-10-20)

What a real sky looks like, why ours does not look like it yet, how the
best shipped systems get there, and how to get there inside the
Quest 3's 13.9 ms. Realism is the target. Optimisation is how we
afford it. Nothing here is gated off the headset.

## 1. What the headset showed (artytest1 `test_graphics`, videos 2026-10-20 01:00-01:02)

| Seen | Cause in our code |
|---|---|
| Stars move in jumps | The sky is rebuilt every `REFRESH_HOURS` = 0.02 game hours. With a 20-minute day that is about once a real second. `celestial` (the star rotation), the sun and moon directions and the panorama only change on a rebuild (`apply_time_of_day` -> `Sky::apply_snapshot`). Between rebuilds only `set_clock` runs. The sky turns 15.04 deg an hour, so every rebuild jumps it 0.3 deg. That is about 7 px at the Quest 3's ~25 px/deg, and wider than the sun's whole disc. |
| No realistic sun | The disc is an analytic disc with one grey limb-darkening factor (`sky_sun_disc`). Nothing makes it read as dazzling: no glare around it, no per-channel limb, no refraction flattening, no reddening across the disc. The tonemapper clips it to a white dot. |
| No realistic moon | Eleven hand-placed "maria" spots on a sphere (`moon_albedo`). The moon also sits exactly on the ecliptic (no 5.1 deg orbit tilt) and has no earthshine. |
| Clouds look poor | They are a single 2D layer: one density read per pixel, one tap toward the light, and a hard coverage threshold. That gives flat "camouflage" blobs with cut-out edges, uniform brown shadows, no dark bases, no silver linings, no depth and no perspective of thickness. At night they glowed brighter than the sky. |
| Sunset colours good | The atmosphere model (Hillaire-style transmittance and multi-scatter tables, ozone included) is sound. Keep it. |

The atmosphere is right. Almost everything else needs to be rebuilt.

## 2. What a real sky looks like, with numbers

**Sun**
- Angular diameter 0.533 deg. Luminance about 1.6e9 cd/m^2 at noon, against 3,000-10,000 cd/m^2 for blue sky (a ratio of about 2e5). No display can show it. What makes a sun read as real to the eye is the **glare around it**: light scattering inside the eye (a veiling halo falling off roughly as 1/theta^2 to 1/theta^3) and the bright Mie aureole in the sky within a few degrees.
- Limb darkening differs by wavelength. The linear law is I(mu)/I(0) = 1 - u(1 - mu), with u about 0.56 at 600 nm and stronger toward the blue (0.95 at 320 nm). The edge is dimmer and redder than the centre.
- At the horizon the air mass is about 38, so transmittance reddens the disc. Refraction lifts it about 34' and lifts the lower limb more than the upper one, so the setting sun is visibly **flattened**. Its edges split by colour, which is where the green rim and green flash come from.

**Twilight**
- Civil, nautical and astronomical twilight end at -6, -12 and -18 deg.
- After sunset the **Earth's shadow** (a dark blue-grey band) rises in the east under the **Belt of Venus** (a pink band 10-20 deg up).
- The deep twilight blue at the zenith comes from ozone (Chappuis band) absorbing, not scattering.
- **High clouds stay sunlit after the ground has gone dark.** A cloud at height h keeps the sun until about sqrt(2h/R) below the horizon: 2.9 deg for 8 km cirrus, 1.4 deg for 2 km cumulus. Only clouds that know their own altitude get this right.

**Moon**
- Angular diameter 0.52 deg. Albedo only about 0.12, as dark as worn asphalt; it looks white only against the night sky.
- A full moon is about 2,500 cd/m^2. Brightness against phase angle is strongly non-linear. The **opposition surge** makes the full moon roughly twice as bright as the quarters' geometry predicts. One magnitude fit is m = -12.73 + 0.026|a| + 4e-9 a^4, with a in degrees.
- At full phase the limb is *not* darkened; it is nearly a flat disc (Lommel-Seeliger and Hapke behaviour).
- **Earthshine** lights the dark part of a crescent faintly. It is brightest when the Earth seen from the Moon is "full", which is when the Moon is a thin crescent.
- The orbit is tilted 5.1 deg to the ecliptic, so the Moon wanders above and below the sun's path.

**Stars**
- About 9,100 stars are brighter than magnitude 6.5 across the whole sky, with about 2,000-4,500 above the horizon at a dark site.
- Each is a point. On a display it must be a stable, energy-correct spot of about 1-2 px that never shimmers as the head turns.
- Colours are subtle (B-V index): Betelgeuse orange, Rigel blue-white. Rod vision washes out the colour of all but the brightest.
- The whole sky turns **smoothly** at 15.04 deg an hour about the celestial pole.
- Twinkling (scintillation) is strong near the horizon and nearly absent at the zenith. Near the horizon it also flickers in colour.
- Background light: a natural dark sky is about 21.6-22 mag/arcsec^2 (~2e-4 cd/m^2; we already use 2.0e-4). The Milky Way is about 20-21 mag/arcsec^2, with dust lanes and the bright core in Sagittarius. Airglow tints the sky faintly green-blue. Zodiacal light rises as a cone after dusk and before dawn.
- Below about 0.01 cd/m^2 the eye shifts to rods (the **Purkinje shift**): blues hold, reds go dark, colour drains away. We already have a scotopic blend (`tonemap::SCOTOPIC_WEIGHTS`); it needs to be driven by real luminance.

**Clouds**
- Cumulus have bases at 0.5-2 km and tops at 1-3 km in fair weather. Cumulonimbus reach 12 km or more. Stratus sit low and flat. Cirrus are ice at 6-12 km and are correctly drawn as a 2D layer.
- **What makes a cloud read as real:**
  - **Dark, flat bases with bright tops:** optical depth is 10 to hundreds, so little light gets through to the underside.
  - **Silver linings:** strong forward scattering (the Mie phase has a huge forward peak, g about 0.85) lights the edges when you look toward the sun.
  - **Dark edges ("powder"):** on the side away from the sun, the thin outer edges have had little in-scattered light built up yet.
  - **Soft wisps** at the edges, and **depth:** near clouds pass in front of far ones, and thickness shows in perspective toward the horizon.
  - **Crepuscular rays** (sunbeams) and **cloud shadows** sweeping across the ground.
- At sunset the undersides turn red and orange while the sun is below the cloud base. At night clouds are lit by the moon and the glow of towns, and in a dark place they are *darker* than the starry sky.

## 3. How the best shipped systems do it

- **Atmosphere:** Hillaire 2020 (Epic), "A Scalable and Production Ready Sky and Atmosphere Rendering Technique".
  - Four small LUTs: transmittance, multiple scattering, **sky-view** (rebuilt every frame, with latitude packed toward the horizon) and **aerial perspective** (a camera froxel volume).
  - The cost does not depend on screen resolution, and it ships on mobile ES 3.1 (Fortnite on an iPhone 6s).
  - The slides give 0.31 ms in total on a GTX 1080 (sky at 720p = 0.14 ms), and LUT cost on mobile at 1.03 ms. The transmittance LUT can be smaller still on mobile.
  - The cloud section of the same course says **cloud rendering was "not reasonably achievable on mobile" in 2020**. That is the wall to take apart in section 4.
- **Clouds:** Schneider and Vos, Guerrilla, "Nubis".
  - **2015:** a 2.5D raymarch through a cloud layer. A **weather map** gives coverage, cloud type and precipitation. Height profiles per type, Perlin-Worley 3D base noise with Worley detail erosion, "beer-powder" lighting. Under 2 ms on PS4, with **1 pixel in each 4x4 tile updated per frame** and the rest reprojected.
  - **2022 (Nubis Evolved):** flyable clouds, superstorms, and internal lightning lighting at "near-zero" cost.
  - **2023 (Nubis Cubed):** voxel clouds with compressed SDFs to skip empty space. They still act as a volumetric skybox with time of day when seen from the ground.
- **Frostbite** (Hillaire 2016):
  - Energy-conserving integration of in-scattering per step: S_int = (S - S·e^(-sigma_t·dt)) / sigma_t. Without it, bright clouds gain energy as the step size changes.
  - Multiple scattering in clouds by Wrenninge's octaves: N octaves with scattering × a^n, extinction × b^n and phase g × c^n. This is what our 3-octave constants already approximate.
  - Beer shadow maps for cloud shadows on the ground and on the atmosphere.
- **Night sky:** Jensen et al. 2001, "A Physically-Based Night Sky Model".
  - The moon from measured albedo and elevation maps with a lunar BRDF.
  - Stars from catalogue position, magnitude and temperature.
  - The Milky Way from a processed photograph.
  - Zodiacal light and airglow.
  - Tone reproduction for scotopic vision, which removes detail rod vision cannot see.
- **Stars on a display:** "Smaller than Pixels" (Eurographics 2025, CosmoScout VR).
  - Sub-pixel stars need their energy spread correctly across the pixels they touch.
  - Pixels toward the edge of a wide view cover less sky, which matters in VR.
  - A perceptual glare filter. 50 M stars in 8 ms at 4K; we need about 9 k.
- **Data** (every download needs your OK first):
  - **Moon:** the NASA SVS CGI Moon Kit (svs.gsfc.nasa.gov/4720). Colour from the LRO Hapke-normalised WAC mosaic, displacement from LOLA (a December 2025 update adds a linear EXR). It is a US-government work, so it should be public domain, but check the page's usage note before shipping it.
  - **Milky Way:** NASA SVS Deep Star Maps 2020 (svs.gsfc.nasa.gov/4851), 1.7 billion stars from Hipparcos-2, Tycho-2 and Gaia DR2, as half-float EXR in celestial coordinates from 4K up.
  - **Bright stars:** the Yale Bright Star Catalogue (BSC5, 9,110 stars), or the Hipparcos main catalogue. Check redistribution terms before bundling (HYG is CC BY-SA).

## 4. The Quest 3 design: realism we can afford

The key fact is that **the sky is at infinity**. With a 64 mm IPD, a cloud 1.5 km up shows 0.002 deg of parallax between the eyes, and walking 100 m shifts it by about one degree. So **sky radiance can be worked out per direction, once, for both eyes**, not per screen pixel per eye. That one change turns cloud cost from "per pixel, per eye, per frame" into "per sky-map texel, a slice per frame". The display pass becomes a few texture fetches, which helps a frame that is limited by pixel fill.

### 4.1 Smooth motion, now (removes the jumps)
- Work out `Astronomy` every frame on the CPU and write it into the uniform: `celestial`, the sun, the moon and the moon's sunward vector. It is a few dozen trig operations. Shadows and the light direction follow it continuously.
- Keep two panoramas, the last one and the next, and **blend** them by the fraction of the way between rebuilds. The sky colour then changes continuously.
- Relighting probes and lightmaps stays thresholded as it is now (`PROBE_RELIGHT_*`), but each relight is blended in over time rather than switched in one step.
- Cost: one extra texture fetch per sky pixel and one panorama's memory (128 KB).

### 4.2 Atmosphere on the GPU (Hillaire)
- **Transmittance** 256x64, **multiple scattering** 32x32: rebuilt only when the atmosphere changes (weather haze, dust).
- **Sky-view** 192x108, with latitude packed toward the horizon and longitude measured from the sun, so the glow around the sun stays sharp. Rebuilt every frame on the GPU.
- **Aerial perspective** 32x32x16 froxels. Distant terrain and the sea then fade into the right haze colour, which matters a lot on the beach and the open terrain.
- The **planet's shadow** inside the sky-view march gives the Earth's-shadow band and the Belt of Venus.
- The moon is a second light at its own irradiance, so a moonlit sky is faintly blue before the rod shift greys it.
- Target under 0.15 ms a frame on the Adreno 740, measured with `pass_timers`.
- This also removes the CPU worker's 256x128 panorama: it is 1.4 deg a texel today, and it is why the rebuilds have to be stepped.

### 4.3 Sun, analytic and cheap
In the sky shader, per pixel near the sun only:
- **Transmittance per pixel across the disc**, so the bottom of a setting sun is redder than the top.
- **Limb darkening per channel**: u of about 0.5, 0.6 and 0.75 for R, G and B. Work the exact values out from Neckel's per-wavelength fit, integrated over each channel.
- **Refraction** (Bennett's formula, with a small difference per channel). This gives the flattened sun at the horizon and the thin green rim.
- **Eye glare as an analytic point-spread function** centred on the sun direction, drawn over geometry as well as sky:
  - It is added in the existing fullscreen pass, so no blur pass is needed (memory: [[glare-in-the-eye]]).
  - It is scaled by how much of the sun is visible: cloud transmittance toward the sun (one fetch from the cloud map) times what the scene leaves showing (a few depth taps at the sun's screen position).
  - Trees and walls then cut the glare off correctly.
- Cost: a branch that only runs near the sun, plus one fullscreen multiply-add term.

### 4.4 Night: real stars, Milky Way, moon
- **Stars:** the BSC5 catalogue, about 9 k stars stored as RA/Dec, magnitude and B-V, under 150 KB.
  - Drawn as instanced quads 2-3 px across, sized from the pixel footprint (`dpdx`), with an energy-normalised Gaussian. A star then keeps the same total light whatever its sub-pixel position, so it is stable when the head turns, unlike the current hash cells.
  - Rotated every frame by sidereal time.
  - Extinction from air mass (already there).
  - Scintillation: a smooth noise at a few Hz, with depth rising with air mass and a colour flicker near the horizon. It must stay coherent from frame to frame, never white noise.
  - About 4 k visible quads is negligible.
- **Milky Way and faint stars:** Deep Star Maps 2020, downsampled to 2048x1024 in celestial coordinates with the catalogue stars subtracted. Stored as RGB9E5 or R11G11B10 (8 MB), or as sRGB ASTC with a scale, since its dynamic range is small. One fetch per sky pixel, rotated with the stars. Zodiacal light and airglow are analytic terms in the sky-view.
- **Moon:**
  - A 256x128 albedo map and a normal map from the CGI Moon Kit. The moon is only about 13 px across on the Quest 3, so 256 texels is already more detail than can be seen.
  - Hapke-lite lighting (Lommel-Seeliger plus the opposition surge).
  - Earthshine set by the Earth's phase as seen from the Moon.
  - Transmittance and refraction near the horizon, like the sun.
  - **Low-precision ephemeris** (Meeus or Schlyter, arcminute accuracy) for the sun and the moon: orbit tilt, the moon's main perturbations, the equation of time. Microseconds a frame.
- **Scotopic vision** driven by the actual adapted luminance (`night_vision_for` already exists): the Purkinje shift and loss of colour.

### 4.5 Clouds: volumetric, time-sliced into a shared sky map
1. **Model (Nubis-style):**
   - A weather map gives coverage, cloud type, precipitation and wind for each altitude band.
   - Height profiles per type: stratus, stratocumulus, cumulus, cumulonimbus.
   - A 128^3 Perlin-Worley base noise and a 32^3 Worley detail noise, both ASTC 3D, or 2D slices if 3D ASTC is not available.
   - Cirrus as its own 2D layer at 8-10 km, the one place 2D is right.
2. **Lighting:**
   - Hillaire's energy-conserving step integration.
   - Wrenninge multiple-scattering octaves (we already have the constants).
   - Two-lobe phase function: forward g 0.8 for silver linings, back g -0.3.
   - The beer-powder term for the dark edges.
   - Ambient from the sky-view's top and bottom.
   - **The sun's colour taken from transmittance at the cloud's own altitude**, so high clouds stay lit after sunset.
   - Lit by the moon at night.
3. **Render into a shared cloud sky map** using the sky-view parameterisation at about 1024x512, with rows packed toward the horizon:
   - Each texel stores in-scattered radiance, transmittance and the distance to the cloud.
   - **Time-sliced:** about 1/16 of the texels are rebuilt each frame, in a Bayer order, into a "next" map. Display blends the complete map into the next one.
   - With no per-pixel reprojection there is nothing to cause VR ghosting, and no TAA is needed.
   - A full rebuild takes about 0.2 s at 72 Hz. A 12 m/s wind at 1.5 km moves the clouds 0.1 deg in that time, which is below what can be seen.
4. **Display:** one bilinear fetch.
   - Detail beyond the map's resolution comes from one fetch of 2D high-frequency erosion noise at display time. It is projected onto the cloud's plane using the stored distance, and it only reshapes the alpha of texels at cloud edges.
   - Edges are then sharp at 25 px/deg while the map stays small.
   - The sun's, moon's and stars' visibility multiplies by the stored transmittance.
5. **Cloud shadows:**
   - A 256^2 Beer shadow map over the play area, rebuilt in slices.
   - The terrain and sun shading read it with one fetch.
   - It also dims the sky-view's in-scattering for crepuscular rays, later and only after pricing them.
6. **Lightning:**
   - The server sends events (position, time, seed), so every headset sees the same strike.
   - At display time a flash term adds `intensity × falloff(distance in the map) × (1 - transmittance)`, so the cloud glows from inside without waiting for a rebuild (Nubis's "near-zero cost" idea).
   - A branching bolt mesh with SpaceWarp motion vectors ([[effects-must-write-motion-vectors]]).
   - Thunder arrives distance / 343 m/s later.
   - The flash lights the scene **as a light**, never by pushing up ambient or exposure. That is the known trap.

**Cost target, to be measured on the headset before it is accepted:**

| Part | Budget | Basis |
|---|---|---|
| Rebuilding the cloud map | 0.6-0.9 ms | 32 k rays a frame (1/16 of 1024x512) × about 48 density steps plus light taps, on the Adreno 740 |
| Displaying the clouds | under 0.1 ms | one or two fetches per sky pixel, which replace the 4-fetch, double-phase shader now run per pixel |

Because the sky-map texel count is fixed, **cost does not grow with resolution or with the number of eyes**. If the measurement comes in over budget, the levers are the slice rate, the step count and SDF-style empty-space skipping (Nubis Cubed), in that order.

### 4.6 Weather on the server ([[server-may-carry-headset-work]])
- The server runs the weather: fronts moving through, coverage changing **by rates** (no pops), cloud type following instability, precipitation from type and thickness, wind by altitude, and lightning events.
- It sends a small weather map: about 64x64 cells over the region, 4 bytes a cell, a few KB, refreshed every few seconds and interpolated on the client.
- Every player then sees the same sky.
- Rain and snow areas (`weather.rs`) come from the precipitation channel.
- Following the protocol rule, the wire field is an `Option`. With `None` the client runs the same simulation locally from the scene's seed, never both at once.

## 5. Order of work

| # | Work | Fixes | Size |
|---|---|---|---|
| 1 | Astronomy every frame, panorama blending | the star, sun and moon jumps | small |
| 2 | Sun disc: per-pixel transmittance, per-channel limb, refraction, eye glare with occlusion | "no realistic sun" | small-medium |
| 3 | Catalogue stars, Milky Way map, moon textures and BRDF, earthshine, ephemeris | "no realistic moon or stars" | medium (needs downloads approved) |
| 4 | GPU sky-view and aerial perspective; planet shadow (Earth's shadow, Belt of Venus) | twilight realism, distant haze | medium |
| 5 | Volumetric cloud sky map with time slicing; types and weather map; cirrus layer | "clouds look poor" | large |
| 6 | Cloud shadows on the ground, then crepuscular rays after pricing them | grounding | medium |
| 7 | Server weather, lightning events and bolt | weather realism | large |

Each step ends with:
- a headset pass-timer measurement against its budget;
- an offline render compared with reference photos;
- a check in Blender Cycles' sky with a volume cloud (memory: [[cycles-reference-harness]]).

## Sources
- Hillaire 2020: [paper](https://diglib.eg.org/handle/10.1111/cgf14050), [SIGGRAPH 2020 slides](https://blog.selfshadow.com/publications/s2020-shading-course/hillaire/s2020_pbs_hillaire_slides.pdf)
- Hillaire 2016, Frostbite sky and clouds: [slides](https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/s2016_pbs_frostbite_sky_clouds.pdf)
- Schneider and Vos 2015: [Real-time Volumetric Cloudscapes of Horizon Zero Dawn](https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn)
- Guerrilla: [Nubis Evolved (2022)](https://www.guerrilla-games.com/read/nubis-evolved), [Nubis Cubed (2023)](https://www.guerrilla-games.com/read/nubis-cubed)
- Jensen et al. 2001: [A Physically-Based Night Sky Model](https://graphics.ucsd.edu/~henrik/papers/nightsky/)
- [Smaller than Pixels: Rendering Millions of Stars in Real-Time (EG 2025)](https://diglib.eg.org/handle/10.2312/egs20251029)
- NASA SVS: [CGI Moon Kit](https://svs.gsfc.nasa.gov/4720), [Deep Star Maps 2020](https://svs.gsfc.nasa.gov/4851)
- Limb darkening: [Tatum, Stellar Atmospheres ch. 6](https://phys.libretexts.org/Bookshelves/Astronomy__Cosmology/Stellar_Atmospheres_(Tatum)/06%3A_Limb_Darkening/6.01%3A_Introduction._The_Empirical_Limb-darkening); flattened suns: [Atmospheric Optics](https://atoptics.co.uk/blog/flattened-suns); colour edges at sunset: [ESO Paranal](https://eso.org/public/belgium-nl/images/paranal-sunset-flashes)
- Mobile cloud optimisation notes: [Vertex Fragment on cloud upsampling](https://www.vertexfragment.com/ramblings/volumetric-cloud-upsampling/), [Unity thread on clouds for a Quest flight sim](https://discussions.unity.com/t/best-approach-to-volumetric-clouds-for-a-flight-sim-on-mobile-quest/890861)
