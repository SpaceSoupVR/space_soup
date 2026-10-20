# Dynamic skies and volumetric clouds: what shipped games do, and what our Quest 3 renderer should take (2026-10-20)

Scope: research only. No code was changed. This report builds on
`space_soup/docs/engine/sky-realism-research.md` (the earlier plan) and the
status in memory `sky-realism-plan-2026-10-20.md`, and does not repeat them.

Code read: `space_soup/src/renderer/cloud_sky.rs` (marcher, map, ground
shadow), `space_soup_sky/src/cloud_layers.rs` (layers, noise, CPU twin),
`space_soup_sky/src/clouds.rs` (constants, the old 2D `CloudFrame`),
`space_soup/src/renderer/sky.rs` (display shader, B-spline read,
`step_clouds`, `tick`), `space_soup_sky/src/time_of_day.rs` (atmosphere and
snapshot).

**The bottom line.**
- Our architecture is right, and a shipped AAA game shows it. Ghost of
  Tsushima rendered its clouds the way we do: a direction-space cloud texture
  (768x768 paraboloid), three textures (two shown, one built over 60 frames),
  and density anti-aliasing from texture derivatives. Ghost of Yōtei (2025)
  then added per-texel depth so the cached clouds could be scrolled with
  parallax.
- What we lack is **not more marching**. It is the cheap things the cached
  map does not do yet:
  - **motion:** the texels are not advected with the wind, so clouds
    cross-fade instead of gliding;
  - **light per altitude:** every layer gets one sun colour;
  - **ambient occlusion:** cloud bases are not darkened;
  - **variety:** spare weather channels are already fetched and left unused;
  - **step placement:** fixed steps waste samples on empty air and
    under-sample the horizon.
- One likely performance cliff sits in the marcher today: local arrays read
  at a runtime index, the same Adreno scratch-memory trap that cost
  0.9-1.6 ms in `sky_irradiance`.

---

## 1. What the leading games and engines do

### 1.1 Guerrilla, Decima "Nubis" (Horizon Zero Dawn 2017, Forbidden West 2022, Burning Shores 2023)

**2015, SIGGRAPH Advances (Schneider):** a 2.5D cloud layer ray-marched from
the ground.
- **Inputs:**
  - a weather map holding coverage, precipitation and cloud type;
  - height-gradient profiles per type (stratus, stratocumulus, cumulus);
  - a 128^3 Perlin-Worley base noise and a 32^3 Worley detail noise;
  - a 2D curl noise that distorts the detail into turbulent wisps.
- **Erosion:** the detail erodes billowy at the top and wispy at the bottom.
  It inverts the Worley noise by height, as our `density()` already does.
- **Lighting:** Beer's law, the "powder" dark-edge term, and Henyey-Greenstein
  silver linings. Light is sampled with 6 taps in a cone toward the sun plus
  one long-distance tap.
- **Marching:**
  - cheap base-only samples until a cloud is hit, then a step back and full
    samples;
  - a return to cheap samples after a run of empty samples;
  - more samples toward the horizon.
- **Reprojection:** quarter resolution, with 1 pixel of every 4x4 block
  updated per frame and the rest reprojected.
- **Cost:** a 2 ms target on PS4; the talk reports about 20 ms brought down
  to about 2 ms.
- Sources: course page and slides [S1]; GPU Pro 7 chapter [S2].
- Our 64^3 noise, height profiles, erosion inversion and per-kind layers
  follow this design.

**2017, Nubis:** regional-scale authoring, animation and transitions between
cloud states, integration with the atmosphere system, and a new lighting
model [S3].

**2022, Nubis Evolved (SIGGRAPH Advances) and "Volumetric Superstorms" (GDC 2022)** [S4][S5]:
- close-up and flyable clouds, at 1080p without temporal upscaling for the
  close range;
- a technique against temporal artefacts in fast-moving clouds;
- "near-zero cost" internal lighting for lightning flashes in superstorms;
- no expensive simulation or lighting, so the system scales from PS4 to PS5.
- Recalled from the slides (the PDFs exceed the fetch limit, so verify these
  before relying on them):
  - a **cloud density profile** (Schneider calls it the "dimensional
    profile"): a field that is 0 at a cloud's edge and rises toward its core;
  - it drives where erosion bites and approximates **ambient occlusion** and
    **dark edges**, so bases and interiors darken without extra light rays.

**2023, Nubis Cubed (SIGGRAPH Advances, Burning Shores)** [S6][S7]:
- voxel clouds modelled with fluid simulation;
- **compressed SDFs for empty-space skipping**;
- detail noise added on top of the voxel density, so dense voxel data does
  not become a memory-bandwidth bottleneck;
- faster light sampling;
- new approximations for dark edges and "inner glow";
- seen from the ground, they still act as a **volumetric skybox with time of
  day**.
- Built in under six months.

**2024, "Synthesizing Realistic Clouds for Video Games"** (Schneider, Lorentz
Center) [S8]: clouds through "the arts of high performance and
approximation". It is the philosophy, not a new technique.

### 1.2 Sucker Punch: Ghost of Tsushima (2020) and Ghost of Yōtei (2025). The closest match to our design.

**Tsushima, Patry, SIGGRAPH Advances 2021** [S9]:
- **Cloud map:** clouds are rendered into a **768x768 paraboloid texture**,
  format R11G11B10F, holding **Mie, Rayleigh and transmittance** as separate
  channels.
- **Three textures:** two used for **scrolling and lerping**, and one being
  rendered, **time-sliced over 60 frames**.
- **Density AA:** cloud density is reduced where the derivatives of cloud
  position against texture coordinate are high. This is the same idea as our
  footprint fade.
- **Phase:** HG with **g that depends on how deep the sample is in the
  cloud**, estimated from the light transmittance and the transmittance of
  the march segment. It runs from about -0.15 (back-scatter) to about 0.85
  (the silver lining).
- **Multiple scattering:** approximated by **scaling back-scatter by about
  2.16**, fitted to a simulation with albedo 0.9.
- **Atmosphere:**
  - pre-computed 3D LUTs with an analytic Earth-shadow term, resampled to 2D
    each frame for the current sun and moon;
  - **cubic upsampling** against banding at sunrise and sunset;
  - a parallel LUT of "radiance divided by phase" to avoid precision problems
    when lighting clouds.
- **Ambient:** sky luminance to SH in 60 µs on PS4.

**Yōtei, PlayStation Blog, 23 Oct 2025** [S10]:
- Each cloud-map texel stores the **average visible depth and average squared
  depth**. The cached clouds can then be **parallax-scrolled** and move
  faster without artefacts.
- It **blends three cloud frames instead of two** for smoother motion.
- The depth statistics give the **opacity distribution along the ray**, which
  is used to put clouds in front of mountains.
- A **light-space cloud shadow map** gives crepuscular rays in the fog, even
  far from the camera.

Takeaway: our direction-space map, the 3-layer rotation and the footprint AA
are the Tsushima design. The parts we have not taken yet are
**scrolling/advection**, **per-texel depth** and the **light-space shadow
map**.

### 1.3 Frostbite (Hillaire 2016) and Unreal Engine (Hillaire 2020; UE5 Volumetric Clouds)

**Frostbite 2016** [S11]:
- energy-conserving step integration;
- Wrenninge-style multiple-scattering octaves [S12];
- a dual-lobe phase function;
- the **Beer Shadow Map (BSM)**, a light-space map holding front depth, mean
  extinction and maximum optical depth, used for shadows on the ground and in
  the atmosphere;
- clouds and atmosphere lighting each other.
- We already have the first three.

**Hillaire 2020** [S13][S14]:
- transmittance, multiple-scattering, sky-view and aerial-perspective LUTs;
- shipped on mobile (Fortnite);
- the course noted that clouds were not reasonably achievable on mobile at
  the time.

**UE5 Volumetric Cloud** (current docs) [S15][S16]:
- **Not supported on mobile.** Sky Atmosphere is supported there.
- **Render modes:**
  - mode 0 traces at quarter resolution, reconstructs at half and upsamples
    to full;
  - mode 1 traces at half resolution;
  - mode 2 is fast but cannot intersect opaque geometry.
- **Multiple-scattering octaves:** 0-2. The defaults are Contribution 0.5,
  Occlusion 0.5 and Eccentricity 0.5, the same as our `OCTAVE_A/B/C`. The
  docs advise **1 octave for games**.
- **Other levers:**
  - Stop Tracing Transmittance Threshold;
  - Conservative Density (a cheap density test that skips the full material);
  - Sky Light Cloud Bottom Occlusion (a ray-free darkening of cloud bases);
  - Cloud Ambient Occlusion (clouds occluding the sky light);
  - Cast Cloud Shadows on Atmosphere (god rays), with the sky-view LUT raised
    to 256x256 for sharper shafts;
  - Beer Shadow Maps recommended on PS4 and Xbox One;
  - Sky Light real-time capture time-sliced over 9 frames.

### 1.4 Rockstar, Red Dead Redemption 2 (Bauer, SIGGRAPH Advances 2019) [S17]

- Sky, clouds, fog and volumetrics are one system: voxelisation plus ray
  marching for scattering and transmittance.
- One system feeds the main view, the reflection maps and a sky-irradiance
  probe grid.
- An ambient model approximates indirect light from the sky.
- Lesson: what you see, what is reflected and what lights the ground come
  from **one** cloud representation. We currently have two:
  - the GPU volumetric map is what is drawn;
  - the CPU's 2D `CloudFrame::shade` model builds the clouded panorama used
    for irradiance and reflections (`time_of_day.rs` ~L806-830).

### 1.5 Unity HDRP Volumetric Clouds [S18][S19]

- **Cloud map:** R coverage, G rain (darkening), B type.
- **Cloud LUT,** indexed by type and height:
  - R the density profile;
  - G erosion susceptibility;
  - B an **ambient-occlusion multiplier**.
- **Layers in Advanced mode:** cumulus, altostratus and cumulonimbus (anvil).
- **Controls:**
  - Powder Effect Intensity, Multi Scattering and Erosion Occlusion;
  - temporal accumulation with ghosting reduction;
  - Num Primary Steps and Num Light Steps;
  - shadow map resolution and distance, with an opacity fallback outside it.
- **Presets:** Sparse, Cloudy, Overcast and Stormy.

### 1.6 Godot 4 [S20]

- The sky shader has `use_half_res_pass` and `use_quarter_res_pass`
  sub-passes, and the docs' example computes clouds only at the reduced
  resolution.
- The radiance cubemap updates only on triggers: TIME, a moving POSITION, a
  light change or a uniform change. There is no built-in volumetric cloud
  system.

### 1.7 Mobile and VR

- **Mobile clouds in UE5** (Ghaemian, 80.lv, 30 Mar 2023) [S21]:
  - Adreno 540, 64 view steps and 2 light steps;
  - rendered into a **320x180 render target** with a 4-tap blur;
  - about 300 ms unoptimised became about 40 ms with step and mip limits;
  - only the low-resolution target reached a solid 60 fps.
  - That low-resolution target is our sky map, except ours is in direction
    space, which is the right space for VR.
- **Screen-space upsampling** (Vertex Fragment) [S22]: 1/16 pixel refresh over
  a 4x4 block.
  - Plain bilinear upsampling pixelates the borders.
  - Refreshing too few pixels per frame shears.
  - Skipping pixels by branching gave nothing, because the wave waits for the
    slowest lane. The work has to shrink at the buffer level, as our map
    slices do.
- **VR specifics:**
  - Billboard cloud particles visibly rotate in VR, so volumetric is preferred
    [S23]. Quest 2 hobbyists fall back to particles [S24].
  - **No published Quest title** with ray-marched volumetric skies was found.
- **Adreno guidance** [S25][S26]:
  - On A5x, 1 texture fetch per fragment at full rate is worth about 16 FP32
    ALU operations, so fetch count is the cost model.
  - Clear or don't-care load ops avoid GMEM loads.
  - Precision: our own measurement found f16 bought nothing
    (memory `half-precision-bought-nothing`).
  - The trap that cost real milliseconds here was **local arrays read at a
    runtime index** (memory `adreno-local-array-cliff`).

### 1.8 Others (thin or secondary evidence; recorded so nobody re-searches it)

- **MSFS 2020/2024:** no primary Asobo technical talk found. Forum
  observations describe low-resolution volumetric clouds with temporal
  upscaling and distance LOD, and halos round aircraft from compositing
  low-resolution clouds [S27]. Not a usable reference.
- **Star Citizen:** a "bricked" voxel format for gas clouds (monthly report,
  June 2016) and volumetric planetary clouds (CitizenCon 2021). Secondary
  sources only [S28].
- **Assassin's Creed / Anvil:** no cloud talk found. The nearest is
  Wronski's GDC 2014 volumetric fog [S29].
- **Genshin Impact and Sky: Children of the Light:** no sky or cloud talk
  found. The miHoYo Unity talk does not cover the sky [S30].
- **Horizon Call of the Mountain (PSVR2):** no source confirms it uses Nubis.

### 1.9 Physics references worth adopting

- **Jendersie and d'Eon 2023 (NVIDIA, SIGGRAPH Talk)** [S31]: an HG + Draine
  blend that matches about 95% of the Mie phase over droplet diameters of
  5-50 µm, analytic and sampleable. It gives the narrow, very bright
  diffraction peak round the sun (the real silver lining), a broad forward
  lobe, and a back-scatter rise toward the glory.
- My recollection of the paper's fits, to check against the PDF before use:
  - `g_HG = exp(-0.0990567/(d-1.67154))`
  - `g_D = exp(-2.20679/(d+3.91029) - 0.428934)`
  - `alpha = exp(3.62489 - 8.29288/(d+5.52825))`
  - `w_D = exp(-0.599085/(d-0.641583) - 0.665888)`
  - At d = 20 µm these give about g_HG 0.995, g_D 0.594, alpha 27.1 and
    w_D 0.498.
- **Spatiotemporal blue noise** (Wolfe et al., EGSR 2022) [S32]: blue noise
  in space and in time. It converges faster and stays more stable under
  temporal filtering than per-frame IGN or white noise when ray-march offsets
  are accumulated. Volumetric rendering is one of its demonstrations.

---

## 2. Our implementation against the field

| Area | Ours (file) | Leading practice | Gap |
|---|---|---|---|
| Architecture | Direction-space map 2048x768 RGBA16F, el = 90°·v^1.5, 16 rows a frame, 48 frames a map, 3 layers rotated, prev/cur blend (`cloud_sky.rs`) | Tsushima: 768² paraboloid, 3 textures, 60-frame slice, **scroll + lerp**; Yōtei: **depth + depth² per texel, parallax scroll, 3-frame blend** | **No advection.** Clouds cross-fade between maps up to ~0.67-1.3 s apart. A 12 m/s wind at 1.5 km overhead moves 0.46°/s ≈ 11 px/s, so the two blended images are about 7 px apart, which shows as soft, pulsing edges. Each map's rows are also marched at different times, so a map is **sheared** by up to 0.67 s of wind from its top row to its bottom row. |
| Map contents | rgb (baked light), a (sky showing) | Tsushima: Mie/Rayleigh/T split; Yōtei: depth stats | **No depth channel.** That blocks advection, lightning flash lookup and the opacity distribution. |
| View march | Fixed counts [10, 30, 8], up to 14 km a layer, IGN jitter only on long steps | Nubis: cheap samples until hit, refine, back off; more steps at the horizon; Nubis³ SDF skipping; UE Conservative Density | Steps are **uniform in distance**. Horizon cumulus rays take 467 m steps, coarser than the clouds (the stripes), while empty air gets the same full-cost samples. |
| Light sampling | Up to 4 taps of full density along the light, same layer only (`light_tau`) | Nubis: 6-tap cone + 1 far tap; Frostbite/UE: **Beer Shadow Map**; Nubis³: accelerated light sampling | About 8 of the ~11 fetches per cumulus sample are light taps. **No cross-layer shadowing in the view march**: an altocumulus sheet does not dim the cumulus under it. |
| Sun colour on clouds | One irradiance at the cumulus base altitude for all layers (`time_of_day.rs` L807-810) | Hillaire/Tsushima: transmittance at the sample's altitude, plus the Earth's shadow | **Cirrus at 8.5 km and altocumulus at about 4 km get the cumulus's sun.** They should stay pink or gold for minutes after the cumulus go grey (2.9° of extra sun for 8 km cirrus). |
| Ambient | `amb_sky·(0.45+0.55h) + amb_ground·0.35(1-h)`, h = height in the *layer* | Nubis density profile; HDRP LUT AO; UE cloud-bottom occlusion; Tsushima SH | **Bases are not occluded** by the cloud above them (the "flat white noon cumulus" complaint). h is measured in the layer, not under the local cloud top. The ambient is one colour with no direction: no warm horizon light on cloud sides at sunset. |
| Phase | 0.7·HG(0.8) + 0.3·HG(-0.3) for every octave (g scaled by c) | Tsushima: g by depth (-0.15 to 0.85); Jendersie-d'Eon HG+Draine | Silver linings are broad and dim compared with Mie's sharp peak. `phase()` is evaluated 3 times per *sample*, although cv is constant along a ray. |
| Shape variety | 4 kinds, each with its own wind and wandering coverage; weather `.r` only | HDRP cloud map RGB + type LUT; Nubis cloud types and anvils | Every cumulus cell has the **same top** (set only by the layer's mean coverage). No wind shear. Weather `.g/.b/.a` are already fetched (the RGBA sample) and unused by `local_coverage`. The detail noise is the base texture read at 4x. |
| Aerial perspective on clouds | Grey `exp(-t·HAZE)` fading to the sky behind | Hillaire AP LUT (per channel) | Fine in structure, but **grey**: distant cumulus should yellow and dim per channel. `sky_extra.extinction` is already in the uniform. |
| Ground shadow | 64² over 1 km, 4 penumbra rays a texel, rebuilt whole every frame; mean beyond | UE/Frostbite BSM cascades (km range); Yōtei light-space map → god rays | **No cloud shadows beyond about 500 m**: distant hills and sea never show shadows sweeping across them, a signature of a real sky. 4 rays × 22 steps × 3,844 texels ≈ 0.7 M fetches a frame. |
| Irradiance and reflections | CPU 2D `CloudFrame::shade` panorama | RDR2: one volumetric representation for view, reflections and probes | **What you see differs from what lights the world and what water reflects** (the memory notes reflections lack the new clouds). |
| Crepuscular rays | none | UE cloud shadow on atmosphere; Yōtei light-space map; Hillaire sky-view march | Waits on step 4 (GPU atmosphere). |
| Lightning | planned (step 7) | Nubis Evolved: near-zero-cost internal flash | Needs the depth channel. |
| Night | Light becomes the moon; scale-normalised maps | RDR2/Tsushima: moon-lit clouds; town glow | Light pollution does not light cloud bases (`amb_ground` holds no skyglow), although `LIGHT_POLLUTION_CD` exists. |
| Shader hygiene | `var steps = {VIEW_STEPS}` read as `steps[k]` in `fs_map`; `var reach = array<f32,4>` read in the `light_tau` loop; `var steps = {SHADOW_STEPS}` in `transmittance_toward` | Adreno: a local array read at a runtime index goes to **scratch memory** (memory `adreno-local-array-cliff`: 0.9-1.6 ms saved by unrolling one such pair) | **Probable silent cost** in the hottest loops of the march and the ground shadow. |

What we already match:
- energy-conserving integration;
- Wrenninge octaves at UE's defaults;
- height-inverted erosion (Schneider);
- footprint density AA (Tsushima's derivative AA);
- the direction-space, time-sliced cache (Tsushima);
- a scale-normalised map for night precision (Tsushima's radiance-over-phase
  LUT solves the same problem);
- penumbra rays;
- a CPU twin tested against the GPU.

---

## 3. Cost model (read before the numbers)

Nothing in the cloud path has been measured on the headset yet (memory: "GPU
cost unmeasured"). The estimates below count texture fetches, since Adreno
fetch cost dominates (one fetch ≈ 16 FP32 ALU on A5x [S25]). They are ranges
for ranking ideas, not budgets.

- **March (fixed cost, independent of eyes and resolution):**
  - 32,768 rays a frame.
  - Fetches per sample with density: cumulus about 11 (weather, base, detail
    and 4 light taps × 2), strato and alto about 7. Empty steps cost 1-2.
  - About 100-200 fetches per ray in a typical sky, so 3-7 M fetches a frame.
  - Rough guess: **0.4-1.5 ms**, more if the local arrays spill to scratch.
- **Ground shadow:** about 0.7 M fetches a frame, roughly 10-20% of the
  march.
- **Display (scales with pixels, both eyes):**
  - Sky covers perhaps 30-50% of 2 × ~4.5 Mpx (less with FFR), so about
    2.5-4 M pixels.
  - Each pixel takes 8 RGBA16F bilinear taps (B-spline × 2 maps) plus the
    panorama and the sun, moon and star logic.
  - The map is magnified about 4.5x, so the cache hit rate is high.
  - Rough guess: **0.4-1.0 ms**.
- **The frame is fill-bound** (memory `quest3-frame-is-fill-bound-not-pass-bound`):
  - The display pass is the one that grows with resolution. Every per-pixel
    tap added there is paid about 3 M times.
  - Work moved into the march is paid 32 k times per frame.
- **Rule used in the ranking:** prefer changes that add work in the march or
  on the CPU, and keep the display at about 8-10 taps.

Before anything else, add pass timers for `cloud_sky_map`, `cloud_sky_shadow`
and the clouded sky display separately (`quest-gpu-profiling-per-draw`).

---

## 4. Ranked recommendations

The ranking is realism gained per ms, with the cost on the Quest 3. "Fits"
says whether the idea fits the cached-map design.

### R0. Do first, no visual change: remove the runtime-indexed local arrays from the march
- **What:** in `cloud_sky_shader()`:
  - `fs_map`: replace `var steps = {VIEW_STEPS}; ... steps[k]` with a
    `select` chain or a uniform.
  - `light_tau`: replace `var reach = array<f32,4>(...)` with explicit
    constants per tap; the loop is at most 4 iterations, so write them out.
  - `transmittance_toward`: the same for `var steps = {SHADOW_STEPS}`.
  - Extend the existing guard test (`the_lights_block_declares_no_local_arrays`)
    to cover `cloud_sky_shader()` and `sky_cloud_shader()`.
- **Gain:** none visually. It is perf headroom for everything below.
- **Cost:** negative. Precedent: 0.9-1.6 ms per view from one such pair in
  `sky_irradiance`. Here it is fixed-cost (32 k rays), so expect less, but it
  is the cheapest win available.
- **Fits:** yes.

### R1. Advect the map: store cloud depth, give each row its build time, and parallax-scroll at display (Tsushima, Yōtei)
- **What:**
  1. Add a second colour target to `map_pipeline`, RG16F, holding:
     - R = the transmittance-weighted mean distance t̄ along the ray (in km);
     - G = t̄² (Yōtei's second moment; also used for the opacity
       distribution and lightning falloff).
  2. Put the map's start time and the rows-per-frame rate in the uniform
     (`cloud_misc`), and each layer's wind velocity as well. A texel's age is
     then exact: `age = now - (start + row/rows_per_frame · dt)`.
  3. In `sky_clouds()`:
     - Fetch t̄ at d (1 bilinear tap) and form `P = d·t̄`.
     - Take the layer from `altitude_along(eye, mu, t̄)` and its wind w_k.
     - Compute `d_src = normalize(P − w_k·age + (eye_now − eye_build))`, then
       run the B-spline read at `uv(d_src)`.
     - Do this per shown map (prev and cur have different ages).
  4. Once motion is advected, raise `MAP_SLICES` from 48 to 96 (or more). This
     halves the march's per-frame cost, because the cross-fade no longer has
     to stay short to hide motion.
- **Gain: high.**
  - Clouds glide continuously instead of breathing or cross-fading.
  - No row shear.
  - The eye's own movement is handled too (it is small, but free).
- **Cost:**
  - Display: +1-2 taps and about 20 ALU per sky pixel, roughly +0.05-0.15 ms.
  - March: one more RG16F write per texel (negligible). With 96 slices, about
    −0.2 to −0.7 ms.
  - Memory: +3 × 6.3 MB.
  - **Net is probably negative.**
- **Fits:** exactly. This is the shipped evolution of our design.

### R2. Light each layer with the sun at its own altitude, including the Earth's shadow
- **What:** on the CPU (`time_of_day.rs` where `CloudFrame` is built, and
  `CloudSkyUniform::new`):
  - Compute `light_k = top · atm.transmittance_at(R + alt_k, mu_sun)` for each
    layer's mid altitude, with mu taken past the horizon dip
    `acos(R/(R+alt_k))` so the planet's shadow switches each layer off at its
    own time.
  - Pass it per layer, in the spare `w` of `layers[k].v[2]` plus one more
    vec4, or as 4 vec4s.
  - `in_scatter` and the cirrus term use `light_k`.
- **Gain: high at sunrise and sunset**, the most-watched moments:
  - cirrus and altocumulus glow pink over grey cumulus;
  - high clouds stay lit for minutes after the ground goes dark;
  - the order of colours across the layers reads as real.
- **Cost:** about 0 (a few CPU LUT reads per snapshot or frame; uniform
  only).
- **Fits:** yes.

### R3. Darken cloud bases: ambient occlusion from depth under the local top, and directional sky ambient
- **What:** in `density()`, also return the local top `top(cover)` (cumulus
  `0.3+0.7·cover`; it is already computed). In `in_scatter`:
  - **Height in the cloud:** `h_c = h / top_local`.
  - **Occlusion above:** about `exp(-k·sigma·d_local·(top_local − h)·thick)`.
    It costs nothing, since it uses only values already in hand. This is the
    Nubis density-profile, HDRP LUT-AO and UE cloud-bottom-occlusion idea.
  - **Ambient:** `amb = amb_top·AO_up·f(h_c) + amb_side·g(h_c) + amb_ground·(1-h_c)`,
    where `amb_side` is L1 SH of the sky evaluated horizontally toward the
    ray's azimuth.
    - Computed once per ray, it gives sunset-warm sides facing the sun and
      blue sides facing away.
    - The SH already exists on the CPU (`SkyIrradiance`). Pass 4 vec4s.
- **Gain: high.** It answers "noon cumulus are flat white": contrast from
  dark grey bases to bright tops, and coloured sides at sunset.
- **Cost:** about 0 (ALU per sample; one SH evaluation per ray). Calibrate
  against the Cycles reference harness.
- **Fits:** yes.

### R4. Variety for free: use the weather channels already fetched, and add wind shear
- **What:** `local_coverage` reads `textureSampleLevel(weather_tex,...).r`;
  the G, B and A channels of that same sample are free.
  - **Per-cell top height:** humilis to congestus, scaling `top` in
    `height_profile`.
  - **Base flatness or type mix:** stratocumulus against cumulus within a
    layer.
  - **Erosion style:** wispy against billowy, the HDRP LUT's erosion
    susceptibility.
  - **Wind shear:** offset the noise xz by `(alt − base)·shear_k`, so towers
    lean downwind and shapes shear with height as real ones do.
  - The CPU twin (`cloud_layers::density`) must follow line for line. The
    `CloudModel` coverage calibration must then sample the varied tops.
- **Gain: high** for "randomized, multiple types, each its own". It breaks
  the cookie-cutter look of equal heights.
- **Cost:** about 0 (no new fetches; a few ALU).
- **Fits:** yes.

### R5. Put steps where the clouds are: empty-space skipping and adaptive steps (Nubis 2015 and Nubis³, UE Conservative Density)
- **What:**
  - Build a **max-coverage mip chain**, or a 2D distance field to cover > 0,
    per layer channel of the 256² weather map. Build it once per weather
    field, on the CPU.
  - In `fs_map`, march each layer in coarse steps that test only coverage (1
    2D fetch) or base-only density. On a hit, step back and take fine steps
    of fixed metres (for example 60-120 m for cumulus) under a budget. Return
    to coarse after N empty fine steps.
  - Keep the total under today's worst case, and let horizon rays spend their
    budget where the cloud is instead of over 14 km of air.
- **Gain: high at the horizon.**
  - Horizon rays resolve 200-500 m cumulus instead of striping.
  - Thin wisps stop being skipped.
- **Cost:** neutral to −30% march fetches. Most sky texels are clear or
  cross little cloud.
- **Fits:** yes. Wavefront coherence is good, because neighbouring texels in
  a row are neighbouring directions.

### R6. One light-space Beer Shadow Map for three jobs: the march's light, far ground shadows, and later the god rays (Frostbite, UE, Yōtei)
- **What:**
  - Build a sun-aligned orthographic map, for example 256² over about 24 km
    (about 94 m a texel), snapped to the eye.
  - Store Frostbite's three values: front depth, mean extinction and maximum
    optical depth. Build it time-sliced at 1/8 a frame from the same density,
    advected by wind × age as in R1.
  - **March light:** inside its extent, replace `light_tau`'s 3 long taps
    with one BSM read: `OD ≈ clamp((s − front)·mean_ext, 0, maxOD)`. Keep the
    single shortest tap (0.05·thick) for small self-shadow detail.
  - **Cross-layer shadowing:** comes with it (all layers are integrated into
    the BSM).
  - **Ground:** keep the fine 64² map for the near 500 m, and read the BSM as
    the **far cascade** to about 12 km instead of the mean.
- **Gain: high.**
  - Shadows sweep across distant hills and the sea.
  - Correct dimming of cumulus under altocumulus and altostratus.
  - It is the prerequisite for R12's shafts.
- **Cost:**
  - Build: 8 k texels a frame × about 30 density evaluations, roughly
    0.5 M fetches, about the same as today's ground shadow.
  - March: cumulus samples drop from about 11 to about 6 fetches
    (−30 to −45% of the march).
  - Terrain shader: +1 fetch per pixel for the far cascade.
  - Watch the terrain reader's I-cache cliff (memory
    `terrain-reader-under-the-cliff`).
- **Fits:** yes (a fixed-cost, light-space cache).

### R7. A truer phase function, hoisted out of the step loop
- **What:**
  - Compute the phase values once per ray in `fs_map`, since cv is constant;
    `in_scatter` calls `phase()` 3 times per sample today.
  - Then upgrade to Jendersie-d'Eon HG+Draine (d of about 15-20 µm) for
    single scattering. Use the broad Draine lobe, or Tsushima's
    depth-dependent g (-0.15 to 0.85 by `tau_l`), for the higher octaves.
  - Keep Tsushima's lesson: let back-scatter rise in dense interiors.
- **Gain: medium-high.**
  - Sharp, very bright silver linings when a cloud edge crosses the sun.
  - A dimmer anti-sun side with a faint back-scatter lift.
  - More natural thin edges.
- **Cost:** negative to about 0 (hoisting saves more ALU than Draine adds).
- **Fits:** yes. Our map at 0.18° resolves the forward peak well enough
  outside the disc.

### R8. Accumulate across rebuilds in map space, with blue-noise jitter
- **What** (needs R1):
  - When marching a texel, also read the shown map at the advected source,
    then write `mix(old, new, α)` with α about 0.35-0.5.
  - Reject history where alpha changed by more than about 0.3 (a new or gone
    cloud).
  - Jitter every step's position per rebuild with STBN [S32], or IGN plus a
    golden-ratio offset per rebuild, instead of jittering only long steps.
- **Gain: medium.** Smoother gradients and edges, and the dot pattern from
  per-texel jitter averages away. In effect 2-3x the steps.
- **Cost:** +1 map fetch per marched texel. Then lower `VIEW_STEPS` from
  [10,30,8] to about [8,20,6] at the same look, a net saving of about
  20-30% of the march.
- **Fits:** yes. Direction-space accumulation has no stereo or ghosting
  issues, unlike screen-space TAA.

### R9. One sky for everything: irradiance SH and the reflection cube from the GPU-composited sky
- **What:**
  - When a map completes (about every 0.7-1.4 s), run a small pass that
    composites the clear panorama with the cloud map into a 64x32 lat-long:
    an 8x8 box per output texel made of 16 bilinear taps.
  - Read it back asynchronously (1-3 frames of latency is harmless at sky
    timescales).
  - Feed the existing SH and reflection builders in place of the CPU's 2D
    `CloudFrame::shade` clouded panorama.
  - The relight thresholds (`PROBE_RELIGHT_*`) stay as they are.
- **Gain: medium-high.**
  - The sea and water reflect the real volumetric clouds (a known gap).
  - Overcast ambient matches the clouds you see.
  - Retires the second cloud model (RDR2's "one representation" lesson).
- **Cost:** about 0.02-0.05 ms a map, about 0 a frame.
- **Fits:** yes.

### R10. Chromatic aerial perspective on clouds, and skyglow on cloud bases at night
- **What:**
  - **Haze:** in `fs_map`, replace grey `exp(-t·HAZE)` on the cloud's rgb
    with a per-channel transmittance from the atmosphere's extinction (as in
    `sky_extra.extinction` and `sky_air`), weighted to the path's mean
    altitude. Keep the grey weight for alpha.
  - Later, step 4's aerial-perspective LUT gives the exact form.
  - **Night:** add the light-pollution radiance (`LIGHT_POLLUTION_CD ·
    POLLUTION_COLOUR`, upward) into `amb_ground`.
- **Gain: medium.**
  - Distant cumulus walls turn cream and amber toward the horizon.
  - Low clouds over towns glow orange, and dark-site clouds stay darker than
    the starry sky.
- **Cost:** about 0.
- **Fits:** yes.

### R11. Optics on thin cloud: the "watery sun", 22° halo and sundogs
- **What:**
  - In `sky_sun_disc` (near-sun branch only), read the cirrus and altocumulus
    optical depth toward the sun (CPU value in the uniform, as
    `sky_halo_through` already does).
  - Use it to broaden the aureole and to add a 22° ring (and sundogs at about
    22° left and right, sun-elevation-dependent) when cirrostratus covers the
    sun.
  - Do the same for a lunar corona through altocumulus.
- **Gain: medium-low.** These are unmistakable "real sky" cues on high-cloud
  days.
- **Cost:** about 0 (an analytic ring under an existing angular test).
- **Fits:** yes (display, analytic).

### R12. Crepuscular and anticrepuscular rays, inside step 4's GPU sky-view LUT
- **What:**
  - When the Hillaire sky-view LUT moves to the GPU, sample R6's BSM in its
    march to shadow the single-scatter sun term (UE "cloud shadow on
    atmosphere"; Yōtei).
  - Raise the sky-view LUT toward 256x128 (UE's guidance for shafts).
  - Optionally store the "fraction of the air path in shadow" in the
    sky-view LUT, so the display needs no extra taps.
- **Gain: high when the sun is low and the air is hazy.** Less often seen,
  hence ranked here.
- **Cost:** about 0.05-0.15 ms (fixed-size LUT).
- **Fits:** yes.

### R13. Crisper overhead detail: its own small detail noise, with optional curl distortion
- **What:**
  - Add a separate tileable 32³ Worley-fbm detail texture (128 KB). Read it
    at about 12-20x the base frequency instead of re-reading the 64³ base at
    4x. The fetch count stays the same, and a smaller texture improves cache
    use.
  - Optionally distort the detail coordinates by a 128² curl texture
    (Schneider 2015).
  - Keep the footprint fade (`fine_a`, `fine_b`, `gone`).
- **Gain: medium.** Cauliflower texture on near and overhead cumulus, where
  the 64³ base is heavily magnified (about 40 m a texel at the default
  scale).
- **Cost:** about 0; +1 2D fetch per detail sample for curl.
- **Fits:** yes.

### R14. Cheaper ground shadow
- **What:**
  - One ray a texel plus a 3x3 tent filter in place of the 4 penumbra rays.
    The penumbra is about 20-40 m and a texel is 16 m, so the tent covers it.
  - Or rebuild a quarter of the map each frame, advected.
- **Gain:** none visually.
- **Cost:** −0.5 M fetches a frame (about −0.05-0.15 ms).
- **Fits:** yes.

### R15. Stop oversampling the zenith
- **What:** azimuth is 2048 texels at every row, so above about 55° texels
  are 2-6x finer than needed.
  - Use a second, half-width band for v > 0.7, or a polar cap.
  - The display picks a band by elevation, with a blend across the seam.
- **Gain:** none visually.
- **Cost:** −15 to −25% of march rays.
- **Fits:** yes, with moderate complexity. Do it after the R1, R5 and R8
  numbers are in.

### R16. Lightning at near-zero cost (with step 7 server events)
- **What:** at display, add
  `I·falloff(|P − strike|)·(1 − T)·tint`, with P from R1's t̄ (and t̄² for
  spread). The clouds light up from inside without a rebuild (Nubis
  Evolved). The flash also lights the scene **as a light**, never by pushing
  up ambient or exposure (the doc's trap).
- **Gain:** high when storms exist.
- **Cost:** about 0 (display ALU only while a flash is live).
- **Fits:** yes.

### R17. Weather states beyond the four kinds
- **What:**
  - an altostratus or nimbostratus deck (featureless grey, the common
    overcast sky);
  - cumulonimbus with an anvil (the HDRP Cb layer; Nubis superstorms);
  - virga and rain shafts from the precipitation channel;
  - transitions by rates, from the server weather map (step 7).
- **Gain:** high for weather realism.
- **Cost:** about 0-0.2 ms, depending on the layers active.
- **Fits:** yes. It is an authoring and model task.

### Considered and not recommended now
- **A relightable map** (store sun-scatter, ambient-scatter and transmittance
  instead of baked RGB, as Tsushima splits Mie and Rayleigh). Per-layer sun
  colours (R2) would need 5 or more channels and a second RGBA16F per map,
  which doubles the display taps. Light changes slowly against the map
  period, and lightning is covered by R1's depth plus R16. Revisit only if
  baked light lag shows.
- **Screen-space quarter-res plus temporal reconstruction** (UE mode 0, HZD
  4x4). This is per eye, per pixel, and reprojection under head motion is
  where VR ghosting comes from. Our direction-space cache is strictly better
  for the headset.
- **f16 in the marcher.** Our own measurement found it bought nothing, and it
  cost 0.45 ms in one view (memory `half-precision-bought-nothing`).
- **Voxel or SDF clouds (Nubis³)** for the ground view. They are worth it only
  for fly-through. R5's 2D coverage distance field is the cheap subset that
  matters from the ground.
- **Neural or precomputed radiance** (for example Disney's Deep Scattering
  2017). Far beyond the Quest's budget, and no shipped mobile precedent.

---

## 5. Summary table

| Rank | Change | Realism | Quest 3 cost (est.) | Files and functions |
|---|---|---|---|---|
| R0 | Remove indexed local arrays | (enabler) | **−** (possibly large) | `cloud_sky.rs` `fs_map`, `light_tau`, `transmittance_toward` |
| R1 | Depth channel + wind advection, per-row age, then 96 slices | High: smooth motion | +0.05-0.15 display, −0.2-0.7 march: **net −** | `CloudSky::encode`, `fs_map`, `sky.rs` `sky_clouds` |
| R2 | Sun at each layer's altitude, with Earth's shadow | High at twilight | ~0 | `time_of_day.rs` snapshot, `CloudSkyUniform`, `in_scatter` |
| R3 | Base AO under the local top + SH-directional ambient | High: noon contrast | ~0 | `density`, `in_scatter`, uniform |
| R4 | Per-cell tops, types and erosion from weather G/B/A; wind shear | High: variety | ~0 | `local_coverage`, `height_profile`, CPU twin, `CloudModel` |
| R5 | Coverage distance field + adaptive steps | High at the horizon | 0 to −30% march | `fs_map`, `CloudField` (build a mip or SDF) |
| R6 | Light-space Beer Shadow Map (march light, far ground shadow) | High: far shadows, layer shading | build ≈ today's shadow; march −30-45% | new pass in `cloud_sky.rs`; terrain read |
| R7 | Hoisted phase; HG+Draine or depth-g | Medium-high: silver linings | − to 0 | `phase`, `in_scatter`, `fs_map` |
| R8 | Map-space accumulation + STBN jitter; fewer steps | Medium: smoothness | net −20-30% march | `fs_map` (needs R1) |
| R9 | SH and reflections from the GPU sky | Medium-high: water, ambient | ~0 | new reduction pass; `time_of_day.rs` clouded pano |
| R10 | Chromatic cloud haze; skyglow on bases | Medium | ~0 | `fs_map`, `CloudFrame` ambient |
| R11 | Watery sun, 22° halo, sundogs, lunar corona | Medium-low | ~0 | `sky_sun_disc`, `sky_moon` |
| R12 | God rays in the GPU sky-view LUT (needs R6, step 4) | High in hazy, low sun | 0.05-0.15 | step 4 |
| R13 | Own 32³ detail noise (+ curl) | Medium (overhead) | ~0 | `cloud_layers.rs` `CloudNoise`, `density` |
| R14 | One-ray tent ground shadow | (perf) | −0.05-0.15 | `fs_shadow` |
| R15 | Zenith band at half width | (perf) | −15-25% rays | `map_dir` / `map_uv`, display |
| R16 | Lightning flash from depth | High in storms | ~0 | display (needs R1) |
| R17 | Altostratus/Ns decks, Cb anvils, virga | High for weather | 0-0.2 | `cloud_layers.rs`, step 7 |

**Suggested order:**
1. R0 and the timers.
2. R2, R3 and R4: zero cost, the biggest visible change for the user's
   complaints.
3. R1 then R8: motion and smoothness, and they pay for themselves.
4. R5.
5. R6, which unlocks R12 and far shadows.
6. R7, R9 and R10.
7. The rest.

Measure each on the headset against the section 4.5 budget of the plan
(0.6-0.9 ms for the map, under 0.1 ms for the display). Note that the
display at 8 taps is probably already over that 0.1 ms. That is worth
knowing before R1 adds taps, and R15 does not help it.

---

## 6. Sources (dates as published; web pages accessed 2026-10-20)

- [S1] Schneider, "The Real-time Volumetric Cloudscapes of Horizon: Zero Dawn", SIGGRAPH 2015 Advances in Real-Time Rendering (Aug 2015). Course page: https://advances.realtimerendering.com/s2015/index.html ; slides: https://advances.realtimerendering.com/s2015/The%20Real-time%20Volumetric%20Cloudscapes%20of%20Horizon%20-%20Zero%20Dawn%20-%20ARTR.pdf
- [S2] Schneider, "Real-Time Volumetric Cloudscapes", GPU Pro 7 (CRC Press, 2016).
- [S3] Schneider, "Nubis: Authoring Real-Time Volumetric Cloudscapes with the Decima Engine", SIGGRAPH 2017 Advances (31 Jul 2017). https://www.guerrilla-games.com/read/nubis-authoring-real-time-volumetric-cloudscapes-with-the-decima-engine ; https://advances.realtimerendering.com/s2017/index.html
- [S4] Schneider, "Nubis, Evolved", SIGGRAPH 2022 Advances (Aug 2022). https://www.guerrilla-games.com/read/nubis-evolved ; https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-NubisEvolved-NoVideos.pdf
- [S5] Schneider, "The Real-Time Volumetric Superstorms of Horizon Forbidden West", GDC 2022. https://www.gdcvault.com/play/1027688/The-Real-Time-Volumetric-Superstorms
- [S6] Schneider, "Nubis Cubed" (Nubis³), SIGGRAPH 2023 Advances (Aug 2023; supplementary voxel cloud pack updated 6 Sep 2023). https://www.guerrilla-games.com/read/nubis-cubed ; https://advances.realtimerendering.com/s2023/
- [S7] Slides PDF: https://advances.realtimerendering.com/s2023/Nubis%20Cubed%20(Advances%202023).pdf (exceeds the fetch limit; details from the abstract and project page)
- [S8] Schneider, "Synthesizing Realistic Clouds for Video Games", Lorentz Center (25 Jul 2024). https://www.guerrilla-games.com/read/synthesizing-realistic-clouds-for-video-games
- [S9] Patry, "Real-Time Samurai Cinema: Lighting, Atmosphere, and Tonemapping in Ghost of Tsushima", SIGGRAPH 2021 Advances (Aug 2021). https://www.advances.realtimerendering.com/s2021/jpatry_advances2021/index.html
- [S10] Bentley (Sucker Punch), "Ghost of Yōtei – tech deep dive", PlayStation Blog (23 Oct 2025). https://blog.playstation.com/?p=411201
- [S11] Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite", SIGGRAPH 2016 PBS course (Jul 2016). https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/s2016_pbs_frostbite_sky_clouds.pdf ; https://www.ea.com/frostbite/news/physically-based-sky-atmosphere-and-cloud-rendering
- [S12] Wrenninge, Kulla, Lundqvist, "Oz: The Great and Volumetric", SIGGRAPH 2013 Talks (multiple-scattering octaves).
- [S13] Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique", EGSR 2020 / CGF 39(4). https://diglib.eg.org/handle/10.1111/cgf14050
- [S14] Hillaire, "Physically Based and Scalable Atmospheres in Unreal Engine", SIGGRAPH 2020 PBS course slides. https://blog.selfshadow.com/publications/s2020-shading-course/hillaire/s2020_pbs_hillaire_slides.pdf
- [S15] Epic, "Volumetric Cloud Component" (UE5 docs, current). https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-cloud-component-in-unreal-engine
- [S16] Epic, "Volumetric Cloud Component Properties" (UE5 docs, current). https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-cloud-component-properties-in-unreal-engine
- [S17] Bauer, "Creating the Atmospheric World of Red Dead Redemption 2: A Complete and Integrated Solution", SIGGRAPH 2019 Advances (slides updated 30 Jul 2019). https://advances.realtimerendering.com/s2019/index.htm
- [S18] Unity, HDRP 17 "Volumetric Clouds Volume Override reference". https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@17.0/manual/volumetric-clouds-volume-override-reference.html
- [S19] Unity, HDRP 17 "Create realistic clouds (volumetric clouds)". https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@17.0/manual/create-realistic-clouds-volumetric-clouds.html
- [S20] Godot 4.x "Sky shaders" reference. https://docs.godotengine.org/en/4.2/tutorials/shaders/shader_reference/sky_shader.html ; "Custom sky shaders in Godot 4.0": https://godotengine.org/article/custom-sky-shaders-godot-4-0/
- [S21] Ghaemian, "Creating Volumetric Clouds for Mobile in Unreal Engine 5", 80.lv (30 Mar 2023). https://80.lv/articles/creating-volumetric-clouds-for-mobile-in-unreal-engine-5
- [S22] Sell, "Volumetric Cloud Upsampling", Vertex Fragment (undated). https://www.vertexfragment.com/ramblings/volumetric-cloud-upsampling/
- [S23] Sundog Software press release on volumetric clouds in VR (vendor, secondary). https://www.army-technology.com/contractors/training/sundog-software/pressreleases/sundog-software-reveals-new-improvements/
- [S24] Babylon.js forum, "Performant volumetric clouds" (Quest 2 advice). https://forum.babylonjs.com/t/performant-volumetric-clouds/22845
- [S25] Qualcomm, Snapdragon Game Developer Guide, Adreno FAQ (texture vs ALU ratio on A5x; filtered output is 16-bit). https://developer.qualcomm.com/sites/default/files/docs/adreno-gpu/snapdragon-game-toolkit/gdg/gpu/faq.html
- [S26] Qualcomm, "Tiling Architecture" and "Avoid GMEM Loads". https://developer.qualcomm.com/sites/default/files/docs/adreno-gpu/snapdragon-game-toolkit/gdg/gpu/best_practices_tiling.html ; https://docs.qualcomm.com/bundle/publicresource/80-71528-1/topics/avoid_gmem_loads.md
- [S27] MSFS community observations (not primary): https://forums.flightsimulator.com/t/best-method-to-reduce-pixelated-clouds-in-msfs-2024/753807 ; https://flyawaysimulation.com/ask/answers/msfs-2020-vs-2024-clouds/
- [S28] Star Citizen volumetric clouds (secondary): https://wccftech.com/star-citizen-gets-gorgeous-volumetric-clouds-but-squadron-42-is-still-mia/amp/
- [S29] Wronski, "Assassin's Creed IV: Black Flag – Road to Next-Gen Graphics", GDC 2014. https://gdcvault.com/play/1020397/Assassin-s-Creed-IV-Black
- [S30] miHoYo, Unity Dojo talk on Genshin Impact rendering (2021). https://docswell.com/s/UnityJapan/KWRPQ5-210617-unity-dojo20211mihoyozhenzhongyi
- [S31] Jendersie, d'Eon, "An Approximate Mie Scattering Function for Fog and Cloud Rendering", SIGGRAPH 2023 Talks (Aug 2023), DOI 10.1145/3587421.3595409. https://research.nvidia.com/labs/rtr/approximate-mie/ ; paper: https://research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf
- [S32] Wolfe, Morrical, Akenine-Möller, Ramamoorthi, "Spatiotemporal Blue Noise Masks", EGSR 2022. https://research.nvidia.com/publication/2022-07_spatiotemporal-blue-noise-masks

**Evidence caveats.**
- Nubis 2022 and 2023 internals beyond the abstracts, and the HG+Draine fit
  constants, are from recollection. The PDFs could not be fetched (size, and
  a fetch error). Verify before implementing.
- MSFS, Star Citizen, Genshin and Sky have no primary technical source.
- All Quest 3 costs are fetch-count estimates, not measurements.
