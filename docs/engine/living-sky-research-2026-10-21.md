# A sky that feels alive: how games move, grow and light their clouds, why ours still looks still and striped, and what to change on Quest 3 (2026-10-21)

Scope: research only. No code was changed. This report builds on three
earlier ones and does not repeat them:

- `sky-research-games-2026-10-20.md` (the "games doc", items R0-R17);
- `cloud-rework-research-2026-10-20.md` (the "rework doc": population,
  envelopes, server weather);
- `soulframe-sky-and-sun-glare-2026-10-20.md`.

This one answers three questions:

1. Why do the clouds still show little visible movement after the 10-21
   advection change?
2. Where do the horizontal striations come from?
3. Which of the motion techniques shipped games use fit a 36 Hz SpaceWarp
   app on an Adreno 740, preferably for less GPU than today?

Code read for the diagnosis:

- `space_soup/src/renderer/cloud_sky.rs`: `fs_map`, `density`, `noise_at`,
  `light_tau`, `encode_at`, `set_population`, `MAP_*`;
- `space_soup/src/renderer/sky.rs`: `sky_clouds`, `step_clouds`, `tick`;
- `space_soup/src/renderer/sky_motion.rs`;
- `space_soup_sky/src/cloud_layers.rs`: `layers_at`, `LAYER_WINDS`,
  `VIEW_STEPS*`;
- `space_soup_sky/src/time_of_day.rs`: `cloud_seconds`;
- `games/artytest1/scenes/test_room.json`: a 20-minute day, cumulus wind
  12 m/s, `weather_minutes` 15, `storms` 0.35.

Evidence tags:

- **[P]** primary source read (slides, notes, docs or paper text);
- **[A]** primary abstract or session page only;
- **[S]** secondary or forum;
- **[R]** recollection, not re-verified today;
- **[C]** read from our own code today;
- **[D]** derived here by arithmetic from the numbers given.

---

## 0. Bottom line

**Why the sky looks still.** There are four causes. All are measurable and
all are fixable without per-pixel cost.

1. **At real-time wind, clouds near the horizon move below or near the
   threshold of human motion detection** [D].
   - Take our cumulus: base 1.8 km, 12 m/s.
     - Overhead it moves 0.38°/s, about 9.5 Quest 3 pixels a second.
     - At 10° elevation it moves 0.066°/s across the view and 0.012°/s
       toward or away from the viewer.
   - Slow-motion detection against a fixed reference is reported at about
     1-2 arcmin/s, or 0.017-0.033°/s [S, W26].
   - A VR player looks mostly within ±25° of the horizon. In that band,
     real-speed clouds are physically correct and perceptually frozen.
2. **Everything in a layer moves rigidly** [C].
   - The display advects each map by its layer's wind, and the noise
     offsets and the envelope move in one wind frame.
   - Shape change appears only in the cross-fade between maps 2.7 s apart.
   - Rigid translation of a whole field reads as a scrolling dome.
     Individual clouds that drift, grow and fray at different rates read
     as weather.
3. **The "boil" is too slow, and it slides rather than churns** [C].
   - Cumulus noise scrolls vertically at 0.25 m/s × `evolution`.
   - Real fair-weather updrafts are 0.5-1.5 m/s under the cloud, and in
     cloud they reach about 5 m/s [P/A, W28].
   - A uniform vertical scroll through a fixed height profile reads as
     texture sliding upward, not as turrets rising and edges evaporating.
4. **The sun runs 72× faster than the clouds** [C].
   - `test_room` has a 20-minute day, and `cloud_seconds` is real time.
   - Light sweeps over clouds that barely move, which is the opposite of
     what a real sky or a time-lapse shows.
   - The recorded rule (memory `cloud-rework-2026-10-20`) is "the cloud
     clock should be real time even when the day is accelerated", and the
     user expects a slower day. This report therefore treats a cloud time
     scale as an **authoring lever with a default of 1**, not as the fix
     (§6, item 6). The real-time-correct fixes come first.

**Why the striations.** Three mechanisms in `fs_map` each produce lines
parallel to the horizon [C]. They can be told apart by three cheap tests
(§4.3):

- **(a) Noise-lattice strata.** In the view march `noise_at(.., eased=true)`
  interpolates the 64³ noise on a smoothstep. Its gradient is zero at every
  texel face, so density terraces on planes 62.5 m apart in altitude for
  cumulus: about 23 strata through a cumulus layer. The vertical boil then
  drags those strata upward through every cloud.
- **(b) The sample lattice.** Samples are taken at `t0 + (i+0.5)·len/n` with
  `n = ceil(len/stride)`, and there is no jitter.
  - Every ray with the same `n` samples the same altitude fractions of the
    layer, which gives stacked plates.
  - Each time `n` increments from one row to the next, all the sample
    planes jump. Below about 20° that happens every 2-6 map rows (about 2 at 10°).
  - The step planes (about 48 m apart at the zenith) also beat against the
    noise planes (62.5 m) with a period of about 200 m.
- **(c) Slice seams.**
  - One map is built over 96 frames (2.7 s).
  - The envelope is re-splatted once a second of cloud time *during* the
    build (`set_population` → `envelope_pending`).
  - The light and ambient come from a time-of-day snapshot that refreshes
    every 0.02 game-hours, about once a real second at a 20-minute day.
  - So one map holds 2-3 bands of rows built against different inputs.

**What to do (ranked in §6).**

- **First, removing the stripes costs about nothing and partly pays for
  itself:**
  1. Rotate the noise domain off the horizontal axes, or ease only in x/z.
  2. Use a fixed-stride lattice with a fractional last step.
  3. Freeze each map's inputs at its moment.
  4. Accumulate jittered builds in map space (a history read in the march,
     not at display), then cut the step counts. Net about −0.2 to −0.35 ms,
     estimated.
- **Then make the sky alive at real speed:**
  - churn at real convective rates (curl and two-phase flow in the march);
  - per-cloud velocity spread and rising tops from the population (CPU);
  - cloud shadows over kilometres of ground (the most visible real-time
    cue);
  - a fast low scud layer under storms;
  - flow-mapped cirrus;
  - server-driven fronts.
- None of this adds a fetch to the sky display shader, which is at its size
  cap (+385 of +400, memory `clouds-drift-on-their-winds`).

---

## 1. How sky motion is perceived (the numbers behind "alive")

### 1.1 Angular speed of a drifting layer [D]

A flat layer at height h above the eye, wind speed v, seen at elevation e,
lies at a slant distance of `h / sin e`. Its angular speed is:

- across the line of sight: `ω⊥ = v·sin e / h`;
- along the line of sight (toward or away): `ω∥ = v·sin² e / h`.

Earth curvature makes the low-elevation values smaller still.

Our `test_room` layers come from `layers_at` with `LAYER_WINDS` and a 12 m/s
cumulus wind:

| Layer (base, wind) | Zenith °/s | 30° ⊥ / ∥ | 10° ⊥ / ∥ | 5° ⊥ / ∥ |
|---|---|---|---|---|
| Stratocumulus 1.08 km, 9.6 m/s | 0.51 | 0.25 / 0.13 | 0.089 / 0.015 | 0.044 / 0.004 |
| Cumulus 1.8 km, 12 m/s | 0.38 | 0.19 / 0.096 | 0.066 / 0.012 | 0.033 / 0.003 |
| Altocumulus 4.0 km, 20.4 m/s | 0.29 | 0.15 / 0.073 | 0.051 / 0.009 | 0.025 / 0.002 |
| Cirrus 8.5 km, 38.4 m/s | 0.26 | 0.13 / 0.065 | 0.045 / 0.008 | 0.023 / 0.002 |
| *Scud 0.5 km, 15 m/s (proposed, storms)* | *1.72* | *0.86 / 0.43* | *0.30 / 0.052* | *0.15 / 0.013* |

To convert to pixels:

- The Quest 3 has 25 pixels per degree (Meta spec [P, W25]).
- The sky map is 0.176°/texel in azimuth (`MAP_WIDTH` 2048), about 4.4
  headset pixels.
- So cumulus at 10° moves 1.6 px/s across the view and 0.3 px/s along it.

Motion detection:

- Detection of slow motion against a stationary reference is reported at
  about **1-2 arcmin/s (0.017-0.033°/s)** in central vision.
- A displacement of about 1.5 arcmin is needed for brief presentations
  (Johnson and Leibowitz, as cited by Legge 1981) [S, W26].
- Every ∥ value at 10° and below in the table sits under that threshold.
  The ⊥ values there are only 2-4× above it.

**What this means:**

- A real sky at real time is mostly still where a standing VR player looks.
  It reads as alive through other cues:
  - fast low cloud;
  - motion overhead;
  - shape change at the edges;
  - shadows racing over the ground;
  - light changing on the clouds.
- Real high cloud moves *slower* across the sky than low cloud despite
  faster winds. Our layer ordering already gets that right, which gives
  free inter-layer parallax.

### 1.2 Shape timescales [P/S]

- **Lifetimes.** Satellite tracking of millions of shallow marine cumulus
  finds a double power law: most clouds live **tens of minutes**, and the
  larger ones live longer [P/A, W27]. Average cumulus lifetimes are under
  an hour (the rework doc §2.2 has the details).
- **Turbulent churn.**
  - Updrafts under fair-weather cumulus average 0.4-0.5 m/s for 100-200 m
    thermals and about 1.5 m/s for large ones [A, W28].
  - A W-band radar study of marine fair-weather cumuli measured about
    5.5 m/s in-cloud updrafts in towers only about 400 m across [A, W28].
  - A 300 m eddy rising at 1-3 m/s therefore turns over in about 100-300 s.
  - At 5 km range, 2 m/s of turret rise is 0.023°/s, right at threshold.
- **Edges evaporate** within tens of seconds to minutes. That is the
  fastest shape cue at real time, and the one most visible at the horizon
  because it does not depend on the wind's direction relative to the view.

### 1.3 Time compression in games [P/A/S]

- **Horizon Zero Dawn**, 2015 talk: a storm demo is "sped up, for effect"
  [P, W1].
- **Ghost of Yōtei** moves its cached clouds faster on purpose, to suggest
  "stormy, unsettled weather". Depth stats per texel make that free of
  artefacts [P, W13].
- **trueSKY** clouds "will not change over time, only move" unless
  keyframes are interpolated over game time. Evolution runs on the game
  clock, so it is compressed with the day [P, W16].
- **No primary source was found** that gives a shipped title's cloud
  time-scale factor.
- **Rule of thumb derived here** [D]:
  - with a day compressed by D, any cloud factor k between 1 and D is a
    compromise;
  - k = √D keeps the clouds and the light equally "wrong" (√72 ≈ 8.5 for
    the 20-minute day);
  - with a 2-4 h day (D = 6-12), k ≈ 2-3.5.

### 1.4 What makes a sky feel dead (checklist used in §6)

1. **Rigid translation.** The whole field moves as one: a scrolling dome.
2. **No change at the silhouettes** between glances.
3. **No ground shadows** sweeping across the land.
4. **No inter-layer parallax**, or layers all moving the same way.
5. **Periodic artefacts that move with the clouds**: strata, stripes,
   tiling. They make the eye read the clouds as a texture.
6. **Pulsing or stepping** at rebuilds or light refreshes.
7. **Light that does not respond** to the cloud: no bright gaps or
   silver-lined edges passing the sun.

Ours today: 1, 2 and 5 fully; 3 beyond about 500 m (64² shadow over 1 km);
6 only faintly since the advection change. Item 4 is present (layers turn
12°, 0°, −25° and 38°).

---

## 2. What shipped games do: animation, temporal reuse, and cost

The games doc §1 covers the lighting and marching details. Here only
motion, evolution, temporal reuse and cost.

### 2.1 Guerrilla: Nubis (HZD 2015/2017, HFW 2022, Burning Shores 2023)

**HZD 2015 slides** [P, W1]:

- **Modelling.**
  - A 128³ base texture: Perlin-Worley plus 3 Worley octaves.
  - A 32³ detail Worley texture.
  - A 128² **curl noise** that is "non divergent and is used to fake fluid
    motion". They "distort our cloud shapes and add a sense of turbulence",
    applied to the detail erosion.
  - Height gradients per cloud type, and a weather map with coverage (R),
    precipitation (G) and type (B).
- **Animation.**
  - "it is all animated in a given wind direction".
  - Coverage and type are "a FUNCTION of our weather system". The weather
    system "modulates these channels with a simulation that progresses
    during gameplay", with bias controls for art direction.
  - High cirrus and alto are **2D tiling textures "scrolling at different
    speeds and directions"** above the volumetric layer.
  - Recollection, from the GPU Pro 7 chapter's code [R]: cloud tops are
    pushed downwind by height fraction, and the sample point is offset by
    `(wind + small up component) · time · speed`, so the shapes rise slowly
    as they drift. That is the same mechanism as our boil.
- **Temporal reuse.**
  - The naive version "costs around 20 milliseconds".
  - The shipped one updates "1 out of 16 pixels for each 4x4 pixel block"
    a frame and reprojects the previous frame.
  - Where it cannot reproject, "like the edge of the screen", it uses a
    low-resolution buffer.
  - Rendering at reduced resolution and upscaling made it "10x faster or
    more". The result is about **2 ms and 20 MB on PS4**.
  - Hillaire's summary: "Update 1 pixel per 4x4 tile: 16 frame latency but
    looks full-res" [P, W6].
- **Banding.** The light samples use a 6-tap cone toward the sun, which
  "smooth's the banding we would normally get with 6 simples". A long tap
  catches distant shadowing [P, W1].

**Nubis 2017** ("Authoring Real-Time Volumetric Cloudscapes"):

- It replaced the 2015 procedural weather simulation with regionally
  authored fields plus "animation and transitions" [A, W2].
- Further optimisation took it below 2 ms [A].

**Nubis Evolved (2022) and the GDC 2022 superstorms** [A, W3, W5]:

- 2D fields are expanded vertically at render time.
- The superstorms were built "without expensive simulations".
- The GDC abstract adds "a solution to render faster moving clouds with
  temporal upscaling". The SIGGRAPH abstract stresses 1080p "without
  temporal upscaling" for fly-through clouds, and "mitigating temporal
  artefacts in fast-moving clouds".
- The slide PDFs (above 10 MB) could not be read today, so the method
  behind that line is unverified.

**Nubis³ (2023)** [A, W4]:

- Voxel clouds modelled from fluid simulations, with compressed SDFs for
  skipping and accelerated light sampling.
- They still act as "traditional volumetric skyboxes" with time of day.
- About 0.2-5 ms by view distance (games doc).
- The voxel content is authored offline, not simulated at runtime.

**For us:**

- The 2015 recipe for motion is wind advection, curl distortion of the
  erosion only, a weather map driven slowly, and 2D high layers on their own
  winds. We have all of it except the curl.
- Their temporal scheme is screen-space. Ours is direction-space, which
  needs no reprojection on head turns and serves both eyes: better for VR.

### 2.2 Sucker Punch: Ghost of Tsushima (2020) and Ghost of Yōtei (2025). The design we follow.

**Tsushima** (Patry, SIGGRAPH Advances 2021) [P, W12]:

- Clouds go into a **768×768 paraboloid**, R11G11B10F with Mie, Rayleigh
  and transmittance in separate channels.
- There are **three textures**: two for **scrolling and lerping**, and one
  being rendered, **time-sliced over 60 frames**.
- Density aliasing is fixed by fading density where the derivatives of
  cloud position against texture coordinate are high.

**Yōtei** (PlayStation Blog deep dive) [P, W13]:

- Each texel stores the average visible depth and squared depth, "to use
  parallax mapping techniques while scrolling each cloud frame".
- It blends **three cloud frames instead of two** for "even smoother
  motion".
- Clouds are moved faster for storms.
- A light-space cloud shadow map feeds god rays far from the camera.

**For us:**

- Our 10-21 change (advection by the height in `alts`) is the Yōtei step.
- Their third blended frame is what smooths *shape change*. On our side the
  same effect can come from a history in the march (§6, item 4) instead of a
  third display read, because our display shader has no room.

### 2.3 Rockstar: Red Dead Redemption 2 (Bauer, SIGGRAPH Advances 2019)

- The talk covers cloud and fog rendering with "a voxelization and
  raymarching approach", shared with reflections and a sky irradiance probe
  grid [A, W11]. The 232 MB PPTX was not read.
- Hillaire's 2020 notes summarise the temporal choice as "Trace at quarter
  resolution with TAA upsampling: instant result but looks blurrier",
  against Nubis's 1/16 update [P, W6].
- No cloud ms figure or animation details were found in a readable primary
  source.

### 2.4 Frostbite (Hillaire 2016) and Unreal Engine 5

**Frostbite 2016**:

- Sky, atmosphere and clouds are treated as one system for dynamic time of
  day and weather [A, W8].
- The course PDF (above 10 MB) was not readable today. Its temporal scheme
  is not verified here.

**Unreal Engine** (Hillaire 2020 slides [P, W6]; UE docs [P, W9]):

- **Sky.** The sky is a low-resolution **Sky-View LUT** in lat-long, with a
  **non-linear latitude mapping** that keeps the horizon's high frequencies
  "while reducing linear interpolation artifacts".
  - The slide's LUT build timings read "0.14 ms, 0.31 ms, 1.03 ms" across
    its PC (GTX 1080) and mobile configurations.
  - "Only cloud rendering is not reasonably achievable on mobile. For now."
- **Clouds.**
  - A volume material graph traced at **quarter resolution, reconstructed
    at half, and upsampled to full** (mode 0).
  - Cloud shadows can be stored "as part of the Sky-View and aerial
    perspective LUTs" when per-pixel tracing is too expensive.
  - The docs state Volumetric Clouds are **not supported on Mobile**; Sky
    Atmosphere is.
  - The sky light's real-time capture is time-sliced over 9 frames [P, W9].
- **Animation.**
  - The docs name no wind parameter. Motion comes from the cloud material,
    i.e. time-offset noise.
  - Ryan Brucks's 4.26 demo painted **density and flow maps** that "steer
    both the cloud's volumetric motion and its density" [S, W10].

### 2.5 Microsoft Flight Simulator 2020/2024 (Asobo)

- **No Asobo technical talk on clouds was found.**
- Weather comes from Meteoblue: about 250 M cells and 60 vertical layers
  [S, W22].
- Forum analysis concludes the shapes are procedural noise "translated
  vertically within the volume, so the clouds do not convect" [S, W22].
  That is our boil, and the community's complaint about it is the one we
  hear.
- Forum reports also describe visible "parallel plates" when flying through
  clouds. That is the slice artefact of §4 [S].

### 2.6 Sea of Thieves (Rare; Ang et al., SIGGRAPH 2018 talk) [P, W14]

- No ray-march. Clouds are **opaque meshes** with per-vertex
  subsurface-like lighting, rendered off-screen, then:
  - downsampled to quarter resolution;
  - Gaussian-blurred, with depth box-blurred;
  - composited with a distortion map for fluffiness and a low/high noise
    blend by distance.
- **Motion.** The meshes are offset by the wind and wrapped around the
  player. The **offset is synchronised across all clients**.
- **Evolution.** Radial **high- and low-pressure zones** push clouds out of,
  or into, regions.
- **For us:** server-authoritative wind phase and moving pressure systems
  are exactly what `space_soup_server` can stream with `Option` fields.

### 2.7 The Witcher 3 (REDengine 3), reverse-engineered [S, W15]

- Cirrus is a stretched dome mesh with **two tiled textures**:
  - normal map plus shape;
  - noise that perturbs the shape.
- UVs are `uv·scale + time·speed`, with different speeds per texture, and
  "no wind model".
- Lighting is N·L from the normal map plus a sun-side term, and the layer
  is drawn twice.
- The stars drift with the moon's frame (part 12).
- This is the classic 2.5D "two scrolling layers" sky. It is cheap and
  readable, and it looks painted up close.

### 2.8 Others

- **Death Stranding / Decima.** The Decima Lighting and AA talk covers fog
  and atmosphere, not clouds. A frame breakdown shows separate cloud and
  volumetric-cloud compute passes [A/S, W24]. The cloud method is
  presumably Nubis; not verified.
- **Star Citizen.**
  - Volumetric clouds shipped in 3.14 (Crusader).
  - CitizenCon 2953 (2024) showed planet-scale volumetric clouds with
    shadows into the atmosphere.
  - A September 2026 monthly report (PCGH) describes a lower-resolution
    cloud buffer whose temporal upsampling "now also takes scene visibility
    into account" [S, W23].
  - No evolution method was found.
- **trueSKY (Simul).** Per-layer cloud **keyframes interpolated by game
  time**: cloudiness, base, height, density, "churn" and others. With one
  keyframe the clouds "only move (if there is wind)" [P, W16].
- **Unity HDRP Cloud Layer**, the cheapest documented dynamic sky
  [P, W17]:
  - a lat-long RGBA cloud map (cumulus, stratus, cirrus and wispy in four
    channels);
  - wind modes None, Horizontal, or **Flowmap** (RG displacement, 0.5 =
    none);
  - **2D ray-marched self-shadow baked to an intermediate texture and
    recomputed only when a parameter changes**;
  - shadows as a light cookie.
- **Assassin's Creed Unity/Odyssey, Soulframe, Horizon Call of the Mountain
  (PSVR2), Red Matter 2, Asgard's Wrath 2, Batman: Arkham Shadow, Behemoth.**
  No public material on their sky or cloud technique was found.
  - Camouflaj's GDC talk "Design and Rendering Strategies for Blockbuster
    VR Games" exists, but it is members-only [A, W30].
  - Red Matter 2 is reported **not** to use AppSW [S, W31].
  - The earlier docs' finding stands: no published Quest title ships
    ray-marched volumetric clouds.

---

## 3. Cheap motion techniques, and how each fits our map

Costs are for our structure:

- the **march** is paid per map texel, 16 k texels a frame;
- the **display** is paid per sky pixel, about 2.5-4 M a frame, both eyes.

Rule: add work in the march or on the CPU, never at display.

| Technique | Who | How it moves clouds | Where it would run here | Cost here | Verdict |
|---|---|---|---|---|---|
| UV scroll of 2+ layers at different speeds and directions | Witcher 3, HZD high clouds | Rigid translation, layer parallax | Already: advection per height | 0 | Have it; rigid on its own |
| Flow map (RG displacement), two-phase blend to hide resets | Valve (Vlachos 2010), Grimes 2011 [S, W21]; HDRP; UE 4.26 | Local, non-rigid drift: shear, swirl, convergence | **March**: distort the cirrus tau lookup and the detail-noise domain | +1 small 2D fetch per sample on the affected layer | **Yes** (cirrus, edges) |
| Curl-noise domain distortion | HZD 2015 (detail erosion) | Swirling, fluid-looking edges; animated by offsetting the curl texture over time | **March**, detail erosion only (view samples, not light taps) | +1 fetch per view density sample (~+5-10% march) | **Yes** |
| Noise-domain vertical scroll ("boil") | HZD [R], MSFS [S], **ours** | Shapes slide upward | — | 0 | Replace by curl, two-phase flow and population rise |
| Keyframed or weather-driven coverage and type | Nubis, trueSKY, HDRP | Regions fill and clear | Population and server weather | CPU | Have it; speed it up for storms |
| Object or particle population | Sea of Thieves, FS2004, **ours** (10-21) | Real identities: born, grow, drift, die | CPU splats | ~0 GPU | Have it; add per-object velocity and rise |
| Cached sky texture, time-sliced and cross-faded | Tsushima, **ours** | Breathing between frames | — | — | Have it |
| Depth per texel, parallax scroll | Yōtei, **ours** (`alts`) | Correct drift between rebuilds | — | — | Have it |
| Third blended frame | Yōtei | Smoother shape change | Would be +1 map read at display | +4 taps a pixel; the display shader is at its cap | **No**; do a history in the march instead |
| Temporal accumulation (1/16 update, TAA) | Nubis, RDR2, UE | Hides undersampling | **March**: history read per map texel | +1-2 fetches per texel (16 k) | **Yes**, in map space |
| Time-lapse flipbook sky with motion-vector frame blending | VFX flipbooks (optical-flow inter-frame blending [S]) | Pre-rendered evolution | Server or offline-baked panoramas | Memory; display +2 taps | No: fixed, and it fights dynamic weather |
| Server-rendered far band | "Clouds in the Cloud" (HPG 2023, rework doc §4.6) | Far horizon streamed | Server | Bandwidth | Later |

---

## 4. Horizontal striations: causes and how shipped games remove them

### 4.1 General causes [P/S]

- **Fixed sample positions.** Every ray samples the same depths, so
  neighbouring pixels err in lockstep and give "slices" or false contours.
  The fix is a per-pixel start jitter of a fraction of a step, which turns
  slices into noise. Blue noise is preferred because its residue is less
  visible and easier to filter. Temporal accumulation then removes the
  noise [S, W20, W34].
  - Toft et al. (Studio Gobo, 2016): 128 steps took 297.7 ms; 8 steps
    2.3 ms; 8 jittered steps 7.5 ms (+5.2 ms, attributed to texture cache
    misses); 8 jittered steps with TAA also 7.5 ms. The claimed result is
    "visually similar results with 1/16 the number of steps" [A, W20].
- **Lattice alignment.** A tiled noise sampled with any interpolation whose
  derivative is discontinuous or zero at texel faces shows the lattice. The
  lattice's horizontal planes become strata on cloud sides.
  - The usual remedies are to rotate or skew the domain, use smoother
    (B-spline or tricubic) reconstruction, or add octaves at non-integer
    lacunarity [R].
- **Step-count switching.** When the integer step count changes between
  neighbours, every sample position jumps, which leaves a seam. Nubis's
  answer is cheap large steps until an isosurface, then a step back and
  full-detail steps [P, W1], or adaptive steps from SDFs (Nubis³).
- **Light-sample banding.** A few fixed light taps give contour lines in the
  shading. HZD's cone of six taps "smooth's the banding" [P, W1].
- **Time-sliced or interleaved updates.** Parts of one image built at
  different moments, or from different inputs, show seams along the slice
  pattern.
  - Toft notes the HZD 4×4 update pattern "may introduce visual artifacts"
    [A, W20].
  - Hillaire notes Nubis's "16 frame latency" [P, W6].
- **Packed elevation.** A non-linear latitude mapping is used to *reduce*
  horizon interpolation artefacts (Hillaire's Sky-View LUT [P, W6]).
  - Ours (`el = 90°·v^1.5`) is in that spirit. At f32 it is not a precision
    source [C].
  - The R16F `alts` quantises 8.5 km to 8 m steps, which is negligible for
    advection [D].

### 4.2 Our three mechanisms [C]

**(a) Strata from the eased noise lattice.** This is most likely the
dominant cause.

- `noise_at(s, eased=true)` is used for every view-march density (base and
  erosion). It maps each texel fraction f to `f²(3−2f)` before the hardware
  trilinear tap, so the density's gradient is zero on every texel face.
- In y the faces are horizontal planes:
  - cumulus tile 4 km / 64 texels = **62.5 m**, so about 23 planes through
    its 1.44 km;
  - stratocumulus about 84 m;
  - altocumulus about 20 m.
- Thresholded density therefore terraces on those planes. At 5-10 km
  range, 62.5 m is 0.36-0.7°, or **9-18 headset pixels between lines**:
  "horizontal line striations".
- The boil (`offset[1]`) moves the noise vertically, so the strata drift up
  through the clouds. Each rebuild redraws them at new heights, and the
  cross-fade makes them shimmer.
- The easing was added to stop altocumulus puffs coming out square
  (2026-10-20). That problem lives in horizontal sections, so it does not
  need the y easing.

**(b) Fixed fractions and integer step counts.**

- `n = clamp(ceil(len/stride), base, 96)` and `t = t0 + (i+0.5)·len/n`,
  with no jitter (removed 10-20 because, with no accumulation, a jitter
  showed as hatching).
- Above about 20° the cumulus ray takes n = 30, so every such ray samples
  the same 30 altitude planes: stacked plates.
  - 1.44 km / 30 = 48 m between sample planes, against 62.5 m between noise
    planes, beats with a period of about 210 m.
- Below about 20°, n climbs from 30 to 96 over about 15° of elevation, one
  increment per 125 m of extra path. With 11.8 map rows per degree at 10°
  (MAP_PACK 1.5), **the whole set of sample planes jumps every 2-6 rows** (about 2 at 10°, 6 at 20°):
  a fine horizontal hatching toward the horizon.
- The light taps sit at fixed fractions (0.05, 0.15, 0.35, 0.8 of the layer
  thickness), so their banding is aligned too.

**(c) Slice seams inside one map.**

- `encode_at` builds 8 rows a frame. The map's moment and eye are frozen
  (good), but two inputs are not:
  1. **Envelope generations.** `set_population` raises `envelope_pending`
     whenever the population's generation changes (once a second of cloud
     time). The next slice re-splats, so the rows of one 2.7 s map see 2-3
     different envelopes.
  2. **Light and ambient.** `CloudSkyUniform::new(.., f.light_irradiance,
     f.ambient_sky, ..)` comes from `c.frame`. That frame changes when a
     time-of-day snapshot arrives, every `REFRESH_HOURS` = 0.02 h, which is
     **about once a real second at a 20-minute day**. Only the light's
     direction glides (`light_now`).
- The result is 2-3 bands of about 290 rows each, with faint brightness or
  shape steps between them. The boundaries move every map, and the
  cross-fade shows two maps' boundaries at once.
- This is weaker than (a) and (b) on a steady sky, but strongest at dawn
  and dusk.

### 4.3 How to tell them apart (all levers or one-line debug changes)

1. **Freeze time** (`Phase::TimeFrozen`, `time_of_day_minutes = 0`):
   - if the stripes vanish, it is (c);
   - if they stay and stop moving, it is (a) or (b).
2. **Draw the map whole every frame** (`whole = true`): removes (c) only.
3. **Set the boil to 0 and turn off easing in y only**: if the lines that
   drifted upward vanish, it is (a).
4. **Colour `fs_map` by `n % 2`**, or raise `VIEW_STEPS_MOST` to 192 and
   `VIEW_STEP_TEXELS` to 1: if the hatching follows the parity bands, it
   is (b).

### 4.4 What fixes them for us (all inside `fs_map`; §6, items 1-4)

- **(a)** Rotate the noise domain:
  - `s' = R·s` with a fixed rotation, e.g. 37° about x then 23° about z;
  - ALU only, and the CPU twin applies the same matrix;
  - or ease in x and z only.
  - Rotation tilts every lattice plane off horizontal, so no stratum is
    parallel to the horizon. What aliasing remains is spread across
    orientations, where the eye accepts it as texture [R].
- **(b)** Make the sample lattice continuous:
  - a fixed stride per layer (metres);
  - `t_i = t0 + (i + o)·stride` with a fractional last step `dt_last`;
  - `n` then only adds a sliver at the far end instead of moving every
    sample.
  - Give `o` a per-map offset (golden-ratio sequence) plus a 4×4 interleave
    by texel, and offset the light taps by the same `o`, ±25%.
- **(c)** Freeze the inputs per map:
  - take light, ambient and envelope from the map's moment `at`;
  - re-splat the envelope only when `starts_map` is true;
  - interpolate the snapshot's irradiance and ambient to `at` (the
    snapshot already carries both ends of its blend).
- **Accumulate** (removes what jitter leaves):
  - after marching texel x of map k, read map k−1 at the advected uv
    (the same `advected_uv` math, one bilinear tap);
  - store `mix(hist, new, α)` with α ≈ 0.35-0.5;
  - clamp `hist` to new ± a tolerance against ghosting when a cloud is born
    or dies.
  - Over 3-4 maps the jittered lattices average into a continuous integral.
    That is Nubis's 1/16 amortisation, done in direction space, where head
    motion costs no reprojection.

---

## 5. Motion vectors for skies under reprojection and SpaceWarp

### 5.1 What the platform says [P, W18, W19]

- AppSW defines motion as the **NDC position difference between the current
  and previous frame**, in 3D with depth motion. The app renders at half
  rate (36 Hz) and the compositor synthesises 72 Hz.
- Materials that write no motion vectors "will appear to stutter, as they
  won't move in AppSW-generated frames".
- A pixel can carry **one** motion. Transparency is ambiguous, and it is
  "usually" not recommended in the MV pass. Distant transparency is
  acceptable because "the actual projected motion is very small from frame
  to frame".
- **"Simple, clean backgrounds"** show warp distortion more than rich ones.
- The MV and depth targets are deliberately low resolution: the Quest 2
  default is 368×400 against a 1440×1584 eye.
- For complex vertex animation, feeding the current position as the
  previous one keeps head and object motion smooth while that animation
  judders a little.
- Neither page names the sky.

### 5.2 What games do

- TAA engines treat the sky as at infinity and give it camera-rotation-only
  motion, since cloud drift is sub-pixel per frame. Nubis reprojects its
  cloud buffer the same way and falls back to a low-resolution buffer at
  disocclusions [P, W1; general practice R].
- Tsushima and Yōtei avoid screen-space history for clouds entirely: the
  cached sky texture is the history [P, W12, W13].

### 5.3 Ours (`sky_motion.rs`) is right [C]

- A far-plane pass writes `motion_of(curr·dir, prev·back)` with w = 0, so
  head rotation is exact and translation gives no parallax.
- `back` adds the wind displacement over the cloud clock's step, weighted
  by cover.
- At k = 1 the wind term is at most 0.38°/s × (1/36) s ≈ 0.27 Quest pixels
  a frame at the zenith, and less elsewhere. It barely matters.
- It starts to matter with a cloud time scale or a scud layer (§6): 4× at
  the zenith is about 1.1 px a frame. Because `misc.y` is the cloud clock's
  dt, a time scale reaches the motion vectors with no further change.
- Keep the cover weighting. It is the one-motion-per-pixel compromise Meta
  describes for distant transparency.

---

## 6. Ranked changes for our engine

Baseline cost [D]: the 10-21 A/B saved 0.70-0.81 ms of GPU when the map
went from 48 to 96 slices, along with other changes in the same commit. So
today's march is roughly **0.6-0.8 ms a frame** (inferred, not timed per
pass). Pass timers on `cloud_sky_map`, `cloud_sky_shadow` and the clouded
sky display are still the first thing to add (games doc §3).

Ranking: look gained per ms, striations first, because they also make
motion read as texture.

### 1. Kill the lattice strata: rotate the noise domain, or ease only in x/z
- **Change:** in `density` (WGSL and the CPU twin in `cloud_layers.rs`),
  sample `noise_at(R·s, ..)` with a fixed rotation matrix R (ALU only). Or
  keep the eased reconstruction in x and z and use plain linear in y.
- **Look:** removes the 62.5 m horizontal strata and their upward crawl
  (§4.2a). Probably the largest single cut in stripes.
- **Cost:** about +0.005 ms (9 FMA per sample × ~0.5 M samples).
- **Risks:**
  - shapes change slightly, so re-tune `DETAIL_EROSION` by eye;
  - the twin tests (`the_gpu_clouds_match_their_twin`, envelopes) need the
    same R on the CPU;
  - the noise's tiling period becomes oblique (fine, and less visible).

### 2. A continuous, jittered step lattice for view and light
- **Change:**
  - fixed stride per layer;
  - `t_i = t0 + (i+o)·stride` with a fractional last step;
  - `o` = golden-ratio offset per map plus a 4×4 per-texel interleave;
  - the light-tap radii scaled by `(1 + 0.5·(o − 0.5))`.
  - Uniform per map or per 4×4 tile, not white per texel, to avoid the
    cache hit Toft measured (+5.2 ms on desktop) [A, W20].
- **Look:** removes the step-count hatching near the horizon and the plate
  beat overhead (§4.2b).
- **Cost:** ~0. It also lets item 4 cut steps.
- **Risk:** without item 4 the jitter shows as a slowly shifting grain at
  each rebuild. Ship 2 and 4 together.

### 3. Freeze each map's inputs at its moment
- **Change:**
  - in `step_clouds` and `encode_at`, splat the envelope only when
    `starts_map` is true (the population may update more often; the map
    takes the newest at its start);
  - build `m` from the light, ambient and irradiance interpolated to the
    map's moment `at`, not from `c.frame` as of this frame.
- **Look:** removes the 2-3 brightness and shape bands per map (§4.2c),
  most visible at dawn and dusk and during storms.
- **Cost:** slightly negative: fewer re-splats.
- **Risk:** the ground shadow, which also reads the envelope, lags by up to
  one map (2.7 s, well under a cloud's life). Use the map-start envelope for
  both, so the shadow and the sky agree.

### 4. Accumulate in map space, then cut the steps
- **Change:**
  - `fs_map` reads the last whole map (and its `alts`) at the texel's
    advected uv;
  - it blends `mix(hist, new, α)` with α ≈ 0.4 and clamps `hist` against the
    new value;
  - then lower `VIEW_STEPS` from [10, 30, 8] to about [6, 16, 6] and
    `VIEW_STEPS_MOST` from 96 to 48-64.
- **Look:** jitter residue and remaining banding average away. Shape change
  between maps gets the "third frame" smoothness Yōtei gets, without a
  display read.
- **Cost:**
  - +2 fetches per map texel (16 k a frame), about +0.01-0.02 ms;
  - −30% to −45% of view samples;
  - net about **−0.2 to −0.35 ms** (estimate).
- **Risks:**
  - ghosting on births and deaths: clamp, and drop the history where the
    newest `alts` and the history's height differ by more than about
    300 m;
  - shape changes lag by about 2-3 maps (6-8 s), which is invisible at real
    lifetimes;
  - the history read must use the same `advected_uv` as the display, so the
    two stay consistent (an existing twin, `advected_uv`).

### 5. Churn at real convective rates (keeps the real-time clock)
- **Change:**
  - **(a) Curl distortion.** Add HZD's curl distortion: a 128² RG8 curl
    texture offsets the *detail* erosion domain by up to about half a
    detail cell. Animate it by sampling two time phases 0.5 apart and
    blending them on a triangle wave (Vlachos and Grimes two-phase flow), so
    it never slides. Period about 60-120 s, the eddy turnover from §1.2.
  - **(b) Retire the uniform boil**, or keep a tenth of it. Instead raise
    each cloud's top from the population (`life_shape`) at 1-3 m/s while it
    grows, and erode its edge faster while it decays (`env.z` phase already
    drives erosion).
- **Look:**
  - silhouettes visibly fray and reform at real time (edges are the
    fastest real-time cue, and the only one that reads at the horizon);
  - towers rise instead of the texture sliding.
- **Cost:**
  - +1 small cached 2D fetch per *view* density sample (not light taps),
    about +5-10% of the march, +0.03-0.07 ms;
  - (b) is CPU only.
- **Risks:**
  - the `fs_map` instruction size (a separate pipeline from the capped sky
    shader, but check with the PIPESTATS proxy);
  - the CPU twin needs the curl texture.

### 6. A cloud time-scale lever (authoring, default 1)
- **Change:**
  - `CloudParams::time_scale` (scene `time_of_day.clouds.time_scale`);
  - cloud seconds = real seconds × scale, applied to drift, population life
    and weather.
  - Default 1 honours the recorded real-time rule. Suggest trying 3-6 in
    `test_room` (20-minute day, √72 ≈ 8.5), or about √D for any compressed
    day.
- **Look:** the largest single lever for "alive". At 4×, cumulus at 10°
  moves 0.26°/s (6.6 px/s) across the view, about 8-16× the detection
  threshold.
- **Cost:** 0. The motion vectors already scale through `misc.y`.
- **Risks:**
  - it contradicts the stated preference, so the user decides;
  - at large factors births and deaths become visible as time-lapse;
  - keep the map period (2.7 s) short compared with a scaled life;
  - storms at 4× feel hurried unless `weather_minutes` scales separately.

### 7. Cloud shadows over kilometres of ground
- **Change:** add a far cascade to `cloud_sun_shadow`, e.g. 128² over 16 km
  centred on the player.
  - Build it like the sky map: time-sliced over 8-16 frames, advected by the
    cumulus wind in the reading shader (a UV offset, ALU only), one ray per
    texel, 1-2 light taps.
  - Read it with the same one fetch the ground already does: use a
    non-linear radial mapping of one texture (fine near, coarse far), so no
    second fetch is added per ground pixel.
- **Look:** **the most visible motion at real time.**
  - A 12 m/s shadow edge crossing a hill 300 m away moves about 2.3°/s, 35×
    faster than the clouds that cast it seen at 10°.
  - It also shows the population's gaps and banks on the land.
  - Matches UE/Frostbite Beer shadow maps and Yōtei's light-space map.
- **Cost:**
  - build about +0.02-0.05 ms (2 k rays a frame amortised);
  - display 0 extra fetches if folded into the existing lookup, otherwise
    about +0.05-0.1 ms of fill.
- **Risk:** it needs the R6/R14 work in the games doc. Price it with a pass
  timer.

### 8. Break the rigidity: per-cloud velocity and gusts (CPU, population)
- **Change:**
  - each cloud gets its own drift: layer wind × (1 ± 0.15) and ±10° of
    direction, from its seed;
  - a slow gust term in the layer wind (±20% over 2-5 min);
  - the envelope splat uses each cloud's own position.
- **Look:**
  - neighbouring clouds slowly overtake, converge and open gaps, the
    difference between weather and a scrolling dome;
  - the display still advects by the layer wind, and the per-cloud
    deviation is caught up at each rebuild (≤ 2 m/s × 2.7 s = 5 m, far
    below a texel).
- **Cost:** 0 GPU, a few µs of CPU a second.
- **Risk:** the merge statistics in the population tests shift. Re-run the
  coverage-asked-vs-drawn guard.

### 9. A fast low scud layer under storms
- **Change:** when `storms` > 0 or coverage > 0.6, lower the stratocumulus
  slot to a 400-700 m base, thin and broken, at 1.3-1.6× the cumulus wind
  (fractus under a storm runs ahead of the main deck).
- **Look:**
  - overhead about 1.7°/s and at 30° about 0.9°/s (table §1.1): unmistakable
    motion at real time;
  - parallax against the deck above;
  - matches storm references, where the low ragged cloud is what visibly
    races.
- **Cost:** 0 (an existing layer re-parameterised). The motion-vector term
  grows to about 1.2 px a frame overhead, which is still fine.
- **Risk:** it needs the weather state to drive it, so it looks wrong on a
  fair day. Gate it.

### 10. Flow-mapped cirrus and altocumulus sheets (the HZD and HDRP 2D layers, done right)
- **Change:**
  - in `cirrus_tau` (map cost only), distort the lookup by a coarse flow
    field: a 64² RG8, or analytic shear along the jet (`LAYER_WINDS`
    cirrus 38°);
  - two-phase blend so it never resets;
  - stretch the streaks along the local flow (`CIRRUS_STRETCH` already
    exists).
- **Look:** cirrus fibres shear and curve instead of translating as a
  sheet. It is high and slow, so the effect is subtle but continuous.
- **Cost:** +1 fetch per cirrus texel per map, negligible.
- **Risk:** low.

### 11. Server weather that moves (fronts and pressure cells)
- **Change:** the rework doc's server field (§3.1), plus Sea of Thieves's
  ideas:
  - a server-authoritative wind phase (all clients see the same sky);
  - moving high- and low-pressure cells that thin and thicken the
    population;
  - fronts with 5-15 min passages.
  - Stream as `Option` fields with the client fallback being today's seeded
    weather.
- **Look:** banks approach from the horizon, the sky clears behind a front,
  and coverage visibly changes over minutes: the "evolving with weather"
  part.
- **Cost:** headset 0; kilobytes a minute.
- **Risk:** protocol work; determinism across reconnects (rework doc §3.3).

### 12. Optional: rebuild the horizon band half as often
- **Change:** rows below about 10° (about 23% of the map at MAP_PACK 1.5)
  are marched every second map. Their angular motion and evolution are the
  slowest, and with item 4's history they hold.
- **Cost:** about −10% of the march (−0.06 to −0.08 ms).
- **Risk:** the light at dusk changes fastest at the horizon. Skip this
  during civil twilight.

### Not recommended now

- **Per-pixel flow or curl distortion at display.** The sky shader is at
  +385 of a +400 cap, and every display tap is paid about 3 M times a frame
  on a fill-bound GPU.
- **Yōtei's third map at display.** Same reason. Item 4 gives the same
  smoothing in the march.
- **Screen-space 1/16 reprojection (Nubis 2015).** Head rotation would
  invalidate the history every frame in VR, and our direction-space cache
  needs none.
- **Voxel clouds (Nubis³).** Memory and fetch bandwidth, and the content is
  offline-simulated anyway.
- **Pre-rendered time-lapse panoramas with motion-vector frame blending.**
  Cheap on the GPU, but frozen weather, and it cannot respond to the server
  or to the sun.

### Summary table (estimates on Quest 3; the baseline march is about 0.6-0.8 ms, inferred)

| # | Change | Look gain | GPU ms | Where |
|---|---|---|---|---|
| 1 | Rotate noise domain / ease x,z only | High (stripes) | +0.005 | march |
| 2 | Continuous jittered lattice (view + light) | High (stripes) | ~0 | march |
| 3 | Freeze map inputs; splat at map start | Medium (bands) | ≤0 | CPU, march |
| 4 | Map-space history, then fewer steps | Medium + perf | −0.2 to −0.35 | march |
| 5 | Curl two-phase churn; population rise | High (alive edges) | +0.03-0.07 | march, CPU |
| 6 | Cloud time-scale lever (default 1) | Very high when >1 | 0 | CPU |
| 7 | Far ground-shadow cascade, advected | Very high | +0.02-0.1 | march, ground |
| 8 | Per-cloud velocity, gusts | Medium | 0 | CPU |
| 9 | Storm scud layer | High in storms | 0 | params |
| 10 | Flow-mapped cirrus | Low-medium | ~0 | march |
| 11 | Server fronts and pressure cells | High over minutes | 0 | server |
| 12 | Horizon rows every 2nd map | — | −0.06 to −0.08 | march |

The net of 1-5 and 8-10 is about −0.15 to −0.3 ms, before item 7.

Order:

1. Items 1-4 together, in one change: they share the `fs_map` edits and
   the twin tests.
2. Then 5 and 8.
3. Then 7 (with pass timers).
4. 6 and 9 are parameters the user can try at any point.

---

## 7. What could not be verified

- **Unreadable primary sources.** The Nubis 2017, Nubis Evolved and Nubis³
  slide PDFs and the Frostbite 2016 course PDF are over 10 MB and could not
  be fetched. The RDR2 slides are a 232 MB PPTX. So the following are
  unverified:
  - how HFW's "temporal upscaling for faster moving clouds" works;
  - Frostbite's cloud temporal scheme;
  - RDR2's animation (beyond Hillaire's one-line summary).
- **Nubis 2015 wind skew and upward offset** in code: recollection of
  GPU Pro 7 [R].
- **No primary source** for:
  - a shipped title's cloud time-scale factor;
  - MSFS cloud internals;
  - Death Stranding's cloud method;
  - AC Unity/Odyssey clouds;
  - Soulframe's clouds;
  - any Quest or PSVR2 title's sky (Red Matter 2, Asgard's Wrath 2,
    Batman: Arkham Shadow, Behemoth, Horizon Call of the Mountain).
- **Our own costs** are inferred from one A/B. They have not been timed per
  pass.
- **The detection threshold** (1-2 arcmin/s) is the cited central-vision
  figure. Periphery and HMD persistence will change it.

## 8. Sources (accessed 2026-10-21)

- W1 Schneider and Vos, "The Real-time Volumetric Cloudscapes of Horizon:
  Zero Dawn", SIGGRAPH 2015 Advances (slides with notes, read):
  https://www.advances.realtimerendering.com/s2015/The%20Real-time%20Volumetric%20Cloudscapes%20of%20Horizon%20-%20Zero%20Dawn%20-%20ARTR.pdf
- W2 Guerrilla, "Nubis: Authoring Real-Time Volumetric Cloudscapes with the
  Decima Engine" (2017):
  https://www.guerrilla-games.com/read/nubis-authoring-real-time-volumetric-cloudscapes-with-the-decima-engine
  and slides
  https://advances.realtimerendering.com/s2017/Nubis%20-%20Authoring%20Realtime%20Volumetric%20Cloudscapes%20with%20the%20Decima%20Engine%20-%20Final%20.pdf
- W3 Guerrilla, "Nubis, Evolved" (2022): https://www.guerrilla-games.com/read/nubis-evolved
  and slides
  https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-NubisEvolved-NoVideos.pdf
- W4 Guerrilla, "Nubis³" (2023): https://www.guerrilla-games.com/read/nubis-cubed
  and slides https://advances.realtimerendering.com/s2023/Nubis%20Cubed%20(Advances%202023).pdf
- W5 Schneider, "The Real-Time Volumetric Superstorms of Horizon Forbidden
  West", GDC 2022: https://www.gdcvault.com/play/1027688/The-Real-Time-Volumetric-Superstorms
- W6 Hillaire, "Physically Based and Scalable Atmospheres in Unreal Engine",
  SIGGRAPH 2020 PBS course (slides with notes, read):
  https://blog.selfshadow.com/publications/s2020-shading-course/hillaire/s2020_pbs_hillaire_slides.pdf
- W7 Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering
  Technique", EGSR 2020: https://diglib.eg.org/items/8a3e5350-18b3-46bd-9274-3add5af88c75/full
- W8 Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in
  Frostbite", SIGGRAPH 2016:
  https://www.ea.com/frostbite/news/physically-based-sky-atmosphere-and-cloud-rendering
- W9 Epic, Volumetric Cloud component:
  https://dev.epicgames.com/documentation/unreal-engine/volumetric-cloud-component-in-unreal-engine
- W10 Digital Production, "Unreal Engine 4.26 Flowmaps Shape Volumetric
  Clouds" (10 Sep 2020): https://digitalproduction.com/2020/09/10/flowmaps-3d-unreal-engine/
- W11 Advances in Real-Time Rendering 2019 (Bauer, RDR2):
  https://advances.realtimerendering.com/s2019/index.htm
- W12 Patry, "Lighting, Atmosphere, and Tonemapping in Ghost of Tsushima",
  SIGGRAPH 2021 Advances:
  https://www.advances.realtimerendering.com/s2021/jpatry_advances2021/index.html
- W13 PlayStation Blog, "Ghost of Yōtei – tech deep dive":
  https://blog.playstation.com/?p=411201
- W14 Ang, Catling, Cifariello Ciardi, Kozin, "The Technical Art of Sea of
  Thieves", SIGGRAPH 2018 Talks (read):
  https://history.siggraph.org/wp-content/uploads/2022/09/2018-Talks-Ang_The-Technical-Art-of-Sea-of-Thieves.pdf
- W15 "Reverse engineering the rendering of The Witcher 3, part 14: cirrus
  clouds": https://astralcode.blogspot.com/2019/05/reverse-engineering-rendering-of.html
  and part 12: https://astralcode.blogspot.com/2019/03/reverse-engineering-rendering-of_17.html
- W16 Simul trueSKY docs: https://docs.simul.co/Reference/Clouds/cloudkeyframe.html ,
  https://docs.simul.co/Tutorials/Sequencer/sequenceroverview.html
- W17 Unity HDRP Cloud Layer:
  https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@14.0/manual/Override-Cloud-Layer.html
- W18 Meta, Application SpaceWarp (native):
  https://developers.meta.com/horizon/documentation/native/android/os-app-spacewarp
- W19 Meta, Application SpaceWarp (Unity):
  https://developers.meta.com/horizon/documentation/unity/os-app-spacewarp/
- W20 Toft, Bowles, Zimmermann, "Optimisations for Real-Time Volumetric
  Cloudscapes" (2016): https://arxiv.org/pdf/1609.05344
- W21 Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010:
  https://www.advances.realtimerendering.com/s2010/Vlachos-Waterflow(SIGGRAPH%202010%20Advanced%20RealTime%20Rendering%20Course).pdf ;
  Grimes, GDC 2011: https://www.hlportal.de/files/GDC2011_BronwenGrimes_NonstandardTextures.pdf
- W22 PC Gamer, "Here's how Microsoft Flight Simulator creates its realistic
  weather":
  https://www.pcgamer.com/uk/heres-how-microsoft-flight-simulator-creates-its-realistic-weather ;
  forum analysis:
  https://forums.flightsimulator.com/t/investigating-how-msfs-draws-clouds-for-weather/482387
- W23 Star Citizen: https://starcitizen.tools/Planet_Tech_v4 ;
  PCGH (Sep 2026):
  https://www.pcgameshardware.de/Star-Citizen-Spiel-3481/News/Wolken-Rendering-Quarter-Res-Monatsbericht-September-2026-1556196/
- W24 Decima, "Advances in Lighting and AA" (2017):
  https://www.guerrilla-games.com/read/decima-engine-advances-in-lighting-and-aa ;
  frame breakdown: https://mamoniem.com/behind-the-pretty-frames-death-stranding/
- W25 Meta Quest compare (Quest 3: 2064×2208 per eye, 25 PPD, 110°×96°):
  https://www.meta.com/quest/compare/
- W26 Legge, "Displacement detection in human vision" (1981):
  https://legge.dl8.umn.edu/sites/legge.psych.umn.edu/files/files/media/legge81_displacement_detection_in_human_vision.pdf ;
  DiBianca et al. 2023: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10665501/
- W27 "Life Cycle of Shallow Marine Cumulus Clouds From Geostationary
  Satellite Observations": https://e-docs.geo-leo.de/handle/11858/9771 ;
  "Do Optically Denser Trade-Wind Cumuli Live Longer?":
  https://e-docs.geo-leo.de/handle/11858/10849
- W28 "Updraft and downdraft characterization with Doppler lidar" (ACP
  2010): https://acp.copernicus.org/articles/10/7845/2010/ ;
  "Radar observations of updrafts, downdrafts and turbulence in fair-weather
  cumuli":
  https://researchconnect.stonybrook.edu/en/publications/radar-observations-of-updrafts-downdrafts-and-turbulence-in-fair-/
- W29 Wikipedia, Cumulus cloud (base heights): https://en.wikipedia.org/wiki/Cumulus_cloud
- W30 GDC, "Design and Rendering Strategies for Blockbuster VR Games -
  Batman: Arkham Shadow": https://gdcvault.com/play/1035431/Design-and-Rendering-Strategies-for
- W31 Mixed, "Red Matter 2 offers great graphics":
  https://mixed-news.com/en/red-matter-2-offers-great-graphics-and-more-upgrades/
- W32 Cinevva, "Volumetric clouds and weather effects in modern games"
  (secondary survey, May 2026):
  https://app.cinevva.com/blog/2026-05-04-volumetric-clouds-and-weather
- W33 Handmade Network, "Fast High-Quality Cloudscape through Noise and
  Reprojection":
  https://handmade.network/p/75/monter/blog/p/7201-engine_optimization__fast_high-quality_cloudscape_through_noise_and_reprojection
- W34 Kummerländer, "Noise and Ray Marching" (2021):
  https://code.kummerlaender.eu/blog_content/tree/articles/2021-09-26_noise_and_ray_marching.org
