# Realistic, visibly moving clouds: what Unreal's sky products do, why ours read as still and soft, and what to change first (2026-10-21)

Scope: research only. No code was changed. This report follows the user's
headset note of 2026-10-21: "the clouds still barely read as moving at all,
definitely not achieving a realistic look". With the head still, a cumulus
drifted about 1.6° in 18 s, with modest reshaping. The clouds also look
soft or blurry, and small clouds look smeared.

The user named "Skywright for Unreal Engine" as a reference, along with Ultra
Dynamic Sky and UE5's built-in clouds. The priority is realism first, then
cost.

It builds on, and does not repeat, these reports:

- `living-sky-research-2026-10-21.md`: angular-speed tables, shipped games,
  striations, SpaceWarp motion vectors;
- `cloud-motion-research-2026-10-21.md`: convection, shear, the diurnal
  cycle, Cb, fronts;
- `cloud-rework-research-2026-10-20.md`: the population and envelopes;
- `sky-research-games-2026-10-20.md`.

**Evidence tags:**

- **[P]** primary source read today (slides, notes, docs or the shipped
  README);
- **[A]** abstract or listing only;
- **[S]** secondary source or forum;
- **[R]** recollection, not re-verified today;
- **[C]** read from our code today;
- **[D]** derived here by arithmetic;
- **[U]** unverified.

**Code read for the diagnosis [C]:**

- `space_soup/src/renderer/cloud_sky.rs`: `MAP_*`, `HISTORY_*`, `fs_map`;
- `space_soup/src/renderer/sky.rs`: `sky_cloud_map`, `sky_clouds`;
- `space_soup_sky/src/clouds.rs`: `CloudParams::time_scale`;
- `space_soup_sky/src/cloud_layers.rs`: `CHURN_*`;
- `space_soup_sky/src/time_of_day.rs`: `cloud_seconds`;
- `games/artytest1/scenes/test_room.json`: a 20-minute day, `wind_speed` 12,
  no `time_scale`, so 1.

---

## 0. Bottom line

1. **The 1.6° in 18 s is the physically correct speed. It is not a bug.**
   - That is 0.089°/s.
   - For a cumulus at 1.8 km in a 12 m/s wind, moving across the line of
     sight, it puts the cloud at about 13.5° elevation [D]. At the 1.53 km
     morning base it puts it at 11.4°.
   - At that elevation a real sky moves this slowly too. Real skies still
     read as alive through:
     - the low, near and overhead clouds;
     - their **edges changing**;
     - **shadows sweeping the ground**;
     - **light changing** as clouds cross the sun;
     - **parallax** between layers.
   - In our sky most of those cues are weak or missing. Our weather is also
     at the slow end of what real skies show: v/h = 12/1800 = 0.0067 s⁻¹.
2. **"Skywright" could not be found** as an Unreal product on Fab, itch.io,
   80.lv, the UE forums, GitHub or the web (searched 2026-10-21) [U].
   - The closest real product is **Sky Creator** by Dmitry Karpukhin.
   - The best documented reference is **Ultra Dynamic Sky (UDS)**. Its
     shipped README says three things that matter here [P]:
     - cloud motion is one **Cloud Phase** parameter that both moves *and
       reforms* the clouds;
     - cloud speed can be tied to wind intensity;
     - an option, **"Clouds Move With Time of Day"**, keeps the clouds in
       step with a sped-up clock "during a short time lapse".
   - So the reference the user named couples cloud time to the game clock
     when asked. We currently run 72× sun over 1× clouds, a combination no
     real sky or real time-lapse shows.
3. **The softness is mostly our own filter chain** [C/D]:
   - The display reads each map through an *approximating* cubic B-spline.
     That blur has σ ≈ 0.58 texel per axis, about 2.6 Quest 3 pixels at
     0.176°/texel.
   - A bilinear history carry is chained at 70% weight per map, about 2.3
     resamples deep on average.
   - The display cross-fades two maps 2.7 s apart.
   - Whatever the carry does not predict (convection rising inside a cloud,
     per-cloud growth, shear lean, life-cycle changes) appears twice,
     offset. That is the "smeared small clouds".
4. **What shipped cached-sky games do** [P]:
   - **Ghost of Tsushima** renders clouds into a 768×768 paraboloid texture
     timesliced over 60 frames. It keeps "two for scrolling & lerping",
     anti-aliases cloud density from texture-coordinate derivatives, and
     uses a depth-dependent Henyey-Greenstein lobe (g 0.85 forward,
     −0.15 back).
   - **Ghost of Yōtei** stores each texel's mean and mean-squared cloud
     depth. It uses them for parallax-correct scrolling, blends **three**
     cloud frames instead of two, and sped cloud motion up on purpose for
     "stormy, unsettled weather".
   - **Nubis** reprojects with the cloud's own scroll added to the
     reprojected position. Its talks say plainly that animating noise
     through a fixed coverage map is "pseudomotion".

**Ranked changes (§6 table):**

1. **Make the motion visible inside the real envelope.**
   - Pick weather that real breezy days have: base about 1.0-1.2 km and
     15-18 m/s, which is v/h ≈ 0.015, 2.2× today.
   - And/or set a per-level cloud `time_scale` of 2-3 (√72 ≈ 8.5 is the
     upper bound for a 20-minute day).
   - Cost: zero GPU.
2. **Sharpen the reconstruction.**
   - Prefilter each finished map so the B-spline interpolates. This is a
     one-off separable pass per map, about 0 ms per frame amortised.
   - Make the history carry bicubic, or carry less of it.
3. **Carry what actually moves.** Write a per-texel motion (including
   per-cloud drift and rise) next to `alts`, and use it in the history carry
   and the display carry, so small clouds do not double-image.
4. **Cloud shadows over kilometres of ground,** and light that responds as
   clouds cross the sun.
5. **Fine-edge churn at 20-60 s timescales** on top of today's 90-150 s
   churn, at no extra taps.
6. **Low fast scud and inter-layer parallax.**
7. **Lighting definition:** dark edges and bases, a depth-dependent forward
   lobe, a separate silver-lining lobe.
8. **Later, when shader space is bought back:** per-pixel display-time edge
   detail.

---

## 1. "Skywright", Ultra Dynamic Sky, Sky Creator and UE5's built-in clouds

### 1.1 "Skywright" [U]

**Searches (2026-10-21):**

- exact "Skywright" with Unreal, UE5, Fab, sky, clouds and weather;
- standard and extended web search;
- fab.com (blocked with HTTP 403 to the fetcher);
- itch.io, 80.lv and the Unreal forums.

**What turned up under that name, none of it an Unreal sky product:**

- a Kotlin Android arcade game;
- a Claude Code game-design-doc plugin;
- a Godot airship game.

**Conclusion:** either it is new or unindexed, or it is a misremembered name.
Candidates, by similarity:

- **Sky Creator** (Dmitry Karpukhin), the most likely:
  - a commercial UE plugin, v1.41.3, listed for UE 4.26-5.8, $89.99 on
    itch.io and also on Fab [P/A];
  - built "upon native Volumetric Cloud & Sky Atmosphere components";
  - a "Unified Multi-layered Volumetric Cloud material with extended quality
    settings" (two layers in 2021; multi-layer announced on 80.lv on
    2024-10-03);
  - 30 weather presets in 2021, 39 now, switchable at runtime;
  - simplified or real sun and moon positions, a star map with moon phases;
  - CPU and GPU particle rain, snow, lightning and "volumetric wind";
  - a "cloud-aware lightning system" in 1.30.
  - Cloud-motion details are **not documented publicly**. The changelog only
    mentions "Cloud Wind Skew" fixes, and Multi Scattering Approximation
    Octave Count and Ground Contribution exposed as quality options
    (v1.04, 2021-06-18).
  - It lowered its default Cloud Shadow Extent from 50 to 25 for "better
    cloud shadows with same performance" (v1.02, 2021-05-17) [P, forum].
  - No mobile or VR statement; the platforms listed are Win, Mac and Linux.
- **Stylized Dynamic Sky & Weather System** (Fab) and a free GitHub
  "DynamicSkySystem" (UE 5.6). Both are minor [A].
- **trueSKY** (Simul), already covered in the living-sky doc.

**Action:** ask the user for the link. Until then this report uses UDS and
Sky Creator as the stand-ins.

### 1.2 Ultra Dynamic Sky (Everett Gunther)

Source: the shipped `README` (via a third-party mirror of a project's
`Content/UltraDynamicSky/Blueprints/README.txt`, version not shown), the Fab
listing, and a 2023 tutorial [P unless marked].

**Sky modes:**

- **Volumetric Clouds**, the default: "a field of full 3D clouds… the most
  realistic visuals, but at a performance cost that may be too much for some
  projects/platforms". It is UE's Volumetric Cloud component with UDS's
  material, and its controls are:
  - Bottom Altitude and Volumetric Clouds Scale;
  - View Sample Scale ("if they're looking too noisy, you can increase
    this");
  - Two Layers ("will increase the cost of cloud rendering very
    significantly");
  - High Frequency Levels ("increasing this above the default of 1 will
    increase cost").
- **2D Dynamic Clouds**: "panning 2D textures" on the sky sphere. Cloud
  Height only fakes perspective.
- **Static Clouds**: "a static texture… with different lighting angles
  packed into it to allow the clouds to approximate a simple lighting effect
  as the sun and moon move". A Static Clouds Authoring Tool renders UDS's
  own volumetric clouds into such textures. This is the same idea as our
  cached map, frozen.
- **Cloud Wisps**: a static, high-altitude grayscale wisp texture drawn
  behind every mode, tinted by time of day.
- No Clouds, Volumetric Aurora and Space.
- A third-party listing claims a "Voxel Clouds" mode in v9.5 (May 2026). It
  is not confirmed on Fab [U].

**Cloud movement (the part that matters here):**

- "The Cloud Movement category contains the settings for how volumetric
  clouds and 2D dynamic clouds will move. You can adjust Cloud Speed and
  Cloud Direction from here." Ultra Dynamic Weather's wind direction
  overrides the direction.
- **Cloud Phase:** "particularly useful when Cloud Speed is set to 0… to
  fine tune an exact formation." Animating Phase "will change the position
  and formation of the clouds" [P, Versluis 2023-09-08].
  - So UDS's clouds **move and reform on one parameter**. Motion is never a
    rigid scroll.
  - Versluis interpolates Phase from 1 to 10 over 30 s to get clouds that
    drift "slightly faster than real time".
- **"Clouds Move With Time of Day** is useful for situations where the speed
  of time can change, to keep cloud movement in sync with time during a
  short time lapse."
- **UDW "Cloud Speed Multiplier**… how Cloud Speed is set on UDS based on
  wind intensity": wind gusts and storms speed the clouds up.
- Randomize Cloud Formation On Run.
- **Time:** Day Length and Night Length are "in minutes… the real time that
  will pass between sunrise and sunset". Simulation Speed "1 being roughly
  realistic speed".
- **UDS's default Cloud Speed number** is not in the README; it is a tooltip
  in the editor [U].

**Other realism cues it ships:**

- **Cloud Shadows** on the directional light, in all modes. In volumetric
  mode they are "accurate to the 3D layer, so that every cloud in the layer
  casts its own correct shadow", with intensity and softness varying with
  coverage.
- **Volumetric Cloud Light Rays**, on by default: "giant additive cards
  controlled by a niagara system" placed at gaps in painted coverage.
  "Individual Cloud Light Rays" casts them through gaps between individual
  clouds at extra cost.
- A Volumetric Cloud Painter for local coverage.

**Mobile and VR:**

- With Feature Level set to Mobile, "incompatible Sky Modes like Volumetric
  Clouds and Volumetric Aurora will be replaced" by configurable mobile
  modes. UDS does not run volumetric clouds on mobile renderers.
- The only VR note is an Instanced Stereo issue with rain-splash particles.
- Its documented cost order: Volumetric > 2D Dynamic > Static > None.

**What makes UDS look realistic** (inferred from the above [D]):

- UE's physically based Sky Atmosphere and volumetric clouds;
- motion coupled with reformation;
- per-cloud ground shadows;
- god-ray cards through gaps;
- high-altitude wisps for depth;
- cloud speed tied to wind and, optionally, to the clock.

### 1.3 UE5 built-in Volumetric Cloud + Sky Atmosphere

**Component** (Epic docs, UE 5.8 page, undated) [P]:

- Layer Bottom and Layer Height (km);
- Tracing Start Max Distance and Tracing Max Distance;
- planet radius and ground albedo;
- per-sample atmosphere transmittance and sky-light cloud-bottom occlusion;
- View, Reflection and Shadow Sample Count Scales;
- Shadow Tracing Distance and Stop Tracing Transmittance Threshold.

Default values are not on the page. Recalled from engine source [R/U]:

- bottom 5 km, height 10 km;
- start max distance 350 km, max distance 50 km;
- shadow tracing 15 km, stop threshold 0.005.

**Material:**

- The sample material is `m_SimpleVolumetricCloud_Inst` in
  `Engine/EngineSky/VolumetricClouds` [P].
- The Volumetric Advanced Material Output defaults are [P]:
  - Phase G 0, G2 0, Phase Blend 0, so dual-lobe HG is opt-in;
  - Multi Scattering Contribution, Occlusion and Eccentricity 0.5 each
    (Wrenninge-style octaves; 0 octaves means single scattering).

**Lighting [P]:**

- The directional light has Cast Cloud Shadows, a cloud shadow map with
  extent (km), resolution scale and ray sample count.
- The sky light has a cloud ambient-occlusion map.

**Temporal reconstruction** (Sébastien Hillaire on the UE forum, 2022-11-15)
[S, author is Epic's sky/cloud engineer]:

- `r.VolumetricRenderTarget.Mode 0` "trace a resolution/4, reconstruct at
  resolution/2". It "looks less sharp, more half res".
- Mode 2 "trace 1 pixel out of 4x4 at full resolution, the other 15 samples
  are reprojected". It "looks sharper… but has lots of latency to it".
- 5.2 added "sharper depth edge reconstruction, less holes, less trails
  during reprojection".
- The trade is exactly ours: amortised tracing buys detail and costs motion
  latency and trails.

**Motion:** UE ships no cloud *simulation*. Motion is whatever the material
does, typically scrolling noise and coverage by a wind vector [R]. Products
like UDS add phase, speed and clock coupling on top.

**Scalability intent:**

- Hillaire's SIGGRAPH 2020 deck lists "scalability because Fortnite runs on
  mobile" as a goal of the atmosphere work [P].
- Volumetric *clouds* are still swapped out on mobile by products such as
  UDS [P].

---

## 2. Why clouds read as moving

### 2.1 The geometry, and what the user measured [D]

A flat layer at height h, wind v, seen at elevation e:

- across the line of sight: `ω⊥ = v·sin e / h`;
- toward or away from the viewer: `ω∥ = v·sin²e / h`.

Degrees per second, ⊥ (∥ is smaller still):

| Case | v/h (s⁻¹) | zenith | 30° | 20° | 15° | 10° | 5° |
|---|---|---|---|---|---|---|---|
| Ours: Cu 1.8 km, 12 m/s | 0.0067 | 0.38 | 0.19 | 0.13 | 0.10 | 0.066 | 0.033 |
| Ours: Cu morning base 1.53 km | 0.0078 | 0.45 | 0.22 | 0.15 | 0.12 | 0.078 | 0.039 |
| Ours: Sc 1.08 km, 9.6 m/s | 0.0089 | 0.51 | 0.26 | 0.17 | 0.13 | 0.088 | 0.044 |
| Breezy real day: Cu 1.0 km, 16 m/s | 0.016 | 0.92 | 0.46 | 0.31 | 0.24 | 0.16 | 0.080 |
| Scud 0.6 km, 15 m/s | 0.025 | 1.43 | 0.72 | 0.49 | 0.37 | 0.25 | 0.13 |

**The measurement:**

- 1.6°/18 s = 0.089°/s, which is our cumulus at about 13.5°.
- A real cumulus there, in that wind, moves the same.
- Detection thresholds for slow motion are about 1-2 arcmin/s (0.017-0.033°/s)
  in central vision, rising with eccentricity to 8-10 arcmin/s at 90°
  [S: MDPI 2025 citing Leibowitz; Burr & Ross 2002].
- So the cloud the user watched was 3-5× above threshold: "barely moving" is
  the honest percept.
- A real observer would also have had the cues in §2.3.

**Stereo:** with a 63 mm IPD the disparity of a cloud 1 km away is
0.0036° = 0.09 px at 25 PPD; at 200 m it is 0.45 px [D]. A mono sky map is
correct for anything over a few hundred metres.

### 2.2 What real weather allows

Apparent speed scales with v/h, not with v alone [D].

- Real cumulus and stratocumulus occur from about 0.5 to 2.5 km with winds at
  cloud level from calm to about 20 m/s or more on breezy, post-frontal or
  maritime days. v/h spans roughly 0.001-0.04 s⁻¹ [D from typical values;
  ranges in the cloud-motion doc §2].
- Ours, 0.0067, is in the lower third.
- A level that wants clouds that visibly move **can choose a breezier, lower
  sky and stay entirely physical**. 2-4× faster across the sky is ordinary
  weather.

### 2.3 What makes motion perceptible besides drift [P/D]

1. **Edges changing.** This is the shape cue that does not depend on the
   wind's direction relative to the view.
   - Eddy turnover scales roughly as τ ≈ L/u(L). With u(L) ≈ 2 m/s·(L/500 m)^⅓
     [D, Kolmogorov-style scaling, illustrative], eddies of 20 m turn over in
     about 30 s, 50 m in about 55 s, 100 m in about 85 s and 300 m in about
     180 s.
   - Our churn periods are 150 s (Sc), 90 s (Cu) and 120 s (Ac) [C], which
     is the turret band.
   - There is **nothing in the 20-60 s band** that makes an edge visibly
     different between two glances 10 s apart.
   - A 20-50 m feature at 2-5 km spans 0.25-1.4°, which is 1.3-8 texels:
     resolvable on nearer clouds.
2. **Parallax between layers.** Different heights and directions give
   different angular speeds. Ours has it: the layers turn 12°, 0°, −25° and
   38° (living-sky doc). But every layer is slow.
3. **Cloud shadows sweeping the ground** at the full wind speed: 12 m/s is
   43 km/h. Seen from 50 m away that is 13.7°/s. This is the strongest
   real-time cue.
   - UDS, Sky Creator, UE and Ghost of Yōtei all ship cloud shadow maps
     [P].
   - Ours covers about 1 km on a 64² grid (living-sky doc).
4. **Light changing** when a cloud crosses the sun:
   - the ground and scene dim and brighten;
   - the sun disc and halo dim (we do this [C]);
   - silver linings appear on edges near the sun;
   - shafts of light come through gaps (UDS's light-ray cards).
5. **Near and overhead clouds.** Overhead cumulus moves 0.38°/s at our
   settings. Low scud under storms moves 1-3°/s.
6. **Coupled move-and-reform.**
   - UDS's Phase, and Nubis's statement: "while the noise in the density
     model is animated, the coverage signal is not… [clouds] only… appear
     to be changing without altering the larger structure" [P, Nubis 2017].
   - Nubis 3's own comparison table rates both its envelope and voxel
     methods "Evolution: Pseudomotion Only". Only the weather-driven
     vertical-profile method gets "Yes" [P, Nubis Cubed 2023].
   - Our population with life cycles is *real* evolution, which shipped
     Nubis skies do not have. Its timescales are real, though, so in 18 s
     it shows almost nothing.

### 2.4 What shipped games do with speed

- **Horizon Zero Dawn (2015):** the storm clip is "sped up, for effect"
  [P]. Nubis (2017) shows "a sped up example" of wind-advected noise with
  `p += wind_direction * time_offset; p += height_fraction * wind_direction * 500.0`
  for skew [P]. No in-game factor is given.
- **Ghost of Yōtei (2025):** the motion was sped up to convey "stormy,
  unsettled weather". It was made artefact-free by per-texel depth statistics
  and three-frame blending [P, Bentley 2025-10-23].
- **UDS:** cloud speed is a free parameter. It can be scaled by wind
  intensity, or locked to the game clock with "Clouds Move With Time of Day"
  [P].
- **Microsoft Flight Simulator:** clouds come from a meteoblue model with
  real wind vectors, so the speeds are real [S, simtuts guide]. No official
  statement on how clouds are advected was found [U].
- **RDR2, Assassin's Creed, Star Citizen, Ghost of Tsushima:** no primary
  statement of a cloud time-scale factor was found [U]. This repeats the
  living-sky doc's finding.

**For accelerated days**, three policies exist [P/D]:

- **Lock to the clock** (UDS's option). This is a coherent time-lapse. At
  D = 72 it is too fast: 27°/s overhead, and 10-40 min lives become 8-33 s.
- **Real time** (ours today). The light races over frozen clouds, which is
  incoherent.
- **A compromise k:**
  - √D (8.5 here) makes the light and the clouds equally "wrong";
  - k = 2-3 keeps the cross-sky speed inside the real v/h envelope (§2.2)
    while lifetimes shrink only to 3-20 min, still within observed cumulus
    lifetimes of a few minutes to tens of minutes [P/A, satellite tracking,
    W27 in the living-sky doc].

Our `CloudParams::time_scale` already scales winds, churn, lives and weather
together, clamped to 0-30 [C]. test_room simply leaves it at 1.

**Comfort caution [U]:** a large, fast-moving overcast fills the upper visual
field and could contribute to vection in VR. No measured threshold was found.
Keep overhead overcast motion at realistic values (well under about 1-2°/s)
and test on the headset.

---

## 3. Why clouds look soft, and how the best systems get crisp edges

### 3.1 Where our softness comes from [C/D]

**1. Display reconstruction is an approximating cubic B-spline.**

- `sky_cloud_map` uses four bilinear taps (Sigg & Hadwiger). It is chosen to
  hide texel squares [C].
- A B-spline does not pass through the samples. Its kernel has variance 1/3
  texel² per axis (σ ≈ 0.58 texel).
- Its response at half the Nyquist frequency (0.25 cycles per texel) is
  sinc⁴ ≈ 0.66 [D].
- At 0.176°/texel in azimuth (`MAP_WIDTH` 2048) this is about 2.6 Quest 3
  pixels of blur on top of the 4.4-pixel texel.

**2. The history carry is chained bilinear.**

- `fs_map` mixes 30% new march with 70% of the newest map, fetched bilinearly
  at the advected uv. That map is itself 70% carried [C].
- The mean chain depth is 0.7/0.3 ≈ 2.3 resamples.
- At fractional offset f a bilinear fetch adds variance f(1−f), up to 0.25
  texel². The chain adds roughly σ ≈ 0.6 texel [D].
- The rejection fires only when alpha differs by 0.2-0.5 [C], so small and
  soft changes are averaged rather than taken.

**3. The display cross-fades two maps 2.7 s apart** [C]. Wherever the carry
mispredicts, the result is two offset copies, and small clouds look smeared.
The carry uses a single wind per texel at the light-weighted height (`alts`).
Not carried:

- rising convection detail inside a cloud;
- per-cloud growth and decay;
- shear lean;
- anything moving differently from `cloud_flow_wind(alt)`.

**4. Noise prefiltering by footprint.** `foot = max(t·texel, 0.25·dt)` [C]
is right against aliasing, but it removes sub-texel erosion by design. Ghost
of Tsushima does the equivalent ("reduce cloud density when derivative is
high") [P].

**Combined** [D]: σ ≈ √(0.33 + 0.4) ≈ 0.85 texel before the texel itself, so
an effective footprint of about 2-3 texels (0.35-0.5°, 9-13 px). A crisp
cumulus edge at 3-5 km has structure at 10-50 m (0.1-1°). We blur away the
lower part of that range.

### 3.2 How shipped systems get crisp detail [P]

**Shape (Nubis, Frostbite, UE):**

- A low-frequency shape noise (Perlin-Worley), "dilated" by coverage through
  a remap.
- Then **eroded at the edges** by a higher-frequency Worley detail noise.
  "If you invert the Worley noise at the base of the clouds you get some
  nice whispy shapes."
- The detail noise is distorted by curl noise, which "is non divergent and is
  used to fake fluid motion" (Schneider 2015).
- Frostbite uses the same split (shape: Perlin-Worley plus Perlin octaves;
  erosion: Worley octaves) and open-sourced the generators (TileableVolumeNoise)
  [P, Hillaire 2016].
- Nubis 3: 128³ four-channel noise at 2 bytes per texel, mip chosen by
  distance, 0.5 m voxel precision [P].

**Lighting that reads as definition:**

- Beer-Lambert plus the "powder sugar" darkening at sun-facing edges
  (Schneider 2015), refined in 2017 into an in-scatter probability:
  - `depth_probability = 0.05 + pow(lodded_density, remap(height, 0.3, 0.85, 0.5, 2.0))`;
  - `vertical_probability = pow(remap(height, 0.07, 0.14, 0.1, 1.0), 0.8)`;
  - these give dark edges and dark bases [P].
- **Silver lining:** `max(HG(cosθ, 0.6), silver_intensity·HG(cosθ, 0.99 − silver_spread))`.
  A second, near-delta forward lobe is kept from darkening clouds at 90°
  from the sun [P, Nubis 2017].
- **Ghost of Tsushima:** HG with g set by how deep the sample is (from sun
  and segment transmittance). It runs g ≈ 0.85 forward to −0.15 back, with
  backscatter scaled about 2.16 to approximate multiple scattering at albedo
  0.9 [P, Patry 2021].
- **Multiple scattering:**
  - Wrenninge octaves: per octave, σs·aⁿ, σe·bⁿ, phase g·cⁿ, with a ≤ b
    (Frostbite: "works very well in practice") [P, Hillaire 2016].
  - Nubis 2017: `max(exp(−d), 0.7·exp(−0.25·d))` [P].
  - Too much multi-scatter or ambient flattens contrast. A cloud looks
    "soft" when its shadowed side is too bright, even when its silhouette is
    sharp [D].
- **Ambient:** Nubis multiplies ambient by height within the cloud and by
  `pow(1 − coarse_density, 0.25)`, so bases go dark [P, Nubis Evolved 2022].
  Frostbite 2016 lists its single-colour ambient as a known weakness [P].

**Resolution and temporal treatment:**

| System | Cloud render | Temporal | Cost |
|---|---|---|---|
| HZD 2015 | 1/16 of pixels per frame in each 4×4 block, reprojected; half-res low buffers at edges | reprojection | about 2 ms PS4 (20 ms without) [P] |
| Nubis 2017 | as above, plus LOD, step size, light only at nonzero density | reprojection | 22 → 8.1 → 3.34 → 1.81 → 1.2 ms PS4 [P] |
| Nubis Evolved 2022 | PS4 960×540, PS5 1920×1080; "without temporal upscaling" | reprojection includes the cloud's own `scroll_direction_2D * inDeltaTime` and a cloud-distance buffer | ≤4 ms PS4, ≤2-3 ms PS5 [P] |
| Nubis Cubed 2023 (voxels) | SDF plus adaptive plus jittered steps (animated hash under 250 m, static beyond) | - | 2.2-4 ms; density sampler 10 ms at 960×540 before optimisation [P] |
| Frostbite 2016 | main view half-res; planar reflection quarter-res | EMA with a constant blend, reprojected | 1.60 ms XB1, worst case [P] |
| UE5 | mode 0: trace /4, reconstruct /2; mode 2: 1/16 per frame at full res | reprojection (latency and trails) | - [S] |
| Ghost of Tsushima | **768×768 paraboloid**, RGB11F (Mie, Rayleigh, transmittance), **timesliced over 60 frames** | two textures "for scrolling & lerping" | - [P] |
| Ghost of Yōtei | cloud map plus per-texel mean and mean² depth | parallax scroll, **3-frame** blend | - [P] |

**Lessons that transfer [D]:**

1. Crispness comes from erosion detail, edge-darkening lighting and an
   *interpolating* reconstruction, not from more rays.
2. Temporal reuse must move the history with the clouds' own motion (Nubis
   Evolved, Yōtei), or it smears.
3. Nubis's PS4 budget used 960×540 for a whole TV view (about 60-70° wide):
   roughly 14 px per degree. Our map is about 5.7 texels per degree in
   azimuth and finer in elevation near the horizon (`MAP_PACK` 1.5 gives
   about 0.08° at 8° elevation). We are not far behind on raw sampling. The
   loss is in reconstruction.

---

## 4. Dome, sky-map and cubemap caches: keeping motion smooth and visible

Techniques found in shipped work [P unless marked]:

1. **Scroll plus lerp between time-offset maps** (Ghost of Tsushima, 2020).
   Three textures: one being rendered over 60 frames, two shown, scrolled
   and lerped. We do the same, with 96 slices and per-height wind [C].
2. **Depth statistics for parallax-correct scrolling** (Ghost of Yōtei,
   2025).
   - Store the mean and mean² of cloud depth per texel.
   - Reconstruct a distribution of opacity along each ray. This also lets
     clouds sit in front of mountains.
   - Scroll every frame with parallax, and blend **three** frames to hide
     transitions at higher speeds.
   - Our `alts` (light-weighted height plus transmittance) is the
     single-moment version [C].
3. **Visual flow on a single panorama** (The Last of Us, Guerrette, GDC
   2014): "vector fields, flow, and some simple math" to suggest clouds
   receding toward the horizon on one matte painting [A]. Frostbite's notes
   call it "very effective" but say it has no variation of shape, weather or
   lighting [P].
4. **Two-phase flow maps** (Vlachos, Valve, 2010).
   - Two layers advected by a flow field, each restarting on a sawtooth,
     offset half a period, cross-faded so the restart never shows [P].
   - Applied to cloud *detail*, this gives unbounded advection without
     stretching.
5. **Static multi-angle light textures** (UDS Static Clouds): a cache with
   several sun angles packed in, so light can change without re-rendering
   [P].
6. **Motion-aware reprojection** (Nubis Evolved): the history is fetched at
   the position the cloud *was*, using a per-pixel cloud distance and the
   scroll vector. The superstorm also rotates through the same path [P].
7. **Density anti-aliasing by derivatives** (Ghost of Tsushima): reduce
   density where the cloud-position derivative with respect to map
   coordinates is high, which avoids shimmering in a low-res cache [P].
8. **Per-pixel high-frequency detail at display time.** Common in 2D sky
   shaders (UDS 2D Dynamic's "noise texture used to generate the clouds"
   [P]). For a cached volumetric map it means modulating alpha near
   partially transparent edges with world-anchored noise advected by the
   wind. No shipped volumetric title was found documenting this exactly
   [U].

---

## 5. VR and mobile specifics

- **No primary source** was found for shipped volumetric clouds on Quest or
  standalone VR. Searches covered Red Matter 2, Meta demos and Quest flight
  games [U].
  - Products targeting mobile (UDS) swap volumetric clouds for 2D or static
    modes [P].
  - Our cached map is already the VR-appropriate design. Ghost of Tsushima
    is the closest shipped analogue [P].
- **Horizon Call of the Mountain (PS VR2)** is by Guerrilla and Firesprite,
  with foveated rendering. No public cloud breakdown was found [U].
- **Microsoft Flight Simulator VR (PC)** uses real-weather volumetric
  clouds [S]. No VR-specific cloud notes were found [U].
- **Quest 3:**
  - 2064×2208 per eye, "25 PPD" at the centre (Meta spec) [P]. Average
    pixels per degree across the field is lower, about 19-21 [S].
  - Our 0.176°/texel is about 4.4 central pixels.
  - Clouds need no stereo beyond a few hundred metres (§2.1) [D].
  - Every moving sky pixel needs SpaceWarp motion vectors. We have them
    (`sky_motion.rs`) [C].
  - Cost scales with pixels (memory notes: fill-bound frame), so per-pixel
    additions to the sky shader cost on every sky pixel. Per-map work costs
    about 1/96 per frame.
- **Shader size:** sky `fs_main` is at +385 of a +400 proxy cap (memory
  `clouds-drift-on-their-winds`) [C]. Anything added at display time must
  be paid for there first.

---

## 6. Recommendations for our system, ranked by realism impact

Costs are **estimates** [D] unless measured. "Map-side" runs in `fs_map` on
16k rays per frame (1/96 of the map). "Display" runs on every sky pixel at
36 Hz.

| # | Change | Why (realism) | Numbers | GPU cost (est) | CPU cost (est) |
|---|---|---|---|---|---|
| 1 | **Visible motion inside the real envelope.** (a) Author test_room as a breezy day: Cu base 1.0-1.2 km, 15-18 m/s, Sc 0.6-0.9 km. (b) And/or `time_scale` 2-3 for the 20-min day, as a per-level value tied to the day's compression (cap at √D). Keep lives scaled with it (they already are). A/B 1 / 2.5 / 8.5 on the headset | The largest perceptual gain; stays physical (v/h ≈ 0.015-0.02 s⁻¹ is ordinary weather); matches UDS's "Clouds Move With Time of Day" behaviour in spirit | Cu at 15° goes 0.10 → 0.24-0.30°/s (7-15× threshold); overhead 0.9-1.1°/s | 0 | Population births and deaths per second × k (worker thread; small) |
| 2 | **Interpolating reconstruction.** When a map completes, run a separable B-spline prefilter (inverse of [1,4,1]/6; exact taps √3·(√3−2)^|k| = 1.732, −0.464, 0.124, −0.033…; a 5-tap truncation normalised to unit gain is [0.118, −0.441, 1.645, −0.441, 0.118] per axis; clamp alpha to 0..1). The display's 4-tap B-spline then interpolates the samples | Removes σ ≈ 0.58 texel (about 2.6 px) of blur; edges and small clouds sharpen; no display taps added | Response at ½ Nyquist (0.25 cycles per texel) 0.66 → 0.98 | One 2-pass filter over 2048×768 per map (every 2.7 s): about 0.1-0.2 ms once, under 0.005 ms per frame amortised; may need a 4th layer or in-place ping-pong (+12.6 MB if a layer) | 0 |
| 3 | **Carry what moves; stop diffusing history.** (a) Fetch history in `fs_map` with Catmull-Rom (5-tap) instead of bilinear, or raise `HISTORY_NEW` to about 0.5 now that the stride lattice no longer needs as much averaging (re-test the grain). (b) Write a per-texel 2D motion (wind at the cloud's height **plus** per-cloud drift and rise, from the population) into `alts` .ba. Use it in both the history carry and the display carry, as Nubis Evolved and Yōtei do. (c) Consider Yōtei's 3-frame blend if crossfade doubling remains | Kills double images and smears of small clouds; keeps temporal noise reduction | Chain blur σ ≈ 0.6 → about 0.2 texel | (a) +4 taps per map texel: about 0.01 ms per frame. (b) `alts` to RGBA16F: +4 B per texel of bandwidth at display (same fetch); map-side ALU only | Per-cloud velocities already exist on the worker |
| 4 | **Cloud shadows over kilometres, and light that responds.** Extend the 64²/1 km ground shadow to cascades, e.g. 128² over 4 km plus 128² over 16 km. Update at map cadence or per frame from the same layer densities; scroll between updates. Dim the scene's sun and indirect light by the shadow at the player. Optionally UDS-style light-shaft cards in gaps near the sun | Shadows racing at 43 km/h over the ground are the most visible real-time cue (13.7°/s at 50 m); passing shadows change everything's light | UE, Sky Creator and UDS ship 25-50+ km shadow extents | Light march per shadow texel, amortised: about 0.05-0.1 ms; +1 tap in ground shaders (watch the terrain I-cache cliff) | 0 |
| 5 | **Fine-edge churn band, 20-60 s.** Add a second, faster, smaller-amplitude rotation to `churn_warp`/`churn_pulse` (period about 30-45 s, warp about ⅓ of `CHURN_WARP`), driven by the shape-noise channels already read. Optionally two-phase (Vlachos) for the erosion coordinates | Edges visibly change between glances, at real turbulent rates (20-50 m eddies turn over in about 30-55 s) | Today 90-150 s only [C] | ALU only (no taps) for one extra rotation; two-phase doubles the erosion tap (+1 tap per march sample: real cost, measure) | 0 |
| 6 | **Low fast scud and stronger inter-layer parallax** (living-sky doc item): a thin, ragged 0.4-0.8 km layer in overcast or storm at 1.3-1.5× the cumulus wind | Low cloud moves 1-3°/s overhead: unmistakable motion, and parallax against the layers above | See the §2.1 table | One more layer in the march when on (only in storms) | 0 |
| 7 | **Lighting definition.** Check the sunlit-to-shaded contrast against photos. Use Nubis's in-scatter probability (dark edges, dark bases). Add a separate silver-lining lobe `max(HG(0.6), s·HG(0.99−spread))` or Tsushima's depth-dependent g (0.85 to −0.15). Limit multi-scatter octaves and ambient on bases | Perceived crispness rises with contrast; silver edges flare as clouds pass the sun, which is itself a motion cue | - | Map-side ALU only | 0 |
| 8 | **Display-time edge detail** (after buying shader space). One world-anchored 2D noise tap at display, applied only where 0.05 < α < 0.95, advected by the per-texel motion from #3, scaled by distance | True per-pixel crispness beyond the 4.4 px texel; the only way past the map's resolution without doubling the map | - | About 1 tap per sky pixel: about 0.05-0.1 ms at 36 Hz on a sky-heavy view (fill-bound); needs about 20-40 shader-size units freed first | 0 |
| 9 | **Higher map resolution** (4096 wide) only if 2+3+8 are not enough | Halves the texel to 0.088° | - | About 2× the map march (measure with the `time_the_sky_map` test); +25 MB | 0 |

**Order of work:**

- 1 and 2 are nearly free and address both complaints directly. Do them
  first and look at them on the headset.
- 3 removes the smear.
- 4 and 5 add the "alive" cues that a real-speed sky depends on.
- 7 is tuning.
- 8 and 9 are last, after measuring what 2 and 3 bought.

**What to measure on the headset after each step:**

- angular drift at 10°, 20° and 45° elevation over 18 s (the user's own
  test);
- edge change between frames 10 s apart (mean |Δα| on edge texels);
- a crispness proxy: the 10-90% alpha rise distance across a cumulus edge,
  in pixels;
- GPU ms on `day_sea`, `beach_south` and `outdoors_field` at GPU level 5.

---

## 7. Not verified

- What "Skywright" is. No product under that name was found (§1.1).
- UDS's default Cloud Speed value, and its claimed v9.5 "Voxel Clouds" mode.
- UE Volumetric Cloud default numbers (recalled from source, not re-checked).
- Any shipped title's numeric cloud time-scale factor.
- How MSFS advects its clouds.
- How Horizon Call of the Mountain renders clouds.
- Any shipped Quest title with volumetric clouds.
- A vection threshold for sky motion in VR.
- The eddy-turnover scaling in §2.3 is illustrative (Kolmogorov-style).
  The 20-60 s band should be checked against time-lapse footage.
- All cost figures in §6 are estimates until benched.

---

## 8. Sources (accessed 2026-10-21; publication dates where given)

**Unreal products:**

- Ultra Dynamic Sky README (shipped with the asset; third-party project
  mirror, version not shown):
  https://gitverse.ru/mramid/MWL/content/develop/Content/UltraDynamicSky/Blueprints/README.txt
- Ultra Dynamic Sky, Fab listing (fetch blocked, 403; listing text via
  search): https://www.fab.com/listings/84fda27a-c79f-49c9-8458-82401fb37cfb
- J. Versluis, "How to keyframe Cloud Movement in Sequencer with Ultra
  Dynamic Sky", 2023-09-08:
  https://www.versluis.com/2023/09/how-to-keyframe-cloud-movement-in-sequencer-with-ultra-dynamic-sky/
- Sky Creator (D. Karpukhin), product page, v1.41.3:
  https://dmkarpukhin.com/sky-creator/ ; itch.io:
  https://dmkarpukhin.itch.io/sky-creator ; UE forum thread from
  2021-04-27: https://forums.unrealengine.com/t/sky-creator/226723 ; 80.lv,
  2024-10-03:
  https://80.lv/articles/sky-creator-plug-in-for-ue5-will-feature-new-volumetric-clouds/
- Sky Creator on Fab: https://www.fab.com/listings/39b1579f-305c-47c5-809e-74fb7d5ec520
- Epic, Volumetric Cloud Component Properties (UE 5.8 docs, undated):
  https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-cloud-component-properties-in-unreal-engine
- Epic, Volumetric Cloud Material:
  https://dev.epicgames.com/documentation/unreal-engine/volumetric-cloud-material-in-unreal-engine
- S. Hillaire, UE forum on `r.VolumetricRenderTarget.Mode`, 2022-11-15 and
  2023-01-15: https://forums.unrealengine.com/t/volumetriccloud/688516
- S. Hillaire, SIGGRAPH 2020 PBS course slides (Unreal atmosphere):
  https://blog.selfshadow.com/publications/s2020-shading-course/hillaire/s2020_pbs_hillaire_slides.pdf

**Shipped-game talks:**

- A. Schneider, N. Vos, "The Real-time Volumetric Cloudscapes of Horizon:
  Zero Dawn", SIGGRAPH 2015 Advances:
  https://advances.realtimerendering.com/s2015/index.html
- A. Schneider, N. Vos, "Nubis: Authoring Real-Time Volumetric Cloudscapes
  with the Decima Engine", SIGGRAPH 2017 (page dated 2017-07-31):
  https://www.guerrilla-games.com/read/nubis-authoring-real-time-volumetric-cloudscapes-with-the-decima-engine
  ; PDF:
  https://d3d3g8mu99pzk9.cloudfront.net/AndrewSchneider/Nubis-Authoring-Realtime-Volumetric-Cloudscapes-with-the-Decima-Engine-Final.pdf
- A. Schneider, "Nubis, Evolved", SIGGRAPH 2022 (page dated 2022-08-05):
  https://www.guerrilla-games.com/read/nubis-evolved ; PDF:
  https://d3d3g8mu99pzk9.cloudfront.net/AndrewSchneider/NubisEvolved/NubisEvolved-NoVideos.pdf
- A. Schneider, "Nubis³", SIGGRAPH 2023 Advances:
  https://advances.realtimerendering.com/s2023/Nubis%20Cubed%20(Advances%202023).pdf
- S. Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in
  Frostbite", SIGGRAPH 2016 course notes:
  https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/s2016-pbs-frostbite-sky-clouds-new.pdf
- J. Patry, "Real-Time Samurai Cinema: Lighting, Atmosphere, and Tonemapping
  in Ghost of Tsushima", SIGGRAPH 2021 Advances:
  https://www.advances.realtimerendering.com/s2021/jpatry_advances2021/index.html
- A. Bentley, "Ghost of Yōtei – tech deep dive", PlayStation Blog,
  2025-10-23: https://blog.playstation.com/?p=411201
- K. Guerrette, "Moving the Heavens: An Artistic and Technical Look at the
  Skies of The Last of Us", GDC 2014 (abstract):
  https://gdcvault.com/play/1020146/Moving-the-Heavens-An-Artistic
- A. Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010:
  https://www.advances.realtimerendering.com/s2010/Vlachos-Waterflow(SIGGRAPH%202010%20Advanced%20RealTime%20Rendering%20Course).pdf
- Microsoft Flight Simulator 2024 live weather (secondary guide):
  https://simtuts.com/guides/msfs-2024-live-weather-guide

**Perception, device, meteorology:**

- Slow-motion detection, 1-2 arcmin/s (citing Leibowitz), MDPI *Inventions*
  10(3):33: https://www.mdpi.com/2411-5134/10/3/33
- D. Burr, J. Ross, "Motion perception", Handbook of Brain Theory draft,
  2002-01-30: https://win.pisavisionlab.org/Downloads/motion_burr.pdf
- Meta Quest 3 specifications (2064×2208 per eye, 25 PPD):
  https://www.meta.com/ca/quest/compare/ ; UploadVR spec table:
  https://uploadvr.com/quest-3-specs
- Shallow marine cumulus life cycle from geostationary satellites:
  https://e-docs.geo-leo.de/entities/publication/aa795daa-5ff5-46b2-aa63-9da95b8a4a56

**Internal:**

- `living-sky-research-2026-10-21.md` (§1 angular speeds, §2 shipped games,
  §6 items);
- `cloud-motion-research-2026-10-21.md`;
- `cloud-rework-research-2026-10-20.md`;
- memory notes `clouds-drift-on-their-winds` and
  `clouds-live-convection-clusters-day-cycle`.
