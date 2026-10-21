# Clouds that live: evolution, real cloud-field statistics, a server weather model, and a cheaper far field (2026-10-20)

> **Status, 2026-10-21: items 2-4 of §6 are built** (client side, stateless
> from the seed, so the server's later `Option` stream is a drop-in):
> `space_soup_sky/src/cloud_population.rs` draws each volumetric layer's
> clouds per (cell, epoch), thinned by the weather's local coverage, sized
> n(D) ∝ D^-3, lives growing with size, each growing, maturing and decaying
> (`life_shape`). The live clouds are splatted once a second of cloud time
> into a 1024² RGBA16F envelope per layer over 128 km in its wind frame
> (`cloud_sky.rs` `fs_splat`; R profile, G x top, B x phase, A squared for
> merging), which `density` reads where it read the weather; past the map
> the weather's mean takes over (`far_share`). Merged clouds rise
> (`MERGE_LIFT`), cores keep the noise's breaks (`COVER_CEIL`), erosion
> follows life phase. Guarded by tests: D^-3 sizes, coverage asked vs drawn
> (within 0.02), no step in a second, the sky renewed in half an hour, no
> correlation at the noise tile (0.19), a weather turn keeps the clouds up,
> and the GPU envelope and shadow against the CPU twin (0.3% / 0.001).
> Still open: items 1 (server weather field), 5-8 (empty-space skipping,
> distance bands, aerial perspective, accumulation), 10, and Cb/rain from
> the population. CPU: an update is 0.3-0.5 ms on the Mac, once a second.

Scope: research only. No code was changed. This report builds on
`sky-research-games-2026-10-20.md` (the "games doc", ranked items R0-R17) and
`sky-realism-research.md` (the plan). Everything those two cover is not
repeated: Nubis lighting and marching, Tsushima/Yōtei map design,
Hillaire/UE5 levers, HDRP, Godot, the 80.lv mobile case, Adreno fetch costs,
and the R-items themselves. Where this report leans on an R-item it names it.

Code read for the diagnosis: `space_soup_sky/src/cloud_layers.rs`
(`layers_at`, `CloudNoise::build`, `local_coverage`, `height_profile`,
`density`), `space_soup_sky/src/clouds.rs` (`CloudParams`, constants),
`space_soup/src/renderer/cloud_sky.rs` (`fs_map`, `local_coverage` WGSL,
`MAP_*`), `space_soup_engine/src/weather.rs`. `space_soup_protocol` carries no
weather today.

**The bottom line.**
- The carpet, the sameness and the horizon band are **not a lighting or
  marching problem**. They come from where cloud shape comes from. Every
  cumulus is a threshold of one Perlin-Worley noise with 4 cells per tile, so
  every cloud has the same size (about 1 km at the default `scale_km` 3). The
  noise repeats every 4 km. Coverage is only nudged by a 256² weather tile
  that repeats every 40 km. Nothing is an object, so nothing can be born,
  grow, merge or die. Scrolling and "boiling" the noise only morphs it.
- What real skies have, and what leading systems imitate, is **clouds as a
  population**:
  - sizes over two decades, with equal cover per size octave (cloud area
    exponent close to 1, robust across climates);
  - lifetimes of tens of minutes, with big clouds living longer;
  - births clustered by the weather and by neighbours, not on a grid;
  - mesoscale organisation (streets, open and closed cells) whose spacing is
    set by the boundary-layer depth;
  - all of it advected by winds that change with height, and modulated by
    fronts.
- **Shipped games almost never simulate this.** The pattern that ships is a
  low-resolution 2D control field, authored or procedurally driven, expanded
  vertically into density and eroded by tiled noise (Nubis 2015-2022, HDRP,
  UE). Evolution comes from scrolling, keyframe interpolation (trueSKY), or
  canned weather states blended over 30-120 s. Guerrilla tried a procedural
  weather simulation in 2015 and dropped it for art control. Nubis³ (2023)
  went to voxel clouds modelled from fluid simulations, but those are offline
  assets. No shipped game was found that simulates individual cloud
  lifecycles at runtime. The research systems that do (Stormscapes 2020,
  Weatherscapes 2021, Thunderscapes 2024) are GPU fluid solvers for 10-20 km
  domains, far beyond a headset.
- **The design that fits us:**
  - The server runs a coarse 2D weather field and a **cloud-object
    population model** with meteorologically measured statistics.
  - It streams kilobytes, not textures.
  - The headset splats the objects into a small per-layer **envelope map**
    (Nubis's "dimensional profile"), which replaces today's weather fetch.
    Each object indexes the noise at its own offset, which kills the tiling.
  - The envelope map doubles as the empty-space-skipping and far-field LOD
    structure.
  - Aerial perspective with real boundary-layer haze fades the horizon.
- **Expected net cost on the headset: negative** (roughly −0.2 to −0.9 ms
  against today, estimated, not measured). The new work is a few hundred
  splats a second and ALU in the march. The savings come from skipping empty
  air, dropping 3D noise in the far band, and longer rebuild periods once
  motion is advected.

---

## 0. Why ours looks the way it does (diagnosis, from the code)

| Symptom on the headset | Mechanism in the code |
|---|---|
| A uniform carpet of near-identical puffs | `CloudNoise::build`: `perlin_fbm3(u, 4, 4, ..)` and `worley_fbm3(u, 4, ..)`, so the base shape has **4 cells per tile**. The cumulus tile is `1.33 × scale_km` (4 km at the default 3 km), so every heap is about 1 km. Coverage only moves the threshold (`CloudModel`), so there are more or fewer clouds **of the same size**, never bigger ones. |
| Identical tops | `height_profile` for cumulus sets `top = 0.3 + 0.7·cover`, where cover is the local coverage, itself the layer mean ± `WEATHER_SWING`·(weather − 0.5). Neighbouring cells get almost the same top. |
| A regular, dense band at the horizon | (a) Periodicity: a 4 km tile at 100 km spans 2.3°, which is **about 13 map texels** of azimuth (0.176° each), so the pattern visibly repeats along the band. The 40 km weather tile repeats every ~23° there. (b) `textureSampleLevel(weather_tex, ..., 0.0)` and the base-noise reads use the top mip at every distance. At 100 km a texel footprint covers several noise cells, so the reads alias into a regular high-frequency pattern instead of averaging to the true mean. (c) A grazing ray marches up to `MAX_MARCH_M` = 14 km of layer, crossing 3-4 tiles. Optical depth saturates, so the band becomes an opaque, even wall. (d) `HAZE_PER_M` = 2e-5 leaves 13.5% transmittance at 100 km and 5% at 150 km. Bright clouds keep contrast there, so the band is not hidden (§5). |
| Nothing forms, grows, merges or dissipates | `layers_at`: shape evolution is a scroll (`offset` x/z by wind) plus a vertical "boil" (`offset[1]`, 0.05-0.25 m/s × `evolution`). Coverage per kind wanders with `wander()` over 12-25 min periods, layer-wide. There is no per-cloud state. |
| No order, but also no organisation | There are no streets, no cells and no clustering. The noise is statistically uniform at every scale above 1 km. |

Perspective makes all of this worse. A flat layer seen at elevation e covers
`1/sin³e` more area per steradian than at the zenith: 8x at 30°, 191x at 10°,
1,510x at 5°. Any statistically uniform field therefore **must** pile up
toward the horizon (see §5).

---

## 1. How shipped games and simulators make clouds evolve

Evidence quality is marked: **[P]** primary source read, **[A]** primary
abstract or session description only, **[S]** secondary or forum,
**[R]** recollection not re-verified.

### 1.1 What is actually documented

- **Guerrilla / Nubis.**
  - The 2015 prototype "ran using a procedural weather simulation". The team
    judged it lacked art-directable control, and Nubis 2017 replaced it with
    regionally authored 2D fields plus "animation and transitions" [A, G1].
  - Nubis Evolved (2022) still models clouds by mapping 2D data fields
    (coverage, type) over world xz and expanding them vertically at render
    time. A "dimensional profile" is generated from a **signed-distance
    field of the cloud**, so each cloud has an outside-to-inside gradient
    that drives erosion and approximates AO. The definition costs about
    2 ms [A/S, G2].
  - Nubis³ (2023) moved to voxel clouds with "fluid simulation-based
    modeling", compressed SDFs for skipping, and ~0.2-5 ms by view distance
    [A, G3].
  - The HFW superstorms were done "without expensive simulations" [A, G4].
  - **Lesson:** even Guerrilla does not simulate lifecycles at runtime. The
    shape authority is a 2D field with an inside-outside profile. That is
    the piece we lack.
- **Sucker Punch, Ghost of Yōtei** (PlayStation Blog, 23 Oct 2025): depth-aware
  cloud maps for parallax scrolling [P, games doc S10]. No SIGGRAPH or GDC
  2026 cloud talk from Sucker Punch was found.
- **Microsoft Flight Simulator 2020/2024.**
  - Weather comes from Meteoblue: about 250 M grid boxes with 60 vertical
    layers, sampled rather than streamed continuously [S, G8].
  - No Asobo technical talk on how fields become clouds was found.
  - Forum reports describe low-resolution, translucent distant clouds and
    aggressive cloud LOD [S].
  - **Lesson:** feeding real NWP data does not by itself give believable
    cumulus. A 1-3 km NWP grid cannot resolve individual cumulus, which must
    still be synthesised.
- **trueSKY (Simul):** evolution is interpolation between sky and cloud
  keyframes (cloudiness, base, height, density, precipitation). Editing the
  keyframe currently being interpolated causes jumps [P docs, G9].
- **Industry norm (survey, secondary):** 4-12 weather states, linear
  transitions over 30-120 s [S, G15]. This is exactly the "hard-set" feel to
  avoid.
- **Sea of Thieves (Rare, SIGGRAPH 2018 talk):** "art-directed cloudscapes
  without expensive raymarching" [A, G6]. Clouds are 3D objects that can
  morph into shapes (skulls, ships); a secondary report gives about 850
  vertices each [S]. Storms are gameplay **entities** that move.
  - **Lesson:** cloud and storm as objects is a proven way to get "this cloud
    is going somewhere".
- **Flight Simulator 2004 (Wang, JGT 9(3), 2004):** clouds as authored
  **objects** (boxes filled with textured sprite puffs), a dozen types from
  stratus to cumulonimbus [A, G7]. This is the classic object-based model,
  and the ancestor of what we propose, minus the sprites.
- **Borderlands 4 (GDC, 13 Mar 2026):**
  - "Demystifying the Clouds of Borderlands 4" (Disbelief) covers adapting
    volumetric cloud techniques to a UE5 open world.
  - A companion talk covers artist-directed "weather forecasts" for an
    open-world weather system [A, G10].
  - Contents beyond the descriptions were not found. This is the most recent
    talk to check in the GDC Vault.
- **Star Citizen:** volumetric "cloud tech" first shipped in 3.14 (Crusader,
  2021). CitizenCon 2953 showed cloud improvements. The procedural weather
  system was announced at CitizenCon 2019; no technical detail on cloud
  evolution was found [S, G11].
- **Just Cause 4 (Apex, 2018):** physics-driven extreme weather (tornadoes,
  storms) with volumetric clouds. No technical talk was found [S, G12].
- **Not found:** Avatar: Frontiers of Pandora (Snowdrop) cloud talk (the GDC
  2024 Snowdrop talks are on audio and world scale), Forza Horizon 5/6 cloud
  internals, Death Stranding 2 cloud internals, and UE 5.4-5.6 release-note
  cloud changes. They are recorded here so nobody re-searches them [S].

### 1.2 The three ways lifecycles can be represented

| Approach | Who | Evolution quality | Headset cost | Fit |
|---|---|---|---|---|
| **Time-varying noise** (scroll, 4D noise, keyframed coverage) | Nubis 2015-22, HDRP, UE, trueSKY, **ours** | Morphing; no identity, no birth or death; single scale unless several are authored | Lowest | Already have it; it is the problem |
| **Cloud objects (particles or cells) splatted into a 2D control field** | FS2004 (sprites), Sea of Thieves (meshes), Nubis's SDF profile (per-cloud field), Goswami & Neyret 2017 (parcels + noise) | Real identities: born, grow, merge, decay, drift; sizes from any distribution | 2D splat (tiny) + same march | **Recommended** |
| **Fluid or thermodynamic simulation** (stable fluids + buoyancy + phase change) | Harris 2003, Dobashi 2000 (cellular automaton), Stormscapes 2020, Weatherscapes 2021, Thunderscapes 2024 | Physically emergent | 3D grids; GPU solvers at "interactive rates" on 10-20 km desktop domains | Server only, and only as a coarse 2D weather field (§3) |

**Measured costs for the simulation family:**
- Harris et al. 2003 (GeForce FX era, read from the paper): a 64³ grid ran at
  under 4 iterations/s. It was **amortised over frames**, viewed at
  40-80 fps while the simulation updated 1-5 times a second [P, R1].
- Stormscapes: cumulus, stratus, stratocumulus and cumulonimbus at
  interactive rates for about 20 × 20 km, up to 13 km high [A, R4].
- Weatherscapes adds ground water and heat exchange, and nowcasts from live
  weather data [A, R5].
- None publishes a per-step ms figure in what could be retrieved.
- Dobashi 2000 is the cheapest physical-ish model: Boolean cellular automaton
  bits `hum`/`act`/`cld` per cell, probabilistic vapour, activation and
  extinction, wind advection [A, R2].
- Goswami & Neyret 2017: coarse **Lagrangian air parcels** (hydrostatic
  atmosphere, thermodynamics, entrainment-like mixing), amplified by
  volumetric noise. Real time at landscape scale, and validated against
  skew-T diagrams [A, R3]. This is the closest research analogue to our
  object model.

---

## 2. Real cloud-field statistics for a believable generator

Numbers to drive the generator. Those I could not pin to a primary number are
flagged.

### 2.1 Sizes

- **Area distribution, global:**
  - n(a) = dn/da ∝ a^-(1+α), with **α = 0.95 ± 0.08** from satellites, over
    at least **3 to 3×10⁵ km²**;
  - "remarkably robust to latitude, season, and land–ocean contrasts";
  - perimeter exponent β = 1.26 ± 0.06 (DeWitt et al., ACP 24:109, Jan 2024)
    [P, C2].
  - Apparent scale breaks are often artefacts of finite domains and
    truncated edge clouds (DeWitt & Garrett, ACP 24:8457, 2024) [A, C3].
- **Chord lengths:**
  - n(L) ∝ L^-1.66±0.04 over 0.1-1,500 km, with fewer clouds above
    1,500 km;
  - clouds ≥ 200 km give about 50% of global cover;
  - clouds < 10 km give about 15%;
  - broken stratocumulus share almost the same exponent as cumulus (Wood &
    Field, J. Climate 2011; definitions read from the paper) [P, C1].
- **Shallow cumulus in LES:** a power law with exponent about −1.7 below a
  **scale break**. Normalising by the break collapses all cases onto one
  curve, so **the break size is the one free parameter**, and it sets the
  "dominant" cloud size for cover and mass flux (Neggers et al., JAS 2003)
  [A, C4]. Spatial organisation raises the maximum size (Neggers et al., JAS
  2019) [A, C5].
- **Usable rule:**
  - Draw cumulus diameters from **n(D) ∝ D^-3**, which is area α = 1. This
    makes **every octave of size contribute equal cover**: each doubling of
    diameter has 1/4 as many clouds and the same total area.
  - Truncate at D_min of about 0.2-0.3 km (below which the map cannot show
    them) and at a break D_max of about 2-5 × the boundary-layer depth for
    unorganised fair-weather cumulus. That break is my reading of the LES
    collapse; **unverified as a ratio**.
  - Raise D_max in organised or merging regimes.
- **Counts this implies** (computed):

| Cover | D range | Clouds per km² | In a 40 km radius | With D ≥ 1 km |
|---|---|---|---|---|
| 0.30 | 0.2-5 km | 1.48 | ~7,400 | ~290 |
| 0.30 | 0.3-4 km | 0.82 | ~4,100 | ~350 |
| 0.15 | 0.2-3 km | 0.88 | ~4,400 | ~160 |

The few hundred large objects are worth streaming; the thousands of small
ones should be spawned deterministically on the client (§3.3).

### 2.2 Lifetimes and growth

- **Shallow marine cumulus, about 250,000 tracked** (MSG-SEVIRI, Canaries,
  August 2015): the lifetime distribution is a **double power law, with most
  clouds living tens of minutes** (Seelig et al., JGR 2021) [A, C6].
- **GOES-16, 1-minute sampling, Caribbean:** confirms the double power law.
  Clouds **live longer when their area exceeds tens of km²**, and optically
  denser ones live longer (Seelig, Müller, Tesche, GRL 2023) [A, C7].
- The exact exponents and break time were not retrievable. **Flag:** take
  them from the papers' figures before tuning. A reasonable starting point is
  a break near 20-30 min, a median around 15-20 min for D < 1 km, and 30-90
  min for D of several km (my estimate).
- **LES tracking** (Dawe & Austin, ACP 2012, BOMEX; Heus & Seifert, GMD 2013,
  with splitting and merging tracked):
  - cloud-base area correlates well with maximum height and upper-level
    area, so **bigger base means taller cloud**;
  - pre-moistened air does not make a cloud more likely to reach the
    inversion [A, C8, C9].
- **Vertical speeds:**
  - fair-weather cumulus updrafts average about 0.5-0.7 m/s, with large
    thermals over 600 m reaching about 1.5 m/s (lidar, Leipzig) [A, C18];
  - the in-cloud mean in marine fair-weather cumulus is about 0.3 m/s (ARM
    Graciosa) [A].
  - **Usable:** a cumulus top rises at about 0.5-2 m/s while growing. A
    500 m deep humilis forms in about 5-15 min; congestus keeps rising at
    2-5 m/s (estimate). Decay (evaporation from the edges, more fibrous and
    then gone) takes about 5-10 min (estimate, consistent with the "tens of
    minutes" totals).
- **Thunderstorm cells:** the mature stage lasts about 15-30 min per cell
  [S, FAA study guide, flag]. Multicell clusters persist for hours by
  spawning new cells on the gust front.
- **Diurnal cycle over land:** shallow cumulus form when the morning mixed
  layer reaches the lifting condensation level. The base rises through the
  afternoon with the boundary layer, and the clouds decay toward evening.
  "Formation and dissipation are tightly coupled with boundary layer
  development" (ARM SGP composites of 76 days) [A]. Exact onset times and
  rise rates are not retrieved; **flag**.

### 2.3 Spatial organisation (placement, clustering, regularity)

- **Clustering versus regularity** (Zhu et al. 1992): with small and large
  clouds included, fields **always cluster**, most strongly at 200-300 m
  diameters. With small clouds removed they tend to regularity [S, C10].
  - 98% of clouds lie within 300 m of a neighbour at 15 m resolution.
  - 75% have a neighbour within 10 equivalent radii [S].
  - Nearest-neighbour spacing grows with cloud size, logarithmically by the
    classic definition (van Laar & Neggers 2021) [A, C11].
  - **Usable rule:** small clouds are born near existing clouds (satellites
    round a big one, and on outflow edges); large clouds keep apart.
- **Convective clustering at about 100 km** in midlatitude summer (Rasp,
  Selz, Craig 2018) [S, C16].
- **Equilibrium theory** (Craig & Cohen 2006): cloud mass fluxes are
  exponentially distributed and the clouds are **randomly located**
  (Poisson) when forcing is uniform [A, C16].
  - The clustering above comes from non-uniform forcing and from cold
    pools. This tells us how to place clouds: **a Poisson process whose rate
    is the weather field, plus neighbour seeding**. No grids and no blue
    noise.
- **Population dynamics:** STOMP (Hagos et al., JAMES 2018) is a master
  equation over cell size, with probabilities of growth and decay and
  mass flux tied to area. It shows aggregation and recharge-discharge under
  steady forcing [A, C17]. It is a ready-made formal model for our birth and
  decay rules.
- **Trade-wind mesoscale patterns** "sugar, gravel, fish, flowers"
  (Stevens/Bony et al. 2020, EUREC4A):
  - sugar: weak winds;
  - gravel: stronger winds and subsidence, with cold pools;
  - flowers: more stability;
  - fish: convergence lines from synoptic disturbances [A, C12].
  - These are good named presets for the weather model.
- **Cloud streets (horizontal rolls):**
  - wavelength about **3× the boundary-layer depth** (AMS Glossary), with an
    observed range of 2:1-6:1 and up to 10:1;
  - wide modes reach 30 km wavelength and an aspect ratio of 10-20;
  - aligned **within 10° of the mean wind most often, and within 30°
    nearly always** [A, C14; Etling & Brown 1993 not read directly].
  - They occur with moderate wind and shear in a convective boundary layer,
    and in cold-air outbreaks.
  - For a 1 km BL that means streets about 2-4 km apart.
- **Stratocumulus mesoscale cells:**
  - cell diameter / BL depth = **30-40**, so closed cells are about 10-60 km
    across, larger in deeper boundary layers;
  - LWP variance is dominated by 10-50 km scales;
  - LES puts the maximum stable aspect ratio at 32-64 (Wood & Hartmann 2006;
    later reviews) [A, C13].
  - Open cells have clear centres ringed by cloud; closed cells are the
    inverse.
  - For a 1 km BL that means **30-40 km cells**, which is why one 2-5 km
    noise tile can never look like a stratocumulus deck.
- **Element sizes by genus** (WMO International Cloud Atlas; apparent width
  of most elements when seen more than 30° above the horizon) [P, C15]:
  - cirrocumulus under 1°;
  - **altocumulus 1-5°**: at 3-5 km above the eye and 60° elevation, about
    **60-500 m elements**;
  - **stratocumulus over 5°**: over 175 m at a 1 km base and 30° elevation,
    in practice 0.3-2 km rolls and lumps.
  - Altocumulus lie in patches and sheets with Bénard-like cells (rising
    centre, clear sinking edges, aspect ratio about 3:1) [A].

### 2.4 Winds, fronts, layers

- **Wind by altitude** (midlatitudes; textbook values, not re-verified):
  - boundary layer and 850 hPa: about 5-10 m/s;
  - 500 hPa (about 5.5 km): about 15-25 m/s;
  - 250 hPa jet: 30-50 m/s typical, winter cores over 50 and events over
    70 m/s (WMO jet threshold 30 m/s) [S].
  - Direction veers (Ekman turning and thermal wind), so upper clouds cross
    lower ones at an angle. Ours already uses shares [0.8, 1, 1.7, 3.2] ×
    8 m/s and turns of +12°/0/−25°/+38°, which is plausible.
- **Fronts:**
  - cold fronts move at about 25-30 mph (11-13 m/s), up to about 60 mph;
  - warm fronts are slower (about 10-15 kt, 5-8 m/s; low-quality source,
    **flag**) [S, C19].
  - **Ahead of a warm front:** cirrus, then cirrostratus, then altostratus
    over hundreds of km, lowering to nimbostratus and rain [S, UCAR]. The
    often-quoted ~1,000 km cirrus lead was **not verified**; ~800 km appears
    in study material.
  - **Cold front:** a line of cumulonimbus and cumulus, then clearing, then
    post-frontal open-cell cumulus.
- **Usable timescales for a play session:**
  - a front crossing a 60 km view takes about 1.5-3 h, so it can be
    time-accelerated;
  - a cumulus field "breathes" on 10-30 min;
  - stratocumulus cells change over hours;
  - cirrus drift visibly at 30-50 m/s.

---

## 3. Cheap simulation, split between server and headset

The rule ([[server-may-carry-headset-work]]): the server carries the
simulation, the wire field is an `Option`, and with `None` the client runs the
same simulation from the scene seed. Never both.

### 3.1 Weather field (server, 2D, coarse)

- **Grid:** 128 × 128 cells over 256 km (2 km a cell), centred on the play
  area and scrolled with it.
- **Per cell:**
  - low-level moisture q;
  - an instability proxy (CAPE-like; surface heating × moisture, from time
    of day, land/sea and terrain);
  - large-scale lift ω (fronts, convergence lines, orography);
  - inversion (BL) height;
  - upper moisture (for cirrus and altostratus);
  - precipitation;
  - wind at three levels (BL, about 600 hPa, about 250 hPa).
- **Dynamics:**
  - semi-Lagrangian advection of the scalars by their level's wind (Stam's
    stable-fluids advection step; no pressure solve is needed for passive
    scalars);
  - relaxation sources and sinks (diurnal heating, rain-out, subsidence
    drying);
  - **fronts as moving parametric features** (a polyline with a speed and a
    cross-front profile of lift, moisture and wind shift) stamped into ω and
    q.
  - Synoptic wind comes from a few keyframed or seeded "pressure systems",
    not a solved momentum equation.
- **Cost:** 16 k cells × about 10 fields × a few dozen flops at 1 Hz, which
  is **well under 1 ms of desktop CPU per step** (estimate). A Harris-style
  3D solver is not needed for this.
- **Outputs per layer:**
  - coverage target;
  - type mix (Cu/Sc/Ac/As/Ns/Ci/Cb);
  - base and top;
  - birth rate for convective objects;
  - organisation regime (streets, open or closed cells, clustered,
    unorganised);
  - precipitation.

### 3.2 Cloud-object population (server)

- **State per object, about 16 B:**
  - position (2 × u16 over 64 km, so 1 m);
  - diameter (u8, log scale);
  - top (u8);
  - phase (u8);
  - growth rate (i8);
  - type (u4);
  - organisation tag (u4);
  - seed (u16);
  - birth time (u16, s).
- **Birth:** a Poisson process with rate = f(instability, moisture, lift)
  from §3.1, so it is random where forcing is uniform (Craig & Cohen), plus:
  - **neighbour seeding:** births near young clouds, and satellites round
    large ones, which reproduces the 200-300 m clustering;
  - **street modulation:** in a streets regime, the rate times
    `1 + k·cos(2π·across/λ)`, with λ = 3 × BL depth and the axis within
    ±10° of the BL wind, plus a random phase that drifts over an hour;
  - **open/closed cells for Sc:** a Voronoi lattice of seeds at 30-40 × BL
    depth spacing, with rate or thickness high on cell edges (open) or in
    cell centres (closed), and the lattice advected and slowly re-seeded.
- **Size:** D drawn from n(D) ∝ D^-3 on [D_min, D_max(regime)].
- **Lifetime:** drawn from the double power law (tune to C6/C7), lengthened
  with D.
- **Growth:** the top rises at 0.5-2 m/s, capped by the inversion or, for
  Cb, by the tropopause.
- **Mature phase.**
- **Decay:** coverage and density fall over 5-10 min, and the edges go wispy
  (phase drives erosion).
- **Merge:** when two objects overlap by more than about 30%, replace them
  with one of the summed area. Its top is the maximum plus a boost (merged
  clouds grow taller), and its lifetime is extended. Merging is how D_max
  grows in organised regimes.
- **Cumulonimbus:** if the top passes the freezing level and the
  instability is high, it becomes Cb. An anvil object spreads downwind at
  the upper-level wind, a precipitation flag feeds `weather.rs` areas, and
  lightning events follow (games doc R16).
- **Motion:** advected with the BL wind plus a small per-object drift.
  Large Cb drift with the mean wind of the cloud layer, and cells propagate
  on their gust fronts.
- **Cost:**
  - **Server:** a few thousand objects at 1 Hz is negligible.
  - **Headset CPU:** decoding and advancing a few hundred objects takes
    under 0.05 ms a frame.

### 3.3 Determinism and wire budget

- **Server sends:**
  - large objects (D ≥ about 1 km, 150-350 within 40 km): a snapshot of
    about 3-6 KB on join, then events (birth, merge, death and rate changes),
    about 10-50 B/s;
  - the weather grid: 128² × 8 B = 128 KB on join, then deltas of about
    1-2 KB/s (refreshed every 30-60 s and interpolated);
  - front and pressure-system descriptors: under 1 KB.
- **The client derives:**
  - the thousands of small cumulus, deterministically from each large
    object's seed and each weather cell's seed. Same seeds give the same
    sky for every player, with no bytes for them.
  - **Between messages the client extrapolates:** objects advance by the
    shared rules and wind, so a late message corrects a little and never
    pops (the correction is blended over 10-30 s).
- **Fallback (`None`):** the client runs §3.1-3.2 locally at 0.2-1 Hz on a
  worker thread, from the scene seed.

### 3.4 Rejected for now

- **A 3D fluid or thermodynamic cloud simulation** (Stormscapes or
  Weatherscapes class) on the server, streaming 3D density: the bandwidth
  (even 128² × 32 × 1 B is 0.5 MB a frame) and the client's 3D textures do
  not fit. Keep it as an **offline authoring tool** for hero storms
  (Nubis³-style voxel assets; a 128³ R8 cloud is 2 MB) if ever needed.
- **NWP (Meteoblue-style) data as the shape source:** too coarse for
  cumulus. As MSFS shows, it still needs a synthesiser. It could seed §3.1
  later.

---

## 4. Rendering it on the Quest 3, cheaper than today

All costs are estimates by fetch count (games doc §3). None are measured.

### 4.1 Envelope maps (the Nubis "dimensional profile", from objects)

- **The map:** per volumetric layer, a 2D RGBA8 map in the **layer's wind
  frame** (so advection is a uniform offset and does not force a re-splat).
  - **Cu:** 512² over 64 km, 125 m a texel, 1 MB.
  - **Sc and Ac:** 256²-512² over 64-128 km.
- **Channels:**
  - **R** = profile, 0 at the edge and 1 at the core: the union of the
    objects' analytic SDFs (an ellipse plus a lumpy radius from the seed);
  - **G** = local top;
  - **B** = the object's noise offset or seed;
  - **A** = life phase (growth billowy, decay wispy).
- **Build:** splat object quads on the GPU, or on a CPU worker and upload.
  Re-splat 1/8 of the map a frame, or only on events, about 0.02-0.05 ms.
  Mips are built on update.
- **In the march:** this replaces the `weather_tex` fetch, so **0 extra
  fetches**:
  - the density threshold becomes the profile;
  - `height_profile` uses G as the top, so tops vary per cloud;
  - the base noise is read at `p·scale + hash(B)·tile`;
  - the detail erosion strength comes from A.
- **What it changes:**
  - every cloud indexes the noise differently, so **the 4 km period
    disappears**;
  - neighbours differ;
  - sizes follow the object distribution, not the noise cell size;
  - the noise now only sculpts the edges and interior of an envelope.
  - This is the 2D half of Nubis Evolved's SDF profile.

### 4.2 Empty-space skipping from the same map

- A max-mip of R and G, or a distance-to-cloud channel, lets coarse steps of
  0.5-2 km test one 2D texel and enter fine stepping only inside an
  envelope.
- This is games doc R5, now with a better field (objects, not noise
  thresholds).
- **Cost:** −20-40% of march fetches in typical 20-50% cover (estimate).

### 4.3 Distance LOD that matches the horizon

The map packs about 80 rows into elevations 0-3°, which for cumulus at 1.8 km
is 33-150 km away. Three bands by distance along the ray:
- **Near (under about 25 km):** full density (base and detail noise,
  envelope).
- **Mid (25-60 km):** base noise only, with the step tied to the texel
  footprint. Sample the envelope and noise at **the mip matching the
  footprint** (today both read LOD 0 everywhere, which aliases).
- **Far (over 60 km, about el < 2°):** **no 3D noise.** Integrate the
  envelope's prefiltered mip (coverage × mean thickness) analytically
  through the layer. Beyond the envelope map's 32 km half-width, use the
  server's 2 km weather grid. The far band is then the **real mean coverage
  of the real weather in that azimuth**: it varies around the horizon, has
  gaps where it is clear, and has no tile period.
- **Cost:** −10-25% of march time (the horizon rows are the longest rays),
  with fewer fetches per far sample.
- **Nubis³ precedent:** a 0.2-5 ms spread by view distance, and the survey
  reports half cost at the same look from per-altitude quality levels
  [A/S].

### 4.4 Aerial perspective that hides the LOD

- Replace grey `exp(-t·HAZE)` with per-channel extinction and in-scatter
  along the path. Add a **boundary-layer aerosol** term: visibility V from
  the weather grid (20-60 km), σ_surface = 3.91/V (Koschmieder), with a
  scale height of about 1.2 km.
- Along a grazing path to 1.8 km clouds, mean σ of about 3-5e-5 /m gives
  T = 0.05 at 60-100 km.
- Far clouds then fade into a warm haze band, as in real photographs, which
  also hides the far-band LOD.
- This extends games doc R10, and is ~0 cost: ALU per segment, plus LUT
  inputs from the CPU.

### 4.5 Temporal and cache behaviour

- **Advect and accumulate in map space** (games doc R1 + R8). Cloud
  lifecycles take minutes, so once motion is advected the full rebuild can
  stretch to 96-192 frames (1.3-2.7 s) without missing an event. That is
  **−0.2 to −0.7 ms of march** (R1 estimate).
- **Jitter coherently.** Toft et al. 2016 (GTX 1080, half resolution, 8
  steps) measured **+5.2 ms from a per-pixel random ray offset**, which they
  attributed to texture-cache misses (2.3 ms without jitter, 7.5 ms with)
  [P, R8].
  - On Adreno, use a per-rebuild offset shared by a row or tile (blue noise
    with small tiles, or golden-ratio offsets per rebuild), not white noise
    per texel.
  - Neighbouring map texels are neighbouring directions, so they stay
    coherent.

### 4.6 Optional, later: let the server render the far band

- **Precedent:** "Clouds in the Cloud" (Weinrauch et al., HPG 2023) caches
  cloud density and lighting on **spheres per cloud layer**, computes them
  remotely and shares them across viewers. Viewers "hundreds of meters
  apart" hardly perceived the sharing, and the caches can be low resolution
  because clouds are smooth [A, R7].
- **For us:** the server would march the el < 3° band (2048 × 80 texels)
  every 10-20 s and stream it, at about 1.3 MB raw RGBA16F, or roughly
  50-150 KB per update compressed (estimate). The headset then **stops
  marching the horizon entirely**.
- **Open questions:** the bandwidth budget, latency at sunrise and sunset
  (light changes quickly), and whether ASTC HDR is available as a
  compression target on Adreno 740 (**not verified**).

### 4.7 Mobile and VR evidence (new since the games doc)

- **Still no published Quest title** with ray-marched volumetric clouds was
  found.
- W4 Games and Meta (July 2026) list sky-rendering optimisation among their
  Godot standalone-XR changes, without a cloud method.
- Oculus Android guidance (quoted secondhand) warns against expensive default
  skyboxes on standalone headsets.
- The fixed-cost, direction-space cache stays the right structure for the
  headset. The work above keeps the display at today's tap count.

---

## 5. Why naive skies "multiply toward the horizon", and how leaders avoid it

1. **Geometry.**
   - A flat layer at height h, seen at elevation e, puts `h²/sin³e` of
     layer area into each steradian (191x the zenith at 10°, 1,510x at 5°).
   - On a curved Earth the layer meets the horizon at 151 km for 1.8 km
     cumulus and 329 km for 8.5 km cirrus.
   - Any field with uniform statistics turns into a dense, finely repeating
     band. Real skies have it too; what differs is everything below.
2. **Haze.** In the boundary layer, visibility of 20-60 km means real horizon
   cumulus at 80-150 km sit at a few percent contrast, tinted warm.
   - Ours (2e-5 /m, grey) leaves 13.5% at 100 km.
   - Hillaire/UE aerial perspective (per channel, LUT) is what leaders
     apply to clouds (games doc R10). Add the aerosol term (§4.4).
3. **Real weather is not uniform at 100 km scales.**
   - Cloud areas are scale invariant up to 3×10⁵ km².
   - Convection clusters at about 100 km.
   - Stratocumulus cells are 10-60 km.
   - So the horizon of a real sky has banks, gaps and regional changes in
     type. Ours repeats a 40 km weather tile and a 4 km shape tile.
   - The fix is non-periodic weather (§3) and per-object noise offsets
     (§4.1).
4. **Aliasing.** Sampling detailed 3D noise at a footprint wider than its
   cells produces regular moiré, not the mean.
   - Leaders reduce detail with distance (Nubis's cheap and full samples,
     HZD's quarter resolution, UE's quarter-resolution trace with
     reconstruction) and Tsushima fades density by derivatives (we have
     footprint AA for detail).
   - Use mips matched to the footprint and the far band from prefiltered
     coverage (§4.3).
5. **Sides, not tops.** Near the horizon you see cloud sides stacked into a
   bank. That reads as real only if the bank's height and brightness vary
   along azimuth: different tops per cloud (§4.1 G channel) and shading by
   aerial perspective.
6. **Clipping distance.** UE exposes a maximum tracing distance, and the
   prior survey reports layered systems blending into a precomputed distant
   sky past a range [S]. Our `MAX_MARCH_M` 14 km per layer is fine **if**
   beyond it we switch to the analytic far band rather than letting
   saturated grazing rays build a wall.

---

## 6. Recommended architecture (in order; cost is headset GPU, net against today)

1. **A server weather field, 2D and coarse.**
   - 128² cells over 256 km (2 km a cell).
   - Fields: moisture, instability, lift, BL height, upper moisture,
     precipitation, and 3-level wind.
   - Semi-Lagrangian advection; fronts and pressure systems as moving
     parametric features; 1 Hz.
   - On the wire: an `Option` with a 128 KB join snapshot and 1-2 KB/s of
     deltas. The client falls back to the same sim from the seed.
   - Weather changes **by rates, never by state lerps**. It also feeds rain
     areas, lightning, particle wind and ambient.
   - **Cost: 0 GPU**; headset CPU 0 (server) or ~0.1 ms at 1 Hz (fallback).
2. **A cloud-object population model on the server.**
   - Poisson births from the weather rate, plus neighbour seeding, street
     modulation (λ = 3 × BL depth, within ±10° of the wind) and Sc
     open/closed Voronoi cells (30-40 × BL depth).
   - D from n(D) ∝ D^-3 (equal cover per octave).
   - Lifetimes from the observed double power law, with bigger clouds
     living longer.
   - Growth at 0.5-2 m/s, merge into bigger and taller clouds, decay over
     5-10 min, Cb with anvils and rain.
   - Stream D ≥ 1 km objects (about 3-6 KB, plus tens of B/s); the client
     derives the small ones from the seeds.
   - **Cost: 0 GPU; under 0.05 ms headset CPU.**
3. **Per-layer envelope maps splatted from the objects.**
   - RGBA8, 512² over 64 km for cumulus, in the wind frame.
   - Channels: profile, local top, noise offset and life phase.
   - They replace the weather texture fetch.
   - **Cost: +0.02-0.05 ms amortised build, 0 extra fetches in the march.**
4. **Density = envelope × per-object-offset noise erosion.**
   - Per-cloud tops (G), erosion style by phase (A), and its own detail
     noise (games doc R13).
   - Altocumulus elements of 60-500 m in patches; cirrus streaks along the
     upper wind and shear.
   - This removes the tile period and the one-size carpet.
   - **Cost: ~0** (ALU only).
5. **Empty-space skipping from the envelope mips** (games doc R5 on a better
   field). **Cost: −20 to −40% of march fetches.**
6. **Three distance bands.**
   - Full detail under 25 km.
   - Base-only, footprint-matched mips at 25-60 km.
   - An analytic prefiltered-coverage far band beyond 60 km (envelope mip,
     then the 2 km weather grid).
   - No LOD-0 reads at distance.
   - **Cost: −10 to −25% of march.**
7. **Aerial perspective on clouds with boundary-layer aerosol.**
   - Visibility from the weather, about a 1.2 km scale height, per channel
     (games doc R10 plus aerosol).
   - Horizon banks fade to warm haze, which hides band transitions.
   - **Cost: ~0.**
8. **Advected map-space accumulation with coherent jitter** (games doc
   R1/R8).
   - Rebuild period of 96-192 frames.
   - Per-row or per-tile jitter rather than per-texel white noise (Toft:
     +5.2 ms from per-pixel jitter on a GTX 1080).
   - **Cost: −0.2 to −0.7 ms.**
9. **Statistical guards in the CPU twin** (tests, no runtime cost). For a
   seeded sky, measure:
   - the size-distribution slope (expect area α ≈ 1);
   - a nearest-neighbour clustering index;
   - the lifetime histogram;
   - the coverage autocorrelation;
   - **no periodicity in the horizon row** at the noise or weather period;
   - a far-band mean that matches the near-field mean coverage.
   Then compare offline renders with reference sky photos (memory
   `cycles-reference-harness`). **Cost: 0.**
10. **Later, optional: a server-rendered far band** ("Clouds in the Cloud").
    Stream the el < 3° band every 10-20 s and stop marching the horizon on
    the headset.
    - **Cost: −(the far-band share of the march)**, at about 5-15 KB/s of
      bandwidth. Needs a network budget and a check of HDR texture
      compression.

**Net estimate for items 1-8:** about −0.2 to −0.9 ms on the headset against
today. Item 3 is the only new GPU pass, and it is tiny. The display shader is
unchanged. **Measure first** (games doc R0 timers: `cloud_sky_map`,
`cloud_sky_shadow`, the display) so these ranges become numbers.

**Items 2-4 are what answer the user's complaint**: motion, formation,
merging, dissipation, variety and no tiling. Items 5-8 pay for them.

---

## 7. What could not be verified

- **Lifetime exponents and break times:**
  - Seelig et al. 2021 and 2023 (abstracts only: "double power law", "tens of
    minutes", "longer above tens of km²");
  - Dawe & Austin 2012 and Heus & Seifert 2013 lifetime histograms (no
    numbers retrieved).
- **D_max ≈ 2-5 × BL depth for unorganised cumulus** is my reading of the
  Neggers 2003 scale-break collapse, not a quoted value.
- **Nubis talks:**
  - Nubis 2017, Nubis Evolved 2022 and Nubis³ 2023 PDFs exceed the fetch
    limit. The 2015 procedural weather simulation, the 2D fields expanded
    vertically, the SDF-derived dimensional profile, about 2 ms, and
    0.2-5 ms come from abstracts and search snippets.
  - How HZD's weather changes over time is **not confirmed**.
- **No primary source found for:**
  - MSFS internals (Meteoblue grid figures are from PC Gamer coverage);
  - Death Stranding 2, Forza Horizon 5/6, Avatar FoP, Just Cause 4 and Star
    Citizen cloud evolution;
  - UE 5.4-5.6 cloud changes;
  - a Sucker Punch 2026 cloud talk.
  - Borderlands 4's GDC 2026 talks are known only by their descriptions.
- **Textbook values** without a retrieved source: the wind climatology by
  level, front speeds (secondary or low quality), cumulus growth and decay
  stage times (my estimates), and thunderstorm stage durations (exam-prep
  guide).
- **Other gaps:**
  - Stormscapes, Weatherscapes and Goswami & Neyret timings were not
    retrieved; "interactive rates" only.
  - Whether ASTC HDR is available on the Adreno 740 under our wgpu path.
  - Every headset cost here is a fetch-count estimate, not a measurement.

---

## 8. Sources (dates as published; web pages accessed 2026-10-20)

Meteorology
- [C1] Wood, R., Field, P. R., "The Distribution of Cloud Horizontal Sizes", J. Climate 24 (2011). PDF: https://atmos.washington.edu/~robwood/papers/my_papers/Wood_Field_JCLI_2011.pdf (text extracted and read for definitions; exponent and cover fractions from the abstract)
- [C2] DeWitt, T. D., Garrett, T. J., et al., "Climatologically invariant scale invariance seen in distributions of cloud horizontal sizes", ACP 24, 109-122 (Jan 2024). https://acp.copernicus.org/articles/24/109/2024/acp-24-109-2024.html (read)
- [C3] DeWitt, Garrett, "Finite domains cause bias in measured and modeled distributions of cloud sizes", ACP 24, 8457 (2024). https://acp.copernicus.org/articles/24/8457/2024/
- [C4] Neggers, Jonker, Siebesma, "Size statistics of cumulus cloud populations in large-eddy simulations", JAS 60 (2003). https://cdn.knmi.nl/system/data_center_publications/files/000/066/549/original/neggers_03_jas.pdf
- [C5] Neggers, Griewank, Heus, "Power-law scaling in the internal variability of cumulus cloud size distributions due to subsampling and spatial organization", JAS (2019), doi:10.1175/JAS-D-18-0194.1
- [C6] Seelig, Deneke, Quaas, Tesche, "Life Cycle of Shallow Marine Cumulus Clouds From Geostationary Satellite Observations", JGR Atmospheres 126(22) (2021), doi:10.1029/2021JD035577. https://e-docs.geo-leo.de/handle/11858/9771
- [C7] Seelig, Müller, Tesche, "Do Optically Denser Trade-Wind Cumuli Live Longer?", GRL (Jul 2023). https://e-docs.geo-leo.de/handle/11858/10849
- [C8] Dawe, Austin, "Statistical analysis of an LES shallow cumulus cloud ensemble using a cloud tracking algorithm", ACP 12, 1101-1119 (2012). https://acp.copernicus.org/articles/12/1101/2012/
- [C9] Heus, Seifert, "Automated tracking of shallow cumulus clouds in large domain, long duration large eddy simulations", GMD 6, 1261-1273 (Aug 2013). https://gmd.copernicus.org/articles/6/1261/2013/
- [C10] Zhu, Lee, Weger, Welch, "Clustering, randomness, and regularity in cloud fields: 2. Cumulus cloud fields", JGR (1992) (via secondary listing). https://www.science.gov/topicpages/r/regular+magnetic+clouds.html
- [C11] van Laar, Neggers, "On the size dependence of cumulus cloud spacing" (2021). https://essopenarchive.org/doi/full/10.1002/essoar.10506509.1 ; EGU21-8377
- [C12] EUREC4A / Bony et al. (2020) sugar-gravel-fish-flowers patterns: https://insidecires.colorado.edu/rendezvous/uploads/Rendezvous_2021_8016_1619629860.pdf ; https://essd.copernicus.org/articles/13/3379/2021/essd-13-3379-2021.html
- [C13] Wood, Hartmann, "Spatial variability of liquid water path in marine low cloud: the importance of mesoscale cellular convection", J. Climate (2006) (via abstract and citing papers). https://modis.gsfc.nasa.gov/sci_team/pubs/abstract_new.php?id=02686 ; https://par.nsf.gov/servlets/purl/10286039
- [C14] AMS Glossary, "horizontal convective rolls" / "longitudinal rolls"; Wikipedia "Horizontal convective rolls"; Etling, Brown, "Roll vortices in the planetary boundary layer: a review", BLM 65 (1993) (cited, not read). https://glossary.ametsoc.org/wiki/horizontal-convective-rolls/ ; https://en.wikipedia.org/wiki/Horizontal_convective_rolls
- [C15] WMO International Cloud Atlas (online), altocumulus vs stratocumulus vs cirrocumulus. https://cloudatlas.wmo.int/en/ac-compared-with-sc.html ; https://cloudatlas.wmo.int/en/cc-compared-with-ac.html
- [C16] Craig, Cohen, "Fluctuations in an Equilibrium Convective Ensemble, Parts I-II", JAS 63 (2006); Rasp, Selz, Craig (2018) on ~100 km clustering. https://ecmwf.net/sites/default/files/elibrary/2005/8803-statistical-mechanics-and-stochastic-convective-parameterisation.pdf
- [C17] Hagos, Feng, Plant, Houze, Xiao, "A Stochastic Framework for Modeling the Population Dynamics of Convective Clouds", JAMES 10(2), 448-465 (Feb 2018). https://www.atmos.washington.edu/MG/PDFs/JAMES18_Hagos-etal_Stochastic.pdf
- [C18] Lidar study of updraft and downdraft sizes and velocities, Leipzig, ACP 10, 7845 (2010). https://acp.copernicus.org/articles/10/7845/2010/
- [C19] UCAR SciEd, "Weather fronts". https://scied.ucar.edu/learning-zone/how-weather-works/weather-fronts ; front speeds (secondary): https://cfinotebook.net/notebook/weather-and-atmosphere/air-masses-and-weather-fronts

Games and engines
- [G1] Schneider, "Nubis: Authoring Real-Time Volumetric Cloudscapes with the Decima Engine", SIGGRAPH 2017 Advances (abstract). https://guerrilla-games.com/read/nubis-authoring-real-time-volumetric-cloudscapes-with-the-decima-engine
- [G2] Schneider, "Nubis, Evolved", SIGGRAPH 2022 Advances. https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-NubisEvolved-NoVideos.pdf (exceeds fetch limit; snippets)
- [G3] Schneider et al., "Nubis Cubed", SIGGRAPH 2023 Advances. https://www.guerrilla-games.com/read/nubis-cubed
- [G4] Schneider, "The Real-Time Volumetric Superstorms of Horizon Forbidden West", GDC 2022. https://www.gdcvault.com/play/1027688/The-Real-Time-Volumetric-Superstorms
- [G6] Rare, "The Technical Art of Sea of Thieves", SIGGRAPH 2018 Talks. https://history.siggraph.org/?p=84316 ; secondary: https://www.tomshw.it/videogioco/come-costruito-lincredibile-mondo-di-sea-of-thieves
- [G7] Wang, N., "Realistic and Fast Cloud Rendering", Journal of Graphics Tools 9(3), 21-40 (2004); SIGGRAPH 2003 talk. https://history.siggraph.org/?p=143002
- [G8] PC Gamer, "Here's how Microsoft Flight Simulator creates its realistic weather" (2020). https://www.pcgamer.com/uk/heres-how-microsoft-flight-simulator-creates-its-realistic-weather ; forum: https://forums.flightsimulator.com/t/clouds-are-very-poor-quality-in-the-distance/709022
- [G9] Simul trueSKY 4.3 docs, Cloud Keyframe / Sky Sequencer. https://docs.simul.co/4.3/Reference/Clouds/cloudkeyframe.html ; https://docs.simul.co/4.3/Tutorials/Sequencer/sequenceroverview.html
- [G10] GDC 2026 (13 Mar 2026), "Demystifying the Clouds of Borderlands 4" and "The Dynamic Weather Systems of Borderlands 4". https://schedule.gdconf.com/session/demystifying-the-clouds-of-borderlands-4-adapting-volumetric-cloud-technologies-to-a-ue5-open-world-game/915656 ; https://schedule.gdconf.com/session/the-dynamic-weather-systems-of-borderlands-4-how-they-can-apply-to-games-without-dynamic-weather/917367
- [G11] Star Citizen cloud tech (secondary). https://www.pcgamer.com/au/star-citizen-wraps-up-its-first-star-system-with-the-games-own-cloud-city ; https://www.dsogaming.com/?p=132602
- [G12] Just Cause 4 weather (secondary). https://gamespot.com/articles/just-cause-4s-demo-showcases-its-extremely-lethal-/1100-6459626/
- [G15] Cinevva, "Volumetric clouds and weather effects in modern games" (4 May 2026; secondary survey, authorship unclear). https://app.cinevva.com/blog/2026-05-04-volumetric-clouds-and-weather
- Ghost of Yōtei tech deep dive, PlayStation Blog (23 Oct 2025): https://blog.playstation.com/?p=411201 (as games doc S10)

Simulation and rendering research
- [R1] Harris, Baxter, Scheuermann, Lastra, "Simulation of Cloud Dynamics on Graphics Hardware", Graphics Hardware 2003. https://users.cg.tuwien.ac.at/bruckner/ss2004/seminar/A3b/Harris2003%20-%20Simulation%20of%20Cloud%20Dynamics%20on%20Graphics%20Hardware.pdf (text extracted and read)
- [R2] Dobashi, Kaneda, Yamashita, Okita, Nishita, "A simple, efficient method for realistic animation of clouds", SIGGRAPH 2000. https://history.siggraph.org/?p=118250
- [R3] Goswami, Neyret, "Real-time Landscape-size Convective Clouds Simulation and Rendering", VRIPHYS 2017. https://diglib.eg.org/handle/10.2312/vriphys20171078 ; https://hal.univ-grenoble-alpes.fr/hal-03748553v2
- [R4] Hädrich et al., "Stormscapes: Simulating Cloud Dynamics in the Now", ACM TOG 39(6) (SIGGRAPH Asia 2020). https://research.google/pubs/stormscapes-simulating-cloud-dynamics-in-the-now/ ; https://80.lv/articles/stormscapes-simulating-cloud-dynamics
- [R5] Amador Herrera et al., "Weatherscapes: Nowcasting Heat Transfer and Water Continuity", ACM TOG (SIGGRAPH Asia 2021). https://research.google/pubs/weatherscapes-nowcasting-heat-transfer-and-water-continuity/
- [R6] Hao, "Thunderscapes: Simulating the Dynamics of Mesoscale Convective System", arXiv:2412.00703 (Dec 2024, v4 Apr 2025). https://arxiv.org/abs/2412.00703
- [R7] Weinrauch, Lorbek, Tatzgern, Stadlbauer, Steinberger, "Clouds in the Cloud: Efficient Cloud-Based Rendering of Real-Time Volumetric Clouds", HPG 2023. https://diglib.eg.org/handle/10.2312/hpg20231138
- [R8] Toft, Bowles, Zimmermann, "Optimisations for Real-Time Volumetric Cloudscapes", arXiv:1609.05344 (2016). https://ar5iv.labs.arxiv.org/html/1609.05344 (read)
- [R9] Webanck, Cortial, Guérin, Galin, "Procedural Cloudscapes", Eurographics 2018 / CGF 37(2) (not read; PDF exceeds fetch limit). https://perso.liris.cnrs.fr/eric.galin/Articles/2018-cloudscapes.pdf

Mobile and VR
- W4 Games, "Where Every Millisecond Counts: Optimizing Godot for Standalone XR" (Jul 2026). https://www.w4games.com/blog/w4-games-news-1/where-every-millisecond-counts-optimizing-godot-for-standalone-xr-159
- Oculus standalone skybox guidance (quoted secondhand). https://discussions.unity.com/t/optimal-skybox-for-mobile-vr-oculus-quest-go-gearvr/764577
