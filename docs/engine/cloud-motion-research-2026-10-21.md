# Clouds that move and change like real ones: convection, shear, the day's cycle, storms, fronts, small cues, and how to pay for them on Quest 3 (2026-10-21)

Scope: research only. No code was changed. This report builds on four
earlier ones and does not repeat them:

- `sky-realism-research.md` (the plan);
- `sky-research-games-2026-10-20.md` (the "games doc", R0-R17);
- `cloud-rework-research-2026-10-20.md` (the "rework doc": population,
  envelopes, server weather, empty-space skipping, distance bands, aerial
  perspective, accumulation);
- `living-sky-research-2026-10-21.md` (the "living doc": perception
  thresholds, striations, curl and two-phase churn, per-cloud drift, far
  shadow cascade, scud, time scale).

Code read today [C]:

- `space_soup_sky/src/cloud_layers.rs`: `CHURN_*`, `churn_warp`,
  `churn_pulse`, `layers_at`, `LAYER_WINDS`;
- `space_soup_sky/src/cloud_population.rs`: `PopulationSpec::new`,
  `life_shape`, `top`;
- `space_soup_sky/src/clouds.rs`: `CloudParams::weather_at`, `time_scale`;
- `space_soup_sky/src/precipitation.rs`;
- `space_soup/src/renderer/cloud_sky.rs`: the WGSL `density`.

**How to read the tags.**

- Source quality:
  - **[P]** primary text read (paper, slides, docs or source code);
  - **[A]** abstract or session page only;
  - **[S]** secondary, forum or search snippet;
  - **[R]** recollection, not re-verified;
  - **[C]** our code, read today.
- Every number is marked **(pub)** published, **(meas)** measured, or
  **(est)** estimated here.
- Quest costs are all **(est)**. Nothing in the cloud path has been timed
  per pass on the headset yet. The working baseline from the living doc is
  a march of about 0.6-0.8 ms (inferred from one A/B):
  - about 16 k map texels a frame;
  - about 0.5-1 M density samples a frame (est).
  - At that rate, 20 ALU per sample is roughly +0.01-0.03 ms (est). One
    extra texture fetch per sample is roughly +5-10% of the march (est,
    living doc).

**Provenance.** The web research was done by five research agents in this
session (2026-10-21). One of them read the full slide text of Nubis Evolved
(SIGGRAPH 2022) and Nubis³ (SIGGRAPH 2023), which earlier docs could not
open. The Nubis³ text came from a 14 MB copy hosted on the speaker's
CloudFront. Those findings are [P] at the agent's reading; I have not
re-read the slides myself.

---

## Ranked recommendations

Ranking is by realism gained per Quest millisecond, for clouds that *move
and change*. Items 1-3 are what the user asked for. Item 4 is what pays for
2 and 6. "Owner" follows [[server-may-carry-headset-work]]: the server
carries the weather and simulation, and streams compact `Option` state. With
`None`, the client runs the same simulation from the seed, never both.

| # | Change | Realism gain | Quest GPU, net (est) and how it is paid | Effort | Owner |
|---|---|---|---|---|---|
| 1 | **Convection in each cloud's own frame**, replacing the layer-wide warp and pulse. Detail rises with the cloud's age, faster in the core (about 1.8× the top) than at the edge, where it sinks. The cap rolls over on closed orbits. The turret pulse runs on a 4-6 min period, staggered per cloud. Billows while growing, wisps while decaying (§1). | **High**, from 1-10 km. Edges change at real rates, and the texture no longer slides as one layer. | **+0.01 to +0.04 ms.** ALU only, using values already fetched (profile, top, phase). Paid by item 4c (skip detail in cores). | Medium | Client. Optional server field: per-cloud rise speed comes from the weather's instability. |
| 2 | **Time of day drives the clouds.** Births ramp in at about 11 local solar time and peak at 12-15. Base rises about 100-200 m/h with the LCL. Births stop by late afternoon. Survivors flatten into stratocumulus. Fog forms at night and lifts into stratus 1.5-3 h after sunrise. Cb fire 13-18 on unstable days (§3). | **High.** The whole day reads as real: a clear dawn, fog, then the first cumulus, then towers. | **0 GPU.** CPU only. | Medium | Server weather (moisture and instability by hour). Client fallback: the same function of solar time and seed. |
| 3 | **Wind shear, per layer, from the weather.** Cumulus tilt as a per-cloud offset of the envelope lookup, `S·h²/(2w)` capped near 45°. Sc rolls aligned with the shear. Cirrus trails behind their heads by `S·d²/(2·v_t)`. Shear values from §2. | **Medium-high.** Leaning towers and sheared tops on windy days; mares' tails. | **~0** (ALU in `density` and `cirrus_tau`). | Small | Server (shear and veer per layer in the weather field). Client fallback from the seed. |
| 4 | **Funding package** (§7): (a) slices balanced by cost rather than row count; (b) head-direction priority for the rebuild; (c) profile-weighted erosion, so cores skip the detail fetch; (d) ASTC sliced-3D noise; (e) a sun-aligned light-density grid (Nubis³) replacing the long light taps. | None by itself. It pays for 1, 5 and 6, and removes slice spikes. | **−0.15 to −0.45 ms** (est) for (c)-(e). (a) and (b) are ≈0 average but cut the worst frame. | Medium | Client |
| 5 | **Fronts as moving parametric objects** that set the type sequence by distance: Ci, then Cs (halo), As, Ns ahead of a warm front; a narrow Cb line at a cold front; clearing behind it, then open-cell showers. These replace the eased state lerp in `weather_at` (§5). | **High over minutes to hours.** Banks arrive from the horizon in the right order. | **0 GPU** (existing layers re-parameterised). | Large | Server (front list, under 1 KB). Client fallback: seeded fronts. |
| 6 | **Cumulonimbus as a population stage.** Analytic storm cells in a uniform buffer, tested once per ray: a tower envelope; an ice anvil plate at the tropopause spreading downwind at about 2 m/s; an overshooting dome; mammatus under the anvil; rain shafts as low-extinction density below the base, slanted by wind × fall time; a shelf cloud on the outflow; lightning as a display glow (§4). | **Very high when storms run.** None on fair days. | **+0.05 to +0.2 ms, only while a Cb is in view.** Paid by 4 and by the far band (rework doc). 0 on fair days. | Large | Server (storm cells, lightning events). Client fallback: seeded cells from the instability. |
| 7 | **Small cues tuned to measurements** (§6). Edge width per kind and phase: cumulus margin about 10% of D, sharp for droplet altocumulus, soft and fibrous for ice. Ground shadows that remove only direct sun (global light about 10% of clear sky under thick Cu). Cloud enhancement: +up to 20% irradiance when the sun is clear beside a cloud. Base darkness scaled by each cloud's thickness. | **Medium.** These are the cues that separate "rendered" from "photographed" in VR. | **~0.** Shadow enhancement is one CPU value a frame. | Small | Client |
| 8 | **Cirrus fall streaks and generating heads** in `cirrus_tau`: hooked trails, bounded by the head's age (§2.4). | Low-medium (high cloud). | ~0 (map only). | Small | Client |

Net of 1-3, 4 and 7-8 on a fair day: about **−0.1 to −0.4 ms** (est). With
a storm in view (item 6), about **−0.05 to +0.1 ms** (est).

**Order.**

1. Item 4's (a) and (c) first, with the pass timers the earlier docs ask
   for, so every later number is measured.
2. Then items 1 and 3. They share `density` and the CPU twin.
3. Then item 2.
4. Then 5 and 6, which need the server weather field (rework doc item 1).
5. Item 7 can be done at any time.

**One conflict to settle first: the compressed day.** `test_room` has a
20-minute day [C]:

- The convective window from onset to decay (about 11:00 to 17:00) is then
  **5 real minutes** (est).
- A 1 km cumulus lives `life_ref_s` = 900 s, 15 real minutes [C].
- So the population cannot follow the day: clouds born at noon would
  outlive the evening.

Either:

- (a) scale cloud lives and births by the day's compression (the existing
  `time_scale` lever, about √D per the living doc); or
- (b) let item 2 act on births, base and top caps *only*, and accept that
  clouds outlive their hour.

For a real-length day (or 2-4 h), the problem disappears. The user decides.

---

## 1. Convective internal motion (rising turrets, rolling tops, outflow, entrainment)

### 1.1 Real numbers

- **Turret and top rise speeds.**
  - Stereo photogrammetry at the ARM Miami site, cameras every 20 s at 50 m
    resolution: shallow cloud tops rise about **1-3 m/s (pub)**, deep ones
    **over 10 m/s (pub)**. ARM feature, 2019-02-19 [P/S]:
    https://arm.gov/news/features/post/53490
  - Matching the observations needs a drag coefficient of about 1.
    Romps & Öktem, GRL, 2015 [A]:
    https://arm.gov/research/highlights/621
- **The core moves faster than the top.**
  - Romps & Charn tracked 1,224 thermals in LES. **Core speed ≈ 1.8 × top
    speed (meas, LES)**; their rule of thumb is ×2, and Hill's spherical
    vortex gives 2.5.
  - Thermals rise at a near-constant terminal speed (drag coefficient 0.6).
    Tops were followed for **6 min on average (meas)**.
  - JAS, 2015-08-03 [P]: https://www.osti.gov/servlets/purl/1511414
- **Thermal sizes and lives.**
  - Most thermals have radius **under 300 m**, rise at **2-4 m/s**, and live
    **4-5 min (pub)**. Deeper convection gives larger, faster thermals.
    Hernández-Deckers & Sherwood, JAS, 2016-09 [S]:
    https://www.researchgate.net/publication/305454073
  - TRACER LES: thermals live **5 min** on average, and **90% are gone by
    9 min**. Whole cells live **24.4 min**, with a median thermal radius of
    **0.7 km (pub)**. ACP 26, 10071, 2026-07-17 [P]:
    https://acp.copernicus.org/articles/26/10071/2026/
- **The toroidal cap** (cumulus mediocris, airborne dual-Doppler W-band
  radar).
  - The ring is about **1 km wide and 0.5 km deep**. The core updraft is
    about **3 m/s**, with flanking downdrafts of similar strength. Horizontal
    vorticity is about **0.03 s⁻¹ (meas)**.
  - Wang & Geerts, JAMC 54(10), 2015 [A]:
    https://researchconnect.suny.edu/en/publications/vertical-plane-dual-doppler-radar-observations-of-cumulus-toroida/
  - Treating it as solid-body rotation gives ζ/2 = 0.015 rad/s, so one cap
    **roll-over takes about 7 min (est)**.
- **Pulses, not steady growth.**
  - Pulsating growth dominates a cumulus's whole life (Heus et al., JGR
    2009 [S]).
  - CuPIDO radar passes about 3 min apart show a toroidal circulation, a
    cloud decaying 6 min later with peak liquid water down 3×, and **new
    turrets rising through the remains of old ones (meas)** [P]:
    https://archive.eol.ucar.edu/projects/cupido/documents/CuPIDO_SOD.html
  - In BOMEX LES, about a third of shallow clouds live **under 4 min**, and
    the histogram runs to about **60 min (pub)**. Dawe & Austin, ACP 2012
    [P].
- **Edges.**
  - The subsiding shell comes from evaporative cooling where cloud mixes
    with clear air at the edge (Heus & Jonker, JAS 2008 [A]). About **80% of
    the compensating downdraft lies within 400 m** of the cloud (Jonker et
    al. 2008, **pub**) [S].
  - Entrained air enters at the front of a thermal and is swept round its
    sides into its wake [S].
  - **No published visible-edge retreat rate was found.** Mixing-limited
    eddy turnover at the edge is about 100 m ÷ 1 m/s ≈ **100 s (est)**.
- **Congestus to Cb.**
  - Congestus over 10 m/s (pub, above).
  - Cb aircraft penetrations: peaks of **25-26 m/s**, one gust of **63 m/s
    (meas)** [A]: https://repository.library.noaa.gov/view/noaa/17638
- **What this says about ours [C].**
  - The churn turns one warp per layer every 90-150 s (`CHURN_PERIOD_S`).
    The erosion pulse runs at the same period, ±35% (`CHURN_PULSE`).
  - Every cumulus in the layer therefore pulses **in step** (apart from a
    phase set by the shape noise `n.g`) and **too fast**: real turrets live
    4-6 min.
  - The cumulus "boil" is a uniform 0.25 m/s vertical scroll of the shape
    noise. That is too slow for the core and the wrong sign for the shell.

### 1.2 How shipped games and papers animate it

- **Nubis³ (Burning Shores, SIGGRAPH 2023) [P].**
  - The voxel clouds are grown **offline** by a fluid solver ("Aero" in
    Houdini: advect density and velocity, enforce non-divergence, buoyancy,
    sourcing), composited into "Frankencloudscapes" and stored in voxel
    grids.
  - **At runtime the only motion is a wind offset**:
    `inSamplePosition -= cCloudWindOffset * voxel_cloud_animation_speed`.
  - The conclusions table lists evolution as **"Pseudomotion Only"** for
    both the envelope and the voxel method. Runtime evolution is left "for
    the future", because the SDFs would have to be rebuilt on the fly.
  - **The useful rule comes from their time-lapse study:** "Decreasing
    Density = Curly Layered Wisps / Increasing Density = Layered Billows".
    - The detail noise is 128³ with 4 channels: low- and high-frequency
      "Curly-Alligator" (alligator noise distorted by curl), and low- and
      high-frequency Alligator.
    - Wisps blend from low to high frequency over the dimensional profile.
      Billows do the same over `pow(profile, 0.25)`.
  - Source: Schneider, Nubis³ slides (2023-08-05):
    https://d3d3g8mu99pzk9.cloudfront.net/AndrewSchneider/Nubis%20Cubed.pdf ;
    https://www.guerrilla-games.com/read/nubis-cubed
- **Nubis Evolved superstorm (SIGGRAPH 2022) [P].**
  - The "M.D.R. Vortex Field" samples the noise at
    `GetRotatedPosition(pos, center, time*ring_rotation_speed[n] + ring_skew[n])`:
    concentric rings, each with its own speed and skew. These are closed
    orbits, so no reset or two-phase blend is needed.
  - Their temporal reprojection rotates the history sample with the storm,
    under a radial mask.
  - https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-NubisEvolved-NoVideos.pdf
- **Curl noise.** Bridson, Hourihan & Nordenstam, SIGGRAPH 2007 [P]:
  https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf
  - The velocity is v = ∇×ψ. Noise of scale L gives vortices about L across.
  - Modulate the potential (∇×(Aψ)), not the velocity, to stay
    divergence-free.
  - Eq. 8 is a ready-made **vortex-ring potential "for smoke rings and
    plumes"**: f(|x−x_C|/R)·(R²−|x−x_C|²)/2·ω_C. That is our turret cap.
- **Flow noise.** Perlin & Neyret 2001 [S]: gradients rotate over time, and
  finer octaves are displaced along coarser gradients.
  - The 3D version is Gustavson & McEwan, JCGT 11(1), 2022 [P]:
    https://jcgt.org/published/0011/01/02/paper-lowres.pdf
  - Cost: **0.047 ns per value on an RTX 3080, 1.84 ns on an Intel UHD 650
    (meas)**, with rotation adding 9%.
  - At 1 M samples a frame that is about 1-3 ms per octave on a mobile-class
    GPU (est, extrapolated). **Too costly per sample.** Since α is
    2π-periodic, it could be baked into a looping 3D texture of K phases
    instead.
- **Advected textures** (Neyret, SCA 2003) [R]: 2-3 layers regenerated at a
  latency set by local deformation, cross-faded.
  **Lagrangian texture advection** (Yu et al., TVCG 2011) [A] keeps the
  spectrum.
  - Both cost 2-3 noise fetches per sample. For us they are only sensible
    applied to the 2D envelope maps, not per sample.
- **Goswami & Neyret, VRIPHYS 2017** [A]: Lagrangian buoyant parcels mixing
  with an implicit environment, amplified by noise. Parcels are the natural
  source of per-cloud rise speeds for our population.
- **Webanck et al., CGF 2018** [A]: keyframe morphing by optimal transport,
  with no internal convection.
- **Dobashi 2000** [R]: a Boolean cellular automaton (bit operations).
- **Other engines.**
  - trueSKY's "Churn" is a keyframe variable for edge churn [S].
  - DCS 2.7 announced "natural cloud cycles" for after launch, not shipped
    (2021-04-09) [S].
  - **No developer statement was found** on cloud evolution in MSFS 2024,
    X-Plane 12 or Frostbite (which only scrolls) [R].
- **Already covered elsewhere, not repeated:** HZD 2015 curl distortion,
  Vlachos/Grimes two-phase flow, UE 4.26 flow maps (living doc §2-3).

### 1.3 Costs here, and the design that fits the map

| Technique | Extra per density sample | Fits our map? |
|---|---|---|
| Layer scroll or boil (ours today) | 1 MAD | Reads as a slide |
| Rotating warp of the detail coordinates (ours today) | ≈10 ALU, 0 fetches | Fine, but layer-wide and in step |
| 2D curl texture (HZD) | +1 fetch | Yes; living doc item 5 |
| Two-phase flow map | 2× detail fetch + 1 flow fetch | Doubles the dominant cost; poor |
| Procedural flow noise | ~1-3 ms per octave per frame (est) | No (bake it instead) |
| **Per-cloud rise, closed-orbit cap roll, staggered pulse (below)** | **≈15-25 ALU, 0 fetches** | **Best fit** |
| Optional outward-direction map for the cap's lateral outflow | +1 RG8 fetch, cap band only | Yes; cumulus layer only |

**The proposed form.** It works because our envelope already carries what
convection needs [C]:

- **R** is the profile p: 0 at the edge, 1 at the core.
- **G** is the local top.
- **B** is the life phase φ.

Each cloud's age is φ × its life. Any per-cloud motion indexed by age is
therefore **bounded by the cloud's life and differs between neighbours**. It
cannot become a sky-wide scroll, which is the failure of a boil.

In `density` (and the CPU twin), with h_c = h / top as the height inside
the cloud:

1. **Rise in the cloud's frame.**
   - Offset the detail coordinate up by `w(p)·age`, where
     `w(p) = w_top·(−0.3 + 2.1·p)` (est). The core rises at about 1.8×
     w_top, and the edge (p < 0.15) sinks slowly: the shell.
   - Use w_top of about 1-3 m/s for humilis and mediocris, and 3-10 m/s for
     congestus. It comes from the cloud's top draw (`PopulationSpec::top`),
     or from the server's instability.
   - Because w depends on p, the texture is **sheared radially**: edges
     curl down while the core goes up. It does not slide as a block.
2. **Cap roll-over.**
   - For h_c > 0.6, rotate the detail coordinate in the (outward, up) plane
     about a ring core at 0.6R and h_c ≈ 0.8. Use rate ω ≈ 0.015 rad/s
     (meas, ζ/2), scaled by w_top / 3 m/s.
   - The angle is ω·age, bounded by life (15 min ≈ 2 turns), so the
     differential-rotation shear that grows without limit in the Nubis
     rings stays bounded.
   - The outward direction is the one thing the envelope does not carry.
     Either:
     - (a) skip lateral roll and rotate in (p, h_c) only, at 0 fetches; or
     - (b) have `fs_splat` write a 2-channel RG8 outward unit vector, which
       it knows from each cloud's centre, and read it only in the cap band.
       That costs +1 fetch on about 20-30% of cumulus samples, roughly
       +0.01-0.02 ms (est).
3. **Turret pulse.** Replace the layer-wide `CHURN_PULSE` period (90-150 s)
   with a **4-6 min period per cloud**, phased by its seed or position
   hash, so neighbours are out of step. Optionally let a pulse peak nudge
   the cloud's top up by 5-10% (the CPU's `life_shape`).
4. **Billows versus wisps** (the Nubis³ rule). Use the sign of growth, which
   phase already gives: growing while φ < `grow`, decaying while
   φ > 1 − `decay` [C].
   - **Growing:** billowy erosion, `pow(p, 0.25)` weighting, low-frequency
     detail, the cut applied mostly at the top.
   - **Decaying:** curly wisps, the high-frequency detail and the curl warp
     at full amplitude, a wider cut from the edge inward.
   - This also makes evaporation the fastest visible change at the
     horizon, which the living doc found is the motion that reads there.
5. **Keep the current rotating warp at a lower amplitude** (about 0.05) as
   the small-eddy texture, or retire it.

**Cost:** about +0.01 to +0.04 ms (est), ALU only, with option (b) at the
top of that range. It is paid by §7's profile-weighted erosion, which skips
the detail fetch in cores. The same p weighting is what makes the core rise
here.

**Risks.**

- The CPU twin and its tests (`the_churn_reshapes_edges_but_keeps_the_cover`,
  GPU against twin) need the same maths.
- Merged clouds blend phase through the splat, so a merged cloud's "age" is
  a blend. That is harmless.
- At map period 2.7 s, 3 m/s of rise is 8 m between builds (est), far below
  a map texel. No seams.

---

## 2. Wind shear

### 2.1 Typical values

| Layer | Speed shear (est from the sources below) | Notes |
|---|---|---|
| Sub-cloud and stratocumulus | 0-2 m/s per km | Veer 15-40° across the boundary layer (pub); corrected ≈20° over land, ≈10° over sea (pub) [S] |
| Fair-weather cumulus layer | **1-3 m/s per km** | Trade-cumulus LES used 0.9-3.6 m/s per km (pub) [P]. Amazon LES used 1.2 and 2.4 m/s per km, about the 33rd and 50th percentiles observed (pub) [P] |
| Mid-troposphere (altocumulus) | 2-5 m/s per km | [R], not verified |
| Upper troposphere (cirrus) | **5-10 m/s per km** | Common near the jet (pub) [S] |
| Jet streak, just above the tropopause | 15-20 m/s per km | ERA5 "strong shear" S² = 4×10⁻⁴ s⁻² ≈ 20 m/s per km, 1-10% of the time, in a layer 1-2 km thick (pub) [A] |
| Bulk 0-6 km by storm type | under 10 m/s: disorganised cells; 10-18: multicells; over 20: supercells | Supercell 10th percentile 29-36 kt against non-supercell 90th percentile 28 kt (pub) [S] |

Sources:

- Helfer et al., 2020-12 [P]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7757185/
- Cecchini et al., ACP 22, 11867, 2022-09-14 [P]:
  https://acp.copernicus.org/articles/22/11867/2022/
- ERA5 shear: WCD 2, 631, 2021 [A]: https://wcd.copernicus.org/articles/2/631/2021/
- NWS Louisville tip sheet [S]: https://www.weather.gov/media/lmk/soo/SvrWx_Fcstg_TipSheet.pdf
- Thompson et al. (SPC), 2003 [S]: https://origin-east-www-spc.woc.noaa.gov/publications/thompson/rucparms.pdf
- Ekman turning [A/S]: https://www.bib.irb.hr/498219 ; https://www.mdpi.com/2073-4433/11/11/1178

Our `LAYER_WINDS` multiples ×0.8, 1, 1.7, 3.2 on 8-12 m/s, with turns of
+12°, 0, −25° and +38° [C], imply about 2-4 m/s per km between layers. That
is plausible as a mean (est).

### 2.2 What shear does to clouds

- **Tilt and cover.** Shear tilts cumulus, forward and backward shear in
  opposite directions. Tilting raises low-cloud cover by about 10-20%
  (pub, LES). Backward shear holds tops near 2 km where unsheared runs reach
  about 7 km (pub, LES). Source: Helfer 2020 [P].
- **Shape.** Under shear, cumulus are about 80-100 m wider, fewer are deeper
  than 1,200 m, they cluster (about 900-1,000 fewer clouds), and core
  updrafts are up to 0.6 m/s weaker (pub, LES). Source: Cecchini 2022 [P].
- **Observed tilt.** At SGP the sub-cloud updraft is slightly asymmetric in
  time, "indicating a down wind tilt" (meas) [P]:
  https://asr.science.energy.gov/meetings/stm/posters/poster_pdf/2017/P001693.pdf
  (2017-03).
- **Stratocumulus rolls** form "in cases with marked wind shear" and align
  with the shear vector. Wood, MWR 2012 [P]:
  https://www.chem.pmf.hr/_download/repository/Review_Sc_clouds_R_Wood_MWR_2012.pdf
- **Cirrus uncinus.**
  - Crystals form in the upshear updraft of a generating head, at about
    1 m/s. Shear carries them into the downshear trail; the head's downshear
    side is a downdraft (pub). Source: 1975, [A]:
    https://impacts.ucar.edu/en/publications/cirrus-uncinus-generating-cells-and-the-evolution-of-cirriform-cl-5/
  - Ice fall speed is **0.2 m/s at −40 °C, rising to 0.6 m/s at −10 °C
    (meas, 17 months of Doppler lidar)** [S]: https://centaur.reading.ac.uk/1983/
- **Anvils.** Nubis Evolved flips the vertical profile so the top spreads,
  and "rotated and skewed the noises… wind-swept" [P, 2022 slides].

### 2.3 How renderers apply it

- **Unity HDRP, source code [P]:**
  `baseNoiseSamplingCoordinates += properties.height * float3(_WindDirection.x,_WindDirection.y,0) * _AltitudeDistortion;`
  - `height` is normalised between the lowest and highest cloud altitude.
    The offset is in noise units and points along the wind only, not along
    a shear vector.
  - Defaults: altitude distortion 0.25 (range −1 to 1); shape speed 1.0;
    erosion speed 0.25; cloud map speed 0.5; vertical shape and erosion wind
    0 (pub).
  - The map, shape and erosion scroll at different rates, which is HDRP's
    churn. Sources:
    https://raw.githubusercontent.com/Unity-Technologies/Graphics/master/Packages/com.unity.render-pipelines.high-definition/Runtime/Lighting/VolumetricClouds/VolumetricClouds.cs
    and `VolumetricCloudsUtilities.hlsl`, read 2026-10-21.
- **Schneider, GPU Pro 7 (2016) [R]:**
  `p += height_fraction * wind_direction * cloud_top_offset` with an offset
  of 500 m, plus `p += (wind + (0, 0.1, 0))·time·speed`.
  - The 2015 slides only say "wind direction varies over altitude" [P].
- **UE5:** per-axis wind controls in the cloud material. No altitude skew is
  documented [P].
- **trueSKY, MSFS:** no shear formula found.
- **No engine checked models tilt or fall streaks from a wind profile.**
  All of them skew noise linearly with height.

### 2.4 How it fits our map

**Cumulus: do not let per-height advection do the tilting.** Today's display
already advects each texel by the wind at its own height [C]. Over one
map's age (≤ 2.7 s) that is a few metres of tilt. Correct, and bounded.

If shear were instead applied as differential advection over a cloud's
*life*, it would grow without limit: Δx = S·Δz·t, which is 2.4 km for
S = 2 m/s per km, Δz = 1 km and t = 20 min (est). The cloud would shred.
Real cumulus keep their low-level momentum and move with a steering wind
near mid-cloud. Our envelope already moves in one wind frame per layer,
which is that steering wind.

**Proposed tilt.** In `density`, offset the **envelope** lookup (not only
the noise) by:

`tilt(h) = Ŝ · min(S·(h·H)²/(2·w_top), (h·H)·tan 45°) · k`

where:

- H is the layer depth;
- k ≈ 0.5, because updrafts keep part of their momentum (est);
- Ŝ is the layer's shear direction (the difference between its own wind and
  the layer below).

With S = 2 m/s per km, w = 2 m/s and h·H = 1 km, that gives about 250 m of
lean at the top (est). Offsetting the envelope moves the whole column; the
noise alone would move only texture. Cost: about 6 ALU per sample.

The CPU twin must follow, and so must the ground-shadow twin.

**Stratocumulus.**

- Align the rolls (the rework doc's street modulation) with the shear
  vector rather than the mean wind.
- The Sc tile already has a `stretch` of 1.5 [C]. Orient it along the
  shear.

**Cirrus.**

- In `cirrus_tau`, draw each streak as a trail behind a head:
  `Δx(d) = S·d²/(2·v_t)`, with v_t of about 0.3-0.6 m/s.
  - The trail hangs vertical under the head and flattens downward, which
    gives the hook.
  - Example: S = 5 m/s per km and d = 1 km give a 5 km trail at slope 0.1
    (est).
- Bound d by the head's age, d ≤ v_t·age.
- The cirrus layer is 2D, so in the map this is an elongated, curved tau
  pattern along Ŝ. Seen from the ground that is what mares' tails look like.
- The rework doc's flow-mapped cirrus (its item 10) is the place for it.

**Owner.** The server weather field carries shear and veer per layer, about
4 bytes per layer per cell. The fallback derives them from the seed and the
weather state:

- 0-2, 1-3, 2-5 and 5-10 m/s per km by layer;
- a storm state picks its 0-6 km bulk from the table.

---

## 3. The diurnal convective cycle

### 3.1 What real days do (midlatitude summer, land)

- **Onset.**
  - A 138-day ARM SGP composite (2011-15): fair-weather cumulus initiate at
    **11:25 CST (meas)**, when the boundary-layer top meets the LCL. That is
    about 10:55 local solar time at 97.5°W (est).
  - Cloud base stays coupled to the boundary-layer top until about
    15:00 CST, then decouples.
  - Peak cloud fraction is **23% (meas)**, and onset is later on days with
    more inhibition.
  - Source [P]: https://asr.science.energy.gov/meetings/stm/posters/poster_pdf/2017/P001693.pdf
- **Peak and thinning.**
  - Cloud fraction peaks at **12-14 CST**, with a maximum near 14:00 (meas).
  - Cloud base **rises all day**. The top rises through the morning, then
    stays flat or falls, so **clouds thin through the afternoon (meas)**.
    Days were selected with bases of 1-3 km.
  - Sources: Berg & Kassianov, 2006 [P]
    (https://arm.gov/publications/proceedings/conf16/extended_abs/berg_lk.pdf);
    PNNL, 2008 [A].
- **What sets depth.**
  - A moister morning boundary layer gives a lower base and deeper clouds;
    "thick" days have a mean depth over 300 m.
  - A lower Bowen ratio and a weaker inversion are secondary, and "shear
    doesn't matter" for fair-weather cumulus.
  - Source: Zhang & Klein, JAS 2013 [A]; ARM slides [P]:
    https://asr.science.energy.gov/meetings/stm/2013/presentations/Klein-sgp-shcu.pdf
  - Clouds are larger on days with earlier onset (2019 poster [P]).
  - **Cloud diameter ≈ LCL height (pub)** (Romps, ~2021 [S]; the PDF could
    not be fetched).
- **Boundary layer.** It peaks in the afternoon at **up to 2,800 m (meas)**
  at SGP [S]. No published hourly growth rate was found.
- **The LCL rule.** z_LCL ≈ **125 m/K × (T − Td)** (pub; Espy with
  Lawrence's coefficient, cited in Romps 2017 [S]). Errors can exceed 200 m.
- **Fog and stratus at dawn.**
  - In one radiation-fog case, fog formed between about 23:00 and 03:00
    local and **cleared 1.7-3 h after sunrise**, with tops of 85-115 m
    (meas). ACP 25, 13493, 2025-10-23 [P]:
    https://acp.copernicus.org/articles/25/13493/2025/
  - At SIRTA (Paris; 250 events over 7 years), **more than half of
    post-sunrise dissipations lift into stratus lasting at least 2 h
    (meas)** [S, same paper].
- **Stratocumulus.**
  - Cover, thickness and liquid water path peak **before sunrise**;
    convective clouds peak in the afternoon. Over the eastern subtropical
    oceans the amplitude exceeds 20% of the mean (pub).
  - Over land, the relative amplitude of stratocumulus cover is about
    **double** that over the ocean (pub, Hahn & Warren 2007).
  - Nocturnal stratocumulus gives way to cumulus by day in transition
    regions.
  - Source: Wood 2012 [P].
- **Deep convection.**
  - Over land, thunderstorm frequency peaks in the **late afternoon**, with
    an amplitude over 50% of the daily mean (pub).
  - Over tropical oceans, rain peaks between **midnight and 04 local solar
    time**, and near 06 near coasts.
  - Source: Dai, J. Climate 2001 [A]:
    https://impacts.ucar.edu/en/publications/global-precipitation-and-thunderstorm-frequencies-part-ii-diurnal/

### 3.2 What games do

- **HZD 2015 [P]:** precipitation turns clouds into cumulonimbus at 70%
  coverage. Beyond 15 km, clouds bias to cumulus at about 50% to keep the
  horizon interesting.
- **Nubis 2022 [P]:** "weather states", each a cloudscape, switched over
  time.
- **trueSKY [S]:** per-layer keyframes interpolated by time of day. Its
  example has cloudiness 0.9 at 11:00 and 0 at 14:00:
  https://docs.simul.co/4.4/Reference/Clouds/cloudkeyframer.html
- **HDRP:** presets only, none tied to the hour (pub, in m):

  | Preset | Bottom | Range | Density |
  |---|---|---|---|
  | Sparse | 3,000 | 1,000 | 0.4 |
  | Cloudy | 1,200 | 2,000 | 0.4 |
  | Overcast | 1,500 | 2,500 | 0.3 |
  | Stormy | 1,000 | 5,000 | 0.35 |

- **MSFS and DCS:** real or forecast data. **No documented time-of-day
  cloud logic.**
- **RDR2 and Sea of Thieves:** no source found.
- **No shipped title was found that drives convective onset, a rising base
  and evening decay from physics.** Doing so would be ahead of the field.

### 3.3 Mapping for our engine (est)

The cumulus base today is a fixed `altitude_m`. The layers' coverage
follows `weather_at` plus `wander` [C]. Nothing depends on the hour.
Proposed, with t in local solar time, sr = sunrise and ss = sunset:

| Window | Sc / fog (the low layer) | Cumulus | Cumulonimbus (moist, unstable, bulk shear) | Ac / Ci |
|---|---|---|---|---|
| 00 → sr | Fog births on calm, moist nights; Sc cover ×1.1-1.2 | births 0 | Over sea only (peak 00-06) | from the weather |
| sr → sr+3 h | Fog lifts to stratus at sr+1.5-3 h; stratus lasts about 2 h, then breaks into Cu | 0 | 0 | ″ |
| about 10:30-11:30 | Remnant Sc breaks up | births ramp 0 → peak over about 1.5 h. Base = 125·(T−Td), about 0.8-1.1 km. Earlier on moist days, later with inhibition | 0 | Ac castellanus on storm mornings [R] |
| 12 → 15 | Sc ×0.8 | Peak births. Cover 0.2-0.3 on fair days, 0.4-0.6 moist. Base rises 100-200 m/h. Depth 300-800 m (moist: to 1.5 km). Top cap = min(inversion, base + depth) | Births from about 13 once Cu have run 2 h or more on unstable days | ″ |
| 15 → ss−1 h | — | Births fall. Tops flat, clouds thinning; base 1.5-2.0 km (dry: 2.5) | Peak 15-18 | Anvil cirrus spreads downwind |
| ss−1 h → ss+2 h | Survivors flatten into Sc near the old top; reform at night if moist | births 0; clouds fade | Decaying; anvils persist [R] | Anvil cirrus stays |

**Implementation.**

- `layers_at` takes the solar hour and the weather's moisture and
  instability. For each layer it returns:
  - a **birth-rate multiplier** (fed to `births_per_m2_s` through the
    coverage it is asked for);
  - the **base and top cap** in metres;
  - **kind weights**.
- The base moves for clouds already up too, since it is a layer parameter.
  Real bases also rise with the LCL, so that is correct.
- Cumulus diameters follow the LCL (D ≈ z_LCL, pub), so `d_min`/`d_max`
  scale with the base: smaller clouds early, larger in mid-afternoon.
- Fog belongs to the ground-fog system, not the cloud map. The lift into
  stratus is the low layer with a 100-300 m base for 2 h.

**Cost:** 0 GPU; CPU microseconds.

**Owner:**

- The server field holds the moisture and instability by hour, and its own
  diurnal heating (rework doc §3.1).
- The fallback evaluates the same function of solar time and seed.
- **Note the compressed-day conflict at the top of this report.**

---

## 4. Cumulonimbus

### 4.1 Real numbers

- **Life.**
  - A single cell lasts **30-60 min (pub)**. Mature storms are **12-18 km
    deep (pub)**. The gust front forms in the mature stage, and a "remnant
    anvil" is all that is left at the end. NWS Spotter Guide [P]:
    https://www.weather.gov/spotterguide/life
  - The stage split (about 10-15 min towering, 15-30 min mature) is only
    from a university text [S].
  - Multicells persist for hours by spawning new cells on the gust front
    (rework doc).
- **Morphology (WMO International Cloud Atlas, 2017) [P].**
  - The top is "nearly always flattened" into an anvil or plume. The base
    is very dark, with ragged low cloud under it, and virga is common.
  - Supplementary features: incus, mamma, praecipitatio, virga, pannus,
    pileus, velum, arcus, murus, cauda, flumen, tuba.
  - **Mamma are seen most often under the projecting part of the anvil.**
  - **Arcus (shelf cloud)** is a dense horizontal roll on the lower front,
    made by downdraught air spreading out ahead. A roll cloud is detached; a
    shelf stays attached.
  - **Murus (wall cloud)** is an abrupt, persistent lowering in the
    rain-free base, under the updraft.
  - Pages: https://cloudatlas.wmo.int/en/cumulonimbus-cb.html ;
    https://cloudatlas.wmo.int/en/supplementary-features-and-accessory-clouds-cumulonimbus.html
- **Updrafts.** 25-26 m/s peaks (meas). Parcel theory w = √(2·CAPE) gives
  about 45 m/s at 1,000 J/kg (est).
- **Downdrafts.** Downbursts fall at **5-25 m/s**; microbursts are under
  4 km across and last **2-5 min (pub)** [S, Stull].
- **Overshooting tops.**
  - A single one lasts **under 30 min** and reaches **1-20 km across**
    (pub) [P EUMETSAT, about 2009; A NOAA].
  - One sounding gives a theoretical overshoot of **5 km** (pub).
  - Cooling of 9 K against the anvil implies about **1.2 km** of height
    (pub, GOES-R ATBD).
  - Pulsing tops last "a few minutes" each [S].
- **Anvils.**
  - Tropical thick anvils average **2.7 km thick** (pub, TRMM).
  - CloudSat gives a modal MCS anvil thickness of **4-5 km**, and anvils
    reach **1.5-2× the rain area's radius, up to 5×** (pub) [A].
  - Model anvil edges spread at **about 2 m/s**, from 10 km to about 50 km
    wide over about 3 h (pub, CRM) [A].
  - **No observed downwind anvil length was found.**
- **Mammatus.**
  - Schultz et al., JAS 63(10), 2006: adequate measurements do not exist,
    and ten mechanisms are proposed [A].
  - Lobes are 1-3 km across and about 0.5 km deep, live about 10 min each,
    and fields last 15 min to hours (pub, uncited) [S].
- **Precipitation.**
  - Rain terminal velocity is about 9-10 m/s for large drops and snow about
    2 m/s (pub, Thunderscapes and Gunn-Kinzer [R]). Large drops fall 25-30%
    slower in turbulence (pub) [A].
  - From a 1.5 km base at 5-9 m/s, a drop falls for 170-300 s. 10 m/s of
    relative wind then **slants the shaft foot 1.7-3 km downwind (est)**.
  - Virga is trails that evaporate before reaching the ground [A, WMO].
  - **No rain-shaft width was found.**
- **Lightning.**
  - Most storms produce under 3-4 flashes/min. Florida air-mass storms
    average **2.4/min**, with a 5-min peak of 30.6 (pub) [A, NASA].
  - Severe storms exceed **60/min**, some reach 100-170 (pub) [A].
  - A flash lasts about **0.2 s (0.03-2 s)**, with **2-4 strokes 40-60 ms
    apart** (pub) [S].
  - Deeper optical sources give **wider, longer-lit footprints** (pub)
    [A, Peterson & Mach 2021].

### 4.2 How games render it

- **HZD 2015 [P]:** a G-channel precipitation signal "transitions the map
  … to cumulonimbus clouds at 70% coverage"; precipitation also drives the
  rain effects. The extracted notes contain no "anvil" or "anvil bias". The
  GPU Pro 7 `anvil_bias` code (coverage bent near the top of the height
  range) is [R].
- **HFW superstorms (GDC 2022) and Nubis Evolved (2022).**
  - Tornadic superstorms, internal lightning and fast-moving clouds,
    "without expensive simulation" [A].
  - The 2022 slides add the ring vortex field, rotated reprojection, a
    flipped profile so the top spreads, and wind-swept skewed noise
    (§1.2, §2.2) [P].
  - Burning Shores storm flashes were "a mix of particles and volumetric
    lighting" [S].
  - Schneider's lighting split treats lightning as a secondary-scattering
    source [A].
- **Unity HDRP [P]:**
  - Advanced mode combines Cumulus, Alto Stratus, **Cumulonimbus** and
    **Rain** maps, each with a multiplier.
  - The Cb map "defines the distribution of anvil shaped cumulonimbus".
  - The LUT has Cb as its own column (R profile, G erosion, B AO).
  - No rain shaft or lightning rendering.
- **X-Plane 12 [S]:** type is separate from coverage, so a cumulus can
  "grow" into a Cb in its 4D field. Lightning lighting the clouds was on
  the roadmap.
- **MSFS 2024 [S]:** SU3 made lightning "far less excessive". Users report
  lightning not tied to cells.
- **DCS:** nothing on Cb, shafts or lightning found.
- **UE5:** nothing on storms; cost levers only.
- **Thunderscapes (Hao et al., 2024/2025) [P]:**
  - Simulation is offline: **0.24-0.81 s per frame on an RTX A6000 (meas)**
    on grids up to 500×376×500 over 30 km, with a 1-min time step.
    Rendering is in Houdini.
  - It notes anvils with sharp edges spreading downwind.
- **Dobashi et al. 2007 [A]:** lightning as a weighted sum of precomputed
  "basis intensities" from point lights, for static clouds.

### 4.3 How it fits our map and population (est)

All Cb timescales are minutes, far longer than a 2.7 s map. Only lightning
must live outside the cache.

**Storm cells as analytic objects in a uniform buffer.** Use up to 4-8
cells. Each has a centre, radius, tower top, anvil radius, downwind axis
and phase. They are spawned by the population from the cumulus layer:

- a cumulus whose top passes the freezing level under high instability
  (rework doc §3.2);
- or a server storm event.

Rules for the march:

- **Test cells once per ray, not per sample.** Intersect the ray with each
  cell's bounding cylinder in `fs_map`. Only rays that hit one add the
  storm segments.
- Uniform arrays read at a runtime index are not the Adreno local-array
  cliff (memory `adreno-local-array-cliff`, which concerns `var` arrays),
  but keep the loop short.
- **Tower:** the cumulus layer's envelope at the cell, with its top lifted
  to 10-12 km. The march segment for that ray extends to the cell top,
  bounded by `VIEW_STEPS_MOST`.
- **Anvil:** a plate at the tropopause, 2.7-5 km thick.
  - Its radius is 1.5-5× the core's, offset downwind along the upper wind,
    with its edge spreading at about 2 m/s.
  - Ice, so soft and fibrous edges (§6).
  - Use the cirrus-style softness with 6-8 samples.
- **Overshooting dome:** 1-15 km across, 1-2 km above the anvil, pulsing on
  a few minutes.
- **Mammatus:** an inverted-bump modulation of the anvil underside, 1-3 km
  lobes about 0.5 km deep, about 10 min each.
  - Lit by a low sun, this is the most photogenic storm feature.
  - Cost: ALU only, in the anvil samples.
- **Rain shaft:** a slanted cylinder of low-extinction density below the
  base.
  - Slanted by wind × fall time (1.7-3 km).
  - 4-6 samples, only on rays that hit it.
  - Precipitation curtains read grey-blue against the light and brighten
    toward the sun through the phase function.
  - **Virga** is the same shaft ending at an evaporation height, under Ac
    and anvils.
  - `precipitation.rs` already decides where showers fall [C]. The shaft
    must be the same object, so rain on the ground and the shaft in the sky
    agree.
- **Shelf cloud:** a low elongated object on the outflow edge ahead of the
  cell (the gust front is where new cells spawn). **Wall cloud:** a local
  lowering of the base on the rain-free side.
- **Lightning.**
  - The display glow from the depth statistics already in the map (games
    doc R16): `I·falloff·(1−T)·tint`, with a falloff radius that grows with
    source depth.
  - Timed per flash at 0.2 s with 2-4 strokes 40-60 ms apart. At the 36 Hz
    app rate (27.8 ms a frame), strokes alias to 1-2 frames each, which
    reads correctly as flicker (est).
  - It is a brightness change, not motion, so the SpaceWarp motion vectors
    need nothing new.
  - Flash rates are 2-4/min for an ordinary cell and 60-170/min severe.

**Cost:**

- 0 on fair days;
- about +0.05 to +0.2 ms while a Cb is in view (est): extra samples on
  storm rays, mostly at long range where the far band applies;
- display: lightning is ALU only while a flash is live.

**Owner:** the server owns the cells and lightning events (position, time,
seed), so all players see the same strike. The fallback spawns cells from
the seeded instability.

---

## 5. Fronts and weather sequences

### 5.1 Real geometry and timing

- **Slopes.** Warm front about **1:200-1:300**; cold front about
  **1:50-1:100 (pub)**. Overrunning ascent is "several centimetres per
  second" (pub). Frontal zones are 20-100 miles across (pub).
  - Penn State METEO 3 [P]: https://courses.ems.psu.edu/meteo3/node/2015
  - Oklahoma Climatological Survey [P]: https://okfirst.mesonet.org/train/meteorology/Fronts.html
  - Britannica [S]
- **The warm-front ladder.**
  - Cirrus about **1,000 km** ahead; As/Ac about **500 km**; the rain belt
    (Ns) the last **300-400 km**. The cloud base falls as the front nears
    (pub). Stratus and possible fog follow.
  - Sources: Oklahoma [P]; Met Office [S]:
    https://weather.metoffice.gov.uk/learn-about/met-office-for-schools/other-content/other-resources/understanding-weather
  - **No primary distance was found for cirrostratus.**
  - Cs to Ns takes "a day or a little more" (pub) [P, PSU].
  - High cloud is seen about **12 h** ahead (pub) [P, UBC ATSC 113, 2020].
  - At 5-8 m/s, 1,000 km is 35-55 h (est). **The sources disagree, so lead
    time is a tunable.**
- **Halo lead.** A cirrostratus halo comes 12-24 h before rain, 6-8 h for a
  fast front (pub) [S]. The folk rule says a halo means rain in 6-10 h, and
  As in about 4 h [S].
- **The warm sector:** cloud thins, the weather turns generally fine (pub)
  [S, Met Office].
- **The cold front.**
  - The cloud band is "usually a few hundred km wide" (pub) [P/S,
    EUMeTrain].
  - The **narrow cold-frontal rainband is 3-5 km wide and tens to hundreds
    of km long**, broken into gaps and cores (pub) [S].
  - A prefrontal squall line runs less than 100 km ahead of the front (pub)
    [P, AMS Glossary].
  - Behind it: clearing, then showers. At ARM ENA (Azores), "widespread open
    cell" cumulus follow a front, sometimes closing into stratocumulus
    within a day, under leftover frontal cirrus (meas) [P]:
    https://lasso-ena.svcs.arm.gov/latest/enacase_20161022.html
- **Rate of visible change:** an overcast thickens and lowers over about
  12-30 h (est). That is two to three orders of magnitude slower than a
  30-120 s game lerp.

### 5.2 How sims and games transition weather

- **X-Plane 12** [P, Laminar, updated 2026-01-05]:
  https://developer.x-plane.com/article/a-metar-does-not-tell-you-the-sims-weather/
  - A coarse GRIB background, with METARs overlaid as point samples that
    fade with distance and age.
  - Per-airport noise stops "neat cylinders", and grid boundaries are
    smoothed.
  - The weather always blends a "before" and an "after" state. On arrival,
    "after" becomes "before", and a precomputed next state takes its place.
  - This is the closest documented analogue to a server field plus seeded
    fallback.
- **MSFS [S]:** Meteoblue on a reported 30 km grid in hourly steps, blended
  with METARs. The blend is undocumented.
- **DCS [S, 2011 dev post]:**
  - Pressure is the primary field: cyclones and anticyclones are Gaussian
    deviations, each with a centre, spread, ellipticity, rotation and
    excess.
  - Wind follows the isobars. Cloud comes from pressure, with rain along
    the fronts.
  - Players report the systems drift.
- **Sea of Thieves [S]:** one storm object on a fixed looping path, about
  4 d 4 h per cycle, synchronised across servers.
- **BotW [S]:** per-region weather rolled every 4 in-game hours.
- **trueSKY [P]:** keyframe interpolation; two close keyframes give a fast
  transition.
- **Far Cry 6** (GDC 2022) [A]: transitions between arbitrary states within
  a continuous day, including drying after storms. The talk is paywalled.
- **Borderlands 4** (GDC, 2026-03-13) [A]: "fully dynamic and localized
  weather", with artists steering forecasts. Contents unknown.
- **Research:** Vimont et al., "Interactive Meso-scale Simulation of
  Skyscapes", CGF 39(2), 2020 [A]. A layered atmosphere with a cloud-form
  classification that upsamples cells into volumetric clouds; it reproduces
  sea breezes, convective cells and orographic cloud. **This is the closest
  published match to a coarse server field feeding a cloud synthesiser.**
  https://people.cs.uct.ac.za/~Jgain/wp-content/papercite-data/pdf/vimont2020.pdf

### 5.3 For us

- `CloudParams::weather_at` holds seeded states for `weather_minutes` and
  eases each into the next over the last 40% [C]. That is the industry
  lerp. The population softens it, since clouds already up keep living, but
  the field still steps between states.
- **Proposal: fronts as moving parametric objects** (rework doc §3.1),
  defined here:
  - a polyline, a speed (warm 5-8 m/s, cold 11-13 m/s), a type, and a
    cross-front profile;
  - that profile maps the signed distance x (km, ahead positive) to layer
    coverage, kind weights, base and precipitation.

  | x from the warm front | Sky |
  |---|---|
  | +1,000 to +600 | Ci, thickening |
  | +600 to +400 | Cs with halo (games doc R11 halo turns on) |
  | +500 to +300 | As, base lowering |
  | +300 to 0 | Ns, rain, low scud (living doc item 9) |
  | warm sector | thin Sc and Cu |
  | cold front ±5 km | Cb line (§4 cells) |
  | behind | clearing within 50-100 km, then open-cell Cu showers |

- The client evaluates x per weather cell, or per layer for the fallback.
  Nothing is lerped: the front moves, and each cell reads its own distance.
- **Time compression.**
  - A real sequence takes 12-30 h. Run fronts on `weather_minutes` time,
    for example a whole warm-front passage in 60-90 min of play (est).
  - Keep convection on the cloud clock (§3, conflict note), so cumulus do
    not time-lapse.
- **Owner:** the server (a front list, under 1 KB, plus speeds). The
  fallback seeds 0-2 fronts per scene day.

**Cost:** 0 GPU. As and Ns are existing layers re-parameterised: the
stratocumulus slot as an Ns deck, and the altocumulus slot as As (games doc
R17).

---

## 6. Small-scale realism cues

- **Edge character by kind** (WMO Cloud Atlas, 2017 [P]):
  - **Cumulus:** "dense and with sharp outlines", "brilliant white" in sun,
    and a base "relatively dark and nearly horizontal"; "sometimes ragged".
  - **Altocumulus:** elements "show sharp outlines when separate", which is
    evidence of water droplets. Once ice forms and the droplets evaporate,
    they "cease to present sharp outlines".
  - **Cirrus:** "delicate filaments", fibrous, a silky sheen.
  - **The rule: liquid is sharp, ice is soft and fibrous.** That applies to
    glaciated Cb tops and anvils too.
  - Pages: https://cloudatlas.wmo.int/en/cumulus-cu.html ;
    https://cloudatlas.wmo.int/en/physical-constitution-altocumulus.html
- **Measured edge widths.**
  - The cumulus margin is within about **200 m, or about 10% of the cloud's
    diameter (meas, 1,624 aircraft penetrations)**. Water content falls
    slowly, then sharply at the edge. Wang, Geerts & French, JAS 66(12),
    2009 [A].
  - The width from clear air to optical depth 2 is about **80 m in polluted
    and 120 m in clean** cumulus (meas, Azores) [S, ARM 2012].
  - Internal transitions are under 10 m (meas) [S].
  - **Koren 2007's "twilight zone":** up to 60% of "clear" pixels are humid
    in-between air (pub), reaching several km [S]. That is a faint haze
    shell round cloud fields, not a sharp edge.
- **For us.**
  - `EDGE_GAIN` 3 gives one ramp for every kind [C].
  - Make the ramp width per kind and phase:
    - cumulus at about 10% of D (from the envelope: the profile gradient
      already scales with D);
    - altocumulus sharper (5%);
    - ice (cirrus, anvils, overshooting tops after glaciation) soft;
    - decaying cumulus wisp-wide (the Nubis³ rule, §1).
  - The twilight zone can be a small aerosol increase in the aerial
    perspective term where the envelope's far mip shows cloud (rework doc
    §4.4). ~0 cost.
- **Silver lining.** Thin edges scatter forward with one or two droplet
  encounters, while the core attenuates [S, Atmospheric Optics]. Covered
  by the games doc R7: HG+Draine (Jendersie & d'Eon 2023, which matches 95%
  of Mie over 5-50 µm droplets [A]). Smaller droplets in decaying or
  polluted cloud change the forward lobe, so the decay phase could lower d.
- **Crepuscular rays.** The beams are parallel; their convergence on the sun
  and the antisolar point is perspective [S]. Covered by the games doc R12.
- **Cloud shadows.**
  - Under thick cumulus, global irradiance falls to **about 10% of clear
    sky**, against about 90% under thin cirrus (pub, model) [S]:
    https://arxiv.org/pdf/2307.06980
  - In shadow, direct sun is essentially gone and only diffuse remains.
  - **Beside a shadow, irradiance can rise up to 20% above clear sky (meas)
    [S]** (cloud enhancement, 2025 preprint):
    https://egusphere.copernicus.org/preprints/2025/egusphere-2025-5808/
  - Shadow motion is measurable from the ground (sensor errors 1.9 m/s and
    22°, meas, Fung et al., AMT 2014 [A]). Shadow speed should equal the
    cloud-level wind (est).
  - **For us:**
    - the ground shadow should scale only the direct term;
    - add a per-frame enhancement factor: when the sun is unoccluded and
      the map's cloud cover within about 10-20° of the sun is high, raise
      diffuse by up to 20% (est). That is one CPU value from the map, ~0
      GPU.
- **Dark bases.** Thickness, and with it optical depth, sets how much light
  reaches the base, so tall Cu and Cb have the darkest bases [S; "optically
  thick, so the bases are dark", Appl. Opt. 2015].
  - For us: the games doc R3's base AO, scaled by each cloud's own top
    (envelope G). Congestus then darken under, while humilis stay grey-white.
- **Engines.**
  - UE's "Sky Light Cloud Bottom Occlusion" is a cheap base darkening [P].
  - Nubis³ adds new approximations for "dark edges and inner glow" [A].
- **Perception.** **No cloud-specific perception study was found.**
  - Rademacher et al., EGWR 2001 [A], tested which factors make an image
    read as photo or CG; soft shadows and rough surfaces raised realism
    [R].
  - A paired-comparison test of clear-sky models (arXiv 1612.04336 [A]) is a
    usable template for our Cycles reference harness.

---

## 7. Performance techniques, newer than the earlier docs

**No primary source with measured volumetric-cloud costs on an Adreno 7xx or
a Quest was found.** That stands from the earlier docs. What transfers:

- **(a) Balance slices by cost, not by rows.**
  - UE's real-time sky capture is time-sliced over 9 frames, and the docs
    note it "is only as expensive as its most costly frame" [P].
  - Ours slices 8 rows a frame [C]. Rows with storms or horizon banks cost
    several times more than clear zenith rows (est).
  - Record per-row sample counts from the last build (a counter is enough)
    and give each frame an equal share of samples.
  - The average cost is unchanged, but the **worst frame drops**, which is
    what matters at 13.9 ms.
  - Cost ≈ 0 (CPU scheduling, est).
- **(b) Head-direction priority.**
  - Rebuild rows and azimuth spans near the head's forward direction more
    often, with a floor for the rest. The Quest 3 has no eye tracking, so
    head direction is the proxy.
  - **No published source;** this is our proposal.
  - It fits a direction-space map naturally. Only the scheduling changes,
    not the shader.
  - Gain: fresher shape change where the player looks, at equal cost, or
    the same freshness at lower cost (est).
- **(c) Profile-weighted erosion, so cores skip detail.**
  - Moinet & Neyret (CGF 44, Eurographics 2025 [A]) treat fBm octaves as
    nested bounds: up to **2× faster on non-constant volumes and 12× on
    opaque ones (pub)**. UE's "Conservative Density" is the production
    version [P].
  - In our `density`, the detail cut can reach 0.95, so a bound on our
    constants rarely saves the fetch [C, est].
  - Nubis weights detail by the dimensional profile anyway: edges eroded,
    cores dense.
  - Weight the cut by `(1 − p)^k` and **skip the detail fetch where the
    weight is under about 0.02**.
  - Estimated **−20 to −40% of detail fetches, about −0.05 to −0.15 ms**.
    It is also the more realistic look (§1, §6).
- **(d) ASTC "sliced 3D" for the noise.**
  - WebGPU `texture-compression-astc-sliced-3d` is on **98.3% of Qualcomm
    Android devices (pub, telemetry)** [P]:
    https://web3dsurvey.com/webgpu/features/texture-compression-astc-sliced-3d/platform/Android
  - wgpu exposes `TEXTURE_COMPRESSION_ASTC_SLICED_3D` [S].
  - ASTC 4×4 is 8 bits per texel against 32 for RGBA8. Our 64³ noise drops
    from 1 MB to 256 KB (est).
  - True 3D-block ASTC (`VK_EXT_texture_compression_astc_3d`, Vulkan 1.4.337,
    Dec 2025) shows **0 Android devices** on gpuinfo (pub) [P].
  - Gain: a better texture-cache hit rate on the march's dominant fetch.
    Unmeasured, perhaps −5 to −15% of the march (est).
  - Cheapest test: query `adapter.features()` on the Quest 3, then A/B the
    noise in bench mode.
  - Caveats: LDR only, and four channels compressed together lose some
    independence. Check the twin tolerance.
- **(e) A sun-aligned light-density grid (Nubis³ "light sampling
  acceleration").**
  - From the PlayStation Blog (2023-03-29) [P]: lighting was "the costliest
    part", and Vos's method made both frame-rate modes possible with
    longer-range cloud-to-cloud shadows.
  - A student reimplementation precomputes a 256×256×32 grid of density
    accumulated toward the sun, reporting **about 30-40% less render time
    (pub, GPU not named)** [P README]:
    https://github.com/YueZhang1027/CIS5650-Final-Project-Frostnova
  - For us it is the 3D form of the games doc R6 Beer shadow map: one
    trilinear fetch replaces the 3 long light taps.
  - It stays correct for self-shadowing inside thick cumulus at a grazing
    sun, which a 2D map loses.
  - Example: 128×128×16 R16F over about 24 km (512 KB), rebuilt
    time-sliced within the 96-frame cycle and advected like the map. Keep
    the short tap for fine self-shadow.
  - Estimated **−20 to −35% of the march**, with the build about equal to
    today's ground shadow. Choose this or R6, not both.
- **Monochrome extinction** (UE "Gray Scale Material" [P]). Ours is already
  grey density [C], so there is nothing to gain.
- **A single multiple-scattering octave** (UE's advice [S]). The games doc
  already covers it.
- **Measured on mobile.** Ghaemian (80.lv, 2023): Adreno 540, about 300 ms
  unoptimised to about 40 ms; 60 fps at a 320×180 target with 64 steps and
  2 light steps (meas) [P]. This confirms that resolution and step count
  dominate. Our fixed map size already captures that.
- **Neural shading** (Olajos, Doggett & Goswami, HPG 2026 [P slides]):
  - mesh plus triplane plus MLP, no ray march;
  - as low as **3.7 ms (pub)**, desktop GPU;
  - trained per cloud for about 20 min on an A100.
  - **Does not fit** procedural, advected, population clouds.
  - MRPNN (SIGGRAPH 2023) runs at 40 Hz on an RTX 2080 (pub) [S]: far
    over budget.
- **Not found:**
  - an MSFS 2024 or Frostbite 2024-25 cloud talk;
  - HoYoverse, Kuro, Infold or Tencent mobile cloud talks;
  - a Meta Quest sky blog;
  - change-detection skipping or gaze-priority sky caches.
  - Lead to check: "Dynamic Environment Lighting in Sky: Children of the
    Light" (GDC 2026) [S, unread].

**Budget across this report (est):**

| Item | GPU ms |
|---|---|
| §7 (c) + (d) + (e) | −0.15 to −0.45 |
| §1 convection | +0.01 to +0.04 |
| §2 shear | ~0 |
| §6 cues | ~0 |
| §4 storms, only when in view | +0.05 to +0.2 |

**The net stays negative on fair days and about neutral in storms.**
Measure (c) and (e) first, with pass timers.

---

## 8. What could not be verified

- **Nubis slides.**
  - The Nubis Evolved and Nubis³ text was read by a research agent, not by
    me. The Nubis³ copy is the speaker's CloudFront PDF, which may differ
    from the Advances version.
  - The GDC 2022 superstorm talk is abstract-only (GDC Vault).
  - The GPU Pro 7 shear and anvil code (`cloud_top_offset` 500,
    `anvil_bias`) is recollection.
- **Hourly cloud-base heights.** No primary hourly table was found. The
  100-200 m/h rise, the 0.8-1.1 km onset base and the 1.5-2.0 km late
  afternoon base are estimates consistent with the SGP composites and the
  LCL rule. The SGP figures (11:25 CST onset, 23% peak) are one site's
  climatology.
- **Convection.**
  - Blyth 1993, Damiani/Vali and the classic photogrammetry (Malkus &
    Scorer, Saunders) gave no retrievable numbers.
  - No visible edge-retreat rate was found (about 100 s is an estimate).
  - The 7-minute cap roll-over is derived from one radar case's vorticity.
- **Shear.** Mid-tropospheric shear (2-5 m/s per km) is recollection. The
  tilt formula `S·h²/(2w)` and its 0.5 momentum factor are my derivation,
  not a published model.
- **Cumulonimbus.** Gaps:
  - an observed downwind anvil length;
  - rain-shaft width;
  - the stage durations (a university text only);
  - mammatus lobe numbers (uncited Wikipedia);
  - the method behind Nubis's "near-zero cost" internal lighting.
- **Fronts.** Gaps:
  - a primary cirrostratus distance;
  - cold-front band width beyond "a few hundred km";
  - the internals of RDR2, Ghost of Tsushima, Kingdom Come, Assassin's
    Creed, Borderlands 4 and MSFS weather transitions;
  - warm-front speeds (still from a low-quality source, rework doc).
- **Cues.**
  - No cloud-specific perception study.
  - No measured cloud-shadow ground speed.
  - The Koren twilight-zone reach is secondary only.
  - Ghost of Tsushima's depth-dependent phase was not re-verified.
- **Performance.**
  - No Adreno or Quest cloud measurement exists in any source found.
  - ASTC HDR on the Adreno 740 under wgpu is unconfirmed.
  - The light-grid saving comes from a student project on an unnamed GPU.
  - Moinet & Neyret's speedups are abstract-only.
- **Ours.** Every Quest cost in this report is an estimate from sample
  counts. The march baseline (0.6-0.8 ms) is inferred from one A/B, not a
  pass timer.

## 9. Sources (beyond the earlier docs; dates as published; accessed 2026-10-21)

Convection
- Romps, Charn, "Sticky thermals", JAS (2015-08-03): https://www.osti.gov/servlets/purl/1511414
- ARM, stereo photogrammetry of cloud tops (2019-02-19): https://arm.gov/news/features/post/53490 ; Romps & Öktem GRL 2015: https://arm.gov/research/highlights/621
- Hernández-Deckers, Sherwood, JAS (2016): https://www.researchgate.net/publication/305454073
- TRACER LES thermals, ACP 26, 10071 (2026-07-17): https://acp.copernicus.org/articles/26/10071/2026/
- Wang, Geerts, cumulus toroidal circulation, JAMC 54(10) (2015): https://researchconnect.suny.edu/en/publications/vertical-plane-dual-doppler-radar-observations-of-cumulus-toroida/
- CuPIDO field summary (about 2006): https://archive.eol.ucar.edu/projects/cupido/documents/CuPIDO_SOD.html
- Heus, Jonker, subsiding shells, JAS (2008): https://engagedscholarship.csuohio.edu/sciphysics_facpub/229
- Bridson, Hourihan, Nordenstam, "Curl-Noise for Procedural Fluid Flow", SIGGRAPH 2007: https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf
- Gustavson, McEwan, "Tiling simplex noise and flow noise in two and three dimensions", JCGT 11(1) (2022): https://jcgt.org/published/0011/01/02/paper-lowres.pdf
- Schneider, Nubis³ slides (2023-08-05): https://d3d3g8mu99pzk9.cloudfront.net/AndrewSchneider/Nubis%20Cubed.pdf ; https://www.guerrilla-games.com/read/nubis-cubed
- Schneider, Nubis Evolved slides (2022-08): https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-NubisEvolved-NoVideos.pdf

Shear and diurnal cycle
- Helfer et al., shear and trade cumulus (2020-12): https://pmc.ncbi.nlm.nih.gov/articles/PMC7757185/
- Cecchini et al., ACP 22, 11867 (2022-09-14): https://acp.copernicus.org/articles/22/11867/2022/
- ERA5 shear near the tropopause, WCD 2, 631 (2021): https://wcd.copernicus.org/articles/2/631/2021/
- Thompson et al., SPC RUC-2 soundings (2003): https://origin-east-www-spc.woc.noaa.gov/publications/thompson/rucparms.pdf
- Cirrus uncinus generating cells (1975): https://impacts.ucar.edu/en/publications/cirrus-uncinus-generating-cells-and-the-evolution-of-cirriform-cl-5/ ; ice fall speeds from Doppler lidar: https://centaur.reading.ac.uk/1983/
- Unity HDRP volumetric clouds source: https://raw.githubusercontent.com/Unity-Technologies/Graphics/master/Packages/com.unity.render-pipelines.high-definition/Runtime/Lighting/VolumetricClouds/VolumetricClouds.cs
- ARM SGP shallow cumulus composite poster (2017-03): https://asr.science.energy.gov/meetings/stm/posters/poster_pdf/2017/P001693.pdf
- Berg, Kassianov, SGP fair-weather cumulus (2006): https://arm.gov/publications/proceedings/conf16/extended_abs/berg_lk.pdf
- Zhang, Klein, SGP shallow cumulus (2013): https://asr.science.energy.gov/meetings/stm/2013/presentations/Klein-sgp-shcu.pdf
- Fog dissipation after sunrise, ACP 25, 13493 (2025-10-23): https://acp.copernicus.org/articles/25/13493/2025/
- Wood, "Stratocumulus clouds", MWR (2012): https://www.chem.pmf.hr/_download/repository/Review_Sc_clouds_R_Wood_MWR_2012.pdf
- Dai, diurnal thunderstorm frequencies, J. Climate (2001): https://impacts.ucar.edu/en/publications/global-precipitation-and-thunderstorm-frequencies-part-ii-diurnal/
- trueSKY cloud keyframer: https://docs.simul.co/4.4/Reference/Clouds/cloudkeyframer.html

Cumulonimbus
- NWS Spotter Guide, thunderstorm life cycle: https://www.weather.gov/spotterguide/life
- WMO International Cloud Atlas, Cb and its supplementary features: https://cloudatlas.wmo.int/en/cumulonimbus-cb.html ; https://cloudatlas.wmo.int/en/supplementary-features-and-accessory-clouds-cumulonimbus.html
- EUMETSAT overshooting tops (about 2009): https://cwg.eumetsat.int/?p=135 ; GOES-R overshooting-top ATBD: https://www.goes-r.gov/products/ATBDs/option2/Aviation_OvershootingTop_v1_no_color.pdf
- CloudSat anvils: https://espo.nasa.gov/arctas/node/30056 ; CRM anvil spreading (about 2005): https://hogback.atmos.colostate.edu/cmmap/research/docs/jul05/krueger-poster.pdf
- Schultz et al., "The mysteries of mammatus clouds", JAS 63(10) (2006): https://digitalcommons.mtu.edu/michigantech-p/12780
- Lightning flash rates (NASA): https://ntrs.nasa.gov/api/citations/19990004337/downloads/19990004337.pdf ; severe-storm rates (2025): https://arxiv.org/pdf/2504.20559 ; Peterson & Mach (2021): https://www.osti.gov/pages/biblio/1855136
- Unity HDRP 17 Volumetric Clouds reference: https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@17.0/manual/volumetric-clouds-volume-override-reference.html
- Hao et al., "Thunderscapes" (arXiv 2024-12, v4 2025): https://arxiv.org/html/2412.00703v4
- PlayStation Blog, Burning Shores clouds (2023-03-29): https://blog.playstation.com/2023/03/29/pushing-the-envelope-achieving-next-level-clouds-in-horizon-forbidden-west-burning-shores/
- Dobashi et al., lightning with multiple scattering (2007): https://ime.ist.hokudai.ac.jp/~doba/projects/lightning_ms/lightning_ms.htm

Fronts and transitions
- Penn State METEO 3 on warm fronts: https://courses.ems.psu.edu/meteo3/node/2015
- Oklahoma Climatological Survey, fronts: https://okfirst.mesonet.org/train/meteorology/Fronts.html
- UBC ATSC 113, stratiform clouds (2020): https://www.eoas.ubc.ca/courses/atsc113/flying/met_concepts/01-met_concepts/01a-clouds/stratiform.html
- EUMeTrain SatManu, cold-front structure: https://rammb.cira.colostate.edu/wmovl/VRL/Tutorials/SatManu-eumetsat/SATMANU/CMs/Cf/structure.htm
- AMS Glossary, prefrontal squall line: https://glossary.ametsoc.org/wiki/Prefrontal_squall_line
- ARM ENA LASSO post-frontal case: https://lasso-ena.svcs.arm.gov/latest/enacase_20161022.html
- X-Plane, "A METAR does not tell you the sim's weather!" (2026-01-05): https://developer.x-plane.com/article/a-metar-does-not-tell-you-the-sims-weather/
- DCS dynamic weather overview (2011): https://forum.dcs.world/topic/57317-dynamic-weather-overview/
- Far Cry 6 tropical weather (GDC 2022): https://www.gdcvault.com/play/1027675/Simulating-Tropical-Weather-in-Far
- Vimont et al., "Interactive Meso-scale Simulation of Skyscapes", CGF 39(2) (2020): https://people.cs.uct.ac.za/~Jgain/wp-content/papercite-data/pdf/vimont2020.pdf

Cues
- WMO Cloud Atlas, cumulus, altocumulus constitution and cirrus: https://cloudatlas.wmo.int/en/cumulus-cu.html ; https://cloudatlas.wmo.int/en/physical-constitution-altocumulus.html ; https://cloudatlas.wmo.int/en/cirrus-ci.html
- Wang, Geerts, French, cumulus margin, JAS 66(12) (2009): https://researchconnect.suny.edu/en/publications/dynamics-of-the-cumulus-cloud-margin-an-observational-study/
- Koren et al. 2007, the twilight zone: https://earthref.org/ERR/110842/
- Cloud-shadow irradiance: https://arxiv.org/pdf/2307.06980 ; cloud enhancement (2025 preprint): https://egusphere.copernicus.org/preprints/2025/egusphere-2025-5808/ ; Fung et al., AMT 7 (2014): https://amt.copernicus.org/articles/7/1693/2014/
- Atmospheric Optics, silver linings and crepuscular rays: https://www.atoptics.co.uk/fz264.htm ; https://atoptics.co.uk/fz654.htm

Performance
- Moinet, Neyret, "Fast Sphere Tracing of Procedural Volumetric Noise…", CGF 44 (Eurographics 2025): https://hal-bioemco.ccsd.cnrs.fr/LJK_GI_MAVERICK/hal-05046040v1
- UE5 Volumetric Cloud component and properties (UE 5.8 docs): https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-cloud-component-properties-in-unreal-engine
- web3dsurvey, ASTC sliced 3D on Android: https://web3dsurvey.com/webgpu/features/texture-compression-astc-sliced-3d/platform/Android ; gpuinfo, `VK_EXT_texture_compression_astc_3d`: https://vulkan.gpuinfo.org/listdevicescoverage.php?extension=VK_EXT_texture_compression_astc_3d&platform=android
- CIS5650 Frostnova, Nubis³ reimplementation: https://github.com/YueZhang1027/CIS5650-Final-Project-Frostnova
- Olajos, Doggett, Goswami, "Environmental Volumetric Neural Shading of Clouds", HPG 2026: https://www.highperformancegraphics.org/2026/_assets/slides/environmental_volumetric_neural_shading_of_clouds_for_real-time_rendering.pdf
- Ghaemian, mobile clouds in UE5, 80.lv (2023-03-30): https://80.lv/articles/creating-volumetric-clouds-for-mobile-in-unreal-engine-5
