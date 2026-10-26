# Soulframe's sky, weather and sun glare, and how we match them on Quest 3

Research date: 2026-10-20. Web research only. No code was edited and nothing was downloaded on purpose.
One PDF was cached automatically by the fetch tool and could not be read because there is no PDF renderer.

Labels used below:
- **[confirmed]**: a dated source says it.
- **[secondary]**: press or a wiki says it, not Digital Extremes (DE).
- **[inference]**: my reasoning. No source says it.
- **[unconfirmed]**: I looked and found nothing.

---

## 0. Bottom line

1. **There is no public technical write-up of Soulframe's sky.** I found no GDC talk, no SIGGRAPH talk, no Digital Foundry analysis and no Devstream segment on its sky, clouds or sun glare. What is public:
   - the engine (Evolution, with the deferred "Enhanced" renderer);
   - the shape of the day/night cycle;
   - the set of weather types added in Preludes 16 (2026-09-18);
   - a stream of patch notes on lighting and volumetric fog.

   Everything about how its sky *looks* has to come from the user's own captures of the game. Section 1.4 lists what to look for.
2. **What to match from Soulframe:**
   - one server-driven day wheel with named moments;
   - weather as a few gameplay-coupled states (rain, wind, fog, lightning storm) that combine with the time of day;
   - scripted storm overrides;
   - volumetric fog that is the same medium as the sky's aerial perspective;
   - glare tuned by an artist rather than left at a physical default. DE patched out "excessive glare" in a cutscene.
3. **Sun glare recommendation.** Reuse our lamp glare pipeline. Treat the sun as one more halo-only `GlareSource` at infinity, drawn over geometry as well as sky. Add three things:
   - **visibility:** how much of the sun shows, measured exactly per eye by the sky pass itself (all geometry, clouds included);
   - **a pupil "squint":** a fast exposure dip of up to about 1.2 stops while the sun is near the line of sight;
   - **optionally, a faint ciliary-corona streak texture and lid shading.**

   Expected cost is about **0.10-0.20 ms per frame while the sun is in view**, and about 0 when it is not. Section 3.4 explains the estimate.
4. **Do not add camera lens-flare ghosts.** They are a camera artefact, not an eye effect. In stereo they are the classic failure: several Unity XR bug reports show flares in the wrong place per eye, doubled, or in one eye only.

---

## 1. Soulframe: sky, day/night, weather and fog

### 1.1 Engine and renderer
- **[confirmed]** Soulframe runs on DE's Evolution engine, shared with Warframe. Creative Director Geoff Crookes said improvements to the technology can benefit both games.
  - Sportskeeda interview (pre-2024): https://www.sportskeeda.com/mmo/news-something-bit-slower-almost-poetic-digital-extremes-geoff-crookes-talks-soulframe-gameplay-community-first-approach
  - Wikipedia: https://en.wikipedia.org/wiki/Soulframe
- **[confirmed]** Warframe's "Enhanced Graphics Engine" (2020-10-06):
  - It writes scene attributes (colour, glossiness, shape) to offscreen buffers, so dynamic lights cost about as much as the old glow sprites.
  - Shadows from light sources are cached and reused when nothing changes.
  - Features carried over from the legacy renderer: static path-traced lighting, parallax-corrected cubemaps, volumetric lights, lit fog and character light probes.
  - Enhanced has been the default since The Duviri Paradox.
  - Sources: https://www.warframe.com/en/amp/enhanced-graphics-engine and https://www.pcgamesn.com/warframe/enhanced-graphics-engine
- **[confirmed, patch notes]** Soulframe Preludes 11 (2025-10-10): "SSAO traces now contribute light bounce for added scene illumination". That is screen-space short-range GI. https://www.midrath.com/patch-notes (a community mirror of the official notes)
- **[secondary]** Players say Soulframe has its own volumetric lighting and GI, "same as Warframe's recent tilesets". Forum: https://forums.soulframe.com/topic/2941-graphical-glitch-with-sky-and-volumetric-fog/

### 1.2 Day/night cycle
- **[secondary, fan wiki, edited 2026-10-13]** The cycle follows two suns, **Mora** and **Saphene**, and a glowing moon, **Iridis**.
  - A day is about **32 real minutes** and a night about **16**.
  - The game announces three moments about every 16 minutes: "Saphene Glows" (dawn), "Mora Rises" (noon) and "Iridis Awakens" (nightfall).
  - The sky runs on the **server clock**, so every player sees the same time.
  - Some spawns are gated by time of day, for example the Vadagar Bear at night (Preludes 13, 2026-02-13).
  - Sources: https://wiki.avakot.org/Time and https://www.midrath.com/patch-notes
- **[confirmed]** Devstream 14's goals for Preludes 15 include "reducing the feeling of a looping day". DE treats a repetitive day as a problem. https://www.soulframe.com/en/news/10000070/soulframe-devstream-14-overview
- **[secondary]** Players report that the day feels too long and the night too short. https://forums.soulframe.com/topic/7466-daynight-cycle-beastiary/
- **[unconfirmed]**
  - how the two suns are drawn, and whether two discs are ever visible at once;
  - whether the sky model is physically based (LUT atmosphere) or artist-keyed gradients.

### 1.3 Weather
- **[confirmed]** Preludes 16, "Of Hook & Hound", released 2026-09-18 with a **dynamic weather system**. From the official preview (2026-09-04):
  - "Dynamic weather is another step toward making Alca feel more immersive to live in!"
  - Lightning storms empower Voltaic damage.
  - Flowers grow more after rain.
  - Source: https://www.soulframe.com/en/news/10000080/soulframe-preludes-16-at-warframes-devstream-197
- **[secondary]** The weather types are rain, wind, fog and lightning, each with a gameplay effect: resources, cover, perception and Voltaic damage. Outlets describe combinations such as "rainy nights, foggy mornings, scenic sunsets".
  - Gameluster (2026-09-20): https://gameluster.com/soulframe-preludes-16/
  - Final Weapon (2026-09-19): https://finalweapon.net/2026/09/19/soulframe-preludes-16-of-hook-hound-available-now/
  - Gaming Trend: https://gamingtrend.com/news/soulframe-drops-preludes-16-of-hook-hound/
- **[confirmed, patch notes]** Weather existed as **scripted story state** before the dynamic system:
  - a "Cloudquell" item restores "the original time and weather" (Preludes 7 Hotfix 4: https://www.soulframe.com/en/patch-notes/0-7-4);
  - Midrath's skies stay **stormy** until a boss (Nimrod) is beaten with a Cloudquell (Preludes 13 Hotfix 6).
- **[secondary]** Devstream 16 (TennoCon, 2026-07-11) showed "a brief snippet of gameplay, with weather changes". https://lordsofgaming.net/2026/07/tennocon-2026-soulframe-devstream-16-breakdown/
- **[unconfirmed]** how weather transitions are rendered: cloud cover ramps, wet surfaces, rain particles, lightning flash lighting.

### 1.4 Fog, aerial perspective, god rays and glare
- **[confirmed, patch notes]** Volumetric fog is the most-patched rendering system. Almost every Prelude says "Continued updates to lighting, VFX, materials, environment, and volumetric fog". Specific fixes include:
  - "flickering in volumetric **and fallback fog when the sky was visible**" (Preludes 12, 2025-11-25);
  - fog density together with **contact shadows** (Preludes 13 Hotfix 1);
  - volumetric fog with DLSS/FSR (Preludes 11 Hotfixes 2-3);
  - fog density too high in the Undercity (Preludes 12 Hotfix 12).

  Source: https://www.midrath.com/patch-notes
- **[secondary]** Turning volumetric fog off visibly changes the *sky*, and glitches it at night. Source: the 2941 forum thread above.
  - **[inference]** So the sky's haze and the froxel fog are one system. Aerial perspective probably comes from the fog volume, not a separate sky-only term.
- **[confirmed]** Preludes 14 (2026-04-17): "Fixed excessive glare in Lady Deora's childhood cutscene". So Soulframe has a glare/bloom effect, and DE tunes it per scene.
- **[unconfirmed]**
  - god rays or light shafts (Warframe's legacy renderer had "volumetric lights");
  - lens flares;
  - an explicit "squint" or eye-adaptation glare;
  - any options menu entry for these.

  None was found in settings guides or patch notes.

### 1.5 Art direction (why it reads as impressive)
- **[confirmed]** The stated influences are **Princess Mononoke, Studio Ghibli and Ico**.
  - GameSpot: https://www.gamespot.com/articles/soulframe-is-heavily-inspired-by-princess-mononoke-studio-ghibli-and-ico/1100-6517404/
  - Game Informer cover story (2026-03-27): https://gameinformer.com/cover-reveal/2026/03/27/cover-reveal-soulframe
  - The art director is Rosa Lee (Wikipedia).
- **[secondary]** Players say it "looks nicer than most AAA games" (Preludes 11, MeinMMO, 2025-10-14): https://mein-mmo.de/en/soulframe-preludes-11-release-patch-notes-founders,1531707
  - A press reviewer says it is not aiming to be a graphical benchmark and is judged on art direction: https://www.gamersyde.com/game_soulframe_5620_en.html
- **[inference — needs the user's captures to confirm]** What a Mononoke-led look usually means in practice:
  - warm, saturated low sun with cool blue-green shade;
  - layered valley fog and strong aerial perspective that separates forest planes;
  - big cumulus with bright lit tops and soft bases;
  - soft wide bloom around the sun rather than hard flares.

  **Action:** ask the user for 3-5 Soulframe screenshots or clips (dawn, noon, sunset, storm, looking at the sun). Judge our sky against those, not against this paragraph.

---

## 2. How sun glare is done: research and games

### 2.1 The physics (what "squinting into the sun" is)
- **Spencer, Shirley, Zimmerman, Greenberg, "Physically-Based Glare Effects for Digital Images", SIGGRAPH 1995**, pp. 325-334. https://w1.graphics.cornell.edu/pubs/1995/SSZG95.pdf
  - Glare is a point-spread function (PSF) applied to the image, in three parts:
    - **bloom**: the veiling glow, from scattering in the cornea, lens and retina;
    - **ciliary corona**: radial streaks, from particles and fibres in the lens;
    - **lenticular halo**: a coloured ring, from diffraction by the lens fibres. It is seen only with a dilated pupil, so mostly at night.
  - Apply it only to light **above display white**, because the viewer's own eye already glares on what the display shows.
  - Its angular size does not depend on distance.
  - Summary page: https://graphics.stanford.edu/courses/cs348b-competition/cs348b-03/glare/
- **Kakimoto et al., "Glare Generation Based on Wave Optics"** (Pacific Graphics 2004, CGF 24(2) 2005). https://diglib.eg.org/handle/10.2312/CGF.v24i2pp185-193
  - Glare is **Fraunhofer diffraction**, the 2D Fourier transform of the aperture. The apertures are the **pupil and the eyelashes**.
  - Glare images are precomputed into textures for interactive rates.
  - The streaks move and rotate as the aperture moves relative to the light. A squint narrows the aperture and the lashes add **streaks perpendicular to the lash line**.
- **Ritschel, Ihrke, Frisvad, Coppens, Myszkowski, Seidel, "Temporal Glare", Eurographics 2009** (CGF 28(2), 183-192). https://diglib.eg.org/handle/10.2312/CGF.v28i2pp183-192 · project page https://resources.mpi-inf.mpg.de/hdr/TemporalGlare/
  - **[confirmed, abstract]**
    - an anatomical eye model simulated in real time on a GPU;
    - glare **fluctuates over time**;
    - their study found the dynamic glare raises perceived brightness and is usually rated more attractive.
  - **[confirmed]** The project page lists "Billboard Video Texture frames". The glare is a precomputed animated sprite drawn per light, not a full-screen convolution.
  - **[unconfirmed — the PDF returned 403]** The aperture parts (pupil with hippus, lens fibres and particles, lashes) and the FFT step. This is my memory of the paper; check the PDF before quoting it.
  - Pupil hippus, the involuntary fluctuation of the pupil size, has a reference implementation with a Moon-Spencer pupil-size mapping: https://people.compute.dtu.dk/jerf/code/hippus/
- **Yoshida et al. 2008, brightness of the glare illusion.** A Gaussian works about as well as the true PSF perceptually. https://www.cl.cam.ac.uk/~rkm38/pdfs/yoshida08bgi.pdf
  - We already use this (memory `glare-in-the-eye`).
- **VR-specific: Luidolt, Wimmer, Krösl, "Gaze-Dependent Simulation of Light Perception in Virtual Reality", IEEE TVCG 26(12), 2020.**
  - https://www.cg.tuwien.ac.at/research/publications/2020/luidolt-2020-lightperceptionVR
  - thesis: https://www.cg.tuwien.ac.at/research/publications/2020/Luidolt-2020-DA/Luidolt-2020-DA-thesis.pdf
  - **[confirmed]** Eye-tracked, real-time VR with gaze-dependent adaptation and glare. Glare intensity follows the viewing direction and its appearance follows pupil size.
  - The pilot study found that "further work is necessary to improve the appearance and movement of our proposed glare kernel".
  - **Lesson:** an animated, eye-anatomy glare kernel is the hard part to get right in a headset.

### 2.2 What shipped games do
- **Ghost of Tsushima.** Jasmin Patry, SIGGRAPH 2021 Advances, "Lighting, Atmosphere, and Tonemapping". https://www.advances.realtimerendering.com/s2021/jpatry_advances2021/index.html
  - **[confirmed]** Sky: one physically based model drives sky, clouds, haze and fog particles.
    - Rayleigh and Mie in 3D LUTs, with multiple scattering and an analytic Earth shadow.
    - The LUTs are resampled each frame for the sun and moon angles.
  - **[confirmed]** Clouds: a 768² paraboloid map, time-sliced over 60 frames.
  - **[confirmed]** Haze: a **128×64×64 froxel grid from 10 cm to 100 km**, about 0.5 ms async on a base PS4.
  - **[confirmed]** Exposure is **illuminance-based**. Luminance is used only for highlights above a threshold.
  - **[confirmed]** Tonemapping and colour:
    - a **bilateral-grid local tonemapper** (about 250 µs);
    - an ACEScg-derived colour space;
    - artist white balance;
    - a **Purkinje** night shader.
  - **[unconfirmed]** The talk does not cover sun glare or bloom.
- **Ghost of Yōtei.** PlayStation Blog tech deep dive, 2025-10-23. https://blog.playstation.com/?p=411201
  - **[confirmed]** Clouds are drawn in front of geometry, so mountains can sit in cloud.
  - **[confirmed]** A light-space **cloud shadow map makes crepuscular rays in the volumetric fog**, and artists place "god ray targets".
  - **[confirmed]** Local fog volumes.
- **Horizon Zero Dawn.** Andrew Schneider, SIGGRAPH 2015, "Real-time Volumetric Cloudscapes". https://www.guerrilla-games.com/read/the-real-time-volumetric-cloudscapes-of-horizon-zero-dawn
  - **[confirmed]** Clouds lit correctly through the day, on a 2 ms GPU budget.
  - **[unconfirmed]** Its sun flare and bloom are not documented.
- **Unreal Engine.** https://dev.epicgames.com/documentation/en-us/unreal-engine/bloom-in-unreal-engine
  - **[confirmed]** "Convolution bloom" is an FFT with a custom kernel image, which can include streaks and diffraction spikes. Epic positions it for cinematics and high-end hardware. The standard mip-chain bloom "has a significant performance advantage" and is meant for most games.
  - UE also has image-based lens flares (ghosts).
- **Call of Duty: Advanced Warfare** (Jimenez, SIGGRAPH 2014): pyramid bloom with no threshold, a 13-tap downsample and a 9-tap tent upsample.
  - **ARM dual-filter** (Bjørge, SIGGRAPH 2015, "Bandwidth-Efficient Rendering") is the mobile version.
  - Both use the bloom-as-blur family: several full-screen passes.
  - Secondary summaries: https://api.qgis.org/api/classQgsBloomRenderView.html and https://qiita.com/UWATechnology/items/8b0eff5650d2c659c019
- **RDR2, Cyberpunk 2077, Horizon (sun flares)**
  - **[unconfirmed]** I found no dated technical source for their sun glare. Only player forum posts. Do not cite them as references.

### 2.3 What fits a Quest 3 (Adreno 740, tiled, stereo, 13.9 ms)

| Technique | Quest fit | Why |
|---|---|---|
| Full-screen FFT convolution (UE convolution bloom, Spencer convolution) | **No** | Several full-resolution passes with off-tile reads. Epic itself calls it high-end only. |
| Mip-chain or dual-filter bloom | **Poor for us** | A full-frame resolve plus about 10 small passes. A Meta forum engineer (2017) called full-screen blurs on tilers "fantastically expensive". Meta's own docs say bloom breaks render-region optimisation. A full-frame copy was quoted at about 2.5 ms on Quest 1 (old). |
| **Analytic PSF over the source's footprint**, one screened overlay quad (what our lamps do) | **Yes** | It reads only its own pixel, which is as cheap as alpha blending on a tiler. The cost scales with the screen area the veil covers. |
| Per-light precomputed glare sprite (Kakimoto, Ritschel's billboards) | **Yes**, as an extra texture tap inside that quad | One texture fetch per veil pixel. |
| Screen-space radial-blur god rays | **No** | A full-screen multi-tap pass. |
| Volumetric shafts from a cloud shadow map (Yōtei) | Only inside a froxel or analytic fog pass we already run | Section 3.5. |
| Lens-flare ghosts | **No** | Camera artefact. Stereo bugs, below. |

Meta sources:
- https://developers.meta.com/horizon/documentation/unity/unity-openxr-settings-quest/ (bloom versus render regions)
- https://communityforums.atmeta.com/discussions/dev-unreal/go-hdr-less-bloom-/645084/replies/645088 (full-screen blur cost on a tiler)
- https://developers.meta.com/horizon/design/lighting-and-effects/ (only generic comfort advice)
- https://github.com/GodotVR/godot_oculus_mobile/issues/45 (the 2.5 ms full-frame copy, secondhand)

### 2.4 Stereo and comfort
- **[confirmed]** Screen-space lens flares break in XR. Unity's issue tracker has:
  - "Lens Flare (SRP) has an incorrect horizontal position in URP VR/XR";
  - "Lens flare VFX is duplicated when observed through a VR headset" (it shows in both lenses' overlap);
  - Quest 3 users report flares in the left eye only (Unity discussions, 2024).

  Sources:
  - https://issuetracker.unity3d.com/issues/lens-flare-srp-has-an-incorrect-horizontal-position-in-the-urp-vr-xr
  - https://issuetracker.unity3d.com/issues/lens-flare-vfx-is-duplicated-when-observed-through-a-vr-headset
  - https://discussions.unity.com/t/do-lens-flare-srp-currently-works-with-vr-quest-3-2022-3-44/1528181
- **[secondary]** Flight-sim and IL-2 VR players complain that a sun bloom so large the horizon disappears is unpleasant.
  - https://forums.flightsimulator.com/t/how-to-reduce-bloom-from-the-sun/467985
  - https://forum.il2-series.com/topic/4213-sun-glare-simulation/page/3/
  - A Quest UE5 guide (third party) recommends turning lens flare off and keeping bloom: https://www.strayspark.studio/blog/vr-game-development-meta-quest-ue5-2026
- **[inference] Rules for stereo-correct eye glare:**
  1. **Centre it on the source's direction, projected through each eye's own view.** The sun is at infinity, so the veil has zero disparity and both eyes put it at infinity. Glare is in the eye, so drawing it over a nearer trunk is right, not a bug.
  2. **Measure visibility per eye.** A branch can hide the sun from one eye and not the other, and real eyes do the same. Average the visibility over the disc's area and smooth it over about 50-100 ms. Otherwise leaves flicker it at the frame rate, and the two eyes flicker differently, which reads as shimmer.
  3. **Any texture detail (corona streaks) must be the same in both eyes**, keyed to the sun's direction, not per eye. Different per-eye patterns cause binocular rivalry.
  4. **Keep animation slow and low-contrast.** Hippus is roughly 0.1-1 Hz. Luidolt et al. found the animated kernel the weakest part in a headset.
  5. **No ghosts and no screen-anchored elements.**

---

## 3. Recommendation for our renderer

### 3.1 What we have (read from source today)
- **Lamp glare:** `space_soup/src/renderer/glare.rs`, memory `glare-in-the-eye`.
  - CIE 146 veil × `VEIL_SHARE` 0.06, windowed. Drawn as a **HALO** (depth Always, screened, `HALO_CAP` 0.04) plus a **CORE**.
  - `GlareSource { halo_only, .. }` already covers the torch and fire, with no core.
  - Occlusion comes from probe-pass depth taps (brushes only) and character capsules.
  - Empty cells are culled. In-run A/B: about 0.14-0.16 ms for a halo covering 8-18% of the view.
- **Sun:** `sky.rs::sky_sun_disc` already draws the same CIE veil (`sky_halo`, `HALO_DEGREES` 20) with limb darkening, reddening and refraction.
  - The veil runs **in the sky pass only**. The sky pipeline is `LessEqual` at far depth, so the veil is cut along every tree, wall and character outline.
  - `sky-realism-plan-2026-10-20` lists "eye glare over geometry" as not done, and `sky-realism-research.md` §4.3 already plans it.
- **Exposure:** `exposure.rs`.
  - Probe and sky bins are 4×4 per cube face, metered as the 0.7-0.98 percentile band. `bright_view_pull` only acts above 0.003 sr.
  - The sun's disc is about 6.8e-5 sr, so **looking at the sun does not stop the eye down** today. Check this with a one-line print; I read it from the code, I did not run it.
  - `exposure.rs` has no sun term at all.

### 3.2 What to build, in order

**A. The sun veil over everything.** This is the core of "squinting into the sun".
- Add a sun (and moon) source to the existing glare build: `halo_only: true`, at infinity, with the angular core of the disc.
  - It is one quad per eye on the sun's direction, sized to where the windowed CIE veil reaches the floor, capped at 20-25°.
  - Draw it last in the scene pass, screened through the scene's ACES like the lamp halos. It lies over trees, terrain and characters.
- **Remove the halo term from `sky_sun_disc` and the cloud twin** at the same time, so it is not counted twice. The disc stays in the sky shader.
  - Sky pixels near the sun then cost less than now. The veil moves from per-sky-pixel shader work to a quad that writes only where its value is above the floor.
- Its strength is `E_sun × visible`. `E_sun` already exists in the sky shader as `e_sun`, and `visible` is item B.
- Clouds: `sky_halo_through()` already exists. Item B folds cloud transmittance in automatically.

**B. Exact per-eye sun visibility, from the sky pass. This needs no new pass.**
- The sky fragment shader runs only on sky pixels nothing else covered: `the_sky_only_fills_where_nothing_was_drawn`. Inside the disc branch, which already exists, `atomicAdd` each fragment's solid angle × (1 − cloud alpha) into a small storage buffer slot for that eye.
- `px` from `dpdx` and `dpdy` is already computed, so the sum is foveation-independent.
- Divide by the disc's solid angle and you get the fraction of the sun this eye sees, through **all** geometry: models, foliage, hands, clouds. The lamp path's probe-pass depth sees brushes only.
- The glare quad's vertex shader reads the **previous frame's** value from a double-buffered slot. wgpu has no barrier inside a render pass, so this frame's value cannot be read.
- Smooth it over about 3-6 frames, either in the shader (EMA in the buffer) or on the CPU, if it is read back anyway for the squint.
- **Check first:**
  - Fragment-stage storage writes and atomics on the Quest's Vulkan adapter (`DownlevelFlags::FRAGMENT_WRITABLE_STORAGE` and the per-stage storage-buffer limits).
  - The disc is only about 12 px across at Quest 3 density, so the count is coarse, but exact.
  - Fallback: depth taps in the glare vertex shader like the lamps use, with a capsule test for characters. That fallback misses foliage models.

**C. Pupil "squint": a fast exposure dip.**
- The real reflex: the pupil narrows within about 0.3-1 s when a bright source enters central vision. A 3 mm → 2 mm pupil is about −1.2 stops.
- Compute `S = smoothstep(…, E_sun × visible × w(angle to gaze))`, with `w` about 1 inside 10° and falling to 0 by about 30°.
- Apply `exposure × 2^(−k·S)` with `k ≈ 1-1.2` stops, attack about 0.3 s and release about 1-2 s. This is **separate** from the slow `EyeAdaptation` meter, so it never fights the room logic.
- CPU only, a few float ops. It is the most noticeable cue of the set: the land around the sun darkens while you look at it, and the veil reads brighter because of it.
- **[inference]** The constants are physiology-based starting points, not measured from any game. Tune them on the headset.

**D. Optional, behind a lever, after A-C are judged on the headset.**
- **Ciliary corona:** a small precomputed polar texture of fine radial streaks (Spencer's flare; Kakimoto and Ritschel compute them from the aperture). Sample it in the sun quad within about 5-8° and modulate the veil by 1 ± a small amount.
  - The pattern is the same in both eyes, keyed to the sun's direction.
  - It rotates or breathes slowly with a hippus noise (0.1-1 Hz).
  - Cost: one texture fetch per veil pixel inside 8°.
- **Lid and lash squint:** when S is high, add faint **vertical** streak lobes (lashes diffract perpendicular to the lid line), and optionally darken a soft head-locked band at the top and bottom of the view by at most about 15-20%.
  - Both eyes identical, slow in and out.
  - **[inference]** Head-locked darkening is a known comfort tool (locomotion vignettes), but test it on the user before shipping. A view that narrows by itself can feel like a restriction.
- **Lenticular halo** (the coloured ring): skip by day. The pupil is small, which is why it only shows at night. It could be added for the moon or lamps at night.

### 3.3 What not to do
- No lens-flare ghosts or sprites anchored to the screen.
- No full-screen bloom or convolution pass for the sun. We already settled on analytic PSF quads for the lamps, for the same cost reasons.
- Do not cut the veil by depth. That was the lamps' first fault: glare cut along nearer outlines.

### 3.4 Expected Quest cost
**[inference, from our own lamp measurements, not a sun measurement]**
- **A:** a 20° veil covers π·20² ≈ 1,260 deg², about 12% of an eye's view of roughly 100°×100°. The lamp halos cost 0.14-0.16 ms at 8-18% of the view, and those shaders are heavier (air grid, table).
  - Sun veil: about **0.08-0.15 ms** while the sun is in view, 0 when it is behind you (the quad is culled).
  - Minus the halo work removed from the sky and cloud shaders near the sun.
- **B:** an atomic add on about 100-200 disc fragments per eye. Negligible, under 0.01 ms.
- **C:** CPU only.
- **D:** one more fetch inside 8° (about 2% of the view), about 0.01-0.02 ms.
- **Total: about 0.1-0.2 ms with the sun in view.** Measure it with `bench.py --ab` (in-run), not cross-run. Cross-run drift is about 0.5 ms (memory `glare-in-the-eye`).

### 3.5 Stereo correctness and SpaceWarp
- Stereo: rules 1-5 in §2.4. Per-eye projection of a source at infinity is correct by construction, and B gives each eye its own visibility.
- **AppSW motion vectors (a trade-off to decide):**
  - The veil is screened over geometry and writes no MVs, so in extrapolated frames it moves with the geometry beneath it, not with the sky. For a smooth, low-frequency veil the error is small.
  - Writing sky-motion MVs under the veil would wrongly move the trunk it covers.
  - The lamp glare already behaves this way (no motion code in `glare.rs`).
  - The `effects-must-write-motion-vectors` rule is about effects that own their pixels. Flag this one to the user rather than decide it silently.

### 3.6 Matching Soulframe beyond the glare (for the sky plan)
1. **Day wheel:** Soulframe's day is about 2:1 against its night (32/16 min) on a server clock with named moments. Our astronomy is exact. Add a game-time warp so the night (and possibly midday) runs faster than dawn and dusk. That is where the colour is, and the user already likes our sunsets. The server owns the clock ([[server-may-carry-headset-work]]).
2. **Weather as a few combinable states:** rain, wind, fog and lightning storm, crossed with time of day ("foggy morning", "rainy night"), plus **scripted overrides**: a story storm and an item that clears it.
   - This fits plan step 7 (server weather map and lightning).
   - The cloud layers' per-kind coverage is the driver: stratocumulus for overcast, cumulus to cumulonimbus for storms.
3. **One medium for fog and sky haze:** Soulframe's sky changes when volumetric fog is off. Ghost of Tsushima uses one physical model for sky, clouds, haze and fog particles.
   - Make the planned GPU aerial perspective LUT (step 4) the same medium as any ground and valley fog.
   - Then get **shafts** the Yōtei way, from the GPU cloud shadow map we already draw (64² over 1 km), but only inside a fog evaluation we already pay for, never as a screen-space pass.
4. **Glare as an artist lever:** DE patched out "excessive glare" for one cutscene. Expose `sun_glare_strength` and `squint_stops` levers next to `glare` and `glare_strength`.

---

## 4. Not confirmed (looked for, not found)
- Any DE talk, GDC, SIGGRAPH or Devstream segment on Soulframe's sky, clouds, atmosphere, sun or glare. No Digital Foundry Soulframe analysis.
- Whether Soulframe's clouds are volumetric.
- How its two suns are drawn.
- Whether it has god rays, lens flares or a squint effect, and what its graphics menu offers for them.
- The "Preludes 8 sky tinting" and "Preludes 1.3 dynamic sky" entries a search summary attributed to the patch-notes mirror. I could not find them in the page.
- RDR2, Cyberpunk and Horizon sun-glare implementations.
- Ritschel 2009's exact aperture components and FFT details (PDF 403). I cite the abstract only.
- On our side: that `SkyIrradiance::radiance` contains no sun lobe that the meter could see. This was read from `exposure.rs`, not run.
- On our side: the fragment-storage and atomics support on the Quest adapter, which item B needs.

---

## 5. Reference frames from the user (2026-10-20)
The user supplied nine Soulframe screenshots: a misty pond at low sun, a high sun with shafts, the sun behind thin cloud, a sunset, a rainy night, dense fog, a pink sunset, and two night skies. They settle some questions from section 4 and the user's decisions settle others.

**User decisions:** slightly exaggerated sun and camera glare are wanted; there is no squint (drop `squint_stops`) and there are no lashes or lid shading. Every draw writes SpaceWarp motion vectors.

**What the frames show:**
1. **Haze makes every shot.** Each daytime frame is depth layered: tree planes and towers fade into a sun-coloured haze. Near the sun the haze is a bright forward-scattering glow that fills a large part of the sky and surrounds silhouettes. Much of what reads as "glare" is in-scattering in the medium, not a lens effect.
2. **Shafts** appear wherever the sun is behind trees or cloud edges (high sun, misty pond). They are volumetric shadowing in that haze.
3. **Camera glare:** a horizontal streak through the sun, a soft wide bloom, and one ghost on the line from the sun through the screen centre (sun behind thin cloud). There is no starburst.
4. **Clouds:** cumulus with bright sunlit or backlit rims and darker bases. At night the clouds are moonlit blue and clearly volumetric against the stars.
5. **Night:** a stylised Milky Way, much brighter and more saturated than the real one (teal dust, orange core), with bloomed bright stars. The sky is brighter than the ground.
6. **Weather:** rain at night has a strong teal grade, long rain streaks at several depths and warm torches against it. Fog is a blue-grey medium where the sky and the fog are the same colour and the trees at middle distance are half gone.
7. **Grade by time and weather:** purple dawn, gold sunset, teal rain night, grey-blue fog.

**Our gaps, read from source on 2026-10-20:**
- Geometry has no aerial perspective or distance fog at all. `fog`, `mist` and `haze` occur only in the cloud map (`HAZE_PER_M`, applied to clouds), the effect slabs and the lamp glare veil. The terrain and brush pipelines have none. This is the largest single difference from these frames, and it moves step 4 (one medium for fog and sky haze) ahead of the cloud refinements.
- There are no shafts and no colour grade. The cloud phase is `phase` in `cloud_sky.rs` and the effects use `henyey_greenstein`.

**Stereo rule for camera glare:** an effect is placed in a direction, not at a screen position. A ghost "through the screen centre" sits at a different place in each eye and reads as an object at a wrong depth. Defined as a world direction (the sun direction reflected about the head's forward axis), it is the same in both eyes and sits at infinity. The streak is defined in head space, horizontal to the head. Both then move with the sky in the motion pass.

**Motion vectors:** haze and shafts are part of the medium in front of a surface, so they take that pixel's surface or sky motion. The glare overlay chooses per pixel which motion wins, as reflections already do.
