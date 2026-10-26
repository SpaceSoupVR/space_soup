# Water

`water_pipeline` draws every water body: wave fields (`water_waves`), refraction of the bed,
reflections, foam, rings and the shore. Water seen from above while wading was the worst
view measured in play (about 21 fps on 2026-10-15), so its size matters most.

## Pipelines and twins

| Twin | Chosen when | Why |
|---|---|---|
| `ringless`, `pipeline` (ringed), `raining` | `for_rings`: no live ring, rings, rain drops | The ring and drop loops cost by presence |
| `waterline` (`water_line_wgsl`) | The eye is in this body | Discards the part above the eye's water line; `the_waterline_twin_is_the_water_plus_its_test` |
| `mirrored[..]` (`mirrored_wgsl`) | This body is the frame's planar mirror | Reads the planar target instead of tracing (see below) |
| `WaterMud::{None, NoPrints, Prints}` (`water_for_mud`, `ensure_mud`) | `WaterMud::of(uniform)`: `look.y == 0`, no print map, or both | Every mud term is multiplied by the mud and every print term by the map; a body without them draws the same pixels without that code |

- **Fallback.** The full set (`Prints`) is always built and right for any body. The lean
  sets are built at `set_water` for the kinds the level has.
- **Guards:**
  - `the_rings_add_less_to_the_water_than_they_did`: every ringed mud twin is at most
    2,424 by the proxy (clear 2,277, muddy 2,322, with prints 2,405).
  - `the_mud_twins_draw_the_waters_own_pixels`: the twins are byte-identical to the full
    shader in 16 views.

## Lossless size cuts (2026-10-19)

These are written once and used twice; pixels are unchanged against the merged shader in
32 views:

- `sky_n = sky_irradiance(n)` feeds both the body light and the foam light.
- `sky_irradiance_up()` (in the lights block) is the SH evaluated straight up, with its
  zero terms dropped. It replaced `sky_irradiance(vec3(0, 1, 0))`.
- `swash` and `swash_amp` are counted once, before the breaker branch, and used by both
  the breaker and the surf. Both sit under the same uniform `water.waves.y > 0.0`.
- One `tonemap` for both the surface colour and the bed seen beyond it (`mapped`, a
  two-pass loop).

## Mechanisms

- **Planar mirror** (`planar_water`, lever `planar_water`, on).
  - One calm body a frame is drawn as a true mirror from a mirrored camera, scissored to
    its screen rect.
  - Cost: about 1.0 ms at the pond and nothing elsewhere.
  - `the_mirror_holds_against_flicker_and_passes_to_a_much_larger_pond` keeps the choice
    from flickering between bodies.
- **Wake** (`water_wake`).
  - A 96×96 grid of 5 cm cells, centred on the player. It solves a dispersive heightfield
    with sources from hands, legs and floats; a still body makes nothing.
  - Its slope and curvature (Rgba16Float) feed the normal and caustics only.
  - It rests after 4 s idle and costs the same in any level size.
- **Rings and drops.**
  - Splash rings are a dispersive train leaving the cavity rim as the crown falls, sized
    by entry energy.
  - Drops landing back make their own rings: the `raining` twin.
  - `RINGS` is the per-body ring count.
- **Sea shelf and lip.**
  - Past the ground map's edge, under a sea, the floor is one shelf colour all round the
    rim (`ground_map::build`, `spacesoup_graphics::ground::{rim_weight, one_shelf}`), so
    the sea floor no longer streaks to the horizon.
  - The rim band scales with the map's size (`map_span`).
  - The water lip counts only where the line grazed the ground (`LIP_BURIED`,
    `LIP_RAMP`), so narrow cliff feet do not smear;
    `the_water_lip_counts_only_where_the_line_grazed_the_ground`.
- **Sea line.**
  - `XrRenderer::sea_line()` is the highest water body that reaches past the ground map.
    Ground below it is soaked by porosity (`TerrainMaterialUniform::sea_line`,
    `SEA_LINE_WGSL`), and the ground map does the same so far ground matches near ground.
- **Shore wet line.**
  - `shore_wet_line` follows the waves up and creeps back;
    `the_wet_line_follows_the_waves_up_and_creeps_back`.
- **Seen bodies and runs.**
  - A body out of sight costs no wave compute, draw or motion (`seen`).
  - `water_runs` draws only the triangles in view.
  - Lever `sky_before_water`: the sky is drawn before the water.
- **Gloss and mud look** (`WaterOptics::{gloss, mud}`, from `WaterDef::{gloss, mud}`).
  - Gloss dims the reflection; mud flattens the ripples into a grainy matte surface.
  - The sky light on the water's body follows the wave normal (`sky_n`).

## Cuts (diagnostics)

`WATER_CUTS` (`env_only`, `fresnel_only`, `surf`, `beyond`, ...) are text edits matched
against exact WGSL lines. If you edit a line a cut names, update the cut.
`every_water_cut_matches_and_validates` fails otherwise.
