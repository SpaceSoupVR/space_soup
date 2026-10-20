# Mud and deform

Mud prints, wet zones and puddles in the ground, and the same prints seen through thin
water. They are meant for the Quest 3 as well as ssstudio.

## Pieces

| Piece | Where | What it does |
|---|---|---|
| `spacesoup_graphics::mud::{MudMap, Walkers}` | spacesoup_graphics | The CPU map: stamping prints from strides, spatter, craters, fading, relaxation (`settle`) |
| `renderer::mud_map::MudBody` | space_soup | The map plus its GPU texture. `walk`, `blast`, `tick`, `flush`, `weight_at`. Shared by the desktop and the headset, so a print looks the same in both |
| `TerrainMaterialUniform::{mud_layer, deform_metres, deform_texel, wet_dark}` | terrain_pipeline | Which splat layer is mud (−1 means none), how deep a full print sinks, and the metres per map texel |
| `deform_tex` (terrain material binding 10) | terrain shader | Read once in `vs_main` (the vertex sinks or rises) and once in `fs_main` (sink, wet, tilt, pool) |
| `mud_tex` (water material binding 7) and `WaterUniform::prints` | water shader | The same map under a thin water film |

Binding numbers: 9 is the ground shape texture on the terrain, and 6 is the wake on the
water. That is why deform and mud sit at 10 and 7.

## The texel format (`terrain_pipeline::DEFORM_FORMAT`, `deform_rows`)

`Rgba16Float`, with values in 0..255 steps:

- R: height (128 is flat);
- G: wetness;
- B: the height's change across the texel in x, `h[x+1] - h[x-1]`;
- A: the same in z.

The neighbours wrap, as the Repeat sampler does.

- **Why the slope is stored.** Bilinear filtering is linear, so one filtered read of the
  stored difference equals the difference of the filtered heights one texel either side.
  The ground used to take 4 extra reads per pixel for that slope.
- **Accuracy.** `one_slope_read_is_the_four_height_reads` checks both against an exact
  CPU bilinear: the same worst error (0.91 of an 8-bit step, set by the GPU's filter
  weight precision).
- **Exactness.** Half floats hold every integer step and difference exactly
  (`the_deform_map_holds_every_step_and_slope_exactly`). Shaders decode with
  `* DEFORM_UNIT` (1/255).
- **Sending changes.** When rows change, `deform_rows_touched` also sends their
  neighbours, because the slopes beside a changed row read it.
- **Cost.** 8 bytes a texel, so 32 MB at the 2048×2048 maximum.

## The mudless build (`GROUND_MUDDY`, `deformless`)

A ground with no stamped map would still pay for the deform code in every pixel (see
[quest3-rules.md](quest3-rules.md), presence cost). So:

- `terrain_shader_built_mud(role, painted, muddy)` builds either the full ground or
  `deformless(src)`, which removes the deform vertex lines and the fragment block from
  `let mudw = mud_weight_f(w);` to `rough_map = mix(rough_map, 0.08, pool);`.
- With a flat map every deform term is zero, so the two draw the same pixels.
- `GROUND_MUDDY` is part of the level's ground build key (`ground_build_moves(painted,
  muddy)`). The headset sets it from `XrRenderer::ground_is_muddy()`, which is true
  while it holds a mud map.
- The desktop scene pipeline always builds muddy (`TerrainPipeline::new`).
- **Guards:**
  - `every_ground_source_builds_with_and_without_its_deform_map` checks 2 deform reads
    when muddy and 0 when not, and that every role validates.
  - `the_muddy_ground_stays_under_the_readers_caps` checks the muddy torch puddle reader
    is at most 2,600 and that the muddy nowhere twin drops the same lookup.

## Stamping

- **ssstudio:** `Renderer::walkers_on_mud(&[(id, feet)])` is fed the simulated walkers,
  plus the camera with `DESKTOP_MUD_CAMERA`. `blast_on_mud` makes craters.
- **Headset:** `XrRenderer` has the same `walkers_on_mud` and `blast_on_mud`.
  - It makes its `MudBody` from the height grid's footprint whenever the material names a
    mud layer (`ensure_mud_map`, run from `rebuild_terrain_material` and
    `set_terrain_heights`).
  - It binds the map into the terrain material (`from_layers_full`) and into every water
    body (`rebind_water_mud`).
  - The app calls `walkers_on_mud` once a frame. The walkers are every player's feet,
    taken under each head on that player's own floor (`avatar_render::build_bodies`),
    keyed by the player id. `walkers_on_mud` also ticks the map: it settles by wall time
    (capped at `MAX_SETTLE_STEP`) and sends what changed.
- **The map's footprint must equal the terrain uv's footprint** (the height grid's
  min/max). The shaders read the map at the ground's own uv, so anything else puts
  prints in the wrong place.
- `MudBody::weight_at` is the CPU twin of the shader's `mud_weight_v`: a print is scaled
  by how much of the ground under the foot is the mud layer.

## Open

- **The water print gradient step.** The water's gradient is taken at
  `prints.zw * MUD_PRINT_STEP`, where `prints.zw` is 1/extent, so the step is ±1 metre
  where ±1 texel was probably meant. If it should be one texel, the water can read the
  stored slope (`.ba`) and drop its 4 extra reads.
- **Not yet in sun shadows or the SpaceWarp motion pass.** Displaced ground is drawn
  undisplaced in both. Prints sink a few centimetres, so this is not visible today.
