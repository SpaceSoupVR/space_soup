# Quest 3 rules

Everything in this renderer ships to a stock Quest 3 from the game's own install. A
desktop-only path is temporary, never a design. When something is too slow on the
headset, the answer is to make it lean there, not to leave it out.

## The frame

- **Budget.** With Application SpaceWarp on, the app renders at half rate. The target
  is 36 fps rendered, 72 shown; the frame budget is 13.9 ms of GPU work per rendered
  frame and eye pair, measured with margin.
- **Fill-bound, not draw-bound.** On our traces the shaders are about 82% busy and
  stalled on memory under 1%. Cost scales with pixels times instructions per pixel.
  - Cutting draw calls bought nothing: the ground went from 6,997 draws to 806 with no
    frame-time change.
  - Bandwidth is not the first lever either.
- **Quality first.** A change may not make the picture less realistic to save time. Win
  the time back elsewhere: a twin, a precomputed map, shared code.

## The instruction-cache cliff

- **The cliff.** A fragment shader past roughly 3,400 PIPESTATS instructions runs about
  35% slower on the Quest 3's GPU, because it no longer fits the instruction cache.
- **The proxy.** `shader_inlining::inlined_sizes(src, "fs_main")` counts each function's
  own size times the copies inlined into the entry point. Multiply by about 1.87 for
  scene readers, or about 2.4 for the reflection pass, to approximate PIPESTATS.
- **Caps are tests.** For example `the_rings_add_less_to_the_water_than_they_did`
  (ringed water at most 2,424) and `the_torch_puddle_reader_walks_its_sky` (at most
  2,600). Do not raise a cap to make a change fit. Make the change fit.
- **Presence cost.** Code compiled into a shader costs even when a uniform branch skips
  it: about 1 ms per eye was measured for skipped code.
  - So a feature many pixels never use goes in a twin, a second pipeline built from the
    same source with that code removed by a checked text edit.
  - Twins are picked on the CPU: per frame, per body, per triangle class, or per level.
  - Examples: `lights::without_spot_shadows`, `without_pool_maps`,
    `terrain_pipeline::deformless`, `water_pipeline::water_for_mud`,
    `ground_shape::unshaped`, the weather kinds, the nowhere/outdoor/open/puddle probe
    passes.
- **Shared code once.** When a function is inlined two or three times with different
  arguments, walk the arguments through one copy in a small loop, with the same calls in
  the same order so the pixels do not change.
  - Examples: `wx_ripples`, the noise octaves in `wx_surface`, `character_card_look`,
    the water's single `tonemap`.
  - Loop-indexed local arrays spill on Adreno, so keep each result in its own variable
    with `select`.
- **Registers.** PIPESTATS also reports registers. Occupancy drops at 24+ registers, and
  values held across a long stretch of shader raise them. Check both.

## Banned or required

- **No centroid interpolation.** It cost 6–9 ms a frame on the ground and stays banned.
  Two guard tests enforce this:
  `every_ground_readers_varyings_fit_the_quest_3s_inter_stage_limit` asserts no ground
  reader asks for it, and `the_brush_does_not_pay_for_centroid_interpolation` covers
  brushes. Removing it from the ground took `cave_back_out` from 37.9 to 29.1 ms.
- **SpaceWarp motion vectors from every moving draw.** Particles, effects, water, bodies
  and anything displaced over time write motion vectors through their motion twins
  (`space_warp`). A draw without them smears in synthesised frames.
- **Inter-stage limit.** At most 15 varyings between vertex and fragment stages on the
  Quest 3. The guard tests are `every_variants_varyings_fit_the_quest_3s_inter_stage_limit`
  and `every_ground_readers_varyings_fit_the_quest_3s_inter_stage_limit`.
- **SpaceWarp slots.** `space_warp` holds 2,048 slots, not 256; levels have more moving
  draws than 256.
- **Every view.** Bench outdoor and moving views too: an indoor-only bench once hid a
  10–13 ms terrain cost.

## Measuring

1. Size by the proxy: `shader_inlining::report`.
2. Registers and instructions by PIPESTATS: launch property `debug.spacesoup.pipestats`.
3. Time by `bench.py` on the headset, A/B against a lever in the same session.
4. Lossless claims by byte-identical renders. For example,
   `the_mud_twins_draw_the_waters_own_pixels` compares a twin against the full shader
   pixel for pixel.
