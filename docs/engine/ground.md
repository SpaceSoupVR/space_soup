# Ground

The terrain is the largest share of most frames. Measured on the Quest 3, it once cost
10–13 ms in outdoor views that an indoor-only bench never showed. Everything here exists to
keep the ground's cost proportional to the pixels it covers, in any level size.

## Chunks and LOD (`ground_lod`, levers `ground_lod`, `ground_lod_strict`, both on)

- **Chunks.** The ground is split into fixed 32-cell chunks, whatever the level's size, kept
  in Morton order so a square of chunks stays together in memory. A quadtree cull picks the
  frame's chunks without walking the whole level. Every pass draws one selection of chunks.
- **LOD steps.** A chunk has 2048, 512, 128, 32 and 8 triangles at its five steps.
  - Borders between any two levels are stitched and watertight by test
    (`every_border_of_every_level_pair_and_side_is_watertight`).
  - Neighbours differ by at most one level.
  - The strict rule keeps every drawn node within a pixel of the full ground
    (`the_strict_rule_keeps_every_drawn_node_within_a_pixel`).
  - Turning or swaying the head changes no level (`turning_the_head_changes_no_level_and_no_stitch`,
    `a_swaying_head_does_not_flip_a_level`), so LOD never pops while you look around.
  - Chunks with a hole or a patch stay at step 1 within 60 m.
- **Why coarse triangles still shade like the full ground** (`ground_shape`): the ground is
  lit by vertex normals, so a coarser triangle would shade differently even inside one
  pixel. A coarse triangle reads the step-1 shape (height, normal, diagonal) from a texture
  (`GroundShape`, `DIAGONAL_BIT`) and shades as the full ground under it
  (`a_coarser_triangle_reads_the_step_1_ground_under_it`).
- **Reader twins** (`ground_shape::unshaped`, `SHAPED_EDITS`). Step-1 triangles keep their
  vertex normal and are drawn by a pipeline without the shape read. Only coarse-list runs
  take it. The read's presence cost was 2.7 ms at `cave_back_out`. The triangle's first
  (provoking) vertex marks which step it is
  (`a_triangles_first_vertex_tells_step_1_from_coarser`).
- **Layout.** Block-major layout cut the scene draws on `stress_2049` from 6,997 to 806. The
  GPU frame time did not change, because the frame is fill-bound.
- **Whole blocks.** A block chosen whole draws as one range in the motion pass (`Wholes::runs`).
- Diagnostics: `debug.spacesoup.groundlod`, `groundlodmrad`, `groundshape`, `groundblocks`,
  `chunks`.

## Slope classes and twins (`ground_twins`, lever `slope_twins`, on)

- Each chunk's triangles are split once, at load, into gentle, mixed and steep (`SlopeSplit`,
  `CellSlopes`, `GENTLE_MARGIN_DEG`). It used to be rebuilt while walking.
  `the_split_made_at_load_is_the_one_the_frames_find` guards that.
- Gentle triangles take `gentle_shader` (no side planes), steep ones `steep_shader` (no top
  plane). Mixed triangles take the full shader. A triangle is gentle only when every vertex
  is under the margin, and steep only when no blend of its normals is gentle.
- Wet and steep wet readers have their own twins (`wet_readers`, `steep_readers`, lever
  `wet_steep_twins`).
- `dedup_passes` / `dedup_pass_shader` (lever `pass_dedup`): the ground probe pass draws each
  triangle once, not once per reader.
- Lever `ground_classes`: classes are looked up through a grid.

## Which ground pass a triangle takes (`outdoor_pass`, levers `outdoor_pass`, `nowhere_pass`)

- `GROUND_FULL`, `GROUND_OUTDOORS`, `GROUND_NOWHERE` index the level's ground pipelines
  (`LevelGround`).
- Open ground (outside every room box but inside the outdoor volume) takes an outdoor twin
  without the indoor room and lamp code. Ground past every box takes the nowhere twin,
  which finds nothing to light it but the sky and sun. Ground in a doorway is never outdoors
  (`ground_in_a_doorway_is_not_outdoors`).
- Reflection twins for open, wet and puddle ground: levers `open_reflections`,
  `wet_reflections`, `puddle_reflections` (all off while being measured).

## The ground's horizon (`ground_horizon`, levers `ground_horizon`, `ground_horizon_caves`, on)

- `GroundOccluder` and `Horizon` (`BINS` = 180 directions): from the eyes, ridges hide what
  stands behind them, so ground and water behind a ridge is not drawn.
- Guarded by `nothing_called_hidden_can_be_seen` and the cave cases
  (`a_cave_through_the_ridge_shows_what_is_behind_it`,
  `from_inside_a_plugged_cave_only_its_mouths_show_the_ground`).
- One horizon stands for the eyes near it (`for_eyes`).

## Uploaded once, in world space (levers `ground_buffers`, `brush_buffers`, on)

- The ground and the brushes keep their buffers on the GPU and are posed in the vertex
  shader, instead of being rewritten when the rig moves (`StaticSolid`, `StaticBrushes`,
  `WORLD_BRUSHES_BY_DEFAULT`).
- `the_same_ground_for_the_same_rig_is_not_written_again`.
- Diagnostics: `debug.spacesoup.worldground`, `debug.spacesoup.worldbrushes`.
- Shaders work in the player frame. Baked, world-anchored data (maps, masks, probes) has to
  be turned into that frame where it is read, or it slides as the rig moves.

## Steep ground (2026-10-15)

Steep ground had streaks, chevrons, pale triangles and wet shine. The fixes, each by test
(`steep_shapes_frame`):

- planes are chosen from the triangle's own face where the smoothed normal disagrees
  (`a_steep_faced_triangle_is_never_gentle`);
- every map blends by the same share;
- mip gradients come from the world position (`the_frame_takes_no_derivative_of_a_chosen_coordinate`);
- side planes no longer lean the normal;
- the rock layer takes the material's roughness override.

On the steep views, ground pixels paying for the reflection trace went from 88% to 43%.

## Ground map (`ground_map`)

- A low-resolution picture of the whole level's ground used past the drawn ground, in
  reflections and in the ground trace. It is built on a worker thread.
- It carries each layer's texture at its own grain (`LayerDetail`), so the pond mirrors the
  rock slope rather than its mean colour.
- `the_transcribed_rules_are_the_terrain_shaders`: the map's colour rules are the terrain
  shader's. If you change the ground's colour rules, change both.
- Under a sea, the rim past the map's edge is one shelf colour (see [water.md](water.md)).

## Wetness

- Ground wetness is by material (porosity), plus a shore wash line that follows the waves
  (see [water.md](water.md)).
- The weather's rain wetting uses its own wet-only twins (`weather.rs`): the wet-only twin
  margin is 250 by the proxy.
- The mud map's wetness channel adds to it (see [mud-and-deform.md](mud-and-deform.md)).

## Cuts (diagnostics)

`ground_cuts` are text edits of the ground readers, matched against exact WGSL lines
(`every_ground_cut_matches_its_shaders`). `log_ground_cuts` sizes each. Lever
`terrain_reader` / `reader_edit` applies one at run time for headset A/B.
