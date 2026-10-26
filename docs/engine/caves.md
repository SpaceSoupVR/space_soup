# Caves

Two kinds of cave geometry: layered cave meshes, shaded by `cave_twins`, and baked caves
from the cave editor, shaded by `baked_cave`.

## Cave twins (`cave_twins`, levers `cave_twins`, `cave_occluder`, `cave_first`, on)

- Every cave triangle is classed as rock (a wall), ground (a floor) or mixed (a bend)
  (`CaveClass`, `classify`, `a_wall_is_rock_a_floor_is_ground_and_a_bend_is_mixed`).
- Each class is drawn by a twin of the cave shader with the other path removed (`CavePath`,
  `twin_source`). A wall pixel no longer pays for the ground code, and a floor pixel no
  longer pays for the rock code.
- Guards:
  - `every_pixel_of_a_class_takes_its_twins_path`: the twin draws the same pixels as the full
    shader for that class;
  - `the_twins_are_the_cave_shader_less_one_path`;
  - `the_partition_keeps_every_triangle_and_its_winding`.
- Only a turn about the vertical keeps the classes (`keeps_vertical`). A tilted placement
  falls back to the full shader.
- `occluder_source` (lever `cave_occluder`): when the eye is inside a cave's box, the cave is
  drawn depth-only in the depth prepass, so everything outside it (the ground above, the
  level beyond) is rejected by depth before shading. Baked caves do the same
  (`draw_occluders`, `occluder_ranges`).
- Lever `cave_first`: when the eye is inside a baked cave, the cave is drawn before the rest
  of the scene for the same reason.
- Cave rock comes in three twins.

## Baked caves (`baked_cave`)

- The cave editor bakes a cave to a file (`BakedCaveMeta`, `BakedCavePrimitive`,
  `BakedCaveVertex`) with chunks (`CaveChunk`) and a lightmap.
  `the_file_says_what_it_is_and_names_its_chunks`.
- The shader (`baked_cave_shader(CaveShader)`) reads tiled layers at the world's coordinate
  on each plane (`CavePlanes`):
  - it reads each layer once a plane, and never a plane with no share
    (`two_planes_blend_by_their_shares_and_a_plane_with_no_share_is_not_read`);
  - it takes no shadow tap of the level's sun (`the_shader_reads_each_layer_once_a_plane_and_takes_no_shadow_tap_of_the_levels_sun`):
    the lightmap holds that light.
- Triangles are reordered by twin (`partition`), so each twin is one range
  (`a_triangle_goes_to_the_twin_that_reads_what_it_reads`).
- At the join with the ground, the rock wears the ground's paint and sky, a band in
  (`at_the_join_the_rock_wears_the_grounds_paint`,
  `at_the_join_the_rock_takes_the_grounds_sky_and_its_own_a_band_in`), so there is no seam.
  `the_cliffs_reading_is_the_terrains_text`: the cliff reading is the terrain shader's own
  text. Change both together.
- Glow paint rides in two spare vertex bytes (`glow_hue_saturation`, `EMISSIVE_GAIN`).
- Soot, the tide band and glow are only in their own twin.
- `CAVE_DEBUG_VIEWS` / `cave_debug_shader`: debug pictures, each validated on every twin.

## Cave chunk cull (lever `cave_chunk_cull`, off)

- A long cave is drawn by the chunks in view, and none in view is lost
  (`a_long_caves_depth_is_drawn_by_the_chunks_in_view_and_none_in_view_is_lost`,
  `visible_ranges`, `covering_chunks`, `occluder_ranges`).
- It is off until it has been measured on the headset in the motion and occluder passes.

## Known issue

The editor's cave tool (tested 2026-10-07): starting from a solid box, the first cuts land in
mid-air and the bake is a floating tube. Not fixed yet.
