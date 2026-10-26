# Lighting and reflections

The parts of the lighting that came with the engine-foundations port. Older lighting (probes,
the sun mask, lamps and glare) is described by its own tests in `lights.rs` and `glare.rs`.

## Torch pools in reflections (lever `pool_bounds`, on)

- A torch pool seen in a reflection is bounded to the stretch the torch actually lit and to
  its front face. Before this, a lit pillar side claimed the far wall its plane crossed.
- The torch's glass shows in reflections only into its beam
  (`a_torchs_glass_glows_in_front_of_it_and_its_body_hides_it_from_behind`).
- The eye meters the torch's pool (lever `pool_metering`).
- The lamp highlight is capped at what physics allows.
- The torch-lit rain reader walks its sky once, which keeps it under the instruction-cache
  cliff (`the_torch_puddle_reader_walks_its_sky`, cap 2,600). `debug.spacesoup.torchskywalk`
  is its diagnostic.
- `spot_caster_drop` (bits 32, 64, 128, 256) is a diagnostic lever that drops caster classes
  out of the torch's shadow tiles, for live tests. It does not ship set.
- `a_torchs_pillar_shadow_on_the_far_ceiling_is_where_the_rays_put_it` checks the torch's
  shadow against ray-traced positions.

## Opaque twins (lever `opaque_models`, on)

- The scene pass's opaque models are drawn by twins that write alpha 1 without blending
  (`the_scene_passs_opaque_twins_build_beside_the_blended_pipelines`,
  `opaque_scene_shaders_write_alpha_one`).
- Why: blending defeats the Adreno's LRZ (low-resolution Z) early rejection, so opaque draws
  through a blended pipeline shaded pixels that were later covered.
- Diagnostic: `debug.spacesoup.opaque`.

## Body depth prepass (lever `body_prepass`, on)

- Bodies are drawn depth-only first, so the expensive body shading runs once per pixel.
- The body has dry and wet twins (`the_bodys_dry_and_wet_twins_validate_and_are_lean`).

## Character cards (lever `character_cards`, on)

- The player's reflection comes from six per-frame cards around the body (`CharacterCards`,
  `card_box`, `CARD_BOX_MARGIN`), not from tracing the skinned mesh.
- `a_reflection_meeting_the_player_shows_the_side_it_meets`.
- `character_card_look` is walked once in a loop (see [quest3-rules.md](quest3-rules.md),
  shared code once).

## Grab glow

- A model the grip will take glows. It uses `ModelUniform` highlight in the same draw, with
  no extra pass.
- It sits behind a uniform branch and is small (`the_grab_glow_is_behind_a_uniform_branch_and_small`).
- It reads in the dark and in daylight (`the_grab_glow_reads_in_the_dark_and_in_daylight`).

## Daylight that holds

Under open daylight sky, turning no longer re-exposes the eye. Door frames read the room
light on the eye's side.

## Scene cull (lever `scene_cull`, on)

- Models and caves out of both eyes' views are not drawn (`scene_cull::sees`, `sees_box`,
  `MARGIN`). One eye seeing a thing is enough.
- A cave is drawn from inside it and when facing it, not with your back to it.

## Frame-rate readout (`fps_hud`, lever `fps_hud`, off)

- A compositor quad layer in head space, about 12° left of and below the middle of the view
  (`POSITION`). It shows fps against the target and the runtime's app GPU and CPU ms, and
  updates every `UPDATE_EVERY` (500 ms).
- It turns amber, then red, below the target.
- The lever shows it without a hand on the controller. The app also toggles it.
- Known issue, not fixed yet: `XrRenderer::fps_hud_frame` halves the runtime's predicted
  display period under SpaceWarp, but that period is already 1/36 s. So the target reads
  /18 and the colour stays green at 22 fps. The fix is the rate 1e9 / period, with no
  halving.

## Sky before water (lever `sky_before_water`, on)

The sky is drawn before the water, so water pixels are not shaded and then covered by sky.
