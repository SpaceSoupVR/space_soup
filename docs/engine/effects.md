# Effects

Particles, fire, smoke, mist, fog, dust and splashes. The goal is that every effect can be
made in the editor, runs optimised on the Quest 3, and loses no realism.

## Effects are data (`effect_asset`, `effects::stack`, `effects::templates`)

- An effect asset is a set of emitters, each made of modules in five stages. It is checked
  against budgets when loaded:
  - a budget overrun is a refusal unless overridden with a reason
    (`a_budget_is_a_refusal_unless_overridden_with_a_reason`);
  - a typo is a load error, not a default (`a_typo_is_a_load_error_not_a_default`);
  - a lasting emitter is costed at its cap.
- `effect_check` (`src/bin/effect_check.rs`) validates assets from the command line.
- Today's fire, coals, smoke, embers, dust and splash are stacks that produce the old
  particles bit for bit (`the_stacks_make_todays_particles_bit_for_bit`). If you change a
  module, that test tells you whether you changed an existing effect.
- `EffectLibrary`, `CompiledEffect`, `Program`: the compiled form.

## Bursts and the governor

- Bursts (a shot's flash, its smoke and port smoke) are spawned by events (`EffectBurst`,
  `BurstSet`, `BURSTS_KEPT` = 1,024).
  - A burst out of view or range is not simulated
    (`a_burst_out_of_view_or_range_is_not_simulated`). Range is how far its particles can
    travel, so nothing visible is ever culled
    (`every_particle_stays_within_its_bursts_reach`).
- **The governor** holds a frame to `FRAME_FILL_BUDGET` (6.0 screens of fill) and
  `FRAME_INSTANCE_BUDGET` (1,500) across bursts, placed effects and splashes (`FrameBudget`).
  - It sheds far-and-small first and counts what it shed
    (`the_governor_thins_the_far_and_small_first_and_counts_it`,
    `the_governor_thins_far_placed_emitters_and_splashes_first`).
  - Automatic fire is held to its cap and thickens instead of multiplying
    (`automatic_fire_is_held_to_its_cap_and_thickens_instead`).
- Emitters behind a shut door or out of view are not simulated; doorway-clipped emitters
  work out only what the doorway shows.

## Variants (`effects::variants`, lever `effect_variants`, on)

- One shader variant per shading kind (`FxVariant`), each with its SpaceWarp motion twin.
  The union shader holds every kind. Each variant is cut from it with the `// <fx:NAME>` /
  `// </fx:NAME>` markers (see [README.md](README.md): **these comment lines are code**).
- `EFFECT_MODULE_CUTS` / `module_cut` cut unused modules the same way.
- `runs` groups the frame's particles into draws within `MAX_OVER_RUNS`.
- Guards:
  - `every_variant_and_its_motion_twin_validates_and_is_cut_from_the_union`;
  - `the_effects_motion_twin_validates_and_is_the_effects_shader`;
  - `every_variants_varyings_fit_the_quest_3s_inter_stage_limit` (15 locations; the motion
    varyings once overflowed it).
- `overdraw_shader` (lever `effects_overdraw`) shows fill.

## Motion vectors

Every particle carries its velocity (`every_particle_carries_its_velocity`,
`every_burst_particle_carries_its_velocity`). Effects write SpaceWarp motion vectors from
their motion twins. A draw without them smears in synthesised frames.

## Mist and fog slabs

- **Pond fog slab.** One instance spans its object and is weighed as one layer
  (`a_fog_slab_is_one_instance_spanning_its_object_weighed_as_one_layer`). An even low layer
  with fine wind streaks; mist follows the hour, thick at dawn and gone by midday
  (`mist_conditions`, `mist_follows_the_hour_thick_at_dawn_gone_by_midday`).
- **Pond mist** hovers over water in soft tufts, a thin layer about a metre deep: three
  gyroid layers, band 0.05–0.85 m. It is not a hugging veil or a smoke plume (user
  direction).
- **Integrated from each eye** (`fx_eye`): a slab integrates from each eye, not their
  midpoint, so the two eyes do not disagree
  (`a_fog_slab_integrates_from_each_eye_not_their_middle`). Its motion vectors come from
  its own density-weighted point.

## Splashes

- Splashes come from impact speed, size and mass. A pebble, a crate, a bullet and a foot
  each splash as what they are.
- The cavity grows as its Froude number; a bigger splash is slower
  (`a_cavity_grows_as_its_froude_number_and_a_bigger_splash_is_slower`).
- Crown, spray and foam; a strike into disturbed water raises no second crown.
- Rings go to the water (`the_rings_that_matter_most_go_to_the_water_and_none_pops`): see
  [water.md](water.md).

## Fire

- Sheets on the bed play a rising-gas flipbook over coals.
- The fire lights the room with a flicker that pulses at the rate its flames puff.
- Its light follows how much flame is in view, and it is metered from its own room, not
  through a wall.
- The tests in `effects::tests` whose names start `a_fire` or `a_flame` are the spec.

## Outdoor light

Effects past every room box take the outdoor light
(`out_in_the_open_past_every_box_the_light_is_the_outdoors`). Only a lamp in the emitter's
room lights it.
