# space_soup engine notes

This folder is the "why" for the renderer. The source has no comments: they are stripped
before merging. So the reasons, the measurements behind them, and the rules a change must
keep live here.

These pages name things the strip cannot remove: types, functions, constants, levers,
`debug.spacesoup.*` properties and test names. Search the code for those names.

**Read [quest3-rules.md](quest3-rules.md) before changing any shader.** Most of the odd
shapes in this renderer exist to satisfy those rules. Code that looks redundant (a twin,
a text edit of a shader, a loop that runs twice) is usually there for a measured reason.

## Comment lines that are code (do not strip these)

A few `//` lines inside WGSL are markers that Rust code searches for. Removing them breaks
the build at run time or in tests, not at compile time:

| Marker | Used by | Fails as |
|---|---|---|
| `// <fx:NAME>` and `// </fx:NAME>` in the effects shader (`effects.rs`) | `effects::variants::cut` and `only`, which cut one effect kind's code out of each variant | `effects variants: no block NAME` (panic at start-up and in tests) |

To strip comments safely, keep any line that starts with `// <fx:` or `// </fx:`.
[porting-across-the-comment-strip.md](porting-across-the-comment-strip.md) describes how to
check for others: list every string literal in the code that contains `//`.

## Pages

| Page | Covers |
|---|---|
| [quest3-rules.md](quest3-rules.md) | The frame budget, the instruction-cache cliff and its proxy, presence cost, twins, the centroid ban, SpaceWarp motion vectors, quality first |
| [ground.md](ground.md) | Chunks, LOD, ground shape, reader twins, the horizon, world-space upload, the ground map, wetness |
| [mud-and-deform.md](mud-and-deform.md) | The mud map, its texel format, the mudless build, desktop and headset stamping |
| [water.md](water.md) | Planar mirror, wake, rings and drops, the waterline twin, the mud twins, the sea shelf and lip |
| [effects.md](effects.md) | Effects as data, variants and their markers, motion twins, the governor, mist and fog slabs, splashes |
| [lighting-and-reflections.md](lighting-and-reflections.md) | Torch pools, opaque twins, the body prepass, character cards, grab glow, the FPS readout |
| [sky-and-clouds.md](sky-and-clouds.md) | The simulated sky, the cloud layer, its CPU twin and shader twin, the per-frame sun dimming |
| [sky-research-games-2026-10-20.md](sky-research-games-2026-10-20.md) | What shipped games do for skies and clouds (Nubis, Tsushima/Yōtei, Frostbite, UE5, RDR2, HDRP), gaps against ours, ranked recommendations with Quest 3 cost estimates |
| [soulframe-sky-and-sun-glare-2026-10-20.md](soulframe-sky-and-sun-glare-2026-10-20.md) | What is public about Soulframe's sky, day/night and weather; how games and research do sun glare; a Quest-sized sun veil, per-eye visibility and pupil squint for ours |
| [caves.md](caves.md) | Cave twins, the baked cave shader, cave chunk culling |
| [porting-across-the-comment-strip.md](porting-across-the-comment-strip.md) | How engine-foundations was merged onto main after the strip, and how to repeat it |

## Where the evidence is

- **Instruction size:** `shader_inlining` (tests only). `report(label, src, "fs_main", n)`
  ranks a shader's inlined functions by size times copies.
- **Measured caps** are test asserts: their names and messages say what each one guards.
  Grep `by the proxy` in the tests.
- **Headset numbers** come from the app's `bench.py` (unattended runs, per-draw traces,
  PIPESTATS register counts). Offline, a change counts only once it is measured on a
  Quest 3.
