# Quest 3 frame-time baseline (C1)

Committed on-device baseline for the renderer. C2/C3/C4 are measured against this — a
performance change without a committed baseline produces claims, not results.

## Instrumentation

`XrRenderer::render_frame_with_meshes` (`src/renderer/xr_renderer.rs`) times each frame via
`FrameStats` and logs a rolling average + max every **120 frames**:

```
PERF: cpu_avg=?.??ms cpu_max=?.??ms | gpu_avg=?.??ms gpu_max=?.??ms | frame=??.??ms (~??.?fps) over 120 frames
```

- **cpu** — CPU time encoding the frame (after the compositor `acquire/wait_image` blocks,
  before `device.poll(Wait)`), i.e. our own draw-call submission cost.
- **gpu** — how long the render thread blocks in `device.poll(Wait)` for the frame's GPU
  work to finish. A proxy for GPU-busy time; no `TIMESTAMP_QUERY` feature required. Moves
  with real GPU load, which is what makes it useful for the C2 MSAA A/B.
- **frame** — wall-clock time between successive frames. ≈ display refresh when hitting
  target framerate; longer when dropping frames.

## Repeatable measurement procedure

1. Build + install: `cd quest_app && ./run.sh` (or the manual `cargo build --target
   aarch64-linux-android --release` → gradle `assembleDebug` → `adb install -r`).
2. **Fixed scene:** the checked-in lobby scene, loaded from the local server (see
   `quest_deploy` notes) so the object set is identical run to run.
3. **Fixed head path:** stand at the play-space origin, look level at the far wall, hold
   still. (Movement changes fill and draw counts, so a still baseline is the comparable one.)
4. Capture ~20 s of logs:
   ```
   adb logcat -c && adb logcat -d -s quest_app | grep PERF
   ```
5. Record the median `PERF` line below.

## Baseline numbers

Device: Quest 3 (`2G0YC5ZG7M02PK`) · MSAA: 1x (none) · still head at origin · 2026-07-31

| metric | value |
|---|---|
| cpu_avg | **1.79 ms** (max ~2.4–2.9 ms) |
| gpu_avg | **9.28 ms** (max ~9.5–9.7 ms) |
| frame   | **13.89 ms (72.0 fps)** — locked to refresh |

Read: the app is **GPU-bound** — 9.3 ms GPU vs 1.8 ms CPU — but comfortably inside the
13.9 ms budget, so there is **~4.6 ms of GPU headroom** at 72 Hz. Captured over five stable
120-frame windows; the numbers barely moved (gpu_avg 9.27–9.31), so this is a solid baseline.

> Reproduce with the procedure above. Numbers are for the current default view (avatar +
> environment, no server scene loaded); re-capture with the lobby scene for a content baseline.

## What this gates

- **C2 (MSAA 4x):** expect `gpu_avg` to rise from 9.3 ms (more fragment work); with ~4.6 ms
  of headroom, `frame` should stay at 13.9 ms / 72 fps. Record the delta against the row above.
- **C3 (multiview):** expect `cpu_avg` to drop from 1.8 ms (draw calls ~halve).
- **C4 (foveation):** expect `gpu_avg` to drop (less peripheral fill).

## C2 (MSAA 4x) — scoping for the Track C owner

The C1 harness above is ready to measure it, but MSAA is **not** a one-line `sample_count: 4`
in this renderer, because of the SSR pipeline's structure:

- Opaque scene geometry is drawn into `scene_targets[eye]` (color **and depth**) at 1× in the
  scene pass, then **sampled as textures** by SSR and by the composite `eye_pass` blit
  (`blit_pipeline` reads scene color; reflective solids + the blit read scene depth).
- A multisampled texture cannot be sampled in a normal shader — it must be resolved first —
  and a **depth** target cannot be given a `resolve_target` at all. So making `scene_targets`
  4× breaks every downstream SSR/blit read.

**Recommended approach (the real work):**
1. Make `scene_targets` color + depth 4× MSAA; add a resolved 1× color texture and resolve
   into it at end of the scene pass (`resolve_target`).
2. For depth, either keep a separate 1× depth pre-pass for SSR/blit sampling, or switch SSR
   to reconstruct depth from the resolved buffer — depth MSAA resolve is the crux.
3. Bump `MultisampleState` to `count: 4` on exactly the pipelines that draw into
   `scene_targets` (solid/wire/mesh/skinned in `pipeline.rs` + `mesh_pipeline.rs`), leaving the
   composite and mirror pipelines at 1× unless they too move to an MSAA target.
4. Thread a `sample_count` param through those pipeline constructors rather than hardcoding.

This is deliberately left unlanded rather than shipped as a partial MSAA on only the composite
pass, which would raise `gpu_avg` without de-aliasing the scene geometry the scope magnifies —
i.e. it would look done without meeting C2's "scope visibly de-aliased" bar.
