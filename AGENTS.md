# space_soup

The renderer and engine library behind ssstudio (desktop) and the Quest 3 app
(`space_soup_app`). **The Quest 3 is the target for everything.** A desktop-only feature is
never the design: a feature too slow for the headset is made lean there, not left out.

The source carries no comments. The reasons behind the code, the measurements and the rules
are in [docs/engine/](docs/engine/README.md):

- **Before changing any shader**, read [docs/engine/quest3-rules.md](docs/engine/quest3-rules.md).
  It covers the instruction-cache cliff, presence cost, twins, the caps that are tests, and
  what is banned.
- **Before stripping comments**, read the marker table in
  [docs/engine/README.md](docs/engine/README.md). Lines starting `// <fx:` and `// </fx:`
  are code.
- **Before merging a commented branch onto the stripped tree**, read
  [docs/engine/porting-across-the-comment-strip.md](docs/engine/porting-across-the-comment-strip.md)
  and use `tools/comment_strip/`.

Checks before a change counts:

```bash
cargo test --release
```

```bash
cargo check --release --target aarch64-linux-android
```

The headset renderer (`renderer::xr_renderer`) compiles only for Android, so the host tests
do not build it. The Android check is the only thing that does.
