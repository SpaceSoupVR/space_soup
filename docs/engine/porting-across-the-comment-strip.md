# Porting across the comment strip

`main` had its comments stripped in commit 1232296 while `engine-foundations` kept adding
commented work (10-08 to 10-15). A plain merge conflicts on nearly every line those commits
touched, because every comment line is a difference. This page is how the 29 commits were
brought onto `main` (merge 8657411), and how to do it again.

## The strip rule

`tools/comment_strip/strip.py` reproduces commit 1232296 exactly on 93 of the 98 files it
changed. The other 5 had hand edits as well. The rule:

1. Any line whose first non-space text is `//` becomes blank, **except** lines starting
   with `// <fx:` or `// </fx:`. Those are effect markers that the code searches for (see
   [README.md](README.md)).
2. A trailing ` // ...` after code is removed, unless the line has an odd number of `"`
   (the `//` is inside a string, for example a URL or WGSL in a Rust string).
3. Runs of blank lines collapse to one. There is no blank line straight after a line ending
   `r#"`, and none at the end of the file.

Only `.rs` and `.wgsl` files are stripped.

**The first strip lost the `<fx:>` markers**, and the effect variants panicked with
"no block fog". Before stripping again, list every string literal that contains `//` and
check whether code matches it:

```bash
grep -rn '"[^"]*//[^"]*"' src --include=*.rs
```

## The merge (`tools/comment_strip/strip_merge.py`)

```bash
python3 -I tools/comment_strip/strip_merge.py <repo> <merge-base> <ours> <theirs>
```

- `<merge-base>` and `<ours>` are pre-strip (commented). `<theirs>` is the stripped `main`.
- For every path, it strips the base and ours, then does a three-way `git merge-file`
  against theirs. Paths only one side changed take that side.
- It writes the result into the work tree and prints the paths that conflicted.
- The plain strip produced 22 conflicted files; matching the colleague's exact rule brought
  that to 4. The 4 were resolved by hand.
- After it: build, run the tests (`cargo test --release`), check Android
  (`cargo check --release --target aarch64-linux-android`), and commit as a merge with
  `main` as the second parent so history shows both sides.

## What the 4 conflicts were, and what was decided

- **Binding numbers.** Both sides had taken binding 9: main's ground shape texture and our
  deform map. Deform moved to 10. Water mud moved to 7, because 6 is the wake.
- **`LIP_BURIED`.** Both sides defined it in the water. Ours was removed.
- **The weather lift** anchored on `v.pos` in ours and on `pos` in main. It was retargeted
  to `pos`.
- **Deform on the desktop only.** It was briefly split so the headset built without it.
  That was reverted: mud and deform are meant for the Quest 3 too, made lean instead (see
  [mud-and-deform.md](mud-and-deform.md)).

## The same for space_soup_protocol

`space_soup_protocol` took Checkpoint 55 (water entries on the wire) the same way: zero
conflicts.
