//! ONE SHADER VARIANT A KIND OF SHADING, cut from the effects' shader.
//!
//! The effects' shader (`effects_shader`, the UNION) shades four kinds of
//! over-blended particle in one `fs_over` -- a bed of coals, a flame, six-way
//! smoke, a plain sprite -- and two kinds of screened light in one
//! `fs_screen`. On the headset a shader's SIZE costs every pixel it draws,
//! whichever branch the pixel takes (the instruction cache), and so do the
//! registers its largest branch holds (occupancy): a puff of smoke paid for
//! the flame's code. So each kind is drawn by a VARIANT: the union's own
//! text with the other kinds' blocks cut out by edits that are each checked
//! to have matched, as the motion twin is made (`motion_shader_of`, which
//! is then applied to each variant for its SpaceWarp twin). A variant draws
//! the same pixels as the union: the lines it keeps are the union's.
//!
//! DRAW ORDER. Screened particles need no order: one draw a variant. Those
//! blended over what is behind them must stay back to front ACROSS kinds, so
//! the sorted list is cut into runs of one kind ([`runs`]) and drawn run by
//! run, at most [`MAX_OVER_RUNS`] draws: where kinds interleave more finely
//! than that (flame sheets among their smoke), the stretches between the
//! largest runs are drawn by the union, which is always built. How much of
//! a frame fell back to it is counted ([`RunStats`]).

use super::{effects_shader, motion_shader_of, EffectFrame, EffectInstance, SMOKE_BACK};

/// A shader variant, and with it a pipeline: `as usize` indexes
/// `EffectsPipeline::variants` and the motion twins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FxVariant {
    /// The whole `fs_over`: any over-blended particle.
    Union = 0,
    /// One lit colour, alpha from a layer: a splash's crown, mist.
    Plain,
    /// Six-way smoke.
    Sixway,
    Flame,
    Coals,
    /// The whole `fs_screen`: any screened particle.
    ScreenUnion,
    /// Light of one colour: embers, dust, drops. `fs_screen` as it was
    /// before a flash existed.
    ScreenPlain,
    /// Light through the heat ramp: a muzzle flash.
    ScreenHeat,
}

/// The line of `fs_screen` that chooses a screened particle's light.
const SCREEN_LIGHT: &str = "    let light = select(in.colour.rgb, fx_flame(in.colour.rgb, t.r), in.layer.z > 3.5);\n";
/// The line every kind's soft fade into the walls is read by.
const DEPTH_FADE: &str = "    let fade = fx_depth_fade(in);\n";
const TAIL_FIRST: &str = "    let six = in.layer.z > 0.5;\n";
const TAIL_LAST: &str = "    return vec4<f32>(aces_fitted(lit) * alpha, alpha);\n";

fn swap(s: &mut String, from: &str, to: &str) {
    assert_eq!(s.matches(from).count(), 1, "effects variants: `{from}` moved in the effects' shader");
    *s = s.replace(from, to);
}

/// Remove a marked block (`// <fx:name>` to `// </fx:name>`) whole.
fn cut(s: &mut String, name: &str) {
    let (open, close) = (format!("    // <fx:{name}>\n"), format!("    // </fx:{name}>\n"));
    let from = s.find(&open).unwrap_or_else(|| panic!("effects variants: no block {name}"));
    let to = s.find(&close).unwrap_or_else(|| panic!("effects variants: block {name} is not closed")) + close.len();
    s.replace_range(from..to, "");
}

/// Keep a marked block's body and drop the `if` round it: the only kind.
fn only(s: &mut String, name: &str, opening: &str) {
    swap(s, &format!("    // <fx:{name}>\n{opening}"), "");
    swap(s, &format!("    }}\n    // </fx:{name}>\n"), "");
}

/// Remove `fs_over`'s tail, the lit sprites: everything after the blocks.
fn cut_tail(s: &mut String) {
    let from = s.find(TAIL_FIRST).expect("effects variants: the lit tail moved");
    let to = s.find(TAIL_LAST).expect("effects variants: the lit tail's end moved") + TAIL_LAST.len();
    s.replace_range(from..to, "");
}

/// The lit tail as one kind only: six-way, or plain (no back layers read).
fn tail(s: &mut String, six: bool) {
    swap(s, TAIL_FIRST, "");
    if six {
        swap(s, &format!("select(layer, layer + {SMOKE_BACK}, six)"), &format!("layer + {SMOKE_BACK}"));
        swap(s, &format!("select(next, next + {SMOKE_BACK}, six)"), &format!("next + {SMOKE_BACK}"));
        swap(s, "select(in.colour.rgb + in.light, in.colour.rgb * around + in.light * toward, six)", "in.colour.rgb * around + in.light * toward");
    } else {
        swap(s, &format!("    let back = select(layer, layer + {SMOKE_BACK}, six);\n"), "");
        swap(s, &format!("    let back_next = select(next, next + {SMOKE_BACK}, six);\n"), "");
        swap(
            s,
            "    let b = mix(textureSampleGrad(fx_atlas, fx_samp, in.uv, back, gx, gy), textureSampleGrad(fx_atlas, fx_samp, in.uv, back_next, gx, gy), in.layer.y);\n",
            "",
        );
        swap(s, "    let lp = max(in.light_q, vec3<f32>(0.0));\n    let ln = max(-in.light_q, vec3<f32>(0.0));\n", "");
        swap(s, "    let toward = dot(lp * lp, a.rgb) + dot(ln * ln, b.rgb);\n    let around = (a.r + a.g + a.b + b.r + b.g + b.b) / 6.0;\n", "");
        swap(s, "select(in.colour.rgb + in.light, in.colour.rgb * around + in.light * toward, six)", "in.colour.rgb + in.light");
    }
}

impl FxVariant {
    pub const ALL: [Self; 8] = [Self::Union, Self::Plain, Self::Sixway, Self::Flame, Self::Coals, Self::ScreenUnion, Self::ScreenPlain, Self::ScreenHeat];

    /// Its pipeline's label, which the headset's PIPESTATS are logged by.
    pub fn label(self) -> &'static str {
        match self {
            Self::Union => "effects_over",
            Self::Plain => "effects_over_plain",
            Self::Sixway => "effects_over_sixway",
            Self::Flame => "effects_over_flame",
            Self::Coals => "effects_over_coals",
            Self::ScreenUnion => "effects_screened_union",
            Self::ScreenPlain => "effects_screened",
            Self::ScreenHeat => "effects_screened_heat",
        }
    }

    pub fn screened(self) -> bool {
        matches!(self, Self::ScreenUnion | Self::ScreenPlain | Self::ScreenHeat)
    }

    /// Its fragment entry point.
    pub fn entry(self) -> &'static str {
        if self.screened() {
            "fs_screen"
        } else {
            "fs_over"
        }
    }

    /// THE VARIANT AN INSTANCE IS DRAWN BY, from how it is shaded
    /// (`EffectInstance::params.w`): the render key of the stack that made
    /// it.
    pub fn of(inst: &EffectInstance, screened: bool) -> Self {
        let shade = inst.params[3];
        match (screened, shade) {
            (true, s) if s > 3.5 => Self::ScreenHeat,
            (true, _) => Self::ScreenPlain,
            (false, s) if s > 2.5 => Self::Coals,
            (false, s) if s > 1.5 => Self::Flame,
            (false, s) if s > 0.5 => Self::Sixway,
            (false, _) => Self::Plain,
        }
    }

    /// Its shader: the union's text, less what it never runs.
    pub fn shader(self) -> String {
        let mut s = effects_shader();
        match self {
            Self::Union | Self::ScreenUnion => {}
            Self::Plain | Self::Sixway => {
                cut(&mut s, "coals");
                cut(&mut s, "flame");
                tail(&mut s, self == Self::Sixway);
            }
            Self::Flame => {
                cut(&mut s, "coals");
                only(&mut s, "flame", "    if (in.layer.z > 1.5) {\n");
                cut_tail(&mut s);
            }
            Self::Coals => {
                only(&mut s, "coals", "    if (in.layer.z > 2.5) {\n");
                cut(&mut s, "flame");
                cut_tail(&mut s);
            }
            Self::ScreenPlain => swap(&mut s, SCREEN_LIGHT, "    let light = in.colour.rgb;\n"),
            Self::ScreenHeat => swap(&mut s, SCREEN_LIGHT, "    let light = fx_flame(in.colour.rgb, t.r);\n"),
        }
        s
    }

    /// Its SpaceWarp motion twin, from its own text, and the twin's
    /// fragment entry point.
    pub fn motion_shader(self) -> (String, &'static str) {
        (motion_shader_of(self.shader()), if self.screened() { "fs_fx_motion_screened" } else { "fs_fx_motion_over" })
    }
}

/// MEASUREMENT: the union with one module cut at a time -- (name, fragment
/// entry point, what is cut). With `debug.spacesoup.pipestats` set each is
/// built once and logged as `effects_cut_<name>`: a module's instructions
/// and registers are the union's (`none`, `screen_none`) less its cut's.
/// Not the same picture; nothing draws with them.
pub const EFFECT_MODULE_CUTS: &[(&str, &str, &str)] = &[
    ("none", "fs_over", "nothing: the union"),
    ("coals", "fs_over", "sprite_bed: the bed of coals"),
    ("flame", "fs_over", "sprite_flame: the heat ramp, the flow-blended frames, the erosion"),
    ("sixway", "fs_over", "sprite_sixway: the back layers' two reads and the six-side weights (what is left is sprite_plain)"),
    ("soft_depth", "fs_over", "soft_depth: the probe pass's depth read and the fade"),
    ("screen_none", "fs_screen", "nothing: the screened union"),
    ("screen_heat", "fs_screen", "sprite_emissive's heat ramp"),
    ("screen_soft_depth", "fs_screen", "soft_depth, in the screened shader"),
];

/// The shader of one of [`EFFECT_MODULE_CUTS`].
pub fn module_cut(name: &str) -> Option<String> {
    let mut s = effects_shader();
    match name {
        "none" | "screen_none" => {}
        "coals" => cut(&mut s, "coals"),
        "flame" => cut(&mut s, "flame"),
        "sixway" => tail(&mut s, false),
        "soft_depth" => swap(&mut s, DEPTH_FADE, "    let fade = 1.0;\n"),
        "screen_heat" => swap(&mut s, SCREEN_LIGHT, "    let light = in.colour.rgb;\n"),
        "screen_soft_depth" => swap(&mut s, "    let strength = in.colour.a * t.a * fx_depth_fade(in);\n", "    let strength = in.colour.a * t.a;\n"),
        _ => return None,
    }
    Some(s)
}

/// DEBUG: the effects' shader with one more fragment stage, a flat eighth
/// of white for every pixel of every particle's cutout: drawn added, a
/// pixel's brightness is the layers it pays for (`Levers::effects_overdraw`).
pub fn overdraw_shader() -> String {
    format!(
        "{}\n@fragment fn fs_overdraw(in: VOut) -> @location(0) vec4<f32> {{\n    return vec4<f32>(vec3<f32>(0.125) * step(0.0, in.uv.x), 0.0);\n}}\n",
        effects_shader()
    )
}

/// One draw: a run of the frame's instances, all of one variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub variant: FxVariant,
    pub first: u32,
    pub count: u32,
}

/// What a frame's runs came to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunStats {
    pub draws: u32,
    /// Runs of one kind the over-blended list had before any were merged.
    pub kinds_runs: u32,
    /// Over-blended instances drawn by the union because kinds interleaved
    /// more finely than [`MAX_OVER_RUNS`] draws.
    pub union_instances: u32,
}

/// The most draws the over-blended particles take.
pub const MAX_OVER_RUNS: usize = 8;

/// THE FRAME'S DRAWS. Over: runs of one kind, back to front; where there
/// are more than [`MAX_OVER_RUNS`], the largest runs keep their variants
/// and each stretch between them is one draw of the union. Screened: one
/// draw a variant (`simulate_all` puts the heat-ramp ones last).
/// `use_variants` false: the union for everything (the A/B).
pub fn runs(frame: &EffectFrame, use_variants: bool) -> (Vec<Run>, RunStats) {
    let (over, screened) = (frame.over, frame.screened);
    let mut out: Vec<Run> = Vec::new();
    let mut stats = RunStats::default();
    if !use_variants {
        if over > 0 {
            out.push(Run { variant: FxVariant::Union, first: 0, count: over });
        }
        if screened > 0 {
            out.push(Run { variant: FxVariant::ScreenUnion, first: over, count: screened });
        }
        stats.draws = out.len() as u32;
        stats.union_instances = over;
        return (out, stats);
    }
    let mut kinds: Vec<Run> = Vec::new();
    for (i, inst) in frame.instances[..over as usize].iter().enumerate() {
        let variant = FxVariant::of(inst, false);
        match kinds.last_mut() {
            Some(run) if run.variant == variant => run.count += 1,
            _ => kinds.push(Run { variant, first: i as u32, count: 1 }),
        }
    }
    stats.kinds_runs = kinds.len() as u32;
    if kinds.len() <= MAX_OVER_RUNS {
        out.extend(kinds);
    } else {
        // The largest few keep their own shaders; with a stretch before,
        // between and after them that is at most MAX_OVER_RUNS draws.
        let anchors = (MAX_OVER_RUNS - 1) / 2;
        let mut by_size: Vec<usize> = (0..kinds.len()).collect();
        by_size.sort_by(|a, b| kinds[*b].count.cmp(&kinds[*a].count).then(a.cmp(b)));
        let mut keep = by_size[..anchors].to_vec();
        keep.sort();
        let stretch = |from: usize, to: usize, out: &mut Vec<Run>, stats: &mut RunStats| match to - from {
            0 => {}
            1 => out.push(kinds[from]),
            _ => {
                let (first, end) = (kinds[from].first, kinds[to - 1].first + kinds[to - 1].count);
                stats.union_instances += end - first;
                out.push(Run { variant: FxVariant::Union, first, count: end - first });
            }
        };
        let mut next = 0;
        for k in keep {
            stretch(next, k, &mut out, &mut stats);
            out.push(kinds[k]);
            next = k + 1;
        }
        stretch(next, kinds.len(), &mut out, &mut stats);
    }
    let heat = frame.screened_heat.min(screened);
    if screened > heat {
        out.push(Run { variant: FxVariant::ScreenPlain, first: over, count: screened - heat });
    }
    if heat > 0 {
        out.push(Run { variant: FxVariant::ScreenHeat, first: over + screened - heat, count: heat });
    }
    stats.draws = out.len() as u32;
    (out, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::shader_inlining::inlined_sizes;

    fn proxy(src: &str, entry: &str) -> usize {
        inlined_sizes(src, entry).iter().map(|f| f.own * f.copies).sum()
    }

    /// PROXY UNITS TO INSTRUCTIONS, until the effects' own PIPESTATS are
    /// logged: the largest ratio measured on this engine's shaders (the
    /// scene readers' 1.87; the ground's is 1.35-1.38), so the limit below
    /// errs toward refusing.
    const PROXY_TO_INSTRUCTIONS: f32 = 1.87;
    /// PROVISIONAL (plan 5.1): the most instructions an effect's variant may
    /// be. Effects are the most overdrawn pixels in the frame.
    const LIMIT_INSTRUCTIONS: f32 = 1200.0;

    /// EVERY VARIANT VALIDATES, with its SpaceWarp twin, and is the union's
    /// own text less whole lines: nothing in a variant is written twice.
    #[test]
    fn every_variant_and_its_motion_twin_validates_and_is_cut_from_the_union() {
        let union = effects_shader();
        let lines: std::collections::HashSet<&str> = union.lines().collect();
        for v in FxVariant::ALL {
            let src = v.shader();
            let error = crate::renderer::multiview::multiview_validation_error_of(&src, false);
            assert!(error.is_none(), "{v:?}: {error:?}");
            let (twin, entry) = v.motion_shader();
            let error = crate::renderer::multiview::multiview_validation_error_of(&twin, false);
            assert!(error.is_none(), "{v:?}'s motion twin: {error:?}");
            assert!(twin.contains(&format!("fn {entry}(")), "{v:?}");
            // A variant's lines are the union's, but for the few rewritten
            // to one kind (which name no `six` and no `select` on the kind).
            let own: Vec<&str> = src.lines().filter(|l| !lines.contains(l.trim_start()) && !lines.contains(l) && !lines.iter().any(|u| u.trim_start() == l.trim_start())).collect();
            assert!(own.len() <= 4, "{v:?} has lines that are not the union's: {own:#?}");
            assert!(src.len() <= union.len(), "{v:?}");
            if !matches!(v, FxVariant::Union | FxVariant::ScreenUnion) {
                assert!(src.len() < union.len(), "{v:?}");
                assert!(v.screened() || !src.contains("// <fx:"), "{v:?} keeps a block that is not its own");
            }
        }
        for (name, entry, _) in EFFECT_MODULE_CUTS {
            let src = module_cut(name).expect(name);
            let error = crate::renderer::multiview::multiview_validation_error_of(&src, false);
            assert!(error.is_none(), "cut {name}: {error:?}");
            assert!(src.contains(&format!("fn {entry}(")));
        }
        assert!(module_cut("no_such").is_none());
        let error = crate::renderer::multiview::multiview_validation_error_of(&overdraw_shader(), false);
        assert!(error.is_none(), "{error:?}");
    }

    /// The screened shader embers, dust and drops are drawn by is the one
    /// they were drawn by before a flash existed: the union's `fs_screen`
    /// with the flash's line gone, and no heat ramp in what it reaches.
    #[test]
    fn the_plain_screened_variant_carries_none_of_a_flashs_code() {
        let plain = FxVariant::ScreenPlain.shader();
        assert!(plain.contains("    let light = in.colour.rgb;\n    return vec4<f32>(aces_fitted(light * strength), 0.0);"));
        assert!(proxy(&plain, "fs_screen") < proxy(&FxVariant::ScreenHeat.shader(), "fs_screen"));
        assert!(proxy(&plain, "fs_screen") < proxy(&effects_shader(), "fs_screen"));
        assert!(!inlined_sizes(&plain, "fs_screen").iter().any(|f| f.name == "fx_flame"));
    }

    /// TODAY'S EFFECT SHADER BRANCHES, by the host proxy: `cargo test
    /// --release --lib todays_effect_shader_branches -- --nocapture` prints
    /// the table (the union, each variant, each module's cut and what the
    /// module therefore costs). And the check the plan asks for: no variant
    /// over the instruction limit by the proxy, and each smaller than the
    /// union it is cut from.
    #[test]
    fn todays_effect_shader_branches_by_the_proxy() {
        let union_over = proxy(&effects_shader(), "fs_over");
        let union_screen = proxy(&effects_shader(), "fs_screen");
        let vertex = proxy(&effects_shader(), "vs_main");
        eprintln!("vertex stage (every variant): {vertex} by the proxy");
        eprintln!("{:<24} {:>6} {:>9} {:>7}  of the union", "variant", "proxy", "~instr", "fetches");
        for v in FxVariant::ALL {
            let src = v.shader();
            let p = proxy(&src, v.entry());
            let against = if v.screened() { union_screen } else { union_over };
            // Texture reads in its own fragment body, as written.
            let body = &src[src.find(&format!("fn {}(", v.entry())).unwrap()..];
            let body = &body[..body.find("\n}\n").unwrap()];
            let fetches = body.matches("textureSample").count() + if body.contains("fx_depth_fade(in)") { 1 } else { 0 };
            eprintln!("{:<24} {:>6} {:>9.0} {:>7}  {:.0}%", v.label(), p, p as f32 * PROXY_TO_INSTRUCTIONS, fetches, 100.0 * p as f32 / against as f32);
            assert!(p <= against, "{v:?} is larger than the union");
            assert!(p as f32 * PROXY_TO_INSTRUCTIONS <= LIMIT_INSTRUCTIONS, "{v:?}: {p} by the proxy is over the limit");
        }
        eprintln!("{:<24} {:>6} {:>9}  the module's own cost", "cut", "proxy", "module");
        for (name, entry, what) in EFFECT_MODULE_CUTS {
            let p = proxy(&module_cut(name).unwrap(), entry);
            let whole = if *entry == "fs_screen" { union_screen } else { union_over };
            eprintln!("{:<24} {:>6} {:>9}  {what}", name, p, whole as i64 - p as i64);
        }
        // What a smoke puff's pixel no longer carries.
        assert!(proxy(&FxVariant::Sixway.shader(), "fs_over") < union_over * 3 / 4);
        assert!(proxy(&FxVariant::Plain.shader(), "fs_over") < proxy(&FxVariant::Sixway.shader(), "fs_over"));
    }

    fn instance(shade: f32) -> EffectInstance {
        EffectInstance { params: [0.0, 0.0, 0.0, shade], ..Default::default() }
    }

    fn frame(over: &[f32], screened: &[f32]) -> EffectFrame {
        let heat = screened.iter().filter(|s| **s > 3.5).count() as u32;
        EffectFrame {
            instances: over.iter().chain(screened).map(|s| instance(*s)).collect(),
            over: over.len() as u32,
            screened: screened.len() as u32,
            screened_heat: heat,
            ..Default::default()
        }
    }

    /// Every instance is drawn exactly once, in order, by a shader that
    /// holds its kind; few kinds take few draws and no union; many
    /// interleaved take at most the cap, the big runs keeping theirs.
    #[test]
    fn a_frames_runs_draw_everything_once_in_order_within_the_cap() {
        let check = |f: &EffectFrame| {
            let (runs, stats) = runs(f, true);
            let mut next = 0;
            for r in &runs {
                assert_eq!(r.first, next, "{runs:?}");
                assert!(r.count > 0);
                for i in r.first..r.first + r.count {
                    let screened = i >= f.over;
                    let own = FxVariant::of(&f.instances[i as usize], screened);
                    assert!(r.variant == own || (!screened && r.variant == FxVariant::Union), "{i}: {own:?} drawn by {:?}", r.variant);
                }
                next += r.count;
            }
            assert_eq!(next, f.over + f.screened);
            assert!(runs.iter().filter(|r| !r.variant.screened()).count() <= MAX_OVER_RUNS);
            assert_eq!(stats.draws as usize, runs.len());
            (runs, stats)
        };
        // A campfire seen so its kinds lie apart: coals, flames, smoke.
        let (runs, stats) = check(&frame(&[3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0], &[0.0, 0.0, -1.0]));
        assert_eq!(runs.iter().map(|r| r.variant).collect::<Vec<_>>(), [FxVariant::Coals, FxVariant::Flame, FxVariant::Sixway, FxVariant::ScreenPlain]);
        assert_eq!((stats.union_instances, stats.kinds_runs), (0, 3));
        // A flash among embers: its own draw, after theirs.
        let (runs, _) = check(&frame(&[1.0], &[0.0, 0.0, 4.0, 4.0]));
        assert_eq!(runs.last().unwrap(), &Run { variant: FxVariant::ScreenHeat, first: 3, count: 2 });
        // Flames and smoke shuffled finely, with a long run of smoke behind
        // and another in front.
        let mut over = vec![1.0; 40];
        over.extend((0..30).map(|k| if k % 2 == 0 { 2.0 } else { 1.0 }));
        over.extend(vec![1.0; 25]);
        let (runs, stats) = check(&frame(&over, &[]));
        assert!(stats.kinds_runs > 20 && stats.union_instances > 0 && stats.union_instances <= 32, "{stats:?}");
        assert!(runs.iter().any(|r| r.variant == FxVariant::Sixway && r.count >= 25), "{runs:?}");
        // The A/B: everything by the union.
        let (all, stats) = super::runs(&frame(&over, &[0.0, 4.0]), false);
        assert_eq!(all.iter().map(|r| r.variant).collect::<Vec<_>>(), [FxVariant::Union, FxVariant::ScreenUnion]);
        assert_eq!(stats.union_instances, over.len() as u32);
        check(&frame(&[], &[]));
    }
}
