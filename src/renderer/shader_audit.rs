//! WHAT A HOT SHADER COMPUTES PER PIXEL THAT IS NOT PER PIXEL, and where its
//! costly built-ins are (tests only; `docs/shader-optimization-2026-10-12.md`).
//!
//! Two reports over a shader's LIVE code (constant branches followed as the
//! compiler follows them, each function weighted by the copies inlining
//! makes -- see `shader_inlining`):
//!
//! - [`hoistable`]: every maximal expression whose inputs are only uniforms,
//!   constants and loop counters -- the same for every pixel of a draw, or
//!   for every pixel at one light of a loop -- with its source text, the
//!   operations it costs by `shader_inlining`'s proxy, and the special
//!   functions in it. Each is work a CPU could do once a frame (or once a
//!   light) and hand over in a uniform. A value read only under such an
//!   expression is found the same way.
//! - [`specials`]: the special-function instructions (reciprocal, square
//!   roots, powers, exponentials, trigonometry) each function's live code
//!   asks for, by kind, times its copies.
//!
//! Proxies read from the source, as `shader_inlining` is: the compiler folds
//! and shares as it likes, so a candidate counts only once the pipeline
//! statistics move (the handoff's acceptance rule 2).

use naga::valid::{Capabilities, FunctionInfo, ValidationFlags, Validator};
use naga::{
    AddressSpace, BinaryOperator, Block, Expression, Function, Handle, LocalVariable, MathFunction, Module, Statement,
    TypeInner,
};
use std::collections::{BTreeMap, HashMap};
use wgpu::naga;

/// How often a value changes: never, once a draw (uniforms), or by pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Rate {
    Constant,
    Uniform,
    Pixel,
}

/// One per-draw (or per-light) expression a shader evaluates per pixel.
pub(crate) struct Hoistable {
    pub function: String,
    /// Copies of the function inlining makes.
    pub copies: usize,
    /// Its source text.
    pub text: String,
    /// Operations in it, by `shader_inlining`'s proxy.
    pub ops: usize,
    /// Special functions in it (see [`special_of`]).
    pub special: usize,
    /// Inside a loop: per iteration (per light), not per frame.
    pub in_loop: bool,
}

struct Walk<'a> {
    module: &'a Module,
    function: &'a Function,
    info: &'a FunctionInfo,
    args: &'a [Rate],
    locals: HashMap<Handle<LocalVariable>, Rate>,
    /// Call results, by the result expression.
    results: HashMap<Handle<Expression>, Rate>,
    /// Live emitted expressions, and whether each is inside a loop.
    emitted: Vec<(Handle<Expression>, bool)>,
    calls: Vec<(Handle<Function>, Vec<Rate>)>,
    returned: Rate,
    changed: bool,
    rates: &'a mut HashMap<(Handle<Function>, Vec<Rate>), Rate>,
}

impl Walk<'_> {
    fn rate(&self, h: Handle<Expression>) -> Rate {
        use Expression as E;
        let max = |a: Rate, b: Rate| a.max(b);
        match self.function.expressions[h] {
            E::Literal(_) | E::Constant(_) | E::ZeroValue(_) | E::Override(_) => Rate::Constant,
            E::GlobalVariable(g) => match self.module.global_variables[g].space {
                AddressSpace::Uniform | AddressSpace::Handle | AddressSpace::Immediate => Rate::Uniform,
                _ => Rate::Pixel,
            },
            E::FunctionArgument(i) => self.args.get(i as usize).copied().unwrap_or(Rate::Pixel),
            E::LocalVariable(v) => self.locals.get(&v).copied().unwrap_or(Rate::Constant),
            E::Load { pointer } => self.rate(pointer),
            E::AccessIndex { base, .. } => self.rate(base),
            E::Access { base, index } => max(self.rate(base), self.rate(index)),
            E::Compose { ref components, .. } => components.iter().map(|&c| self.rate(c)).max().unwrap_or(Rate::Constant),
            E::Splat { value, .. } => self.rate(value),
            E::Swizzle { vector, .. } => self.rate(vector),
            E::Unary { expr, .. } => self.rate(expr),
            E::Binary { left, right, .. } => max(self.rate(left), self.rate(right)),
            E::Select { condition, accept, reject } => max(self.rate(condition), max(self.rate(accept), self.rate(reject))),
            E::Relational { argument, .. } => self.rate(argument),
            E::As { expr, .. } => self.rate(expr),
            E::Math { arg, arg1, arg2, arg3, .. } => {
                [Some(arg), arg1, arg2, arg3].into_iter().flatten().map(|a| self.rate(a)).max().unwrap()
            }
            E::ImageQuery { .. } => Rate::Uniform,
            E::CallResult(_) => self.results.get(&h).copied().unwrap_or(Rate::Pixel),
            _ => Rate::Pixel,
        }
    }

    fn store(&mut self, pointer: Handle<Expression>, value: Rate) {
        // The variable under the pointer, through any indexing.
        let mut p = pointer;
        let mut rate = value;
        loop {
            match self.function.expressions[p] {
                Expression::AccessIndex { base, .. } => p = base,
                Expression::Access { base, index } => {
                    rate = rate.max(self.rate(index));
                    p = base;
                }
                _ => break,
            }
        }
        if let Expression::LocalVariable(v) = self.function.expressions[p] {
            let was = self.locals.get(&v).copied().unwrap_or(Rate::Constant);
            if rate > was {
                self.locals.insert(v, rate);
                self.changed = true;
            }
        }
    }

    /// Walks live statements under `control` (the rate of the conditions
    /// that decide whether they run). Returns whether the block always ends
    /// the function, and whether it can leave early by a pixel's choice.
    fn walk(&mut self, block: &Block, mut control: Rate, in_loop: bool, record: bool) -> (bool, bool) {
        let mut leaves = false;
        for statement in block.iter() {
            match statement {
                Statement::Emit(range) => {
                    if record {
                        self.emitted.extend(range.clone().map(|h| (h, in_loop)));
                    }
                }
                Statement::Block(b) => {
                    let (ends, l) = self.walk(b, control, in_loop, record);
                    leaves |= l;
                    if l {
                        control = Rate::Pixel;
                    }
                    if ends {
                        return (true, leaves);
                    }
                }
                Statement::If { condition, accept, reject } => {
                    let known = super::shader_inlining::const_bool(self.module, &self.function.expressions, *condition);
                    match known {
                        Some(true) | Some(false) => {
                            let (ends, l) = self.walk(if known == Some(true) { accept } else { reject }, control, in_loop, record);
                            leaves |= l;
                            if l {
                                control = Rate::Pixel;
                            }
                            if ends {
                                return (true, leaves);
                            }
                        }
                        None => {
                            let inner = control.max(self.rate(*condition));
                            let (a, la) = self.walk(accept, inner, in_loop, record);
                            let (r, lr) = self.walk(reject, inner, in_loop, record);
                            let exits = has_exit(accept) || has_exit(reject);
                            if la || lr || (exits && inner == Rate::Pixel) {
                                leaves = true;
                                control = Rate::Pixel;
                            }
                            if a && r {
                                return (true, leaves);
                            }
                        }
                    }
                }
                Statement::Switch { selector, cases } => {
                    let inner = control.max(self.rate(*selector));
                    for c in cases {
                        let (_, l) = self.walk(&c.body, inner, in_loop, record);
                        leaves |= l;
                    }
                }
                Statement::Loop { body, continuing, break_if } => {
                    // The counter's own step runs at the loop's rate; what a
                    // pixel's early exit leaves behind is the body's affair.
                    let (_, _) = self.walk(body, control, true, record);
                    self.walk(continuing, control, true, record);
                    if let Some(b) = break_if {
                        if self.rate(*b) == Rate::Pixel {
                            control = Rate::Pixel;
                        }
                    }
                }
                Statement::Call { function, arguments, result } => {
                    let args: Vec<Rate> = arguments.iter().map(|&a| self.rate(a).max(control)).collect();
                    let r = function_rate(self.module, *function, &args, self.rates);
                    if let Some(result) = result {
                        self.results.insert(*result, r);
                    }
                    if record {
                        self.calls.push((*function, args));
                    }
                }
                Statement::Store { pointer, value } => {
                    let r = self.rate(*value).max(control);
                    self.store(*pointer, r);
                }
                Statement::Return { value } => {
                    let r = value.map_or(Rate::Constant, |v| self.rate(v)).max(control);
                    self.returned = self.returned.max(r);
                    return (true, leaves);
                }
                Statement::Kill => return (true, leaves),
                _ => {}
            }
        }
        (false, leaves)
    }
}

fn has_exit(block: &Block) -> bool {
    block.iter().any(|s| match s {
        Statement::Break | Statement::Continue | Statement::Return { .. } | Statement::Kill => true,
        Statement::Block(b) => has_exit(b),
        Statement::If { accept, reject, .. } => has_exit(accept) || has_exit(reject),
        _ => false,
    })
}

thread_local! {
    static INFO: std::cell::RefCell<Option<naga::valid::ModuleInfo>> = const { std::cell::RefCell::new(None) };
}

/// The rate of what `f` returns when called with arguments at `args`.
fn function_rate(module: &Module, f: Handle<Function>, args: &[Rate], rates: &mut HashMap<(Handle<Function>, Vec<Rate>), Rate>) -> Rate {
    let key = (f, args.to_vec());
    if let Some(&r) = rates.get(&key) {
        return r;
    }
    // Recursion is not WGSL; a placeholder all the same.
    rates.insert(key.clone(), Rate::Pixel);
    let r = INFO.with(|info| {
        let info = info.borrow();
        let info = info.as_ref().expect("module info set");
        analyse(module, &module.functions[f], &info[f], args, rates).returned
    });
    rates.insert(key, r);
    r
}

struct Analysed {
    returned: Rate,
    emitted: Vec<(Handle<Expression>, bool, Rate)>,
    calls: Vec<(Handle<Function>, Vec<Rate>)>,
}

fn analyse(
    module: &Module,
    function: &Function,
    info: &FunctionInfo,
    args: &[Rate],
    rates: &mut HashMap<(Handle<Function>, Vec<Rate>), Rate>,
) -> Analysed {
    let mut locals = HashMap::new();
    for (v, var) in function.local_variables.iter() {
        if var.init.is_some() {
            locals.insert(v, Rate::Constant);
        }
    }
    let mut w = Walk {
        module,
        function,
        info,
        args,
        locals,
        results: HashMap::new(),
        emitted: Vec::new(),
        calls: Vec::new(),
        returned: Rate::Constant,
        changed: true,
        rates,
    };
    // Local variables' rates to a fixed point, then once more to record.
    let mut rounds = 0;
    while w.changed && rounds < 16 {
        w.changed = false;
        w.returned = Rate::Constant;
        // A local's initialiser.
        for (v, var) in function.local_variables.iter() {
            if let Some(init) = var.init {
                let r = w.rate(init);
                if r > w.locals.get(&v).copied().unwrap_or(Rate::Constant) {
                    w.locals.insert(v, r);
                    w.changed = true;
                }
            }
        }
        w.walk(&function.body, Rate::Constant, false, false);
        rounds += 1;
    }
    w.returned = Rate::Constant;
    w.walk(&function.body, Rate::Constant, false, true);
    let emitted = w.emitted.iter().map(|&(h, l)| (h, l, w.rate(h))).collect();
    let _ = w.info;
    Analysed { returned: w.returned, emitted, calls: w.calls }
}

fn width(module: &Module, info: &FunctionInfo, h: Handle<Expression>) -> usize {
    match *info[h].ty.inner_with(&module.types) {
        TypeInner::Vector { size, .. } => size as usize,
        TypeInner::Matrix { columns, rows, .. } => columns as usize * rows as usize,
        _ => 1,
    }
}

/// `shader_inlining`'s proxy for one expression's operations.
fn ops_of(module: &Module, function: &Function, info: &FunctionInfo, h: Handle<Expression>) -> usize {
    match function.expressions[h] {
        Expression::Binary { .. } | Expression::Unary { .. } | Expression::Select { .. } | Expression::As { .. } => width(module, info, h),
        Expression::Relational { argument, .. } => width(module, info, argument),
        Expression::Math { fun, arg, .. } => {
            let n = width(module, info, arg);
            match fun {
                MathFunction::Dot | MathFunction::Length => n,
                MathFunction::Distance | MathFunction::Normalize | MathFunction::Reflect => 2 * n + 1,
                MathFunction::Mix | MathFunction::Clamp | MathFunction::Cross => 2 * n,
                MathFunction::Pow | MathFunction::FaceForward => 3 * n,
                MathFunction::SmoothStep | MathFunction::Refract => 5 * n,
                _ => n,
            }
        }
        _ => 0,
    }
}

/// The special-function instructions one expression asks for, by kind: what
/// the Adreno's statistics count as "complex" -- a reciprocal, a square root
/// or its inverse, a power, an exponential or logarithm, trigonometry. A
/// division by a constant is a multiply and counts nothing; `normalize`,
/// `length` and `distance` are one (inverse) square root; `pow` is a
/// logarithm and an exponential a component.
pub(crate) fn special_of(module: &Module, function: &Function, info: &FunctionInfo, h: Handle<Expression>) -> Option<(&'static str, usize)> {
    let is_const = |e: Handle<Expression>| {
        matches!(function.expressions[e], Expression::Literal(_) | Expression::Constant(_) | Expression::ZeroValue(_))
    };
    match function.expressions[h] {
        Expression::Binary { op: BinaryOperator::Divide, right, .. } if !is_const(right) => {
            match *info[h].ty.inner_with(&module.types) {
                TypeInner::Scalar(s) | TypeInner::Vector { scalar: s, .. } if s.kind == naga::ScalarKind::Float => {
                    Some(("rcp", width(module, info, right)))
                }
                _ => None,
            }
        }
        Expression::Math { fun, arg, .. } => {
            let n = width(module, info, arg);
            Some(match fun {
                MathFunction::Sin | MathFunction::Cos | MathFunction::Tan => ("sin/cos", n),
                MathFunction::Asin | MathFunction::Acos | MathFunction::Atan | MathFunction::Atan2 => ("atan/acos", n),
                MathFunction::Pow => ("pow", 2 * n),
                MathFunction::Exp | MathFunction::Exp2 => ("exp", n),
                MathFunction::Log | MathFunction::Log2 => ("log", n),
                MathFunction::Sqrt => ("sqrt", n),
                MathFunction::InverseSqrt => ("rsqrt", n),
                MathFunction::Normalize => ("rsqrt (normalize)", 1),
                MathFunction::Length | MathFunction::Distance => ("sqrt (length)", 1),
                MathFunction::Refract => ("sqrt", 1),
                _ => return None,
            })
        }
        _ => None,
    }
}

fn children(function: &Function, h: Handle<Expression>) -> Vec<Handle<Expression>> {
    use Expression as E;
    match function.expressions[h] {
        E::AccessIndex { base, .. } => vec![base],
        E::Access { base, index } => vec![base, index],
        E::Compose { ref components, .. } => components.clone(),
        E::Splat { value, .. } => vec![value],
        E::Swizzle { vector, .. } => vec![vector],
        E::Unary { expr, .. } => vec![expr],
        E::Binary { left, right, .. } => vec![left, right],
        E::Select { condition, accept, reject } => vec![condition, accept, reject],
        E::Relational { argument, .. } => vec![argument],
        E::As { expr, .. } => vec![expr],
        E::Math { arg, arg1, arg2, arg3, .. } => [Some(arg), arg1, arg2, arg3].into_iter().flatten().collect(),
        E::Load { pointer } => vec![pointer],
        _ => Vec::new(),
    }
}

fn parse(src: &str) -> (Module, naga::valid::ModuleInfo) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
    (module, info)
}

/// Everything live from `entry`, as `(function or None for the entry point,
/// its argument rates, copies, analysis)`.
fn reach<'m>(module: &'m Module, info: &naga::valid::ModuleInfo, entry: &str) -> Vec<(Option<Handle<Function>>, usize, Analysed)> {
    let index = module.entry_points.iter().position(|e| e.name == entry).unwrap_or_else(|| panic!("no entry point {entry}"));
    INFO.with(|slot| *slot.borrow_mut() = Some(info.clone()));
    let mut rates = HashMap::new();
    let ep = &module.entry_points[index].function;
    let ep_args = vec![Rate::Pixel; ep.arguments.len()];
    let top = analyse(module, ep, info.get_entry_point(index), &ep_args, &mut rates);
    // Copies of each (function, argument rates), down the live call tree.
    let mut copies: BTreeMap<(usize, Vec<u8>), (Handle<Function>, Vec<Rate>, usize)> = BTreeMap::new();
    let mut stack = top.calls.clone();
    let mut done: HashMap<(Handle<Function>, Vec<Rate>), Vec<(Handle<Function>, Vec<Rate>)>> = HashMap::new();
    while let Some((f, args)) = stack.pop() {
        let key = (f.index(), args.iter().map(|&r| r as u8).collect::<Vec<u8>>());
        copies.entry(key).or_insert((f, args.clone(), 0)).2 += 1;
        let calls = done.entry((f, args.clone())).or_insert_with(|| analyse(module, &module.functions[f], &info[f], &args, &mut rates).calls);
        stack.extend(calls.iter().cloned());
    }
    let mut out = vec![(None, 1, top)];
    for (_, (f, args, n)) in copies {
        out.push((Some(f), n, analyse(module, &module.functions[f], &info[f], &args, &mut rates)));
    }
    out
}

/// Every maximal per-draw (or per-light) expression `entry`'s live code
/// evaluates per pixel, the costliest first.
pub(crate) fn hoistable(src: &str, entry: &str) -> Vec<Hoistable> {
    let (module, info) = parse(src);
    let index = module.entry_points.iter().position(|e| e.name == entry).unwrap();
    let mut merged: BTreeMap<(String, String), Hoistable> = BTreeMap::new();
    for (f, copies, a) in reach(&module, &info, entry) {
        let (function, finfo, name) = match f {
            Some(h) => (&module.functions[h], &info[h], module.functions[h].name.clone().unwrap_or_default()),
            None => (&module.entry_points[index].function, info.get_entry_point(index), entry.to_string()),
        };
        let rate: HashMap<Handle<Expression>, (Rate, bool)> = a.emitted.iter().map(|&(h, l, r)| (h, (r, l))).collect();
        let weighted = |h: Handle<Expression>| ops_of(&module, function, finfo, h) > 0;
        let uniform = |h: Handle<Expression>| rate.get(&h).is_some_and(|&(r, _)| r == Rate::Uniform);
        // Uniform-rate weighted expressions that feed another one are part of it.
        let mut inner: std::collections::HashSet<Handle<Expression>> = Default::default();
        for &(h, _, r) in &a.emitted {
            if r == Rate::Uniform {
                // Children through weightless wrappers (loads, swizzles, composes) too.
                let mut stack = children(function, h);
                while let Some(c) = stack.pop() {
                    if uniform(c) {
                        inner.insert(c);
                    } else if !rate.contains_key(&c) {
                        stack.extend(children(function, c));
                    }
                }
            }
        }
        for &(h, in_loop, r) in &a.emitted {
            if r != Rate::Uniform || inner.contains(&h) {
                continue;
            }
            // Its whole subtree's cost.
            let (mut ops, mut special) = (0, 0);
            let mut seen: std::collections::HashSet<Handle<Expression>> = Default::default();
            let mut stack = vec![h];
            while let Some(e) = stack.pop() {
                if !seen.insert(e) {
                    continue;
                }
                if uniform(e) || e == h {
                    ops += ops_of(&module, function, finfo, e);
                    special += special_of(&module, function, finfo, e).map_or(0, |(_, n)| n);
                }
                stack.extend(children(function, e));
            }
            if ops == 0 || !(weighted(h) || special > 0) {
                continue;
            }
            // A condition (a loop's bound) is not work worth handing over.
            if matches!(*finfo[h].ty.inner_with(&module.types), TypeInner::Scalar(s) | TypeInner::Vector { scalar: s, .. } if s.kind == naga::ScalarKind::Bool) {
                continue;
            }
            // Its source: the expression's own text where the span covers it
            // (operators), its whole line where the span is only a built-in's
            // name.
            let span = function.expressions.get_span(h);
            let own = span.to_range().and_then(|r| src.get(r.clone()).map(|t| (r, t)));
            let text = match own {
                Some((r, t)) if !t.contains(['(', ' ', '*', '+', '-', '/']) => {
                    let start = src[..r.start].rfind('\n').map_or(0, |i| i + 1);
                    let end = src[r.end..].find('\n').map_or(src.len(), |i| r.end + i);
                    format!("{t}: {}", src[start..end].trim())
                }
                Some((_, t)) => t.split_whitespace().collect::<Vec<_>>().join(" "),
                None => "?".to_string(),
            };
            let key = (name.clone(), text.clone());
            let e = merged.entry(key).or_insert(Hoistable { function: name.clone(), copies, text, ops: 0, special: 0, in_loop });
            e.ops = e.ops.max(ops);
            e.special = e.special.max(special);
        }
    }
    let mut out: Vec<Hoistable> = merged.into_values().collect();
    out.sort_by(|a, b| (b.ops * b.copies + 4 * b.special * b.copies).cmp(&(a.ops * a.copies + 4 * a.special * a.copies)));
    out
}

/// The special functions `entry`'s live code asks for: per function, its
/// copies and its count of each kind; and each kind's total.
pub(crate) fn specials(src: &str, entry: &str) -> (Vec<(String, usize, BTreeMap<&'static str, usize>)>, BTreeMap<&'static str, usize>) {
    let (module, info) = parse(src);
    let index = module.entry_points.iter().position(|e| e.name == entry).unwrap();
    let mut rows: BTreeMap<String, (usize, BTreeMap<&'static str, usize>)> = BTreeMap::new();
    for (f, copies, a) in reach(&module, &info, entry) {
        let (function, finfo, name) = match f {
            Some(h) => (&module.functions[h], &info[h], module.functions[h].name.clone().unwrap_or_default()),
            None => (&module.entry_points[index].function, info.get_entry_point(index), entry.to_string()),
        };
        let row = rows.entry(name).or_insert((0, BTreeMap::new()));
        row.0 += copies;
        let mut own = BTreeMap::new();
        for &(h, _, _) in &a.emitted {
            if let Some((kind, n)) = special_of(&module, function, finfo, h) {
                *own.entry(kind).or_insert(0) += n;
            }
        }
        // The same function at several argument rates is the same code.
        for (k, n) in own {
            let e = row.1.entry(k).or_insert(0);
            *e = (*e).max(n);
        }
    }
    let mut total = BTreeMap::new();
    let mut out: Vec<(String, usize, BTreeMap<&'static str, usize>)> = Vec::new();
    for (name, (copies, kinds)) in rows {
        for (k, n) in &kinds {
            *total.entry(*k).or_insert(0) += n * copies;
        }
        if !kinds.is_empty() {
            out.push((name, copies, kinds));
        }
    }
    out.sort_by_key(|(_, copies, kinds)| std::cmp::Reverse(copies * kinds.values().sum::<usize>()));
    (out, total)
}

/// Both reports as text.
pub(crate) fn report(label: &str, src: &str, entry: &str, rows: usize) -> String {
    let mut out = format!("== {label}\n");
    let (by_fn, total) = specials(src, entry);
    let all: usize = total.values().sum();
    out += &format!("special functions (source, live, x copies): {all}  {total:?}\n");
    for (name, copies, kinds) in by_fn.iter().take(rows) {
        out += &format!("  {:>4} x{copies:<2} {name}  {kinds:?}\n", copies * kinds.values().sum::<usize>());
    }
    let h = hoistable(src, entry);
    let ops: usize = h.iter().map(|e| e.ops * e.copies).sum();
    let special: usize = h.iter().map(|e| e.special * e.copies).sum();
    out += &format!("per-draw or per-light expressions evaluated per pixel: {} ({ops} ops, {special} special)\n", h.len());
    for e in h.iter().take(rows) {
        let text: String = e.text.chars().take(110).collect();
        out += &format!(
            "  {:>4} ops {:>2} special x{:<2} {}{}: {}\n",
            e.ops * e.copies,
            e.special * e.copies,
            e.copies,
            e.function,
            if e.in_loop { " [loop]" } else { "" },
            text
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE AUDIT of the hot shaders: each one's generated source and both
    /// reports, written under `docs/bench/shader_opt_1012/audit/`.
    /// `cargo test --lib audit_the_hot_shaders -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn audit_the_hot_shaders() {
        use crate::renderer::terrain_pipeline::{terrain_shader_for, TerrainRole};
        use crate::renderer::{cave_twins, ground_twins, weather};
        let mut sources = crate::renderer::brush_pipeline::audit_sources();
        let ground = terrain_shader_for(TerrainRole::Read);
        let gentle = ground_twins::gentle_shader(&ground).expect("gentle");
        sources.push(("terrain_read_gentle", gentle.clone()));
        sources.push(("terrain_read", ground));
        if let Some(wet) = weather::with_weather(&gentle) {
            sources.push(("terrain_weather_gentle_read", wet));
        }
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        if let Some(wet) = weather::with_weather(&pass) {
            sources.push(("terrain_weather_probe_pass", wet));
        }
        sources.push(("terrain_probe_pass", pass));
        sources.push(("cave_full", crate::renderer::layered_mesh_pipeline::layered_mesh_shader()));
        sources.push(("cave_rock", cave_twins::twin_source(cave_twins::CavePath::Rock).unwrap()));
        sources.push(("cave_ground", cave_twins::twin_source(cave_twins::CavePath::Ground).unwrap()));
        sources.push(("water", crate::renderer::water_pipeline::water_shader_src()));
        sources.push(("mesh", crate::renderer::mesh_pipeline::mesh_shader_src()));
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/bench/shader_opt_1012/audit");
        std::fs::create_dir_all(&dir).unwrap();
        let mut all = String::new();
        for (label, src) in sources {
            let src = crate::renderer::multiview::ViewMode::Mono.shader(src);
            std::fs::write(dir.join(format!("{label}.wgsl")), &src).unwrap();
            let inlined = crate::renderer::shader_inlining::report(label, &src, "fs_main", 14);
            let r = report(label, &src, "fs_main", 60);
            all += &r;
            all += &inlined;
            all += "\n";
        }
        std::fs::write(dir.join("audit.txt"), &all).unwrap();
        println!("{all}");
    }

    const SRC: &str = r#"
struct Camera { yaw: f32, tint: vec3<f32>, count: u32, lights: array<vec4<f32>, 4> }
@group(0) @binding(0) var<uniform> camera: Camera;
fn turn(d: vec3<f32>) -> vec3<f32> {
    let s = sin(camera.yaw);
    let c = cos(camera.yaw);
    return vec3<f32>(c * d.x + s * d.z, d.y, c * d.z - s * d.x);
}
fn up() -> vec3<f32> {
    return turn(vec3<f32>(0.0, 1.0, 0.0));
}
@fragment fn fs_main(@location(0) n: vec3<f32>) -> @location(0) vec4<f32> {
    let a = turn(n);
    let b = turn(normalize(n));
    var sum = dot(a, b) * pow(camera.tint.x, 2.2) + up().y;
    for (var i = 0u; i < camera.count; i = i + 1u) {
        let range = camera.lights[i].w * camera.lights[i].w;
        if (dot(n, camera.lights[i].xyz) > range) {
            continue;
        }
        sum = sum + exp(-sum) / range;
    }
    return vec4<f32>(vec3<f32>(sum), 1.0);
}
"#;

    #[test]
    fn a_uniforms_sine_and_a_lights_square_are_found_and_a_pixels_are_not() {
        let h = hoistable(SRC, "fs_main");
        let text = |needle: &str| h.iter().find(|e| e.text.contains(needle));
        // The yaw's sine and cosine, in each of `turn`'s copies.
        let s = text("sin(camera.yaw)").expect("the sine");
        assert_eq!((s.function.as_str(), s.special), ("turn", 1));
        assert!(text("cos(camera.yaw)").is_some());
        // The tint's power: per draw.
        assert!(text("pow(camera.tint.x, 2.2)").is_some_and(|e| e.special == 2 && !e.in_loop));
        // `up()` turns a constant: the whole of that copy of `turn` is per draw.
        assert!(h.iter().any(|e| e.function == "turn" && e.copies == 1 && e.text.starts_with("vec3<f32>(c * d.x")));
        // A loop's bound is not listed.
        assert!(text("i < camera.count").is_none());
        // A light's range squared: per light.
        assert!(text("camera.lights[i].w * camera.lights[i].w").is_some_and(|e| e.in_loop));
        // Nothing that reads the pixel's normal is.
        assert!(h.iter().all(|e| !e.text.starts_with("normalize") && !e.text.starts_with("dot") && !e.text.starts_with("exp")), "{:?}", h.iter().map(|e| &e.text).collect::<Vec<_>>());
    }

    #[test]
    fn special_functions_are_counted_by_kind_and_copy() {
        let (rows, total) = specials(SRC, "fs_main");
        // `turn` three times (two from the entry point, one through `up`).
        let turn = rows.iter().find(|(n, _, _)| n == "turn").expect("turn");
        assert_eq!(turn.1, 3);
        assert_eq!(total["sin/cos"], 6);
        assert_eq!(total["pow"], 2);
        assert_eq!(total["exp"], 1);
        assert_eq!(total["rcp"], 1);
        assert_eq!(total["rsqrt (normalize)"], 1);
    }
}
