use wgpu::naga;
use naga::valid::{Capabilities, FunctionInfo, ValidationFlags, Validator};
use naga::{Arena, BinaryOperator, Block, Expression, Function, Handle, Literal, LocalVariable, MathFunction, Module, Statement, TypeInner, UnaryOperator};
use std::collections::HashMap;

pub(crate) struct Inlined {
    pub name: String,

    pub own: usize,

    pub copies: usize,

    pub callers: Vec<(String, usize)>,
}

impl Inlined {

    pub fn repeated(&self) -> usize {
        self.own * self.copies.saturating_sub(1)
    }
}

pub(crate) fn inlined_sizes(src: &str, entry: &str) -> Vec<Inlined> {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
    let index = module
        .entry_points
        .iter()
        .position(|e| e.name == entry)
        .unwrap_or_else(|| panic!("no entry point {entry}"));

    let mut functions: HashMap<Handle<Function>, (Vec<Handle<Function>>, usize)> = HashMap::new();
    for (handle, function) in module.functions.iter() {
        let mut live = Live { module: &module, function, info: &info[handle], calls: Vec::new(), own: 0, temps: HashMap::new() };
        live.walk(&function.body);
        functions.insert(handle, (live.calls, live.own));
    }
    let ep = &module.entry_points[index].function;
    let mut live = Live { module: &module, function: ep, info: info.get_entry_point(index), calls: Vec::new(), own: 0, temps: HashMap::new() };
    live.walk(&ep.body);

    let mut copies: HashMap<Handle<Function>, usize> = HashMap::new();
    let from_entry = live.calls.clone();
    let mut stack = live.calls;
    while let Some(f) = stack.pop() {
        *copies.entry(f).or_default() += 1;
        stack.extend(functions[&f].0.iter().copied());
    }

    let name = |h: Handle<Function>| module.functions[h].name.clone().unwrap_or_else(|| format!("{h:?}"));
    let mut callers: HashMap<Handle<Function>, Vec<(String, usize)>> = HashMap::new();
    let mut add = |callee: Handle<Function>, caller: String, n: usize| {
        let list = callers.entry(callee).or_default();
        match list.iter_mut().find(|(c, _)| *c == caller) {
            Some((_, m)) => *m += n,
            None => list.push((caller, n)),
        }
    };
    for &f in &from_entry {
        add(f, entry.to_string(), 1);
    }
    for (&g, &n) in &copies {
        for &f in &functions[&g].0 {
            add(f, name(g), n);
        }
    }
    let mut rows = vec![Inlined { name: entry.to_string(), own: live.own, copies: 1, callers: Vec::new() }];
    for (handle, _) in module.functions.iter() {
        if let Some(&n) = copies.get(&handle) {
            rows.push(Inlined {
                name: name(handle),
                own: functions[&handle].1,
                copies: n,
                callers: callers.remove(&handle).unwrap_or_default(),
            });
        }
    }
    rows
}

pub(crate) fn report(label: &str, src: &str, entry: &str, rows: usize) -> String {
    let mut all = inlined_sizes(src, entry);
    let total: usize = all.iter().map(|r| r.own * r.copies).sum();
    let repeated: usize = all.iter().map(Inlined::repeated).sum();
    all.sort_by(|a, b| b.repeated().cmp(&a.repeated()).then(b.own.cmp(&a.own)));
    let mut out = format!("{label}: {total} by the proxy, {repeated} of it in repeated copies\n");
    out += "  repeated  copies   own  function\n";
    for r in all.iter().take(rows).filter(|r| r.repeated() > 0) {
        let from: Vec<String> = r.callers.iter().map(|(c, n)| format!("{c} x{n}")).collect();
        out += &format!("  {:>8}  {:>6}  {:>4}  {}  <- {}\n", r.repeated(), r.copies, r.own, r.name, from.join(", "));
    }
    all.sort_by(|a, b| (b.own * b.copies).cmp(&(a.own * a.copies)));
    out += "  largest  copies   own  function\n";
    for r in all.iter().take(rows) {
        out += &format!("  {:>7}  {:>6}  {:>4}  {}\n", r.own * r.copies, r.copies, r.own, r.name);
    }
    out
}

struct Live<'a> {
    module: &'a Module,
    function: &'a Function,
    info: &'a FunctionInfo,
    calls: Vec<Handle<Function>>,
    own: usize,

    temps: HashMap<Handle<LocalVariable>, Option<bool>>,
}

impl Live<'_> {

    fn walk(&mut self, block: &Block) -> bool {
        for statement in block.iter() {
            let ends = match statement {
                Statement::Emit(range) => {
                    for h in range.clone() {
                        self.own += self.weight(h);
                    }
                    false
                }
                Statement::Block(b) => self.walk(b),
                Statement::If { condition, accept, reject } => match self.condition(*condition) {
                    Some(true) => self.walk(accept),
                    Some(false) => self.walk(reject),
                    None => {
                        let a = self.walk(accept);
                        let r = self.walk(reject);
                        a && r
                    }
                },
                Statement::Switch { cases, .. } => {
                    cases.iter().for_each(|c| {
                        self.walk(&c.body);
                    });
                    false
                }
                Statement::Loop { body, continuing, .. } => {
                    self.walk(body);
                    self.walk(continuing);
                    false
                }
                Statement::Call { function, .. } => {
                    self.calls.push(*function);
                    false
                }
                Statement::Store { pointer, value } => {
                    if let Some(v) = self.temp(*pointer) {
                        let now = self.condition(*value);
                        self.temps.entry(v).and_modify(|was| if *was != now { *was = None }).or_insert(now);
                    }
                    false
                }
                Statement::Return { .. } | Statement::Kill => true,
                _ => false,
            };
            if ends {
                return true;
            }
        }
        false
    }

    fn temp(&self, pointer: Handle<Expression>) -> Option<Handle<LocalVariable>> {
        match self.function.expressions[pointer] {
            Expression::LocalVariable(v) if self.function.local_variables[v].name.is_none() => Some(v),
            _ => None,
        }
    }

    fn condition(&self, h: Handle<Expression>) -> Option<bool> {
        match self.function.expressions[h] {
            Expression::Load { pointer } => self.temp(pointer).and_then(|v| self.temps.get(&v).copied().flatten()),
            Expression::Unary { op: UnaryOperator::LogicalNot, expr } => self.condition(expr).map(|b| !b),
            Expression::Binary { op: BinaryOperator::LogicalAnd, left, right } => and(self.condition(left), self.condition(right)),
            Expression::Binary { op: BinaryOperator::LogicalOr, left, right } => or(self.condition(left), self.condition(right)),
            _ => const_bool(self.module, &self.function.expressions, h),
        }
    }

    fn width(&self, h: Handle<Expression>) -> usize {
        match *self.info[h].ty.inner_with(&self.module.types) {
            TypeInner::Vector { size, .. } => size as usize,
            TypeInner::Matrix { columns, rows, .. } => columns as usize * rows as usize,
            _ => 1,
        }
    }

    fn weight(&self, h: Handle<Expression>) -> usize {
        match self.function.expressions[h] {
            Expression::Binary { .. }
            | Expression::Unary { .. }
            | Expression::Select { .. }
            | Expression::As { .. }
            | Expression::Derivative { .. } => self.width(h),
            Expression::Relational { argument, .. } => self.width(argument),
            Expression::ImageSample { .. } | Expression::ImageLoad { .. } | Expression::ImageQuery { .. } => 1,
            Expression::Math { fun, arg, .. } => {
                let n = self.width(arg);
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
}

pub(super) fn const_bool(module: &Module, arena: &Arena<Expression>, h: Handle<Expression>) -> Option<bool> {
    match arena[h] {
        Expression::Literal(Literal::Bool(b)) => Some(b),
        Expression::Constant(c) => const_bool(module, &module.global_expressions, module.constants[c].init),
        Expression::Unary { op: UnaryOperator::LogicalNot, expr } => const_bool(module, arena, expr).map(|b| !b),
        Expression::Binary { op: BinaryOperator::LogicalAnd, left, right } => {
            and(const_bool(module, arena, left), const_bool(module, arena, right))
        }
        Expression::Binary { op: BinaryOperator::LogicalOr, left, right } => {
            or(const_bool(module, arena, left), const_bool(module, arena, right))
        }
        _ => None,
    }
}

fn and(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

fn or(a: Option<bool>, b: Option<bool>) -> Option<bool> {
    match (a, b) {
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        _ => None,
    }
}
