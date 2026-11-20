//! Random worlds: pseudo-word lexicon, layered signature, rule templates.
//!
//! Layer discipline keeps every run finite and every Skolem/function term
//! shallow: all rules except `Prop` write only predicates in a strictly higher
//! layer than anything they read, and `Prop` (spreading a property along a
//! relation of no higher layer) never builds a new term. Latent predicates sit
//! in layer 0 and no rule writes layer 0, so a hidden cause can only enter a
//! run through the initial state, which means it has to be abduced.

use crate::rewrite::{Rule, State};
use crate::rng::Rng;
use crate::term::{Graph, Kind, NodeId, SymId};
use crate::util::round;
use std::collections::{HashMap, HashSet};

// ------------------------------------------------------------------ lexicon

const ONSETS: &[&str] = &[
    "b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "sh", "kr", "tr", "pl",
    "gl", "br", "st", "th",
];
const VOWELS: &[&str] = &["a", "e", "i", "o", "u", "ae", "ia", "ou"];
const CODAS: &[&str] = &["", "", "", "n", "r", "s", "l", "x"];

/// Source of pronounceable nonsense names.
pub struct Lexicon {
    used: HashSet<String>,
}

impl Lexicon {
    pub fn new() -> Lexicon {
        Lexicon { used: HashSet::new() }
    }

    /// A fresh pseudo-word of `lo..=hi` syllables; never returns a word twice.
    pub fn word(&mut self, rng: &mut Rng, lo: usize, hi: usize) -> String {
        let mut miss = 0usize;
        loop {
            let n = rng.range(lo.max(1), hi.max(1)) + miss / 16;
            let mut w = String::new();
            for i in 0..n {
                w.push_str(rng.pick(ONSETS));
                w.push_str(rng.pick(VOWELS));
                if i + 1 == n {
                    w.push_str(rng.pick(CODAS));
                }
            }
            if self.used.insert(w.clone()) {
                return w;
            }
            miss += 1;
        }
    }
}

fn cap(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

// ---------------------------------------------------------------- parameters

/// Difficulty knobs. `Params::level(1..=6)` gives a sensible ramp.
#[derive(Clone, Debug)]
pub struct Params {
    pub consts: usize,      // entities
    pub layers: usize,      // predicate layers (>= 3)
    pub width: usize,       // visible predicates per layer
    pub latent: usize,      // hidden-cause predicates
    pub rules: usize,       // rule budget, mandatory latent-cause rules included
    pub funcs: usize,       // unary function symbols
    pub leak: f64,          // chance that a rule can silently fail
    pub consume: f64,       // chance that a chain/conjunction eats its premise
    pub obs: usize,         // facts to explain
    pub known: f64,         // chance that a visible initial fact is disclosed
    pub max_expl: usize,    // cap on undisclosed facts in the explanation
    pub trials: usize,      // Monte-Carlo runs per candidate
    pub distractors: usize, // wrong answers kept next to the gold one
    pub margin: f64,        // required score gap (nats) between gold and the rest
    pub horizon: f64,       // CTMC time horizon
    pub max_steps: usize,   // reaction cap per run
    pub attempts: usize,    // rejection-sampling budget in `abduce::build`
}

impl Params {
    pub fn level(l: u32) -> Params {
        let l = l.clamp(1, 6) as usize;
        Params {
            consts: 6 + l,
            layers: 2 + (l + 1) / 2,
            width: 2 + l / 3,
            latent: 1 + l / 3,
            rules: 4 + 2 * l,
            funcs: l / 2,
            leak: 0.25,
            consume: 0.2,
            obs: 1 + l / 2,
            known: 0.7,
            max_expl: 4 + l / 2,
            trials: 48,
            distractors: 5,
            margin: 1.0,
            horizon: 8.0,
            max_steps: 150,
            attempts: 64,
        }
    }
}

// --------------------------------------------------------------------- world

#[derive(Clone, Debug)]
pub struct PredInfo {
    pub sym: SymId,
    pub arity: usize,
    pub layer: usize,
    pub latent: bool,
    pub prior: f64, // P(ground atom is in the initial state); layer 0 only
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RuleKind {
    Chain,
    Conj,
    Prop,
    Fan,
    Split,
    Consume,
    Invent,
    Func,
    Latent,
}

impl RuleKind {
    pub fn label(self) -> &'static str {
        match self {
            RuleKind::Chain => "chain",
            RuleKind::Conj => "conjunction",
            RuleKind::Prop => "propagation",
            RuleKind::Fan => "fan-out",
            RuleKind::Split => "split",
            RuleKind::Consume => "consume",
            RuleKind::Invent => "invention",
            RuleKind::Func => "function",
            RuleKind::Latent => "latent-cause",
        }
    }
}

pub struct World {
    pub g: Graph,
    pub preds: Vec<PredInfo>,
    pub consts: Vec<NodeId>,
    pub funcs: Vec<SymId>,
    pub rules: Vec<Rule>,
    pub kinds: Vec<RuleKind>,
    by_sym: HashMap<SymId, usize>, // lookups only, never iterated
    lex: Lexicon,
}

// ------------------------------------------------------------------- helpers

/// `k` distinct variables out of `vs`, in random order (`k <= vs.len()`).
fn pick_args(rng: &mut Rng, vs: &[NodeId], k: usize) -> Vec<NodeId> {
    let mut idx: Vec<usize> = (0..vs.len()).collect();
    rng.shuffle(&mut idx);
    idx.truncate(k);
    idx.into_iter().map(|i| vs[i]).collect()
}

/// One certain outcome, or (with chance `leak`) one that can silently fail.
fn outcomes(rng: &mut Rng, leak: f64, fx: Vec<NodeId>) -> Vec<(f64, Vec<NodeId>)> {
    if rng.chance(leak) {
        let e = round(rng.uniform(0.05, 0.30), 2);
        vec![(round(1.0 - e, 2), fx), (e, Vec::new())]
    } else {
        vec![(1.0, fx)]
    }
}

/// Probabilities of `k >= 2` outcomes, each about 0.1 or more, summing to 1.
fn probs(rng: &mut Rng, k: usize) -> Vec<f64> {
    let mut left = 1.0f64;
    let mut out: Vec<f64> = Vec::new();
    for i in 0..k - 1 {
        let hi = (left - 0.1 * ((k - 1 - i) as f64)).max(0.1);
        let x = round(rng.uniform(0.1, hi), 2);
        out.push(x);
        left = round(left - x, 2);
    }
    out.push(left);
    out
}

/// Random arities (1 or 2) for one layer; keeps at least one unary predicate,
/// and at least one binary one when `need2`.
fn arities(rng: &mut Rng, n: usize, p2: f64, need2: bool) -> Vec<usize> {
    let mut a: Vec<usize> = Vec::with_capacity(n);
    for _ in 0..n {
        a.push(if rng.chance(p2) { 2 } else { 1 });
    }
    if n >= 2 {
        if need2 && !a.contains(&2) {
            a[0] = 2;
        }
        if !a.contains(&1) {
            a[n - 1] = 1;
        }
    }
    a
}

fn collect_vars(g: &Graph, id: NodeId, out: &mut Vec<usize>) {
    if let Some(i) = g.var_index(id) {
        out.push(i);
        return;
    }
    for &a in g.args(id).iter() {
        collect_vars(g, a, out);
    }
}

// ---------------------------------------------------------------- generation

impl World {
    pub fn generate(rng: &mut Rng, p: &Params) -> World {
        let mut w = World {
            g: Graph::new(),
            preds: Vec::new(),
            consts: Vec::new(),
            funcs: Vec::new(),
            rules: Vec::new(),
            kinds: Vec::new(),
            by_sym: HashMap::new(),
            lex: Lexicon::new(),
        };
        for _ in 0..p.consts.max(3) {
            let name = cap(&w.lex.word(rng, 1, 2));
            let s = w.g.sym(&name, Kind::Func, 0);
            let n = w.g.mk(s, &[]);
            w.consts.push(n);
        }
        for _ in 0..p.funcs {
            let name = w.lex.word(rng, 2, 3);
            let s = w.g.sym(&name, Kind::Func, 1);
            w.funcs.push(s);
        }
        let nl = p.layers.max(3);
        for a in arities(rng, p.width.max(1), 0.35, true) {
            w.add_pred(rng, 0, a, false);
        }
        let mut hidden: Vec<usize> = Vec::new();
        for _ in 0..p.latent {
            let a = if rng.chance(0.2) { 2 } else { 1 };
            hidden.push(w.add_pred(rng, 0, a, true));
        }
        for l in 1..nl {
            for a in arities(rng, p.width.max(1), 0.3, l == 1) {
                w.add_pred(rng, l, a, false);
            }
        }
        // every hidden cause gets a rule that makes it observable
        for &h in hidden.iter() {
            for _ in 0..8 {
                if w.t_latent(rng, p, h) {
                    break;
                }
            }
        }
        // random rules
        let mut guard = 0usize;
        while w.rules.len() < p.rules && guard < p.rules * 24 {
            guard += 1;
            match rng.weighted(&[3.0, 3.0, 1.5, 2.0, 1.5, 2.0, 1.0, 1.0]) {
                0 => w.t_chain(rng, p),
                1 => w.t_conj(rng, p),
                2 => w.t_prop(rng, p),
                3 => w.t_fan(rng, p),
                4 => w.t_split(rng, p),
                5 => w.t_consume(rng, p),
                6 => w.t_invent(rng, p),
                _ => w.t_func(rng, p),
            };
        }
        // coverage: every derived predicate is produced by at least one rule
        for q in 0..w.preds.len() {
            if w.preds[q].layer > 0 && !w.produced(q) {
                for _ in 0..8 {
                    if w.t_into(rng, p, q) {
                        break;
                    }
                }
            }
        }
        w
    }

    fn add_pred(&mut self, rng: &mut Rng, layer: usize, arity: usize, latent: bool) -> usize {
        let name = self.lex.word(rng, 1, 3);
        let sym = self.g.sym(&name, Kind::Pred, arity);
        let prior = match (layer, arity, latent) {
            (0, 1, false) => rng.uniform(0.22, 0.40),
            (0, 1, true) => rng.uniform(0.10, 0.22),
            (0, _, false) => rng.uniform(0.05, 0.14),
            (0, _, true) => rng.uniform(0.03, 0.08),
            _ => 0.0,
        };
        let idx = self.preds.len();
        self.preds.push(PredInfo { sym, arity, layer, latent, prior: round(prior, 3) });
        self.by_sym.insert(sym, idx);
        idx
    }

    fn vars(&mut self, n: usize) -> Vec<NodeId> {
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(self.g.var(i));
        }
        v
    }

    fn atom(&mut self, p: usize, args: &[NodeId]) -> NodeId {
        let s = self.preds[p].sym;
        self.g.mk(s, args)
    }

    /// Predicate indices passing a filter, in index order.
    fn select(&self, f: impl Fn(&PredInfo) -> bool) -> Vec<usize> {
        (0..self.preds.len()).filter(|&i| f(&self.preds[i])).collect()
    }

    fn produced(&self, q: usize) -> bool {
        let s = self.preds[q].sym;
        self.rules
            .iter()
            .any(|r| r.alts.iter().any(|a| a.1.iter().any(|&f| self.g.head(f) == s)))
    }

    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        rng: &mut Rng,
        kind: RuleKind,
        lhs: Vec<NodeId>,
        take: usize,
        alts: Vec<(f64, Vec<NodeId>)>,
        nvars: usize,
        skolem: Vec<SymId>,
    ) -> bool {
        let dup = self.rules.iter().any(|r| {
            r.lhs == lhs
                && r.take == take
                && r.alts.len() == alts.len()
                && r.alts.iter().zip(alts.iter()).all(|(x, y)| x.1 == y.1)
        });
        if dup {
            return false;
        }
        let rate = round(rng.uniform(0.6, 2.5), 2);
        self.rules.push(Rule { lhs, take, alts, rate, nvars, skolem });
        self.kinds.push(kind);
        true
    }

    // p(x..) => q(x..)
    fn t_chain(&mut self, rng: &mut Rng, pr: &Params) -> bool {
        let p = rng.below(self.preds.len());
        let (pl, pa) = (self.preds[p].layer, self.preds[p].arity);
        let dst = self.select(|q| q.layer > pl && q.arity <= pa && !q.latent);
        if dst.is_empty() {
            return false;
        }
        let q = dst[rng.below(dst.len())];
        let vs = self.vars(pa);
        let qargs = pick_args(rng, &vs, self.preds[q].arity);
        let lhs = self.atom(p, &vs);
        let rhs = self.atom(q, &qargs);
        let take: usize = if rng.chance(pr.consume) { 1 } else { 0 };
        let kind = if take > 0 { RuleKind::Consume } else { RuleKind::Chain };
        let alts = outcomes(rng, pr.leak, vec![rhs]);
        self.add(rng, kind, vec![lhs], take, alts, pa, Vec::new())
    }

    // p(x,y) & r(y,z) => q(x,z)   (the join variable is the last arg of p, first of r)
    fn t_conj(&mut self, rng: &mut Rng, pr: &Params) -> bool {
        let p = rng.below(self.preds.len());
        let r = rng.below(self.preds.len());
        let (pa, ra) = (self.preds[p].arity, self.preds[r].arity);
        if p == r && pa == 1 {
            return false;
        }
        let top = self.preds[p].layer.max(self.preds[r].layer);
        let nv = pa + ra - 1;
        let dst = self.select(|q| q.layer > top && q.arity <= nv && !q.latent);
        if dst.is_empty() {
            return false;
        }
        let q = dst[rng.below(dst.len())];
        let vs = self.vars(nv);
        let mut rargs: Vec<NodeId> = vec![vs[pa - 1]];
        rargs.extend_from_slice(&vs[pa..]);
        let qargs = pick_args(rng, &vs, self.preds[q].arity);
        let l1 = self.atom(p, &vs[..pa]);
        let l2 = self.atom(r, &rargs);
        let rhs = self.atom(q, &qargs);
        let take: usize = if rng.chance(pr.consume) { 1 } else { 0 };
        let kind = if take > 0 { RuleKind::Consume } else { RuleKind::Conj };
        let alts = outcomes(rng, pr.leak, vec![rhs]);
        self.add(rng, kind, vec![l1, l2], take, alts, nv, Vec::new())
    }

    // e(x,y) & p(x) => p(y)
    fn t_prop(&mut self, rng: &mut Rng, pr: &Params) -> bool {
        let uni = self.select(|q| q.arity == 1 && q.layer >= 1);
        if uni.is_empty() {
            return false;
        }
        let p = uni[rng.below(uni.len())];
        let pl = self.preds[p].layer;
        let rel = self.select(|q| q.arity == 2 && q.layer <= pl);
        if rel.is_empty() {
            return false;
        }
        let e = rel[rng.below(rel.len())];
        let vs = self.vars(2);
        let (a, b) = if rng.chance(0.5) { (vs[0], vs[1]) } else { (vs[1], vs[0]) };
        let l1 = self.atom(e, &[a, b]);
        let l2 = self.atom(p, &[vs[0]]);
        let rhs = self.atom(p, &[vs[1]]);
        let alts = outcomes(rng, pr.leak, vec![rhs]);
        self.add(rng, RuleKind::Prop, vec![l1, l2], 0, alts, 2, Vec::new())
    }

    // p(x) => q1(x), q2(x) [, q3(x)]
    fn t_fan(&mut self, rng: &mut Rng, pr: &Params) -> bool {
        let p = rng.below(self.preds.len());
        let (pl, pa) = (self.preds[p].layer, self.preds[p].arity);
        let mut dst = self.select(|q| q.layer > pl && q.arity <= pa && !q.latent);
        if dst.len() < 2 {
            return false;
        }
        rng.shuffle(&mut dst);
        let k = rng.range(2, dst.len().min(3));
        dst.truncate(k);
        let vs = self.vars(pa);
        let lhs = self.atom(p, &vs);
        let mut fx: Vec<NodeId> = Vec::new();
        for &q in dst.iter() {
            let a = pick_args(rng, &vs, self.preds[q].arity);
            fx.push(self.atom(q, &a));
        }
        let alts = outcomes(rng, pr.leak, fx);
        self.add(rng, RuleKind::Fan, vec![lhs], 0, alts, pa, Vec::new())
    }

    // p(x) => { 0.6: q1(x) | 0.4: q2(x) }
    fn t_split(&mut self, rng: &mut Rng, _pr: &Params) -> bool {
        let p = rng.below(self.preds.len());
        let (pl, pa) = (self.preds[p].layer, self.preds[p].arity);
        let mut dst = self.select(|q| q.layer > pl && q.arity <= pa && !q.latent);
        if dst.len() < 2 {
            return false;
        }
        rng.shuffle(&mut dst);
        let k = rng.range(2, dst.len().min(3));
        dst.truncate(k);
        let vs = self.vars(pa);
        let lhs = self.atom(p, &vs);
        let pv = probs(rng, k);
        let mut alts: Vec<(f64, Vec<NodeId>)> = Vec::new();
        for (i, &q) in dst.iter().enumerate() {
            let a = pick_args(rng, &vs, self.preds[q].arity);
            let f = self.atom(q, &a);
            alts.push((pv[i], vec![f]));
        }
        self.add(rng, RuleKind::Split, vec![lhs], 0, alts, pa, Vec::new())
    }

    // -p(x..) & c(x) => q(x..)   (p is eaten, the catalyst c stays)
    fn t_consume(&mut self, rng: &mut Rng, pr: &Params) -> bool {
        let p = rng.below(self.preds.len());
        let (pl, pa) = (self.preds[p].layer, self.preds[p].arity);
        let cs = self.select(|q| q.arity == 1);
        if cs.is_empty() {
            return false;
        }
        let c = cs[rng.below(cs.len())];
        if c == p {
            return false;
        }
        let top = pl.max(self.preds[c].layer);
        let dst = self.select(|q| q.layer > top && q.arity <= pa && !q.latent);
        if dst.is_empty() {
            return false;
        }
        let q = dst[rng.below(dst.len())];
        let vs = self.vars(pa);
        let k = rng.below(pa);
        let l1 = self.atom(p, &vs);
        let l2 = self.atom(c, &[vs[k]]);
        let qargs = pick_args(rng, &vs, self.preds[q].arity);
        let rhs = self.atom(q, &qargs);
        let alts = outcomes(rng, pr.leak, vec![rhs]);
        self.add(rng, RuleKind::Consume, vec![l1, l2], 1, alts, pa, Vec::new())
    }

    // p(x) => q(x, ?z), r(?z)   with ?z = sk(x): a freshly invented entity
    fn t_invent(&mut self, rng: &mut Rng, pr: &Params) -> bool {
        let p = rng.below(self.preds.len());
        let (pl, pa) = (self.preds[p].layer, self.preds[p].arity);
        let two = self.select(|q| q.layer > pl && q.arity == 2 && !q.latent);
        let one = self.select(|q| q.layer > pl && q.arity == 1 && !q.latent);
        if two.is_empty() {
            return false;
        }
        let q = two[rng.below(two.len())];
        let vs = self.vars(pa);
        let z = self.g.var(pa);
        let src = vs[rng.below(pa)];
        let mut fx: Vec<NodeId> = vec![self.atom(q, &[src, z])];
        if !one.is_empty() && rng.chance(0.7) {
            let r = one[rng.below(one.len())];
            fx.push(self.atom(r, &[z]));
        }
        let name = self.lex.word(rng, 2, 3);
        let sk = self.g.sym(&name, Kind::Func, pa);
        let lhs = self.atom(p, &vs);
        let take: usize = if rng.chance(pr.consume) { 1 } else { 0 };
        let alts = outcomes(rng, pr.leak, fx);
        self.add(rng, RuleKind::Invent, vec![lhs], take, alts, pa, vec![sk])
    }

    // p(x) => q(f(x))   and, often, the destructor   q(f(x)) => r(x)
    fn t_func(&mut self, rng: &mut Rng, pr: &Params) -> bool {
        if self.funcs.is_empty() {
            return false;
        }
        let f = self.funcs[rng.below(self.funcs.len())];
        let uni = self.select(|q| q.arity == 1);
        if uni.is_empty() {
            return false;
        }
        let p = uni[rng.below(uni.len())];
        let pl = self.preds[p].layer;
        let mid = self.select(|q| q.arity == 1 && q.layer > pl && !q.latent);
        if mid.is_empty() {
            return false;
        }
        let q = mid[rng.below(mid.len())];
        let ql = self.preds[q].layer;
        let x = self.g.var(0);
        let fx = self.g.mk(f, &[x]);
        let l1 = self.atom(p, &[x]);
        let r1 = self.atom(q, &[fx]);
        let alts = outcomes(rng, pr.leak, vec![r1]);
        let ok = self.add(rng, RuleKind::Func, vec![l1], 0, alts, 1, Vec::new());
        let outs = self.select(|r| r.arity == 1 && r.layer > ql && !r.latent);
        if !outs.is_empty() && rng.chance(0.7) {
            let r = outs[rng.below(outs.len())];
            let l2 = self.atom(q, &[fx]);
            let r2 = self.atom(r, &[x]);
            let alts2 = outcomes(rng, pr.leak, vec![r2]);
            let ok2 = self.add(rng, RuleKind::Func, vec![l2], 0, alts2, 1, Vec::new());
            return ok || ok2;
        }
        ok
    }

    // H(x..) => e1(x..), e2(x..)   with H a hidden predicate: the common cause
    fn t_latent(&mut self, rng: &mut Rng, pr: &Params, h: usize) -> bool {
        let ha = self.preds[h].arity;
        let mut dst = self.select(|q| q.layer > 0 && q.arity <= ha && !q.latent);
        if dst.len() < 2 {
            return false;
        }
        rng.shuffle(&mut dst);
        let k = rng.range(2, dst.len().min(3));
        dst.truncate(k);
        let vs = self.vars(ha);
        let lhs = self.atom(h, &vs);
        let mut fx: Vec<NodeId> = Vec::new();
        for &q in dst.iter() {
            let a = pick_args(rng, &vs, self.preds[q].arity);
            fx.push(self.atom(q, &a));
        }
        // a hidden cause may be imperfect: sometimes only its first effect shows
        let alts = if fx.len() > 1 && rng.chance(0.4) {
            let a = round(rng.uniform(0.6, 0.9), 2);
            vec![(a, fx.clone()), (round(1.0 - a, 2), vec![fx[0]])]
        } else {
            outcomes(rng, pr.leak, fx)
        };
        let take: usize = if rng.chance(pr.consume) { 1 } else { 0 };
        self.add(rng, RuleKind::Latent, vec![lhs], take, alts, ha, Vec::new())
    }

    // coverage rule: some lower-layer predicate feeds q
    fn t_into(&mut self, rng: &mut Rng, pr: &Params, q: usize) -> bool {
        let (ql, qa) = (self.preds[q].layer, self.preds[q].arity);
        let srcs = self.select(|s| s.layer < ql && s.arity >= qa);
        if srcs.is_empty() {
            return false;
        }
        let p = srcs[rng.below(srcs.len())];
        let pa = self.preds[p].arity;
        let vs = self.vars(pa);
        let qargs = pick_args(rng, &vs, qa);
        let lhs = self.atom(p, &vs);
        let rhs = self.atom(q, &qargs);
        let alts = outcomes(rng, pr.leak, vec![rhs]);
        self.add(rng, RuleKind::Chain, vec![lhs], 0, alts, pa, Vec::new())
    }

    // ------------------------------------------------------------- queries

    pub fn pred_of(&self, f: NodeId) -> Option<usize> {
        self.by_sym.get(&self.g.head(f)).copied()
    }

    pub fn pred_name(&self, i: usize) -> &str {
        self.g.name(self.preds[i].sym)
    }

    pub fn is_latent(&self, f: NodeId) -> bool {
        self.pred_of(f).map_or(false, |i| self.preds[i].latent)
    }

    /// ln(prior / (1 - prior)) of a ground atom being initially present;
    /// very negative for anything that can never be an initial fact.
    pub fn logit_prior(&self, f: NodeId) -> f64 {
        match self.pred_of(f) {
            Some(p) if self.preds[p].layer == 0 => {
                let x = self.preds[p].prior.clamp(1e-6, 1.0 - 1e-6);
                (x / (1.0 - x)).ln()
            }
            _ => -20.0,
        }
    }

    /// Random initial state over layer-0 predicates (each ground atom present
    /// with its predicate's prior); every hidden cause occurs at least once.
    pub fn sample_init(&mut self, rng: &mut Rng) -> State {
        let mut st = State::default();
        let n = self.consts.len();
        for pi in 0..self.preds.len() {
            let (layer, arity, latent, prior) = {
                let d = &self.preds[pi];
                (d.layer, d.arity, d.latent, d.prior)
            };
            if layer != 0 {
                continue;
            }
            let before = st.facts.len();
            if arity == 1 {
                for i in 0..n {
                    if rng.chance(prior) {
                        let c = self.consts[i];
                        let f = self.atom(pi, &[c]);
                        st.facts.insert(f);
                    }
                }
            } else {
                for i in 0..n {
                    for j in 0..n {
                        if i != j && rng.chance(prior) {
                            let (a, b) = (self.consts[i], self.consts[j]);
                            let f = self.atom(pi, &[a, b]);
                            st.facts.insert(f);
                        }
                    }
                }
            }
            if latent && st.facts.len() == before {
                let a = self.consts[rng.below(n)];
                let mut args = vec![a];
                if arity == 2 {
                    let mut b = self.consts[rng.below(n)];
                    while b == a {
                        b = self.consts[rng.below(n)];
                    }
                    args.push(b);
                }
                let f = self.atom(pi, &args);
                st.facts.insert(f);
            }
        }
        st
    }

    /// Structural sanity of every rule (variables bound, probabilities, Skolem arity).
    pub fn validate(&self) -> Result<(), String> {
        for (i, r) in self.rules.iter().enumerate() {
            if r.lhs.is_empty() || r.take > r.lhs.len() {
                return Err(format!("rule {i}: bad lhs/take"));
            }
            if r.alts.is_empty() {
                return Err(format!("rule {i}: no outcomes"));
            }
            let total: f64 = r.alts.iter().map(|a| a.0).sum();
            if (total - 1.0).abs() > 1e-6 {
                return Err(format!("rule {i}: probabilities sum to {total}"));
            }
            if !(r.rate > 0.0) {
                return Err(format!("rule {i}: non-positive rate"));
            }
            let mut bound: Vec<usize> = Vec::new();
            for &p in r.lhs.iter() {
                collect_vars(&self.g, p, &mut bound);
            }
            for &v in bound.iter() {
                if v >= r.nvars {
                    return Err(format!("rule {i}: lhs uses ?{v} beyond nvars"));
                }
            }
            for v in 0..r.nvars {
                if !bound.contains(&v) {
                    return Err(format!("rule {i}: ?{v} does not occur in the lhs"));
                }
            }
            let mut used: Vec<usize> = Vec::new();
            for a in r.alts.iter() {
                for &f in a.1.iter() {
                    collect_vars(&self.g, f, &mut used);
                }
            }
            for &v in used.iter() {
                if v >= r.nvars + r.skolem.len() {
                    return Err(format!("rule {i}: rhs uses undeclared ?{v}"));
                }
            }
            for &s in r.skolem.iter() {
                if self.g.info(s).arity != r.nvars {
                    return Err(format!("rule {i}: skolem arity differs from nvars"));
                }
            }
        }
        Ok(())
    }

    /// One-line rule text: `-` marks a consumed premise, `@` the rate.
    pub fn rule_text(&self, i: usize) -> String {
        let r = &self.rules[i];
        let mut s = String::new();
        for (k, &pat) in r.lhs.iter().enumerate() {
            if k > 0 {
                s.push_str(" & ");
            }
            if k < r.take {
                s.push('-');
            }
            s.push_str(&self.g.show(pat));
        }
        s.push_str(" => ");
        for (a, alt) in r.alts.iter().enumerate() {
            if a > 0 {
                s.push_str(" | ");
            }
            if r.alts.len() > 1 {
                s.push_str(&format!("{}: ", alt.0));
            }
            if alt.1.is_empty() {
                s.push_str("nothing");
            }
            for (k, &f) in alt.1.iter().enumerate() {
                if k > 0 {
                    s.push_str(", ");
                }
                s.push_str(&self.g.show(f));
            }
        }
        if r.skolem.len() == 1 {
            s.push_str(&format!("  [new ?{}]", r.nvars));
        } else if r.skolem.len() > 1 {
            s.push_str(&format!("  [new ?{}..?{}]", r.nvars, r.nvars + r.skolem.len() - 1));
        }
        s.push_str(&format!("  @{}", r.rate));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_rules_are_well_formed() {
        for seed in 0..30u64 {
            for level in 1..=6u32 {
                let mut rng = Rng::new(seed);
                let w = World::generate(&mut rng, &Params::level(level));
                w.validate().unwrap();
                assert!(!w.rules.is_empty());
            }
        }
    }

    #[test]
    fn init_contains_every_hidden_cause() {
        let mut rng = Rng::new(3);
        let mut w = World::generate(&mut rng, &Params::level(3));
        let st = w.sample_init(&mut rng);
        for (i, p) in w.preds.iter().enumerate() {
            if p.latent {
                assert!(st.facts.iter().any(|&f| w.pred_of(f) == Some(i)));
            }
        }
    }
}