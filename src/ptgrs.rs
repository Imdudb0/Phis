use crate::percept::{random_renaming, Percept};
use crate::util::*;
use std::collections::HashMap;
pub const NONE: u32 = u32::MAX;
const NIL: usize = usize::MAX;
const IND: u16 = u16::MAX;
const CL_FMT: u32 = 1;
const CL_VAR: u32 = 2;
const CL_LHS: u32 = 3;
const CL_RHS: u32 = 4;
const TAG_SCHEMA: u64 = 0x5343_4845_4D41_0008;
const VARN: [char; 6] = ['x', 'y', 'z', 'w', 'u', 'v'];
const CONN: [char; 8] = ['a', 'b', 'c', 'd', 'e', 'i', 'j', 'l'];
const FUNN: [char; 8] = ['h', 'f', 'g', 'p', 'q', 'r', 's', 't'];
#[inline]
fn ent(class: u32, a: usize, b: usize) -> u32 {
    (class << 24) | (((a as u32) & 0xFFF) << 12) | ((b as u32) & 0xFFF)
}
fn kind_of(e: u32) -> Kind {
    match e >> 24 {
        CL_FMT => Kind::Format,
        CL_VAR => Kind::Var,
        _ => Kind::Rule,
    }
}
fn tok(s: u16) -> Cell {
    Cell { sym: s, uid: NONE, ent: ent(CL_FMT, s as usize, 0) }
}
fn rr(rng: &mut Rng, r: (usize, usize)) -> usize {
    rng.range(r.0.min(r.1), r.0.max(r.1))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Share,
    Rule,
    Var,
    Format,
    Schema,
}
impl Kind {
    pub const ALL: [Kind; 5] = [Kind::Share, Kind::Rule, Kind::Var, Kind::Format, Kind::Schema];
    pub fn weight(self) -> f64 {
        if self == Kind::Format { 0.25 } else { 1.0 }
    }
    pub fn name(self) -> &'static str {
        match self {
            Kind::Share => "share",
            Kind::Rule => "rule",
            Kind::Var => "var",
            Kind::Format => "format",
            Kind::Schema => "schema",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Link {
    pub to: usize,
    pub from: usize,
    pub kind: Kind,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub sym: u16,
    pub uid: u32,
    pub ent: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    Rule,
    Step,
    Eval,
}
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub kind: BlockKind,
    pub rule: usize,
    pub step: usize,
    pub start: usize,
    pub end: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spec {
    Off,
    Head,
    Inline,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Style {
    pub spec: Spec,
    pub steps: bool,
    pub eval: bool,
}
impl Style {
    pub const QA: Style = Style { spec: Spec::Head, steps: false, eval: true };
    pub const CLOSED: Style = Style { spec: Spec::Head, steps: true, eval: true };
    pub const WORKED: Style = Style { spec: Spec::Inline, steps: true, eval: false };
    pub const TRACE: Style = Style { spec: Spec::Off, steps: true, eval: false };
    pub const ALL: [Style; 4] = [Style::QA, Style::CLOSED, Style::WORKED, Style::TRACE];
    pub fn norm(mut self) -> Style {
        if self.spec == Spec::Inline && !self.steps {
            self.spec = Spec::Head;
        }
        if self.spec == Spec::Off && !self.steps && !self.eval {
            self.eval = true;
        }
        self
    }
}
#[derive(Clone, Debug)]
pub struct Cfg {
    pub funs: (usize, usize),
    pub cons: (usize, usize),
    pub vars: usize,
    pub arity: usize,
    pub rules: (usize, usize),
    pub nest: f64,
    pub lit: f64,
    pub rhs: usize,
    pub leaf: f64,
    pub qdepth: usize,
    pub share: f64,
    pub steps: (usize, usize),
    pub min_steps: usize,
    pub rec_max: usize,
    pub term_max: usize,
    pub style: Option<Style>,
    pub align: usize,
    pub tries: usize,
}
impl Default for Cfg {
    fn default() -> Cfg {
        Cfg {
            funs: (3, 3),
            cons: (2, 3),
            vars: 4,
            arity: 2,
            rules: (2, 3),
            nest: 0.2,
            lit: 0.1,
            rhs: 5,
            leaf: 0.35,
            qdepth: 2,
            share: 0.15,
            steps: (1, 3),
            min_steps: 1,
            rec_max: 8,
            term_max: 9,
            style: None,
            align: 0,
            tries: 24,
        }
    }
}
impl Cfg {
    pub fn level(k: usize) -> Cfg {
        let mut c = Cfg::default();
        match k {
            0 => {
                c.funs = (2, 2);
                c.cons = (2, 2);
                c.vars = 2;
                c.rules = (1, 1);
                c.nest = 0.0;
                c.lit = 0.0;
                c.rhs = 3;
                c.qdepth = 1;
                c.share = 0.0;
                c.steps = (1, 1);
                c.rec_max = 6;
                c.term_max = 6;
                c.style = Some(Style::QA);
            }
            1 => {
                c.funs = (2, 3);
                c.cons = (2, 3);
                c.vars = 3;
                c.rules = (1, 2);
                c.nest = 0.0;
                c.lit = 0.0;
                c.rhs = 4;
                c.qdepth = 1;
                c.share = 0.1;
                c.steps = (1, 2);
                c.rec_max = 7;
                c.term_max = 7;
            }
            2 => {
                c.rules = (2, 2);
                c.nest = 0.15;
                c.lit = 0.05;
            }
            3 => {}
            _ => {
                c.funs = (3, 4);
                c.cons = (3, 4);
                c.rules = (3, 4);
                c.rhs = 6;
                c.steps = (2, 4);
                c.nest = 0.3;
                c.share = 0.2;
                c.rec_max = 10;
                c.term_max = 12;
            }
        }
        c
    }
    pub fn fit(mut self, np: usize, aligned: bool) -> Cfg {
        let h = (np.saturating_sub(2) / 2).max(3);
        self.rec_max = self.rec_max.min(h);
        self.term_max = self.term_max.min(h).max(4);
        self.rhs = self.rhs.min(h);
        self.align = if aligned { np } else { 0 };
        self
    }
}
#[derive(Clone, Debug)]
pub struct Sig {
    pub arity: Vec<usize>,
    pub name: Vec<char>,
    pub arrow: u16,
    pub sep: u16,
    pub pad: u16,
    pub vars: Vec<u16>,
    pub cons: Vec<u16>,
    pub funs: Vec<u16>,
}
impl Sig {
    pub fn n(&self) -> usize {
        self.arity.len()
    }
    fn add(&mut self, a: usize, c: char) -> u16 {
        self.arity.push(a);
        self.name.push(c);
        (self.arity.len() - 1) as u16
    }
}
fn sample_sig(rng: &mut Rng, cfg: &Cfg) -> Sig {
    let nf = rr(rng, cfg.funs).clamp(2, FUNN.len());
    let nc = rr(rng, cfg.cons).clamp(1, CONN.len());
    let nv = cfg.vars.clamp(1, VARN.len());
    let am = cfg.arity.clamp(1, 3);
    let mut s = Sig {
        arity: vec![0; 3],
        name: vec!['>', ';', '_'],
        arrow: 0,
        sep: 1,
        pad: 2,
        vars: Vec::new(),
        cons: Vec::new(),
        funs: Vec::new(),
    };
    for i in 0..nv {
        let id = s.add(0, VARN[i]);
        s.vars.push(id);
    }
    for i in 0..nc {
        let id = s.add(0, CONN[i]);
        s.cons.push(id);
    }
    for i in 0..nf {
        let a = if am >= 2 && rng.chance(0.65) { 2 } else { rng.range(1, am) };
        let id = s.add(a, FUNN[i]);
        s.funs.push(id);
    }
    s
}
#[derive(Clone, Debug, PartialEq)]
pub enum Pat {
    Var(usize),
    App(u16, Vec<Pat>),
}
#[derive(Clone, Debug)]
pub struct Rule {
    pub lhs: Pat,
    pub rhs: Pat,
    pub w: f64,
    pub nv: usize,
}
impl Rule {
    pub fn head(&self) -> u16 {
        match &self.lhs {
            Pat::App(s, _) => *s,
            Pat::Var(_) => IND,
        }
    }
}
fn has_var(p: &Pat) -> bool {
    match p {
        Pat::Var(_) => true,
        Pat::App(_, ks) => ks.iter().any(has_var),
    }
}
fn lhs_arg(rng: &mut Rng, sig: &Sig, cfg: &Cfg, lvl: usize, next: &mut usize) -> Pat {
    let nv = sig.vars.len();
    if lvl >= 1 && rng.chance(cfg.nest) {
        let h = sig.funs[rng.below(lvl)];
        let ah = sig.arity[h as usize];
        if *next + ah <= nv {
            let mut vs = Vec::with_capacity(ah);
            for _ in 0..ah {
                vs.push(Pat::Var(*next));
                *next += 1;
            }
            return Pat::App(h, vs);
        }
    }
    if *next >= nv || rng.chance(cfg.lit) {
        return Pat::App(sig.cons[rng.below(sig.cons.len())], Vec::new());
    }
    *next += 1;
    Pat::Var(*next - 1)
}
fn rhs_app(rng: &mut Rng, sig: &Sig, cfg: &Cfg, lvl: usize, nv: usize, left: &mut usize) -> Pat {
    let f = sig.funs[rng.below(lvl)];
    *left = (*left).saturating_sub(1);
    let mut ks = Vec::with_capacity(sig.arity[f as usize]);
    for _ in 0..sig.arity[f as usize] {
        ks.push(rhs_node(rng, sig, cfg, lvl, nv, left));
    }
    Pat::App(f, ks)
}
fn rhs_node(rng: &mut Rng, sig: &Sig, cfg: &Cfg, lvl: usize, nv: usize, left: &mut usize) -> Pat {
    if *left > 1 && !rng.chance(cfg.leaf) {
        return rhs_app(rng, sig, cfg, lvl, nv, left);
    }
    *left = (*left).saturating_sub(1);
    if nv > 0 && rng.chance(0.7) {
        Pat::Var(rng.below(nv))
    } else {
        Pat::App(sig.cons[rng.below(sig.cons.len())], Vec::new())
    }
}
fn make_rule(rng: &mut Rng, sig: &Sig, cfg: &Cfg, lvl: usize) -> Rule {
    let head = sig.funs[lvl];
    let mut next = 0usize;
    let mut kids = Vec::new();
    for _ in 0..sig.arity[head as usize] {
        kids.push(lhs_arg(rng, sig, cfg, lvl, &mut next));
    }
    let lhs = Pat::App(head, kids);
    let nv = next;
    let mut rhs = Pat::App(sig.cons[0], Vec::new());
    for _ in 0..16 {
        let mut left = cfg.rhs.max(1);
        rhs = if left >= 2 && rng.chance(0.85) {
            rhs_app(rng, sig, cfg, lvl, nv, &mut left)
        } else {
            rhs_node(rng, sig, cfg, lvl, nv, &mut left)
        };
        if nv == 0 || has_var(&rhs) {
            break;
        }
    }
    Rule { lhs, rhs, w: 0.5 + rng.f64(), nv }
}
fn sample_rules(rng: &mut Rng, sig: &Sig, cfg: &Cfg) -> Vec<Rule> {
    let nr = rr(rng, cfg.rules).max(1);
    let mut heads: Vec<usize> = (1..sig.funs.len()).collect();
    rng.shuffle(&mut heads);
    let mut out = Vec::with_capacity(nr);
    for i in 0..nr {
        let lvl = if i < heads.len() { heads[i] } else { heads[rng.below(heads.len())] };
        out.push(make_rule(rng, sig, cfg, lvl));
    }
    out
}
// ===================== 5a-2/3 =====================
// Terme en préfixe sans parenthèses (arités dans Sig) : 1 nœud = 1 cellule = 1 octet de E.
// Cell.uid = identité du nœud du graphe (partage, persistance d'un pas à l'autre).
// Cell.ent = rôle dans la cause (format, variable de règle, côté gauche/droit d'une règle).
// Un pas s'écrit `avant > après ;` avec le terme entier de chaque côté.
// Cfg.rec_max = longueur max (en pas) d'une dérivation menée à la forme normale.
// Cfg.term_max = taille max (cellules, partage déplié) de tout terme écrit.
// Cfg.align = np : aucun bloc ne chevauche deux trames, bourrage `_` jusqu'à la frontière.

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub sym: u16,
    pub kids: [usize; 3],
    pub uid: u32,
}
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
}
#[derive(Clone, Debug)]
pub struct Redex {
    pub node: usize,
    pub rule: usize,
    pub bind: Vec<usize>,
}
#[derive(Clone, Copy, Debug)]
pub struct Hot<'a> {
    pub node: usize,
    pub rule: usize,
    pub side: u32,
    pub pat: &'a Pat,
}
impl Graph {
    pub fn add(&mut self, sym: u16, kids: &[usize]) -> usize {
        let mut k = [NIL; 3];
        for (j, &x) in kids.iter().take(3).enumerate() {
            k[j] = x;
        }
        let id = self.nodes.len();
        self.nodes.push(Node { sym, kids: k, uid: id as u32 });
        id
    }
    pub fn deref(&self, mut i: usize) -> usize {
        while self.nodes[i].sym == IND {
            i = self.nodes[i].kids[0];
        }
        i
    }
    fn chain(&self, mut i: usize, via: usize) -> (usize, bool) {
        let mut hit = false;
        loop {
            if i == via {
                hit = true;
            }
            if self.nodes[i].sym != IND {
                return (i, hit);
            }
            i = self.nodes[i].kids[0];
        }
    }
    /// Taille de l'arbre déplié, saturée à cap + 1.
    pub fn size(&self, root: usize, cap: usize) -> usize {
        let mut n = 0usize;
        let mut st: Vec<usize> = vec![root];
        while let Some(i) = st.pop() {
            n += 1;
            if n > cap {
                return n;
            }
            let d = self.deref(i);
            for k in 0..3 {
                let c = self.nodes[d].kids[k];
                if c != NIL {
                    st.push(c);
                }
            }
        }
        n
    }
    pub fn matches(&self, p: &Pat, i: usize, bind: &mut [usize]) -> bool {
        let d = self.deref(i);
        match p {
            Pat::Var(v) => {
                if bind[*v] == NIL {
                    bind[*v] = d;
                    true
                } else {
                    bind[*v] == d
                }
            }
            Pat::App(s, ks) => {
                let n = self.nodes[d];
                if n.sym != *s {
                    return false;
                }
                for (k, q) in ks.iter().enumerate() {
                    if k >= 3 || n.kids[k] == NIL || !self.matches(q, n.kids[k], bind) {
                        return false;
                    }
                }
                true
            }
        }
    }
    pub fn redexes(&self, rules: &[Rule], root: usize) -> Vec<Redex> {
        let mut out: Vec<Redex> = Vec::new();
        let mut seen = vec![false; self.nodes.len()];
        let mut st: Vec<usize> = vec![root];
        while let Some(i) = st.pop() {
            let d = self.deref(i);
            if seen[d] {
                continue;
            }
            seen[d] = true;
            for (r, rule) in rules.iter().enumerate() {
                let mut bind = vec![NIL; rule.nv];
                if self.matches(&rule.lhs, d, &mut bind) {
                    out.push(Redex { node: d, rule: r, bind });
                }
            }
            let n = self.nodes[d];
            for k in (0..3).rev() {
                if n.kids[k] != NIL {
                    st.push(n.kids[k]);
                }
            }
        }
        out
    }
    pub fn build(&mut self, p: &Pat, bind: &[usize]) -> usize {
        match p {
            Pat::Var(v) => bind[*v],
            Pat::App(s, ks) => {
                let n = ks.len().min(3);
                let mut kids = [NIL; 3];
                for k in 0..n {
                    kids[k] = self.build(&ks[k], bind);
                }
                self.add(*s, &kids[..n])
            }
        }
    }
    /// Construit le membre droit, puis le nœud du radical devient une indirection vers le résultat.
    pub fn apply(&mut self, rules: &[Rule], rd: &Redex) -> usize {
        let res = self.build(&rules[rd.rule].rhs, &rd.bind);
        if res != rd.node {
            let n = &mut self.nodes[rd.node];
            n.sym = IND;
            n.kids = [res, NIL, NIL];
        }
        res
    }
    /// Écrit le terme en préfixe. Avec `hot`, la région du radical est parcourue avec son motif et étiquetée.
    pub fn emit(&self, i: usize, hot: Option<&Hot<'_>>, out: &mut Vec<Cell>) {
        let via = match hot {
            Some(h) => h.node,
            None => NIL,
        };
        let (d, hit) = self.chain(i, via);
        if hit {
            if let Some(h) = hot {
                let mut pos = 0usize;
                self.walk(d, h, h.pat, &mut pos, out);
                return;
            }
        }
        let n = self.nodes[d];
        out.push(Cell { sym: n.sym, uid: n.uid, ent: NONE });
        for k in 0..3 {
            if n.kids[k] != NIL {
                self.emit(n.kids[k], hot, out);
            }
        }
    }
    fn walk(&self, i: usize, h: &Hot<'_>, p: &Pat, pos: &mut usize, out: &mut Vec<Cell>) {
        let at = *pos;
        *pos += 1;
        match p {
            Pat::Var(v) => {
                let s = out.len();
                self.emit(i, None, out);
                out[s].ent = ent(CL_VAR, h.rule, *v);
            }
            Pat::App(_, ks) => {
                let d = self.deref(i);
                let n = self.nodes[d];
                out.push(Cell { sym: n.sym, uid: n.uid, ent: ent(h.side, h.rule, at) });
                for (k, q) in ks.iter().enumerate() {
                    if k < 3 && n.kids[k] != NIL {
                        self.walk(n.kids[k], h, q, pos, out);
                    }
                }
            }
        }
    }
    /// Terme avant le pas : le radical est écrit comme instance du membre gauche de la règle.
    pub fn emit_lhs(&self, root: usize, rules: &[Rule], rd: &Redex, out: &mut Vec<Cell>) {
        let hot = Hot { node: rd.node, rule: rd.rule, side: CL_LHS, pat: &rules[rd.rule].lhs };
        self.emit(root, Some(&hot), out);
    }
    /// Terme après le pas : le résultat est écrit comme instance du membre droit.
    pub fn emit_rhs(&self, root: usize, rules: &[Rule], rd: &Redex, out: &mut Vec<Cell>) {
        let hot = Hot { node: rd.node, rule: rd.rule, side: CL_RHS, pat: &rules[rd.rule].rhs };
        self.emit(root, Some(&hot), out);
    }
}

#[derive(Clone, Debug)]
pub struct Step {
    pub rule: usize,
    pub before: Vec<Cell>,
    pub after: Vec<Cell>,
}
#[derive(Clone, Debug)]
pub struct Sim {
    pub steps: Vec<Step>,
    pub first: Vec<Cell>,
    pub last: Vec<Cell>,
    pub normal: bool,
}

struct QGen<'a> {
    sig: &'a Sig,
    rules: &'a [Rule],
    cfg: &'a Cfg,
    pool: Vec<usize>,
}
impl<'a> QGen<'a> {
    fn leaf(&mut self, rng: &mut Rng, g: &mut Graph) -> usize {
        let c = self.sig.cons[rng.below(self.sig.cons.len())];
        let id = g.add(c, &[]);
        self.pool.push(id);
        id
    }
    fn node(&mut self, rng: &mut Rng, g: &mut Graph, depth: usize, top: bool) -> usize {
        if !top && !self.pool.is_empty() && rng.chance(self.cfg.share) {
            return self.pool[rng.below(self.pool.len())];
        }
        let p_inst = if top { 0.8 } else { 0.4 };
        if depth > 0 && !self.rules.is_empty() && rng.chance(p_inst) {
            let rules = self.rules;
            let lhs = &rules[rng.below(rules.len())].lhs;
            return self.inst(rng, g, lhs, depth);
        }
        if depth == 0 || (!top && rng.chance(0.2)) {
            return self.leaf(rng, g);
        }
        let f = self.sig.funs[rng.below(self.sig.funs.len())];
        let ar = self.sig.arity[f as usize].min(3);
        let mut kids = [NIL; 3];
        for k in 0..ar {
            kids[k] = self.node(rng, g, depth - 1, false);
        }
        let id = g.add(f, &kids[..ar]);
        self.pool.push(id);
        id
    }
    fn inst(&mut self, rng: &mut Rng, g: &mut Graph, p: &Pat, depth: usize) -> usize {
        match p {
            Pat::Var(_) => self.node(rng, g, depth.saturating_sub(1), false),
            Pat::App(s, ks) => {
                let n = ks.len().min(3);
                let mut kids = [NIL; 3];
                for k in 0..n {
                    kids[k] = self.inst(rng, g, &ks[k], depth);
                }
                let id = g.add(*s, &kids[..n]);
                self.pool.push(id);
                id
            }
        }
    }
}

/// Question initiale : instance (probable) d'un membre gauche, sous-termes partagés avec la probabilité cfg.share.
pub fn sample_query(rng: &mut Rng, sig: &Sig, rules: &[Rule], cfg: &Cfg, g: &mut Graph) -> usize {
    let mut q = QGen { sig, rules, cfg, pool: Vec::new() };
    q.node(rng, g, cfg.qdepth.max(1), true)
}

/// Réécriture probabiliste : à chaque pas, un radical tiré avec le poids Rule.w parmi tous ceux du graphe.
/// Un pas qui dépasse term_max est écarté ; `None` si la question elle-même dépasse term_max.
pub fn simulate(rng: &mut Rng, g: &mut Graph, rules: &[Rule], root: usize, cfg: &Cfg, limit: usize) -> Option<Sim> {
    let cap = cfg.term_max;
    if g.size(root, cap) > cap {
        return None;
    }
    let mut first: Vec<Cell> = Vec::new();
    g.emit(root, None, &mut first);
    let mut steps: Vec<Step> = Vec::new();
    for _ in 0..limit {
        let mut cand = g.redexes(rules, root);
        let mut pick: Option<(Graph, Redex)> = None;
        while !cand.is_empty() {
            let w: Vec<f64> = cand.iter().map(|r| rules[r.rule].w).collect();
            let rd = cand.swap_remove(rng.weighted(&w));
            let mut g2 = g.clone();
            g2.apply(rules, &rd);
            if g2.size(root, cap) <= cap {
                pick = Some((g2, rd));
                break;
            }
        }
        let (g2, rd) = match pick {
            Some(x) => x,
            None => break,
        };
        let mut before: Vec<Cell> = Vec::new();
        g.emit_lhs(root, rules, &rd, &mut before);
        let mut after: Vec<Cell> = Vec::new();
        g2.emit_rhs(root, rules, &rd, &mut after);
        steps.push(Step { rule: rd.rule, before, after });
        *g = g2;
    }
    let mut last: Vec<Cell> = Vec::new();
    g.emit(root, None, &mut last);
    let normal = g.redexes(rules, root).is_empty();
    Some(Sim { steps, first, last, normal })
}

fn emit_pat(sig: &Sig, p: &Pat, rule: usize, side: u32, pos: &mut usize, out: &mut Vec<Cell>) {
    let at = *pos;
    *pos += 1;
    match p {
        Pat::Var(v) => out.push(Cell { sym: sig.vars[*v], uid: NONE, ent: ent(CL_VAR, rule, *v) }),
        Pat::App(s, ks) => {
            out.push(Cell { sym: *s, uid: NONE, ent: ent(side, rule, at) });
            for q in ks {
                emit_pat(sig, q, rule, side, pos, out);
            }
        }
    }
}

struct Asm<'a> {
    sig: &'a Sig,
    cells: Vec<Cell>,
    blocks: Vec<Block>,
    align: usize,
}
impl<'a> Asm<'a> {
    fn pad_to_frame(&mut self) {
        if self.align > 0 {
            while self.cells.len() % self.align != 0 {
                self.cells.push(tok(self.sig.pad));
            }
        }
    }
    fn put(&mut self, kind: BlockKind, rule: usize, step: usize, body: Vec<Cell>) {
        let used = if self.align > 0 { self.cells.len() % self.align } else { 0 };
        if used > 0 && used + body.len() > self.align {
            self.pad_to_frame();
        }
        let start = self.cells.len();
        self.cells.extend(body);
        self.blocks.push(Block { kind, rule, step, start, end: self.cells.len() });
    }
    fn rule(&mut self, rules: &[Rule], r: usize, step: usize) {
        let mut body: Vec<Cell> = Vec::new();
        let mut pos = 0usize;
        emit_pat(self.sig, &rules[r].lhs, r, CL_LHS, &mut pos, &mut body);
        body.push(tok(self.sig.arrow));
        let mut pos = 0usize;
        emit_pat(self.sig, &rules[r].rhs, r, CL_RHS, &mut pos, &mut body);
        body.push(tok(self.sig.sep));
        self.put(BlockKind::Rule, r, step, body);
    }
    fn line(&mut self, kind: BlockKind, rule: usize, step: usize, l: &[Cell], r: &[Cell]) {
        let mut body: Vec<Cell> = Vec::with_capacity(l.len() + r.len() + 2);
        body.extend_from_slice(l);
        body.push(tok(self.sig.arrow));
        body.extend_from_slice(r);
        body.push(tok(self.sig.sep));
        self.put(kind, rule, step, body);
    }
}

/// Texte abstrait : règles (Head, ou Inline avant chaque pas), pas écrits, puis éval `question > réponse ;`.
/// En style QA (sans pas écrits), l'éval porte les étiquettes du premier et du dernier pas de la dérivation cachée.
pub fn assemble(sig: &Sig, rules: &[Rule], style: Style, sim: &Sim, show: usize, align: usize) -> (Vec<Cell>, Vec<Block>) {
    let mut a = Asm { sig, cells: Vec::new(), blocks: Vec::new(), align };
    if style.spec == Spec::Head {
        for r in 0..rules.len() {
            a.rule(rules, r, NIL);
        }
    }
    let n = if style.steps { show.min(sim.steps.len()) } else { 0 };
    for (s, st) in sim.steps.iter().take(n).enumerate() {
        if style.spec == Spec::Inline {
            a.rule(rules, st.rule, s);
        }
        a.line(BlockKind::Step, st.rule, s, &st.before, &st.after);
    }
    if style.eval {
        match (sim.steps.first(), sim.steps.last()) {
            (Some(f), Some(l)) if !style.steps => a.line(BlockKind::Eval, NIL, NIL, &f.before, &l.after),
            _ => a.line(BlockKind::Eval, NIL, NIL, &sim.first, &sim.last),
        }
    }
    a.pad_to_frame();
    (a.cells, a.blocks)
}

/// Cause cachée sous forme de liens : chaque occurrence d'une même entité (ent) ou d'un même nœud (uid)
/// est reliée à la première (from < to), plus, par bloc, racine du membre gauche -> racine du résultat.
pub fn make_links(cells: &[Cell], blocks: &[Block]) -> Vec<Link> {
    let mut by_ent: HashMap<u32, usize> = HashMap::new();
    let mut by_uid: HashMap<u32, usize> = HashMap::new();
    let mut out: Vec<Link> = Vec::new();
    for (p, c) in cells.iter().enumerate() {
        if c.ent != NONE {
            let first = by_ent.get(&c.ent).copied();
            match first {
                Some(f) => out.push(Link { to: p, from: f, kind: kind_of(c.ent) }),
                None => {
                    by_ent.insert(c.ent, p);
                }
            }
        }
        if c.uid != NONE {
            let first = by_uid.get(&c.uid).copied();
            match first {
                Some(f) => out.push(Link { to: p, from: f, kind: Kind::Share }),
                None => {
                    by_uid.insert(c.uid, p);
                }
            }
        }
    }
    for b in blocks {
        let end = b.end.min(cells.len());
        let arrow = match (b.start..end).find(|&p| (cells[p].ent >> 24) == CL_FMT) {
            Some(a) => a,
            None => continue,
        };
        let from = match b.kind {
            BlockKind::Step => (b.start..arrow).find(|&p| cells[p].ent == ent(CL_LHS, b.rule, 0)),
            _ => Some(b.start),
        };
        if let Some(f) = from {
            let to = arrow + 1 + (f - b.start);
            if to + 1 < end {
                out.push(Link { to, from: f, kind: Kind::Rule });
            }
        }
    }
    out.sort_by_key(|l| (l.to, l.from, l.kind as usize));
    out.dedup_by(|x, y| x.to == y.to && x.from == y.from && x.kind == y.kind);
    out
}

#[derive(Clone, Debug)]
pub struct Abs {
    pub sig: Sig,
    pub rules: Vec<Rule>,
    pub style: Style,
    pub cells: Vec<Cell>,
    pub blocks: Vec<Block>,
    pub links: Vec<Link>,
    pub steps: usize,
    pub deriv: usize,
    pub normal: bool,
    pub align: usize,
}

fn fits(blocks: &[Block], align: usize) -> bool {
    align == 0 || blocks.iter().all(|b| b.end - b.start <= align)
}

fn finish(
    rng: &mut Rng,
    cfg: &Cfg,
    sig: Sig,
    rules: Vec<Rule>,
    style: Style,
    mut g: Graph,
    root: usize,
    strict: bool,
) -> Option<Abs> {
    let n_want = rr(rng, cfg.steps).max(cfg.min_steps).max(1);
    let limit = if style.steps { n_want } else { cfg.rec_max.max(1) };
    let sim = simulate(rng, &mut g, &rules, root, cfg, limit)?;
    let l = sim.steps.len();
    if l < cfg.min_steps.max(1) || (style.eval && !sim.normal) {
        return None;
    }
    let show = if style.steps { l } else { 0 };
    let (cells, blocks) = assemble(&sig, &rules, style, &sim, show, cfg.align);
    if strict && !fits(&blocks, cfg.align) {
        return None;
    }
    let links = make_links(&cells, &blocks);
    Some(Abs { sig, rules, style, cells, blocks, links, steps: show, deriv: l, normal: sim.normal, align: cfg.align })
}

/// Un tirage : schéma (flux forké par TAG_SCHEMA), style, question, dérivation, texte, liens. `None` si rejeté.
pub fn sample_once(rng: &mut Rng, cfg: &Cfg) -> Option<Abs> {
    let mut sr = rng.fork(TAG_SCHEMA);
    let sig = sample_sig(&mut sr, cfg);
    let rules = sample_rules(&mut sr, &sig, cfg);
    let mut style = match cfg.style {
        Some(s) => s,
        None => *rng.pick(&Style::ALL),
    }
    .norm();
    if !style.steps && !style.eval {
        style.eval = true;
    }
    let mut g = Graph::default();
    let root = sample_query(rng, &sig, &rules, cfg, &mut g);
    finish(rng, cfg, sig, rules, style, g, root, true)
}

fn fallback(rng: &mut Rng, cfg: &Cfg) -> Abs {
    let mut c = cfg.clone();
    c.term_max = c.term_max.max(4);
    c.rec_max = c.rec_max.max(1);
    c.min_steps = 1;
    c.steps = (1, c.steps.1.max(c.steps.0).max(1));
    let mut sr = rng.fork(TAG_SCHEMA);
    let sig = sample_sig(&mut sr, &c);
    let top = sig.funs[1];
    let ar = sig.arity[top as usize].clamp(1, 3);
    let nv = ar.min(sig.vars.len());
    let kp: Vec<Pat> = (0..ar)
        .map(|i| if i < nv { Pat::Var(i) } else { Pat::App(sig.cons[0], Vec::new()) })
        .collect();
    let rules = vec![Rule { lhs: Pat::App(top, kp), rhs: Pat::Var(0), w: 1.0, nv }];
    let mut g = Graph::default();
    let leaf = g.add(sig.cons[0], &[]);
    let root = g.add(top, &vec![leaf; ar]);
    let mut style = c.style.unwrap_or(Style::TRACE).norm();
    if !style.steps && !style.eval {
        style.eval = true;
    }
    finish(rng, &c, sig, rules, style, g, root, false).expect("ptgrs: fallback instance")
}

/// Rejet jusqu'à cfg.tries fois, puis instance minimale garantie.
pub fn sample(rng: &mut Rng, cfg: &Cfg) -> Abs {
    for _ in 0..cfg.tries.max(1) {
        if let Some(a) = sample_once(rng, cfg) {
            return a;
        }
    }
    fallback(rng, cfg)
}

fn prefix_ok(sig: &Sig, cells: &[Cell], end: usize, i: &mut usize) -> bool {
    if *i >= end {
        return false;
    }
    let s = cells[*i].sym as usize;
    if s < 3 || s >= sig.n() {
        return false;
    }
    *i += 1;
    for _ in 0..sig.arity[s] {
        if !prefix_ok(sig, cells, end, i) {
            return false;
        }
    }
    true
}

/// Contrôle structurel d'une instance : symboles, blocs (`terme > terme ;`), trames, liens.
pub fn verify(a: &Abs) -> Result<(), String> {
    let n = a.cells.len();
    if n == 0 {
        return Err("empty text".to_string());
    }
    for (p, c) in a.cells.iter().enumerate() {
        if (c.sym as usize) >= a.sig.n() {
            return Err(format!("cell {}: unknown symbol {}", p, c.sym));
        }
    }
    if a.align > 0 && n % a.align != 0 {
        return Err(format!("length {} is not a multiple of {}", n, a.align));
    }
    let mut prev = 0usize;
    for (k, b) in a.blocks.iter().enumerate() {
        if b.start < prev || b.end <= b.start || b.end > n {
            return Err(format!("block {}: bad range {}..{}", k, b.start, b.end));
        }
        if a.align > 0 {
            let len = b.end - b.start;
            if len <= a.align && b.start / a.align != (b.end - 1) / a.align {
                return Err(format!("block {} straddles two frames", k));
            }
            if len > a.align && b.start % a.align != 0 {
                return Err(format!("long block {} does not start on a frame", k));
            }
        }
        let mut i = b.start;
        let mut ok = prefix_ok(&a.sig, &a.cells, b.end, &mut i);
        if ok && i < b.end && a.cells[i].sym == a.sig.arrow {
            i += 1;
            ok = prefix_ok(&a.sig, &a.cells, b.end, &mut i)
                && i + 1 == b.end
                && a.cells[i].sym == a.sig.sep;
        } else {
            ok = false;
        }
        if !ok {
            return Err(format!("block {}: not `term > term ;`", k));
        }
        prev = b.end;
    }
    for l in &a.links {
        if l.from >= l.to || l.to >= n {
            return Err(format!("link {:?}: bad positions", l));
        }
        let (x, y) = (&a.cells[l.to], &a.cells[l.from]);
        let ok = match l.kind {
            Kind::Share => x.uid != NONE && x.uid == y.uid && x.sym == y.sym,
            Kind::Format => x.ent == y.ent && x.sym == y.sym,
            Kind::Var => x.ent == y.ent && (x.ent >> 24) == CL_VAR,
            Kind::Rule | Kind::Schema => true,
        };
        if !ok {
            return Err(format!("link {:?}: inconsistent cells", l));
        }
    }
    Ok(())
}

pub fn show_cells(sig: &Sig, cells: &[Cell]) -> String {
    cells
        .iter()
        .map(|c| sig.name.get(c.sym as usize).copied().unwrap_or('?'))
        .collect()
}

pub fn show_abs(a: &Abs) -> String {
    let mut s = String::new();
    for b in &a.blocks {
        let kind = match b.kind {
            BlockKind::Rule => "rule",
            BlockKind::Step => "step",
            BlockKind::Eval => "eval",
        };
        let end = b.end.min(a.cells.len());
        s.push_str(&format!("{:>4} {:<4} {}\n", b.start, kind, show_cells(&a.sig, &a.cells[b.start..end])));
    }
    s
}