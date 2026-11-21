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