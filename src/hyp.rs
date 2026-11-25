//! hyp : hypothèse H (grammaire à classes), coups, forêt d'analyse, rejeu, compilation en PTGRS.
//! H n'utilise que les symboles canoniques de E. Une production = nœuds partagés (slots) + corps ;
//! `S(k)` répété = même nœud instancié une fois (accord), exactement la sémantique de `Builder::rule`.
use crate::canon::Canon;
use crate::ptgrs::{OpId, Ptgrs, RNode, Rule, Sym};
use crate::util::{fnv64, Meter};
use std::collections::HashMap;

pub type Cls = u32;
pub const MAX_CLASSES: usize = u16::MAX as usize;
pub const NONE: u32 = u32::MAX;
pub const FIT_ROUNDS: u32 = 3;
const MAX_LEN: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Item {
    A(Sym),
    C(Cls),
    S(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Prod {
    pub slots: Vec<Cls>,
    pub body: Vec<Item>,
    pub w: f64,
}

impl Prod {
    pub fn same(&self, o: &Prod) -> bool {
        self.slots == o.slots && self.body == o.body
    }
    /// Forme canonique : slots numérotés par première occurrence, slots inutilisés supprimés.
    pub fn normalize(&mut self) {
        let mut map: Vec<u32> = vec![NONE; self.slots.len()];
        let mut slots: Vec<Cls> = Vec::new();
        for it in self.body.iter_mut() {
            if let Item::S(k) = *it {
                let k = k as usize;
                if map[k] == NONE {
                    map[k] = slots.len() as u32;
                    slots.push(self.slots[k]);
                }
                *it = Item::S(map[k]);
            }
        }
        self.slots = slots;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Class {
    pub prods: Vec<Prod>,
    pub alive: bool,
}

/// Hypothèse sur E. `classes[0]` est le start (jamais référencé). Les ids ne sont jamais réutilisés.
#[derive(Clone, Debug, PartialEq)]
pub struct Hyp {
    pub na: usize,
    pub e: Vec<Sym>,
    pub classes: Vec<Class>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Chunk,
    Merge,
    Share,
    Unchunk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Move {
    /// Extrait `body[at..at+len]` de la production en une classe (réutilise une classe identique
    /// existante). `all` : remplace toutes les occurrences équivalentes, sinon la seule désignée.
    Chunk { cls: Cls, prod: u32, at: u32, len: u32, all: bool },
    /// Unit deux classes (la plus petite survit) : généralisation.
    Merge { a: Cls, b: Cls },
    /// Ajoute une production où deux items de même classe deviennent un même nœud partagé.
    Share { cls: Cls, prod: u32, i: u32, j: u32 },
    /// Inverse de Chunk : inline d'une classe à production unique.
    Unchunk { cls: Cls },
}

impl Move {
    pub fn kind(&self) -> Kind {
        match self {
            Move::Chunk { .. } => Kind::Chunk,
            Move::Merge { .. } => Kind::Merge,
            Move::Share { .. } => Kind::Share,
            Move::Unchunk { .. } => Kind::Unchunk,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveError {
    Dead,
    Bounds,
    Start,
    Window,
    Locality,
    Kind,
    NoOp,
    Blocked,
    Cycle,
    Limit,
    Invalid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fail {
    Move(MoveError),
    NoParse,
    Budget,
    Cycle,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Sig {
    items: Vec<Item>,
    slots: Vec<Cls>,
}

fn dedupe(prods: &mut Vec<Prod>) {
    let mut out: Vec<Prod> = Vec::with_capacity(prods.len());
    for p in prods.drain(..) {
        if let Some(q) = out.iter_mut().find(|q| q.same(&p)) {
            q.w += p.w;
        } else {
            out.push(p);
        }
    }
    *prods = out;
}

/// Signature d'une fenêtre ; None si elle coupe un nœud partagé (slot non local).
fn window_sig(p: &Prod, at: usize, len: usize) -> Option<Sig> {
    let win = &p.body[at..at + len];
    let mut map: Vec<(u32, u32)> = Vec::new();
    let mut slots: Vec<Cls> = Vec::new();
    let mut items: Vec<Item> = Vec::with_capacity(len);
    for &it in win {
        match it {
            Item::S(k) => {
                let found = map.iter().find(|&&(o, _)| o == k).map(|&(_, n)| n);
                let nk = match found {
                    Some(n) => n,
                    None => {
                        let n = map.len() as u32;
                        map.push((k, n));
                        slots.push(p.slots[k as usize]);
                        n
                    }
                };
                items.push(Item::S(nk));
            }
            other => items.push(other),
        }
    }
    for &(o, _) in &map {
        let total = p.body.iter().filter(|&&x| x == Item::S(o)).count();
        let inside = win.iter().filter(|&&x| x == Item::S(o)).count();
        if total != inside {
            return None;
        }
    }
    Some(Sig { items, slots })
}

/// Remplace (gauche→droite, sans chevauchement) les fenêtres de signature `sig` par `C(n)`.
fn replace_in(p: &Prod, sig: &Sig, n: Cls, only: Option<usize>) -> Option<(Prod, usize)> {
    let len = sig.items.len();
    let mut body: Vec<Item> = Vec::with_capacity(p.body.len());
    let mut t = 0usize;
    let mut cnt = 0usize;
    while t < p.body.len() {
        let here = only.map_or(true, |a| a == t) && t + len <= p.body.len();
        if here {
            if let Some(s) = window_sig(p, t, len) {
                if s == *sig {
                    body.push(Item::C(n));
                    t += len;
                    cnt += 1;
                    continue;
                }
            }
        }
        body.push(p.body[t]);
        t += 1;
    }
    if cnt == 0 {
        return None;
    }
    let mut q = Prod { slots: p.slots.clone(), body, w: p.w };
    q.normalize();
    Some((q, cnt))
}

fn item_class(p: &Prod, it: Item) -> Option<Cls> {
    match it {
        Item::C(x) => Some(x),
        Item::S(k) => p.slots.get(k as usize).copied(),
        Item::A(_) => None,
    }
}

fn share_prod(p: &Prod, i: usize, j: usize) -> Result<Prod, MoveError> {
    let (i, j) = if i <= j { (i, j) } else { (j, i) };
    if j >= p.body.len() {
        return Err(MoveError::Bounds);
    }
    if i == j {
        return Err(MoveError::NoOp);
    }
    let ci = item_class(p, p.body[i]).ok_or(MoveError::Kind)?;
    let cj = item_class(p, p.body[j]).ok_or(MoveError::Kind)?;
    if ci != cj {
        return Err(MoveError::Kind);
    }
    let mut q = p.clone();
    match (p.body[i], p.body[j]) {
        (Item::C(_), Item::C(_)) => {
            let k = q.slots.len() as u32;
            q.slots.push(ci);
            q.body[i] = Item::S(k);
            q.body[j] = Item::S(k);
        }
        (Item::S(k), Item::C(_)) => q.body[j] = Item::S(k),
        (Item::C(_), Item::S(k)) => q.body[i] = Item::S(k),
        (Item::S(k1), Item::S(k2)) => {
            if k1 == k2 {
                return Err(MoveError::NoOp);
            }
            for it in q.body.iter_mut() {
                if *it == Item::S(k2) {
                    *it = Item::S(k1);
                }
            }
        }
        _ => return Err(MoveError::Kind),
    }
    q.normalize();
    Ok(q)
}

impl Hyp {
    /// H0 : la mémorisation littérale de E (régénère E avec probabilité 1).
    pub fn literal(e: &[Sym], na: usize) -> Hyp {
        let body: Vec<Item> = e.iter().map(|&a| Item::A(a)).collect();
        Hyp {
            na,
            e: e.to_vec(),
            classes: vec![Class { prods: vec![Prod { slots: Vec::new(), body, w: 1.0 }], alive: true }],
        }
    }

    pub fn from_canon(c: &Canon) -> Hyp {
        Hyp::literal(&c.syms, c.n())
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.classes.len();
        if n == 0 || !self.classes[0].alive {
            return Err("start absent".to_string());
        }
        if n > MAX_CLASSES {
            return Err("trop de classes".to_string());
        }
        for (c, cl) in self.classes.iter().enumerate() {
            if !cl.alive {
                if !cl.prods.is_empty() {
                    return Err(format!("classe {c}: morte mais non vide"));
                }
                continue;
            }
            if cl.prods.is_empty() {
                return Err(format!("classe {c}: aucune production"));
            }
            for (pi, p) in cl.prods.iter().enumerate() {
                if !(p.w.is_finite() && p.w > 0.0) {
                    return Err(format!("classe {c} prod {pi}: poids invalide"));
                }
                if p.body.is_empty() && c != 0 {
                    return Err(format!("classe {c} prod {pi}: corps vide"));
                }
                let mut used = vec![0usize; p.slots.len()];
                for &s in &p.slots {
                    let s = s as usize;
                    if s == 0 || s >= n || !self.classes[s].alive {
                        return Err(format!("classe {c} prod {pi}: slot invalide"));
                    }
                }
                for &it in &p.body {
                    match it {
                        Item::A(a) => {
                            if (a as usize) >= self.na {
                                return Err(format!("classe {c} prod {pi}: atome hors bornes"));
                            }
                        }
                        Item::C(x) => {
                            let x = x as usize;
                            if x == 0 || x >= n || !self.classes[x].alive {
                                return Err(format!("classe {c} prod {pi}: appel invalide"));
                            }
                        }
                        Item::S(k) => {
                            let k = k as usize;
                            if k >= used.len() {
                                return Err(format!("classe {c} prod {pi}: slot hors bornes"));
                            }
                            used[k] += 1;
                        }
                    }
                }
                if used.iter().any(|&u| u == 0) {
                    return Err(format!("classe {c} prod {pi}: slot inutilisé"));
                }
                let mut q = p.clone();
                q.normalize();
                if !q.same(p) {
                    return Err(format!("classe {c} prod {pi}: slots non canoniques"));
                }
                for (pj, r) in cl.prods.iter().enumerate().take(pi) {
                    if r.same(p) {
                        return Err(format!("classe {c}: productions {pj} et {pi} identiques"));
                    }
                }
            }
        }
        if self.unit_cycle() {
            return Err("cycle unitaire".to_string());
        }
        Ok(())
    }

    /// Cycle de productions unitaires (A→B→…→A) : le seul moyen d'avoir une forêt cyclique.
    fn unit_cycle(&self) -> bool {
        let n = self.classes.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut indeg = vec![0usize; n];
        for (c, cl) in self.classes.iter().enumerate() {
            if !cl.alive {
                continue;
            }
            for p in &cl.prods {
                if p.body.len() != 1 {
                    continue;
                }
                let d = match p.body[0] {
                    Item::C(x) => Some(x as usize),
                    Item::S(k) => p.slots.get(k as usize).map(|&x| x as usize),
                    Item::A(_) => None,
                };
                if let Some(d) = d {
                    if d < n {
                        adj[c].push(d);
                        indeg[d] += 1;
                    }
                }
            }
        }
        let mut stack: Vec<usize> = (0..n).filter(|&c| indeg[c] == 0).collect();
        let mut seen = 0usize;
        while let Some(c) = stack.pop() {
            seen += 1;
            for &d in &adj[c] {
                indeg[d] -= 1;
                if indeg[d] == 0 {
                    stack.push(d);
                }
            }
        }
        seen < n
    }

    /// Empreinte de structure (poids exclus), pour détecter les états revisités.
    pub fn fingerprint(&self) -> u64 {
        let mut b: Vec<u8> = Vec::new();
        for (c, cl) in self.classes.iter().enumerate() {
            if !cl.alive {
                continue;
            }
            b.extend_from_slice(&(c as u32).to_le_bytes());
            for p in &cl.prods {
                b.push(0xFE);
                for &s in &p.slots {
                    b.extend_from_slice(&s.to_le_bytes());
                }
                b.push(0xFD);
                for it in &p.body {
                    match *it {
                        Item::A(a) => {
                            b.push(0);
                            b.extend_from_slice(&a.to_le_bytes());
                        }
                        Item::C(x) => {
                            b.push(1);
                            b.extend_from_slice(&x.to_le_bytes());
                        }
                        Item::S(k) => {
                            b.push(2);
                            b.extend_from_slice(&k.to_le_bytes());
                        }
                    }
                }
            }
            b.push(0xFF);
        }
        fnv64(&b)
    }

    pub fn dump(&self) -> String {
        let mut s = String::new();
        for (c, cl) in self.classes.iter().enumerate() {
            if !cl.alive {
                continue;
            }
            for p in &cl.prods {
                let body: Vec<String> = p
                    .body
                    .iter()
                    .map(|it| match it {
                        Item::A(a) => format!("a{a}"),
                        Item::C(x) => format!("@{x}"),
                        Item::S(k) => format!("${k}"),
                    })
                    .collect();
                let slots: Vec<String> = p.slots.iter().map(|x| format!("@{x}")).collect();
                s.push_str(&format!("c{c} w={:.3} slots=[{}] : {}\n", p.w, slots.join(","), body.join(" ")));
            }
        }
        s
    }

    /// Applique un coup de façon transactionnelle : `self` est inchangé en cas d'erreur.
    pub fn apply(&mut self, m: &Move) -> Result<(), MoveError> {
        let mut g = self.clone();
        match *m {
            Move::Chunk { cls, prod, at, len, all } => g.chunk(cls, prod, at, len, all)?,
            Move::Merge { a, b } => g.merge(a, b)?,
            Move::Share { cls, prod, i, j } => g.share(cls, prod, i, j)?,
            Move::Unchunk { cls } => g.unchunk(cls)?,
        }
        if g.unit_cycle() {
            return Err(MoveError::Cycle);
        }
        if g.validate().is_err() {
            return Err(MoveError::Invalid);
        }
        *self = g;
        Ok(())
    }

    fn chunk(&mut self, cls: Cls, prod: u32, at: u32, len: u32, all: bool) -> Result<(), MoveError> {
        let c = cls as usize;
        let cl = self.classes.get(c).ok_or(MoveError::Bounds)?;
        if !cl.alive {
            return Err(MoveError::Dead);
        }
        let p = cl.prods.get(prod as usize).ok_or(MoveError::Bounds)?;
        let (at, len) = (at as usize, len as usize);
        if len == 0 || at + len > p.body.len() {
            return Err(MoveError::Window);
        }
        let sig = window_sig(p, at, len).ok_or(MoveError::Locality)?;
        let mut target: Option<usize> = None;
        for (x, k) in self.classes.iter().enumerate() {
            if x != 0 && k.alive && k.prods.len() == 1 && k.prods[0].slots == sig.slots && k.prods[0].body == sig.items {
                target = Some(x);
                break;
            }
        }
        let nid = match target {
            Some(x) => x,
            None => {
                if self.classes.len() >= MAX_CLASSES {
                    return Err(MoveError::Limit);
                }
                self.classes.len()
            }
        };
        let mut replaced = 0usize;
        for (ci, k) in self.classes.iter_mut().enumerate() {
            if !k.alive || Some(ci) == target {
                continue;
            }
            for (pi, q) in k.prods.iter_mut().enumerate() {
                let only = if all {
                    None
                } else if ci == c && pi == prod as usize {
                    Some(at)
                } else {
                    continue;
                };
                if let Some((r, cnt)) = replace_in(q, &sig, nid as Cls, only) {
                    q.body = r.body;
                    q.slots = r.slots;
                    replaced += cnt;
                }
            }
            dedupe(&mut k.prods);
        }
        if replaced == 0 {
            return Err(MoveError::NoOp);
        }
        if target.is_none() {
            self.classes.push(Class {
                prods: vec![Prod { slots: sig.slots, body: sig.items, w: replaced as f64 }],
                alive: true,
            });
        }
        Ok(())
    }

    fn merge(&mut self, a: Cls, b: Cls) -> Result<(), MoveError> {
        let (a, b) = (a.min(b) as usize, a.max(b) as usize);
        if a == b {
            return Err(MoveError::NoOp);
        }
        if b >= self.classes.len() {
            return Err(MoveError::Bounds);
        }
        if !self.classes[a].alive || !self.classes[b].alive {
            return Err(MoveError::Dead);
        }
        if a == 0 {
            return Err(MoveError::Start);
        }
        let moved = std::mem::take(&mut self.classes[b].prods);
        self.classes[b].alive = false;
        self.classes[a].prods.extend(moved);
        let (a32, b32) = (a as Cls, b as Cls);
        for k in self.classes.iter_mut() {
            if !k.alive {
                continue;
            }
            for p in k.prods.iter_mut() {
                for it in p.body.iter_mut() {
                    if *it == Item::C(b32) {
                        *it = Item::C(a32);
                    }
                }
                for s in p.slots.iter_mut() {
                    if *s == b32 {
                        *s = a32;
                    }
                }
            }
            dedupe(&mut k.prods);
        }
        Ok(())
    }

    fn share(&mut self, cls: Cls, prod: u32, i: u32, j: u32) -> Result<(), MoveError> {
        let c = cls as usize;
        let cl = self.classes.get(c).ok_or(MoveError::Bounds)?;
        if !cl.alive {
            return Err(MoveError::Dead);
        }
        let p = cl.prods.get(prod as usize).ok_or(MoveError::Bounds)?;
        let q = share_prod(p, i as usize, j as usize)?;
        if cl.prods.iter().any(|r| r.same(&q)) {
            return Err(MoveError::NoOp);
        }
        self.classes[c].prods.push(q);
        Ok(())
    }

    pub fn unchunk_ok(&self, x: Cls) -> Result<(), MoveError> {
        let xi = x as usize;
        let cl = self.classes.get(xi).ok_or(MoveError::Bounds)?;
        if !cl.alive {
            return Err(MoveError::Dead);
        }
        if xi == 0 {
            return Err(MoveError::Start);
        }
        if cl.prods.len() != 1 || cl.prods[0].body.contains(&Item::C(x)) {
            return Err(MoveError::Blocked);
        }
        for k in &self.classes {
            if k.alive && k.prods.iter().any(|p| p.slots.contains(&x)) {
                return Err(MoveError::Blocked);
            }
        }
        Ok(())
    }

    fn unchunk(&mut self, x: Cls) -> Result<(), MoveError> {
        self.unchunk_ok(x)?;
        let xi = x as usize;
        let def = self.classes[xi].prods[0].clone();
        for (ci, k) in self.classes.iter_mut().enumerate() {
            if !k.alive || ci == xi {
                continue;
            }
            for p in k.prods.iter_mut() {
                if !p.body.contains(&Item::C(x)) {
                    continue;
                }
                let mut body: Vec<Item> = Vec::with_capacity(p.body.len() + def.body.len());
                let mut slots = p.slots.clone();
                for &it in &p.body {
                    if it == Item::C(x) {
                        let base = slots.len() as u32;
                        slots.extend_from_slice(&def.slots);
                        for &jt in &def.body {
                            body.push(match jt {
                                Item::S(s) => Item::S(s + base),
                                other => other,
                            });
                        }
                    } else {
                        body.push(it);
                    }
                }
                p.body = body;
                p.slots = slots;
                p.normalize();
            }
            dedupe(&mut k.prods);
        }
        self.classes[xi].alive = false;
        self.classes[xi].prods.clear();
        Ok(())
    }
}

// ───────────── forêt d'analyse ─────────────

#[derive(Clone, Debug)]
pub struct Deriv {
    pub prod: u32,
    pub cuts: Vec<u32>, // fin de chaque item du corps
    pub kids: Vec<u32>, // nœuds des items parsés (C et première occurrence de chaque S), dans l'ordre
}

#[derive(Clone, Debug)]
pub struct FNode {
    pub cls: Cls,
    pub i: u32,
    pub j: u32,
    pub derivs: Vec<Deriv>,
}

/// Forêt empaquetée de E sous H : nœuds (classe, empan), enfants avant parents, racine en dernier.
#[derive(Clone, Debug)]
pub struct Forest {
    pub n: usize,
    pub nodes: Vec<FNode>,
    pub root: u32,
    pub best: Vec<u32>, // dérivation de Viterbi par nœud
    pub logp: f64,      // log P(E|H), somme sur toutes les dérivations
    pub vlogp: f64,     // log-proba de la meilleure dérivation
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TKind {
    Atom(Sym),
    Cls { cls: Cls, prod: u32 },
    Copy { of: u32 }, // 2e occurrence (ou +) d'un nœud partagé
}

#[derive(Clone, Debug)]
pub struct TNode {
    pub kind: TKind,
    pub i: u32,
    pub j: u32,
    pub parent: u32, // NONE pour la racine
    pub item: u32,   // indice de l'item dans le corps de la production du parent
    pub kids: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Tree {
    pub nodes: Vec<TNode>,
}

struct Chart {
    nodes: Vec<FNode>,
    ix: HashMap<u64, u32>,
    by_start: Vec<Vec<u32>>,
}

#[inline]
fn key(c: Cls, i: u32, j: u32) -> u64 {
    ((c as u64) << 42) | ((i as u64) << 21) | (j as u64)
}

fn logadd(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

struct Walk<'a> {
    s: &'a [Sym],
    ch: &'a Chart,
    meter: &'a mut Meter,
    cuts: Vec<u32>,
    kids: Vec<u32>,
    span: Vec<(u32, u32)>,
    out: Vec<(u32, Deriv)>,
}

impl<'a> Walk<'a> {
    /// Énumère toutes les façons de couvrir le corps de `p` depuis `pos` (item k).
    fn go(&mut self, p: &Prod, pidx: u32, mut k: usize, mut pos: usize) -> Result<(), Fail> {
        let c0 = self.cuts.len();
        let k0 = self.kids.len();
        let n = self.s.len();
        loop {
            self.meter.spend(1);
            if self.meter.exhausted() {
                return Err(Fail::Budget);
            }
            if k == p.body.len() {
                self.out.push((pos as u32, Deriv { prod: pidx, cuts: self.cuts.clone(), kids: self.kids.clone() }));
                break;
            }
            match p.body[k] {
                Item::A(a) => {
                    if pos < n && self.s[pos] == a {
                        pos += 1;
                        self.cuts.push(pos as u32);
                        k += 1;
                    } else {
                        break;
                    }
                }
                Item::C(d) => {
                    self.branch(p, pidx, k, pos, d, None)?;
                    break;
                }
                Item::S(sk) => {
                    let sk = sk as usize;
                    let (x, y) = self.span[sk];
                    if x == NONE {
                        let d = p.slots[sk];
                        self.branch(p, pidx, k, pos, d, Some(sk))?;
                        break;
                    }
                    let len = (y - x) as usize;
                    if pos + len <= n && self.s[x as usize..y as usize] == self.s[pos..pos + len] {
                        pos += len;
                        self.cuts.push(pos as u32);
                        k += 1;
                    } else {
                        break;
                    }
                }
            }
        }
        self.cuts.truncate(c0);
        self.kids.truncate(k0);
        Ok(())
    }

    fn branch(&mut self, p: &Prod, pidx: u32, k: usize, pos: usize, d: Cls, slot: Option<usize>) -> Result<(), Fail> {
        let ch = self.ch;
        for &nid in &ch.by_start[pos] {
            let nd = &ch.nodes[nid as usize];
            if nd.cls != d {
                continue;
            }
            self.cuts.push(nd.j);
            self.kids.push(nid);
            if let Some(sk) = slot {
                self.span[sk] = (nd.i, nd.j);
            }
            let r = self.go(p, pidx, k + 1, nd.j as usize);
            if let Some(sk) = slot {
                self.span[sk] = (NONE, NONE);
            }
            self.cuts.pop();
            self.kids.pop();
            r?;
        }
        Ok(())
    }
}

/// Garde les nœuds atteignables depuis la racine, en ordre topologique (enfants d'abord).
fn trim(nodes: Vec<FNode>, root: u32) -> Result<(Vec<FNode>, u32), Fail> {
    let nn = nodes.len();
    let mut state = vec![0u8; nn];
    let mut order: Vec<u32> = Vec::new();
    let mut stack: Vec<(u32, u32, u32)> = vec![(root, 0, 0)];
    state[root as usize] = 1;
    while let Some(&(v, di, ki)) = stack.last() {
        let top = stack.len() - 1;
        let nd = &nodes[v as usize];
        if (di as usize) >= nd.derivs.len() {
            state[v as usize] = 2;
            order.push(v);
            stack.pop();
            continue;
        }
        let d = &nd.derivs[di as usize];
        if (ki as usize) >= d.kids.len() {
            stack[top].1 = di + 1;
            stack[top].2 = 0;
            continue;
        }
        let c = d.kids[ki as usize];
        stack[top].2 = ki + 1;
        match state[c as usize] {
            0 => {
                state[c as usize] = 1;
                stack.push((c, 0, 0));
            }
            1 => return Err(Fail::Cycle),
            _ => {}
        }
    }
    let mut remap = vec![NONE; nn];
    for (new, &old) in order.iter().enumerate() {
        remap[old as usize] = new as u32;
    }
    let mut slots: Vec<Option<FNode>> = nodes.into_iter().map(Some).collect();
    let mut out: Vec<FNode> = Vec::with_capacity(order.len());
    for &old in &order {
        if let Some(mut nd) = slots[old as usize].take() {
            for d in nd.derivs.iter_mut() {
                for k in d.kids.iter_mut() {
                    *k = remap[*k as usize];
                }
            }
            out.push(nd);
        }
    }
    Ok((out, remap[root as usize]))
}

impl Forest {
    /// Inside (somme) et Viterbi (max) en log, avec les poids courants de `g`.
    pub fn score(&mut self, g: &Hyp) {
        let lt = g.log_theta();
        let nn = self.nodes.len();
        let mut lb = vec![f64::NEG_INFINITY; nn];
        let mut vb = vec![f64::NEG_INFINITY; nn];
        let mut best = vec![0u32; nn];
        for v in 0..nn {
            let nd = &self.nodes[v];
            let mut acc = f64::NEG_INFINITY;
            let mut mx = f64::NEG_INFINITY;
            let mut arg = 0u32;
            for (di, d) in nd.derivs.iter().enumerate() {
                let lp = lt[nd.cls as usize][d.prod as usize];
                let mut s = lp;
                let mut m = lp;
                for &k in &d.kids {
                    s += lb[k as usize];
                    m += vb[k as usize];
                }
                acc = logadd(acc, s);
                if m > mx {
                    mx = m;
                    arg = di as u32;
                }
            }
            lb[v] = acc;
            vb[v] = mx;
            best[v] = arg;
        }
        let r = self.root as usize;
        self.logp = lb[r];
        self.vlogp = vb[r];
        self.best = best;
    }

    fn best_counts(&self, g: &Hyp) -> Vec<Vec<f64>> {
        let mut cnt: Vec<Vec<f64>> = g.classes.iter().map(|c| vec![0.0; c.prods.len()]).collect();
        let mut stack = vec![self.root];
        while let Some(v) = stack.pop() {
            let nd = &self.nodes[v as usize];
            let d = &nd.derivs[self.best[v as usize] as usize];
            cnt[nd.cls as usize][d.prod as usize] += 1.0;
            for &k in &d.kids {
                stack.push(k);
            }
        }
        cnt
    }

    /// Retire les dérivations dont la production a disparu (newp[c][p] == NONE) et re-élague.
    fn retain_prods(&mut self, newp: &[Vec<u32>]) -> Result<(), Fail> {
        let nn = self.nodes.len();
        let mut alive = vec![false; nn];
        for v in 0..nn {
            let cls = self.nodes[v].cls as usize;
            let mut ds = std::mem::take(&mut self.nodes[v].derivs);
            ds.retain_mut(|d| {
                let np = newp[cls][d.prod as usize];
                if np == NONE || d.kids.iter().any(|&k| !alive[k as usize]) {
                    return false;
                }
                d.prod = np;
                true
            });
            alive[v] = !ds.is_empty();
            self.nodes[v].derivs = ds;
        }
        if !alive[self.root as usize] {
            return Err(Fail::NoParse);
        }
        let nodes = std::mem::take(&mut self.nodes);
        let (nodes, root) = trim(nodes, self.root)?;
        self.nodes = nodes;
        self.root = root;
        Ok(())
    }

    /// Arbre de la meilleure dérivation (ids parents < enfants ; racine = 0).
    pub fn tree(&self, g: &Hyp) -> Tree {
        let r = self.root as usize;
        let rd = &self.nodes[r].derivs[self.best[r] as usize];
        let mut t: Vec<TNode> = vec![TNode {
            kind: TKind::Cls { cls: self.nodes[r].cls, prod: rd.prod },
            i: self.nodes[r].i,
            j: self.nodes[r].j,
            parent: NONE,
            item: NONE,
            kids: Vec::new(),
        }];
        let mut stack: Vec<(u32, u32)> = vec![(self.root, 0)];
        while let Some((fv, tid)) = stack.pop() {
            let nd = &self.nodes[fv as usize];
            let d = &nd.derivs[self.best[fv as usize] as usize];
            let p = &g.classes[nd.cls as usize].prods[d.prod as usize];
            let mut first: Vec<u32> = vec![NONE; p.slots.len()];
            let mut ki = 0usize;
            let mut prev = nd.i;
            let mut kids: Vec<u32> = Vec::with_capacity(p.body.len());
            for (ti, it) in p.body.iter().enumerate() {
                let end = d.cuts[ti];
                let id = t.len() as u32;
                let mut child: Option<u32> = None;
                match *it {
                    Item::A(a) => t.push(TNode { kind: TKind::Atom(a), i: prev, j: end, parent: tid, item: ti as u32, kids: Vec::new() }),
                    Item::C(_) => {
                        child = Some(d.kids[ki]);
                        ki += 1;
                    }
                    Item::S(k) => {
                        let k = k as usize;
                        if first[k] == NONE {
                            first[k] = id;
                            child = Some(d.kids[ki]);
                            ki += 1;
                        } else {
                            t.push(TNode { kind: TKind::Copy { of: first[k] }, i: prev, j: end, parent: tid, item: ti as u32, kids: Vec::new() });
                        }
                    }
                }
                if let Some(kf) = child {
                    let kp = self.nodes[kf as usize].derivs[self.best[kf as usize] as usize].prod;
                    t.push(TNode {
                        kind: TKind::Cls { cls: self.nodes[kf as usize].cls, prod: kp },
                        i: prev,
                        j: end,
                        parent: tid,
                        item: ti as u32,
                        kids: Vec::new(),
                    });
                    stack.push((kf, id));
                }
                kids.push(id);
                prev = end;
            }
            t[tid as usize].kids = kids;
        }
        Tree { nodes: t }
    }
}

// ───────────── analyse, ajustement, pas ─────────────

impl Hyp {
    fn log_theta(&self) -> Vec<Vec<f64>> {
        self.classes
            .iter()
            .map(|cl| {
                let tot: f64 = cl.prods.iter().map(|p| p.w).sum();
                cl.prods
                    .iter()
                    .map(|p| if tot > 0.0 && p.w > 0.0 { (p.w / tot).ln() } else { f64::NEG_INFINITY })
                    .collect()
            })
            .collect()
    }

    /// Forêt d'analyse de `s` sous H (chart ascendant, départs de droite à gauche, point fixe par départ).
    /// `Err(NoParse)` : `s` n'est pas dans le langage de H ; `Err(Budget)` : budget épuisé.
    pub fn parse(&self, s: &[Sym], meter: &mut Meter) -> Result<Forest, Fail> {
        let n = s.len();
        if n >= MAX_LEN {
            return Err(Fail::Budget);
        }
        let mut ch = Chart { nodes: Vec::new(), ix: HashMap::new(), by_start: vec![Vec::new(); n + 1] };
        for i in (0..=n).rev() {
            let mut first = true;
            loop {
                let mut found: Vec<(Cls, u32, Deriv)> = Vec::new();
                {
                    let mut w = Walk {
                        s,
                        ch: &ch,
                        meter: &mut *meter,
                        cuts: Vec::new(),
                        kids: Vec::new(),
                        span: Vec::new(),
                        out: Vec::new(),
                    };
                    for (c, cl) in self.classes.iter().enumerate() {
                        if !cl.alive || (c == 0 && i != 0) {
                            continue;
                        }
                        for (pi, p) in cl.prods.iter().enumerate() {
                            if !first && matches!(p.body.first(), Some(Item::A(_)) | None) {
                                continue;
                            }
                            w.span.clear();
                            w.span.resize(p.slots.len(), (NONE, NONE));
                            w.out.clear();
                            w.go(p, pi as u32, 0, i)?;
                            for (end, d) in w.out.drain(..) {
                                found.push((c as Cls, end, d));
                            }
                        }
                    }
                }
                first = false;
                let mut changed = false;
                for (c, end, d) in found {
                    let kk = key(c, i as u32, end);
                    let id = match ch.ix.get(&kk).copied() {
                        Some(id) => id,
                        None => {
                            let id = ch.nodes.len() as u32;
                            ch.nodes.push(FNode { cls: c, i: i as u32, j: end, derivs: Vec::new() });
                            ch.ix.insert(kk, id);
                            ch.by_start[i].push(id);
                            changed = true;
                            id
                        }
                    };
                    let nd = &mut ch.nodes[id as usize];
                    if !nd.derivs.iter().any(|x| x.prod == d.prod && x.cuts == d.cuts) {
                        nd.derivs.push(d);
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        let root = match ch.ix.get(&key(0, 0, n as u32)).copied() {
            Some(r) => r,
            None => return Err(Fail::NoParse),
        };
        meter.spend(ch.nodes.len() as u64);
        let (nodes, root) = trim(ch.nodes, root)?;
        let mut f = Forest { n, nodes, root, best: Vec::new(), logp: f64::NEG_INFINITY, vlogp: f64::NEG_INFINITY };
        f.score(self);
        Ok(f)
    }

    pub fn analyze(&self, meter: &mut Meter) -> Result<Forest, Fail> {
        self.parse(&self.e, meter)
    }

    /// Poids = comptes de la dérivation de Viterbi ; productions jamais utilisées supprimées,
    /// classes devenues inaccessibles tuées. Retourne true si quelque chose a changé.
    fn refit(&mut self, f: &mut Forest) -> Result<bool, Fail> {
        let cnt = f.best_counts(self);
        let mut changed = false;
        let mut newp: Vec<Vec<u32>> = Vec::with_capacity(self.classes.len());
        for (c, cl) in self.classes.iter_mut().enumerate() {
            let mut map = vec![NONE; cl.prods.len()];
            if cl.alive {
                let old = std::mem::take(&mut cl.prods);
                for (pi, mut p) in old.into_iter().enumerate() {
                    let k = cnt[c][pi];
                    if k > 0.0 {
                        if p.w != k {
                            changed = true;
                        }
                        p.w = k;
                        map[pi] = cl.prods.len() as u32;
                        cl.prods.push(p);
                    } else {
                        changed = true;
                    }
                }
                if cl.prods.is_empty() {
                    cl.alive = false;
                }
            }
            newp.push(map);
        }
        f.retain_prods(&newp)?;
        f.score(self);
        Ok(changed)
    }

    /// Analyse puis ré-estimation (hard-EM de Viterbi) jusqu'au point fixe ou `rounds` tours.
    pub fn fit(&mut self, rounds: u32, meter: &mut Meter) -> Result<Forest, Fail> {
        if meter.exhausted() {
            return Err(Fail::Budget);
        }
        let mut f = self.analyze(meter)?;
        for _ in 0..rounds {
            meter.spend(f.nodes.len() as u64 + 1);
            if !self.refit(&mut f)? {
                break;
            }
        }
        Ok(f)
    }

    /// Transition canonique : coup + analyse + ré-estimation. Transactionnelle.
    pub fn step(&mut self, m: &Move, rounds: u32, meter: &mut Meter) -> Result<Forest, Fail> {
        if meter.exhausted() {
            return Err(Fail::Budget);
        }
        let mut next = self.clone();
        next.apply(m).map_err(Fail::Move)?;
        let f = next.fit(rounds, meter)?;
        *self = next;
        Ok(f)
    }
}