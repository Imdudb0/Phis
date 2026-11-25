//! PTGRS : Probabilistic Term Graph Rewriting System.
//!
//! Nœuds : `Atom` (symbole), `Cat` (concaténation), `Call` (non-terminal).
//! Un `Call` est réécrit EN PLACE par l'une des règles de son op, tirée selon les
//! poids. Un nœud référencé par plusieurs parents n'est donc réécrit qu'une fois :
//! les occurrences du nœud partagé donnent exactement la même chaîne (copie,
//! accord), alors que deux `Call` distincts d'un même op tirent indépendamment
//! (variation). E est le rendu (yield) du graphe en forme normale.
//!
//! La cause cachée C n'est jamais utilisée pour entraîner : `training_e` ne
//! renvoie que E. C ne sort que par `draw_instance`, réservé à l'évaluation.

use crate::canon::{Canon, ABSENT};
use crate::util::Rng;
use std::collections::BTreeMap;

pub type Sym = u16;
pub type OpId = u16;

/// Nœud d'un graphe de membre droit. Les enfants d'un `Cat` ont un indice
/// strictement inférieur à celui du `Cat` (ordre topologique : DAG garanti).
#[derive(Clone, Debug, PartialEq)]
pub enum RNode {
    Atom(Sym),
    Call(OpId),
    Cat(Vec<usize>),
}

#[derive(Clone, Debug)]
pub struct Rule {
    pub nodes: Vec<RNode>,
    pub root: usize,
    pub weight: f64,
    /// 0 = aucune. Sinon, l'étendue du rendu du nœud est consignée (évaluation).
    pub tag: u16,
}

impl Rule {
    pub fn has_call(&self) -> bool {
        self.nodes.iter().any(|n| matches!(n, RNode::Call(_)))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Ptgrs {
    pub n_atoms: usize,
    /// ops[o] = alternatives (règles) de l'op o.
    pub ops: Vec<Vec<Rule>>,
    pub start: OpId,
}

#[derive(Clone, Debug)]
pub struct Limits {
    pub max_steps: usize,
    pub max_len: usize,
    pub max_depth: usize,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits { max_steps: 20_000, max_len: 4096, max_depth: 64 }
    }
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub syms: Vec<Sym>,
    /// (tag, début, fin) dans l'ordre post-fixe.
    pub spans: Vec<(u16, usize, usize)>,
    pub steps: usize,
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Atom(Sym),
    Cat,
    Call(OpId),
}

struct Gn {
    kind: Kind,
    kids: Vec<u32>,
    tag: u16,
    done: bool,
}

impl Ptgrs {
    pub fn new(n_atoms: usize) -> Ptgrs {
        Ptgrs { n_atoms, ops: Vec::new(), start: 0 }
    }

    pub fn add_op(&mut self) -> OpId {
        self.ops.push(Vec::new());
        (self.ops.len() - 1) as OpId
    }

    pub fn add_rule(&mut self, op: OpId, rule: Rule) {
        self.ops[op as usize].push(rule);
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.ops.is_empty() {
            return Err("aucun op".to_string());
        }
        if (self.start as usize) >= self.ops.len() {
            return Err("start hors bornes".to_string());
        }
        for (o, alts) in self.ops.iter().enumerate() {
            for (r, rule) in alts.iter().enumerate() {
                if rule.root >= rule.nodes.len() {
                    return Err(format!("op {o} règle {r}: racine hors bornes"));
                }
                if !(rule.weight >= 0.0) {
                    return Err(format!("op {o} règle {r}: poids invalide"));
                }
                for (i, nd) in rule.nodes.iter().enumerate() {
                    match nd {
                        RNode::Atom(a) => {
                            if (*a as usize) >= self.n_atoms {
                                return Err(format!("op {o} règle {r}: atome hors bornes"));
                            }
                        }
                        RNode::Call(c) => {
                            if (*c as usize) >= self.ops.len() {
                                return Err(format!("op {o} règle {r}: appel hors bornes"));
                            }
                        }
                        RNode::Cat(ks) => {
                            for &k in ks {
                                if k >= i {
                                    return Err(format!("op {o} règle {r}: cycle ou ordre"));
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Renomme les atomes (map[ancien] = nouveau) ; `n_new` = nouvelle taille d'alphabet.
    pub fn relabel(&self, map: &[Sym], n_new: usize) -> Ptgrs {
        let mut g = self.clone();
        g.n_atoms = n_new;
        for alts in g.ops.iter_mut() {
            for r in alts.iter_mut() {
                for nd in r.nodes.iter_mut() {
                    if let RNode::Atom(a) = nd {
                        *a = map[*a as usize];
                    }
                }
            }
        }
        g
    }

    /// Un tirage de E par réécriture probabiliste de graphe. `None` si une limite saute.
    pub fn generate(&self, rng: &mut Rng, lim: &Limits) -> Option<Sample> {
        let mut g: Vec<Gn> = vec![Gn { kind: Kind::Call(self.start), kids: Vec::new(), tag: 0, done: false }];
        let mut steps = 0usize;
        if !self.reduce(&mut g, 0, rng, lim, &mut steps, 0) {
            return None;
        }
        let mut syms: Vec<Sym> = Vec::new();
        let mut spans: Vec<(u16, usize, usize)> = Vec::new();
        if !Self::flatten(&g, 0, &mut syms, &mut spans, lim.max_len) {
            return None;
        }
        Some(Sample { syms, spans, steps })
    }

    fn choose_rule(&self, op: usize, rng: &mut Rng, depth: usize, lim: &Limits) -> usize {
        let alts = &self.ops[op];
        if alts.len() == 1 {
            return 0;
        }
        let mut w: Vec<f64> = alts.iter().map(|r| r.weight.max(0.0)).collect();
        // Profondeur avancée : on privilégie les règles terminales pour borner la récursion.
        if depth * 2 >= lim.max_depth && alts.iter().any(|r| !r.has_call()) {
            for (i, r) in alts.iter().enumerate() {
                if r.has_call() {
                    w[i] = 0.0;
                }
            }
        }
        rng.weighted(&w)
    }

    fn reduce(&self, g: &mut Vec<Gn>, n: usize, rng: &mut Rng, lim: &Limits, steps: &mut usize, depth: usize) -> bool {
        if g[n].done {
            return true;
        }
        if depth > lim.max_depth {
            return false;
        }
        loop {
            let op = match g[n].kind {
                Kind::Call(op) => op as usize,
                _ => break,
            };
            if *steps >= lim.max_steps || self.ops[op].is_empty() {
                return false;
            }
            *steps += 1;
            let ri = self.choose_rule(op, rng, depth, lim);
            let rule = &self.ops[op][ri];
            let base = g.len();
            for nd in &rule.nodes {
                let (kind, kids) = match nd {
                    RNode::Atom(a) => (Kind::Atom(*a), Vec::new()),
                    RNode::Call(o) => (Kind::Call(*o), Vec::new()),
                    RNode::Cat(ix) => (Kind::Cat, ix.iter().map(|&i| (base + i) as u32).collect()),
                };
                g.push(Gn { kind, kids, tag: 0, done: false });
            }
            let r = base + rule.root;
            let (k, ks) = (g[r].kind, g[r].kids.clone());
            g[n].kind = k;
            g[n].kids = ks;
            if rule.tag != 0 {
                g[n].tag = rule.tag;
            }
        }
        g[n].done = true;
        let kids = g[n].kids.clone();
        for k in kids {
            if !self.reduce(g, k as usize, rng, lim, steps, depth + 1) {
                return false;
            }
        }
        true
    }

    fn flatten(g: &[Gn], n: usize, out: &mut Vec<Sym>, spans: &mut Vec<(u16, usize, usize)>, max_len: usize) -> bool {
        let start = out.len();
        match g[n].kind {
            Kind::Atom(a) => {
                if out.len() >= max_len {
                    return false;
                }
                out.push(a);
            }
            Kind::Cat => {
                for &k in &g[n].kids {
                    if !Self::flatten(g, k as usize, out, spans, max_len) {
                        return false;
                    }
                }
            }
            Kind::Call(_) => return false,
        }
        if g[n].tag != 0 {
            spans.push((g[n].tag, start, out.len()));
        }
        true
    }

    pub fn dump(&self) -> String {
        let mut s = String::new();
        for (o, alts) in self.ops.iter().enumerate() {
            for r in alts {
                let body: Vec<String> = r
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(i, n)| match n {
                        RNode::Atom(a) => format!("n{i}=a{a}"),
                        RNode::Call(c) => format!("n{i}=@{c}"),
                        RNode::Cat(k) => format!("n{i}=cat{k:?}"),
                    })
                    .collect();
                s.push_str(&format!("op{o} w={:.3} tag={} root=n{} : {}\n", r.weight, r.tag, r.root, body.join(" ")));
            }
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Construction de causes
// ---------------------------------------------------------------------------

/// Morceau d'une règle `Cat`. `S(k)` référence le k-ième nœud partagé de la règle
/// (un `Call` instancié une seule fois puis réutilisé : accord / copie).
#[derive(Clone, Copy, Debug)]
pub enum Part {
    A(Sym),
    C(OpId),
    S(usize),
}

pub struct Builder {
    pub g: Ptgrs,
}

impl Builder {
    /// L'op 0 est l'op de départ.
    pub fn new(n_atoms: usize) -> Builder {
        let mut g = Ptgrs::new(n_atoms);
        g.add_op();
        Builder { g }
    }

    pub fn op(&mut self) -> OpId {
        self.g.add_op()
    }

    pub fn word(rng: &mut Rng, n_atoms: usize, lo: usize, hi: usize) -> Vec<Sym> {
        let len = rng.range(lo, hi);
        (0..len).map(|_| rng.below(n_atoms) as Sym).collect()
    }

    pub fn atoms(w: &[Sym]) -> Vec<Part> {
        w.iter().map(|&a| Part::A(a)).collect()
    }

    /// Ajoute à `op` une règle : nœuds partagés `shared` puis `Cat(parts)`.
    pub fn rule(&mut self, op: OpId, weight: f64, shared: &[OpId], parts: &[Part]) {
        let mut nodes: Vec<RNode> = shared.iter().map(|&o| RNode::Call(o)).collect();
        let mut kids: Vec<usize> = Vec::new();
        for p in parts {
            match p {
                Part::S(k) => kids.push(*k),
                Part::A(a) => {
                    nodes.push(RNode::Atom(*a));
                    kids.push(nodes.len() - 1);
                }
                Part::C(o) => {
                    nodes.push(RNode::Call(*o));
                    kids.push(nodes.len() - 1);
                }
            }
        }
        nodes.push(RNode::Cat(kids));
        let root = nodes.len() - 1;
        self.g.add_rule(op, Rule { nodes, root, weight, tag: 0 });
    }

    /// Op déterministe : un mot (chunk).
    pub fn chunk(&mut self, w: &[Sym]) -> OpId {
        let op = self.op();
        self.rule(op, 1.0, &[], &Self::atoms(w));
        op
    }

    /// Op à alternatives : un mot par alternative.
    pub fn choice(&mut self, alts: &[Vec<Sym>], weights: &[f64]) -> OpId {
        let op = self.op();
        for (i, w) in alts.iter().enumerate() {
            self.rule(op, weights[i], &[], &Self::atoms(w));
        }
        op
    }

    /// Cadre : pre, filler (un seul tirage dans la règle), post.
    pub fn frame(&mut self, pre: &[Sym], slot: OpId, post: &[Sym]) -> OpId {
        let op = self.op();
        let mut parts = Self::atoms(pre);
        parts.push(Part::C(slot));
        parts.extend(Self::atoms(post));
        self.rule(op, 1.0, &[], &parts);
        op
    }

    /// Cadre à accord : pre, X, mid, X, post avec le MÊME nœud X (réécrit une fois).
    pub fn agree(&mut self, pre: &[Sym], slot: OpId, mid: &[Sym], post: &[Sym]) -> OpId {
        let op = self.op();
        let mut parts = Self::atoms(pre);
        parts.push(Part::S(0));
        parts.extend(Self::atoms(mid));
        parts.push(Part::S(0));
        parts.extend(Self::atoms(post));
        self.rule(op, 1.0, &[slot], &parts);
        op
    }

    /// Op de catégorie aléatoire (2 à 4 alternatives).
    pub fn new_slot(&mut self, rng: &mut Rng) -> OpId {
        let na = self.g.n_atoms;
        let k = rng.range(2, 4);
        let alts: Vec<Vec<Sym>> = (0..k).map(|_| Self::word(rng, na, 1, 3)).collect();
        let ws: Vec<f64> = (0..k).map(|_| 0.3 + rng.f64()).collect();
        self.choice(&alts, &ws)
    }

    /// Règle de départ.
    pub fn finish(mut self, top: &[Part]) -> Ptgrs {
        self.rule(0, 1.0, &[], top);
        self.g
    }
}

// ---------------------------------------------------------------------------
// Familles d'ENTRAÎNEMENT (les familles hors-distribution vivent dans `ladder`)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Repeat,
    Frame,
    Agree,
    Hier,
    List,
    Mixed,
}

pub const TRAIN_FAMILIES: [Family; 6] =
    [Family::Repeat, Family::Frame, Family::Agree, Family::Hier, Family::List, Family::Mixed];

fn frag_repeat(b: &mut Builder, rng: &mut Rng) -> Vec<Part> {
    let na = b.g.n_atoms;
    let nm = rng.range(1, 2);
    let mut ms: Vec<OpId> = Vec::new();
    for _ in 0..nm {
        let w = Builder::word(rng, na, 2, 4);
        ms.push(b.chunk(&w));
    }
    let mut top: Vec<Part> = Vec::new();
    for _ in 0..rng.range(3, 7) {
        top.push(Part::C(ms[rng.below(ms.len())]));
        if rng.chance(0.5) {
            top.push(Part::A(rng.below(na) as Sym));
        }
    }
    top
}

fn frag_frame(b: &mut Builder, rng: &mut Rng) -> Vec<Part> {
    let na = b.g.n_atoms;
    let pre = Builder::word(rng, na, 1, 3);
    let post = Builder::word(rng, na, 1, 2);
    let slot = b.new_slot(rng);
    let f = b.frame(&pre, slot, &post);
    (0..rng.range(4, 8)).map(|_| Part::C(f)).collect()
}

fn frag_agree(b: &mut Builder, rng: &mut Rng) -> Vec<Part> {
    let na = b.g.n_atoms;
    let pre = Builder::word(rng, na, 1, 2);
    let mid = Builder::word(rng, na, 1, 2);
    let post = Builder::word(rng, na, 1, 2);
    let slot = b.new_slot(rng);
    let t = b.agree(&pre, slot, &mid, &post);
    (0..rng.range(3, 6)).map(|_| Part::C(t)).collect()
}

fn frag_hier(b: &mut Builder, rng: &mut Rng, depth: usize) -> Vec<Part> {
    let na = b.g.n_atoms;
    let mut level: Vec<OpId> = Vec::new();
    for _ in 0..rng.range(2, 3) {
        let w = Builder::word(rng, na, 2, 3);
        level.push(b.chunk(&w));
    }
    for _ in 0..depth {
        let mut next: Vec<OpId> = Vec::new();
        for _ in 0..rng.range(1, 2) {
            let op = b.op();
            let k = rng.range(2, 3);
            let parts: Vec<Part> = (0..k).map(|_| Part::C(level[rng.below(level.len())])).collect();
            b.rule(op, 1.0, &[], &parts);
            next.push(op);
        }
        level = next;
    }
    (0..rng.range(3, 5)).map(|_| Part::C(level[rng.below(level.len())])).collect()
}

fn frag_list(b: &mut Builder, rng: &mut Rng) -> Vec<Part> {
    let na = b.g.n_atoms;
    let slot = b.new_slot(rng);
    let sep = rng.below(na) as Sym;
    let r = rng.range(4, 8);
    let mut top: Vec<Part> = Vec::new();
    for i in 0..r {
        top.push(Part::C(slot));
        if i + 1 < r {
            top.push(Part::A(sep));
        }
    }
    top
}

fn fragment(b: &mut Builder, rng: &mut Rng, idx: usize) -> Vec<Part> {
    match idx % 5 {
        0 => frag_repeat(b, rng),
        1 => frag_frame(b, rng),
        2 => frag_agree(b, rng),
        3 => {
            let d = rng.range(1, 2);
            frag_hier(b, rng, d)
        }
        _ => frag_list(b, rng),
    }
}

/// Construit une cause de la famille demandée sur `n_atoms` atomes.
pub fn build_cause(rng: &mut Rng, fam: Family, n_atoms: usize) -> Ptgrs {
    let mut b = Builder::new(n_atoms);
    let top: Vec<Part> = match fam {
        Family::Repeat => fragment(&mut b, rng, 0),
        Family::Frame => fragment(&mut b, rng, 1),
        Family::Agree => fragment(&mut b, rng, 2),
        Family::Hier => fragment(&mut b, rng, 3),
        Family::List => fragment(&mut b, rng, 4),
        Family::Mixed => {
            let i = rng.below(5);
            let j = (i + 1 + rng.below(4)) % 5;
            let mut a = fragment(&mut b, rng, i);
            let c = fragment(&mut b, rng, j);
            if rng.chance(0.5) {
                a.extend(c);
                a
            } else {
                let mut c = c;
                c.extend(a);
                c
            }
        }
    };
    b.finish(&top)
}

// ---------------------------------------------------------------------------
// Instances (E + vérité terrain réservée à l'évaluation)
// ---------------------------------------------------------------------------

pub struct Instance {
    pub e: Vec<u8>,
    /// Cause cachée : ne sert qu'à évaluer après coup.
    pub cause: Ptgrs,
    /// atome du générateur -> octet.
    pub atom_byte: Vec<u8>,
    pub spans: Vec<(u16, usize, usize)>,
}

impl Instance {
    /// Cause exprimée dans l'alphabet canonique de E (les atomes absents de E
    /// reçoivent des identifiants >= canon.n(), jamais confondus avec ceux de H).
    pub fn cause_in_canon(&self, canon: &Canon) -> Ptgrs {
        let mut map: Vec<Sym> = vec![0; self.cause.n_atoms];
        let mut extra = canon.n();
        for a in 0..self.cause.n_atoms {
            let s = canon.sym_of[self.atom_byte[a] as usize];
            if s != ABSENT {
                map[a] = s as Sym;
            } else {
                map[a] = extra as Sym;
                extra += 1;
            }
        }
        self.cause.relabel(&map, extra)
    }
}

/// Alphabet aléatoire : n octets distincts (ASCII imprimable une fois sur deux).
pub fn random_alphabet(rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut pool: Vec<u8> = if rng.chance(0.5) { (33..=126u8).collect() } else { (0..=255u8).collect() };
    rng.shuffle(&mut pool);
    pool.truncate(n.min(pool.len()));
    pool
}

/// Tire une cause et un E ; `None` si E sort de [len_lo, len_hi] ou dégénère.
pub fn draw_instance(rng: &mut Rng, fam: Family, len_lo: usize, len_hi: usize) -> Option<Instance> {
    let n_atoms = rng.range(4, 10);
    let cause = build_cause(rng, fam, n_atoms);
    if cause.validate().is_err() {
        return None;
    }
    let lim = Limits { max_len: len_hi, ..Limits::default() };
    let s = cause.generate(rng, &lim)?;
    if s.syms.len() < len_lo || s.syms.len() > len_hi {
        return None;
    }
    let atom_byte = random_alphabet(rng, n_atoms);
    let e: Vec<u8> = s.syms.iter().map(|&a| atom_byte[a as usize]).collect();
    let mut seen = [false; 256];
    let mut distinct = 0;
    for &b in &e {
        if !seen[b as usize] {
            seen[b as usize] = true;
            distinct += 1;
        }
    }
    if distinct < 3 {
        return None;
    }
    Some(Instance { e, cause, atom_byte, spans: s.spans })
}

/// Seule porte d'entrée de l'entraînement : E, sans la cause.
pub fn training_e(rng: &mut Rng) -> Vec<u8> {
    for _ in 0..200 {
        let fam = TRAIN_FAMILIES[rng.below(TRAIN_FAMILIES.len())];
        if let Some(inst) = draw_instance(rng, fam, 24, 160) {
            return inst.e;
        }
    }
    b"abcabcabcabcabcabcabcabc".to_vec()
}

// ---------------------------------------------------------------------------
// Équivalence observationnelle (pas égalité de graphes)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ObsDist {
    /// JS sur les chaînes exactes ; `None` si le support est quasi injectif.
    pub strings: Option<f64>,
    /// JS moyen sur les profils de 1-, 2-, 3-grammes.
    pub ngrams: f64,
    /// JS sur la distribution des longueurs.
    pub lengths: f64,
}

impl ObsDist {
    pub fn total(&self) -> f64 {
        match self.strings {
            Some(s) => (s + self.ngrams + self.lengths) / 3.0,
            None => (self.ngrams + self.lengths) / 2.0,
        }
    }
    pub fn quality(&self) -> f64 {
        1.0 - self.total()
    }
}

fn bump<K: Ord>(m: &mut BTreeMap<K, f64>, k: K) {
    *m.entry(k).or_insert(0.0) += 1.0;
}

/// Divergence de Jensen-Shannon (base 2), dans [0, 1].
fn js<K: Ord + Clone>(p: &BTreeMap<K, f64>, q: &BTreeMap<K, f64>) -> f64 {
    let sp: f64 = p.values().sum();
    let sq: f64 = q.values().sum();
    if sp <= 0.0 || sq <= 0.0 {
        return 1.0;
    }
    let mut both: BTreeMap<K, (f64, f64)> = BTreeMap::new();
    for (k, v) in p {
        both.entry(k.clone()).or_insert((0.0, 0.0)).0 = v / sp;
    }
    for (k, v) in q {
        both.entry(k.clone()).or_insert((0.0, 0.0)).1 = v / sq;
    }
    let mut d = 0.0;
    for (_, &(a, b)) in both.iter() {
        let m = 0.5 * (a + b);
        if a > 0.0 {
            d += 0.5 * a * (a / m).log2();
        }
        if b > 0.0 {
            d += 0.5 * b * (b / m).log2();
        }
    }
    d.clamp(0.0, 1.0)
}

fn profile(samples: &[Vec<Sym>], n: usize) -> BTreeMap<Vec<Sym>, f64> {
    let mut m: BTreeMap<Vec<Sym>, f64> = BTreeMap::new();
    for s in samples {
        if s.len() >= n {
            for w in s.windows(n) {
                bump(&mut m, w.to_vec());
            }
        }
    }
    m
}

pub fn sample_many(sys: &Ptgrs, rng: &mut Rng, n: usize, lim: &Limits) -> Vec<Vec<Sym>> {
    let mut out: Vec<Vec<Sym>> = Vec::new();
    let mut tries = 0;
    while out.len() < n && tries < n * 4 {
        tries += 1;
        if let Some(s) = sys.generate(rng, lim) {
            out.push(s.syms);
        }
    }
    out
}

/// Distance observationnelle entre deux systèmes (mêmes symboles, p. ex. C
/// ramenée à l'alphabet canonique de E et H compilée en PTGRS).
pub fn obs_dist(a: &Ptgrs, b: &Ptgrs, rng: &mut Rng, n: usize, lim: &Limits) -> Option<ObsDist> {
    let sa = sample_many(a, rng, n, lim);
    let sb = sample_many(b, rng, n, lim);
    if sa.is_empty() || sb.is_empty() || sa.len() < n / 2 || sb.len() < n / 2 {
        return None;
    }
    let mut ma: BTreeMap<Vec<Sym>, f64> = BTreeMap::new();
    let mut mb: BTreeMap<Vec<Sym>, f64> = BTreeMap::new();
    for s in &sa {
        bump(&mut ma, s.clone());
    }
    for s in &sb {
        bump(&mut mb, s.clone());
    }
    let uniq_a = ma.len() as f64 / sa.len() as f64;
    let uniq_b = mb.len() as f64 / sb.len() as f64;
    let strings = if uniq_a > 0.5 || uniq_b > 0.5 { None } else { Some(js(&ma, &mb)) };
    let mut ng = 0.0;
    for k in 1..=3 {
        ng += js(&profile(&sa, k), &profile(&sb, k));
    }
    ng /= 3.0;
    let mut la: BTreeMap<usize, f64> = BTreeMap::new();
    let mut lb: BTreeMap<usize, f64> = BTreeMap::new();
    for s in &sa {
        bump(&mut la, s.len() / 4);
    }
    for s in &sb {
        bump(&mut lb, s.len() / 4);
    }
    Some(ObsDist { strings, ngrams: ng, lengths: js(&la, &lb) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn familles_valides_et_generatives() {
        let mut rng = Rng::new(11);
        for fam in TRAIN_FAMILIES {
            let mut ok = false;
            for _ in 0..50 {
                if let Some(inst) = draw_instance(&mut rng, fam, 8, 400) {
                    assert!(inst.cause.validate().is_ok());
                    assert!(!inst.e.is_empty());
                    ok = true;
                    break;
                }
            }
            assert!(ok, "famille {:?} ne produit rien", fam);
        }
    }

    #[test]
    fn partage_donne_copie_exacte() {
        // T -> [X ; X ; X] avec X partagé : les trois occurrences sont identiques.
        let mut b = Builder::new(4);
        let slot = b.choice(&[vec![0], vec![1], vec![2, 3]], &[1.0, 1.0, 1.0]);
        let t = b.op();
        b.rule(t, 1.0, &[slot], &[Part::S(0), Part::A(3), Part::S(0)]);
        let g = b.finish(&[Part::C(t)]);
        let mut rng = Rng::new(5);
        for _ in 0..30 {
            let s = g.generate(&mut rng, &Limits::default()).unwrap().syms;
            let k = (s.len() - 1) / 2;
            assert_eq!(s[..k], s[k + 1..]);
        }
    }

    #[test]
    fn obs_dist_soi_meme_faible() {
        let mut rng = Rng::new(21);
        let inst = loop {
            if let Some(i) = draw_instance(&mut rng, Family::Frame, 8, 400) {
                break i;
            }
        };
        let mut r2 = Rng::new(22);
        let d = obs_dist(&inst.cause, &inst.cause, &mut r2, 100, &Limits::default()).unwrap();
        assert!(d.total() < 0.35);
    }
}