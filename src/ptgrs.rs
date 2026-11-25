//! PTGRS — Probabilistic Term Graph Rewriting System.
//!
//! RÈGLE DE FERMETURE : ce module ne fabrique que E pour l'entraînement. La cause
//! cachée C n'est jamais sérialisée dans le canal d'entrée, jamais rendue en
//! octets, jamais décrite. Elle n'est lue qu'après coup, par le harness, pour
//! l'étalonnage et l'équivalence observationnelle.
//!
//! Le probabiliste porte sur le CHOIX de la dérivation (grammaire, germe, suite
//! de pas). L'exécution est déterministe étant ces choix : C est donc un
//! transducteur exécutable sur des sondes, ce qui rend « équivalence
//! observationnelle » calculable sans comparer des graphes.
//!
//! Les mini-dictionnaires (barreau 2 de l'échelle) sont verrouillés par un mode
//! d'évaluation que le chemin d'entraînement ne pose jamais.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::eff::Perm;
use crate::rng::Rng;

pub const LIT: u8 = 255;
const MAX_MATCHES: usize = 256;
const MAX_PARSE_NODES: usize = 512;
const MAX_DEPTH: usize = 64;

// ---------------------------------------------------------------- graphe

#[derive(Clone, Debug)]
pub struct TermGraph {
    pub lab: Vec<u8>,
    pub ch: Vec<Vec<u32>>,
    pub lit: Vec<u8>,
    pub root: u32,
}

impl TermGraph {
    pub fn new() -> Self {
        TermGraph { lab: Vec::new(), ch: Vec::new(), lit: Vec::new(), root: 0 }
    }

    pub fn node(&mut self, l: u8, ch: Vec<u32>, lit: u8) -> u32 {
        self.lab.push(l);
        self.ch.push(ch);
        self.lit.push(lit);
        (self.lab.len() - 1) as u32
    }

    pub fn len(&self) -> usize {
        self.lab.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lab.is_empty()
    }

    /// Ensemble reachable, en ordre croissant d'identifiant. Ordre canonique :
    /// il fixe l'énumération des appariements, donc le rejeu déterministe.
    pub fn reachable(&self) -> Vec<u32> {
        if self.is_empty() {
            return Vec::new();
        }
        let mut seen = vec![false; self.len()];
        let mut stack = vec![self.root];
        while let Some(v) = stack.pop() {
            let vi = v as usize;
            if vi >= self.len() || seen[vi] {
                continue;
            }
            seen[vi] = true;
            for &c in self.ch[vi].iter() {
                stack.push(c);
            }
        }
        (0..self.len()).filter(|&i| seen[i]).map(|i| i as u32).collect()
    }
}

impl Default for TermGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub enum Pat {
    Var(u8),
    Node(u8, Vec<Pat>),
}

#[derive(Clone, Debug)]
pub enum Tpl {
    Var(u8),
    New(u8, Vec<Tpl>),
}

#[derive(Clone, Debug)]
pub struct Rule {
    pub lhs: Pat,
    pub rhs: Tpl,
    pub w: f32,
}

#[derive(Clone, Debug)]
pub struct Grammar {
    pub arity: Vec<u8>,
    pub rules: Vec<Rule>,
    pub byte_of: Vec<u8>,
    pub ref_byte: u8,
    pub family: u16,
}

impl Grammar {
    pub fn label_of_byte(&self, b: u8) -> Option<u8> {
        self.byte_of.iter().position(|&x| x == b).map(|i| i as u8)
    }
}

// ---------------------------------------------------------------- appariement

fn match_pat(
    g: &TermGraph,
    node: u32,
    pat: &Pat,
    binds: &mut Vec<Option<u32>>,
    undo: &mut Vec<u8>,
) -> bool {
    match pat {
        Pat::Var(v) => {
            let vi = *v as usize;
            while binds.len() <= vi {
                binds.push(None);
            }
            match binds[vi] {
                // non-linéarité : deux occurrences de la même variable exigent
                // le MÊME nœud, i.e. une contrainte de partage, pas de copie.
                Some(b) => b == node,
                None => {
                    binds[vi] = Some(node);
                    undo.push(*v);
                    true
                }
            }
        }
        Pat::Node(l, ps) => {
            let ni = node as usize;
            if ni >= g.len() || g.lab[ni] != *l || g.ch[ni].len() != ps.len() {
                return false;
            }
            for (k, p) in ps.iter().enumerate() {
                let c = g.ch[ni][k];
                if !match_pat(g, c, p, binds, undo) {
                    return false;
                }
            }
            true
        }
    }
}

/// Appariements d'une règle, en ordre canonique (nœuds reachables croissants).
fn matches(g: &TermGraph, reach: &[u32], rule: &Rule) -> Vec<(u32, Vec<Option<u32>>)> {
    let mut out = Vec::new();
    for &n in reach.iter() {
        let mut binds: Vec<Option<u32>> = Vec::new();
        let mut undo: Vec<u8> = Vec::new();
        if match_pat(g, n, &rule.lhs, &mut binds, &mut undo) {
            out.push((n, binds));
            if out.len() >= MAX_MATCHES {
                break;
            }
        }
    }
    out
}

fn instantiate(tpl: &Tpl, binds: &[Option<u32>], out: &mut TermGraph, arity: &[u8]) -> u32 {
    match tpl {
        Tpl::Var(v) => {
            let vi = *v as usize;
            if vi < binds.len() {
                if let Some(n) = binds[vi] {
                    return n;
                }
            }
            out.node(0, Vec::new(), 0)
        }
        Tpl::New(l, cs) => {
            let mut ch = Vec::with_capacity(cs.len());
            for c in cs.iter() {
                ch.push(instantiate(c, binds, out, arity));
            }
            out.node(*l, ch, 0)
        }
    }
}

/// Redirection globale : toutes les arêtes entrantes vers le nœud apparié sont
/// redirigées vers la nouvelle racine. Le motif a toujours une racine `Node`,
/// donc les variables lient des descendants stricts : aucun cycle n'est créé.
fn apply(g: &Grammar, tg: &mut TermGraph, target: u32, binds: &[Option<u32>], rule: &Rule) {
    let new_root = instantiate(&rule.rhs, binds, tg, &g.arity);
    for i in 0..tg.ch.len() {
        for c in tg.ch[i].iter_mut() {
            if *c == target {
                *c = new_root;
            }
        }
    }
    if tg.root == target {
        tg.root = new_root;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub rule: u16,
    pub pick: u16,
}

/// Exécution déterministe. Un pas sans appariement est un no-op enregistré :
/// la cause reste la même fonction, y compris sur des sondes où rien ne matche.
pub fn derive(g: &Grammar, seed: &TermGraph, steps: &[Step], max_nodes: usize) -> TermGraph {
    let mut tg = seed.clone();
    for s in steps.iter() {
        if tg.len() >= max_nodes {
            break;
        }
        let ri = s.rule as usize;
        if ri >= g.rules.len() {
            continue;
        }
        let reach = tg.reachable();
        let ms = matches(&tg, &reach, &g.rules[ri]);
        if ms.is_empty() {
            continue;
        }
        let k = (s.pick as usize) % ms.len();
        let (target, binds) = ms[k].clone();
        apply(g, &mut tg, target, &binds, &g.rules[ri]);
    }
    tg
}

/// Octets réservés par le format : image des étiquettes + marqueur de partage.
fn reserved(g: &Grammar) -> [bool; 256] {
    let mut r = [false; 256];
    for &b in g.byte_of.iter() {
        r[b as usize] = true;
    }
    r[g.ref_byte as usize] = true;
    r
}

fn fallback_byte(res: &[bool; 256]) -> u8 {
    (0..256usize).find(|&i| !res[i]).map(|i| i as u8).unwrap_or(0)
}

pub fn serialize(g: &Grammar, tg: &TermGraph) -> Vec<u8> {
    let mut out = Vec::new();
    if tg.is_empty() {
        return out;
    }
    let res = reserved(g);
    let fb = fallback_byte(&res);
    let mut index = vec![u32::MAX; tg.len()];
    let mut counter = 0u32;
    let mut stack: Vec<u32> = vec![tg.root];
    let mut guard = 0usize;
    while let Some(v) = stack.pop() {
        guard += 1;
        if guard > 4 * MAX_PARSE_NODES + 16 {
            break;
        }
        let vi = v as usize;
        if vi >= tg.len() {
            continue;
        }
        let seen = index[vi] != u32::MAX;
        if seen && index[vi] <= 255 {
            out.push(g.ref_byte);
            out.push(index[vi] as u8);
            continue;
        }
        if !seen {
            index[vi] = counter;
            counter += 1;
        }
        if tg.lab[vi] == LIT {
            // Un littéral opaque ne doit jamais émettre un octet réservé :
            // sinon il est relu comme en-tête de format.
            let b = tg.lit[vi];
            out.push(if res[b as usize] { fb } else { b });
            continue;
        }
        let l = tg.lab[vi] as usize;
        out.push(if l < g.byte_of.len() { g.byte_of[l] } else { fb });
        for &c in tg.ch[vi].iter().rev() {
            stack.push(c);
        }
    }
    out
}

/// Un littéral occupe un slot d'index, exactement comme dans `serialize`.
fn lit_node(out: &mut TermGraph, emitted: &mut Vec<u32>, b: u8) -> u32 {
    let n = out.node(LIT, Vec::new(), b);
    emitted.push(n);
    n
}

fn parse_rec(
    g: &Grammar,
    cur: &mut Cursor,
    out: &mut TermGraph,
    emitted: &mut Vec<u32>,
    depth: usize,
    sentinel: u32,
) -> u32 {
    if depth > MAX_DEPTH || out.len() >= MAX_PARSE_NODES {
        return sentinel;
    }
    let b = match cur.take() {
        Some(x) => x,
        None => return lit_node(out, emitted, 0),
    };
    if b == g.ref_byte {
        let idx = cur.take().unwrap_or(0) as usize;
        if idx < emitted.len() {
            return emitted[idx];
        }
        // flux malforme : un partage ne consomme pas de slot d'index.
        return out.node(LIT, Vec::new(), 0);
    }
    match g.label_of_byte(b) {
        None => lit_node(out, emitted, b),
        Some(l) => {
            let a = g.arity[l as usize] as usize;
            let me = out.node(l, Vec::new(), 0);
            emitted.push(me);
            let mut ch = Vec::with_capacity(a);
            for _ in 0..a {
                ch.push(parse_rec(g, cur, out, emitted, depth + 1, sentinel));
            }
            out.ch[me as usize] = ch;
            me
        }
    }
}

pub fn parse(g: &Grammar, bytes: &[u8]) -> TermGraph {
    let mut out = TermGraph::new();
    // Sentinelle HORS registre d'index : réutilisée à la borne, pour que le
    // nombre de nœuds reste borné sans décaler les index de partage.
    let sentinel = out.node(LIT, Vec::new(), 0);
    let mut cur = Cursor { b: bytes, i: 0 };
    let mut emitted = Vec::new();
    if bytes.is_empty() {
        out.root = sentinel;
        return out;
    }
    out.root = parse_rec(g, &mut cur, &mut out, &mut emitted, 0, sentinel);
    out
}

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self) -> Option<u8> {
        if self.i < self.b.len() {
            let x = self.b[self.i];
            self.i += 1;
            Some(x)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------- familles

#[derive(Clone, Copy, Debug)]
pub struct FamilySpec {
    pub id: u16,
    pub n_labels: usize,
    pub max_arity: usize,
    pub n_rules: usize,
    pub depth: usize,
    pub nonlinear: f32,
    pub recursive: bool,
    pub byte_lo: u8,
    pub byte_hi: u8,
    pub steps: usize,
    pub max_nodes: usize,
}

impl FamilySpec {
    /// Distribution d'entraînement. Seule source de E pour `train`.
    pub fn train() -> Self {
        FamilySpec {
            id: 1,
            n_labels: 6,
            max_arity: 2,
            n_rules: 5,
            depth: 2,
            nonlinear: 0.15,
            recursive: false,
            byte_lo: 16,
            byte_hi: 96,
            steps: 24,
            max_nodes: 200,
        }
    }

    /// Familles hors-distribution : espace d'étiquettes disjoint, arité 3,
    /// non-linéarité forte, composition profonde, récursion. Jamais entraînées.
    pub fn ood(kind: usize) -> Self {
        let mut s = Self::train();
        s.id = 100 + kind as u16;
        s.byte_lo = 128;
        s.byte_hi = 240;
        match kind {
            0 => {
                s.n_labels = 8;
            }
            1 => {
                s.max_arity = 3;
                s.n_labels = 7;
            }
            2 => {
                s.nonlinear = 0.75;
                s.n_rules = 7;
            }
            3 => {
                s.depth = 3;
                s.steps = 56;
                s.max_nodes = 240;
            }
            _ => {
                s.recursive = true;
                s.steps = 48;
                s.max_nodes = 240;
            }
        }
        s
    }
}

fn gen_pat(
    arity: &[u8],
    depth: usize,
    rng: &mut Rng,
    vars: &mut u8,
    used: &mut Vec<u8>,
    nonlinear: f32,
) -> Pat {
    if depth == 0 || rng.bernoulli(0.45) {
        if !used.is_empty() && rng.bernoulli(nonlinear) {
            return Pat::Var(used[rng.below(used.len())]);
        }
        let v = *vars;
        *vars = vars.saturating_add(1);
        used.push(v);
        return Pat::Var(v);
    }
    let cands: Vec<usize> = (0..arity.len()).filter(|&l| arity[l] > 0 && arity[l] <= 2).collect();
    if cands.is_empty() {
        let v = *vars;
        *vars = vars.saturating_add(1);
        used.push(v);
        return Pat::Var(v);
    }
    let l = cands[rng.below(cands.len())];
    let a = arity[l] as usize;
    let mut ch = Vec::new();
    for _ in 0..a {
        ch.push(gen_pat(arity, depth - 1, rng, vars, used, nonlinear));
    }
    Pat::Node(l as u8, ch)
}

fn gen_tpl(arity: &[u8], used: &[u8], rng: &mut Rng, root_lab: u8, recursive: bool, depth: usize) -> Tpl {
    let cands: Vec<usize> = (0..arity.len())
        .filter(|&l| arity[l] >= 1 && arity[l] <= 2)
        .collect();
    let lab = if recursive && rng.bernoulli(0.5) {
        root_lab as usize
    } else if cands.is_empty() {
        0
    } else {
        cands[rng.below(cands.len())]
    };
    let a = arity[lab] as usize;
    let mut ch = Vec::new();
    for k in 0..a {
        let reuse = !used.is_empty() && (k == 0 || rng.bernoulli(0.55));
        if reuse {
            ch.push(Tpl::Var(used[rng.below(used.len())]));
        } else if depth > 0 && rng.bernoulli(0.3) {
            ch.push(gen_tpl(arity, used, rng, root_lab, false, depth - 1));
        } else {
            let leaves: Vec<usize> = (0..arity.len()).filter(|&l| arity[l] == 0).collect();
            let lf = if leaves.is_empty() { 0 } else { leaves[rng.below(leaves.len())] };
            ch.push(Tpl::New(lf as u8, Vec::new()));
        }
    }
    Tpl::New(lab as u8, ch)
}

pub fn sample_grammar(spec: &FamilySpec, rng: &mut Rng) -> Grammar {
    let nl = spec.n_labels.clamp(3, 24);
    let mut arity = vec![0u8; nl];
    arity[0] = 0;
    if nl > 1 {
        arity[1] = 0;
    }
    for l in 2..nl {
        arity[l] = rng.between(1, spec.max_arity.max(1)) as u8;
    }
    if !arity.iter().any(|&a| a > 0) {
        arity[nl - 1] = 1;
    }

    let span = (spec.byte_hi as usize).saturating_sub(spec.byte_lo as usize) + 1;
    let mut pool: Vec<u8> = (0..span).map(|i| (spec.byte_lo as usize + i) as u8).collect();
    rng.shuffle(&mut pool);
    let need = nl + 1;
    let byte_of: Vec<u8> = pool.iter().take(need.min(pool.len())).cloned().collect();
    let ref_byte = *byte_of.last().unwrap_or(&0);
    let byte_of = byte_of[..byte_of.len().saturating_sub(1)].to_vec();

    let mut rules = Vec::new();
    for _ in 0..spec.n_rules.clamp(2, 12) {
        let roots: Vec<usize> = (0..nl).filter(|&l| arity[l] > 0).collect();
        let root = if roots.is_empty() { nl - 1 } else { roots[rng.below(roots.len())] };
        let mut vars = 0u8;
        let mut used = Vec::new();
        let a = arity[root] as usize;
        let mut ch = Vec::new();
        for _ in 0..a {
            ch.push(gen_pat(&arity, spec.depth, rng, &mut vars, &mut used, spec.nonlinear));
        }
        let lhs = Pat::Node(root as u8, ch);
        let rhs = gen_tpl(&arity, &used, rng, root as u8, spec.recursive, 1);
        rules.push(Rule { lhs, rhs, w: 0.5 + rng.next_f32() });
    }

    Grammar { arity, rules, byte_of, ref_byte, family: spec.id }
}

pub fn sample_seed(g: &Grammar, rng: &mut Rng) -> TermGraph {
    let mut tg = TermGraph::new();
    let leaves: Vec<usize> = (0..g.arity.len()).filter(|&l| g.arity[l] == 0).collect();
    let inner: Vec<usize> = (0..g.arity.len()).filter(|&l| g.arity[l] > 0).collect();
    let nleaf = rng.between(2, 4);
    for _ in 0..nleaf {
        let l = if leaves.is_empty() { 0 } else { leaves[rng.below(leaves.len())] };
        tg.node(l as u8, Vec::new(), 0);
    }
    let ninner = rng.between(2, 6);
    for _ in 0..ninner {
        if inner.is_empty() || tg.is_empty() {
            break;
        }
        let l = inner[rng.below(inner.len())];
        let a = g.arity[l] as usize;
        let mut ch = Vec::with_capacity(a);
        for _ in 0..a {
            ch.push(rng.below(tg.len()) as u32);
        }
        tg.node(l as u8, ch, 0);
    }
    tg.root = (tg.len() - 1) as u32;
    tg
}

fn sample_steps(g: &Grammar, tg: &mut TermGraph, rng: &mut Rng, n: usize, max_nodes: usize) -> Vec<Step> {
    let mut steps = Vec::new();
    for _ in 0..n {
        if tg.len() >= max_nodes {
            break;
        }
        let reach = tg.reachable();
        let mut ms: Vec<Vec<(u32, Vec<Option<u32>>)>> = Vec::with_capacity(g.rules.len());
        let mut w: Vec<f32> = Vec::with_capacity(g.rules.len());
        for r in g.rules.iter() {
            let m = matches(tg, &reach, r);  
            w.push(if m.is_empty() { 0.0 } else { r.w * (m.len() as f32).sqrt() });
            ms.push(m);
        }
        if w.iter().all(|&x| x == 0.0) {
            break;
        }
        let ri = rng.weighted(&w);
        let k = rng.below(ms[ri].len().max(1));
        let (target, binds) = ms[ri][k].clone();
        apply(g, tg, target, &binds, &g.rules[ri]);
        steps.push(Step { rule: ri as u16, pick: k as u16 });
    }
    steps
}

// ---------------------------------------------------------------- causes

#[derive(Clone, Debug)]
pub struct Cause {
    pub g: Grammar,
    pub seed: TermGraph,
    pub steps: Vec<Step>,
    pub max_nodes: usize,
}

impl Cause {
    pub fn regenerate(&self) -> Vec<u8> {
        serialize(&self.g, &derive(&self.g, &self.seed, &self.steps, self.max_nodes))
    }

    /// La cause vue comme transducteur : sondes d'intervention.
    /// C'est ICI, et seulement ici, qu'on lit C — jamais dans le canal d'entrée.
    pub fn observe(&self, x: &[u8]) -> Vec<u8> {
        let tg = parse(&self.g, x);
        serialize(&self.g, &derive(&self.g, &tg, &self.steps, self.max_nodes * 2))
    }
}

pub struct Episode {
    pub e: Vec<u8>,
    pub cause: Cause,
}

/// Unique fabrique de E pour l'entraînement. Ne produit jamais de mise en page
/// de dictionnaire, jamais de texte, jamais d'auto-description.
pub fn episode(spec: &FamilySpec, rng: &mut Rng) -> Episode {
    let mut best: Option<Episode> = None;
    for _ in 0..32 {
        let g = sample_grammar(spec, rng);
        let seed = sample_seed(&g, rng);
        let mut tg = seed.clone();
        let steps = sample_steps(&g, &mut tg, rng, spec.steps, spec.max_nodes);
        let e = serialize(&g, &tg);
        let ok = e.len() >= 24 && e.len() <= 2400;
        let ep = Episode { e, cause: Cause { g, seed, steps, max_nodes: spec.max_nodes } };
        if ok {
            return ep;
        }
        if best.is_none() {
            best = Some(ep);
        }
    }
    best.expect("episode : au moins une tentative")
}

// ---------------------------------------------------------------- observation

fn grams(b: &[u8]) -> BTreeSet<u32> {
    let mut s = BTreeSet::new();
    if b.len() >= 4 {
        for i in 0..=b.len() - 4 {
            s.insert(u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]));
        }
    }
    s
}

/// Distance comportementale. Définition opérationnelle, critiquable : elle est
/// grossière et ne sert QU'À l'évaluation, jamais de signal d'entraînement.
pub fn behavior_distance(a: &[u8], b: &[u8]) -> f32 {
    if a == b {
        return 0.0;
    }
    let ga = grams(a);
    let gb = grams(b);
    if ga.is_empty() && gb.is_empty() {
        return 1.0;
    }
    let inter = ga.intersection(&gb).count();
    let uni = ga.union(&gb).count();
    if uni == 0 {
        1.0
    } else {
        1.0 - inter as f32 / uni as f32
    }
}

pub fn probes(c: &Cause, rng: &mut Rng, k: usize) -> Vec<Vec<u8>> {
    let e = c.regenerate();
    let mut out: Vec<Vec<u8>> = Vec::new();
    out.push(serialize(&c.g, &c.seed));
    if e.len() > 2 {
        out.push(e[..e.len() / 2].to_vec());
        out.push(e[..(3 * e.len()) / 4].to_vec());
    }
    let p = Perm::random(rng);
    out.push(p.apply(&e));
    if e.len() > 8 {
        let mut sw = e.clone();
        let q = e.len() / 4;
        let (a, b) = sw.split_at_mut(2 * q);
        a[..q].swap_with_slice(&mut b[..q]);
        out.push(sw);
    }
    let rnd: Vec<u8> = (0..e.len().max(8)).map(|_| rng.below(256) as u8).collect();
    out.push(rnd);
    let half = c.steps.len() / 2;
    out.push(serialize(&c.g, &derive(&c.g, &c.seed, &c.steps[..half], c.max_nodes)));
    while out.len() < k {
        let i = rng.below(e.len().max(1));
        let mut v = e.clone();
        if i < v.len() {
            v[i] = rng.below(256) as u8;
        }
        out.push(v);
    }
    out.truncate(k.max(1));
    out
}

/// Équivalence observationnelle : moyenne des distances comportementales sur
/// les sondes. Pas d'égalité de graphes, pas d'égalité d'hypothèses.
pub fn obs_distance<F: Fn(&[u8]) -> Vec<u8>>(c: &Cause, f: F, ps: &[Vec<u8>]) -> f32 {
    if ps.is_empty() {
        return 1.0;
    }
    let mut s = 0.0f32;
    for x in ps.iter() {
        s += behavior_distance(&c.observe(x), &f(x));
    }
    s / ps.len() as f32
}

/// MDL : INSTRUMENT DE COMPARAISON. Jamais un objectif, jamais dans le signal.
pub fn mdl_bits(b: &[u8]) -> f32 {
    if b.is_empty() {
        return 0.0;
    }
    let mut dict: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
    for i in 0..256u32 {
        dict.insert(vec![i as u8], i);
    }
    let mut next = 256u32;
    let mut w: Vec<u8> = Vec::new();
    let mut nsym = 0usize;
    for &c in b {
        let mut wc = w.clone();
        wc.push(c);
        if dict.contains_key(&wc) {
            w = wc;
        } else {
            nsym += 1;
            dict.insert(wc, next);
            next += 1;
            w = vec![c];
        }
    }
    if !w.is_empty() {
        nsym += 1;
    }
    nsym as f32 * (next.max(2) as f32).log2()
}

// ---------------------------------------------------------------- barreau 2

static EVAL_MODE: AtomicBool = AtomicBool::new(false);

/// Posé uniquement par le harness. `train` ne le pose jamais : tout appel à
/// `minidict` hors évaluation est une violation de fermeture, et il avorte.
pub fn enter_eval() {
    EVAL_MODE.store(true, Ordering::SeqCst);
}

pub struct MiniDict {
    pub e: Vec<u8>,
    pub entries: Vec<(usize, usize)>,
    pub target: usize,
    pub query: (usize, usize),
    pub causes: Vec<Cause>,
}

/// Mini-dictionnaire engendré par PTGRS, C connue. ÉVALUATION SEULEMENT.
/// La requête partage la CAUSE d'une entrée (même suite de pas, autre germe)
/// mais son alphabet est renommé sur des octets DISJOINTS de ceux du
/// dictionnaire : zéro octet commun, donc similarité lexicale = hasard par
/// construction. Le seul lien restant est relationnel (forme canonique).
pub fn minidict(spec: &FamilySpec, rng: &mut Rng, k_entries: usize) -> MiniDict {
    assert!(EVAL_MODE.load(Ordering::SeqCst), "minidict hors evaluation : fermeture violee");
    let k = k_entries.clamp(3, 12);
    let g = sample_grammar(spec, rng);
    let mut causes = Vec::new();
    let mut bodies: Vec<Vec<u8>> = Vec::new();
    for _ in 0..k {
        let seed = sample_seed(&g, rng);
        let mut tg = seed.clone();
        let steps = sample_steps(&g, &mut tg, rng, spec.steps, spec.max_nodes);
        bodies.push(serialize(&g, &tg));
        causes.push(Cause { g: g.clone(), seed, steps, max_nodes: spec.max_nodes });
    }
    let target = rng.below(k);

    let mut used = [false; 256];
    for b in bodies.iter() {
        for &x in b.iter() {
            used[x as usize] = true;
        }
    }
    let sep: u8 = (0..=255u8).find(|&i| !used[i as usize]).unwrap_or(0);
    used[sep as usize] = true;
    let colon: u8 = (0..=255u8).find(|&i| !used[i as usize]).unwrap_or(1);
    used[colon as usize] = true;
    let qmark: u8 = (0..=255u8).find(|&i| !used[i as usize]).unwrap_or(2);
    used[qmark as usize] = true;

    let mut e: Vec<u8> = Vec::new();
    let mut entries = Vec::new();
    for (i, body) in bodies.iter().enumerate() {
        let head = [((i + 1) as u8).wrapping_mul(31).wrapping_add(3), 7u8];
        for &h in head.iter() {
            if !used[h as usize] {
                used[h as usize] = true;
                e.push(h);
            }
        }
        e.push(colon);
        let s = e.len();
        e.extend_from_slice(body);
        entries.push((s, e.len()));
        e.push(sep);
    }

    let qseed = sample_seed(&g, rng);
    let mut qtg = qseed.clone();
    let _ = sample_steps(&g, &mut qtg, rng, 0, spec.max_nodes);
    let qbody = serialize(&g, &derive(&g, &qseed, &causes[target].steps, spec.max_nodes));
    let free: Vec<u8> = (0..256).filter(|&i| !used[i]).map(|i| i as u8).collect();
    let mut map = [0u8; 256];
    for i in 0..256 {
        map[i] = i as u8;
    }
    let mut qused: Vec<u8> = Vec::new();
    for &x in qbody.iter() {
        if !qused.contains(&x) {
            qused.push(x);
        }
    }
    for (j, &x) in qused.iter().enumerate() {
        map[x as usize] = if j < free.len() { free[j] } else { x };
    }
    e.push(qmark);
    let qs = e.len();
    for &x in qbody.iter() {
        e.push(map[x as usize]);
    }
    let query = (qs, e.len());

    MiniDict { e, entries, target, query, causes }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_serialisation() {
        let mut rng = Rng::new(11);
        let spec = FamilySpec::train();
        for _ in 0..20 {
            let ep = episode(&spec, &mut rng);
            assert_eq!(ep.cause.regenerate(), ep.e, "C doit régénérer E exactement");
            let tg = parse(&ep.cause.g, &ep.e);
            assert_eq!(serialize(&ep.cause.g, &tg), ep.e, "round-trip");
        }
    }

    #[test]
    fn parseur_total() {
        let mut rng = Rng::new(12);
        let g = sample_grammar(&FamilySpec::train(), &mut rng);
        for n in 0..64 {
            let b: Vec<u8> = (0..n).map(|_| rng.below(256) as u8).collect();
            let tg = parse(&g, &b);
            assert!(!tg.is_empty());
            let _ = serialize(&g, &tg);
        }
    }

    #[test]
    fn ood_disjoint() {
        let a = FamilySpec::train();
        for k in 0..5 {
            let b = FamilySpec::ood(k);
            assert!(b.byte_lo > a.byte_hi, "espace d'étiquettes disjoint");
        }
    }

    #[test]
    fn distance_comportementale() {
        assert_eq!(behavior_distance(b"abcdabcd", b"abcdabcd"), 0.0);
        assert!(behavior_distance(b"abcdabcd", b"wxyzwxyz") > 0.9);
    }

    #[test]
    #[should_panic(expected = "fermeture violee")]
    fn minidict_interdit_hors_eval() {
        let mut rng = Rng::new(3);
        let _ = minidict(&FamilySpec::train(), &mut rng, 4);
    }

    #[test]
    fn canal_observation_point_fixe() {
        let mut rng = Rng::new(77);
        for k in 0..5 {
            let ep = episode(&FamilySpec::ood(k), &mut rng);
            for x in probes(&ep.cause, &mut rng, 8).iter() {
                let o1 = ep.cause.observe(x);
                let o2 = serialize(&ep.cause.g, &parse(&ep.cause.g, &o1));
                let o3 = serialize(&ep.cause.g, &parse(&ep.cause.g, &o2));
                assert_eq!(o2, o3, "canal d'observation instable : index replie");
            }
        }
    }
}