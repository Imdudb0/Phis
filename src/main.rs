#![forbid(unsafe_code)]
//! PTGRS-Abducer — abduction « universelle » méta-apprise en contexte.
//! Générateur : Probabilistic Term Graph Rewriting System (graphe hash-consé, règles probabilistes).
//! Réseau : petit Transformer causal (RMSNorm, RoPE, SiLU) + autodiff maison. Rust 2024, std seule.

use std::collections::{HashMap, HashSet};
use std::env;
use std::f32::consts::PI;

// ───────────────────────── Vocabulaire ─────────────────────────
const NV: usize = 4; // variables de règle X0..X3
const NS: usize = 12; // symboles globaux ; arités ré-tirées à chaque épisode
const T_BOS: usize = 0;
const T_SIG: usize = 1;
const T_RULE: usize = 2;
const T_ARROW: usize = 3;
const T_BAR: usize = 4;
const T_OBS: usize = 5;
const T_HYP: usize = 6;
const T_END: usize = 7;
const T_LP: usize = 8;
const T_RP: usize = 9;
const T_P: usize = 10; // +0..3 : proba quantifiée .25 .5 .75 1
const T_AR: usize = 14; // +0..3 : arité
const T_VAR: usize = 18; // +0..NV
const T_SYM: usize = 22; // +0..NS
const VOCAB: usize = T_SYM + NS;

const LMAX: usize = 4; // niveaux = nb max d'étapes de réécriture
const HMAX: usize = 12; // taille max d'une cause (arbre déplié)
const OMAX: usize = 20; // taille max d'une observation
const MAXT: usize = 256; // longueur max d'une séquence
const MAXH: usize = 48; // longueur max d'une hypothèse décodée

// ───────────────────────── RNG (SplitMix64) ─────────────────────────
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        let mut r = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xDEAD_BEEF_CAFE_F00D);
        r.next_u64();
        r
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn chance(&mut self, p: f32) -> bool {
        self.unif() < p
    }
    fn normal(&mut self) -> f32 {
        let u1 = self.unif().max(1e-7);
        let u2 = self.unif();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// ──────────────── Graphe de termes hash-consé + réécriture ────────────────
type Id = u32;
type Bind = [Option<Id>; NV];

#[derive(Clone, PartialEq, Eq, Hash)]
struct Node {
    sym: usize,
    kids: Vec<Id>,
}

#[derive(Clone, PartialEq)]
enum Pat {
    Var(u8),
    App(usize, Vec<Pat>),
}

struct Rule {
    lhs: Pat,
    outs: Vec<(f32, Pat)>, // (probabilité, membre droit)
}

struct Arena {
    nodes: Vec<Node>,
    tsize: Vec<u32>, // taille de l'arbre déplié
    map: HashMap<Node, Id>,
}

impl Arena {
    fn new() -> Arena {
        Arena { nodes: Vec::new(), tsize: Vec::new(), map: HashMap::new() }
    }

    /// Partage maximal : égalité structurelle ⇔ égalité d'Id.
    fn mk(&mut self, sym: usize, kids: Vec<Id>) -> Id {
        let n = Node { sym, kids };
        if let Some(&id) = self.map.get(&n) {
            return id;
        }
        let id = self.nodes.len() as Id;
        let mut sz = 1u32;
        for &k in &n.kids {
            sz = sz.saturating_add(self.tsize[k as usize]);
        }
        self.tsize.push(sz);
        self.nodes.push(n.clone());
        self.map.insert(n, id);
        id
    }

    fn size(&self, id: Id) -> usize {
        self.tsize[id as usize] as usize
    }

    /// Filtrage d'un motif sur un nœud (variable répétée ⇒ même sous-graphe).
    fn matches(&self, p: &Pat, id: Id, b: &mut Bind) -> bool {
        match p {
            Pat::Var(v) => {
                let v = *v as usize;
                let cur = b[v];
                match cur {
                    Some(x) => x == id,
                    None => {
                        b[v] = Some(id);
                        true
                    }
                }
            }
            Pat::App(s, ps) => {
                let n = &self.nodes[id as usize];
                if n.sym != *s || n.kids.len() != ps.len() {
                    return false;
                }
                for (q, &k) in ps.iter().zip(n.kids.iter()) {
                    if !self.matches(q, k, b) {
                        return false;
                    }
                }
                true
            }
        }
    }

    fn inst(&mut self, p: &Pat, b: &Bind) -> Id {
        match p {
            Pat::Var(v) => b[*v as usize].expect("variable libre dans un membre droit"),
            Pat::App(s, ps) => {
                let mut kids = Vec::with_capacity(ps.len());
                for q in ps {
                    kids.push(self.inst(q, b));
                }
                self.mk(*s, kids)
            }
        }
    }

    fn reach(&self, root: Id) -> Vec<Id> {
        let mut seen: HashSet<Id> = HashSet::new();
        let mut out = Vec::new();
        let mut st = vec![root];
        while let Some(i) = st.pop() {
            if !seen.insert(i) {
                continue;
            }
            out.push(i);
            st.extend(self.nodes[i as usize].kids.iter().copied());
        }
        out
    }

    /// Tous les redex (nœud, règle, liaisons) du graphe enraciné en `root`.
    fn redexes(&self, root: Id, rules: &[Rule]) -> Vec<(Id, usize, Bind)> {
        let mut out = Vec::new();
        let mut ids = self.reach(root);
        ids.sort_unstable();
        for id in ids {
            for (ri, r) in rules.iter().enumerate() {
                let mut b: Bind = [None; NV];
                if self.matches(&r.lhs, id, &mut b) {
                    out.push((id, ri, b));
                }
            }
        }
        out
    }

    /// Réécriture de graphe : le nœud `target` est remplacé par `rep` pour tous ses parents.
    fn replace(&mut self, root: Id, target: Id, rep: Id, memo: &mut HashMap<Id, Id>) -> Id {
        if root == target {
            return rep;
        }
        if let Some(&m) = memo.get(&root) {
            return m;
        }
        let n = self.nodes[root as usize].clone();
        let mut kids = Vec::with_capacity(n.kids.len());
        let mut changed = false;
        for &k in &n.kids {
            let nk = self.replace(k, target, rep, memo);
            changed |= nk != k;
            kids.push(nk);
        }
        let r = if changed { self.mk(n.sym, kids) } else { root };
        memo.insert(root, r);
        r
    }

    /// Un pas stochastique : redex uniforme, issue tirée selon les probabilités de la règle.
    fn step(&mut self, root: Id, rules: &[Rule], rng: &mut Rng) -> Option<Id> {
        let rx = self.redexes(root, rules);
        if rx.is_empty() {
            return None;
        }
        let (node, ri, b) = rx[rng.below(rx.len())];
        let outs = &rules[ri].outs;
        let mut u = rng.unif();
        let mut chosen = &outs[outs.len() - 1].1;
        for (p, rhs) in outs {
            if u < *p {
                chosen = rhs;
                break;
            }
            u -= *p;
        }
        let rep = self.inst(chosen, &b);
        Some(self.replace(root, node, rep, &mut HashMap::new()))
    }

    fn ser(&self, id: Id, out: &mut Vec<usize>) {
        let n = &self.nodes[id as usize];
        out.push(T_SYM + n.sym);
        if !n.kids.is_empty() {
            out.push(T_LP);
            for &k in &n.kids {
                self.ser(k, out);
            }
            out.push(T_RP);
        }
    }

    fn show(&self, id: Id) -> String {
        let n = &self.nodes[id as usize];
        if n.kids.is_empty() {
            return format!("s{}", n.sym);
        }
        let ks: Vec<String> = n.kids.iter().map(|&k| self.show(k)).collect();
        format!("s{}({})", n.sym, ks.join(","))
    }
}

fn ser_pat(p: &Pat, out: &mut Vec<usize>) {
    match p {
        Pat::Var(v) => out.push(T_VAR + *v as usize),
        Pat::App(s, ps) => {
            out.push(T_SYM + *s);
            if !ps.is_empty() {
                out.push(T_LP);
                for q in ps {
                    ser_pat(q, out);
                }
                out.push(T_RP);
            }
        }
    }
}

fn show_pat(p: &Pat) -> String {
    match p {
        Pat::Var(v) => format!("X{}", v),
        Pat::App(s, ps) if ps.is_empty() => format!("s{}", s),
        Pat::App(s, ps) => {
            let ks: Vec<String> = ps.iter().map(show_pat).collect();
            format!("s{}({})", s, ks.join(","))
        }
    }
}

fn pat_size(p: &Pat) -> usize {
    match p {
        Pat::Var(_) => 1,
        Pat::App(_, ps) => 1 + ps.iter().map(pat_size).sum::<usize>(),
    }
}

fn pat_vars(p: &Pat, seen: &mut [bool; NV]) {
    match p {
        Pat::Var(v) => seen[*v as usize] = true,
        Pat::App(_, ps) => {
            for q in ps {
                pat_vars(q, seen);
            }
        }
    }
}

// ───────────────────────── Générateur PTGRS ─────────────────────────
struct Sig {
    syms: Vec<usize>,
    arity: [usize; NS],
    has: [bool; NS],
}

fn random_sig(rng: &mut Rng) -> Sig {
    let n = 5 + rng.below(4); // 5..=8 symboles
    let mut ids: Vec<usize> = (0..NS).collect();
    for i in (1..NS).rev() {
        let j = rng.below(i + 1);
        ids.swap(i, j);
    }
    ids.truncate(n);
        let ncst = 2 + rng.below(2); // 2..=3 constantes
    let mut sig = Sig { syms: ids.clone(), arity: [0; NS], has: [false; NS] };
    for (i, &s) in ids.iter().enumerate() {
        sig.has[s] = true;
        sig.arity[s] = if i < ncst {
            0
        } else {
            let u = rng.unif();
            if u < 0.3 { 1 } else if u < 0.9 { 2 } else { 3 }
        };
    }
    sig.syms.sort_unstable();
    sig
}

fn pick_sym(rng: &mut Rng, sig: &Sig, compound: bool) -> usize {
    let c: Vec<usize> = sig.syms.iter().copied().filter(|&s| (sig.arity[s] > 0) == compound).collect();
    c[rng.below(c.len())]
}

fn lhs_slot(rng: &mut Rng, sig: &Sig, depth: usize, nv: &mut usize) -> Pat {
    if depth > 0 && rng.chance(0.3) {
        let comp = rng.chance(0.8);
        let s = pick_sym(rng, sig, comp);
        let a = sig.arity[s];
        let mut ks = Vec::with_capacity(a);
        for _ in 0..a {
            ks.push(lhs_slot(rng, sig, depth - 1, nv));
        }
        Pat::App(s, ks)
    } else if *nv > 0 && (*nv >= NV || rng.chance(0.08)) {
        Pat::Var(rng.below(*nv) as u8)
    } else {
        *nv += 1;
        Pat::Var((*nv - 1) as u8)
    }
}

fn rand_lhs(rng: &mut Rng, sig: &Sig) -> (Pat, usize) {
    loop {
        let mut nv = 0usize;
        let p = if rng.chance(0.1) {
            Pat::App(pick_sym(rng, sig, false), Vec::new())
        } else {
            let s = pick_sym(rng, sig, true);
            let mut ks = Vec::new();
            for _ in 0..sig.arity[s] {
                ks.push(lhs_slot(rng, sig, 1, &mut nv));
            }
            Pat::App(s, ks)
        };
        if pat_size(&p) <= 7 {
            return (p, nv);
        }
    }
}

fn rand_rhs(rng: &mut Rng, sig: &Sig, depth: usize, nv: usize, unused: &mut Vec<usize>) -> Pat {
    let pc = if depth == 2 { 0.9 } else { 0.45 };
    if depth > 0 && rng.chance(pc) {
        let s = pick_sym(rng, sig, true);
        let mut ks = Vec::new();
        for _ in 0..sig.arity[s] {
            ks.push(rand_rhs(rng, sig, depth - 1, nv, unused));
        }
        Pat::App(s, ks)
    } else if !unused.is_empty() && rng.chance(0.85) {
        let i = rng.below(unused.len());
        Pat::Var(unused.swap_remove(i) as u8)
    } else if nv > 0 && rng.chance(0.4) {
        Pat::Var(rng.below(nv) as u8)
    } else {
        Pat::App(pick_sym(rng, sig, false), Vec::new())
    }
}

fn random_rules(rng: &mut Rng, sig: &Sig, nr: usize) -> Vec<Rule> {
    let mut rules: Vec<Rule> = Vec::new();
    while rules.len() < nr {
        let (lhs, nv) = rand_lhs(rng, sig);
        let n_out = if rng.chance(0.35) { 2 } else { 1 };
        let mut rhss: Vec<Pat> = Vec::new();
        let mut tries = 0;
        while rhss.len() < n_out && tries < 50 {
            tries += 1;
            let mut unused: Vec<usize> = (0..nv).collect();
            let rhs = rand_rhs(rng, sig, 2, nv, &mut unused);
            if pat_size(&rhs) <= 8 && rhs != lhs && !rhss.contains(&rhs) {
                rhss.push(rhs);
            }
        }
        if rhss.is_empty() {
            continue;
        }
        let probs: Vec<f32> = if rhss.len() == 1 {
            vec![1.0]
        } else {
            let p = [0.25f32, 0.5, 0.75][rng.below(3)];
            vec![p, 1.0 - p]
        };
        rules.push(Rule { lhs, outs: probs.into_iter().zip(rhss).collect() });
    }
    rules
}

/// Terme initial aléatoire avec des instances de membres gauches « plantées » (redex garantis).
fn gen_term(ar: &mut Arena, rng: &mut Rng, sig: &Sig, rules: &[Rule], depth: usize) -> Id {
    if depth == 0 || rng.chance(0.15) {
        let c = pick_sym(rng, sig, false);
        return ar.mk(c, Vec::new());
    }
    if rng.chance(0.4) {
        let r = &rules[rng.below(rules.len())];
        let mut seen = [false; NV];
        pat_vars(&r.lhs, &mut seen);
        let mut b: Bind = [None; NV];
        for v in 0..NV {
            if seen[v] {
                b[v] = Some(gen_term(ar, rng, sig, rules, depth - 1));
            }
        }
        return ar.inst(&r.lhs, &b);
    }
    let f = pick_sym(rng, sig, true);
    let mut ks = Vec::new();
    for _ in 0..sig.arity[f] {
        ks.push(gen_term(ar, rng, sig, rules, depth - 1));
    }
    ar.mk(f, ks)
}

// ───────────────────────── Épisodes ─────────────────────────
struct Episode {
    sig: Sig,
    rules: Vec<Rule>,
    ar: Arena,
    h: Id,       // cause vraie
    o: Id,       // observation
    kmax: usize, // nb max d'étapes (= niveau)
    toks: Vec<usize>,
    plen: usize, // longueur du prompt (finit par HYP)
}

fn ptok(p: f32) -> usize {
    if p > 0.99 { 3 } else if p > 0.7 { 2 } else if p > 0.4 { 1 } else { 0 }
}

/// BOS SIG (sym arité)* (RULE lhs ARROW P rhs (BAR P rhs)*)* OBS obs HYP [cause END]
fn encode(sig: &Sig, rules: &[Rule], ar: &Arena, o: Id, h: Option<Id>) -> (Vec<usize>, usize) {
    let mut t = vec![T_BOS, T_SIG];
    for &s in &sig.syms {
        t.push(T_SYM + s);
        t.push(T_AR + sig.arity[s]);
    }
    for r in rules {
        t.push(T_RULE);
        ser_pat(&r.lhs, &mut t);
        t.push(T_ARROW);
        for (i, (p, rhs)) in r.outs.iter().enumerate() {
            if i > 0 {
                t.push(T_BAR);
            }
            t.push(T_P + ptok(*p));
            ser_pat(rhs, &mut t);
        }
    }
    t.push(T_OBS);
    ar.ser(o, &mut t);
    t.push(T_HYP);
    let plen = t.len();
    if let Some(h) = h {
        ar.ser(h, &mut t);
        t.push(T_END);
    }
    (t, plen)
}

fn make_episode(rng: &mut Rng, level: usize) -> Episode {
    loop {
        let sig = random_sig(rng);
        let nr = 1 + rng.below((level + 1).min(4));
        let rules = random_rules(rng, &sig, nr);
        let mut ar = Arena::new();
        let depth = if level >= 3 { 3 } else { 2 };
        let h = gen_term(&mut ar, rng, &sig, &rules, depth);
        if ar.size(h) > HMAX {
            continue;
        }
        let k = 1 + rng.below(level);
        let mut o = h;
        let mut ok = true;
        for _ in 0..k {
            match ar.step(o, &rules, rng) {
                Some(n) if ar.size(n) <= OMAX => o = n,
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok || o == h {
            continue;
        }
        let (toks, plen) = encode(&sig, &rules, &ar, o, Some(h));
        if toks.len() > MAXT {
            continue;
        }
        return Episode { sig, rules, ar, h, o, kmax: level, toks, plen };
    }
}

// ───────────────────────── Autodiff minimal (tenseurs 2D f32) ─────────────────────────
fn mm(a: &[f32], b: &[f32], n: usize, k: usize, m: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; n * m];
    for i in 0..n {
        let orow = &mut o[i * m..(i + 1) * m];
        for p in 0..k {
            let av = a[i * k + p];
            let brow = &b[p * m..(p + 1) * m];
            for j in 0..m {
                orow[j] += av * brow[j];
            }
        }
    }
    o
}

/// RoPE en place ; sign = +1 (avant) ou -1 (rotation inverse = rétropropagation).
fn rope_apply(x: &mut [f32], t: usize, d: usize, heads: usize, sign: f32) {
    let dh = d / heads;
    let half = dh / 2;
    for pos in 0..t {
        for h in 0..heads {
            let base = pos * d + h * dh;
            for c in 0..half {
                let freq = 10000f32.powf(-(c as f32) / half as f32);
                let (s, co) = (pos as f32 * freq).sin_cos();
                let s = s * sign;
                let (a, b) = (x[base + c], x[base + half + c]);
                x[base + c] = a * co - b * s;
                x[base + half + c] = a * s + b * co;
            }
        }
    }
}

/// Attention causale multi-têtes ; renvoie (sortie, probabilités T×T par tête).
fn attn_fwd(q: &[f32], k: &[f32], v: &[f32], t: usize, d: usize, heads: usize) -> (Vec<f32>, Vec<f32>) {
    let dh = d / heads;
    let sc = 1.0 / (dh as f32).sqrt();
    let mut out = vec![0.0f32; t * d];
    let mut p = vec![0.0f32; heads * t * t];
    for h in 0..heads {
        let o = h * dh;
        for i in 0..t {
            let pr = &mut p[(h * t + i) * t..(h * t + i + 1) * t];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..=i {
                let mut s = 0.0f32;
                for c in 0..dh {
                    s += q[i * d + o + c] * k[j * d + o + c];
                }
                s *= sc;
                pr[j] = s;
                if s > mx {
                    mx = s;
                }
            }
            let mut z = 0.0f32;
            for j in 0..=i {
                let e = (pr[j] - mx).exp();
                pr[j] = e;
                z += e;
            }
            for j in 0..=i {
                let w = pr[j] / z;
                pr[j] = w;
                for c in 0..dh {
                    out[i * d + o + c] += w * v[j * d + o + c];
                }
            }
        }
    }
    (out, p)
}

fn attn_bwd(
    q: &[f32], k: &[f32], v: &[f32], p: &[f32], g: &[f32], t: usize, d: usize, heads: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dh = d / heads;
    let sc = 1.0 / (dh as f32).sqrt();
    let mut dq = vec![0.0f32; t * d];
    let mut dk = vec![0.0f32; t * d];
    let mut dv = vec![0.0f32; t * d];
    let mut dp = vec![0.0f32; t];
    for h in 0..heads {
        let o = h * dh;
        for i in 0..t {
            let pr = &p[(h * t + i) * t..(h * t + i + 1) * t];
            let mut dot = 0.0f32;
            for j in 0..=i {
                let mut s = 0.0f32;
                for c in 0..dh {
                    s += g[i * d + o + c] * v[j * d + o + c];
                    dv[j * d + o + c] += pr[j] * g[i * d + o + c];
                }
                dp[j] = s;
                dot += pr[j] * s;
            }
            for j in 0..=i {
                let ds = pr[j] * (dp[j] - dot) * sc;
                for c in 0..dh {
                    dq[i * d + o + c] += ds * k[j * d + o + c];
                    dk[j * d + o + c] += ds * q[i * d + o + c];
                }
            }
        }
    }
    (dq, dk, dv)
}

enum Op {
    Leaf,
    MatMul(usize, usize),
    Add(usize, usize),
    Silu(usize),
    Rms(usize, usize),                         // x, gain
    Embed(usize, Vec<usize>),                  // table, ids
    Rope(usize, usize),                        // x, heads
    Attn(usize, usize, usize, usize, Vec<f32>), // q, k, v, heads, probas
    Xent(usize, Vec<usize>, Vec<bool>, f32),   // logits, cibles, masque, échelle
}

struct Tape {
    val: Vec<Vec<f32>>,
    shp: Vec<(usize, usize)>,
    ops: Vec<Op>,
    grad: Vec<Vec<f32>>,
}

impl Tape {
    fn new() -> Tape {
        Tape { val: Vec::new(), shp: Vec::new(), ops: Vec::new(), grad: Vec::new() }
    }
    fn push(&mut self, v: Vec<f32>, r: usize, c: usize, op: Op) -> usize {
        self.val.push(v);
        self.shp.push((r, c));
        self.ops.push(op);
        self.val.len() - 1
    }
    fn leaf(&mut self, v: Vec<f32>, r: usize, c: usize) -> usize {
        self.push(v, r, c, Op::Leaf)
    }
    fn matmul(&mut self, a: usize, b: usize) -> usize {
        let (n, k) = self.shp[a];
        let (k2, m) = self.shp[b];
        assert_eq!(k, k2, "dimensions incompatibles dans matmul");
        let o = mm(&self.val[a], &self.val[b], n, k, m);
        self.push(o, n, m, Op::MatMul(a, b))
    }
    fn add(&mut self, a: usize, b: usize) -> usize {
        let (n, m) = self.shp[a];
        let o: Vec<f32> = self.val[a].iter().zip(&self.val[b]).map(|(x, y)| x + y).collect();
        self.push(o, n, m, Op::Add(a, b))
    }
    fn silu(&mut self, a: usize) -> usize {
        let (n, m) = self.shp[a];
        let o: Vec<f32> = self.val[a].iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        self.push(o, n, m, Op::Silu(a))
    }
    fn rms(&mut self, x: usize, g: usize) -> usize {
        let (n, d) = self.shp[x];
        let mut o = vec![0.0f32; n * d];
        {
            let xv = &self.val[x];
            let gv = &self.val[g];
            for i in 0..n {
                let row = &xv[i * d..(i + 1) * d];
                let r = (row.iter().map(|v| v * v).sum::<f32>() / d as f32 + 1e-5).sqrt();
                for j in 0..d {
                    o[i * d + j] = gv[j] * row[j] / r;
                }
            }
        }
        self.push(o, n, d, Op::Rms(x, g))
    }
    fn embed(&mut self, tab: usize, ids: &[usize]) -> usize {
        let (_, d) = self.shp[tab];
        let mut o = Vec::with_capacity(ids.len() * d);
        for &i in ids {
            o.extend_from_slice(&self.val[tab][i * d..(i + 1) * d]);
        }
        self.push(o, ids.len(), d, Op::Embed(tab, ids.to_vec()))
    }
    fn rope(&mut self, x: usize, heads: usize) -> usize {
        let (t, d) = self.shp[x];
        let mut o = self.val[x].clone();
        rope_apply(&mut o, t, d, heads, 1.0);
        self.push(o, t, d, Op::Rope(x, heads))
    }
    fn attn(&mut self, q: usize, k: usize, v: usize, heads: usize) -> usize {
        let (t, d) = self.shp[q];
        let (o, p) = attn_fwd(&self.val[q], &self.val[k], &self.val[v], t, d, heads);
        self.push(o, t, d, Op::Attn(q, k, v, heads, p))
    }
    /// Perte = échelle × Σ_{lignes masquées} NLL(softmax(logits), cible).
    fn xent(&mut self, lg: usize, tg: Vec<usize>, mk: Vec<bool>, sc: f32) -> usize {
        let (n, c) = self.shp[lg];
        let mut loss = 0.0f32;
        for i in 0..n {
            if !mk[i] {
                continue;
            }
            let row = &self.val[lg][i * c..(i + 1) * c];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = mx + row.iter().map(|&x| (x - mx).exp()).sum::<f32>().ln();
            loss += lse - row[tg[i]];
        }
        self.push(vec![loss * sc], 1, 1, Op::Xent(lg, tg, mk, sc))
    }

    fn backward(&mut self, root: usize) {
        let Tape { val, shp, ops, grad } = self;
        *grad = val.iter().map(|v| vec![0.0f32; v.len()]).collect();
        grad[root][0] = 1.0;
        for i in (0..=root).rev() {
            let (lo, hi) = grad.split_at_mut(i);
            let g = &hi[0];
            match &ops[i] {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    for (d, x) in lo[*a].iter_mut().zip(g.iter()) {
                        *d += *x;
                    }
                    for (d, x) in lo[*b].iter_mut().zip(g.iter()) {
                        *d += *x;
                    }
                }
                Op::MatMul(a, b) => {
                    let (n, k) = shp[*a];
                    let (_, m) = shp[*b];
                    let bv = &val[*b];
                    let da = &mut lo[*a];
                    for r in 0..n {
                        let grow = &g[r * m..(r + 1) * m];
                        for p in 0..k {
                            let brow = &bv[p * m..(p + 1) * m];
                            let mut s = 0.0f32;
                            for j in 0..m {
                                s += grow[j] * brow[j];
                            }
                            da[r * k + p] += s;
                        }
                    }
                    let av = &val[*a];
                    let db = &mut lo[*b];
                    for r in 0..n {
                        let grow = &g[r * m..(r + 1) * m];
                        for p in 0..k {
                            let x = av[r * k + p];
                            let drow = &mut db[p * m..(p + 1) * m];
                            for j in 0..m {
                                drow[j] += x * grow[j];
                            }
                        }
                    }
                }
                Op::Silu(a) => {
                    let xv = &val[*a];
                    let da = &mut lo[*a];
                    for j in 0..xv.len() {
                        let x = xv[j];
                        let s = 1.0 / (1.0 + (-x).exp());
                        da[j] += g[j] * s * (1.0 + x * (1.0 - s));
                    }
                }
                Op::Rms(xi, gi) => {
                    let (n, d) = shp[*xi];
                    let xv = &val[*xi];
                    let gv = &val[*gi];
                    let mut dx = vec![0.0f32; n * d];
                    let mut dg = vec![0.0f32; d];
                    for r in 0..n {
                        let row = &xv[r * d..(r + 1) * d];
                        let gr = &g[r * d..(r + 1) * d];
                        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32 + 1e-5;
                        let rr = ms.sqrt();
                        let mut dot = 0.0f32;
                        for j in 0..d {
                            dot += gv[j] * gr[j] * row[j];
                            dg[j] += gr[j] * row[j] / rr;
                        }
                        let coef = dot / (d as f32 * rr * ms);
                        for j in 0..d {
                            dx[r * d + j] = gv[j] * gr[j] / rr - row[j] * coef;
                        }
                    }
                    for (a, b) in lo[*xi].iter_mut().zip(&dx) {
                        *a += *b;
                    }
                    for (a, b) in lo[*gi].iter_mut().zip(&dg) {
                        *a += *b;
                    }
                }
                Op::Embed(tab, ids) => {
                    let (_, d) = shp[*tab];
                    let dt = &mut lo[*tab];
                    for (r, &id) in ids.iter().enumerate() {
                        for j in 0..d {
                            dt[id * d + j] += g[r * d + j];
                        }
                    }
                }
                Op::Rope(x, heads) => {
                    let (t, d) = shp[*x];
                    let mut dx = g.clone();
                    rope_apply(&mut dx, t, d, *heads, -1.0);
                    for (a, b) in lo[*x].iter_mut().zip(&dx) {
                        *a += *b;
                    }
                }
                Op::Attn(q, k, v, heads, p) => {
                    let (t, d) = shp[*q];
                    let (dq, dk, dv) = attn_bwd(&val[*q], &val[*k], &val[*v], p, g, t, d, *heads);
                    for (a, b) in lo[*q].iter_mut().zip(&dq) {
                        *a += *b;
                    }
                    for (a, b) in lo[*k].iter_mut().zip(&dk) {
                        *a += *b;
                    }
                    for (a, b) in lo[*v].iter_mut().zip(&dv) {
                        *a += *b;
                    }
                }
                Op::Xent(lg, tg, mk, sc) => {
                    let (n, c) = shp[*lg];
                    let lv = &val[*lg];
                    let dl = &mut lo[*lg];
                    let g0 = g[0] * *sc;
                    for r in 0..n {
                        if !mk[r] {
                            continue;
                        }
                        let row = &lv[r * c..(r + 1) * c];
                        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let z: f32 = row.iter().map(|&x| (x - mx).exp()).sum();
                        for j in 0..c {
                            let pr = (row[j] - mx).exp() / z;
                            let tgt = if j == tg[r] { 1.0 } else { 0.0 };
                            dl[r * c + j] += g0 * (pr - tgt);
                        }
                    }
                }
            }
        }
    }
}

// ═══════════════ SUITE de src/main.rs (à coller après `impl Tape { … }`) ═══════════════

// ───────────────────────── Modèle ─────────────────────────
const D: usize = 48; // largeur
const NH: usize = 4; // têtes (D/NH = 12, pair pour RoPE)
const NL: usize = 3; // couches
const FF: usize = 128; // largeur du MLP
const PL: usize = 8; // tenseurs par couche : g1 wq wk wv wo g2 w1 w2
const NT: usize = 3 + NL * PL; // emb + couches + gain final + tête (≈ 68 k paramètres)

type Seq = (Vec<usize>, usize); // (tokens, longueur du prompt)

struct Model {
    w: Vec<Vec<f32>>,
    shp: Vec<(usize, usize)>,
}

impl Model {
    fn new(rng: &mut Rng) -> Model {
        let mut shp = vec![(VOCAB, D)];
        for _ in 0..NL {
            shp.extend_from_slice(&[(1, D), (D, D), (D, D), (D, D), (D, D), (1, D), (D, FF), (FF, D)]);
        }
        shp.push((1, D));
        shp.push((D, VOCAB));
        let resid = 1.0 / ((2 * NL) as f32).sqrt();
        let mut w: Vec<Vec<f32>> = Vec::with_capacity(NT);
        for (i, &(r, c)) in shp.iter().enumerate() {
            let sd = if r == 1 {
                0.0
            } else if i == 0 {
                0.1
            } else if i == NT - 1 {
                0.05
            } else {
                let base = 1.0 / (r as f32).sqrt();
                if matches!((i - 1) % PL, 4 | 7) { base * resid } else { base }
            };
            w.push((0..r * c).map(|_| if r == 1 { 1.0 } else { sd * rng.normal() }).collect());
        }
        Model { w, shp }
    }

    fn count(&self) -> usize {
        self.w.iter().map(|v| v.len()).sum()
    }

    /// Nouvelle bande d'autodiff : ses NT premiers nœuds sont les paramètres (copiés).
    fn tape(&self) -> Tape {
        let mut tp = Tape::new();
        for (v, &(r, c)) in self.w.iter().zip(self.shp.iter()) {
            tp.leaf(v.clone(), r, c);
        }
        tp
    }
}

/// Passe avant causale ; renvoie le nœud des logits (T × VOCAB).
fn forward(tp: &mut Tape, toks: &[usize]) -> usize {
    let mut x = tp.embed(0, toks);
    for l in 0..NL {
        let b = 1 + l * PL;
        let h = tp.rms(x, b);
        let q = tp.matmul(h, b + 1);
        let k = tp.matmul(h, b + 2);
        let v = tp.matmul(h, b + 3);
        let q = tp.rope(q, NH);
        let k = tp.rope(k, NH);
        let a = tp.attn(q, k, v, NH);
        let o = tp.matmul(a, b + 4);
        x = tp.add(x, o);
        let h = tp.rms(x, b + 5);
        let u = tp.matmul(h, b + 6);
        let u = tp.silu(u);
        let o = tp.matmul(u, b + 7);
        x = tp.add(x, o);
    }
    let h = tp.rms(x, NT - 2);
    tp.matmul(h, NT - 1)
}

/// Perte = scale × Σ NLL sur les tokens de la cause (+ END). Avec `acc` : rétropropage et accumule.
fn seq_run(m: &Model, toks: &[usize], plen: usize, scale: f32, acc: Option<&mut Vec<Vec<f32>>>) -> f32 {
    let n = toks.len() - 1;
    let mut tp = m.tape();
    let lg = forward(&mut tp, &toks[..n]);
    let mask: Vec<bool> = (0..n).map(|i| i + 1 >= plen).collect();
    let loss = tp.xent(lg, toks[1..].to_vec(), mask, scale);
    let val = tp.val[loss][0];
    if let Some(a) = acc {
        tp.backward(loss);
        for (dst, src) in a.iter_mut().zip(tp.grad.iter()) {
            for (x, y) in dst.iter_mut().zip(src.iter()) {
                *x += *y;
            }
        }
    }
    val
}

// ───────────────────────── Entraînement parallèle ─────────────────────────
fn worker(m: &Model, seqs: &[Seq], scale: f32) -> (f32, Vec<Vec<f32>>) {
    let mut acc: Vec<Vec<f32>> = m.w.iter().map(|v| vec![0.0f32; v.len()]).collect();
    let mut loss = 0.0f32;
    for (t, p) in seqs {
        loss += seq_run(m, t, *p, scale, Some(&mut acc));
    }
    (loss, acc)
}

/// Un lot réparti sur `nthreads` threads ; renvoie (perte moyenne par token cible, gradients sommés).
fn train_step(m: &Model, batch: &[Seq], nthreads: usize) -> (f32, Vec<Vec<f32>>) {
    let ntgt: usize = batch.iter().map(|(t, p)| t.len() - *p).sum();
    let scale = 1.0 / ntgt as f32;
    let per = ((batch.len() + nthreads - 1) / nthreads).max(1);
    let mut loss = 0.0f32;
    let mut grads: Vec<Vec<f32>> = m.w.iter().map(|v| vec![0.0f32; v.len()]).collect();
    std::thread::scope(|sc| {
        let mut hs = Vec::new();
        for ch in batch.chunks(per) {
            hs.push(sc.spawn(move || worker(m, ch, scale)));
        }
        for h in hs {
            let (l, g) = h.join().expect("thread d'entraînement");
            loss += l;
            for (a, b) in grads.iter_mut().zip(g.iter()) {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
            }
        }
    });
    (loss, grads)
}

fn make_batch(rng: &mut Rng, size: usize, maxlv: usize) -> Vec<Seq> {
    (0..size)
        .map(|_| {
            let lv = 1 + rng.below(maxlv);
            let e = make_episode(rng, lv);
            (e.toks, e.plen)
        })
        .collect()
}

struct Adam {
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: i32,
}

impl Adam {
    fn new(w: &[Vec<f32>]) -> Adam {
        let z = || w.iter().map(|x| vec![0.0f32; x.len()]).collect::<Vec<_>>();
        Adam { m: z(), v: z(), t: 0 }
    }

    /// AdamW (décroissance découplée, sauf sur les gains RMSNorm).
    fn step(&mut self, w: &mut [Vec<f32>], g: &[Vec<f32>], shp: &[(usize, usize)], lr: f32, wd: f32) {
        self.t += 1;
        let (b1, b2) = (0.9f32, 0.98f32);
        let (c1, c2) = (1.0 - b1.powi(self.t), 1.0 - b2.powi(self.t));
        for i in 0..w.len() {
            let dec = if shp[i].0 == 1 { 0.0 } else { wd };
            for j in 0..w[i].len() {
                let gj = g[i][j];
                self.m[i][j] = b1 * self.m[i][j] + (1.0 - b1) * gj;
                self.v[i][j] = b2 * self.v[i][j] + (1.0 - b2) * gj * gj;
                let upd = (self.m[i][j] / c1) / ((self.v[i][j] / c2).sqrt() + 1e-8);
                w[i][j] -= lr * (upd + dec * w[i][j]);
            }
        }
    }
}

/// Échauffement linéaire puis cosinus jusqu'à 10 % du taux de base.
fn lr_at(step: usize, total: usize, base: f32) -> f32 {
    let warm = (total / 20).clamp(10, 200);
    if step < warm {
        return base * (step + 1) as f32 / warm as f32;
    }
    let pr = (step - warm) as f32 / total.saturating_sub(warm).max(1) as f32;
    base * (0.1 + 0.45 * (1.0 + (PI * pr).cos()))
}

// ───────────────────────── Inférence incrémentale (cache K/V) ─────────────────────────
#[derive(Clone)]
struct Cache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
}

impl Cache {
    fn new() -> Cache {
        Cache { k: vec![Vec::new(); NL], v: vec![Vec::new(); NL], len: 0 }
    }
}

fn rms_row(x: &[f32], g: &[f32]) -> Vec<f32> {
    let r = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + 1e-5).sqrt();
    x.iter().zip(g).map(|(a, b)| a * b / r).collect()
}

fn rope_row(x: &mut [f32], pos: usize) {
    let dh = D / NH;
    let half = dh / 2;
    for h in 0..NH {
        for c in 0..half {
            let freq = 10000f32.powf(-(c as f32) / half as f32);
            let (s, co) = (pos as f32 * freq).sin_cos();
            let (a, b) = (x[h * dh + c], x[h * dh + half + c]);
            x[h * dh + c] = a * co - b * s;
            x[h * dh + half + c] = a * s + b * co;
        }
    }
}

/// Consomme un token, met à jour le cache, renvoie les logits du token suivant.
fn infer_step(m: &Model, ca: &mut Cache, tok: usize) -> Vec<f32> {
    let pos = ca.len;
    let dh = D / NH;
    let sc = 1.0 / (dh as f32).sqrt();
    let mut x = m.w[0][tok * D..(tok + 1) * D].to_vec();
    for l in 0..NL {
        let b = 1 + l * PL;
        let h = rms_row(&x, &m.w[b]);
        let mut q = mm(&h, &m.w[b + 1], 1, D, D);
        let mut k = mm(&h, &m.w[b + 2], 1, D, D);
        let v = mm(&h, &m.w[b + 3], 1, D, D);
        rope_row(&mut q, pos);
        rope_row(&mut k, pos);
        ca.k[l].extend_from_slice(&k);
        ca.v[l].extend_from_slice(&v);
        let mut a = vec![0.0f32; D];
        for hd in 0..NH {
            let o = hd * dh;
            let mut p = vec![0.0f32; pos + 1];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..=pos {
                let kj = &ca.k[l][j * D + o..j * D + o + dh];
                let s = q[o..o + dh].iter().zip(kj).map(|(u, w)| u * w).sum::<f32>() * sc;
                p[j] = s;
                mx = mx.max(s);
            }
            let mut z = 0.0f32;
            for j in 0..=pos {
                p[j] = (p[j] - mx).exp();
                z += p[j];
            }
            for j in 0..=pos {
                let wgt = p[j] / z;
                for c in 0..dh {
                    a[o + c] += wgt * ca.v[l][j * D + o + c];
                }
            }
        }
        let ao = mm(&a, &m.w[b + 4], 1, D, D);
        for i in 0..D {
            x[i] += ao[i];
        }
        let h2 = rms_row(&x, &m.w[b + 5]);
        let mut u = mm(&h2, &m.w[b + 6], 1, D, FF);
        for z in u.iter_mut() {
            *z /= 1.0 + (-*z).exp();
        }
        let dn = mm(&u, &m.w[b + 7], 1, FF, D);
        for i in 0..D {
            x[i] += dn[i];
        }
    }
    ca.len += 1;
    let h = rms_row(&x, &m.w[NT - 2]);
    mm(&h, &m.w[NT - 1], 1, D, VOCAB)
}

fn prefill(m: &Model, prompt: &[usize]) -> (Cache, Vec<f32>) {
    let mut ca = Cache::new();
    let mut lg = Vec::new();
    for &t in prompt {
        lg = infer_step(m, &mut ca, t);
    }
    (ca, lg)
}

// ───────────────────────── Décodage contraint par grammaire ─────────────────────────
/// terme := SYM | SYM '(' terme^arité ')'. Taille d'arbre bornée par HMAX (comme les vraies causes).
struct Gram {
    stack: Vec<usize>, // enfants restant à commencer, par compound ouvert
    nodes: usize,
    lp: bool,
    done: bool,
}

impl Gram {
    fn new() -> Gram {
        Gram { stack: Vec::new(), nodes: 0, lp: false, done: false }
    }

    fn need(&self) -> usize {
        self.stack.iter().sum::<usize>() + if self.nodes == 0 { 1 } else { 0 }
    }

    fn allowed(&self, sig: &Sig) -> Vec<usize> {
        if self.done {
            return vec![T_END];
        }
        if self.lp {
            return vec![T_LP];
        }
        if self.stack.last() == Some(&0) {
            return vec![T_RP];
        }
        let need = self.need();
        sig.syms
            .iter()
            .filter(|&&s| sig.arity[s] == 0 || self.nodes + need + sig.arity[s] <= HMAX)
            .map(|&s| T_SYM + s)
            .collect()
    }

    fn push(&mut self, tok: usize, sig: &Sig) {
        if tok == T_LP {
            self.lp = false;
        } else if tok == T_RP {
            self.stack.pop();
            if self.stack.is_empty() {
                self.done = true;
            }
        } else if tok >= T_SYM {
            let a = sig.arity[tok - T_SYM];
            self.nodes += 1;
            if let Some(top) = self.stack.last_mut() {
                *top -= 1;
            }
            if a > 0 {
                self.stack.push(a);
                self.lp = true;
            } else if self.stack.is_empty() {
                self.done = true;
            }
        }
    }
}

fn pick(lg: &[f32], allowed: &[usize], temp: f32, uniform: bool, rng: &mut Rng) -> usize {
    if allowed.len() == 1 {
        return allowed[0];
    }
    if uniform {
        return allowed[rng.below(allowed.len())];
    }
    if temp <= 0.0 {
        let mut best = allowed[0];
        for &t in allowed {
            if lg[t] > lg[best] {
                best = t;
            }
        }
        return best;
    }
    let mx = allowed.iter().map(|&t| lg[t]).fold(f32::NEG_INFINITY, f32::max);
    let ws: Vec<f32> = allowed.iter().map(|&t| ((lg[t] - mx) / temp).exp()).collect();
    let mut u = rng.unif() * ws.iter().sum::<f32>();
    for (i, w) in ws.iter().enumerate() {
        if u < *w {
            return allowed[i];
        }
        u -= *w;
    }
    allowed[allowed.len() - 1]
}

/// temp = 0 : glouton ; temp = 1 : échantillon du postérieur appris ; `uniform` : référence sans réseau.
fn decode(m: &Model, base: &Cache, first: &[f32], sig: &Sig, temp: f32, uniform: bool, rng: &mut Rng) -> Vec<usize> {
    let mut ca = base.clone();
    let mut lg = first.to_vec();
    let mut g = Gram::new();
    let mut out = Vec::new();
    for _ in 0..MAXH {
        let tok = pick(&lg, &g.allowed(sig), temp, uniform, rng);
        out.push(tok);
        if tok == T_END {
            break;
        }
        g.push(tok, sig);
        if !uniform {
            lg = infer_step(m, &mut ca, tok);
        }
    }
    out
}

fn parse(ar: &mut Arena, sig: &Sig, t: &[usize], pos: &mut usize) -> Option<Id> {
    let tk = *t.get(*pos)?;
    if tk < T_SYM || tk >= T_SYM + NS || !sig.has[tk - T_SYM] {
        return None;
    }
    let s = tk - T_SYM;
    *pos += 1;
    let mut kids = Vec::new();
    if sig.arity[s] > 0 {
        if t.get(*pos) != Some(&T_LP) {
            return None;
        }
        *pos += 1;
        for _ in 0..sig.arity[s] {
            kids.push(parse(ar, sig, t, pos)?);
        }
        if t.get(*pos) != Some(&T_RP) {
            return None;
        }
        *pos += 1;
    }
    Some(ar.mk(s, kids))
}

fn parse_cause(ar: &mut Arena, sig: &Sig, out: &[usize]) -> Option<Id> {
    let mut pos = 0;
    let id = parse(ar, sig, out, &mut pos)?;
    if out.get(pos) == Some(&T_END) { Some(id) } else { None }
}

// ───────────────────────── Vérificateur exact ─────────────────────────
/// P(obs | cause) = 1/kmax · Σ_{k=1..kmax} P(X_k = obs | X_0 = cause), en propageant la loi de la chaîne
/// de Markov du PTGRS (états = racines du graphe hash-consé ; redex uniforme ; issue selon les probas
/// de la règle). Même rejet de taille (OMAX) que le générateur ; les états bloqués perdent leur masse.
fn likelihood(ar: &mut Arena, rules: &[Rule], h: Id, o: Id, kmax: usize) -> f64 {
    let mut cur: HashMap<Id, f64> = HashMap::new();
    cur.insert(h, 1.0);
    let mut total = 0.0f64;
    for _ in 0..kmax {
        let mut nxt: HashMap<Id, f64> = HashMap::new();
        for (&s, &p) in cur.iter() {
            let rx = ar.redexes(s, rules);
            if rx.is_empty() {
                continue;
            }
            let pr = p / rx.len() as f64;
            for (node, ri, b) in rx {
                for (q, rhs) in &rules[ri].outs {
                    let rep = ar.inst(rhs, &b);
                    let n = ar.replace(s, node, rep, &mut HashMap::new());
                    if ar.size(n) <= OMAX {
                        *nxt.entry(n).or_insert(0.0) += pr * *q as f64;
                    }
                }
            }
        }
        total += nxt.get(&o).copied().unwrap_or(0.0);
        cur = nxt;
    }
    total / kmax as f64
}

// ───────────────────────── Évaluation ─────────────────────────
#[derive(Default)]
struct Stats {
    n: usize,
    nll: f64,
    ntok: usize,
    g_valid: usize,
    g_exact: usize,
    s_valid: usize,
    s_any: usize,
    u_valid: usize,
    u_any: usize,
}

impl Stats {
    fn add(&mut self, o: &Stats) {
        self.n += o.n;
        self.nll += o.nll;
        self.ntok += o.ntok;
        self.g_valid += o.g_valid;
        self.g_exact += o.g_exact;
        self.s_valid += o.s_valid;
        self.s_any += o.s_any;
        self.u_valid += o.u_valid;
        self.u_any += o.u_any;
    }
}

fn pct(a: usize, b: usize) -> f64 {
    100.0 * a as f64 / b.max(1) as f64
}

/// « Valide » = la cause proposée explique l'observation avec probabilité > 0 en ≤ LMAX pas.
fn eval_chunk(m: &Model, eps: &mut [Episode], k: usize, seed: u64) -> Stats {
    let mut st = Stats::default();
    let mut rng = Rng::new(seed);
    for ep in eps.iter_mut() {
        st.n += 1;
        st.nll += seq_run(m, &ep.toks, ep.plen, 1.0, None) as f64;
        st.ntok += ep.toks.len() - ep.plen;
        let (base, first) = prefill(m, &ep.toks[..ep.plen]);
        let out = decode(m, &base, &first, &ep.sig, 0.0, false, &mut rng);
        if let Some(id) = parse_cause(&mut ep.ar, &ep.sig, &out) {
            if id == ep.h {
                st.g_exact += 1;
            }
            if likelihood(&mut ep.ar, &ep.rules, id, ep.o, LMAX) > 0.0 {
                st.g_valid += 1;
            }
        }
        for uni in [false, true] {
            let mut any = false;
            for _ in 0..k {
                let out = decode(m, &base, &first, &ep.sig, 1.0, uni, &mut rng);
                let ok = match parse_cause(&mut ep.ar, &ep.sig, &out) {
                    Some(id) => likelihood(&mut ep.ar, &ep.rules, id, ep.o, LMAX) > 0.0,
                    None => false,
                };
                any |= ok;
                if uni {
                    st.u_valid += ok as usize;
                } else {
                    st.s_valid += ok as usize;
                }
            }
            if uni {
                st.u_any += any as usize;
            } else {
                st.s_any += any as usize;
            }
        }
    }
    st
}

fn evaluate(m: &Model, n: usize, k: usize, nthreads: usize, seed: u64) {
    for level in 1..=LMAX {
        let mut rng = Rng::new(seed ^ (level as u64).wrapping_mul(7919));
        let mut eps: Vec<Episode> = (0..n).map(|_| make_episode(&mut rng, level)).collect();
        let per = ((n + nthreads - 1) / nthreads).max(1);
        let mut st = Stats::default();
        std::thread::scope(|sc| {
            let mut hs = Vec::new();
            for (i, ch) in eps.chunks_mut(per).enumerate() {
                let sd = seed.wrapping_add(1 + i as u64 + 97 * level as u64);
                hs.push(sc.spawn(move || eval_chunk(m, ch, k, sd)));
            }
            for h in hs {
                st.add(&h.join().expect("thread d'évaluation"));
            }
        });
        println!(
            "  niv {level} │ NLL/tok {:.3} │ glouton : {:.0}% valide, {:.0}% exact │ échantillons : {:.0}% valides, ≥1 sur {k} : {:.0}% │ uniforme : {:.0}% valides, ≥1 sur {k} : {:.0}%",
            st.nll / st.ntok.max(1) as f64,
            pct(st.g_valid, st.n), pct(st.g_exact, st.n),
            pct(st.s_valid, st.n * k), pct(st.s_any, st.n),
            pct(st.u_valid, st.n * k), pct(st.u_any, st.n)
        );
    }
}

// ───────────────────────── Vérifications numériques ─────────────────────────
/// Écart max entre les logits de la passe complète (autodiff) et ceux du chemin incrémental (cache K/V).
fn consistency(m: &Model, toks: &[usize]) -> f32 {
    let n = toks.len() - 1;
    let mut tp = m.tape();
    let lg = forward(&mut tp, &toks[..n]);
    let mut ca = Cache::new();
    let mut worst = 0.0f32;
    for i in 0..n {
        let row = infer_step(m, &mut ca, toks[i]);
        for c in 0..VOCAB {
            worst = worst.max((row[c] - tp.val[lg][i * VOCAB + c]).abs());
        }
    }
    worst
}

fn grad_check(seed: u64) {
    let mut rng = Rng::new(seed);
    let mut m = Model::new(&mut rng);
    let ep = make_episode(&mut rng, 2);
    let ntgt = ep.toks.len() - ep.plen;
    let scale = 1.0 / ntgt as f32;
    println!("check : {} paramètres, séquence de {} tokens ({} cibles)", m.count(), ep.toks.len(), ntgt);
    let mut acc: Vec<Vec<f32>> = m.w.iter().map(|v| vec![0.0f32; v.len()]).collect();
    let l0 = seq_run(&m, &ep.toks, ep.plen, scale, Some(&mut acc));
    println!("perte initiale {:.3} (loi uniforme : {:.3})", l0, (VOCAB as f32).ln());
    let eps = 2e-2f32;
    let (mut bad, mut tot, mut worst) = (0usize, 0usize, 0.0f32);
    for i in 0..m.w.len() {
        let mut idx = vec![0usize]; // plus gros gradient + 3 indices au hasard
        for j in 0..acc[i].len() {
            if acc[i][j].abs() > acc[i][idx[0]].abs() {
                idx[0] = j;
            }
        }
        for _ in 0..3 {
            idx.push(rng.below(acc[i].len()));
        }
        for &j in &idx {
            let o = m.w[i][j];
            m.w[i][j] = o + eps;
            let lp = seq_run(&m, &ep.toks, ep.plen, scale, None);
            m.w[i][j] = o - eps;
            let lm = seq_run(&m, &ep.toks, ep.plen, scale, None);
            m.w[i][j] = o;
            let num = (lp - lm) / (2.0 * eps);
            let ana = acc[i][j];
            let ratio = (num - ana).abs() / (3e-3 + 0.05 * num.abs().max(ana.abs()));
            worst = worst.max(ratio);
            tot += 1;
            if ratio > 1.0 {
                bad += 1;
                println!("  ✗ tenseur {i} idx {j} : analytique {ana:.5} vs numérique {num:.5}");
            }
        }
    }
    println!("différences finies : {bad}/{tot} hors tolérance (pire ratio {worst:.2}, < 1 attendu)");
    let diff = consistency(&m, &ep.toks);
    println!("cache K/V vs passe complète : écart max des logits {diff:.2e}");
    println!("{}", if bad == 0 && diff < 1e-3 { "OK" } else { "ATTENTION : voir ci-dessus" });
}

// ───────────────────────── Démo qualitative ─────────────────────────
fn demo(m: &Model, seed: u64, k: usize) {
    let mut rng = Rng::new(seed ^ 0xDE30);
    let mut ep = make_episode(&mut rng, 3);
    println!("— Exemple (niveau 3) —");
    let sg: Vec<String> = ep.sig.syms.iter().map(|&s| format!("s{}/{}", s, ep.sig.arity[s])).collect();
    println!("signature   : {}", sg.join(" "));
    for r in &ep.rules {
        let outs: Vec<String> = r.outs.iter().map(|(p, rhs)| format!("{:.2}:{}", p, show_pat(rhs))).collect();
        println!("règle       : {} -> {}", show_pat(&r.lhs), outs.join(" | "));
    }
    let truth = likelihood(&mut ep.ar, &ep.rules, ep.h, ep.o, ep.kmax);
    println!("observation : {}", ep.ar.show(ep.o));
    println!("cause vraie : {}   (P(obs|cause) = {:.4}, ≤ {} pas)", ep.ar.show(ep.h), truth, ep.kmax);
    let (base, first) = prefill(m, &ep.toks[..ep.plen]);
    for i in 0..=k {
        let temp = if i == 0 { 0.0 } else { 1.0 };
        let out = decode(m, &base, &first, &ep.sig, temp, false, &mut rng);
        match parse_cause(&mut ep.ar, &ep.sig, &out) {
            Some(id) => {
                let l = likelihood(&mut ep.ar, &ep.rules, id, ep.o, LMAX);
                let tag = if i == 0 { "glouton    " } else { "échantillon" };
                println!("{} : {}   P(obs|cause) = {:.4}", tag, ep.ar.show(id), l);
            }
            None => println!("(cause mal formée)"),
        }
    }
}

// ───────────────────────── main ─────────────────────────
fn arg<T: std::str::FromStr>(kv: &HashMap<String, String>, key: &str, def: T) -> T {
    kv.get(key).and_then(|v| v.parse().ok()).unwrap_or(def)
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn main() {
    let mut kv: HashMap<String, String> = HashMap::new();
    for a in env::args().skip(1) {
        if let Some((k, v)) = a.split_once('=') {
            kv.insert(k.to_string(), v.to_string());
        }
    }
    let seed: u64 = arg(&kv, "seed", 1);
    let threads: usize = arg::<usize>(&kv, "threads", default_threads()).max(1);
    if arg::<u32>(&kv, "check", 0) == 1 {
        grad_check(seed);
        return;
    }
    let steps: usize = arg(&kv, "steps", 4000);
    let bs: usize = arg(&kv, "batch", 32);
    let lr: f32 = arg(&kv, "lr", 2e-3);
    let wd: f32 = arg(&kv, "wd", 0.01);
    let clip: f32 = arg(&kv, "clip", 1.0);
    let log_every: usize = arg::<usize>(&kv, "log", 50).max(1);
    let eval_every: usize = arg(&kv, "eval_every", 1000);
    let eval_n: usize = arg(&kv, "eval_n", 32);
    let ks: usize = arg(&kv, "k", 8);
    let curriculum: u32 = arg(&kv, "curriculum", 1);

    let mut rng = Rng::new(seed);
    let mut m = Model::new(&mut rng);
    let mut adam = Adam::new(&m.w);
    println!(
        "PTGRS-Abducer : {} paramètres | {} threads | lot {} | {} pas | lr {:.1e}",
        m.count(), threads, bs, steps, lr
    );
    let t0 = std::time::Instant::now();
    let ramp = (steps * 3 / 5).max(1); // curriculum : un niveau de plus tous les ~15 % du budget
    for step in 0..steps {
        let maxlv = if curriculum == 1 { (1 + step * LMAX / ramp).min(LMAX) } else { LMAX };
        let batch = make_batch(&mut rng, bs, maxlv);
        let (loss, mut g) = train_step(&m, &batch, threads);
        let gn = g.iter().map(|v| v.iter().map(|x| x * x).sum::<f32>()).sum::<f32>().sqrt();
        if !loss.is_finite() || !gn.is_finite() {
            eprintln!("divergence au pas {step} (perte {loss}) : essayez un lr plus petit");
            return;
        }
        if gn > clip {
            for v in g.iter_mut() {
                for x in v.iter_mut() {
                    *x *= clip / gn;
                }
            }
        }
        let lr_t = lr_at(step, steps, lr);
        adam.step(&mut m.w, &g, &m.shp, lr_t, wd);
        if step == 0 || (step + 1) % log_every == 0 {
            println!(
                "pas {:>5}/{} │ niv≤{} │ perte {:.4} │ ‖g‖ {:.2} │ lr {:.1e} │ {:.0}s",
                step + 1, steps, maxlv, loss, gn, lr_t, t0.elapsed().as_secs_f32()
            );
        }
        if eval_every > 0 && (step + 1) % eval_every == 0 && step + 1 < steps {
            evaluate(&m, eval_n, ks, threads, seed ^ 0xE5A1);
        }
    }
    println!("— évaluation finale —");
    evaluate(&m, eval_n.max(64), ks, threads, seed ^ 0xE5A1);
    demo(&m, seed, ks);
}