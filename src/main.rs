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

fn main() {}