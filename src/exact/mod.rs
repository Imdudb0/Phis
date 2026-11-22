//! Part 5b — exact reference for phi_sharp on small graphs.
//!
//! 5b-1 mod.rs    config, tractability, exact marginals, exact phi_s, exact complexes
//! 5b-2 dx.rs     exact distinctions, exact residual
//! 5b-3 relx.rs   exact relations
//! 5b-4 run.rs    exact structure, scan, Analysis, score, render
//! 5b-5 calib.rs  estimator vs exact (draws R, sampled cuts, relations)

mod dx;
pub use dx::{residuals_exact, Dx};
mod relx;
pub use relx::{gap, relations_exact, RelGap, RelsX};
// mod run;    // 5b-4
// mod calib;  // 5b-5

use crate::model::Model;
use crate::percept::Percept;
use crate::phi::{ipow, PHI_EPS};
use crate::sim::Ctx;
use crate::sys::{split_mask, System};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

const REL_TOL: f64 = 1e-12;
const TIE_REL: f64 = 1e-9;

/// Where the repertoires come from.
/// `Exact`: `Ctx::marg_exact` (full enumeration of the joint states).
/// `Mc`: the estimator's own Monte-Carlo `Ctx::marg`; everything else stays exhaustive.
/// Exact vs Mc isolates the sampling noise; Mc vs estimator isolates the structural
/// restrictions (neighbourhoods, sampled cuts, pair+closure relations).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Exact,
    Mc,
}

#[derive(Clone, Debug)]
pub struct ExactCfg {
    pub backend: Backend,
    /// largest mechanism (0 = every node of the complex)
    pub kmax: usize,
    /// largest purview (0 = every node of the complex)
    pub qmax: usize,
    pub max_states: usize,
    pub max_mech: usize,
    /// refuse when kmax + qmax exceeds this (Bell number of the partitions)
    pub max_part: usize,
    pub grain_max: usize,
    pub max_cx: usize,
    pub max_nodes: usize,
    /// refuse when |A(E)|^n exceeds this (exact backend)
    pub max_joint: usize,
    /// largest relation order (0 = every subset of distinctions)
    pub rel_order: usize,
    /// cap on enumerated relations; beyond it the result is flagged incomplete
    pub max_rel: usize,
    /// Monte-Carlo draws R (only used by Backend::Mc)
    pub draws: Option<usize>,
}

impl ExactCfg {
    /// Same kmax / qmax / grain / max_cx as the estimator, but exhaustive otherwise.
    pub fn from_model(m: &Model) -> ExactCfg {
        ExactCfg {
            backend: Backend::Exact,
            kmax: m.hy.kmax.clamp(1, 3),
            qmax: m.hy.qmax.clamp(1, 3),
            max_states: 8192,
            max_mech: 4096,
            max_part: 9,
            grain_max: m.hy.grain_max.max(1),
            max_cx: m.hy.max_cx.max(1),
            max_nodes: 6,
            max_joint: 2048,
            rel_order: 0,
            max_rel: 200_000,
            draws: None,
        }
    }
    /// Every mechanism and every purview (needs n <= 4 with the default max_part).
    pub fn full(m: &Model) -> ExactCfg {
        let mut c = ExactCfg::from_model(m);
        c.kmax = 0;
        c.qmax = 0;
        c
    }
    pub fn with_backend(mut self, b: Backend) -> ExactCfg {
        self.backend = b;
        self
    }
    pub fn with_rel_order(mut self, k: usize) -> ExactCfg {
        self.rel_order = k;
        self
    }
    pub fn with_draws(mut self, r: usize) -> ExactCfg {
        self.draws = Some(r);
        self
    }
}

/// Can the exact reference run on this (model, E)?
pub fn tractable(m: &Model, e: &Percept, cfg: &ExactCfg) -> Result<(), String> {
    let n = m.n();
    let top = cfg.max_nodes.min(30);
    if n > top {
        return Err(format!("{} nodes: above max_nodes = {}", n, top));
    }
    let kk = if cfg.kmax == 0 { n } else { cfg.kmax.min(n) };
    let qq = if cfg.qmax == 0 { n } else { cfg.qmax.min(n) };
    if kk + qq > cfg.max_part {
        return Err(format!(
            "partitions of {}+{} elements: above max_part = {}",
            kk, qq, cfg.max_part
        ));
    }
    if cfg.backend == Backend::Exact {
        let a = e.a.max(1);
        let joint = ipow(a, n);
        let lim = if cfg.grain_max > 1 { 2_048 } else { 4_000_000 };
        let lim = lim.min(cfg.max_joint);
        if joint > lim {
            return Err(format!(
                "|A(E)|^n = {}^{} = {}: above the exact limit {}",
                a, n, joint, lim
            ));
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ helpers

fn comb_rec(nodes: &[usize], start: usize, size: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    if cur.len() == size {
        out.push(cur.clone());
        return;
    }
    for i in start..nodes.len() {
        if nodes.len() - i < size - cur.len() {
            break;
        }
        cur.push(nodes[i]);
        comb_rec(nodes, i + 1, size, cur, out);
        cur.pop();
    }
}

/// Every subset of `nodes` with min_size..=max_size elements, by size then lexicographically.
pub fn subsets(nodes: &[usize], min_size: usize, max_size: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for size in min_size.max(1)..=max_size.min(nodes.len()) {
        comb_rec(nodes, 0, size, &mut cur, &mut out);
    }
    out
}

fn norm(v: &[usize]) -> Vec<usize> {
    let mut x = v.to_vec();
    x.sort_unstable();
    x.dedup();
    x
}

// ---------------------------------------------------------------- marginals

/// Cached marginal of one node given a clamped subset, under the Ctx background.
/// Lives only as long as the borrow of the Ctx, so the background cannot change under it.
struct Mg<'c, 'm> {
    cx: &'c Ctx<'m>,
    be: Backend,
    tau: usize,
    cache: RefCell<HashMap<Vec<usize>, Rc<Vec<f64>>>>,
    bad: Cell<bool>,
}

impl<'c, 'm> Mg<'c, 'm> {
    fn new(cx: &'c Ctx<'m>, be: Backend, tau: usize) -> Mg<'c, 'm> {
        Mg { cx, be, tau: tau.max(1), cache: RefCell::new(HashMap::new()), bad: Cell::new(false) }
    }

    fn get(&self, j: usize, clamp: &[usize], syms: &[u16]) -> Rc<Vec<f64>> {
        let mut pr: Vec<(usize, u16)> = clamp.iter().copied().zip(syms.iter().copied()).collect();
        pr.sort_unstable();
        let mut key: Vec<usize> = Vec::with_capacity(2 + 2 * pr.len());
        key.push(j);
        key.push(self.tau);
        for &(c, s) in &pr {
            key.push(c);
            key.push(s as usize);
        }
        let hit = self.cache.borrow().get(&key).cloned();
        if let Some(v) = hit {
            return v;
        }
        let cl: Vec<usize> = pr.iter().map(|p| p.0).collect();
        let sy: Vec<u16> = pr.iter().map(|p| p.1).collect();
        let mut out = vec![0.0f64; self.cx.a];
        let done = match self.be {
            Backend::Exact => self.cx.marg_exact(j, &cl, self.tau, &sy, &mut out),
            Backend::Mc => false,
        };
        if !done {
            if self.be == Backend::Exact {
                self.bad.set(true);
            }
            self.cx.marg(j, &cl, self.tau, &sy, &mut out);
        }
        let rc = Rc::new(out);
        self.cache.borrow_mut().insert(key, Rc::clone(&rc));
        rc
    }
}

// ------------------------------------------------------------ exact phi_s

#[derive(Clone, Copy, Debug)]
pub struct SysX {
    pub phi: f64,
    pub src: u32,
    /// false: some repertoire fell back to Monte-Carlo
    pub ok: bool,
}

fn all_vals(cx: &Ctx<'_>, y: usize) -> Vec<u16> {
    match cx.e_sym(y) {
        Some(s) => vec![s],
        None => (0..cx.a as u16).collect(),
    }
}

/// max over the product of allowed states of prod p * |sum r|, by full enumeration.
fn max_prod_all(p: &[Rc<Vec<f64>>], r: &[Vec<f64>], vals: &[Vec<u16>]) -> f64 {
    let d = p.len();
    if d == 0 {
        return 0.0;
    }
    let mut idx = vec![0usize; d];
    let mut best = 0.0f64;
    loop {
        let mut pr = 1.0f64;
        let mut s = 0.0f64;
        for y in 0..d {
            let v = vals[y][idx[y]] as usize;
            pr *= p[y][v];
            s += r[y][v];
        }
        let val = pr * s.abs();
        if val > best {
            best = val;
        }
        let mut y = 0usize;
        while y < d {
            idx[y] += 1;
            if idx[y] < vals[y].len() {
                break;
            }
            idx[y] = 0;
            y += 1;
        }
        if y == d {
            break;
        }
    }
    best
}

/// Value of the directed cut "bit i of mask = 1 puts k[i] in the source side A":
/// connections A -> B are replaced by noise, so only nodes of B change.
fn cut_value_exact(
    mg: &Mg<'_, '_>,
    k: &[usize],
    sk: &[u16],
    p: &[Rc<Vec<f64>>],
    vals: &[Vec<u16>],
    pmax: &[f64],
    mask: u32,
) -> f64 {
    let m = k.len();
    let a = mg.cx.a;
    let mut fa = 1.0f64;
    let mut bpos: Vec<usize> = Vec::new();
    for i in 0..m {
        if ((mask >> i) & 1) == 1 {
            fa *= pmax[i];
        } else {
            bpos.push(i);
        }
    }
    let bnodes: Vec<usize> = bpos.iter().map(|&i| k[i]).collect();
    let bsyms: Vec<u16> = bpos.iter().map(|&i| sk[i]).collect();
    let mut pb: Vec<Rc<Vec<f64>>> = Vec::with_capacity(bpos.len());
    let mut rb: Vec<Vec<f64>> = Vec::with_capacity(bpos.len());
    let mut vb: Vec<Vec<u16>> = Vec::with_capacity(bpos.len());
    let mut changed = false;
    for &i in &bpos {
        let q = mg.get(k[i], &bnodes, &bsyms);
        let mut r = vec![0.0f64; a];
        for v in 0..a {
            r[v] = (p[i][v] / q[v]).log2();
            if r[v].abs() > 1e-12 {
                changed = true;
            }
        }
        pb.push(Rc::clone(&p[i]));
        rb.push(r);
        vb.push(vals[i].clone());
    }
    if changed { fa * max_prod_all(&pb, &rb, &vb) } else { 0.0 }
}

/// Exact integrated information of the candidate system `k` at grain `tau`:
/// min over EVERY directed bipartition, max over EVERY state of the system
/// (anchored nodes at E's symbol). Same definition as sys.rs, nothing sampled.
pub fn system_phi_exact(cx: &mut Ctx<'_>, be: Backend, state: &[u16], tau: usize, k: &[usize]) -> SysX {
    let m = k.len();
    if m < 2 {
        return SysX { phi: 0.0, src: 0, ok: true };
    }
    assert!(m <= 30, "a candidate system has at most 30 nodes");
    let n = cx.n;
    let mut inside = vec![false; n];
    for &j in k {
        inside[j] = true;
    }
    let mut bn: Vec<usize> = Vec::new();
    let mut bs: Vec<u16> = Vec::new();
    for j in 0..n {
        if !inside[j] {
            bn.push(j);
            bs.push(state[j]);
        }
    }
    cx.set_background(&bn, &bs);
    let cx: &Ctx<'_> = &*cx;
    let a = cx.a;
    let amax = (a - 1) as u16;
    let mg = Mg::new(cx, be, tau);
    let sk: Vec<u16> = k.iter().map(|&j| state[j].min(amax)).collect();
    let mut p: Vec<Rc<Vec<f64>>> = Vec::with_capacity(m);
    let mut vals: Vec<Vec<u16>> = Vec::with_capacity(m);
    let mut pmax: Vec<f64> = Vec::with_capacity(m);
    for &y in k {
        let py = mg.get(y, k, &sk);
        let vs = all_vals(cx, y);
        let mut mx = 0.0f64;
        for &v in &vs {
            if py[v as usize] > mx {
                mx = py[v as usize];
            }
        }
        p.push(py);
        vals.push(vs);
        pmax.push(mx);
    }
    let full: u32 = (1u32 << m) - 1;
    let mut best = f64::INFINITY;
    let mut bm = 0u32;
    for mask in 1..full {
        let v = cut_value_exact(&mg, k, &sk, &p, &vals, &pmax, mask);
        if v < best {
            best = v;
            bm = mask;
        }
        if best <= PHI_EPS {
            break;
        }
    }
    let phi = if best.is_finite() { best.max(0.0) } else { 0.0 };
    SysX { phi, src: bm, ok: !mg.bad.get() }
}

// ----------------------------------------------------------- exact complexes

#[derive(Clone, Debug)]
pub struct FoundX {
    pub tau: usize,
    pub systems: Vec<System>,
    pub ok: bool,
}

fn best_system_exact(
    cx: &mut Ctx<'_>,
    cfg: &ExactCfg,
    state: &[u16],
    tau: usize,
    rem: &[usize],
    ok: &mut bool,
) -> Option<(Vec<usize>, f64, u32)> {
    if rem.len() < 2 {
        return None;
    }
    let mut best: Option<(Vec<usize>, f64, u32)> = None;
    for k in subsets(rem, 2, rem.len()) {
        let s = system_phi_exact(cx, cfg.backend, state, tau, &k);
        if !s.ok {
            *ok = false;
        }
        let better = match &best {
            None => true,
            Some(b) => s.phi > b.1 * (1.0 + REL_TOL),
        };
        if better {
            best = Some((k, s.phi, s.src));
        }
    }
    match best {
        Some(b) if b.1 > PHI_EPS => Some(b),
        _ => None,
    }
}

/// Exclusion, recursively: grain tau* = the one whose best system has the largest phi_s
/// (smaller grain wins ties), then the best system among the remaining nodes, and so on.
pub fn find_complexes_exact(cx: &mut Ctx<'_>, state: &[u16], cfg: &ExactCfg) -> FoundX {
    let n = cx.n;
    let all: Vec<usize> = (0..n).collect();
    let mut ok = true;
    let mut tau_star = 1usize;
    let mut first: Option<(Vec<usize>, f64, u32)> = None;
    for tau in 1..=cfg.grain_max.max(1) {
        if let Some(c) = best_system_exact(cx, cfg, state, tau, &all, &mut ok) {
            let better = match &first {
                None => true,
                Some(f) => c.1 > f.1 * (1.0 + REL_TOL),
            };
            if better {
                tau_star = tau;
                first = Some(c);
            }
        }
    }
    let mut chosen: Vec<(Vec<usize>, f64, u32)> = Vec::new();
    if let Some(f) = first {
        let mut remaining: Vec<usize> = all.iter().copied().filter(|j| !f.0.contains(j)).collect();
        chosen.push(f);
        while chosen.len() < cfg.max_cx.max(1) && remaining.len() >= 2 {
            match best_system_exact(cx, cfg, state, tau_star, &remaining, &mut ok) {
                Some(c) => {
                    remaining.retain(|j| !c.0.contains(j));
                    chosen.push(c);
                }
                None => break,
            }
        }
    }
    let systems: Vec<System> = chosen
        .into_iter()
        .map(|(k, phi, src)| {
            let (a, b) = split_mask(&k, src);
            System { nodes: k, phi_s: phi, cut_src: a, cut_dst: b, exact: true }
        })
        .collect();
    FoundX { tau: tau_star, systems, ok }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Hyper;
    use crate::sys::{system_phi, SharpCfg};
    use crate::util::Rng;

    fn small(seed: u64, np: usize, nl: usize) -> Model {
        let hy = Hyper { budget: 4, grain_max: 1, ..Hyper::default() };
        Model::random(np, nl, hy, &mut Rng::new(seed))
    }

    #[test]
    fn exhaustive_cuts_reproduce_the_estimator_when_it_enumerates() {
        let m = small(11, 3, 2);
        let e = Percept::from_bytes(b"abb");
        let mut cx = Ctx::new(&m, &e, 0);
        let state = cx.initial_state();
        let cfg = SharpCfg::from_model(&m);
        for k in subsets(&[0, 1, 2, 3, 4], 2, 5) {
            let est = system_phi(&mut cx, &state, 1, &cfg, &k).phi;
            let exh = system_phi_exact(&mut cx, Backend::Mc, &state, 1, &k).phi;
            assert!((est - exh).abs() <= 1e-9 * est.max(1.0), "{:?}: {} vs {}", k, est, exh);
        }
    }

    #[test]
    fn acyclic_graph_has_no_complex() {
        let mut m = small(5, 3, 2);
        let n = m.n();
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    m.w[i * n + j] = 0.0;
                }
            }
        }
        let e = Percept::from_bytes(b"aba");
        let mut cx = Ctx::new(&m, &e, 0);
        let state = cx.initial_state();
        let f = find_complexes_exact(&mut cx, &state, &ExactCfg::from_model(&m));
        assert!(f.systems.is_empty());
        assert!(f.ok);
    }
}