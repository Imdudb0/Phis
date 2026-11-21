//! phi.rs — mechanism level of phi_sharp: repertoires, partitions, purview selection and
//! anchored distinctions. Part 3 of 6; complexes, phi_s, relations, the residual and
//! phi_sharp(E) are in sharp.rs (part 4).
//!
//! A distinction is d = (m ; Z_c ; Z_e). Everything is read from W through `Ctx::marg`
//! (the "TPM"), with node states in A(E):
//!
//!   effect repertoire  pi_e(z) = prod_{y in Z} marg(y, m, tau, s_m)[z_y]
//!   cause repertoire   pi_c(z) = L(z) / sum_z' L(z'),   L(z) = prod_{j in m} marg(j, Z, tau, z)[s_j]
//!                      (Bayesian inversion, flat prior on A(E)^|Z|)
//!   partition psi      a set partition of the elements of m (at t) and of Z (at t+tau for the
//!                      effect, t-tau for the cause) into at least two blocks (IIT 4.0 mechanism
//!                      partitions, Bell(|m|+|Z|) - 1 of them, all enumerated). Connections inside
//!                      a block are kept; connections between blocks are cut in the direction of
//!                      the repertoire (m -> Z for the effect, Z -> m for the cause) and replaced
//!                      by uniform noise on A(E):
//!                        pi_e^psi(z) = prod_y marg(y, m_block(y), tau, s_m_block(y))[z_y]
//!                        pi_c^psi(z) = prod_blocks pi_c(z_Zb | m_b)  (uniform if m_b is empty,
//!                                                                     factor 1 if Z_b is empty)
//!   ID(p, q) = max_z p(z) |log2 p(z)/q(z)|        phi_x(m, Z) = min_psi ID(pi_x, pi_x^psi)
//!
//! - E-anchoring (first assumed deviation from IIT 4.0): on the effect side, the coordinates of
//!   Z that carry a byte of E are evaluated AT E, not at the argmax; latent coordinates keep the
//!   argmax. The cause side keeps the argmax: the cause state is what abduction infers.
//! - Specified states are read at the MIP (the minimising partition), ties kept.
//! - Exclusion: Z_c and Z_e are selected separately (argmax of phi_c, phi_e over the candidate
//!   purviews, ties kept up to MAX_TIES); phi_d = min(phi_c, phi_e); reducible means 0.
//! - Support (second assumed deviation): `Dist::anchored` is true when a purview of d touches a
//!   byte of E. Only anchored distinctions belong to Dist_E.
//! - Estimator: mechanisms are the connected motifs (size <= kmax) of the capped neighbourhood
//!   graph; purviews are subsets (size <= qmax, |A(E)|^|Z| <= max_states) of the directed
//!   neighbourhood within tau hops. Mechanism partitions are enumerated exactly (a min over a
//!   subset of cuts would overestimate phi); the approximate cuts are for phi_s (part 4).
//!
//! A `Phi` is bound to one background (the system is `cx.free_nodes()`; call
//! `cx.set_background` BEFORE building it, the borrow checker then keeps it fixed), one state
//! s_t and one grain tau. It is not Send/Sync: build one per thread.

use crate::model::Model;
use crate::sim::{Ctx, IdBuild};
use crate::util::{hash_usize, show_byte};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Below this a phi is zero: the mechanism / purview is reducible. Exact cuts of absent
/// connections give exact zeros, so this only absorbs rounding.
pub const PHI_EPS: f64 = 1e-9;
/// Relative tolerance of ties ("ex aequo kept").
const TIE_REL: f64 = 1e-9;
/// Maximum number of tied purviews, and of tied specified states, kept per side.
pub const MAX_TIES: usize = 4;
const TINY: f64 = 1e-300;
/// Cap of the cause-table cache, in f64 entries (about 32 MB).
const CACHE_CAP: usize = 4_000_000;
/// Separator inside cache keys (node ids are far below it).
const SEP: usize = usize::MAX / 2;
const TAG_CTAB: u64 = 0x4354_4142_0000_0007;

// ---------- small numerics ----------

/// a^k, saturating.
#[inline]
pub fn ipow(a: usize, k: usize) -> usize {
    let mut r = 1usize;
    for _ in 0..k {
        r = r.saturating_mul(a);
    }
    r
}

/// One term of the intrinsic difference: p * |log2(p/q)|.
#[inline]
pub fn id_term(p: f64, q: f64) -> f64 {
    p * (p.max(TINY) / q.max(TINY)).log2().abs()
}

/// ID(p, q) = max_z p(z) |log2 p(z)/q(z)| over all indices; returns (value, argmax).
pub fn id_max(p: &[f64], q: &[f64]) -> (f64, usize) {
    let mut best = 0.0f64;
    let mut arg = 0usize;
    for z in 0..p.len().min(q.len()) {
        let v = id_term(p[z], q[z]);
        if v > best {
            best = v;
            arg = z;
        }
    }
    (best, arg)
}

/// Digits of `idx` in base `a` (least significant first), written into `out`.
pub fn decode(mut idx: usize, a: usize, out: &mut [u16]) {
    for o in out.iter_mut() {
        *o = (idx % a) as u16;
        idx /= a;
    }
}

/// Index, inside the sub-purview made of the positions `pos` (increasing), of the joint state `idx`.
#[inline]
fn sub_index(idx: usize, a: usize, pos: &[usize], apow: &[usize]) -> usize {
    let mut s = 0usize;
    let mut mul = 1usize;
    for &p in pos {
        s += ((idx / apow[p]) % a) * mul;
        mul *= a;
    }
    s
}

/// All set partitions of km + kz elements (mechanism elements first, then purview elements)
/// into at least two blocks, as restricted-growth label strings. Count: Bell(km + kz) - 1.
pub fn set_partitions(km: usize, kz: usize) -> Vec<Vec<u8>> {
    fn rec(i: usize, mx: u8, lab: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
        if i == lab.len() {
            if mx >= 1 {
                out.push(lab.clone());
            }
            return;
        }
        for l in 0..=(mx + 1) {
            lab[i] = l;
            rec(i + 1, mx.max(l), lab, out);
        }
    }
    let n = km + kz;
    let mut out: Vec<Vec<u8>> = Vec::new();
    if n < 2 {
        return out;
    }
    let mut lab = vec![0u8; n];
    rec(1, 0, &mut lab, &mut out);
    out
}

fn combos(u: &[usize], size: usize, start: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    if cur.len() == size {
        out.push(cur.clone());
        return;
    }
    for i in start..u.len() {
        if u.len() - i < size - cur.len() {
            break;
        }
        cur.push(u[i]);
        combos(u, size, i + 1, cur, out);
        cur.pop();
    }
}

/// Ids sorted by decreasing weight (ties by id), at most `cap` of them.
fn top_ids(mut v: Vec<(usize, f64)>, cap: usize) -> Vec<usize> {
    v.sort_by(|x, y| {
        y.1.partial_cmp(&x.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.0.cmp(&y.0))
    });
    v.truncate(cap);
    v.iter().map(|p| p.0).collect()
}

fn norm(v: &[usize]) -> Vec<usize> {
    let mut x = v.to_vec();
    x.sort_unstable();
    x.dedup();
    x
}

fn join_ids(v: &[usize]) -> String {
    v.iter().map(|j| j.to_string()).collect::<Vec<String>>().join(",")
}

// ---------- types ----------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Cause,
    Effect,
}

/// Estimator budgets of the mechanism level.
#[derive(Clone, Debug)]
pub struct PhiCfg {
    /// max mechanism size (connected motifs), clamped to 1..=3
    pub kmax: usize,
    /// max purview size, clamped to 1..=3
    pub qmax: usize,
    /// directed-neighbourhood cap per node
    pub nbr: usize,
    /// purviews with |A(E)|^|Z| above this are not proposed (their states are enumerated)
    pub max_states: usize,
    /// cap on the number of candidate mechanisms
    pub max_mech: usize,
}

impl PhiCfg {
    pub fn from_model(m: &Model) -> PhiCfg {
        PhiCfg {
            kmax: m.hy.kmax.clamp(1, 3),
            qmax: m.hy.qmax.clamp(1, 3),
            nbr: m.hy.nbr.max(1),
            max_states: 8192,
            max_mech: 2048,
        }
    }
}

/// One maximally irreducible purview of a mechanism, with its specified state(s).
#[derive(Clone, Debug)]
pub struct Face {
    /// purview nodes, increasing ids
    pub nodes: Vec<usize>,
    /// phi_c or phi_e of this purview
    pub phi: f64,
    /// specified state(s) read at the MIP: symbols of A(E) aligned with `nodes` (ties kept)
    pub states: Vec<Vec<u16>>,
}

/// A mechanism with its maximally irreducible cause and effect. Mechanisms with only one
/// irreducible side are returned too (phi_d = 0): the residual of part 4 reads what the min
/// throws away. They never enter the sum of phi_d.
#[derive(Clone, Debug)]
pub struct Dist {
    /// mechanism nodes, increasing ids
    pub mech: Vec<usize>,
    /// symbols of the mechanism nodes in s_t
    pub mech_state: Vec<u16>,
    /// maximally irreducible cause purviews (empty when phi_c = 0); several only for ties
    pub cause: Vec<Face>,
    /// maximally irreducible effect purviews (empty when phi_e = 0); several only for ties
    pub effect: Vec<Face>,
    pub phi_c: f64,
    pub phi_e: f64,
    /// min(phi_c, phi_e)
    pub phi_d: f64,
    /// a purview of d touches a byte of E
    pub anchored: bool,
}

impl Dist {
    pub fn is_distinction(&self) -> bool {
        self.phi_d > PHI_EPS
    }
}

/// Sum of phi_d over the anchored distinctions: the first term of phi_sharp for one complex.
pub fn sum_phi_d(ds: &[Dist]) -> f64 {
    ds.iter()
        .filter(|d| d.anchored && d.is_distinction())
        .map(|d| d.phi_d)
        .sum()
}

struct SideEval {
    phi: f64,
    states: Vec<Vec<u16>>,
    /// false when the evaluation stopped early because the running minimum fell below the floor
    exact: bool,
}

// ---------- the evaluator ----------

pub struct Phi<'c, 'm> {
    pub cx: &'c Ctx<'m>,
    /// grain: micro-steps per macro-step
    pub tau: usize,
    /// full state s_t of the graph, one symbol of A(E) per node
    pub state: Vec<u16>,
    pub cfg: PhiCfg,
    /// nin[j]: strongest free sources of j; nout[i]: strongest free targets of i
    nin: Vec<Vec<usize>>,
    nout: Vec<Vec<usize>>,
    /// cause tables keyed by (mechanism block, purview block); state, tau and background are fixed
    ctab: RefCell<HashMap<u64, Rc<Vec<f64>>, IdBuild>>,
    ctab_sz: Cell<usize>,
}

impl<'c, 'm> Phi<'c, 'm> {
    pub fn new(cx: &'c Ctx<'m>, tau: usize, state: &[u16], cfg: PhiCfg) -> Phi<'c, 'm> {
        let n = cx.n;
        assert!(state.len() >= n, "state must hold one symbol per node");
        let amax = (cx.a - 1) as u16;
        let st: Vec<u16> = state[..n].iter().map(|&s| s.min(amax)).collect();
        let mut nin: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut nout: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &j in cx.free_nodes() {
            let mut vi: Vec<(usize, f64)> = Vec::new();
            let mut vo: Vec<(usize, f64)> = Vec::new();
            for &i in cx.free_nodes() {
                let wi = cx.weight(i, j).abs(); // i -> j
                if wi > 0.0 {
                    vi.push((i, wi));
                }
                let wo = cx.weight(j, i).abs(); // j -> i
                if wo > 0.0 {
                    vo.push((i, wo));
                }
            }
            nin[j] = top_ids(vi, cfg.nbr);
            nout[j] = top_ids(vo, cfg.nbr);
        }
        Phi {
            cx,
            tau: tau.max(1),
            state: st,
            cfg,
            nin,
            nout,
            ctab: RefCell::new(HashMap::default()),
            ctab_sz: Cell::new(0),
        }
    }

    /// `new` with the budgets stored in the model.
    pub fn standard(cx: &'c Ctx<'m>, tau: usize, state: &[u16]) -> Phi<'c, 'm> {
        Phi::new(cx, tau, state, PhiCfg::from_model(cx.m))
    }

    // ---------- candidates ----------

    /// Connected motifs (size <= kmax) of the capped undirected neighbourhood graph of the
    /// free nodes, by increasing size then lexicographic order.
    pub fn mechanisms(&self) -> Vec<Vec<usize>> {
        let n = self.cx.n;
        let free = self.cx.free_nodes();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &i in free {
            for &j in self.nin[i].iter().chain(self.nout[i].iter()) {
                if i != j {
                    if !adj[i].contains(&j) {
                        adj[i].push(j);
                    }
                    if !adj[j].contains(&i) {
                        adj[j].push(i);
                    }
                }
            }
        }
        let mut all: Vec<Vec<usize>> = free.iter().map(|&i| vec![i]).collect();
        let mut layer: Vec<Vec<usize>> = all.clone();
        for _ in 2..=self.cfg.kmax {
            let mut seen: HashSet<Vec<usize>> = HashSet::new();
            let mut next: Vec<Vec<usize>> = Vec::new();
            for ms in &layer {
                for &x in ms {
                    for &y in &adj[x] {
                        if ms.contains(&y) {
                            continue;
                        }
                        let mut v = ms.clone();
                        v.push(y);
                        v.sort_unstable();
                        if seen.insert(v.clone()) {
                            next.push(v);
                        }
                    }
                }
            }
            next.sort();
            all.extend(next.iter().cloned());
            layer = next;
            if layer.is_empty() || all.len() >= self.cfg.max_mech {
                break;
            }
        }
        all.sort_by(|x, y| x.len().cmp(&y.len()).then_with(|| x.cmp(y)));
        all.truncate(self.cfg.max_mech);
        all
    }

    /// Directed neighbourhood of the mechanism `m` within tau hops (sources for the cause,
    /// targets for the effect), strongest first round-robin per hop, capped at 2*nbr, sorted.
    pub fn universe(&self, m: &[usize], dir: Dir) -> Vec<usize> {
        let lists = match dir {
            Dir::Cause => &self.nin,
            Dir::Effect => &self.nout,
        };
        let cap = 2 * self.cfg.nbr;
        let mut seen = vec![false; self.cx.n];
        let mut out: Vec<usize> = Vec::new();
        let mut front: Vec<usize> = m.to_vec();
        'hops: for _ in 0..self.tau {
            let mut next: Vec<usize> = Vec::new();
            for r in 0..self.cfg.nbr {
                for &x in &front {
                    if let Some(&y) = lists[x].get(r) {
                        if !seen[y] {
                            seen[y] = true;
                            next.push(y);
                            out.push(y);
                            if out.len() >= cap {
                                break 'hops;
                            }
                        }
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            front = next;
        }
        out.sort_unstable();
        out
    }

    /// Candidate purviews: non-empty subsets of the (sorted) universe `u`, by increasing size,
    /// with size <= qmax and |A(E)|^size <= max_states.
    pub fn purviews(&self, u: &[usize]) -> Vec<Vec<usize>> {
        let a = self.cx.a;
        let mut out: Vec<Vec<usize>> = Vec::new();
        for size in 1..=self.cfg.qmax.min(u.len()) {
            if ipow(a, size) > self.cfg.max_states {
                break;
            }
            let mut cur: Vec<usize> = Vec::new();
            combos(u, size, 0, &mut cur, &mut out);
        }
        out
    }

    // ---------- cause side ----------

    /// Cause repertoire of the mechanism block `m` over the purview block `z` (both sorted),
    /// as a table over the joint states of `z` (index = sum_i digit_i * a^i, digit_i = symbol of
    /// the i-th node of `z`): the Bayesian inversion of the TPM with a flat prior on A(E)^|z|.
    pub(crate) fn cause_table(&self, m: &[usize], z: &[usize]) -> Rc<Vec<f64>> {
        let mut ks: Vec<usize> = Vec::with_capacity(m.len() + z.len() + 1);
        ks.extend_from_slice(m);
        ks.push(SEP);
        ks.extend_from_slice(z);
        let key = hash_usize(TAG_CTAB, &ks);
        let hit = self.ctab.borrow().get(&key).cloned();
        if let Some(t) = hit {
            return t;
        }
        let a = self.cx.a;
        let k = z.len();
        let total = ipow(a, k);
        let sm: Vec<u16> = m.iter().map(|&j| self.state[j]).collect();
        let mut tab = vec![0.0f64; total];
        let mut zs = vec![0u16; k];
        let mut out = vec![0.0f64; a];
        let mut sum = 0.0f64;
        for idx in 0..total {
            decode(idx, a, &mut zs);
            let mut l = 1.0f64;
            for (t, &j) in m.iter().enumerate() {
                self.cx.marg(j, z, self.tau, &zs, &mut out);
                l *= out[sm[t] as usize];
            }
            tab[idx] = l;
            sum += l;
        }
        if sum > TINY {
            let inv = 1.0 / sum;
            for v in tab.iter_mut() {
                *v *= inv;
            }
        } else {
            tab.fill(1.0 / total as f64);
        }
        let rc = Rc::new(tab);
        let mut c = self.ctab.borrow_mut();
        if self.ctab_sz.get() + total > CACHE_CAP {
            c.clear();
            self.ctab_sz.set(0);
        }
        self.ctab_sz.set(self.ctab_sz.get() + total);
        c.insert(key, Rc::clone(&rc));
        rc
    }

    /// Partitioned cause repertoire over all joint states of `z`, written into `q`.
    /// `lab` labels the elements of m (first) and z (after) by block.
    fn partitioned_cause(&self, m: &[usize], z: &[usize], lab: &[u8], apow: &[usize], q: &mut [f64]) {
        let a = self.cx.a;
        let km = m.len();
        let kz = z.len();
        let nb = lab.iter().copied().max().map_or(0, |x| x as usize + 1);
        q.fill(1.0);
        for b in 0..nb {
            let bb = b as u8;
            let mb: Vec<usize> = (0..km).filter(|&i| lab[i] == bb).map(|i| m[i]).collect();
            let zpos: Vec<usize> = (0..kz).filter(|&i| lab[km + i] == bb).collect();
            if zpos.is_empty() {
                continue; // a mechanism part without purview contributes the factor 1
            }
            if mb.is_empty() {
                // purview part without mechanism: unconstrained, uniform on A(E)^|Z_b|
                let f = 1.0 / (ipow(a, zpos.len()) as f64);
                for v in q.iter_mut() {
                    *v *= f;
                }
            } else {
                let zb: Vec<usize> = zpos.iter().map(|&i| z[i]).collect();
                let tab = self.cause_table(&mb, &zb);
                for (idx, v) in q.iter_mut().enumerate() {
                    *v *= tab[sub_index(idx, a, &zpos, apow)];
                }
            }
        }
    }

    /// phi_c(m, Z) = min_psi ID(pi_c, pi_c^psi), with the specified cause state(s) read at the
    /// MIP. Stops early (exact = false) when the running minimum drops below `floor`.
    fn cause_eval(&self, m: &[usize], z: &[usize], floor: f64) -> SideEval {
        let a = self.cx.a;
        let km = m.len();
        let kz = z.len();
        let total = ipow(a, kz);
        let p = self.cause_table(m, z);
        let apow: Vec<usize> = (0..kz).map(|i| ipow(a, i)).collect();
        let parts = set_partitions(km, kz);
        let mut q = vec![0.0f64; total];
        let mut best = f64::INFINITY;
        let mut best_q: Vec<f64> = Vec::new();
        for lab in &parts {
            self.partitioned_cause(m, z, lab, &apow, &mut q);
            let (v, _) = id_max(&p, &q);
            if v < best {
                best = v;
                best_q.clear();
                best_q.extend_from_slice(&q);
            }
            if best <= PHI_EPS {
                return SideEval { phi: best, states: Vec::new(), exact: true };
            }
            if best < floor {
                return SideEval { phi: best, states: Vec::new(), exact: false };
            }
        }
        if !best.is_finite() {
            return SideEval { phi: 0.0, states: Vec::new(), exact: true };
        }
        let tol = TIE_REL * best.max(1.0);
        let mut states: Vec<Vec<u16>> = Vec::new();
        for idx in 0..total {
            if id_term(p[idx], best_q[idx]) >= best - tol {
                let mut zs = vec![0u16; kz];
                decode(idx, a, &mut zs);
                states.push(zs);
                if states.len() >= MAX_TIES {
                    break;
                }
            }
        }
        SideEval { phi: best, states, exact: true }
    }

    // ---------- effect side ----------

    /// phi_e(m, Z) = min_psi ID_E(pi_e, pi_e^psi): the coordinates of Z that carry a byte of E
    /// are fixed to E, the latent ones range over A(E) (argmax). Specified states read at the MIP.
    fn effect_eval(&self, m: &[usize], z: &[usize], floor: f64) -> SideEval {
        let a = self.cx.a;
        let km = m.len();
        let kz = z.len();
        let sm: Vec<u16> = m.iter().map(|&j| self.state[j]).collect();
        // unpartitioned effect repertoire: one marginal per purview node
        let pz: Vec<Vec<f64>> = z
            .iter()
            .map(|&y| self.cx.marg_vec(y, m, self.tau, &sm))
            .collect();
        let fixed: Vec<Option<u16>> = z.iter().map(|&y| self.cx.e_sym(y)).collect();
        let lat: Vec<usize> = (0..kz).filter(|&i| fixed[i].is_none()).collect();
        let nstate = ipow(a, lat.len());
        // candidate joint states (anchored coordinates = E) and their unpartitioned probability
        let mut cand = vec![0u16; nstate * kz];
        let mut pc = vec![1.0f64; nstate];
        let mut digits = vec![0u16; lat.len()];
        for c in 0..nstate {
            decode(c, a, &mut digits);
            let mut t = 0usize;
            for i in 0..kz {
                let s = match fixed[i] {
                    Some(s) => s,
                    None => {
                        let d = digits[t];
                        t += 1;
                        d
                    }
                };
                cand[c * kz + i] = s;
                pc[c] *= pz[i][s as usize];
            }
        }
        let parts = set_partitions(km, kz);
        let mut qy: Vec<Vec<f64>> = vec![vec![0.0f64; a]; kz];
        let mut qc = vec![0.0f64; nstate];
        let mut best = f64::INFINITY;
        let mut best_q: Vec<f64> = Vec::new();
        for lab in &parts {
            for i in 0..kz {
                let bb = lab[km + i];
                let mut mb: Vec<usize> = Vec::new();
                let mut smb: Vec<u16> = Vec::new();
                for t in 0..km {
                    if lab[t] == bb {
                        mb.push(m[t]);
                        smb.push(sm[t]);
                    }
                }
                self.cx.marg(z[i], &mb, self.tau, &smb, &mut qy[i]);
            }
            let mut v = 0.0f64;
            for c in 0..nstate {
                let mut q = 1.0f64;
                for i in 0..kz {
                    q *= qy[i][cand[c * kz + i] as usize];
                }
                qc[c] = q;
                let t = id_term(pc[c], q);
                if t > v {
                    v = t;
                }
            }
            if v < best {
                best = v;
                best_q.clear();
                best_q.extend_from_slice(&qc);
            }
            if best <= PHI_EPS {
                return SideEval { phi: best, states: Vec::new(), exact: true };
            }
            if best < floor {
                return SideEval { phi: best, states: Vec::new(), exact: false };
            }
        }
        if !best.is_finite() {
            return SideEval { phi: 0.0, states: Vec::new(), exact: true };
        }
        let tol = TIE_REL * best.max(1.0);
        let mut states: Vec<Vec<u16>> = Vec::new();
        for c in 0..nstate {
            if id_term(pc[c], best_q[c]) >= best - tol {
                states.push(cand[c * kz..(c + 1) * kz].to_vec());
                if states.len() >= MAX_TIES {
                    break;
                }
            }
        }
        SideEval { phi: best, states, exact: true }
    }

    // ---------- public mechanism-purview values ----------

    /// phi_c(m, Z) and its specified cause state(s). Nodes must be free (not background).
    pub fn phi_cause(&self, m: &[usize], z: &[usize]) -> (f64, Vec<Vec<u16>>) {
        let (m, z) = (norm(m), norm(z));
        if m.is_empty() || z.is_empty() {
            return (0.0, Vec::new());
        }
        debug_assert!(m.iter().chain(z.iter()).all(|&i| self.cx.pinned(i).is_none()));
        let ev = self.cause_eval(&m, &z, -1.0);
        if ev.phi <= PHI_EPS { (0.0, Vec::new()) } else { (ev.phi, ev.states) }
    }

    /// phi_e(m, Z) and its specified effect state(s) (anchored coordinates at E).
    pub fn phi_effect(&self, m: &[usize], z: &[usize]) -> (f64, Vec<Vec<u16>>) {
        let (m, z) = (norm(m), norm(z));
        if m.is_empty() || z.is_empty() {
            return (0.0, Vec::new());
        }
        debug_assert!(m.iter().chain(z.iter()).all(|&i| self.cx.pinned(i).is_none()));
        let ev = self.effect_eval(&m, &z, -1.0);
        if ev.phi <= PHI_EPS { (0.0, Vec::new()) } else { (ev.phi, ev.states) }
    }

    // ---------- exclusion and distinctions ----------

    /// Maximally irreducible purview(s) of `m` on one side: argmax of phi over `cands`,
    /// ties kept (up to MAX_TIES). Returns (phi_max, faces); (0, []) when nothing is irreducible.
    fn best_side(&self, m: &[usize], cands: &[Vec<usize>], dir: Dir) -> (f64, Vec<Face>) {
        let mut best = 0.0f64;
        let mut faces: Vec<Face> = Vec::new();
        for z in cands {
            let floor = if best > PHI_EPS { best - TIE_REL * best.max(1.0) } else { -1.0 };
            let ev = match dir {
                Dir::Cause => self.cause_eval(m, z, floor),
                Dir::Effect => self.effect_eval(m, z, floor),
            };
            if !ev.exact || ev.phi <= PHI_EPS {
                continue;
            }
            let tol = TIE_REL * best.max(ev.phi).max(1.0);
            if ev.phi > best + tol {
                best = ev.phi;
                faces.clear();
                faces.push(Face { nodes: z.clone(), phi: ev.phi, states: ev.states });
            } else if ev.phi >= best - tol {
                if faces.len() < MAX_TIES {
                    faces.push(Face { nodes: z.clone(), phi: ev.phi, states: ev.states });
                }
                if ev.phi > best {
                    best = ev.phi;
                }
            }
        }
        (best, faces)
    }

    /// Every mechanism of the system with at least one irreducible side, with its maximally
    /// irreducible purviews. With `only_anchored`, mechanisms whose neighbourhoods contain no
    /// byte of E are skipped (they cannot be anchored). Order: by mechanism size, then id.
    pub fn distinctions(&self, only_anchored: bool) -> Vec<Dist> {
        let mut out: Vec<Dist> = Vec::new();
        for m in self.mechanisms() {
            let uc = self.universe(&m, Dir::Cause);
            let ue = self.universe(&m, Dir::Effect);
            if only_anchored && !(self.cx.touches_e(&uc) || self.cx.touches_e(&ue)) {
                continue;
            }
            let cc = self.purviews(&uc);
            let ce = self.purviews(&ue);
            let (phi_c, fc) = self.best_side(&m, &cc, Dir::Cause);
            let (phi_e, fe) = self.best_side(&m, &ce, Dir::Effect);
            if phi_c <= PHI_EPS && phi_e <= PHI_EPS {
                continue;
            }
            let anchored = fc.iter().chain(fe.iter()).any(|f| self.cx.touches_e(&f.nodes));
            let mech_state: Vec<u16> = m.iter().map(|&j| self.state[j]).collect();
            out.push(Dist {
                mech: m,
                mech_state,
                cause: fc,
                effect: fe,
                phi_c,
                phi_e,
                phi_d: phi_c.min(phi_e),
                anchored,
            });
        }
        out
    }

    /// Dist_E: the anchored distinctions (phi_d > 0) of the system.
    pub fn dist_e(&self) -> Vec<Dist> {
        self.distinctions(true)
            .into_iter()
            .filter(|d| d.anchored && d.is_distinction())
            .collect()
    }

    /// Positions of E touched by the purviews of d (its footprint in E), increasing.
    pub fn footprint(&self, d: &Dist) -> Vec<usize> {
        let mut v: Vec<usize> = Vec::new();
        for f in d.cause.iter().chain(d.effect.iter()) {
            for &j in &f.nodes {
                if let Some(p) = self.cx.e_pos(j) {
                    v.push(p);
                }
            }
        }
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Per-node marginals of the unpartitioned repertoire of (m, Z): marginals of the Bayesian
    /// inversion for the cause, per-node effect marginals for the effect. Each row sums to 1.
    /// (Used in part 4 to localise the residual on purview nodes.)
    pub fn repertoire_marginals(&self, m: &[usize], z: &[usize], dir: Dir) -> Vec<Vec<f64>> {
        let a = self.cx.a;
        let (m, z) = (norm(m), norm(z));
        let kz = z.len();
        let mut out = vec![vec![0.0f64; a]; kz];
        if m.is_empty() || kz == 0 {
            for row in out.iter_mut() {
                row.fill(1.0 / a as f64);
            }
            return out;
        }
        match dir {
            Dir::Effect => {
                let sm: Vec<u16> = m.iter().map(|&j| self.state[j]).collect();
                for (i, &y) in z.iter().enumerate() {
                    self.cx.marg(y, &m, self.tau, &sm, &mut out[i]);
                }
            }
            Dir::Cause => {
                let p = self.cause_table(&m, &z);
                let apow: Vec<usize> = (0..kz).map(|i| ipow(a, i)).collect();
                for idx in 0..p.len() {
                    for i in 0..kz {
                        out[i][(idx / apow[i]) % a] += p[idx];
                    }
                }
            }
        }
        out
    }

    // ---------- display ----------

    fn fmt_face(&self, f: &Face) -> String {
        let mut s = format!("{{{}}}=", join_ids(&f.nodes));
        for (k, st) in f.states.iter().enumerate() {
            if k > 0 {
                s.push('|');
            }
            for &v in st {
                match self.cx.alphabet.get(v as usize) {
                    Some(&b) => s.push_str(&show_byte(b)),
                    None => s.push('?'),
                }
            }
        }
        s.push_str(&format!(" phi={:.4}", f.phi));
        s
    }

    /// Readable form of a distinction, states decoded to bytes of E's alphabet.
    pub fn describe(&self, d: &Dist) -> String {
        let mut s = format!(
            "m{{{}}} phi_d={:.4} (c={:.4} e={:.4}){}",
            join_ids(&d.mech),
            d.phi_d,
            d.phi_c,
            d.phi_e,
            if d.anchored { " [E]" } else { "" }
        );
        for f in &d.cause {
            s.push_str(&format!("\n  cause  {}", self.fmt_face(f)));
        }
        for f in &d.effect {
            s.push_str(&format!("\n  effect {}", self.fmt_face(f)));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Hyper;
    use crate::percept::{random_renaming, rename, Percept};
    use crate::util::Rng;

    fn sig(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Node 0 is a perception node anchored to E[0]; node 1 is latent.
    /// w01 is the connection 0 -> 1, w10 is 1 -> 0 (set directly, no non-reciprocity pass).
    fn two_node(w01: f64, w10: f64, hold: f64) -> Model {
        let mut rng = Rng::new(1);
        let mut m = Model::random(1, 1, Hyper::default(), &mut rng);
        m.w = vec![0.0, w01, w10, 0.0];
        m.hold = hold;
        m.beta = 1.0;
        m
    }

    #[test]
    fn partitions_are_bell_minus_one() {
        assert_eq!(set_partitions(1, 1).len(), 1);
        assert_eq!(set_partitions(1, 2).len(), 4);
        assert_eq!(set_partitions(2, 2).len(), 14);
        assert_eq!(set_partitions(3, 2).len(), 51);
        for lab in set_partitions(2, 2) {
            assert_eq!(lab.len(), 4);
            let mut mx = 0u8;
            for (i, &l) in lab.iter().enumerate() {
                let cap = if i == 0 { 0 } else { mx + 1 };
                assert!(l <= cap);
                mx = mx.max(l);
            }
            assert!(mx >= 1);
        }
    }

    #[test]
    fn id_max_picks_the_largest_term() {
        let (v, k) = id_max(&[0.5, 0.5], &[0.25, 0.75]);
        assert!((v - 0.5).abs() < 1e-12);
        assert_eq!(k, 0);
        let (v0, _) = id_max(&[0.3, 0.7], &[0.3, 0.7]);
        assert_eq!(v0, 0.0);
    }

    #[test]
    fn latent_effect_and_cause_match_closed_forms() {
        let m = two_node(1.0, -0.5, 2.0);
        let e = Percept::from_bytes(b"ab");
        let cx = Ctx::new(&m, &e, 0);
        let ph = Phi::new(&cx, 1, &[0, 0], PhiCfg::from_model(&m));
        let p0 = sig(1.0);
        let want = p0 * (2.0 * p0).log2().abs();
        // effect of node 0 on the latent node 1: the unconstrained effect is exactly uniform
        let (fe, se) = ph.phi_effect(&[0], &[1]);
        assert!((fe - want).abs() < 1e-9, "{} vs {}", fe, want);
        assert_eq!(se, vec![vec![0u16]]);
        // cause of node 1 in node 0 (anchored, but the cause side keeps the argmax)
        let (fc, sc) = ph.phi_cause(&[1], &[0]);
        assert!((fc - want).abs() < 1e-9, "{} vs {}", fc, want);
        assert_eq!(sc, vec![vec![0u16]]);
        let t = ph.cause_table(&[1], &[0]);
        assert!((t.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn effect_on_anchored_node_is_evaluated_at_e_not_at_argmax() {
        let (w10, hold) = (3.0f64, 2.0f64);
        let m = two_node(-0.5, w10, hold);
        let e = Percept::from_bytes(b"ab");
        let cx = Ctx::new(&m, &e, 0);
        assert_eq!(cx.e_sym(0), Some(0));
        let ph = Phi::new(&cx, 1, &[0, 1], PhiCfg::from_model(&m));
        // node 0 reads node 1 (= symbol 1): field [hold, w10] -> p(sym 0) = sig(hold - w10)
        let p0 = sig(hold - w10);
        // cut: node 1 becomes uniform noise: x1 = 0 -> field [hold + w10, 0]; x1 = 1 -> [hold, w10]
        let q0 = 0.5 * (sig(hold + w10) + sig(hold - w10));
        let at_e = p0 * (p0 / q0).log2().abs();
        let (p1, q1) = (1.0 - p0, 1.0 - q0);
        let at_argmax = p1 * (p1 / q1).log2().abs();
        assert!(at_argmax > at_e + 0.1);
        let (fe, se) = ph.phi_effect(&[1], &[0]);
        assert!((fe - at_e).abs() < 1e-9, "{} vs {}", fe, at_e);
        assert_eq!(se, vec![vec![0u16]]);
    }

    #[test]
    fn unconnected_graph_has_no_irreducible_structure() {
        let m = two_node(0.0, 0.0, 2.0);
        let e = Percept::from_bytes(b"ab");
        let cx = Ctx::new(&m, &e, 0);
        let ph = Phi::new(&cx, 1, &[0, 1], PhiCfg::from_model(&m));
        assert_eq!(ph.phi_effect(&[0], &[1]).0, 0.0);
        assert_eq!(ph.phi_cause(&[0], &[1]).0, 0.0);
        assert!(ph.distinctions(false).is_empty());
    }

    #[test]
    fn distinctions_of_the_two_node_graph() {
        let (w01, w10, hold) = (1.0f64, -0.5f64, 2.0f64);
        let m = two_node(w01, w10, hold);
        let e = Percept::from_bytes(b"ab");
        let cx = Ctx::new(&m, &e, 0);
        let ph = Phi::new(&cx, 1, &[0, 0], PhiCfg::from_model(&m));
        let ds = ph.distinctions(false);
        for d in &ds {
            assert!((d.phi_d - d.phi_c.min(d.phi_e)).abs() < 1e-15);
            assert!(d.phi_c >= 0.0 && d.phi_e >= 0.0);
            for f in d.cause.iter().chain(d.effect.iter()) {
                assert!(f.nodes.windows(2).all(|w| w[0] < w[1]));
                assert!(!f.states.is_empty());
                assert!(f.states.iter().all(|s| s.len() == f.nodes.len()));
            }
        }
        // mechanism {0} (state 0): cause and effect purview are the latent node 1, not anchored
        let l0 = sig(hold + w10); // node 0 reads node 1 = 0
        let l1 = sig(hold - w10); // node 0 reads node 1 = 1
        let (c0, c1) = (l0 / (l0 + l1), l1 / (l0 + l1));
        let want_c0 = (c0 * (2.0 * c0).log2().abs()).max(c1 * (2.0 * c1).log2().abs());
        let pe = sig(w01);
        let want_x = pe * (2.0 * pe).log2().abs();
        let d0 = ds.iter().find(|d| d.mech == vec![0]).expect("distinction of node 0");
        assert!((d0.phi_c - want_c0).abs() < 1e-9, "{} vs {}", d0.phi_c, want_c0);
        assert!((d0.phi_e - want_x).abs() < 1e-9, "{} vs {}", d0.phi_e, want_x);
        assert_eq!(d0.cause[0].nodes, vec![1]);
        assert_eq!(d0.effect[0].nodes, vec![1]);
        assert!(!d0.anchored);
        // mechanism {1} (state 0): purview {0} is anchored, its effect is read at E
        let pa = sig(hold + w10);
        let qa = 0.5 * (sig(hold + w10) + sig(hold - w10));
        let want_e1 = pa * (pa / qa).log2().abs();
        let d1 = ds.iter().find(|d| d.mech == vec![1]).expect("distinction of node 1");
        assert!((d1.phi_c - want_x).abs() < 1e-9, "{} vs {}", d1.phi_c, want_x);
        assert!((d1.phi_e - want_e1).abs() < 1e-9, "{} vs {}", d1.phi_e, want_e1);
        assert!(d1.anchored && d1.is_distinction());
        assert_eq!(ph.footprint(d1), vec![0]);
        assert!(sum_phi_d(&ds) >= d1.phi_d - 1e-12);
        assert!(!ph.describe(d1).is_empty());
    }

    #[test]
    fn distinctions_are_renaming_invariant_and_deterministic() {
        let mut rng = Rng::new(7);
        let m = Model::random(3, 2, Hyper::default(), &mut rng);
        let perm = random_renaming(&mut rng);
        let raw = b"abcab";
        let e1 = Percept::from_bytes(raw);
        let e2 = Percept::from_bytes(&rename(raw, &perm));
        let cx1 = Ctx::new(&m, &e1, 0);
        let cx2 = Ctx::new(&m, &e2, 0);
        let s1 = cx1.trajectory();
        let s2 = cx2.trajectory();
        assert_eq!(s1, s2);
        let d1 = Phi::standard(&cx1, 1, &s1[2]).distinctions(false);
        let d2 = Phi::standard(&cx2, 1, &s2[2]).distinctions(false);
        let d3 = Phi::standard(&cx1, 1, &s1[2]).distinctions(false);
        assert!(!d1.is_empty());
        assert_eq!(d1.len(), d2.len());
        assert_eq!(d1.len(), d3.len());
        for ((x, y), z) in d1.iter().zip(d2.iter()).zip(d3.iter()) {
            assert_eq!(x.mech, y.mech);
            assert_eq!(x.phi_d.to_bits(), y.phi_d.to_bits());
            assert_eq!(x.phi_d.to_bits(), z.phi_d.to_bits());
            assert_eq!(x.cause.len(), y.cause.len());
            assert_eq!(x.effect.len(), y.effect.len());
        }
    }

    #[test]
    fn repertoire_marginals_are_distributions() {
        let mut rng = Rng::new(3);
        let m = Model::random(3, 2, Hyper::default(), &mut rng);
        let e = Percept::from_bytes(b"abcab");
        let cx = Ctx::new(&m, &e, 0);
        let st = cx.initial_state();
        let ph = Phi::standard(&cx, 1, &st);
        for dir in [Dir::Cause, Dir::Effect] {
            let mg = ph.repertoire_marginals(&[0, 3], &[1, 4], dir);
            assert_eq!(mg.len(), 2);
            for row in &mg {
                assert_eq!(row.len(), cx.a);
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            }
        }
    }
}