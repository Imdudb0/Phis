//! sim.rs — analysis context, stochastic dynamics and the macro-marginal engine.
//!
//! Every repertoire of phi.rs is built from one primitive, `Ctx::marg(j, clamp, tau, syms, out)`:
//! the distribution over A(E) of node j, tau micro-steps after a state in which the nodes
//! `clamp` hold the symbols `syms` and every other free node is uniform noise on A(E).
//!
//! - Effect side: P(Z at t+tau | m at t) is the product over y in Z of marg(y, m, tau, s_m).
//! - Cause side: P(m at t | Z at t-tau = z) is the product over j in m of marg(j, Z, tau, z).
//!   The Bayesian inversion (flat prior on A(E)) is done in phi.rs.
//! - A directed cut is expressed by the clamp: a source that is not clamped is replaced by
//!   uniform noise on A(E) for that target, so "cut i -> j" means "marg(j, clamp without i)".
//!   Cuts act on macro-connections (i at t-tau to j at t), the grain being tau micro-steps.
//! - Background: nodes outside the candidate system are pinned to their state (constants,
//!   never updated, never noised). With no background every node is free.
//!
//! Estimation: R = hy.draws coupled draws. The initial noise comes from a table built once per
//! context (common random numbers, stratified per node by blocks of permutations of A(E)),
//! intermediate micro-steps are sampled from hash-based uniforms, and the last step is exact
//! (Rao-Blackwellised softmax). Every output is therefore smooth, strictly positive and a pure
//! function of its arguments: it can be cached, and the difference between two repertoires
//! (what phi measures) has low variance because both share the same noise.
//!
//! A Ctx is Send but not Sync (interior cache): build one per thread.

use crate::model::{Model, WEPS};
use crate::percept::Percept;
use crate::util::*;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Floor of every probability produced here, so that every logarithm downstream is finite.
pub const PFLOOR: f64 = 1e-12;

const TAG_NOISE: u64 = 0x4E4F_4953_4500_0001;
const TAG_STEP: u64 = 0x5354_4550_0000_0002;
const TAG_TRAJ: u64 = 0x5452_414A_0000_0003;
const TAG_INIT: u64 = 0x494E_4954_0000_0004;
const TAG_KEY: u64 = 0x4B45_5900_0000_0005;
const TAG_BG: u64 = 0x4247_0000_0000_0006;

/// Cache keys are already well-mixed 64-bit hashes: hashing them again would be wasted work.
#[derive(Default, Clone, Copy)]
pub struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

pub type IdBuild = BuildHasherDefault<IdHasher>;

/// p = softmax(beta * f), floored at PFLOOR and renormalised. `f` and `p` have the same length (>= 1).
pub fn softmax_into(f: &[f64], beta: f64, p: &mut [f64]) {
    let mut mx = f[0];
    for &v in &f[1..] {
        if v > mx {
            mx = v;
        }
    }
    let mut s = 0.0f64;
    for k in 0..f.len() {
        let e = (beta * (f[k] - mx)).exp();
        p[k] = e;
        s += e;
    }
    let inv = 1.0 / s;
    let mut s2 = 0.0f64;
    for k in 0..f.len() {
        let v = (p[k] * inv).max(PFLOOR);
        p[k] = v;
        s2 += v;
    }
    let inv2 = 1.0 / s2;
    for k in 0..f.len() {
        p[k] *= inv2;
    }
}

/// Inverse-CDF draw of a symbol from `p` with a uniform `u` in [0,1).
#[inline]
fn sample(p: &[f64], u: f64) -> u16 {
    let mut x = u;
    for (k, &q) in p.iter().enumerate() {
        if x < q {
            return k as u16;
        }
        x -= q;
    }
    (p.len() - 1) as u16
}

/// Node ids sorted by decreasing weight (ties by id), at most `cap` of them.
fn top_by_weight(mut v: Vec<(usize, f64)>, cap: usize) -> Vec<usize> {
    v.sort_by(|x, y| {
        y.1.partial_cmp(&x.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.0.cmp(&y.0))
    });
    v.truncate(cap);
    v.iter().map(|p| p.0).collect()
}

struct Scratch {
    x: Vec<u16>,
    y: Vec<u16>,
    f: Vec<f64>,
    p: Vec<f64>,
}

/// Analysis context for one (model, E, frame): effective weights, E anchoring, background,
/// noise table and marginal cache.
pub struct Ctx<'a> {
    pub m: &'a Model,
    pub n: usize,
    /// |A(E)| (at least 1)
    pub a: usize,
    /// alphabet[code] = original byte; used to decode, never by the dynamics
    pub alphabet: Vec<u8>,
    /// (start, len) of the frame of E carried by the perception nodes
    pub frame: (usize, usize),
    /// inn[j]: strongest sources i of connections i -> j (at most hy.nbr)
    pub inn: Vec<Vec<usize>>,
    /// outn[i]: strongest targets j of connections i -> j (at most hy.nbr)
    pub outn: Vec<Vec<usize>>,
    /// E symbol biasing node j (perception nodes that carry a byte of the frame)
    bias: Vec<Option<u16>>,
    /// common-random-numbers seed: model seed + recoded fingerprint of E + frame start
    seed: u64,
    /// R: Monte-Carlo draws per marginal
    draws: usize,
    /// effective weights stored by target: win[j*n+i] = w_ij, zero when |w_ij| < WEPS
    win: Vec<f64>,
    pin: Vec<Option<u16>>,
    free: Vec<usize>,
    /// background field: E bias + contribution of the pinned sources, n*a
    bgf: Vec<f64>,
    bg_id: u64,
    /// initial noise, noise[d*n+i] = symbol of node i in draw d
    noise: Vec<u16>,
    cache: RefCell<HashMap<u64, Box<[f64]>, IdBuild>>,
    scratch: RefCell<Scratch>,
    calls: Cell<u64>,
    hits: Cell<u64>,
}

impl<'a> Ctx<'a> {
    /// Number of frames of E for this model (at least 1, even for an empty E).
    pub fn n_frames(m: &Model, e: &Percept) -> usize {
        e.frames(m.np).len().max(1)
    }

    /// Context for frame number `frame` of E (clamped into range). Perception node k of the
    /// frame is anchored to the byte E[start + k]; the other nodes are not anchored.
    pub fn new(m: &'a Model, e: &Percept, frame: usize) -> Ctx<'a> {
        let n = m.n();
        let fr = e.frames(m.np);
        let (start, len) = if fr.is_empty() {
            (0usize, 0usize)
        } else {
            fr[frame.min(fr.len() - 1)]
        };
        let a = e.a.max(1);
        let mut bias: Vec<Option<u16>> = vec![None; n];
        for k in 0..len.min(m.np) {
            bias[k] = Some(e.codes[start + k]);
        }
        let mut win = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let w = m.w[i * n + j];
                if i != j && w.abs() >= WEPS {
                    win[j * n + i] = w;
                }
            }
        }
        let cap = m.hy.nbr.max(1);
        let mut inn: Vec<Vec<usize>> = Vec::with_capacity(n);
        let mut outn: Vec<Vec<usize>> = Vec::with_capacity(n);
        for k in 0..n {
            let mut vi: Vec<(usize, f64)> = Vec::new();
            let mut vo: Vec<(usize, f64)> = Vec::new();
            for o in 0..n {
                let wi = win[k * n + o].abs(); // o -> k
                if wi > 0.0 {
                    vi.push((o, wi));
                }
                let wo = win[o * n + k].abs(); // k -> o
                if wo > 0.0 {
                    vo.push((o, wo));
                }
            }
            inn.push(top_by_weight(vi, cap));
            outn.push(top_by_weight(vo, cap));
        }
        let seed = mix3(m.hy.seed, e.fingerprint, start as u64);
        let mut c = Ctx {
            m,
            n,
            a,
            alphabet: e.alphabet.clone(),
            frame: (start, len),
            inn,
            outn,
            bias,
            seed,
            draws: m.hy.draws.max(1),
            win,
            pin: vec![None; n],
            free: (0..n).collect(),
            bgf: vec![0.0; n * a],
            bg_id: 0,
            noise: Vec::new(),
            cache: RefCell::new(HashMap::default()),
            scratch: RefCell::new(Scratch {
                x: vec![0; n],
                y: vec![0; n],
                f: vec![0.0; a],
                p: vec![0.0; a],
            }),
            calls: Cell::new(0),
            hits: Cell::new(0),
        };
        c.build_noise();
        c.set_background(&[], &[]);
        c
    }

    fn build_noise(&mut self) {
        let (n, a, r) = (self.n, self.a, self.draws);
        let blocks = r.div_ceil(a);
        let mut noise = vec![0u16; r * n];
        let mut perm: Vec<u16> = Vec::with_capacity(a);
        for i in 0..n {
            for b in 0..blocks {
                perm.clear();
                perm.extend(0..a as u16);
                let mut g = Rng::new(mix4(self.seed, TAG_NOISE, i as u64, b as u64));
                g.shuffle(&mut perm);
                for k in 0..a {
                    let d = b * a + k;
                    if d < r {
                        noise[d * n + i] = perm[k];
                    }
                }
            }
        }
        self.noise = noise;
    }

    /// Change R (rebuilds the noise table and empties the cache).
    pub fn set_draws(&mut self, r: usize) {
        self.draws = r.max(1);
        self.build_noise();
        self.cache.get_mut().clear();
    }

    /// Pin `nodes` to `syms` (distinct nodes). Cached marginals of other backgrounds stay valid
    /// for when that background comes back.
    pub fn set_background(&mut self, nodes: &[usize], syms: &[u16]) {
        let n = self.n;
        let a = self.a;
        let amax = (a - 1) as u16;
        let mut pin: Vec<Option<u16>> = vec![None; n];
        for (&i, &s) in nodes.iter().zip(syms.iter()) {
            if i < n {
                pin[i] = Some(s.min(amax));
            }
        }
        let mut id = TAG_BG;
        for i in 0..n {
            if let Some(s) = pin[i] {
                id = id.wrapping_add(mix3(TAG_BG, i as u64, s as u64));
            }
        }
        let mut bgf = vec![0.0f64; n * a];
        for j in 0..n {
            if let Some(b) = self.bias[j] {
                bgf[j * a + b as usize] += self.m.hold;
            }
            for i in 0..n {
                if let Some(s) = pin[i] {
                    bgf[j * a + s as usize] += self.win[j * n + i];
                }
            }
        }
        self.free = (0..n).filter(|&i| pin[i].is_none()).collect();
        self.pin = pin;
        self.bgf = bgf;
        self.bg_id = id;
    }

    pub fn clear_background(&mut self) {
        self.set_background(&[], &[]);
    }

    // ---------- views ----------

    pub fn draws(&self) -> usize {
        self.draws
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Nodes that are not pinned (the current system), in increasing order.
    pub fn free_nodes(&self) -> &[usize] {
        &self.free
    }

    pub fn pinned(&self, i: usize) -> Option<u16> {
        self.pin[i]
    }

    pub fn anchored(&self, j: usize) -> bool {
        self.bias[j].is_some()
    }

    /// Symbol of E that biases node j (None if j is not anchored in E).
    pub fn e_sym(&self, j: usize) -> Option<u16> {
        self.bias[j]
    }

    /// Position in E of the byte that node j carries.
    pub fn e_pos(&self, j: usize) -> Option<usize> {
        if self.bias[j].is_some() { Some(self.frame.0 + j) } else { None }
    }

    /// Original byte that node j carries (decoding only).
    pub fn byte_of(&self, j: usize) -> Option<u8> {
        self.bias[j].map(|c| self.alphabet[c as usize])
    }

    /// True when at least one of `nodes` carries a byte of E (an anchored purview).
    pub fn touches_e(&self, nodes: &[usize]) -> bool {
        nodes.iter().any(|&i| self.bias[i].is_some())
    }

    /// Effective weight of i -> j (0 when |w| < WEPS).
    pub fn weight(&self, i: usize, j: usize) -> f64 {
        self.win[j * self.n + i]
    }

    /// Causal influence of the connection i -> j (beta*|w|), used to weight min-cuts.
    pub fn infl(&self, i: usize, j: usize) -> f64 {
        self.m.beta * self.weight(i, j).abs()
    }

    /// (marg calls, cache hits)
    pub fn stats(&self) -> (u64, u64) {
        (self.calls.get(), self.hits.get())
    }

    pub fn cache_len(&self) -> usize {
        self.cache.borrow().len()
    }

    // ---------- dynamics ----------

    /// Field of node j given the FULL state x: E bias + sum of w_ij on the symbol held by i.
    fn field_all(&self, j: usize, x: &[u16], f: &mut [f64]) {
        let n = self.n;
        f.fill(0.0);
        if let Some(b) = self.bias[j] {
            f[b as usize] += self.m.hold;
        }
        let row = &self.win[j * n..(j + 1) * n];
        for i in 0..n {
            let w = row[i];
            if w != 0.0 {
                f[x[i] as usize] += w;
            }
        }
    }

    /// Same field, but pinned sources come from the precomputed background field and only
    /// the free sources are read from x.
    fn field_free(&self, j: usize, x: &[u16], f: &mut [f64]) {
        let a = self.a;
        let n = self.n;
        f.copy_from_slice(&self.bgf[j * a..(j + 1) * a]);
        let row = &self.win[j * n..(j + 1) * n];
        for &i in &self.free {
            let w = row[i];
            if w != 0.0 {
                f[x[i] as usize] += w;
            }
        }
    }

    /// One synchronous micro-step of the free nodes (pinned entries of y are untouched).
    fn step_free(&self, x: &[u16], y: &mut [u16], ds: u64, t: u64, f: &mut [f64], p: &mut [f64]) {
        let beta = self.m.beta;
        for &j in &self.free {
            self.field_free(j, x, f);
            softmax_into(f, beta, p);
            y[j] = sample(p, unit(mix3(ds, t, j as u64)));
        }
    }

    /// Exact one-step conditional distribution of node j given the FULL state x (length n,
    /// pinned nodes included in x). Does not use the background shortcut.
    pub fn cond(&self, j: usize, x: &[u16], out: &mut [f64]) {
        let mut f = vec![0.0f64; self.a];
        self.field_all(j, x, &mut f);
        softmax_into(&f, self.m.beta, &mut out[..self.a]);
    }

    /// s_0: perception nodes carry the frame's symbols, every other node starts on a
    /// hash-drawn symbol. Depends on E only through its recoded codes.
    pub fn initial_state(&self) -> Vec<u16> {
        let mut s = Vec::with_capacity(self.n);
        for j in 0..self.n {
            s.push(match self.bias[j] {
                Some(b) => b,
                None => below(mix3(self.seed, TAG_INIT, j as u64), self.a) as u16,
            });
        }
        s
    }

    /// One micro-step of the whole graph (background ignored) with hash-based uniforms.
    pub fn step_all(&self, x: &[u16], t: u64, y: &mut [u16]) {
        let mut f = vec![0.0f64; self.a];
        let mut p = vec![0.0f64; self.a];
        for j in 0..self.n {
            self.field_all(j, x, &mut f);
            softmax_into(&f, self.m.beta, &mut p);
            y[j] = sample(&p, unit(mix3(self.seed ^ TAG_TRAJ, t, j as u64)));
        }
    }

    /// The B+1 states s_0..s_B of the inference dynamics (background ignored). Deterministic
    /// given (model, E, frame) and invariant under byte renaming of E.
    pub fn trajectory(&self) -> Vec<Vec<u16>> {
        let b = self.m.hy.budget;
        let mut out: Vec<Vec<u16>> = Vec::with_capacity(b + 1);
        out.push(self.initial_state());
        for t in 1..=b {
            let mut y = vec![0u16; self.n];
            self.step_all(&out[t - 1], t as u64, &mut y);
            out.push(y);
        }
        out
    }

    // ---------- the macro-marginal engine ----------

    /// Macro-marginal: the distribution (length a, written to `out[..a]`) of node `j` after
    /// `tau` micro-steps, from a state where `clamp[k]` holds `syms[k]` and every other free
    /// node is uniform noise on A(E). `clamp` must hold distinct nodes. Pinned nodes are
    /// constants: clamping one is ignored, and asking for one returns a one-hot.
    /// Cached by (background, j, tau, clamp as a set).
    pub fn marg(&self, j: usize, clamp: &[usize], tau: usize, syms: &[u16], out: &mut [f64]) {
        let a = self.a;
        let n = self.n;
        debug_assert_eq!(clamp.len(), syms.len());
        debug_assert!(out.len() >= a);
        self.calls.set(self.calls.get() + 1);
        if let Some(s) = self.pin[j] {
            out[..a].fill(0.0);
            out[s as usize] = 1.0;
            return;
        }
        let tau = tau.max(1);
        let mut kc = 0u64;
        for (&c, &s) in clamp.iter().zip(syms.iter()) {
            kc = kc.wrapping_add(mix3(TAG_KEY, c as u64, s as u64));
        }
        let key = mix4(self.bg_id, j as u64, tau as u64, kc);
        {
            let cache = self.cache.borrow();
            if let Some(v) = cache.get(&key) {
                out[..a].copy_from_slice(v);
                self.hits.set(self.hits.get() + 1);
                return;
            }
        }
        let amax = (a - 1) as u16;
        let r = self.draws;
        let beta = self.m.beta;
        out[..a].fill(0.0);
        {
            let mut guard = self.scratch.borrow_mut();
            let Scratch { x, y, f, p } = &mut *guard;
            for d in 0..r {
                let base = d * n;
                for &i in &self.free {
                    x[i] = self.noise[base + i];
                }
                for (&c, &s) in clamp.iter().zip(syms.iter()) {
                    if c < n && self.pin[c].is_none() {
                        x[c] = s.min(amax);
                    }
                }
                if tau > 1 {
                    let ds = mix(self.seed ^ TAG_STEP, d as u64);
                    for t in 1..tau {
                        self.step_free(&x[..], &mut y[..], ds, t as u64, &mut f[..], &mut p[..]);
                        std::mem::swap(x, y);
                    }
                }
                self.field_free(j, &x[..], &mut f[..]);
                softmax_into(&f[..], beta, &mut p[..]);
                for v in 0..a {
                    out[v] += p[v];
                }
            }
        }
        let inv = 1.0 / r as f64;
        for v in 0..a {
            out[v] *= inv;
        }
        {
            let mut cache = self.cache.borrow_mut();
            let cap = ((64usize << 20) / (8 * a)).max(1024);
            if cache.len() >= cap {
                cache.clear();
            }
            cache.insert(key, out[..a].to_vec().into_boxed_slice());
        }
    }

    pub fn marg_vec(&self, j: usize, clamp: &[usize], tau: usize, syms: &[u16]) -> Vec<f64> {
        let mut o = vec![0.0f64; self.a];
        self.marg(j, clamp, tau, syms, &mut o);
        o
    }

    fn decode_free(&self, mut idx: usize, x: &mut [u16]) {
        for &i in &self.free {
            x[i] = (idx % self.a) as u16;
            idx /= self.a;
        }
    }

    /// Exact counterpart of `marg` by full enumeration of the free nodes' joint states
    /// (calibration on tiny graphs only). Returns false, leaving `out` untouched, when the
    /// enumeration would be too large (a^free > 4e6 for tau = 1, > 2048 for tau > 1).
    pub fn marg_exact(&self, j: usize, clamp: &[usize], tau: usize, syms: &[u16], out: &mut [f64]) -> bool {
        let n = self.n;
        let a = self.a;
        if let Some(s) = self.pin[j] {
            out[..a].fill(0.0);
            out[s as usize] = 1.0;
            return true;
        }
        let tau = tau.max(1);
        let k = self.free.len();
        let mut total: usize = 1;
        for _ in 0..k {
            total = match total.checked_mul(a) {
                Some(t) if t <= 4_000_000 => t,
                _ => return false,
            };
        }
        if tau > 1 && total > 2_048 {
            return false;
        }
        let amax = (a - 1) as u16;
        let mut fixed: Vec<Option<u16>> = vec![None; n];
        for (&c, &s) in clamp.iter().zip(syms.iter()) {
            if c < n && self.pin[c].is_none() {
                fixed[c] = Some(s.min(amax));
            }
        }
        let mut x = vec![0u16; n];
        for i in 0..n {
            if let Some(s) = self.pin[i] {
                x[i] = s;
            }
        }
        let mut f = vec![0.0f64; a];
        let mut pj = vec![0.0f64; a];
        let mut dist = vec![0.0f64; k * a];
        // initial distribution over the joint states of the free nodes
        let mut p = vec![0.0f64; total];
        for idx in 0..total {
            self.decode_free(idx, &mut x);
            let mut w = 1.0f64;
            for &i in &self.free {
                w *= match fixed[i] {
                    Some(s) => {
                        if x[i] == s {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    None => 1.0 / a as f64,
                };
            }
            p[idx] = w;
        }
        // tau - 1 exact micro-steps (product of per-node conditionals)
        for _ in 1..tau {
            let mut q = vec![0.0f64; total];
            for src in 0..total {
                if p[src] == 0.0 {
                    continue;
                }
                self.decode_free(src, &mut x);
                for (t, &i) in self.free.iter().enumerate() {
                    self.field_all(i, &x, &mut f);
                    softmax_into(&f, self.m.beta, &mut dist[t * a..(t + 1) * a]);
                }
                for dst in 0..total {
                    let mut pr = p[src];
                    let mut d = dst;
                    for t in 0..k {
                        pr *= dist[t * a + d % a];
                        d /= a;
                    }
                    q[dst] += pr;
                }
            }
            p = q;
        }
        // last step: exact softmax of node j
        out[..a].fill(0.0);
        for idx in 0..total {
            if p[idx] == 0.0 {
                continue;
            }
            self.decode_free(idx, &mut x);
            self.field_all(j, &x, &mut f);
            softmax_into(&f, self.m.beta, &mut pj);
            for v in 0..a {
                out[v] += p[idx] * pj[v];
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Hyper;
    use crate::percept::{random_renaming, rename};

    fn small(np: usize, nl: usize, seed: u64) -> Model {
        let mut rng = Rng::new(seed);
        Model::random(np, nl, Hyper::default(), &mut rng)
    }

    #[test]
    fn cond_matches_closed_form() {
        let mut m = small(1, 1, 3);
        // w[i*n+j] is i -> j: w_01 = 1.0 (node 0 -> node 1), w_10 = -0.5 (node 1 -> node 0)
        m.w = vec![0.0, 1.0, -0.5, 0.0];
        m.hold = 2.0;
        m.beta = 1.0;
        let e = Percept::from_bytes(b"ab");
        let c = Ctx::new(&m, &e, 0);
        assert_eq!(c.a, 2);
        assert_eq!(c.e_sym(0), Some(0));
        assert_eq!(c.e_sym(1), None);
        let mut out = [0.0f64; 2];
        // latent node 1 reads node 0 (symbol 0): field = [1, 0]
        c.cond(1, &[0, 1], &mut out);
        let z = 1.0f64.exp() + 1.0;
        assert!((out[0] - 1.0f64.exp() / z).abs() < 1e-9);
        // perception node 0: bias 2.0 on symbol 0, w_10 = -0.5 on symbol 1 (held by node 1)
        c.cond(0, &[0, 1], &mut out);
        let z = 2.0f64.exp() + (-0.5f64).exp();
        assert!((out[0] - 2.0f64.exp() / z).abs() < 1e-9);
    }

    #[test]
    fn cond_is_equivariant_under_symbol_permutation() {
        let m = small(3, 3, 11);
        let e = Percept::from_bytes(b"abcab");
        let c = Ctx::new(&m, &e, 0);
        let mut c2 = Ctx::new(&m, &e, 0);
        let a = c.a;
        assert_eq!(a, 3);
        let perm: Vec<u16> = vec![2, 0, 1]; // symbol v -> perm[v]
        for b in c2.bias.iter_mut() {
            if let Some(s) = b {
                *s = perm[*s as usize];
            }
        }
        let x: Vec<u16> = vec![0, 1, 2, 2, 1, 0];
        let x2: Vec<u16> = x.iter().map(|&s| perm[s as usize]).collect();
        for j in 0..c.n {
            let mut o1 = vec![0.0f64; a];
            let mut o2 = vec![0.0f64; a];
            c.cond(j, &x, &mut o1);
            c2.cond(j, &x2, &mut o2);
            for v in 0..a {
                assert!((o1[v] - o2[perm[v] as usize]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn trajectory_is_deterministic_and_renaming_invariant() {
        let m = small(4, 4, 5);
        let mut rng = Rng::new(99);
        let perm = random_renaming(&mut rng);
        let raw = b"the cat sat on the mat";
        let e1 = Percept::from_bytes(raw);
        let e2 = Percept::from_bytes(&rename(raw, &perm));
        let t1 = Ctx::new(&m, &e1, 0).trajectory();
        let t2 = Ctx::new(&m, &e2, 0).trajectory();
        assert_eq!(t1.len(), m.hy.budget + 1);
        assert_eq!(t1, t2);
        for s in &t1 {
            for &v in s {
                assert!((v as usize) < e1.a);
            }
        }
    }

    #[test]
    fn noise_is_stratified() {
        let m = small(2, 3, 1);
        let e = Percept::from_bytes(b"abcd"); // a = 4
        let mut c = Ctx::new(&m, &e, 0);
        c.set_draws(8); // two full blocks
        for i in 0..c.n {
            let mut cnt = vec![0usize; c.a];
            for d in 0..8 {
                cnt[c.noise[d * c.n + i] as usize] += 1;
            }
            assert!(cnt.iter().all(|&k| k == 2));
        }
    }

    #[test]
    fn marg_matches_exact_enumeration() {
        let m = small(2, 2, 21);
        let e = Percept::from_bytes(b"abcab"); // a = 3
        let mut c = Ctx::new(&m, &e, 0);
        assert_eq!(c.a, 3);
        c.set_draws(3000);
        let clamp = [0usize, 2usize];
        let syms = [1u16, 2u16];
        for &(j, tau) in &[(2usize, 1usize), (0, 1), (3, 2), (1, 2)] {
            let got = c.marg_vec(j, &clamp, tau, &syms);
            let mut want = vec![0.0f64; c.a];
            assert!(c.marg_exact(j, &clamp, tau, &syms, &mut want));
            for v in 0..c.a {
                assert!(
                    (got[v] - want[v]).abs() < 0.05,
                    "j={} tau={} v={} got={} want={}",
                    j, tau, v, got[v], want[v]
                );
            }
        }
    }

    #[test]
    fn background_pins_are_respected() {
        let m = small(2, 2, 8);
        let e = Percept::from_bytes(b"abab"); // a = 2
        let mut c = Ctx::new(&m, &e, 0);
        c.set_draws(4000);
        c.set_background(&[3], &[1]);
        assert_eq!(c.free_nodes(), &[0usize, 1, 2][..]);
        let clamp = [1usize];
        let syms = [0u16];
        for tau in 1..=2usize {
            let got = c.marg_vec(2, &clamp, tau, &syms);
            let mut want = vec![0.0f64; c.a];
            assert!(c.marg_exact(2, &clamp, tau, &syms, &mut want));
            for v in 0..c.a {
                assert!(
                    (got[v] - want[v]).abs() < 0.05,
                    "tau={} v={} got={} want={}",
                    tau, v, got[v], want[v]
                );
            }
        }
        // a pinned node is a constant
        let pin = c.marg_vec(3, &[], 1, &[]);
        assert_eq!(pin, vec![0.0, 1.0]);
    }

    #[test]
    fn marg_is_cached_and_deterministic() {
        let m = small(3, 3, 4);
        let e = Percept::from_bytes(b"abcabc"); // a = 3
        let c = Ctx::new(&m, &e, 0);
        let o1 = c.marg_vec(4, &[0, 1], 2, &[1, 2]);
        let o2 = c.marg_vec(4, &[1, 0], 2, &[2, 1]); // same clamp, other order
        assert_eq!(o1, o2);
        assert_eq!(c.stats(), (2, 1));
        let s: f64 = o1.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
        assert!(o1.iter().all(|&p| p > 0.0));
    }
}