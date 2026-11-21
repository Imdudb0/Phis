use crate::model::{Model, WEPS};
use crate::percept::Percept;
use crate::util::*;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
pub const PFLOOR: f64 = 1e-12;
const TAG_NOISE: u64 = 0x4E4F_4953_4500_0001;
const TAG_STEP: u64 = 0x5354_4550_0000_0002;
const TAG_TRAJ: u64 = 0x5452_414A_0000_0003;
const TAG_INIT: u64 = 0x494E_4954_0000_0004;
const TAG_KEY: u64 = 0x4B45_5900_0000_0005;
const TAG_BG: u64 = 0x4247_0000_0000_0006;
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
pub struct Ctx<'a> {
    pub m: &'a Model,
    pub n: usize,
    pub a: usize,
    pub alphabet: Vec<u8>,
    pub frame: (usize, usize),
    pub inn: Vec<Vec<usize>>,
    pub outn: Vec<Vec<usize>>,
    bias: Vec<Option<u16>>,
    seed: u64,
    draws: usize,
    win: Vec<f64>,
    pin: Vec<Option<u16>>,
    free: Vec<usize>,
    bgf: Vec<f64>,
    bg_id: u64,
    noise: Vec<u16>,
    cache: RefCell<HashMap<u64, Box<[f64]>, IdBuild>>,
    scratch: RefCell<Scratch>,
    calls: Cell<u64>,
    hits: Cell<u64>,
}
impl<'a> Ctx<'a> {
    pub fn n_frames(m: &Model, e: &Percept) -> usize {
        e.frames(m.np).len().max(1)
    }
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
                let wi = win[k * n + o].abs(); 
                if wi > 0.0 {
                    vi.push((o, wi));
                }
                let wo = win[o * n + k].abs(); 
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
    pub fn set_draws(&mut self, r: usize) {
        self.draws = r.max(1);
        self.build_noise();
        self.cache.get_mut().clear();
    }
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
    pub fn draws(&self) -> usize {
        self.draws
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub fn free_nodes(&self) -> &[usize] {
        &self.free
    }
    pub fn pinned(&self, i: usize) -> Option<u16> {
        self.pin[i]
    }
    pub fn anchored(&self, j: usize) -> bool {
        self.bias[j].is_some()
    }
    pub fn e_sym(&self, j: usize) -> Option<u16> {
        self.bias[j]
    }
    pub fn e_pos(&self, j: usize) -> Option<usize> {
        if self.bias[j].is_some() { Some(self.frame.0 + j) } else { None }
    }
    pub fn byte_of(&self, j: usize) -> Option<u8> {
        self.bias[j].map(|c| self.alphabet[c as usize])
    }
    pub fn touches_e(&self, nodes: &[usize]) -> bool {
        nodes.iter().any(|&i| self.bias[i].is_some())
    }
    pub fn weight(&self, i: usize, j: usize) -> f64 {
        self.win[j * self.n + i]
    }
    pub fn infl(&self, i: usize, j: usize) -> f64 {
        self.m.beta * self.weight(i, j).abs()
    }
    pub fn stats(&self) -> (u64, u64) {
        (self.calls.get(), self.hits.get())
    }
    pub fn cache_len(&self) -> usize {
        self.cache.borrow().len()
    }
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
    fn step_free(&self, x: &[u16], y: &mut [u16], ds: u64, t: u64, f: &mut [f64], p: &mut [f64]) {
        let beta = self.m.beta;
        for &j in &self.free {
            self.field_free(j, x, f);
            softmax_into(f, beta, p);
            y[j] = sample(p, unit(mix3(ds, t, j as u64)));
        }
    }
    pub fn cond(&self, j: usize, x: &[u16], out: &mut [f64]) {
        let mut f = vec![0.0f64; self.a];
        self.field_all(j, x, &mut f);
        softmax_into(&f, self.m.beta, &mut out[..self.a]);
    }
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
    pub fn step_all(&self, x: &[u16], t: u64, y: &mut [u16]) {
        let mut f = vec![0.0f64; self.a];
        let mut p = vec![0.0f64; self.a];
        for j in 0..self.n {
            self.field_all(j, x, &mut f);
            softmax_into(&f, self.m.beta, &mut p);
            y[j] = sample(&p, unit(mix3(self.seed ^ TAG_TRAJ, t, j as u64)));
        }
    }
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
