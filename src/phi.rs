use crate::model::Model;
use crate::sim::{Ctx, IdBuild};
use crate::util::{hash_usize, show_byte};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
pub const PHI_EPS: f64 = 1e-9;
const TIE_REL: f64 = 1e-9;
pub const MAX_TIES: usize = 4;
const TINY: f64 = 1e-300;
const CACHE_CAP: usize = 4_000_000;
const SEP: usize = usize::MAX / 2;
const TAG_CTAB: u64 = 0x4354_4142_0000_0007;
#[inline]
pub fn ipow(a: usize, k: usize) -> usize {
    let mut r = 1usize;
    for _ in 0..k {
        r = r.saturating_mul(a);
    }
    r
}
#[inline]
pub fn id_term(p: f64, q: f64) -> f64 {
    p * (p.max(TINY) / q.max(TINY)).log2().abs()
}
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
pub fn decode(mut idx: usize, a: usize, out: &mut [u16]) {
    for o in out.iter_mut() {
        *o = (idx % a) as u16;
        idx /= a;
    }
}
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Cause,
    Effect,
}
#[derive(Clone, Debug)]
pub struct PhiCfg {
    pub kmax: usize,
    pub qmax: usize,
    pub nbr: usize,
    pub max_states: usize,
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
#[derive(Clone, Debug)]
pub struct Face {
    pub nodes: Vec<usize>,
    pub phi: f64,
    pub states: Vec<Vec<u16>>,
}
#[derive(Clone, Debug)]
pub struct Dist {
    pub mech: Vec<usize>,
    pub mech_state: Vec<u16>,
    pub cause: Vec<Face>,
    pub effect: Vec<Face>,
    pub phi_c: f64,
    pub phi_e: f64,
    pub phi_d: f64,
    pub anchored: bool,
}
impl Dist {
    pub fn is_distinction(&self) -> bool {
        self.phi_d > PHI_EPS
    }
}
pub fn sum_phi_d(ds: &[Dist]) -> f64 {
    ds.iter()
        .filter(|d| d.anchored && d.is_distinction())
        .map(|d| d.phi_d)
        .sum()
}
struct SideEval {
    phi: f64,
    states: Vec<Vec<u16>>,
    exact: bool,
}
pub struct Phi<'c, 'm> {
    pub cx: &'c Ctx<'m>,
    pub tau: usize,
    pub state: Vec<u16>,
    pub cfg: PhiCfg,
    nin: Vec<Vec<usize>>,
    nout: Vec<Vec<usize>>,
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
                let wi = cx.weight(i, j).abs(); 
                if wi > 0.0 {
                    vi.push((i, wi));
                }
                let wo = cx.weight(j, i).abs(); 
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
    pub fn standard(cx: &'c Ctx<'m>, tau: usize, state: &[u16]) -> Phi<'c, 'm> {
        Phi::new(cx, tau, state, PhiCfg::from_model(cx.m))
    }
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
                continue; 
            }
            if mb.is_empty() {
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
    fn effect_eval(&self, m: &[usize], z: &[usize], floor: f64) -> SideEval {
        let a = self.cx.a;
        let km = m.len();
        let kz = z.len();
        let sm: Vec<u16> = m.iter().map(|&j| self.state[j]).collect();
        let pz: Vec<Vec<f64>> = z
            .iter()
            .map(|&y| self.cx.marg_vec(y, m, self.tau, &sm))
            .collect();
        let fixed: Vec<Option<u16>> = z.iter().map(|&y| self.cx.e_sym(y)).collect();
        let lat: Vec<usize> = (0..kz).filter(|&i| fixed[i].is_none()).collect();
        let nstate = ipow(a, lat.len());
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
    pub fn phi_cause(&self, m: &[usize], z: &[usize]) -> (f64, Vec<Vec<u16>>) {
        let (m, z) = (norm(m), norm(z));
        if m.is_empty() || z.is_empty() {
            return (0.0, Vec::new());
        }
        debug_assert!(m.iter().chain(z.iter()).all(|&i| self.cx.pinned(i).is_none()));
        let ev = self.cause_eval(&m, &z, -1.0);
        if ev.phi <= PHI_EPS { (0.0, Vec::new()) } else { (ev.phi, ev.states) }
    }
    pub fn phi_effect(&self, m: &[usize], z: &[usize]) -> (f64, Vec<Vec<u16>>) {
        let (m, z) = (norm(m), norm(z));
        if m.is_empty() || z.is_empty() {
            return (0.0, Vec::new());
        }
        debug_assert!(m.iter().chain(z.iter()).all(|&i| self.cx.pinned(i).is_none()));
        let ev = self.effect_eval(&m, &z, -1.0);
        if ev.phi <= PHI_EPS { (0.0, Vec::new()) } else { (ev.phi, ev.states) }
    }
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
    pub fn dist_e(&self) -> Vec<Dist> {
        self.distinctions(true)
            .into_iter()
            .filter(|d| d.anchored && d.is_distinction())
            .collect()
    }
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
