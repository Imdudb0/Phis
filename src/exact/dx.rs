use super::{norm, subsets, ExactCfg, Mg, TIE_REL};
use crate::phi::{decode, id_max, id_term, ipow, set_partitions, Dir, Dist, Face, MAX_TIES, PHI_EPS};
use crate::resid::{norm_entropy, Residual, MASS_EPS};
use crate::sim::Ctx;
use crate::util::show_byte;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

const TINY: f64 = 1e-300;

fn sub_index(idx: usize, a: usize, pos: &[usize], apow: &[usize]) -> usize {
    let mut s = 0usize;
    let mut mul = 1usize;
    for &p in pos {
        s += ((idx / apow[p]) % a) * mul;
        mul *= a;
    }
    s
}

fn ids(v: &[usize]) -> String {
    v.iter().map(|j| j.to_string()).collect::<Vec<String>>().join(",")
}

/// Exact distinctions of one complex (the Ctx background must already pin everything else).
pub struct Dx<'c, 'm> {
    cx: &'c Ctx<'m>,
    cfg: ExactCfg,
    tau: usize,
    state: Vec<u16>,
    mg: Mg<'c, 'm>,
    ct: RefCell<HashMap<(Vec<usize>, Vec<usize>), Rc<Vec<f64>>>>,
    pt: RefCell<HashMap<(usize, usize), Rc<Vec<Vec<u8>>>>>,
}

impl<'c, 'm> Dx<'c, 'm> {
    pub fn new(cx: &'c Ctx<'m>, tau: usize, state: &[u16], cfg: &ExactCfg) -> Dx<'c, 'm> {
        let n = cx.n;
        assert!(state.len() >= n, "state must hold one symbol per node");
        let amax = (cx.a - 1) as u16;
        let st: Vec<u16> = state[..n].iter().map(|&s| s.min(amax)).collect();
        Dx {
            cx,
            cfg: cfg.clone(),
            tau: tau.max(1),
            state: st,
            mg: Mg::new(cx, cfg.backend, tau),
            ct: RefCell::new(HashMap::new()),
            pt: RefCell::new(HashMap::new()),
        }
    }

    pub fn tau(&self) -> usize {
        self.tau
    }

    /// true if some repertoire fell back to Monte-Carlo (exact backend only)
    pub fn inexact(&self) -> bool {
        self.mg.bad.get()
    }

    fn parts(&self, km: usize, kz: usize) -> Rc<Vec<Vec<u8>>> {
        let hit = self.pt.borrow().get(&(km, kz)).cloned();
        if let Some(p) = hit {
            return p;
        }
        let p = Rc::new(set_partitions(km, kz));
        self.pt.borrow_mut().insert((km, kz), Rc::clone(&p));
        p
    }

    /// Every mechanism of the complex up to kmax (0 = all): connected or not.
    pub fn mechanisms(&self) -> Vec<Vec<usize>> {
        let free = self.cx.free_nodes();
        let kmax = if self.cfg.kmax == 0 { free.len() } else { self.cfg.kmax.min(free.len()) };
        let mut all = subsets(free, 1, kmax);
        all.truncate(self.cfg.max_mech);
        all
    }

    /// Every purview of the complex up to qmax (0 = all), capped by |A(E)|^size.
    pub fn purviews(&self) -> Vec<Vec<usize>> {
        let free = self.cx.free_nodes();
        let a = self.cx.a;
        let mut qm = if self.cfg.qmax == 0 { free.len() } else { self.cfg.qmax.min(free.len()) };
        while qm > 0 && ipow(a, qm) > self.cfg.max_states {
            qm -= 1;
        }
        subsets(free, 1, qm)
    }

    fn cause_table(&self, m: &[usize], z: &[usize]) -> Rc<Vec<f64>> {
        let key = (m.to_vec(), z.to_vec());
        let hit = self.ct.borrow().get(&key).cloned();
        if let Some(t) = hit {
            return t;
        }
        let a = self.cx.a;
        let k = z.len();
        let total = ipow(a, k);
        let sm: Vec<u16> = m.iter().map(|&j| self.state[j]).collect();
        let mut tab = vec![0.0f64; total];
        let mut zs = vec![0u16; k];
        let mut sum = 0.0f64;
        for idx in 0..total {
            decode(idx, a, &mut zs);
            let mut l = 1.0f64;
            for (t, &j) in m.iter().enumerate() {
                let out = self.mg.get(j, z, &zs);
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
        self.ct.borrow_mut().insert(key, Rc::clone(&rc));
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

    /// phi_c(m, z): min over every disintegrating partition of ID(pi_c, pi_c^psi).
    fn cause_eval(&self, m: &[usize], z: &[usize]) -> (f64, Vec<Vec<u16>>) {
        let a = self.cx.a;
        let km = m.len();
        let kz = z.len();
        let total = ipow(a, kz);
        let p = self.cause_table(m, z);
        let apow: Vec<usize> = (0..kz).map(|i| ipow(a, i)).collect();
        let parts = self.parts(km, kz);
        let mut q = vec![0.0f64; total];
        let mut best = f64::INFINITY;
        let mut best_q: Vec<f64> = Vec::new();
        for lab in parts.iter() {
            self.partitioned_cause(m, z, lab, &apow, &mut q);
            let (v, _) = id_max(&p, &q);
            if v < best {
                best = v;
                best_q.clear();
                best_q.extend_from_slice(&q);
            }
            if best <= PHI_EPS {
                return (best, Vec::new());
            }
        }
        if !best.is_finite() {
            return (0.0, Vec::new());
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
        (best, states)
    }

    /// phi_e(m, z): purview nodes carrying bytes of E are evaluated at E, not at the argmax.
    fn effect_eval(&self, m: &[usize], z: &[usize]) -> (f64, Vec<Vec<u16>>) {
        let a = self.cx.a;
        let km = m.len();
        let kz = z.len();
        let sm: Vec<u16> = m.iter().map(|&j| self.state[j]).collect();
        let pz: Vec<Rc<Vec<f64>>> = z.iter().map(|&y| self.mg.get(y, m, &sm)).collect();
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
        let parts = self.parts(km, kz);
        let mut qc = vec![0.0f64; nstate];
        let mut best = f64::INFINITY;
        let mut best_q: Vec<f64> = Vec::new();
        for lab in parts.iter() {
            let mut qy: Vec<Rc<Vec<f64>>> = Vec::with_capacity(kz);
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
                qy.push(self.mg.get(z[i], &mb, &smb));
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
                return (best, Vec::new());
            }
        }
        if !best.is_finite() {
            return (0.0, Vec::new());
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
        (best, states)
    }

    /// Single (mechanism, purview) values, to compare with `Phi::phi_cause` / `phi_effect`.
    pub fn phi_cause(&self, m: &[usize], z: &[usize]) -> f64 {
        let (m, z) = (norm(m), norm(z));
        if m.is_empty() || z.is_empty() {
            return 0.0;
        }
        let v = self.cause_eval(&m, &z).0;
        if v <= PHI_EPS { 0.0 } else { v }
    }

    pub fn phi_effect(&self, m: &[usize], z: &[usize]) -> f64 {
        let (m, z) = (norm(m), norm(z));
        if m.is_empty() || z.is_empty() {
            return 0.0;
        }
        let v = self.effect_eval(&m, &z).0;
        if v <= PHI_EPS { 0.0 } else { v }
    }

    fn best_side(&self, m: &[usize], cands: &[Vec<usize>], dir: Dir) -> (f64, Vec<Face>) {
        let mut best = 0.0f64;
        let mut faces: Vec<Face> = Vec::new();
        for z in cands {
            let (phi, states) = match dir {
                Dir::Cause => self.cause_eval(m, z),
                Dir::Effect => self.effect_eval(m, z),
            };
            if phi <= PHI_EPS {
                continue;
            }
            let tol = TIE_REL * best.max(phi).max(1.0);
            if phi > best + tol {
                best = phi;
                faces.clear();
                faces.push(Face { nodes: z.clone(), phi, states });
            } else if phi >= best - tol {
                if faces.len() < MAX_TIES {
                    faces.push(Face { nodes: z.clone(), phi, states });
                }
                if phi > best {
                    best = phi;
                }
            }
        }
        (best, faces)
    }

    /// Every (anchored or not) distinction and half-distinction of the complex;
    /// the caller keeps `anchored`, like `sharp::structure` does.
    pub fn distinctions(&self) -> Vec<Dist> {
        let cands = self.purviews();
        let mut out: Vec<Dist> = Vec::new();
        for m in self.mechanisms() {
            let (phi_c, fc) = self.best_side(&m, &cands, Dir::Cause);
            let (phi_e, fe) = self.best_side(&m, &cands, Dir::Effect);
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

    /// Bytes of E (global positions) touched by the purviews of d.
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
                    let r = self.mg.get(y, m, &sm);
                    out[i].copy_from_slice(&r);
                }
            }
            Dir::Cause => {
                let p = self.cause_table(m, z);
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

    fn sym(&self, v: u16) -> String {
        match self.cx.alphabet.get(v as usize) {
            Some(&b) => show_byte(b),
            None => "?".to_string(),
        }
    }

    fn fmt_face(&self, f: &Face) -> String {
        let mut s = format!("{{{}}}=", ids(&f.nodes));
        for (k, st) in f.states.iter().enumerate() {
            if k > 0 {
                s.push('|');
            }
            for &v in st {
                s.push_str(&self.sym(v));
            }
        }
        s.push_str(&format!(" phi={:.4}", f.phi));
        s
    }

    pub fn describe(&self, d: &Dist) -> String {
        let mut s = format!(
            "m{{{}}} phi_d={:.4} (c={:.4} e={:.4}){}",
            ids(&d.mech),
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

/// Exact residual, same rules as resid::residuals:
/// rho = |phi_c - phi_e|, located on the nodes of the weak side's purviews
/// (those of the strong side if the weak side has none), weighted by their repertoire entropy.
pub fn residuals_exact(dx: &Dx<'_, '_>, dists: &[Dist]) -> Vec<Residual> {
    let mut out: Vec<Residual> = Vec::new();
    for d in dists {
        let rho = (d.phi_c - d.phi_e).abs();
        if rho <= PHI_EPS {
            continue;
        }
        let weak = if d.phi_c < d.phi_e { Dir::Cause } else { Dir::Effect };
        let (wf, sf) = match weak {
            Dir::Cause => (&d.cause, &d.effect),
            Dir::Effect => (&d.effect, &d.cause),
        };
        let from_strong = wf.is_empty();
        let faces = if from_strong { sf } else { wf };
        let ties: usize = wf.iter().map(|f| f.states.len()).sum();
        let mut acc: Vec<(usize, f64)> = Vec::new();
        for f in faces {
            let rows = dx.repertoire_marginals(&d.mech, &f.nodes, weak);
            for (i, &j) in f.nodes.iter().enumerate() {
                let h = norm_entropy(&rows[i]);
                match acc.iter().position(|x| x.0 == j) {
                    Some(p) => {
                        if h > acc[p].1 {
                            acc[p].1 = h;
                        }
                    }
                    None => acc.push((j, h)),
                }
            }
        }
        acc.sort_by_key(|x| x.0);
        let tot: f64 = acc.iter().map(|x| x.1).sum();
        let k = acc.len().max(1) as f64;
        let nodes: Vec<(usize, f64)> = acc
            .iter()
            .map(|&(j, h)| (j, rho * if tot > MASS_EPS { h / tot } else { 1.0 / k }))
            .collect();
        out.push(Residual { mech: d.mech.clone(), weak, rho, ties, nodes, from_strong });
    }
    out
}