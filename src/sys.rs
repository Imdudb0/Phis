use crate::model::Model;
use crate::phi::{PhiCfg, PHI_EPS};
use crate::sim::Ctx;
use crate::util::*;
use std::cmp::Ordering;
use std::collections::HashMap;
const REL_TOL: f64 = 1e-12;
#[derive(Clone, Debug)]
pub struct SharpCfg {
    pub phi: PhiCfg,
    pub ncuts: usize,
    pub grain_max: usize,
    pub max_cx: usize,
    pub sys_cap: usize,
    pub exh_cuts: usize,
    pub enum_lim: usize,
    pub seeds: usize,
}
impl SharpCfg {
    pub fn from_model(m: &Model) -> SharpCfg {
        SharpCfg {
            phi: PhiCfg::from_model(m),
            ncuts: m.hy.ncuts,
            grain_max: m.hy.grain_max.max(1),
            max_cx: m.hy.max_cx.max(1),
            sys_cap: 12,
            exh_cuts: 62,
            enum_lim: 4096,
            seeds: 3,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct SysPhi {
    pub phi: f64,
    pub src: u32,
    pub exact: bool,
}
#[derive(Clone, Debug)]
pub struct System {
    pub nodes: Vec<usize>,
    pub phi_s: f64,
    pub cut_src: Vec<usize>,
    pub cut_dst: Vec<usize>,
    pub exact: bool,
}
#[derive(Clone, Debug)]
pub struct Found {
    pub tau: usize,
    pub systems: Vec<System>,
}
type SysCache = HashMap<Vec<usize>, SysPhi>;
fn max_flow(cap0: &[f64], m: usize, s: usize, t: usize) -> (f64, u32) {
    let mut r = cap0.to_vec();
    let mut flow = 0.0f64;
    let mut par = vec![usize::MAX; m];
    let mut queue: Vec<usize> = Vec::with_capacity(m);
    loop {
        par.fill(usize::MAX);
        par[s] = s;
        queue.clear();
        queue.push(s);
        let mut head = 0usize;
        while head < queue.len() && par[t] == usize::MAX {
            let u = queue[head];
            head += 1;
            for v in 0..m {
                if par[v] == usize::MAX && r[u * m + v] > 1e-12 {
                    par[v] = u;
                    queue.push(v);
                }
            }
        }
        if par[t] == usize::MAX {
            let mut mask = 0u32;
            for v in 0..m {
                if par[v] != usize::MAX {
                    mask |= 1u32 << v;
                }
            }
            return (flow, mask);
        }
        let mut b = f64::INFINITY;
        let mut v = t;
        while v != s {
            let u = par[v];
            b = b.min(r[u * m + v]);
            v = u;
        }
        v = t;
        while v != s {
            let u = par[v];
            r[u * m + v] -= b;
            r[v * m + u] += b;
            v = u;
        }
        flow += b;
    }
}
fn max_prod_abs(p: &[Vec<f64>], r: &[Vec<f64>], vals: &[Vec<u16>], lim: usize) -> f64 {
    let d = p.len();
    if d == 0 {
        return 0.0;
    }
    let mut total = 1usize;
    for v in vals {
        total = total.saturating_mul(v.len());
    }
    let mut best = 0.0f64;
    if total <= lim {
        let mut idx = vec![0usize; d];
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
        return best;
    }
    let mut z = vec![0usize; d];
    for start in 0..4usize {
        for y in 0..d {
            let mut pick = 0usize;
            let mut bs = f64::NEG_INFINITY;
            for (k, &v) in vals[y].iter().enumerate() {
                let v = v as usize;
                let sc = match start {
                    0 => p[y][v],
                    1 => r[y][v],
                    2 => -r[y][v],
                    _ => unit(mix3(0xC0FF_EE, y as u64, k as u64)),
                };
                if sc > bs {
                    bs = sc;
                    pick = k;
                }
            }
            z[y] = pick;
        }
        for _sweep in 0..32 {
            let mut improved = false;
            for y in 0..d {
                let mut pr = 1.0f64;
                let mut sr = 0.0f64;
                for u in 0..d {
                    if u != y {
                        let v = vals[u][z[u]] as usize;
                        pr *= p[u][v];
                        sr += r[u][v];
                    }
                }
                let cur = vals[y][z[y]] as usize;
                let mut bv = z[y];
                let mut bval = pr * p[y][cur] * (sr + r[y][cur]).abs();
                for (k, &v) in vals[y].iter().enumerate() {
                    let v = v as usize;
                    let val = pr * p[y][v] * (sr + r[y][v]).abs();
                    if val > bval * (1.0 + REL_TOL) {
                        bval = val;
                        bv = k;
                    }
                }
                if bv != z[y] {
                    z[y] = bv;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        let mut pr = 1.0f64;
        let mut sr = 0.0f64;
        for y in 0..d {
            let v = vals[y][z[y]] as usize;
            pr *= p[y][v];
            sr += r[y][v];
        }
        let val = pr * sr.abs();
        if val > best {
            best = val;
        }
    }
    best
}
struct SysEval<'c, 'm> {
    cx: &'c Ctx<'m>,
    tau: usize,
    k: Vec<usize>,
    sk: Vec<u16>,
    p: Vec<Vec<f64>>,
    vals: Vec<Vec<u16>>,
    pmax: Vec<f64>,
    lim: usize,
    memo: HashMap<u32, f64>,
}
impl<'c, 'm> SysEval<'c, 'm> {
    fn new(cx: &'c Ctx<'m>, tau: usize, state: &[u16], k: &[usize], lim: usize) -> SysEval<'c, 'm> {
        let a = cx.a;
        let amax = (a - 1) as u16;
        let sk: Vec<u16> = k.iter().map(|&j| state[j].min(amax)).collect();
        let mut p: Vec<Vec<f64>> = Vec::with_capacity(k.len());
        let mut vals: Vec<Vec<u16>> = Vec::with_capacity(k.len());
        let mut pmax: Vec<f64> = Vec::with_capacity(k.len());
        for &y in k {
            let py = cx.marg_vec(y, k, tau, &sk);
            let vs: Vec<u16> = match cx.e_sym(y) {
                Some(s) => vec![s],
                None => (0..a as u16).collect(),
            };
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
        SysEval { cx, tau, k: k.to_vec(), sk, p, vals, pmax, lim, memo: HashMap::new() }
    }
    fn cut_value(&mut self, mask: u32) -> f64 {
        if let Some(&v) = self.memo.get(&mask) {
            return v;
        }
        let m = self.k.len();
        let a = self.cx.a;
        let mut fa = 1.0f64;
        let mut bpos: Vec<usize> = Vec::new();
        for i in 0..m {
            if ((mask >> i) & 1) == 1 {
                fa *= self.pmax[i];
            } else {
                bpos.push(i);
            }
        }
        let bnodes: Vec<usize> = bpos.iter().map(|&i| self.k[i]).collect();
        let bsyms: Vec<u16> = bpos.iter().map(|&i| self.sk[i]).collect();
        let mut pb: Vec<Vec<f64>> = Vec::with_capacity(bpos.len());
        let mut rb: Vec<Vec<f64>> = Vec::with_capacity(bpos.len());
        let mut vb: Vec<Vec<u16>> = Vec::with_capacity(bpos.len());
        let mut changed = false;
        let mut q = vec![0.0f64; a];
        for (t, &i) in bpos.iter().enumerate() {
            self.cx.marg(bnodes[t], &bnodes, self.tau, &bsyms, &mut q);
            let mut r = vec![0.0f64; a];
            for v in 0..a {
                r[v] = (self.p[i][v] / q[v]).log2();
                if r[v].abs() > 1e-12 {
                    changed = true;
                }
            }
            pb.push(self.p[i].clone());
            rb.push(r);
            vb.push(self.vals[i].clone());
        }
        let val = if changed { fa * max_prod_abs(&pb, &rb, &vb, self.lim) } else { 0.0 };
        self.memo.insert(mask, val);
        val
    }
    fn eval_cut(&mut self, mk: u32, evald: &mut Vec<(f64, u32)>) -> f64 {
        let fresh = !self.memo.contains_key(&mk);
        let v = self.cut_value(mk);
        if fresh {
            evald.push((v, mk));
        }
        v
    }
    fn min_cuts(&self) -> Vec<u32> {
        let m = self.k.len();
        let mut cap = vec![0.0f64; m * m];
        for i in 0..m {
            for j in 0..m {
                if i != j {
                    cap[i * m + j] = self.cx.infl(self.k[i], self.k[j]);
                }
            }
        }
        let mut found: Vec<(f64, u32)> = Vec::new();
        for t in 1..m {
            found.push(max_flow(&cap, m, 0, t));
            found.push(max_flow(&cap, m, t, 0));
        }
        found.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal).then(x.1.cmp(&y.1)));
        let mut out: Vec<u32> = Vec::new();
        for (_, mk) in found {
            if !out.contains(&mk) {
                out.push(mk);
                if out.len() >= 6 {
                    break;
                }
            }
        }
        out
    }
    fn search(&mut self, seed: u64, ncuts: usize, exh: usize) -> (f64, u32, bool) {
        let m = self.k.len();
        let full: u32 = (1u32 << m) - 1;
        let total = (1usize << m) - 2;
        let mut best = f64::INFINITY;
        let mut bm = 0u32;
        if total <= exh {
            for mk in 1..full {
                let v = self.cut_value(mk);
                if v < best {
                    best = v;
                    bm = mk;
                }
                if best <= PHI_EPS {
                    break;
                }
            }
            return (best, bm, true);
        }
        let mut evald: Vec<(f64, u32)> = Vec::new();
        let mut cand: Vec<u32> = Vec::new();
        for i in 0..m {
            cand.push(1u32 << i);
            cand.push(full ^ (1u32 << i));
        }
        cand.extend(self.min_cuts());
        let mut rng = Rng::new(seed);
        for _ in 0..ncuts {
            let mut mk = 0u32;
            for i in 0..m {
                if rng.chance(0.5) {
                    mk |= 1u32 << i;
                }
            }
            if mk != 0 && mk != full {
                cand.push(mk);
            }
        }
        for mk in cand {
            let v = self.eval_cut(mk, &mut evald);
            if v < best {
                best = v;
                bm = mk;
            }
            if best <= PHI_EPS {
                return (best, bm, false);
            }
        }
        evald.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal).then(x.1.cmp(&y.1)));
        let starts: Vec<u32> = evald.iter().take(3).map(|e| e.1).collect();
        let mut budget = 4 * m + 32;
        for s0 in starts {
            let mut cur = s0;
            let mut cv = self.cut_value(cur);
            loop {
                let mut improved = false;
                for i in 0..m {
                    if budget == 0 {
                        break;
                    }
                    let nm = cur ^ (1u32 << i);
                    if nm == 0 || nm == full {
                        continue;
                    }
                    budget -= 1;
                    let v = self.cut_value(nm);
                    if v < cv * (1.0 - REL_TOL) {
                        cur = nm;
                        cv = v;
                        improved = true;
                    }
                    if cv <= PHI_EPS {
                        break;
                    }
                }
                if !improved || budget == 0 || cv <= PHI_EPS {
                    break;
                }
            }
            if cv < best {
                best = cv;
                bm = cur;
            }
            if best <= PHI_EPS {
                break;
            }
        }
        (best, bm, false)
    }
}
pub fn system_phi(cx: &mut Ctx<'_>, state: &[u16], tau: usize, cfg: &SharpCfg, k: &[usize]) -> SysPhi {
    assert!(k.len() <= 31, "a candidate system has at most 31 nodes");
    if k.len() < 2 {
        return SysPhi { phi: 0.0, src: 0, exact: true };
    }
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
    let seed = hash_usize(mix(cx.seed(), tau as u64), k);
    let (phi, src, exact) = {
        let mut ev = SysEval::new(&*cx, tau, state, k, cfg.enum_lim);
        ev.search(seed, cfg.ncuts, cfg.exh_cuts)
    };
    SysPhi { phi: if phi.is_finite() { phi.max(0.0) } else { 0.0 }, src, exact }
}
pub fn split_mask(k: &[usize], mask: u32) -> (Vec<usize>, Vec<usize>) {
    let mut a: Vec<usize> = Vec::new();
    let mut b: Vec<usize> = Vec::new();
    for (i, &j) in k.iter().enumerate() {
        if ((mask >> i) & 1) == 1 {
            a.push(j);
        } else {
            b.push(j);
        }
    }
    (a, b)
}
fn consider(
    cx: &mut Ctx<'_>,
    state: &[u16],
    tau: usize,
    cfg: &SharpCfg,
    cache: &mut SysCache,
    seen: &mut Vec<(Vec<usize>, SysPhi)>,
    k: Vec<usize>,
) -> f64 {
    if k.len() < 2 {
        return 0.0;
    }
    if let Some(s) = seen.iter().find(|s| s.0 == k) {
        return s.1.phi;
    }
    let cached: Option<SysPhi> = cache.get(&k).copied();
    let sp = match cached {
        Some(v) => v,
        None => {
            let v = system_phi(cx, state, tau, cfg, &k);
            cache.insert(k.clone(), v);
            v
        }
    };
    seen.push((k, sp));
    sp.phi
}
fn scc_candidates(cx: &Ctx<'_>, rem: &[usize], cap: usize) -> Vec<Vec<usize>> {
    let m = rem.len();
    let mut vals: Vec<f64> = Vec::new();
    for x in 0..m {
        for y in 0..m {
            if x != y {
                let c = cx.infl(rem[x], rem[y]);
                if c > 0.0 {
                    vals.push(c);
                }
            }
        }
    }
    let mut out: Vec<Vec<usize>> = Vec::new();
    if vals.is_empty() {
        return out;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    for &q in &[0.0f64, 0.25, 0.5, 0.75] {
        let thr = vals[((vals.len() - 1) as f64 * q) as usize];
        let mut reach = vec![false; m * m];
        for x in 0..m {
            reach[x * m + x] = true;
            for y in 0..m {
                if x != y {
                    let c = cx.infl(rem[x], rem[y]);
                    if c > 0.0 && c >= thr {
                        reach[x * m + y] = true;
                    }
                }
            }
        }
        for k in 0..m {
            for x in 0..m {
                if reach[x * m + k] {
                    for y in 0..m {
                        if reach[k * m + y] {
                            reach[x * m + y] = true;
                        }
                    }
                }
            }
        }
        let mut done = vec![false; m];
        for x in 0..m {
            if done[x] {
                continue;
            }
            let mut comp: Vec<usize> = Vec::new();
            for y in 0..m {
                if reach[x * m + y] && reach[y * m + x] {
                    comp.push(y);
                    done[y] = true;
                }
            }
            if comp.len() < 2 {
                continue;
            }
            if comp.len() > cap {
                let mut sc: Vec<(usize, f64)> = Vec::new();
                for &y in &comp {
                    let mut s = 0.0f64;
                    for &z in &comp {
                        if z != y {
                            s += cx.infl(rem[y], rem[z]) + cx.infl(rem[z], rem[y]);
                        }
                    }
                    sc.push((y, s));
                }
                sc.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0)));
                sc.truncate(cap);
                comp = sc.iter().map(|p| p.0).collect();
                comp.sort_unstable();
            }
            out.push(comp.iter().map(|&y| rem[y]).collect());
        }
    }
    out
}
fn best_system(
    cx: &mut Ctx<'_>,
    state: &[u16],
    tau: usize,
    cfg: &SharpCfg,
    cache: &mut SysCache,
    rem: &[usize],
) -> Option<(Vec<usize>, SysPhi)> {
    if rem.len() < 2 {
        return None;
    }
    let cap = cfg.sys_cap.clamp(2, 24);
    let mut seen: Vec<(Vec<usize>, SysPhi)> = Vec::new();
    for x in 0..rem.len() {
        for y in (x + 1)..rem.len() {
            let (i, j) = (rem[x], rem[y]);
            if cx.weight(i, j) != 0.0 && cx.weight(j, i) != 0.0 {
                consider(cx, state, tau, cfg, cache, &mut seen, vec![i, j]);
            }
        }
    }
    let clusters = scc_candidates(&*cx, rem, cap);
    for k in clusters {
        consider(cx, state, tau, cfg, cache, &mut seen, k);
    }
    let mut starts: Vec<(Vec<usize>, f64)> = seen.iter().map(|s| (s.0.clone(), s.1.phi)).collect();
    starts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    starts.truncate(cfg.seeds.max(1));
    for (k0, v0) in starts {
        if v0 <= PHI_EPS {
            continue;
        }
        let mut cur = k0;
        let mut cv = v0;
        for _ in 0..cap {
            let mut moves: Vec<Vec<usize>> = Vec::new();
            if cur.len() < cap {
                for &x in rem {
                    if cur.contains(&x) {
                        continue;
                    }
                    let into = cur.iter().any(|&c| cx.weight(c, x) != 0.0);
                    let outof = cur.iter().any(|&c| cx.weight(x, c) != 0.0);
                    if into && outof {
                        let mut nk = cur.clone();
                        nk.push(x);
                        nk.sort_unstable();
                        moves.push(nk);
                    }
                }
            }
            if cur.len() > 2 {
                for i in 0..cur.len() {
                    let mut nk = cur.clone();
                    nk.remove(i);
                    moves.push(nk);
                }
            }
            let mut bm: Option<(Vec<usize>, f64)> = None;
            for nk in moves {
                let v = consider(cx, state, tau, cfg, cache, &mut seen, nk.clone());
                let better = match &bm {
                    None => true,
                    Some(b) => v > b.1,
                };
                if better {
                    bm = Some((nk, v));
                }
            }
            match bm {
                Some((nk, v)) if v > cv * (1.0 + REL_TOL) => {
                    cur = nk;
                    cv = v;
                }
                _ => break,
            }
        }
    }
    let mut best: Option<(Vec<usize>, SysPhi)> = None;
    for s in seen {
        let better = match &best {
            None => true,
            Some(b) => s.1.phi > b.1.phi * (1.0 + REL_TOL),
        };
        if better {
            best = Some(s);
        }
    }
    match best {
        Some(b) if b.1.phi > PHI_EPS => Some(b),
        _ => None,
    }
}
pub fn find_complexes(cx: &mut Ctx<'_>, state: &[u16], cfg: &SharpCfg) -> Found {
    let n = cx.n;
    let gm = cfg.grain_max.max(1);
    let all: Vec<usize> = (0..n).collect();
    let mut caches: Vec<SysCache> = (0..=gm).map(|_| HashMap::new()).collect();
    let mut tau_star = 1usize;
    let mut first: Option<(Vec<usize>, SysPhi)> = None;
    for tau in 1..=gm {
        if let Some(c) = best_system(cx, state, tau, cfg, &mut caches[tau], &all) {
            let better = match &first {
                None => true,
                Some(f) => c.1.phi > f.1.phi * (1.0 + REL_TOL),
            };
            if better {
                tau_star = tau;
                first = Some(c);
            }
        }
    }
    let mut chosen: Vec<(Vec<usize>, SysPhi)> = Vec::new();
    if let Some(f) = first {
        let mut remaining: Vec<usize> = all.iter().copied().filter(|j| !f.0.contains(j)).collect();
        chosen.push(f);
        while chosen.len() < cfg.max_cx.max(1) && remaining.len() >= 2 {
            match best_system(cx, state, tau_star, cfg, &mut caches[tau_star], &remaining) {
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
        .map(|(k, sp)| {
            let (a, b) = split_mask(&k, sp.src);
            System { nodes: k, phi_s: sp.phi, cut_src: a, cut_dst: b, exact: sp.exact }
        })
        .collect();
    Found { tau: tau_star, systems }
}
