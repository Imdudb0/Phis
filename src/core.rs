//! Perception plastique, contrôleur de routage, recherche abductive COMPLÈTE.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

use crate::algebra::{values, Builder, Cause, Node, IDENT, Op};
use crate::phi::{phi, underdetermined, PhiInfo};
use crate::rng::{NormalRng, PyRandom};

#[derive(Clone, Debug)]
pub struct Cfg {
    pub lmax: usize,
    pub max_atoms_per_item: usize,
    pub pops: usize,
    pub completes: usize,
    pub phi_n: usize,
    pub phi_n_final: usize,
    pub lift_cap: usize,
    pub d: usize,
    pub h: usize,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            lmax: 24,
            max_atoms_per_item: 6,
            pops: 3000,
            completes: 8,
            phi_n: 12,
            phi_n_final: 64,
            lift_cap: 6,
            d: 12,
            h: 10,
        }
    }
}

pub fn pattern(bs: &[u8]) -> Vec<usize> {
    let mut seen: HashMap<u8, usize> = HashMap::new();
    let mut out = Vec::with_capacity(bs.len());
    for &b in bs {
        let len = seen.len();
        let rank = *seen.entry(b).or_insert(len);
        out.push(rank);
    }
    out
}

#[derive(Clone, Debug)]
pub struct Item {
    pub i: usize,
    pub j: usize,
    pub kind: Op,
    pub lit: Vec<u8>,
    pub k: usize,
    pub a0: usize,
    pub a1: usize,
}

#[inline]
fn find_subslice(haystack: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    if needle.is_empty() {
        return if start <= haystack.len() { Some(start) } else { None };
    }
    if start + needle.len() > haystack.len() {
        return None;
    }
    haystack[start..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|pos| start + pos)
}

#[inline]
fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    find_subslice(haystack, needle, 0).is_some()
}

/// R régional : tout est dérivé de E. Plasticité locale à l'inférence.
pub struct Perception {
    pub e: Vec<u8>,
    pub n: usize,
    pub cfg: Cfg,
    pub alpha: Vec<u8>,
    pub f: HashMap<(u8, u8), f64>,
    pub b: HashMap<(u8, u8), f64>,
    pub sur: Vec<f64>,
    pub counts: HashMap<Vec<u8>, usize>,
    count_order: Vec<Vec<u8>>,
    pub maxreps: Vec<Vec<u8>>,
    pub atoms: Vec<(usize, usize)>,
    pub items: Vec<Item>,
    pub by_start: HashMap<usize, Vec<usize>>,
    pub item_at: HashMap<(usize, usize), usize>,
    pub partner: HashMap<usize, Vec<usize>>,
    pub earlier: HashMap<usize, Vec<usize>>,
    pub later: HashMap<usize, Vec<usize>>,
    pub sep: Option<u8>,
    pub records: Vec<(usize, usize)>,
    pub frame: Vec<Vec<u8>>,
    pub slots: Vec<(usize, usize, usize, usize)>,
    pub fmat: Vec<[f64; 12]>,
    pub pf: Vec<[f64; 12]>,
    pub pb: Vec<[f64; 12]>,
    frames_ready: bool
}

impl Perception {
    pub fn new(e: Vec<u8>, cfg: &Cfg) -> Self {
        let n = e.len();
        let mut present = [false; 256];
        for &byte in &e {
            present[byte as usize] = true;
        }
        let alpha: Vec<u8> = (0u8..=255)
            .filter(|&byte| present[byte as usize])
            .collect();

        let mut per = Self {
            e,
            n,
            cfg: cfg.clone(),
            alpha,
            f: HashMap::new(),
            b: HashMap::new(),
            sur: Vec::new(),
            counts: HashMap::new(),
            count_order: Vec::new(),
            maxreps: Vec::new(),
            atoms: Vec::new(),
            items: Vec::new(),
            by_start: HashMap::new(),
            item_at: HashMap::new(),
            partner: HashMap::new(),
            earlier: HashMap::new(),
            later: HashMap::new(),
            sep: None,
            records: Vec::new(),
            frame: Vec::new(),
            slots: Vec::new(),
            fmat: Vec::new(),
            pf: Vec::new(),
            pb: Vec::new(),
            frames_ready: false,
        };
        per.plastic();
        per.compute_counts();
        per.compute_atoms();
        per.compute_items();
        per.compute_partners();
        per.compute_features();
        per
    }

    pub fn ensure_frames(&mut self) {
        if !self.frames_ready {
            self.compute_frames();
            self.frames_ready = true;
        }
    }

    fn plastic(&mut self) {
        let n = self.n;
        let mut f = HashMap::new();
        let mut b = HashMap::new();
        let mut sur = vec![0.0; n];
        for i in 0..n {
            let a = if i > 0 { self.e[i - 1] } else { 0 };
            let byte = self.e[i];
            let p = *f.get(&(a, byte)).unwrap_or(&0.0);
            let s = 1.0 - p;
            sur[i] = s;
            f.insert((a, byte), p + 0.25 * (1.0 - p));
            let q = *b.get(&(byte, a)).unwrap_or(&0.0);
            b.insert((byte, a), q + 0.15 * s * (1.0 - q));
        }
        self.f = f;
        self.b = b;
        self.sur = sur;
    }

    fn compute_counts(&mut self) {
        let n = self.n;
        let l = self.cfg.lmax;
        let mut c: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut order = Vec::new();
        for i in 0..n {
            let j_max = n.min(i + l);
            for j in (i + 1)..=j_max {
                let s = self.e[i..j].to_vec();
                let entry = c.entry(s.clone()).or_insert_with(|| {
                    order.push(s);
                    0
                });
                *entry += 1;
            }
        }
        self.counts = c;
        self.count_order = order;
    }

    fn compute_atoms(&mut self) {
        let n = self.n;
        let mut maximal = Vec::new();
        for s in &self.count_order {
            let cnt = self.counts.get(s).copied().unwrap_or(0);
            if cnt < 2 {
                continue;
            }
            let left_ext = self.alpha.iter().any(|&a| {
                let mut ext = Vec::with_capacity(s.len() + 1);
                ext.push(a);
                ext.extend_from_slice(s);
                self.counts.get(&ext).copied().unwrap_or(0) >= 2
            });
            if left_ext {
                continue;
            }
            let right_ext = self.alpha.iter().any(|&b| {
                let mut ext = Vec::with_capacity(s.len() + 1);
                ext.extend_from_slice(s);
                ext.push(b);
                self.counts.get(&ext).copied().unwrap_or(0) >= 2
            });
            if right_ext {
                continue;
            }
            maximal.push(s.clone());
        }

        let mut marked = vec![0u8; n];
        let mut longest_at: HashMap<usize, usize> = HashMap::new();
        for s in &maximal {
            let mut st = 0usize;
            while let Some(k) = find_subslice(&self.e, s, st) {
                for t in k..(k + s.len()) {
                    marked[t] = 1;
                }
                let prev = longest_at.get(&k).copied().unwrap_or(0);
                if s.len() > prev {
                    longest_at.insert(k, s.len());
                }
                st = k + 1;
            }
        }
        self.maxreps = maximal;

        let mut atoms = Vec::new();
        let mut i = 0usize;
        while i < n {
            let l_at = longest_at.get(&i).copied().unwrap_or(0);
            if marked[i] != 0 && l_at > 0 {
                atoms.push((i, i + l_at));
                i += l_at;
            } else {
                let mut j = i;
                while j < n && marked[j] == 0 {
                    j += 1;
                }
                if j == i {
                    j = i + 1;
                }
                atoms.push((i, j));
                i = j;
            }
        }
        self.atoms = atoms;
    }

    fn compute_items(&mut self) {
        let t_len = self.atoms.len();
        let mut items = Vec::new();
        for t in 0..t_len {
            let i = self.atoms[t].0;
            let u_max = t_len.min(t + self.cfg.max_atoms_per_item);
            for u in t..u_max {
                let j = self.atoms[u].1;
                let s = &self.e[i..j];
                if s.len() > 2 * self.cfg.lmax {
                    break;
                }
                items.push(Item {
                    i,
                    j,
                    kind: Op::Lit,
                    lit: s.to_vec(),
                    k: 1,
                    a0: t,
                    a1: u + 1,
                });
                let ln = j - i;
                for k in [2usize, 3, 4] {
                    if ln % k == 0 {
                        let unit = ln / k;
                        let pat = &self.e[i..i + unit];
                        if (0..k).all(|m| &self.e[i + m * unit..i + (m + 1) * unit] == pat) {
                            items.push(Item {
                                i,
                                j,
                                kind: Op::Rep,
                                lit: pat.to_vec(),
                                k,
                                a0: t,
                                a1: u + 1,
                            });
                        }
                    }
                }
            }
        }

        let mut by_start: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut item_at: HashMap<(usize, usize), usize> = HashMap::new();
        for (idx, it) in items.iter().enumerate() {
            by_start.entry(it.i).or_default().push(idx);
            if it.kind == Op::Lit {
                item_at.entry((it.i, it.j)).or_insert(idx);
            }
        }
        self.items = items;
        self.by_start = by_start;
        self.item_at = item_at;
    }

    fn compute_partners(&mut self) {
        let mut groups: HashMap<(usize, Vec<usize>), Vec<usize>> = HashMap::new();
        for (idx, it) in self.items.iter().enumerate() {
            if it.kind != Op::Lit {
                continue;
            }
            groups
                .entry((it.lit.len(), pattern(&it.lit)))
                .or_default()
                .push(idx);
        }

        let mut partner = HashMap::new();
        let mut earlier = HashMap::new();
        let mut later = HashMap::new();
        for (_, mut g) in groups {
            if g.len() < 2 {
                continue;
            }
            g.sort_by_key(|&x| (self.items[x].i, self.items[x].j));
            for (t, &idx) in g.iter().enumerate() {
                let e_slice = g[t.saturating_sub(3)..t].to_vec();
                let l_slice = g[t + 1..(t + 4).min(g.len())].to_vec();
                let mut p_slice = Vec::with_capacity(4);
                for &x in e_slice.iter().chain(l_slice.iter()) {
                    if p_slice.len() == 4 {
                        break;
                    }
                    p_slice.push(x);
                }
                earlier.insert(idx, e_slice);
                later.insert(idx, l_slice);
                partner.insert(idx, p_slice);
            }
        }
        self.partner = partner;
        self.earlier = earlier;
        self.later = later;
    }

    pub fn key_between(&self, a_idx: usize, b_idx: usize) -> Option<[u8; 256]> {
        let a_bs = &self.items[a_idx].lit;
        let b_bs = &self.items[b_idx].lit;
        if a_bs.len() != b_bs.len() {
            return None;
        }
        let mut m: HashMap<u8, u8> = HashMap::new();
        for (&x, &y) in a_bs.iter().zip(b_bs.iter()) {
            if let Some(&prev) = m.get(&x) {
                if prev != y {
                    return None;
                }
            }
            m.insert(x, y);
        }
        let used: HashSet<u8> = m.values().copied().collect();
        if used.len() != m.len() {
            return None;
        }
        let rs: Vec<u8> = self
            .alpha
            .iter()
            .copied()
            .filter(|x| !m.contains_key(x))
            .collect();
        let rt: Vec<u8> = self
            .alpha
            .iter()
            .copied()
            .filter(|y| !used.contains(y))
            .collect();
        if rs.len() != rt.len() {
            return None;
        }
        for (x, y) in rs.into_iter().zip(rt.into_iter()) {
            m.insert(x, y);
        }
        let mut key = IDENT;
        for (x, y) in m {
            key[x as usize] = y;
        }
        Some(key)
    }

    fn shared_pieces(&self, recs: &[&[u8]]) -> (Vec<Vec<u8>>, usize) {
        let r_cnt = recs.len();
        let mut cnt: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut order = Vec::new();
        for &r in recs {
            let mut seen: HashSet<&[u8]> = HashSet::new();
            let mut seen_order: Vec<&[u8]> = Vec::new();
            for i in 0..r.len() {
                let j_max = r.len().min(i + self.cfg.lmax);
                for j in (i + 1)..=j_max {
                    let sub = &r[i..j];
                    if seen.insert(sub) {
                        seen_order.push(sub);
                    }
                }
            }
            for s in seen_order {
                let sv = s.to_vec();
                let entry = cnt.entry(sv.clone()).or_insert_with(|| {
                    order.push(sv);
                    0
                });
                *entry += 1;
            }
        }
        let thr = if r_cnt >= 3 { r_cnt } else { 2.max(r_cnt) };
        let pieces: Vec<Vec<u8>> = order
            .into_iter()
            .filter(|s| s.len() >= 2 && cnt.get(s).copied().unwrap_or(0) >= thr)
            .collect();
        let mut maximal: Vec<Vec<u8>> = pieces
            .iter()
            .filter(|s| !pieces.iter().any(|t| *t != **s && contains_subslice(t, s)))
            .cloned()
            .collect();
        if r_cnt >= 2 && !recs[0].is_empty() {
            let tail = &recs[0][recs[0].len() - 1..];
            if recs
                .iter()
                .all(|r| r.ends_with(tail) && !contains_subslice(&r[..r.len() - 1], tail))
            {
                maximal.push(tail.to_vec());
            }
        }
        let cov = maximal.iter().map(|s| s.len()).sum::<usize>() * r_cnt;
        (maximal, cov)
    }

    fn split_records(&self, sep_byte: u8) -> Vec<(usize, usize)> {
        let mut parts = Vec::new();
        let mut st = 0usize;
        for (i, &x) in self.e.iter().enumerate() {
            if x == sep_byte {
                if i > st {
                    parts.push((st, i));
                }
                st = i + 1;
            }
        }
        if st < self.n {
            parts.push((st, self.n));
        }
        parts
    }

    fn compute_frames(&mut self) {
        let n = self.n;
        let mut freq_counts = [0usize; 256];
        let mut freq_order = Vec::new();
        for &b in &self.e {
            if freq_counts[b as usize] == 0 {
                freq_order.push(b);
            }
            freq_counts[b as usize] += 1;
        }

        let mut best: Option<u8> = None;
        let mut bs = -1.0f64;
        let max_sep_cnt = (n / 2).max(2);
        for b in freq_order {
            let c = freq_counts[b as usize];
            if c < 2 || c > max_sep_cnt {
                continue;
            }
            let parts = self.split_records(b);
            if parts.len() < 2 {
                continue;
            }
            let slices: Vec<&[u8]> = parts.iter().map(|&(a, e)| &self.e[a..e]).collect();
            let (_pieces, cov) = self.shared_pieces(&slices);
            if (cov as f64) > bs {
                bs = cov as f64;
                best = Some(b);
            }
        }

        self.sep = best;
        self.records = match best {
            None => {
                if n > 0 {
                    vec![(0, n)]
                } else {
                    Vec::new()
                }
            }
            Some(sep_byte) => self.split_records(sep_byte),
        };

        let recs: Vec<&[u8]> = self.records.iter().map(|&(a, b)| &self.e[a..b]).collect();
        let (maximal, _cov) = self.shared_pieces(&recs);
        self.frame = maximal;

        let mut slots = Vec::new();
        for (ri, &(a, b)) in self.records.iter().enumerate() {
            let r = &self.e[a..b];
            let mut marks: Vec<(usize, usize)> = Vec::new();
            for p in &self.frame {
                let mut st = 0usize;
                while let Some(k) = find_subslice(r, p, st) {
                    marks.push((k, k + p.len()));
                    st = k + 1;
                }
            }
            marks.sort_by(|&(a1, b1), &(a2, b2)| {
                (b2 - a2).cmp(&(b1 - a1)).then_with(|| a1.cmp(&a2))
            });

            let mut kept: Vec<(usize, usize)> = Vec::new();
            for (x, y) in marks {
                if kept.iter().all(|&(u, v)| y <= u || x >= v) {
                    kept.push((x, y));
                }
            }
            kept.sort();

            let mut gaps = Vec::new();
            let mut cur = 0usize;
            for (x, y) in kept {
                if x > cur {
                    gaps.push((cur, x));
                }
                cur = y;
            }
            if cur < r.len() {
                gaps.push((cur, r.len()));
            }
            for (ci, (x, y)) in gaps.into_iter().enumerate() {
                slots.push((a + x, a + y, ci, ri));
            }
        }
        self.slots = slots;
    }

    fn compute_features(&mut self) {
        let n = self.n;
        let mut fmat = Vec::with_capacity(self.items.len());
        for (idx, it) in self.items.iter().enumerate() {
            let i = it.i;
            let j = it.j;
            let s = &it.lit;
            let ln = j - i;
            let cnt = self.counts.get(s).copied().unwrap_or(1);
            let ne = self.earlier.get(&idx).map_or(0, |v| v.len());
            let nl = self.later.get(&idx).map_or(0, |v| v.len());
            let sur = self.sur[i..j].iter().sum::<f64>() / (ln.max(1) as f64);
            let bnd = if i < n { self.sur[i] } else { 1.0 };
            let mut seen_bytes = [false; 256];
            let mut uniq = 0usize;
            for &byte in s {
                if !seen_bytes[byte as usize] {
                    seen_bytes[byte as usize] = true;
                    uniq += 1;
                }
            }
            let div = uniq as f64 / (ln.max(1) as f64);
            fmat.push([
                ln as f64 / self.cfg.lmax as f64,
                1.0f64.min((cnt as f64).ln_1p() / 3.0),
                if ne > 0 { 1.0 } else { 0.0 },
                if nl > 0 { 1.0 } else { 0.0 },
                (ne + nl) as f64 / 4.0,
                (it.k as f64 - 1.0) / 3.0,
                1.0,
                (it.a1 - it.a0) as f64 / 6.0,
                sur,
                bnd,
                div,
                1.0,
            ]);
        }

        let d = fmat.len();
        let mut pf = vec![[0.0; 12]; d];
        let mut pb = vec![[0.0; 12]; d];
        for idx in 0..d {
            if let Some(e_list) = self.earlier.get(&idx) {
                if !e_list.is_empty() {
                    let denom = e_list.len() as f64;
                    for &e_idx in e_list {
                        for k in 0..12 {
                            pf[idx][k] += fmat[e_idx][k];
                        }
                    }
                    for k in 0..12 {
                        pf[idx][k] /= denom;
                    }
                }
            }
            if let Some(l_list) = self.later.get(&idx) {
                if !l_list.is_empty() {
                    let denom = l_list.len() as f64;
                    for &l_idx in l_list {
                        for k in 0..12 {
                            pb[idx][k] += fmat[l_idx][k];
                        }
                    }
                    for k in 0..12 {
                        pb[idx][k] /= denom;
                    }
                }
            }
        }
        self.fmat = fmat;
        self.pf = pf;
        self.pb = pb;
    }
}

/// Contrôleur de routage graphe. Trois facteurs, local, non réciproque.
#[derive(Clone, Debug)]
pub struct Router {
    pub d: usize,
    pub h: usize,
    pub wf: Vec<Vec<f64>>,
    pub wp: Vec<Vec<f64>>,
    pub wb: Vec<Vec<f64>>,
    pub u: Vec<f64>,
    pub baseline: f64, 
    pub m2: f64,
}

impl Router {
    pub fn new(d: usize, h: usize, seed: u64) -> Self {
        let mut rng = NormalRng::new(seed);
        let mut sample_mat = |rows: usize, cols: usize| -> Vec<Vec<f64>> {
            (0..rows)
                .map(|_| (0..cols).map(|_| rng.standard_normal() * 0.3).collect())
                .collect()
        };
        let wf = sample_mat(d, h);
        let wp = sample_mat(d, h);
        let wb = sample_mat(d, h);
        let u: Vec<f64> = (0..h).map(|_| rng.standard_normal() * 0.3).collect();
        Self {
            d,
            h,
            wf,
            wp,
            wb,
            u,
            baseline: 0.0,
        }
    }

    pub fn score(&self, per: &Perception) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n_items = per.fmat.len();
        let mut s = Vec::with_capacity(n_items);
        let mut hm = Vec::with_capacity(n_items);
        let cols = self.d.min(12);

        for r in 0..n_items {
            let mut row_h = vec![0.0; self.h];
            let mut logit = 0.0;
            for c in 0..self.h {
                let mut acc = 0.0;
                for k in 0..cols {
                    acc += per.fmat[r][k] * self.wf[k][c]
                        + per.pf[r][k] * self.wp[k][c]
                        + per.pb[r][k] * self.wb[k][c];
                }
                let th = acc.tanh();
                row_h[c] = th;
                logit += th * self.u[c];
            }
            s.push(1.0 / (1.0 + (-logit).exp()));
            hm.push(row_h);
        }
        (s, hm)
    }
    
    pub fn update(
        &mut self,
        per: &Perception,
        hm: &[Vec<f64>],
        used: &[usize],
        reward: f64,
        eta: f64,
    ) {
        let adv_raw = reward - self.baseline;
        self.baseline += 0.05 * adv_raw;
        self.m2       += 0.05 * (reward * reward - self.m2);
        let var = (self.m2 - self.baseline * self.baseline).max(1e-8);
        let adv = (adv_raw / var.sqrt()).clamp(-3.0, 3.0);
        if used.is_empty() { return; }
    
        self.baseline += 0.05 * adv;
        let m = used.len().max(1) as f64;
        let cols = self.d.min(12);
        let mut mean_hm = vec![0.0; self.h];
        for &idx in used {
            for c in 0..self.h {
                let h_val = hm[idx][c];
                mean_hm[c] += h_val;
                let post_c = h_val * self.u[c];
                for k in 0..cols {
                    self.wf[k][c] += eta * adv * (per.fmat[idx][k] * post_c) / m;
                    self.wp[k][c] += eta * adv * (per.pf[idx][k] * post_c) / m;
                    self.wb[k][c] += eta * adv * (per.pb[idx][k] * post_c) / m;
                }
            }
        }
        for c in 0..self.h {
            self.u[c] += eta * adv * (mean_hm[c] / m);
        }
    }

    /// Wp != Wb^T.
    pub fn nonreciprocal(&self) -> bool {
        if self.d != self.h {
            return true;
        }
        for i in 0..self.d {
            for j in 0..self.h {
                let a = self.wp[i][j];
                let b = self.wb[j][i];
                if (a - b).abs() > 1e-8 + 1e-5 * b.abs() {
                    return true;
                }
            }
        }
        false
    }

    /// Jacobien directionnel du graphe de routage : M[i, j] = d(score_i)/d(F_j) · F_j.
    /// j prédécesseur de i => Wp ; j successeur de i => Wb.
    pub fn influence(&self, per: &Perception) -> Vec<Vec<f64>> {
        let (s, hm) = self.score(per);
        let n = per.items.len();
        let cols = self.d.min(12);

        let mut wpi = vec![vec![0.0; cols]; n];
        let mut wbi = vec![vec![0.0; cols]; n];
        for i in 0..n {
            let ds = s[i] * (1.0 - s[i]);
            for k in 0..cols {
                let mut acc_p = 0.0;
                let mut acc_b = 0.0;
                for c in 0..self.h {
                    let g_ic = ds * (1.0 - hm[i][c] * hm[i][c]) * self.u[c];
                    acc_p += self.wp[k][c] * g_ic;
                    acc_b += self.wb[k][c] * g_ic;
                }
                wpi[i][k] = acc_p;
                wbi[i][k] = acc_b;
            }
        }

        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n {
            if let Some(e_list) = per.earlier.get(&i) {
                if !e_list.is_empty() {
                    let denom = e_list.len() as f64;
                    for &j in e_list {
                        let mut dot = 0.0;
                        for k in 0..cols {
                            dot += per.fmat[j][k] * wpi[i][k];
                        }
                        m[i][j] = dot / denom;
                    }
                }
            }
            if let Some(l_list) = per.later.get(&i) {
                if !l_list.is_empty() {
                    let denom = l_list.len() as f64;
                    for &j in l_list {
                        let mut dot = 0.0;
                        for k in 0..cols {
                            dot += per.fmat[j][k] * wbi[i][k];
                        }
                        m[i][j] = dot / denom;
                    }
                }
            }
        }
        m
    }
}

#[derive(Clone, Debug)]
pub struct Abduction {
    pub cause: Option<Cause>,
    pub phi: f64,
    pub info: Option<PhiInfo>,
    pub pops: usize,
    pub used: Vec<usize>,
    pub cover: Vec<usize>,
    pub complete: bool,
}

fn build_variant(
    per: &Perception,
    chosen: &[usize],
    links: &[(usize, usize, [u8; 256])],
) -> Cause {
    let mut b = Builder::new(true);
    let mut tops = Vec::with_capacity(chosen.len());
    for &idx in chosen {
        let it = &per.items[idx];
        if it.kind == Op::Lit {
            tops.push(b.add(Node::lit(it.lit.clone())));
        } else {
            let base = b.add(Node::lit(it.lit.clone()));
            tops.push(b.add(Node::rep(base, it.k)));
        }
    }
    let pos: HashMap<usize, usize> = chosen
        .iter()
        .enumerate()
        .map(|(t, &idx)| (idx, t))
        .collect();
    for &(t, s, key) in links {
        tops[t] = b.add(Node::rec(tops[pos[&s]], key));
    }
    let root = if tops.len() > 1 {
        b.add(Node::seq(tops))
    } else {
        tops[0]
    };
    b.cause(root)
}

#[derive(Clone, Debug)]
struct HeapEntry {
    f: f64,
    counter: usize,
    pos: usize,
    chosen: Vec<usize>,
    g: f64,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.f.to_bits() == other.f.to_bits() && self.counter == other.counter
    }
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.f.total_cmp(&other.f) {
            Ordering::Equal => other.counter.cmp(&self.counter),
            ord => ord,
        }
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn key_cmp(a: (f64, isize, isize), b: (f64, isize, isize)) -> Ordering {
    a.0.total_cmp(&b.0)
        .then_with(|| a.1.cmp(&b.1))
        .then_with(|| a.2.cmp(&b.2))
}

/// Recherche COMPLÈTE : le contrôleur réordonne, il n'élague jamais.
pub fn abduce(
    per: &Perception,
    a: usize,
    b: usize,
    scores: &[f64],
    cfg: &Cfg,
    rng: &mut PyRandom,
    pops_budget: Option<usize>,
) -> Abduction {
    let empty = |pops| Abduction {
        cause: None,
        phi: 0.0,
        info: None,
        pops,
        used: Vec::new(),
        cover: Vec::new(),
        complete: false,
    };

    if b <= a {
        return empty(0);
    }

    let mut starts: HashMap<usize, Vec<usize>> = HashMap::new();
    for pos in a..b {
        if let Some(lst) = per.by_start.get(&pos) {
            for &idx in lst {
                if per.items[idx].j <= b {
                    starts.entry(pos).or_default().push(idx);
                }
            }
        }
    }
    if !starts.contains_key(&a) {
        return empty(0);
    }

    let budget = pops_budget.unwrap_or(cfg.pops);
    let maxd = starts
        .values()
        .flat_map(|lst| lst.iter().map(|&i| scores[i]))
        .reduce(f64::max)
        .unwrap_or(1.0);

    let mut heap = BinaryHeap::new();
    heap.push(HeapEntry {
        f: (b - a) as f64 * maxd,
        counter: 0,
        pos: a,
        chosen: Vec::new(),
        g: 0.0,
    });

    let mut counter = 1usize;
    let mut pops = 0usize;
    let mut completes: Vec<Vec<usize>> = Vec::new();
    while !heap.is_empty() && pops < budget && completes.len() < cfg.completes {
        let entry = heap.pop().unwrap();
        pops += 1;
        if entry.pos == b {
            completes.push(entry.chosen);
            continue;
        }
        if let Some(lst) = starts.get(&entry.pos) {
            for &idx in lst {
                let nj = per.items[idx].j;
                let ng = entry.g + scores[idx];
                let hh = (b - nj) as f64 * maxd;
                let mut next_chosen = entry.chosen.clone();
                next_chosen.push(idx);
                heap.push(HeapEntry {
                    f: ng + hh,
                    counter,
                    pos: nj,
                    chosen: next_chosen,
                    g: ng,
                });
                counter += 1;
            }
        }
    }

    let target = &per.e[a..b];
    let mut best = empty(pops);
    let mut best_key: (f64, isize, isize) = (-1.0, -1, -1);
    let mut extra_keys: Vec<[u8; 256]> = Vec::new();

    for chosen in &completes {
        let chosen_set: HashSet<usize> = chosen.iter().copied().collect();
        let mut linkable: Vec<(usize, usize, [u8; 256])> = Vec::new();
        for (t, &idx) in chosen.iter().enumerate() {
            if let Some(earl) = per.earlier.get(&idx) {
                for &s in earl {
                    if chosen_set.contains(&s) && per.items[s].j <= per.items[idx].i {
                        if per.items[s].lit == per.items[idx].lit {
                            continue;
                        }
                        if let Some(key) = per.key_between(s, idx) {
                            linkable.push((t, s, key));
                            extra_keys.push(key);
                            break;
                        }
                    }
                }
            }
        }
        linkable.truncate(cfg.lift_cap);
        for mask in 0usize..(1usize << linkable.len()) {
            let links: Vec<(usize, usize, [u8; 256])> = linkable
                .iter()
                .enumerate()
                .filter(|&(t, _)| ((mask >> t) & 1) != 0)
                .map(|(_, &item)| item)
                .collect();
            let c = build_variant(per, chosen, &links);
            if values(&c, None)[c.root].as_slice() != target {
                continue;
            }
            let extra_slice = &extra_keys[..extra_keys.len().min(3)];
            let (ph, info) = phi(&c, target, rng, cfg.phi_n, extra_slice, 3);
            let nshared = info
                .sources
                .iter()
                .filter(|d| d.iota > 0.0 && d.eps > 0.0)
                .count() as isize;
            let key_t = (ph, nshared, c.nodes.len() as isize);
            if key_cmp(key_t, best_key) == Ordering::Greater {
                best_key = key_t;
                best = Abduction {
                    cause: Some(c),
                    phi: ph,
                    info: Some(info),
                    pops,
                    used: chosen.clone(),
                    cover: chosen.clone(),
                    complete: true,
                };
            }
        }
    }

    if let Some(ref cause) = best.cause {
        let extra_slice = &extra_keys[..extra_keys.len().min(3)];
        let (ph, info) = phi(cause, target, rng, cfg.phi_n_final, extra_slice, 3);
        best.phi = ph;
        best.info = Some(info);
    }

    best
}

#[derive(Clone, Debug)]
pub struct Residual {
    pub gap: Vec<(usize, usize)>,
    pub inert: Vec<(usize, usize)>,
    pub undet: Vec<(usize, usize, Vec<usize>)>,
    pub open: Vec<(usize, usize, Vec<usize>)>,
}

pub fn residual(per: &Perception, a: usize, b: usize, ab: &Abduction) -> Residual {
    let mut covered = HashSet::new();
    if ab.complete {
        if let Some(ref info) = ab.info {
            for d in &info.sources {
                if d.iota > 0.0 && d.eps > 0.0 {
                    for &(x, y) in &info.occ[d.node] { for p in x..y { covered.insert(p); } }
                }
            }
        }
    }
    let mut inert = Vec::new();
    let mut cur: Option<usize> = None;
    for t in 0..(b - a) {
        if !covered.contains(&t) { if cur.is_none() { cur = Some(t); } }
        else if let Some(c) = cur { inert.push((a + c, a + t)); cur = None; }
    }
    if let Some(c) = cur { inert.push((a + c, b)); }

    let incomplete = !ab.complete || ab.cause.is_none();
    let gap = if incomplete { vec![(a, b)] } else { Vec::new() };
    let (undet, open_alt) = match ab.cause {
        Some(ref c) if ab.complete => underdetermined(c, &per.e[a..b]),
        _ => (Vec::new(), Vec::new()),
    };
    debug_assert!(incomplete || covered.len() + inert.iter().map(|(x,y)| y-x).sum::<usize>() == b - a,
                  "rho doit couvrir [a,b)");
    Residual { gap, inert,
               undet: undet.into_iter().map(|(n,vs)| (a,n,vs)).collect(),
               open:  open_alt.into_iter().map(|(n,ss)| (a,n,ss)).collect() }
}