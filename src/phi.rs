// φ_sharp : φ_s d'IIT 4.0 (TPM, répertoires cause/effet, bipartitions unidirectionnelles, MIP, complexe).
// État : bit i = unité i. Tpm : p[x * n + i] = P(U_i(t+1) = 1 | U(t) = x), unités indépendantes sachant x.
const NONE: usize = usize::MAX;
const EPS: f64 = 1e-9;
const TIE: f64 = 1e-12;

#[derive(Clone, Debug)]
pub struct Tpm {
    pub n: usize,
    pub p: Vec<f64>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Partition {
    pub from: usize,
    pub to: usize,
}
impl Partition {
    pub fn connections(&self) -> usize {
        (self.from.count_ones() * self.to.count_ones()) as usize
    }
}
// Les états (state, cause_state, effect_state, index des répertoires) sont compactés :
// le bit r est la r-ième unité de `units`. `global` les ramène aux bits du substrat.
#[derive(Clone, Debug)]
pub struct Sia {
    pub units: usize,
    pub state: usize,
    pub cause_state: usize,
    pub effect_state: usize,
    pub phi: f64,
    pub phi_cause: f64,
    pub phi_effect: f64,
    pub normalized: f64,
    pub mip: Option<Partition>,
    pub cause: Vec<f64>,
    pub effect: Vec<f64>,
    pub cause_cut: Vec<f64>,
    pub effect_cut: Vec<f64>,
}
impl Sia {
    pub fn global(&self, local: usize) -> usize {
        scatter(self.units, local)
    }
}
struct Best {
    norm: f64,
    phi: f64,
    pc: f64,
    pe: f64,
    cut: usize,
}
impl Tpm {
    pub fn new(n: usize, p: Vec<f64>) -> Tpm {
        assert!(n >= 1 && n < usize::BITS as usize);
        assert_eq!(p.len(), n << n);
        assert!(p.iter().all(|v| (0.0..=1.0).contains(v)));
        Tpm { n, p }
    }
    pub fn from_fn<F: FnMut(usize, usize) -> f64>(n: usize, mut f: F) -> Tpm {
        let mut p = Vec::with_capacity(n << n);
        for x in 0..1usize << n {
            for i in 0..n {
                p.push(f(x, i));
            }
        }
        Tpm::new(n, p)
    }
    // Fond figé : les unités hors `units` gardent leur état courant.
    pub fn condition(&self, units: usize, state: usize) -> Tpm {
        let n = self.n;
        let fixed = state & !units & ((1usize << n) - 1);
        let idx: Vec<usize> = (0..n).filter(|&i| units >> i & 1 == 1).collect();
        let k = idx.len();
        let mut p = Vec::with_capacity(k << k);
        for x in 0..1usize << k {
            let full = fixed | scatter(units, x);
            for &i in &idx {
                p.push(self.p[full * n + i]);
            }
        }
        Tpm { n: k, p }
    }
    // Coupure from → to : pour chaque unité de `to`, les entrées issues de `from`
    // sont remplacées par du bruit uniforme (marginalisation).
    pub fn cut(&self, from: usize, to: usize) -> Tpm {
        let n = self.n;
        let size = 1usize << n;
        let keep = (size - 1) & !from;
        let w = 1.0 / (1usize << from.count_ones()) as f64;
        let mut p = self.p.clone();
        let mut acc = vec![0.0f64; size];
        for j in 0..n {
            if to >> j & 1 == 0 {
                continue;
            }
            acc.fill(0.0);
            for x in 0..size {
                acc[x & keep] += self.p[x * n + j];
            }
            for x in 0..size {
                p[x * n + j] = acc[x & keep] * w;
            }
        }
        Tpm { n, p }
    }
    // π_e(z|s) = Π_j p(z_j|s)
    pub fn effect_at(&self, s: usize, z: usize) -> f64 {
        let row = &self.p[s * self.n..(s + 1) * self.n];
        (0..self.n)
            .map(|j| if z >> j & 1 == 1 { row[j] } else { 1.0 - row[j] })
            .product::<f64>()
    }
    pub fn effect(&self, s: usize) -> Vec<f64> {
        (0..1usize << self.n).map(|z| self.effect_at(s, z)).collect()
    }
    // π_c(z|s) ∝ Π_j p(s_j|z), prior uniforme ; nul partout si s est inatteignable.
    pub fn cause(&self, s: usize) -> Vec<f64> {
        let n = self.n;
        let mut l: Vec<f64> = (0..1usize << n)
            .map(|z| {
                let row = &self.p[z * n..(z + 1) * n];
                (0..n)
                    .map(|j| if s >> j & 1 == 1 { row[j] } else { 1.0 - row[j] })
                    .product::<f64>()
            })
            .collect();
        let tot: f64 = l.iter().sum();
        if tot > 0.0 {
            for v in &mut l {
                *v /= tot;
            }
        }
        l
    }
    // π_e(z) : état d'entrée uniforme, produit des marginales d'unités.
    pub fn unconstrained_effect(&self) -> Vec<f64> {
        let n = self.n;
        let size = 1usize << n;
        let q: Vec<f64> = (0..n)
            .map(|j| (0..size).map(|x| self.p[x * n + j]).sum::<f64>() / size as f64)
            .collect();
        (0..size)
            .map(|z| {
                (0..n)
                    .map(|j| if z >> j & 1 == 1 { q[j] } else { 1.0 - q[j] })
                    .product::<f64>()
            })
            .collect()
    }
}
pub fn scatter(mask: usize, x: usize) -> usize {
    let mut out = 0usize;
    let mut m = mask;
    let mut r = 0;
    while m != 0 {
        let b = m & m.wrapping_neg();
        if x >> r & 1 == 1 {
            out |= b;
        }
        m ^= b;
        r += 1;
    }
    out
}
pub fn gather(mask: usize, x: usize) -> usize {
    let mut out = 0usize;
    let mut m = mask;
    let mut r = 0;
    while m != 0 {
        let b = m & m.wrapping_neg();
        if x & b != 0 {
            out |= 1 << r;
        }
        m ^= b;
        r += 1;
    }
    out
}
fn bits(p: f64, q: f64) -> f64 {
    if p <= 0.0 {
        0.0
    } else if q <= 0.0 {
        f64::INFINITY
    } else {
        p * (p / q).log2()
    }
}
fn gap(p: f64, q: f64) -> f64 {
    bits(p, q).abs()
}
fn specified(ii: &[f64]) -> Vec<usize> {
    let m = ii.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m <= TIE {
        return Vec::new();
    }
    (0..ii.len()).filter(|&z| ii[z] >= m - TIE).collect()
}
fn local_cuts(k: usize) -> Vec<(usize, usize)> {
    if k == 1 {
        return vec![(1, 1)];
    }
    let full = (1usize << k) - 1;
    (1..full).map(|a| (a, full ^ a)).collect()
}
// Θ(S) : les 2^k − 2 bipartitions unidirectionnelles (k = 1 : coupure de l'auto-connexion).
pub fn partitions(units: usize) -> Vec<Partition> {
    local_cuts(units.count_ones() as usize)
        .into_iter()
        .map(|(a, b)| Partition { from: scatter(units, a), to: scatter(units, b) })
        .collect()
}
// s'_x = argmax_z π_x(z|s)·log2(π_x(z|s)/π_x(z))
// φ_x(θ) = π_x(s'_x|s)·|log2(π_x(s'_x|s)/π_x^θ(s'_x|s))|, φ_s(θ) = min(φ_c, φ_e)
// MIP = argmin_θ φ_s(θ)/(|S¹|·|S²|), φ_s = φ_s(MIP)
pub fn analyze(tpm: &Tpm, state: usize, units: usize) -> Sia {
    let sys = tpm.condition(units, state);
    let k = sys.n;
    let s = gather(units, state);
    let cause = sys.cause(s);
    let effect = sys.effect(s);
    let free = sys.unconstrained_effect();
    let uniform = 1.0 / (1usize << k) as f64;
    let ii_c: Vec<f64> = cause.iter().map(|&p| bits(p, uniform)).collect();
    let ii_e: Vec<f64> = effect.iter().zip(&free).map(|(&p, &q)| bits(p, q)).collect();
    let tc = specified(&ii_c);
    let te = specified(&ii_e);
    if tc.is_empty() || te.is_empty() {
        return Sia {
            units,
            state: s,
            cause_state: 0,
            effect_state: 0,
            phi: 0.0,
            phi_cause: 0.0,
            phi_effect: 0.0,
            normalized: 0.0,
            mip: None,
            cause_cut: cause.clone(),
            effect_cut: effect.clone(),
            cause,
            effect,
        };
    }
    let cuts = local_cuts(k);
    let width = te.len();
    let mut best: Vec<Best> = (0..tc.len() * width)
        .map(|_| Best { norm: f64::INFINITY, phi: 0.0, pc: 0.0, pe: 0.0, cut: NONE })
        .collect();
    for (ci, &(a, b)) in cuts.iter().enumerate() {
        let cut = sys.cut(a, b);
        let cc = cut.cause(s);
        let links = (a.count_ones() * b.count_ones()) as f64;
        let pcs: Vec<f64> = tc.iter().map(|&z| gap(cause[z], cc[z])).collect();
        let pes: Vec<f64> = te.iter().map(|&z| gap(effect[z], cut.effect_at(s, z))).collect();
        for (i, &pc) in pcs.iter().enumerate() {
            for (j, &pe) in pes.iter().enumerate() {
                let phi = pc.min(pe);
                let norm = phi / links;
                let slot = &mut best[i * width + j];
                if slot.cut == NONE || norm < slot.norm - TIE {
                    *slot = Best { norm, phi, pc, pe, cut: ci };
                }
            }
        }
    }
    let mut pick = 0;
    for i in 1..best.len() {
        if best[i].phi > best[pick].phi + TIE {
            pick = i;
        }
    }
    let top = &best[pick];
    let (a, b) = cuts[top.cut];
    let cut = sys.cut(a, b);
    let part = Partition { from: scatter(units, a), to: scatter(units, b) };
    let phi = if top.phi < EPS { 0.0 } else { top.phi };
    Sia {
        units,
        state: s,
        cause_state: tc[pick / width],
        effect_state: te[pick % width],
        phi,
        phi_cause: top.pc,
        phi_effect: top.pe,
        normalized: phi / part.connections() as f64,
        mip: Some(part),
        cause_cut: cut.cause(s),
        effect_cut: cut.effect(s),
        cause,
        effect,
    }
}
// Complexes : φ_s maximal, égalité → plus grand système ; exclusion des systèmes chevauchants.
pub fn complexes(tpm: &Tpm, state: usize) -> Vec<Sia> {
    let mut ranked: Vec<(i64, u32, usize)> = Vec::new();
    for units in 1..1usize << tpm.n {
        let phi = analyze(tpm, state, units).phi;
        if phi > 0.0 {
            ranked.push(((phi / EPS).round() as i64, units.count_ones(), units));
        }
    }
    ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    let mut covered = 0usize;
    let mut out: Vec<Sia> = Vec::new();
    for &(_, _, units) in &ranked {
        if units & covered == 0 {
            covered |= units;
            out.push(analyze(tpm, state, units));
        }
    }
    out
}
pub fn complex(tpm: &Tpm, state: usize) -> Option<Sia> {
    complexes(tpm, state).into_iter().next()
}
pub fn phi_sharp(tpm: &Tpm, state: usize) -> f64 {
    complex(tpm, state).map_or(0.0, |c| c.phi)
}