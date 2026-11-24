//! Substrat : K unités binaires stochastiques ; C = 5 matrices K×K de poids (une par canal de
//! relation du percept) fondamentalement non-réciproques : les unités forment un tournoi circulaire,
//! chaque paire {i,j} n'admet qu'un seul sens d'influence, tous canaux confondus.
//! Le percept ne fixe pas les poids, il gate la connectivité efficace (Evidence) : la TPM, donc
//! φ_sharp, est toujours relative à E.
use crate::percept::{Chan, Percept};
use crate::phi::Tpm;
use crate::rng::Rng;
use std::path::Path;

pub const CHANS: [Chan; 5] = [
    Chan::Identity,
    Chan::Forward,
    Chan::Backward,
    Chan::Iso,
    Chan::IsoBytes,
];
pub const C: usize = CHANS.len();
pub const F: usize = 3 + 2 * C;
pub const K_MIN: usize = 3;
pub const K_DEFAULT: usize = 5;
pub const K_MAX: usize = 12;
const PMIN: f64 = 1.0 / 256.0;
const GATE_EPS: f64 = 1e-9;
const MAGIC: [u8; 4] = *b"PHIS";
const VERSION: u32 = 1;
const HEADER: usize = 20;

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn maxof(v: &[f64]) -> f64 {
    v.iter().copied().fold(0.0f64, f64::max)
}

fn unit(v: f64, mx: f64) -> f64 {
    if mx > 0.0 { v / mx } else { 0.0 }
}

fn strength(c: usize, w: f32) -> f64 {
    let w = f64::from(w).max(0.0);
    match CHANS[c] {
        Chan::Forward | Chan::Backward => 1.0 / (1.0 + w),
        Chan::Identity | Chan::Iso | Chan::IsoBytes => (1.0 + w).ln(),
    }
}

// Tournoi circulaire : pour toute paire {i,j}, exactement un des deux arcs existe.
fn orient(k: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(k * (k - 1) / 2);
    for i in 0..k {
        for d in 1..=(k - 1) / 2 {
            v.push((i, (i + d) % k));
        }
    }
    if k % 2 == 0 {
        for i in 0..k / 2 {
            v.push((i, i + k / 2));
        }
    }
    v
}

#[derive(Clone, Debug)]
pub struct Net {
    pub k: usize,
    arrows: Vec<(usize, usize)>,
    // [profils K×F | biais d'appartenance K | poids C×arcs | seuils K | netteté 1]
    g: Vec<f32>,
}

impl Net {
    fn dim(k: usize) -> usize {
        k * F + k + C * (k * (k - 1) / 2) + k + 1
    }
    fn o_bm(&self) -> usize {
        self.k * F
    }
    fn o_w(&self) -> usize {
        self.k * F + self.k
    }
    fn o_th(&self) -> usize {
        self.o_w() + C * self.arrows.len()
    }
    fn o_beta(&self) -> usize {
        self.o_th() + self.k
    }

    pub fn new(k: usize, rng: &mut Rng) -> Net {
        assert!((K_MIN..=K_MAX).contains(&k));
        let mut net = Net { k, arrows: orient(k), g: vec![0.0; Net::dim(k)] };
        let (bm, w, th, be) = (net.o_bm(), net.o_w(), net.o_th(), net.o_beta());
        for i in 0..be {
            let s = if i < bm {
                1.5
            } else if i < w {
                1.0
            } else if i < th {
                5.0
            } else {
                1.0
            };
            net.g[i] = (rng.normal() * s) as f32;
        }
        net
    }

    pub fn params(&self) -> &[f32] {
        &self.g
    }
    pub fn params_mut(&mut self) -> &mut [f32] {
        &mut self.g
    }
    pub fn arrows(&self) -> &[(usize, usize)] {
        &self.arrows
    }

    /// Matrice dense K×K du canal c (ligne = source, colonne = cible) ; W[i][j]·W[j][i] = 0.
    pub fn matrix(&self, c: usize) -> Vec<f32> {
        assert!(c < C);
        let k = self.k;
        let na = self.arrows.len();
        let ow = self.o_w();
        let mut m = vec![0.0f32; k * k];
        for (ai, &(from, to)) in self.arrows.iter().enumerate() {
            m[from * k + to] = self.g[ow + c * na + ai];
        }
        m
    }

    /// Netteté de la politique de recrutement, dans (0, 16).
    pub fn sharpness(&self) -> f64 {
        16.0 * sigmoid(f64::from(self.g[self.o_beta()]))
    }

    fn member(&self, i: usize, feat: &[f64]) -> f64 {
        let mut z = f64::from(self.g[self.o_bm() + i]);
        for f in 0..F {
            z += f64::from(self.g[i * F + f]) * feat[f];
        }
        sigmoid(z)
    }

    /// TPM des K unités sous l'évidence d'une hypothèse :
    /// P(i=1 | x) = σ(θ_i + Σ_j x_j · Σ_c W_c[j→i] · G_c[j][i]).
    pub fn tpm(&self, ev: &Evidence) -> Tpm {
        let k = self.k;
        let na = self.arrows.len();
        let ow = self.o_w();
        let th = self.o_th();
        let mut t = vec![0.0f64; k * k];
        for (ai, &(from, to)) in self.arrows.iter().enumerate() {
            let mut s = 0.0f64;
            for c in 0..C {
                s += f64::from(self.g[ow + c * na + ai]) * ev.gate(c, from, to);
            }
            t[from * k + to] = s;
        }
        Tpm::from_fn(k, |x, i| {
            let mut z = f64::from(self.g[th + i]);
            for j in 0..k {
                if x >> j & 1 == 1 {
                    z += t[j * k + i];
                }
            }
            let p = sigmoid(z);
            if p.is_nan() { 0.5 } else { p.clamp(PMIN, 1.0 - PMIN) }
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(HEADER + 4 * self.g.len());
        b.extend_from_slice(&MAGIC);
        for v in [VERSION, self.k as u32, C as u32, F as u32] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.g {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    pub fn from_bytes(b: &[u8]) -> Result<Net, String> {
        if b.len() < HEADER || b[..4] != MAGIC[..] {
            return Err("fichier .phis invalide".into());
        }
        let rd = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        if rd(4) != VERSION {
            return Err(format!("version {} non supportée", rd(4)));
        }
        let k = rd(8) as usize;
        if rd(12) as usize != C || rd(16) as usize != F || !(K_MIN..=K_MAX).contains(&k) {
            return Err("architecture incompatible".into());
        }
        let n = Net::dim(k);
        if b.len() != HEADER + 4 * n {
            return Err("taille de modèle incohérente".into());
        }
        let mut g = Vec::with_capacity(n);
        for i in 0..n {
            let o = HEADER + 4 * i;
            let v = f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            if !v.is_finite() {
                return Err("paramètre non fini".into());
            }
            g.push(v);
        }
        Ok(Net { k, arrows: orient(k), g })
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        if let Some(dir) = path.as_ref().parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        std::fs::write(path, self.to_bytes())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Net, String> {
        let b = std::fs::read(path.as_ref()).map_err(|e| e.to_string())?;
        Net::from_bytes(&b)
    }
}

/// Lecture du percept par le substrat : appartenance de chaque ancre à chaque unité (rôle),
/// calculée à partir de traits purement relationnels/informationnels de l'ancre.
#[derive(Clone, Debug)]
pub struct Field {
    pub k: usize,
    pub na: usize,
    pub nc: usize,
    pub class_of: Vec<usize>,
    pub by_class: Vec<Vec<usize>>,
    pub cands: Vec<usize>,
    pub m: Vec<f64>,
    pub mbar: Vec<f64>,
    pub sal: Vec<f64>,
    ostr: Vec<f64>,
    edges: Vec<Vec<(usize, usize, f64)>>,
}

impl Field {
    pub fn new(net: &Net, p: &Percept) -> Field {
        let k = net.k;
        let na = p.anchors.len();
        let nc = p.repeats.len();
        let class_of: Vec<usize> = p.anchors.iter().map(|a| a.class).collect();
        let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); nc];
        for (a, &c) in class_of.iter().enumerate() {
            by_class[c].push(a);
        }
        let cands: Vec<usize> = (0..nc).filter(|&c| !by_class[c].is_empty()).collect();
        let mut edges: Vec<Vec<(usize, usize, f64)>> = Vec::with_capacity(C);
        let mut ostr = vec![0.0f64; C * na];
        let mut lout = vec![0.0f64; C * na];
        let mut lin = vec![0.0f64; C * na];
        for c in 0..C {
            let mut v: Vec<(usize, usize, f64)> = Vec::new();
            for e in p.rel.channel(CHANS[c]) {
                let s = strength(c, e.w);
                v.push((e.from, e.to, s));
                ostr[c * na + e.from] += s;
                lout[c * na + e.from] += 1.0;
                lin[c * na + e.to] += 1.0;
            }
            edges.push(v);
        }
        for v in lout.iter_mut().chain(lin.iter_mut()) {
            *v = (1.0 + *v).ln();
        }
        let mo: Vec<f64> = (0..C).map(|c| maxof(&lout[c * na..(c + 1) * na])).collect();
        let mi: Vec<f64> = (0..C).map(|c| maxof(&lin[c * na..(c + 1) * na])).collect();
        let lb: Vec<f64> = p
            .anchors
            .iter()
            .map(|a| (1.0 + p.repeats[a.class].bits).ln())
            .collect();
        let lc: Vec<f64> = p
            .anchors
            .iter()
            .map(|a| (p.repeats[a.class].count() as f64).ln())
            .collect();
        let ll: Vec<f64> = p.anchors.iter().map(|a| (a.len as f64).ln()).collect();
        let (mb, mc, ml) = (maxof(&lb), maxof(&lc), maxof(&ll));
        let mut m = vec![0.0f64; na * k];
        let mut feat = [0.0f64; F];
        for a in 0..na {
            feat[0] = unit(lb[a], mb);
            feat[1] = unit(lc[a], mc);
            feat[2] = unit(ll[a], ml);
            for c in 0..C {
                feat[3 + c] = unit(lout[c * na + a], mo[c]);
                feat[3 + C + c] = unit(lin[c * na + a], mi[c]);
            }
            for i in 0..k {
                m[a * k + i] = net.member(i, &feat);
            }
        }
        let mut mbar = vec![0.0f64; nc * k];
        for &c in &cands {
            let inv = 1.0 / by_class[c].len() as f64;
            for &a in &by_class[c] {
                for i in 0..k {
                    mbar[c * k + i] += m[a * k + i] * inv;
                }
            }
        }
        Field { k, na, nc, class_of, by_class, cands, m, mbar, sal: lb, ostr, edges }
    }

    /// R : état des K unités déclenché par la perception (réponse poolée, pondérée par la
    /// surprise des ancres, au moins égale à la moyenne des unités).
    pub fn response(&self) -> usize {
        let k = self.k;
        let tot: f64 = self.sal.iter().sum();
        let mut r = vec![0.0f64; k];
        if tot > 0.0 {
            for a in 0..self.na {
                for i in 0..k {
                    r[i] += self.sal[a] * self.m[a * k + i];
                }
            }
            for v in r.iter_mut() {
                *v /= tot;
            }
        }
        let mean = r.iter().sum::<f64>() / k as f64;
        let mut s = 0usize;
        for i in 0..k {
            if r[i] >= mean - 1e-12 {
                s |= 1 << i;
            }
        }
        s
    }

    /// Alignement de rôle ∈ [-1, 1] entre une classe de répétitions et un état d'unités z.
    pub fn align(&self, class: usize, z: usize) -> f64 {
        let k = self.k;
        let mut s = 0.0f64;
        for i in 0..k {
            let u = if z >> i & 1 == 1 { 1.0 } else { -1.0 };
            s += u * (2.0 * self.mbar[class * k + i] - 1.0);
        }
        s / k as f64
    }

    /// Région de E déclenchée par R : classes dont l'alignement avec R dépasse la moyenne.
    pub fn region(&self, r: usize) -> Vec<bool> {
        let mut sel = vec![false; self.nc];
        if self.cands.is_empty() {
            return sel;
        }
        let al: Vec<f64> = self.cands.iter().map(|&c| self.align(c, r)).collect();
        let mean = al.iter().sum::<f64>() / al.len() as f64;
        for (i, &c) in self.cands.iter().enumerate() {
            if al[i] >= mean - 1e-12 {
                sel[c] = true;
            }
        }
        sel
    }
}

/// Évidence relationnelle d'une hypothèse S (ensemble de classes de répétitions) :
/// flow[c][j][i] = force des relations du canal c allant d'une ancre de S de rôle j vers une ancre
/// de S de rôle i ; mass[c][j] = force sortante totale (y compris hors de S) des ancres de rôle j.
/// Les relations qui quittent S ne sont pas expliquées : elles diluent le couplage.
#[derive(Clone, Debug)]
pub struct Evidence {
    k: usize,
    flow: Vec<f64>,
    mass: Vec<f64>,
}

impl Evidence {
    pub fn of(f: &Field, sel: &[bool]) -> Evidence {
        let k = f.k;
        let mut flow = vec![0.0f64; C * k * k];
        let mut mass = vec![0.0f64; C * k];
        for a in 0..f.na {
            if !sel[f.class_of[a]] {
                continue;
            }
            for c in 0..C {
                let s = f.ostr[c * f.na + a];
                if s > 0.0 {
                    for j in 0..k {
                        mass[c * k + j] += f.m[a * k + j] * s;
                    }
                }
            }
        }
        for c in 0..C {
            for &(a, b, s) in &f.edges[c] {
                if !(sel[f.class_of[a]] && sel[f.class_of[b]]) {
                    continue;
                }
                for j in 0..k {
                    let v = s * f.m[a * k + j];
                    for i in 0..k {
                        flow[(c * k + j) * k + i] += v * f.m[b * k + i];
                    }
                }
            }
        }
        Evidence { k, flow, mass }
    }

    /// G_c[from][to] ∈ [0, 1].
    pub fn gate(&self, c: usize, from: usize, to: usize) -> f64 {
        self.flow[(c * self.k + from) * self.k + to] / (self.mass[c * self.k + from] + GATE_EPS)
    }
}