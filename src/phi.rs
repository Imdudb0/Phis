//! φ_sharp — substitut de puissance cause-effet intégrée, inspiré d'IIT 4.0.
//!
//! SUBSTITUT DÉCLARÉ, RÉFUTABLE PAR L'ÉTALONNAGE. Écarts assumés vis-à-vis
//! d'IIT 4.0, tous consignés comme risques :
//!  (1) répertoires = produits de marginales sur Q états ORDONNÉS par unité ;
//!  (2) distance = EMD 1-D exacte sur ces états (et non EMD sur l'espace d'états
//!      joint, qui est hors budget) ;
//!  (3) recherche MIP bornée à `mip` bipartitions déterministes, et restreinte
//!      aux bipartitions dont les quatre parts sont non vides — restriction qui
//!      ne peut que SUR-estimer Φ ; c'est le point le plus attaquable ;
//!  (4) le mécanisme est induit par l'hypothèse courante (masse des pointeurs),
//!      ce qui est ce qui rend le témoin H := E structurellement nul.
//!
//! Le mécanisme est FIXÉ pendant une évaluation de φ (sémantique IIT : φ est
//! calculée POUR un mécanisme). Les gains d'arêtes sont donc des constantes de
//! la tape ; le gradient passe par les répertoires, et la sélection du mécanisme
//! est apprise par le terme de couverture/résidu et par REINFORCE (lot 3bis).

use crate::ad::{Params, Tape};
use crate::eff::{EffView, REL_ADJ, REL_CONTAIN};
use crate::substrate::{self, Fwd, Itv, Wiring, P, Q};

#[derive(Clone, Copy, Debug)]
pub struct PhiCfg {
    pub mech: usize,
    pub purview: usize,
    pub mip: usize,
    pub perturb: bool,
}

impl Default for PhiCfg {
    fn default() -> Self {
        PhiCfg { mech: 5, purview: 4, mip: 6, perturb: true }
    }
}

#[derive(Clone)]
pub struct Mech {
    pub m: Vec<usize>,
    pub pe: Vec<usize>,
    pub pc: Vec<usize>,
    pub gain: Vec<f32>,
    pub in_m: Vec<bool>,
}

#[inline]
fn mask(n: usize, s: &[usize]) -> Vec<bool> {
    let mut m = vec![false; n];
    for &i in s {
        if i < n {
            m[i] = true;
        }
    }
    m
}

fn topk(s: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..s.len()).filter(|&i| s[i] > 0.0).collect();
    idx.sort_unstable_by(|&a, &b| {
        s[b].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// Mécanisme induit par l'hypothèse : masse de pointeur × porte du mécanisme.
/// Le purview d'effet est ce que le mécanisme affecte ; le purview de cause est
/// ce qui l'affecte. Tous deux bornés et ordonnés déterminiquement.
pub fn build_mech(t: &Tape, f: &Fwd, v: &EffView, w: &Wiring, cfg: &PhiCfg) -> Mech {
    let n = v.n();
    let mut gain = vec![0.0f32; v.edges.len()];
    let mut score = vec![0.0f32; w.na];
    for a in 0..w.na {
        let gate = t.at(f.mech, a);
        let mut s = 0.0f32;
        for k in 0..P {
            let eid = w.ptr_eid[a * P + k];
            if eid == usize::MAX {
                continue;
            }
            let pr = t.at(f.ptr, a * (P + 1) + k + 1);
            let m = pr * gate;
            gain[eid] += m;
            s += m;
        }
        score[a] = s;
    }
    let m = topk(&score, cfg.mech);
    let in_m = mask(n, &m);
    let mut we = vec![0.0f32; n];
    let mut wc = vec![0.0f32; n];
    for (ei, e) in v.edges.iter().enumerate() {
        let g = gain[ei];
        if g <= 0.0 {
            continue;
        }
        if in_m[e.src as usize] {
            we[e.dst as usize] += g;
        }
        if in_m[e.dst as usize] {
            wc[e.src as usize] += g;
        }
    }
    let pe = topk(&we, cfg.purview);
    let pc = topk(&wc, cfg.purview);
    Mech { m, pe, pc, gain, in_m }
}

fn mask_itv(v: &EffView, gain: &[f32], keep: impl Fn(usize, usize) -> bool) -> Itv {
    let mut g = vec![0.0f32; v.edges.len()];
    for (ei, e) in v.edges.iter().enumerate() {
        if keep(e.src as usize, e.dst as usize) {
            g[ei] = gain[ei];
        }
    }
    Itv { edge_gain: Some(g), group_gain: [1.0; substrate::NW], rounds: 1 }
}

fn emd_norm(t: &mut Tape, a: usize, b: usize, k: usize) -> usize {
    let d = t.emd(a, b);
    t.scale(d, 1.0 / ((k.max(1) * (Q - 1)) as f32))
}

#[inline]
fn rep_rel(w: &Wiring, k: usize) -> u64 {
    (k * w.rel_count.max(1)) as u64
}

/// Bipartitions candidates. Déterministes, bornées. Une bipartition dont une
/// part est vide est écartée : sinon la MIP serait trivialement nulle partout.
fn candidates(v: &EffView, srcs: &[usize], tgts: &[usize], mip: usize) -> Vec<Vec<bool>> {
    let n = v.n();
    let mut out: Vec<Vec<bool>> = Vec::new();

    let mut lab = vec![false; n];
    for i in 0..n {
        lab[i] = i % 2 == 1;
    }
    out.push(lab);

    let mut lab = vec![false; n];
    for i in 0..n {
        lab[i] = v.nodes[i].scale > 0;
    }
    out.push(lab);

    let mut uni: Vec<usize> = srcs.to_vec();
    uni.extend_from_slice(tgts);
    uni.sort_unstable();
    uni.dedup();
    let mut lab = vec![false; n];
    for (r, &i) in uni.iter().enumerate() {
        lab[i] = r >= uni.len() / 2;
    }
    out.push(lab);

    for &s in srcs.iter().take(3) {
        if let Some(&tt) = tgts.iter().find(|&&x| x != s) {
            let mut lab = vec![false; n];
            lab[s] = true;
            lab[tt] = true;
            out.push(lab);
        }
    }
    out.truncate(mip.max(1));
    out
}

/// φ d'un côté. `information` = EMD au répertoire non contraint (uniforme) ;
/// `intégration` = EMD à la partition minimale. min des deux : pas de
/// compensation entre « c'est informatif » et « c'est intégré ».
fn side_phi(
    t: &mut Tape,
    p: &Params,
    h: usize,
    v: &EffView,
    w: &Wiring,
    srcs: &[usize],
    tgts: &[usize],
    gain: &[f32],
    effect: bool,
    cfg: &PhiCfg,
) -> (usize, usize, usize, u64) {
    let zero = t.scalar(0.0);
    let k = tgts.len();
    if srcs.len() < 2 || k < 2 {
        return (zero, zero, zero, 0);
    }
    let in_s = mask(v.n(), srcs);
    let in_t = mask(v.n(), tgts);
    let whole_itv = mask_itv(v, gain, |s, d| in_s[s] && in_t[d]);
    let whole = substrate::local_rep(t, p, h, w, tgts, &whole_itv, effect);
    let mut rel = rep_rel(w, k);

    let u = t.input(k, Q, &vec![1.0 / Q as f32; k * Q]);
    let info = emd_norm(t, whole, u, k);

    let mut best: Option<usize> = None;
    for lab in candidates(v, srcs, tgts, cfg.mip) {
        let s1: Vec<usize> = srcs.iter().cloned().filter(|&i| !lab[i]).collect();
        let s2: Vec<usize> = srcs.iter().cloned().filter(|&i| lab[i]).collect();
        let t1: Vec<usize> = tgts.iter().cloned().filter(|&i| !lab[i]).collect();
        let t2: Vec<usize> = tgts.iter().cloned().filter(|&i| lab[i]).collect();
        if s1.is_empty() || s2.is_empty() || t1.is_empty() || t2.is_empty() {
            continue;
        }
        let i1 = mask_itv(v, gain, |s, d| !lab[s] && !lab[d] && in_s[s] && in_t[d]);
        let i2 = mask_itv(v, gain, |s, d| lab[s] && lab[d] && in_s[s] && in_t[d]);
        let ra = substrate::local_rep(t, p, h, w, &t1, &i1, effect);
        let rb = substrate::local_rep(t, p, h, w, &t2, &i2, effect);
        rel += rep_rel(w, t1.len()) + rep_rel(w, t2.len());
        let cat = t.concat_rows(ra, rb);
        let mut pos = vec![0usize; k];
        let (mut c1, mut c2) = (0usize, 0usize);
        for (i, &nd) in tgts.iter().enumerate() {
            if !lab[nd] {
                pos[i] = c1;
                c1 += 1;
            } else {
                pos[i] = t1.len() + c2;
                c2 += 1;
            }
        }
        let rp = t.gather_rows(cat, &pos);
        let d = emd_norm(t, whole, rp, k);
        best = Some(match best {
            None => d,
            Some(b) => t.min2(b, d),
        });
    }
    let integ = best.unwrap_or_else(|| t.scalar(0.0));
    let ph = t.min2(info, integ);
    (ph, info, integ, rel)
}

fn shift_srcs(v: &EffView, w: &Wiring, m: &[usize], kind: u8, fallback_adj: bool) -> Vec<usize> {
    let mut out = Vec::with_capacity(m.len());
    for &a in m {
        let mut found = None;
        for &ei in w.inc[a].iter() {
            let e = v.edges[ei];
            if e.kind == kind && e.src as usize != a {
                found = Some(e.src as usize);
                break;
            }
        }
        match found {
            Some(x) => out.push(x),
            None => {
                if fallback_adj && w.na > 1 {
                    out.push(if a + 1 < w.na { a + 1 } else { a - 1 });
                } else {
                    out.push(a);
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Stabilité différentielle : le répertoire continue-t-il d'expliquer quand on
/// essaie activement de le casser ? Trois interventions minimales.
fn stability(
    t: &mut Tape,
    p: &Params,
    h: usize,
    v: &EffView,
    w: &Wiring,
    mech: &Mech,
    whole: usize,
) -> (usize, u64) {
    let k = mech.pe.len();
    if k == 0 || mech.m.is_empty() {
        return (t.scalar(1.0), 0);
    }
    let in_pe = mask(v.n(), &mech.pe);
    let in_m = mask(v.n(), &mech.m);
    let mut rel = 0u64;
    let zero = t.scalar(0.0);
    let one = t.scalar(1.0);
    let mut best = one;

    // 1. réfutation : on coupe l'arête la plus porteuse du mécanisme.
    let mut g2 = mech.gain.clone();
    let mut bi = usize::MAX;
    let mut bv = 0.0f32;
    for (ei, e) in v.edges.iter().enumerate() {
        if in_m[e.src as usize] && in_pe[e.dst as usize] && mech.gain[ei] > bv {
            bv = mech.gain[ei];
            bi = ei;
        }
    }
    if bi != usize::MAX {
        g2[bi] = 0.0;
    }
    let i_ref = mask_itv(v, &g2, |s, d| in_m[s] && in_pe[d]);
    let r_ref = substrate::local_rep(t, p, h, w, &mech.pe, &i_ref, true);
    rel += rep_rel(w, k);
    let d = emd_norm(t, whole, r_ref, k);
    let nd = t.neg(d);
    let s1 = t.add_const(nd, 1.0);
    let s1 = t.max2(zero, s1);
    best = t.min2(best, s1);

    // 2. déplacement : re-indexation relationnelle du mécanisme.
    let md = shift_srcs(v, w, &mech.m, REL_ADJ, true);
    let in_md = mask(v.n(), &md);
    let i_dis = mask_itv(v, &mech.gain, |s, d| in_md[s] && in_pe[d]);
    let r_dis = substrate::local_rep(t, p, h, w, &mech.pe, &i_dis, true);
    rel += rep_rel(w, k);
    let d = emd_norm(t, whole, r_dis, k);
    let nd = t.neg(d);
    let s2 = t.add_const(nd, 1.0);
    let s2 = t.max2(zero, s2);
    best = t.min2(best, s2);

    // 3. abstraction : grossissement par containment.
    let ma = shift_srcs(v, w, &mech.m, REL_CONTAIN, false);
    let in_ma = mask(v.n(), &ma);
    let i_abs = mask_itv(v, &mech.gain, |s, d| in_ma[s] && in_pe[d]);
    let r_abs = substrate::local_rep(t, p, h, w, &mech.pe, &i_abs, true);
    rel += rep_rel(w, k);
    let d = emd_norm(t, whole, r_abs, k);
    let nd = t.neg(d);
    let s3 = t.add_const(nd, 1.0);
    let s3 = t.max2(zero, s3);
    best = t.min2(best, s3);

    (best, rel)
}

pub struct PhiVal {
    pub sharp: usize,
    pub bigphi: usize,
    pub stab: usize,
    pub cov: usize,
    pub res_vec: usize,
    pub info_e: usize,
    pub integ_e: usize,
    pub info_c: usize,
    pub integ_c: usize,
    pub sharp_f: f32,
    pub bigphi_f: f32,
    pub stab_f: f32,
    pub cov_f: f32,
    pub res_node: usize,
    pub rel: u64,
}

pub fn evaluate(
    t: &mut Tape,
    p: &Params,
    v: &EffView,
    w: &Wiring,
    f: &Fwd,
    mech: &Mech,
    cfg: &PhiCfg,
) -> PhiVal {
    let (ph_e, info_e, integ_e, rel_e) =
        side_phi(t, p, f.h, v, w, &mech.m, &mech.pe, &mech.gain, true, cfg);
    let (ph_c, info_c, integ_c, rel_c) =
        side_phi(t, p, f.h, v, w, &mech.pc, &mech.m, &mech.gain, false, cfg);
    let bigphi = t.min2(ph_e, ph_c);

    let in_m = mask(v.n(), &mech.m);
    let in_pe = mask(v.n(), &mech.pe);
    let witv = mask_itv(v, &mech.gain, |s, d| in_m[s] && in_pe[d]);
    let whole = substrate::local_rep(t, p, f.h, w, &mech.pe, &witv, true);
    let (stab, rel_s) = if cfg.perturb {
        stability(t, p, f.h, v, w, mech, whole)
    } else {
        (t.scalar(1.0), 0)
    };
    let sharp = t.mul(bigphi, stab);

    // couverture douce : masse hors NULL. Différentiable par la tête de pointeur.
    let mut sel = vec![0.0f32; P + 1];
    sel[0] = 1.0;
    let selv = t.input(P + 1, 1, &sel);
    let nullcol = t.matmul(f.ptr, selv);
    let one = t.input(w.na, 1, &vec![1.0; w.na]);
    let hit = t.sub(one, nullcol);
    let cov = t.mean_all(hit);

    let atoms: Vec<usize> = (0..w.na).collect();
    let res_a = t.gather_rows(f.res, &atoms);
    let res_vec = t.mul(nullcol, res_a);
    let mut res_node = 0usize;
    let mut rv = f32::NEG_INFINITY;
    for a in 0..w.na {
        let x = t.at(res_vec, a);
        if x > rv {
            rv = x;
            res_node = a;
        }
    }

    PhiVal {
        sharp,
        bigphi,
        stab,
        cov,
        res_vec,
        info_e,
        integ_e,
        info_c,
        integ_c,
        sharp_f: t.scalar_of(sharp),
        bigphi_f: t.scalar_of(bigphi),
        stab_f: t.scalar_of(stab),
        cov_f: t.scalar_of(cov),
        res_node,
        rel: rel_e + rel_c + rel_s + rep_rel(w, mech.pe.len()) + (w.na * (P + 1)) as u64,
    }
}

/// Sonde d'addition contrefactuelle : on rend au résidu désigné la masse qui lui
/// manque, et on mesure si Φ monte. C'est le terme qui interdit au système
/// d'optimiser la seule structure de son hypothèse.
pub fn residual_gain(
    t: &mut Tape,
    p: &Params,
    v: &EffView,
    w: &Wiring,
    f: &Fwd,
    mech: &Mech,
    cfg: &PhiCfg,
    extra: usize,
    base: usize,
) -> usize {
    if extra >= w.na || mech.in_m[extra] {
        return t.scalar(0.0);
    }
    let mut k0 = 0usize;
    let mut bw = f32::NEG_INFINITY;
    for k in 0..P {
        let x = w.ptr_w[extra * P + k];
        if x > bw {
            bw = x;
            k0 = k;
        }
    }
    let eid = w.ptr_eid[extra * P + k0];
    if eid == usize::MAX {
        return t.scalar(0.0);
    }
    let nz = mech.gain.iter().filter(|&&x| x > 0.0).count().max(1);
    let mean = mech.gain.iter().sum::<f32>() / nz as f32;
    let mut g2 = mech.gain.clone();
    if g2[eid] < mean {
        g2[eid] = mean;
    }
    let mut m2 = mech.m.clone();
    m2.push(extra);
    let (a, _, _, _) = side_phi(t, p, f.h, v, w, &m2, &mech.pe, &g2, true, cfg);
    let (b, _, _, _) = side_phi(t, p, f.h, v, w, &mech.pc, &m2, &g2, false, cfg);
    let with = t.min2(a, b);
    t.sub(with, base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eff::{build, ViewCfg};
    use crate::rng::Rng;

    fn setup(n: usize) -> (Vec<u8>, EffView, Wiring, Params) {
        let b: Vec<u8> = (0..n).map(|i| ((i * 3) % 7) as u8).collect();
        let v = build(&b, &ViewCfg::default());
        let w = substrate::wire(&v);
        let mut rng = Rng::new(21);
        let p = substrate::init(&mut rng);
        (b, v, w, p)
    }

    #[test]
    fn phi_borne_et_fini() {
        let (_b, v, w, p) = setup(96);
        let mut t = Tape::new();
        let f = substrate::forward(&mut t, &p, &v, &w, &Itv::default());
        let m = build_mech(&t, &f, &v, &w, &PhiCfg::default());
        let pv = evaluate(&mut t, &p, &v, &w, &f, &m, &PhiCfg::default());
        assert!(pv.sharp_f.is_finite());
        assert!(pv.sharp_f >= 0.0 && pv.sharp_f <= 1.0, "sharp {}", pv.sharp_f);
        assert!(pv.cov_f >= 0.0 && pv.cov_f <= 1.0);
        assert!(pv.rel > 0);
    }

    /// Témoin structurel : un mécanisme réduit aux boucles SELF n'a AUCUNE
    /// bipartition qui coupe quoi que ce soit -> intégration nulle -> φ nul.
    #[test]
    fn mecanisme_separe_integration_nulle() {
        let (_b, v, w, p) = setup(96);
        let mut t = Tape::new();
        let f = substrate::forward(&mut t, &p, &v, &w, &Itv::default());
        let mut m = build_mech(&t, &f, &v, &w, &PhiCfg::default());
        let mut g = vec![0.0f32; v.edges.len()];
        for (ei, e) in v.edges.iter().enumerate() {
            if e.kind == crate::eff::REL_SELF {
                g[ei] = 1.0;
            }
        }
        m.gain = g;
        m.pe = m.m.clone();
        m.pc = m.m.clone();
        let cfg = PhiCfg::default();
        let (ph, _i, integ, _) =
            side_phi(&mut t, &p, f.h, &v, &w, &m.m, &m.pe, &m.gain, true, &cfg);
        assert_eq!(t.scalar_of(integ), 0.0);
        assert_eq!(t.scalar_of(ph), 0.0);
    }
}