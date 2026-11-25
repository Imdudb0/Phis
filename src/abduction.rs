//! Recherche abductive. Le seul méta-a-priori.
//!
//! L'état de recherche est un ensemble de GAINS persistants sur les relations
//! (arêtes et types). Un opérateur ne manipule jamais un programme symbolique :
//! il modifie la dynamique relationnelle, et l'hypothèse est la configuration
//! d'activations qui en résulte. Le programme de pointeurs n'est lu qu'à la fin,
//! comme mesure. C'est ce qui interdit le neuro-symbolique déguisé.
//!
//! Budget : `op` = une application d'opérateur ; `rel` = applications de relation
//! (arêtes × tours pour la dynamique, cibles × slots pour les répertoires).
//! Budget nul = encodage + une lecture, zéro opérateur : c'est la condition
//! d'ablation « la réponse était-elle déjà là ? ».
//!
//! Apprentissage : gradient du terme différentiable J sur l'état atteint, plus
//! REINFORCE sur le choix discret (opérateur, site) avec avantage = Δφ_sharp.
//! Aucun texte, aucune définition, aucun rendu de H n'entre dans ce gradient.

use crate::ad::{accumulate, Params, Tape};
use crate::eff::{EffView, REL_ADJ, REL_CONTAIN, REL_EQ, REL_ORD, REL_SIM};
use crate::phi::{self, Mech, PhiCfg, PhiVal};
use crate::rng::Rng;
use crate::substrate::{self, Fwd, Itv, Program, Wiring, N_OPS, P};

pub const OP_BIND: usize = 0;
pub const OP_COMPOSE: usize = 1;
pub const OP_SPLIT: usize = 2;
pub const OP_LIFT: usize = 3;
pub const OP_DISPLACE: usize = 4;
pub const OP_REFUTE: usize = 5;
pub const OP_INSTANTIATE: usize = 6;
pub const OP_HALT: usize = 7;

pub const OP_NAMES: [&str; N_OPS] =
    ["bind", "compose", "split", "lift", "displace", "refute", "instantiate", "halt"];

#[derive(Clone, Copy, Debug)]
pub struct SearchCfg {
    pub max_ops: usize,
    pub max_rel: u64,
    pub rounds: usize,
    pub allow_refutation: bool,
    pub explore: bool,
    pub phi: PhiCfg,
    pub lambda_cov: f32,
    pub mu_res: f32,
}

impl Default for SearchCfg {
    fn default() -> Self {
        SearchCfg {
            max_ops: 48,
            max_rel: 400_000_000,
            rounds: 2,
            allow_refutation: true,
            explore: true,
            phi: PhiCfg::default(),
            lambda_cov: 0.25,
            mu_res: 0.10,
        }
    }
}

impl SearchCfg {
    /// Ablation « budget coupé ».
    pub fn budget_zero() -> Self {
        SearchCfg { max_ops: 0, max_rel: 0, ..Self::default() }
    }

    /// Ablation « réfutation désactivée ».
    pub fn no_refutation() -> Self {
        SearchCfg { allow_refutation: false, ..Self::default() }
    }
}

#[derive(Clone, Default)]
pub struct Trace {
    pub step: Vec<usize>,
    pub ops: Vec<usize>,
    pub rel: Vec<u64>,
    pub sharp: Vec<f32>,
    pub bigphi: Vec<f32>,
    pub stab: Vec<f32>,
    pub cov: Vec<f32>,
    pub op: Vec<u8>,
    pub site: Vec<u32>,
    pub residue: Vec<u32>,
}

impl Trace {
    fn push(&mut self, step: usize, ops: usize, rel: u64, pv: &PhiVal, op: u8, site: u32) {
        self.step.push(step);
        self.ops.push(ops);
        self.rel.push(rel);
        self.sharp.push(pv.sharp_f);
        self.bigphi.push(pv.bigphi_f);
        self.stab.push(pv.stab_f);
        self.cov.push(pv.cov_f);
        self.op.push(op);
        self.site.push(site);
        self.residue.push(pv.res_node as u32);
    }

    pub fn len(&self) -> usize {
        self.step.len()
    }

    pub fn is_empty(&self) -> bool {
        self.step.is_empty()
    }
}

pub struct Outcome {
    pub program: Program,
    pub trace: Trace,
    pub phi: f32,
    pub bigphi: f32,
    pub stab: f32,
    pub cov: f32,
    pub residue: Vec<(usize, f32)>,
    pub ops: usize,
    pub rel: u64,
    pub itv: Itv,
}

#[inline]
fn forward_rel(v: &EffView, rounds: usize) -> u64 {
    (v.edges.len() as u64) * ((rounds + 1) as u64)
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let mx = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut o: Vec<f32> = x.iter().map(|v| (v - mx).clamp(-30.0, 30.0).exp()).collect();
    let s: f32 = o.iter().sum();
    let inv = 1.0 / s.max(1e-12);
    for v in o.iter_mut() {
        *v *= inv;
    }
    o
}

fn argmax(x: &[f32]) -> usize {
    let mut b = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv {
            bv = v;
            b = i;
        }
    }
    b
}

fn choose(t: &Tape, f: &Fwd, cfg: &SearchCfg, rng: &mut Rng, w: &Wiring) -> (usize, usize) {
    let mut lg: Vec<f32> = (0..N_OPS).map(|i| t.at(f.policy, i)).collect();
    if !cfg.allow_refutation {
        lg[OP_REFUTE] = -1e9;
    }
    let pr = softmax(&lg);
    let op = if cfg.explore { rng.weighted(&pr) } else { argmax(&pr) };
    let sl: Vec<f32> = (0..w.na).map(|a| t.at(f.prop, a)).collect();
    let sp = softmax(&sl);
    let site = if cfg.explore { rng.weighted(&sp) } else { argmax(&sp) };
    (op, site)
}

/// rank-ième candidat de pointeur d'un atome, lu dans la distribution courante.
fn top_edge(t: &Tape, f: &Fwd, w: &Wiring, a: usize, rank: usize) -> Option<usize> {
    if a >= w.na {
        return None;
    }
    let mut s: Vec<(f32, usize)> = Vec::with_capacity(P);
    for k in 0..P {
        let eid = w.ptr_eid[a * P + k];
        if eid == usize::MAX {
            continue;
        }
        s.push((t.at(f.ptr, a * (P + 1) + k + 1), eid));
    }
    s.sort_unstable_by(|x, y| {
        y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal).then(x.1.cmp(&y.1))
    });
    s.get(rank).map(|&(_, e)| e)
}

fn neighbor(v: &EffView, w: &Wiring, a: usize, kind: u8) -> Option<usize> {
    for &ei in w.inc[a].iter() {
        let e = v.edges[ei];
        if e.kind == kind && e.src as usize != a {
            return Some(e.src as usize);
        }
    }
    None
}

#[inline]
fn cl(x: f32) -> f32 {
    x.clamp(0.05, 4.0)
}

/// Les huit opérateurs. Tous agissent sur des gains relationnels : le substrat
/// reste le seul lieu de la représentation.
fn apply_op(op: usize, site: usize, itv: &Itv, v: &EffView, w: &Wiring, t: &Tape, f: &Fwd) -> Itv {
    let mut g = itv.edge_gain.clone().unwrap_or_else(|| vec![1.0; v.edges.len()]);
    let mut gg = itv.group_gain;
    let a = site.min(w.na.saturating_sub(1));
    match op {
        OP_BIND => {
            if let Some(e) = top_edge(t, f, w, a, 0) {
                g[e] = cl(g[e] * 1.6);
            }
        }
        OP_COMPOSE => {
            if let Some(e) = top_edge(t, f, w, a, 0) {
                g[e] = cl(g[e] * 1.4);
                let s = v.edges[e].src as usize;
                if s < w.na {
                    if let Some(e2) = top_edge(t, f, w, s, 0) {
                        g[e2] = cl(g[e2] * 1.4);
                    }
                }
            }
        }
        OP_SPLIT => {
            let mut l = w.inc[a].clone();
            l.sort_unstable_by(|&x, &y| {
                v.edges[y]
                    .w
                    .partial_cmp(&v.edges[x].w)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(x.cmp(&y))
            });
            for &e in l.iter().rev().take(2) {
                g[e] = cl(g[e] * 0.4);
            }
        }
        OP_LIFT => {
            gg[REL_CONTAIN as usize * 2] = cl(gg[REL_CONTAIN as usize * 2] * 1.3);
            gg[REL_CONTAIN as usize * 2 + 1] = cl(gg[REL_CONTAIN as usize * 2 + 1] * 1.3);
            gg[REL_ADJ as usize * 2] = cl(gg[REL_ADJ as usize * 2] * 0.85);
            gg[REL_ORD as usize * 2] = cl(gg[REL_ORD as usize * 2] * 0.85);
            if let Some(e) = neighbor(v, w, a, REL_CONTAIN) {
                let _ = e;
            }
        }
        OP_DISPLACE => {
            if let Some(nb) = neighbor(v, w, a, REL_ADJ) {
                if let Some(e) = top_edge(t, f, w, a, 0) {
                    g[e] = cl(g[e] * 0.6);
                }
                if nb < w.na {
                    if let Some(e2) = top_edge(t, f, w, nb, 0) {
                        g[e2] = cl(g[e2] * 1.5);
                    }
                }
            }
        }
        OP_REFUTE => {
            if let Some(e) = top_edge(t, f, w, a, 0) {
                g[e] = cl(g[e] * 0.15);
            }
            if let Some(e2) = top_edge(t, f, w, a, 1) {
                g[e2] = cl(g[e2] * 1.5);
            }
        }
        OP_INSTANTIATE => {
            gg[REL_CONTAIN as usize * 2] = cl(gg[REL_CONTAIN as usize * 2] * 0.7);
            gg[REL_CONTAIN as usize * 2 + 1] = cl(gg[REL_CONTAIN as usize * 2 + 1] * 0.7);
            gg[REL_EQ as usize * 2] = cl(gg[REL_EQ as usize * 2] * 1.2);
            gg[REL_SIM as usize * 2] = cl(gg[REL_SIM as usize * 2] * 1.2);
        }
        _ => {}
    }
    Itv { edge_gain: Some(g), group_gain: gg, rounds: itv.rounds }
}

fn logprob(t: &mut Tape, f: &Fwd, op: usize, site: usize, na: usize, allow_ref: bool) -> usize {
    let pol = if allow_ref {
        f.policy
    } else {
        let mut m = vec![0.0f32; N_OPS];
        m[OP_REFUTE] = -1e9;
        let mv = t.input(1, N_OPS, &m);
        t.add(f.policy, mv)
    };
    let lp = t.log_softmax_rows(pol);
    let o = t.gather_rows(lp, &[op]);
    let lpo = t.sum_all(o);

    // log-softmax sur les sites sans transposition : le max est une constante
    // (log-sum-exp est invariant par décalage, le gradient est exact).
    let idx: Vec<usize> = (0..na).collect();
    let col = t.gather_rows(f.prop, &idx);
    let mx = col_val_max(t, col);
    let sh = t.add_const(col, -mx);
    let ex = t.exp(sh);
    let s = t.sum_all(ex);
    let ls = t.log(s);
    let lse = t.add_const(ls, mx);
    let xs = t.gather_rows(f.prop, &[site]);
    let x = t.sum_all(xs);
    let lps = t.sub(x, lse);
    t.add(lpo, lps)
}

fn col_val_max(t: &Tape, v: usize) -> f32 {
    t.val(v).iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

fn objective(t: &mut Tape, pv: &PhiVal, cfg: &SearchCfg) -> usize {
    let one = t.scalar(1.0);
    let miss = t.sub(one, pv.cov);
    let pen = t.scale(miss, cfg.lambda_cov);
    t.sub(pv.sharp, pen)
}

pub fn search(
    p: &Params,
    bytes: &[u8],
    v: &EffView,
    w: &Wiring,
    cfg: &SearchCfg,
    seed: u64,
    mut grads: Option<&mut Vec<Vec<f32>>>,
) -> Outcome {
    let mut rng = Rng::new(seed);
    let mut itv = Itv {
        edge_gain: Some(vec![1.0; v.edges.len()]),
        group_gain: [1.0; substrate::NW],
        rounds: cfg.rounds.max(1),
    };
    let mut trace = Trace::default();
    let mut rel: u64 = 0;
    let mut ops: usize = 0;
    let mut baseline = 0.0f32;
    let mut nrew = 0usize;

    let mut t0 = Tape::new();
    let mut f0 = substrate::forward(&mut t0, p, v, w, &itv);
    rel += forward_rel(v, itv.rounds);
    let m = phi::build_mech(&t0, &f0, v, w, &cfg.phi);
    let mut pv0 = phi::evaluate(&mut t0, p, v, w, &f0, &m, &cfg.phi);
    rel += pv0.rel;
    trace.push(0, ops, rel, &pv0, 255, 0);

    while ops < cfg.max_ops && rel < cfg.max_rel {
        let (op, site) = choose(&t0, &f0, cfg, &mut rng, w);
        if op == OP_HALT {
            trace.push(ops + 1, ops, rel, &pv0, OP_HALT as u8, site as u32);
            break;
        }
        let itv2 = apply_op(op, site, &itv, v, w, &t0, &f0);
        let mut t1 = Tape::new();
        let f1 = substrate::forward(&mut t1, p, v, w, &itv2);
        rel += forward_rel(v, itv2.rounds);
        let m1 = phi::build_mech(&t1, &f1, v, w, &cfg.phi);
        let pv1 = phi::evaluate(&mut t1, p, v, w, &f1, &m1, &cfg.phi);
        rel += pv1.rel;
        ops += 1;

        let reward = pv1.sharp_f - pv0.sharp_f;
        if let Some(g) = grads.as_deref_mut() {
            let adv = reward - baseline;
            let lp = logprob(&mut t0, &f0, op, site, w.na, cfg.allow_refutation);
            let loss = t0.scale(lp, -adv);
            let gb = t0.backward(loss, p.nblocks());
            accumulate(g, &gb);

            let mut j = objective(&mut t1, &pv1, cfg);
            if cfg.mu_res > 0.0 {
                let rg =
                    phi::residual_gain(&mut t1, p, v, w, &f1, &m1, &cfg.phi, pv1.res_node, pv1.bigphi);
                let rgs = t1.scale(rg, cfg.mu_res);
                j = t1.add(j, rgs);
            }
            let lj = t1.neg(j);
            let gb1 = t1.backward(lj, p.nblocks());
            accumulate(g, &gb1);
        }
        nrew += 1;
        baseline += (reward - baseline) / nrew as f32;

        trace.push(ops, ops, rel, &pv1, op as u8, site as u32);
        itv = itv2;
        t0 = t1;
        f0 = f1;
        pv0 = pv1;
    }

    let program = substrate::read_program(&t0, &f0, bytes, v, w);
    let mut res: Vec<(usize, f32)> = (0..w.na).map(|a| (a, t0.at(pv0.res_vec, a))).collect();
    res.sort_unstable_by(|x, y| {
        y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal).then(x.0.cmp(&y.0))
    });
    res.truncate(8);

    Outcome {
        program,
        trace,
        phi: pv0.sharp_f,
        bigphi: pv0.bigphi_f,
        stab: pv0.stab_f,
        cov: pv0.cov_f,
        residue: res,
        ops,
        rel,
        itv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eff::{build, ViewCfg};
    use crate::rng::Rng;

    fn setup() -> (Vec<u8>, EffView, Wiring, Params) {
        let b: Vec<u8> = (0..72).map(|i| ((i * 5) % 9) as u8).collect();
        let v = build(&b, &ViewCfg::default());
        let w = substrate::wire(&v);
        let mut rng = Rng::new(31);
        let p = substrate::init(&mut rng);
        (b, v, w, p)
    }

    fn small() -> SearchCfg {
        SearchCfg {
            max_ops: 2,
            max_rel: 50_000_000,
            phi: PhiCfg { mech: 3, purview: 3, mip: 2, perturb: true },
            ..SearchCfg::default()
        }
    }

    #[test]
    fn budget_nul_est_une_lecture_seule() {
        let (b, v, w, p) = setup();
        let cfg = SearchCfg { max_ops: 0, ..small() };
        let o = search(&p, &b, &v, &w, &cfg, 7, None);
        assert_eq!(o.ops, 0);
        assert_eq!(o.trace.len(), 1, "budget nul : aucun opérateur, une lecture");
    }

    #[test]
    fn determinisme_de_trajectoire() {
        let (b, v, w, p) = setup();
        let cfg = small();
        let a = search(&p, &b, &v, &w, &cfg, 1234, None);
        let c = search(&p, &b, &v, &w, &cfg, 1234, None);
        assert_eq!(a.trace.op, c.trace.op);
        assert_eq!(a.trace.site, c.trace.site);
        assert_eq!(a.trace.sharp, c.trace.sharp);
        assert_eq!(a.rel, c.rel);
    }

    #[test]
    fn ablation_refutation_retire_operateur() {
        let (b, v, w, p) = setup();
        let cfg = SearchCfg { max_ops: 3, allow_refutation: false, ..small() };
        let o = search(&p, &b, &v, &w, &cfg, 99, None);
        assert!(o.trace.op.iter().all(|&x| x != OP_REFUTE as u8));
    }

    #[test]
    fn budget_monotone() {
        let (b, v, w, p) = setup();
        let o = search(&p, &b, &v, &w, &small(), 5, None);
        for i in 1..o.trace.len() {
            assert!(o.trace.rel[i] >= o.trace.rel[i - 1]);
            assert!(o.trace.ops[i] >= o.trace.ops[i - 1]);
        }
    }

    #[test]
    fn gradient_non_nul() {
        let (b, v, w, p) = setup();
        let mut g: Vec<Vec<f32>> = p.data.iter().map(|x| vec![0.0f32; x.len()]).collect();
        let _ = search(&p, &b, &v, &w, &small(), 17, Some(&mut g));
        let n: f32 = g.iter().map(|x| x.iter().map(|v| v.abs()).sum::<f32>()).sum();
        assert!(n > 0.0, "le signal φ_sharp doit produire un gradient");
    }
}