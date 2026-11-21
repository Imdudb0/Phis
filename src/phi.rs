use crate::graph::*;
use crate::dynamics::regenerate;

const K: usize = 3;
const PHI_L: usize = 24;

struct Rep { hist: Vec<[f32; 256]>, gh: Vec<[f32; 8]> }

fn rollouts(g: &Graph, ws: &[S], c0: &[S], l: usize, cut: Option<EdgeKind>,
            rng: &mut Rng, budget: &mut Budget) -> Rep {
    let mut hist = vec![[0.0; 256]; l];
    let mut gh = vec![[0.0; 8]; N_G];
    for _ in 0..K {
        let mut c = c0.to_vec();
        for v in c.iter_mut() { for d in 0..D { v[d] += rng.norm() * 0.05; } }
        let mut x = zero_state();
        x[WS0..C0].copy_from_slice(ws);
        x[C0..G0].copy_from_slice(&c);
        let gain = vec![1.0f32; N];
        for p in 0..l {
            x = gen_step(g, &x, &gain, 0.05, rng, budget, cut);
            let pr = softmax_bytes(&x[0..N_BYTE]);
            for b in 0..256 { hist[p][b] += pr[b]; }
            for gi in 0..N_G {
                let v = x[G0 + gi][0];
                let bin = (((v * 0.5 + 0.5) * 8.0).floor() as usize).min(7);
                gh[gi][bin] += 1.0;
            }
        }
    }
    for p in 0..l { let s: f32 = hist[p].iter().sum(); if s > 0.0 { for b in 0..256 { hist[p][b] /= s; } } }
    for gi in 0..N_G { let s: f32 = gh[gi].iter().sum(); if s > 0.0 { for b in 0..8 { gh[gi][b] /= s; } } }
    Rep { hist, gh }
}

fn divergence(a: &Rep, b: &Rep) -> f32 {
    let l = a.hist.len().min(b.hist.len());
    if l == 0 { return 0.0; }
    let mut s = 0.0;
    for p in 0..l { s += tv(&a.hist[p], &b.hist[p]); }
    s / l as f32
}

// Puissance cause-effet intégrée, relative à E.
// intégration = irréductibilité à la plus petite partition (MIP approchée sur coupes dirigées).
pub fn compute(g: &Graph, ws: &[S], c0: &[S], e: &[u8], rng: &mut Rng, budget: &mut Budget) -> f32 {
    let l = e.len().min(PHI_L);
    if l == 0 { return 0.0; }
    let whole = rollouts(g, ws, c0, l, None, rng, budget);
    let mut min_div = f32::INFINITY;
    for cut in [EdgeKind::CG, EdgeKind::WsC, EdgeKind::CC, EdgeKind::GByte] {
        let part = rollouts(g, ws, c0, l, Some(cut), rng, budget);
        let d = divergence(&whole, &part);
        if d < min_div { min_div = d; }
    }
    if !min_div.is_finite() { min_div = 0.0; }
    let integration = min_div;
    let mut cp = 0.0;
    for p in 0..l { cp += entropy_bits(&whole.hist[p]) / 8.0; }
    let cause_power = (1.0 - cp / l as f32).clamp(0.0, 1.0);
    let mut ep = 0.0;
    for gi in 0..N_G { ep += entropy_bits(&whole.gh[gi]) / 3.0; }
    let effect_power = (1.0 - ep / N_G as f32).clamp(0.0, 1.0);
    let mut hit = 0;
    for p in 0..l { if argmax_byte(&whole.hist[p]) == e[p] { hit += 1; } }
    let fid = hit as f32 / l as f32;
    (fid * integration * 0.5 * (cause_power + effect_power)).clamp(0.0, 1.0)
}

// Résidu causal seulement: responsabilité par lésion de la cause, aucune similarité lexicale.
pub fn responsibility(g: &Graph, ws: &[S], c0: &[S], e: &[u8], base: &[[f32; 256]],
                      rng: &mut Rng, budget: &mut Budget) -> Vec<Vec<f32>> {
    let l = base.len();
    let mut resp = vec![vec![0.0; l]; N_C];
    for k in 0..N_C {
        let (_, pk) = regenerate(g, ws, c0, l, 0.0, None, Some(k), rng, budget);
        for p in 0..l { resp[k][p] = (base[p][e[p] as usize] - pk[p][e[p] as usize]).max(0.0); }
    }
    resp
}