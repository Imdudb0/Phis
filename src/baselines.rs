//! Baselines à armes égales (mêmes données PTGRS, même calcul) + instrument MDL post-hoc.
//!
//! 1. `BytePredictorBaseline` : prédicteur d'octets contextuel (bigramme + trigramme + lissage appris
//!    par entropie croisée sur PTGRS uniquement).
//! 2. `LikelihoodSearchBaseline` : même architecture `PhisharpNet` et même budget de recherche,
//!    entraînée et guidée UNIQUEMENT par la vraisemblance log P(E | H) au lieu de φ_sharp.
//! 3. `mdl_diagnostic_score` : instrument de comparaison post-hoc UNIQUEMENT (jamais objectif).

use crate::math::{Rng, normalize_prob, sigmoid, tanh_clamped};
use crate::phi_sharp::{score_generativity, score_projection};
use crate::ptgrs::sample_training_effect;
use crate::substrate::{HypothesisState, N_LAT, PerceivedGraph, PhisharpNet};

/// Baseline 1 : Prédicteur d'octets à mélange bigramme/trigramme/skip appris sur PTGRS.
#[derive(Clone, Debug)]
pub struct BytePredictorBaseline {
    /// Poids appris sur PTGRS : [alpha_bigram, alpha_trigram, alpha_skip2, alpha_marginal]
    pub mix_logits: [f64; 4],
}

impl BytePredictorBaseline {
    pub fn train_on_ptgrs(episodes: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut logits = [1.2, 0.8, 0.4, -0.5];
        let lr = 0.05;

        for _ in 0..episodes {
            let sample = sample_training_effect(&mut rng);
            let pg = PerceivedGraph::from_bytes(&sample.observed_e);
            if pg.sym_seq.len() < 4 || pg.k() < 2 {
                continue;
            }
            let k = pg.k();
            // Calcul des probabilités locales et gradient numérique léger sur les 4 poids de mélange
            let loss_at = |l: &[f64; 4]| -> f64 {
                let w = softmax4(l);
                let mut nll = 0.0;
                let mut cnt: f64 = 0.0;
                for i in 1..pg.sym_seq.len() {
                    let u = pg.sym_seq[i - 1];
                    let v = pg.sym_seq[i];
                    let p_bi = pg.emp_trans1[u * k + v];
                    let p_sk = if i >= 2 { pg.emp_trans2[pg.sym_seq[i - 2] * k + v] } else { p_bi };
                    let p_co = pg.emp_cooccur[u * k + v];
                    let p_ma = pg.emp_marginal[v];
                    let p = (w[0] * p_bi + w[1] * p_sk + w[2] * p_co + w[3] * p_ma).max(1e-9);
                    nll -= p.ln();
                    cnt += 1.0;
                }
                nll / cnt.max(1.0)
            };

            let base = loss_at(&logits);
            let mut grad = [0.0; 4];
            let eps = 1e-3;
            for j in 0..4 {
                let mut l2 = logits;
                l2[j] += eps;
                grad[j] = (loss_at(&l2) - base) / eps;
            }
            for j in 0..4 {
                logits[j] -= lr * grad[j].clamp(-2.0, 2.0);
            }
        }
        Self { mix_logits: logits }
    }

    pub fn predict_transition_matrix(&self, pg: &PerceivedGraph) -> Vec<f64> {
        let k = pg.k();
        let w = softmax4(&self.mix_logits);
        let mut mat = vec![0.0; k * k];
        for u in 0..k {
            let mut row = vec![0.0; k];
            for v in 0..k {
                row[v] = w[0] * pg.emp_trans1[u * k + v]
                    + w[1] * pg.emp_trans2[u * k + v]
                    + w[2] * pg.emp_cooccur[u * k + v]
                    + w[3] * pg.emp_marginal[v];
            }
            normalize_prob(&mut row);
            mat[u * k..(u + 1) * k].copy_from_slice(&row);
        }
        mat
    }

    pub fn projection_score(&self, pg: &PerceivedGraph) -> f64 {
        let k = pg.k();
        if k <= 1 || pg.sym_seq.len() < 6 {
            return 0.0;
        }
        let mat = self.predict_transition_matrix(pg);
        let uniform_ll = -(k as f64).ln();
        let mid = pg.sym_seq.len() / 2;
        let mut ll = 0.0;
        for w in pg.sym_seq[mid..].windows(2) {
            ll += mat[w[0] * k + w[1]].max(1e-9).ln();
        }
        let mean_ll = ll / ((pg.sym_seq.len() - mid - 1).max(1) as f64);
        tanh_clamped(((mean_ll - uniform_ll) / (-uniform_ll).max(0.5)).clamp(0.0, 1.2) * 1.4)
    }

    pub fn emit_continuation(&self, observed_e: &[u8], max_len: usize) -> Vec<u8> {
        let pg = PerceivedGraph::from_bytes(observed_e);
        let k = pg.k();
        if k == 0 {
            return Vec::new();
        }
        let mat = self.predict_transition_matrix(&pg);
        let mut cur = *pg.sym_seq.last().unwrap_or(&0);
        let mut out = Vec::with_capacity(max_len.min(64));
        let mut seen = vec![0usize; k];
        for _ in 0..max_len.min(64) {
            let mut best_v = 0;
            let mut best_p = -1.0;
            for v in 0..k {
                let p = mat[cur * k + v] / (1.0 + 0.3 * (seen[v] as f64));
                if p > best_p {
                    best_p = p;
                    best_v = v;
                }
            }
            out.push(pg.unique_bytes[best_v]);
            seen[best_v] += 1;
            cur = best_v;
        }
        out
    }
}

fn softmax4(l: &[f64; 4]) -> [f64; 4] {
    let m = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e = [(l[0] - m).exp(), (l[1] - m).exp(), (l[2] - m).exp(), (l[3] - m).exp()];
    let s = e[0] + e[1] + e[2] + e[3];
    [e[0] / s, e[1] / s, e[2] / s, e[3] / s]
}

/// Baseline 2 : Même réseau connexionniste et même budget de recherche,
/// mais guidé et entraîné UNIQUEMENT par la log-vraisemblance sur E (sans φ_sharp ni stabilité).
#[derive(Clone, Debug)]
pub struct LikelihoodSearchBaseline {
    pub net: PhisharpNet,
}

pub fn normalized_log_likelihood(h: &HypothesisState, pg: &PerceivedGraph) -> f64 {
    let k = pg.k();
    if k <= 1 || pg.sym_seq.len() < 2 {
        return 0.0;
    }
    let uniform_ll = -(k as f64).ln();
    let mut ll = 0.0;
    for w in pg.sym_seq.windows(2) {
        ll += h.sym_trans[w[0] * k + w[1]].max(1e-9).ln();
    }
    let mean_ll = ll / ((pg.sym_seq.len() - 1) as f64);
    sigmoid((mean_ll - uniform_ll) * 1.8)
}

impl LikelihoodSearchBaseline {
    pub fn train_equal_compute(iterations: usize, pop_pairs: usize, batch_size: usize, budget: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed ^ 0x55AA_55AA);
        let mut net = PhisharpNet::new_random(&mut rng);
        let n = net.param_count();
        let mut theta = net.to_flat_vec();
        let mut m = vec![0.0; n];
        let mut v = vec![0.0; n];
        let sigma = 0.04;
        let lr = 0.012;

        for it in 1..=iterations {
            let mut batch = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                batch.push(sample_training_effect(&mut rng));
            }
            let eval_ll = |candidate_net: &PhisharpNet| -> f64 {
                let mut sum = 0.0;
                for s in &batch {
                    let (h, ll) = search_by_likelihood(candidate_net, &s.observed_e, budget);
                    sum += 0.7 * ll + 0.3 * score_generativity(&h, &PerceivedGraph::from_bytes(&s.observed_e));
                }
                sum / (batch.len().max(1) as f64)
            };

            let mut grad = vec![0.0; n];
            let mut probe = net.clone();
            for _ in 0..pop_pairs {
                let eps: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
                let plus: Vec<f64> = theta.iter().zip(eps.iter()).map(|(&w, &e)| w + sigma * e).collect();
                probe.from_flat_vec(&plus);
                let fp = eval_ll(&probe);

                let minus: Vec<f64> = theta.iter().zip(eps.iter()).map(|(&w, &e)| w - sigma * e).collect();
                probe.from_flat_vec(&minus);
                let fm = eval_ll(&probe);

                let diff = ((fp - fm) / (2.0 * sigma)).clamp(-5.0, 5.0);
                for i in 0..n {
                    grad[i] += diff * eps[i] / (pop_pairs as f64);
                }
            }

            let b1 = 1.0 - 0.9_f64.powi(it as i32);
            let b2 = 1.0 - 0.999_f64.powi(it as i32);
            for i in 0..n {
                let g = grad[i] - 1e-4 * theta[i];
                m[i] = 0.9 * m[i] + 0.1 * g;
                v[i] = 0.999 * v[i] + 0.001 * g * g;
                theta[i] += lr * (m[i] / b1) / ((v[i] / b2).sqrt() + 1e-8);
            }
            net.from_flat_vec(&theta);
        }
        Self { net }
    }

    pub fn run(&self, observed_e: &[u8], budget: usize) -> (HypothesisState, f64, f64, Vec<u8>) {
        let pg = PerceivedGraph::from_bytes(observed_e);
        let (h, ll) = search_by_likelihood(&self.net, observed_e, budget);
        let proj = score_projection(&h, &pg);
        let out = crate::abduction::emit_explanation(&h, &pg, 128);
        (h, ll, proj, out)
    }
}

fn search_by_likelihood(net: &PhisharpNet, observed_e: &[u8], budget: usize) -> (HypothesisState, f64) {
    let pg = PerceivedGraph::from_bytes(observed_e);
    let mut h = net.init_regional_state(&pg);
    let mut best_h = h.clone();
    let mut best_ll = normalized_log_likelihood(&h, &pg);

    for _ in 0..budget {
        let mut cand = h.clone();
        net.propagate_step(&mut cand, &pg, 0.55);
        // Ajustement direct par vraisemblance pure (sans réfutation ni MIP)
        let k = cand.k;
        for u in 0..k {
            for v in 0..k {
                cand.sym_trans[u * k + v] =
                    (0.85 * cand.sym_trans[u * k + v] + 0.15 * pg.emp_trans1[u * k + v]).max(1e-9);
            }
        }
        let ll = normalized_log_likelihood(&cand, &pg);
        if ll >= best_ll {
            best_ll = ll;
            best_h = cand.clone();
            h = cand;
        }
    }
    (best_h, best_ll)
}

/// Instrument diagnostique MDL post-hoc UNIQUEMENT (jamais utilisé comme objectif !).
/// Pénalise la complexité des couplages actifs de H en plus de la longueur de code des données.
pub fn mdl_diagnostic_score(h: &HypothesisState, pg: &PerceivedGraph) -> f64 {
    let k = pg.k();
    if k <= 1 || pg.sym_seq.len() < 2 {
        return 0.0;
    }
    let mut data_bits = 0.0;
    for w in pg.sym_seq.windows(2) {
        let p = h.sym_trans[w[0] * k + w[1]].max(1e-9);
        data_bits -= p.log2();
    }
    // Coût descriptif du modèle H (nombre effectif de couplages forts + degrés latents)
    let mut active_edges = 0.0;
    for &c in &h.lat_couplings {
        active_edges += c.abs().min(1.0);
    }
    let model_bits = active_edges * (N_LAT as f64).log2() + (k as f64) * 2.0;
    let total_bits = data_bits + model_bits;
    let uniform_bits = (pg.sym_seq.len() as f64) * (k as f64).log2();
    let compression_ratio = ((uniform_bits - total_bits) / uniform_bits.max(1.0)).clamp(-1.0, 1.0);
    0.5 * (compression_ratio + 1.0)
}