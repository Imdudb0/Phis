//! Entraînement FERMÉ de Phisharp v0.2.0 : PTGRS (E seul) + φ_sharp + Sens + Résidu.

use std::path::Path;

use crate::abduction::{SearchConfig, run_abductive_search};
use crate::math::Rng;
use crate::ptgrs::{PtgrsSample, sample_training_effect};
use crate::substrate::PhisharpNet;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub iterations: usize,
    pub pop_pairs: usize,
    pub batch_size: usize,
    pub search_budget: usize,
    pub sigma: f64,
    pub lr: f64,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            iterations: 80,
            pop_pairs: 16,
            batch_size: 4,
            search_budget: 8,
            sigma: 0.045,
            lr: 0.015,
            seed: 2026_11_24,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainLogEntry {
    pub iter: usize,
    pub mean_phi_sharp: f64,
    pub mean_phi_t0: f64,
    pub mean_sens_min: f64,
    pub mean_residue: f64,
    pub non_reciprocity: f64,
}

fn evaluate_batch_fitness(
    net: &PhisharpNet,
    batch: &[PtgrsSample],
    budget: usize,
) -> (f64, f64, f64, f64, f64) {
    let cfg = SearchConfig {
        budget_steps: budget,
        disable_refutation: false,
        max_explain_len: 96,
    };

    let mut sum_obj = 0.0;
    let mut sum_phi = 0.0;
    let mut sum_phi0 = 0.0;
    let mut sum_sens = 0.0;
    let mut sum_res = 0.0;

    for sample in batch {
        let res = run_abductive_search(net, &sample.observed_e, cfg);
        let phi0 = res.trajectory.first().map(|p| p.report.phi_sharp).unwrap_or(0.0);
        let rep = &res.final_report;

        let traj_gain = (rep.phi_sharp - phi0).max(0.0);
        let obj = rep.phi_sharp + 0.50 * traj_gain + 0.35 * rep.sens_min - 0.25 * rep.residue_score;

        sum_obj += obj;
        sum_phi += rep.phi_sharp;
        sum_phi0 += phi0;
        sum_sens += rep.sens_min;
        sum_res += rep.residue_score;
    }

    let inv = 1.0 / (batch.len().max(1) as f64);
    (
        sum_obj * inv,
        sum_phi * inv,
        sum_phi0 * inv,
        sum_sens * inv,
        sum_res * inv,
    )
}

pub fn train_closed_ptgrs(cfg: &TrainConfig, save_path: Option<&Path>) -> (PhisharpNet, Vec<TrainLogEntry>) {
    let mut rng = Rng::new(cfg.seed);
    let mut net = PhisharpNet::new_random(&mut rng);
    let n_params = net.param_count();
    let mut theta = net.to_flat_vec();

    let mut m = vec![0.0; n_params];
    let mut v = vec![0.0; n_params];
    let beta1 = 0.9_f64;
    let beta2 = 0.999_f64;
    let eps_adam = 1e-8_f64;

    let mut logs = Vec::with_capacity(cfg.iterations);
    let mut best_theta = theta.clone();
    let mut best_obj = f64::NEG_INFINITY;

    for it in 1..=cfg.iterations {
        let mut batch = Vec::with_capacity(cfg.batch_size);
        for _ in 0..cfg.batch_size {
            batch.push(sample_training_effect(&mut rng));
        }

        let mut grad = vec![0.0; n_params];
        let mut probe_net = net.clone();
        let mut noises = Vec::with_capacity(cfg.pop_pairs);
        let mut diffs = Vec::with_capacity(cfg.pop_pairs);

        for _ in 0..cfg.pop_pairs {
            let eps: Vec<f64> = (0..n_params).map(|_| rng.next_normal()).collect();

            let plus: Vec<f64> = theta.iter().zip(eps.iter()).map(|(&w, &e)| w + cfg.sigma * e).collect();
            probe_net.from_flat_vec(&plus);
            let (f_plus, _, _, _, _) = evaluate_batch_fitness(&probe_net, &batch, cfg.search_budget);

            let minus: Vec<f64> = theta.iter().zip(eps.iter()).map(|(&w, &e)| w - cfg.sigma * e).collect();
            probe_net.from_flat_vec(&minus);
            let (f_minus, _, _, _, _) = evaluate_batch_fitness(&probe_net, &batch, cfg.search_budget);

            diffs.push(f_plus - f_minus);
            noises.push(eps);
        }

        let mean_abs = diffs.iter().map(|d| d.abs()).sum::<f64>() / (diffs.len().max(1) as f64);
        let norm_scale = if mean_abs > 1e-8 { 1.0 / mean_abs } else { 1.0 };

        let denom = 2.0 * (cfg.pop_pairs as f64) * cfg.sigma;
        for (p_idx, eps) in noises.iter().enumerate() {
            let w_diff = (diffs[p_idx] * norm_scale).clamp(-3.0, 3.0);
            for i in 0..n_params {
                grad[i] += w_diff * eps[i] / denom;
            }
        }

        let b1_corr = 1.0 - beta1.powi(it as i32);
        let b2_corr = 1.0 - beta2.powi(it as i32);
        for i in 0..n_params {
            let g = grad[i] - 1e-4 * theta[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            let m_hat = m[i] / b1_corr;
            let v_hat = v[i] / b2_corr;
            theta[i] += cfg.lr * m_hat / (v_hat.sqrt() + eps_adam);
        }

        net.from_flat_vec(&theta);
        let (obj_cur, mean_phi, mean_phi0, mean_sens, mean_res) =
            evaluate_batch_fitness(&net, &batch, cfg.search_budget);

        if obj_cur > best_obj {
            best_obj = obj_cur;
            best_theta = theta.clone();
        }

        let entry = TrainLogEntry {
            iter: it,
            mean_phi_sharp: mean_phi,
            mean_phi_t0: mean_phi0,
            mean_sens_min: mean_sens,
            mean_residue: mean_res,
            non_reciprocity: net.non_reciprocity_index(),
        };

        if it == 1 || it % 10 == 0 || it == cfg.iterations {
            eprintln!(
                "[train {}/{}] phi_sharp={:.4} (t0={:.4}) sens_min={:.4} res={:.4} non_recip={:.3}",
                entry.iter,
                cfg.iterations,
                entry.mean_phi_sharp,
                entry.mean_phi_t0,
                entry.mean_sens_min,
                entry.mean_residue,
                entry.non_reciprocity
            );
        }
        logs.push(entry);
    }

    net.from_flat_vec(&best_theta);
    if let Some(p) = save_path {
        if let Err(e) = net.save_to_file(p) {
            eprintln!("Erreur sauvegarde modèle {:?}: {}", p, e);
        }
    }
    (net, logs)
}