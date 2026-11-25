use crate::graph::{NUM_PARAMS, PhiGraph, STATE_DIM, extract_relations};
use crate::phi::{compute_phi_sharp, cosine_sim};
use crate::ptgrs::{CauseFamily, sample_episode, sample_episode_in_range};
use crate::rng::Rng;
use std::thread;
use std::time::Instant;

pub struct TrainConfig {
    pub steps: usize,
    pub pop_pairs: usize,
    pub sigma: f32,
    pub lr: f32,
    pub seed: u64,
}

/// Pénalité anti-effondrement en cône (sans jamais utiliser les étiquettes cachées).
fn batch_collapse_penalty(states: &[[f32; STATE_DIM]]) -> f32 {
    let mut sum = 0.0f32;
    let mut pairs = 0usize;
    for i in 0..states.len() {
        for j in (i + 1)..states.len() {
            let c = cosine_sim(&states[i], &states[j]).max(0.0);
            sum += c * c;
            pairs += 1;
        }
    }
    if pairs == 0 {
        0.0
    } else {
        sum / (pairs as f32)
    }
}

fn eval_batch(graph: &PhiGraph, batch: &[Vec<u8>], eval_seed: u64) -> f32 {
    let mut rng = Rng::new(eval_seed);
    let mut sum = 0.0f32;
    let mut states = Vec::with_capacity(batch.len());
    for effect in batch {
        let rep = compute_phi_sharp(graph, effect, &mut rng, false);
        sum += rep.phi_sharp;
        states.push(rep.trace.final_state);
    }
    let mean_phi = sum / (batch.len() as f32);
    let collapse = batch_collapse_penalty(&states);
    mean_phi - 0.22 * collapse
}

pub fn train_phisharp(cfg: &TrainConfig) -> PhiGraph {
    let start = Instant::now();
    let mut rng = Rng::new(cfg.seed);
    let mut theta = PhiGraph::random(&mut rng).weights;

    let mut m = vec![0.0f32; NUM_PARAMS];
    let mut v = vec![0.0f32; NUM_PARAMS];
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps_adam = 1e-8f32;

    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(cfg.pop_pairs.max(1));

    println!(
        "Entraînement Phisharp v0.2 ({} paramètres, {} étapes, {} paires ES, {} threads)...",
        NUM_PARAMS, cfg.steps, cfg.pop_pairs, n_threads
    );

    for step in 1..=cfg.steps {
        let batch: Vec<Vec<u8>> = CauseFamily::ALL
            .iter()
            .map(|&fam| sample_episode(&mut rng, fam).effect)
            .collect();
        let eval_seed = rng.next_u64();
        let pair_seeds: Vec<u64> = (0..cfg.pop_pairs).map(|_| rng.next_u64()).collect();

        let results: Vec<(u64, f32, f32)> = thread::scope(|s| {
            let chunk_size = cfg.pop_pairs.div_ceil(n_threads);
            let mut handles = Vec::new();
            for chunk in pair_seeds.chunks(chunk_size) {
                let theta_ref = &theta;
                let batch_ref = &batch;
                let sigma = cfg.sigma;
                handles.push(s.spawn(move || {
                    let mut local = Vec::with_capacity(chunk.len());
                    for &pseed in chunk {
                        let mut prng = Rng::new(pseed);
                        let mut w_pos = Vec::with_capacity(NUM_PARAMS);
                        let mut w_neg = Vec::with_capacity(NUM_PARAMS);
                        for &w in theta_ref {
                            let noise = prng.next_normal() * sigma;
                            w_pos.push(w + noise);
                            w_neg.push(w - noise);
                        }
                        let f_pos = eval_batch(&PhiGraph::from_weights(w_pos), batch_ref, eval_seed);
                        let f_neg = eval_batch(&PhiGraph::from_weights(w_neg), batch_ref, eval_seed);
                        local.push((pseed, f_pos, f_neg));
                    }
                    local
                }));
            }
            let mut merged = Vec::with_capacity(cfg.pop_pairs);
            for h in handles {
                merged.extend(h.join().unwrap());
            }
            merged
        });

        let diffs: Vec<f32> = results.iter().map(|&(_, fp, fn_)| fp - fn_).collect();
        let var: f32 = diffs.iter().map(|d| d * d).sum::<f32>() / (diffs.len() as f32);
        let std = var.sqrt().max(1e-6);

        let mut grad = vec![0.0f32; NUM_PARAMS];
        let scale = 1.0 / (2.0 * (cfg.pop_pairs as f32) * cfg.sigma * std);
        for (&(pseed, _, _), &diff) in results.iter().zip(diffs.iter()) {
            let mut prng = Rng::new(pseed);
            let coeff = diff * scale;
            for g in &mut grad {
                *g += coeff * prng.next_normal();
            }
        }

        let t = step as i32;
        let b1_corr = 1.0 - beta1.powi(t);
        let b2_corr = 1.0 - beta2.powi(t);
        for i in 0..NUM_PARAMS {
            let g = grad[i] - 1e-4 * theta[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            let m_hat = m[i] / b1_corr;
            let v_hat = v[i] / b2_corr;
            theta[i] += cfg.lr * m_hat / (v_hat.sqrt() + eps_adam);
        }

        if step == 1 || step % 25 == 0 || step == cfg.steps {
            let model = PhiGraph::from_weights(theta.clone());
            let mut diag_rng = Rng::new(eval_seed);
            let mut phi = 0.0;
            let mut gen_sum = 0.0;
            let mut disc = 0.0;
            let mut proj = 0.0;
            let mut integ = 0.0;
            let mut res = 0.0;
            let mut bounce = 0.0;
            let mut states = Vec::with_capacity(batch.len());
            for eff in &batch {
                let r = compute_phi_sharp(&model, eff, &mut diag_rng, false);
                phi += r.phi_sharp;
                gen_sum += r.generativity;
                disc += r.discrimination;
                proj += r.projection;
                integ += r.integration;
                res += r.residual;
                bounce += r.bounce_pen;
                states.push(r.trace.final_state);
            }
            let n = batch.len() as f32;
            let coll = batch_collapse_penalty(&states);
            println!(
                "step {:>4}/{:<4} | φ={:+.4} | Gen={:.3} Disc={:.3} Proj={:.3} IIT={:.3} | res={:.3} bnc={:.3} coll={:.3} ({:.1}s)",
                step,
                cfg.steps,
                phi / n,
                gen_sum / n,
                disc / n,
                proj / n,
                integ / n,
                res / n,
                bounce / n,
                coll,
                start.elapsed().as_secs_f32()
            );
        }
    }

    let final_model = PhiGraph::from_weights(theta);
    evaluate_meta_relational_abstraction(&final_model, cfg.seed ^ 0xDEAD_BEEF);
    final_model
}

fn evaluate_meta_relational_abstraction(model: &PhiGraph, seed: u64) {
    let mut rng = Rng::new(seed);
    let mut same_sum = 0.0f32;
    let mut diff_sum = 0.0f32;
    let trials = 48;

    for _ in 0..trials {
        let f1 = CauseFamily::ALL[rng.next_usize(CauseFamily::ALL.len())];
        let mut f2 = CauseFamily::ALL[rng.next_usize(CauseFamily::ALL.len())];
        while f2 == f1 {
            f2 = CauseFamily::ALL[rng.next_usize(CauseFamily::ALL.len())];
        }

        let e_a = sample_episode_in_range(&mut rng, f1, 32, 75);
        let e_b = sample_episode_in_range(&mut rng, f1, 140, 220);
        let e_c = sample_episode_in_range(&mut rng, f2, 140, 220);

        let h_a = model.abduce(&extract_relations(&e_a.effect), None).final_state;
        let h_b = model.abduce(&extract_relations(&e_b.effect), None).final_state;
        let h_c = model.abduce(&extract_relations(&e_c.effect), None).final_state;

        same_sum += cosine_sim(&h_a, &h_b);
        diff_sum += cosine_sim(&h_a, &h_c);
    }

    let same_avg = same_sum / (trials as f32);
    let diff_avg = diff_sum / (trials as f32);
    println!("\n--- Évaluation post-hoc (Abstraction Méta-relationnelle, alphabets disjoints) ---");
    println!("  Même cause cachée (0 octet commun) : cos = {:+.4}", same_avg);
    println!("  Causes cachées différentes         : cos = {:+.4}", diff_avg);
    println!("  Marge d'abstraction                : {:+.4}", same_avg - diff_avg);
}
