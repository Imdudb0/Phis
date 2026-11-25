use crate::graph::{NUM_PARAMS, PhiGraph, STATE_DIM};
use crate::phi::{
    PreparedComposite, PreparedEpisode, compute_composite_phi_prepared,
    compute_phi_sharp_prepared, cosine_sim, project_query_to_molds, segment_effect,
};
use crate::ptgrs::{
    CauseFamily, sample_composite_episode, sample_episode, sample_episode_in_range,
};
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

fn eval_batch(
    graph: &PhiGraph,
    singles: &[PreparedEpisode],
    composites: &[PreparedComposite],
) -> f32 {
    let mut sum_phi = 0.0f32;
    let mut states = Vec::with_capacity(singles.len());
    for ep in singles {
        let rep = compute_phi_sharp_prepared(graph, ep, false);
        sum_phi += rep.phi_sharp;
        states.push(rep.trace.final_state);
    }
    let mean_phi = sum_phi / (singles.len() as f32);
    let collapse = batch_collapse_penalty(&states);

    let mut comp_sum = 0.0f32;
    for comp in composites {
        comp_sum += compute_composite_phi_prepared(graph, comp).score;
    }
    let mean_comp = if composites.is_empty() {
        0.0
    } else {
        comp_sum / (composites.len() as f32)
    };

    mean_phi - 0.22 * collapse + 0.50 * mean_comp
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
        "Entraînement Phisharp v0.4 ({} paramètres dont 140 capteur relationnel, {} étapes, {} paires ES, {} threads)...",
        NUM_PARAMS, cfg.steps, cfg.pop_pairs, n_threads
    );

    for step in 1..=cfg.steps {
        let mut prep_rng = Rng::new(rng.next_u64());
        let singles: Vec<PreparedEpisode> = CauseFamily::ALL
            .iter()
            .map(|&fam| {
                let eff = sample_episode(&mut rng, fam).effect;
                PreparedEpisode::from_bytes(&eff, &mut prep_rng)
            })
            .collect();

        let composites = vec![
            PreparedComposite::from_bytes(
                &sample_composite_episode(&mut rng, 3, false, false, false).effect,
            ),
            PreparedComposite::from_bytes(
                &sample_composite_episode(&mut rng, 3, true, false, false).effect,
            ),
        ];

        let pair_seeds: Vec<u64> = (0..cfg.pop_pairs).map(|_| rng.next_u64()).collect();

        let results: Vec<(u64, f32, f32)> = thread::scope(|s| {
            let chunk_size = cfg.pop_pairs.div_ceil(n_threads);
            let mut handles = Vec::new();
            for chunk in pair_seeds.chunks(chunk_size) {
                let theta_ref = &theta;
                let singles_ref = &singles;
                let comp_ref = &composites;
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
                        let f_pos = eval_batch(
                            &PhiGraph::from_weights(w_pos),
                            singles_ref,
                            comp_ref,
                        );
                        let f_neg = eval_batch(
                            &PhiGraph::from_weights(w_neg),
                            singles_ref,
                            comp_ref,
                        );
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
            let mut phi = 0.0;
            let mut gen_sum = 0.0;
            let mut disc = 0.0;
            let mut proj = 0.0;
            let mut integ = 0.0;
            let mut res = 0.0;
            let mut states = Vec::with_capacity(singles.len());
            for ep in &singles {
                let r = compute_phi_sharp_prepared(&model, ep, false);
                phi += r.phi_sharp;
                gen_sum += r.generativity;
                disc += r.discrimination;
                proj += r.projection;
                integ += r.integration;
                res += r.residual;
                states.push(r.trace.final_state);
            }
            let n = singles.len() as f32;
            let coll = batch_collapse_penalty(&states);
            let sel: f32 = composites
                .iter()
                .map(|c| compute_composite_phi_prepared(&model, c).selectivity)
                .sum::<f32>()
                / (composites.len() as f32);
            println!(
                "step {:>4}/{:<4} | φ={:+.4} | Gen={:.3} Disc={:.3} Proj={:.3} IIT={:.3} RefSel={:.3} | res={:.3} coll={:.3} ({:.1}s)",
                step,
                cfg.steps,
                phi / n,
                gen_sum / n,
                disc / n,
                proj / n,
                integ / n,
                sel,
                res / n,
                coll,
                start.elapsed().as_secs_f32()
            );
        }
    }

    let final_model = PhiGraph::from_weights(theta);
    evaluate_post_hoc(&final_model, cfg.seed ^ 0xDEAD_BEEF);
    final_model
}

fn evaluate_post_hoc(model: &PhiGraph, seed: u64) {
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
        let e_a = sample_episode_in_range(&mut rng, f1, 33, 90);
        let e_b = sample_episode_in_range(&mut rng, f1, 140, 220);
        let e_c = sample_episode_in_range(&mut rng, f2, 140, 220);

        let h_a = model.abduce(&model.encode_bytes(&e_a.effect), None).final_state;
        let h_b = model.abduce(&model.encode_bytes(&e_b.effect), None).final_state;
        let h_c = model.abduce(&model.encode_bytes(&e_c.effect), None).final_state;

        same_sum += cosine_sim(&h_a, &h_b);
        diff_sum += cosine_sim(&h_a, &h_c);
    }
    let same_avg = same_sum / (trials as f32);
    let diff_avg = diff_sum / (trials as f32);

    let proj_trials = 50;
    let mut hits_raw = 0usize;
    let mut hits_tagged = 0usize;
    let mut avg_margin = 0.0f32;

    for _ in 0..proj_trials {
        let ep_raw = sample_composite_episode(&mut rng, 4, false, false, true);
        let spans_raw = segment_effect(&ep_raw.effect);
        let m_raw =
            project_query_to_molds(model, &ep_raw.effect, &spans_raw, ep_raw.hidden_query_idx);
        if m_raw.first().map(|m| m.dst_idx) == Some(ep_raw.hidden_match_idx) {
            hits_raw += 1;
        }

        let ep_tag = sample_composite_episode(&mut rng, 4, true, true, true);
        let spans_tag = segment_effect(&ep_tag.effect);
        let m_tag =
            project_query_to_molds(model, &ep_tag.effect, &spans_tag, ep_tag.hidden_query_idx);
        if m_tag.first().map(|m| m.dst_idx) == Some(ep_tag.hidden_match_idx) {
            hits_tagged += 1;
        }
        if m_tag.len() >= 2 {
            avg_margin += m_tag[0].score - m_tag[1].score;
        }
    }

    println!("\n--- Évaluation post-hoc 1 : Abstraction Méta-relationnelle (0 octet commun) ---");
    println!("  Même cause cachée : cos = {:+.4}", same_avg);
    println!("  Causes différentes: cos = {:+.4}", diff_avg);
    println!("  Marge             : {:+.4}", same_avg - diff_avg);

    println!("\n--- Évaluation post-hoc 2 : Projection Référentielle (4 moules, hasard = 25%) ---");
    println!(
        "  Moules bruts (0 octet commun)      : {}/{} ({:.1}%)",
        hits_raw,
        proj_trials,
        100.0 * (hits_raw as f32) / (proj_trials as f32)
    );
    println!(
        "  Moules étiquetés (0 octet commun)  : {}/{} ({:.1}%)  [marge Top1-Top2 = {:+.4}]",
        hits_tagged,
        proj_trials,
        100.0 * (hits_tagged as f32) / (proj_trials as f32),
        avg_margin / (proj_trials as f32)
    );
}
