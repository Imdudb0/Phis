use crate::graph::{NUM_PARAMS, OFF_REF_START, PhiGraph, STATE_DIM};
use crate::multihop::{PreparedRefScene, analyze_multihop, eval_ref_scene_phi, prepare_ref_scene};
use crate::phi::{
    PreparedComposite, PreparedEpisode, compute_composite_phi_prepared,
    compute_phi_sharp_prepared, cosine_sim, discover_causal_regimes, project_query_to_regimes,
};
use crate::ptgrs::{
    CauseFamily, SepMode, sample_composite_episode, sample_episode, sample_episode_in_range,
    sample_multihop_episode,
};
use crate::rng::Rng;
use std::thread;
use std::time::Instant;

const REF_PARAMS: usize = NUM_PARAMS - OFF_REF_START;

pub struct TrainConfig {
    pub steps: usize,
    pub pop_pairs: usize,
    pub sigma: f32,
    pub lr: f32,
    pub seed: u64,
}

fn span_iou(a: (usize, usize), b: (usize, usize)) -> f32 {
    let inter_s = a.0.max(b.0);
    let inter_e = a.1.min(b.1);
    if inter_e <= inter_s {
        return 0.0;
    }
    let inter = (inter_e - inter_s) as f32;
    let union = ((a.1 - a.0) + (b.1 - b.0)) as f32 - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn concept_aligned(pred: &[u8], truth: &[u8]) -> bool {
    if pred == truth {
        return true;
    }
    if pred.len() >= 3 && truth.windows(pred.len()).any(|w| w == pred) {
        return true;
    }
    if truth.len() >= 3
        && pred.len() <= truth.len() + 3
        && pred.windows(truth.len()).any(|w| w == truth)
    {
        return true;
    }
    false
}

fn batch_collapse_penalty(states: &[[f32; STATE_DIM]]) -> f32 {
    let mut sum = 0.0f32;
    let mut max_sq = 0.0f32;
    let mut pairs = 0usize;
    for i in 0..states.len() {
        for j in (i + 1)..states.len() {
            let c = cosine_sim(&states[i], &states[j]).max(0.0);
            let sq = c * c;
            sum += sq;
            if sq > max_sq {
                max_sq = sq;
            }
            pairs += 1;
        }
    }
    if pairs == 0 {
        0.0
    } else {
        0.5 * (sum / (pairs as f32)) + 0.5 * max_sq
    }
}

fn cross_specificity(
    graph: &PhiGraph,
    singles: &[PreparedEpisode],
    states: &[[f32; STATE_DIM]],
    residuals: &[f32],
) -> f32 {
    let n = singles.len();
    if n < 2 {
        return 0.0;
    }
    let rels_list: Vec<_> = singles
        .iter()
        .map(|ep| graph.encode_relations(&ep.orig))
        .collect();
    let mut spec_sum = 0.0f32;
    for i in 0..n {
        let mut min_other = f32::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let (_, _, err_ij) = graph.evaluate_hypothesis(&states[i], &rels_list[j]);
            if err_ij < min_other {
                min_other = err_ij;
            }
        }
        spec_sum += ((min_other - residuals[i]) / (min_other + 0.05)).clamp(0.0, 1.0);
    }
    spec_sum / (n as f32)
}

fn eval_batch(
    graph: &PhiGraph,
    singles: &[PreparedEpisode],
    composites: &[PreparedComposite],
    comp_w: f32,
) -> f32 {
    let mut sum_phi = 0.0f32;
    let mut sum_gate_c = 0.0f32;
    let mut min_gate_c = 1.0f32;
    let mut states = Vec::with_capacity(singles.len());
    let mut residuals = Vec::with_capacity(singles.len());
    for ep in singles {
        let rep = compute_phi_sharp_prepared(graph, ep, false);
        sum_phi += rep.phi_sharp;
        sum_gate_c += rep.gate_contrast;
        if rep.gate_contrast < min_gate_c {
            min_gate_c = rep.gate_contrast;
        }
        states.push(rep.trace.final_state);
        residuals.push(rep.residual);
    }
    let n = singles.len() as f32;
    let mean_phi = sum_phi / n;
    let mean_gate_c = sum_gate_c / n;
    let collapse = batch_collapse_penalty(&states);
    let spec = cross_specificity(graph, singles, &states, &residuals);

    let mut comp_sum = 0.0f32;
    for comp in composites {
        comp_sum += compute_composite_phi_prepared(graph, comp).score;
    }
    let mean_comp = if composites.is_empty() {
        0.0
    } else {
        comp_sum / (composites.len() as f32)
    };

    mean_phi
        + 0.55 * spec
        + 0.20 * mean_gate_c
        + 0.30 * min_gate_c
        - 0.40 * collapse
        + comp_w * mean_comp
}

fn eval_ref_batch(graph: &PhiGraph, scenes: &[PreparedRefScene]) -> f32 {
    if scenes.is_empty() {
        return 0.0;
    }
    let sum: f32 = scenes
        .iter()
        .map(|sc| eval_ref_scene_phi(graph, sc))
        .sum();
    sum / (scenes.len() as f32)
}

pub fn train_phisharp(cfg: &TrainConfig) -> PhiGraph {
    let start = Instant::now();
    let mut rng = Rng::new(cfg.seed);
    let mut theta = PhiGraph::random(&mut rng).weights;

    let mut m = vec![0.0f32; OFF_REF_START];
    let mut v = vec![0.0f32; OFF_REF_START];
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps_adam = 1e-8f32;

    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(cfg.pop_pairs.max(1));

    println!(
        "Entraînement Phisharp v0.7 ({} paramètres : {} Hop-1 + {} Champ Réf., {} étapes, {} threads)...",
        NUM_PARAMS, OFF_REF_START, REF_PARAMS, cfg.steps, n_threads
    );

    // --- Phase A : Capteur relationnel + Porte causale + Boucle abductive (1541 poids) ---
    for step in 1..=cfg.steps {
        let comp_w = 0.10 + 0.45 * ((step as f32) / 120.0).min(1.0);
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
                &sample_composite_episode(&mut rng, 3, false, false, false, SepMode::None).effect,
            ),
            PreparedComposite::from_bytes(
                &sample_composite_episode(&mut rng, 3, true, false, false, SepMode::Random).effect,
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
                        let mut w_pos = theta_ref.clone();
                        let mut w_neg = theta_ref.clone();
                        for i in 0..OFF_REF_START {
                            let noise = prng.next_normal() * sigma;
                            w_pos[i] += noise;
                            w_neg[i] -= noise;
                        }
                        let f_pos = eval_batch(
                            &PhiGraph::from_weights(w_pos),
                            singles_ref,
                            comp_ref,
                            comp_w,
                        );
                        let f_neg = eval_batch(
                            &PhiGraph::from_weights(w_neg),
                            singles_ref,
                            comp_ref,
                            comp_w,
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

        let mut grad = vec![0.0f32; OFF_REF_START];
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
        for i in 0..OFF_REF_START {
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
            let mut integ = 0.0;
            let mut gate_c = 0.0;
            let mut min_gate_c = 1.0f32;
            let mut res = 0.0;
            let mut states = Vec::with_capacity(singles.len());
            let mut residuals = Vec::with_capacity(singles.len());
            for ep in &singles {
                let r = compute_phi_sharp_prepared(&model, ep, false);
                phi += r.phi_sharp;
                gen_sum += r.generativity;
                disc += r.discrimination;
                integ += r.integration;
                gate_c += r.gate_contrast;
                if r.gate_contrast < min_gate_c {
                    min_gate_c = r.gate_contrast;
                }
                res += r.residual;
                states.push(r.trace.final_state);
                residuals.push(r.residual);
            }
            let n = singles.len() as f32;
            let coll = batch_collapse_penalty(&states);
            let spec = cross_specificity(&model, &singles, &states, &residuals);
            let sel: f32 = composites
                .iter()
                .map(|c| compute_composite_phi_prepared(&model, c).selectivity)
                .sum::<f32>()
                / (composites.len() as f32);
            println!(
                "step {:>4}/{:<4} | φ={:+.4} | Gen={:.3} Disc={:.3} Spec={:.3} Gate={:.2}/{:.2} IIT={:.3} RefSel={:.3} | res={:.3} coll={:.3} ({:.1}s)",
                step,
                cfg.steps,
                phi / n,
                gen_sum / n,
                disc / n,
                spec,
                gate_c / n,
                min_gate_c,
                integ / n,
                sel,
                res / n,
                coll,
                start.elapsed().as_secs_f32()
            );
        }
    }

    // --- Phase B : Champ Référentiel Connexionniste (97 poids, ES non-supervisé dans la même séance) ---
    let ref_steps = (cfg.steps / 2).clamp(20, 150);
    let ref_sigma = 0.14f32;
    let ref_lr = 0.055f32;
    let mut rm = vec![0.0f32; REF_PARAMS];
    let mut rv = vec![0.0f32; REF_PARAMS];
    let base_model = PhiGraph::from_weights(theta.clone());

    println!(
        "\nPhase B — Champ Référentiel Connexionniste ({} poids, {} étapes ES non-supervisées)...",
        REF_PARAMS, ref_steps
    );

    for rstep in 1..=ref_steps {
        let mut scenes: Vec<PreparedRefScene> = Vec::with_capacity(3);
        let mut tries = 0usize;
        while scenes.len() < 3 && tries < 10 {
            let flat = scenes.len() == 2;
            let ep = sample_multihop_episode(&mut rng, flat, false);
            if let Some(sc) = prepare_ref_scene(&base_model, &ep.effect) {
                scenes.push(sc);
            }
            tries += 1;
        }
        if scenes.is_empty() {
            continue;
        }

        let pair_seeds: Vec<u64> = (0..cfg.pop_pairs).map(|_| rng.next_u64()).collect();
        let results: Vec<(u64, f32, f32)> = thread::scope(|s| {
            let chunk_size = cfg.pop_pairs.div_ceil(n_threads);
            let mut handles = Vec::new();
            for chunk in pair_seeds.chunks(chunk_size) {
                let theta_ref = &theta;
                let scenes_ref = &scenes;
                handles.push(s.spawn(move || {
                    let mut local = Vec::with_capacity(chunk.len());
                    for &pseed in chunk {
                        let mut prng = Rng::new(pseed);
                        let mut w_pos = theta_ref.clone();
                        let mut w_neg = theta_ref.clone();
                        for i in 0..REF_PARAMS {
                            let noise = prng.next_normal() * ref_sigma;
                            w_pos[OFF_REF_START + i] += noise;
                            w_neg[OFF_REF_START + i] -= noise;
                        }
                        let f_pos = eval_ref_batch(&PhiGraph::from_weights(w_pos), scenes_ref);
                        let f_neg = eval_ref_batch(&PhiGraph::from_weights(w_neg), scenes_ref);
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

        let mut grad = vec![0.0f32; REF_PARAMS];
        let scale = 1.0 / (2.0 * (cfg.pop_pairs as f32) * ref_sigma * std);
        for (&(pseed, _, _), &diff) in results.iter().zip(diffs.iter()) {
            let mut prng = Rng::new(pseed);
            let coeff = diff * scale;
            for g in &mut grad {
                *g += coeff * prng.next_normal();
            }
        }

        let t = rstep as i32;
        let b1_corr = 1.0 - beta1.powi(t);
        let b2_corr = 1.0 - beta2.powi(t);
        for i in 0..REF_PARAMS {
            let idx = OFF_REF_START + i;
            let g = grad[i] - 1e-4 * theta[idx];
            rm[i] = beta1 * rm[i] + (1.0 - beta1) * g;
            rv[i] = beta2 * rv[i] + (1.0 - beta2) * g * g;
            let m_hat = rm[i] / b1_corr;
            let v_hat = rv[i] / b2_corr;
            theta[idx] += ref_lr * m_hat / (v_hat.sqrt() + eps_adam);
        }

        if rstep == 1 || rstep % 30 == 0 || rstep == ref_steps {
            let cur_model = PhiGraph::from_weights(theta.clone());
            let score = eval_ref_batch(&cur_model, &scenes);
            println!(
                "ref  {:>4}/{:<4} | φ_ref={:+.4} ({:.1}s)",
                rstep,
                ref_steps,
                score,
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

    // 1. Abstraction Méta-relationnelle
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

    // 2. Projection 1-saut sur flux continus auto-segmentés
    let proj_trials = 50;
    let mut hits_glued = 0usize;
    let mut hits_tagged = 0usize;
    let mut ref_hits = 0usize;

    for _ in 0..proj_trials {
        let ep_raw = sample_composite_episode(&mut rng, 4, false, false, true, SepMode::None);
        let regs_raw = discover_causal_regimes(model, &ep_raw.effect, true);
        if regs_raw.len() >= 2 {
            let q_idx = regs_raw.len() - 1;
            let q_reg = regs_raw[q_idx];
            let true_q = ep_raw.hidden_core_spans[ep_raw.hidden_query_idx];
            let true_m = ep_raw.hidden_core_spans[ep_raw.hidden_match_idx];
            if span_iou((q_reg.core_start, q_reg.core_end), true_q) >= 0.45 {
                let matches = project_query_to_regimes(model, &ep_raw.effect, &regs_raw, q_idx);
                if let Some(top) = matches.first()
                    && span_iou((top.core_start, top.core_end), true_m) >= 0.45
                {
                    hits_glued += 1;
                }
            }
        }

        let ep_tag = sample_composite_episode(&mut rng, 4, true, true, true, SepMode::Random);
        let regs_tag = discover_causal_regimes(model, &ep_tag.effect, true);
        if regs_tag.len() >= 2 {
            let q_idx = regs_tag.len() - 1;
            let q_reg = regs_tag[q_idx];
            let true_q = ep_tag.hidden_core_spans[ep_tag.hidden_query_idx];
            let true_m = ep_tag.hidden_core_spans[ep_tag.hidden_match_idx];
            let true_tag = ep_tag.hidden_tag_spans[ep_tag.hidden_match_idx];
            if span_iou((q_reg.core_start, q_reg.core_end), true_q) >= 0.45 {
                let matches = project_query_to_regimes(model, &ep_tag.effect, &regs_tag, q_idx);
                if let Some(top) = matches.first()
                    && span_iou((top.core_start, top.core_end), true_m) >= 0.45
                {
                    hits_tagged += 1;
                    if let (Some(ext_ref), Some(gt_tag)) = (top.referent, true_tag)
                        && span_iou(ext_ref, gt_tag) >= 0.50
                    {
                        ref_hits += 1;
                    }
                }
            }
        }
    }

    // 3. Multi-Sauts 100 % Connexionniste (Champ Référentiel appris, 4 concepts, hasard = 25%)
    let mh_trials = 50;
    let mut mh_decomposed = 0usize;
    let mut mh_exact = 0usize;
    let mut mh_aligned = 0usize;
    let mut mh_flat_exact = 0usize;
    let mut mh_flat_aligned = 0usize;
    let mut mh_margin = 0.0f32;

    for _ in 0..mh_trials {
        let ep = sample_multihop_episode(&mut rng, false, false);
        if let Some(ana) = analyze_multihop(model, &ep.effect) {
            if ana.target_spans.len() >= 2 {
                mh_decomposed += 1;
            }
            if let Some(top) = ana.concepts.first() {
                if top.name_bytes == ep.hidden_concept_name {
                    mh_exact += 1;
                }
                if concept_aligned(&top.name_bytes, &ep.hidden_concept_name) {
                    mh_aligned += 1;
                    if ana.concepts.len() >= 2 {
                        mh_margin += top.score - ana.concepts[1].score;
                    }
                }
            }
        }

        let ep_flat = sample_multihop_episode(&mut rng, true, false);
        if let Some(ana) = analyze_multihop(model, &ep_flat.effect)
            && let Some(top) = ana.concepts.first()
        {
            if top.name_bytes == ep_flat.hidden_concept_name {
                mh_flat_exact += 1;
            }
            if concept_aligned(&top.name_bytes, &ep_flat.hidden_concept_name) {
                mh_flat_aligned += 1;
            }
        }
    }

    println!("\n--- Évaluation post-hoc 1 : Abstraction Méta-relationnelle (0 octet commun) ---");
    println!("  Même cause cachée : cos = {:+.4}", same_avg);
    println!("  Causes différentes: cos = {:+.4}", diff_avg);
    println!("  Marge             : {:+.4}", same_avg - diff_avg);

    println!("\n--- Évaluation post-hoc 2 : Flux continu 1-saut auto-segmenté (4 moules, hasard = 25%) ---");
    println!(
        "  Moules collés SANS séparateur (0 octet commun)     : {}/{} ({:.1}%)",
        hits_glued,
        proj_trials,
        100.0 * (hits_glued as f32) / (proj_trials as f32)
    );
    println!(
        "  Moules étiquetés, sép. aléatoire (0 octet commun)  : {}/{} ({:.1}%)  [référent exact={}/{}]",
        hits_tagged,
        proj_trials,
        100.0 * (hits_tagged as f32) / (proj_trials as f32),
        ref_hits,
        hits_tagged
    );

    println!(
        "\n--- Évaluation post-hoc 3 : Multi-Sauts 100% Connexionniste (4 concepts, hasard = 25%) ---"
    );
    println!(
        "  Décomposition 2 causes (cible hybride)             : {}/{} ({:.1}%)",
        mh_decomposed,
        mh_trials,
        100.0 * (mh_decomposed as f32) / (mh_trials as f32)
    );
    println!(
        "  Crête alignée / exacte (multiligne, 0 octet com.)  : {}/{} ({:.1}%) | exact={}/{}  [marge={:+.4}]",
        mh_aligned,
        mh_trials,
        100.0 * (mh_aligned as f32) / (mh_trials as f32),
        mh_exact,
        mh_trials,
        if mh_aligned > 0 {
            mh_margin / (mh_aligned as f32)
        } else {
            0.0
        }
    );
    println!(
        "  Crête alignée / exacte (flux plat, 0 octet com.)   : {}/{} ({:.1}%) | exact={}/{}",
        mh_flat_aligned,
        mh_trials,
        100.0 * (mh_flat_aligned as f32) / (mh_trials as f32),
        mh_flat_exact,
        mh_trials
    );
}
