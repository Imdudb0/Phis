use crate::graph::{NUM_PARAMS, PhiGraph, STATE_DIM, WM_PARAM_START};
use crate::phi::{
    PreparedComposite, PreparedEpisode, PreparedMultiHop, compute_composite_phi_prepared,
    compute_phi_sharp_prepared, compute_wm_phi_prepared, cosine_sim, discover_causal_regimes,
    project_query_to_regimes, run_working_memory,
};
use crate::ptgrs::{
    CauseFamily, SepMode, sample_composite_episode, sample_episode, sample_episode_in_range,
    sample_multihop_episode,
};
use crate::rng::Rng;
use std::ops::Range;
use std::thread;
use std::time::Instant;

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

fn span_hits_concept(pred: (usize, usize), truth: (usize, usize)) -> bool {
    let inter_s = pred.0.max(truth.0);
    let inter_e = pred.1.min(truth.1);
    if inter_e <= inter_s {
        return false;
    }
    let inter = inter_e - inter_s;
    let truth_len = truth.1.saturating_sub(truth.0).max(1);
    inter * 2 >= truth_len && span_iou(pred, truth) >= 0.30
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

fn eval_hop1_batch(
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
    + 0.30 * mean_gate_c
    + 0.55 * min_gate_c
    - 0.40 * collapse
    + comp_w * mean_comp
}

fn eval_wm_batch(graph: &PhiGraph, multihops: &[PreparedMultiHop]) -> f32 {
    if multihops.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut w_tot = 0.0f32;
    for (i, mh) in multihops.iter().enumerate() {
        let w = if i == 0 { 1.2 } else { 1.0 };
        sum += w * compute_wm_phi_prepared(graph, mh);
        w_tot += w;
    }
    sum / w_tot
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Hop1,
    WorkingMemory,
    Joint,
}

pub fn train_phisharp(cfg: &TrainConfig) -> PhiGraph {
    let start = Instant::now();
    let mut rng = Rng::new(cfg.seed);
    let mut theta = PhiGraph::random(&mut rng).weights;

    let mut m = vec![0.0f32; NUM_PARAMS];
    let mut v = vec![0.0f32; NUM_PARAMS];
    let mut param_steps = vec![0i32; NUM_PARAMS];
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps_adam = 1e-8f32;

    let n_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(cfg.pop_pairs.max(1));

    // Une séance fluide : Hop-1 (55%) -> Mémoire de Travail (35%) -> Joint (10%)
    let p1_end = ((cfg.steps * 60) / 100).max(1);
    let p2_end = ((cfg.steps * 90) / 100).max(p1_end);

    println!(
        "Entraînement Phisharp v0.7.2 ({} params = {} Hop-1 + {} Mémoire de Travail, {} étapes, {} paires ES, {} threads)...",
        NUM_PARAMS,
        WM_PARAM_START,
        NUM_PARAMS - WM_PARAM_START,
        cfg.steps,
        cfg.pop_pairs,
        n_threads
    );
    println!(
        "  Curriculum 1 séance : étapes 1..={} (Hop-1) -> {}..={} (Mémoire de Travail) -> {}..={} (Joint)",
        p1_end,
        p1_end + 1,
        p2_end,
        p2_end + 1,
        cfg.steps
    );

    for step in 1..=cfg.steps {
        let phase = if step <= p1_end {
            Phase::Hop1
        } else if step <= p2_end {
            Phase::WorkingMemory
        } else {
            Phase::Joint
        };

        let active_range: Range<usize> = match phase {
            Phase::Hop1 => 0..WM_PARAM_START,
            Phase::WorkingMemory => WM_PARAM_START..NUM_PARAMS,
            Phase::Joint => 0..NUM_PARAMS,
        };
        let step_sigma = match phase {
            Phase::Hop1 => cfg.sigma,
            Phase::WorkingMemory => cfg.sigma * 1.25,
            Phase::Joint => cfg.sigma * 0.55,
        };
        let step_lr = match phase {
            Phase::Hop1 => cfg.lr,
            Phase::WorkingMemory => cfg.lr * 1.35,
            Phase::Joint => cfg.lr * 0.35,
        };
        let comp_w = 0.10 + 0.45 * ((step as f32) / 120.0).min(1.0);

        let mut singles: Vec<PreparedEpisode> = Vec::new();
        let mut composites: Vec<PreparedComposite> = Vec::new();
        let mut multihops: Vec<PreparedMultiHop> = Vec::new();

        if phase != Phase::WorkingMemory {
            let mut prep_rng = Rng::new(rng.next_u64());
            singles = CauseFamily::ALL
                .iter()
                .map(|&fam| {
                    let eff = sample_episode(&mut rng, fam).effect;
                    PreparedEpisode::from_bytes(&eff, &mut prep_rng)
                })
                .collect();
            composites = vec![
                PreparedComposite::from_bytes(
                    &sample_composite_episode(&mut rng, 3, false, false, false, SepMode::None)
                        .effect,
                ),
                PreparedComposite::from_bytes(
                    &sample_composite_episode(&mut rng, 3, true, false, false, SepMode::Random)
                        .effect,
                ),
            ];
        }

        if phase != Phase::Hop1 {
            // Mélange 1-saut étiqueté + Hop-2 (multiligne & plat) + Hop-3 récursif (E seul)
            multihops = vec![
                PreparedMultiHop::from_bytes(
                    &sample_composite_episode(&mut rng, 4, true, false, true, SepMode::Newline)
                        .effect,
                ),
                PreparedMultiHop::from_bytes(
                    &sample_multihop_episode(&mut rng, false, false, 2).effect,
                ),
                PreparedMultiHop::from_bytes(
                    &sample_multihop_episode(&mut rng, true, false, 2).effect,
                ),
                PreparedMultiHop::from_bytes(
                    &sample_multihop_episode(&mut rng, false, false, 3).effect,
                ),
            ];
        }

        let pair_seeds: Vec<u64> = (0..cfg.pop_pairs).map(|_| rng.next_u64()).collect();
        let a_start = active_range.start;
        let a_end = active_range.end;

        let results: Vec<(u64, f32, f32)> = thread::scope(|s| {
            let chunk_size = cfg.pop_pairs.div_ceil(n_threads);
            let mut handles = Vec::new();
            for chunk in pair_seeds.chunks(chunk_size) {
                let theta_ref = &theta;
                let singles_ref = &singles;
                let comp_ref = &composites;
                let mh_ref = &multihops;
                handles.push(s.spawn(move || {
                    let mut local = Vec::with_capacity(chunk.len());
                    for &pseed in chunk {
                        let mut prng = Rng::new(pseed);
                        let mut w_pos = theta_ref.clone();
                        let mut w_neg = theta_ref.clone();
                        for i in a_start..a_end {
                            let noise = prng.next_normal() * step_sigma;
                            w_pos[i] += noise;
                            w_neg[i] -= noise;
                        }
                        let g_pos = PhiGraph::from_weights(w_pos);
                        let g_neg = PhiGraph::from_weights(w_neg);
                        let (f_pos, f_neg) = match phase {
                            Phase::Hop1 => (
                                eval_hop1_batch(&g_pos, singles_ref, comp_ref, comp_w),
                                eval_hop1_batch(&g_neg, singles_ref, comp_ref, comp_w),
                            ),
                            Phase::WorkingMemory => (
                                eval_wm_batch(&g_pos, mh_ref),
                                eval_wm_batch(&g_neg, mh_ref),
                            ),
                            Phase::Joint => (
                                eval_hop1_batch(&g_pos, singles_ref, comp_ref, comp_w)
                                    + 0.75 * eval_wm_batch(&g_pos, mh_ref),
                                eval_hop1_batch(&g_neg, singles_ref, comp_ref, comp_w)
                                    + 0.75 * eval_wm_batch(&g_neg, mh_ref),
                            ),
                        };
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

        let mut grad = vec![0.0f32; a_end - a_start];
        let scale = 1.0 / (2.0 * (cfg.pop_pairs as f32) * step_sigma * std);
        for (&(pseed, _, _), &diff) in results.iter().zip(diffs.iter()) {
            let mut prng = Rng::new(pseed);
            let coeff = diff * scale;
            for g in &mut grad {
                *g += coeff * prng.next_normal();
            }
        }

        for (offset, &raw_g) in grad.iter().enumerate() {
            let i = a_start + offset;
            param_steps[i] += 1;
            let t = param_steps[i];
            let b1_corr = 1.0 - beta1.powi(t);
            let b2_corr = 1.0 - beta2.powi(t);
            let g = raw_g - 1e-4 * theta[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            let m_hat = m[i] / b1_corr;
            let v_hat = v[i] / b2_corr;
            theta[i] += step_lr * m_hat / (v_hat.sqrt() + eps_adam);
        }

        let log_now = step == 1
            || step == p1_end
            || step == p1_end + 1
            || step == p2_end
            || step == cfg.steps
            || step % 25 == 0;

        if log_now {
            let model = PhiGraph::from_weights(theta.clone());
            match phase {
                Phase::Hop1 => {
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
                        "[H1] step {:>4}/{:<4} | φ={:+.4} | Gen={:.3} Disc={:.3} Spec={:.3} Gate={:.2}/{:.2} IIT={:.3} RefSel={:.3} | res={:.3} coll={:.3} ({:.1}s)",
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
                Phase::WorkingMemory => {
                    let wm_phi = eval_wm_batch(&model, &multihops);
                    let mut drng = Rng::new(cfg.seed ^ (step as u64).wrapping_mul(0x9E37_79B9));
                    let ep = sample_multihop_episode(&mut drng, false, false, 2);
                    if let Some(r) = run_working_memory(&model, &ep.effect) {
                        println!(
                            "[WM] step {:>4}/{:<4} | φ_wm={:+.4} | align={:.3} (cut={:.3}, IIT_echo={:+.3}) | HSel={:.3} RSel={:.3} RDisc={:.3} ({:.1}s)",
                            step,
                            cfg.steps,
                            wm_phi,
                            r.head_cos,
                            r.head_cos_cut,
                            r.iit_echo_gain,
                            r.head_sel,
                            r.readout_sel,
                            r.readout_disc,
                            start.elapsed().as_secs_f32()
                        );
                    } else {
                        println!(
                            "[WM] step {:>4}/{:<4} | φ_wm={:+.4} ({:.1}s)",
                            step,
                            cfg.steps,
                            wm_phi,
                            start.elapsed().as_secs_f32()
                        );
                    }
                }
                Phase::Joint => {
                    let h1_phi: f32 = singles
                        .iter()
                        .map(|ep| compute_phi_sharp_prepared(&model, ep, false).phi_sharp)
                        .sum::<f32>()
                        / (singles.len() as f32);
                    let wm_phi = eval_wm_batch(&model, &multihops);
                    println!(
                        "[JT] step {:>4}/{:<4} | φ_h1={:+.4} | φ_wm={:+.4} ({:.1}s)",
                        step,
                        cfg.steps,
                        h1_phi,
                        wm_phi,
                        start.elapsed().as_secs_f32()
                    );
                }
            }
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

    // 2. Flux continu 1-saut auto-segmenté
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

    // 3. Mémoire de Travail 100% Connexionniste : Hop-2 & Hop-3 (P(t) et pic delta IIT)
    let wm_trials = 50;
    let mut h2_hits = 0usize;
    let mut h2_delta_hits = 0usize;
    let mut h2_flat_hits = 0usize;
    let mut h3_hits = 0usize;
    let mut h3_delta_hits = 0usize;
    let mut h2_iit_sum = 0.0f32;
    let mut h3_iit_sum = 0.0f32;
    let mut h2_margin_sum = 0.0f32;

    for _ in 0..wm_trials {
        let ep2 = sample_multihop_episode(&mut rng, false, false, 2);
        if let Some(rep) = run_working_memory(model, &ep2.effect) {
            h2_iit_sum += rep.iit_echo_gain;
            if span_hits_concept(rep.delta_peak_span, ep2.hidden_concept_span) {
                h2_delta_hits += 1;
            }
            if let Some(top) = rep.top_spans.first()
                && span_hits_concept((top.start, top.end), ep2.hidden_concept_span)
            {
                h2_hits += 1;
                if let Some(sec) = rep.top_spans.get(1) {
                    h2_margin_sum += top.peak_p - sec.peak_p;
                }
            }
        }

        let ep2f = sample_multihop_episode(&mut rng, true, false, 2);
        if let Some(rep) = run_working_memory(model, &ep2f.effect)
            && let Some(top) = rep.top_spans.first()
            && span_hits_concept((top.start, top.end), ep2f.hidden_concept_span)
        {
            h2_flat_hits += 1;
        }

        let ep3 = sample_multihop_episode(&mut rng, false, false, 3);
        if let Some(rep) = run_working_memory(model, &ep3.effect) {
            h3_iit_sum += rep.iit_echo_gain;
            if span_hits_concept(rep.delta_peak_span, ep3.hidden_concept_span) {
                h3_delta_hits += 1;
            }
            if let Some(top) = rep.top_spans.first()
                && span_hits_concept((top.start, top.end), ep3.hidden_concept_span)
            {
                h3_hits += 1;
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
        "\n--- Évaluation post-hoc 3 : Mémoire de Travail 100% Connexionniste (0 parsing symbolique) ---"
    );
    println!(
        "  Hop-2 multiligne (4 déf., hasard = 25%) : P(t)={}/{} ({:.1}%) | Δ_IIT={}/{} ({:.1}%)  [marge P={:+.3}, IIT_echo={:+.3}]",
        h2_hits,
        wm_trials,
        100.0 * (h2_hits as f32) / (wm_trials as f32),
        h2_delta_hits,
        wm_trials,
        100.0 * (h2_delta_hits as f32) / (wm_trials as f32),
        if h2_hits > 0 {
            h2_margin_sum / (h2_hits as f32)
        } else {
            0.0
        },
        h2_iit_sum / (wm_trials as f32)
    );
    println!(
        "  Hop-2 flux plat  (4 déf., hasard = 25%) : P(t)={}/{} ({:.1}%)",
        h2_flat_hits,
        wm_trials,
        100.0 * (h2_flat_hits as f32) / (wm_trials as f32)
    );
    println!(
        "  Hop-3 récursif   (5 déf., 3 causes)     : P(t)={}/{} ({:.1}%) | Δ_IIT={}/{} ({:.1}%)  [IIT_echo={:+.3}]",
        h3_hits,
        wm_trials,
        100.0 * (h3_hits as f32) / (wm_trials as f32),
        h3_delta_hits,
        wm_trials,
        100.0 * (h3_delta_hits as f32) / (wm_trials as f32),
        h3_iit_sum / (wm_trials as f32)
    );
}