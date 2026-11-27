use std::path::Path;
use std::thread;

use crate::core::*;
use crate::model::*;

// ============================================================================
// 1. Entraînement fermé multi-objectif (Untrained / LL / MDL / φ_sharp)
//    Architecture V3 identique, PTGRS générique identique, multi-budget (T, T+2).
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrainObjective {
    Untrained,
    Likelihood,
    Mdl,
    PhiSharp,
}

impl TrainObjective {
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s {
            "untrained" | "step0" | "random" => Some(Self::Untrained),
            "ll" | "likelihood" | "vraisemblance" => Some(Self::Likelihood),
            "mdl" => Some(Self::Mdl),
            "phi" | "phisharp" | "phi_sharp" => Some(Self::PhiSharp),
            _ => None,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Untrained => "Step 0 (aléatoire)",
            Self::Likelihood => "Baseline Vraisemblance (LL)",
            Self::Mdl => "Baseline MDL",
            Self::PhiSharp => "Phisharp V3 (φ_sharp)",
        }
    }
}

pub(crate) fn train_model(
    out_path: &Path,
    steps: usize,
    seed: u64,
    objective: TrainObjective,
) -> PhisharpV3 {
    let mut rng = Rng::new(seed);
    let mut model = PhisharpV3::new_random(&mut rng);

    if objective == TrainObjective::Untrained || steps == 0 {
        model.save(out_path).expect("Échec écriture modèle Step 0");
        println!(
            "Modèle {} (0 steps, seed={}) sauvegardé : {}",
            objective.label(),
            seed,
            out_path.display()
        );
        return model;
    }

    let pop_pairs = 12usize; // 24 évaluations antithétiques par étape
    let sigma: f64 = 0.075;
    let lr: f64 = 0.055;
    let batch_size = 5usize;

    let mut m_adam = vec![0.0_f64; PARAM_COUNT];
    let mut v_adam = vec![0.0_f64; PARAM_COUNT];
    let beta1: f64 = 0.9;
    let beta2: f64 = 0.999;

    println!(
        "=== ENTRAÎNEMENT FERMÉ V3 [{}] (PTGRS générique, circulation pure, {} params) ===",
        objective.label(),
        PARAM_COUNT
    );

    for step in 1..=steps {
        let t_budget = 3 + rng.range_usize(5); // T dans 3..=7
        let t_next = t_budget + 2;

        let mut batch_views = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let cause = PtgrsCause::sample_generic(&mut rng);
            let depth = 3 + rng.range_usize(3);
            let bytes = cause.emit_effect(&mut rng, depth, 150);
            batch_views.push(RelationalView::from_bytes(&bytes));
        }

        let mut epsilons: Vec<Vec<f64>> = Vec::with_capacity(pop_pairs);
        for _ in 0..pop_pairs {
            let mut e = vec![0.0_f64; PARAM_COUNT];
            for val in &mut e {
                *val = rng.next_normal();
            }
            epsilons.push(e);
        }

        let base_params = &model.params;
        let views_ref = &batch_views;
        let mut diffs = vec![0.0_f64; pop_pairs];

        thread::scope(|s| {
            let mut handles = Vec::with_capacity(pop_pairs);
            for (idx, eps) in epsilons.iter().enumerate() {
                handles.push(s.spawn(move || {
                    let mut p_plus = base_params.clone();
                    let mut p_minus = base_params.clone();
                    for i in 0..PARAM_COUNT {
                        p_plus[i] += sigma * eps[i];
                        p_minus[i] -= sigma * eps[i];
                    }
                    let m_plus = PhisharpV3 { params: p_plus };
                    let m_minus = PhisharpV3 { params: p_minus };

                    let eval_cand = |m: &PhisharpV3| -> f64 {
                        let mut acc = 0.0_f64;
                        let snaps = [t_budget, t_next];
                        for view in views_ref {
                            let tr = m.run_abduction(view, t_next, true, &snaps, false);
                            match objective {
                                TrainObjective::PhiSharp => {
                                    let r1 = m.evaluate_phi_state(
                                        view,
                                        &tr.snapshots[0].1,
                                        t_budget,
                                        true,
                                    );
                                    let r2 = m.evaluate_phi_state(
                                        view,
                                        &tr.snapshots[1].1,
                                        t_next,
                                        true,
                                    );
                                    let drop_pen = (r1.phi_sharp - r2.phi_sharp).max(0.0_f64);
                                    acc += 0.45_f64 * r1.shaping_score
                                        + 0.45_f64 * r2.shaping_score
                                        + 0.10_f64 * r1.shaping_score.min(r2.shaping_score)
                                        - 0.20_f64 * drop_pen;
                                }
                                TrainObjective::Likelihood => {
                                    let u1 = m.unfold_top_down(&tr.snapshots[0].1, None, false);
                                    let u2 = m.unfold_top_down(&tr.snapshots[1].1, None, false);
                                    let s1 =
                                        log_likelihood_graph(&view.graph, &u1.r_hat, view.v_size);
                                    let s2 =
                                        log_likelihood_graph(&view.graph, &u2.r_hat, view.v_size);
                                    let drop_pen = (s1 - s2).max(0.0_f64);
                                    acc += 0.45_f64 * s1
                                        + 0.45_f64 * s2
                                        + 0.10_f64 * s1.min(s2)
                                        - 0.20_f64 * drop_pen;
                                }
                                TrainObjective::Mdl => {
                                    let u1 = m.unfold_top_down(&tr.snapshots[0].1, None, false);
                                    let u2 = m.unfold_top_down(&tr.snapshots[1].1, None, false);
                                    let s1 = mdl_graph_v3(
                                        &view.graph,
                                        &u1.r_hat,
                                        &tr.snapshots[0].1,
                                        view.seq.len(),
                                        view.v_size,
                                    );
                                    let s2 = mdl_graph_v3(
                                        &view.graph,
                                        &u2.r_hat,
                                        &tr.snapshots[1].1,
                                        view.seq.len(),
                                        view.v_size,
                                    );
                                    let drop_pen = (s1 - s2).max(0.0_f64);
                                    acc += 0.45_f64 * s1
                                        + 0.45_f64 * s2
                                        + 0.10_f64 * s1.min(s2)
                                        - 0.20_f64 * drop_pen;
                                }
                                TrainObjective::Untrained => {}
                            }
                        }
                        acc / (views_ref.len() as f64)
                    };

                    (idx, eval_cand(&m_plus) - eval_cand(&m_minus))
                }));
            }
            for h in handles {
                let (idx, d) = h.join().unwrap();
                diffs[idx] = d;
            }
        });

        let std_d = (diffs.iter().map(|d| d * d).sum::<f64>() / (pop_pairs as f64))
            .sqrt()
            .max(1e-6_f64);

        let mut grad = vec![0.0_f64; PARAM_COUNT];
        for (idx, eps) in epsilons.iter().enumerate() {
            let coeff = (diffs[idx] / std_d) / ((2 * pop_pairs) as f64 * sigma);
            for i in 0..PARAM_COUNT {
                grad[i] += coeff * eps[i];
            }
        }

        let b1_corr = 1.0_f64 - beta1.powi(step as i32);
        let b2_corr = 1.0_f64 - beta2.powi(step as i32);
        for i in 0..PARAM_COUNT {
            m_adam[i] = beta1 * m_adam[i] + (1.0_f64 - beta1) * grad[i];
            v_adam[i] = beta2 * v_adam[i] + (1.0_f64 - beta2) * grad[i] * grad[i];
            let m_hat = m_adam[i] / b1_corr;
            let v_hat = v_adam[i] / b2_corr;
            model.params[i] += lr * m_hat / (v_hat.sqrt() + 1e-8_f64);
            model.params[i] = model.params[i].clamp(-4.0_f64, 4.0_f64);
        }

        if step == 1 || step % 50 == 0 || step == steps {
            let mut g_s = 0.0_f64;
            let mut d_s = 0.0_f64;
            let mut pt_s = 0.0_f64;
            let mut ps_s = 0.0_f64;
            let mut cut_s = 0.0_f64;
            let mut sr_s = 0.0_f64;
            let mut sd_s = 0.0_f64;
            let mut sa_s = 0.0_f64;
            let mut res_s = 0.0_f64;
            let mut phi_t = 0.0_f64;
            let mut phi_t2 = 0.0_f64;
            let snaps = [t_budget, t_next];

            for view in &batch_views {
                let tr = model.run_abduction(view, t_next, true, &snaps, false);
                let r1 = model.evaluate_phi_state(view, &tr.snapshots[0].1, t_budget, true);
                let r2 = model.evaluate_phi_state(view, &tr.snapshots[1].1, t_next, true);
                g_s += r1.generativity;
                d_s += r1.discrimination;
                pt_s += r1.proj_temp;
                ps_s += r1.proj_scale;
                cut_s += r1.phi_cut;
                sr_s += r1.s_ref;
                sd_s += r1.s_disp;
                sa_s += r1.s_abs;
                res_s += r1.residual_norm;
                phi_t += r1.phi_sharp;
                phi_t2 += r2.phi_sharp;
            }
            let nb = batch_views.len() as f64;
            println!(
                "step {:4}/{} (T={},{}) | φ={:.4}->{:.4} | G={:.2} D={:.2} Pt={:.2} Ps={:.2} | Φc={:.2} R/D/A={:.2}/{:.2}/{:.2} res={:.2}",
                step,
                steps,
                t_budget,
                t_next,
                phi_t / nb,
                phi_t2 / nb,
                g_s / nb,
                d_s / nb,
                pt_s / nb,
                ps_s / nb,
                cut_s / nb,
                sr_s / nb,
                sd_s / nb,
                sa_s / nb,
                res_s / nb
            );
        }
    }

    model.save(out_path).expect("Échec écriture modèle V3");
    println!(
        "Modèle [{}] figé sauvegardé : {}",
        objective.label(),
        out_path.display()
    );
    model
}

// ============================================================================
// 2. Porte d'étalonnage V3 (C connue après coup + 6 témoins négatifs stricts)
// ============================================================================

pub(crate) struct CalibrationSummary {
    pub(crate) r_phi: f64,
    pub(crate) r_ll: f64,
    pub(crate) r_mdl: f64,
    pub(crate) s_phi: f64,
    pub(crate) s_ll: f64,
    pub(crate) s_mdl: f64,
    pub(crate) within_r_phi: f64,
    pub(crate) within_r_ll: f64,
    pub(crate) within_r_mdl: f64,
    pub(crate) mean_phi: f64,
    pub(crate) mean_eq: f64,
    pub(crate) rand_phi: f64,
    pub(crate) h_eq_e_phi: f64,
    pub(crate) overgen_phi: f64,
    pub(crate) perm_mean: f64,
    pub(crate) perm_max: f64,
    pub(crate) phi_clean: f64,
    pub(crate) phi_l1_dead: f64,
    pub(crate) phi_l1_clone: f64,
    pub(crate) phi_l2_clone: f64,
    pub(crate) max_phi_diff: f64,
    pub(crate) ask_exact: bool,
    pub(crate) signal_alive: bool,
    pub(crate) pass_corr: bool,
    pub(crate) pass_neg: bool,
    pub(crate) pass_inv: bool,
}

pub(crate) fn evaluate_calibration(model: &PhisharpV3, seed: u64) -> CalibrationSummary {
    let mut rng = Rng::new(seed);
    let mut eval_rng = Rng::new(seed ^ 0x55AA_CC33);

    let mut eq_scores = Vec::new();
    let mut phi_scores = Vec::new();
    let mut ll_scores = Vec::new();
    let mut mdl_scores = Vec::new();

    for trial in 0..36 {
        let cause = PtgrsCause::sample_generic(&mut rng);
        let depth = 3 + (trial % 3);
        let len = match trial % 3 {
            0 => 96,
            1 => 144,
            _ => 192,
        };
        let bytes = cause.emit_effect(&mut rng, depth, len);
        let view = RelationalView::from_bytes(&bytes);
        let truth = cause.ground_truth_multi_scale(&mut eval_rng, depth);

        let budget = match trial % 6 {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 4,
            4 => 6,
            _ => 8,
        };
        let use_ref = (trial % 4) != 1;
        let trace = model.run_abduction(&view, budget, use_ref, &[], false);
        let rep = model.evaluate_phi_state(&view, &trace.state, budget, use_ref);
        let unfold = model.unfold_top_down(&trace.state, None, false);

        let eq = observational_equivalence_excess(&view, &unfold.r_hat, &truth);
        let ll = log_likelihood_graph(&view.graph, &unfold.r_hat, view.v_size);
        let mdl = mdl_graph_v3(
            &view.graph,
            &unfold.r_hat,
            &trace.state,
            view.seq.len(),
            view.v_size,
        );

        eq_scores.push(eq);
        phi_scores.push(rep.phi_sharp);
        ll_scores.push(ll);
        mdl_scores.push(mdl);
    }

    let r_phi = pearson_corr(&phi_scores, &eq_scores);
    let r_ll = pearson_corr(&ll_scores, &eq_scores);
    let r_mdl = pearson_corr(&mdl_scores, &eq_scores);

    let s_phi = spearman_corr(&phi_scores, &eq_scores);
    let s_ll = spearman_corr(&ll_scores, &eq_scores);
    let s_mdl = spearman_corr(&mdl_scores, &eq_scores);

    let mean_phi = phi_scores.iter().sum::<f64>() / (phi_scores.len() as f64);
    let mean_eq = eq_scores.iter().sum::<f64>() / (eq_scores.len() as f64);

    let traj_budgets = [0usize, 1, 2, 4, 6, 8];
    let mut within_r_phi = 0.0_f64;
    let mut within_r_ll = 0.0_f64;
    let mut within_r_mdl = 0.0_f64;
    for _ in 0..12 {
        let cause = PtgrsCause::sample_generic(&mut rng);
        let bytes = cause.emit_effect(&mut rng, 4, 160);
        let view = RelationalView::from_bytes(&bytes);
        let truth = cause.ground_truth_multi_scale(&mut eval_rng, 4);

        let tr = model.run_abduction(&view, 8, true, &traj_budgets, false);
        let mut eqs = Vec::with_capacity(traj_budgets.len());
        let mut phis = Vec::with_capacity(traj_budgets.len());
        let mut lls = Vec::with_capacity(traj_budgets.len());
        let mut mdls = Vec::with_capacity(traj_budgets.len());

        for (b, st) in &tr.snapshots {
            let rep = model.evaluate_phi_state(&view, st, *b, true);
            let unf = model.unfold_top_down(st, None, false);
            eqs.push(observational_equivalence_excess(&view, &unf.r_hat, &truth));
            phis.push(rep.phi_sharp);
            lls.push(log_likelihood_graph(&view.graph, &unf.r_hat, view.v_size));
            mdls.push(mdl_graph_v3(
                &view.graph,
                &unf.r_hat,
                st,
                view.seq.len(),
                view.v_size,
            ));
        }
        within_r_phi += pearson_corr(&phis, &eqs);
        within_r_ll += pearson_corr(&lls, &eqs);
        within_r_mdl += pearson_corr(&mdls, &eqs);
    }
    within_r_phi /= 12.0_f64;
    within_r_ll /= 12.0_f64;
    within_r_mdl /= 12.0_f64;

    let mut rand_phi = 0.0_f64;
    for _ in 0..10 {
        let bytes: Vec<u8> = (0..160).map(|_| (rng.next_u64() % 10) as u8).collect();
        let view = RelationalView::from_bytes(&bytes);
        rand_phi += model.evaluate_phi(&view, DEFAULT_BUDGET, true).phi_sharp;
    }
    rand_phi /= 10.0_f64;

    let cause_ref = PtgrsCause::sample_generic(&mut rng);
    let bytes_ref = cause_ref.emit_effect(&mut rng, 4, 160);
    let view_ref = RelationalView::from_bytes(&bytes_ref);
    let empty_state = CircState::zeros(view_ref.v_size, 0, 0);
    let h_eq_e_phi = model
        .evaluate_phi_state(&view_ref, &empty_state, DEFAULT_BUDGET, true)
        .phi_sharp;

    let zero_state = CircState::zeros(view_ref.v_size, K1, K2);
    let overgen_phi = model
        .evaluate_phi_state(&view_ref, &zero_state, DEFAULT_BUDGET, true)
        .phi_sharp;

    let mut perm_sum = 0.0_f64;
    let mut perm_max = 0.0_f64;
    for _ in 0..10 {
        let c = PtgrsCause::sample_generic(&mut rng);
        let mut b = c.emit_effect(&mut rng, 4, 160);
        rng.shuffle(&mut b);
        let vp = RelationalView::from_bytes(&b);
        let p = model.evaluate_phi(&vp, DEFAULT_BUDGET, true).phi_sharp;
        perm_sum += p;
        if p > perm_max {
            perm_max = p;
        }
    }
    let perm_mean = perm_sum / 10.0_f64;

    let trace_ref = model.run_abduction(&view_ref, DEFAULT_BUDGET, true, &[], false);
    let phi_clean = model
        .evaluate_phi_state(&view_ref, &trace_ref.state, DEFAULT_BUDGET, true)
        .phi_sharp;

    let l1_dead = trace_ref.state.with_dead_l1_node();
    let phi_l1_dead = model
        .evaluate_phi_state(&view_ref, &l1_dead, DEFAULT_BUDGET, true)
        .phi_sharp;

    let l1_clone = trace_ref.state.with_cloned_l1_node(0);
    let phi_l1_clone = model
        .evaluate_phi_state(&view_ref, &l1_clone, DEFAULT_BUDGET, true)
        .phi_sharp;

    let l2_clone = trace_ref.state.with_cloned_l2_node(0);
    let phi_l2_clone = model
        .evaluate_phi_state(&view_ref, &l2_clone, DEFAULT_BUDGET, true)
        .phi_sharp;

    let mut bij: Vec<u8> = (0u8..=255).collect();
    rng.shuffle(&mut bij);
    let mut inv_bij = [0u8; 256];
    for (orig, &mapped) in bij.iter().enumerate() {
        inv_bij[mapped as usize] = orig as u8;
    }
    let renamed_bytes: Vec<u8> = bytes_ref.iter().map(|&b| bij[b as usize]).collect();
    let view_ren = RelationalView::from_bytes(&renamed_bytes);

    let tr_orig = model.run_abduction(&view_ref, DEFAULT_BUDGET, true, &[], true);
    let tr_ren = model.run_abduction(&view_ren, DEFAULT_BUDGET, true, &[], true);
    let max_phi_diff = tr_orig
        .phi_history
        .iter()
        .zip(tr_ren.phi_history.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let out_orig = model.ask_emit(&view_ref, DEFAULT_BUDGET, 24);
    let out_ren = model.ask_emit(&view_ren, DEFAULT_BUDGET, 24);
    let out_ren_inv: Vec<u8> = out_ren.iter().map(|&b| inv_bij[b as usize]).collect();
    let ask_exact = out_orig == out_ren_inv;

    let signal_alive = mean_phi > 0.015_f64 && phi_clean > 0.015_f64;
    let pass_corr = signal_alive
        && ((r_phi > r_ll && r_phi > r_mdl && r_phi > 0.35_f64)
            || (s_phi > s_ll && s_phi > s_mdl && s_phi > 0.40_f64 && r_phi > 0.30_f64));
    let pass_neg = signal_alive
        && rand_phi < 0.03_f64
        && h_eq_e_phi < 1e-9_f64
        && overgen_phi < 0.02_f64
        && perm_mean < 0.03_f64
        && perm_max < 0.08_f64
        && phi_l1_dead < 1e-9_f64
        && phi_l1_clone < 1e-9_f64
        && phi_l2_clone < 1e-9_f64;
    let pass_inv = max_phi_diff < 1e-9_f64 && ask_exact;

    CalibrationSummary {
        r_phi,
        r_ll,
        r_mdl,
        s_phi,
        s_ll,
        s_mdl,
        within_r_phi,
        within_r_ll,
        within_r_mdl,
        mean_phi,
        mean_eq,
        rand_phi,
        h_eq_e_phi,
        overgen_phi,
        perm_mean,
        perm_max,
        phi_clean,
        phi_l1_dead,
        phi_l1_clone,
        phi_l2_clone,
        max_phi_diff,
        ask_exact,
        signal_alive,
        pass_corr,
        pass_neg,
        pass_inv,
    }
}

pub(crate) fn run_calibration(model: &PhisharpV3, seed: u64) {
    let c = evaluate_calibration(model, seed);
    println!("=== PORTE D'ÉTALONNAGE V3 (PTGRS générique, récupération excédentaire C) ===");
    println!(
        "Inter-causes Pearson  vs Rec_exc(C) : φ_sharp={:+.4} | Vraisemblance={:+.4} | MDL={:+.4}",
        c.r_phi, c.r_ll, c.r_mdl
    );
    println!(
        "Inter-causes Spearman vs Rec_exc(C) : φ_sharp={:+.4} | Vraisemblance={:+.4} | MDL={:+.4}",
        c.s_phi, c.s_ll, c.s_mdl
    );
    println!(
        "Intra-cause  Pearson  (t=0..8, 12x) : φ_sharp={:+.4} | Vraisemblance={:+.4} | MDL={:+.4}",
        c.within_r_phi, c.within_r_ll, c.within_r_mdl
    );
    println!(
        "Moyennes PTGRS (t=0..8)             : φ_sharp={:.4} | Rec_exc(C)={:+.4}",
        c.mean_phi, c.mean_eq
    );
    println!("--- Témoins négatifs ---");
    println!("  Octets aléatoires (10x)       : φ_sharp = {:.4}", c.rand_phi);
    println!("  H := E (sans intégration)     : φ_sharp = {:.4}", c.h_eq_e_phi);
    println!("  Cause sur-générale            : φ_sharp = {:.4}", c.overgen_phi);
    println!(
        "  E permuté à marginales égales : moy={:.4} | max={:.4}",
        c.perm_mean, c.perm_max
    );
    println!(
        "  Branches épiphénoménales      : propre={:.4} -> L1_morte={:.4} | L1_clone={:.4} | L2_clone={:.4}",
        c.phi_clean, c.phi_l1_dead, c.phi_l1_clone, c.phi_l2_clone
    );
    println!(
        "  Invariance bijective          : max|Δφ(t)|={:.2e} | ask(π(E))==π(ask(E)) : {}",
        c.max_phi_diff, c.ask_exact
    );
    println!("--- VERDICT ÉTALONNAGE V3 ---");
    if c.pass_corr && c.pass_neg && c.pass_inv {
        println!("PASS : A est lisible.");
    } else {
        println!(
            "ARRÊT : Étalonnage échoué (alive={}, corr_ok={}, neg_ok={}, inv_ok={}).",
            c.signal_alive, c.pass_corr, c.pass_neg, c.pass_inv
        );
    }
}

// ============================================================================
// 3. Barreau 1 : PTGRS OOD (anytime, ablations, Rec_exc(C), triple résidu)
// ============================================================================

pub(crate) struct OodDepthStats {
    pub(crate) depth: usize,
    pub(crate) curves: Vec<f64>,
    pub(crate) no_ref_t6: f64,
    pub(crate) rec_exc_t6: f64,
    pub(crate) rec_exc_t10: f64,
    pub(crate) g_t6: f64,
    pub(crate) p_t6: f64,
    pub(crate) phi_cut_t6: f64,
    pub(crate) s_abs_t6: f64,
}

pub(crate) struct OodSummary {
    pub(crate) depths: Vec<OodDepthStats>,
    pub(crate) gain_true: f64,
    pub(crate) gain_foreign: f64,
    pub(crate) gain_echo: f64,
}

pub(crate) fn evaluate_ood(model: &PhisharpV3, seed: u64) -> OodSummary {
    let mut rng = Rng::new(seed);
    let mut truth_rng = Rng::new(seed ^ 0x7711_9933);
    let max_t = 10usize;
    let all_snaps: Vec<usize> = (0..=max_t).collect();
    let mut depths = Vec::new();

    for &depth in &[2usize, 4, 8] {
        let mut curves = vec![0.0_f64; max_t + 1];
        let mut no_ref_sum = 0.0_f64;
        let mut rec6_sum = 0.0_f64;
        let mut rec10_sum = 0.0_f64;
        let mut g6_sum = 0.0_f64;
        let mut p6_sum = 0.0_f64;
        let mut cut6_sum = 0.0_f64;
        let mut abs6_sum = 0.0_f64;
        let trials = 12usize;

        for _ in 0..trials {
            let cause = PtgrsCause::sample_ood(&mut rng, depth);
            let bytes = cause.emit_effect(&mut rng, depth, 180);
            let view = RelationalView::from_bytes(&bytes);
            let truth = cause.ground_truth_multi_scale(&mut truth_rng, depth);

            let tr = model.run_abduction(&view, max_t, true, &all_snaps, false);
            for (step, st) in &tr.snapshots {
                let rep = model.evaluate_phi_state(&view, st, *step, true);
                curves[*step] += rep.phi_sharp;
                if *step == DEFAULT_BUDGET {
                    g6_sum += rep.generativity;
                    p6_sum += rep.projection;
                    cut6_sum += rep.phi_cut;
                    abs6_sum += rep.s_abs;
                    let unf = model.unfold_top_down(st, None, false);
                    rec6_sum += observational_equivalence_excess(&view, &unf.r_hat, &truth);
                } else if *step == max_t {
                    let unf = model.unfold_top_down(st, None, false);
                    rec10_sum += observational_equivalence_excess(&view, &unf.r_hat, &truth);
                }
            }
            no_ref_sum += model.evaluate_phi(&view, DEFAULT_BUDGET, false).phi_sharp;
        }

        let n = trials as f64;
        for val in &mut curves {
            *val /= n;
        }
        depths.push(OodDepthStats {
            depth,
            curves,
            no_ref_t6: no_ref_sum / n,
            rec_exc_t6: rec6_sum / n,
            rec_exc_t10: rec10_sum / n,
            g_t6: g6_sum / n,
            p_t6: p6_sum / n,
            phi_cut_t6: cut6_sum / n,
            s_abs_t6: abs6_sum / n,
        });
    }

    let mut gain_true = 0.0_f64;
    let mut gain_foreign = 0.0_f64;
    let mut gain_echo = 0.0_f64;
    let trials_r = 15usize;

    for _ in 0..trials_r {
        let c_true = PtgrsCause::sample_ood(&mut rng, 4);
        let c_other = PtgrsCause::sample_ood(&mut rng, 4);

        let full_true = c_true.emit_effect(&mut rng, 4, 180);
        let foreign_chunk = c_other.emit_effect(&mut rng, 4, 60);

        let partial = &full_true[0..120];
        let mut with_true = partial.to_vec();
        with_true.extend_from_slice(&full_true[120..180]);

        let mut with_foreign = partial.to_vec();
        with_foreign.extend_from_slice(&foreign_chunk);

        let mut with_echo = partial.to_vec();
        with_echo.extend_from_slice(&partial[0..60]);

        let phi_partial = model
            .evaluate_phi(&RelationalView::from_bytes(partial), DEFAULT_BUDGET, true)
            .phi_sharp;
        let phi_true = model
            .evaluate_phi(&RelationalView::from_bytes(&with_true), DEFAULT_BUDGET, true)
            .phi_sharp;
        let phi_foreign = model
            .evaluate_phi(
                &RelationalView::from_bytes(&with_foreign),
                DEFAULT_BUDGET,
                true,
            )
            .phi_sharp;
        let phi_echo = model
            .evaluate_phi(&RelationalView::from_bytes(&with_echo), DEFAULT_BUDGET, true)
            .phi_sharp;

        gain_true += phi_true - phi_partial;
        gain_foreign += phi_foreign - phi_partial;
        gain_echo += phi_echo - phi_partial;
    }

    OodSummary {
        depths,
        gain_true: gain_true / (trials_r as f64),
        gain_foreign: gain_foreign / (trials_r as f64),
        gain_echo: gain_echo / (trials_r as f64),
    }
}

pub(crate) fn run_eval_ood(model: &PhisharpV3, seed: u64) {
    let ood = evaluate_ood(model, seed);
    println!("=== ÉVALUATION BARREAU 1 V3 (PTGRS OOD : profondeurs d=2, 4, 8) ===");
    for d in &ood.depths {
        print!("Profondeur d={} | Anytime φ(t=0..10): ", d.depth);
        for (b, val) in d.curves.iter().enumerate() {
            print!("t{}={:.3} ", b, val);
        }
        println!();
        let ret = if d.curves[DEFAULT_BUDGET] > 1e-6_f64 {
            d.curves[10] / d.curves[DEFAULT_BUDGET]
        } else {
            0.0_f64
        };
        println!(
            "  Ablations (d={}) : t0={:.4} | no_ref={:.4} vs ref={:.4} | t10/t6={:.2}x | Φc={:.2} S_abs={:.2} | Rec(C) t6={:+.4} t10={:+.4}",
            d.depth,
            d.curves[0],
            d.no_ref_t6,
            d.curves[DEFAULT_BUDGET],
            ret,
            d.phi_cut_t6,
            d.s_abs_t6,
            d.rec_exc_t6,
            d.rec_exc_t10
        );
    }
    println!("--- Test du résidu (vrai complément vs étranger C' vs écho) ---");
    println!(
        "Δφ_sharp moyen : vrai complément={:+.4} | étranger C'={:+.4} | écho={:+.4}",
        ood.gain_true, ood.gain_foreign, ood.gain_echo
    );
}

// ============================================================================
// 4. Test de rupture de générateur hors-PTGRS (CA 1D & algorithmique)
// ============================================================================

fn gen_elementary_ca(rule: u8, seed: u64, target_len: usize) -> Vec<u8> {
    let width = 12usize;
    let mut cells = vec![0u8; width];
    let mut rng = Rng::new(seed);
    for c in &mut cells {
        *c = (rng.next_u64() & 1) as u8;
    }
    if cells.iter().all(|&x| x == 0) {
        cells[width / 2] = 1;
    }

    for _ in 0..6 {
        let mut next = vec![0u8; width];
        for i in 0..width {
            let l = cells[(i + width - 1) % width];
            let m = cells[i];
            let r = cells[(i + 1) % width];
            let pat = (l << 2) | (m << 1) | r;
            next[i] = (rule >> pat) & 1;
        }
        cells = next;
    }

    let mut out = Vec::with_capacity(target_len + 4);
    while out.len() < target_len {
        for blk in 0..(width / 3) {
            let sym = (cells[3 * blk] << 2) | (cells[3 * blk + 1] << 1) | cells[3 * blk + 2];
            out.push(sym);
        }
        let mut next = vec![0u8; width];
        for i in 0..width {
            let l = cells[(i + width - 1) % width];
            let m = cells[i];
            let r = cells[(i + 1) % width];
            let pat = (l << 2) | (m << 1) | r;
            next[i] = (rule >> pat) & 1;
        }
        cells = next;
    }
    out.truncate(target_len);
    out
}

fn gen_fibonacci_mod(modulus: u8, seed: u64, target_len: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut a = (1 + (rng.next_u64() as u8) % (modulus - 1)) % modulus;
    let mut b = (1 + (rng.next_u64() as u8) % (modulus - 1)) % modulus;
    let mut out = Vec::with_capacity(target_len);
    for _ in 0..target_len {
        out.push(a);
        let c = (a.wrapping_add(b)) % modulus;
        a = b;
        b = c;
    }
    out
}

fn gen_collatz_trace(seed: u64, target_len: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut n = 27u64 + (rng.next_u64() % 500);
    let mut out = Vec::with_capacity(target_len);
    while out.len() < target_len {
        out.push((n % 8) as u8);
        if n <= 1 {
            n = 27u64 + (rng.next_u64() % 500);
        } else if n % 2 == 0 {
            n /= 2;
        } else {
            n = 3 * n + 1;
        }
    }
    out
}

fn gen_thue_morse_blocks(offset: u64, target_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let idx = offset + (i as u64);
        let b0 = (idx.count_ones() & 1) as u8;
        let b1 = ((idx + 1).count_ones() & 1) as u8;
        let b2 = ((idx + 2).count_ones() & 1) as u8;
        out.push((b0 << 2) | (b1 << 1) | b2);
    }
    out
}

fn gen_structured_frames(seed: u64, target_len: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut counter = (rng.next_u64() % 5) as u8;
    let mut out = Vec::with_capacity(target_len + 4);
    while out.len() < target_len {
        let field = (counter % 3) + 1;
        let val = (counter + field) % 5;
        let chk = (field + val + 2) % 5;
        out.push(9u8);
        out.push(5u8 + field);
        out.push(val);
        out.push(chk);
        counter = (counter + 1) % 5;
    }
    out.truncate(target_len);
    out
}

pub(crate) struct TransferFamilyStats {
    pub(crate) name: &'static str,
    pub(crate) phi_real: f64,
    pub(crate) phi_perm: f64,
    pub(crate) g: f64,
    pub(crate) d: f64,
    pub(crate) p: f64,
    pub(crate) phi_cut: f64,
    pub(crate) s_abs: f64,
    pub(crate) no_ref: f64,
    pub(crate) retention: f64,
}

pub(crate) struct TransferSummary {
    pub(crate) families: Vec<TransferFamilyStats>,
    pub(crate) passed_count: usize,
    pub(crate) total_structured: usize,
    pub(crate) mean_phi_structured: f64,
    pub(crate) mean_p_structured: f64,
    pub(crate) mean_cut_structured: f64,
    pub(crate) mean_sabs_structured: f64,
    pub(crate) chaos_phi: f64,
}

pub(crate) fn evaluate_transfer(model: &PhisharpV3, seed: u64) -> TransferSummary {
    let mut rng = Rng::new(seed);
    let specs: Vec<(&'static str, fn(&mut Rng) -> Vec<u8>, bool)> = vec![
        ("CA 1D Rule 110 (Turing)", |r| gen_elementary_ca(110, r.next_u64(), 180), true),
        ("CA 1D Rule 54 (Gliders)", |r| gen_elementary_ca(54, r.next_u64(), 180), true),
        ("CA 1D Rule 90 (Fractal)", |r| gen_elementary_ca(90, r.next_u64(), 180), true),
        ("CA 1D Rule 30 (Chaos)",   |r| gen_elementary_ca(30, r.next_u64(), 180), false),
        ("Fibonacci mod 11",        |r| gen_fibonacci_mod(11, r.next_u64(), 180), true),
        ("Collatz mod 8",           |r| gen_collatz_trace(r.next_u64(), 180), true),
        ("Thue-Morse (blocs 3b)",   |r| gen_thue_morse_blocks(r.next_u64() % 1024, 180), true),
        ("Trames protocole/log",    |r| gen_structured_frames(r.next_u64(), 180), true),
    ];

    let mut families = Vec::with_capacity(specs.len());
    let mut passed_count = 0usize;
    let mut total_structured = 0usize;
    let mut sum_phi_s = 0.0_f64;
    let mut sum_p_s = 0.0_f64;
    let mut sum_cut_s = 0.0_f64;
    let mut sum_abs_s = 0.0_f64;
    let mut chaos_phi = 0.0_f64;
    let snaps = [DEFAULT_BUDGET, 10];

    for (name, sampler, expect_structure) in specs {
        let trials = 8usize;
        let mut phi_real = 0.0_f64;
        let mut phi_perm = 0.0_f64;
        let mut g_sum = 0.0_f64;
        let mut d_sum = 0.0_f64;
        let mut p_sum = 0.0_f64;
        let mut cut_sum = 0.0_f64;
        let mut abs_sum = 0.0_f64;
        let mut phi_noref = 0.0_f64;
        let mut phi_t10 = 0.0_f64;

        for _ in 0..trials {
            let bytes = sampler(&mut rng);
            let view = RelationalView::from_bytes(&bytes);
            let tr = model.run_abduction(&view, 10, true, &snaps, false);
            let r6 = model.evaluate_phi_state(&view, &tr.snapshots[0].1, DEFAULT_BUDGET, true);
            let r10 = model.evaluate_phi_state(&view, &tr.snapshots[1].1, 10, true);
            let rn = model.evaluate_phi(&view, DEFAULT_BUDGET, false);

            let mut perm_bytes = bytes.clone();
            rng.shuffle(&mut perm_bytes);
            let vp = RelationalView::from_bytes(&perm_bytes);
            let rp = model.evaluate_phi(&vp, DEFAULT_BUDGET, true);

            phi_real += r6.phi_sharp;
            phi_perm += rp.phi_sharp;
            g_sum += r6.generativity;
            d_sum += r6.discrimination;
            p_sum += r6.projection;
            cut_sum += r6.phi_cut;
            abs_sum += r6.s_abs;
            phi_noref += rn.phi_sharp;
            phi_t10 += r10.phi_sharp;
        }

        let n = trials as f64;
        phi_real /= n;
        phi_perm /= n;
        g_sum /= n;
        d_sum /= n;
        p_sum /= n;
        cut_sum /= n;
        abs_sum /= n;
        phi_noref /= n;
        phi_t10 /= n;

        let ret = if phi_real > 1e-6_f64 {
            phi_t10 / phi_real
        } else {
            0.0_f64
        };
        let passed = expect_structure
            && phi_real > 0.025_f64
            && phi_real > phi_perm + 0.02_f64
            && ret >= 0.75_f64;

        if expect_structure {
            total_structured += 1;
            sum_phi_s += phi_real;
            sum_p_s += p_sum;
            sum_cut_s += cut_sum;
            sum_abs_s += abs_sum;
            if passed {
                passed_count += 1;
            }
        } else {
            chaos_phi = phi_real;
        }

        families.push(TransferFamilyStats {
            name,
            phi_real,
            phi_perm,
            g: g_sum,
            d: d_sum,
            p: p_sum,
            phi_cut: cut_sum,
            s_abs: abs_sum,
            no_ref: phi_noref,
            retention: ret,
        });
    }

    let denom = total_structured.max(1) as f64;
    TransferSummary {
        families,
        passed_count,
        total_structured,
        mean_phi_structured: sum_phi_s / denom,
        mean_p_structured: sum_p_s / denom,
        mean_cut_structured: sum_cut_s / denom,
        mean_sabs_structured: sum_abs_s / denom,
        chaos_phi,
    }
}

pub(crate) fn run_eval_transfer(model: &PhisharpV3, seed: u64) {
    let tr = evaluate_transfer(model, seed);
    println!("=== TEST DE RUPTURE DE GÉNÉRATEUR V3 (HORS-PTGRS : CA 1D & ALGORITHMIQUE) ===");
    println!(
        "{:<24} | {:>7} | {:>7} | {:>5} {:>5} {:>5} {:>5} {:>5} | {:>7} | {:>6}",
        "Générateur hors-PTGRS", "φ_réel", "φ_perm", "G", "D", "P", "Φ_cut", "S_abs", "no_ref", "t10/t6"
    );
    println!("{}", "-".repeat(98));
    for f in &tr.families {
        println!(
            "{:<24} | {:>7.4} | {:>7.4} | {:>5.3} {:>5.3} {:>5.3} {:>5.3} {:>5.3} | {:>7.4} | {:>5.2}x",
            f.name, f.phi_real, f.phi_perm, f.g, f.d, f.p, f.phi_cut, f.s_abs, f.no_ref, f.retention
        );
    }
    println!("{}", "-".repeat(98));
    println!(
        "VERDICT RUPTURE DE GÉNÉRATEUR V3 : {}/{} familles structurées transférées (Rule 30 = témoin chaotique).",
        tr.passed_count, tr.total_structured
    );
}

// ============================================================================
// 5. Comparaison contrôlée 4 voies (Step 0 vs LL vs MDL vs Phisharp V3)
// ============================================================================

pub(crate) fn run_compare(
    models: &[(&str, &PhisharpV3)],
    seed_cal: u64,
    seed_ood: u64,
    seed_tr: u64,
) {
    let mut cals = Vec::with_capacity(models.len());
    let mut oods = Vec::with_capacity(models.len());
    let mut trs = Vec::with_capacity(models.len());

    for (_, m) in models {
        cals.push(evaluate_calibration(m, seed_cal));
        oods.push(evaluate_ood(m, seed_ood));
        trs.push(evaluate_transfer(m, seed_tr));
    }

    println!();
    println!("=============================================================================================");
    println!("=== COMPARAISON CONTRÔLÉE V3 : Step 0 vs Vraisemblance (LL) vs MDL vs Phisharp V3         ===");
    println!("=== (Circulation pure L0-L1-L2 sans fast-weights, seeds éval {}/{}/{})             ===", seed_cal, seed_ood, seed_tr);
    println!("=============================================================================================");

    println!();
    println!("[1/3] PORTE D'ÉTALONNAGE V3 (PTGRS générique, C connue après coup, seed={})", seed_cal);
    println!(
        "{:<16} | {:>8} | {:>9} | {:>7} {:>7} | {:>7} | {:>7} | {:>7} | {:>6}",
        "Modèle", "φ_moy", "Rec_exc(C)", "Pear(φ)", "Spear(φ)", "E_perm", "Epiph", "Δφ_bij", "Verdict"
    );
    println!("{}", "-".repeat(94));
    for (i, (label, _)) in models.iter().enumerate() {
        let c = &cals[i];
        let epiph_max = c.phi_l1_dead.max(c.phi_l1_clone).max(c.phi_l2_clone);
        let verdict = if c.pass_corr && c.pass_neg && c.pass_inv {
            "PASS"
        } else {
            "ARRÊT"
        };
        println!(
            "{:<16} | {:>8.4} | {:>+9.4} | {:>+7.4} {:>+7.4} | {:>7.4} | {:>7.4} | {:>7.1e} | {:>6}",
            label, c.mean_phi, c.mean_eq, c.r_phi, c.s_phi, c.perm_max, epiph_max, c.max_phi_diff, verdict
        );
    }

    println!();
    println!("[2/3] BARREAU 1 : PTGRS HORS-DISTRIBUTION (d=2, 4, 8, seed={})", seed_ood);
    println!(
        "{:<16} | {:>4} | {:>7} {:>7} {:>6} | {:>7} | {:>5} {:>5} {:>5} {:>5} | {:>9} {:>9}",
        "Modèle", "Prof", "φ(t=6)", "φ(t=10)", "t10/t6", "no_ref", "G(t6)", "P(t6)", "Φ_cut", "S_abs", "Rec(C)t6", "Rec(C)t10"
    );
    println!("{}", "-".repeat(106));
    for depth_idx in 0..3 {
        for (i, (label, _)) in models.iter().enumerate() {
            let d = &oods[i].depths[depth_idx];
            let ret = if d.curves[DEFAULT_BUDGET] > 1e-6_f64 {
                d.curves[10] / d.curves[DEFAULT_BUDGET]
            } else {
                0.0_f64
            };
            println!(
                "{:<16} | d={:<2} | {:>7.4} {:>7.4} {:>5.2}x | {:>7.4} | {:>5.3} {:>5.3} {:>5.3} {:>5.3} | {:>+9.4} {:>+9.4}",
                label,
                d.depth,
                d.curves[DEFAULT_BUDGET],
                d.curves[10],
                ret,
                d.no_ref_t6,
                d.g_t6,
                d.p_t6,
                d.phi_cut_t6,
                d.s_abs_t6,
                d.rec_exc_t6,
                d.rec_exc_t10
            );
        }
        println!("{}", "-".repeat(106));
    }

    println!("  Résidu OOD (Δφ_sharp moyen, d=4 — test anti-écho) :");
    for (i, (label, _)) in models.iter().enumerate() {
        let o = &oods[i];
        println!(
            "    {:<16} : vrai={:+.4} | étranger C'={:+.4} | écho={:+.4}",
            label, o.gain_true, o.gain_foreign, o.gain_echo
        );
    }

    println!();
    println!("[3/3] RUPTURE DE GÉNÉRATEUR HORS-PTGRS (φ_réel / P_réel à t=6, seed={})", seed_tr);
    print!("{:<24}", "Générateur hors-PTGRS");
    for (label, _) in models {
        print!(" | {:>15}", label);
    }
    println!();
    print!("{:<24}", "");
    for _ in models {
        print!(" | {:>7} {:>7}", "φ_réel", "P_réel");
    }
    println!();
    println!("{}", "-".repeat(24 + models.len() * 18));

    let n_fam = trs[0].families.len();
    for f_idx in 0..n_fam {
        print!("{:<24}", trs[0].families[f_idx].name);
        for tr in &trs {
            let f = &tr.families[f_idx];
            print!(" | {:>7.4} {:>7.3}", f.phi_real, f.p);
        }
        println!();
    }
    println!("{}", "-".repeat(24 + models.len() * 18));
    print!("{:<24}", "Moy 7 structurées");
    for tr in &trs {
        print!(" | {:>7.4} {:>7.3}", tr.mean_phi_structured, tr.mean_p_structured);
    }
    println!();
    print!("{:<24}", "Φ_cut / S_abs moy (7x)");
    for tr in &trs {
        print!(" | {:>7.3} {:>7.3}", tr.mean_cut_structured, tr.mean_sabs_structured);
    }
    println!();
    print!("{:<24}", "Rule 30 (chaos, cible=0)");
    for tr in &trs {
        print!(" | {:>15.4}", tr.chaos_phi);
    }
    println!();
    print!("{:<24}", "Familles transférées /7");
    for tr in &trs {
        let s = format!("{}/{}", tr.passed_count, tr.total_structured);
        print!(" | {:>15}", s);
    }
    println!();

    if models.len() == 4 {
        let ood_rec_mean = |idx: usize| -> f64 {
            oods[idx].depths.iter().map(|d| d.rec_exc_t6).sum::<f64>() / 3.0_f64
        };
        println!();
        println!("--- SYNTHÈSE CONTRÔLE SCEPTIQUE V3 ---");
        println!(
            "  OOD Rec_exc(C) moy (externe) : Step0={:+.4} | LL={:+.4} | MDL={:+.4} | V3 φ={:+.4}",
            ood_rec_mean(0),
            ood_rec_mean(1),
            ood_rec_mean(2),
            ood_rec_mean(3)
        );
        println!(
            "  Transfert P_réel moy (7x)    : Step0={:.3}   | LL={:.3}   | MDL={:.3}   | V3 φ={:.3}",
            trs[0].mean_p_structured,
            trs[1].mean_p_structured,
            trs[2].mean_p_structured,
            trs[3].mean_p_structured
        );
        println!(
            "  Transfert φ_réel moy (7x)    : Step0={:.4}  | LL={:.4}  | MDL={:.4}  | V3 φ={:.4}",
            trs[0].mean_phi_structured,
            trs[1].mean_phi_structured,
            trs[2].mean_phi_structured,
            trs[3].mean_phi_structured
        );
    }
}
