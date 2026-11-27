use std::path::Path;
use std::thread;

use crate::autograd::Tape;
use crate::core::*;
use crate::model::*;

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
            Self::PhiSharp => "Phisharp V4 (φ_sharp)",
        }
    }
}

fn combine_two_budgets(
    t: &mut Tape,
    score1: usize,
    score2: usize,
    mon1: usize,
    mon2: usize,
) -> usize {
    let zero = t.leaf_scalar(0.0_f64);
    let diff = t.sub(mon1, mon2);
    let drop_pen = t.max_scalar(diff, zero);
    let m12 = t.min_scalar(score1, score2);
    let w1 = t.scale(score1, 0.45_f64);
    let w2 = t.scale(score2, 0.45_f64);
    let s = t.add(w1, w2);
    let wm = t.scale(m12, 0.10_f64);
    let s2 = t.add(s, wm);
    let wp = t.scale(drop_pen, 0.20_f64);
    t.sub(s2, wp)
}

fn sample_objective_and_grad(
    model: &PhisharpV4,
    view: &RelationalView,
    t_budget: usize,
    t_next: usize,
    objective: TrainObjective,
    need_grad: bool,
) -> (f64, Vec<f64>) {
    let mut t = Tape::new(need_grad);
    let p = ParamsTape::from_model(&mut t, model);
    let snaps = [t_budget, t_next];

    let (_, captured) = PhisharpV4::run_abduction_tape(
        &mut t,
        &p,
        view,
        &view.obs,
        t_next,
        true,
        [true, true, true],
        &snaps,
    );
    let st1 = &captured[0].1;
    let st2 = &captured[1].1;

    let score_node = match objective {
        TrainObjective::PhiSharp => {
            let mid = view.seq.len() / 2;
            let first_half = view.subslice_same_vocab(0, mid);
            let (_, cap_pt) = PhisharpV4::run_abduction_tape(
                &mut t,
                &p,
                &first_half,
                &first_half.obs,
                t_next,
                true,
                [true, true, true],
                &snaps,
            );
            let (_, cap_pc) = PhisharpV4::run_abduction_tape(
                &mut t,
                &p,
                view,
                &view.obs,
                t_next,
                false,
                [true, true, false],
                &snaps,
            );

            let r1 = model.evaluate_phi_tape(
                &mut t,
                &p,
                view,
                st1,
                t_budget,
                true,
                Some(&cap_pt[0].1),
                Some(&cap_pc[0].1),
            );
            let r2 = model.evaluate_phi_tape(
                &mut t,
                &p,
                view,
                st2,
                t_next,
                true,
                Some(&cap_pt[1].1),
                Some(&cap_pc[1].1),
            );
            combine_two_budgets(
                &mut t,
                r1.shaping_score,
                r2.shaping_score,
                r1.phi_sharp,
                r2.phi_sharp,
            )
        }
        TrainObjective::Likelihood => {
            let (_, e1) = PhisharpV4::unfold_tables(
                &mut t,
                &p,
                view,
                &view.obs,
                st1,
                None,
                false,
                [true, true, true],
            );
            let (_, e2) = PhisharpV4::unfold_tables(
                &mut t,
                &p,
                view,
                &view.obs,
                st2,
                None,
                false,
                [true, true, true],
            );
            combine_two_budgets(&mut t, e1.ll_mix, e2.ll_mix, e1.ll_mix, e2.ll_mix)
        }
        TrainObjective::Mdl => {
            let (_, e1) = PhisharpV4::unfold_tables(
                &mut t,
                &p,
                view,
                &view.obs,
                st1,
                None,
                false,
                [true, true, true],
            );
            let (_, e2) = PhisharpV4::unfold_tables(
                &mut t,
                &p,
                view,
                &view.obs,
                st2,
                None,
                false,
                [true, true, true],
            );
            let s1 = mdl_tape_v4(&mut t, e1.ll_mix, st1, view.seq.len());
            let s2 = mdl_tape_v4(&mut t, e2.ll_mix, st2, view.seq.len());
            combine_two_budgets(&mut t, s1, s2, s1, s2)
        }
        TrainObjective::Untrained => t.leaf_scalar(0.0_f64),
    };

    let val = t.scalar(score_node);
    let grad = if need_grad {
        t.backward(score_node, p.root)
    } else {
        Vec::new()
    };
    (val, grad)
}

pub(crate) fn run_grad_check(seed: u64) {
    let mut rng = Rng::new(seed);
    let model = PhisharpV4::new_random(&mut rng);
    let cause = PtgrsCause::sample_generic(&mut rng);
    let bytes = cause.emit_effect(&mut rng, 4, 140);
    let view = RelationalView::from_bytes(&bytes);
    let eps = 1e-6_f64;

    println!("=== VÉRIFICATION AUTOGRAD V4 (différences finies centrées, 16 coords) ===");
    for obj in [
        TrainObjective::Likelihood,
        TrainObjective::Mdl,
        TrainObjective::PhiSharp,
    ] {
        let (_, grad) = sample_objective_and_grad(&model, &view, 3, 5, obj, true);
        let mut ag_vec = Vec::with_capacity(16);
        let mut fd_vec = Vec::with_capacity(16);
        let mut max_err = 0.0_f64;

        for _ in 0..16 {
            let idx = rng.range_usize(PARAM_COUNT);
            let mut mp = model.clone();
            let mut mm = model.clone();
            mp.params[idx] += eps;
            mm.params[idx] -= eps;
            let (vp, _) = sample_objective_and_grad(&mp, &view, 3, 5, obj, false);
            let (vm, _) = sample_objective_and_grad(&mm, &view, 3, 5, obj, false);
            let fd = (vp - vm) / (2.0_f64 * eps);
            let ag = grad[idx];
            ag_vec.push(ag);
            fd_vec.push(fd);
            max_err = max_err.max((ag - fd).abs());
        }

        let num: f64 = ag_vec.iter().zip(fd_vec.iter()).map(|(a, b)| a * b).sum();
        let na: f64 = ag_vec.iter().map(|a| a * a).sum::<f64>().sqrt();
        let nb: f64 = fd_vec.iter().map(|b| b * b).sum::<f64>().sqrt();
        let cos = if na * nb > 1e-18_f64 {
            num / (na * nb)
        } else {
            1.0_f64
        };
        println!(
            "  {:<28} | cos(AG, FD) = {:+.6} | max|AG-FD| = {:.2e} | ||AG|| = {:.2e}",
            obj.label(),
            cos,
            max_err,
            na
        );
    }
}

pub(crate) fn train_model(
    out_path: &Path,
    steps: usize,
    seed: u64,
    objective: TrainObjective,
) -> PhisharpV4 {
    let mut rng = Rng::new(seed);
    let mut model = PhisharpV4::new_random(&mut rng);

    if objective == TrainObjective::Untrained || steps == 0 {
        model.save(out_path).expect("Échec écriture Step 0");
        println!(
            "Modèle {} (0 steps, seed={}) sauvegardé : {}",
            objective.label(),
            seed,
            out_path.display()
        );
        return model;
    }

    let batch_size = 6usize;
    let lr: f64 = 0.012;
    let beta1: f64 = 0.9;
    let beta2: f64 = 0.999;
    let mut m_adam = vec![0.0_f64; PARAM_COUNT];
    let mut v_adam = vec![0.0_f64; PARAM_COUNT];

    println!(
        "=== ENTRAÎNEMENT FERMÉ V4 [{}] (Adam + autograd exact, {} params) ===",
        objective.label(),
        PARAM_COUNT
    );

    for step in 1..=steps {
        let t_budget = 3 + rng.range_usize(4);
        let t_next = t_budget + 2;

        let mut batch_views = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let cause = PtgrsCause::sample_generic(&mut rng);
            let depth = 3 + rng.range_usize(3);
            let bytes = cause.emit_effect(&mut rng, depth, 160);
            batch_views.push(RelationalView::from_bytes(&bytes));
        }

        let model_ref = &model;
        let views_ref = &batch_views;
        let mut grad_acc = vec![0.0_f64; PARAM_COUNT];

        thread::scope(|s| {
            let mut handles = Vec::with_capacity(batch_size);
            for view in views_ref {
                handles.push(s.spawn(move || {
                    sample_objective_and_grad(model_ref, view, t_budget, t_next, objective, true)
                }));
            }
            for h in handles {
                let (_, g) = h.join().unwrap();
                for i in 0..PARAM_COUNT {
                    grad_acc[i] += g[i];
                }
            }
        });

        let inv_b = 1.0_f64 / (batch_size as f64);
        let mut norm_sq = 0.0_f64;
        for g in &mut grad_acc {
            *g *= inv_b;
            norm_sq += (*g) * (*g);
        }
        let g_norm = norm_sq.sqrt();
        if g_norm > 5.0_f64 {
            let sc = 5.0_f64 / g_norm;
            for g in &mut grad_acc {
                *g *= sc;
            }
        }

        let b1_corr = 1.0_f64 - beta1.powi(step as i32);
        let b2_corr = 1.0_f64 - beta2.powi(step as i32);
        for i in 0..PARAM_COUNT {
            m_adam[i] = beta1 * m_adam[i] + (1.0_f64 - beta1) * grad_acc[i];
            v_adam[i] = beta2 * v_adam[i] + (1.0_f64 - beta2) * grad_acc[i] * grad_acc[i];
            let m_hat = m_adam[i] / b1_corr;
            let v_hat = v_adam[i] / b2_corr;
            model.params[i] += lr * m_hat / (v_hat.sqrt() + 1e-8_f64);
            model.params[i] = model.params[i].clamp(-4.0_f64, 4.0_f64);
        }

        if step == 1 || step % 25 == 0 || step == steps {
            let mut g_s = 0.0_f64;
            let mut d_s = 0.0_f64;
            let mut pt_s = 0.0_f64;
            let mut pc_s = 0.0_f64;
            let mut cut_s = 0.0_f64;
            let mut sr_s = 0.0_f64;
            let mut sd_s = 0.0_f64;
            let mut sa_s = 0.0_f64;
            let mut phi_t = 0.0_f64;
            let mut phi_t2 = 0.0_f64;
            let snaps = [t_budget, t_next];

            for view in &batch_views {
                let (_, caps) = model.run_abduction_raw(view, t_next, true, &snaps);
                let r1 = model.evaluate_phi_raw(view, &caps[0].1, t_budget, true);
                let r2 = model.evaluate_phi_raw(view, &caps[1].1, t_next, true);
                g_s += r1.generativity;
                d_s += r1.discrimination;
                pt_s += r1.proj_temp;
                pc_s += r1.proj_comp;
                cut_s += r1.phi_cut;
                sr_s += r1.s_ref;
                sd_s += r1.s_disp;
                sa_s += r1.s_abs;
                phi_t += r1.phi_sharp;
                phi_t2 += r2.phi_sharp;
            }
            let nb = batch_views.len() as f64;
            println!(
                "step {:4}/{} (T={},{}) | φ={:.4}->{:.4} | G={:.2} D={:.2} Pt={:.2} Pc={:.2} | Φc={:.2} R/D/A={:.2}/{:.2}/{:.2} |g|={:.2}",
                step,
                steps,
                t_budget,
                t_next,
                phi_t / nb,
                phi_t2 / nb,
                g_s / nb,
                d_s / nb,
                pt_s / nb,
                pc_s / nb,
                cut_s / nb,
                sr_s / nb,
                sd_s / nb,
                sa_s / nb,
                g_norm
            );
        }
    }

    model.save(out_path).expect("Échec écriture modèle V4");
    println!(
        "Modèle [{}] figé sauvegardé : {}",
        objective.label(),
        out_path.display()
    );
    model
}

pub(crate) struct CalibrationSummary {
    pub(crate) r_phi: f64,
    pub(crate) r_ll: f64,
    pub(crate) r_mdl: f64,
    pub(crate) s_phi: f64,
    pub(crate) s_ll: f64,
    pub(crate) s_mdl: f64,
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

pub(crate) fn evaluate_calibration(model: &PhisharpV4, seed: u64) -> CalibrationSummary {
    let mut rng = Rng::new(seed);
    let mut eval_rng = Rng::new(seed ^ 0x55AA_CC33);

    let mut eq_scores = Vec::new();
    let mut phi_scores = Vec::new();
    let mut ll_scores = Vec::new();
    let mut mdl_scores = Vec::new();

    for trial in 0..30 {
        let cause = PtgrsCause::sample_generic(&mut rng);
        let depth = 3 + (trial % 3);
        let len = match trial % 3 {
            0 => 110,
            1 => 150,
            _ => 190,
        };
        let bytes = cause.emit_effect(&mut rng, depth, len);
        let view = RelationalView::from_bytes(&bytes);
        let truth = cause.ground_truth_non_markov(&mut eval_rng, depth);

        let budget = match trial % 5 {
            0 => 0,
            1 => 1,
            2 => 3,
            3 => 5,
            _ => 7,
        };
        let use_ref = (trial % 4) != 1;
        let (raw, _) = model.run_abduction_raw(&view, budget, use_ref, &[]);
        let rep = model.evaluate_phi_raw(&view, &raw, budget, use_ref);
        let (preds, ll, mdl, _) = model.unfold_predictions_raw(&view, &raw, false);
        let eq = observational_equivalence_nm(&view, &preds, &truth);

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

    let mut rand_phi = 0.0_f64;
    for _ in 0..8 {
        let bytes: Vec<u8> = (0..160).map(|_| (rng.next_u64() % 8) as u8).collect();
        let view = RelationalView::from_bytes(&bytes);
        rand_phi += model.evaluate_phi(&view, DEFAULT_BUDGET, true).phi_sharp;
    }
    rand_phi /= 8.0_f64;

    let cause_ref = PtgrsCause::sample_generic(&mut rng);
    let bytes_ref = cause_ref.emit_effect(&mut rng, 4, 160);
    let view_ref = RelationalView::from_bytes(&bytes_ref);
    let h_eq_e_phi = model
        .evaluate_phi_raw(&view_ref, &RawState::zeros(0, 0), DEFAULT_BUDGET, true)
        .phi_sharp;
    let overgen_phi = model
        .evaluate_phi_raw(&view_ref, &RawState::zeros(K1, K2), DEFAULT_BUDGET, true)
        .phi_sharp;

    let mut perm_sum = 0.0_f64;
    let mut perm_max = 0.0_f64;
    for _ in 0..8 {
        let c = PtgrsCause::sample_generic(&mut rng);
        let mut b = c.emit_effect(&mut rng, 4, 160);
        rng.shuffle(&mut b);
        let vp = RelationalView::from_bytes(&b);
        let p = model.evaluate_phi(&vp, DEFAULT_BUDGET, true).phi_sharp;
        perm_sum += p;
        perm_max = perm_max.max(p);
    }
    let perm_mean = perm_sum / 8.0_f64;

    let (raw_ref, _) = model.run_abduction_raw(&view_ref, DEFAULT_BUDGET, true, &[]);
    let phi_clean = model
        .evaluate_phi_raw(&view_ref, &raw_ref, DEFAULT_BUDGET, true)
        .phi_sharp;
    let phi_l1_dead = model
        .evaluate_phi_raw(&view_ref, &raw_ref.with_dead_l1(), DEFAULT_BUDGET, true)
        .phi_sharp;
    let phi_l1_clone = model
        .evaluate_phi_raw(&view_ref, &raw_ref.with_cloned_l1(0), DEFAULT_BUDGET, true)
        .phi_sharp;
    let phi_l2_clone = model
        .evaluate_phi_raw(&view_ref, &raw_ref.with_cloned_l2(0), DEFAULT_BUDGET, true)
        .phi_sharp;

    let mut bij: Vec<u8> = (0u8..=255).collect();
    rng.shuffle(&mut bij);
    let mut inv_bij = [0u8; 256];
    for (orig, &mapped) in bij.iter().enumerate() {
        inv_bij[mapped as usize] = orig as u8;
    }
    let renamed: Vec<u8> = bytes_ref.iter().map(|&b| bij[b as usize]).collect();
    let view_ren = RelationalView::from_bytes(&renamed);

    let snaps_bij = [0usize, 2, 4, DEFAULT_BUDGET];
    let (_, s_orig) = model.run_abduction_raw(&view_ref, DEFAULT_BUDGET, true, &snaps_bij);
    let (_, s_ren) = model.run_abduction_raw(&view_ren, DEFAULT_BUDGET, true, &snaps_bij);
    let mut max_phi_diff = 0.0_f64;
    for ((b, a), (_, c)) in s_orig.iter().zip(s_ren.iter()) {
        let p1 = model.evaluate_phi_raw(&view_ref, a, *b, true).phi_sharp;
        let p2 = model.evaluate_phi_raw(&view_ren, c, *b, true).phi_sharp;
        max_phi_diff = max_phi_diff.max((p1 - p2).abs());
    }
    let out_orig = model.ask_emit(&view_ref, DEFAULT_BUDGET, 24);
    let out_ren = model.ask_emit(&view_ren, DEFAULT_BUDGET, 24);
    let out_inv: Vec<u8> = out_ren.iter().map(|&b| inv_bij[b as usize]).collect();
    let ask_exact = out_orig == out_inv;

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

pub(crate) fn run_calibration(model: &PhisharpV4, seed: u64) {
    let c = evaluate_calibration(model, seed);
    println!("=== PORTE D'ÉTALONNAGE V4 (PTGRS non-markovien, C connue après coup) ===");
    println!(
        "Pearson  vs Rec_nm(C) : φ_sharp={:+.4} | LL={:+.4} | MDL={:+.4}",
        c.r_phi, c.r_ll, c.r_mdl
    );
    println!(
        "Spearman vs Rec_nm(C) : φ_sharp={:+.4} | LL={:+.4} | MDL={:+.4}",
        c.s_phi, c.s_ll, c.s_mdl
    );
    println!(
        "Moyennes PTGRS        : φ_sharp={:.4} | Rec_nm(C)={:+.4}",
        c.mean_phi, c.mean_eq
    );
    println!(
        "Témoins négatifs      : rand={:.4} | H=E={:.4} | overgen={:.4} | E_perm={:.4}(max {:.4})",
        c.rand_phi, c.h_eq_e_phi, c.overgen_phi, c.perm_mean, c.perm_max
    );
    println!(
        "Épiphénoménal         : propre={:.4} -> L1_morte={:.4} | L1_clone={:.4} | L2_clone={:.4}",
        c.phi_clean, c.phi_l1_dead, c.phi_l1_clone, c.phi_l2_clone
    );
    println!(
        "Invariance bijective  : max|Δφ(t)|={:.2e} | ask(π(E))==π(ask(E)) : {}",
        c.max_phi_diff, c.ask_exact
    );
    if c.pass_corr && c.pass_neg && c.pass_inv {
        println!("VERDICT V4 : PASS (A est lisible).");
    } else {
        println!(
            "VERDICT V4 : ARRÊT (alive={}, corr={}, neg={}, inv={}).",
            c.signal_alive, c.pass_corr, c.pass_neg, c.pass_inv
        );
    }
}

pub(crate) struct OodDepthStats {
    pub(crate) depth: usize,
    pub(crate) phi_t6: f64,
    pub(crate) phi_t10: f64,
    pub(crate) no_ref_t6: f64,
    pub(crate) rec_t6: f64,
    pub(crate) rec_t10: f64,
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

pub(crate) fn evaluate_ood(model: &PhisharpV4, seed: u64) -> OodSummary {
    let mut rng = Rng::new(seed);
    let mut truth_rng = Rng::new(seed ^ 0x7711_9933);
    let snaps = [DEFAULT_BUDGET, 10];
    let mut depths = Vec::new();

    for &depth in &[2usize, 4, 8] {
        let trials = 10usize;
        let (mut p6, mut p10, mut nr, mut r6, mut r10) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut g6, mut pr6, mut c6, mut a6) = (0.0, 0.0, 0.0, 0.0);

        for _ in 0..trials {
            let cause = PtgrsCause::sample_ood(&mut rng, depth);
            let bytes = cause.emit_effect(&mut rng, depth, 180);
            let view = RelationalView::from_bytes(&bytes);
            let truth = cause.ground_truth_non_markov(&mut truth_rng, depth);

            let (_, caps) = model.run_abduction_raw(&view, 10, true, &snaps);
            let rep6 = model.evaluate_phi_raw(&view, &caps[0].1, DEFAULT_BUDGET, true);
            let rep10 = model.evaluate_phi_raw(&view, &caps[1].1, 10, true);
            let rep_nr = model.evaluate_phi(&view, DEFAULT_BUDGET, false);

            let (pr_6, _, _, _) = model.unfold_predictions_raw(&view, &caps[0].1, false);
            let (pr_10, _, _, _) = model.unfold_predictions_raw(&view, &caps[1].1, false);

            p6 += rep6.phi_sharp;
            p10 += rep10.phi_sharp;
            nr += rep_nr.phi_sharp;
            r6 += observational_equivalence_nm(&view, &pr_6, &truth);
            r10 += observational_equivalence_nm(&view, &pr_10, &truth);
            g6 += rep6.generativity;
            pr6 += rep6.projection;
            c6 += rep6.phi_cut;
            a6 += rep6.s_abs;
        }
        let n = trials as f64;
        depths.push(OodDepthStats {
            depth,
            phi_t6: p6 / n,
            phi_t10: p10 / n,
            no_ref_t6: nr / n,
            rec_t6: r6 / n,
            rec_t10: r10 / n,
            g_t6: g6 / n,
            p_t6: pr6 / n,
            phi_cut_t6: c6 / n,
            s_abs_t6: a6 / n,
        });
    }

    let (mut gt, mut gf, mut ge) = (0.0_f64, 0.0_f64, 0.0_f64);
    let tr_n = 12usize;
    for _ in 0..tr_n {
        let c1 = PtgrsCause::sample_ood(&mut rng, 4);
        let c2 = PtgrsCause::sample_ood(&mut rng, 4);
        let full = c1.emit_effect(&mut rng, 4, 180);
        let foreign = c2.emit_effect(&mut rng, 4, 60);
        let part = &full[0..120];

        let mut wt = part.to_vec();
        wt.extend_from_slice(&full[120..180]);
        let mut wf = part.to_vec();
        wf.extend_from_slice(&foreign);
        let mut we = part.to_vec();
        we.extend_from_slice(&part[0..60]);

        let p0 = model
            .evaluate_phi(&RelationalView::from_bytes(part), DEFAULT_BUDGET, true)
            .phi_sharp;
        gt += model
            .evaluate_phi(&RelationalView::from_bytes(&wt), DEFAULT_BUDGET, true)
            .phi_sharp
            - p0;
        gf += model
            .evaluate_phi(&RelationalView::from_bytes(&wf), DEFAULT_BUDGET, true)
            .phi_sharp
            - p0;
        ge += model
            .evaluate_phi(&RelationalView::from_bytes(&we), DEFAULT_BUDGET, true)
            .phi_sharp
            - p0;
    }

    OodSummary {
        depths,
        gain_true: gt / (tr_n as f64),
        gain_foreign: gf / (tr_n as f64),
        gain_echo: ge / (tr_n as f64),
    }
}

fn gen_ca(rule: u8, seed: u64, len: usize) -> Vec<u8> {
    let w = 12usize;
    let mut cells = vec![0u8; w];
    let mut rng = Rng::new(seed);
    for c in &mut cells {
        *c = (rng.next_u64() & 1) as u8;
    }
    if cells.iter().all(|&x| x == 0) {
        cells[w / 2] = 1;
    }
    for _ in 0..6 {
        let mut nxt = vec![0u8; w];
        for i in 0..w {
            let pat = (cells[(i + w - 1) % w] << 2) | (cells[i] << 1) | cells[(i + 1) % w];
            nxt[i] = (rule >> pat) & 1;
        }
        cells = nxt;
    }
    let mut out = Vec::with_capacity(len + 4);
    while out.len() < len {
        for blk in 0..(w / 3) {
            out.push((cells[3 * blk] << 2) | (cells[3 * blk + 1] << 1) | cells[3 * blk + 2]);
        }
        let mut nxt = vec![0u8; w];
        for i in 0..w {
            let pat = (cells[(i + w - 1) % w] << 2) | (cells[i] << 1) | cells[(i + 1) % w];
            nxt[i] = (rule >> pat) & 1;
        }
        cells = nxt;
    }
    out.truncate(len);
    out
}

fn gen_fib(modulus: u8, seed: u64, len: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut a = (1 + (rng.next_u64() as u8) % (modulus - 1)) % modulus;
    let mut b = (1 + (rng.next_u64() as u8) % (modulus - 1)) % modulus;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(a);
        let c = (a + b) % modulus;
        a = b;
        b = c;
    }
    out
}

fn gen_collatz(seed: u64, len: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut n = 27u64 + (rng.next_u64() % 500);
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        out.push((n % 8) as u8);
        n = if n <= 1 {
            27u64 + (rng.next_u64() % 500)
        } else if n % 2 == 0 {
            n / 2
        } else {
            3 * n + 1
        };
    }
    out
}

fn gen_thue(offset: u64, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| {
            let idx = offset + (i as u64);
            let b0 = (idx.count_ones() & 1) as u8;
            let b1 = ((idx + 1).count_ones() & 1) as u8;
            let b2 = ((idx + 2).count_ones() & 1) as u8;
            (b0 << 2) | (b1 << 1) | b2
        })
        .collect()
}

fn gen_frames(seed: u64, len: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut cnt = (rng.next_u64() % 5) as u8;
    let mut out = Vec::with_capacity(len + 4);
    while out.len() < len {
        let f = (cnt % 3) + 1;
        let v = (cnt + f) % 5;
        let chk = (f + v + 2) % 5;
        out.extend_from_slice(&[9u8, 5u8 + f, v, chk]);
        cnt = (cnt + 1) % 5;
    }
    out.truncate(len);
    out
}

fn gen_dyck(seed: u64, len: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut stack: Vec<u8> = Vec::new();
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        if stack.is_empty() || (stack.len() < 4 && rng.next_f64() < 0.55_f64) {
            let k = (rng.next_u64() % 3) as u8;
            out.push(2 * k);
            stack.push(2 * k + 1);
        } else {
            out.push(stack.pop().unwrap());
        }
    }
    out
}

pub(crate) struct TransferFamilyStats {
    pub(crate) name: &'static str,
    pub(crate) phi_real: f64,
    pub(crate) phi_perm: f64,
    pub(crate) p: f64,
    pub(crate) phi_cut: f64,
    pub(crate) s_abs: f64,
    pub(crate) ret: f64,
}

pub(crate) struct TransferSummary {
    pub(crate) families: Vec<TransferFamilyStats>,
    pub(crate) passed: usize,
    pub(crate) total_struct: usize,
    pub(crate) mean_phi: f64,
    pub(crate) mean_p: f64,
    pub(crate) mean_cut: f64,
    pub(crate) mean_sabs: f64,
    pub(crate) chaos_phi: f64,
}

pub(crate) fn evaluate_transfer(model: &PhisharpV4, seed: u64) -> TransferSummary {
    let mut rng = Rng::new(seed);
    let specs: Vec<(&'static str, fn(&mut Rng) -> Vec<u8>, bool)> = vec![
        ("CA 1D Rule 110 (Turing)", |r| gen_ca(110, r.next_u64(), 180), true),
        ("CA 1D Rule 54 (Gliders)", |r| gen_ca(54, r.next_u64(), 180), true),
        ("CA 1D Rule 90 (Fractal)", |r| gen_ca(90, r.next_u64(), 180), true),
        ("CA 1D Rule 30 (Chaos)",   |r| gen_ca(30, r.next_u64(), 180), false),
        ("Fibonacci mod 11",        |r| gen_fib(11, r.next_u64(), 180), true),
        ("Collatz mod 8",           |r| gen_collatz(r.next_u64(), 180), true),
        ("Thue-Morse (blocs 3b)",   |r| gen_thue(r.next_u64() % 1024, 180), true),
        ("Trames protocole/log",    |r| gen_frames(r.next_u64(), 180), true),
        ("Dyck-3 (emboîtement)",    |r| gen_dyck(r.next_u64(), 180), true),
    ];

    let snaps = [DEFAULT_BUDGET, 10];
    let mut families = Vec::new();
    let (mut passed, mut total_struct) = (0usize, 0usize);
    let (mut sp, mut spr, mut sc, mut sa, mut chaos) = (0.0, 0.0, 0.0, 0.0, 0.0);

    for (name, sampler, is_struct) in specs {
        let trials = 6usize;
        let (mut pr, mut pp, mut p_s, mut cut_s, mut abs_s, mut p10) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        for _ in 0..trials {
            let bytes = sampler(&mut rng);
            let view = RelationalView::from_bytes(&bytes);
            let (_, caps) = model.run_abduction_raw(&view, 10, true, &snaps);
            let r6 = model.evaluate_phi_raw(&view, &caps[0].1, DEFAULT_BUDGET, true);
            let r10 = model.evaluate_phi_raw(&view, &caps[1].1, 10, true);

            let mut perm = bytes.clone();
            rng.shuffle(&mut perm);
            let rp = model.evaluate_phi(&RelationalView::from_bytes(&perm), DEFAULT_BUDGET, true);

            pr += r6.phi_sharp;
            pp += rp.phi_sharp;
            p_s += r6.projection;
            cut_s += r6.phi_cut;
            abs_s += r6.s_abs;
            p10 += r10.phi_sharp;
        }
        let n = trials as f64;
        pr /= n;
        pp /= n;
        p_s /= n;
        cut_s /= n;
        abs_s /= n;
        p10 /= n;
        let ret = if pr > 1e-6_f64 { p10 / pr } else { 0.0_f64 };

        if is_struct {
            total_struct += 1;
            sp += pr;
            spr += p_s;
            sc += cut_s;
            sa += abs_s;
            if pr > 0.025_f64 && pr > pp + 0.02_f64 && ret >= 0.75_f64 {
                passed += 1;
            }
        } else {
            chaos = pr;
        }
        families.push(TransferFamilyStats {
            name,
            phi_real: pr,
            phi_perm: pp,
            p: p_s,
            phi_cut: cut_s,
            s_abs: abs_s,
            ret,
        });
    }

    let d = total_struct.max(1) as f64;
    TransferSummary {
        families,
        passed,
        total_struct,
        mean_phi: sp / d,
        mean_p: spr / d,
        mean_cut: sc / d,
        mean_sabs: sa / d,
        chaos_phi: chaos,
    }
}

pub(crate) fn run_compare(
    models: &[(&str, &PhisharpV4)],
    seed_cal: u64,
    seed_ood: u64,
    seed_tr: u64,
) {
    let cals: Vec<_> = models
        .iter()
        .map(|(_, m)| evaluate_calibration(m, seed_cal))
        .collect();
    let oods: Vec<_> = models
        .iter()
        .map(|(_, m)| evaluate_ood(m, seed_ood))
        .collect();
    let trs: Vec<_> = models
        .iter()
        .map(|(_, m)| evaluate_transfer(m, seed_tr))
        .collect();

    println!();
    println!("=============================================================================================");
    println!("=== COMPARAISON V4 (NON-MARKOVIEN + AUTOGRAD ADAM EXACT + REPÈRE L0 FIXE)                 ===");
    println!("=============================================================================================");

    println!("\n[1/3] PORTE D'ÉTALONNAGE V4 (seed={})", seed_cal);
    println!(
        "{:<14} | {:>7} | {:>9} | {:>7} {:>7} | {:>6} {:>6} | {:>7} | {:>6}",
        "Modèle", "φ_moy", "Rec_nm(C)", "Pear(φ)", "Spear(φ)", "E_perm", "Epiph", "Δφ_bij", "Verdict"
    );
    println!("{}", "-".repeat(90));
    for (i, (label, _)) in models.iter().enumerate() {
        let c = &cals[i];
        let ep = c.phi_l1_dead.max(c.phi_l1_clone).max(c.phi_l2_clone);
        let v = if c.pass_corr && c.pass_neg && c.pass_inv {
            "PASS"
        } else {
            "ARRÊT"
        };
        println!(
            "{:<14} | {:>7.4} | {:>+9.4} | {:>+7.4} {:>+7.4} | {:>6.4} {:>6.4} | {:>7.1e} | {:>6}",
            label, c.mean_phi, c.mean_eq, c.r_phi, c.s_phi, c.perm_max, ep, c.max_phi_diff, v
        );
    }

    println!("\n[2/3] BARREAU 1 : PTGRS OOD NON-MARKOVIEN (d=2, 4, 8, seed={})", seed_ood);
    println!(
        "{:<14} | {:>4} | {:>7} {:>7} {:>6} | {:>7} | {:>5} {:>5} {:>5} {:>5} | {:>8} {:>8}",
        "Modèle", "Prof", "φ(t=6)", "φ(t=10)", "t10/t6", "no_ref", "G(t6)", "P(t6)", "Φ_cut", "S_abs", "Rec(t6)", "Rec(t10)"
    );
    println!("{}", "-".repeat(102));
    for d_i in 0..3 {
        for (i, (label, _)) in models.iter().enumerate() {
            let d = &oods[i].depths[d_i];
            let ret = if d.phi_t6 > 1e-6_f64 {
                d.phi_t10 / d.phi_t6
            } else {
                0.0_f64
            };
            println!(
                "{:<14} | d={:<2} | {:>7.4} {:>7.4} {:>5.2}x | {:>7.4} | {:>5.3} {:>5.3} {:>5.3} {:>5.3} | {:>+8.4} {:>+8.4}",
                label, d.depth, d.phi_t6, d.phi_t10, ret, d.no_ref_t6, d.g_t6, d.p_t6, d.phi_cut_t6, d.s_abs_t6, d.rec_t6, d.rec_t10
            );
        }
        println!("{}", "-".repeat(102));
    }
    println!("  Résidu OOD (Δφ_sharp moyen, d=4 — test anti-écho) :");
    for (i, (label, _)) in models.iter().enumerate() {
        let o = &oods[i];
        println!(
            "    {:<14} : vrai={:+.4} | étranger C'={:+.4} | écho={:+.4}",
            label, o.gain_true, o.gain_foreign, o.gain_echo
        );
    }

    println!("\n[3/3] RUPTURE DE GÉNÉRATEUR HORS-PTGRS (φ_réel / P_réel à t=6, seed={})", seed_tr);
    print!("{:<24}", "Générateur hors-PTGRS");
    for (label, _) in models {
        print!(" | {:>15}", label);
    }
    println!();
    println!("{}", "-".repeat(24 + models.len() * 18));
    for f_i in 0..trs[0].families.len() {
        print!("{:<24}", trs[0].families[f_i].name);
        for tr in &trs {
            let f = &tr.families[f_i];
            print!(" | {:>7.4} {:>7.3}", f.phi_real, f.p);
        }
        println!();
    }
    println!("{}", "-".repeat(24 + models.len() * 18));
    print!("{:<24}", "Moy 8 structurées");
    for tr in &trs {
        print!(" | {:>7.4} {:>7.3}", tr.mean_phi, tr.mean_p);
    }
    println!();
    print!("{:<24}", "Φ_cut / S_abs (8x)");
    for tr in &trs {
        print!(" | {:>7.3} {:>7.3}", tr.mean_cut, tr.mean_sabs);
    }
    println!();
    print!("{:<24}", "Rule 30 (chaos, cible=0)");
    for tr in &trs {
        print!(" | {:>15.4}", tr.chaos_phi);
    }
    println!();
    print!("{:<24}", "Familles transférées /8");
    for tr in &trs {
        print!(" | {:>15}", format!("{}/{}", tr.passed, tr.total_struct));
    }
    println!();

    if models.len() == 4 {
        let ood_r = |idx: usize| oods[idx].depths.iter().map(|d| d.rec_t6).sum::<f64>() / 3.0_f64;
        println!("\n--- SYNTHÈSE CONTRÔLE SCEPTIQUE V4 ---");
        println!(
            "  OOD Rec_nm(C) moy (externe) : Step0={:+.4} | LL={:+.4} | MDL={:+.4} | V4 φ={:+.4}",
            ood_r(0),
            ood_r(1),
            ood_r(2),
            ood_r(3)
        );
        println!(
            "  Transfert P_réel moy (8x)   : Step0={:.3}   | LL={:.3}   | MDL={:.3}   | V4 φ={:.3}",
            trs[0].mean_p, trs[1].mean_p, trs[2].mean_p, trs[3].mean_p
        );
        println!(
            "  Transfert φ_réel moy (8x)   : Step0={:.4}  | LL={:.4}  | MDL={:.4}  | V4 φ={:.4}",
            trs[0].mean_phi, trs[1].mean_phi, trs[2].mean_phi, trs[3].mean_phi
        );
    }
}
