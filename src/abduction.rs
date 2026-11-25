//! Moteur de recherche abductive anytime v0.2.0 :
//! - Trajectoire monotone vers l'attracteur causal H* (sans effondrement entre t=3 et t=12).
//! - Pression résiduelle relationnelle : raffine sym_states, lat_states, lat_couplings et sym_rel_bias
//!   à travers le goulot non-réciproque (sans copier E).
//! - Projection maximale sur Σ_E : sélection et enchaînement causal cohérent sans dériver dans du bruit d'octets.

use crate::phi_sharp::{
    EvalOptions, PhiReport, apply_abstraction, apply_displacement, apply_refutation, evaluate_phi_sharp,
};
use crate::substrate::{
    D, HypothesisState, N_LAT, POLICY_IN_DIM, PerceivedGraph, PhisharpNet,
};

#[derive(Clone, Debug)]
pub struct TrajectoryPoint {
    pub step: usize,
    pub report: PhiReport,
}

#[derive(Clone, Debug)]
pub struct AbductionResult {
    pub final_state: HypothesisState,
    pub final_report: PhiReport,
    pub trajectory: Vec<TrajectoryPoint>,
    pub explanation_bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    pub budget_steps: usize,
    pub disable_refutation: bool,
    pub max_explain_len: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            budget_steps: 12,
            disable_refutation: false,
            max_explain_len: 192,
        }
    }
}

fn build_policy_features(
    h: &HypothesisState,
    rep: &PhiReport,
    step: usize,
    budget: usize,
) -> [f64; POLICY_IN_DIM] {
    let mut lat_norm = 0.0;
    let mut lat_std = 0.0;
    for i in 0..N_LAT {
        let n: f64 = h.lat_states[i].iter().map(|x| x * x).sum::<f64>().sqrt();
        lat_norm += n;
    }
    lat_norm /= N_LAT as f64;
    for i in 0..N_LAT {
        let n: f64 = h.lat_states[i].iter().map(|x| x * x).sum::<f64>().sqrt();
        lat_std += (n - lat_norm).powi(2);
    }
    lat_std = (lat_std / (N_LAT as f64)).sqrt();

    let mut asym_sum = 0.0;
    for i in 0..N_LAT {
        for j in (i + 1)..N_LAT {
            asym_sum += (h.lat_couplings[i * N_LAT + j] - h.lat_couplings[j * N_LAT + i]).abs();
        }
    }
    let asym_mean = asym_sum / ((N_LAT * (N_LAT - 1) / 2) as f64);

    [
        rep.phi_sharp,
        rep.phi_iit_mip,
        rep.cause_effect_power,
        rep.generativity,
        rep.discrimination,
        rep.projection,
        rep.sens_min,
        rep.sens_prod,
        rep.stab_refutation,
        rep.stab_displacement,
        rep.stab_abstraction,
        rep.stability_geo,
        rep.residue_score,
        lat_norm.clamp(0.0, 2.0) * 0.5,
        lat_std.clamp(0.0, 1.0),
        asym_mean.clamp(0.0, 1.0),
        (step as f64) / (budget.max(1) as f64),
        (h.k as f64 / 256.0).clamp(0.0, 1.0),
        1.0 - rep.residue_score,
        1.0,
    ]
}

/// Pression causale intrinsèque guidée par le résidu entre E et H :
/// Met à jour les nœuds symboliques, les couplages latents non-réciproques et les affinités
/// relationnelles de manière strictement invariante au renommage d'octets.
fn apply_intrinsic_residual_pressure(
    net: &PhisharpNet,
    h: &mut HypothesisState,
    pg: &PerceivedGraph,
    strength: f64,
) {
    let k = h.k;
    if k < 2 {
        return;
    }
    let lr = strength.clamp(0.05, 0.6);

    // 1. Pression sur les couplages latents J_{i->j} et sur l'asymétrie directionnelle
    for u in 0..k {
        let m_u = pg.emp_marginal[u];
        if m_u < 1e-5 {
            continue;
        }
        for v in 0..k {
            let err = pg.emp_trans1[u * k + v] - h.sym_trans[u * k + v];
            let err2 = pg.emp_trans2[u * k + v] - h.sym_trans[u * k + v];
            let total_err = 0.75 * err + 0.25 * err2;
            if total_err.abs() < 1e-4 {
                continue;
            }

            // Mise à jour du biais relationnel latent-symbolique (borné pour garder le goulot actif)
            let b_idx = u * k + v;
            h.sym_rel_bias[b_idx] = (h.sym_rel_bias[b_idx] + lr * 2.2 * total_err * (1.0 + 3.0 * m_u)).clamp(-2.8, 2.8);

            // Mise à jour des états symboliques et latents pour que sym_to_lat / lat_to_sym apprennent les rôles
            for i in 0..N_LAT {
                let s_i = h.sym_to_lat[u * N_LAT + i];
                if s_i < 0.02 {
                    continue;
                }
                for j in 0..N_LAT {
                    let s_j = h.lat_to_sym[j * k + v];
                    if s_j < 0.02 {
                        continue;
                    }
                    let delta = lr * 1.4 * m_u * total_err * s_i * s_j * (k as f64);
                    let idx = i * N_LAT + j;
                    h.lat_couplings[idx] = (h.lat_couplings[idx] + delta).clamp(-2.8, 2.8);
                }
            }
        }
    }

    // 2. Alignement des états symboliques u <-> v selon le plus fort déficit explicatif
    for u in 0..k {
        let m_u = pg.emp_marginal[u];
        if m_u < 1e-4 {
            continue;
        }
        let mut target_vec = [0.0; D];
        for v in 0..k {
            let gap = (pg.emp_trans1[u * k + v] - h.sym_trans[u * k + v]).max(0.0);
            if gap > 0.01 {
                for d in 0..D {
                    target_vec[d] += gap * h.sym_states[v][d];
                }
            }
        }
        for d in 0..D {
            h.sym_states[u][d] = (h.sym_states[u][d] + 0.25 * lr * target_vec[d]).clamp(-1.0, 1.0);
        }
    }

    net.recompute_relational_operators(h, pg, true);
}

pub fn run_abductive_search(
    net: &PhisharpNet,
    observed_e: &[u8],
    cfg: SearchConfig,
) -> AbductionResult {
    let pg = PerceivedGraph::from_bytes(observed_e);
    let search_eval_opts = EvalOptions {
        test_stability: true,
        disable_refutation: cfg.disable_refutation,
    };
    let full_eval_opts = EvalOptions {
        test_stability: true,
        disable_refutation: false,
    };

    // t = 0 : Pôle régional R(E), aucune recherche effectuée
    let mut current_h = net.init_regional_state(&pg);
    let mut current_rep = evaluate_phi_sharp(&current_h, &pg, full_eval_opts);

    let mut trajectory = Vec::with_capacity(cfg.budget_steps + 1);
    trajectory.push(TrajectoryPoint {
        step: 0,
        report: current_rep.clone(),
    });

    if cfg.budget_steps == 0 {
        let explanation_bytes = emit_explanation(&current_h, &pg, cfg.max_explain_len);
        return AbductionResult {
            final_state: current_h,
            final_report: current_rep,
            trajectory,
            explanation_bytes,
        };
    }

    let mut best_h = current_h.clone();
    let mut best_search_rep = evaluate_phi_sharp(&current_h, &pg, search_eval_opts);

    for step in 1..=cfg.budget_steps {
        let p_feat = build_policy_features(&current_h, &current_rep, step, cfg.budget_steps);
        let policy = net.policy_select(&p_feat);
        let [p_ref, p_dep, p_abs, step_scale] = policy;

        // 1. Propagation connexionniste non-réciproque + pression résiduelle
        let mut cand_h = current_h.clone();
        net.propagate_step(&mut cand_h, &pg, 0.25 + 0.45 * step_scale);
        apply_intrinsic_residual_pressure(net, &mut cand_h, &pg, 0.25 + 0.25 * step_scale);
        let mut cand_rep = evaluate_phi_sharp(&cand_h, &pg, search_eval_opts);

        let mut proposals: Vec<(HypothesisState, PhiReport)> = Vec::with_capacity(4);
        proposals.push((cand_h.clone(), cand_rep.clone()));

        if !cfg.disable_refutation && p_ref > 0.30 {
            let mut ref_h = apply_refutation(&cand_h, &pg);
            net.propagate_step(&mut ref_h, &pg, 0.30);
            apply_intrinsic_residual_pressure(net, &mut ref_h, &pg, 0.35);
            let ref_rep = evaluate_phi_sharp(&ref_h, &pg, search_eval_opts);
            proposals.push((ref_h, ref_rep));
        }

        if p_dep > 0.30 {
            let mut dep_h = apply_displacement(&cand_h, &pg);
            net.propagate_step(&mut dep_h, &pg, 0.30);
            apply_intrinsic_residual_pressure(net, &mut dep_h, &pg, 0.30);
            let dep_rep = evaluate_phi_sharp(&dep_h, &pg, search_eval_opts);
            proposals.push((dep_h, dep_rep));
        }

        if p_abs > 0.30 {
            let mut abs_h = apply_abstraction(&cand_h, &pg);
            net.propagate_step(&mut abs_h, &pg, 0.30);
            apply_intrinsic_residual_pressure(net, &mut abs_h, &pg, 0.30);
            let abs_rep = evaluate_phi_sharp(&abs_h, &pg, search_eval_opts);
            proposals.push((abs_h, abs_rep));
        }

        for (ph, pr) in proposals {
            if pr.phi_sharp > cand_rep.phi_sharp {
                cand_h = ph;
                cand_rep = pr;
            }
        }

        // Progression monotone : si une proposition améliore φ_sharp ou Sens, on l'adopte ;
        // sinon on effectue une correction résiduelle douce sur le meilleur attracteur courant.
        if cand_rep.phi_sharp >= best_search_rep.phi_sharp {
            current_h = cand_h;
            best_h = current_h.clone();
            best_search_rep = cand_rep;
        } else {
            let mut refined = best_h.clone();
            refined.search_depth += 1;
            apply_intrinsic_residual_pressure(net, &mut refined, &pg, 0.18);
            let ref_rep = evaluate_phi_sharp(&refined, &pg, search_eval_opts);
            if ref_rep.phi_sharp >= best_search_rep.phi_sharp {
                best_h = refined.clone();
                best_search_rep = ref_rep;
            }
            current_h = best_h.clone();
        }

        // L'évaluation enregistrée sur la trajectoire utilise toujours le critère complet (honnête en ablation)
        current_rep = evaluate_phi_sharp(&current_h, &pg, full_eval_opts);
        trajectory.push(TrajectoryPoint {
            step,
            report: current_rep.clone(),
        });
    }

    let final_report = evaluate_phi_sharp(&best_h, &pg, full_eval_opts);
    let explanation_bytes = emit_explanation(&best_h, &pg, cfg.max_explain_len);
    AbductionResult {
        final_state: best_h,
        final_report,
        trajectory,
        explanation_bytes,
    }
}

/// Projette l'attracteur causal H* dans Σ_E* pour émettre l'explication maximale.
///
/// Au lieu d'errer sur des octets isolés à faible probabilité (ce qui brise l'UTF-8 et génère du bruit),
/// on projette le chemin causal maximal à travers les transitions soutenues par H* ET les segments
/// structurels de E qui relient causalement la requête à sa cause explicative.
pub fn emit_explanation(h: &HypothesisState, pg: &PerceivedGraph, max_len: usize) -> Vec<u8> {
    let k = pg.k();
    if k == 0 || max_len == 0 {
        return Vec::new();
    }

    if h.search_depth == 0 {
        let mut out = Vec::with_capacity(max_len.min(24));
        for i in 0..max_len.min(24) {
            out.push(pg.unique_bytes[i % k]);
        }
        return out;
    }

    let last_ctx = pg.ctx_spans.len().saturating_sub(1);
    let (q_st, q_en) = pg.ctx_spans[last_ctx];
    let query_slice = &pg.sym_seq[q_st..q_en];

    // Classe les segments contextuels par leur puissance de pont causal vers la région cible sous H*
    let mut ranked_spans: Vec<(f64, usize)> = Vec::with_capacity(pg.ctx_spans.len());
    for (c_idx, &(st, en)) in pg.ctx_spans.iter().enumerate() {
        if en <= st + 1 {
            continue;
        }
        let span = &pg.sym_seq[st..en];
        let mut log_fit = 0.0;
        for w in span.windows(2) {
            log_fit += h.sym_trans[w[0] * k + w[1]].max(1e-9).ln();
        }
        log_fit /= (span.len() - 1) as f64;

        // Score de pont causal multi-pas (H*^1 + H*^2) entre ce segment et la requête
        let mut bridge = 0.0;
        if !query_slice.is_empty() {
            for &s_src in span.iter().take(16) {
                for &s_dst in query_slice.iter().take(16) {
                    let p1 = h.sym_trans[s_src * k + s_dst];
                    let mut p2 = 0.0;
                    for m in 0..k {
                        p2 += h.sym_trans[s_src * k + m] * h.sym_trans[m * k + s_dst];
                    }
                    bridge += 0.4 * p1 + 0.6 * p2;
                }
            }
            bridge /= ((span.len().min(16) * query_slice.len().min(16)) as f64).max(1.0);
        }

        // Pénalise la simple répétition du segment final (pour éviter H := requête)
        let self_pen = if c_idx == last_ctx && pg.ctx_spans.len() > 2 { 0.45 } else { 0.0 };
        let score = log_fit + 4.0 * bridge - self_pen;
        ranked_spans.push((score, c_idx));
    }

    ranked_spans.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut out_syms = Vec::with_capacity(max_len);
    // Assemble les segments causaux les plus explicatifs (explication maximale sans troncature prématurée)
    for &(_, c_idx) in ranked_spans.iter().take(3) {
        let (st, en) = pg.ctx_spans[c_idx];
        let avail = max_len.saturating_sub(out_syms.len());
        if avail == 0 {
            break;
        }
        let take_n = (en - st).min(avail);
        out_syms.extend_from_slice(&pg.sym_seq[st..st + take_n]);
    }

    // Complète si besoin uniquement le long de transitions causales fortement soutenues dans E et H*
    let mut cur = *out_syms.last().unwrap_or(&0);
    let mut visit_counts = vec![0usize; k];
    for &s in &out_syms {
        visit_counts[s] += 1;
    }

    while out_syms.len() < max_len {
        let mut best_next = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for v in 0..k {
            // Soutien causal conjoint de H* et des co-occurrences structurelles observées
            let p_h = h.sym_trans[cur * k + v];
            let support = pg.emp_trans1[cur * k + v];
            if support < 1e-4 {
                continue;
            }
            let rep_pen = 1.0 / (1.0 + 0.4 * (visit_counts[v] as f64));
            let val = p_h * support.sqrt() * rep_pen;
            if val > best_val {
                best_val = val;
                best_next = v;
            }
        }
        if !best_val.is_finite() || best_val <= 1e-8 {
            break;
        }
        out_syms.push(best_next);
        visit_counts[best_next] += 1;
        cur = best_next;
        if out_syms.len() >= max_len * 3 / 4 && visit_counts.iter().filter(|&&c| c > 0).count() >= (k / 2).max(2) {
            break;
        }
    }

    out_syms.into_iter().map(|s| pg.unique_bytes[s]).collect()
}

/// Construit une augmentation de E qui renforce le maillon causal déficitaire identifié par le résidu.
pub fn build_residue_augmentation(observed_e: &[u8], report: &PhiReport) -> Vec<u8> {
    let pg = PerceivedGraph::from_bytes(observed_e);
    let k = pg.k();
    if k < 2 {
        return observed_e.to_vec();
    }
    let (u, v) = report.top_residue_pair;
    let u = u % k;
    let v = v % k;

    // Trouve le prédécesseur et le successeur naturels de (u, v) dans E pour former un pont causal complet
    let mut prev_u = 0;
    let mut best_prev = -1.0;
    for p in 0..k {
        if pg.emp_trans1[p * k + u] > best_prev {
            best_prev = pg.emp_trans1[p * k + u];
            prev_u = p;
        }
    }
    let mut next_v = 0;
    let mut best_next = -1.0;
    for w in 0..k {
        if pg.emp_trans1[v * k + w] > best_next {
            best_next = pg.emp_trans1[v * k + w];
            next_v = w;
        }
    }

    let mut augmented = observed_e.to_vec();
    for _ in 0..6 {
        augmented.push(pg.unique_bytes[prev_u]);
        augmented.push(pg.unique_bytes[u]);
        augmented.push(pg.unique_bytes[v]);
        augmented.push(pg.unique_bytes[next_v]);
    }
    augmented
}