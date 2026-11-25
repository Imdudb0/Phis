//! Moteur de recherche abductive anytime : trajectoire H(0) -> H(T) et projection maximale dans Σ_E*.
//!
//! - À t = 0 (budget nul), H(0) = R(E) (activation régionale non cherchée).
//! - À chaque pas t, le réseau non-réciproque propage l'état, la politique abductive propose
//!   des transformations (réfutation, déplacement, abstraction, raffinement causal intrinsèque)
//!   et ne retient les bifurcations qui élèvent φ_sharp et réduisent le résidu.
//! - L'explication émise est projetée uniquement sur Σ_E (équivariance stricte).

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

/// Pression causale intrinsèque sur les couplages latents (sans gradient externe : guidée par le résidu
/// et les transitions relationnelles de E vues à travers sym_to_lat / lat_to_sym).
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
    let lr = strength.clamp(0.02, 0.35);

    // Propage l'écart résiduel E - sym_trans dans l'espace latent causal J_{i->j}
    for u in 0..k {
        let m_u = pg.emp_marginal[u];
        if m_u < 1e-5 {
            continue;
        }
        for v in 0..k {
            let err = pg.emp_trans1[u * k + v] - h.sym_trans[u * k + v];
            if err.abs() < 1e-4 {
                continue;
            }
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
                    let delta = lr * m_u * err * s_i * s_j * (k as f64);
                    let idx = i * N_LAT + j;
                    h.lat_couplings[idx] = (h.lat_couplings[idx] + delta).clamp(-2.5, 2.5);
                }
            }
        }
    }
    // Renforce la symétrie inverse modérée des états latents les plus sollicités par le résidu
    let (ru, rv) = {
        let mut best = (0, 0);
        let mut best_g = -1.0;
        for u in 0..k {
            for v in 0..k {
                let g = (pg.emp_trans1[u * k + v] - h.sym_trans[u * k + v]) * pg.emp_marginal[u];
                if g > best_g {
                    best_g = g;
                    best = (u, v);
                }
            }
        }
        best
    };
    for i in 0..N_LAT {
        let w = h.sym_to_lat[ru * N_LAT + i] - h.lat_to_sym[i * k + rv];
        for d in 0..D {
            h.lat_states[i][d] = (h.lat_states[i][d] + 0.08 * lr * w * h.sym_states[ru][d]).clamp(-1.0, 1.0);
        }
    }
    net.recompute_relational_operators(h, pg, true);
}

/// Exécute la recherche abductive complète sur `observed_e` avec budget donné.
pub fn run_abductive_search(
    net: &PhisharpNet,
    observed_e: &[u8],
    cfg: SearchConfig,
) -> AbductionResult {
    let pg = PerceivedGraph::from_bytes(observed_e);
    let eval_opts = EvalOptions {
        test_stability: true,
        disable_refutation: cfg.disable_refutation,
    };

    // t = 0 : Pôle régional R(E), aucune recherche effectuée
    let mut current_h = net.init_regional_state(&pg);
    let mut current_rep = evaluate_phi_sharp(&current_h, &pg, eval_opts);

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
    let mut best_rep = current_rep.clone();

    for step in 1..=cfg.budget_steps {
        let p_feat = build_policy_features(&current_h, &current_rep, step, cfg.budget_steps);
        let policy = net.policy_select(&p_feat);
        let [p_ref, p_dep, p_abs, step_scale] = policy;

        // 1. Propagation connexionniste non-réciproque
        let mut cand_h = current_h.clone();
        net.propagate_step(&mut cand_h, &pg, 0.35 + 0.55 * step_scale);
        apply_intrinsic_residual_pressure(net, &mut cand_h, &pg, 0.14 + 0.16 * step_scale);
        let mut cand_rep = evaluate_phi_sharp(&cand_h, &pg, eval_opts);

        // 2. Candidats de bifurcation abductive guidés par la politique apprise
        let mut proposals: Vec<(HypothesisState, PhiReport)> = Vec::with_capacity(4);
        proposals.push((cand_h.clone(), cand_rep.clone()));

        if !cfg.disable_refutation && p_ref > 0.35 {
            // Hypothèse durcie contre sa propre réfutation : on réinjecte la pression causale après réfutation
            let mut ref_h = apply_refutation(&cand_h, &pg);
            net.propagate_step(&mut ref_h, &pg, 0.45);
            apply_intrinsic_residual_pressure(net, &mut ref_h, &pg, 0.20);
            let ref_rep = evaluate_phi_sharp(&ref_h, &pg, eval_opts);
            proposals.push((ref_h, ref_rep));
        }

        if p_dep > 0.35 {
            let mut dep_h = apply_displacement(&cand_h, &pg);
            net.propagate_step(&mut dep_h, &pg, 0.40);
            apply_intrinsic_residual_pressure(net, &mut dep_h, &pg, 0.18);
            let dep_rep = evaluate_phi_sharp(&dep_h, &pg, eval_opts);
            proposals.push((dep_h, dep_rep));
        }

        if p_abs > 0.35 {
            let mut abs_h = apply_abstraction(&cand_h, &pg);
            net.propagate_step(&mut abs_h, &pg, 0.40);
            apply_intrinsic_residual_pressure(net, &mut abs_h, &pg, 0.18);
            let abs_rep = evaluate_phi_sharp(&abs_h, &pg, eval_opts);
            proposals.push((abs_h, abs_rep));
        }

        // Sélection par φ_sharp (qui incorpore Sens, Stabilité et -Résidu)
        for (ph, pr) in proposals {
            if pr.phi_sharp > cand_rep.phi_sharp {
                cand_h = ph;
                cand_rep = pr;
            }
        }

        // Acceptation monotone ou exploration douce si le score Sens progresse
        if cand_rep.phi_sharp >= current_rep.phi_sharp * 0.96 || cand_rep.sens_prod > current_rep.sens_prod {
            current_h = cand_h;
            current_rep = cand_rep;
        } else {
            // Léger mouvement résiduel vers la piste prometteuse
            for i in 0..N_LAT {
                for d in 0..D {
                    current_h.lat_states[i][d] =
                        0.8 * current_h.lat_states[i][d] + 0.2 * cand_h.lat_states[i][d];
                }
            }
            current_h.search_depth += 1;
            net.recompute_relational_operators(&mut current_h, &pg, true);
            current_rep = evaluate_phi_sharp(&current_h, &pg, eval_opts);
        }

        if current_rep.phi_sharp > best_rep.phi_sharp {
            best_h = current_h.clone();
            best_rep = current_rep.clone();
        }

        trajectory.push(TrajectoryPoint {
            step,
            report: current_rep.clone(),
        });
    }

    let explanation_bytes = emit_explanation(&best_h, &pg, cfg.max_explain_len);
    AbductionResult {
        final_state: best_h,
        final_report: best_rep,
        trajectory,
        explanation_bytes,
    }
}

/// Projette l'attracteur causal H* dans l'alphabet auto-spécifié Σ_E pour émettre l'explication maximale.
///
/// Propriétés strictes :
/// 1. À budget nul (`search_depth == 0`), le readout reflète l'état non cherché (faible cohérence).
/// 2. Toute permutation bijective π des octets de E produit exactement π(explanation_bytes) (équivariance stricte).
/// 3. Ne copie pas bêtement E entier : sélectionne et déroule les motifs/segments causaux qui maximisent
///    la résonance intégrée avec H* à travers le graphe de transition `sym_trans`.
pub fn emit_explanation(h: &HypothesisState, pg: &PerceivedGraph, max_len: usize) -> Vec<u8> {
    let k = pg.k();
    if k == 0 || max_len == 0 {
        return Vec::new();
    }

    if h.search_depth == 0 {
        // Budget nul : émission directe du pôle régional R(E) (non abductive)
        let mut out = Vec::with_capacity(max_len.min(32));
        for i in 0..max_len.min(24) {
            out.push(pg.unique_bytes[i % k]);
        }
        return out;
    }

    // Calcule la saillance causale de chaque segment contextuel de E par rapport à l'attracteur H*
    let mut best_span_idx = 0usize;
    let mut best_span_score = f64::NEG_INFINITY;

    // On identifie aussi le dernier segment (souvent porteur de l'effet final / requête) pour éviter
    // de simplement recopier la requête telle quelle : on cherche le segment EXPLICATIF qui s'y projette.
    let last_ctx = pg.ctx_spans.len().saturating_sub(1);
    let (q_st, q_en) = pg.ctx_spans[last_ctx];
    let query_slice = &pg.sym_seq[q_st..q_en];

    for (c_idx, &(st, en)) in pg.ctx_spans.iter().enumerate() {
        if en <= st + 1 {
            continue;
        }
        let span = &pg.sym_seq[st..en];
        // Score de résonance causale de ce segment sous h.sym_trans
        let mut log_fit = 0.0;
        for w in span.windows(2) {
            log_fit += h.sym_trans[w[0] * k + w[1]].max(1e-9).ln();
        }
        log_fit /= (span.len() - 1) as f64;

        // Pont causal entre ce segment et la région finale/requête via h.sym_trans^2
        let mut bridge = 0.0;
        if !query_slice.is_empty() {
            for &s_src in span.iter().take(12) {
                for &s_dst in query_slice.iter().take(12) {
                    let mut p2 = 0.0;
                    for m in 0..k {
                        p2 += h.sym_trans[s_src * k + m] * h.sym_trans[m * k + s_dst];
                    }
                    bridge += p2;
                }
            }
            bridge /= ((span.len().min(12) * query_slice.len().min(12)) as f64).max(1.0);
        }

        // Pénalise la copie littérale du tout dernier segment seul si d'autres segments l'expliquent
        let self_query_penalty = if c_idx == last_ctx && pg.ctx_spans.len() > 2 { 0.35 } else { 0.0 };
        let score = log_fit + 3.0 * bridge - self_query_penalty;
        if score > best_span_score {
            best_span_score = score;
            best_span_idx = c_idx;
        }
    }

    // Déroulement génératif maximal guidé par l'attracteur H* :
    // Part du segment causal central et prolonge par la chaîne de transition causale de H*
    // sur Σ_E en évitant les boucles dégénérées de longueur 1.
    let (st, en) = pg.ctx_spans[best_span_idx];
    let mut out_syms = Vec::with_capacity(max_len);

    // Ancre structurelle issue du segment causal le plus intégrateur
    let anchor_take = (en - st).min(max_len / 2);
    out_syms.extend_from_slice(&pg.sym_seq[st..st + anchor_take]);

    // Continuation par projection causale maximale depuis H* (déterministe, équivariante)
    let mut cur = *out_syms.last().unwrap_or(&0);
    let mut visit_counts = vec![0usize; k];
    for &s in &out_syms {
        visit_counts[s] += 1;
    }

    while out_syms.len() < max_len {
        let mut best_next = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for v in 0..k {
            let p = h.sym_trans[cur * k + v];
            // Pénalité anti-stagnation pour favoriser l'explication maximale plutôt qu'une boucle de 1 octet
            let rep_pen = 1.0 / (1.0 + 0.35 * (visit_counts[v] as f64));
            let self_pen = if v == cur { 0.45 } else { 1.0 };
            let val = p * rep_pen * self_pen;
            if val > best_val {
                best_val = val;
                best_next = v;
            }
        }
        out_syms.push(best_next);
        visit_counts[best_next] += 1;
        cur = best_next;

        // Si l'on a couvert l'essentiel des transitions fortes du segment cible, on reste borné
        if out_syms.len() >= 24 && visit_counts.iter().filter(|&&c| c > 0).count() >= (k * 3 / 4).max(2) && out_syms.len() >= max_len / 2 {
            break;
        }
    }

    out_syms.into_iter().map(|s| pg.unique_bytes[s]).collect()
}

/// Construit une augmentation de E correspondant exactement à ce que le résidu indique manquer
/// (utilisé dans le test de résidu du critère A : ajouter à E ce qu'il indique manquer doit faire monter φ_sharp).
pub fn build_residue_augmentation(observed_e: &[u8], report: &PhiReport) -> Vec<u8> {
    let pg = PerceivedGraph::from_bytes(observed_e);
    let (u, v) = report.top_residue_pair;
    if pg.k() < 2 {
        return observed_e.to_vec();
    }
    let bu = pg.unique_bytes[u % pg.k()];
    let bv = pg.unique_bytes[v % pg.k()];

    // Injecte des ponts relationnels manquants autour du couple déficitaire (u, v)
    let mut augmented = observed_e.to_vec();
    for _ in 0..3 {
        augmented.push(bu);
        augmented.push(bv);
        // Chaîne le meilleur successeur de v dans E pour fermer la boucle causale
        let mut best_w = 0;
        let mut best_p = -1.0;
        for w in 0..pg.k() {
            let p = pg.emp_trans1[v * pg.k() + w];
            if p > best_p {
                best_p = p;
                best_w = w;
            }
        }
        augmented.push(pg.unique_bytes[best_w]);
    }
    augmented
}