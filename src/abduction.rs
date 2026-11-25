//! Moteur de recherche abductive authentique v0.3.0 :
//! - Aucune béquille codée en dur sur emp_trans1 : toute évolution passe par `net.propagate_step`
//!   (matrices non-réciproques + projections apprises du résidu) et les défis abductifs.
//! - Conclusion émise par résonance causale dans l'espace latent V_lat entre la clause cible
//!   (requête) et les clauses explicatives émergentes de E (sans couper de mots ni babiller).

use crate::math::{cosine_similarity, entropy_bits, normalize_prob};
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
            max_explain_len: 256,
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

        // Propagation purement connexionniste (aucune béquille sur emp_trans1)
        let mut cand_h = current_h.clone();
        net.propagate_step(&mut cand_h, &pg, 0.28 + 0.50 * step_scale);
        let mut cand_rep = evaluate_phi_sharp(&cand_h, &pg, search_eval_opts);

        let mut proposals: Vec<(HypothesisState, PhiReport)> = Vec::with_capacity(4);
        proposals.push((cand_h.clone(), cand_rep.clone()));

        if !cfg.disable_refutation && p_ref > 0.25 {
            let mut ref_h = apply_refutation(&cand_h, &pg);
            net.propagate_step(&mut ref_h, &pg, 0.35);
            let ref_rep = evaluate_phi_sharp(&ref_h, &pg, search_eval_opts);
            proposals.push((ref_h, ref_rep));
        }

        if p_dep > 0.25 {
            let mut dep_h = apply_displacement(&cand_h, &pg);
            net.propagate_step(&mut dep_h, &pg, 0.35);
            let dep_rep = evaluate_phi_sharp(&dep_h, &pg, search_eval_opts);
            proposals.push((dep_h, dep_rep));
        }

        if p_abs > 0.25 {
            let mut abs_h = apply_abstraction(&cand_h, &pg);
            net.propagate_step(&mut abs_h, &pg, 0.35);
            let abs_rep = evaluate_phi_sharp(&abs_h, &pg, search_eval_opts);
            proposals.push((abs_h, abs_rep));
        }

        for (ph, pr) in proposals {
            if pr.phi_sharp > cand_rep.phi_sharp {
                cand_h = ph;
                cand_rep = pr;
            }
        }

        if cand_rep.phi_sharp >= best_search_rep.phi_sharp {
            current_h = cand_h;
            best_h = current_h.clone();
            best_search_rep = cand_rep;
        } else {
            // Pas de raffinement connexionniste léger autour du meilleur attracteur
            let mut refined = best_h.clone();
            net.propagate_step(&mut refined, &pg, 0.16);
            let ref_rep = evaluate_phi_sharp(&refined, &pg, search_eval_opts);
            if ref_rep.phi_sharp >= best_search_rep.phi_sharp {
                best_h = refined.clone();
                best_search_rep = ref_rep;
            }
            current_h = best_h.clone();
        }

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

/// Calcule le profil latent causal (distribution sur N_LAT et vecteur de rôle moyen dans R^D)
/// d'une séquence de symboles à travers les matrices de l'attracteur H*.
fn span_latent_signature(h: &HypothesisState, span: &[usize]) -> ([f64; N_LAT], [f64; D], f64) {
    let mut lat_dist = [0.0; N_LAT];
    let mut vec_rep = [0.0; D];
    if span.is_empty() {
        normalize_prob(&mut lat_dist);
        return (lat_dist, vec_rep, 0.0);
    }

    // Distribution de branchement / incertitude causale interne de cette clause dans H*
    let mut trans_ent_sum = 0.0;
    for (idx, &s) in span.iter().enumerate() {
        for l in 0..N_LAT {
            lat_dist[l] += h.sym_to_lat[s * N_LAT + l];
        }
        for d in 0..D {
            vec_rep[d] += h.sym_states[s][d];
        }
        if idx + 1 < span.len() {
            let mut row_ent = 0.0;
            for v in 0..h.k {
                let p = h.sym_trans[s * h.k + v].max(1e-12);
                row_ent -= p * p.ln();
            }
            trans_ent_sum += row_ent / ((h.k as f64).max(2.0).ln());
        }
    }
    let nf = span.len() as f64;
    for d in 0..D {
        vec_rep[d] /= nf;
    }
    normalize_prob(&mut lat_dist);
    let mean_branch_ent = if span.len() > 1 { trans_ent_sum / (nf - 1.0) } else { 0.0 };
    (lat_dist, vec_rep, mean_branch_ent)
}

/// Projette l'attracteur causal H* dans Σ_E* pour émettre la conclusion maximale :
/// 1. Infère la signature causale latente de la requête finale (incluant son entropie de branchement
///    causal = multiplicité de lectures concurrentes dans H*).
/// 2. Propulse les clauses candidates à travers les couplages causaux J_{i->j} de H*.
/// 3. Émet les clauses émergentes entières qui unifient le mieux l'attracteur requête -> cause,
///    sans troncature arbitraire au milieu des unités ni babil bigramme.
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

    let n_ctx = pg.ctx_spans.len();
    let last_ctx = n_ctx.saturating_sub(1);
    let (q_st, q_en) = pg.ctx_spans[last_ctx];
    let query_slice = &pg.sym_seq[q_st..q_en];

    let (q_lat_raw, q_vec, q_branch_ent) = span_latent_signature(h, query_slice);

    // Inversion causale arrière de la requête via J^T (quelle cause latente explique cette requête ?)
    let mut q_cause_lat = [0.0; N_LAT];
    for i in 0..N_LAT {
        let mut acc = 0.0;
        for j in 0..N_LAT {
            let w_ij = (2.0 * h.lat_couplings[i * N_LAT + j]).clamp(-8.0, 8.0).exp();
            acc += w_ij * q_lat_raw[j];
        }
        q_cause_lat[i] = acc;
    }
    normalize_prob(&mut q_cause_lat);
    let q_lat_entropy = entropy_bits(&q_cause_lat) / (N_LAT as f64).log2();

    let mut ranked_spans: Vec<(f64, usize)> = Vec::with_capacity(n_ctx);
    for (c_idx, &(st, en)) in pg.ctx_spans.iter().enumerate() {
        if en <= st + 1 {
            continue;
        }
        let span = &pg.sym_seq[st..en];
        let (c_lat, c_vec, c_branch_ent) = span_latent_signature(h, span);

        // Projection avant de cette clause à travers J_{i->j} vers l'espace des effets
        let mut c_eff_lat = [0.0; N_LAT];
        for j in 0..N_LAT {
            let mut acc = 0.0;
            for i in 0..N_LAT {
                let w_ij = (2.0 * h.lat_couplings[i * N_LAT + j]).clamp(-8.0, 8.0).exp();
                acc += c_lat[i] * w_ij;
            }
            c_eff_lat[j] = acc;
        }
        normalize_prob(&mut c_eff_lat);

        // Résonance cause-effet dans l'espace latent (pas de copie de surface)
        let lat_resonance: f64 = c_eff_lat.iter().zip(q_cause_lat.iter()).map(|(a, b)| (a * b).sqrt()).sum();
        let vec_resonance = cosine_similarity(&c_vec, &q_vec).max(-1.0);

        // Concordance de multiplicité causale (si la requête a un fort branchement latent / ambiguïté,
        // privilégie les clauses qui expriment structurellement une pluralité de chemins dans H*)
        let c_lat_entropy = entropy_bits(&c_lat) / (N_LAT as f64).log2();
        let branching_match = 1.0 - (q_branch_ent - c_branch_ent).abs() - 0.5 * (q_lat_entropy - c_lat_entropy).abs();

        // Qualité générative intrinsèque de cette clause sous H*
        let mut log_fit = 0.0;
        for w in span.windows(2) {
            log_fit += h.sym_trans[w[0] * k + w[1]].max(1e-9).ln();
        }
        log_fit /= (span.len() - 1) as f64;
        let norm_fit = (log_fit / (k as f64).max(2.0).ln() + 1.0).clamp(0.0, 1.0);

        // Pénalise la simple répétition de la requête elle-même
        let self_pen = if c_idx == last_ctx && n_ctx > 2 { 0.55 } else { 0.0 };

        let total = 2.2 * lat_resonance + 0.8 * vec_resonance + 1.1 * branching_match + 0.6 * norm_fit - self_pen;
        ranked_spans.push((total, c_idx));
    }

    ranked_spans.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut out_syms = Vec::with_capacity(max_len);
    for &(_, c_idx) in ranked_spans.iter().take(2) {
        let (st, en) = pg.ctx_spans[c_idx];
        let avail = max_len.saturating_sub(out_syms.len());
        if avail == 0 {
            break;
        }
        let take_n = (en - st).min(avail);
        out_syms.extend_from_slice(&pg.sym_seq[st..st + take_n]);
    }

    out_syms.into_iter().map(|s| pg.unique_bytes[s]).collect()
}

/// Construit une augmentation équilibrée de E en insérant la clause relationnelle manquante
/// indiquée par le résidu (sans briser l'équilibre pairs/impairs de projection).
pub fn build_residue_augmentation(observed_e: &[u8], report: &PhiReport) -> Vec<u8> {
    let pg = PerceivedGraph::from_bytes(observed_e);
    let k = pg.k();
    if k < 2 || pg.ctx_spans.is_empty() {
        return observed_e.to_vec();
    }
    let (u, v) = report.top_residue_pair;
    let u = u % k;
    let v = v % k;

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

    // Symbole séparateur émergent (le plus fort en fin des spans découverts)
    let sep_sym = pg.sym_seq[pg.ctx_spans[0].1.saturating_sub(1)];
    let bridge = [
        pg.unique_bytes[prev_u],
        pg.unique_bytes[u],
        pg.unique_bytes[v],
        pg.unique_bytes[next_v],
        pg.unique_bytes[sep_sym],
    ];

    // Insertion symétrique (milieu + fin) pour clarifier la causalité des deux vues
    let mid = observed_e.len() / 2;
    let mut augmented = Vec::with_capacity(observed_e.len() + 20);
    augmented.extend_from_slice(&observed_e[..mid]);
    augmented.extend_from_slice(&bridge);
    augmented.extend_from_slice(&bridge);
    augmented.extend_from_slice(&observed_e[mid..]);
    augmented.extend_from_slice(&bridge);
    augmented.extend_from_slice(&bridge);
    augmented
}