use crate::graph::{PhiGraph, REF_EDGE_IN, RawSubstrate};
use crate::phi::{
    DiscoveredRegime, ProjectionMatch, abductive_discontinuity, byte_overlap_ratio, cosine_sim,
    cut_discontinuity, discover_causal_regimes, fnv_hash,
};
use crate::rng::Rng;

const MAX_CAUSES: usize = 4;

#[derive(Clone, Debug)]
pub struct ResidualSplit {
    pub parts: Vec<(usize, usize)>,
    pub single_err: f32,
    pub split_err: f32,
    pub cross_cos: f32,
}

#[derive(Clone, Debug)]
pub struct DiscoveredConcept {
    pub span: (usize, usize),
    pub name_bytes: Vec<u8>,
    pub score: f32,
    pub coverage: f32,
    pub precision: f32,
    pub mass: f32,
    pub hop_depth: usize,
}

#[derive(Clone, Debug)]
pub struct MultiHopAnalysis {
    pub mold_regimes: Vec<DiscoveredRegime>,
    pub target_spans: Vec<(usize, usize)>,
    pub residual_info: Option<(f32, f32, f32)>,
    pub hop1_per_part: Vec<Vec<ProjectionMatch>>,
    pub concepts: Vec<DiscoveredConcept>,
    pub demoted_meta_count: usize,
    pub field_peak: f32,
    pub field_selectivity: f32,
}

/// Arête d'écho diagonale précalculée uniquement à partir de l'égalité d'octets M(t, j).
#[derive(Clone, Debug)]
struct EchoEdge {
    dst: usize,
    src_id: usize,
    diag_len: usize,
    feat: [f32; REF_EDGE_IN],
}

/// Statistiques locales de répétition dans M sur les zones passives (zéro règle ASCII).
#[derive(Clone, Debug)]
struct PassiveFieldStats {
    n: usize,
    is_passive: Vec<bool>,
    rep1_norm: Vec<f32>,
    rep2_norm: Vec<f32>,
    rep3_norm: Vec<f32>,
    causal_gate: Vec<f32>,
    def_zones: Vec<(usize, usize)>,
}

/// Source continue injectée dans le champ référentiel (moule Hop-1 ou crête Hop-2).
#[derive(Clone, Debug)]
struct RefSource {
    span: (usize, usize),
    cov: [f32; MAX_CAUSES],
    prec: f32,
    mass: f32,
    hop: usize,
}

/// Scène précalculée une seule fois par étape ES pour entraîner sans bruit les 97 poids référentiels.
#[derive(Clone, Debug)]
pub struct PreparedRefScene {
    n_causes: usize,
    stats: PassiveFieldStats,
    sources: Vec<RefSource>,
    edges: Vec<EchoEdge>,
    shuf_stats: PassiveFieldStats,
    shuf_edges: Vec<EchoEdge>,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-12.0, 12.0)).exp())
}

/// Longueur du segment diagonal d'égalité passant par `(t, j)` dans M, avec `(back, fwd)`.
#[inline]
fn diag_extent(bytes: &[u8], t: usize, j: usize) -> (usize, usize, usize) {
    let n = bytes.len();
    let mut b = 0usize;
    while b < 24 && t > b && j > b && bytes[t - b - 1] == bytes[j - b - 1] {
        b += 1;
    }
    let mut f = 0usize;
    while f < 24 && t + f + 1 < n && j + f + 1 < n && bytes[t + f + 1] == bytes[j + f + 1] {
        f += 1;
    }
    (b, f, b + 1 + f)
}

fn build_byte_buckets(bytes: &[u8]) -> [Vec<usize>; 256] {
    let mut buckets: [Vec<usize>; 256] = std::array::from_fn(|_| Vec::new());
    for (i, &b) in bytes.iter().enumerate() {
        buckets[b as usize].push(i);
    }
    buckets
}

/// Décomposition multi-causes guidée par le résidu sur un bloc cible `bytes`.
pub fn decompose_by_residual(graph: &PhiGraph, bytes: &[u8]) -> ResidualSplit {
    let n = bytes.len();
    let sub = RawSubstrate::from_bytes(bytes);
    let rels = graph.encode_relations(&sub);
    let tr0 = graph.abduce(&rels, None);
    let single_err = tr0.final_mean_err;

    if n < 30 {
        return ResidualSplit {
            parts: vec![(0, n)],
            single_err,
            split_err: single_err,
            cross_cos: 1.0,
        };
    }

    let mut pref_err = vec![0.0f32; n + 1];
    for t in 0..n {
        pref_err[t + 1] = pref_err[t] + tr0.final_pos_err[t];
    }

    let mut proposals: Vec<(usize, f32)> = Vec::new();
    let mut p = 14usize;
    while p + 14 <= n {
        let ml = pref_err[p] / (p as f32);
        let mr = (pref_err[n] - pref_err[p]) / ((n - p) as f32);
        let res_jump = (ml - mr).abs();
        let cut_d = cut_discontinuity(&sub, 0, n, p);
        let abd_d = if p % 4 == 0 {
            let l0 = p.saturating_sub(16);
            let r1 = (p + 16).min(n);
            abductive_discontinuity(graph, &rels, (l0, p), (p, r1), 4)
        } else {
            0.0
        };
        proposals.push((p, res_jump + 0.45 * cut_d + 0.55 * abd_d));
        p += 2;
    }
    proposals.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut cands: Vec<usize> = Vec::new();
    for (cp, _) in proposals {
        if cands.iter().all(|&x| x.abs_diff(cp) >= 6) {
            cands.push(cp);
            if cands.len() >= 4 {
                break;
            }
        }
    }
    if cands.iter().all(|&x| x.abs_diff(n / 2) >= 6) {
        cands.push(n / 2);
    }

    let eval_cut = |cp: usize| -> (f32, f32, f32, f32) {
        let rl = graph.encode_bytes(&bytes[0..cp]);
        let rr = graph.encode_bytes(&bytes[cp..n]);
        let tl = graph.abduce(&rl, None);
        let tr = graph.abduce(&rr, None);
        let split_e =
            ((cp as f32) * tl.final_mean_err + ((n - cp) as f32) * tr.final_mean_err) / (n as f32);
        let cos_lr = cosine_sim(&tl.final_state, &tr.final_state);
        let (_, _, e_lr) = graph.evaluate_hypothesis(&tl.final_state, &rr);
        let (_, _, e_rl) = graph.evaluate_hypothesis(&tr.final_state, &rl);
        let cross_e = 0.5 * (e_lr + e_rl);
        let cross_gap = ((cross_e - split_e) / (cross_e + 0.08)).clamp(0.0, 1.0);
        let obj = (single_err - split_e) + 0.30 * cross_gap * (1.0 - cos_lr.max(0.0));
        (obj, split_e, cos_lr, cross_gap)
    };

    let mut best_p = n / 2;
    let mut best_obj = f32::NEG_INFINITY;
    let mut best_res = (single_err, 1.0f32, 0.0f32);

    for cp in cands {
        let (obj, se, cos_lr, cg) = eval_cut(cp);
        if obj > best_obj {
            best_obj = obj;
            best_p = cp;
            best_res = (se, cos_lr, cg);
        }
    }

    let lo = best_p.saturating_sub(2).max(12);
    let hi = (best_p + 2).min(n.saturating_sub(12));
    for cp in lo..=hi {
        if cp == best_p {
            continue;
        }
        let (obj, se, cos_lr, cg) = eval_cut(cp);
        if obj > best_obj {
            best_obj = obj;
            best_p = cp;
            best_res = (se, cos_lr, cg);
        }
    }

    let (split_err, cross_cos, cross_gap) = best_res;
    let significant_drop = (single_err - split_err) > 0.025 || split_err < 0.88 * single_err;
    if cross_cos < 0.55 && cross_gap > 0.22 && significant_drop {
        ResidualSplit {
            parts: vec![(0, best_p), (best_p, n)],
            single_err,
            split_err,
            cross_cos,
        }
    } else {
        ResidualSplit {
            parts: vec![(0, n)],
            single_err,
            split_err: single_err,
            cross_cos: 1.0,
        }
    }
}

/// Démotion purement relationnelle des blocs méta-langage : un régime qui ne fait
/// qu'échoïser par diagonales M(t,j) les zones passives spécifiques de moules antérieurs
/// est reclassé en zone définitionnelle passive (zéro tokenisation, zéro ASCII).
fn demote_metalanguage_regimes(
    bytes: &[u8],
    raw: &[DiscoveredRegime],
) -> (Vec<DiscoveredRegime>, usize) {
    if raw.len() < 3 {
        return (raw.to_vec(), 0);
    }

    let buckets = build_byte_buckets(bytes);
    let mut kept: Vec<DiscoveredRegime> = Vec::with_capacity(raw.len());
    let mut kept_tags: Vec<(usize, usize)> = Vec::with_capacity(raw.len());
    let mut demoted = 0usize;

    for (idx, &reg) in raw.iter().enumerate() {
        let prev_end = if kept.is_empty() {
            0
        } else {
            kept.last().unwrap().core_end
        };

        let mut echoed_molds = 0usize;
        let mut echoed_bytes = 0usize;
        if kept.len() >= 2 && idx + 1 < raw.len() {
            let mut mold_hit = vec![false; kept.len()];
            for pos in reg.core_start..reg.core_end {
                let mut best_m: Option<usize> = None;
                let mut hit_count = 0usize;
                for (m_i, &(ts, te)) in kept_tags.iter().enumerate() {
                    let mut matched_mi = false;
                    for &j in &buckets[bytes[pos] as usize] {
                        if j >= ts && j < te {
                            let (_, _, dlen) = diag_extent(bytes, pos, j);
                            if dlen >= 4 {
                                matched_mi = true;
                                break;
                            }
                        }
                    }
                    if matched_mi {
                        hit_count += 1;
                        best_m = Some(m_i);
                    }
                }
                // Uniquement les échos spécifiques à un seul moule antérieur (exclut "MOLD_", ": ")
                if hit_count == 1
                    && let Some(mi) = best_m
                {
                    mold_hit[mi] = true;
                    echoed_bytes += 1;
                }
            }
            echoed_molds = mold_hit.iter().filter(|&&h| h).count();
        }

        let core_len = reg.core_end.saturating_sub(reg.core_start).max(1);
        if echoed_molds >= 2 || (echoed_molds >= 1 && echoed_bytes * 4 >= core_len) {
            demoted += 1;
            continue;
        }

        let tag_start = prev_end.max(reg.core_start.saturating_sub(24));
        kept_tags.push((tag_start, reg.core_start));
        kept.push(reg);
    }

    let k = kept.len();
    if k == 0 {
        return (raw.to_vec(), 0);
    }
    let lead_gap = kept[0].core_start;
    let trail_gap = bytes.len().saturating_sub(kept[k - 1].core_end);
    let suffix_mode = trail_gap >= 3 && lead_gap < 3;

    for i in 0..k {
        let prev_end = if i == 0 { 0 } else { kept[i - 1].core_end };
        let next_start = if i + 1 == k {
            bytes.len()
        } else {
            kept[i + 1].core_start
        };
        kept[i].referent = if suffix_mode {
            if next_start.saturating_sub(kept[i].core_end) >= 3 {
                Some((kept[i].core_end, next_start))
            } else {
                None
            }
        } else if kept[i].core_start.saturating_sub(prev_end) >= 3 {
            Some((prev_end, kept[i].core_start))
        } else {
            None
        };
    }

    (kept, demoted)
}

fn project_span_to_molds(
    graph: &PhiGraph,
    bytes: &[u8],
    q_span: (usize, usize),
    molds: &[DiscoveredRegime],
) -> Vec<ProjectionMatch> {
    let q_bytes = &bytes[q_span.0..q_span.1];
    let q_rels = graph.encode_bytes(q_bytes);
    let q_trace = graph.abduce(&q_rels, None);

    let mut matches = Vec::with_capacity(molds.len());
    for (idx, reg) in molds.iter().enumerate() {
        let core_bytes = &bytes[reg.core_start..reg.core_end];
        let core_rels = graph.encode_bytes(core_bytes);
        let core_trace = graph.abduce(&core_rels, None);

        let (_, _, fwd_err) = graph.evaluate_hypothesis(&q_trace.final_state, &core_rels);
        let (_, _, bwd_err) = graph.evaluate_hypothesis(&core_trace.final_state, &q_rels);
        let cross_err = 0.5 * (fwd_err + bwd_err);

        let cross_gen = (-2.2 * cross_err).exp().clamp(0.0, 1.0);
        let cos_sim = cosine_sim(&q_trace.final_state, &core_trace.final_state);
        let cos_pos = 0.5 * (1.0 + cos_sim);
        let score = (cross_gen * cos_pos).sqrt();

        let byte_overlap = byte_overlap_ratio(q_bytes, core_bytes);
        matches.push(ProjectionMatch {
            dst_idx: idx,
            score,
            cross_err,
            cos_sim,
            core_start: reg.core_start,
            core_end: reg.core_end,
            referent: reg.referent,
            byte_overlap,
        });
    }
    matches.sort_by(|a, b| b.score.total_cmp(&a.score));
    matches
}

/// Calcule les statistiques purement relationnelles de chaque octet dans les zones passives.
fn compute_passive_stats(
    bytes: &[u8],
    causal_gate: Vec<f32>,
    def_zones: Vec<(usize, usize)>,
) -> PassiveFieldStats {
    let n = bytes.len();
    let mut is_passive = vec![false; n];
    for &(ds, de) in &def_zones {
        for flag in &mut is_passive[ds..de.min(n)] {
            *flag = true;
        }
    }

    let buckets = build_byte_buckets(bytes);
    let mut rep1_norm = vec![0.0f32; n];
    let mut rep2_norm = vec![0.0f32; n];
    let mut rep3_norm = vec![0.0f32; n];

    for &(ds, de) in &def_zones {
        for t in ds..de.min(n) {
            let mut c1 = 0usize;
            let mut c2 = 0usize;
            let mut c3 = 0usize;
            for &j in &buckets[bytes[t] as usize] {
                if j == t || !is_passive[j] {
                    continue;
                }
                c1 += 1;
                let left_eq = t > 0 && j > 0 && bytes[t - 1] == bytes[j - 1];
                let right_eq = t + 1 < n && j + 1 < n && bytes[t + 1] == bytes[j + 1];
                if left_eq || right_eq {
                    c2 += 1;
                    let (_, _, dlen) = diag_extent(bytes, t, j);
                    if dlen >= 3 {
                        c3 += 1;
                    }
                }
            }
            rep1_norm[t] = (c1 as f32 / 4.0).tanh();
            rep2_norm[t] = (c2 as f32 / 2.5).tanh();
            rep3_norm[t] = (c3 as f32 / 2.0).tanh();
        }
    }

    PassiveFieldStats {
        n,
        is_passive,
        rep1_norm,
        rep2_norm,
        rep3_norm,
        causal_gate,
        def_zones,
    }
}

/// Extrait les arêtes d'échos diagonaux (longueur >= 3 dans M) depuis un ensemble de sources
/// vers les zones définitionnelles passives, avec leurs 6 features purement relationnelles.
fn build_echo_edges(
    bytes: &[u8],
    stats: &PassiveFieldStats,
    sources: &[RefSource],
) -> Vec<EchoEdge> {
    let n = bytes.len();
    let buckets = build_byte_buckets(bytes);
    let mut edges = Vec::new();

    for (src_id, src) in sources.iter().enumerate() {
        let (ss, se) = src.span;
        if se <= ss {
            continue;
        }
        // Pré-calcul de la multiplicité inter-sources de chaque position `t` dans `src`
        let mut src_mult = vec![0usize; se - ss];
        for t in ss..se {
            let mut mult = 0usize;
            for (other_id, other) in sources.iter().enumerate() {
                if other_id == src_id {
                    continue;
                }
                let (os, oe) = other.span;
                let mut hit = false;
                for &j in &buckets[bytes[t] as usize] {
                    if j >= os && j < oe {
                        let (_, _, dlen) = diag_extent(bytes, t, j);
                        if dlen >= 3 {
                            hit = true;
                            break;
                        }
                    }
                }
                if hit {
                    mult += 1;
                }
            }
            src_mult[t - ss] = mult;
        }

        // Recherche des échos diagonaux dans les zones passives (hors de la source elle-même)
        for &(ds, de) in &stats.def_zones {
            for j in ds..de.min(n) {
                if j >= ss && j < se {
                    continue;
                }
                // Garde le meilleur écho diagonal provenant de cette source vers `j`
                let mut best: Option<(usize, usize, usize, usize)> = None; // (dlen, t, back, fwd)
                for &t in &buckets[bytes[j] as usize] {
                    if t < ss || t >= se {
                        continue;
                    }
                    let (back, fwd, dlen) = diag_extent(bytes, t, j);
                    // Limite la diagonale à l'intérieur de la zone source `ss..se`
                    let b_lim = back.min(t - ss);
                    let f_lim = fwd.min(se - 1 - t);
                    let eff_len = b_lim + 1 + f_lim;
                    if eff_len < 3 {
                        continue;
                    }
                    // Calcule la spécificité moyenne sur le segment source
                    let ok = match best {
                        None => true,
                        Some((bl, bt, _, _)) => {
                            eff_len > bl
                                || (eff_len == bl && src_mult[t - ss] < src_mult[bt - ss])
                        }
                    };
                    if ok {
                        best = Some((eff_len, t, b_lim, f_lim));
                    }
                }

                if let Some((dlen, t, b_lim, f_lim)) = best {
                    // Multiplicité minimale du segment source (évite qu'un bord partagé pénalise le cœur)
                    let mut seg_mult = 0usize;
                    let seg_s = t - b_lim;
                    let seg_e = t + f_lim + 1;
                    for p in seg_s..seg_e {
                        seg_mult += src_mult[p - ss];
                    }
                    let avg_src_mult = (seg_mult as f32) / (dlen as f32);

                    // Contexte de bordure purement relationnel aux extrémités du segment échoïsé en `j`
                    let run_l = j.saturating_sub(b_lim);
                    let run_r = (j + f_lim + 1).min(n);
                    let flank_l = if run_l == ds || !stats.is_passive[run_l - 1] {
                        1.0
                    } else {
                        0.45 * stats.rep1_norm[run_l - 1] + 0.55 * stats.rep2_norm[run_l - 1]
                    };
                    let flank_r = if run_r >= de || run_r >= n || !stats.is_passive[run_r] {
                        1.0
                    } else {
                        0.45 * stats.rep1_norm[run_r] + 0.55 * stats.rep2_norm[run_r]
                    };
                    let min_flank = flank_l.min(flank_r);

                    let feat = [
                        (dlen as f32 / 6.0).tanh(),
                        1.0 / (1.0 + avg_src_mult).sqrt(),
                        1.0 - 0.65 * stats.rep3_norm[j],
                        (t.abs_diff(j) as f32 / 64.0).tanh(),
                        if j > t { 1.0 } else { -1.0 },
                        min_flank,
                    ];

                    edges.push(EchoEdge {
                        dst: j,
                        src_id,
                        diag_len: dlen,
                        feat,
                    });
                }
            }
        }
    }

    edges
}

#[derive(Clone, Debug)]
struct FieldPassOutput {
    phi: Vec<f32>,
    cov_field: Vec<[f32; MAX_CAUSES]>,
    prec_field: Vec<f32>,
    mass_field: Vec<f32>,
    hop_field: Vec<usize>,
    echo_str: Vec<f32>,
    p_head: Vec<f32>,
    p_boiler: Vec<f32>,
    echo_spec: f32,
    echo_src_cov: f32,
}

/// Exécute une passe complète du Champ Référentiel Connexionniste (échos longue portée +
/// cellule de rôle + intégration récurrente non-réciproque sur les zones passives).
fn run_ref_field_pass(
    graph: &PhiGraph,
    stats: &PassiveFieldStats,
    sources: &[RefSource],
    edges: &[EchoEdge],
    n_causes: usize,
) -> FieldPassOutput {
    let n = stats.n;
    let mut echo_str = vec![0.0f32; n];
    let mut echo_sum = vec![0.0f32; n];
    let mut echo_cov = vec![[0.0f32; MAX_CAUSES]; n];
    let mut echo_prec = vec![0.0f32; n];
    let mut echo_step_mass = vec![0.0f32; n];
    let mut echo_hop = vec![1usize; n];
    let mut src_max_w = vec![0.0f32; sources.len()];

    for edge in edges {
        let w = graph.eval_echo_gate(&edge.feat);
        let j = edge.dst;
        echo_sum[j] += w;
        if w > src_max_w[edge.src_id] {
            src_max_w[edge.src_id] = w;
        }
        if w > echo_str[j] {
            let src = &sources[edge.src_id];
            echo_str[j] = w;
            echo_cov[j] = src.cov;
            echo_prec[j] = src.prec;
            echo_step_mass[j] = src.mass / (edge.diag_len.max(3) as f32);
            echo_hop[j] = src.hop;
        }
    }

    // Spécificité non-supervisée des échos : chaque position échoïsée doit pointer 1 source nette
    let mut spec_sum = 0.0f32;
    let mut spec_cnt = 0usize;
    for j in 0..n {
        if echo_str[j] > 0.25 {
            spec_sum += echo_str[j] / (echo_sum[j] + 1e-4);
            spec_cnt += 1;
        }
    }
    let echo_spec = if spec_cnt == 0 {
        0.0
    } else {
        spec_sum / (spec_cnt as f32)
    };
    let echo_src_cov = if sources.is_empty() {
        0.0
    } else {
        src_max_w.iter().sum::<f32>() / (sources.len() as f32)
    };

    // Cellule de rôle purement connexionniste (zéro règle ASCII)
    let mut p_head = vec![0.0f32; n];
    let mut p_boiler = vec![0.0f32; n];
    for &(ds, de) in &stats.def_zones {
        for t in ds..de.min(n) {
            let rf = [
                echo_str[t],
                stats.rep1_norm[t],
                stats.rep2_norm[t],
                stats.rep3_norm[t],
                stats.causal_gate[t],
            ];
            let (ph, pb) = graph.eval_role_cell(&rf);
            p_head[t] = ph;
            p_boiler[t] = pb;
        }
    }

    // Intégration non-réciproque dans une direction donnée (`rev = true` pour R->L, `false` pour L->R)
    let integrate_dir = |rev: bool| -> (Vec<f32>, Vec<[f32; MAX_CAUSES]>, Vec<f32>, Vec<f32>, Vec<usize>, f32) {
        let mut phi = vec![0.0f32; n];
        let mut cov_f = vec![[0.0f32; MAX_CAUSES]; n];
        let mut prec_f = vec![0.0f32; n];
        let mut mass_f = vec![0.0f32; n];
        let mut hop_f = vec![2usize; n];
        let mut best_peak = 0.0f32;

        for &(ds, de) in &stats.def_zones {
            let end = de.min(n);
            if end <= ds {
                continue;
            }
            let mut acc_cov = [0.0f32; MAX_CAUSES];
            let mut acc_prec_sum = 0.0f32;
            let mut acc_mass = 0.0f32;
            let mut acc_max_hop = 1usize;
            let mut head_seen = 0.0f32;

            let len = end - ds;
            for step in 0..len {
                let t = if rev { end - 1 - step } else { ds + step };
                let pe = echo_str[t];
                let ph = p_head[t];
                let pb = p_boiler[t];

                let (keep, next_hs) = graph.eval_clause_gates(&[pe, ph, pb, head_seen]);
                if keep < 0.15 {
                    acc_max_hop = 1;
                }
                for c in acc_cov.iter_mut().take(n_causes) {
                    *c *= keep;
                }
                acc_prec_sum *= keep;
                acc_mass *= keep;

                if pe > 0.20 {
                    for k in 0..n_causes {
                        let cand = pe * echo_cov[t][k];
                        if cand > acc_cov[k] {
                            acc_cov[k] = cand;
                        }
                    }
                    let dm = pe * echo_step_mass[t];
                    acc_prec_sum += dm * echo_prec[t];
                    acc_mass += dm;
                    if echo_hop[t] > acc_max_hop {
                        acc_max_hop = echo_hop[t];
                    }
                }

                let cov_t = acc_cov[0..n_causes].iter().sum::<f32>() / (n_causes.max(1) as f32);
                let prec_t = if acc_mass > 1e-4 {
                    (acc_prec_sum / acc_mass).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                // Porte continue de multi-composition (requiert >= ~2 causes distinctes)
                let multi_gate = sigmoid(4.0 * (acc_mass - 1.25));
                let head_clean = ph * (1.0 - pe) * (1.0 - pb);
                let score_t = head_clean * (cov_t * prec_t).sqrt() * multi_gate;

                phi[t] = score_t;
                cov_f[t] = acc_cov;
                prec_f[t] = prec_t;
                mass_f[t] = acc_mass;
                hop_f[t] = acc_max_hop + 1;
                if score_t > best_peak {
                    best_peak = score_t;
                }

                head_seen = next_hs;
            }
        }
        (phi, cov_f, prec_f, mass_f, hop_f, best_peak)
    };

    let rl = integrate_dir(true);
    let lr = integrate_dir(false);
    let (phi, cov_field, prec_field, mass_field, hop_field, _) = if rl.5 >= lr.5 { rl } else { lr };

    FieldPassOutput {
        phi,
        cov_field,
        prec_field,
        mass_field,
        hop_field,
        echo_str,
        p_head,
        p_boiler,
        echo_spec,
        echo_src_cov,
    }
}

/// Extrait les plateaux de crête continus du champ d'explication `phi[t]` (zéro parsing de chaîne).
fn extract_field_ridges(
    bytes: &[u8],
    stats: &PassiveFieldStats,
    pass: &FieldPassOutput,
    n_causes: usize,
) -> Vec<(DiscoveredConcept, [f32; MAX_CAUSES])> {
    let mut peak_global = 0.0f32;
    for &v in &pass.phi {
        if v > peak_global {
            peak_global = v;
        }
    }
    if peak_global < 0.15 {
        return Vec::new();
    }
    let thresh = (0.45 * peak_global).max(0.14);

    let mut out = Vec::new();
    for &(ds, de) in &stats.def_zones {
        let end = de.min(bytes.len());
        let mut t = ds;
        while t < end {
            if pass.phi[t] < thresh {
                t += 1;
                continue;
            }
            let s = t;
            while t < end && pass.phi[t] >= thresh {
                t += 1;
            }
            let e = t;
            if e - s < 2 {
                continue;
            }

            let len_f = (e - s) as f32;
            let mut mean_score = 0.0f32;
            let mut mean_prec = 0.0f32;
            let mut mean_mass = 0.0f32;
            let mut mean_cov_vec = [0.0f32; MAX_CAUSES];
            let mut max_hop = 2usize;

            for pos in s..e {
                mean_score += pass.phi[pos] / len_f;
                mean_prec += pass.prec_field[pos] / len_f;
                mean_mass += pass.mass_field[pos] / len_f;
                for k in 0..n_causes {
                    mean_cov_vec[k] += pass.cov_field[pos][k] / len_f;
                }
                if pass.hop_field[pos] > max_hop {
                    max_hop = pass.hop_field[pos];
                }
            }
            let coverage = mean_cov_vec[0..n_causes].iter().sum::<f32>() / (n_causes.max(1) as f32);

            out.push((
                DiscoveredConcept {
                    span: (s, e),
                    name_bytes: bytes[s..e].to_vec(),
                    score: mean_score,
                    coverage,
                    precision: mean_prec,
                    mass: mean_mass,
                    hop_depth: max_hop,
                },
                mean_cov_vec,
            ));
        }
    }

    out.sort_by(|a, b| b.0.score.total_cmp(&a.0.score));
    out
}

/// Partitionne les régimes découverts en moules primitifs, composantes cibles et zones passives.
fn partition_scene(
    graph: &PhiGraph,
    bytes: &[u8],
) -> Option<(
    Vec<DiscoveredRegime>,
    Vec<(usize, usize)>,
    Option<(f32, f32, f32)>,
    Vec<Vec<ProjectionMatch>>,
    Vec<(usize, usize)>,
    Vec<(usize, usize)>,
    Vec<f32>,
    usize,
)> {
    let sub = RawSubstrate::from_bytes(bytes);
    let (_, gates_ctx, _) = graph.encode_with_gate(&sub);
    let raw_regimes = discover_causal_regimes(graph, bytes, true);
    if raw_regimes.len() < 2 {
        return None;
    }

    let (regimes, demoted_meta_count) = demote_metalanguage_regimes(bytes, &raw_regimes);
    if regimes.len() < 2 {
        return None;
    }

    let buckets = build_byte_buckets(bytes);

    // 1. Regroupe les régimes finaux formant une cible multi-causes (couture <= 14 sans écho de tag)
    let k = regimes.len();
    let mut split_t = k - 1;
    while split_t >= 3 && (k - split_t) < MAX_CAUSES {
        let cand_idx = split_t - 1;
        let gap_s = regimes[cand_idx].core_end;
        let gap_e = regimes[split_t].core_start;
        let gap_to_next = gap_e.saturating_sub(gap_s);
        if gap_to_next > 14 {
            break;
        }
        let prev_end = regimes[cand_idx - 1].core_end;
        let gap_before = regimes[cand_idx].core_start.saturating_sub(prev_end);
        if gap_before > 14 && gap_before < 20 {
            break;
        }

        // Vérifie que la couture entre `cand_idx` et `split_t` n'est pas une zone de définitions
        let mut gap_echoes_tag = false;
        if gap_to_next >= 4 {
            for pos in gap_s..gap_e {
                for m_i in 0..cand_idx {
                    let pe = if m_i == 0 {
                        0
                    } else {
                        regimes[m_i - 1].core_end
                    };
                    let ts = pe.max(regimes[m_i].core_start.saturating_sub(24));
                    let te = regimes[m_i].core_start;
                    for &j in &buckets[bytes[pos] as usize] {
                        if j >= ts && j < te && diag_extent(bytes, pos, j).2 >= 4 {
                            gap_echoes_tag = true;
                            break;
                        }
                    }
                    if gap_echoes_tag {
                        break;
                    }
                }
                if gap_echoes_tag {
                    break;
                }
            }
        }
        if gap_echoes_tag {
            break;
        }

        let earlier = &regimes[0..cand_idx];
        let proj = project_span_to_molds(
            graph,
            bytes,
            (regimes[cand_idx].core_start, regimes[cand_idx].core_end),
            earlier,
        );
        let strong_match = proj
            .first()
            .is_some_and(|m| m.score >= 0.60 && m.cos_sim >= 0.40);
        if strong_match {
            split_t = cand_idx;
        } else {
            break;
        }
    }

    let mold_regimes = regimes[0..split_t].to_vec();
    let mut target_spans: Vec<(usize, usize)> = regimes[split_t..]
        .iter()
        .map(|r| (r.core_start, r.core_end))
        .collect();

    if target_spans.len() >= 2 {
        let mut merged_t: Vec<(usize, usize)> = vec![target_spans[0]];
        for &(s, e) in &target_spans[1..] {
            let prev = *merged_t.last().unwrap();
            let h_prev = graph
                .abduce(&graph.encode_bytes(&bytes[prev.0..prev.1]), None)
                .final_state;
            let h_cur = graph
                .abduce(&graph.encode_bytes(&bytes[s..e]), None)
                .final_state;
            let ov = byte_overlap_ratio(&bytes[prev.0..prev.1], &bytes[s..e]);
            if cosine_sim(&h_prev, &h_cur) > 0.72 && ov > 0.50 {
                merged_t.last_mut().unwrap().1 = e;
            } else {
                merged_t.push((s, e));
            }
        }
        target_spans = merged_t;
    }

    // 2. Décomposition guidée par le résidu si cible monolithique
    let mut residual_info: Option<(f32, f32, f32)> = None;
    if target_spans.len() == 1 {
        let (ts, te) = target_spans[0];
        let dec = decompose_by_residual(graph, &bytes[ts..te]);
        if dec.parts.len() >= 2 {
            target_spans = dec
                .parts
                .into_iter()
                .map(|(a, b)| (ts + a, ts + b))
                .collect();
            residual_info = Some((dec.single_err, dec.split_err, dec.cross_cos));
        }
    } else if target_spans.len() >= 2 {
        let (s0, e0) = target_spans[0];
        let (s1, e1) = target_spans[1];
        let full_err = graph
            .abduce(&graph.encode_bytes(&bytes[s0..e1]), None)
            .final_mean_err;
        let t0 = graph.abduce(&graph.encode_bytes(&bytes[s0..e0]), None);
        let t1 = graph.abduce(&graph.encode_bytes(&bytes[s1..e1]), None);
        let l0 = (e0 - s0) as f32;
        let l1 = (e1 - s1) as f32;
        let split_err = (l0 * t0.final_mean_err + l1 * t1.final_mean_err) / (l0 + l1).max(1.0);
        let cos_lr = cosine_sim(&t0.final_state, &t1.final_state);
        residual_info = Some((full_err, split_err, cos_lr));
    }

    if target_spans.len() > MAX_CAUSES {
        target_spans.truncate(MAX_CAUSES);
    }

    // 3. Projection Hop-1
    let mut hop1_per_part: Vec<Vec<ProjectionMatch>> = Vec::with_capacity(target_spans.len());
    for &span in &target_spans {
        hop1_per_part.push(project_span_to_molds(graph, bytes, span, &mold_regimes));
    }

    // 4. Zones d'étiquettes locales des moules et zones définitionnelles passives
    let mut tag_spans: Vec<(usize, usize)> = Vec::with_capacity(mold_regimes.len());
    let mut def_zones: Vec<(usize, usize)> = Vec::new();

    for i in 0..mold_regimes.len() {
        let prev_end = if i == 0 {
            0
        } else {
            mold_regimes[i - 1].core_end
        };
        let cs = mold_regimes[i].core_start;
        if cs.saturating_sub(prev_end) > 28 {
            def_zones.push((prev_end, cs - 24));
            tag_spans.push((cs - 24, cs));
        } else {
            tag_spans.push((prev_end, cs));
        }
    }

    let after_molds = mold_regimes.last().unwrap().core_end;
    let before_target = target_spans[0].0;
    if before_target.saturating_sub(after_molds) >= 12 {
        def_zones.push((after_molds, before_target));
    }
    let after_target = target_spans.last().unwrap().1;
    if bytes.len().saturating_sub(after_target) >= 12 {
        def_zones.push((after_target, bytes.len()));
    }

    Some((
        mold_regimes,
        target_spans,
        residual_info,
        hop1_per_part,
        tag_spans,
        def_zones,
        gates_ctx,
        demoted_meta_count,
    ))
}

fn build_hop1_sources(
    mold_count: usize,
    tag_spans: &[(usize, usize)],
    hop1_per_part: &[Vec<ProjectionMatch>],
) -> Vec<RefSource> {
    let n_causes = hop1_per_part.len().min(MAX_CAUSES);
    let mut sources = Vec::with_capacity(mold_count);
    for m in 0..mold_count {
        let mut cov = [0.0f32; MAX_CAUSES];
        let mut prec = 0.0f32;
        for (c_idx, matches) in hop1_per_part.iter().take(n_causes).enumerate() {
            if let Some(found) = matches.iter().find(|x| x.dst_idx == m) {
                cov[c_idx] = found.score;
                if found.score > prec {
                    prec = found.score;
                }
            }
        }
        sources.push(RefSource {
            span: tag_spans[m],
            cov,
            prec,
            mass: 1.0,
            hop: 1,
        });
    }
    sources
}

/// Prépare une scène référentielle (originale + contrefactuelle à zones passives mélangées)
/// une seule fois par étape ES pour entraîner les 97 poids référentiels sans aucune étiquette.
pub fn prepare_ref_scene(base: &PhiGraph, bytes: &[u8]) -> Option<PreparedRefScene> {
    let (mold_regimes, target_spans, _, hop1_per_part, tag_spans, def_zones, gates_ctx, _) =
        partition_scene(base, bytes)?;
    if def_zones.is_empty() || mold_regimes.len() < 2 {
        return None;
    }

    let n_causes = target_spans.len().min(MAX_CAUSES);
    let sources = build_hop1_sources(mold_regimes.len(), &tag_spans, &hop1_per_part);
    let stats = compute_passive_stats(bytes, gates_ctx.clone(), def_zones.clone());
    let edges = build_echo_edges(bytes, &stats, &sources);
    if edges.len() < 6 {
        return None;
    }

    // Scène contrefactuelle : mélange les octets des zones définitionnelles passives
    let mut shuf_bytes = bytes.to_vec();
    let mut rng = Rng::new(fnv_hash(bytes) ^ 0xCAFE_BABE);
    for &(ds, de) in &def_zones {
        if de > ds && de <= shuf_bytes.len() {
            rng.shuffle(&mut shuf_bytes[ds..de]);
        }
    }
    let shuf_stats = compute_passive_stats(&shuf_bytes, gates_ctx, def_zones);
    let shuf_edges = build_echo_edges(&shuf_bytes, &shuf_stats, &sources);

    Some(PreparedRefScene {
        n_causes,
        stats,
        sources,
        edges,
        shuf_stats,
        shuf_edges,
    })
}

/// Signal 100 % non-supervisé pour entraîner les 97 poids du Champ Référentiel Connexionniste :
/// combine sélectivité des crêtes, discrimination contrefactuelle (vs zones passives mélangées),
/// spécificité des échos et séparation purement relationnelle des rôles tête / connecteur.
pub fn eval_ref_scene_phi(graph: &PhiGraph, scene: &PreparedRefScene) -> f32 {
    let pass = run_ref_field_pass(
        graph,
        &scene.stats,
        &scene.sources,
        &scene.edges,
        scene.n_causes,
    );
    let shuf = run_ref_field_pass(
        graph,
        &scene.shuf_stats,
        &scene.sources,
        &scene.shuf_edges,
        scene.n_causes,
    );

    // 1. Alignement purement relationnel des rôles (tête = faible répétition, connecteur = forte répétition)
    let mut role_sum = 0.0f32;
    let mut clean_sum = 0.0f32;
    let mut mass_pen = 0.0f32;
    let mut p_cnt = 0usize;
    for &(ds, de) in &scene.stats.def_zones {
        for t in ds..de.min(scene.stats.n) {
            let pe = pass.echo_str[t];
            let ph = pass.p_head[t];
            let pb = pass.p_boiler[t];
            let r2 = scene.stats.rep2_norm[t];

            let head_target = (1.0 - pe) * (1.0 - r2);
            let boiler_target = (1.0 - pe) * r2;
            let r_score = ph * head_target + (1.0 - ph) * (1.0 - head_target)
                + pb * boiler_target
                + (1.0 - pb) * (1.0 - boiler_target);
            role_sum += 0.5 * r_score;
            clean_sum += (1.0 - ph * pb) * (1.0 - ph * pe);
            // Pénalise la fuite inter-clauses (accumulation excessive > 3.2 sur une cible 2-causes)
            if pass.mass_field[t] > (scene.n_causes as f32) + 1.2 {
                mass_pen += (pass.mass_field[t] - ((scene.n_causes as f32) + 1.2)).min(2.0);
            }
            p_cnt += 1;
        }
    }
    let inv_p = 1.0 / (p_cnt.max(1) as f32);
    let role_align = role_sum * inv_p;
    let role_clean = clean_sum * inv_p;
    let leak_pen = mass_pen * inv_p;

    // 2. Crêtes continues (sans besoin de bytes : on mesure les plateaux de pass.phi)
    let mut peak = 0.0f32;
    for &v in &pass.phi {
        if v > peak {
            peak = v;
        }
    }
    let mut shuf_peak = 0.0f32;
    for &v in &shuf.phi {
        if v > shuf_peak {
            shuf_peak = v;
        }
    }

    let mut ridge_scores: Vec<f32> = Vec::new();
    let mut single_byte_pen = 0.0f32;
    if peak >= 0.12 {
        let thresh = (0.45 * peak).max(0.10);
        for &(ds, de) in &scene.stats.def_zones {
            let end = de.min(scene.stats.n);
            let mut t = ds;
            while t < end {
                if pass.phi[t] < thresh {
                    t += 1;
                    continue;
                }
                let s = t;
                while t < end && pass.phi[t] >= thresh {
                    t += 1;
                }
                let len = t - s;
                if len >= 2 {
                    let mean_s = pass.phi[s..t].iter().sum::<f32>() / (len as f32);
                    ridge_scores.push(mean_s);
                } else {
                    single_byte_pen += 0.05;
                }
            }
        }
    }
    ridge_scores.sort_by(|a, b| b.total_cmp(a));

    let (sel_score, margin_score, count_bonus) = if ridge_scores.len() >= 2 {
        let s1 = ridge_scores[0];
        let s2 = ridge_scores[1];
        let s_rest =
            ridge_scores[1..].iter().sum::<f32>() / ((ridge_scores.len() - 1) as f32);
        let sel = s1 * (1.0 - s_rest);
        let marg = (s1 - s2).clamp(0.0, 0.35);
        let c_bon = if (2..=6).contains(&ridge_scores.len()) {
            0.15
        } else {
            0.0
        };
        (sel, marg, c_bon)
    } else if ridge_scores.len() == 1 {
        (0.15 * ridge_scores[0], 0.0, 0.0)
    } else {
        (0.0, 0.0, -0.15)
    };

    let disc = (peak - shuf_peak).clamp(0.0, 1.0);
    let echo_q = pass.echo_spec * pass.echo_src_cov;

    0.45 * role_align
        + 0.25 * role_clean
        + 0.45 * echo_q
        + 0.55 * disc
        + 0.90 * sel_score
        + 0.80 * margin_score
        + count_bonus
        - 0.35 * leak_pen
        - single_byte_pen.min(0.25)
}

/// Analyse complète 100 % connexionniste : Hop-1 + Champ Référentiel Hop-2 + second passage Hop-3.
pub fn analyze_multihop(graph: &PhiGraph, bytes: &[u8]) -> Option<MultiHopAnalysis> {
    let (
        mold_regimes,
        target_spans,
        residual_info,
        hop1_per_part,
        tag_spans,
        def_zones,
        gates_ctx,
        demoted_meta_count,
    ) = partition_scene(graph, bytes)?;

    let n_causes = target_spans.len().min(MAX_CAUSES);
    let mut concepts: Vec<DiscoveredConcept> = Vec::new();
    let mut field_peak = 0.0f32;
    let mut field_selectivity = 0.0f32;

    if !def_zones.is_empty() && mold_regimes.len() >= 2 {
        let sources_h1 = build_hop1_sources(mold_regimes.len(), &tag_spans, &hop1_per_part);
        let stats = compute_passive_stats(bytes, gates_ctx, def_zones);
        let edges_h1 = build_echo_edges(bytes, &stats, &sources_h1);

        if !edges_h1.is_empty() {
            // Passage 1 : Hop-2 (moules -> concepts composés)
            let pass1 = run_ref_field_pass(graph, &stats, &sources_h1, &edges_h1, n_causes);
            let ridges1 = extract_field_ridges(bytes, &stats, &pass1, n_causes);

            // Passage 2 : Hop-3 (exactement le même champ, sourcé par Hop-1 + crêtes Hop-2)
            let mut ext_sources = sources_h1;
            for (c, cov_vec) in &ridges1 {
                if c.span.1.saturating_sub(c.span.0) >= 3 && c.score >= 0.22 {
                    ext_sources.push(RefSource {
                        span: c.span,
                        cov: *cov_vec,
                        prec: c.precision,
                        mass: c.mass.clamp(1.5, 2.5),
                        hop: c.hop_depth,
                    });
                }
            }

            let mut all_concepts: Vec<DiscoveredConcept> =
                ridges1.into_iter().map(|(c, _)| c).collect();

            if ext_sources.len() > mold_regimes.len() {
                let edges_h2 = build_echo_edges(bytes, &stats, &ext_sources);
                let pass2 = run_ref_field_pass(graph, &stats, &ext_sources, &edges_h2, n_causes);
                let ridges2 = extract_field_ridges(bytes, &stats, &pass2, n_causes);

                for (c3, _) in ridges2 {
                    if c3.hop_depth >= 3 {
                        let overlaps = all_concepts.iter().any(|old| {
                            let s = old.span.0.max(c3.span.0);
                            let e = old.span.1.min(c3.span.1);
                            e > s
                        });
                        if !overlaps {
                            all_concepts.push(c3);
                        }
                    }
                }
            }

            all_concepts.sort_by(|a, b| b.score.total_cmp(&a.score));
            if let Some(top) = all_concepts.first() {
                field_peak = top.score;
                let rest = if all_concepts.len() > 1 {
                    all_concepts[1..].iter().map(|c| c.score).sum::<f32>()
                        / ((all_concepts.len() - 1) as f32)
                } else {
                    0.0
                };
                field_selectivity = top.score * (1.0 - rest);
            }
            concepts = all_concepts;
        }
    }

    Some(MultiHopAnalysis {
        mold_regimes,
        target_spans,
        residual_info,
        hop1_per_part,
        concepts,
        demoted_meta_count,
        field_peak,
        field_selectivity,
    })
}

pub fn analyze_monolithic_residual(graph: &PhiGraph, bytes: &[u8]) -> ResidualSplit {
    decompose_by_residual(graph, bytes)
}
