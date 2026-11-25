//! Substrat connexionniste 100% relationnel à matrices non-réciproques (v0.4.0).
//!
//! Avancées structurelles :
//! 1. Découverte de motifs multi-octets émergents par cohésion PMI (sans rien coder en dur,
//!    strictement invariant sous permutation bijective d'octets).
//! 2. Graphe relationnel inter-clauses NON-SÉQUENTIEL (`ctx_adj`) reliant les clauses par
//!    recouvrement de motifs pondéré par IDF et flux asymétrique (fini `c-1 -> c+1`).
//! 3. Segmentation émergente adaptative (`MAX_CTX = 48`, `min_span = 6`) capable d'isoler
//!    proprement des requêtes courtes ("Who are social?") sans les fusionner dans un article voisin.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::math::{Rng, entropy_bits, matvec, normalize_prob, sigmoid, softplus, tanh_clamped};

pub const D: usize = 12;
pub const N_LAT: usize = 12;
pub const MAX_CTX: usize = 48;
pub const MAX_MOTIFS: usize = 72;
pub const SYM_FEAT_DIM: usize = 14;
pub const CTX_FEAT_DIM: usize = 10;
pub const POLICY_IN_DIM: usize = 20;
pub const POLICY_OUT_DIM: usize = 4;

#[derive(Clone, Debug)]
pub struct PerceivedGraph {
    pub raw_len: usize,
    pub unique_bytes: Vec<u8>,
    pub sym_seq: Vec<usize>,
    pub sym_features: Vec<[f64; SYM_FEAT_DIM]>,
    pub emp_trans1: Vec<f64>,
    pub emp_trans2: Vec<f64>,
    pub emp_cooccur: Vec<f64>,
    pub emp_marginal: Vec<f64>,
    /// Motifs émergents multi-octets (séquences d'indices symboliques, invariants à π)
    pub motifs: Vec<Vec<usize>>,
    /// Spans des motifs découverts dans sym_seq : Vec<(start_pos, motif_id)>
    pub motif_occurrences: Vec<(usize, usize)>,
    pub motif_idf: Vec<f64>,
    /// Poids TF-IDF des motifs par clause (n_ctx * n_motifs)
    pub ctx_motif_weights: Vec<f64>,
    /// Matrice d'adjacence relationnelle NON-SÉQUENTIELLE entre clauses (n_ctx * n_ctx)
    pub ctx_adj: Vec<f64>,
    pub ctx_features: Vec<[f64; CTX_FEAT_DIM]>,
    pub ctx_sym_weights: Vec<f64>,
    pub ctx_spans: Vec<(usize, usize)>,
}

/// Découvre les frontières naturelles de clauses (sans délimiteur codé en dur),
/// avec détection renforcée des séparateurs récurrents (ex. doubles retours, ponctuations terminales).
fn discover_emergent_spans(
    sym_seq: &[usize],
    emp_trans1: &[f64],
    emp_marginal: &[f64],
    sym_features: &[[f64; SYM_FEAT_DIM]],
    k: usize,
) -> Vec<(usize, usize)> {
    let n = sym_seq.len();
    if n < 12 || k < 2 {
        return vec![(0, n.max(1))];
    }

    let mut sym_boundary_affinity = vec![0.0; k];
    for u in 0..k {
        let freq = emp_marginal[u];
        let h_out = sym_features[u][4];
        let h_in = sym_features[u][5];
        let asym = sym_features[u][7];
        let self_rep = sym_features[u][8];
        let cohesion = sym_features[u][12];
        // Un séparateur de paragraphe/clause a forte entropie gauche/droite, faible cohésion
        // interne, et apparaît parfois doublé ou en fin de motif.
        let rep_bonus = if self_rep > 0.05 && freq < 0.08 { 0.45 * self_rep } else { 0.0 };
        sym_boundary_affinity[u] =
            freq.sqrt() * (0.45 * h_out + 0.35 * h_in + 0.20 * asym) * (1.20 - 0.55 * cohesion) + rep_bonus;
    }

    let mut raw_scores = vec![0.0; n];
    for i in 1..(n - 1) {
        let u = sym_seq[i];
        let v = sym_seq[i + 1];
        let p_uv = emp_trans1[u * k + v].max(1e-6);
        let surprisal = (-p_uv.ln()).clamp(0.0, 6.0) / 6.0;

        let lo = i.saturating_sub(3);
        let hi = (i + 5).min(n);
        let left_slice = &sym_seq[lo..=i];
        let right_slice = &sym_seq[(i + 1)..hi];
        let mut overlap = 0.0;
        for &ls in left_slice {
            if right_slice.contains(&ls) {
                overlap += 1.0;
            }
        }
        let discont = 1.0 - (overlap / (left_slice.len().max(1) as f64));
        // Double répétition locale (ex. deux délimiteurs consécutifs)
        let double_sep = if u == sym_seq[i - 1] && emp_marginal[u] < 0.08 { 0.30 } else { 0.0 };
        raw_scores[i] = 0.55 * sym_boundary_affinity[u] + 0.25 * surprisal + 0.20 * discont + double_sep;
    }

    let mean = raw_scores[1..(n - 1)].iter().sum::<f64>() / ((n - 2).max(1) as f64);
    let var = raw_scores[1..(n - 1)]
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f64>()
        / ((n - 2).max(1) as f64);
    let threshold = mean + 0.90 * var.sqrt();

    // min_span bas (6..16) pour ne jamais avaler une requête courte dans le paragraphe précédent
    let min_span = (n / 120).clamp(6, 16);
    let mut peaks: Vec<(f64, usize)> = Vec::new();
    for i in min_span..(n.saturating_sub(min_span)) {
        if raw_scores[i] > threshold
            && raw_scores[i] >= raw_scores[i - 1]
            && raw_scores[i] >= raw_scores[i + 1]
        {
            peaks.push((raw_scores[i], i + 1));
        }
    }
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut cuts: Vec<usize> = Vec::with_capacity(MAX_CTX - 1);
    for &(_, idx) in &peaks {
        if cuts.len() >= MAX_CTX - 1 {
            break;
        }
        if !cuts.iter().any(|&c| c.abs_diff(idx) < min_span) {
            cuts.push(idx);
        }
    }

    if cuts.len() < 2 {
        cuts.clear();
        let segs = 6.min(n / 6).max(1);
        for c in 1..segs {
            cuts.push((c * n) / segs);
        }
    }

    cuts.sort_unstable();
    let mut spans = Vec::with_capacity(cuts.len() + 1);
    let mut prev = 0;
    for c in cuts {
        if c > prev {
            spans.push((prev, c));
            prev = c;
        }
    }
    if prev < n {
        spans.push((prev, n));
    }
    if spans.is_empty() {
        spans.push((0, n));
    }
    spans
}

/// Découvre des motifs multi-octets cohésifs par fusions itératives guidées par PMI
/// (100 % invariant à toute permutation bijective π des octets).
fn discover_emergent_motifs(
    sym_seq: &[usize],
    k: usize,
    ctx_spans: &[(usize, usize)],
) -> (Vec<Vec<usize>>, Vec<(usize, usize)>) {
    let n = sym_seq.len();
    if n < 12 || k < 2 {
        return (Vec::new(), Vec::new());
    }

    // Chaque position porte (unit_id, start_in_sym_seq, len_in_sym_seq)
    let mut units: Vec<Vec<usize>> = (0..k).map(|s| vec![s]).collect();
    let mut seq: Vec<(usize, usize)> = sym_seq.iter().enumerate().map(|(pos, &s)| (s, pos)).collect();

    // 4 passes de fusion cohésive (permet de découvrir des motifs jusqu'à 8-12 octets)
    let mut merged_motifs: Vec<Vec<usize>> = Vec::new();
    for pass in 0..4 {
        if seq.len() < 6 || merged_motifs.len() >= MAX_MOTIFS {
            break;
        }
        let u_count = units.len();
        let mut counts = vec![0usize; u_count];
        for &(u, _) in &seq {
            counts[u] += 1;
        }

        let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
        for w in seq.windows(2) {
            // Évite de fusionner deux unités de longueur > 6 ensemble trop vite
            if units[w[0].0].len() + units[w[1].0].len() <= 12 {
                *pair_counts.entry((w[0].0, w[1].0)).or_insert(0) += 1;
            }
        }

        let seq_f = seq.len() as f64;
        let mut scored_pairs: Vec<(f64, (usize, usize), usize)> = Vec::new();
        let min_freq = if n > 1000 { 3 } else { 2 };

        for ((a, b), c_ab) in pair_counts {
            if c_ab < min_freq {
                continue;
            }
            let ca = counts[a] as f64;
            let cb = counts[b] as f64;
            let pmi = ((c_ab as f64) * seq_f / (ca * cb).max(1.0)).max(1e-6).ln();
            if pmi <= 0.4 {
                continue;
            }
            // Score de cohésion : PMI * sqrt(freq)
            let score = pmi * (c_ab as f64).sqrt();
            scored_pairs.push((score, (a, b), c_ab));
        }

        scored_pairs.sort_by(|x, y| y.0.total_cmp(&x.0));
        let top_k_merge = if pass < 2 { 18 } else { 14 };
        let mut merge_map: HashMap<(usize, usize), usize> = HashMap::new();

        for &(_, pair, _) in scored_pairs.iter().take(top_k_merge) {
            if merged_motifs.len() >= MAX_MOTIFS {
                break;
            }
            let mut combined = units[pair.0].clone();
            combined.extend_from_slice(&units[pair.1]);
            let new_id = units.len();
            units.push(combined.clone());
            merged_motifs.push(combined);
            merge_map.insert(pair, new_id);
        }

        if merge_map.is_empty() {
            break;
        }

        let mut next_seq = Vec::with_capacity(seq.len());
        let mut idx = 0;
        while idx < seq.len() {
            if idx + 1 < seq.len() {
                if let Some(&new_u) = merge_map.get(&(seq[idx].0, seq[idx + 1].0)) {
                    next_seq.push((new_u, seq[idx].1));
                    idx += 2;
                    continue;
                }
            }
            next_seq.push(seq[idx]);
            idx += 1;
        }
        seq = next_seq;
    }

    // Ajoute aussi les n-grammes récurrents multi-clauses (de longueur 2..6) qui apparaissent
    // dans >= 2 clauses distinctes afin d'assurer la couverture des ponts référentiels inter-clauses.
    if merged_motifs.len() < MAX_MOTIFS && ctx_spans.len() >= 2 {
        let mut clause_of_pos = vec![0usize; n];
        for (c_idx, &(st, en)) in ctx_spans.iter().enumerate() {
            for p in st..en.min(n) {
                clause_of_pos[p] = c_idx;
            }
        }
        let mut span_clauses: HashMap<&[usize], (usize, usize)> = HashMap::new();
        for len in [4usize, 3, 5, 2] {
            if n < len {
                continue;
            }
            span_clauses.clear();
            for pos in 0..=(n - len) {
                let slice = &sym_seq[pos..pos + len];
                let c = clause_of_pos[pos];
                span_clauses
                    .entry(slice)
                    .and_modify(|e| {
                        e.0 += 1;
                        if e.1 != c {
                            e.1 = usize::MAX; // vu dans >= 2 clauses distinctes
                        }
                    })
                    .or_insert((1, c));
            }
            let mut cross_candidates: Vec<(f64, Vec<usize>)> = span_clauses
                .iter()
                .filter(|&(_, &(cnt, marker))| marker == usize::MAX && cnt >= 2)
                .map(|(sl, &(cnt, _))| ((cnt as f64) * (len as f64).sqrt(), sl.to_vec()))
                .collect();
            cross_candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (_, cand) in cross_candidates {
                if merged_motifs.len() >= MAX_MOTIFS {
                    break;
                }
                if !merged_motifs.iter().any(|m| m == &cand) {
                    merged_motifs.push(cand);
                }
            }
        }
    }

    // Indexe toutes les occurrences des motifs retenus dans sym_seq
    let mut occs = Vec::new();
    for (mid, motif) in merged_motifs.iter().enumerate() {
        let mlen = motif.len();
        if mlen < 2 || mlen > n {
            continue;
        }
        for pos in 0..=(n - mlen) {
            if &sym_seq[pos..pos + mlen] == motif.as_slice() {
                occs.push((pos, mid));
            }
        }
    }
    occs.sort_unstable_by_key(|x| x.0);
    (merged_motifs, occs)
}

impl PerceivedGraph {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self {
                raw_len: 0,
                unique_bytes: vec![0],
                sym_seq: vec![0],
                sym_features: vec![[0.0; SYM_FEAT_DIM]],
                emp_trans1: vec![1.0],
                emp_trans2: vec![1.0],
                emp_cooccur: vec![1.0],
                emp_marginal: vec![1.0],
                motifs: Vec::new(),
                motif_occurrences: Vec::new(),
                motif_idf: Vec::new(),
                ctx_motif_weights: Vec::new(),
                ctx_adj: vec![1.0],
                ctx_features: vec![[0.0; CTX_FEAT_DIM]],
                ctx_sym_weights: vec![1.0],
                ctx_spans: vec![(0, 1)],
            };
        }

        let mut byte_to_sym = [usize::MAX; 256];
        let mut unique_bytes = Vec::with_capacity(64);
        let mut sym_seq = Vec::with_capacity(bytes.len());

        for &b in bytes {
            let idx = byte_to_sym[b as usize];
            if idx == usize::MAX {
                let new_idx = unique_bytes.len();
                byte_to_sym[b as usize] = new_idx;
                unique_bytes.push(b);
                sym_seq.push(new_idx);
            } else {
                sym_seq.push(idx);
            }
        }

        let k = unique_bytes.len();
        let n = sym_seq.len();
        let nf = n as f64;

        let mut counts = vec![0usize; k];
        let mut first_pos = vec![usize::MAX; k];
        let mut sum_pos = vec![0.0; k];
        let mut sum_pos2 = vec![0.0; k];
        let mut self_adj = vec![0usize; k];
        let mut self_skip2 = vec![0usize; k];

        for (pos, &s) in sym_seq.iter().enumerate() {
            counts[s] += 1;
            if first_pos[s] == usize::MAX {
                first_pos[s] = pos;
            }
            let p_norm = (pos as f64) / nf.max(1.0);
            sum_pos[s] += p_norm;
            sum_pos2[s] += p_norm * p_norm;
            if pos + 1 < n && sym_seq[pos + 1] == s {
                self_adj[s] += 1;
            }
            if pos + 2 < n && sym_seq[pos + 2] == s {
                self_skip2[s] += 1;
            }
        }

        let mut emp_marginal = vec![0.0; k];
        for s in 0..k {
            emp_marginal[s] = (counts[s] as f64) / nf;
        }
        normalize_prob(&mut emp_marginal);

        let mut trans1_raw = vec![0.0; k * k];
        let mut trans1_rev = vec![0.0; k * k];
        let mut trans2_raw = vec![0.0; k * k];
        let mut cooccur_raw = vec![0.0; k * k];

        for i in 0..n {
            let u = sym_seq[i];
            if i + 1 < n {
                let v = sym_seq[i + 1];
                trans1_raw[u * k + v] += 1.0;
                trans1_rev[v * k + u] += 1.0;
            }
            if i + 2 < n {
                let v2 = sym_seq[i + 2];
                trans2_raw[u * k + v2] += 1.0;
            }
            let lo = i.saturating_sub(3);
            let hi = (i + 4).min(n);
            for j in lo..hi {
                if i != j {
                    let vj = sym_seq[j];
                    let dist = (i as isize - j as isize).unsigned_abs() as f64;
                    cooccur_raw[u * k + vj] += 1.0 / dist;
                }
            }
        }

        let mut emp_trans1 = vec![0.0; k * k];
        let mut emp_trans2 = vec![0.0; k * k];
        let mut emp_cooccur = vec![0.0; k * k];
        let mut sym_features = vec![[0.0; SYM_FEAT_DIM]; k];
        let log_k = (k as f64).max(2.0).log2();

        for u in 0..k {
            let mut p_out = trans1_raw[u * k..(u + 1) * k].to_vec();
            normalize_prob(&mut p_out);
            let mut p_in = trans1_rev[u * k..(u + 1) * k].to_vec();
            normalize_prob(&mut p_in);
            let mut p_skip = trans2_raw[u * k..(u + 1) * k].to_vec();
            normalize_prob(&mut p_skip);
            let mut p_co = cooccur_raw[u * k..(u + 1) * k].to_vec();
            normalize_prob(&mut p_co);

            emp_trans1[u * k..(u + 1) * k].copy_from_slice(&p_out);
            emp_trans2[u * k..(u + 1) * k].copy_from_slice(&p_skip);
            emp_cooccur[u * k..(u + 1) * k].copy_from_slice(&p_co);

            let c = (counts[u] as f64).max(1.0);
            let mean_pos = sum_pos[u] / c;
            let var_pos = (sum_pos2[u] / c - mean_pos * mean_pos).max(0.0);
            let h_out = entropy_bits(&p_out) / log_k;
            let h_in = entropy_bits(&p_in) / log_k;
            let h_skip = entropy_bits(&p_skip) / log_k;

            let mut asym = 0.0;
            let mut top_out_mass = 0.0;
            let mut top_in_mass = 0.0;
            for v in 0..k {
                asym += (p_out[v] - p_in[v]).abs();
                if p_out[v] > top_out_mass {
                    top_out_mass = p_out[v];
                }
                if p_in[v] > top_in_mass {
                    top_in_mass = p_in[v];
                }
            }
            asym = (0.5 * asym).clamp(0.0, 1.0);

            let mut motif_cohesion = 0.0;
            for v in 0..k {
                let p_joint = emp_marginal[u] * p_out[v];
                if p_joint > 1e-6 && emp_marginal[v] > 1e-6 {
                    let pmi = (p_out[v] / emp_marginal[v]).max(1e-6).ln().clamp(0.0, 4.0) / 4.0;
                    motif_cohesion += p_out[v] * pmi;
                }
            }

            let strong_branches = p_out.iter().filter(|&&p| p > 1.5 / (k as f64).max(2.0)).count() as f64;
            let branch_mult = (strong_branches / (k as f64).max(2.0)).clamp(0.0, 1.0);

            let period_est = if counts[u] > 1 {
                let mut sum_gap = 0.0;
                let mut prev = None;
                let mut gaps = 0.0;
                for (pos, &s) in sym_seq.iter().enumerate() {
                    if s == u {
                        if let Some(pr) = prev {
                            sum_gap += (pos - pr) as f64;
                            gaps += 1.0;
                        }
                        prev = Some(pos);
                    }
                }
                if gaps > 0.0 { (sum_gap / gaps / nf.max(1.0)).clamp(0.0, 1.0) } else { 1.0 }
            } else {
                1.0
            };

            sym_features[u] = [
                (c / nf).sqrt().clamp(0.0, 1.0),
                (first_pos[u] as f64) / nf.max(1.0),
                mean_pos.clamp(0.0, 1.0),
                var_pos.sqrt().clamp(0.0, 1.0),
                h_out.clamp(0.0, 1.0),
                h_in.clamp(0.0, 1.0),
                h_skip.clamp(0.0, 1.0),
                asym,
                ((self_adj[u] as f64) / c).clamp(0.0, 1.0),
                ((self_skip2[u] as f64) / c).clamp(0.0, 1.0),
                period_est,
                (top_out_mass - top_in_mass).clamp(-1.0, 1.0),
                motif_cohesion.clamp(0.0, 1.0),
                branch_mult,
            ];
        }

        let ctx_spans = discover_emergent_spans(&sym_seq, &emp_trans1, &emp_marginal, &sym_features, k);
        let n_ctx = ctx_spans.len();
        let (motifs, motif_occurrences) = discover_emergent_motifs(&sym_seq, k, &ctx_spans);
        let n_motifs = motifs.len();

        let mut ctx_features = vec![[0.0; CTX_FEAT_DIM]; n_ctx];
        let mut ctx_sym_weights = vec![0.0; n_ctx * k];
        let mut ctx_motif_raw = vec![0.0; n_ctx * n_motifs.max(1)];
        let mut motif_df = vec![0usize; n_motifs.max(1)];

        // Assigne les occurrences de motifs aux clauses
        if n_motifs > 0 {
            let mut c_ptr = 0usize;
            for &(pos, mid) in &motif_occurrences {
                while c_ptr + 1 < n_ctx && pos >= ctx_spans[c_ptr].1 {
                    c_ptr += 1;
                }
                if pos >= ctx_spans[c_ptr].0 && pos < ctx_spans[c_ptr].1 {
                    let mlen = motifs[mid].len() as f64;
                    ctx_motif_raw[c_ptr * n_motifs + mid] += mlen.sqrt();
                }
            }
            for mid in 0..n_motifs {
                let mut df = 0usize;
                for c in 0..n_ctx {
                    if ctx_motif_raw[c * n_motifs + mid] > 0.0 {
                        df += 1;
                    }
                }
                motif_df[mid] = df;
            }
        }

        let mut motif_idf = vec![0.0; n_motifs];
        for mid in 0..n_motifs {
            motif_idf[mid] = (((n_ctx as f64) + 1.0) / ((motif_df[mid] as f64) + 1.0)).ln().max(0.1);
        }

        let mut ctx_motif_weights = vec![0.0; n_ctx * n_motifs];
        if n_motifs > 0 {
            for c in 0..n_ctx {
                let mut norm2 = 0.0;
                for mid in 0..n_motifs {
                    let w = ctx_motif_raw[c * n_motifs + mid] * motif_idf[mid];
                    ctx_motif_weights[c * n_motifs + mid] = w;
                    norm2 += w * w;
                }
                let inv = if norm2 > 1e-12 { 1.0 / norm2.sqrt() } else { 0.0 };
                for mid in 0..n_motifs {
                    ctx_motif_weights[c * n_motifs + mid] *= inv;
                }
            }
        }

        // Caractéristiques intrinsèques des clauses (invariantes à la permutation de clauses)
        for (c_idx, &(start, end)) in ctx_spans.iter().enumerate() {
            let span = &sym_seq[start..end];
            let span_len = (span.len() as f64).max(1.0);
            let mut local_dist = vec![0.0; k];
            let mut local_trans_novelty = 0.0;
            let mut local_rep = 0.0;

            for (pos, &s) in span.iter().enumerate() {
                local_dist[s] += 1.0;
                if pos + 1 < span.len() {
                    let s2 = span[pos + 1];
                    if s == s2 {
                        local_rep += 1.0;
                    }
                    let p_emp = emp_trans1[s * k + s2];
                    local_trans_novelty += (-p_emp.max(1e-6).ln()).min(6.0) / 6.0;
                }
            }
            normalize_prob(&mut local_dist);
            ctx_sym_weights[c_idx * k..(c_idx + 1) * k].copy_from_slice(&local_dist);

            let local_ent = (entropy_bits(&local_dist) / log_k).clamp(0.0, 1.0);
            let unique_ratio = (local_dist.iter().filter(|&&x| x > 1.0 / (span_len * 2.0)).count() as f64
                / (k as f64).max(1.0))
            .clamp(0.0, 1.0);

            let global_overlap: f64 = local_dist.iter().zip(emp_marginal.iter()).map(|(a, b)| a.min(*b)).sum();

            let mut mean_asym = 0.0;
            let mut mean_branch = 0.0;
            for (s, &w) in local_dist.iter().enumerate() {
                mean_asym += w * sym_features[s][7];
                mean_branch += w * sym_features[s][13];
            }

            // Densité de motifs partagés à l'extérieur (valence relationnelle ouverte)
            let mut ext_pull = 0.0;
            if n_motifs > 0 {
                for mid in 0..n_motifs {
                    if ctx_motif_weights[c_idx * n_motifs + mid] > 0.0 && motif_df[mid] > 1 {
                        ext_pull += ctx_motif_weights[c_idx * n_motifs + mid] * motif_idf[mid];
                    }
                }
            }

            ctx_features[c_idx] = [
                (span_len / nf).clamp(0.0, 1.0),
                local_ent,
                unique_ratio,
                (local_trans_novelty / span_len).clamp(0.0, 1.0),
                (local_rep / span_len).clamp(0.0, 1.0),
                global_overlap.clamp(0.0, 1.0),
                mean_asym.clamp(0.0, 1.0),
                mean_branch.clamp(0.0, 1.0),
                (ext_pull / 3.0).clamp(0.0, 1.0),
                (1.0 / span_len.sqrt()).clamp(0.0, 1.0),
            ];
        }

        // Graphe relationnel NON-SÉQUENTIEL `ctx_adj` entre clauses (invariante à tout réordonnancement)
        let mut ctx_adj = vec![0.0; n_ctx * n_ctx];
        for i in 0..n_ctx {
            let mut row_sum = 0.0;
            for j in 0..n_ctx {
                if i == j {
                    continue;
                }
                let motif_sim = if n_motifs > 0 {
                    let mut s = 0.0;
                    for mid in 0..n_motifs {
                        s += ctx_motif_weights[i * n_motifs + mid] * ctx_motif_weights[j * n_motifs + mid];
                    }
                    s
                } else {
                    0.0
                };
                let sym_sim: f64 = ctx_sym_weights[i * k..(i + 1) * k]
                    .iter()
                    .zip(ctx_sym_weights[j * k..(j + 1) * k].iter())
                    .map(|(a, b)| a.min(*b))
                    .sum();
                let w = 0.75 * motif_sim + 0.25 * sym_sim + 1e-4;
                ctx_adj[i * n_ctx + j] = w;
                row_sum += w;
            }
            if row_sum > 1e-12 {
                let inv = 1.0 / row_sum;
                for j in 0..n_ctx {
                    ctx_adj[i * n_ctx + j] *= inv;
                }
            }
        }

        Self {
            raw_len: bytes.len(),
            unique_bytes,
            sym_seq,
            sym_features,
            emp_trans1,
            emp_trans2,
            emp_cooccur,
            emp_marginal,
            motifs,
            motif_occurrences,
            motif_idf,
            ctx_motif_weights,
            ctx_adj,
            ctx_features,
            ctx_sym_weights,
            ctx_spans,
        }
    }

    #[inline] pub fn k(&self) -> usize { self.unique_bytes.len() }
    #[inline] pub fn n_ctx(&self) -> usize { self.ctx_spans.len() }
    #[inline] pub fn n_motifs(&self) -> usize { self.motifs.len() }
}

#[derive(Clone, Debug)]
pub struct HypothesisState {
    pub k: usize,
    pub lat_states: [[f64; D]; N_LAT],
    pub ctx_states: Vec<[f64; D]>,
    pub sym_states: Vec<[f64; D]>,
    pub lat_couplings: [f64; N_LAT * N_LAT],
    pub lat_to_sym: Vec<f64>,
    pub sym_to_lat: Vec<f64>,
    pub lat_op_mods: [[f64; D]; N_LAT],
    pub sym_query_vecs: Vec<[f64; D]>,
    pub sym_key_vecs: Vec<[f64; D]>,
    pub sym_trans: Vec<f64>,
    pub search_depth: usize,
}

impl HypothesisState {
    pub fn trivial_identity_from(pg: &PerceivedGraph) -> Self {
        let k = pg.k();
        let mut lat_states = [[0.0; D]; N_LAT];
        for i in 0..N_LAT {
            lat_states[i][0] = 0.05;
        }
        let mut lat_couplings = [0.0; N_LAT * N_LAT];
        for i in 0..N_LAT {
            lat_couplings[i * N_LAT + i] = 0.1;
        }
        let mut sym_states = vec![[0.0; D]; k];
        for s in 0..k {
            for d in 0..SYM_FEAT_DIM.min(D) {
                sym_states[s][d] = pg.sym_features[s][d];
            }
        }
        Self {
            k,
            lat_states,
            ctx_states: vec![[0.0; D]; pg.n_ctx()],
            sym_states,
            lat_couplings,
            lat_to_sym: vec![1.0 / (k as f64).max(1.0); N_LAT * k],
            sym_to_lat: vec![1.0 / (N_LAT as f64); k * N_LAT],
            lat_op_mods: [[0.0; D]; N_LAT],
            sym_query_vecs: vec![[0.0; D]; k],
            sym_key_vecs: vec![[0.0; D]; k],
            sym_trans: pg.emp_trans1.clone(),
            search_depth: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PhisharpNet {
    pub w_sym_enc: Vec<f64>,
    pub w_ctx_enc: Vec<f64>,
    pub w_lat_bias: Vec<f64>,
    pub w_fwd_effect: Vec<f64>,
    pub w_bwd_cause: Vec<f64>,
    pub w_rel_lat_fwd: Vec<f64>,
    pub w_rel_lat_bwd: Vec<f64>,
    pub w_ctx_to_lat: Vec<f64>,
    pub w_lat_to_ctx: Vec<f64>,
    pub w_edge_q: Vec<f64>,
    pub w_edge_k: Vec<f64>,
    pub w_upd_gate: Vec<f64>,
    pub w_upd_cand: Vec<f64>,
    pub w_res_effect: Vec<f64>,
    pub w_res_cause: Vec<f64>,
    pub w_res_to_lat: Vec<f64>,
    pub w_op_mod: Vec<f64>,
    pub w_policy: Vec<f64>,
    pub w_read_query: Vec<f64>,
    pub w_read_key: Vec<f64>,
    pub w_temp_logits: Vec<f64>,
}

impl PhisharpNet {
    pub fn new_random(rng: &mut Rng) -> Self {
        let init = |r: &mut Rng, len: usize, scale: f64| -> Vec<f64> {
            (0..len).map(|_| r.next_normal() * scale).collect()
        };
        let s_d = (1.0 / (D as f64)).sqrt();
        let s_3d = (1.0 / ((3 * D) as f64)).sqrt();
        Self {
            w_sym_enc: init(rng, D * SYM_FEAT_DIM, (1.0 / (SYM_FEAT_DIM as f64)).sqrt()),
            w_ctx_enc: init(rng, D * CTX_FEAT_DIM, (1.0 / (CTX_FEAT_DIM as f64)).sqrt()),
            w_lat_bias: init(rng, N_LAT * D, 0.45),
            w_fwd_effect: init(rng, D * D, s_d),
            w_bwd_cause: init(rng, D * D, s_d),
            w_rel_lat_fwd: init(rng, D * D, s_d),
            w_rel_lat_bwd: init(rng, D * D, s_d),
            w_ctx_to_lat: init(rng, D * D, s_d),
            w_lat_to_ctx: init(rng, D * D, s_d),
            w_edge_q: init(rng, D * D, s_d),
            w_edge_k: init(rng, D * D, s_d),
            w_upd_gate: init(rng, D * (3 * D), s_3d),
            w_upd_cand: init(rng, D * (3 * D), s_3d),
            w_res_effect: init(rng, D * D, s_d),
            w_res_cause: init(rng, D * D, s_d),
            w_res_to_lat: init(rng, D * D, s_d),
            w_op_mod: init(rng, D * D, s_d * 1.4),
            w_policy: init(rng, POLICY_OUT_DIM * POLICY_IN_DIM, (1.0 / (POLICY_IN_DIM as f64)).sqrt()),
            w_read_query: init(rng, D * D, s_d),
            w_read_key: init(rng, D * D, s_d),
            w_temp_logits: vec![1.6, 1.5, 1.8, 2.2],
        }
    }

    pub fn param_count(&self) -> usize {
        self.w_sym_enc.len()
            + self.w_ctx_enc.len()
            + self.w_lat_bias.len()
            + self.w_fwd_effect.len()
            + self.w_bwd_cause.len()
            + self.w_rel_lat_fwd.len()
            + self.w_rel_lat_bwd.len()
            + self.w_ctx_to_lat.len()
            + self.w_lat_to_ctx.len()
            + self.w_edge_q.len()
            + self.w_edge_k.len()
            + self.w_upd_gate.len()
            + self.w_upd_cand.len()
            + self.w_res_effect.len()
            + self.w_res_cause.len()
            + self.w_res_to_lat.len()
            + self.w_op_mod.len()
            + self.w_policy.len()
            + self.w_read_query.len()
            + self.w_read_key.len()
            + self.w_temp_logits.len()
    }

    pub fn to_flat_vec(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.param_count());
        v.extend_from_slice(&self.w_sym_enc);
        v.extend_from_slice(&self.w_ctx_enc);
        v.extend_from_slice(&self.w_lat_bias);
        v.extend_from_slice(&self.w_fwd_effect);
        v.extend_from_slice(&self.w_bwd_cause);
        v.extend_from_slice(&self.w_rel_lat_fwd);
        v.extend_from_slice(&self.w_rel_lat_bwd);
        v.extend_from_slice(&self.w_ctx_to_lat);
        v.extend_from_slice(&self.w_lat_to_ctx);
        v.extend_from_slice(&self.w_edge_q);
        v.extend_from_slice(&self.w_edge_k);
        v.extend_from_slice(&self.w_upd_gate);
        v.extend_from_slice(&self.w_upd_cand);
        v.extend_from_slice(&self.w_res_effect);
        v.extend_from_slice(&self.w_res_cause);
        v.extend_from_slice(&self.w_res_to_lat);
        v.extend_from_slice(&self.w_op_mod);
        v.extend_from_slice(&self.w_policy);
        v.extend_from_slice(&self.w_read_query);
        v.extend_from_slice(&self.w_read_key);
        v.extend_from_slice(&self.w_temp_logits);
        v
    }

    pub fn from_flat_vec(&mut self, v: &[f64]) {
        assert_eq!(v.len(), self.param_count());
        let mut off = 0;
        let mut take = |dst: &mut [f64]| {
            dst.copy_from_slice(&v[off..off + dst.len()]);
            off += dst.len();
        };
        take(&mut self.w_sym_enc);
        take(&mut self.w_ctx_enc);
        take(&mut self.w_lat_bias);
        take(&mut self.w_fwd_effect);
        take(&mut self.w_bwd_cause);
        take(&mut self.w_rel_lat_fwd);
        take(&mut self.w_rel_lat_bwd);
        take(&mut self.w_ctx_to_lat);
        take(&mut self.w_lat_to_ctx);
        take(&mut self.w_edge_q);
        take(&mut self.w_edge_k);
        take(&mut self.w_upd_gate);
        take(&mut self.w_upd_cand);
        take(&mut self.w_res_effect);
        take(&mut self.w_res_cause);
        take(&mut self.w_res_to_lat);
        take(&mut self.w_op_mod);
        take(&mut self.w_policy);
        take(&mut self.w_read_query);
        take(&mut self.w_read_key);
        take(&mut self.w_temp_logits);
    }

    pub fn non_reciprocity_index(&self) -> f64 {
        let mut diff2 = 0.0;
        let mut norm2 = 0.0;
        for r in 0..D {
            for c in 0..D {
                let a = self.w_fwd_effect[r * D + c];
                let b = self.w_bwd_cause[c * D + r];
                diff2 += (a - b) * (a - b);
                norm2 += a * a + b * b;
            }
        }
        if norm2 < 1e-12 { 0.0 } else { (diff2.sqrt() / norm2.sqrt()).clamp(0.0, 1.0) }
    }

    #[inline]
    pub fn temp_scales(&self) -> [f64; 4] {
        [
            (1.0 + softplus(self.w_temp_logits[0])).clamp(1.0, 9.0),
            (1.0 + softplus(self.w_temp_logits[1])).clamp(1.0, 9.0),
            (1.0 + softplus(self.w_temp_logits[2])).clamp(1.0, 9.0),
            (1.0 + softplus(self.w_temp_logits[3])).clamp(1.0, 10.0),
        ]
    }

    pub fn init_regional_state(&self, pg: &PerceivedGraph) -> HypothesisState {
        let k = pg.k();
        let mut sym_states = vec![[0.0; D]; k];
        for s in 0..k {
            let mut buf = [0.0; D];
            matvec(&self.w_sym_enc, D, SYM_FEAT_DIM, &pg.sym_features[s], &mut buf);
            for d in 0..D {
                sym_states[s][d] = tanh_clamped(buf[d]);
            }
        }

        let n_ctx = pg.n_ctx();
        let mut ctx_states = vec![[0.0; D]; n_ctx];
        for c in 0..n_ctx {
            let mut buf = [0.0; D];
            matvec(&self.w_ctx_enc, D, CTX_FEAT_DIM, &pg.ctx_features[c], &mut buf);
            for d in 0..D {
                ctx_states[c][d] = tanh_clamped(buf[d]);
            }
        }

        let mut lat_states = [[0.0; D]; N_LAT];
        for i in 0..N_LAT {
            for d in 0..D {
                lat_states[i][d] = tanh_clamped(self.w_lat_bias[i * D + d]);
            }
        }

        let mut h = HypothesisState {
            k,
            lat_states,
            ctx_states,
            sym_states,
            lat_couplings: [0.0; N_LAT * N_LAT],
            lat_to_sym: vec![0.0; N_LAT * k],
            sym_to_lat: vec![0.0; k * N_LAT],
            lat_op_mods: [[0.0; D]; N_LAT],
            sym_query_vecs: vec![[0.0; D]; k],
            sym_key_vecs: vec![[0.0; D]; k],
            sym_trans: vec![1.0 / (k as f64).max(1.0); k * k],
            search_depth: 0,
        };
        self.recompute_relational_operators(&mut h, pg, false);
        h
    }

    pub fn recompute_relational_operators(
        &self,
        h: &mut HypothesisState,
        _pg: &PerceivedGraph,
        active_causal_bottleneck: bool,
    ) {
        let k = h.k;
        let [b_s2l, b_lat, b_l2s, b_out] = self.temp_scales();
        let base_scale = 1.0 / (D as f64).sqrt();

        let mut q_lat = [[0.0; D]; N_LAT];
        let mut k_lat = [[0.0; D]; N_LAT];
        for i in 0..N_LAT {
            matvec(&self.w_edge_q, D, D, &h.lat_states[i], &mut q_lat[i]);
            matvec(&self.w_edge_k, D, D, &h.lat_states[i], &mut k_lat[i]);
            let mut mod_buf = [0.0; D];
            matvec(&self.w_op_mod, D, D, &h.lat_states[i], &mut mod_buf);
            for d in 0..D {
                h.lat_op_mods[i][d] = 1.5 * tanh_clamped(mod_buf[d]);
            }
        }

        for i in 0..N_LAT {
            for j in 0..N_LAT {
                let mut dot_ij = 0.0;
                for d in 0..D {
                    dot_ij += q_lat[i][d] * k_lat[j][d];
                }
                let val = tanh_clamped(dot_ij * base_scale * 1.5);
                h.lat_couplings[i * N_LAT + j] = if i == j { val * 0.2 } else { val };
            }
        }

        for s in 0..k {
            matvec(&self.w_read_query, D, D, &h.sym_states[s], &mut h.sym_query_vecs[s]);
            matvec(&self.w_read_key, D, D, &h.sym_states[s], &mut h.sym_key_vecs[s]);
        }

        for s in 0..k {
            let mut max_sc = f64::NEG_INFINITY;
            let mut scores = [0.0; N_LAT];
            for l in 0..N_LAT {
                let mut sc = 0.0;
                for d in 0..D {
                    sc += h.sym_query_vecs[s][d] * k_lat[l][d];
                }
                let v = sc * base_scale * b_s2l;
                scores[l] = v;
                if v > max_sc {
                    max_sc = v;
                }
            }
            let mut row_sum = 0.0;
            for l in 0..N_LAT {
                let w = (scores[l] - max_sc).clamp(-12.0, 0.0).exp();
                h.sym_to_lat[s * N_LAT + l] = w;
                row_sum += w;
            }
            let inv = 1.0 / row_sum.max(1e-12);
            for l in 0..N_LAT {
                h.sym_to_lat[s * N_LAT + l] *= inv;
            }
        }

        for l in 0..N_LAT {
            let mut max_sc = f64::NEG_INFINITY;
            let mut scores = vec![0.0; k];
            for s in 0..k {
                let mut sc = 0.0;
                for d in 0..D {
                    sc += q_lat[l][d] * h.sym_key_vecs[s][d];
                }
                let v = sc * base_scale * b_l2s;
                scores[s] = v;
                if v > max_sc {
                    max_sc = v;
                }
            }
            let mut row_sum = 0.0;
            for s in 0..k {
                let w = (scores[s] - max_sc).clamp(-12.0, 0.0).exp();
                h.lat_to_sym[l * k + s] = w;
                row_sum += w;
            }
            let inv = 1.0 / row_sum.max(1e-12);
            for s in 0..k {
                h.lat_to_sym[l * k + s] *= inv;
            }
        }

        if !active_causal_bottleneck {
            let u = 1.0 / (k as f64).max(1.0);
            for x in h.sym_trans.iter_mut() {
                *x = u;
            }
            return;
        }

        let mut lat_trans = [0.0; N_LAT * N_LAT];
        for i in 0..N_LAT {
            let mut max_v = f64::NEG_INFINITY;
            for j in 0..N_LAT {
                let v = b_lat * h.lat_couplings[i * N_LAT + j];
                if v > max_v {
                    max_v = v;
                }
            }
            let mut sum = 0.0;
            for j in 0..N_LAT {
                let w = (b_lat * h.lat_couplings[i * N_LAT + j] - max_v).clamp(-12.0, 0.0).exp();
                lat_trans[i * N_LAT + j] = w;
                sum += w;
            }
            let inv = 1.0 / sum.max(1e-12);
            for j in 0..N_LAT {
                lat_trans[i * N_LAT + j] *= inv;
            }
        }

        for u in 0..k {
            let mut lat_after = [0.0; N_LAT];
            for j in 0..N_LAT {
                let mut acc = 0.0;
                for i in 0..N_LAT {
                    acc += h.sym_to_lat[u * N_LAT + i] * lat_trans[i * N_LAT + j];
                }
                lat_after[j] = acc;
            }

            let mut eff_op = [0.0; D];
            for j in 0..N_LAT {
                let w = lat_after[j];
                for d in 0..D {
                    eff_op[d] += w * h.lat_op_mods[j][d];
                }
            }
            let mut eff_vec = [0.0; D];
            for d in 0..D {
                eff_vec[d] = eff_op[d] * h.sym_query_vecs[u][d];
            }

            let mut logits = vec![0.0; k];
            let mut max_logit = f64::NEG_INFINITY;
            for v in 0..k {
                let mut p_uv = 0.0;
                for j in 0..N_LAT {
                    p_uv += lat_after[j] * h.lat_to_sym[j * k + v];
                }
                let mut bil = 0.0;
                for d in 0..D {
                    bil += eff_vec[d] * h.sym_key_vecs[v][d];
                }
                let l_uv = 0.5 * b_out * p_uv.max(1e-9).ln() + b_out * (bil * base_scale);
                logits[v] = l_uv;
                if l_uv > max_logit {
                    max_logit = l_uv;
                }
            }

            let mut row_sum = 0.0;
            for v in 0..k {
                let w = (logits[v] - max_logit).clamp(-14.0, 0.0).exp().max(1e-9);
                h.sym_trans[u * k + v] = w;
                row_sum += w;
            }
            let inv = 1.0 / row_sum.max(1e-12);
            for v in 0..k {
                h.sym_trans[u * k + v] *= inv;
            }
        }
    }

    pub fn propagate_step(&self, h: &mut HypothesisState, pg: &PerceivedGraph, step_gain: f64) {
        let k = h.k;
        let n_ctx = h.ctx_states.len();
        let gain = step_gain.clamp(0.05, 1.0);

        let mut msg_eff = [0.0; D];
        let mut msg_cau = [0.0; D];
        let mut msg_lat = [0.0; D];
        let mut concat = [0.0; 3 * D];
        let mut gate = [0.0; D];
        let mut cand = [0.0; D];

        let mut new_sym = vec![[0.0; D]; k];
        for u in 0..k {
            let mut agg_fwd = [0.0; D];
            let mut agg_bwd = [0.0; D];
            let mut res_eff_vec = [0.0; D];
            let mut res_cau_vec = [0.0; D];

            for v in 0..k {
                let w_out = pg.emp_trans1[u * k + v];
                let w_in = pg.emp_trans1[v * k + u];
                let gap_eff = w_out - h.sym_trans[u * k + v];
                let gap_cau = w_in - h.sym_trans[v * k + u];
                for d in 0..D {
                    agg_fwd[d] += w_out * h.sym_states[v][d];
                    agg_bwd[d] += w_in * h.sym_states[v][d];
                    res_eff_vec[d] += gap_eff * h.sym_states[v][d];
                    res_cau_vec[d] += gap_cau * h.sym_states[v][d];
                }
            }
            matvec(&self.w_fwd_effect, D, D, &agg_fwd, &mut msg_eff);
            matvec(&self.w_bwd_cause, D, D, &agg_bwd, &mut msg_cau);

            let mut res_e_proj = [0.0; D];
            let mut res_c_proj = [0.0; D];
            matvec(&self.w_res_effect, D, D, &res_eff_vec, &mut res_e_proj);
            matvec(&self.w_res_cause, D, D, &res_cau_vec, &mut res_c_proj);

            let mut agg_lat = [0.0; D];
            for l in 0..N_LAT {
                let w = h.sym_to_lat[u * N_LAT + l];
                for d in 0..D {
                    agg_lat[d] += w * h.lat_states[l][d];
                }
            }
            matvec(&self.w_lat_to_ctx, D, D, &agg_lat, &mut msg_lat);

            for d in 0..D {
                concat[d] = msg_eff[d] + 0.80 * res_e_proj[d];
                concat[D + d] = msg_cau[d] + 0.80 * res_c_proj[d];
                concat[2 * D + d] = msg_lat[d];
            }
            matvec(&self.w_upd_gate, D, 3 * D, &concat, &mut gate);
            matvec(&self.w_upd_cand, D, 3 * D, &concat, &mut cand);
            for d in 0..D {
                let z = sigmoid(gate[d]) * gain;
                new_sym[u][d] = (1.0 - z) * h.sym_states[u][d] + z * tanh_clamped(cand[d]);
            }
        }
        h.sym_states = new_sym;

        // Mise à jour NON-SÉQUENTIELLE des clauses via `pg.ctx_adj` (invariante à l'ordre des clauses)
        let mut new_ctx = vec![[0.0; D]; n_ctx];
        for c in 0..n_ctx {
            let mut agg_sym = [0.0; D];
            for s in 0..k {
                let w = pg.ctx_sym_weights[c * k + s];
                for d in 0..D {
                    agg_sym[d] += w * h.sym_states[s][d];
                }
            }
            let mut agg_out_ctx = [0.0; D];
            let mut agg_in_ctx = [0.0; D];
            for other in 0..n_ctx {
                if other == c {
                    continue;
                }
                let w_out = pg.ctx_adj[c * n_ctx + other];
                let w_in = pg.ctx_adj[other * n_ctx + c];
                for d in 0..D {
                    agg_out_ctx[d] += w_out * h.ctx_states[other][d];
                    agg_in_ctx[d] += w_in * h.ctx_states[other][d];
                }
            }

            matvec(&self.w_fwd_effect, D, D, &agg_out_ctx, &mut msg_eff);
            matvec(&self.w_bwd_cause, D, D, &agg_in_ctx, &mut msg_cau);
            matvec(&self.w_ctx_to_lat, D, D, &agg_sym, &mut msg_lat);

            for d in 0..D {
                concat[d] = msg_eff[d];
                concat[D + d] = msg_cau[d];
                concat[2 * D + d] = msg_lat[d];
            }
            matvec(&self.w_upd_gate, D, 3 * D, &concat, &mut gate);
            matvec(&self.w_upd_cand, D, 3 * D, &concat, &mut cand);
            for d in 0..D {
                let z = sigmoid(gate[d]) * gain;
                new_ctx[c][d] = (1.0 - z) * h.ctx_states[c][d] + z * tanh_clamped(cand[d]);
            }
        }
        h.ctx_states = new_ctx;

        let mut sym_res_by_lat = [[0.0; D]; N_LAT];
        for u in 0..k {
            let m_u = pg.emp_marginal[u];
            let mut res_u = [0.0; D];
            for v in 0..k {
                let err = pg.emp_trans1[u * k + v] - h.sym_trans[u * k + v];
                for d in 0..D {
                    res_u[d] += err * h.sym_states[v][d];
                }
            }
            for i in 0..N_LAT {
                let w = m_u * h.sym_to_lat[u * N_LAT + i];
                for d in 0..D {
                    sym_res_by_lat[i][d] += w * res_u[d];
                }
            }
        }

        let mut new_lat = [[0.0; D]; N_LAT];
        for i in 0..N_LAT {
            let mut agg_out = [0.0; D];
            let mut agg_in = [0.0; D];
            let mut sum_out = 1e-6;
            let mut sum_in = 1e-6;
            for j in 0..N_LAT {
                let w_ij = h.lat_couplings[i * N_LAT + j].abs() + 0.05;
                let w_ji = h.lat_couplings[j * N_LAT + i].abs() + 0.05;
                sum_out += w_ij;
                sum_in += w_ji;
                for d in 0..D {
                    agg_out[d] += h.lat_couplings[i * N_LAT + j] * h.lat_states[j][d];
                    agg_in[d] += h.lat_couplings[j * N_LAT + i] * h.lat_states[j][d];
                }
            }
            for d in 0..D {
                agg_out[d] /= sum_out;
                agg_in[d] /= sum_in;
            }

            let mut agg_ctx = [0.0; D];
            for c in 0..n_ctx {
                for d in 0..D {
                    agg_ctx[d] += h.ctx_states[c][d];
                }
            }
            for d in 0..D {
                agg_ctx[d] /= (n_ctx.max(1)) as f64;
            }

            let mut res_lat_proj = [0.0; D];
            matvec(&self.w_res_to_lat, D, D, &sym_res_by_lat[i], &mut res_lat_proj);

            matvec(&self.w_rel_lat_fwd, D, D, &agg_out, &mut msg_eff);
            matvec(&self.w_rel_lat_bwd, D, D, &agg_in, &mut msg_cau);
            matvec(&self.w_ctx_to_lat, D, D, &agg_ctx, &mut msg_lat);

            for d in 0..D {
                concat[d] = msg_eff[d];
                concat[D + d] = msg_cau[d];
                concat[2 * D + d] = msg_lat[d] + 0.90 * res_lat_proj[d];
            }
            matvec(&self.w_upd_gate, D, 3 * D, &concat, &mut gate);
            matvec(&self.w_upd_cand, D, 3 * D, &concat, &mut cand);
            for d in 0..D {
                let z = sigmoid(gate[d]) * gain;
                new_lat[i][d] = (1.0 - z) * h.lat_states[i][d] + z * tanh_clamped(cand[d]);
            }
        }
        h.lat_states = new_lat;
        h.search_depth += 1;
        self.recompute_relational_operators(h, pg, true);
    }

    pub fn policy_select(&self, features: &[f64; POLICY_IN_DIM]) -> [f64; POLICY_OUT_DIM] {
        let mut out = [0.0; POLICY_OUT_DIM];
        matvec(&self.w_policy, POLICY_OUT_DIM, POLICY_IN_DIM, features, &mut out);
        [
            sigmoid(out[0]),
            sigmoid(out[1]),
            sigmoid(out[2]),
            sigmoid(out[3]),
        ]
    }

    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);
        w.write_all(b"PHIS0004")?;
        let flat = self.to_flat_vec();
        let len_u64 = flat.len() as u64;
        w.write_all(&len_u64.to_le_bytes())?;
        for val in flat {
            w.write_all(&val.to_le_bytes())?;
        }
        w.flush()?;
        Ok(())
    }

    pub fn load_from_file(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != b"PHIS0004" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "En-tête .phis invalide (attendu PHIS0004)",
            ));
        }
        let mut len_buf = [0u8; 8];
        r.read_exact(&mut len_buf)?;
        let n = u64::from_le_bytes(len_buf) as usize;
        let mut rng = Rng::new(1);
        let mut net = Self::new_random(&mut rng);
        if n != net.param_count() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Nombre de paramètres incompatible dans .phis",
            ));
        }
        let mut flat = Vec::with_capacity(n);
        let mut fbuf = [0u8; 8];
        for _ in 0..n {
            r.read_exact(&mut fbuf)?;
            flat.push(f64::from_le_bytes(fbuf));
        }
        net.from_flat_vec(&flat);
        Ok(net)
    }
}