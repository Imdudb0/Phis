//! Substrat connexionniste 100% relationnel à matrices de poids fondamentalement non-réciproques.
//! v0.2.0 : Températures relationnelles apprises pour permettre des rôles causaux nets sans
//! écraser sym_trans vers l'uniforme, tout en conservant l'invariance bijective exacte.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::math::{Rng, entropy_bits, matvec, normalize_prob, sigmoid, softplus, tanh_clamped};

pub const D: usize = 12;           // Dimension d'état par nœud
pub const N_LAT: usize = 10;       // Nombre de nœuds causaux latents
pub const N_CTX: usize = 12;       // Nombre de nœuds de segments contextuels
pub const SYM_FEAT_DIM: usize = 12;
pub const CTX_FEAT_DIM: usize = 10;
pub const POLICY_IN_DIM: usize = 20;
pub const POLICY_OUT_DIM: usize = 4;

/// Représentation purement relationnelle de E = perceived.eff.
/// Strictement invariante sous renommage bijectif d'octets.
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
    pub ctx_features: Vec<[f64; CTX_FEAT_DIM]>,
    pub ctx_sym_weights: Vec<f64>,
    pub ctx_spans: Vec<(usize, usize)>,
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
                ctx_features: vec![[0.0; CTX_FEAT_DIM]; N_CTX],
                ctx_sym_weights: vec![1.0; N_CTX],
                ctx_spans: vec![(0, 1); N_CTX],
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
            for v in 0..k {
                asym += (p_out[v] - p_in[v]).abs();
            }
            asym = (0.5 * asym).clamp(0.0, 1.0);

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

            let max_out = p_out.iter().copied().fold(0.0_f64, f64::max);
            let max_in = p_in.iter().copied().fold(0.0_f64, f64::max);

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
                (max_out - max_in).clamp(-1.0, 1.0),
            ];
        }

        let mut ctx_features = vec![[0.0; CTX_FEAT_DIM]; N_CTX];
        let mut ctx_sym_weights = vec![0.0; N_CTX * k];
        let mut ctx_spans = Vec::with_capacity(N_CTX);

        for c_idx in 0..N_CTX {
            let start = (c_idx * n) / N_CTX;
            let end = (((c_idx + 1) * n) / N_CTX).max(start + 1).min(n);
            ctx_spans.push((start, end));

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

            let prev_overlap = if c_idx > 0 {
                let prev_slice = &ctx_sym_weights[(c_idx - 1) * k..c_idx * k];
                local_dist.iter().zip(prev_slice.iter()).map(|(a, b)| a.min(*b)).sum::<f64>()
            } else {
                0.0
            };
            let global_overlap: f64 = local_dist.iter().zip(emp_marginal.iter()).map(|(a, b)| a.min(*b)).sum();

            let mut mean_asym = 0.0;
            let mut mean_hout = 0.0;
            for (s, &w) in local_dist.iter().enumerate() {
                mean_asym += w * sym_features[s][7];
                mean_hout += w * sym_features[s][4];
            }

            let pos_norm = (c_idx as f64) / ((N_CTX - 1) as f64);
            ctx_features[c_idx] = [
                pos_norm,
                (span_len / nf).clamp(0.0, 1.0),
                local_ent,
                unique_ratio,
                (local_trans_novelty / span_len).clamp(0.0, 1.0),
                (local_rep / span_len).clamp(0.0, 1.0),
                prev_overlap.clamp(0.0, 1.0),
                global_overlap.clamp(0.0, 1.0),
                mean_asym.clamp(0.0, 1.0),
                mean_hout.clamp(0.0, 1.0),
            ];
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
            ctx_features,
            ctx_sym_weights,
            ctx_spans,
        }
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.unique_bytes.len()
    }
}

#[derive(Clone, Debug)]
pub struct HypothesisState {
    pub k: usize,
    pub lat_states: [[f64; D]; N_LAT],
    pub ctx_states: [[f64; D]; N_CTX],
    pub sym_states: Vec<[f64; D]>,
    pub lat_couplings: [f64; N_LAT * N_LAT],
    pub lat_to_sym: Vec<f64>, // N_LAT * K
    pub sym_to_lat: Vec<f64>, // K * N_LAT
    /// Biais relationnel appris entre paires de symboles (via projections non-réciproques)
    pub sym_rel_bias: Vec<f64>, // K * K
    pub sym_trans: Vec<f64>,    // K * K
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
            ctx_states: [[0.0; D]; N_CTX],
            sym_states,
            lat_couplings,
            lat_to_sym: vec![1.0 / (k as f64).max(1.0); N_LAT * k],
            sym_to_lat: vec![1.0 / (N_LAT as f64); k * N_LAT],
            sym_rel_bias: vec![0.0; k * k],
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
    pub w_policy: Vec<f64>,
    pub w_read_query: Vec<f64>,
    pub w_read_key: Vec<f64>,
    /// Paramètres d'échelle de sélectivité (inverse-températures) : [beta_s2l, beta_lat, beta_l2s, beta_out]
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
            w_policy: init(rng, POLICY_OUT_DIM * POLICY_IN_DIM, (1.0 / (POLICY_IN_DIM as f64)).sqrt()),
            w_read_query: init(rng, D * D, s_d),
            w_read_key: init(rng, D * D, s_d),
            w_temp_logits: vec![1.8, 1.6, 2.0, 2.2],
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
            (1.0 + softplus(self.w_temp_logits[0])).clamp(1.0, 8.0),
            (1.0 + softplus(self.w_temp_logits[1])).clamp(1.0, 8.0),
            (1.0 + softplus(self.w_temp_logits[2])).clamp(1.0, 8.0),
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

        let mut ctx_states = [[0.0; D]; N_CTX];
        for c in 0..N_CTX {
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
            sym_rel_bias: vec![0.0; k * k],
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
        }

        for i in 0..N_LAT {
            for j in 0..N_LAT {
                let mut dot_ij = 0.0;
                for d in 0..D {
                    dot_ij += q_lat[i][d] * k_lat[j][d];
                }
                let val = tanh_clamped(dot_ij * base_scale * 1.6);
                h.lat_couplings[i * N_LAT + j] = if i == j { val * 0.2 } else { val };
            }
        }

        let mut rq_sym = vec![[0.0; D]; k];
        let mut rk_sym = vec![[0.0; D]; k];
        for s in 0..k {
            matvec(&self.w_read_query, D, D, &h.sym_states[s], &mut rq_sym[s]);
            matvec(&self.w_read_key, D, D, &h.sym_states[s], &mut rk_sym[s]);
        }

        for s in 0..k {
            let mut max_sc = f64::NEG_INFINITY;
            let mut scores = [0.0; N_LAT];
            for l in 0..N_LAT {
                let mut sc = 0.0;
                for d in 0..D {
                    sc += rq_sym[s][d] * k_lat[l][d];
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
                    sc += q_lat[l][d] * rk_sym[s][d];
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

        // Passage net par le goulot causal non-réciproque + accentuation log-linéaire
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

            // Calcul des logits combinant le goulot latent et la projection bilinéaire asymétrique rq_sym . rk_sym
            let mut logits = vec![0.0; k];
            let mut max_logit = f64::NEG_INFINITY;
            for v in 0..k {
                let mut p_uv = 0.0;
                for j in 0..N_LAT {
                    p_uv += lat_after[j] * h.lat_to_sym[j * k + v];
                }
                let mut bil = 0.0;
                for d in 0..D {
                    bil += rq_sym[u][d] * rk_sym[v][d];
                }
                let l_uv = b_out * p_uv.max(1e-9).ln() * 0.45
                    + (bil * base_scale * 1.4)
                    + h.sym_rel_bias[u * k + v];
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
        let gain = step_gain.clamp(0.05, 1.0);

        let mut new_sym = vec![[0.0; D]; k];
        let mut msg_eff = [0.0; D];
        let mut msg_cau = [0.0; D];
        let mut msg_lat = [0.0; D];
        let mut concat = [0.0; 3 * D];
        let mut gate = [0.0; D];
        let mut cand = [0.0; D];

        for u in 0..k {
            let mut agg_fwd = [0.0; D];
            let mut agg_bwd = [0.0; D];
            for v in 0..k {
                let w_out = pg.emp_trans1[u * k + v];
                let w_in = pg.emp_trans1[v * k + u];
                for d in 0..D {
                    agg_fwd[d] += w_out * h.sym_states[v][d];
                    agg_bwd[d] += w_in * h.sym_states[v][d];
                }
            }
            matvec(&self.w_fwd_effect, D, D, &agg_fwd, &mut msg_eff);
            matvec(&self.w_bwd_cause, D, D, &agg_bwd, &mut msg_cau);

            let mut agg_lat = [0.0; D];
            for l in 0..N_LAT {
                let w = h.sym_to_lat[u * N_LAT + l];
                for d in 0..D {
                    agg_lat[d] += w * h.lat_states[l][d];
                }
            }
            matvec(&self.w_lat_to_ctx, D, D, &agg_lat, &mut msg_lat);

            for d in 0..D {
                concat[d] = msg_eff[d];
                concat[D + d] = msg_cau[d];
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

        let mut new_ctx = [[0.0; D]; N_CTX];
        for c in 0..N_CTX {
            let mut agg_sym = [0.0; D];
            for s in 0..k {
                let w = pg.ctx_sym_weights[c * k + s];
                for d in 0..D {
                    agg_sym[d] += w * h.sym_states[s][d];
                }
            }
            let prev_c = if c > 0 { h.ctx_states[c - 1] } else { [0.0; D] };
            let next_c = if c + 1 < N_CTX { h.ctx_states[c + 1] } else { [0.0; D] };

            matvec(&self.w_fwd_effect, D, D, &prev_c, &mut msg_eff);
            matvec(&self.w_bwd_cause, D, D, &next_c, &mut msg_cau);
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
            for c in 0..N_CTX {
                let w = ((c + i) % 3 == 0) as usize as f64 * 0.5 + 0.5;
                for d in 0..D {
                    agg_ctx[d] += w * h.ctx_states[c][d];
                }
            }
            for d in 0..D {
                agg_ctx[d] /= N_CTX as f64;
            }

            matvec(&self.w_rel_lat_fwd, D, D, &agg_out, &mut msg_eff);
            matvec(&self.w_rel_lat_bwd, D, D, &agg_in, &mut msg_cau);
            matvec(&self.w_ctx_to_lat, D, D, &agg_ctx, &mut msg_lat);

            for d in 0..D {
                concat[d] = msg_eff[d];
                concat[D + d] = msg_cau[d];
                concat[2 * D + d] = msg_lat[d];
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
        w.write_all(b"PHIS0002")?;
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
        if &magic != b"PHIS0002" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "En-tête .phis invalide (attendu PHIS0002)",
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