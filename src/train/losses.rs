use crate::autodiff::{ParamStore, Tape, ValId};
use crate::model::controller::{
    AbductiveTrace, causal_closure_predict, forward_causal_intervention, readout_train,
    run_abductive_search,
};
use crate::model::graph::GraphParams;
use crate::synth::codec::{BlockKind, SpanMeta};
use crate::synth::episode::Episode;

#[derive(Clone, Debug)]
pub struct LossWeights {
    pub w_iso: f32,
    pub w_do: f32,
    pub w_max_rel: f32,
    pub w_closure: f32,
    pub w_teleology: f32,
    pub w_halt: f32,
    pub w_residual: f32,
    pub w_div: f32
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            w_iso: 1.5,
            w_do: 1.5,
            w_max_rel: 1.5,
            w_closure: 1.0,
            w_teleology: 1.0,
            w_halt: 0.35,
            w_residual: 0.25,
            w_div: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LossBreakdown {
    pub total: f32,
    pub iso: f32,
    pub do_causal: f32,
    pub max_rel: f32,
    pub closure: f32,
    pub teleology: f32,
    pub halt: f32,
    pub residual: f32,
    pub div: f32,
}

fn build_span_pool_matrix(spans: &[SpanMeta], num_blocks: usize, seq_len: usize) -> Vec<f32> {
    let mut pool = vec![0.0f32; num_blocks * seq_len];
    for sp in spans {
        if sp.block_id >= num_blocks || sp.end <= sp.start || sp.end > seq_len {
            continue;
        }
        let inv_len = 1.0 / ((sp.end - sp.start) as f32);
        for p in sp.start..sp.end {
            pool[sp.block_id * seq_len + p] = inv_len;
        }
    }
    pool
}

/// Calcule `acc + w * term` sans imbriquer d'emprunts mutables sur la tape.
fn weighted_add(tape: &mut Tape, acc: ValId, term: ValId, w: f32) -> ValId {
    let scaled = tape.scale(term, w);
    tape.add(acc, scaled)
}

/// Extrait pour une vue :
/// - P : [B, K] matrice de routage bloc -> nœuds du substrat
/// - Z : [B, d] représentations explicatives normalisées par bloc
/// - C_norm : [B, B] couplages causaux dirigés bloc -> bloc (non-réciproques)
fn extract_block_causality(
    tape: &mut Tape,
    params: &GraphParams,
    trace: &AbductiveTrace,
    spans: &[SpanMeta],
    num_blocks: usize,
    seq_len: usize,
) -> (ValId, ValId, ValId) {
    let k = params.cfg.num_nodes;
    let d = params.cfg.dim;
    let e = k * k;

    let pool_data = build_span_pool_matrix(spans, num_blocks, seq_len);
    let pool = tape.tensor(pool_data, vec![num_blocks, seq_len]);
    let attn_t = tape.transpose_2d(trace.attn_final); // [L, K]
    let p_mat = tape.matmul(pool, attn_t); // [B, K]

    let z_raw = tape.matmul(p_mat, trace.h_final); // [B, d]
    let z = tape.rmsnorm_rows(z_raw, 1e-5); // [B, d]

    let w_flat = tape.reshape(trace.w_eff_final, vec![e, d * d]);
    let w_sq = tape.square(w_flat);
    let edge_energy = tape.sum_axis1(w_sq); // [K*K, 1]
    let edge_strength = tape.mul(edge_energy, trace.gates_final); // [K*K, 1]
    let g_mat = tape.reshape(edge_strength, vec![k, k]); // [K, K]

    let p_t = tape.transpose_2d(p_mat); // [K, B]
    let g_pt = tape.matmul(g_mat, p_t); // [K, B]
    let c_raw = tape.matmul(p_mat, g_pt); // [B, B]
    let c_scaled = tape.scale(c_raw, 1.0 / (d as f32));
    let c_norm = tape.tanh(c_scaled); // [B, B]

    (p_mat, z, c_norm)
}

pub fn compute_episode_losses(
    tape: &mut Tape,
    store: &ParamStore,
    params: &GraphParams,
    ep: &Episode,
    weights: &LossWeights,
    loss_scale: f32,
) -> (ValId, LossBreakdown) {
    let cfg = &params.cfg;
    let d = cfg.dim;
    let b = ep.num_blocks;

    // Recherche abductive complète sur les deux vues multi-codecs indépendantes
    let trace_v1 = run_abductive_search(tape, store, params, &ep.bytes_v1, Some(cfg.max_steps), false);
    let trace_v2 = run_abductive_search(tape, store, params, &ep.bytes_v2, Some(cfg.max_steps), false);

    let (p1, z1, c1_norm) = extract_block_causality(
        tape,
        params,
        &trace_v1,
        &ep.spans_v1,
        b,
        ep.bytes_v1.len(),
    );
    let (_p2, z2, c2_norm) = extract_block_causality(
        tape,
        params,
        &trace_v2,
        &ep.spans_v2,
        b,
        ep.bytes_v2.len(),
    );

    // -------------------------------------------------------------------------
    // PILIER 1 : Isomorphisme inter-codecs (abduction invariante à la surface)
    // -------------------------------------------------------------------------
    let inv_d = 1.0 / (d as f32);
    let z1_t = tape.transpose_2d(z1);
    let z1_gram = tape.matmul(z1, z1_t);
    let r1 = tape.scale(z1_gram, inv_d); // [B, B]
    let z2_t = tape.transpose_2d(z2);
    let z2_gram = tape.matmul(z2, z2_t);
    let r2 = tape.scale(z2_gram, inv_d); // [B, B]
    let l_iso_rep = tape.mse_loss(r1, r2);
    let l_iso_causal = tape.mse_loss(c1_norm, c2_norm);
    let l_iso = weighted_add(tape, l_iso_rep, l_iso_causal, 0.5);

    // -------------------------------------------------------------------------
    // PILIER 2 : Fidélité directionnelle sous intervention do()
    // -------------------------------------------------------------------------
    let seed_block_id = ep
        .spans_v1
        .iter()
        .find(|sp| sp.kind == BlockKind::RuleSpec && ep.do_affected_blocks[sp.block_id] > 0.5)
        .map(|sp| sp.block_id)
        .or_else(|| ep.do_affected_blocks.iter().position(|&v| v > 0.5))
        .unwrap_or(0);

    let mut seed_vec_data = vec![0.0f32; b];
    seed_vec_data[seed_block_id] = 1.0;
    let seed_vec = tape.tensor(seed_vec_data, vec![1, b]);
    let seed_node_row = tape.matmul(seed_vec, p1); // [1, K]
    let seed_node_w = tape.transpose_2d(seed_node_row); // [K, 1]

    let node_delta = forward_causal_intervention(tape, params, &trace_v1, seed_node_w, 2); // [K, 1]
    let block_delta = tape.matmul(p1, node_delta); // [B, 1]
    // Borner strictement dans [1e-3, 0.999] pour gradient lisse partout
    let delta_tanh = tape.tanh(block_delta);
    let delta_scaled = tape.scale(delta_tanh, 0.998);
    let eps_floor = tape.tensor(vec![1e-3], vec![1]);
    let pred_do = tape.add(delta_scaled, eps_floor);
    let l_do = tape.bce_with_probs(pred_do, ep.do_affected_blocks.clone(), 1e-5);

    // -------------------------------------------------------------------------
    // PILIER 3 : Saturation relationnelle maximale (Anti-MDL contraint)
    // -------------------------------------------------------------------------
    let w_rel = tape.param(store, params.w_rel_proj);
    let b_rel = tape.param(store, params.b_rel);
    let z1_proj = tape.matmul(z1, w_rel);
    let z1_rel = tape.rmsnorm_rows(z1_proj, 1e-5); // [B, d]
    let inv_sqrt_d = 1.0 / (d as f32).sqrt();
    let z1_rel_t = tape.transpose_2d(z1_rel);
    let sim_raw = tape.matmul(z1_rel, z1_rel_t);
    let sim_scaled = tape.scale(sim_raw, inv_sqrt_d);
    let sim_logits = tape.add(sim_scaled, b_rel);
    let c1_t = tape.transpose_2d(c1_norm);
    let c1_sym = tape.add(c1_norm, c1_t);
    let rel_logits = weighted_add(tape, sim_logits, c1_sym, 0.5);

    let l_rel_bce = tape.bce_with_logits(rel_logits, ep.homology_matrix.clone());

    let mut pos_mask_data = vec![0.0f32; b * b];
    let mut pos_count = 0.0f32;
    for i in 0..b {
        for j in 0..b {
            if i != j && ep.homology_matrix[i * b + j] > 0.5 {
                pos_mask_data[i * b + j] = 1.0;
                pos_count += 1.0;
            }
        }
    }
    let l_max_rel = if pos_count > 0.0 {
        let pos_mask = tape.tensor(pos_mask_data, vec![b, b]);
        let p_rel = tape.sigmoid(rel_logits);
        let log_p = tape.safe_log(p_rel, 1e-6);
        let neg_log_p = tape.neg(log_p);
        let masked = tape.mul(pos_mask, neg_log_p);
        let masked_sum = tape.sum_all(masked);
        let sat_loss = tape.scale(masked_sum, 1.0 / pos_count);
        weighted_add(tape, l_rel_bce, sat_loss, 0.5)
    } else {
        l_rel_bce
    };

    // -------------------------------------------------------------------------
    // PILIER 4a : Fermeture causale sans fuite d'octets (masque par 0u8)
    // -------------------------------------------------------------------------
    let (c_start, c_end) = ep.closure_span_v1;
    let l_closure = if c_end > c_start && !ep.closure_target_bytes.is_empty() {
        let mut bytes_masked = ep.bytes_v1.clone();
        bytes_masked[c_start..c_end].fill(0u8);
        let trace_masked = run_abductive_search(
            tape,
            store,
            params,
            &bytes_masked,
            Some(cfg.max_steps),
            false,
        );
        let closure_logits = causal_closure_predict(
       tape,
       store,
       params,
       &trace_masked,
       &bytes_masked,
       c_start,
       c_end,
   );
        let closure_lp = tape.masked_log_softmax_rows(closure_logits, ep.alphabet_mask_v1.clone());
        let targets: Vec<usize> = ep.closure_target_bytes.iter().map(|&x| x as usize).collect();
        tape.nll_loss(closure_lp, targets)
    } else {
        tape.scalar(0.0)
    };

    // -------------------------------------------------------------------------
    // PILIER 4b : Résolution téléologique dans l'alphabet de E1
    // -------------------------------------------------------------------------
    let (ro_log_probs, ro_stop_logits) = readout_train(
tape,
store,
params,
&trace_v1,
&ep.bytes_v1,
&ep.alphabet_mask_v1,
&ep.target_bytes,
);
    let ro_targets: Vec<usize> = ep.target_bytes.iter().map(|&x| x as usize).collect();
    let l_ro_bytes = tape.nll_loss(ro_log_probs, ro_targets);

    let mut stop_targets = vec![0.0f32; ep.target_bytes.len() + 1];
    stop_targets[ep.target_bytes.len()] = 1.0;
    let l_ro_stop = tape.bce_with_logits(ro_stop_logits, stop_targets);
    let l_teleology = weighted_add(tape, l_ro_bytes, l_ro_stop, 0.5);

    // -------------------------------------------------------------------------
    // Calibration de la convergence / arrêt adaptatif
    // -------------------------------------------------------------------------
    let mut halt_seq = trace_v1.halt_logits[0];
    let num_steps = trace_v1.halt_logits.len();
    let mut halt_targets = Vec::with_capacity(num_steps);
    for t in 0..num_steps {
        if t > 0 {
            halt_seq = tape.concat_rows(halt_seq, trace_v1.halt_logits[t]);
        }
        let r_val = tape.data(trace_v1.residuals[t])[0];
        let tgt = if t + 1 < cfg.min_steps {
            0.0
        } else if t + 1 == num_steps || r_val < 0.08 {
            1.0
        } else {
            (-3.0 * r_val).exp().min(0.65)
        };
        halt_targets.push(tgt);
    }
    let l_halt = tape.bce_with_logits(halt_seq, halt_targets);
    let l_res = *trace_v1
        .residuals
        .last()
        .expect("at least one step executed");

    let k = cfg.num_nodes;
    let mut off_data = vec![1.0f32; k * k];
    for idx in 0..k {
        off_data[idx * k + idx] = 0.0;
    }
    let off_mask = tape.tensor(off_data, vec![k, k]);

let h_final_t = tape.transpose_2d(trace_v1.h_final);
let gram_f_raw = tape.matmul(trace_v1.h_final, h_final_t);
let gram_f = tape.scale(gram_f_raw, inv_d);

let h_regional_t = tape.transpose_2d(trace_v1.h_regional);
let gram_r_raw = tape.matmul(trace_v1.h_regional, h_regional_t);
let gram_r = tape.scale(gram_r_raw, inv_d);

let m_a_prod = tape.mul(gram_f, off_mask);
let m_a = tape.square(m_a_prod);

let m_a_sq   = tape.square(m_a);
let mean_sq  = tape.mean_all(m_a_sq);
let scaled   = tape.scale(mean_sq, 0.5);
let mean_a   = tape.mean_all(m_a);
let l_div    = tape.add(mean_a, scaled);

    // Agrégation pondérée totale
    let mut total = tape.scale(l_iso, weights.w_iso);
    total = weighted_add(tape, total, l_do, weights.w_do);
    total = weighted_add(tape, total, l_max_rel, weights.w_max_rel);
    total = weighted_add(tape, total, l_closure, weights.w_closure);
    total = weighted_add(tape, total, l_teleology, weights.w_teleology);
    total = weighted_add(tape, total, l_halt, weights.w_halt);
    total = weighted_add(tape, total, l_res, weights.w_residual);
    total = weighted_add(tape, total, l_div, weights.w_div);

    let breakdown = LossBreakdown {
        total: tape.data(total)[0],
        iso: tape.data(l_iso)[0],
        do_causal: tape.data(l_do)[0],
        max_rel: tape.data(l_max_rel)[0],
        closure: tape.data(l_closure)[0],
        teleology: tape.data(l_teleology)[0],
        halt: tape.data(l_halt)[0],
        residual: tape.data(l_res)[0],
        div: tape.data(l_div)[0],
    };

    let scaled_total = if (loss_scale - 1.0).abs() > 1e-6 {
        tape.scale(total, loss_scale)
    } else {
        total
    };

    (scaled_total, breakdown)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::graph::ModelConfig;
    use crate::rng::Rng;
    use crate::synth::episode::{EpisodeConfig, sample_episode};

    #[test]
    fn four_pillars_losses_flow_gradients_to_all_subsystems() {
        let mut rng = Rng::from_seed(4242);
        let mut store = ParamStore::new();
        let cfg = ModelConfig {
            num_nodes: 6,
            dim: 8,
            pos_dim: 6,
            min_steps: 2,
            max_steps: 3,
            halt_threshold: 0.8,
            max_readout_bytes: 32,
        };
        let params = GraphParams::init(cfg, &mut store, &mut rng);
        let ep = sample_episode(&mut rng, &EpisodeConfig::default());
        let weights = LossWeights::default();

        let mut tape = Tape::new();
        let (loss, br) = compute_episode_losses(&mut tape, &store, &params, &ep, &weights, 1.0);
        assert!(br.total.is_finite());
        assert!(br.iso.is_finite());
        assert!(br.do_causal.is_finite());
        assert!(br.max_rel.is_finite());
        assert!(br.closure.is_finite());
        assert!(br.teleology.is_finite());

        tape.backward(loss, &mut store);

        let check_nonzero = |pid: crate::autodiff::ParamId| {
            let norm: f32 = store.grads[pid.0].iter().map(|g| g * g).sum();
            assert!(
                norm > 1e-12,
                "Gradient nul pour paramètre {} !",
                store.names[pid.0]
            );
        };
        check_nonzero(params.byte_embed);
        check_nonzero(params.w_slow_edges);
        check_nonzero(params.w_route_gate);
        check_nonzero(params.w_plast_err);
        check_nonzero(params.w_rel_proj);
        check_nonzero(params.w_closure_out);
        check_nonzero(params.w_ro_byte);
    }
}