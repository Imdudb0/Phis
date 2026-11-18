use crate::autodiff::{ParamStore, Tape, ValId};
use crate::model::graph::{
    GraphParams, encode_sensory_field, positional_features, trigger_regional_r,
};

#[derive(Clone, Debug)]
pub struct AbductiveTrace {
    pub sens: ValId,
    pub sens_keys: ValId,
    pub h_regional: ValId,
    pub h_final: ValId,
    pub w_dyn_final: ValId,
    pub w_eff_final: ValId,
    pub attn_final: ValId,
    pub gates_final: ValId,
    pub e_local_final: ValId,
    pub halt_logits: Vec<ValId>,
    pub residuals: Vec<ValId>,
    pub steps_executed: usize,
}

pub fn alphabet_mask_from_bytes(bytes: &[u8]) -> Vec<f32> {
    let mut mask = vec![0.0f32; 256];
    for &b in bytes {
        mask[b as usize] = 1.0;
    }
    mask
}

pub fn build_byte_incidence(bytes: &[u8]) -> Vec<f32> {
    let mut mat = vec![0.0f32; bytes.len() * 256];
    for (p, &b) in bytes.iter().enumerate() {
        mat[p * 256 + (b as usize)] = 1.0;
    }
    mat
}

pub fn run_abductive_search(
    tape: &mut Tape,
    store: &ParamStore,
    params: &GraphParams,
    bytes: &[u8],
    steps_override: Option<usize>,
    adaptive_halt: bool,
) -> AbductiveTrace {
    let cfg = &params.cfg;
    let k = cfg.num_nodes;
    let d = cfg.dim;
    let e = k * k;
    let max_steps = steps_override.unwrap_or(cfg.max_steps).max(1);

    let (sens, sens_keys) = encode_sensory_field(tape, store, params, bytes);
    let (h0, attn0) = trigger_regional_r(tape, store, params, sens, sens_keys);

    let w_slow = tape.param(store, params.w_slow_edges);
    let mut w_dyn = tape.tensor(vec![0.0; e * d * d], vec![e, d, d]);
    let mut h = h0;
    let mut attn = attn0;

    let w_read_q = tape.param(store, params.w_read_q);
    let w_route_gate = tape.param(store, params.w_route_gate);
    let b_route_gate = tape.param(store, params.b_route_gate);
    let w_upd = tape.param(store, params.w_upd);
    let b_upd = tape.param(store, params.b_upd);
    let w_beta = tape.param(store, params.w_beta);
    let b_beta = tape.param(store, params.b_beta);
    let w_plast_pre = tape.param(store, params.w_plast_pre);
    let w_plast_err = tape.param(store, params.w_plast_err);
    let w_plast_post = tape.param(store, params.w_plast_post);
    let w_eta = tape.param(store, params.w_eta);
    let b_eta = tape.param(store, params.b_eta);
    let w_gamma = tape.param(store, params.w_gamma);
    let b_gamma = tape.param(store, params.b_gamma);
    let w_decay = tape.param(store, params.w_decay);
    let b_decay = tape.param(store, params.b_decay);
    let w_halt = tape.param(store, params.w_halt);
    let b_halt = tape.param(store, params.b_halt);

    let sens_keys_t = tape.transpose_2d(sens_keys);
    let attn_scale = 2.0 / (d as f32).sqrt();
    let one_scalar = tape.scalar(1.0);
    let inv_k = 1.0 / (k as f32);

    let mut halt_logits = Vec::with_capacity(max_steps);
    let mut residuals = Vec::with_capacity(max_steps);

    let mut gates = tape.tensor(vec![0.0; e], vec![e, 1]);
    let mut e_local = tape.tensor(vec![0.0; k * d], vec![k, d]);
    let mut steps_executed = 0usize;

    for step in 0..max_steps {
        let q_lin = tape.matmul(h, w_read_q);
        let q = tape.rmsnorm_rows(q_lin, 1e-5);
        let qk = tape.matmul(q, sens_keys_t);
        let scores = tape.scale(qk, attn_scale);
        attn = tape.softmax_rows(scores);
        let r_sens = tape.matmul(attn, sens);

        let h_src = tape.gather_rows(h, params.src_idx.clone());
        let h_dst = tape.gather_rows(h, params.dst_idx.clone());
        let r_dst = tape.gather_rows(r_sens, params.dst_idx.clone());
        let edge_ctx_1 = tape.concat_cols(h_src, h_dst);
        let edge_ctx = tape.concat_cols(edge_ctx_1, r_dst);

        let gate_lin = tape.matmul(edge_ctx, w_route_gate);
        let gate_b = tape.add(gate_lin, b_route_gate);
        gates = tape.sigmoid(gate_b);

        let w_eff = tape.add(w_slow, w_dyn);
        let h_src_col = tape.reshape(h_src, vec![e, d, 1]);
        let raw_msg_3d = tape.batched_matmul(w_eff, h_src_col);
        let raw_msg = tape.reshape(raw_msg_3d, vec![e, d]);
        let gated_msg = tape.mul(raw_msg, gates);
        let agg_msg = tape.scatter_add_rows(gated_msg, params.dst_idx.clone(), k);

        let agg_norm = tape.rmsnorm_rows(agg_msg, 1e-5);
        let causal_pred = tape.tanh(agg_norm);
        e_local = tape.sub(r_sens, causal_pred);

        let upd_1 = tape.concat_cols(h, agg_msg);
        let upd_in = tape.concat_cols(upd_1, r_sens);
        let dh_lin = tape.matmul(upd_in, w_upd);
        let dh_b = tape.add(dh_lin, b_upd);
        let delta_h = tape.tanh(dh_b);
        let beta_lin = tape.matmul(upd_in, w_beta);
        let beta_b = tape.add(beta_lin, b_beta);
        let beta = tape.sigmoid(beta_b);
        let beta_dh = tape.mul(beta, delta_h);
        let h_sum = tape.add(h, beta_dh);
        let h_next = tape.rmsnorm_rows(h_sum, 1e-5);

        let pre_lin = tape.matmul(h, w_plast_pre);
        let pre = tape.tanh(pre_lin);
        let perr_lin = tape.matmul(e_local, w_plast_err);
        let post_err = tape.tanh(perr_lin);
        let pst_lin = tape.matmul(h_next, w_plast_post);
        let post_state = tape.tanh(pst_lin);

        let pre_g = tape.gather_rows(pre, params.src_idx.clone());
        let pre_e = tape.reshape(pre_g, vec![e, 1, d]);
        let perr_g = tape.gather_rows(post_err, params.dst_idx.clone());
        let post_err_e = tape.reshape(perr_g, vec![e, d, 1]);
        let pst_g = tape.gather_rows(post_state, params.dst_idx.clone());
        let post_st_e = tape.reshape(pst_g, vec![e, d, 1]);

        let hebb_err = tape.batched_matmul(post_err_e, pre_e);
        let hebb_st = tape.batched_matmul(post_st_e, pre_e);

        let eta_lin = tape.matmul(edge_ctx, w_eta);
        let eta_b = tape.add(eta_lin, b_eta);
        let eta_sig = tape.sigmoid(eta_b);
        let eta_sc = tape.scale(eta_sig, 0.35);
        let eta = tape.reshape(eta_sc, vec![e, 1, 1]);

        let gam_lin = tape.matmul(edge_ctx, w_gamma);
        let gam_b = tape.add(gam_lin, b_gamma);
        let gam_t = tape.tanh(gam_b);
        let gamma = tape.reshape(gam_t, vec![e, 1, 1]);

        let dec_lin = tape.matmul(edge_ctx, w_decay);
        let dec_b = tape.add(dec_lin, b_decay);
        let dec_sig = tape.sigmoid(dec_b);
        let dec_sc = tape.scale(dec_sig, 0.15);
        let decay = tape.reshape(dec_sc, vec![e, 1, 1]);

        let g_hebb = tape.mul(gamma, hebb_st);
        let plast_drive = tape.add(hebb_err, g_hebb);
        let delta_w = tape.mul(eta, plast_drive);
        let keep = tape.sub(one_scalar, decay);
        let keep_w = tape.mul(keep, w_dyn);
        let w_raw_next = tape.add(keep_w, delta_w);
        let w_half = tape.scale(w_raw_next, 0.5);
        let w_tanh = tape.tanh(w_half);
        w_dyn = tape.scale(w_tanh, 2.0);

        let e_sq = tape.square(e_local);
        let res_err = tape.mean_all(e_sq);
        let dh_diff = tape.sub(h_next, h);
        let dh_sq = tape.square(dh_diff);
        let res_dh = tape.mean_all(dh_sq);
        let res_total = tape.add(res_err, res_dh);
        residuals.push(res_total);

        let sum_h = tape.sum_axis0(h_next);
        let mean_h = tape.scale(sum_h, inv_k);
        let sum_e = tape.sum_axis0(e_local);
        let mean_e = tape.scale(sum_e, inv_k);
        let res_col = tape.reshape(res_total, vec![1, 1]);
        let he_cat = tape.concat_cols(mean_h, mean_e);
        let halt_in = tape.concat_cols(he_cat, res_col);
        let h_lin = tape.matmul(halt_in, w_halt);
        let h_logit = tape.add(h_lin, b_halt);
        halt_logits.push(h_logit);

        h = h_next;
        steps_executed = step + 1;

        if adaptive_halt && steps_executed >= cfg.min_steps {
            let z = tape.data(h_logit)[0];
            let p_halt = 1.0 / (1.0 + (-z).exp());
            if p_halt >= cfg.halt_threshold {
                break;
            }
        }
    }

    let w_eff = tape.add(w_slow, w_dyn);

    AbductiveTrace {
        sens,
        sens_keys,
        h_regional: h0,
        h_final: h,
        w_dyn_final: w_dyn,
        w_eff_final: w_eff,
        attn_final: attn,
        gates_final: gates,
        e_local_final: e_local,
        halt_logits,
        residuals,
        steps_executed,
    }
}

pub fn forward_causal_intervention(
    tape: &mut Tape,
    params: &GraphParams,
    trace: &AbductiveTrace,
    seed_node_weights: ValId,
    prop_steps: usize,
) -> ValId {
    let k = params.cfg.num_nodes;
    let d = params.cfg.dim;
    let e = k * k;

    let mut delta_h = tape.mul(trace.h_final, seed_node_weights);

    for _ in 0..prop_steps {
        let dh_src = tape.gather_rows(delta_h, params.src_idx.clone());
        let dh_col = tape.reshape(dh_src, vec![e, d, 1]);
        let prop_3d = tape.batched_matmul(trace.w_eff_final, dh_col);
        let prop = tape.reshape(prop_3d, vec![e, d]);
        let gated = tape.mul(prop, trace.gates_final);
        let agg = tape.scatter_add_rows(gated, params.dst_idx.clone(), k);
        let dh_sum = tape.add(delta_h, agg);
        delta_h = tape.tanh(dh_sum);
    }

    let dh_sq = tape.square(delta_h);
    tape.sum_axis1(dh_sq)
}

fn settled_causal_substrate(tape: &mut Tape, params: &GraphParams, trace: &AbductiveTrace) -> ValId {
    let k = params.cfg.num_nodes;
    let d = params.cfg.dim;
    let e = k * k;
    let h_src = tape.gather_rows(trace.h_final, params.src_idx.clone());
    let h_col = tape.reshape(h_src, vec![e, d, 1]);
    let msg_3d = tape.batched_matmul(trace.w_eff_final, h_col);
    let msg = tape.reshape(msg_3d, vec![e, d]);
    let gated = tape.mul(msg, trace.gates_final);
    let agg = tape.scatter_add_rows(gated, params.dst_idx.clone(), k);
    let agg_t = tape.tanh(agg);
    let h_sum = tape.add(trace.h_final, agg_t);
    tape.rmsnorm_rows(h_sum, 1e-5)
}

pub fn causal_closure_predict(
    tape: &mut Tape,
    store: &ParamStore,
    params: &GraphParams,
    trace: &AbductiveTrace,
    bytes_masked: &[u8],
    span_start: usize,
    span_end: usize,
) -> ValId {
    let total_len = bytes_masked.len();
    let span_len = (span_end - span_start).max(1);
    let pos_global = positional_features(total_len, span_start, span_end, params.cfg.pos_dim);
    let pos_local = positional_features(span_len, 0, span_len, params.cfg.pos_dim);
    let pos_gv = tape.tensor(pos_global.data, pos_global.shape);
    let pos_lv = tape.tensor(pos_local.data, pos_local.shape);
    let pos_g_sc = tape.scale(pos_gv, 0.35);
    let pos_l_sc = tape.scale(pos_lv, 0.85);
    let pos_v = tape.add(pos_g_sc, pos_l_sc);

    let h_causal = settled_causal_substrate(tape, params, trace);

    let w_cq = tape.param(store, params.w_closure_q);
    let w_ck = tape.param(store, params.w_closure_k);
    let w_csq = tape.param(store, params.w_closure_sens_q);
    let w_cout = tape.param(store, params.w_closure_out);
    let b_cout = tape.param(store, params.b_closure_out);

    let q = tape.matmul(pos_v, w_cq);
    let k_proj = tape.matmul(h_causal, w_ck);
    let k_sub = tape.rmsnorm_rows(k_proj, 1e-5);
    let k_sub_t = tape.transpose_2d(k_sub);
    let scale = 2.0 / (params.cfg.dim as f32).sqrt();
    let qk = tape.matmul(q, k_sub_t);
    let qk_sc = tape.scale(qk, scale);
    let alpha = tape.softmax_rows(qk_sc);
    let ctx = tape.matmul(alpha, h_causal);

    let feat = tape.concat_cols(q, ctx);
    let stat_lin = tape.matmul(feat, w_cout);
    let logits_static = tape.add(stat_lin, b_cout);

    let q_sens_lin = tape.matmul(feat, w_csq);
    let q_sens = tape.rmsnorm_rows(q_sens_lin, 1e-5);
    let sk_t = tape.transpose_2d(trace.sens_keys);
    let c_qk = tape.matmul(q_sens, sk_t);
    let c_sc = tape.scale(c_qk, scale);
    let c_attn = tape.softmax_rows(c_sc);

    let occ_mat = build_byte_incidence(bytes_masked);
    let occ_t = tape.tensor(occ_mat, vec![total_len, 256]);
    let p_assoc = tape.matmul(c_attn, occ_t);
    let eps_mix = tape.scalar(1e-3);
    let p_safe = tape.add(p_assoc, eps_mix);
    let log_assoc = tape.safe_log(p_safe, 1e-6);

    tape.add(logits_static, log_assoc)
}

pub fn readout_train(
    tape: &mut Tape,
    store: &ParamStore,
    params: &GraphParams,
    trace: &AbductiveTrace,
    bytes: &[u8],
    alphabet_mask: &[f32],
    target_bytes: &[u8],
) -> (ValId, ValId) {
    assert!(!target_bytes.is_empty(), "target_bytes must not be empty");
    let k = params.cfg.num_nodes;
    let d = params.cfg.dim;
    let len = bytes.len();
    let inv_k = 1.0 / (k as f32);
    let scale = 2.0 / (d as f32).sqrt();

    let h_causal = settled_causal_substrate(tape, params, trace);
    let sens_keys_t = tape.transpose_2d(trace.sens_keys);
    let occ_t = tape.tensor(build_byte_incidence(bytes), vec![len, 256]);
    let eps_mix = tape.scalar(1e-3);

    let byte_embed = tape.param(store, params.byte_embed);
    let w_ro_init = tape.param(store, params.w_ro_init);
    let w_ro_k = tape.param(store, params.w_ro_k);
    let w_ro_sens_q = tape.param(store, params.w_ro_sens_q);
    let w_ro_byte = tape.param(store, params.w_ro_byte);
    let b_ro_byte = tape.param(store, params.b_ro_byte);
    let w_ro_stop = tape.param(store, params.w_ro_stop);
    let b_ro_stop = tape.param(store, params.b_ro_stop);
    let w_ro_step = tape.param(store, params.w_ro_step);
    let b_ro_step = tape.param(store, params.b_ro_step);

    let sum_h = tape.sum_axis0(h_causal);
    let pooled = tape.scale(sum_h, inv_k);
    let q_init_lin = tape.matmul(pooled, w_ro_init);
    let mut q = tape.tanh(q_init_lin);

    let k_proj = tape.matmul(h_causal, w_ro_k);
    let k_sub = tape.rmsnorm_rows(k_proj, 1e-5);
    let k_sub_t = tape.transpose_2d(k_sub);
    let mut cum_attn = tape.tensor(vec![0.0f32; len], vec![1, len]);

    let mut byte_logits_seq: Option<ValId> = None;
    let mut stop_logits_seq: Option<ValId> = None;

    for s in 0..=target_bytes.len() {
        let qk = tape.matmul(q, k_sub_t);
        let qk_sc = tape.scale(qk, scale);
        let alpha = tape.softmax_rows(qk_sc);
        let ctx = tape.matmul(alpha, h_causal);
        let ro_feat = tape.concat_cols(q, ctx);

        let stop_lin = tape.matmul(ro_feat, w_ro_stop);
        let stop_l = tape.add(stop_lin, b_ro_stop);
        stop_logits_seq = Some(match stop_logits_seq {
            None => stop_l,
            Some(prev) => tape.concat_rows(prev, stop_l),
        });

        if s < target_bytes.len() {
            let qs_lin = tape.matmul(ro_feat, w_ro_sens_q);
            let q_sens = tape.rmsnorm_rows(qs_lin, 1e-5);
            let rs_qk = tape.matmul(q_sens, sens_keys_t);
            let raw_scores = tape.scale(rs_qk, scale);
            let cov_pen = tape.scale(cum_attn, 1.8);
            let cov_scores = tape.sub(raw_scores, cov_pen);
            let ro_sens_attn = tape.softmax_rows(cov_scores);
            let cum_decayed = tape.scale(cum_attn, 0.85);
            cum_attn = tape.add(cum_decayed, ro_sens_attn);

            let p_assoc = tape.matmul(ro_sens_attn, occ_t);
            let p_safe = tape.add(p_assoc, eps_mix);
            let log_assoc = tape.safe_log(p_safe, 1e-6);
            let stat_lin = tape.matmul(ro_feat, w_ro_byte);
            let logits_static = tape.add(stat_lin, b_ro_byte);
            let byte_l = tape.add(logits_static, log_assoc);

            byte_logits_seq = Some(match byte_logits_seq {
                None => byte_l,
                Some(prev) => tape.concat_rows(prev, byte_l),
            });

            let y_emb = tape.gather_rows(byte_embed, vec![target_bytes[s] as usize]);
            let sens_ctx = tape.matmul(ro_sens_attn, trace.sens);
            let step_cat = tape.concat_cols(ro_feat, y_emb);
            let step_in = tape.concat_cols(step_cat, sens_ctx);
            let step_lin = tape.matmul(step_in, w_ro_step);
            let step_b = tape.add(step_lin, b_ro_step);
            let step_t = tape.tanh(step_b);
            q = tape.rmsnorm_rows(step_t, 1e-5);
        }
    }

    let raw_logits = byte_logits_seq.expect("non-empty target_bytes guarantees logits");
    let log_probs = tape.masked_log_softmax_rows(raw_logits, alphabet_mask.to_vec());
    let stop_logits = stop_logits_seq.expect("stop logits must exist");
    (log_probs, stop_logits)
}

pub fn readout_infer(store: &ParamStore, params: &GraphParams, bytes: &[u8]) -> Vec<u8> {
    if bytes.is_empty() {
        return Vec::new();
    }
    let alphabet_mask = alphabet_mask_from_bytes(bytes);
    let mut tape = Tape::new();
    let trace = run_abductive_search(&mut tape, store, params, bytes, None, true);

    let k = params.cfg.num_nodes;
    let d = params.cfg.dim;
    let len = bytes.len();
    let inv_k = 1.0 / (k as f32);
    let scale = 2.0 / (d as f32).sqrt();

    let h_causal = settled_causal_substrate(&mut tape, params, &trace);
    let sens_keys_t = tape.transpose_2d(trace.sens_keys);
    let occ_t = tape.tensor(build_byte_incidence(bytes), vec![len, 256]);
    let eps_mix = tape.scalar(1e-3);

    let byte_embed = tape.param(store, params.byte_embed);
    let w_ro_init = tape.param(store, params.w_ro_init);
    let w_ro_k = tape.param(store, params.w_ro_k);
    let w_ro_sens_q = tape.param(store, params.w_ro_sens_q);
    let w_ro_byte = tape.param(store, params.w_ro_byte);
    let b_ro_byte = tape.param(store, params.b_ro_byte);
    let w_ro_stop = tape.param(store, params.w_ro_stop);
    let b_ro_stop = tape.param(store, params.b_ro_stop);
    let w_ro_step = tape.param(store, params.w_ro_step);
    let b_ro_step = tape.param(store, params.b_ro_step);

    let sum_h = tape.sum_axis0(h_causal);
    let pooled = tape.scale(sum_h, inv_k);
    let q_init_lin = tape.matmul(pooled, w_ro_init);
    let mut q = tape.tanh(q_init_lin);

    let k_proj = tape.matmul(h_causal, w_ro_k);
    let k_sub = tape.rmsnorm_rows(k_proj, 1e-5);
    let k_sub_t = tape.transpose_2d(k_sub);
    let mut cum_attn = tape.tensor(vec![0.0f32; len], vec![1, len]);

    let mut out_bytes = Vec::new();

    for s in 0..params.cfg.max_readout_bytes {
        let qk = tape.matmul(q, k_sub_t);
        let qk_sc = tape.scale(qk, scale);
        let alpha = tape.softmax_rows(qk_sc);
        let ctx = tape.matmul(alpha, h_causal);
        let ro_feat = tape.concat_cols(q, ctx);

        let stop_lin = tape.matmul(ro_feat, w_ro_stop);
        let stop_l = tape.add(stop_lin, b_ro_stop);
        let z_stop = tape.data(stop_l)[0];
        let p_stop = 1.0 / (1.0 + (-z_stop).exp());
        if s > 0 && p_stop >= 0.5 {
            break;
        }

        let qs_lin = tape.matmul(ro_feat, w_ro_sens_q);
        let q_sens = tape.rmsnorm_rows(qs_lin, 1e-5);
        let rs_qk = tape.matmul(q_sens, sens_keys_t);
        let raw_scores = tape.scale(rs_qk, scale);
        let cov_pen = tape.scale(cum_attn, 1.8);
        let cov_scores = tape.sub(raw_scores, cov_pen);
        let ro_sens_attn = tape.softmax_rows(cov_scores);
        let cum_decayed = tape.scale(cum_attn, 0.85);
        cum_attn = tape.add(cum_decayed, ro_sens_attn);

        let p_assoc = tape.matmul(ro_sens_attn, occ_t);
        let p_safe = tape.add(p_assoc, eps_mix);
        let log_assoc = tape.safe_log(p_safe, 1e-6);
        let stat_lin = tape.matmul(ro_feat, w_ro_byte);
        let logits_static = tape.add(stat_lin, b_ro_byte);
        let byte_l = tape.add(logits_static, log_assoc);

        let logits = tape.data(byte_l);
        let mut best_b = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (b_idx, &lv) in logits.iter().enumerate() {
            if alphabet_mask[b_idx] > 0.5 && lv > best_v {
                best_v = lv;
                best_b = b_idx;
            }
        }
        out_bytes.push(best_b as u8);

        let y_emb = tape.gather_rows(byte_embed, vec![best_b]);
        let sens_ctx = tape.matmul(ro_sens_attn, trace.sens);
        let step_cat = tape.concat_cols(ro_feat, y_emb);
        let step_in = tape.concat_cols(step_cat, sens_ctx);
        let step_lin = tape.matmul(step_in, w_ro_step);
        let step_b = tape.add(step_lin, b_ro_step);
        let step_t = tape.tanh(step_b);
        q = tape.rmsnorm_rows(step_t, 1e-5);
    }

    out_bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::graph::non_reciprocity_ratio;
    use crate::rng::Rng;

    #[test]
    fn non_reciprocal_search_and_closed_alphabet_readout() {
        let mut rng = Rng::from_seed(555);
        let mut store = ParamStore::new();
        let cfg = ModelConfig {
            num_nodes: 8,
            dim: 8,
            pos_dim: 6,
            min_steps: 2,
            max_steps: 4,
            halt_threshold: 0.9,
            max_readout_bytes: 12,
        };
        let params = GraphParams::init(cfg, &mut store, &mut rng);

        let input = b"abcabc_xyz";
        let mut tape = Tape::new();
        let trace = run_abductive_search(&mut tape, &store, &params, input, None, false);

        let ratio = non_reciprocity_ratio(&tape, &params, trace.w_eff_final);
        assert!(ratio > 0.5, "Non-reciprocity ratio too low: {}", ratio);
        assert_eq!(trace.steps_executed, 4);

        let output = readout_infer(&store, &params, input);
        assert!(!output.is_empty());
        let mask = alphabet_mask_from_bytes(input);
        for &b in &output {
            assert_eq!(
                mask[b as usize], 1.0,
                "Octet émis {} absent de l'alphabet d'entrée!",
                b
            );
        }
    }
}