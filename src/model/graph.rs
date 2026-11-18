use crate::autodiff::{ParamId, ParamStore, Tape, TensorData, ValId};
use crate::rng::Rng;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub num_nodes: usize,
    pub dim: usize,
    pub pos_dim: usize,
    pub min_steps: usize,
    pub max_steps: usize,
    pub halt_threshold: f32,
    pub max_readout_bytes: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            num_nodes: 16,
            dim: 16,
            pos_dim: 16,
            min_steps: 2,
            max_steps: 8,
            halt_threshold: 0.75,
            max_readout_bytes: 64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GraphParams {
    pub cfg: ModelConfig,
    pub src_idx: Vec<usize>,
    pub dst_idx: Vec<usize>,
    pub rev_edge_idx: Vec<usize>,

    // Encodeur sensoriel pur octets (fenêtre 5 octets) + déclenchement régional R
    pub byte_embed: ParamId,
    pub w_sens_in: ParamId,
    pub b_sens_in: ParamId,
    pub w_sens_k: ParamId,
    pub regional_receptors: ParamId,

    // Substrat non-réciproque lent et contrôleur de routage + plasticité locale
    pub w_slow_edges: ParamId,
    pub w_read_q: ParamId,
    pub w_route_gate: ParamId,
    pub b_route_gate: ParamId,
    pub w_upd: ParamId,
    pub b_upd: ParamId,
    pub w_beta: ParamId,
    pub b_beta: ParamId,
    pub w_plast_pre: ParamId,
    pub w_plast_err: ParamId,
    pub w_plast_post: ParamId,
    pub w_eta: ParamId,
    pub b_eta: ParamId,
    pub w_gamma: ParamId,
    pub b_gamma: ParamId,
    pub w_decay: ParamId,
    pub b_decay: ParamId,
    pub w_halt: ParamId,
    pub b_halt: ParamId,

    // Tête de fermeture causale (hybride statique + rappel associatif sensoriel)
    pub w_closure_q: ParamId,
    pub w_closure_k: ParamId,
    pub w_closure_sens_q: ParamId,
    pub w_closure_out: ParamId,
    pub b_closure_out: ParamId,

    // Tête de lecture téléologique (en aval exclusif de l'explication stabilisée)
    pub w_ro_init: ParamId,
    pub w_ro_k: ParamId,
    pub w_ro_sens_q: ParamId,
    pub w_ro_byte: ParamId,
    pub b_ro_byte: ParamId,
    pub w_ro_stop: ParamId,
    pub b_ro_stop: ParamId,
    pub w_ro_step: ParamId,
    pub b_ro_step: ParamId,

    // Tête de couplage relationnel
    pub w_rel_proj: ParamId,
    pub b_rel: ParamId,
}

pub fn build_edge_topology(num_nodes: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let total = num_nodes * num_nodes;
    let mut src = Vec::with_capacity(total);
    let mut dst = Vec::with_capacity(total);
    let mut rev = Vec::with_capacity(total);
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            src.push(i);
            dst.push(j);
            rev.push(j * num_nodes + i);
        }
    }
    (src, dst, rev)
}

impl GraphParams {
    pub fn init(cfg: ModelConfig, store: &mut ParamStore, rng: &mut Rng) -> Self {
        let k = cfg.num_nodes;
        let d = cfg.dim;
        let e = k * k;
        let p = cfg.pos_dim;
        let (src_idx, dst_idx, rev_edge_idx) = build_edge_topology(k);

        let byte_std = 1.0 / (d as f32).sqrt();
        let byte_embed = store.add_normal("byte_embed", vec![256, d], byte_std, rng);
        let w_sens_in = store.add_xavier("w_sens_in", 5 * d + p, d, rng);
        let b_sens_in = store.add_zeros("b_sens_in", vec![1, d]);
        let w_sens_k = store.add_xavier("w_sens_k", d, d, rng);
        // Variance accrue pour différencier nettement les K récepteurs régionaux dès t=0
        let regional_receptors = store.add_normal("regional_receptors", vec![k, d], 1.5 * byte_std, rng);

        let edge_std = 0.45 / ((d * k) as f32).sqrt();
        let w_slow_edges = store.add_normal("w_slow_edges", vec![e, d, d], edge_std, rng);

        let w_read_q = store.add_xavier("w_read_q", d, d, rng);
        let w_route_gate = store.add_xavier("w_route_gate", 3 * d, 1, rng);
        let b_route_gate = store.add_zeros("b_route_gate", vec![1, 1]);

        let w_upd = store.add_xavier("w_upd", 3 * d, d, rng);
        let b_upd = store.add_zeros("b_upd", vec![1, d]);
        let w_beta = store.add_xavier("w_beta", 3 * d, d, rng);
        let b_beta = store.add_param("b_beta", vec![-0.5; d], vec![1, d]);

        let w_plast_pre = store.add_xavier("w_plast_pre", d, d, rng);
        let w_plast_err = store.add_xavier("w_plast_err", d, d, rng);
        let w_plast_post = store.add_xavier("w_plast_post", d, d, rng);

        let w_eta = store.add_xavier("w_eta", 3 * d, 1, rng);
        let b_eta = store.add_param("b_eta", vec![-0.5], vec![1, 1]);
        let w_gamma = store.add_xavier("w_gamma", 3 * d, 1, rng);
        let b_gamma = store.add_zeros("b_gamma", vec![1, 1]);
        let w_decay = store.add_xavier("w_decay", 3 * d, 1, rng);
        let b_decay = store.add_param("b_decay", vec![-1.0], vec![1, 1]);

        let w_halt = store.add_xavier("w_halt", 2 * d + 1, 1, rng);
        let b_halt = store.add_param("b_halt", vec![-1.2], vec![1, 1]);

        let w_closure_q = store.add_xavier("w_closure_q", p, d, rng);
        let w_closure_k = store.add_xavier("w_closure_k", d, d, rng);
        let w_closure_sens_q = store.add_xavier("w_closure_sens_q", 2 * d, d, rng);
        let w_closure_out = store.add_xavier("w_closure_out", 2 * d, 256, rng);
        let b_closure_out = store.add_zeros("b_closure_out", vec![1, 256]);

        let w_ro_init = store.add_xavier("w_ro_init", d, d, rng);
        let w_ro_k = store.add_xavier("w_ro_k", d, d, rng);
        let w_ro_sens_q = store.add_xavier("w_ro_sens_q", 2 * d, d, rng);
        let w_ro_byte = store.add_xavier("w_ro_byte", 2 * d, 256, rng);
        let b_ro_byte = store.add_zeros("b_ro_byte", vec![1, 256]);
        let w_ro_stop = store.add_xavier("w_ro_stop", 2 * d, 1, rng);
        let b_ro_stop = store.add_param("b_ro_stop", vec![-1.0], vec![1, 1]);
        let w_ro_step = store.add_xavier("w_ro_step", 4 * d, d, rng);
        let b_ro_step = store.add_zeros("b_ro_step", vec![1, d]);

        let w_rel_proj = store.add_xavier("w_rel_proj", d, d, rng);
        let b_rel = store.add_param("b_rel", vec![-0.5], vec![1, 1]);

        Self {
            cfg,
            src_idx,
            dst_idx,
            rev_edge_idx,
            byte_embed,
            w_sens_in,
            b_sens_in,
            w_sens_k,
            regional_receptors,
            w_slow_edges,
            w_read_q,
            w_route_gate,
            b_route_gate,
            w_upd,
            b_upd,
            w_beta,
            b_beta,
            w_plast_pre,
            w_plast_err,
            w_plast_post,
            w_eta,
            b_eta,
            w_gamma,
            b_gamma,
            w_decay,
            b_decay,
            w_halt,
            b_halt,
            w_closure_q,
            w_closure_k,
            w_closure_sens_q,
            w_closure_out,
            b_closure_out,
            w_ro_init,
            w_ro_k,
            w_ro_sens_q,
            w_ro_byte,
            b_ro_byte,
            w_ro_stop,
            b_ro_stop,
            w_ro_step,
            b_ro_step,
            w_rel_proj,
            b_rel,
        }
    }

    pub fn from_store(cfg: ModelConfig, store: &ParamStore) -> Result<Self, String> {
        let mut by_name: HashMap<&str, ParamId> = HashMap::with_capacity(store.names.len());
        for (idx, name) in store.names.iter().enumerate() {
            by_name.insert(name.as_str(), ParamId(idx));
        }
        let get = |name: &str, expected: &[usize]| -> Result<ParamId, String> {
            let pid = *by_name
                .get(name)
                .ok_or_else(|| format!("Paramètre manquant dans .phis: {}", name))?;
            if store.shapes[pid.0] != expected {
                return Err(format!(
                    "Forme invalide pour {}: attendu {:?}, reçu {:?}",
                    name, expected, store.shapes[pid.0]
                ));
            }
            Ok(pid)
        };

        let k = cfg.num_nodes;
        let d = cfg.dim;
        let e = k * k;
        let p = cfg.pos_dim;
        let (src_idx, dst_idx, rev_edge_idx) = build_edge_topology(k);

        Ok(Self {
            byte_embed: get("byte_embed", &[256, d])?,
            w_sens_in: get("w_sens_in", &[5 * d + p, d])?,
            b_sens_in: get("b_sens_in", &[1, d])?,
            w_sens_k: get("w_sens_k", &[d, d])?,
            regional_receptors: get("regional_receptors", &[k, d])?,
            w_slow_edges: get("w_slow_edges", &[e, d, d])?,
            w_read_q: get("w_read_q", &[d, d])?,
            w_route_gate: get("w_route_gate", &[3 * d, 1])?,
            b_route_gate: get("b_route_gate", &[1, 1])?,
            w_upd: get("w_upd", &[3 * d, d])?,
            b_upd: get("b_upd", &[1, d])?,
            w_beta: get("w_beta", &[3 * d, d])?,
            b_beta: get("b_beta", &[1, d])?,
            w_plast_pre: get("w_plast_pre", &[d, d])?,
            w_plast_err: get("w_plast_err", &[d, d])?,
            w_plast_post: get("w_plast_post", &[d, d])?,
            w_eta: get("w_eta", &[3 * d, 1])?,
            b_eta: get("b_eta", &[1, 1])?,
            w_gamma: get("w_gamma", &[3 * d, 1])?,
            b_gamma: get("b_gamma", &[1, 1])?,
            w_decay: get("w_decay", &[3 * d, 1])?,
            b_decay: get("b_decay", &[1, 1])?,
            w_halt: get("w_halt", &[2 * d + 1, 1])?,
            b_halt: get("b_halt", &[1, 1])?,
            w_closure_q: get("w_closure_q", &[p, d])?,
            w_closure_k: get("w_closure_k", &[d, d])?,
            w_closure_sens_q: get("w_closure_sens_q", &[2 * d, d])?,
            w_closure_out: get("w_closure_out", &[2 * d, 256])?,
            b_closure_out: get("b_closure_out", &[1, 256])?,
            w_ro_init: get("w_ro_init", &[d, d])?,
            w_ro_k: get("w_ro_k", &[d, d])?,
            w_ro_sens_q: get("w_ro_sens_q", &[2 * d, d])?,
            w_ro_byte: get("w_ro_byte", &[2 * d, 256])?,
            b_ro_byte: get("b_ro_byte", &[1, 256])?,
            w_ro_stop: get("w_ro_stop", &[2 * d, 1])?,
            b_ro_stop: get("b_ro_stop", &[1, 1])?,
            w_ro_step: get("w_ro_step", &[4 * d, d])?,
            b_ro_step: get("b_ro_step", &[1, d])?,
            w_rel_proj: get("w_rel_proj", &[d, d])?,
            b_rel: get("b_rel", &[1, 1])?,
            cfg,
            src_idx,
            dst_idx,
            rev_edge_idx,
        })
    }
}

pub fn positional_features(total_len: usize, start: usize, end: usize, pos_dim: usize) -> TensorData {
    assert!(end > start && end <= total_len);
    let rows = end - start;
    let mut data = Vec::with_capacity(rows * pos_dim);
    let denom = (total_len.saturating_sub(1)).max(1) as f32;
    let num_freqs = pos_dim / 2;

    for pos in start..end {
        let u = (pos as f32) / denom;
        for f in 0..num_freqs {
            let freq = std::f32::consts::PI * (1u32 << (f.min(11))) as f32;
            data.push((u * freq).sin());
            data.push((u * freq).cos());
        }
        if pos_dim % 2 == 1 {
            data.push(u * 2.0 - 1.0);
        }
    }
    TensorData::new(data, vec![rows, pos_dim])
}

pub fn encode_sensory_field(
    tape: &mut Tape,
    store: &ParamStore,
    params: &GraphParams,
    bytes: &[u8],
) -> (ValId, ValId) {
    assert!(!bytes.is_empty(), "perceived bytes E must not be empty");
    let len = bytes.len();
    let idx_m2: Vec<usize> = (0..len).map(|p| bytes[p.saturating_sub(2)] as usize).collect();
    let idx_m1: Vec<usize> = (0..len).map(|p| bytes[p.saturating_sub(1)] as usize).collect();
    let idx_0: Vec<usize> = bytes.iter().map(|&b| b as usize).collect();
    let idx_p1: Vec<usize> = (0..len).map(|p| bytes[(p + 1).min(len - 1)] as usize).collect();
    let idx_p2: Vec<usize> = (0..len).map(|p| bytes[(p + 2).min(len - 1)] as usize).collect();

    let embed = tape.param(store, params.byte_embed);
    let em2 = tape.gather_rows(embed, idx_m2);
    let em1 = tape.gather_rows(embed, idx_m1);
    let e0 = tape.gather_rows(embed, idx_0);
    let ep1 = tape.gather_rows(embed, idx_p1);
    let ep2 = tape.gather_rows(embed, idx_p2);

    let pos_td = positional_features(len, 0, len, params.cfg.pos_dim);
    let pos_v = tape.tensor(pos_td.data, pos_td.shape);

    let c1 = tape.concat_cols(em2, em1);
    let c2 = tape.concat_cols(c1, e0);
    let c3 = tape.concat_cols(c2, ep1);
    let c4 = tape.concat_cols(c3, ep2);
    let sens_raw = tape.concat_cols(c4, pos_v);

let w_in = tape.param(store, params.w_sens_in);
let b_in = tape.param(store, params.b_sens_in);
let proj_raw = tape.matmul(sens_raw, w_in);
let proj = tape.add(proj_raw, b_in);
let proj_norm = tape.rmsnorm_rows(proj, 1e-5);
let sens = tape.tanh(proj_norm);

let w_k = tape.param(store, params.w_sens_k);
let keys_raw = tape.matmul(sens, w_k);
let sens_keys = tape.rmsnorm_rows(keys_raw, 1e-5);
(sens, sens_keys)
}

pub fn trigger_regional_r(
    tape: &mut Tape,
    store: &ParamStore,
    params: &GraphParams,
    sens: ValId,
    sens_keys: ValId,
) -> (ValId, ValId) {
    let rec_raw = tape.param(store, params.regional_receptors);
    let rec = tape.rmsnorm_rows(rec_raw, 1e-5);

    let scale = 2.5 / (params.cfg.dim as f32).sqrt();
    let k_t = tape.transpose_2d(sens_keys);
    let logits_raw = tape.matmul(rec, k_t);
    let logits = tape.scale(logits_raw, scale);
    let attn0 = tape.softmax_rows(logits);

    let r_raw = tape.matmul(attn0, sens);
    let r = tape.rmsnorm_rows(r_raw, 1e-5);
    (r, attn0)
}

pub fn non_reciprocity_ratio(tape: &Tape, params: &GraphParams, w_eff: ValId) -> f32 {
    let data = tape.data(w_eff);
    let d = params.cfg.dim;
    let stride = d * d;
    let e = params.cfg.num_nodes * params.cfg.num_nodes;
    let mut diff_sq = 0.0f64;
    let mut norm_sq = 0.0f64;

    for edge in 0..e {
        let rev = params.rev_edge_idx[edge];
        if edge == rev {
            continue;
        }
        for r in 0..d {
            for c in 0..d {
                let w_ij = data[edge * stride + r * d + c] as f64;
                let w_ji_t = data[rev * stride + c * d + r] as f64;
                let diff = w_ij - w_ji_t;
                diff_sq += diff * diff;
                norm_sq += w_ij * w_ij;
            }
        }
    }
    if norm_sq < 1e-12 {
        0.0
    } else {
        (diff_sq / norm_sq).sqrt() as f32
    }
}