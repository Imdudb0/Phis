use std::collections::HashSet;
use std::fs;
use std::path::Path;

use crate::autograd::Tape;
use crate::core::*;

pub(crate) const MAGIC: &[u8; 8] = b"PHIS24V4";

const D2: usize = DIM * DIM;

const OFF_W_LAT1: usize = 0;
const OFF_W_LAT2: usize = OFF_W_LAT1 + K1 * K1;
const OFF_W_UP_12: usize = OFF_W_LAT2 + K2 * K2;
const OFF_W_DOWN_21: usize = OFF_W_UP_12 + K2 * K1;

const OFF_U_A: usize = OFF_W_DOWN_21 + K1 * K2;
const OFF_U_B: usize = OFF_U_A + D2;
const OFF_U_MA: usize = OFF_U_B + D2;
const OFF_U_MB: usize = OFF_U_MA + D2;
const OFF_U_DOWN_21: usize = OFF_U_MB + D2;
const OFF_U_MOD: usize = OFF_U_DOWN_21 + D2;
const OFF_U_FLOW: usize = OFF_U_MOD + D2;
const OFF_U_META: usize = OFF_U_FLOW + D2;
const OFF_U_OUT: usize = OFF_U_META + D2;
const OFF_U_MUL: usize = OFF_U_OUT + D2;
const OFF_W_READ: usize = OFF_U_MUL + D2;
const OFF_U_RES: usize = OFF_W_READ + D2;
const OFF_U_CTX: usize = OFF_U_RES + D2;
const OFF_U_LAT1: usize = OFF_U_CTX + D2;
const OFF_U_UP_12: usize = OFF_U_LAT1 + D2;
const OFF_U_LAT2: usize = OFF_U_UP_12 + D2;
const OFF_U_ABS_21: usize = OFF_U_LAT2 + D2;
const OFF_U_REF1: usize = OFF_U_ABS_21 + D2;
const OFF_U_REF2: usize = OFF_U_REF1 + D2;

const OFF_B1: usize = OFF_U_REF2 + D2;
const OFF_B2: usize = OFF_B1 + DIM;
const OFF_B_ABS: usize = OFF_B2 + DIM;
const OFF_GATES: usize = OFF_B_ABS + DIM;
pub(crate) const PARAM_COUNT: usize = OFF_GATES + 8;

#[derive(Clone, Copy)]
pub(crate) struct ParamsTape {
    pub(crate) root: usize,
    u_a: usize,
    u_b: usize,
    u_ma: usize,
    u_mb: usize,
    u_down_21: usize,
    u_mod: usize,
    u_flow: usize,
    u_meta: usize,
    u_out: usize,
    u_mul: usize,
    w_read: usize,
    u_res: usize,
    u_ctx: usize,
    u_lat1: usize,
    u_up_12: usize,
    u_lat2: usize,
    u_abs_21: usize,
    u_ref1: usize,
    u_ref2: usize,
    b1: usize,
    b2: usize,
    b_abs: usize,
}

impl ParamsTape {
    pub(crate) fn from_model(t: &mut Tape, model: &PhisharpV4) -> Self {
        let root = t.leaf(model.params.clone());
        Self {
            root,
            u_a: t.slice(root, OFF_U_A, D2),
            u_b: t.slice(root, OFF_U_B, D2),
            u_ma: t.slice(root, OFF_U_MA, D2),
            u_mb: t.slice(root, OFF_U_MB, D2),
            u_down_21: t.slice(root, OFF_U_DOWN_21, D2),
            u_mod: t.slice(root, OFF_U_MOD, D2),
            u_flow: t.slice(root, OFF_U_FLOW, D2),
            u_meta: t.slice(root, OFF_U_META, D2),
            u_out: t.slice(root, OFF_U_OUT, D2),
            u_mul: t.slice(root, OFF_U_MUL, D2),
            w_read: t.slice(root, OFF_W_READ, D2),
            u_res: t.slice(root, OFF_U_RES, D2),
            u_ctx: t.slice(root, OFF_U_CTX, D2),
            u_lat1: t.slice(root, OFF_U_LAT1, D2),
            u_up_12: t.slice(root, OFF_U_UP_12, D2),
            u_lat2: t.slice(root, OFF_U_LAT2, D2),
            u_abs_21: t.slice(root, OFF_U_ABS_21, D2),
            u_ref1: t.slice(root, OFF_U_REF1, D2),
            u_ref2: t.slice(root, OFF_U_REF2, D2),
            b1: t.slice(root, OFF_B1, DIM),
            b2: t.slice(root, OFF_B2, DIM),
            b_abs: t.slice(root, OFF_B_ABS, DIM),
        }
    }

    #[inline]
    fn w_lat1(&self, t: &mut Tape, dst: usize, src: usize) -> Option<usize> {
        if dst < K1 && src < K1 {
            Some(t.slice(self.root, OFF_W_LAT1 + dst * K1 + src, 1))
        } else {
            None
        }
    }

    #[inline]
    fn w_lat2(&self, t: &mut Tape, dst: usize, src: usize) -> Option<usize> {
        if dst < K2 && src < K2 {
            Some(t.slice(self.root, OFF_W_LAT2 + dst * K2 + src, 1))
        } else {
            None
        }
    }

    #[inline]
    fn w_up_12(&self, t: &mut Tape, dst2: usize, src1: usize) -> Option<usize> {
        if dst2 < K2 && src1 < K1 {
            Some(t.slice(self.root, OFF_W_UP_12 + dst2 * K1 + src1, 1))
        } else {
            None
        }
    }

    #[inline]
    fn w_down_21(&self, t: &mut Tape, dst1: usize, src2: usize) -> Option<usize> {
        if dst1 < K1 && src2 < K2 {
            Some(t.slice(self.root, OFF_W_DOWN_21 + dst1 * K2 + src2, 1))
        } else {
            None
        }
    }

    #[inline]
    fn gate(&self, t: &mut Tape, idx: usize) -> usize {
        t.slice(self.root, OFF_GATES + idx, 1)
    }
}

#[derive(Clone)]
pub(crate) struct NmState {
    pub(crate) h1: Vec<usize>,
    pub(crate) h2: Vec<usize>,
}

#[derive(Clone)]
pub(crate) struct RawState {
    pub(crate) h1: Vec<[f64; DIM]>,
    pub(crate) h2: Vec<[f64; DIM]>,
}

impl RawState {
    pub(crate) fn zeros(k1: usize, k2: usize) -> Self {
        Self {
            h1: vec![[0.0_f64; DIM]; k1],
            h2: vec![[0.0_f64; DIM]; k2],
        }
    }

    pub(crate) fn from_tape(t: &Tape, st: &NmState) -> Self {
        let mut h1 = Vec::with_capacity(st.h1.len());
        for &id in &st.h1 {
            let mut v = [0.0_f64; DIM];
            v.copy_from_slice(t.val(id));
            h1.push(v);
        }
        let mut h2 = Vec::with_capacity(st.h2.len());
        for &id in &st.h2 {
            let mut v = [0.0_f64; DIM];
            v.copy_from_slice(t.val(id));
            h2.push(v);
        }
        Self { h1, h2 }
    }

    pub(crate) fn into_tape(&self, t: &mut Tape) -> NmState {
        let h1 = self.h1.iter().map(|v| t.leaf(v.to_vec())).collect();
        let h2 = self.h2.iter().map(|v| t.leaf(v.to_vec())).collect();
        NmState { h1, h2 }
    }

    pub(crate) fn with_dead_l1(&self) -> Self {
        let mut out = self.clone();
        out.h1.push([0.0_f64; DIM]);
        out
    }

    pub(crate) fn with_cloned_l1(&self, idx: usize) -> Self {
        let mut out = self.clone();
        out.h1.push(self.h1[idx]);
        out
    }

    pub(crate) fn with_cloned_l2(&self, idx: usize) -> Self {
        let mut out = self.clone();
        out.h2.push(self.h2[idx]);
        out
    }
}

pub(crate) struct OpFields {
    h1_eff: Vec<usize>,
    v_flow: Vec<usize>,
    gate_h: Vec<usize>,
    g_act: usize,
    inv_tau: usize,
    basis: Vec<usize>,
    ua_e: Vec<usize>,
    ub_e: Vec<usize>,
    uma_e: Vec<usize>,
    umb_e: Vec<usize>,
    read_k: Vec<usize>,
}

pub(crate) struct TablesEval {
    pub(crate) task_gains: [usize; 3],
    pub(crate) gain_mix: usize,
    pub(crate) ll_mix: usize,
    pub(crate) residual_l1: usize,
    pub(crate) ctx_vecs: [Vec<usize>; 3],
    pub(crate) attn_vecs: [Vec<usize>; 3],
    pub(crate) probs: [Vec<usize>; 3],
}

#[derive(Clone, Copy)]
pub(crate) enum CutSpec {
    L1(u32),
    L2(u32),
    Vertical,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PhiReport {
    pub(crate) generativity: f64,
    pub(crate) discrimination: f64,
    pub(crate) projection: f64,
    pub(crate) proj_temp: f64,
    pub(crate) proj_comp: f64,
    pub(crate) phi_cut: f64,
    pub(crate) s_ref: f64,
    pub(crate) s_disp: f64,
    pub(crate) s_abs: f64,
    pub(crate) residual_norm: f64,
    pub(crate) phi_sharp: f64,
    pub(crate) shaping_score: f64,
}

pub(crate) struct PhiTapeOut {
    pub(crate) phi_sharp: usize,
    pub(crate) shaping_score: usize,
    pub(crate) report: PhiReport,
}

#[derive(Clone)]
pub(crate) struct PhisharpV4 {
    pub(crate) params: Vec<f64>,
}

impl PhisharpV4 {
    pub(crate) fn new_random(rng: &mut Rng) -> Self {
        let mut params = vec![0.0_f64; PARAM_COUNT];
        for (i, p) in params.iter_mut().enumerate() {
            let scale = if i < OFF_U_A {
                0.45_f64
            } else if i < OFF_B1 {
                0.32_f64
            } else {
                0.10_f64
            };
            *p = rng.next_normal() * scale;
        }
        Self { params }
    }

    pub(crate) fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut buf = Vec::with_capacity(16 + self.params.len() * 8);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(self.params.len() as u64).to_le_bytes());
        for &p in &self.params {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        fs::write(path, buf)
    }

    pub(crate) fn load(path: &Path) -> std::io::Result<Self> {
        let data = fs::read(path)?;
        if data.len() < 16 || &data[0..8] != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Format de modèle .phis V4 invalide",
            ));
        }
        let count = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        if count != PARAM_COUNT || data.len() != 16 + count * 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Nombre de paramètres V4 incompatible",
            ));
        }
        let mut params = Vec::with_capacity(count);
        for chunk in data[16..].chunks_exact(8) {
            params.push(f64::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(Self { params })
    }

    pub(crate) fn seeded_state(t: &mut Tape) -> NmState {
        let mut h1 = Vec::with_capacity(K1);
        for k in 0..K1 {
            let mut v = vec![0.0_f64; DIM];
            for d in 0..DIM {
                v[d] = 0.25_f64 * (((k + 1) * (d + 3)) as f64 * 1.19_f64).sin();
            }
            let id = t.leaf(v);
            h1.push(t.rms_norm(id, 0.25_f64));
        }
        let mut h2 = Vec::with_capacity(K2);
        for m in 0..K2 {
            let mut v = vec![0.0_f64; DIM];
            for d in 0..DIM {
                v[d] = 0.25_f64 * (((m + 2) * (d + 1)) as f64 * 1.29_f64).cos();
            }
            let id = t.leaf(v);
            h2.push(t.rms_norm(id, 0.25_f64));
        }
        NmState { h1, h2 }
    }

    pub(crate) fn zero_state(t: &mut Tape, k1: usize, k2: usize) -> NmState {
        let h1 = (0..k1).map(|_| t.leaf(vec![0.0_f64; DIM])).collect();
        let h2 = (0..k2).map(|_| t.leaf(vec![0.0_f64; DIM])).collect();
        NmState { h1, h2 }
    }

    pub(crate) fn regenerate_h1_from_h2(
        t: &mut Tape,
        p: &ParamsTape,
        h2: &[usize],
        k1: usize,
    ) -> Vec<usize> {
        let k2 = h2.len();
        if k2 == 0 {
            return (0..k1).map(|_| t.leaf(vec![0.0_f64; DIM])).collect();
        }
        let msg2: Vec<usize> = h2
            .iter()
            .map(|&hm| t.mat_vec(p.u_abs_21, hm, DIM, DIM))
            .collect();

        let mut h1_abs = Vec::with_capacity(k1);
        for k in 0..k1 {
            let mut acc = p.b_abs;
            for m in 0..k2 {
                if let Some(w) = p.w_down_21(t, k, m) {
                    let term = t.scalar_mul(msg2[m], w);
                    acc = t.add(acc, term);
                }
            }
            let th = t.tanh(acc);
            h1_abs.push(t.rms_norm(th, 0.25_f64));
        }
        h1_abs
    }

    pub(crate) fn prepare_operator(
        t: &mut Tape,
        p: &ParamsTape,
        view: &RelationalView,
        state: &NmState,
        cut_l1: Option<u32>,
        cut_vertical: bool,
    ) -> OpFields {
        let k1 = state.h1.len();
        let k2 = state.h2.len();

        let n1 = if k1 == 0 {
            t.leaf_scalar(0.0_f64)
        } else {
            let stacked = t.stack(state.h1.clone());
            t.smooth_norm(stacked)
        };
        let n2 = if k2 == 0 {
            t.leaf_scalar(0.0_f64)
        } else {
            let stacked = t.stack(state.h2.clone());
            t.smooth_norm(stacked)
        };
        let th1 = t.tanh(n1);
        let g_act = if cut_vertical {
            th1
        } else {
            let th2 = t.tanh(n2);
            t.mul(th1, th2)
        };

        let mut h1_eff = state.h1.clone();
        let mut meta2 = Vec::with_capacity(k2);
        if !cut_vertical && k2 > 0 {
            let down21: Vec<usize> = state
                .h2
                .iter()
                .map(|&hm| t.mat_vec(p.u_down_21, hm, DIM, DIM))
                .collect();
            for &hm in &state.h2 {
                meta2.push(t.mat_vec(p.u_meta, hm, DIM, DIM));
            }
            for k in 0..k1 {
                let mut td = t.leaf(vec![0.0_f64; DIM]);
                for m in 0..k2 {
                    if let Some(w) = p.w_down_21(t, k, m) {
                        let term = t.scalar_mul(down21[m], w);
                        td = t.add(td, term);
                    }
                }
                let td_th = t.tanh(td);
                let mod_h = t.mat_vec(p.u_mod, state.h1[k], DIM, DIM);
                let gated = t.mul(td_th, mod_h);
                let sum = t.add(state.h1[k], gated);
                h1_eff[k] = t.rms_norm(sum, 0.25_f64);
            }
        }

        let flow_msg: Vec<usize> = h1_eff
            .iter()
            .map(|&hk| t.mat_vec(p.u_flow, hk, DIM, DIM))
            .collect();
        let mut v_flow = Vec::with_capacity(k1);
        let mut gate_h = Vec::with_capacity(k1);

        for k in 0..k1 {
            let mut acc = t.leaf(vec![0.0_f64; DIM]);
            for j in 0..k1 {
                if let Some(mask) = cut_l1 {
                    let sk = ((mask >> k.min(30)) & 1) != 0;
                    let sj = ((mask >> j.min(30)) & 1) != 0;
                    if sk != sj {
                        continue;
                    }
                }
                if let Some(w) = p.w_lat1(t, k, j) {
                    let term = t.scalar_mul(flow_msg[j], w);
                    acc = t.add(acc, term);
                }
            }
            if !cut_vertical {
                for m in 0..k2 {
                    if let Some(w) = p.w_down_21(t, k, m) {
                        let term = t.scalar_mul(meta2[m], w);
                        acc = t.add(acc, term);
                    }
                }
            }
            v_flow.push(t.mat_vec(p.u_out, acc, DIM, DIM));
            gate_h.push(t.mat_vec(p.u_mul, h1_eff[k], DIM, DIM));
        }

        let basis: Vec<usize> = view.basis.iter().map(|b| t.leaf(b.clone())).collect();
        let ua_e: Vec<usize> = basis
            .iter()
            .map(|&e| t.mat_vec(p.u_a, e, DIM, DIM))
            .collect();
        let ub_e: Vec<usize> = basis
            .iter()
            .map(|&e| t.mat_vec(p.u_b, e, DIM, DIM))
            .collect();
        let uma_e: Vec<usize> = basis
            .iter()
            .map(|&e| t.mat_vec(p.u_ma, e, DIM, DIM))
            .collect();
        let umb_e: Vec<usize> = basis
            .iter()
            .map(|&e| t.mat_vec(p.u_mb, e, DIM, DIM))
            .collect();
        let read_k: Vec<usize> = basis
            .iter()
            .map(|&e| t.mat_vec(p.w_read, e, DIM, DIM))
            .collect();

        let g_tau = p.gate(t, 4);
        let sig_tau = t.sigmoid(g_tau);
        let scaled_tau = t.scale(sig_tau, 3.6_f64);
        let inv_tau = t.add_scalar(scaled_tau, 0.80_f64);

        OpFields {
            h1_eff,
            v_flow,
            gate_h,
            g_act,
            inv_tau,
            basis,
            ua_e,
            ub_e,
            uma_e,
            umb_e,
            read_k,
        }
    }

    fn apply_op(t: &mut Tape, f: &OpFields, c: usize) -> (usize, usize) {
        let k1 = f.h1_eff.len();
        if k1 == 0 {
            let z = t.leaf(vec![0.0_f64; DIM]);
            let a = t.leaf(Vec::new());
            return (z, a);
        }
        let inv_sqrt_d = 1.0_f64 / (DIM as f64).sqrt();
        let mut dots = Vec::with_capacity(k1);
        for k in 0..k1 {
            let d = t.dot(c, f.h1_eff[k]);
            dots.push(t.scale(d, inv_sqrt_d));
        }
        let attn_logits = t.stack(dots);
        let attn = t.softmax(attn_logits);

        let mut branches = Vec::with_capacity(k1);
        for k in 0..k1 {
            let cm = t.mul(c, f.gate_h[k]);
            let sum = t.add(f.v_flow[k], cm);
            branches.push(t.tanh(sum));
        }
        let mix = t.weighted_sum(attn, branches);
        let normed = t.rms_norm(mix, 0.25_f64);
        let y = t.scalar_mul(normed, f.g_act);
        (y, attn)
    }

    fn read_logits(t: &mut Tape, f: &OpFields, y: usize) -> usize {
        let v = f.read_k.len();
        let mut dots = Vec::with_capacity(v);
        for w in 0..v {
            dots.push(t.dot(y, f.read_k[w]));
        }
        let raw = t.stack(dots);
        t.scalar_mul(raw, f.inv_tau)
    }

    pub(crate) fn unfold_tables(
        t: &mut Tape,
        p: &ParamsTape,
        view: &RelationalView,
        tables: &NonMarkovTables,
        state: &NmState,
        cut_l1: Option<u32>,
        cut_vertical: bool,
        task_mask: [bool; 3],
    ) -> (OpFields, TablesEval) {
        let f = Self::prepare_operator(t, p, view, state, cut_l1, cut_vertical);
        let mut task_gains = [
            t.leaf_scalar(0.0_f64),
            t.leaf_scalar(0.0_f64),
            t.leaf_scalar(0.0_f64),
        ];
        let mut task_lls = [
            t.leaf_scalar(0.0_f64),
            t.leaf_scalar(0.0_f64),
            t.leaf_scalar(0.0_f64),
        ];
        let mut task_l1s = [
            t.leaf_scalar(0.0_f64),
            t.leaf_scalar(0.0_f64),
            t.leaf_scalar(0.0_f64),
        ];

        let mut ctx_vecs: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut attn_vecs: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut probs: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];

        for m in 0..3 {
            if !task_mask[m] || tables.tables[m].is_empty() {
                continue;
            }
            let entries = &tables.tables[m];
            let mut nll_acc = t.leaf_scalar(0.0_f64);
            let mut l1_acc = t.leaf_scalar(0.0_f64);

            for entry in entries {
                let (c, y, attn) = match m {
                    0 => {
                        let a = f.ua_e[entry.u];
                        let b = f.ub_e[entry.v];
                        let ab = t.mul(a, b);
                        let s1 = t.add(a, b);
                        let s2 = t.add(s1, ab);
                        let c0 = t.tanh(s2);
                        let (y0, at0) = Self::apply_op(t, &f, c0);
                        (c0, y0, at0)
                    }
                    1 => {
                        let a = f.uma_e[entry.u];
                        let b = f.umb_e[entry.v];
                        let ab = t.mul(a, b);
                        let s1 = t.add(a, b);
                        let s2 = t.add(s1, ab);
                        let c1 = t.tanh(s2);
                        let (y1, at1) = Self::apply_op(t, &f, c1);
                        (c1, y1, at1)
                    }
                    _ => {
                        let a = f.ua_e[entry.u];
                        let b = f.ub_e[entry.v];
                        let ab = t.mul(a, b);
                        let s1 = t.add(a, b);
                        let s2 = t.add(s1, ab);
                        let c_first = t.tanh(s2);
                        let (y_first, _) = Self::apply_op(t, &f, c_first);

                        let a2 = f.ua_e[entry.v];
                        let b2 = t.mat_vec(p.u_b, y_first, DIM, DIM);
                        let ab2 = t.mul(a2, b2);
                        let s3 = t.add(a2, b2);
                        let s4 = t.add(s3, ab2);
                        let c2 = t.tanh(s4);
                        let (y2, at2) = Self::apply_op(t, &f, c2);
                        (c2, y2, at2)
                    }
                };

                let logits = Self::read_logits(t, &f, y);
                let pr = t.softmax(logits);
                let ce = t.cross_entropy_logits(logits, entry.target.clone());
                let w_ce = t.scale(ce, entry.weight);
                nll_acc = t.add(nll_acc, w_ce);

                let tgt_node = t.leaf(entry.target.clone());
                let l1 = t.smooth_l1_half(pr, tgt_node);
                let w_l1 = t.scale(l1, entry.weight);
                l1_acc = t.add(l1_acc, w_l1);

                ctx_vecs[m].push(c);
                attn_vecs[m].push(attn);
                probs[m].push(pr);
            }

            let hm = tables.h_marg[m].max(0.05_f64);
            let neg_nll = t.scale(nll_acc, -1.0_f64);
            let diff = t.add_scalar(neg_nll, hm);
            let raw_g = t.scale(diff, 1.0_f64 / hm);
            task_gains[m] = t.clamp(raw_g, -1.0_f64, 1.0_f64);
            task_lls[m] = neg_nll;
            task_l1s[m] = l1_acc;
        }

        let mut w_sum = 0.0_f64;
        for m in 0..3 {
            if task_mask[m] && !tables.tables[m].is_empty() {
                w_sum += TASK_W[m];
            }
        }
        let inv_w = if w_sum > 1e-9_f64 { 1.0_f64 / w_sum } else { 1.0_f64 };

        let mut gain_mix = t.leaf_scalar(0.0_f64);
        let mut ll_mix = t.leaf_scalar(0.0_f64);
        let mut residual_l1 = t.leaf_scalar(0.0_f64);
        for m in 0..3 {
            if task_mask[m] && !tables.tables[m].is_empty() {
                let cw = TASK_W[m] * inv_w;
                let g_term = t.scale(task_gains[m], cw);
                let l_term = t.scale(task_lls[m], cw);
                let r_term = t.scale(task_l1s[m], cw);
                gain_mix = t.add(gain_mix, g_term);
                ll_mix = t.add(ll_mix, l_term);
                residual_l1 = t.add(residual_l1, r_term);
            }
        }

        (
            f,
            TablesEval {
                task_gains,
                gain_mix,
                ll_mix,
                residual_l1,
                ctx_vecs,
                attn_vecs,
                probs,
            },
        )
    }

    pub(crate) fn abductive_step(
        t: &mut Tape,
        p: &ParamsTape,
        view: &RelationalView,
        tables: &NonMarkovTables,
        state: &NmState,
        use_refutation: bool,
        task_mask: [bool; 3],
        cut_l1: Option<u32>,
        cut_l2: Option<u32>,
        cut_vertical: bool,
    ) -> NmState {
        let k1 = state.h1.len();
        let k2 = state.h2.len();

        let (f, ev) =
            Self::unfold_tables(t, p, view, tables, state, cut_l1, cut_vertical, task_mask);

        let g0 = p.gate(t, 0);
        let g1 = p.gate(t, 1);
        let g2 = p.gate(t, 2);
        let g3 = p.gate(t, 3);

        let a1_mul = t.mul(g0, ev.residual_l1);
        let a1_pre = t.add(a1_mul, g1);
        let a1_sig = t.sigmoid(a1_pre);
        let a1_sc = t.scale(a1_sig, 0.56_f64);
        let alpha1 = t.add_scalar(a1_sc, 0.14_f64);
        let neg_a1 = t.scale(alpha1, -1.0_f64);
        let one_m_a1 = t.add_scalar(neg_a1, 1.0_f64);

        let a2_mul = t.mul(g2, ev.residual_l1);
        let a2_pre = t.add(a2_mul, g3);
        let a2_sig = t.sigmoid(a2_pre);
        let a2_sc = t.scale(a2_sig, 0.56_f64);
        let alpha2 = t.add_scalar(a2_sc, 0.14_f64);
        let neg_a2 = t.scale(alpha2, -1.0_f64);
        let one_m_a2 = t.add_scalar(neg_a2, 1.0_f64);

        let mut w_sum = 0.0_f64;
        for m in 0..3 {
            if task_mask[m] && !tables.tables[m].is_empty() {
                w_sum += TASK_W[m];
            }
        }
        let inv_w = if w_sum > 1e-9_f64 { 1.0_f64 / w_sum } else { 1.0_f64 };

        let mut e_up: Vec<usize> = (0..k1).map(|_| t.leaf(vec![0.0_f64; DIM])).collect();

        for m in 0..3 {
            if !task_mask[m] || tables.tables[m].is_empty() {
                continue;
            }
            let mw = TASK_W[m] * inv_w;
            for (i, entry) in tables.tables[m].iter().enumerate() {
                let tgt = t.leaf(entry.target.clone());
                let diff = t.sub(tgt, ev.probs[m][i]);
                let r_vec = t.weighted_sum(diff, f.basis.clone());
                let ur = t.mat_vec(p.u_res, r_vec, DIM, DIM);
                let uc = t.mat_vec(p.u_ctx, r_vec, DIM, DIM);
                let inter = t.mul(ev.ctx_vecs[m][i], uc);
                let sum_s = t.add(ur, inter);
                let s_vec = t.tanh(sum_s);
                let s_scaled = t.scale(s_vec, mw * entry.weight * 2.5_f64);

                for k in 0..k1 {
                    let ak = t.slice(ev.attn_vecs[m][i], k, 1);
                    let ak_s = t.scale(ak, 0.85_f64);
                    let gate_k = t.add_scalar(ak_s, 0.15_f64);
                    let contrib = t.scalar_mul(s_scaled, gate_k);
                    e_up[k] = t.add(e_up[k], contrib);
                }
            }
        }

        let lat1_msg: Vec<usize> = state
            .h1
            .iter()
            .map(|&hk| t.mat_vec(p.u_lat1, hk, DIM, DIM))
            .collect();
        let down2_msg: Vec<usize> = state
            .h2
            .iter()
            .map(|&hm| t.mat_vec(p.u_down_21, hm, DIM, DIM))
            .collect();

        let mut next_h1 = Vec::with_capacity(k1);
        for dst in 0..k1 {
            let mut acc = t.add(e_up[dst], p.b1);
            for src in 0..k1 {
                if let Some(mask) = cut_l1 {
                    let sd = ((mask >> dst.min(30)) & 1) != 0;
                    let ss = ((mask >> src.min(30)) & 1) != 0;
                    if sd != ss {
                        continue;
                    }
                }
                if let Some(w) = p.w_lat1(t, dst, src) {
                    let term = t.scalar_mul(lat1_msg[src], w);
                    acc = t.add(acc, term);
                }
            }
            if !cut_vertical {
                for m in 0..k2 {
                    if let Some(w) = p.w_down_21(t, dst, m) {
                        let term = t.scalar_mul(down2_msg[m], w);
                        let sc = t.scale(term, 0.65_f64);
                        acc = t.add(acc, sc);
                    }
                }
            }
            let drv = t.tanh(acc);
            let old_part = t.scalar_mul(state.h1[dst], one_m_a1);
            let new_part = t.scalar_mul(drv, alpha1);
            let raw = t.add(old_part, new_part);
            next_h1.push(t.rms_norm(raw, 0.25_f64));
        }

        let g5 = p.gate(t, 5);
        let sig_g5 = t.sigmoid(g5);
        let sc_g5 = t.scale(sig_g5, 0.80_f64);
        let beta_dyn = t.add_scalar(sc_g5, 0.20_f64);

        let mut e1_reified = Vec::with_capacity(k1);
        for k in 0..k1 {
            let dh = t.sub(next_h1[k], state.h1[k]);
            let dh_s = t.scalar_mul(dh, beta_dyn);
            let s1 = t.add(next_h1[k], dh_s);
            let e_sc = t.scale(e_up[k], 0.70_f64);
            let s2 = t.add(s1, e_sc);
            e1_reified.push(t.mat_vec(p.u_up_12, s2, DIM, DIM));
        }

        let lat2_msg: Vec<usize> = state
            .h2
            .iter()
            .map(|&hm| t.mat_vec(p.u_lat2, hm, DIM, DIM))
            .collect();

        let mut next_h2 = Vec::with_capacity(k2);
        for dst in 0..k2 {
            let mut acc = p.b2;
            if !cut_vertical {
                for src1 in 0..k1 {
                    if let Some(w) = p.w_up_12(t, dst, src1) {
                        let term = t.scalar_mul(e1_reified[src1], w);
                        acc = t.add(acc, term);
                    }
                }
            }
            for src in 0..k2 {
                if let Some(mask) = cut_l2 {
                    let sd = ((mask >> dst.min(30)) & 1) != 0;
                    let ss = ((mask >> src.min(30)) & 1) != 0;
                    if sd != ss {
                        continue;
                    }
                }
                if let Some(w) = p.w_lat2(t, dst, src) {
                    let term = t.scalar_mul(lat2_msg[src], w);
                    acc = t.add(acc, term);
                }
            }
            let drv = t.tanh(acc);
            let old_part = t.scalar_mul(state.h2[dst], one_m_a2);
            let new_part = t.scalar_mul(drv, alpha2);
            let raw = t.add(old_part, new_part);
            next_h2.push(t.rms_norm(raw, 0.25_f64));
        }

        let mut next_state = NmState {
            h1: next_h1,
            h2: next_h2,
        };

        if use_refutation
            && cut_l1.is_none()
            && cut_l2.is_none()
            && !cut_vertical
            && k1 > 0
            && k2 > 0
            && task_mask == [true, true, true]
        {
            let peak1 = (0..k1)
                .max_by(|&a, &b| {
                    let na: f64 = t.val(next_state.h1[a]).iter().map(|x| x * x).sum();
                    let nb: f64 = t.val(next_state.h1[b]).iter().map(|x| x * x).sum();
                    na.total_cmp(&nb)
                })
                .unwrap_or(0);
            let peak2 = (0..k2)
                .max_by(|&a, &b| {
                    let na: f64 = t.val(next_state.h2[a]).iter().map(|x| x * x).sum();
                    let nb: f64 = t.val(next_state.h2[b]).iter().map(|x| x * x).sum();
                    na.total_cmp(&nb)
                })
                .unwrap_or(0);

            let mut refuted = next_state.clone();
            let mv1 = t.mat_vec(p.u_ref1, refuted.h1[peak1], DIM, DIM);
            let p1 = t.tanh(mv1);
            let p1_sc = t.scale(p1, 0.35_f64);
            let r1 = t.sub(refuted.h1[peak1], p1_sc);
            refuted.h1[peak1] = t.rms_norm(r1, 0.25_f64);

            let mv2 = t.mat_vec(p.u_ref2, refuted.h2[peak2], DIM, DIM);
            let p2 = t.tanh(mv2);
            let p2_sc = t.scale(p2, 0.35_f64);
            let r2 = t.sub(refuted.h2[peak2], p2_sc);
            refuted.h2[peak2] = t.rms_norm(r2, 0.25_f64);

            let (_, ev_base) =
                Self::unfold_tables(t, p, view, tables, &next_state, None, false, task_mask);
            let (_, ev_ref) =
                Self::unfold_tables(t, p, view, tables, &refuted, None, false, task_mask);

            if t.scalar(ev_ref.gain_mix) > t.scalar(ev_base.gain_mix) + 1e-4_f64 {
                next_state = refuted;
            } else if t.scalar(ev_base.gain_mix) >= t.scalar(ev.gain_mix) {
                let d1 = t.sub(next_state.h1[peak1], refuted.h1[peak1]);
                let d1_sc = t.scale(d1, 0.10_f64);
                let c1 = t.add(next_state.h1[peak1], d1_sc);
                next_state.h1[peak1] = t.rms_norm(c1, 0.25_f64);

                let d2 = t.sub(next_state.h2[peak2], refuted.h2[peak2]);
                let d2_sc = t.scale(d2, 0.10_f64);
                let c2 = t.add(next_state.h2[peak2], d2_sc);
                next_state.h2[peak2] = t.rms_norm(c2, 0.25_f64);
            }
        }

        next_state
    }

    pub(crate) fn run_abduction_tape(
        t: &mut Tape,
        p: &ParamsTape,
        view: &RelationalView,
        tables: &NonMarkovTables,
        budget: usize,
        use_refutation: bool,
        task_mask: [bool; 3],
        capture_steps: &[usize],
    ) -> (NmState, Vec<(usize, NmState)>) {
        let mut st = Self::zero_state(t, K1, K2);
        let mut snaps = Vec::new();
        if capture_steps.contains(&0) {
            snaps.push((0, st.clone()));
        }
        if budget == 0 {
            return (st, snaps);
        }

        st = Self::seeded_state(t);
        for step in 1..=budget {
            st = Self::abductive_step(
                t,
                p,
                view,
                tables,
                &st,
                use_refutation,
                task_mask,
                None,
                None,
                false,
            );
            if capture_steps.contains(&step) {
                snaps.push((step, st.clone()));
            }
        }
        (st, snaps)
    }

    fn find_min_cut_offtape(&self, view: &RelationalView, raw: &RawState) -> Option<CutSpec> {
        let k1 = raw.h1.len();
        let k2 = raw.h2.len();
        if k1 <= 1 || k2 <= 1 {
            return None;
        }
        let mut ft = Tape::new(false);
        let fp = ParamsTape::from_model(&mut ft, self);
        let st = raw.into_tape(&mut ft);

        let full_next = Self::abductive_step(
            &mut ft,
            &fp,
            view,
            &view.obs,
            &st,
            false,
            [true, true, true],
            None,
            None,
            false,
        );
        let (_, ev_full) = Self::unfold_tables(
            &mut ft,
            &fp,
            view,
            &view.obs,
            &full_next,
            None,
            false,
            [true, true, true],
        );

        let tvd_eval = |ft: &Tape, ev_cut: &TablesEval| -> f64 {
            let mut tot = 0.0_f64;
            for m in 0..3 {
                let mut sc = 0.0_f64;
                for (i, e) in view.obs.tables[m].iter().enumerate() {
                    let pa = ft.val(ev_full.probs[m][i]);
                    let pb = ft.val(ev_cut.probs[m][i]);
                    let mut l1 = 0.0_f64;
                    for (&a, &b) in pa.iter().zip(pb.iter()) {
                        l1 += (a - b).abs();
                    }
                    sc += e.weight * 0.5_f64 * l1;
                }
                tot += TASK_W[m] * sc;
            }
            tot
        };

        let mut best_spec = CutSpec::Vertical;
        let mut min_div = f64::INFINITY;

        let masks1 = 1u32 << (k1.min(10) - 1);
        for mask in 1..masks1 {
            let cn = Self::abductive_step(
                &mut ft,
                &fp,
                view,
                &view.obs,
                &st,
                false,
                [true, true, true],
                Some(mask),
                None,
                false,
            );
            let (_, ev_c) = Self::unfold_tables(
                &mut ft,
                &fp,
                view,
                &view.obs,
                &cn,
                Some(mask),
                false,
                [true, true, true],
            );
            let d = tvd_eval(&ft, &ev_c);
            if d < min_div {
                min_div = d;
                best_spec = CutSpec::L1(mask);
            }
        }

        let masks2 = 1u32 << (k2.min(10) - 1);
        for mask in 1..masks2 {
            let cn = Self::abductive_step(
                &mut ft,
                &fp,
                view,
                &view.obs,
                &st,
                false,
                [true, true, true],
                None,
                Some(mask),
                false,
            );
            let (_, ev_c) = Self::unfold_tables(
                &mut ft,
                &fp,
                view,
                &view.obs,
                &cn,
                None,
                false,
                [true, true, true],
            );
            let d = tvd_eval(&ft, &ev_c);
            if d < min_div {
                min_div = d;
                best_spec = CutSpec::L2(mask);
            }
        }

        let vn = Self::abductive_step(
            &mut ft,
            &fp,
            view,
            &view.obs,
            &st,
            false,
            [true, true, true],
            None,
            None,
            true,
        );
        let (_, ev_v) = Self::unfold_tables(
            &mut ft,
            &fp,
            view,
            &view.obs,
            &vn,
            None,
            true,
            [true, true, true],
        );
        let dv = tvd_eval(&ft, &ev_v);
        if dv < min_div {
            best_spec = CutSpec::Vertical;
        }

        Some(best_spec)
    }

    fn tables_pred_tvd(
        t: &mut Tape,
        tables: &NonMarkovTables,
        a: &TablesEval,
        b: &TablesEval,
    ) -> usize {
        let mut tot = t.leaf_scalar(0.0_f64);
        for m in 0..3 {
            if tables.tables[m].is_empty() {
                continue;
            }
            let mut sc = t.leaf_scalar(0.0_f64);
            for (i, e) in tables.tables[m].iter().enumerate() {
                let d = t.smooth_l1_half(a.probs[m][i], b.probs[m][i]);
                let wd = t.scale(d, e.weight);
                sc = t.add(sc, wd);
            }
            let wsc = t.scale(sc, TASK_W[m]);
            tot = t.add(tot, wsc);
        }
        tot
    }

    pub(crate) fn evaluate_phi_tape(
        &self,
        t: &mut Tape,
        p: &ParamsTape,
        view: &RelationalView,
        state: &NmState,
        budget: usize,
        use_refutation: bool,
        pre_proj_temp: Option<&NmState>,
        pre_proj_comp: Option<&NmState>,
    ) -> PhiTapeOut {
        if view.v_size <= 1 || view.seq.len() < 16 || view.obs.tables[0].is_empty() {
            let z = t.leaf_scalar(0.0_f64);
            let neg = t.leaf_scalar(-1.0_f64);
            return PhiTapeOut {
                phi_sharp: z,
                shaping_score: neg,
                report: PhiReport {
                    generativity: 0.0,
                    discrimination: 0.0,
                    projection: 0.0,
                    proj_temp: 0.0,
                    proj_comp: 0.0,
                    phi_cut: 0.0,
                    s_ref: 0.0,
                    s_disp: 0.0,
                    s_abs: 0.0,
                    residual_norm: 1.0,
                    phi_sharp: 0.0,
                    shaping_score: -1.0,
                },
            };
        }

        let (_, ev_e) = Self::unfold_tables(
            t,
            p,
            view,
            &view.obs,
            state,
            None,
            false,
            [true, true, true],
        );
        let gen_raw = ev_e.gain_mix;

        let (_, ev_neg) = Self::unfold_tables(
            t,
            p,
            view,
            &view.shuf,
            state,
            None,
            false,
            [true, true, true],
        );
        let disc_diff = t.sub(gen_raw, ev_neg.gain_mix);
        let disc_raw = t.clamp(disc_diff, -1.0_f64, 1.0_f64);

        let (pt_raw, pc_raw) = if budget == 0 || state.h1.is_empty() || state.h2.is_empty() {
            (t.leaf_scalar(0.0_f64), t.leaf_scalar(0.0_f64))
        } else {
            let mid = view.seq.len() / 2;
            let first_half = view.subslice_same_vocab(0, mid);
            let second_half = view.subslice_same_vocab(mid, view.seq.len());

            let owned_pt;
            let st_pt = if let Some(s) = pre_proj_temp {
                s
            } else {
                owned_pt = Self::run_abduction_tape(
                    t,
                    p,
                    &first_half,
                    &first_half.obs,
                    budget,
                    use_refutation,
                    [true, true, true],
                    &[],
                )
                .0;
                &owned_pt
            };
            let (_, ev_pt) = Self::unfold_tables(
                t,
                p,
                view,
                &second_half.obs,
                st_pt,
                None,
                false,
                [true, true, true],
            );

            let owned_pc;
            let st_pc = if let Some(s) = pre_proj_comp {
                s
            } else {
                owned_pc = Self::run_abduction_tape(
                    t,
                    p,
                    view,
                    &view.obs,
                    budget,
                    false,
                    [true, true, false],
                    &[],
                )
                .0;
                &owned_pc
            };
            let (_, ev_pc) = Self::unfold_tables(
                t,
                p,
                view,
                &view.obs,
                st_pc,
                None,
                false,
                [false, false, true],
            );
            (ev_pt.gain_mix, ev_pc.task_gains[2])
        };
        let proj_raw = t.min_scalar(pt_raw, pc_raw);

        let k1 = state.h1.len();
        let k2 = state.h2.len();
        let phi_cut = if budget == 0 || k1 <= 1 || k2 <= 1 {
            t.leaf_scalar(0.0_f64)
        } else {
            let mut min_n1 = t.smooth_norm(state.h1[0]);
            for k in 1..k1 {
                let nk = t.smooth_norm(state.h1[k]);
                min_n1 = t.min_scalar(min_n1, nk);
            }
            let mut min_n2 = t.smooth_norm(state.h2[0]);
            for m in 1..k2 {
                let nm = t.smooth_norm(state.h2[m]);
                min_n2 = t.min_scalar(min_n2, nm);
            }
            let mut min_dist = t.smooth_dist(state.h1[0], state.h1[1]);
            for a in 0..k1 {
                for b in (a + 1)..k1 {
                    let d = t.smooth_dist(state.h1[a], state.h1[b]);
                    min_dist = t.min_scalar(min_dist, d);
                }
            }
            for a in 0..k2 {
                for b in (a + 1)..k2 {
                    let d = t.smooth_dist(state.h2[a], state.h2[b]);
                    min_dist = t.min_scalar(min_dist, d);
                }
            }
            let mn1_s = t.scale(min_n1, 3.0_f64);
            let f1 = t.tanh(mn1_s);
            let mn2_s = t.scale(min_n2, 3.0_f64);
            let f2 = t.tanh(mn2_s);
            let md_s = t.scale(min_dist, 3.0_f64);
            let fd = t.tanh(md_s);
            let f12 = t.mul(f1, f2);
            let phi_diff = t.mul(f12, fd);

            if t.scalar(phi_diff) <= 1e-9_f64 {
                t.leaf_scalar(0.0_f64)
            } else {
                let raw_st = RawState::from_tape(t, state);
                if let Some(best_cut) = self.find_min_cut_offtape(view, &raw_st) {
                    let full_next = Self::abductive_step(
                        t,
                        p,
                        view,
                        &view.obs,
                        state,
                        false,
                        [true, true, true],
                        None,
                        None,
                        false,
                    );
                    let (_, ev_full) = Self::unfold_tables(
                        t,
                        p,
                        view,
                        &view.obs,
                        &full_next,
                        None,
                        false,
                        [true, true, true],
                    );

                    let (cl1, cl2, cv) = match best_cut {
                        CutSpec::L1(m) => (Some(m), None, false),
                        CutSpec::L2(m) => (None, Some(m), false),
                        CutSpec::Vertical => (None, None, true),
                    };
                    let cut_next = Self::abductive_step(
                        t,
                        p,
                        view,
                        &view.obs,
                        state,
                        false,
                        [true, true, true],
                        cl1,
                        cl2,
                        cv,
                    );
                    let (_, ev_cut) = Self::unfold_tables(
                        t,
                        p,
                        view,
                        &view.obs,
                        &cut_next,
                        cl1,
                        cv,
                        [true, true, true],
                    );
                    let div = Self::tables_pred_tvd(t, &view.obs, &ev_full, &ev_cut);
                    let div_s = t.scale(div, 10.0_f64);
                    let div_th = t.tanh(div_s);
                    let mip = t.clamp(div_th, 0.0_f64, 1.0_f64);
                    t.mul(mip, phi_diff)
                } else {
                    t.leaf_scalar(0.0_f64)
                }
            }
        };

        let (s_ref_raw, s_disp_raw, s_abs_raw) =
            if budget == 0 || t.scalar(gen_raw) <= 1e-6_f64 || k1 == 0 || k2 == 0 {
                (
                    t.leaf_scalar(0.0_f64),
                    t.leaf_scalar(0.0_f64),
                    t.leaf_scalar(0.0_f64),
                )
            } else {
                let floor = t.leaf_scalar(0.05_f64);
                let denom = t.max_scalar(gen_raw, floor);

                let peak1 = (0..k1)
                    .max_by(|&a, &b| {
                        let na: f64 = t.val(state.h1[a]).iter().map(|x| x * x).sum();
                        let nb: f64 = t.val(state.h1[b]).iter().map(|x| x * x).sum();
                        na.total_cmp(&nb)
                    })
                    .unwrap_or(0);
                let mut att = state.clone();
                att.h1[peak1] = t.leaf(vec![0.0_f64; DIM]);
                for _ in 0..2 {
                    att = Self::abductive_step(
                        t,
                        p,
                        view,
                        &view.obs,
                        &att,
                        false,
                        [true, true, true],
                        None,
                        None,
                        false,
                    );
                }
                let (_, ev_rep) = Self::unfold_tables(
                    t,
                    p,
                    view,
                    &view.obs,
                    &att,
                    None,
                    false,
                    [true, true, true],
                );
                let sr = t.div_scalar(ev_rep.gain_mix, denom);

                let q1 = view.seq.len() / 4;
                let q3 = (3 * view.seq.len()) / 4;
                let shifted = view.subslice_same_vocab(q1, q3);
                let (_, ev_disp) = Self::unfold_tables(
                    t,
                    p,
                    view,
                    &shifted.obs,
                    state,
                    None,
                    false,
                    [true, true, true],
                );
                let sd = t.div_scalar(ev_disp.gain_mix, denom);

                let h1_abs = Self::regenerate_h1_from_h2(t, p, &state.h2, k1);
                let abs_st = NmState {
                    h1: h1_abs,
                    h2: state.h2.clone(),
                };
                let (_, ev_abs) = Self::unfold_tables(
                    t,
                    p,
                    view,
                    &view.obs,
                    &abs_st,
                    None,
                    false,
                    [true, true, true],
                );
                let sa = t.div_scalar(ev_abs.gain_mix, denom);

                (sr, sd, sa)
            };

        let s_ref = t.clamp(s_ref_raw, 0.0_f64, 1.0_f64);
        let s_disp = t.clamp(s_disp_raw, 0.0_f64, 1.0_f64);
        let s_abs = t.clamp(s_abs_raw, 0.0_f64, 1.0_f64);

        let s_rd = t.mul(s_ref, s_disp);
        let s_prod = t.mul(s_rd, s_abs);
        let s_abd = t.pow_scalar(s_prod, 1.0_f64 / 3.0_f64);

        let res_c = t.clamp(ev_e.residual_l1, 0.0_f64, 1.0_f64);
        let res_neg = t.scale(res_c, -0.5_f64);
        let res_fac = t.add_scalar(res_neg, 1.0_f64);

        let g_c = t.clamp(gen_raw, 0.0_f64, 1.0_f64);
        let d_c = t.clamp(disc_raw, 0.0_f64, 1.0_f64);
        let p_c = t.clamp(proj_raw, 0.0_f64, 1.0_f64);

        let min_gd = t.min_scalar(g_c, d_c);
        let base_min = t.min_scalar(min_gd, p_c);
        let p1 = t.mul(base_min, phi_cut);
        let p2 = t.mul(p1, s_abd);
        let phi_sharp = t.mul(p2, res_fac);

        let shaping_score = if t.scalar(gen_raw) > 0.0_f64
            && t.scalar(disc_raw) > 0.0_f64
            && t.scalar(proj_raw) > 0.0_f64
        {
            let sa_br = t.clamp(s_abs_raw, -0.5_f64, 1.0_f64);
            let sr_br = t.clamp(s_ref_raw, -0.5_f64, 1.0_f64);
            let sd_br = t.clamp(s_disp_raw, -0.5_f64, 1.0_f64);

            let w_bm = t.scale(base_min, 0.08_f64);
            let b0 = t.add(phi_sharp, w_bm);
            let w_sa = t.scale(sa_br, 0.06_f64);
            let b1 = t.add(b0, w_sa);
            let w_sr = t.scale(sr_br, 0.04_f64);
            let b2 = t.add(b1, w_sr);
            let w_sd = t.scale(sd_br, 0.04_f64);
            let b3 = t.add(b2, w_sd);
            let w_pc = t.scale(phi_cut, 0.04_f64);
            t.add(b3, w_pc)
        } else {
            let min_gd_r = t.min_scalar(gen_raw, disc_raw);
            let min_raw = t.min_scalar(min_gd_r, proj_raw);
            let sum_gd = t.add(gen_raw, disc_raw);
            let sum_gdp = t.add(sum_gd, proj_raw);
            let avg_raw = t.scale(sum_gdp, 1.0_f64 / 3.0_f64);
            let w_min = t.scale(min_raw, 0.55_f64);
            let w_avg = t.scale(avg_raw, 0.25_f64);
            let s1 = t.add(w_min, w_avg);
            let w_res = t.scale(res_c, 0.10_f64);
            t.sub(s1, w_res)
        };

        let report = PhiReport {
            generativity: t.scalar(g_c),
            discrimination: t.scalar(d_c),
            projection: t.scalar(p_c),
            proj_temp: t.scalar(pt_raw).clamp(0.0_f64, 1.0_f64),
            proj_comp: t.scalar(pc_raw).clamp(0.0_f64, 1.0_f64),
            phi_cut: t.scalar(phi_cut),
            s_ref: t.scalar(s_ref),
            s_disp: t.scalar(s_disp),
            s_abs: t.scalar(s_abs),
            residual_norm: t.scalar(res_c),
            phi_sharp: t.scalar(phi_sharp),
            shaping_score: t.scalar(shaping_score),
        };

        PhiTapeOut {
            phi_sharp,
            shaping_score,
            report,
        }
    }

    pub(crate) fn run_abduction_raw(
        &self,
        view: &RelationalView,
        budget: usize,
        use_refutation: bool,
        capture_steps: &[usize],
    ) -> (RawState, Vec<(usize, RawState)>) {
        let mut t = Tape::new(false);
        let p = ParamsTape::from_model(&mut t, self);
        let (st, snaps) = Self::run_abduction_tape(
            &mut t,
            &p,
            view,
            &view.obs,
            budget,
            use_refutation,
            [true, true, true],
            capture_steps,
        );
        let raw_snaps = snaps
            .into_iter()
            .map(|(step, s)| (step, RawState::from_tape(&t, &s)))
            .collect();
        (RawState::from_tape(&t, &st), raw_snaps)
    }

    pub(crate) fn evaluate_phi_raw(
        &self,
        view: &RelationalView,
        raw: &RawState,
        budget: usize,
        use_refutation: bool,
    ) -> PhiReport {
        let mut t = Tape::new(false);
        let p = ParamsTape::from_model(&mut t, self);
        let st = raw.into_tape(&mut t);
        self.evaluate_phi_tape(&mut t, &p, view, &st, budget, use_refutation, None, None)
            .report
    }

    pub(crate) fn evaluate_phi(
        &self,
        view: &RelationalView,
        budget: usize,
        use_refutation: bool,
    ) -> PhiReport {
        let (raw, _) = self.run_abduction_raw(view, budget, use_refutation, &[]);
        self.evaluate_phi_raw(view, &raw, budget, use_refutation)
    }

    pub(crate) fn unfold_predictions_raw(
        &self,
        view: &RelationalView,
        raw: &RawState,
        cut_vertical: bool,
    ) -> ([Vec<Vec<f64>>; 3], f64, f64, f64) {
        let mut t = Tape::new(false);
        let p = ParamsTape::from_model(&mut t, self);
        let st = raw.into_tape(&mut t);
        let (_, ev) = Self::unfold_tables(
            &mut t,
            &p,
            view,
            &view.obs,
            &st,
            None,
            cut_vertical,
            [true, true, true],
        );
        let mut preds: [Vec<Vec<f64>>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for m in 0..3 {
            for &pid in &ev.probs[m] {
                preds[m].push(t.val(pid).to_vec());
            }
        }
        let ll = t.scalar(ev.ll_mix);
        let mdl_node = mdl_tape_v4(&mut t, ev.ll_mix, &st, view.seq.len());
        let mdl = t.scalar(mdl_node);
        let gain = t.scalar(ev.gain_mix);
        (preds, ll, mdl, gain)
    }

    pub(crate) fn ask_emit(&self, view: &RelationalView, budget: usize, out_len: usize) -> Vec<u8> {
        if view.vocab.is_empty() || view.obs.tables[0].is_empty() {
            return Vec::new();
        }
        let (raw, _) = self.run_abduction_raw(view, budget, true, &[]);
        let (preds, _, _, _) = self.unfold_predictions_raw(view, &raw, false);
        let phi_base = self.evaluate_phi_raw(view, &raw, budget, true).phi_sharp;

        let mut pair_res: Vec<(usize, usize, f64)> = view.obs.tables[0]
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let mut l1 = 0.0_f64;
                for (&pt, &pq) in e.target.iter().zip(preds[0][i].iter()) {
                    l1 += (pt - pq).abs();
                }
                (e.u, e.v, 0.5_f64 * l1)
            })
            .collect();
        pair_res.sort_by(|a, b| {
            b.2.total_cmp(&a.2)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });

        let mut e_4grams: HashSet<[usize; 4]> = HashSet::new();
        for w in view.seq.windows(4) {
            e_4grams.insert([w[0], w[1], w[2], w[3]]);
        }

        let n_cands = pair_res.len().min(4).max(1);
        let mut best_cand = Vec::new();
        let mut best_score = f64::NEG_INFINITY;

        for c_i in 0..n_cands {
            let (u0, v0, _) = pair_res[c_i];
            let mut t = Tape::new(false);
            let p = ParamsTape::from_model(&mut t, self);
            let st = raw.into_tape(&mut t);
            let f = Self::prepare_operator(&mut t, &p, view, &st, None, false);

            let mut z_prev = f.basis[u0];
            let mut z_curr = f.basis[v0];
            let mut cand_idx = vec![u0, v0];
            let mut visits = vec![0usize; view.v_size];
            visits[u0] += 1;
            visits[v0] += 1;

            for _ in 2..out_len {
                let a = t.mat_vec(p.u_a, z_prev, DIM, DIM);
                let b = t.mat_vec(p.u_b, z_curr, DIM, DIM);
                let ab = t.mul(a, b);
                let s1 = t.add(a, b);
                let s2 = t.add(s1, ab);
                let c = t.tanh(s2);
                let (y, _) = Self::apply_op(&mut t, &f, c);
                let logits = Self::read_logits(&mut t, &f, y);
                let pr = t.softmax(logits);
                let pv = t.val(pr);

                let mut best_w = 0usize;
                let mut best_p = f64::NEG_INFINITY;
                for (w, &pw) in pv.iter().enumerate() {
                    let sc = pw / (1.0_f64 + 0.35_f64 * (visits[w] as f64));
                    if sc > best_p {
                        best_p = sc;
                        best_w = w;
                    }
                }
                visits[best_w] += 1;
                cand_idx.push(best_w);
                z_prev = z_curr;
                z_curr = y;
            }

            let mut ext_seq = view.seq.clone();
            ext_seq.extend_from_slice(&cand_idx);
            let ext_view = RelationalView::from_indexed(view.vocab.clone(), ext_seq);
            let phi_ext = self.evaluate_phi(&ext_view, budget, true).phi_sharp;

            let echo_ratio = if cand_idx.len() >= 4 {
                let mut hits = 0usize;
                for w in cand_idx.windows(4) {
                    if e_4grams.contains(&[w[0], w[1], w[2], w[3]]) {
                        hits += 1;
                    }
                }
                (hits as f64) / ((cand_idx.len() - 3) as f64)
            } else {
                0.0_f64
            };

            let closure = (phi_ext - phi_base) - 0.15_f64 * echo_ratio;
            if closure > best_score {
                best_score = closure;
                best_cand = cand_idx;
            }
        }

        best_cand.into_iter().map(|i| view.vocab[i]).collect()
    }
}

pub(crate) fn mdl_tape_v4(
    t: &mut Tape,
    ll_mix: usize,
    state: &NmState,
    seq_len: usize,
) -> usize {
    let mut energy = t.leaf_scalar(0.0_f64);
    for &hid in state.h1.iter().chain(state.h2.iter()) {
        let e = t.smooth_abs_sum(hid);
        energy = t.add(energy, e);
    }
    let n = seq_len.max(2) as f64;
    let pen_coeff = 0.012_f64 * n.ln() / n;
    let pen = t.scale(energy, pen_coeff);
    t.sub(ll_mix, pen)
}
