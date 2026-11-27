use std::collections::HashSet;
use std::fs;
use std::path::Path;

use crate::core::*;

pub(crate) const MAGIC: &[u8; 8] = b"PHIS24V3";

// ============================================================================
// Disposition des paramètres (1329 paramètres, poids non-réciproques)
// Zéro fast-weight (L_A / L_E / L_G supprimés) ; zéro pénalité IB/MDL.
// ============================================================================

const OFF_W_LAT1: usize = 0;
const OFF_W_LAT2: usize = OFF_W_LAT1 + K1 * K1;
const OFF_W_UP_12: usize = OFF_W_LAT2 + K2 * K2;
const OFF_W_DOWN_21: usize = OFF_W_UP_12 + K2 * K1;

const OFF_U_X_IN: usize = OFF_W_DOWN_21 + K1 * K2;
const OFF_U_X_OUT: usize = OFF_U_X_IN + DIM * DIM;
const OFF_U_X_TD: usize = OFF_U_X_OUT + DIM * DIM;
const OFF_W_UP_01: usize = OFF_U_X_TD + DIM * DIM;
const OFF_U_OBS_1: usize = OFF_W_UP_01 + DIM * DIM;
const OFF_U_LAT1: usize = OFF_U_OBS_1 + DIM * DIM;
const OFF_U_UP_12: usize = OFF_U_LAT1 + DIM * DIM;
const OFF_U_LAT2: usize = OFF_U_UP_12 + DIM * DIM;
const OFF_U_DOWN_21: usize = OFF_U_LAT2 + DIM * DIM;
const OFF_U_MOD: usize = OFF_U_DOWN_21 + DIM * DIM;
const OFF_U_FLOW: usize = OFF_U_MOD + DIM * DIM;
const OFF_U_META: usize = OFF_U_FLOW + DIM * DIM;
const OFF_U_DOWN_10: usize = OFF_U_META + DIM * DIM;
const OFF_U_ROT: usize = OFF_U_DOWN_10 + DIM * DIM;
const OFF_U_GATE: usize = OFF_U_ROT + DIM * DIM;
const OFF_W_READ: usize = OFF_U_GATE + DIM * DIM;
const OFF_U_ABS_21: usize = OFF_W_READ + DIM * DIM;
const OFF_U_REF1: usize = OFF_U_ABS_21 + DIM * DIM;
const OFF_U_REF2: usize = OFF_U_REF1 + DIM * DIM;

const OFF_B1: usize = OFF_U_REF2 + DIM * DIM;
const OFF_B2: usize = OFF_B1 + DIM;
const OFF_B_ABS: usize = OFF_B2 + DIM;
const OFF_GATES: usize = OFF_B_ABS + DIM;
pub(crate) const PARAM_COUNT: usize = OFF_GATES + 8;

/// État dynamique pur de circulation : uniquement des activations de nœuds (L0, L1, L2).
#[derive(Clone)]
pub(crate) struct CircState {
    pub(crate) x: Vec<[f64; DIM]>,
    pub(crate) h1: Vec<[f64; DIM]>,
    pub(crate) h2: Vec<[f64; DIM]>,
}

impl CircState {
    pub(crate) fn zeros(v_size: usize, k1: usize, k2: usize) -> Self {
        Self {
            x: vec![[0.0_f64; DIM]; v_size],
            h1: vec![[0.0_f64; DIM]; k1],
            h2: vec![[0.0_f64; DIM]; k2],
        }
    }

    /// Initialisation strictement invariante à toute bijection π de l'alphabet :
    /// dépend uniquement de `node_init[u]` et de l'indice d'ordre d'apparition `u`.
    pub(crate) fn seeded(view: &RelationalView) -> Self {
        let mut x = vec![[0.0_f64; DIM]; view.v_size];
        for u in 0..view.v_size {
            for d in 0..DIM {
                let sym_break = 0.28_f64 * (((u + 1) * (d + 2)) as f64 * 1.37_f64).sin();
                x[u][d] = 0.55_f64 * view.node_init[u][d] + sym_break;
            }
            rms_norm(&mut x[u]);
        }
        let mut h1 = vec![[0.0_f64; DIM]; K1];
        for (k, v) in h1.iter_mut().enumerate() {
            for d in 0..DIM {
                v[d] = 0.22_f64 * (((k + 1) * (d + 3)) as f64 * 1.19_f64).sin();
            }
            rms_norm(v);
        }
        let mut h2 = vec![[0.0_f64; DIM]; K2];
        for (m, v) in h2.iter_mut().enumerate() {
            for d in 0..DIM {
                v[d] = 0.22_f64 * (((m + 2) * (d + 1)) as f64 * 1.29_f64).cos();
            }
            rms_norm(v);
        }
        Self { x, h1, h2 }
    }

    pub(crate) fn with_dead_l1_node(&self) -> Self {
        let mut out = self.clone();
        out.h1.push([0.0_f64; DIM]);
        out
    }

    pub(crate) fn with_cloned_l1_node(&self, src_k: usize) -> Self {
        let mut out = self.clone();
        out.h1.push(self.h1[src_k]);
        out
    }

    pub(crate) fn with_cloned_l2_node(&self, src_m: usize) -> Self {
        let mut out = self.clone();
        out.h2.push(self.h2[src_m]);
        out
    }
}

pub(crate) struct CircUnfold {
    pub(crate) h1_eff: Vec<[f64; DIM]>,
    pub(crate) a_ground: Vec<f64>,   // V * k1
    pub(crate) y: [Vec<[f64; DIM]>; 3], // T^1(X), T^2(X), T^3(X)
    pub(crate) r_hat: [Vec<f64>; 3], // V * V
}

#[derive(Clone)]
pub(crate) struct AbductionTrace {
    pub(crate) state: CircState,
    pub(crate) snapshots: Vec<(usize, CircState)>,
    pub(crate) phi_history: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PhiReport {
    pub(crate) generativity: f64,
    pub(crate) discrimination: f64,
    pub(crate) projection: f64,
    pub(crate) proj_temp: f64,
    pub(crate) proj_scale: f64,
    pub(crate) phi_cut: f64,
    pub(crate) s_ref: f64,
    pub(crate) s_disp: f64,
    pub(crate) s_abs: f64,
    pub(crate) residual_norm: f64,
    pub(crate) phi_sharp: f64,
    pub(crate) shaping_score: f64,
}

#[derive(Clone)]
pub(crate) struct PhisharpV3 {
    pub(crate) params: Vec<f64>,
}

impl PhisharpV3 {
    pub(crate) fn new_random(rng: &mut Rng) -> Self {
        let mut params = vec![0.0_f64; PARAM_COUNT];
        for (i, p) in params.iter_mut().enumerate() {
            let scale = if i < OFF_U_X_IN {
                0.45_f64
            } else if i < OFF_B1 {
                0.35_f64
            } else {
                0.15_f64
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
                "Format de modèle .phis V3 invalide",
            ));
        }
        let count = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        if count != PARAM_COUNT || data.len() != 16 + count * 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Nombre de paramètres V3 incompatible",
            ));
        }
        let mut params = Vec::with_capacity(count);
        for chunk in data[16..].chunks_exact(8) {
            params.push(f64::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(Self { params })
    }

    #[inline]
    fn w_lat1(&self, dst: usize, src: usize) -> f64 {
        if dst < K1 && src < K1 {
            self.params[OFF_W_LAT1 + dst * K1 + src]
        } else {
            0.0_f64
        }
    }
    #[inline]
    fn w_lat2(&self, dst: usize, src: usize) -> f64 {
        if dst < K2 && src < K2 {
            self.params[OFF_W_LAT2 + dst * K2 + src]
        } else {
            0.0_f64
        }
    }
    #[inline]
    fn w_up_12(&self, dst2: usize, src1: usize) -> f64 {
        if dst2 < K2 && src1 < K1 {
            self.params[OFF_W_UP_12 + dst2 * K1 + src1]
        } else {
            0.0_f64
        }
    }
    #[inline]
    fn w_down_21(&self, dst1: usize, src2: usize) -> f64 {
        if dst1 < K1 && src2 < K2 {
            self.params[OFF_W_DOWN_21 + dst1 * K2 + src2]
        } else {
            0.0_f64
        }
    }

    fn u_x_in(&self) -> &[f64] { &self.params[OFF_U_X_IN..OFF_U_X_OUT] }
    fn u_x_out(&self) -> &[f64] { &self.params[OFF_U_X_OUT..OFF_U_X_TD] }
    fn u_x_td(&self) -> &[f64] { &self.params[OFF_U_X_TD..OFF_W_UP_01] }
    fn w_up_01(&self) -> &[f64] { &self.params[OFF_W_UP_01..OFF_U_OBS_1] }
    fn u_obs_1(&self) -> &[f64] { &self.params[OFF_U_OBS_1..OFF_U_LAT1] }
    fn u_lat1(&self) -> &[f64] { &self.params[OFF_U_LAT1..OFF_U_UP_12] }
    fn u_up_12(&self) -> &[f64] { &self.params[OFF_U_UP_12..OFF_U_LAT2] }
    fn u_lat2(&self) -> &[f64] { &self.params[OFF_U_LAT2..OFF_U_DOWN_21] }
    fn u_down_21(&self) -> &[f64] { &self.params[OFF_U_DOWN_21..OFF_U_MOD] }
    fn u_mod(&self) -> &[f64] { &self.params[OFF_U_MOD..OFF_U_FLOW] }
    fn u_flow(&self) -> &[f64] { &self.params[OFF_U_FLOW..OFF_U_META] }
    fn u_meta(&self) -> &[f64] { &self.params[OFF_U_META..OFF_U_DOWN_10] }
    fn u_down_10(&self) -> &[f64] { &self.params[OFF_U_DOWN_10..OFF_U_ROT] }
    fn u_rot(&self) -> &[f64] { &self.params[OFF_U_ROT..OFF_U_GATE] }
    fn u_gate(&self) -> &[f64] { &self.params[OFF_U_GATE..OFF_W_READ] }
    fn w_read(&self) -> &[f64] { &self.params[OFF_W_READ..OFF_U_ABS_21] }
    fn u_abs_21(&self) -> &[f64] { &self.params[OFF_U_ABS_21..OFF_U_REF1] }
    fn u_ref1(&self) -> &[f64] { &self.params[OFF_U_REF1..OFF_U_REF2] }
    fn u_ref2(&self) -> &[f64] { &self.params[OFF_U_REF2..OFF_B1] }
    fn b1(&self) -> &[f64] { &self.params[OFF_B1..OFF_B2] }
    fn b2(&self) -> &[f64] { &self.params[OFF_B2..OFF_B_ABS] }
    fn b_abs(&self) -> &[f64] { &self.params[OFF_B_ABS..OFF_GATES] }
    fn gates(&self) -> &[f64] { &self.params[OFF_GATES..PARAM_COUNT] }

    /// Régénère H1 purement de haut en bas depuis H2 (utilisé par l'épreuve d'Abstraction S_abs).
    pub(crate) fn regenerate_h1_from_h2(&self, h2: &[[f64; DIM]], k1: usize) -> Vec<[f64; DIM]> {
        let k2 = h2.len();
        let mut h1_abs = vec![[0.0_f64; DIM]; k1];
        if k2 == 0 {
            return h1_abs;
        }
        let b_abs = self.b_abs();
        let mut msg2 = vec![[0.0_f64; DIM]; k2];
        for m in 0..k2 {
            msg2[m] = mat_vec(self.u_abs_21(), &h2[m]);
        }
        for k in 0..k1 {
            let mut acc = [0.0_f64; DIM];
            for m in 0..k2 {
                let w = self.w_down_21(k, m);
                if w != 0.0_f64 {
                    for d in 0..DIM {
                        acc[d] += w * msg2[m][d];
                    }
                }
            }
            for d in 0..DIM {
                h1_abs[k][d] = (acc[d] + b_abs[d]).tanh();
            }
            rms_norm(&mut h1_abs[k]);
        }
        h1_abs
    }

    /// Prépare les champs de circulation (h1_eff, flux v_k, portes u_gate(h1_eff), g_act)
    /// pour l'opérateur vertical T_{H1, H2}.
    fn prepare_operator(
        &self,
        state: &CircState,
        cut_l1: Option<u32>,
        cut_vertical: bool,
    ) -> (Vec<[f64; DIM]>, Vec<[f64; DIM]>, Vec<[f64; DIM]>, f64) {
        let k1 = state.h1.len();
        let k2 = state.h2.len();

        let n1_sq: f64 = state.h1.iter().map(|h| dot(h, h)).sum();
        let n2_sq: f64 = state.h2.iter().map(|h| dot(h, h)).sum();
        let n1 = n1_sq.sqrt();
        let n2 = n2_sq.sqrt();

        // Si budget = 0 ou si H1 ou H2 est absent/nul (hors coupe MIP verticale), g_act = 0
        let g_act = if cut_vertical {
            n1.tanh()
        } else {
            n1.tanh() * n2.tanh()
        };

        let mut h1_eff = state.h1.clone();
        let mut down21 = vec![[0.0_f64; DIM]; k2];
        let mut meta2 = vec![[0.0_f64; DIM]; k2];

        if !cut_vertical && k2 > 0 {
            for m in 0..k2 {
                down21[m] = mat_vec(self.u_down_21(), &state.h2[m]);
                meta2[m] = mat_vec(self.u_meta(), &state.h2[m]);
            }
            for k in 0..k1 {
                if vec_norm(&state.h1[k]) <= 1e-12_f64 {
                    continue;
                }
                let mut td = [0.0_f64; DIM];
                for m in 0..k2 {
                    let w = self.w_down_21(k, m);
                    if w != 0.0_f64 {
                        for d in 0..DIM {
                            td[d] += w * down21[m][d];
                        }
                    }
                }
                let mod_h = mat_vec(self.u_mod(), &state.h1[k]);
                for d in 0..DIM {
                    h1_eff[k][d] += td[d].tanh() * mod_h[d];
                }
                rms_norm(&mut h1_eff[k]);
            }
        }

        // Flux causal dirigé v_k dans L1 (+ gouvernance descendante L2 -> L1)
        let mut flow_msg = vec![[0.0_f64; DIM]; k1];
        for j in 0..k1 {
            flow_msg[j] = mat_vec(self.u_flow(), &h1_eff[j]);
        }

        let mut v_flow = vec![[0.0_f64; DIM]; k1];
        let mut gate_h = vec![[0.0_f64; DIM]; k1];

        for k in 0..k1 {
            let mut acc = [0.0_f64; DIM];
            for j in 0..k1 {
                if let Some(mask) = cut_l1 {
                    let sk = ((mask >> k.min(30)) & 1) != 0;
                    let sj = ((mask >> j.min(30)) & 1) != 0;
                    if sk != sj {
                        continue;
                    }
                }
                let w = self.w_lat1(k, j);
                if w != 0.0_f64 {
                    for d in 0..DIM {
                        acc[d] += w * flow_msg[j][d];
                    }
                }
            }
            if !cut_vertical {
                for m in 0..k2 {
                    let w = self.w_down_21(k, m);
                    if w != 0.0_f64 {
                        for d in 0..DIM {
                            acc[d] += w * meta2[m][d];
                        }
                    }
                }
            }
            v_flow[k] = mat_vec(self.u_down_10(), &acc);
            gate_h[k] = mat_vec(self.u_gate(), &h1_eff[k]);
        }

        (h1_eff, v_flow, gate_h, g_act)
    }

    /// Applique une fois l'opérateur de circulation T_{H1, H2}(z) dans R^d.
    fn apply_operator_once(
        &self,
        z: &[f64; DIM],
        h1_eff: &[[f64; DIM]],
        v_flow: &[[f64; DIM]],
        gate_h: &[[f64; DIM]],
        g_act: f64,
        attn_out: Option<&mut [f64]>,
    ) -> [f64; DIM] {
        let k1 = h1_eff.len();
        if k1 == 0 || g_act <= 1e-12_f64 {
            if let Some(a) = attn_out {
                let u = 1.0_f64 / (a.len().max(1) as f64);
                for x in a.iter_mut() {
                    *x = u;
                }
            }
            return [0.0_f64; DIM];
        }

        let inv_sqrt_d = 1.0_f64 / (DIM as f64).sqrt();
        let q = mat_vec(self.w_up_01(), z);
        let mut logits = [0.0_f64; 16];
        let mut weights = [0.0_f64; 16];
        let k_use = k1.min(16);

        for k in 0..k_use {
            if vec_norm(&h1_eff[k]) <= 1e-12_f64 {
                logits[k] = -1e9_f64;
            } else {
                logits[k] = dot(&q, &h1_eff[k]) * inv_sqrt_d;
            }
        }
        softmax_slice(&logits[..k_use], &mut weights[..k_use]);

        if let Some(a) = attn_out {
            for k in 0..k_use.min(a.len()) {
                a[k] = weights[k];
            }
        }

        let rot_z = mat_vec(self.u_rot(), z);
        let mut delta = [0.0_f64; DIM];
        for k in 0..k_use {
            let ak = weights[k];
            if ak <= 1e-12_f64 {
                continue;
            }
            for d in 0..DIM {
                delta[d] += ak * (v_flow[k][d] + rot_z[d] * gate_h[k][d]).tanh();
            }
        }

        let mut next_z = [0.0_f64; DIM];
        for d in 0..DIM {
            next_z[d] = z[d] + delta[d];
        }
        rms_norm(&mut next_z);
        for d in 0..DIM {
            next_z[d] *= g_act;
        }
        next_z
    }

    /// Déploiement vertical pur sans autorégression ni fast-weights :
    /// y^(1) = T(X), y^(2) = T(y^(1)), y^(3) = T(y^(2)), R_hat^(s) = softmax(<y^(s), W_read X> / tau).
    pub(crate) fn unfold_top_down(
        &self,
        state: &CircState,
        cut_l1: Option<u32>,
        cut_vertical: bool,
    ) -> CircUnfold {
        let v = state.x.len();
        let k1 = state.h1.len();

        let (h1_eff, v_flow, gate_h, g_act) = self.prepare_operator(state, cut_l1, cut_vertical);

        let mut a_ground = vec![0.0_f64; v * k1];
        let mut y1 = vec![[0.0_f64; DIM]; v];
        let mut y2 = vec![[0.0_f64; DIM]; v];
        let mut y3 = vec![[0.0_f64; DIM]; v];

        if v == 0 || k1 == 0 || g_act <= 1e-12_f64 {
            let unif = if v > 0 { 1.0_f64 / (v as f64) } else { 0.0_f64 };
            let r = vec![unif; v * v];
            return CircUnfold {
                h1_eff,
                a_ground,
                y: [y1, y2, y3],
                r_hat: [r.clone(), r.clone(), r],
            };
        }

        for u in 0..v {
            let row = &mut a_ground[u * k1..(u + 1) * k1];
            y1[u] = self.apply_operator_once(&state.x[u], &h1_eff, &v_flow, &gate_h, g_act, Some(row));
            y2[u] = self.apply_operator_once(&y1[u], &h1_eff, &v_flow, &gate_h, g_act, None);
            y3[u] = self.apply_operator_once(&y2[u], &h1_eff, &v_flow, &gate_h, g_act, None);
        }

        // Clés de lecture W_read * x_w
        let mut read_keys = vec![[0.0_f64; DIM]; v];
        for w in 0..v {
            read_keys[w] = mat_vec(self.w_read(), &state.x[w]);
        }

        let inv_tau = 0.60_f64 + 3.40_f64 * sigmoid(self.gates()[6]);
        let mut r_hat = [
            vec![0.0_f64; v * v],
            vec![0.0_f64; v * v],
            vec![0.0_f64; v * v],
        ];
        let mut logits = vec![0.0_f64; v];

        for (s, ys) in [&y1, &y2, &y3].into_iter().enumerate() {
            for u in 0..v {
                for w in 0..v {
                    logits[w] = inv_tau * dot(&ys[u], &read_keys[w]);
                }
                softmax_slice(&logits, &mut r_hat[s][u * v..(u + 1) * v]);
            }
        }

        CircUnfold {
            h1_eff,
            a_ground,
            y: [y1, y2, y3],
            r_hat,
        }
    }

    /// Une étape abductive pure de circulation L0 <-> L1 <-> L2.
    /// - `scale_mask`: masque certaines échelles de R_obs (ex: [true, false, true] pour P_scale).
    /// - `freeze_h`: verrouille (H1, H2) pour tester le ré-ancrage seul de X sous Déplacement.
    pub(crate) fn abductive_step(
        &self,
        view: &RelationalView,
        state: &CircState,
        use_refutation: bool,
        scale_mask: [bool; 3],
        freeze_h: bool,
        cut_l1: Option<u32>,
        cut_l2: Option<u32>,
        cut_vertical: bool,
    ) -> CircState {
        let v = view.v_size;
        let k1 = state.h1.len();
        let k2 = state.h2.len();

        let unfold = self.unfold_top_down(state, cut_l1, cut_vertical);
        let res_norm = multi_scale_l1(&view.graph, &unfold.r_hat, v);

        // Forces résiduelles duales (sortante r_u et entrante r_tilde_w) sur les échelles actives
        let mut r_out = vec![[0.0_f64; DIM]; v];
        let mut r_in = vec![[0.0_f64; DIM]; v];
        let mut mask_w_sum = 0.0_f64;
        for s in 0..3 {
            if scale_mask[s] {
                mask_w_sum += SCALE_W[s];
            }
        }
        let inv_mw = if mask_w_sum > 1e-9_f64 {
            1.0_f64 / mask_w_sum
        } else {
            1.0_f64
        };

        for s in 0..3 {
            if !scale_mask[s] {
                continue;
            }
            let sw = SCALE_W[s] * inv_mw;
            for u in 0..v {
                let wu = view.graph.w[s][u];
                if wu <= 0.0_f64 {
                    continue;
                }
                let base = u * v;
                let yu = &unfold.y[s][u];
                for w in 0..v {
                    let diff = view.graph.r[s][base + w] - unfold.r_hat[s][base + w];
                    if diff.abs() <= 1e-12_f64 {
                        continue;
                    }
                    let c_out = sw * diff;
                    let c_in = sw * wu * diff * (v as f64);
                    for d in 0..DIM {
                        r_out[u][d] += c_out * state.x[w][d];
                        r_in[w][d] += c_in * yu[d];
                    }
                }
            }
        }

        let g = self.gates();
        let alpha0 = 0.14_f64 + 0.50_f64 * sigmoid(g[0] * res_norm + g[1]);
        let alpha1 = 0.14_f64 + 0.50_f64 * sigmoid(g[2] * res_norm + g[3]);
        let alpha2 = 0.14_f64 + 0.50_f64 * sigmoid(g[4] * res_norm + g[5]);

        // 1. Réorganisation géométrique de L0 (ancrage relationnel sans stocker la loi)
        let mut next_x = vec![[0.0_f64; DIM]; v];
        for u in 0..v {
            let m_in = mat_vec(self.u_x_in(), &r_in[u]);
            let m_out = mat_vec(self.u_x_out(), &r_out[u]);
            let mut h_mix = [0.0_f64; DIM];
            for k in 0..k1 {
                let ak = unfold.a_ground[u * k1 + k];
                for d in 0..DIM {
                    h_mix[d] += ak * unfold.h1_eff[k][d];
                }
            }
            let m_td = mat_vec(self.u_x_td(), &h_mix);

            for d in 0..DIM {
                let drv = (m_in[d] + m_out[d] + 0.35_f64 * m_td[d]).tanh();
                next_x[u][d] = (1.0_f64 - alpha0) * state.x[u][d] + alpha0 * drv;
            }
            rms_norm(&mut next_x[u]);
        }

        if freeze_h {
            return CircState {
                x: next_x,
                h1: state.h1.clone(),
                h2: state.h2.clone(),
            };
        }

        // 2. Abduction à Niveau 1 (résidu agrégé + couplage latéral W_lat1 + L2 -> L1)
        let mut next_h1 = vec![[0.0_f64; DIM]; k1];
        let b1 = self.b1();

        let mut obs1_msg = vec![[0.0_f64; DIM]; v];
        for u in 0..v {
            obs1_msg[u] = mat_vec(self.u_obs_1(), &r_out[u]);
        }
        let mut lat1_msg = vec![[0.0_f64; DIM]; k1];
        for src in 0..k1 {
            lat1_msg[src] = mat_vec(self.u_lat1(), &state.h1[src]);
        }
        let mut down2_msg = vec![[0.0_f64; DIM]; k2];
        for m in 0..k2 {
            down2_msg[m] = mat_vec(self.u_down_21(), &state.h2[m]);
        }

        let mut e_up = vec![[0.0_f64; DIM]; k1];
        for dst in 0..k1 {
            let mut ek = [0.0_f64; DIM];
            for u in 0..v {
                let wu = view.graph.w[0][u];
                if wu <= 0.0_f64 {
                    continue;
                }
                let coeff = wu * (0.20_f64 + 0.80_f64 * unfold.a_ground[u * k1 + dst]);
                for d in 0..DIM {
                    ek[d] += coeff * obs1_msg[u][d];
                }
            }
            e_up[dst] = ek;

            let mut acc = ek;
            for src in 0..k1 {
                if let Some(mask) = cut_l1 {
                    let s_dst = ((mask >> dst.min(30)) & 1) != 0;
                    let s_src = ((mask >> src.min(30)) & 1) != 0;
                    if s_dst != s_src {
                        continue;
                    }
                }
                let w = self.w_lat1(dst, src);
                if w != 0.0_f64 {
                    for d in 0..DIM {
                        acc[d] += w * lat1_msg[src][d];
                    }
                }
            }

            if !cut_vertical {
                for m in 0..k2 {
                    let w = self.w_down_21(dst, m);
                    if w != 0.0_f64 {
                        for d in 0..DIM {
                            acc[d] += 0.65_f64 * w * down2_msg[m][d];
                        }
                    }
                }
            }

            for d in 0..DIM {
                next_h1[dst][d] =
                    (1.0_f64 - alpha1) * state.h1[dst][d] + alpha1 * (acc[d] + b1[d]).tanh();
            }
            rms_norm(&mut next_h1[dst]);
        }

        // 3. Réification pure L1 -> L2 (zéro accès direct à L0)
        let mut next_h2 = vec![[0.0_f64; DIM]; k2];
        let b2 = self.b2();
        let beta_dyn = 0.20_f64 + 0.80_f64 * sigmoid(g[7]);

        let mut e1_reified = vec![[0.0_f64; DIM]; k1];
        for k in 0..k1 {
            let mut combined = [0.0_f64; DIM];
            for d in 0..DIM {
                let dh = next_h1[k][d] - state.h1[k][d];
                combined[d] = next_h1[k][d] + beta_dyn * dh + 0.65_f64 * e_up[k][d];
            }
            e1_reified[k] = mat_vec(self.u_up_12(), &combined);
        }

        let mut lat2_msg = vec![[0.0_f64; DIM]; k2];
        for src in 0..k2 {
            lat2_msg[src] = mat_vec(self.u_lat2(), &state.h2[src]);
        }

        for dst in 0..k2 {
            let mut acc = [0.0_f64; DIM];

            if !cut_vertical {
                for src1 in 0..k1 {
                    let w = self.w_up_12(dst, src1);
                    if w != 0.0_f64 {
                        for d in 0..DIM {
                            acc[d] += w * e1_reified[src1][d];
                        }
                    }
                }
            }

            for src in 0..k2 {
                if let Some(mask) = cut_l2 {
                    let s_dst = ((mask >> dst.min(30)) & 1) != 0;
                    let s_src = ((mask >> src.min(30)) & 1) != 0;
                    if s_dst != s_src {
                        continue;
                    }
                }
                let w = self.w_lat2(dst, src);
                if w != 0.0_f64 {
                    for d in 0..DIM {
                        acc[d] += w * lat2_msg[src][d];
                    }
                }
            }

            for d in 0..DIM {
                next_h2[dst][d] =
                    (1.0_f64 - alpha2) * state.h2[dst][d] + alpha2 * (acc[d] + b2[d]).tanh();
            }
            rms_norm(&mut next_h2[dst]);
        }

        let mut next_state = CircState {
            x: next_x,
            h1: next_h1,
            h2: next_h2,
        };

        // 4. Réfutation endogène conservatrice (ne freine jamais une trajectoire déjà en progrès)
        if use_refutation
            && cut_l1.is_none()
            && cut_l2.is_none()
            && !cut_vertical
            && k1 > 0
            && k2 > 0
            && scale_mask == [true, true, true]
        {
            let gain_curr = multi_scale_gain(&view.graph, &unfold.r_hat, v);
            let unf_base = self.unfold_top_down(&next_state, None, false);
            let gain_base = multi_scale_gain(&view.graph, &unf_base.r_hat, v);

            // Teste activement une contre-hypothèse sur le nœud dominant L1 et L2
            let peak1 = next_state
                .h1
                .iter()
                .enumerate()
                .max_by(|a, b| dot(a.1, a.1).total_cmp(&dot(b.1, b.1)))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let peak2 = next_state
                .h2
                .iter()
                .enumerate()
                .max_by(|a, b| dot(a.1, a.1).total_cmp(&dot(b.1, b.1)))
                .map(|(i, _)| i)
                .unwrap_or(0);

            let mut refuted = next_state.clone();
            let p1 = mat_vec(self.u_ref1(), &refuted.h1[peak1]);
            for d in 0..DIM {
                refuted.h1[peak1][d] -= 0.35_f64 * p1[d].tanh();
            }
            rms_norm(&mut refuted.h1[peak1]);

            let p2 = mat_vec(self.u_ref2(), &refuted.h2[peak2]);
            for d in 0..DIM {
                refuted.h2[peak2][d] -= 0.35_f64 * p2[d].tanh();
            }
            rms_norm(&mut refuted.h2[peak2]);

            let unf_ref = self.unfold_top_down(&refuted, None, false);
            let gain_ref = multi_scale_gain(&view.graph, &unf_ref.r_hat, v);

            if gain_ref > gain_base + 1e-4_f64 {
                next_state = refuted;
            } else if gain_base >= gain_curr {
                // Consolide ce qui a survécu à la tentative de réfutation
                for d in 0..DIM {
                    let d1 = next_state.h1[peak1][d] - refuted.h1[peak1][d];
                    next_state.h1[peak1][d] += 0.10_f64 * d1;
                    let d2 = next_state.h2[peak2][d] - refuted.h2[peak2][d];
                    next_state.h2[peak2][d] += 0.10_f64 * d2;
                }
                rms_norm(&mut next_state.h1[peak1]);
                rms_norm(&mut next_state.h2[peak2]);
            }
        }

        next_state
    }

    pub(crate) fn run_abduction(
        &self,
        view: &RelationalView,
        budget: usize,
        use_refutation: bool,
        capture_steps: &[usize],
        record_phi: bool,
    ) -> AbductionTrace {
        self.run_abduction_masked(
            view,
            budget,
            use_refutation,
            [true, true, true],
            capture_steps,
            record_phi,
        )
    }

    pub(crate) fn run_abduction_masked(
        &self,
        view: &RelationalView,
        budget: usize,
        use_refutation: bool,
        scale_mask: [bool; 3],
        capture_steps: &[usize],
        record_phi: bool,
    ) -> AbductionTrace {
        let mut state = CircState::zeros(view.v_size, K1, K2);
        let mut snapshots = Vec::new();
        let mut phi_history = Vec::new();

        if capture_steps.contains(&0) {
            snapshots.push((0, state.clone()));
        }
        if record_phi {
            phi_history.push(
                self.evaluate_phi_state(view, &state, 0, use_refutation)
                    .phi_sharp,
            );
        }
        if budget == 0 {
            return AbductionTrace {
                state,
                snapshots,
                phi_history,
            };
        }

        state = CircState::seeded(view);
        for step in 1..=budget {
            state = self.abductive_step(
                view,
                &state,
                use_refutation,
                scale_mask,
                false,
                None,
                None,
                false,
            );
            if capture_steps.contains(&step) {
                snapshots.push((step, state.clone()));
            }
            if record_phi {
                phi_history.push(
                    self.evaluate_phi_state(view, &state, step, use_refutation)
                        .phi_sharp,
                );
            }
        }

        AbductionTrace {
            state,
            snapshots,
            phi_history,
        }
    }

    pub(crate) fn evaluate_phi(
        &self,
        view: &RelationalView,
        budget: usize,
        use_refutation: bool,
    ) -> PhiReport {
        let trace = self.run_abduction(view, budget, use_refutation, &[], false);
        self.evaluate_phi_state(view, &trace.state, budget, use_refutation)
    }

    /// φ_sharp V3 :
    /// - Conjonction sans compensation : min(G, D, P) avec P = min(P_temp, P_scale)
    /// - Φ_cut = Φ_mip(coupes de canaux seules) * Φ_diff(différenciation, sans punir DropL1)
    /// - S_abd = (S_ref * S_disp * S_abs)^(1/3) : réparation sous attaque + déplacement + abstraction H2->H1
    /// - Terme de résidu + verrou anti-Goodhart strict
    pub(crate) fn evaluate_phi_state(
        &self,
        view: &RelationalView,
        state: &CircState,
        budget: usize,
        use_refutation: bool,
    ) -> PhiReport {
        let v = view.v_size;
        if v <= 1 || view.seq.len() < 16 {
            return PhiReport {
                generativity: 0.0,
                discrimination: 0.0,
                projection: 0.0,
                proj_temp: 0.0,
                proj_scale: 0.0,
                phi_cut: 0.0,
                s_ref: 0.0,
                s_disp: 0.0,
                s_abs: 0.0,
                residual_norm: 1.0,
                phi_sharp: 0.0,
                shaping_score: -1.0,
            };
        }

        let unfold = self.unfold_top_down(state, None, false);

        // 1. Générativité et Discrimination au-dessus des marginales P_marg
        let gen_raw = multi_scale_gain(&view.graph, &unfold.r_hat, v);
        let neg_gain = multi_scale_gain(&view.shuf_graph, &unfold.r_hat, v);
        let disc_raw = (gen_raw - neg_gain).clamp(-1.0_f64, 1.0_f64);

        // 2. Projection immunisée contre l'écho : min(P_temp, P_scale)
        let (proj_temp_raw, proj_scale_raw) = if budget == 0 || state.h1.is_empty() || state.h2.is_empty() {
            (0.0_f64, 0.0_f64)
        } else {
            let mid = view.seq.len() / 2;
            let first_half = view.subslice_same_vocab(0, mid);
            let second_half = view.subslice_same_vocab(mid, view.seq.len());
            let tr_half = self.run_abduction(&first_half, budget, use_refutation, &[], false);
            let unf_half = self.unfold_top_down(&tr_half.state, None, false);
            let pt = multi_scale_gain(&second_half.graph, &unf_half.r_hat, v);

            // Projection inter-échelles : abduit en masquant lag-2, déplie T^2 sur R^(2)
            let tr_sc = self.run_abduction_masked(
                view,
                budget,
                false,
                [true, false, true],
                &[],
                false,
            );
            let unf_sc = self.unfold_top_down(&tr_sc.state, None, false);
            let ps = scale_gain(
                &view.graph.r[1],
                &unf_sc.r_hat[1],
                &view.graph.w[1],
                view.graph.h_marg[1],
                v,
            );
            (pt, ps)
        };
        let proj_raw = proj_temp_raw.min(proj_scale_raw);

        // 3. Irréductibilité de circulation Φ_cut = Φ_mip * Φ_diff (ZÉRO ablation de nœud)
        let k1 = state.h1.len();
        let k2 = state.h2.len();
        let phi_cut = if budget == 0 || k1 <= 1 || k2 <= 1 {
            0.0_f64
        } else {
            let mut min_n1 = f64::INFINITY;
            for h in &state.h1 {
                min_n1 = min_n1.min(vec_norm(h));
            }
            let mut min_n2 = f64::INFINITY;
            for h in &state.h2 {
                min_n2 = min_n2.min(vec_norm(h));
            }
            let mut min_dist = f64::INFINITY;
            for a in 0..k1 {
                for b in (a + 1)..k1 {
                    min_dist = min_dist.min(vec_dist(&state.h1[a], &state.h1[b]));
                }
            }
            for a in 0..k2 {
                for b in (a + 1)..k2 {
                    min_dist = min_dist.min(vec_dist(&state.h2[a], &state.h2[b]));
                }
            }
            let phi_diff = (min_n1 * 3.0_f64).tanh()
                * (min_n2 * 3.0_f64).tanh()
                * (min_dist * 3.0_f64).tanh();

            if phi_diff <= 1e-9_f64 {
                0.0_f64
            } else {
                let full_next = self.abductive_step(
                    view,
                    state,
                    false,
                    [true, true, true],
                    false,
                    None,
                    None,
                    false,
                );
                let pred_full = self.unfold_top_down(&full_next, None, false);

                let mut min_div = f64::INFINITY;

                // Coupes de canaux internes L1
                let masks1 = 1u32 << (k1.min(10) - 1);
                for mask in 1..masks1 {
                    let cut_next = self.abductive_step(
                        view,
                        state,
                        false,
                        [true, true, true],
                        false,
                        Some(mask),
                        None,
                        false,
                    );
                    let pred_cut = self.unfold_top_down(&cut_next, Some(mask), false);
                    let div =
                        multi_scale_pred_tvd(&pred_full.r_hat, &pred_cut.r_hat, &view.graph.w, v);
                    if div < min_div {
                        min_div = div;
                    }
                }

                // Coupes de canaux internes L2
                let masks2 = 1u32 << (k2.min(10) - 1);
                for mask in 1..masks2 {
                    let cut_next = self.abductive_step(
                        view,
                        state,
                        false,
                        [true, true, true],
                        false,
                        None,
                        Some(mask),
                        false,
                    );
                    let pred_cut = self.unfold_top_down(&cut_next, None, false);
                    let div =
                        multi_scale_pred_tvd(&pred_full.r_hat, &pred_cut.r_hat, &view.graph.w, v);
                    if div < min_div {
                        min_div = div;
                    }
                }

                // Coupe verticale L1 <-> L2
                let vcut_next = self.abductive_step(
                    view,
                    state,
                    false,
                    [true, true, true],
                    false,
                    None,
                    None,
                    true,
                );
                let pred_vcut = self.unfold_top_down(&vcut_next, None, true);
                let vdiv =
                    multi_scale_pred_tvd(&pred_full.r_hat, &pred_vcut.r_hat, &view.graph.w, v);
                if vdiv < min_div {
                    min_div = vdiv;
                }

                ((min_div * 10.0_f64).tanh().clamp(0.0_f64, 1.0_f64)) * phi_diff
            }
        };

        // 4. Survie sous les 3 épreuves abductives : Réfutation, Déplacement, Abstraction
        let (s_ref, s_disp, s_abs) = if budget == 0 || gen_raw <= 1e-6_f64 || k1 == 0 || k2 == 0 {
            (0.0_f64, 0.0_f64, 0.0_f64)
        } else {
            // (a) S_ref : attaque du nœud L1 dominant puis 2 pas de réparation active
            let mut attacked = state.clone();
            let peak1 = attacked
                .h1
                .iter()
                .enumerate()
                .max_by(|a, b| dot(a.1, a.1).total_cmp(&dot(b.1, b.1)))
                .map(|(i, _)| i)
                .unwrap_or(0);
            attacked.h1[peak1] = [0.0_f64; DIM];
            for _ in 0..2 {
                attacked = self.abductive_step(
                    view,
                    &attacked,
                    false,
                    [true, true, true],
                    false,
                    None,
                    None,
                    false,
                );
            }
            let unf_rep = self.unfold_top_down(&attacked, None, false);
            let g_rep = multi_scale_gain(&view.graph, &unf_rep.r_hat, v);
            let sr = (g_rep / gen_raw).clamp(0.0_f64, 1.0_f64);

            // (b) S_disp : (H1*, H2*) verrouillés, ré-ancrage seul de X sur fenêtre déplacée
            let q1 = view.seq.len() / 4;
            let q3 = (3 * view.seq.len()) / 4;
            let shifted = view.subslice_same_vocab(q1, q3);
            let mut disp_st = state.clone();
            for _ in 0..2 {
                disp_st = self.abductive_step(
                    &shifted,
                    &disp_st,
                    false,
                    [true, true, true],
                    true,
                    None,
                    None,
                    false,
                );
            }
            let unf_disp = self.unfold_top_down(&disp_st, None, false);
            let g_disp = multi_scale_gain(&shifted.graph, &unf_disp.r_hat, v);
            let sd = (g_disp / gen_raw).clamp(0.0_f64, 1.0_f64);

            // (c) S_abs : efface H1* et force la régénération pure H2* -> H1_abs (verrou anti-gel)
            let h1_abs = self.regenerate_h1_from_h2(&state.h2, k1);
            let abs_st = CircState {
                x: state.x.clone(),
                h1: h1_abs,
                h2: state.h2.clone(),
            };
            let unf_abs = self.unfold_top_down(&abs_st, None, false);
            let g_abs = multi_scale_gain(&view.graph, &unf_abs.r_hat, v);
            let sa = (g_abs / gen_raw).clamp(0.0_f64, 1.0_f64);

            (sr, sd, sa)
        };

        let s_abd = (s_ref * s_disp * s_abs).cbrt();
        let residual_norm = multi_scale_l1(&view.graph, &unfold.r_hat, v).clamp(0.0_f64, 1.0_f64);

        let g_c = gen_raw.clamp(0.0_f64, 1.0_f64);
        let d_c = disc_raw.clamp(0.0_f64, 1.0_f64);
        let p_c = proj_raw.clamp(0.0_f64, 1.0_f64);

        let base_min = g_c.min(d_c).min(p_c);
        let phi_sharp = base_min * phi_cut * s_abd * (1.0_f64 - 0.5_f64 * residual_norm);

        // Verrou anti-Goodhart strict : zéro récompense sur Φ_cut ou S_abd tant que G, D, P <= 0
        let shaping_score = if gen_raw > 0.0_f64 && disc_raw > 0.0_f64 && proj_raw > 0.0_f64 {
            phi_sharp + 0.05_f64 * base_min
        } else {
            let min_raw = gen_raw.min(disc_raw).min(proj_raw);
            0.55_f64 * min_raw
                + 0.25_f64 * (gen_raw + disc_raw + proj_raw) / 3.0_f64
                - 0.10_f64 * residual_norm
        };

        PhiReport {
            generativity: g_c,
            discrimination: d_c,
            projection: p_c,
            proj_temp: proj_temp_raw.clamp(0.0_f64, 1.0_f64),
            proj_scale: proj_scale_raw.clamp(0.0_f64, 1.0_f64),
            phi_cut,
            s_ref,
            s_disp,
            s_abs,
            residual_norm,
            phi_sharp,
            shaping_score,
        }
    }

    /// Canal `ask` pur : itération verticale y_{t+1} = T_{H1*, H2*}(y_t) dans R^d,
    /// sélectionnée par clôture de résidu auto-référentielle max [φ_sharp(E ⊕ ΔE) - φ_sharp(E)].
    pub(crate) fn ask_emit(&self, view: &RelationalView, budget: usize, out_len: usize) -> Vec<u8> {
        if view.vocab.is_empty() || view.seq.is_empty() {
            return Vec::new();
        }
        let v = view.v_size;
        let trace = self.run_abduction(view, budget, true, &[], false);
        let unfold = self.unfold_top_down(&trace.state, None, false);
        let phi_base = self
            .evaluate_phi_state(view, &trace.state, budget, true)
            .phi_sharp;

        let (h1_eff, v_flow, gate_h, g_act) = self.prepare_operator(&trace.state, None, false);
        let mut read_keys = vec![[0.0_f64; DIM]; v];
        for w in 0..v {
            read_keys[w] = mat_vec(self.w_read(), &trace.state.x[w]);
        }
        let inv_tau = 0.60_f64 + 3.40_f64 * sigmoid(self.gates()[6]);

        let mut node_res: Vec<(usize, f64)> = (0..v)
            .filter(|&u| view.graph.w[0][u] > 0.0_f64)
            .map(|u| {
                let mut err = 0.0_f64;
                for s in 0..3 {
                    let mut l1 = 0.0_f64;
                    for w in 0..v {
                        l1 += (view.graph.r[s][u * v + w] - unfold.r_hat[s][u * v + w]).abs();
                    }
                    err += SCALE_W[s] * 0.5_f64 * l1;
                }
                (u, err)
            })
            .collect();
        node_res.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut e_4grams: HashSet<[usize; 4]> = HashSet::new();
        for w in view.seq.windows(4) {
            e_4grams.insert([w[0], w[1], w[2], w[3]]);
        }

        let n_cands = node_res.len().min(4).max(1);
        let mut best_cand = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut logits = vec![0.0_f64; v];
        let mut probs = vec![0.0_f64; v];

        for c_i in 0..n_cands {
            let anchor = if c_i < node_res.len() {
                node_res[c_i].0
            } else {
                0
            };

            let mut z = trace.state.x[anchor];
            let mut cand_idx = Vec::with_capacity(out_len);
            cand_idx.push(anchor);
            let mut visits = vec![0usize; v];
            visits[anchor] = 1;

            for _ in 1..out_len {
                z = self.apply_operator_once(&z, &h1_eff, &v_flow, &gate_h, g_act, None);
                for w in 0..v {
                    logits[w] = inv_tau * dot(&z, &read_keys[w]);
                }
                softmax_slice(&logits, &mut probs);

                let mut best_w = 0usize;
                let mut best_p = f64::NEG_INFINITY;
                for w in 0..v {
                    let p = probs[w] / (1.0_f64 + 0.35_f64 * (visits[w] as f64));
                    if p > best_p {
                        best_p = p;
                        best_w = w;
                    }
                }
                visits[best_w] += 1;
                cand_idx.push(best_w);
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

/// Baseline MDL adaptée à l'état V3 (pénalise la longueur de description de X, H1, H2).
pub(crate) fn mdl_graph_v3(
    graph: &ScaleGraph,
    r_hat: &[Vec<f64>; 3],
    state: &CircState,
    seq_len: usize,
    v: usize,
) -> f64 {
    let ll = log_likelihood_graph(graph, r_hat, v);
    let mut lat_energy = 0.0_f64;
    for h in state.h1.iter().chain(state.h2.iter()) {
        for &val in h {
            if val.abs() > 0.05_f64 {
                lat_energy += val.abs();
            }
        }
    }
    let mut x_energy = 0.0_f64;
    for x in &state.x {
        for &val in x {
            if val.abs() > 0.05_f64 {
                x_energy += val.abs();
            }
        }
    }
    let n = seq_len.max(2) as f64;
    ll - 0.008_f64 * (lat_energy + 0.35_f64 * x_energy) * n.ln() / n
}
