use crate::core::flow::{run_abductive_flow, FlowConfig};
use crate::core::phi_sharp::{
    evaluate_phi_sharp, residual_weights, smooth_cap, smooth_hinge, EffectSummary, ModelParams,
    PhiSharpConfig, PrecomputedHeads, StabilityProbes, BI_DIM, BYTE_VOCAB, REL_DIM,
};
use crate::core::projection::{project_witness_mirror_ascent, ProjectionConfig};
use crate::core::spectral::CausalLaplacian;
use crate::core::tensor::{Mat, RelTensor};
use crate::ho_ptgrs::generator::{CurriculumConfig, HoPtgrsGenerator};
use crate::ho_ptgrs::rng::Rng;

/// Configuration d'une séance unique de méta-entraînement sur HO-PTGRS.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub episodes: usize,
    pub seed: u64,
    pub lr: f64,
    pub weight_decay: f64,
    pub grad_clip_norm: f64,
    /// Écart-type de la perturbation antithétique pour les paramètres de trajectoire (\Psi_0, W_sol).
    pub es_sigma: f64,
    /// Fréquence (en épisodes) de la mise à jour antithétique (0 = désactivé).
    pub es_every: usize,
    /// Sondes de stabilité différentielle pour V_stab.
    pub num_rot_probes: usize,
    pub probe_angle: f64,
    pub probe_abs_kappa: f64,
    /// Curriculum : bornes de début et de fin pour (horizon_k, distractors, noise_ratio).
    pub start_horizon_k: usize,
    pub end_horizon_k: usize,
    pub start_distractors: usize,
    pub end_distractors: usize,
    pub start_noise: f64,
    pub end_noise: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            episodes: 30,
            seed: 42,
            lr: 0.02,
            weight_decay: 1e-4,
            grad_clip_norm: 5.0,
            es_sigma: 0.02,
            es_every: 2,
            num_rot_probes: 2,
            probe_angle: 0.06,
            probe_abs_kappa: 0.12,
            start_horizon_k: 3,
            end_horizon_k: 5,
            start_distractors: 2,
            end_distractors: 5,
            start_noise: 0.15,
            end_noise: 0.75,
        }
    }
}

/// Journal d'un épisode d'entraînement (métriques purement intrinsèques à \phi_\sharp).
#[derive(Debug, Clone)]
pub struct EpisodeTrainLog {
    pub episode: usize,
    pub horizon_k: usize,
    pub distractors: usize,
    pub eff_bytes: usize,
    pub phi_init: f64,
    pub phi_final: f64,
    pub lambda2_final: f64,
    pub residual_init: f64,
    pub residual_final: f64,
    pub v_stab_final: f64,
    pub active_edges: usize,
    pub flow_steps: usize,
    pub grad_norm: f64,
}

/// Bilan d'évaluation post-hoc (la cause cachée n'est lue qu'ICI, jamais pendant le training).
#[derive(Debug, Clone)]
pub struct PostHocEvalSummary {
    pub episodes_tested: usize,
    pub mean_phi_gain: f64,
    pub mean_lambda2: f64,
    pub mean_residual_drop: f64,
    pub mean_witness_alphabet_ratio: f64,
    pub mean_target_byte_overlap: f64,
}

/// Tampon de gradients aligné 1-pour-1 sur tous les paramètres S_256-équivariants de `ModelParams`.
#[derive(Debug, Clone)]
pub struct ParamGrads {
    pub w_dir: Vec<f64>,
    pub w_comp: Vec<f64>,
    pub b_edge: Vec<f64>,
    pub w_rel_uni: Vec<f64>,
    pub b_rel_uni: Vec<f64>,
    pub w_rel_bi: Vec<f64>,
    pub b_rel_bi: Vec<f64>,
    pub w_enc_uni: Vec<f64>,
    pub w_enc_bi: Vec<f64>,
    pub w_init_asym: Vec<f64>,
    pub w_init_ov: Vec<f64>,
    pub w_sol: Mat,
    pub nu_sol: Vec<f64>,
}

impl ParamGrads {
    pub fn zeros_like(p: &ModelParams) -> Self {
        Self {
            w_dir: vec![0.0; p.w_dir.len()],
            w_comp: vec![0.0; p.w_comp.len()],
            b_edge: vec![0.0; p.b_edge.len()],
            w_rel_uni: vec![0.0; p.w_rel_uni.len()],
            b_rel_uni: vec![0.0; p.b_rel_uni.len()],
            w_rel_bi: vec![0.0; p.w_rel_bi.len()],
            b_rel_bi: vec![0.0; p.b_rel_bi.len()],
            w_enc_uni: vec![0.0; p.w_enc_uni.len()],
            w_enc_bi: vec![0.0; p.w_enc_bi.len()],
            w_init_asym: vec![0.0; p.w_init_asym.len()],
            w_init_ov: vec![0.0; p.w_init_ov.len()],
            w_sol: Mat::zeros(p.n),
            nu_sol: vec![0.0; p.nu_sol.len()],
        }
    }

    pub fn l2_norm(&self) -> f64 {
        let mut s = 0.0;
        for slice in [
            self.w_dir.as_slice(),
            self.w_comp.as_slice(),
            self.b_edge.as_slice(),
            self.w_rel_uni.as_slice(),
            self.b_rel_uni.as_slice(),
            self.w_rel_bi.as_slice(),
            self.b_rel_bi.as_slice(),
            self.w_enc_uni.as_slice(),
            self.w_enc_bi.as_slice(),
            self.w_init_asym.as_slice(),
            self.w_init_ov.as_slice(),
            self.w_sol.data.as_slice(),
            self.nu_sol.as_slice(),
        ] {
            for &v in slice {
                s += v * v;
            }
        }
        s.sqrt()
    }

    pub fn clip_norm(&mut self, max_norm: f64) -> f64 {
        let norm = self.l2_norm();
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for slice in [
                self.w_dir.as_mut_slice(),
                self.w_comp.as_mut_slice(),
                self.b_edge.as_mut_slice(),
                self.w_rel_uni.as_mut_slice(),
                self.b_rel_uni.as_mut_slice(),
                self.w_rel_bi.as_mut_slice(),
                self.b_rel_bi.as_mut_slice(),
                self.w_enc_uni.as_mut_slice(),
                self.w_enc_bi.as_mut_slice(),
                self.w_init_asym.as_mut_slice(),
                self.w_init_ov.as_mut_slice(),
                self.w_sol.data.as_mut_slice(),
                self.nu_sol.as_mut_slice(),
            ] {
                for v in slice {
                    *v *= scale;
                }
            }
        }
        norm
    }
}

/// État interne de l'optimiseur AdamW en ascension sur \phi_\sharp.
struct AdamWState {
    step: usize,
    m: ParamGrads,
    v: ParamGrads,
}

impl AdamWState {
    fn new(p: &ModelParams) -> Self {
        Self {
            step: 0,
            m: ParamGrads::zeros_like(p),
            v: ParamGrads::zeros_like(p),
        }
    }

    fn step_ascent(
        &mut self,
        params: &mut ModelParams,
        grads: &ParamGrads,
        lr: f64,
        weight_decay: f64,
    ) {
        self.step += 1;
        let b1 = 0.9;
        let b2 = 0.999;
        let eps = 1e-8;
        let bc1 = 1.0 - b1.powi(self.step as i32);
        let bc2 = 1.0 - b2.powi(self.step as i32);

        let update = |p: &mut [f64], g: &[f64], m: &mut [f64], v: &mut [f64], wd: f64| {
            for i in 0..p.len() {
                let gi = g[i];
                m[i] = b1 * m[i] + (1.0 - b1) * gi;
                v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
                let m_hat = m[i] / bc1;
                let v_hat = v[i] / bc2;
                p[i] = p[i] * (1.0 - lr * wd) + lr * m_hat / (v_hat.sqrt() + eps);
            }
        };

        update(&mut params.w_dir, &grads.w_dir, &mut self.m.w_dir, &mut self.v.w_dir, weight_decay);
        update(&mut params.w_comp, &grads.w_comp, &mut self.m.w_comp, &mut self.v.w_comp, weight_decay);
        update(&mut params.b_edge, &grads.b_edge, &mut self.m.b_edge, &mut self.v.b_edge, 0.0);
        update(&mut params.w_rel_uni, &grads.w_rel_uni, &mut self.m.w_rel_uni, &mut self.v.w_rel_uni, weight_decay);
        update(&mut params.b_rel_uni, &grads.b_rel_uni, &mut self.m.b_rel_uni, &mut self.v.b_rel_uni, 0.0);
        update(&mut params.w_rel_bi, &grads.w_rel_bi, &mut self.m.w_rel_bi, &mut self.v.w_rel_bi, weight_decay);
        update(&mut params.b_rel_bi, &grads.b_rel_bi, &mut self.m.b_rel_bi, &mut self.v.b_rel_bi, 0.0);
        update(&mut params.w_enc_uni, &grads.w_enc_uni, &mut self.m.w_enc_uni, &mut self.v.w_enc_uni, weight_decay);
        update(&mut params.w_enc_bi, &grads.w_enc_bi, &mut self.m.w_enc_bi, &mut self.v.w_enc_bi, weight_decay);
        update(&mut params.w_init_asym, &grads.w_init_asym, &mut self.m.w_init_asym, &mut self.v.w_init_asym, weight_decay);
        update(&mut params.w_init_ov, &grads.w_init_ov, &mut self.m.w_init_ov, &mut self.v.w_init_ov, weight_decay);
        update(&mut params.w_sol.data, &grads.w_sol.data, &mut self.m.w_sol.data, &mut self.v.w_sol.data, weight_decay);
        update(&mut params.nu_sol, &grads.nu_sol, &mut self.m.nu_sol, &mut self.v.nu_sol, weight_decay);
    }
}

/// Accumule `scale * \nabla_\theta \rho_KL(R, E)` dans `grads` (rétropropagation exacte sur W_rel).
fn accumulate_residual_grad_theta(
    params: &ModelParams,
    r: &RelTensor,
    effect: &EffectSummary,
    heads: &PrecomputedHeads,
    lambda_bi: f64,
    scale: f64,
    grads: &mut ParamGrads,
) {
    if scale == 0.0 || effect.k_uni == 0 {
        return;
    }
    let (w_uni, w_bi) = residual_weights(effect.has_bigrams, lambda_bi);

    let n = params.n;
    let d = params.d;
    let h = params.h;
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();

    let comp: Vec<Mat> = (0..d)
        .map(|c| r.channels[c].matmul(&r.channels[c]))
        .collect();

    let mut m_edge = vec![0.0; n * n * h];
    let mut h_nodes = vec![0.0; n * h];

    for i in 0..n {
        for j in 0..n {
            let base = (i * n + j) * h;
            for r_idx in 0..h {
                let mut z = params.b_edge[r_idx];
                let w_d = &params.w_dir[r_idx * d..(r_idx + 1) * d];
                let w_c = &params.w_comp[r_idx * d..(r_idx + 1) * d];
                for c in 0..d {
                    z += w_d[c] * r.get(i, j, c) + w_c[c] * comp[c].get(i, j);
                }
                let m = z.tanh();
                m_edge[base + r_idx] = m;
                h_nodes[i * h + r_idx] += m;
            }
        }
        for r_idx in 0..h {
            h_nodes[i * h + r_idx] *= inv_sqrt_n;
        }
    }

    let mut grad_h = vec![0.0; n * h];
    let mut probs = vec![0.0; heads.k_uni.max(heads.k_bi).max(1)];

    for i in 0..n {
        let hi = &h_nodes[i * h..(i + 1) * h];
        let ghi = &mut grad_h[i * h..(i + 1) * h];

        if effect.uni_mass[i] > 0.0 {
            bilinear_head_grad_theta(
                hi,
                &heads.u_uni,
                &heads.u0_uni,
                &effect.rel_feats,
                REL_DIM,
                &effect.uni[i * heads.k_uni..(i + 1) * heads.k_uni],
                effect.uni_mass[i],
                scale * w_uni,
                &mut probs[..heads.k_uni],
                ghi,
                &mut grads.w_rel_uni,
                &mut grads.b_rel_uni,
            );
        }
        if w_bi > 0.0 && heads.k_bi > 0 && effect.bi_mass[i] > 0.0 {
            bilinear_head_grad_theta(
                hi,
                &heads.u_bi,
                &heads.u0_bi,
                &effect.bi_feats,
                BI_DIM,
                &effect.bi[i * heads.k_bi..(i + 1) * heads.k_bi],
                effect.bi_mass[i],
                scale * w_bi,
                &mut probs[..heads.k_bi],
                ghi,
                &mut grads.w_rel_bi,
                &mut grads.b_rel_bi,
            );
        }
    }

    for i in 0..n {
        let ghi = &grad_h[i * h..(i + 1) * h];
        for j in 0..n {
            let base = (i * n + j) * h;
            for r_idx in 0..h {
                let m = m_edge[base + r_idx];
                let dz = inv_sqrt_n * ghi[r_idx] * (1.0 - m * m);
                if dz == 0.0 {
                    continue;
                }
                grads.b_edge[r_idx] += dz;
                for c in 0..d {
                    grads.w_dir[r_idx * d + c] += dz * r.get(i, j, c);
                    grads.w_comp[r_idx * d + c] += dz * comp[c].get(i, j);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn bilinear_head_grad_theta(
    h_node: &[f64],
    u_mat: &[f64],
    u0: &[f64],
    feats: &[f64],
    feat_dim: usize,
    target: &[f64],
    mass: f64,
    weight: f64,
    probs: &mut [f64],
    grad_h: &mut [f64],
    grad_w_rel: &mut [f64],
    grad_b_rel: &mut [f64],
) {
    let k = u0.len();
    let h = h_node.len();
    let mut max_l = f64::NEG_INFINITY;

    for a in 0..k {
        let row = &u_mat[a * h..(a + 1) * h];
        let mut acc = u0[a];
        for r in 0..h {
            acc += row[r] * h_node[r];
        }
        probs[a] = acc;
        if acc > max_l {
            max_l = acc;
        }
    }

    let mut sum_exp = 0.0;
    for v in probs.iter_mut().take(k) {
        *v = (*v - max_l).exp();
        sum_exp += *v;
    }
    let inv_sum = 1.0 / sum_exp;

    // Factorisation exacte : g_feat[m] = sum_a dl_a * feat[a, m]
    let mut g_feat = vec![0.0; feat_dim];
    for a in 0..k {
        let p_a = probs[a] * inv_sum;
        let dl = weight * (mass * p_a - target[a]);
        if dl == 0.0 {
            continue;
        }
        let u_row = &u_mat[a * h..(a + 1) * h];
        for r in 0..h {
            grad_h[r] += dl * u_row[r];
        }
        let f_row = &feats[a * feat_dim..(a + 1) * feat_dim];
        for m in 0..feat_dim {
            g_feat[m] += dl * f_row[m];
        }
    }

    for m in 0..feat_dim {
        let gm = g_feat[m];
        if gm == 0.0 {
            continue;
        }
        grad_b_rel[m] += gm;
        for r in 0..h {
            grad_w_rel[r * feat_dim + m] += h_node[r] * gm;
        }
    }
}

/// Calcule le gradient analytique EXACT \partial \phi_\sharp(R, E) / \partial \theta à R fixé.
pub fn grad_theta_phi_sharp(
    params: &ModelParams,
    r: &RelTensor,
    effect: &EffectSummary,
    probes: &StabilityProbes,
    cfg: &PhiSharpConfig,
) -> (f64, ParamGrads) {
    let (eval, _) = evaluate_phi_sharp(params, r, effect, probes, cfg, false);
    let mut grads = ParamGrads::zeros_like(params);
    let heads = params.precompute_heads(effect);

    let n = r.n;
    let d = r.d;
    let rho = eval.residual;
    let rho_bg = eval.rho_background;
    let p_bar = (-rho).exp();
    let g_w = CausalLaplacian::grad_directed_weights(&eval.fiedler.v2);

    let mut coeff_on_rho = -cfg.alpha;

    // 1. Contribution de \beta_iit * \lambda_2(L_{\bar{w}}) à travers toutes les ablations actives
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let gw = cfg.beta_iit * g_w.get(i, j);
            if gw <= 0.0 || (0..d).all(|c| r.get(i, j, c) == 0.0) {
                continue;
            }
            let r_abl = r.with_ablated_edge(i, j);
            let rho_ij = params.residual_only_precomputed(&r_abl, effect, &heads, cfg.lambda_bi);
            let (s_ij, ds_drho_ij) = smooth_cap(rho_ij, rho_bg, cfg.cap_eps);
            let (h_val, h_prime) = smooth_hinge(s_ij - rho, cfg.hinge_tau);
            if h_val > 0.0 {
                let dw_drho = -p_bar * (h_val + h_prime);
                let dw_drho_ij = p_bar * h_prime * ds_drho_ij;
                coeff_on_rho += gw * dw_drho;
                accumulate_residual_grad_theta(
                    params,
                    &r_abl,
                    effect,
                    &heads,
                    cfg.lambda_bi,
                    gw * dw_drho_ij,
                    &mut grads,
                );
            }
        }
    }

    // 2. Contribution de -\beta * V_stab(R, E) sur les sondes de déplacement & abstraction
    let total_probes = probes.probe_count();
    if total_probes > 0 && cfg.beta > 0.0 {
        let inv_m = 1.0 / (total_probes as f64);

        for q in &probes.rotations {
            let r_rot = StabilityProbes::apply_rotation(q, r);
            let rho_rot =
                params.residual_only_precomputed(&r_rot, effect, &heads, cfg.lambda_bi);
            let (_, hp) = smooth_hinge(rho_rot - rho, cfg.hinge_tau);
            if hp > 0.0 {
                let scale = -cfg.beta * inv_m * hp;
                coeff_on_rho -= scale;
                accumulate_residual_grad_theta(
                    params,
                    &r_rot,
                    effect,
                    &heads,
                    cfg.lambda_bi,
                    scale,
                    &mut grads,
                );
            }
        }

        if probes.abs_kappa > 0.0 {
            let r_abs = probes.apply_abstraction(r);
            let rho_abs =
                params.residual_only_precomputed(&r_abs, effect, &heads, cfg.lambda_bi);
            let (_, hp) = smooth_hinge(rho_abs - rho, cfg.hinge_tau);
            if hp > 0.0 {
                let scale = -cfg.beta * inv_m * hp;
                coeff_on_rho -= scale;
                accumulate_residual_grad_theta(
                    params,
                    &r_abs,
                    effect,
                    &heads,
                    cfg.lambda_bi,
                    scale,
                    &mut grads,
                );
            }
        }
    }

    // 3. Terme principal \nabla_\theta \rho_KL(R, E)
    accumulate_residual_grad_theta(
        params,
        r,
        effect,
        &heads,
        cfg.lambda_bi,
        coeff_on_rho,
        &mut grads,
    );

    (eval.phi_sharp, grads)
}

/// Estimation antithétique de \nabla_{\Psi_0, W_{sol}} \phi_\sharp(R_\theta(T; E), E)
/// pilotée exclusivement par le scalaire \phi_\sharp.
fn accumulate_trajectory_es_grad(
    params: &ModelParams,
    effect: &EffectSummary,
    probes: &StabilityProbes,
    phi_cfg: &PhiSharpConfig,
    flow_cfg: &FlowConfig,
    sigma: f64,
    rng: &mut Rng,
    grads: &mut ParamGrads,
) {
    if sigma <= 0.0 {
        return;
    }

    let sample_dir = |len: usize, rng: &mut Rng| -> Vec<f64> {
        (0..len)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
            .collect()
    };

    let d_enc_u = sample_dir(params.w_enc_uni.len(), rng);
    let d_enc_b = sample_dir(params.w_enc_bi.len(), rng);
    let d_asym = sample_dir(params.w_init_asym.len(), rng);
    let d_ov = sample_dir(params.w_init_ov.len(), rng);
    let d_sol = sample_dir(params.w_sol.data.len(), rng);
    let d_nu = sample_dir(params.nu_sol.len(), rng);

    let make_perturbed = |sign: f64| -> ModelParams {
        let mut p = params.clone();
        let s = sign * sigma;
        for (w, &dv) in p.w_enc_uni.iter_mut().zip(d_enc_u.iter()) {
            *w += s * dv;
        }
        for (w, &dv) in p.w_enc_bi.iter_mut().zip(d_enc_b.iter()) {
            *w += s * dv;
        }
        for (w, &dv) in p.w_init_asym.iter_mut().zip(d_asym.iter()) {
            *w += s * dv;
        }
        for (w, &dv) in p.w_init_ov.iter_mut().zip(d_ov.iter()) {
            *w += s * dv;
        }
        for (w, &dv) in p.w_sol.data.iter_mut().zip(d_sol.iter()) {
            *w += s * dv;
        }
        for (w, &dv) in p.nu_sol.iter_mut().zip(d_nu.iter()) {
            *w += s * dv;
        }
        p
    };

    let mut short_flow = flow_cfg.clone();
    short_flow.max_steps = flow_cfg.max_steps.min(10);

    let p_plus = make_perturbed(1.0);
    let p_minus = make_perturbed(-1.0);
    let phi_plus = run_abductive_flow(&p_plus, effect, probes, phi_cfg, &short_flow, None)
        .final_eval
        .phi_sharp;
    let phi_minus = run_abductive_flow(&p_minus, effect, probes, phi_cfg, &short_flow, None)
        .final_eval
        .phi_sharp;

    let coeff = (phi_plus - phi_minus) / (2.0 * sigma);
    for (g, &dv) in grads.w_enc_uni.iter_mut().zip(d_enc_u.iter()) {
        *g += coeff * dv;
    }
    for (g, &dv) in grads.w_enc_bi.iter_mut().zip(d_enc_b.iter()) {
        *g += coeff * dv;
    }
    for (g, &dv) in grads.w_init_asym.iter_mut().zip(d_asym.iter()) {
        *g += coeff * dv;
    }
    for (g, &dv) in grads.w_init_ov.iter_mut().zip(d_ov.iter()) {
        *g += coeff * dv;
    }
    for (g, &dv) in grads.w_sol.data.iter_mut().zip(d_sol.iter()) {
        *g += coeff * dv;
    }
    for (g, &dv) in grads.nu_sol.iter_mut().zip(d_nu.iter()) {
        *g += coeff * dv;
    }
}

/// Exécute la séance unique de méta-entraînement sur HO-PTGRS (signal exclusif : \phi_\sharp).
pub fn train_session<F>(
    params: &mut ModelParams,
    phi_cfg: &PhiSharpConfig,
    flow_cfg: &FlowConfig,
    train_cfg: &TrainConfig,
    mut on_episode: F,
) -> Vec<EpisodeTrainLog>
where
    F: FnMut(&EpisodeTrainLog),
{
    let mut adam = AdamWState::new(params);
    let mut es_rng = Rng::from_seed(train_cfg.seed ^ 0x55AA_33CC_11EE_7799);
    let mut logs = Vec::with_capacity(train_cfg.episodes);

    let total_eps = train_cfg.episodes.max(1);

    for ep_idx in 0..train_cfg.episodes {
        let progress = (ep_idx as f64) / ((total_eps.saturating_sub(1)).max(1) as f64);

        let cur_k = lerp_usize(train_cfg.start_horizon_k, train_cfg.end_horizon_k, progress);
        let cur_d = lerp_usize(
            train_cfg.start_distractors,
            train_cfg.end_distractors,
            progress,
        );
        let cur_noise =
            train_cfg.start_noise + progress * (train_cfg.end_noise - train_cfg.start_noise);

        let cur_cfg = CurriculumConfig {
            horizon_k: cur_k,
            num_distractors: cur_d,
            noise_ratio: cur_noise,
            max_attempts: 64,
        };
        let mut gen_ = HoPtgrsGenerator::new(
            train_cfg
                .seed
                .wrapping_add((ep_idx as u64).wrapping_mul(0x9E37_79B9)),
            cur_cfg,
        );

        let ep = match gen_.generate_episode(None) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Seul `ep.perceived_eff` est visible ; `ep.hidden_cause` est ignoré.
        let effect = EffectSummary::from_bytes(&ep.perceived_eff, params.n);
        let probes = StabilityProbes::build(
            params.n,
            train_cfg.num_rot_probes,
            train_cfg.probe_angle,
            train_cfg.probe_abs_kappa,
            train_cfg.seed.wrapping_add((ep_idx as u64) * 101 + 7),
        );

        // 1. Trajectoire abductive R(0) -> R(T)
        let traj = run_abductive_flow(params, &effect, &probes, phi_cfg, flow_cfg, None);
        let first = traj.trace.first().unwrap();
        let last = traj.trace.last().unwrap();

        // 2. Gradient analytique exact \partial \phi_\sharp(R*, E) / \partial \theta
        let (_, mut grads) =
            grad_theta_phi_sharp(params, &traj.r_final, &effect, &probes, phi_cfg);

        // 3. Estimation antithétique sur \Psi_0 et W_sol (toujours via \phi_\sharp seul)
        if train_cfg.es_every > 0 && ep_idx % train_cfg.es_every == 0 {
            accumulate_trajectory_es_grad(
                params,
                &effect,
                &probes,
                phi_cfg,
                flow_cfg,
                train_cfg.es_sigma,
                &mut es_rng,
                &mut grads,
            );
        }

        let raw_grad_norm = grads.clip_norm(train_cfg.grad_clip_norm);

        let lr_t = train_cfg.lr
            * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()));
        adam.step_ascent(params, &grads, lr_t, train_cfg.weight_decay);

        let log = EpisodeTrainLog {
            episode: ep_idx + 1,
            horizon_k: cur_k,
            distractors: cur_d,
            eff_bytes: ep.perceived_eff.len(),
            phi_init: first.phi_sharp,
            phi_final: last.phi_sharp,
            lambda2_final: last.lambda2,
            residual_init: first.residual,
            residual_final: last.residual,
            v_stab_final: last.v_stab,
            active_edges: last.active_causal_edges,
            flow_steps: traj.trace.len().saturating_sub(1),
            grad_norm: raw_grad_norm,
        };
        on_episode(&log);
        logs.push(log);
    }

    logs
}

/// Évaluation post-hoc sur des épisodes HO-PTGRS frais (n'altère aucun poids).
pub fn evaluate_post_hoc(
    params: &ModelParams,
    phi_cfg: &PhiSharpConfig,
    flow_cfg: &FlowConfig,
    proj_cfg: &ProjectionConfig,
    num_episodes: usize,
    seed: u64,
) -> PostHocEvalSummary {
    let mut gen_ = HoPtgrsGenerator::new(seed, CurriculumConfig::default());
    let mut tested = 0usize;
    let mut sum_phi_gain = 0.0;
    let mut sum_lam2 = 0.0;
    let mut sum_res_drop = 0.0;
    let mut sum_alph_ratio = 0.0;
    let mut sum_target_overlap = 0.0;

    for i in 0..num_episodes {
        let ep = match gen_.generate_episode(None) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let effect = EffectSummary::from_bytes(&ep.perceived_eff, params.n);
        let probes = StabilityProbes::build(params.n, 2, 0.06, 0.12, seed + i as u64 + 13);
        let traj = run_abductive_flow(params, &effect, &probes, phi_cfg, flow_cfg, None);
        let proj = project_witness_mirror_ascent(
            params,
            &traj.r_final,
            &traj.final_eval,
            &ep.perceived_eff,
            proj_cfg,
        );

        let first = traj.trace.first().unwrap();
        let last = traj.trace.last().unwrap();
        sum_phi_gain += last.phi_sharp - first.phi_sharp;
        sum_lam2 += last.lambda2;
        sum_res_drop += first.residual - last.residual;

        let mut eff_present = [false; BYTE_VOCAB];
        for &b in &ep.perceived_eff {
            eff_present[b as usize] = true;
        }
        let in_alph = proj
            .witness_bytes
            .iter()
            .filter(|&&b| eff_present[b as usize])
            .count();
        sum_alph_ratio += (in_alph as f64) / (proj.witness_bytes.len().max(1) as f64);

        let target_bytes = &ep.hidden_cause.target_concept_bytes;
        let hits = target_bytes
            .iter()
            .filter(|b| proj.witness_bytes.contains(b))
            .count();
        sum_target_overlap += (hits as f64) / (target_bytes.len().max(1) as f64);

        tested += 1;
    }

    let denom = tested.max(1) as f64;
    PostHocEvalSummary {
        episodes_tested: tested,
        mean_phi_gain: sum_phi_gain / denom,
        mean_lambda2: sum_lam2 / denom,
        mean_residual_drop: sum_res_drop / denom,
        mean_witness_alphabet_ratio: sum_alph_ratio / denom,
        mean_target_byte_overlap: sum_target_overlap / denom,
    }
}

fn lerp_usize(a: usize, b: usize, t: f64) -> usize {
    let v = (a as f64) + t * ((b as f64) - (a as f64));
    v.round().max(1.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_theta_phi_sharp_exact_finite_diff() {
        let mut rng = Rng::from_seed(8080);
        let n = 5;
        let d = 2;
        let h = 6;
        let mut params = ModelParams::random_init(n, d, h, &mut rng);
        for v in &mut params.w_rel_uni {
            *v *= 2.5;
        }
        for v in &mut params.w_rel_bi {
            *v *= 2.5;
        }

        // Deux régimes structurels distincts pour obtenir un gap spectral net
        let mut bytes = Vec::with_capacity(80);
        for _ in 0..8 {
            bytes.extend_from_slice(&[12u8, 45, 78, 12, 45]);
        }
        for _ in 0..8 {
            bytes.extend_from_slice(&[190u8, 210, 230, 210, 230]);
        }
        let eff = EffectSummary::from_bytes(&bytes, n);
        let probes = StabilityProbes::build(n, 2, 0.06, 0.12, 1234);
        let cfg = PhiSharpConfig {
            beta_iit: 10.0,
            alpha: 0.4,
            beta: 0.3,
            lambda_bi: 0.5,
            hinge_tau: 1e-4,
            cap_eps: 1e-3,
            max_krylov: 16,
            max_inv_power: 6,
        };

        let mut r = params.encode_regional_pole(&eff, 4.0);
        for _ in 0..20 {
            let (_, g) = params.residual_and_grad_r(&r, &eff, cfg.lambda_bi);
            r.axpy(-1.2, &g);
            r.clamp_to_ball(4.0);
        }
        for c in 0..d {
            for i in 0..n {
                for j in 0..n {
                    if (i < 2) != (j < 2) {
                        r.set(i, j, c, r.get(i, j, c) * 0.45);
                    }
                }
            }
        }
        for _ in 0..3 {
            let (_, g) = params.residual_and_grad_r(&r, &eff, cfg.lambda_bi);
            r.axpy(-0.4, &g);
            r.clamp_to_ball(4.0);
        }

        let (_, grads) = grad_theta_phi_sharp(&params, &r, &eff, &probes, &cfg);
        let eps = 1e-6;

        let check = |label: &str, analytic: f64, mutate: &dyn Fn(&mut ModelParams, f64)| {
            let mut pp = params.clone();
            let mut pm = params.clone();
            mutate(&mut pp, eps);
            mutate(&mut pm, -eps);
            let (ep, _) = evaluate_phi_sharp(&pp, &r, &eff, &probes, &cfg, false);
            let (em, _) = evaluate_phi_sharp(&pm, &r, &eff, &probes, &cfg, false);
            let fd = (ep.phi_sharp - em.phi_sharp) / (2.0 * eps);
            assert!(
                (fd - analytic).abs() < 1e-4,
                "Param grad mismatch {label}: fd={fd}, analytic={analytic}"
            );
        };

        check("w_dir[0]", grads.w_dir[0], &|p, d| p.w_dir[0] += d);
        check("w_comp[1]", grads.w_comp[1], &|p, d| p.w_comp[1] += d);
        check("b_edge[2]", grads.b_edge[2], &|p, d| p.b_edge[2] += d);
        check("w_rel_uni[3]", grads.w_rel_uni[3], &|p, d| p.w_rel_uni[3] += d);
        check("b_rel_uni[4]", grads.b_rel_uni[4], &|p, d| p.b_rel_uni[4] += d);
        check("w_rel_bi[5]", grads.w_rel_bi[5], &|p, d| p.w_rel_bi[5] += d);
        check("b_rel_bi[6]", grads.b_rel_bi[6], &|p, d| p.b_rel_bi[6] += d);
    }

    #[test]
    fn test_short_training_session_runs_end_to_end() {
        let mut rng = Rng::from_seed(42);
        let mut params = ModelParams::random_init(5, 2, 6, &mut rng);
        let phi_cfg = PhiSharpConfig::default();
        let flow_cfg = FlowConfig {
            max_steps: 8,
            ..FlowConfig::default()
        };
        let train_cfg = TrainConfig {
            episodes: 3,
            es_every: 2,
            ..TrainConfig::default()
        };

        let logs = train_session(&mut params, &phi_cfg, &flow_cfg, &train_cfg, |_| {});
        assert_eq!(logs.len(), 3);
        for l in &logs {
            assert!(l.phi_final.is_finite());
            assert!(l.phi_final >= l.phi_init - 1e-6);
        }
    }
}
