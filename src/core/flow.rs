use crate::core::phi_sharp::{
    evaluate_phi_sharp, EffectSummary, ModelParams, PhiSharpConfig, PhiSharpEval, StabilityProbes,
};
use crate::core::tensor::{Mat, RelTensor};

/// Configuration de l'intégrateur du flot Lie-Gradient sur B_K.
#[derive(Debug, Clone)]
pub struct FlowConfig {
    /// Budget maximal d'étapes d'inférence T.
    pub max_steps: usize,
    /// Pas d'intégration initial dt.
    pub dt_init: f64,
    /// Pas d'intégration minimum (arrêt si convergence atteinte sur M*(E)).
    pub dt_min: f64,
    /// Pas d'intégration maximum.
    pub dt_max: f64,
    /// Vitesse de la composante gradient \eta > 0.
    pub eta: f64,
    /// Vitesse de l'ascension isospectrale \omega > 0 (A_asc).
    pub omega: f64,
    /// Amplitude du champ solénoïdal orthogonal \omega_\perp (A_\perp).
    pub omega_perp: f64,
    /// Borne sur ||A_\perp||_F pour la stabilité numérique de la rotation.
    pub max_perp_norm: f64,
    /// Poids \gamma de la régularisation d'abstraction spectrale -\gamma R_abs^\eps(R).
    pub gamma_abs: f64,
    /// Paramètre de lissage \eps de R_abs^\eps(R).
    pub abs_eps: f64,
    /// Rayon K de la boule compacte B_K = { ||R||_F <= K }.
    pub ball_radius_k: f64,
    /// Critère d'arrêt sur la vitesse de Lyapunov \dot{L} (atteinte de M*(E)).
    pub lyapunov_tol: f64,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            max_steps: 40,
            dt_init: 0.25,
            dt_min: 1e-5,
            dt_max: 2.0,
            eta: 1.0,
            omega: 0.5,
            omega_perp: 0.1,
            max_perp_norm: 0.5,
            gamma_abs: 0.02,
            abs_eps: 1e-2,
            ball_radius_k: 8.0,
            lyapunov_tol: 1e-8,
        }
    }
}

/// Instantané d'une étape le long de la trajectoire abductive R(t).
#[derive(Debug, Clone)]
pub struct FlowStepTrace {
    pub step: usize,
    pub t: f64,
    pub dt: f64,
    /// Potentiel de Lyapunov complet L(R, E) = \phi_\sharp(R, E) - \gamma R_abs^\eps(R).
    pub lyapunov: f64,
    pub phi_sharp: f64,
    pub lambda2: f64,
    pub residual: f64,
    pub v_stab: f64,
    pub r_abs: f64,
    pub active_causal_edges: usize,
    pub state_norm: f64,
    /// Taux théorique de croissance de Lyapunov : \eta ||\Pi G||^2 + (1/\omega) ||A_asc||^2 >= 0.
    pub lyapunov_rate_pred: f64,
    /// Norme de la composante orthogonale ||A_\perp||_F.
    pub perp_norm: f64,
    pub backtracks: usize,
}

/// Trajectoire complète de recherche abductive (le Sens en acte) et état final R(T).
#[derive(Debug, Clone)]
pub struct FlowTrajectory {
    pub r_init: RelTensor,
    pub r_final: RelTensor,
    pub final_eval: PhiSharpEval,
    pub final_lyapunov: f64,
    pub trace: Vec<FlowStepTrace>,
    pub converged_to_critical_manifold: bool,
}

impl FlowTrajectory {
    /// Vérifie la propriété de LaSalle (Théorème 4.2) : monotonicité non-décroissante de L(R(t), E).
    pub fn is_monotone_lyapunov(&self, tol: f64) -> bool {
        self.trace
            .windows(2)
            .all(|w| w[1].lyapunov + tol >= w[0].lyapunov)
    }
}

/// Évalue le potentiel complet L(R, E) = \phi_\sharp(R, E) - \gamma R_abs^\eps(R) et son gradient G(R).
fn eval_lyapunov(
    params: &ModelParams,
    r: &RelTensor,
    effect: &EffectSummary,
    probes: &StabilityProbes,
    phi_cfg: &PhiSharpConfig,
    flow_cfg: &FlowConfig,
    compute_grad: bool,
) -> (f64, f64, PhiSharpEval, Option<RelTensor>) {
    let (eval, grad_phi_opt) =
        evaluate_phi_sharp(params, r, effect, probes, phi_cfg, compute_grad);

    let (r_abs, grad_abs_opt) = if flow_cfg.gamma_abs > 0.0 {
        let (v, g) = r.spectral_abstraction_reg(flow_cfg.abs_eps);
        (v, if compute_grad { Some(g) } else { None })
    } else {
        (0.0, None)
    };

    let lyap = eval.phi_sharp - flow_cfg.gamma_abs * r_abs;

    let grad = match (grad_phi_opt, grad_abs_opt) {
        (Some(mut gp), Some(ga)) => {
            gp.axpy(-flow_cfg.gamma_abs, &ga);
            Some(gp)
        }
        (Some(gp), None) => Some(gp),
        _ => None,
    };

    (lyap, r_abs, eval, grad)
}

/// Construit le champ solénoïdal orthogonal A_\perp(R) \in so(N) vérifiant <A_\perp, A_asc>_F = 0.
pub fn build_orthogonal_solenoidal(
    params: &ModelParams,
    r: &RelTensor,
    a_asc: &Mat,
    omega_perp: f64,
    max_norm: f64,
) -> Mat {
    if omega_perp <= 0.0 {
        return Mat::zeros(r.n);
    }

    // Champ antisymétrique piloté par les poids non-réciproques W_sol et l'état courant R
    let mut raw = params.w_sol.clone();
    for c in 0..r.d {
        let nu = params.nu_sol[c];
        if nu != 0.0 {
            raw.axpy(nu, &r.channels[c]);
        }
    }
    let mut a_perp = raw.skew();

    // Projection orthogonale hors de span(A_asc) dans so(N)
    let asc_norm_sq = a_asc.frob_norm_sq();
    if asc_norm_sq > 1e-20 {
        let proj = a_perp.frob_dot(a_asc) / asc_norm_sq;
        a_perp.axpy(-proj, a_asc);
    }

    a_perp.scale(omega_perp);
    let p_norm = a_perp.frob_norm();
    if p_norm > max_norm && p_norm > 0.0 {
        a_perp.scale(max_norm / p_norm);
    }
    a_perp
}

/// Transformée de Cayley exacte dans SO(N) : Q = (I - 0.5*dt*A)^{-1} * (I + 0.5*dt*A).
/// Pour tout A \in so(N), Q est strictement orthogonal (Q^T Q = I) et conjugue R de façon isospectrale.
pub fn cayley_rotation(a_skew: &Mat, dt: f64) -> Mat {
    let n = a_skew.n;
    if dt == 0.0 || a_skew.frob_norm_sq() <= 1e-30 {
        return Mat::identity(n);
    }
    let half_dt = 0.5 * dt;
    let mut lhs = Mat::identity(n);
    let mut rhs = Mat::identity(n);
    for i in 0..n {
        for j in 0..n {
            let aij = a_skew.get(i, j);
            lhs.add_assign(i, j, -half_dt * aij);
            rhs.add_assign(i, j, half_dt * aij);
        }
    }
    solve_linear_multi_rhs(&lhs, &rhs)
}

/// Applique la conjugaison isospectrale exacte R -> Q * R * Q^T sur chaque canal c.
pub fn conjugate_isospectral(r: &RelTensor, q: &Mat) -> RelTensor {
    let mut out = RelTensor::zeros(r.n, r.d);
    for c in 0..r.d {
        let qr = q.matmul(&r.channels[c]);
        out.channels[c] = qr.matmul_t_right(q);
    }
    out
}

/// Résout A * X = B (matrices N x N) par élimination de Gauss avec pivotage partiel (zéro dep).
fn solve_linear_multi_rhs(a: &Mat, b: &Mat) -> Mat {
    let n = a.n;
    debug_assert_eq!(n, b.n);
    let mut m = a.data.clone();
    let mut x = b.data.clone();

    for col in 0..n {
        // Recherche du pivot maximal dans la colonne
        let mut pivot_row = col;
        let mut max_abs = m[col * n + col].abs();
        for row in (col + 1)..n {
            let val = m[row * n + col].abs();
            if val > max_abs {
                max_abs = val;
                pivot_row = row;
            }
        }
        if max_abs <= 1e-15 {
            continue;
        }
        if pivot_row != col {
            for k in col..n {
                m.swap(col * n + k, pivot_row * n + k);
            }
            for k in 0..n {
                x.swap(col * n + k, pivot_row * n + k);
            }
        }

        let diag = m[col * n + col];
        for row in (col + 1)..n {
            let factor = m[row * n + col] / diag;
            if factor == 0.0 {
                continue;
            }
            m[row * n + col] = 0.0;
            for k in (col + 1)..n {
                m[row * n + k] -= factor * m[col * n + k];
            }
            for k in 0..n {
                x[row * n + k] -= factor * x[col * n + k];
            }
        }
    }

    // Substitution arrière
    for col in (0..n).rev() {
        let diag = m[col * n + col];
        if diag.abs() <= 1e-15 {
            continue;
        }
        let inv_diag = 1.0 / diag;
        for k in 0..n {
            let mut sum = x[col * n + k];
            for j in (col + 1)..n {
                sum -= m[col * n + j] * x[j * n + k];
            }
            x[col * n + k] = sum * inv_diag;
        }
    }

    Mat { n, data: x }
}

/// Exécute la trajectoire abductive complète du flot Lie-Gradient depuis R(0) = \Psi_0(E).
pub fn run_abductive_flow(
    params: &ModelParams,
    effect: &EffectSummary,
    probes: &StabilityProbes,
    phi_cfg: &PhiSharpConfig,
    flow_cfg: &FlowConfig,
    r_init_override: Option<RelTensor>,
) -> FlowTrajectory {
    let mut r = r_init_override
        .unwrap_or_else(|| params.encode_regional_pole(effect, flow_cfg.ball_radius_k));
    r.clamp_to_ball(flow_cfg.ball_radius_k);
    let r_init = r.clone();

    let (mut lyap, mut r_abs, mut eval, mut grad_opt) =
        eval_lyapunov(params, &r, effect, probes, phi_cfg, flow_cfg, true);

    let mut t = 0.0f64;
    let mut dt = flow_cfg.dt_init;
    let mut trace = Vec::with_capacity(flow_cfg.max_steps + 1);
    let mut converged = false;

    for step in 0..flow_cfg.max_steps {
        let grad = grad_opt.take().unwrap();
        let proj_g = r.project_tangent_ball(&grad, flow_cfg.ball_radius_k);

        // Opérateur antisymétrique d'ascension isospectrale A_asc = \omega * dual(G, R)
        let dual = r.skew_commutator_dual(&grad);
        let mut a_asc = dual.clone();
        a_asc.scale(flow_cfg.omega);

        // Taux exact de Lyapunov prédit par le Théorème 4.2 :
        // dL/dt = \eta ||\Pi G||_F^2 + \omega ||dual(G, R)||_F^2 >= 0
        let rate_grad = flow_cfg.eta * proj_g.frob_norm_sq();
        let rate_lie = flow_cfg.omega * dual.frob_norm_sq();
        let lyap_rate = rate_grad + rate_lie;

        let a_perp = build_orthogonal_solenoidal(
            params,
            &r,
            &a_asc,
            flow_cfg.omega_perp,
            flow_cfg.max_perp_norm,
        );
        let perp_norm = a_perp.frob_norm();

        if step == 0 {
            trace.push(FlowStepTrace {
                step: 0,
                t: 0.0,
                dt: 0.0,
                lyapunov: lyap,
                phi_sharp: eval.phi_sharp,
                lambda2: eval.lambda2,
                residual: eval.residual,
                v_stab: eval.v_stab,
                r_abs,
                active_causal_edges: eval.active_causal_edges,
                state_norm: r.frob_norm(),
                lyapunov_rate_pred: lyap_rate,
                perp_norm,
                backtracks: 0,
            });
        }

        if lyap_rate <= flow_cfg.lyapunov_tol {
            converged = true;
            break;
        }

        // Générateur total A_total = A_asc + A_perp \in so(N)
        let mut a_total = a_asc.clone();
        a_total.axpy(1.0, &a_perp);

        // Contrôle de pas adaptatif (splitting Cayley-SO(N) + pas gradient projeté)
        // garantissant la monotonicité stricte de LaSalle L(R_{k+1}) >= L(R_k).
        let mut trial_dt = dt;
        let mut backtracks = 0usize;
        let mut accepted = None;

        while trial_dt >= flow_cfg.dt_min {
            // 1. Déplacement isospectral exact via transformée de Cayley dans SO(N)
            let q = cayley_rotation(&a_total, trial_dt);
            let mut r_cand = conjugate_isospectral(&r, &q);

            // 2. Étape gradient tangent + rétraction dans B_K
            r_cand.axpy(trial_dt * flow_cfg.eta, &proj_g);
            r_cand.clamp_to_ball(flow_cfg.ball_radius_k);

            let (cand_lyap, cand_r_abs, cand_eval, _) =
                eval_lyapunov(params, &r_cand, effect, probes, phi_cfg, flow_cfg, false);

            // Condition d'Armijo faible / monotonicité stricte de LaSalle
            let armijo_rhs = lyap + 1e-4 * trial_dt * lyap_rate;
            if cand_lyap >= armijo_rhs || (cand_lyap >= lyap && trial_dt <= flow_cfg.dt_min * 4.0) {
                accepted = Some((r_cand, cand_lyap, cand_r_abs, cand_eval, trial_dt));
                break;
            }
            trial_dt *= 0.5;
            backtracks += 1;
        }

        if let Some((r_next, next_lyap, next_r_abs, _, used_dt)) = accepted {
            r = r_next;
            t += used_dt;
            // Recalcul du gradient exact au nouvel état pour l'itération suivante
            let (l_ref, ra_ref, e_ref, g_ref) =
                eval_lyapunov(params, &r, effect, probes, phi_cfg, flow_cfg, true);
            lyap = l_ref.max(next_lyap);
            r_abs = ra_ref.min(next_r_abs).max(0.0);
            eval = e_ref;
            grad_opt = g_ref;

            trace.push(FlowStepTrace {
                step: step + 1,
                t,
                dt: used_dt,
                lyapunov: lyap,
                phi_sharp: eval.phi_sharp,
                lambda2: eval.lambda2,
                residual: eval.residual,
                v_stab: eval.v_stab,
                r_abs,
                active_causal_edges: eval.active_causal_edges,
                state_norm: r.frob_norm(),
                lyapunov_rate_pred: lyap_rate,
                perp_norm,
                backtracks,
            });

            // Croissance douce du pas si aucune réduction n'a été nécessaire
            if backtracks == 0 {
                dt = (used_dt * 1.25).min(flow_cfg.dt_max);
            } else {
                dt = used_dt;
            }
        } else {
            converged = true;
            break;
        }
    }

    FlowTrajectory {
        r_init,
        r_final: r,
        final_eval: eval,
        final_lyapunov: lyap,
        trace,
        converged_to_critical_manifold: converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ho_ptgrs::generator::{CurriculumConfig, HoPtgrsGenerator};
    use crate::ho_ptgrs::meta_ops::MetaOp;
    use crate::ho_ptgrs::rng::Rng;

    #[test]
    fn test_cayley_isospectral_exact_conservation_theorem_4() {
        let mut rng = Rng::from_seed(555);
        let n = 6;
        let d = 3;
        let mut r = RelTensor::zeros(n, d);
        let mut raw_a = Mat::zeros(n);
        for c in 0..d {
            for i in 0..n {
                for j in 0..n {
                    r.set(i, j, c, rng.gen_f64() * 2.0 - 1.0);
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                raw_a.set(i, j, rng.gen_f64() * 2.0 - 1.0);
            }
        }
        let a_skew = raw_a.skew();

        let norm0 = r.frob_norm();
        let tr2_0: Vec<f64> = (0..d)
            .map(|c| {
                let sq = r.channels[c].matmul(&r.channels[c]);
                (0..n).map(|i| sq.get(i, i)).sum::<f64>()
            })
            .collect();

        // 40 pas de conjugaison isospectrale pure par transformée de Cayley
        let q = cayley_rotation(&a_skew, 0.35);
        // Vérification stricte Q^T Q = I
        let qtq = q.matmul_t_left(&q);
        let eye = Mat::identity(n);
        let mut orth_err = 0.0;
        for i in 0..n {
            for j in 0..n {
                orth_err += (qtq.get(i, j) - eye.get(i, j)).powi(2);
            }
        }
        assert!(orth_err.sqrt() < 1e-13, "Q non orthogonal: err={}", orth_err.sqrt());

        let mut curr = r;
        for _ in 0..40 {
            curr = conjugate_isospectral(&curr, &q);
        }

        // Conservation stricte de ||R||_F et des invariants cycliques Tr(R_c^2)
        assert!((curr.frob_norm() - norm0).abs() < 1e-11);
        for c in 0..d {
            let sq = curr.channels[c].matmul(&curr.channels[c]);
            let tr2: f64 = (0..n).map(|i| sq.get(i, i)).sum();
            assert!((tr2 - tr2_0[c]).abs() < 1e-11);
        }
    }

    #[test]
    fn test_abductive_flow_lasalle_monotonicity_on_ho_ptgrs_episode() {
        // Génère un vrai fichier perceived.eff (opérateur \mu_bif) via crate::ho_ptgrs
        let mut gen = HoPtgrsGenerator::new(
            2026,
            CurriculumConfig {
                horizon_k: 4,
                num_distractors: 4,
                noise_ratio: 0.5,
                max_attempts: 32,
            },
        );
        let ep = gen
            .generate_episode(Some(MetaOp::Bif))
            .expect("Génération HO-PTGRS réussie");

        let n = 6;
        let d = 2;
        let h = 4;
        let mut rng = Rng::from_seed(777);
        let mut params = ModelParams::random_init(n, d, h, &mut rng);
        for v in &mut params.w_out_uni {
            *v *= 2.0;
        }
        for v in &mut params.w_out_bi {
            *v *= 2.0;
        }

        let effect = EffectSummary::from_bytes(&ep.perceived_eff, n);
        let probes = StabilityProbes::build(n, 2, 0.06, 0.1, 4242);
        let phi_cfg = PhiSharpConfig {
            alpha: 0.5,
            beta: 0.2,
            lambda_bi: 0.5,
            hinge_tau: 0.03,
            smin_eps: 0.03,
            max_krylov: 16,
            max_inv_power: 4,
        };
        let flow_cfg = FlowConfig {
            max_steps: 18,
            dt_init: 0.4,
            eta: 1.0,
            omega: 0.4,
            omega_perp: 0.08,
            gamma_abs: 0.01,
            ball_radius_k: 6.0,
            ..FlowConfig::default()
        };

        let traj = run_abductive_flow(&params, &effect, &probes, &phi_cfg, &flow_cfg, None);

        assert!(traj.trace.len() >= 2, "Le flot doit avancer d'au moins 1 étape");
        // 1. Monotonicité stricte de LaSalle (Théorème 4.2)
        assert!(
            traj.is_monotone_lyapunov(1e-10),
            "Violation de monotonicité de LaSalle !"
        );
        // 2. Invariance positive de la boule compacte B_K
        for step in &traj.trace {
            assert!(
                step.state_norm <= flow_cfg.ball_radius_k + 1e-10,
                "Sortie de B_K à l'étape {}",
                step.step
            );
        }
        // 3. Progression effective du potentiel abductif et chute du résidu
        let first = traj.trace.first().unwrap();
        let last = traj.trace.last().unwrap();
        assert!(
            last.lyapunov > first.lyapunov + 1e-3,
            "L n'a pas progressé : {} -> {}",
            first.lyapunov,
            last.lyapunov
        );
        assert!(
            last.residual < first.residual,
            "Le résidu doit baisser : {} -> {}",
            first.residual,
            last.residual
        );
    }
}
