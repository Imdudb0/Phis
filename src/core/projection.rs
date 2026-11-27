use crate::core::phi_sharp::{
    fold_nibble, spatial_assignment, ModelParams, PhiSharpEval, BYTE_VOCAB,
};
use crate::core::tensor::{Mat, RelTensor};

/// Configuration de la projection référentielle par montée miroir sur \Delta_{255}^{L'}.
#[derive(Debug, Clone)]
pub struct ProjectionConfig {
    /// Longueur de la séquence témoin d'octets L'.
    pub out_len: usize,
    /// Nombre maximal d'itérations K de montée miroir parallèle.
    pub max_steps: usize,
    /// Pas initial \eta_Y de la mise à jour exponentielle de Bregman.
    pub eta_init: f64,
    /// Pas minimal avant arrêt par convergence.
    pub eta_min: f64,
    /// Pas maximal autorisé.
    pub eta_max: f64,
    /// Poids relatif du couplage bigramme inter-positions dans \Psi.
    pub lambda_bi: f64,
    /// Poids d'ancrage sur l'alphabet observé dans `perceived.eff` (auto-spécification).
    pub gamma_alphabet: f64,
    /// Pénalité anti-copie positionnelle triviale de `perceived.eff`.
    pub gamma_anticopy: f64,
    /// Poids de pureté quadratique (Tsallis, gradient borné sur \Delta_{255}^{L'}).
    pub tau_sharp: f64,
    /// Seuil d'arrêt sur ||Y^{(k+1)} - Y^{(k)}||_{1,2}.
    pub tol_step_l12: f64,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            out_len: 16,
            max_steps: 80,
            eta_init: 4.0,
            eta_min: 1e-4,
            eta_max: 64.0,
            lambda_bi: 0.5,
            gamma_alphabet: 0.35,
            gamma_anticopy: 0.25,
            tau_sharp: 0.15,
            tol_step_l12: 1e-5,
        }
    }
}

/// Instantané d'une itération de montée miroir (Théorème 5).
#[derive(Debug, Clone)]
pub struct ProjectionStepTrace {
    pub step: usize,
    pub psi: f64,
    pub eta: f64,
    pub kl_old_new: f64,
    pub step_l12: f64,
}

/// Résultat complet de la projection référentielle non-autorégressive.
#[derive(Debug, Clone)]
pub struct ProjectionResult {
    /// Distribution finale sur le produit de simplexes \Delta_{255}^{L'} (row-major L' x 256).
    pub soft_bytes: Vec<f64>,
    /// Séquence témoin décodée par argmax parallèle.
    pub witness_bytes: Vec<u8>,
    /// Confiance moyenne max_b Y_{\ell, b} sur les L' positions.
    pub mean_confidence: f64,
    /// Valeur finale de \Psi(Y^{(K)}; R^*, E).
    pub final_psi: f64,
    /// Trace de convergence (permet de vérifier le Théorème 5).
    pub trace: Vec<ProjectionStepTrace>,
    pub converged: bool,
}

impl ProjectionResult {
    /// Vérifie la croissance monotone du Théorème 5.1 à tolérance numérique près.
    pub fn is_monotone_psi(&self, tol: f64) -> bool {
        self.trace.windows(2).all(|w| w[1].psi + tol >= w[0].psi)
    }

    /// Représentation lisible UTF-8 / ASCII imprimable des octets témoins.
    pub fn display_lossy_nl(&self) -> String {
        self.witness_bytes
            .iter()
            .map(|&b| {
                if (0x20..=0x7E).contains(&b) {
                    b as char
                } else {
                    '.'
                }
            })
            .collect()
    }
}

/// Champs précalculés dépendant uniquement de (R^*, E) pour évaluer \Psi(Y) et \nabla_Y \Psi(Y) en O(L' * 256).
struct ProjectionField {
    out_len: usize,
    w_uni: f64,
    w_bi: f64,
    /// Potentiel linéaire combiné c_{\ell, b} = (w_uni * u_{\ell, b} + gamma_alph * log q_E(b)) / L'.
    linear_pot: Vec<f64>,
    /// Potentiel bilinéaire B_{\ell, r, s} pour \ell \in 1..L' (stocké en (L'-1) x 256).
    bi_pot: Vec<f64>,
    /// Préfixe de `perceived.eff` pour la pénalité anti-copie positionnelle.
    copy_ref: Vec<u8>,
    gamma_anticopy: f64,
    tau_sharp: f64,
}

impl ProjectionField {
    fn build(
        params: &ModelParams,
        r_star: &RelTensor,
        eval_star: &PhiSharpEval,
        perceived_eff: &[u8],
        cfg: &ProjectionConfig,
    ) -> Self {
        let n = params.n;
        let d = params.d;
        let h = params.h;
        let out_len = cfg.out_len.max(1);
        let inv_sqrt_n = 1.0 / (n as f64).sqrt();

        // 1. Calcul des log-distributions d'émission log p_i^{uni}(b | R^*) et log p_i^{bi}(g | R^*)
        let comp: Vec<Mat> = (0..d)
            .map(|c| r_star.channels[c].matmul(&r_star.channels[c]))
            .collect();

        let mut log_p_uni = vec![0.0; n * BYTE_VOCAB];
        let mut log_p_bi = vec![0.0; n * BYTE_VOCAB];
        let mut h_node = vec![0.0; h];

        for i in 0..n {
            h_node.fill(0.0);
            for j in 0..n {
                for r_idx in 0..h {
                    let mut z = params.b_edge[r_idx];
                    let w_d = &params.w_dir[r_idx * d..(r_idx + 1) * d];
                    let w_c = &params.w_comp[r_idx * d..(r_idx + 1) * d];
                    for c in 0..d {
                        z += w_d[c] * r_star.get(i, j, c) + w_c[c] * comp[c].get(i, j);
                    }
                    h_node[r_idx] += z.tanh();
                }
            }
            for v in &mut h_node {
                *v *= inv_sqrt_n;
            }

            compute_log_softmax(
                &h_node,
                &params.w_out_uni,
                &params.b_out_uni,
                &mut log_p_uni[i * BYTE_VOCAB..(i + 1) * BYTE_VOCAB],
            );
            compute_log_softmax(
                &h_node,
                &params.w_out_bi,
                &params.b_out_bi,
                &mut log_p_bi[i * BYTE_VOCAB..(i + 1) * BYTE_VOCAB],
            );
        }

        // 2. Poids d'importance causale des nœuds (degré causal symétrisé + énergie de Fiedler v_2^2)
        let mut causal_deg = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                if i != j {
                    sum += 0.5
                        * (eval_star.causal_weights.get(i, j)
                            + eval_star.causal_weights.get(j, i));
                }
            }
            causal_deg[i] = sum;
        }
        let max_deg = causal_deg
            .iter()
            .copied()
            .fold(0.0f64, f64::max)
            .max(1e-12);
        let mut node_imp = vec![1.0; n];
        for i in 0..n {
            let deg_norm = causal_deg[i] / max_deg;
            let fiedler_e = eval_star
                .fiedler
                .v2
                .get(i)
                .map_or(0.0, |v| v * v * (n as f64));
            node_imp[i] = 1.0 + deg_norm + 0.5 * fiedler_e;
        }

        // 3. Support d'alphabet observé dans `perceived.eff` (auto-spécification)
        let mut counts = [0usize; BYTE_VOCAB];
        for &b in perceived_eff {
            counts[b as usize] += 1;
        }
        let present_types = counts.iter().filter(|&&c| c > 0).count();
        let mut log_q_alph = [0.0f64; BYTE_VOCAB];
        if present_types > 0 && present_types < BYTE_VOCAB {
            let eps_out = 0.02;
            let p_out = eps_out / ((BYTE_VOCAB - present_types) as f64);
            let total = perceived_eff.len() as f64;
            for b in 0..BYTE_VOCAB {
                let p = if counts[b] > 0 {
                    // Mélange uniforme sur l'alphabet présent + fréquence douce
                    let p_unif = (1.0 - eps_out) * 0.7 / (present_types as f64);
                    let p_emp = (1.0 - eps_out) * 0.3 * (counts[b] as f64) / total;
                    p_unif + p_emp
                } else {
                    p_out
                };
                log_q_alph[b] = p.ln();
            }
        } else {
            let unif_log = (1.0 / (BYTE_VOCAB as f64)).ln();
            log_q_alph.fill(unif_log);
        }

        let has_bi = out_len >= 2 && cfg.lambda_bi > 0.0;
        let (w_uni, w_bi) = if has_bi {
            let denom = 1.0 + cfg.lambda_bi;
            (1.0 / denom, cfg.lambda_bi / denom)
        } else {
            (1.0, 0.0)
        };

        // 4. Construction des potentiels spatialisés par position \ell \in 0..L'
        let inv_l = 1.0 / (out_len as f64);
        let mut linear_pot = vec![0.0; out_len * BYTE_VOCAB];
        let mut bi_pot = if has_bi {
            vec![0.0; (out_len - 1) * BYTE_VOCAB]
        } else {
            Vec::new()
        };
        let mut assign = vec![0.0; n];

        for pos in 0..out_len {
            spatial_assignment(pos, out_len, n, &mut assign);
            let mut z_sum = 0.0;
            for i in 0..n {
                assign[i] *= node_imp[i];
                z_sum += assign[i];
            }
            let inv_z = 1.0 / z_sum.max(1e-12);
            for a in &mut assign {
                *a *= inv_z;
            }

            let lin_row = &mut linear_pot[pos * BYTE_VOCAB..(pos + 1) * BYTE_VOCAB];
            for b in 0..BYTE_VOCAB {
                let mut u_val = 0.0;
                for i in 0..n {
                    u_val += assign[i] * log_p_uni[i * BYTE_VOCAB + b];
                }
                lin_row[b] = inv_l * (w_uni * u_val + cfg.gamma_alphabet * log_q_alph[b]);
            }

            if has_bi && pos >= 1 {
                let bi_row = &mut bi_pot[(pos - 1) * BYTE_VOCAB..pos * BYTE_VOCAB];
                for g in 0..BYTE_VOCAB {
                    let mut b_val = 0.0;
                    for i in 0..n {
                        b_val += assign[i] * log_p_bi[i * BYTE_VOCAB + g];
                    }
                    bi_row[g] = b_val;
                }
            }
        }

        let copy_len = out_len.min(perceived_eff.len());
        Self {
            out_len,
            w_uni,
            w_bi,
            linear_pot,
            bi_pot,
            copy_ref: perceived_eff[..copy_len].to_vec(),
            gamma_anticopy: cfg.gamma_anticopy,
            tau_sharp: cfg.tau_sharp,
        }
    }

    /// Évalue exactement \Psi(Y; R^*, E) et (optionnellement) \nabla_Y \Psi(Y).
    fn eval_and_grad(&self, y: &[f64], grad_out: Option<&mut [f64]>) -> f64 {
        let l = self.out_len;
        let inv_l = 1.0 / (l as f64);
        let mut psi = 0.0;

        let mut grad = grad_out;
        if let Some(g) = grad.as_deref_mut() {
            g.fill(0.0);
        }

        // 1. Terme linéaire (unigramme + ancrage alphabet) + pureté quadratique Tsallis
        for pos in 0..l {
            let y_row = &y[pos * BYTE_VOCAB..(pos + 1) * BYTE_VOCAB];
            let c_row = &self.linear_pot[pos * BYTE_VOCAB..(pos + 1) * BYTE_VOCAB];
            for b in 0..BYTE_VOCAB {
                let yb = y_row[b];
                psi += yb * c_row[b] + 0.5 * self.tau_sharp * inv_l * yb * yb;
                if let Some(g) = grad.as_deref_mut() {
                    g[pos * BYTE_VOCAB + b] += c_row[b] + self.tau_sharp * inv_l * yb;
                }
            }
        }

        // 2. Pénalité anti-copie positionnelle triviale
        if self.gamma_anticopy > 0.0 {
            let coef = self.gamma_anticopy * inv_l;
            for (pos, &eb) in self.copy_ref.iter().enumerate() {
                let b = eb as usize;
                let yb = y[pos * BYTE_VOCAB + b];
                psi -= 0.5 * coef * yb * yb;
                if let Some(g) = grad.as_deref_mut() {
                    g[pos * BYTE_VOCAB + b] -= coef * yb;
                }
            }
        }

        // 3. Couplage bilinéaire bigramme inter-positions (en parallèle sur 0..L')
        if self.w_bi > 0.0 && l >= 2 {
            let bi_scale = self.w_bi / ((l - 1) as f64);
            let mut nib = vec![0.0; l * 16];
            for pos in 0..l {
                let y_row = &y[pos * BYTE_VOCAB..(pos + 1) * BYTE_VOCAB];
                let n_row = &mut nib[pos * 16..(pos + 1) * 16];
                for b in 0..BYTE_VOCAB {
                    n_row[fold_nibble(b as u8)] += y_row[b];
                }
            }

            let mut nib_grad = if grad.is_some() {
                vec![0.0; l * 16]
            } else {
                Vec::new()
            };

            for pos in 1..l {
                let b_mat = &self.bi_pot[(pos - 1) * BYTE_VOCAB..pos * BYTE_VOCAB];
                let prev_n = &nib[(pos - 1) * 16..pos * 16];
                let curr_n = &nib[pos * 16..(pos + 1) * 16];
                for r in 0..16 {
                    let pr = prev_n[r];
                    let row = &b_mat[r * 16..(r + 1) * 16];
                    for s in 0..16 {
                        let brs = bi_scale * row[s];
                        psi += pr * curr_n[s] * brs;
                        if !nib_grad.is_empty() {
                            nib_grad[(pos - 1) * 16 + r] += curr_n[s] * brs;
                            nib_grad[pos * 16 + s] += pr * brs;
                        }
                    }
                }
            }

            if let Some(g) = grad.as_deref_mut() {
                for pos in 0..l {
                    let ng = &nib_grad[pos * 16..(pos + 1) * 16];
                    let g_row = &mut g[pos * BYTE_VOCAB..(pos + 1) * BYTE_VOCAB];
                    for b in 0..BYTE_VOCAB {
                        g_row[b] += ng[fold_nibble(b as u8)];
                    }
                }
            }
        }

        psi
    }
}

/// Exécute la montée miroir parallèle sur le simplexe \Delta_{255}^{L'} (Théorème 5).
pub fn project_witness_mirror_ascent(
    params: &ModelParams,
    r_star: &RelTensor,
    eval_star: &PhiSharpEval,
    perceived_eff: &[u8],
    cfg: &ProjectionConfig,
) -> ProjectionResult {
    let l = cfg.out_len.max(1);
    let field = ProjectionField::build(params, r_star, eval_star, perceived_eff, cfg);

    // Centre barycentrique uniforme du produit de simplexes \Delta_{255}^{L'}
    let mut y = vec![1.0 / (BYTE_VOCAB as f64); l * BYTE_VOCAB];
    let mut y_cand = vec![0.0; l * BYTE_VOCAB];
    let mut grad = vec![0.0; l * BYTE_VOCAB];

    let mut psi = field.eval_and_grad(&y, Some(&mut grad));
    let mut eta = cfg.eta_init;
    let mut converged = false;

    let mut trace = Vec::with_capacity(cfg.max_steps + 1);
    trace.push(ProjectionStepTrace {
        step: 0,
        psi,
        eta: 0.0,
        kl_old_new: 0.0,
        step_l12: 0.0,
    });

    for step in 1..=cfg.max_steps {
        let mut trial_eta = eta;
        let mut accepted = None;

        while trial_eta >= cfg.eta_min {
            mirror_step(&y, &grad, l, trial_eta, &mut y_cand);
            let cand_psi = field.eval_and_grad(&y_cand, None);
            let kl_old_new = kl_divergence_simplex(&y, &y_cand);

            // Condition exacte du Théorème 5.1 :
            // \Psi(Y^{(k+1)}) - \Psi(Y^{(k)}) >= (1 / \eta_Y) * D_KL(Y^{(k)} || Y^{(k+1)})
            if cand_psi + 1e-12 >= psi + kl_old_new / trial_eta {
                let l12 = norm_l12_diff(&y, &y_cand, l);
                accepted = Some((cand_psi, kl_old_new, l12, trial_eta));
                break;
            }
            trial_eta *= 0.5;
        }

        if let Some((next_psi, kl_old_new, step_l12, used_eta)) = accepted {
            y.copy_from_slice(&y_cand);
            psi = field.eval_and_grad(&y, Some(&mut grad));
            if psi < next_psi {
                psi = next_psi;
            }
            trace.push(ProjectionStepTrace {
                step,
                psi,
                eta: used_eta,
                kl_old_new,
                step_l12,
            });
            if step_l12 <= cfg.tol_step_l12 {
                converged = true;
                break;
            }
            eta = (used_eta * 1.15).min(cfg.eta_max);
        } else {
            converged = true;
            break;
        }
    }

    // Décodage parallèle par argmax sur chaque simplexe \ell \in 0..L'
    let mut witness_bytes = Vec::with_capacity(l);
    let mut conf_sum = 0.0;
    for pos in 0..l {
        let row = &y[pos * BYTE_VOCAB..(pos + 1) * BYTE_VOCAB];
        let mut best_b = 0usize;
        let mut best_p = -1.0f64;
        for (b, &p) in row.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best_b = b;
            }
        }
        witness_bytes.push(best_b as u8);
        conf_sum += best_p;
    }

    ProjectionResult {
        soft_bytes: y,
        witness_bytes,
        mean_confidence: conf_sum / (l as f64),
        final_psi: psi,
        trace,
        converged,
    }
}

/// Mise à jour exponentielle miroir numériquement stable en espace log.
fn mirror_step(y: &[f64], grad: &[f64], out_len: usize, eta: f64, out: &mut [f64]) {
    const FLOOR: f64 = 1e-300;
    for pos in 0..out_len {
        let off = pos * BYTE_VOCAB;
        let mut max_log = f64::NEG_INFINITY;
        for b in 0..BYTE_VOCAB {
            let v = y[off + b].max(FLOOR).ln() + eta * grad[off + b];
            out[off + b] = v;
            if v > max_log {
                max_log = v;
            }
        }
        let mut sum = 0.0;
        for b in 0..BYTE_VOCAB {
            let ex = (out[off + b] - max_log).exp();
            out[off + b] = ex;
            sum += ex;
        }
        let inv_sum = 1.0 / sum.max(FLOOR);
        for b in 0..BYTE_VOCAB {
            out[off + b] *= inv_sum;
        }
    }
}

/// Divergence KL produit D_KL(P || Q) = sum_{\ell, b} P_{\ell, b} log(P_{\ell, b} / Q_{\ell, b}).
pub fn kl_divergence_simplex(p: &[f64], q: &[f64]) -> f64 {
    const FLOOR: f64 = 1e-300;
    let mut kl = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > FLOOR {
            kl += pi * (pi.ln() - qi.max(FLOOR).ln());
        }
    }
    kl.max(0.0)
}

/// Norme produit ||Y - Z||_{1,2} = ( sum_\ell ||Y_\ell - Z_\ell||_1^2 )^{1/2}.
pub fn norm_l12_diff(a: &[f64], b: &[f64], out_len: usize) -> f64 {
    let mut sum_sq = 0.0;
    for pos in 0..out_len {
        let off = pos * BYTE_VOCAB;
        let mut l1 = 0.0;
        for k in 0..BYTE_VOCAB {
            l1 += (a[off + k] - b[off + k]).abs();
        }
        sum_sq += l1 * l1;
    }
    sum_sq.sqrt()
}

fn compute_log_softmax(h_node: &[f64], w_out: &[f64], b_out: &[f64], out: &mut [f64]) {
    let h = h_node.len();
    let mut max_l = f64::NEG_INFINITY;
    for b in 0..BYTE_VOCAB {
        let row = &w_out[b * h..(b + 1) * h];
        let mut acc = b_out[b];
        for r in 0..h {
            acc += row[r] * h_node[r];
        }
        out[b] = acc;
        if acc > max_l {
            max_l = acc;
        }
    }
    let mut sum_exp = 0.0;
    for &v in out.iter() {
        sum_exp += (v - max_l).exp();
    }
    let lse = max_l + sum_exp.ln();
    for v in out.iter_mut() {
        *v -= lse;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::flow::{run_abductive_flow, FlowConfig};
    use crate::core::phi_sharp::{EffectSummary, PhiSharpConfig, StabilityProbes};
    use crate::ho_ptgrs::rng::Rng;

    #[test]
    fn test_projection_gradient_exact_finite_diff() {
        let mut rng = Rng::from_seed(321);
        let n = 4;
        let d = 2;
        let h = 6;
        let params = ModelParams::random_init(n, d, h, &mut rng);
        let eff_bytes: Vec<u8> = b"abductive meaning trajectory in bytes".to_vec();
        let eff = EffectSummary::from_bytes(&eff_bytes, n);
        let probes = StabilityProbes::build(n, 2, 0.05, 0.1, 11);
        let phi_cfg = PhiSharpConfig::default();
        let flow_cfg = FlowConfig {
            max_steps: 5,
            ..FlowConfig::default()
        };
        let traj = run_abductive_flow(&params, &eff, &probes, &phi_cfg, &flow_cfg, None);

        let proj_cfg = ProjectionConfig {
            out_len: 4,
            ..ProjectionConfig::default()
        };
        let field = ProjectionField::build(
            &params,
            &traj.r_final,
            &traj.final_eval,
            &eff_bytes,
            &proj_cfg,
        );

        // Point intérieur aléatoire sur \Delta_{255}^4
        let mut y = vec![0.0; 4 * BYTE_VOCAB];
        for pos in 0..4 {
            let mut sum = 0.0;
            for b in 0..BYTE_VOCAB {
                let v = rng.gen_f64() + 0.05;
                y[pos * BYTE_VOCAB + b] = v;
                sum += v;
            }
            for b in 0..BYTE_VOCAB {
                y[pos * BYTE_VOCAB + b] /= sum;
            }
        }

        let mut grad = vec![0.0; y.len()];
        let _ = field.eval_and_grad(&y, Some(&mut grad));

        let eps = 1e-6;
        // Vérification sur un échantillon couvrant toutes les positions et plusieurs octets
        for pos in 0..4 {
            for b in [0usize, 13, 65, 97, 100, 200, 255] {
                let idx = pos * BYTE_VOCAB + b;
                let mut yp = y.clone();
                let mut ym = y.clone();
                yp[idx] += eps;
                ym[idx] -= eps;
                let vp = field.eval_and_grad(&yp, None);
                let vm = field.eval_and_grad(&ym, None);
                let fd = (vp - vm) / (2.0 * eps);
                let an = grad[idx];
                assert!(
                    (fd - an).abs() < 1e-6,
                    "Grad projection mismatch ({pos},{b}): fd={fd}, an={an}"
                );
            }
        }
    }

    #[test]
    fn test_theorem_5_mirror_ascent_monotonicity_and_simplex() {
        let mut rng = Rng::from_seed(909);
        let n = 5;
        let d = 2;
        let h = 6;
        let params = ModelParams::random_init(n, d, h, &mut rng);
        let eff_bytes = b"ambiguite: deux interpretations distinctes pour un meme effet";
        let eff = EffectSummary::from_bytes(eff_bytes, n);
        let probes = StabilityProbes::build(n, 2, 0.05, 0.1, 77);
        let phi_cfg = PhiSharpConfig::default();
        let flow_cfg = FlowConfig {
            max_steps: 6,
            ..FlowConfig::default()
        };
        let traj = run_abductive_flow(&params, &eff, &probes, &phi_cfg, &flow_cfg, None);

        let proj_cfg = ProjectionConfig {
            out_len: 12,
            max_steps: 40,
            ..ProjectionConfig::default()
        };
        let res = project_witness_mirror_ascent(
            &params,
            &traj.r_final,
            &traj.final_eval,
            eff_bytes,
            &proj_cfg,
        );

        assert!(res.trace.len() >= 2);
        assert!(res.is_monotone_psi(1e-10), "Violation de monotonicité Théorème 5");
        assert!(res.final_psi > res.trace[0].psi);
        assert_eq!(res.witness_bytes.len(), 12);

        // Vérifier que chaque ligne appartient exactement au simplexe \Delta_{255}
        for pos in 0..12 {
            let row = &res.soft_bytes[pos * BYTE_VOCAB..(pos + 1) * BYTE_VOCAB];
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
            assert!(row.iter().all(|&p| p >= 0.0));
        }
    }
}
