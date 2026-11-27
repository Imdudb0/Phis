use crate::core::phi_sharp::{
    spatial_assignment, EffectSummary, ModelParams, PhiSharpEval, BI_DIM, REL_DIM,
};
use crate::core::tensor::{Mat, RelTensor};
use std::collections::BTreeMap;

/// Configuration de la projection référentielle par montée miroir sur \Delta_{|A(E)|-1}^{L'}.
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
    /// Poids relatif du couplage transitions inter-positions dans \Psi.
    pub lambda_bi: f64,
    /// Poids d'ancrage sur la distribution empirique de l'alphabet observé A(E).
    pub gamma_alphabet: f64,
    /// Pénalité anti-copie positionnelle triviale de `perceived.eff`.
    pub gamma_anticopy: f64,
    /// Poids de pureté quadratique (Tsallis, gradient borné sur le simplexe).
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

/// Résultat complet de la projection référentielle non-autorégressive sur A(E).
#[derive(Debug, Clone)]
pub struct ProjectionResult {
    /// Alphabet observé A(E) (taille k_uni).
    pub alphabet: Vec<u8>,
    pub k_uni: usize,
    /// Distribution finale sur \Delta_{k_uni - 1}^{L'} (row-major L' x k_uni).
    pub soft_bytes: Vec<f64>,
    /// Séquence témoin décodée par argmax parallèle dans A(E).
    pub witness_bytes: Vec<u8>,
    /// Confiance moyenne max_a Y_{\ell, a} sur les L' positions.
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

/// Champs précalculés dépendant uniquement de (R^*, E), strictement S_256-équivariants.
struct ProjectionField {
    out_len: usize,
    k_uni: usize,
    w_uni: f64,
    w_bi: f64,
    /// Potentiel linéaire c_{\ell, a} (row-major L' x k_uni).
    linear_pot: Vec<f64>,
    /// Paires de transitions observées (u, v) dans A(E) x A(E).
    obs_pairs: Vec<(usize, usize)>,
    /// Scores d'attraction >= 0 pour chaque position \ell \in 1..L' et chaque paire observée.
    /// Stockage row-major (L' - 1) x obs_pairs.len().
    bi_pot: Vec<f64>,
    /// Préfixe de `perceived.eff` exprimé dans les indices de A(E) pour l'anti-copie.
    copy_ref_alpha: Vec<usize>,
    gamma_anticopy: f64,
    tau_sharp: f64,
}

impl ProjectionField {
    fn build(
        params: &ModelParams,
        r_star: &RelTensor,
        eval_star: &PhiSharpEval,
        effect: &EffectSummary,
        perceived_eff: &[u8],
        cfg: &ProjectionConfig,
    ) -> Self {
        let n = params.n;
        let d = params.d;
        let h = params.h;
        let k_uni = effect.k_uni;
        let out_len = cfg.out_len.max(1);
        let inv_sqrt_n = 1.0 / (n as f64).sqrt();

        // 1. Calcul des états nodaux h_i(R^*)
        let comp: Vec<Mat> = (0..d)
            .map(|c| r_star.channels[c].matmul(&r_star.channels[c]))
            .collect();

        let mut h_nodes = vec![0.0; n * h];
        for i in 0..n {
            for j in 0..n {
                for r_idx in 0..h {
                    let mut z = params.b_edge[r_idx];
                    let w_d = &params.w_dir[r_idx * d..(r_idx + 1) * d];
                    let w_c = &params.w_comp[r_idx * d..(r_idx + 1) * d];
                    for c in 0..d {
                        z += w_d[c] * r_star.get(i, j, c) + w_c[c] * comp[c].get(i, j);
                    }
                    h_nodes[i * h + r_idx] += z.tanh();
                }
            }
            for r_idx in 0..h {
                h_nodes[i * h + r_idx] *= inv_sqrt_n;
            }
        }

        // 2. Log-distributions unigrammes log p_i^{uni}(a | R^*, E) sur A(E) via W_rel_uni
        let heads = params.precompute_heads(effect);
        let mut log_p_uni = vec![0.0; n * k_uni];
        for i in 0..n {
            let hi = &h_nodes[i * h..(i + 1) * h];
            compute_bilinear_log_softmax(
                hi,
                &heads.u_uni,
                &heads.u0_uni,
                &mut log_p_uni[i * k_uni..(i + 1) * k_uni],
            );
        }

        // 3. Extraction S_256-équivariante des paires de transitions observées (u, v) et scores W_rel_bi
        let has_bi = out_len >= 2 && cfg.lambda_bi > 0.0 && perceived_eff.len() >= 2;
        let (obs_pairs, bi_node_scores) = if has_bi {
            build_observed_transition_scores(params, &h_nodes, effect, perceived_eff)
        } else {
            (Vec::new(), Vec::new())
        };
        let num_obs = obs_pairs.len();
        let active_bi = has_bi && num_obs > 0;

        // 4. Importance causale des nœuds (degré causal symétrisé + énergie de Fiedler v_2^2)
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

        // 5. Prior empirique doux sur A(E) (strictement S_256-équivariant)
        let mut counts = vec![0usize; k_uni];
        for &b in perceived_eff {
            if let Some(a) = effect.byte_to_alpha[b as usize] {
                counts[a] += 1;
            }
        }
        let total = (perceived_eff.len().max(1)) as f64;
        let mut log_q_alph = vec![0.0; k_uni];
        for a in 0..k_uni {
            let p_unif = 0.7 / (k_uni as f64);
            let p_emp = 0.3 * (counts[a] as f64) / total;
            log_q_alph[a] = (p_unif + p_emp).max(1e-12).ln();
        }

        let (w_uni, w_bi) = if active_bi {
            let denom = 1.0 + cfg.lambda_bi;
            (1.0 / denom, cfg.lambda_bi / denom)
        } else {
            (1.0, 0.0)
        };

        // 6. Assemblage des potentiels spatialisés par position \ell \in 0..L'
        let inv_l = 1.0 / (out_len as f64);
        let mut linear_pot = vec![0.0; out_len * k_uni];
        let mut bi_pot = if active_bi {
            vec![0.0; (out_len - 1) * num_obs]
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

            let lin_row = &mut linear_pot[pos * k_uni..(pos + 1) * k_uni];
            for a in 0..k_uni {
                let mut u_val = 0.0;
                for i in 0..n {
                    u_val += assign[i] * log_p_uni[i * k_uni + a];
                }
                lin_row[a] = inv_l * (w_uni * u_val + cfg.gamma_alphabet * log_q_alph[a]);
            }

            if active_bi && pos >= 1 {
                let bi_row = &mut bi_pot[(pos - 1) * num_obs..pos * num_obs];
                for e_idx in 0..num_obs {
                    let mut b_val = 0.0;
                    for i in 0..n {
                        b_val += assign[i] * bi_node_scores[i * num_obs + e_idx];
                    }
                    bi_row[e_idx] = b_val;
                }
            }
        }

        let copy_len = out_len.min(perceived_eff.len());
        let copy_ref_alpha: Vec<usize> = perceived_eff[..copy_len]
            .iter()
            .filter_map(|&b| effect.byte_to_alpha[b as usize])
            .collect();

        Self {
            out_len,
            k_uni,
            w_uni,
            w_bi,
            linear_pot,
            obs_pairs,
            bi_pot,
            copy_ref_alpha,
            gamma_anticopy: cfg.gamma_anticopy,
            tau_sharp: cfg.tau_sharp,
        }
    }

    /// Évalue exactement \Psi(Y; R^*, E) et (optionnellement) \nabla_Y \Psi(Y).
    fn eval_and_grad(&self, y: &[f64], grad_out: Option<&mut [f64]>) -> f64 {
        let l = self.out_len;
        let k = self.k_uni;
        let inv_l = 1.0 / (l as f64);
        let mut psi = 0.0;

        let mut grad = grad_out;
        if let Some(g) = grad.as_deref_mut() {
            g.fill(0.0);
        }

        // 1. Terme linéaire + pureté quadratique Tsallis
        for pos in 0..l {
            let off = pos * k;
            let y_row = &y[off..off + k];
            let c_row = &self.linear_pot[off..off + k];
            for a in 0..k {
                let ya = y_row[a];
                psi += ya * c_row[a] + 0.5 * self.tau_sharp * inv_l * ya * ya;
                if let Some(g) = grad.as_deref_mut() {
                    g[off + a] += c_row[a] + self.tau_sharp * inv_l * ya;
                }
            }
        }

        // 2. Pénalité anti-copie positionnelle triviale
        if self.gamma_anticopy > 0.0 {
            let coef = self.gamma_anticopy * inv_l;
            for (pos, &a) in self.copy_ref_alpha.iter().enumerate() {
                let idx = pos * k + a;
                let ya = y[idx];
                psi -= 0.5 * coef * ya * ya;
                if let Some(g) = grad.as_deref_mut() {
                    g[idx] -= coef * ya;
                }
            }
        }

        // 3. Couplage bilinéaire creux sur les transitions observées (u, v) \in A(E) x A(E)
        let num_obs = self.obs_pairs.len();
        if self.w_bi > 0.0 && l >= 2 && num_obs > 0 {
            let bi_scale = self.w_bi / ((l - 1) as f64);
            for pos in 1..l {
                let prev_off = (pos - 1) * k;
                let curr_off = pos * k;
                let b_row = &self.bi_pot[(pos - 1) * num_obs..pos * num_obs];
                for (e_idx, &(u, v)) in self.obs_pairs.iter().enumerate() {
                    let w = bi_scale * b_row[e_idx];
                    let yu = y[prev_off + u];
                    let yv = y[curr_off + v];
                    psi += yu * yv * w;
                    if let Some(g) = grad.as_deref_mut() {
                        g[prev_off + u] += yv * w;
                        g[curr_off + v] += yu * w;
                    }
                }
            }
        }

        psi
    }
}

/// Calcule les scores d'attraction S_256-équivariants >= 0 de chaque transition observée (u, v)
/// pour chaque nœud i, via l'opérateur relationnel W_rel_bi / b_rel_bi.
fn build_observed_transition_scores(
    params: &ModelParams,
    h_nodes: &[f64],
    effect: &EffectSummary,
    bytes: &[u8],
) -> (Vec<(usize, usize)>, Vec<f64>) {
    let n = params.n;
    let h = params.h;
    let k_uni = effect.k_uni;
    let len = bytes.len();
    let ln_len1 = ((len as f64) + 1.0).ln().max(1e-12);

    let mut counts = vec![0usize; k_uni];
    let mut out_totals = vec![0usize; k_uni];
    let mut in_totals = vec![0usize; k_uni];
    let mut bi_counts: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut bi_pos_sum: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    let mut bi_pos_sq: BTreeMap<(usize, usize), f64> = BTreeMap::new();

    for &b in bytes {
        counts[effect.byte_to_alpha[b as usize].unwrap()] += 1;
    }
    for pos in 1..len {
        let u = effect.byte_to_alpha[bytes[pos - 1] as usize].unwrap();
        let v = effect.byte_to_alpha[bytes[pos] as usize].unwrap();
        out_totals[u] += 1;
        in_totals[v] += 1;
        *bi_counts.entry((u, v)).or_insert(0) += 1;
        let x = ((pos as f64) + 0.5) / (len as f64);
        *bi_pos_sum.entry((u, v)).or_insert(0.0) += x;
        *bi_pos_sq.entry((u, v)).or_insert(0.0) += x * x;
    }

    let obs_pairs: Vec<(usize, usize)> = bi_counts.keys().copied().collect();
    let num_obs = obs_pairs.len();
    if num_obs == 0 {
        return (obs_pairs, Vec::new());
    }

    // Projection relationnelle des transitions observées par W_rel_bi et b_rel_bi
    let mut u_obs = vec![0.0; num_obs * h];
    let mut u0_obs = vec![0.0; num_obs];

    for (idx, &(u, v)) in obs_pairs.iter().enumerate() {
        let c_uv = bi_counts[&(u, v)] as f64;
        let c_vu = bi_counts.get(&(v, u)).copied().unwrap_or(0) as f64;
        let c_u_tot = (counts[u] as f64).max(1.0);
        let c_v_tot = (counts[v] as f64).max(1.0);
        let c_u_out = (out_totals[u] as f64).max(1.0);
        let c_v_in = (in_totals[v] as f64).max(1.0);

        let g0 = (c_uv + 1.0).ln() / ln_len1;
        let pmi = ((c_uv * (len as f64)) / (c_u_tot * c_v_tot))
            .max(1e-12)
            .ln();
        let g1 = (pmi / 3.0).tanh();
        let g2 = c_uv / c_u_out;
        let g3 = c_uv / c_v_in;
        let g4 = c_vu / (c_uv + c_vu);
        let mean_p = bi_pos_sum[&(u, v)] / c_uv;
        let mean_sq = bi_pos_sq[&(u, v)] / c_uv;
        let std_p = (mean_sq - mean_p * mean_p).max(0.0).sqrt();
        let g7 = if u == v { 1.0 } else { 0.0 };
        let g8 = effect.rel_feats[v * REL_DIM + 3] - effect.rel_feats[u * REL_DIM + 3];
        let g9 = effect.rel_feats[v * REL_DIM] - effect.rel_feats[u * REL_DIM];

        let feat = [g0, g1, g2, g3, g4, mean_p, std_p, g7, g8, g9];

        let mut b0 = 0.0;
        for m in 0..BI_DIM {
            b0 += params.b_rel_bi[m] * feat[m];
        }
        u0_obs[idx] = b0;

        for r in 0..h {
            let w_row = &params.w_rel_bi[r * BI_DIM..(r + 1) * BI_DIM];
            let mut acc = 0.0;
            for m in 0..BI_DIM {
                acc += w_row[m] * feat[m];
            }
            u_obs[idx * h + r] = acc;
        }
    }

    // Pour chaque nœud i : log-softmax sur obs_pairs puis décalage positif >= 0.5
    // afin que toute transition observée cohérente reste strictement préférée aux transitions absentes (0).
    let mut node_scores = vec![0.0; n * num_obs];
    for i in 0..n {
        let hi = &h_nodes[i * h..(i + 1) * h];
        let row = &mut node_scores[i * num_obs..(i + 1) * num_obs];
        compute_bilinear_log_softmax(hi, &u_obs, &u0_obs, row);
        let min_lp = row.iter().copied().fold(f64::INFINITY, f64::min);
        for v in row.iter_mut() {
            *v = (*v - min_lp) + 0.5;
        }
    }

    (obs_pairs, node_scores)
}

/// Exécute la montée miroir parallèle sur \Delta_{|A(E)|-1}^{L'} (Théorème 5).
pub fn project_witness_mirror_ascent(
    params: &ModelParams,
    r_star: &RelTensor,
    eval_star: &PhiSharpEval,
    perceived_eff: &[u8],
    cfg: &ProjectionConfig,
) -> ProjectionResult {
    let l = cfg.out_len.max(1);
    let effect = EffectSummary::from_bytes(perceived_eff, params.n);
    let k = effect.k_uni;

    if k == 0 {
        return ProjectionResult {
            alphabet: Vec::new(),
            k_uni: 0,
            soft_bytes: Vec::new(),
            witness_bytes: Vec::new(),
            mean_confidence: 0.0,
            final_psi: 0.0,
            trace: Vec::new(),
            converged: true,
        };
    }

    let field = ProjectionField::build(params, r_star, eval_star, &effect, perceived_eff, cfg);

    // Centre barycentrique uniforme du produit de simplexes \Delta_{k-1}^{L'}
    let mut y = vec![1.0 / (k as f64); l * k];
    let mut y_cand = vec![0.0; l * k];
    let mut grad = vec![0.0; l * k];

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
            mirror_step(&y, &grad, l, k, trial_eta, &mut y_cand);
            let cand_psi = field.eval_and_grad(&y_cand, None);
            let kl_old_new = kl_divergence_simplex(&y, &y_cand);

            // Condition exacte du Théorème 5.1 :
            // \Psi(Y^{(k+1)}) - \Psi(Y^{(k)}) >= (1 / \eta_Y) * D_KL(Y^{(k)} || Y^{(k+1)})
            if cand_psi + 1e-12 >= psi + kl_old_new / trial_eta {
                let l12 = norm_l12_diff(&y, &y_cand, l, k);
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

    // Décodage parallèle par argmax sur chaque simplexe \ell \in 0..L' dans A(E)
    let mut witness_bytes = Vec::with_capacity(l);
    let mut conf_sum = 0.0;
    for pos in 0..l {
        let row = &y[pos * k..(pos + 1) * k];
        let mut best_a = 0usize;
        let mut best_p = -1.0f64;
        for (a, &p) in row.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best_a = a;
            }
        }
        witness_bytes.push(effect.alphabet[best_a]);
        conf_sum += best_p;
    }

    ProjectionResult {
        alphabet: effect.alphabet,
        k_uni: k,
        soft_bytes: y,
        witness_bytes,
        mean_confidence: conf_sum / (l as f64),
        final_psi: psi,
        trace,
        converged,
    }
}

/// Mise à jour exponentielle miroir numériquement stable en espace log sur \Delta_{k-1}^{L'}.
fn mirror_step(y: &[f64], grad: &[f64], out_len: usize, k: usize, eta: f64, out: &mut [f64]) {
    const FLOOR: f64 = 1e-300;
    for pos in 0..out_len {
        let off = pos * k;
        let mut max_log = f64::NEG_INFINITY;
        for a in 0..k {
            let v = y[off + a].max(FLOOR).ln() + eta * grad[off + a];
            out[off + a] = v;
            if v > max_log {
                max_log = v;
            }
        }
        let mut sum = 0.0;
        for a in 0..k {
            let ex = (out[off + a] - max_log).exp();
            out[off + a] = ex;
            sum += ex;
        }
        let inv_sum = 1.0 / sum.max(FLOOR);
        for a in 0..k {
            out[off + a] *= inv_sum;
        }
    }
}

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

pub fn norm_l12_diff(a: &[f64], b: &[f64], out_len: usize, k: usize) -> f64 {
    let mut sum_sq = 0.0;
    for pos in 0..out_len {
        let off = pos * k;
        let mut l1 = 0.0;
        for j in 0..k {
            l1 += (a[off + j] - b[off + j]).abs();
        }
        sum_sq += l1 * l1;
    }
    sum_sq.sqrt()
}

fn compute_bilinear_log_softmax(h_node: &[f64], u_mat: &[f64], u0: &[f64], out: &mut [f64]) {
    let k = u0.len();
    let h = h_node.len();
    let mut max_l = f64::NEG_INFINITY;
    for a in 0..k {
        let row = &u_mat[a * h..(a + 1) * h];
        let mut acc = u0[a];
        for r in 0..h {
            acc += row[r] * h_node[r];
        }
        out[a] = acc;
        if acc > max_l {
            max_l = acc;
        }
    }
    let mut sum_exp = 0.0;
    for &v in out.iter().take(k) {
        sum_exp += (v - max_l).exp();
    }
    let lse = max_l + sum_exp.ln();
    for v in out.iter_mut().take(k) {
        *v -= lse;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::flow::{run_abductive_flow, FlowConfig};
    use crate::core::phi_sharp::{PhiSharpConfig, StabilityProbes};
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
            &eff,
            &eff_bytes,
            &proj_cfg,
        );

        let k = eff.k_uni;
        let mut y = vec![0.0; 4 * k];
        for pos in 0..4 {
            let mut sum = 0.0;
            for a in 0..k {
                let v = rng.gen_f64() + 0.05;
                y[pos * k + a] = v;
                sum += v;
            }
            for a in 0..k {
                y[pos * k + a] /= sum;
            }
        }

        let mut grad = vec![0.0; y.len()];
        let _ = field.eval_and_grad(&y, Some(&mut grad));

        let eps = 1e-6;
        for pos in 0..4 {
            for a in 0..k {
                let idx = pos * k + a;
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
                    "Grad projection mismatch ({pos},{a}): fd={fd}, an={an}"
                );
            }
        }
    }

    #[test]
    fn test_theorem_5_mirror_ascent_monotonicity_and_s256_equivariance() {
        let mut rng = Rng::from_seed(909);
        let n = 5;
        let d = 2;
        let h = 6;
        let params = ModelParams::random_init(n, d, h, &mut rng);
        let eff_bytes = b"ambiguite: deux interpretations distinctes pour un meme effet";

        // Permutation aléatoire \pi \in S_256
        let mut perm_rng = Rng::from_seed(424242);
        let mut perm: Vec<u8> = (0u8..=255).collect();
        perm_rng.shuffle(&mut perm);
        let perm_bytes: Vec<u8> = eff_bytes.iter().map(|&b| perm[b as usize]).collect();

        let eff1 = EffectSummary::from_bytes(eff_bytes, n);
        let eff2 = EffectSummary::from_bytes(&perm_bytes, n);
        let probes = StabilityProbes::build(n, 2, 0.05, 0.1, 77);
        let phi_cfg = PhiSharpConfig::default();
        let flow_cfg = FlowConfig {
            max_steps: 6,
            ..FlowConfig::default()
        };

        let traj1 = run_abductive_flow(&params, &eff1, &probes, &phi_cfg, &flow_cfg, None);
        let traj2 = run_abductive_flow(&params, &eff2, &probes, &phi_cfg, &flow_cfg, None);

        let proj_cfg = ProjectionConfig {
            out_len: 12,
            max_steps: 40,
            ..ProjectionConfig::default()
        };
        let res1 = project_witness_mirror_ascent(
            &params,
            &traj1.r_final,
            &traj1.final_eval,
            eff_bytes,
            &proj_cfg,
        );
        let res2 = project_witness_mirror_ascent(
            &params,
            &traj2.r_final,
            &traj2.final_eval,
            &perm_bytes,
            &proj_cfg,
        );

        assert!(res1.trace.len() >= 2);
        assert!(res1.is_monotone_psi(1e-10), "Violation de monotonicité Théorème 5");
        assert!(res1.final_psi > res1.trace[0].psi);
        assert_eq!(res1.witness_bytes.len(), 12);

        // Vérification stricte de l'équivariance S_256 : proj(\pi(E)) == \pi(proj(E))
        assert!((res1.final_psi - res2.final_psi).abs() < 1e-10);
        let mapped_w1: Vec<u8> = res1.witness_bytes.iter().map(|&b| perm[b as usize]).collect();
        assert_eq!(
            mapped_w1, res2.witness_bytes,
            "La projection miroir doit être strictement équivariante sous S_256"
        );

        // Vérifier que chaque ligne appartient au simplexe \Delta_{k-1}
        for pos in 0..12 {
            let row = &res1.soft_bytes[pos * res1.k_uni..(pos + 1) * res1.k_uni];
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
            assert!(row.iter().all(|&p| p >= 0.0));
        }
    }
}
