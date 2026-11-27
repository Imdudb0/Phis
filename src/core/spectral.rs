use crate::core::tensor::Mat;

/// Résultat spectral du Laplacien de sensibilité causale L_{\bar{w}}.
#[derive(Debug, Clone)]
pub struct FiedlerResult {
    /// Connectivité algébrique \lambda_2(L_{\bar{w}}) >= 0.
    pub lambda2: f64,
    /// Seconde valeur propre non triviale \lambda_3 (diagnostic de simplicité / gap spectral).
    pub lambda3: f64,
    /// Vecteur propre unitaire de Fiedler v_2 orthogonal à 1_N.
    pub v2: Vec<f64>,
    /// Résidu spectral ||L v_2 - \lambda_2 v_2||_2.
    pub residual_norm: f64,
    /// Nombre d'itérations de Lanczos effectuées.
    pub lanczos_iters: usize,
    /// Nombre d'itérations de raffinement par puissance inverse CG.
    pub inverse_power_iters: usize,
}

/// Laplacien symétrisé L_{\bar{w}} = D_{\bar{w}} - \bar{W}, avec \bar{w}_{ij} = 0.5*(w_{ij} + w_{ji}).
#[derive(Debug, Clone)]
pub struct CausalLaplacian {
    pub n: usize,
    pub deg: Vec<f64>,
    /// Poids symétrisés \bar{W} (diagonale nulle, valeurs >= 0).
    pub w_sym: Mat,
}

impl CausalLaplacian {
    /// Construit L_{\bar{w}} à partir d'une matrice de poids causaux dirigés W (w_{ij} >= 0).
    pub fn from_directed_weights(w_dir: &Mat) -> Self {
        let n = w_dir.n;
        let mut w_sym = Mat::zeros(n);
        let mut deg = vec![0.0; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let wij = w_dir.get(i, j).max(0.0);
                let wji = w_dir.get(j, i).max(0.0);
                let bar = 0.5 * (wij + wji);
                if bar > 0.0 {
                    w_sym.set(i, j, bar);
                    w_sym.set(j, i, bar);
                    deg[i] += bar;
                    deg[j] += bar;
                }
            }
        }
        Self { n, deg, w_sym }
    }

    /// Produit matrice-vecteur exact y = L_{\bar{w}} * x en O(N^2).
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        debug_assert_eq!(n, x.len());
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut acc = self.deg[i] * x[i];
            let row = i * n;
            for j in 0..n {
                let w = self.w_sym.data[row + j];
                if w != 0.0 {
                    acc -= w * x[j];
                }
            }
            y[i] = acc;
        }
        y
    }

    /// Calcule \lambda_2(L_{\bar{w}}) et v_2 par Lanczos redémarré sur 1_N^\perp,
    /// suivi d'un raffinement par puissance inverse (résolution de L z = v par CG déflaté).
    pub fn compute_fiedler(&self, max_krylov: usize, max_inv_power: usize) -> FiedlerResult {
        let n = self.n;
        if n <= 1 {
            return FiedlerResult {
                lambda2: 0.0,
                lambda3: 0.0,
                v2: vec![0.0; n],
                residual_norm: 0.0,
                lanczos_iters: 0,
                inverse_power_iters: 0,
            };
        }
        if n == 2 {
            let w01 = self.w_sym.get(0, 1);
            let s = std::f64::consts::FRAC_1_SQRT_2;
            return FiedlerResult {
                lambda2: 2.0 * w01,
                lambda3: 2.0 * w01,
                v2: vec![s, -s],
                residual_norm: 0.0,
                lanczos_iters: 1,
                inverse_power_iters: 0,
            };
        }

        // Cas dégénéré : graphe complètement vide
        if self.deg.iter().all(|&d| d <= 1e-15) {
            let mut v = deterministic_unit_perp_ones(n, 0xA5A5_1234_5678_90EF);
            normalize(&mut v);
            return FiedlerResult {
                lambda2: 0.0,
                lambda3: 0.0,
                v2: v,
                residual_norm: 0.0,
                lanczos_iters: 0,
                inverse_power_iters: 0,
            };
        }

        let m_max = max_krylov.clamp(2, n - 1);
        let mut basis: Vec<Vec<f64>> = Vec::with_capacity(m_max);
        let mut alpha: Vec<f64> = Vec::with_capacity(m_max);
        let mut beta: Vec<f64> = Vec::with_capacity(m_max);

        let mut seed_state: u64 = 0x9E37_79B9_7F4A_7C15;
        let q0 = deterministic_unit_perp_ones(n, next_seed(&mut seed_state));
        basis.push(q0);

        // 1. Construction de la base de Krylov par Lanczos déflaté (avec redémarrage si sous-espace invariant)
        for j in 0..m_max {
            let mut w = self.apply(&basis[j]);
            project_perp_ones(&mut w);

            if j > 0 && beta[j - 1] > 0.0 {
                let b_prev = beta[j - 1];
                let q_prev = &basis[j - 1];
                for i in 0..n {
                    w[i] -= b_prev * q_prev[i];
                }
            }

            let a_j = dot(&basis[j], &w);
            alpha.push(a_j);

            for i in 0..n {
                w[i] -= a_j * basis[j][i];
            }

            // Double re-orthogonalisation complète contre 1_N et la base Q_j
            reorthogonalize(&mut w, &basis);
            let mut b_j = norm2(&w);

            if j + 1 < m_max {
                if b_j < 1e-11 {
                    // Sous-espace invariant atteint (ex. composante connexe isolée) :
                    // redémarrage sur un nouveau vecteur orthogonal avec beta[j] = 0.0 (bloc diagonal).
                    if let Some(fresh) = fresh_orthogonal_vector(n, &basis, &mut seed_state) {
                        beta.push(0.0);
                        basis.push(fresh);
                    } else {
                        break;
                    }
                } else {
                    for x in &mut w {
                        *x /= b_j;
                    }
                    beta.push(b_j);
                    basis.push(w);
                }
            }
        }

        let m = basis.len();
        let mut t_mat = Mat::zeros(m);
        for i in 0..m {
            t_mat.set(i, i, alpha[i]);
            if i + 1 < m && i < beta.len() {
                t_mat.set(i, i + 1, beta[i]);
                t_mat.set(i + 1, i, beta[i]);
            }
        }

        let (ritz_vals, ritz_vecs) = t_mat.jacobi_eigen_sym(60, 1e-14);
        let mut lambda2 = ritz_vals[0].max(0.0);
        let lambda3 = if m >= 2 {
            ritz_vals[1].max(0.0)
        } else {
            lambda2
        };

        // Reconstruction du vecteur de Ritz v_2 = Q_m * y_0
        let mut v2 = vec![0.0; n];
        for col in 0..m {
            let coeff = ritz_vecs.get(col, 0);
            for i in 0..n {
                v2[i] += coeff * basis[col][i];
            }
        }
        project_perp_ones(&mut v2);
        normalize(&mut v2);

        let mut res = self.eigen_residual(&v2, lambda2);
        let mut inv_iters = 0usize;

        // 2. Raffinement par puissance inverse (L z = v_2 résolu par CG sur 1_N^\perp)
        // Uniquement si le graphe est connexe (\lambda_2 > 1e-9), où L est SPD sur 1_N^\perp.
        if lambda2 > 1e-9 && res > 1e-11 && max_inv_power > 0 {
            for _ in 0..max_inv_power {
                if res <= 1e-11 {
                    break;
                }
                if let Some(mut z) = self.solve_cg_perp_ones(&v2, 2 * n, 1e-12) {
                    project_perp_ones(&mut z);
                    if normalize(&mut z) > 1e-15 {
                        let lz = self.apply(&z);
                        let new_lam = dot(&z, &lz).max(0.0);
                        let new_res = residual_from_lv(&z, &lz, new_lam);
                        if new_lam <= lambda2 + 1e-12 && new_res <= res + 1e-12 {
                            v2 = z;
                            lambda2 = new_lam;
                            res = new_res;
                            inv_iters += 1;
                            continue;
                        }
                    }
                }
                break;
            }
        }

        // Fixation de signe canonique (premier élément de magnitude > 1e-9 positif) pour déterminisme
        if let Some(&first_sig) = v2.iter().find(|x| x.abs() > 1e-9) {
            if first_sig < 0.0 {
                for x in &mut v2 {
                    *x = -*x;
                }
            }
        }

        FiedlerResult {
            lambda2,
            lambda3,
            v2,
            residual_norm: res,
            lanczos_iters: m,
            inverse_power_iters: inv_iters,
        }
    }

    /// Gradient analytique exact de Hellmann-Feynman de \lambda_2 par rapport aux poids dirigés W :
    /// d(\lambda_2) / d(w_{ij}) = 0.5 * (v_{2,i} - v_{2,j})^2 pour i != j, et 0 sur la diagonale.
    pub fn grad_directed_weights(v2: &[f64]) -> Mat {
        let n = v2.len();
        let mut grad = Mat::zeros(n);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let diff = v2[i] - v2[j];
                    grad.set(i, j, 0.5 * diff * diff);
                }
            }
        }
        grad
    }

    fn eigen_residual(&self, v: &[f64], lam: f64) -> f64 {
        let lv = self.apply(v);
        residual_from_lv(v, &lv, lam)
    }

    /// Résout L x = b dans le sous-espace 1_N^\perp par Gradient Conjugué (L y est SPD si connexe).
    fn solve_cg_perp_ones(&self, b: &[f64], max_iter: usize, tol: f64) -> Option<Vec<f64>> {
        let n = self.n;
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        project_perp_ones(&mut r);
        let mut p = r.clone();
        let mut rs_old = dot(&r, &r);
        let b_norm = rs_old.sqrt();
        if b_norm <= 1e-15 {
            return None;
        }

        for _ in 0..max_iter {
            let mut ap = self.apply(&p);
            project_perp_ones(&mut ap);
            let denom = dot(&p, &ap);
            if denom <= 1e-15 {
                return None;
            }
            let alpha = rs_old / denom;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            project_perp_ones(&mut r);
            let rs_new = dot(&r, &r);
            if rs_new.sqrt() <= tol * b_norm {
                return Some(x);
            }
            let beta = rs_new / rs_old;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            project_perp_ones(&mut p);
            rs_old = rs_new;
        }
        Some(x)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: &mut [f64]) -> f64 {
    let n = norm2(a);
    if n > 0.0 {
        for x in a.iter_mut() {
            *x /= n;
        }
    }
    n
}

fn project_perp_ones(v: &mut [f64]) {
    let mean = v.iter().sum::<f64>() / (v.len() as f64);
    for x in v.iter_mut() {
        *x -= mean;
    }
}

fn residual_from_lv(v: &[f64], lv: &[f64], lam: f64) -> f64 {
    let mut sq = 0.0;
    for i in 0..v.len() {
        let d = lv[i] - lam * v[i];
        sq += d * d;
    }
    sq.sqrt()
}

fn reorthogonalize(w: &mut [f64], basis: &[Vec<f64>]) {
    for _ in 0..2 {
        project_perp_ones(w);
        for q in basis {
            let proj = dot(q, w);
            for i in 0..w.len() {
                w[i] -= proj * q[i];
            }
        }
    }
}

fn next_seed(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn deterministic_unit_perp_ones(n: usize, mut seed: u64) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let r = next_seed(&mut seed);
        let f = ((r >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0;
        v.push(f);
    }
    project_perp_ones(&mut v);
    normalize(&mut v);
    v
}

fn fresh_orthogonal_vector(n: usize, basis: &[Vec<f64>], seed_state: &mut u64) -> Option<Vec<f64>> {
    for _ in 0..8 {
        let mut cand = deterministic_unit_perp_ones(n, next_seed(seed_state));
        reorthogonalize(&mut cand, basis);
        if normalize(&mut cand) > 1e-10 {
            return Some(cand);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_graph_exact_fiedler_value() {
        // Pour un graphe chaîne P_N à poids symétriques unitaires (w_dir = 1.0 dans chaque sens),
        // \lambda_2(P_N) = 2 * (1 - cos(pi / N)).
        let n = 10;
        let mut w = Mat::zeros(n);
        for i in 0..(n - 1) {
            w.set(i, i + 1, 1.0);
            w.set(i + 1, i, 1.0);
        }
        let lap = CausalLaplacian::from_directed_weights(&w);
        let res = lap.compute_fiedler(32, 4);

        let expected = 2.0 * (1.0 - (std::f64::consts::PI / (n as f64)).cos());
        assert!(
            (res.lambda2 - expected).abs() < 1e-10,
            "got {}, expected {}",
            res.lambda2,
            expected
        );
        assert!(res.residual_norm < 1e-10);
    }

    #[test]
    fn test_hellmann_feynman_gradient_finite_diff() {
        // Graphe dirigé non symétrique quelconque (w_{ij} != w_{ji}) avec \lambda_2 simple
        let n = 7;
        let mut w = Mat::zeros(n);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let val = 0.2 + ((i * 5 + j * 11 + 3) % 17) as f64 * 0.07;
                    w.set(i, j, val);
                }
            }
        }
        // Renforcer deux clusters pour avoir un gap spectral net entre \lambda_2 et \lambda_3
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    w.add_assign(i, j, 2.0);
                }
            }
        }
        for i in 3..n {
            for j in 3..n {
                if i != j {
                    w.add_assign(i, j, 2.0);
                }
            }
        }

        let lap = CausalLaplacian::from_directed_weights(&w);
        let res = lap.compute_fiedler(32, 4);
        assert!(res.lambda3 - res.lambda2 > 1e-2, "lambda2 doit être simple");

        let grad = CausalLaplacian::grad_directed_weights(&res.v2);
        let h = 1e-6;

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mut wp = w.clone();
                let mut wm = w.clone();
                wp.add_assign(i, j, h);
                wm.add_assign(i, j, -h);

                let lp = CausalLaplacian::from_directed_weights(&wp).compute_fiedler(32, 4);
                let lm = CausalLaplacian::from_directed_weights(&wm).compute_fiedler(32, 4);
                let fd = (lp.lambda2 - lm.lambda2) / (2.0 * h);
                let an = grad.get(i, j);
                assert!(
                    (fd - an).abs() < 1e-6,
                    "Mismatch ({i},{j}): fd={fd}, analytic={an}"
                );
            }
        }
    }

    #[test]
    fn test_theorem_6_disconnected_bridge_gradient() {
        // Deux sous-graphes connexes disjoints E_1 = {0,1,2} et E_2 = {3,4,5} :
        // \lambda_2 = 0, et le gradient de Hellmann-Feynman est 0 à l'intérieur des blocs
        // et strictement positif UNIQUEMENT sur les ponts inter-blocs (Théorème 6).
        let n = 6;
        let mut w = Mat::zeros(n);
        for &(u, v) in &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)] {
            w.set(u, v, 1.5);
            w.set(v, u, 1.5);
        }

        let lap = CausalLaplacian::from_directed_weights(&w);
        let res = lap.compute_fiedler(32, 4);
        assert!(res.lambda2 < 1e-11, "lambda2 attendu ~0, obtenu {}", res.lambda2);

        let grad = CausalLaplacian::grad_directed_weights(&res.v2);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let same_block = (i < 3) == (j < 3);
                if same_block {
                    assert!(
                        grad.get(i, j) < 1e-10,
                        "Le gradient intra-bloc ({i},{j}) devrait être nul, obtenu {}",
                        grad.get(i, j)
                    );
                } else {
                    assert!(
                        grad.get(i, j) > 0.1,
                        "Le gradient inter-bloc ({i},{j}) devrait pousser à créer le pont méta-relationnel"
                    );
                }
            }
        }
    }
}
