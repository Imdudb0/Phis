/// Matrice carrée dense N x N en stockage row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    pub n: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0; n * n],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] = v;
    }

    #[inline]
    pub fn add_assign(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] += v;
    }

    pub fn frob_dot(&self, other: &Mat) -> f64 {
        debug_assert_eq!(self.n, other.n);
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn frob_norm_sq(&self) -> f64 {
        self.frob_dot(self)
    }

    pub fn frob_norm(&self) -> f64 {
        self.frob_norm_sq().sqrt()
    }

    pub fn scale(&mut self, s: f64) {
        for v in &mut self.data {
            *v *= s;
        }
    }

    pub fn axpy(&mut self, alpha: f64, other: &Mat) {
        debug_assert_eq!(self.n, other.n);
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += alpha * *b;
        }
    }

    /// Projection sur l'algèbre de Lie so(N) : skew(X) = 0.5 * (X - X^T).
    pub fn skew(&self) -> Mat {
        let n = self.n;
        let mut out = Mat::zeros(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let v = 0.5 * (self.get(i, j) - self.get(j, i));
                out.set(i, j, v);
                out.set(j, i, -v);
            }
        }
        out
    }

    /// Produit matriciel C = A * B.
    pub fn matmul(&self, rhs: &Mat) -> Mat {
        let n = self.n;
        debug_assert_eq!(n, rhs.n);
        let mut out = Mat::zeros(n);
        for i in 0..n {
            let row_a = i * n;
            let row_c = i * n;
            for k in 0..n {
                let aik = self.data[row_a + k];
                if aik == 0.0 {
                    continue;
                }
                let row_b = k * n;
                for j in 0..n {
                    out.data[row_c + j] += aik * rhs.data[row_b + j];
                }
            }
        }
        out
    }

    /// Produit matriciel avec transposée gauche : A^T * B.
    pub fn matmul_t_left(&self, rhs: &Mat) -> Mat {
        let n = self.n;
        debug_assert_eq!(n, rhs.n);
        let mut out = Mat::zeros(n);
        for k in 0..n {
            let row_a = k * n;
            let row_b = k * n;
            for i in 0..n {
                let aki = self.data[row_a + i];
                if aki == 0.0 {
                    continue;
                }
                let row_c = i * n;
                for j in 0..n {
                    out.data[row_c + j] += aki * rhs.data[row_b + j];
                }
            }
        }
        out
    }

    /// Produit matriciel avec transposée droite : A * B^T.
    pub fn matmul_t_right(&self, rhs: &Mat) -> Mat {
        let n = self.n;
        debug_assert_eq!(n, rhs.n);
        let mut out = Mat::zeros(n);
        for i in 0..n {
            let row_a = i * n;
            for j in 0..n {
                let row_b = j * n;
                let mut acc = 0.0;
                for k in 0..n {
                    acc += self.data[row_a + k] * rhs.data[row_b + k];
                }
                out.data[i * n + j] = acc;
            }
        }
        out
    }

    /// Produit matrice-vecteur y = M * x.
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        debug_assert_eq!(n, x.len());
        let mut y = vec![0.0; n];
        for i in 0..n {
            let row = i * n;
            let mut acc = 0.0;
            for j in 0..n {
                acc += self.data[row + j] * x[j];
            }
            y[i] = acc;
        }
        y
    }

    /// Diagonalisation exacte par rotations de Jacobi d'une matrice symétrique M = V * diag(evals) * V^T.
    /// Retourne `(evals, V)` triés par ordre croissant des valeurs propres ; la colonne `k` de `V`
    /// est le vecteur propre unitaire associé à `evals[k]`.
    pub fn jacobi_eigen_sym(&self, max_sweeps: usize, tol: f64) -> (Vec<f64>, Mat) {
        let n = self.n;
        if n == 0 {
            return (Vec::new(), Mat::zeros(0));
        }
        if n == 1 {
            return (vec![self.data[0]], Mat::identity(1));
        }

        // Symétrisation stricte par sécurité numérique
        let mut a = Mat::zeros(n);
        for i in 0..n {
            a.set(i, i, self.get(i, i));
            for j in (i + 1)..n {
                let sym = 0.5 * (self.get(i, j) + self.get(j, i));
                a.set(i, j, sym);
                a.set(j, i, sym);
            }
        }
        let mut v = Mat::identity(n);
        let max_iters = max_sweeps * n * n;

        for _ in 0..max_iters {
            // Pivot maximal hors-diagonale
            let mut p = 0usize;
            let mut q = 1usize;
            let mut max_off = 0.0f64;
            for i in 0..n {
                let row = i * n;
                for j in (i + 1)..n {
                    let val = a.data[row + j].abs();
                    if val > max_off {
                        max_off = val;
                        p = i;
                        q = j;
                    }
                }
            }
            if max_off <= tol {
                break;
            }

            let apq = a.get(p, q);
            let app = a.get(p, p);
            let aqq = a.get(q, q);
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            a.set(p, p, app - t * apq);
            a.set(q, q, aqq + t * apq);
            a.set(p, q, 0.0);
            a.set(q, p, 0.0);

            for k in 0..n {
                if k != p && k != q {
                    let akp = a.get(k, p);
                    let akq = a.get(k, q);
                    let new_kp = c * akp - s * akq;
                    let new_kq = s * akp + c * akq;
                    a.set(k, p, new_kp);
                    a.set(p, k, new_kp);
                    a.set(k, q, new_kq);
                    a.set(q, k, new_kq);
                }
            }

            for k in 0..n {
                let vkp = v.get(k, p);
                let vkq = v.get(k, q);
                v.set(k, p, c * vkp - s * vkq);
                v.set(k, q, s * vkp + c * vkq);
            }
        }

        // Tri croissant des paires propres
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            a.get(i, i)
                .partial_cmp(&a.get(j, j))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let evals: Vec<f64> = order.iter().map(|&idx| a.get(idx, idx)).collect();
        let mut sorted_v = Mat::zeros(n);
        for (new_col, &old_col) in order.iter().enumerate() {
            for row in 0..n {
                sorted_v.set(row, new_col, v.get(row, old_col));
            }
        }
        (evals, sorted_v)
    }
}

/// Tenseur relationnel dirigé R \in R^{N x N x d}.
/// Stockage par canal (channel-first : `channels[c]` est une matrice N x N)
/// pour accélérer les produits matriciels, le commutateur [A, R] et les SVD par canal.
#[derive(Debug, Clone, PartialEq)]
pub struct RelTensor {
    pub n: usize,
    pub d: usize,
    pub channels: Vec<Mat>,
}

impl RelTensor {
    pub fn zeros(n: usize, d: usize) -> Self {
        Self {
            n,
            d,
            channels: (0..d).map(|_| Mat::zeros(n)).collect(),
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize, c: usize) -> f64 {
        self.channels[c].get(i, j)
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, c: usize, v: f64) {
        self.channels[c].set(i, j, v);
    }

    pub fn frob_dot(&self, other: &RelTensor) -> f64 {
        debug_assert_eq!(self.n, other.n);
        debug_assert_eq!(self.d, other.d);
        self.channels
            .iter()
            .zip(other.channels.iter())
            .map(|(a, b)| a.frob_dot(b))
            .sum()
    }

    pub fn frob_norm_sq(&self) -> f64 {
        self.frob_dot(self)
    }

    pub fn frob_norm(&self) -> f64 {
        self.frob_norm_sq().sqrt()
    }

    pub fn scale(&mut self, s: f64) {
        for ch in &mut self.channels {
            ch.scale(s);
        }
    }

    pub fn axpy(&mut self, alpha: f64, other: &RelTensor) {
        debug_assert_eq!(self.n, other.n);
        debug_assert_eq!(self.d, other.d);
        for (a, b) in self.channels.iter_mut().zip(other.channels.iter()) {
            a.axpy(alpha, b);
        }
    }

    /// Rétraction radiale sur la boule compacte B_K = { ||R||_F <= K }.
    pub fn clamp_to_ball(&mut self, radius_k: f64) {
        let norm = self.frob_norm();
        if norm > radius_k && norm > 0.0 {
            self.scale(radius_k / norm);
        }
    }

    /// Projection orthogonale d'un gradient G sur le cône tangent T_R B_K.
    /// Si ||R||_F >= K et <G, R>_F > 0, supprime la composante radiale sortante.
    pub fn project_tangent_ball(&self, grad: &RelTensor, radius_k: f64) -> RelTensor {
        let norm_sq = self.frob_norm_sq();
        let k_sq = radius_k * radius_k;
        let mut out = grad.clone();
        if norm_sq >= k_sq * (1.0 - 1e-12) && norm_sq > 0.0 {
            let dot = grad.frob_dot(self);
            if dot > 0.0 {
                out.axpy(-dot / norm_sq, self);
            }
        }
        out
    }

    /// Commutateur isospectral [A, R] = A*R - R*A appliqué sur chaque canal relationnel c.
    /// Pour tout A \in so(N), vérifie strictement <[A, R], R>_F = 0.
    pub fn commutator_with(&self, a_skew: &Mat) -> RelTensor {
        debug_assert_eq!(self.n, a_skew.n);
        let mut out = RelTensor::zeros(self.n, self.d);
        for c in 0..self.d {
            let ar = a_skew.matmul(&self.channels[c]);
            let ra = self.channels[c].matmul(a_skew);
            for idx in 0..(self.n * self.n) {
                out.channels[c].data[idx] = ar.data[idx] - ra.data[idx];
            }
        }
        out
    }

    /// Opérateur dual exact du commutateur dans so(N) (Théorème 4) :
    /// dual(G, R) = skew( sum_c (G_c * R_c^T - R_c^T * G_c) )
    /// satisfaisant <[A, R], G>_F = <A, dual(G, R)>_F pour tout A \in so(N).
    pub fn skew_commutator_dual(&self, grad: &RelTensor) -> Mat {
        debug_assert_eq!(self.n, grad.n);
        debug_assert_eq!(self.d, grad.d);
        let mut acc = Mat::zeros(self.n);
        for c in 0..self.d {
            let grt = grad.channels[c].matmul_t_right(&self.channels[c]);
            let rtg = self.channels[c].matmul_t_left(&grad.channels[c]);
            for idx in 0..(self.n * self.n) {
                acc.data[idx] += grt.data[idx] - rtg.data[idx];
            }
        }
        acc.skew()
    }

    /// Régularisation d'abstraction spectrale C^\infty :
    /// R_abs^\eps(R) = sum_c [ Tr(sqrt(R_c^T R_c + eps^2 I)) - sqrt(||R_c||_F^2 + eps^2) ]
    /// Retourne `(valeur, gradient_exact_par_rapport_a_R)`.
    pub fn spectral_abstraction_reg(&self, eps: f64) -> (f64, RelTensor) {
        let eps_sq = eps * eps;
        let mut total_val = 0.0;
        let mut grad = RelTensor::zeros(self.n, self.d);

        for c in 0..self.d {
            let m = &self.channels[c];
            let s = m.matmul_t_left(m); // S = M^T M (symétrique PSD)
            let (evals, v) = s.jacobi_eigen_sym(50, 1e-13);

            // Terme nucléaire lissé : sum_i sqrt(max(evals[i], 0) + eps^2)
            let mut nuc_smooth = 0.0;
            let mut inv_sqrt = vec![0.0; self.n];
            for i in 0..self.n {
                let sq = (evals[i].max(0.0) + eps_sq).sqrt();
                nuc_smooth += sq;
                inv_sqrt[i] = 1.0 / sq;
            }

            // (M^T M + eps^2 I)^{-1/2} = V * diag(inv_sqrt) * V^T
            let mut inv_mat = Mat::zeros(self.n);
            for i in 0..self.n {
                for j in i..self.n {
                    let mut sum = 0.0;
                    for k in 0..self.n {
                        sum += v.get(i, k) * inv_sqrt[k] * v.get(j, k);
                    }
                    inv_mat.set(i, j, sum);
                    inv_mat.set(j, i, sum);
                }
            }

            // Gradient nucléaire exact : M * (M^T M + eps^2 I)^{-1/2}
            let grad_nuc = m.matmul(&inv_mat);

            // Terme Frobenius lissé : sqrt(||M||_F^2 + eps^2)
            let frob_smooth = (m.frob_norm_sq() + eps_sq).sqrt();
            total_val += nuc_smooth - frob_smooth;

            let inv_frob = 1.0 / frob_smooth;
            for idx in 0..(self.n * self.n) {
                grad.channels[c].data[idx] = grad_nuc.data[idx] - m.data[idx] * inv_frob;
            }
        }

        (total_val, grad)
    }

    /// Copie avec ablation stricte de l'arête dirigée (i, j) sur tous les canaux : R^{\setminus(i,j)}.
    pub fn with_ablated_edge(&self, i: usize, j: usize) -> RelTensor {
        let mut copy = self.clone();
        for c in 0..self.d {
            copy.set(i, j, c, 0.0);
        }
        copy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rel(n: usize, d: usize) -> RelTensor {
        let mut r = RelTensor::zeros(n, d);
        let mut x: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || -> f64 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        for c in 0..d {
            for i in 0..n {
                for j in 0..n {
                    r.set(i, j, c, next());
                }
            }
        }
        r
    }

    #[test]
    fn test_commutator_orthogonality_and_duality_theorem_4() {
        let r = sample_rel(6, 3);
        let g = sample_rel(6, 3);
        let mut raw_a = Mat::zeros(6);
        for i in 0..6 {
            for j in 0..6 {
                raw_a.set(i, j, ((i * 7 + j * 13) as f64).sin());
            }
        }
        let a = raw_a.skew();

        let comm = r.commutator_with(&a);
        // Orthogonalité radiale exacte : <[A, R], R>_F = 0
        assert!(comm.frob_dot(&r).abs() < 1e-12);

        // Identité duale (*) : <[A, R], G>_F = <A, dual(G, R)>_F
        let lhs = comm.frob_dot(&g);
        let dual = r.skew_commutator_dual(&g);
        let rhs = a.frob_dot(&dual);
        assert!((lhs - rhs).abs() < 1e-12, "lhs={lhs}, rhs={rhs}");
    }

    #[test]
    fn test_spectral_abstraction_gradient_finite_diff() {
        let r = sample_rel(5, 2);
        let eps = 1e-2;
        let (_, analytic) = r.spectral_abstraction_reg(eps);
        let h = 1e-6;

        for c in 0..r.d {
            for i in 0..r.n {
                for j in 0..r.n {
                    let mut rp = r.clone();
                    let mut rm = r.clone();
                    rp.set(i, j, c, r.get(i, j, c) + h);
                    rm.set(i, j, c, r.get(i, j, c) - h);
                    let (vp, _) = rp.spectral_abstraction_reg(eps);
                    let (vm, _) = rm.spectral_abstraction_reg(eps);
                    let fd = (vp - vm) / (2.0 * h);
                    let an = analytic.get(i, j, c);
                    assert!(
                        (fd - an).abs() < 1e-6,
                        "Mismatch at ({i},{j},{c}): fd={fd}, analytic={an}"
                    );
                }
            }
        }
    }
}
