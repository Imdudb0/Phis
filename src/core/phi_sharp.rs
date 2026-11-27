use crate::core::spectral::{CausalLaplacian, FiedlerResult};
use crate::core::tensor::{Mat, RelTensor};
use crate::ho_ptgrs::rng::Rng;

pub const BYTE_VOCAB: usize = 256;
/// Résidu de fond uniforme par octet : log(256).
pub const UNIFORM_BYTE_RESIDUAL: f64 = 5.545_177_444_479_562;

/// Hyperparamètres analytiques de \phi_\sharp(R, E).
#[derive(Debug, Clone)]
pub struct PhiSharpConfig {
    /// Poids \alpha du résidu abductif -\alpha * \bar{\rho}(R, E).
    pub alpha: f64,
    /// Poids \beta de la fragilité différentielle -\beta * V_stab(R, E).
    pub beta: f64,
    /// Poids relatif de la composante bigramme dans \bar{\rho} (normalisé dans [0, 1]).
    pub lambda_bi: f64,
    /// Rayon de lissage C^1 de la charnière \psi_\tau sur \Delta_{ij} et V_stab.
    pub hinge_tau: f64,
    /// Température du minimum lisse smin_\eps(\bar{\rho}_{abl}, \bar{\rho}_0).
    pub smin_eps: f64,
    /// Dimension maximale de Krylov pour Lanczos.
    pub max_krylov: usize,
    /// Itérations maximales de raffinement par puissance inverse CG.
    pub max_inv_power: usize,
}

impl Default for PhiSharpConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.25,
            lambda_bi: 0.5,
            hinge_tau: 0.05,
            smin_eps: 0.05,
            max_krylov: 32,
            max_inv_power: 4,
        }
    }
}

/// Résumé statistique exact de `perceived.eff` projeté sur la partition spatiale des N nœuds.
/// Compatible à la fois avec des octets bruts &[u8] et des distributions souples Y \in \Delta_{255}^L.
#[derive(Debug, Clone)]
pub struct EffectSummary {
    pub n: usize,
    pub len: usize,
    /// Distribution empirique unigramme pondérée Q_{i, b} (N x 256), somme totale = 1.
    pub uni: Vec<f64>,
    /// Masse spatiale par nœud q_mass[i] = sum_b Q_{i, b}.
    pub uni_mass: Vec<f64>,
    /// Distribution empirique des transitions bigrammes (nibbles 16x16 = 256 classes) T_{i, g}.
    pub bi: Vec<f64>,
    /// Masse bigramme par nœud t_mass[i] = sum_g T_{i, g}.
    pub bi_mass: Vec<f64>,
    /// Indique si des transitions bigrammes sont présentes (len >= 2).
    pub has_bigrams: bool,
}

impl EffectSummary {
    /// Construit les statistiques exactes depuis le flux brut d'octets `perceived.eff`.
    pub fn from_bytes(bytes: &[u8], n: usize) -> Self {
        let len = bytes.len();
        let mut uni = vec![0.0; n * BYTE_VOCAB];
        let mut uni_mass = vec![0.0; n];
        let mut bi = vec![0.0; n * BYTE_VOCAB];
        let mut bi_mass = vec![0.0; n];

        if len == 0 || n == 0 {
            return Self {
                n,
                len: 0,
                uni,
                uni_mass,
                bi,
                bi_mass,
                has_bigrams: false,
            };
        }

        let inv_l = 1.0 / (len as f64);
        let mut assign = vec![0.0; n];

        for (pos, &b) in bytes.iter().enumerate() {
            spatial_assignment(pos, len, n, &mut assign);
            let b_idx = b as usize;
            for i in 0..n {
                let w = assign[i] * inv_l;
                uni[i * BYTE_VOCAB + b_idx] += w;
                uni_mass[i] += w;
            }
        }

        let has_bigrams = len >= 2;
        if has_bigrams {
            let inv_bm = 1.0 / ((len - 1) as f64);
            for pos in 1..len {
                spatial_assignment(pos, len, n, &mut assign);
                // Classe de transition 16x16 couplant nibble bas(prev) et nibble bas(curr)
                // plus parité haute via XOR pour sensibilité à tous les bits.
                let g_idx = bigram_bucket(bytes[pos - 1], bytes[pos]);
                for i in 0..n {
                    let w = assign[i] * inv_bm;
                    bi[i * BYTE_VOCAB + g_idx] += w;
                    bi_mass[i] += w;
                }
            }
        }

        Self {
            n,
            len,
            uni,
            uni_mass,
            bi,
            bi_mass,
            has_bigrams,
        }
    }

    /// Construit les statistiques depuis une séquence souple Y \in \Delta_{255}^L
    /// (stockage row-major `y[pos * 256 + byte]`), pour la projection de la Partie 3.
    pub fn from_soft_bytes(y: &[f64], len: usize, n: usize) -> Self {
        debug_assert_eq!(y.len(), len * BYTE_VOCAB);
        let mut uni = vec![0.0; n * BYTE_VOCAB];
        let mut uni_mass = vec![0.0; n];
        let mut bi = vec![0.0; n * BYTE_VOCAB];
        let mut bi_mass = vec![0.0; n];

        if len == 0 || n == 0 {
            return Self {
                n,
                len: 0,
                uni,
                uni_mass,
                bi,
                bi_mass,
                has_bigrams: false,
            };
        }

        let inv_l = 1.0 / (len as f64);
        let mut assign = vec![0.0; n];

        // Agrégation souple vers 16 classes de nibbles par position
        let mut nib = vec![0.0; len * 16];
        for pos in 0..len {
            let row = &y[pos * BYTE_VOCAB..(pos + 1) * BYTE_VOCAB];
            spatial_assignment(pos, len, n, &mut assign);
            for (b, &p) in row.iter().enumerate() {
                let nb = fold_nibble(b as u8);
                nib[pos * 16 + nb] += p;
                for i in 0..n {
                    let w = assign[i] * inv_l * p;
                    uni[i * BYTE_VOCAB + b] += w;
                }
            }
            for i in 0..n {
                uni_mass[i] += assign[i] * inv_l;
            }
        }

        let has_bigrams = len >= 2;
        if has_bigrams {
            let inv_bm = 1.0 / ((len - 1) as f64);
            for pos in 1..len {
                spatial_assignment(pos, len, n, &mut assign);
                let prev_nib = &nib[(pos - 1) * 16..pos * 16];
                let curr_nib = &nib[pos * 16..(pos + 1) * 16];
                for r in 0..16 {
                    let pr = prev_nib[r];
                    if pr == 0.0 {
                        continue;
                    }
                    for s in 0..16 {
                        let prs = pr * curr_nib[s];
                        if prs == 0.0 {
                            continue;
                        }
                        let g_idx = r * 16 + s;
                        for i in 0..n {
                            let w = assign[i] * inv_bm * prs;
                            bi[i * BYTE_VOCAB + g_idx] += w;
                        }
                    }
                }
                for i in 0..n {
                    bi_mass[i] += assign[i] * inv_bm;
                }
            }
        }

        Self {
            n,
            len,
            uni,
            uni_mass,
            bi,
            bi_mass,
            has_bigrams,
        }
    }
}

#[inline]
pub fn fold_nibble(b: u8) -> usize {
    ((b & 0x0F) ^ (b >> 4)) as usize
}

#[inline]
pub fn bigram_bucket(prev: u8, curr: u8) -> usize {
    fold_nibble(prev) * 16 + fold_nibble(curr)
}

/// Partition de l'unité gaussienne lisse alignant la position `pos` sur les N nœuds.
pub fn spatial_assignment(pos: usize, len: usize, n: usize, out: &mut [f64]) {
    let x = ((pos as f64) + 0.5) / (len as f64);
    let sigma = (0.65 / (n as f64)).max(1e-4);
    let inv_2sig2 = 0.5 / (sigma * sigma);

    let mut max_logit = f64::NEG_INFINITY;
    for i in 0..n {
        let center = ((i as f64) + 0.5) / (n as f64);
        let d = x - center;
        let v = -d * d * inv_2sig2;
        out[i] = v;
        if v > max_logit {
            max_logit = v;
        }
    }
    let mut sum = 0.0;
    for v in out.iter_mut() {
        *v = (*v - max_logit).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in out.iter_mut() {
        *v *= inv_sum;
    }
}

/// Paramètres appris \theta de l'opérateur génératif relationnel P_\theta(E | R)
/// et du pôle régional initial \Psi_0(E) -> R(0).
#[derive(Debug, Clone)]
pub struct ModelParams {
    pub n: usize,
    pub d: usize,
    pub h: usize,
    /// Poids directs W_dir (h x d) : R_{ij, c} -> pré-activation d'arête.
    pub w_dir: Vec<f64>,
    /// Poids de composition 2-FWL W_comp (h x d) : (R_c R_c)_{ij} -> pré-activation d'arête.
    pub w_comp: Vec<f64>,
    /// Biais d'arête b_edge (h).
    pub b_edge: Vec<f64>,
    /// Tête d'émission unigramme W_out_uni (256 x h) et biais (256).
    pub w_out_uni: Vec<f64>,
    pub b_out_uni: Vec<f64>,
    /// Tête d'émission bigramme W_out_bi (256 x h) et biais (256).
    pub w_out_bi: Vec<f64>,
    pub b_out_bi: Vec<f64>,
    /// Encodeur du pôle régional \Psi_0(E) : projection (d x 256) + couplage non-réciproque (d x d).
    pub w_enc_uni: Vec<f64>,
    pub w_enc_bi: Vec<f64>,
    pub w_init_asym: Vec<f64>,
    /// Matrice de poids non-réciproques W_sol (N x N) générant le champ solénoïdal A_\perp.
    pub w_sol: Mat,
    /// Couplage de l'état R dans le champ solénoïdal (d).
    pub nu_sol: Vec<f64>,
}

impl ModelParams {
    pub fn random_init(n: usize, d: usize, h: usize, rng: &mut Rng) -> Self {
        // closure NE capture PAS rng : elle le reçoit en argument
        let init_vec = |len: usize, scale: f64, rng: &mut Rng| -> Vec<f64> {
            (0..len)
                .map(|_| (rng.gen_f64() * 2.0 - 1.0) * scale)
                .collect()
        };

        let s_edge = (1.0 / (2.0 * d as f64)).sqrt();
        let s_out = (1.0 / (h as f64)).sqrt() * 0.5;
        let s_enc = (1.0 / (BYTE_VOCAB as f64)).sqrt() * 2.0;

        let mut w_sol = Mat::zeros(n);
        let s_sol = 0.2 / (n as f64).sqrt();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    w_sol.set(i, j, (rng.gen_f64() * 2.0 - 1.0) * s_sol);
                }
            }
        }

        Self {
            n,
            d,
            h,
            w_dir: init_vec(h * d, s_edge, rng),
            w_comp: init_vec(h * d, s_edge * 0.5, rng),
            b_edge: vec![0.0; h],
            w_out_uni: init_vec(BYTE_VOCAB * h, s_out, rng),
            b_out_uni: vec![0.0; BYTE_VOCAB],
            w_out_bi: init_vec(BYTE_VOCAB * h, s_out, rng),
            b_out_bi: vec![0.0; BYTE_VOCAB],
            w_enc_uni: init_vec(d * BYTE_VOCAB, s_enc, rng),
            w_enc_bi: init_vec(d * BYTE_VOCAB, s_enc, rng),
            w_init_asym: init_vec(d * d, 0.5 / (d as f64).sqrt(), rng),
            w_sol,
            nu_sol: init_vec(d, 0.15, rng),
        }
    }

    /// Encode le pôle régional initial R(0) = \Psi_0(E) \in B_K (non-réciproque par construction).
    pub fn encode_regional_pole(&self, effect: &EffectSummary, radius_k: f64) -> RelTensor {
        debug_assert_eq!(effect.n, self.n);
        let n = self.n;
        let d = self.d;

        // Signatures nodales s_i \in R^d normalisées par la masse locale du nœud
        let mut sigs = vec![0.0; n * d];
        for i in 0..n {
            let inv_um = if effect.uni_mass[i] > 1e-12 {
                1.0 / effect.uni_mass[i]
            } else {
                0.0
            };
            let inv_bm = if effect.has_bigrams && effect.bi_mass[i] > 1e-12 {
                1.0 / effect.bi_mass[i]
            } else {
                0.0
            };
            for c in 0..d {
                let mut acc = 0.0;
                let w_u = &self.w_enc_uni[c * BYTE_VOCAB..(c + 1) * BYTE_VOCAB];
                let q_i = &effect.uni[i * BYTE_VOCAB..(i + 1) * BYTE_VOCAB];
                for b in 0..BYTE_VOCAB {
                    acc += w_u[b] * (q_i[b] * inv_um);
                }
                if inv_bm > 0.0 {
                    let w_b = &self.w_enc_bi[c * BYTE_VOCAB..(c + 1) * BYTE_VOCAB];
                    let t_i = &effect.bi[i * BYTE_VOCAB..(i + 1) * BYTE_VOCAB];
                    for g in 0..BYTE_VOCAB {
                        acc += 0.5 * w_b[g] * (t_i[g] * inv_bm);
                    }
                }
                sigs[i * d + c] = acc.tanh();
            }
        }

        // Transformation asymétrique t_j = W_init_asym * s_j pour briser la symétrie dès t = 0
        let mut target_sigs = vec![0.0; n * d];
        for j in 0..n {
            for c in 0..d {
                let mut acc = 0.0;
                for a in 0..d {
                    acc += self.w_init_asym[c * d + a] * sigs[j * d + a];
                }
                target_sigs[j * d + c] = acc;
            }
        }

        let mut r0 = RelTensor::zeros(n, d);
        for c in 0..d {
            for i in 0..n {
                for j in 0..n {
                    let val = (sigs[i * d + c] * target_sigs[j * d + c] + 0.25 * (sigs[i * d + c] - sigs[j * d + c])).tanh();
                    r0.set(i, j, c, val * 0.3);
                }
            }
        }
        r0.clamp_to_ball(radius_k);
        r0
    }

    /// Calcule \bar{\rho}(R, E) (passe avant seule, sans allocation de gradient).
    pub fn residual_only(
        &self,
        r: &RelTensor,
        effect: &EffectSummary,
        lambda_bi: f64,
    ) -> f64 {
        let (w_uni, w_bi) = residual_weights(effect.has_bigrams, lambda_bi);
        let n = self.n;
        let d = self.d;
        let h = self.h;

        // Composition 2-FWL : C_c = R_c * R_c
        let comp: Vec<Mat> = (0..d)
            .map(|c| r.channels[c].matmul(&r.channels[c]))
            .collect();

        let inv_n = 1.0 / (n as f64).sqrt();
        let mut h_node = vec![0.0; h];
        let mut logits = [0.0f64; BYTE_VOCAB];
        let mut total = 0.0;

        for i in 0..n {
            h_node.fill(0.0);
            for j in 0..n {
                for r_idx in 0..h {
                    let mut z = self.b_edge[r_idx];
                    let w_d = &self.w_dir[r_idx * d..(r_idx + 1) * d];
                    let w_c = &self.w_comp[r_idx * d..(r_idx + 1) * d];
                    for c in 0..d {
                        z += w_d[c] * r.get(i, j, c) + w_c[c] * comp[c].get(i, j);
                    }
                    h_node[r_idx] += z.tanh();
                }
            }
            for val in &mut h_node {
                *val *= inv_n;
            }

            if effect.uni_mass[i] > 0.0 {
                let loss_u = head_cross_entropy(
                    &h_node,
                    &self.w_out_uni,
                    &self.b_out_uni,
                    &effect.uni[i * BYTE_VOCAB..(i + 1) * BYTE_VOCAB],
                    effect.uni_mass[i],
                    &mut logits,
                );
                total += w_uni * loss_u;
            }
            if w_bi > 0.0 && effect.bi_mass[i] > 0.0 {
                let loss_b = head_cross_entropy(
                    &h_node,
                    &self.w_out_bi,
                    &self.b_out_bi,
                    &effect.bi[i * BYTE_VOCAB..(i + 1) * BYTE_VOCAB],
                    effect.bi_mass[i],
                    &mut logits,
                );
                total += w_bi * loss_b;
            }
        }
        total
    }

    /// Calcule exactement (\bar{\rho}(R, E), \nabla_R \bar{\rho}(R, E)).
    pub fn residual_and_grad_r(
        &self,
        r: &RelTensor,
        effect: &EffectSummary,
        lambda_bi: f64,
    ) -> (f64, RelTensor) {
        let (w_uni, w_bi) = residual_weights(effect.has_bigrams, lambda_bi);
        let n = self.n;
        let d = self.d;
        let h = self.h;
        let inv_n = 1.0 / (n as f64).sqrt();

        let comp: Vec<Mat> = (0..d)
            .map(|c| r.channels[c].matmul(&r.channels[c]))
            .collect();

        // Stockage des activations m_{ij, r} = tanh(z_{ij, r}) pour le backward exact
        let mut m_edge = vec![0.0; n * n * h];
        let mut h_nodes = vec![0.0; n * h];

        for i in 0..n {
            for j in 0..n {
                let base = (i * n + j) * h;
                for r_idx in 0..h {
                    let mut z = self.b_edge[r_idx];
                    let w_d = &self.w_dir[r_idx * d..(r_idx + 1) * d];
                    let w_c = &self.w_comp[r_idx * d..(r_idx + 1) * d];
                    for c in 0..d {
                        z += w_d[c] * r.get(i, j, c) + w_c[c] * comp[c].get(i, j);
                    }
                    let m = z.tanh();
                    m_edge[base + r_idx] = m;
                    h_nodes[i * h + r_idx] += m;
                }
            }
            for r_idx in 0..h {
                h_nodes[i * h + r_idx] *= inv_n;
            }
        }

        let mut total_loss = 0.0;
        let mut grad_h = vec![0.0; n * h];
        let mut probs = [0.0f64; BYTE_VOCAB];

        for i in 0..n {
            let hi = &h_nodes[i * h..(i + 1) * h];
            let ghi = &mut grad_h[i * h..(i + 1) * h];

            if effect.uni_mass[i] > 0.0 {
                let lu = head_cross_entropy_and_grad_h(
                    hi,
                    &self.w_out_uni,
                    &self.b_out_uni,
                    &effect.uni[i * BYTE_VOCAB..(i + 1) * BYTE_VOCAB],
                    effect.uni_mass[i],
                    w_uni,
                    &mut probs,
                    ghi,
                );
                total_loss += w_uni * lu;
            }
            if w_bi > 0.0 && effect.bi_mass[i] > 0.0 {
                let lb = head_cross_entropy_and_grad_h(
                    hi,
                    &self.w_out_bi,
                    &self.b_out_bi,
                    &effect.bi[i * BYTE_VOCAB..(i + 1) * BYTE_VOCAB],
                    effect.bi_mass[i],
                    w_bi,
                    &mut probs,
                    ghi,
                );
                total_loss += w_bi * lb;
            }
        }

        // Rétropropagation exacte vers R (branche directe + branche composée C_c = R_c R_c)
        let mut grad_r = RelTensor::zeros(n, d);
        let mut grad_comp: Vec<Mat> = (0..d).map(|_| Mat::zeros(n)).collect();

        for i in 0..n {
            let ghi = &grad_h[i * h..(i + 1) * h];
            for j in 0..n {
                let base = (i * n + j) * h;
                for r_idx in 0..h {
                    let m = m_edge[base + r_idx];
                    let dz = inv_n * ghi[r_idx] * (1.0 - m * m);
                    if dz == 0.0 {
                        continue;
                    }
                    let w_d = &self.w_dir[r_idx * d..(r_idx + 1) * d];
                    let w_c = &self.w_comp[r_idx * d..(r_idx + 1) * d];
                    for c in 0..d {
                        grad_r.channels[c].add_assign(i, j, dz * w_d[c]);
                        grad_comp[c].add_assign(i, j, dz * w_c[c]);
                    }
                }
            }
        }

        // Adjoint de C_c = R_c R_c : dR_c += G_c * R_c^T + R_c^T * G_c
        for c in 0..d {
            let term1 = grad_comp[c].matmul_t_right(&r.channels[c]);
            let term2 = r.channels[c].matmul_t_left(&grad_comp[c]);
            grad_r.channels[c].axpy(1.0, &term1);
            grad_r.channels[c].axpy(1.0, &term2);
        }

        (total_loss, grad_r)
    }
}

/// Sondes déterministes de stabilité différentielle \mathcal{T}_{inv} = \mathcal{T}_{dep} \cup \mathcal{T}_{abs}.
/// Fixées une fois par épisode afin que \phi_\sharp(R, E) soit un potentiel déterministe C^1.
#[derive(Debug, Clone)]
pub struct StabilityProbes {
    /// Matrices orthogonales Q_m \in SO(N) pour tester l'invariance au déplacement (Q R Q^T).
    pub rotations: Vec<Mat>,
    /// Matrice symétrique P_avg pour la sonde d'abstraction spectrale (1-\kappa) R + \kappa P R P.
    pub p_avg: Mat,
    /// Intensité \kappa \in (0, 1) de la contraction d'abstraction.
    pub abs_kappa: f64,
}

impl StabilityProbes {
    /// Construit `num_rotations` sondes orthogonales exactes (compositions de rotations de Givens)
    /// et une sonde de contraction macro-relationnelle.
    pub fn build(n: usize, num_rotations: usize, max_angle: f64, abs_kappa: f64, seed: u64) -> Self {
        let mut rng = Rng::from_seed(seed);
        let mut rotations = Vec::with_capacity(num_rotations);

        for _ in 0..num_rotations {
            let mut q = Mat::identity(n);
            let num_planes = n.max(2);
            for _ in 0..num_planes {
                if n < 2 {
                    break;
                }
                let p = rng.gen_range(n);
                let mut r = rng.gen_range(n - 1);
                if r >= p {
                    r += 1;
                }
                let theta = (rng.gen_f64() * 2.0 - 1.0) * max_angle;
                let c = theta.cos();
                let s = theta.sin();
                // Multiplication à gauche par la rotation de Givens G(p, r, theta)
                for col in 0..n {
                    let qp = q.get(p, col);
                    let qr = q.get(r, col);
                    q.set(p, col, c * qp - s * qr);
                    q.set(r, col, s * qp + c * qr);
                }
            }
            rotations.push(q);
        }

        // P_avg = (1 - mu) I + (mu / N) * 1 1^T (symétrique, doublement stochastique)
        let mu = 0.35;
        let off = mu / (n.max(1) as f64);
        let mut p_avg = Mat::zeros(n);
        for i in 0..n {
            for j in 0..n {
                p_avg.set(i, j, if i == j { (1.0 - mu) + off } else { off });
            }
        }

        Self {
            rotations,
            p_avg,
            abs_kappa,
        }
    }

    pub fn probe_count(&self) -> usize {
        self.rotations.len() + if self.abs_kappa > 0.0 { 1 } else { 0 }
    }

    fn apply_rotation(q: &Mat, r: &RelTensor) -> RelTensor {
        let mut out = RelTensor::zeros(r.n, r.d);
        for c in 0..r.d {
            let qr = q.matmul(&r.channels[c]);
            out.channels[c] = qr.matmul_t_right(q);
        }
        out
    }

    fn adjoint_rotation(q: &Mat, grad_rot: &RelTensor) -> RelTensor {
        let mut out = RelTensor::zeros(grad_rot.n, grad_rot.d);
        for c in 0..grad_rot.d {
            let qtg = q.matmul_t_left(&grad_rot.channels[c]);
            out.channels[c] = qtg.matmul(q);
        }
        out
    }

    fn apply_abstraction(&self, r: &RelTensor) -> RelTensor {
        let k = self.abs_kappa;
        let one_minus_k = 1.0 - k;
        let mut out = RelTensor::zeros(r.n, r.d);
        for c in 0..r.d {
            let prp = self.p_avg.matmul(&r.channels[c]).matmul(&self.p_avg);
            for idx in 0..(r.n * r.n) {
                out.channels[c].data[idx] =
                    one_minus_k * r.channels[c].data[idx] + k * prp.data[idx];
            }
        }
        out
    }

    /// Auto-adjoint car P_avg = P_avg^T.
    fn adjoint_abstraction(&self, grad_abs: &RelTensor) -> RelTensor {
        self.apply_abstraction(grad_abs)
    }
}

/// Bilan complet d'une évaluation de \phi_\sharp(R, E).
#[derive(Debug, Clone)]
pub struct PhiSharpEval {
    /// Scalaire global \phi_\sharp(R, E) = \lambda_2 - \alpha \bar{\rho} - \beta V_stab.
    pub phi_sharp: f64,
    /// Connectivité algébrique \lambda_2(L_{\bar{w}}).
    pub lambda2: f64,
    /// Résidu abductif normalisé par octet \bar{\rho}(R, E).
    pub residual: f64,
    /// Fragilité différentielle V_stab(R, E).
    pub v_stab: f64,
    /// Matrice des poids causaux dirigés w_{ij}(R, E).
    pub causal_weights: Mat,
    /// Nombre d'arêtes dirigées causalement actives (w_{ij} > 0).
    pub active_causal_edges: usize,
    /// Détail spectral de Fiedler.
    pub fiedler: FiedlerResult,
}

/// Évalue \phi_\sharp(R, E) et (si `compute_grad = true`) son gradient analytique EXACT \nabla_R \phi_\sharp.
pub fn evaluate_phi_sharp(
    params: &ModelParams,
    r: &RelTensor,
    effect: &EffectSummary,
    probes: &StabilityProbes,
    cfg: &PhiSharpConfig,
    compute_grad: bool,
) -> (PhiSharpEval, Option<RelTensor>) {
    let n = r.n;
    let d = r.d;

    // 1. Résidu principal \bar{\rho}(R, E) et son gradient \nabla_R \bar{\rho}
    let (rho, grad_rho_opt) = if compute_grad {
        let (val, g) = params.residual_and_grad_r(r, effect, cfg.lambda_bi);
        (val, Some(g))
    } else {
        (params.residual_only(r, effect, cfg.lambda_bi), None)
    };

    let p_bar = (-rho).exp();

    // 2. Poids causaux dirigés par ablation exacte w_{ij}(R, E)
    let mut causal_weights = Mat::zeros(n);
    // Stockage intermédiaire pour le backward sans recalculer \bar{\rho}_{ij}
    let mut abl_deriv_rho = Mat::zeros(n);
    let mut abl_deriv_rho_ij = Mat::zeros(n);
    let mut active_causal_edges = 0usize;

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // Raccourci exact : si l'arête R_{ij} est nulle sur tous les canaux,
            // R^{\setminus(i,j)} == R, donc w_{ij} == 0 exactement.
            let is_zero = (0..d).all(|c| r.get(i, j, c) == 0.0);
            if is_zero {
                continue;
            }

            let r_abl = r.with_ablated_edge(i, j);
            let rho_ij = params.residual_only(&r_abl, effect, cfg.lambda_bi);
            let (s_ij, ds_drho_ij) = smooth_cap(rho_ij, UNIFORM_BYTE_RESIDUAL, cfg.smin_eps);
            let delta = s_ij - rho;
            let (h_val, h_prime) = smooth_hinge(delta, cfg.hinge_tau);

            if h_val > 0.0 {
                let wij = p_bar * h_val;
                causal_weights.set(i, j, wij);
                active_causal_edges += 1;

                if compute_grad {
                    abl_deriv_rho.set(i, j, -p_bar * (h_val + h_prime));
                    abl_deriv_rho_ij.set(i, j, p_bar * h_prime * ds_drho_ij);
                }
            }
        }
    }

    // 3. Connectivité algébrique \lambda_2(L_{\bar{w}}) par Lanczos + puissance inverse CG
    let lap = CausalLaplacian::from_directed_weights(&causal_weights);
    let fiedler = lap.compute_fiedler(cfg.max_krylov, cfg.max_inv_power);
    let lambda2 = fiedler.lambda2;

    // 4. Stabilité différentielle V_stab(R, E) le long des sondes \mathcal{T}_{inv}
    let total_probes = probes.probe_count();
    let mut v_stab = 0.0;
    let mut grad_r = if compute_grad {
        Some(RelTensor::zeros(n, d))
    } else {
        None
    };

    // Accumulation scalaire devant \nabla_R \bar{\rho}(R, E)
    let mut coeff_on_grad_rho = -cfg.alpha;

    if compute_grad {
        // Contribution de \lambda_2 : Hellmann-Feynman sur W puis chaîne dans chaque ablation active
        let g_w = CausalLaplacian::grad_directed_weights(&fiedler.v2);
        let grad_acc = grad_r.as_mut().unwrap();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let gw = g_w.get(i, j);
                if gw <= 0.0 {
                    continue;
                }
                let dw_drho = abl_deriv_rho.get(i, j);
                let dw_drho_ij = abl_deriv_rho_ij.get(i, j);
                if dw_drho == 0.0 && dw_drho_ij == 0.0 {
                    continue;
                }

                coeff_on_grad_rho += gw * dw_drho;

                let mult = gw * dw_drho_ij;
                if mult.abs() > 0.0 {
                    let r_abl = r.with_ablated_edge(i, j);
                    let (_, mut g_abl) = params.residual_and_grad_r(&r_abl, effect, cfg.lambda_bi);
                    // Masquage strict de l'arête ablatée (d R^{\setminus(i,j)} / d R_{ij} = 0)
                    for c in 0..d {
                        g_abl.set(i, j, c, 0.0);
                    }
                    grad_acc.axpy(mult, &g_abl);
                }
            }
        }
    }

    if total_probes > 0 && cfg.beta > 0.0 {
        let inv_m = 1.0 / (total_probes as f64);

        // 4a. Sondes de déplacement orthogonal Q R Q^T
        for q in &probes.rotations {
            let r_rot = StabilityProbes::apply_rotation(q, r);
            let rho_rot = params.residual_only(&r_rot, effect, cfg.lambda_bi);
            let (hv, hp) = smooth_hinge(rho_rot - rho, cfg.hinge_tau);
            v_stab += inv_m * hv;

            if compute_grad && hp > 0.0 {
                let scale = -cfg.beta * inv_m * hp;
                coeff_on_grad_rho -= scale; // car d(rho_rot - rho)/d(rho) = -1
                let (_, g_rot) = params.residual_and_grad_r(&r_rot, effect, cfg.lambda_bi);
                let g_back = StabilityProbes::adjoint_rotation(q, &g_rot);
                grad_r.as_mut().unwrap().axpy(scale, &g_back);
            }
        }

        // 4b. Sonde d'abstraction spectrale (1-\kappa) R + \kappa P R P
        if probes.abs_kappa > 0.0 {
            let r_abs = probes.apply_abstraction(r);
            let rho_abs = params.residual_only(&r_abs, effect, cfg.lambda_bi);
            let (hv, hp) = smooth_hinge(rho_abs - rho, cfg.hinge_tau);
            v_stab += inv_m * hv;

            if compute_grad && hp > 0.0 {
                let scale = -cfg.beta * inv_m * hp;
                coeff_on_grad_rho -= scale;
                let (_, g_abs) = params.residual_and_grad_r(&r_abs, effect, cfg.lambda_bi);
                let g_back = probes.adjoint_abstraction(&g_abs);
                grad_r.as_mut().unwrap().axpy(scale, &g_back);
            }
        }
    }

    if let (Some(g_acc), Some(g_rho)) = (grad_r.as_mut(), grad_rho_opt.as_ref()) {
        g_acc.axpy(coeff_on_grad_rho, g_rho);
    }

    let phi_sharp = lambda2 - cfg.alpha * rho - cfg.beta * v_stab;

    let eval = PhiSharpEval {
        phi_sharp,
        lambda2,
        residual: rho,
        v_stab,
        causal_weights,
        active_causal_edges,
        fiedler,
    };
    (eval, grad_r)
}

#[inline]
fn residual_weights(has_bigrams: bool, lambda_bi: f64) -> (f64, f64) {
    if !has_bigrams || lambda_bi <= 0.0 {
        (1.0, 0.0)
    } else {
        let denom = 1.0 + lambda_bi;
        (1.0 / denom, lambda_bi / denom)
    }
}

/// Charnière quadratique C^1 : 0 si x <= 0, x^2/(2\tau) sur (0, \tau), x - \tau/2 si x >= \tau.
#[inline]
pub fn smooth_hinge(x: f64, tau: f64) -> (f64, f64) {
    if x <= 0.0 {
        (0.0, 0.0)
    } else if x < tau {
        (0.5 * x * x / tau, x / tau)
    } else {
        (x - 0.5 * tau, 1.0)
    }
}

/// Plafond lisse C^1 sans biais sous le seuil b :
/// - pour a <= b : vaut exactement `a` (dérivée = 1, zéro zone morte sous log(256))
/// - pour a >= b + eps : plafonné à `b + 0.5*eps` (dérivée = 0)
#[inline]
pub fn smooth_cap(a: f64, b: f64, eps: f64) -> (f64, f64) {
    let (h, hp) = smooth_hinge(a - b, eps);
    (a - h, 1.0 - hp)
}

fn head_cross_entropy(
    h_node: &[f64],
    w_out: &[f64],
    b_out: &[f64],
    target: &[f64],
    mass: f64,
    logits: &mut [f64; BYTE_VOCAB],
) -> f64 {
    let h = h_node.len();
    let mut max_l = f64::NEG_INFINITY;
    for b in 0..BYTE_VOCAB {
        let row = &w_out[b * h..(b + 1) * h];
        let mut acc = b_out[b];
        for r in 0..h {
            acc += row[r] * h_node[r];
        }
        logits[b] = acc;
        if acc > max_l {
            max_l = acc;
        }
    }
    let mut sum_exp = 0.0;
    for &l in logits.iter() {
        sum_exp += (l - max_l).exp();
    }
    let lse = max_l + sum_exp.ln();
    let mut dot_qt = 0.0;
    for b in 0..BYTE_VOCAB {
        if target[b] > 0.0 {
            dot_qt += target[b] * logits[b];
        }
    }
    mass * lse - dot_qt
}

#[allow(clippy::too_many_arguments)]
fn head_cross_entropy_and_grad_h(
    h_node: &[f64],
    w_out: &[f64],
    b_out: &[f64],
    target: &[f64],
    mass: f64,
    loss_weight: f64,
    probs: &mut [f64; BYTE_VOCAB],
    grad_h_out: &mut [f64],
) -> f64 {
    let h = h_node.len();
    let mut max_l = f64::NEG_INFINITY;
    for b in 0..BYTE_VOCAB {
        let row = &w_out[b * h..(b + 1) * h];
        let mut acc = b_out[b];
        for r in 0..h {
            acc += row[r] * h_node[r];
        }
        probs[b] = acc;
        if acc > max_l {
            max_l = acc;
        }
    }
    let mut sum_exp = 0.0;
    let mut dot_qt = 0.0;
    for b in 0..BYTE_VOCAB {
        let l = probs[b];
        if target[b] > 0.0 {
            dot_qt += target[b] * l;
        }
        let ex = (l - max_l).exp();
        probs[b] = ex;
        sum_exp += ex;
    }
    let lse = max_l + sum_exp.ln();
    let inv_sum = 1.0 / sum_exp;

    for b in 0..BYTE_VOCAB {
        let p_b = probs[b] * inv_sum;
        let dl = loss_weight * (mass * p_b - target[b]);
        if dl == 0.0 {
            continue;
        }
        let row = &w_out[b * h..(b + 1) * h];
        for r in 0..h {
            grad_h_out[r] += dl * row[r];
        }
    }

    mass * lse - dot_qt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theorem_2_diagonal_memorization_has_zero_lambda2() {
        let mut rng = Rng::from_seed(77);
        let n = 6;
        let d = 2;
        let h = 4;
        let params = ModelParams::random_init(n, d, h, &mut rng);
        let bytes: Vec<u8> = (0..64).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        let eff = EffectSummary::from_bytes(&bytes, n);
        let probes = StabilityProbes::build(n, 2, 0.05, 0.1, 99);
        let cfg = PhiSharpConfig::default();

        // État strictement diagonal R_copie (aucune relation hors-diagonale)
        let mut r_diag = RelTensor::zeros(n, d);
        for c in 0..d {
            for i in 0..n {
                r_diag.set(i, i, c, 0.8);
            }
        }

        let (eval, _) = evaluate_phi_sharp(&params, &r_diag, &eff, &probes, &cfg, false);
        assert_eq!(eval.active_causal_edges, 0);
        assert_eq!(eval.lambda2, 0.0);
        assert!(eval.phi_sharp <= 0.0);
    }

    #[test]
    fn test_residual_gradient_exact_finite_diff() {
        let mut rng = Rng::from_seed(101);
        let n = 4;
        let d = 2;
        let h = 3;
        let params = ModelParams::random_init(n, d, h, &mut rng);
        let bytes: Vec<u8> = (0..40).map(|i| ((i * 53 + 7) % 256) as u8).collect();
        let eff = EffectSummary::from_bytes(&bytes, n);

        let mut r = RelTensor::zeros(n, d);
        for c in 0..d {
            for i in 0..n {
                for j in 0..n {
                    r.set(i, j, c, (rng.gen_f64() * 2.0 - 1.0) * 0.6);
                }
            }
        }

        let (_, analytic) = params.residual_and_grad_r(&r, &eff, 0.5);
        let eps = 1e-6;
        for c in 0..d {
            for i in 0..n {
                for j in 0..n {
                    let mut rp = r.clone();
                    let mut rm = r.clone();
                    rp.set(i, j, c, r.get(i, j, c) + eps);
                    rm.set(i, j, c, r.get(i, j, c) - eps);
                    let vp = params.residual_only(&rp, &eff, 0.5);
                    let vm = params.residual_only(&rm, &eff, 0.5);
                    let fd = (vp - vm) / (2.0 * eps);
                    let an = analytic.get(i, j, c);
                    assert!(
                        (fd - an).abs() < 1e-6,
                        "Residual grad mismatch ({i},{j},{c}): fd={fd}, an={an}"
                    );
                }
            }
        }
    }
    
        #[test]
    fn test_full_phi_sharp_gradient_exact_finite_diff() {
        // Vérifie le gradient analytique COMPLET de \phi_\sharp(R, E) :
        // \lambda_2(L_{\bar{w}}) (via toutes les ablations + Fiedler) - \alpha \bar{\rho} - \beta V_stab
        let mut rng = Rng::from_seed(2026);
        let n = 5;
        let d = 2;
        let h = 8;
        let mut params = ModelParams::random_init(n, d, h, &mut rng);
        for v in &mut params.w_out_uni {
            *v *= 4.0;
        }
        for v in &mut params.w_out_bi {
            *v *= 4.0;
        }

        // Deux régimes d'octets distincts (nœuds {0,1} vs {2,3,4}) pour briser la dégénérescence K_5
        // et obtenir une coupe de Fiedler simple (\lambda_3 - \lambda_2 grand).
        let mut bytes = Vec::with_capacity(60);
        for _ in 0..12 {
            bytes.extend_from_slice(&[12u8, 45]);
        }
        for _ in 0..18 {
            bytes.extend_from_slice(&[201u8, 234]);
        }

        let eff = EffectSummary::from_bytes(&bytes, n);
        let probes = StabilityProbes::build(n, 2, 0.08, 0.15, 31415);
        let cfg = PhiSharpConfig {
            alpha: 0.4,
            beta: 0.3,
            lambda_bi: 0.5,
            hinge_tau: 0.005,
            smin_eps: 0.02,
            max_krylov: 16,
            max_inv_power: 6,
        };

        let mut r = params.encode_regional_pole(&eff, 4.0);
        // Descente préliminaire pour abaisser \bar{\rho} et activer les arêtes causales
        for _ in 0..25 {
            let (_, g_rho) = params.residual_and_grad_r(&r, &eff, cfg.lambda_bi);
            r.axpy(-1.5, &g_rho);
            r.clamp_to_ball(4.0);
        }
        // Léger goulot sur les arêtes inter-clusters ({0,1} <-> {2,3,4}) puis 3 pas de réalignement
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
            let (_, g_rho) = params.residual_and_grad_r(&r, &eff, cfg.lambda_bi);
            r.axpy(-0.5, &g_rho);
            r.clamp_to_ball(4.0);
        }

        let (eval, grad_opt) = evaluate_phi_sharp(&params, &r, &eff, &probes, &cfg, true);
        let analytic = grad_opt.unwrap();
        assert!(
            eval.active_causal_edges > 0,
            "Des arêtes causales doivent être actives pour tester Hellmann-Feynman"
        );
        assert!(
            eval.lambda2 > 1e-5 && (eval.fiedler.lambda3 - eval.lambda2).abs() > 1e-5,
            "lambda2={} doit être simple et > 0 (lambda3={})",
            eval.lambda2,
            eval.fiedler.lambda3
        );

        let eps = 1e-6;
        let mut max_err = 0.0f64;
        for c in 0..d {
            for i in 0..n {
                for j in 0..n {
                    let mut rp = r.clone();
                    let mut rm = r.clone();
                    rp.set(i, j, c, r.get(i, j, c) + eps);
                    rm.set(i, j, c, r.get(i, j, c) - eps);
                    let (ep, _) = evaluate_phi_sharp(&params, &rp, &eff, &probes, &cfg, false);
                    let (em, _) = evaluate_phi_sharp(&params, &rm, &eff, &probes, &cfg, false);
                    let fd = (ep.phi_sharp - em.phi_sharp) / (2.0 * eps);
                    let an = analytic.get(i, j, c);
                    let err = (fd - an).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    assert!(
                        err < 1e-4,
                        "Phi_sharp grad mismatch ({i},{j},{c}): fd={fd}, an={an}, err={err}"
                    );
                }
            }
        }
        assert!(max_err < 1e-4);
    }
}