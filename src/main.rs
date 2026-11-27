//! Prototype minimal 100% connexionniste (zéro axiome hardcodé, zéro dep externe).
//! - Entrée : (I = X X^T, S, S^T, I_L) strictement S_256-invariante
//! - Couches relationnelles génériques : R, R*R, R⊙R
//! - Exécuteur aveugle (ne voit JAMAIS I) + Copy-Transport T X
//! - Tests OOD : alphabet ASCII inédit ("ababababab"), longueurs inédites, périodes inédites.

use std::f64;

// ============================================================================
// 1. PRNG autonome (Xoshiro256**)
// ============================================================================

struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn from_seed(mut seed: u64) -> Self {
        let mut sm = || -> u64 {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self {
            s: [sm(), sm(), sm(), sm()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    fn gen_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() % (upper as u64)) as usize
    }
}

// ============================================================================
// 2. Opérations matricielles L x L
// ============================================================================

fn matmul(a: &[f64], b: &[f64], l: usize) -> Vec<f64> {
    let mut out = vec![0.0; l * l];
    for i in 0..l {
        let ra = i * l;
        for k in 0..l {
            let aik = a[ra + k];
            if aik == 0.0 {
                continue;
            }
            let rb = k * l;
            for j in 0..l {
                out[ra + j] += aik * b[rb + j];
            }
        }
    }
    out
}

fn matmul_t_right(a: &[f64], b: &[f64], l: usize) -> Vec<f64> {
    let mut out = vec![0.0; l * l];
    for i in 0..l {
        let ra = i * l;
        for j in 0..l {
            let rb = j * l;
            let mut acc = 0.0;
            for k in 0..l {
                acc += a[ra + k] * b[rb + k];
            }
            out[ra + j] = acc;
        }
    }
    out
}

fn matmul_t_left(a: &[f64], b: &[f64], l: usize) -> Vec<f64> {
    let mut out = vec![0.0; l * l];
    for k in 0..l {
        let ra = k * l;
        let rb = k * l;
        for i in 0..l {
            let aki = a[ra + i];
            if aki == 0.0 {
                continue;
            }
            let rc = i * l;
            for j in 0..l {
                out[rc + j] += aki * b[rb + j];
            }
        }
    }
    out
}

/// Construit (I = X X^T, S, S^T, I_L) depuis une séquence brute d'octets.
fn build_raw_relations(bytes: &[u8]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let l = bytes.len();
    let mut id_eq = vec![0.0; l * l];
    let mut shift = vec![0.0; l * l];
    let mut shift_t = vec![0.0; l * l];
    let mut eye = vec![0.0; l * l];

    for i in 0..l {
        eye[i * l + i] = 1.0;
        for j in 0..l {
            if bytes[i] == bytes[j] {
                id_eq[i * l + j] = 1.0;
            }
        }
        if i + 1 < l {
            shift[(i + 1) * l + i] = 1.0; // S_{t, t-1} = 1
            shift_t[i * l + (i + 1)] = 1.0; // S^T_{t-1, t} = 1
        }
    }
    (id_eq, shift, shift_t, eye)
}

// ============================================================================
// 3. Couche relationnelle générique (R, R*R, R⊙R) + Backward exact
// ============================================================================

#[derive(Clone)]
struct RelLayer {
    c_in: usize,
    c_out: usize,
    w1: Vec<f64>, // (c_out x c_in) sur R
    w2: Vec<f64>, // (c_out x c_in) sur (R * R) / sqrt(L)
    w3: Vec<f64>, // (c_out x c_in) sur R_c ⊙ R_{(c+1)%c_in}
    b: Vec<f64>,  // (c_out)
}

struct RelLayerGrad {
    w1: Vec<f64>,
    w2: Vec<f64>,
    w3: Vec<f64>,
    b: Vec<f64>,
}

struct RelLayerCache {
    l: usize,
    x: Vec<Vec<f64>>,
    sq: Vec<Vec<f64>>,
    had: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
}

impl RelLayer {
    fn random_init(c_in: usize, c_out: usize, rng: &mut Rng) -> Self {
        let scale = (1.0 / (3.0 * c_in as f64)).sqrt();
        let mut init = |n: usize| -> Vec<f64> {
            (0..n).map(|_| (rng.gen_f64() * 2.0 - 1.0) * scale).collect()
        };
        Self {
            c_in,
            c_out,
            w1: init(c_out * c_in),
            w2: init(c_out * c_in),
            w3: init(c_out * c_in),
            b: vec![0.0; c_out],
        }
    }

    fn zero_grad(&self) -> RelLayerGrad {
        RelLayerGrad {
            w1: vec![0.0; self.w1.len()],
            w2: vec![0.0; self.w2.len()],
            w3: vec![0.0; self.w3.len()],
            b: vec![0.0; self.b.len()],
        }
    }

    fn forward(&self, x: &[Vec<f64>], l: usize) -> (Vec<Vec<f64>>, RelLayerCache) {
        let inv_sqrt_l = 1.0 / (l as f64).sqrt();
        let mut sq = Vec::with_capacity(self.c_in);
        let mut had = Vec::with_capacity(self.c_in);

        for c in 0..self.c_in {
            let mut m = matmul(&x[c], &x[c], l);
            for v in &mut m {
                *v *= inv_sqrt_l;
            }
            sq.push(m);

            let c_next = (c + 1) % self.c_in;
            let h: Vec<f64> = x[c]
                .iter()
                .zip(x[c_next].iter())
                .map(|(a, b)| a * b)
                .collect();
            had.push(h);
        }

        let mut y = vec![vec![0.0; l * l]; self.c_out];
        for o in 0..self.c_out {
            let bo = self.b[o];
            let mut z = vec![bo; l * l];
            for c in 0..self.c_in {
                let w1 = self.w1[o * self.c_in + c];
                let w2 = self.w2[o * self.c_in + c];
                let w3 = self.w3[o * self.c_in + c];
                for idx in 0..(l * l) {
                    z[idx] += w1 * x[c][idx] + w2 * sq[c][idx] + w3 * had[c][idx];
                }
            }
            for idx in 0..(l * l) {
                y[o][idx] = z[idx].tanh();
            }
        }

        let cache = RelLayerCache {
            l,
            x: x.to_vec(),
            sq,
            had,
            y: y.clone(),
        };
        (y, cache)
    }

    fn backward(
        &self,
        dy: &[Vec<f64>],
        cache: &RelLayerCache,
        grad: &mut RelLayerGrad,
    ) -> Vec<Vec<f64>> {
        let l = cache.l;
        let inv_sqrt_l = 1.0 / (l as f64).sqrt();
        let mut dx = vec![vec![0.0; l * l]; self.c_in];
        let mut dsq = vec![vec![0.0; l * l]; self.c_in];
        let mut dhad = vec![vec![0.0; l * l]; self.c_in];

        for o in 0..self.c_out {
            let mut dz = vec![0.0; l * l];
            let mut sum_dz = 0.0;
            for idx in 0..(l * l) {
                let yv = cache.y[o][idx];
                let v = dy[o][idx] * (1.0 - yv * yv);
                dz[idx] = v;
                sum_dz += v;
            }
            grad.b[o] += sum_dz;

            for c in 0..self.c_in {
                let w1 = self.w1[o * self.c_in + c];
                let w2 = self.w2[o * self.c_in + c];
                let w3 = self.w3[o * self.c_in + c];
                let mut g1 = 0.0;
                let mut g2 = 0.0;
                let mut g3 = 0.0;
                for idx in 0..(l * l) {
                    let dzv = dz[idx];
                    g1 += dzv * cache.x[c][idx];
                    g2 += dzv * cache.sq[c][idx];
                    g3 += dzv * cache.had[c][idx];
                    dx[c][idx] += w1 * dzv;
                    dsq[c][idx] += w2 * dzv;
                    dhad[c][idx] += w3 * dzv;
                }
                grad.w1[o * self.c_in + c] += g1;
                grad.w2[o * self.c_in + c] += g2;
                grad.w3[o * self.c_in + c] += g3;
            }
        }

        for c in 0..self.c_in {
            // Adjoint de (1/sqrt(L)) * X * X
            let t1 = matmul_t_right(&dsq[c], &cache.x[c], l);
            let t2 = matmul_t_left(&cache.x[c], &dsq[c], l);
            for idx in 0..(l * l) {
                dx[c][idx] += inv_sqrt_l * (t1[idx] + t2[idx]);
            }

            // Adjoint de X_c ⊙ X_{(c+1)%c_in}
            let c_next = (c + 1) % self.c_in;
            for idx in 0..(l * l) {
                let dh = dhad[c][idx];
                dx[c][idx] += dh * cache.x[c_next][idx];
                dx[c_next][idx] += dh * cache.x[c][idx];
            }
        }

        dx
    }
}

// ============================================================================
// 4. Modèle complet : Abducteur (I, S) -> (s, h) + Exécuteur aveugle (S, h) -> T
// ============================================================================

const H_DIM: usize = 4;
const C_HID: usize = 6;

#[derive(Clone)]
struct AbductiveNet {
    abd1: RelLayer,
    abd2: RelLayer,
    // Tête de graine s_t = sigmoid(b_s + sum_c w_s_diag[c]*R[t,t] + w_s_row[c]*mean_j R[t,j])
    w_s_diag: Vec<f64>,
    w_s_row: Vec<f64>,
    b_s: f64,
    // Goulot d'hypothèse h = tanh(W_h * mean_{i,j}(R) + b_h)
    w_h: Vec<f64>, // (H_DIM x C_HID)
    b_h: Vec<f64>, // (H_DIM)
    // Exécuteur aveugle : NE VOIT JAMAIS I (entrée : S, S^T, I_L, broadcast(h))
    exe1: RelLayer,
    exe2: RelLayer,
    exe3: RelLayer,
    w_t: Vec<f64>, // (C_HID) -> logits de transport Z_{t,u}
}

struct NetGrad {
    abd1: RelLayerGrad,
    abd2: RelLayerGrad,
    w_s_diag: Vec<f64>,
    w_s_row: Vec<f64>,
    b_s: f64,
    w_h: Vec<f64>,
    b_h: Vec<f64>,
    exe1: RelLayerGrad,
    exe2: RelLayerGrad,
    exe3: RelLayerGrad,
    w_t: Vec<f64>,
}

struct ForwardOut {
    objective: f64,
    seed: Vec<f64>,
    p_hat: Vec<f64>,
}

// --- Fonction libre, à placer avant impl AbductiveNet ---
fn init_vec(rng: &mut Rng, n: usize, s: f64) -> Vec<f64> {
    (0..n).map(|_| (rng.gen_f64() * 2.0 - 1.0) * s).collect()
}

impl AbductiveNet {
    fn random_init(rng: &mut Rng) -> Self {
        // Ordre d'appel identique à l'original (déterminisme préservé).
        let abd1 = RelLayer::random_init(4, C_HID, rng);
        let abd2 = RelLayer::random_init(C_HID, C_HID, rng);
        let w_s_diag = init_vec(rng, C_HID, 0.4);
        let w_s_row  = init_vec(rng, C_HID, 0.4);
        let w_h      = init_vec(rng, H_DIM * C_HID, 0.5);
        let exe1 = RelLayer::random_init(3 + H_DIM, C_HID, rng);
        let exe2 = RelLayer::random_init(C_HID, C_HID, rng);
        let exe3 = RelLayer::random_init(C_HID, C_HID, rng);
        let w_t  = init_vec(rng, C_HID, 0.8);

        Self {
            abd1,
            abd2,
            w_s_diag,
            w_s_row,
            b_s: -0.5,
            w_h,
            b_h: vec![0.0; H_DIM],
            exe1,
            exe2,
            exe3,
            w_t,
        }
    }

    fn zero_grad(&self) -> NetGrad {
        NetGrad {
            abd1: self.abd1.zero_grad(),
            abd2: self.abd2.zero_grad(),
            w_s_diag: vec![0.0; C_HID],
            w_s_row: vec![0.0; C_HID],
            b_s: 0.0,
            w_h: vec![0.0; H_DIM * C_HID],
            b_h: vec![0.0; H_DIM],
            exe1: self.exe1.zero_grad(),
            exe2: self.exe2.zero_grad(),
            exe3: self.exe3.zero_grad(),
            w_t: vec![0.0; C_HID],
        }
    }

    /// Passe avant + rétropropagation analytique exacte sur l'objectif de générativité causale J.
    fn forward_and_backward(&self, bytes: &[u8], grad: Option<&mut NetGrad>) -> ForwardOut {
        let l = bytes.len();
        let inv_l = 1.0 / (l as f64);
        let inv_l2 = 1.0 / ((l * l) as f64);
        let (id_eq, shift, shift_t, eye) = build_raw_relations(bytes);

        // --- Étage 1 : Abducteur sur (I, S, S^T, I_L) ---
        let abd_in = vec![id_eq.clone(), shift.clone(), shift_t.clone(), eye.clone()];
        let (r1, c_abd1) = self.abd1.forward(&abd_in, l);
        let (r2, c_abd2) = self.abd2.forward(&r1, l);

        // Masque de graine s \in (0, 1)^L
        let mut row_mean = vec![vec![0.0; l]; C_HID];
        for c in 0..C_HID {
            for t in 0..l {
                let mut acc = 0.0;
                for j in 0..l {
                    acc += r2[c][t * l + j];
                }
                row_mean[c][t] = acc * inv_l;
            }
        }
        let mut seed = vec![0.0; l];
        for t in 0..l {
            let mut u = self.b_s;
            for c in 0..C_HID {
                u += self.w_s_diag[c] * r2[c][t * l + t] + self.w_s_row[c] * row_mean[c][t];
            }
            seed[t] = 1.0 / (1.0 + (-u).exp());
        }

        // Goulot d'hypothèse h \in R^{H_DIM}
        let mut pool = vec![0.0; C_HID];
        for c in 0..C_HID {
            pool[c] = r2[c].iter().sum::<f64>() * inv_l2;
        }
        let mut h_vec = vec![0.0; H_DIM];
        for k in 0..H_DIM {
            let mut z = self.b_h[k];
            for c in 0..C_HID {
                z += self.w_h[k * C_HID + c] * pool[c];
            }
            h_vec[k] = z.tanh();
        }

        // --- Étage 2 : Exécuteur aveugle sur (S, S^T, I_L, h) — JAMAIS I ! ---
        let mut exe_in = Vec::with_capacity(3 + H_DIM);
        exe_in.push(shift);
        exe_in.push(shift_t);
        exe_in.push(eye);
        for k in 0..H_DIM {
            exe_in.push(vec![h_vec[k]; l * l]);
        }

        let (e1, c_exe1) = self.exe1.forward(&exe_in, l);
        let (e2, c_exe2) = self.exe2.forward(&e1, l);
        let (e3, c_exe3) = self.exe3.forward(&e2, l);

        // Logits de transport Z_{t, u} et transport masqué par la graine s_u
        let mut z_mat = vec![0.0; l * l];
        for c in 0..C_HID {
            let wt = self.w_t[c];
            for idx in 0..(l * l) {
                z_mat[idx] += wt * e3[c][idx];
            }
        }

        const EPS: f64 = 1e-8;
        let mut exp_z = vec![0.0; l * l];
        let mut denom = vec![0.0; l];
        let mut t_mat = vec![0.0; l * l];
        let mut p_hat = vec![0.0; l];
        let mut objective = 0.0;

        for t in 0..l {
            let row = t * l;
            let mut max_z = f64::NEG_INFINITY;
            for u in 0..l {
                if z_mat[row + u] > max_z {
                    max_z = z_mat[row + u];
                }
            }
            let mut d_sum = EPS;
            for u in 0..l {
                let ez = (z_mat[row + u] - max_z).exp();
                exp_z[row + u] = ez;
                d_sum += seed[u] * ez;
            }
            denom[t] = d_sum;

            let mut pt = 0.0;
            for u in 0..l {
                let tu = (seed[u] * exp_z[row + u]) / d_sum;
                t_mat[row + u] = tu;
                pt += tu * id_eq[row + u];
            }
            p_hat[t] = pt;
            objective += inv_l * (1.0 - seed[t]) * pt;
        }

        // --- Rétropropagation exacte ---
        if let Some(g) = grad {
            let mut d_seed = vec![0.0; l];
            let mut d_z = vec![0.0; l * l];

            for t in 0..l {
                let row = t * l;
                d_seed[t] -= inv_l * p_hat[t];
                let dp = inv_l * (1.0 - seed[t]);
                let inv_den = 1.0 / denom[t];
                for u in 0..l {
                    let diff = id_eq[row + u] - p_hat[t];
                    d_z[row + u] = dp * t_mat[row + u] * diff;
                    d_seed[u] += dp * (exp_z[row + u] * inv_den) * diff;
                }
            }

            // Backward tête transport -> Exécuteur
            let mut de3 = vec![vec![0.0; l * l]; C_HID];
            for c in 0..C_HID {
                let mut g_wt = 0.0;
                let wt = self.w_t[c];
                for idx in 0..(l * l) {
                    g_wt += d_z[idx] * e3[c][idx];
                    de3[c][idx] = d_z[idx] * wt;
                }
                g.w_t[c] += g_wt;
            }

            let de2 = self.exe3.backward(&de3, &c_exe3, &mut g.exe3);
            let de1 = self.exe2.backward(&de2, &c_exe2, &mut g.exe2);
            let de_in = self.exe1.backward(&de1, &c_exe1, &mut g.exe1);

            // Gradient vers le goulot h
            let mut dr2 = vec![vec![0.0; l * l]; C_HID];
            for k in 0..H_DIM {
                let dh: f64 = de_in[3 + k].iter().sum();
                let dz_h = dh * (1.0 - h_vec[k] * h_vec[k]);
                g.b_h[k] += dz_h;
                for c in 0..C_HID {
                    g.w_h[k * C_HID + c] += dz_h * pool[c];
                    let add = dz_h * self.w_h[k * C_HID + c] * inv_l2;
                    for idx in 0..(l * l) {
                        dr2[c][idx] += add;
                    }
                }
            }

            // Gradient depuis le masque de graine s -> r2
            for t in 0..l {
                let du = d_seed[t] * seed[t] * (1.0 - seed[t]);
                g.b_s += du;
                for c in 0..C_HID {
                    g.w_s_diag[c] += du * r2[c][t * l + t];
                    g.w_s_row[c] += du * row_mean[c][t];
                    dr2[c][t * l + t] += du * self.w_s_diag[c];
                    let row_add = du * self.w_s_row[c] * inv_l;
                    for j in 0..l {
                        dr2[c][t * l + j] += row_add;
                    }
                }
            }

            let dr1 = self.abd2.backward(&dr2, &c_abd2, &mut g.abd2);
            let _ = self.abd1.backward(&dr1, &c_abd1, &mut g.abd1);
        }

        ForwardOut {
            objective,
            seed,
            p_hat,
        }
    }

    fn flat_params_mut(&mut self) -> Vec<&mut f64> {
        let mut out = Vec::new();
        for l in [&mut self.abd1, &mut self.abd2, &mut self.exe1, &mut self.exe2, &mut self.exe3] {
            out.extend(l.w1.iter_mut());
            out.extend(l.w2.iter_mut());
            out.extend(l.w3.iter_mut());
            out.extend(l.b.iter_mut());
        }
        out.extend(self.w_s_diag.iter_mut());
        out.extend(self.w_s_row.iter_mut());
        out.push(&mut self.b_s);
        out.extend(self.w_h.iter_mut());
        out.extend(self.b_h.iter_mut());
        out.extend(self.w_t.iter_mut());
        out
    }
}

impl NetGrad {
    fn flat_vec(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for l in [&self.abd1, &self.abd2, &self.exe1, &self.exe2, &self.exe3] {
            out.extend_from_slice(&l.w1);
            out.extend_from_slice(&l.w2);
            out.extend_from_slice(&l.w3);
            out.extend_from_slice(&l.b);
        }
        out.extend_from_slice(&self.w_s_diag);
        out.extend_from_slice(&self.w_s_row);
        out.push(self.b_s);
        out.extend_from_slice(&self.w_h);
        out.extend_from_slice(&self.b_h);
        out.extend_from_slice(&self.w_t);
        out
    }
}

struct Adam {
    t: usize,
    m: Vec<f64>,
    v: Vec<f64>,
}

impl Adam {
    fn new(n_params: usize) -> Self {
        Self {
            t: 0,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
        }
    }

    fn step_ascent(&mut self, net: &mut AbductiveNet, grad: &NetGrad, lr: f64) {
        self.t += 1;
        let g = grad.flat_vec();
        let mut params = net.flat_params_mut();
        let b1: f64 = 0.9;
        let b2: f64 = 0.999;
        let eps: f64 = 1e-8;
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);

        // Clipping global L2
        let norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        let scale = if norm > 5.0 { 5.0 / norm } else { 1.0 };

        for i in 0..params.len() {
            let gi = g[i] * scale;
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            *params[i] += lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

// ============================================================================
// 5. Mini-PTGRS : séquences aléatoires S_256 (périodes 2 et 3, longueurs 6..9)
// ============================================================================

fn sample_periodic_s256(rng: &mut Rng, period: usize, len: usize) -> Vec<u8> {
    let mut seed_bytes = Vec::with_capacity(period);
    while seed_bytes.len() < period {
        let b = rng.gen_range(256) as u8;
        if !seed_bytes.contains(&b) {
            seed_bytes.push(b);
        }
    }
    (0..len).map(|i| seed_bytes[i % period]).collect()
}

fn inspect_case(net: &AbductiveNet, label: &str, input: &[u8]) {
    let out = net.forward_and_backward(input, None);
    let extracted: Vec<u8> = input
        .iter()
        .zip(out.seed.iter())
        .filter_map(|(&b, &s)| if s >= 0.5 { Some(b) } else { None })
        .collect();
    let extracted_str = String::from_utf8_lossy(&extracted);
    let input_str = String::from_utf8_lossy(input);
    let mask_str: String = out
        .seed
        .iter()
        .map(|&s| if s >= 0.5 { '1' } else { '0' })
        .collect();
    println!(
        "[{:<18}] entrée=\"{}\" -> graine=\"{}\" (masque={}, J={:.3})",
        label, input_str, extracted_str, mask_str, out.objective
    );
}

fn main() {
    let mut rng = Rng::from_seed(2026);
    let mut net = AbductiveNet::random_init(&mut rng);
    let n_params = net.flat_params_mut().len();
    let mut adam = Adam::new(n_params);

    println!("=== Phisharp :: Prototype 100% connexionniste (zéro axiome hardcodé) ===");
    println!("Paramètres appris : {} | Entraînement : périodes p in {{2,3}}, L in {{6,8,9}}, octets S_256 aléatoires", n_params);

    let steps = 400;
    let batch_size = 8;

    for step in 1..=steps {
        let mut grad = net.zero_grad();
        let mut avg_j = 0.0;

        for _ in 0..batch_size {
            let period = if rng.gen_range(2) == 0 { 2 } else { 3 };
            let len = if period == 2 {
                if rng.gen_range(2) == 0 { 6 } else { 8 }
            } else {
                if rng.gen_range(2) == 0 { 6 } else { 9 }
            };
            let seq = sample_periodic_s256(&mut rng, period, len);
            let out = net.forward_and_backward(&seq, Some(&mut grad));
            avg_j += out.objective;
        }
        avg_j /= batch_size as f64;

        let lr = 0.015 * (0.2 + 0.8 * (1.0 - (step as f64) / (steps as f64)));
        adam.step_ascent(&mut net, &grad, lr);

        if step % 80 == 0 || step == 1 {
            println!("  step {:>3}/{} | J moyen = {:.4}", step, steps, avg_j);
        }
    }

    println!("\n=== Évaluation OOD / Transfert (jamais vu à l'entraînement) ===");
    // 1. Alphabet ASCII inédit + longueur inédite (L=10)
    inspect_case(&net, "OOD len=10 p=2", b"ababababab");
    // 2. Même structure, autre alphabet (vérifie invariance S_256)
    inspect_case(&net, "OOD len=10 S256", b"7#7#7#7#7#");
    // 3. Période 3, longueur inédite (L=12)
    inspect_case(&net, "OOD len=12 p=3", b"xyzxyzxyzxyz");
    // 4. Période INÉDITE (p=4, jamais vue à l'entraînement)
    inspect_case(&net, "OOD p=4 INEDIT", b"abcdabcd");
    // 5. Loi INÉDITE (symétrie miroir, jamais vue à l'entraînement)
    inspect_case(&net, "OOD sym INEDIT", b"abccba");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytical_gradient_exact_finite_diff() {
        let mut rng = Rng::from_seed(123);
        let net = AbductiveNet::random_init(&mut rng);
        let seq = b"ababab";

        let mut g = net.zero_grad();
        let _ = net.forward_and_backward(seq, Some(&mut g));
        let analytic = g.flat_vec();

        let eps = 1e-6;
        // Vérification sur un échantillon couvrant toutes les couches (Abductor + Executor)
        for idx in [0, 5, 25, 80, 150, 220, analytic.len() - 1] {
            let mut np = net.clone();
            let mut nm = net.clone();
            *np.flat_params_mut()[idx] += eps;
            *nm.flat_params_mut()[idx] -= eps;
            let jp = np.forward_and_backward(seq, None).objective;
            let jm = nm.forward_and_backward(seq, None).objective;
            let fd = (jp - jm) / (2.0 * eps);
            assert!(
                (fd - analytic[idx]).abs() < 1e-5,
                "Grad mismatch idx={idx}: fd={fd}, an={}",
                analytic[idx]
            );
        }
    }

    #[test]
    fn test_exact_s256_invariance() {
        let mut rng = Rng::from_seed(999);
        let net = AbductiveNet::random_init(&mut rng);
        let o1 = net.forward_and_backward(b"ababababab", None);
        let o2 = net.forward_and_backward(b"Q9Q9Q9Q9Q9", None);
        assert!((o1.objective - o2.objective).abs() < 1e-12);
        assert_eq!(o1.seed, o2.seed);
    }
}