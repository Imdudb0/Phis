//! Primitives mathématiques, PRNG déterministe, théorie de l'information et métriques IIT 4.0.
//! Zéro dépendance externe, compatible Rust 2024 (aucun identifiant `gen`).

#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut splitmix = || -> u64 {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self {
            state: [splitmix(), splitmix(), splitmix(), splitmix()],
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[0].wrapping_add(self.state[3]))
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let v = self.next_u64() >> 11;
        (v as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    #[inline]
    pub fn usize_below(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = self.usize_below(i + 1);
            slice.swap(i, j);
        }
    }

    pub fn sample_categorical(&mut self, probs: &[f64]) -> usize {
        if probs.is_empty() {
            return 0;
        }
        let sum: f64 = probs.iter().copied().filter(|x| *x > 0.0).sum();
        if sum <= 1e-15 {
            return self.usize_below(probs.len());
        }
        let mut r = self.next_f64() * sum;
        for (i, &p) in probs.iter().enumerate() {
            if p > 0.0 {
                r -= p;
                if r <= 0.0 {
                    return i;
                }
            }
        }
        probs.len() - 1
    }
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
pub fn tanh_clamped(x: f64) -> f64 {
    x.clamp(-20.0, 20.0).tanh()
}

#[inline]
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

pub fn softmax_into(logits: &[f64], temp: f64, out: &mut [f64]) {
    let t = temp.max(1e-6);
    let max_v = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for (o, &x) in out.iter_mut().zip(logits.iter()) {
        let v = ((x - max_v) / t).exp();
        *o = v;
        sum += v;
    }
    let inv = if sum > 1e-30 { 1.0 / sum } else { 1.0 / (out.len().max(1) as f64) };
    for o in out.iter_mut() {
        *o *= inv;
    }
}

pub fn normalize_prob(v: &mut [f64]) {
    let eps = 1e-12;
    let mut sum = 0.0;
    for x in v.iter_mut() {
        if !x.is_finite() || *x < 0.0 {
            *x = 0.0;
        }
        *x += eps;
        sum += *x;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn matvec(mat: &[f64], rows: usize, cols: usize, vec: &[f64], out: &mut [f64]) {
    debug_assert_eq!(mat.len(), rows * cols);
    debug_assert_eq!(vec.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for r in 0..rows {
        let row = &mat[r * cols..(r + 1) * cols];
        out[r] = dot(row, vec);
    }
}

pub fn l2_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        (dot(a, b) / (na * nb)).clamp(-1.0, 1.0)
    }
}

pub fn entropy_bits(p: &[f64]) -> f64 {
    let mut h = 0.0;
    for &pi in p {
        if pi > 1e-15 {
            h -= pi * pi.log2();
        }
    }
    h.max(0.0)
}

pub fn kl_divergence_bits(p: &[f64], q: &[f64]) -> f64 {
    let mut kl = 0.0;
    let eps = 1e-12;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let pp = pi.max(0.0);
        if pp > eps {
            let qq = qi.max(eps);
            kl += pp * (pp / qq).log2();
        }
    }
    kl.max(0.0)
}

/// Différence Intrinsèque (Intrinsic Difference, IIT 4.0) :
/// Mesure la sélectivité causale-effet : max_x p(x) * log2(p(x) / q(x)) combinée
/// à l'information intrinsèque intégrée sur la distribution contrainte.
pub fn iit_intrinsic_difference(p: &[f64], q: &[f64]) -> f64 {
    let eps = 1e-12;
    let mut max_pointwise = 0.0_f64;
    let mut expected = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let pp = pi.max(0.0);
        if pp > eps {
            let qq = qi.max(eps);
            let ratio = (pp / qq).log2();
            if ratio > 0.0 {
                let term = pp * ratio;
                if term > max_pointwise {
                    max_pointwise = term;
                }
                expected += term;
            }
        }
    }
    // IIT 4.0 ID pure pour un état sélectif + intégration distributionnelle
    0.5 * max_pointwise + 0.5 * expected
}

/// Distance d'intégration sous partition (Earth-Mover / ID-loss entre système entier et partitionné).
pub fn iit_partition_loss(whole: &[f64], partitioned: &[f64]) -> f64 {
    let d1 = iit_intrinsic_difference(whole, partitioned);
    let d2 = kl_divergence_bits(whole, partitioned);
    // Prise de min non-compensatoire ou borne inférieure stable
    (d1 * d2).sqrt().max(0.0)
}

pub fn pearson_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }
    let mx = xs[..n].iter().sum::<f64>() / (n as f64);
    let my = ys[..n].iter().sum::<f64>() / (n as f64);
    let mut num = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        num += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom < 1e-12 { 0.0 } else { (num / denom).clamp(-1.0, 1.0) }
}

pub fn spearman_rank_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }
    let rank_vec = |vals: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
        let mut ranks = vec![0.0; n];
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && (vals[idx[j]] - vals[idx[i]]).abs() < 1e-12 {
                j += 1;
            }
            let avg_rank = 0.5 * ((i + j - 1) as f64);
            for k in i..j {
                ranks[idx[k]] = avg_rank;
            }
            i = j;
        }
        ranks
    };
    let rx = rank_vec(&xs[..n]);
    let ry = rank_vec(&ys[..n]);
    pearson_correlation(&rx, &ry)
}