// Moteur de différentiation inverse (reverse-mode autograd) sur tenseurs 1D/2D.
// Zéro dépendance externe. Permet le même code en inférence pure et en entraînement exact.

#[derive(Clone)]
enum Op {
    Leaf,
    Slice {
        x: usize,
        offset: usize,
        len: usize,
    },
    Stack(Vec<usize>),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Scale {
        x: usize,
        c: f64,
    },
    AddScalar {
        x: usize,
        c: f64,
    },
    ScalarMul {
        vec: usize,
        scalar: usize,
    },
    MatVec {
        mat: usize,
        vec: usize,
        rows: usize,
        cols: usize,
    },
    Dot(usize, usize),
    WeightedSum {
        weights: usize,
        vecs: Vec<usize>,
    },
    Tanh(usize),
    Sigmoid(usize),
    RmsNorm {
        x: usize,
        eps: f64,
        denom: f64,
    },
    Softmax(usize),
    CrossEntropyLogits {
        logits: usize,
        targets: Vec<f64>,
        probs: Vec<f64>,
        t_sum: f64,
    },
    SmoothL1Half {
        a: usize,
        b: usize,
        eps: f64,
    },
    SmoothNorm {
        x: usize,
        eps: f64,
        raw_norm: f64,
    },
    SmoothDist {
        a: usize,
        b: usize,
        eps: f64,
        raw_norm: f64,
    },
    SmoothAbsSum {
        x: usize,
        eps: f64,
    },
    Clamp {
        x: usize,
        lo: f64,
        hi: f64,
    },
    Min(usize, usize),
    Max(usize, usize),
    Div(usize, usize),
    Pow {
        x: usize,
        p: f64,
    },
}

struct Node {
    val: Vec<f64>,
    op: Op,
}

pub(crate) struct Tape {
    nodes: Vec<Node>,
    track_grad: bool,
}

impl Tape {
    pub(crate) fn new(track_grad: bool) -> Self {
        Self {
            nodes: Vec::with_capacity(512),
            track_grad,
        }
    }

    #[inline]
    fn push(&mut self, val: Vec<f64>, op: Op) -> usize {
        let id = self.nodes.len();
        let stored_op = if self.track_grad { op } else { Op::Leaf };
        self.nodes.push(Node { val, op: stored_op });
        id
    }

    #[inline]
    pub(crate) fn val(&self, id: usize) -> &[f64] {
        &self.nodes[id].val
    }

    #[inline]
    pub(crate) fn scalar(&self, id: usize) -> f64 {
        self.nodes[id].val[0]
    }

    pub(crate) fn leaf(&mut self, val: Vec<f64>) -> usize {
        self.push(val, Op::Leaf)
    }

    pub(crate) fn leaf_scalar(&mut self, v: f64) -> usize {
        self.push(vec![v], Op::Leaf)
    }

    pub(crate) fn slice(&mut self, x: usize, offset: usize, len: usize) -> usize {
        let v = self.nodes[x].val[offset..offset + len].to_vec();
        self.push(v, Op::Slice { x, offset, len })
    }

    pub(crate) fn stack(&mut self, inputs: Vec<usize>) -> usize {
        let total: usize = inputs.iter().map(|&id| self.nodes[id].val.len()).sum();
        let mut out = Vec::with_capacity(total);
        for &id in &inputs {
            out.extend_from_slice(&self.nodes[id].val);
        }
        self.push(out, Op::Stack(inputs))
    }

    pub(crate) fn add(&mut self, a: usize, b: usize) -> usize {
        let va = &self.nodes[a].val;
        let vb = &self.nodes[b].val;
        let out: Vec<f64> = va.iter().zip(vb.iter()).map(|(&x, &y)| x + y).collect();
        self.push(out, Op::Add(a, b))
    }

    pub(crate) fn sub(&mut self, a: usize, b: usize) -> usize {
        let va = &self.nodes[a].val;
        let vb = &self.nodes[b].val;
        let out: Vec<f64> = va.iter().zip(vb.iter()).map(|(&x, &y)| x - y).collect();
        self.push(out, Op::Sub(a, b))
    }

    pub(crate) fn mul(&mut self, a: usize, b: usize) -> usize {
        let va = &self.nodes[a].val;
        let vb = &self.nodes[b].val;
        let out: Vec<f64> = va.iter().zip(vb.iter()).map(|(&x, &y)| x * y).collect();
        self.push(out, Op::Mul(a, b))
    }

    pub(crate) fn scale(&mut self, x: usize, c: f64) -> usize {
        let out: Vec<f64> = self.nodes[x].val.iter().map(|&v| v * c).collect();
        self.push(out, Op::Scale { x, c })
    }

    pub(crate) fn add_scalar(&mut self, x: usize, c: f64) -> usize {
        let out: Vec<f64> = self.nodes[x].val.iter().map(|&v| v + c).collect();
        self.push(out, Op::AddScalar { x, c })
    }

    pub(crate) fn scalar_mul(&mut self, vec: usize, scalar: usize) -> usize {
        let s = self.nodes[scalar].val[0];
        let out: Vec<f64> = self.nodes[vec].val.iter().map(|&v| v * s).collect();
        self.push(out, Op::ScalarMul { vec, scalar })
    }

    pub(crate) fn mat_vec(&mut self, mat: usize, vec: usize, rows: usize, cols: usize) -> usize {
        let m = &self.nodes[mat].val;
        let x = &self.nodes[vec].val;
        let mut out = vec![0.0_f64; rows];
        for r in 0..rows {
            let base = r * cols;
            let mut acc = 0.0_f64;
            for c in 0..cols {
                acc += m[base + c] * x[c];
            }
            out[r] = acc;
        }
        self.push(
            out,
            Op::MatVec {
                mat,
                vec,
                rows,
                cols,
            },
        )
    }

    pub(crate) fn dot(&mut self, a: usize, b: usize) -> usize {
        let va = &self.nodes[a].val;
        let vb = &self.nodes[b].val;
        let mut acc = 0.0_f64;
        for (&x, &y) in va.iter().zip(vb.iter()) {
            acc += x * y;
        }
        self.push(vec![acc], Op::Dot(a, b))
    }

    pub(crate) fn weighted_sum(&mut self, weights: usize, vecs: Vec<usize>) -> usize {
        let w = &self.nodes[weights].val;
        let dim = self.nodes[vecs[0]].val.len();
        let mut out = vec![0.0_f64; dim];
        for (k, &vid) in vecs.iter().enumerate() {
            let wk = w[k];
            let vk = &self.nodes[vid].val;
            for d in 0..dim {
                out[d] += wk * vk[d];
            }
        }
        self.push(out, Op::WeightedSum { weights, vecs })
    }

    pub(crate) fn tanh(&mut self, x: usize) -> usize {
        let out: Vec<f64> = self.nodes[x].val.iter().map(|&v| v.tanh()).collect();
        self.push(out, Op::Tanh(x))
    }

    pub(crate) fn sigmoid(&mut self, x: usize) -> usize {
        let out: Vec<f64> = self.nodes[x]
            .val
            .iter()
            .map(|&v| 1.0_f64 / (1.0_f64 + (-v.clamp(-20.0_f64, 20.0_f64)).exp()))
            .collect();
        self.push(out, Op::Sigmoid(x))
    }

    pub(crate) fn rms_norm(&mut self, x: usize, eps: f64) -> usize {
        let vx = &self.nodes[x].val;
        let n = vx.len().max(1) as f64;
        let mean_sq = vx.iter().map(|&v| v * v).sum::<f64>() / n;
        let denom = (mean_sq + eps).sqrt();
        let out: Vec<f64> = vx.iter().map(|&v| v / denom).collect();
        self.push(out, Op::RmsNorm { x, eps, denom })
    }

    pub(crate) fn softmax(&mut self, x: usize) -> usize {
        let vx = &self.nodes[x].val;
        let max_l = vx.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut out = vec![0.0_f64; vx.len()];
        let mut sum_e = 0.0_f64;
        for (o, &l) in out.iter_mut().zip(vx.iter()) {
            let e = if l < -1e8_f64 {
                0.0_f64
            } else {
                (l - max_l).exp()
            };
            *o = e;
            sum_e += e;
        }
        let inv = 1.0_f64 / sum_e.max(1e-15_f64);
        for o in &mut out {
            *o *= inv;
        }
        self.push(out, Op::Softmax(x))
    }

    /// Entropie croisée numériquement exacte depuis les logits : gradient = (sum t) * p - t.
    pub(crate) fn cross_entropy_logits(&mut self, logits: usize, targets: Vec<f64>) -> usize {
        let vz = &self.nodes[logits].val;
        let max_l = vz.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut exps = vec![0.0_f64; vz.len()];
        let mut sum_e = 0.0_f64;
        for (e, &l) in exps.iter_mut().zip(vz.iter()) {
            let v = (l - max_l).exp();
            *e = v;
            sum_e += v;
        }
        let log_z = max_l + sum_e.max(1e-15_f64).ln();
        let inv_sum = 1.0_f64 / sum_e.max(1e-15_f64);
        let mut probs = vec![0.0_f64; vz.len()];
        let mut loss = 0.0_f64;
        let mut t_sum = 0.0_f64;
        for i in 0..vz.len() {
            probs[i] = exps[i] * inv_sum;
            let ti = targets[i];
            if ti > 0.0_f64 {
                loss -= ti * (vz[i] - log_z);
                t_sum += ti;
            }
        }
        self.push(
            vec![loss],
            Op::CrossEntropyLogits {
                logits,
                targets,
                probs,
                t_sum,
            },
        )
    }

    /// Distance TVD lissée : 0.5 * sum_i (sqrt((a_i - b_i)^2 + eps) - sqrt(eps)).
    pub(crate) fn smooth_l1_half(&mut self, a: usize, b: usize) -> usize {
        let eps = 1e-10_f64;
        let s_eps = eps.sqrt();
        let va = &self.nodes[a].val;
        let vb = &self.nodes[b].val;
        let mut acc = 0.0_f64;
        for (&x, &y) in va.iter().zip(vb.iter()) {
            let d = x - y;
            acc += (d * d + eps).sqrt() - s_eps;
        }
        self.push(vec![0.5_f64 * acc], Op::SmoothL1Half { a, b, eps })
    }

    /// Norme euclidienne lissée valant exactement 0.0 en x = 0.
    pub(crate) fn smooth_norm(&mut self, x: usize) -> usize {
        let eps = 1e-10_f64;
        let sq: f64 = self.nodes[x].val.iter().map(|&v| v * v).sum();
        let raw_norm = (sq + eps).sqrt();
        self.push(vec![raw_norm - eps.sqrt()], Op::SmoothNorm { x, eps, raw_norm })
    }

    /// Distance euclidienne lissée valant exactement 0.0 quand a == b.
    pub(crate) fn smooth_dist(&mut self, a: usize, b: usize) -> usize {
        let eps = 1e-10_f64;
        let va = &self.nodes[a].val;
        let vb = &self.nodes[b].val;
        let mut sq = 0.0_f64;
        for (&x, &y) in va.iter().zip(vb.iter()) {
            let d = x - y;
            sq += d * d;
        }
        let raw_norm = (sq + eps).sqrt();
        self.push(
            vec![raw_norm - eps.sqrt()],
            Op::SmoothDist {
                a,
                b,
                eps,
                raw_norm,
            },
        )
    }

    /// Norme L1 lissée (utilisée pour la pénalité MDL).
    pub(crate) fn smooth_abs_sum(&mut self, x: usize) -> usize {
        let eps = 1e-6_f64;
        let s_eps = eps.sqrt();
        let acc: f64 = self.nodes[x]
            .val
            .iter()
            .map(|&v| (v * v + eps).sqrt() - s_eps)
            .sum();
        self.push(vec![acc], Op::SmoothAbsSum { x, eps })
    }

    pub(crate) fn clamp(&mut self, x: usize, lo: f64, hi: f64) -> usize {
        let out: Vec<f64> = self.nodes[x].val.iter().map(|&v| v.clamp(lo, hi)).collect();
        self.push(out, Op::Clamp { x, lo, hi })
    }

    pub(crate) fn min_scalar(&mut self, a: usize, b: usize) -> usize {
        let va = self.nodes[a].val[0];
        let vb = self.nodes[b].val[0];
        self.push(vec![va.min(vb)], Op::Min(a, b))
    }

    pub(crate) fn max_scalar(&mut self, a: usize, b: usize) -> usize {
        let va = self.nodes[a].val[0];
        let vb = self.nodes[b].val[0];
        self.push(vec![va.max(vb)], Op::Max(a, b))
    }

    pub(crate) fn div_scalar(&mut self, a: usize, b: usize) -> usize {
        let va = self.nodes[a].val[0];
        let vb = self.nodes[b].val[0];
        let safe_b = if vb.abs() < 1e-9_f64 {
            if vb >= 0.0_f64 { 1e-9_f64 } else { -1e-9_f64 }
        } else {
            vb
        };
        self.push(vec![va / safe_b], Op::Div(a, b))
    }

    pub(crate) fn pow_scalar(&mut self, x: usize, p: f64) -> usize {
        let vx = self.nodes[x].val[0];
        let val = if vx <= 1e-12_f64 {
            0.0_f64
        } else {
            vx.powf(p)
        };
        self.push(vec![val], Op::Pow { x, p })
    }

    /// Rétropropagation inverse exacte depuis un nœud scalaire `root`.
    /// Retourne le gradient accumulé pour le nœud `target_leaf` (ex. les paramètres).
    pub(crate) fn backward(&self, root: usize, target_leaf: usize) -> Vec<f64> {
        let n = self.nodes.len();
        let mut grads: Vec<Vec<f64>> = self
            .nodes
            .iter()
            .map(|nd| vec![0.0_f64; nd.val.len()])
            .collect();
        grads[root][0] = 1.0_f64;

        for idx in (0..=root).rev() {
            let g_out = grads[idx].clone();
            if g_out.iter().all(|&g| g == 0.0_f64) {
                continue;
            }
            match &self.nodes[idx].op {
                Op::Leaf => {}
                Op::Slice { x, offset, len } => {
                    let gx = &mut grads[*x];
                    for i in 0..*len {
                        gx[*offset + i] += g_out[i];
                    }
                }
                Op::Stack(inputs) => {
                    let mut off = 0usize;
                    for &inp in inputs {
                        let len = self.nodes[inp].val.len();
                        let gi = &mut grads[inp];
                        for i in 0..len {
                            gi[i] += g_out[off + i];
                        }
                        off += len;
                    }
                }
                Op::Add(a, b) => {
                    for (ga, &g) in grads[*a].iter_mut().zip(g_out.iter()) {
                        *ga += g;
                    }
                    for (gb, &g) in grads[*b].iter_mut().zip(g_out.iter()) {
                        *gb += g;
                    }
                }
                Op::Sub(a, b) => {
                    for (ga, &g) in grads[*a].iter_mut().zip(g_out.iter()) {
                        *ga += g;
                    }
                    for (gb, &g) in grads[*b].iter_mut().zip(g_out.iter()) {
                        *gb -= g;
                    }
                }
                Op::Mul(a, b) => {
                    let va = &self.nodes[*a].val;
                    let vb = &self.nodes[*b].val;
                    for i in 0..g_out.len() {
                        grads[*a][i] += g_out[i] * vb[i];
                        grads[*b][i] += g_out[i] * va[i];
                    }
                }
                Op::Scale { x, c } => {
                    for (gx, &g) in grads[*x].iter_mut().zip(g_out.iter()) {
                        *gx += g * (*c);
                    }
                }
                Op::AddScalar { x, .. } => {
                    for (gx, &g) in grads[*x].iter_mut().zip(g_out.iter()) {
                        *gx += g;
                    }
                }
                Op::ScalarMul { vec, scalar } => {
                    let s = self.nodes[*scalar].val[0];
                    let vv = &self.nodes[*vec].val;
                    let mut gs = 0.0_f64;
                    for i in 0..g_out.len() {
                        grads[*vec][i] += g_out[i] * s;
                        gs += g_out[i] * vv[i];
                    }
                    grads[*scalar][0] += gs;
                }
                Op::MatVec {
                    mat,
                    vec,
                    rows,
                    cols,
                } => {
                    let vm = &self.nodes[*mat].val;
                    let vx = &self.nodes[*vec].val;
                    for r in 0..*rows {
                        let gr = g_out[r];
                        if gr == 0.0_f64 {
                            continue;
                        }
                        let base = r * (*cols);
                        for c in 0..*cols {
                            grads[*mat][base + c] += gr * vx[c];
                            grads[*vec][c] += gr * vm[base + c];
                        }
                    }
                }
                Op::Dot(a, b) => {
                    let g = g_out[0];
                    let va = &self.nodes[*a].val;
                    let vb = &self.nodes[*b].val;
                    for i in 0..va.len() {
                        grads[*a][i] += g * vb[i];
                        grads[*b][i] += g * va[i];
                    }
                }
                Op::WeightedSum { weights, vecs } => {
                    let vw = &self.nodes[*weights].val;
                    for (k, &vid) in vecs.iter().enumerate() {
                        let wk = vw[k];
                        let vk = &self.nodes[vid].val;
                        let mut gw = 0.0_f64;
                        for d in 0..g_out.len() {
                            gw += g_out[d] * vk[d];
                            grads[vid][d] += wk * g_out[d];
                        }
                        grads[*weights][k] += gw;
                    }
                }
                Op::Tanh(x) => {
                    let vy = &self.nodes[idx].val;
                    for i in 0..g_out.len() {
                        grads[*x][i] += g_out[i] * (1.0_f64 - vy[i] * vy[i]);
                    }
                }
                Op::Sigmoid(x) => {
                    let vy = &self.nodes[idx].val;
                    for i in 0..g_out.len() {
                        grads[*x][i] += g_out[i] * vy[i] * (1.0_f64 - vy[i]);
                    }
                }
                Op::RmsNorm { x, denom, .. } => {
                    let vy = &self.nodes[idx].val;
                    let inv_d = 1.0_f64 / (*denom);
                    let inv_n = 1.0_f64 / (vy.len().max(1) as f64);
                    let mut dot_gy = 0.0_f64;
                    for i in 0..vy.len() {
                        dot_gy += g_out[i] * vy[i];
                    }
                    for i in 0..vy.len() {
                        grads[*x][i] += inv_d * (g_out[i] - vy[i] * dot_gy * inv_n);
                    }
                }
                Op::Softmax(x) => {
                    let vy = &self.nodes[idx].val;
                    let mut dot_gy = 0.0_f64;
                    for i in 0..vy.len() {
                        dot_gy += g_out[i] * vy[i];
                    }
                    for i in 0..vy.len() {
                        grads[*x][i] += vy[i] * (g_out[i] - dot_gy);
                    }
                }
                Op::CrossEntropyLogits {
                    logits,
                    targets,
                    probs,
                    t_sum,
                } => {
                    let g = g_out[0];
                    for i in 0..probs.len() {
                        grads[*logits][i] += g * ((*t_sum) * probs[i] - targets[i]);
                    }
                }
                Op::SmoothL1Half { a, b, eps } => {
                    let g = 0.5_f64 * g_out[0];
                    let va = &self.nodes[*a].val;
                    let vb = &self.nodes[*b].val;
                    for i in 0..va.len() {
                        let d = va[i] - vb[i];
                        let s = g * d / (d * d + *eps).sqrt();
                        grads[*a][i] += s;
                        grads[*b][i] -= s;
                    }
                }
                Op::SmoothNorm { x, raw_norm, .. } => {
                    let g = g_out[0] / (*raw_norm);
                    let vx = &self.nodes[*x].val;
                    for i in 0..vx.len() {
                        grads[*x][i] += g * vx[i];
                    }
                }
                Op::SmoothDist { a, b, raw_norm, .. } => {
                    let g = g_out[0] / (*raw_norm);
                    let va = &self.nodes[*a].val;
                    let vb = &self.nodes[*b].val;
                    for i in 0..va.len() {
                        let s = g * (va[i] - vb[i]);
                        grads[*a][i] += s;
                        grads[*b][i] -= s;
                    }
                }
                Op::SmoothAbsSum { x, eps } => {
                    let g = g_out[0];
                    let vx = &self.nodes[*x].val;
                    for i in 0..vx.len() {
                        grads[*x][i] += g * vx[i] / (vx[i] * vx[i] + *eps).sqrt();
                    }
                }
                Op::Clamp { x, lo, hi } => {
                    let vx = &self.nodes[*x].val;
                    for i in 0..vx.len() {
                        if vx[i] > *lo && vx[i] < *hi {
                            grads[*x][i] += g_out[i];
                        }
                    }
                }
                Op::Min(a, b) => {
                    let va = self.nodes[*a].val[0];
                    let vb = self.nodes[*b].val[0];
                    let g = g_out[0];
                    if va < vb {
                        grads[*a][0] += g;
                    } else if vb < va {
                        grads[*b][0] += g;
                    } else {
                        grads[*a][0] += 0.5_f64 * g;
                        grads[*b][0] += 0.5_f64 * g;
                    }
                }
                Op::Max(a, b) => {
                    let va = self.nodes[*a].val[0];
                    let vb = self.nodes[*b].val[0];
                    let g = g_out[0];
                    if va > vb {
                        grads[*a][0] += g;
                    } else if vb > va {
                        grads[*b][0] += g;
                    } else {
                        grads[*a][0] += 0.5_f64 * g;
                        grads[*b][0] += 0.5_f64 * g;
                    }
                }
                Op::Div(a, b) => {
                    let va = self.nodes[*a].val[0];
                    let vb = self.nodes[*b].val[0];
                    let safe_b = if vb.abs() < 1e-9_f64 {
                        if vb >= 0.0_f64 { 1e-9_f64 } else { -1e-9_f64 }
                    } else {
                        vb
                    };
                    let g = g_out[0];
                    grads[*a][0] += g / safe_b;
                    grads[*b][0] -= g * va / (safe_b * safe_b);
                }
                Op::Pow { x, p } => {
                    let vx = self.nodes[*x].val[0];
                    if vx > 1e-12_f64 {
                        grads[*x][0] += g_out[0] * (*p) * vx.powf(*p - 1.0_f64);
                    }
                }
            }
        }

        let _ = n;
        grads.swap_remove(target_leaf)
    }
}
