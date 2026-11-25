//! Différentiation inverse sur tape, f32, zéro dépendance.
//!
//! Ce graphe de calcul ne voit JAMAIS : texte de langue, définition, E auto-décrit,
//! rendu de H en octets, intention dans la cause cachée, prior linguistique.
//! Il porte uniquement (a) la dynamique relationnelle du substrat, (b) le calcul
//! de φ_sharp et de ses termes de stabilité/résidu, (c) les logits de la politique
//! d'opérateurs abductifs. Les choix discrets d'opérateurs passent par un estimateur
//! de score-function dont la récompense est un incrément de φ_sharp (lot 4).
//!
//! Convention : tout tenseur est (r, c) à plat, ligne-major. Un vecteur est (n, 1).
//! Une distribution catégorielle est une LIGNE de (r, c) : c états ordonnés.
//! `cumsum` fait des sommes préfixes le long des colonnes, ce qui rend l'EMD 1-D
//! exacte et différentiable : EMD(P,Q) = Σ_j |CDF_P(j) − CDF_Q(j)|.

use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub r: usize,
    pub c: usize,
}

impl Shape {
    #[inline]
    pub fn n(self) -> usize {
        self.r * self.c
    }
}

#[derive(Clone, Debug)]
enum Op {
    Leaf,
    Param(usize),
    MatMul(usize, usize),
    Add(usize, usize),
    AddRow(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    MulRow(usize, usize),
    Scale(usize, f32),
    AddConst(usize, f32),
    Tanh(usize),
    Sigmoid(usize),
    Relu(usize),
    Abs(usize),
    Exp(usize),
    Log(usize),
    SoftmaxRows(usize),
    LogSoftmaxRows(usize),
    SumAll(usize),
    SumRows(usize),
    MeanAll(usize),
    GatherRows(usize, Vec<usize>),
    Cumsum(usize),
    LayerNormRows(usize),
    Min2(usize, usize),
    Max2(usize, usize),
    ConcatRows(usize, usize),
    StopGrad(usize),
}

struct Node {
    op: Op,
    shape: Shape,
    val: Vec<f32>,
}

/// Magasin de paramètres, indépendant de la tape (une tape par pas de calcul).
#[derive(Clone, Default)]
pub struct Params {
    pub shapes: Vec<Shape>,
    pub data: Vec<Vec<f32>>,
}

impl Params {
    pub fn block(&mut self, r: usize, c: usize, init: &[f32]) -> usize {
        assert_eq!(r * c, init.len(), "bloc de parametres : taille incoherente");
        self.shapes.push(Shape { r, c });
        self.data.push(init.to_vec());
        self.data.len() - 1
    }

    pub fn zeros(&mut self, r: usize, c: usize) -> usize {
        self.block(r, c, &vec![0.0f32; r * c])
    }

    pub fn glorot(&mut self, r: usize, c: usize, rng: &mut Rng) -> usize {
        let s = (6.0f32 / (r + c).max(1) as f32).sqrt();
        let v: Vec<f32> = (0..r * c).map(|_| (rng.next_f32() * 2.0 - 1.0) * s).collect();
        self.block(r, c, &v)
    }

    pub fn total(&self) -> usize {
        self.data.iter().map(|b| b.len()).sum()
    }

    pub fn nblocks(&self) -> usize {
        self.data.len()
    }
}

pub struct Tape {
    nodes: Vec<Node>,
    params: Vec<(usize, usize)>, // (nœud, bloc)
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl Tape {
    pub fn new() -> Self {
        Tape { nodes: Vec::new(), params: Vec::new() }
    }

    fn push(&mut self, op: Op, shape: Shape, val: Vec<f32>) -> usize {
        debug_assert_eq!(shape.n(), val.len());
        self.nodes.push(Node { op, shape, val });
        self.nodes.len() - 1
    }

    pub fn input(&mut self, r: usize, c: usize, data: &[f32]) -> usize {
        assert_eq!(r * c, data.len());
        self.push(Op::Leaf, Shape { r, c }, data.to_vec())
    }

    pub fn scalar(&mut self, x: f32) -> usize {
        self.push(Op::Leaf, Shape { r: 1, c: 1 }, vec![x])
    }

    pub fn param(&mut self, p: &Params, b: usize) -> usize {
        let sh = p.shapes[b];
        let v = self.push(Op::Param(b), sh, p.data[b].clone());
        self.params.push((v, b));
        v
    }

    #[inline]
    pub fn shape(&self, v: usize) -> Shape {
        self.nodes[v].shape
    }

    #[inline]
    pub fn val(&self, v: usize) -> &[f32] {
        &self.nodes[v].val
    }

    #[inline]
    pub fn at(&self, v: usize, i: usize) -> f32 {
        self.nodes[v].val[i]
    }

    #[inline]
    pub fn scalar_of(&self, v: usize) -> f32 {
        self.nodes[v].val[0]
    }

    // ---------------- opérations ----------------

    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
        let sa = self.shape(a);
        let sb = self.shape(b);
        assert_eq!(sa.c, sb.r, "matmul : dimensions internes");
        let (n, k, m) = (sa.r, sa.c, sb.c);
        let av = self.val(a).to_vec();
        let bv = self.val(b).to_vec();
        let mut out = vec![0.0f32; n * m];
        for i in 0..n {
            for t in 0..k {
                let x = av[i * k + t];
                if x == 0.0 {
                    continue;
                }
                let row = &bv[t * m..(t + 1) * m];
                for j in 0..m {
                    out[i * m + j] += x * row[j];
                }
            }
        }
        self.push(Op::MatMul(a, b), Shape { r: n, c: m }, out)
    }

    fn same_shape(&self, a: usize, b: usize) {
        assert_eq!(self.shape(a), self.shape(b), "forme incompatible");
    }

    pub fn add(&mut self, a: usize, b: usize) -> usize {
        self.same_shape(a, b);
        let bv = self.val(b).to_vec();
        let out: Vec<f32> = self.val(a).iter().zip(bv.iter()).map(|(x, y)| x + y).collect();
        let sh = self.shape(a);
        self.push(Op::Add(a, b), sh, out)
    }

    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        self.same_shape(a, b);
        let bv = self.val(b).to_vec();
        let out: Vec<f32> = self.val(a).iter().zip(bv.iter()).map(|(x, y)| x - y).collect();
        let sh = self.shape(a);
        self.push(Op::Sub(a, b), sh, out)
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        self.same_shape(a, b);
        let bv = self.val(b).to_vec();
        let out: Vec<f32> = self.val(a).iter().zip(bv.iter()).map(|(x, y)| x * y).collect();
        let sh = self.shape(a);
        self.push(Op::Mul(a, b), sh, out)
    }

    /// (r,c) + (1,c), diffusion sur les lignes.
    pub fn add_row(&mut self, a: usize, b: usize) -> usize {
        let sa = self.shape(a);
        let sb = self.shape(b);
        assert_eq!(sb.r, 1);
        assert_eq!(sa.c, sb.c);
        let bv = self.val(b).to_vec();
        let mut out = self.val(a).to_vec();
        for i in 0..sa.r {
            for j in 0..sa.c {
                out[i * sa.c + j] += bv[j];
            }
        }
        self.push(Op::AddRow(a, b), sa, out)
    }

    /// (r,c) * (1,c), gain par colonne.
    pub fn mul_row(&mut self, a: usize, b: usize) -> usize {
        let sa = self.shape(a);
        let sb = self.shape(b);
        assert_eq!(sb.r, 1);
        assert_eq!(sa.c, sb.c);
        let bv = self.val(b).to_vec();
        let mut out = self.val(a).to_vec();
        for i in 0..sa.r {
            for j in 0..sa.c {
                out[i * sa.c + j] *= bv[j];
            }
        }
        self.push(Op::MulRow(a, b), sa, out)
    }

    pub fn scale(&mut self, a: usize, k: f32) -> usize {
        let out: Vec<f32> = self.val(a).iter().map(|x| x * k).collect();
        let sh = self.shape(a);
        self.push(Op::Scale(a, k), sh, out)
    }

    pub fn neg(&mut self, a: usize) -> usize {
        self.scale(a, -1.0)
    }

    pub fn add_const(&mut self, a: usize, k: f32) -> usize {
        let out: Vec<f32> = self.val(a).iter().map(|x| x + k).collect();
        let sh = self.shape(a);
        self.push(Op::AddConst(a, k), sh, out)
    }

    pub fn tanh(&mut self, a: usize) -> usize {
        let out: Vec<f32> = self.val(a).iter().map(|x| x.tanh()).collect();
        let sh = self.shape(a);
        self.push(Op::Tanh(a), sh, out)
    }

    pub fn sigmoid(&mut self, a: usize) -> usize {
        let out: Vec<f32> = self.val(a).iter().map(|x| 1.0 / (1.0 + (-x.clamp(-40.0, 40.0)).exp())).collect();
        let sh = self.shape(a);
        self.push(Op::Sigmoid(a), sh, out)
    }

    pub fn relu(&mut self, a: usize) -> usize {
        let out: Vec<f32> = self.val(a).iter().map(|x| if *x > 0.0 { *x } else { 0.0 }).collect();
        let sh = self.shape(a);
        self.push(Op::Relu(a), sh, out)
    }

    pub fn abs(&mut self, a: usize) -> usize {
        let out: Vec<f32> = self.val(a).iter().map(|x| x.abs()).collect();
        let sh = self.shape(a);
        self.push(Op::Abs(a), sh, out)
    }

    pub fn exp(&mut self, a: usize) -> usize {
        let out: Vec<f32> = self.val(a).iter().map(|x| x.clamp(-30.0, 30.0).exp()).collect();
        let sh = self.shape(a);
        self.push(Op::Exp(a), sh, out)
    }

    pub fn log(&mut self, a: usize) -> usize {
        let out: Vec<f32> = self.val(a).iter().map(|x| x.max(1e-12).ln()).collect();
        let sh = self.shape(a);
        self.push(Op::Log(a), sh, out)
    }

    pub fn softmax_rows(&mut self, a: usize) -> usize {
        let sh = self.shape(a);
        let av = self.val(a).to_vec();
        let mut out = vec![0.0f32; sh.n()];
        for i in 0..sh.r {
            let row = &av[i * sh.c..(i + 1) * sh.c];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0f32;
            for j in 0..sh.c {
                let e = (row[j] - mx).clamp(-30.0, 30.0).exp();
                out[i * sh.c + j] = e;
                s += e;
            }
            let inv = 1.0 / s.max(1e-12);
            for j in 0..sh.c {
                out[i * sh.c + j] *= inv;
            }
        }
        self.push(Op::SoftmaxRows(a), sh, out)
    }

    pub fn log_softmax_rows(&mut self, a: usize) -> usize {
        let sh = self.shape(a);
        let av = self.val(a).to_vec();
        let mut out = vec![0.0f32; sh.n()];
        for i in 0..sh.r {
            let row = &av[i * sh.c..(i + 1) * sh.c];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = row.iter().map(|x| (x - mx).clamp(-30.0, 30.0).exp()).sum();
            let lse = mx + s.max(1e-12).ln();
            for j in 0..sh.c {
                out[i * sh.c + j] = row[j] - lse;
            }
        }
        self.push(Op::LogSoftmaxRows(a), sh, out)
    }

    pub fn sum_all(&mut self, a: usize) -> usize {
        let s: f32 = self.val(a).iter().sum();
        self.push(Op::SumAll(a), Shape { r: 1, c: 1 }, vec![s])
    }

    pub fn mean_all(&mut self, a: usize) -> usize {
        let n = self.shape(a).n().max(1);
        let s: f32 = self.val(a).iter().sum();
        self.push(Op::MeanAll(a), Shape { r: 1, c: 1 }, vec![s / n as f32])
    }

    pub fn sum_rows(&mut self, a: usize) -> usize {
        let sh = self.shape(a);
        let mut out = vec![0.0f32; sh.r];
        for i in 0..sh.r {
            out[i] = self.val(a)[i * sh.c..(i + 1) * sh.c].iter().sum();
        }
        self.push(Op::SumRows(a), Shape { r: sh.r, c: 1 }, out)
    }

    pub fn gather_rows(&mut self, a: usize, idx: &[usize]) -> usize {
        let sh = self.shape(a);
        let mut out = vec![0.0f32; idx.len() * sh.c];
        for (t, &i) in idx.iter().enumerate() {
            assert!(i < sh.r);
            out[t * sh.c..(t + 1) * sh.c].copy_from_slice(&self.val(a)[i * sh.c..(i + 1) * sh.c]);
        }
        self.push(Op::GatherRows(a, idx.to_vec()), Shape { r: idx.len(), c: sh.c }, out)
    }

    /// Sommes préfixes le long des colonnes, ligne par ligne.
    pub fn cumsum(&mut self, a: usize) -> usize {
        let sh = self.shape(a);
        let mut out = vec![0.0f32; sh.n()];
        for i in 0..sh.r {
            let mut acc = 0.0f32;
            for j in 0..sh.c {
                acc += self.val(a)[i * sh.c + j];
                out[i * sh.c + j] = acc;
            }
        }
        self.push(Op::Cumsum(a), sh, out)
    }

    pub fn layer_norm_rows(&mut self, a: usize) -> usize {
        let sh = self.shape(a);
        let av = self.val(a).to_vec();
        let mut out = vec![0.0f32; sh.n()];
        let c = sh.c.max(1) as f32;
        for i in 0..sh.r {
            let row = &av[i * sh.c..(i + 1) * sh.c];
            let mu: f32 = row.iter().sum::<f32>() / c;
            let var: f32 = row.iter().map(|x| (x - mu) * (x - mu)).sum::<f32>() / c;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for j in 0..sh.c {
                out[i * sh.c + j] = (row[j] - mu) * inv;
            }
        }
        self.push(Op::LayerNormRows(a), sh, out)
    }

    /// min scalaire. Sous-gradient vers l'argmin (égalité -> a).
    /// C'est l'opérateur de non-compensation : min(cause,effet), min sur MIP,
    /// min(Générativité, Discrimination, Projection).
    pub fn min2(&mut self, a: usize, b: usize) -> usize {
        assert_eq!(self.shape(a).n(), 1);
        assert_eq!(self.shape(b).n(), 1);
        let x = self.scalar_of(a);
        let y = self.scalar_of(b);
        self.push(Op::Min2(a, b), Shape { r: 1, c: 1 }, vec![x.min(y)])
    }

    pub fn max2(&mut self, a: usize, b: usize) -> usize {
        assert_eq!(self.shape(a).n(), 1);
        assert_eq!(self.shape(b).n(), 1);
        let x = self.scalar_of(a);
        let y = self.scalar_of(b);
        self.push(Op::Max2(a, b), Shape { r: 1, c: 1 }, vec![x.max(y)])
    }

    pub fn concat_rows(&mut self, a: usize, b: usize) -> usize {
        let sa = self.shape(a);
        let sb = self.shape(b);
        assert_eq!(sa.c, sb.c);
        let mut out = self.val(a).to_vec();
        out.extend_from_slice(self.val(b));
        self.push(Op::ConcatRows(a, b), Shape { r: sa.r + sb.r, c: sa.c }, out)
    }

    pub fn stop_grad(&mut self, a: usize) -> usize {
        let sh = self.shape(a);
        let v = self.val(a).to_vec();
        self.push(Op::StopGrad(a), sh, v)
    }

    // ---------------- distances ----------------

    pub fn dot(&mut self, a: usize, b: usize) -> usize {
        let m = self.mul(a, b);
        self.sum_all(m)
    }

    pub fn l1(&mut self, a: usize, b: usize) -> usize {
        let d = self.sub(a, b);
        let ad = self.abs(d);
        self.sum_all(ad)
    }

    /// EMD 1-D exacte, sommé sur les lignes. Espaces d'états ordonnés.
    pub fn emd(&mut self, a: usize, b: usize) -> usize {
        let d = self.sub(a, b);
        let cd = self.cumsum(d);
        let ac = self.abs(cd);
        self.sum_all(ac)
    }

    // ---------------- rétropropagation ----------------

    /// Retourne un gradient par bloc de `Params` (taille `nblocks`), les blocs
    /// non touchés restant à zéro. `loss` doit être scalaire.
    pub fn backward(&mut self, loss: usize, nblocks: usize) -> Vec<Vec<f32>> {
        assert_eq!(self.shape(loss).n(), 1, "la perte doit etre scalaire");
        let mut g: Vec<Vec<f32>> = self.nodes.iter().map(|n| vec![0.0f32; n.shape.n()]).collect();
        g[loss][0] = 1.0;

        for i in (0..self.nodes.len()).rev() {
            let gi = g[i].clone();
            if gi.iter().all(|&x| x == 0.0) {
                continue;
            }
            let op = self.nodes[i].op.clone();
            let sh = self.nodes[i].shape;
            match op {
                Op::Leaf | Op::Param(_) | Op::StopGrad(_) => {}

                Op::MatMul(a, b) => {
                    let sa = self.nodes[a].shape;
                    let sb = self.nodes[b].shape;
                    let av = self.nodes[a].val.clone();
                    let bv = self.nodes[b].val.clone();
                    let (n, k, m) = (sa.r, sa.c, sb.c);
                    for x in 0..n {
                        for j in 0..m {
                            let gg = gi[x * m + j];
                            if gg == 0.0 {
                                continue;
                            }
                            for t in 0..k {
                                g[a][x * k + t] += gg * bv[t * m + j];
                                g[b][t * m + j] += gg * av[x * k + t];
                            }
                        }
                    }
                }

                Op::Add(a, b) => {
                    for t in 0..gi.len() {
                        g[a][t] += gi[t];
                        g[b][t] += gi[t];
                    }
                }

                Op::Sub(a, b) => {
                    for t in 0..gi.len() {
                        g[a][t] += gi[t];
                        g[b][t] -= gi[t];
                    }
                }

                Op::Mul(a, b) => {
                    let av = self.nodes[a].val.clone();
                    let bv = self.nodes[b].val.clone();
                    for t in 0..gi.len() {
                        g[a][t] += gi[t] * bv[t];
                        g[b][t] += gi[t] * av[t];
                    }
                }

                Op::AddRow(a, b) => {
                    let c = sh.c;
                    for x in 0..sh.r {
                        for j in 0..c {
                            let gg = gi[x * c + j];
                            g[a][x * c + j] += gg;
                            g[b][j] += gg;
                        }
                    }
                }

                Op::MulRow(a, b) => {
                    let c = sh.c;
                    let av = self.nodes[a].val.clone();
                    let bv = self.nodes[b].val.clone();
                    for x in 0..sh.r {
                        for j in 0..c {
                            let gg = gi[x * c + j];
                            g[a][x * c + j] += gg * bv[j];
                            g[b][j] += gg * av[x * c + j];
                        }
                    }
                }

                Op::Scale(a, k) => {
                    for t in 0..gi.len() {
                        g[a][t] += k * gi[t];
                    }
                }

                Op::AddConst(a, _) => {
                    for t in 0..gi.len() {
                        g[a][t] += gi[t];
                    }
                }

                Op::Tanh(a) => {
                    let y = self.nodes[i].val.clone();
                    for t in 0..gi.len() {
                        g[a][t] += gi[t] * (1.0 - y[t] * y[t]);
                    }
                }

                Op::Sigmoid(a) => {
                    let y = self.nodes[i].val.clone();
                    for t in 0..gi.len() {
                        g[a][t] += gi[t] * y[t] * (1.0 - y[t]);
                    }
                }

                Op::Relu(a) => {
                    let av = self.nodes[a].val.clone();
                    for t in 0..gi.len() {
                        if av[t] > 0.0 {
                            g[a][t] += gi[t];
                        }
                    }
                }

                Op::Abs(a) => {
                    let av = self.nodes[a].val.clone();
                    for t in 0..gi.len() {
                        g[a][t] += gi[t] * if av[t] >= 0.0 { 1.0 } else { -1.0 };
                    }
                }

                Op::Exp(a) => {
                    let y = self.nodes[i].val.clone();
                    for t in 0..gi.len() {
                        g[a][t] += gi[t] * y[t];
                    }
                }

                Op::Log(a) => {
                    let av = self.nodes[a].val.clone();
                    for t in 0..gi.len() {
                        g[a][t] += gi[t] / av[t].max(1e-12);
                    }
                }

                Op::SoftmaxRows(a) => {
                    let y = self.nodes[i].val.clone();
                    let c = sh.c;
                    for x in 0..sh.r {
                        let mut s = 0.0f32;
                        for j in 0..c {
                            s += gi[x * c + j] * y[x * c + j];
                        }
                        for j in 0..c {
                            g[a][x * c + j] += y[x * c + j] * (gi[x * c + j] - s);
                        }
                    }
                }

                Op::LogSoftmaxRows(a) => {
                    let y = self.nodes[i].val.clone();
                    let c = sh.c;
                    for x in 0..sh.r {
                        let s: f32 = gi[x * c..(x + 1) * c].iter().sum();
                        for j in 0..c {
                            let p = y[x * c + j].clamp(-30.0, 0.0).exp();
                            g[a][x * c + j] += gi[x * c + j] - p * s;
                        }
                    }
                }

                Op::SumAll(a) => {
                    for t in 0..g[a].len() {
                        g[a][t] += gi[0];
                    }
                }

                Op::MeanAll(a) => {
                    let n = g[a].len().max(1) as f32;
                    for t in 0..g[a].len() {
                        g[a][t] += gi[0] / n;
                    }
                }

                Op::SumRows(a) => {
                    let c = sh.c;
                    for x in 0..sh.r {
                        for j in 0..c {
                            g[a][x * c + j] += gi[x];
                        }
                    }
                }

                Op::GatherRows(a, idx) => {
                    let c = sh.c;
                    for (t, &src) in idx.iter().enumerate() {
                        for j in 0..c {
                            g[a][src * c + j] += gi[t * c + j];
                        }
                    }
                }

                Op::Cumsum(a) => {
                    let c = sh.c;
                    for x in 0..sh.r {
                        let mut acc = 0.0f32;
                        for j in (0..c).rev() {
                            acc += gi[x * c + j];
                            g[a][x * c + j] += acc;
                        }
                    }
                }

                Op::LayerNormRows(a) => {
                    let c = sh.c;
                    let av = self.nodes[a].val.clone();
                    let y = self.nodes[i].val.clone();
                    let cf = c.max(1) as f32;
                    for x in 0..sh.r {
                        let row = &av[x * c..(x + 1) * c];
                        let mu: f32 = row.iter().sum::<f32>() / cf;
                        let var: f32 = row.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / cf;
                        let inv = 1.0 / (var + 1e-5).sqrt();
                        let mg: f32 = gi[x * c..(x + 1) * c].iter().sum::<f32>() / cf;
                        let mgy: f32 = (0..c).map(|j| gi[x * c + j] * y[x * c + j]).sum::<f32>() / cf;
                        for j in 0..c {
                            g[a][x * c + j] += inv * (gi[x * c + j] - mg - y[x * c + j] * mgy);
                        }
                    }
                }

                Op::Min2(a, b) => {
                    let x = self.nodes[a].val[0];
                    let y = self.nodes[b].val[0];
                    if x <= y {
                        g[a][0] += gi[0];
                    } else {
                        g[b][0] += gi[0];
                    }
                }

                Op::Max2(a, b) => {
                    let x = self.nodes[a].val[0];
                    let y = self.nodes[b].val[0];
                    if x >= y {
                        g[a][0] += gi[0];
                    } else {
                        g[b][0] += gi[0];
                    }
                }

                Op::ConcatRows(a, b) => {
                    let na = self.nodes[a].shape.n();
                    for t in 0..na {
                        g[a][t] += gi[t];
                    }
                    for t in na..gi.len() {
                        g[b][t - na] += gi[t];
                    }
                }
            }
        }

        let mut out = vec![Vec::new(); nblocks];
        for b in 0..nblocks {
            out[b] = Vec::new();
        }
        for &(node, block) in self.params.iter() {
            if block >= nblocks {
                continue;
            }
            if out[block].is_empty() {
                out[block] = vec![0.0f32; self.nodes[node].shape.n()];
            }
            for t in 0..out[block].len() {
                out[block][t] += g[node][t];
            }
        }
        for b in 0..nblocks {
            if out[b].is_empty() {
                out[b] = vec![0.0f32; 0];
            }
        }
        out
    }
}

pub fn accumulate(dst: &mut [Vec<f32>], src: &[Vec<f32>]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        if d.len() != s.len() {
            continue;
        }
        for t in 0..d.len() {
            d[t] += s[t];
        }
    }
}

/// Adam avec écrêtage global de norme. Zéro dépendance.
pub struct Adam {
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: u64,
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
    pub clip: f32,
}

impl Adam {
    pub fn new(p: &Params, lr: f32) -> Self {
        Adam {
            m: p.data.iter().map(|b| vec![0.0f32; b.len()]).collect(),
            v: p.data.iter().map(|b| vec![0.0f32; b.len()]).collect(),
            t: 0,
            lr,
            b1: 0.9,
            b2: 0.999,
            eps: 1e-8,
            clip: 5.0,
        }
    }

    /// `grads` peut contenir des blocs vides (non touchés) : ils sont ignorés.
    pub fn step(&mut self, p: &mut Params, grads: &[Vec<f32>]) {
        self.t += 1;
        let mut norm = 0.0f32;
        for gb in grads.iter() {
            for &x in gb.iter() {
                norm += x * x;
            }
        }
        let norm = norm.sqrt();
        let scale = if norm > self.clip && norm > 0.0 { self.clip / norm } else { 1.0 };
        let c1 = 1.0 - self.b1.powi(self.t as i32);
        let c2 = 1.0 - self.b2.powi(self.t as i32);
        for b in 0..p.data.len() {
            if b >= grads.len() || grads[b].len() != p.data[b].len() {
                continue;
            }
            for t in 0..p.data[b].len() {
                let g = grads[b][t] * scale;
                self.m[b][t] = self.b1 * self.m[b][t] + (1.0 - self.b1) * g;
                self.v[b][t] = self.b2 * self.v[b][t] + (1.0 - self.b2) * g * g;
                let mh = self.m[b][t] / c1;
                let vh = self.v[b][t] / c2;
                p.data[b][t] -= self.lr * mh / (vh.sqrt() + self.eps);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Contrôle numérique du tape : gradient de sum((W x)^2) par différences finies.
    #[test]
    fn grad_check_matmul() {
        let mut p = Params::default();
        let w = p.block(2, 2, &[0.3, -0.7, 0.5, 0.2]);
        let x = [0.4f32, -0.9];
        let f = |p: &Params| -> f32 {
            let mut t = Tape::new();
            let wv = t.param(p, w);
            let xv = t.input(2, 1, &x);
            let y = t.matmul(wv, xv);
            let s = t.mul(y, y);
            let l = t.sum_all(s);
            t.scalar_of(l)
        };
        let mut t = Tape::new();
        let wv = t.param(&p, w);
        let xv = t.input(2, 1, &x);
        let y = t.matmul(wv, xv);
        let s = t.mul(y, y);
        let l = t.sum_all(s);
        let g = t.backward(l, p.nblocks());
        let h = 1e-3;
        for k in 0..4 {
            let save = p.data[w][k];
            p.data[w][k] = save + h;
            let a = f(&p);
            p.data[w][k] = save - h;
            let b = f(&p);
            p.data[w][k] = save;
            let num = (a - b) / (2.0 * h);
            assert!((num - g[w][k]).abs() < 1e-2, "bloc {k}: {num} vs {}", g[w][k]);
        }
    }

    #[test]
    fn emd_nulle_sur_distribution_identique() {
        let mut t = Tape::new();
        let a = t.input(1, 4, &[0.1, 0.2, 0.3, 0.4]);
        let b = t.input(1, 4, &[0.1, 0.2, 0.3, 0.4]);
        let d = t.emd(a, b);
        assert!(t.scalar_of(d).abs() < 1e-6);
    }

    #[test]
    fn adam_descend() {
        let mut p = Params::default();
        let b = p.block(1, 1, &[1.0]);
        let mut opt = Adam::new(&p, 0.05);
        for _ in 0..200 {
            let g = vec![vec![2.0 * p.data[b][0]]];
            opt.step(&mut p, &g);
        }
        assert!(p.data[b][0].abs() < 0.05, "Adam doit descendre : x = {}", p.data[b][0]);
    }
}