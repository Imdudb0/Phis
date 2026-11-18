use crate::rng::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParamId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValId(pub usize);

#[derive(Clone, Debug)]
pub struct TensorData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl TensorData {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "TensorData length {} does not match shape {:?}",
            data.len(),
            shape
        );
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct ParamStore {
    pub names: Vec<String>,
    pub shapes: Vec<Vec<usize>>,
    pub values: Vec<Vec<f32>>,
    pub grads: Vec<Vec<f32>>,
}

impl ParamStore {
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            shapes: Vec::new(),
            values: Vec::new(),
            grads: Vec::new(),
        }
    }

    pub fn add_param(&mut self, name: impl Into<String>, data: Vec<f32>, shape: Vec<usize>) -> ParamId {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "param shape mismatch");
        let id = self.values.len();
        self.names.push(name.into());
        self.shapes.push(shape);
        self.values.push(data);
        self.grads.push(vec![0.0; expected]);
        ParamId(id)
    }

    pub fn add_zeros(&mut self, name: impl Into<String>, shape: Vec<usize>) -> ParamId {
        let len: usize = shape.iter().product();
        self.add_param(name, vec![0.0; len], shape)
    }

    pub fn add_normal(&mut self, name: impl Into<String>, shape: Vec<usize>, std: f32, rng: &mut Rng) -> ParamId {
        let len: usize = shape.iter().product();
        let data = (0..len).map(|_| rng.normal_f32(0.0, std)).collect();
        self.add_param(name, data, shape)
    }

    pub fn add_xavier(&mut self, name: impl Into<String>, rows: usize, cols: usize, rng: &mut Rng) -> ParamId {
        let std = (2.0 / (rows + cols) as f32).sqrt();
        self.add_normal(name, vec![rows, cols], std, rng)
    }

    pub fn zero_grad(&mut self) {
        for g in &mut self.grads {
            g.fill(0.0);
        }
    }

    pub fn accum_grad(&mut self, pid: ParamId, grad: &[f32]) {
        let dst = &mut self.grads[pid.0];
        assert_eq!(dst.len(), grad.len());
        for (d, &s) in dst.iter_mut().zip(grad.iter()) {
            *d += s;
        }
    }

    pub fn total_param_count(&self) -> usize {
        self.values.iter().map(|v| v.len()).sum()
    }
}

impl Default for ParamStore {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub step_count: u64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamW {
    pub fn new(store: &ParamStore, lr: f32, weight_decay: f32, max_grad_norm: f32) -> Self {
        let m = store.values.iter().map(|p| vec![0.0; p.len()]).collect();
        let v = store.values.iter().map(|p| vec![0.0; p.len()]).collect();
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            max_grad_norm,
            step_count: 0,
            m,
            v,
        }
    }

    pub fn step(&mut self, store: &mut ParamStore) -> f32 {
        self.step_count += 1;

        let mut sq_norm = 0.0f64;
        for g in &store.grads {
            for &val in g {
                sq_norm += (val as f64) * (val as f64);
            }
        }
        let grad_norm = sq_norm.sqrt() as f32;
        let clip_scale = if self.max_grad_norm > 0.0 && grad_norm > self.max_grad_norm {
            self.max_grad_norm / (grad_norm + 1e-12)
        } else {
            1.0
        };

        let b1_t = 1.0 - self.beta1.powi(self.step_count as i32);
        let b2_t = 1.0 - self.beta2.powi(self.step_count as i32);

        for i in 0..store.values.len() {
            let val = &mut store.values[i];
            let grad = &store.grads[i];
            let m = &mut self.m[i];
            let v = &mut self.v[i];
            let decay = if store.shapes[i].len() >= 2 {
                self.weight_decay
            } else {
                0.0
            };

            for j in 0..val.len() {
                let g = grad[j] * clip_scale;
                if !g.is_finite() {
                    continue;
                }
                if decay != 0.0 {
                    val[j] -= self.lr * decay * val[j];
                }
                m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * g;
                v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * g * g;
                let m_hat = m[j] / b1_t;
                let v_hat = v[j] / b2_t;
                val[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }

        grad_norm
    }
}

#[derive(Clone, Debug)]
enum BroadcastKind {
    SameShape,
    RightScalar,
    LeftScalar,
    RightTrailing { outer: usize, inner: usize },
    LeftTrailing { outer: usize, inner: usize },
    RightLeading { rows: usize, cols: usize },
    LeftLeading { rows: usize, cols: usize },
    General {
        a_strides: Vec<usize>,
        b_strides: Vec<usize>,
        out_shape: Vec<usize>,
    },
}

#[derive(Clone, Debug)]
enum Op {
    Const,
    Param(ParamId),
    Add {
        a: ValId,
        b: ValId,
        kind: BroadcastKind,
    },
    Sub {
        a: ValId,
        b: ValId,
        kind: BroadcastKind,
    },
    Mul {
        a: ValId,
        b: ValId,
        kind: BroadcastKind,
    },
    Neg(ValId),
    Scale {
        a: ValId,
        s: f32,
    },
    Tanh(ValId),
    Sigmoid(ValId),
    Relu(ValId),
    Silu(ValId),
    Exp(ValId),
    SafeLog {
        a: ValId,
        eps: f32,
    },
    SafeSqrt {
        a: ValId,
        eps: f32,
    },
    Square(ValId),
    MatMul {
        a: ValId,
        b: ValId,
        m: usize,
        k: usize,
        n: usize,
    },
    BatchedMatMul {
        a: ValId,
        b: ValId,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    },
    Transpose2D {
        a: ValId,
        rows: usize,
        cols: usize,
    },
    TransposeLast2 {
        a: ValId,
        batch: usize,
        rows: usize,
        cols: usize,
    },
    RmsNormRows {
        a: ValId,
        rows: usize,
        cols: usize,
        inv_rms: Vec<f32>,
    },
    SoftmaxRows {
        a: ValId,
        rows: usize,
        cols: usize,
    },
    MaskedLogSoftmaxRows {
        a: ValId,
        rows: usize,
        cols: usize,
        mask: Vec<f32>,
    },
    SumAll(ValId),
    MeanAll(ValId),
    SumAxis0 {
        a: ValId,
        rows: usize,
        cols: usize,
    },
    SumAxis1 {
        a: ValId,
        rows: usize,
        cols: usize,
    },
    SumBatch {
        a: ValId,
        batch: usize,
        stride: usize,
    },
    Reshape {
        a: ValId,
    },
    ConcatCols {
        a: ValId,
        b: ValId,
        rows: usize,
        cols_a: usize,
        cols_b: usize,
    },
    ConcatRows {
        a: ValId,
        b: ValId,
        rows_a: usize,
        cols: usize,
    },
    SliceRows {
        a: ValId,
        start: usize,
        len: usize,
        cols: usize,
    },
    SliceCols {
        a: ValId,
        rows: usize,
        total_cols: usize,
        start: usize,
        len: usize,
    },
    GatherRows {
        a: ValId,
        indices: Vec<usize>,
        cols: usize,
    },
    ScatterAddRows {
        a: ValId,
        indices: Vec<usize>,
        cols: usize,
    },
    NllLoss {
        log_probs: ValId,
        targets: Vec<usize>,
        cols: usize,
    },
    BceWithLogits {
        logits: ValId,
        targets: Vec<f32>,
    },
    BceWithProbs {
        probs: ValId,
        targets: Vec<f32>,
        eps: f32,
    },
}

pub struct Tape {
    vals: Vec<TensorData>,
    grads: Vec<Vec<f32>>,
    ops: Vec<Op>,
}

fn broadcast_spec(a: &[usize], b: &[usize]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let ndim = a.len().max(b.len());
    let mut pa = vec![1usize; ndim];
    let mut pb = vec![1usize; ndim];
    for i in 0..a.len() {
        pa[ndim - a.len() + i] = a[i];
    }
    for i in 0..b.len() {
        pb[ndim - b.len() + i] = b[i];
    }
    let mut out_shape = vec![0usize; ndim];
    for k in 0..ndim {
        assert!(
            pa[k] == pb[k] || pa[k] == 1 || pb[k] == 1,
            "incompatible broadcast shapes {:?} and {:?}",
            a,
            b
        );
        out_shape[k] = pa[k].max(pb[k]);
    }

    let make_strides = |padded: &[usize]| -> Vec<usize> {
        let mut strides = vec![0usize; ndim];
        let mut stride = 1usize;
        for k in (0..ndim).rev() {
            if padded[k] == 1 {
                strides[k] = 0;
            } else {
                strides[k] = stride;
                stride *= padded[k];
            }
        }
        strides
    };

    (out_shape, make_strides(&pa), make_strides(&pb))
}

fn is_suffix_without_leading_ones(big: &[usize], small: &[usize]) -> Option<(usize, usize)> {
    let mut s = small;
    while s.first() == Some(&1) {
        s = &s[1..];
    }
    if s.is_empty() || s.len() > big.len() {
        return None;
    }
    let split = big.len() - s.len();
    if &big[split..] == s {
        let rows: usize = big[..split].iter().product();
        let cols: usize = s.iter().product();
        if rows > 0 && cols > 0 {
            return Some((rows, cols));
        }
    }
    None
}

fn is_prefix_without_trailing_ones(big: &[usize], small: &[usize]) -> Option<(usize, usize)> {
    let mut s = small;
    while s.last() == Some(&1) {
        s = &s[..s.len() - 1];
    }
    if s.is_empty() || s.len() > big.len() {
        return None;
    }
    if &big[..s.len()] == s {
        let outer: usize = s.iter().product();
        let inner: usize = big[s.len()..].iter().product();
        if outer > 0 && inner > 0 {
            return Some((outer, inner));
        }
    }
    None
}

fn classify_broadcast(sa: &[usize], sb: &[usize]) -> (Vec<usize>, BroadcastKind) {
    let (out_shape, a_strides, b_strides) = broadcast_spec(sa, sb);
    let a_len: usize = sa.iter().product();
    let b_len: usize = sb.iter().product();

    let kind = if sa == sb {
        BroadcastKind::SameShape
    } else if b_len == 1 {
        BroadcastKind::RightScalar
    } else if a_len == 1 {
        BroadcastKind::LeftScalar
    } else if let Some((outer, inner)) = is_prefix_without_trailing_ones(sa, sb)
        .filter(|&(o, i)| o == b_len && o * i == a_len)
    {
        BroadcastKind::RightTrailing { outer, inner }
    } else if let Some((outer, inner)) = is_prefix_without_trailing_ones(sb, sa)
        .filter(|&(o, i)| o == a_len && o * i == b_len)
    {
        BroadcastKind::LeftTrailing { outer, inner }
    } else if let Some((rows, cols)) = is_suffix_without_leading_ones(sa, sb)
        .filter(|&(r, c)| c == b_len && r * c == a_len)
    {
        BroadcastKind::RightLeading { rows, cols }
    } else if let Some((rows, cols)) = is_suffix_without_leading_ones(sb, sa)
        .filter(|&(r, c)| c == a_len && r * c == b_len)
    {
        BroadcastKind::LeftLeading { rows, cols }
    } else {
        BroadcastKind::General {
            a_strides,
            b_strides,
            out_shape: out_shape.clone(),
        }
    };

    (out_shape, kind)
}

#[inline]
fn next_multi_idx(coords: &mut [usize], shape: &[usize]) {
    for k in (0..shape.len()).rev() {
        coords[k] += 1;
        if coords[k] < shape[k] {
            return;
        }
        coords[k] = 0;
    }
}

#[inline]
fn offset_from_coords(coords: &[usize], strides: &[usize]) -> usize {
    let mut off = 0usize;
    for i in 0..coords.len() {
        off += coords[i] * strides[i];
    }
    off
}

#[inline]
fn borrow_two_mut<T>(slice: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert_ne!(i, j);
    if i < j {
        let (left, right) = slice.split_at_mut(j);
        (&mut left[i], &mut right[0])
    } else {
        let (left, right) = slice.split_at_mut(i);
        (&mut right[0], &mut left[j])
    }
}

impl Tape {
    pub fn new() -> Self {
        Self {
            vals: Vec::new(),
            grads: Vec::new(),
            ops: Vec::new(),
        }
    }

    fn push_node(&mut self, data: Vec<f32>, shape: Vec<usize>, op: Op) -> ValId {
        let len = data.len();
        let id = self.vals.len();
        self.vals.push(TensorData::new(data, shape));
        self.grads.push(vec![0.0; len]);
        self.ops.push(op);
        ValId(id)
    }

    pub fn tensor(&mut self, data: Vec<f32>, shape: Vec<usize>) -> ValId {
        self.push_node(data, shape, Op::Const)
    }

    pub fn scalar(&mut self, v: f32) -> ValId {
        self.push_node(vec![v], vec![1], Op::Const)
    }

    pub fn param(&mut self, store: &ParamStore, pid: ParamId) -> ValId {
        let data = store.values[pid.0].clone();
        let shape = store.shapes[pid.0].clone();
        self.push_node(data, shape, Op::Param(pid))
    }

    pub fn val(&self, id: ValId) -> &TensorData {
        &self.vals[id.0]
    }

    pub fn data(&self, id: ValId) -> &[f32] {
        &self.vals[id.0].data
    }

    pub fn shape(&self, id: ValId) -> &[usize] {
        &self.vals[id.0].shape
    }

    pub fn grad(&self, id: ValId) -> &[f32] {
        &self.grads[id.0]
    }

    pub fn add(&mut self, a: ValId, b: ValId) -> ValId {
        let (out_shape, kind) = classify_broadcast(&self.vals[a.0].shape, &self.vals[b.0].shape);
        let da = &self.vals[a.0].data;
        let db = &self.vals[b.0].data;
        let out_len: usize = out_shape.iter().product();
        let mut out = vec![0.0f32; out_len];

        match &kind {
            BroadcastKind::SameShape => {
                for i in 0..out_len {
                    out[i] = da[i] + db[i];
                }
            }
            BroadcastKind::RightScalar => {
                let s = db[0];
                for i in 0..out_len {
                    out[i] = da[i] + s;
                }
            }
            BroadcastKind::LeftScalar => {
                let s = da[0];
                for i in 0..out_len {
                    out[i] = s + db[i];
                }
            }
            BroadcastKind::RightTrailing { outer, inner } => {
                for c in 0..*outer {
                    let s = db[c];
                    let dst = &mut out[c * inner..(c + 1) * inner];
                    let src = &da[c * inner..(c + 1) * inner];
                    for j in 0..*inner {
                        dst[j] = src[j] + s;
                    }
                }
            }
            BroadcastKind::LeftTrailing { outer, inner } => {
                for c in 0..*outer {
                    let s = da[c];
                    let dst = &mut out[c * inner..(c + 1) * inner];
                    let src = &db[c * inner..(c + 1) * inner];
                    for j in 0..*inner {
                        dst[j] = s + src[j];
                    }
                }
            }
            BroadcastKind::RightLeading { rows, cols } => {
                for r in 0..*rows {
                    let dst = &mut out[r * cols..(r + 1) * cols];
                    let src = &da[r * cols..(r + 1) * cols];
                    for j in 0..*cols {
                        dst[j] = src[j] + db[j];
                    }
                }
            }
            BroadcastKind::LeftLeading { rows, cols } => {
                for r in 0..*rows {
                    let dst = &mut out[r * cols..(r + 1) * cols];
                    let src = &db[r * cols..(r + 1) * cols];
                    for j in 0..*cols {
                        dst[j] = da[j] + src[j];
                    }
                }
            }
            BroadcastKind::General {
                a_strides,
                b_strides,
                out_shape: os,
            } => {
                let mut coords = vec![0usize; os.len()];
                for o in &mut out {
                    let ia = offset_from_coords(&coords, a_strides);
                    let ib = offset_from_coords(&coords, b_strides);
                    *o = da[ia] + db[ib];
                    next_multi_idx(&mut coords, os);
                }
            }
        }
        self.push_node(out, out_shape, Op::Add { a, b, kind })
    }

    pub fn sub(&mut self, a: ValId, b: ValId) -> ValId {
        let (out_shape, kind) = classify_broadcast(&self.vals[a.0].shape, &self.vals[b.0].shape);
        let da = &self.vals[a.0].data;
        let db = &self.vals[b.0].data;
        let out_len: usize = out_shape.iter().product();
        let mut out = vec![0.0f32; out_len];

        match &kind {
            BroadcastKind::SameShape => {
                for i in 0..out_len {
                    out[i] = da[i] - db[i];
                }
            }
            BroadcastKind::RightScalar => {
                let s = db[0];
                for i in 0..out_len {
                    out[i] = da[i] - s;
                }
            }
            BroadcastKind::LeftScalar => {
                let s = da[0];
                for i in 0..out_len {
                    out[i] = s - db[i];
                }
            }
            BroadcastKind::RightTrailing { outer, inner } => {
                for c in 0..*outer {
                    let s = db[c];
                    let dst = &mut out[c * inner..(c + 1) * inner];
                    let src = &da[c * inner..(c + 1) * inner];
                    for j in 0..*inner {
                        dst[j] = src[j] - s;
                    }
                }
            }
            BroadcastKind::LeftTrailing { outer, inner } => {
                for c in 0..*outer {
                    let s = da[c];
                    let dst = &mut out[c * inner..(c + 1) * inner];
                    let src = &db[c * inner..(c + 1) * inner];
                    for j in 0..*inner {
                        dst[j] = s - src[j];
                    }
                }
            }
            BroadcastKind::RightLeading { rows, cols } => {
                for r in 0..*rows {
                    let dst = &mut out[r * cols..(r + 1) * cols];
                    let src = &da[r * cols..(r + 1) * cols];
                    for j in 0..*cols {
                        dst[j] = src[j] - db[j];
                    }
                }
            }
            BroadcastKind::LeftLeading { rows, cols } => {
                for r in 0..*rows {
                    let dst = &mut out[r * cols..(r + 1) * cols];
                    let src = &db[r * cols..(r + 1) * cols];
                    for j in 0..*cols {
                        dst[j] = da[j] - src[j];
                    }
                }
            }
            BroadcastKind::General {
                a_strides,
                b_strides,
                out_shape: os,
            } => {
                let mut coords = vec![0usize; os.len()];
                for o in &mut out {
                    let ia = offset_from_coords(&coords, a_strides);
                    let ib = offset_from_coords(&coords, b_strides);
                    *o = da[ia] - db[ib];
                    next_multi_idx(&mut coords, os);
                }
            }
        }
        self.push_node(out, out_shape, Op::Sub { a, b, kind })
    }

    pub fn mul(&mut self, a: ValId, b: ValId) -> ValId {
        let (out_shape, kind) = classify_broadcast(&self.vals[a.0].shape, &self.vals[b.0].shape);
        let da = &self.vals[a.0].data;
        let db = &self.vals[b.0].data;
        let out_len: usize = out_shape.iter().product();
        let mut out = vec![0.0f32; out_len];

        match &kind {
            BroadcastKind::SameShape => {
                for i in 0..out_len {
                    out[i] = da[i] * db[i];
                }
            }
            BroadcastKind::RightScalar => {
                let s = db[0];
                for i in 0..out_len {
                    out[i] = da[i] * s;
                }
            }
            BroadcastKind::LeftScalar => {
                let s = da[0];
                for i in 0..out_len {
                    out[i] = s * db[i];
                }
            }
            BroadcastKind::RightTrailing { outer, inner } => {
                for c in 0..*outer {
                    let s = db[c];
                    let dst = &mut out[c * inner..(c + 1) * inner];
                    let src = &da[c * inner..(c + 1) * inner];
                    for j in 0..*inner {
                        dst[j] = src[j] * s;
                    }
                }
            }
            BroadcastKind::LeftTrailing { outer, inner } => {
                for c in 0..*outer {
                    let s = da[c];
                    let dst = &mut out[c * inner..(c + 1) * inner];
                    let src = &db[c * inner..(c + 1) * inner];
                    for j in 0..*inner {
                        dst[j] = s * src[j];
                    }
                }
            }
            BroadcastKind::RightLeading { rows, cols } => {
                for r in 0..*rows {
                    let dst = &mut out[r * cols..(r + 1) * cols];
                    let src = &da[r * cols..(r + 1) * cols];
                    for j in 0..*cols {
                        dst[j] = src[j] * db[j];
                    }
                }
            }
            BroadcastKind::LeftLeading { rows, cols } => {
                for r in 0..*rows {
                    let dst = &mut out[r * cols..(r + 1) * cols];
                    let src = &db[r * cols..(r + 1) * cols];
                    for j in 0..*cols {
                        dst[j] = da[j] * src[j];
                    }
                }
            }
            BroadcastKind::General {
                a_strides,
                b_strides,
                out_shape: os,
            } => {
                let mut coords = vec![0usize; os.len()];
                for o in &mut out {
                    let ia = offset_from_coords(&coords, a_strides);
                    let ib = offset_from_coords(&coords, b_strides);
                    *o = da[ia] * db[ib];
                    next_multi_idx(&mut coords, os);
                }
            }
        }
        self.push_node(out, out_shape, Op::Mul { a, b, kind })
    }

    pub fn neg(&mut self, a: ValId) -> ValId {
        let out: Vec<f32> = self.vals[a.0].data.iter().map(|&x| -x).collect();
        let shape = self.vals[a.0].shape.clone();
        self.push_node(out, shape, Op::Neg(a))
    }

    pub fn scale(&mut self, a: ValId, s: f32) -> ValId {
        let out: Vec<f32> = self.vals[a.0].data.iter().map(|&x| x * s).collect();
        let shape = self.vals[a.0].shape.clone();
        self.push_node(out, shape, Op::Scale { a, s })
    }

    pub fn tanh(&mut self, a: ValId) -> ValId {
        let out: Vec<f32> = self.vals[a.0].data.iter().map(|&x| x.tanh()).collect();
        let shape = self.vals[a.0].shape.clone();
        self.push_node(out, shape, Op::Tanh(a))
    }

    pub fn sigmoid(&mut self, a: ValId) -> ValId {
        let out: Vec<f32> = self.vals[a.0]
            .data
            .iter()
            .map(|&x| {
                if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let ex = x.exp();
                    ex / (1.0 + ex)
                }
            })
            .collect();
        let shape = self.vals[a.0].shape.clone();
        self.push_node(out, shape, Op::Sigmoid(a))
    }

    pub fn relu(&mut self, a: ValId) -> ValId {
        let out: Vec<f32> = self.vals[a.0].data.iter().map(|&x| x.max(0.0)).collect();
        let shape = self.vals[a.0].shape.clone();
        self.push_node(out, shape, Op::Relu(a))
    }

    pub fn silu(&mut self, a: ValId) -> ValId {
        let out: Vec<f32> = self.vals[a.0]
            .data
            .iter()
            .map(|&x| {
                let sig = if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let ex = x.exp();
                    ex / (1.0 + ex)
                };
                x * sig
            })
            .collect();
        let shape = self.vals[a.0].shape.clone();
        self.push_node(out, shape, Op::Silu(a))
    }

    pub fn exp(&mut self, a: ValId) -> ValId {
        let out: Vec<f32> = self.vals[a.0].data.iter().map(|&x| x.clamp(-60.0, 60.0).exp()).collect();
        let shape = self.vals[a.0].shape.clone();
        self.push_node(out, shape, Op::Exp(a))
    }

    pub fn safe_log(&mut self, a: ValId, eps: f32) -> ValId {
        let out: Vec<f32> = self.vals[a.0].data.iter().map(|&x| x.max(eps).ln()).collect();
        let shape = self.vals[a.0].shape.clone();
        self.push_node(out, shape, Op::SafeLog { a, eps })
    }

    pub fn safe_sqrt(&mut self, a: ValId, eps: f32) -> ValId {
        let out: Vec<f32> = self.vals[a.0].data.iter().map(|&x| (x.max(0.0) + eps).sqrt()).collect();
        let shape = self.vals[a.0].shape.clone();
        self.push_node(out, shape, Op::SafeSqrt { a, eps })
    }

    pub fn square(&mut self, a: ValId) -> ValId {
        let out: Vec<f32> = self.vals[a.0].data.iter().map(|&x| x * x).collect();
        let shape = self.vals[a.0].shape.clone();
        self.push_node(out, shape, Op::Square(a))
    }

    pub fn matmul(&mut self, a: ValId, b: ValId) -> ValId {
        let sa = &self.vals[a.0].shape;
        let sb = &self.vals[b.0].shape;
        assert_eq!(sa.len(), 2, "matmul a must be 2D, got {:?}", sa);
        assert_eq!(sb.len(), 2, "matmul b must be 2D, got {:?}", sb);
        let (m, k) = (sa[0], sa[1]);
        let (k2, n) = (sb[0], sb[1]);
        assert_eq!(k, k2, "matmul inner dim mismatch: {} vs {}", k, k2);

        let da = &self.vals[a.0].data;
        let db = &self.vals[b.0].data;
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            let a_row = &da[i * k..(i + 1) * k];
            let out_row = &mut out[i * n..(i + 1) * n];
            for p in 0..k {
                let a_ip = a_row[p];
                let b_row = &db[p * n..(p + 1) * n];
                for j in 0..n {
                    out_row[j] += a_ip * b_row[j];
                }
            }
        }
        self.push_node(out, vec![m, n], Op::MatMul { a, b, m, k, n })
    }

    pub fn batched_matmul(&mut self, a: ValId, b: ValId) -> ValId {
        let sa = &self.vals[a.0].shape;
        let sb = &self.vals[b.0].shape;
        assert_eq!(sa.len(), 3, "batched_matmul a must be 3D, got {:?}", sa);
        assert_eq!(sb.len(), 3, "batched_matmul b must be 3D, got {:?}", sb);
        let (batch, m, k) = (sa[0], sa[1], sa[2]);
        let (batch2, k2, n) = (sb[0], sb[1], sb[2]);
        assert_eq!(batch, batch2, "batch mismatch: {} vs {}", batch, batch2);
        assert_eq!(k, k2, "batched_matmul inner dim mismatch: {} vs {}", k, k2);

        let da = &self.vals[a.0].data;
        let db = &self.vals[b.0].data;
        let mut out = vec![0.0f32; batch * m * n];

        if n == 1 {
            for bi in 0..batch {
                let a_base = &da[bi * m * k..(bi + 1) * m * k];
                let b_vec = &db[bi * k..(bi + 1) * k];
                let o_vec = &mut out[bi * m..(bi + 1) * m];
                for i in 0..m {
                    let a_row = &a_base[i * k..(i + 1) * k];
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc += a_row[p] * b_vec[p];
                    }
                    o_vec[i] = acc;
                }
            }
        } else if k == 1 {
            for bi in 0..batch {
                let a_vec = &da[bi * m..(bi + 1) * m];
                let b_vec = &db[bi * n..(bi + 1) * n];
                let o_base = &mut out[bi * m * n..(bi + 1) * m * n];
                for i in 0..m {
                    let ai = a_vec[i];
                    let o_row = &mut o_base[i * n..(i + 1) * n];
                    for j in 0..n {
                        o_row[j] = ai * b_vec[j];
                    }
                }
            }
        } else {
            for bi in 0..batch {
                let a_base = &da[bi * m * k..(bi + 1) * m * k];
                let b_base = &db[bi * k * n..(bi + 1) * k * n];
                let o_base = &mut out[bi * m * n..(bi + 1) * m * n];
                for i in 0..m {
                    let a_row = &a_base[i * k..(i + 1) * k];
                    let o_row = &mut o_base[i * n..(i + 1) * n];
                    for p in 0..k {
                        let a_ip = a_row[p];
                        let b_row = &b_base[p * n..(p + 1) * n];
                        for j in 0..n {
                            o_row[j] += a_ip * b_row[j];
                        }
                    }
                }
            }
        }

        self.push_node(
            out,
            vec![batch, m, n],
            Op::BatchedMatMul {
                a,
                b,
                batch,
                m,
                k,
                n,
            },
        )
    }

    pub fn transpose_2d(&mut self, a: ValId) -> ValId {
        let s = &self.vals[a.0].shape;
        assert_eq!(s.len(), 2, "transpose_2d requires 2D tensor");
        let (rows, cols) = (s[0], s[1]);
        let da = &self.vals[a.0].data;
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = da[r * cols + c];
            }
        }
        self.push_node(out, vec![cols, rows], Op::Transpose2D { a, rows, cols })
    }

    pub fn transpose_last2(&mut self, a: ValId) -> ValId {
        let s = &self.vals[a.0].shape;
        assert_eq!(s.len(), 3, "transpose_last2 requires 3D tensor");
        let (batch, rows, cols) = (s[0], s[1], s[2]);
        let da = &self.vals[a.0].data;
        let mut out = vec![0.0f32; batch * rows * cols];
        for bi in 0..batch {
            let off = bi * rows * cols;
            for r in 0..rows {
                for c in 0..cols {
                    out[off + c * rows + r] = da[off + r * cols + c];
                }
            }
        }
        self.push_node(
            out,
            vec![batch, cols, rows],
            Op::TransposeLast2 {
                a,
                batch,
                rows,
                cols,
            },
        )
    }

    pub fn rmsnorm_rows(&mut self, a: ValId, eps: f32) -> ValId {
        let s = &self.vals[a.0].shape;
        assert_eq!(s.len(), 2, "rmsnorm_rows requires 2D tensor");
        let (rows, cols) = (s[0], s[1]);
        let da = &self.vals[a.0].data;
        let mut out = vec![0.0f32; rows * cols];
        let mut inv_rms = vec![0.0f32; rows];
        for r in 0..rows {
            let row = &da[r * cols..(r + 1) * cols];
            let mean_sq = row.iter().map(|&x| x * x).sum::<f32>() / (cols as f32);
            let ir = 1.0 / (mean_sq + eps).sqrt();
            inv_rms[r] = ir;
            let dst = &mut out[r * cols..(r + 1) * cols];
            for c in 0..cols {
                dst[c] = row[c] * ir;
            }
        }
        self.push_node(
            out,
            vec![rows, cols],
            Op::RmsNormRows {
                a,
                rows,
                cols,
                inv_rms,
            },
        )
    }

    pub fn softmax_rows(&mut self, a: ValId) -> ValId {
        let s = &self.vals[a.0].shape;
        assert_eq!(s.len(), 2, "softmax_rows requires 2D tensor");
        let (rows, cols) = (s[0], s[1]);
        let da = &self.vals[a.0].data;
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let row = &da[r * cols..(r + 1) * cols];
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            let dst = &mut out[r * cols..(r + 1) * cols];
            for c in 0..cols {
                let e = (row[c] - max_v).exp();
                dst[c] = e;
                sum_exp += e;
            }
            let inv = 1.0 / sum_exp.max(1e-20);
            for v in dst.iter_mut() {
                *v *= inv;
            }
        }
        self.push_node(out, vec![rows, cols], Op::SoftmaxRows { a, rows, cols })
    }

    pub fn masked_log_softmax_rows(&mut self, a: ValId, mask: Vec<f32>) -> ValId {
        let s = &self.vals[a.0].shape;
        assert_eq!(s.len(), 2, "masked_log_softmax_rows requires 2D tensor");
        let (rows, cols) = (s[0], s[1]);
        assert_eq!(mask.len(), cols, "mask length must equal cols");
        let da = &self.vals[a.0].data;
        let mut out = vec![-1e9f32; rows * cols];
        for r in 0..rows {
            let row = &da[r * cols..(r + 1) * cols];
            let mut max_v = f32::NEG_INFINITY;
            for c in 0..cols {
                if mask[c] > 0.5 && row[c] > max_v {
                    max_v = row[c];
                }
            }
            assert!(max_v.is_finite(), "masked_log_softmax_rows: empty mask!");
            let mut sum_exp = 0.0f32;
            for c in 0..cols {
                if mask[c] > 0.5 {
                    sum_exp += (row[c] - max_v).exp();
                }
            }
            let lse = max_v + sum_exp.max(1e-20).ln();
            let dst = &mut out[r * cols..(r + 1) * cols];
            for c in 0..cols {
                if mask[c] > 0.5 {
                    dst[c] = row[c] - lse;
                }
            }
        }
        self.push_node(
            out,
            vec![rows, cols],
            Op::MaskedLogSoftmaxRows {
                a,
                rows,
                cols,
                mask,
            },
        )
    }

    pub fn sum_all(&mut self, a: ValId) -> ValId {
        let s: f32 = self.vals[a.0].data.iter().sum();
        self.push_node(vec![s], vec![1], Op::SumAll(a))
    }

    pub fn mean_all(&mut self, a: ValId) -> ValId {
        let len = self.vals[a.0].data.len().max(1) as f32;
        let s: f32 = self.vals[a.0].data.iter().sum::<f32>() / len;
        self.push_node(vec![s], vec![1], Op::MeanAll(a))
    }

    pub fn sum_axis0(&mut self, a: ValId) -> ValId {
        let s = &self.vals[a.0].shape;
        assert_eq!(s.len(), 2, "sum_axis0 requires 2D");
        let (rows, cols) = (s[0], s[1]);
        let da = &self.vals[a.0].data;
        let mut out = vec![0.0f32; cols];
        for r in 0..rows {
            let row = &da[r * cols..(r + 1) * cols];
            for c in 0..cols {
                out[c] += row[c];
            }
        }
        self.push_node(out, vec![1, cols], Op::SumAxis0 { a, rows, cols })
    }

    pub fn sum_axis1(&mut self, a: ValId) -> ValId {
        let s = &self.vals[a.0].shape;
        assert_eq!(s.len(), 2, "sum_axis1 requires 2D");
        let (rows, cols) = (s[0], s[1]);
        let da = &self.vals[a.0].data;
        let mut out = vec![0.0f32; rows];
        for r in 0..rows {
            out[r] = da[r * cols..(r + 1) * cols].iter().sum();
        }
        self.push_node(out, vec![rows, 1], Op::SumAxis1 { a, rows, cols })
    }

    pub fn sum_batch(&mut self, a: ValId) -> ValId {
        let s = &self.vals[a.0].shape;
        assert_eq!(s.len(), 3, "sum_batch requires 3D");
        let (batch, rows, cols) = (s[0], s[1], s[2]);
        let stride = rows * cols;
        let da = &self.vals[a.0].data;
        let mut out = vec![0.0f32; stride];
        for b in 0..batch {
            let src = &da[b * stride..(b + 1) * stride];
            for i in 0..stride {
                out[i] += src[i];
            }
        }
        self.push_node(out, vec![rows, cols], Op::SumBatch { a, batch, stride })
    }

    pub fn reshape(&mut self, a: ValId, new_shape: Vec<usize>) -> ValId {
        let expected: usize = new_shape.iter().product();
        assert_eq!(self.vals[a.0].data.len(), expected, "reshape element count mismatch");
        let data = self.vals[a.0].data.clone();
        self.push_node(data, new_shape, Op::Reshape { a })
    }

    pub fn concat_cols(&mut self, a: ValId, b: ValId) -> ValId {
        let sa = &self.vals[a.0].shape;
        let sb = &self.vals[b.0].shape;
        assert_eq!(sa.len(), 2);
        assert_eq!(sb.len(), 2);
        assert_eq!(sa[0], sb[0], "concat_cols row mismatch");
        let (rows, cols_a, cols_b) = (sa[0], sa[1], sb[1]);
        let da = &self.vals[a.0].data;
        let db = &self.vals[b.0].data;
        let mut out = Vec::with_capacity(rows * (cols_a + cols_b));
        for r in 0..rows {
            out.extend_from_slice(&da[r * cols_a..(r + 1) * cols_a]);
            out.extend_from_slice(&db[r * cols_b..(r + 1) * cols_b]);
        }
        self.push_node(
            out,
            vec![rows, cols_a + cols_b],
            Op::ConcatCols {
                a,
                b,
                rows,
                cols_a,
                cols_b,
            },
        )
    }

    pub fn concat_rows(&mut self, a: ValId, b: ValId) -> ValId {
        let sa = &self.vals[a.0].shape;
        let sb = &self.vals[b.0].shape;
        assert_eq!(sa.len(), 2);
        assert_eq!(sb.len(), 2);
        assert_eq!(sa[1], sb[1], "concat_rows col mismatch");
        let (rows_a, rows_b, cols) = (sa[0], sb[0], sa[1]);
        let mut out = Vec::with_capacity((rows_a + rows_b) * cols);
        out.extend_from_slice(&self.vals[a.0].data);
        out.extend_from_slice(&self.vals[b.0].data);
        self.push_node(
            out,
            vec![rows_a + rows_b, cols],
            Op::ConcatRows {
                a,
                b,
                rows_a,
                cols,
            },
        )
    }

    pub fn slice_rows(&mut self, a: ValId, start: usize, len: usize) -> ValId {
        let s = &self.vals[a.0].shape;
        assert_eq!(s.len(), 2);
        assert!(start + len <= s[0], "slice_rows out of bounds");
        let cols = s[1];
        let out = self.vals[a.0].data[start * cols..(start + len) * cols].to_vec();
        self.push_node(
            out,
            vec![len, cols],
            Op::SliceRows {
                a,
                start,
                len,
                cols,
            },
        )
    }

    pub fn slice_cols(&mut self, a: ValId, start: usize, len: usize) -> ValId {
        let s = &self.vals[a.0].shape;
        assert_eq!(s.len(), 2);
        let (rows, total_cols) = (s[0], s[1]);
        assert!(start + len <= total_cols, "slice_cols out of bounds");
        let da = &self.vals[a.0].data;
        let mut out = Vec::with_capacity(rows * len);
        for r in 0..rows {
            out.extend_from_slice(&da[r * total_cols + start..r * total_cols + start + len]);
        }
        self.push_node(
            out,
            vec![rows, len],
            Op::SliceCols {
                a,
                rows,
                total_cols,
                start,
                len,
            },
        )
    }

    pub fn gather_rows(&mut self, a: ValId, indices: Vec<usize>) -> ValId {
        let s = &self.vals[a.0].shape;
        assert_eq!(s.len(), 2);
        let (rows, cols) = (s[0], s[1]);
        let da = &self.vals[a.0].data;
        let mut out = Vec::with_capacity(indices.len() * cols);
        for &idx in &indices {
            assert!(idx < rows, "gather_rows index {} >= {}", idx, rows);
            out.extend_from_slice(&da[idx * cols..(idx + 1) * cols]);
        }
        self.push_node(
            out,
            vec![indices.len(), cols],
            Op::GatherRows { a, indices, cols },
        )
    }

    pub fn scatter_add_rows(&mut self, a: ValId, indices: Vec<usize>, out_rows: usize) -> ValId {
        let s = &self.vals[a.0].shape;
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], indices.len());
        let cols = s[1];
        let da = &self.vals[a.0].data;
        let mut out = vec![0.0f32; out_rows * cols];
        for (i, &dst) in indices.iter().enumerate() {
            assert!(dst < out_rows, "scatter_add_rows dst {} >= {}", dst, out_rows);
            let src_row = &da[i * cols..(i + 1) * cols];
            let dst_row = &mut out[dst * cols..(dst + 1) * cols];
            for c in 0..cols {
                dst_row[c] += src_row[c];
            }
        }
        self.push_node(
            out,
            vec![out_rows, cols],
            Op::ScatterAddRows { a, indices, cols },
        )
    }

    pub fn mse_loss(&mut self, pred: ValId, target: ValId) -> ValId {
        let diff = self.sub(pred, target);
        let sq = self.square(diff);
        self.mean_all(sq)
    }

    pub fn nll_loss(&mut self, log_probs: ValId, targets: Vec<usize>) -> ValId {
        let s = &self.vals[log_probs.0].shape;
        assert_eq!(s.len(), 2);
        let (rows, cols) = (s[0], s[1]);
        assert_eq!(rows, targets.len());
        let lp = &self.vals[log_probs.0].data;
        let mut loss = 0.0f32;
        for (r, &t) in targets.iter().enumerate() {
            assert!(t < cols, "target {} >= {}", t, cols);
            loss -= lp[r * cols + t];
        }
        loss /= rows.max(1) as f32;
        self.push_node(
            vec![loss],
            vec![1],
            Op::NllLoss {
                log_probs,
                targets,
                cols,
            },
        )
    }

    pub fn bce_with_logits(&mut self, logits: ValId, targets: Vec<f32>) -> ValId {
        let dz = &self.vals[logits.0].data;
        assert_eq!(dz.len(), targets.len());
        let n = dz.len().max(1) as f32;
        let mut loss = 0.0f32;
        for (&z, &t) in dz.iter().zip(targets.iter()) {
            let max_z = z.max(0.0);
            loss += max_z - z * t + (1.0 + (-z.abs()).exp()).ln();
        }
        loss /= n;
        self.push_node(vec![loss], vec![1], Op::BceWithLogits { logits, targets })
    }

    pub fn bce_with_probs(&mut self, probs: ValId, targets: Vec<f32>, eps: f32) -> ValId {
        let dp = &self.vals[probs.0].data;
        assert_eq!(dp.len(), targets.len());
        let n = dp.len().max(1) as f32;
        let mut loss = 0.0f32;
        for (&p, &t) in dp.iter().zip(targets.iter()) {
            let pc = p.clamp(eps, 1.0 - eps);
            loss -= t * pc.ln() + (1.0 - t) * (1.0 - pc).ln();
        }
        loss /= n;
        self.push_node(vec![loss], vec![1], Op::BceWithProbs { probs, targets, eps })
    }

    pub fn backward(&mut self, root: ValId, store: &mut ParamStore) {
        assert_eq!(self.vals[root.0].len(), 1, "backward root must be scalar");
        for g in &mut self.grads {
            g.fill(0.0);
        }
        self.grads[root.0][0] = 1.0;

        for idx in (0..=root.0).rev() {
            if self.grads[idx].iter().all(|&x| x == 0.0) {
                continue;
            }
            let g_out = std::mem::take(&mut self.grads[idx]);

            match &self.ops[idx] {
                Op::Const => {}
                Op::Param(pid) => {
                    store.accum_grad(*pid, &g_out);
                }
                Op::Add { a, b, kind } => match kind {
                    BroadcastKind::SameShape => {
                        for (ga, &go) in self.grads[a.0].iter_mut().zip(g_out.iter()) {
                            *ga += go;
                        }
                        for (gb, &go) in self.grads[b.0].iter_mut().zip(g_out.iter()) {
                            *gb += go;
                        }
                    }
                    BroadcastKind::RightScalar => {
                        let mut acc = 0.0f32;
                        for (ga, &go) in self.grads[a.0].iter_mut().zip(g_out.iter()) {
                            *ga += go;
                            acc += go;
                        }
                        self.grads[b.0][0] += acc;
                    }
                    BroadcastKind::LeftScalar => {
                        let mut acc = 0.0f32;
                        for (gb, &go) in self.grads[b.0].iter_mut().zip(g_out.iter()) {
                            *gb += go;
                            acc += go;
                        }
                        self.grads[a.0][0] += acc;
                    }
                    BroadcastKind::RightTrailing { outer, inner } => {
                        for (ga, &go) in self.grads[a.0].iter_mut().zip(g_out.iter()) {
                            *ga += go;
                        }
                        let gb = &mut self.grads[b.0];
                        for c in 0..*outer {
                            let s: f32 = g_out[c * inner..(c + 1) * inner].iter().sum();
                            gb[c] += s;
                        }
                    }
                    BroadcastKind::LeftTrailing { outer, inner } => {
                        for (gb, &go) in self.grads[b.0].iter_mut().zip(g_out.iter()) {
                            *gb += go;
                        }
                        let ga = &mut self.grads[a.0];
                        for c in 0..*outer {
                            let s: f32 = g_out[c * inner..(c + 1) * inner].iter().sum();
                            ga[c] += s;
                        }
                    }
                    BroadcastKind::RightLeading { rows, cols } => {
                        for (ga, &go) in self.grads[a.0].iter_mut().zip(g_out.iter()) {
                            *ga += go;
                        }
                        let gb = &mut self.grads[b.0];
                        for r in 0..*rows {
                            let go_row = &g_out[r * cols..(r + 1) * cols];
                            for j in 0..*cols {
                                gb[j] += go_row[j];
                            }
                        }
                    }
                    BroadcastKind::LeftLeading { rows, cols } => {
                        for (gb, &go) in self.grads[b.0].iter_mut().zip(g_out.iter()) {
                            *gb += go;
                        }
                        let ga = &mut self.grads[a.0];
                        for r in 0..*rows {
                            let go_row = &g_out[r * cols..(r + 1) * cols];
                            for j in 0..*cols {
                                ga[j] += go_row[j];
                            }
                        }
                    }
                    BroadcastKind::General {
                        a_strides,
                        b_strides,
                        out_shape,
                    } => {
                        let mut coords = vec![0usize; out_shape.len()];
                        for &go in &g_out {
                            let ia = offset_from_coords(&coords, a_strides);
                            let ib = offset_from_coords(&coords, b_strides);
                            self.grads[a.0][ia] += go;
                            self.grads[b.0][ib] += go;
                            next_multi_idx(&mut coords, out_shape);
                        }
                    }
                },
                Op::Sub { a, b, kind } => match kind {
                    BroadcastKind::SameShape => {
                        for (ga, &go) in self.grads[a.0].iter_mut().zip(g_out.iter()) {
                            *ga += go;
                        }
                        for (gb, &go) in self.grads[b.0].iter_mut().zip(g_out.iter()) {
                            *gb -= go;
                        }
                    }
                    BroadcastKind::RightScalar => {
                        let mut acc = 0.0f32;
                        for (ga, &go) in self.grads[a.0].iter_mut().zip(g_out.iter()) {
                            *ga += go;
                            acc += go;
                        }
                        self.grads[b.0][0] -= acc;
                    }
                    BroadcastKind::LeftScalar => {
                        let mut acc = 0.0f32;
                        for (gb, &go) in self.grads[b.0].iter_mut().zip(g_out.iter()) {
                            *gb -= go;
                            acc += go;
                        }
                        self.grads[a.0][0] += acc;
                    }
                    BroadcastKind::RightTrailing { outer, inner } => {
                        for (ga, &go) in self.grads[a.0].iter_mut().zip(g_out.iter()) {
                            *ga += go;
                        }
                        let gb = &mut self.grads[b.0];
                        for c in 0..*outer {
                            let s: f32 = g_out[c * inner..(c + 1) * inner].iter().sum();
                            gb[c] -= s;
                        }
                    }
                    BroadcastKind::LeftTrailing { outer, inner } => {
                        for (gb, &go) in self.grads[b.0].iter_mut().zip(g_out.iter()) {
                            *gb -= go;
                        }
                        let ga = &mut self.grads[a.0];
                        for c in 0..*outer {
                            let s: f32 = g_out[c * inner..(c + 1) * inner].iter().sum();
                            ga[c] += s;
                        }
                    }
                    BroadcastKind::RightLeading { rows, cols } => {
                        for (ga, &go) in self.grads[a.0].iter_mut().zip(g_out.iter()) {
                            *ga += go;
                        }
                        let gb = &mut self.grads[b.0];
                        for r in 0..*rows {
                            let go_row = &g_out[r * cols..(r + 1) * cols];
                            for j in 0..*cols {
                                gb[j] -= go_row[j];
                            }
                        }
                    }
                    BroadcastKind::LeftLeading { rows, cols } => {
                        for (gb, &go) in self.grads[b.0].iter_mut().zip(g_out.iter()) {
                            *gb -= go;
                        }
                        let ga = &mut self.grads[a.0];
                        for r in 0..*rows {
                            let go_row = &g_out[r * cols..(r + 1) * cols];
                            for j in 0..*cols {
                                ga[j] += go_row[j];
                            }
                        }
                    }
                    BroadcastKind::General {
                        a_strides,
                        b_strides,
                        out_shape,
                    } => {
                        let mut coords = vec![0usize; out_shape.len()];
                        for &go in &g_out {
                            let ia = offset_from_coords(&coords, a_strides);
                            let ib = offset_from_coords(&coords, b_strides);
                            self.grads[a.0][ia] += go;
                            self.grads[b.0][ib] -= go;
                            next_multi_idx(&mut coords, out_shape);
                        }
                    }
                },
                Op::Mul { a, b, kind } => {
                    let da = &self.vals[a.0].data;
                    let db = &self.vals[b.0].data;
                    match kind {
                        BroadcastKind::SameShape => {
                            if a.0 != b.0 {
                                let (ga, gb) = borrow_two_mut(&mut self.grads, a.0, b.0);
                                for i in 0..g_out.len() {
                                    let go = g_out[i];
                                    ga[i] += go * db[i];
                                    gb[i] += go * da[i];
                                }
                            } else {
                                let ga = &mut self.grads[a.0];
                                for i in 0..g_out.len() {
                                    ga[i] += 2.0 * g_out[i] * da[i];
                                }
                            }
                        }
                        BroadcastKind::RightScalar => {
                            let s = db[0];
                            let mut acc = 0.0f32;
                            let ga = &mut self.grads[a.0];
                            for i in 0..g_out.len() {
                                let go = g_out[i];
                                ga[i] += go * s;
                                acc += go * da[i];
                            }
                            self.grads[b.0][0] += acc;
                        }
                        BroadcastKind::LeftScalar => {
                            let s = da[0];
                            let mut acc = 0.0f32;
                            let gb = &mut self.grads[b.0];
                            for i in 0..g_out.len() {
                                let go = g_out[i];
                                gb[i] += go * s;
                                acc += go * db[i];
                            }
                            self.grads[a.0][0] += acc;
                        }
                        BroadcastKind::RightTrailing { outer, inner } => {
                            let (ga, gb) = borrow_two_mut(&mut self.grads, a.0, b.0);
                            for c in 0..*outer {
                                let s = db[c];
                                let go_c = &g_out[c * inner..(c + 1) * inner];
                                let da_c = &da[c * inner..(c + 1) * inner];
                                let ga_c = &mut ga[c * inner..(c + 1) * inner];
                                let mut acc = 0.0f32;
                                for j in 0..*inner {
                                    let go = go_c[j];
                                    ga_c[j] += go * s;
                                    acc += go * da_c[j];
                                }
                                gb[c] += acc;
                            }
                        }
                        BroadcastKind::LeftTrailing { outer, inner } => {
                            let (ga, gb) = borrow_two_mut(&mut self.grads, a.0, b.0);
                            for c in 0..*outer {
                                let s = da[c];
                                let go_c = &g_out[c * inner..(c + 1) * inner];
                                let db_c = &db[c * inner..(c + 1) * inner];
                                let gb_c = &mut gb[c * inner..(c + 1) * inner];
                                let mut acc = 0.0f32;
                                for j in 0..*inner {
                                    let go = go_c[j];
                                    gb_c[j] += go * s;
                                    acc += go * db_c[j];
                                }
                                ga[c] += acc;
                            }
                        }
                        BroadcastKind::RightLeading { rows, cols } => {
                            let (ga, gb) = borrow_two_mut(&mut self.grads, a.0, b.0);
                            for r in 0..*rows {
                                let go_r = &g_out[r * cols..(r + 1) * cols];
                                let da_r = &da[r * cols..(r + 1) * cols];
                                let ga_r = &mut ga[r * cols..(r + 1) * cols];
                                for j in 0..*cols {
                                    let go = go_r[j];
                                    ga_r[j] += go * db[j];
                                    gb[j] += go * da_r[j];
                                }
                            }
                        }
                        BroadcastKind::LeftLeading { rows, cols } => {
                            let (ga, gb) = borrow_two_mut(&mut self.grads, a.0, b.0);
                            for r in 0..*rows {
                                let go_r = &g_out[r * cols..(r + 1) * cols];
                                let db_r = &db[r * cols..(r + 1) * cols];
                                let gb_r = &mut gb[r * cols..(r + 1) * cols];
                                for j in 0..*cols {
                                    let go = go_r[j];
                                    gb_r[j] += go * da[j];
                                    ga[j] += go * db_r[j];
                                }
                            }
                        }
                        BroadcastKind::General {
                            a_strides,
                            b_strides,
                            out_shape,
                        } => {
                            let mut coords = vec![0usize; out_shape.len()];
                            for &go in &g_out {
                                let ia = offset_from_coords(&coords, a_strides);
                                let ib = offset_from_coords(&coords, b_strides);
                                self.grads[a.0][ia] += go * db[ib];
                                self.grads[b.0][ib] += go * da[ia];
                                next_multi_idx(&mut coords, out_shape);
                            }
                        }
                    }
                }
                Op::Neg(a) => {
                    for (ga, &go) in self.grads[a.0].iter_mut().zip(g_out.iter()) {
                        *ga -= go;
                    }
                }
                Op::Scale { a, s } => {
                    for (ga, &go) in self.grads[a.0].iter_mut().zip(g_out.iter()) {
                        *ga += go * *s;
                    }
                }
                Op::Tanh(a) => {
                    let y = &self.vals[idx].data;
                    for ((ga, &yi), &go) in self.grads[a.0].iter_mut().zip(y.iter()).zip(g_out.iter()) {
                        *ga += go * (1.0 - yi * yi);
                    }
                }
                Op::Sigmoid(a) => {
                    let y = &self.vals[idx].data;
                    for ((ga, &yi), &go) in self.grads[a.0].iter_mut().zip(y.iter()).zip(g_out.iter()) {
                        *ga += go * yi * (1.0 - yi);
                    }
                }
                Op::Relu(a) => {
                    let x = &self.vals[a.0].data;
                    for ((ga, &xi), &go) in self.grads[a.0].iter_mut().zip(x.iter()).zip(g_out.iter()) {
                        if xi > 0.0 {
                            *ga += go;
                        }
                    }
                }
                Op::Silu(a) => {
                    let x = &self.vals[a.0].data;
                    for ((ga, &xi), &go) in self.grads[a.0].iter_mut().zip(x.iter()).zip(g_out.iter()) {
                        let sig = if xi >= 0.0 {
                            1.0 / (1.0 + (-xi).exp())
                        } else {
                            let ex = xi.exp();
                            ex / (1.0 + ex)
                        };
                        *ga += go * (sig * (1.0 + xi * (1.0 - sig)));
                    }
                }
                Op::Exp(a) => {
                    let x = &self.vals[a.0].data;
                    let y = &self.vals[idx].data;
                    for (((ga, &xi), &yi), &go) in self.grads[a.0]
                        .iter_mut()
                        .zip(x.iter())
                        .zip(y.iter())
                        .zip(g_out.iter())
                    {
                        if (-60.0..=60.0).contains(&xi) {
                            *ga += go * yi;
                        }
                    }
                }
                Op::SafeLog { a, eps } => {
                    let x = &self.vals[a.0].data;
                    for ((ga, &xi), &go) in self.grads[a.0].iter_mut().zip(x.iter()).zip(g_out.iter()) {
                        if xi > *eps {
                            *ga += go / xi;
                        }
                    }
                }
                Op::SafeSqrt { a, .. } => {
                    let x = &self.vals[a.0].data;
                    let y = &self.vals[idx].data;
                    for (((ga, &xi), &yi), &go) in self.grads[a.0]
                        .iter_mut()
                        .zip(x.iter())
                        .zip(y.iter())
                        .zip(g_out.iter())
                    {
                        if xi > 0.0 {
                            *ga += go / (2.0 * yi);
                        }
                    }
                }
                Op::Square(a) => {
                    let x = &self.vals[a.0].data;
                    for ((ga, &xi), &go) in self.grads[a.0].iter_mut().zip(x.iter()).zip(g_out.iter()) {
                        *ga += go * 2.0 * xi;
                    }
                }
                Op::MatMul { a, b, m, k, n } => {
                    let da = &self.vals[a.0].data;
                    let db = &self.vals[b.0].data;
                    let (m, k, n) = (*m, *k, *n);
                    if a.0 != b.0 {
                        let (ga, gb) = borrow_two_mut(&mut self.grads, a.0, b.0);
                        for i in 0..m {
                            let go_row = &g_out[i * n..(i + 1) * n];
                            let a_row = &da[i * k..(i + 1) * k];
                            let ga_row = &mut ga[i * k..(i + 1) * k];
                            for p in 0..k {
                                let a_ip = a_row[p];
                                let db_row = &db[p * n..(p + 1) * n];
                                let gb_row = &mut gb[p * n..(p + 1) * n];
                                let mut acc = 0.0f32;
                                for j in 0..n {
                                    let go = go_row[j];
                                    acc += go * db_row[j];
                                    gb_row[j] += a_ip * go;
                                }
                                ga_row[p] += acc;
                            }
                        }
                    } else {
                        let mut ga_buf = vec![0.0f32; m * k];
                        let mut gb_buf = vec![0.0f32; k * n];
                        for i in 0..m {
                            let go_row = &g_out[i * n..(i + 1) * n];
                            for p in 0..k {
                                let a_ip = da[i * k + p];
                                let db_row = &db[p * n..(p + 1) * n];
                                let gb_row = &mut gb_buf[p * n..(p + 1) * n];
                                let mut acc = 0.0f32;
                                for j in 0..n {
                                    let go = go_row[j];
                                    acc += go * db_row[j];
                                    gb_row[j] += a_ip * go;
                                }
                                ga_buf[i * k + p] += acc;
                            }
                        }
                        let ga = &mut self.grads[a.0];
                        for t in 0..ga.len() {
                            ga[t] += ga_buf[t] + gb_buf[t];
                        }
                    }
                }
                Op::BatchedMatMul {
                    a,
                    b,
                    batch,
                    m,
                    k,
                    n,
                } => {
                    let da = &self.vals[a.0].data;
                    let db = &self.vals[b.0].data;
                    let (batch, m, k, n) = (*batch, *m, *k, *n);
                    assert_ne!(a.0, b.0, "self batched_matmul not used");
                    let (ga, gb) = borrow_two_mut(&mut self.grads, a.0, b.0);

                    if n == 1 {
                        for bi in 0..batch {
                            let go_vec = &g_out[bi * m..(bi + 1) * m];
                            let a_base = &da[bi * m * k..(bi + 1) * m * k];
                            let b_vec = &db[bi * k..(bi + 1) * k];
                            let ga_base = &mut ga[bi * m * k..(bi + 1) * m * k];
                            let gb_vec = &mut gb[bi * k..(bi + 1) * k];
                            for i in 0..m {
                                let go = go_vec[i];
                                let a_row = &a_base[i * k..(i + 1) * k];
                                let ga_row = &mut ga_base[i * k..(i + 1) * k];
                                for p in 0..k {
                                    ga_row[p] += go * b_vec[p];
                                    gb_vec[p] += a_row[p] * go;
                                }
                            }
                        }
                    } else if k == 1 {
                        for bi in 0..batch {
                            let go_base = &g_out[bi * m * n..(bi + 1) * m * n];
                            let a_vec = &da[bi * m..(bi + 1) * m];
                            let b_vec = &db[bi * n..(bi + 1) * n];
                            let ga_vec = &mut ga[bi * m..(bi + 1) * m];
                            let gb_vec = &mut gb[bi * n..(bi + 1) * n];
                            for i in 0..m {
                                let ai = a_vec[i];
                                let go_row = &go_base[i * n..(i + 1) * n];
                                let mut acc = 0.0f32;
                                for j in 0..n {
                                    let go = go_row[j];
                                    acc += go * b_vec[j];
                                    gb_vec[j] += ai * go;
                                }
                                ga_vec[i] += acc;
                            }
                        }
                    } else {
                        for bi in 0..batch {
                            let go_base = &g_out[bi * m * n..(bi + 1) * m * n];
                            let a_base = &da[bi * m * k..(bi + 1) * m * k];
                            let b_base = &db[bi * k * n..(bi + 1) * k * n];
                            let ga_base = &mut ga[bi * m * k..(bi + 1) * m * k];
                            let gb_base = &mut gb[bi * k * n..(bi + 1) * k * n];
                            for i in 0..m {
                                let go_row = &go_base[i * n..(i + 1) * n];
                                let a_row = &a_base[i * k..(i + 1) * k];
                                let ga_row = &mut ga_base[i * k..(i + 1) * k];
                                for p in 0..k {
                                    let a_ip = a_row[p];
                                    let b_row = &b_base[p * n..(p + 1) * n];
                                    let gb_row = &mut gb_base[p * n..(p + 1) * n];
                                    let mut acc = 0.0f32;
                                    for j in 0..n {
                                        let go = go_row[j];
                                        acc += go * b_row[j];
                                        gb_row[j] += a_ip * go;
                                    }
                                    ga_row[p] += acc;
                                }
                            }
                        }
                    }
                }
                Op::Transpose2D { a, rows, cols } => {
                    let (rows, cols) = (*rows, *cols);
                    let ga = &mut self.grads[a.0];
                    for r in 0..rows {
                        for c in 0..cols {
                            ga[r * cols + c] += g_out[c * rows + r];
                        }
                    }
                }
                Op::TransposeLast2 {
                    a,
                    batch,
                    rows,
                    cols,
                } => {
                    let (batch, rows, cols) = (*batch, *rows, *cols);
                    let ga = &mut self.grads[a.0];
                    for bi in 0..batch {
                        let off = bi * rows * cols;
                        for r in 0..rows {
                            for c in 0..cols {
                                ga[off + r * cols + c] += g_out[off + c * rows + r];
                            }
                        }
                    }
                }
                Op::RmsNormRows {
                    a,
                    rows,
                    cols,
                    inv_rms,
                } => {
                    let (rows, cols) = (*rows, *cols);
                    let y = &self.vals[idx].data;
                    let ga = &mut self.grads[a.0];
                    for r in 0..rows {
                        let ir = inv_rms[r];
                        let y_row = &y[r * cols..(r + 1) * cols];
                        let g_row = &g_out[r * cols..(r + 1) * cols];
                        let ga_row = &mut ga[r * cols..(r + 1) * cols];
                        let dot_gy: f32 = g_row.iter().zip(y_row.iter()).map(|(&g, &yi)| g * yi).sum();
                        let mean_gy = dot_gy / (cols as f32);
                        for c in 0..cols {
                            ga_row[c] += ir * (g_row[c] - y_row[c] * mean_gy);
                        }
                    }
                }
                Op::SoftmaxRows { a, rows, cols } => {
                    let (rows, cols) = (*rows, *cols);
                    let p = &self.vals[idx].data;
                    let ga = &mut self.grads[a.0];
                    for r in 0..rows {
                        let p_row = &p[r * cols..(r + 1) * cols];
                        let g_row = &g_out[r * cols..(r + 1) * cols];
                        let ga_row = &mut ga[r * cols..(r + 1) * cols];
                        let dot: f32 = p_row.iter().zip(g_row.iter()).map(|(&pi, &gi)| pi * gi).sum();
                        for c in 0..cols {
                            ga_row[c] += p_row[c] * (g_row[c] - dot);
                        }
                    }
                }
                Op::MaskedLogSoftmaxRows {
                    a,
                    rows,
                    cols,
                    mask,
                } => {
                    let (rows, cols) = (*rows, *cols);
                    let lp = &self.vals[idx].data;
                    let ga = &mut self.grads[a.0];
                    for r in 0..rows {
                        let lp_row = &lp[r * cols..(r + 1) * cols];
                        let g_row = &g_out[r * cols..(r + 1) * cols];
                        let ga_row = &mut ga[r * cols..(r + 1) * cols];
                        let mut sum_g = 0.0f32;
                        for c in 0..cols {
                            if mask[c] > 0.5 {
                                sum_g += g_row[c];
                            }
                        }
                        for c in 0..cols {
                            if mask[c] > 0.5 {
                                let p = lp_row[c].exp();
                                ga_row[c] += g_row[c] - p * sum_g;
                            }
                        }
                    }
                }
                Op::SumAll(a) => {
                    let go = g_out[0];
                    for ga in &mut self.grads[a.0] {
                        *ga += go;
                    }
                }
                Op::MeanAll(a) => {
                    let n = self.grads[a.0].len().max(1) as f32;
                    let go = g_out[0] / n;
                    for ga in &mut self.grads[a.0] {
                        *ga += go;
                    }
                }
                Op::SumAxis0 { a, rows, cols } => {
                    let (rows, cols) = (*rows, *cols);
                    let ga = &mut self.grads[a.0];
                    for r in 0..rows {
                        let ga_row = &mut ga[r * cols..(r + 1) * cols];
                        for c in 0..cols {
                            ga_row[c] += g_out[c];
                        }
                    }
                }
                Op::SumAxis1 { a, rows, cols } => {
                    let (rows, cols) = (*rows, *cols);
                    let ga = &mut self.grads[a.0];
                    for r in 0..rows {
                        let go = g_out[r];
                        let ga_row = &mut ga[r * cols..(r + 1) * cols];
                        for c in 0..cols {
                            ga_row[c] += go;
                        }
                    }
                }
                Op::SumBatch { a, batch, stride } => {
                    let (batch, stride) = (*batch, *stride);
                    let ga = &mut self.grads[a.0];
                    for b in 0..batch {
                        let dst = &mut ga[b * stride..(b + 1) * stride];
                        for i in 0..stride {
                            dst[i] += g_out[i];
                        }
                    }
                }
                Op::Reshape { a } => {
                    for (ga, &go) in self.grads[a.0].iter_mut().zip(g_out.iter()) {
                        *ga += go;
                    }
                }
                Op::ConcatCols {
                    a,
                    b,
                    rows,
                    cols_a,
                    cols_b,
                } => {
                    let (rows, cols_a, cols_b) = (*rows, *cols_a, *cols_b);
                    let total = cols_a + cols_b;
                    let (ga, gb) = borrow_two_mut(&mut self.grads, a.0, b.0);
                    for r in 0..rows {
                        let go_row = &g_out[r * total..(r + 1) * total];
                        let ga_row = &mut ga[r * cols_a..(r + 1) * cols_a];
                        let gb_row = &mut gb[r * cols_b..(r + 1) * cols_b];
                        for c in 0..cols_a {
                            ga_row[c] += go_row[c];
                        }
                        for c in 0..cols_b {
                            gb_row[c] += go_row[cols_a + c];
                        }
                    }
                }
                Op::ConcatRows {
                    a,
                    b,
                    rows_a,
                    cols,
                } => {
                    let split = rows_a * cols;
                    for (ga, &go) in self.grads[a.0].iter_mut().zip(g_out[..split].iter()) {
                        *ga += go;
                    }
                    for (gb, &go) in self.grads[b.0].iter_mut().zip(g_out[split..].iter()) {
                        *gb += go;
                    }
                }
                Op::SliceRows { a, start, len, cols } => {
                    let off = start * cols;
                    let ga = &mut self.grads[a.0][off..off + len * cols];
                    for i in 0..(len * cols) {
                        ga[i] += g_out[i];
                    }
                }
                Op::SliceCols {
                    a,
                    rows,
                    total_cols,
                    start,
                    len,
                } => {
                    let (rows, total_cols, start, len) = (*rows, *total_cols, *start, *len);
                    let ga = &mut self.grads[a.0];
                    for r in 0..rows {
                        let ga_row = &mut ga[r * total_cols + start..r * total_cols + start + len];
                        let go_row = &g_out[r * len..(r + 1) * len];
                        for c in 0..len {
                            ga_row[c] += go_row[c];
                        }
                    }
                }
                Op::GatherRows { a, indices, cols } => {
                    let cols = *cols;
                    let ga = &mut self.grads[a.0];
                    for (i, &src_row) in indices.iter().enumerate() {
                        let ga_row = &mut ga[src_row * cols..(src_row + 1) * cols];
                        let go_row = &g_out[i * cols..(i + 1) * cols];
                        for c in 0..cols {
                            ga_row[c] += go_row[c];
                        }
                    }
                }
                Op::ScatterAddRows { a, indices, cols } => {
                    let cols = *cols;
                    let ga = &mut self.grads[a.0];
                    for (i, &dst_row) in indices.iter().enumerate() {
                        let ga_row = &mut ga[i * cols..(i + 1) * cols];
                        let go_row = &g_out[dst_row * cols..(dst_row + 1) * cols];
                        for c in 0..cols {
                            ga_row[c] += go_row[c];
                        }
                    }
                }
                Op::NllLoss {
                    log_probs,
                    targets,
                    cols,
                } => {
                    let cols = *cols;
                    let rows = targets.len().max(1) as f32;
                    let go = g_out[0] / rows;
                    let glp = &mut self.grads[log_probs.0];
                    for (r, &t) in targets.iter().enumerate() {
                        glp[r * cols + t] -= go;
                    }
                }
                Op::BceWithLogits { logits, targets } => {
                    let dz = &self.vals[logits.0].data;
                    let n = dz.len().max(1) as f32;
                    let go = g_out[0] / n;
                    let gz = &mut self.grads[logits.0];
                    for ((g_dst, &z), &t) in gz.iter_mut().zip(dz.iter()).zip(targets.iter()) {
                        let sig = if z >= 0.0 {
                            1.0 / (1.0 + (-z).exp())
                        } else {
                            let ex = z.exp();
                            ex / (1.0 + ex)
                        };
                        *g_dst += go * (sig - t);
                    }
                }
                Op::BceWithProbs {
                    probs,
                    targets,
                    eps,
                } => {
                    let dp = &self.vals[probs.0].data;
                    let n = dp.len().max(1) as f32;
                    let go = g_out[0] / n;
                    let gp = &mut self.grads[probs.0];
                    let eps = *eps;
                    for ((g_dst, &p), &t) in gp.iter_mut().zip(dp.iter()).zip(targets.iter()) {
                        if p > eps && p < 1.0 - eps {
                            *g_dst += go * ((1.0 - t) / (1.0 - p) - t / p);
                        }
                    }
                }
            }
        }
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite_diff_check<F>(store: &mut ParamStore, mut build_loss: F, eps: f32, tol: f32)
    where
        F: FnMut(&mut Tape, &ParamStore) -> ValId,
    {
        store.zero_grad();
        let mut tape = Tape::new();
        let loss = build_loss(&mut tape, store);
        tape.backward(loss, store);
        let analytic_grads = store.grads.clone();

        for pid in 0..store.values.len() {
            for idx in 0..store.values[pid].len() {
                let orig = store.values[pid][idx];

                store.values[pid][idx] = orig + eps;
                let mut t_plus = Tape::new();
                let l_plus_id = build_loss(&mut t_plus, store);
                let l_plus = t_plus.data(l_plus_id)[0];

                store.values[pid][idx] = orig - eps;
                let mut t_minus = Tape::new();
                let l_minus_id = build_loss(&mut t_minus, store);
                let l_minus = t_minus.data(l_minus_id)[0];

                store.values[pid][idx] = orig;

                let num_g = (l_plus - l_minus) / (2.0 * eps);
                let ana_g = analytic_grads[pid][idx];
                let diff = (num_g - ana_g).abs();
                let scale = num_g.abs().max(ana_g.abs()).max(1.0);
                assert!(
                    diff <= tol * scale,
                    "Grad mismatch at param {}[{}]: analytic={}, numeric={}, diff={}",
                    store.names[pid],
                    idx,
                    ana_g,
                    num_g,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_graph_ops_gradients() {
        let mut rng = Rng::from_seed(1337);
        let mut store = ParamStore::new();
        let p_edges = store.add_normal("edges", vec![4, 3, 3], 0.5, &mut rng);
        let p_nodes = store.add_normal("nodes", vec![2, 3], 0.5, &mut rng);
        let p_bias = store.add_normal("bias", vec![1, 3], 0.3, &mut rng);
        let p_head = store.add_normal("head", vec![6, 5], 0.4, &mut rng);

        let src_idx = vec![0usize, 0, 1, 1];
        let dst_idx = vec![0usize, 1, 0, 1];
        let mask = vec![1.0, 0.0, 1.0, 1.0, 0.0];
        let targets = vec![2usize, 3usize];

        finite_diff_check(
            &mut store,
            |tape, s| {
                let edges = tape.param(s, p_edges);
                let nodes = tape.param(s, p_nodes);
                let bias = tape.param(s, p_bias);
                let head = tape.param(s, p_head);

                let gathered = tape.gather_rows(nodes, src_idx.clone());
                let gathered_col = tape.reshape(gathered, vec![4, 3, 1]);
                let msg = tape.batched_matmul(edges, gathered_col);
                let msg_2d = tape.reshape(msg, vec![4, 3]);
                let agg = tape.scatter_add_rows(msg_2d, dst_idx.clone(), 2);
                let biased = tape.add(agg, bias);
                let norm = tape.rmsnorm_rows(biased, 1e-5);
                let act = tape.tanh(norm);
                let joined = tape.concat_cols(nodes, act);
                let logits = tape.matmul(joined, head);
                let log_p = tape.masked_log_softmax_rows(logits, mask.clone());
                let nll = tape.nll_loss(log_p, targets.clone());

                let edges_t = tape.transpose_last2(edges);
                let asym = tape.sub(edges, edges_t);
                let reg = tape.mean_all(tape.square(asym));
                let w_reg = tape.scale(reg, 0.1);
                tape.add(nll, w_reg)
            },
            1e-3,
            5e-2,
        );
    }

    #[test]
    fn test_fast_broadcast_and_outer_gradients() {
        let mut rng = Rng::from_seed(321);
        let mut store = ParamStore::new();
        let p_mat = store.add_normal("mat", vec![4, 3, 3], 0.5, &mut rng);
        let p_tr = store.add_normal("tr", vec![4, 1, 1], 0.4, &mut rng);
        let p_sc = store.add_normal("sc", vec![1], 0.3, &mut rng);
        let p_a = store.add_normal("a", vec![4, 3, 1], 0.5, &mut rng);
        let p_b = store.add_normal("b", vec![4, 1, 3], 0.5, &mut rng);

        finite_diff_check(
            &mut store,
            |tape, s| {
                let mat = tape.param(s, p_mat);
                let tr = tape.param(s, p_tr);
                let sc = tape.param(s, p_sc);
                let a = tape.param(s, p_a);
                let b = tape.param(s, p_b);

                let outer = tape.batched_matmul(a, b);
                let m1 = tape.mul(mat, tr);
                let m2 = tape.add(m1, outer);
                let m3 = tape.sub(m2, sc);
                tape.mean_all(tape.square(m3))
            },
            1e-3,
            5e-2,
        );
    }

    #[test]
    fn test_bce_softmax_slice_gradients() {
        let mut rng = Rng::from_seed(99);
        let mut store = ParamStore::new();
        let p_x = store.add_normal("x", vec![3, 4], 0.6, &mut rng);
        let p_w = store.add_normal("w", vec![2, 2], 0.6, &mut rng);

        finite_diff_check(
            &mut store,
            |tape, s| {
                let x = tape.param(s, p_x);
                let w = tape.param(s, p_w);
                let probs = tape.softmax_rows(x);
                let sub = tape.slice_rows(probs, 0, 2);
                let sub_c = tape.slice_cols(sub, 1, 2);
                let out = tape.matmul(sub_c, w);
                let sig = tape.sigmoid(out);
                let l1 = tape.bce_with_logits(out, vec![1.0, 0.0, 0.5, 1.0]);
                let l2 = tape.bce_with_probs(sig, vec![0.0, 1.0, 0.5, 0.0], 1e-5);
                tape.add(l1, l2)
            },
            1e-3,
            5e-2,
        );
    }
}