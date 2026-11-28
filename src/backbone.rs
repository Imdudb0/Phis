#![forbid(unsafe_code)]

use std::collections::{BTreeMap, VecDeque};

use crate::hypothesis_graph::{
    AddNodeKind, BackwardAction, BackwardMask, Edge, EdgeKind, ForwardAction, ForwardMask,
    GraphConfig, HypothesisGraph, NodeId, NodeKind, SymbolId,
};
use crate::memory_controller::{
    AbductivePolicy, MaskedForwardView, ResidualGuidedHeuristicPolicy,
};
use crate::projector::{
    NodeAlignmentStatus, ObservationGraph, ResidualEdgeKind, ResidualGraph, project_and_residual,
};
use crate::ptgrs_synth::TrainingSample;
use crate::refutation::{RefutationConfig, evaluate_phi_sharp};

// ============================================================================
// 1. Constantes topologiques du graphe tripartite (G_E, H_t, R_t)
// ============================================================================

/// Dimension fixe des traits relationnels d'entrée par nœud (aucun ID brut d'octet).
pub const NODE_FEAT_DIM: usize = 14;
/// Dimension du résumé scalaire du résidu R_t injecté dans le contexte global `g`.
pub const GLOBAL_RES_DIM: usize = 5;
/// Nombre de types d'arêtes dans le graphe tripartite.
pub const NUM_TRIPARTITE_EDGE_TYPES: usize = 12;
/// Catégories d'arêtes pour les têtes bilinéaires `CONNECT` et `DEL_EDGE` :
/// [InSeed, InLhs, InRhs, Struct(0), Struct(>=1), SameAs].
pub const NUM_CONNECT_CATS: usize = 6;
/// Nombre de traits relationnels de paire injectés dans la tête `CONNECT` (0 ID brut d'octet).
pub const CONNECT_PAIR_FEAT_DIM: usize = 6;

struct ConnectPairStats {
    obs_edge_sigs: BTreeMap<(SymbolId, u8, SymbolId), usize>,
    missing_edge_sigs: BTreeMap<(SymbolId, u8, SymbolId), usize>,
    missing_node_syms: BTreeMap<SymbolId, usize>,
    obs_node_syms: BTreeMap<SymbolId, usize>,
    max_obs_out: usize,
    max_obs_in: usize,
    obs_has_2cycle: bool,
    hyp_struct_out: Vec<usize>,
    hyp_struct_in: Vec<usize>,
    total_obs_edges: f64,
    total_missing_edges: f64,
    total_obs_nodes: f64,
    total_missing_nodes: f64,
}

impl ConnectPairStats {
    fn build(state: &HypothesisGraph, obs: &ObservationGraph, residual: &ResidualGraph) -> Self {
        let n_obs = obs.node_count();
        let mut obs_out = vec![0usize; n_obs];
        let mut obs_in = vec![0usize; n_obs];
        let mut obs_has_2cycle = false;
        let mut obs_edge_sigs: BTreeMap<(SymbolId, u8, SymbolId), usize> = BTreeMap::new();

        for &e in &obs.edges {
            obs_out[e.src] += 1;
            obs_in[e.dst] += 1;
            if obs
                .edges
                .contains(&crate::projector::ObsEdge::new(e.dst, e.src, e.rel))
            {
                obs_has_2cycle = true;
            }
            let sig = (obs.nodes[e.src], e.rel, obs.nodes[e.dst]);
            *obs_edge_sigs.entry(sig).or_insert(0) += 1;
        }

        let mut missing_edge_sigs: BTreeMap<(SymbolId, u8, SymbolId), usize> = BTreeMap::new();
        let mut total_missing_edges = 0usize;
        for re in &residual.edges {
            if let (ResidualEdgeKind::Missing(rel), Some((ou, ov))) = (re.kind, re.obs_endpoints) {
                let sig = (obs.nodes[ou], rel, obs.nodes[ov]);
                *missing_edge_sigs.entry(sig).or_insert(0) += 1;
                total_missing_edges += 1;
            }
        }

        let mut obs_node_syms: BTreeMap<SymbolId, usize> = BTreeMap::new();
        for &sym in &obs.nodes {
            *obs_node_syms.entry(sym).or_insert(0) += 1;
        }
        let mut missing_node_syms: BTreeMap<SymbolId, usize> = BTreeMap::new();
        for &(_, sym) in &residual.missing_nodes {
            *missing_node_syms.entry(sym).or_insert(0) += 1;
        }

        let n_hyp = state.node_count();
        let mut hyp_struct_out = vec![0usize; n_hyp];
        let mut hyp_struct_in = vec![0usize; n_hyp];
        for &e in state.edges() {
            if matches!(e.kind, EdgeKind::Struct(_)) {
                hyp_struct_out[e.src] += 1;
                hyp_struct_in[e.dst] += 1;
            }
        }

        Self {
            obs_edge_sigs,
            missing_edge_sigs,
            missing_node_syms,
            obs_node_syms,
            max_obs_out: obs_out.into_iter().max().unwrap_or(1).max(1),
            max_obs_in: obs_in.into_iter().max().unwrap_or(1).max(1),
            obs_has_2cycle,
            hyp_struct_out,
            hyp_struct_in,
            total_obs_edges: obs.edge_count().max(1) as f64,
            total_missing_edges: total_missing_edges.max(1) as f64,
            total_obs_nodes: n_obs.max(1) as f64,
            total_missing_nodes: residual.missing_nodes.len().max(1) as f64,
        }
    }

    fn features(&self, state: &HypothesisGraph, edge: Edge) -> Vec<f64> {
        let mut f = vec![0.0; CONNECT_PAIR_FEAT_DIM];
        match edge.kind {
            EdgeKind::Struct(rel) => {
                if let (NodeKind::Conc(su), NodeKind::Conc(sv)) =
                    (state.nodes()[edge.src], state.nodes()[edge.dst])
                {
                    let sig = (su, rel, sv);
                    f[0] = (*self.missing_edge_sigs.get(&sig).unwrap_or(&0) as f64)
                        / self.total_missing_edges;
                    f[1] = (*self.obs_edge_sigs.get(&sig).unwrap_or(&0) as f64)
                        / self.total_obs_edges;
                }
                let rev_exists = state
                    .edges()
                    .contains(&Edge::new(edge.dst, edge.src, EdgeKind::Struct(rel)));
                if rev_exists && !self.obs_has_2cycle {
                    f[2] = 1.0;
                }
                if self.hyp_struct_out[edge.src] >= self.max_obs_out
                    || self.hyp_struct_in[edge.dst] >= self.max_obs_in
                {
                    f[3] = 1.0;
                }
            }
            EdgeKind::InSeed => {
                if let NodeKind::Conc(sym) = state.nodes()[edge.dst] {
                    f[4] = (*self.missing_node_syms.get(&sym).unwrap_or(&0) as f64)
                        / self.total_missing_nodes;
                }
                f[5] = 1.0;
            }
            EdgeKind::InLhs | EdgeKind::InRhs => {
                if let NodeKind::Conc(sym) = state.nodes()[edge.dst] {
                    f[4] = (*self.obs_node_syms.get(&sym).unwrap_or(&0) as f64)
                        / self.total_obs_nodes;
                }
                f[5] = 1.0;
            }
            EdgeKind::SameAs => {}
        }
        f
    }
}

#[inline]
fn connect_category(kind: EdgeKind) -> usize {
    match kind {
        EdgeKind::InSeed => 0,
        EdgeKind::InLhs => 1,
        EdgeKind::InRhs => 2,
        EdgeKind::Struct(0) => 3,
        EdgeKind::Struct(_) => 4,
        EdgeKind::SameAs => 5,
    }
}

/// Calcule le plancher de pas atomiques nécessaire pour couvrir un effet `obs`
/// sans couper prématurément les trajectoires de construction.
#[inline]
fn effective_traj_steps(base_max: usize, obs: &ObservationGraph) -> usize {
    let min_for_effect = 2 * obs.node_count() + obs.edge_count() + 12;
    base_max.max(min_for_effect)
}

/// Échantillonne une trajectoire complète H_0 -> ... -> H_T -> STOP avec n'importe quelle `AbductivePolicy`
/// (utilisé pour l'amorçage off-policy via `ResidualGuidedHeuristicPolicy`, sans jamais lire H_true).
pub fn sample_policy_trajectory<P: AbductivePolicy>(
    policy: &mut P,
    obs: &ObservationGraph,
    graph_cfg: GraphConfig,
    ref_cfg: RefutationConfig,
    max_steps: usize,
    temperature: f64,
    epsilon_uniform: f64,
    rng_seed: u64,
) -> AbductiveTrajectory {
    let mut rng = SplitMix64::new(rng_seed);
    let mut state = HypothesisGraph::new(graph_cfg);
    let mut states = Vec::new();
    let mut residuals = Vec::new();
    let mut actions = Vec::new();

    let step_limit = max_steps.max(2);

    for step in 0..step_limit {
        let (_, residual) =
            project_and_residual(&state, obs, ref_cfg.projector_cfg, ref_cfg.residual_weights);
        let mask = state.forward_mask();
        let legal = mask.to_actions(&state.config().alphabet);

        let chosen = if step + 1 == step_limit && mask.allow_stop {
            ForwardAction::Stop
        } else {
            let masked_view = MaskedForwardView {
                mask,
                legal_actions: legal.clone(),
                blocked_count: 0,
            };
            let logits = policy.score_forward(&state, obs, &residual, &masked_view);
            let idx = sample_from_log_probs(&logits, temperature, epsilon_uniform, &mut rng);
            legal[idx]
        };

        states.push(state.clone());
        residuals.push(residual);
        actions.push(chosen);

        if matches!(chosen, ForwardAction::Stop) {
            break;
        }
        state = state
            .apply_forward(&chosen)
            .expect("Action issue de ForwardMask toujours légale");

        if state
            .forward_mask()
            .to_actions(&state.config().alphabet)
            .is_empty()
        {
            break;
        }
    }

    if actions.last() != Some(&ForwardAction::Stop) {
        let (_, residual) =
            project_and_residual(&state, obs, ref_cfg.projector_cfg, ref_cfg.residual_weights);
        states.push(state);
        residuals.push(residual);
        actions.push(ForwardAction::Stop);
    }

    let terminal_h = states.last().expect("Au moins H_1 présent");
    let report = evaluate_phi_sharp(terminal_h, obs, ref_cfg);

    AbductiveTrajectory {
        obs: obs.clone(),
        states,
        residuals,
        actions,
        log_phi_sharp: report.log_phi_sharp,
        terminal_residual_norm: report.residual.norm,
    }
}

// ============================================================================
// 2. PRNG déterministe minimal (SplitMix64 — zéro dépendance externe)
// ============================================================================

#[derive(Clone, Debug)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Flottant uniforme dans [0.0, 1.0).
    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Flottant uniforme dans [-limit, +limit].
    fn uniform_sym(&mut self, limit: f64) -> f64 {
        (2.0 * self.next_f64() - 1.0) * limit
    }
}

fn sample_from_log_probs(
    log_probs: &[f64],
    temperature: f64,
    epsilon_uniform: f64,
    rng: &mut SplitMix64,
) -> usize {
    debug_assert!(!log_probs.is_empty());
    if log_probs.len() == 1 {
        return 0;
    }
    if epsilon_uniform > 0.0 && rng.next_f64() < epsilon_uniform {
        return (rng.next_u64() as usize) % log_probs.len();
    }
    let temp = temperature.max(1e-6);
    let max_lp = log_probs
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut weights = Vec::with_capacity(log_probs.len());
    let mut sum = 0.0;
    for &lp in log_probs {
        let w = if lp.is_finite() {
            ((lp - max_lp) / temp).clamp(-500.0, 0.0).exp()
        } else {
            0.0
        };
        weights.push(w);
        sum += w;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return (rng.next_u64() as usize) % log_probs.len();
    }
    let mut target = rng.next_f64() * sum;
    for (idx, &w) in weights.iter().enumerate() {
        target -= w;
        if target <= 0.0 {
            return idx;
        }
    }
    log_probs.len() - 1
}

// ============================================================================
// 3. Moteur autograd vectoriel léger (mode inverse exact)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ParamSlice {
    offset: usize,
    len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ParamMat {
    offset: usize,
    rows: usize,
    cols: usize,
}

type TapeNode = usize;

enum TapeOp {
    Const,
    Param(ParamSlice),
    Add(TapeNode, TapeNode),
    AddBias(TapeNode, ParamSlice),
    Tanh(TapeNode),
    MatVec {
        mat: ParamMat,
        x: TapeNode,
    },
    Dot(TapeNode, TapeNode),
    Bilinear {
        mat: ParamMat,
        u: TapeNode,
        v: TapeNode,
    },
    MeanPool(Vec<TapeNode>),
    Concat(Vec<TapeNode>),
    LogSoftmax(TapeNode),
    Index(TapeNode, usize),
    LogSumExp(Vec<TapeNode>),
    LinearComb {
        const_bias: f64,
        terms: Vec<(f64, TapeNode)>,
    },
    Square(TapeNode),
}

struct TensorTape {
    values: Vec<Vec<f64>>,
    ops: Vec<TapeOp>,
}

impl TensorTape {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            ops: Vec::new(),
        }
    }

    fn push(&mut self, val: Vec<f64>, op: TapeOp) -> TapeNode {
        let id = self.values.len();
        self.values.push(val);
        self.ops.push(op);
        id
    }

    #[inline]
    fn val(&self, node: TapeNode) -> &[f64] {
        &self.values[node]
    }

    #[inline]
    fn scalar(&self, node: TapeNode) -> f64 {
        debug_assert_eq!(self.values[node].len(), 1);
        self.values[node][0]
    }

    fn const_vec(&mut self, v: Vec<f64>) -> TapeNode {
        self.push(v, TapeOp::Const)
    }

    fn const_scalar(&mut self, c: f64) -> TapeNode {
        self.push(vec![c], TapeOp::Const)
    }

    fn param(&mut self, params: &[f64], s: ParamSlice) -> TapeNode {
        self.push(params[s.offset..s.offset + s.len].to_vec(), TapeOp::Param(s))
    }

    fn add(&mut self, a: TapeNode, b: TapeNode) -> TapeNode {
        let va = &self.values[a];
        let vb = &self.values[b];
        debug_assert_eq!(va.len(), vb.len());
        let out: Vec<f64> = va.iter().zip(vb.iter()).map(|(&x, &y)| x + y).collect();
        self.push(out, TapeOp::Add(a, b))
    }

    fn add_bias(&mut self, params: &[f64], x: TapeNode, bias: ParamSlice) -> TapeNode {
        let vx = &self.values[x];
        debug_assert_eq!(vx.len(), bias.len);
        let b = &params[bias.offset..bias.offset + bias.len];
        let out: Vec<f64> = vx.iter().zip(b.iter()).map(|(&xi, &bi)| xi + bi).collect();
        self.push(out, TapeOp::AddBias(x, bias))
    }

    fn tanh(&mut self, x: TapeNode) -> TapeNode {
        let out: Vec<f64> = self.values[x].iter().map(|&v| v.tanh()).collect();
        self.push(out, TapeOp::Tanh(x))
    }

    fn matvec(&mut self, params: &[f64], mat: ParamMat, x: TapeNode) -> TapeNode {
        let vx = &self.values[x];
        debug_assert_eq!(vx.len(), mat.cols);
        let w = &params[mat.offset..mat.offset + mat.rows * mat.cols];
        let mut out = vec![0.0; mat.rows];
        for r in 0..mat.rows {
            let row = &w[r * mat.cols..(r + 1) * mat.cols];
            let mut acc = 0.0;
            for c in 0..mat.cols {
                acc += row[c] * vx[c];
            }
            out[r] = acc;
        }
        self.push(out, TapeOp::MatVec { mat, x })
    }

    fn linear(&mut self, params: &[f64], mat: ParamMat, bias: ParamSlice, x: TapeNode) -> TapeNode {
        let mv = self.matvec(params, mat, x);
        self.add_bias(params, mv, bias)
    }

    fn dot(&mut self, a: TapeNode, b: TapeNode) -> TapeNode {
        let va = &self.values[a];
        let vb = &self.values[b];
        debug_assert_eq!(va.len(), vb.len());
        let s: f64 = va.iter().zip(vb.iter()).map(|(&x, &y)| x * y).sum();
        self.push(vec![s], TapeOp::Dot(a, b))
    }

    fn bilinear(&mut self, params: &[f64], mat: ParamMat, u: TapeNode, v: TapeNode) -> TapeNode {
        let vu = &self.values[u];
        let vv = &self.values[v];
        debug_assert_eq!(vu.len(), mat.rows);
        debug_assert_eq!(vv.len(), mat.cols);
        let w = &params[mat.offset..mat.offset + mat.rows * mat.cols];
        let mut acc = 0.0;
        for r in 0..mat.rows {
            let ur = vu[r];
            let row = &w[r * mat.cols..(r + 1) * mat.cols];
            let mut inner = 0.0;
            for c in 0..mat.cols {
                inner += row[c] * vv[c];
            }
            acc += ur * inner;
        }
        self.push(vec![acc], TapeOp::Bilinear { mat, u, v })
    }

    fn mean_pool(&mut self, nodes: Vec<TapeNode>, dim: usize) -> TapeNode {
        if nodes.is_empty() {
            return self.const_vec(vec![0.0; dim]);
        }
        let inv_k = 1.0 / (nodes.len() as f64);
        let mut out = vec![0.0; dim];
        for &nid in &nodes {
            let v = &self.values[nid];
            debug_assert_eq!(v.len(), dim);
            for i in 0..dim {
                out[i] += v[i] * inv_k;
            }
        }
        self.push(out, TapeOp::MeanPool(nodes))
    }

    fn concat(&mut self, nodes: Vec<TapeNode>) -> TapeNode {
        let total: usize = nodes.iter().map(|&n| self.values[n].len()).sum();
        let mut out = Vec::with_capacity(total);
        for &nid in &nodes {
            out.extend_from_slice(&self.values[nid]);
        }
        self.push(out, TapeOp::Concat(nodes))
    }

    fn log_softmax(&mut self, x: TapeNode) -> TapeNode {
        let vx = &self.values[x];
        debug_assert!(!vx.is_empty());
        let max_v = vx.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = vx.iter().map(|&v| (v - max_v).exp()).sum();
        let lse = max_v + sum_exp.ln();
        let out: Vec<f64> = vx.iter().map(|&v| v - lse).collect();
        self.push(out, TapeOp::LogSoftmax(x))
    }

    fn index(&mut self, x: TapeNode, idx: usize) -> TapeNode {
        let v = self.values[x][idx];
        self.push(vec![v], TapeOp::Index(x, idx))
    }

    fn log_sum_exp(&mut self, scalars: Vec<TapeNode>) -> TapeNode {
        debug_assert!(!scalars.is_empty());
        if scalars.len() == 1 {
            return scalars[0];
        }
        let max_v = scalars
            .iter()
            .map(|&n| self.values[n][0])
            .fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = scalars
            .iter()
            .map(|&n| (self.values[n][0] - max_v).exp())
            .sum();
        let out = max_v + sum_exp.ln();
        self.push(vec![out], TapeOp::LogSumExp(scalars))
    }

    fn linear_comb(&mut self, const_bias: f64, terms: Vec<(f64, TapeNode)>) -> TapeNode {
        let mut acc = const_bias;
        for &(c, n) in &terms {
            acc += c * self.values[n][0];
        }
        self.push(vec![acc], TapeOp::LinearComb { const_bias, terms })
    }

    fn square(&mut self, x: TapeNode) -> TapeNode {
        let v = self.values[x][0];
        self.push(vec![v * v], TapeOp::Square(x))
    }

    /// Rétropropagation exacte depuis un nœud scalaire `loss_node` vers `param_grads`.
    fn backward(&self, loss_node: TapeNode, params: &[f64], param_grads: &mut [f64]) {
        debug_assert_eq!(self.values[loss_node].len(), 1);
        debug_assert_eq!(params.len(), param_grads.len());

        let mut grads: Vec<Vec<f64>> = self
            .values
            .iter()
            .map(|v| vec![0.0; v.len()])
            .collect();
        grads[loss_node][0] = 1.0;

        for node_id in (0..=loss_node).rev() {
            let gy = grads[node_id].clone();
            if gy.iter().all(|&g| g == 0.0) {
                continue;
            }
            match &self.ops[node_id] {
                TapeOp::Const => {}
                TapeOp::Param(s) => {
                    for i in 0..s.len {
                        param_grads[s.offset + i] += gy[i];
                    }
                }
                TapeOp::Add(a, b) => {
                    for i in 0..gy.len() {
                        grads[*a][i] += gy[i];
                        grads[*b][i] += gy[i];
                    }
                }
                TapeOp::AddBias(x, bias) => {
                    for i in 0..bias.len {
                        grads[*x][i] += gy[i];
                        param_grads[bias.offset + i] += gy[i];
                    }
                }
                TapeOp::Tanh(x) => {
                    let y = &self.values[node_id];
                    for i in 0..gy.len() {
                        grads[*x][i] += gy[i] * (1.0 - y[i] * y[i]);
                    }
                }
                TapeOp::MatVec { mat, x } => {
                    let vx = &self.values[*x];
                    let w = &params[mat.offset..mat.offset + mat.rows * mat.cols];
                    let gx = &mut grads[*x];
                    for r in 0..mat.rows {
                        let gyr = gy[r];
                        let row_off = mat.offset + r * mat.cols;
                        let row_w = &w[r * mat.cols..(r + 1) * mat.cols];
                        for c in 0..mat.cols {
                            param_grads[row_off + c] += gyr * vx[c];
                            gx[c] += gyr * row_w[c];
                        }
                    }
                }
                TapeOp::Dot(a, b) => {
                    let g = gy[0];
                    let va = &self.values[*a];
                    let vb = &self.values[*b];
                    for i in 0..va.len() {
                        grads[*a][i] += g * vb[i];
                        grads[*b][i] += g * va[i];
                    }
                }
                TapeOp::Bilinear { mat, u, v } => {
                    let g = gy[0];
                    let vu = &self.values[*u];
                    let vv = &self.values[*v];
                    let w = &params[mat.offset..mat.offset + mat.rows * mat.cols];
                    let mut gu_add = vec![0.0; mat.rows];
                    let mut gv_add = vec![0.0; mat.cols];
                    for r in 0..mat.rows {
                        let ur = vu[r];
                        let row_off = mat.offset + r * mat.cols;
                        let row_w = &w[r * mat.cols..(r + 1) * mat.cols];
                        let mut acc_u = 0.0;
                        for c in 0..mat.cols {
                            let vc = vv[c];
                            param_grads[row_off + c] += g * ur * vc;
                            acc_u += row_w[c] * vc;
                            gv_add[c] += g * ur * row_w[c];
                        }
                        gu_add[r] = g * acc_u;
                    }
                    for r in 0..mat.rows {
                        grads[*u][r] += gu_add[r];
                    }
                    for c in 0..mat.cols {
                        grads[*v][c] += gv_add[c];
                    }
                }
                TapeOp::MeanPool(nodes) => {
                    let inv_k = 1.0 / (nodes.len() as f64);
                    for &nid in nodes {
                        for i in 0..gy.len() {
                            grads[nid][i] += gy[i] * inv_k;
                        }
                    }
                }
                TapeOp::Concat(nodes) => {
                    let mut pos = 0usize;
                    for &nid in nodes {
                        let len = self.values[nid].len();
                        for i in 0..len {
                            grads[nid][i] += gy[pos + i];
                        }
                        pos += len;
                    }
                }
                TapeOp::LogSoftmax(x) => {
                    let y = &self.values[node_id];
                    let sum_gy: f64 = gy.iter().sum();
                    for i in 0..gy.len() {
                        grads[*x][i] += gy[i] - y[i].exp() * sum_gy;
                    }
                }
                TapeOp::Index(x, idx) => {
                    grads[*x][*idx] += gy[0];
                }
                TapeOp::LogSumExp(scalars) => {
                    let g = gy[0];
                    let lse = self.values[node_id][0];
                    for &nid in scalars {
                        let p = (self.values[nid][0] - lse).exp();
                        grads[nid][0] += g * p;
                    }
                }
                TapeOp::LinearComb { const_bias: _, terms } => {
                    let g = gy[0];
                    for &(c, nid) in terms {
                        grads[nid][0] += g * c;
                    }
                }
                TapeOp::Square(x) => {
                    let vx = self.values[*x][0];
                    grads[*x][0] += 2.0 * vx * gy[0];
                }
            }
        }
    }
}

// ============================================================================
// 4. Construction du graphe tripartite relationnel (G_E, H_t, R_t)
// ============================================================================

struct TripartiteGraph {
    /// Traits initiaux par nœud (taille `NODE_FEAT_DIM`).
    node_feats: Vec<Vec<f64>>,
    /// Résumé global du résidu (taille `GLOBAL_RES_DIM`).
    global_res: Vec<f64>,
    /// Indices des nœuds d'observation G_E dans `node_feats`.
    obs_range: (usize, usize),
    /// Indices des nœuds d'hypothèse H_t dans `node_feats`.
    hyp_range: (usize, usize),
    /// Indices des ancres symboliques Sigma_E dans `node_feats` (alignés sur `config.alphabet`).
    sym_range: (usize, usize),
    /// Arêtes dirigées `(src, dst, edge_type)` dans `0..NUM_TRIPARTITE_EDGE_TYPES`.
    edges: Vec<(usize, usize, usize)>,
}

fn build_tripartite_graph(
    state: &HypothesisGraph,
    obs: &ObservationGraph,
    residual: &ResidualGraph,
) -> TripartiteGraph {
    let n_obs = obs.node_count();
    let n_hyp = state.node_count();
    let alphabet = &state.config().alphabet;
    let n_sym = alphabet.len();

    let obs_offset = 0usize;
    let hyp_offset = n_obs;
    let sym_offset = n_obs + n_hyp;
    let total_nodes = n_obs + n_hyp + n_sym;

    let mut sym_to_local: BTreeMap<SymbolId, usize> = BTreeMap::new();
    for (idx, &sym) in alphabet.iter().enumerate() {
        sym_to_local.insert(sym, idx);
    }

    // Statuts d'alignement des nœuds d'observation dans R_t.
    let mut obs_matched = vec![false; n_obs];
    let mut obs_conflict = vec![false; n_obs];
    let mut obs_missing = vec![false; n_obs];
    for al in &residual.aligned_nodes {
        match al.status {
            NodeAlignmentStatus::Matched(_) => obs_matched[al.obs_node] = true,
            NodeAlignmentStatus::Conflict { .. } => obs_conflict[al.obs_node] = true,
        }
    }
    for &(u, _) in &residual.missing_nodes {
        obs_missing[u] = true;
    }

    // Comptes résiduels par symbole (manquants / excédentaires) pour les ancres Sigma_E.
    let mut sym_missing_cnt = vec![0usize; n_sym];
    for &(_, sym) in &residual.missing_nodes {
        if let Some(&idx) = sym_to_local.get(&sym) {
            sym_missing_cnt[idx] += 1;
        }
    }
    let mut sym_excess_cnt = vec![0usize; n_sym];
    for &(_, sym) in &residual.excess_nodes {
        if let Some(&idx) = sym_to_local.get(&sym) {
            sym_excess_cnt[idx] += 1;
        }
    }

    // Comptes d'arêtes manquantes/conflits incidentes par nœud d'observation.
    let mut obs_err_edges = vec![0usize; n_obs];
    let mut edges: Vec<(usize, usize, usize)> = Vec::new();

    // 1. Arêtes structurelles de G_E (types 0 et 1).
    for &e in &obs.edges {
        let et = if e.rel == 0 { 0 } else { 1 };
        edges.push((obs_offset + e.src, obs_offset + e.dst, et));
    }

    // 2. Arêtes d'erreur du résidu R_t sur G_E (types 2 : Missing, 3 : Conflict).
    for re in &residual.edges {
        if let Some((u, v)) = re.obs_endpoints {
            match re.kind {
                ResidualEdgeKind::Missing(_) => {
                    edges.push((obs_offset + u, obs_offset + v, 2));
                    obs_err_edges[u] += 1;
                    obs_err_edges[v] += 1;
                }
                ResidualEdgeKind::Conflict { .. } => {
                    edges.push((obs_offset + u, obs_offset + v, 3));
                    obs_err_edges[u] += 1;
                    obs_err_edges[v] += 1;
                }
                ResidualEdgeKind::Matched(_) | ResidualEdgeKind::Excess(_) => {}
            }
        }
    }

    // 3. Arêtes de l'hypothèse H_t (types 4..=9).
    for &e in state.edges() {
        let u = hyp_offset + e.src;
        let v = hyp_offset + e.dst;
        match e.kind {
            EdgeKind::InSeed => edges.push((u, v, 4)),
            EdgeKind::InLhs => edges.push((u, v, 5)),
            EdgeKind::InRhs => edges.push((u, v, 6)),
            EdgeKind::Struct(0) => edges.push((u, v, 7)),
            EdgeKind::Struct(_) => edges.push((u, v, 8)),
            EdgeKind::SameAs => {
                // Co-référence symétrique entre deux VAR.
                edges.push((u, v, 9));
                edges.push((v, u, 9));
            }
        }
    }

    // 4. Arêtes d'ancrage symbolique (type 10 : Obs->Sym, type 11 : Hyp->Sym).
    for (u, &sym) in obs.nodes.iter().enumerate() {
        if let Some(&s_idx) = sym_to_local.get(&sym) {
            edges.push((obs_offset + u, sym_offset + s_idx, 10));
        }
    }
    for (v, &kind) in state.nodes().iter().enumerate() {
        if let NodeKind::Conc(sym) = kind {
            if let Some(&s_idx) = sym_to_local.get(&sym) {
                edges.push((hyp_offset + v, sym_offset + s_idx, 11));
            }
        }
    }

    // Degrés entrants et sortants sur le graphe tripartite.
    let mut in_deg = vec![0usize; total_nodes];
    let mut out_deg = vec![0usize; total_nodes];
    for &(u, v, _) in &edges {
        out_deg[u] += 1;
        in_deg[v] += 1;
    }

    let norm_scaled = residual.norm / (residual.norm + 4.0);
    let denom_obs = (n_obs.max(1)) as f64;
    let mut node_feats = vec![vec![0.0; NODE_FEAT_DIM]; total_nodes];

    // Traits des nœuds d'observation.
    for u in 0..n_obs {
        let f = &mut node_feats[obs_offset + u];
        f[0] = 1.0;
        f[6] = f64::from(u8::from(obs_matched[u]));
        f[7] = f64::from(u8::from(obs_missing[u]));
        f[8] = f64::from(u8::from(obs_conflict[u]));
        let err_e = obs_err_edges[u] as f64;
        f[9] = err_e / (err_e + 1.0);
        f[11] = (in_deg[obs_offset + u] as f64) / ((in_deg[obs_offset + u] + 1) as f64);
        f[12] = (out_deg[obs_offset + u] as f64) / ((out_deg[obs_offset + u] + 1) as f64);
        f[13] = norm_scaled;
    }

    // Traits des nœuds d'hypothèse.
    for (v, &kind) in state.nodes().iter().enumerate() {
        let gid = hyp_offset + v;
        let f = &mut node_feats[gid];
        match kind {
            NodeKind::Seed => f[1] = 1.0,
            NodeKind::Rule => f[2] = 1.0,
            NodeKind::Conc(_) => f[3] = 1.0,
            NodeKind::Var => f[4] = 1.0,
        }
        let has_mem = state.membership(v).is_some();
        f[9] = f64::from(u8::from(has_mem));
        let deg_h = state.total_degree(v) as f64;
        f[10] = deg_h / (deg_h + 1.0);
        f[11] = (in_deg[gid] as f64) / ((in_deg[gid] + 1) as f64);
        f[12] = (out_deg[gid] as f64) / ((out_deg[gid] + 1) as f64);
        f[13] = norm_scaled;
    }

    // Traits des ancres symboliques Sigma_E.
    for s in 0..n_sym {
        let gid = sym_offset + s;
        let f = &mut node_feats[gid];
        f[5] = 1.0;
        f[9] = (sym_missing_cnt[s] as f64) / denom_obs;
        f[10] = (sym_excess_cnt[s] as f64) / denom_obs;
        f[11] = (in_deg[gid] as f64) / ((in_deg[gid] + 1) as f64);
        f[12] = (out_deg[gid] as f64) / ((out_deg[gid] + 1) as f64);
        f[13] = norm_scaled;
    }

    let s = &residual.summary;
    let global_res = vec![
        norm_scaled,
        (s.missing_nodes as f64) / denom_obs,
        (s.excess_nodes as f64) / denom_obs,
        (s.conflict_nodes as f64) / denom_obs,
        ((s.missing_edges + s.excess_edges + s.conflict_edges) as f64)
            / ((obs.edge_count().max(1)) as f64),
    ];

    TripartiteGraph {
        node_feats,
        global_res,
        obs_range: (obs_offset, obs_offset + n_obs),
        hyp_range: (hyp_offset, hyp_offset + n_hyp),
        sym_range: (sym_offset, sym_offset + n_sym),
        edges,
    }
}

// ============================================================================
// 5. Architecture du Backbone Relationnel Non-Réciproque & Têtes GFlowNet
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackboneConfig {
    /// Dimension cachée D des représentations de nœuds et du contexte global.
    pub hidden_dim: usize,
    /// Nombre L de couches de passage de messages non-réciproques.
    pub num_layers: usize,
}

impl Default for BackboneConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 24,
            num_layers: 2,
        }
    }
}

#[derive(Clone, Debug)]
struct LayerLayout {
    /// Matrices de messages sens direct (src -> dst) pour chaque type d'arête.
    w_fwd: [ParamMat; NUM_TRIPARTITE_EDGE_TYPES],
    /// Matrices de messages sens inverse (dst -> src), distinctes (non-réciproques).
    w_bwd: [ParamMat; NUM_TRIPARTITE_EDGE_TYPES],
    /// Transformation propre du nœud dans la mise à jour résiduelle.
    w_self: ParamMat,
    b_layer: ParamSlice,
}

#[derive(Clone, Debug)]
struct ParamLayout {
    in_w: ParamMat,
    in_b: ParamSlice,
    layers: Vec<LayerLayout>,

    // Agrégation globale g = tanh(W_g * [pool_obs, pool_hyp, pool_sym, global_res] + b_g)
    global_w: ParamMat,
    global_b: ParamSlice,

    // Tête scalaire log F_theta(H_t, E)
    flow_w1: ParamMat,
    flow_b1: ParamSlice,
    flow_w2: ParamSlice,
    flow_b2: ParamSlice,

    // Têtes factorisées P_F
    pf_fam_w: ParamMat,
    pf_fam_b: ParamSlice,
    pf_add_rule_w: ParamSlice,
    pf_add_rule_b: ParamSlice,
    pf_add_conc_sym_w: ParamSlice,
    pf_add_conc_ctx_w: ParamSlice,
    pf_add_conc_b: ParamSlice,
    pf_conn_type_w: ParamMat,
    pf_conn_type_b: ParamSlice,
    pf_conn_bilin: [ParamMat; NUM_CONNECT_CATS],
    pf_conn_pair_w: ParamSlice,
    pf_abs_w: ParamSlice,
    pf_abs_b: ParamSlice,

    // Têtes factorisées P_B
    pb_fam_w: ParamMat,
    pb_fam_b: ParamSlice,
    pb_del_node_w: ParamSlice,
    pb_del_node_b: ParamSlice,
    pb_del_edge_bilin: [ParamMat; NUM_CONNECT_CATS],
    pb_del_edge_b: ParamSlice,
    pb_unabs_node_w: ParamSlice,
    pb_unabs_sym_bilin: ParamMat,

    total_params: usize,
}

struct ParamAllocator {
    offset: usize,
}

impl ParamAllocator {
    fn new() -> Self {
        Self { offset: 0 }
    }

    fn slice(&mut self, len: usize) -> ParamSlice {
        let s = ParamSlice {
            offset: self.offset,
            len,
        };
        self.offset += len;
        s
    }

    fn mat(&mut self, rows: usize, cols: usize) -> ParamMat {
        let m = ParamMat {
            offset: self.offset,
            rows,
            cols,
        };
        self.offset += rows * cols;
        m
    }
}

impl ParamLayout {
    fn build(cfg: BackboneConfig) -> Self {
        let d = cfg.hidden_dim;
        let mut a = ParamAllocator::new();

        let in_w = a.mat(d, NODE_FEAT_DIM);
        let in_b = a.slice(d);

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            let w_fwd = std::array::from_fn(|_| a.mat(d, d));
            let w_bwd = std::array::from_fn(|_| a.mat(d, d));
            let w_self = a.mat(d, d);
            let b_layer = a.slice(d);
            layers.push(LayerLayout {
                w_fwd,
                w_bwd,
                w_self,
                b_layer,
            });
        }

        let global_w = a.mat(d, 3 * d + GLOBAL_RES_DIM);
        let global_b = a.slice(d);

        let flow_w1 = a.mat(d, d);
        let flow_b1 = a.slice(d);
        let flow_w2 = a.slice(d);
        let flow_b2 = a.slice(1);

        let pf_fam_w = a.mat(4, d);
        let pf_fam_b = a.slice(4);
        let pf_add_rule_w = a.slice(d);
        let pf_add_rule_b = a.slice(1);
        let pf_add_conc_sym_w = a.slice(d);
        let pf_add_conc_ctx_w = a.slice(d);
        let pf_add_conc_b = a.slice(1);
        let pf_conn_type_w = a.mat(NUM_CONNECT_CATS, d);
        let pf_conn_type_b = a.slice(NUM_CONNECT_CATS);
        let pf_conn_bilin = std::array::from_fn(|_| a.mat(d, d));
        let pf_conn_pair_w = a.slice(CONNECT_PAIR_FEAT_DIM);
        let pf_abs_w = a.slice(d);
        let pf_abs_b = a.slice(1);

        let pb_fam_w = a.mat(4, d);
        let pb_fam_b = a.slice(4);
        let pb_del_node_w = a.slice(d);
        let pb_del_node_b = a.slice(1);
        let pb_del_edge_bilin = std::array::from_fn(|_| a.mat(d, d));
        let pb_del_edge_b = a.slice(NUM_CONNECT_CATS);
        let pb_unabs_node_w = a.slice(d);
        let pb_unabs_sym_bilin = a.mat(d, d);

        Self {
            in_w,
            in_b,
            layers,
            global_w,
            global_b,
            flow_w1,
            flow_b1,
            flow_w2,
            flow_b2,
            pf_fam_w,
            pf_fam_b,
            pf_add_rule_w,
            pf_add_rule_b,
            pf_add_conc_sym_w,
            pf_add_conc_ctx_w,
            pf_add_conc_b,
            pf_conn_type_w,
            pf_conn_type_b,
            pf_conn_bilin,
            pf_conn_pair_w,
            pf_abs_w,
            pf_abs_b,
            pb_fam_w,
            pb_fam_b,
            pb_del_node_w,
            pb_del_node_b,
            pb_del_edge_bilin,
            pb_del_edge_b,
            pb_unabs_node_w,
            pb_unabs_sym_bilin,
            total_params: a.offset,
        }
    }
}

struct EncodedState {
    hyp_nodes: Vec<TapeNode>,
    sym_nodes: Vec<TapeNode>,
    global_ctx: TapeNode,
}

/// Backbone relationnel non-réciproque complet de Phisharp V1.
#[derive(Clone, Debug)]
pub struct RelationalBackbone {
    cfg: BackboneConfig,
    layout: ParamLayout,
    params: Vec<f64>,
}

impl RelationalBackbone {
    /// Initialise le réseau avec Xavier/Glorot uniforme déterministe.
    pub fn new(cfg: BackboneConfig, seed: u64) -> Result<Self, &'static str> {
        if cfg.hidden_dim < 2 {
            return Err("hidden_dim doit être >= 2");
        }
        if cfg.num_layers == 0 {
            return Err("num_layers doit être >= 1");
        }
        let layout = ParamLayout::build(cfg);
        let mut params = vec![0.0; layout.total_params];
        let mut rng = SplitMix64::new(seed);

        let init_mat = |params: &mut [f64], m: ParamMat, rng: &mut SplitMix64, scale: f64| {
            let limit = scale * (6.0 / ((m.rows + m.cols) as f64)).sqrt();
            for i in 0..(m.rows * m.cols) {
                params[m.offset + i] = rng.uniform_sym(limit);
            }
        };
        let init_vec = |params: &mut [f64], s: ParamSlice, rng: &mut SplitMix64, scale: f64| {
            let limit = scale / (s.len as f64).sqrt();
            for i in 0..s.len {
                params[s.offset + i] = rng.uniform_sym(limit);
            }
        };

        init_mat(&mut params, layout.in_w, &mut rng, 1.0);
        for layer in &layout.layers {
            // Deux passes indépendantes garantissent l'asymétrie initiale W_fwd != W_bwd.
            for &m in &layer.w_fwd {
                init_mat(&mut params, m, &mut rng, 0.5);
            }
            for &m in &layer.w_bwd {
                init_mat(&mut params, m, &mut rng, 0.5);
            }
            init_mat(&mut params, layer.w_self, &mut rng, 0.7);
        }

        init_mat(&mut params, layout.global_w, &mut rng, 1.0);
        init_mat(&mut params, layout.flow_w1, &mut rng, 1.0);
        init_vec(&mut params, layout.flow_w2, &mut rng, 0.5);

        init_mat(&mut params, layout.pf_fam_w, &mut rng, 0.5);
        init_vec(&mut params, layout.pf_add_rule_w, &mut rng, 0.5);
        init_vec(&mut params, layout.pf_add_conc_sym_w, &mut rng, 0.5);
        init_vec(&mut params, layout.pf_add_conc_ctx_w, &mut rng, 0.5);
        init_mat(&mut params, layout.pf_conn_type_w, &mut rng, 0.5);
        for &m in &layout.pf_conn_bilin {
            init_mat(&mut params, m, &mut rng, 0.5);
        }
        init_vec(&mut params, layout.pf_conn_pair_w, &mut rng, 0.5);
        init_vec(&mut params, layout.pf_abs_w, &mut rng, 0.5);

        init_mat(&mut params, layout.pb_fam_w, &mut rng, 0.5);
        init_vec(&mut params, layout.pb_del_node_w, &mut rng, 0.5);
        for &m in &layout.pb_del_edge_bilin {
            init_mat(&mut params, m, &mut rng, 0.5);
        }
        init_vec(&mut params, layout.pb_unabs_node_w, &mut rng, 0.5);
        init_mat(&mut params, layout.pb_unabs_sym_bilin, &mut rng, 0.5);

        Ok(Self {
            cfg,
            layout,
            params,
        })
    }

    /// Reconstruit le réseau depuis un vecteur de poids chargé (ex. `out/sharp.phis`).
    pub fn from_params(cfg: BackboneConfig, params: Vec<f64>) -> Result<Self, &'static str> {
        if cfg.hidden_dim < 2 || cfg.num_layers == 0 {
            return Err("BackboneConfig invalide");
        }
        let layout = ParamLayout::build(cfg);
        if params.len() != layout.total_params {
            return Err("Nombre de paramètres incompatible avec BackboneConfig");
        }
        Ok(Self {
            cfg,
            layout,
            params,
        })
    }

    #[inline]
    pub fn config(&self) -> BackboneConfig {
        self.cfg
    }

    #[inline]
    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    #[inline]
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    #[inline]
    pub fn params_mut(&mut self) -> &mut [f64] {
        &mut self.params
    }

    /// Encode (G_E, H_t, R_t) sur le `TensorTape` via L couches non-réciproques.
    fn encode_on_tape(
        &self,
        tape: &mut TensorTape,
        state: &HypothesisGraph,
        obs: &ObservationGraph,
        residual: &ResidualGraph,
    ) -> EncodedState {
        let d = self.cfg.hidden_dim;
        let tri = build_tripartite_graph(state, obs, residual);
        let n = tri.node_feats.len();

        // Projection initiale des traits relationnels.
        let mut h: Vec<TapeNode> = Vec::with_capacity(n);
        for feat in tri.node_feats {
            let x = tape.const_vec(feat);
            let lin = tape.linear(&self.params, self.layout.in_w, self.layout.in_b, x);
            h.push(tape.tanh(lin));
        }

        // L couches de passage de messages dirigés à matrices non-réciproques.
        for layer in &self.layout.layers {
            let mut inbox: Vec<Vec<TapeNode>> = vec![Vec::new(); n];
            for &(u, v, et) in &tri.edges {
                // Message direct u -> v via W_fwd[et].
                let msg_fwd = tape.matvec(&self.params, layer.w_fwd[et], h[u]);
                inbox[v].push(msg_fwd);
                // Message retour v -> u via W_bwd[et] (strictement non-réciproque).
                let msg_bwd = tape.matvec(&self.params, layer.w_bwd[et], h[v]);
                inbox[u].push(msg_bwd);
            }

            let mut next_h = Vec::with_capacity(n);
            for (u, msgs) in inbox.into_iter().enumerate() {
                let self_term = tape.matvec(&self.params, layer.w_self, h[u]);
                let pre = if msgs.is_empty() {
                    tape.add_bias(&self.params, self_term, layer.b_layer)
                } else {
                    let agg = tape.mean_pool(msgs, d);
                    let sum = tape.add(self_term, agg);
                    tape.add_bias(&self.params, sum, layer.b_layer)
                };
                let delta = tape.tanh(pre);
                // Connexion résiduelle.
                next_h.push(tape.add(h[u], delta));
            }
            h = next_h;
        }

        let obs_pool = tape.mean_pool(h[tri.obs_range.0..tri.obs_range.1].to_vec(), d);
        let hyp_pool = tape.mean_pool(h[tri.hyp_range.0..tri.hyp_range.1].to_vec(), d);
        let sym_pool = tape.mean_pool(h[tri.sym_range.0..tri.sym_range.1].to_vec(), d);
        let res_vec = tape.const_vec(tri.global_res);

        let concat_ctx = tape.concat(vec![obs_pool, hyp_pool, sym_pool, res_vec]);
        let g_lin = tape.linear(
            &self.params,
            self.layout.global_w,
            self.layout.global_b,
            concat_ctx,
        );
        let global_ctx = tape.tanh(g_lin);

        EncodedState {
            hyp_nodes: h[tri.hyp_range.0..tri.hyp_range.1].to_vec(),
            sym_nodes: h[tri.sym_range.0..tri.sym_range.1].to_vec(),
            global_ctx,
        }
    }

    /// Tête scalaire log F_theta(H_t, E) sur le tape.
    fn log_flow_on_tape(&self, tape: &mut TensorTape, enc: &EncodedState) -> TapeNode {
        let h1_lin = tape.linear(
            &self.params,
            self.layout.flow_w1,
            self.layout.flow_b1,
            enc.global_ctx,
        );
        let h1 = tape.tanh(h1_lin);
        let w2 = tape.param(&self.params, self.layout.flow_w2);
        let dot = tape.dot(w2, h1);
        tape.add_bias(&self.params, dot, self.layout.flow_b2)
    }

    fn forward_log_probs_on_tape(
        &self,
        tape: &mut TensorTape,
        enc: &EncodedState,
        state: &HypothesisGraph,
        obs: &ObservationGraph,
        residual: &ResidualGraph,
        mask: &ForwardMask,
    ) -> Vec<TapeNode>
        let active_fams: Vec<usize> = (0..4).filter(|&i| mask.family_mask[i]).collect();
        if active_fams.is_empty() {
            return Vec::new();
        }

        // 1. Probabilité de famille p(op | g) masquée aux familles actives.
        let fam_all = tape.linear(
            &self.params,
            self.layout.pf_fam_w,
            self.layout.pf_fam_b,
            enc.global_ctx,
        );
        let fam_active_logits: Vec<TapeNode> = active_fams
            .iter()
            .map(|&f| tape.index(fam_all, f))
            .collect();
        let fam_concat = tape.concat(fam_active_logits);
        let fam_log_probs = tape.log_softmax(fam_concat);

        let mut fam_lp: [Option<TapeNode>; 4] = [None; 4];
        for (pos, &f) in active_fams.iter().enumerate() {
            fam_lp[f] = Some(tape.index(fam_log_probs, pos));
        }

        let mut action_log_probs = Vec::new();

        // 2. Famille 0 : ADD_NODE (Rule + Conc(c)).
        if let Some(lp_fam0) = fam_lp[0] {
            let mut add_scores = Vec::new();
            if mask.allow_add_rule {
                let w = tape.param(&self.params, self.layout.pf_add_rule_w);
                let dot = tape.dot(w, enc.global_ctx);
                add_scores.push(tape.add_bias(&self.params, dot, self.layout.pf_add_rule_b));
            }
            let w_sym = tape.param(&self.params, self.layout.pf_add_conc_sym_w);
            let w_ctx = tape.param(&self.params, self.layout.pf_add_conc_ctx_w);
            let ctx_term = tape.dot(w_ctx, enc.global_ctx);
            for (sym_idx, &allowed) in mask.allow_add_conc.iter().enumerate() {
                if allowed {
                    let sym_term = tape.dot(w_sym, enc.sym_nodes[sym_idx]);
                    let sum = tape.add(sym_term, ctx_term);
                    add_scores.push(tape.add_bias(&self.params, sum, self.layout.pf_add_conc_b));
                }
            }
            let add_cat = tape.concat(add_scores);
            let add_lps = tape.log_softmax(add_cat);
            let count = tape.val(add_lps).len();
            for i in 0..count {
                let lp_i = tape.index(add_lps, i);
                action_log_probs.push(tape.add(lp_fam0, lp_i));
            }
        }

        // 3. Famille 1 : CONNECT(u, v, rel) factorisée par type de relation puis score bilinéaire.
        if let Some(lp_fam1) = fam_lp[1] {
            let mut by_cat: [Vec<(usize, Edge)>; NUM_CONNECT_CATS] =
                std::array::from_fn(|_| Vec::new());
            for (orig_idx, &edge) in mask.connect_edges.iter().enumerate() {
                by_cat[connect_category(edge.kind)].push((orig_idx, edge));
            }

            let active_cats: Vec<usize> = (0..NUM_CONNECT_CATS)
                .filter(|&c| !by_cat[c].is_empty())
                .collect();

            let cat_all = tape.linear(
                &self.params,
                self.layout.pf_conn_type_w,
                self.layout.pf_conn_type_b,
                enc.global_ctx,
            );
            let cat_logits: Vec<TapeNode> = active_cats
                .iter()
                .map(|&c| tape.index(cat_all, c))
                .collect();
            let cat_vec = tape.concat(cat_logits);
            let cat_lps = tape.log_softmax(cat_vec);
            
            let pair_stats = ConnectPairStats::build(state, obs, residual);
            let w_pair = tape.param(&self.params, self.layout.pf_conn_pair_w);

            let mut scattered: Vec<Option<TapeNode>> = vec![None; mask.connect_edges.len()];
            for (cat_pos, &cat) in active_cats.iter().enumerate() {
                let lp_cat = tape.index(cat_lps, cat_pos);
                let lp_prefix = tape.add(lp_fam1, lp_cat);

                let pair_scores: Vec<TapeNode> = by_cat[cat]
                    .iter()
                    .map(|&(_, edge)| {
                        let bil = tape.bilinear(
                            &self.params,
                            self.layout.pf_conn_bilin[cat],
                            enc.hyp_nodes[edge.src],
                            enc.hyp_nodes[edge.dst],
                        );
                        let pf = tape.const_vec(pair_stats.features(state, edge));
                        let pf_dot = tape.dot(w_pair, pf);
                        tape.add(bil, pf_dot)
                    })
                    .collect();
                let pair_vec = tape.concat(pair_scores);
                let pair_lps = tape.log_softmax(pair_vec);

                for (p_idx, &(orig_idx, _)) in by_cat[cat].iter().enumerate() {
                    let lp_pair = tape.index(pair_lps, p_idx);
                    scattered[orig_idx] = Some(tape.add(lp_prefix, lp_pair));
                }
            }

            for lp in scattered {
                action_log_probs.push(lp.expect("Chaque arête légale CONNECT possède un log_p"));
            }
        }

        // 4. Famille 2 : ABSTRACT(v).
        if let Some(lp_fam2) = fam_lp[2] {
            let w_abs = tape.param(&self.params, self.layout.pf_abs_w);
            let mut abs_scores = Vec::new();
            for (node, &allowed) in mask.abstract_nodes.iter().enumerate() {
                if allowed {
                    let dot = tape.dot(w_abs, enc.hyp_nodes[node]);
                    abs_scores.push(tape.add_bias(&self.params, dot, self.layout.pf_abs_b));
                }
            }
            let abs_vec = tape.concat(abs_scores);
            let abs_lps = tape.log_softmax(abs_vec);
            let count = tape.val(abs_lps).len();
            for i in 0..count {
                let lp_i = tape.index(abs_lps, i);
                action_log_probs.push(tape.add(lp_fam2, lp_i));
            }
        }

        // 5. Famille 3 : STOP.
        if let Some(lp_fam3) = fam_lp[3] {
            action_log_probs.push(lp_fam3);
        }

        action_log_probs
    }

    /// Tête factorisée P_B sur le tape : retourne les log-probabilités exactes
    /// alignées 1-à-1 avec `mask.to_actions(state.edges(), &state.config().alphabet)`.
    fn backward_log_probs_on_tape(
        &self,
        tape: &mut TensorTape,
        enc: &EncodedState,
        state: &HypothesisGraph,
        mask: &BackwardMask,
    ) -> Vec<TapeNode> {
        let active_fams: Vec<usize> = (0..4).filter(|&i| mask.family_mask[i]).collect();
        if active_fams.is_empty() {
            return Vec::new();
        }
        // Sur un état STOP, `UnStop` est l'unique parent déterministe (log P_B = 0.0).
        if mask.allow_un_stop && active_fams.len() == 1 {
            return vec![tape.const_scalar(0.0)];
        }

        let fam_all = tape.linear(
            &self.params,
            self.layout.pb_fam_w,
            self.layout.pb_fam_b,
            enc.global_ctx,
        );
        let fam_active_logits: Vec<TapeNode> = active_fams
            .iter()
            .map(|&f| tape.index(fam_all, f))
            .collect();
        let fam_concat = tape.concat(fam_active_logits);
        let fam_log_probs = tape.log_softmax(fam_concat);

        let mut fam_lp: [Option<TapeNode>; 4] = [None; 4];
        for (pos, &f) in active_fams.iter().enumerate() {
            fam_lp[f] = Some(tape.index(fam_log_probs, pos));
        }

        let mut action_log_probs = Vec::new();

        // 1. Famille 0 : DEL_NODE(v).
        if let Some(lp_fam0) = fam_lp[0] {
            let w_dn = tape.param(&self.params, self.layout.pb_del_node_w);
            let mut scores = Vec::new();
            for (node, &allowed) in mask.del_nodes.iter().enumerate() {
                if allowed {
                    let dot = tape.dot(w_dn, enc.hyp_nodes[node]);
                    scores.push(tape.add_bias(&self.params, dot, self.layout.pb_del_node_b));
                }
            }
            let vec_s = tape.concat(scores);
            let lps = tape.log_softmax(vec_s);
            let count = tape.val(lps).len();
            for i in 0..count {
                let lp_i = tape.index(lps, i);
                action_log_probs.push(tape.add(lp_fam0, lp_i));
            }
        }

        // 2. Famille 1 : DEL_EDGE(e).
        if let Some(lp_fam1) = fam_lp[1] {
            let b_all = tape.param(&self.params, self.layout.pb_del_edge_b);
            let mut scores = Vec::new();
            for (&edge, &allowed) in state.edges().iter().zip(mask.del_edges.iter()) {
                if allowed {
                    let cat = connect_category(edge.kind);
                    let bil = tape.bilinear(
                        &self.params,
                        self.layout.pb_del_edge_bilin[cat],
                        enc.hyp_nodes[edge.src],
                        enc.hyp_nodes[edge.dst],
                    );
                    let b_cat = tape.index(b_all, cat);
                    scores.push(tape.add(bil, b_cat));
                }
            }
            let vec_s = tape.concat(scores);
            let lps = tape.log_softmax(vec_s);
            let count = tape.val(lps).len();
            for i in 0..count {
                let lp_i = tape.index(lps, i);
                action_log_probs.push(tape.add(lp_fam1, lp_i));
            }
        }

        // 3. Famille 2 : UN_ABSTRACT(v, c) factorisée P(v) * P(c | v).
        if let Some(lp_fam2) = fam_lp[2] {
            let active_vars: Vec<NodeId> = mask
                .un_abstract_nodes
                .iter()
                .enumerate()
                .filter_map(|(v, &ok)| ok.then_some(v))
                .collect();

            let w_un = tape.param(&self.params, self.layout.pb_unabs_node_w);
            let node_scores: Vec<TapeNode> = active_vars
                .iter()
                .map(|&v| tape.dot(w_un, enc.hyp_nodes[v]))
                .collect();
            let node_vec = tape.concat(node_scores);
            let node_lps = tape.log_softmax(node_vec);

            let n_sym = enc.sym_nodes.len();
            for (v_pos, &v) in active_vars.iter().enumerate() {
                let lp_v = tape.index(node_lps, v_pos);
                let lp_prefix = tape.add(lp_fam2, lp_v);

                let sym_scores: Vec<TapeNode> = (0..n_sym)
                    .map(|s_idx| {
                        tape.bilinear(
                            &self.params,
                            self.layout.pb_unabs_sym_bilin,
                            enc.hyp_nodes[v],
                            enc.sym_nodes[s_idx],
                        )
                    })
                    .collect();
                let sym_vec = tape.concat(sym_scores);
                let sym_lps = tape.log_softmax(sym_vec);

                for s_idx in 0..n_sym {
                    let lp_c = tape.index(sym_lps, s_idx);
                    action_log_probs.push(tape.add(lp_prefix, lp_c));
                }
            }
        }

        // 4. Famille 3 : UN_STOP.
        if let Some(lp_fam3) = fam_lp[3] {
            action_log_probs.push(lp_fam3);
        }

        action_log_probs
    }

    /// Inférence pure (sans rétropropagation) : log F_theta(H_t, E).
    pub fn predict_log_flow(
        &self,
        state: &HypothesisGraph,
        obs: &ObservationGraph,
        residual: &ResidualGraph,
    ) -> f64 {
        let mut tape = TensorTape::new();
        let enc = self.encode_on_tape(&mut tape, state, obs, residual);
        let lf = self.log_flow_on_tape(&mut tape, &enc);
        tape.scalar(lf)
    }

    /// Inférence pure : log P_F aligné sur `mask.to_actions(&state.config().alphabet)`.
    pub fn forward_log_probs(
        &self,
        state: &HypothesisGraph,
        obs: &ObservationGraph,
        residual: &ResidualGraph,
        mask: &ForwardMask,
    ) -> Vec<f64> {
        let mut tape = TensorTape::new();
        let enc = self.encode_on_tape(&mut tape, state, obs, residual);
        let lps = self.forward_log_probs_on_tape(&mut tape, &enc, state, obs, residual, mask);
        lps.into_iter().map(|n| tape.scalar(n)).collect()
    }

    /// Inférence pure : log P_B aligné sur `mask.to_actions(state.edges(), &state.config().alphabet)`.
    pub fn backward_log_probs(
        &self,
        state: &HypothesisGraph,
        obs: &ObservationGraph,
        residual: &ResidualGraph,
        mask: &BackwardMask,
    ) -> Vec<f64> {
        let mut tape = TensorTape::new();
        let enc = self.encode_on_tape(&mut tape, state, obs, residual);
        let lps = self.backward_log_probs_on_tape(&mut tape, &enc, state, mask);
        lps.into_iter().map(|n| tape.scalar(n)).collect()
    }
}

/// Branchement direct du `RelationalBackbone` sur `AskController` (`memory_controller.rs`).
impl AbductivePolicy for RelationalBackbone {
    fn score_forward(
        &mut self,
        state: &HypothesisGraph,
        obs: &ObservationGraph,
        residual: &ResidualGraph,
        masked: &MaskedForwardView,
    ) -> Vec<f64> {
        self.forward_log_probs(state, obs, residual, &masked.mask)
    }

    fn score_backward(
        &mut self,
        state: &HypothesisGraph,
        obs: &ObservationGraph,
        residual: &ResidualGraph,
        mask: &BackwardMask,
        _legal_backward: &[BackwardAction],
    ) -> Vec<f64> {
        self.backward_log_probs(state, obs, residual, mask)
    }
}

// ============================================================================
// 6. Trajectoires, regroupement canonique exact et perte Sub-Trajectory Balance
// ============================================================================

/// Trajectoire abductive complète H_0 -> ... -> H_T -> STOP avec son signal phi_sharp.
#[derive(Clone, Debug)]
pub struct AbductiveTrajectory {
    pub obs: ObservationGraph,
    /// États non-stoppés `[H_0, H_1, ..., H_T]` (longueur T + 1 >= 2).
    pub states: Vec<HypothesisGraph>,
    /// Résidus associés `[R_0, R_1, ..., R_T]` (même longueur que `states`).
    pub residuals: Vec<ResidualGraph>,
    /// Actions `[a_0, ..., a_{T-1}, Stop]` (longueur T + 1).
    pub actions: Vec<ForwardAction>,
    /// Signal terminal exact log phi_sharp(H_T, E) évalué par le banc de réfutation.
    pub log_phi_sharp: f64,
    /// Norme résiduelle finale ||R(H_T, E)|| (suivi métrique).
    pub terminal_residual_norm: f64,
}

/// Regroupe les indices des actions P_F de `legal_actions` menant au même graphe canonique que `chosen`.
fn canonical_forward_equivalence_indices(
    state: &HypothesisGraph,
    legal_actions: &[ForwardAction],
    chosen: &ForwardAction,
    next_state: &HypothesisGraph,
) -> Vec<usize> {
    match *chosen {
        // AddNode et Stop sont toujours canoniquement uniques.
        ForwardAction::AddNode(_) | ForwardAction::Stop => legal_actions
            .iter()
            .position(|a| a == chosen)
            .into_iter()
            .collect(),
        ForwardAction::Connect(target_edge) => {
            let target_key = next_state.canonical_key();
            let mut indices = Vec::new();
            for (idx, a) in legal_actions.iter().enumerate() {
                if let ForwardAction::Connect(e) = *a {
                    if e == target_edge {
                        indices.push(idx);
                    } else if e.kind == target_edge.kind {
                        if let Ok(cand) = state.apply_forward(a) {
                            if cand.canonical_key() == target_key {
                                indices.push(idx);
                            }
                        }
                    }
                }
            }
            indices
        }
        ForwardAction::Abstract(target_node) => {
            let target_kind = state.nodes()[target_node];
            let target_key = next_state.canonical_key();
            let mut indices = Vec::new();
            for (idx, a) in legal_actions.iter().enumerate() {
                if let ForwardAction::Abstract(v) = *a {
                    if v == target_node {
                        indices.push(idx);
                    } else if state.nodes()[v] == target_kind {
                        if let Ok(cand) = state.apply_forward(a) {
                            if cand.canonical_key() == target_key {
                                indices.push(idx);
                            }
                        }
                    }
                }
            }
            indices
        }
    }
}

/// Regroupe les indices des actions P_B sur `next_state` menant au même graphe canonique que `prev_state`.
fn canonical_backward_equivalence_indices(
    prev_state: &HypothesisGraph,
    next_state: &HypothesisGraph,
    chosen_forward: &ForwardAction,
    legal_backward: &[BackwardAction],
) -> Vec<usize> {
    let prev_key = prev_state.canonical_key();
    let mut indices = Vec::new();

    match *chosen_forward {
        ForwardAction::AddNode(kind) => {
            let expected_node_kind = match kind {
                AddNodeKind::Rule => NodeKind::Rule,
                AddNodeKind::Conc(c) => NodeKind::Conc(c),
            };
            for (idx, ba) in legal_backward.iter().enumerate() {
                if let BackwardAction::DelNode(v) = *ba {
                    if next_state.nodes()[v] == expected_node_kind {
                        if let Ok(cand) = next_state.apply_backward(ba) {
                            if cand.canonical_key() == prev_key {
                                indices.push(idx);
                            }
                        }
                    }
                }
            }
        }
        ForwardAction::Connect(target_edge) => {
            for (idx, ba) in legal_backward.iter().enumerate() {
                if let BackwardAction::DelEdge(e) = *ba {
                    if e == target_edge {
                        indices.push(idx);
                    } else if e.kind == target_edge.kind {
                        if let Ok(cand) = next_state.apply_backward(ba) {
                            if cand.canonical_key() == prev_key {
                                indices.push(idx);
                            }
                        }
                    }
                }
            }
        }
        ForwardAction::Abstract(target_node) => {
            let NodeKind::Conc(orig_sym) = prev_state.nodes()[target_node] else {
                return indices;
            };
            for (idx, ba) in legal_backward.iter().enumerate() {
                if let BackwardAction::UnAbstract { node, symbol } = *ba {
                    if symbol == orig_sym {
                        if node == target_node {
                            indices.push(idx);
                        } else if let Ok(cand) = next_state.apply_backward(ba) {
                            if cand.canonical_key() == prev_key {
                                indices.push(idx);
                            }
                        }
                    }
                }
            }
        }
        ForwardAction::Stop => {}
    }

    indices
}

/// Évalue la perte Sub-Trajectory Balance sur une trajectoire et accumule son gradient
/// (pondéré par `loss_scale`) dans `param_grads`.
pub fn subtb_loss_and_accumulate_grad(
    backbone: &RelationalBackbone,
    traj: &AbductiveTrajectory,
    subtb_lambda: f64,
    loss_scale: f64,
    param_grads: &mut [f64],
) -> f64 {
    let m = traj.states.len();
    assert!(m >= 2, "Une trajectoire SubTB doit contenir au moins H_0 et H_1");
    assert_eq!(traj.residuals.len(), m);
    assert_eq!(traj.actions.len(), m);
    assert_eq!(traj.actions[m - 1], ForwardAction::Stop);

    let mut tape = TensorTape::new();

    // 1. Encodage de tous les états H_0 .. H_T sur le même tape.
    let mut log_f: Vec<TapeNode> = Vec::with_capacity(m);
    let mut log_pf: Vec<TapeNode> = Vec::with_capacity(m);
    let mut log_pb: Vec<TapeNode> = Vec::with_capacity(m - 1);

    for t in 0..m {
        let state = &traj.states[t];
        let residual = &traj.residuals[t];
        let enc = backbone.encode_on_tape(&mut tape, state, &traj.obs, residual);

        log_f.push(backbone.log_flow_on_tape(&mut tape, &enc));

        // Transition avant t -> t + 1 (ou STOP si t == m - 1).
        let f_mask = state.forward_mask();
        let f_actions = f_mask.to_actions(&state.config().alphabet);
        let f_lps = backbone.forward_log_probs_on_tape(&mut tape, &enc, state, obs, residual, &f_mask);
        debug_assert_eq!(f_actions.len(), f_lps.len());

        let chosen_f = &traj.actions[t];
        let eq_f = if t + 1 < m {
            canonical_forward_equivalence_indices(state, &f_actions, chosen_f, &traj.states[t + 1])
        } else {
            f_actions
                .iter()
                .position(|a| a == &ForwardAction::Stop)
                .into_iter()
                .collect()
        };
        assert!(
            !eq_f.is_empty(),
            "Action avant {:?} introuvable à l'étape {t}",
            chosen_f
        );
        let f_nodes: Vec<TapeNode> = eq_f.into_iter().map(|i| f_lps[i]).collect();
        log_pf.push(tape.log_sum_exp(f_nodes));

        // Transition arrière t -> t - 1 (pour t >= 1).
        if t >= 1 {
            let b_mask = state.backward_mask();
            let b_actions = b_mask.to_actions(state.edges(), &state.config().alphabet);
            let b_lps = backbone.backward_log_probs_on_tape(&mut tape, &enc, state, &b_mask);
            debug_assert_eq!(b_actions.len(), b_lps.len());

            let eq_b = canonical_backward_equivalence_indices(
                &traj.states[t - 1],
                state,
                &traj.actions[t - 1],
                &b_actions,
            );
            assert!(
                !eq_b.is_empty(),
                "Inverse arrière introuvable à l'étape {t} pour {:?}",
                traj.actions[t - 1]
            );
            let b_nodes: Vec<TapeNode> = eq_b.into_iter().map(|i| b_lps[i]).collect();
            log_pb.push(tape.log_sum_exp(b_nodes));
        }
    }

    // 2. Assemblage exact de la perte Sub-Trajectory Balance sur tous les couples 0 <= i < j <= m.
    // Indice m représente le puits terminal H_T^top avec log F_m = log_phi_sharp et log P_B(m->m-1) = 0.
    let lam = subtb_lambda.clamp(1e-4, 1.0);
    let mut weight_sum = 0.0;
    let mut weighted_terms: Vec<(f64, TapeNode)> = Vec::with_capacity((m * (m + 1)) / 2);

    for i in 0..m {
        for j in (i + 1)..=m {
            let w_ij = lam.powi((j - i) as i32);
            weight_sum += w_ij;

            let mut terms: Vec<(f64, TapeNode)> = Vec::with_capacity(2 * (j - i) + 2);
            terms.push((1.0, log_f[i]));
            for &lp in &log_pf[i..j] {
                terms.push((1.0, lp));
            }
            let const_bias = if j < m {
                terms.push((-1.0, log_f[j]));
                0.0
            } else {
                // Condition frontière terminale : -log F(H_T^top, E) = -log phi_sharp(H_T, E).
                -traj.log_phi_sharp
            };
            let pb_end = j.min(m - 1);
            for &lp in &log_pb[i..pb_end] {
                terms.push((-1.0, lp));
            }

            let delta = tape.linear_comb(const_bias, terms);
            let sq = tape.square(delta);
            weighted_terms.push((w_ij, sq));
        }
    }

    let inv_w = 1.0 / weight_sum.max(1e-12);
    let scaled_terms: Vec<(f64, TapeNode)> = weighted_terms
        .into_iter()
        .map(|(w, n)| (w * inv_w * loss_scale, n))
        .collect();
    let scaled_loss = tape.linear_comb(0.0, scaled_terms);
    let raw_loss = tape.scalar(scaled_loss) / loss_scale.max(1e-12);

    tape.backward(scaled_loss, backbone.params(), param_grads);
    raw_loss
}

/// Évalue uniquement la valeur scalaire de la perte SubTB (sans garder les gradients).
pub fn eval_subtb_loss(
    backbone: &RelationalBackbone,
    traj: &AbductiveTrajectory,
    subtb_lambda: f64,
) -> f64 {
    let mut scratch = vec![0.0; backbone.param_count()];
    subtb_loss_and_accumulate_grad(backbone, traj, subtb_lambda, 1.0, &mut scratch)
}

// ============================================================================
// 7. Échantillonnage de trajectoires (on-policy + Back-and-Forth off-policy)
// ============================================================================

/// Échantillonne une trajectoire complète H_0 -> ... -> H_T -> STOP sur un `TrainingSample`.
pub fn sample_on_policy_trajectory(
    backbone: &RelationalBackbone,
    obs: &ObservationGraph,
    graph_cfg: GraphConfig,
    ref_cfg: RefutationConfig,
    max_steps: usize,
    temperature: f64,
    epsilon_uniform: f64,
    rng_seed: u64,
) -> AbductiveTrajectory {
    let mut rng = SplitMix64::new(rng_seed);
    let mut state = HypothesisGraph::new(graph_cfg);
    let mut states = Vec::new();
    let mut residuals = Vec::new();
    let mut actions = Vec::new();

    let step_limit = max_steps.max(2);

    for step in 0..step_limit {
        let (_, residual) =
            project_and_residual(&state, obs, ref_cfg.projector_cfg, ref_cfg.residual_weights);
        let mask = state.forward_mask();
        let legal = mask.to_actions(&state.config().alphabet);

        let chosen = if step + 1 == step_limit && mask.allow_stop {
            ForwardAction::Stop
        } else {
            let lps = backbone.forward_log_probs(&state, obs, &residual, &mask);
            let idx = sample_from_log_probs(&lps, temperature, epsilon_uniform, &mut rng);
            legal[idx]
        };

        states.push(state.clone());
        residuals.push(residual);
        actions.push(chosen);

        if matches!(chosen, ForwardAction::Stop) {
            break;
        }
        state = state
            .apply_forward(&chosen)
            .expect("Action issue de ForwardMask toujours légale");

        // Si plus aucune action sauf STOP n'existe au prochain état, la boucle le choisira.
        if state.forward_mask().to_actions(&state.config().alphabet).is_empty() {
            break;
        }
    }

    // Assure que la trajectoire se termine par STOP.
    if actions.last() != Some(&ForwardAction::Stop) {
        let (_, residual) =
            project_and_residual(&state, obs, ref_cfg.projector_cfg, ref_cfg.residual_weights);
        states.push(state);
        residuals.push(residual);
        actions.push(ForwardAction::Stop);
    }

    let terminal_h = states.last().expect("Au moins H_1 présent");
    let report = evaluate_phi_sharp(terminal_h, obs, ref_cfg);

    AbductiveTrajectory {
        obs: obs.clone(),
        states,
        residuals,
        actions,
        log_phi_sharp: report.log_phi_sharp,
        terminal_residual_norm: report.residual.norm,
    }
}

/// Construit une trajectoire *off-policy* Back-and-Forth à partir d'une trajectoire existante :
/// recule de `k` pas via P_B, remonte canoniquement jusqu'à H_0 pour obtenir un préfixe valide,
/// puis déroule P_F vers une nouvelle branche.
pub fn sample_back_and_forth_trajectory(
    backbone: &RelationalBackbone,
    base: &AbductiveTrajectory,
    back_steps: usize,
    max_total_steps: usize,
    ref_cfg: RefutationConfig,
    temperature: f64,
    epsilon_uniform: f64,
    rng_seed: u64,
) -> Option<AbductiveTrajectory> {
    let mut rng = SplitMix64::new(rng_seed);
    let start_h = base.states.last()?.clone();
    let cfg = start_h.config().clone();

    // 1. Recule de `k` pas selon P_B pour atteindre l'état pivot `pivot`.
    let mut pivot = start_h;
    let steps_to_take = back_steps.min(pivot.potential().saturating_sub(1)).max(1);
    for _ in 0..steps_to_take {
        if pivot.is_initial() {
            break;
        }
        let (_, res) = project_and_residual(
            &pivot,
            &base.obs,
            ref_cfg.projector_cfg,
            ref_cfg.residual_weights,
        );
        let b_mask = pivot.backward_mask();
        let legal_b = b_mask.to_actions(pivot.edges(), &cfg.alphabet);
        let lps = backbone.backward_log_probs(&pivot, &base.obs, &res, &b_mask);
        let idx = sample_from_log_probs(&lps, temperature, epsilon_uniform, &mut rng);
        pivot = pivot.apply_backward(&legal_b[idx]).ok()?;
    }

    // 2. Continue jusqu'à H_0 pour obtenir la chaîne de clés canoniques H_0 -> ... -> pivot.
    let mut keys_rev = vec![pivot.canonical_key()];
    let mut cur = pivot;
    while !cur.is_initial() {
        let legal_b = cur.legal_backward_actions();
        let chosen_b = legal_b[(rng.next_u64() as usize) % legal_b.len()];
        cur = cur.apply_backward(&chosen_b).ok()?;
        keys_rev.push(cur.canonical_key());
    }
    keys_rev.reverse();

    // 3. Reconstruit un préfixe avant strictement valide H_0 -> ... -> pivot'.
    let mut state = HypothesisGraph::new(cfg);
    let mut states = Vec::new();
    let mut residuals = Vec::new();
    let mut actions = Vec::new();

    for target_key in keys_rev.iter().skip(1) {
        let (_, res) = project_and_residual(
            &state,
            &base.obs,
            ref_cfg.projector_cfg,
            ref_cfg.residual_weights,
        );
        let legal_f = state.legal_forward_actions();
        let fa = legal_f.into_iter().find(|a| {
            state
                .apply_forward(a)
                .is_ok_and(|next| &next.canonical_key() == target_key)
        })?;
        let next = state.apply_forward(&fa).ok()?;
        states.push(state);
        residuals.push(res);
        actions.push(fa);
        state = next;
    }

    // 4. Poursuit en avant avec P_F depuis pivot' jusqu'à STOP ou `max_total_steps`.
    let limit = max_total_steps.max(states.len() + 2);
    while states.len() < limit {
        let (_, res) = project_and_residual(
            &state,
            &base.obs,
            ref_cfg.projector_cfg,
            ref_cfg.residual_weights,
        );
        let mask = state.forward_mask();
        let legal = mask.to_actions(&state.config().alphabet);
        let chosen = if states.len() + 1 == limit && mask.allow_stop {
            ForwardAction::Stop
        } else {
            let lps = backbone.forward_log_probs(&state, &base.obs, &res, &mask);
            let idx = sample_from_log_probs(&lps, temperature, epsilon_uniform, &mut rng);
            legal[idx]
        };

        states.push(state.clone());
        residuals.push(res);
        actions.push(chosen);

        if matches!(chosen, ForwardAction::Stop) {
            break;
        }
        state = state.apply_forward(&chosen).ok()?;
    }

    let terminal_h = states.last()?;
    let report = evaluate_phi_sharp(terminal_h, &base.obs, ref_cfg);
    Some(AbductiveTrajectory {
        obs: base.obs.clone(),
        states,
        residuals,
        actions,
        log_phi_sharp: report.log_phi_sharp,
        terminal_residual_norm: report.residual.norm,
    })
}

// ============================================================================
// 8. Replay Buffer off-policy, Optimiseur Adam et Entraîneur SubTB
// ============================================================================

/// Buffer mixte conservant les meilleures trajectoires (élite phi_sharp) et les récentes.
#[derive(Clone, Debug)]
pub struct TrajectoryReplayBuffer {
    max_elite: usize,
    max_recent: usize,
    elite: Vec<AbductiveTrajectory>,
    recent: VecDeque<AbductiveTrajectory>,
}

impl TrajectoryReplayBuffer {
    pub fn new(max_elite: usize, max_recent: usize) -> Self {
        Self {
            max_elite: max_elite.max(1),
            max_recent: max_recent.max(1),
            elite: Vec::new(),
            recent: VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.elite.len() + self.recent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elite.is_empty() && self.recent.is_empty()
    }

    pub fn push(&mut self, traj: AbductiveTrajectory) {
        if self.recent.len() >= self.max_recent {
            self.recent.pop_front();
        }
        self.recent.push_back(traj.clone());

        if self.elite.len() < self.max_elite {
            self.elite.push(traj);
            self.sort_elite();
        } else if let Some(worst) = self.elite.last() {
            if traj.log_phi_sharp > worst.log_phi_sharp {
                let last = self.elite.len() - 1;
                self.elite[last] = traj;
                self.sort_elite();
            }
        }
    }

    fn sort_elite(&mut self) {
        self.elite
            .sort_by(|a, b| b.log_phi_sharp.total_cmp(&a.log_phi_sharp));
    }

    fn sample(&self, rng: &mut SplitMix64) -> Option<&AbductiveTrajectory> {
        if self.is_empty() {
            return None;
        }
        // 60 % élite, 40 % récent pour équilibrer exploitation et couverture de modes.
        let use_elite = !self.elite.is_empty() && (self.recent.is_empty() || rng.next_f64() < 0.6);
        if use_elite {
            let idx = (rng.next_u64() as usize) % self.elite.len();
            Some(&self.elite[idx])
        } else {
            let idx = (rng.next_u64() as usize) % self.recent.len();
            self.recent.get(idx)
        }
    }
}

/// Optimiseur AdamW léger avec clipping de norme L2 des gradients.
#[derive(Clone, Debug)]
pub struct AdamOptimizer {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub max_grad_norm: f64,
    step: u64,
    m: Vec<f64>,
    v: Vec<f64>,
}

impl AdamOptimizer {
    pub fn new(param_count: usize, lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-4,
            max_grad_norm: 5.0,
            step: 0,
            m: vec![0.0; param_count],
            v: vec![0.0; param_count],
        }
    }

    /// Applique une mise à jour AdamW sur `params` et retourne la norme L2 brute du gradient.
    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) -> f64 {
        debug_assert_eq!(params.len(), grads.len());
        debug_assert_eq!(params.len(), self.m.len());

        let norm_sq: f64 = grads.iter().map(|&g| g * g).sum();
        let grad_norm = norm_sq.sqrt();
        let clip_scale = if grad_norm > self.max_grad_norm && grad_norm > 0.0 {
            self.max_grad_norm / grad_norm
        } else {
            1.0
        };

        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);

        for i in 0..params.len() {
            let g = grads[i] * clip_scale;
            if !g.is_finite() {
                continue;
            }
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * params[i]);
        }

        grad_norm
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SubTbTrainConfig {
    pub learning_rate: f64,
    pub subtb_lambda: f64,
    pub max_trajectory_steps: usize,
    pub sample_temperature: f64,
    pub epsilon_uniform: f64,
    pub replay_samples_per_step: usize,
    pub use_back_and_forth: bool,
    pub back_and_forth_steps: usize,
    /// Probabilité par pas d'injecter une trajectoire off-policy via `ResidualGuidedHeuristicPolicy`.
    pub guided_explore_prob: f64,
    /// Température softmax utilisée pour la politique heuristique d'amorçage.
    pub guided_temperature: f64,
    pub refutation_cfg: RefutationConfig,
}

impl Default for SubTbTrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-3,
            subtb_lambda: 0.9,
            max_trajectory_steps: 36,
            sample_temperature: 1.1,
            epsilon_uniform: 0.1,
            replay_samples_per_step: 2,
            use_back_and_forth: true,
            back_and_forth_steps: 4,
            guided_explore_prob: 0.6,
            guided_temperature: 0.45,
            refutation_cfg: RefutationConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainStepMetrics {
    pub mean_subtb_loss: f64,
    pub mean_log_phi_sharp: f64,
    pub mean_terminal_residual: f64,
    pub grad_norm: f64,
    pub trajectories_used: usize,
}

/// Orchestrateur d'entraînement SubTB (on-policy + replay off-policy + Back-and-Forth).
pub struct SubTbTrainer {
    cfg: SubTbTrainConfig,
    optimizer: AdamOptimizer,
    replay: TrajectoryReplayBuffer,
    rng: SplitMix64,
}

impl SubTbTrainer {
    pub fn new(
        backbone: &RelationalBackbone,
        cfg: SubTbTrainConfig,
        replay_elite: usize,
        replay_recent: usize,
        seed: u64,
    ) -> Self {
        Self {
            cfg,
            optimizer: AdamOptimizer::new(backbone.param_count(), cfg.learning_rate),
            replay: TrajectoryReplayBuffer::new(replay_elite, replay_recent),
            rng: SplitMix64::new(seed),
        }
    }

    pub fn replay_len(&self) -> usize {
        self.replay.len()
    }

    pub fn train_step(
        &mut self,
        backbone: &mut RelationalBackbone,
        batch: &[TrainingSample],
    ) -> TrainStepMetrics {
        assert!(!batch.is_empty(), "Le batch d'entraînement ne peut pas être vide");
        let mut trajectories =
            Vec::with_capacity(batch.len() + self.cfg.replay_samples_per_step + 2);

        // 1. Trajectoires on-policy sur les effets E du batch PTGRS.
        for sample in batch {
            let step_budget = effective_traj_steps(self.cfg.max_trajectory_steps, &sample.effect);
            let s_seed = self.rng.next_u64();
            let traj = sample_on_policy_trajectory(
                backbone,
                &sample.effect,
                sample.recommended_graph_cfg.clone(),
                self.cfg.refutation_cfg,
                step_budget,
                self.cfg.sample_temperature,
                self.cfg.epsilon_uniform,
                s_seed,
            );
            self.replay.push(traj.clone());
            trajectories.push(traj);
        }

        // 1b. Amorçage off-policy guidé par résidu (ResidualGuidedHeuristicPolicy, zéro fuite H_true).
        if self.cfg.guided_explore_prob > 0.0
            && self.rng.next_f64() < self.cfg.guided_explore_prob
        {
            let pick = (self.rng.next_u64() as usize) % batch.len();
            let sample = &batch[pick];
            let step_budget = effective_traj_steps(self.cfg.max_trajectory_steps, &sample.effect);
            let g_seed = self.rng.next_u64();
            let mut heuristic = ResidualGuidedHeuristicPolicy;
            let g_traj = sample_policy_trajectory(
                &mut heuristic,
                &sample.effect,
                sample.recommended_graph_cfg.clone(),
                self.cfg.refutation_cfg,
                step_budget,
                self.cfg.guided_temperature,
                0.05,
                g_seed,
            );
            self.replay.push(g_traj.clone());
            trajectories.push(g_traj);
        }

        // 2. Trajectoire off-policy Back-and-Forth depuis un mode du replay buffer.
        if self.cfg.use_back_and_forth {
            if let Some(base) = self.replay.sample(&mut self.rng).cloned() {
                let bf_budget = effective_traj_steps(self.cfg.max_trajectory_steps, &base.obs);
                let bf_seed = self.rng.next_u64();
                if let Some(bf_traj) = sample_back_and_forth_trajectory(
                    backbone,
                    &base,
                    self.cfg.back_and_forth_steps,
                    bf_budget,
                    self.cfg.refutation_cfg,
                    self.cfg.sample_temperature,
                    self.cfg.epsilon_uniform,
                    bf_seed,
                ) {
                    self.replay.push(bf_traj.clone());
                    trajectories.push(bf_traj);
                }
            }
        }

        // 3. Rejeu off-policy depuis le buffer.
        for _ in 0..self.cfg.replay_samples_per_step {
            if let Some(t) = self.replay.sample(&mut self.rng).cloned() {
                trajectories.push(t);
            }
        }

        self.update_on_trajectories(backbone, &trajectories)
    }

    /// Met à jour les poids du backbone sur une liste explicite de trajectoires.
    pub fn update_on_trajectories(
        &mut self,
        backbone: &mut RelationalBackbone,
        trajectories: &[AbductiveTrajectory],
    ) -> TrainStepMetrics {
        assert!(!trajectories.is_empty());
        let n = trajectories.len() as f64;
        let inv_n = 1.0 / n;
        let mut grads = vec![0.0; backbone.param_count()];

        let mut total_loss = 0.0;
        let mut total_log_phi = 0.0;
        let mut total_res = 0.0;

        for traj in trajectories {
            let loss = subtb_loss_and_accumulate_grad(
                backbone,
                traj,
                self.cfg.subtb_lambda,
                inv_n,
                &mut grads,
            );
            total_loss += loss;
            total_log_phi += traj.log_phi_sharp;
            total_res += traj.terminal_residual_norm;
        }

        let grad_norm = self.optimizer.step(backbone.params_mut(), &grads);
        TrainStepMetrics {
            mean_subtb_loss: total_loss * inv_n,
            mean_log_phi_sharp: total_log_phi * inv_n,
            mean_terminal_residual: total_res * inv_n,
            grad_norm,
            trajectories_used: trajectories.len(),
        }
    }
}

// ============================================================================
// Tests unitaires complets (vérification numérique des gradients incluse)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_controller::{AskConfig, AskController};
    use crate::ptgrs_synth::{PtgrsSynthesizer, SynthConfig};

    fn build_rich_trajectory(
        obs: &ObservationGraph,
        cfg: GraphConfig,
        ref_cfg: RefutationConfig,
    ) -> AbductiveTrajectory {
        // Exerce toutes les familles : AddNode(Rule/Conc), Connect(InSeed/InLhs/InRhs/SameAs), Abstract, Stop.
        let actions = vec![
            ForwardAction::AddNode(AddNodeKind::Conc(b'a')),       // 1
            ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
            ForwardAction::AddNode(AddNodeKind::Rule),             // 2
            ForwardAction::AddNode(AddNodeKind::Conc(b'a')),       // 3
            ForwardAction::AddNode(AddNodeKind::Conc(b'b')),       // 4
            ForwardAction::Connect(Edge::new(2, 3, EdgeKind::InLhs)),
            ForwardAction::Connect(Edge::new(2, 4, EdgeKind::InRhs)),
            ForwardAction::Abstract(3),
            ForwardAction::Abstract(4),
            ForwardAction::Connect(Edge::new(3, 4, EdgeKind::SameAs)),
            ForwardAction::Stop,
        ];

        let mut state = HypothesisGraph::new(cfg);
        let mut states = Vec::with_capacity(actions.len());
        let mut residuals = Vec::with_capacity(actions.len());

        for &a in &actions {
            let (_, res) =
                project_and_residual(&state, obs, ref_cfg.projector_cfg, ref_cfg.residual_weights);
            states.push(state.clone());
            residuals.push(res);
            if a != ForwardAction::Stop {
                state = state.apply_forward(&a).expect("Action P_F valide");
            }
        }

        let report = evaluate_phi_sharp(states.last().unwrap(), obs, ref_cfg);
        AbductiveTrajectory {
            obs: obs.clone(),
            states,
            residuals,
            actions,
            log_phi_sharp: report.log_phi_sharp,
            terminal_residual_norm: report.residual.norm,
        }
    }

    #[test]
    fn test_factorized_policies_normalize_strictly_to_one() {
        let obs = ObservationGraph::from_bytes(b"aba");
        let cfg = GraphConfig::new(vec![b'a', b'b'], 2, 10, 2, 14).unwrap();
        let ref_cfg = RefutationConfig::default();
        let backbone = RelationalBackbone::new(
            BackboneConfig {
                hidden_dim: 8,
                num_layers: 1,
            },
            101,
        )
        .unwrap();

        let traj = build_rich_trajectory(&obs, cfg, ref_cfg);
        for (t, (state, res)) in traj.states.iter().zip(traj.residuals.iter()).enumerate() {
            // Vérification exacte sum(P_F) == 1.0
            let f_mask = state.forward_mask();
            let f_actions = f_mask.to_actions(&state.config().alphabet);
            let f_lps = backbone.forward_log_probs(state, &obs, res, &f_mask);
            assert_eq!(f_lps.len(), f_actions.len());
            let f_sum: f64 = f_lps.iter().map(|&lp| lp.exp()).sum();
            assert!(
                (f_sum - 1.0).abs() < 1e-11,
                "P_F non normalisé à t={t} : sum={f_sum}"
            );

            // Vérification exacte sum(P_B) == 1.0 pour tout état non initial
            if t > 0 {
                let b_mask = state.backward_mask();
                let b_actions = b_mask.to_actions(state.edges(), &state.config().alphabet);
                let b_lps = backbone.backward_log_probs(state, &obs, res, &b_mask);
                assert_eq!(b_lps.len(), b_actions.len());
                let b_sum: f64 = b_lps.iter().map(|&lp| lp.exp()).sum();
                assert!(
                    (b_sum - 1.0).abs() < 1e-11,
                    "P_B non normalisé à t={t} : sum={b_sum}"
                );
            }
        }
    }

    #[test]
    fn test_analytical_gradients_match_finite_differences() {
        let obs = ObservationGraph::from_bytes(b"ab");
        let cfg = GraphConfig::new(vec![b'a', b'b'], 1, 8, 2, 12).unwrap();
        let ref_cfg = RefutationConfig::default();
        let mut backbone = RelationalBackbone::new(
            BackboneConfig {
                hidden_dim: 4,
                num_layers: 1,
            },
            2026,
        )
        .unwrap();

        let traj = build_rich_trajectory(&obs, cfg, ref_cfg);
        let mut analytical = vec![0.0; backbone.param_count()];
        let loss = subtb_loss_and_accumulate_grad(&backbone, &traj, 0.85, 1.0, &mut analytical);
        assert!(loss.is_finite() && loss >= 0.0);

        // Sélectionne des indices répartis sur tous les blocs de paramètres (y compris ceux à fort gradient).
        let mut indexed: Vec<(usize, f64)> = analytical
            .iter()
            .enumerate()
            .map(|(i, &g)| (i, g.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut check_indices: Vec<usize> = indexed.iter().take(40).map(|&(i, _)| i).collect();
        let stride = (backbone.param_count() / 30).max(1);
        for i in (0..backbone.param_count()).step_by(stride) {
            if !check_indices.contains(&i) {
                check_indices.push(i);
            }
        }

        let eps = 1e-6;
        let mut non_zero_checked = 0usize;
        for idx in check_indices {
            let orig = backbone.params()[idx];

            backbone.params_mut()[idx] = orig + eps;
            let lp = eval_subtb_loss(&backbone, &traj, 0.85);

            backbone.params_mut()[idx] = orig - eps;
            let lm = eval_subtb_loss(&backbone, &traj, 0.85);

            backbone.params_mut()[idx] = orig;

            let numeric = (lp - lm) / (2.0 * eps);
            let anal = analytical[idx];
            if numeric.abs() > 1e-4 {
                non_zero_checked += 1;
            }
            let diff = (numeric - anal).abs();
            let scale = 1.0_f64.max(numeric.abs()).max(anal.abs());
            assert!(
                diff / scale < 1e-5,
                "Gradient discordant à param[{idx}] : analytique={anal}, numérique={numeric}, diff={diff}"
            );
        }

        assert!(
            non_zero_checked >= 25,
            "Trop peu de paramètres actifs vérifiés : {non_zero_checked}"
        );
    }

    #[test]
    fn test_subtb_training_reduces_loss_on_ptgrs_trajectories() {
        let synth_cfg = SynthConfig {
            difficulty: 0,
            min_effect_nodes: 3,
            max_effect_nodes: 6,
            ..SynthConfig::default()
        };
        let mut synth = PtgrsSynthesizer::new(synth_cfg, 77).unwrap();
        let sample = synth
            .next_training_sample()
            .expect("PTGRS doit produire un échantillon d'entraînement");

        let mut backbone = RelationalBackbone::new(
            BackboneConfig {
                hidden_dim: 8,
                num_layers: 1,
            },
            31415,
        )
        .unwrap();

        let train_cfg = SubTbTrainConfig {
            learning_rate: 1.5e-2,
            subtb_lambda: 0.9,
            max_trajectory_steps: 8,
            use_back_and_forth: false,
            replay_samples_per_step: 0,
            ..SubTbTrainConfig::default()
        };
        let mut trainer = SubTbTrainer::new(&backbone, train_cfg, 8, 8, 999);

        // Fixe deux trajectoires issues de sample pour mesurer la convergence SubTB exacte.
        let t1 = sample_on_policy_trajectory(
            &backbone,
            &sample.effect,
            sample.recommended_graph_cfg.clone(),
            train_cfg.refutation_cfg,
            8,
            1.0,
            0.2,
            1,
        );
        let t2 = sample_on_policy_trajectory(
            &backbone,
            &sample.effect,
            sample.recommended_graph_cfg.clone(),
            train_cfg.refutation_cfg,
            8,
            1.0,
            0.2,
            2,
        );
        let fixed_batch = vec![t1, t2];

        let initial_loss = 0.5
            * (eval_subtb_loss(&backbone, &fixed_batch[0], train_cfg.subtb_lambda)
                + eval_subtb_loss(&backbone, &fixed_batch[1], train_cfg.subtb_lambda));

        let mut final_loss = initial_loss;
        for _ in 0..18 {
            let m = trainer.update_on_trajectories(&mut backbone, &fixed_batch);
            assert!(m.mean_subtb_loss.is_finite());
            final_loss = m.mean_subtb_loss;
        }

        assert!(
            final_loss < initial_loss * 0.5,
            "La perte SubTB doit chuter nettement : init={initial_loss}, fin={final_loss}"
        );

        // Vérifie aussi un pas complet end-to-end avec replay + Back-and-Forth.
        let mut full_trainer = SubTbTrainer::new(
            &backbone,
            SubTbTrainConfig {
                max_trajectory_steps: 6,
                replay_samples_per_step: 1,
                use_back_and_forth: true,
                back_and_forth_steps: 2,
                ..SubTbTrainConfig::default()
            },
            8,
            8,
            1234,
        );
        let metrics = full_trainer.train_step(&mut backbone, &[sample]);
        assert!(metrics.mean_subtb_loss.is_finite());
        assert!(metrics.trajectories_used >= 2);
        assert!(full_trainer.replay_len() >= 2);
    }

    #[test]
    fn test_backbone_plugs_directly_into_ask_controller_and_generalizes_ood_bytes() {
        // Entraîné/initialisé sur un alphabet {a, b}, exécuté zéro-shot sur des octets OOD {0xE1, 0xF2}.
        let mut backbone = RelationalBackbone::new(
            BackboneConfig {
                hidden_dim: 8,
                num_layers: 2,
            },
            555,
        )
        .unwrap();

        let ood_obs = ObservationGraph::from_bytes(&[0xE1, 0xF2, 0xE1]);
        let ood_cfg = GraphConfig::new(ood_obs.alphabet(), 1, 8, 2, 12).unwrap();

        let controller = AskController::new(AskConfig {
            budget_steps: 28,
            max_forward_run: 10,
            ..AskConfig::default()
        });

        let result = controller.ask(&ood_obs, ood_cfg, &mut backbone, 42);
        assert!(result.stats.steps_used <= 28);
        assert!(result.stats.forward_steps > 0);
        assert!(!result.modes.is_empty());
        assert!(result.modes[0].report.log_phi_sharp.is_finite());
    }
}