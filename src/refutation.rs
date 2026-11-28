//! Phisharp V1 — Banc de réfutation et calcul de log phi_sharp(H, E).
//!
//! À déclarer dans `src/lib.rs` :
//!   pub mod hypothesis_graph;
//!   pub mod projector;
//!   pub mod refutation;
//!
//! Puis exécuter :
//!   cargo test

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::hypothesis_graph::{
    AddNodeKind, Block, Edge, EdgeKind, ForwardAction, HypothesisGraph, NodeId, NodeKind, RelId,
};
use crate::projector::{
    ObsEdge, ObservationGraph, ProjectionOutcome, ProjectorConfig, ResidualGraph, ResidualWeights,
    TerminationReason, compute_residual, project_and_residual,
};

/// Poids de la longueur de description structurelle |H| (parcimonie anti-trivialité).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParsimonyWeights {
    pub conc_node: f64,
    /// Légèrement inférieur à `conc_node` pour favoriser l'abstraction en variables.
    pub var_node: f64,
    pub rule_hub: f64,
    pub membership_edge: f64,
    pub struct_edge: f64,
    pub same_as_edge: f64,
    /// Malus sur tout nœud isolé (degré total nul, hors SEED).
    pub orphan_node: f64,
    /// Malus sur chaque règle incomplète ou non exécutable.
    pub skipped_rule: f64,
}

impl Default for ParsimonyWeights {
    fn default() -> Self {
        Self {
            conc_node: 1.0,
            var_node: 0.7,
            rule_hub: 0.8,
            membership_edge: 0.25,
            struct_edge: 0.4,
            same_as_edge: 0.2,
            orphan_node: 1.5,
            skipped_rule: 1.5,
        }
    }
}

/// Configuration complète du banc de réfutation phi_sharp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefutationConfig {
    /// Poids alpha de la fidélité résiduelle (-alpha * ||R(H, E)||).
    pub alpha: f64,
    /// Poids beta de la stabilité sous réfutation active (+beta * Stab_ref(H, E)).
    pub beta: f64,
    /// Poids lambda de la parcimonie structurelle (-lambda * |H|).
    pub lambda: f64,
    /// Part du test de masquage contrefactuel dans Stab_ref.
    pub mask_weight: f64,
    /// Part du test d'ablation causale dans Stab_ref.
    pub ablation_weight: f64,
    /// Malus appliqué dans Stab_ref par règle parasite ou malformée.
    pub parasitic_rule_penalty: f64,
    /// Malus appliqué dans Stab_ref si Pi(H) n'atteint pas une forme normale.
    pub non_normal_penalty: f64,
    /// Nombre maximal de fenêtres masquées sur G_E (borne le coût d'entraînement).
    pub max_masks: usize,
    /// Rayon (en sauts) de chaque sous-région masquée (0 = un nœud + ses arêtes incidentes).
    pub mask_radius: usize,
    /// Gain minimal de résidu Delta R(r) pour valider la nécessité causale d'une règle.
    pub ablation_epsilon: f64,
    pub parsimony: ParsimonyWeights,
    pub projector_cfg: ProjectorConfig,
    pub residual_weights: ResidualWeights,
}

impl Default for RefutationConfig {
    fn default() -> Self {
        Self {
            alpha: 2.5,
            beta: 3.0,
            lambda: 0.35,
            mask_weight: 0.55,
            ablation_weight: 0.45,
            parasitic_rule_penalty: 1.2,
            non_normal_penalty: 1.5,
            max_masks: 8,
            mask_radius: 0,
            ablation_epsilon: 1e-6,
            parsimony: ParsimonyWeights::default(),
            projector_cfg: ProjectorConfig::default(),
            residual_weights: ResidualWeights {
                conflict_node: 2.2,
                ..ResidualWeights::default()
            },
        }
    }
}

/// Statut causal d'un hub `RULE` après test d'ablation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleStatus {
    /// Supprimer cette règle dégrade strictement le résidu (Delta R > epsilon).
    Necessary,
    /// Règle exécutable mais inutile ou nuisible (Delta R <= epsilon) : épicycle.
    Parasitic,
    /// Règle incomplète (LHS vide ou VAR non lié en RHS), ignorée par Pi.
    MalformedOrSkipped,
}

/// Diagnostic d'ablation pour une règle individuelle de H.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuleAblationReport {
    pub rule_node: NodeId,
    pub status: RuleStatus,
    /// ||R(H \ r, E)|| - ||R(H, E)|| (0.0 si malformée).
    pub delta_residual: f64,
    /// Contribution normalisée dans [0, 1] : delta / (delta + 1).
    pub necessity_score: f64,
}

/// Rapport complet produit par le banc de réfutation (pour SubTB et mémoire `nogoods`).
#[derive(Clone, Debug, PartialEq)]
pub struct RefutationReport {
    pub outcome: ProjectionOutcome,
    pub residual: ResidualGraph,
    /// Score moyen de reconstruction aveugle sur les fenêtres masquées, dans [0, 1].
    pub mask_score: f64,
    pub masks_tested: usize,
    /// Score moyen de nécessité causale des règles, dans [0, 1].
    pub ablation_score: f64,
    pub rule_reports: Vec<RuleAblationReport>,
    /// Longueur de description structurelle |H|.
    pub structural_complexity: f64,
    /// Stabilité composite Stab_ref(H, E).
    pub stab_ref: f64,
    /// Signal terminal direct pour Sub-Trajectory Balance : log F(H_T, E) = log phi_sharp.
    pub log_phi_sharp: f64,
    /// Score positif strict exp(clamp(log_phi_sharp)).
    pub phi_sharp: f64,
}

impl RefutationReport {
    /// Retourne les identifiants des hubs `RULE` réfutés (parasites ou malformés)
    /// pour alimenter le registre des `nogoods` en mode inférence.
    pub fn nogood_rule_nodes(&self) -> Vec<NodeId> {
        self.rule_reports
            .iter()
            .filter(|r| r.status != RuleStatus::Necessary)
            .map(|r| r.rule_node)
            .collect()
    }

    pub fn necessary_rule_count(&self) -> usize {
        self.rule_reports
            .iter()
            .filter(|r| r.status == RuleStatus::Necessary)
            .count()
    }

    pub fn parasitic_rule_count(&self) -> usize {
        self.rule_reports
            .iter()
            .filter(|r| r.status != RuleStatus::Necessary)
            .count()
    }
}

/// Évalue intégralement une hypothèse H face à l'observation G_E.
pub fn evaluate_phi_sharp(
    h: &HypothesisGraph,
    obs: &ObservationGraph,
    cfg: RefutationConfig,
) -> RefutationReport {
    // 1. Projection principale et résidu de base ||R(H, E)||.
    let (outcome, residual) =
        project_and_residual(h, obs, cfg.projector_cfg, cfg.residual_weights);

    // 2. Test de masquage contrefactuel aveugle.
    let (mask_score, masks_tested) = evaluate_masked_reconstruction(obs, &outcome.predicted, cfg);

    // 3. Test d'ablation causale règle par règle.
    let (ablation_score, rule_reports) = evaluate_rule_ablations(h, obs, residual.norm, cfg);

    // 4. Parcimonie structurelle |H|.
    let structural_complexity =
        compute_structural_complexity(h, outcome.skipped_rules, cfg.parsimony);

    // 5. Assemblage de Stab_ref et log phi_sharp(H, E).
    let parasitic_count = rule_reports
        .iter()
        .filter(|r| r.status != RuleStatus::Necessary)
        .count();
    let term_penalty = if outcome.termination == TerminationReason::NormalForm {
        0.0
    } else {
        cfg.non_normal_penalty
    };

    let stab_ref = cfg.mask_weight * mask_score + cfg.ablation_weight * ablation_score
        - (parasitic_count as f64) * cfg.parasitic_rule_penalty
        - term_penalty;

    let log_phi_sharp =
        -cfg.alpha * residual.norm + cfg.beta * stab_ref - cfg.lambda * structural_complexity;
    // Clamp exponentiel pour éviter tout underflow à 0.0 ou overflow à +inf.
    let phi_sharp = log_phi_sharp.clamp(-500.0, 500.0).exp();

    RefutationReport {
        outcome,
        residual,
        mask_score,
        masks_tested,
        ablation_score,
        rule_reports,
        structural_complexity,
        stab_ref,
        log_phi_sharp,
        phi_sharp,
    }
}

// ============================================================================
// 1. Parcimonie structurelle |H|
// ============================================================================

pub fn compute_structural_complexity(
    h: &HypothesisGraph,
    skipped_rules: usize,
    w: ParsimonyWeights,
) -> f64 {
    let mut complexity = 0.0;

    // On ignore le nœud 0 (SEED fixe commun à toutes les hypothèses).
    for (v, &kind) in h.nodes().iter().enumerate().skip(1) {
        complexity += match kind {
            NodeKind::Conc(_) => w.conc_node,
            NodeKind::Var => w.var_node,
            NodeKind::Rule => w.rule_hub,
            NodeKind::Seed => 0.0,
        };
        if h.total_degree(v) == 0 {
            complexity += w.orphan_node;
        }
    }

    for &edge in h.edges() {
        complexity += match edge.kind {
            EdgeKind::InSeed | EdgeKind::InLhs | EdgeKind::InRhs => w.membership_edge,
            EdgeKind::Struct(_) => w.struct_edge,
            EdgeKind::SameAs => w.same_as_edge,
        };
    }

    complexity += (skipped_rules as f64) * w.skipped_rule;
    complexity
}

// ============================================================================
// 2. Test de masquage contrefactuel aveugle
// ============================================================================

fn evaluate_masked_reconstruction(
    obs: &ObservationGraph,
    pred: &ObservationGraph,
    cfg: RefutationConfig,
) -> (f64, usize) {
    let n_obs = obs.node_count();
    if n_obs == 0 {
        let score = if pred.node_count() == 0 { 1.0 } else { 0.0 };
        return (score, 0);
    }
    if n_obs == 1 || cfg.max_masks == 0 {
        let res = compute_residual(obs, pred, cfg.residual_weights);
        let score = if res.is_zero() { 1.0 } else { 0.0 };
        return (score, 0);
    }
    if pred.node_count() == 0 {
        return (0.0, n_obs.min(cfg.max_masks));
    }

    // Adjacences de G_E (pour extraire les voisinages masqués et les arêtes de frontière).
    let mut obs_out: Vec<Vec<(NodeId, RelId)>> = vec![Vec::new(); n_obs];
    let mut obs_in: Vec<Vec<(NodeId, RelId)>> = vec![Vec::new(); n_obs];
    for &e in &obs.edges {
        obs_out[e.src].push((e.dst, e.rel));
        obs_in[e.dst].push((e.src, e.rel));
    }

    // Sélection déterministe et uniformément répartie des centres de masques.
    let mask_count = n_obs.min(cfg.max_masks);
    let mut centers = Vec::with_capacity(mask_count);
    for k in 0..mask_count {
        centers.push((k * n_obs) / mask_count);
    }

    let mut total_score = 0.0;
    let mut tested = 0usize;

    for center in centers {
        let masked = collect_masked_region(center, n_obs, &obs_out, &obs_in, cfg.mask_radius);
        if masked.is_empty() || masked.len() >= n_obs {
            continue;
        }

        let score = score_single_mask(obs, pred, &masked, &obs_out, &obs_in, cfg.residual_weights);
        total_score += score;
        tested += 1;
    }

    if tested == 0 {
        (0.0, 0)
    } else {
        (total_score / (tested as f64), tested)
    }
}

/// Construit une sous-région masquée par BFS non-orienté autour de `center`
/// tout en garantissant qu'au moins un nœud de contexte reste visible.
fn collect_masked_region(
    center: NodeId,
    n_obs: usize,
    obs_out: &[Vec<(NodeId, RelId)>],
    obs_in: &[Vec<(NodeId, RelId)>],
    radius: usize,
) -> BTreeSet<NodeId> {
    let mut masked = BTreeSet::new();
    let mut queue = VecDeque::new();
    masked.insert(center);
    queue.push_back((center, 0usize));

    while let Some((u, dist)) = queue.pop_front() {
        if dist >= radius || masked.len() + 1 >= n_obs {
            continue;
        }
        let neighbors = obs_out[u]
            .iter()
            .chain(obs_in[u].iter())
            .map(|&(v, _)| v);
        for v in neighbors {
            if masked.len() + 1 >= n_obs {
                break;
            }
            if masked.insert(v) {
                queue.push_back((v, dist + 1));
            }
        }
    }

    masked
}

/// Aligne E_hat sur le seul contexte visible (G_E \ masked), prédit aveuglément
/// la zone masquée par topologie de frontière, puis mesure la fidélité démasquée.
fn score_single_mask(
    obs: &ObservationGraph,
    pred: &ObservationGraph,
    masked: &BTreeSet<NodeId>,
    obs_out: &[Vec<(NodeId, RelId)>],
    obs_in: &[Vec<(NodeId, RelId)>],
    weights: ResidualWeights,
) -> f64 {
    let n_obs = obs.node_count();
    let n_pred = pred.node_count();

    // 1. Construction du sous-graphe strictement visible G_E \ masked.
    let mut obs_to_vis: Vec<Option<NodeId>> = vec![None; n_obs];
    let mut vis_to_obs: Vec<NodeId> = Vec::with_capacity(n_obs - masked.len());
    let mut vis_nodes = Vec::with_capacity(n_obs - masked.len());

    for (u, &sym) in obs.nodes.iter().enumerate() {
        if !masked.contains(&u) {
            obs_to_vis[u] = Some(vis_nodes.len());
            vis_to_obs.push(u);
            vis_nodes.push(sym);
        }
    }

    let mut vis_edges = Vec::new();
    for &e in &obs.edges {
        if let (Some(vs), Some(vd)) = (obs_to_vis[e.src], obs_to_vis[e.dst]) {
            vis_edges.push(ObsEdge::new(vs, vd, e.rel));
        }
    }
    let visible_obs = ObservationGraph::new(vis_nodes, vis_edges);

    // 2. Alignement aveugle aux nœuds/arêtes masqués.
    let vis_residual = compute_residual(&visible_obs, pred, weights);
    let mut obs_to_pred: Vec<Option<NodeId>> = vec![None; n_obs];
    let mut pred_used = vec![false; n_pred];
    for aligned in &vis_residual.aligned_nodes {
        let orig_u = vis_to_obs[aligned.obs_node];
        obs_to_pred[orig_u] = Some(aligned.pred_node);
        pred_used[aligned.pred_node] = true;
    }

    // 3. Prédiction aveugle des nœuds masqués par propagation topologique depuis la frontière.
    let mut unpredicted: BTreeSet<NodeId> = masked.clone();
    loop {
        let mut best_choice: Option<(usize, NodeId, NodeId)> = None;

        for &w in &unpredicted {
            // Collecte les contraintes relationnelles depuis les voisins déjà ancrés.
            let mut in_anchors: Vec<(NodeId, RelId)> = Vec::new();
            for &(u, rel) in &obs_in[w] {
                if let Some(pu) = obs_to_pred[u] {
                    in_anchors.push((pu, rel));
                }
            }
            let mut out_anchors: Vec<(NodeId, RelId)> = Vec::new();
            for &(v, rel) in &obs_out[w] {
                if let Some(pv) = obs_to_pred[v] {
                    out_anchors.push((pv, rel));
                }
            }
            if in_anchors.is_empty() && out_anchors.is_empty() {
                continue;
            }

            // Note : on ne consulte JAMAIS `obs.nodes[w]` ici (prédiction strictement aveugle).
            for cand in 0..n_pred {
                if pred_used[cand] {
                    continue;
                }
                let mut matches = 0usize;
                for &(pu, rel) in &in_anchors {
                    if pred.edges.contains(&ObsEdge::new(pu, cand, rel)) {
                        matches += 1;
                    }
                }
                for &(pv, rel) in &out_anchors {
                    if pred.edges.contains(&ObsEdge::new(cand, pv, rel)) {
                        matches += 1;
                    }
                }
                if matches == 0 {
                    continue;
                }

                let is_better = match best_choice {
                    None => true,
                    Some((best_m, best_w, best_c)) => {
                        matches > best_m
                            || (matches == best_m && (w, cand) < (best_w, best_c))
                    }
                };
                if is_better {
                    best_choice = Some((matches, w, cand));
                }
            }
        }

        match best_choice {
            Some((_, w, cand)) => {
                obs_to_pred[w] = Some(cand);
                pred_used[cand] = true;
                unpredicted.remove(&w);
            }
            None => break,
        }
    }

    // 4. Démasquage et évaluation de la reconstruction exacte (nœuds + arêtes masqués).
    let mut total_items = 0usize;
    let mut correct_items = 0usize;

    for &w in masked {
        total_items += 1;
        if let Some(pw) = obs_to_pred[w] {
            if pred.nodes[pw] == obs.nodes[w] {
                correct_items += 1;
            }
        }
    }

    for &e in &obs.edges {
        if masked.contains(&e.src) || masked.contains(&e.dst) {
            total_items += 1;
            if let (Some(ps), Some(pd)) = (obs_to_pred[e.src], obs_to_pred[e.dst]) {
                if pred.edges.contains(&ObsEdge::new(ps, pd, e.rel)) {
                    correct_items += 1;
                }
            }
        }
    }

    if total_items == 0 {
        1.0
    } else {
        (correct_items as f64) / (total_items as f64)
    }
}

// ============================================================================
// 3. Test d'ablation causale règle par règle
// ============================================================================

fn evaluate_rule_ablations(
    h: &HypothesisGraph,
    obs: &ObservationGraph,
    base_residual_norm: f64,
    cfg: RefutationConfig,
) -> (f64, Vec<RuleAblationReport>) {
    let rule_nodes: Vec<NodeId> = h
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(id, &k)| (k == NodeKind::Rule).then_some(id))
        .collect();

    if rule_nodes.is_empty() {
        return (0.0, Vec::new());
    }

    let mut reports = Vec::with_capacity(rule_nodes.len());
    let mut necessity_sum = 0.0;

    for rule_id in rule_nodes {
        if !is_rule_executable(h, rule_id) {
            reports.push(RuleAblationReport {
                rule_node: rule_id,
                status: RuleStatus::MalformedOrSkipped,
                delta_residual: 0.0,
                necessity_score: 0.0,
            });
            continue;
        }

        let ablated = build_ablated_hypothesis(h, rule_id);
        let (_, ablated_res) =
            project_and_residual(&ablated, obs, cfg.projector_cfg, cfg.residual_weights);
        let delta = ablated_res.norm - base_residual_norm;

        if delta > cfg.ablation_epsilon {
            let score = delta / (delta + 1.0);
            necessity_sum += score;
            reports.push(RuleAblationReport {
                rule_node: rule_id,
                status: RuleStatus::Necessary,
                delta_residual: delta,
                necessity_score: score,
            });
        } else {
            reports.push(RuleAblationReport {
                rule_node: rule_id,
                status: RuleStatus::Parasitic,
                delta_residual: delta,
                necessity_score: 0.0,
            });
        }
    }

    let ablation_score = necessity_sum / (reports.len() as f64);
    (ablation_score, reports)
}

/// Vérifie si un hub `RULE` satisfait les critères d'exécutabilité de `CompiledPtgrs`
/// (LHS non vide et chaque VAR du RHS liée par `SameAs` à au moins une VAR du LHS).
fn is_rule_executable(h: &HypothesisGraph, rule_id: NodeId) -> bool {
    let mut lhs_vars = BTreeSet::new();
    let mut rhs_vars = Vec::new();
    let mut has_lhs = false;

    for v in 1..h.node_count() {
        match h.membership(v) {
            Some(Block::RuleLhs(r)) if r == rule_id => {
                has_lhs = true;
                if h.nodes()[v] == NodeKind::Var {
                    lhs_vars.insert(v);
                }
            }
            Some(Block::RuleRhs(r)) if r == rule_id => {
                if h.nodes()[v] == NodeKind::Var {
                    rhs_vars.push(v);
                }
            }
            _ => {}
        }
    }

    if !has_lhs {
        return false;
    }
    if rhs_vars.is_empty() {
        return true;
    }

    // Fermeture transitive des `SameAs` de la règle par BFS depuis `lhs_vars`.
    let mut same_adj: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
    for &e in h.edges() {
        if e.kind == EdgeKind::SameAs
            && h.owner_rule(e.src) == Some(rule_id)
            && h.owner_rule(e.dst) == Some(rule_id)
        {
            same_adj.entry(e.src).or_default().push(e.dst);
            same_adj.entry(e.dst).or_default().push(e.src);
        }
    }

    let mut bound = lhs_vars.clone();
    let mut queue: VecDeque<NodeId> = lhs_vars.into_iter().collect();
    while let Some(u) = queue.pop_front() {
        if let Some(neighbors) = same_adj.get(&u) {
            for &v in neighbors {
                if bound.insert(v) {
                    queue.push_back(v);
                }
            }
        }
    }

    rhs_vars.iter().all(|v| bound.contains(v))
}

/// Reconstruit H \ {drop_rule} en passant exclusivement par l'API publique de `HypothesisGraph`
/// (garantissant tous les invariants structurels et la validité DAG).
fn build_ablated_hypothesis(h: &HypothesisGraph, drop_rule: NodeId) -> HypothesisGraph {
    let cfg = h.config().clone();
    let fallback_sym = cfg.alphabet[0];
    let mut ablated = HypothesisGraph::new(cfg);

    let mut old_to_new: Vec<Option<NodeId>> = vec![None; h.node_count()];
    old_to_new[0] = Some(0); // SEED fixe

    // 1. Recréation des nœuds conservés.
    for v in 1..h.node_count() {
        if v == drop_rule || h.owner_rule(v) == Some(drop_rule) {
            continue;
        }
        let new_id = ablated.node_count();
        match h.nodes()[v] {
            NodeKind::Rule => {
                ablated = ablated
                    .apply_forward(&ForwardAction::AddNode(AddNodeKind::Rule))
                    .expect("Ajout RULE valide lors de l'ablation");
            }
            NodeKind::Conc(sym) => {
                ablated = ablated
                    .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(sym)))
                    .expect("Ajout CONC valide lors de l'ablation");
            }
            NodeKind::Var => {
                ablated = ablated
                    .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(fallback_sym)))
                    .expect("Ajout CONC temporaire valide")
                    .apply_forward(&ForwardAction::Abstract(new_id))
                    .expect("Abstraction VAR valide lors de l'ablation");
            }
            NodeKind::Seed => {}
        }
        old_to_new[v] = Some(new_id);
    }

    // 2. Arêtes d'appartenance d'abord (précondition stricte pour STRUCT et SAME_AS).
    for &e in h.edges() {
        if e.kind.is_membership() {
            if let (Some(ns), Some(nd)) = (old_to_new[e.src], old_to_new[e.dst]) {
                ablated = ablated
                    .apply_forward(&ForwardAction::Connect(Edge::new(ns, nd, e.kind)))
                    .expect("Arête d'appartenance valide lors de l'ablation");
            }
        }
    }

    // 3. Arêtes internes STRUCT et SAME_AS ensuite.
    for &e in h.edges() {
        if !e.kind.is_membership() {
            if let (Some(ns), Some(nd)) = (old_to_new[e.src], old_to_new[e.dst]) {
                ablated = ablated
                    .apply_forward(&ForwardAction::Connect(Edge::new(ns, nd, e.kind)))
                    .expect("Arête interne valide lors de l'ablation");
            }
        }
    }

    ablated
}

// ============================================================================
// Tests unitaires complets
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypothesis_graph::GraphConfig;

    fn apply_seq(mut g: HypothesisGraph, actions: &[ForwardAction]) -> HypothesisGraph {
        for a in actions {
            g = g.apply_forward(a).expect("Action P_F valide");
        }
        g
    }

    /// Construit la copie triviale H = E (mémorisation brute dans SEED, 0 règle).
    fn build_verbatim_copy(bytes: &[u8], cfg: GraphConfig) -> HypothesisGraph {
        let mut h = HypothesisGraph::new(cfg);
        for (i, &b) in bytes.iter().enumerate() {
            let nid = i + 1;
            h = h
                .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(b)))
                .unwrap()
                .apply_forward(&ForwardAction::Connect(Edge::new(
                    0,
                    nid,
                    EdgeKind::InSeed,
                )))
                .unwrap();
            if i > 0 {
                h = h
                    .apply_forward(&ForwardAction::Connect(Edge::new(
                        nid - 1,
                        nid,
                        EdgeKind::Struct(0),
                    )))
                    .unwrap();
            }
        }
        h
    }

    #[test]
    fn test_causal_rule_beats_verbatim_copy_and_epicycle() {
        // Effet structuré : "abcabcabc" (9 octets).
        let obs = ObservationGraph::from_bytes(b"abcabcabc");
        let cfg = GraphConfig::new(vec![b'a', b'b', b'c', b'x', b'z'], 1, 20, 4, 32).unwrap();
        let ref_cfg = RefutationConfig::default();

        // 1. Hypothèse triviale H_copy = E (9 nœuds dans SEED, 0 règle).
        let h_copy = build_verbatim_copy(b"abcabcabc", cfg.clone());
        let rep_copy = evaluate_phi_sharp(&h_copy, &obs, ref_cfg);
        assert!(rep_copy.residual.is_zero());
        assert_eq!(rep_copy.ablation_score, 0.0);
        assert_eq!(rep_copy.necessary_rule_count(), 0);

        // 2. Hypothèse causale compacte : SEED = x -> x -> x ; Règle : x => a -> b -> c.
        let h_causal = apply_seq(
            HypothesisGraph::new(cfg),
            &[
                ForwardAction::AddNode(AddNodeKind::Conc(b'x')), // 1
                ForwardAction::AddNode(AddNodeKind::Conc(b'x')), // 2
                ForwardAction::AddNode(AddNodeKind::Conc(b'x')), // 3
                ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(0, 2, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(0, 3, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(1, 2, EdgeKind::Struct(0))),
                ForwardAction::Connect(Edge::new(2, 3, EdgeKind::Struct(0))),
                ForwardAction::AddNode(AddNodeKind::Rule),       // 4
                ForwardAction::AddNode(AddNodeKind::Conc(b'x')), // 5 (LHS)
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 6 (RHS)
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 7 (RHS)
                ForwardAction::AddNode(AddNodeKind::Conc(b'c')), // 8 (RHS)
                ForwardAction::Connect(Edge::new(4, 5, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(4, 6, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(4, 7, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(4, 8, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(6, 7, EdgeKind::Struct(0))),
                ForwardAction::Connect(Edge::new(7, 8, EdgeKind::Struct(0))),
            ],
        );
        let rep_causal = evaluate_phi_sharp(&h_causal, &obs, ref_cfg);
        assert!(rep_causal.residual.is_zero());
        assert_eq!(rep_causal.necessary_rule_count(), 1);
        assert_eq!(rep_causal.parasitic_rule_count(), 0);
        assert!(
            rep_causal.ablation_score > 0.85,
            "Score d'ablation trop bas : {}",
            rep_causal.ablation_score
        );
        assert!(
            rep_causal.mask_score > 0.99,
            "Reconstruction masquée attendue à 1.0 : {}",
            rep_causal.mask_score
        );

        // 3. Ajout d'un épicycle parasite : Règle morte z => a (ne se déclenche jamais).
        let h_epicycle = apply_seq(
            h_causal.clone(),
            &[
                ForwardAction::AddNode(AddNodeKind::Rule),       // 9
                ForwardAction::AddNode(AddNodeKind::Conc(b'z')), // 10 (LHS)
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 11 (RHS)
                ForwardAction::Connect(Edge::new(9, 10, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(9, 11, EdgeKind::InRhs)),
            ],
        );
        let rep_epicycle = evaluate_phi_sharp(&h_epicycle, &obs, ref_cfg);
        assert!(rep_epicycle.residual.is_zero());
        assert_eq!(rep_epicycle.necessary_rule_count(), 1);
        assert_eq!(rep_epicycle.parasitic_rule_count(), 1);
        assert_eq!(rep_epicycle.nogood_rule_nodes(), vec![9]);

        // Hiérarchie stricte : Cause compacte > Copie triviale ET Cause compacte > Épicycle.
        assert!(
            rep_causal.log_phi_sharp > rep_copy.log_phi_sharp + 1.0,
            "La cause compacte ({}) doit nettement battre H = E ({})",
            rep_causal.log_phi_sharp,
            rep_copy.log_phi_sharp
        );
        assert!(
            rep_causal.log_phi_sharp > rep_epicycle.log_phi_sharp + 2.0,
            "La cause compacte ({}) doit nettement battre l'épicycle ({})",
            rep_causal.log_phi_sharp,
            rep_epicycle.log_phi_sharp
        );
    }

    #[test]
    fn test_ablation_detects_necessary_parasitic_and_malformed_rules() {
        // SEED = m -> a -> y ; Règle 1 (nécessaire, abstraite) : m -> X => X -> X -> "aay"
        // Règle 2 (parasite) : z => a ; Règle 3 (malformée) : LHS vide.
        let obs = ObservationGraph::from_bytes(b"aay");
        let cfg = GraphConfig::new(vec![b'm', b'a', b'y', b'z'], 1, 20, 4, 28).unwrap();

        let h = apply_seq(
            HypothesisGraph::new(cfg),
            &[
                ForwardAction::AddNode(AddNodeKind::Conc(b'm')), // 1
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 2
                ForwardAction::AddNode(AddNodeKind::Conc(b'y')), // 3
                ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(0, 2, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(0, 3, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(1, 2, EdgeKind::Struct(0))),
                ForwardAction::Connect(Edge::new(2, 3, EdgeKind::Struct(0))),
                // Règle 1 (id = 4) : (m -> X) => (X -> X)
                ForwardAction::AddNode(AddNodeKind::Rule),       // 4
                ForwardAction::AddNode(AddNodeKind::Conc(b'm')), // 5
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 6 -> VAR
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 7 -> VAR
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 8 -> VAR
                ForwardAction::Connect(Edge::new(4, 5, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(4, 6, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(5, 6, EdgeKind::Struct(0))),
                ForwardAction::Connect(Edge::new(4, 7, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(4, 8, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(7, 8, EdgeKind::Struct(0))),
                ForwardAction::Abstract(6),
                ForwardAction::Abstract(7),
                ForwardAction::Abstract(8),
                ForwardAction::Connect(Edge::new(6, 7, EdgeKind::SameAs)),
                ForwardAction::Connect(Edge::new(7, 8, EdgeKind::SameAs)),
                // Règle 2 (id = 9) : parasite (z => a)
                ForwardAction::AddNode(AddNodeKind::Rule),       // 9
                ForwardAction::AddNode(AddNodeKind::Conc(b'z')), // 10
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 11
                ForwardAction::Connect(Edge::new(9, 10, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(9, 11, EdgeKind::InRhs)),
                // Règle 3 (id = 12) : malformée (LHS vide)
                ForwardAction::AddNode(AddNodeKind::Rule),       // 12
            ],
        );

        let report = evaluate_phi_sharp(&h, &obs, RefutationConfig::default());
        assert!(report.residual.is_zero());
        assert_eq!(report.rule_reports.len(), 3);
        assert_eq!(report.rule_reports[0].rule_node, 4);
        assert_eq!(report.rule_reports[0].status, RuleStatus::Necessary);
        assert!(report.rule_reports[0].delta_residual > 0.0);

        assert_eq!(report.rule_reports[1].rule_node, 9);
        assert_eq!(report.rule_reports[1].status, RuleStatus::Parasitic);

        assert_eq!(report.rule_reports[2].rule_node, 12);
        assert_eq!(report.rule_reports[2].status, RuleStatus::MalformedOrSkipped);

        assert_eq!(report.nogood_rule_nodes(), vec![9, 12]);
    }

    #[test]
    fn test_masked_reconstruction_distinguishes_coherent_from_corrupted() {
        let obs = ObservationGraph::from_bytes(b"abcdef");
        let cfg = GraphConfig::new(vec![b'a', b'b', b'c', b'd', b'e', b'f', b'X'], 1, 16, 2, 24)
            .unwrap();
        let ref_cfg = RefutationConfig::default();

        let h_exact = build_verbatim_copy(b"abcdef", cfg.clone());
        let h_corrupt = build_verbatim_copy(b"abXdef", cfg.clone());
        let h_short = build_verbatim_copy(b"abde", cfg);

        let r_exact = evaluate_phi_sharp(&h_exact, &obs, ref_cfg);
        let r_corrupt = evaluate_phi_sharp(&h_corrupt, &obs, ref_cfg);
        let r_short = evaluate_phi_sharp(&h_short, &obs, ref_cfg);

        assert_eq!(r_exact.masks_tested, 6);
        assert!((r_exact.mask_score - 1.0).abs() < 1e-12);
        assert!(
            r_corrupt.mask_score < r_exact.mask_score,
            "Un symbole corrompu doit réduire mask_score ({} vs {})",
            r_corrupt.mask_score,
            r_exact.mask_score
        );
        assert!(
            r_short.mask_score < r_corrupt.mask_score,
            "Des nœuds manquants doivent réduire encore plus mask_score ({} vs {})",
            r_short.mask_score,
            r_corrupt.mask_score
        );
    }

    #[test]
    fn test_cycle_penalty_and_numerical_safety_for_gflownet() {
        let obs = ObservationGraph::from_bytes(b"a");
        let cfg = GraphConfig::new(vec![b'a', b'b'], 1, 10, 3, 12).unwrap();
        let ref_cfg = RefutationConfig::default();

        // Hypothèse cyclique : SEED = a ; a => b ; b => a.
        let h_cycle = apply_seq(
            HypothesisGraph::new(cfg.clone()),
            &[
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 1
                ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
                ForwardAction::AddNode(AddNodeKind::Rule),       // 2
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 3
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 4
                ForwardAction::Connect(Edge::new(2, 3, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(2, 4, EdgeKind::InRhs)),
                ForwardAction::AddNode(AddNodeKind::Rule),       // 5
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 6
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 7
                ForwardAction::Connect(Edge::new(5, 6, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(5, 7, EdgeKind::InRhs)),
            ],
        );

        let rep_cycle = evaluate_phi_sharp(&h_cycle, &obs, ref_cfg);
        assert_eq!(rep_cycle.outcome.termination, TerminationReason::CycleDetected);
        assert!(rep_cycle.log_phi_sharp.is_finite());
        assert!(rep_cycle.phi_sharp.is_finite() && rep_cycle.phi_sharp > 0.0);

        // Comparaison avec l'état initial vide H_0 : toujours fini et strictement positif.
        let rep_empty = evaluate_phi_sharp(&HypothesisGraph::new(cfg), &obs, ref_cfg);
        assert!(rep_empty.log_phi_sharp.is_finite());
        assert!(rep_empty.phi_sharp.is_finite() && rep_empty.phi_sharp > 0.0);

        // Comparaison avec une forme normale simple SEED = a (sans cycle).
        let h_simple = build_verbatim_copy(b"a", h_cycle.config().clone());
        let rep_simple = evaluate_phi_sharp(&h_simple, &obs, ref_cfg);
        assert!(
            rep_simple.log_phi_sharp > rep_cycle.log_phi_sharp,
            "La forme normale doit battre le cycle ({} vs {})",
            rep_simple.log_phi_sharp,
            rep_cycle.log_phi_sharp
        );
    }
}