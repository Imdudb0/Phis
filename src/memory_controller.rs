//! Phisharp V1 — Contrôleur d'inférence `ask`, Mémoire Épisodique et boucle Back-and-Forth.
//!
//! À déclarer dans `src/lib.rs` :
//!   pub mod hypothesis_graph;
//!   pub mod projector;
//!   pub mod refutation;
//!   pub mod memory_controller;
//!
//! Puis exécuter :
//!   cargo test

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use crate::hypothesis_graph::{
    AddNodeKind, BackwardAction, BackwardMask, Block, CanonicalKey, Edge, EdgeKind, ForwardAction,
    ForwardMask, GraphConfig, HypothesisGraph, NodeId, NodeKind,
};
use crate::projector::{
    ObservationGraph, ProjectionOutcome, ResidualGraph, project_and_residual,
};
use crate::refutation::{
    RefutationConfig, RefutationReport, RuleStatus, evaluate_phi_sharp,
};

// ============================================================================
// 1. PRNG déterministe minimal (SplitMix64 — zéro dépendance externe)
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
}

/// Échantillonne un indice proportionnellement à `exp(logit / temperature)`.
fn sample_softmax(logits: &[f64], temperature: f64, rng: &mut SplitMix64) -> usize {
    debug_assert!(!logits.is_empty());
    if logits.len() == 1 {
        return 0;
    }
    let temp = temperature.max(1e-6);
    let max_val = logits
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut weights = Vec::with_capacity(logits.len());
    let mut sum = 0.0;
    for &l in logits {
        let w = if l.is_finite() {
            ((l - max_val) / temp).clamp(-500.0, 0.0).exp()
        } else {
            0.0
        };
        weights.push(w);
        sum += w;
    }

    if sum <= 0.0 || !sum.is_finite() {
        return (rng.next_u64() as usize) % logits.len();
    }

    let mut target = rng.next_f64() * sum;
    for (idx, &w) in weights.iter().enumerate() {
        target -= w;
        if target <= 0.0 {
            return idx;
        }
    }
    logits.len() - 1
}

// ============================================================================
// 2. Cache canonique basé sur `CanonicalKey`
// ============================================================================

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub residual_hits: usize,
    pub residual_misses: usize,
    pub refutation_hits: usize,
    pub refutation_misses: usize,
}

#[derive(Clone, Debug)]
struct CachedResidualEntry {
    outcome: ProjectionOutcome,
    residual: ResidualGraph,
}

/// Cache épisodique indexé par forme canonique (invariant par isomorphisme).
#[derive(Clone, Debug, Default)]
pub struct CanonicalCache {
    residuals: BTreeMap<CanonicalKey, CachedResidualEntry>,
    refutations: BTreeMap<CanonicalKey, RefutationReport>,
    pub stats: CacheStats,
}

impl CanonicalCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Retourne `(ProjectionOutcome, ResidualGraph)` avec mise en cache par `CanonicalKey`.
    pub fn get_or_compute_residual(
        &mut self,
        h: &HypothesisGraph,
        key: &CanonicalKey,
        obs: &ObservationGraph,
        ref_cfg: RefutationConfig,
    ) -> (ProjectionOutcome, ResidualGraph) {
        if let Some(entry) = self.residuals.get(key) {
            self.stats.residual_hits += 1;
            return (entry.outcome.clone(), entry.residual.clone());
        }
        self.stats.residual_misses += 1;
        let (outcome, residual) =
            project_and_residual(h, obs, ref_cfg.projector_cfg, ref_cfg.residual_weights);
        self.residuals.insert(
            key.clone(),
            CachedResidualEntry {
                outcome: outcome.clone(),
                residual: residual.clone(),
            },
        );
        (outcome, residual)
    }

    /// Retourne `RefutationReport` avec mise en cache par `CanonicalKey` (et alimente le cache résidu).
    pub fn get_or_compute_refutation(
        &mut self,
        h: &HypothesisGraph,
        key: &CanonicalKey,
        obs: &ObservationGraph,
        ref_cfg: RefutationConfig,
    ) -> RefutationReport {
        if let Some(report) = self.refutations.get(key) {
            self.stats.refutation_hits += 1;
            return report.clone();
        }
        self.stats.refutation_misses += 1;
        let report = evaluate_phi_sharp(h, obs, ref_cfg);
        self.residuals.entry(key.clone()).or_insert_with(|| CachedResidualEntry {
            outcome: report.outcome.clone(),
            residual: report.residual.clone(),
        });
        self.refutations.insert(key.clone(), report.clone());
        report
    }

    pub fn cached_residual_count(&self) -> usize {
        self.residuals.len()
    }

    pub fn cached_refutation_count(&self) -> usize {
        self.refutations.len()
    }
}

// ============================================================================
// 3. Registre des `nogoods` et masquage dynamique de `ForwardMask`
// ============================================================================

/// Extrait la clé canonique isolée d'un hub `RULE` (invariante par renumérotation).
pub fn extract_rule_pattern_key(h: &HypothesisGraph, rule_id: NodeId) -> Option<CanonicalKey> {
    if rule_id >= h.node_count() || h.nodes()[rule_id] != NodeKind::Rule {
        return None;
    }

    let cfg = h.config().clone();
    let fallback_sym = cfg.alphabet[0];
    let mut mini = HypothesisGraph::new(cfg);

    // Le hub RULE est placé à l'indice 1 dans le mini-graphe.
    mini = mini
        .apply_forward(&ForwardAction::AddNode(AddNodeKind::Rule))
        .ok()?;
    let mut map: BTreeMap<NodeId, NodeId> = BTreeMap::new();
    map.insert(rule_id, 1);

    for v in 1..h.node_count() {
        if h.owner_rule(v) != Some(rule_id) {
            continue;
        }
        let new_id = mini.node_count();
        match h.nodes()[v] {
            NodeKind::Conc(sym) => {
                mini = mini
                    .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(sym)))
                    .ok()?;
            }
            NodeKind::Var => {
                mini = mini
                    .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(fallback_sym)))
                    .ok()?
                    .apply_forward(&ForwardAction::Abstract(new_id))
                    .ok()?;
            }
            NodeKind::Seed | NodeKind::Rule => {}
        }
        map.insert(v, new_id);
    }

    // Arêtes d'appartenance d'abord, puis arêtes internes STRUCT / SAME_AS.
    for &e in h.edges() {
        if e.kind.is_membership() {
            if let (Some(&ns), Some(&nd)) = (map.get(&e.src), map.get(&e.dst)) {
                mini = mini
                    .apply_forward(&ForwardAction::Connect(Edge::new(ns, nd, e.kind)))
                    .ok()?;
            }
        }
    }
    for &e in h.edges() {
        if !e.kind.is_membership() {
            if let (Some(&ns), Some(&nd)) = (map.get(&e.src), map.get(&e.dst)) {
                mini = mini
                    .apply_forward(&ForwardAction::Connect(Edge::new(ns, nd, e.kind)))
                    .ok()?;
            }
        }
    }

    Some(mini.canonical_key())
}

/// Résultat du filtrage d'un `ForwardMask` par le registre des nogoods.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskedForwardView {
    pub mask: ForwardMask,
    pub legal_actions: Vec<ForwardAction>,
    pub blocked_count: usize,
}

/// Registre épisodique des états impasses et motifs de règles parasites réfutées.
#[derive(Clone, Debug, Default)]
pub struct NogoodRegistry {
    /// États canoniques morts (impasses ou terminaux déjà explorés).
    dead_states: BTreeSet<CanonicalKey>,
    /// Motifs canoniques de règles exécutables réfutées comme parasites (`RuleStatus::Parasitic`).
    parasitic_rule_patterns: BTreeSet<CanonicalKey>,
    /// Interdit de s'arrêter deux fois exactement sur le même état terminal `STOP`.
    evaluated_terminals: BTreeSet<CanonicalKey>,
}

impl NogoodRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_dead_state(&mut self, key: CanonicalKey) {
        self.dead_states.insert(key);
    }

    pub fn mark_evaluated_terminal(&mut self, key: CanonicalKey) {
        self.evaluated_terminals.insert(key);
    }

    /// Enregistre uniquement les règles *exécutables et parasites* (jamais les règles incomplètes).
    pub fn record_from_refutation(
        &mut self,
        h: &HypothesisGraph,
        report: &RefutationReport,
    ) -> usize {
        let mut added = 0usize;
        for rr in &report.rule_reports {
            if rr.status == RuleStatus::Parasitic {
                if let Some(pattern_key) = extract_rule_pattern_key(h, rr.rule_node) {
                    if self.parasitic_rule_patterns.insert(pattern_key) {
                        added += 1;
                    }
                }
            }
        }
        added
    }

    pub fn dead_state_count(&self) -> usize {
        self.dead_states.len()
    }

    pub fn parasitic_pattern_count(&self) -> usize {
        self.parasitic_rule_patterns.len()
    }

    /// Vérifie si `candidate` contient au moins une règle correspondant à un motif parasite connu.
    fn contains_parasitic_pattern(&self, candidate: &HypothesisGraph) -> bool {
        if self.parasitic_rule_patterns.is_empty() {
            return false;
        }
        for (v, &kind) in candidate.nodes().iter().enumerate() {
            if kind == NodeKind::Rule {
                if let Some(pat) = extract_rule_pattern_key(candidate, v) {
                    if self.parasitic_rule_patterns.contains(&pat) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn is_action_blocked(&self, state: &HypothesisGraph, action: &ForwardAction) -> bool {
        let Ok(next) = state.apply_forward(action) else {
            return true;
        };
        let next_key = next.canonical_key();

        if self.dead_states.contains(&next_key) {
            return true;
        }
        if matches!(action, ForwardAction::Stop) && self.evaluated_terminals.contains(&next_key) {
            return true;
        }
        // Si l'action clôt ou conserve un motif de règle parasite déjà réfuté, on la bloque.
        if !matches!(action, ForwardAction::AddNode(_)) && self.contains_parasitic_pattern(&next) {
            return true;
        }
        false
    }

    /// Filtre `raw_mask` en désactivant toute action menant à un nogood, et met à jour `family_mask`.
    pub fn apply_mask(
        &self,
        state: &HypothesisGraph,
        raw_mask: &ForwardMask,
    ) -> MaskedForwardView {
        let alphabet = &state.config().alphabet;
        let mut mask = raw_mask.clone();
        let mut blocked_count = 0usize;

        if mask.allow_add_rule
            && self.is_action_blocked(state, &ForwardAction::AddNode(AddNodeKind::Rule))
        {
            mask.allow_add_rule = false;
            blocked_count += 1;
        }

        for (i, allowed) in mask.allow_add_conc.iter_mut().enumerate() {
            if *allowed
                && self.is_action_blocked(state, &ForwardAction::AddNode(AddNodeKind::Conc(alphabet[i])))
            {
                *allowed = false;
                blocked_count += 1;
            }
        }

        let mut kept_edges = Vec::with_capacity(mask.connect_edges.len());
        for &edge in &mask.connect_edges {
            if self.is_action_blocked(state, &ForwardAction::Connect(edge)) {
                blocked_count += 1;
            } else {
                kept_edges.push(edge);
            }
        }
        mask.connect_edges = kept_edges;

        for (node, allowed) in mask.abstract_nodes.iter_mut().enumerate() {
            if *allowed && self.is_action_blocked(state, &ForwardAction::Abstract(node)) {
                *allowed = false;
                blocked_count += 1;
            }
        }

        if mask.allow_stop && self.is_action_blocked(state, &ForwardAction::Stop) {
            mask.allow_stop = false;
            blocked_count += 1;
        }

        mask.family_mask = [
            mask.allow_add_rule || mask.allow_add_conc.iter().any(|&b| b),
            !mask.connect_edges.is_empty(),
            mask.abstract_nodes.iter().any(|&b| b),
            mask.allow_stop,
        ];

        let legal_actions = mask.to_actions(alphabet);
        MaskedForwardView {
            mask,
            legal_actions,
            blocked_count,
        }
    }
}

// ============================================================================
// 4. Distance structurelle, Archive multi-modes et Indice d'ambiguïté causale
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum StructuralFeature {
    SeedNode(NodeKind),
    SeedEdge(NodeKind, u8, NodeKind),
    RulePattern(CanonicalKey),
}

fn extract_structural_multiset(h: &HypothesisGraph) -> BTreeMap<StructuralFeature, usize> {
    let mut bag: BTreeMap<StructuralFeature, usize> = BTreeMap::new();

    for (v, &kind) in h.nodes().iter().enumerate().skip(1) {
        if h.membership(v) == Some(Block::Seed) {
            *bag.entry(StructuralFeature::SeedNode(kind)).or_insert(0) += 1;
        } else if kind == NodeKind::Rule {
            if let Some(pat) = extract_rule_pattern_key(h, v) {
                *bag.entry(StructuralFeature::RulePattern(pat)).or_insert(0) += 1;
            }
        }
    }

    for &e in h.edges() {
        if let EdgeKind::Struct(rel) = e.kind {
            if h.membership(e.src) == Some(Block::Seed) && h.membership(e.dst) == Some(Block::Seed)
            {
                let feat = StructuralFeature::SeedEdge(h.nodes()[e.src], rel, h.nodes()[e.dst]);
                *bag.entry(feat).or_insert(0) += 1;
            }
        }
    }

    bag
}

/// Distance structurelle métrique dans [0.0, 1.0] (Jaccard pondéré sur traits canoniques).
/// Vaut 0.0 ssi les deux graphes sont strictement isomorphes.
pub fn structural_distance(h1: &HypothesisGraph, h2: &HypothesisGraph) -> f64 {
    let k1 = h1.canonical_key();
    let k2 = h2.canonical_key();
    if k1 == k2 {
        return 0.0;
    }

    let m1 = extract_structural_multiset(h1);
    let m2 = extract_structural_multiset(h2);

    let mut inter = 0usize;
    let mut union = 0usize;
    let mut all_keys: BTreeSet<&StructuralFeature> = BTreeSet::new();
    all_keys.extend(m1.keys());
    all_keys.extend(m2.keys());

    for feat in all_keys {
        let c1 = m1.get(feat).copied().unwrap_or(0);
        let c2 = m2.get(feat).copied().unwrap_or(0);
        inter += c1.min(c2);
        union += c1.max(c2);
    }

    if union == 0 {
        return 0.05;
    }
    let jaccard = 1.0 - (inter as f64) / (union as f64);
    // Plancher minimal non nul si non-isomorphes (ex. arêtes non-SEED distinctes).
    jaccard.max(0.05)
}

/// Étape élémentaire enregistrée dans la trajectoire d'un mode archivé.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrajectoryStep {
    Forward(ForwardAction),
    Backward(BackwardAction),
}

/// Bassin causal survivant stocké dans l'archive multi-modes.
#[derive(Clone, Debug)]
pub struct ArchivedMode {
    pub graph: HypothesisGraph,
    pub canonical_key: CanonicalKey,
    pub trajectory: Vec<TrajectoryStep>,
    pub report: RefutationReport,
}

/// Diagnostic quantitatif d'ambiguïté causale sur les bassins survivants.
#[derive(Clone, Debug, PartialEq)]
pub struct AmbiguityReport {
    pub is_ambiguous: bool,
    pub mode_count: usize,
    /// Probabilités normalisées de Boltzmann/GFlowNet sur les bassins archivés.
    pub mode_weights: Vec<f64>,
    /// Entropie normalisée dans [0.0, 1.0] (0 = cause unique dominante, 1 = équi-ambiguïté).
    pub normalized_entropy: f64,
    /// Nombre effectif de causes concurrentes N_eff = exp(H) dans [1.0, mode_count].
    pub effective_modes: f64,
    /// Écart log_phi_sharp(1er) - log_phi_sharp(2e) (None si < 2 modes).
    pub top_score_gap: Option<f64>,
}

/// Archive des K meilleurs bassins causaux séparés d'au moins `min_distance`.
#[derive(Clone, Debug)]
pub struct MultiModeArchive {
    max_modes: usize,
    min_distance: f64,
    ambiguity_margin: f64,
    ambiguity_temperature: f64,
    modes: Vec<ArchivedMode>,
}

impl MultiModeArchive {
    pub fn new(
        max_modes: usize,
        min_distance: f64,
        ambiguity_margin: f64,
        ambiguity_temperature: f64,
    ) -> Self {
        Self {
            max_modes: max_modes.max(1),
            min_distance: min_distance.clamp(0.0, 1.0),
            ambiguity_margin: ambiguity_margin.max(0.0),
            ambiguity_temperature: ambiguity_temperature.max(1e-6),
            modes: Vec::new(),
        }
    }

    pub fn modes(&self) -> &[ArchivedMode] {
        &self.modes
    }

    pub fn best(&self) -> Option<&ArchivedMode> {
        self.modes.first()
    }

    /// Insère une hypothèse si elle ouvre un nouveau bassin ou améliore son bassin voisin.
    /// Retourne `true` si l'archive a été mise à jour.
    pub fn consider(
        &mut self,
        graph: HypothesisGraph,
        canonical_key: CanonicalKey,
        trajectory: Vec<TrajectoryStep>,
        report: RefutationReport,
    ) -> bool {
        // 1. Recherche du bassin le plus proche en distance structurelle.
        let mut nearest: Option<(usize, f64)> = None;
        for (idx, existing) in self.modes.iter().enumerate() {
            let d = structural_distance(&existing.graph, &graph);
            if nearest.is_none_or(|(_, best_d)| d < best_d) {
                nearest = Some((idx, d));
            }
        }

        // 2. Si dans le même bassin (d < delta), remplace uniquement si le score progresse.
        if let Some((idx, d)) = nearest {
            if d < self.min_distance {
                if report.log_phi_sharp > self.modes[idx].report.log_phi_sharp + 1e-12 {
                    self.modes[idx] = ArchivedMode {
                        graph,
                        canonical_key,
                        trajectory,
                        report,
                    };
                    self.sort_modes();
                    return true;
                }
                return false;
            }
        }

        // 3. Nouveau bassin distinct : insertion directe si place disponible, sinon éviction du plus faible.
        let entry = ArchivedMode {
            graph,
            canonical_key,
            trajectory,
            report,
        };
        if self.modes.len() < self.max_modes {
            self.modes.push(entry);
            self.sort_modes();
            true
        } else if let Some(worst) = self.modes.last() {
            if entry.report.log_phi_sharp > worst.report.log_phi_sharp + 1e-12 {
                let last_idx = self.modes.len() - 1;
                self.modes[last_idx] = entry;
                self.sort_modes();
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    fn sort_modes(&mut self) {
        self.modes.sort_by(|a, b| {
            b.report
                .log_phi_sharp
                .total_cmp(&a.report.log_phi_sharp)
                .then_with(|| a.canonical_key.cmp(&b.canonical_key))
        });
    }

    /// Calcule l'indice d'ambiguïté causale sur les bassins survivants.
    pub fn ambiguity_report(&self) -> AmbiguityReport {
        let m = self.modes.len();
        if m == 0 {
            return AmbiguityReport {
                is_ambiguous: false,
                mode_count: 0,
                mode_weights: Vec::new(),
                normalized_entropy: 0.0,
                effective_modes: 0.0,
                top_score_gap: None,
            };
        }
        if m == 1 {
            return AmbiguityReport {
                is_ambiguous: false,
                mode_count: 1,
                mode_weights: vec![1.0],
                normalized_entropy: 0.0,
                effective_modes: 1.0,
                top_score_gap: None,
            };
        }

        let max_log = self.modes[0].report.log_phi_sharp;
        let mut raw_w = Vec::with_capacity(m);
        let mut sum = 0.0;
        for mode in &self.modes {
            let scaled = ((mode.report.log_phi_sharp - max_log) / self.ambiguity_temperature)
                .clamp(-500.0, 0.0);
            let w = scaled.exp();
            raw_w.push(w);
            sum += w;
        }
        let mode_weights: Vec<f64> = raw_w.into_iter().map(|w| w / sum).collect();

        let entropy: f64 = mode_weights
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        let normalized_entropy = (entropy / (m as f64).ln()).clamp(0.0, 1.0);
        let effective_modes = entropy.exp();

        let top_gap = self.modes[0].report.log_phi_sharp - self.modes[1].report.log_phi_sharp;
        let is_ambiguous = top_gap <= self.ambiguity_margin && effective_modes >= 1.5;

        AmbiguityReport {
            is_ambiguous,
            mode_count: m,
            mode_weights,
            normalized_entropy,
            effective_modes,
            top_score_gap: Some(top_gap),
        }
    }
}

// ============================================================================
// 5. Interface `AbductivePolicy` et implémentations prêtes à l'emploi
// ============================================================================

/// Contrat commun entre le futur Backbone Relationnel GFlowNet et le contrôleur `ask`.
pub trait AbductivePolicy {
    /// Retourne un logit par action dans `masked.legal_actions`.
    fn score_forward(
        &mut self,
        state: &HypothesisGraph,
        obs: &ObservationGraph,
        residual: &ResidualGraph,
        masked: &MaskedForwardView,
    ) -> Vec<f64>;

    /// Retourne un logit par action dans `legal_backward`.
    fn score_backward(
        &mut self,
        state: &HypothesisGraph,
        obs: &ObservationGraph,
        residual: &ResidualGraph,
        mask: &BackwardMask,
        legal_backward: &[BackwardAction],
    ) -> Vec<f64>;
}

/// Politique uniforme (référence neutre et tests de couverture).
#[derive(Clone, Copy, Debug, Default)]
pub struct UniformPolicy;

impl AbductivePolicy for UniformPolicy {
    fn score_forward(
        &mut self,
        _state: &HypothesisGraph,
        _obs: &ObservationGraph,
        _residual: &ResidualGraph,
        masked: &MaskedForwardView,
    ) -> Vec<f64> {
        vec![0.0; masked.legal_actions.len()]
    }

    fn score_backward(
        &mut self,
        _state: &HypothesisGraph,
        _obs: &ObservationGraph,
        _residual: &ResidualGraph,
        _mask: &BackwardMask,
        legal_backward: &[BackwardAction],
    ) -> Vec<f64> {
        vec![0.0; legal_backward.len()]
    }
}

/// Politique heuristique légère guidée directement par le résidu R_t (sans poids entraînés).
#[derive(Clone, Copy, Debug, Default)]
pub struct ResidualGuidedHeuristicPolicy;

impl AbductivePolicy for ResidualGuidedHeuristicPolicy {
    fn score_forward(
        &mut self,
        state: &HypothesisGraph,
        _obs: &ObservationGraph,
        residual: &ResidualGraph,
        masked: &MaskedForwardView,
    ) -> Vec<f64> {
        let missing_syms: BTreeSet<u8> = residual.missing_nodes.iter().map(|&(_, s)| s).collect();
        let has_orphan_conc = (1..state.node_count()).any(|v| {
            matches!(state.nodes()[v], NodeKind::Conc(_)) && state.membership(v).is_none()
        });

        masked
            .legal_actions
            .iter()
            .map(|action| match *action {
                ForwardAction::Stop => {
                    if residual.is_zero() {
                        6.0
                    } else if residual.norm < 2.0 {
                        1.5
                    } else {
                        -2.0
                    }
                }
                ForwardAction::AddNode(AddNodeKind::Conc(sym)) => {
                    if missing_syms.contains(&sym) && !has_orphan_conc {
                        3.0
                    } else if missing_syms.contains(&sym) {
                        1.2
                    } else {
                        -0.5
                    }
                }
                ForwardAction::AddNode(AddNodeKind::Rule) => {
                    if !residual.is_zero() && state.rule_count() == 0 {
                        1.0
                    } else {
                        -0.5
                    }
                }
                ForwardAction::Connect(edge) => match edge.kind {
                    EdgeKind::InSeed => {
                        if !residual.missing_nodes.is_empty() {
                            3.5
                        } else {
                            0.2
                        }
                    }
                    EdgeKind::Struct(_) => {
                        if residual.summary.missing_edges > 0 {
                            3.2
                        } else {
                            0.0
                        }
                    }
                    EdgeKind::InLhs | EdgeKind::InRhs | EdgeKind::SameAs => 1.0,
                },
                ForwardAction::Abstract(_) => 0.4,
            })
            .collect()
    }

    fn score_backward(
        &mut self,
        state: &HypothesisGraph,
        _obs: &ObservationGraph,
        residual: &ResidualGraph,
        _mask: &BackwardMask,
        legal_backward: &[BackwardAction],
    ) -> Vec<f64> {
        let excess_syms: BTreeSet<u8> = residual.excess_nodes.iter().map(|&(_, s)| s).collect();
        legal_backward
            .iter()
            .map(|action| match *action {
                BackwardAction::UnStop => 5.0,
                BackwardAction::DelNode(v) => match state.nodes()[v] {
                    NodeKind::Conc(s) if excess_syms.contains(&s) => 2.5,
                    _ => 1.0,
                },
                BackwardAction::DelEdge(_) => 1.2,
                BackwardAction::UnAbstract { .. } => 0.3,
            })
            .collect()
    }
}

// ============================================================================
// 6. Contrôleur `ask` et boucle Back-and-Forth sous budget fixe B
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AskConfig {
    /// Budget strict B en nombre maximal de transitions (P_F + P_B) durant une session `ask`.
    pub budget_steps: usize,
    /// Profondeur maximale d'une marche avant continue avant évaluation forcée.
    pub max_forward_run: usize,
    /// Recul minimal k_min lors d'une phase P_B.
    pub min_backtrack_steps: usize,
    /// Recul maximal k_max (si résidu très élevé, peut aller jusqu'au redémarrage H_0).
    pub max_backtrack_steps: usize,
    /// Échelle de résidu r_0 calibrant le recul proportionnel k(R_T).
    pub residual_backtrack_scale: f64,
    /// Température softmax pour P_F.
    pub forward_temperature: f64,
    /// Température softmax pour P_B.
    pub backward_temperature: f64,
    /// Nombre maximal K de bassins conservés dans l'archive multi-modes.
    pub max_modes: usize,
    /// Distance structurelle minimale delta entre deux bassins distincts.
    pub mode_min_distance: f64,
    /// Écart maximal de log_phi_sharp pour signaler une ambiguïté causale.
    pub ambiguity_margin: f64,
    /// Température de Boltzmann pour l'entropie d'ambiguïté.
    pub ambiguity_temperature: f64,
    /// Si vrai, stoppe tôt si un mode a résidu nul et que `early_stop_after_modes` bassins sont trouvés.
    pub early_stop_after_modes: Option<usize>,
    pub refutation_cfg: RefutationConfig,
}

impl Default for AskConfig {
    fn default() -> Self {
        Self {
            budget_steps: 120,
            max_forward_run: 24,
            min_backtrack_steps: 2,
            max_backtrack_steps: 12,
            residual_backtrack_scale: 4.0,
            forward_temperature: 1.0,
            backward_temperature: 1.0,
            max_modes: 4,
            mode_min_distance: 0.25,
            ambiguity_margin: 0.75,
            ambiguity_temperature: 1.0,
            early_stop_after_modes: None,
            refutation_cfg: RefutationConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionStats {
    pub steps_used: usize,
    pub forward_steps: usize,
    pub backward_steps: usize,
    pub cycles_completed: usize,
    pub full_restarts: usize,
    pub actions_masked_by_nogoods: usize,
    pub parasitic_patterns_learned: usize,
    pub dead_states_recorded: usize,
    pub cache_stats: CacheStats,
}

#[derive(Clone, Debug)]
pub struct AskResult {
    pub modes: Vec<ArchivedMode>,
    pub ambiguity: AmbiguityReport,
    pub stats: SessionStats,
}

/// Contrôleur épisodique exécutant `ask --model out/sharp.phis --file perceived.eff`.
pub struct AskController {
    cfg: AskConfig,
}

impl AskController {
    pub fn new(cfg: AskConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> AskConfig {
        self.cfg
    }

    /// Exécute une session abductive Back-and-Forth bornée par `cfg.budget_steps`.
    pub fn ask<P: AbductivePolicy>(
        &self,
        obs: &ObservationGraph,
        graph_cfg: GraphConfig,
        policy: &mut P,
        seed: u64,
    ) -> AskResult {
        let mut rng = SplitMix64::new(seed);
        let mut cache = CanonicalCache::new();
        let mut nogoods = NogoodRegistry::new();
        let mut archive = MultiModeArchive::new(
            self.cfg.max_modes,
            self.cfg.mode_min_distance,
            self.cfg.ambiguity_margin,
            self.cfg.ambiguity_temperature,
        );
        let mut stats = SessionStats::default();

        let mut state = HypothesisGraph::new(graph_cfg.clone());
        let mut trajectory: Vec<TrajectoryStep> = Vec::new();

        while stats.steps_used < self.cfg.budget_steps {
            // ================================================================
            // Phase 1 : Marche avant P_F guidée par résidu R_t et masquée par nogoods
            // ================================================================
            let mut forward_in_cycle = 0usize;
            let mut hit_dead_end = false;

            while stats.steps_used < self.cfg.budget_steps
                && forward_in_cycle < self.cfg.max_forward_run
                && !state.is_stopped()
            {
                let key = state.canonical_key();
                let (_, residual) =
                    cache.get_or_compute_residual(&state, &key, obs, self.cfg.refutation_cfg);

                let raw_mask = state.forward_mask();
                let masked = nogoods.apply_mask(&state, &raw_mask);
                stats.actions_masked_by_nogoods += masked.blocked_count;

                if masked.legal_actions.is_empty() {
                    hit_dead_end = true;
                    break;
                }

                let logits = policy.score_forward(&state, obs, &residual, &masked);
                debug_assert_eq!(logits.len(), masked.legal_actions.len());
                let chosen_idx = sample_softmax(&logits, self.cfg.forward_temperature, &mut rng);
                let action = masked.legal_actions[chosen_idx];

                state = state
                    .apply_forward(&action)
                    .expect("Action issue de MaskedForwardView toujours légale");
                trajectory.push(TrajectoryStep::Forward(action));
                stats.steps_used += 1;
                stats.forward_steps += 1;
                forward_in_cycle += 1;

                if matches!(action, ForwardAction::Stop) {
                    break;
                }
            }

            // ================================================================
            // Phase 2 : Réfutation phi_sharp, mise à jour de l'archive et des nogoods
            // ================================================================
            let eval_residual_norm = if !state.is_initial() {
                let eval_graph = if state.is_stopped() {
                    state
                        .apply_backward(&BackwardAction::UnStop)
                        .expect("UnStop toujours valide sur un état STOP")
                } else {
                    state.clone()
                };
                let eval_key = eval_graph.canonical_key();
                let report = cache.get_or_compute_refutation(
                    &eval_graph,
                    &eval_key,
                    obs,
                    self.cfg.refutation_cfg,
                );
                let norm = report.residual.norm;

                // Enregistre les règles parasites détectées par l'ablation.
                stats.parasitic_patterns_learned +=
                    nogoods.record_from_refutation(&eval_graph, &report);

                // Marque le terminal STOP comme déjà exploré pour diversifier les prochains cycles.
                if state.is_stopped() {
                    nogoods.mark_evaluated_terminal(state.canonical_key());
                }
                if hit_dead_end {
                    nogoods.mark_dead_state(eval_key.clone());
                }

                archive.consider(eval_graph, eval_key, trajectory.clone(), report);
                stats.cycles_completed += 1;
                norm
            } else {
                // Même H_0 est bloqué : rien de plus à explorer.
                if hit_dead_end {
                    break;
                }
                self.cfg.residual_backtrack_scale
            };

            // Option d'arrêt anticipé si assez de modes parfaits ont été trouvés.
            if let Some(target_modes) = self.cfg.early_stop_after_modes {
                let perfect_modes = archive
                    .modes()
                    .iter()
                    .filter(|m| m.report.residual.is_zero())
                    .count();
                if perfect_modes >= target_modes {
                    break;
                }
            }

            if stats.steps_used >= self.cfg.budget_steps {
                break;
            }

            // ================================================================
            // Phase 3 : Recul Back-and-Forth (P_B) proportionnel au résidu
            // ================================================================
            let target_back = self.compute_backtrack_depth(eval_residual_norm, &state, hit_dead_end);
            let mut stepped_back = 0usize;

            while stepped_back < target_back
                && stats.steps_used < self.cfg.budget_steps
                && !state.is_initial()
            {
                let b_mask = state.backward_mask();
                let legal_b = b_mask.to_actions(state.edges(), &state.config().alphabet);
                if legal_b.is_empty() {
                    break;
                }

                let action = if state.is_stopped() {
                    BackwardAction::UnStop
                } else {
                    let key = state.canonical_key();
                    let (_, residual) =
                        cache.get_or_compute_residual(&state, &key, obs, self.cfg.refutation_cfg);
                    let logits =
                        policy.score_backward(&state, obs, &residual, &b_mask, &legal_b);
                    debug_assert_eq!(logits.len(), legal_b.len());
                    let idx = sample_softmax(&logits, self.cfg.backward_temperature, &mut rng);
                    legal_b[idx]
                };

                state = state
                    .apply_backward(&action)
                    .expect("Action P_B légale garantie par BackwardMask");
                trajectory.push(TrajectoryStep::Backward(action));
                stats.steps_used += 1;
                stats.backward_steps += 1;
                stepped_back += 1;
            }

            // Si le recul complet a ramené à H_0 ou si l'état courant est mort, redémarrage propre.
            if state.is_initial() {
                stats.full_restarts += 1;
                trajectory.clear();
            } else if hit_dead_end {
                let cur_mask = nogoods.apply_mask(&state, &state.forward_mask());
                if cur_mask.legal_actions.is_empty() {
                    nogoods.mark_dead_state(state.canonical_key());
                    state = HypothesisGraph::new(graph_cfg.clone());
                    trajectory.clear();
                    stats.full_restarts += 1;
                }
            }
        }

        stats.dead_states_recorded = nogoods.dead_state_count();
        stats.cache_stats = cache.stats;

        let ambiguity = archive.ambiguity_report();
        AskResult {
            modes: archive.modes().to_vec(),
            ambiguity,
            stats,
        }
    }

    /// Calcule la profondeur de recul k(R_T) : plus le résidu est élevé, plus le recul est profond.
    fn compute_backtrack_depth(
        &self,
        residual_norm: f64,
        state: &HypothesisGraph,
        hit_dead_end: bool,
    ) -> usize {
        let max_possible = state.potential().saturating_sub(1);
        if max_possible == 0 {
            return 0;
        }
        if hit_dead_end {
            // Sur une impasse, recul appuyé pour sortir du sous-arbre mort.
            return self.cfg.max_backtrack_steps.min(max_possible).max(1);
        }

        let scale = self.cfg.residual_backtrack_scale.max(1e-6);
        let ratio = (residual_norm / (residual_norm + scale)).clamp(0.0, 1.0);
        let span = self
            .cfg
            .max_backtrack_steps
            .saturating_sub(self.cfg.min_backtrack_steps) as f64;
        let raw = (self.cfg.min_backtrack_steps as f64 + span * ratio).ceil() as usize;

        // Si le résidu reste très élevé (> 2 * scale), autorise un recul jusqu'à H_0.
        if residual_norm > 2.0 * scale {
            max_possible
        } else {
            raw.clamp(1, max_possible)
        }
    }
}

// ============================================================================
// Tests unitaires complets
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_seq(mut g: HypothesisGraph, actions: &[ForwardAction]) -> HypothesisGraph {
        for a in actions {
            g = g.apply_forward(a).expect("Action P_F valide");
        }
        g
    }

    #[test]
    fn test_canonical_cache_hits_on_isomorphic_graphs() {
        let obs = ObservationGraph::from_bytes(b"ab");
        let cfg = GraphConfig::new(vec![b'a', b'b'], 1, 8, 2, 10).unwrap();
        let ref_cfg = RefutationConfig::default();

        // Ordre A : ajoute 'a' puis 'b'.
        let ga = apply_seq(
            HypothesisGraph::new(cfg.clone()),
            &[
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 1
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 2
                ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(0, 2, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(1, 2, EdgeKind::Struct(0))),
            ],
        );

        // Ordre B : ajoute 'b' puis 'a', même structure isomorphe.
        let gb = apply_seq(
            HypothesisGraph::new(cfg),
            &[
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 1
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 2
                ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(0, 2, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(2, 1, EdgeKind::Struct(0))),
            ],
        );

        assert_ne!(ga.nodes(), gb.nodes());
        let ka = ga.canonical_key();
        let kb = gb.canonical_key();
        assert_eq!(ka, kb);

        let mut cache = CanonicalCache::new();
        let (_, res_a) = cache.get_or_compute_residual(&ga, &ka, &obs, ref_cfg);
        let (_, res_b) = cache.get_or_compute_residual(&gb, &kb, &obs, ref_cfg);
        assert_eq!(res_a.norm, res_b.norm);
        assert_eq!(cache.stats.residual_misses, 1);
        assert_eq!(cache.stats.residual_hits, 1);

        let rep_a = cache.get_or_compute_refutation(&ga, &ka, &obs, ref_cfg);
        let rep_b = cache.get_or_compute_refutation(&gb, &kb, &obs, ref_cfg);
        assert!((rep_a.log_phi_sharp - rep_b.log_phi_sharp).abs() < 1e-12);
        assert_eq!(cache.stats.refutation_misses, 1);
        assert_eq!(cache.stats.refutation_hits, 1);
    }

    #[test]
    fn test_nogood_registry_masks_parasitic_rule_and_dead_state() {
        let obs = ObservationGraph::from_bytes(b"a");
        let cfg = GraphConfig::new(vec![b'a', b'z'], 1, 10, 3, 14).unwrap();
        let ref_cfg = RefutationConfig::default();

        // État de base : SEED = a, plus règle parasite z => a et règle incomplète vide.
        let base = apply_seq(
            HypothesisGraph::new(cfg),
            &[
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 1
                ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
                ForwardAction::AddNode(AddNodeKind::Rule),       // 2 (z => a : parasite)
                ForwardAction::AddNode(AddNodeKind::Conc(b'z')), // 3
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 4
                ForwardAction::Connect(Edge::new(2, 3, EdgeKind::InLhs)),
            ],
        );
        let closing_edge = Edge::new(2, 4, EdgeKind::InRhs);
        let with_parasitic = base
            .apply_forward(&ForwardAction::Connect(closing_edge))
            .unwrap()
            // Ajout d'une règle incomplète (MalformedOrSkipped) pour vérifier qu'elle n'est PAS bannie.
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Rule))
            .unwrap();

        let report = evaluate_phi_sharp(&with_parasitic, &obs, ref_cfg);
        let mut nogoods = NogoodRegistry::new();
        let added = nogoods.record_from_refutation(&with_parasitic, &report);

        // Seule la règle parasite exécutable (z => a) doit être enregistrée, pas la règle vide.
        assert_eq!(added, 1);
        assert_eq!(nogoods.parasitic_pattern_count(), 1);

        // Depuis `base`, l'arête qui referme z => a doit maintenant être masquée !
        let raw_mask = base.forward_mask();
        assert!(raw_mask.connect_edges.contains(&closing_edge));

        let masked = nogoods.apply_mask(&base, &raw_mask);
        assert!(!masked.mask.connect_edges.contains(&closing_edge));
        assert!(masked.blocked_count >= 1);

        // L'ajout d'un nouveau hub `RULE` reste parfaitement autorisé (règles incomplètes non bannies).
        assert!(masked.mask.allow_add_rule);

        // Un état impasse explicite doit aussi bloquer l'action qui y mène.
        let dead_action = ForwardAction::AddNode(AddNodeKind::Rule);
        let dead_state = base.apply_forward(&dead_action).unwrap();
        nogoods.mark_dead_state(dead_state.canonical_key());
        let masked2 = nogoods.apply_mask(&base, &raw_mask);
        assert!(!masked2.mask.allow_add_rule);
    }

    #[test]
    fn test_multimode_archive_detects_causal_ambiguity_and_deduplicates() {
        // Effet "cdb" avec deux causes symétriques distinctes de score identique :
        // Mode A : SEED = x -> b ; Règle : x => c -> d
        // Mode B : SEED = c -> y ; Règle : y => d -> b
        let obs = ObservationGraph::from_bytes(b"cdb");
        let cfg = GraphConfig::new(vec![b'b', b'c', b'd', b'x', b'y'], 1, 14, 2, 18).unwrap();
        let ref_cfg = RefutationConfig::default();

        let mode_a = apply_seq(
            HypothesisGraph::new(cfg.clone()),
            &[
                ForwardAction::AddNode(AddNodeKind::Conc(b'x')), // 1
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 2
                ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(0, 2, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(1, 2, EdgeKind::Struct(0))),
                ForwardAction::AddNode(AddNodeKind::Rule),       // 3
                ForwardAction::AddNode(AddNodeKind::Conc(b'x')), // 4
                ForwardAction::AddNode(AddNodeKind::Conc(b'c')), // 5
                ForwardAction::AddNode(AddNodeKind::Conc(b'd')), // 6
                ForwardAction::Connect(Edge::new(3, 4, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(3, 5, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(3, 6, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(5, 6, EdgeKind::Struct(0))),
            ],
        );

        let mode_b = apply_seq(
            HypothesisGraph::new(cfg.clone()),
            &[
                ForwardAction::AddNode(AddNodeKind::Conc(b'c')), // 1
                ForwardAction::AddNode(AddNodeKind::Conc(b'y')), // 2
                ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(0, 2, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(1, 2, EdgeKind::Struct(0))),
                ForwardAction::AddNode(AddNodeKind::Rule),       // 3
                ForwardAction::AddNode(AddNodeKind::Conc(b'y')), // 4
                ForwardAction::AddNode(AddNodeKind::Conc(b'd')), // 5
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 6
                ForwardAction::Connect(Edge::new(3, 4, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(3, 5, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(3, 6, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(5, 6, EdgeKind::Struct(0))),
            ],
        );

        // Vérifie que la distance structurelle sépare nettement les deux causes.
        let dist_ab = structural_distance(&mode_a, &mode_b);
        assert!(dist_ab > 0.5, "Distance attendue > 0.5, obtenue : {dist_ab}");

        let rep_a = evaluate_phi_sharp(&mode_a, &obs, ref_cfg);
        let rep_b = evaluate_phi_sharp(&mode_b, &obs, ref_cfg);
        assert!(rep_a.residual.is_zero() && rep_b.residual.is_zero());
        assert!((rep_a.log_phi_sharp - rep_b.log_phi_sharp).abs() < 1e-9);

        let mut archive = MultiModeArchive::new(4, 0.25, 0.5, 1.0);
        assert!(archive.consider(mode_a.clone(), mode_a.canonical_key(), Vec::new(), rep_a.clone()));
        assert!(!archive.ambiguity_report().is_ambiguous);

        // Ré-insertion du même bassin isomorphe : doit être dédupliquée (pas de doublon).
        assert!(!archive.consider(mode_a.clone(), mode_a.canonical_key(), Vec::new(), rep_a));
        assert_eq!(archive.modes().len(), 1);

        // Insertion du second bassin distinct : doit déclencher la détection d'ambiguïté causale.
        assert!(archive.consider(mode_b.clone(), mode_b.canonical_key(), Vec::new(), rep_b));
        assert_eq!(archive.modes().len(), 2);

        let amb = archive.ambiguity_report();
        assert!(amb.is_ambiguous, "Ambiguïté causale attendue : {amb:?}");
        assert!(amb.effective_modes > 1.95);
        assert!(amb.normalized_entropy > 0.99);
        assert!(amb.top_score_gap.unwrap() < 1e-6);

        // Cas non ambigu : archive contenant mode_a + une hypothèse médiocre (SEED = c seul).
        let weak = apply_seq(
            HypothesisGraph::new(cfg),
            &[
                ForwardAction::AddNode(AddNodeKind::Conc(b'c')),
                ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
            ],
        );
        let rep_weak = evaluate_phi_sharp(&weak, &obs, ref_cfg);
        let mut clear_archive = MultiModeArchive::new(4, 0.25, 0.5, 1.0);
        clear_archive.consider(
            mode_a.clone(),
            mode_a.canonical_key(),
            Vec::new(),
            evaluate_phi_sharp(&mode_a, &obs, ref_cfg),
        );
        clear_archive.consider(weak.clone(), weak.canonical_key(), Vec::new(), rep_weak);
        let clear_amb = clear_archive.ambiguity_report();
        assert!(!clear_amb.is_ambiguous, "Ne doit pas signaler d'ambiguïté : {clear_amb:?}");
    }

    #[test]
    fn test_ask_back_and_forth_respects_budget_and_improves_residual() {
        let obs = ObservationGraph::from_bytes(b"ab");
        let graph_cfg = GraphConfig::new(obs.alphabet(), 1, 8, 2, 10).unwrap();

        let ask_cfg = AskConfig {
            budget_steps: 80,
            max_forward_run: 14,
            forward_temperature: 0.7,
            backward_temperature: 1.0,
            ..AskConfig::default()
        };
        let controller = AskController::new(ask_cfg);
        let mut policy = ResidualGuidedHeuristicPolicy;

        let result = controller.ask(&obs, graph_cfg.clone(), &mut policy, 0x5048_4953);

        // Respect strict du budget B.
        assert!(result.stats.steps_used <= ask_cfg.budget_steps);
        assert!(result.stats.forward_steps > 0);
        assert!(result.stats.backward_steps > 0);
        assert!(result.stats.cycles_completed >= 1);
        assert!(!result.modes.is_empty());

        // La meilleure hypothèse trouvée doit nettement réduire le résidu par rapport à H_0.
        let empty_rep = evaluate_phi_sharp(
            &HypothesisGraph::new(graph_cfg),
            &obs,
            ask_cfg.refutation_cfg,
        );
        let best = &result.modes[0];
        assert!(
            best.report.residual.norm < empty_rep.residual.norm,
            "Résidu non réduit : {} vs {}",
            best.report.residual.norm,
            empty_rep.residual.norm
        );
    }
}
