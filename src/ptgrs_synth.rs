//! Phisharp V1 — Synthétiseur universel de tâches PTGRS (entraînement & validation).
//!
//! À déclarer dans `src/lib.rs` :
//!   pub mod hypothesis_graph;
//!   pub mod projector;
//!   pub mod refutation;
//!   pub mod memory_controller;
//!   pub mod ptgrs_synth;
//!
//! Puis exécuter :
//!   cargo test

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use crate::hypothesis_graph::{
    AddNodeKind, Edge, EdgeKind, ForwardAction, GraphConfig, HypothesisGraph, NodeId, NodeKind,
    RelId, SymbolId,
};
use crate::projector::{
    ObservationGraph, TerminationReason, project,
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

    fn range_usize(&mut self, low: usize, high_inclusive: usize) -> usize {
        if high_inclusive <= low {
            return low;
        }
        let span = high_inclusive - low + 1;
        low + ((self.next_u64() as usize) % span)
    }

    fn next_bool(&mut self, num: u64, den: u64) -> bool {
        debug_assert!(den > 0);
        (self.next_u64() % den) < num
    }

    fn pick<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        debug_assert!(!slice.is_empty());
        let idx = (self.next_u64() as usize) % slice.len();
        &slice[idx]
    }

    /// Tire `k` éléments distincts sans remise dans `slice`.
    fn pick_distinct<T: Copy>(&mut self, slice: &[T], k: usize) -> Option<Vec<T>> {
        if k > slice.len() {
            return None;
        }
        let mut pool = slice.to_vec();
        let mut out = Vec::with_capacity(k);
        for _ in 0..k {
            let idx = (self.next_u64() as usize) % pool.len();
            out.push(pool.swap_remove(idx));
        }
        Some(out)
    }
}

// ============================================================================
// 2. Types publics (isolation stricte TrainingSample vs EvalSample)
// ============================================================================

/// Familles structurelles couvertes par le synthétiseur PTGRS.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuleArchetype {
    /// Expansion déterministe d'un motif concret répété.
    ConcreteExpansion,
    /// Duplication abstraite d'un motif arbitraire via VAR + SAME_AS.
    AbstractDuplication,
    /// Permutation de deux variables distinctes (X, Y) via deux classes SAME_AS.
    AbstractSwap,
    /// Encadrement d'une variable abstraite par un gabarit concret.
    AbstractBracketing,
    /// Réécriture non-linéaire multi-relations (embranchement / DAG).
    MultiRelTransform,
    /// Chaîne causale à 2 règles séquentielles (la règle 1 déclenche la règle 2).
    ChainedRules,
    /// Synthèse aléatoire ouverte (germe DAG + règles aléatoires filtrées).
    FreeRandom,
}

/// Configuration du synthétiseur PTGRS et du curriculum.
#[derive(Clone, Debug)]
pub struct SynthConfig {
    /// Réservoir d'octets disponibles pour fabriquer les effets.
    pub alphabet_pool: Vec<SymbolId>,
    /// Nombre de relations structurelles `Struct(0..num_struct_rels)`.
    pub num_struct_rels: u8,
    /// Taille minimale (en nœuds) de l'effet observable G_E.
    pub min_effect_nodes: usize,
    /// Taille maximale (en nœuds) de l'effet observable G_E.
    pub max_effect_nodes: usize,
    /// Nombre minimal d'étapes de réécriture effectivement déclenchées.
    pub min_rewrite_steps: usize,
    /// Niveau de curriculum :
    /// - 0 : 1 règle concrète ou duplication simple
    /// - 1 : 1 règle abstraite riche (duplication, swap, encadrement, expansion)
    /// - 2 : ajout multi-relations et chaînes à 2 règles
    /// - 3 : distribution universelle complète incluant `FreeRandom`
    pub difficulty: u8,
    /// Si vrai, garantit que tout symbole concret de H_true apparaît dans G_E
    /// (alphabet 100 % auto-spécifié par `perceived.eff`).
    pub require_self_specified: bool,
    /// Si vrai, vérifie via `evaluate_phi_sharp` que chaque règle de H_true est `Necessary`.
    pub require_all_rules_necessary: bool,
    /// Si vrai, n'accepte que les tâches où H_true bat strictement la copie triviale H = E.
    pub require_beats_verbatim: bool,
    /// Tentatives maximales avant d'abandonner un tirage.
    pub max_attempts_per_sample: usize,
    /// Marge de nœuds ajoutée à `recommended_graph_cfg` pour la recherche GFlowNet.
    pub headroom_nodes: usize,
    /// Marge de règles ajoutée à `recommended_graph_cfg`.
    pub headroom_rules: usize,
    /// Marge d'arêtes ajoutée à `recommended_graph_cfg`.
    pub headroom_edges: usize,
    /// Configuration du banc de réfutation utilisée pour valider H_true.
    pub refutation_cfg: RefutationConfig,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            alphabet_pool: b"abcdefgh".to_vec(),
            num_struct_rels: 2,
            min_effect_nodes: 3,
            max_effect_nodes: 14,
            min_rewrite_steps: 1,
            difficulty: 3,
            require_self_specified: true,
            require_all_rules_necessary: true,
            require_beats_verbatim: false,
            max_attempts_per_sample: 160,
            headroom_nodes: 4,
            headroom_rules: 2,
            headroom_edges: 10,
            refutation_cfg: RefutationConfig::default(),
        }
    }
}

/// Échantillon livré à la boucle d'entraînement : NE CONTIENT JAMAIS la cause cachée H_true.
#[derive(Clone, Debug)]
pub struct TrainingSample {
    pub sample_id: u64,
    pub effect: ObservationGraph,
    pub recommended_graph_cfg: GraphConfig,
    pub difficulty: u8,
}

/// Échantillon complet réservé à l'évaluation hors-ligne après coup.
#[derive(Clone, Debug)]
pub struct EvalSample {
    pub training: TrainingSample,
    pub hidden_cause: HypothesisGraph,
    pub hidden_report: RefutationReport,
    pub archetype: RuleArchetype,
    pub rewrite_steps: usize,
}

/// Compteurs de diagnostic du filtrage PTGRS.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SynthStats {
    pub accepted: usize,
    pub rejected_construction: usize,
    pub rejected_non_normal: usize,
    pub rejected_too_few_steps: usize,
    pub rejected_effect_size: usize,
    pub rejected_not_self_specified: usize,
    pub rejected_parasitic_rule: usize,
    pub rejected_not_beating_verbatim: usize,
    pub by_archetype: BTreeMap<RuleArchetype, usize>,
}

// ============================================================================
// 3. Aide à la construction sûre de `HypothesisGraph` (via API publique P_F)
// ============================================================================

struct GraphBuilder {
    graph: HypothesisGraph,
}

impl GraphBuilder {
    fn new(cfg: GraphConfig) -> Self {
        Self {
            graph: HypothesisGraph::new(cfg),
        }
    }

    fn finish(self) -> HypothesisGraph {
        self.graph
    }

    fn add_rule(&mut self) -> Option<NodeId> {
        let id = self.graph.node_count();
        self.graph = self
            .graph
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Rule))
            .ok()?;
        Some(id)
    }

    fn add_seed_conc(&mut self, sym: SymbolId) -> Option<NodeId> {
        let id = self.graph.node_count();
        self.graph = self
            .graph
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(sym)))
            .ok()?
            .apply_forward(&ForwardAction::Connect(Edge::new(0, id, EdgeKind::InSeed)))
            .ok()?;
        Some(id)
    }

    fn add_rule_conc(&mut self, rule: NodeId, in_lhs: bool, sym: SymbolId) -> Option<NodeId> {
        let id = self.graph.node_count();
        let kind = if in_lhs {
            EdgeKind::InLhs
        } else {
            EdgeKind::InRhs
        };
        self.graph = self
            .graph
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(sym)))
            .ok()?
            .apply_forward(&ForwardAction::Connect(Edge::new(rule, id, kind)))
            .ok()?;
        Some(id)
    }

    fn add_rule_var(&mut self, rule: NodeId, in_lhs: bool) -> Option<NodeId> {
        let fallback = self.graph.config().alphabet[0];
        let id = self.add_rule_conc(rule, in_lhs, fallback)?;
        self.graph = self
            .graph
            .apply_forward(&ForwardAction::Abstract(id))
            .ok()?;
        Some(id)
    }

    fn connect_struct(&mut self, src: NodeId, dst: NodeId, rel: RelId) -> Option<()> {
        self.graph = self
            .graph
            .apply_forward(&ForwardAction::Connect(Edge::new(
                src,
                dst,
                EdgeKind::Struct(rel),
            )))
            .ok()?;
        Some(())
    }

    fn try_connect_struct(&mut self, src: NodeId, dst: NodeId, rel: RelId) -> bool {
        let action = ForwardAction::Connect(Edge::new(src, dst, EdgeKind::Struct(rel)));
        if let Ok(next) = self.graph.apply_forward(&action) {
            self.graph = next;
            true
        } else {
            false
        }
    }

    fn connect_chain(&mut self, nodes: &[NodeId], rel: RelId) -> Option<()> {
        for pair in nodes.windows(2) {
            self.connect_struct(pair[0], pair[1], rel)?;
        }
        Some(())
    }

    fn connect_same_as(&mut self, u: NodeId, v: NodeId) -> Option<()> {
        self.graph = self
            .graph
            .apply_forward(&ForwardAction::Connect(Edge::new(u, v, EdgeKind::SameAs)))
            .ok()?;
        Some(())
    }
}

// ============================================================================
// 4. Synthétiseur PTGRS principal
// ============================================================================

pub struct PtgrsSynthesizer {
    cfg: SynthConfig,
    rng: SplitMix64,
    next_id: u64,
    stats: SynthStats,
}

impl PtgrsSynthesizer {
    pub fn new(mut cfg: SynthConfig, seed: u64) -> Result<Self, &'static str> {
        cfg.alphabet_pool.sort_unstable();
        cfg.alphabet_pool.dedup();
        if cfg.alphabet_pool.len() < 2 {
            return Err("alphabet_pool doit contenir au moins 2 symboles distincts");
        }
        if cfg.num_struct_rels == 0 {
            return Err("num_struct_rels doit être >= 1");
        }
        if cfg.min_effect_nodes == 0 || cfg.min_effect_nodes > cfg.max_effect_nodes {
            return Err("Bornes min_effect_nodes / max_effect_nodes invalides");
        }
        Ok(Self {
            cfg,
            rng: SplitMix64::new(seed),
            next_id: 0,
            stats: SynthStats::default(),
        })
    }

    pub fn config(&self) -> &SynthConfig {
        &self.cfg
    }

    pub fn stats(&self) -> &SynthStats {
        &self.stats
    }

    /// Ajuste le niveau de curriculum (0..=3) à la volée entre deux époques.
    pub fn set_difficulty(&mut self, difficulty: u8) {
        self.cfg.difficulty = difficulty.min(3);
    }

    /// Livre UNIQUEMENT l'effet observable G_E pour l'entraînement (cause cachée écartée).
    pub fn next_training_sample(&mut self) -> Option<TrainingSample> {
        self.next_eval_sample().map(|eval| eval.training)
    }

    /// Livre un lot d'échantillons d'entraînement isolés.
    pub fn next_training_batch(&mut self, batch_size: usize) -> Vec<TrainingSample> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            if let Some(sample) = self.next_training_sample() {
                batch.push(sample);
            } else {
                break;
            }
        }
        batch
    }

    /// Livre un échantillon complet (effet + cause cachée) pour l'évaluation hors-ligne.
    pub fn next_eval_sample(&mut self) -> Option<EvalSample> {
        for _ in 0..self.cfg.max_attempts_per_sample {
            let archetype = self.choose_archetype();
            if let Some(sample) = self.try_single_sample(archetype) {
                return Some(sample);
            }
        }
        None
    }

    /// Livre un lot d'échantillons complets pour un banc de validation.
    pub fn next_eval_batch(&mut self, batch_size: usize) -> Vec<EvalSample> {
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            if let Some(sample) = self.next_eval_sample() {
                batch.push(sample);
            } else {
                break;
            }
        }
        batch
    }

    /// Synthétise un échantillon d'une famille précise (utile pour tests ciblés et benchmarks).
    pub fn sample_archetype(&mut self, archetype: RuleArchetype) -> Option<EvalSample> {
        for _ in 0..self.cfg.max_attempts_per_sample {
            if let Some(sample) = self.try_single_sample(archetype) {
                return Some(sample);
            }
        }
        None
    }

    fn choose_archetype(&mut self) -> RuleArchetype {
        let allow_multi_rel = self.cfg.num_struct_rels >= 2;
        let allow_chain = self.cfg.alphabet_pool.len() >= 3 && self.cfg.max_effect_nodes >= 5;

        let mut pool = Vec::new();
        match self.cfg.difficulty {
            0 => {
                pool.push(RuleArchetype::ConcreteExpansion);
                pool.push(RuleArchetype::AbstractDuplication);
            }
            1 => {
                pool.push(RuleArchetype::ConcreteExpansion);
                pool.push(RuleArchetype::AbstractDuplication);
                pool.push(RuleArchetype::AbstractSwap);
                pool.push(RuleArchetype::AbstractBracketing);
            }
            2 => {
                pool.push(RuleArchetype::ConcreteExpansion);
                pool.push(RuleArchetype::AbstractDuplication);
                pool.push(RuleArchetype::AbstractSwap);
                pool.push(RuleArchetype::AbstractBracketing);
                if allow_multi_rel {
                    pool.push(RuleArchetype::MultiRelTransform);
                }
                if allow_chain {
                    pool.push(RuleArchetype::ChainedRules);
                }
            }
            _ => {
                pool.push(RuleArchetype::ConcreteExpansion);
                pool.push(RuleArchetype::AbstractDuplication);
                pool.push(RuleArchetype::AbstractSwap);
                pool.push(RuleArchetype::AbstractBracketing);
                if allow_multi_rel {
                    pool.push(RuleArchetype::MultiRelTransform);
                }
                if allow_chain {
                    pool.push(RuleArchetype::ChainedRules);
                }
                pool.push(RuleArchetype::FreeRandom);
            }
        }

        *self.rng.pick(&pool)
    }

    fn internal_graph_cfg(&self) -> GraphConfig {
        GraphConfig::new(
            self.cfg.alphabet_pool.clone(),
            self.cfg.num_struct_rels,
            32,
            4,
            64,
        )
        .expect("SynthConfig déjà validée")
    }

    fn try_single_sample(&mut self, archetype: RuleArchetype) -> Option<EvalSample> {
        let raw_cause = match archetype {
            RuleArchetype::ConcreteExpansion => self.build_concrete_expansion(),
            RuleArchetype::AbstractDuplication => self.build_abstract_duplication(),
            RuleArchetype::AbstractSwap => self.build_abstract_swap(),
            RuleArchetype::AbstractBracketing => self.build_abstract_bracketing(),
            RuleArchetype::MultiRelTransform => self.build_multi_rel_transform(),
            RuleArchetype::ChainedRules => self.build_chained_rules(),
            RuleArchetype::FreeRandom => self.build_free_random(),
        };

        let Some(raw_cause) = raw_cause else {
            self.stats.rejected_construction += 1;
            return None;
        };

        // 1. Exécution du projecteur Pi(H_true) pour obtenir l'effet observable G_E.
        let outcome = project(&raw_cause, self.cfg.refutation_cfg.projector_cfg);
        if outcome.termination != TerminationReason::NormalForm {
            self.stats.rejected_non_normal += 1;
            return None;
        }
        if outcome.steps_executed < self.cfg.min_rewrite_steps {
            self.stats.rejected_too_few_steps += 1;
            return None;
        }
        let effect = outcome.predicted;
        if effect.node_count() < self.cfg.min_effect_nodes
            || effect.node_count() > self.cfg.max_effect_nodes
        {
            self.stats.rejected_effect_size += 1;
            return None;
        }

        // 2. Vérification d'auto-spécification : tous les symboles concrets de H_true dans G_E.
        let effect_alpha = effect.alphabet();
        let effect_alpha_set: BTreeSet<SymbolId> = effect_alpha.iter().copied().collect();
        let cause_symbols = collect_concrete_symbols(&raw_cause);
        let is_self_specified = cause_symbols.is_subset(&effect_alpha_set);
        if self.cfg.require_self_specified && !is_self_specified {
            self.stats.rejected_not_self_specified += 1;
            return None;
        }

        // 3. Construction de `recommended_graph_cfg` et reconstruction exacte de H_true dedans.
        let rec_alpha = if self.cfg.require_self_specified {
            effect_alpha
        } else {
            let mut merged: BTreeSet<SymbolId> = effect_alpha_set;
            merged.extend(cause_symbols);
            merged.into_iter().collect()
        };

        let max_nodes = raw_cause
            .node_count()
            .max(effect.node_count() + 1)
            + self.cfg.headroom_nodes;
        let max_rules = raw_cause.rule_count().max(1) + self.cfg.headroom_rules;
        let max_edges = raw_cause
            .edges()
            .len()
            .max(effect.edge_count() + effect.node_count())
            + self.cfg.headroom_edges;

        let recommended_graph_cfg = GraphConfig::new(
            rec_alpha,
            self.cfg.num_struct_rels,
            max_nodes,
            max_rules,
            max_edges,
        )
        .ok()?;

        let hidden_cause = rebuild_with_config(&raw_cause, recommended_graph_cfg.clone())?;

        // 4. Vérification de nécessité causale et (optionnellement) supériorité sur H = E.
        let hidden_report = evaluate_phi_sharp(&hidden_cause, &effect, self.cfg.refutation_cfg);
        if !hidden_report.residual.is_zero() {
            self.stats.rejected_non_normal += 1;
            return None;
        }
        if self.cfg.require_all_rules_necessary {
            let all_necessary = !hidden_report.rule_reports.is_empty()
                && hidden_report
                    .rule_reports
                    .iter()
                    .all(|r| r.status == RuleStatus::Necessary);
            if !all_necessary {
                self.stats.rejected_parasitic_rule += 1;
                return None;
            }
        }
        if self.cfg.require_beats_verbatim {
            let verbatim = build_verbatim_copy(&effect, recommended_graph_cfg.clone())?;
            let verbatim_report = evaluate_phi_sharp(&verbatim, &effect, self.cfg.refutation_cfg);
            if hidden_report.log_phi_sharp <= verbatim_report.log_phi_sharp + 1e-6 {
                self.stats.rejected_not_beating_verbatim += 1;
                return None;
            }
        }

        let sample_id = self.next_id;
        self.next_id += 1;
        self.stats.accepted += 1;
        *self.stats.by_archetype.entry(archetype).or_insert(0) += 1;

        let training = TrainingSample {
            sample_id,
            effect,
            recommended_graph_cfg,
            difficulty: self.cfg.difficulty,
        };

        Some(EvalSample {
            training,
            hidden_cause,
            hidden_report,
            archetype,
            rewrite_steps: outcome.steps_executed,
        })
    }

    // ========================================================================
    // 5. Constructeurs d'archétypes PTGRS (terminaison garantie)
    // ========================================================================

    /// Famille 1 : Expansion concrète répétée.
    /// - Mode latent (!self_specified) : SEED = m^k ; Règle : m => a -> b -> c
    /// - Mode auto-spécifié (>= 3 sym) : SEED = (a -> b)^k ; Règle : a -> b => c -> [d ->] b -> a
    /// - Mode auto-spécifié (2 sym)    : SEED = (a -> a -> b)^k ; Règle : a -> a => a -> b -> a
    fn build_concrete_expansion(&mut self) -> Option<HypothesisGraph> {
        let mut b = GraphBuilder::new(self.internal_graph_cfg());

        // Mode marqueur latent (uniquement si auto-spécification désactivée).
        if !self.cfg.require_self_specified && self.cfg.alphabet_pool.len() >= 4 {
            let syms = self.rng.pick_distinct(&self.cfg.alphabet_pool, 4)?;
            let (m, s0, s1, s2) = (syms[0], syms[1], syms[2], syms[3]);
            let max_k = (self.cfg.max_effect_nodes / 3).clamp(1, 4);
            let min_k = (self.cfg.min_effect_nodes.div_ceil(3)).clamp(1, max_k);
            let k = self.rng.range_usize(min_k, max_k);

            let mut seed_ids = Vec::with_capacity(k);
            for _ in 0..k {
                seed_ids.push(b.add_seed_conc(m)?);
            }
            b.connect_chain(&seed_ids, 0)?;

            let rule = b.add_rule()?;
            let _lhs = b.add_rule_conc(rule, true, m)?;
            let r0 = b.add_rule_conc(rule, false, s0)?;
            let r1 = b.add_rule_conc(rule, false, s1)?;
            let r2 = b.add_rule_conc(rule, false, s2)?;
            b.connect_chain(&[r0, r1, r2], 0)?;
            return Some(b.finish());
        }

        if self.cfg.alphabet_pool.len() >= 3 {
            let use_len4 = self.cfg.alphabet_pool.len() >= 4
                && self.cfg.max_effect_nodes >= 8
                && (self.cfg.require_beats_verbatim || self.rng.next_bool(1, 2));
            let syms = self
                .rng
                .pick_distinct(&self.cfg.alphabet_pool, if use_len4 { 4 } else { 3 })?;
            let (sym_a, sym_b, sym_c) = (syms[0], syms[1], syms[2]);
            let rhs_len = if use_len4 { 4 } else { 3 };

            let max_k = (self.cfg.max_effect_nodes / rhs_len).clamp(1, 4);
            let min_k = self
                .cfg
                .min_effect_nodes
                .div_ceil(rhs_len)
                .max(if self.cfg.require_beats_verbatim { 3 } else { 1 })
                .min(max_k);
            let k = self.rng.range_usize(min_k, max_k);

            // SEED = (a -> b)^k : aucune arête `b -> b`, donc a -> b => c -> [d ->] b -> a termine en k pas.
            let mut seed_ids = Vec::with_capacity(2 * k);
            for _ in 0..k {
                seed_ids.push(b.add_seed_conc(sym_a)?);
                seed_ids.push(b.add_seed_conc(sym_b)?);
            }
            b.connect_chain(&seed_ids, 0)?;

            let rule = b.add_rule()?;
            let l0 = b.add_rule_conc(rule, true, sym_a)?;
            let l1 = b.add_rule_conc(rule, true, sym_b)?;
            b.connect_struct(l0, l1, 0)?;

            let mut rhs_ids = Vec::with_capacity(rhs_len);
            rhs_ids.push(b.add_rule_conc(rule, false, sym_c)?);
            if use_len4 {
                rhs_ids.push(b.add_rule_conc(rule, false, syms[3])?);
            }
            rhs_ids.push(b.add_rule_conc(rule, false, sym_b)?);
            rhs_ids.push(b.add_rule_conc(rule, false, sym_a)?);
            b.connect_chain(&rhs_ids, 0)?;
            Some(b.finish())
        } else {
            // Repli 2 symboles : a -> a => a -> b -> a
            let syms = self.rng.pick_distinct(&self.cfg.alphabet_pool, 2)?;
            let (sym_a, sym_b) = (syms[0], syms[1]);
            let max_k = ((self.cfg.max_effect_nodes + 1) / 4).clamp(1, 3);
            let k = self.rng.range_usize(1, max_k);

            let mut seed_ids = Vec::new();
            for block in 0..k {
                if block > 0 {
                    seed_ids.push(b.add_seed_conc(sym_b)?);
                }
                seed_ids.push(b.add_seed_conc(sym_a)?);
                seed_ids.push(b.add_seed_conc(sym_a)?);
            }
            b.connect_chain(&seed_ids, 0)?;

            let rule = b.add_rule()?;
            let l0 = b.add_rule_conc(rule, true, sym_a)?;
            let l1 = b.add_rule_conc(rule, true, sym_a)?;
            b.connect_struct(l0, l1, 0)?;

            let r0 = b.add_rule_conc(rule, false, sym_a)?;
            let r1 = b.add_rule_conc(rule, false, sym_b)?;
            let r2 = b.add_rule_conc(rule, false, sym_a)?;
            b.connect_chain(&[r0, r1, r2], 0)?;
            Some(b.finish())
        }
    }

    /// Famille 2 : Duplication abstraite (VAR + SAME_AS).
    /// Règle : a -> b -> X => b -> a -> X -> X  (avec X != b : terminaison exacte en k pas).
    fn build_abstract_duplication(&mut self) -> Option<HypothesisGraph> {
        let mut b = GraphBuilder::new(self.internal_graph_cfg());
        let syms = self.rng.pick_distinct(&self.cfg.alphabet_pool, 2)?;
        let (sym_a, sym_b) = (syms[0], syms[1]);

        let payload_candidates: Vec<SymbolId> = self
            .cfg
            .alphabet_pool
            .iter()
            .copied()
            .filter(|&s| s != sym_b)
            .collect();
        if payload_candidates.is_empty() {
            return None;
        }

        let max_k = (self.cfg.max_effect_nodes / 4).clamp(1, 3);
        let min_k = if self.cfg.require_beats_verbatim {
            2.min(max_k)
        } else {
            1
        };
        let k = self.rng.range_usize(min_k, max_k);

        let mut seed_ids = Vec::with_capacity(3 * k);
        for _ in 0..k {
            let payload = *self.rng.pick(&payload_candidates);
            seed_ids.push(b.add_seed_conc(sym_a)?);
            seed_ids.push(b.add_seed_conc(sym_b)?);
            seed_ids.push(b.add_seed_conc(payload)?);
        }
        b.connect_chain(&seed_ids, 0)?;

        let rule = b.add_rule()?;
        let l0 = b.add_rule_conc(rule, true, sym_a)?;
        let l1 = b.add_rule_conc(rule, true, sym_b)?;
        let lx = b.add_rule_var(rule, true)?;
        b.connect_chain(&[l0, l1, lx], 0)?;

        let r0 = b.add_rule_conc(rule, false, sym_b)?;
        let r1 = b.add_rule_conc(rule, false, sym_a)?;
        let rx1 = b.add_rule_var(rule, false)?;
        let rx2 = b.add_rule_var(rule, false)?;
        b.connect_chain(&[r0, r1, rx1, rx2], 0)?;

        // Fermeture SAME_AS : lx == rx1 == rx2
        b.connect_same_as(lx, rx1)?;
        b.connect_same_as(rx1, rx2)?;
        Some(b.finish())
    }

    /// Famille 3 : Permutation (SWAP) de deux variables distinctes (X, Y) via deux classes SAME_AS.
    /// Règle : a -> b -> X -> Y => b -> a -> Y -> X  (avec X != Y et X, Y != b).
    fn build_abstract_swap(&mut self) -> Option<HypothesisGraph> {
        if self.cfg.alphabet_pool.len() < 3 || self.cfg.max_effect_nodes < 4 {
            return None;
        }
        let mut b = GraphBuilder::new(self.internal_graph_cfg());
        let trigger = self.rng.pick_distinct(&self.cfg.alphabet_pool, 2)?;
        let (sym_a, sym_b) = (trigger[0], trigger[1]);

        let payload_pool: Vec<SymbolId> = self
            .cfg
            .alphabet_pool
            .iter()
            .copied()
            .filter(|&s| s != sym_b)
            .collect();
        if payload_pool.len() < 2 {
            return None;
        }

        let max_k = (self.cfg.max_effect_nodes / 4).clamp(1, 3);
        let k = self.rng.range_usize(1, max_k);

        let mut seed_ids = Vec::with_capacity(4 * k);
        for _ in 0..k {
            let pair = self.rng.pick_distinct(&payload_pool, 2)?;
            seed_ids.push(b.add_seed_conc(sym_a)?);
            seed_ids.push(b.add_seed_conc(sym_b)?);
            seed_ids.push(b.add_seed_conc(pair[0])?);
            seed_ids.push(b.add_seed_conc(pair[1])?);
        }
        b.connect_chain(&seed_ids, 0)?;

        let rule = b.add_rule()?;
        let l0 = b.add_rule_conc(rule, true, sym_a)?;
        let l1 = b.add_rule_conc(rule, true, sym_b)?;
        let lx = b.add_rule_var(rule, true)?;
        let ly = b.add_rule_var(rule, true)?;
        b.connect_chain(&[l0, l1, lx, ly], 0)?;

        let r0 = b.add_rule_conc(rule, false, sym_b)?;
        let r1 = b.add_rule_conc(rule, false, sym_a)?;
        let ry = b.add_rule_var(rule, false)?;
        let rx = b.add_rule_var(rule, false)?;
        b.connect_chain(&[r0, r1, ry, rx], 0)?;

        b.connect_same_as(lx, rx)?;
        b.connect_same_as(ly, ry)?;
        Some(b.finish())
    }

    /// Famille 4 : Encadrement d'une variable abstraite par un gabarit concret.
    /// Règle : a -> b -> X => c -> X -> b -> a  (terminaison exacte en k pas).
    fn build_abstract_bracketing(&mut self) -> Option<HypothesisGraph> {
        if self.cfg.alphabet_pool.len() < 3 || self.cfg.max_effect_nodes < 4 {
            return None;
        }
        let mut b = GraphBuilder::new(self.internal_graph_cfg());
        let frame = self.rng.pick_distinct(&self.cfg.alphabet_pool, 3)?;
        let (sym_a, sym_b, sym_c) = (frame[0], frame[1], frame[2]);

        let payload_pool: Vec<SymbolId> = self
            .cfg
            .alphabet_pool
            .iter()
            .copied()
            .filter(|&s| s != sym_b)
            .collect();

        let max_k = (self.cfg.max_effect_nodes / 4).clamp(1, 3);
        let k = self.rng.range_usize(1, max_k);

        let mut seed_ids = Vec::with_capacity(3 * k);
        for _ in 0..k {
            let payload = *self.rng.pick(&payload_pool);
            seed_ids.push(b.add_seed_conc(sym_a)?);
            seed_ids.push(b.add_seed_conc(sym_b)?);
            seed_ids.push(b.add_seed_conc(payload)?);
        }
        b.connect_chain(&seed_ids, 0)?;

        let rule = b.add_rule()?;
        let l0 = b.add_rule_conc(rule, true, sym_a)?;
        let l1 = b.add_rule_conc(rule, true, sym_b)?;
        let lx = b.add_rule_var(rule, true)?;
        b.connect_chain(&[l0, l1, lx], 0)?;

        let r0 = b.add_rule_conc(rule, false, sym_c)?;
        let rx = b.add_rule_var(rule, false)?;
        let r2 = b.add_rule_conc(rule, false, sym_b)?;
        let r3 = b.add_rule_conc(rule, false, sym_a)?;
        b.connect_chain(&[r0, rx, r2, r3], 0)?;

        b.connect_same_as(lx, rx)?;
        Some(b.finish())
    }

    /// Famille 5 : Réécriture multi-relations non-linéaire (embranchement en triangle DAG).
    /// Règle : a -(rel 0)-> X => a -(rel 1)-> X1, a -(rel 1)-> X2, X1 -(rel 0)-> X2 (X != a).
    fn build_multi_rel_transform(&mut self) -> Option<HypothesisGraph> {
        if self.cfg.num_struct_rels < 2 || self.cfg.max_effect_nodes < 3 {
            return None;
        }
        let mut b = GraphBuilder::new(self.internal_graph_cfg());
        let syms = self.rng.pick_distinct(&self.cfg.alphabet_pool, 2)?;
        let sym_a = syms[0];

        let payload_pool: Vec<SymbolId> = self
            .cfg
            .alphabet_pool
            .iter()
            .copied()
            .filter(|&s| s != sym_a)
            .collect();
        if payload_pool.is_empty() {
            return None;
        }

        // SEED : une ou deux racines `a -(0)-> x` (éventuellement prolongées par `-(0)-> tail`).
        let max_k = (self.cfg.max_effect_nodes / 3).clamp(1, 3);
        let k = self.rng.range_usize(1, max_k);
        let mut prev_tail: Option<NodeId> = None;

        for _ in 0..k {
            let payload = *self.rng.pick(&payload_pool);
            let na = b.add_seed_conc(sym_a)?;
            let nx = b.add_seed_conc(payload)?;
            b.connect_struct(na, nx, 0)?;
            if let Some(pt) = prev_tail {
                b.connect_struct(pt, na, 1)?;
            }
            prev_tail = Some(nx);
        }

        let rule = b.add_rule()?;
        let la = b.add_rule_conc(rule, true, sym_a)?;
        let lx = b.add_rule_var(rule, true)?;
        b.connect_struct(la, lx, 0)?;

        let ra = b.add_rule_conc(rule, false, sym_a)?;
        let rx1 = b.add_rule_var(rule, false)?;
        let rx2 = b.add_rule_var(rule, false)?;
        b.connect_struct(ra, rx1, 1)?;
        b.connect_struct(ra, rx2, 1)?;
        b.connect_struct(rx1, rx2, 0)?;

        b.connect_same_as(lx, rx1)?;
        b.connect_same_as(rx1, rx2)?;
        Some(b.finish())
    }

    /// Famille 6 : Chaîne causale séquentielle à 2 règles (R1 produit le redex de R2).
    /// SEED = a -> a -> x ; R1 : a -> a => a -> b -> c ; R2 : b -> c -> X => c -> X -> b -> X
    /// Effet final : a -> c -> x -> b -> x (2 étapes, auto-spécifié, R1 et R2 strictement nécessaires).
    fn build_chained_rules(&mut self) -> Option<HypothesisGraph> {
        if self.cfg.alphabet_pool.len() < 3 || self.cfg.max_effect_nodes < 5 {
            return None;
        }
        let mut b = GraphBuilder::new(self.internal_graph_cfg());
        let base = self.rng.pick_distinct(&self.cfg.alphabet_pool, 3)?;
        let (sym_a, sym_b, sym_c) = (base[0], base[1], base[2]);

        // Charge utile X différente de `a` et `c` pour garantir l'arrêt exact après R2.
        let payload_pool: Vec<SymbolId> = self
            .cfg
            .alphabet_pool
            .iter()
            .copied()
            .filter(|&s| s != sym_a && s != sym_c)
            .collect();
        let payload = *self.rng.pick(&payload_pool);

        let max_k = (self.cfg.max_effect_nodes / 5).clamp(1, 2);
        let k = self.rng.range_usize(1, max_k);

        let mut seed_ids = Vec::with_capacity(3 * k);
        for _ in 0..k {
            seed_ids.push(b.add_seed_conc(sym_a)?);
            seed_ids.push(b.add_seed_conc(sym_a)?);
            seed_ids.push(b.add_seed_conc(payload)?);
        }
        b.connect_chain(&seed_ids, 0)?;

        // Règle 1 : a -> a => a -> b -> c
        let r1 = b.add_rule()?;
        let r1_l0 = b.add_rule_conc(r1, true, sym_a)?;
        let r1_l1 = b.add_rule_conc(r1, true, sym_a)?;
        b.connect_struct(r1_l0, r1_l1, 0)?;

        let r1_r0 = b.add_rule_conc(r1, false, sym_a)?;
        let r1_r1 = b.add_rule_conc(r1, false, sym_b)?;
        let r1_r2 = b.add_rule_conc(r1, false, sym_c)?;
        b.connect_chain(&[r1_r0, r1_r1, r1_r2], 0)?;

        // Règle 2 : b -> c -> X => c -> X -> b -> X
        let r2 = b.add_rule()?;
        let r2_l0 = b.add_rule_conc(r2, true, sym_b)?;
        let r2_l1 = b.add_rule_conc(r2, true, sym_c)?;
        let r2_lx = b.add_rule_var(r2, true)?;
        b.connect_chain(&[r2_l0, r2_l1, r2_lx], 0)?;

        let r2_r0 = b.add_rule_conc(r2, false, sym_c)?;
        let r2_rx1 = b.add_rule_var(r2, false)?;
        let r2_r2 = b.add_rule_conc(r2, false, sym_b)?;
        let r2_rx2 = b.add_rule_var(r2, false)?;
        b.connect_chain(&[r2_r0, r2_rx1, r2_r2, r2_rx2], 0)?;

        b.connect_same_as(r2_lx, r2_rx1)?;
        b.connect_same_as(r2_rx1, r2_rx2)?;
        Some(b.finish())
    }

    /// Famille 7 : Échantillonneur libre (germe DAG aléatoire + 1 à 2 règles aléatoires).
    fn build_free_random(&mut self) -> Option<HypothesisGraph> {
        let mut b = GraphBuilder::new(self.internal_graph_cfg());
        let sub_size = self.cfg.alphabet_pool.len().min(4);
        let local_alpha = self.rng.pick_distinct(&self.cfg.alphabet_pool, sub_size)?;

        // 1. Germe aléatoire connexe de 2 à 5 nœuds (éventuellement multi-relations).
        let seed_len = self.rng.range_usize(2, 5);
        let mut seed_ids = Vec::with_capacity(seed_len);
        for _ in 0..seed_len {
            let sym = *self.rng.pick(&local_alpha);
            seed_ids.push(b.add_seed_conc(sym)?);
        }
        // Arbre couvrant dirigé + 0 à 2 arêtes supplémentaires.
        for i in 1..seed_len {
            let parent = if self.rng.next_bool(2, 3) {
                i - 1
            } else {
                self.rng.range_usize(0, i - 1)
            };
            let rel = self.rng.range_usize(0, usize::from(self.cfg.num_struct_rels - 1)) as RelId;
            b.connect_struct(seed_ids[parent], seed_ids[i], rel)?;
        }
        if seed_len >= 3 && self.rng.next_bool(1, 3) {
            let u = self.rng.range_usize(0, seed_len - 2);
            let v = self.rng.range_usize(u + 1, seed_len - 1);
            let rel = self.rng.range_usize(0, usize::from(self.cfg.num_struct_rels - 1)) as RelId;
            let _ = b.try_connect_struct(seed_ids[u], seed_ids[v], rel);
        }

        // 2. Ajout de 1 ou 2 règles aléatoires exécutables (avec ou sans VAR/SAME_AS).
        let num_rules = if self.rng.next_bool(3, 4) { 1 } else { 2 };
        for _ in 0..num_rules {
            let rule = b.add_rule()?;
            let lhs_len = self.rng.range_usize(1, 3);
            let rhs_len = self.rng.range_usize(1, 4);

            let mut lhs_ids = Vec::with_capacity(lhs_len);
            let mut lhs_vars = Vec::new();
            for i in 0..lhs_len {
                // Garde au moins un nœud concret en LHS pour limiter les boucles triviales.
                let make_var = i > 0 && self.rng.next_bool(1, 2);
                if make_var {
                    let v = b.add_rule_var(rule, true)?;
                    lhs_ids.push(v);
                    lhs_vars.push(v);
                } else {
                    let sym = *self.rng.pick(&local_alpha);
                    lhs_ids.push(b.add_rule_conc(rule, true, sym)?);
                }
            }
            for i in 1..lhs_len {
                let rel =
                    self.rng.range_usize(0, usize::from(self.cfg.num_struct_rels - 1)) as RelId;
                b.connect_struct(lhs_ids[i - 1], lhs_ids[i], rel)?;
            }

            let mut rhs_ids = Vec::with_capacity(rhs_len);
            for _ in 0..rhs_len {
                let make_var = !lhs_vars.is_empty() && self.rng.next_bool(1, 2);
                if make_var {
                    let rv = b.add_rule_var(rule, false)?;
                    let anchor = *self.rng.pick(&lhs_vars);
                    b.connect_same_as(anchor, rv)?;
                    rhs_ids.push(rv);
                } else {
                    let sym = *self.rng.pick(&local_alpha);
                    rhs_ids.push(b.add_rule_conc(rule, false, sym)?);
                }
            }
            for i in 1..rhs_len {
                let parent = if self.rng.next_bool(3, 4) {
                    i - 1
                } else {
                    self.rng.range_usize(0, i - 1)
                };
                let rel =
                    self.rng.range_usize(0, usize::from(self.cfg.num_struct_rels - 1)) as RelId;
                b.connect_struct(rhs_ids[parent], rhs_ids[i], rel)?;
            }
        }

        Some(b.finish())
    }
}

// ============================================================================
// 6. Utilitaires canoniques (symboles, reconstruction de config, copie H = E)
// ============================================================================

fn collect_concrete_symbols(h: &HypothesisGraph) -> BTreeSet<SymbolId> {
    let mut set = BTreeSet::new();
    for &kind in h.nodes() {
        if let NodeKind::Conc(sym) = kind {
            set.insert(sym);
        }
    }
    set
}

/// Reconstruit exactement un `HypothesisGraph` dans une nouvelle `GraphConfig` cible.
fn rebuild_with_config(h: &HypothesisGraph, target_cfg: GraphConfig) -> Option<HypothesisGraph> {
    let fallback = target_cfg.alphabet[0];
    let mut out = HypothesisGraph::new(target_cfg);

    for v in 1..h.node_count() {
        match h.nodes()[v] {
            NodeKind::Rule => {
                out = out
                    .apply_forward(&ForwardAction::AddNode(AddNodeKind::Rule))
                    .ok()?;
            }
            NodeKind::Conc(sym) => {
                out = out
                    .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(sym)))
                    .ok()?;
            }
            NodeKind::Var => {
                out = out
                    .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(fallback)))
                    .ok()?
                    .apply_forward(&ForwardAction::Abstract(v))
                    .ok()?;
            }
            NodeKind::Seed => return None,
        }
    }

    for &e in h.edges() {
        if e.kind.is_membership() {
            out = out.apply_forward(&ForwardAction::Connect(e)).ok()?;
        }
    }
    for &e in h.edges() {
        if !e.kind.is_membership() {
            out = out.apply_forward(&ForwardAction::Connect(e)).ok()?;
        }
    }
    Some(out)
}

/// Construit la copie triviale H = E (mémorisation complète de `obs` dans SEED, 0 règle).
pub fn build_verbatim_copy(obs: &ObservationGraph, cfg: GraphConfig) -> Option<HypothesisGraph> {
    let mut b = GraphBuilder::new(cfg);
    let mut ids = Vec::with_capacity(obs.node_count());
    for &sym in &obs.nodes {
        ids.push(b.add_seed_conc(sym)?);
    }
    for &e in &obs.edges {
        if e.src != e.dst {
            b.connect_struct(ids[e.src], ids[e.dst], e.rel)?;
        }
    }
    Some(b.finish())
}

// ============================================================================
// Tests unitaires complets
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_seven_archetypes_produce_valid_necessary_causes() {
        let cfg = SynthConfig::default();
        let mut synth = PtgrsSynthesizer::new(cfg, 0x5048_4953_0001).unwrap();

        let all = [
            RuleArchetype::ConcreteExpansion,
            RuleArchetype::AbstractDuplication,
            RuleArchetype::AbstractSwap,
            RuleArchetype::AbstractBracketing,
            RuleArchetype::MultiRelTransform,
            RuleArchetype::ChainedRules,
            RuleArchetype::FreeRandom,
        ];

        for archetype in all {
            let sample = synth
                .sample_archetype(archetype)
                .unwrap_or_else(|| panic!("Échec de synthèse sur {archetype:?}"));

            // 1. Invariants DAG stricts et configuration alignée.
            sample
                .hidden_cause
                .check_invariants()
                .expect("Invariants DAG violés sur H_true");
            assert_eq!(
                sample.hidden_cause.config(),
                &sample.training.recommended_graph_cfg
            );

            // 2. Auto-spécification stricte : alphabet de H_true == alphabet de E.
            assert_eq!(
                sample.training.recommended_graph_cfg.alphabet,
                sample.training.effect.alphabet()
            );

            // 3. Rejeu exact par le projecteur Pi : résidu nul et forme normale.
            let replay = project(
                &sample.hidden_cause,
                synth.config().refutation_cfg.projector_cfg,
            );
            assert_eq!(replay.termination, TerminationReason::NormalForm);
            assert_eq!(replay.predicted, sample.training.effect);
            assert!(sample.rewrite_steps >= 1);
            assert!(sample.hidden_report.residual.is_zero());

            // 4. Aucune règle parasite ou malformée.
            assert!(sample.hidden_report.necessary_rule_count() >= 1);
            assert_eq!(sample.hidden_report.parasitic_rule_count(), 0);

            // Vérifications spécifiques aux familles abstraites, multi-relations et chaînées.
            match archetype {
                RuleArchetype::AbstractDuplication
                | RuleArchetype::AbstractSwap
                | RuleArchetype::AbstractBracketing => {
                    assert!(
                        sample.hidden_cause.var_count() >= 2,
                        "VAR attendus pour {archetype:?}"
                    );
                    assert!(
                        sample
                            .hidden_cause
                            .edges()
                            .iter()
                            .any(|e| e.kind == EdgeKind::SameAs),
                        "SAME_AS attendu pour {archetype:?}"
                    );
                }
                RuleArchetype::MultiRelTransform => {
                    let rels: BTreeSet<RelId> = sample
                        .training
                        .effect
                        .edges
                        .iter()
                        .map(|e| e.rel)
                        .collect();
                    assert!(
                        rels.len() >= 2,
                        "Effet multi-relations attendu, obtenu : {rels:?}"
                    );
                }
                RuleArchetype::ChainedRules => {
                    assert_eq!(sample.hidden_cause.rule_count(), 2);
                    assert!(sample.rewrite_steps >= 2);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_determinism_and_training_isolation() {
        let cfg = SynthConfig::default();
        let mut s1 = PtgrsSynthesizer::new(cfg.clone(), 42).unwrap();
        let mut s2 = PtgrsSynthesizer::new(cfg, 42).unwrap();

        let batch1 = s1.next_training_batch(10);
        let batch2 = s2.next_training_batch(10);

        assert_eq!(batch1.len(), 10);
        assert_eq!(batch2.len(), 10);
        for (a, b) in batch1.iter().zip(batch2.iter()) {
            assert_eq!(a.sample_id, b.sample_id);
            assert_eq!(a.effect, b.effect);
            assert_eq!(a.recommended_graph_cfg, b.recommended_graph_cfg);
        }
    }

    #[test]
    fn test_curriculum_levels_and_beats_verbatim_mode() {
        // 1. Curriculum : difficulté 0 ne produit ni ChainedRules ni FreeRandom.
        let cfg0 = SynthConfig {
            difficulty: 0,
            ..SynthConfig::default()
        };
        let mut synth0 = PtgrsSynthesizer::new(cfg0, 99).unwrap();
        let eval0 = synth0.next_eval_batch(12);
        assert_eq!(eval0.len(), 12);
        for e in &eval0 {
            assert!(matches!(
                e.archetype,
                RuleArchetype::ConcreteExpansion | RuleArchetype::AbstractDuplication
            ));
        }

        // Passage dynamique à difficulté 3.
        synth0.set_difficulty(3);
        let next_s = synth0.next_training_sample().unwrap();
        assert_eq!(next_s.difficulty, 3);

        // 2. Mode exigeant que H_true batte strictement la copie triviale H = E (phi_sharp).
        let cfg_strict = SynthConfig {
            min_effect_nodes: 8,
            max_effect_nodes: 16,
            require_beats_verbatim: true,
            ..SynthConfig::default()
        };
        let mut synth_strict = PtgrsSynthesizer::new(cfg_strict, 777).unwrap();
        let sample = synth_strict
            .sample_archetype(RuleArchetype::ConcreteExpansion)
            .expect("Doit produire une expansion auto-spécifiée qui bat H = E");

        let verbatim = build_verbatim_copy(
            &sample.training.effect,
            sample.training.recommended_graph_cfg.clone(),
        )
        .unwrap();
        let rep_verbatim = evaluate_phi_sharp(
            &verbatim,
            &sample.training.effect,
            synth_strict.config().refutation_cfg,
        );
        assert!(
            sample.hidden_report.log_phi_sharp > rep_verbatim.log_phi_sharp,
            "H_true ({}) doit battre H = E ({})",
            sample.hidden_report.log_phi_sharp,
            rep_verbatim.log_phi_sharp
        );
    }

    #[test]
    fn test_latent_marker_mode_when_self_specified_disabled() {
        let cfg = SynthConfig {
            require_self_specified: false,
            require_beats_verbatim: true,
            min_effect_nodes: 9,
            max_effect_nodes: 12,
            ..SynthConfig::default()
        };
        let mut synth = PtgrsSynthesizer::new(cfg, 12345).unwrap();
        let sample = synth
            .sample_archetype(RuleArchetype::ConcreteExpansion)
            .expect("Mode marqueur latent doit réussir");

        // Le marqueur m est absent de G_E mais inclus dans recommended_graph_cfg.
        assert!(
            sample.training.recommended_graph_cfg.alphabet.len()
                > sample.training.effect.alphabet().len()
        );
        assert!(sample.hidden_report.residual.is_zero());
    }
}
