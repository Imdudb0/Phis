//! Phisharp V1 — Moteur de graphe d'hypothèses PTGRS compatible GFlowNet.
//!
//! Compilation et tests (Rust 2024, zéro dépendance externe) :
//!   rustc --edition 2024 --test hypothesis_graph.rs && ./hypothesis_graph

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

pub type NodeId = usize;
pub type SymbolId = u8;
pub type RelId = u8;

/// Configuration bornée du DAG d'hypothèses pour un fichier `perceived.eff`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphConfig {
    /// Alphabet Sigma_E trié et dédupliqué (octets observés dans E).
    pub alphabet: Vec<SymbolId>,
    /// Nombre de relations structurelles internes `Struct(0..num_struct_rels)`.
    pub num_struct_rels: u8,
    /// Borne supérieure V_max (inclut le nœud SEED fixe).
    pub max_nodes: usize,
    /// Borne supérieure R_max sur le nombre de hubs RULE.
    pub max_rules: usize,
    /// Borne supérieure E_max sur le nombre total d'arêtes.
    pub max_edges: usize,
}

impl GraphConfig {
    pub fn new(
        mut alphabet: Vec<SymbolId>,
        num_struct_rels: u8,
        max_nodes: usize,
        max_rules: usize,
        max_edges: usize,
    ) -> Result<Self, &'static str> {
        alphabet.sort_unstable();
        alphabet.dedup();
        if alphabet.is_empty() {
            return Err("L'alphabet Sigma_E ne peut pas être vide");
        }
        if max_nodes < 2 {
            return Err("max_nodes doit être >= 2 (SEED + au moins un nœud)");
        }
        Ok(Self {
            alphabet,
            num_struct_rels,
            max_nodes,
            max_rules,
            max_edges,
        })
    }

    #[inline]
    pub fn has_symbol(&self, sym: SymbolId) -> bool {
        self.alphabet.binary_search(&sym).is_ok()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeKind {
    Seed,
    Rule,
    Conc(SymbolId),
    Var,
}

impl NodeKind {
    fn tag(self) -> u32 {
        match self {
            NodeKind::Seed => 0,
            NodeKind::Rule => 1,
            NodeKind::Var => 2,
            NodeKind::Conc(c) => 3 + u32::from(c),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EdgeKind {
    InSeed,
    InLhs,
    InRhs,
    Struct(RelId),
    SameAs,
}

impl EdgeKind {
    fn tag(self) -> u32 {
        match self {
            EdgeKind::InSeed => 0,
            EdgeKind::InLhs => 1,
            EdgeKind::InRhs => 2,
            EdgeKind::SameAs => 3,
            EdgeKind::Struct(r) => 4 + u32::from(r),
        }
    }

    #[inline]
    pub fn is_membership(self) -> bool {
        matches!(self, EdgeKind::InSeed | EdgeKind::InLhs | EdgeKind::InRhs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge {
    pub src: NodeId,
    pub dst: NodeId,
    pub kind: EdgeKind,
}

impl Edge {
    /// Construit une arête canonique (`src < dst` pour `SameAs`).
    #[inline]
    pub fn new(src: NodeId, dst: NodeId, kind: EdgeKind) -> Self {
        if kind == EdgeKind::SameAs && src > dst {
            Self {
                src: dst,
                dst: src,
                kind,
            }
        } else {
            Self { src, dst, kind }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Block {
    Seed,
    RuleLhs(NodeId),
    RuleRhs(NodeId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AddNodeKind {
    Rule,
    Conc(SymbolId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ForwardAction {
    AddNode(AddNodeKind),
    Connect(Edge),
    Abstract(NodeId),
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackwardAction {
    DelNode(NodeId),
    DelEdge(Edge),
    UnAbstract { node: NodeId, symbol: SymbolId },
    UnStop,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionError {
    IllegalForwardAction(ForwardAction),
    IllegalBackwardAction(BackwardAction),
}

impl fmt::Display for ActionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActionError::IllegalForwardAction(a) => write!(f, "Action P_F illégale : {a:?}"),
            ActionError::IllegalBackwardAction(a) => write!(f, "Action P_B illégale : {a:?}"),
        }
    }
}

/// Masques factorisés prêts pour les têtes neuronales de P_F.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForwardMask {
    /// Ordre : `[ADD_NODE, CONNECT, ABSTRACT, STOP]`.
    pub family_mask: [bool; 4],
    pub allow_add_rule: bool,
    /// Aligné sur `config.alphabet`.
    pub allow_add_conc: Vec<bool>,
    /// Ensemble exact des arêtes candidates légales pour la tête bilinéaire.
    pub connect_edges: Vec<Edge>,
    /// Aligné sur `0..graph.node_count()`.
    pub abstract_nodes: Vec<bool>,
    pub allow_stop: bool,
}

impl ForwardMask {
    pub fn to_actions(&self, alphabet: &[SymbolId]) -> Vec<ForwardAction> {
        let mut actions = Vec::new();
        if self.allow_add_rule {
            actions.push(ForwardAction::AddNode(AddNodeKind::Rule));
        }
        for (&allowed, &sym) in self.allow_add_conc.iter().zip(alphabet.iter()) {
            if allowed {
                actions.push(ForwardAction::AddNode(AddNodeKind::Conc(sym)));
            }
        }
        for &edge in &self.connect_edges {
            actions.push(ForwardAction::Connect(edge));
        }
        for (node, &allowed) in self.abstract_nodes.iter().enumerate() {
            if allowed {
                actions.push(ForwardAction::Abstract(node));
            }
        }
        if self.allow_stop {
            actions.push(ForwardAction::Stop);
        }
        actions
    }
}

/// Masques factorisés prêts pour les têtes neuronales de P_B.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackwardMask {
    /// Ordre : `[DEL_NODE, DEL_EDGE, UN_ABSTRACT, UN_STOP]`.
    pub family_mask: [bool; 4],
    /// Aligné sur `0..graph.node_count()`.
    pub del_nodes: Vec<bool>,
    /// Aligné sur l'ordre déterministe de `graph.edges()`.
    pub del_edges: Vec<bool>,
    /// Aligné sur `0..graph.node_count()` (chaque nœud vrai accepte tout `c` de `alphabet`).
    pub un_abstract_nodes: Vec<bool>,
    pub allow_un_stop: bool,
}

impl BackwardMask {
    pub fn to_actions(&self, edges: &BTreeSet<Edge>, alphabet: &[SymbolId]) -> Vec<BackwardAction> {
        let mut actions = Vec::new();
        for (node, &allowed) in self.del_nodes.iter().enumerate() {
            if allowed {
                actions.push(BackwardAction::DelNode(node));
            }
        }
        for (&edge, &allowed) in edges.iter().zip(self.del_edges.iter()) {
            if allowed {
                actions.push(BackwardAction::DelEdge(edge));
            }
        }
        for (node, &allowed) in self.un_abstract_nodes.iter().enumerate() {
            if allowed {
                for &symbol in alphabet {
                    actions.push(BackwardAction::UnAbstract { node, symbol });
                }
            }
        }
        if self.allow_un_stop {
            actions.push(BackwardAction::UnStop);
        }
        actions
    }
}

/// Clé canonique invariante par isomorphisme de graphe (pour cache, nogoods et SubTB).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalKey {
    pub stopped: bool,
    pub nodes: Vec<NodeKind>,
    pub edges: Vec<Edge>,
}

/// Graphe d'hypothèse PTGRS H_t de Phisharp V1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HypothesisGraph {
    config: GraphConfig,
    nodes: Vec<NodeKind>,
    edges: BTreeSet<Edge>,
    stopped: bool,
}

impl HypothesisGraph {
    /// Crée l'état initial minimal H_0 = ({SEED}, ∅).
    pub fn new(config: GraphConfig) -> Self {
        Self {
            config,
            nodes: vec![NodeKind::Seed],
            edges: BTreeSet::new(),
            stopped: false,
        }
    }

    #[inline]
    pub fn config(&self) -> &GraphConfig {
        &self.config
    }

    #[inline]
    pub fn nodes(&self) -> &[NodeKind] {
        &self.nodes
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn edges(&self) -> &BTreeSet<Edge> {
        &self.edges
    }

    #[inline]
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    #[inline]
    pub fn is_initial(&self) -> bool {
        !self.stopped && self.nodes.len() == 1 && self.edges.is_empty()
    }

    pub fn rule_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|&&k| k == NodeKind::Rule)
            .count()
    }

    pub fn var_count(&self) -> usize {
        self.nodes.iter().filter(|&&k| k == NodeKind::Var).count()
    }

    /// Fonction de potentiel strictement croissante S(H_t) = |V| + |E| + |V_var| + 1_{stopped}.
    /// Chaque transition P_F augmente S de +1 exactement (garantie DAG).
    pub fn potential(&self) -> usize {
        self.nodes.len() + self.edges.len() + self.var_count() + usize::from(self.stopped)
    }

    /// Retourne le bloc unique d'appartenance d'un nœud (`SEED`, `LHS(r)` ou `RHS(r)`).
    pub fn membership(&self, node: NodeId) -> Option<Block> {
        for edge in &self.edges {
            if edge.dst == node {
                match edge.kind {
                    EdgeKind::InSeed => return Some(Block::Seed),
                    EdgeKind::InLhs => return Some(Block::RuleLhs(edge.src)),
                    EdgeKind::InRhs => return Some(Block::RuleRhs(edge.src)),
                    EdgeKind::Struct(_) | EdgeKind::SameAs => {}
                }
            }
        }
        None
    }

    /// Retourne l'identifiant du hub `RULE` propriétaire si `node` est dans un `LHS` ou `RHS`.
    pub fn owner_rule(&self, node: NodeId) -> Option<NodeId> {
        match self.membership(node) {
            Some(Block::RuleLhs(r) | Block::RuleRhs(r)) => Some(r),
            _ => None,
        }
    }

    pub fn total_degree(&self, node: NodeId) -> usize {
        self.edges
            .iter()
            .filter(|e| e.src == node || e.dst == node)
            .count()
    }

    pub fn struct_degree(&self, node: NodeId) -> usize {
        self.edges
            .iter()
            .filter(|e| matches!(e.kind, EdgeKind::Struct(_)) && (e.src == node || e.dst == node))
            .count()
    }

    pub fn same_as_degree(&self, node: NodeId) -> usize {
        self.edges
            .iter()
            .filter(|e| e.kind == EdgeKind::SameAs && (e.src == node || e.dst == node))
            .count()
    }

    /// Vérifie les préconditions strictes d'une action avant P_F.
    pub fn is_legal_forward(&self, action: &ForwardAction) -> bool {
        if self.stopped {
            return false;
        }
        match *action {
            ForwardAction::AddNode(AddNodeKind::Rule) => {
                self.nodes.len() < self.config.max_nodes
                    && self.rule_count() < self.config.max_rules
            }
            ForwardAction::AddNode(AddNodeKind::Conc(sym)) => {
                self.nodes.len() < self.config.max_nodes && self.config.has_symbol(sym)
            }
            ForwardAction::Connect(edge) => self.is_legal_connect(edge),
            ForwardAction::Abstract(node) => {
                node < self.nodes.len()
                    && matches!(self.nodes[node], NodeKind::Conc(_))
                    && self.membership(node) != Some(Block::Seed)
            }
            ForwardAction::Stop => !self.is_initial(),
        }
    }

    fn is_legal_connect(&self, edge: Edge) -> bool {
        if self.edges.len() >= self.config.max_edges {
            return false;
        }
        if edge.src >= self.nodes.len() || edge.dst >= self.nodes.len() {
            return false;
        }
        if edge.kind == EdgeKind::SameAs && edge.src >= edge.dst {
            return false;
        }
        if self.edges.contains(&edge) {
            return false;
        }

        match edge.kind {
            EdgeKind::InSeed => {
                edge.src == 0
                    && matches!(self.nodes[edge.dst], NodeKind::Conc(_))
                    && self.membership(edge.dst).is_none()
            }
            EdgeKind::InLhs | EdgeKind::InRhs => {
                self.nodes[edge.src] == NodeKind::Rule
                    && matches!(self.nodes[edge.dst], NodeKind::Conc(_) | NodeKind::Var)
                    && self.membership(edge.dst).is_none()
            }
            EdgeKind::Struct(rel) => {
                rel < self.config.num_struct_rels
                    && edge.src != edge.dst
                    && matches!(self.nodes[edge.src], NodeKind::Conc(_) | NodeKind::Var)
                    && matches!(self.nodes[edge.dst], NodeKind::Conc(_) | NodeKind::Var)
                    && matches!(
                        (self.membership(edge.src), self.membership(edge.dst)),
                        (Some(b1), Some(b2)) if b1 == b2
                    )
            }
            EdgeKind::SameAs => {
                edge.src < edge.dst
                    && self.nodes[edge.src] == NodeKind::Var
                    && self.nodes[edge.dst] == NodeKind::Var
                    && matches!(
                        (self.owner_rule(edge.src), self.owner_rule(edge.dst)),
                        (Some(r1), Some(r2)) if r1 == r2
                    )
            }
        }
    }

    /// Vérifie les préconditions strictes d'une action arrière P_B (bijection exacte).
    pub fn is_legal_backward(&self, action: &BackwardAction) -> bool {
        if self.stopped {
            return matches!(action, BackwardAction::UnStop);
        }
        match *action {
            BackwardAction::UnStop => false,
            BackwardAction::DelNode(node) => {
                node > 0
                    && node < self.nodes.len()
                    && matches!(self.nodes[node], NodeKind::Rule | NodeKind::Conc(_))
                    && self.total_degree(node) == 0
            }
            BackwardAction::DelEdge(edge) => {
                if !self.edges.contains(&edge) {
                    return false;
                }
                if edge.kind.is_membership() {
                    self.struct_degree(edge.dst) == 0 && self.same_as_degree(edge.dst) == 0
                } else {
                    true
                }
            }
            BackwardAction::UnAbstract { node, symbol } => {
                node < self.nodes.len()
                    && self.nodes[node] == NodeKind::Var
                    && self.same_as_degree(node) == 0
                    && self.config.has_symbol(symbol)
            }
        }
    }

    /// Calcule les masques factorisés complets pour P_F.
    pub fn forward_mask(&self) -> ForwardMask {
        let n = self.nodes.len();
        if self.stopped {
            return ForwardMask {
                family_mask: [false; 4],
                allow_add_rule: false,
                allow_add_conc: vec![false; self.config.alphabet.len()],
                connect_edges: Vec::new(),
                abstract_nodes: vec![false; n],
                allow_stop: false,
            };
        }

        let can_add_node = n < self.config.max_nodes;
        let allow_add_rule = can_add_node && self.rule_count() < self.config.max_rules;
        let allow_add_conc = vec![can_add_node; self.config.alphabet.len()];

        let mut connect_edges = Vec::new();
        if self.edges.len() < self.config.max_edges {
            // Pré-calcul des appartenances et règles propriétaires pour O(V^2).
            let memberships: Vec<Option<Block>> = (0..n).map(|v| self.membership(v)).collect();
            let owners: Vec<Option<NodeId>> = memberships
                .iter()
                .map(|m| match *m {
                    Some(Block::RuleLhs(r) | Block::RuleRhs(r)) => Some(r),
                    _ => None,
                })
                .collect();

            // 1. IN_SEED
            for (dst, &kind) in self.nodes.iter().enumerate().skip(1) {
                if matches!(kind, NodeKind::Conc(_)) && memberships[dst].is_none() {
                    let e = Edge::new(0, dst, EdgeKind::InSeed);
                    if !self.edges.contains(&e) {
                        connect_edges.push(e);
                    }
                }
            }

            // 2. IN_LHS et IN_RHS
            for (rule, &rk) in self.nodes.iter().enumerate().skip(1) {
                if rk != NodeKind::Rule {
                    continue;
                }
                for (dst, &dk) in self.nodes.iter().enumerate().skip(1) {
                    if matches!(dk, NodeKind::Conc(_) | NodeKind::Var) && memberships[dst].is_none()
                    {
                        for kind in [EdgeKind::InLhs, EdgeKind::InRhs] {
                            let e = Edge::new(rule, dst, kind);
                            if !self.edges.contains(&e) {
                                connect_edges.push(e);
                            }
                        }
                    }
                }
            }

            // 3. STRUCT(rel) dans un même bloc
            for u in 1..n {
                let Some(bu) = memberships[u] else {
                    continue;
                };
                for v in 1..n {
                    if u == v || memberships[v] != Some(bu) {
                        continue;
                    }
                    for rel in 0..self.config.num_struct_rels {
                        let e = Edge::new(u, v, EdgeKind::Struct(rel));
                        if !self.edges.contains(&e) {
                            connect_edges.push(e);
                        }
                    }
                }
            }

            // 4. SAME_AS entre deux VAR d'une même règle
            for u in 1..n {
                if self.nodes[u] != NodeKind::Var {
                    continue;
                }
                let Some(ru) = owners[u] else {
                    continue;
                };
                for v in (u + 1)..n {
                    if self.nodes[v] == NodeKind::Var && owners[v] == Some(ru) {
                        let e = Edge::new(u, v, EdgeKind::SameAs);
                        if !self.edges.contains(&e) {
                            connect_edges.push(e);
                        }
                    }
                }
            }
        }

        let abstract_nodes: Vec<bool> = (0..n)
            .map(|v| {
                matches!(self.nodes[v], NodeKind::Conc(_)) && self.membership(v) != Some(Block::Seed)
            })
            .collect();

        let allow_stop = !self.is_initial();

        let family_mask = [
            allow_add_rule || allow_add_conc.iter().any(|&b| b),
            !connect_edges.is_empty(),
            abstract_nodes.iter().any(|&b| b),
            allow_stop,
        ];

        ForwardMask {
            family_mask,
            allow_add_rule,
            allow_add_conc,
            connect_edges,
            abstract_nodes,
            allow_stop,
        }
    }

    /// Calcule les masques factorisés complets pour P_B.
    pub fn backward_mask(&self) -> BackwardMask {
        let n = self.nodes.len();
        if self.stopped {
            return BackwardMask {
                family_mask: [false, false, false, true],
                del_nodes: vec![false; n],
                del_edges: vec![false; self.edges.len()],
                un_abstract_nodes: vec![false; n],
                allow_un_stop: true,
            };
        }

        let del_nodes: Vec<bool> = (0..n)
            .map(|v| {
                v > 0
                    && matches!(self.nodes[v], NodeKind::Rule | NodeKind::Conc(_))
                    && self.total_degree(v) == 0
            })
            .collect();

        let del_edges: Vec<bool> = self
            .edges
            .iter()
            .map(|e| {
                if e.kind.is_membership() {
                    self.struct_degree(e.dst) == 0 && self.same_as_degree(e.dst) == 0
                } else {
                    true
                }
            })
            .collect();

        let un_abstract_nodes: Vec<bool> = (0..n)
            .map(|v| self.nodes[v] == NodeKind::Var && self.same_as_degree(v) == 0)
            .collect();

        let family_mask = [
            del_nodes.iter().any(|&b| b),
            del_edges.iter().any(|&b| b),
            un_abstract_nodes.iter().any(|&b| b),
            false,
        ];

        BackwardMask {
            family_mask,
            del_nodes,
            del_edges,
            un_abstract_nodes,
            allow_un_stop: false,
        }
    }

    pub fn legal_forward_actions(&self) -> Vec<ForwardAction> {
        self.forward_mask().to_actions(&self.config.alphabet)
    }

    pub fn legal_backward_actions(&self) -> Vec<BackwardAction> {
        self.backward_mask()
            .to_actions(&self.edges, &self.config.alphabet)
    }

    /// Applique une action avant P_F (immuable, retourne le nouvel état).
    pub fn apply_forward(&self, action: &ForwardAction) -> Result<Self, ActionError> {
        if !self.is_legal_forward(action) {
            return Err(ActionError::IllegalForwardAction(*action));
        }
        let mut next = self.clone();
        match *action {
            ForwardAction::AddNode(AddNodeKind::Rule) => next.nodes.push(NodeKind::Rule),
            ForwardAction::AddNode(AddNodeKind::Conc(sym)) => next.nodes.push(NodeKind::Conc(sym)),
            ForwardAction::Connect(edge) => {
                next.edges.insert(edge);
            }
            ForwardAction::Abstract(node) => {
                next.nodes[node] = NodeKind::Var;
            }
            ForwardAction::Stop => {
                next.stopped = true;
            }
        }
        Ok(next)
    }

    /// Applique une action arrière P_B (immuable, retourne l'état parent).
    pub fn apply_backward(&self, action: &BackwardAction) -> Result<Self, ActionError> {
        if !self.is_legal_backward(action) {
            return Err(ActionError::IllegalBackwardAction(*action));
        }
        let mut next = self.clone();
        match *action {
            BackwardAction::UnStop => {
                next.stopped = false;
            }
            BackwardAction::DelNode(node) => {
                next.nodes.remove(node);
                let remap = |id: NodeId| if id > node { id - 1 } else { id };
                next.edges = next
                    .edges
                    .into_iter()
                    .map(|e| Edge::new(remap(e.src), remap(e.dst), e.kind))
                    .collect();
            }
            BackwardAction::DelEdge(edge) => {
                next.edges.remove(&edge);
            }
            BackwardAction::UnAbstract { node, symbol } => {
                next.nodes[node] = NodeKind::Conc(symbol);
            }
        }
        Ok(next)
    }

    /// Vérifie tous les invariants structurels du graphe (utile en test/debug).
    pub fn check_invariants(&self) -> Result<(), String> {
        if self.nodes.is_empty() || self.nodes.len() > self.config.max_nodes {
            return Err(format!("Nombre de nœuds hors bornes : {}", self.nodes.len()));
        }
        if self.nodes[0] != NodeKind::Seed {
            return Err("Le nœud 0 doit toujours être SEED".into());
        }
        for (i, &k) in self.nodes.iter().enumerate().skip(1) {
            if k == NodeKind::Seed {
                return Err(format!("Nœud SEED dupliqué à l'indice {i}"));
            }
            if let NodeKind::Conc(c) = k {
                if !self.config.has_symbol(c) {
                    return Err(format!("Symbole {c} absent de l'alphabet"));
                }
            }
        }
        if self.rule_count() > self.config.max_rules {
            return Err("Nombre de règles > max_rules".into());
        }
        if self.edges.len() > self.config.max_edges {
            return Err("Nombre d'arêtes > max_edges".into());
        }
        if self.stopped && self.nodes.len() == 1 {
            return Err("L'état H_0 ne peut pas être dans l'état STOP".into());
        }

        // Chaque nœud possède au plus une arête d'appartenance.
        let mut membership_counts = vec![0usize; self.nodes.len()];
        for &edge in &self.edges {
            if edge.src >= self.nodes.len() || edge.dst >= self.nodes.len() {
                return Err(format!("Arête hors bornes : {edge:?}"));
            }
            if edge.kind.is_membership() {
                membership_counts[edge.dst] += 1;
            }
        }
        if let Some((v, &cnt)) = membership_counts.iter().enumerate().find(|(_, c)| **c > 1) {
            return Err(format!("Nœud {v} possède {cnt} arêtes d'appartenance"));
        }

        // Validation du typage de chaque arête.
        for &edge in &self.edges {
            match edge.kind {
                EdgeKind::InSeed => {
                    if edge.src != 0 || !matches!(self.nodes[edge.dst], NodeKind::Conc(_)) {
                        return Err(format!("Arête InSeed invalide : {edge:?}"));
                    }
                }
                EdgeKind::InLhs | EdgeKind::InRhs => {
                    if self.nodes[edge.src] != NodeKind::Rule
                        || !matches!(self.nodes[edge.dst], NodeKind::Conc(_) | NodeKind::Var)
                    {
                        return Err(format!("Arête InLhs/InRhs invalide : {edge:?}"));
                    }
                }
                EdgeKind::Struct(rel) => {
                    if rel >= self.config.num_struct_rels
                        || edge.src == edge.dst
                        || self.membership(edge.src).is_none()
                        || self.membership(edge.src) != self.membership(edge.dst)
                    {
                        return Err(format!("Arête Struct invalide : {edge:?}"));
                    }
                }
                EdgeKind::SameAs => {
                    if edge.src >= edge.dst
                        || self.nodes[edge.src] != NodeKind::Var
                        || self.nodes[edge.dst] != NodeKind::Var
                        || self.owner_rule(edge.src).is_none()
                        || self.owner_rule(edge.src) != self.owner_rule(edge.dst)
                    {
                        return Err(format!("Arête SameAs invalide : {edge:?}"));
                    }
                }
            }
        }
        Ok(())
    }

    /// Calcule une représentation canonique exacte (invariante par isomorphisme).
    pub fn canonical_key(&self) -> CanonicalKey {
        let n = self.nodes.len();
        let mut out_adj: Vec<Vec<(NodeId, u32)>> = vec![Vec::new(); n];
        let mut in_adj: Vec<Vec<(NodeId, u32)>> = vec![Vec::new(); n];

        for &edge in &self.edges {
            let tag = edge.kind.tag();
            out_adj[edge.src].push((edge.dst, tag));
            in_adj[edge.dst].push((edge.src, tag));
            if edge.kind == EdgeKind::SameAs {
                // Symétrisation pour le raffinement des arêtes non-orientées.
                out_adj[edge.dst].push((edge.src, tag));
                in_adj[edge.src].push((edge.dst, tag));
            }
        }
        for list in out_adj.iter_mut().chain(in_adj.iter_mut()) {
            list.sort_unstable();
        }

        // Coloration initiale basée uniquement sur NodeKind.
        let mut initial_tags: Vec<u32> = self.nodes.iter().map(|k| k.tag()).collect();
        initial_tags.sort_unstable();
        initial_tags.dedup();
        let colors: Vec<usize> = self
            .nodes
            .iter()
            .map(|k| initial_tags.binary_search(&k.tag()).unwrap())
            .collect();

        let mut best: Option<CanonicalKey> = None;
        self.canonical_search(colors, &out_adj, &in_adj, &mut best);
        best.expect("Le graphe possède au moins un étiquetage canonique")
    }

    fn refine_colors(
        colors: &mut [usize],
        out_adj: &[Vec<(NodeId, u32)>],
        in_adj: &[Vec<(NodeId, u32)>],
    ) {
        let n = colors.len();
        let mut prev_classes = 0usize;
        loop {
            let mut signatures: Vec<(usize, Vec<(u32, usize)>, Vec<(u32, usize)>)> =
                Vec::with_capacity(n);
            for u in 0..n {
                let mut out_sig: Vec<(u32, usize)> = out_adj[u]
                    .iter()
                    .map(|&(v, tag)| (tag, colors[v]))
                    .collect();
                out_sig.sort_unstable();

                let mut in_sig: Vec<(u32, usize)> = in_adj[u]
                    .iter()
                    .map(|&(v, tag)| (tag, colors[v]))
                    .collect();
                in_sig.sort_unstable();

                signatures.push((colors[u], out_sig, in_sig));
            }

            let mut unique = signatures.clone();
            unique.sort();
            unique.dedup();

            for (u, sig) in signatures.iter().enumerate() {
                colors[u] = unique.binary_search(sig).unwrap();
            }

            if unique.len() == prev_classes || unique.len() == n {
                break;
            }
            prev_classes = unique.len();
        }
    }

    /// Vérifie si `u` et `v` sont des jumeaux automorphes (transposition valide).
    fn are_twins(
        u: NodeId,
        v: NodeId,
        out_adj: &[Vec<(NodeId, u32)>],
        in_adj: &[Vec<(NodeId, u32)>],
    ) -> bool {
        let ext_out_u = out_adj[u].iter().copied().filter(|&(w, _)| w != v);
        let ext_out_v = out_adj[v].iter().copied().filter(|&(w, _)| w != u);
        if !ext_out_u.eq(ext_out_v) {
            return false;
        }

        let ext_in_u = in_adj[u].iter().copied().filter(|&(w, _)| w != v);
        let ext_in_v = in_adj[v].iter().copied().filter(|&(w, _)| w != u);
        if !ext_in_u.eq(ext_in_v) {
            return false;
        }

        let uv = out_adj[u]
            .iter()
            .filter(|&&(w, _)| w == v)
            .map(|&(_, tag)| tag);
        let vu = out_adj[v]
            .iter()
            .filter(|&&(w, _)| w == u)
            .map(|&(_, tag)| tag);
        uv.eq(vu)
    }

    fn canonical_search(
        &self,
        mut colors: Vec<usize>,
        out_adj: &[Vec<(NodeId, u32)>],
        in_adj: &[Vec<(NodeId, u32)>],
        best: &mut Option<CanonicalKey>,
    ) {
        Self::refine_colors(&mut colors, out_adj, in_adj);
        let n = colors.len();

        let mut classes: BTreeMap<usize, Vec<NodeId>> = BTreeMap::new();
        for (u, &c) in colors.iter().enumerate() {
            classes.entry(c).or_default().push(u);
        }

        let target_class = classes
            .into_iter()
            .find(|(_, members)| members.len() > 1);

        let Some((cls_color, members)) = target_class else {
            // Partition discrète : colors[u] est une permutation exacte 0..n.
            let mut relabeled_nodes = vec![NodeKind::Seed; n];
            for (u, &new_id) in colors.iter().enumerate() {
                relabeled_nodes[new_id] = self.nodes[u];
            }
            let mut relabeled_edges: Vec<Edge> = self
                .edges
                .iter()
                .map(|e| Edge::new(colors[e.src], colors[e.dst], e.kind))
                .collect();
            relabeled_edges.sort_unstable();

            let candidate = CanonicalKey {
                stopped: self.stopped,
                nodes: relabeled_nodes,
                edges: relabeled_edges,
            };
            if best.as_ref().is_none_or(|b| &candidate < b) {
                *best = Some(candidate);
            }
            return;
        };

        // Élagage exact des nœuds jumeaux au sein de la classe choisie.
        let mut reps: Vec<NodeId> = Vec::new();
        for u in members {
            if !reps.iter().any(|&r| Self::are_twins(u, r, out_adj, in_adj)) {
                reps.push(u);
            }
        }

        for rep in reps {
            let mut next_colors = colors.clone();
            for (u, c) in next_colors.iter_mut().enumerate() {
                if *c > cls_color {
                    *c += 1;
                } else if *c == cls_color && u != rep {
                    *c = cls_color + 1;
                }
            }
            self.canonical_search(next_colors, out_adj, in_adj, best);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PRNG déterministe minimal (SplitMix64) — zéro dépendance externe.
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

        fn pick_index(&mut self, len: usize) -> usize {
            (self.next_u64() as usize) % len
        }
    }

    #[test]
    fn test_random_trajectories_dag_and_exact_reversibility() {
        let config = GraphConfig::new(vec![b'a', b'b', b'c'], 2, 10, 3, 14).unwrap();
        let mut rng = SplitMix64::new(0x5048_4953_4841_5250);

        let mut observed_forward_families = [false; 4];
        let mut observed_backward_families = [false; 4];

        for _episode in 0..120 {
            let mut state = HypothesisGraph::new(config.clone());
            assert!(state.is_initial());
            assert_eq!(state.potential(), 1);
            assert!(state.legal_backward_actions().is_empty());

            // 1. Marche avant aléatoire P_F jusqu'à STOP ou budget atteint.
            for _step in 0..25 {
                let f_mask = state.forward_mask();
                let f_actions = state.legal_forward_actions();

                // Cohérence stricte entre masques de famille et liste d'actions.
                assert_eq!(
                    f_mask.family_mask[0],
                    f_actions
                        .iter()
                        .any(|a| matches!(a, ForwardAction::AddNode(_)))
                );
                assert_eq!(
                    f_mask.family_mask[1],
                    f_actions
                        .iter()
                        .any(|a| matches!(a, ForwardAction::Connect(_)))
                );
                assert_eq!(
                    f_mask.family_mask[2],
                    f_actions
                        .iter()
                        .any(|a| matches!(a, ForwardAction::Abstract(_)))
                );
                assert_eq!(
                    f_mask.family_mask[3],
                    f_actions.iter().any(|a| matches!(a, ForwardAction::Stop))
                );

                for action in &f_actions {
                    assert!(state.is_legal_forward(action));
                }

                if f_actions.is_empty() {
                    break;
                }

                // Échantillonnage uniforme par famille (couvre mieux Connect/Abstract/Stop).
                let active_families: Vec<usize> = (0..4)
                    .filter(|&i| f_mask.family_mask[i])
                    .collect();
                let chosen_fam = active_families[rng.pick_index(active_families.len())];
                observed_forward_families[chosen_fam] = true;

                let candidates: Vec<ForwardAction> = f_actions
                    .into_iter()
                    .filter(|a| match (chosen_fam, a) {
                        (0, ForwardAction::AddNode(_)) => true,
                        (1, ForwardAction::Connect(_)) => true,
                        (2, ForwardAction::Abstract(_)) => true,
                        (3, ForwardAction::Stop) => true,
                        _ => false,
                    })
                    .collect();
                let chosen = candidates[rng.pick_index(candidates.len())];

                let prev_canon = state.canonical_key();
                let prev_potential = state.potential();

                let next = state.apply_forward(&chosen).expect("Action P_F légale");
                next.check_invariants()
                    .expect("Invariants violés après P_F");

                // Propriété DAG : croissance stricte de +1.
                assert_eq!(
                    next.potential(),
                    prev_potential + 1,
                    "Le potentiel doit croître exactement de +1 sur {chosen:?}"
                );

                // Réversibilité P_F -> P_B : l'état précédent doit être atteignable par P_B.
                let b_actions = next.legal_backward_actions();
                assert!(!b_actions.is_empty());
                let found_parent = b_actions.iter().any(|ba| {
                    assert!(next.is_legal_backward(ba));
                    let parent = next.apply_backward(ba).expect("Action P_B légale");
                    parent.canonical_key() == prev_canon
                });
                assert!(
                    found_parent,
                    "Aucun inverse P_B trouvé pour l'action avant {chosen:?}"
                );

                // Réversibilité P_B -> P_F : TOUTE action P_B valide mène à un parent
                // depuis lequel une action P_F légale reconstruit exactement `next`.
                let next_canon = next.canonical_key();
                let b_mask = next.backward_mask();
                for (i, &active) in b_mask.family_mask.iter().enumerate() {
                    if active {
                        observed_backward_families[i] = true;
                    }
                }
                for ba in &b_actions {
                    let parent = next.apply_backward(ba).expect("Action P_B légale");
                    parent
                        .check_invariants()
                        .expect("Invariants violés après P_B");
                    assert_eq!(parent.potential() + 1, next.potential());

                    let rev_ok = parent.legal_forward_actions().iter().any(|fa| {
                        let rebuilt = parent.apply_forward(fa).expect("Action P_F légale");
                        rebuilt.canonical_key() == next_canon
                    });
                    assert!(
                        rev_ok,
                        "Le parent produit par {ba:?} ne peut pas reconstruire l'état courant via P_F"
                    );
                }

                state = next;
                if state.is_stopped() {
                    break;
                }
            }

            // 2. Marche arrière aléatoire P_B : doit TOUJOURS revenir à H_0 en S(H_T) - 1 pas.
            let expected_back_steps = state.potential() - 1;
            let mut back_steps = 0usize;
            while !state.is_initial() {
                let b_actions = state.legal_backward_actions();
                assert!(
                    !b_actions.is_empty(),
                    "Impasse arrière atteinte avant H_0 sur {state:?}"
                );
                let ba = b_actions[rng.pick_index(b_actions.len())];
                let prev_pot = state.potential();
                state = state.apply_backward(&ba).expect("Action P_B légale");
                state
                    .check_invariants()
                    .expect("Invariants violés en marche arrière");
                assert_eq!(state.potential() + 1, prev_pot);
                back_steps += 1;
            }
            assert_eq!(back_steps, expected_back_steps);
        }

        assert!(
            observed_forward_families.iter().all(|&x| x),
            "Toutes les familles P_F doivent être exercées : {observed_forward_families:?}"
        );
        assert!(
            observed_backward_families.iter().all(|&x| x),
            "Toutes les familles P_B doivent être exercées : {observed_backward_families:?}"
        );
    }

    #[test]
    fn test_strict_precondition_rejections() {
        let config = GraphConfig::new(vec![b'a', b'b'], 1, 8, 2, 10).unwrap();
        let h0 = HypothesisGraph::new(config);

        // STOP interdit sur H_0 ; DEL_NODE interdit sur SEED (0).
        assert!(h0.apply_forward(&ForwardAction::Stop).is_err());
        assert!(h0.apply_backward(&BackwardAction::DelNode(0)).is_err());

        // Construction : Rule(1), Conc('a')(2), Conc('b')(3), Conc('a')(4)
        let g = h0
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Rule))
            .unwrap()
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(b'a')))
            .unwrap()
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(b'b')))
            .unwrap()
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(b'a')))
            .unwrap();

        // Attachement de 2 au SEED : ABSTRACT interdit sur un nœud de SEED.
        let g = g
            .apply_forward(&ForwardAction::Connect(Edge::new(0, 2, EdgeKind::InSeed)))
            .unwrap();
        assert!(g.apply_forward(&ForwardAction::Abstract(2)).is_err());

        // Attachement de 3 en LHS et 4 en RHS de Rule(1).
        let g = g
            .apply_forward(&ForwardAction::Connect(Edge::new(1, 3, EdgeKind::InLhs)))
            .unwrap()
            .apply_forward(&ForwardAction::Connect(Edge::new(1, 4, EdgeKind::InRhs)))
            .unwrap();

        // STRUCT interdit entre blocs différents (LHS vs RHS).
        assert!(
            g.apply_forward(&ForwardAction::Connect(Edge::new(
                3,
                4,
                EdgeKind::Struct(0)
            )))
            .is_err()
        );

        // SAME_AS interdit sur des nœuds CONC.
        assert!(
            g.apply_forward(&ForwardAction::Connect(Edge::new(
                3,
                4,
                EdgeKind::SameAs
            )))
            .is_err()
        );

        // Abstraction de 3 et 4 en VAR puis liaison SAME_AS.
        let g = g
            .apply_forward(&ForwardAction::Abstract(3))
            .unwrap()
            .apply_forward(&ForwardAction::Abstract(4))
            .unwrap()
            .apply_forward(&ForwardAction::Connect(Edge::new(
                3,
                4,
                EdgeKind::SameAs
            )))
            .unwrap();

        // Interdits stricts sur P_B tant que SAME_AS existe :
        // 1. UN_ABSTRACT interdit si deg_SAME_AS > 0
        assert!(
            g.apply_backward(&BackwardAction::UnAbstract {
                node: 3,
                symbol: b'a'
            })
            .is_err()
        );
        // 2. DEL_EDGE sur IN_LHS interdit si deg_SAME_AS > 0
        assert!(
            g.apply_backward(&BackwardAction::DelEdge(Edge::new(1, 3, EdgeKind::InLhs)))
                .is_err()
        );
        // 3. DEL_NODE interdit sur un nœud VAR même une fois isolé
        let g_isolated_var = g
            .apply_backward(&BackwardAction::DelEdge(Edge::new(3, 4, EdgeKind::SameAs)))
            .unwrap()
            .apply_backward(&BackwardAction::DelEdge(Edge::new(1, 3, EdgeKind::InLhs)))
            .unwrap();
        assert_eq!(g_isolated_var.total_degree(3), 0);
        assert!(
            g_isolated_var
                .apply_backward(&BackwardAction::DelNode(3))
                .is_err()
        );
        // Il faut d'abord UN_ABSTRACT avant DEL_NODE.
        let g_conc = g_isolated_var
            .apply_backward(&BackwardAction::UnAbstract {
                node: 3,
                symbol: b'b',
            })
            .unwrap();
        assert!(g_conc.apply_backward(&BackwardAction::DelNode(3)).is_ok());
    }

    #[test]
    fn test_canonical_isomorphism_invariance() {
        let config = GraphConfig::new(vec![b'x', b'y'], 1, 8, 2, 10).unwrap();

        // Ordre A : crée Rule(1), Conc('x')(2), Conc('y')(3) -> abstrait en VAR + SAME_AS
        let ga = HypothesisGraph::new(config.clone())
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Rule))
            .unwrap()
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(b'x')))
            .unwrap()
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(b'y')))
            .unwrap()
            .apply_forward(&ForwardAction::Connect(Edge::new(1, 2, EdgeKind::InLhs)))
            .unwrap()
            .apply_forward(&ForwardAction::Connect(Edge::new(1, 3, EdgeKind::InRhs)))
            .unwrap()
            .apply_forward(&ForwardAction::Abstract(2))
            .unwrap()
            .apply_forward(&ForwardAction::Abstract(3))
            .unwrap()
            .apply_forward(&ForwardAction::Connect(Edge::new(
                2,
                3,
                EdgeKind::SameAs,
            )))
            .unwrap();

        // Ordre B : crée Conc('y')(1), Conc('x')(2), Rule(3), abstrait avant de connecter
        let gb = HypothesisGraph::new(config)
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(b'y')))
            .unwrap()
            .apply_forward(&ForwardAction::Abstract(1))
            .unwrap()
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Conc(b'x')))
            .unwrap()
            .apply_forward(&ForwardAction::Abstract(2))
            .unwrap()
            .apply_forward(&ForwardAction::AddNode(AddNodeKind::Rule))
            .unwrap()
            .apply_forward(&ForwardAction::Connect(Edge::new(3, 1, EdgeKind::InRhs)))
            .unwrap()
            .apply_forward(&ForwardAction::Connect(Edge::new(3, 2, EdgeKind::InLhs)))
            .unwrap()
            .apply_forward(&ForwardAction::Connect(Edge::new(
                1,
                2,
                EdgeKind::SameAs,
            )))
            .unwrap();

        // Les IDs internes diffèrent mais les formes canoniques doivent être identiques.
        assert_ne!(ga.nodes(), gb.nodes());
        assert_eq!(ga.canonical_key(), gb.canonical_key());
    }

    #[test]
    fn test_exhaustive_small_dag_bijection() {
        // Espace réduit couvrant SEED, RULE, IN_LHS, IN_RHS, STRUCT, SAME_AS, ABSTRACT et STOP.
        let config = GraphConfig::new(vec![b'a'], 1, 4, 1, 3).unwrap();
        let root = HypothesisGraph::new(config);
        let root_key = root.canonical_key();

        let mut visited: BTreeMap<CanonicalKey, HypothesisGraph> = BTreeMap::new();
        let mut queue: VecDeque<CanonicalKey> = VecDeque::new();
        visited.insert(root_key.clone(), root);
        queue.push_back(root_key);

        let mut forward_edges: BTreeSet<(CanonicalKey, CanonicalKey)> = BTreeSet::new();
        let mut backward_edges_reversed: BTreeSet<(CanonicalKey, CanonicalKey)> = BTreeSet::new();
        let mut saw_same_as = false;
        let mut saw_struct = false;

        while let Some(key) = queue.pop_front() {
            let state = visited[&key].clone();
            state.check_invariants().unwrap();

            for edge in state.edges() {
                if edge.kind == EdgeKind::SameAs {
                    saw_same_as = true;
                }
                if matches!(edge.kind, EdgeKind::Struct(_)) {
                    saw_struct = true;
                }
            }

            for fa in state.legal_forward_actions() {
                let next = state.apply_forward(&fa).unwrap();
                assert_eq!(next.potential(), state.potential() + 1);
                let next_key = next.canonical_key();
                forward_edges.insert((key.clone(), next_key.clone()));
                if !visited.contains_key(&next_key) {
                    visited.insert(next_key.clone(), next);
                    queue.push_back(next_key);
                }
            }

            for ba in state.legal_backward_actions() {
                let prev = state.apply_backward(&ba).unwrap();
                assert_eq!(prev.potential() + 1, state.potential());
                let prev_key = prev.canonical_key();
                backward_edges_reversed.insert((prev_key, key.clone()));
            }
        }

        assert!(saw_same_as, "L'exploration exhaustive doit couvrir SAME_AS");
        assert!(saw_struct, "L'exploration exhaustive doit couvrir STRUCT");
        assert_eq!(
            forward_edges, backward_edges_reversed,
            "Le graphe canonique P_F et le graphe inversé P_B doivent être strictement identiques"
        );
    }
}
