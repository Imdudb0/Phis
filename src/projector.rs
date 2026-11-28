#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use crate::hypothesis_graph::{Block, EdgeKind, HypothesisGraph, NodeId, NodeKind, RelId, SymbolId};

/// Arête dirigée et typée dans un graphe d'observation (G_E ou E_hat).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObsEdge {
    pub src: NodeId,
    pub dst: NodeId,
    pub rel: RelId,
}

impl ObsEdge {
    #[inline]
    pub const fn new(src: NodeId, dst: NodeId, rel: RelId) -> Self {
        Self { src, dst, rel }
    }
}

/// Représentation unifiée de l'observation E (G_E) et de la prédiction E_hat = Pi(H_t).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObservationGraph {
    pub nodes: Vec<SymbolId>,
    pub edges: BTreeSet<ObsEdge>,
}

impl ObservationGraph {
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            edges: BTreeSet::new(),
        }
    }

    pub fn new(nodes: Vec<SymbolId>, edges: impl IntoIterator<Item = ObsEdge>) -> Self {
        let mut set = BTreeSet::new();
        for e in edges {
            debug_assert!(e.src < nodes.len() && e.dst < nodes.len());
            set.insert(e);
        }
        Self { nodes, edges: set }
    }

    /// Encode directement un flux d'octets `perceived.eff` en graphe séquentiel (rel = 0 : NEXT).
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut edges = BTreeSet::new();
        for i in 0..bytes.len().saturating_sub(1) {
            edges.insert(ObsEdge::new(i, i + 1, 0));
        }
        Self {
            nodes: bytes.to_vec(),
            edges,
        }
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Extrait l'alphabet Sigma_E trié et dédupliqué pour initialiser `GraphConfig`.
    pub fn alphabet(&self) -> Vec<SymbolId> {
        let mut alpha = self.nodes.clone();
        alpha.sort_unstable();
        alpha.dedup();
        alpha
    }
}

// ============================================================================
// 1. Compilation et exécution déterministe du projecteur Pi(H_t)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum PatternAtom {
    Conc(SymbolId),
    Var(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CompiledPattern {
    atoms: Vec<PatternAtom>,
    edges: Vec<ObsEdge>,
    entries: Vec<usize>,
    exits: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct CompiledRule {
    rule_node: NodeId,
    lhs: CompiledPattern,
    rhs: CompiledPattern,
    /// Pour chaque classe de variable, indices des nœuds RHS appartenant à cette classe.
    rhs_var_positions: BTreeMap<usize, Vec<usize>>,
}

#[derive(Clone, Debug)]
pub struct CompiledPtgrs {
    seed: ObservationGraph,
    rules: Vec<CompiledRule>,
    pub skipped_rules: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectorConfig {
    /// Budget maximal d'étapes de réécriture.
    pub max_steps: usize,
    /// Taille maximale autorisée pour E_hat (coupe les explosions exponentielles).
    pub max_nodes: usize,
    /// Arrêt immédiat si un état exact du graphe hôte est revu.
    pub stop_on_cycle: bool,
}

impl Default for ProjectorConfig {
    fn default() -> Self {
        Self {
            max_steps: 64,
            max_nodes: 128,
            stop_on_cycle: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminationReason {
    /// Plus aucun redex applicable (forme normale atteinte).
    NormalForm,
    /// État déjà visité détecté pendant le déroulement.
    CycleDetected,
    /// Budget `max_steps` atteint.
    MaxStepsReached,
    /// Borne `max_nodes` atteinte avant d'appliquer la règle suivante.
    MaxNodesReached,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionOutcome {
    pub predicted: ObservationGraph,
    pub steps_executed: usize,
    pub skipped_rules: usize,
    pub termination: TerminationReason,
}

/// Union-Find minimal pour résoudre la fermeture transitive des arêtes `SameAs`.
struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = self.parent[p];
            x = p;
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            if ra < rb {
                self.parent[rb] = ra;
            } else {
                self.parent[ra] = rb;
            }
        }
    }
}

/// Calcule les nœuds d'entrée (degré entrant minimal) et de sortie (degré sortant minimal).
fn compute_boundary_nodes(n: usize, edges: &[ObsEdge]) -> (Vec<usize>, Vec<usize>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut in_deg = vec![0usize; n];
    let mut out_deg = vec![0usize; n];
    for e in edges {
        out_deg[e.src] += 1;
        in_deg[e.dst] += 1;
    }

    let min_in = *in_deg.iter().min().unwrap_or(&0);
    let min_out = *out_deg.iter().min().unwrap_or(&0);

    let entries: Vec<usize> = (0..n).filter(|&i| in_deg[i] == min_in).collect();
    let exits: Vec<usize> = (0..n).filter(|&i| out_deg[i] == min_out).collect();
    (entries, exits)
}

impl CompiledPtgrs {
    /// Extrait S_0 et les règles exécutables depuis un `HypothesisGraph` (même partiel).
    pub fn compile(h: &HypothesisGraph) -> Self {
        let n = h.node_count();
        let memberships: Vec<Option<Block>> = (0..n).map(|v| h.membership(v)).collect();

        // 1. Extraction du germe S_0.
        let mut seed_map: BTreeMap<NodeId, usize> = BTreeMap::new();
        let mut seed_nodes: Vec<SymbolId> = Vec::new();
        for (v, &m) in memberships.iter().enumerate() {
            if m == Some(Block::Seed) {
                if let NodeKind::Conc(sym) = h.nodes()[v] {
                    seed_map.insert(v, seed_nodes.len());
                    seed_nodes.push(sym);
                }
            }
        }
        let mut seed_edges = BTreeSet::new();
        for &e in h.edges() {
            if let EdgeKind::Struct(rel) = e.kind {
                if let (Some(&s), Some(&d)) = (seed_map.get(&e.src), seed_map.get(&e.dst)) {
                    seed_edges.insert(ObsEdge::new(s, d, rel));
                }
            }
        }
        let seed = ObservationGraph {
            nodes: seed_nodes,
            edges: seed_edges,
        };

        // 2. Extraction des règles exécutables.
        let mut rules = Vec::new();
        let mut skipped_rules = 0usize;

        for (rule_id, &kind) in h.nodes().iter().enumerate() {
            if kind != NodeKind::Rule {
                continue;
            }

            let mut lhs_ids: Vec<NodeId> = Vec::new();
            let mut rhs_ids: Vec<NodeId> = Vec::new();
            for (v, &m) in memberships.iter().enumerate() {
                match m {
                    Some(Block::RuleLhs(r)) if r == rule_id => lhs_ids.push(v),
                    Some(Block::RuleRhs(r)) if r == rule_id => rhs_ids.push(v),
                    _ => {}
                }
            }

            // Une règle sans LHS ne peut pas être appariée : on l'ignore proprement.
            if lhs_ids.is_empty() {
                skipped_rules += 1;
                continue;
            }

            // Fermeture transitive des `SameAs` au sein de la règle.
            let mut all_rule_nodes: Vec<NodeId> = Vec::with_capacity(lhs_ids.len() + rhs_ids.len());
            all_rule_nodes.extend_from_slice(&lhs_ids);
            all_rule_nodes.extend_from_slice(&rhs_ids);

            let mut local_idx: BTreeMap<NodeId, usize> = BTreeMap::new();
            for (i, &nid) in all_rule_nodes.iter().enumerate() {
                local_idx.insert(nid, i);
            }
            let mut dsu = DisjointSet::new(all_rule_nodes.len());
            for &e in h.edges() {
                if e.kind == EdgeKind::SameAs {
                    if let (Some(&a), Some(&b)) = (local_idx.get(&e.src), local_idx.get(&e.dst)) {
                        dsu.union(a, b);
                    }
                }
            }

            // Numérotation compacte des classes de variables apparaissant dans le LHS.
            let mut root_to_class: BTreeMap<usize, usize> = BTreeMap::new();
            let mut lhs_atoms = Vec::with_capacity(lhs_ids.len());
            let mut lhs_pos: BTreeMap<NodeId, usize> = BTreeMap::new();

            for (pos, &nid) in lhs_ids.iter().enumerate() {
                lhs_pos.insert(nid, pos);
                match h.nodes()[nid] {
                    NodeKind::Conc(sym) => lhs_atoms.push(PatternAtom::Conc(sym)),
                    NodeKind::Var => {
                        let root = dsu.find(local_idx[&nid]);
                        let next_id = root_to_class.len();
                        let class_id = *root_to_class.entry(root).or_insert(next_id);
                        lhs_atoms.push(PatternAtom::Var(class_id));
                    }
                    NodeKind::Seed | NodeKind::Rule => {}
                }
            }

            // Vérifie que toute variable RHS est liée à au moins une variable LHS.
            let mut rhs_atoms = Vec::with_capacity(rhs_ids.len());
            let mut rhs_pos: BTreeMap<NodeId, usize> = BTreeMap::new();
            let mut rhs_var_positions: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            let mut has_unbound_var = false;

            for (pos, &nid) in rhs_ids.iter().enumerate() {
                rhs_pos.insert(nid, pos);
                match h.nodes()[nid] {
                    NodeKind::Conc(sym) => rhs_atoms.push(PatternAtom::Conc(sym)),
                    NodeKind::Var => {
                        let root = dsu.find(local_idx[&nid]);
                        if let Some(&class_id) = root_to_class.get(&root) {
                            rhs_atoms.push(PatternAtom::Var(class_id));
                            rhs_var_positions.entry(class_id).or_default().push(pos);
                        } else {
                            has_unbound_var = true;
                            break;
                        }
                    }
                    NodeKind::Seed | NodeKind::Rule => {}
                }
            }

            if has_unbound_var {
                skipped_rules += 1;
                continue;
            }

            let mut lhs_edges = Vec::new();
            let mut rhs_edges = Vec::new();
            for &e in h.edges() {
                if let EdgeKind::Struct(rel) = e.kind {
                    if let (Some(&s), Some(&d)) = (lhs_pos.get(&e.src), lhs_pos.get(&e.dst)) {
                        lhs_edges.push(ObsEdge::new(s, d, rel));
                    } else if let (Some(&s), Some(&d)) = (rhs_pos.get(&e.src), rhs_pos.get(&e.dst))
                    {
                        rhs_edges.push(ObsEdge::new(s, d, rel));
                    }
                }
            }
            lhs_edges.sort_unstable();
            rhs_edges.sort_unstable();

            let (lhs_entries, lhs_exits) = compute_boundary_nodes(lhs_atoms.len(), &lhs_edges);
            let (rhs_entries, rhs_exits) = compute_boundary_nodes(rhs_atoms.len(), &rhs_edges);

            rules.push(CompiledRule {
                rule_node: rule_id,
                lhs: CompiledPattern {
                    atoms: lhs_atoms,
                    edges: lhs_edges,
                    entries: lhs_entries,
                    exits: lhs_exits,
                },
                rhs: CompiledPattern {
                    atoms: rhs_atoms,
                    edges: rhs_edges,
                    entries: rhs_entries,
                    exits: rhs_exits,
                },
                rhs_var_positions,
            });
        }

        // Ordre déterministe : règles plus spécifiques (LHS plus grand) prioritaires.
        rules.sort_by(|a, b| {
            b.lhs
                .atoms
                .len()
                .cmp(&a.lhs.atoms.len())
                .then_with(|| b.lhs.edges.len().cmp(&a.lhs.edges.len()))
                .then_with(|| a.lhs.cmp(&b.lhs))
                .then_with(|| a.rhs.cmp(&b.rhs))
                .then_with(|| a.rule_node.cmp(&b.rule_node))
        });

        Self {
            seed,
            rules,
            skipped_rules,
        }
    }

    /// Exécute le PTGRS compilé pour produire E_hat = Pi(H_t).
    pub fn execute(&self, cfg: ProjectorConfig) -> ProjectionOutcome {
        let mut host = self.seed.clone();
        if host.node_count() > cfg.max_nodes {
            return ProjectionOutcome {
                predicted: host,
                steps_executed: 0,
                skipped_rules: self.skipped_rules,
                termination: TerminationReason::MaxNodesReached,
            };
        }

        let mut visited: BTreeSet<ObservationGraph> = BTreeSet::new();
        if cfg.stop_on_cycle {
            visited.insert(host.clone());
        }

        let mut steps = 0usize;
        while steps < cfg.max_steps {
            let mut applied = false;
            for rule in &self.rules {
                if let Some((mapping, bindings)) = find_first_match(&host, &rule.lhs) {
                    let projected_size =
                        host.node_count() - rule.lhs.atoms.len() + rule.rhs.atoms.len();
                    if projected_size > cfg.max_nodes {
                        return ProjectionOutcome {
                            predicted: host,
                            steps_executed: steps,
                            skipped_rules: self.skipped_rules,
                            termination: TerminationReason::MaxNodesReached,
                        };
                    }

                    host = apply_rewrite(&host, rule, &mapping, &bindings);
                    steps += 1;
                    applied = true;

                    if cfg.stop_on_cycle && !visited.insert(host.clone()) {
                        return ProjectionOutcome {
                            predicted: host,
                            steps_executed: steps,
                            skipped_rules: self.skipped_rules,
                            termination: TerminationReason::CycleDetected,
                        };
                    }
                    break;
                }
            }

            if !applied {
                return ProjectionOutcome {
                    predicted: host,
                    steps_executed: steps,
                    skipped_rules: self.skipped_rules,
                    termination: TerminationReason::NormalForm,
                };
            }
        }

        ProjectionOutcome {
            predicted: host,
            steps_executed: steps,
            skipped_rules: self.skipped_rules,
            termination: TerminationReason::MaxStepsReached,
        }
    }
}

/// Raccourci principal : projette directement `HypothesisGraph` en `ProjectionOutcome`.
pub fn project(h: &HypothesisGraph, cfg: ProjectorConfig) -> ProjectionOutcome {
    CompiledPtgrs::compile(h).execute(cfg)
}

/// Trouve le premier homomorphisme injectif lexicographique de `lhs` dans `host`.
fn find_first_match(
    host: &ObservationGraph,
    lhs: &CompiledPattern,
) -> Option<(Vec<NodeId>, BTreeMap<usize, SymbolId>)> {
    let k = lhs.atoms.len();
    if k == 0 || k > host.node_count() {
        return None;
    }

    let mut mapping = vec![usize::MAX; k];
    let mut used = vec![false; host.node_count()];
    let mut bindings: BTreeMap<usize, SymbolId> = BTreeMap::new();

    if backtrack_match(0, host, lhs, &mut mapping, &mut used, &mut bindings) {
        Some((mapping, bindings))
    } else {
        None
    }
}

fn backtrack_match(
    idx: usize,
    host: &ObservationGraph,
    lhs: &CompiledPattern,
    mapping: &mut [NodeId],
    used: &mut [bool],
    bindings: &mut BTreeMap<usize, SymbolId>,
) -> bool {
    if idx == lhs.atoms.len() {
        return true;
    }

    for h_node in 0..host.node_count() {
        if used[h_node] {
            continue;
        }
        let h_sym = host.nodes[h_node];

        // 1. Contrainte d'atome (`Conc` ou cohérence de classe `Var`).
        let mut inserted_var: Option<usize> = None;
        match lhs.atoms[idx] {
            PatternAtom::Conc(expected) => {
                if h_sym != expected {
                    continue;
                }
            }
            PatternAtom::Var(class_id) => {
                if let Some(&bound) = bindings.get(&class_id) {
                    if bound != h_sym {
                        continue;
                    }
                } else {
                    bindings.insert(class_id, h_sym);
                    inserted_var = Some(class_id);
                }
            }
        }

        // 2. Contraintes d'arêtes déjà closes avec les indices < idx.
        let mut edges_ok = true;
        for e in &lhs.edges {
            if e.src == idx && e.dst < idx {
                if !host.edges.contains(&ObsEdge::new(h_node, mapping[e.dst], e.rel)) {
                    edges_ok = false;
                    break;
                }
            } else if e.dst == idx && e.src < idx {
                if !host.edges.contains(&ObsEdge::new(mapping[e.src], h_node, e.rel)) {
                    edges_ok = false;
                    break;
                }
            } else if e.src == idx && e.dst == idx {
                if !host.edges.contains(&ObsEdge::new(h_node, h_node, e.rel)) {
                    edges_ok = false;
                    break;
                }
            }
        }

        if edges_ok {
            mapping[idx] = h_node;
            used[h_node] = true;
            if backtrack_match(idx + 1, host, lhs, mapping, used, bindings) {
                return true;
            }
            used[h_node] = false;
            mapping[idx] = usize::MAX;
        }

        if let Some(cid) = inserted_var {
            bindings.remove(&cid);
        }
    }
    false
}

/// Applique la réécriture `rule` sur le redex `mapping` avec reconnexion de frontière.
fn apply_rewrite(
    host: &ObservationGraph,
    rule: &CompiledRule,
    mapping: &[NodeId],
    bindings: &BTreeMap<usize, SymbolId>,
) -> ObservationGraph {
    let n = host.node_count();
    let mut host_to_lhs: Vec<Option<usize>> = vec![None; n];
    for (lhs_idx, &h_node) in mapping.iter().enumerate() {
        host_to_lhs[h_node] = Some(lhs_idx);
    }

    // 1. Copie des nœuds contextuels non consommés.
    let mut new_nodes: Vec<SymbolId> = Vec::with_capacity(n - mapping.len() + rule.rhs.atoms.len());
    let mut old_to_new: Vec<Option<NodeId>> = vec![None; n];
    for (h_node, &sym) in host.nodes.iter().enumerate() {
        if host_to_lhs[h_node].is_none() {
            old_to_new[h_node] = Some(new_nodes.len());
            new_nodes.push(sym);
        }
    }

    // 2. Instanciation des nœuds RHS avec les symboles liés.
    let rhs_offset = new_nodes.len();
    for atom in &rule.rhs.atoms {
        let sym = match *atom {
            PatternAtom::Conc(c) => c,
            PatternAtom::Var(cid) => bindings[&cid],
        };
        new_nodes.push(sym);
    }

    let mut new_edges: BTreeSet<ObsEdge> = BTreeSet::new();

    // 3. Arêtes internes du RHS.
    for e in &rule.rhs.edges {
        new_edges.insert(ObsEdge::new(rhs_offset + e.src, rhs_offset + e.dst, e.rel));
    }

    let lhs_entry_set: BTreeSet<usize> = rule.lhs.entries.iter().copied().collect();
    let lhs_exit_set: BTreeSet<usize> = rule.lhs.exits.iter().copied().collect();

    // 4. Arêtes contextuelles et reconnexion des frontières.
    for &e in &host.edges {
        match (host_to_lhs[e.src], host_to_lhs[e.dst]) {
            // Arête purement externe : conservée intacte.
            (None, None) => {
                let ns = old_to_new[e.src].unwrap();
                let nd = old_to_new[e.dst].unwrap();
                new_edges.insert(ObsEdge::new(ns, nd, e.rel));
            }
            // Arête purement interne au redex : supprimée (remplacée par le RHS).
            (Some(_), Some(_)) => {}
            // Arête entrante vers le redex : reconnectée aux entrées RHS (ou partenaire VAR unique).
            (None, Some(lhs_idx)) => {
                if rule.rhs.atoms.is_empty() {
                    continue;
                }
                let ns = old_to_new[e.src].unwrap();
                let targets = resolve_incoming_targets(rule, lhs_idx, &lhs_entry_set);
                for &rhs_idx in targets {
                    new_edges.insert(ObsEdge::new(ns, rhs_offset + rhs_idx, e.rel));
                }
            }
            // Arête sortante du redex : reconnectée depuis les sorties RHS (ou partenaire VAR unique).
            (Some(lhs_idx), None) => {
                if rule.rhs.atoms.is_empty() {
                    continue;
                }
                let nd = old_to_new[e.dst].unwrap();
                let sources = resolve_outgoing_sources(rule, lhs_idx, &lhs_exit_set);
                for &rhs_idx in sources {
                    new_edges.insert(ObsEdge::new(rhs_offset + rhs_idx, nd, e.rel));
                }
            }
        }
    }

    ObservationGraph {
        nodes: new_nodes,
        edges: new_edges,
    }
}

fn resolve_incoming_targets<'a>(
    rule: &'a CompiledRule,
    lhs_idx: usize,
    lhs_entry_set: &BTreeSet<usize>,
) -> &'a [usize] {
    if lhs_entry_set.contains(&lhs_idx) {
        return &rule.rhs.entries;
    }
    if let PatternAtom::Var(cid) = rule.lhs.atoms[lhs_idx] {
        if let Some(pos) = rule.rhs_var_positions.get(&cid) {
            if pos.len() == 1 {
                return pos.as_slice();
            }
        }
    }
    &rule.rhs.entries
}

fn resolve_outgoing_sources<'a>(
    rule: &'a CompiledRule,
    lhs_idx: usize,
    lhs_exit_set: &BTreeSet<usize>,
) -> &'a [usize] {
    if lhs_exit_set.contains(&lhs_idx) {
        return &rule.rhs.exits;
    }
    if let PatternAtom::Var(cid) = rule.lhs.atoms[lhs_idx] {
        if let Some(pos) = rule.rhs_var_positions.get(&cid) {
            if pos.len() == 1 {
                return pos.as_slice();
            }
        }
    }
    &rule.rhs.exits
}

// ============================================================================
// 2. Graphe différentiel de résidu structuré R_t = G_E (-) E_hat
// ============================================================================

/// Poids d'édition utilisés à la fois par l'aligneur et par la norme ||R_t||.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualWeights {
    pub missing_node: f64,
    pub excess_node: f64,
    pub conflict_node: f64,
    pub missing_edge: f64,
    pub excess_edge: f64,
    pub conflict_edge: f64,
}

impl Default for ResidualWeights {
    fn default() -> Self {
        Self {
            missing_node: 1.0,
            excess_node: 1.0,
            // > missing_node + excess_node (2.0) pour ne jamais forcer un conflit entre
            // deux nœuds sans arête commune, tout en restant < 4.0 pour préférer un conflit
            // localisé dès qu'au moins 1 arête incidente coïncide (ex. "axc" vs "ayc").
            conflict_node: 2.2,
            missing_edge: 1.0,
            excess_edge: 1.0,
            conflict_edge: 1.2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeAlignmentStatus {
    Matched(SymbolId),
    Conflict {
        expected: SymbolId,
        predicted: SymbolId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlignedNode {
    pub obs_node: NodeId,
    pub pred_node: NodeId,
    pub status: NodeAlignmentStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualEdgeKind {
    Matched(RelId),
    Missing(RelId),
    Excess(RelId),
    Conflict {
        expected_rel: RelId,
        predicted_rel: RelId,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidualEdge {
    pub kind: ResidualEdgeKind,
    pub obs_endpoints: Option<(NodeId, NodeId)>,
    pub pred_endpoints: Option<(NodeId, NodeId)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidualSummary {
    pub matched_nodes: usize,
    pub missing_nodes: usize,
    pub excess_nodes: usize,
    pub conflict_nodes: usize,
    pub matched_edges: usize,
    pub missing_edges: usize,
    pub excess_edges: usize,
    pub conflict_edges: usize,
}

/// Graphe différentiel structuré R_t réinjecté dans le Backbone Relationnel à chaque pas t.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualGraph {
    pub aligned_nodes: Vec<AlignedNode>,
    pub missing_nodes: Vec<(NodeId, SymbolId)>,
    pub excess_nodes: Vec<(NodeId, SymbolId)>,
    pub edges: Vec<ResidualEdge>,
    pub summary: ResidualSummary,
    pub norm: f64,
}

impl ResidualGraph {
    /// Vrai ssi G_E et E_hat sont exactement isomorphes (résidu strictement nul).
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.summary.missing_nodes == 0
            && self.summary.excess_nodes == 0
            && self.summary.conflict_nodes == 0
            && self.summary.missing_edges == 0
            && self.summary.excess_edges == 0
            && self.summary.conflict_edges == 0
    }
}

/// Calcule le graphe différentiel R_t = G_E (-) E_hat avec alignement structurel optimal/raffiné.
pub fn compute_residual(
    obs: &ObservationGraph,
    pred: &ObservationGraph,
    weights: ResidualWeights,
) -> ResidualGraph {
    let mapping = align_graphs(obs, pred, weights);
    build_residual_from_mapping(obs, pred, &mapping, weights)
}

/// Calcule directement R_t à partir de H_t et G_E en une étape.
pub fn project_and_residual(
    h: &HypothesisGraph,
    obs: &ObservationGraph,
    proj_cfg: ProjectorConfig,
    weights: ResidualWeights,
) -> (ProjectionOutcome, ResidualGraph) {
    let outcome = project(h, proj_cfg);
    let residual = compute_residual(obs, &outcome.predicted, weights);
    (outcome, residual)
}

/// Construit la structure complète `ResidualGraph` depuis un alignement partiel `obs -> Option<pred>`.
fn build_residual_from_mapping(
    obs: &ObservationGraph,
    pred: &ObservationGraph,
    obs_to_pred: &[Option<NodeId>],
    weights: ResidualWeights,
) -> ResidualGraph {
    let n_obs = obs.node_count();
    let n_pred = pred.node_count();
    let mut pred_to_obs: Vec<Option<NodeId>> = vec![None; n_pred];

    let mut aligned_nodes = Vec::new();
    let mut missing_nodes = Vec::new();
    let mut summary = ResidualSummary::default();

    for u in 0..n_obs {
        match obs_to_pred[u] {
            Some(pu) => {
                pred_to_obs[pu] = Some(u);
                let expected = obs.nodes[u];
                let predicted = pred.nodes[pu];
                let status = if expected == predicted {
                    summary.matched_nodes += 1;
                    NodeAlignmentStatus::Matched(expected)
                } else {
                    summary.conflict_nodes += 1;
                    NodeAlignmentStatus::Conflict {
                        expected,
                        predicted,
                    }
                };
                aligned_nodes.push(AlignedNode {
                    obs_node: u,
                    pred_node: pu,
                    status,
                });
            }
            None => {
                summary.missing_nodes += 1;
                missing_nodes.push((u, obs.nodes[u]));
            }
        }
    }

    let mut excess_nodes = Vec::new();
    for pu in 0..n_pred {
        if pred_to_obs[pu].is_none() {
            summary.excess_nodes += 1;
            excess_nodes.push((pu, pred.nodes[pu]));
        }
    }

    // Regroupement des arêtes par paires d'extrémités pour isoler proprement les conflits de relation.
    let mut obs_pair_rels: BTreeMap<(NodeId, NodeId), Vec<RelId>> = BTreeMap::new();
    for &e in &obs.edges {
        obs_pair_rels.entry((e.src, e.dst)).or_default().push(e.rel);
    }
    let mut pred_pair_rels: BTreeMap<(NodeId, NodeId), Vec<RelId>> = BTreeMap::new();
    for &e in &pred.edges {
        pred_pair_rels.entry((e.src, e.dst)).or_default().push(e.rel);
    }

    let mut edges = Vec::new();
    let mut consumed_pred_pairs: BTreeSet<(NodeId, NodeId)> = BTreeSet::new();

    for (&(u, v), o_rels) in &obs_pair_rels {
        match (obs_to_pred[u], obs_to_pred[v]) {
            (Some(pu), Some(pv)) => {
                consumed_pred_pairs.insert((pu, pv));
                let empty = Vec::new();
                let p_rels = pred_pair_rels.get(&(pu, pv)).unwrap_or(&empty);

                let mut only_obs = Vec::new();
                let mut only_pred = Vec::new();
                for &r in o_rels {
                    if p_rels.binary_search(&r).is_ok() {
                        summary.matched_edges += 1;
                        edges.push(ResidualEdge {
                            kind: ResidualEdgeKind::Matched(r),
                            obs_endpoints: Some((u, v)),
                            pred_endpoints: Some((pu, pv)),
                        });
                    } else {
                        only_obs.push(r);
                    }
                }
                for &r in p_rels {
                    if o_rels.binary_search(&r).is_err() {
                        only_pred.push(r);
                    }
                }

                let conflicts = only_obs.len().min(only_pred.len());
                for i in 0..conflicts {
                    summary.conflict_edges += 1;
                    edges.push(ResidualEdge {
                        kind: ResidualEdgeKind::Conflict {
                            expected_rel: only_obs[i],
                            predicted_rel: only_pred[i],
                        },
                        obs_endpoints: Some((u, v)),
                        pred_endpoints: Some((pu, pv)),
                    });
                }
                for &r in &only_obs[conflicts..] {
                    summary.missing_edges += 1;
                    edges.push(ResidualEdge {
                        kind: ResidualEdgeKind::Missing(r),
                        obs_endpoints: Some((u, v)),
                        pred_endpoints: None,
                    });
                }
                for &r in &only_pred[conflicts..] {
                    summary.excess_edges += 1;
                    edges.push(ResidualEdge {
                        kind: ResidualEdgeKind::Excess(r),
                        obs_endpoints: None,
                        pred_endpoints: Some((pu, pv)),
                    });
                }
            }
            _ => {
                for &r in o_rels {
                    summary.missing_edges += 1;
                    edges.push(ResidualEdge {
                        kind: ResidualEdgeKind::Missing(r),
                        obs_endpoints: Some((u, v)),
                        pred_endpoints: None,
                    });
                }
            }
        }
    }

    for (&(pu, pv), p_rels) in &pred_pair_rels {
        if consumed_pred_pairs.contains(&(pu, pv)) {
            continue;
        }
        for &r in p_rels {
            summary.excess_edges += 1;
            edges.push(ResidualEdge {
                kind: ResidualEdgeKind::Excess(r),
                obs_endpoints: None,
                pred_endpoints: Some((pu, pv)),
            });
        }
    }

    let norm = summary.missing_nodes as f64 * weights.missing_node
        + summary.excess_nodes as f64 * weights.excess_node
        + summary.conflict_nodes as f64 * weights.conflict_node
        + summary.missing_edges as f64 * weights.missing_edge
        + summary.excess_edges as f64 * weights.excess_edge
        + summary.conflict_edges as f64 * weights.conflict_edge;

    ResidualGraph {
        aligned_nodes,
        missing_nodes,
        excess_nodes,
        edges,
        summary,
        norm,
    }
}

// ============================================================================
// 3. Moteur d'alignement structurel (Branch-and-Bound exact + Seed-and-Extend/2-opt)
// ============================================================================

struct IndexedGraph {
    nodes: Vec<SymbolId>,
    out_adj: Vec<Vec<(NodeId, RelId)>>,
    in_adj: Vec<Vec<(NodeId, RelId)>>,
    pair_rels: BTreeMap<(NodeId, NodeId), Vec<RelId>>,
}

impl IndexedGraph {
    fn from_obs(g: &ObservationGraph) -> Self {
        let n = g.node_count();
        let mut out_adj = vec![Vec::new(); n];
        let mut in_adj = vec![Vec::new(); n];
        let mut pair_rels: BTreeMap<(NodeId, NodeId), Vec<RelId>> = BTreeMap::new();
        for &e in &g.edges {
            out_adj[e.src].push((e.dst, e.rel));
            in_adj[e.dst].push((e.src, e.rel));
            pair_rels.entry((e.src, e.dst)).or_default().push(e.rel);
        }
        Self {
            nodes: g.nodes.clone(),
            out_adj,
            in_adj,
            pair_rels,
        }
    }

    fn rels_between(&self, u: NodeId, v: NodeId) -> &[RelId] {
        self.pair_rels.get(&(u, v)).map_or(&[], Vec::as_slice)
    }
}

/// Calcule le coût d'édition exact d'un alignement `obs_to_pred` (cohérent avec `ResidualGraph.norm`).
fn evaluate_alignment_cost(
    obs: &IndexedGraph,
    pred: &IndexedGraph,
    obs_to_pred: &[Option<NodeId>],
    w: ResidualWeights,
) -> f64 {
    let mut pred_used = vec![false; pred.nodes.len()];
    let mut cost = 0.0;

    for (u, &mapped) in obs_to_pred.iter().enumerate() {
        match mapped {
            Some(pu) => {
                pred_used[pu] = true;
                if obs.nodes[u] != pred.nodes[pu] {
                    cost += w.conflict_node;
                }
            }
            None => cost += w.missing_node,
        }
    }
    for &used in &pred_used {
        if !used {
            cost += w.excess_node;
        }
    }

    let mut consumed_pred_pairs: BTreeSet<(NodeId, NodeId)> = BTreeSet::new();
    for (&(u, v), o_rels) in &obs.pair_rels {
        match (obs_to_pred[u], obs_to_pred[v]) {
            (Some(pu), Some(pv)) => {
                consumed_pred_pairs.insert((pu, pv));
                let p_rels = pred.rels_between(pu, pv);
                cost += pair_edge_cost(o_rels, p_rels, w);
            }
            _ => {
                cost += o_rels.len() as f64 * w.missing_edge;
            }
        }
    }
    for (&(pu, pv), p_rels) in &pred.pair_rels {
        if !consumed_pred_pairs.contains(&(pu, pv)) {
            cost += p_rels.len() as f64 * w.excess_edge;
        }
    }
    cost
}

#[inline]
fn pair_edge_cost(o_rels: &[RelId], p_rels: &[RelId], w: ResidualWeights) -> f64 {
    let mut matched = 0usize;
    for &r in o_rels {
        if p_rels.binary_search(&r).is_ok() {
            matched += 1;
        }
    }
    let only_o = o_rels.len() - matched;
    let only_p = p_rels.len() - matched;
    let conflicts = only_o.min(only_p);
    conflicts as f64 * w.conflict_edge
        + (only_o - conflicts) as f64 * w.missing_edge
        + (only_p - conflicts) as f64 * w.excess_edge
}

fn align_graphs(
    obs: &ObservationGraph,
    pred: &ObservationGraph,
    w: ResidualWeights,
) -> Vec<Option<NodeId>> {
    let n_obs = obs.node_count();
    let n_pred = pred.node_count();
    if n_obs == 0 {
        return Vec::new();
    }
    if n_pred == 0 {
        return vec![None; n_obs];
    }

    let i_obs = IndexedGraph::from_obs(obs);
    let i_pred = IndexedGraph::from_obs(pred);

    // 1. Heuristique multi-graines (Seed-and-Extend + affinité WL) + raffinement 2-opt.
    let (mut best_map, mut best_cost) = heuristic_align(&i_obs, &i_pred, w);
    if best_cost == 0.0 {
        return best_map;
    }

    // 2. Branch-and-Bound exact pour les petits graphes (garantit l'optimum global).
    if n_obs <= 9 && n_pred <= 9 {
        let mut ctx = ExactAlignCtx {
            obs: &i_obs,
            pred: &i_pred,
            w,
            best_cost,
            best_map: best_map.clone(),
            budget_left: 60_000,
        };
        let mut current = vec![None; n_obs];
        let mut pred_used = vec![false; n_pred];
        ctx.search(0, 0.0, &mut current, &mut pred_used);
        if ctx.best_cost < best_cost {
            best_map = ctx.best_map;
            best_cost = ctx.best_cost;
        }
    }

    let _ = best_cost;
    best_map
}

struct ExactAlignCtx<'a> {
    obs: &'a IndexedGraph,
    pred: &'a IndexedGraph,
    w: ResidualWeights,
    best_cost: f64,
    best_map: Vec<Option<NodeId>>,
    budget_left: usize,
}

impl<'a> ExactAlignCtx<'a> {
    /// Borne inférieure admissible O(|Sigma|) sur le coût des nœuds restants.
    fn remaining_node_lower_bound(&self, obs_idx: usize, pred_used: &[bool]) -> f64 {
        let rem_o = self.obs.nodes.len() - obs_idx;
        let rem_p = pred_used.iter().filter(|&&u| !u).count();

        let mut o_counts = [0usize; 256];
        for &sym in &self.obs.nodes[obs_idx..] {
            o_counts[usize::from(sym)] += 1;
        }
        let mut p_counts = [0usize; 256];
        for (pu, &used) in pred_used.iter().enumerate() {
            if !used {
                p_counts[usize::from(self.pred.nodes[pu])] += 1;
            }
        }
        let max_exact: usize = o_counts
            .iter()
            .zip(p_counts.iter())
            .map(|(&a, &b)| a.min(b))
            .sum();

        let max_k = rem_o.min(rem_p);
        let candidates = [0usize, max_exact.min(max_k), max_k];
        let mut min_lb = f64::INFINITY;
        for k in candidates {
            let conflicts = k.saturating_sub(max_exact);
            let lb = (rem_o - k) as f64 * self.w.missing_node
                + (rem_p - k) as f64 * self.w.excess_node
                + conflicts as f64 * self.w.conflict_node;
            if lb < min_lb {
                min_lb = lb;
            }
        }
        min_lb
    }

    /// Coût incrémental des arêtes nouvellement closes lorsqu'on décide `current[u]`.
    fn incremental_edge_cost(&self, u: usize, current: &[Option<NodeId>]) -> f64 {
        let mut delta = 0.0;
        let pu_opt = current[u];

        // Boucle propre (u -> u).
        let o_self = self.obs.rels_between(u, u);
        match pu_opt {
            Some(pu) => {
                let p_self = self.pred.rels_between(pu, pu);
                if !o_self.is_empty() || !p_self.is_empty() {
                    delta += pair_edge_cost(o_self, p_self, self.w);
                }
            }
            None => {
                delta += o_self.len() as f64 * self.w.missing_edge;
            }
        }

        // Paires avec les nœuds déjà décidés v < u.
        for v in 0..u {
            let pv_opt = current[v];
            let o_uv = self.obs.rels_between(u, v);
            let o_vu = self.obs.rels_between(v, u);

            match (pu_opt, pv_opt) {
                (Some(pu), Some(pv)) => {
                    let p_uv = self.pred.rels_between(pu, pv);
                    let p_vu = self.pred.rels_between(pv, pu);
                    if !o_uv.is_empty() || !p_uv.is_empty() {
                        delta += pair_edge_cost(o_uv, p_uv, self.w);
                    }
                    if !o_vu.is_empty() || !p_vu.is_empty() {
                        delta += pair_edge_cost(o_vu, p_vu, self.w);
                    }
                }
                _ => {
                    delta += (o_uv.len() + o_vu.len()) as f64 * self.w.missing_edge;
                }
            }
        }
        delta
    }

    fn search(
        &mut self,
        u: usize,
        acc_cost: f64,
        current: &mut [Option<NodeId>],
        pred_used: &mut [bool],
    ) {
        if self.budget_left == 0 {
            return;
        }
        self.budget_left -= 1;

        if acc_cost + self.remaining_node_lower_bound(u, pred_used) >= self.best_cost - 1e-12 {
            return;
        }

        if u == self.obs.nodes.len() {
            // Ajoute les nœuds excédentaires et arêtes excédentaires incidentes à au moins un nœud non aligné.
            let total = evaluate_alignment_cost(self.obs, self.pred, current, self.w);
            if total < self.best_cost - 1e-12 {
                self.best_cost = total;
                self.best_map = current.to_vec();
            }
            return;
        }

        // Ordonnancement des candidats : même symbole d'abord, puis autres, puis None.
        let mut candidates: Vec<NodeId> = (0..self.pred.nodes.len())
            .filter(|&pu| !pred_used[pu])
            .collect();
        let target_sym = self.obs.nodes[u];
        candidates.sort_by_key(|&pu| usize::from(self.pred.nodes[pu] != target_sym));

        for pu in candidates {
            pred_used[pu] = true;
            current[u] = Some(pu);
            let node_delta = if self.pred.nodes[pu] == target_sym {
                0.0
            } else {
                self.w.conflict_node
            };
            let edge_delta = self.incremental_edge_cost(u, current);
            self.search(u + 1, acc_cost + node_delta + edge_delta, current, pred_used);
            current[u] = None;
            pred_used[pu] = false;
            if self.best_cost == 0.0 || self.budget_left == 0 {
                return;
            }
        }

        // Option : laisser u non aligné (`Missing`).
        current[u] = None;
        let edge_delta = self.incremental_edge_cost(u, current);
        self.search(
            u + 1,
            acc_cost + self.w.missing_node + edge_delta,
            current,
            pred_used,
        );
    }
}

/// Heuristique déterministe multi-graines (affinité WL + Seed-and-Extend) suivie d'un 2-opt exact.
fn heuristic_align(
    obs: &IndexedGraph,
    pred: &IndexedGraph,
    w: ResidualWeights,
) -> (Vec<Option<NodeId>>, f64) {
    let n_obs = obs.nodes.len();
    let n_pred = pred.nodes.len();

    // 1. Matrice d'affinité locale (symbole + voisinage entrant/sortant).
    let mut affinity = vec![vec![0.0f64; n_pred]; n_obs];
    for u in 0..n_obs {
        for pu in 0..n_pred {
            let mut score = if obs.nodes[u] == pred.nodes[pu] {
                3.0
            } else {
                0.2
            };
            // Chevauchement des voisins sortants (rel, symbole).
            let mut o_out: Vec<(RelId, SymbolId)> = obs.out_adj[u]
                .iter()
                .map(|&(v, r)| (r, obs.nodes[v]))
                .collect();
            let mut p_out: Vec<(RelId, SymbolId)> = pred.out_adj[pu]
                .iter()
                .map(|&(v, r)| (r, pred.nodes[v]))
                .collect();
            score += multiset_overlap(&mut o_out, &mut p_out) as f64 * 1.5;

            let mut o_in: Vec<(RelId, SymbolId)> = obs.in_adj[u]
                .iter()
                .map(|&(v, r)| (r, obs.nodes[v]))
                .collect();
            let mut p_in: Vec<(RelId, SymbolId)> = pred.in_adj[pu]
                .iter()
                .map(|&(v, r)| (r, pred.nodes[v]))
                .collect();
            score += multiset_overlap(&mut o_in, &mut p_in) as f64 * 1.5;

            affinity[u][pu] = score;
        }
    }

    let mut best_map = vec![None; n_obs];
    let mut best_cost = evaluate_alignment_cost(obs, pred, &best_map, w);

    // Candidat A : assignation gloutonne par affinité décroissante.
    let mut pairs: Vec<(f64, NodeId, NodeId)> = Vec::with_capacity(n_obs * n_pred);
    for (u, row) in affinity.iter().enumerate() {
        for (pu, &s) in row.iter().enumerate() {
            pairs.push((s, u, pu));
        }
    }
    pairs.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    let mut greedy_map = vec![None; n_obs];
    let mut used_p = vec![false; n_pred];
    for &(_, u, pu) in &pairs {
        if greedy_map[u].is_none() && !used_p[pu] {
            greedy_map[u] = Some(pu);
            used_p[pu] = true;
        }
    }
    let greedy_cost = refine_local_search(obs, pred, &mut greedy_map, w);
    if greedy_cost < best_cost {
        best_cost = greedy_cost;
        best_map = greedy_map;
    }

    // Candidat B : Seed-and-Extend depuis les meilleures ancres structurelles.
    let max_anchors = 16.min(pairs.len());
    for &(_, anchor_u, anchor_pu) in pairs.iter().take(max_anchors) {
        if best_cost == 0.0 {
            break;
        }
        let mut map = seed_and_extend(obs, pred, anchor_u, anchor_pu, &affinity);
        let cost = refine_local_search(obs, pred, &mut map, w);
        if cost < best_cost {
            best_cost = cost;
            best_map = map;
        }
    }

    (best_map, best_cost)
}

fn multiset_overlap(a: &mut [(RelId, SymbolId)], b: &mut [(RelId, SymbolId)]) -> usize {
    a.sort_unstable();
    b.sort_unstable();
    let (mut i, mut j, mut count) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    count
}

/// Propage un alignement par BFS à partir d'une paire ancre `(anchor_u, anchor_pu)`.
fn seed_and_extend(
    obs: &IndexedGraph,
    pred: &IndexedGraph,
    anchor_u: NodeId,
    anchor_pu: NodeId,
    affinity: &[Vec<f64>],
) -> Vec<Option<NodeId>> {
    let n_obs = obs.nodes.len();
    let n_pred = pred.nodes.len();
    let mut map = vec![None; n_obs];
    let mut used_p = vec![false; n_pred];
    let mut frontier = std::collections::VecDeque::new();

    map[anchor_u] = Some(anchor_pu);
    used_p[anchor_pu] = true;
    frontier.push_back((anchor_u, anchor_pu));

    while let Some((u, pu)) = frontier.pop_front() {
        // Extension le long des arêtes sortantes de même relation.
        for &(v, rel) in &obs.out_adj[u] {
            if map[v].is_some() {
                continue;
            }
            let best_pv = pred.out_adj[pu]
                .iter()
                .filter(|&&(pv, prel)| prel == rel && !used_p[pv])
                .max_by(|&&(pv1, _), &&(pv2, _)| {
                    affinity[v][pv1]
                        .total_cmp(&affinity[v][pv2])
                        .then_with(|| pv2.cmp(&pv1))
                })
                .map(|&(pv, _)| pv);
            if let Some(pv) = best_pv {
                map[v] = Some(pv);
                used_p[pv] = true;
                frontier.push_back((v, pv));
            }
        }
        // Extension le long des arêtes entrantes de même relation.
        for &(v, rel) in &obs.in_adj[u] {
            if map[v].is_some() {
                continue;
            }
            let best_pv = pred.in_adj[pu]
                .iter()
                .filter(|&&(pv, prel)| prel == rel && !used_p[pv])
                .max_by(|&&(pv1, _), &&(pv2, _)| {
                    affinity[v][pv1]
                        .total_cmp(&affinity[v][pv2])
                        .then_with(|| pv2.cmp(&pv1))
                })
                .map(|&(pv, _)| pv);
            if let Some(pv) = best_pv {
                map[v] = Some(pv);
                used_p[pv] = true;
                frontier.push_back((v, pv));
            }
        }
    }

    // Complète les composantes isolées restantes par affinité gloutonne.
    for u in 0..n_obs {
        if map[u].is_some() {
            continue;
        }
        let best_pv = (0..n_pred)
            .filter(|&pv| !used_p[pv])
            .max_by(|&a, &b| affinity[u][a].total_cmp(&affinity[u][b]).then_with(|| b.cmp(&a)));
        if let Some(pv) = best_pv {
            map[u] = Some(pv);
            used_p[pv] = true;
        }
    }

    map
}

/// Descente locale exacte (échanges 2-opt, réassignations, désalignements) sur le coût d'édition.
fn refine_local_search(
    obs: &IndexedGraph,
    pred: &IndexedGraph,
    map: &mut [Option<NodeId>],
    w: ResidualWeights,
) -> f64 {
    let n_obs = obs.nodes.len();
    let n_pred = pred.nodes.len();
    let mut best_cost = evaluate_alignment_cost(obs, pred, map, w);

    loop {
        let mut improved = false;

        // 1. Échange entre deux nœuds observés (u1, u2).
        for u1 in 0..n_obs {
            for u2 in (u1 + 1)..n_obs {
                if map[u1] == map[u2] {
                    continue;
                }
                map.swap(u1, u2);
                let c = evaluate_alignment_cost(obs, pred, map, w);
                if c < best_cost - 1e-12 {
                    best_cost = c;
                    improved = true;
                } else {
                    map.swap(u1, u2);
                }
            }
        }

        // 2. Désalignement ou réassignation vers un nœud prédit libre.
        let mut used_p = vec![false; n_pred];
        for &opt in map.iter() {
            if let Some(pu) = opt {
                used_p[pu] = true;
            }
        }
        for u in 0..n_obs {
            let old = map[u];
            // Test de désalignement (None).
            if let Some(pu) = old {
                map[u] = None;
                let c = evaluate_alignment_cost(obs, pred, map, w);
                if c < best_cost - 1e-12 {
                    best_cost = c;
                    used_p[pu] = false;
                    improved = true;
                    continue;
                } else {
                    map[u] = old;
                }
            }
            // Test vers chaque nœud prédit libre.
            for pu in 0..n_pred {
                if used_p[pu] {
                    continue;
                }
                map[u] = Some(pu);
                let c = evaluate_alignment_cost(obs, pred, map, w);
                if c < best_cost - 1e-12 {
                    best_cost = c;
                    if let Some(old_pu) = old {
                        used_p[old_pu] = false;
                    }
                    used_p[pu] = true;
                    improved = true;
                    break;
                } else {
                    map[u] = old;
                }
            }
        }

        if !improved || best_cost == 0.0 {
            break;
        }
    }

    best_cost
}

// ============================================================================
// Tests unitaires complets
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypothesis_graph::{AddNodeKind, Edge, ForwardAction, GraphConfig};

    fn apply_seq(mut g: HypothesisGraph, actions: &[ForwardAction]) -> HypothesisGraph {
        for a in actions {
            g = g.apply_forward(a).expect("Action P_F valide");
        }
        g
    }

    #[test]
    fn test_projection_concrete_expansion_and_zero_residual() {
        // SEED : a -> b ; Règle : a => c -> d ; Attendu E_hat : c -> d -> b
        let cfg = GraphConfig::new(vec![b'a', b'b', b'c', b'd'], 1, 12, 2, 16).unwrap();
        let h = apply_seq(
            HypothesisGraph::new(cfg),
            &[
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 1
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 2
                ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(0, 2, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(1, 2, EdgeKind::Struct(0))),
                ForwardAction::AddNode(AddNodeKind::Rule),       // 3
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 4 (LHS)
                ForwardAction::AddNode(AddNodeKind::Conc(b'c')), // 5 (RHS)
                ForwardAction::AddNode(AddNodeKind::Conc(b'd')), // 6 (RHS)
                ForwardAction::Connect(Edge::new(3, 4, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(3, 5, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(3, 6, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(5, 6, EdgeKind::Struct(0))),
            ],
        );

        let obs = ObservationGraph::from_bytes(b"cdb");
        let (out, res) =
            project_and_residual(&h, &obs, ProjectorConfig::default(), ResidualWeights::default());

        assert_eq!(out.termination, TerminationReason::NormalForm);
        assert_eq!(out.steps_executed, 1);
        assert_eq!(out.skipped_rules, 0);
        assert!(res.is_zero(), "Résidu attendu nul, obtenu : {:?}", res.summary);
        assert_eq!(res.norm, 0.0);
        assert_eq!(res.summary.matched_nodes, 3);
        assert_eq!(res.summary.matched_edges, 2);
    }

    #[test]
    fn test_projection_var_duplication_and_swap_with_same_as() {
        // 1) Duplication abstraite via SAME_AS : SEED = m -> a -> y ; Règle : (m -> X) => (X -> X)
        // Résultat attendu : a -> a -> y
        let cfg = GraphConfig::new(vec![b'm', b'a', b'y'], 1, 14, 2, 18).unwrap();
        let h_dup = apply_seq(
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
                ForwardAction::AddNode(AddNodeKind::Rule),       // 4
                ForwardAction::AddNode(AddNodeKind::Conc(b'm')), // 5 (LHS)
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 6 (LHS -> VAR X)
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 7 (RHS -> VAR X)
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 8 (RHS -> VAR X)
                ForwardAction::Connect(Edge::new(4, 5, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(4, 6, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(5, 6, EdgeKind::Struct(0))),
                ForwardAction::Connect(Edge::new(4, 7, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(4, 8, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(7, 8, EdgeKind::Struct(0))),
                ForwardAction::Abstract(6),
                ForwardAction::Abstract(7),
                ForwardAction::Abstract(8),
                // Fermeture transitive : 6 == 7 et 7 == 8
                ForwardAction::Connect(Edge::new(6, 7, EdgeKind::SameAs)),
                ForwardAction::Connect(Edge::new(7, 8, EdgeKind::SameAs)),
            ],
        );

        let obs_dup = ObservationGraph::from_bytes(b"aay");
        let (out_dup, res_dup) = project_and_residual(
            &h_dup,
            &obs_dup,
            ProjectorConfig::default(),
            ResidualWeights::default(),
        );
        assert_eq!(out_dup.termination, TerminationReason::NormalForm);
        assert_eq!(out_dup.steps_executed, 1);
        assert!(res_dup.is_zero(), "Duplication échouée : {:?}", res_dup);

        // 2) Permutation (swap) via SAME_AS : SEED = p -> m -> a -> b -> q
        // Règle : (m -> X -> Y) => (Y -> X)  =>  Attendu : p -> b -> a -> q
        let cfg2 = GraphConfig::new(vec![b'p', b'm', b'a', b'b', b'q'], 1, 16, 2, 24).unwrap();
        let h_swap = apply_seq(
            HypothesisGraph::new(cfg2),
            &[
                ForwardAction::AddNode(AddNodeKind::Conc(b'p')), // 1
                ForwardAction::AddNode(AddNodeKind::Conc(b'm')), // 2
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 3
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 4
                ForwardAction::AddNode(AddNodeKind::Conc(b'q')), // 5
                ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(0, 2, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(0, 3, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(0, 4, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(0, 5, EdgeKind::InSeed)),
                ForwardAction::Connect(Edge::new(1, 2, EdgeKind::Struct(0))),
                ForwardAction::Connect(Edge::new(2, 3, EdgeKind::Struct(0))),
                ForwardAction::Connect(Edge::new(3, 4, EdgeKind::Struct(0))),
                ForwardAction::Connect(Edge::new(4, 5, EdgeKind::Struct(0))),
                ForwardAction::AddNode(AddNodeKind::Rule),       // 6
                ForwardAction::AddNode(AddNodeKind::Conc(b'm')), // 7 (LHS)
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 8 (LHS VAR X)
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 9 (LHS VAR Y)
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 10 (RHS VAR Y)
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 11 (RHS VAR X)
                ForwardAction::Connect(Edge::new(6, 7, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(6, 8, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(6, 9, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(7, 8, EdgeKind::Struct(0))),
                ForwardAction::Connect(Edge::new(8, 9, EdgeKind::Struct(0))),
                ForwardAction::Connect(Edge::new(6, 10, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(6, 11, EdgeKind::InRhs)),
                ForwardAction::Connect(Edge::new(10, 11, EdgeKind::Struct(0))),
                ForwardAction::Abstract(8),
                ForwardAction::Abstract(9),
                ForwardAction::Abstract(10),
                ForwardAction::Abstract(11),
                ForwardAction::Connect(Edge::new(8, 11, EdgeKind::SameAs)), // X
                ForwardAction::Connect(Edge::new(9, 10, EdgeKind::SameAs)), // Y
            ],
        );

        let obs_swap = ObservationGraph::from_bytes(b"pbaq");
        let (out_swap, res_swap) = project_and_residual(
            &h_swap,
            &obs_swap,
            ProjectorConfig::default(),
            ResidualWeights::default(),
        );
        assert_eq!(out_swap.termination, TerminationReason::NormalForm);
        assert!(res_swap.is_zero(), "Swap échoué : {:?}", res_swap);
    }

    #[test]
    fn test_partial_hypothesis_and_cycle_guard() {
        // 1. Hypothèse partielle : une règle vide + une règle avec VAR non lié en RHS.
        let cfg = GraphConfig::new(vec![b'a', b'b'], 1, 10, 3, 12).unwrap();
        let h_partial = apply_seq(
            HypothesisGraph::new(cfg.clone()),
            &[
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 1
                ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
                ForwardAction::AddNode(AddNodeKind::Rule),       // 2 (LHS vide)
                ForwardAction::AddNode(AddNodeKind::Rule),       // 3 (RHS VAR libre)
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 4 (LHS)
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 5 (RHS -> VAR non lié)
                ForwardAction::Connect(Edge::new(3, 4, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(3, 5, EdgeKind::InRhs)),
                ForwardAction::Abstract(5),
            ],
        );
        let out_partial = project(&h_partial, ProjectorConfig::default());
        assert_eq!(out_partial.skipped_rules, 2);
        assert_eq!(out_partial.predicted.nodes, vec![b'a']);
        assert_eq!(out_partial.termination, TerminationReason::NormalForm);

        // 2. Garde anti-cycle : a => b et b => a.
        let h_cycle = apply_seq(
            HypothesisGraph::new(cfg),
            &[
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 1
                ForwardAction::Connect(Edge::new(0, 1, EdgeKind::InSeed)),
                ForwardAction::AddNode(AddNodeKind::Rule),       // 2 : a => b
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 3
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 4
                ForwardAction::Connect(Edge::new(2, 3, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(2, 4, EdgeKind::InRhs)),
                ForwardAction::AddNode(AddNodeKind::Rule),       // 5 : b => a
                ForwardAction::AddNode(AddNodeKind::Conc(b'b')), // 6
                ForwardAction::AddNode(AddNodeKind::Conc(b'a')), // 7
                ForwardAction::Connect(Edge::new(5, 6, EdgeKind::InLhs)),
                ForwardAction::Connect(Edge::new(5, 7, EdgeKind::InRhs)),
            ],
        );
        let out_cycle = project(&h_cycle, ProjectorConfig::default());
        assert_eq!(out_cycle.termination, TerminationReason::CycleDetected);
        assert_eq!(out_cycle.steps_executed, 2);
    }

    #[test]
    fn test_residual_classes_and_permutation_invariance() {
        let w = ResidualWeights::default();

        // Cas 1 : Manquant (G_E = "abc", E_hat = "bc")
        let res_miss = compute_residual(
            &ObservationGraph::from_bytes(b"abc"),
            &ObservationGraph::from_bytes(b"bc"),
            w,
        );
        assert_eq!(res_miss.summary.matched_nodes, 2);
        assert_eq!(res_miss.summary.missing_nodes, 1);
        assert_eq!(res_miss.summary.excess_nodes, 0);
        assert_eq!(res_miss.summary.conflict_nodes, 0);
        assert_eq!(res_miss.summary.missing_edges, 1);
        assert_eq!(res_miss.missing_nodes, vec![(0, b'a')]);

        // Cas 2 : Excédentaire (G_E = "bc", E_hat = "abc")
        let res_exc = compute_residual(
            &ObservationGraph::from_bytes(b"bc"),
            &ObservationGraph::from_bytes(b"abc"),
            w,
        );
        assert_eq!(res_exc.summary.matched_nodes, 2);
        assert_eq!(res_exc.summary.missing_nodes, 0);
        assert_eq!(res_exc.summary.excess_nodes, 1);
        assert_eq!(res_exc.summary.excess_edges, 1);
        assert_eq!(res_exc.excess_nodes, vec![(0, b'a')]);

        // Cas 3 : Conflit de symbole localisé (G_E = "axc", E_hat = "ayc")
        let res_conf_n = compute_residual(
            &ObservationGraph::from_bytes(b"axc"),
            &ObservationGraph::from_bytes(b"ayc"),
            w,
        );
        assert_eq!(res_conf_n.summary.matched_nodes, 2);
        assert_eq!(res_conf_n.summary.conflict_nodes, 1);
        assert_eq!(res_conf_n.summary.matched_edges, 2);
        assert_eq!(res_conf_n.summary.missing_nodes, 0);
        assert_eq!(res_conf_n.summary.excess_nodes, 0);

        // Cas 4 : Conflit de type de relation d'arête (rel 0 vs rel 1)
        let obs_rel = ObservationGraph::new(vec![b'a', b'b'], [ObsEdge::new(0, 1, 0)]);
        let pred_rel = ObservationGraph::new(vec![b'a', b'b'], [ObsEdge::new(0, 1, 1)]);
        let res_conf_e = compute_residual(&obs_rel, &pred_rel, w);
        assert_eq!(res_conf_e.summary.matched_nodes, 2);
        assert_eq!(res_conf_e.summary.conflict_edges, 1);
        assert_eq!(res_conf_e.summary.missing_edges, 0);
        assert_eq!(res_conf_e.summary.excess_edges, 0);

        // Cas 5 : Invariance du résidu par permutation des IDs de nœuds dans E_hat
        // E_hat = "ayc" réindexé comme [0:'c', 1:'a', 2:'y'] avec arêtes 1->2 et 2->0.
        let pred_perm = ObservationGraph::new(
            vec![b'c', b'a', b'y'],
            [ObsEdge::new(1, 2, 0), ObsEdge::new(2, 0, 0)],
        );
        let res_perm = compute_residual(&ObservationGraph::from_bytes(b"axc"), &pred_perm, w);
        assert_eq!(res_perm.summary, res_conf_n.summary);
        assert!((res_perm.norm - res_conf_n.norm).abs() < 1e-12);
    }

    #[test]
    fn test_larger_graph_heuristic_alignment() {
        // 16 nœuds (> seuil exact 9) avec 1 nœud manquant en tête et 1 conflit interne.
        let obs = ObservationGraph::from_bytes(b"0123456789abcdef");
        let pred = ObservationGraph::from_bytes(b"1234567X89abcdef");
        let res = compute_residual(&obs, &pred, ResidualWeights::default());

        assert_eq!(res.summary.missing_nodes, 1);
        assert_eq!(res.summary.excess_nodes, 1);
        assert_eq!(res.summary.matched_nodes, 15);
    }
}