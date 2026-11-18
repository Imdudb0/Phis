use crate::rng::Rng;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SymbolId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PatSymbol {
    Concrete(SymbolId),
    SymVar(usize),
}

#[derive(Clone, Debug)]
pub struct Signature {
    pub arities: Vec<usize>,
}

impl Signature {
    pub fn sample(rng: &mut Rng, num_symbols: usize) -> Self {
        assert!(num_symbols >= 4, "Signature requires at least 4 symbols");
        let mut arities = vec![0usize; num_symbols];
        let half = num_symbols / 2;
        for (start, end) in [(0, half), (half, num_symbols)] {
            let span = end - start;
            let num_leaves = (span / 3).max(1).min(span - 1);
            let mut indices: Vec<usize> = (start..end).collect();
            rng.shuffle(&mut indices);
            for &idx in &indices[..num_leaves] {
                arities[idx] = 0;
            }
            for &idx in &indices[num_leaves..] {
                arities[idx] = rng.range_usize(1, 4); // arités 1, 2 ou 3
            }
        }
        Self { arities }
    }

    pub fn arity(&self, s: SymbolId) -> usize {
        self.arities[s.0]
    }
}

#[derive(Clone, Debug)]
pub struct TermNode {
    pub sym: SymbolId,
    pub args: Vec<NodeId>,
}

#[derive(Clone, Debug)]
pub struct TermGraph {
    pub nodes: Vec<TermNode>,
    pub root: NodeId,
}

impl TermGraph {
    pub fn new_empty(root_sym: SymbolId) -> Self {
        Self {
            nodes: vec![TermNode {
                sym: root_sym,
                args: Vec::new(),
            }],
            root: NodeId(0),
        }
    }

    pub fn add_node(&mut self, sym: SymbolId, args: Vec<NodeId>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(TermNode { sym, args });
        id
    }

    pub fn reachable_from(&self, start: NodeId, limit: usize) -> Vec<NodeId> {
        let mut visited = HashSet::new();
        let mut stack = vec![start];
        let mut order = Vec::new();
        while let Some(nid) = stack.pop() {
            if !visited.insert(nid) {
                continue;
            }
            order.push(nid);
            if order.len() >= limit {
                break;
            }
            for &arg in self.nodes[nid.0].args.iter().rev() {
                stack.push(arg);
            }
        }
        order
    }

    pub fn extract_subgraph(&self, start: NodeId, max_nodes: usize) -> TermGraph {
        let order = self.reachable_from(start, max_nodes);
        let mut map = HashMap::new();
        for (new_idx, &old_id) in order.iter().enumerate() {
            map.insert(old_id, NodeId(new_idx));
        }
        let mut nodes = Vec::with_capacity(order.len());
        for &old_id in &order {
            let old = &self.nodes[old_id.0];
            let mut new_args = Vec::with_capacity(old.args.len());
            let mut valid = true;
            for &a in &old.args {
                if let Some(&mapped) = map.get(&a) {
                    new_args.push(mapped);
                } else {
                    valid = false;
                    break;
                }
            }
            nodes.push(TermNode {
                sym: old.sym,
                args: if valid { new_args } else { Vec::new() },
            });
        }
        TermGraph {
            nodes,
            root: NodeId(0),
        }
    }

    pub fn sample_random(
        rng: &mut Rng,
        sig: &Signature,
        pool: &[SymbolId],
        max_depth: usize,
        allow_sharing: bool,
    ) -> Self {
        let leaves: Vec<SymbolId> = pool
            .iter()
            .copied()
            .filter(|&s| sig.arity(s) == 0)
            .collect();
        let internals: Vec<SymbolId> = pool
            .iter()
            .copied()
            .filter(|&s| sig.arity(s) > 0)
            .collect();
        assert!(!leaves.is_empty(), "pool must contain at least one 0-ary leaf symbol");

        let mut nodes: Vec<TermNode> = Vec::new();

        fn build(
            rng: &mut Rng,
            sig: &Signature,
            leaves: &[SymbolId],
            internals: &[SymbolId],
            nodes: &mut Vec<TermNode>,
            depth: usize,
            max_depth: usize,
            allow_sharing: bool,
        ) -> NodeId {
            // Partage de sous-terme réel (DAG) si autorisé et nœuds existants
            if allow_sharing && !nodes.is_empty() && depth > 0 && rng.bernoulli(0.22) {
                let idx = rng.index(nodes.len());
                return NodeId(idx);
            }
            let sym = if depth >= max_depth || internals.is_empty() || (depth > 0 && rng.bernoulli(0.3)) {
                *rng.choose(leaves)
            } else {
                *rng.choose(internals)
            };
            let arity = sig.arity(sym);
            let mut args = Vec::with_capacity(arity);
            for _ in 0..arity {
                args.push(build(
                    rng,
                    sig,
                    leaves,
                    internals,
                    nodes,
                    depth + 1,
                    max_depth,
                    allow_sharing,
                ));
            }
            let id = NodeId(nodes.len());
            nodes.push(TermNode { sym, args });
            id
        }

        let root = build(rng, sig, &leaves, &internals, &mut nodes, 0, max_depth, allow_sharing);
        TermGraph { nodes, root }
    }
}

#[derive(Clone, Debug)]
pub enum PatNodeKind {
    Var(usize),
    App {
        sym: PatSymbol,
        args: Vec<usize>,
    },
}

#[derive(Clone, Debug)]
pub struct PatternGraph {
    pub nodes: Vec<PatNodeKind>,
    pub root: usize,
}

#[derive(Clone, Debug)]
pub struct RewriteRule {
    pub id: usize,
    pub lhs: PatternGraph,
    pub rhs: PatternGraph,
    pub weight: f32,
    pub enabled: bool,
}

#[derive(Clone, Debug)]
pub struct MatchBinding {
    pub root_node: NodeId,
    pub node_vars: HashMap<usize, NodeId>,
    pub sym_vars: HashMap<usize, SymbolId>,
    pub matched_nodes: Vec<NodeId>,
}

impl PatternGraph {
    pub fn sample_rule_pair(
        rng: &mut Rng,
        sig: &Signature,
        pool: &[SymbolId],
        polymorphic: bool,
    ) -> (PatternGraph, PatternGraph) {
        let leaves: Vec<SymbolId> = pool
            .iter()
            .copied()
            .filter(|&s| sig.arity(s) == 0)
            .collect();
        let internals: Vec<SymbolId> = pool
            .iter()
            .copied()
            .filter(|&s| sig.arity(s) > 0)
            .collect();
        assert!(!leaves.is_empty());

        let mut lhs_nodes: Vec<PatNodeKind> = Vec::new();
        let mut bound_vars: Vec<usize> = Vec::new();
        let mut bound_sym_vars: Vec<(usize, usize)> = Vec::new(); // (sym_var_id, arity)

        let mut next_var = 0usize;
        let mut next_sym_var = 0usize;

        // LHS : profondeur 1 à 2 pour garantir des déclenchements riches
        let root_sym = if !internals.is_empty() {
            *rng.choose(&internals)
        } else {
            *rng.choose(&leaves)
        };
        let root_arity = sig.arity(root_sym);
        let root_pat_sym = if polymorphic && rng.bernoulli(0.5) {
            let sv = next_sym_var;
            next_sym_var += 1;
            bound_sym_vars.push((sv, root_arity));
            PatSymbol::SymVar(sv)
        } else {
            PatSymbol::Concrete(root_sym)
        };

        let mut root_args = Vec::with_capacity(root_arity);
        for _ in 0..root_arity {
            // Soit une variable de nœud (réutilisable -> partage!), soit un sous-motif profondeur 1
            if !bound_vars.is_empty() && rng.bernoulli(0.25) {
                let v = *rng.choose(&bound_vars);
                let idx = lhs_nodes.len();
                lhs_nodes.push(PatNodeKind::Var(v));
                root_args.push(idx);
            } else if !internals.is_empty() && rng.bernoulli(0.4) {
                let child_sym = *rng.choose(&internals);
                let child_arity = sig.arity(child_sym);
                let child_pat_sym = if polymorphic && rng.bernoulli(0.45) {
                    let sv = next_sym_var;
                    next_sym_var += 1;
                    bound_sym_vars.push((sv, child_arity));
                    PatSymbol::SymVar(sv)
                } else {
                    PatSymbol::Concrete(child_sym)
                };
                let mut cargs = Vec::with_capacity(child_arity);
                for _ in 0..child_arity {
                    let v = next_var;
                    next_var += 1;
                    bound_vars.push(v);
                    let idx = lhs_nodes.len();
                    lhs_nodes.push(PatNodeKind::Var(v));
                    cargs.push(idx);
                }
                let cidx = lhs_nodes.len();
                lhs_nodes.push(PatNodeKind::App {
                    sym: child_pat_sym,
                    args: cargs,
                });
                root_args.push(cidx);
            } else {
                let v = next_var;
                next_var += 1;
                bound_vars.push(v);
                let idx = lhs_nodes.len();
                lhs_nodes.push(PatNodeKind::Var(v));
                root_args.push(idx);
            }
        }
        let lhs_root = lhs_nodes.len();
        lhs_nodes.push(PatNodeKind::App {
            sym: root_pat_sym,
            args: root_args,
        });
        let lhs = PatternGraph {
            nodes: lhs_nodes,
            root: lhs_root,
        };

        // RHS : construit en réutilisant variables de nœuds/symboles liées ou nouveaux symboles du pool
        let mut rhs_nodes: Vec<PatNodeKind> = Vec::new();
        let rhs_root = build_rhs(
            rng,
            sig,
            pool,
            &leaves,
            &internals,
            &bound_vars,
            &bound_sym_vars,
            &mut rhs_nodes,
            0,
            2,
        );
        let rhs = PatternGraph {
            nodes: rhs_nodes,
            root: rhs_root,
        };
        (lhs, rhs)
    }

    pub fn match_at(&self, graph: &TermGraph, target: NodeId, sig: &Signature) -> Option<MatchBinding> {
        let mut node_vars = HashMap::new();
        let mut sym_vars = HashMap::new();
        let mut matched_nodes = Vec::new();

        fn rec(
            pat: &PatternGraph,
            pat_idx: usize,
            graph: &TermGraph,
            g_id: NodeId,
            sig: &Signature,
            node_vars: &mut HashMap<usize, NodeId>,
            sym_vars: &mut HashMap<usize, SymbolId>,
            matched_nodes: &mut Vec<NodeId>,
            depth: usize,
        ) -> bool {
            if depth > 32 {
                return false;
            }
            match &pat.nodes[pat_idx] {
                PatNodeKind::Var(v) => {
                    if let Some(&prev) = node_vars.get(v) {
                        prev == g_id
                    } else {
                        node_vars.insert(*v, g_id);
                        matched_nodes.push(g_id);
                        true
                    }
                }
                PatNodeKind::App { sym, args } => {
                    let g_node = &graph.nodes[g_id.0];
                    if g_node.args.len() != args.len() {
                        return false;
                    }
                    match sym {
                        PatSymbol::Concrete(cs) => {
                            if g_node.sym != *cs {
                                return false;
                            }
                        }
                        PatSymbol::SymVar(sv) => {
                            if let Some(&prev_sym) = sym_vars.get(sv) {
                                if prev_sym != g_node.sym {
                                    return false;
                                }
                            } else {
                                if sig.arity(g_node.sym) != args.len() {
                                    return false;
                                }
                                sym_vars.insert(*sv, g_node.sym);
                            }
                        }
                    }
                    matched_nodes.push(g_id);
                    for (&p_child, &g_child) in args.iter().zip(g_node.args.iter()) {
                        if !rec(
                            pat,
                            p_child,
                            graph,
                            g_child,
                            sig,
                            node_vars,
                            sym_vars,
                            matched_nodes,
                            depth + 1,
                        ) {
                            return false;
                        }
                    }
                    true
                }
            }
        }

        if rec(
            self,
            self.root,
            graph,
            target,
            sig,
            &mut node_vars,
            &mut sym_vars,
            &mut matched_nodes,
            0,
        ) {
            Some(MatchBinding {
                root_node: target,
                node_vars,
                sym_vars,
                matched_nodes,
            })
        } else {
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_rhs(
    rng: &mut Rng,
    sig: &Signature,
    _pool: &[SymbolId],
    leaves: &[SymbolId],
    internals: &[SymbolId],
    bound_vars: &[usize],
    bound_sym_vars: &[(usize, usize)],
    rhs_nodes: &mut Vec<PatNodeKind>,
    depth: usize,
    max_depth: usize,
) -> usize {
    if !bound_vars.is_empty() && (depth >= max_depth || (depth > 0 && rng.bernoulli(0.55))) {
        let v = *rng.choose(bound_vars);
        let idx = rhs_nodes.len();
        rhs_nodes.push(PatNodeKind::Var(v));
        return idx;
    }
    if depth >= max_depth || internals.is_empty() {
        if !bound_vars.is_empty() && rng.bernoulli(0.6) {
            let v = *rng.choose(bound_vars);
            let idx = rhs_nodes.len();
            rhs_nodes.push(PatNodeKind::Var(v));
            return idx;
        }
        let s = *rng.choose(leaves);
        let idx = rhs_nodes.len();
        rhs_nodes.push(PatNodeKind::App {
            sym: PatSymbol::Concrete(s),
            args: Vec::new(),
        });
        return idx;
    }

    // Choisir éventuellement un symbole variable lié d'arité > 0 ou un symbole interne concret
    let (pat_sym, arity) = if !bound_sym_vars.is_empty() && rng.bernoulli(0.4) {
        let &(sv, ar) = rng.choose(bound_sym_vars);
        (PatSymbol::SymVar(sv), ar)
    } else {
        let s = *rng.choose(internals);
        (PatSymbol::Concrete(s), sig.arity(s))
    };

    let mut args = Vec::with_capacity(arity);
    for _ in 0..arity {
        args.push(build_rhs(
            rng,
            sig,
            _pool,
            leaves,
            internals,
            bound_vars,
            bound_sym_vars,
            rhs_nodes,
            depth + 1,
            max_depth,
        ));
    }
    let idx = rhs_nodes.len();
    rhs_nodes.push(PatNodeKind::App { sym: pat_sym, args });
    idx
}

#[derive(Clone, Debug)]
pub struct TraceStep {
    pub step: usize,
    pub rule_id: usize,
    pub target_node: NodeId,
    pub matched_nodes: Vec<NodeId>,
    pub new_root: NodeId,
    pub candidate_count: usize,
}

#[derive(Clone, Debug)]
pub struct DerivationResult {
    pub initial_graph: TermGraph,
    pub final_graph: TermGraph,
    pub steps: Vec<TraceStep>,
}

pub fn instantiate_rhs(
    rhs: &PatternGraph,
    binding: &MatchBinding,
    graph: &mut TermGraph,
    fallback_leaf: SymbolId,
) -> NodeId {
    let mut memo: HashMap<usize, NodeId> = HashMap::new();

    fn rec(
        rhs: &PatternGraph,
        idx: usize,
        binding: &MatchBinding,
        graph: &mut TermGraph,
        memo: &mut HashMap<usize, NodeId>,
        fallback_leaf: SymbolId,
    ) -> NodeId {
        if let Some(&existing) = memo.get(&idx) {
            return existing;
        }
        let res = match &rhs.nodes[idx] {
            PatNodeKind::Var(v) => binding
                .node_vars
                .get(v)
                .copied()
                .unwrap_or_else(|| graph.add_node(fallback_leaf, Vec::new())),
            PatNodeKind::App { sym, args } => {
                let concrete_sym = match sym {
                    PatSymbol::Concrete(s) => *s,
                    PatSymbol::SymVar(sv) => binding
                        .sym_vars
                        .get(sv)
                        .copied()
                        .unwrap_or(fallback_leaf),
                };
                let mut concrete_args = Vec::with_capacity(args.len());
                for &a in args {
                    concrete_args.push(rec(rhs, a, binding, graph, memo, fallback_leaf));
                }
                graph.add_node(concrete_sym, concrete_args)
            }
        };
        memo.insert(idx, res);
        res
    }

    rec(rhs, rhs.root, binding, graph, &mut memo, fallback_leaf)
}

pub fn run_ptgrs(
    rng: &mut Rng,
    sig: &Signature,
    initial: &TermGraph,
    rules: &[RewriteRule],
    max_steps: usize,
    max_nodes: usize,
) -> DerivationResult {
    let mut graph = initial.clone();
    let mut steps = Vec::new();
    let fallback_leaf = sig
        .arities
        .iter()
        .position(|&a| a == 0)
        .map(SymbolId)
        .unwrap_or(SymbolId(0));

    for step in 0..max_steps {
        if graph.nodes.len() >= max_nodes {
            break;
        }
        let active_nodes = graph.reachable_from(graph.root, max_nodes);
        let mut candidates: Vec<(usize, MatchBinding, f32)> = Vec::new();

        for rule in rules {
            if !rule.enabled || rule.weight <= 0.0 {
                continue;
            }
            for &nid in &active_nodes {
                if let Some(binding) = rule.lhs.match_at(&graph, nid, sig) {
                    candidates.push((rule.id, binding, rule.weight));
                }
            }
        }

        if candidates.is_empty() {
            break;
        }

        let total_w: f32 = candidates.iter().map(|c| c.2).sum();
        let mut draw = rng.next_f32_01() * total_w;
        let mut chosen_idx = candidates.len() - 1;
        for (i, c) in candidates.iter().enumerate() {
            draw -= c.2;
            if draw <= 0.0 {
                chosen_idx = i;
                break;
            }
        }

        let candidate_count = candidates.len();
        let (rule_id, binding, _) = candidates.swap_remove(chosen_idx);
        let rule = rules
            .iter()
            .find(|r| r.id == rule_id)
            .expect("rule_id must exist");

        let new_subroot = instantiate_rhs(&rule.rhs, &binding, &mut graph, fallback_leaf);

        // Rediriger tous les pointeurs vers binding.root_node (préserve le graphe de termes DAG)
        if binding.root_node == graph.root {
            graph.root = new_subroot;
        } else {
            let old_target = binding.root_node;
            for node in &mut graph.nodes {
                for arg in &mut node.args {
                    if *arg == old_target {
                        *arg = new_subroot;
                    }
                }
            }
        }

        steps.push(TraceStep {
            step,
            rule_id,
            target_node: binding.root_node,
            matched_nodes: binding.matched_nodes,
            new_root: new_subroot,
            candidate_count,
        });
    }

    let compacted = graph.extract_subgraph(graph.root, max_nodes);
    DerivationResult {
        initial_graph: initial.clone(),
        final_graph: compacted,
        steps,
    }
}

// 3) Remplacer `structural_similarity` :
pub fn structural_similarity(a: &TermGraph, b: &TermGraph) -> f32 {
    if a.nodes.is_empty() || b.nodes.is_empty() {
        return 0.0;
    }

    let a_reach = a.reachable_from(a.root, a.nodes.len());
    let b_reach = b.reachable_from(b.root, b.nodes.len());

    let mut node_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut rev_node_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut sym_map: HashMap<SymbolId, SymbolId> = HashMap::new();
    let mut rev_sym_map: HashMap<SymbolId, SymbolId> = HashMap::new();

    #[allow(clippy::too_many_arguments)]
    fn check_iso(
        a: &TermGraph,
        na: NodeId,
        b: &TermGraph,
        nb: NodeId,
        node_map: &mut HashMap<NodeId, NodeId>,
        rev_node_map: &mut HashMap<NodeId, NodeId>,
        sym_map: &mut HashMap<SymbolId, SymbolId>,
        rev_sym_map: &mut HashMap<SymbolId, SymbolId>,
        depth: usize,
    ) -> bool {
        if depth > 24 {
            return true;
        }
        if let Some(&prev_b) = node_map.get(&na) {
            return prev_b == nb;
        }
        if let Some(&prev_a) = rev_node_map.get(&nb) {
            if prev_a != na {
                return false;
            }
        }
        let ga = &a.nodes[na.0];
        let gb = &b.nodes[nb.0];
        if ga.args.len() != gb.args.len() {
            return false;
        }
        if let Some(&mapped_sb) = sym_map.get(&ga.sym) {
            if mapped_sb != gb.sym {
                return false;
            }
        } else if let Some(&mapped_sa) = rev_sym_map.get(&gb.sym) {
            if mapped_sa != ga.sym {
                return false;
            }
        } else {
            sym_map.insert(ga.sym, gb.sym);
            rev_sym_map.insert(gb.sym, ga.sym);
        }

        node_map.insert(na, nb);
        rev_node_map.insert(nb, na);
        for (&ca, &cb) in ga.args.iter().zip(gb.args.iter()) {
            if !check_iso(
                a,
                ca,
                b,
                cb,
                node_map,
                rev_node_map,
                sym_map,
                rev_sym_map,
                depth + 1,
            ) {
                return false;
            }
        }
        true
    }

    if check_iso(
        a,
        a.root,
        b,
        b.root,
        &mut node_map,
        &mut rev_node_map,
        &mut sym_map,
        &mut rev_sym_map,
        0,
    ) {
        return 1.0;
    }

    // Plongement structurel sur les nœuds atteignables uniquement (jamais les nœuds morts)
    let (small, small_len, large, large_reach) = if a_reach.len() <= b_reach.len() {
        (a, a_reach.len(), b, &b_reach)
    } else {
        (b, b_reach.len(), a, &a_reach)
    };
    if small_len >= 2 {
        for &nid in large_reach {
            node_map.clear();
            rev_node_map.clear();
            sym_map.clear();
            rev_sym_map.clear();
            if check_iso(
                small,
                small.root,
                large,
                nid,
                &mut node_map,
                &mut rev_node_map,
                &mut sym_map,
                &mut rev_sym_map,
                0,
            ) {
                return 0.75;
            }
        }
    }

    0.0
}

// 4) Remplacer le bloc #[cfg(test)] de ptgrs.rs par :
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptgrs_rewrites_and_alpha_homology() {
        let mut rng = Rng::from_seed(2024);
        let sig = Signature::sample(&mut rng, 12);
        let pool_a: Vec<SymbolId> = (0..6).map(SymbolId).collect();
        let pool_b: Vec<SymbolId> = (6..12).map(SymbolId).collect();
        assert!(pool_a.iter().any(|&s| sig.arity(s) == 0));
        assert!(pool_b.iter().any(|&s| sig.arity(s) == 0));

        let custom_sig = Signature {
            arities: vec![0, 0, 2, 0, 0, 2],
        };
        let mut ga = TermGraph::new_empty(SymbolId(0));
        let a_leaf = ga.add_node(SymbolId(1), vec![]);
        ga.root = ga.add_node(SymbolId(2), vec![a_leaf, a_leaf]);

        let mut gb = TermGraph::new_empty(SymbolId(3));
        let b_leaf = gb.add_node(SymbolId(4), vec![]);
        gb.root = gb.add_node(SymbolId(5), vec![b_leaf, b_leaf]);
        assert_eq!(structural_similarity(&ga, &gb), 1.0);

        // Nouveau graphe distinct avec deux feuilles différentes (sans nœuds morts)
        let mut gc = TermGraph::new_empty(SymbolId(3));
        let c_leaf1 = gc.add_node(SymbolId(4), vec![]);
        let c_leaf2 = gc.add_node(SymbolId(3), vec![]);
        gc.root = gc.add_node(SymbolId(5), vec![c_leaf1, c_leaf2]);
        assert_eq!(structural_similarity(&ga, &gc), 0.0);

        // Exécution PTGRS déterministe
        let (lhs, rhs) = PatternGraph::sample_rule_pair(&mut rng, &custom_sig, &pool_a[..3], true);
        let rule = RewriteRule {
            id: 0,
            lhs,
            rhs,
            weight: 1.0,
            enabled: true,
        };
        let res = run_ptgrs(&mut rng, &custom_sig, &ga, &[rule], 6, 32);
        assert!(!res.final_graph.nodes.is_empty());
    }
}