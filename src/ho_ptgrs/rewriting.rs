use crate::term_graph::{Node, SymbolId, TermGraph};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

/// Nœud d'un motif de règle DPO (interface K portée par les variables Var(id)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternNode {
    Var(u32),
    Sym(SymbolId, Vec<usize>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternGraph {
    pub nodes: Vec<PatternNode>,
    pub root: usize,
}

impl PatternGraph {
    pub fn builder() -> PatternBuilder {
        PatternBuilder { nodes: Vec::new() }
    }

    pub fn symbols(&self) -> BTreeSet<SymbolId> {
        self.nodes
            .iter()
            .filter_map(|n| match n {
                PatternNode::Sym(s, _) => Some(*s),
                PatternNode::Var(_) => None,
            })
            .collect()
    }
}

pub struct PatternBuilder {
    nodes: Vec<PatternNode>,
}

impl PatternBuilder {
    pub fn var(&mut self, id: u32) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(PatternNode::Var(id));
        idx
    }

    pub fn sym(&mut self, s: SymbolId, succ: Vec<usize>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(PatternNode::Sym(s, succ));
        idx
    }

    pub fn finish(self, root: usize) -> PatternGraph {
        PatternGraph {
            nodes: self.nodes,
            root,
        }
    }
}

/// Règle de réécriture DPO probabiliste de la Strate 0.
#[derive(Debug, Clone)]
pub struct Rule {
    pub id: u32,
    pub lhs: PatternGraph,
    pub rhs: PatternGraph,
    pub prob: f64,
}

impl Rule {
    pub fn symbols(&self) -> BTreeSet<SymbolId> {
        let mut set = self.lhs.symbols();
        set.extend(self.rhs.symbols());
        set
    }
}

/// Tente d'apparier `pat` enraciné au nœud `target` de `graph`.
fn match_pattern(
    pat: &PatternGraph,
    graph: &TermGraph,
    target: usize,
) -> Option<HashMap<u32, usize>> {
    let mut var_bindings: HashMap<u32, usize> = HashMap::new();
    let mut pat_bindings: HashMap<usize, usize> = HashMap::new();

    fn recurse(
        p_idx: usize,
        g_idx: usize,
        pat: &PatternGraph,
        graph: &TermGraph,
        var_bindings: &mut HashMap<u32, usize>,
        pat_bindings: &mut HashMap<usize, usize>,
    ) -> bool {
        if let Some(&bound) = pat_bindings.get(&p_idx) {
            return bound == g_idx;
        }
        pat_bindings.insert(p_idx, g_idx);

        match &pat.nodes[p_idx] {
            PatternNode::Var(vid) => {
                if let Some(&bound) = var_bindings.get(vid) {
                    bound == g_idx
                } else {
                    var_bindings.insert(*vid, g_idx);
                    true
                }
            }
            PatternNode::Sym(sym, p_children) => {
                let g_node = &graph.nodes[g_idx];
                if g_node.label != *sym || g_node.succ.len() != p_children.len() {
                    return false;
                }
                for (&pc, &gc) in p_children.iter().zip(g_node.succ.iter()) {
                    if !recurse(pc, gc, pat, graph, var_bindings, pat_bindings) {
                        return false;
                    }
                }
                true
            }
        }
    }

    if recurse(pat.root, target, pat, graph, &mut var_bindings, &mut pat_bindings) {
        Some(var_bindings)
    } else {
        None
    }
}

fn instantiate_rhs(
    r_idx: usize,
    rhs: &PatternGraph,
    var_bindings: &HashMap<u32, usize>,
    nodes: &mut Vec<Node>,
    rhs_map: &mut [Option<usize>],
) -> usize {
    if let Some(existing) = rhs_map[r_idx] {
        return existing;
    }
    match &rhs.nodes[r_idx] {
        PatternNode::Var(vid) => {
            let bound = var_bindings[vid];
            rhs_map[r_idx] = Some(bound);
            bound
        }
        PatternNode::Sym(sym, r_children) => {
            let slot = nodes.len();
            nodes.push(Node {
                label: *sym,
                succ: Vec::new(),
            });
            rhs_map[r_idx] = Some(slot);
            let mut child_ids = Vec::with_capacity(r_children.len());
            for &rc in r_children {
                child_ids.push(instantiate_rhs(rc, rhs, var_bindings, nodes, rhs_map));
            }
            nodes[slot].succ = child_ids;
            slot
        }
    }
}

/// Applique la règle `rule` à toutes les occurrences valides dans `graph`.
pub fn apply_rule_everywhere(graph: &TermGraph, rule: &Rule) -> Vec<TermGraph> {
    let mut results = Vec::new();
    for target in 0..graph.nodes.len() {
        if let Some(var_bindings) = match_pattern(&rule.lhs, graph, target) {
            let mut nodes = graph.nodes.clone();
            let orig_len = nodes.len();
            let mut rhs_map = vec![None; rule.rhs.nodes.len()];
            let new_subroot = instantiate_rhs(
                rule.rhs.root,
                &rule.rhs,
                &var_bindings,
                &mut nodes,
                &mut rhs_map,
            );

            for n in &mut nodes[..orig_len] {
                for child in &mut n.succ {
                    if *child == target {
                        *child = new_subroot;
                    }
                }
            }
            let new_root = if graph.root == target {
                new_subroot
            } else {
                graph.root
            };
            results.push(TermGraph { nodes, root: new_root }.canonicalize());
        }
    }
    results
}

/// Graphe de dérivation réifié à horizon k : D_k(G, R_0) = (U, A).
#[derive(Debug, Clone)]
pub struct DerivationGraph {
    pub states: Vec<TermGraph>,
    /// Arêtes dirigées (src_state, dst_state, rule_id).
    pub edges: Vec<(usize, usize, u32)>,
    pub root: usize,
    pub used_rules: BTreeSet<u32>,
}

impl DerivationGraph {
    /// Construit exactement D_k(G_0, R_0) par exploration BFS bornée à l'horizon `horizon_k`.
    pub fn build(
        root_graph: &TermGraph,
        rules: &[Rule],
        horizon_k: usize,
        max_states: usize,
    ) -> Self {
        let canon_root = root_graph.canonicalize();
        let mut states: Vec<TermGraph> = vec![canon_root.clone()];
        let mut state_index: HashMap<TermGraph, usize> = HashMap::new();
        state_index.insert(canon_root, 0);

        let mut edges: Vec<(usize, usize, u32)> = Vec::new();
        let mut edge_set: BTreeSet<(usize, usize, u32)> = BTreeSet::new();
        let mut used_rules: BTreeSet<u32> = BTreeSet::new();
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
        queue.push_back((0, 0));

        while let Some((u, depth)) = queue.pop_front() {
            if depth >= horizon_k {
                continue;
            }
            let current = states[u].clone();
            for rule in rules {
                for next_g in apply_rule_everywhere(&current, rule) {
                    let v = if let Some(&existing) = state_index.get(&next_g) {
                        existing
                    } else {
                        if states.len() >= max_states {
                            continue;
                        }
                        let id = states.len();
                        states.push(next_g.clone());
                        state_index.insert(next_g, id);
                        queue.push_back((id, depth + 1));
                        id
                    };
                    if edge_set.insert((u, v, rule.id)) {
                        edges.push((u, v, rule.id));
                        used_rules.insert(rule.id);
                    }
                }
            }
        }

        Self {
            states,
            edges,
            root: 0,
            used_rules,
        }
    }

    /// Ensemble des symboles apparaissant dans au moins un état atteignable de D_k.
    pub fn reachable_symbols(&self) -> BTreeSet<SymbolId> {
        let mut set = BTreeSet::new();
        for g in &self.states {
            set.extend(g.symbols());
        }
        set
    }

    /// Formes normales (sommets de degré sortant 0).
    pub fn normal_forms(&self) -> Vec<usize> {
        let mut out_deg = vec![0usize; self.states.len()];
        for &(u, _, _) in &self.edges {
            out_deg[u] += 1;
        }
        (0..self.states.len()).filter(|&u| out_deg[u] == 0).collect()
    }

    /// Détecte exactement l'existence d'un cycle dirigé (b_1 dirigé >= 1).
    pub fn has_directed_cycle(&self) -> bool {
        let n = self.states.len();
        let mut adj = vec![Vec::new(); n];
        for &(u, v, _) in &self.edges {
            adj[u].push(v);
        }
        // 0 = blanc, 1 = gris, 2 = noir
        let mut color = vec![0u8; n];

        fn dfs(u: usize, adj: &[Vec<usize>], color: &mut [u8]) -> bool {
            color[u] = 1;
            for &v in &adj[u] {
                if color[v] == 1 {
                    return true;
                }
                if color[v] == 0 && dfs(v, adj, color) {
                    return true;
                }
            }
            color[u] = 2;
            false
        }

        for i in 0..n {
            if color[i] == 0 && dfs(i, &adj, &mut color) {
                return true;
            }
        }
        false
    }

    /// Matrice d'adjacence simple non étiquetée (arêtes parallèles fusionnées).
    fn unlabeled_adj(&self) -> (Vec<BTreeSet<usize>>, Vec<usize>, Vec<usize>) {
        let n = self.states.len();
        let mut adj = vec![BTreeSet::new(); n];
        let mut in_deg = vec![0usize; n];
        let mut out_deg = vec![0usize; n];
        for &(u, v, _) in &self.edges {
            if adj[u].insert(v) {
                out_deg[u] += 1;
                in_deg[v] += 1;
            }
        }
        (adj, in_deg, out_deg)
    }

    /// Test exact d'isomorphisme de graphes dirigés enracinés non étiquetés (pour \mu_iso).
    pub fn is_unlabeled_isomorphic(&self, other: &DerivationGraph) -> bool {
        let n = self.states.len();
        if n != other.states.len() {
            return false;
        }
        let (adj1, in1, out1) = self.unlabeled_adj();
        let (adj2, in2, out2) = other.unlabeled_adj();

        if in1[self.root] != in2[other.root] || out1[self.root] != out2[other.root] {
            return false;
        }

        let mut sig1: Vec<(usize, usize)> = (0..n).map(|i| (in1[i], out1[i])).collect();
        let mut sig2: Vec<(usize, usize)> = (0..n).map(|i| (in2[i], out2[i])).collect();
        sig1.sort_unstable();
        sig2.sort_unstable();
        if sig1 != sig2 {
            return false;
        }

        let mut mapping: Vec<Option<usize>> = vec![None; n];
        let mut used = vec![false; n];
        mapping[self.root] = Some(other.root);
        used[other.root] = true;

        // Ordonner les sommets restants par degré total décroissant pour élaguer vite.
        let mut order: Vec<usize> = (0..n).filter(|&i| i != self.root).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(in1[i] + out1[i]));

        fn backtrack(
            pos: usize,
            order: &[usize],
            adj1: &[BTreeSet<usize>],
            adj2: &[BTreeSet<usize>],
            in1: &[usize],
            out1: &[usize],
            in2: &[usize],
            out2: &[usize],
            mapping: &mut [Option<usize>],
            used: &mut [bool],
        ) -> bool {
            if pos == order.len() {
                return true;
            }
            let u = order[pos];
            for v in 0..adj2.len() {
                if used[v] || in1[u] != in2[v] || out1[u] != out2[v] {
                    continue;
                }
                let mut ok = true;
                for (u_prev, m_opt) in mapping.iter().enumerate() {
                    if let Some(v_prev) = *m_opt {
                        if adj1[u].contains(&u_prev) != adj2[v].contains(&v_prev)
                            || adj1[u_prev].contains(&u) != adj2[v_prev].contains(&v)
                        {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    mapping[u] = Some(v);
                    used[v] = true;
                    if backtrack(
                        pos + 1, order, adj1, adj2, in1, out1, in2, out2, mapping, used,
                    ) {
                        return true;
                    }
                    mapping[u] = None;
                    used[v] = false;
                }
            }
            false
        }

        backtrack(
            0, &order, &adj1, &adj2, &in1, &out1, &in2, &out2, &mut mapping, &mut used,
        )
    }
}

/// Détecte structurellement les symboles absorbeurs (\bot) purement à partir de R_0 :
/// un symbole nul-aire z est un absorbeur si z n'a aucune règle sortante et qu'il existe
/// au moins un constructeur f tel que f(..., z, ...) -> z pour chaque position d'argument.
pub fn detect_absorber_symbols(
    rules: &[Rule],
    arities: &BTreeMap<SymbolId, usize>,
) -> BTreeSet<SymbolId> {
    let mut nullary_candidates: BTreeSet<SymbolId> = arities
        .iter()
        .filter_map(|(&s, &a)| if a == 0 { Some(s) } else { None })
        .collect();

    // Exclure tout symbole nul-aire qui possède une règle directe z -> ...
    for r in rules {
        if let PatternNode::Sym(s, ch) = &r.lhs.nodes[r.lhs.root] {
            if ch.is_empty() {
                nullary_candidates.remove(s);
            }
        }
    }

    let mut absorbers = BTreeSet::new();
    for z in nullary_candidates {
        // Chercher un constructeur f d'arité a >= 1 absorbé par z à toutes ses positions
        let mut absorbed_positions: HashMap<SymbolId, BTreeSet<usize>> = HashMap::new();
        for r in rules {
            let rhs_is_z = matches!(&r.rhs.nodes[r.rhs.root], PatternNode::Sym(s, ch) if *s == z && ch.is_empty());
            if !rhs_is_z {
                continue;
            }
            if let PatternNode::Sym(f, ch) = &r.lhs.nodes[r.lhs.root] {
                if !ch.is_empty() {
                    let mut z_pos = None;
                    let mut all_others_vars = true;
                    for (idx, &c) in ch.iter().enumerate() {
                        match &r.lhs.nodes[c] {
                            PatternNode::Sym(s, sub) if *s == z && sub.is_empty() && z_pos.is_none() => {
                                z_pos = Some(idx);
                            }
                            PatternNode::Var(_) => {}
                            _ => {
                                all_others_vars = false;
                                break;
                            }
                        }
                    }
                    if all_others_vars {
                        if let Some(pos) = z_pos {
                            absorbed_positions.entry(*f).or_default().insert(pos);
                        }
                    }
                }
            }
        }

        let is_absorber = absorbed_positions.iter().any(|(f, positions)| {
            arities.get(f).copied().unwrap_or(0) == positions.len() && !positions.is_empty()
        });
        if is_absorber {
            absorbers.insert(z);
        }
    }
    absorbers
}
