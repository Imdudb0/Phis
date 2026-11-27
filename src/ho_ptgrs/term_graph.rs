use std::collections::{BTreeSet, HashMap};

/// Identifiant de symbole dans la signature \Sigma.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(pub u32);

/// Nœud d'un graphe de termes : symbole étiqueté + successeurs ordonnés (partage et cycles autorisés).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Node {
    pub label: SymbolId,
    pub succ: Vec<usize>,
}

/// Graphe de termes enraciné G = (V, lab, succ, r).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TermGraph {
    pub nodes: Vec<Node>,
    pub root: usize,
}

impl TermGraph {
    pub fn new(nodes: Vec<Node>, root: usize) -> Self {
        Self { nodes, root }.canonicalize()
    }

    /// Numérotation canonique par parcours DFS ordonné depuis la racine.
    /// Deux graphes de termes enracinés accessibles sont isomorphes ssi leurs formes canoniques sont égales.
    pub fn canonicalize(&self) -> Self {
        if self.nodes.is_empty() {
            return self.clone();
        }
        let mut mapping: Vec<Option<usize>> = vec![None; self.nodes.len()];
        let mut order: Vec<usize> = Vec::with_capacity(self.nodes.len());

        fn dfs(
            u: usize,
            nodes: &[Node],
            mapping: &mut [Option<usize>],
            order: &mut Vec<usize>,
        ) {
            if mapping[u].is_some() {
                return;
            }
            let new_id = order.len();
            mapping[u] = Some(new_id);
            order.push(u);
            for &v in &nodes[u].succ {
                dfs(v, nodes, mapping, order);
            }
        }

        dfs(self.root, &self.nodes, &mut mapping, &mut order);

        let new_nodes = order
            .iter()
            .map(|&old_id| {
                let n = &self.nodes[old_id];
                Node {
                    label: n.label,
                    succ: n.succ.iter().map(|&c| mapping[c].unwrap()).collect(),
                }
            })
            .collect();

        Self {
            nodes: new_nodes,
            root: 0,
        }
    }

    pub fn symbols(&self) -> BTreeSet<SymbolId> {
        self.nodes.iter().map(|n| n.label).collect()
    }

    /// Constructeur utilitaire pour bâtir un graphe enraciné proprement.
    pub fn builder() -> TermGraphBuilder {
        TermGraphBuilder { nodes: Vec::new() }
    }
}

pub struct TermGraphBuilder {
    nodes: Vec<Node>,
}

impl TermGraphBuilder {
    pub fn add(&mut self, label: SymbolId, succ: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node { label, succ });
        id
    }

    pub fn finish(self, root: usize) -> TermGraph {
        TermGraph::new(self.nodes, root)
    }
}

/// Nœud d'un moule (template) issu de l'anti-unification sur graphes de termes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateNode {
    Sym(SymbolId, Vec<usize>),
    Hole(usize),
}

/// Résultat d'anti-unification (plus générale généralisation commune M).
#[derive(Debug, Clone)]
pub struct TemplateGraph {
    pub nodes: Vec<TemplateNode>,
    pub root: usize,
    pub hole_count: usize,
}

impl TemplateGraph {
    /// Nombre de constructeurs internes (arité > 0) partagés par tous les graphes.
    pub fn internal_constructor_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n, TemplateNode::Sym(_, succ) if !succ.is_empty()))
            .count()
    }

    /// Un moule est non trivial s'il partage au moins 2 constructeurs internes et au moins 1 trou.
    pub fn is_nontrivial(&self) -> bool {
        self.internal_constructor_count() >= 2 && self.hole_count >= 1
    }
}

/// Anti-unification exacte sur une famille de graphes de termes (gère partage et cycles).
pub fn anti_unify(graphs: &[TermGraph]) -> Option<TemplateGraph> {
    if graphs.is_empty() {
        return None;
    }

    let mut memo: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut nodes: Vec<TemplateNode> = Vec::new();
    let mut hole_count: usize = 0;

    fn recurse(
        tuple: Vec<usize>,
        graphs: &[TermGraph],
        memo: &mut HashMap<Vec<usize>, usize>,
        nodes: &mut Vec<TemplateNode>,
        hole_count: &mut usize,
    ) -> usize {
        if let Some(&existing) = memo.get(&tuple) {
            return existing;
        }

        let first_node = &graphs[0].nodes[tuple[0]];
        let all_match = graphs.iter().enumerate().all(|(i, g)| {
            let n = &g.nodes[tuple[i]];
            n.label == first_node.label && n.succ.len() == first_node.succ.len()
        });

        if all_match {
            let arity = first_node.succ.len();
            let slot = nodes.len();
            nodes.push(TemplateNode::Sym(first_node.label, Vec::new()));
            memo.insert(tuple.clone(), slot);

            let mut children = Vec::with_capacity(arity);
            for arg_idx in 0..arity {
                let child_tuple: Vec<usize> = graphs
                    .iter()
                    .enumerate()
                    .map(|(i, g)| g.nodes[tuple[i]].succ[arg_idx])
                    .collect();
                let cid = recurse(child_tuple, graphs, memo, nodes, hole_count);
                children.push(cid);
            }

            if let TemplateNode::Sym(_, ref mut succ) = nodes[slot] {
                *succ = children;
            }
            slot
        } else {
            let hid = *hole_count;
            *hole_count += 1;
            let slot = nodes.len();
            nodes.push(TemplateNode::Hole(hid));
            memo.insert(tuple, slot);
            slot
        }
    }

    let roots: Vec<usize> = graphs.iter().map(|g| g.root).collect();
    let root = recurse(roots, graphs, &mut memo, &mut nodes, &mut hole_count);

    Some(TemplateGraph {
        nodes,
        root,
        hole_count,
    })
}
