use crate::meta_ops::{DefGraph, MetaOp};
use std::collections::{BTreeMap, HashMap};

/// Rapport de vérification de rigidité définitionnelle (Théorème B, Condition 1).
#[derive(Debug, Clone)]
pub struct RigidityReport {
    pub is_rigid: bool,
    pub iterations: usize,
    pub total_color_classes: usize,
    /// Multiplicité de la classe de couleur 2-FWL de chaque nœud concept v_{\mu}.
    pub concept_multiplicities: BTreeMap<MetaOp, usize>,
}

/// Vérifie la rigidité structurelle de G_def par raffinement de couleurs 2-FWL (Folklore 2-WL).
///
/// IMPORTANT : tous les nœuds concepts v_{\mu} reçoivent la MÊME couleur initiale (masquage)
/// afin que 2-FWL ne puisse les distinguer QUE par la topologie pure de leur définition.
/// Si deux définitions sont structurellement indiscernables, leur multiplicité sera >= 2 et le test échoue.
pub fn verify_rigidity_2fwl(def_graph: &DefGraph) -> RigidityReport {
    let n = def_graph.node_symbols.len();
    if n == 0 {
        return RigidityReport {
            is_rigid: false,
            iterations: 0,
            total_color_classes: 0,
            concept_multiplicities: BTreeMap::new(),
        };
    }

    // Indexation des arêtes dirigées par paire (u, v) -> liste triée de types de relations
    let mut edge_labels: Vec<Vec<Vec<u32>>> = vec![vec![Vec::new(); n]; n];
    for &(u, rel, v) in &def_graph.edges {
        edge_labels[u][v].push(rel.0);
    }
    for row in &mut edge_labels {
        for cell in row {
            cell.sort_unstable();
        }
    }

    // Initialisation des couleurs de paires c^{(0)}(i, j)
    // - Sur la diagonale (i == j) : 0 si nœud concept (masqué !), 1 si nœud anonyme de schéma,
    //   ou identifiant de hub partagé si hub ontologique (degré entrant > 1 de plusieurs définitions).
    //   Pour un test maximalement strict, même les nœuds non-concepts sont masqués en 1 !
    let mut init_keys: Vec<(u8, Vec<u32>, Vec<u32>)> = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                let role = if def_graph.is_concept_node[i] { 0u8 } else { 1u8 };
                init_keys.push((role, Vec::new(), Vec::new()));
            } else {
                init_keys.push((2u8, edge_labels[i][j].clone(), edge_labels[j][i].clone()));
            }
        }
    }

    let (mut colors, mut num_classes) = compress_colors(&init_keys);
    let mut iterations = 0usize;

    // Boucle de raffinement 2-FWL :
    // c^{(t+1)}(i, j) = compress( c^{(t)}(i, j), multiset_{k in 0..n} (c^{(t)}(i, k), c^{(t)}(k, j)) )
    loop {
        iterations += 1;
        let mut next_keys: Vec<(u32, Vec<(u32, u32)>)> = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let mut multiset = Vec::with_capacity(n);
                for k in 0..n {
                    multiset.push((colors[i * n + k], colors[k * n + j]));
                }
                multiset.sort_unstable();
                next_keys.push((colors[i * n + j], multiset));
            }
        }

        let (next_colors, next_classes) = compress_colors(&next_keys);
        colors = next_colors;
        if next_classes == num_classes {
            break;
        }
        num_classes = next_classes;
        if iterations >= n * n {
            break;
        }
    }

    // Compter la multiplicité de chaque couleur diagonale c^{(\infty)}(v, v)
    let mut diag_counts: HashMap<u32, usize> = HashMap::new();
    for v in 0..n {
        *diag_counts.entry(colors[v * n + v]).or_insert(0) += 1;
    }

    let mut concept_multiplicities = BTreeMap::new();
    let mut is_rigid = true;
    for (op, &node_idx) in &def_graph.concept_nodes {
        let mult = diag_counts[&colors[node_idx * n + node_idx]];
        if mult != 1 {
            is_rigid = false;
        }
        concept_multiplicities.insert(op.clone(), mult);
    }

    RigidityReport {
        is_rigid,
        iterations,
        total_color_classes: num_classes,
        concept_multiplicities,
    }
}

fn compress_colors<K: Ord>(keys: &[K]) -> (Vec<u32>, usize) {
    let mut indexed: Vec<(&K, usize)> = keys.iter().enumerate().map(|(i, k)| (k, i)).collect();
    indexed.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let mut result = vec![0u32; keys.len()];
    let mut current_id = 0u32;
    for idx in 0..indexed.len() {
        if idx > 0 && indexed[idx].0 != indexed[idx - 1].0 {
            current_id += 1;
        }
        result[indexed[idx].1] = current_id;
    }
    let count = if keys.is_empty() {
        0
    } else {
        (current_id as usize) + 1
    };
    (result, count)
}
