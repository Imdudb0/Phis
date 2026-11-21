//! φ_sharp : scalaire de puissance cause-effet intégrée (inspirée IIT 4.0),
//! relatif à E. On mesure, sur une bipartition, la perte d'information
//! causale en coupant les arêtes traversantes.

use crate::graph::{Graph, D};

pub fn phi_sharp(g: &Graph, effect_nodes: &[usize]) -> f64 {
    if g.nodes.is_empty() || effect_nodes.is_empty() { return 0.0; }
    let n = g.nodes.len();

    // Activer les nœuds cibles (E) au démarrage.
    let mut base = g.clone_graph();
    for &e in effect_nodes {
        if e < base.nodes.len() {
            for k in 0..D { base.nodes[e].emb[k] = 1.0; }
        }
    }
    base.propagate(3);
    let base_state: Vec<[f32; D]> = base.nodes.iter().map(|n| n.emb).collect();

    // Bipartitions aléatoires mais déterministes sur un échantillon :
    // exhaustif pour n<=12, sinon 64 tirages déterministes.
    let partitions = sample_partitions(n);

    let mut phi_min = f64::INFINITY;
    for (a_mask, b_mask) in &partitions {
        let mut cut = g.clone_graph();
        cut.edges.retain(|e| {
            let s_in_a = a_mask[e.src];
            let d_in_a = a_mask[e.dst];
            let s_in_b = b_mask[e.src];
            let d_in_b = b_mask[e.dst];
            !((s_in_a && d_in_b) || (s_in_b && d_in_a))
        });
        cut.propagate(3);
        let mut cut_state = cut.nodes.iter().map(|n| n.emb);
        let mut delta = 0.0f64;
        for (i, c) in cut_state.by_ref().enumerate() {
            for k in 0..D {
                let d = (base_state[i][k] - c[k]) as f64;
                delta += d * d;
            }
        }
        // Information intégrée au sens "minimum over cuts".
        let phi_cut = delta / (n as f64 + 1.0);
        if phi_cut < phi_min { phi_min = phi_cut; }
    }
    if phi_min.is_finite() { phi_min } else { 0.0 }
}

fn sample_partitions(n: usize) -> Vec<(Vec<bool>, Vec<bool>)> {
    let mut out = vec![];
    if n <= 12 {
        for mask in 1u32..(1u32 << n) - 1 {
            let mut a = vec![false; n];
            let mut b = vec![false; n];
            for i in 0..n {
                if mask & (1 << i) != 0 { a[i] = true; } else { b[i] = true; }
            }
            out.push((a, b));
        }
    } else {
        let mut r = crate::rng::Rng::new(0xA5A5_5A5A);
        for _ in 0..64 {
            let mut a = vec![false; n];
            let mut b = vec![false; n];
            for i in 0..n {
                if r.next_f32() < 0.5 { a[i] = true; } else { b[i] = true; }
            }
            out.push((a, b));
        }
    }
    out
}