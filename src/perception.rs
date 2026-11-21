//! E = bytes. La perception construit un graphe-terme déterministe :
//! nœuds Byte chaînés par transitions + nœuds Term pour chaque
//! sous-chaîne qui matche un dictionnaire.

use crate::graph::{Graph, NodeKind, D};
use crate::model::Model;

pub struct Perception {
    pub bytes: Vec<u8>,
    pub graph: Graph,
    /// Nœuds "cible" (Byte) dont l'emb doit être reproduite par l'abduction.
    pub effect_nodes: Vec<usize>,
}

pub fn perceive(bytes: &[u8], model: &Model) -> Perception {
    let mut g = Graph::new();
    let mut prev: Option<usize> = None;
    let mut effect_nodes = vec![];
    for &b in bytes {
        let emb = model.byte_emb[b as usize];
        let idx = g.add_node(emb, NodeKind::Byte, b as u32);
        effect_nodes.push(idx);
        if let Some(p) = prev {
            // matrice de transition = non-réciproque par construction
            g.add_edge(p, idx, model.transition.clone());
        }
        prev = Some(idx);
    }
    // Attache les occurrences de termes du dictionnaire (matching naïf
    // mais complet : toute sous-chaîne égale à un terme du dico).
    for (ti, term) in model.term_dict.iter().enumerate() {
        let mut start = 0;
        while start + term.len() <= bytes.len() {
            if &bytes[start..start + term.len()] == term.as_slice() {
                let tid = g.add_node(model.term_emb[ti], NodeKind::Term, ti as u32);
                // relie chaque octet couvert au term (matrice ∝ transition)
                for off in 0..term.len() {
                    g.add_edge(start + off, tid, model.transition.clone());
                }
                start += 1;
            } else {
                start += 1;
            }
        }
    }
    Perception { bytes: bytes.to_vec(), graph: g, effect_nodes }
}