//! Dynamique abductive : depuis E, on cherche une cause (graphe) qui
//! régénère E et maximise φ_sharp. Aucun MDL : on maximise l'explication.

use crate::graph::{Graph, NodeKind};
use crate::model::Model;
use crate::perception::Perception;
use crate::phi::phi_sharp;
use crate::rng::Rng;

pub struct Abduction {
    pub max_iters: usize,
    pub temperature: f32,
}

impl Abduction {
    pub fn new() -> Self { Abduction { max_iters: 64, temperature: 0.5 } }

    pub fn run(&self, model: &Model, perc: &Perception) -> (Graph, f64) {
        let mut rng = Rng::new(0x1234_5678 ^ (perc.bytes.len() as u64));

        let mut best = perc.graph.clone_graph();
        best.propagate(2);
        let mut best_phi = phi_sharp(&best, &perc.effect_nodes);

        for _ in 0..self.max_iters {
            let mut hyp = best.clone_graph();

            // Proposition de réécriture : ajouter un nœud Concept
            // (abduction créative), le connecter à des nœuds existants
            // via des matrices non-réciproques, et éventuellement un
            // lien de verbalisation vers un Term du dictionnaire.
            let n_existing = hyp.nodes.len();
            if n_existing == 0 { break; }

            // concept tiré parmi les concepts émergents du modèle si dispo,
            // sinon nouveau concept aléatoire (induit par E)
            let (c_emb, c_label) = if !model.concept_emb.is_empty() {
                let i = rng.next_range(model.concept_emb.len());
                (model.concept_emb[i], i as u32)
            } else {
                // concept aléatoire dérivé d'un byte de E
                let b = perc.bytes[rng.next_range(perc.bytes.len().max(1))];
                (model.byte_emb[b as usize], u32::MAX)
            };

            let c_idx = hyp.add_node(c_emb, NodeKind::Concept, c_label);

            // connecte le concept à k nœuds existants (matrices non-réciproques)
            let k = 1 + rng.next_range(3);
            for _ in 0..k {
                let target = rng.next_range(n_existing);
                let m = Graph::random_matrix(&mut rng);
                hyp.add_edge(c_idx, target, m);
                // contre-champ non-réciproque (i.e. matrice indépendante)
                let m2 = Graph::random_matrix(&mut rng);
                hyp.add_edge(target, c_idx, m2);
            }

            hyp.propagate(2);
            let phi = phi_sharp(&hyp, &perc.effect_nodes);

            // Critère : on accepte si φ augmente, sinon acceptation
            // métropolis (exploration — l'abduction n'est pas un gradient
            // greedy fermé).
            let accept = phi > best_phi
                || rng.next_f32() < (-((phi - best_phi) / self.temperature as f64)).exp() as f32;
            if accept {
                best = hyp;
                best_phi = phi;
            }
        }

        (best, best_phi)
    }
}