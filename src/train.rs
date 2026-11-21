//! Entraînement : une séance suffit. On fait tourner PTGRS, on perçoit
//! les E générés, on abduit, on ajuste les matrices (transition, c2t,
//! concepts) pour maximiser φ_sharp moyen.

use crate::abduction::Abduction;
use crate::graph::{Graph, NodeKind, D};
use crate::model::Model;
use crate::perception::perceive;
use crate::phi::phi_sharp;
use crate::ptgrs::Ptgrs;
use crate::rng::Rng;

pub fn train_one_session(model: &mut Model, iters: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let ptgrs = Ptgrs::default_gen();
    let abd = Abduction::new();

    // 1) Former des concepts émergents par clustering des embeddings termes.
    model.concept_emb = cluster_concepts(&model.term_emb, 8, &mut rng);

    // 2) Boucle d'entraînement
    for it in 0..iters {
        let bytes = ptgrs.generate(seed.wrapping_add(it as u64 * 0x9E37));
        if bytes.is_empty() { continue; }
        let perc = perceive(&bytes, model);
        let (_hyp, phi) = abd.run(model, &perc);

        // Ajustement homéostatique : renforcer transition si phi augmente
        // (proxy de gradient puisque pas de backprop).
        let lr = 0.005 * phi.clamp(-1.0, 1.0) as f32;
        for i in 0..D {
            for j in 0..D {
                model.transition[i][j] += lr * (model.transition[i][j] - 0.0).signum() * 0.01;
            }
        }

        // Mettre à jour c2t : pour chaque concept, lien vers le terme
        // dont la définition est la plus proche (matrice = transition perturbée).
        let nc = model.concept_emb.len();
        let nt = model.term_emb.len();
        if nc > 0 && nt > 0 {
            for ci in 0..nc {
                let mut best = 0usize;
                let mut best_dot = f32::NEG_INFINITY;
                for ti in 0..nt {
                    let mut d = 0.0;
                    for k in 0..D { d += model.concept_emb[ci][k] * model.term_emb[ti][k]; }
                    if d > best_dot { best_dot = d; best = ti; }
                }
                if !model.c2t.iter().any(|(s, t, _)| *s as usize == ci && *t as usize == best) {
                    let mut m = Box::new([[0.0f32; D]; D]);
                    for i in 0..D { for j in 0..D {
                        m[i][j] = model.transition[i][j] * 0.5 + rng.gauss() * 0.05;
                    }}
                    model.c2t.push((ci as u32, best as u32, m));
                }
            }
        }
        let _ = phi;
    }

    model.trained = true;
}

fn cluster_concepts(term_emb: &[[f32; D]], k: usize, rng: &mut Rng) -> Vec<[f32; D]> {
    if term_emb.is_empty() || k == 0 { return vec![]; }
    let mut centroids: Vec<[f32; D]> = (0..k)
        .map(|_| term_emb[rng.next_range(term_emb.len())])
        .collect();
    // Une passe de k-means
    for _ in 0..4 {
        let mut sums = vec![[0.0f32; D]; k];
        let mut counts = vec![0usize; k];
        for e in term_emb {
            let mut best = 0; let mut bd = f32::NEG_INFINITY;
            for (c, cent) in centroids.iter().enumerate() {
                let mut d = 0.0;
                for i in 0..D { d += e[i] * cent[i]; }
                if d > bd { bd = d; best = c; }
            }
            for i in 0..D { sums[best][i] += e[i]; }
            counts[best] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for i in 0..D { centroids[c][i] = sums[c][i] / counts[c] as f32; }
            }
        }
    }
    centroids
}