use phisharp::ho_ptgrs::generator::{CurriculumConfig, HoPtgrsGenerator};
use phisharp::ho_ptgrs::meta_ops::MetaOp;

use std::env;
use std::fs;

fn main() {
    let mut seed: u64 = 42;
    let mut horizon_k: usize = 4;
    let mut num_distractors: usize = 4;
    let mut noise_ratio: f64 = 0.5;
    let mut out_path: Option<String> = None;
    let mut forced_op: Option<MetaOp> = None;

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().expect("seed invalide");
                i += 2;
            }
            "--horizon" if i + 1 < args.len() => {
                horizon_k = args[i + 1].parse().expect("horizon invalide");
                i += 2;
            }
            "--distractors" if i + 1 < args.len() => {
                num_distractors = args[i + 1].parse().expect("distractors invalide");
                i += 2;
            }
            "--noise" if i + 1 < args.len() => {
                noise_ratio = args[i + 1].parse().expect("noise invalide");
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--op" if i + 1 < args.len() => {
                forced_op = Some(parse_op(&args[i + 1]));
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    let config = CurriculumConfig {
        horizon_k,
        num_distractors,
        noise_ratio,
        max_attempts: 64,
    };

    let mut gen = HoPtgrsGenerator::new(seed, config);
    let ep = gen
        .generate_episode(forced_op)
        .expect("Impossible de générer un épisode valide");

    println!("=== Épisode HO-PTGRS généré ===");
    println!("Cause cachée (eval post-hoc) : {:?}", ep.hidden_cause.target_op);
    println!("Distracteurs G_def           : {:?}", ep.hidden_cause.distractors);
    println!(
        "Rigidité 2-FWL               : {} ({} itérations, {} classes de couleurs)",
        ep.rigidity.is_rigid, ep.rigidity.iterations, ep.rigidity.total_color_classes
    );
    println!(
        "Codec préfixe-libre / S_256  : invariants OK = {}",
        ep.stats.codec_invariants_ok
    );
    println!(
        "Taille perceived.eff         : {} octets (G_def: {} nœuds / {} arêtes | G_inst: {} règles dont {} bruit)",
        ep.stats.total_bytes,
        ep.stats.def_nodes,
        ep.stats.def_edges,
        ep.stats.inst_rules_total,
        ep.stats.noise_rules_added
    );
    println!(
        "Code octets cible (post-hoc) : {:02X?}",
        ep.hidden_cause.target_concept_bytes
    );

    if let Some(path) = out_path {
        fs::write(&path, &ep.perceived_eff).expect("Échec d'écriture de perceived.eff");
        println!("Écrit dans : {}", path);
    }
}

fn parse_op(s: &str) -> MetaOp {
    match s.to_lowercase().as_str() {
        "bif" => MetaOp::Bif,
        "iso" => MetaOp::Iso,
        "bot" => MetaOp::Bot,
        "fix" => MetaOp::Fix,
        "quot" => MetaOp::Quot,
        "comp_bif_fix" => MetaOp::comp(MetaOp::Bif, MetaOp::Fix),
        "comp_iso_bif" => MetaOp::comp(MetaOp::Iso, MetaOp::Bif),
        "comp_iso_fix" => MetaOp::comp(MetaOp::Iso, MetaOp::Fix),
        "comp_quot_bif" => MetaOp::comp(MetaOp::Quot, MetaOp::Bif),
        other => panic!("Opérateur inconnu: {}", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta_ops::{build_def_graph, DefGraph};
    use crate::rigidity::verify_rigidity_2fwl;
    use crate::term_graph::SymbolId;
    use std::collections::BTreeMap;

    #[test]
    fn test_all_canonical_and_composite_operators() {
        let cfg = CurriculumConfig {
            horizon_k: 4,
            num_distractors: 5,
            noise_ratio: 0.75,
            max_attempts: 32,
        };
        let mut gen = HoPtgrsGenerator::new(2026, cfg);

        for op in MetaOp::catalog() {
            let ep = gen
                .generate_episode(Some(op.clone()))
                .unwrap_or_else(|e| panic!("Échec pour {:?}: {}", op, e));

            assert_eq!(ep.hidden_cause.target_op, op);
            assert!(ep.rigidity.is_rigid, "G_def non rigide pour {:?}", op);
            assert!(ep.stats.codec_invariants_ok, "Invariant codec violé");
            assert!(!ep.perceived_eff.is_empty());

            // Vérifie que tous les concepts ont une multiplicité 2-FWL égale à 1
            for (&ref c_op, &mult) in &ep.rigidity.concept_multiplicities {
                assert_eq!(mult, 1, "Multiplicité != 1 pour {:?}", c_op);
            }
        }
    }

    #[test]
    fn test_rigidity_rejects_symmetric_definitions() {
        // Test négatif : deux concepts avec exactement la même structure relationnelle
        // doivent être fusionnés par 2-FWL (puisque les identités des concepts sont masquées).
        let mut concept_nodes = BTreeMap::new();
        concept_nodes.insert(MetaOp::Bif, 0);
        concept_nodes.insert(MetaOp::Fix, 1);

        let sym_def = DefGraph {
            node_symbols: vec![SymbolId(10), SymbolId(11), SymbolId(12), SymbolId(13)],
            edges: vec![
                (0, SymbolId(1), 2),
                (1, SymbolId(1), 3),
                (2, SymbolId(2), 3),
                (3, SymbolId(2), 2),
            ],
            concept_nodes,
            is_concept_node: vec![true, true, false, false],
        };

        let report = verify_rigidity_2fwl(&sym_def);
        assert!(
            !report.is_rigid,
            "2-FWL aurait dû rejeter deux définitions symétriques !"
        );
        assert_eq!(report.concept_multiplicities[&MetaOp::Bif], 2);
        assert_eq!(report.concept_multiplicities[&MetaOp::Fix], 2);
    }

    #[test]
    fn test_full_catalog_def_graph_is_globally_rigid() {
        let all = MetaOp::catalog();
        let def_graph = build_def_graph(&all);
        let report = verify_rigidity_2fwl(&def_graph);
        assert!(
            report.is_rigid,
            "Le dictionnaire complet des 9 opérateurs doit être simultanément rigide à 2-FWL : {:?}",
            report.concept_multiplicities
        );
    }
}