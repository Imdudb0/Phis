mod graph;
mod multihop;
mod phi;
mod ptgrs;
mod rng;
mod train;

use crate::graph::PhiGraph;
use crate::multihop::{MultiHopAnalysis, analyze_monolithic_residual, analyze_multihop};
use crate::phi::{
    compute_composite_phi, compute_phi_sharp, extract_abductive_core, fnv_hash, format_partition,
};
use crate::ptgrs::{
    CauseFamily, SepMode, sample_composite_episode, sample_episode, sample_multihop_episode,
};
use crate::rng::Rng;
use crate::train::{TrainConfig, train_phisharp};
use std::env;
use std::fs;
use std::process;

fn usage() -> ! {
    eprintln!("Usage:");
    eprintln!("  phisharp train --out out/sharp.phis [--steps 300] [--seed 42]");
    eprintln!(
        "  phisharp sample --out perceived.eff [--family periodic|symmetric|nested|shared|substitution|grammar|composite|stream|multihop|multihop_flat] [--seed 7]"
    );
    eprintln!("  phisharp ask --model out/sharp.phis --file perceived.eff");
    process::exit(1);
}

fn arg_val(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
}

fn printable_preview(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|&b| {
            if (32..=126).contains(&b) {
                b as char
            } else {
                '.'
            }
        })
        .collect()
}

fn trim_ctrl(bytes: &[u8]) -> &[u8] {
    let mut s = 0usize;
    let mut e = bytes.len();
    while s < e && bytes[s] < 32 {
        s += 1;
    }
    while e > s && bytes[e - 1] < 32 {
        e -= 1;
    }
    &bytes[s..e]
}

fn hex_preview(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage();
    }

    match args[1].as_str() {
        "train" => {
            let out_path = arg_val(&args, "--out").unwrap_or_else(|| "out/sharp.phis".to_string());
            let steps = arg_val(&args, "--steps")
                .and_then(|s| s.parse().ok())
                .unwrap_or(300);
            let seed = arg_val(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(42);

            let cfg = TrainConfig {
                steps,
                pop_pairs: 24,
                sigma: 0.08,
                lr: 0.03,
                seed,
            };
            let model = train_phisharp(&cfg);
            if let Err(e) = model.save(&out_path) {
                eprintln!("Erreur écriture modèle '{}': {}", out_path, e);
                process::exit(1);
            }
            println!("\nModèle sauvegardé dans '{}'.", out_path);
        }

        "sample" => {
            let out_path = arg_val(&args, "--out").unwrap_or_else(|| "perceived.eff".to_string());
            let seed = arg_val(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(7);
            let mut rng = Rng::new(seed);

            match arg_val(&args, "--family").as_deref() {
                Some("composite") => {
                    let ep =
                        sample_composite_episode(&mut rng, 4, true, true, true, SepMode::Newline);
                    if let Err(e) = fs::write(&out_path, &ep.effect) {
                        eprintln!("Erreur écriture '{}': {}", out_path, e);
                        process::exit(1);
                    }
                    println!(
                        "Scène composite générée dans '{}' ({} octets, moule caché = bloc {} [{}])",
                        out_path,
                        ep.effect.len(),
                        ep.hidden_match_idx,
                        ep.hidden_families[ep.hidden_match_idx].name()
                    );
                    println!("  FLUX : {}", printable_preview(&ep.effect));
                    return;
                }
                Some("stream") => {
                    let ep = sample_composite_episode(&mut rng, 4, true, true, true, SepMode::None);
                    if let Err(e) = fs::write(&out_path, &ep.effect) {
                        eprintln!("Erreur écriture '{}': {}", out_path, e);
                        process::exit(1);
                    }
                    println!(
                        "Flux continu SANS séparateur généré dans '{}' ({} octets, moule caché = {} [{}])",
                        out_path,
                        ep.effect.len(),
                        ep.hidden_match_idx,
                        ep.hidden_families[ep.hidden_match_idx].name()
                    );
                    println!("  FLUX : {}", printable_preview(&ep.effect));
                    return;
                }
                Some("multihop") | Some("multihop_flat") => {
                    let flat = arg_val(&args, "--family").as_deref() == Some("multihop_flat");
                    let ep = sample_multihop_episode(&mut rng, flat, true);
                    if let Err(e) = fs::write(&out_path, &ep.effect) {
                        eprintln!("Erreur écriture '{}': {}", out_path, e);
                        process::exit(1);
                    }
                    println!(
                        "Scène multi-sauts{} générée dans '{}' ({} octets, concept caché = {} [{} + {}])",
                        if flat { " (flux plat)" } else { "" },
                        out_path,
                        ep.effect.len(),
                        printable_preview(&ep.hidden_concept_name),
                        ep.hidden_cause_pair.0.name(),
                        ep.hidden_cause_pair.1.name(),
                    );
                    println!("  Moules : {}", ep.hidden_prim_names.iter().map(|n| printable_preview(n)).collect::<Vec<_>>().join(", "));
                    println!("\n{}", String::from_utf8_lossy(&ep.effect));
                    return;
                }
                _ => {}
            }

            let family = match arg_val(&args, "--family") {
                Some(name) => CauseFamily::from_name(&name).unwrap_or_else(|| {
                    eprintln!(
                        "Famille inconnue '{}'. Choix: periodic, symmetric, nested, shared, substitution, grammar, composite, stream, multihop, multihop_flat.",
                        name
                    );
                    process::exit(1);
                }),
                None => CauseFamily::ALL[rng.next_usize(CauseFamily::ALL.len())],
            };
            let ep = sample_episode(&mut rng, family);
            if let Err(e) = fs::write(&out_path, &ep.effect) {
                eprintln!("Erreur écriture '{}': {}", out_path, e);
                process::exit(1);
            }
            println!(
                "Effet généré dans '{}' ({} octets, cause cachée = {})",
                out_path,
                ep.effect.len(),
                ep.hidden_cause.name()
            );
            println!("  ASCII : {}", printable_preview(&ep.effect));
        }

        "ask" => {
            let model_path = arg_val(&args, "--model").unwrap_or_else(|| usage());
            let file_path = arg_val(&args, "--file").unwrap_or_else(|| usage());

            let model = PhiGraph::load(&model_path).unwrap_or_else(|e| {
                eprintln!("Erreur chargement modèle '{}': {}", model_path, e);
                process::exit(1);
            });
            let bytes = fs::read(&file_path).unwrap_or_else(|e| {
                eprintln!("Erreur lecture effet '{}': {}", file_path, e);
                process::exit(1);
            });
            if bytes.len() < 4 {
                eprintln!("'{}' doit contenir au moins 4 octets.", file_path);
                process::exit(1);
            }

            if let Some(ana) = analyze_multihop(&model, &bytes) {
                run_ask_continuous(&model, &file_path, &bytes, &ana);
            } else {
                run_ask_monolithic(&model, &file_path, &bytes);
            }
        }

        _ => usage(),
    }
}

fn run_ask_monolithic(model: &PhiGraph, file_path: &str, bytes: &[u8]) {
    let mut rng = Rng::new(fnv_hash(bytes));
    let rep = compute_phi_sharp(model, bytes, &mut rng, true);
    let rels = model.encode_bytes(bytes);
    let res_split = analyze_monolithic_residual(model, bytes);

    println!("=== PHISHARP ABDUCTION REPORT (v0.6 — Monolithique) ===");
    println!("Effet (E)       : '{}' ({} octets)", file_path, bytes.len());
    println!("φ_sharp         : {:+.4}", rep.phi_sharp);
    println!(
        "  Générativité  : {:.4}  (résidu moyen = {:.4})",
        rep.generativity, rep.residual
    );
    println!("  Discrimination: {:.4}", rep.discrimination);
    println!(
        "  Projection    : {:.4}  [réfut={:.3} | dépl={:.3} | abst={:.3}]",
        rep.projection, rep.refutation_stab, rep.displacement_stab, rep.abstraction_stab
    );
    println!(
        "  Porte causale : {:.4}  (contraste={:.3})",
        rep.mean_gate, rep.gate_contrast
    );
    println!(
        "  Intégration   : {:.4}  (MIP exacte sur 31 coupes : perte={:.4}, coupe {})",
        rep.integration,
        rep.mip_loss,
        format_partition(rep.mip_mask)
    );

    let traj_str = rep
        .trace
        .step_residuals
        .iter()
        .enumerate()
        .map(|(t, r)| format!("t{}={:.3}", t, r))
        .collect::<Vec<_>>()
        .join(" -> ");
    println!("\nTrajectoire abductive (résidu) :");
    println!(
        "  {}  (gain/t0={:+.3}, rebond={:.4})",
        traj_str, rep.traj_gain, rep.bounce_pen
    );

    if res_split.parts.len() >= 2 {
        println!(
            "\nDécomposition multi-causes induite par le résidu (res {:.4} -> {:.4}, cos={:+.3}) :",
            res_split.single_err, res_split.split_err, res_split.cross_cos
        );
        for (i, &(a, b)) in res_split.parts.iter().enumerate() {
            println!("  cause #{} ({}..{}) : \"{}\"", i + 1, a, b, printable_preview(&bytes[a..b]));
        }
    }

    let (start, motif, err_all) =
        extract_abductive_core(model, bytes, &rels, &rep.trace.final_state);
    println!(
        "\nNoyau abductif auto-projeté (offset {}, régénère tout E avec résidu {:.4}) :",
        start, err_all
    );
    println!("  ASCII : \"{}\"", printable_preview(&motif));
    println!("  HEX   : {}", hex_preview(&motif));

    let mut ranked: Vec<(usize, f32)> = rep
        .trace
        .final_pos_err
        .iter()
        .copied()
        .enumerate()
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    println!("\nZones de tension (résidu maximal dans perceived.eff) :");
    for (pos, err) in ranked.into_iter().take(5) {
        let b = bytes[pos];
        let ch = if (32..=126).contains(&b) {
            b as char
        } else {
            '.'
        };
        println!(
            "  offset {:>3} | 0x{:02x} '{}' | résidu = {:.4}",
            pos, b, ch, err
        );
    }
}

fn run_ask_continuous(
    model: &PhiGraph,
    file_path: &str,
    bytes: &[u8],
    ana: &MultiHopAnalysis,
) {
    let comp = compute_composite_phi(model, bytes);
    let mode = if ana.definitions_found > 0 {
        "Multi-Sauts & Définitions Composées"
    } else {
        "Flux Continu Auto-Segmenté"
    };

    println!("=== PHISHARP ABDUCTION REPORT (v0.6 — {}) ===", mode);
    println!(
        "Effet (E)       : '{}' ({} octets, {} moules + {} cause(s) cible, {} déf. composées, {} méta-bloc(s) démote)",
        file_path,
        bytes.len(),
        ana.mold_regimes.len(),
        ana.target_spans.len(),
        ana.definitions_found,
        ana.demoted_meta_count
    );
    println!(
        "Cohérence scène : sélectivité={:.4} (R_top={:.3} vs R_rest={:.3}, paire [{}]<->[{}])",
        comp.selectivity, comp.r_top, comp.r_rest, comp.top_pair.0, comp.top_pair.1
    );

    if let Some((e_single, e_split, c_lr)) = ana.residual_info {
        println!(
            "Résidu cible    : décomposition multi-causes validée (res {:.4} -> {:.4}, cos inter-causes={:+.3})",
            e_single, e_split, c_lr
        );
    }

    for (ci, &(ts, te)) in ana.target_spans.iter().enumerate() {
        let q_bytes = &bytes[ts..te];
        let mut rng = Rng::new(fnv_hash(q_bytes));
        let q_rep = compute_phi_sharp(model, q_bytes, &mut rng, true);
        println!(
            "\nCause cible #{} (octets {}..{}) | φ={:+.4} [Gen={:.3} Disc={:.3} Proj={:.3} IIT={:.3} res={:.4}] :",
            ci + 1,
            ts,
            te,
            q_rep.phi_sharp,
            q_rep.generativity,
            q_rep.discrimination,
            q_rep.projection,
            q_rep.integration,
            q_rep.residual
        );
        println!("  \"{}\"", printable_preview(q_bytes));

        let matches = &ana.hop1_per_part[ci];
        let show_n = if ana.definitions_found > 0 {
            2.min(matches.len())
        } else {
            matches.len()
        };
        for (rank, m) in matches.iter().take(show_n).enumerate() {
            let key_label = ana.mold_keys[m.dst_idx]
                .as_ref()
                .map(|k| format!("clé=\"{}\"", printable_preview(k)))
                .unwrap_or_else(|| "sans clé".to_string());
            println!(
                "    Hop-1 #{} moule [{}] ({}) | résonance={:.4} | err={:.4} | cos={:+.3} | octets_communs={:.0}%",
                rank + 1,
                m.dst_idx,
                key_label,
                m.score,
                m.cross_err,
                m.cos_sim,
                m.byte_overlap * 100.0
            );
            println!(
                "       noyau : \"{}\"",
                printable_preview(&bytes[m.core_start..m.core_end])
            );
            if let Some((rs, re)) = m.referent {
                println!(
                    "       réf.  : \"{}\"",
                    printable_preview(trim_ctrl(&bytes[rs..re]))
                );
            }
        }
    }

    if !ana.hop2_matches.is_empty() {
        println!("\nProjection Référentielle Multi-Sauts (définitions composées découvertes) :");
        for (rank, hm) in ana.hop2_matches.iter().enumerate() {
            let refs = hm
                .concept
                .ref_labels
                .iter()
                .map(|r| printable_preview(r))
                .collect::<Vec<_>>()
                .join(" + ");
            println!(
                "  #{} [Hop-{}] \"{}\"  | score={:.4} (couv={:.3}, préc={:.3})  <-  {}",
                rank + 1,
                hm.concept.hop_depth,
                printable_preview(&hm.concept.name_bytes),
                hm.score,
                hm.coverage,
                hm.precision,
                refs
            );
            println!(
                "     clause ({}..{}) : \"{}\"",
                hm.concept.clause_span.0,
                hm.concept.clause_span.1,
                printable_preview(trim_ctrl(
                    &bytes[hm.concept.clause_span.0..hm.concept.clause_span.1]
                ))
            );
        }
    }

    // Choix final de l'explication maximale projetée (Hop-2/3 si multi-cause bat Hop-1)
    let best_hop1 = ana
        .hop1_per_part
        .first()
        .and_then(|v| v.first());
    let best_hop2 = ana.hop2_matches.first();

    let use_multihop = match (best_hop2, best_hop1) {
        (Some(h2), Some(h1)) => ana.target_spans.len() >= 2 || h2.score > h1.score,
        (Some(_), None) => true,
        _ => false,
    };

    if use_multihop && let Some(h2) = best_hop2 {
        let margin = ana
            .hop2_matches
            .get(1)
            .map(|s| h2.score - s.score)
            .unwrap_or(h2.score);
        println!(
            "\nExplication projetée (multi-sauts Hop-{}, alphabet de E, marge={:+.4}) :",
            h2.concept.hop_depth, margin
        );
        println!("  {}", printable_preview(&h2.concept.name_bytes));
    } else if let Some(best) = best_hop1 {
        let margin = ana.hop1_per_part[0]
            .get(1)
            .map(|s| best.score - s.score)
            .unwrap_or(best.score);
        let out_slice = match best.referent {
            Some((rs, re)) => trim_ctrl(&bytes[rs..re]),
            None => &bytes[best.core_start..best.core_end],
        };
        println!(
            "\nExplication projetée (Hop-1, alphabet de E, marge={:+.4}) :",
            margin
        );
        println!("  {}", printable_preview(out_slice));
    }
}
