mod graph;
mod phi;
mod ptgrs;
mod rng;
mod train;

use crate::graph::PhiGraph;
use crate::phi::{
    DiscoveredRegime, compute_composite_phi, compute_phi_sharp, discover_causal_regimes,
    extract_abductive_core, fnv_hash, format_partition, project_query_to_regimes,
};
use crate::ptgrs::{CauseFamily, SepMode, sample_composite_episode, sample_episode};
use crate::rng::Rng;
use crate::train::{TrainConfig, train_phisharp};
use std::env;
use std::fs;
use std::process;

fn usage() -> ! {
    eprintln!("Usage:");
    eprintln!("  phisharp train --out out/sharp.phis [--steps 300] [--seed 42]");
    eprintln!(
        "  phisharp sample --out perceived.eff [--family periodic|symmetric|nested|shared|substitution|grammar|composite|stream] [--seed 7]"
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
                    // Flux continu sans aucun retour à la ligne (étiquettes + moules + requête collés)
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
                _ => {}
            }

            let family = match arg_val(&args, "--family") {
                Some(name) => CauseFamily::from_name(&name).unwrap_or_else(|| {
                    eprintln!(
                        "Famille inconnue '{}'. Choix: periodic, symmetric, nested, shared, substitution, grammar, composite, stream.",
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

            // Découverte 100% connexionniste des régimes causaux sur le flux brut
            let regimes = discover_causal_regimes(&model, &bytes, true);
            if regimes.len() >= 2 {
                run_ask_continuous(&model, &file_path, &bytes, &regimes);
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

    println!("=== PHISHARP ABDUCTION REPORT (v0.5 — Monolithique) ===");
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
    regimes: &[DiscoveredRegime],
) {
    let comp = compute_composite_phi(model, bytes);
    let q_idx = regimes.len() - 1;
    let q_reg = regimes[q_idx];
    let q_bytes = &bytes[q_reg.core_start..q_reg.core_end];

    let mut rng = Rng::new(fnv_hash(q_bytes));
    let q_rep = compute_phi_sharp(model, q_bytes, &mut rng, true);
    let matches = project_query_to_regimes(model, bytes, regimes, q_idx);

    println!("=== PHISHARP ABDUCTION REPORT (v0.5 — Flux Continu Auto-Segmenté) ===");
    println!(
        "Effet (E)       : '{}' ({} octets, {} régimes causaux découverts sans séparateur)",
        file_path,
        bytes.len(),
        regimes.len()
    );
    println!(
        "Cohérence scène : sélectivité={:.4} (R_top={:.3} vs R_rest={:.3}, paire [{}]<->[{}])",
        comp.selectivity, comp.r_top, comp.r_rest, comp.top_pair.0, comp.top_pair.1
    );
    println!(
        "φ_sharp(cible)  : {:+.4}  [Gen={:.3} Disc={:.3} Proj={:.3} Gate={:.3} IIT={:.3}, res={:.4}]",
        q_rep.phi_sharp,
        q_rep.generativity,
        q_rep.discrimination,
        q_rep.projection,
        q_rep.mean_gate,
        q_rep.integration,
        q_rep.residual
    );

    println!(
        "\nRégime cible [{}] (octets {}..{}, gate={:.3}) :",
        q_idx, q_reg.core_start, q_reg.core_end, q_reg.mean_gate
    );
    println!("  \"{}\"", printable_preview(q_bytes));

    println!("\nProjection référentielle vers les régimes découverts dans perceived.eff :");
    for (rank, m) in matches.iter().enumerate() {
        let reg = regimes[m.dst_idx];
        let core = &bytes[m.core_start..m.core_end];
        println!(
            "  #{} régime [{}] ({}..{}, gate={:.3}) | résonance={:.4} | err_croisée={:.4} | cos={:+.3} | octets_communs={:.0}%",
            rank + 1,
            m.dst_idx,
            m.core_start,
            m.core_end,
            reg.mean_gate,
            m.score,
            m.cross_err,
            m.cos_sim,
            m.byte_overlap * 100.0
        );
        println!("     noyau   : \"{}\"", printable_preview(core));
        if let Some((rs, re)) = m.referent {
            println!(
                "     référent ({}..{}) : \"{}\"",
                rs,
                re,
                printable_preview(trim_ctrl(&bytes[rs..re]))
            );
        }
    }

    if let Some(best) = matches.first() {
        let margin = matches
            .get(1)
            .map(|second| best.score - second.score)
            .unwrap_or(best.score);
        let out_slice = match best.referent {
            Some((rs, re)) => trim_ctrl(&bytes[rs..re]),
            None => &bytes[best.core_start..best.core_end],
        };
        println!(
            "\nExplication projetée (alphabet de E, marge={:+.4}) :",
            margin
        );
        println!("  {}", printable_preview(out_slice));
    }
}
