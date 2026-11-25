mod graph;
mod phi;
mod ptgrs;
mod rng;
mod train;

use crate::graph::PhiGraph;
use crate::phi::{compute_phi_sharp, extract_core_motif, format_partition};
use crate::ptgrs::{CauseFamily, sample_episode};
use crate::rng::Rng;
use crate::train::{TrainConfig, train_phisharp};
use std::env;
use std::fs;
use std::process;

fn usage() -> ! {
    eprintln!("Usage:");
    eprintln!("  phisharp train --out out/sharp.phis [--steps 250] [--seed 42]");
    eprintln!("  phisharp sample --out perceived.eff [--family periodic|symmetric|nested|shared|arithmetic] [--seed 7]");
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
                .unwrap_or(250);
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
            let family = match arg_val(&args, "--family") {
                Some(name) => CauseFamily::from_name(&name).unwrap_or_else(|| {
                    eprintln!("Famille inconnue '{}'. Choix: periodic, symmetric, nested, shared, arithmetic.", name);
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

            // Graine déterministe dérivée de E pour une évaluation reproductible
            let mut hash = 0xCBF2_9CE4_8422_2325u64;
            for &b in &bytes {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100_0000_01B3);
            }
            let mut rng = Rng::new(hash);
            let rep = compute_phi_sharp(&model, &bytes, &mut rng, true);

            println!("=== PHISHARP ABDUCTION REPORT ===");
            println!("Effet (E)       : '{}' ({} octets)", file_path, bytes.len());
            println!("φ_sharp         : {:+.4}", rep.phi_sharp);
            println!("  Générativité  : {:.4}  (résidu moyen = {:.4})", rep.generativity, rep.residual);
            println!("  Discrimination: {:.4}", rep.discrimination);
            println!(
                "  Projection    : {:.4}  [réfut={:.3} | dépl={:.3} | abst={:.3}]",
                rep.projection, rep.refutation_stab, rep.displacement_stab, rep.abstraction_stab
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
            println!("  {}  (gain={:+.4})", traj_str, rep.traj_gain);

            let (start, motif) = extract_core_motif(&bytes, &rep.trace.final_pos_err);
            println!("\nNoyau causal auto-spécifié (offset {}, alphabet de E) :", start);
            println!("  ASCII : \"{}\"", printable_preview(&motif));
            println!("  HEX   : {}", hex_preview(&motif));

            // Top 5 positions qui résistent le plus à l'explication (le résidu)
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
                println!("  offset {:>3} | 0x{:02x} '{}' | résidu = {:.4}", pos, b, ch, err);
            }
        }

        _ => usage(),
    }
}
