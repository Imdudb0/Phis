mod graph;
mod phi;
mod ptgrs;
mod rng;
mod train;

use crate::graph::PhiGraph;
use crate::phi::{
    WmReport, compute_composite_phi, compute_phi_sharp, extract_abductive_core, fnv_hash,
    format_partition, run_working_memory,
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
    eprintln!("  phisharp train --out out/sharp.phis [--steps 450] [--seed 42]");
    eprintln!(
        "  phisharp sample --out perceived.eff [--family periodic|symmetric|nested|shared|substitution|grammar|composite|stream|multihop|multihop_flat|hop3] [--seed 7]"
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
                .unwrap_or(450);
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
                Some("multihop") | Some("multihop_flat") | Some("hop3") => {
                    let fam = arg_val(&args, "--family").unwrap();
                    let flat = fam == "multihop_flat";
                    let hops = if fam == "hop3" { 3 } else { 2 };
                    let ep = sample_multihop_episode(&mut rng, flat, true, hops);
                    if let Err(e) = fs::write(&out_path, &ep.effect) {
                        eprintln!("Erreur écriture '{}': {}", out_path, e);
                        process::exit(1);
                    }
                    let causes_str = ep
                        .hidden_causes
                        .iter()
                        .map(|c| c.name())
                        .collect::<Vec<_>>()
                        .join(" + ");
                    println!(
                        "Scène Hop-{}{} générée dans '{}' ({} octets, concept caché = {} [{}])",
                        hops,
                        if flat { " (flux plat)" } else { "" },
                        out_path,
                        ep.effect.len(),
                        printable_preview(&ep.hidden_concept_name),
                        causes_str
                    );
                    println!(
                        "  Moules : {}",
                        ep.hidden_prim_names
                            .iter()
                            .map(|n| printable_preview(n))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    println!("\n{}", String::from_utf8_lossy(&ep.effect));
                    return;
                }
                _ => {}
            }

            let family = match arg_val(&args, "--family") {
                Some(name) => CauseFamily::from_name(&name).unwrap_or_else(|| {
                    eprintln!(
                        "Famille inconnue '{}'. Choix: periodic, symmetric, nested, shared, substitution, grammar, composite, stream, multihop, multihop_flat, hop3.",
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

            if let Some(wm) = run_working_memory(&model, &bytes) {
                run_ask_wm(&model, &file_path, &bytes, &wm);
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

    println!("=== PHISHARP ABDUCTION REPORT (v0.7 — Monolithique) ===");
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

fn run_ask_wm(model: &PhiGraph, file_path: &str, bytes: &[u8], wm: &WmReport) {
    let comp = compute_composite_phi(model, bytes);

    println!("=== PHISHARP ABDUCTION REPORT (v0.7 — Mémoire de Travail 100% Connexionniste) ===");
    println!(
        "Effet (E)       : '{}' ({} octets, contexte 0..{}, {} moules, {} cause(s) agrégée(s) en Z_Q)",
        file_path,
        bytes.len(),
        wm.context_end,
        wm.molds_count,
        wm.target_causes
    );
    println!(
        "φ_wm (mémoire)  : {:+.4}  [cos_tête={:.3}, coupe_échos={:.3}, gain IIT échos={:+.3}]",
        wm.wm_phi, wm.head_cos, wm.head_cos_cut, wm.iit_echo_gain
    );
    println!(
        "Sélectivité WM  : HSel={:.3} | RSel={:.3} | RDisc={:.3}  (scène: sélec={:.3}, paire [{}]<->[{}])",
        wm.head_sel, wm.readout_sel, wm.readout_disc, comp.selectivity, comp.top_pair.0, comp.top_pair.1
    );

    println!("\nMoules primitifs injectés dans Z^(0) :");
    for (idx, reg) in wm.mold_regimes.iter().enumerate() {
        let ref_txt = reg
            .referent
            .map(|(rs, re)| format!(" | réf=\"{}\"", printable_preview(trim_ctrl(&bytes[rs..re]))))
            .unwrap_or_default();
        println!(
            "  [{}] ({}..{}, gate={:.2}) \"{}\"{}",
            idx,
            reg.core_start,
            reg.core_end,
            reg.mean_gate,
            printable_preview(&bytes[reg.core_start..reg.core_end]),
            ref_txt
        );
    }

    for (ci, &(ts, te)) in wm.target_spans.iter().enumerate() {
        let q_bytes = &bytes[ts..te];
        let mut rng = Rng::new(fnv_hash(q_bytes));
        let q_rep = compute_phi_sharp(model, q_bytes, &mut rng, true);
        println!(
            "\nRégime cible #{} ({}..{}) | φ={:+.4} [Gen={:.3} Disc={:.3} Proj={:.3} IIT={:.3} res={:.4}] :",
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
    }

    println!("\nChamp continu de lecture P(t) sur le contexte (zéro parsing symbolique) :");
    for (rank, sp) in wm.top_spans.iter().enumerate() {
        println!(
            "  #{} ({:>3}..{:<3}) | P_pic={:.4} (moy={:.3}) | cos(Z,Zq)={:+.3} | ||Z||={:.2} | g={:.2} | écho={:.2} | \"{}\"",
            rank + 1,
            sp.start,
            sp.end,
            sp.peak_p,
            sp.mean_p,
            sp.mean_cos,
            sp.mean_norm,
            sp.mean_gate,
            sp.mean_back_echo,
            printable_preview(trim_ctrl(&bytes[sp.start..sp.end]))
        );
    }

    if let Some(best) = wm.top_spans.first() {
        let margin = wm
            .top_spans
            .get(1)
            .map(|sec| best.peak_p - sec.peak_p)
            .unwrap_or(best.peak_p);
        println!(
            "\nExplication projetée (champ neuronal P(t), alphabet de E, marge={:+.4}) :",
            margin
        );
        println!("  {}", printable_preview(trim_ctrl(&bytes[best.start..best.end])));
    }
}
