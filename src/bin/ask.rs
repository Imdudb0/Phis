//! Binaire strict conforme au protocole :
//! `ask --model out/sharp.phis --file perceived.eff`
//!
//! Avec `--trace`, affiche sur stderr la trajectoire φ_sharp(t), la clause focale détectée
//! relationnellement ainsi que les « trous » identifiés par le résidu dans perceived.eff.

use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::process;

use phisharp::abduction::{SearchConfig, run_abductive_search};
use phisharp::substrate::PhisharpNet;

fn print_usage() {
    eprintln!("Usage: ask --model <model.phis> --file <perceived.eff> [--budget <steps>] [--trace]");
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut model_path: Option<PathBuf> = None;
    let mut file_path: Option<PathBuf> = None;
    let mut budget_steps: usize = 12;
    let mut trace = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                if i < args.len() {
                    model_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--file" => {
                i += 1;
                if i < args.len() {
                    file_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--budget" => {
                i += 1;
                if i < args.len() {
                    budget_steps = args[i].parse().unwrap_or(12);
                }
            }
            "--trace" => {
                trace = true;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            other => {
                eprintln!("Argument inconnu: {}", other);
                print_usage();
                process::exit(2);
            }
        }
        i += 1;
    }

    let (Some(mpath), Some(fpath)) = (model_path, file_path) else {
        print_usage();
        process::exit(2);
    };

    let net = match PhisharpNet::load_from_file(&mpath) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Impossible de charger le modele {:?}: {}", mpath, e);
            process::exit(1);
        }
    };

    let perceived_bytes = match std::fs::read(&fpath) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Impossible de lire {:?}: {}", fpath, e);
            process::exit(1);
        }
    };

    let cfg = SearchConfig {
        budget_steps,
        disable_refutation: false,
        max_explain_len: 320,
    };

    let result = run_abductive_search(&net, &perceived_bytes, cfg);

    if trace {
        for pt in &result.trajectory {
            eprintln!(
                "[step {:2}] phi_sharp={:.4} mip={:.4} sens_min={:.3} (G={:.3}, D={:.3}, P={:.3}) res={:.3}",
                pt.step,
                pt.report.phi_sharp,
                pt.report.phi_iit_mip,
                pt.report.sens_min,
                pt.report.generativity,
                pt.report.discrimination,
                pt.report.projection,
                pt.report.residue_score
            );
        }
        eprintln!("[focal_clause_idx] {}", result.focal_clause_idx);
        let holes: Vec<String> = result
            .final_report
            .residue_holes
            .iter()
            .map(|b| format!("{:?}", String::from_utf8_lossy(b)))
            .collect();
        eprintln!("[residue_holes] {}", holes.join(", "));
    }

    let mut stdout = std::io::stdout().lock();
    if let Err(e) = stdout.write_all(&result.explanation_bytes) {
        eprintln!("Erreur ecriture stdout: {}", e);
        process::exit(1);
    }
    let _ = stdout.flush();
}