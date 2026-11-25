//! Orchestrateur complet Phisharp : entraînement fermé PTGRS et protocole d'évaluation.

use std::env;
use std::path::PathBuf;
use std::process;

use phisharp::eval::run_full_protocol;
use phisharp::math::Rng;
use phisharp::substrate::PhisharpNet;
use phisharp::train::{TrainConfig, train_closed_ptgrs};

fn print_usage() {
    eprintln!(
        "Usage:\n  phisharp train [--iters N] [--out out/sharp.phis] [--seed S]\n  phisharp eval [--model out/sharp.phis] [--seed S]\n  phisharp train-and-eval [--iters N] [--out out/sharp.phis] [--seed S]"
    );
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        process::exit(2);
    }

    let cmd = args[0].as_str();
    let mut iters: usize = 80;
    let mut out_path = PathBuf::from("out/sharp.phis");
    let mut seed: u64 = 2026_11_24;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => {
                i += 1;
                if i < args.len() {
                    iters = args[i].parse().unwrap_or(80);
                }
            }
            "--out" | "--model" => {
                i += 1;
                if i < args.len() {
                    out_path = PathBuf::from(&args[i]);
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    seed = args[i].parse().unwrap_or(2026_11_24);
                }
            }
            other => {
                eprintln!("Option inconnue: {}", other);
                print_usage();
                process::exit(2);
            }
        }
        i += 1;
    }

    match cmd {
        "train" => {
            let cfg = TrainConfig {
                iterations: iters,
                seed,
                ..TrainConfig::default()
            };
            train_closed_ptgrs(&cfg, Some(&out_path));
            println!("Modele sauvegarde dans {:?}", out_path);
        }
        "eval" => {
            let net = match PhisharpNet::load_from_file(&out_path) {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("Chargement impossible {:?} ({}), utilisation d'un modele non entraine pour diagnostic.", out_path, e);
                    let mut rng = Rng::new(seed);
                    PhisharpNet::new_random(&mut rng)
                }
            };
            run_full_protocol(&net, seed);
        }
        "train-and-eval" => {
            let cfg = TrainConfig {
                iterations: iters,
                seed,
                ..TrainConfig::default()
            };
            let (net, _) = train_closed_ptgrs(&cfg, Some(&out_path));
            run_full_protocol(&net, seed);
        }
        _ => {
            print_usage();
            process::exit(2);
        }
    }
}