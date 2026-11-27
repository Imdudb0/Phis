mod autograd;
mod core;
mod eval_iso;
mod model;
mod train_eval;

use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::core::{DEFAULT_BUDGET, RelationalView};
use crate::model::PhisharpV4;
use crate::train_eval::{
    TrainObjective, run_calibration, run_compare, run_grad_check, train_model,
};

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage (Phisharp V4 — non-markovien + autograd Adam exact):");
        eprintln!("  phisharp grad-check  [--seed 42]");
        eprintln!("  phisharp train       --out out/v4_phi.phis [--objective phi|ll|mdl|untrained] [--steps 120] [--seed 42]");
        eprintln!("  phisharp calibrate   --model out/v4_phi.phis [--seed 101]");
        eprintln!("  phisharp compare     --untrained out/v4_step0.phis --ll out/v4_ll.phis --mdl out/v4_mdl.phis --phi out/v4_phi.phis");
        eprintln!("  phisharp eval-iso    --untrained out/v4_step0.phis --ll out/v4_ll.phis --mdl out/v4_mdl.phis --phi out/v4_phi.phis [--seed 404]");
        eprintln!("  phisharp compare-all [--dir out] [--steps 120] [--seed 42]");
        eprintln!("  phisharp ask         --model out/v4_phi.phis --file perceived.eff");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "grad-check" => {
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(42);
            run_grad_check(seed);
        }
        "train" => {
            let out = get_arg(&args, "--out").unwrap_or_else(|| "out/v4_phi.phis".to_string());
            let obj_str = get_arg(&args, "--objective").unwrap_or_else(|| "phi".to_string());
            let obj = TrainObjective::from_str(&obj_str)
                .expect("--objective invalide (phi, ll, mdl, untrained)");
            let steps: usize = get_arg(&args, "--steps")
                .and_then(|s| s.parse().ok())
                .unwrap_or(120);
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(42);
            train_model(Path::new(&out), steps, seed, obj);
        }
        "calibrate" => {
            let path = get_arg(&args, "--model").unwrap_or_else(|| "out/v4_phi.phis".to_string());
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(101);
            let model = PhisharpV4::load(Path::new(&path)).expect("Chargement modèle V4");
            run_calibration(&model, seed);
        }
        "compare" => {
            let m0 = PhisharpV4::load(Path::new(&get_arg(&args, "--untrained").expect("--untrained")))
                .expect("load step0");
            let m_ll = PhisharpV4::load(Path::new(&get_arg(&args, "--ll").expect("--ll")))
                .expect("load ll");
            let m_mdl = PhisharpV4::load(Path::new(&get_arg(&args, "--mdl").expect("--mdl")))
                .expect("load mdl");
            let m_phi = PhisharpV4::load(Path::new(&get_arg(&args, "--phi").expect("--phi")))
                .expect("load phi");
            let list = [
                ("Step0 (rand)", &m0),
                ("Baseline LL", &m_ll),
                ("Baseline MDL", &m_mdl),
                ("Phisharp V4", &m_phi),
            ];
            run_compare(&list, 101, 202, 303);
        }
        "eval-iso" => {
            let m0 = PhisharpV4::load(Path::new(&get_arg(&args, "--untrained").expect("--untrained")))
                .expect("load step0");
            let m_ll = PhisharpV4::load(Path::new(&get_arg(&args, "--ll").expect("--ll")))
                .expect("load ll");
            let m_mdl = PhisharpV4::load(Path::new(&get_arg(&args, "--mdl").expect("--mdl")))
                .expect("load mdl");
            let m_phi = PhisharpV4::load(Path::new(&get_arg(&args, "--phi").expect("--phi")))
                .expect("load phi");
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(404);
            let list = [
                ("Step0 (rand)", &m0),
                ("Baseline LL", &m_ll),
                ("Baseline MDL", &m_mdl),
                ("Phisharp V4", &m_phi),
            ];
            eval_iso::run_eval_iso(&list, seed);
        }
        "compare-all" => {
            let dir = get_arg(&args, "--dir").unwrap_or_else(|| "out".to_string());
            let steps: usize = get_arg(&args, "--steps")
                .and_then(|s| s.parse().ok())
                .unwrap_or(120);
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(42);

            let m0 = train_model(&Path::new(&dir).join("v4_step0.phis"), 0, seed, TrainObjective::Untrained);
            let m_ll = train_model(&Path::new(&dir).join("v4_ll.phis"), steps, seed, TrainObjective::Likelihood);
            let m_mdl = train_model(&Path::new(&dir).join("v4_mdl.phis"), steps, seed, TrainObjective::Mdl);
            let m_phi = train_model(&Path::new(&dir).join("v4_phi.phis"), steps, seed, TrainObjective::PhiSharp);

            let list = [
                ("Step0 (rand)", &m0),
                ("Baseline LL", &m_ll),
                ("Baseline MDL", &m_mdl),
                ("Phisharp V4", &m_phi),
            ];
            run_compare(&list, 101, 202, 303);
            eval_iso::run_eval_iso(&list, 404);
        }
        "ask" => {
            let path = get_arg(&args, "--model").expect("--model requis");
            let file = get_arg(&args, "--file").expect("--file requis");
            let model = PhisharpV4::load(Path::new(&path)).expect("Chargement modèle V4");
            let raw = fs::read(&file).expect("Lecture perceived.eff");
            let view = RelationalView::from_bytes(&raw);
            let out_bytes = model.ask_emit(&view, DEFAULT_BUDGET + 2, 48);
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(&out_bytes).expect("Écriture stdout");
        }
        other => {
            eprintln!("Commande inconnue : {}", other);
            std::process::exit(1);
        }
    }
}
