mod core;
mod eval_iso;
mod model;
mod train_eval;

use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::core::{DEFAULT_BUDGET, RelationalView};
use crate::model::PhisharpV3;
use crate::train_eval::{
    TrainObjective, run_calibration, run_compare, run_eval_ood, run_eval_transfer, train_model,
};

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage (Phisharp V3 — circulation pure L0-L1-L2):");
        eprintln!("  phisharp train         --out out/v3_phi.phis [--objective phi|ll|mdl|untrained] [--steps 250] [--seed 42]");
        eprintln!("  phisharp calibrate     --model out/v3_phi.phis [--seed 101]");
        eprintln!("  phisharp eval-ood      --model out/v3_phi.phis [--seed 202]");
        eprintln!("  phisharp eval-transfer --model out/v3_phi.phis [--seed 303]");
        eprintln!("  phisharp compare       --untrained out/v3_step0.phis --ll out/v3_ll.phis --mdl out/v3_mdl.phis --phi out/v3_phi.phis");
        eprintln!("  phisharp eval-iso      --untrained out/v3_step0.phis --ll out/v3_ll.phis --mdl out/v3_mdl.phis --phi out/v3_phi.phis [--seed 404]");
        eprintln!("  phisharp compare-all   [--dir out] [--steps 250] [--seed 42]");
        eprintln!("  phisharp ask           --model out/v3_phi.phis --file perceived.eff");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "train" => {
            let out = get_arg(&args, "--out").unwrap_or_else(|| "out/v3_phi.phis".to_string());
            let obj_str = get_arg(&args, "--objective").unwrap_or_else(|| "phi".to_string());
            let objective = TrainObjective::from_str(&obj_str)
                .expect("--objective invalide (attendu: phi, ll, mdl, untrained)");
            let steps: usize = get_arg(&args, "--steps")
                .and_then(|s| s.parse().ok())
                .unwrap_or(250);
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(42);
            train_model(Path::new(&out), steps, seed, objective);
        }
        "calibrate" => {
            let model_path =
                get_arg(&args, "--model").unwrap_or_else(|| "out/v3_phi.phis".to_string());
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(101);
            let model = PhisharpV3::load(Path::new(&model_path)).expect("Chargement modèle V3");
            run_calibration(&model, seed);
        }
        "eval-ood" => {
            let model_path =
                get_arg(&args, "--model").unwrap_or_else(|| "out/v3_phi.phis".to_string());
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(202);
            let model = PhisharpV3::load(Path::new(&model_path)).expect("Chargement modèle V3");
            run_eval_ood(&model, seed);
        }
        "eval-transfer" => {
            let model_path =
                get_arg(&args, "--model").unwrap_or_else(|| "out/v3_phi.phis".to_string());
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(303);
            let model = PhisharpV3::load(Path::new(&model_path)).expect("Chargement modèle V3");
            run_eval_transfer(&model, seed);
        }
        "compare" => {
            let p0 = get_arg(&args, "--untrained").expect("--untrained requis");
            let p_ll = get_arg(&args, "--ll").expect("--ll requis");
            let p_mdl = get_arg(&args, "--mdl").expect("--mdl requis");
            let p_phi = get_arg(&args, "--phi").expect("--phi requis");

            let seed_cal: u64 = get_arg(&args, "--seed-cal")
                .and_then(|s| s.parse().ok())
                .unwrap_or(101);
            let seed_ood: u64 = get_arg(&args, "--seed-ood")
                .and_then(|s| s.parse().ok())
                .unwrap_or(202);
            let seed_tr: u64 = get_arg(&args, "--seed-transfer")
                .and_then(|s| s.parse().ok())
                .unwrap_or(303);

            let m0 = PhisharpV3::load(Path::new(&p0)).expect("Chargement --untrained");
            let m_ll = PhisharpV3::load(Path::new(&p_ll)).expect("Chargement --ll");
            let m_mdl = PhisharpV3::load(Path::new(&p_mdl)).expect("Chargement --mdl");
            let m_phi = PhisharpV3::load(Path::new(&p_phi)).expect("Chargement --phi");

            let list = [
                ("Step0 (rand)", &m0),
                ("Baseline LL", &m_ll),
                ("Baseline MDL", &m_mdl),
                ("Phisharp V3", &m_phi),
            ];
            run_compare(&list, seed_cal, seed_ood, seed_tr);
        }
        "eval-iso" => {
            let p0 = get_arg(&args, "--untrained").expect("--untrained requis");
            let p_ll = get_arg(&args, "--ll").expect("--ll requis");
            let p_mdl = get_arg(&args, "--mdl").expect("--mdl requis");
            let p_phi = get_arg(&args, "--phi").expect("--phi requis");
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(404);

            let m0 = PhisharpV3::load(Path::new(&p0)).expect("Chargement --untrained");
            let m_ll = PhisharpV3::load(Path::new(&p_ll)).expect("Chargement --ll");
            let m_mdl = PhisharpV3::load(Path::new(&p_mdl)).expect("Chargement --mdl");
            let m_phi = PhisharpV3::load(Path::new(&p_phi)).expect("Chargement --phi");

            let list = [
                ("Step0 (rand)", &m0),
                ("Baseline LL", &m_ll),
                ("Baseline MDL", &m_mdl),
                ("Phisharp V3", &m_phi),
            ];
            eval_iso::run_eval_iso(&list, seed);
        }
        "compare-all" => {
            let dir = get_arg(&args, "--dir").unwrap_or_else(|| "out".to_string());
            let steps: usize = get_arg(&args, "--steps")
                .and_then(|s| s.parse().ok())
                .unwrap_or(250);
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(42);

            let p0 = Path::new(&dir).join("v3_step0.phis");
            let p_ll = Path::new(&dir).join("v3_ll.phis");
            let p_mdl = Path::new(&dir).join("v3_mdl.phis");
            let p_phi = Path::new(&dir).join("v3_phi.phis");

            let m0 = train_model(&p0, 0, seed, TrainObjective::Untrained);
            let m_ll = train_model(&p_ll, steps, seed, TrainObjective::Likelihood);
            let m_mdl = train_model(&p_mdl, steps, seed, TrainObjective::Mdl);
            let m_phi = train_model(&p_phi, steps, seed, TrainObjective::PhiSharp);

            let list = [
                ("Step0 (rand)", &m0),
                ("Baseline LL", &m_ll),
                ("Baseline MDL", &m_mdl),
                ("Phisharp V3", &m_phi),
            ];
            run_compare(&list, 101, 202, 303);
            eval_iso::run_eval_iso(&list, 404);
        }
        "ask" => {
            let model_path = get_arg(&args, "--model").expect("--model requis");
            let file_path = get_arg(&args, "--file").expect("--file requis");
            let model = PhisharpV3::load(Path::new(&model_path)).expect("Chargement modèle V3");
            let raw = fs::read(&file_path).expect("Lecture perceived.eff");
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
