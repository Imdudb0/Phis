use phisharp::core::flow::{run_abductive_flow, FlowConfig};
use phisharp::core::io::PhisCheckpoint;
use phisharp::core::phi_sharp::{EffectSummary, ModelParams, PhiSharpConfig, StabilityProbes};
use phisharp::core::projection::{project_witness_mirror_ascent, ProjectionConfig};
use phisharp::core::train::{evaluate_post_hoc, train_session, TrainConfig};
use phisharp::ho_ptgrs::generator::{CurriculumConfig, HoPtgrsGenerator};
use phisharp::ho_ptgrs::meta_ops::MetaOp;
use phisharp::ho_ptgrs::rng::Rng;

use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "ask" => cmd_ask(&args[2..]),
        "train" => cmd_train(&args[2..]),
        "gen-eff" => cmd_gen_eff(&args[2..]),
        "-h" | "--help" | "help" => print_usage(),
        other => {
            eprintln!("Commande inconnue : '{}'", other);
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("Phisharp — Moteur de compréhension générale par abduction");
    eprintln!();
    eprintln!("Commandes :");
    eprintln!("  ask     --model out/sharp.phis --file perceived.eff [--steps N] [--out-len L] [--raw]");
    eprintln!("  train   --out out/sharp.phis [--episodes 30] [--n 6] [--d 2] [--h 8] [--lr 0.02] [--seed 42]");
    eprintln!("  gen-eff --out perceived.eff [--op bif|iso|bot|fix|quot] [--horizon 4] [--distractors 4] [--noise 0.5] [--seed 42]");
}

fn cmd_ask(args: &[String]) {
    let mut model_path: Option<String> = None;
    let mut file_path: Option<String> = None;
    let mut steps_override: Option<usize> = None;
    let mut out_len_override: Option<usize> = None;
    let mut raw_output = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => {
                model_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--file" if i + 1 < args.len() => {
                file_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--steps" if i + 1 < args.len() => {
                steps_override = Some(args[i + 1].parse().expect("--steps invalide"));
                i += 2;
            }
            "--out-len" if i + 1 < args.len() => {
                out_len_override = Some(args[i + 1].parse().expect("--out-len invalide"));
                i += 2;
            }
            "--raw" => {
                raw_output = true;
                i += 1;
            }
            other => {
                eprintln!("Option ignorée : {}", other);
                i += 1;
            }
        }
    }

    let model_path = model_path.unwrap_or_else(|| {
        eprintln!("Erreur : --model <chemin.phis> requis.");
        process::exit(1);
    });
    let file_path = file_path.unwrap_or_else(|| {
        eprintln!("Erreur : --file <perceived.eff> requis.");
        process::exit(1);
    });

    let mut ckpt = PhisCheckpoint::load(&model_path).unwrap_or_else(|e| {
        eprintln!("Erreur de chargement du modèle {} : {}", model_path, e);
        process::exit(1);
    });
    let perceived_eff = fs::read(&file_path).unwrap_or_else(|e| {
        eprintln!("Erreur de lecture de {} : {}", file_path, e);
        process::exit(1);
    });

    if let Some(s) = steps_override {
        ckpt.flow_cfg.max_steps = s;
    }
    if let Some(l) = out_len_override {
        ckpt.proj_cfg.out_len = l;
    }

    let effect = EffectSummary::from_bytes(&perceived_eff, ckpt.params.n);
    // Sondes de stabilité déterministes ancrées sur le contenu de E
    let probe_seed = deterministic_content_seed(&perceived_eff);
    let probes = StabilityProbes::build(ckpt.params.n, 2, 0.06, 0.12, probe_seed);

    let traj = run_abductive_flow(
        &ckpt.params,
        &effect,
        &probes,
        &ckpt.phi_cfg,
        &ckpt.flow_cfg,
        None,
    );
    let proj = project_witness_mirror_ascent(
        &ckpt.params,
        &traj.r_final,
        &traj.final_eval,
        &perceived_eff,
        &ckpt.proj_cfg,
    );

    if raw_output {
        io::stdout()
            .write_all(&proj.witness_bytes)
            .expect("Erreur d'écriture stdout");
        return;
    }

    let first = traj.trace.first().unwrap();
    let last = traj.trace.last().unwrap();

    println!("=== Phisharp :: Trajectoire Abductive ===");
    println!(
        "Entrée        : {} ({} octets) | Graphe N={}, d={}, h={}",
        file_path,
        perceived_eff.len(),
        ckpt.params.n,
        ckpt.params.d,
        ckpt.params.h
    );
    println!(
        "Trajectoire   : {} étapes (t={:.3}, variété critique={}, LaSalle monotone={})",
        traj.trace.len().saturating_sub(1),
        last.t,
        traj.converged_to_critical_manifold,
        traj.is_monotone_lyapunov(1e-9)
    );
    println!(
        "phi_sharp     : {:.6} -> {:.6}  (lambda_2={:.6}, arêtes causales={})",
        first.phi_sharp, last.phi_sharp, last.lambda2, last.active_causal_edges
    );
    println!(
        "Résidu / Stab : rho {:.5} -> {:.5} | V_stab={:.5} | R_abs={:.5}",
        first.residual, last.residual, last.v_stab, last.r_abs
    );
    println!(
        "Projection    : {} itérations miroir (Psi={:.5}, confiance={:.1}%, monotone={})",
        proj.trace.len().saturating_sub(1),
        proj.final_psi,
        proj.mean_confidence * 100.0,
        proj.is_monotone_psi(1e-9)
    );
    println!("Témoin (hex)  : {:02X?}", proj.witness_bytes);
    println!("Témoin (NL)   : {}", proj.display_lossy_nl());
}

fn cmd_train(args: &[String]) {
    let mut out_path = String::from("out/sharp.phis");
    let mut n: usize = 6;
    let mut d: usize = 2;
    let mut h: usize = 8;
    let mut train_cfg = TrainConfig::default();
    let mut flow_steps: usize = 16;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" if i + 1 < args.len() => {
                out_path = args[i + 1].clone();
                i += 2;
            }
            "--episodes" if i + 1 < args.len() => {
                train_cfg.episodes = args[i + 1].parse().expect("--episodes invalide");
                i += 2;
            }
            "--n" if i + 1 < args.len() => {
                n = args[i + 1].parse().expect("--n invalide");
                i += 2;
            }
            "--d" if i + 1 < args.len() => {
                d = args[i + 1].parse().expect("--d invalide");
                i += 2;
            }
            "--h" if i + 1 < args.len() => {
                h = args[i + 1].parse().expect("--h invalide");
                i += 2;
            }
            "--lr" if i + 1 < args.len() => {
                train_cfg.lr = args[i + 1].parse().expect("--lr invalide");
                i += 2;
            }
            "--seed" if i + 1 < args.len() => {
                train_cfg.seed = args[i + 1].parse().expect("--seed invalide");
                i += 2;
            }
            "--flow-steps" if i + 1 < args.len() => {
                flow_steps = args[i + 1].parse().expect("--flow-steps invalide");
                i += 2;
            }
            other => {
                eprintln!("Option ignorée : {}", other);
                i += 1;
            }
        }
    }

    let mut rng = Rng::from_seed(train_cfg.seed);
    let mut params = ModelParams::random_init(n, d, h, &mut rng);
    let phi_cfg = PhiSharpConfig::default();
    let flow_cfg = FlowConfig {
        max_steps: flow_steps,
        dt_init: 0.8,
        dt_max: 8.0,
        ..FlowConfig::default()
    };
    let proj_cfg = ProjectionConfig::default();

    println!(
        "=== Phisharp :: Séance unique de méta-entraînement HO-PTGRS ({} épisodes) ===",
        train_cfg.episodes
    );
    println!("Architecture : N={}, d={}, h={} | signal exclusif : phi_sharp(E)", n, d, h);

    let _logs = train_session(&mut params, &phi_cfg, &flow_cfg, &train_cfg, |log| {
        println!(
            "[ep {:>3} | k={} D={}] bytes={:<4} | phi {:+.4} -> {:+.4} | lam2={:.5} | rho {:.3}->{:.3} | edges={:<2} | |g|={:.3}",
            log.episode,
            log.horizon_k,
            log.distractors,
            log.eff_bytes,
            log.phi_init,
            log.phi_final,
            log.lambda2_final,
            log.residual_init,
            log.residual_final,
            log.active_edges,
            log.grad_norm
        );
    });

    // Évaluation post-hoc (strictement hors entraînement)
    let eval = evaluate_post_hoc(
        &params,
        &phi_cfg,
        &flow_cfg,
        &proj_cfg,
        6,
        train_cfg.seed ^ 0xDEAD_BEEF,
    );
    println!("=== Évaluation post-hoc ({} épisodes HO-PTGRS frais) ===", eval.episodes_tested);
    println!(
        "Gain moyen phi_sharp : {:+.5} | lambda_2 moyen : {:.5} | chute rho : {:.4}",
        eval.mean_phi_gain, eval.mean_lambda2, eval.mean_residual_drop
    );
    println!(
        "Témoin dans alphabet(E) : {:.1}% | recouvrement concept caché : {:.1}%",
        eval.mean_witness_alphabet_ratio * 100.0,
        eval.mean_target_byte_overlap * 100.0
    );

    let ckpt = PhisCheckpoint {
        params,
        phi_cfg,
        flow_cfg,
        proj_cfg,
    };
    ckpt.save(&out_path).unwrap_or_else(|e| {
        eprintln!("Échec de sauvegarde {} : {}", out_path, e);
        process::exit(1);
    });
    println!("Modèle sauvegardé dans : {}", out_path);
}

fn cmd_gen_eff(args: &[String]) {
    let mut out_path = String::from("perceived.eff");
    let mut seed: u64 = 42;
    let mut horizon_k: usize = 4;
    let mut num_distractors: usize = 4;
    let mut noise_ratio: f64 = 0.5;
    let mut forced_op: Option<MetaOp> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" if i + 1 < args.len() => {
                out_path = args[i + 1].clone();
                i += 2;
            }
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().expect("--seed invalide");
                i += 2;
            }
            "--horizon" if i + 1 < args.len() => {
                horizon_k = args[i + 1].parse().expect("--horizon invalide");
                i += 2;
            }
            "--distractors" if i + 1 < args.len() => {
                num_distractors = args[i + 1].parse().expect("--distractors invalide");
                i += 2;
            }
            "--noise" if i + 1 < args.len() => {
                noise_ratio = args[i + 1].parse().expect("--noise invalide");
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

    let mut gen = HoPtgrsGenerator::new(
        seed,
        CurriculumConfig {
            horizon_k,
            num_distractors,
            noise_ratio,
            max_attempts: 64,
        },
    );
    let ep = gen
        .generate_episode(forced_op)
        .expect("Échec génération HO-PTGRS");

    fs::write(&out_path, &ep.perceived_eff).expect("Échec écriture perceived.eff");
    println!(
        "Écrit {} ({} octets) | cause cachée post-hoc={:?} | code cible={:02X?}",
        out_path,
        ep.perceived_eff.len(),
        ep.hidden_cause.target_op,
        ep.hidden_cause.target_concept_bytes
    );
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
        other => panic!("Opérateur inconnu : {}", other),
    }
}

fn deterministic_content_seed(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
