#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use phisharp::backbone::{
    BackboneConfig, RelationalBackbone, SubTbTrainConfig, SubTbTrainer,
};
use phisharp::hypothesis_graph::{
    Block, EdgeKind, GraphConfig, HypothesisGraph, NodeId, NodeKind,
};
use phisharp::memory_controller::{AskConfig, AskController, ArchivedMode};
use phisharp::model_io::{ModelMetadata, load_phis, save_phis};
use phisharp::projector::ObservationGraph;
use phisharp::ptgrs_synth::{PtgrsSynthesizer, SynthConfig};
use phisharp::refutation::{RefutationConfig, RuleStatus, evaluate_phi_sharp};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print_usage();
        return ExitCode::SUCCESS;
    }

    let result = match args[0].as_str() {
        "train" => run_train(&args[1..]),
        "ask" => run_ask(&args[1..]),
        "inspect" => run_inspect(&args[1..]),
        other => Err(format!("Commande inconnue : '{other}'. Utilise --help.")),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("Erreur : {err}");
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!(
        "Phisharp V1 — Moteur abductif PTGRS + GFlowNet SubTB (Rust 2024)\n\
         \n\
         COMMANDES :\n\
           train [OPTIONS]\n\
               Entraîne le backbone relationnel sur le synthétiseur PTGRS avec curriculum\n\
               et sauvegarde les poids au format binaire `.phis`.\n\
               Options :\n\
                 --out <PATH>            Fichier de sortie (défaut : out/sharp.phis)\n\
                 --resume <PATH>         Reprend depuis un checkpoint `.phis` existant\n\
                 --steps <N>             Pas d'entraînement SubTB (défaut : 120)\n\
                 --batch-size <N>        Effets PTGRS par pas (défaut : 2)\n\
                 --hidden-dim <D>        Dimension cachée D (défaut : 24)\n\
                 --layers <L>            Couches non-réciproques L (défaut : 2)\n\
                 --lr <F>                Taux d'apprentissage AdamW (défaut : 0.003)\n\
                 --max-traj-steps <N>    Profondeur max de trajectoire SubTB (défaut : 36)\n\
                 --max-difficulty <0..3> Niveau final du curriculum PTGRS (défaut : 3)\n\
                 --eval-samples <N>      Tâches d'évaluation hors-ligne en fin de run (défaut : 4)\n\
                 --eval-budget <B>       Budget Back-and-Forth en évaluation (défaut : 120)\n\
                 --log-every <N>         Fréquence d'affichage des métriques (défaut : 10)\n\
                 --seed <U64>            Graine déterministe (défaut : 1346914643)\n\
         \n\
           ask --model <PATH> --file <PATH> [OPTIONS]\n\
               Exécute la recherche abductive Back-and-Forth guidée par résidu sur `perceived.eff`.\n\
               Options :\n\
                 --model <PATH>          Modèle binaire `.phis` (ex. out/sharp.phis)\n\
                 --file <PATH>           Fichier d'effet brut en octets (ex. perceived.eff)\n\
                 --budget <B>            Budget fixe de transitions P_F + P_B (défaut : 120)\n\
                 --max-forward <N>       Longueur max d'une marche avant (défaut : 36)\n\
                 --max-modes <K>         Nombre max de bassins causaux archivés (défaut : 4)\n\
                 --max-nodes <N>         Borne V_max du graphe d'hypothèse (défaut : auto)\n\
                 --max-rules <N>         Borne R_max sur les règles (défaut : 3)\n\
                 --max-edges <N>         Borne E_max sur les arêtes (défaut : auto)\n\
                 --struct-rels <U8>      Nombre de relations structurelles (défaut : 1)\n\
                 --temp-f <F>            Température softmax P_F (défaut : 0.85)\n\
                 --temp-b <F>            Température softmax P_B (défaut : 1.0)\n\
                 --seed <U64>            Graine d'inférence (défaut : 1346914643)\n\
                 --show-trace            Affiche la trajectoire complète de chaque bassin\n\
         \n\
           inspect --model <PATH>\n\
               Vérifie le checksum et affiche les métadonnées d'un fichier `.phis`."
    );
}

// ============================================================================
// Parsing d'arguments minimal (zéro dépendance)
// ============================================================================

fn next_val<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| format!("Valeur manquante après {flag}"))
}

fn parse_val<T: std::str::FromStr>(raw: &str, flag: &str) -> Result<T, String> {
    raw.parse::<T>()
        .map_err(|_| format!("Valeur invalide pour {flag} : '{raw}'"))
}

// ============================================================================
// Commande 1 : `train` (Curriculum PTGRS + GFlowNet SubTB + Validation)
// ============================================================================

struct TrainCliArgs {
    out_path: PathBuf,
    resume_path: Option<PathBuf>,
    steps: usize,
    batch_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    lr: f64,
    max_traj_steps: usize,
    guided_prob: f64,
    max_difficulty: u8,
    eval_samples: usize,
    eval_budget: usize,
    log_every: usize,
    seed: u64,
}

impl Default for TrainCliArgs {
    fn default() -> Self {
        Self {
            out_path: PathBuf::from("out/sharp.phis"),
            resume_path: None,
            steps: 120,
            batch_size: 2,
            hidden_dim: 24,
            num_layers: 2,
            lr: 3e-3,
            max_traj_steps: 36,
            guided_prob: 0.6,
            max_difficulty: 3,
            eval_samples: 4,
            eval_budget: 120,
            log_every: 10,
            seed: 0x5048_4953,
        }
    }
}

fn parse_train_args(args: &[String]) -> Result<TrainCliArgs, String> {
    let mut cli = TrainCliArgs::default();
    let mut i = 0usize;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--out" => cli.out_path = PathBuf::from(next_val(args, &mut i, flag)?),
            "--resume" => cli.resume_path = Some(PathBuf::from(next_val(args, &mut i, flag)?)),
            "--steps" => cli.steps = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--batch-size" => cli.batch_size = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--hidden-dim" => cli.hidden_dim = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--layers" => cli.num_layers = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--lr" => cli.lr = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--max-traj-steps" => {
                cli.max_traj_steps = parse_val(next_val(args, &mut i, flag)?, flag)?
            }
            "--max-difficulty" => {
                let d: u8 = parse_val(next_val(args, &mut i, flag)?, flag)?;
                if d > 3 {
                    return Err("--max-difficulty doit être compris entre 0 et 3".into());
                }
                cli.max_difficulty = d;
            }
            "--guided-prob" => {
                let p: f64 = parse_val(next_val(args, &mut i, flag)?, flag)?;
                if !(0.0..=1.0).contains(&p) {
                    return Err("--guided-prob doit être compris entre 0.0 et 1.0".into());
                }
                cli.guided_prob = p;
            }
            "--eval-samples" => cli.eval_samples = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--eval-budget" => cli.eval_budget = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--log-every" => cli.log_every = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--seed" => cli.seed = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("Option inconnue pour train : '{other}'")),
        }
        i += 1;
    }
    if cli.steps == 0 || cli.batch_size == 0 {
        return Err("--steps et --batch-size doivent être >= 1".into());
    }
    Ok(cli)
}

fn run_train(args: &[String]) -> Result<(), String> {
    let cli = parse_train_args(args)?;
    let start_time = Instant::now();

    let (mut backbone, mut prev_steps) = if let Some(ref resume) = cli.resume_path {
        let loaded = load_phis(resume).map_err(|e| format!("Chargement {:?} : {e}", resume))?;
        println!(
            "Reprise depuis {:?} (D={}, L={}, {} params, {} pas précédents)",
            resume,
            loaded.backbone.config().hidden_dim,
            loaded.backbone.config().num_layers,
            loaded.backbone.param_count(),
            loaded.metadata.trained_steps
        );
        (loaded.backbone, loaded.metadata.trained_steps)
    } else {
        let bb = RelationalBackbone::new(
            BackboneConfig {
                hidden_dim: cli.hidden_dim,
                num_layers: cli.num_layers,
            },
            cli.seed,
        )
        .map_err(|e| e.to_string())?;
        println!(
            "Initialisation Phisharp V1 (D={}, L={}, {} paramètres)",
            bb.config().hidden_dim,
            bb.config().num_layers,
            bb.param_count()
        );
        (bb, 0)
    };

    let synth_cfg = SynthConfig {
        difficulty: 0,
        min_effect_nodes: 3,
        max_effect_nodes: 10,
        ..SynthConfig::default()
    };
    let mut synth = PtgrsSynthesizer::new(synth_cfg, cli.seed ^ 0xA5A5_5A5A)
        .map_err(|e| e.to_string())?;

    let train_cfg = SubTbTrainConfig {
        learning_rate: cli.lr,
        max_trajectory_steps: cli.max_traj_steps,
        guided_explore_prob: cli.guided_prob,
        ..SubTbTrainConfig::default()
    };
    let mut trainer = SubTbTrainer::new(&backbone, train_cfg, 32, 64, cli.seed ^ 0x1234_5678);

    let mut last_loss = 0.0;
    let mut last_log_phi = 0.0;
    let mut current_diff = 0u8;
    let stages = usize::from(cli.max_difficulty) + 1;

    println!(
        "Démarrage training SubTB : {} pas, batch={}, curriculum 0..={}",
        cli.steps, cli.batch_size, cli.max_difficulty
    );

    for step in 0..cli.steps {
        let target_diff = ((step * stages) / cli.steps).min(usize::from(cli.max_difficulty)) as u8;
        if target_diff != current_diff {
            current_diff = target_diff;
            synth.set_difficulty(current_diff);
            println!("  [Curriculum] Passage au niveau de difficulté {current_diff}");
        }

        let batch = synth.next_training_batch(cli.batch_size);
        if batch.is_empty() {
            return Err("Le synthétiseur PTGRS n'a pas pu produire de batch".into());
        }

        let metrics = trainer.train_step(&mut backbone, &batch);
        last_loss = metrics.mean_subtb_loss;
        last_log_phi = metrics.mean_log_phi_sharp;
        prev_steps += 1;

        let step_num = step + 1;
        if cli.log_every > 0
            && (step_num == 1 || step_num % cli.log_every == 0 || step_num == cli.steps)
        {
            println!(
                "  pas {:>4}/{} | diff={} | loss={:>8.4} | log_phi={:>7.3} | res={:>5.2} | |g|={:>6.3} | replay={}",
                step_num,
                cli.steps,
                current_diff,
                metrics.mean_subtb_loss,
                metrics.mean_log_phi_sharp,
                metrics.mean_terminal_residual,
                metrics.grad_norm,
                trainer.replay_len()
            );
        }
    }

    let meta = ModelMetadata {
        trained_steps: prev_steps,
        max_difficulty: cli.max_difficulty,
        last_mean_loss: last_loss,
        last_mean_log_phi: last_log_phi,
    };
    let written_bytes = save_phis(&cli.out_path, &backbone, meta)
        .map_err(|e| format!("Sauvegarde {:?} : {e}", cli.out_path))?;
    println!(
        "Modèle sauvegardé dans {:?} ({written_bytes} octets, {:.2}s)",
        cli.out_path,
        start_time.elapsed().as_secs_f64()
    );

    // Validation hors-ligne optionnelle sur EvalSample (cause cachée utilisée uniquement ici).
    if cli.eval_samples > 0 {
        run_offline_validation(
            &mut backbone,
            cli.max_difficulty,
            cli.eval_samples,
            cli.eval_budget,
            cli.seed ^ 0xDEAD_BEEF,
        )?;
    }

    Ok(())
}

fn run_offline_validation(
    backbone: &mut RelationalBackbone,
    difficulty: u8,
    eval_samples: usize,
    eval_budget: usize,
    seed: u64,
) -> Result<(), String> {
    let eval_synth_cfg = SynthConfig {
        difficulty,
        min_effect_nodes: 3,
        max_effect_nodes: 8,
        ..SynthConfig::default()
    };
    let mut eval_synth = PtgrsSynthesizer::new(eval_synth_cfg, seed).map_err(|e| e.to_string())?;
    let eval_batch = eval_synth.next_eval_batch(eval_samples);

    let ask_cfg = AskConfig {
        budget_steps: eval_budget,
        max_forward_run: 36,
        forward_temperature: 0.85,
        ..AskConfig::default()
    };
    
    let controller = AskController::new(ask_cfg);

    let mut exact_hits = 0usize;
    let mut sum_res_ratio = 0.0;
    let mut sum_phi_gap = 0.0;

    println!("\nValidation hors-ligne ({} tâches PTGRS inédites) :", eval_batch.len());
    for (idx, sample) in eval_batch.iter().enumerate() {
        let base_res = evaluate_phi_sharp(
            &HypothesisGraph::new(sample.training.recommended_graph_cfg.clone()),
            &sample.training.effect,
            RefutationConfig::default(),
        )
        .residual
        .norm;

        let res = controller.ask(
            &sample.training.effect,
            sample.training.recommended_graph_cfg.clone(),
            backbone,
            seed.wrapping_add(idx as u64),
        );

        if let Some(best) = res.modes.first() {
            let ratio = if base_res > 0.0 {
                best.report.residual.norm / base_res
            } else {
                0.0
            };
            let phi_gap = sample.hidden_report.log_phi_sharp - best.report.log_phi_sharp;
            if best.report.residual.is_zero() {
                exact_hits += 1;
            }
            sum_res_ratio += ratio;
            sum_phi_gap += phi_gap;
            println!(
                "  #{:<2} {:<20?} | |E|={:<2} | R_0={:>5.2} -> R*={:>5.2} | log_phi*={:>7.3} (H_true={:>7.3}) | modes={}",
                idx + 1,
                sample.archetype,
                sample.training.effect.node_count(),
                base_res,
                best.report.residual.norm,
                best.report.log_phi_sharp,
                sample.hidden_report.log_phi_sharp,
                res.modes.len()
            );
        }
    }

    if !eval_batch.is_empty() {
        let n = eval_batch.len() as f64;
        println!(
            "  Bilan eval : résidus nuls = {exact_hits}/{} | ratio résiduel moyen = {:.3} | écart moyen log_phi = {:.3}",
            eval_batch.len(),
            sum_res_ratio / n,
            sum_phi_gap / n
        );
    }
    Ok(())
}

// ============================================================================
// Commande 2 : `ask --model out/sharp.phis --file perceived.eff`
// ============================================================================

struct AskCliArgs {
    model_path: Option<PathBuf>,
    file_path: Option<PathBuf>,
    budget_steps: usize,
    max_forward_run: usize,
    max_modes: usize,
    max_nodes: Option<usize>,
    max_rules: usize,
    max_edges: Option<usize>,
    num_struct_rels: u8,
    temp_f: f64,
    temp_b: f64,
    seed: u64,
    show_trace: bool,
}

impl Default for AskCliArgs {
    fn default() -> Self {
        Self {
            model_path: None,
            file_path: None,
            budget_steps: 120,
            max_forward_run: 36,
            max_modes: 4,
            max_nodes: None,
            max_rules: 3,
            max_edges: None,
            num_struct_rels: 1,
            temp_f: 0.85,
            temp_b: 1.0,
            seed: 0x5048_4953,
            show_trace: false,
        }
    }
}

fn parse_ask_args(args: &[String]) -> Result<AskCliArgs, String> {
    let mut cli = AskCliArgs::default();
    let mut i = 0usize;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--model" => cli.model_path = Some(PathBuf::from(next_val(args, &mut i, flag)?)),
            "--file" => cli.file_path = Some(PathBuf::from(next_val(args, &mut i, flag)?)),
            "--budget" => cli.budget_steps = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--max-forward" => {
                cli.max_forward_run = parse_val(next_val(args, &mut i, flag)?, flag)?
            }
            "--max-modes" => cli.max_modes = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--max-nodes" => {
                cli.max_nodes = Some(parse_val(next_val(args, &mut i, flag)?, flag)?)
            }
            "--max-rules" => cli.max_rules = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--max-edges" => {
                cli.max_edges = Some(parse_val(next_val(args, &mut i, flag)?, flag)?)
            }
            "--struct-rels" => {
                cli.num_struct_rels = parse_val(next_val(args, &mut i, flag)?, flag)?
            }
            "--temp-f" => cli.temp_f = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--temp-b" => cli.temp_b = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--seed" => cli.seed = parse_val(next_val(args, &mut i, flag)?, flag)?,
            "--show-trace" => cli.show_trace = true,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("Option inconnue pour ask : '{other}'")),
        }
        i += 1;
    }
    Ok(cli)
}

fn run_ask(args: &[String]) -> Result<(), String> {
    let cli = parse_ask_args(args)?;
    let model_path = cli
        .model_path
        .ok_or("Option requise manquante : --model <PATH>")?;
    let file_path = cli
        .file_path
        .ok_or("Option requise manquante : --file <PATH>")?;

    let mut loaded =
        load_phis(&model_path).map_err(|e| format!("Chargement {:?} : {e}", model_path))?;
    let raw_bytes =
        fs::read(&file_path).map_err(|e| format!("Lecture {:?} : {e}", file_path))?;
    if raw_bytes.is_empty() {
        return Err(format!("Le fichier {:?} est vide (0 octet)", file_path));
    }

    let obs = ObservationGraph::from_bytes(&raw_bytes);
    let alphabet = obs.alphabet();
    let max_nodes = cli.max_nodes.unwrap_or_else(|| (obs.node_count() + 6).max(12));
    let max_edges = cli
        .max_edges
        .unwrap_or_else(|| (obs.edge_count() + obs.node_count() + 14).max(20));

    let graph_cfg = GraphConfig::new(
        alphabet.clone(),
        cli.num_struct_rels.max(1),
        max_nodes,
        cli.max_rules,
        max_edges,
    )
    .map_err(|e| e.to_string())?;

    let ask_cfg = AskConfig {
        budget_steps: cli.budget_steps,
        max_forward_run: cli.max_forward_run,
        max_modes: cli.max_modes,
        forward_temperature: cli.temp_f,
        backward_temperature: cli.temp_b,
        ..AskConfig::default()
    };

    let t0 = Instant::now();
    let controller = AskController::new(ask_cfg);
    let result = controller.ask(&obs, graph_cfg, &mut loaded.backbone, cli.seed);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("=== Phisharp V1 — Résultat abductif ===");
    println!(
        "Modèle : {:?} (D={}, L={}, {} pas d'entraînement)",
        model_path,
        loaded.backbone.config().hidden_dim,
        loaded.backbone.config().num_layers,
        loaded.metadata.trained_steps
    );
    println!(
        "Effet  : {:?} ({} octets, |Sigma_E|={}, aperçu={})",
        file_path,
        raw_bytes.len(),
        alphabet.len(),
        format_byte_preview(&raw_bytes, 48)
    );
    println!(
        "Session: {}/{} pas (P_F={}, P_B={}), {} cycles, {} restarts, {} actions masquées (nogoods), {} règles parasites apprises ({:.1} ms)",
        result.stats.steps_used,
        cli.budget_steps,
        result.stats.forward_steps,
        result.stats.backward_steps,
        result.stats.cycles_completed,
        result.stats.full_restarts,
        result.stats.actions_masked_by_nogoods,
        result.stats.parasitic_patterns_learned,
        elapsed_ms
    );
    let cache = &result.stats.cache_stats;
    println!(
        "Cache  : résidus {} hits / {} misses | réfutations {} hits / {} misses",
        cache.residual_hits, cache.residual_misses, cache.refutation_hits, cache.refutation_misses
    );

    let amb = &result.ambiguity;
    println!(
        "\nAmbiguïté causale : {} | bassins={} | N_eff={:.2} | entropie_norm={:.3} | écart_top12={}",
        if amb.is_ambiguous {
            "DÉTECTÉE (multi-causes concurrentes)"
        } else {
            "non détectée"
        },
        amb.mode_count,
        amb.effective_modes,
        amb.normalized_entropy,
        amb.top_score_gap
            .map_or_else(|| "n/a".into(), |g| format!("{g:.3}"))
    );

    if result.modes.is_empty() {
        println!("\nAucun bassin causal découvert dans le budget imparti.");
        return Ok(());
    }

    for (rank, mode) in result.modes.iter().enumerate() {
        let weight = amb.mode_weights.get(rank).copied().unwrap_or(0.0);
        print_archived_mode(rank + 1, weight, mode, cli.show_trace);
    }
    Ok(())
}

// ============================================================================
// Commande 3 : `inspect --model out/sharp.phis`
// ============================================================================

fn run_inspect(args: &[String]) -> Result<(), String> {
    let mut model_path: Option<PathBuf> = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => model_path = Some(PathBuf::from(next_val(args, &mut i, "--model")?)),
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            other => return Err(format!("Option inconnue pour inspect : '{other}'")),
        }
        i += 1;
    }
    let path = model_path.ok_or("Option requise manquante : --model <PATH>")?;
    let loaded = load_phis(&path).map_err(|e| format!("Inspection {:?} : {e}", path))?;
    let cfg = loaded.backbone.config();
    let params = loaded.backbone.params();
    let l2: f64 = params.iter().map(|&w| w * w).sum::<f64>().sqrt();
    let max_abs: f64 = params.iter().map(|&w| w.abs()).fold(0.0, f64::max);

    println!("Fichier .phis valide (checksum FNV-1a OK) : {:?}", path);
    println!(
        "  Architecture : hidden_dim={}, num_layers={}, params={}",
        cfg.hidden_dim,
        cfg.num_layers,
        params.len()
    );
    println!(
        "  Entraînement : steps={}, max_difficulty={}, last_loss={:.4}, last_log_phi={:.3}",
        loaded.metadata.trained_steps,
        loaded.metadata.max_difficulty,
        loaded.metadata.last_mean_loss,
        loaded.metadata.last_mean_log_phi
    );
    println!("  Poids        : norme L2={l2:.4}, max|w|={max_abs:.4}");
    Ok(())
}

// ============================================================================
// Affichage lisible d'une cause PTGRS (SEED, Règles, VAR/SAME_AS, Résidu)
// ============================================================================

fn format_byte(b: u8) -> String {
    if (0x21..=0x7E).contains(&b) {
        format!("'{}'", b as char)
    } else {
        format!("0x{b:02X}")
    }
}

fn format_byte_preview(bytes: &[u8], limit: usize) -> String {
    let mut out = String::new();
    for (idx, &b) in bytes.iter().take(limit).enumerate() {
        if idx > 0 {
            out.push(' ');
        }
        out.push_str(&format_byte(b));
    }
    if bytes.len() > limit {
        out.push_str(" ...");
    }
    out
}

fn print_archived_mode(rank: usize, weight: f64, mode: &ArchivedMode, show_trace: bool) {
    let rep = &mode.report;
    println!("\n--- Bassin #{rank} (poids={:.1}%) ---", weight * 100.0);
    println!(
        "  phi_sharp : log={:.3} (val={:.4e}) | résidu ||R||={:.2}{} | mask={:.2} ({} tests) | ablation={:.2} | |H|={:.2}",
        rep.log_phi_sharp,
        rep.phi_sharp,
        rep.residual.norm,
        if rep.residual.is_zero() { " [EXACT]" } else { "" },
        rep.mask_score,
        rep.masks_tested,
        rep.ablation_score,
        rep.structural_complexity
    );
    println!(
        "  Exécution : {:?} en {} pas | E_hat={} nœuds, {} arêtes ({})",
        rep.outcome.termination,
        rep.outcome.steps_executed,
        rep.outcome.predicted.node_count(),
        rep.outcome.predicted.edge_count(),
        format_byte_preview(&rep.outcome.predicted.nodes, 32)
    );

    print_hypothesis_structure(&mode.graph, rep);

    if show_trace {
        println!("  Trajectoire ({} actions) :", mode.trajectory.len());
        for (t, step) in mode.trajectory.iter().enumerate() {
            println!("    [{t:>2}] {step:?}");
        }
    }
}

fn print_hypothesis_structure(h: &HypothesisGraph, rep: &phisharp::refutation::RefutationReport) {
    // 1. Affichage du germe SEED.
    let mut seed_nodes = Vec::new();
    for v in 1..h.node_count() {
        if h.membership(v) == Some(Block::Seed) {
            seed_nodes.push(v);
        }
    }
    let seed_labels: Vec<String> = seed_nodes
        .iter()
        .map(|&v| match h.nodes()[v] {
            NodeKind::Conc(b) => format!("n{v}:{}", format_byte(b)),
            _ => format!("n{v}:?"),
        })
        .collect();
    let seed_edges: Vec<String> = h
        .edges()
        .iter()
        .filter_map(|e| match e.kind {
            EdgeKind::Struct(r)
                if h.membership(e.src) == Some(Block::Seed)
                    && h.membership(e.dst) == Some(Block::Seed) =>
            {
                Some(format!("n{}-(r{r})->n{}", e.src, e.dst))
            }
            _ => None,
        })
        .collect();
    println!(
        "  SEED      : [{}]  arêtes: [{}]",
        seed_labels.join(", "),
        seed_edges.join(", ")
    );

    // 2. Affichage de chaque hub RULE avec ses classes de variables SAME_AS et son statut d'ablation.
    let status_map: BTreeMap<NodeId, (RuleStatus, f64)> = rep
        .rule_reports
        .iter()
        .map(|r| (r.rule_node, (r.status, r.delta_residual)))
        .collect();

    for (rule_id, &kind) in h.nodes().iter().enumerate() {
        if kind != NodeKind::Rule {
            continue;
        }
        let var_names = resolve_rule_var_names(h, rule_id);
        let lhs_str = format_rule_side(h, Block::RuleLhs(rule_id), &var_names);
        let rhs_str = format_rule_side(h, Block::RuleRhs(rule_id), &var_names);
        let status_txt = match status_map.get(&rule_id) {
            Some((RuleStatus::Necessary, dr)) => format!("Necessary, dR=+{dr:.2}"),
            Some((RuleStatus::Parasitic, dr)) => format!("Parasitic, dR={dr:+.2}"),
            Some((RuleStatus::MalformedOrSkipped, _)) => "MalformedOrSkipped".into(),
            None => "Inconnu".into(),
        };
        println!("  RULE n{rule_id}  : {lhs_str}  ==>  {rhs_str}   [{status_txt}]");
    }
}

fn resolve_rule_var_names(h: &HypothesisGraph, rule_id: NodeId) -> BTreeMap<NodeId, String> {
    let vars: Vec<NodeId> = (1..h.node_count())
        .filter(|&v| h.owner_rule(v) == Some(rule_id) && h.nodes()[v] == NodeKind::Var)
        .collect();
    let mut parent: BTreeMap<NodeId, NodeId> = vars.iter().map(|&v| (v, v)).collect();

    let find = |parent: &mut BTreeMap<NodeId, NodeId>, mut x: NodeId| -> NodeId {
        while parent[&x] != x {
            let p = parent[&x];
            x = p;
        }
        x
    };

    for &e in h.edges() {
        if e.kind == EdgeKind::SameAs
            && h.owner_rule(e.src) == Some(rule_id)
            && h.owner_rule(e.dst) == Some(rule_id)
        {
            let ra = find(&mut parent, e.src);
            let rb = find(&mut parent, e.dst);
            if ra < rb {
                parent.insert(rb, ra);
            } else if rb < ra {
                parent.insert(ra, rb);
            }
        }
    }

    let mut root_to_idx: BTreeMap<NodeId, usize> = BTreeMap::new();
    let mut out = BTreeMap::new();
    for v in vars {
        let r = find(&mut parent, v);
        let next = root_to_idx.len();
        let idx = *root_to_idx.entry(r).or_insert(next);
        out.insert(v, format!("X{idx}"));
    }
    out
}

fn format_rule_side(
    h: &HypothesisGraph,
    block: Block,
    var_names: &BTreeMap<NodeId, String>,
) -> String {
    let nodes: Vec<String> = (1..h.node_count())
        .filter(|&v| h.membership(v) == Some(block))
        .map(|v| match h.nodes()[v] {
            NodeKind::Conc(b) => format!("n{v}:{}", format_byte(b)),
            NodeKind::Var => format!("n{v}:{}", var_names.get(&v).map_or("X?", String::as_str)),
            _ => format!("n{v}:?"),
        })
        .collect();
    let edges: Vec<String> = h
        .edges()
        .iter()
        .filter_map(|e| match e.kind {
            EdgeKind::Struct(r)
                if h.membership(e.src) == Some(block) && h.membership(e.dst) == Some(block) =>
            {
                Some(format!("n{}-(r{r})->n{}", e.src, e.dst))
            }
            _ => None,
        })
        .collect();
    if edges.is_empty() {
        format!("[{}]", nodes.join(", "))
    } else {
        format!("[{} | {}]", nodes.join(", "), edges.join(", "))
    }
}