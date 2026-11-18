use phisharp::autodiff::{ParamStore, Tape};
use phisharp::model::controller::{
    alphabet_mask_from_bytes, forward_causal_intervention, readout_infer, run_abductive_search,
};
use phisharp::model::graph::{GraphParams, ModelConfig, non_reciprocity_ratio};
use phisharp::model::phis::load_phis;
use phisharp::rng::Rng;
use phisharp::synth::codec::BlockKind;
use phisharp::synth::episode::{EpisodeConfig, sample_episode};
use phisharp::train::losses::{LossBreakdown, LossWeights, compute_episode_losses};
use phisharp::train::trainer::{TrainConfig, run_training};
use std::env;
use std::io::{self, Write};
use std::process;

fn print_help_and_exit(code: i32) -> ! {
    let _ = writeln!(
        io::stderr(),
        "Phisharp CLI\n\
         Commandes :\n\
           phisharp train [--steps N] [--batch N] [--lr F] [--nodes K] [--dim D] [--max-search N] [--out out/sharp.phis] [--seed S]\n\
           phisharp eval  [--model out/sharp.phis] [--episodes N] [--seed S]\n\
           phisharp ask   --model <out/sharp.phis> --file <perceived.eff>"
    );
    process::exit(code);
}

fn parse_usize(args: &[String], idx: &mut usize) -> usize {
    *idx += 1;
    if *idx >= args.len() {
        print_help_and_exit(2);
    }
    args[*idx].parse::<usize>().unwrap_or_else(|_| print_help_and_exit(2))
}

fn parse_u64(args: &[String], idx: &mut usize) -> u64 {
    *idx += 1;
    if *idx >= args.len() {
        print_help_and_exit(2);
    }
    args[*idx].parse::<u64>().unwrap_or_else(|_| print_help_and_exit(2))
}

fn parse_f32(args: &[String], idx: &mut usize) -> f32 {
    *idx += 1;
    if *idx >= args.len() {
        print_help_and_exit(2);
    }
    args[*idx].parse::<f32>().unwrap_or_else(|_| print_help_and_exit(2))
}

fn parse_string(args: &[String], idx: &mut usize) -> String {
    *idx += 1;
    if *idx >= args.len() {
        print_help_and_exit(2);
    }
    args[*idx].clone()
}

fn cmd_train(args: &[String]) {
    let mut cfg = TrainConfig::default();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => cfg.total_steps = parse_usize(args, &mut i),
            "--batch" => cfg.batch_accum = parse_usize(args, &mut i),
            "--lr" => cfg.lr = parse_f32(args, &mut i),
            "--nodes" => cfg.model_cfg.num_nodes = parse_usize(args, &mut i),
            "--dim" => cfg.model_cfg.dim = parse_usize(args, &mut i),
            "--max-search" => cfg.model_cfg.max_steps = parse_usize(args, &mut i),
            "--out" => cfg.save_path = parse_string(args, &mut i),
            "--seed" => cfg.seed = parse_u64(args, &mut i),
            _ => print_help_and_exit(2),
        }
        i += 1;
    }

    if let Err(err) = run_training(&cfg, true) {
        let _ = writeln!(io::stderr(), "Échec entraînement: {}", err);
        process::exit(1);
    }
}

fn evaluate_model_on_episodes(
    store: &ParamStore,
    params: &GraphParams,
    episodes: usize,
    seed: u64,
) -> (LossBreakdown, f32, f32, f32) {
    let mut rng = Rng::from_seed(seed);
    // Distribution hors-entraînement : plus de symboles, plus de règles, dérivations plus profondes
    let ep_cfg = EpisodeConfig {
        num_symbols: 14,
        num_rules: 5,
        max_rewrite_steps: 7,
        max_graph_nodes: 56,
        max_target_bytes: 48,
    };
    let weights = LossWeights::default();

    let mut agg = LossBreakdown::default();
    let mut non_recip_sum = 0.0f32;
    let mut alpha_ok = 0usize;
    let mut alpha_total = 0usize;
    let mut dir_asym_sum = 0.0f32;

    for _ in 0..episodes {
        let ep = sample_episode(&mut rng, &ep_cfg);
        let mut tape = Tape::new();
        let (_loss, br) = compute_episode_losses(&mut tape, store, params, &ep, &weights, 1.0);
        agg.total += br.total;
        agg.iso += br.iso;
        agg.do_causal += br.do_causal;
        agg.max_rel += br.max_rel;
        agg.closure += br.closure;
        agg.teleology += br.teleology;
        agg.halt += br.halt;
        agg.residual += br.residual;

        // Contrôle non-réciprocité et asymétrie directionnelle sous intervention
        let mut eval_tape = Tape::new();
        let trace = run_abductive_search(
            &mut eval_tape,
            store,
            params,
            &ep.bytes_v1,
            None,
            true,
        );
        non_recip_sum += non_reciprocity_ratio(&eval_tape, params, trace.w_eff_final);

        // Vérifier fermeture stricte de l'alphabet sur l'inférence autonome
        let out = readout_infer(store, params, &ep.bytes_v1);
        let mask = alphabet_mask_from_bytes(&ep.bytes_v1);
        for &b in &out {
            alpha_total += 1;
            if mask[b as usize] > 0.5 {
                alpha_ok += 1;
            }
        }

        // Mesurer la différence entre propagation causale directe et propagation inversée
        let seed_blk = ep
            .spans_v1
            .iter()
            .find(|sp| sp.kind == BlockKind::RuleSpec && ep.do_affected_blocks[sp.block_id] > 0.5)
            .map(|sp| sp.block_id)
            .unwrap_or(0);
        let mut seed_node = vec![0.0f32; params.cfg.num_nodes];
        let sp = ep.spans_v1.iter().find(|s| s.block_id == seed_blk);
        if let Some(sp) = sp {
            let attn_d = eval_tape.data(trace.attn_final);
            let len = ep.bytes_v1.len();
            let span_len = (sp.end - sp.start).max(1) as f32;
            for k_idx in 0..params.cfg.num_nodes {
                let mut sum_a = 0.0f32;
                for p in sp.start..sp.end {
                    sum_a += attn_d[k_idx * len + p];
                }
                seed_node[k_idx] = sum_a / span_len;
            }
        }
        let seed_v = eval_tape.tensor(seed_node, vec![params.cfg.num_nodes, 1]);
        let fwd_delta = forward_causal_intervention(&mut eval_tape, params, &trace, seed_v, 2);

        // Construire trace avec arêtes transposées pour vérifier que le sens causal n'est pas symétrique
        let mut rev_trace = trace.clone();
        rev_trace.w_eff_final = eval_tape.transpose_last2(trace.w_eff_final);
        let rev_delta = forward_causal_intervention(&mut eval_tape, params, &rev_trace, seed_v, 2);

        let fd = eval_tape.data(fwd_delta);
        let rd = eval_tape.data(rev_delta);
        let diff: f32 = fd
            .iter()
            .zip(rd.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f32>()
            / (params.cfg.num_nodes as f32);
        dir_asym_sum += diff;
    }

    let inv_n = 1.0 / (episodes.max(1) as f32);
    agg.total *= inv_n;
    agg.iso *= inv_n;
    agg.do_causal *= inv_n;
    agg.max_rel *= inv_n;
    agg.closure *= inv_n;
    agg.teleology *= inv_n;
    agg.halt *= inv_n;
    agg.residual *= inv_n;

    let alpha_acc = if alpha_total == 0 {
        1.0
    } else {
        (alpha_ok as f32) / (alpha_total as f32)
    };
    (agg, non_recip_sum * inv_n, alpha_acc, dir_asym_sum * inv_n)
}

fn cmd_eval(args: &[String]) {
    let mut model_path = "out/sharp.phis".to_string();
    let mut episodes = 40usize;
    let mut seed = 999_999u64;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => model_path = parse_string(args, &mut i),
            "--episodes" => episodes = parse_usize(args, &mut i),
            "--seed" => seed = parse_u64(args, &mut i),
            _ => print_help_and_exit(2),
        }
        i += 1;
    }

    let (cfg, params_trained, store_trained) = load_phis(&model_path).unwrap_or_else(|e| {
        let _ = writeln!(io::stderr(), "Erreur chargement '{}': {}", model_path, e);
        process::exit(1);
    });

    // Référence aléatoire non entraînée de même architecture pour comparaison honnête
    let mut base_rng = Rng::from_seed(seed ^ 0x1234_5678);
    let mut store_base = ParamStore::new();
    let params_base = GraphParams::init(cfg, &mut store_base, &mut base_rng);

    let (br_base, nr_base, _, _) = evaluate_model_on_episodes(&store_base, &params_base, episodes, seed);
    let (br_tr, nr_tr, alpha_tr, asym_tr) =
        evaluate_model_on_episodes(&store_trained, &params_trained, episodes, seed);

    println!("=== Évaluation Hors-Distribution (Codecs jamais vus, 14 symboles, 5 règles) ===");
    println!(
        "Référence aléatoire : total={:.4} (iso={:.4} do={:.4} rel={:.4} clos={:.4} tel={:.4}) | non-récip={:.3}",
        br_base.total, br_base.iso, br_base.do_causal, br_base.max_rel, br_base.closure, br_base.teleology, nr_base
    );
    println!(
        "Modèle entraîné     : total={:.4} (iso={:.4} do={:.4} rel={:.4} clos={:.4} tel={:.4}) | non-récip={:.3}",
        br_tr.total, br_tr.iso, br_tr.do_causal, br_tr.max_rel, br_tr.closure, br_tr.teleology, nr_tr
    );
    println!(
        "Respect strict alphabet d'entrée : {:.2}% | Asymétrie directionnelle do() : {:.4}",
        alpha_tr * 100.0,
        asym_tr
    );
}

fn cmd_ask(args: &[String]) {
    let mut model_path: Option<String> = None;
    let mut file_path: Option<String> = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => model_path = Some(parse_string(args, &mut i)),
            "--file" => file_path = Some(parse_string(args, &mut i)),
            _ => print_help_and_exit(2),
        }
        i += 1;
    }
    let model_path = model_path.unwrap_or_else(|| print_help_and_exit(2));
    let file_path = file_path.unwrap_or_else(|| print_help_and_exit(2));

    let (_cfg, params, store) = load_phis(&model_path).unwrap_or_else(|e| {
        let _ = writeln!(io::stderr(), "Erreur chargement '{}': {}", model_path, e);
        process::exit(1);
    });
    let bytes = std::fs::read(&file_path).unwrap_or_else(|e| {
        let _ = writeln!(io::stderr(), "Erreur lecture '{}': {}", file_path, e);
        process::exit(1);
    });
    let out = readout_infer(&store, &params, &bytes);
    let _ = io::stdout().write_all(&out);
    let _ = io::stdout().flush();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_help_and_exit(2);
    }
    match args[1].as_str() {
        "train" => cmd_train(&args[2..]),
        "eval" => cmd_eval(&args[2..]),
        "ask" => cmd_ask(&args[2..]),
        _ => print_help_and_exit(2),
    }
}