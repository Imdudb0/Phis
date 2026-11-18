use crate::autodiff::{AdamW, ParamStore, Tape};
use crate::model::graph::{GraphParams, ModelConfig, non_reciprocity_ratio};
use crate::model::phis::save_phis;
use crate::rng::Rng;
use crate::synth::episode::{Episode, EpisodeConfig, sample_episode};
use crate::train::losses::{LossBreakdown, LossWeights, compute_episode_losses};

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub seed: u64,
    pub total_steps: usize,
    pub batch_accum: usize,
    pub lr: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub log_every: usize,
    pub save_path: String,
    pub model_cfg: ModelConfig,
    pub episode_cfg: EpisodeConfig,
    pub loss_weights: LossWeights,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            seed: 20260401,
            total_steps: 400,
            batch_accum: 4,
            lr: 2.5e-3,
            weight_decay: 1e-4,
            max_grad_norm: 1.0,
            log_every: 20,
            save_path: "out/sharp.phis".to_string(),
            model_cfg: ModelConfig::default(),
            episode_cfg: EpisodeConfig::default(),
            loss_weights: LossWeights::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainHistory {
    pub step_losses: Vec<LossBreakdown>,
    pub final_non_reciprocity: f32,
}

fn add_breakdown(dst: &mut LossBreakdown, br: &LossBreakdown, scale: f32) {
    dst.total += br.total * scale;
    dst.iso += br.iso * scale;
    dst.do_causal += br.do_causal * scale;
    dst.max_rel += br.max_rel * scale;
    dst.closure += br.closure * scale;
    dst.teleology += br.teleology * scale;
    dst.halt += br.halt * scale;
    dst.residual += br.residual * scale;
    dst.div += br.div * scale;
}

pub fn run_training(cfg: &TrainConfig, verbose: bool) -> Result<TrainHistory, String> {
    let mut rng = Rng::from_seed(cfg.seed);
    let mut store = ParamStore::new();
    let params = GraphParams::init(cfg.model_cfg.clone(), &mut store, &mut rng);
    let mut opt = AdamW::new(&store, cfg.lr, cfg.weight_decay, cfg.max_grad_norm);

    let accum = cfg.batch_accum.max(1);
    let inv_accum = 1.0 / (accum as f32);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(accum);
    let mut history = Vec::with_capacity(cfg.total_steps);

    if verbose {
        println!(
            "[Phisharp] Entraînement : {} paramètres lents | K={} d={} | {} pas (batch_accum={}, threads={})",
            store.total_param_count(),
            cfg.model_cfg.num_nodes,
            cfg.model_cfg.dim,
            cfg.total_steps,
            accum,
            threads
        );
    }

    for step in 0..cfg.total_steps {
        store.zero_grad();
        let mut agg = LossBreakdown::default();

        let episodes: Vec<Episode> = (0..accum)
            .map(|_| sample_episode(&mut rng, &cfg.episode_cfg))
            .collect();

        if threads <= 1 {
            for ep in &episodes {
                let mut tape = Tape::new();
                let (loss, br) = compute_episode_losses(
                    &mut tape,
                    &store,
                    &params,
                    ep,
                    &cfg.loss_weights,
                    inv_accum,
                );
                tape.backward(loss, &mut store);
                add_breakdown(&mut agg, &br, inv_accum);
            }
        } else {
            let chunk_size = accum.div_ceil(threads);
            let store_ref = &store;
            let params_ref = &params;
            let weights_ref = &cfg.loss_weights;

            let results: Vec<(Vec<Vec<f32>>, LossBreakdown)> = std::thread::scope(|s| {
                let mut handles = Vec::new();
                for chunk in episodes.chunks(chunk_size) {
                    handles.push(s.spawn(move || {
                        let mut local_store = ParamStore {
                            names: Vec::new(),
                            shapes: store_ref.shapes.clone(),
                            values: store_ref.values.clone(),
                            grads: store_ref.values.iter().map(|v| vec![0.0; v.len()]).collect(),
                        };
                        let mut local_br = LossBreakdown::default();
                        for ep in chunk {
                            let mut tape = Tape::new();
                            let (loss, br) = compute_episode_losses(
                                &mut tape,
                                &local_store,
                                params_ref,
                                ep,
                                weights_ref,
                                inv_accum,
                            );
                            tape.backward(loss, &mut local_store);
                            add_breakdown(&mut local_br, &br, inv_accum);
                        }
                        (local_store.grads, local_br)
                    }));
                }
                handles.into_iter().map(|h| h.join().expect("worker panic")).collect()
            });

            for (local_grads, local_br) in results {
                for (dst, src) in store.grads.iter_mut().zip(local_grads.iter()) {
                    for (d, &s) in dst.iter_mut().zip(src.iter()) {
                        *d += s;
                    }
                }
                add_breakdown(&mut agg, &local_br, 1.0);
            }
        }

        let g_norm = opt.step(&mut store);
        history.push(agg.clone());

        if verbose && (step == 0 || (step + 1) % cfg.log_every == 0 || step + 1 == cfg.total_steps) {
            println!(
                "pas {:>4}/{} | total={:.4} | iso={:.4} do={:.4} rel={:.4} clos={:.4} tel={:.4} res={:.4} div={:.4} | gnorm={:.2}",
                step + 1,
                cfg.total_steps,
                agg.total,
                agg.iso,
                agg.do_causal,
                agg.max_rel,
                agg.closure,
                agg.teleology,
                agg.residual,
                agg.div,
                g_norm
            );
        }
    }

    let probe_ep = sample_episode(&mut rng, &cfg.episode_cfg);
    let mut probe_tape = Tape::new();
    let probe_trace = crate::model::controller::run_abductive_search(
        &mut probe_tape,
        &store,
        &params,
        &probe_ep.bytes_v1,
        None,
        true,
    );
    let final_non_reciprocity = non_reciprocity_ratio(&probe_tape, &params, probe_trace.w_eff_final);

    save_phis(&cfg.save_path, &cfg.model_cfg, &store)
        .map_err(|e| format!("Erreur sauvegarde {}: {}", cfg.save_path, e))?;

    if verbose {
        println!(
            "[Phisharp] Modèle sauvegardé dans {} | non-réciprocité={:.3}",
            cfg.save_path, final_non_reciprocity
        );
    }

    Ok(TrainHistory {
        step_losses: history,
        final_non_reciprocity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::phis::load_phis;

    #[test]
    fn trainer_smoke_run_converges_and_saves_phis() {
        let tmp_phis = std::env::temp_dir().join("phisharp_smoke_train.phis");
        let cfg = TrainConfig {
            seed: 777,
            total_steps: 10,
            batch_accum: 2,
            lr: 4e-3,
            weight_decay: 1e-4,
            max_grad_norm: 1.0,
            log_every: 5,
            save_path: tmp_phis.to_string_lossy().into_owned(),
            model_cfg: ModelConfig {
                num_nodes: 6,
                dim: 8,
                pos_dim: 6,
                min_steps: 2,
                max_steps: 3,
                halt_threshold: 0.8,
                max_readout_bytes: 32,
            },
            episode_cfg: EpisodeConfig {
                num_symbols: 8,
                num_rules: 3,
                max_rewrite_steps: 4,
                max_graph_nodes: 24,
                max_target_bytes: 24,
            },
            loss_weights: LossWeights::default(),
        };

        let hist = run_training(&cfg, false).expect("training failed");
        assert_eq!(hist.step_losses.len(), 10);
        assert!(hist.final_non_reciprocity > 0.4);
        let first = hist.step_losses[0].total;
        let last = hist.step_losses.last().unwrap().total;
        assert!(first.is_finite() && last.is_finite());

        let (loaded_cfg, _, _) = load_phis(&tmp_phis).expect("load_phis failed");
        assert_eq!(loaded_cfg, cfg.model_cfg);
        let _ = std::fs::remove_file(&tmp_phis);
    }
}