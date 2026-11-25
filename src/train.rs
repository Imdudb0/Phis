//! Entraînement FERMÉ : PTGRS et φ_sharp, rien d'autre.
//!
//! Sont absents de ce fichier, et doivent le rester : texte de langue,
//! définitions, E auto-décrits, rendu de H ou de la trajectoire en octets,
//! intentions dans la cause cachée, prior linguistique pré-entraîné, tête de
//! sortie entraînée au NL. La cause cachée est DÉTRUITE avant tout calcul.
//!
//! Aucun choix de checkpoint : on sauve l'état final d'une séance, à un nombre
//! d'épisodes pré-enregistré. La courbe enregistrée ne contient que des
//! grandeurs PTGRS (φ, couverture, budget) — jamais une métrique NL.

use std::io;

use crate::abduction::{self, SearchCfg};
use crate::ad::{Adam, Params};
use crate::eff::{self, ViewCfg};
use crate::emit::EmitCfg;
use crate::model::{Closure, Model, CLOSED_EXPERIMENT};
use crate::phi::PhiCfg;
use crate::ptgrs::{self, FamilySpec};
use crate::rng::{tag, Rng};
use crate::substrate;

#[derive(Clone, Copy, Debug)]
pub struct TrainCfg {
    pub seed: u64,
    pub episodes: usize,
    pub batch: usize,
    pub lr: f32,
    pub view: ViewCfg,
    /// budget de recherche PENDANT l'entraînement
    pub search: SearchCfg,
    /// budget de recherche À L'INFÉRENCE, figé dans l'artefact
    pub infer: SearchCfg,
    pub emit: EmitCfg,
    pub log_every: usize,
}

impl Default for TrainCfg {
    fn default() -> Self {
        TrainCfg {
            seed: 20_261_125,
            episodes: 240,
            batch: 4,
            lr: 3e-3,
            view: ViewCfg::default(),
            search: SearchCfg {
                max_ops: 10,
                max_rel: 60_000_000,
                phi: PhiCfg { mech: 4, purview: 3, mip: 4, perturb: true },
                ..SearchCfg::default()
            },
            infer: SearchCfg::default(),
            emit: EmitCfg::default(),
            log_every: 20,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CurveRow {
    pub ep: usize,
    pub phi: f32,
    pub cov: f32,
    pub ops: usize,
    pub rel: u64,
}

pub struct TrainReport {
    pub episodes: usize,
    pub steps: usize,
    pub nan_steps: usize,
    pub curve: Vec<CurveRow>,
    pub final_phi_mean: f32,
    pub fingerprint: u64,
    pub total_rel: u64,
}

fn finite(g: &[Vec<f32>]) -> bool {
    g.iter().all(|b| b.iter().all(|x| x.is_finite()))
}

pub fn train(cfg: &TrainCfg, out_path: &str) -> io::Result<(Model, TrainReport)> {
    let mut rng = Rng::new(cfg.seed);
    let mut p: Params = substrate::init(&mut rng.fork(tag("init")));
    let mut opt = Adam::new(&p, cfg.lr);
    let mut grads: Vec<Vec<f32>> = p.data.iter().map(|b| vec![0.0f32; b.len()]).collect();

    let spec = FamilySpec::train();
    let mut curve: Vec<CurveRow> = Vec::new();
    let mut steps = 0usize;
    let mut nan_steps = 0usize;
    let mut total_rel = 0u64;
    let mut pending = 0usize;
    let batch = cfg.batch.max(1);

    for ep in 0..cfg.episodes {
        let mut er = rng.fork(tag("ptgrs").wrapping_add(ep as u64));
        let draw = ptgrs::episode(&spec, &mut er);
        let e = draw.e;
        // La cause cachée ne sert qu'à l'évaluation après coup. Ici, détruite.
        drop(draw.cause);

        let v = eff::build(&e, &cfg.view);
        let w = substrate::wire(&v);
        let seed = rng.fork(tag("search")).next_u64();
        let out = abduction::search(&p, &e, &v, &w, &cfg.search, seed, Some(&mut grads));

        total_rel = total_rel.saturating_add(out.rel);
        curve.push(CurveRow { ep, phi: out.phi, cov: out.cov, ops: out.ops, rel: out.rel });
        pending += 1;

        if pending >= batch {
            if finite(&grads) {
                opt.step(&mut p, &grads);
                steps += 1;
            } else {
                nan_steps += 1;
            }
            for b in grads.iter_mut() {
                for x in b.iter_mut() {
                    *x = 0.0;
                }
            }
            pending = 0;
        }

        if cfg.log_every > 0 && (ep + 1) % cfg.log_every == 0 {
            let k = curve.len().min(cfg.log_every);
            let m: f32 = curve[curve.len() - k..].iter().map(|r| r.phi).sum::<f32>() / k as f32;
            eprintln!("ep {:>5}  phi_moy {:.6}  steps {}  rel {}", ep + 1, m, steps, total_rel);
        }
    }

    if pending > 0 {
        if finite(&grads) {
            opt.step(&mut p, &grads);
            steps += 1;
        } else {
            nan_steps += 1;
        }
    }

    let model = Model {
        params: p,
        view: cfg.view,
        search: cfg.infer,
        emit: cfg.emit,
        seed: cfg.seed,
        episodes: cfg.episodes as u64,
        steps: steps as u64,
        closure: Closure::closed(),
        experiment: CLOSED_EXPERIMENT.to_string(),
        fingerprint: 0,
    };
    let fp = model.save(out_path)?;

    let k = curve.len().min(20).max(1);
    let final_phi_mean = if curve.is_empty() {
        0.0
    } else {
        curve[curve.len() - k..].iter().map(|r| r.phi).sum::<f32>() / k as f32
    };

    Ok((
        model,
        TrainReport {
            episodes: cfg.episodes,
            steps,
            nan_steps,
            curve,
            final_phi_mean,
            fingerprint: fp,
            total_rel,
        },
    ))
}

/// Manifeste de version : ce qui a été vu, ce qui ne l'a pas été, le budget.
/// Écrit à côté de l'artefact. C'est la pièce de traçabilité, pas un résultat.
pub fn write_manifest(path: &str, cfg: &TrainCfg, rep: &TrainReport) -> io::Result<()> {
    let mut s = String::new();
    s.push_str(&format!("experiment\t{}\n", CLOSED_EXPERIMENT));
    s.push_str(&format!("fingerprint\t{:016x}\n", rep.fingerprint));
    s.push_str(&format!("seed\t{}\n", cfg.seed));
    s.push_str(&format!("episodes\t{}\n", rep.episodes));
    s.push_str(&format!("optimizer_steps\t{}\n", rep.steps));
    s.push_str(&format!("nan_steps\t{}\n", rep.nan_steps));
    s.push_str(&format!("total_rel\t{}\n", rep.total_rel));
    s.push_str(&format!("final_phi_mean\t{:.8}\n", rep.final_phi_mean));
    s.push_str(&format!("train_max_ops\t{}\n", cfg.search.max_ops));
    s.push_str(&format!("infer_max_ops\t{}\n", cfg.infer.max_ops));
    s.push_str(&format!("phi_mech\t{}\n", cfg.search.phi.mech));
    s.push_str(&format!("phi_mip\t{}\n", cfg.search.phi.mip));
    s.push_str(&format!("emit_phi_min\t{:.8}\n", cfg.emit.phi_min));
    s.push_str(&format!("emit_stab_min\t{:.8}\n", cfg.emit.stab_min));
    s.push_str(&format!("emit_delim\t{}\n", cfg.emit.delim));
    s.push_str("training_signal\tphi_sharp+coverage+residue\n");
    s.push_str("excluded\tnl_text,definitions,self_described_e,rendered_h,intentions,pretrained_prior,nl_head\n");
    s.push_str("checkpoint_selection\tnone\n");
    for r in rep.curve.iter() {
        s.push_str(&format!(
            "curve\t{}\t{:.8}\t{:.8}\t{}\t{}\n",
            r.ep, r.phi, r.cov, r.ops, r.rel
        ));
    }
    std::fs::write(path, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seance_minuscule_et_gel() {
        let mut p = std::env::temp_dir();
        p.push(format!("phis_train_{}.phis", std::process::id()));
        let path = p.to_string_lossy().to_string();
        let cfg = TrainCfg {
            seed: 7,
            episodes: 2,
            batch: 2,
            lr: 1e-3,
            search: SearchCfg {
                max_ops: 1,
                max_rel: 20_000_000,
                phi: PhiCfg { mech: 3, purview: 3, mip: 2, perturb: true },
                ..SearchCfg::default()
            },
            log_every: 0,
            ..TrainCfg::default()
        };
        let (m, rep) = train(&cfg, &path).unwrap();
        assert!(m.closure.is_closed());
        assert_eq!(rep.episodes, 2);
        assert_eq!(rep.curve.len(), 2);
        let l = Model::load(&path).unwrap();
        assert_eq!(l.fingerprint, rep.fingerprint);
        let _ = std::fs::remove_file(&path);
    }
}