use crate::graph::*;
use crate::dynamics::{self, Elig};
use crate::{model, ptgrs};

pub fn run(out: &str, samples: usize, seed: u64) -> Result<(), String> {
    let mut rng = Rng::new(seed);
    let mut g = Graph::build(&mut rng);
    let mut elig = Elig(vec![[0.0; D * D]; g.edges.len()]);
    let mut budget = Budget { ops: 0, max: 2_000_000_000 };
    let mut phi_ema = 0.0f32; let eta = 0.3f32;
    let mut phi_sum = 0.0f32; let mut phi_prev_mean = 0.0f32;

    // UNE séance. Signal = phi_sharp relatif à E. Récompense non-évanescente.
    for s in 0..samples {
        let depth = 2 + (s * 10 / samples.max(1)).min(10);
        let (e, _cause) = ptgrs::generate(&mut rng, depth);
        for v in elig.0.iter_mut() { *v = [0.0; D * D]; }
        let eg = g.meta.explore_gain;
        let res = dynamics::search(&g, &e, 32, &mut budget, &mut rng, eg, Some(&mut elig));
        let reward = ((res.phi - phi_ema) / (phi_ema + 0.05)).clamp(-1.0, 1.0);
        for (ei, ed) in g.edges.iter_mut().enumerate() {
            if ed.kind.is_fast() { continue; }
            let mut nrm = 0.0;
            for k in 0..D * D { nrm += elig.0[ei][k] * elig.0[ei][k]; }
            let nrm = nrm.sqrt() + 1e-6;
            for k in 0..D * D { ed.w[k] = (ed.w[k] + eta * reward * elig.0[ei][k] / nrm).clamp(-2.0, 2.0); }
        }
        phi_ema = 0.9 * phi_ema + 0.1 * res.phi;
        phi_sum += res.phi;
        let mean = phi_sum / (s + 1) as f32;
        if mean > phi_prev_mean { g.meta.explore_gain = (g.meta.explore_gain * 0.99).max(0.3); }
        else { g.meta.explore_gain = (g.meta.explore_gain * 1.01).min(3.0); }
        phi_prev_mean = mean;
        g.meta.phi_ema = phi_ema;
        eprintln!("séance {}/{} prof={} phi={:.4} fid={:.3} résidu={:.2} exploration={:.3}",
            s + 1, samples, depth, res.phi, res.fidelity, res.residual, g.meta.explore_gain);
        if budget.ops >= budget.max { eprintln!("budget mesuré atteint: {} ops", budget.ops); break; }
    }
    model::save(&g, out)?;
    eprintln!("modèle écrit: {out}");
    eval_post(&mut g, seed.wrapping_add(0x9E3779B9), 8);
    Ok(())
}

fn eval_post(g: &mut Graph, seed: u64, n: usize) {
    let mut rng = Rng::new(seed);
    let mut budget = Budget { ops: 0, max: 1_000_000_000 };
    let mut fid_sum = 0.0; let mut proj = 0usize;
    for i in 0..n {
        let depth = 4 + (i % 5);
        let (e, _cause) = ptgrs::generate(&mut rng, depth);
        let eg = g.meta.explore_gain;
        let res = dynamics::search(g, &e, 32, &mut budget, &mut rng, eg, None);
        let (links, _ans) = crate::abstraction::project(&res, &e);
        fid_sum += res.fidelity;
        let sans_mots = links.iter().any(|l| !l.shared);
        if sans_mots { proj += 1; }
        eprintln!("éval {}/{} fid={:.3} phi={:.4} projection_sans_mots_communs={}",
            i + 1, n, res.fidelity, res.phi, sans_mots);
    }
    eprintln!("MOYENNES fid={:.3} projection={}/{} (cause cachée: évaluation après coup seulement)",
        fid_sum / n as f32, proj, n);
}