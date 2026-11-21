use crate::graph::*;
use crate::dynamics::{self, Elig};
use crate::{model, ptgrs};

pub fn run(out: &str, samples: usize, seed: u64) -> Result<(), String> {
    let mut rng = Rng::new(seed);
    let mut g = Graph::build(&mut rng);
    let mut elig = Elig(vec![[0.0; D * D]; g.edges.len()]);
    let mut budget = Budget { ops: 0, max: 4_000_000_000 };
    let mut phi_ema = 0.0f32; let eta = 0.05f32;
    let mut phi_sum = 0.0f32; let mut phi_prev_mean = 0.0f32;

    // UNE séance: une seule passe. Le signal est phi_sharp relatif à E.
    for s in 0..samples {
        let depth = 2 + (s * 10 / samples.max(1)).min(10);
        let (e, _cause) = ptgrs::generate(&mut rng, depth);
        for v in elig.0.iter_mut() { *v = [0.0; D * D]; }
        let ef = g.meta.eta_fast; let eg = g.meta.explore_gain;
        let res = dynamics::search(&mut g, &e, 24, &mut budget, &mut rng, ef, eg, Some(&mut elig));
        let phi = res.phi;
        let adv = phi - phi_ema;
        for (ei, ed) in g.edges.iter_mut().enumerate() {
            if ed.kind.is_fast() { continue; }
            let mut nrm = 0.0;
            for k in 0..D * D { nrm += elig.0[ei][k] * elig.0[ei][k]; }
            let nrm = nrm.sqrt() + 1e-6;
            for k in 0..D * D { ed.w[k] = (ed.w[k] + eta * adv * elig.0[ei][k] / nrm).clamp(-2.0, 2.0); }
        }
        phi_ema = 0.9 * phi_ema + 0.1 * phi;
        phi_sum += phi;
        let mean = phi_sum / (s + 1) as f32;
        // apprend à chercher: disposition d'exploration, pas une procédure.
        if mean > phi_prev_mean { g.meta.explore_gain = (g.meta.explore_gain * 0.99).max(0.3); }
        else { g.meta.explore_gain = (g.meta.explore_gain * 1.01).min(3.0); }
        phi_prev_mean = mean;
        g.meta.phi_ema = phi_ema;
        eprintln!("séance {}/{} profondeur={} phi_sharp={:.4} fidélité={:.3} exploration={:.3}",
            s + 1, samples, depth, phi, res.fidelity, g.meta.explore_gain);
        if budget.ops >= budget.max { eprintln!("budget mesuré atteint: {} ops", budget.ops); break; }
    }
    model::save(&g, out)?;
    eprintln!("modèle écrit: {out}");
    eval_post(&mut g, seed.wrapping_add(0x9E3779B9), 8);
    Ok(())
}

// La cause cachée ne sert QU'ICI, après coup.
fn eval_post(g: &mut Graph, seed: u64, n: usize) {
    let mut rng = Rng::new(seed);
    let mut budget = Budget { ops: 0, max: 1_000_000_000 };
    let mut fid_sum = 0.0; let mut proj = 0usize; let mut cov_sum = 0.0;
    for i in 0..n {
        let depth = 4 + (i % 5);
        let (e, cause) = ptgrs::generate(&mut rng, depth);
        let ef = g.meta.eta_fast; let eg = g.meta.explore_gain;
        let res = dynamics::search(g, &e, 32, &mut budget, &mut rng, ef, eg, None);
        fid_sum += res.fidelity;
        let sans_mots = res.links.iter().any(|l| !l.shared);
        if sans_mots { proj += 1; }
        let mut cov = 0usize;
        for &(s, t) in &cause.spans {
            if t <= s { continue; }
            let ok = res.assemblies.iter().any(|a| {
                let (u, v) = a.span;
                let istart = s.max(u); let iend = t.min(v);
                let il = if iend > istart { iend - istart } else { 0 };
                let un = (t - s) + (v - u) - il;
                un > 0 && il as f32 / un as f32 > 0.3
            });
            if ok { cov += 1; }
        }
        let cv = if cause.spans.is_empty() { 0.0 } else { cov as f32 / cause.spans.len() as f32 };
        cov_sum += cv;
        eprintln!("éval {}/{} fid={:.3} projection_sans_mots_communs={} couverture_cause={:.2}",
            i + 1, n, res.fidelity, sans_mots, cv);
    }
    eprintln!("MOYENNES fid={:.3} projection={}/{} couverture={:.2} (cause cachée: évaluation après coup seulement)",
        fid_sum / n as f32, proj, n, cov_sum / n as f32);
}