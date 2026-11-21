use crate::graph::*;
use crate::dynamics;
use crate::{abstraction, model, phi};
use std::fs;

pub fn run(model_path: &str, file_path: &str) -> Result<(), String> {
    let g = model::load(model_path)?;
    let e = fs::read(file_path).map_err(|x| format!("lecture {file_path}: {x}"))?;
    let mut rng = Rng::new(0x5EED);
    let mut budget = Budget { ops: 0, max: 500_000_000 };
    let res = dynamics::search(&g, &e, 64, &mut budget, &mut rng, g.meta.explore_gain, None);
    let (links, ans) = abstraction::project(&res, &e);
    let resp = phi::responsibility(&res.b, &res.drive, res.probe, res.assemblies.len());

    // Maximale en sens, concise en volume: la réponse projetée + la chaîne causale + le résidu.
    let mut out = String::new();
    out.push_str(&ans.phrase);
    out.push('\n');
    let probe = abstraction::span_text(&e, &res.assemblies[res.probe]);
    out.push_str(&probe);
    if let Some(h) = ans.top_head {
        out.push_str(&format!(" → {}", abstraction::span_text(&e, &res.assemblies[h])));
    }
    if let Some(p) = ans.projected {
        out.push_str(&format!(" → {}{}", abstraction::span_text(&e, &res.assemblies[p]),
            if ans.no_shared { " [∅]" } else { "" }));
    }
    out.push('\n');
    out.push_str(&format!("résidu: {:.2} de E non lié par la cause.\n", res.residual));
    print!("{out}");

    eprintln!("phi_sharp={:.4} fid={:.3} intégration={:.3} spécificité={:.3} étapes={} ops={}",
        res.phi, res.fidelity, res.integration, res.specificity, res.steps, res.ops);
    eprintln!("liens={} dont_sans_mots_communs={}", links.len(), links.iter().filter(|l| !l.shared).count());
    let mut idx: Vec<usize> = (0..res.assemblies.len()).collect();
    idx.sort_by(|&a, &c| resp[c].partial_cmp(&resp[a]).unwrap());
    for i in idx.iter().take(3) {
        eprintln!("résidu causal: « {} » responsabilité={:.3}",
            abstraction::span_text(&e, &res.assemblies[*i]), resp[*i]);
    }
    Ok(())
}