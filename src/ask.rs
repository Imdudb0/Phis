use crate::graph::*;
use crate::dynamics::{self, SearchResult};
use crate::{model, phi};
use std::fs;

pub fn run(model_path: &str, file_path: &str) -> Result<(), String> {
    let mut g = model::load(model_path)?;
    let e = fs::read(file_path).map_err(|x| format!("lecture {file_path}: {x}"))?;
    let mut rng = Rng::new(0x5EED);
    let mut budget = Budget { ops: 0, max: 500_000_000 };
    let ef = g.meta.eta_fast; let eg = g.meta.explore_gain;
    // zéro-shot: aucun apprentissage lent; seule la plasticité rapide intrinsèque joue.
    let res = dynamics::search(&mut g, &e, 64, &mut budget, &mut rng, ef, eg, None);
    let resp = phi::responsibility(&g, &res.best_ws, &res.best_c, &e, &res.best_probs, &mut rng, &mut budget);

    println!("{}", explain(&e, &res));

    eprintln!("phi_sharp={:.4} fidélité={:.3} cycles={} ops={} résidu_moyen={:.3}",
        res.phi, res.fidelity, res.cycles, res.ops, res.residual_mean);
    let zero = res.links.iter().filter(|l| !l.shared).count();
    eprintln!("liens_méta_relationnels={} dont_sans_mots_communs={}", res.links.len(), zero);
    let l = res.best_probs.len();
    let mut positions: Vec<usize> = (0..e.len()).collect();
    positions.sort_by(|&a, &b| {
        let ra = if a < l { 1.0 - res.best_probs[a][e[a] as usize] } else { 1.0 };
        let rb = if b < l { 1.0 - res.best_probs[b][e[b] as usize] } else { 1.0 };
        rb.partial_cmp(&ra).unwrap()
    });
    for &p in positions.iter().take(8) {
        let got = res.best_e.get(p).copied().unwrap_or(0);
        let mut top_k = 0; let mut top_v = 0.0f32;
        for k in 0..N_C { if p < l && resp[k][p] > top_v { top_v = resp[k][p]; top_k = k; } }
        let nature = if p >= l { "hors_budget" }
            else if top_v < 0.05 { "structure_manquante_dans_la_perception" }
            else { "liaison_causale_à_ajuster" };
        eprintln!("résidu@{p}: attendu=0x{:02X} régénéré=0x{:02X} cause_responsable=c{top_k} ({nature})", e[p], got);
    }
    Ok(())
}

fn span_text(e: &[u8], s: usize, t: usize) -> String {
    String::from_utf8_lossy(&e[s..t]).into_owned()
}

// Explication maximale: pas d'élagage MDL. Tout ce que la recherche a trouvé.
fn explain(e: &[u8], res: &SearchResult) -> String {
    let mut out = String::new();
    out.push_str(&String::from_utf8_lossy(&res.best_e));
    out.push('\n');
    let mut order: Vec<usize> = (0..res.assemblies.len()).collect();
    let score = |i: usize| -> f32 { res.assemblies[i].sig.iter().take(N_C).sum() };
    order.sort_by(|&a, &b| score(b).partial_cmp(&score(a)).unwrap());
    for i in order {
        let (s, t) = res.assemblies[i].span;
        out.push_str(&span_text(e, s, t));
        out.push('\n');
    }
    for lk in &res.links {
        let (sa, ta) = res.assemblies[lk.a].span;
        let (sb, tb) = res.assemblies[lk.b].span;
        out.push_str(&format!("« {} » ↔ « {} »{}\n",
            span_text(e, sa, ta), span_text(e, sb, tb),
            if lk.shared { "" } else { " [∅]" }));
    }
    let l = res.best_probs.len();
    for p in 0..e.len() {
        if p >= l || res.best_e[p] != e[p] {
            let got = res.best_e.get(p).copied().unwrap_or(0);
            out.push_str(&format!("@{p}:0x{:02X}→0x{:02X}\n", e[p], got));
        }
    }
    out
}