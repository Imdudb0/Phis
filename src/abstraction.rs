use crate::graph::*;
use crate::dynamics::{Assembly, Link};

// Signature = rôle causal + flux dirigés + influence inter-assemblées. Pas le contenu.
pub fn signatures(g: &Graph, asms: &[Assembly], rng: &mut Rng, budget: &mut Budget) -> Vec<Vec<f32>> {
    let gain = vec![1.0f32; N]; let zero = zero_state();
    let mut out = Vec::new();
    for a in asms {
        let mut x = zero_state();
        for i in 0..N_WS { x[WS0 + i] = a.ws[i]; }
        for _ in 0..4 { x = step_full(g, &x, &zero, &gain, 0.0, rng, budget, None); }
        let mut sig = Vec::new();
        for k in 0..N_C { sig.push(norm(&x[C0 + k])); }
        for (s, t) in [(0usize, N_BYTE), (WS0, C0), (C0, G0), (G0, A0), (A0, N)] {
            let m: f32 = (s..t).map(|i| norm(&x[i])).sum::<f32>() / (t - s) as f32;
            sig.push(m);
        }
        for b in asms {
            let mut d = 0.0; let mut na = 0.0; let mut nb = 0.0;
            for i in 0..N_WS {
                let u = x[WS0 + i][0]; let v = b.ws[i][0];
                d += u * v; na += u * u; nb += v * v;
            }
            sig.push(d / (na.sqrt() * nb.sqrt() + 1e-6));
        }
        out.push(sig);
    }
    out
}

fn words(e: &[u8], span: (usize, usize)) -> Vec<Vec<u8>> {
    let mut out = Vec::new(); let mut cur = Vec::new();
    for &b in &e[span.0..span.1] {
        if b.is_ascii_alphanumeric() { cur.push(b); }
        else if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

// Critère majeur: lien entre assemblées SANS mots communs.
pub fn project(g: &Graph, asms: &mut [Assembly], e: &[u8], rng: &mut Rng, budget: &mut Budget) -> Vec<Link> {
    let sigs = signatures(g, asms, rng, budget);
    for (a, s) in asms.iter_mut().zip(sigs.into_iter()) { a.sig = s; }
    let mut links = Vec::new();
    for i in 0..asms.len() {
        for j in i + 1..asms.len() {
            let score = cosine(&asms[i].sig, &asms[j].sig);
            if score > 0.7 {
                let wa = words(e, asms[i].span); let wb = words(e, asms[j].span);
                let shared = wa.iter().any(|w| wb.contains(w));
                links.push(Link { a: i, b: j, score, shared });
            }
        }
    }
    links.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    links
}