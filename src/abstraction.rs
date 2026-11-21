use crate::graph::*;
use crate::dynamics::{Assembly, SearchResult};

#[derive(Clone)]
pub struct Link { pub a: usize, pub b: usize, pub score: f32, pub shared: bool }

pub struct Answer {
    pub top_head: Option<usize>, pub bound_def: Option<usize>, pub projected: Option<usize>,
    pub phrase: String, pub no_shared: bool,
}

fn role(b: &[Vec<f32>], i: usize) -> Vec<f32> {
    let n = b.len();
    let mut r = Vec::with_capacity(2 * n);
    for j in 0..n { r.push(b[i][j]); }
    for j in 0..n { r.push(b[j][i]); }
    r
}
fn overlap(a: &[Vec<u8>], b: &[Vec<u8>]) -> bool { a.iter().any(|w| b.contains(w)) }

fn core_phrase(e: &[u8], span: (usize, usize)) -> String {
    let s = &e[span.0..span.1];
    let mut end = s.len();
    let bytes = s;
    let pats: [&[u8]; 4] = [b" who", b" which", b" that", b","];
    for p in pats {
        if let Some(pos) = bytes.windows(p.len()).position(|w| w == p) { end = end.min(pos); }
    }
    String::from_utf8_lossy(&s[..end]).trim().to_string()
}

// Projection référentielle: rôles causaux (pas le contenu), lien ∅ = critère majeur.
pub fn project(res: &SearchResult, e: &[u8]) -> (Vec<Link>, Answer) {
    let n = res.assemblies.len();
    let b = &res.b; let drv = &res.drive;
    let mut links = Vec::new();
    for i in 0..n { for j in i + 1..n {
        if drv[i] < 0.05 && drv[j] < 0.05 { continue; }
        let score = cosine(&role(b, i), &role(b, j));
        if score > 0.6 {
            let shared = overlap(&res.assemblies[i].words, &res.assemblies[j].words);
            links.push(Link { a: i, b: j, score, shared });
        }
    }}
    links.sort_by(|a, c| c.score.partial_cmp(&a.score).unwrap());

    let probe_words = &res.assemblies[res.probe].words;
    let top_head = (0..n).filter(|&i| res.assemblies[i].kind == 1 && b[res.probe][i] > 0.0)
        .max_by(|&a, &c| drv[c].partial_cmp(&drv[a]).unwrap());
    let bound_def = top_head.and_then(|h|
        (0..n).filter(|&i| res.assemblies[i].head == Some(h))
            .max_by(|&a, &c| drv[c].partial_cmp(&drv[a]).unwrap()));
    let projected = bound_def.and_then(|d|
        (0..n).filter(|&i| res.assemblies[i].kind == 1
            && !overlap(&res.assemblies[i].words, probe_words)
            && (b[d][i] > 0.0 || crate::dynamics::words_of(e, res.assemblies[d].span.0, res.assemblies[d].span.1)
                .iter().any(|w| res.assemblies[i].words.contains(w))))
            .max_by(|&a, &c| drv[c].partial_cmp(&drv[a]).unwrap()));

    let phrase = match bound_def {
        Some(d) => core_phrase(e, res.assemblies[d].span),
        None => String::from_utf8_lossy(&e[res.assemblies[res.probe].span.0..res.assemblies[res.probe].span.1]).into_owned(),
    };
    let no_shared = match projected {
        Some(p) => !overlap(&res.assemblies[p].words, probe_words),
        None => false,
    };
    (links, Answer { top_head, bound_def, projected, phrase, no_shared })
}

pub fn span_text(e: &[u8], a: &Assembly) -> String {
    String::from_utf8_lossy(&e[a.span.0..a.span.1]).trim().to_string()
}