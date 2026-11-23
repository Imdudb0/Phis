use crate::percept::Percept;
use crate::search::{Outcome, Span, Step};
use std::cmp::Ordering;
const NONE: usize = usize::MAX;
fn is_cont(b: u8) -> bool {
    b & 0xC0 == 0x80
}
fn width(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 0,
    }
}
fn glyph(eff: &[u8], at: usize) -> Option<(usize, usize)> {
    if at == 0 || at >= eff.len() || !is_cont(eff[at]) {
        return None;
    }
    let mut a = at;
    while a > 0 && at - a < 3 && is_cont(eff[a]) {
        a -= 1;
    }
    let end = a + width(eff[a]);
    if end > at && end <= eff.len() && eff[at..end].iter().all(|&b| is_cont(b)) {
        Some((a, end))
    } else {
        None
    }
}
fn down(eff: &[u8], at: usize) -> usize {
    glyph(eff, at).map_or(at, |g| g.0)
}
fn up(eff: &[u8], at: usize) -> usize {
    glyph(eff, at).map_or(at, |g| g.1)
}
fn cover(p: &Percept, spans: &[Span]) -> Vec<Span> {
    let mut raw: Vec<(usize, usize)> = spans
        .iter()
        .map(|s| (down(&p.eff, s.start), up(&p.eff, s.end)))
        .collect();
    raw.sort_unstable();
    let mut out: Vec<Span> = Vec::with_capacity(raw.len());
    for (a, b) in raw {
        let join = out.last().is_some_and(|l| a <= l.end);
        if join {
            if let Some(l) = out.last_mut() {
                l.end = l.end.max(b);
            }
        } else {
            out.push(Span { start: a, end: b, bits: 0.0 });
        }
    }
    for s in out.iter_mut() {
        s.bits = p.mass(s.start, s.end);
    }
    out
}
fn lacking(p: &Percept, o: &Outcome) -> Vec<Span> {
    let mut v: Vec<Span> = Vec::with_capacity(o.residual.spans.len());
    for s in &o.residual.spans {
        let (a, b) = (up(&p.eff, s.start), down(&p.eff, s.end));
        if a < b {
            v.push(Span { start: a, end: b, bits: p.mass(a, b) });
        }
    }
    v.sort_by(|x, y| y.bits.partial_cmp(&x.bits).unwrap_or(Ordering::Equal));
    v
}
pub fn progress(s: &Step) -> String {
    let act = if s.class == NONE {
        String::from("départ")
    } else if s.added {
        format!("ajout classe {}", s.class)
    } else {
        format!("retrait classe {}", s.class)
    };
    let verdict = if s.accepted { "retenu" } else { "écarté" };
    format!(
        "éval {:>5} | {:<18} | {} | φ_sharp {:>10.6} | meilleur {:>10.6} | taille {:>4}",
        s.eval, act, verdict, s.phi, s.best, s.size
    )
}
pub fn explanation(p: &Percept, o: &Outcome) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    for (i, s) in cover(p, &o.explanation.spans).iter().enumerate() {
        if i > 0 {
            out.push(b'\n');
        }
        out.extend_from_slice(&p.eff[s.start..s.end]);
    }
    out
}
pub fn residual(p: &Percept, o: &Outcome) -> Vec<u8> {
    let r = &o.residual;
    let mut out = format!(
        "Ce qui manque pour une explication plus claire, plus expressive, plus juste (0 : rien, 1 : tout) :\n  clarté       {:.6}\n  expressivité {:.6}\n  justesse     {:.6}\n",
        r.unclear, r.inexpressive, r.unjust
    )
    .into_bytes();
    let zones = lacking(p, o);
    if !zones.is_empty() {
        let head = format!(
            "Non expliqué, par masse d'information décroissante ({} zones) :\n",
            zones.len()
        );
        out.extend_from_slice(head.as_bytes());
        for z in &zones {
            let tag = format!("[{}..{}] {:.3} bits\n", z.start, z.end, z.bits);
            out.extend_from_slice(tag.as_bytes());
            out.extend_from_slice(&p.eff[z.start..z.end]);
            out.push(b'\n');
        }
    }
    out
}
pub fn report(p: &Percept, o: &Outcome) -> Vec<u8> {
    let mut out = explanation(p, o);
    if out.is_empty() {
        out.extend_from_slice("Aucun sens trouvé.".as_bytes());
    }
    let head = format!("\n\nφ_sharp = {:.6} ({} évaluations)\n", o.phi, o.evals);
    out.extend_from_slice(head.as_bytes());
    out.extend_from_slice(&residual(p, o));
    out
}