use crate::net::{Evidence, Field, Net};
use crate::percept::Percept;
use crate::phi::{complex, Sia, Tpm};
use crate::rng::Rng;
use std::collections::HashMap;
const NONE: usize = usize::MAX;
const TOL: f64 = 1e-9;
const RIVALS: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub evals: usize,
    pub patience: usize,
}
impl Budget {
    pub fn measured(p: &Percept) -> Budget {
        let evals = p.repeats.len().clamp(32, 1024);
        Budget { evals, patience: (evals / 4).max(8) }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub bits: f64,
}
#[derive(Clone, Debug, Default)]
pub struct Explanation {
    pub classes: Vec<usize>,
    pub spans: Vec<Span>,
}
impl Explanation {
    pub fn text(&self, eff: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for (i, s) in self.spans.iter().enumerate() {
            if i > 0 {
                out.push(b'\n');
            }
            out.extend_from_slice(&eff[s.start..s.end]);
        }
        out
    }
}
#[derive(Clone, Debug)]
pub struct Residual {
    pub unclear: f64,
    pub inexpressive: f64,
    pub unjust: f64,
    pub spans: Vec<Span>,
}
#[derive(Clone, Debug)]
pub struct Step {
    pub eval: usize,
    pub class: usize,
    pub added: bool,
    pub accepted: bool,
    pub phi: f64,
    pub best: f64,
    pub size: usize,
    pub cause: usize,
}

fn draw_rival(rng: &mut Rng, f: &Field, size: usize) -> Vec<bool> {
    let mut ids = f.cands.clone();
    rng.shuffle(&mut ids);
    let mut sel = vec![false; f.nc];
    for &c in ids.iter().take(size) {
        sel[c] = true;
    }
    sel
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub r: usize,
    pub phi: f64,
    pub rival: f64,
    pub cause: usize,
    pub effect: usize,
    pub sia: Option<Sia>,
    pub explanation: Explanation,
    pub residual: Residual,
    pub evals: usize,
    pub trace: Vec<Step>,
}
impl Outcome {
    pub fn gain(&self) -> f64 {
        self.phi - self.rival
    }
}
struct Judged {
    tpm: Tpm,
    sia: Option<Sia>,
    phi: f64,
}
fn judge(net: &Net, f: &Field, sel: &[bool], state: usize) -> Judged {
    let tpm = net.tpm(&Evidence::of(f, sel));
    let sia = complex(&tpm, state);
    let phi = sia.as_ref().map_or(0.0, |s| s.phi);
    Judged { tpm, sia, phi }
}
fn abduce(rng: &mut Rng, cur: &Judged, state: usize) -> usize {
    match &cur.sia {
        Some(s) => (state & !s.units) | s.global(rng.weighted(&s.cause)),
        None => rng.weighted(&cur.tpm.cause(state)),
    }
}
fn count(f: &Field, sel: &[bool]) -> usize {
    f.cands.iter().filter(|&&c| sel[c]).count()
}
fn unit_gap(p: &Percept) -> usize {
    let mut h: HashMap<usize, usize> = HashMap::new();
    for e in &p.rel.forward {
        let g = e.w as usize;
        if g > 0 {
            *h.entry(g).or_insert(0) += 1;
        }
    }
    h.into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
        .map_or(0, |(g, _)| g)
}
fn explain(p: &Percept, f: &Field, sel: &[bool], gap: usize) -> Explanation {
    let classes: Vec<usize> = f.cands.iter().copied().filter(|&c| sel[c]).collect();
    let mut spans: Vec<Span> = Vec::new();
    for a in &p.anchors {
        if !sel[a.class] {
            continue;
        }
        let (s, e) = (a.pos, a.end());
        let join = spans.last().is_some_and(|l| s <= l.end + gap);
        if join {
            if let Some(l) = spans.last_mut() {
                l.end = l.end.max(e);
            }
        } else {
            spans.push(Span { start: s, end: e, bits: 0.0 });
        }
    }
    for sp in spans.iter_mut() {
        sp.bits = p.mass(sp.start, sp.end);
    }
    Explanation { classes, spans }
}
fn leftover(p: &Percept, cur: &Judged, state: usize, ex: &Explanation) -> Residual {
    let n = p.eff.len();
    let total = p.cum[n];
    let (unclear, unjust) = match &cur.sia {
        Some(s) => {
            let z = (state & !s.units) | s.global(s.cause_state);
            (
                (1.0 - s.cause[s.cause_state]).clamp(0.0, 1.0),
                (1.0 - cur.tpm.effect_at(z, state)).clamp(0.0, 1.0),
            )
        }
        None => (1.0, 1.0),
    };
    let covered: f64 = ex.spans.iter().map(|s| s.bits).sum();
    let inexpressive = if total > 0.0 { (1.0 - covered / total).clamp(0.0, 1.0) } else { 0.0 };
    let mut spans: Vec<Span> = Vec::new();
    let mut at = 0usize;
    for s in &ex.spans {
        if s.start > at {
            spans.push(Span { start: at, end: s.start, bits: p.mass(at, s.start) });
        }
        at = at.max(s.end);
    }
    if at < n {
        spans.push(Span { start: at, end: n, bits: p.mass(at, n) });
    }
    spans.sort_by(|a, b| b.bits.partial_cmp(&a.bits).unwrap_or(std::cmp::Ordering::Equal));
    Residual { unclear, inexpressive, unjust, spans }
}
pub fn run(
    net: &Net,
    p: &Percept,
    rng: &mut Rng,
    budget: Budget,
    on_step: &mut dyn FnMut(&Step),
) -> Outcome {
    let field = Field::new(net, p);
    let state = field.response();
    let mut sel = field.region(state);
    let mut cur = judge(net, &field, &sel, state);
    let beta = net.sharpness();
    let mut evals = 1usize;
    let mut since = 0usize;
    let mut tabu = vec![false; field.cands.len()];
    let mut trace: Vec<Step> = Vec::new();
    let first = Step {
        eval: 0,
        class: NONE,
        added: true,
        accepted: true,
        phi: cur.phi,
        best: cur.phi,
        size: count(&field, &sel),
        cause: state,
    };
    on_step(&first);
    trace.push(first);
    while evals < budget.evals && since < budget.patience {
        let z = abduce(rng, &cur, state);
        let mut w = vec![0.0f64; field.cands.len()];
        let mut total = 0.0f64;
        for (i, &c) in field.cands.iter().enumerate() {
            if tabu[i] {
                continue;
            }
            let sgn = if sel[c] { -1.0 } else { 1.0 };
            let v = (beta * sgn * field.align(c, z)).exp();
            w[i] = v;
            total += v;
        }
        if !(total > 0.0) {
            break;
        }
        let i = rng.weighted(&w);
        let c = field.cands[i];
        let adding = !sel[c];
        sel[c] = adding;
        let next = judge(net, &field, &sel, state);
        evals += 1;
        let tried = next.phi;
        let ok = if adding { tried >= cur.phi - TOL } else { tried > cur.phi + TOL };
        if ok {
            cur = next;
            tabu.fill(false);
            since = 0;
        } else {
            sel[c] = !adding;
            tabu[i] = true;
            since += 1;
        }
        let step = Step {
            eval: evals - 1,
            class: c,
            added: adding,
            accepted: ok,
            phi: tried,
            best: cur.phi,
            size: count(&field, &sel),
            cause: z,
        };
        on_step(&step);
        trace.push(step);
    }
    let size = count(&field, &sel);
    let mut rivals = 0.0f64;
    for _ in 0..RIVALS {
        let alt = draw_rival(rng, &field, size);
        rivals += judge(net, &field, &alt, state).phi;
    }
    let rival = rivals / RIVALS as f64;
    evals += RIVALS;
    let found = cur.sia.is_some();
    let explanation = if found {
        explain(p, &field, &sel, unit_gap(p))
    } else {
        Explanation::default()
    };
    let residual = leftover(p, &cur, state, &explanation);
    let (cause, effect) = match &cur.sia {
        Some(s) => (s.global(s.cause_state), s.global(s.effect_state)),
        None => (0, 0),
    };
    Outcome {
        r: state,
        phi: cur.phi,
        rival,
        cause,
        effect,
        sia: cur.sia,
        explanation,
        residual,
        evals,
        trace,
    }
}
pub fn ask(net: &Net, p: &Percept, rng: &mut Rng) -> Outcome {
    run(net, p, rng, Budget::measured(p), &mut |_: &Step| {})
}