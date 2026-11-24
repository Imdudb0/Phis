use crate::net::{Evidence, Field, Net};
use crate::percept::Percept;
use crate::phi::{complex, Sia, Tpm};
use crate::rng::Rng;
use std::collections::HashMap;

const NONE: usize = usize::MAX;
const TOL: f64 = 1e-9;
const CHANCE_DRAWS: usize = 8;                     

/// Budget mesuré sur le percept : une évaluation de φ_sharp par classe de répétitions.
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

/// L'explication s'exprime dans l'alphabet de E : empans d'octets de E.
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

/// Ce qui manque pour une explication plus claire, plus expressive, plus juste.
#[derive(Clone, Debug)]
pub struct Residual {
    /// 1 − P(état-cause spécifié) dans le répertoire-cause du complexe.
    pub unclear: f64,
    /// Part de l'information de E (bits) non couverte par l'explication.
    pub inexpressive: f64,
    /// 1 − P(R | cause) : la cause abduite ne régénère pas R.
    pub unjust: f64,
    /// Trous de E non expliqués, triés par bits décroissants.
    pub spans: Vec<Span>,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub eval: usize,
    pub class: usize,
    pub added: bool,
    pub accepted: bool,
    /// φ_sharp de l'hypothèse testée à ce pas.
    pub phi: f64,
    /// φ_sharp de l'hypothèse courante après décision (le sens en train de se découvrir).
    pub best: f64,
    pub size: usize,
    /// Cause abduite (masque d'unités) qui a guidé ce pas.
    pub cause: usize,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    /// R : état des unités déclenché par la perception.
    pub r: usize,
    pub phi: f64,
    pub chance: f64,   // + φ moyen de sélections aléatoires de même taille
    /// Cause et effet spécifiés par le complexe (bits positionnés sur `sia.units`).
    pub cause: usize,
    pub effect: usize,
    pub sia: Option<Sia>,
    pub explanation: Explanation,
    pub residual: Residual,
    pub evals: usize,
    pub trace: Vec<Step>,
}

struct Judged {
    tpm: Tpm,
    sia: Option<Sia>,
    phi: f64,
    unjust: f64,
}

fn judge(net: &Net, f: &Field, sel: &[bool], state: usize) -> Judged {
    let tpm = net.tpm(&Evidence::of(f, sel));
    let sia = complex(&tpm, state);
    let phi = sia.as_ref().map_or(0.0, |s| s.phi);
    let unjust = match &sia {
        Some(s) => {
            let z = (state & !s.units) | s.global(s.cause_state);
            (1.0 - tpm.effect_at(z, state)).clamp(0.0, 1.0)
        }
        None => 1.0,
    };
    Judged { tpm, sia, phi, unjust }
}

// Abduction créative : on tire la cause dans le répertoire-cause de R (complexe si φ > 0,
// système entier sinon) au lieu d'en prendre le mode.
fn abduce(rng: &mut Rng, cur: &Judged, state: usize) -> usize {
    match &cur.sia {
        Some(s) => (state & !s.units) | s.global(rng.weighted(&s.cause)),
        None => rng.weighted(&cur.tpm.cause(state)),
    }
}

fn count(f: &Field, sel: &[bool]) -> usize {
    f.cands.iter().filter(|&&c| sel[c]).count()
}

fn baseline(net: &Net, f: &Field, rng: &mut Rng, state: usize, sel: &[bool]) -> f64 {
    let weight = |c: usize| -> f64 { f.by_class[c].iter().map(|&a| f.sal[a]).sum() };
    let target: f64 = f.cands.iter().filter(|&&c| sel[c]).map(|&c| weight(c)).sum();
    let mut pool: Vec<usize> = f.cands.clone();
    let mut sum = 0.0f64;
    for _ in 0..CHANCE_DRAWS {
        rng.shuffle(&mut pool);
        let mut pick = vec![false; f.nc];
        let mut got = 0.0_f64;
        for &c in &pool {
            if got >= target {
                break;
            }
            pick[c] = true;
            got += weight(c);
        }
        sum += judge(net, f, &pick, state).phi;
    }
    sum / CHANCE_DRAWS as f64
}

fn mass_of(f: &Field, sel: &[bool]) -> f64 {
    (0..f.na).filter(|&a| sel[f.class_of[a]]).map(|a| f.sal[a]).sum()
}

struct Curve(Vec<(f64, f64)>); // (masse, phi) trié
impl Curve {
    fn at(&self, m: f64) -> f64 {
        let p = &self.0;
        if p.is_empty() { return 0.0; }
        if m <= p[0].0 { return p[0].1; }
        for w in p.windows(2) {
            if m <= w[1].0 {
                let t = (m - w[0].0) / (w[1].0 - w[0].0).max(1e-9);
                return w[0].1 + t * (w[1].1 - w[0].1);
            }
        }
        p.last().unwrap().1
    }
}

fn chance_curve(net: &Net, f: &Field, rng: &mut Rng, state: usize) -> Curve {
    let total: f64 = f.sal.iter().sum();
    let mut pts = Vec::new();
    for i in 0..=10 {
        let target = total * i as f64 / 10.0;
        for _ in 0..2 {
            let mut pool = f.cands.clone();
            rng.shuffle(&mut pool);
            let mut pick = vec![false; f.nc];
            let mut got = 0.0;
            for &c in &pool {
                if got >= target { break; }
                pick[c] = true;
                got += f.by_class[c].iter().map(|&a| f.sal[a]).sum::<f64>();
            }
            pts.push((mass_of(f, &pick), judge(net, f, &pick, state).phi));
        }
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    Curve(pts)
}

// Échelle de séparation propre à E : écart positif le plus fréquent entre ancres consécutives.
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
        Some(s) => ((1.0 - s.cause[s.cause_state]).clamp(0.0, 1.0), cur.unjust),
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
    let curve = chance_curve(net, &field, &mut chance_rng, state);
    let mut cur = judge(net, &field, &sel, state);
    let mut cur_obj = cur.phi - curve.at(mass_of(&field, &sel));
    let mut chance_rng = rng.fork();   // + flux dédié : mêmes tirages pour tous les réseaux
    let mut sel = field.region(state);
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
            // à ajouter si aligné avec la cause abduite ; à retirer si opposé à elle
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
        
        /*let tried = next.phi;
        let ok = tried > cur.phi + TOL
            || (adding && (tried - cur.phi).abs() <= TOL && next.unjust < cur.unjust - TOL);*/
            
        let next = judge(net, &field, &sel, state);
        let next_obj = next.phi - curve.at(mass_of(&field, &sel));
        let ok = next_obj > cur_obj + TOL
            || (adding && (next_obj - cur_obj).abs() <= TOL && next.unjust < cur.unjust - TOL);
            
        if ok { cur = next; cur_obj = next_obj; tabu.fill(false); since = 0; } else { sel[c] = !adding; tabu[i] = true; since += 1; }
        
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
    let found = cur.sia.is_some();
    let explanation = if found {
        explain(p, &field, &sel, unit_gap(p))
    } else {
        Explanation::default()
    };
    let residual = leftover(p, &cur, state, &explanation);
    let chance = baseline(net, &field, &mut chance_rng, state, &sel);
    let (cause, effect) = match &cur.sia {
        Some(s) => (s.global(s.cause_state), s.global(s.effect_state)),
        None => (0, 0),
    };
    Outcome {
        r: state,
        phi: cur.phi,
        chance,                        
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