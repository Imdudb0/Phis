use crate::model::{self, Answer, Model};
use crate::net::Net;
use crate::percept::Percept;
use crate::ptgrs::{Cause, Ptgrs, Role, Sample};
use crate::rng::Rng;
use crate::search;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicUsize, Ordering as Atomic};
pub const ROLES: usize = 5;
const NAMES: [&str; ROLES] = [
    "règles utilisées",
    "règles inutilisées",
    "instances",
    "bruit",
    "leurres",
];
const SIGMA0: f64 = 0.3;
const NULL_SALT: u64 = 0x4e55_4c4c;
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub k: usize,
    pub seed: u64,
    pub dim: usize,
    pub lambda: usize,
    pub mu: usize,
    pub batch: usize,
    pub epochs: usize,
    pub holdout: usize,
}
impl Plan {
    pub fn measured(k: usize, seed: u64) -> Plan {
        let dim = Net::new(k, &mut Rng::new(seed)).params().len();
        let lambda = 4 + (3.0 * (dim as f64).ln()) as usize;
        let mu = lambda / 2;
        let epochs: usize = std::env::var("PHIS_EPOCHS").ok().and_then(|v| v.parse().ok()).unwrap_or(dim);
        Plan { k, seed, dim, lambda, mu, batch: mu, epochs, holdout: 2 * lambda }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Epoch {
    pub epoch: usize,
    pub sigma: f64,
    pub center: f64,
    pub best: f64,
    pub parents: f64,
    pub mean: f64,
    pub evals: usize,
    pub total: usize,
}
#[derive(Clone, Debug)]
pub struct Eval {
    pub samples: usize,
    pub phi: f64,
    pub null_phi: f64,    // + φ sur E permuté
    pub chance_phi: f64,  // + φ d'une sélection aléatoire
    pub phi_sd: f64,      // + écart-type entre contenus (0 = effondré)
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub roles: [f64; ROLES],
    pub links: f64,
    pub tops: f64,
    pub phi_f1: f64,
    pub phi_steps: f64,
    pub evals: usize,
    pub purity: f64,
}
#[derive(Clone, Debug)]
pub struct Session {
    pub plan: Plan,
    pub model: Model,
    pub before: Eval,
    pub after: Eval,
    pub trace: Vec<Epoch>,
    pub evals: usize,
}
struct Row {
    phi: f64,
    null: f64,      // +
    chance: f64,
    steps: f64,
    evals: usize,
    precision: f64,
    recall: f64,
    f1: f64,
    cover: [f64; ROLES],
    bits: [f64; ROLES],
    links: f64,
    nlinks: usize,
    tops: f64,
    ntops: usize,
}
fn slot(role: Role) -> usize {
    match role {
        Role::Used => 0,
        Role::Unused => 1,
        Role::Instance => 2,
        Role::Noise => 3,
        Role::Decoy => 4,
    }
}
fn quot(x: f64, y: f64) -> f64 {
    if y > 0.0 { x / y } else { 0.0 }
}
fn mean(v: &[f64]) -> f64 {
    quot(v.iter().sum::<f64>(), v.len() as f64)
}
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (mean(x), mean(y));
    let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in x.iter().zip(y) {
        let (da, db) = (a - mx, b - my);
        sxy += da * db;
        sxx += da * da;
        syy += db * db;
    }
    quot(sxy, (sxx * syy).sqrt())
}
fn spread(x: &[f64]) -> f64 {
    let m = mean(x);
    let v: Vec<f64> = x.iter().map(|a| (a - m) * (a - m)).collect();
    mean(&v).sqrt()
}

fn control(eff: &[u8]) -> Vec<u8> {
    let mut v = eff.to_vec();
    Rng::new(model::seed(eff) ^ NULL_SALT).shuffle(&mut v);
    v
}

fn quiet(net: &Net, p: &Percept, budget: search::Budget) -> search::Outcome {
    let mut rng = Rng::new(model::seed(&p.eff));
    search::run(net, p, &mut rng, budget, &mut |_: &search::Step| {})
}

fn by_net(cells: Vec<(f64, usize)>, b: usize) -> Vec<(f64, usize)> {
    cells
        .chunks(b)
        .map(|r| (r.iter().map(|c| c.0).sum::<f64>() / b as f64, r.iter().map(|c| c.1).sum::<usize>()))
        .collect()
}

fn null_eff(s: &Sample) -> Vec<u8> {
    let eff = &s.eff; let mut out = eff.clone();
    let mut rng = Rng::new(model::seed(eff) ^ NULL_SALT);
    for l in &s.cause.lines {
        let line = &eff[l.start..l.end];
        let mut cnt = [0usize; 256];
        for &b in line { cnt[b as usize] += 1; }
        let sep = (0..256).max_by_key(|&b| cnt[b]).unwrap() as u8;
        let mut toks: Vec<&[u8]> = line.split(|&b| b==sep).filter(|t| !t.is_empty()).collect();
        rng.shuffle(&mut toks);
        let mut rb = Vec::new();
        for (i,t) in toks.iter().enumerate() { if i>0 { rb.push(sep); } rb.extend_from_slice(t); }
        let n = line.len().min(rb.len());
        out[l.start..l.start+n].copy_from_slice(&rb[..n]);
    }
    out
}

fn fitness(nets: &[Net], batch: &[Sample], percepts: &[Percept]) -> Vec<(f64, usize)> {
    let nulls: Vec<Percept> = batch.iter().map(|s| Percept::new(&null_eff(s))).collect();
    let b = percepts.len();
    by_net(par_map(nets.len()*b, |t| {
        let (p, q) = (&percepts[t % b], &nulls[t % b]);
        let o = quiet(&nets[t/b], p, search::Budget::measured(p));
        let n = quiet(&nets[t/b], q, search::Budget::measured(p));
        (o.phi - n.phi, o.evals + n.evals)   // le signal est le CONTRASTE de contenu
    }), b)
}

// DIAGNOSTIC seulement : F1 contre la cause cachée comme fitness
fn fitness_oracle(nets: &[Net], batch: &[Sample], percepts: &[Percept]) -> Vec<(f64, usize)> {
    let b = percepts.len();
    by_net(
        par_map(nets.len() * b, |t| {
            let (p, s) = (&percepts[t % b], &batch[t % b]);
            let outcome = quiet(&nets[t / b], p, search::Budget::measured(p));
            let evals = outcome.evals;
            let a = Answer { percept: p.clone(), outcome };
            (mark(&s.cause, &a, 0.0).f1, evals)
        }),
        b,
    )
}

fn par_map<R, F>(n: usize, f: F) -> Vec<R>
where
    R: Send + 'static,
    F: Fn(usize) -> R + Sync,
{
    let workers = std::thread::available_parallelism()
        .map_or(1, |w| w.get())
        .clamp(1, n.max(1));
    let next = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, R)>> = std::thread::scope(|s| {
        let mut hs = Vec::with_capacity(workers);
        for _ in 0..workers {
            hs.push(s.spawn(|| {
                let mut out: Vec<(usize, R)> = Vec::new();
                loop {
                    let i = next.fetch_add(1, Atomic::Relaxed);
                    if i >= n {
                        break;
                    }
                    out.push((i, f(i)));
                }
                out
            }));
        }
        hs.into_iter()
            .map(|h| h.join().expect("évaluation interrompue"))
            .collect()
    });
    let mut slots: Vec<Option<R>> = (0..n).map(|_| None).collect();
    for part in parts {
        for (i, r) in part {
            slots[i] = Some(r);
        }
    }
    slots
        .into_iter()
        .map(|r| r.expect("tâche manquante"))
        .collect()
}
pub fn draw(rng: &mut Rng, count: usize) -> Vec<Sample> {
    let mut out = Vec::with_capacity(count);
    for j in 0..count {
        let scale = (j as f64 + rng.unit()) / count as f64;
        out.push(Ptgrs::at_scale(rng, scale));
    }
    out
}
fn instantiate(template: &Net, x: &[f64]) -> Net {
    let mut net = template.clone();
    for (d, &v) in net.params_mut().iter_mut().zip(x) {
        *d = v as f32;
    }
    net
}
fn mark(cause: &Cause, a: &Answer, null: f64) -> Row {
    let p = &a.percept;
    let spans = &a.outcome.explanation.spans;
    let mut cover = [0.0f64; ROLES];
    let mut bits = [0.0f64; ROLES];
    let mut frac: Vec<f64> = Vec::with_capacity(cause.lines.len());
    let (mut hit, mut want) = (0.0f64, 0.0f64);
    for (i, l) in cause.lines.iter().enumerate() {
        let total = p.mass(l.start, l.end);
        let mut got = 0.0f64;
        for s in spans {
            let (x, y) = (s.start.max(l.start), s.end.min(l.end));
            if x < y {
                got += p.mass(x, y);
            }
        }
        let r = slot(l.role);
        cover[r] += got;
        bits[r] += total;
        if cause.is_cause_line(i) {
            hit += got;
            want += total;
        }
        frac.push(quot(got, total));
    }
    let said: f64 = spans.iter().map(|s| s.bits).sum();
    let recall = quot(hit, want);
    let precision = quot(hit, said);
    let f1 = quot(2.0 * precision * recall, precision + recall);
    let mut links = 0.0f64;
    for &(x, y) in &cause.links {
        links += frac[x].min(frac[y]);
    }
    let mut tops = 0.0f64;
    for &t in &cause.tops {
        tops += frac[t];
    }
    Row {
        phi: a.outcome.phi,
        null,
        chance: a.outcome.chance,
        steps: cause.steps as f64,
        evals: a.outcome.evals,
        precision,
        recall,
        f1,
        cover,
        bits,
        links,
        nlinks: cause.links.len(),
        tops,
        ntops: cause.tops.len(),
    }
}
impl Eval {
    fn of(rows: &[Row]) -> Eval {
        let phi: Vec<f64> = rows.iter().map(|r| r.phi).collect();
        let f1: Vec<f64> = rows.iter().map(|r| r.f1).collect();
        let steps: Vec<f64> = rows.iter().map(|r| r.steps).collect();
        let prec: Vec<f64> = rows.iter().map(|r| r.precision).collect();
        let rec: Vec<f64> = rows.iter().map(|r| r.recall).collect();
        let mut roles = [0.0f64; ROLES];
        for (j, v) in roles.iter_mut().enumerate() {
            let seen: Vec<f64> = rows
                .iter()
                .filter(|r| r.bits[j] > 0.0)
                .map(|r| r.cover[j] / r.bits[j])
                .collect();
            *v = mean(&seen);
        }
        let links = quot(
            rows.iter().map(|r| r.links).sum::<f64>(),
            rows.iter().map(|r| r.nlinks).sum::<usize>() as f64,
        );
        let tops = quot(
            rows.iter().map(|r| r.tops).sum::<f64>(),
            rows.iter().map(|r| r.ntops).sum::<usize>() as f64,
        );
        let nul: Vec<f64> = rows.iter().map(|r| r.null).collect();
        let cha: Vec<f64> = rows.iter().map(|r| r.chance).collect();
        let pur: Vec<f64> = rows
            .iter()
            .map(|r| quot(r.cover[0] + r.cover[1] + r.cover[2], r.cover.iter().sum::<f64>()))
            .collect();
        Eval {
            samples: rows.len(),
            phi: mean(&phi),
            null_phi: mean(&nul),
            chance_phi: mean(&cha),
            phi_sd: spread(&phi),
            precision: mean(&prec),
            recall: mean(&rec),
            f1: mean(&f1),
            roles,
            links,
            tops,
            phi_f1: pearson(&phi, &f1),
            phi_steps: pearson(&phi, &steps),
            evals: rows.iter().map(|r| r.evals).sum(),
            purity: mean(&pur),
        }
    }
}
pub fn evaluate(model: &Model, samples: &[Sample]) -> Eval {
    let rows = par_map(samples.len(), |i| {
        let s = &samples[i];
        let a = model.ask(&s.eff);
        let q = Percept::new(&control(&s.eff));
        let n = quiet(&model.net, &q, search::Budget::measured(&a.percept));
        mark(&s.cause, &a, n.phi)
    });
    Eval::of(&rows)
}
pub fn session(plan: &Plan, on_epoch: &mut dyn FnMut(&Epoch)) -> Session {
    assert!(plan.mu >= 1 && plan.mu <= plan.lambda && plan.batch >= 1, "plan invalide");
    let mut master = Rng::new(plan.seed);
    let mut r_init = master.fork();
    let mut r_step = master.fork();
    let mut r_data = master.fork();
    let mut r_hold = master.fork();
    let start = Net::new(plan.k, &mut r_init);
    let dim = start.params().len();
    assert_eq!(dim, plan.dim, "plan incohérent avec l'architecture");
    let hold = draw(&mut r_hold, plan.holdout);
    let before = evaluate(&Model::new(start.clone()), &hold);
    let nf = dim as f64;
    let mut w: Vec<f64> = (1..=plan.mu)
        .map(|i| (plan.mu as f64 + 0.5).ln() - (i as f64).ln())
        .collect();
    let wsum: f64 = w.iter().sum();
    for v in w.iter_mut() {
        *v /= wsum;
    }
    let mueff = 1.0 / w.iter().map(|v| v * v).sum::<f64>();
    let cs = (mueff + 2.0) / (nf + mueff + 5.0);
    let ds = 1.0 + 2.0 * (((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
    let chi = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));
    let gain = (cs * (2.0 - cs) * mueff).sqrt();
    let mut m: Vec<f64> = start.params().iter().map(|&v| f64::from(v)).collect();
    let rms = (m.iter().map(|v| v * v).sum::<f64>() / nf).sqrt();
    let mut sigma = if rms > 0.0 { SIGMA0 * rms } else { 1.0 };
    let floor = f64::from(f32::EPSILON) * rms;
    let mut path = vec![0.0f64; dim];
    let mut trace: Vec<Epoch> = Vec::with_capacity(plan.epochs);
    let mut total = 0usize;
    let oracle = std::env::var_os("PHIS_ORACLE").is_some();
    for epoch in 1..=plan.epochs {
        let batch = draw(&mut r_data, plan.batch);
        let percepts: Vec<Percept> = batch.iter().map(|s| Percept::new(&s.eff)).collect();
        let mut z: Vec<Vec<f64>> = Vec::with_capacity(plan.lambda);
        for _ in 0..plan.lambda {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(r_step.normal());
            }
            z.push(v);
        }
        let mut nets: Vec<Net> = z
            .iter()
            .map(|zi| {
                let x: Vec<f64> = m.iter().zip(zi).map(|(&c, &d)| c + sigma * d).collect();
                instantiate(&start, &x)
            })
            .collect();
        nets.push(instantiate(&start, &m));
        let scores = if oracle { fitness_oracle(&nets, &batch, &percepts) } else { fitness(&nets, &batch, &percepts) };
        let center = scores[plan.lambda].0;
        let mut order: Vec<usize> = (0..plan.lambda).collect();
        order.sort_by(|&p, &q| {
            scores[q].0.partial_cmp(&scores[p].0).unwrap_or(Ordering::Equal).then(p.cmp(&q))
        });
        let used = sigma;
        let mut yw = vec![0.0f64; dim];
        for (r, &i) in order.iter().take(plan.mu).enumerate() {
            for d in 0..dim {
                yw[d] += w[r] * z[i][d];
            }
        }
        for d in 0..dim {
            m[d] += used * yw[d];
            path[d] = (1.0 - cs) * path[d] + gain * yw[d];
        }
        let norm = path.iter().map(|v| v * v).sum::<f64>().sqrt();
        sigma = used * ((cs / ds) * (norm / chi - 1.0)).exp();
        let spent: usize = scores.iter().map(|s| s.1).sum();
        total += spent;
        let rec = Epoch {
            epoch,
            sigma: used,
            center,
            best: scores[order[0]].0,
            parents: order[..plan.mu].iter().map(|&i| scores[i].0).sum::<f64>() / plan.mu as f64,
            mean: scores[..plan.lambda].iter().map(|s| s.0).sum::<f64>() / plan.lambda as f64,
            evals: spent,
            total,
        };
        on_epoch(&rec);
        trace.push(rec);
        if sigma < floor {
            break;
        }
    }
    let tuned = instantiate(&start, &m);
    assert!(
        tuned.params().iter().all(|v| v.is_finite()),
        "paramètres non finis après entraînement"
    );
    let trained = Model::new(tuned);
    let after = evaluate(&trained, &hold);
    Session { plan: *plan, model: trained, before, after, trace, evals: total }
}
pub fn header(p: &Plan) -> String {
    format!(
        "Budget mesuré : {} unités, {} paramètres | population {} (parents {}) | lot {} contenus | {} générations | {} contenus d'évaluation | graine {}",
        p.k, p.dim, p.lambda, p.mu, p.batch, p.epochs, p.holdout, p.seed
    )
}
pub fn progress(e: &Epoch) -> String {
    format!(
        "gén {:>4} | σ {:>10.3e} | φ_sharp centre {:>9.6} | meilleur {:>9.6} | parents {:>9.6} | moyen {:>9.6} | éval {:>8} | cumul {:>11}",
        e.epoch, e.sigma, e.center, e.best, e.parents, e.mean, e.evals, e.total
    )
}
pub fn report(s: &Session) -> String {
    let (b, a) = (&s.before, &s.after);
    let mut out = format!(
        "Séance terminée : {} générations, {} évaluations de φ_sharp.\n\
         Évaluation a posteriori sur la cause cachée ({} contenus inédits) :\n\
         {:<38}{:>12}{:>12}\n",
        s.trace.len(),
        s.evals,
        a.samples,
        "",
        "avant",
        "après"
    );
    let mut row = |label: &str, x: f64, y: f64| {
        out.push_str(&format!("{label:<38}{x:>12.6}{y:>12.6}\n"));
    };
    row("φ_sharp moyen", b.phi, a.phi);
    row("φ_sharp contrôle (E permuté)", b.null_phi, a.null_phi);
    row("φ_sharp hasard (même taille)", b.chance_phi, a.chance_phi);
    row("écart-type φ_sharp (0 = effondré)", b.phi_sd, a.phi_sd);
    row("précision (masse en bits)", b.precision, a.precision);
    row("rappel (masse en bits)", b.recall, a.recall);
    row("F1", b.f1, a.f1);
    for r in 0..ROLES {
        row(&format!("couverture : {}", NAMES[r]), b.roles[r], a.roles[r]);
    }
    row("liens cachés retrouvés", b.links, a.links);
    row("règles de tête retrouvées", b.tops, a.tops);
    row("corrélation φ_sharp / F1", b.phi_f1, a.phi_f1);
    row("corrélation φ_sharp / étapes", b.phi_steps, a.phi_steps);
    row(
        "évaluations de φ par requête",
        b.evals as f64 / b.samples.max(1) as f64,
        a.evals as f64 / a.samples.max(1) as f64,
    );
    row("pureté (grammaire / expliqué)", b.purity, a.purity);
    out
}