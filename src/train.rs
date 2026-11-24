use crate::model::{self, Answer, Model};
use crate::net::{Evidence, Field, Net};
use crate::percept::Percept;
use crate::phi;
use crate::ptgrs::{Cause, Ptgrs, Role, Sample};
use crate::rng::Rng;
use crate::search::{self, Budget, Outcome, Step};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicUsize, Ordering as Atomic};

const FLAT: f64 = 1e-9;
const FLAT_MAX: usize = 15;
const LUCK: u64 = 0xA5A5_5A5A_C3C3_3C3C;

pub const ROLES: usize = 5;
const NAMES: [&str; ROLES] = [
    "règles utilisées",
    "règles inutilisées",
    "instances",
    "bruit",
    "leurres",
];

const SIGMA0: f64 = 0.3;

#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub k: usize,
    pub seed: u64,
    pub dim: usize,
    pub lambda: usize,
    pub mu: usize,
    pub batch: usize,
    /// Nombre de fenêtres disjointes dans le pool fixe (pool = batch × windows).
    pub windows: usize,
    /// Générations consécutives passées sur la même fenêtre : c'est ce qui rend
    /// l'objectif stationnaire par blocs et les nombres aléatoires réellement communs.
    pub block: usize,
    pub epochs: usize,
    pub holdout: usize,
}

impl Plan {
    pub fn measured(k: usize, seed: u64) -> Plan {
        let dim = Net::new(k, &mut Rng::new(seed)).params().len();
        let lambda = 4 + (3.0 * (dim as f64).ln()) as usize;
        let mu = lambda / 2;
        Plan {
            k, seed, dim, lambda, mu,
            batch: mu,
            windows: 4,
            block: 8,
            epochs: dim,
            holdout: 2 * lambda,
        }
    }
    pub fn pool(&self) -> usize { self.batch * self.windows }
}

#[derive(Clone, Copy, Debug)]
pub struct Epoch {
    pub epoch: usize,
    pub window: usize,
    pub sigma: f64,
    pub center: f64,
    pub best: f64,
    pub parents: f64,
    pub mean: f64,
    /// Δφ du centre sur la fenêtre 0, identique à chaque génération : seule colonne
    /// comparable d'une génération à l'autre.
    pub probe: f64,
    pub real: f64,
    pub base: f64,
    pub evals: usize,
    pub total: usize,
}

#[derive(Clone, Debug)]
pub struct Eval {
    pub samples: usize,
    pub phi: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub roles: [f64; ROLES],
    /// Moyenne appariée par contenu de couverture(Used) − couverture(Unused).
    pub selectivity: f64,
    pub selectivity_n: usize,
    pub links: f64,
    pub tops: f64,
    pub phi_f1: f64,
    pub phi_sel: f64,
    pub phi_steps: f64,
    pub evals: usize,
}

/// Comparaison appariée après − avant sur exactement les mêmes contenus.
#[derive(Clone, Debug)]
pub struct Paired {
    pub label: &'static str,
    pub n: usize,
    pub delta: f64,
    pub se: f64,
    pub t: f64,
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
}

impl Paired {
    fn of(label: &'static str, before: &[f64], after: &[f64]) -> Paired {
        assert_eq!(before.len(), after.len(), "échantillons non appariés");
        let d: Vec<f64> = after.iter().zip(before).map(|(a, b)| a - b).collect();
        let n = d.len();
        let delta = mean(&d);
        let (mut wins, mut losses, mut ties) = (0usize, 0usize, 0usize);
        for &v in &d {
            if v > FLAT { wins += 1; } else if v < -FLAT { losses += 1; } else { ties += 1; }
        }
        let se = if n >= 2 {
            let var = d.iter().map(|v| (v - delta) * (v - delta)).sum::<f64>() / (n - 1) as f64;
            (var / n as f64).sqrt()
        } else { 0.0 };
        Paired { label, n, delta, se, t: quot(delta, se), wins, losses, ties }
    }
}

#[derive(Clone, Debug)]
pub struct Session {
    pub plan: Plan,
    pub model: Model,
    pub before: Eval,
    pub after: Eval,
    pub paired: Vec<Paired>,
    pub trace: Vec<Epoch>,
    pub evals: usize,
}

struct Row {
    phi: f64,
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

impl Row {
    /// Défini seulement si le contenu contient les deux rôles : sinon la différence
    /// n'a pas de sens et il ne faut pas la moyenner avec les autres.
    fn selectivity(&self) -> Option<f64> {
        if self.bits[0] > 0.0 && self.bits[1] > 0.0 {
            Some(self.cover[0] / self.bits[0] - self.cover[1] / self.bits[1])
        } else {
            None
        }
    }
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

fn quot(x: f64, y: f64) -> f64 { if y > 0.0 { x / y } else { 0.0 } }

fn mean(v: &[f64]) -> f64 { quot(v.iter().sum::<f64>(), v.len() as f64) }

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
                    if i >= n { break; }
                    out.push((i, f(i)));
                }
                out
            }));
        }
        hs.into_iter().map(|h| h.join().expect("évaluation interrompue")).collect()
    });
    let mut slots: Vec<Option<R>> = (0..n).map(|_| None).collect();
    for part in parts {
        for (i, r) in part { slots[i] = Some(r); }
    }
    slots.into_iter().map(|r| r.expect("tâche manquante")).collect()
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
    for (d, &v) in net.params_mut().iter_mut().zip(x) { *d = v as f32; }
    net
}

#[derive(Clone, Copy, Debug)]
struct Score {
    delta: f64,
    real: f64,
    base: f64,
    evals: usize,
}

fn null_of(p: &Percept, rng: &mut Rng) -> Percept {
    let mut e = p.eff.clone();
    rng.shuffle(&mut e);
    Percept::new(&e)
}

fn chance(net: &Net, p: &Percept, o: &Outcome, rng: &mut Rng) -> f64 {
    let f = Field::new(net, p);
    let mut pool = f.cands.clone();
    rng.shuffle(&mut pool);
    let mut sel = vec![false; f.nc];
    for &c in pool.iter().take(o.explanation.classes.len()) { sel[c] = true; }
    phi::phi_sharp(&net.tpm(&Evidence::of(&f, &sel)), o.r)
}

/// Inchangée. Les graines sont dérivées du contenu : pool fixe ⇒ nombres aléatoires
/// communs à tous les candidats et à toutes les générations d'un même bloc.
fn fitness(nets: &[Net], percepts: &[Percept], nulls: &[Percept]) -> Vec<Score> {
    let b = percepts.len();
    let cells: Vec<Score> = par_map(nets.len() * b, |t| {
        let (net, i) = (&nets[t / b], t % b);
        let (p, q) = (&percepts[i], &nulls[i]);
        let budget = Budget::measured(p);
        let seed = model::seed(&p.eff);
        let real = search::run(net, p, &mut Rng::new(seed), budget, &mut |_: &Step| {});
        let null = search::run(net, q, &mut Rng::new(model::seed(&q.eff)), budget, &mut |_: &Step| {});
        let base = null.phi.max(chance(net, p, &real, &mut Rng::new(seed ^ LUCK)));
        Score { delta: real.phi - base, real: real.phi, base, evals: real.evals + null.evals }
    });
    let n = b as f64;
    cells
        .chunks(b)
        .map(|row| Score {
            delta: row.iter().map(|c| c.delta).sum::<f64>() / n,
            real: row.iter().map(|c| c.real).sum::<f64>() / n,
            base: row.iter().map(|c| c.base).sum::<f64>() / n,
            evals: row.iter().map(|c| c.evals).sum::<usize>(),
        })
        .collect()
}

fn mark(cause: &Cause, a: &Answer) -> Row {
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
            if x < y { got += p.mass(x, y); }
        }
        let r = slot(l.role);
        cover[r] += got;
        bits[r] += total;
        if cause.is_cause_line(i) { hit += got; want += total; }
        frac.push(quot(got, total));
    }
    let said: f64 = spans.iter().map(|s| s.bits).sum();
    let recall = quot(hit, want);
    let precision = quot(hit, said);
    let f1 = quot(2.0 * precision * recall, precision + recall);
    let mut links = 0.0f64;
    for &(x, y) in &cause.links { links += frac[x].min(frac[y]); }
    let mut tops = 0.0f64;
    for &t in &cause.tops { tops += frac[t]; }
    Row {
        phi: a.outcome.phi, steps: cause.steps as f64, evals: a.outcome.evals,
        precision, recall, f1, cover, bits,
        links, nlinks: cause.links.len(), tops, ntops: cause.tops.len(),
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
            let seen: Vec<f64> = rows.iter()
                .filter(|r| r.bits[j] > 0.0)
                .map(|r| r.cover[j] / r.bits[j])
                .collect();
            *v = mean(&seen);
        }
        let sel: Vec<f64> = rows.iter().filter_map(|r| r.selectivity()).collect();
        let pairs: Vec<(f64, f64)> = rows.iter()
            .filter_map(|r| r.selectivity().map(|s| (r.phi, s)))
            .collect();
        let px: Vec<f64> = pairs.iter().map(|p| p.0).collect();
        let py: Vec<f64> = pairs.iter().map(|p| p.1).collect();
        let links = quot(
            rows.iter().map(|r| r.links).sum::<f64>(),
            rows.iter().map(|r| r.nlinks).sum::<usize>() as f64,
        );
        let tops = quot(
            rows.iter().map(|r| r.tops).sum::<f64>(),
            rows.iter().map(|r| r.ntops).sum::<usize>() as f64,
        );
        Eval {
            samples: rows.len(),
            phi: mean(&phi),
            precision: mean(&prec),
            recall: mean(&rec),
            f1: mean(&f1),
            roles,
            selectivity: mean(&sel),
            selectivity_n: sel.len(),
            links, tops,
            phi_f1: pearson(&phi, &f1),
            phi_sel: pearson(&px, &py),
            phi_steps: pearson(&phi, &steps),
            evals: rows.iter().map(|r| r.evals).sum(),
        }
    }
}

fn rows(model: &Model, samples: &[Sample]) -> Vec<Row> {
    par_map(samples.len(), |i| {
        let s = &samples[i];
        mark(&s.cause, &model.ask(&s.eff))
    })
}

pub fn evaluate(model: &Model, samples: &[Sample]) -> Eval {
    Eval::of(&rows(model, samples))
}

pub fn session(plan: &Plan, on_epoch: &mut dyn FnMut(&Epoch)) -> Session {
    assert!(plan.mu >= 1 && plan.mu <= plan.lambda && plan.batch >= 1, "plan invalide");
    assert!(plan.windows >= 1 && plan.block >= 1, "pool invalide");
    let mut master = Rng::new(plan.seed);
    let mut r_init = master.fork();
    let mut r_step = master.fork();
    let mut r_data = master.fork();
    let mut r_hold = master.fork();
    let mut r_null = master.fork();
    let start = Net::new(plan.k, &mut r_init);
    let dim = start.params().len();
    assert_eq!(dim, plan.dim, "plan incohérent avec l'architecture");

    let hold = draw(&mut r_hold, plan.holdout);
    let before_rows = rows(&Model::new(start.clone()), &hold);
    let before = Eval::of(&before_rows);

    // ---- POOL FIXE : tiré une seule fois, plus jamais retiré ----
    let pool: Vec<Percept> = draw(&mut r_data, plan.pool())
        .iter().map(|s| Percept::new(&s.eff)).collect();
    let nulls: Vec<Percept> = pool.iter().map(|p| null_of(p, &mut r_null)).collect();
    let (probe_p, probe_q) = (&pool[..plan.batch], &nulls[..plan.batch]);

    let nf = dim as f64;
    let mut w: Vec<f64> = (1..=plan.mu)
        .map(|i| (plan.mu as f64 + 0.5).ln() - (i as f64).ln())
        .collect();
    let wsum: f64 = w.iter().sum();
    for v in w.iter_mut() { *v /= wsum; }
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
    let mut flat = 0usize;

    for epoch in 1..=plan.epochs {
        let window = ((epoch - 1) / plan.block) % plan.windows;
        let lo = window * plan.batch;
        let percepts = &pool[lo..lo + plan.batch];
        let nulls_w = &nulls[lo..lo + plan.batch];

        let mut z: Vec<Vec<f64>> = Vec::with_capacity(plan.lambda);
        for _ in 0..plan.lambda {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim { v.push(r_step.normal()); }
            z.push(v);
        }
        let mut nets: Vec<Net> = z.iter()
            .map(|zi| {
                let x: Vec<f64> = m.iter().zip(zi).map(|(&c, &d)| c + sigma * d).collect();
                instantiate(&start, &x)
            })
            .collect();
        nets.push(instantiate(&start, &m));

        let scores = fitness(&nets, percepts, nulls_w);
        let center = scores[plan.lambda];
        // Sonde stationnaire : le centre sur la fenêtre 0, gratuite quand window == 0.
        let probe = if window == 0 {
            center.delta
        } else {
            fitness(&nets[plan.lambda..plan.lambda + 1], probe_p, probe_q)[0].delta
        };
        let kids = &scores[..plan.lambda];
        let mut order: Vec<usize> = (0..plan.lambda).collect();
        order.sort_by(|&p, &q| {
            kids[q].delta.partial_cmp(&kids[p].delta).unwrap_or(Ordering::Equal).then(p.cmp(&q))
        });
        let probe_idx = (1 + plan.lambda / 4).min(plan.lambda - 1);
        let used = sigma;
        if kids[order[0]].delta - kids[order[probe_idx]].delta > FLAT {
            flat = 0;
            let mut yw = vec![0.0f64; dim];
            for (r, &i) in order.iter().take(plan.mu).enumerate() {
                for d in 0..dim { yw[d] += w[r] * z[i][d]; }
            }
            for d in 0..dim {
                m[d] += used * yw[d];
                path[d] = (1.0 - cs) * path[d] + gain * yw[d];
            }
            let norm = path.iter().map(|v| v * v).sum::<f64>().sqrt();
            sigma = used * ((cs / ds) * (norm / chi - 1.0)).exp();
        } else {
            flat += 1;
            sigma = used * (0.2 + cs / ds).exp();
        }
        let spent: usize = scores.iter().map(|s| s.evals).sum();
        total += spent;
        let rec = Epoch {
            epoch, window, sigma: used,
            center: center.delta,
            best: kids[order[0]].delta,
            parents: order[..plan.mu].iter().map(|&i| kids[i].delta).sum::<f64>() / plan.mu as f64,
            mean: kids.iter().map(|s| s.delta).sum::<f64>() / plan.lambda as f64,
            probe,
            real: center.real,
            base: center.base,
            evals: spent,
            total,
        };
        on_epoch(&rec);
        trace.push(rec);
        if flat >= FLAT_MAX {
            eprintln!("plateau : {flat} générations sans information de sélection, arrêt");
            break;
        }
        if sigma < floor { break; }
    }

    let tuned = instantiate(&start, &m);
    assert!(tuned.params().iter().all(|v| v.is_finite()), "paramètres non finis");
    let trained = Model::new(tuned);
    let after_rows = rows(&trained, &hold);
    let after = Eval::of(&after_rows);

    let col = |rs: &[Row], f: fn(&Row) -> f64| -> Vec<f64> { rs.iter().map(f).collect() };
    let (mut sx, mut sy) = (Vec::new(), Vec::new());
    for (x, y) in before_rows.iter().zip(&after_rows) {
        if let (Some(u), Some(v)) = (x.selectivity(), y.selectivity()) {
            sx.push(u); sy.push(v);
        }
    }
    let paired = vec![
        Paired::of("F1", &col(&before_rows, |r| r.f1), &col(&after_rows, |r| r.f1)),
        Paired::of("précision", &col(&before_rows, |r| r.precision), &col(&after_rows, |r| r.precision)),
        Paired::of("rappel", &col(&before_rows, |r| r.recall), &col(&after_rows, |r| r.recall)),
        Paired::of("sélectivité", &sx, &sy),
    ];

    Session { plan: *plan, model: trained, before, after, paired, trace, evals: total }
}

pub fn header(p: &Plan) -> String {
    format!(
        "Budget mesuré : {} unités, {} paramètres | population {} (parents {}) | \
         pool fixe {} = lot {} × {} fenêtres, bloc {} gén | {} générations | \
         {} contenus d'évaluation | graine {}",
        p.k, p.dim, p.lambda, p.mu, p.pool(), p.batch, p.windows, p.block,
        p.epochs, p.holdout, p.seed
    )
}

pub fn progress(e: &Epoch) -> String {
    format!(
        "gén {:>4} | fen {:>2} | σ {:>10.3e} | Δφ centre {:>9.6} | sonde {:>9.6} | \
         meilleur {:>9.6} | parents {:>9.6} | moyen {:>9.6} | φ(E) {:>9.6} | base {:>9.6} | \
         éval {:>8} | cumul {:>11}",
        e.epoch, e.window, e.sigma, e.center, e.probe,
        e.best, e.parents, e.mean, e.real, e.base, e.evals, e.total
    )
}

pub fn report(s: &Session) -> String {
    let (b, a) = (&s.before, &s.after);
    let mut out = format!(
        "Séance terminée : {} générations, {} évaluations de φ_sharp.\n\
         Évaluation a posteriori sur la cause cachée ({} contenus inédits) :\n\
         {:<38}{:>12}{:>12}\n",
        s.trace.len(), s.evals, a.samples, "", "avant", "après"
    );
    let mut row = |label: &str, x: f64, y: f64| {
        out.push_str(&format!("{label:<38}{x:>12.6}{y:>12.6}\n"));
    };
    row("φ_sharp moyen", b.phi, a.phi);
    row("précision (masse en bits)", b.precision, a.precision);
    row("rappel (masse en bits)", b.recall, a.recall);
    row("F1", b.f1, a.f1);
    for r in 0..ROLES {
        row(&format!("couverture : {}", NAMES[r]), b.roles[r], a.roles[r]);
    }
    row("sélectivité (utilisées − inutil.)", b.selectivity, a.selectivity);
    row("liens cachés retrouvés", b.links, a.links);
    row("règles de tête retrouvées", b.tops, a.tops);
    row("corrélation φ_sharp / F1", b.phi_f1, a.phi_f1);
    row("corrélation φ_sharp / sélectivité", b.phi_sel, a.phi_sel);
    row("corrélation φ_sharp / étapes", b.phi_steps, a.phi_steps);
    row(
        "évaluations de φ par requête",
        b.evals as f64 / b.samples.max(1) as f64,
        a.evals as f64 / a.samples.max(1) as f64,
    );
    out.push_str(&format!(
        "\n(sélectivité : {} contenus possédant à la fois des règles utilisées et inutilisées)\n",
        a.selectivity_n
    ));
    out.push_str(&format!(
        "\nDifférences appariées (après − avant), mêmes contenus :\n\
         {:<26}{:>11}{:>11}{:>8}{:>18}\n",
        "métrique", "Δ", "err. type", "t", "gains/pertes/nuls"
    ));
    for p in &s.paired {
        let tally = format!("{}/{}/{}", p.wins, p.losses, p.ties);
        out.push_str(&format!(
            "{:<26}{:>+11.6}{:>11.6}{:>8.2}{:>18}\n",
            p.label, p.delta, p.se, p.t, tally
        ));
    }
    out.push_str("IC 95 % ≈ Δ ± 1.96·err. type ; |t| < 2 ⇒ gain non établi.\n");
    out
}