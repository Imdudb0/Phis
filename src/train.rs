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
        Plan { k, seed, dim, lambda, mu, batch: mu, epochs: dim, holdout: 2 * lambda }
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
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub roles: [f64; ROLES],
    pub links: f64,
    pub tops: f64,
    pub phi_f1: f64,
    pub phi_steps: f64,
    pub evals: usize,
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

// Tirage stratifié par échelle : un contenu par strate de [0, 1].
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

// Signal d'entraînement : φ_sharp moyen relatif à E, par réseau, sur le même lot.
fn fitness(nets: &[Net], percepts: &[Percept]) -> Vec<(f64, usize)> {
    let b = percepts.len();
    let cells: Vec<(f64, usize)> = par_map(nets.len() * b, |t| {
        let p = &percepts[t % b];
        let mut rng = Rng::new(model::seed(&p.eff));
        let o = search::ask(&nets[t / b], p, &mut rng);
        (o.phi, o.evals)
    });
    cells
        .chunks(b)
        .map(|row| {
            let phi = row.iter().map(|c| c.0).sum::<f64>() / b as f64;
            let evals = row.iter().map(|c| c.1).sum::<usize>();
            (phi, evals)
        })
        .collect()
}

// Seul endroit où la cause cachée est lue.
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

impl E