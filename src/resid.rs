use crate::phi::{Dir, Dist, Phi, PHI_EPS};
pub const MASS_EPS: f64 = 1e-12;
#[derive(Clone, Debug)]
pub struct Residual {
    pub mech: Vec<usize>,
    pub weak: Dir,
    pub rho: f64,
    pub ties: usize,
    pub nodes: Vec<(usize, f64)>,
    pub from_strong: bool,
}
impl Residual {
    pub fn need(&self) -> &'static str {
        match self.weak {
            Dir::Cause => "a discriminant (clarté)",
            Dir::Effect => "reach or context (expressivité)",
        }
    }
}
pub fn norm_entropy(p: &[f64]) -> f64 {
    if p.len() < 2 {
        return 0.0;
    }
    let mut h = 0.0f64;
    for &x in p {
        if x > 0.0 {
            h -= x * x.ln();
        }
    }
    (h / (p.len() as f64).ln()).clamp(0.0, 1.0)
}
pub fn residuals(ph: &Phi<'_, '_>, dists: &[Dist]) -> Vec<Residual> {
    let mut out: Vec<Residual> = Vec::new();
    for d in dists {
        let rho = (d.phi_c - d.phi_e).abs();
        if rho <= PHI_EPS {
            continue;
        }
        let weak = if d.phi_c < d.phi_e { Dir::Cause } else { Dir::Effect };
        let (wf, sf) = match weak {
            Dir::Cause => (&d.cause, &d.effect),
            Dir::Effect => (&d.effect, &d.cause),
        };
        let from_strong = wf.is_empty();
        let faces = if from_strong { sf } else { wf };
        let ties: usize = wf.iter().map(|f| f.states.len()).sum();
        let mut acc: Vec<(usize, f64)> = Vec::new();
        for f in faces {
            let rows = ph.repertoire_marginals(&d.mech, &f.nodes, weak);
            for (i, &j) in f.nodes.iter().enumerate() {
                let h = norm_entropy(&rows[i]);
                match acc.iter().position(|x| x.0 == j) {
                    Some(p) => {
                        if h > acc[p].1 {
                            acc[p].1 = h;
                        }
                    }
                    None => acc.push((j, h)),
                }
            }
        }
        acc.sort_by_key(|x| x.0);
        let tot: f64 = acc.iter().map(|x| x.1).sum();
        let k = acc.len().max(1) as f64;
        let nodes: Vec<(usize, f64)> = acc
            .iter()
            .map(|&(j, h)| (j, rho * if tot > MASS_EPS { h / tot } else { 1.0 / k }))
            .collect();
        out.push(Residual { mech: d.mech.clone(), weak, rho, ties, nodes, from_strong });
    }
    out
}
#[derive(Clone, Debug)]
pub struct ResidualMap {
    pub by_pos: Vec<f64>,
    pub latent: f64,
    pub cause: f64,
    pub effect: f64,
}
impl ResidualMap {
    pub fn total(&self) -> f64 {
        self.cause + self.effect
    }
    pub fn top(&self, k: usize) -> Vec<(usize, f64)> {
        let mut v: Vec<(usize, f64)> = self
            .by_pos
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, m)| m > MASS_EPS)
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        if k > 0 {
            v.truncate(k);
        }
        v
    }
}