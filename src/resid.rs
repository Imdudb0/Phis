//! resid.rs — the typed, localised and bounded residual (part 4b-1 of 6). Causal only.
//!
//! For an anchored mechanism with phi_c != phi_e (half-distinctions included: they are what the
//! min throws away entirely):
//!     rho = |phi_c - phi_e|
//!     phi_c < phi_e : rho_cause, effects are sharp, the cause is diffuse or tied: a discriminant
//!                     is missing (clarity)
//!     phi_e < phi_c : rho_effet, the cause is sharp, the effects are diffuse: reach or context is
//!                     missing (expressivity)
//! Localisation, per purview node: the nodes are those of the weak side's irreducible purview(s),
//! ties merged. When the weak side has none, the strong side's purview nodes are used (`from_strong`).
//! Each node is weighted by the normalised entropy of its marginal in the unpartitioned weak-side
//! repertoire (the flattest nodes are where that side lacks discrimination), and rho is shared in
//! proportion: the shares sum to rho. A node that carries a byte of E is read back through
//! `Ctx::e_pos` (done in sharp.rs); latent nodes keep their share apart.
//! Bound: filling the weak side up to the strong one raises phi_d by exactly rho, strong side
//! unchanged, so no filling can gain more than rho. Nothing is filled here.

use crate::phi::{Dir, Dist, Phi, PHI_EPS};

/// Shares below this are not reported.
pub const MASS_EPS: f64 = 1e-12;

/// What the min of phi_c and phi_e throws away for one anchored mechanism.
#[derive(Clone, Debug)]
pub struct Residual {
    pub mech: Vec<usize>,
    /// the missing side: Cause when phi_c < phi_e, Effect when phi_e < phi_c
    pub weak: Dir,
    /// |phi_c - phi_e|
    pub rho: f64,
    /// tied specified states on the weak side (0: that side has no irreducible purview)
    pub ties: usize,
    /// (purview node, share of rho), by node id; the shares sum to rho
    pub nodes: Vec<(usize, f64)>,
    /// the nodes are the strong side's (the weak side has no irreducible purview)
    pub from_strong: bool,
}

impl Residual {
    /// What the weak side lacks, in the vocabulary of the spec.
    pub fn need(&self) -> &'static str {
        match self.weak {
            Dir::Cause => "a discriminant (clarté)",
            Dir::Effect => "reach or context (expressivité)",
        }
    }
}

/// Entropy of a distribution divided by ln |A(E)|: 0 = sharp, 1 = flat.
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

/// Residuals of the anchored mechanisms `dists` (state, grain and background fixed by `ph`).
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
        // flattest reading over tied purviews, per node
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

/// Where the residual of the whole E sits.
#[derive(Clone, Debug)]
pub struct ResidualMap {
    /// mass per byte of E (decodable in E)
    pub by_pos: Vec<f64>,
    /// mass on latent purview nodes
    pub latent: f64,
    /// sum of rho_cause: effects are sharp, the cause is diffuse or tied
    pub cause: f64,
    /// sum of rho_effet: the cause is sharp, the effects are diffuse
    pub effect: f64,
}

impl ResidualMap {
    pub fn total(&self) -> f64 {
        self.cause + self.effect
    }

    /// Bytes of E by decreasing residual mass (k = 0: all of them).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_is_normalised() {
        assert!((norm_entropy(&[0.5, 0.5]) - 1.0).abs() < 1e-12);
        assert_eq!(norm_entropy(&[1.0, 0.0]), 0.0);
        assert_eq!(norm_entropy(&[1.0]), 0.0);
        let h = norm_entropy(&[0.7, 0.2, 0.1]);
        assert!(h > 0.0 && h < 1.0);
    }

    #[test]
    fn top_orders_by_mass_then_position() {
        let m = ResidualMap { by_pos: vec![0.0, 0.5, 0.2, 0.5], latent: 0.0, cause: 0.0, effect: 1.2 };
        assert_eq!(m.top(0), vec![(1, 0.5), (3, 0.5), (2, 0.2)]);
        assert_eq!(m.top(1), vec![(1, 0.5)]);
        assert!((m.total() - 1.2).abs() < 1e-12);
    }
}