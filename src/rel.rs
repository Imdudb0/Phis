//! rel.rs — relations of Dist_E (part 4b-1 of 6).
//!
//! A relatum is one purview face (cause or effect) of a distinction of Dist_E, with its specified
//! state(s). Two relata overlap on the LATENT nodes (not anchored in E) that both contain and on
//! which their specified symbols agree (congruence). A byte never overlaps, so a surface matcher
//! has no relation at all. With o the congruent overlap and u = phi_d / |Z| the power per node of
//! a relatum:
//!     phi_r = |o| * min_i u_i  <=  min_i phi_d(i)          (|o| <= |Z_i| for every member i)
//! A relation is a set of distinctions (at least two), counted once, at its best choice of faces
//! and of tied states: ex aequo never multiplies it. Self-relations (a distinction's own cause and
//! effect) are not counted: relations "relient" distinctions.
//!
//! Estimator: every pair (found through the shared latent nodes) plus, per (shared latent node,
//! symbol), one closure gathering every distinction that has a face with that symbol on that node.
//! Intermediate orders are not enumerated, so the sum of phi_r can only under-count.
//! `cross`: the members' footprints in E are pairwise disjoint. The share of phi_sharp carried by
//! such relations is the abstraction A(E) of the spec.

use crate::phi::Dist;

/// A relation among distinctions of Dist_E.
#[derive(Clone, Debug)]
pub struct Relation {
    /// indices into the list of distinctions, increasing, at least two
    pub members: Vec<usize>,
    /// latent nodes shared by the chosen purviews with the same specified symbol, increasing
    pub overlap: Vec<usize>,
    pub phi: f64,
    /// pairwise disjoint footprints in E: a relation of meaning, not of bytes
    pub cross: bool,
}

/// One purview face of a distinction, as seen by the relation search.
struct Relatum {
    /// index of the distinction
    d: usize,
    /// purview nodes, increasing
    z: Vec<usize>,
    /// z[i] is latent (not anchored in E)
    lat: Vec<bool>,
    /// specified states (ties kept), aligned with z
    states: Vec<Vec<u16>>,
    /// phi_d / |z|
    unit: f64,
}

/// Smallest node that is latent in both relata, if any.
fn first_common(a: &Relatum, b: &Relatum) -> Option<usize> {
    for (i, &x) in a.z.iter().enumerate() {
        if a.lat[i] {
            if let Ok(k) = b.z.binary_search(&x) {
                if b.lat[k] {
                    return Some(x);
                }
            }
        }
    }
    None
}

/// Largest congruent latent overlap over the tied states of the two relata.
fn congruent(a: &Relatum, b: &Relatum) -> Vec<usize> {
    let mut best: Vec<usize> = Vec::new();
    for sa in &a.states {
        for sb in &b.states {
            let mut o: Vec<usize> = Vec::new();
            for (i, &x) in a.z.iter().enumerate() {
                if !a.lat[i] {
                    continue;
                }
                if let Ok(k) = b.z.binary_search(&x) {
                    if b.lat[k] && sa[i] == sb[k] {
                        o.push(x);
                    }
                }
            }
            if o.len() > best.len() {
                best = o;
            }
        }
    }
    best
}

/// Overlap and phi_r of the closure on node `x` holding symbol `v`: `pick` holds one relatum per
/// distinction. The overlap is every latent node present in all chosen purviews with one symbol.
fn common_overlap(rel: &[Relatum], pick: &[(usize, usize)], x: usize, v: u16) -> Option<(Vec<usize>, f64)> {
    let mut sel: Vec<(&Relatum, &Vec<u16>)> = Vec::with_capacity(pick.len());
    for &(_, ri) in pick {
        let r = &rel[ri];
        let i = r.z.binary_search(&x).ok()?;
        let s = r.states.iter().find(|s| s[i] == v)?;
        sel.push((r, s));
    }
    let (r0, s0) = sel[0];
    let mut o: Vec<usize> = Vec::new();
    for (i, &y) in r0.z.iter().enumerate() {
        if !r0.lat[i] {
            continue;
        }
        let ok = sel[1..].iter().all(|(r, s)| match r.z.binary_search(&y) {
            Ok(k) => r.lat[k] && s[k] == s0[i],
            Err(_) => false,
        });
        if ok {
            o.push(y);
        }
    }
    if o.is_empty() {
        return None;
    }
    let u = sel.iter().map(|(r, _)| r.unit).fold(f64::INFINITY, f64::min);
    let phi = o.len() as f64 * u;
    Some((o, phi))
}

/// True when the footprints of the members are pairwise disjoint.
fn disjoint(foot: &[Vec<usize>], mem: &[usize]) -> bool {
    for p in 0..mem.len() {
        for q in (p + 1)..mem.len() {
            let (a, b) = (&foot[mem[p]], &foot[mem[q]]);
            if a.iter().any(|x| b.binary_search(x).is_ok()) {
                return false;
            }
        }
    }
    true
}

/// Relations among the distinctions of `dists` (only those with phi_d > 0 take part).
/// `latent[j]`: node j is not anchored in E. `foot[i]`: sorted footprint in E of `dists[i]`.
/// Output order: pairs by (members, decreasing phi), then closures by members. Deterministic.
pub fn relations(dists: &[Dist], latent: &[bool], foot: &[Vec<usize>]) -> Vec<Relation> {
    let n = latent.len();
    let mut rel: Vec<Relatum> = Vec::new();
    for (di, d) in dists.iter().enumerate() {
        if !d.is_distinction() {
            continue;
        }
        for f in d.cause.iter().chain(d.effect.iter()) {
            let lat: Vec<bool> = f.nodes.iter().map(|&j| latent[j]).collect();
            if f.states.is_empty() || !lat.iter().any(|&b| b) {
                continue;
            }
            rel.push(Relatum {
                d: di,
                z: f.nodes.clone(),
                lat,
                states: f.states.clone(),
                unit: d.phi_d / f.nodes.len() as f64,
            });
        }
    }
    let mut by_node: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (ri, r) in rel.iter().enumerate() {
        for (i, &x) in r.z.iter().enumerate() {
            if r.lat[i] {
                by_node[x].push(ri);
            }
        }
    }
    // pairs, each relatum pair evaluated once: at the smallest latent node they share
    let mut pairs: Vec<(usize, usize, f64, Vec<usize>)> = Vec::new();
    for x in 0..n {
        let lst = &by_node[x];
        for p in 0..lst.len() {
            for q in (p + 1)..lst.len() {
                let (a, b) = (&rel[lst[p]], &rel[lst[q]]);
                if a.d == b.d || first_common(a, b) != Some(x) {
                    continue;
                }
                let o = congruent(a, b);
                if o.is_empty() {
                    continue;
                }
                let phi = o.len() as f64 * a.unit.min(b.unit);
                pairs.push((a.d.min(b.d), a.d.max(b.d), phi, o));
            }
        }
    }
    pairs.sort_by(|u, v| {
        u.0.cmp(&v.0)
            .then(u.1.cmp(&v.1))
            .then(v.2.total_cmp(&u.2))
            .then_with(|| u.3.cmp(&v.3))
    });
    pairs.dedup_by(|later, first| later.0 == first.0 && later.1 == first.1);
    // closures on shared latent nodes (three distinctions or more)
    let mut cl: Vec<(Vec<usize>, f64, Vec<usize>)> = Vec::new();
    let mut trip: Vec<(u16, usize, usize)> = Vec::new();
    for x in 0..n {
        if by_node[x].len() < 3 {
            continue;
        }
        trip.clear();
        for &ri in &by_node[x] {
            let r = &rel[ri];
            if let Ok(i) = r.z.binary_search(&x) {
                for s in &r.states {
                    trip.push((s[i], r.d, ri));
                }
            }
        }
        trip.sort_unstable();
        let mut g = 0usize;
        while g < trip.len() {
            let v = trip[g].0;
            let mut h = g;
            while h < trip.len() && trip[h].0 == v {
                h += 1;
            }
            // per distinction, the relatum with the largest power per node
            let mut pick: Vec<(usize, usize)> = Vec::new();
            for &(_, d, ri) in &trip[g..h] {
                let same = matches!(pick.last(), Some(c) if c.0 == d);
                if same {
                    if let Some(c) = pick.last_mut() {
                        if rel[ri].unit > rel[c.1].unit {
                            c.1 = ri;
                        }
                    }
                } else {
                    pick.push((d, ri));
                }
            }
            if pick.len() >= 3 {
                if let Some((o, phi)) = common_overlap(&rel, &pick, x, v) {
                    cl.push((pick.iter().map(|p| p.0).collect(), phi, o));
                }
            }
            g = h;
        }
    }
    cl.sort_by(|u, v| u.0.cmp(&v.0).then(v.1.total_cmp(&u.1)).then_with(|| u.2.cmp(&v.2)));
    cl.dedup_by(|later, first| later.0 == first.0);
    let mut out: Vec<Relation> = Vec::with_capacity(pairs.len() + cl.len());
    for (a, b, phi, o) in pairs {
        out.push(Relation { members: vec![a, b], overlap: o, phi, cross: disjoint(foot, &[a, b]) });
    }
    for (mem, phi, o) in cl {
        let cross = disjoint(foot, &mem);
        out.push(Relation { members: mem, overlap: o, phi, cross });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phi::Face;

    fn face(nodes: &[usize], states: &[&[u16]], phi: f64) -> Face {
        Face { nodes: nodes.to_vec(), phi, states: states.iter().map(|s| s.to_vec()).collect() }
    }

    fn dist(mech: usize, cause: Vec<Face>, effect: Vec<Face>, phi_d: f64) -> Dist {
        Dist {
            mech: vec![mech],
            mech_state: vec![0],
            cause,
            effect,
            phi_c: phi_d,
            phi_e: phi_d,
            phi_d,
            anchored: true,
        }
    }

    #[test]
    fn pairs_and_closure_by_hand() {
        // nodes 0 and 1 carry bytes of E; nodes 2..=7 are latent
        let latent = [false, false, true, true, true, true, true, true];
        let ds = vec![
            dist(0, vec![face(&[0, 2], &[&[0, 1]], 0.6)], vec![face(&[3], &[&[0]], 0.6)], 0.6),
            dist(1, vec![face(&[1, 2], &[&[1, 1]], 0.4)], vec![face(&[4], &[&[0]], 0.4)], 0.4),
            dist(2, vec![face(&[2, 5], &[&[0, 0]], 0.8)], vec![face(&[6], &[&[0]], 0.8)], 0.8),
            dist(3, vec![face(&[2, 7], &[&[1, 0]], 0.9)], vec![face(&[1], &[&[1]], 0.9)], 0.9),
        ];
        let foot = vec![vec![0usize], vec![1], vec![4], vec![1]];
        let r = relations(&ds, &latent, &foot);
        // node 2 holds symbol 1 for d0, d1, d3 and symbol 0 for d2 (incongruent with the others)
        assert_eq!(r.len(), 4);
        assert_eq!(r[0].members, vec![0, 1]);
        assert_eq!(r[0].overlap, vec![2]);
        assert!((r[0].phi - 0.2).abs() < 1e-12); // 1 * min(0.6/2, 0.4/2)
        assert!(r[0].cross);
        assert_eq!(r[1].members, vec![0, 3]);
        assert!((r[1].phi - 0.3).abs() < 1e-12); // 1 * min(0.6/2, 0.9/2)
        assert!(r[1].cross);
        assert_eq!(r[2].members, vec![1, 3]);
        assert!((r[2].phi - 0.2).abs() < 1e-12);
        assert!(!r[2].cross); // d1 and d3 both touch byte 1
        assert_eq!(r[3].members, vec![0, 1, 3]); // closure on (node 2, symbol 1)
        assert!((r[3].phi - 0.2).abs() < 1e-12);
        assert!(!r[3].cross);
        for rl in &r {
            let lo = rl.members.iter().map(|&i| ds[i].phi_d).fold(f64::INFINITY, f64::min);
            assert!(rl.phi <= lo + 1e-12, "phi_r above min phi_d");
            assert!(!rl.overlap.is_empty());
            assert!(rl.overlap.iter().all(|&x| latent[x]));
        }
    }

    #[test]
    fn a_shared_byte_is_not_a_relation() {
        let latent = [false, false, true];
        let ds = vec![
            dist(0, vec![face(&[0], &[&[0]], 0.5)], vec![face(&[1], &[&[0]], 0.5)], 0.5),
            dist(1, vec![face(&[0], &[&[0]], 0.7)], vec![face(&[1], &[&[0]], 0.7)], 0.7),
        ];
        let foot = vec![vec![0usize], vec![0]];
        assert!(relations(&ds, &latent, &foot).is_empty());
    }
}