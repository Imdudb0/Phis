//! Part 5b-3 — exact relations.
//!
//! Same objects and same numbers as `rel.rs`, enumerated without its two restrictions
//! (pairs, plus one closure per shared (node, state)).
//!
//! A relation is a set S of at least two distinctions (phi_d > 0) for which some choice of
//! ONE specification per member has a non-empty overlap. A specification is a face (cause or
//! effect purview) read in one of its tied states; the overlap is the set of latent nodes
//! present in every chosen face with the same symbol, never bytes of E.
//!
//!     phi_r!(S) = max over those choices of |overlap| * min_{d in S} phi_d / |face_d|
//!
//! |overlap| <= |face_d| for every member, so phi_r <= min_{d in S} phi_d by construction.
//! One relation per set S (its best choice; ties are kept as ties and the smallest overlap
//! is listed), the way `rel.rs` deduplicates by members.
//!
//! Enumeration. Adding a member cannot create overlap, so the sets with a non-empty overlap
//! are down-closed: depth-first search over increasing member indices, cut as soon as no
//! choice keeps a node. What is carried down is the Pareto front of (overlap, smallest unit)
//! over every choice made so far: later members can only intersect and take minima, both
//! monotone, so a dominated point never matters. Exact, nothing sampled, and the sets that
//! overlap nothing are never visited.
//!
//! The pairs coincide with those of `rel.rs` (same formula). Its closure is ONE relation per
//! shared (node, state); here every subset of it is one. On equal distinctions the estimator
//! is therefore at or below the exact sum.

use super::{ExactCfg, REL_TOL};
use crate::phi::Dist;
use crate::rel::Relation;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

/// (latent node, specified symbol), sorted by node.
type Cells = Vec<(usize, u16)>;

/// A point of the search: cells shared so far and the smallest phi_d / |face| met so far.
/// For a single specification: one face read in one tied state.
#[derive(Clone, Debug)]
struct Pt {
    o: Cells,
    u: f64,
}

/// A distinction with its non-dominated specifications.
struct Item {
    d: usize,
    specs: Vec<Pt>,
}

fn meet_cells(a: &[(usize, u16)], b: &[(usize, u16)]) -> Cells {
    let (mut i, mut j) = (0usize, 0usize);
    let mut o: Cells = Vec::new();
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                if a[i].1 == b[j].1 {
                    o.push(a[i]);
                }
                i += 1;
                j += 1;
            }
        }
    }
    o
}

fn subset_cells(a: &[(usize, u16)], b: &[(usize, u16)]) -> bool {
    let mut j = 0usize;
    for x in a {
        while j < b.len() && b[j] < *x {
            j += 1;
        }
        if j >= b.len() || b[j] != *x {
            return false;
        }
        j += 1;
    }
    true
}

/// Keeps the non-dominated points: (O1, u1) is dropped if some kept (O2, u2) has
/// O2 containing O1 and u2 >= u1. Exact comparisons only, so nothing but dominated points goes.
fn pareto(mut v: Vec<Pt>) -> Vec<Pt> {
    v.sort_by(|x, y| {
        y.o.len()
            .cmp(&x.o.len())
            .then(y.u.total_cmp(&x.u))
            .then_with(|| x.o.cmp(&y.o))
    });
    let mut keep: Vec<Pt> = Vec::new();
    'next: for f in v {
        for k in &keep {
            if k.u >= f.u && subset_cells(&f.o, &k.o) {
                continue 'next;
            }
        }
        keep.push(f);
    }
    keep
}

/// Best value |O| * u of a front, ties on the value broken by the smallest node list.
fn best_of(fr: &[Pt]) -> (f64, Vec<usize>) {
    let mut bv = 0.0f64;
    let mut bo: Vec<usize> = Vec::new();
    for f in fr {
        let v = f.o.len() as f64 * f.u;
        let nodes: Vec<usize> = f.o.iter().map(|c| c.0).collect();
        if bo.is_empty() || v > bv * (1.0 + REL_TOL) {
            bv = v;
            bo = nodes;
        } else if v >= bv * (1.0 - REL_TOL) && nodes < bo {
            bv = bv.max(v);
            bo = nodes;
        }
    }
    (bv, bo)
}

/// Front after adding one more member (one of `specs`).
fn extend(fr: &[Pt], specs: &[Pt]) -> Vec<Pt> {
    let mut out: Vec<Pt> = Vec::new();
    for f in fr {
        for s in specs {
            let o = meet_cells(&f.o, &s.o);
            if !o.is_empty() {
                out.push(Pt { o, u: f.u.min(s.u) });
            }
        }
    }
    pareto(out)
}

fn meets(a: &[usize], b: &[usize]) -> bool {
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => return true,
        }
    }
    false
}

fn foot_of(foot: &[Vec<usize>], d: usize) -> &[usize] {
    foot.get(d).map(|v| v.as_slice()).unwrap_or(&[])
}

/// Same relata as `rel::relations`: faces with states and at least one latent node.
/// Indices refer to positions in `dists`; half-distinctions (phi_d = 0) are skipped.
fn build_items(dists: &[Dist], latent: &[bool]) -> Vec<Item> {
    let mut items: Vec<Item> = Vec::new();
    for (di, d) in dists.iter().enumerate() {
        if !d.is_distinction() {
            continue;
        }
        let mut pts: Vec<Pt> = Vec::new();
        for f in d.cause.iter().chain(d.effect.iter()) {
            if f.nodes.is_empty() {
                continue;
            }
            let u = d.phi_d / f.nodes.len() as f64;
            for st in &f.states {
                let mut o: Cells = Vec::new();
                for (i, &j) in f.nodes.iter().enumerate() {
                    if latent.get(j).copied().unwrap_or(false) {
                        if let Some(&s) = st.get(i) {
                            o.push((j, s));
                        }
                    }
                }
                o.sort_unstable();
                o.dedup_by_key(|c| c.0);
                if !o.is_empty() {
                    pts.push(Pt { o, u });
                }
            }
        }
        let specs = pareto(pts);
        if !specs.is_empty() {
            items.push(Item { d: di, specs });
        }
    }
    items
}

struct Search<'a> {
    items: &'a [Item],
    foot: &'a [Vec<usize>],
    kmax: usize,
    cap: usize,
    out: Vec<Relation>,
    capped: bool,
    order_cut: bool,
}

impl<'a> Search<'a> {
    /// `mem`: positions in `items` of the current set; `fr`: its front; `disj`: footprints
    /// pairwise disjoint so far (once false, false for every superset).
    fn grow(&mut self, mem: &mut Vec<usize>, fr: &[Pt], disj: bool, from: usize) {
        let items = self.items;
        let foot = self.foot;
        for k in from..items.len() {
            if self.capped {
                return;
            }
            let nf = extend(fr, &items[k].specs);
            if nf.is_empty() {
                continue;
            }
            let dj = disj
                && mem
                    .iter()
                    .all(|&m| !meets(foot_of(foot, items[m].d), foot_of(foot, items[k].d)));
            if self.out.len() >= self.cap {
                self.capped = true;
                return;
            }
            mem.push(k);
            let (phi, overlap) = best_of(&nf);
            self.out.push(Relation {
                members: mem.iter().map(|&m| items[m].d).collect(),
                overlap,
                phi,
                cross: dj,
            });
            if mem.len() < self.kmax {
                self.grow(mem, &nf, dj, k + 1);
            } else if !self.order_cut {
                for k2 in (k + 1)..items.len() {
                    if !extend(&nf, &items[k2].specs).is_empty() {
                        self.order_cut = true;
                        break;
                    }
                }
            }
            mem.pop();
        }
    }
}

/// Exact relations of one complex. `rels` are plain `rel::Relation` (members index `dists`),
/// sorted by (order, members): the pairs come first, in the estimator's order.
#[derive(Clone, Debug, Default)]
pub struct RelsX {
    pub rels: Vec<Relation>,
    /// `max_rel` reached: the list stops there, the sum is a lower bound
    pub capped: bool,
    /// `rel_order` removed relations that exist: the sum is a lower bound
    pub order_cut: bool,
}

impl RelsX {
    pub fn complete(&self) -> bool {
        !self.capped && !self.order_cut
    }
    pub fn sum(&self) -> f64 {
        self.rels.iter().map(|r| r.phi).sum()
    }
    pub fn sum_cross(&self) -> f64 {
        self.rels.iter().filter(|r| r.cross).map(|r| r.phi).sum()
    }
    /// (order, count, sum phi_r), by increasing order
    pub fn orders(&self) -> Vec<(usize, usize, f64)> {
        let mut m: BTreeMap<usize, (usize, f64)> = BTreeMap::new();
        for r in &self.rels {
            let e = m.entry(r.members.len()).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += r.phi;
        }
        m.into_iter().map(|(k, (n, s))| (k, n, s)).collect()
    }
    pub fn line(&self) -> String {
        let o: Vec<String> = self
            .orders()
            .iter()
            .map(|&(k, n, s)| format!("{}:{}x{:.4}", k, n, s))
            .collect();
        format!(
            "relations exact: {} (sum {:.4}, cross {:.4}) by order [{}]{}{}",
            self.rels.len(),
            self.sum(),
            self.sum_cross(),
            o.join(" "),
            if self.capped { "  CAPPED: lower bound" } else { "" },
            if self.order_cut { "  ORDER-CUT: lower bound" } else { "" }
        )
    }
}

/// Same arguments as `rel::relations`, plus the config (`rel_order`: 0 = every order,
/// values below 2 are read as 2; `max_rel`: cap on the number of relations).
pub fn relations_exact(
    dists: &[Dist],
    latent: &[bool],
    foot: &[Vec<usize>],
    cfg: &ExactCfg,
) -> RelsX {
    let items = build_items(dists, latent);
    let kmax = if cfg.rel_order == 0 { usize::MAX } else { cfg.rel_order.max(2) };
    let mut s = Search {
        items: &items,
        foot,
        kmax,
        cap: cfg.max_rel.max(1),
        out: Vec::new(),
        capped: false,
        order_cut: false,
    };
    for k0 in 0..items.len() {
        if s.capped {
            break;
        }
        let fr0: Vec<Pt> = items[k0].specs.clone();
        let mut mem: Vec<usize> = vec![k0];
        s.grow(&mut mem, &fr0, true, k0 + 1);
    }
    let Search { out, capped, order_cut, .. } = s;
    let mut rels = out;
    rels.sort_by(|x, y| {
        x.members
            .len()
            .cmp(&y.members.len())
            .then_with(|| x.members.cmp(&y.members))
    });
    RelsX { rels, capped, order_cut }
}

/// Estimator against exact, on the same distinctions.
#[derive(Clone, Debug, Default)]
pub struct RelGap {
    pub n_exact: usize,
    pub n_est: usize,
    pub sum_exact: f64,
    pub sum_est: f64,
    pub pair_exact: f64,
    pub pair_est: f64,
    /// member sets of the exact the estimator does not list
    pub only_exact: usize,
    /// member sets of the estimator the exact does not list (0 if the exact is complete)
    pub only_est: usize,
    /// same members, estimator phi above the exact one (0 if the exact is complete)
    pub over: usize,
    /// same members, estimator phi below the exact one
    pub under: usize,
}

impl RelGap {
    pub fn ratio(&self) -> f64 {
        if self.sum_exact > 1e-12 {
            self.sum_est / self.sum_exact
        } else if self.sum_est <= 1e-12 {
            1.0
        } else {
            f64::INFINITY
        }
    }
    /// Estimator below exact, set by set. Only meaningful when the exact result is complete.
    pub fn dominated(&self) -> bool {
        self.only_est == 0 && self.over == 0
    }
    pub fn line(&self) -> String {
        format!(
            "relations: exact {} (sum {:.4}, pairs {:.4}) | estimator {} (sum {:.4}, pairs {:.4}) | ratio {:.4} | estimator lacks {} set(s), exact lacks {}, estimator above {}, below {}",
            self.n_exact,
            self.sum_exact,
            self.pair_exact,
            self.n_est,
            self.sum_est,
            self.pair_est,
            self.ratio(),
            self.only_exact,
            self.only_est,
            self.over,
            self.under
        )
    }
}

pub fn gap(exact: &[Relation], est: &[Relation]) -> RelGap {
    let mut g = RelGap::default();
    let mut ex: HashMap<Vec<usize>, (f64, bool)> = HashMap::new();
    for r in exact {
        g.n_exact += 1;
        g.sum_exact += r.phi;
        if r.members.len() == 2 {
            g.pair_exact += r.phi;
        }
        ex.insert(r.members.clone(), (r.phi, false));
    }
    for r in est {
        g.n_est += 1;
        g.sum_est += r.phi;
        if r.members.len() == 2 {
            g.pair_est += r.phi;
        }
        match ex.get_mut(&r.members) {
            None => g.only_est += 1,
            Some(e) => {
                e.1 = true;
                let tol = 1e-9 * e.0.max(r.phi).max(1.0);
                if r.phi > e.0 + tol {
                    g.over += 1;
                } else if r.phi < e.0 - tol {
                    g.under += 1;
                }
            }
        }
    }
    g.only_exact = ex.values().filter(|e| !e.1).count();
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Hyper, Model};
    use crate::phi::Face;
    use crate::rel::relations;
    use crate::util::Rng;

    fn cfg() -> ExactCfg {
        let m = Model::random(2, 2, Hyper::default(), &mut Rng::new(3));
        ExactCfg::from_model(&m)
    }

    /// nodes 0 and 1 carry bytes of E; 2..=7 are latent
    fn lat() -> Vec<bool> {
        vec![false, false, true, true, true, true, true, true]
    }

    fn face(nodes: &[usize], states: Vec<Vec<u16>>, phi: f64) -> Face {
        Face { nodes: nodes.to_vec(), phi, states }
    }

    fn dist(phi_d: f64, cause: Vec<Face>, effect: Vec<Face>) -> Dist {
        Dist {
            mech: vec![0],
            mech_state: vec![0],
            cause,
            effect,
            phi_c: phi_d,
            phi_e: phi_d,
            phi_d,
            anchored: true,
        }
    }

    /// units: d0 1.0, d1 2.0, d2 1.5; every pair overlaps on node 2 = 7 only
    fn trio() -> Vec<Dist> {
        vec![
            dist(2.0, vec![face(&[2, 3], vec![vec![7, 8]], 2.0)], vec![]),
            dist(4.0, vec![face(&[2, 3], vec![vec![7, 9]], 4.0)], vec![]),
            dist(3.0, vec![], vec![face(&[2, 4], vec![vec![7, 5]], 3.0)]),
        ]
    }

    #[test]
    fn hand_computed_triplet() {
        let ds = trio();
        let foot = vec![vec![0], vec![1], vec![2]];
        let rx = relations_exact(&ds, &lat(), &foot, &cfg());
        assert!(rx.complete());
        let want = vec![
            (vec![0usize, 1], 1.0f64),
            (vec![0, 2], 1.0),
            (vec![1, 2], 1.5),
            (vec![0, 1, 2], 1.0),
        ];
        assert_eq!(rx.rels.len(), want.len());
        for (r, w) in rx.rels.iter().zip(want.iter()) {
            assert_eq!(r.members, w.0);
            assert!((r.phi - w.1).abs() < 1e-12, "{:?}: {}", r.members, r.phi);
            assert_eq!(r.overlap, vec![2usize]);
            assert!(r.cross);
        }
        assert!((rx.sum() - 4.5).abs() < 1e-12);
        // one triple only: the estimator finds the same four relations
        let est = relations(&ds, &lat(), &foot);
        let g = gap(&rx.rels, &est);
        assert!(g.dominated());
        assert_eq!(g.only_exact, 0);
        assert!((g.ratio() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cross_means_pairwise_disjoint_footprints() {
        let ds = trio();
        let foot = vec![vec![0], vec![1], vec![1]];
        let rx = relations_exact(&ds, &lat(), &foot, &cfg());
        let flags: Vec<bool> = rx.rels.iter().map(|r| r.cross).collect();
        assert_eq!(flags, vec![true, true, false, false]); // (0,1) (0,2) (1,2) (0,1,2)
        assert!((rx.sum_cross() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn half_distinctions_are_skipped_but_keep_their_index() {
        let mut ds = trio();
        ds.insert(1, dist(0.0, vec![face(&[2], vec![vec![7]], 0.0)], vec![]));
        let foot = vec![vec![0], vec![5], vec![1], vec![2]];
        let rx = relations_exact(&ds, &lat(), &foot, &cfg());
        let got: Vec<Vec<usize>> = rx.rels.iter().map(|r| r.members.clone()).collect();
        assert_eq!(got, vec![vec![0, 2], vec![0, 3], vec![2, 3], vec![0, 2, 3]]);
    }

    #[test]
    fn overlap_is_latent_and_congruent_only() {
        let ds = vec![
            dist(2.0, vec![face(&[0, 2], vec![vec![5, 7]], 2.0)], vec![]),
            // shares only the byte node 0 with d0: no relation
            dist(2.0, vec![face(&[0, 3], vec![vec![5, 7]], 2.0)], vec![]),
            // node 2, but another symbol: no relation
            dist(1.0, vec![face(&[2], vec![vec![6]], 1.0)], vec![]),
        ];
        let foot = vec![vec![0], vec![0], vec![1]];
        let rx = relations_exact(&ds, &lat(), &foot, &cfg());
        assert!(rx.rels.is_empty());
        // a tied state that agrees creates it
        let mut ds2 = ds.clone();
        ds2[2] = dist(1.0, vec![face(&[2], vec![vec![6], vec![7]], 1.0)], vec![]);
        let rx2 = relations_exact(&ds2, &lat(), &foot, &cfg());
        assert_eq!(rx2.rels.len(), 1);
        assert_eq!(rx2.rels[0].members, vec![0, 2]);
        assert_eq!(rx2.rels[0].overlap, vec![2usize]);
        assert!((rx2.rels[0].phi - 1.0).abs() < 1e-12);
    }

    fn four_on_one_node() -> (Vec<Dist>, Vec<Vec<usize>>) {
        let ds: Vec<Dist> = (0..4)
            .map(|_| dist(1.0, vec![face(&[2], vec![vec![7]], 1.0)], vec![]))
            .collect();
        let foot: Vec<Vec<usize>> = (0..4).map(|k| vec![k]).collect();
        (ds, foot)
    }

    #[test]
    fn every_subset_counts_not_only_the_closure() {
        let (ds, foot) = four_on_one_node();
        let l = lat();
        let full = relations_exact(&ds, &l, &foot, &cfg());
        assert!(full.complete());
        assert_eq!(full.rels.len(), 11); // 6 pairs + 4 triples + 1 quadruple
        assert_eq!(full.orders(), vec![(2, 6, 6.0), (3, 4, 4.0), (4, 1, 1.0)]);
        let est = relations(&ds, &l, &foot);
        let g = gap(&full.rels, &est);
        assert_eq!(g.n_est, 7); // 6 pairs + the closure
        assert!((g.sum_est - 7.0).abs() < 1e-12);
        assert!((g.sum_exact - 11.0).abs() < 1e-12);
        assert!(g.dominated());
        assert_eq!(g.only_exact, 4); // the four triples
        // order cut: only a lower bound, and it says so
        let two = relations_exact(&ds, &l, &foot, &cfg().with_rel_order(2));
        assert!(two.order_cut && !two.complete());
        assert!((two.sum() - 6.0).abs() < 1e-12);
        let one = relations_exact(&ds, &l, &foot, &cfg().with_rel_order(1)); // read as 2
        assert_eq!(one.rels.len(), 6);
        let three = relations_exact(&ds, &l, &foot, &cfg().with_rel_order(3));
        assert!(three.order_cut);
        assert!((three.sum() - 10.0).abs() < 1e-12);
        assert!(relations_exact(&ds, &l, &foot, &cfg().with_rel_order(4)).complete());
        // cap: flagged only if one more relation exists
        let mut c = cfg();
        c.max_rel = 10;
        let cut = relations_exact(&ds, &l, &foot, &c);
        assert!(cut.capped && !cut.complete());
        assert_eq!(cut.rels.len(), 10);
        c.max_rel = 11;
        assert!(relations_exact(&ds, &l, &foot, &c).complete());
    }

    #[test]
    fn k_distinctions_on_one_latent_state_give_2k_minus_k_minus_1_relations() {
        for k in 2..=8usize {
            let ds: Vec<Dist> = (0..k)
                .map(|_| dist(1.0, vec![face(&[2], vec![vec![7]], 1.0)], vec![]))
                .collect();
            let foot: Vec<Vec<usize>> = (0..k).map(|i| vec![i]).collect();
            let rx = relations_exact(&ds, &lat(), &foot, &cfg());
            assert!(rx.complete());
            let want = (1usize << k) - k - 1;
            assert_eq!(rx.rels.len(), want);
            // sum phi_d = k, sum phi_r = 2^k - k - 1
            assert!((rx.sum() - want as f64).abs() < 1e-9);
        }
    }

    fn rand_dists(rng: &mut Rng, nd: usize, a: usize) -> (Vec<Dist>, Vec<Vec<usize>>) {
        let mut ds: Vec<Dist> = Vec::new();
        let mut foot: Vec<Vec<usize>> = Vec::new();
        for _ in 0..nd {
            let phi = 0.5 + 2.5 * rng.f64();
            let mut faces: Vec<Face> = Vec::new();
            for _ in 0..rng.range(1, 3) {
                let size = rng.range(1, 2);
                let mut nodes: Vec<usize> = Vec::new();
                while nodes.len() < size {
                    let j = 2 + rng.below(6);
                    if !nodes.contains(&j) {
                        nodes.push(j);
                    }
                }
                if rng.chance(0.3) {
                    nodes.push(0); // a byte node inside the purview
                }
                nodes.sort_unstable();
                let mut states: Vec<Vec<u16>> = Vec::new();
                for _ in 0..rng.range(1, 2) {
                    let st: Vec<u16> = (0..nodes.len()).map(|_| rng.below(a) as u16).collect();
                    states.push(st);
                }
                states.sort();
                states.dedup();
                faces.push(Face { nodes, phi, states });
            }
            let effect = faces.split_off(faces.len() / 2);
            ds.push(dist(phi, faces, effect));
            foot.push((0..4).filter(|_| rng.chance(0.4)).collect());
        }
        (ds, foot)
    }

    /// The raw definition: every subset, every (face, state) choice per member, no pruning.
    fn brute(ds: &[Dist], lat: &[bool]) -> Vec<(Vec<usize>, f64)> {
        let mut raw: Vec<Vec<(Vec<(usize, u16)>, f64)>> = Vec::new();
        for d in ds {
            let mut v: Vec<(Vec<(usize, u16)>, f64)> = Vec::new();
            if d.is_distinction() {
                for f in d.cause.iter().chain(d.effect.iter()) {
                    for st in &f.states {
                        let cells: Vec<(usize, u16)> = f
                            .nodes
                            .iter()
                            .enumerate()
                            .filter(|&(_, &j)| lat[j])
                            .map(|(i, &j)| (j, st[i]))
                            .collect();
                        if !cells.is_empty() {
                            v.push((cells, d.phi_d / f.nodes.len() as f64));
                        }
                    }
                }
            }
            raw.push(v);
        }
        let n = ds.len();
        let mut out: Vec<(Vec<usize>, f64)> = Vec::new();
        for mask in 1u32..(1u32 << n) {
            let mem: Vec<usize> = (0..n).filter(|&i| (mask >> i) & 1 == 1).collect();
            if mem.len() < 2 || mem.iter().any(|&d| raw[d].is_empty()) {
                continue;
            }
            let mut idx = vec![0usize; mem.len()];
            let mut best = 0.0f64;
            loop {
                let first = &raw[mem[0]][idx[0]];
                let mut cnt = 0usize;
                for &(j, s) in &first.0 {
                    if (1..mem.len()).all(|t| raw[mem[t]][idx[t]].0.contains(&(j, s))) {
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    let u = (0..mem.len())
                        .map(|t| raw[mem[t]][idx[t]].1)
                        .fold(f64::INFINITY, f64::min);
                    best = best.max(cnt as f64 * u);
                }
                let mut t = 0usize;
                while t < mem.len() {
                    idx[t] += 1;
                    if idx[t] < raw[mem[t]].len() {
                        break;
                    }
                    idx[t] = 0;
                    t += 1;
                }
                if t == mem.len() {
                    break;
                }
            }
            if best > 0.0 {
                out.push((mem, best));
            }
        }
        out.sort_by(|x, y| x.0.len().cmp(&y.0.len()).then_with(|| x.0.cmp(&y.0)));
        out
    }

    #[test]
    fn dfs_matches_brute_force_on_random_populations() {
        let l = lat();
        for seed in 0..60u64 {
            let mut rng = Rng::new(0xB10C_0000 + seed);
            let nd = rng.range(2, 6);
            let (ds, foot) = rand_dists(&mut rng, nd, 2 + (seed % 2) as usize);
            let ex = relations_exact(&ds, &l, &foot, &cfg());
            let bf = brute(&ds, &l);
            assert!(ex.complete(), "seed {}", seed);
            assert_eq!(ex.rels.len(), bf.len(), "seed {}", seed);
            for (r, b) in ex.rels.iter().zip(bf.iter()) {
                assert_eq!(r.members, b.0, "seed {}", seed);
                assert!(
                    (r.phi - b.1).abs() < 1e-12,
                    "seed {}: {:?} {} vs {}",
                    seed,
                    r.members,
                    r.phi,
                    b.1
                );
                assert!(!r.overlap.is_empty(), "seed {}", seed);
            }
            let two = relations_exact(&ds, &l, &foot, &cfg().with_rel_order(2));
            let n2 = bf.iter().filter(|b| b.0.len() == 2).count();
            assert_eq!(two.rels.len(), n2, "seed {}", seed);
        }
    }

    #[test]
    fn exact_dominates_the_estimator_and_shares_its_pairs() {
        let l = lat();
        for seed in 0..60u64 {
            let mut rng = Rng::new(0xD0_0000 + seed);
            let nd = rng.range(2, 8);
            let (ds, foot) = rand_dists(&mut rng, nd, 2 + (seed % 2) as usize);
            let ex = relations_exact(&ds, &l, &foot, &cfg());
            assert!(ex.complete(), "seed {}", seed);
            let est = relations(&ds, &l, &foot);
            let g = gap(&ex.rels, &est);
            assert!(g.dominated(), "seed {}: {}", seed, g.line());
            assert!(g.sum_est <= g.sum_exact + 1e-9, "seed {}: {}", seed, g.line());
            // pairs: same members, same numbers, same cross flag
            let mut pe: Vec<&Relation> = ex.rels.iter().filter(|r| r.members.len() == 2).collect();
            let mut ps: Vec<&Relation> = est.iter().filter(|r| r.members.len() == 2).collect();
            pe.sort_by(|a, b| a.members.cmp(&b.members));
            ps.sort_by(|a, b| a.members.cmp(&b.members));
            assert_eq!(pe.len(), ps.len(), "seed {}", seed);
            for (x, y) in pe.iter().zip(ps.iter()) {
                assert_eq!(x.members, y.members, "seed {}", seed);
                assert!((x.phi - y.phi).abs() < 1e-12, "seed {}: {:?}", seed, x.members);
                assert_eq!(x.cross, y.cross, "seed {}", seed);
            }
            // phi_r <= min phi_d, relation by relation (not for the sum)
            for r in &ex.rels {
                let lo = r.members.iter().map(|&d| ds[d].phi_d).fold(f64::INFINITY, f64::min);
                assert!(r.phi <= lo + 1e-9, "seed {}: {:?}", seed, r.members);
            }
        }
    }
}