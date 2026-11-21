//! sharp.rs — phi_sharp(E) (part 4b-2 of 6): structure of a state, phi_sharp(t), maximum over
//! t <= B, frames, unit, `score` (the loss) and `analyze` (what `ask` reads).
//!
//!   phi_sharp(E) = max_{t<=B} sum_{K in Cx(s_t)} [ sum_{d in Dist_E(K)} phi_d + sum_{r in Rel_E(K)} phi_r ]
//!
//! - State: `structure` finds the complexes at the grain tau* (sys.rs), pins V\K to s_t for each one,
//!   keeps the anchored mechanisms (a purview touches a byte of E), sums phi_d over Dist_E (phi_d > 0)
//!   and phi_r over their relations (rel.rs). Half-distinctions (one irreducible side) never enter
//!   the sums; with `Detail::Full` they feed the residual (resid.rs).
//! - Time: the B+1 states of the inference trajectory are scored (a state seen twice once).
//!   phi_sharp(E) is the maximum, t* the first maximiser, the explanation is read at s_{t*}.
//!   "Findable within B" is read as: the plateau is reached before B (t* < B). The area is the mean
//!   of the running maximum over t = 0..=B, which is what the speed-learning loss integrates.
//! - Frames: E is cut into windows of np bytes (the perception nodes). Each frame is analysed
//!   independently, in the A(E) of the whole E, and the frames add up. A relation never links two frames.
//! - Units: everything inside `Step`, `FrameResult` and `Structure` is in bits (log2, as in phi.rs).
//!   `Unit::Symbols` divides by log2|A(E)|; every phi is a log-ratio, so the rescaling is exact.
//!   The top-level numbers of `Analysis` and `Score` are in the requested unit.
//! - The A(E) of the spec is `abstraction`: (sum of phi_r over relations between disjoint footprints
//!   in E) / phi_sharp.
//!
//! Nothing here feeds back into the dynamics: W is only read through `Ctx`, states come from the
//! trajectory. Everything is deterministic and invariant under byte renaming of E.

use crate::model::Model;
use crate::percept::Percept;
use crate::phi::{sum_phi_d, Dir, Dist, Phi, PHI_EPS};
use crate::rel::{relations, Relation};
use crate::resid::{residuals, Residual, ResidualMap};
use crate::sim::Ctx;
use crate::sys::{find_complexes, SharpCfg, System};
use std::collections::HashMap;

/// A new maximum must beat the best so far by this relative margin (first maximiser = t*).
const T_TOL: f64 = 1e-12;

// ---------- units and detail ----------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Bits,
    /// symbols of A(E): bits / log2 |A(E)|
    Symbols,
}

/// Factor turning bits into `unit` for an alphabet of `a` symbols (1 when a < 2: phi is 0 there anyway).
pub fn unit_scale(unit: Unit, a: usize) -> f64 {
    match unit {
        Unit::Bits => 1.0,
        Unit::Symbols => {
            if a >= 2 {
                1.0 / (a as f64).log2()
            } else {
                1.0
            }
        }
    }
}

/// `Lean` computes what phi_sharp needs; `Full` adds the residual and the text of each distinction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    Lean,
    Full,
}

// ---------- structure of one state ----------

/// One complex of a state with its anchored structure (bits).
#[derive(Clone, Debug)]
pub struct ComplexStruct {
    pub sys: System,
    /// anchored mechanisms: Dist_E (phi_d > 0) and half-distinctions (phi_d = 0, residual only)
    pub dists: Vec<Dist>,
    /// footprint in E of each entry of `dists`
    pub foot: Vec<Vec<usize>>,
    pub rels: Vec<Relation>,
    pub sum_d: f64,
    pub sum_r: f64,
    /// part of `sum_r` carried by relations between disjoint footprints
    pub sum_r_cross: f64,
    /// Detail::Full only
    pub resid: Vec<Residual>,
    /// readable form of each entry of `dists` (Detail::Full only)
    pub text: Vec<String>,
}

impl ComplexStruct {
    pub fn phi(&self) -> f64 {
        self.sum_d + self.sum_r
    }
}

/// The structure of one state s_t: its complexes at the grain tau*.
#[derive(Clone, Debug)]
pub struct Structure {
    pub tau: usize,
    pub complexes: Vec<ComplexStruct>,
}

impl Structure {
    pub fn phi(&self) -> f64 {
        self.complexes.iter().map(|c| c.phi()).sum()
    }
    pub fn sum_d(&self) -> f64 {
        self.complexes.iter().map(|c| c.sum_d).sum()
    }
    pub fn sum_r(&self) -> f64 {
        self.complexes.iter().map(|c| c.sum_r).sum()
    }
    pub fn sum_r_cross(&self) -> f64 {
        self.complexes.iter().map(|c| c.sum_r_cross).sum()
    }
}

/// Structure of `state` (one symbol of A(E) per node). Leaves the background of `cx` cleared.
pub fn structure(cx: &mut Ctx<'_>, state: &[u16], cfg: &SharpCfg, detail: Detail) -> Structure {
    let n = cx.n;
    let found = find_complexes(cx, state, cfg);
    let tau = found.tau;
    let latent: Vec<bool> = (0..n).map(|j| !cx.anchored(j)).collect();
    let mut complexes: Vec<ComplexStruct> = Vec::with_capacity(found.systems.len());
    for sys in found.systems {
        let mut inside = vec![false; n];
        for &j in &sys.nodes {
            inside[j] = true;
        }
        let mut bn: Vec<usize> = Vec::new();
        let mut bs: Vec<u16> = Vec::new();
        for j in 0..n {
            if !inside[j] {
                bn.push(j);
                bs.push(state[j]);
            }
        }
        cx.set_background(&bn, &bs);
        let (dists, foot, rels, resid, text) = {
            let ph = Phi::new(&*cx, tau, state, cfg.phi.clone());
            let dists: Vec<Dist> = ph
                .distinctions(true)
                .into_iter()
                .filter(|d| d.anchored)
                .collect();
            let foot: Vec<Vec<usize>> = dists.iter().map(|d| ph.footprint(d)).collect();
            let rels = relations(&dists, &latent, &foot);
            let (resid, text) = match detail {
                Detail::Full => (
                    residuals(&ph, &dists),
                    dists.iter().map(|d| ph.describe(d)).collect::<Vec<String>>(),
                ),
                Detail::Lean => (Vec::new(), Vec::new()),
            };
            (dists, foot, rels, resid, text)
        };
        let sum_d = sum_phi_d(&dists);
        let sum_r: f64 = rels.iter().map(|r| r.phi).sum();
        let sum_r_cross: f64 = rels.iter().filter(|r| r.cross).map(|r| r.phi).sum();
        complexes.push(ComplexStruct { sys, dists, foot, rels, sum_d, sum_r, sum_r_cross, resid, text });
    }
    cx.clear_background();
    Structure { tau, complexes }
}

// ---------- phi_sharp(t) ----------

/// First maximiser of `v`: a later value must exceed the best by a relative T_TOL. 0 when empty.
pub fn first_max(v: &[f64]) -> usize {
    let mut best = -1.0f64;
    let mut at = 0usize;
    for (t, &x) in v.iter().enumerate() {
        if x > best + T_TOL * best.max(1.0) {
            best = x;
            at = t;
        }
    }
    at
}

/// Mean over t of the running maximum of `v`: the area of phi_sharp(t) on [0, B].
pub fn running_area(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut run = v[0];
    let mut s = 0.0f64;
    for &x in v {
        run = run.max(x);
        s += run;
    }
    s / v.len() as f64
}

/// What one state contributes (bits).
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub t: usize,
    pub phi: f64,
    pub sum_d: f64,
    pub sum_r: f64,
    pub sum_r_cross: f64,
    /// grain tau* of this state
    pub tau: usize,
    pub n_cx: usize,
}

struct Scan {
    steps: Vec<Step>,
    traj: Vec<Vec<u16>>,
    t_star: usize,
    area: f64,
}

/// Score the B+1 states of the inference trajectory (Lean). A state seen twice is analysed once.
fn scan(cx: &mut Ctx<'_>, cfg: &SharpCfg) -> Scan {
    let traj = cx.trajectory();
    let mut seen: HashMap<Vec<u16>, usize> = HashMap::new();
    let mut steps: Vec<Step> = Vec::with_capacity(traj.len());
    for (t, s) in traj.iter().enumerate() {
        let known: Option<usize> = seen.get(s).copied();
        let step = match known {
            Some(k) => Step { t, ..steps[k] },
            None => {
                let st = structure(cx, s, cfg, Detail::Lean);
                seen.insert(s.clone(), t);
                Step {
                    t,
                    phi: st.phi(),
                    sum_d: st.sum_d(),
                    sum_r: st.sum_r(),
                    sum_r_cross: st.sum_r_cross(),
                    tau: st.tau,
                    n_cx: st.complexes.len(),
                }
            }
        };
        steps.push(step);
    }
    let phis: Vec<f64> = steps.iter().map(|s| s.phi).collect();
    let t_star = first_max(&phis);
    let area = running_area(&phis);
    Scan { steps, traj, t_star, area }
}

/// One frame of E, analysed over the whole trajectory (bits).
#[derive(Clone, Debug)]
pub struct FrameResult {
    pub frame: usize,
    /// (start, len) of the frame in E
    pub start: usize,
    pub len: usize,
    /// the B+1 states, in order
    pub steps: Vec<Step>,
    /// first maximiser of phi_sharp(t)
    pub t_star: usize,
    /// the plateau is reached before the budget: t* < B
    pub plateau: bool,
    /// s_{t*}: the state the explanation is read from
    pub state: Vec<u16>,
    /// structure of s_{t*} (with the residual and the text under Detail::Full)
    pub st: Structure,
    /// max_t phi_sharp(t) = steps[t*].phi
    pub phi: f64,
    /// mean of the running maximum over t = 0..=B
    pub area: f64,
}

fn frame_result(cx: &mut Ctx<'_>, f: usize, cfg: &SharpCfg, detail: Detail) -> FrameResult {
    let sc = scan(cx, cfg);
    let state = sc.traj[sc.t_star].clone();
    let st = structure(cx, &state, cfg, detail);
    let phi = sc.steps[sc.t_star].phi;
    let budget = cx.m.hy.budget;
    FrameResult {
        frame: f,
        start: cx.frame.0,
        len: cx.frame.1,
        plateau: sc.t_star < budget,
        steps: sc.steps,
        t_star: sc.t_star,
        state,
        st,
        phi,
        area: sc.area,
    }
}

// ---------- whole E ----------

/// Everything `ask` needs. Top-level numbers are in `unit`; the frames keep bits.
#[derive(Clone, Debug)]
pub struct Analysis {
    pub unit: Unit,
    /// bits -> unit
    pub scale: f64,
    /// |A(E)| and |E|
    pub a: usize,
    pub len: usize,
    /// phi_sharp(E): sum over the frames of the maximum over t, in `unit`
    pub phi: f64,
    pub phi_bits: f64,
    /// sum of phi_d and of phi_r at t*, in `unit`
    pub sum_d: f64,
    pub sum_r: f64,
    /// share of phi_sharp carried by relations between disjoint footprints (A(E) of the spec)
    pub abstraction: f64,
    /// area of phi_sharp(t) on [0, B], in `unit`
    pub area: f64,
    /// residual of the whole E at t*, in `unit` (empty under Detail::Lean)
    pub resid: ResidualMap,
    pub frames: Vec<FrameResult>,
}

impl Analysis {
    /// Every frame reached its plateau before B.
    pub fn plateau(&self) -> bool {
        self.frames.iter().all(|f| f.plateau)
    }

    /// Largest t* over the frames.
    pub fn t_star(&self) -> usize {
        self.frames.iter().map(|f| f.t_star).max().unwrap_or(0)
    }
}

/// Loss pieces of one E, in the requested unit (Lean, nothing but the scan).
#[derive(Clone, Copy, Debug)]
pub struct Score {
    /// phi_sharp(E)
    pub phi: f64,
    /// area of phi_sharp(t) on [0, B]
    pub area: f64,
    /// A(E) of the spec
    pub abstraction: f64,
}

/// Where the residual of `st` sits: bytes of E through `Ctx::e_pos`, latent nodes apart.
fn add_residuals(rm: &mut ResidualMap, cx: &Ctx<'_>, st: &Structure) {
    for c in &st.complexes {
        for r in &c.resid {
            match r.weak {
                Dir::Cause => rm.cause += r.rho,
                Dir::Effect => rm.effect += r.rho,
            }
            for &(j, share) in &r.nodes {
                match cx.e_pos(j) {
                    Some(p) if p < rm.by_pos.len() => rm.by_pos[p] += share,
                    _ => rm.latent += share,
                }
            }
        }
    }
}

fn ratio(cross: f64, phi: f64) -> f64 {
    if phi > PHI_EPS { (cross / phi).clamp(0.0, 1.0) } else { 0.0 }
}

/// phi_sharp(E) pieces for the loss. Lean: no residual, no text, no structure at t*.
pub fn score(m: &Model, e: &Percept, unit: Unit) -> Score {
    let cfg = SharpCfg::from_model(m);
    let scale = unit_scale(unit, e.a);
    let (mut phi, mut area, mut cross) = (0.0f64, 0.0f64, 0.0f64);
    for f in 0..Ctx::n_frames(m, e) {
        let mut cx = Ctx::new(m, e, f);
        let sc = scan(&mut cx, &cfg);
        phi += sc.steps[sc.t_star].phi;
        cross += sc.steps[sc.t_star].sum_r_cross;
        area += sc.area;
    }
    Score { phi: phi * scale, area: area * scale, abstraction: ratio(cross, phi) }
}

/// Full reading of E. `draws`: R for this reading (None: the model's R; raise it for a final reading).
pub fn analyze(m: &Model, e: &Percept, detail: Detail, unit: Unit, draws: Option<usize>) -> Analysis {
    let cfg = SharpCfg::from_model(m);
    let scale = unit_scale(unit, e.a);
    let nf = Ctx::n_frames(m, e);
    let mut frames: Vec<FrameResult> = Vec::with_capacity(nf);
    let mut rm = ResidualMap { by_pos: vec![0.0; e.len()], latent: 0.0, cause: 0.0, effect: 0.0 };
    for f in 0..nf {
        let mut cx = Ctx::new(m, e, f);
        if let Some(r) = draws {
            cx.set_draws(r);
        }
        let fr = frame_result(&mut cx, f, &cfg, detail);
        add_residuals(&mut rm, &cx, &fr.st);
        frames.push(fr);
    }
    let (mut phi_bits, mut sum_d, mut sum_r, mut cross, mut area) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for fr in &frames {
        let s = &fr.steps[fr.t_star];
        phi_bits += s.phi;
        sum_d += s.sum_d;
        sum_r += s.sum_r;
        cross += s.sum_r_cross;
        area += fr.area;
    }
    rm.cause *= scale;
    rm.effect *= scale;
    rm.latent *= scale;
    for v in rm.by_pos.iter_mut() {
        *v *= scale;
    }
    Analysis {
        unit,
        scale,
        a: e.a,
        len: e.len(),
        phi: phi_bits * scale,
        phi_bits,
        sum_d: sum_d * scale,
        sum_r: sum_r * scale,
        abstraction: ratio(cross, phi_bits),
        area: area * scale,
        resid: rm,
        frames,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Hyper;
    use crate::percept::{random_renaming, rename};
    use crate::util::Rng;

    fn small_model(np: usize, nl: usize, seed: u64) -> Model {
        let mut rng = Rng::new(seed);
        let mut hy = Hyper::default();
        hy.budget = 4;
        hy.draws = 4;
        Model::random(np, nl, hy, &mut rng)
    }

    /// Node 0 is a perception node anchored to the byte of the frame; node 1 is latent.
    /// w01 is the connection 0 -> 1, w10 is 1 -> 0 (set directly).
    fn two_node(w01: f64, w10: f64, hold: f64) -> Model {
        let mut m = small_model(1, 1, 1);
        m.w = vec![0.0, w01, w10, 0.0];
        m.hold = hold;
        m.beta = 1.0;
        m
    }

    #[test]
    fn unit_conversion() {
        assert_eq!(unit_scale(Unit::Bits, 7), 1.0);
        assert!((unit_scale(Unit::Symbols, 4) - 0.5).abs() < 1e-15);
        assert!((unit_scale(Unit::Symbols, 2) - 1.0).abs() < 1e-15);
        assert_eq!(unit_scale(Unit::Symbols, 1), 1.0);
        assert_eq!(unit_scale(Unit::Symbols, 0), 1.0);
    }

    #[test]
    fn first_maximiser_and_area() {
        assert_eq!(first_max(&[0.0, 1.0, 3.0, 3.0, 2.0]), 2);
        assert_eq!(first_max(&[0.0, 0.0, 0.0]), 0);
        assert_eq!(first_max(&[1.0, 1.0 + 1e-14]), 0);
        assert_eq!(first_max(&[1.0, 2.0, 3.0]), 2);
        assert_eq!(first_max(&[]), 0);
        assert!((running_area(&[0.0, 1.0, 3.0, 3.0, 2.0]) - 2.0).abs() < 1e-12);
        assert_eq!(running_area(&[]), 0.0);
    }

    #[test]
    fn feedforward_graph_explains_nothing() {
        let m = two_node(1.0, 0.0, 2.0);
        let e = Percept::from_bytes(b"ab");
        let an = analyze(&m, &e, Detail::Full, Unit::Bits, None);
        assert_eq!(an.phi_bits, 0.0);
        assert_eq!(an.phi, 0.0);
        assert_eq!(an.abstraction, 0.0);
        assert!(an.frames.iter().all(|f| f.t_star == 0));
        assert!(an.frames.iter().all(|f| f.steps.iter().all(|s| s.n_cx == 0)));
    }

    #[test]
    fn an_attractor_floating_outside_e_scores_zero() {
        let mut m = small_model(2, 2, 3);
        // perception nodes 0, 1 isolated; latent cycle 2 -> 3 -> 2 (w[i*4+j] is i -> j)
        m.w = vec![
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.5,
            0.0, 0.0, -1.0, 0.0,
        ];
        m.hold = 2.0;
        m.beta = 1.0;
        let e = Percept::from_bytes(b"abab");
        let an = analyze(&m, &e, Detail::Full, Unit::Bits, None);
        // the attractor is a complex (phi_s > 0) in every state, but nothing in it touches E
        assert!(an.frames.iter().all(|f| f.steps.iter().all(|s| s.n_cx >= 1)));
        assert_eq!(an.phi_bits, 0.0);
    }

    #[test]
    fn a_loop_through_e_has_positive_phi_sharp() {
        let m = two_node(1.0, -0.5, 2.0);
        let e = Percept::from_bytes(b"ab");
        let an = analyze(&m, &e, Detail::Full, Unit::Bits, None);
        assert_eq!(an.frames.len(), 2);
        assert!(an.phi_bits > 0.0);
        for fr in &an.frames {
            assert!(fr.phi > 0.0);
            assert!(fr.st.complexes.iter().any(|c| c.dists.iter().any(|d| d.anchored && d.is_distinction())));
        }
    }

    #[test]
    fn invariants_on_a_random_model() {
        let m = small_model(3, 2, 11);
        let e = Percept::from_bytes(b"abcabca");
        let an = analyze(&m, &e, Detail::Full, Unit::Bits, None);
        assert_eq!(an.frames.len(), 3); // 7 bytes, frames of 3
        let mut tot = 0.0f64;
        for fr in &an.frames {
            assert_eq!(fr.steps.len(), m.hy.budget + 1);
            let top = fr.steps.iter().map(|s| s.phi).fold(0.0f64, f64::max);
            assert!((fr.phi - top).abs() <= 1e-9 * top.max(1.0));
            assert!(fr.steps[..fr.t_star].iter().all(|s| s.phi <= fr.phi * (1.0 + 1e-9) + 1e-12));
            assert!(fr.area >= 0.0 && fr.area <= fr.phi + 1e-12);
            // Full re-analysis at t* gives the same phi as the Lean scan
            assert!((fr.st.phi() - fr.phi).abs() <= 1e-9 * fr.phi.max(1.0));
            tot += fr.phi;
            for c in &fr.st.complexes {
                assert_eq!(c.text.len(), c.dists.len());
                assert_eq!(c.foot.len(), c.dists.len());
                assert!((c.sum_d - sum_phi_d(&c.dists)).abs() < 1e-12);
                assert!((c.phi() - c.sum_d - c.sum_r).abs() < 1e-12);
                assert!(c.sum_r_cross <= c.sum_r + 1e-12);
                for r in &c.rels {
                    let lo = r.members.iter().map(|&i| c.dists[i].phi_d).fold(f64::INFINITY, f64::min);
                    assert!(r.phi <= lo + 1e-9, "phi_r above min phi_d");
                }
                for f in &c.foot {
                    assert!(f.iter().all(|&p| p < e.len()));
                }
            }
        }
        assert!((tot - an.phi_bits).abs() < 1e-9);
        assert!(an.abstraction >= 0.0 && an.abstraction <= 1.0);
        assert!(an.area >= 0.0 && an.area <= an.phi_bits + 1e-12);
        // the shares of the residual add up: bytes of E + latent nodes = rho_cause + rho_effet
        let rsum: f64 = an.resid.by_pos.iter().sum::<f64>() + an.resid.latent;
        assert!((rsum - an.resid.total()).abs() < 1e-9);
        // deterministic
        let again = analyze(&m, &e, Detail::Lean, Unit::Bits, None);
        assert_eq!(an.phi_bits.to_bits(), again.phi_bits.to_bits());
    }

    #[test]
    fn analysis_is_renaming_invariant() {
        let m = small_model(3, 2, 5);
        let mut rng = Rng::new(77);
        let perm = random_renaming(&mut rng);
        let raw = b"abcabcab";
        let e1 = Percept::from_bytes(raw);
        let e2 = Percept::from_bytes(&rename(raw, &perm));
        let a1 = analyze(&m, &e1, Detail::Full, Unit::Bits, None);
        let a2 = analyze(&m, &e2, Detail::Full, Unit::Bits, None);
        assert_eq!(a1.phi_bits.to_bits(), a2.phi_bits.to_bits());
        assert_eq!(a1.abstraction.to_bits(), a2.abstraction.to_bits());
        assert_eq!(a1.area.to_bits(), a2.area.to_bits());
        for (x, y) in a1.frames.iter().zip(a2.frames.iter()) {
            assert_eq!(x.t_star, y.t_star);
            assert_eq!(x.state, y.state);
        }
        let b1: Vec<u64> = a1.resid.by_pos.iter().map(|v| v.to_bits()).collect();
        let b2: Vec<u64> = a2.resid.by_pos.iter().map(|v| v.to_bits()).collect();
        assert_eq!(b1, b2);
    }

    #[test]
    fn symbols_rescale_bits_exactly_and_score_matches_analyze() {
        let m = small_model(3, 2, 9);
        let e = Percept::from_bytes(b"abcabc"); // |A(E)| = 3
        let ab = analyze(&m, &e, Detail::Lean, Unit::Bits, None);
        let asy = analyze(&m, &e, Detail::Lean, Unit::Symbols, None);
        let k = 3.0f64.log2();
        assert!((asy.phi * k - ab.phi_bits).abs() < 1e-9);
        assert!((asy.area * k - ab.area).abs() < 1e-9);
        let sc = score(&m, &e, Unit::Symbols);
        assert!((sc.phi - asy.phi).abs() < 1e-12);
        assert!((sc.area - asy.area).abs() < 1e-12);
        assert!((sc.abstraction - asy.abstraction).abs() < 1e-12);
    }

    #[test]
    fn degenerate_percepts_score_zero() {
        let m = small_model(3, 2, 2);
        for raw in [&b""[..], &b"aaaaaaa"[..], &b"z"[..]] {
            let e = Percept::from_bytes(raw);
            let an = analyze(&m, &e, Detail::Full, Unit::Symbols, None);
            assert_eq!(an.phi_bits, 0.0);
            assert_eq!(an.abstraction, 0.0);
            assert_eq!(an.resid.by_pos.len(), raw.len());
        }
    }
}