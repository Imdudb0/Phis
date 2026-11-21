use crate::model::Model;
use crate::percept::Percept;
use crate::phi::{sum_phi_d, Dir, Dist, Phi, PHI_EPS};
use crate::rel::{relations, Relation};
use crate::resid::{residuals, Residual, ResidualMap};
use crate::sim::Ctx;
use crate::sys::{find_complexes, SharpCfg, System};
use std::collections::HashMap;
const T_TOL: f64 = 1e-12;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Bits,
    Symbols,
}
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail {
    Lean,
    Full,
}
#[derive(Clone, Debug)]
pub struct ComplexStruct {
    pub sys: System,
    pub dists: Vec<Dist>,
    pub foot: Vec<Vec<usize>>,
    pub rels: Vec<Relation>,
    pub sum_d: f64,
    pub sum_r: f64,
    pub sum_r_cross: f64,
    pub resid: Vec<Residual>,
    pub text: Vec<String>,
}
impl ComplexStruct {
    pub fn phi(&self) -> f64 {
        self.sum_d + self.sum_r
    }
}
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
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub t: usize,
    pub phi: f64,
    pub sum_d: f64,
    pub sum_r: f64,
    pub sum_r_cross: f64,
    pub tau: usize,
    pub n_cx: usize,
}
struct Scan {
    steps: Vec<Step>,
    traj: Vec<Vec<u16>>,
    t_star: usize,
    area: f64,
}
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
#[derive(Clone, Debug)]
pub struct FrameResult {
    pub frame: usize,
    pub start: usize,
    pub len: usize,
    pub steps: Vec<Step>,
    pub t_star: usize,
    pub plateau: bool,
    pub state: Vec<u16>,
    pub st: Structure,
    pub phi: f64,
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
#[derive(Clone, Debug)]
pub struct Analysis {
    pub unit: Unit,
    pub scale: f64,
    pub a: usize,
    pub len: usize,
    pub phi: f64,
    pub phi_bits: f64,
    pub sum_d: f64,
    pub sum_r: f64,
    pub abstraction: f64,
    pub area: f64,
    pub resid: ResidualMap,
    pub frames: Vec<FrameResult>,
}
impl Analysis {
    pub fn plateau(&self) -> bool {
        self.frames.iter().all(|f| f.plateau)
    }
    pub fn t_star(&self) -> usize {
        self.frames.iter().map(|f| f.t_star).max().unwrap_or(0)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Score {
    pub phi: f64,
    pub area: f64,
    pub abstraction: f64,
}
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
