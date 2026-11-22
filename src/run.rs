use super::{
    find_complexes_exact, relations_exact, residuals_exact, tractable, Backend, Dx, ExactCfg,
};
use crate::phi::find_complexes_exact;
use crate::model::Model;
use crate::percept::Percept;
use crate::phi::{sum_phi_d, Dir, Dist, PHI_EPS};
use crate::render::{one_line, render, RenderCfg};
use crate::resid::ResidualMap;
use crate::sharp::{
    first_max, running_area, unit_scale, Analysis, ComplexStruct, Detail, FrameResult, Score, Step,
    Structure, Unit,
};
use crate::sim::Ctx;
use std::collections::HashMap;

const LEGEND_EXACT: &str = "  exact reference: repertoires by full enumeration, minima over every cut and partition; mechanisms and purviews limited only by kmax, qmax (header)";
const LEGEND_MC: &str = "  exact enumeration of cuts, partitions and relations over Monte-Carlo repertoires (R draws): only the sampling noise remains";

// ------------------------------------------------------------------- flags

/// What the estimator's types cannot say about a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flags {
    /// every repertoire came from the requested backend (no fallback to Monte-Carlo)
    pub ok: bool,
    /// no relation list was cut by `max_rel` or `rel_order` (otherwise sum phi_r is a lower bound)
    pub rel_complete: bool,
}

impl Flags {
    pub const CLEAN: Flags = Flags { ok: true, rel_complete: true };

    pub fn and(self, o: Flags) -> Flags {
        Flags { ok: self.ok && o.ok, rel_complete: self.rel_complete && o.rel_complete }
    }

    pub fn clean(self) -> bool {
        self.ok && self.rel_complete
    }
}

// --------------------------------------------------------------- structure

/// A `sharp::Structure` and what is known about how it was obtained.
#[derive(Clone, Debug)]
pub struct StructX {
    pub st: Structure,
    pub flags: Flags,
}

/// `sharp::structure` without its restrictions, for one state. The Ctx background is set to
/// the complement of each complex while its distinctions are enumerated, and cleared at the end.
pub fn structure_exact(
    cx: &mut Ctx<'_>,
    state: &[u16],
    cfg: &ExactCfg,
    detail: Detail,
) -> StructX {
    let n = cx.n;
    let found = find_complexes_exact(cx, state, cfg);
    let tau = found.tau;
    let latent: Vec<bool> = (0..n).map(|j| !cx.anchored(j)).collect();
    let mut flags = Flags { ok: found.ok, rel_complete: true };
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
        let (dists, foot, rx, resid, text, inexact) = {
            let dx = Dx::new(&*cx, tau, state, cfg);
            let dists: Vec<Dist> = dx.distinctions().into_iter().filter(|d| d.anchored).collect();
            let foot: Vec<Vec<usize>> = dists.iter().map(|d| dx.footprint(d)).collect();
            let rx = relations_exact(&dists, &latent, &foot, cfg);
            let (resid, text) = match detail {
                Detail::Full => (
                    residuals_exact(&dx, &dists),
                    dists.iter().map(|d| dx.describe(d)).collect::<Vec<String>>(),
                ),
                Detail::Lean => (Vec::new(), Vec::new()),
            };
            (dists, foot, rx, resid, text, dx.inexact())
        };
        flags.ok &= !inexact;
        flags.rel_complete &= rx.complete();
        let sum_d = sum_phi_d(&dists);
        let sum_r = rx.sum();
        let sum_r_cross = rx.sum_cross();
        complexes.push(ComplexStruct {
            sys,
            dists,
            foot,
            rels: rx.rels,
            sum_d,
            sum_r,
            sum_r_cross,
            resid,
            text,
        });
    }
    cx.clear_background();
    StructX { st: Structure { tau, complexes }, flags }
}

// -------------------------------------------------------------------- scan

/// phi_sharp(t) along the trajectory of one frame.
#[derive(Clone, Debug)]
pub struct ScanX {
    pub steps: Vec<Step>,
    pub traj: Vec<Vec<u16>>,
    pub t_star: usize,
    pub area: f64,
    pub flags: Flags,
}

/// `sharp::scan`: one structure per distinct state of `Ctx::trajectory` (a repeated state
/// reuses the first evaluation), t* = first maximum, area = mean running maximum.
pub fn scan_exact(cx: &mut Ctx<'_>, cfg: &ExactCfg) -> ScanX {
    let traj = cx.trajectory();
    let mut seen: HashMap<Vec<u16>, usize> = HashMap::new();
    let mut steps: Vec<Step> = Vec::with_capacity(traj.len());
    let mut flags = Flags::CLEAN;
    for (t, s) in traj.iter().enumerate() {
        let known: Option<usize> = seen.get(s).copied();
        let step = match known {
            Some(k) => Step { t, ..steps[k] },
            None => {
                let sx = structure_exact(cx, s, cfg, Detail::Lean);
                flags = flags.and(sx.flags);
                seen.insert(s.clone(), t);
                Step {
                    t,
                    phi: sx.st.phi(),
                    sum_d: sx.st.sum_d(),
                    sum_r: sx.st.sum_r(),
                    sum_r_cross: sx.st.sum_r_cross(),
                    tau: sx.st.tau,
                    n_cx: sx.st.complexes.len(),
                }
            }
        };
        steps.push(step);
    }
    let phis: Vec<f64> = steps.iter().map(|s| s.phi).collect();
    let t_star = first_max(&phis);
    let area = running_area(&phis);
    ScanX { steps, traj, t_star, area, flags }
}

fn frame_exact(
    cx: &mut Ctx<'_>,
    f: usize,
    cfg: &ExactCfg,
    detail: Detail,
) -> (FrameResult, Flags) {
    let sc = scan_exact(cx, cfg);
    let state = sc.traj[sc.t_star].clone();
    let sx = structure_exact(cx, &state, cfg, detail);
    let flags = sc.flags.and(sx.flags);
    let phi = sc.steps[sc.t_star].phi;
    let budget = cx.m.hy.budget;
    let fr = FrameResult {
        frame: f,
        start: cx.frame.0,
        len: cx.frame.1,
        plateau: sc.t_star < budget,
        steps: sc.steps,
        t_star: sc.t_star,
        state,
        st: sx.st,
        phi,
        area: sc.area,
    };
    (fr, flags)
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

// ---------------------------------------------------------------- analysis

/// A `sharp::Analysis` computed by the exact reference, with its provenance.
#[derive(Clone, Debug)]
pub struct AnalysisX {
    pub an: Analysis,
    pub backend: Backend,
    pub cfg: ExactCfg,
    /// the model's budget B (steps of dynamics)
    pub budget: usize,
    /// all frames together
    pub flags: Flags,
    /// one entry per frame
    pub frame_flags: Vec<Flags>,
}

impl AnalysisX {
    /// The value is the exact one: exact repertoires, nothing fell back, no relation list cut.
    pub fn is_exact(&self) -> bool {
        self.backend == Backend::Exact && self.flags.clean()
    }

    pub fn verdict(&self) -> String {
        if self.is_exact() {
            return "EXACT: no repertoire fell back to Monte-Carlo, no relation list was cut"
                .to_string();
        }
        let mut why: Vec<&str> = Vec::new();
        if self.backend == Backend::Mc {
            why.push("Monte-Carlo repertoires");
        }
        if !self.flags.ok {
            why.push("some repertoire fell back to Monte-Carlo");
        }
        if !self.flags.rel_complete {
            why.push("a relation list was cut by max_rel or rel_order: sum phi_r is a lower bound");
        }
        format!("NOT EXACT: {}", why.join("; "))
    }

    pub fn summary(&self) -> String {
        format!("{}  [{}]", one_line(&self.an), if self.is_exact() { "exact" } else { "not exact" })
    }
}

/// `sharp::analyze` without its restrictions. The Monte-Carlo draws of `cfg` only matter for
/// `Backend::Mc`. Refuses (Err) what `tractable` refuses.
pub fn analyze_exact(
    m: &Model,
    e: &Percept,
    detail: Detail,
    unit: Unit,
    cfg: &ExactCfg,
) -> Result<AnalysisX, String> {
    tractable(m, e, cfg)?;
    let scale = unit_scale(unit, e.a);
    let nf = Ctx::n_frames(m, e);
    let mut frames: Vec<FrameResult> = Vec::with_capacity(nf);
    let mut per: Vec<Flags> = Vec::with_capacity(nf);
    let mut all = Flags::CLEAN;
    let mut rm = ResidualMap { by_pos: vec![0.0; e.len()], latent: 0.0, cause: 0.0, effect: 0.0 };
    for f in 0..nf {
        let mut cx = Ctx::new(m, e, f);
        if let Some(r) = cfg.draws {
            cx.set_draws(r);
        }
        let (fr, fl) = frame_exact(&mut cx, f, cfg, detail);
        add_residuals(&mut rm, &cx, &fr.st);
        all = all.and(fl);
        per.push(fl);
        frames.push(fr);
    }
    let (mut phi_bits, mut sum_d, mut sum_r, mut cross, mut area) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
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
    let an = Analysis {
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
    };
    Ok(AnalysisX {
        an,
        backend: cfg.backend,
        cfg: cfg.clone(),
        budget: m.hy.budget,
        flags: all,
        frame_flags: per,
    })
}

// ------------------------------------------------------------------- score

#[derive(Clone, Copy, Debug)]
pub struct ScoreX {
    pub score: Score,
    pub flags: Flags,
}

/// `sharp::score` without its restrictions: no final structure, no residual.
pub fn score_exact(m: &Model, e: &Percept, unit: Unit, cfg: &ExactCfg) -> Result<ScoreX, String> {
    tractable(m, e, cfg)?;
    let scale = unit_scale(unit, e.a);
    let (mut phi, mut area, mut cross) = (0.0f64, 0.0f64, 0.0f64);
    let mut flags = Flags::CLEAN;
    for f in 0..Ctx::n_frames(m, e) {
        let mut cx = Ctx::new(m, e, f);
        if let Some(r) = cfg.draws {
            cx.set_draws(r);
        }
        let sc = scan_exact(&mut cx, cfg);
        phi += sc.steps[sc.t_star].phi;
        cross += sc.steps[sc.t_star].sum_r_cross;
        area += sc.area;
        flags = flags.and(sc.flags);
    }
    let score = Score { phi: phi * scale, area: area * scale, abstraction: ratio(cross, phi) };
    Ok(ScoreX { score, flags })
}

// ------------------------------------------------------------------ report

fn banner(ax: &AnalysisX, m: &Model) -> String {
    let c = &ax.cfg;
    let word = |k: usize| if k == 0 { "all".to_string() } else { k.to_string() };
    let order = if c.rel_order == 0 { "all".to_string() } else { c.rel_order.max(2).to_string() };
    let mut s = String::new();
    s.push_str("phisharp exact reference (part 5b)\n");
    match ax.backend {
        Backend::Exact => s.push_str(
            "  repertoires  : exact, every joint state of the complex enumerated (the model's R is not used)\n",
        ),
        Backend::Mc => s.push_str(&format!(
            "  repertoires  : Monte-Carlo, R = {} draws (the estimator's own): sampling noise only\n",
            c.draws.unwrap_or(m.hy.draws)
        )),
    }
    s.push_str(&format!(
        "  enumerated   : every node subset as candidate complex (grains 1..={}); mechanisms of up to {} node(s); purviews of up to {}; every disintegrating partition; every directed cut; relations of order up to {}\n",
        c.grain_max,
        word(c.kmax),
        word(c.qmax),
        order
    ));
    s.push_str(
        "  as estimator : the states s_t (one sample path fixed by the seed); phi_s reads the effect side only\n",
    );
    s.push_str(&format!("  verdict      : {}\n", ax.verdict()));
    let bad: Vec<String> = ax
        .frame_flags
        .iter()
        .enumerate()
        .filter(|p| !p.1.clean())
        .map(|p| p.0.to_string())
        .collect();
    if !bad.is_empty() {
        s.push_str(&format!("  flagged frames: {}\n", bad.join(" ")));
    }
    s.push('\n');
    s
}

/// The standard report of `render::render` under a banner that says what was enumerated and
/// whether the number is the exact one. The estimator's legend line is replaced.
pub fn render_exact(ax: &AnalysisX, m: &Model, e: &Percept, rc: &RenderCfg) -> String {
    let mut cfg = rc.clone();
    cfg.draws = match ax.backend {
        Backend::Mc => ax.cfg.draws,
        Backend::Exact => None,
    };
    let legend = match ax.backend {
        Backend::Exact => LEGEND_EXACT,
        Backend::Mc => LEGEND_MC,
    };
    let mut out = banner(ax, m);
    let body = render(&ax.an, m, e, &cfg);
    for line in body.lines() {
        if line.trim_start().starts_with("estimator, not exact") {
            out.push_str(legend);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

// ------------------------------------------------------------------ verify

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

fn disjoint_feet(foot: &[Vec<usize>], mem: &[usize]) -> bool {
    for p in 0..mem.len() {
        for q in (p + 1)..mem.len() {
            if foot[mem[p]].iter().any(|x| foot[mem[q]].binary_search(x).is_ok()) {
                return false;
            }
        }
    }
    true
}

fn check_complex(fr: &FrameResult, ci: usize, c: &ComplexStruct) -> Result<(), String> {
    let tag = format!("frame {} complex {}", fr.frame, ci);
    let nd = c.dists.len();
    if c.foot.len() != nd {
        return Err(format!("{}: {} footprints for {} distinctions", tag, c.foot.len(), nd));
    }
    let hi = fr.start + fr.len;
    for (i, d) in c.dists.iter().enumerate() {
        if d.phi_c < 0.0 || d.phi_e < 0.0 || !near(d.phi_d, d.phi_c.min(d.phi_e)) {
            return Err(format!(
                "{}: d{} has phi_d {} but phi_c {} and phi_e {}",
                tag, i, d.phi_d, d.phi_c, d.phi_e
            ));
        }
        let f = &c.foot[i];
        let sorted = f.windows(2).all(|w| w[0] < w[1]);
        if !d.anchored || f.is_empty() || !sorted || f.iter().any(|&p| p < fr.start || p >= hi) {
            return Err(format!("{}: d{} is not anchored in E (footprint {:?})", tag, i, f));
        }
    }
    let sd: f64 = c.dists.iter().filter(|d| d.anchored && d.is_distinction()).map(|d| d.phi_d).sum();
    let sr: f64 = c.rels.iter().map(|r| r.phi).sum();
    let sx: f64 = c.rels.iter().filter(|r| r.cross).map(|r| r.phi).sum();
    if !near(sd, c.sum_d) || !near(sr, c.sum_r) || !near(sx, c.sum_r_cross) {
        return Err(format!("{}: sums do not match their parts", tag));
    }
    for (ri, r) in c.rels.iter().enumerate() {
        let sorted = r.members.windows(2).all(|w| w[0] < w[1]);
        if r.members.len() < 2
            || !sorted
            || r.members.iter().any(|&k| k >= nd || !c.dists[k].is_distinction())
        {
            return Err(format!("{}: r{} has bad members {:?}", tag, ri, r.members));
        }
        if r.overlap.is_empty() || r.phi <= 0.0 {
            return Err(format!("{}: r{} has an empty overlap or no weight", tag, ri));
        }
        // phi_r <= min phi_d, relation by relation (the spec's "borné par phi_d")
        let lo = r.members.iter().map(|&k| c.dists[k].phi_d).fold(f64::INFINITY, f64::min);
        if r.phi > lo + 1e-9 * lo.max(1.0) {
            return Err(format!("{}: r{} has phi_r {} above min phi_d {}", tag, ri, r.phi, lo));
        }
        if disjoint_feet(&c.foot, &r.members) != r.cross {
            return Err(format!("{}: r{} cross flag disagrees with the footprints", tag, ri));
        }
    }
    for r in &c.resid {
        let d = match c.dists.iter().find(|d| d.mech == r.mech) {
            Some(d) => d,
            None => return Err(format!("{}: residual of an unknown mechanism {:?}", tag, r.mech)),
        };
        let rho = (d.phi_c - d.phi_e).abs();
        let tot: f64 = r.nodes.iter().map(|x| x.1).sum();
        if !near(rho, r.rho) || !near(tot, r.rho) || r.nodes.iter().any(|x| x.1 < 0.0) {
            return Err(format!(
                "{}: residual of m{:?}: rho {} (expected {}), shares sum to {}",
                tag, r.mech, r.rho, rho, tot
            ));
        }
        if (r.weak == Dir::Cause) != (d.phi_c < d.phi_e) {
            return Err(format!("{}: residual of m{:?} names the wrong weak side", tag, r.mech));
        }
        // filling the weak side gains at most rho(d), the strong side unchanged
        if !near(d.phi_d + rho, d.phi_c.max(d.phi_e)) {
            return Err(format!("{}: residual of m{:?} is not bounded by rho", tag, r.mech));
        }
    }
    Ok(())
}

/// The spec's guarantees, checked on one analysis. Err names the first violation.
/// Checks: phi_d = min(phi_c, phi_e); phi_r <= min phi_d per relation; cross = pairwise
/// disjoint footprints; rho(d) = |phi_c - phi_e| localised on nodes and bounded; t* = first
/// maximum and the plateau flag; frames contiguous and covering E; A(E) in [0, 1]; the sums.
pub fn verify(ax: &AnalysisX) -> Result<(), String> {
    let an = &ax.an;
    let mut at = 0usize;
    let mut phi_bits = 0.0f64;
    let mut rho_sum = 0.0f64;
    for fr in &an.frames {
        let tag = format!("frame {}", fr.frame);
        if fr.start != at {
            return Err(format!("{}: starts at {}, expected {}", tag, fr.start, at));
        }
        at += fr.len;
        if fr.steps.len() != ax.budget + 1 || fr.t_star > ax.budget {
            return Err(format!(
                "{}: {} steps, t* {}, B {}",
                tag,
                fr.steps.len(),
                fr.t_star,
                ax.budget
            ));
        }
        for (t, s) in fr.steps.iter().enumerate() {
            if s.t != t {
                return Err(format!("{}: step {} is labelled t = {}", tag, t, s.t));
            }
        }
        let phis: Vec<f64> = fr.steps.iter().map(|s| s.phi).collect();
        if first_max(&phis) != fr.t_star {
            return Err(format!("{}: t* {} is not the first maximum", tag, fr.t_star));
        }
        if fr.plateau != (fr.t_star < ax.budget) {
            return Err(format!("{}: plateau flag disagrees with t* = {}", tag, fr.t_star));
        }
        if !near(fr.phi, phis[fr.t_star])
            || !near(fr.st.phi(), fr.phi)
            || !near(fr.area, running_area(&phis))
        {
            return Err(format!("{}: phi or area do not match the scan", tag));
        }
        phi_bits += fr.phi;
        for (ci, c) in fr.st.complexes.iter().enumerate() {
            check_complex(fr, ci, c)?;
            for r in &c.resid {
                rho_sum += r.rho;
            }
        }
    }
    if at != an.len {
        return Err(format!("frames cover {} bytes of {}", at, an.len));
    }
    if !near(an.phi_bits, phi_bits)
        || !near(an.phi, an.phi_bits * an.scale)
        || !near(an.phi, an.sum_d + an.sum_r)
    {
        return Err("phi_sharp does not match its frames or its parts".to_string());
    }
    if !(0.0..=1.0).contains(&an.abstraction) {
        return Err(format!("A(E) = {} is outside [0, 1]", an.abstraction));
    }
    let rm = &an.resid;
    let on: f64 = rm.by_pos.iter().sum();
    if !near(rho_sum * an.scale, rm.total()) || !near(on + rm.latent, rm.total()) {
        return Err("residual map does not add up to the sum of rho".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Hyper;
    use crate::util::Rng;

    fn small(seed: u64, np: usize, nl: usize) -> Model {
        let hy = Hyper { budget: 4, grain_max: 1, ..Hyper::default() };
        Model::random(np, nl, hy, &mut Rng::new(seed))
    }

    fn analyse(m: &Model, e: &Percept, detail: Detail) -> AnalysisX {
        analyze_exact(m, e, detail, Unit::Bits, &ExactCfg::from_model(m)).expect("tractable")
    }

    #[test]
    fn acyclic_model_scores_zero() {
        let mut m = small(5, 3, 2);
        let n = m.n();
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    m.w[i * n + j] = 0.0;
                }
            }
        }
        let e = Percept::from_bytes(b"aba");
        let cfg = ExactCfg::from_model(&m);
        let ax = analyze_exact(&m, &e, Detail::Full, Unit::Bits, &cfg).expect("tractable");
        assert!(ax.is_exact());
        assert_eq!(ax.an.phi, 0.0);
        assert_eq!(ax.an.abstraction, 0.0);
        assert!(ax.an.frames.iter().all(|f| f.st.complexes.is_empty()));
        verify(&ax).unwrap();
        let s = score_exact(&m, &e, Unit::Bits, &cfg).expect("tractable");
        assert_eq!(s.score.phi, 0.0);
        assert!(s.flags.clean());
    }

    #[test]
    fn lean_and_full_agree_on_the_value() {
        let m = small(11, 3, 2);
        let e = Percept::from_bytes(b"abb");
        let lean = analyse(&m, &e, Detail::Lean);
        let full = analyse(&m, &e, Detail::Full);
        assert!(lean.is_exact() && full.is_exact());
        assert!(near(lean.an.phi, full.an.phi));
        assert_eq!(lean.an.frames[0].t_star, full.an.frames[0].t_star);
        assert_eq!(lean.an.resid.total(), 0.0);
        for fr in &full.an.frames {
            for c in &fr.st.complexes {
                assert_eq!(c.text.len(), c.dists.len());
            }
        }
        verify(&lean).unwrap();
        verify(&full).unwrap();
    }

    #[test]
    fn score_is_the_analysis_value() {
        let m = small(7, 3, 2);
        let e = Percept::from_bytes(b"abc");
        let cfg = ExactCfg::from_model(&m);
        let a = analyze_exact(&m, &e, Detail::Lean, Unit::Symbols, &cfg).unwrap();
        let s = score_exact(&m, &e, Unit::Symbols, &cfg).unwrap();
        assert!(near(s.score.phi, a.an.phi));
        assert!(near(s.score.area, a.an.area));
        assert!(near(s.score.abstraction, a.an.abstraction));
        assert_eq!(s.flags, a.flags);
    }

    #[test]
    fn two_frames_are_consistent() {
        let m = small(13, 3, 2);
        let e = Percept::from_bytes(b"abbaab");
        let ax = analyse(&m, &e, Detail::Full);
        assert_eq!(ax.an.frames.len(), 2);
        assert_eq!(ax.frame_flags.len(), 2);
        let sum: f64 = ax.an.frames.iter().map(|f| f.phi).sum();
        assert!(near(ax.an.phi_bits, sum));
        verify(&ax).unwrap();
    }

    #[test]
    fn a_run_is_reproducible() {
        let m = small(17, 3, 2);
        let e = Percept::from_bytes(b"abbab");
        let r = RenderCfg::default();
        let x = render_exact(&analyse(&m, &e, Detail::Full), &m, &e, &r);
        let y = render_exact(&analyse(&m, &e, Detail::Full), &m, &e, &r);
        assert_eq!(x, y);
    }

    #[test]
    fn intractable_input_is_refused() {
        let m = small(3, 4, 3); // 7 nodes, above max_nodes = 6
        let e = Percept::from_bytes(b"abab");
        let cfg = ExactCfg::from_model(&m);
        assert!(analyze_exact(&m, &e, Detail::Lean, Unit::Bits, &cfg).is_err());
        assert!(score_exact(&m, &e, Unit::Bits, &cfg).is_err());
    }

    #[test]
    fn verify_catches_a_tampered_analysis() {
        let m = small(11, 3, 2);
        let e = Percept::from_bytes(b"abb");
        let mut ax = analyse(&m, &e, Detail::Full);
        assert!(verify(&ax).is_ok());
        ax.an.phi_bits += 1.0;
        assert!(verify(&ax).is_err());
    }

    #[test]
    fn report_says_exact_and_drops_the_estimator_legend() {
        let m = small(11, 3, 2);
        let e = Percept::from_bytes(b"abb");
        let ax = analyse(&m, &e, Detail::Full);
        let r = render_exact(&ax, &m, &e, &RenderCfg::default());
        assert!(r.starts_with("phisharp exact reference"));
        assert!(r.contains("phisharp report"));
        assert!(r.contains("verdict"));
        assert!(!r.contains("estimator, not exact"));
    }

    #[test]
    fn monte_carlo_backend_is_never_called_exact() {
        let m = small(19, 3, 2);
        let e = Percept::from_bytes(b"abb");
        let cfg = ExactCfg::from_model(&m).with_backend(Backend::Mc).with_draws(4);
        let ax = analyze_exact(&m, &e, Detail::Lean, Unit::Bits, &cfg).unwrap();
        assert!(!ax.is_exact());
        assert!(ax.verdict().starts_with("NOT EXACT"));
        verify(&ax).unwrap();
        let r = render_exact(&ax, &m, &e, &RenderCfg::default());
        assert!(r.contains("R = 4"));
    }

    // cargo test exact::run::tests::print_a_small_exact_report -- --ignored --nocapture
    #[test]
    //#[ignore]
    fn print_a_small_exact_report() {
        let m = small(21, 3, 2);
        let e = Percept::from_bytes(b"abbabb");
        let ax = analyse(&m, &e, Detail::Full);
        println!("{}", render_exact(&ax, &m, &e, &RenderCfg::default()));
    }
}