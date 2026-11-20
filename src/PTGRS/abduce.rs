//! Creative-abduction instances on top of a PTGRS world.
//!
//! `build` pipeline:
//!   1. generate a world, sample an initial state, run the hidden CTMC;
//!   2. disclose each visible initial fact with probability `known`; facts of
//!      hidden-cause (latent) predicates are never disclosed;
//!   3. observe derived facts whose derivation needs undisclosed facts. The
//!      gold hypothesis is the backward slice of the trace restricted to those
//!      undisclosed facts (a minimal explanation that postulates at least one
//!      hidden cause); the slice itself is the replayable certificate;
//!   4. rivals: wrong entity, wrong predicate, insufficient, redundant, and
//!      alternative derivations found by random backward chaining;
//!   5. every hypothesis gets a Monte-Carlo posterior score
//!          score(H) = ln P(obs | disclosed ∪ H) + Σ_{f∈H} logit(prior(f))
//!      and the instance is kept only if the gold hypothesis beats every rival,
//!      and the empty hypothesis, by `Params::margin` nats;
//!   6. latent predicates are renamed to opaque `H1`, `H2`, ... (masking).
//!
//! The result is a pure function of the seed: nothing iterates a hash map.

use super::rewrite::{replay, simulate, slice_trace, State, Step, Sub};
use super::rng::Rng;
use super::term::NodeId;
use super::util::{mix, round};
use super::world::{Params, World};

use std::collections::BTreeSet;

/// A fresh world is drawn after this many failed attempts on the current one.
const PER_WORLD: usize = 16;

// -------------------------------------------------------------------- types

/// Why a hypothesis is in the candidate list (metadata, never shown to a solver).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Origin {
    Gold,
    WrongEntity,    // one argument replaced by another entity
    WrongPredicate, // one atom re-typed to another layer-0 predicate
    Insufficient,   // one needed fact dropped
    Redundant,      // one improbable extra fact added
    RivalCause,     // alternative derivation found by backward chaining
}

impl Origin {
    pub fn label(self) -> &'static str {
        match self {
            Origin::Gold => "gold",
            Origin::WrongEntity => "wrong-entity",
            Origin::WrongPredicate => "wrong-predicate",
            Origin::Insufficient => "insufficient",
            Origin::Redundant => "redundant",
            Origin::RivalCause => "rival-cause",
        }
    }
}

/// A hypothesis (a set of undisclosed layer-0 facts) with its Monte-Carlo score.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub facts: Vec<NodeId>, // sorted ascending, no duplicates
    pub origin: Origin,
    pub hits: usize,   // runs (out of `Params::trials`) in which every observation held
    pub loglik: f64,   // ln((hits + 1/2) / (trials + 1))
    pub logprior: f64, // sum of logit(prior) over `facts`
    pub score: f64,    // loglik + logprior
}

/// One complete instance.
pub struct Problem {
    pub seed: u64,
    pub params: Params,
    pub world: World,               // latent predicates are already masked as H1, H2, ...
    pub disclosed: Vec<NodeId>,     // background facts (visible, layer 0)
    pub observed: Vec<NodeId>,      // derived facts to explain
    pub candidates: Vec<Candidate>, // shuffled; exactly one has `Origin::Gold`
    pub gold: usize,                // index of the gold candidate
    pub null: Candidate,            // "assume nothing" baseline (not an answer option)
    pub certificate: Vec<Step>,     // backward slice of the hidden run
    pub truth: Vec<NodeId>,         // complete initial state of the hidden run
    pub trace_len: usize,           // length of the full hidden trace
    pub mc_seed: u64,               // Monte-Carlo seed behind the scores
    pub attempts: usize,            // attempt on which this instance was accepted
    pub worlds: usize,              // worlds drawn so far (1 = the first one worked)
}

// ------------------------------------------------------------------ scoring

/// Monte-Carlo posterior score of `hyp` given the background `base`.
/// Trial `t` uses stream `mix(mc, t)`, so scores are exactly reproducible.
fn judge(
    w: &mut World,
    base: &[NodeId],
    hyp: &[NodeId],
    origin: Origin,
    obs: &[NodeId],
    mc: u64,
    p: &Params,
) -> Candidate {
    let mut init = State::default();
    for &f in base.iter().chain(hyp.iter()) {
        init.facts.insert(f);
    }
    let mut hits = 0usize;
    for t in 0..p.trials {
        let mut rng = Rng::new(mix(mc, t as u64));
        let (fin, _) = simulate(&mut w.g, &w.rules, &init, &mut rng, p.horizon, p.max_steps);
        if obs.iter().all(|f| fin.facts.contains(f)) {
            hits += 1;
        }
    }
    let ll = ((hits as f64 + 0.5) / (p.trials as f64 + 1.0)).ln();
    let lp: f64 = hyp.iter().map(|&f| w.logit_prior(f)).sum();
    Candidate {
        facts: hyp.to_vec(),
        origin,
        hits,
        loglik: round(ll, 4),
        logprior: round(lp, 4),
        score: round(ll + lp, 4),
    }
}

// ------------------------------------------------------------------- rivals

fn atom(w: &mut World, p: usize, args: &[NodeId]) -> NodeId {
    let s = w.preds[p].sym;
    w.g.mk(s, args)
}

fn sorted(mut v: Vec<NodeId>) -> Vec<NodeId> {
    v.sort_unstable();
    v.dedup();
    v
}

/// Random backward chaining: pushes layer-0 atoms whose presence would let
/// `goal` be derived. Variables that occur only in a rule's premises get random
/// entities; Skolem variables must agree with the Skolem term of the binding.
fn backchain(
    w: &mut World,
    rng: &mut Rng,
    goal: NodeId,
    depth: usize,
    out: &mut Vec<NodeId>,
    budget: &mut usize,
) -> bool {
    if depth > 6 || *budget == 0 {
        return false;
    }
    *budget -= 1;
    let pi = match w.pred_of(goal) {
        Some(i) => i,
        None => return false,
    };
    if w.preds[pi].layer == 0 {
        // abducible; sampled worlds never relate an entity to itself
        let a = w.g.args(goal);
        if a.len() == 2 && a[0] == a[1] {
            return false;
        }
        out.push(goal);
        return true;
    }
    let head = w.g.head(goal);
    let mut opts: Vec<(usize, usize, usize)> = Vec::new();
    for (ri, r) in w.rules.iter().enumerate() {
        for (ai, alt) in r.alts.iter().enumerate() {
            for (k, &f) in alt.1.iter().enumerate() {
                if w.g.head(f) == head {
                    opts.push((ri, ai, k));
                }
            }
        }
    }
    rng.shuffle(&mut opts);
    for (ri, ai, k) in opts {
        let nv = w.rules[ri].nvars;
        let ns = w.rules[ri].skolem.len();
        let pat = w.rules[ri].alts[ai].1[k];
        let mut sub: Sub = vec![None; nv + ns];
        if !w.g.match_term(pat, goal, &mut sub) {
            continue;
        }
        for v in 0..nv {
            if sub[v].is_none() {
                sub[v] = Some(*rng.pick(&w.consts));
            }
        }
        let binding: Vec<NodeId> = sub[..nv].iter().map(|x| x.unwrap_or(0)).collect();
        let mut full = sub.clone();
        let mut sound = true;
        for s in 0..ns {
            let sk = w.rules[ri].skolem[s];
            let t = w.g.mk(sk, &binding);
            if let Some(b) = sub[nv + s] {
                if b != t {
                    sound = false;
                    break;
                }
            }
            full[nv + s] = Some(t);
        }
        if !sound {
            continue;
        }
        let lhs: Vec<NodeId> = w.rules[ri].lhs.clone();
        let mut prem: Vec<NodeId> = Vec::with_capacity(lhs.len());
        for q in lhs {
            prem.push(w.g.subst(q, &full));
        }
        let mark = out.len();
        let mut good = true;
        for q in prem {
            if !backchain(w, rng, q, depth + 1, out, budget) {
                good = false;
                break;
            }
        }
        if good {
            return true;
        }
        out.truncate(mark);
    }
    false
}

/// One rival hypothesis of the given kind, or `None` if this try failed.
#[allow(clippy::too_many_arguments)]
fn mutate(
    w: &mut World,
    rng: &mut Rng,
    kind: Origin,
    p: &Params,
    disclosed: &BTreeSet<NodeId>,
    gold: &[NodeId],
    obs: &[NodeId],
    per_obs: &[BTreeSet<NodeId>],
) -> Option<Vec<NodeId>> {
    match kind {
        Origin::Gold => None,
        Origin::RivalCause => {
            // re-derive some observations from scratch, keep the gold leaves of the rest
            let n = obs.len();
            let mut redo: Vec<bool> = (0..n).map(|_| rng.chance(0.5)).collect();
            if !redo.iter().any(|&b| b) {
                let k = rng.below(n);
                redo[k] = true;
            }
            let mut leaves: BTreeSet<NodeId> = BTreeSet::new();
            for i in 0..n {
                if redo[i] {
                    let mut acc: Vec<NodeId> = Vec::new();
                    let mut budget = 40usize;
                    if !backchain(w, rng, obs[i], 0, &mut acc, &mut budget) {
                        return None;
                    }
                    leaves.extend(acc);
                } else {
                    leaves.extend(per_obs[i].iter().cloned());
                }
            }
            let h: Vec<NodeId> = leaves.into_iter().filter(|f| !disclosed.contains(f)).collect();
            if h.is_empty() || h.len() > p.max_expl + 2 {
                return None;
            }
            Some(h)
        }
        Origin::WrongEntity | Origin::WrongPredicate => {
            let f = *rng.pick(gold);
            let pi = w.pred_of(f)?;
            let mut args: Vec<NodeId> = w.g.args(f).to_vec();
            let mut q = pi;
            if kind == Origin::WrongEntity {
                let k = rng.below(args.len());
                let c = *rng.pick(&w.consts);
                if c == args[k] {
                    return None;
                }
                args[k] = c;
            } else {
                let same: Vec<usize> = (0..w.preds.len())
                    .filter(|&i| i != pi && w.preds[i].layer == 0 && w.preds[i].arity == args.len())
                    .collect();
                if same.is_empty() {
                    return None;
                }
                q = same[rng.below(same.len())];
            }
            if args.len() == 2 && args[0] == args[1] {
                return None;
            }
            let nf = atom(w, q, &args);
            if disclosed.contains(&nf) || gold.contains(&nf) {
                return None;
            }
            let mut h: Vec<NodeId> = gold.iter().cloned().filter(|&x| x != f).collect();
            h.push(nf);
            Some(sorted(h))
        }
        Origin::Insufficient => {
            if gold.len() < 2 {
                return None;
            }
            let f = *rng.pick(gold);
            let h: Vec<NodeId> = gold.iter().cloned().filter(|&x| x != f).collect();
            Some(h)
        }
        Origin::Redundant => {
            // only facts that are already clearly improbable a priori
            let l0: Vec<usize> = (0..w.preds.len()).filter(|&i| w.preds[i].layer == 0).collect();
            let q = l0[rng.below(l0.len())];
            let mut args: Vec<NodeId> = Vec::new();
            for _ in 0..w.preds[q].arity {
                args.push(*rng.pick(&w.consts));
            }
            if args.len() == 2 && args[0] == args[1] {
                return None;
            }
            let nf = atom(w, q, &args);
            if disclosed.contains(&nf) || gold.contains(&nf) || w.logit_prior(nf) > -(p.margin + 0.5) {
                return None;
            }
            let mut h = gold.to_vec();
            h.push(nf);
            Some(sorted(h))
        }
    }
}

/// Up to `p.distractors` distinct rivals, cycling through the kinds.
fn make_rivals(
    w: &mut World,
    rng: &mut Rng,
    p: &Params,
    disclosed: &BTreeSet<NodeId>,
    gold: &[NodeId],
    obs: &[NodeId],
    per_obs: &[BTreeSet<NodeId>],
) -> Vec<(Origin, Vec<NodeId>)> {
    const CYCLE: [Origin; 5] = [
        Origin::RivalCause,
        Origin::WrongEntity,
        Origin::WrongPredicate,
        Origin::Insufficient,
        Origin::Redundant,
    ];
    let mut seen: BTreeSet<Vec<NodeId>> = BTreeSet::new();
    seen.insert(gold.to_vec());
    let mut out: Vec<(Origin, Vec<NodeId>)> = Vec::new();
    let mut tries = 0usize;
    while out.len() < p.distractors && tries < 12 * p.distractors.max(1) {
        let kind = CYCLE[tries % CYCLE.len()];
        tries += 1;
        if let Some(h) = mutate(w, rng, kind, p, disclosed, gold, obs, per_obs) {
            if seen.insert(h.clone()) {
                out.push((kind, h));
            }
        }
    }
    out
}

// -------------------------------------------------------------------- build

struct Draft {
    disclosed: Vec<NodeId>,
    observed: Vec<NodeId>,
    candidates: Vec<Candidate>, // gold first
    null: Candidate,
    certificate: Vec<Step>,
    truth: Vec<NodeId>,
    trace_len: usize,
}

/// One attempt on the current world; `None` if any structural or ranking test fails.
fn draft(w: &mut World, rng: &mut Rng, p: &Params, mc: u64) -> Option<Draft> {
    let init = w.sample_init(rng);
    let (fin, steps) = simulate(&mut w.g, &w.rules, &init, rng, p.horizon, p.max_steps);
    if steps.is_empty() {
        return None;
    }
    // background: each visible initial fact is disclosed with chance `known`
    let mut bg: BTreeSet<NodeId> = BTreeSet::new();
    for &f in init.facts.iter() {
        if !w.is_latent(f) && rng.chance(p.known) {
            bg.insert(f);
        }
    }
    let need_latent = w.preds.iter().any(|d| d.latent);
    // derived facts, each with the undisclosed initial facts it depends on
    let mut pool: Vec<(NodeId, BTreeSet<NodeId>)> = Vec::new();
    for &f in fin.facts.iter() {
        if init.facts.contains(&f) {
            continue;
        }
        let (_, leaves) = slice_trace(&steps, &[f]);
        let hidden: BTreeSet<NodeId> = leaves.difference(&bg).cloned().collect();
        if hidden.is_empty() || hidden.len() > p.max_expl {
            continue;
        }
        pool.push((f, hidden));
    }
    let seeds: Vec<usize> = (0..pool.len())
        .filter(|&i| !need_latent || pool[i].1.iter().any(|&h| w.is_latent(h)))
        .collect();
    if seeds.is_empty() {
        return None;
    }
    // observations: one that needs a hidden cause, then others sharing its facts first
    let first = seeds[rng.below(seeds.len())];
    let mut chosen: Vec<usize> = vec![first];
    let mut hidden: BTreeSet<NodeId> = pool[first].1.clone();
    let mut rest: Vec<usize> = (0..pool.len()).filter(|&i| i != first).collect();
    rng.shuffle(&mut rest);
    rest.sort_by_key(|&i| std::cmp::Reverse(pool[i].1.intersection(&pool[first].1).count()));
    for &i in rest.iter() {
        if chosen.len() >= p.obs.max(1) {
            break;
        }
        let merged: BTreeSet<NodeId> = hidden.union(&pool[i].1).cloned().collect();
        if merged.len() <= p.max_expl {
            hidden = merged;
            chosen.push(i);
        }
    }
    let mut obs: Vec<NodeId> = chosen.iter().map(|&i| pool[i].0).collect();
    obs.sort_unstable();
    // gold hypothesis = undisclosed leaves of the joint slice
    let (used, leaves) = slice_trace(&steps, &obs);
    let gold: Vec<NodeId> = leaves.iter().cloned().filter(|f| !bg.contains(f)).collect();
    if gold.is_empty() || gold.len() > p.max_expl {
        return None;
    }
    if need_latent && !gold.iter().any(|&f| w.is_latent(f)) {
        return None;
    }
    let per_obs: Vec<BTreeSet<NodeId>> = obs.iter().map(|&o| slice_trace(&steps, &[o]).1).collect();
    let rivals = make_rivals(w, rng, p, &bg, &gold, &obs, &per_obs);
    if rivals.len() < p.distractors.min(3) {
        return None;
    }
    // ranking: cheap tests first, the gold and the empty hypothesis before the rivals
    let base: Vec<NodeId> = bg.iter().cloned().collect();
    let top = judge(w, &base, &gold, Origin::Gold, &obs, mc, p);
    if top.hits * 6 < p.trials {
        return None; // too improbable to rank reliably
    }
    let null = judge(w, &base, &[], Origin::Insufficient, &obs, mc, p);
    if top.score - null.score < p.margin {
        return None; // the background alone already explains the observations
    }
    let mut cands: Vec<Candidate> = vec![top];
    for (origin, h) in rivals.iter() {
        let c = judge(w, &base, h, *origin, &obs, mc, p);
        if cands[0].score - c.score < p.margin {
            return None;
        }
        cands.push(c);
    }
    let certificate: Vec<Step> = used.iter().map(|&i| steps[i].clone()).collect();
    Some(Draft {
        disclosed: base,
        observed: obs,
        candidates: cands,
        null,
        certificate,
        truth: init.facts.iter().cloned().collect(),
        trace_len: steps.len(),
    })
}

/// Latent predicates become `H1`, `H2`, ... Every other name starts with a
/// different letter (constants and predicates have no `h` onset), so `H` is unambiguous.
fn mask_latent(w: &mut World) {
    let mut k = 0usize;
    for i in 0..w.preds.len() {
        if w.preds[i].latent {
            k += 1;
            let s = w.preds[i].sym;
            w.g.rename_sym(s, &format!("H{k}"));
        }
    }
}

/// Build one instance. Deterministic in `(seed, p)`; `Err` if no attempt met the
/// acceptance rules (raise `attempts`, lower `margin` or `trials`).
pub fn build(seed: u64, p: &Params) -> Result<Problem, String> {
    let mut rng = Rng::new(mix(seed, 0xAB5D));
    let mut world = World::generate(&mut rng, p);
    let mut worlds = 1usize;
    let total = p.attempts.max(1);
    for attempt in 1..=total {
        if attempt > 1 && (attempt - 1) % PER_WORLD == 0 {
            world = World::generate(&mut rng, p);
            worlds += 1;
        }
        let mc = mix(seed, 0x4D43_0000 + attempt as u64);
        let Some(d) = draft(&mut world, &mut rng, p, mc) else {
            continue;
        };
        // shuffle the answer options; the gold one was first
        let mut order: Vec<usize> = (0..d.candidates.len()).collect();
        rng.shuffle(&mut order);
        let gold = order.iter().position(|&i| i == 0).unwrap_or(0);
        let candidates: Vec<Candidate> = order.iter().map(|&i| d.candidates[i].clone()).collect();
        mask_latent(&mut world);
        return Ok(Problem {
            seed,
            params: p.clone(),
            world,
            disclosed: d.disclosed,
            observed: d.observed,
            candidates,
            gold,
            null: d.null,
            certificate: d.certificate,
            truth: d.truth,
            trace_len: d.trace_len,
            mc_seed: mc,
            attempts: attempt,
            worlds,
        });
    }
    Err(format!(
        "no instance met the acceptance rules after {total} attempts (seed {seed})"
    ))
}

// ------------------------------------------------------------------ Problem

impl Problem {
    pub fn text(&self, f: NodeId) -> String {
        self.world.g.show(f)
    }

    pub fn gold_candidate(&self) -> &Candidate {
        &self.candidates[self.gold]
    }

    /// Softmax of the scores over the answer options (sums to about 1).
    pub fn posterior(&self) -> Vec<f64> {
        let top = self.candidates.iter().map(|c| c.score).fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = self.candidates.iter().map(|c| (c.score - top).exp()).collect();
        let z: f64 = e.iter().sum();
        e.into_iter().map(|x| round(x / z, 4)).collect()
    }

    /// Structure, certificate, minimality, ranking and masking.
    pub fn verify(&mut self) -> Result<(), String> {
        self.world.validate()?;
        if self.observed.is_empty() {
            return Err("no observations".to_string());
        }
        if self.gold >= self.candidates.len()
            || self.candidates[self.gold].origin != Origin::Gold
            || self.candidates.iter().filter(|c| c.origin == Origin::Gold).count() != 1
        {
            return Err("exactly one candidate must be the gold one".to_string());
        }
        let gold_facts: Vec<NodeId> = self.candidates[self.gold].facts.clone();
        for &f in self.disclosed.iter() {
            if self.world.is_latent(f) {
                return Err("a hidden-cause fact was disclosed".to_string());
            }
        }
        for (i, c) in self.candidates.iter().enumerate() {
            if c.facts.is_empty() {
                return Err(format!("candidate {i}: empty hypothesis"));
            }
            for &f in c.facts.iter() {
                let abducible = match self.world.pred_of(f) {
                    Some(pi) => self.world.preds[pi].layer == 0,
                    None => false,
                };
                if !abducible || self.disclosed.contains(&f) {
                    return Err(format!(
                        "candidate {i}: {} is not an undisclosed layer-0 fact",
                        self.world.g.show(f)
                    ));
                }
            }
        }
        // the gold hypothesis and the background are part of the hidden truth
        for &f in self.disclosed.iter().chain(gold_facts.iter()) {
            if !self.truth.contains(&f) {
                return Err(format!(
                    "{} is not part of the hidden initial state",
                    self.world.g.show(f)
                ));
            }
        }
        for &o in self.observed.iter() {
            if self.truth.contains(&o) {
                return Err(format!("observation {} is an initial fact", self.world.g.show(o)));
            }
        }
        // certificate: replays from background + gold and derives every observation
        let mut init = State::default();
        for &f in self.disclosed.iter().chain(gold_facts.iter()) {
            init.facts.insert(f);
        }
        let fin = replay(&mut self.world.g, &self.world.rules, &init, &self.certificate, false)?;
        for &o in self.observed.iter() {
            if !fin.facts.contains(&o) {
                return Err(format!("certificate does not derive {}", self.world.g.show(o)));
            }
        }
        // minimality: every gold fact is needed by the certificate
        for &skip in gold_facts.iter() {
            let mut reduced = State::default();
            for &f in self.disclosed.iter().chain(gold_facts.iter()) {
                if f != skip {
                    reduced.facts.insert(f);
                }
            }
            let ok = replay(&mut self.world.g, &self.world.rules, &reduced, &self.certificate, false).is_ok();
            if ok {
                return Err(format!(
                    "gold fact {} is not needed by the certificate",
                    self.world.g.show(skip)
                ));
            }
        }
        // ranking
        let best = self.candidates[self.gold].score;
        for (i, c) in self.candidates.iter().enumerate() {
            if i != self.gold && best - c.score < self.params.margin - 1e-9 {
                return Err(format!(
                    "candidate {i} ({}) is within the margin of the gold one",
                    c.origin.label()
                ));
            }
        }
        if best - self.null.score < self.params.margin - 1e-9 {
            return Err("the empty hypothesis is within the margin of the gold one".to_string());
        }
        // creativity and masking
        if self.world.preds.iter().any(|d| d.latent) {
            if !gold_facts.iter().any(|&f| self.world.is_latent(f)) {
                return Err("the gold hypothesis postulates no hidden cause".to_string());
            }
            for (i, d) in self.world.preds.iter().enumerate() {
                if d.latent && !self.world.pred_name(i).starts_with('H') {
                    return Err(format!("hidden predicate {i} is not masked"));
                }
            }
        }
        Ok(())
    }

    /// Re-runs the Monte-Carlo for every candidate and the baseline; the scores
    /// must come out identical. Costs `(candidates + 1) * trials` simulations.
    pub fn verify_scores(&mut self) -> Result<(), String> {
        for i in 0..=self.candidates.len() {
            let old = if i < self.candidates.len() {
                self.candidates[i].clone()
            } else {
                self.null.clone()
            };
            let c = judge(
                &mut self.world,
                &self.disclosed,
                &old.facts,
                old.origin,
                &self.observed,
                self.mc_seed,
                &self.params,
            );
            if c.hits != old.hits || (c.score - old.score).abs() > 1e-9 {
                return Err(format!("candidate {i}: Monte-Carlo score is not reproducible"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instances_verify_and_reproduce() {
        let mut made = 0;
        for seed in 0..4u64 {
            for level in 1..=2u32 {
                let mut p = Params::level(level);
                p.trials = 16; // keep debug-mode tests quick
                if let Ok(mut a) = build(seed, &p) {
                    made += 1;
                    a.verify().unwrap();
                    a.verify_scores().unwrap();
                    let b = build(seed, &p).unwrap();
                    assert_eq!(a.disclosed, b.disclosed);
                    assert_eq!(a.observed, b.observed);
                    assert_eq!(a.gold, b.gold);
                    let sa: Vec<f64> = a.candidates.iter().map(|c| c.score).collect();
                    let sb: Vec<f64> = b.candidates.iter().map(|c| c.score).collect();
                    assert_eq!(sa, sb);
                }
            }
        }
        assert!(made > 0, "no instance could be built");
    }
}