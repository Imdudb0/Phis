//! Probabilistic term-graph rewriting over fact sets (multi-rooted jungles).
//!
//! A `Rule` is `consume ∪ require ⇒ {p_i : add_i}` with rate λ. The first
//! `take` patterns of `lhs` are consumed, the rest are only required.
//! Existential variables (indices >= nvars) are Skolemized: each becomes
//! `sk(θ(x_0), …, θ(x_{n-1}))`, a deterministic function term, which keeps the
//! system confluent w.r.t. naming and preserves sharing in the graph.
//!
//! Dynamics are a CTMC (Gillespie): every unfired match (rule, θ) is a channel
//! with propensity λ; the next event time is Exp(Σλ); the channel is picked
//! proportionally; the outcome is drawn from the rule's outcome probabilities.

use crate::rng::Rng;
use crate::term::{Graph, NodeId, SymId};
use std::collections::{BTreeSet, HashMap, HashSet};

pub type Sub = Vec<Option<NodeId>>;

#[derive(Clone, Debug)]
pub struct Rule {
    pub lhs: Vec<NodeId>,
    pub take: usize,
    pub alts: Vec<(f64, Vec<NodeId>)>,
    pub rate: f64,
    pub nvars: usize,
    pub skolem: Vec<SymId>,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub t: f64,
    pub rule: usize,
    pub alt: usize,
    pub binding: Vec<NodeId>,
    pub consumed: Vec<NodeId>,
    pub kept: Vec<NodeId>,
    pub added: Vec<NodeId>,
}

#[derive(Clone, Debug, Default)]
pub struct State {
    pub facts: BTreeSet<NodeId>,
}

fn heads_index(g: &Graph, st: &State) -> HashMap<SymId, Vec<NodeId>> {
    let mut m: HashMap<SymId, Vec<NodeId>> = HashMap::new();
    for &f in st.facts.iter() {
        m.entry(g.head(f)).or_default().push(f);
    }
    m
}

fn match_rec(
    g: &Graph,
    pats: &[NodeId],
    k: usize,
    idx: &HashMap<SymId, Vec<NodeId>>,
    used: &mut Vec<NodeId>,
    sub: &Sub,
    out: &mut Vec<Sub>,
    cap: usize,
) {
    if out.len() >= cap {
        return;
    }
    if k == pats.len() {
        out.push(sub.clone());
        return;
    }
    let cands = match idx.get(&g.head(pats[k])) {
        Some(c) => c,
        None => return,
    };
    for &f in cands.iter() {
        if used.contains(&f) {
            continue; // distinct patterns bind distinct facts
        }
        let mut s2 = sub.clone();
        if g.match_term(pats[k], f, &mut s2) {
            used.push(f);
            match_rec(g, pats, k + 1, idx, used, &s2, out, cap);
            used.pop();
            if out.len() >= cap {
                return;
            }
        }
    }
}

/// All matches of a rule's left-hand side against the indexed state.
pub fn find_matches(g: &Graph, r: &Rule, idx: &HashMap<SymId, Vec<NodeId>>, cap: usize) -> Vec<Sub> {
    let mut out: Vec<Sub> = Vec::new();
    let mut used: Vec<NodeId> = Vec::new();
    let sub: Sub = vec![None; r.nvars];
    match_rec(g, &r.lhs, 0, idx, &mut used, &sub, &mut out, cap);
    out
}

fn binding_of(r: &Rule, sub: &Sub) -> Vec<NodeId> {
    sub.iter().take(r.nvars).map(|x| x.unwrap_or(u32::MAX)).collect()
}

/// Apply rule `ri` with a full universal binding and outcome `alt`.
/// Pure function of (state, rule, binding, alt): this is what `replay` re-runs.
pub fn apply_binding(
    g: &mut Graph,
    r: &Rule,
    ri: usize,
    st: &mut State,
    binding: &[NodeId],
    alt: usize,
    t: f64,
) -> Result<Step, String> {
    if binding.len() != r.nvars {
        return Err("binding arity mismatch".to_string());
    }
    if alt >= r.alts.len() {
        return Err("bad outcome index".to_string());
    }
    let mut full: Sub = binding.iter().map(|&b| Some(b)).collect();
    for &sk in r.skolem.iter() {
        let n = g.mk(sk, binding);
        full.push(Some(n));
    }
    let matched: Vec<NodeId> = r.lhs.iter().map(|&p| g.subst(p, &full)).collect();
    for &f in matched.iter() {
        if !st.facts.contains(&f) {
            return Err(format!("missing precondition {}", g.show(f)));
        }
    }
    let consumed: Vec<NodeId> = matched[..r.take].to_vec();
    let kept: Vec<NodeId> = matched[r.take..].to_vec();
    for f in consumed.iter() {
        st.facts.remove(f);
    }
    let rhs: Vec<NodeId> = r.alts[alt].1.iter().map(|&p| g.subst(p, &full)).collect();
    let mut added: Vec<NodeId> = Vec::new();
    for f in rhs {
        if st.facts.insert(f) {
            added.push(f);
        }
    }
    Ok(Step { t, rule: ri, alt, binding: binding.to_vec(), consumed, kept, added })
}

/// Gillespie simulation from `init` until `horizon` (or quiescence / `max_steps`).
pub fn simulate(
    g: &mut Graph,
    rules: &[Rule],
    init: &State,
    rng: &mut Rng,
    horizon: f64,
    max_steps: usize,
) -> (State, Vec<Step>) {
    let mut st = init.clone();
    let mut steps: Vec<Step> = Vec::new();
    // Non-consuming rule instances fire at most once; consuming ones are
    // naturally limited by the facts they remove.
    let mut fired: HashSet<(usize, Vec<NodeId>)> = HashSet::new();
    let mut t = 0.0f64;
    let mut guard = 0usize;
    while steps.len() < max_steps && guard < max_steps * 4 {
        guard += 1;
        let idx = heads_index(g, &st);
        let mut chans: Vec<(usize, Sub)> = Vec::new();
        let mut rates: Vec<f64> = Vec::new();
        for (ri, r) in rules.iter().enumerate() {
            for sub in find_matches(g, r, &idx, 64) {
                if r.take == 0 && fired.contains(&(ri, binding_of(r, &sub))) {
                    continue;
                }
                rates.push(r.rate);
                chans.push((ri, sub));
            }
        }
        if chans.is_empty() {
            break;
        }
        let total: f64 = rates.iter().sum();
        t += rng.exp(total);
        if t > horizon {
            break;
        }
        let pick = rng.weighted(&rates);
        let (ri, sub) = chans.swap_remove(pick);
        let r = &rules[ri];
        let probs: Vec<f64> = r.alts.iter().map(|a| a.0).collect();
        let alt = rng.weighted(&probs);
        let binding = binding_of(r, &sub);
        if r.take == 0 {
            fired.insert((ri, binding.clone()));
        }
        if let Ok(step) = apply_binding(g, r, ri, &mut st, &binding, alt, t) {
            if !step.consumed.is_empty() || !step.added.is_empty() {
                steps.push(step);
            }
        }
    }
    (st, steps)
}

/// Re-execute a trace. `strict` additionally demands identical effects per step
/// (full traces); a sliced trace is replayed with `strict = false`.
pub fn replay(
    g: &mut Graph,
    rules: &[Rule],
    init: &State,
    steps: &[Step],
    strict: bool,
) -> Result<State, String> {
    let mut st = init.clone();
    for (i, s) in steps.iter().enumerate() {
        let r = match rules.get(s.rule) {
            Some(r) => r,
            None => return Err(format!("step {i}: unknown rule")),
        };
        let new = apply_binding(g, r, s.rule, &mut st, &s.binding, s.alt, s.t)
            .map_err(|e| format!("step {i}: {e}"))?;
        if strict && (new.added != s.added || new.consumed != s.consumed || new.kept != s.kept) {
            return Err(format!("step {i}: effect mismatch"));
        }
    }
    Ok(st)
}

/// Backward slice of a trace w.r.t. `targets`. Returns the indices of the
/// relevant steps (ascending) and the leaf facts that must hold initially.
/// Those leaves are the minimal explanation extracted from this derivation.
pub fn slice_trace(steps: &[Step], targets: &[NodeId]) -> (Vec<usize>, BTreeSet<NodeId>) {
    let mut need: BTreeSet<NodeId> = targets.iter().cloned().collect();
    let mut used: Vec<usize> = Vec::new();
    for i in (0..steps.len()).rev() {
        let s = &steps[i];
        if s.added.iter().any(|f| need.contains(f)) {
            for f in s.added.iter() {
                need.remove(f);
            }
            for f in s.consumed.iter().chain(s.kept.iter()) {
                need.insert(*f);
            }
            used.push(i);
        }
    }
    used.reverse();
    (used, need)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::term::Kind;

    #[test]
    fn chain_rule_fires_slices_and_replays() {
        let mut g = Graph::new();
        let a = g.sym("a", Kind::Func, 0);
        let p = g.sym("p", Kind::Pred, 1);
        let q = g.sym("q", Kind::Pred, 1);
        let na = g.mk(a, &[]);
        let x = g.var(0);
        let px = g.mk(p, &[x]);
        let qx = g.mk(q, &[x]);
        let rule = Rule {
            lhs: vec![px],
            take: 0,
            alts: vec![(1.0, vec![qx])],
            rate: 5.0,
            nvars: 1,
            skolem: vec![],
        };
        let pa = g.mk(p, &[na]);
        let qa = g.mk(q, &[na]);
        let mut init = State::default();
        init.facts.insert(pa);
        let mut rng = Rng::new(7);
        let (fin, steps) = simulate(&mut g, &[rule.clone()], &init, &mut rng, 1e9, 100);
        assert!(fin.facts.contains(&qa));
        assert_eq!(steps.len(), 1);
        let (used, need) = slice_trace(&steps, &[qa]);
        assert_eq!(used, vec![0]);
        assert!(need.contains(&pa));
        let re = replay(&mut g, &[rule], &init, &steps, true).unwrap();
        assert_eq!(re.facts, fin.facts);
    }
}