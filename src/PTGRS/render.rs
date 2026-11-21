//! Rendering of a `Problem` into plain `Value` trees.
//!
//! * `task` is everything a solver may see: how the world works, the entities,
//!   the disclosed and observed facts and the lettered options. Origins, scores,
//!   the certificate and the hidden initial state never enter it, and every fact
//!   list is sorted by its text, so not even the order in which the generator
//!   created its atoms leaks.
//! * `key` is everything only the grader needs: the right letter, where each
//!   option came from and how it scored, the certificate, the hidden initial
//!   state, and the parameters and seeds that reproduce the instance.
//!
//! Both trees are built from `Value`s alone, so every format of `codec` carries
//! them. Seeds are written as 16 hex digits because `Value` has no `u64`.

use super::abduce::{Candidate, Problem, build, Origin};
use super::rewrite::Step;
use super::term::NodeId;
use super::util::round;
use super::value::Value;
use super::world::Params;
use super::codec::Format;

/// Layout version of the task and key trees.
pub const VERSION: u32 = 1;

/// The rules of the game as the solver reads them. Nothing here may name an
/// answer, an origin or a score; a test greps the task for that.
const BRIEF: &[&str] = &[
    "A hidden run of a stochastic rewriting system started from a set of ground facts (the initial state) and stopped at the horizon.",
    "Facts are atoms such as kita(Ba,Zo). Capitalised names are entities, and a function symbol applied to entities names another entity.",
    "A rule reads premises => outcomes @rate. Premises are joined by &. A premise marked - is removed when the rule fires, the others are only required. Different premises match different facts.",
    "?0, ?1, ... are variables for entities. A variable listed as [new ?k] does not occur in the premises: it names a new entity determined by the matched ones.",
    "Several outcomes, written p: facts | q: facts, are alternatives drawn with those probabilities. 'nothing' changes nothing. A single outcome is certain.",
    "Every match of a rule against the current state is a reaction with the rule's rate: waiting times are exponential, the next reaction is drawn in proportion to the rates, and a rule that consumes nothing fires at most once per match. The run ends at the horizon, after max_steps reactions, or when nothing can fire.",
    "Only predicates marked initial can hold at the start. Each ground atom of one (two different entities for a binary predicate) was present independently with the predicate's prior probability, and each hidden predicate H1, H2, ... holds for at least one entity.",
    "disclosed facts were in the initial state. observed facts hold at the end and were not initial. The initial state also held further facts that were not disclosed.",
    "Each option names undisclosed initial facts that were present besides the disclosed ones; no other undisclosed fact was present.",
    "Choose the most probable option: the chance that a run from disclosed + option ends with every observed fact, times the product of prior/(1-prior) over the option's facts.",
    "Answer with the label of one option.",
];

/// Spreadsheet-style option labels: A..Z, AA, AB, ...
pub fn letter(i: usize) -> String {
    let mut n = i;
    let mut s = String::new();
    loop {
        s.insert(0, (b'A' + (n % 26) as u8) as char);
        if n < 26 {
            return s;
        }
        n = n / 26 - 1;
    }
}

/// A 64-bit seed or id as 16 hex digits.
pub fn hex(x: u64) -> String {
    format!("{x:016x}")
}

fn text(p: &Problem, f: NodeId) -> String {
    if f == NodeId::MAX { "?".to_string() } else { p.text(f) }
}

/// Fact texts; `sort` orders them lexicographically and drops repeats.
fn list(p: &Problem, fs: &[NodeId], sort: bool) -> Value {
    let mut t: Vec<String> = fs.iter().map(|&f| text(p, f)).collect();
    if sort {
        t.sort();
        t.dedup();
    }
    Value::seq(t)
}

/// What the solver sees.
pub fn task(p: &Problem) -> Value {
    let w = &p.world;
    let mut preds = Value::arr();
    for (i, d) in w.preds.iter().enumerate() {
        let mut o = Value::obj()
            .with("name", w.pred_name(i))
            .with("arity", d.arity)
            .with("initial", d.layer == 0);
        if d.layer == 0 {
            o.put("prior", d.prior);
        }
        preds.push(o);
    }
    let mut options = Value::arr();
    for (i, c) in p.candidates.iter().enumerate() {
        options.push(
            Value::obj()
                .with("label", letter(i))
                .with("facts", list(p, &c.facts, true)),
        );
    }
    let entities: Vec<String> = w.consts.iter().map(|&c| text(p, c)).collect();
    let rules: Vec<String> = (0..w.rules.len()).map(|i| w.rule_text(i)).collect();
    let dynamics = Value::obj()
        .with("horizon", p.params.horizon)
        .with("max_steps", p.params.max_steps);
    Value::obj()
        .with("kind", "task")
        .with("version", VERSION)
        .with("id", hex(p.seed))
        .with("instructions", Value::seq(BRIEF.iter().copied()))
        .with("dynamics", dynamics)
        .with("entities", Value::seq(entities))
        .with("predicates", preds)
        .with("rules", Value::seq(rules))
        .with("disclosed", list(p, &p.disclosed, true))
        .with("observed", list(p, &p.observed, true))
        .with("options", options)
}

fn row(p: &Problem, c: &Candidate, label: String, post: f64) -> Value {
    Value::obj()
        .with("label", label)
        .with("origin", c.origin.label())
        .with("facts", list(p, &c.facts, true))
        .with("hits", c.hits)
        .with("loglik", c.loglik)
        .with("logprior", c.logprior)
        .with("score", c.score)
        .with("posterior", post)
}

fn step(p: &Problem, s: &Step) -> Value {
    Value::obj()
        .with("t", round(s.t, 4))
        .with("rule", s.rule)
        .with("alt", s.alt)
        .with("binding", list(p, &s.binding, false))
        .with("consumed", list(p, &s.consumed, false))
        .with("kept", list(p, &s.kept, false))
        .with("added", list(p, &s.added, false))
}

/// All difficulty knobs of an instance.
pub fn params(q: &Params) -> Value {
    Value::obj()
        .with("consts", q.consts)
        .with("layers", q.layers)
        .with("width", q.width)
        .with("latent", q.latent)
        .with("rules", q.rules)
        .with("funcs", q.funcs)
        .with("leak", q.leak)
        .with("consume", q.consume)
        .with("obs", q.obs)
        .with("known", q.known)
        .with("max_expl", q.max_expl)
        .with("trials", q.trials)
        .with("distractors", q.distractors)
        .with("margin", q.margin)
        .with("horizon", q.horizon)
        .with("max_steps", q.max_steps)
        .with("attempts", q.attempts)
}

/// What only the grader sees.
pub fn key(p: &Problem) -> Value {
    let post = p.posterior();
    let mut options = Value::arr();
    for (i, c) in p.candidates.iter().enumerate() {
        options.push(row(p, c, letter(i), post[i]));
    }
    let certificate: Vec<Value> = p.certificate.iter().map(|s| step(p, s)).collect();
    let n = &p.null;
    let baseline = Value::obj()
        .with("hits", n.hits)
        .with("loglik", n.loglik)
        .with("score", n.score);
    Value::obj()
        .with("kind", "key")
        .with("version", VERSION)
        .with("id", hex(p.seed))
        .with("answer", letter(p.gold))
        .with("options", options)
        .with("baseline", baseline)
        .with("certificate", certificate)
        .with("truth", list(p, &p.truth, true))
        .with("rule_kinds", Value::seq(p.world.kinds.iter().map(|k| k.label())))
        .with("trace_len", p.trace_len)
        .with("mc_seed", hex(p.mc_seed))
        .with("attempts", p.attempts)
        .with("worlds", p.worlds)
        .with("params", params(&p.params))
}

/// One line for logs: id, build effort, size and the weight of the right answer.
pub fn summary(p: &Problem) -> String {
    let post = p.posterior();
    format!(
        "id={} attempts={} worlds={} rules={} options={} observed={} answer={} p={:.3}",
        hex(p.seed),
        p.attempts,
        p.worlds,
        p.world.rules.len(),
        p.candidates.len(),
        p.observed.len(),
        letter(p.gold),
        post[p.gold]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small instance, or `None` if this seed finds no acceptable one.
    fn instance(seed: u64) -> Option<Problem> {
        let mut q = Params::level(1);
        q.trials = 16; // keep debug-mode tests quick
        build(seed, &q).ok()
    }

    fn keys(v: &Value) -> Vec<&str> {
        match v {
            Value::Map(m) => m.iter().map(|(k, _)| k.as_str()).collect(),
            _ => Vec::new(),
        }
    }

    fn items(v: Option<&Value>) -> &[Value] {
        match v {
            Some(Value::List(l)) => l,
            _ => &[],
        }
    }

    fn strs(v: Option<&Value>) -> Vec<&str> {
        items(v).iter().filter_map(Value::as_str).collect()
    }

    #[test]
    fn letters_run_like_spreadsheet_columns() {
        for (i, want) in [
            (0, "A"),
            (1, "B"),
            (25, "Z"),
            (26, "AA"),
            (27, "AB"),
            (51, "AZ"),
            (52, "BA"),
            (701, "ZZ"),
            (702, "AAA"),
        ] {
            assert_eq!(letter(i), want);
        }
    }

    #[test]
    fn the_task_hides_the_answer_and_the_key_matches_it() {
        let mut made = 0;
        for seed in 0..6u64 {
            let Some(p) = instance(seed) else { continue };
            made += 1;
            let (t, k) = (task(&p), key(&p));
            assert_eq!(
                keys(&t),
                [
                    "kind", "version", "id", "instructions", "dynamics", "entities", "predicates",
                    "rules", "disclosed", "observed", "options"
                ]
            );
            let flat = String::from_utf8(Format::Json.encode(&t).unwrap()).unwrap();
            for o in [
                Origin::Gold,
                Origin::WrongEntity,
                Origin::WrongPredicate,
                Origin::Insufficient,
                Origin::Redundant,
                Origin::RivalCause,
            ] {
                assert!(!flat.contains(o.label()), "the task mentions {}", o.label());
            }
            assert!(!strs(t.get("observed")).is_empty());
            for field in ["disclosed", "observed"] {
                let l = strs(t.get(field));
                assert!(l.windows(2).all(|w| w[0] < w[1]), "{field} is not sorted");
            }
            let opts = items(t.get("options"));
            let rows = items(k.get("options"));
            assert_eq!(opts.len(), p.candidates.len());
            assert_eq!(rows.len(), opts.len());
            assert_eq!(
                k.get("answer").and_then(Value::as_str),
                Some(letter(p.gold).as_str())
            );
            for (i, o) in opts.iter().enumerate() {
                assert_eq!(keys(o), ["label", "facts"]);
                assert_eq!(
                    o.get("label").and_then(Value::as_str),
                    Some(letter(i).as_str())
                );
                let l = strs(o.get("facts"));
                assert!(!l.is_empty() && l.windows(2).all(|w| w[0] < w[1]));
                let r = &rows[i];
                assert_eq!(r.get("label"), o.get("label"));
                assert!(r.get("facts").unwrap().same(o.get("facts").unwrap()));
                let is_gold = r.get("origin").and_then(Value::as_str) == Some("gold");
                assert_eq!(is_gold, i == p.gold);
            }
        }
        assert!(made > 0, "no instance could be built");
    }

    #[test]
    fn rendering_is_reproducible() {
        let mut made = 0;
        for seed in 0..4u64 {
            if let (Some(a), Some(b)) = (instance(seed), instance(seed)) {
                made += 1;
                assert!(task(&a).same(&task(&b)));
                assert!(key(&a).same(&key(&b)));
            }
        }
        assert!(made > 0, "no instance could be built");
    }

    #[test]
    fn task_and_key_survive_every_format() {
        let mut made = 0;
        for seed in 0..3u64 {
            let Some(p) = instance(seed) else { continue };
            made += 1;
            for v in [task(&p), key(&p)] {
                for f in Format::ALL {
                    let enc = f.encode(&v).unwrap();
                    assert!(f.decode(&enc).unwrap().same(&v), "{}", f.name());
                }
            }
        }
        assert!(made > 0, "no instance could be built");
    }
}