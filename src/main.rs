//! ptgrs: universal synthetic generator for creative abduction.
//!
//! A probabilistic term-graph rewriting world (PTGRS) becomes a multiple-choice
//! abduction task, and a random multi-codec engine serializes it. Zero
//! dependencies; given a seed, every output byte is reproducible.
//!
//! This file is the command line and the self-test. `ptgrs help` prints the usage.
//! Exit status: 0 success, 1 a failure (or nothing could be built), 2 a usage error.

use phisharp::PTGRS::abduce::{build, Problem};
use phisharp::PTGRS::armor::Armor;
use phisharp::PTGRS::codec::Format;
use phisharp::PTGRS::pipeline::Pipeline;
use phisharp::PTGRS::rng::Rng;
use std::io::{self, Read, Write};
use phisharp::PTGRS::util::mix;
use phisharp::PTGRS::value::{random_value, Value};
use phisharp::PTGRS::world::Params;

// ------------------------------------------------------------------ constants

/// First word of every record header.
const MAGIC: &str = "#ptgrs";
/// Instance `i` of a batch with master seed `s` gets the seed `mix(s, SALT + i)`.
const SALT: u64 = 0x5EED_0000;
/// A seed that does not build is replaced by `mix(seed, RETRY + k)`, `k = 1..=RETRIES`.
const RETRY: u64 = 0x2E72_0000;
const RETRIES: u64 = 8;
/// An instance's codec is drawn from `Rng::new(mix(id, CODEC))`.
const CODEC: u64 = 0xC0DE_C000;
/// The seed of `selftest` unless `--seed` is given: a self-test must repeat.
const SELFTEST_SEED: u64 = 0x5E1F_7E57;
const MAX_COUNT: usize = 1_000_000;

// --------------------------------------------------------------------- tables

/// Option name, value placeholder (empty for a flag), meaning.
static OPTIONS: [(&str, &str, &str); 6] = [
    ("seed", "N", "master seed, decimal or 0x hex (without it the clock is used and the seed is printed)"),
    ("level", "L|A-B", "difficulty 1..=6, or a range whose levels are used in turn"),
    ("codec", "random|SPEC", "how tasks are serialized; random draws a SPEC from each instance's id"),
    ("key", "", "also write the answer key, as a second record in jsonp|raw"),
    ("count", "N", "number of instances (or specs)"),
    ("set", "K=V", "override one parameter, repeatable (see PARAMETERS)"),
];

/// Commands that take options: the options each accepts, and a summary.
static COMMANDS: [(&str, &[&str], &str); 5] = [
    ("gen", &["seed", "level", "codec", "key", "set"], "one instance; --seed is its id"),
    ("batch", &["seed", "level", "codec", "key", "count", "set"], "COUNT instances (default 10)"),
    ("verify", &["seed", "level", "codec", "count", "set"], "build, verify and round-trip what `batch` would write"),
    ("selftest", &["seed", "level", "count", "set"], "verify, rebuild and fuzz (levels 1-6, 6 instances, fixed seed, 24 trials)"),
    ("codec random", &["seed", "count"], "random SPECs, one per line (default 10)"),
];

/// Parameter name, smallest and largest accepted value, meaning.
static KNOBS: [(&str, f64, f64, &str); 17] = [
    ("consts", 3.0, 64.0, "entities"),
    ("layers", 3.0, 12.0, "predicate layers; layer 0 is the abducible vocabulary"),
    ("width", 1.0, 16.0, "visible predicates per layer"),
    ("latent", 0.0, 8.0, "hidden-cause predicates"),
    ("rules", 1.0, 200.0, "rule budget, hidden-cause rules included"),
    ("funcs", 0.0, 8.0, "unary function symbols"),
    ("leak", 0.0, 0.9, "chance that a rule can silently fail"),
    ("consume", 0.0, 1.0, "chance that a chain or conjunction eats its premise"),
    ("obs", 1.0, 16.0, "facts to explain"),
    ("known", 0.0, 1.0, "chance that a visible initial fact is disclosed"),
    ("max_expl", 1.0, 16.0, "cap on undisclosed facts in the explanation"),
    ("trials", 1.0, 100000.0, "Monte-Carlo runs per candidate"),
    ("distractors", 1.0, 25.0, "wrong answers kept next to the gold one"),
    ("margin", 0.0, 10.0, "required score gap in nats"),
    ("horizon", 0.1, 1000.0, "CTMC time horizon"),
    ("max_steps", 1.0, 100000.0, "reaction cap per run"),
    ("attempts", 1.0, 100000.0, "rejection-sampling budget of `build`"),
];

// ----------------------------------------------------------------- parameters

enum Field<'a> {
    Int(&'a mut usize),
    Real(&'a mut f64),
}

fn field<'a>(p: &'a mut Params, name: &str) -> Option<Field<'a>> {
    Some(match name {
        "consts" => Field::Int(&mut p.consts),
        "layers" => Field::Int(&mut p.layers),
        "width" => Field::Int(&mut p.width),
        "latent" => Field::Int(&mut p.latent),
        "rules" => Field::Int(&mut p.rules),
        "funcs" => Field::Int(&mut p.funcs),
        "leak" => Field::Real(&mut p.leak),
        "consume" => Field::Real(&mut p.consume),
        "obs" => Field::Int(&mut p.obs),
        "known" => Field::Real(&mut p.known),
        "max_expl" => Field::Int(&mut p.max_expl),
        "trials" => Field::Int(&mut p.trials),
        "distractors" => Field::Int(&mut p.distractors),
        "margin" => Field::Real(&mut p.margin),
        "horizon" => Field::Real(&mut p.horizon),
        "max_steps" => Field::Int(&mut p.max_steps),
        "attempts" => Field::Int(&mut p.attempts),
        _ => return None,
    })
}

fn is_int(name: &str) -> bool {
    let mut scratch = Params::level(1);
    matches!(field(&mut scratch, name), Some(Field::Int(_)))
}

/// Canonical decimal: digits only, no sign, no leading zeros.
fn dec(s: &str) -> Option<usize> {
    let plain = !s.is_empty()
        && s.bytes().all(|b| b.is_ascii_digit())
        && (s == "0" || !s.starts_with('0'));
    if plain { s.parse::<usize>().ok() } else { None }
}

/// One `--set` pair, checked against the table: nothing is clamped silently.
fn knob(name: &str, val: &str) -> Result<(&'static str, f64), String> {
    let Some(&(key, lo, hi, _)) = KNOBS.iter().find(|k| k.0 == name) else {
        let all: Vec<&str> = KNOBS.iter().map(|k| k.0).collect();
        return Err(format!("unknown parameter {name:?} (expected one of: {})", all.join(", ")));
    };
    let x = if is_int(key) {
        match dec(val) {
            Some(n) => n as f64,
            None => return Err(format!("{key}: expected a plain decimal integer, got {val:?}")),
        }
    } else {
        match val.parse::<f64>() {
            Ok(x) if x.is_finite() => x,
            _ => return Err(format!("{key}: expected a number, got {val:?}")),
        }
    };
    if !(lo <= x && x <= hi) {
        return Err(format!("{key}: {val} is outside {lo}..={hi}"));
    }
    Ok((key, x))
}

fn apply(p: &mut Params, set: &[(&'static str, f64)]) {
    for &(name, x) in set {
        match field(p, name) {
            Some(Field::Int(r)) => *r = x as usize,
            Some(Field::Real(r)) => *r = x,
            None => {}
        }
    }
}

fn params_for(level: u32, set: &[(&'static str, f64)]) -> Params {
    let mut p = Params::level(level);
    apply(&mut p, set);
    p
}

// -------------------------------------------------------------------- options

#[derive(Clone, Debug, PartialEq)]
enum Codec {
    Random,
    Fixed(Pipeline),
}

#[derive(Clone, Debug)]
struct Opts {
    seed: Option<u64>,
    level: Option<(u32, u32)>,
    codec: Codec,
    key: bool,
    count: Option<usize>,
    set: Vec<(&'static str, f64)>,
}

#[derive(Debug)]
enum Cmd {
    Help,
    Gen(Opts),
    Batch(Opts),
    Verify(Opts),
    SelfTest(Opts),
    Random(Opts),
    Encode(Pipeline),
    Decode(Option<Pipeline>),
}

fn accepts(cmd: &str) -> &'static [&'static str] {
    for c in COMMANDS.iter() {
        if c.0 == cmd {
            return c.1;
        }
    }
    &[]
}

fn parse_seed(s: &str) -> Result<u64, String> {
    let r = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) if !h.is_empty() && h.bytes().all(|b| b.is_ascii_hexdigit()) => {
            u64::from_str_radix(h, 16).ok()
        }
        Some(_) => None,
        None if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) => s.parse::<u64>().ok(),
        None => None,
    };
    r.ok_or_else(|| {
        format!("bad seed {s:?} (expected a decimal number or 0x and hex digits, below 2^64)")
    })
}

fn parse_level(s: &str) -> Result<(u32, u32), String> {
    let bad = || format!("bad level {s:?} (expected 1..=6 or a range such as 2-5)");
    let (a, b) = match s.split_once('-') {
        Some((a, b)) => (a, b),
        None => (s, s),
    };
    let (Some(lo), Some(hi)) = (dec(a), dec(b)) else {
        return Err(bad());
    };
    if lo < 1 || hi > 6 || lo > hi {
        return Err(bad());
    }
    Ok((lo as u32, hi as u32))
}

/// `--name value`, `--name=value` and bare flags; positional arguments are errors.
fn parse_opts(cmd: &str, args: &[String]) -> Result<Opts, String> {
    let allowed = accepts(cmd);
    let mut o = Opts {
        seed: None,
        level: None,
        codec: Codec::Random,
        key: false,
        count: None,
        set: Vec::new(),
    };
    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        i += 1;
        let Some(body) = a.strip_prefix("--") else {
            return Err(format!("unexpected argument {a:?} for `{cmd}`"));
        };
        let (name, inline) = match body.split_once('=') {
            Some((n, v)) => (n, Some(v)),
            None => (body, None),
        };
        let Some(&(_, arg, _)) = OPTIONS.iter().find(|x| x.0 == name && allowed.contains(&x.0)) else {
            return Err(format!("unknown option --{name} for `{cmd}`"));
        };
        let v: &str = if arg.is_empty() {
            if inline.is_some() {
                return Err(format!("--{name} takes no value"));
            }
            ""
        } else if let Some(v) = inline {
            v
        } else {
            let Some(v) = args.get(i) else {
                return Err(format!("--{name} needs a value ({arg})"));
            };
            i += 1;
            v.as_str()
        };
        match name {
            "seed" => o.seed = Some(parse_seed(v)?),
            "level" => o.level = Some(parse_level(v)?),
            "codec" => {
                o.codec = if v == "random" {
                    Codec::Random
                } else {
                    Codec::Fixed(Pipeline::parse(v).map_err(|e| format!("--codec: {e}"))?)
                }
            }
            "key" => o.key = true,
            "count" => match dec(v) {
                Some(n) if (1..=MAX_COUNT).contains(&n) => o.count = Some(n),
                _ => return Err(format!("bad count {v:?} (expected 1 to {MAX_COUNT})")),
            },
            "set" => {
                let Some((k, x)) = v.split_once('=') else {
                    return Err(format!("--set expects K=V, got {v:?}"));
                };
                o.set.push(knob(k, x)?);
            }
            _ => return Err(format!("unknown option --{name} for `{cmd}`")),
        }
    }
    Ok(o)
}

fn parse(args: &[String]) -> Result<Cmd, String> {
    let Some(first) = args.first() else {
        return Ok(Cmd::Help);
    };
    let rest = &args[1..];
    Ok(match first.as_str() {
        "help" | "-h" | "--help" => Cmd::Help,
        "gen" => {
            let o = parse_opts("gen", rest)?;
            if let Some((lo, hi)) = o.level {
                if lo != hi {
                    return Err("gen takes a single level, not a range (use batch)".to_string());
                }
            }
            Cmd::Gen(o)
        }
        "batch" => Cmd::Batch(parse_opts("batch", rest)?),
        "verify" => Cmd::Verify(parse_opts("verify", rest)?),
        "selftest" => Cmd::SelfTest(parse_opts("selftest", rest)?),
        "codec" => match rest.split_first() {
            None => return Err("codec needs encode, decode or random".to_string()),
            Some((sub, more)) => match (sub.as_str(), more) {
                ("encode", [spec]) => Cmd::Encode(Pipeline::parse(spec)?),
                ("encode", _) => return Err("usage: codec encode SPEC".to_string()),
                ("decode", []) => Cmd::Decode(None),
                ("decode", [spec]) => Cmd::Decode(Some(Pipeline::parse(spec)?)),
                ("decode", _) => return Err("usage: codec decode [SPEC]".to_string()),
                ("random", _) => Cmd::Random(parse_opts("codec random", more)?),
                (other, _) => {
                    return Err(format!(
                        "unknown codec command {other:?} (expected encode, decode or random)"
                    ));
                }
            },
        },
        other => return Err(format!("unknown command {other:?} (try `ptgrs help`)")),
    })
}

// ------------------------------------------------------------ seeds and builds

/// Instance `i` of a run: a batch mixes the master seed with `i`; `gen` uses it as is.
fn seed_for(base: u64, i: usize, direct: bool) -> u64 {
    if direct { base } else { mix(base, SALT + i as u64) }
}

/// The levels of a range are used in turn.
fn level_for(level: (u32, u32), i: usize) -> u32 {
    let span = (level.1 - level.0 + 1) as usize;
    level.0 + (i % span) as u32
}

fn clock_seed() -> u64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x5EED);
    mix(t, std::process::id() as u64)
}

fn base_seed(o: &Opts, fixed: Option<u64>) -> u64 {
    if let Some(s) = o.seed.or(fixed) {
        return s;
    }
    let s = clock_seed();
    eprintln!("ptgrs: no --seed given, using 0x{s:016x}");
    s
}

/// Build an instance. A seed that does not build is replaced by derived seeds;
/// the one that worked is `Problem::seed`, and it is the id written to records.
fn make(
    seed: u64,
    level: u32,
    set: &[(&'static str, f64)],
    cap: Option<usize>,
) -> Result<Problem, String> {
    let mut p = params_for(level, set);
    if let Some(c) = cap {
        p.trials = p.trials.min(c);
    }
    let mut last = String::new();
    for k in 0..=RETRIES {
        let s = if k == 0 { seed } else { mix(seed, RETRY + k) };
        match build(s, &p) {
            Ok(a) => return Ok(a),
            Err(e) => last = e,
        }
    }
    Err(last)
}

// -------------------------------------------------------------------- records

/// `#ptgrs <kind> id=<16 hex> level=<n> bytes=<n> spec=<SPEC>`, the body of
/// exactly `bytes` bytes, then a blank line.
fn frame(kind: &str, id: u64, level: u32, spec: &str, body: &str) -> String {
    format!(
        "{MAGIC} {kind} id={} level={level} bytes={} spec={spec}\n{body}\n\n",
        render::hex(id),
        body.len()
    )
}

struct Rec<'a> {
    kind: &'a str,
    id: &'a str,
    level: usize,
    spec: &'a str,
    body: &'a str,
}

/// Read a stream of records; white space between records is skipped.
fn unframe(text: &str) -> Result<Vec<Rec<'_>>, String> {
    let mut out: Vec<Rec<'_>> = Vec::new();
    let mut at = 0usize;
    loop {
        let rest = text[at..].trim_start();
        at = text.len() - rest.len();
        if rest.is_empty() {
            return Ok(out);
        }
        let n = out.len() + 1;
        let Some(eol) = rest.find('\n') else {
            return Err(format!("record {n}: the header has no end of line"));
        };
        let head = rest[..eol].trim_end();
        let f: Vec<&str> = head.split(' ').collect();
        if f.len() != 6 || f[0] != MAGIC {
            return Err(format!(
                "record {n}: expected `{MAGIC} <kind> id=.. level=.. bytes=.. spec=..`"
            ));
        }
        if f[1] != "task" && f[1] != "key" {
            return Err(format!("record {n}: unknown kind {:?}", f[1]));
        }
        let id = match f[2].strip_prefix("id=") {
            Some(h) if h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()) => h,
            _ => return Err(format!("record {n}: bad id {:?}", f[2])),
        };
        let Some(level) = f[3].strip_prefix("level=").and_then(dec) else {
            return Err(format!("record {n}: bad level {:?}", f[3]));
        };
        let Some(len) = f[4].strip_prefix("bytes=").and_then(dec) else {
            return Err(format!("record {n}: bad byte count {:?}", f[4]));
        };
        let spec = match f[5].strip_prefix("spec=") {
            Some(s) if !s.is_empty() => s,
            _ => return Err(format!("record {n}: bad spec {:?}", f[5])),
        };
        let start = at + eol + 1;
        let end = match start.checked_add(len) {
            Some(e) if e <= text.len() => e,
            _ => {
                return Err(format!(
                    "record {n}: truncated body (the header claims {len} bytes)"
                ));
            }
        };
        let Some(body) = text.get(start..end) else {
            return Err(format!("record {n}: the body ends inside a character"));
        };
        out.push(Rec { kind: f[1], id, level, spec, body });
        at = end;
    }
}

/// Decode one record and check that the value agrees with its header.
fn decode_rec(r: &Rec<'_>) -> Result<Value, String> {
    let pipe = Pipeline::parse(r.spec)?;
    let v = pipe.decode(r.body)?;
    if v.get("kind").and_then(Value::as_str) != Some(r.kind) {
        return Err(format!("the {} record holds a different kind of value", r.kind));
    }
    if v.get("id").and_then(Value::as_str) != Some(r.id) {
        return Err(format!("the {} record holds another instance than its header says", r.kind));
    }
    Ok(v)
}

fn pipeline_for(id: u64, c: &Codec) -> Pipeline {
    match c {
        Codec::Fixed(p) => p.clone(),
        Codec::Random => Pipeline::random(&mut Rng::new(mix(id, CODEC))),
    }
}

fn key_pipe() -> Pipeline {
    Pipeline::new(Format::JsonPretty, Vec::new(), Armor::Raw, 0)
        .expect("jsonp|raw is a legal pipeline")
}

/// The task record, and with `with_key` the key record as well.
fn records(p: &Problem, level: u32, o: &Opts, with_key: bool) -> Result<String, String> {
    let pipe = pipeline_for(p.seed, &o.codec);
    let body = pipe.encode(&render::task(p)).map_err(|e| format!("task: {e}"))?;
    let mut out = frame("task", p.seed, level, &pipe.spec(), &body);
    if with_key {
        let kp = key_pipe();
        let kb = kp.encode(&render::key(p)).map_err(|e| format!("key: {e}"))?;
        out.push_str(&frame("key", p.seed, level, &kp.spec(), &kb));
    }
    Ok(out)
}

// --------------------------------------------------------------------- checks

/// Build-time checks, then the whole chain: render, encode, frame, unframe,
/// decode, compare with `Value::same`.
fn check_one(p: &mut Problem, level: u32, o: &Opts) -> Result<(), String> {
    p.verify()?;
    p.verify_scores()?;
    let text = records(p, level, o, true)?;
    let recs = unframe(&text)?;
    if recs.len() != 2 {
        return Err(format!("expected 2 records, read {}", recs.len()));
    }
    let want = [render::task(p), render::key(p)];
    for (r, w) in recs.iter().zip(want.iter()) {
        if r.id != render::hex(p.seed).as_str() || r.level != level as usize {
            return Err(format!("the {} record's header disagrees with the instance", r.kind));
        }
        let back = decode_rec(r)?;
        if !back.same(w) {
            return Err(format!("the {} record changed in transit through {}", r.kind, r.spec));
        }
    }
    Ok(())
}

/// The same seed must rebuild the same task and key.
fn rebuild(p: &Problem, level: u32, o: &Opts, cap: Option<usize>) -> Result<(), String> {
    let q = make(p.seed, level, &o.set, cap)?;
    if q.seed != p.seed {
        return Err("the seed does not rebuild the same instance".to_string());
    }
    if !render::task(&q).same(&render::task(p)) || !render::key(&q).same(&render::key(p)) {
        return Err("a rebuilt instance differs".to_string());
    }
    Ok(())
}

fn round_trips(n: usize, seed: u64) -> Result<(), String> {
    let mut rng = Rng::new(mix(seed, 0x7E57));
    for i in 0..n {
        let pipe = Pipeline::random(&mut rng);
        let depth = rng.below(5);
        let v = random_value(&mut rng, depth);
        let text = pipe.encode(&v).map_err(|e| format!("round trip {i}, {pipe}: {e}"))?;
        let back = pipe.decode(&text).map_err(|e| format!("round trip {i}, {pipe}: {e}"))?;
        if !back.same(&v) {
            return Err(format!("round trip {i}, {pipe}: the value changed"));
        }
    }
    Ok(())
}

// ------------------------------------------------------------------- commands

/// Write to standard output; a closed pipe ends the program quietly.
fn put(s: &str) {
    if let Err(e) = io::stdout().write_all(s.as_bytes()) {
        if e.kind() != io::ErrorKind::BrokenPipe {
            eprintln!("ptgrs: cannot write to standard output: {e}");
            std::process::exit(1);
        }
        std::process::exit(0);
    }
}

fn fail(msg: &str) -> i32 {
    eprintln!("ptgrs: {msg}");
    1
}

fn read_stdin() -> Result<String, String> {
    let mut s = String::new();
    io::stdin()
        .read_to_string(&mut s)
        .map_err(|e| format!("cannot read standard input: {e}"))?;
    Ok(s)
}

fn json_line(v: &Value) -> Result<String, String> {
    let b = Format::Json.encode(v)?;
    String::from_utf8(b).map_err(|_| "JSON output is not UTF-8".to_string())
}

/// `gen` (one instance) and `batch`: verified instances as records on stdout.
fn cmd_make(o: &Opts, single: bool) -> i32 {
    let base = base_seed(o, None);
    let levels = o.level.unwrap_or((3, 3));
    let n = if single { 1 } else { o.count.unwrap_or(10) };
    let mut bad = 0usize;
    for i in 0..n {
        let level = level_for(levels, i);
        let want = seed_for(base, i, single);
        let mut p = match make(want, level, &o.set, None) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("ptgrs: instance {} (level {level}): {e}", i + 1);
                bad += 1;
                continue;
            }
        };
        if p.seed != want {
            eprintln!("ptgrs: seed 0x{want:016x} did not build; using 0x{:016x}", p.seed);
        }
        if let Err(e) = p.verify() {
            eprintln!(
                "ptgrs: instance {} ({}) failed verification: {e}",
                i + 1,
                render::hex(p.seed)
            );
            bad += 1;
            continue;
        }
        match records(&p, level, o, o.key) {
            Ok(text) => {
                put(&text);
                eprintln!("ptgrs: {}", render::summary(&p));
            }
            Err(e) => {
                eprintln!("ptgrs: instance {} ({}): {e}", i + 1, render::hex(p.seed));
                bad += 1;
            }
        }
    }
    if bad > 0 { 1 } else { 0 }
}

/// `verify` (same instances as `batch`) and, with `deep`, `selftest`.
fn cmd_check(o: &Opts, deep: bool) -> i32 {
    let base = base_seed(o, if deep { Some(SELFTEST_SEED) } else { None });
    let levels = o.level.unwrap_or(if deep { (1, 6) } else { (3, 3) });
    let n = o.count.unwrap_or(if deep { 6 } else { 10 });
    let cap = if deep { Some(24) } else { None };
    let (mut ok, mut bad, mut skipped) = (0usize, 0usize, 0usize);
    for i in 0..n {
        let level = level_for(levels, i);
        let seed = seed_for(base, i, false);
        let mut p = match make(seed, level, &o.set, cap) {
            Ok(p) => p,
            Err(e) => {
                put(&format!("skip  {} level {level}: {e}\n", render::hex(seed)));
                skipped += 1;
                continue;
            }
        };
        let mut res = check_one(&mut p, level, o);
        if deep && res.is_ok() {
            res = rebuild(&p, level, o, cap);
        }
        match res {
            Ok(()) => {
                ok += 1;
                put(&format!("ok    {}\n", render::summary(&p)));
            }
            Err(e) => {
                bad += 1;
                put(&format!("FAIL  {} level {level}: {e}\n", render::hex(p.seed)));
            }
        }
    }
    if deep {
        match round_trips(300, base) {
            Ok(()) => put("ok    300 random value x pipeline round trips\n"),
            Err(e) => {
                bad += 1;
                put(&format!("FAIL  {e}\n"));
            }
        }
    }
    put(&format!("{ok} ok, {bad} failed, {skipped} skipped\n"));
    if bad > 0 || ok == 0 { 1 } else { 0 }
}

fn cmd_encode(pipe: &Pipeline) -> i32 {
    let input = match read_stdin() {
        Ok(s) => s,
        Err(e) => return fail(&e),
    };
    let v = match Format::Json.decode(input.as_bytes()) {
        Ok(v) => v,
        Err(e) => return fail(&format!("standard input is not JSON: {e}")),
    };
    match pipe.encode(&v) {
        Ok(text) => {
            put(&format!("{text}\n"));
            0
        }
        Err(e) => fail(&e),
    }
}

fn cmd_decode(pipe: Option<&Pipeline>) -> i32 {
    let input = match read_stdin() {
        Ok(s) => s,
        Err(e) => return fail(&e),
    };
    if let Some(p) = pipe {
        return match p.decode(&input).and_then(|v| json_line(&v)) {
            Ok(line) => {
                put(&format!("{line}\n"));
                0
            }
            Err(e) => fail(&e),
        };
    }
    let recs = match unframe(&input) {
        Ok(r) => r,
        Err(e) => return fail(&e),
    };
    if recs.is_empty() {
        return fail("no records on standard input");
    }
    let mut bad = 0usize;
    for (i, r) in recs.iter().enumerate() {
        match decode_rec(r).and_then(|v| json_line(&v)) {
            Ok(line) => put(&format!("{line}\n")),
            Err(e) => {
                eprintln!("ptgrs: record {}: {e}", i + 1);
                bad += 1;
            }
        }
    }
    if bad > 0 { 1 } else { 0 }
}

fn cmd_random(o: &Opts) -> i32 {
    let mut rng = Rng::new(base_seed(o, None));
    for _ in 0..o.count.unwrap_or(10) {
        put(&format!("{}\n", Pipeline::random(&mut rng)));
    }
    0
}

/// Generated from the same tables the parser reads, so it cannot drift.
fn help() -> String {
    let formats: Vec<&str> = Format::ALL.iter().map(|f| f.name()).collect();
    let armors: Vec<&str> = Armor::ALL.iter().map(|a| a.name()).collect();
    let mut s = String::from(
        "ptgrs: synthetic creative-abduction tasks from stochastic rewriting worlds\n\nUSAGE\n",
    );
    for (name, _, about) in COMMANDS.iter() {
        s.push_str(&format!("  ptgrs {name:<12} [OPTIONS]  {about}\n"));
    }
    s.push_str("  ptgrs codec encode SPEC     JSON on stdin, encoded text on stdout\n");
    s.push_str("  ptgrs codec decode [SPEC]   text on stdin, one JSON line per value; without SPEC a stream of records\n");
    s.push_str("  ptgrs help\n\nOPTIONS\n");
    for (name, arg, about) in OPTIONS.iter() {
        let left = if arg.is_empty() { format!("--{name}") } else { format!("--{name} {arg}") };
        let users: Vec<&str> =
            COMMANDS.iter().filter(|c| c.1.contains(name)).map(|c| c.0).collect();
        s.push_str(&format!("  {left:<22}{about} [{}]\n", users.join(", ")));
    }
    s.push_str(&format!(
        "\nSPEC  format|stage|...|armor[@wrap]\n  format  {}\n  stage   rle, lzss, delta, rev, mtf, xor:<1 to 16 hex digits>\n  armor   {}\n  wrap    1 to {} columns; left out = one line\n  raw armor: text formats only, no stages, no wrap\n  example: cbor|delta|lzss|xor:00000000000000ff|z85@76\n",
        formats.join(", "),
        armors.join(", "),
        pipeline::MAX_WRAP
    ));
    s.push_str("\nRECORDS\n  #ptgrs <task|key> id=<16 hex> level=<n> bytes=<n> spec=<SPEC>\n  <exactly n bytes of encoded text>\n  <blank line>\n  The id is the seed that built the instance: `gen --seed 0x<id> --level <n>`, with the same\n  --set flags, rewrites the record byte for byte. `verify` takes the same --seed, --level,\n  --codec and --set as `batch` and checks the instances `batch` would write.\n");
    s.push_str("\nPARAMETERS  (--set K=V; the level gives the defaults)\n");
    for &(name, lo, hi, about) in KNOBS.iter() {
        s.push_str(&format!("  {name:<12}{:<14}{about}\n", format!("{lo}..={hi}")));
    }
    s.push_str("\nEXAMPLES\n  ptgrs batch --seed 7 --level 2-5 --count 20 --key > tasks.txt\n  ptgrs verify --seed 7 --level 2-5 --count 20\n  ptgrs gen --seed 0x1f --level 4 --codec 'cbor|lzss|b64@76' --set trials=24\n  ptgrs codec decode < tasks.txt\n\nEXIT STATUS  0 success, 1 failure (or nothing could be built), 2 usage error\n");
    s
}

fn run(cmd: Cmd) -> i32 {
    match cmd {
        Cmd::Help => {
            put(&help());
            0
        }
        Cmd::Gen(o) => cmd_make(&o, true),
        Cmd::Batch(o) => cmd_make(&o, false),
        Cmd::Verify(o) => cmd_check(&o, false),
        Cmd::SelfTest(o) => cmd_check(&o, true),
        Cmd::Random(o) => cmd_random(&o),
        Cmd::Encode(p) => cmd_encode(&p),
        Cmd::Decode(p) => cmd_decode(p.as_ref()),
    }
}

fn main() {
    let args: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    let code = match parse(&args) {
        Ok(cmd) => run(cmd),
        Err(e) => {
            eprintln!("ptgrs: {e}\ntry `ptgrs help`");
            2
        }
    };
    let _ = io::stdout().flush();
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    fn cmd(s: &str) -> Cmd {
        parse(&args(s)).unwrap_or_else(|e| panic!("{s:?}: {e}"))
    }

    fn opts(c: Cmd) -> Opts {
        match c {
            Cmd::Gen(o) | Cmd::Batch(o) | Cmd::Verify(o) | Cmd::SelfTest(o) | Cmd::Random(o) => o,
            other => panic!("no options in {other:?}"),
        }
    }

    #[test]
    fn arguments_parse_into_commands_and_options() {
        let o = opts(cmd(
            "batch --seed 0xFF --level 2-4 --count=7 --key --codec cbor|lzss|b64@76 --set trials=12 --set=leak=0.5",
        ));
        assert_eq!(o.seed, Some(255));
        assert_eq!(o.level, Some((2, 4)));
        assert_eq!(o.count, Some(7));
        assert!(o.key);
        assert_eq!(o.codec, Codec::Fixed(Pipeline::parse("cbor|lzss|b64@76").unwrap()));
        assert_eq!(o.set, vec![("trials", 12.0), ("leak", 0.5)]);
        let q = params_for(2, &o.set);
        assert_eq!((q.trials, q.leak), (12, 0.5));
        assert_eq!(q.consts, Params::level(2).consts);

        let d = opts(cmd("gen"));
        assert_eq!((d.seed, d.level, d.count, d.key), (None, None, None, false));
        assert_eq!(d.codec, Codec::Random);
        assert!(d.set.is_empty());

        for (s, want) in [
            ("0", 0u64),
            ("007", 7),
            ("0X1f", 31),
            ("18446744073709551615", u64::MAX),
            ("0xffffffffffffffff", u64::MAX),
        ] {
            assert_eq!(parse_seed(s), Ok(want), "{s}");
        }
        assert_eq!(parse_level("4"), Ok((4, 4)));
        assert_eq!(parse_level("1-6"), Ok((1, 6)));

        assert!(matches!(cmd(""), Cmd::Help));
        assert!(matches!(cmd("help"), Cmd::Help) && matches!(cmd("--help"), Cmd::Help));
        assert!(matches!(cmd("verify --level 1-2"), Cmd::Verify(_)));
        assert!(matches!(cmd("selftest"), Cmd::SelfTest(_)));
        assert!(matches!(cmd("codec random --count 3 --seed 1"), Cmd::Random(_)));
        assert!(matches!(cmd("codec decode"), Cmd::Decode(None)));
        match cmd("codec encode json|xor:ff|hex@40") {
            Cmd::Encode(p) => assert_eq!(p.spec(), "json|xor:00000000000000ff|hex@40"),
            other => panic!("{other:?}"),
        }
        match cmd("codec decode cbor|b58") {
            Cmd::Decode(Some(p)) => assert_eq!(p.spec(), "cbor|b58"),
            other => panic!("{other:?}"),
        }

        // the seed schedule: gen uses the seed as is, a batch mixes in the index
        assert_eq!(seed_for(9, 0, true), 9);
        assert_eq!(seed_for(9, 3, false), mix(9, SALT + 3));
        assert_ne!(seed_for(9, 1, false), seed_for(9, 2, false));
        let lv: Vec<u32> = (0..7).map(|i| level_for((2, 4), i)).collect();
        assert_eq!(lv, vec![2, 3, 4, 2, 3, 4, 2]);
    }

    #[test]
    fn bad_arguments_are_usage_errors() {
        for (line, why) in [
            ("frobnicate", "unknown command"),
            ("gen --bogus", "unknown option --bogus"),
            ("gen --count 3", "unknown option --count"),
            ("gen --seed", "needs a value"),
            ("gen --seed 12x", "bad seed"),
            ("gen --seed 0x", "bad seed"),
            ("gen --seed 0xzz", "bad seed"),
            ("gen --seed 18446744073709551616", "bad seed"),
            ("gen --seed -1", "bad seed"),
            ("gen --level 0", "bad level"),
            ("gen --level 7", "bad level"),
            ("gen --level 4-2", "bad level"),
            ("gen --level x", "bad level"),
            ("gen --level 1-3", "single level"),
            ("gen --key=yes", "takes no value"),
            ("gen extra", "unexpected argument"),
            ("gen --codec yaml|b64", "--codec"),
            ("gen --codec cbor|raw", "--codec"),
            ("batch --count 0", "bad count"),
            ("batch --count 007", "bad count"),
            ("batch --count 1000001", "bad count"),
            ("batch --set trials", "expects K=V"),
            ("batch --set nope=1", "unknown parameter"),
            ("batch --set trials=0", "outside"),
            ("batch --set trials=1.5", "plain decimal"),
            ("batch --set leak=2", "outside"),
            ("batch --set leak=nan", "expected a number"),
            ("batch --set leak=abc", "expected a number"),
            ("verify --key", "unknown option --key"),
            ("selftest --codec json|b64", "unknown option --codec"),
            ("codec", "needs"),
            ("codec squeeze", "unknown codec command"),
            ("codec encode", "usage"),
            ("codec encode json|b64 extra", "usage"),
            ("codec encode yaml|b64", "unknown format"),
            ("codec decode a b", "usage"),
            ("codec random --level 2", "unknown option --level"),
        ] {
            let err = parse(&args(line)).err().unwrap_or_else(|| panic!("accepted {line:?}"));
            assert!(err.contains(why), "{line:?}: {err}");
        }
    }

    #[test]
    fn the_knob_table_covers_every_parameter() {
        // the same names as the rendered params, so a new field cannot be forgotten
        let shown = render::params(&Params::level(1));
        let Value::Map(rows) = &shown else { panic!("params is not a map") };
        let mut theirs: Vec<&str> = rows.iter().map(|(k, _)| k.as_str()).collect();
        let mut ours: Vec<&str> = KNOBS.iter().map(|k| k.0).collect();
        theirs.sort_unstable();
        ours.sort_unstable();
        assert_eq!(ours, theirs);
        assert_eq!(ours.len(), 17);
        // the defaults of every level lie inside the accepted ranges
        for level in 1..=6u32 {
            let r = render::params(&Params::level(level));
            for &(name, lo, hi, _) in KNOBS.iter() {
                let x = r.get(name).and_then(Value::as_f64).expect(name);
                assert!(lo <= x && x <= hi, "{name} = {x} at level {level}");
            }
        }
        // each bound is accepted and lands in `Params`; one step beyond it is refused
        for &(name, lo, hi, _) in KNOBS.iter() {
            let int = is_int(name);
            let text = |x: f64| if int { format!("{}", x as i64) } else { format!("{x}") };
            for x in [lo, hi] {
                let (k, v) = knob(name, &text(x)).unwrap();
                assert_eq!((k, v), (name, x));
                let mut p = Params::level(3);
                apply(&mut p, &[(k, v)]);
                assert_eq!(render::params(&p).get(name).and_then(Value::as_f64), Some(x), "{name}");
            }
            assert!(knob(name, &text(hi + 1.0)).is_err(), "{name} above its range");
            assert!(knob(name, &text(lo - 1.0)).is_err(), "{name} below its range");
        }
        assert!(is_int("trials") && !is_int("leak") && !is_int("nope"));
    }

    #[test]
    fn records_frame_and_unframe_exactly() {
        assert_eq!(
            frame("task", 255, 3, "cbor|hex", "abc"),
            "#ptgrs task id=00000000000000ff level=3 bytes=3 spec=cbor|hex\nabc\n\n"
        );
        let bodies = [
            "",
            "x",
            "two\nlines",
            "ends with a newline\n",
            "blank\n\nline inside",
            "#ptgrs key id=0000000000000000 level=1 bytes=0 spec=json|raw\nlooks like a header",
            "é漢😀",
            "  padded  ",
        ];
        let mut stream = String::new();
        for (i, b) in bodies.iter().enumerate() {
            let kind = if i % 2 == 0 { "task" } else { "key" };
            stream.push_str(&frame(kind, 0xDEAD_0000 + i as u64, 1 + (i % 6) as u32, "json|raw", b));
        }
        let recs = unframe(&stream).unwrap();
        assert_eq!(recs.len(), bodies.len());
        for (i, (r, b)) in recs.iter().zip(bodies.iter()).enumerate() {
            assert_eq!(r.body, *b);
            assert_eq!(r.kind, if i % 2 == 0 { "task" } else { "key" });
            assert_eq!(r.id, render::hex(0xDEAD_0000 + i as u64).as_str());
            assert_eq!((r.level, r.spec), (1 + i % 6, "json|raw"));
        }
        // blank lines and spaces between records are fine
        let loose = format!(
            "\n\n{}\n\n\n{}  \n",
            frame("task", 1, 2, "json|raw", "ab"),
            frame("key", 2, 2, "json|raw", "cd")
        );
        let recs = unframe(&loose).unwrap();
        assert_eq!((recs[0].body, recs[1].body), ("ab", "cd"));
    }

    #[test]
    fn malformed_streams_are_refused() {
        let h = |kind: &str, id: &str, level: &str, bytes: &str, spec: &str| {
            format!("#ptgrs {kind} id={id} level={level} bytes={bytes} spec={spec}\nabcd\n\n")
        };
        let good = "0000000000000001";
        assert_eq!(unframe(&h("task", good, "2", "4", "json|raw")).unwrap().len(), 1);
        assert!(unframe("").unwrap().is_empty());
        assert!(unframe(" \n\n").unwrap().is_empty());
        for (bad, why) in [
            (h("note", good, "2", "4", "json|raw"), "unknown kind"),
            (h("task", "1", "2", "4", "json|raw"), "bad id"),
            (h("task", "000000000000000g", "2", "4", "json|raw"), "bad id"),
            (h("task", good, "x", "4", "json|raw"), "bad level"),
            (h("task", good, "02", "4", "json|raw"), "bad level"),
            (h("task", good, "2", "-1", "json|raw"), "bad byte count"),
            (h("task", good, "2", "04", "json|raw"), "bad byte count"),
            (h("task", good, "2", "99999999999999999999999", "json|raw"), "bad byte count"),
            (h("task", good, "2", "4", ""), "bad spec"),
            (h("task", good, "2", "7", "json|raw"), "truncated"),
            (h("task", good, "2", "4294967295", "json|raw"), "truncated"),
            (h("task", good, "2", "3", "json|raw"), "expected"),
            ("garbage\n".to_string(), "expected"),
            (
                "#ptgrs task id=0000000000000001 level=2 bytes=4 spec=json|raw".to_string(),
                "no end of line",
            ),
            (
                "#ptgrs task id=0000000000000001 level=2 bytes=1 spec=json|raw\n\u{e9}\n\n".to_string(),
                "inside a character",
            ),
            (format!("{}tail", h("task", good, "2", "4", "json|raw")), "record 2"),
        ] {
            let err = unframe(&bad).err().unwrap_or_else(|| panic!("accepted {bad:?}"));
            assert!(err.contains(why), "{bad:?}: {err}");
        }
        // the decoded value has to agree with its header
        let body = Pipeline::parse("json|raw")
            .unwrap()
            .encode(&Value::obj().with("kind", "task").with("id", good))
            .unwrap();
        let text = frame("key", 1, 2, "json|raw", &body);
        let err = decode_rec(&unframe(&text).unwrap()[0]).unwrap_err();
        assert!(err.contains("different kind"), "{err}");
        let text = frame("task", 2, 2, "json|raw", &body);
        let err = decode_rec(&unframe(&text).unwrap()[0]).unwrap_err();
        assert!(err.contains("another instance"), "{err}");
        let text = frame("task", 1, 2, "json|raw", &body);
        assert!(decode_rec(&unframe(&text).unwrap()[0]).is_ok());
    }

    #[test]
    fn an_instance_survives_the_whole_chain() {
        let set = vec![("trials", 16.0)];
        let random = Opts {
            seed: None,
            level: None,
            codec: Codec::Random,
            key: true,
            count: None,
            set: set.clone(),
        };
        let fixed = Opts {
            codec: Codec::Fixed(Pipeline::parse("cbor|lzss|b58@64").unwrap()),
            ..random.clone()
        };
        let mut made = 0;
        for seed in 0..4u64 {
            let Ok(mut a) = make(seed, 1, &set, None) else { continue };
            made += 1;
            check_one(&mut a, 1, &random).unwrap();
            check_one(&mut a, 1, &fixed).unwrap();
            assert!(records(&a, 1, &fixed, false).unwrap().contains("spec=cbor|lzss|b58@64"));
            // the id is the seed that worked: it rewrites the very same records
            let b = make(a.seed, 1, &set, None).unwrap();
            assert_eq!(b.seed, a.seed);
            assert_eq!(
                records(&a, 1, &random, true).unwrap(),
                records(&b, 1, &random, true).unwrap()
            );
        }
        assert!(made > 0, "no instance could be built");
    }
}