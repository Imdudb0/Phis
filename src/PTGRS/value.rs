//! Generic data model of the codec engine: a JSON-like tree with ordered maps.
//!
//! Every format in `codec.rs` represents every `Value` exactly: maps keep their
//! insertion order, floats are finite and survive bit for bit, strings are UTF-8.
//! So `decode(encode(v))` is the identity for every format and every pipeline.

use super::rng::Rng;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64), // finite only; `Value::num` turns NaN and infinities into `Null`
    Str(String),
    List(Vec<Value>),
    Map(Vec<(String, Value)>), // insertion-ordered; duplicate keys are tolerated
}

impl Value {
    pub fn obj() -> Value {
        Value::Map(Vec::new())
    }

    pub fn arr() -> Value {
        Value::List(Vec::new())
    }

    /// List from any iterator of convertible items.
    pub fn seq<I, V>(items: I) -> Value
    where
        I: IntoIterator<Item = V>,
        V: Into<Value>,
    {
        Value::List(items.into_iter().map(|x| x.into()).collect())
    }

    /// Float, or `Null` if it is not finite.
    pub fn num(x: f64) -> Value {
        if x.is_finite() { Value::Float(x) } else { Value::Null }
    }

    /// Append `key: v` to a map (no-op for other kinds).
    pub fn put<V: Into<Value>>(&mut self, key: &str, v: V) {
        if let Value::Map(m) = self {
            m.push((key.to_string(), v.into()));
        }
    }

    /// Builder form of `put`.
    pub fn with<V: Into<Value>>(mut self, key: &str, v: V) -> Value {
        self.put(key, v);
        self
    }

    /// Append to a list (no-op for other kinds).
    pub fn push<V: Into<Value>>(&mut self, v: V) {
        if let Value::List(l) = self {
            l.push(v.into());
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn at(&self, i: usize) -> Option<&Value> {
        match self {
            Value::List(l) => l.get(i),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Ints and floats alike.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(x) => Some(*x),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Map(_) => "map",
        }
    }

    /// Number of values in the tree (map keys are not counted).
    pub fn nodes(&self) -> usize {
        1 + match self {
            Value::List(l) => l.iter().map(Value::nodes).sum::<usize>(),
            Value::Map(m) => m.iter().map(|(_, v)| v.nodes()).sum::<usize>(),
            _ => 0,
        }
    }

    pub fn depth(&self) -> usize {
        1 + match self {
            Value::List(l) => l.iter().map(Value::depth).max().unwrap_or(0),
            Value::Map(m) => m.iter().map(|(_, v)| v.depth()).max().unwrap_or(0),
            _ => 0,
        }
    }

    /// Structural equality with bitwise float comparison (`-0.0` differs from `0.0`).
    pub fn same(&self, o: &Value) -> bool {
        match (self, o) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::List(a), Value::List(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.same(y))
            }
            (Value::Map(a), Value::Map(b)) => {
                a.len() == b.len()
                    && a.iter().zip(b.iter()).all(|(x, y)| x.0 == y.0 && x.1.same(&y.1))
            }
            _ => false,
        }
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Bool(b)
    }
}
impl From<i32> for Value {
    fn from(n: i32) -> Value {
        Value::Int(n as i64)
    }
}
impl From<u32> for Value {
    fn from(n: u32) -> Value {
        Value::Int(n as i64)
    }
}
impl From<i64> for Value {
    fn from(n: i64) -> Value {
        Value::Int(n)
    }
}
impl From<usize> for Value {
    fn from(n: usize) -> Value {
        Value::Int(i64::try_from(n).unwrap_or(i64::MAX))
    }
}
impl From<f64> for Value {
    fn from(x: f64) -> Value {
        Value::num(x)
    }
}
impl From<&str> for Value {
    fn from(s: &str) -> Value {
        Value::Str(s.to_string())
    }
}
impl From<String> for Value {
    fn from(s: String) -> Value {
        Value::Str(s)
    }
}
impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Value {
        Value::List(v)
    }
}

// ------------------------------------------------------- random test values

fn random_int(rng: &mut Rng) -> i64 {
    const SPECIAL: &[i64] = &[
        0, 1, -1, 23, 24, -24, -25, 255, 256, 65_535, 65_536, 4_294_967_296, i64::MAX, i64::MIN,
    ];
    if rng.chance(0.4) {
        return SPECIAL[rng.below(SPECIAL.len())];
    }
    let bits = rng.range(1, 63) as u32;
    let x = (rng.next_u64() >> (64 - bits)) as i64; // below 2^63, so non-negative
    if rng.chance(0.5) { -x } else { x }
}

fn random_float(rng: &mut Rng) -> f64 {
    match rng.below(7) {
        0 => 0.0,
        1 => -0.0,
        2 => f64::MIN_POSITIVE / 8.0, // subnormal
        3 => f64::MAX,
        4 => -f64::MAX / 3.0,
        5 => rng.uniform(-1.0, 1.0),
        _ => rng.uniform(-1.0, 1.0) * 10f64.powi(rng.range(0, 60) as i32 - 30),
    }
}

fn random_string(rng: &mut Rng) -> String {
    const PIECES: &[&str] = &[
        "a", "b", "zz", " ", "\"", "\\", "/", "\n", "\t", "\u{0}", "\u{1f}", "\u{7f}", "é", "ß",
        "漢", "😀", "\u{2028}", ",", ":", "(", ")", "{", "}", "[", "]", "'", ";", "#", "i", "e",
        "0", "-", ".", "nil", "@",
    ];
    let n = rng.below(7);
    let mut s = String::new();
    for _ in 0..n {
        s.push_str(rng.pick(PIECES));
    }
    s
}

/// Random tree for round-trip testing: awkward strings, extreme numbers,
/// empty containers. Leaves only at `depth == 0`.
pub fn random_value(rng: &mut Rng, depth: usize) -> Value {
    let kinds = if depth == 0 { 5 } else { 7 };
    match rng.below(kinds) {
        0 => Value::Null,
        1 => Value::Bool(rng.chance(0.5)),
        2 => Value::Int(random_int(rng)),
        3 => Value::Float(random_float(rng)),
        4 => Value::Str(random_string(rng)),
        5 => {
            let n = rng.below(5);
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                items.push(random_value(rng, depth - 1));
            }
            Value::List(items)
        }
        _ => {
            let n = rng.below(5);
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                let k = random_string(rng);
                items.push((k, random_value(rng, depth - 1)));
            }
            Value::Map(items)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite(v: &Value) -> bool {
        match v {
            Value::Float(x) => x.is_finite(),
            Value::List(l) => l.iter().all(finite),
            Value::Map(m) => m.iter().all(|(_, x)| finite(x)),
            _ => true,
        }
    }

    #[test]
    fn builders_accessors_and_exact_equality() {
        let v = Value::obj()
            .with("a", 1)
            .with("b", "x")
            .with("c", Value::seq(vec![1, 2, 3]))
            .with("d", 2.5)
            .with("e", true);
        assert_eq!(v.get("a").and_then(Value::as_int), Some(1));
        assert_eq!(v.get("b").and_then(Value::as_str), Some("x"));
        assert_eq!(v.get("c").and_then(|c| c.at(2)).and_then(Value::as_int), Some(3));
        assert_eq!(v.get("d").and_then(Value::as_f64), Some(2.5));
        assert_eq!(v.get("e").and_then(Value::as_bool), Some(true));
        assert!(v.get("zz").is_none());
        assert_eq!(v.nodes(), 9);
        assert_eq!(v.depth(), 3);
        assert_eq!(Value::Float(0.0), Value::Float(-0.0));
        assert!(!Value::Float(0.0).same(&Value::Float(-0.0)));
        assert!(!Value::Int(1).same(&Value::Float(1.0)));
    }

    #[test]
    fn random_values_are_finite_varied_and_reproducible() {
        let (mut a, mut b) = (Rng::new(9), Rng::new(9));
        let mut kinds = std::collections::BTreeSet::new();
        for _ in 0..300 {
            let x = random_value(&mut a, 4);
            assert!(x.same(&random_value(&mut b, 4)));
            assert!(finite(&x));
            kinds.insert(x.kind());
        }
        assert_eq!(kinds.len(), 7);
    }
}