//! Text formats: JSON (compact or indented) and S-expressions.
//!
//! Both run on one small cursor (`Tx`) and share the string, number and float
//! rules, so a string or a float means the same in either of them:
//!
//! * `Int` is a number without `.`, `e` or `E` and must fit an `i64`; `Float` is
//!   written in Rust's shortest round-trip form and always carries a `.` or an
//!   exponent, so it can never be read back as an `Int`.
//! * Strings escape `"`, `\` and control characters (`\n`, `\r`, `\t`, `\u00XX`);
//!   all other text, non-ASCII included, is written as plain UTF-8.
//! * S-expressions: `nil`, `#t`, `#f`, numbers, strings, `(list a b ..)` and
//!   `(map ("k" v) ..)`.

use super::MAX_DEPTH;
use crate::PTGRS::value::Value;

/// `indent` is the number of spaces per level; 0 gives the compact form.
pub fn json_encode(v: &Value, indent: usize) -> String {
    let mut out = String::new();
    json_write(v, indent, 0, &mut out);
    out
}

pub fn json_decode(s: &str) -> Result<Value, String> {
    let mut p = Tx::new(s);
    let v = json_read(&mut p, 1)?;
    p.finish()?;
    Ok(v)
}

pub fn sexp_encode(v: &Value) -> String {
    let mut out = String::new();
    sexp_write(v, &mut out);
    out
}

pub fn sexp_decode(s: &str) -> Result<Value, String> {
    let mut p = Tx::new(s);
    let v = sexp_read(&mut p, 1)?;
    p.finish()?;
    Ok(v)
}

// ------------------------------------------------------------------ cursor

/// Cursor over UTF-8 text. `i` always rests on a character boundary: it only
/// advances over ASCII bytes or over whole runs that end before an ASCII byte.
struct Tx<'a> {
    s: &'a str,
    i: usize,
}

impl<'a> Tx<'a> {
    fn new(s: &'a str) -> Tx<'a> {
        Tx { s, i: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.s.as_bytes().get(self.i).copied()
    }

    fn ws(&mut self) {
        while self.peek().is_some_and(|c| c.is_ascii_whitespace()) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        let hit = self.peek() == Some(c);
        if hit {
            self.i += 1;
        }
        hit
    }

    fn fail<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at byte {}", self.i))
    }

    /// Only whitespace may remain.
    fn finish(&mut self) -> Result<(), String> {
        self.ws();
        if self.i < self.s.len() {
            return self.fail("trailing data");
        }
        Ok(())
    }

    /// Exactly the literal `w` (JSON `null`, `true`, `false`).
    fn word(&mut self, w: &str) -> Result<(), String> {
        if !self.s[self.i..].starts_with(w) {
            return self.fail("unexpected token");
        }
        self.i += w.len();
        Ok(())
    }

    /// A run of atom characters: S-expression symbols and numbers.
    fn atom(&mut self) -> &'a str {
        let s = self.s;
        let start = self.i;
        while self
            .peek()
            .is_some_and(|c| !c.is_ascii_whitespace() && !matches!(c, b'(' | b')' | b'"'))
        {
            self.i += 1;
        }
        &s[start..self.i]
    }

    /// A JSON number token.
    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_digit() || matches!(c, b'+' | b'-' | b'.' | b'e' | b'E'))
        {
            self.i += 1;
        }
        num(&self.s[start..self.i]).map_err(|e| format!("{e} at byte {start}"))
    }

    /// The string that starts at the opening quote under the cursor.
    fn string(&mut self) -> Result<String, String> {
        let s = self.s;
        let b = s.as_bytes();
        self.i += 1; // the opening quote
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < b.len() && b[self.i] >= 0x20 && b[self.i] != b'"' && b[self.i] != b'\\' {
                self.i += 1;
            }
            out.push_str(&s[start..self.i]);
            match b.get(self.i).copied() {
                None => return self.fail("unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let Some(c) = b.get(self.i).copied() else {
                        return self.fail("unterminated escape");
                    };
                    self.i += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let ch = self.unicode()?;
                            out.push(ch);
                        }
                        _ => return self.fail("bad escape"),
                    }
                }
                Some(_) => return self.fail("control character in string"),
            }
        }
    }

    /// The part of `\uXXXX` after the `u`, including a trailing `\uXXXX` low surrogate.
    fn unicode(&mut self) -> Result<char, String> {
        let hi = self.hex4()?;
        let cp = if (0xD800..0xDC00).contains(&hi) {
            if !self.s[self.i..].starts_with("\\u") {
                return self.fail("lone surrogate");
            }
            self.i += 2;
            let lo = self.hex4()?;
            if !(0xDC00..0xE000).contains(&lo) {
                return self.fail("bad surrogate pair");
            }
            0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
        } else {
            hi
        };
        match char::from_u32(cp) {
            Some(c) => Ok(c),
            None => self.fail("invalid code point"),
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let b = self.s.as_bytes();
        if self.i + 4 > b.len() {
            return self.fail("short \\u escape");
        }
        let mut v = 0u32;
        for k in 0..4 {
            match (b[self.i + k] as char).to_digit(16) {
                Some(d) => v = v * 16 + d,
                None => return self.fail("bad \\u escape"),
            }
        }
        self.i += 4;
        Ok(v)
    }
}

// -------------------------------------------------------- shared scalar rules

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Shortest round-trip form; the safety net guarantees a `.` or an exponent.
fn write_float(x: f64, out: &mut String) {
    let t = format!("{x:?}");
    out.push_str(&t);
    if !t.contains(|c: char| matches!(c, '.' | 'e' | 'E')) {
        out.push_str(".0");
    }
}

/// A number token: `Float` if it has `.`, `e` or `E` (and is finite), else `Int`.
fn num(tok: &str) -> Result<Value, String> {
    if tok.contains(|c: char| matches!(c, '.' | 'e' | 'E')) {
        match tok.parse::<f64>() {
            Ok(x) if x.is_finite() => Ok(Value::Float(x)),
            _ => Err(format!("bad float {tok:?}")),
        }
    } else {
        tok.parse::<i64>()
            .map(Value::Int)
            .map_err(|_| format!("bad integer {tok:?}"))
    }
}

fn newline(out: &mut String, ind: usize, lvl: usize) {
    if ind > 0 {
        out.push('\n');
        for _ in 0..ind * lvl {
            out.push(' ');
        }
    }
}

// --------------------------------------------------------------------- JSON

fn json_write(v: &Value, ind: usize, lvl: usize, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(i) => out.push_str(&i.to_string()),
        Value::Float(x) => write_float(*x, out),
        Value::Str(s) => write_str(s, out),
        Value::List(l) => {
            if l.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (k, x) in l.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                newline(out, ind, lvl + 1);
                json_write(x, ind, lvl + 1, out);
            }
            newline(out, ind, lvl);
            out.push(']');
        }
        Value::Map(m) => {
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (k, (key, x)) in m.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                newline(out, ind, lvl + 1);
                write_str(key, out);
                out.push(':');
                if ind > 0 {
                    out.push(' ');
                }
                json_write(x, ind, lvl + 1, out);
            }
            newline(out, ind, lvl);
            out.push('}');
        }
    }
}

fn json_read(p: &mut Tx<'_>, depth: usize) -> Result<Value, String> {
    if depth > MAX_DEPTH {
        return p.fail("nesting too deep");
    }
    p.ws();
    match p.peek() {
        None => p.fail("unexpected end of input"),
        Some(b'n') => p.word("null").map(|_| Value::Null),
        Some(b't') => p.word("true").map(|_| Value::Bool(true)),
        Some(b'f') => p.word("false").map(|_| Value::Bool(false)),
        Some(b'"') => p.string().map(Value::Str),
        Some(b'[') => {
            p.i += 1;
            let mut items = Vec::new();
            p.ws();
            if p.eat(b']') {
                return Ok(Value::List(items));
            }
            loop {
                items.push(json_read(p, depth + 1)?);
                p.ws();
                if p.eat(b',') {
                    continue;
                }
                if p.eat(b']') {
                    return Ok(Value::List(items));
                }
                return p.fail("expected ',' or ']'");
            }
        }
        Some(b'{') => {
            p.i += 1;
            let mut items = Vec::new();
            p.ws();
            if p.eat(b'}') {
                return Ok(Value::Map(items));
            }
            loop {
                p.ws();
                if p.peek() != Some(b'"') {
                    return p.fail("expected a string key");
                }
                let k = p.string()?;
                p.ws();
                if !p.eat(b':') {
                    return p.fail("expected ':'");
                }
                let x = json_read(p, depth + 1)?;
                items.push((k, x));
                p.ws();
                if p.eat(b',') {
                    continue;
                }
                if p.eat(b'}') {
                    return Ok(Value::Map(items));
                }
                return p.fail("expected ',' or '}'");
            }
        }
        Some(c) if c == b'-' || c.is_ascii_digit() => p.number(),
        Some(_) => p.fail("unexpected character"),
    }
}

// ------------------------------------------------------------ S-expressions

fn sexp_write(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("nil"),
        Value::Bool(b) => out.push_str(if *b { "#t" } else { "#f" }),
        Value::Int(i) => out.push_str(&i.to_string()),
        Value::Float(x) => write_float(*x, out),
        Value::Str(s) => write_str(s, out),
        Value::List(l) => {
            out.push_str("(list");
            for x in l {
                out.push(' ');
                sexp_write(x, out);
            }
            out.push(')');
        }
        Value::Map(m) => {
            out.push_str("(map");
            for (k, x) in m {
                out.push_str(" (");
                write_str(k, out);
                out.push(' ');
                sexp_write(x, out);
                out.push(')');
            }
            out.push(')');
        }
    }
}

fn sexp_read(p: &mut Tx<'_>, depth: usize) -> Result<Value, String> {
    if depth > MAX_DEPTH {
        return p.fail("nesting too deep");
    }
    p.ws();
    match p.peek() {
        None => p.fail("unexpected end of input"),
        Some(b'"') => p.string().map(Value::Str),
        Some(b')') => p.fail("unexpected ')'"),
        Some(b'(') => {
            p.i += 1;
            p.ws();
            match p.atom() {
                "list" => {
                    let mut items = Vec::new();
                    loop {
                        p.ws();
                        if p.eat(b')') {
                            break;
                        }
                        items.push(sexp_read(p, depth + 1)?);
                    }
                    Ok(Value::List(items))
                }
                "map" => {
                    let mut items = Vec::new();
                    loop {
                        p.ws();
                        if p.eat(b')') {
                            break;
                        }
                        if !p.eat(b'(') {
                            return p.fail("expected '(' or ')'");
                        }
                        p.ws();
                        if p.peek() != Some(b'"') {
                            return p.fail("expected a string key");
                        }
                        let k = p.string()?;
                        let x = sexp_read(p, depth + 1)?;
                        p.ws();
                        if !p.eat(b')') {
                            return p.fail("expected ')'");
                        }
                        items.push((k, x));
                    }
                    Ok(Value::Map(items))
                }
                _ => p.fail("expected 'list' or 'map'"),
            }
        }
        Some(_) => {
            let at = p.i;
            match p.atom() {
                "nil" => Ok(Value::Null),
                "#t" => Ok(Value::Bool(true)),
                "#f" => Ok(Value::Bool(false)),
                a => num(a).map_err(|e| format!("{e} at byte {at}")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Value {
        Value::obj()
            .with("a", 1)
            .with("b", Value::seq(vec![2, 3]))
            .with("s", "x\"y")
            .with("f", 2.5)
            .with("n", Value::Null)
            .with("t", true)
            .with("e", Value::arr())
    }

    #[test]
    fn json_and_sexp_have_the_documented_shape() {
        let v = sample();
        let compact = r#"{"a":1,"b":[2,3],"s":"x\"y","f":2.5,"n":null,"t":true,"e":[]}"#;
        let pretty = r#"{
  "a": 1,
  "b": [
    2,
    3
  ],
  "s": "x\"y",
  "f": 2.5,
  "n": null,
  "t": true,
  "e": []
}"#;
        let sexp = r##"(map ("a" 1) ("b" (list 2 3)) ("s" "x\"y") ("f" 2.5) ("n" nil) ("t" #t) ("e" (list)))"##;
        assert_eq!(json_encode(&v, 0), compact);
        assert_eq!(json_encode(&v, 2), pretty);
        assert_eq!(sexp_encode(&v), sexp);
        for s in [compact, pretty] {
            assert!(json_decode(s).unwrap().same(&v));
        }
        assert!(sexp_decode(sexp).unwrap().same(&v));
    }

    #[test]
    fn foreign_documents_are_read() {
        let doc = " [ \"\\u00e9\\ud83d\\ude00\\n\\/\" , 1E2 , -0 , {\"k\" : [ ] } ] ";
        let want = Value::List(vec![
            Value::from("é😀\n/"),
            Value::Float(100.0),
            Value::Int(0),
            Value::obj().with("k", Value::arr()),
        ]);
        assert!(json_decode(doc).unwrap().same(&want));

        let sx = sexp_decode("  ( map\n (\"a\"  ( list 1 -2 3.5e0 nil #t #f \"s\" ) )\n ) ").unwrap();
        let items = vec![
            Value::Int(1),
            Value::Int(-2),
            Value::Float(3.5),
            Value::Null,
            Value::Bool(true),
            Value::Bool(false),
            Value::from("s"),
        ];
        assert!(sx.same(&Value::obj().with("a", Value::List(items))));
    }

    #[test]
    fn floats_keep_their_kind_and_bits() {
        for x in [100.0, -0.0, 0.1, 1e21, 1e-7, 5e-324, f64::MAX, f64::MIN_POSITIVE / 8.0] {
            let s = json_encode(&Value::Float(x), 0);
            assert!(s.contains(|c: char| matches!(c, '.' | 'e')), "{s}");
            assert!(json_decode(&s).unwrap().same(&Value::Float(x)), "{s}");
            assert!(sexp_decode(&sexp_encode(&Value::Float(x))).unwrap().same(&Value::Float(x)));
        }
        assert_eq!(json_encode(&Value::Float(100.0), 0), "100.0");
        assert_eq!(json_encode(&Value::Int(100), 0), "100");
        assert!(json_decode("100").unwrap().same(&Value::Int(100)));
        assert!(json_decode("-9223372036854775808").unwrap().same(&Value::Int(i64::MIN)));
        assert!(json_decode("9223372036854775808").is_err()); // beyond i64: never a silent float
    }

    #[test]
    fn malformed_text_is_rejected() {
        for bad in [
            "", "[", "[1,]", "[1 2]", "{\"a\" 1}", "{\"a\":1,}", "{1:2}", "\"abc", "\"\\x\"",
            "\"\\ud800\"", "\"\\u12\"", "\"\u{1}\"", "nul", "tru", "1e999", "--1", "1.2.3", "[1] x",
        ] {
            assert!(json_decode(bad).is_err(), "json accepted {bad:?}");
        }
        for bad in [
            "", "(", "(list 1", "(foo)", "(map (1 2))", "(map (\"a\"))", "(map (\"a\" 1)", "nil nil",
            ")", "(list))", "#x", "1e999", "\"abc",
        ] {
            assert!(sexp_decode(bad).is_err(), "sexp accepted {bad:?}");
        }
    }
}