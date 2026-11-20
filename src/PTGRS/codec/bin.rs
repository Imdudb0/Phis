//! Binary formats: bencode (with extensions), CBOR and MessagePack.
//!
//! All three decoders run on one bounds-checked cursor (`Rd`). A claimed length
//! or count is compared with the bytes actually left before anything is read or
//! allocated, nesting is capped at `MAX_DEPTH`, and the input must be exactly one
//! value, so hostile data costs time linear in its size and never panics. Decoded
//! floats are finite and decoded strings valid UTF-8, so every result is a legal
//! `Value`.
//!
//! * bencode: `i<n>e`, `<len>:<bytes>`, `l..e`, `d..e`, plus `n` (null), `t`, `f`
//!   and `r<16 hex digits>e` (the bits of a float). Dictionary keys stay in
//!   insertion order and duplicates are kept: a dialect, not canonical bencode.
//! * CBOR (RFC 8949): shortest integer and length heads, definite lengths, text
//!   strings, floats always as float64. Read-only extras: float16, float32 and
//!   heads that are not the shortest. Everything outside `Value` is an error.
//! * MessagePack: smallest int / str / array / map forms, floats always as
//!   float64. Read-only extra: float32. `bin` and `ext` types are errors.

use super::MAX_DEPTH;
use super::value::Value;

// ------------------------------------------------------------------ cursor

/// Bounds-checked cursor over a byte slice; `i <= b.len()` always holds.
struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Rd<'a> {
    fn new(b: &'a [u8]) -> Rd<'a> {
        Rd { b, i: 0 }
    }

    fn left(&self) -> usize {
        self.b.len() - self.i
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn fail<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at byte {}", self.i))
    }

    fn byte(&mut self) -> Result<u8, String> {
        let Some(c) = self.peek() else {
            return self.fail("unexpected end of input");
        };
        self.i += 1;
        Ok(c)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.left() {
            return self.fail("unexpected end of input");
        }
        let b = self.b;
        let s = &b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }

    /// Big-endian unsigned integer from `n <= 8` bytes.
    fn be(&mut self, n: usize) -> Result<u64, String> {
        let s = self.take(n)?;
        Ok(s.iter().fold(0u64, |acc, &x| (acc << 8) | x as u64))
    }

    /// The bytes before the next `term`; the terminator is consumed as well.
    fn until(&mut self, term: u8) -> Result<&'a [u8], String> {
        let b = self.b;
        let rest = &b[self.i..];
        match rest.iter().position(|&x| x == term) {
            Some(k) => {
                self.i += k + 1;
                Ok(&rest[..k])
            }
            None => self.fail("unterminated token"),
        }
    }

    /// `n` bytes that must be valid UTF-8.
    fn text(&mut self, n: u64) -> Result<String, String> {
        let len = match usize::try_from(n) {
            Ok(l) if l <= self.left() => l,
            _ => return self.fail("claimed length exceeds the data"),
        };
        let at = self.i;
        let raw = self.take(len)?;
        match std::str::from_utf8(raw) {
            Ok(s) => Ok(s.to_string()),
            Err(_) => Err(format!("invalid UTF-8 at byte {at}")),
        }
    }

    /// A claimed element count, checked against the bytes left: every element
    /// takes at least `each` of them.
    fn count(&self, n: u64, each: usize) -> Result<usize, String> {
        match usize::try_from(n) {
            Ok(c) if c <= self.left() / each => Ok(c),
            _ => self.fail("claimed count exceeds the data"),
        }
    }

    fn float(&self, x: f64) -> Result<Value, String> {
        if x.is_finite() {
            Ok(Value::Float(x))
        } else {
            self.fail("non-finite float")
        }
    }

    /// The input must end here.
    fn finish(&self) -> Result<(), String> {
        if self.i < self.b.len() {
            return self.fail("trailing data");
        }
        Ok(())
    }
}

// ----------------------------------------------------------------- bencode

pub fn bencode_encode(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    ben_write(v, &mut out);
    out
}

pub fn bencode_decode(data: &[u8]) -> Result<Value, String> {
    let mut r = Rd::new(data);
    let v = ben_read(&mut r, 1)?;
    r.finish()?;
    Ok(v)
}

fn ben_put_str(s: &str, out: &mut Vec<u8>) {
    out.extend_from_slice(s.len().to_string().as_bytes());
    out.push(b':');
    out.extend_from_slice(s.as_bytes());
}

fn ben_write(v: &Value, out: &mut Vec<u8>) {
    match v {
        Value::Null => out.push(b'n'),
        Value::Bool(b) => out.push(if *b { b't' } else { b'f' }),
        Value::Int(i) => {
            out.push(b'i');
            out.extend_from_slice(i.to_string().as_bytes());
            out.push(b'e');
        }
        Value::Float(x) => out.extend_from_slice(format!("r{:016x}e", x.to_bits()).as_bytes()),
        Value::Str(s) => ben_put_str(s, out),
        Value::List(l) => {
            out.push(b'l');
            for x in l {
                ben_write(x, out);
            }
            out.push(b'e');
        }
        Value::Map(m) => {
            out.push(b'd');
            for (k, x) in m {
                ben_put_str(k, out);
                ben_write(x, out);
            }
            out.push(b'e');
        }
    }
}

/// Canonical decimal integer: no `+`, no leading zeros, no `-0`.
fn ben_int(tok: &[u8]) -> Option<i64> {
    let (neg, digits) = match tok.split_first() {
        Some((b'-', rest)) => (true, rest),
        _ => (false, tok),
    };
    let canonical = !digits.is_empty()
        && digits.iter().all(|c| c.is_ascii_digit())
        && (digits[0] != b'0' || (digits.len() == 1 && !neg));
    if !canonical {
        return None;
    }
    std::str::from_utf8(tok).ok()?.parse().ok()
}

/// Canonical string length: digits without leading zeros.
fn ben_len(tok: &[u8]) -> Option<u64> {
    let plain = !tok.is_empty()
        && tok.iter().all(|c| c.is_ascii_digit())
        && (tok[0] != b'0' || tok.len() == 1);
    if !plain {
        return None;
    }
    std::str::from_utf8(tok).ok()?.parse().ok()
}

/// `<len>:<bytes>`, with the cursor on the first digit.
fn ben_str(r: &mut Rd<'_>) -> Result<String, String> {
    let tok = r.until(b':')?;
    match ben_len(tok) {
        Some(n) => r.text(n),
        None => r.fail("bad string length"),
    }
}

fn ben_read(r: &mut Rd<'_>, depth: usize) -> Result<Value, String> {
    if depth > MAX_DEPTH {
        return r.fail("nesting too deep");
    }
    let Some(c) = r.peek() else {
        return r.fail("unexpected end of input");
    };
    if c.is_ascii_digit() {
        return ben_str(r).map(Value::Str);
    }
    r.i += 1;
    match c {
        b'n' => Ok(Value::Null),
        b't' => Ok(Value::Bool(true)),
        b'f' => Ok(Value::Bool(false)),
        b'i' => match ben_int(r.until(b'e')?) {
            Some(n) => Ok(Value::Int(n)),
            None => r.fail("bad integer"),
        },
        b'r' => {
            // exactly 16 hex digits: scanning for 'e' would stop inside them
            let mut bits = 0u64;
            for &d in r.take(16)? {
                match (d as char).to_digit(16) {
                    Some(h) => bits = (bits << 4) | h as u64,
                    None => return r.fail("bad hex digit in a float"),
                }
            }
            if r.byte()? != b'e' {
                return r.fail("a float must end with 'e'");
            }
            r.float(f64::from_bits(bits))
        }
        b'l' => {
            let mut items = Vec::new();
            loop {
                match r.peek() {
                    Some(b'e') => {
                        r.i += 1;
                        return Ok(Value::List(items));
                    }
                    Some(_) => items.push(ben_read(r, depth + 1)?),
                    None => return r.fail("unterminated list"),
                }
            }
        }
        b'd' => {
            let mut items = Vec::new();
            loop {
                match r.peek() {
                    Some(b'e') => {
                        r.i += 1;
                        return Ok(Value::Map(items));
                    }
                    Some(k) if k.is_ascii_digit() => {
                        let key = ben_str(r)?;
                        items.push((key, ben_read(r, depth + 1)?));
                    }
                    Some(_) => return r.fail("dictionary keys must be strings"),
                    None => return r.fail("unterminated dictionary"),
                }
            }
        }
        _ => r.fail("unexpected byte"),
    }
}

// -------------------------------------------------------------------- CBOR

pub fn cbor_encode(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    cbor_write(v, &mut out);
    out
}

pub fn cbor_decode(data: &[u8]) -> Result<Value, String> {
    let mut r = Rd::new(data);
    let v = cbor_read(&mut r, 1)?;
    r.finish()?;
    Ok(v)
}

/// Initial byte of major type `major`, followed by the shortest form of `n`.
fn cbor_head(out: &mut Vec<u8>, major: u8, n: u64) {
    let m = major << 5;
    if n < 24 {
        out.push(m | n as u8);
    } else if n <= 0xff {
        out.push(m | 24);
        out.push(n as u8);
    } else if n <= 0xffff {
        out.push(m | 25);
        out.extend_from_slice(&(n as u16).to_be_bytes());
    } else if n <= 0xffff_ffff {
        out.push(m | 26);
        out.extend_from_slice(&(n as u32).to_be_bytes());
    } else {
        out.push(m | 27);
        out.extend_from_slice(&n.to_be_bytes());
    }
}

fn cbor_put_str(s: &str, out: &mut Vec<u8>) {
    cbor_head(out, 3, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

fn cbor_write(v: &Value, out: &mut Vec<u8>) {
    match v {
        Value::Null => out.push(0xf6),
        Value::Bool(b) => out.push(if *b { 0xf5 } else { 0xf4 }),
        Value::Int(i) if *i >= 0 => cbor_head(out, 0, *i as u64),
        // a negative integer is stored as n = -1 - i, which is `!i`
        Value::Int(i) => cbor_head(out, 1, (!*i) as u64),
        Value::Float(x) => {
            out.push(0xfb);
            out.extend_from_slice(&x.to_bits().to_be_bytes());
        }
        Value::Str(s) => cbor_put_str(s, out),
        Value::List(l) => {
            cbor_head(out, 4, l.len() as u64);
            for x in l {
                cbor_write(x, out);
            }
        }
        Value::Map(m) => {
            cbor_head(out, 5, m.len() as u64);
            for (k, x) in m {
                cbor_put_str(k, out);
                cbor_write(x, out);
            }
        }
    }
}

/// The argument that follows an initial byte with additional information `info`
/// (majors 0 to 6 only; major 7 reads its own payload).
fn cbor_arg(r: &mut Rd<'_>, info: u8) -> Result<u64, String> {
    match info {
        0..=23 => Ok(info as u64),
        24 => r.be(1),
        25 => r.be(2),
        26 => r.be(4),
        27 => r.be(8),
        31 => r.fail("indefinite lengths are not supported"),
        _ => r.fail("reserved additional information"),
    }
}

/// IEEE 754 binary16 to f64, exactly. Infinities and NaN come out non-finite and
/// are refused by the caller.
fn half(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 31) as i32;
    let man = (h & 0x3ff) as f64;
    let mag = match exp {
        0 => man * 2f64.powi(-24),
        31 if man == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        e => (1.0 + man / 1024.0) * 2f64.powi(e - 15),
    };
    sign * mag
}

fn cbor_read(r: &mut Rd<'_>, depth: usize) -> Result<Value, String> {
    if depth > MAX_DEPTH {
        return r.fail("nesting too deep");
    }
    let ib = r.byte()?;
    let (major, info) = (ib >> 5, ib & 31);
    match major {
        0 | 1 => {
            let n = cbor_arg(r, info)?;
            match i64::try_from(n) {
                Ok(i) if major == 0 => Ok(Value::Int(i)),
                Ok(i) => Ok(Value::Int(-1 - i)),
                Err(_) => r.fail("integer beyond the i64 range"),
            }
        }
        2 => r.fail("byte strings are not supported"),
        3 => {
            let n = cbor_arg(r, info)?;
            r.text(n).map(Value::Str)
        }
        4 => {
            let n = cbor_arg(r, info)?;
            let n = r.count(n, 1)?;
            let mut items = Vec::with_capacity(n.min(256));
            for _ in 0..n {
                items.push(cbor_read(r, depth + 1)?);
            }
            Ok(Value::List(items))
        }
        5 => {
            let n = cbor_arg(r, info)?;
            let n = r.count(n, 2)?;
            let mut items = Vec::with_capacity(n.min(256));
            for _ in 0..n {
                let kb = r.byte()?;
                if kb >> 5 != 3 {
                    return r.fail("map keys must be text strings");
                }
                let kn = cbor_arg(r, kb & 31)?;
                let key = r.text(kn)?;
                items.push((key, cbor_read(r, depth + 1)?));
            }
            Ok(Value::Map(items))
        }
        6 => r.fail("tags are not supported"),
        _ => match info {
            20 => Ok(Value::Bool(false)),
            21 => Ok(Value::Bool(true)),
            22 => Ok(Value::Null),
            25 => {
                let h = r.be(2)? as u16;
                r.float(half(h))
            }
            26 => {
                let x = f32::from_bits(r.be(4)? as u32);
                r.float(f64::from(x))
            }
            27 => {
                let x = f64::from_bits(r.be(8)?);
                r.float(x)
            }
            _ => r.fail("unsupported simple value"),
        },
    }
}

// ------------------------------------------------------------- MessagePack

pub fn msgpack_encode(v: &Value) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    mp_write(v, &mut out)?;
    Ok(out)
}

pub fn msgpack_decode(data: &[u8]) -> Result<Value, String> {
    let mut r = Rd::new(data);
    let v = mp_read(&mut r, 1)?;
    r.finish()?;
    Ok(v)
}

/// Header of a string, array or map with `n` bytes or entries. `fix` is the tag
/// of the one-byte form (valid up to `max`), `t8` the 8-bit form (strings only),
/// `t16` and `t32` the wider forms.
fn mp_head(
    out: &mut Vec<u8>,
    n: usize,
    fix: u8,
    max: u64,
    t8: Option<u8>,
    t16: u8,
    t32: u8,
) -> Result<(), String> {
    let n = n as u64;
    if n <= max {
        out.push(fix | n as u8);
    } else if let (Some(t), true) = (t8, n <= 0xff) {
        out.push(t);
        out.push(n as u8);
    } else if n <= 0xffff {
        out.push(t16);
        out.extend_from_slice(&(n as u16).to_be_bytes());
    } else if n <= 0xffff_ffff {
        out.push(t32);
        out.extend_from_slice(&(n as u32).to_be_bytes());
    } else {
        return Err("string, array or map of 2^32 or more entries".to_string());
    }
    Ok(())
}

fn mp_put_str(s: &str, out: &mut Vec<u8>) -> Result<(), String> {
    mp_head(out, s.len(), 0xa0, 31, Some(0xd9), 0xda, 0xdb)?;
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

fn mp_int(i: i64, out: &mut Vec<u8>) {
    if i >= 0 {
        let n = i as u64;
        if n < 128 {
            out.push(n as u8);
        } else if n <= 0xff {
            out.push(0xcc);
            out.push(n as u8);
        } else if n <= 0xffff {
            out.push(0xcd);
            out.extend_from_slice(&(n as u16).to_be_bytes());
        } else if n <= 0xffff_ffff {
            out.push(0xce);
            out.extend_from_slice(&(n as u32).to_be_bytes());
        } else {
            out.push(0xcf);
            out.extend_from_slice(&n.to_be_bytes());
        }
    } else if i >= -32 {
        out.push(i as i8 as u8);
    } else if i >= i8::MIN as i64 {
        out.push(0xd0);
        out.push(i as i8 as u8);
    } else if i >= i16::MIN as i64 {
        out.push(0xd1);
        out.extend_from_slice(&(i as i16).to_be_bytes());
    } else if i >= i32::MIN as i64 {
        out.push(0xd2);
        out.extend_from_slice(&(i as i32).to_be_bytes());
    } else {
        out.push(0xd3);
        out.extend_from_slice(&i.to_be_bytes());
    }
}

fn mp_write(v: &Value, out: &mut Vec<u8>) -> Result<(), String> {
    match v {
        Value::Null => out.push(0xc0),
        Value::Bool(b) => out.push(if *b { 0xc3 } else { 0xc2 }),
        Value::Int(i) => mp_int(*i, out),
        Value::Float(x) => {
            out.push(0xcb);
            out.extend_from_slice(&x.to_bits().to_be_bytes());
        }
        Value::Str(s) => mp_put_str(s, out)?,
        Value::List(l) => {
            mp_head(out, l.len(), 0x90, 15, None, 0xdc, 0xdd)?;
            for x in l {
                mp_write(x, out)?;
            }
        }
        Value::Map(m) => {
            mp_head(out, m.len(), 0x80, 15, None, 0xde, 0xdf)?;
            for (k, x) in m {
                mp_put_str(k, out)?;
                mp_write(x, out)?;
            }
        }
    }
    Ok(())
}

fn mp_list(r: &mut Rd<'_>, n: u64, depth: usize) -> Result<Value, String> {
    let n = r.count(n, 1)?;
    let mut items = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        items.push(mp_read(r, depth + 1)?);
    }
    Ok(Value::List(items))
}

fn mp_map(r: &mut Rd<'_>, n: u64, depth: usize) -> Result<Value, String> {
    let n = r.count(n, 2)?;
    let mut items = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        let key = mp_key(r)?;
        items.push((key, mp_read(r, depth + 1)?));
    }
    Ok(Value::Map(items))
}

/// A map key: only the string forms are allowed.
fn mp_key(r: &mut Rd<'_>) -> Result<String, String> {
    let t = r.byte()?;
    let n = match t {
        0xa0..=0xbf => (t & 31) as u64,
        0xd9 => r.be(1)?,
        0xda => r.be(2)?,
        0xdb => r.be(4)?,
        _ => return r.fail("map keys must be strings"),
    };
    r.text(n)
}

fn mp_read(r: &mut Rd<'_>, depth: usize) -> Result<Value, String> {
    if depth > MAX_DEPTH {
        return r.fail("nesting too deep");
    }
    let t = r.byte()?;
    match t {
        0x00..=0x7f => Ok(Value::Int(t as i64)),
        0xe0..=0xff => Ok(Value::Int(t as i8 as i64)),
        0x80..=0x8f => mp_map(r, (t & 15) as u64, depth),
        0x90..=0x9f => mp_list(r, (t & 15) as u64, depth),
        0xa0..=0xbf => r.text((t & 31) as u64).map(Value::Str),
        0xc0 => Ok(Value::Null),
        0xc1 => r.fail("reserved type byte 0xc1"),
        0xc2 => Ok(Value::Bool(false)),
        0xc3 => Ok(Value::Bool(true)),
        0xc4..=0xc6 => r.fail("binary data is not supported"),
        0xc7..=0xc9 | 0xd4..=0xd8 => r.fail("extension types are not supported"),
        0xca => {
            let x = f32::from_bits(r.be(4)? as u32);
            r.float(f64::from(x))
        }
        0xcb => {
            let x = f64::from_bits(r.be(8)?);
            r.float(x)
        }
        0xcc => Ok(Value::Int(r.be(1)? as i64)),
        0xcd => Ok(Value::Int(r.be(2)? as i64)),
        0xce => Ok(Value::Int(r.be(4)? as i64)),
        0xcf => {
            let n = r.be(8)?;
            match i64::try_from(n) {
                Ok(i) => Ok(Value::Int(i)),
                Err(_) => r.fail("integer beyond the i64 range"),
            }
        }
        0xd0 => Ok(Value::Int(r.be(1)? as u8 as i8 as i64)),
        0xd1 => Ok(Value::Int(r.be(2)? as u16 as i16 as i64)),
        0xd2 => Ok(Value::Int(r.be(4)? as u32 as i32 as i64)),
        0xd3 => Ok(Value::Int(r.be(8)? as i64)),
        0xd9 => {
            let n = r.be(1)?;
            r.text(n).map(Value::Str)
        }
        0xda => {
            let n = r.be(2)?;
            r.text(n).map(Value::Str)
        }
        0xdb => {
            let n = r.be(4)?;
            r.text(n).map(Value::Str)
        }
        0xdc => {
            let n = r.be(2)?;
            mp_list(r, n, depth)
        }
        0xdd => {
            let n = r.be(4)?;
            mp_list(r, n, depth)
        }
        0xde => {
            let n = r.be(2)?;
            mp_map(r, n, depth)
        }
        0xdf => {
            let n = r.be(4)?;
            mp_map(r, n, depth)
        }
    }
}