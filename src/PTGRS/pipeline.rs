//! Pipelines of the codec engine: one spec string says how a `Value` becomes text
//! and how that text becomes the `Value` again.
//!
//! ```text
//! Value -> Format -> bytes -> envelope -> Stage* -> bytes -> Armor -> text
//! ```
//!
//! Decoding runs the same chain backwards, so `decode(encode(v))` is the identity
//! for every legal pipeline.
//!
//! Spec string: `format ("|" stage)* "|" armor ("@" wrap)?`
//!
//! ```text
//! format  json | jsonp | sexp | bencode | cbor | msgpack
//! stage   rle | lzss | delta | rev | mtf | xor:<1 to 16 hex digits>
//! armor   raw | hex | b64 | b64u | b32 | z85 | b58 | words
//! wrap    1..=4096 columns; left out = one line
//! ```
//!
//! For example `cbor|delta|lzss|xor:00000000000000ff|z85@76`. The armor `raw` is
//! legal only after a text format, with no stages and no wrap: anything else is
//! not UTF-8.
//!
//! The envelope is `PTG1:<checksum, 8 hex digits>:<payload length>:<payload>`. The
//! checksum is the CRC-32 of the format name, a NUL byte and the payload. The
//! envelope sits inside the stages, so a wrong format, stage or armor is stopped
//! by the armor decoder, the magic or the checksum instead of returning garbage,
//! and damaged data is stopped before the format decoder runs. The header is
//! ASCII, so `json|raw` still reads as text. ASCII whitespace after the payload
//! is ignored, for pasted bodies.

use super::armor::{Armor, Stage};
use super::codec::Format;
use super::rng::Rng;
use super::util::crc32;
use super::value::{Value, random_value};

use std::fmt;

/// Most stages a spec may name.
pub const MAX_STAGES: usize = 16;
/// Widest line a spec may ask for.
pub const MAX_WRAP: usize = 4096;

const MAGIC: &str = "PTG1:";
/// Line widths the random generator picks from (it also picks "one line").
const WRAPS: [usize; 6] = [48, 64, 72, 76, 80, 100];

// ----------------------------------------------------------------- envelope

/// CRC-32 of the format name, a NUL byte and the payload. With the format in the
/// checksum a wrong format is refused instead of being read as another value
/// (the same bytes can be a valid but different value in CBOR and MessagePack).
fn checksum(format: Format, payload: &[u8]) -> u32 {
    let name = format.name().as_bytes();
    let mut buf = Vec::with_capacity(name.len() + 1 + payload.len());
    buf.extend_from_slice(name);
    buf.push(0);
    buf.extend_from_slice(payload);
    crc32(&buf)
}

/// `PTG1:<checksum, 8 hex digits>:<payload length>:<payload>`.
pub fn seal(format: Format, payload: &[u8]) -> Vec<u8> {
    let head = format!("{}{:08x}:{}:", MAGIC, checksum(format, payload), payload.len());
    let mut out = Vec::with_capacity(head.len() + payload.len());
    out.extend_from_slice(head.as_bytes());
    out.extend_from_slice(payload);
    out
}

/// The payload of a sealed buffer. The checksum field is 8 hex digits, the length
/// canonical decimal, the payload exactly that long and its checksum right; only
/// ASCII whitespace may follow it. Never panics, whatever the input.
pub fn open(format: Format, data: &[u8]) -> Result<&[u8], String> {
    let Some(rest) = data.strip_prefix(MAGIC.as_bytes()) else {
        return Err("envelope: bad magic (wrong pipeline spec or damaged data)".to_string());
    };
    if rest.len() < 9 || rest[8] != b':' {
        return Err("envelope: bad checksum field".to_string());
    }
    let mut want = 0u32;
    for &d in &rest[..8] {
        match (d as char).to_digit(16) {
            Some(h) => want = (want << 4) | h,
            None => return Err("envelope: bad checksum field".to_string()),
        }
    }
    let rest = &rest[9..];
    // at most 20 digits (the width of u64::MAX), then a colon
    let Some(colon) = rest.iter().take(21).position(|&b| b == b':') else {
        return Err("envelope: bad length field".to_string());
    };
    let digits = &rest[..colon];
    let canonical = !digits.is_empty()
        && digits.iter().all(u8::is_ascii_digit)
        && (digits[0] != b'0' || digits.len() == 1);
    let len = match std::str::from_utf8(digits).ok().and_then(|s| s.parse::<usize>().ok()) {
        Some(n) if canonical => n,
        _ => return Err("envelope: bad length field".to_string()),
    };
    let body = &rest[colon + 1..];
    if len > body.len() {
        return Err("envelope: truncated payload".to_string());
    }
    let (payload, tail) = body.split_at(len);
    if !tail.iter().all(u8::is_ascii_whitespace) {
        return Err("envelope: unexpected data after the payload".to_string());
    }
    if checksum(format, payload) != want {
        return Err("envelope: checksum mismatch (wrong pipeline spec or damaged data)".to_string());
    }
    Ok(payload)
}

// ----------------------------------------------------------------- pipeline

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Pipeline {
    pub format: Format,
    pub stages: Vec<Stage>,
    pub armor: Armor,
    pub wrap: usize, // columns of the armored text; 0 = one line
}

fn parse_wrap(s: &str) -> Result<usize, String> {
    let plain = !s.is_empty() && !s.starts_with('0') && s.bytes().all(|b| b.is_ascii_digit());
    match s.parse::<usize>() {
        Ok(n) if plain && n <= MAX_WRAP => Ok(n),
        _ => Err(format!(
            "bad wrap {:?} (expected a decimal number from 1 to {})",
            s, MAX_WRAP
        )),
    }
}

impl Pipeline {
    /// A pipeline, or the reason it is not legal.
    pub fn new(
        format: Format,
        stages: Vec<Stage>,
        armor: Armor,
        wrap: usize,
    ) -> Result<Pipeline, String> {
        let p = Pipeline { format, stages, armor, wrap };
        p.check()?;
        Ok(p)
    }

    /// Legality: `raw` armor needs a text format, no stages and no wrap; the
    /// number of stages and the wrap width are capped.
    pub fn check(&self) -> Result<(), String> {
        if self.stages.len() > MAX_STAGES {
            return Err(format!(
                "too many stages ({}, at most {})",
                self.stages.len(),
                MAX_STAGES
            ));
        }
        if self.wrap > MAX_WRAP {
            return Err(format!("wrap {} is above {}", self.wrap, MAX_WRAP));
        }
        if self.armor == Armor::Raw {
            if !self.format.is_text() {
                let texts: Vec<&str> = Format::ALL
                    .iter()
                    .filter(|f| f.is_text())
                    .map(|f| f.name())
                    .collect();
                return Err(format!(
                    "armor raw needs a text format ({}), not {}",
                    texts.join(", "),
                    self.format.name()
                ));
            }
            if !self.stages.is_empty() {
                return Err("armor raw cannot follow stages (their output is not text)".to_string());
            }
            if self.wrap != 0 {
                return Err("armor raw takes no wrap width".to_string());
            }
        }
        Ok(())
    }

    /// Parse a spec string such as `cbor|delta|xor:00000000000000ff|z85@76`.
    pub fn parse(spec: &str) -> Result<Pipeline, String> {
        let spec = spec.trim();
        let parts: Vec<&str> = spec.split('|').collect();
        if parts.len() < 2 {
            return Err(format!(
                "bad spec {:?}: needs at least a format and an armor, like \"json|b64\"",
                spec
            ));
        }
        let Some(format) = Format::parse(parts[0]) else {
            let all: Vec<&str> = Format::ALL.iter().map(|f| f.name()).collect();
            return Err(format!(
                "unknown format {:?} (expected one of: {})",
                parts[0],
                all.join(", ")
            ));
        };
        let mut stages: Vec<Stage> = Vec::new();
        for &part in &parts[1..parts.len() - 1] {
            match Stage::parse(part) {
                Some(s) => stages.push(s),
                None => {
                    return Err(format!(
                        "unknown stage {:?} (expected rle, lzss, delta, rev, mtf or xor:<1 to 16 hex digits>)",
                        part
                    ));
                }
            }
        }
        let last = parts[parts.len() - 1];
        let (name, wrap) = match last.split_once('@') {
            Some((n, w)) => (n, parse_wrap(w)?),
            None => (last, 0),
        };
        let Some(armor) = Armor::parse(name) else {
            let all: Vec<&str> = Armor::ALL.iter().map(|a| a.name()).collect();
            return Err(format!(
                "unknown armor {:?} (expected one of: {})",
                name,
                all.join(", ")
            ));
        };
        Pipeline::new(format, stages, armor, wrap)
    }

    /// The spec string; `parse(&p.spec())` gives `p` back.
    pub fn spec(&self) -> String {
        let mut s = String::from(self.format.name());
        for st in self.stages.iter() {
            s.push('|');
            s.push_str(&st.name());
        }
        s.push('|');
        s.push_str(self.armor.name());
        if self.wrap > 0 {
            s.push('@');
            s.push_str(&self.wrap.to_string());
        }
        s
    }

    /// A random legal pipeline: up to four stages, never the same kind twice in a
    /// row, `raw` armor only where it is legal.
    pub fn random(rng: &mut Rng) -> Pipeline {
        let format = Format::random(rng);
        let n = *rng.pick(&[0usize, 0, 1, 1, 2, 2, 3, 4]);
        let mut stages: Vec<Stage> = Vec::with_capacity(n);
        while stages.len() < n {
            let s = Stage::random(rng);
            if stages.last().is_some_and(|l| l.kind() == s.kind()) {
                continue;
            }
            stages.push(s);
        }
        let raw_ok = format.is_text() && stages.is_empty();
        let armors: Vec<Armor> = Armor::ALL
            .iter()
            .copied()
            .filter(|&a| a != Armor::Raw || raw_ok)
            .collect();
        let armor = *rng.pick(&armors);
        let wrap = if armor == Armor::Raw || rng.chance(0.4) { 0 } else { *rng.pick(&WRAPS) };
        Pipeline { format, stages, armor, wrap }
    }

    /// `Value` to text. `Err` for an illegal pipeline or for a value the format
    /// cannot carry (nesting beyond `codec::MAX_DEPTH`).
    pub fn encode(&self, v: &Value) -> Result<String, String> {
        self.check()?;
        let body = self
            .format
            .encode(v)
            .map_err(|e| format!("{}: {e}", self.format.name()))?;
        let mut buf = seal(self.format, &body);
        for s in self.stages.iter() {
            buf = s.apply(&buf);
        }
        self.armor.encode(&buf, self.wrap)
    }

    /// Text to `Value`. `Err` if the text is damaged or was written by another spec.
    pub fn decode(&self, text: &str) -> Result<Value, String> {
        self.check()?;
        let mut buf = self.armor.decode(text)?;
        for s in self.stages.iter().rev() {
            buf = s.undo(&buf)?;
        }
        let payload = open(self.format, &buf)?;
        self.format
            .decode(payload)
            .map_err(|e| format!("{}: {e}", self.format.name()))
    }
}

impl fmt::Display for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.spec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn spec_of(s: &str) -> Pipeline {
        Pipeline::parse(s).unwrap_or_else(|e| panic!("{s}: {e}"))
    }

    fn sample() -> Value {
        Value::obj()
            .with("name", "pipeline")
            .with("n", 42)
            .with("pi", 3.25)
            .with("ok", true)
            .with("none", Value::Null)
            .with("list", Value::seq([1, 2, 3]))
            .with("text", "é漢😀 \"quoted\"\n")
    }

    /// Replace, drop, insert or cut one or two characters of `text`.
    fn mutate(rng: &mut Rng, text: &str) -> String {
        const POOL: &[char] = &[
            '0', '1', '9', 'a', 'f', 'A', 'Z', 'z', '+', '/', '=', ' ', '\n', ':', '"', '}', 'é',
        ];
        let mut cs: Vec<char> = text.chars().collect();
        for _ in 0..rng.range(1, 2) {
            if cs.is_empty() {
                break;
            }
            let i = rng.below(cs.len());
            match rng.below(4) {
                0 => cs[i] = *rng.pick(POOL),
                1 => {
                    cs.remove(i);
                }
                2 => cs.insert(i, *rng.pick(POOL)),
                _ => cs.truncate(i),
            }
        }
        cs.into_iter().collect()
    }

    #[test]
    fn crc32_and_envelope_layout() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b"The quick brown fox jumps over the lazy dog"), 0x414F_A339);
        // the checksum covers the format name, a NUL byte and the payload
        let want = format!("PTG1:{:08x}:9:123456789", crc32(b"json\x00123456789"));
        assert_eq!(seal(Format::Json, b"123456789"), want.into_bytes());
        let want = format!("PTG1:{:08x}:0:", crc32(b"cbor\x00"));
        assert_eq!(seal(Format::Cbor, b""), want.into_bytes());
        assert_ne!(seal(Format::Cbor, b"x"), seal(Format::MsgPack, b"x"));
        // the length is plain decimal, however many digits it needs
        for n in [0usize, 1, 9, 10, 99, 100, 12_345] {
            let blob = vec![b'a'; n];
            let s = seal(Format::Bencode, &blob);
            assert_eq!(s.len(), 5 + 8 + 1 + n.to_string().len() + 1 + n);
            assert_eq!(open(Format::Bencode, &s).unwrap(), &blob[..]);
        }
        // binary payloads (colons included) and trailing whitespace
        let blob: Vec<u8> = (0..=255u8).collect();
        let mut s = seal(Format::Cbor, &blob);
        assert_eq!(open(Format::Cbor, &s).unwrap(), &blob[..]);
        s.extend_from_slice(b" \r\n\t\x0c");
        assert_eq!(open(Format::Cbor, &s).unwrap(), &blob[..]);
    }

    #[test]
    fn damaged_envelopes_are_refused() {
        let c = format!("{:08x}", crc32(b"json\x00123456789"));
        let env = |crc: &str, len: &str, body: &str| {
            format!("PTG1:{crc}:{len}:{body}").into_bytes()
        };
        let ok = env(&c, "9", "123456789");
        assert_eq!(open(Format::Json, &ok).unwrap(), &b"123456789"[..]);
        let other = format!("{:08x}", crc32(b"json\x00123456780"));
        let bad: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            b"PTG1".to_vec(),
            b"PTG1:".to_vec(),
            format!(" PTG1:{c}:9:123456789").into_bytes(), // leading whitespace
            format!("ptg1:{c}:9:123456789").into_bytes(),
            format!("PTG2:{c}:9:123456789").into_bytes(),
            env(&c[..7], "9", "123456789"),                  // seven digits
            env(&format!("{c}0"), "9", "123456789"),         // nine digits
            env(&format!("{}g", &c[..7]), "9", "123456789"), // not hex
            env(&c, "09", "123456789"),                      // leading zero
            env(&c, "", "123456789"),                        // no length
            env(&c, "+9", "123456789"),
            env(&c, "10", "123456789"),                      // longer than the data
            env(&c, "8", "123456789"),                       // data left over
            env(&c, "9", "12345678"),
            env(&c, "9", "123456789x"),
            env(&c, "9", "123456780"),                       // damaged payload
            env(&other, "9", "123456789"),                   // checksum of another payload
            env(&c, "99999999999999999999999", "1"),         // too many digits
            env(&c, "18446744073709551616", ""),             // overflows usize
            env(&c, "18446744073709551615", "1"),            // huge claim, no data
            format!("PTG1:{c}:9").into_bytes(),
            format!("PTG1:{c}:9;123456789").into_bytes(),
        ];
        for b in bad.iter() {
            assert!(
                open(Format::Json, b).is_err(),
                "accepted {:?}",
                String::from_utf8_lossy(b)
            );
        }
        // the same bytes under another format name have another checksum
        for f in Format::ALL {
            if f != Format::Json {
                assert!(open(f, &ok).is_err(), "{}", f.name());
            }
        }
        let mut padded = ok.clone();
        padded.extend_from_slice(b" \r\n");
        assert!(open(Format::Json, &padded).is_ok());
        padded.push(b'x');
        assert!(open(Format::Json, &padded).is_err());
    }

    #[test]
    fn spec_strings_parse_and_print() {
        let s = "cbor|delta|lzss|xor:00000000000000ff|z85@76";
        let p = spec_of(s);
        assert_eq!(p.format, Format::Cbor);
        assert_eq!(p.stages, vec![Stage::Delta, Stage::Lzss, Stage::Xor(0xff)]);
        assert_eq!((p.armor, p.wrap), (Armor::Z85, 76));
        assert_eq!(p.spec(), s);
        assert_eq!(p.to_string(), s);
        for t in [
            "json|b64",
            "jsonp|hex",
            "sexp|raw",
            "bencode|rev|rev|b32@40",
            "msgpack|mtf|words",
            "json|raw",
            "json|b58@4096",
        ] {
            assert_eq!(spec_of(t).spec(), t);
        }
        // short keys are padded, whitespace around the whole spec is ignored
        assert_eq!(spec_of(" json|xor:ff|hex\n").spec(), "json|xor:00000000000000ff|hex");
        // every format and armor pair: legal ones print and parse back, illegal ones are refused
        for f in Format::ALL {
            for a in Armor::ALL {
                let t = format!("{}|{}", f.name(), a.name());
                match Pipeline::new(f, Vec::new(), a, 0) {
                    Ok(p) => assert_eq!(Pipeline::parse(&t).unwrap(), p, "{t}"),
                    Err(_) => assert!(Pipeline::parse(&t).is_err(), "{t}"),
                }
            }
        }
    }

    #[test]
    fn bad_specs_are_refused_with_a_reason() {
        let many = format!("json|{}b64", "rev|".repeat(MAX_STAGES + 1));
        let long_key = format!("json|xor:{}|b64", "0".repeat(17));
        for (bad, why) in [
            ("", "format and an armor"),
            ("json", "format and an armor"),
            ("|", "unknown format"),
            ("yaml|b64", "unknown format"),
            ("JSON|b64", "unknown format"),
            ("json|", "unknown armor"),
            ("json|base64", "unknown armor"),
            ("json|B64", "unknown armor"),
            ("json||b64", "unknown stage"),
            ("json|gzip|b64", "unknown stage"),
            ("json|xor|b64", "unknown stage"),
            ("json|xor:|b64", "unknown stage"),
            ("json|xor:zz|b64", "unknown stage"),
            (long_key.as_str(), "unknown stage"),
            ("json|rle@5|b64", "unknown stage"),
            ("json|b64@", "bad wrap"),
            ("json|b64@0", "bad wrap"),
            ("json|b64@07", "bad wrap"),
            ("json|b64@-1", "bad wrap"),
            ("json|b64@x", "bad wrap"),
            ("json|b64@4097", "bad wrap"),
            ("json|b64@99999999999999999999", "bad wrap"),
            ("json|b64@7@8", "bad wrap"),
            ("cbor|raw", "text format"),
            ("msgpack|raw", "text format"),
            ("json|rle|raw", "cannot follow stages"),
            ("json|raw@40", "no wrap"),
            (many.as_str(), "too many stages"),
        ] {
            let err = Pipeline::parse(bad).unwrap_err();
            assert!(err.contains(why), "{bad:?}: {err}");
        }
    }

    #[test]
    fn layers_run_in_the_documented_order() {
        let v = Value::obj().with("a", Value::seq([1, 2, 3]));
        // peel `json|delta|rev|hex` by hand: armor first, then the stages in reverse
        let text = spec_of("json|delta|rev|hex").encode(&v).unwrap();
        let mut bytes = Armor::Hex.decode(&text).unwrap();
        bytes = Stage::Rev.undo(&bytes).unwrap();
        bytes = Stage::Delta.undo(&bytes).unwrap();
        let payload = open(Format::Json, &bytes).unwrap();
        assert_eq!(payload, &br#"{"a":[1,2,3]}"#[..]);
        // `raw` leaves the envelope readable
        let plain = spec_of("json|raw").encode(&v).unwrap();
        let crc = crc32(b"json\x00{\"a\":[1,2,3]}");
        assert_eq!(plain, format!("PTG1:{:08x}:13:{{\"a\":[1,2,3]}}", crc));
    }

    #[test]
    fn every_format_and_armor_pair_round_trips() {
        let mut rng = Rng::new(21);
        for f in Format::ALL {
            for a in Armor::ALL {
                let legal = a != Armor::Raw || f.is_text();
                assert_eq!(
                    Pipeline::new(f, vec![], a, 0).is_ok(),
                    legal,
                    "{}|{}",
                    f.name(),
                    a.name()
                );
                if !legal {
                    continue;
                }
                let wraps: &[usize] = if a == Armor::Raw { &[0] } else { &[0, 40] };
                for &wrap in wraps {
                    let p = Pipeline::new(f, vec![], a, wrap).unwrap();
                    let mut values = vec![sample()];
                    for _ in 0..8 {
                        values.push(random_value(&mut rng, 4));
                    }
                    for v in values.iter() {
                        let text = p.encode(v).unwrap();
                        if a != Armor::Raw {
                            assert!(text.is_ascii(), "{p}");
                        }
                        if wrap > 0 {
                            assert!(text.lines().all(|l| l.len() <= wrap), "{p}");
                        }
                        let back = p.decode(&text).unwrap_or_else(|e| panic!("{p}: {e}"));
                        assert!(back.same(v), "{p} changed {v:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn every_stage_and_armor_combine() {
        let mut rng = Rng::new(22);
        let stages = [
            Stage::Rle,
            Stage::Lzss,
            Stage::Delta,
            Stage::Rev,
            Stage::Mtf,
            Stage::Xor(0x1234_5678_9abc_def0),
        ];
        for st in stages {
            for a in Armor::ALL {
                if a == Armor::Raw {
                    assert!(Pipeline::new(Format::Json, vec![st], a, 0).is_err());
                    continue;
                }
                let p = Pipeline::new(Format::Cbor, vec![st], a, 64).unwrap();
                let mut values = vec![sample()];
                for _ in 0..4 {
                    values.push(random_value(&mut rng, 4));
                }
                for v in values.iter() {
                    let text = p.encode(v).unwrap();
                    assert!(p.decode(&text).unwrap().same(v), "{p}");
                }
            }
        }
        // a hand-built illegal pipeline fails cleanly in both directions
        let bad = Pipeline { format: Format::Cbor, stages: vec![], armor: Armor::Raw, wrap: 0 };
        assert!(bad.encode(&sample()).is_err());
        assert!(bad.decode("anything").is_err());
    }

    #[test]
    fn large_values_survive_long_pipelines() {
        let rows: Vec<Value> = (0..400i32)
            .map(|i| {
                Value::obj()
                    .with("id", i)
                    .with("name", "item-number")
                    .with("ok", i % 3 == 0)
                    .with("w", Value::seq([i, i / 2, i * 7]))
            })
            .collect();
        let v = Value::obj()
            .with("rows", Value::List(rows))
            .with("note", "x".repeat(5000));
        for t in [
            "json|raw",
            "jsonp|lzss|b64@76",
            "sexp|rle|delta|b32@64",
            "bencode|mtf|b58",
            "cbor|lzss|xor:0123456789abcdef|z85@72",
            "msgpack|rev|delta|rle|hex@100",
            "json|lzss|mtf|words@80",
        ] {
            let p = spec_of(t);
            let text = p.encode(&v).unwrap();
            let back = p.decode(&text).unwrap_or_else(|e| panic!("{t}: {e}"));
            assert!(back.same(&v), "{t}");
        }
    }

    #[test]
    fn random_pipelines_are_legal_and_round_trip() {
        let mut rng = Rng::new(24);
        for _ in 0..300 {
            let p = Pipeline::random(&mut rng);
            p.check().unwrap();
            assert!(p.stages.len() <= 4, "{p}");
            assert!(p.stages.windows(2).all(|w| w[0].kind() != w[1].kind()), "{p}");
            assert_eq!(Pipeline::parse(&p.spec()).unwrap(), p);
            assert_eq!(p.to_string(), p.spec());
            let v = random_value(&mut rng, 4);
            let text = p.encode(&v).unwrap();
            let back = p.decode(&text).unwrap_or_else(|e| panic!("{p}: {e}"));
            assert!(back.same(&v), "{p} changed {v:?}");
        }
    }

    #[test]
    fn the_generator_is_deterministic_and_covers_every_choice() {
        let (mut a, mut b) = (Rng::new(5), Rng::new(5));
        let mut formats = [false; 6];
        let mut armors = [false; 8];
        let mut kinds = [false; 6];
        let mut lens = [false; 5];
        let (mut wrapped, mut unwrapped) = (false, false);
        for _ in 0..2000 {
            let p = Pipeline::random(&mut a);
            assert_eq!(p, Pipeline::random(&mut b));
            formats[Format::ALL.iter().position(|&f| f == p.format).unwrap()] = true;
            armors[Armor::ALL.iter().position(|&x| x == p.armor).unwrap()] = true;
            for s in p.stages.iter() {
                kinds[s.kind()] = true;
            }
            lens[p.stages.len()] = true;
            if p.wrap > 0 {
                wrapped = true;
            } else {
                unwrapped = true;
            }
        }
        assert!(formats.iter().all(|&x| x) && armors.iter().all(|&x| x));
        assert!(kinds.iter().all(|&x| x) && lens.iter().all(|&x| x));
        assert!(wrapped && unwrapped);
    }

    #[test]
    fn pasted_bodies_may_carry_trailing_whitespace() {
        let v = sample();
        for t in ["json|raw", "jsonp|raw", "sexp|raw", "bencode|raw", "cbor|hex@40", "msgpack|b58@60"] {
            let p = spec_of(t);
            let mut text = p.encode(&v).unwrap();
            text.push_str(" \r\n\n\t");
            assert!(p.decode(&text).unwrap().same(&v), "{t}");
        }
        let p = spec_of("json|raw");
        let mut text = p.encode(&v).unwrap();
        text.push_str(" x");
        assert!(p.decode(&text).is_err());
    }

    #[test]
    fn corruption_is_caught_or_harmless() {
        let mut rng = Rng::new(23);
        let (mut caught, mut total) = (0usize, 0usize);
        for _ in 0..500 {
            let p = Pipeline::random(&mut rng);
            let v = random_value(&mut rng, 4);
            let text = p.encode(&v).unwrap();
            let bad = mutate(&mut rng, &text);
            total += 1;
            match p.decode(&bad) {
                Ok(w) => assert!(w.same(&v), "{p}: corrupted text decoded to another value"),
                Err(_) => caught += 1,
            }
        }
        assert!(caught * 2 > total, "only {caught} of {total} corruptions were caught");
    }

    #[test]
    fn a_wrong_spec_is_refused() {
        let v = Value::obj().with("a", Value::seq([1, 2, 3])).with("b", "text");
        for (right, wrong) in [
            ("json|xor:0000000000000001|b64", "json|xor:0000000000000002|b64"),
            ("json|rev|hex", "json|hex"),
            ("json|hex", "json|rev|hex"),
            ("json|delta|hex", "json|hex"),
            ("json|delta|rev|hex", "json|rev|delta|hex"),
            ("json|b64", "json|hex"),
            // same bytes, other format: the checksum names the format
            ("cbor|hex", "msgpack|hex"),
            ("msgpack|hex", "cbor|hex"),
            ("json|hex", "jsonp|hex"),
            ("cbor|hex", "json|hex"),
            ("sexp|raw", "json|raw"),
        ] {
            let text = spec_of(right).encode(&v).unwrap();
            assert!(spec_of(right).decode(&text).unwrap().same(&v), "{right}");
            assert!(spec_of(wrong).decode(&text).is_err(), "{wrong} read what {right} wrote");
        }
        // random pairs: refused, or at least never read as another value
        let mut rng = Rng::new(25);
        let (mut pairs, mut refused) = (0usize, 0usize);
        for _ in 0..300 {
            let (p, q) = (Pipeline::random(&mut rng), Pipeline::random(&mut rng));
            if p == q {
                continue;
            }
            let v = random_value(&mut rng, 3);
            let text = p.encode(&v).unwrap();
            pairs += 1;
            match q.decode(&text) {
                Ok(w) => assert!(w.same(&v), "{q} read the text of {p} as another value"),
                Err(_) => refused += 1,
            }
        }
        assert!(refused * 10 >= pairs * 9, "only {refused} of {pairs} wrong specs were refused");
    }

    #[test]
    fn noise_never_panics() {
        let mut rng = Rng::new(26);
        for _ in 0..300 {
            let p = Pipeline::random(&mut rng);
            let n = rng.below(80);
            let junk: String = (0..n).map(|_| (b' ' + rng.below(95) as u8) as char).collect();
            assert!(p.decode(&junk).is_err(), "{p} accepted noise");
        }
    }
}