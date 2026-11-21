//! Cross-format tests of the codec facade.
//!
//! * known answers: CBOR from RFC 8949 Appendix A (the part of it this engine
//!   reads), MessagePack from its specification, bencode from BEP 3 plus this
//!   engine's extensions, and one value spelled out in all six formats;
//! * exact round trips of awkward and of random values;
//! * every proper prefix and a padded tail are refused, nesting and size bombs
//!   are refused, and noise never panics.

use super::{bin, text, Format, MAX_DEPTH};
use crate::rng::Rng;
use crate::value::{random_value, Value};

// ------------------------------------------------------------------ helpers

/// Hex literal to bytes; ASCII whitespace is ignored.
fn hex(s: &str) -> Vec<u8> {
    let d: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    assert!(d.len() % 2 == 0, "odd hex literal {s:?}");
    d.chunks(2)
        .map(|p| {
            let h = (p[0] as char).to_digit(16).expect("hex digit");
            let l = (p[1] as char).to_digit(16).expect("hex digit");
            (h * 16 + l) as u8
        })
        .collect()
}

fn finite(v: &Value) -> bool {
    match v {
        Value::Float(x) => x.is_finite(),
        Value::List(l) => l.iter().all(finite),
        Value::Map(m) => m.iter().all(|(_, x)| finite(x)),
        _ => true,
    }
}

/// `v` encodes to exactly `want`, and `want` decodes back to `v`.
fn kat(f: Format, v: &Value, want: &[u8]) {
    let got = f.encode(v).unwrap();
    assert_eq!(got, want, "{} writes {v:?} differently", f.name());
    let back = f.decode(want).unwrap();
    assert!(back.same(v), "{} reads {want:?} as {back:?}", f.name());
}

/// The hex input decodes to `v` (an encoding this engine never writes itself).
fn reads(f: Format, h: &str, v: &Value) {
    let back = f
        .decode(&hex(h))
        .unwrap_or_else(|e| panic!("{} rejected {h}: {e}", f.name()));
    assert!(back.same(v), "{} reads {h} as {back:?}", f.name());
}

/// Every hex input is refused.
fn refuses(f: Format, inputs: &[&str]) {
    for h in inputs {
        assert!(f.decode(&hex(h)).is_err(), "{} accepted {h}", f.name());
    }
}

/// `v` starts with the bytes `head` in format `f` and decodes back.
fn starts(f: Format, v: &Value, head: &str) {
    let enc = f.encode(v).unwrap();
    let h = hex(head);
    assert_eq!(&enc[..h.len()], &h[..], "{} header", f.name());
    assert!(f.decode(&enc).unwrap().same(v), "{} round trip", f.name());
}

/// A value of exactly `depth` levels: maps and lists alternating around a null.
fn nested(depth: usize) -> Value {
    let mut v = Value::Null;
    for k in 1..depth {
        v = if k % 2 == 0 {
            Value::List(vec![v])
        } else {
            Value::Map(vec![("k".to_string(), v)])
        };
    }
    v
}

/// What each format would write if the facade did not enforce the depth cap.
fn raw_encode(f: Format, v: &Value) -> Vec<u8> {
    match f {
        Format::Json => text::json_encode(v, 0).into_bytes(),
        Format::JsonPretty => text::json_encode(v, 2).into_bytes(),
        Format::Sexp => text::sexp_encode(v).into_bytes(),
        Format::Bencode => bin::bencode_encode(v),
        Format::Cbor => bin::cbor_encode(v),
        Format::MsgPack => bin::msgpack_encode(v).unwrap(),
    }
}

fn awkward() -> Vec<Value> {
    vec![
        Value::Null,
        Value::Bool(true),
        Value::Bool(false),
        Value::Int(0),
        Value::Int(-1),
        Value::Int(i64::MAX),
        Value::Int(i64::MIN),
        Value::Float(0.0),
        Value::Float(-0.0),
        Value::Float(f64::MAX),
        Value::Float(-f64::MAX),
        Value::Float(f64::MIN_POSITIVE),
        Value::Float(5e-324),
        Value::Float(0.1),
        Value::Float(1e21),
        Value::Float(123_456_789.125),
        Value::from(""),
        Value::from("\u{0}\u{1f}\u{7f}\"\\/\n\r\t"),
        Value::from("é漢😀\u{2028}"),
        Value::from("nil #t (list) [1,2] {\"a\":1} i1e 3:abc"),
        Value::arr(),
        Value::obj(),
        // order and duplicate keys must survive
        Value::obj().with("b", 1).with("a", 2).with("b", 3),
        Value::obj().with("", Value::Null).with("", Value::Null),
        Value::List(vec![Value::arr(), Value::obj(), Value::List(vec![Value::Null])]),
    ]
}

// ------------------------------------------------------------- the registry

#[test]
fn names_parse_back_and_random_covers_all() {
    let mut names: Vec<&str> = Format::ALL.iter().map(|f| f.name()).collect();
    for f in Format::ALL {
        assert_eq!(Format::parse(f.name()), Some(f));
    }
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), Format::ALL.len());
    assert_eq!(Format::parse("yaml"), None);
    assert_eq!(Format::parse(""), None);
    let mut rng = Rng::new(1);
    let mut seen = [false; 6];
    for _ in 0..300 {
        let f = Format::random(&mut rng);
        seen[Format::ALL.iter().position(|&g| g == f).unwrap()] = true;
    }
    assert!(seen.iter().all(|&b| b));
    assert_eq!(Format::ALL.iter().filter(|f| f.is_text()).count(), 4);
}

// ------------------------------------------------------------ known answers

#[test]
fn cbor_matches_rfc_8949_appendix_a() {
    let f = Format::Cbor;
    let ints: &[(i64, &str)] = &[
        (0, "00"),
        (1, "01"),
        (10, "0a"),
        (23, "17"),
        (24, "1818"),
        (25, "1819"),
        (100, "1864"),
        (255, "18ff"),
        (256, "190100"),
        (1000, "1903e8"),
        (65_535, "19ffff"),
        (65_536, "1a00010000"),
        (1_000_000, "1a000f4240"),
        (4_294_967_295, "1affffffff"),
        (4_294_967_296, "1b0000000100000000"),
        (1_000_000_000_000, "1b000000e8d4a51000"),
        (i64::MAX, "1b7fffffffffffffff"),
        (-1, "20"),
        (-10, "29"),
        (-24, "37"),
        (-25, "3818"),
        (-100, "3863"),
        (-256, "38ff"),
        (-257, "390100"),
        (-1000, "3903e7"),
        (-4_294_967_296, "3affffffff"),
        (i64::MIN, "3b7fffffffffffffff"),
    ];
    for &(i, h) in ints {
        kat(f, &Value::Int(i), &hex(h));
    }
    // floats are always written as float64
    let floats: &[(f64, &str)] = &[
        (0.0, "fb0000000000000000"),
        (-0.0, "fb8000000000000000"),
        (1.1, "fb3ff199999999999a"),
        (1.0e300, "fb7e37e43c8800759c"),
        (-4.1, "fbc010666666666666"),
    ];
    for &(x, h) in floats {
        kat(f, &Value::Float(x), &hex(h));
    }
    kat(f, &Value::Bool(false), &hex("f4"));
    kat(f, &Value::Bool(true), &hex("f5"));
    kat(f, &Value::Null, &hex("f6"));
    let strings = [
        ("", "60"),
        ("a", "6161"),
        ("IETF", "6449455446"),
        ("\"\\", "62225c"),
        ("\u{fc}", "62c3bc"),
        ("\u{6c34}", "63e6b0b4"),
        ("\u{10151}", "64f0908591"),
    ];
    for (t, h) in strings {
        kat(f, &Value::from(t), &hex(h));
    }
    kat(f, &Value::arr(), &hex("80"));
    kat(f, &Value::seq([1, 2, 3]), &hex("83010203"));
    let tree = Value::List(vec![Value::Int(1), Value::seq([2, 3]), Value::seq([4, 5])]);
    kat(f, &tree, &hex("8301820203820405"));
    kat(
        f,
        &Value::seq(1..=25i32),
        &hex("98190102030405060708090a0b0c0d0e0f101112131415161718181819"),
    );
    kat(f, &Value::obj(), &hex("a0"));
    let ab = Value::obj().with("a", 1).with("b", Value::seq([2, 3]));
    kat(f, &ab, &hex("a26161016162820203"));
    let mixed = Value::List(vec![Value::from("a"), Value::obj().with("b", "c")]);
    kat(f, &mixed, &hex("826161a161626163"));
    let mut abcde = Value::obj();
    for (k, v) in [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D"), ("e", "E")] {
        abcde.put(k, v);
    }
    kat(f, &abcde, &hex("a56161614161626142616361436164614461656145"));

    // read-only: heads that are not the shortest, float16 and float32
    reads(f, "1800", &Value::Int(0));
    reads(f, "1b0000000000000005", &Value::Int(5));
    reads(f, "9800", &Value::arr());
    reads(f, "f90000", &Value::Float(0.0));
    reads(f, "f98000", &Value::Float(-0.0));
    reads(f, "f93c00", &Value::Float(1.0));
    reads(f, "f93e00", &Value::Float(1.5));
    reads(f, "f97bff", &Value::Float(65504.0));
    reads(f, "f90001", &Value::Float(f64::from_bits(0x3e70_0000_0000_0000))); // 2^-24
    reads(f, "f90400", &Value::Float(0.000_061_035_156_25));
    reads(f, "f9c400", &Value::Float(-4.0));
    reads(f, "fa47c35000", &Value::Float(100_000.0));
    reads(f, "fa7f7fffff", &Value::Float(f64::from(f32::MAX)));
    reads(f, "fb3ff199999999999a", &Value::Float(1.1));

    refuses(
        f,
        &[
            "f97c00", "f9fc00", "f97e00", // float16 infinities and NaN
            "fa7f800000", "faff800000", "fa7fc00000", // float32
            "fb7ff0000000000000", "fbfff0000000000000", "fb7ff8000000000000", // float64
            "f7", "f0", "f8ff", "ff", // undefined, simple values, a stray break
            "40", "4401020304", // byte strings
            "c074323031332d30332d32315432303a30343a30305a", // a tag
            "5fff", "7f657374726561646d696e67ff", "9fff", "9f018202039f0405ffff", // indefinite
            "1bffffffffffffffff", "3bffffffffffffffff", // beyond i64
            "a201020304", // integer map keys
            "1c", "1d", "1e", // reserved additional information
            "62c328", "61ff", // invalid UTF-8
            "18", "1900", "1a000000", "1b00000000000000", // truncated heads
            "61", "6200", "8201", "a16161", // truncated bodies
            "0000", // two values
            "",
        ],
    );
}

#[test]
fn msgpack_matches_the_specification() {
    let f = Format::MsgPack;
    let ints: &[(i64, &str)] = &[
        (0, "00"),
        (127, "7f"),
        (128, "cc80"),
        (255, "ccff"),
        (256, "cd0100"),
        (65_535, "cdffff"),
        (65_536, "ce00010000"),
        (4_294_967_295, "ceffffffff"),
        (4_294_967_296, "cf0000000100000000"),
        (i64::MAX, "cf7fffffffffffffff"),
        (-1, "ff"),
        (-32, "e0"),
        (-33, "d0df"),
        (-128, "d080"),
        (-129, "d1ff7f"),
        (-32_768, "d18000"),
        (-32_769, "d2ffff7fff"),
        (-2_147_483_648, "d280000000"),
        (-2_147_483_649, "d3ffffffff7fffffff"),
        (i64::MIN, "d38000000000000000"),
    ];
    for &(i, h) in ints {
        kat(f, &Value::Int(i), &hex(h));
    }
    kat(f, &Value::Null, &hex("c0"));
    kat(f, &Value::Bool(false), &hex("c2"));
    kat(f, &Value::Bool(true), &hex("c3"));
    kat(f, &Value::Float(1.5), &hex("cb3ff8000000000000"));
    kat(f, &Value::from(""), &hex("a0"));
    kat(f, &Value::from("a"), &hex("a161"));
    kat(f, &Value::from("IETF"), &hex("a449455446"));
    kat(f, &Value::from("\u{fc}"), &hex("a2c3bc"));
    kat(f, &Value::arr(), &hex("90"));
    kat(f, &Value::seq([1, 2, 3]), &hex("93010203"));
    let tree = Value::List(vec![Value::Int(1), Value::seq([2, 3]), Value::seq([4, 5])]);
    kat(f, &tree, &hex("9301920203920405"));
    kat(f, &Value::obj(), &hex("80"));
    let ab = Value::obj().with("a", 1).with("b", Value::seq([2, 3]));
    kat(f, &ab, &hex("82a16101a1629203"));
    let mixed = Value::List(vec![Value::from("a"), Value::obj().with("b", "c")]);
    kat(f, &mixed, &hex("92a16181a162a163"));

    // read-only: wider-than-needed forms and float32
    reads(f, "cd0001", &Value::Int(1));
    reads(f, "cf0000000000000001", &Value::Int(1));
    reads(f, "d0ff", &Value::Int(-1));
    reads(f, "d1ffff", &Value::Int(-1));
    reads(f, "d3ffffffffffffffff", &Value::Int(-1));
    reads(f, "d200000005", &Value::Int(5));
    reads(f, "d903616263", &Value::from("abc"));
    reads(f, "dc000101", &Value::seq([1]));
    reads(f, "de0001a16101", &Value::obj().with("a", 1));
    reads(f, "ca3fc00000", &Value::Float(1.5));

    refuses(
        f,
        &[
            "c1", // never used
            "c40100", "c5000100", "c6000000 0100", // bin 8/16/32
            "c7", "c8", "c9", "d40100", "d5", "d6", "d7", "d8", // ext types
            "cf8000000000000000", "cfffffffffffffffff", // uint64 above i64::MAX
            "cbfff0000000000000", "cb7ff0000000000000", "cb7ff8000000000000", // float64
            "ca7f800000", "caff800000", "ca7fc00000", // float32
            "810101", "81c0c0", "819001", // non-string keys
            "a1ff", "d901ff", // invalid UTF-8
            "cd01", "ce000100", "cf00", "d0", "d100", "ca3fc000", "cb3ff80000", // truncated
            "a261", "d90561", "dc000201", "dd00000002c0", "de0001a161", "81a161",
            "c0c0", // two values
            "",
        ],
    );
}

#[test]
fn bencode_matches_bep_3_and_its_extensions() {
    let f = Format::Bencode;
    kat(f, &Value::from("spam"), b"4:spam");
    kat(f, &Value::from(""), b"0:");
    kat(f, &Value::from("\u{e9}"), "2:\u{e9}".as_bytes());
    kat(f, &Value::Int(3), b"i3e");
    kat(f, &Value::Int(-3), b"i-3e");
    kat(f, &Value::Int(0), b"i0e");
    kat(f, &Value::Int(i64::MIN), b"i-9223372036854775808e");
    kat(f, &Value::Int(i64::MAX), b"i9223372036854775807e");
    kat(f, &Value::seq(["spam", "eggs"]), b"l4:spam4:eggse");
    let cow = Value::obj().with("cow", "moo").with("spam", "eggs");
    kat(f, &cow, b"d3:cow3:moo4:spam4:eggse");
    let nest = Value::obj().with("spam", Value::seq(["a", "b"]));
    kat(f, &nest, b"d4:spaml1:a1:bee");
    kat(f, &Value::arr(), b"le");
    kat(f, &Value::obj(), b"de");
    // the extensions: null, booleans, floats as the hex of their bits
    kat(f, &Value::Null, b"n");
    kat(f, &Value::Bool(true), b"t");
    kat(f, &Value::Bool(false), b"f");
    kat(f, &Value::Float(1.5), b"r3ff8000000000000e");
    kat(f, &Value::Float(-0.0), b"r8000000000000000e");
    // duplicate keys are kept, in order
    let dup = Value::obj().with("a", 1).with("a", 2);
    kat(f, &dup, b"d1:ai1e1:ai2ee");

    for bad in [
        "", "i-0e", "i05e", "i00e", "ie", "i-e", "i+3e", "i3", "i3ex", "i12345678901234567890e",
        "i9223372036854775808e", "04:spam", "4:spa", "4spam", ":", "-1:a", "l", "li1e", "d", "d1:a",
        "d1:ai1e", "di1ei2ee", "l4:spam4:eggs", "4:spam4:eggs", "x", "e", "r3ff800000000000e",
        "r3ff8000000000000", "rzzzzzzzzzzzzzzze", "r7ff8000000000000e", "rfff0000000000000e",
    ] {
        assert!(f.decode(bad.as_bytes()).is_err(), "bencode accepted {bad:?}");
    }
    assert!(f.decode(b"1:\xff").is_err()); // invalid UTF-8
}

#[test]
fn one_value_in_all_six_formats() {
    let v = Value::obj().with("compact", true).with("schema", 0);
    let table: [(Format, Vec<u8>); 6] = [
        (Format::Json, br#"{"compact":true,"schema":0}"#.to_vec()),
        (
            Format::JsonPretty,
            b"{\n  \"compact\": true,\n  \"schema\": 0\n}".to_vec(),
        ),
        (Format::Sexp, br##"(map ("compact" #t) ("schema" 0))"##.to_vec()),
        (Format::Bencode, b"d7:compactt6:schemai0ee".to_vec()),
        (Format::Cbor, hex("a267636f6d70616374f566736368656d6100")),
        (Format::MsgPack, hex("82a7636f6d70616374c3a6736368656d6100")),
    ];
    for (f, bytes) in table.iter() {
        kat(*f, &v, bytes);
    }
}

#[test]
fn length_heads_use_the_smallest_form() {
    let text_of = |n: usize| Value::Str("a".repeat(n));
    let list_of = |n: usize| Value::List(vec![Value::Null; n]);
    let map_of = |n: usize| {
        Value::Map((0..n).map(|i| (i.to_string(), Value::Null)).collect())
    };
    let mp = Format::MsgPack;
    for (n, h) in [
        (0, "a0"),
        (31, "bf"),
        (32, "d920"),
        (255, "d9ff"),
        (256, "da0100"),
        (65_535, "daffff"),
        (65_536, "db00010000"),
    ] {
        starts(mp, &text_of(n), h);
    }
    for (n, h) in [(0, "90"), (15, "9f"), (16, "dc0010"), (65_535, "dcffff"), (65_536, "dd00010000")] {
        starts(mp, &list_of(n), h);
    }
    for (n, h) in [(0, "80"), (15, "8f"), (16, "de0010"), (65_535, "deffff"), (65_536, "df00010000")] {
        starts(mp, &map_of(n), h);
    }
    let cb = Format::Cbor;
    for (n, h) in [
        (0, "60"),
        (23, "77"),
        (24, "7818"),
        (255, "78ff"),
        (256, "790100"),
        (65_535, "79ffff"),
        (65_536, "7a00010000"),
    ] {
        starts(cb, &text_of(n), h);
    }
    for (n, h) in [
        (0, "80"),
        (23, "97"),
        (24, "9818"),
        (255, "98ff"),
        (256, "990100"),
        (65_535, "99ffff"),
        (65_536, "9a00010000"),
    ] {
        starts(cb, &list_of(n), h);
    }
    for (n, h) in [
        (0, "a0"),
        (23, "b7"),
        (24, "b818"),
        (255, "b8ff"),
        (256, "b90100"),
        (65_535, "b9ffff"),
        (65_536, "ba00010000"),
    ] {
        starts(cb, &map_of(n), h);
    }
}

// -------------------------------------------------------------- round trips

#[test]
fn awkward_values_round_trip_exactly() {
    for v in awkward() {
        for f in Format::ALL {
            let enc = f.encode(&v).unwrap();
            let back = f
                .decode(&enc)
                .unwrap_or_else(|e| panic!("{} on {v:?}: {e}", f.name()));
            assert!(back.same(&v), "{}: {v:?} came back as {back:?}", f.name());
        }
    }
}

#[test]
fn random_values_round_trip_in_every_format() {
    let mut rng = Rng::new(11);
    for round in 0..400 {
        let depth = rng.below(6);
        let v = random_value(&mut rng, depth);
        for f in Format::ALL {
            let enc = f.encode(&v).unwrap();
            if f.is_text() {
                assert!(std::str::from_utf8(&enc).is_ok(), "{} wrote invalid UTF-8", f.name());
            }
            let back = f
                .decode(&enc)
                .unwrap_or_else(|e| panic!("round {round}, {}: {e}", f.name()));
            assert!(back.same(&v), "round {round}: {} changed {v:?}", f.name());
            assert_eq!(f.encode(&back).unwrap(), enc, "{} is not canonical", f.name());
        }
    }
}

#[test]
fn whitespace_is_tolerated_by_json_and_sexp_only() {
    let mut rng = Rng::new(12);
    for _ in 0..100 {
        let v = random_value(&mut rng, 4);
        // the two JSON layouts read each other
        let compact = Format::Json.encode(&v).unwrap();
        let pretty = Format::JsonPretty.encode(&v).unwrap();
        assert!(Format::Json.decode(&pretty).unwrap().same(&v));
        assert!(Format::JsonPretty.decode(&compact).unwrap().same(&v));
        for f in Format::ALL {
            let mut padded = b" \n\t".to_vec();
            padded.extend_from_slice(&f.encode(&v).unwrap());
            padded.extend_from_slice(b" \r\n");
            let loose = matches!(f, Format::Json | Format::JsonPretty | Format::Sexp);
            assert_eq!(f.decode(&padded).is_ok(), loose, "{}", f.name());
        }
    }
}

// ----------------------------------------------------------- hostile input

#[test]
fn every_proper_prefix_and_a_padded_tail_are_refused() {
    let mut rng = Rng::new(13);
    for _ in 0..200 {
        let depth = rng.below(4);
        let v = random_value(&mut rng, depth);
        for f in Format::ALL {
            let enc = f.encode(&v).unwrap();
            // "12" is a valid prefix of "123" in JSON and S-expressions; everything
            // else, and every value in the three formats of bin.rs, is prefix-free
            let bare_number = matches!(f, Format::Json | Format::JsonPretty | Format::Sexp)
                && matches!(v, Value::Int(_) | Value::Float(_));
            if !bare_number {
                for cut in 0..enc.len() {
                    assert!(
                        f.decode(&enc[..cut]).is_err(),
                        "{} accepted the first {cut} of {} bytes of {v:?}",
                        f.name(),
                        enc.len()
                    );
                }
            }
            let mut padded = enc.clone();
            padded.push(0);
            assert!(f.decode(&padded).is_err(), "{} accepted a trailing NUL", f.name());
        }
    }
    assert!(Format::Json.decode(&[0xff]).is_err());
    assert!(Format::Sexp.decode(b"(list \"\xff\")").is_err());
}

#[test]
fn nesting_is_capped_at_max_depth() {
    let ok = nested(MAX_DEPTH);
    let deep = nested(MAX_DEPTH + 1);
    assert_eq!((ok.depth(), deep.depth()), (MAX_DEPTH, MAX_DEPTH + 1));
    for f in Format::ALL {
        let enc = f.encode(&ok).unwrap();
        assert!(f.decode(&enc).unwrap().same(&ok), "{}", f.name());
        assert!(f.encode(&deep).is_err(), "{} wrote a value above the cap", f.name());
        // the raw writers have no cap; the decoders must stop one level later
        let err = f.decode(&raw_encode(f, &deep)).unwrap_err();
        assert!(err.contains("too deep"), "{}: {err}", f.name());
    }
}

#[test]
fn unbalanced_openers_cost_a_bounded_stack() {
    let n = 200_000;
    let bombs: Vec<(Format, Vec<u8>)> = vec![
        (Format::Json, b"[".repeat(n)),
        (Format::Json, b"{\"a\":".repeat(n)),
        (Format::JsonPretty, b"[\n  ".repeat(n)),
        (Format::Sexp, b"(list ".repeat(n)),
        (Format::Sexp, b"(map (\"a\" ".repeat(n)),
        (Format::Bencode, b"l".repeat(n)),
        (Format::Bencode, b"d1:a".repeat(n)),
        (Format::Cbor, vec![0x81u8; n]),
        (Format::Cbor, vec![0xa1u8, 0x61, 0x61].repeat(n)),
        (Format::MsgPack, vec![0x91u8; n]),
        (Format::MsgPack, vec![0x81u8, 0xa1, 0x61].repeat(n)),
    ];
    for (f, data) in bombs.iter() {
        let err = f.decode(data).unwrap_err();
        assert!(err.contains("too deep"), "{}: {err}", f.name());
    }
}

#[test]
fn claimed_sizes_beyond_the_data_are_refused() {
    refuses(
        Format::Cbor,
        &[
            "9b1000000000000000", // array of 2^60 entries
            "bb1000000000000000", // map of 2^60 entries
            "7b1000000000000000", // text of 2^60 bytes
            "9bffffffffffffffff",
            "9a00ffffff",
            "9903e8 00000000000000000000", // 1000 entries, 10 bytes left
        ],
    );
    refuses(
        Format::MsgPack,
        &[
            "ddffffffff", "dfffffffff", "dbffffffff", "dcffff", "deffff", "daffff",
            "dd00000010 00",
        ],
    );
    for bad in [
        "1152921504606846976:abc",
        "18446744073709551616:",
        "99999999999999999999:a",
        "4294967296:a",
    ] {
        assert!(Format::Bencode.decode(bad.as_bytes()).is_err(), "{bad}");
    }
}

#[test]
fn noise_and_mutations_never_panic_and_accepted_input_re_encodes() {
    let mut rng = Rng::new(31);
    for _ in 0..400 {
        let v = random_value(&mut rng, 4);
        for f in Format::ALL {
            let n = rng.below(64);
            let noise: Vec<u8> = (0..n).map(|_| rng.next_u64() as u8).collect();
            let mut m = f.encode(&v).unwrap();
            for _ in 0..rng.range(1, 3) {
                if m.is_empty() {
                    break;
                }
                let i = rng.below(m.len());
                match rng.below(4) {
                    0 => m[i] = rng.next_u64() as u8,
                    1 => {
                        m.remove(i);
                    }
                    2 => m.insert(i, rng.next_u64() as u8),
                    _ => m.truncate(i),
                }
            }
            for data in [noise, m] {
                // may succeed or fail, must not panic; what it accepts is a legal value
                if let Ok(w) = f.decode(&data) {
                    assert!(finite(&w), "{} produced a non-finite float", f.name());
                    let e = f.encode(&w).unwrap();
                    assert!(f.decode(&e).unwrap().same(&w), "{} re-encode", f.name());
                }
            }
        }
    }
}