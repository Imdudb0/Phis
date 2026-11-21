//! Serialization formats of the codec engine, written from scratch.
//!
//! A `Format` turns a `Value` into bytes and back without loss: `decode(encode(v))`
//! is the identity, with floats bit for bit, maps in insertion order and duplicate
//! keys kept. Decoders are hardened against hostile input: nesting is capped at
//! `MAX_DEPTH`, claimed lengths are checked against the bytes actually left, and
//! no input makes them panic.
//!
//! | name      | kind   | shape                                                        |
//! |-----------|--------|--------------------------------------------------------------|
//! | `json`    | text   | compact JSON                                                 |
//! | `jsonp`   | text   | JSON with a two-space indent                                 |
//! | `sexp`    | text   | `nil`, `#t`, `#f`, `(list ..)`, `(map ("k" v) ..)`           |
//! | `bencode` | text   | bencode plus `n`, `t`, `f` and `r<16 hex digits>e` (a float) |
//! | `cbor`    | binary | RFC 8949 subset: definite lengths, floats as float64         |
//! | `msgpack` | binary | MessagePack subset, always the smallest int/str/array/map    |
//!
//! `json`, `jsonp`, `cbor` and `msgpack` write what standard tools accept; `sexp`
//! and `bencode` are this engine's own dialects. The code lives in two private
//! submodules: `text` (JSON, S-expressions) and `bin` (bencode, CBOR, MessagePack).

mod bin;
//mod text;
#[cfg(test)]
mod tests;

use crate::PTGRS::rng::Rng;
use crate::PTGRS::value::Value;

/// Deepest nesting (`Value::depth()`) that encoders accept and decoders allow.
pub const MAX_DEPTH: usize = 128;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Json,
    JsonPretty,
    Sexp,
    Bencode,
    Cbor,
    MsgPack,
}

impl Format {
    pub const ALL: [Format; 6] = [
        Format::Json,
        Format::JsonPretty,
        Format::Sexp,
        Format::Bencode,
        Format::Cbor,
        Format::MsgPack,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Format::Json => "json",
            Format::JsonPretty => "jsonp",
            Format::Sexp => "sexp",
            Format::Bencode => "bencode",
            Format::Cbor => "cbor",
            Format::MsgPack => "msgpack",
        }
    }

    pub fn parse(s: &str) -> Option<Format> {
        Format::ALL.iter().copied().find(|f| f.name() == s)
    }

    pub fn random(rng: &mut Rng) -> Format {
        *rng.pick(&Format::ALL)
    }

    /// Text formats always yield valid UTF-8, so `Armor::Raw` can carry them.
    pub fn is_text(self) -> bool {
        !matches!(self, Format::Cbor | Format::MsgPack)
    }

    /// `Err` only for a value nested deeper than `MAX_DEPTH` (the decoders would
    /// refuse it) or, in MessagePack, a container beyond 2^32 - 1 entries.
    pub fn encode(self, v: &Value) -> Result<Vec<u8>, String> {
        if v.depth() > MAX_DEPTH {
            return Err(format!("value nests deeper than {MAX_DEPTH} levels"));
        }
        match self {
            Format::Json => Ok(text::json_encode(v, 0).into_bytes()),
            Format::JsonPretty => Ok(text::json_encode(v, 2).into_bytes()),
            Format::Sexp => Ok(text::sexp_encode(v).into_bytes()),
            Format::Bencode => Ok(bin::bencode_encode(v)),
            Format::Cbor => Ok(bin::cbor_encode(v)),
            Format::MsgPack => bin::msgpack_encode(v),
        }
    }

    /// The whole input must be exactly one value.
    pub fn decode(self, data: &[u8]) -> Result<Value, String> {
        match self {
            Format::Json | Format::JsonPretty | Format::Sexp => {
                let s = std::str::from_utf8(data)
                    .map_err(|_| "text is not valid UTF-8".to_string())?;
                if self == Format::Sexp {
                    text::sexp_decode(s)
                } else {
                    text::json_decode(s)
                }
            }
            Format::Bencode => bin::bencode_decode(data),
            Format::Cbor => bin::cbor_decode(data),
            Format::MsgPack => bin::msgpack_decode(data),
        }
    }
}