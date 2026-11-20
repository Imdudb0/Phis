//! Byte-level layers of the codec engine, written from scratch.
//!
//! * `Armor` maps bytes to printable text and back. Every armor except `Raw` and
//!   `Words` ignores ASCII whitespace when decoding, so a body can be wrapped at
//!   any width; for `Words` whitespace separates the words.
//! * `Stage` is a reversible byte filter. A pipeline applies its stages in order
//!   before armoring and undoes them in reverse order after de-armoring.

use crate::rng::Rng;

/// Cap on the size of any decoded buffer; a guard against hostile input.
pub const LIMIT: usize = 64 << 20;

const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64U: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
const B32: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
const Z85: &[u8] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";
const B58: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
/// Characters per base58 block, indexed by the number of bytes in the block.
const B58_CHARS: [usize; 9] = [0, 2, 3, 5, 6, 7, 9, 10, 11];
const SYL_HI: [&str; 16] = [
    "ba", "de", "fi", "go", "hu", "ja", "ke", "li", "mo", "nu", "pa", "re", "si", "to", "vu", "za",
];
const SYL_LO: [&str; 16] = [
    "bi", "do", "fu", "ga", "he", "ji", "ko", "lu", "ma", "ne", "pi", "ro", "su", "ta", "ve", "zo",
];

// ------------------------------------------------------------------- armors

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Armor {
    Raw,    // UTF-8 pass-through: text formats without stages only
    Hex,    // lowercase hexadecimal
    B64,    // RFC 4648 base64, padded
    B64Url, // RFC 4648 URL-safe base64, no padding
    B32,    // RFC 4648 base32, padded
    Z85,    // Z85 alphabet; the last character is the pad length (0..=3)
    B58,    // Bitcoin alphabet, 8-byte blocks of 11 characters (Monero style)
    Words,  // two pseudo-syllables per byte, e.g. "babi dofu"
}

impl Armor {
    pub const ALL: [Armor; 8] = [
        Armor::Raw,
        Armor::Hex,
        Armor::B64,
        Armor::B64Url,
        Armor::B32,
        Armor::Z85,
        Armor::B58,
        Armor::Words,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Armor::Raw => "raw",
            Armor::Hex => "hex",
            Armor::B64 => "b64",
            Armor::B64Url => "b64u",
            Armor::B32 => "b32",
            Armor::Z85 => "z85",
            Armor::B58 => "b58",
            Armor::Words => "words",
        }
    }

    pub fn parse(s: &str) -> Option<Armor> {
        Armor::ALL.iter().copied().find(|a| a.name() == s)
    }

    /// Armor `data`; `wrap > 0` breaks the text into lines of at most that width.
    /// Only `Raw` can fail (the bytes must be valid UTF-8).
    pub fn encode(self, data: &[u8], wrap: usize) -> Result<String, String> {
        let flat = match self {
            Armor::Raw => {
                return String::from_utf8(data.to_vec())
                    .map_err(|_| "raw armor: data is not valid UTF-8".to_string());
            }
            Armor::Words => return Ok(wrap_words(&words_enc(data), wrap)),
            Armor::Hex => hex_enc(data),
            Armor::B64 => b64_enc(data, B64, true),
            Armor::B64Url => b64_enc(data, B64U, false),
            Armor::B32 => b32_enc(data),
            Armor::Z85 => z85_enc(data),
            Armor::B58 => b58_enc(data),
        };
        Ok(wrap_fixed(&flat, wrap))
    }

    pub fn decode(self, text: &str) -> Result<Vec<u8>, String> {
        if self == Armor::Raw {
            return Ok(text.as_bytes().to_vec());
        }
        if self == Armor::Words {
            return words_dec(text);
        }
        let c = squash(text);
        match self {
            Armor::Hex => hex_dec(&c),
            Armor::B64 => b64_dec(&c, B64),
            Armor::B64Url => b64_dec(&c, B64U),
            Armor::B32 => b32_dec(&c),
            Armor::Z85 => z85_dec(&c),
            Armor::B58 => b58_dec(&c),
            Armor::Raw | Armor::Words => Ok(Vec::new()), // handled above
        }
    }
}

fn squash(t: &str) -> Vec<u8> {
    t.bytes().filter(|b| !b.is_ascii_whitespace()).collect()
}

/// Break ASCII text into lines of `width` characters (0 = leave as is).
fn wrap_fixed(s: &str, width: usize) -> String {
    if width == 0 || s.len() <= width {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + s.len() / width + 1);
    for (i, ch) in s.as_bytes().chunks(width).enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(std::str::from_utf8(ch).unwrap_or(""));
    }
    out
}

/// Join words with spaces, starting a new line before a word that would overflow.
fn wrap_words(words: &[String], width: usize) -> String {
    let mut out = String::new();
    let mut col = 0usize;
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            if width > 0 && col + 1 + w.len() > width {
                out.push('\n');
                col = 0;
            } else {
                out.push(' ');
                col += 1;
            }
        }
        out.push_str(w);
        col += w.len();
    }
    out
}

fn hex_enc(d: &[u8]) -> String {
    const H: &[u8] = b"0123456789abcdef";
    let mut s = String::with_capacity(d.len() * 2);
    for &b in d {
        s.push(H[(b >> 4) as usize] as char);
        s.push(H[(b & 15) as usize] as char);
    }
    s
}

fn hex_dec(c: &[u8]) -> Result<Vec<u8>, String> {
    if c.len() % 2 != 0 {
        return Err("hex: odd number of digits".to_string());
    }
    let mut out = Vec::with_capacity(c.len() / 2);
    for p in c.chunks(2) {
        match ((p[0] as char).to_digit(16), (p[1] as char).to_digit(16)) {
            (Some(h), Some(l)) => out.push((h * 16 + l) as u8),
            _ => return Err("hex: bad digit".to_string()),
        }
    }
    Ok(out)
}

fn b64_enc(d: &[u8], tab: &[u8], pad: bool) -> String {
    let mut s = String::with_capacity(d.len() / 3 * 4 + 4);
    for ch in d.chunks(3) {
        let b = |k: usize| ch.get(k).copied().unwrap_or(0) as u32;
        let n = (b(0) << 16) | (b(1) << 8) | b(2);
        s.push(tab[(n >> 18) as usize & 63] as char);
        s.push(tab[(n >> 12) as usize & 63] as char);
        if ch.len() > 1 {
            s.push(tab[(n >> 6) as usize & 63] as char);
        } else if pad {
            s.push('=');
        }
        if ch.len() > 2 {
            s.push(tab[n as usize & 63] as char);
        } else if pad {
            s.push('=');
        }
    }
    s
}

fn b64_dec(c: &[u8], tab: &[u8]) -> Result<Vec<u8>, String> {
    let mut rev = [255u8; 256];
    for (i, &x) in tab.iter().enumerate() {
        rev[x as usize] = i as u8;
    }
    let mut vals: Vec<u8> = Vec::with_capacity(c.len());
    let mut pads = 0usize;
    for &x in c {
        if x == b'=' {
            pads += 1;
            continue;
        }
        if pads > 0 {
            return Err("base64: data after padding".to_string());
        }
        let v = rev[x as usize];
        if v == 255 {
            return Err(format!("base64: bad character {:?}", x as char));
        }
        vals.push(v);
    }
    if pads > 2 || vals.len() % 4 == 1 || (pads > 0 && (vals.len() + pads) % 4 != 0) {
        return Err("base64: bad length".to_string());
    }
    let mut out = Vec::with_capacity(vals.len() * 3 / 4 + 3);
    for ch in vals.chunks(4) {
        let mut n = 0u32;
        for (k, &v) in ch.iter().enumerate() {
            n |= (v as u32) << (18 - 6 * k as u32);
        }
        out.push((n >> 16) as u8);
        if ch.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if ch.len() > 3 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

fn b32_enc(d: &[u8]) -> String {
    let mut s = String::with_capacity((d.len() + 4) / 5 * 8);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &b in d {
        acc = (acc << 8) | b as u32;
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            s.push(B32[((acc >> bits) & 31) as usize] as char);
        }
        acc &= (1u32 << bits) - 1;
    }
    if bits > 0 {
        s.push(B32[((acc << (5 - bits)) & 31) as usize] as char);
    }
    while s.len() % 8 != 0 {
        s.push('=');
    }
    s
}

fn b32_dec(c: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(c.len() * 5 / 8 + 1);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &x in c {
        let v = match x {
            b'=' => continue,
            b'A'..=b'Z' => x - b'A',
            b'a'..=b'z' => x - b'a',
            b'2'..=b'7' => x - b'2' + 26,
            _ => return Err(format!("base32: bad character {:?}", x as char)),
        };
        acc = (acc << 5) | v as u32;
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1u32 << bits) - 1;
        }
    }
    Ok(out)
}

/// Z85 groups of 4 bytes -> 5 characters; the input is zero-padded to a multiple
/// of 4 and one final character records how many pad bytes to drop.
fn z85_enc(d: &[u8]) -> String {
    let pad = (4 - d.len() % 4) % 4;
    let mut buf = d.to_vec();
    buf.resize(d.len() + pad, 0);
    let mut s = String::with_capacity(buf.len() / 4 * 5 + 1);
    for ch in buf.chunks(4) {
        let mut v = u32::from_be_bytes([ch[0], ch[1], ch[2], ch[3]]);
        let mut g = [0u8; 5];
        for k in (0..5).rev() {
            g[k] = Z85[(v % 85) as usize];
            v /= 85;
        }
        for &c in g.iter() {
            s.push(c as char);
        }
    }
    s.push(Z85[pad] as char);
    s
}

fn z85_dec(c: &[u8]) -> Result<Vec<u8>, String> {
    if c.len() % 5 != 1 {
        return Err("z85: bad length".to_string());
    }
    let mut rev = [255u8; 256];
    for (i, &x) in Z85.iter().enumerate() {
        rev[x as usize] = i as u8;
    }
    let val = |x: u8| -> Result<u64, String> {
        match rev[x as usize] {
            255 => Err(format!("z85: bad character {:?}", x as char)),
            r => Ok(r as u64),
        }
    };
    let pad = val(c[c.len() - 1])? as usize;
    if pad > 3 {
        return Err("z85: bad padding".to_string());
    }
    let mut out = Vec::with_capacity(c.len() / 5 * 4);
    for g in c[..c.len() - 1].chunks(5) {
        let mut v = 0u64;
        for &x in g {
            v = v * 85 + val(x)?;
        }
        if v > u32::MAX as u64 {
            return Err("z85: group out of range".to_string());
        }
        out.extend_from_slice(&(v as u32).to_be_bytes());
    }
    if pad > out.len() {
        return Err("z85: padding exceeds data".to_string());
    }
    out.truncate(out.len() - pad);
    Ok(out)
}

/// Base58 in blocks of up to 8 bytes (11 characters for a full block), so the
/// cost is linear in the input size.
fn b58_enc(d: &[u8]) -> String {
    let mut s = String::with_capacity(d.len() / 8 * 11 + 11);
    for blk in d.chunks(8) {
        let mut v: u64 = 0;
        for &b in blk {
            v = (v << 8) | b as u64;
        }
        let n = B58_CHARS[blk.len()];
        let mut tmp = [0u8; 11];
        for k in (0..n).rev() {
            tmp[k] = B58[(v % 58) as usize];
            v /= 58;
        }
        for &c in &tmp[..n] {
            s.push(c as char);
        }
    }
    s
}

fn b58_dec(c: &[u8]) -> Result<Vec<u8>, String> {
    let mut rev = [255u8; 256];
    for (i, &x) in B58.iter().enumerate() {
        rev[x as usize] = i as u8;
    }
    let mut out = Vec::with_capacity(c.len() / 11 * 8 + 8);
    for blk in c.chunks(11) {
        let k = match B58_CHARS.iter().position(|&n| n == blk.len()) {
            Some(k) if k > 0 => k,
            _ => return Err("base58: bad block length".to_string()),
        };
        let mut v: u128 = 0;
        for &x in blk {
            let d = rev[x as usize];
            if d == 255 {
                return Err(format!("base58: bad character {:?}", x as char));
            }
            v = v * 58 + d as u128;
        }
        if (v >> (8 * k)) != 0 {
            return Err("base58: block out of range".to_string());
        }
        for j in (0..k).rev() {
            out.push((v >> (8 * j)) as u8);
        }
    }
    Ok(out)
}

fn words_enc(d: &[u8]) -> Vec<String> {
    d.iter()
        .map(|&b| format!("{}{}", SYL_HI[(b >> 4) as usize], SYL_LO[(b & 15) as usize]))
        .collect()
}

fn words_dec(t: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for w in t.split_whitespace() {
        let (a, b) = match (w.get(..2), w.get(2..)) {
            (Some(a), Some(b)) => (a, b),
            _ => return Err(format!("words: bad word '{w}'")),
        };
        match (
            SYL_HI.iter().position(|&s| s == a),
            SYL_LO.iter().position(|&s| s == b),
        ) {
            (Some(h), Some(l)) => out.push((h * 16 + l) as u8),
            _ => return Err(format!("words: bad word '{w}'")),
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------- stages

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    Rle,      // PackBits run-length coding
    Lzss,     // LZ77 with 4095-byte window and 3..=18 byte matches
    Delta,    // byte-wise difference to the previous byte
    Rev,      // reverse the byte order
    Mtf,      // move-to-front transform
    Xor(u64), // XOR with a keystream seeded from the key
}

impl Stage {
    pub fn name(self) -> String {
        match self {
            Stage::Rle => "rle".to_string(),
            Stage::Lzss => "lzss".to_string(),
            Stage::Delta => "delta".to_string(),
            Stage::Rev => "rev".to_string(),
            Stage::Mtf => "mtf".to_string(),
            Stage::Xor(k) => format!("xor:{k:016x}"),
        }
    }

    pub fn parse(s: &str) -> Option<Stage> {
        match s {
            "rle" => Some(Stage::Rle),
            "lzss" => Some(Stage::Lzss),
            "delta" => Some(Stage::Delta),
            "rev" => Some(Stage::Rev),
            "mtf" => Some(Stage::Mtf),
            _ => {
                let k = s.strip_prefix("xor:")?;
                if k.is_empty() || k.len() > 16 || !k.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return None;
                }
                u64::from_str_radix(k, 16).ok().map(Stage::Xor)
            }
        }
    }

    /// Index of the stage kind (ignores the XOR key); used to avoid repeats.
    pub fn kind(self) -> usize {
        match self {
            Stage::Rle => 0,
            Stage::Lzss => 1,
            Stage::Delta => 2,
            Stage::Rev => 3,
            Stage::Mtf => 4,
            Stage::Xor(_) => 5,
        }
    }

    pub fn is_compressor(self) -> bool {
        matches!(self, Stage::Rle | Stage::Lzss)
    }