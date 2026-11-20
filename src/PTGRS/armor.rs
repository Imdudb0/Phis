//! Byte-level layers of the codec engine, written from scratch.
//!
//! * `Armor` maps bytes to printable text and back. Every armor except `Raw` and
//!   `Words` ignores ASCII whitespace when decoding, so a body can be wrapped at
//!   any width; for `Words` whitespace separates the words.
//! * `Stage` is a reversible byte filter. A pipeline applies its stages in order
//!   before armoring and undoes them in reverse order after de-armoring.

use super::rng::Rng;

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

    /// A random stage; `Xor` gets a fresh key.
    pub fn random(rng: &mut Rng) -> Stage {
        match rng.below(6) {
            0 => Stage::Rle,
            1 => Stage::Lzss,
            2 => Stage::Delta,
            3 => Stage::Rev,
            4 => Stage::Mtf,
            _ => Stage::Xor(rng.next_u64()),
        }
    }

    /// Run the filter forward. Never fails.
    pub fn apply(self, data: &[u8]) -> Vec<u8> {
        match self {
            Stage::Rle => rle_enc(data),
            Stage::Lzss => lzss_enc(data),
            Stage::Delta => delta_enc(data),
            Stage::Rev => data.iter().rev().copied().collect(),
            Stage::Mtf => mtf_enc(data),
            Stage::Xor(key) => xor_stream(data, key),
        }
    }

    /// Undo the filter. `Err` on corrupt input or when the output would pass `LIMIT`.
    pub fn undo(self, data: &[u8]) -> Result<Vec<u8>, String> {
        match self {
            Stage::Rle => rle_dec(data),
            Stage::Lzss => lzss_dec(data),
            Stage::Delta => Ok(delta_dec(data)),
            Stage::Rev => Ok(data.iter().rev().copied().collect()),
            Stage::Mtf => Ok(mtf_dec(data)),
            Stage::Xor(key) => Ok(xor_stream(data, key)),
        }
    }
}

// ------------------------------------------------------------ stage codecs

/// PackBits. A control byte `n` in `0..=127` copies the next `n + 1` bytes;
/// `129..=255` repeats the next byte `257 - n` times (2..=128). `128` is never
/// written and the decoder rejects it.
fn rle_enc(d: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(d.len() + d.len() / 128 + 2);
    let mut i = 0usize;
    while i < d.len() {
        let mut run = 1usize;
        while run < 128 && i + run < d.len() && d[i + run] == d[i] {
            run += 1;
        }
        if run >= 2 {
            out.push((257 - run) as u8);
            out.push(d[i]);
            i += run;
        } else {
            // literal stretch: ends where a run of two or more starts, or at 128 bytes
            let start = i;
            i += 1;
            while i < d.len() && i - start < 128 && !(i + 1 < d.len() && d[i] == d[i + 1]) {
                i += 1;
            }
            out.push((i - start - 1) as u8);
            out.extend_from_slice(&d[start..i]);
        }
    }
    out
}

fn rle_dec(d: &[u8]) -> Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::new();
    let mut i = 0usize;
    while i < d.len() {
        let n = d[i] as usize;
        i += 1;
        if n < 128 {
            let len = n + 1;
            if i + len > d.len() {
                return Err("rle: literal runs past the end".to_string());
            }
            out.extend_from_slice(&d[i..i + len]);
            i += len;
        } else if n > 128 {
            if i >= d.len() {
                return Err("rle: run without a byte".to_string());
            }
            let b = d[i];
            i += 1;
            out.resize(out.len() + (257 - n), b);
        } else {
            return Err("rle: reserved control byte".to_string());
        }
        if out.len() > LIMIT {
            return Err("rle: output exceeds the size limit".to_string());
        }
    }
    Ok(out)
}

fn delta_enc(d: &[u8]) -> Vec<u8> {
    let mut prev = 0u8;
    d.iter()
        .map(|&b| {
            let r = b.wrapping_sub(prev);
            prev = b;
            r
        })
        .collect()
}

fn delta_dec(d: &[u8]) -> Vec<u8> {
    let mut acc = 0u8;
    d.iter()
        .map(|&r| {
            acc = acc.wrapping_add(r);
            acc
        })
        .collect()
}

fn mtf_enc(d: &[u8]) -> Vec<u8> {
    let mut tab: Vec<u8> = (0..=255u8).collect();
    let mut out = Vec::with_capacity(d.len());
    for &b in d {
        let pos = tab.iter().position(|&x| x == b).unwrap_or(0);
        out.push(pos as u8);
        tab.copy_within(0..pos, 1);
        tab[0] = b;
    }
    out
}

fn mtf_dec(d: &[u8]) -> Vec<u8> {
    let mut tab: Vec<u8> = (0..=255u8).collect();
    let mut out = Vec::with_capacity(d.len());
    for &i in d {
        let pos = i as usize;
        let b = tab[pos];
        out.push(b);
        tab.copy_within(0..pos, 1);
        tab[0] = b;
    }
    out
}

/// XOR with a keystream: `Rng::new(key)`, eight bytes per draw, little-endian.
/// Applying it twice restores the input.
fn xor_stream(d: &[u8], key: u64) -> Vec<u8> {
    let mut rng = Rng::new(key);
    let mut out = Vec::with_capacity(d.len());
    for chunk in d.chunks(8) {
        let ks = rng.next_u64().to_le_bytes();
        for (k, &b) in chunk.iter().enumerate() {
            out.push(b ^ ks[k]);
        }
    }
    out
}

// LZSS: a flag byte per eight items (low bit first, 1 = literal). A match is two
// bytes, `(distance << 4) | (length - 3)` big-endian, distance 1..=4095, length
// 3..=18. No header: the stream ends with the data.
const LZ_WIN: usize = 4095; // farthest back a match may reach
const LZ_MIN: usize = 3;
const LZ_MAX: usize = 18;
const LZ_DEPTH: usize = 24; // hash-chain steps tried per position
const LZ_HASH_BITS: u32 = 15;
const LZ_NONE: usize = usize::MAX;

fn lz_hash(d: &[u8], i: usize) -> usize {
    let v = (d[i] as u32) | ((d[i + 1] as u32) << 8) | ((d[i + 2] as u32) << 16);
    (v.wrapping_mul(0x9E37_79B1) >> (32 - LZ_HASH_BITS)) as usize
}

fn lzss_enc(d: &[u8]) -> Vec<u8> {
    let n = d.len();
    let mut head = vec![LZ_NONE; 1 << LZ_HASH_BITS];
    // ring of chain links indexed by `position & LZ_WIN`; a slot is overwritten
    // only 4096 positions later, which is already outside the window
    let mut prev = vec![LZ_NONE; LZ_WIN + 1];
    let mut out: Vec<u8> = Vec::with_capacity(n / 2 + 16);
    let mut flag_at = 0usize;
    let mut bit = 8u32; // 8 = the next item needs a new flag byte
    let mut i = 0usize;
    while i < n {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        if i + LZ_MIN <= n {
            let limit = LZ_MAX.min(n - i);
            let mut cand = head[lz_hash(d, i)];
            let mut depth = 0usize;
            while cand != LZ_NONE && i - cand <= LZ_WIN && depth < LZ_DEPTH {
                let mut l = 0usize;
                while l < limit && d[cand + l] == d[i + l] {
                    l += 1;
                }
                if l > best_len {
                    best_len = l;
                    best_dist = i - cand;
                    if l == limit {
                        break;
                    }
                }
                cand = prev[cand & LZ_WIN];
                depth += 1;
            }
        }
        if bit == 8 {
            flag_at = out.len();
            out.push(0);
            bit = 0;
        }
        let step = if best_len >= LZ_MIN {
            let v = (best_dist << 4) | (best_len - LZ_MIN);
            out.push((v >> 8) as u8);
            out.push(v as u8);
            best_len
        } else {
            out[flag_at] |= 1u8 << bit;
            out.push(d[i]);
            1
        };
        bit += 1;
        // enter every covered position into the chains
        for j in i..i + step {
            if j + LZ_MIN <= n {
                let h = lz_hash(d, j);
                prev[j & LZ_WIN] = head[h];
                head[h] = j;
            }
        }
        i += step;
    }
    out
}

fn lzss_dec(d: &[u8]) -> Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::with_capacity(d.len().saturating_mul(2).min(LIMIT));
    let mut i = 0usize;
    while i < d.len() {
        let flags = d[i];
        i += 1;
        for bit in 0..8u32 {
            if i >= d.len() {
                break;
            }
            if flags & (1u8 << bit) != 0 {
                out.push(d[i]);
                i += 1;
            } else {
                if i + 1 >= d.len() {
                    return Err("lzss: truncated match".to_string());
                }
                let v = ((d[i] as usize) << 8) | d[i + 1] as usize;
                i += 2;
                let dist = v >> 4;
                let len = (v & 15) + LZ_MIN;
                if dist == 0 || dist > out.len() {
                    return Err("lzss: match reaches before the start".to_string());
                }
                let from = out.len() - dist;
                for k in 0..len {
                    let b = out[from + k]; // overlapping copies are intended
                    out.push(b);
                }
            }
            if out.len() > LIMIT {
                return Err("lzss: output exceeds the size limit".to_string());
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte strings of assorted shapes: noise, long runs, repeated phrases, ramps.
    fn sample(rng: &mut Rng) -> Vec<u8> {
        let n = match rng.below(4) {
            0 => rng.below(9),
            1 => rng.below(70),
            2 => rng.below(700),
            _ => rng.below(9000),
        };
        let mut v: Vec<u8> = Vec::with_capacity(n + 300);
        match rng.below(5) {
            0 => {
                while v.len() < n {
                    v.push(rng.next_u64() as u8);
                }
            }
            1 => {
                while v.len() < n {
                    let b = rng.next_u64() as u8;
                    for _ in 0..rng.range(1, 300) {
                        v.push(b);
                    }
                }
            }
            2 => {
                let mut phrase: Vec<u8> = Vec::new();
                for _ in 0..rng.range(1, 40) {
                    phrase.push(rng.next_u64() as u8);
                }
                while v.len() < n {
                    v.extend_from_slice(&phrase);
                    if rng.chance(0.2) {
                        v.push(rng.next_u64() as u8); // a little noise now and then
                    }
                }
            }
            3 => {
                let step = rng.range(1, 5) as u8;
                let mut x = 0u8;
                while v.len() < n {
                    v.push(x);
                    x = x.wrapping_add(step);
                }
            }
            _ => {
                const POOL: [u8; 5] = [b'a', b'b', b'c', 0, 255];
                while v.len() < n {
                    v.push(POOL[rng.below(POOL.len())]);
                }
            }
        }
        v.truncate(n);
        v
    }

    #[test]
    fn standard_vectors() {
        let cases: [(&str, &str, &str); 7] = [
            ("", "", ""),
            ("f", "Zg==", "MY======"),
            ("fo", "Zm8=", "MZXQ===="),
            ("foo", "Zm9v", "MZXW6==="),
            ("foob", "Zm9vYg==", "MZXW6YQ="),
            ("fooba", "Zm9vYmE=", "MZXW6YTB"),
            ("foobar", "Zm9vYmFy", "MZXW6YTBOI======"),
        ];
        for (plain, b64, b32) in cases {
            assert_eq!(Armor::B64.encode(plain.as_bytes(), 0).unwrap(), b64);
            assert_eq!(Armor::B32.encode(plain.as_bytes(), 0).unwrap(), b32);
            assert_eq!(Armor::B64.decode(b64).unwrap(), plain.as_bytes());
            assert_eq!(Armor::B32.decode(b32).unwrap(), plain.as_bytes());
        }
        assert_eq!(Armor::B64.decode(" Zm9v\r\nYmFy ").unwrap(), b"foobar");
        assert_eq!(Armor::Hex.encode(&[0x00, 0xab, 0xff], 0).unwrap(), "00abff");
        assert_eq!(Armor::B64.encode(&[0xfb, 0xff], 0).unwrap(), "+/8=");
        assert_eq!(Armor::B64Url.encode(&[0xfb, 0xff], 0).unwrap(), "-_8");
        // the ZeroMQ Z85 vector, followed by this crate's pad-length character
        let z = [0x86, 0x4F, 0xD2, 0x6F, 0xB5, 0x59, 0xF7, 0x5B];
        assert_eq!(Armor::Z85.encode(&z, 0).unwrap(), "HelloWorld0");
        assert_eq!(Armor::B58.encode(&[0xff], 0).unwrap(), "5Q");
        let one = Armor::B58.encode(&[0, 0, 0, 0, 0, 0, 0, 1], 0).unwrap();
        assert_eq!(one, format!("{}2", "1".repeat(10)));
        assert_eq!(Armor::Words.encode(&[0x01, 0xf0], 0).unwrap(), "bado zabi");
    }

    #[test]
    fn armors_round_trip_at_every_wrap() {
        let mut rng = Rng::new(1);
        for _ in 0..150 {
            let data = sample(&mut rng);
            for a in Armor::ALL {
                if a == Armor::Raw {
                    continue;
                }
                for wrap in [0usize, 7, 76] {
                    let text = a.encode(&data, wrap).unwrap();
                    assert!(text.is_ascii());
                    if wrap > 0 {
                        assert!(text.lines().all(|l| l.len() <= wrap), "{} wrap {wrap}", a.name());
                    }
                    assert_eq!(a.decode(&text).unwrap(), data, "{} wrap {wrap}", a.name());
                }
            }
        }
    }

    #[test]
    fn raw_armor_is_verbatim_utf8() {
        let s = "héllo\n wörld 漢";
        assert_eq!(Armor::Raw.encode(s.as_bytes(), 5).unwrap(), s);
        assert_eq!(Armor::Raw.decode(s).unwrap(), s.as_bytes());
        assert!(Armor::Raw.encode(&[0xff, 0xfe], 0).is_err());
    }

    #[test]
    fn corrupt_text_is_rejected_and_noise_never_panics() {
        for (a, bad) in [
            (Armor::Hex, "abc"),
            (Armor::Hex, "zz"),
            (Armor::B64, "Z"),
            (Armor::B64, "Zg=x"),
            (Armor::B64, "Zm9é"),
            (Armor::B32, "M1"),
            (Armor::Z85, "abcd"),
            (Armor::Z85, "#####0"), // group above u32::MAX
            (Armor::B58, "1"),
            (Armor::B58, "0O"),
            (Armor::Words, "ba"),
            (Armor::Words, "xyzw"),
        ] {
            assert!(a.decode(bad).is_err(), "{} accepted {bad:?}", a.name());
        }
        assert!(Armor::B58.decode(&"z".repeat(11)).is_err()); // block above 2^64
        let mut rng = Rng::new(5);
        for _ in 0..300 {
            let n = rng.below(40);
            let junk: String = (0..n).map(|_| (b' ' + rng.below(95) as u8) as char).collect();
            for a in Armor::ALL {
                let _ = a.decode(&junk); // may succeed or fail, must not panic
            }
        }
    }

    #[test]
    fn stages_round_trip_alone_and_stacked() {
        let mut rng = Rng::new(2);
        for _ in 0..200 {
            let data = sample(&mut rng);
            let each = [
                Stage::Rle,
                Stage::Lzss,
                Stage::Delta,
                Stage::Rev,
                Stage::Mtf,
                Stage::Xor(0),
                Stage::Xor(rng.next_u64()),
            ];
            for s in each {
                let enc = s.apply(&data);
                assert_eq!(s.undo(&enc).unwrap(), data, "{}", s.name());
            }
            // a random pipeline: apply in order, undo in reverse order
            let n = rng.range(1, 5);
            let mut pipe: Vec<Stage> = Vec::new();
            for _ in 0..n {
                pipe.push(Stage::random(&mut rng));
            }
            let mut buf = data.clone();
            for s in pipe.iter() {
                buf = s.apply(&buf);
            }
            for s in pipe.iter().rev() {
                buf = s.undo(&buf).unwrap();
            }
            assert_eq!(buf, data);
        }
    }

    #[test]
    fn compressors_compress_and_the_rest_keep_the_length() {
        let flat = vec![7u8; 5000];
        assert!(Stage::Rle.apply(&flat).len() < 100);
        let text: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
            .iter()
            .cycle()
            .take(4000)
            .copied()
            .collect();
        assert!(Stage::Lzss.apply(&text).len() < text.len() / 4);
        for s in [Stage::Delta, Stage::Rev, Stage::Mtf, Stage::Xor(5)] {
            assert_eq!(s.apply(&text).len(), text.len());
        }
        // incompressible input grows by at most one flag byte per eight items
        let mut rng = Rng::new(8);
        let noise: Vec<u8> = (0..3000).map(|_| rng.next_u64() as u8).collect();
        assert!(Stage::Lzss.apply(&noise).len() <= noise.len() + noise.len() / 8 + 1);
        // a repeat 6000 bytes back is out of reach of the 4095-byte window
        let mut twice: Vec<u8> = (0..6000).map(|_| rng.next_u64() as u8).collect();
        twice.extend_from_within(..);
        let enc = Stage::Lzss.apply(&twice);
        assert!(enc.len() > twice.len());
        assert_eq!(Stage::Lzss.undo(&enc).unwrap(), twice);
    }

    #[test]
    fn corrupt_stage_input_is_rejected() {
        assert!(Stage::Rle.undo(&[128]).is_err()); // reserved control byte
        assert!(Stage::Rle.undo(&[5, 1, 2]).is_err()); // literal cut short
        assert!(Stage::Rle.undo(&[200]).is_err()); // run without its byte
        assert!(Stage::Lzss.undo(&[0, 0x00]).is_err()); // match cut short
        assert!(Stage::Lzss.undo(&[0, 0x00, 0x10]).is_err()); // match before the start
        let bomb = [129u8, 0].repeat(LIMIT / 128 + 1); // every pair expands to 128 bytes
        assert!(Stage::Rle.undo(&bomb).is_err());
        let mut rng = Rng::new(4);
        for _ in 0..300 {
            let junk = sample(&mut rng);
            for s in [Stage::Rle, Stage::Lzss] {
                let _ = s.undo(&junk); // may succeed or fail, must not panic
            }
        }
    }

    #[test]
    fn stage_names_parse_back_and_keys_matter() {
        let mut rng = Rng::new(3);
        let mut seen = [false; 6];
        for _ in 0..200 {
            let s = Stage::random(&mut rng);
            seen[s.kind()] = true;
            assert_eq!(Stage::parse(&s.name()), Some(s));
        }
        assert!(seen.iter().all(|&b| b));
        assert_eq!(Stage::parse("xor:"), None);
        assert_eq!(Stage::parse("xor:0123456789abcdef0"), None); // 17 digits
        assert_eq!(Stage::parse("xor:zz"), None);
        assert_eq!(Stage::parse("gzip"), None);
        assert!(Stage::Lzss.is_compressor() && !Stage::Xor(1).is_compressor());
        let (a, b) = (Stage::Xor(1), Stage::Xor(2));
        assert_eq!(a.kind(), b.kind());
        assert_ne!(a.apply(&[0; 32]), b.apply(&[0; 32]));
    }
}