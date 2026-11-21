//! percept.rs — E (the perceived effect), its alphabet A(E), and frames.
//! Symbols are numbered by first appearance, so any byte renaming of E yields
//! the exact same `codes`: every downstream quantity is renaming-invariant.

use crate::util::*;

#[derive(Clone, Debug)]
pub struct Percept {
    pub raw: Vec<u8>,
    /// E recoded as indices into A(E) (first-appearance order).
    pub codes: Vec<u16>,
    /// alphabet[code] = original byte.
    pub alphabet: Vec<u8>,
    /// |A(E)|
    pub a: usize,
    /// Hash of `codes` (invariant under renaming of bytes).
    pub fingerprint: u64,
}

impl Percept {
    pub fn from_bytes(raw: &[u8]) -> Percept {
        let mut idx = [u16::MAX; 256];
        let mut alphabet: Vec<u8> = Vec::new();
        let mut codes: Vec<u16> = Vec::with_capacity(raw.len());
        for &b in raw {
            if idx[b as usize] == u16::MAX {
                idx[b as usize] = alphabet.len() as u16;
                alphabet.push(b);
            }
            codes.push(idx[b as usize]);
        }
        let fingerprint = hash_u16(0x5048_4953, &codes);
        let a = alphabet.len();
        Percept { raw: raw.to_vec(), codes, alphabet, a, fingerprint }
    }

    pub fn len(&self) -> usize { self.codes.len() }

    pub fn is_empty(&self) -> bool { self.codes.is_empty() }

    pub fn byte(&self, code: u16) -> u8 { self.alphabet[code as usize] }

    /// Consecutive frames of at most `np` bytes, as (start, len).
    /// Node values always live in the A(E) of the whole E, whatever the frame.
    pub fn frames(&self, np: usize) -> Vec<(usize, usize)> {
        let np = np.max(1);
        let mut v = Vec::new();
        let mut s = 0usize;
        while s < self.len() {
            let l = np.min(self.len() - s);
            v.push((s, l));
            s += l;
        }
        v
    }
}

/// Random injective renaming of the 256 byte values.
pub fn random_renaming(rng: &mut Rng) -> [u8; 256] {
    let mut p = [0u8; 256];
    for k in 0..256usize {
        p[k] = k as u8;
    }
    rng.shuffle(&mut p);
    p
}

pub fn rename(raw: &[u8], perm: &[u8; 256]) -> Vec<u8> {
    raw.iter().map(|&b| perm[b as usize]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recoding_is_renaming_invariant() {
        let mut rng = Rng::new(5);
        let perm = random_renaming(&mut rng);
        let e1 = Percept::from_bytes(b"hello world, hello");
        let e2 = Percept::from_bytes(&rename(b"hello world, hello", &perm));
        assert_eq!(e1.codes, e2.codes);
        assert_eq!(e1.fingerprint, e2.fingerprint);
        assert_eq!(e1.a, e2.a);
    }

    #[test]
    fn frames_cover_everything() {
        let e = Percept::from_bytes(b"abcdefghij");
        let f = e.frames(4);
        assert_eq!(f, vec![(0, 4), (4, 4), (8, 2)]);
    }
}