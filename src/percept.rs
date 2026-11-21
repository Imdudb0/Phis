use crate::util::*;
#[derive(Clone, Debug)]
pub struct Percept {
    pub raw: Vec<u8>,
    pub codes: Vec<u16>,
    pub alphabet: Vec<u8>,
    pub a: usize,
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