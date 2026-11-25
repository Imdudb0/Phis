//! Canonicalisation de l'alphabet : chaque octet est remplacé par le rang de sa
//! première apparition dans E. Le substrat ne voit jamais l'identité d'un octet,
//! seulement ses relations : l'invariance par renommage bijectif en découle.

use crate::util::Rng;

pub const ABSENT: u16 = u16::MAX;

#[derive(Clone, Debug)]
pub struct Canon {
    /// E en symboles canoniques (0..n).
    pub syms: Vec<u16>,
    /// symbole canonique -> octet d'origine.
    pub byte_of: Vec<u8>,
    /// octet -> symbole canonique (ABSENT si l'octet n'apparaît pas).
    pub sym_of: [u16; 256],
}

impl Canon {
    pub fn new(bytes: &[u8]) -> Canon {
        let mut sym_of = [ABSENT; 256];
        let mut byte_of: Vec<u8> = Vec::new();
        let mut syms: Vec<u16> = Vec::with_capacity(bytes.len());
        for &b in bytes {
            let slot = &mut sym_of[b as usize];
            if *slot == ABSENT {
                *slot = byte_of.len() as u16;
                byte_of.push(b);
            }
            syms.push(*slot);
        }
        Canon { syms, byte_of, sym_of }
    }

    /// Taille de l'alphabet.
    pub fn n(&self) -> usize {
        self.byte_of.len()
    }

    pub fn len(&self) -> usize {
        self.syms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.syms.is_empty()
    }

    /// Symboles canoniques -> octets (les symboles hors alphabet sont ignorés).
    pub fn bytes_of(&self, syms: &[u16]) -> Vec<u8> {
        syms.iter().filter_map(|&s| self.byte_of.get(s as usize).copied()).collect()
    }
}

/// Renommage bijectif d'un contenu octet par octet.
pub fn permute(bytes: &[u8], perm: &[u8; 256]) -> Vec<u8> {
    bytes.iter().map(|&b| perm[b as usize]).collect()
}

pub fn random_perm(rng: &mut Rng) -> [u8; 256] {
    let mut v: Vec<u8> = (0..=255u8).collect();
    rng.shuffle(&mut v);
    let mut p = [0u8; 256];
    p.copy_from_slice(&v);
    p
}

pub fn inverse(perm: &[u8; 256]) -> [u8; 256] {
    let mut inv = [0u8; 256];
    for (i, &p) in perm.iter().enumerate() {
        inv[p as usize] = i as u8;
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invariance_renommage() {
        let a = b"abracadabra abracadabra";
        let mut rng = Rng::new(3);
        let p = random_perm(&mut rng);
        let b = permute(a, &p);
        assert_eq!(Canon::new(a).syms, Canon::new(&b).syms);
        let back = permute(&b, &inverse(&p));
        assert_eq!(&back[..], &a[..]);
    }
}