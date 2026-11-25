//! Déterminisme intégral. Aucune source d'aléa externe, aucun crate.
//! Toute stochasticité (PTGRS, init, recherche, MIP) dérive d'une graine unique
//! consignée dans le manifeste de version. Deux exécutions de même graine et
//! même budget produisent la même trajectoire, octet pour octet.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

#[inline]
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut x = seed ^ 0xA076_1D64_78BD_642F;
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            *v = splitmix64(&mut x);
        }
        if s.iter().all(|&v| v == 0) {
            s[0] = 1;
        }
        Rng { s }
    }

    /// Sous-flux indépendant, étiqueté. Empêche tout couplage accidentel entre
    /// le tirage d'une règle PTGRS et celui d'un opérateur abductif.
    pub fn fork(&mut self, tag: u64) -> Rng {
        let a = self.next_u64();
        Rng::new(a ^ tag.wrapping_mul(0xD1B5_4A32_D192_ED03))
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // xoshiro256**
        let result = self.s[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniforme dans [0,1), 24 bits de mantisse.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % (n as u64)) as usize
        }
    }

    /// Bornes incluses. Retourne `lo` si l'intervalle est vide.
    #[inline]
    pub fn between(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo {
            lo
        } else {
            lo + self.below(hi - lo + 1)
        }
    }

    #[inline]
    pub fn bernoulli(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }

    pub fn gauss(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        let n = v.len();
        if n < 2 {
            return;
        }
        for i in (1..n).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }

    /// Tirage proportionnel. Poids négatifs tronqués à 0 ; somme nulle -> uniforme.
    pub fn weighted(&mut self, w: &[f32]) -> usize {
        if w.is_empty() {
            return 0;
        }
        let total: f32 = w.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).sum();
        if !(total > 0.0) {
            return self.below(w.len());
        }
        let mut t = self.next_f32() * total;
        for (i, &x) in w.iter().enumerate() {
            let x = if x > 0.0 { x } else { 0.0 };
            if t < x {
                return i;
            }
            t -= x;
        }
        w.len() - 1
    }

    pub fn pick<T: Copy>(&mut self, v: &[T]) -> T {
        v[self.below(v.len())]
    }
}

/// FNV-1a 64. Utilisé pour les empreintes de version, les formes canoniques et
/// les étiquettes de flux. Jamais comme signal d'apprentissage.
pub fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn fnv1a_u16(data: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in data {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

pub fn tag(tag: &str) -> u64 {
    fnv1a(tag.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinisme() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn fork_independant() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        let f1 = a.fork(tag("ptgrs"));
        let f2 = b.fork(tag("search"));
        assert_ne!(f1.s, f2.s);
    }
}