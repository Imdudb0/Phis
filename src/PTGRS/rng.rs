//! Deterministic PRNG: xoshiro256** seeded through SplitMix64. Zero dependencies.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut sm = seed;
        let mut s = [0u64; 4];
        for x in s.iter_mut() {
            *x = splitmix(&mut sm);
        }
        Rng { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Unbiased integer in `0..n` (Lemire). Returns 0 when `n <= 1`.
    pub fn below(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let n64 = n as u64;
        let mut x = self.next_u64();
        let mut m = (x as u128) * (n64 as u128);
        let mut l = m as u64;
        if l < n64 {
            let t = n64.wrapping_neg() % n64;
            while l < t {
                x = self.next_u64();
                m = (x as u128) * (n64 as u128);
                l = m as u64;
            }
        }
        (m >> 64) as usize
    }

    /// Inclusive range `lo..=hi`.
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo {
            return lo;
        }
        lo + self.below(hi - lo + 1)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }

    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }

    /// Index drawn proportionally to non-negative weights (uniform if all zero).
    pub fn weighted(&mut self, w: &[f64]) -> usize {
        let total: f64 = w.iter().sum();
        if !(total > 0.0) {
            return self.below(w.len());
        }
        let mut x = self.unit() * total;
        for (i, &wi) in w.iter().enumerate() {
            if x < wi {
                return i;
            }
            x -= wi;
        }
        w.len() - 1
    }

    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }

    /// Exponential variate with the given rate.
    pub fn exp(&mut self, rate: f64) -> f64 {
        -(1.0 - self.unit()).ln() / rate
    }

    /// Independent child stream.
    pub fn fork(&mut self) -> Rng {
        Rng::new(self.next_u64())
    }
}