//! Deterministic RNGs: CPython-compatible MT19937 (`PyRandom`) + SplitMix64/Box-Muller (`NormalRng`).

pub struct PyRandom {
    mt: [u32; 624],
    index: usize,
}

impl PyRandom {
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            mt: [0u32; 624],
            index: 624,
        };
        rng.seed(seed);
        rng
    }

    pub fn seed(&mut self, seed: u64) {
        let lo = (seed & 0xFFFF_FFFF) as u32;
        let hi = (seed >> 32) as u32;
        if hi != 0 {
            self.init_by_array(&[lo, hi]);
        } else {
            self.init_by_array(&[lo]);
        }
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for mti in 1..624usize {
            let prev = self.mt[mti - 1];
            self.mt[mti] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(mti as u32);
        }
        self.index = 624;
    }

    fn init_by_array(&mut self, init_key: &[u32]) {
        self.init_genrand(19650218u32);
        let mut i: usize = 1;
        let mut j: usize = 0;
        let mut k: usize = 624usize.max(init_key.len());
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525u32))
                .wrapping_add(init_key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                self.mt[0] = self.mt[623];
                i = 1;
            }
            if j >= init_key.len() {
                j = 0;
            }
            k -= 1;
        }
        for _ in 0..623 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941u32))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                self.mt[0] = self.mt[623];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000u32;
        self.index = 624;
    }

    pub fn genrand_uint32(&mut self) -> u32 {
        const N: usize = 624;
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;

        if self.index >= N {
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                let mag = if y & 1 != 0 { MATRIX_A } else { 0 };
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag;
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                let mag = if y & 1 != 0 { MATRIX_A } else { 0 };
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ mag;
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            let mag = if y & 1 != 0 { MATRIX_A } else { 0 };
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag;
            self.index = 0;
        }

        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_uint32() >> 5) as f64;
        let b = (self.genrand_uint32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    fn getrandbits_usize(&mut self, k: u32) -> usize {
        if k == 0 {
            return 0;
        }
        if k <= 32 {
            return (self.genrand_uint32() >> (32 - k)) as usize;
        }
        let mut out: usize = 0;
        let mut shift: u32 = 0;
        let mut rem = k;
        while rem > 0 {
            let mut r = self.genrand_uint32();
            if rem < 32 {
                r >>= 32 - rem;
                out |= (r as usize) << shift;
                break;
            }
            out |= (r as usize) << shift;
            shift += 32;
            rem -= 32;
        }
        out
    }

    pub fn randbelow(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = self.getrandbits_usize(k);
            if r < n {
                return r;
            }
        }
    }

    pub fn randrange(&mut self, stop: usize) -> usize {
        assert!(stop > 0, "empty range for randrange");
        self.randbelow(stop)
    }

    pub fn randrange_range(&mut self, start: usize, stop: usize) -> usize {
        assert!(stop > start, "empty range for randrange_range");
        start + self.randbelow(stop - start)
    }

    pub fn randint(&mut self, a: usize, b: usize) -> usize {
        assert!(b >= a, "empty range for randint");
        a + self.randbelow(b - a + 1)
    }

    pub fn choice<'a, T>(&mut self, seq: &'a [T]) -> &'a T {
        assert!(!seq.is_empty(), "cannot choose from an empty sequence");
        let idx = self.randbelow(seq.len());
        &seq[idx]
    }

    pub fn shuffle<T>(&mut self, seq: &mut [T]) {
        if seq.len() < 2 {
            return;
        }
        for i in (1..seq.len()).rev() {
            let j = self.randbelow(i + 1);
            seq.swap(i, j);
        }
    }

    pub fn choices_one<'a, T>(&mut self, population: &'a [T], weights: &[f64]) -> &'a T {
        assert!(!population.is_empty() && population.len() == weights.len());
        let mut cum = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for &w in weights {
            acc += w;
            cum.push(acc);
        }
        let total = acc + 0.0;
        let x = self.random() * total;
        let mut lo = 0usize;
        let mut hi = population.len() - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if x < cum[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        &population[lo]
    }
}

pub struct NormalRng {
    state: u64,
    spare: Option<f64>,
}

impl NormalRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed, spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn standard_normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9007199254740992.0);
        let u2 = (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}