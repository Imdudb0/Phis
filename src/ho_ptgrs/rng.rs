/// Générateur pseudo-aléatoire autonome (SplitMix64 -> Xoshiro256**), zéro dépendance.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn from_seed(mut seed: u64) -> Self {
        let mut splitmix = || -> u64 {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self {
            s: [splitmix(), splitmix(), splitmix(), splitmix()],
        }
    }

    #[inline]
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

    /// Entier uniforme dans [0, upper) sans biais modulo.
    pub fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        let u = upper as u64;
        let zone = u64::MAX - (u64::MAX % u);
        loop {
            let v = self.next_u64();
            if v < zone {
                return (v % u) as usize;
            }
        }
    }

    /// Entier uniforme dans [low, high].
    pub fn gen_range_inclusive(&mut self, low: usize, high: usize) -> usize {
        if high <= low {
            return low;
        }
        low + self.gen_range(high - low + 1)
    }

    pub fn gen_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    pub fn gen_bool(&mut self, prob: f64) -> bool {
        self.gen_f64() < prob
    }

    /// Mélange de Fisher-Yates en place.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.gen_range(i + 1);
            slice.swap(i, j);
        }
    }
}
