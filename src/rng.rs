#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 4],
}

#[inline]
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl Rng {
    pub fn from_seed(seed: u64) -> Self {
        let mut s = seed ^ 0x517c_c1b7_2722_0a95;
        let s0 = splitmix64(&mut s);
        let s1 = splitmix64(&mut s);
        let s2 = splitmix64(&mut s);
        let s3 = splitmix64(&mut s);
        let state = if (s0 | s1 | s2 | s3) == 0 {
            [1, 2, 3, 4]
        } else {
            [s0, s1, s2, s3]
        };
        Self { state }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    #[inline]
    pub fn next_f64_01(&mut self) -> f64 {
        let upper = self.next_u64() >> 11;
        (upper as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    pub fn next_f32_01(&mut self) -> f32 {
        let upper = self.next_u64() >> 40;
        (upper as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    #[inline]
    pub fn uniform_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32_01()
    }

    pub fn normal_f32(&mut self, mean: f32, std: f32) -> f32 {
        let u1 = self.next_f64_01().max(1e-12);
        let u2 = self.next_f64_01();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * (z0 as f32)
    }

    #[inline]
    pub fn index(&mut self, bound: usize) -> usize {
        assert!(bound > 0, "Rng::index bound must be > 0");
        ((self.next_u64() as u128 * bound as u128) >> 64) as usize
    }

    #[inline]
    pub fn range_usize(&mut self, lo: usize, hi_exclusive: usize) -> usize {
        assert!(hi_exclusive > lo, "invalid range");
        lo + self.index(hi_exclusive - lo)
    }

    #[inline]
    pub fn bernoulli(&mut self, p: f32) -> bool {
        self.next_f32_01() < p
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        if n <= 1 {
            return;
        }
        for i in (1..n).rev() {
            let j = self.index(i + 1);
            slice.swap(i, j);
        }
    }

    pub fn choose<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        assert!(!slice.is_empty(), "cannot choose from empty slice");
        let idx = self.index(slice.len());
        &slice[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn deterministic_sequence_and_bounds() {
        let mut r1 = Rng::from_seed(42);
        let mut r2 = Rng::from_seed(42);
        for _ in 0..1000 {
            assert_eq!(r1.next_u64(), r2.next_u64());
            let v = r1.next_f32_01();
            let _ = r2.next_f32_01();
            assert!((0.0..1.0).contains(&v));
        }
        for _ in 0..500 {
            let i = r1.index(17);
            assert!(i < 17);
        }
    }
}