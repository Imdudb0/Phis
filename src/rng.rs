//! PRNG déterministe (xorshift64*), zéro dépendance.

pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0xDEADBEEFCAFEBABE)
            | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as u32 as f32) / ((1u32 << 24) as f32)
    }
    pub fn next_range(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next_u64() as usize) % n }
    }
    /// Box–Muller gaussien.
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}