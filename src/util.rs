#[inline]
pub fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
#[inline]
pub fn mix(a: u64, b: u64) -> u64 {
    let mut s = a;
    let x = splitmix64(&mut s);
    let mut t = x ^ b.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    splitmix64(&mut t)
}
#[inline]
pub fn mix3(a: u64, b: u64, c: u64) -> u64 { mix(mix(a, b), c) }
#[inline]
pub fn mix4(a: u64, b: u64, c: u64, d: u64) -> u64 { mix(mix(mix(a, b), c), d) }
#[inline]
pub fn unit(h: u64) -> f64 { (h >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0) }
#[inline]
pub fn below(h: u64, n: usize) -> usize {
    if n <= 1 { 0 } else { (((h >> 32) * (n as u64)) >> 32) as usize }
}
pub fn hash_u16(seed: u64, xs: &[u16]) -> u64 {
    let mut h = mix(seed, xs.len() as u64);
    for &x in xs { h = mix(h, x as u64 + 1); }
    h
}
pub fn hash_usize(seed: u64, xs: &[usize]) -> u64 {
    let mut h = mix(seed, xs.len() as u64);
    for &x in xs { h = mix(h, x as u64 + 1); }
    h
}
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}
impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Rng { s, spare: None }
    }
    pub fn next_u64(&mut self) -> u64 {
        let r = self.s[0].wrapping_add(self.s[3]).rotate_left(23).wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }
    pub fn f64(&mut self) -> f64 { unit(self.next_u64()) }
    pub fn below(&mut self, n: usize) -> usize {
        if n <= 1 { 0 } else { (((self.next_u64() >> 32) * (n as u64)) >> 32) as usize }
    }
    pub fn range(&mut self, lo: usize, hi: usize) -> usize { lo + self.below(hi - lo + 1) }
    pub fn chance(&mut self, p: f64) -> bool { self.f64() < p }
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let mut u = self.f64();
        while u <= 1e-300 {
            u = self.f64();
        }
        let v = self.f64();
        let r = (-2.0 * u.ln()).sqrt();
        let th = 2.0 * std::f64::consts::PI * v;
        self.spare = Some(r * th.sin());
        r * th.cos()
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
    pub fn pick<'a, T>(&mut self, v: &'a [T]) -> &'a T { &v[self.below(v.len())] }
    pub fn weighted(&mut self, w: &[f64]) -> usize {
        let tot: f64 = w.iter().sum();
        let mut x = self.f64() * tot;
        for (k, &p) in w.iter().enumerate() {
            if x < p {
                return k;
            }
            x -= p;
        }
        w.len().saturating_sub(1)
    }
    pub fn fork(&mut self, tag: u64) -> Rng { Rng::new(mix(self.next_u64(), tag)) }
}
pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() { 0.0 } else { x.iter().sum::<f64>() / x.len() as f64 }
}
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mx = mean(&x[..n]);
    let my = mean(&y[..n]);
    let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..n {
        let dx = x[k] - mx;
        let dy = y[k] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 { 0.0 } else { sxy / (sxx * syy).sqrt() }
}
pub fn show_byte(b: u8) -> String {
    match b {
        b' ' => "\u{2423}".to_string(),
        b'\n' => "\\n".to_string(),
        b'\t' => "\\t".to_string(),
        0x21..=0x7E => (b as char).to_string(),
        _ => format!("\\x{:02x}", b),
    }
}
pub fn show_bytes(bs: &[u8]) -> String {
    let mut s = String::new();
    for &b in bs {
        s.push_str(&show_byte(b));
    }
    s
}