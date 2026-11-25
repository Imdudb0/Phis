//! Utilitaires sans dépendance : PRNG, hachage, budget mesuré, numérique, horodatage.

use std::time::{SystemTime, UNIX_EPOCH};

const INV53: f64 = 1.0 / 9_007_199_254_740_992.0;

/// SplitMix64 : déterministe et portable.
#[derive(Clone, Debug)]
pub struct Rng {
    s: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { s: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * INV53
    }

    pub fn below(&mut self, n: usize) -> usize {
        if n <= 1 { 0 } else { (self.next_u64() % (n as u64)) as usize }
    }

    /// Entier uniforme dans [lo, hi], bornes incluses.
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo { lo } else { lo + self.below(hi - lo + 1) }
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    pub fn normal(&mut self) -> f64 {
        let u1 = self.f64().max(1e-300);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        let n = v.len();
        for i in (1..n).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }

    /// Tirage catégoriel proportionnel aux poids (poids nuls ou vides : uniforme).
    pub fn weighted(&mut self, w: &[f64]) -> usize {
        if w.is_empty() {
            return 0;
        }
        let tot: f64 = w.iter().sum();
        if !(tot > 0.0) {
            return self.below(w.len());
        }
        let mut x = self.f64() * tot;
        for (i, &wi) in w.iter().enumerate() {
            if x < wi {
                return i;
            }
            x -= wi;
        }
        w.len() - 1
    }

    pub fn fork(&mut self) -> Rng {
        Rng::new(self.next_u64())
    }
}

/// FNV-1a 64 bits (empreintes d'intégrité, pas de cryptographie).
pub fn fnv64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn mix64(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_add(a << 6).wrapping_add(a >> 2);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^ (z >> 29)
}

/// Budget mesuré en « ticks » : unités abstraites d'opérations élémentaires
/// (candidat évalué, nœud créé, nœud visité). Indépendant de l'horloge.
#[derive(Clone, Debug, Default)]
pub struct Meter {
    pub ticks: u64,
    pub limit: u64,
}

impl Meter {
    pub fn new(limit: u64) -> Meter {
        Meter { ticks: 0, limit }
    }
    pub fn spend(&mut self, n: u64) {
        self.ticks = self.ticks.saturating_add(n);
    }
    pub fn exhausted(&self) -> bool {
        self.ticks >= self.limit
    }
    pub fn used(&self) -> f64 {
        if self.limit == 0 { 1.0 } else { (self.ticks as f64 / self.limit as f64).min(1.0) }
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}

pub fn std_dev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}

pub fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// log-somme-exp (base e).
pub fn lse(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NEG_INFINITY;
    }
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
}

pub fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Date ISO-8601 UTC (algorithme civil-from-days), pour dater le protocole.
pub fn iso_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    if mo <= 2 {
        y += 1;
    }
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, m, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_deterministe() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn date_epoch() {
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
    }
}