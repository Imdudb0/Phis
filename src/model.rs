use crate::util::*;
use std::io::{Error, ErrorKind, Result as IoResult};
pub const MAGIC: &[u8; 8] = b"PHISHARP";
pub const VERSION: u32 = 1;
pub const WEPS: f64 = 0.02;
pub const WMAX: f64 = 4.0;
pub const NONRECIP: f64 = 0.25;
#[derive(Clone, Debug)]
pub struct Hyper {
    pub kmax: usize,
    pub qmax: usize,
    pub nbr: usize,
    pub budget: usize,
    pub draws: usize,
    pub ncuts: usize,
    pub grain_max: usize,
    pub max_cx: usize,
    pub seed: u64,
}
impl Default for Hyper {
    fn default() -> Hyper {
        Hyper {
            kmax: 2,
            qmax: 2,
            nbr: 6,
            budget: 12,
            draws: 8,
            ncuts: 24,
            grain_max: 2,
            max_cx: 3,
            seed: 0x5048_4953_4841_5250,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Model {
    pub np: usize,
    pub nl: usize,
    pub w: Vec<f64>,
    pub hold: f64,
    pub beta: f64,
    pub hy: Hyper,
    pub iters: u64,
    pub best: f64,
}
impl Model {
    pub fn n(&self) -> usize { self.np + self.nl }
    pub fn random(np: usize, nl: usize, hy: Hyper, rng: &mut Rng) -> Model {
        let n = np + nl;
        assert!(n >= 2, "need at least two nodes");
        let mut w = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    w[i * n + j] = 0.7 * rng.normal();
                }
            }
        }
        let mut m = Model { np, nl, w, hold: 2.0, beta: 1.0, hy, iters: 0, best: 0.0 };
        m.enforce_nonreciprocal(NONRECIP);
        m
    }
    pub fn enforce_nonreciprocal(&mut self, delta: f64) {
        let n = self.n();
        for i in 0..n {
            self.w[i * n + i] = 0.0;
            for j in (i + 1)..n {
                let a = self.w[i * n + j];
                let b = self.w[j * n + i];
                let d = a - b;
                if d.abs() < delta {
                    let push = (delta - d.abs()) * 0.5;
                    let sgn = if d > 0.0 || (d == 0.0 && (i + j) % 2 == 0) { 1.0 } else { -1.0 };
                    self.w[i * n + j] = a + sgn * push;
                    self.w[j * n + i] = b - sgn * push;
                }
            }
        }
    }
    pub fn nonreciprocity(&self) -> f64 {
        let n = self.n();
        let mut m = f64::INFINITY;
        for i in 0..n {
            for j in (i + 1)..n {
                m = m.min((self.w[i * n + j] - self.w[j * n + i]).abs());
            }
        }
        m
    }
    pub fn n_params(&self) -> usize {
        let n = self.n();
        n * (n - 1) + 2
    }
    pub fn params(&self) -> Vec<f64> {
        let n = self.n();
        let mut p = Vec::with_capacity(self.n_params());
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    p.push(self.w[i * n + j]);
                }
            }
        }
        p.push(self.hold);
        p.push(self.beta.ln());
        p
    }
    pub fn set_params(&mut self, p: &[f64]) {
        let n = self.n();
        let mut k = 0usize;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    self.w[i * n + j] = p[k].clamp(-WMAX, WMAX);
                    k += 1;
                } else {
                    self.w[i * n + j] = 0.0;
                }
            }
        }
        self.hold = p[k].clamp(0.0, 8.0);
        self.beta = p[k + 1].clamp(-2.0, 2.0).exp();
        self.enforce_nonreciprocal(NONRECIP);
    }
    pub fn describe(&self) -> String {
        format!(
            "np={} nl={} beta={:.3} hold={:.3} nonrecip>={:.3} iters={} best={:.4} B={} R={} kmax={} qmax={} grains=1..{}",
            self.np, self.nl, self.beta, self.hold, self.nonreciprocity(),
            self.iters, self.best, self.hy.budget, self.hy.draws, self.hy.kmax,
            self.hy.qmax, self.hy.grain_max
        )
    }
    pub fn save(&self, path: &str) -> IoResult<()> {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        let ints = [
            self.np, self.nl, self.hy.kmax, self.hy.qmax, self.hy.nbr,
            self.hy.budget, self.hy.draws, self.hy.ncuts, self.hy.grain_max, self.hy.max_cx,
        ];
        for v in ints {
            b.extend_from_slice(&(v as u32).to_le_bytes());
        }
        b.extend_from_slice(&self.hy.seed.to_le_bytes());
        b.extend_from_slice(&self.iters.to_le_bytes());
        for v in [self.hold, self.beta, self.best] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for &x in &self.w {
            b.extend_from_slice(&x.to_le_bytes());
        }
        if let Some(dir) = std::path::Path::new(path).parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        std::fs::write(path, b)
    }
    pub fn load(path: &str) -> IoResult<Model> {
        let bytes = std::fs::read(path)?;
        let mut r = Rd { b: &bytes, p: 0 };
        if r.take(8)? != &MAGIC[..] {
            return Err(Error::new(ErrorKind::InvalidData, "not a .phis file"));
        }
        if r.u32()? != VERSION {
            return Err(Error::new(ErrorKind::InvalidData, "unsupported .phis version"));
        }
        let np = r.u32()? as usize;
        let nl = r.u32()? as usize;
        let kmax = r.u32()? as usize;
        let qmax = r.u32()? as usize;
        let nbr = r.u32()? as usize;
        let budget = r.u32()? as usize;
        let draws = r.u32()? as usize;
        let ncuts = r.u32()? as usize;
        let grain_max = r.u32()? as usize;
        let max_cx = r.u32()? as usize;
        let seed = r.u64()?;
        let iters = r.u64()?;
        let hold = r.f64()?;
        let beta = r.f64()?;
        let best = r.f64()?;
        let n = np + nl;
        if n < 2 || n > 4096 {
            return Err(Error::new(ErrorKind::InvalidData, "bad node count"));
        }
        let mut w = Vec::with_capacity(n * n);
        for _ in 0..n * n {
            w.push(r.f64()?);
        }
        let hy = Hyper { kmax, qmax, nbr, budget, draws, ncuts, grain_max, max_cx, seed };
        Ok(Model { np, nl, w, hold, beta, hy, iters, best })
    }
}
struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Rd<'a> {
    fn take(&mut self, k: usize) -> IoResult<&'a [u8]> {
        if self.p + k > self.b.len() {
            return Err(Error::new(ErrorKind::UnexpectedEof, "truncated .phis file"));
        }
        let s = &self.b[self.p..self.p + k];
        self.p += k;
        Ok(s)
    }
    fn u32(&mut self) -> IoResult<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> IoResult<u64> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
    }
    fn f64(&mut self) -> IoResult<f64> {
        let s = self.take(8)?;
        Ok(f64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
    }
}