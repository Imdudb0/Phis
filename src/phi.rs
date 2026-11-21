use crate::graph::*;
use crate::dynamics::drive;

fn sum_norm(a: &[f32]) -> Vec<f32> {
    let s: f32 = a.iter().sum();
    if s > 0.0 { a.iter().map(|v| v / s).collect() } else { a.to_vec() }
}
fn flat_cosine(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    let mut d = 0.0; let mut na = 0.0; let mut nb = 0.0;
    for i in 0..a.len() { for j in 0..a[i].len() {
        let u = a[i][j]; let v = b[i][j];
        d += u * v; na += u * u; nb += v * v;
    }}
    d / (na.sqrt() * nb.sqrt() + 1e-9)
}
fn lesion(b: &[Vec<f32>], h: usize) -> Vec<Vec<f32>> {
    let mut bl = b.to_vec();
    for j in 0..bl.len() { bl[h][j] = 0.0; bl[j][h] = 0.0; }
    bl
}

// φ_sharp: puissance cause-effet intégrée, relative à E (structure régénérée).
pub fn compute(b: &[Vec<f32>], t: &[Vec<f32>], drv: &[f32], probe: usize, n: usize,
               budget: &mut Budget) -> (f32, f32, f32, f32) {
    if n == 0 { return (0.0, 0.0, 0.0, 0.0); }
    let fid = flat_cosine(b, t).clamp(0.0, 1.0);
    let base = sum_norm(drv);
    let mut hubs: Vec<usize> = (0..n).collect();
    hubs.sort_by(|&a, &c| drv[c].partial_cmp(&drv[a]).unwrap());
    hubs.truncate(8);
    let mut integ = f32::INFINITY;
    for h in hubs {
        let ah = sum_norm(&drive(&lesion(b, h), probe, n));
        let d = tv(&base, &ah);
        if d < integ { integ = d; }
        budget.ops += (n * n * 3) as u64;
    }
    if !integ.is_finite() { integ = 0.0; }
    let spec = (1.0 - entropy_bits(&base) / (n as f32).log2().max(1.0)).clamp(0.0, 1.0);
    let phi = (fid * integ * spec).clamp(0.0, 1.0);
    (fid, integ, spec, phi)
}

// Résidu causal seulement: structure de E non régénérée par la cause.
pub fn residual(b: &[Vec<f32>], t: &[Vec<f32>], drv: &[f32]) -> f32 {
    let mut r = 0.0; let mut tot = 0.0;
    for i in 0..t.len() { for j in 0..t[i].len() {
        if t[i][j] > 0.0 {
            tot += t[i][j];
            if b[i][j] == 0.0 { r += t[i][j] * drv[i].max(0.05); }
        }
    }}
    if tot > 0.0 { r / tot } else { 0.0 }
}

pub fn responsibility(b: &[Vec<f32>], drv: &[f32], probe: usize, n: usize) -> Vec<f32> {
    let base = sum_norm(drv);
    let mut hubs: Vec<usize> = (0..n).collect();
    hubs.sort_by(|&a, &c| drv[c].partial_cmp(&drv[a]).unwrap());
    hubs.truncate(8);
    let mut resp = vec![0.0f32; n];
    for h in hubs {
        let ah = sum_norm(&drive(&lesion(b, h), probe, n));
        for i in 0..n { resp[i] += (base[i] - ah[i]).max(0.0); }
    }
    resp
}