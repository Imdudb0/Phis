use crate::rng::Rng;
use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;

pub const NODES: usize = 6;
pub const DIM: usize = 4;
pub const REL_DIM: usize = 8;
pub const SEARCH_STEPS: usize = 6;
pub const STATE_DIM: usize = NODES * DIM;
pub const INPUT_DIM: usize = REL_DIM * 3;

const OFF_EDGE: usize = 0; // 576
const OFF_BIAS: usize = OFF_EDGE + NODES * NODES * DIM * DIM; // 576, len 24
const OFF_IN: usize = OFF_BIAS + NODES * DIM; // 600, len 576
const OFF_G1: usize = OFF_IN + NODES * DIM * INPUT_DIM; // 1176, len 32
const OFF_G2: usize = OFF_G1 + DIM * (2 * DIM); // 1208, len 32
const OFF_BH: usize = OFF_G2 + DIM * (2 * DIM); // 1240, len 32
const OFF_CTX: usize = OFF_BH + DIM * (2 * DIM); // 1272, len 32
const OFF_U: usize = OFF_CTX + DIM * REL_DIM; // 1304, len 16
const OFF_OUT: usize = OFF_U + DIM * DIM; // 1320, len 32
pub const NUM_PARAMS: usize = OFF_OUT + REL_DIM * DIM; // 1352

const MAGIC: &[u8; 4] = b"PHIS";
const VERSION: u32 = 3;

#[derive(Clone, Debug)]
pub struct PhiGraph {
    pub weights: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct AbductionTrace {
    pub final_state: [f32; STATE_DIM],
    pub step_residuals: Vec<f32>,
    pub final_pos_err: Vec<f32>,
    pub final_mean_err: f32,
}

/// Champ purement relationnel multi-échelle (portée 256 octets, invariant par bijection affine).
pub fn extract_relations(bytes: &[u8]) -> Vec<[f32; REL_DIM]> {
    let n = bytes.len();
    let mut freq = [0u32; 256];
    let mut max_freq = 1u32;
    for &b in bytes {
        freq[b as usize] += 1;
        if freq[b as usize] > max_freq {
            max_freq = freq[b as usize];
        }
    }
    let max_f = max_freq as f32;
    let win_cap = 256usize;
    let log_cap = (n.min(win_cap).max(2) as f32).ln();

    let mut out = Vec::with_capacity(n);
    for t in 0..n {
        let b = bytes[t];
        let start = t.saturating_sub(win_cap);

        let const_step = if t >= 2 {
            let d1 = b.wrapping_sub(bytes[t - 1]);
            let d0 = bytes[t - 1].wrapping_sub(bytes[t - 2]);
            if d1 != 0 && d1 == d0 { 1.0 } else { -1.0 }
        } else {
            -1.0
        };

        let mut last_uni: Option<usize> = None;
        for j in (start..t).rev() {
            if bytes[j] == b {
                last_uni = Some(t - j);
                break;
            }
        }
        let uni_rec = match last_uni {
            Some(dist) => 1.0 - ((dist as f32).ln() / log_cap).clamp(0.0, 1.0),
            None => -1.0,
        };

        let mut bi_dist: Option<usize> = None;
        if t >= 1 {
            let b_prev = bytes[t - 1];
            for j in (start + 1..t).rev() {
                if bytes[j] == b && bytes[j - 1] == b_prev {
                    bi_dist = Some(t - j);
                    break;
                }
            }
        }
        let bi_rec = if bi_dist.is_some() { 1.0 } else { -1.0 };
        let scale_rec = match bi_dist {
            Some(dist) => 2.0 * ((dist as f32).ln() / log_cap).clamp(0.0, 1.0) - 1.0,
            None => -1.0,
        };

        let mut tri_hit = false;
        if t >= 2 {
            let b1 = bytes[t - 1];
            let b2 = bytes[t - 2];
            for j in (start + 2..t).rev() {
                if bytes[j] == b && bytes[j - 1] == b1 && bytes[j - 2] == b2 {
                    tri_hit = true;
                    break;
                }
            }
        }
        let tri_rec = if tri_hit { 1.0 } else { -1.0 };

        let mut mirror_hit = false;
        if t >= 1 {
            let b_prev = bytes[t - 1];
            if b == b_prev {
                mirror_hit = true;
            } else {
                let m_start = t.saturating_sub(24);
                for j in (m_start + 1..t).rev() {
                    if bytes[j - 1] == b && bytes[j] == b_prev {
                        mirror_hit = true;
                        break;
                    }
                }
            }
        }
        let mirror = if mirror_hit { 1.0 } else { -1.0 };

        let hub = 2.0 * (freq[b as usize] as f32 / max_f) - 1.0;
        let boundary = if t >= 1 {
            (freq[b as usize] as f32 - freq[bytes[t - 1] as usize] as f32) / max_f
        } else {
            0.0
        };

        out.push([
            const_step, uni_rec, bi_rec, tri_rec, mirror, hub, boundary, scale_rec,
        ]);
    }
    out
}

impl PhiGraph {
    pub fn random(rng: &mut Rng) -> Self {
        let mut weights = Vec::with_capacity(NUM_PARAMS);
        for _ in 0..NUM_PARAMS {
            weights.push(rng.next_normal() * 0.22);
        }
        Self { weights }
    }

    pub fn from_weights(weights: Vec<f32>) -> Self {
        assert_eq!(weights.len(), NUM_PARAMS);
        Self { weights }
    }

    pub fn evaluate_hypothesis(
        &self,
        state: &[f32; STATE_DIM],
        rels: &[[f32; REL_DIM]],
    ) -> (Vec<[f32; REL_DIM]>, Vec<f32>, f32) {
        let w = &self.weights;

        let mut g1 = [0.0f32; DIM];
        let mut g2 = [0.0f32; DIM];
        let mut bh = [0.0f32; DIM];

        for d in 0..DIM {
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            let mut s3 = 0.0f32;
            for k in 0..(2 * DIM) {
                s1 += w[OFF_G1 + d * (2 * DIM) + k] * state[k];
                s2 += w[OFF_G2 + d * (2 * DIM) + k] * state[2 * DIM + k];
                s3 += w[OFF_BH + d * (2 * DIM) + k] * state[4 * DIM + k];
            }
            g1[d] = s1.tanh();
            g2[d] = s2.tanh();
            bh[d] = s3.tanh();
        }

        let mut u = [0.0f32; DIM];
        let mut preds = Vec::with_capacity(rels.len());
        let mut pos_err = Vec::with_capacity(rels.len());
        let mut total_err = 0.0f32;

        for t in 0..rels.len() {
            let prev = if t == 0 {
                [0.0f32; REL_DIM]
            } else {
                rels[t - 1]
            };

            let mut u_next = [0.0f32; DIM];
            for d in 0..DIM {
                let mut ctx = 0.0f32;
                for r in 0..REL_DIM {
                    ctx += w[OFF_CTX + d * REL_DIM + r] * prev[r];
                }
                let mut rec = 0.0f32;
                for k in 0..DIM {
                    rec += w[OFF_U + d * DIM + k] * u[k];
                }
                u_next[d] = (g1[d] * ctx + g2[d] * rec + bh[d]).tanh();
            }
            u = u_next;

            let mut pred = [0.0f32; REL_DIM];
            let mut err_t = 0.0f32;
            for r in 0..REL_DIM {
                let mut acc = 0.0f32;
                for d in 0..DIM {
                    acc += w[OFF_OUT + r * DIM + d] * u[d];
                }
                pred[r] = acc.tanh();
                let diff = rels[t][r] - pred[r];
                err_t += diff * diff;
            }
            err_t /= REL_DIM as f32;
            preds.push(pred);
            pos_err.push(err_t);
            total_err += err_t;
        }

        let mean_err = if rels.is_empty() {
            0.0
        } else {
            total_err / (rels.len() as f32)
        };
        (preds, pos_err, mean_err)
    }

    pub fn abduce(&self, rels: &[[f32; REL_DIM]], cut_mask: Option<u8>) -> AbductionTrace {
        let w = &self.weights;
        let mut state = [0.0f32; STATE_DIM];
        let mut step_residuals = Vec::with_capacity(SEARCH_STEPS + 1);

        let mid = (rels.len() / 2).max(1);
        let inv_a = 1.0 / (mid as f32);
        let inv_b = 1.0 / (rels.len().saturating_sub(mid).max(1) as f32);
        let len_inv = 1.0 / (rels.len().max(1) as f32);

        for _ in 0..SEARCH_STEPS {
            let (preds, _, mean_err) = self.evaluate_hypothesis(&state, rels);
            step_residuals.push(mean_err);

            let mut inp = [0.0f32; INPUT_DIM];
            for t in 0..rels.len() {
                let prev = if t == 0 {
                    [0.0f32; REL_DIM]
                } else {
                    rels[t - 1]
                };
                for r in 0..REL_DIM {
                    let err = rels[t][r] - preds[t][r];
                    if t < mid {
                        inp[r] += err * inv_a;
                    } else {
                        inp[REL_DIM + r] += err * inv_b;
                    }
                    inp[2 * REL_DIM + r] += (prev[r] * err) * len_inv;
                }
            }

            let mut next_state = [0.0f32; STATE_DIM];
            for i in 0..NODES {
                let part_i = cut_mask.map(|m| (m >> i) & 1);
                for d in 0..DIM {
                    let mut acc = w[OFF_BIAS + i * DIM + d];

                    for j in 0..NODES {
                        if let Some(pi) = part_i {
                            let pj = (cut_mask.unwrap() >> j) & 1;
                            if i != j && pi != pj {
                                continue;
                            }
                        }
                        let base = OFF_EDGE + (i * NODES + j) * DIM * DIM + d * DIM;
                        for k in 0..DIM {
                            acc += w[base + k] * state[j * DIM + k];
                        }
                    }

                    let in_base = OFF_IN + (i * DIM + d) * INPUT_DIM;
                    for m in 0..INPUT_DIM {
                        acc += w[in_base + m] * inp[m];
                    }

                    let idx = i * DIM + d;
                    next_state[idx] = 0.35 * state[idx] + 0.65 * acc.tanh();
                }
            }
            state = next_state;
        }

        let (_, final_pos_err, final_mean_err) = self.evaluate_hypothesis(&state, rels);
        step_residuals.push(final_mean_err);

        AbductionTrace {
            final_state: state,
            step_residuals,
            final_pos_err,
            final_mean_err,
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        if let Some(parent) = path.as_ref().parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let mut buf = Vec::with_capacity(12 + NUM_PARAMS * 4);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(NUM_PARAMS as u32).to_le_bytes());
        for &val in &self.weights {
            buf.extend_from_slice(&val.to_le_bytes());
        }
        fs::write(path, buf)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let data = fs::read(path)?;
        if data.len() < 12 || &data[0..4] != MAGIC {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Fichier .phis invalide (en-tête manquant)",
            ));
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let count = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        if version != VERSION || count != NUM_PARAMS || data.len() != 12 + count * 4 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Version ou taille du modèle .phis incompatible (ré-entraîne avec v0.3)",
            ));
        }
        let mut weights = Vec::with_capacity(count);
        for chunk in data[12..].chunks_exact(4) {
            weights.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(Self { weights })
    }
}
