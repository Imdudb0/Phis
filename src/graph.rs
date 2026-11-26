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

pub const EDGE_IN: usize = 12;
pub const EDGE_HID: usize = 4;
pub const NODE_IN: usize = EDGE_HID * 2 + 2;

pub const GATE_IN: usize = EDGE_HID * 2 + 2;
pub const GATE_HID: usize = 4;

const OFF_REL_EDGE_W: usize = 0; // 48
const OFF_REL_EDGE_B: usize = OFF_REL_EDGE_W + EDGE_HID * EDGE_IN; // 48, len 4
const OFF_REL_NODE_W: usize = OFF_REL_EDGE_B + EDGE_HID; // 52, len 80
const OFF_REL_NODE_B: usize = OFF_REL_NODE_W + REL_DIM * NODE_IN; // 132, len 8

const OFF_GATE_W1: usize = OFF_REL_NODE_B + REL_DIM; // 140, len 40
const OFF_GATE_B1: usize = OFF_GATE_W1 + GATE_HID * GATE_IN; // 180, len 4
const OFF_GATE_W2: usize = OFF_GATE_B1 + GATE_HID; // 184, len 4
const OFF_GATE_B2: usize = OFF_GATE_W2 + GATE_HID; // 188, len 1

const OFF_EDGE: usize = OFF_GATE_B2 + 1; // 189, len 576
const OFF_BIAS: usize = OFF_EDGE + NODES * NODES * DIM * DIM; // 765, len 24
const OFF_IN: usize = OFF_BIAS + NODES * DIM; // 789, len 576
const OFF_G1: usize = OFF_IN + NODES * DIM * INPUT_DIM; // 1365, len 32
const OFF_G2: usize = OFF_G1 + DIM * (2 * DIM); // 1397, len 32
const OFF_BH: usize = OFF_G2 + DIM * (2 * DIM); // 1429, len 32
const OFF_CTX: usize = OFF_BH + DIM * (2 * DIM); // 1461, len 32
const OFF_U: usize = OFF_CTX + DIM * REL_DIM; // 1493, len 16
const OFF_OUT: usize = OFF_U + DIM * DIM; // 1509, len 32
pub const NUM_PARAMS: usize = OFF_OUT + REL_DIM * DIM; // 1541

const MAGIC: &[u8; 4] = b"PHIS";
const VERSION: u32 = 5;

/// Substrat relationnel brut à échelle invariante (fenêtre fixe ±24, degré local ±12).
#[derive(Clone, Debug)]
pub struct RawSubstrate {
    pub len: usize,
    pub offsets: Vec<usize>,
    pub edges: Vec<[f32; EDGE_IN]>,
    pub edge_targets: Vec<u32>,
    pub deg: Vec<f32>,
}

impl RawSubstrate {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let n = bytes.len();
        let win_cap = 24usize;
        let deg_rad = 12usize;
        let max_edges = 12usize;
        let log_cap = (win_cap as f32).ln();

        let m_nb = |a: isize, b: isize, ti: isize, ji: isize| -> f32 {
            if a >= 0 && b >= 0 && a != b && !(a == ji && b == ti) {
                let au = a as usize;
                let bu = b as usize;
                if au < n && bu < n && bytes[au] == bytes[bu] {
                    return 1.0;
                }
            }
            0.0
        };

        let mut offsets = Vec::with_capacity(n + 1);
        let mut edges = Vec::with_capacity(n * 6);
        let mut edge_targets = Vec::with_capacity(n * 6);
        let mut deg = Vec::with_capacity(n);
        let mut candidates: Vec<(usize, usize, f32)> = Vec::with_capacity(32);

        offsets.push(0);
        for t in 0..n {
            let b = bytes[t];
            let lo = t.saturating_sub(win_cap);
            let hi = (t + win_cap + 1).min(n);
            let deg_lo = t.saturating_sub(deg_rad);
            let deg_hi = (t + deg_rad + 1).min(n);
            let deg_span = (deg_hi - deg_lo).saturating_sub(1).max(1);
            let ti = t as isize;

            candidates.clear();
            let mut local_matches = 0usize;
            for j in lo..hi {
                if j != t && bytes[j] == b {
                    let dist = t.abs_diff(j);
                    if dist <= deg_rad {
                        local_matches += 1;
                    }
                    let ji = j as isize;
                    let neigh = m_nb(ti - 1, ji - 1, ti, ji)
                        + m_nb(ti + 1, ji + 1, ti, ji)
                        + m_nb(ti - 2, ji - 2, ti, ji)
                        + m_nb(ti + 2, ji + 2, ti, ji)
                        + m_nb(ti - 1, ji + 1, ti, ji)
                        + m_nb(ti + 1, ji - 1, ti, ji)
                        + m_nb(ti - 2, ji + 2, ti, ji)
                        + m_nb(ti + 2, ji - 2, ti, ji);
                    candidates.push((j, dist, neigh));
                }
            }

            deg.push(2.0 * (local_matches as f32 / deg_span as f32) - 1.0);

            let match_count = candidates.len();
            let mut selected: Vec<usize> = Vec::with_capacity(max_edges.min(match_count));
            if match_count <= max_edges {
                for &(j, _, _) in &candidates {
                    selected.push(j);
                }
            } else {
                let mut by_dist = candidates.clone();
                by_dist.sort_unstable_by_key(|c| c.1);
                for &(j, _, _) in by_dist.iter().take(6) {
                    selected.push(j);
                }
                candidates.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.1.cmp(&b.1)));
                for &(j, _, _) in &candidates {
                    if selected.len() >= max_edges {
                        break;
                    }
                    if !selected.contains(&j) {
                        selected.push(j);
                    }
                }
            }

            for j in selected {
                let ji = j as isize;
                let dist = t.abs_diff(j) as f32;
                let log_dist = (dist.ln() / log_cap).clamp(0.0, 1.0);
                let dir = if j < t { -1.0 } else { 1.0 };
                edges.push([
                    m_nb(ti - 1, ji - 1, ti, ji),
                    m_nb(ti + 1, ji + 1, ti, ji),
                    m_nb(ti - 2, ji - 2, ti, ji),
                    m_nb(ti + 2, ji + 2, ti, ji),
                    m_nb(ti - 1, ji + 1, ti, ji),
                    m_nb(ti + 1, ji - 1, ti, ji),
                    m_nb(ti - 2, ji + 2, ti, ji),
                    m_nb(ti + 2, ji - 2, ti, ji),
                    m_nb(ti - 1, ji, ti, ji),
                    m_nb(ti, ji - 1, ti, ji),
                    log_dist,
                    dir,
                ]);
                edge_targets.push(j as u32);
            }
            offsets.push(edges.len());
        }

        Self {
            len: n,
            offsets,
            edges,
            edge_targets,
            deg,
        }
    }
}

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

    /// Capteur relationnel 100% appris + porte causale neuronale contextuelle (`±2`) et ponctuelle.
    pub fn encode_with_gate(&self, sub: &RawSubstrate) -> (Vec<[f32; REL_DIM]>, Vec<f32>, Vec<f32>) {
        let w = &self.weights;
        let n = sub.len;
        if n == 0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let mut pre = Vec::with_capacity(n);
        let mut e_max_all = Vec::with_capacity(n);
        let mut has_all = Vec::with_capacity(n);
        let mut energy = 0.0f32;

        for t in 0..n {
            let slice = &sub.edges[sub.offsets[t]..sub.offsets[t + 1]];
            let mut e_max = [-1.0f32; EDGE_HID];
            let mut e_mean = [-1.0f32; EDGE_HID];
            let has_edge;

            if !slice.is_empty() {
                has_edge = 1.0f32;
                e_max = [f32::NEG_INFINITY; EDGE_HID];
                e_mean = [0.0f32; EDGE_HID];
                let inv_c = 1.0 / (slice.len() as f32);

                for feat in slice {
                    for h in 0..EDGE_HID {
                        let base = OFF_REL_EDGE_W + h * EDGE_IN;
                        let mut acc = w[OFF_REL_EDGE_B + h];
                        for k in 0..EDGE_IN {
                            acc += w[base + k] * feat[k];
                        }
                        let val = acc.tanh();
                        if val > e_max[h] {
                            e_max[h] = val;
                        }
                        e_mean[h] += val * inv_c;
                    }
                }
            } else {
                has_edge = -1.0f32;
            }

            e_max_all.push(e_max);
            has_all.push(has_edge);

            let node_in = [
                e_max[0],
                e_max[1],
                e_max[2],
                e_max[3],
                e_mean[0],
                e_mean[1],
                e_mean[2],
                e_mean[3],
                sub.deg[t],
                has_edge,
            ];

            let mut row = [0.0f32; REL_DIM];
            for r in 0..REL_DIM {
                let base = OFF_REL_NODE_W + r * NODE_IN;
                let mut acc = 0.25 * w[OFF_REL_NODE_B + r];
                for k in 0..NODE_IN {
                    acc += w[base + k] * node_in[k];
                }
                row[r] = acc;
                energy += acc * acc;
            }
            pre.push(row);
        }

        let rms = (energy / ((n * REL_DIM) as f32) + 1e-4).sqrt();
        let inv = 1.25 / rms;
        for row in &mut pre {
            for val in row.iter_mut() {
                *val = (*val * inv).tanh();
            }
        }

        let eval_gate = |g_in: &[f32; GATE_IN]| -> f32 {
            let mut logit = w[OFF_GATE_B2];
            for h in 0..GATE_HID {
                let base = OFF_GATE_W1 + h * GATE_IN;
                let mut acc = w[OFF_GATE_B1 + h];
                for k in 0..GATE_IN {
                    acc += w[base + k] * g_in[k];
                }
                logit += w[OFF_GATE_W2 + h] * acc.tanh();
            }
            1.0 / (1.0 + (-logit.clamp(-12.0, 12.0)).exp())
        };

        let mut gates_ctx = Vec::with_capacity(n);
        let mut gates_pt = Vec::with_capacity(n);
        for t in 0..n {
            let lo = t.saturating_sub(2);
            let hi = (t + 2).min(n - 1);
            let inv_w = 1.0 / ((hi - lo + 1) as f32);

            let mut loc_max = [0.0f32; EDGE_HID];
            let mut loc_deg = 0.0f32;
            let mut loc_has = 0.0f32;
            for j in lo..=hi {
                for h in 0..EDGE_HID {
                    loc_max[h] += e_max_all[j][h] * inv_w;
                }
                loc_deg += sub.deg[j] * inv_w;
                loc_has += has_all[j] * inv_w;
            }

            let g_in_ctx = [
                loc_max[0],
                loc_max[1],
                loc_max[2],
                loc_max[3],
                e_max_all[t][0],
                e_max_all[t][1],
                e_max_all[t][2],
                e_max_all[t][3],
                loc_deg,
                loc_has,
            ];
            let g_in_pt = [
                e_max_all[t][0],
                e_max_all[t][1],
                e_max_all[t][2],
                e_max_all[t][3],
                e_max_all[t][0],
                e_max_all[t][1],
                e_max_all[t][2],
                e_max_all[t][3],
                sub.deg[t],
                has_all[t],
            ];
            gates_ctx.push(eval_gate(&g_in_ctx));
            gates_pt.push(eval_gate(&g_in_pt));
        }

        (pre, gates_ctx, gates_pt)
    }

    pub fn encode_relations(&self, sub: &RawSubstrate) -> Vec<[f32; REL_DIM]> {
        self.encode_with_gate(sub).0
    }

    pub fn encode_bytes(&self, bytes: &[u8]) -> Vec<[f32; REL_DIM]> {
        let sub = RawSubstrate::from_bytes(bytes);
        self.encode_relations(&sub)
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

    pub fn abduce_with_steps(
        &self,
        rels: &[[f32; REL_DIM]],
        cut_mask: Option<u8>,
        steps: usize,
    ) -> AbductionTrace {
        let w = &self.weights;
        let mut state = [0.0f32; STATE_DIM];
        let mut step_residuals = Vec::with_capacity(steps + 1);

        let mid = (rels.len() / 2).max(1);
        let inv_a = 1.0 / (mid as f32);
        let inv_b = 1.0 / (rels.len().saturating_sub(mid).max(1) as f32);
        let len_inv = 1.0 / (rels.len().max(1) as f32);

        for _ in 0..steps {
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

    pub fn abduce(&self, rels: &[[f32; REL_DIM]], cut_mask: Option<u8>) -> AbductionTrace {
        self.abduce_with_steps(rels, cut_mask, SEARCH_STEPS)
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
                "Version ou taille du modèle .phis incompatible (ré-entraîne avec v0.5.2)",
            ));
        }
        let mut weights = Vec::with_capacity(count);
        for chunk in data[12..].chunks_exact(4) {
            weights.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(Self { weights })
    }
}
