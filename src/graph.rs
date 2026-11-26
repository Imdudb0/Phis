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

// --- Mémoire de Travail Positionnelle (directement dans l'espace orthogonal 24D de H) ---
pub const WM_DIM: usize = STATE_DIM; // 24
pub const WM_STEPS: usize = 4;
const ECHO_IN: usize = 7;
const ECHO_HID: usize = 6;
const SCAN_IN: usize = 7;
const SCAN_CTRL: usize = 5;
const SCAN_BLOCK: usize =
    SCAN_CTRL * SCAN_IN + SCAN_CTRL * SCAN_CTRL + SCAN_CTRL + 3 * SCAN_CTRL + 3; // 83
const WM_UPD_IN: usize = 5;
const WM_UPD_HID: usize = 4;
const READ_IN: usize = 7;
const READ_HID: usize = 6;

// --- Paramètres Hop-1 (0..1541) ---
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

// --- Paramètres Mémoire de Travail (1541..1846, +305 params seulement) ---
pub const WM_PARAM_START: usize = OFF_OUT + REL_DIM * DIM; // 1541
const OFF_ECHO_W1: usize = WM_PARAM_START; // 1541, len 42
const OFF_ECHO_B1: usize = OFF_ECHO_W1 + ECHO_HID * ECHO_IN; // 1583, len 6
const OFF_ECHO_W2: usize = OFF_ECHO_B1 + ECHO_HID; // 1589, len 6
const OFF_ECHO_B2: usize = OFF_ECHO_W2 + ECHO_HID; // 1595, len 1
const OFF_SCAN_BWD: usize = OFF_ECHO_B2 + 1; // 1596, len 83
const OFF_SCAN_FWD: usize = OFF_SCAN_BWD + SCAN_BLOCK; // 1679, len 83
const OFF_WM_UPD_W1: usize = OFF_SCAN_FWD + SCAN_BLOCK; // 1762, len 20
const OFF_WM_UPD_B1: usize = OFF_WM_UPD_W1 + WM_UPD_HID * WM_UPD_IN; // 1782, len 4
const OFF_WM_UPD_W2: usize = OFF_WM_UPD_B1 + WM_UPD_HID; // 1786, len 4
const OFF_WM_UPD_B2: usize = OFF_WM_UPD_W2 + WM_UPD_HID; // 1790, len 1
const OFF_READ_W1: usize = OFF_WM_UPD_B2 + 1; // 1791, len 42
const OFF_READ_B1: usize = OFF_READ_W1 + READ_HID * READ_IN; // 1833, len 6
const OFF_READ_W2: usize = OFF_READ_B1 + READ_HID; // 1839, len 6
const OFF_READ_B2: usize = OFF_READ_W2 + READ_HID; // 1845, len 1
pub const NUM_PARAMS: usize = OFF_READ_B2 + 1; // 1846

const MAGIC: &[u8; 4] = b"PHIS";
const VERSION: u32 = 8;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-12.0, 12.0)).exp())
}

pub fn wm_norm(v: &[f32; WM_DIM]) -> f32 {
    let mut s = 0.0f32;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

pub fn wm_normalize(v: &[f32; WM_DIM]) -> [f32; WM_DIM] {
    let nrm = wm_norm(v);
    if nrm < 1e-8 {
        return [0.0; WM_DIM];
    }
    let inv = 1.0 / nrm;
    let mut out = [0.0f32; WM_DIM];
    for d in 0..WM_DIM {
        out[d] = v[d] * inv;
    }
    out
}

pub fn wm_cosine(a: &[f32; WM_DIM], b: &[f32; WM_DIM]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for d in 0..WM_DIM {
        dot += a[d] * b[d];
        na += a[d] * a[d];
        nb += b[d] * b[d];
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-8 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Alignement à norme amortie : tue les fuites microscopiques (`||v|| ~ 0.01 -> ~0.04`)
/// tout en distinguant composition exacte Hop-2 (`0.85`) > sur-ensemble (`0.71`) > Hop-1 (`0.57`).
pub fn wm_align(v: &[f32; WM_DIM], q: &[f32; WM_DIM]) -> f32 {
    let mut dot = 0.0f32;
    let mut nv = 0.0f32;
    let mut nq = 0.0f32;
    for d in 0..WM_DIM {
        dot += v[d] * q[d];
        nv += v[d] * v[d];
        nq += q[d] * q[d];
    }
    let qn = nq.sqrt();
    if qn < 1e-8 {
        return 0.0;
    }
    let denom = (nv.sqrt() + 0.25) * qn;
    (dot / denom).clamp(-1.0, 1.0)
}

/// Nouveauté orthogonale Gram-Schmidt exacte et idempotente :
/// - `u1` reçu sur `u1` -> `0` (idempotent)
/// - `u1` reçu sur `u1 + u2` -> `0` (n'efface jamais `u2`)
/// - `u1 + u2` reçu sur `u1` -> `u2` (préserve 100% la composante nouvelle)
/// - `u3` reçu sur `u1 + u2` -> `u3` (composition exacte `u1 + u2 + u3`).
pub fn orth_novelty(msg: &[f32; WM_DIM], cur: &[f32; WM_DIM]) -> [f32; WM_DIM] {
    let mut dot = 0.0f32;
    let mut m2 = 0.0f32;
    let mut z2 = 0.0f32;
    for d in 0..WM_DIM {
        dot += msg[d] * cur[d];
        m2 += msg[d] * msg[d];
        z2 += cur[d] * cur[d];
    }
    if m2 < 1e-8 {
        return [0.0; WM_DIM];
    }
    if dot <= 0.0 || z2 < 1e-8 {
        return *msg;
    }
    let gate = ((1.0 - dot / m2) * 2.0).clamp(0.0, 1.0);
    if gate < 1e-6 {
        return [0.0; WM_DIM];
    }
    let proj = (dot / (z2 + 1e-6)).min(1.0);
    let mut out = [0.0f32; WM_DIM];
    for d in 0..WM_DIM {
        out[d] = gate * (msg[d] - proj * cur[d]);
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub struct EchoEdge {
    pub src: u32,
    pub norm_len: f32,
    pub log_deg: f32,
    pub rel_dist: f32,
    pub freq_src: f32,
}

#[derive(Clone, Debug)]
pub struct RawSubstrate {
    pub len: usize,
    pub offsets: Vec<usize>,
    pub edges: Vec<[f32; EDGE_IN]>,
    pub edge_targets: Vec<u32>,
    pub deg: Vec<f32>,
    pub global_freq: Vec<f32>,
    pub back_echo_len: Vec<f32>,
    pub fwd_echo_len: Vec<f32>,
    pub echo_log_deg: Vec<f32>,
    pub echo_offsets: Vec<usize>,
    pub echo_edges: Vec<EchoEdge>,
}

impl RawSubstrate {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let n = bytes.len();
        let win_cap = 30usize;
        let deg_rad = 12usize;
        let max_edges = 12usize;
        let log_cap = (win_cap as f32).ln();

        let mut counts = [0u32; 256];
        let mut max_c = 1u32;
        for &b in bytes {
            counts[b as usize] += 1;
            if counts[b as usize] > max_c {
                max_c = counts[b as usize];
            }
        }
        let inv_max_c = 1.0 / (max_c as f32);
        let global_freq: Vec<f32> = bytes
            .iter()
            .map(|&b| counts[b as usize] as f32 * inv_max_c)
            .collect();

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

        // Diagonales d'écho longue portée (|a - b| > 16, L >= 3)
        let mut back_echo_len = vec![0.0f32; n];
        let mut fwd_echo_len = vec![0.0f32; n];
        let mut echo_deg_raw = vec![0u32; n];
        let mut raw_links: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];

        if n > 20 {
            let max_d = n.min(1024);
            for d in 17..max_d {
                let mut i = 0usize;
                while i + d < n {
                    if bytes[i] == bytes[i + d] {
                        let start = i;
                        while i + d < n && bytes[i] == bytes[i + d] {
                            i += 1;
                        }
                        let run_len = i - start;
                        if run_len >= 3 {
                            let nl = (run_len as f32 / 10.0).min(1.0);
                            for k in 0..run_len {
                                let a = start + k;
                                let b = a + d;
                                echo_deg_raw[a] += 1;
                                echo_deg_raw[b] += 1;
                                if nl > fwd_echo_len[a] {
                                    fwd_echo_len[a] = nl;
                                }
                                if nl > back_echo_len[b] {
                                    back_echo_len[b] = nl;
                                }
                                raw_links[b].push((a, run_len));
                                raw_links[a].push((b, run_len));
                            }
                        }
                    } else {
                        i += 1;
                    }
                }
            }
        }

        let echo_log_deg: Vec<f32> = echo_deg_raw
            .iter()
            .map(|&c| {
                if c == 0 {
                    0.0
                } else {
                    (((c + 1) as f32).ln() / 3.0).clamp(0.0, 1.0)
                }
            })
            .collect();

        let max_echo_per_pos = 6usize;
        let inv_n = 1.0 / (n.max(1) as f32);
        let mut echo_offsets = Vec::with_capacity(n + 1);
        let mut echo_edges = Vec::new();
        echo_offsets.push(0);

        for dst in 0..n {
            let list = &mut raw_links[dst];
            if list.len() > max_echo_per_pos {
                list.sort_unstable_by(|a, b| {
                    b.1.cmp(&a.1)
                        .then_with(|| dst.abs_diff(a.0).cmp(&dst.abs_diff(b.0)))
                });
                list.truncate(max_echo_per_pos);
            }
            for &(src, run_len) in list.iter() {
                echo_edges.push(EchoEdge {
                    src: src as u32,
                    norm_len: (run_len as f32 / 10.0).min(1.0),
                    log_deg: echo_log_deg[dst],
                    rel_dist: ((dst as f32) - (src as f32)) * inv_n,
                    freq_src: global_freq[src],
                });
            }
            echo_offsets.push(echo_edges.len());
        }

        Self {
            len: n,
            offsets,
            edges,
            edge_targets,
            deg,
            global_freq,
            back_echo_len,
            fwd_echo_len,
            echo_log_deg,
            echo_offsets,
            echo_edges,
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
            sigmoid(logit)
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

    // =========================================================================
    // MÉMOIRE DE TRAVAIL POSITIONNELLE (directement dans l'espace 24D de H)
    // =========================================================================

    fn eval_echo_weight(&self, feat: &[f32; ECHO_IN]) -> f32 {
        let w = &self.weights;
        let mut logit = w[OFF_ECHO_B2];
        for h in 0..ECHO_HID {
            let base = OFF_ECHO_W1 + h * ECHO_IN;
            let mut acc = w[OFF_ECHO_B1 + h];
            for k in 0..ECHO_IN {
                acc += w[base + k] * feat[k];
            }
            logit += w[OFF_ECHO_W2 + h] * acc.tanh();
        }
        sigmoid(logit)
    }

    fn eval_scanner_step(
        &self,
        off: usize,
        ctrl: &mut [f32; SCAN_CTRL],
        inp: &[f32; SCAN_IN],
    ) -> (f32, f32, f32) {
        let w = &self.weights;
        let off_in = off;
        let off_rec = off_in + SCAN_CTRL * SCAN_IN;
        let off_b = off_rec + SCAN_CTRL * SCAN_CTRL;
        let off_out_w = off_b + SCAN_CTRL;
        let off_out_b = off_out_w + 3 * SCAN_CTRL;

        let mut next_ctrl = [0.0f32; SCAN_CTRL];
        for h in 0..SCAN_CTRL {
            let mut acc = w[off_b + h];
            for k in 0..SCAN_IN {
                acc += w[off_in + h * SCAN_IN + k] * inp[k];
            }
            for j in 0..SCAN_CTRL {
                acc += w[off_rec + h * SCAN_CTRL + j] * ctrl[j];
            }
            next_ctrl[h] = acc.tanh();
        }
        *ctrl = next_ctrl;

        let mut gates = [0.0f32; 3];
        // Prior d'init non-évanescent sur `keep` (+1.5) pour franchir 10-15 octets dès l'étape 1
        let init_bias = [1.5f32, 0.4f32, 0.4f32];
        for g in 0..3 {
            let mut acc = w[off_out_b + g] + init_bias[g];
            for h in 0..SCAN_CTRL {
                acc += w[off_out_w + g * SCAN_CTRL + h] * ctrl[h];
            }
            gates[g] = sigmoid(acc);
        }
        (gates[0], gates[1], gates[2])
    }

    /// Exécute `WM_STEPS` itérations de la mémoire de travail :
    /// Sub-step A : transport par arêtes d'écho longue portée (appliqué immédiatement)
    /// Sub-step B : balayages directionnels arrière (`←`) et avant (`→`) non-réciproques
    /// Fusion par nouveauté orthogonale Gram-Schmidt exacte.
    pub fn propagate_working_memory(
        &self,
        sub: &RawSubstrate,
        gates_pt: &[f32],
        init_z: &[[f32; WM_DIM]],
        cut_echoes: bool,
    ) -> Vec<[f32; WM_DIM]> {
        let n = sub.len;
        if n == 0 {
            return Vec::new();
        }
        let w = &self.weights;
        let mut z = init_z.to_vec();

        let mut msg_echo = vec![[0.0f32; WM_DIM]; n];
        let mut msg_bwd = vec![[0.0f32; WM_DIM]; n];
        let mut msg_fwd = vec![[0.0f32; WM_DIM]; n];
        let mut norms = vec![0.0f32; n];

        for _ in 0..WM_STEPS {
            for t in 0..n {
                norms[t] = wm_norm(&z[t]);
            }

            // Sub-step A : transport synchrone par échos longue portée de M(a,b), appliqué avant les scans
            if !cut_echoes {
                for dst in 0..n {
                    msg_echo[dst] = [0.0; WM_DIM];
                    let slice = &sub.echo_edges[sub.echo_offsets[dst]..sub.echo_offsets[dst + 1]];
                    if slice.is_empty() {
                        continue;
                    }
                    let mut acc = [0.0f32; WM_DIM];
                    for edge in slice {
                        let src = edge.src as usize;
                        if norms[src] < 0.05 {
                            continue;
                        }
                        let feat = [
                            edge.norm_len,
                            edge.log_deg,
                            edge.rel_dist,
                            edge.freq_src,
                            (norms[src] / 2.0).min(1.0),
                            (norms[dst] / 2.0).min(1.0),
                            gates_pt[src],
                        ];
                        let alpha = self.eval_echo_weight(&feat);
                        if alpha > 0.05 {
                            let nov = orth_novelty(&z[src], &acc);
                            for d in 0..WM_DIM {
                                acc[d] += alpha * nov[d];
                            }
                        }
                    }
                    msg_echo[dst] = acc;
                }
                for t in 0..n {
                    if wm_norm(&msg_echo[t]) > 1e-4 {
                        let nov = orth_novelty(&msg_echo[t], &z[t]);
                        for d in 0..WM_DIM {
                            z[t][d] += nov[d];
                        }
                        let nrm = wm_norm(&z[t]);
                        if nrm > 3.0 {
                            let sc = 3.0 / nrm;
                            for x in &mut z[t] {
                                *x *= sc;
                            }
                        }
                        norms[t] = nrm.min(3.0);
                    }
                }
            }

            // Sub-step B1 : balayage directionnel arrière (t = n-1 -> 0)
            let mut ctrl_b = [0.0f32; SCAN_CTRL];
            let mut carry_b = [0.0f32; WM_DIM];
            for t in (0..n).rev() {
                let c_norm = (wm_norm(&carry_b) / 2.0).min(1.0);
                let inp = [
                    gates_pt[t],
                    sub.back_echo_len[t],
                    sub.fwd_echo_len[t],
                    sub.echo_log_deg[t],
                    sub.global_freq[t],
                    (norms[t] / 2.0).min(1.0),
                    c_norm,
                ];
                let (keep, absorb, write) = self.eval_scanner_step(OFF_SCAN_BWD, &mut ctrl_b, &inp);

                for d in 0..WM_DIM {
                    carry_b[d] *= keep;
                }
                if norms[t] > 0.05 {
                    let nov = orth_novelty(&z[t], &carry_b);
                    for d in 0..WM_DIM {
                        carry_b[d] += absorb * nov[d];
                    }
                    let cn = wm_norm(&carry_b);
                    if cn > 3.0 {
                        let sc = 3.0 / cn;
                        for x in &mut carry_b {
                            *x *= sc;
                        }
                    }
                }
                for d in 0..WM_DIM {
                    msg_bwd[t][d] = write * carry_b[d];
                }
            }

            // Sub-step B2 : balayage directionnel avant (t = 0 -> n-1)
            let mut ctrl_f = [0.0f32; SCAN_CTRL];
            let mut carry_f = [0.0f32; WM_DIM];
            for t in 0..n {
                let c_norm = (wm_norm(&carry_f) / 2.0).min(1.0);
                let inp = [
                    gates_pt[t],
                    sub.back_echo_len[t],
                    sub.fwd_echo_len[t],
                    sub.echo_log_deg[t],
                    sub.global_freq[t],
                    (norms[t] / 2.0).min(1.0),
                    c_norm,
                ];
                let (keep, absorb, write) = self.eval_scanner_step(OFF_SCAN_FWD, &mut ctrl_f, &inp);

                for d in 0..WM_DIM {
                    carry_f[d] *= keep;
                }
                if norms[t] > 0.05 {
                    let nov = orth_novelty(&z[t], &carry_f);
                    for d in 0..WM_DIM {
                        carry_f[d] += absorb * nov[d];
                    }
                    let cn = wm_norm(&carry_f);
                    if cn > 3.0 {
                        let sc = 3.0 / cn;
                        for x in &mut carry_f {
                            *x *= sc;
                        }
                    }
                }
                for d in 0..WM_DIM {
                    msg_fwd[t][d] = write * carry_f[d];
                }
            }

            // Sub-step C : fusion orthogonale dans Z
            for t in 0..n {
                let nb = (wm_norm(&msg_bwd[t]) / 2.0).min(1.0);
                let nf = (wm_norm(&msg_fwd[t]) / 2.0).min(1.0);
                if nb + nf < 1e-4 {
                    continue;
                }
                let upd_in = [
                    (norms[t] / 2.0).min(1.0),
                    nb,
                    nf,
                    gates_pt[t],
                    sub.back_echo_len[t],
                ];
                let mut logit = w[OFF_WM_UPD_B2] + 0.5;
                for h in 0..WM_UPD_HID {
                    let base = OFF_WM_UPD_W1 + h * WM_UPD_IN;
                    let mut acc = w[OFF_WM_UPD_B1 + h];
                    for k in 0..WM_UPD_IN {
                        acc += w[base + k] * upd_in[k];
                    }
                    logit += w[OFF_WM_UPD_W2 + h] * acc.tanh();
                }
                let gamma = sigmoid(logit);

                let mut z_next = z[t];
                let nov_b = orth_novelty(&msg_bwd[t], &z_next);
                for d in 0..WM_DIM {
                    z_next[d] += gamma * nov_b[d];
                }
                let nov_f = orth_novelty(&msg_fwd[t], &z_next);
                for d in 0..WM_DIM {
                    z_next[d] += gamma * nov_f[d];
                }

                let nrm = wm_norm(&z_next);
                if nrm > 3.0 {
                    let sc = 3.0 / nrm;
                    for x in &mut z_next {
                        *x *= sc;
                    }
                }
                z[t] = z_next;
            }
        }

        z
    }

    /// Tête de lecture connexionniste inspirée IIT : compare l'alignement intact (`z_full`)
    /// et l'alignement coupé (`z_cut`) pour unifier Hop-1 et Hop-2/Hop-3 en un seul champ `P(t)`.
    pub fn readout_explanation(
        &self,
        sub: &RawSubstrate,
        gates_pt: &[f32],
        z_full: &[[f32; WM_DIM]],
        z_cut: &[[f32; WM_DIM]],
        z_query: &[f32; WM_DIM],
    ) -> Vec<f32> {
        let w = &self.weights;
        let n = sub.len;
        let mut out = Vec::with_capacity(n);

        for t in 0..n {
            let align_full = wm_align(&z_full[t], z_query);
            let align_cut = wm_align(&z_cut[t], z_query);
            let align_delta = (align_full - align_cut).max(0.0);
            let nrm = (wm_norm(&z_full[t]) / 2.0).min(1.0);
            let feat = [
                align_full,
                align_cut,
                align_delta,
                nrm,
                gates_pt[t],
                sub.back_echo_len[t],
                sub.global_freq[t],
            ];

            let mut logit = w[OFF_READ_B2];
            for h in 0..READ_HID {
                let base = OFF_READ_W1 + h * READ_IN;
                let mut acc = w[OFF_READ_B1 + h];
                for k in 0..READ_IN {
                    acc += w[base + k] * feat[k];
                }
                logit += w[OFF_READ_W2 + h] * acc.tanh();
            }
            out.push(sigmoid(logit));
        }

        out
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
                "Version ou taille du modèle .phis incompatible (ré-entraîne avec v0.7.1)",
            ));
        }
        let mut weights = Vec::with_capacity(count);
        for chunk in data[12..].chunks_exact(4) {
            weights.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(Self { weights })
    }
}
