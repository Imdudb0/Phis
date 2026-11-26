use crate::graph::{
    AbductionTrace, NODES, PhiGraph, REL_DIM, RawSubstrate, SEARCH_STEPS, STATE_DIM, WM_DIM,
    orth_novelty, wm_cosine, wm_norm,
};
use crate::rng::Rng;

const SINGLETON_MASKS: [u8; 6] = [
    0b000001, 0b000010, 0b000100, 0b001000, 0b010000, 0b011111,
];

#[derive(Clone, Copy, Debug)]
pub struct DiscoveredRegime {
    pub core_start: usize,
    pub core_end: usize,
    pub referent: Option<(usize, usize)>,
    pub mean_gate: f32,
}

#[derive(Clone, Debug)]
pub struct PreparedEpisode {
    pub orig: RawSubstrate,
    pub neg: RawSubstrate,
    pub passive: RawSubstrate,
    pub refut: RawSubstrate,
    pub shift: RawSubstrate,
    pub abs: RawSubstrate,
    pub cuts: [u8; 6],
}

#[derive(Clone, Debug)]
pub struct PreparedComposite {
    pub stream: RawSubstrate,
}

#[derive(Clone, Debug)]
pub struct PreparedMultiHop {
    pub stream: RawSubstrate,
}

#[derive(Clone, Debug)]
pub struct PhiReport {
    pub phi_sharp: f32,
    pub generativity: f32,
    pub discrimination: f32,
    pub projection: f32,
    pub integration: f32,
    pub gate_contrast: f32,
    pub mean_gate: f32,
    pub mip_loss: f32,
    pub mip_mask: u8,
    pub residual: f32,
    pub traj_gain: f32,
    pub bounce_pen: f32,
    pub refutation_stab: f32,
    pub displacement_stab: f32,
    pub abstraction_stab: f32,
    pub trace: AbductionTrace,
}

#[derive(Clone, Debug)]
pub struct ProjectionMatch {
    pub dst_idx: usize,
    pub score: f32,
    pub cross_err: f32,
    pub cos_sim: f32,
    pub core_start: usize,
    pub core_end: usize,
    pub referent: Option<(usize, usize)>,
    pub byte_overlap: f32,
}

#[derive(Clone, Debug)]
pub struct CompositePhi {
    pub score: f32,
    pub r_top: f32,
    pub r_rest: f32,
    pub selectivity: f32,
    pub top_pair: (usize, usize),
}

#[derive(Clone, Debug)]
pub struct WmSpan {
    pub start: usize,
    pub end: usize,
    pub peak_p: f32,
    pub mean_p: f32,
    pub mean_cos: f32,
    pub mean_norm: f32,
    pub mean_gate: f32,
    pub mean_back_echo: f32,
}

#[derive(Clone, Debug)]
pub struct WmReport {
    pub wm_phi: f32,
    pub head_cos: f32,
    pub head_cos_cut: f32,
    pub iit_echo_gain: f32,
    pub head_sel: f32,
    pub readout_sel: f32,
    pub readout_disc: f32,
    pub context_end: usize,
    pub molds_count: usize,
    pub target_causes: usize,
    pub target_spans: Vec<(usize, usize)>,
    pub mold_regimes: Vec<DiscoveredRegime>,
    pub top_spans: Vec<WmSpan>,
}

pub fn fnv_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100_0000_01B3);
    }
    hash
}

pub fn cosine_sim(a: &[f32; STATE_DIM], b: &[f32; STATE_DIM]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..STATE_DIM {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-8 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

fn sample_training_cuts(rng: &mut Rng) -> [u8; 6] {
    let mut sing = SINGLETON_MASKS;
    rng.shuffle(&mut sing);

    let mut balanced = [0u8; 25];
    let mut idx = 0;
    for m in 1u8..(1u8 << (NODES - 1)) {
        if !SINGLETON_MASKS.contains(&m) {
            balanced[idx] = m;
            idx += 1;
        }
    }
    rng.shuffle(&mut balanced);

    [
        sing[0],
        sing[1],
        sing[2],
        balanced[0],
        balanced[1],
        balanced[2],
    ]
}

impl PreparedEpisode {
    pub fn from_bytes(bytes: &[u8], rng: &mut Rng) -> Self {
        let orig = RawSubstrate::from_bytes(bytes);

        let mut neg_bytes = bytes.to_vec();
        rng.shuffle(&mut neg_bytes);
        let neg = RawSubstrate::from_bytes(&neg_bytes);

        let mut pool: [u8; 256] = std::array::from_fn(|i| i as u8);
        rng.shuffle(&mut pool);
        let mut passive_bytes = Vec::with_capacity(bytes.len());
        let mut cursor = 0usize;
        for _ in 0..bytes.len() {
            if cursor >= 6 && rng.next_usize(100) < 15 {
                passive_bytes.push(pool[rng.next_usize(cursor.min(14))]);
            } else {
                passive_bytes.push(pool[cursor % 256]);
                cursor += 1;
            }
        }
        let passive = RawSubstrate::from_bytes(&passive_bytes);

        let mut ref_bytes = bytes.to_vec();
        let corrupt_n = (bytes.len() / 6).max(1);
        for _ in 0..corrupt_n {
            let pos = rng.next_usize(ref_bytes.len());
            ref_bytes[pos] = ref_bytes[pos].wrapping_add(1 + rng.next_u8() % 250);
        }
        let refut = RawSubstrate::from_bytes(&ref_bytes);

        let mut shift_bytes = bytes.to_vec();
        let shift = (bytes.len() / 3).max(1) % bytes.len();
        shift_bytes.rotate_left(shift);
        let shift_sub = RawSubstrate::from_bytes(&shift_bytes);

        let mut perm: [u8; 256] = std::array::from_fn(|i| i as u8);
        rng.shuffle(&mut perm);
        let sub_len = ((bytes.len() * 3) / 4).max(8).min(bytes.len());
        let start_off = rng.next_usize(bytes.len() - sub_len + 1);
        let abs_bytes: Vec<u8> = bytes[start_off..start_off + sub_len]
            .iter()
            .map(|&b| perm[b as usize])
            .collect();
        let abs = RawSubstrate::from_bytes(&abs_bytes);

        let cuts = sample_training_cuts(rng);
        Self {
            orig,
            neg,
            passive,
            refut,
            shift: shift_sub,
            abs,
            cuts,
        }
    }
}

impl PreparedComposite {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            stream: RawSubstrate::from_bytes(bytes),
        }
    }
}

impl PreparedMultiHop {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            stream: RawSubstrate::from_bytes(bytes),
        }
    }
}

pub fn byte_overlap_ratio(a: &[u8], b: &[u8]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let mut seen = [false; 256];
    for &x in b {
        seen[x as usize] = true;
    }
    let shared = a.iter().filter(|&&x| seen[x as usize]).count();
    (shared as f32) / (a.len() as f32)
}

pub fn cut_discontinuity(sub: &RawSubstrate, s: usize, e: usize, p: usize) -> f32 {
    let l_lo = p.saturating_sub(10).max(s);
    let r_hi = (p + 10).min(e);
    if p <= l_lo || r_hi <= p {
        return 0.0;
    }

    let mut l_cross = 0usize;
    for t in l_lo..p {
        let mut has_cross = false;
        for idx in sub.offsets[t]..sub.offsets[t + 1] {
            let j = sub.edge_targets[idx] as usize;
            if j >= p && j < e {
                let f = &sub.edges[idx];
                if f[0] + f[1] + f[4] + f[5] >= 1.0 {
                    has_cross = true;
                    break;
                }
            }
        }
        if has_cross {
            l_cross += 1;
        }
    }

    let mut r_cross = 0usize;
    for t in p..r_hi {
        let mut has_cross = false;
        for idx in sub.offsets[t]..sub.offsets[t + 1] {
            let j = sub.edge_targets[idx] as usize;
            if j >= s && j < p {
                let f = &sub.edges[idx];
                if f[0] + f[1] + f[4] + f[5] >= 1.0 {
                    has_cross = true;
                    break;
                }
            }
        }
        if has_cross {
            r_cross += 1;
        }
    }

    let f_left = (l_cross as f32) / ((p - l_lo) as f32);
    let f_right = (r_cross as f32) / ((r_hi - p) as f32);
    (1.0 - 0.5 * (f_left + f_right)).clamp(0.0, 1.0)
}

pub fn abductive_discontinuity(
    graph: &PhiGraph,
    rels: &[[f32; REL_DIM]],
    left: (usize, usize),
    right: (usize, usize),
    steps: usize,
) -> f32 {
    let l_rels = &rels[left.0..left.1];
    let r_rels = &rels[right.0..right.1];
    let hl = graph.abduce_with_steps(l_rels, None, steps);
    let hr = graph.abduce_with_steps(r_rels, None, steps);

    let cos_lr = cosine_sim(&hl.final_state, &hr.final_state);
    let state_disc = 0.5 * (1.0 - cos_lr);

    let (_, _, e_lr) = graph.evaluate_hypothesis(&hl.final_state, r_rels);
    let (_, _, e_rl) = graph.evaluate_hypothesis(&hr.final_state, l_rels);
    let self_e = 0.5 * (hl.final_mean_err + hr.final_mean_err);
    let cross_e = 0.5 * (e_lr + e_rl);
    let err_disc = ((cross_e - self_e) / (cross_e + 0.08)).clamp(0.0, 1.0);

    0.55 * state_disc + 0.45 * err_disc
}

fn split_run_by_abduction(
    graph: &PhiGraph,
    sub: &RawSubstrate,
    rels: &[[f32; REL_DIM]],
    s: usize,
    e: usize,
    exact: bool,
) -> Vec<(usize, usize)> {
    let len = e.saturating_sub(s);
    if len < 32 {
        return vec![(s, e)];
    }

    let step = if exact { 4 } else { 6 };
    let win = if exact { 16 } else { 18 };
    let steps = if exact { SEARCH_STEPS } else { 4 };
    let offset = win / step;

    let mut anchors = Vec::new();
    let mut pos = s;
    while pos + win <= e {
        anchors.push(pos);
        pos += step;
    }
    if anchors.len() <= offset {
        return vec![(s, e)];
    }

    let traces: Vec<AbductionTrace> = anchors
        .iter()
        .map(|&a| graph.abduce_with_steps(&rels[a..a + win], None, steps))
        .collect();

    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for k in 0..(anchors.len() - offset) {
        let grid_p = anchors[k] + win;
        if grid_p < s + 12 || e.saturating_sub(grid_p) < 12 {
            continue;
        }
        let hl = &traces[k];
        let hr = &traces[k + offset];
        let l_rels = &rels[anchors[k]..anchors[k] + win];
        let r_rels = &rels[anchors[k + offset]..anchors[k + offset] + win];

        let state_disc = 0.5 * (1.0 - cosine_sim(&hl.final_state, &hr.final_state));
        let (_, _, e_lr) = graph.evaluate_hypothesis(&hl.final_state, r_rels);
        let (_, _, e_rl) = graph.evaluate_hypothesis(&hr.final_state, l_rels);
        let self_e = 0.5 * (hl.final_mean_err + hr.final_mean_err);
        let cross_e = 0.5 * (e_lr + e_rl);
        let err_disc = ((cross_e - self_e) / (cross_e + 0.08)).clamp(0.0, 1.0);
        let d_abd = 0.55 * state_disc + 0.45 * err_disc;

        let p_lo = grid_p.saturating_sub(step / 2).max(s + 12);
        let p_hi = (grid_p + step / 2).min(e.saturating_sub(12));
        let mut best_p = grid_p;
        let mut best_cut = -1.0f32;
        for cp in p_lo..=p_hi {
            let c = cut_discontinuity(sub, s, e, cp);
            if c > best_cut {
                best_cut = c;
                best_p = cp;
            }
        }

        let mut score = 0.45 * d_abd + 0.55 * best_cut;
        if best_cut >= 0.85 {
            score = score.max(0.75 * best_cut);
        }
        candidates.push((best_p, score));
    }

    let mut peaks: Vec<(usize, f32)> = Vec::new();
    for i in 0..candidates.len() {
        let (p, sc) = candidates[i];
        if sc <= 0.52 {
            continue;
        }
        let left_ok = i == 0 || sc >= candidates[i - 1].1;
        let right_ok = i + 1 == candidates.len() || sc >= candidates[i + 1].1;
        if left_ok && right_ok {
            peaks.push((p, sc));
        }
    }
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut chosen: Vec<usize> = Vec::new();
    for (p, _) in peaks {
        if chosen.iter().all(|&c| c.abs_diff(p) >= 16) {
            chosen.push(p);
        }
    }
    if chosen.is_empty() {
        return vec![(s, e)];
    }
    chosen.sort_unstable();

    if exact {
        for p in &mut chosen {
            let mut best_p = *p;
            let mut best_sc = f32::NEG_INFINITY;
            let lo = (*p).saturating_sub(3).max(s + 12);
            let hi = (*p + 3).min(e.saturating_sub(12));
            for cand in lo..=hi {
                let l_start = cand.saturating_sub(16).max(s);
                let r_end = (cand + 16).min(e);
                let d_abd = abductive_discontinuity(
                    graph,
                    rels,
                    (l_start, cand),
                    (cand, r_end),
                    SEARCH_STEPS,
                );
                let d_cut = cut_discontinuity(sub, s, e, cand);
                let sc = 0.35 * d_abd + 0.65 * d_cut;
                if sc > best_sc {
                    best_sc = sc;
                    best_p = cand;
                }
            }
            *p = best_p;
        }
    }

    let mut segs = Vec::with_capacity(chosen.len() + 1);
    let mut cur = s;
    for p in chosen {
        if p >= cur + 12 && e >= p + 12 {
            segs.push((cur, p));
            cur = p;
        }
    }
    segs.push((cur, e));
    segs
}

fn attach_passive_referents(
    cores: &[(usize, usize)],
    gates_ctx: &[f32],
    n: usize,
) -> Vec<DiscoveredRegime> {
    let k = cores.len();
    if k == 0 {
        return Vec::new();
    }
    let lead_gap = cores[0].0;
    let trail_gap = n.saturating_sub(cores[k - 1].1);
    let suffix_mode = trail_gap >= 3 && lead_gap < 3;

    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let (cs, ce) = cores[i];
        let mg = gates_ctx[cs..ce].iter().sum::<f32>() / ((ce - cs).max(1) as f32);

        let prev_end = if i == 0 { 0 } else { cores[i - 1].1 };
        let next_start = if i + 1 == k { n } else { cores[i + 1].0 };

        let referent = if suffix_mode {
            if next_start.saturating_sub(ce) >= 3 {
                Some((ce, next_start))
            } else {
                None
            }
        } else if cs.saturating_sub(prev_end) >= 3 {
            Some((prev_end, cs))
        } else {
            None
        };

        out.push(DiscoveredRegime {
            core_start: cs,
            core_end: ce,
            referent,
            mean_gate: mg,
        });
    }
    out
}

fn tighten_regime_isolated(
    graph: &PhiGraph,
    bytes: &[u8],
    mut cs: usize,
    mut ce: usize,
) -> (usize, usize) {
    for _ in 0..2 {
        if ce.saturating_sub(cs) < 12 {
            break;
        }
        let iso_sub = RawSubstrate::from_bytes(&bytes[cs..ce]);
        let (_, _, iso_pt) = graph.encode_with_gate(&iso_sub);
        let len = iso_pt.len();

        let mut l = 0usize;
        while l + 10 < len && iso_pt[l] < 0.44 {
            l += 1;
        }
        let mut r = len;
        while l + 10 < r && iso_pt[r - 1] < 0.44 {
            r -= 1;
        }
        if l == 0 && r == len {
            break;
        }
        cs += l;
        ce = cs + (r - l);
    }
    (cs, ce)
}

fn has_supported_bridge(sub: &RawSubstrate, left: (usize, usize), right: (usize, usize)) -> bool {
    let mut bridges = 0usize;
    for t in left.0..left.1 {
        for idx in sub.offsets[t]..sub.offsets[t + 1] {
            let j = sub.edge_targets[idx] as usize;
            if j >= right.0 && j < right.1 {
                let f = &sub.edges[idx];
                if f[0] + f[1] + f[4] + f[5] >= 1.0 {
                    bridges += 1;
                    if bridges >= 2 {
                        return true;
                    }
                    break;
                }
            }
        }
    }
    false
}

pub fn discover_causal_regimes_from_encoded(
    graph: &PhiGraph,
    sub: &RawSubstrate,
    rels: &[[f32; REL_DIM]],
    gates_ctx: &[f32],
    gates_pt: &[f32],
    exact: bool,
) -> Vec<DiscoveredRegime> {
    let n = rels.len();
    if n < 12 {
        let mg = if n == 0 {
            0.0
        } else {
            gates_ctx.iter().sum::<f32>() / (n as f32)
        };
        return vec![DiscoveredRegime {
            core_start: 0,
            core_end: n,
            referent: None,
            mean_gate: mg,
        }];
    }

    let mut raw_runs: Vec<(usize, usize)> = Vec::new();
    let mut in_run = false;
    let mut run_start = 0usize;
    for t in 0..n {
        if gates_ctx[t] >= 0.46 {
            if !in_run {
                in_run = true;
                run_start = t;
            }
        } else if in_run {
            raw_runs.push((run_start, t));
            in_run = false;
        }
    }
    if in_run {
        raw_runs.push((run_start, n));
    }

    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in raw_runs {
        if let Some(last) = merged.last_mut() {
            let gap = s.saturating_sub(last.1);
            let both_solid = last.1.saturating_sub(last.0) >= 8 && e.saturating_sub(s) >= 8;
            let bridged = gap <= 6
                && last.1.saturating_sub(last.0) >= 6
                && e.saturating_sub(s) >= 6
                && has_supported_bridge(sub, *last, (s, e));
            if gap <= 1 || (gap <= 3 && both_solid) || bridged {
                last.1 = e;
                continue;
            }
        }
        merged.push((s, e));
    }

    let mut active_runs: Vec<(usize, usize)> = merged
        .into_iter()
        .filter(|&(s, e)| e.saturating_sub(s) >= 10)
        .collect();

    if active_runs.is_empty() {
        let mg = gates_ctx.iter().sum::<f32>() / (n as f32);
        return vec![DiscoveredRegime {
            core_start: 0,
            core_end: n,
            referent: None,
            mean_gate: mg,
        }];
    }

    for i in 0..active_runs.len() {
        let min_s = if i == 0 { 0 } else { active_runs[i - 1].1 };
        let max_e = if i + 1 == active_runs.len() {
            n
        } else {
            active_runs[i + 1].0
        };
        let (mut s, mut e) = active_runs[i];

        let mut trim_l = 0usize;
        while s + 10 < e && trim_l < 14 && gates_pt[s] < 0.44 {
            s += 1;
            trim_l += 1;
        }
        let mut trim_r = 0usize;
        while s + 10 < e && trim_r < 14 && gates_pt[e - 1] < 0.44 {
            e -= 1;
            trim_r += 1;
        }

        if trim_l == 0 {
            let mut ext_l = 0usize;
            while s > min_s && ext_l < 2 && gates_pt[s - 1] >= 0.68 {
                s -= 1;
                ext_l += 1;
            }
        }
        if trim_r == 0 {
            let mut ext_r = 0usize;
            while e < max_e && ext_r < 2 && gates_pt[e] >= 0.68 {
                e += 1;
                ext_r += 1;
            }
        }

        active_runs[i] = (s, e);
    }

    let mut cores: Vec<(usize, usize)> = Vec::new();
    for (s, e) in active_runs {
        for (mut cs, mut ce) in split_run_by_abduction(graph, sub, rels, s, e, exact) {
            let mut tl = 0usize;
            while cs + 10 < ce && tl < 4 && gates_pt[cs] < 0.40 {
                cs += 1;
                tl += 1;
            }
            let mut tr = 0usize;
            while cs + 10 < ce && tr < 4 && gates_pt[ce - 1] < 0.40 {
                ce -= 1;
                tr += 1;
            }
            cores.push((cs, ce));
        }
    }

    let max_mg = cores
        .iter()
        .map(|&(cs, ce)| gates_ctx[cs..ce].iter().sum::<f32>() / ((ce - cs).max(1) as f32))
        .fold(0.0f32, f32::max);
    if max_mg >= 0.72 && cores.len() >= 2 {
        let filtered: Vec<(usize, usize)> = cores
            .iter()
            .copied()
            .filter(|&(cs, ce)| {
                let mg = gates_ctx[cs..ce].iter().sum::<f32>() / ((ce - cs).max(1) as f32);
                let len = ce.saturating_sub(cs);
                mg >= 0.62 * max_mg && (len >= 14 || mg >= 0.68)
            })
            .collect();
        if !filtered.is_empty() {
            cores = filtered;
        }
    }

    attach_passive_referents(&cores, gates_ctx, n)
}

pub fn discover_causal_regimes(
    graph: &PhiGraph,
    bytes: &[u8],
    exact: bool,
) -> Vec<DiscoveredRegime> {
    let sub = RawSubstrate::from_bytes(bytes);
    let (rels, gates_ctx, gates_pt) = graph.encode_with_gate(&sub);
    let raw =
        discover_causal_regimes_from_encoded(graph, &sub, &rels, &gates_ctx, &gates_pt, exact);
    if !exact || raw.is_empty() {
        return raw;
    }

    let mut tightened: Vec<(usize, usize)> = Vec::with_capacity(raw.len());
    for reg in raw {
        let (cs, ce) = tighten_regime_isolated(graph, bytes, reg.core_start, reg.core_end);
        if ce.saturating_sub(cs) >= 10 {
            tightened.push((cs, ce));
        }
    }
    if tightened.is_empty() {
        return vec![DiscoveredRegime {
            core_start: 0,
            core_end: bytes.len(),
            referent: None,
            mean_gate: gates_ctx.iter().sum::<f32>() / (bytes.len().max(1) as f32),
        }];
    }
    attach_passive_referents(&tightened, &gates_ctx, bytes.len())
}

pub fn compute_phi_sharp_prepared(
    graph: &PhiGraph,
    prep: &PreparedEpisode,
    exact_mip: bool,
) -> PhiReport {
    let (rels, gates_orig_ctx, gates_orig_pt) = graph.encode_with_gate(&prep.orig);
    let trace = graph.abduce(&rels, None);
    let residual = trace.final_mean_err;

    let r0 = trace.step_residuals[0].max(1e-4);
    let traj_gain = ((r0 - residual) / r0).clamp(0.0, 1.0);
    let mut bounce_pen = 0.0f32;
    for pair in trace.step_residuals.windows(2) {
        if pair[1] > pair[0] {
            bounce_pen += pair[1] - pair[0];
        }
    }

    let generativity = (-2.2 * residual).exp().clamp(1e-4, 1.0);

    let (neg_rels, gates_neg_ctx, gates_neg_pt) = graph.encode_with_gate(&prep.neg);
    let (_, gates_pass_ctx, gates_pass_pt) = graph.encode_with_gate(&prep.passive);
    let (_, _, cross_err) = graph.evaluate_hypothesis(&trace.final_state, &neg_rels);
    let neg_trace = graph.abduce(&neg_rels, None);

    let err_gap = ((cross_err - residual) / (cross_err + 0.05)).clamp(0.0, 1.0);
    let state_gap = (1.0 - cosine_sim(&trace.final_state, &neg_trace.final_state)).clamp(0.0, 1.0);
    let discrimination = (0.65 * err_gap + 0.35 * state_gap).clamp(1e-4, 1.0);

    let inv_orig = 1.0 / (gates_orig_ctx.len().max(1) as f32);
    let inv_neg = 1.0 / (gates_neg_ctx.len().max(1) as f32);
    let inv_pass = 1.0 / (gates_pass_ctx.len().max(1) as f32);

    let mean_g_orig = 0.5
        * (gates_orig_ctx.iter().sum::<f32>() * inv_orig
            + gates_orig_pt.iter().sum::<f32>() * inv_orig);
    let mean_g_neg = 0.5
        * (gates_neg_ctx.iter().sum::<f32>() * inv_neg
            + gates_neg_pt.iter().sum::<f32>() * inv_neg);
    let mean_g_pass = 0.5
        * (gates_pass_ctx.iter().sum::<f32>() * inv_pass
            + gates_pass_pt.iter().sum::<f32>() * inv_pass);

    let worst_bad_g = mean_g_neg.max(mean_g_pass);
    let gate_contrast = (mean_g_orig * (1.0 - worst_bad_g)).clamp(0.0, 1.0);

    let ref_rels = graph.encode_relations(&prep.refut);
    let ref_trace = graph.abduce(&ref_rels, None);
    let (_, _, ref_on_orig) = graph.evaluate_hypothesis(&ref_trace.final_state, &rels);
    let ref_cos = 0.5 * (1.0 + cosine_sim(&trace.final_state, &ref_trace.final_state));
    let refutation_stab = (ref_cos * (-1.5 * ref_on_orig).exp()).clamp(0.0, 1.0);

    let shift_rels = graph.encode_relations(&prep.shift);
    let shift_trace = graph.abduce(&shift_rels, None);
    let displacement_stab =
        (0.5 * (1.0 + cosine_sim(&trace.final_state, &shift_trace.final_state))).clamp(0.0, 1.0);

    let abs_rels = graph.encode_relations(&prep.abs);
    let abs_trace = graph.abduce(&abs_rels, None);
    let abstraction_stab =
        (0.5 * (1.0 + cosine_sim(&trace.final_state, &abs_trace.final_state))).clamp(0.0, 1.0);

    let projection =
        ((refutation_stab + displacement_stab + abstraction_stab) / 3.0).clamp(1e-4, 1.0);

    let masks: Vec<u8> = if exact_mip {
        (1u8..(1u8 << (NODES - 1))).collect()
    } else {
        prep.cuts.to_vec()
    };

    let mut mip_loss = f32::INFINITY;
    let mut mip_mask = masks[0];
    for mask in masks {
        let cut_trace = graph.abduce(&rels, Some(mask));
        let err_drop = (cut_trace.final_mean_err - residual).max(0.0);
        let state_drop = (1.0 - cosine_sim(&trace.final_state, &cut_trace.final_state)).max(0.0);
        let loss = err_drop + 0.25 * state_drop;
        if loss < mip_loss {
            mip_loss = loss;
            mip_mask = mask;
        }
    }
    let integration = (1.0 - (-3.5 * mip_loss).exp()).clamp(0.0, 1.0);

    let conj = (generativity * discrimination * projection * (0.2 + 0.8 * integration)).powf(0.25);
    let phi_sharp =
        0.85 * conj + 0.10 * traj_gain + 0.15 * gate_contrast - 0.30 * residual - 0.30 * bounce_pen;

    PhiReport {
        phi_sharp,
        generativity,
        discrimination,
        projection,
        integration,
        gate_contrast,
        mean_gate: mean_g_orig,
        mip_loss,
        mip_mask,
        residual,
        traj_gain,
        bounce_pen,
        refutation_stab,
        displacement_stab,
        abstraction_stab,
        trace,
    }
}

pub fn compute_phi_sharp(
    graph: &PhiGraph,
    bytes: &[u8],
    rng: &mut Rng,
    exact_mip: bool,
) -> PhiReport {
    let prep = PreparedEpisode::from_bytes(bytes, rng);
    compute_phi_sharp_prepared(graph, &prep, exact_mip)
}

pub fn compute_composite_phi_prepared(
    graph: &PhiGraph,
    prep: &PreparedComposite,
) -> CompositePhi {
    let (rels, gates_ctx, gates_pt) = graph.encode_with_gate(&prep.stream);
    let regimes = discover_causal_regimes_from_encoded(
        graph,
        &prep.stream,
        &rels,
        &gates_ctx,
        &gates_pt,
        false,
    );
    let m = regimes.len();
    if m < 2 {
        return CompositePhi {
            score: -0.25,
            r_top: 0.0,
            r_rest: 0.0,
            selectivity: 0.0,
            top_pair: (0, 0),
        };
    }

    let mut traces = Vec::with_capacity(m);
    let mut sum_res = 0.0f32;
    for reg in &regimes {
        let tr = graph.abduce(&rels[reg.core_start..reg.core_end], None);
        sum_res += tr.final_mean_err;
        traces.push(tr);
    }
    let mean_residual = sum_res / (m as f32);

    let mut pairs: Vec<(f32, usize, usize)> = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        let ri = &rels[regimes[i].core_start..regimes[i].core_end];
        for j in (i + 1)..m {
            let rj = &rels[regimes[j].core_start..regimes[j].core_end];
            let (_, _, e_ij) = graph.evaluate_hypothesis(&traces[i].final_state, rj);
            let (_, _, e_ji) = graph.evaluate_hypothesis(&traces[j].final_state, ri);
            let cross_gen = (-2.2 * 0.5 * (e_ij + e_ji)).exp().clamp(0.0, 1.0);
            let cos_res = 0.5 * (1.0 + cosine_sim(&traces[i].final_state, &traces[j].final_state));
            let r_ij = (cross_gen * cos_res).sqrt();
            pairs.push((r_ij, i, j));
        }
    }

    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (r_top, top_i, top_j) = pairs[0];
    let r_rest = if pairs.len() > 1 {
        pairs[1..].iter().map(|p| p.0).sum::<f32>() / ((pairs.len() - 1) as f32)
    } else {
        0.0
    };
    let selectivity = r_top * (1.0 - r_rest);
    let score = selectivity - 0.18 * mean_residual;

    CompositePhi {
        score,
        r_top,
        r_rest,
        selectivity,
        top_pair: (top_i, top_j),
    }
}

pub fn compute_composite_phi(graph: &PhiGraph, bytes: &[u8]) -> CompositePhi {
    let prep = PreparedComposite::from_bytes(bytes);
    compute_composite_phi_prepared(graph, &prep)
}

pub fn project_query_to_regimes(
    graph: &PhiGraph,
    bytes: &[u8],
    regimes: &[DiscoveredRegime],
    query_idx: usize,
) -> Vec<ProjectionMatch> {
    let q_reg = regimes[query_idx];
    let q_bytes = &bytes[q_reg.core_start..q_reg.core_end];
    let q_rels = graph.encode_bytes(q_bytes);
    let q_trace = graph.abduce(&q_rels, None);

    let mut matches = Vec::with_capacity(regimes.len().saturating_sub(1));
    for (idx, reg) in regimes.iter().enumerate() {
        if idx == query_idx {
            continue;
        }
        let core_bytes = &bytes[reg.core_start..reg.core_end];
        let core_rels = graph.encode_bytes(core_bytes);
        let core_trace = graph.abduce(&core_rels, None);

        let (_, _, fwd_err) = graph.evaluate_hypothesis(&q_trace.final_state, &core_rels);
        let (_, _, bwd_err) = graph.evaluate_hypothesis(&core_trace.final_state, &q_rels);
        let cross_err = 0.5 * (fwd_err + bwd_err);

        let cross_gen = (-2.2 * cross_err).exp().clamp(0.0, 1.0);
        let cos_sim = cosine_sim(&q_trace.final_state, &core_trace.final_state);
        let cos_pos = 0.5 * (1.0 + cos_sim);
        let score = (cross_gen * cos_pos).sqrt();

        let byte_overlap = byte_overlap_ratio(q_bytes, core_bytes);
        matches.push(ProjectionMatch {
            dst_idx: idx,
            score,
            cross_err,
            cos_sim,
            core_start: reg.core_start,
            core_end: reg.core_end,
            referent: reg.referent,
            byte_overlap,
        });
    }

    matches.sort_by(|a, b| b.score.total_cmp(&a.score));
    matches
}

// =============================================================================
// MÉMOIRE DE TRAVAIL CONNEXIONNISTE : φ_wm NON-SUPERVISÉ & LECTURE CONTINUE P(t)
// =============================================================================

/// Fraction des octets d'un régime qui échoent vers une zone passive antérieure (méta-langage).
/// 100% relationnel : utilise uniquement `echo_edges` de M(a,b) et `gates_pt` appris.
fn passive_back_echo_ratio(sub: &RawSubstrate, gates_pt: &[f32], cs: usize, ce: usize) -> f32 {
    if ce <= cs {
        return 0.0;
    }
    let mut hits = 0usize;
    for t in cs..ce {
        let slice = &sub.echo_edges[sub.echo_offsets[t]..sub.echo_offsets[t + 1]];
        if slice
            .iter()
            .any(|e| (e.src as usize) < cs && gates_pt[e.src as usize] < 0.52)
        {
            hits += 1;
        }
    }
    (hits as f32) / ((ce - cs) as f32)
}

fn best_window_mean(vals: &[f32], mask: &[bool], len: usize, win: usize) -> (f32, usize) {
    if len == 0 {
        return (0.0, 0);
    }
    let w = win.min(len).max(1);
    let mut best_val = -1.0f32;
    let mut best_center = 0usize;

    for s in 0..=(len - w) {
        let mut sum = 0.0f32;
        let mut cnt = 0usize;
        for j in s..(s + w) {
            if mask[j] {
                sum += vals[j];
                cnt += 1;
            }
        }
        if cnt * 5 >= w * 3 {
            let avg = sum / (cnt as f32);
            if avg > best_val {
                best_val = avg;
                best_center = s + w / 2;
            }
        }
    }

    if best_val < 0.0 {
        // Repli sur la meilleure position valide si aucune fenêtre contiguë n'atteint la taille `w`
        for j in 0..len {
            if mask[j] && vals[j] > best_val {
                best_val = vals[j];
                best_center = j;
            }
        }
    }
    (best_val.max(0.0), best_center)
}

fn extract_top_wm_spans(
    sub: &RawSubstrate,
    gates_pt: &[f32],
    z: &[[f32; WM_DIM]],
    z_query: &[f32; WM_DIM],
    p_field: &[f32],
    context_end: usize,
    max_spans: usize,
) -> Vec<WmSpan> {
    if context_end == 0 {
        return Vec::new();
    }
    let mut smooth = vec![0.0f32; context_end];
    for t in 0..context_end {
        let lo = t.saturating_sub(1);
        let hi = (t + 1).min(context_end - 1);
        let sum: f32 = p_field[lo..=hi].iter().sum();
        smooth[t] = sum / ((hi - lo + 1) as f32);
    }

    let mut used = vec![false; context_end];
    let mut spans = Vec::with_capacity(max_spans);

    for _ in 0..max_spans {
        let mut best_t = None;
        let mut best_s = 0.05f32;
        for t in 0..context_end {
            if !used[t] && smooth[t] > best_s {
                best_s = smooth[t];
                best_t = Some(t);
            }
        }
        let Some(pt) = best_t else {
            break;
        };

        let peak_p = p_field[pt].max(best_s);
        let thresh = (peak_p * 0.55).max(0.18);

        let mut s = pt;
        while s > 0 && !used[s - 1] && p_field[s - 1] >= thresh && pt.saturating_sub(s) < 16 {
            s -= 1;
        }
        let mut e = pt + 1;
        while e < context_end && !used[e] && p_field[e] >= thresh && e.saturating_sub(pt) < 16 {
            e += 1;
        }

        let len = (e - s).max(1) as f32;
        let mut sum_p = 0.0f32;
        let mut sum_cos = 0.0f32;
        let mut sum_norm = 0.0f32;
        let mut sum_gate = 0.0f32;
        let mut sum_be = 0.0f32;
        let mut max_p = 0.0f32;
        for t in s..e {
            if p_field[t] > max_p {
                max_p = p_field[t];
            }
            sum_p += p_field[t];
            sum_cos += wm_cosine(&z[t], z_query);
            sum_norm += wm_norm(&z[t]);
            sum_gate += gates_pt[t];
            sum_be += sub.back_echo_len[t];
        }

        spans.push(WmSpan {
            start: s,
            end: e,
            peak_p: max_p,
            mean_p: sum_p / len,
            mean_cos: sum_cos / len,
            mean_norm: sum_norm / len,
            mean_gate: sum_gate / len,
            mean_back_echo: sum_be / len,
        });

        let pad_s = s.saturating_sub(4);
        let pad_e = (e + 4).min(context_end);
        for u in &mut used[pad_s..pad_e] {
            *u = true;
        }
    }

    spans
}

fn run_wm_core(
    graph: &PhiGraph,
    sub: &RawSubstrate,
    raw_bytes: Option<&[u8]>,
    exact: bool,
) -> Option<WmReport> {
    let n = sub.len;
    if n < 24 {
        return None;
    }

    let (rels, gates_ctx, gates_pt) = graph.encode_with_gate(sub);
    let mut regimes =
        discover_causal_regimes_from_encoded(graph, sub, &rels, &gates_ctx, &gates_pt, exact);

    if exact && let Some(bytes) = raw_bytes {
        let mut tightened = Vec::with_capacity(regimes.len());
        for reg in &regimes {
            let (cs, ce) = tighten_regime_isolated(graph, bytes, reg.core_start, reg.core_end);
            if ce.saturating_sub(cs) >= 10 {
                tightened.push((cs, ce));
            }
        }
        if tightened.len() >= 2 {
            regimes = attach_passive_referents(&tightened, &gates_ctx, n);
        }
    }

    if regimes.len() < 2 {
        return None;
    }

    // 1. Démotion purement relationnelle des blocs méta-linguistiques (citant des zones passives antérieures)
    let last_idx = regimes.len() - 1;
    let mut non_meta: Vec<DiscoveredRegime> = Vec::with_capacity(regimes.len());
    for (idx, &reg) in regimes.iter().enumerate() {
        if idx < last_idx && passive_back_echo_ratio(sub, &gates_pt, reg.core_start, reg.core_end) >= 0.25 {
            continue;
        }
        non_meta.push(reg);
    }
    if non_meta.len() < 2 {
        non_meta = regimes;
    }

    // Abduit les régimes retenus une seule fois
    let traces: Vec<AbductionTrace> = non_meta
        .iter()
        .map(|reg| graph.abduce(&rels[reg.core_start..reg.core_end], None))
        .collect();

    // 2. Regroupe les régimes finaux appartenant à une cible multi-causes (sans utiliser de labels)
    let k = non_meta.len();
    let mut split_t = k - 1;
    while split_t >= 3 && (k - split_t) < 4 {
        let cand = split_t - 1;
        let gap_next = non_meta[split_t]
            .core_start
            .saturating_sub(non_meta[cand].core_end);
        if gap_next > 8 {
            break;
        }
        let gap_prev = non_meta[cand]
            .core_start
            .saturating_sub(non_meta[cand - 1].core_end);
        if gap_prev > 8 && gap_prev < 20 {
            break;
        }
        let max_cos = (0..cand)
            .map(|i| cosine_sim(&traces[cand].final_state, &traces[i].final_state))
            .fold(-1.0f32, f32::max);
        if max_cos >= 0.45 {
            split_t = cand;
        } else {
            break;
        }
    }

    let mold_regimes = non_meta[0..split_t].to_vec();
    let target_regs = &non_meta[split_t..];
    let target_spans: Vec<(usize, usize)> = target_regs
        .iter()
        .map(|r| (r.core_start, r.core_end))
        .collect();
    let context_end = target_regs[0].core_start;
    if context_end < 12 {
        return None;
    }

    // 3. Initialisation de la Mémoire de Travail Z^(0) sur les moules primitifs (0..context_end)
    let mut init_z = vec![[0.0f32; WM_DIM]; n];
    for (idx, reg) in mold_regimes.iter().enumerate() {
        let u = graph.project_cause_to_wm(&traces[idx].final_state);
        for t in reg.core_start..reg.core_end.min(context_end) {
            let w_g = gates_pt[t].max(0.55);
            for d in 0..WM_DIM {
                init_z[t][d] = w_g * u[d];
            }
        }
    }

    // 4. Agrégation continue de la cible Q en vecteur mémoire `z_query` par nouveauté orthogonale
    //    (gère 1, 2 ou 3 causes sans heuristique de coupe discrète)
    let mut win_cands: Vec<(f32, [f32; WM_DIM])> = Vec::new();
    for (offset, reg) in target_regs.iter().enumerate() {
        let len = reg.core_end.saturating_sub(reg.core_start);
        if len < 34 {
            let tr = &traces[split_t + offset];
            win_cands.push((tr.final_mean_err, graph.project_cause_to_wm(&tr.final_state)));
        } else {
            let w = 18usize;
            let stride = 8usize;
            let mut s = reg.core_start;
            while s + w <= reg.core_end {
                let tr = graph.abduce(&rels[s..s + w], None);
                win_cands.push((tr.final_mean_err, graph.project_cause_to_wm(&tr.final_state)));
                s += stride;
            }
            let tail_s = reg.core_end - w;
            if tail_s > reg.core_start {
                let tr = graph.abduce(&rels[tail_s..reg.core_end], None);
                win_cands.push((tr.final_mean_err, graph.project_cause_to_wm(&tr.final_state)));
            }
        }
    }
    win_cands.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut z_query = [0.0f32; WM_DIM];
    let mut target_causes = 0usize;
    if let Some(&(best_err, first_u)) = win_cands.first() {
        z_query = first_u;
        target_causes = 1;
        let err_cap = (best_err + 0.14).max(0.24);
        for &(err, ref u) in &win_cands[1..] {
            if err > err_cap || target_causes >= 3 {
                continue;
            }
            let nov = orth_novelty(u, &z_query);
            if wm_norm(&nov) >= 0.62 {
                for d in 0..WM_DIM {
                    z_query[d] += nov[d];
                }
                target_causes += 1;
            }
        }
    }

    // 5. Propagation complète (full) vs coupure IIT des arêtes d'écho longue portée (cut)
    let z_full = graph.propagate_working_memory(sub, &gates_pt, &init_z, false);
    let z_cut = graph.propagate_working_memory(sub, &gates_pt, &init_z, true);

    // 6. Évaluation 100% non-supervisée sur les têtes passives non-écho (0..context_end)
    let mut head_mask = vec![false; context_end];
    let mut head_count = 0usize;
    for t in 0..context_end {
        if gates_pt[t] < 0.48 && sub.back_echo_len[t] < 0.05 {
            head_mask[t] = true;
            head_count += 1;
        }
    }
    if head_count < 4 {
        for m in &mut head_mask {
            *m = true;
        }
    }

    let mut cos_full = vec![0.0f32; context_end];
    let mut cos_cut = vec![0.0f32; context_end];
    for t in 0..context_end {
        cos_full[t] = wm_cosine(&z_full[t], &z_query).max(0.0);
        cos_cut[t] = wm_cosine(&z_cut[t], &z_query).max(0.0);
    }

    let (head_cos, peak_center) = best_window_mean(&cos_full, &head_mask, context_end, 5);
    let (head_cos_cut, _) = best_window_mean(&cos_cut, &head_mask, context_end, 5);
    let iit_echo_gain = (head_cos - head_cos_cut).max(0.0);

    let mut bg_sum = 0.0f32;
    let mut bg_cnt = 0usize;
    for t in 0..context_end {
        if head_mask[t] && t.abs_diff(peak_center) > 10 {
            bg_sum += cos_full[t];
            bg_cnt += 1;
        }
    }
    let bg_head_cos = if bg_cnt == 0 {
        0.0
    } else {
        bg_sum / (bg_cnt as f32)
    };
    let head_sel = head_cos * (1.0 - bg_head_cos);

    // 7. Tête de lecture connexionniste P(t) + contrefactuel (-z_query)
    let p_field = graph.readout_explanation(sub, &gates_pt, &z_full, &z_query);
    let mut neg_q = [0.0f32; WM_DIM];
    for d in 0..WM_DIM {
        neg_q[d] = -z_query[d];
    }
    let p_neg = graph.readout_explanation(sub, &gates_pt, &z_full, &neg_q);

    let all_mask = vec![true; context_end];
    let (top_p, p_center) = best_window_mean(&p_field[..context_end], &all_mask, context_end, 5);
    let (top_p_neg, _) = best_window_mean(&p_neg[..context_end], &all_mask, context_end, 5);

    let mut bg_p_sum = 0.0f32;
    let mut bg_p_cnt = 0usize;
    for t in 0..context_end {
        if t.abs_diff(p_center) > 10 {
            bg_p_sum += p_field[t];
            bg_p_cnt += 1;
        }
    }
    let bg_p = if bg_p_cnt == 0 {
        0.0
    } else {
        bg_p_sum / (bg_p_cnt as f32)
    };
    let readout_sel = top_p * (1.0 - bg_p);
    let readout_disc = (top_p - top_p_neg).max(0.0);

    // Aligne le pic de P(t) sur un vrai vecteur mémoire passif non-écho (0 label)
    let al_lo = p_center.saturating_sub(2);
    let al_hi = (p_center + 3).min(context_end);
    let mut al_sum = 0.0f32;
    for t in al_lo..al_hi {
        let passive_head = if head_count >= 4 {
            (1.0 - gates_pt[t]) * (1.0 - sub.back_echo_len[t])
        } else {
            1.0
        };
        al_sum += cos_full[t] * passive_head;
    }
    let readout_align = al_sum / ((al_hi - al_lo).max(1) as f32);

    let iit_norm = (iit_echo_gain / 0.28).clamp(0.0, 1.0);
    let wm_phi = 0.30 * head_cos
        + 0.25 * head_sel
        + 0.35 * iit_norm
        + 0.20 * readout_sel
        + 0.15 * readout_disc
        + 0.25 * readout_align;

    let top_spans = if exact {
        extract_top_wm_spans(
            sub,
            &gates_pt,
            &z_full,
            &z_query,
            &p_field,
            context_end,
            5,
        )
    } else {
        Vec::new()
    };

    Some(WmReport {
        wm_phi,
        head_cos,
        head_cos_cut,
        iit_echo_gain,
        head_sel,
        readout_sel,
        readout_disc,
        context_end,
        molds_count: mold_regimes.len(),
        target_causes,
        target_spans,
        mold_regimes,
        top_spans,
    })
}

pub fn compute_wm_phi_prepared(graph: &PhiGraph, prep: &PreparedMultiHop) -> f32 {
    run_wm_core(graph, &prep.stream, None, false)
        .map(|r| r.wm_phi)
        .unwrap_or(-0.20)
}

pub fn run_working_memory(graph: &PhiGraph, bytes: &[u8]) -> Option<WmReport> {
    let sub = RawSubstrate::from_bytes(bytes);
    run_wm_core(graph, &sub, Some(bytes), true)
}

pub fn extract_abductive_core(
    graph: &PhiGraph,
    bytes: &[u8],
    rels: &[[f32; REL_DIM]],
    global_state: &[f32; STATE_DIM],
) -> (usize, Vec<u8>, f32) {
    let n = bytes.len();
    let win = (n / 4).clamp(10, 24).min(n);
    if n <= win {
        let (_, _, err) = graph.evaluate_hypothesis(global_state, rels);
        return (0, bytes.to_vec(), err);
    }
    let stride = (win / 4).max(1);
    let mut best_start = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    let mut best_err = f32::INFINITY;

    let mut s = 0usize;
    while s + win <= n {
        let sub_rels = graph.encode_bytes(&bytes[s..s + win]);
        let sub_trace = graph.abduce(&sub_rels, None);
        let (_, _, err_on_all) = graph.evaluate_hypothesis(&sub_trace.final_state, rels);
        let sim = 0.5 * (1.0 + cosine_sim(&sub_trace.final_state, global_state));
        let score = (-2.2 * err_on_all).exp() * sim;
        if score > best_score {
            best_score = score;
            best_start = s;
            best_err = err_on_all;
        }
        s += stride;
    }
    (
        best_start,
        bytes[best_start..best_start + win].to_vec(),
        best_err,
    )
}

pub fn format_partition(mask: u8) -> String {
    let mut a = Vec::new();
    let mut b = Vec::new();
    for i in 0..NODES {
        if ((mask >> i) & 1) == 0 {
            a.push(i.to_string());
        } else {
            b.push(i.to_string());
        }
    }
    format!("{{{}}} | {{{}}}", a.join(","), b.join(","))
}
