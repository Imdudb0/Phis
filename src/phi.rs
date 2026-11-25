use crate::graph::{AbductionTrace, NODES, PhiGraph, REL_DIM, RawSubstrate, SEARCH_STEPS, STATE_DIM};
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
    pub refut: RawSubstrate,
    pub shift: RawSubstrate,
    pub abs: RawSubstrate,
    pub cuts: [u8; 6],
}

/// Flux continu brut : aucun découpage préalable sur `\n` ou autre délimiteur.
#[derive(Clone, Debug)]
pub struct PreparedComposite {
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
    pub mean_residual: f32,
    pub regimes_found: usize,
    pub top_pair: (usize, usize),
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

fn byte_overlap_ratio(a: &[u8], b: &[u8]) -> f32 {
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

fn abductive_discontinuity(
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

/// Scinde un îlot actif continu aux transitions de régime causal par discontinuité abductive D(t),
/// sans aucun séparateur codé en dur.
fn split_run_by_abduction(
    graph: &PhiGraph,
    rels: &[[f32; REL_DIM]],
    s: usize,
    e: usize,
    exact: bool,
) -> Vec<(usize, usize)> {
    let len = e.saturating_sub(s);
    if len < 34 {
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

    // Une seule abduction par fenêtre ancrée : chaque point de coupe compare fenêtre k et fenêtre k + offset.
    let traces: Vec<AbductionTrace> = anchors
        .iter()
        .map(|&a| graph.abduce_with_steps(&rels[a..a + win], None, steps))
        .collect();

    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for k in 0..(anchors.len() - offset) {
        let split_p = anchors[k] + win;
        if split_p < s + 14 || e.saturating_sub(split_p) < 14 {
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
        let disc = 0.55 * state_disc + 0.45 * err_disc;
        candidates.push((split_p, disc));
    }

    // Sélection des pics locaux de discontinuité > 0.36 espacés d'au moins 14 octets
    let mut peaks: Vec<(usize, f32)> = Vec::new();
    for i in 0..candidates.len() {
        let (p, d) = candidates[i];
        if d <= 0.36 {
            continue;
        }
        let left_ok = i == 0 || d >= candidates[i - 1].1;
        let right_ok = i + 1 == candidates.len() || d >= candidates[i + 1].1;
        if left_ok && right_ok {
            peaks.push((p, d));
        }
    }
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut chosen: Vec<usize> = Vec::new();
    for (p, _) in peaks {
        if chosen.iter().all(|&c| c.abs_diff(p) >= 14) {
            chosen.push(p);
        }
    }
    if chosen.is_empty() {
        return vec![(s, e)];
    }
    chosen.sort_unstable();

    // En mode `ask` (exact), raffine chaque frontière à l'octet près dans [p-3..=p+3]
    if exact {
        for p in &mut chosen {
            let mut best_p = *p;
            let mut best_d = f32::NEG_INFINITY;
            let lo = (*p).saturating_sub(3).max(s + 12);
            let hi = (*p + 3).min(e.saturating_sub(12));
            for cand in lo..=hi {
                let l_start = cand.saturating_sub(16).max(s);
                let r_end = (cand + 16).min(e);
                let d = abductive_discontinuity(graph, rels, (l_start, cand), (cand, r_end), SEARCH_STEPS);
                if d > best_d {
                    best_d = d;
                    best_p = cand;
                }
            }
            *p = best_p;
        }
    }

    let mut segs = Vec::with_capacity(chosen.len() + 1);
    let mut cur = s;
    for p in chosen {
        if p >= cur + 10 && e >= p + 10 {
            segs.push((cur, p));
            cur = p;
        }
    }
    segs.push((cur, e));
    segs
}

/// Découvre automatiquement les régimes causaux et leurs référents passifs sur un flux
/// d'octets brut continu : zéro `\n`, zéro séparateur, 100% piloté par `PhiGraph`.
pub fn discover_causal_regimes_from_encoded(
    graph: &PhiGraph,
    rels: &[[f32; REL_DIM]],
    gates: &[f32],
    exact: bool,
) -> Vec<DiscoveredRegime> {
    let n = rels.len();
    if n < 12 {
        let mg = if n == 0 {
            0.0
        } else {
            gates.iter().sum::<f32>() / (n as f32)
        };
        return vec![DiscoveredRegime {
            core_start: 0,
            core_end: n,
            referent: None,
            mean_gate: mg,
        }];
    }

    // Lissage minimal (rayon 1) pour éviter les micro-coupures d'un octet
    let mut smooth = vec![0.0f32; n];
    for t in 0..n {
        let lo = t.saturating_sub(1);
        let hi = (t + 1).min(n - 1);
        let sum: f32 = gates[lo..=hi].iter().sum();
        smooth[t] = sum / ((hi - lo + 1) as f32);
    }

    // 1. Îlots actifs issus de la porte causale neuronale g(t)
    let mut raw_runs: Vec<(usize, usize)> = Vec::new();
    let mut in_run = false;
    let mut run_start = 0usize;
    for t in 0..n {
        if smooth[t] >= 0.45 {
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

    // Fusionne les micro-trous <= 3 octets (ponts uniques internes à un moule)
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in raw_runs {
        if let Some(last) = merged.last_mut()
            && s.saturating_sub(last.1) <= 3
        {
            last.1 = e;
            continue;
        }
        merged.push((s, e));
    }

    // Élimine les micro-blips accidentels < 10 octets (ex: mini-palindrome dans une étiquette)
    let active_runs: Vec<(usize, usize)> = merged
        .into_iter()
        .filter(|&(s, e)| e.saturating_sub(s) >= 10)
        .collect();

    if active_runs.is_empty() {
        let mg = gates.iter().sum::<f32>() / (n as f32);
        return vec![DiscoveredRegime {
            core_start: 0,
            core_end: n,
            referent: None,
            mean_gate: mg,
        }];
    }

    // 2. Scinde les îlots qui contiennent plusieurs régimes causaux collés via D(t)
    let mut cores: Vec<(usize, usize)> = Vec::new();
    for (s, e) in active_runs {
        let split = split_run_by_abduction(graph, rels, s, e, exact);
        cores.extend(split);
    }

    // 3. Associe les zones passives adjacentes comme référents induits (préfixe ou suffixe)
    let k = cores.len();
    let lead_gap = cores[0].0;
    let trail_gap = n.saturating_sub(cores[k - 1].1);
    let suffix_mode = trail_gap >= 3 && lead_gap < 3;

    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let (cs, ce) = cores[i];
        let mg = gates[cs..ce].iter().sum::<f32>() / ((ce - cs).max(1) as f32);

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

pub fn discover_causal_regimes(
    graph: &PhiGraph,
    bytes: &[u8],
    exact: bool,
) -> Vec<DiscoveredRegime> {
    let sub = RawSubstrate::from_bytes(bytes);
    let (rels, gates) = graph.encode_with_gate(&sub);
    discover_causal_regimes_from_encoded(graph, &rels, &gates, exact)
}

pub fn compute_phi_sharp_prepared(
    graph: &PhiGraph,
    prep: &PreparedEpisode,
    exact_mip: bool,
) -> PhiReport {
    let (rels, gates_orig) = graph.encode_with_gate(&prep.orig);
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

    // 1. Discrimination + contraste de la porte causale neuronale (structuré vs mélangé)
    let (neg_rels, gates_neg) = graph.encode_with_gate(&prep.neg);
    let (_, _, cross_err) = graph.evaluate_hypothesis(&trace.final_state, &neg_rels);
    let neg_trace = graph.abduce(&neg_rels, None);

    let err_gap = ((cross_err - residual) / (cross_err + 0.05)).clamp(0.0, 1.0);
    let state_gap = (1.0 - cosine_sim(&trace.final_state, &neg_trace.final_state)).clamp(0.0, 1.0);
    let discrimination = (0.65 * err_gap + 0.35 * state_gap).clamp(1e-4, 1.0);

    let mean_g_orig = gates_orig.iter().sum::<f32>() / (gates_orig.len().max(1) as f32);
    let mean_g_neg = gates_neg.iter().sum::<f32>() / (gates_neg.len().max(1) as f32);
    let gate_contrast = (mean_g_orig * (1.0 - mean_g_neg)).clamp(0.0, 1.0);

    // 2. Projection (réfutation, déplacement, abstraction par bijection arbitraire)
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

    // 3. Intégration cause-effet (IIT 4.0)
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

    // 4. Agrégation
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

/// Signal non-supervisé sur flux continu brut : découvre les régimes via `PhiGraph` seul
/// puis mesure la sélectivité causale de la matrice de résonance réciproque.
pub fn compute_composite_phi_prepared(
    graph: &PhiGraph,
    prep: &PreparedComposite,
) -> CompositePhi {
    let (rels, gates) = graph.encode_with_gate(&prep.stream);
    let regimes = discover_causal_regimes_from_encoded(graph, &rels, &gates, false);
    let m = regimes.len();
    if m < 2 {
        return CompositePhi {
            score: -0.25,
            r_top: 0.0,
            r_rest: 0.0,
            selectivity: 0.0,
            mean_residual: 1.0,
            regimes_found: m,
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
        mean_residual,
        regimes_found: m,
        top_pair: (top_i, top_j),
    }
}

pub fn compute_composite_phi(graph: &PhiGraph, bytes: &[u8]) -> CompositePhi {
    let prep = PreparedComposite::from_bytes(bytes);
    compute_composite_phi_prepared(graph, &prep)
}

/// Projection Référentielle réciproque sur les régimes découverts par `PhiGraph`.
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
    (best_start, bytes[best_start..best_start + win].to_vec(), best_err)
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
