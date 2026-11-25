use crate::graph::{AbductionTrace, NODES, PhiGraph, REL_DIM, STATE_DIM, extract_relations};
use crate::rng::Rng;

const SINGLETON_MASKS: [u8; 6] = [
    0b000001, 0b000010, 0b000100, 0b001000, 0b010000, 0b011111,
];

#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug)]
pub struct PhiReport {
    pub phi_sharp: f32,
    pub generativity: f32,
    pub discrimination: f32,
    pub projection: f32,
    pub integration: f32,
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
    pub top_pair: (usize, usize),
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

/// Découpe `perceived.eff` en blocs si des frontières de ligne créent >= 2 blocs de taille >= 6.
pub fn segment_effect(bytes: &[u8]) -> Vec<Span> {
    let mut spans = Vec::new();
    let mut start = 0usize;
    for i in 0..=bytes.len() {
        if i == bytes.len() || bytes[i] == b'\n' {
            let s = start;
            let mut e = i;
            while e > s && bytes[e - 1] == b'\r' {
                e -= 1;
            }
            if e.saturating_sub(s) >= 6 {
                spans.push(Span { start: s, end: e });
            }
            start = i + 1;
        }
    }
    if spans.len() >= 2 {
        spans
    } else {
        vec![Span {
            start: 0,
            end: bytes.len(),
        }]
    }
}

/// Trouve la sous-fenêtre contiguë la mieux expliquée (résidu minimal, légère préférence pour
/// la couverture totale afin de ne découper que lorsqu'un préfixe/suffixe résiste vraiment).
pub fn best_explained_window(pos_err: &[f32], min_win: usize) -> (usize, usize, f32) {
    let n = pos_err.len();
    if n == 0 {
        return (0, 0, 0.0);
    }
    let min_w = min_win.min(n).max(1);
    let mut pref = vec![0.0f32; n + 1];
    for i in 0..n {
        pref[i + 1] = pref[i] + pos_err[i];
    }
    if n <= min_w {
        return (0, n, pref[n] / (n as f32));
    }

    let mut widths = [
        n,
        ((n * 3) / 4).max(min_w).min(n),
        ((n * 3) / 5).max(min_w).min(n),
        ((n * 9) / 20).max(min_w).min(n),
    ];
    widths.sort_unstable();

    let mut best_start = 0usize;
    let mut best_end = n;
    let mut best_avg = pref[n] / (n as f32);
    let mut best_obj = best_avg;

    let mut prev_w = 0usize;
    for &w in &widths {
        if w == prev_w || w == 0 {
            continue;
        }
        prev_w = w;
        let len_pen = 0.04 * (1.0 - (w as f32) / (n as f32));
        for s in 0..=(n - w) {
            let avg = (pref[s + w] - pref[s]) / (w as f32);
            let obj = avg + len_pen;
            if obj < best_obj {
                best_obj = obj;
                best_avg = avg;
                best_start = s;
                best_end = s + w;
            }
        }
    }
    (best_start, best_end, best_avg)
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

pub fn compute_phi_sharp(
    graph: &PhiGraph,
    bytes: &[u8],
    rng: &mut Rng,
    exact_mip: bool,
) -> PhiReport {
    let rels = extract_relations(bytes);
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

    // 1. Discrimination
    let mut neg_bytes = bytes.to_vec();
    rng.shuffle(&mut neg_bytes);
    let neg_rels = extract_relations(&neg_bytes);
    let (_, _, cross_err) = graph.evaluate_hypothesis(&trace.final_state, &neg_rels);
    let neg_trace = graph.abduce(&neg_rels, None);

    let err_gap = ((cross_err - residual) / (cross_err + 0.05)).clamp(0.0, 1.0);
    let state_gap = (1.0 - cosine_sim(&trace.final_state, &neg_trace.final_state)).clamp(0.0, 1.0);
    let discrimination = (0.65 * err_gap + 0.35 * state_gap).clamp(1e-4, 1.0);

    // 2. Projection (réfutation, déplacement, abstraction d'alphabet)
    let mut ref_bytes = bytes.to_vec();
    let corrupt_n = (bytes.len() / 6).max(1);
    for _ in 0..corrupt_n {
        let pos = rng.next_usize(ref_bytes.len());
        ref_bytes[pos] = ref_bytes[pos].wrapping_add(1 + rng.next_u8() % 250);
    }
    let ref_rels = extract_relations(&ref_bytes);
    let ref_trace = graph.abduce(&ref_rels, None);
    let (_, _, ref_on_orig) = graph.evaluate_hypothesis(&ref_trace.final_state, &rels);
    let ref_cos = 0.5 * (1.0 + cosine_sim(&trace.final_state, &ref_trace.final_state));
    let refutation_stab = (ref_cos * (-1.5 * ref_on_orig).exp()).clamp(0.0, 1.0);

    let mut shift_bytes = bytes.to_vec();
    let shift = (bytes.len() / 3).max(1) % bytes.len();
    shift_bytes.rotate_left(shift);
    let shift_rels = extract_relations(&shift_bytes);
    let shift_trace = graph.abduce(&shift_rels, None);
    let displacement_stab =
        (0.5 * (1.0 + cosine_sim(&trace.final_state, &shift_trace.final_state))).clamp(0.0, 1.0);

    let mul = rng.next_u8() | 1;
    let add = rng.next_u8().max(1);
    let sub_len = ((bytes.len() * 3) / 4).max(8).min(bytes.len());
    let start_off = rng.next_usize(bytes.len() - sub_len + 1);
    let abs_bytes: Vec<u8> = bytes[start_off..start_off + sub_len]
        .iter()
        .map(|&b| b.wrapping_mul(mul).wrapping_add(add))
        .collect();
    let abs_rels = extract_relations(&abs_bytes);
    let abs_trace = graph.abduce(&abs_rels, None);
    let abstraction_stab =
        (0.5 * (1.0 + cosine_sim(&trace.final_state, &abs_trace.final_state))).clamp(0.0, 1.0);

    let projection =
        ((refutation_stab + displacement_stab + abstraction_stab) / 3.0).clamp(1e-4, 1.0);

    // 3. Intégration cause-effet (IIT 4.0)
    let masks: Vec<u8> = if exact_mip {
        (1u8..(1u8 << (NODES - 1))).collect()
    } else {
        sample_training_cuts(rng).to_vec()
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
    let phi_sharp = 0.90 * conj + 0.10 * traj_gain - 0.30 * residual - 0.30 * bounce_pen;

    PhiReport {
        phi_sharp,
        generativity,
        discrimination,
        projection,
        integration,
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

/// Signal non-supervisé de Projection Référentielle sur un fichier composite `E` :
/// récompense une matrice de résonance causale sélective (paires qui s'expliquent
/// mutuellement avec un fort contraste contre les autres paires), sans jamais voir d'étiquettes.
pub fn compute_composite_phi(graph: &PhiGraph, bytes: &[u8]) -> CompositePhi {
    let spans = segment_effect(bytes);
    let m = spans.len();
    if m < 2 {
        return CompositePhi {
            score: 0.0,
            r_top: 0.0,
            r_rest: 0.0,
            selectivity: 0.0,
            mean_residual: 1.0,
            top_pair: (0, 0),
        };
    }

    let mut rels_list = Vec::with_capacity(m);
    let mut traces = Vec::with_capacity(m);
    let mut sum_res = 0.0f32;
    for sp in &spans {
        let r = extract_relations(&bytes[sp.start..sp.end]);
        let tr = graph.abduce(&r, None);
        sum_res += tr.final_mean_err;
        rels_list.push(r);
        traces.push(tr);
    }
    let mean_residual = sum_res / (m as f32);

    let mut pairs: Vec<(f32, usize, usize)> = Vec::with_capacity(m * (m - 1) / 2);
    for i in 0..m {
        for j in (i + 1)..m {
            let (_, pos_j, full_j) =
                graph.evaluate_hypothesis(&traces[i].final_state, &rels_list[j]);
            let (_, _, foc_j) = best_explained_window(&pos_j, 12);
            let e_ij = 0.25 * full_j + 0.75 * foc_j;

            let (_, pos_i, full_i) =
                graph.evaluate_hypothesis(&traces[j].final_state, &rels_list[i]);
            let (_, _, foc_i) = best_explained_window(&pos_i, 12);
            let e_ji = 0.25 * full_i + 0.75 * foc_i;

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
    let score = selectivity - 0.15 * mean_residual;

    CompositePhi {
        score,
        r_top,
        r_rest,
        selectivity,
        mean_residual,
        top_pair: (top_i, top_j),
    }
}

/// Projection Référentielle réciproque focalisée :
/// projette le bloc `query_idx` sur tous les autres blocs de `E`, isole la sous-zone qui
/// résonne (`M*`), vérifie l'explication réciproque `H(M*) -> Q` et détecte le référent induit.
pub fn project_query_to_molds(
    graph: &PhiGraph,
    bytes: &[u8],
    spans: &[Span],
    query_idx: usize,
) -> Vec<ProjectionMatch> {
    let q_sp = spans[query_idx];
    let q_bytes = &bytes[q_sp.start..q_sp.end];
    let q_rels = extract_relations(q_bytes);
    let q_trace = graph.abduce(&q_rels, None);

    let mut matches = Vec::with_capacity(spans.len().saturating_sub(1));
    for (idx, sp) in spans.iter().enumerate() {
        if idx == query_idx {
            continue;
        }
        let d_bytes = &bytes[sp.start..sp.end];
        let d_rels = extract_relations(d_bytes);

        // 1. Projection avant H(Q) -> M : isole la sous-zone causale M*
        let (_, pos_d, full_fwd) = graph.evaluate_hypothesis(&q_trace.final_state, &d_rels);
        let (c_start, c_end, foc_fwd) = best_explained_window(&pos_d, 10);

        // 2. Abduction réciproque sur M* -> H(M*) -> Q
        let core_bytes = &d_bytes[c_start..c_end];
        let core_rels = extract_relations(core_bytes);
        let core_trace = graph.abduce(&core_rels, None);
        let (_, pos_q, full_bwd) = graph.evaluate_hypothesis(&core_trace.final_state, &q_rels);
        let (_, _, foc_bwd) = best_explained_window(&pos_q, 10);

        let fwd_err = 0.15 * full_fwd + 0.85 * foc_fwd;
        let bwd_err = 0.25 * full_bwd + 0.75 * foc_bwd;
        let cross_err = 0.5 * (fwd_err + bwd_err);

        let cross_gen = (-2.2 * cross_err).exp().clamp(0.0, 1.0);
        let cos_sim = cosine_sim(&q_trace.final_state, &core_trace.final_state);
        let cos_pos = 0.5 * (1.0 + cos_sim);
        let score = (cross_gen * cos_pos).sqrt();

        // 3. Détection automatique d'un référent/étiquette complémentaire (sans séparateur codé en dur)
        let referent = if c_start >= 3 && d_bytes.len().saturating_sub(c_end) <= 2 {
            let comp_err = pos_d[0..c_start].iter().sum::<f32>() / (c_start as f32);
            if comp_err > foc_fwd * 1.25 + 0.05 {
                Some((0, c_start))
            } else {
                None
            }
        } else if d_bytes.len().saturating_sub(c_end) >= 3 && c_start <= 2 {
            let comp_len = d_bytes.len() - c_end;
            let comp_err = pos_d[c_end..].iter().sum::<f32>() / (comp_len as f32);
            if comp_err > foc_fwd * 1.25 + 0.05 {
                Some((c_end, d_bytes.len()))
            } else {
                None
            }
        } else {
            None
        };

        let byte_overlap = byte_overlap_ratio(q_bytes, core_bytes);
        matches.push(ProjectionMatch {
            dst_idx: idx,
            score,
            cross_err,
            cos_sim,
            core_start: c_start,
            core_end: c_end,
            referent,
            byte_overlap,
        });
    }

    matches.sort_by(|a, b| b.score.total_cmp(&a.score));
    matches
}

/// Noyau abductif sans heuristique pour fichier monolithique :
/// cherche la sous-fenêtre de `E` dont l'abduction locale régénère le mieux tout `E`.
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
        let sub_rels = extract_relations(&bytes[s..s + win]);
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
