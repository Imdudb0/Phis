use crate::graph::{AbductionTrace, NODES, PhiGraph, REL_DIM, STATE_DIM, extract_relations};
use crate::rng::Rng;

const SINGLETON_MASKS: [u8; 6] = [
    0b000001, 0b000010, 0b000100, 0b001000, 0b010000, 0b011111,
];

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

/// Calcule φ_sharp(E) : conjonction Générativité x Discrimination x Projection x Intégration,
/// trajectoire monotone depuis t0 (anti-sabotage) et coupes IIT incluant les singletons.
pub fn compute_phi_sharp(
    graph: &PhiGraph,
    bytes: &[u8],
    rng: &mut Rng,
    exact_mip: bool,
) -> PhiReport {
    let rels = extract_relations(bytes);
    let trace = graph.abduce(&rels, None);
    let residual = trace.final_mean_err;

    // Ancrage anti-sabotage : t0 est indépendant des poids (state = 0 => pred = 0).
    let r0 = trace.step_residuals[0].max(1e-4);
    let traj_gain = ((r0 - residual) / r0).clamp(0.0, 1.0);
    let mut bounce_pen = 0.0f32;
    for pair in trace.step_residuals.windows(2) {
        if pair[1] > pair[0] {
            bounce_pen += pair[1] - pair[0];
        }
    }

    let generativity = (-2.2 * residual).exp().clamp(1e-4, 1.0);

    // 1. Discrimination : même multiset d'octets, structure causale détruite par permutation
    let mut neg_bytes = bytes.to_vec();
    rng.shuffle(&mut neg_bytes);
    let neg_rels = extract_relations(&neg_bytes);
    let (_, _, cross_err) = graph.evaluate_hypothesis(&trace.final_state, &neg_rels);
    let neg_trace = graph.abduce(&neg_rels, None);

    let err_gap = ((cross_err - residual) / (cross_err + 0.05)).clamp(0.0, 1.0);
    let state_gap = (1.0 - cosine_sim(&trace.final_state, &neg_trace.final_state)).clamp(0.0, 1.0);
    let discrimination = (0.65 * err_gap + 0.35 * state_gap).clamp(1e-4, 1.0);

    // 2. Projection : stabilité sous réfutation, déplacement et abstraction non-triviale
    // 2a. Réfutation (corruption active de ~15% de E)
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

    // 2b. Déplacement (rotation cyclique : change les segments moitié 1 / moitié 2)
    let mut shift_bytes = bytes.to_vec();
    let shift = (bytes.len() / 3).max(1) % bytes.len();
    shift_bytes.rotate_left(shift);
    let shift_rels = extract_relations(&shift_bytes);
    let shift_trace = graph.abduce(&shift_rels, None);
    let displacement_stab =
        (0.5 * (1.0 + cosine_sim(&trace.final_state, &shift_trace.final_state))).clamp(0.0, 1.0);

    // 2c. Abstraction méta-relationnelle : bijection d'alphabet (0 mot commun) + fenêtre 75%
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

    // 3. Intégration cause-effet (MIP sur toutes les 31 coupes en ask, ou 3 singletons + 3 équilibrées)
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

    // 4. Agrégation bornée <= 1.0 + pénalité de résidu et d'anti-rebond
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

/// Extrait la fenêtre à plus forte densité explicative : faible résidu ET liens relationnels actifs.
pub fn extract_core_motif(
    bytes: &[u8],
    rels: &[[f32; REL_DIM]],
    pos_err: &[f32],
) -> (usize, Vec<u8>) {
    let win = (bytes.len() / 5).clamp(4, 10).min(bytes.len());
    let mut best_start = 0;
    let mut best_score = f32::NEG_INFINITY;

    for start in 0..=bytes.len().saturating_sub(win) {
        let mut score = 0.0f32;
        for t in start..start + win {
            let active = rels[t][0].max(0.0)
                + rels[t][2].max(0.0)
                + rels[t][3].max(0.0)
                + rels[t][4].max(0.0);
            score += (0.25 + active) / (pos_err[t] + 0.08);
        }
        score /= win as f32;
        if score > best_score {
            best_score = score;
            best_start = start;
        }
    }
    (best_start, bytes[best_start..best_start + win].to_vec())
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
