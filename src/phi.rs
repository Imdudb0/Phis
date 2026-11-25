use crate::graph::{AbductionTrace, NODES, PhiGraph, STATE_DIM, extract_relations};
use crate::rng::Rng;

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

/// Calcule φ_sharp(E) : stabilité différentielle cause-effet sous réfutation,
/// déplacement, abstraction d'alphabet et bipartitions IIT, ancrée par le résidu.
pub fn compute_phi_sharp(
    graph: &PhiGraph,
    bytes: &[u8],
    rng: &mut Rng,
    exact_mip: bool,
) -> PhiReport {
    let rels = extract_relations(bytes);
    let trace = graph.abduce(&rels, None);
    let residual = trace.final_mean_err;

    let generativity = (-2.5 * residual).exp().clamp(1e-4, 1.0);
    let r1 = trace.step_residuals.get(1).copied().unwrap_or(residual);
    let traj_gain = (r1 - residual).max(0.0);

    // 1. Discrimination : même alphabet/fréquences, ordre causal détruit par permutation
    let mut neg_bytes = bytes.to_vec();
    rng.shuffle(&mut neg_bytes);
    let neg_rels = extract_relations(&neg_bytes);
    let (_, _, cross_err) = graph.evaluate_hypothesis(&trace.final_state, &neg_rels);
    let neg_trace = graph.abduce(&neg_rels, None);

    let err_gap = ((cross_err - residual) / (cross_err + 0.05)).clamp(0.0, 1.0);
    let state_gap = (1.0 - cosine_sim(&trace.final_state, &neg_trace.final_state)).clamp(0.0, 1.0);
    let discrimination = (0.6 * err_gap + 0.4 * state_gap).clamp(1e-4, 1.0);

    // 2. Projection : stabilité sous réfutation, déplacement et abstraction
    // 2a. Réfutation (corruption active d'une fraction de E)
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

    // 2b. Déplacement (rotation cyclique)
    let mut shift_bytes = bytes.to_vec();
    let shift = (bytes.len() / 3).max(1) % bytes.len();
    shift_bytes.rotate_left(shift);
    let shift_rels = extract_relations(&shift_bytes);
    let shift_trace = graph.abduce(&shift_rels, None);
    let displacement_stab =
        (0.5 * (1.0 + cosine_sim(&trace.final_state, &shift_trace.final_state))).clamp(0.0, 1.0);

    // 2c. Abstraction méta-relationnelle (bijection affine sur les octets : 0 mot commun)
    let mul = rng.next_u8() | 1; // impair => bijectif mod 256
    let add = rng.next_u8().max(1);
    let abs_bytes: Vec<u8> = bytes
        .iter()
        .map(|&b| b.wrapping_mul(mul).wrapping_add(add))
        .collect();
    let abs_rels = extract_relations(&abs_bytes);
    let abs_trace = graph.abduce(&abs_rels, None);
    let abstraction_stab =
        (0.5 * (1.0 + cosine_sim(&trace.final_state, &abs_trace.final_state))).clamp(0.0, 1.0);

    let projection = ((refutation_stab + displacement_stab + abstraction_stab) / 3.0).clamp(1e-4, 1.0);

    // 3. Intégration cause-effet (Partition d'Information Minimale inspirée IIT 4.0)
    let masks: Vec<u8> = if exact_mip {
        (1u8..(1u8 << (NODES - 1))).collect()
    } else {
        vec![0b000111, 0b001011, 0b010101, 0b000011, 0b001100, 0b011000]
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
    let integration = (1.0 - (-3.0 * mip_loss).exp()).clamp(0.0, 1.0);

    // 4. Agrégation conjonctive + ancrage par le résidu (empêche l'auto-optimisation vide)
    let conj = (generativity * discrimination * projection * (0.2 + 0.8 * integration)).powf(0.25);
    let phi_sharp = conj - 0.35 * residual + 0.15 * traj_gain;

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
        refutation_stab,
        displacement_stab,
        abstraction_stab,
        trace,
    }
}

/// Extrait le noyau causal auto-spécifié (fenêtre la plus cohérente dans l'alphabet de E).
pub fn extract_core_motif(bytes: &[u8], pos_err: &[f32]) -> (usize, Vec<u8>) {
    let win = (bytes.len() / 4).clamp(4, 8).min(bytes.len());
    let mut best_start = 0;
    let mut best_score = f32::INFINITY;
    for start in 0..=bytes.len().saturating_sub(win) {
        let avg: f32 = pos_err[start..start + win].iter().sum::<f32>() / (win as f32);
        if avg < best_score {
            best_score = avg;
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
