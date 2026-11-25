//! φ_sharp (puissance cause-effet intégrée inspirée d'IIT 4.0), Sens(H, E) et Résidu.
//!
//! Principes implémentés sans compromis :
//! 1. Répertoires Cause et Effet distincts via les couplages non-réciproques.
//! 2. Partition d'Information Minimale (MIP) : toute hypothèse séparable ou copie triviale H := E
//!    a une intégration φ_MIP ≈ 0.
//! 3. Stabilité différentielle sous réfutation, déplacement et abstraction.
//! 4. Sens(H, E) = Générativité ∧ Discrimination ∧ Projection (par produit et min, sans compensation, sans MDL).
//! 5. Résidu couplé directement au score afin d'empêcher l'optimisation cosmétique de H sans expliquer E.

use crate::math::{iit_intrinsic_difference, iit_partition_loss, normalize_prob, tanh_clamped};
use crate::substrate::{D, HypothesisState, N_CTX, N_LAT, PerceivedGraph};

#[derive(Clone, Debug)]
pub struct PhiReport {
    pub phi_sharp: f64,
    pub phi_iit_mip: f64,
    pub cause_effect_power: f64,
    pub generativity: f64,
    pub discrimination: f64,
    pub projection: f64,
    pub sens_min: f64,
    pub sens_prod: f64,
    pub stab_refutation: f64,
    pub stab_displacement: f64,
    pub stab_abstraction: f64,
    pub stability_geo: f64,
    pub residue_score: f64,
    /// Couple de symboles (u, v) présentant le plus fort déficit explicatif dans E
    pub top_residue_pair: (usize, usize),
}

#[derive(Clone, Copy, Debug)]
pub struct EvalOptions {
    pub test_stability: bool,
    pub disable_refutation: bool,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            test_stability: true,
            disable_refutation: false,
        }
    }
}

/// Construit les répertoires Cause (arrière) et Effet (avant) sur les N_LAT nœuds latents,
/// en respectant la non-réciprocité J_{i->j} != J_{j->i}.
fn build_cause_effect_repertoires(
    h: &HypothesisState,
    cut_mask: Option<&[bool; N_LAT]>,
) -> ([f64; N_LAT], [f64; N_LAT], [f64; N_LAT], [f64; N_LAT]) {
    let mut act = [0.0; N_LAT];
    for i in 0..N_LAT {
        let norm: f64 = h.lat_states[i].iter().map(|x| x * x).sum::<f64>().sqrt();
        act[i] = (0.25 + norm).min(2.5);
    }

    let mut eff = [0.0; N_LAT];
    let mut cau = [0.0; N_LAT];
    let mut u_eff = [0.0; N_LAT];
    let mut u_cau = [0.0; N_LAT];

    for j in 0..N_LAT {
        let mut s_eff = 0.0;
        let mut s_cau = 0.0;
        let mut su_e = 0.0;
        let mut su_c = 0.0;
        for i in 0..N_LAT {
            let cut = match cut_mask {
                Some(mask) => mask[i] != mask[j],
                None => false,
            };
            let w_ij = if cut { 0.0 } else { h.lat_couplings[i * N_LAT + j] };
            let w_ji = if cut { 0.0 } else { h.lat_couplings[j * N_LAT + i] };
            s_eff += act[i] * w_ij;
            s_cau += act[i] * w_ji;
            su_e += w_ij;
            su_c += w_ji;
        }
        eff[j] = (1.6 * s_eff).clamp(-8.0, 8.0).exp();
        cau[j] = (1.6 * s_cau).clamp(-8.0, 8.0).exp();
        u_eff[j] = (0.4 * su_e).clamp(-8.0, 8.0).exp();
        u_cau[j] = (0.4 * su_c).clamp(-8.0, 8.0).exp();
    }

    normalize_prob(&mut eff);
    normalize_prob(&mut cau);
    normalize_prob(&mut u_eff);
    normalize_prob(&mut u_cau);
    (eff, cau, u_eff, u_cau)
}

/// Calcule φ_MIP (IIT 4.0) et la puissance cause-effet intrinsèque brute.
/// Détecte et annule les configurations dégénérées de copie directe H := E.
pub fn compute_iit_mip(h: &HypothesisState, pg: &PerceivedGraph) -> (f64, f64) {
    let (eff, cau, u_eff, u_cau) = build_cause_effect_repertoires(h, None);
    let id_eff = iit_intrinsic_difference(&eff, &u_eff);
    let id_cau = iit_intrinsic_difference(&cau, &u_cau);
    let cause_effect_power = (id_eff.min(id_cau)).clamp(0.0, 2.0);

    // Pénalité explicite de copie triviale H := E :
    // Si sym_trans est quasiment identique à emp_trans1 ET que les couplages hors-diagonale sont faibles,
    // il n'y a aucune intégration causale latente.
    let mut off_diag_energy = 0.0;
    let mut asym_energy = 0.0;
    for i in 0..N_LAT {
        for j in 0..N_LAT {
            if i != j {
                let a = h.lat_couplings[i * N_LAT + j];
                let b = h.lat_couplings[j * N_LAT + i];
                off_diag_energy += a.abs();
                asym_energy += (a - b).abs();
            }
        }
    }
    let mean_off = off_diag_energy / ((N_LAT * (N_LAT - 1)) as f64);
    let mean_asym = asym_energy / ((N_LAT * (N_LAT - 1)) as f64);

    if mean_off < 0.03 {
        return (0.0, 0.0);
    }

    // Test de proximité avec la copie littérale emp_trans1
    let k = h.k;
    let mut copy_l1 = 0.0;
    for idx in 0..(k * k) {
        copy_l1 += (h.sym_trans[idx] - pg.emp_trans1[idx]).abs();
    }
    let mean_copy_diff = copy_l1 / (k as f64).max(1.0);
    // Si search_depth == 0 ou si H est copie littérale non-asymétrique, l'intégration est quasi nulle
    let non_trivial_factor = if h.search_depth == 0 {
        0.08
    } else {
        let diff_factor = (mean_copy_diff * 4.0).clamp(0.15, 1.0);
        let asym_factor = (mean_asym * 5.0).clamp(0.1, 1.0);
        (diff_factor * asym_factor).sqrt()
    };

    // Recherche de la Partition d'Information Minimale (MIP) :
    // On évalue toutes les coupes 1-vs-(N-1) ainsi que plusieurs bipartitions équilibrées.
    let mut min_phi = f64::INFINITY;

    for single in 0..N_LAT {
        let mut mask = [false; N_LAT];
        mask[single] = true;
        let (p_eff, p_cau, _, _) = build_cause_effect_repertoires(h, Some(&mask));
        let loss_e = iit_partition_loss(&eff, &p_eff);
        let loss_c = iit_partition_loss(&cau, &p_cau);
        let phi_cut = loss_e.min(loss_c);
        if phi_cut < min_phi {
            min_phi = phi_cut;
        }
    }

    // Bipartitions équilibrées (parité, moitié haute/basse, alternance par blocs de 2 et 3)
    for pattern in 0..4 {
        let mut mask = [false; N_LAT];
        for i in 0..N_LAT {
            mask[i] = match pattern {
                0 => i < N_LAT / 2,
                1 => i % 2 == 0,
                2 => (i / 2) % 2 == 0,
                _ => (i / 3) % 2 == 0,
            };
        }
        let (p_eff, p_cau, _, _) = build_cause_effect_repertoires(h, Some(&mask));
        let loss_e = iit_partition_loss(&eff, &p_eff);
        let loss_c = iit_partition_loss(&cau, &p_cau);
        let phi_cut = loss_e.min(loss_c);
        if phi_cut < min_phi {
            min_phi = phi_cut;
        }
    }

    if !min_phi.is_finite() {
        min_phi = 0.0;
    }

    // Normalisation douce dans [0, 1] via tanh
    let phi_mip = tanh_clamped(min_phi * 4.5 * non_trivial_factor) * (mean_off * 2.5).clamp(0.0, 1.0);
    let ce_norm = tanh_clamped(cause_effect_power * 3.0 * non_trivial_factor);
    (phi_mip.max(0.0), ce_norm.max(0.0))
}

/// Score continu de Générativité(H, E) dans [0, 1] sans pénalité MDL.
/// Mesure la capacité de H à régénérer la dynamique transitionnelle et sautée de E.
pub fn score_generativity(h: &HypothesisState, pg: &PerceivedGraph) -> f64 {
    let k = pg.k();
    if k <= 1 || pg.sym_seq.len() < 2 {
        return 0.0;
    }
    let uniform_ll = -(k as f64).ln();

    // Pas 1 : prédiction directe via sym_trans
    let mut ll1 = 0.0;
    let mut count1 = 0.0;
    for w in pg.sym_seq.windows(2) {
        let p = h.sym_trans[w[0] * k + w[1]].max(1e-9);
        ll1 += p.ln();
        count1 += 1.0;
    }
    let mean_ll1 = ll1 / count1.max(1.0);

    // Pas 2 : régénération à 2 pas (sym_trans^2) comparée à E à pas 2 (structure relationnelle profonde)
    let mut ll2 = 0.0;
    let mut count2 = 0.0;
    if pg.sym_seq.len() >= 3 {
        let mut trans2 = vec![0.0; k * k];
        for u in 0..k {
            for v in 0..k {
                let mut s = 0.0;
                for m in 0..k {
                    s += h.sym_trans[u * k + m] * h.sym_trans[m * k + v];
                }
                trans2[u * k + v] = s.max(1e-9);
            }
        }
        for i in 0..(pg.sym_seq.len() - 2) {
            let u = pg.sym_seq[i];
            let v2 = pg.sym_seq[i + 2];
            ll2 += trans2[u * k + v2].ln();
            count2 += 1.0;
        }
    }
    let mean_ll2 = if count2 > 0.0 { ll2 / count2 } else { uniform_ll };

    let gain1 = ((mean_ll1 - uniform_ll) / (-uniform_ll).max(0.5)).clamp(0.0, 1.5);
    let gain2 = ((mean_ll2 - uniform_ll) / (-uniform_ll).max(0.5)).clamp(0.0, 1.5);

    // Combinaison multiplicative douce sans compensation extrême
    let raw = 0.65 * gain1 + 0.35 * gain2;
    tanh_clamped(raw * 1.4).clamp(0.0, 1.0)
}

/// Score continu de Discrimination(H, E) dans [0, 1].
/// Compare l'ajustement de H sur le vrai E contre des leurres qui brisent la structure de E :
/// 1) séquence inversée (teste la flèche causale non-réciproque),
/// 2) permutation déterministe des blocs,
/// 3) décalage circulaire brisé.
pub fn score_discrimination(h: &HypothesisState, pg: &PerceivedGraph) -> f64 {
    let k = pg.k();
    let n = pg.sym_seq.len();
    if k <= 1 || n < 4 {
        return 0.0;
    }

    let seq_ll = |seq: &[usize]| -> f64 {
        let mut s = 0.0;
        for w in seq.windows(2) {
            s += h.sym_trans[w[0] * k + w[1]].max(1e-9).ln();
        }
        s / ((seq.len() - 1).max(1) as f64)
    };

    let ll_true = seq_ll(&pg.sym_seq);

    // Leurre 1 : inversion temporelle (détecte si H capture la causalité asymétrique)
    let mut rev = pg.sym_seq.clone();
    rev.reverse();
    let ll_rev = seq_ll(&rev);

    // Leurre 2 : permutation déterministe intra-blocs qui détruit la grammaire locale
    let mut scrambled = pg.sym_seq.clone();
    for i in (0..n.saturating_sub(1)).step_by(2) {
        let j = (i * 7 + 3) % n;
        scrambled.swap(i, j);
    }
    let ll_scr = seq_ll(&scrambled);

    // Leurre 3 : distribution marginale plate (non-spécificité)
    let mut ll_marg = 0.0;
    for w in pg.sym_seq.windows(2) {
        let p_mix = 0.5 * h.sym_trans[w[0] * k + w[1]] + 0.5 * pg.emp_marginal[w[1]];
        ll_marg += p_mix.max(1e-9).ln();
    }
    ll_marg /= (n - 1) as f64;

    let margin_scr = (ll_true - ll_scr).max(0.0);
    let margin_rev = (ll_true - ll_rev).max(0.0);
    let margin_marg = (ll_true - ll_marg).max(0.0);

    // Spécificité non-compensatoire : H doit battre le brouillage ET ne pas être isotrope
    let combined = 0.55 * margin_scr + 0.30 * margin_rev + 0.15 * margin_marg;
    tanh_clamped(combined * 2.2).clamp(0.0, 1.0)
}

/// Score continu de Projection(H, E) dans [0, 1].
/// Teste si la structure relationnelle de H se projette de façon cohérente sur des vues partielles
/// (moitié gauche -> moitié droite, et segments pairs -> segments impairs).
pub fn score_projection(h: &HypothesisState, pg: &PerceivedGraph) -> f64 {
    let k = pg.k();
    let n = pg.sym_seq.len();
    if k <= 1 || n < 6 {
        return 0.0;
    }

    let uniform_ll = -(k as f64).ln();
    let mid = n / 2;

    let span_gain = |slice: &[usize]| -> f64 {
        if slice.len() < 2 {
            return 0.0;
        }
        let mut ll = 0.0;
        for w in slice.windows(2) {
            ll += h.sym_trans[w[0] * k + w[1]].max(1e-9).ln();
        }
        let mean_ll = ll / ((slice.len() - 1) as f64);
        ((mean_ll - uniform_ll) / (-uniform_ll).max(0.5)).clamp(0.0, 1.2)
    };

    let g_left = span_gain(&pg.sym_seq[..mid]);
    let g_right = span_gain(&pg.sym_seq[mid..]);

    // Consistance de projection croisée : H ne doit pas expliquer seulement un côté de E
    let min_half = g_left.min(g_right);
    let balance = 1.0 - (g_left - g_right).abs().min(1.0);

    // Cohérence entre la projection latente globale et les poids empiriques des segments
    let mut ctx_align = 0.0;
    let mut ctx_count = 0.0;
    for c in 0..N_CTX {
        let (st, en) = pg.ctx_spans[c];
        if en > st + 1 {
            let mut local_score = 0.0;
            for w in pg.sym_seq[st..en].windows(2) {
                local_score += h.sym_trans[w[0] * k + w[1]];
            }
            ctx_align += (local_score / ((en - st - 1) as f64) * (k as f64)).clamp(0.0, 2.0) / 2.0;
            ctx_count += 1.0;
        }
    }
    let ctx_proj = if ctx_count > 0.0 { ctx_align / ctx_count } else { 0.0 };

    let raw = min_half * (0.5 + 0.5 * balance) * (0.4 + 0.6 * ctx_proj);
    tanh_clamped(raw * 1.6).clamp(0.0, 1.0)
}

/// Calcule le résidu explicatif : écart non expliqué entre E et la régénération par H,
/// plus déficit de fermeture causale dans H.
pub fn compute_residue(h: &HypothesisState, pg: &PerceivedGraph) -> (f64, (usize, usize)) {
    let k = pg.k();
    if k <= 1 {
        return (1.0, (0, 0));
    }

    let mut total_deficit = 0.0;
    let mut max_pair = (0usize, 0usize);
    let mut max_val = -1.0;

    for u in 0..k {
        let m_u = pg.emp_marginal[u];
        for v in 0..k {
            let emp = pg.emp_trans1[u * k + v];
            let pred = h.sym_trans[u * k + v];
            let gap = (emp - pred).max(0.0) * m_u;
            total_deficit += gap;
            if gap > max_val {
                max_val = gap;
                max_pair = (u, v);
            }
        }
    }

    // Déficit d'ouverture causale : entropie moyenne résiduelle de sym_trans
    let mut mean_ent_ratio = 0.0;
    let log_k = (k as f64).max(2.0).ln();
    for u in 0..k {
        let mut ent = 0.0;
        for v in 0..k {
            let p = h.sym_trans[u * k + v].max(1e-12);
            ent -= p * p.ln();
        }
        mean_ent_ratio += pg.emp_marginal[u] * (ent / log_k).clamp(0.0, 1.0);
    }

    let res = (0.65 * total_deficit * 2.2 + 0.35 * mean_ent_ratio).clamp(0.0, 1.0);
    (res, max_pair)
}

/// Transformation abductive 1 : Réfutation (perturbe les 25% de couplages causaux les plus forts).
pub fn apply_refutation(h: &HypothesisState, pg: &PerceivedGraph) -> HypothesisState {
    let mut ref_h = h.clone();
    let mut edges: Vec<(f64, usize)> = ref_h
        .lat_couplings
        .iter()
        .enumerate()
        .filter(|(idx, _)| idx / N_LAT != idx % N_LAT)
        .map(|(idx, &v)| (v.abs(), idx))
        .collect();
    edges.sort_by(|a, b| b.0.total_cmp(&a.0));

    let cut_n = (edges.len() / 4).max(1);
    for &(_, idx) in edges.iter().take(cut_n) {
        ref_h.lat_couplings[idx] *= -0.15; // Inhibition active de la piste dominante
    }
    rebuild_sym_trans_from_lat(&mut ref_h, pg);
    ref_h
}

/// Transformation abductive 2 : Déplacement (décale les ancrages contextuels et rôles latents).
pub fn apply_displacement(h: &HypothesisState, pg: &PerceivedGraph) -> HypothesisState {
    let mut dep_h = h.clone();
    dep_h.ctx_states.rotate_left(1);
    // Déplacement cyclique léger des rôles sur les nœuds latents tout en préservant le graphe relatif
    let mut shifted_lat = [[0.0; D]; N_LAT];
    for i in 0..N_LAT {
        let next = (i + 1) % N_LAT;
        for d in 0..D {
            shifted_lat[i][d] = 0.85 * dep_h.lat_states[i][d] + 0.15 * dep_h.lat_states[next][d];
        }
    }
    dep_h.lat_states = shifted_lat;
    // Décale partiellement les lignes de sym_to_lat / lat_to_sym pour tester l'équivariance des rôles
    let k = dep_h.k;
    let mut new_s2l = dep_h.sym_to_lat.clone();
    let mut new_l2s = dep_h.lat_to_sym.clone();
    for s in 0..k {
        for l in 0..N_LAT {
            let l2 = (l + 1) % N_LAT;
            new_s2l[s * N_LAT + l] = 0.8 * dep_h.sym_to_lat[s * N_LAT + l] + 0.2 * dep_h.sym_to_lat[s * N_LAT + l2];
            new_l2s[l * k + s] = 0.8 * dep_h.lat_to_sym[l * k + s] + 0.2 * dep_h.lat_to_sym[l2 * k + s];
        }
    }
    dep_h.sym_to_lat = new_s2l;
    dep_h.lat_to_sym = new_l2s;
    rebuild_sym_trans_from_lat(&mut dep_h, pg);
    dep_h
}

/// Transformation abductive 3 : Abstraction (regroupe les nœuds latents en paires d'invariants supérieurs).
pub fn apply_abstraction(h: &HypothesisState, pg: &PerceivedGraph) -> HypothesisState {
    let mut abs_h = h.clone();
    for pair in (0..N_LAT).step_by(2) {
        let partner = (pair + 1) % N_LAT;
        for d in 0..D {
            let mean = 0.5 * (abs_h.lat_states[pair][d] + abs_h.lat_states[partner][d]);
            abs_h.lat_states[pair][d] = 0.75 * abs_h.lat_states[pair][d] + 0.25 * mean;
            abs_h.lat_states[partner][d] = 0.75 * abs_h.lat_states[partner][d] + 0.25 * mean;
        }
        for j in 0..N_LAT {
            let m_out = 0.5 * (abs_h.lat_couplings[pair * N_LAT + j] + abs_h.lat_couplings[partner * N_LAT + j]);
            abs_h.lat_couplings[pair * N_LAT + j] = 0.75 * abs_h.lat_couplings[pair * N_LAT + j] + 0.25 * m_out;
            abs_h.lat_couplings[partner * N_LAT + j] = 0.75 * abs_h.lat_couplings[partner * N_LAT + j] + 0.25 * m_out;
        }
    }
    rebuild_sym_trans_from_lat(&mut abs_h, pg);
    abs_h
}

fn rebuild_sym_trans_from_lat(h: &mut HypothesisState, pg: &PerceivedGraph) {
    let k = h.k;
    if h.search_depth == 0 {
        return;
    }
    let mut lat_trans = [0.0; N_LAT * N_LAT];
    for i in 0..N_LAT {
        let mut sum = 0.0;
        for j in 0..N_LAT {
            let w = (2.2 * h.lat_couplings[i * N_LAT + j]).clamp(-8.0, 8.0).exp();
            lat_trans[i * N_LAT + j] = w;
            sum += w;
        }
        let inv = 1.0 / sum.max(1e-12);
        for j in 0..N_LAT {
            lat_trans[i * N_LAT + j] *= inv;
        }
    }

    for u in 0..k {
        let mut lat_after = [0.0; N_LAT];
        for j in 0..N_LAT {
            let mut acc = 0.0;
            for i in 0..N_LAT {
                acc += h.sym_to_lat[u * N_LAT + i] * lat_trans[i * N_LAT + j];
            }
            lat_after[j] = acc;
        }
        let mut row_sum = 0.0;
        for v in 0..k {
            let mut p_uv = 0.0;
            for j in 0..N_LAT {
                p_uv += lat_after[j] * h.lat_to_sym[j * k + v];
            }
            let val = (p_uv * (0.85 + 0.30 * pg.emp_cooccur[u * k + v])).max(1e-9);
            h.sym_trans[u * k + v] = val;
            row_sum += val;
        }
        let inv = 1.0 / row_sum.max(1e-12);
        for v in 0..k {
            h.sym_trans[u * k + v] *= inv;
        }
    }
}

/// Évaluation complète de φ_sharp(H, E), Sens(H, E), stabilité différentielle et résidu.
pub fn evaluate_phi_sharp(h: &HypothesisState, pg: &PerceivedGraph, opts: EvalOptions) -> PhiReport {
    let (phi_iit_mip, cause_effect_power) = compute_iit_mip(h, pg);
    let generativity = score_generativity(h, pg);
    let discrimination = score_discrimination(h, pg);
    let projection = score_projection(h, pg);

    let sens_min = generativity.min(discrimination).min(projection);
    let sens_prod = generativity * discrimination * projection;
    let (residue_score, top_residue_pair) = compute_residue(h, pg);

    let (stab_refutation, stab_displacement, stab_abstraction) = if opts.test_stability && h.search_depth > 0 {
        let base_g = generativity.max(1e-4);
        let base_p = phi_iit_mip.max(1e-4);

        let sr = if opts.disable_refutation {
            1.0 // En ablation sans réfutation, on ne pénalise ni ne vérifie la réfutation
        } else {
            let h_ref = apply_refutation(h, pg);
            let (p_ref, _) = compute_iit_mip(&h_ref, pg);
            let g_ref = score_generativity(&h_ref, pg);
            let r_p = (p_ref / base_p).clamp(0.0, 1.0);
            let r_g = (g_ref / base_g).clamp(0.0, 1.0);
            (r_p * r_g).sqrt()
        };

        let h_dep = apply_displacement(h, pg);
        let (p_dep, _) = compute_iit_mip(&h_dep, pg);
        let g_dep = score_generativity(&h_dep, pg);
        let sd = ((p_dep / base_p).clamp(0.0, 1.0) * (g_dep / base_g).clamp(0.0, 1.0)).sqrt();

        let h_abs = apply_abstraction(h, pg);
        let (p_abs, _) = compute_iit_mip(&h_abs, pg);
        let g_abs = score_generativity(&h_abs, pg);
        let sa = ((p_abs / base_p).clamp(0.0, 1.0) * (g_abs / base_g).clamp(0.0, 1.0)).sqrt();

        (sr, sd, sa)
    } else if h.search_depth == 0 {
        (0.05, 0.05, 0.05)
    } else {
        (1.0, 1.0, 1.0)
    };

    let stability_geo = (stab_refutation * stab_displacement * stab_abstraction).cbrt();
    let res_factor = (1.0 - 0.65 * residue_score).clamp(0.05, 1.0);
    let sens_factor = sens_prod.cbrt();

    // φ_sharp intègre : IIT-MIP * Stabilité différentielle * Sens non-compensatoire * (1 - Résidu)
    let phi_sharp = (phi_iit_mip * (0.35 + 0.65 * stability_geo) * (0.25 + 0.75 * sens_factor) * res_factor)
        .clamp(0.0, 1.0);

    PhiReport {
        phi_sharp,
        phi_iit_mip,
        cause_effect_power,
        generativity,
        discrimination,
        projection,
        sens_min,
        sens_prod,
        stab_refutation,
        stab_displacement,
        stab_abstraction,
        stability_geo,
        residue_score,
        top_residue_pair,
    }
}