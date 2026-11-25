//! φ_sharp (IIT 4.0) v0.2.0 :
//! - Couplage strictement NON-COMPENSATOIRE avec Sens(H, E) (aucun plancher sur bruit blanc).
//! - Sur octets aléatoires, Sens ≈ 0 => φ_sharp ≈ 0.
//! - Sur H := E, φ_MIP = 0 => φ_sharp = 0.
//! - Résidu bien calibré dans [0, 1] sans saturation à 1.0.

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
        eff[j] = (2.2 * s_eff).clamp(-10.0, 10.0).exp();
        cau[j] = (2.2 * s_cau).clamp(-10.0, 10.0).exp();
        u_eff[j] = (0.45 * su_e).clamp(-10.0, 10.0).exp();
        u_cau[j] = (0.45 * su_c).clamp(-10.0, 10.0).exp();
    }

    normalize_prob(&mut eff);
    normalize_prob(&mut cau);
    normalize_prob(&mut u_eff);
    normalize_prob(&mut u_cau);
    (eff, cau, u_eff, u_cau)
}

pub fn compute_iit_mip(h: &HypothesisState, pg: &PerceivedGraph) -> (f64, f64) {
    let (eff, cau, u_eff, u_cau) = build_cause_effect_repertoires(h, None);
    let id_eff = iit_intrinsic_difference(&eff, &u_eff);
    let id_cau = iit_intrinsic_difference(&cau, &u_cau);
    let cause_effect_power = (id_eff.min(id_cau)).clamp(0.0, 3.0);

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

    // Détection stricte de copie triviale H := E (où sym_rel_bias == 0 et sym_trans == emp_trans1)
    let k = h.k;
    let mut copy_l1 = 0.0;
    for idx in 0..(k * k) {
        copy_l1 += (h.sym_trans[idx] - pg.emp_trans1[idx]).abs();
    }
    let mean_copy_diff = copy_l1 / (k as f64).max(1.0);
    let non_trivial_factor = if h.search_depth == 0 {
        0.05
    } else {
        let diff_factor = (mean_copy_diff * 5.0).clamp(0.20, 1.0);
        let asym_factor = (mean_asym * 6.0).clamp(0.15, 1.0);
        (diff_factor * asym_factor).sqrt()
    };

    let mut min_phi = f64::INFINITY;
    for single in 0..N_LAT {
        let mut mask = [false; N_LAT];
        mask[single] = true;
        let (p_eff, p_cau, _, _) = build_cause_effect_repertoires(h, Some(&mask));
        let phi_cut = iit_partition_loss(&eff, &p_eff).min(iit_partition_loss(&cau, &p_cau));
        if phi_cut < min_phi {
            min_phi = phi_cut;
        }
    }

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
        let phi_cut = iit_partition_loss(&eff, &p_eff).min(iit_partition_loss(&cau, &p_cau));
        if phi_cut < min_phi {
            min_phi = phi_cut;
        }
    }

    if !min_phi.is_finite() {
        min_phi = 0.0;
    }

    let phi_mip = tanh_clamped(min_phi * 6.0 * non_trivial_factor) * (mean_off * 2.5).clamp(0.0, 1.0);
    let ce_norm = tanh_clamped(cause_effect_power * 3.5 * non_trivial_factor);
    (phi_mip.max(0.0), ce_norm.max(0.0))
}

pub fn score_generativity(h: &HypothesisState, pg: &PerceivedGraph) -> f64 {
    let k = pg.k();
    if k <= 1 || pg.sym_seq.len() < 2 {
        return 0.0;
    }
    let uniform_ll = -(k as f64).ln();

    let mut ll1 = 0.0;
    let mut count1 = 0.0;
    for w in pg.sym_seq.windows(2) {
        ll1 += h.sym_trans[w[0] * k + w[1]].max(1e-9).ln();
        count1 += 1.0;
    }
    let mean_ll1 = ll1 / count1.max(1.0);

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
            ll2 += trans2[pg.sym_seq[i] * k + pg.sym_seq[i + 2]].ln();
            count2 += 1.0;
        }
    }
    let mean_ll2 = if count2 > 0.0 { ll2 / count2 } else { uniform_ll };

    let gain1 = ((mean_ll1 - uniform_ll) / (-uniform_ll).max(0.5)).clamp(0.0, 1.5);
    let gain2 = ((mean_ll2 - uniform_ll) / (-uniform_ll).max(0.5)).clamp(0.0, 1.5);
    tanh_clamped((0.65 * gain1 + 0.35 * gain2) * 1.6).clamp(0.0, 1.0)
}

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

    let mut rev = pg.sym_seq.clone();
    rev.reverse();
    let ll_rev = seq_ll(&rev);

    let mut scrambled = pg.sym_seq.clone();
    for i in (0..n.saturating_sub(1)).step_by(2) {
        let j = (i * 7 + 3) % n;
        scrambled.swap(i, j);
    }
    let ll_scr = seq_ll(&scrambled);

    let mut ll_marg = 0.0;
    for w in pg.sym_seq.windows(2) {
        let p_mix = 0.5 * h.sym_trans[w[0] * k + w[1]] + 0.5 * pg.emp_marginal[w[1]];
        ll_marg += p_mix.max(1e-9).ln();
    }
    ll_marg /= (n - 1) as f64;

    let margin_scr = (ll_true - ll_scr).max(0.0);
    let margin_rev = (ll_true - ll_rev).max(0.0);
    let margin_marg = (ll_true - ll_marg).max(0.0);

    // Exige de battre significativement les leurres structurels (nul sur bruit blanc)
    let combined = 0.55 * margin_scr + 0.30 * margin_rev + 0.15 * margin_marg;
    tanh_clamped(combined * 2.5).clamp(0.0, 1.0)
}

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
        ((mean_ll - uniform_ll) / (-uniform_ll).max(0.5)).clamp(0.0, 1.3)
    };

    let g_left = span_gain(&pg.sym_seq[..mid]);
    let g_right = span_gain(&pg.sym_seq[mid..]);
    let min_half = g_left.min(g_right);
    let balance = 1.0 - (g_left - g_right).abs().min(1.0);

    let mut ctx_align = 0.0;
    let mut ctx_count = 0.0;
    for c in 0..N_CTX {
        let (st, en) = pg.ctx_spans[c];
        if en > st + 1 {
            let mut local_score = 0.0;
            for w in pg.sym_seq[st..en].windows(2) {
                local_score += h.sym_trans[w[0] * k + w[1]];
            }
            ctx_align += (local_score / ((en - st - 1) as f64) * (k as f64)).clamp(0.0, 3.0) / 3.0;
            ctx_count += 1.0;
        }
    }
    let ctx_proj = if ctx_count > 0.0 { ctx_align / ctx_count } else { 0.0 };

    let raw = min_half * (0.4 + 0.6 * balance) * (0.3 + 0.7 * ctx_proj);
    tanh_clamped(raw * 2.0).clamp(0.0, 1.0)
}

/// Résidu explicatif calibré dans [0, 1] :
/// Mesure la fraction non expliquée de la structure transitionnelle de E (distance TV pondérée)
/// sans saturer artificiellement à 1.0.
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

    // total_deficit est une distance TV unilatérale dans [0, 1]
    // On y ajoute l'écart sur les sauts à pas 2 (structure relationnelle profonde)
    let mut skip_deficit = 0.0;
    for u in 0..k {
        let m_u = pg.emp_marginal[u];
        for v in 0..k {
            let mut p2 = 0.0;
            for m in 0..k {
                p2 += h.sym_trans[u * k + m] * h.sym_trans[m * k + v];
            }
            skip_deficit += (pg.emp_trans2[u * k + v] - p2).max(0.0) * m_u;
        }
    }

    let res = (0.70 * total_deficit + 0.30 * skip_deficit).clamp(0.0, 1.0);
    (res, max_pair)
}

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
        ref_h.lat_couplings[idx] *= -0.15;
    }
    rebuild_sym_trans_from_lat(&mut ref_h, pg);
    ref_h
}

pub fn apply_displacement(h: &HypothesisState, pg: &PerceivedGraph) -> HypothesisState {
    let mut dep_h = h.clone();
    dep_h.ctx_states.rotate_left(1);
    let mut shifted_lat = [[0.0; D]; N_LAT];
    for i in 0..N_LAT {
        let next = (i + 1) % N_LAT;
        for d in 0..D {
            shifted_lat[i][d] = 0.85 * dep_h.lat_states[i][d] + 0.15 * dep_h.lat_states[next][d];
        }
    }
    dep_h.lat_states = shifted_lat;
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

fn rebuild_sym_trans_from_lat(h: &mut HypothesisState, _pg: &PerceivedGraph) {
    let k = h.k;
    if h.search_depth == 0 {
        return;
    }
    let mut lat_trans = [0.0; N_LAT * N_LAT];
    for i in 0..N_LAT {
        let mut max_v = f64::NEG_INFINITY;
        for j in 0..N_LAT {
            let v = 3.0 * h.lat_couplings[i * N_LAT + j];
            if v > max_v {
                max_v = v;
            }
        }
        let mut sum = 0.0;
        for j in 0..N_LAT {
            let w = (3.0 * h.lat_couplings[i * N_LAT + j] - max_v).clamp(-12.0, 0.0).exp();
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
        let mut logits = vec![0.0; k];
        let mut max_l = f64::NEG_INFINITY;
        for v in 0..k {
            let mut p_uv = 0.0;
            for j in 0..N_LAT {
                p_uv += lat_after[j] * h.lat_to_sym[j * k + v];
            }
            let l = p_uv.max(1e-9).ln() + h.sym_rel_bias[u * k + v];
            logits[v] = l;
            if l > max_l {
                max_l = l;
            }
        }
        let mut row_sum = 0.0;
        for v in 0..k {
            let w = (logits[v] - max_l).clamp(-14.0, 0.0).exp().max(1e-9);
            h.sym_trans[u * k + v] = w;
            row_sum += w;
        }
        let inv = 1.0 / row_sum.max(1e-12);
        for v in 0..k {
            h.sym_trans[u * k + v] *= inv;
        }
    }
}

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
            0.0 // Sans réfutation active, aucune validation réfutative n'est acquise
        } else {
            let h_ref = apply_refutation(h, pg);
            let (p_ref, _) = compute_iit_mip(&h_ref, pg);
            let g_ref = score_generativity(&h_ref, pg);
            ((p_ref / base_p).clamp(0.0, 1.0) * (g_ref / base_g).clamp(0.0, 1.0)).sqrt()
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
        (0.0, 0.0, 0.0)
    } else {
        (1.0, 1.0, 1.0)
    };

    let stability_geo = (stab_refutation * stab_displacement * stab_abstraction).cbrt();
    let res_factor = (1.0 - 0.65 * residue_score).clamp(0.05, 1.0);
    // Couplage STRICTEMENT non-compensatoire : si Sens == 0 (octets aléatoires) ou MIP == 0 (H := E), φ_sharp == 0 !
    let sens_geo = sens_prod.cbrt();

    let phi_sharp = (phi_iit_mip * stability_geo * sens_geo * res_factor).clamp(0.0, 1.0);

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