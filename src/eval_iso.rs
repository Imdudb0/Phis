use std::collections::HashSet;

use crate::core::*;
use crate::model::*;

// ============================================================================
// Utilitaires : bijection disjointe, signatures invariantes, rangs sans biais
// ============================================================================

fn remap_into_range(bytes: &[u8], low: u8, high_incl: u8, rng: &mut Rng) -> Vec<u8> {
    let mut pool: Vec<u8> = (low..=high_incl).collect();
    rng.shuffle(&mut pool);

    let mut table = [u8::MAX; 256];
    let mut used = [false; 256];
    let mut next_slot = 0usize;
    let mut out = Vec::with_capacity(bytes.len());

    for &b in bytes {
        let idx = b as usize;
        if !used[idx] {
            used[idx] = true;
            table[idx] = pool[next_slot];
            next_slot += 1;
        }
        out.push(table[idx]);
    }
    out
}

fn assert_disjoint_bytes(a: &[u8], b: &[u8]) {
    let set_a: HashSet<u8> = a.iter().copied().collect();
    for &x in b {
        assert!(
            !set_a.contains(&x),
            "Violation du protocole : octet commun détecté entre requête et candidat"
        );
    }
}

fn cosine_slice(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut num = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for i in 0..n {
        num += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let den = (na * nb).sqrt();
    if den < 1e-12_f64 {
        0.0_f64
    } else {
        (num / den).clamp(-1.0_f64, 1.0_f64)
    }
}

fn h2_perm_cosine(a: &[[f64; DIM]], b: &[[f64; DIM]]) -> f64 {
    if a.len() != K2 || b.len() != K2 {
        return 0.0_f64;
    }
    let na: f64 = a.iter().map(|v| dot(v, v)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|v| dot(v, v)).sum::<f64>().sqrt();
    if na < 1e-7_f64 || nb < 1e-7_f64 {
        return 0.0_f64;
    }

    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let mut best = f64::NEG_INFINITY;
    for p in PERMS {
        let mut num = 0.0_f64;
        for m in 0..K2 {
            num += dot(&a[m], &b[p[m]]);
        }
        let c = num / (na * nb);
        if c > best {
            best = c;
        }
    }
    best.clamp(-1.0_f64, 1.0_f64)
}

/// Signature interne invariante aux permutations de (H1, H2) et de l'alphabet.
fn circ_signature(state: &CircState, unfold: &CircUnfold) -> Vec<f64> {
    let k1 = state.h1.len();
    let k2 = state.h2.len();

    let mut h1_gram = Vec::with_capacity((k1 * (k1 - 1)) / 2);
    for i in 0..k1 {
        for j in (i + 1)..k1 {
            h1_gram.push(dot(&state.h1[i], &state.h1[j]));
        }
    }
    h1_gram.sort_by(|a, b| a.total_cmp(b));

    let mut h2_gram = Vec::with_capacity((k2 * (k2 - 1)) / 2);
    for i in 0..k2 {
        for j in (i + 1)..k2 {
            h2_gram.push(dot(&state.h2[i], &state.h2[j]));
        }
    }
    h2_gram.sort_by(|a, b| a.total_cmp(b));

    // Invariants de circulation de l'opérateur T : cosinus moyens <x_u, T^s(x_u)>
    let v = state.x.len().max(1);
    let mut self_ret = [0.0_f64; 3];
    for s in 0..3 {
        let mut acc = 0.0_f64;
        for u in 0..state.x.len() {
            let nx = vec_norm(&state.x[u]).max(1e-9_f64);
            let ny = vec_norm(&unfold.y[s][u]).max(1e-9_f64);
            acc += dot(&state.x[u], &unfold.y[s][u]) / (nx * ny);
        }
        self_ret[s] = acc / (v as f64);
    }

    let mut sig = Vec::with_capacity(h1_gram.len() + h2_gram.len() + 3);
    sig.extend(h1_gram);
    sig.extend(h2_gram);
    sig.extend(self_ret);
    sig
}

/// Baseline de surface stricte : P_marg trié + entropies H1, H2, H3 triées.
fn surface_signature(view: &RelationalView) -> Vec<f64> {
    let v = view.v_size;
    let mut pm = view.p_marg.clone();
    pm.sort_by(|a, b| a.total_cmp(b));

    let ln_v = (v.max(2) as f64).ln();
    let mut hs = [vec![0.0_f64; v], vec![0.0_f64; v], vec![0.0_f64; v]];
    for s in 0..3 {
        for u in 0..v {
            let mut h = 0.0_f64;
            for w in 0..v {
                let p = view.graph.r[s][u * v + w];
                if p > 1e-12_f64 {
                    h -= p * p.ln();
                }
            }
            hs[s][u] = h / ln_v;
        }
        hs[s].sort_by(|a, b| a.total_cmp(b));
    }

    let mut sig = Vec::with_capacity(4 * v);
    sig.extend(pm);
    for s in 0..3 {
        sig.extend(&hs[s]);
    }
    sig
}

fn tie_aware_rank_metrics(scores: &[f64], pos_idx: usize) -> (f64, f64) {
    let target = scores[pos_idx];
    let mut better = 0usize;
    let mut tied = 0usize;
    for (i, &s) in scores.iter().enumerate() {
        if i == pos_idx {
            continue;
        }
        if s > target + 1e-10_f64 {
            better += 1;
        } else if (s - target).abs() <= 1e-10_f64 {
            tied += 1;
        }
    }

    let top1 = if better == 0 {
        1.0_f64 / ((tied + 1) as f64)
    } else {
        0.0_f64
    };

    let mut mrr_sum = 0.0_f64;
    for offset in 0..=tied {
        mrr_sum += 1.0_f64 / ((better + 1 + offset) as f64);
    }
    (top1, mrr_sum / ((tied + 1) as f64))
}

// ============================================================================
// 4 graphes 2-réguliers strictement iso-marginaux et iso-entropiques sur V = 8
//   - Stationnaire uniforme 1/8 partout
//   - Lag-1 : {0.5, 0.5} -> H1 = ln 2 partout
//   - Lag-2 : {0.25, 0.50, 0.25} -> H2 = 1.5 ln 2 partout
//   - Lag-3 : {1/8, 3/8, 3/8, 1/8} -> H3 identique partout
// ============================================================================

#[derive(Clone, Copy)]
enum IsoTopo {
    ChiralCycle12,   // Z_8, générateurs (+1, +2) : non-biparti, chiral (0 arête symétrique)
    SymmetricRing,   // Z_8, générateurs (+1, -1) : biparti, 100 % symétrique/réversible
    TwinBasinPrism,  // Z_2 x Z_4, ((0,+1), (1,0)) : semi-symétrique (deux bassins couplés)
    BipartiteChiral, // Z_8, générateurs (+1, +3) : biparti, chiral (0 arête symétrique)
}

impl IsoTopo {
    fn neighbors(self, u: u8) -> [u8; 2] {
        match self {
            Self::ChiralCycle12 => [(u + 1) % 8, (u + 2) % 8],
            Self::SymmetricRing => [(u + 1) % 8, (u + 7) % 8],
            Self::TwinBasinPrism => {
                let b = u / 4;
                let k = u % 4;
                [b * 4 + ((k + 1) % 4), (1 - b) * 4 + k]
            }
            Self::BipartiteChiral => [(u + 1) % 8, (u + 3) % 8],
        }
    }
}

/// Génère une marche aléatoire équilibrée (sacs locaux 50/50 de taille 6 par sommet)
/// afin d'éliminer les fuites de variance empirique sur P_marg et H1..H3.
fn emit_iso_walk(topo: IsoTopo, rng: &mut Rng, len: usize) -> Vec<u8> {
    let mut bags: Vec<Vec<usize>> = vec![Vec::new(); 8];
    let mut cur = (rng.next_u64() % 8) as u8;
    let mut out = Vec::with_capacity(len);

    for _ in 0..len {
        out.push(cur);
        let u = cur as usize;
        if bags[u].is_empty() {
            let mut b = vec![0usize, 0, 0, 1, 1, 1];
            rng.shuffle(&mut b);
            bags[u] = b;
        }
        let choice = bags[u].pop().unwrap();
        cur = topo.neighbors(cur)[choice];
    }
    out
}

struct IsoTrial {
    query: Vec<u8>,
    cands: Vec<Vec<u8>>, // 4 candidats sur [128..=255] (0 octet commun avec query)
    pos_idx: usize,
}

#[derive(Default, Clone, Copy)]
struct IsoMetrics {
    h2_top1: f64,
    h2_mrr: f64,
    circ_top1: f64,
    circ_mrr: f64,
    joint_top1: f64,
    joint_mrr: f64,
    h2_norm: f64,
}

fn build_pure_iso_trials(seed: u64, trials_per_topo: usize) -> Vec<IsoTrial> {
    let mut rng = Rng::new(seed);
    let topos = [
        IsoTopo::ChiralCycle12,
        IsoTopo::SymmetricRing,
        IsoTopo::TwinBasinPrism,
        IsoTopo::BipartiteChiral,
    ];

    let mut trials = Vec::with_capacity(4 * trials_per_topo);
    for (t_idx, &topo_q) in topos.iter().enumerate() {
        for _ in 0..trials_per_topo {
            let q_raw = emit_iso_walk(topo_q, &mut rng, 320);
            let query = remap_into_range(&q_raw, 0, 127, &mut rng);

            let mut tagged: Vec<(bool, Vec<u8>)> = Vec::with_capacity(4);
            for (c_idx, &topo_c) in topos.iter().enumerate() {
                let c_raw = emit_iso_walk(topo_c, &mut rng, 320);
                let c_bytes = remap_into_range(&c_raw, 128, 255, &mut rng);
                assert_disjoint_bytes(&query, &c_bytes);
                tagged.push((c_idx == t_idx, c_bytes));
            }
            rng.shuffle(&mut tagged);
            let pos_idx = tagged.iter().position(|(is_pos, _)| *is_pos).unwrap();
            let cands = tagged.into_iter().map(|(_, b)| b).collect();
            trials.push(IsoTrial {
                query,
                cands,
                pos_idx,
            });
        }
    }
    trials
}

fn eval_surface_baseline(trials: &[IsoTrial]) -> (f64, f64) {
    let mut t1 = 0.0_f64;
    let mut mrr = 0.0_f64;
    for tr in trials {
        let vq = RelationalView::from_bytes(&tr.query);
        let sq = surface_signature(&vq);
        let mut scores = Vec::with_capacity(4);
        for c in &tr.cands {
            let vc = RelationalView::from_bytes(c);
            let sc = surface_signature(&vc);
            scores.push(cosine_slice(&sq, &sc));
        }
        let (a, b) = tie_aware_rank_metrics(&scores, tr.pos_idx);
        t1 += a;
        mrr += b;
    }
    let n = trials.len().max(1) as f64;
    (t1 / n, mrr / n)
}

fn eval_model_on_iso(model: &PhisharpV3, trials: &[IsoTrial]) -> IsoMetrics {
    let mut m = IsoMetrics::default();
    let n = trials.len().max(1) as f64;

    for tr in trials {
        let vq = RelationalView::from_bytes(&tr.query);
        let tq = model.run_abduction(&vq, DEFAULT_BUDGET, true, &[], false);
        let uq = model.unfold_top_down(&tq.state, None, false);
        let gq = multi_scale_gain(&vq.graph, &uq.r_hat, vq.v_size);
        let sig_q = circ_signature(&tq.state, &uq);

        m.h2_norm += tq
            .state
            .h2
            .iter()
            .map(|v| vec_norm(v))
            .sum::<f64>()
            / (K2 as f64);

        let mut sc_h2 = Vec::with_capacity(4);
        let mut sc_circ = Vec::with_capacity(4);
        let mut sc_joint = Vec::with_capacity(4);

        for cand in &tr.cands {
            let vc = RelationalView::from_bytes(cand);
            let tc = model.run_abduction(&vc, DEFAULT_BUDGET, true, &[], false);
            let uc = model.unfold_top_down(&tc.state, None, false);
            let gc = multi_scale_gain(&vc.graph, &uc.r_hat, vc.v_size);
            let sig_c = circ_signature(&tc.state, &uc);

            sc_h2.push(h2_perm_cosine(&tq.state.h2, &tc.state.h2));
            sc_circ.push(cosine_slice(&sig_q, &sig_c));

            let mut joint_bytes = Vec::with_capacity(tr.query.len() + cand.len());
            joint_bytes.extend_from_slice(&tr.query);
            joint_bytes.extend_from_slice(cand);
            let vj = RelationalView::from_bytes(&joint_bytes);
            let tj = model.run_abduction(&vj, DEFAULT_BUDGET, true, &[], false);
            let uj = model.unfold_top_down(&tj.state, None, false);
            let gj = multi_scale_gain(&vj.graph, &uj.r_hat, vj.v_size);
            sc_joint.push(gj - 0.5_f64 * (gq + gc));
        }

        let (a1, b1) = tie_aware_rank_metrics(&sc_h2, tr.pos_idx);
        let (a2, b2) = tie_aware_rank_metrics(&sc_circ, tr.pos_idx);
        let (a3, b3) = tie_aware_rank_metrics(&sc_joint, tr.pos_idx);
        m.h2_top1 += a1;
        m.h2_mrr += b1;
        m.circ_top1 += a2;
        m.circ_mrr += b2;
        m.joint_top1 += a3;
        m.joint_mrr += b3;
    }

    m.h2_top1 /= n;
    m.h2_mrr /= n;
    m.circ_top1 /= n;
    m.circ_mrr /= n;
    m.joint_top1 /= n;
    m.joint_mrr /= n;
    m.h2_norm /= n;
    m
}

// ============================================================================
// Épreuve 2 : Résistance aux perturbations & test anti-gel de H2 (PTGRS OOD)
// ============================================================================

#[derive(Default, Clone, Copy)]
struct PerturbMetrics {
    rec_noise: [f64; 4], // 0%, 10%, 20%, 30%
    rec_drop_l1: f64,    // Ablation L1 à t=3 -> réparation jusqu'à t=6
    rec_cut_l2: f64,     // Coupure L2 à t=3 -> jusqu'à t=6
    s_abs_clean: f64,    // Survie sous abstraction H2 -> H1
    s_ref_clean: f64,    // Survie sous attaque + réparation
    rec_rot: f64,
    rec_blk: f64,
    h2_cos_rot: f64,
    h2_cos_blk: f64,
    h2_stab_20: f64,
}

fn corrupt_bytes_same_vocab(bytes: &[u8], rate: f64, rng: &mut Rng) -> Vec<u8> {
    let mut vocab: Vec<u8> = bytes
        .iter()
        .copied()
        .collect::<HashSet<u8>>()
        .into_iter()
        .collect();
    vocab.sort_unstable();
    let mut out = bytes.to_vec();
    if vocab.len() <= 1 || rate <= 0.0_f64 {
        return out;
    }
    for b in &mut out {
        if rng.next_f64() < rate {
            let mut cand = vocab[rng.range_usize(vocab.len())];
            if cand == *b {
                cand = vocab[(vocab.iter().position(|&x| x == *b).unwrap_or(0) + 1) % vocab.len()];
            }
            *b = cand;
        }
    }
    out
}

fn block_shuffle(bytes: &[u8], block_len: usize, rng: &mut Rng) -> Vec<u8> {
    let mut blocks: Vec<&[u8]> = bytes.chunks(block_len).collect();
    rng.shuffle(&mut blocks);
    let mut out = Vec::with_capacity(bytes.len());
    for blk in blocks {
        out.extend_from_slice(blk);
    }
    out
}

fn eval_model_perturbations(model: &PhisharpV3, seed: u64) -> PerturbMetrics {
    let mut rng = Rng::new(seed);
    let mut truth_rng = Rng::new(seed ^ 0x3141_5926);
    let trials = 16usize;
    let noise_rates = [0.0_f64, 0.10_f64, 0.20_f64, 0.30_f64];

    let mut m = PerturbMetrics::default();

    for _ in 0..trials {
        let cause = PtgrsCause::sample_ood(&mut rng, 4);
        let clean = cause.emit_effect(&mut rng, 4, 180);
        let truth = cause.ground_truth_multi_scale(&mut truth_rng, 4);

        let v_clean = RelationalView::from_bytes(&clean);
        let tr_clean = model.run_abduction(&v_clean, DEFAULT_BUDGET, true, &[3], false);
        let unf_clean = model.unfold_top_down(&tr_clean.state, None, false);
        let rep_clean = model.evaluate_phi_state(&v_clean, &tr_clean.state, DEFAULT_BUDGET, true);

        m.rec_noise[0] += observational_equivalence_excess(&v_clean, &unf_clean.r_hat, &truth);
        m.s_abs_clean += rep_clean.s_abs;
        m.s_ref_clean += rep_clean.s_ref;

        for (idx, &rate) in noise_rates.iter().enumerate().skip(1) {
            let noisy = corrupt_bytes_same_vocab(&clean, rate, &mut rng);
            let vn = RelationalView::from_bytes(&noisy);
            let tr_n = model.run_abduction(&vn, DEFAULT_BUDGET, true, &[], false);
            let unf_n = model.unfold_top_down(&tr_n.state, None, false);
            m.rec_noise[idx] += observational_equivalence_excess(&vn, &unf_n.r_hat, &truth);
            if idx == 2 {
                m.h2_stab_20 += h2_perm_cosine(&tr_clean.state.h2, &tr_n.state.h2);
            }
        }

        let mid_state = &tr_clean.snapshots[0].1;

        // DropL1 : ablation du nœud L1 dominant à t=3, puis réparation active jusqu'à t=6
        let mut st_d1 = mid_state.clone();
        let peak1 = st_d1
            .h1
            .iter()
            .enumerate()
            .max_by(|a, b| dot(a.1, a.1).total_cmp(&dot(b.1, b.1)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        st_d1.h1[peak1] = [0.0_f64; DIM];
        for _ in 3..DEFAULT_BUDGET {
            st_d1 = model.abductive_step(
                &v_clean,
                &st_d1,
                true,
                [true, true, true],
                false,
                None,
                None,
                false,
            );
        }
        let unf_d1 = model.unfold_top_down(&st_d1, None, false);
        m.rec_drop_l1 += observational_equivalence_excess(&v_clean, &unf_d1.r_hat, &truth);

        // CutL2 : coupure verticale de L2 à t=3 -> t=6
        let mut st_c2 = mid_state.clone();
        for h in &mut st_c2.h2 {
            *h = [0.0_f64; DIM];
        }
        for _ in 3..DEFAULT_BUDGET {
            st_c2 = model.abductive_step(
                &v_clean,
                &st_c2,
                true,
                [true, true, true],
                false,
                None,
                None,
                true,
            );
        }
        let unf_c2 = model.unfold_top_down(&st_c2, None, true);
        m.rec_cut_l2 += observational_equivalence_excess(&v_clean, &unf_c2.r_hat, &truth);

        // Cause préservée (rotation + bijection) vs cause brisée (blocs de 4 mélangés)
        let shift = 45usize;
        let mut rot = Vec::with_capacity(clean.len());
        rot.extend_from_slice(&clean[shift..]);
        rot.extend_from_slice(&clean[..shift]);

        let mut bij: Vec<u8> = (0u8..=255).collect();
        rng.shuffle(&mut bij);
        let rot_bij: Vec<u8> = rot.iter().map(|&b| bij[b as usize]).collect();

        let v_rot_plain = RelationalView::from_bytes(&rot);
        let v_rot_bij = RelationalView::from_bytes(&rot_bij);

        let tr_rot_plain = model.run_abduction(&v_rot_plain, DEFAULT_BUDGET, true, &[], false);
        let unf_rot_plain = model.unfold_top_down(&tr_rot_plain.state, None, false);
        m.rec_rot += observational_equivalence_excess(&v_rot_plain, &unf_rot_plain.r_hat, &truth);

        let tr_rot_bij = model.run_abduction(&v_rot_bij, DEFAULT_BUDGET, true, &[], false);
        m.h2_cos_rot += h2_perm_cosine(&tr_clean.state.h2, &tr_rot_bij.state.h2);

        let blk = block_shuffle(&clean, 4, &mut rng);
        let v_blk = RelationalView::from_bytes(&blk);
        let tr_blk = model.run_abduction(&v_blk, DEFAULT_BUDGET, true, &[], false);
        let unf_blk = model.unfold_top_down(&tr_blk.state, None, false);
        m.rec_blk += observational_equivalence_excess(&v_blk, &unf_blk.r_hat, &truth);
        m.h2_cos_blk += h2_perm_cosine(&tr_clean.state.h2, &tr_blk.state.h2);
    }

    let n = trials as f64;
    for r in &mut m.rec_noise {
        *r /= n;
    }
    m.rec_drop_l1 /= n;
    m.rec_cut_l2 /= n;
    m.s_abs_clean /= n;
    m.s_ref_clean /= n;
    m.rec_rot /= n;
    m.rec_blk /= n;
    m.h2_cos_rot /= n;
    m.h2_cos_blk /= n;
    m.h2_stab_20 /= n;
    m
}

// ============================================================================
// Point d'entrée `eval-iso`
// ============================================================================

pub(crate) fn run_eval_iso(models: &[(&str, &PhisharpV3)], seed: u64) {
    println!();
    println!("================================================================================================");
    println!("=== BANC V3 : ISOMORPHISME STRICTEMENT ISO-MARGINAL/ISO-ENTROPIQUE & PERTURBATIONS           ===");
    println!("=== (4 graphes 2-réguliers V=8, H1=ln2, H2=1.5ln2, H3 égaux, 0 octet commun, seed={})       ===", seed);
    println!("================================================================================================");

    let trials = build_pure_iso_trials(seed, 12); // 48 essais (12 par topologie)
    let (surf_t1, surf_mrr) = eval_surface_baseline(&trials);
    let surface_clean = surf_t1 <= 0.40_f64;

    println!();
    println!(
        "[1/2] ISOMORPHISME MÉTA-RELATIONNEL PUR (48 essais, 4 choix, 0 octet commun vérifié)"
    );
    println!(
        "      Hasard pur : Top1=25.0% MRR=0.521 | Baseline Surface (P_marg,H1,H2,H3) : Top1={:5.1}% MRR={:.3} [{}]",
        surf_t1 * 100.0_f64,
        surf_mrr,
        if surface_clean { "PROPRE : zéro fuite de surface" } else { "ALERTE : fuite de surface" }
    );
    println!(
        "{:<16} | {:>15} | {:>15} | {:>15} | {:>8}",
        "Modèle", "H2 seul (T1/MRR)", "Circ(H1,H2,T)", "Joint (T1/MRR)", "||H2||"
    );
    println!("{}", "-".repeat(80));

    let mut iso_res = Vec::with_capacity(models.len());
    for (label, model) in models {
        let m = eval_model_on_iso(model, &trials);
        println!(
            "{:<16} | {:>5.1}% / {:>5.3}  | {:>5.1}% / {:>5.3}  | {:>5.1}% / {:>5.3}  | {:>8.4}",
            label,
            m.h2_top1 * 100.0_f64,
            m.h2_mrr,
            m.circ_top1 * 100.0_f64,
            m.circ_mrr,
            m.joint_top1 * 100.0_f64,
            m.joint_mrr,
            m.h2_norm
        );
        iso_res.push(m);
    }

    println!();
    println!("[2/2] RÉSISTANCE AUX PERTURBATIONS & TEST ANTI-GEL DE H2 (PTGRS OOD d=4, 16 causes)");
    println!(
        "{:<16} | {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} | {:>5} {:>5} | {:>11}",
        "Modèle", "Rec0%", "Rec10%", "Rec20%", "Rec30%", "DropL1", "CutL2", "S_ref", "S_abs", "ΔcosH2(r-b)"
    );
    println!("{}", "-".repeat(104));

    let mut pert_res = Vec::with_capacity(models.len());
    for (label, model) in models {
        let p = eval_model_perturbations(model, seed ^ 0x7777_8888);
        let d_cos = p.h2_cos_rot - p.h2_cos_blk;
        println!(
            "{:<16} | {:>+7.4} {:>+7.4} {:>+7.4} {:>+7.4} | {:>+7.4} {:>+7.4} | {:>5.3} {:>5.3} | {:>+11.4}",
            label,
            p.rec_noise[0],
            p.rec_noise[1],
            p.rec_noise[2],
            p.rec_noise[3],
            p.rec_drop_l1,
            p.rec_cut_l2,
            p.s_ref_clean,
            p.s_abs_clean,
            d_cos
        );
        pert_res.push(p);
    }

    println!("  Détail anti-gel H2 & sensibilité externe :");
    for (i, (label, _)) in models.iter().enumerate() {
        let p = &pert_res[i];
        println!(
            "    {:<16} : Rec(rot)={:+.4} vs Rec(blk4)={:+.4} | cosH2(rot)={:+.3} cosH2(blk4)={:+.3} | stabH2@20%={:+.3}",
            label, p.rec_rot, p.rec_blk, p.h2_cos_rot, p.h2_cos_blk, p.h2_stab_20
        );
    }

    if models.len() == 4 {
        let mdl_h2 = iso_res[2].h2_top1 * 100.0_f64;
        let phi_h2 = iso_res[3].h2_top1 * 100.0_f64;
        let mdl_rob = 0.5_f64 * (pert_res[2].rec_noise[2] + pert_res[2].rec_noise[3]);
        let phi_rob = 0.5_f64 * (pert_res[3].rec_noise[2] + pert_res[3].rec_noise[3]);
        let mdl_dcos = pert_res[2].h2_cos_rot - pert_res[2].h2_cos_blk;
        let phi_dcos = pert_res[3].h2_cos_rot - pert_res[3].h2_cos_blk;

        println!();
        println!("--- VERDICT PRÉ-ENREGISTRÉ V3 : MDL vs PHISHARP V3 ---");
        println!(
            "  Isomorphisme pur H2 (Top1)   : Surface={:4.1}% | Step0={:4.1}% | LL={:4.1}% | MDL={:4.1}% | V3 φ={:4.1}%",
            surf_t1 * 100.0_f64,
            iso_res[0].h2_top1 * 100.0_f64,
            iso_res[1].h2_top1 * 100.0_f64,
            mdl_h2,
            phi_h2
        );
        println!(
            "  Robustesse bruit 20-30% (Rec): MDL={:+.4} vs V3 φ={:+.4} | DropL1 : MDL={:+.4} vs V3 φ={:+.4}",
            mdl_rob, phi_rob, pert_res[2].rec_drop_l1, pert_res[3].rec_drop_l1
        );
        println!(
            "  Anti-gel H2 Δcos(rot-blk4)   : MDL={:+.4} vs V3 φ={:+.4} (cible >= +0.1500)",
            mdl_dcos, phi_dcos
        );
    }
}
