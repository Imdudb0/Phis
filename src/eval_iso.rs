use std::collections::HashSet;

use crate::autograd::Tape;
use crate::core::*;
use crate::model::*;

fn remap_into_range(bytes: &[u8], low: u8, high_incl: u8, rng: &mut Rng) -> Vec<u8> {
    let mut pool: Vec<u8> = (low..=high_incl).collect();
    rng.shuffle(&mut pool);
    let mut table = [u8::MAX; 256];
    let mut used = [false; 256];
    let mut slot = 0usize;
    let mut out = Vec::with_capacity(bytes.len());
    for &b in bytes {
        let idx = b as usize;
        if !used[idx] {
            used[idx] = true;
            table[idx] = pool[slot];
            slot += 1;
        }
        out.push(table[idx]);
    }
    out
}

fn assert_disjoint_bytes(a: &[u8], b: &[u8]) {
    let set_a: HashSet<u8> = a.iter().copied().collect();
    for &x in b {
        assert!(!set_a.contains(&x), "Octet commun interdit entre requête et candidat");
    }
}

fn dot_arr(a: &[f64; DIM], b: &[f64; DIM]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn cosine_slice(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (mut num, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..n {
        num += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let den = (na * nb).sqrt();
    if den < 1e-12_f64 { 0.0_f64 } else { (num / den).clamp(-1.0_f64, 1.0_f64) }
}

fn h2_perm_cosine(a: &[[f64; DIM]], b: &[[f64; DIM]]) -> f64 {
    if a.len() != K2 || b.len() != K2 {
        return 0.0_f64;
    }
    let na: f64 = a.iter().map(|v| dot_arr(v, v)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|v| dot_arr(v, v)).sum::<f64>().sqrt();
    if na < 1e-7_f64 || nb < 1e-7_f64 {
        return 0.0_f64;
    }
    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0],
    ];
    let mut best = f64::NEG_INFINITY;
    for p in PERMS {
        let mut num = 0.0_f64;
        for m in 0..K2 {
            num += dot_arr(&a[m], &b[p[m]]);
        }
        best = best.max(num / (na * nb));
    }
    best.clamp(-1.0_f64, 1.0_f64)
}

fn gram_signature(raw: &RawState) -> Vec<f64> {
    let mut g1 = Vec::new();
    for i in 0..raw.h1.len() {
        for j in (i + 1)..raw.h1.len() {
            g1.push(dot_arr(&raw.h1[i], &raw.h1[j]));
        }
    }
    g1.sort_by(|a, b| a.total_cmp(b));
    let mut g2 = Vec::new();
    for i in 0..raw.h2.len() {
        for j in (i + 1)..raw.h2.len() {
            g2.push(dot_arr(&raw.h2[i], &raw.h2[j]));
        }
    }
    g2.sort_by(|a, b| a.total_cmp(b));
    g1.extend(g2);
    g1
}

/// Baseline de surface (sans modèle) : P_marg trié + entropies H1, H2, H3 triées.
fn surface_signature(seq: &[usize], v: usize, p_marg: &[f64]) -> Vec<f64> {
    let mut pm = p_marg.to_vec();
    pm.sort_by(|a, b| a.total_cmp(b));
    let ln_v = (v.max(2) as f64).ln();
    let mut sig = pm;
    for lag in [1usize, 2, 3] {
        let mut cnt = vec![0.0_f64; v * v];
        let mut row = vec![0.0_f64; v];
        if seq.len() > lag {
            for t in 0..(seq.len() - lag) {
                cnt[seq[t] * v + seq[t + lag]] += 1.0_f64;
                row[seq[t]] += 1.0_f64;
            }
        }
        let mut hs = vec![0.0_f64; v];
        for u in 0..v {
            if row[u] > 0.0_f64 {
                let mut h = 0.0_f64;
                for w in 0..v {
                    let p = cnt[u * v + w] / row[u];
                    if p > 1e-12_f64 {
                        h -= p * p.ln();
                    }
                }
                hs[u] = h / ln_v;
            }
        }
        hs.sort_by(|a, b| a.total_cmp(b));
        sig.extend(hs);
    }
    sig
}

fn tie_aware_metrics(scores: &[f64], pos_idx: usize) -> (f64, f64) {
    let tgt = scores[pos_idx];
    let (mut better, mut tied) = (0usize, 0usize);
    for (i, &s) in scores.iter().enumerate() {
        if i == pos_idx {
            continue;
        }
        if s > tgt + 1e-10_f64 {
            better += 1;
        } else if (s - tgt).abs() <= 1e-10_f64 {
            tied += 1;
        }
    }
    let top1 = if better == 0 { 1.0_f64 / ((tied + 1) as f64) } else { 0.0_f64 };
    let mut mrr = 0.0_f64;
    for off in 0..=tied {
        mrr += 1.0_f64 / ((better + 1 + off) as f64);
    }
    (top1, mrr / ((tied + 1) as f64))
}

// ============================================================================
// 4 graphes 2-réguliers sur V = 8 + marches EULÉRIENNES exactement équilibrées
//   Chaque sommet est visité exactement 2m fois et chaque arête exactement m fois.
// ============================================================================

#[derive(Clone, Copy)]
enum IsoTopo {
    ChiralCycle12,   // Z_8, (+1, +2)
    SymmetricRing,   // Z_8, (+1, -1)
    TwinBasinPrism,  // Z_2 x Z_4
    BipartiteChiral, // Z_8, (+1, +3)
}

impl IsoTopo {
    fn neighbors(self, u: u8) -> [u8; 2] {
        match self {
            Self::ChiralCycle12 => [(u + 1) % 8, (u + 2) % 8],
            Self::SymmetricRing => [(u + 1) % 8, (u + 7) % 8],
            Self::TwinBasinPrism => {
                let (b, k) = (u / 4, u % 4);
                [b * 4 + ((k + 1) % 4), (1 - b) * 4 + k]
            }
            Self::BipartiteChiral => [(u + 1) % 8, (u + 3) % 8],
        }
    }
}

/// Circuit eulérien aléatoire (Hierholzer) : P_marg == 1/8 et H1 == ln 2 à 1e-16 près.
fn emit_eulerian_walk(topo: IsoTopo, rng: &mut Rng, mult: usize) -> Vec<u8> {
    let mut adj: Vec<Vec<u8>> = Vec::with_capacity(8);
    for u in 0u8..8 {
        let nb = topo.neighbors(u);
        let mut edges = Vec::with_capacity(2 * mult);
        for _ in 0..mult {
            edges.push(nb[0]);
            edges.push(nb[1]);
        }
        rng.shuffle(&mut edges);
        adj.push(edges);
    }
    let start = (rng.next_u64() % 8) as u8;
    let mut stack = vec![start];
    let mut circuit = Vec::with_capacity(16 * mult + 1);
    while let Some(&u) = stack.last() {
        if let Some(nxt) = adj[u as usize].pop() {
            stack.push(nxt);
        } else {
            circuit.push(stack.pop().unwrap());
        }
    }
    circuit.reverse();
    circuit
}

struct IsoTrial {
    query: Vec<u8>,
    cands: Vec<Vec<u8>>,
    pos_idx: usize,
}

#[derive(Default, Clone, Copy)]
struct IsoMetrics {
    h2_t1: f64,
    h2_mrr: f64,
    gram_t1: f64,
    gram_mrr: f64,
    joint_t1: f64,
    joint_mrr: f64,
    h2_norm: f64,
}

fn build_eulerian_trials(seed: u64, per_topo: usize) -> Vec<IsoTrial> {
    let mut rng = Rng::new(seed);
    let topos = [
        IsoTopo::ChiralCycle12,
        IsoTopo::SymmetricRing,
        IsoTopo::TwinBasinPrism,
        IsoTopo::BipartiteChiral,
    ];
    let mut trials = Vec::with_capacity(4 * per_topo);
    for (ti, &tq) in topos.iter().enumerate() {
        for _ in 0..per_topo {
            let query = remap_into_range(&emit_eulerian_walk(tq, &mut rng, 20), 0, 127, &mut rng);
            let mut tagged = Vec::with_capacity(4);
            for (ci, &tc) in topos.iter().enumerate() {
                let c = remap_into_range(&emit_eulerian_walk(tc, &mut rng, 20), 128, 255, &mut rng);
                assert_disjoint_bytes(&query, &c);
                tagged.push((ci == ti, c));
            }
            rng.shuffle(&mut tagged);
            let pos_idx = tagged.iter().position(|(is_p, _)| *is_p).unwrap();
            let cands = tagged.into_iter().map(|(_, b)| b).collect();
            trials.push(IsoTrial { query, cands, pos_idx });
        }
    }
    trials
}

fn eval_surface(trials: &[IsoTrial]) -> (f64, f64) {
    let (mut t1, mut mrr) = (0.0_f64, 0.0_f64);
    for tr in trials {
        let vq = RelationalView::from_bytes(&tr.query);
        let sq = surface_signature(&vq.seq, vq.v_size, &vq.p_marg);
        let scores: Vec<f64> = tr
            .cands
            .iter()
            .map(|c| {
                let vc = RelationalView::from_bytes(c);
                cosine_slice(&sq, &surface_signature(&vc.seq, vc.v_size, &vc.p_marg))
            })
            .collect();
        let (a, b) = tie_aware_metrics(&scores, tr.pos_idx);
        t1 += a;
        mrr += b;
    }
    let n = trials.len().max(1) as f64;
    (t1 / n, mrr / n)
}

fn eval_model_iso(model: &PhisharpV4, trials: &[IsoTrial]) -> IsoMetrics {
    let mut m = IsoMetrics::default();
    let n = trials.len().max(1) as f64;

    for tr in trials {
        let vq = RelationalView::from_bytes(&tr.query);
        let (rq, _) = model.run_abduction_raw(&vq, DEFAULT_BUDGET, true, &[]);
        let (_, _, _, gq) = model.unfold_predictions_raw(&vq, &rq, false);
        let gq_sig = gram_signature(&rq);
        m.h2_norm += rq.h2.iter().map(|v| dot_arr(v, v).sqrt()).sum::<f64>() / (K2 as f64);

        let (mut sc_h2, mut sc_g, mut sc_j) = (Vec::new(), Vec::new(), Vec::new());
        for cand in &tr.cands {
            let vc = RelationalView::from_bytes(cand);
            let (rc, _) = model.run_abduction_raw(&vc, DEFAULT_BUDGET, true, &[]);
            let (_, _, _, gc) = model.unfold_predictions_raw(&vc, &rc, false);
            sc_h2.push(h2_perm_cosine(&rq.h2, &rc.h2));
            sc_g.push(cosine_slice(&gq_sig, &gram_signature(&rc)));

            let mut jb = Vec::with_capacity(tr.query.len() + cand.len());
            jb.extend_from_slice(&tr.query);
            jb.extend_from_slice(cand);
            let vj = RelationalView::from_bytes(&jb);
            let (rj, _) = model.run_abduction_raw(&vj, DEFAULT_BUDGET, true, &[]);
            let (_, _, _, gj) = model.unfold_predictions_raw(&vj, &rj, false);
            sc_j.push(gj - 0.5_f64 * (gq + gc));
        }

        let (a1, b1) = tie_aware_metrics(&sc_h2, tr.pos_idx);
        let (a2, b2) = tie_aware_metrics(&sc_g, tr.pos_idx);
        let (a3, b3) = tie_aware_metrics(&sc_j, tr.pos_idx);
        m.h2_t1 += a1;
        m.h2_mrr += b1;
        m.gram_t1 += a2;
        m.gram_mrr += b2;
        m.joint_t1 += a3;
        m.joint_mrr += b3;
    }

    m.h2_t1 /= n;
    m.h2_mrr /= n;
    m.gram_t1 /= n;
    m.gram_mrr /= n;
    m.joint_t1 /= n;
    m.joint_mrr /= n;
    m.h2_norm /= n;
    m
}

// ============================================================================
// Épreuve 2 : Résistance aux perturbations & test anti-gel strict de H2
// ============================================================================

#[derive(Default, Clone, Copy)]
struct PerturbMetrics {
    rec_noise: [f64; 4],
    rec_drop_l1: f64,
    rec_cut_l2: f64,
    s_ref: f64,
    s_abs: f64,
    rec_rot: f64,
    rec_blk: f64,
    h2_rot: f64,
    h2_blk: f64,
    h2_stab20: f64,
}

fn corrupt_same_vocab(bytes: &[u8], rate: f64, rng: &mut Rng) -> Vec<u8> {
    let mut vocab: Vec<u8> = bytes.iter().copied().collect::<HashSet<u8>>().into_iter().collect();
    vocab.sort_unstable();
    let mut out = bytes.to_vec();
    if vocab.len() <= 1 || rate <= 0.0_f64 {
        return out;
    }
    for b in &mut out {
        if rng.next_f64() < rate {
            let mut c = vocab[rng.range_usize(vocab.len())];
            if c == *b {
                c = vocab[(vocab.iter().position(|&x| x == *b).unwrap_or(0) + 1) % vocab.len()];
            }
            *b = c;
        }
    }
    out
}

fn block_shuffle(bytes: &[u8], blk: usize, rng: &mut Rng) -> Vec<u8> {
    let mut chunks: Vec<&[u8]> = bytes.chunks(blk).collect();
    rng.shuffle(&mut chunks);
    let mut out = Vec::with_capacity(bytes.len());
    for c in chunks {
        out.extend_from_slice(c);
    }
    out
}

fn eval_perturb(model: &PhisharpV4, seed: u64) -> PerturbMetrics {
    let mut rng = Rng::new(seed);
    let mut truth_rng = Rng::new(seed ^ 0x3141_5926);
    let trials = 14usize;
    let rates = [0.0_f64, 0.10_f64, 0.20_f64, 0.30_f64];
    let mut m = PerturbMetrics::default();

    for _ in 0..trials {
        let cause = PtgrsCause::sample_ood(&mut rng, 4);
        let clean = cause.emit_effect(&mut rng, 4, 180);
        let truth = cause.ground_truth_non_markov(&mut truth_rng, 4);

        let vc = RelationalView::from_bytes(&clean);
        let (rc, snaps) = model.run_abduction_raw(&vc, DEFAULT_BUDGET, true, &[3]);
        let (pc, _, _, _) = model.unfold_predictions_raw(&vc, &rc, false);
        let rep = model.evaluate_phi_raw(&vc, &rc, DEFAULT_BUDGET, true);

        m.rec_noise[0] += observational_equivalence_nm(&vc, &pc, &truth);
        m.s_ref += rep.s_ref;
        m.s_abs += rep.s_abs;

        for (idx, &rate) in rates.iter().enumerate().skip(1) {
            let noisy = corrupt_same_vocab(&clean, rate, &mut rng);
            let vn = RelationalView::from_bytes(&noisy);
            let (rn, _) = model.run_abduction_raw(&vn, DEFAULT_BUDGET, true, &[]);
            let (pn, _, _, _) = model.unfold_predictions_raw(&vn, &rn, false);
            m.rec_noise[idx] += observational_equivalence_nm(&vn, &pn, &truth);
            if idx == 2 {
                m.h2_stab20 += h2_perm_cosine(&rc.h2, &rn.h2);
            }
        }

        // DropL1 à t=3 -> réparation jusqu'à t=6
        let mut st_d1 = snaps[0].1.clone();
        let pk = st_d1
            .h1
            .iter()
            .enumerate()
            .max_by(|a, b| dot_arr(a.1, a.1).total_cmp(&dot_arr(b.1, b.1)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        st_d1.h1[pk] = [0.0_f64; DIM];
        {
            let mut t = Tape::new(false);
            let p = ParamsTape::from_model(&mut t, model);
            let mut cur = st_d1.into_tape(&mut t);
            for _ in 3..DEFAULT_BUDGET {
                cur = PhisharpV4::abductive_step(
                    &mut t, &p, &vc, &vc.obs, &cur, true, [true, true, true], None, None, false,
                );
            }
            let r_d1 = RawState::from_tape(&t, &cur);
            let (pd1, _, _, _) = model.unfold_predictions_raw(&vc, &r_d1, false);
            m.rec_drop_l1 += observational_equivalence_nm(&vc, &pd1, &truth);
        }

        // CutL2 à t=3 -> t=6
        let mut st_c2 = snaps[0].1.clone();
        for h in &mut st_c2.h2 {
            *h = [0.0_f64; DIM];
        }
        {
            let mut t = Tape::new(false);
            let p = ParamsTape::from_model(&mut t, model);
            let mut cur = st_c2.into_tape(&mut t);
            for _ in 3..DEFAULT_BUDGET {
                cur = PhisharpV4::abductive_step(
                    &mut t, &p, &vc, &vc.obs, &cur, true, [true, true, true], None, None, true,
                );
            }
            let r_c2 = RawState::from_tape(&t, &cur);
            let (pc2, _, _, _) = model.unfold_predictions_raw(&vc, &r_c2, true);
            m.rec_cut_l2 += observational_equivalence_nm(&vc, &pc2, &truth);
        }

        // Sensibilité : rotation + bijection (préserve la cause) vs blocs de 4 (brise la cause)
        let mut rot = Vec::with_capacity(clean.len());
        rot.extend_from_slice(&clean[45..]);
        rot.extend_from_slice(&clean[..45]);
        let mut bij: Vec<u8> = (0u8..=255).collect();
        rng.shuffle(&mut bij);
        let rot_bij: Vec<u8> = rot.iter().map(|&b| bij[b as usize]).collect();

        let vr = RelationalView::from_bytes(&rot);
        let (rr, _) = model.run_abduction_raw(&vr, DEFAULT_BUDGET, true, &[]);
        let (pr, _, _, _) = model.unfold_predictions_raw(&vr, &rr, false);
        m.rec_rot += observational_equivalence_nm(&vr, &pr, &truth);

        let vrb = RelationalView::from_bytes(&rot_bij);
        let (rrb, _) = model.run_abduction_raw(&vrb, DEFAULT_BUDGET, true, &[]);
        m.h2_rot += h2_perm_cosine(&rc.h2, &rrb.h2);

        let blk = block_shuffle(&clean, 4, &mut rng);
        let vb = RelationalView::from_bytes(&blk);
        let (rb, _) = model.run_abduction_raw(&vb, DEFAULT_BUDGET, true, &[]);
        let (pb, _, _, _) = model.unfold_predictions_raw(&vb, &rb, false);
        m.rec_blk += observational_equivalence_nm(&vb, &pb, &truth);
        m.h2_blk += h2_perm_cosine(&rc.h2, &rb.h2);
    }

    let n = trials as f64;
    for r in &mut m.rec_noise {
        *r /= n;
    }
    m.rec_drop_l1 /= n;
    m.rec_cut_l2 /= n;
    m.s_ref /= n;
    m.s_abs /= n;
    m.rec_rot /= n;
    m.rec_blk /= n;
    m.h2_rot /= n;
    m.h2_blk /= n;
    m.h2_stab20 /= n;
    m
}

pub(crate) fn run_eval_iso(models: &[(&str, &PhisharpV4)], seed: u64) {
    println!();
    println!("================================================================================================");
    println!("=== BANC V4 : ISOMORPHISME EULÉRIEN STRICT (P_marg=1/8, H1=ln2) & PERTURBATIONS              ===");
    println!("================================================================================================");

    let trials = build_eulerian_trials(seed, 10); // 40 essais, 4 choix
    let (surf_t1, surf_mrr) = eval_surface(&trials);
    let clean_tag = if surf_t1 <= 0.35_f64 {
        "PROPRE : zéro fuite de surface"
    } else {
        "ALERTE : fuite résiduelle"
    };

    println!("\n[1/2] ISOMORPHISME MÉTA-RELATIONNEL PUR (40 essais eulériens, 0 octet commun vérifié)");
    println!(
        "      Hasard : 25.0% / 0.521 | Baseline Surface : {:5.1}% / {:.3} [{}]",
        surf_t1 * 100.0_f64, surf_mrr, clean_tag
    );
    println!(
        "{:<14} | {:>15} | {:>15} | {:>15} | {:>8}",
        "Modèle", "H2 seul (T1/MRR)", "Gram(H1,H2)", "Joint (T1/MRR)", "||H2||"
    );
    println!("{}", "-".repeat(78));

    let mut iso_res = Vec::with_capacity(models.len());
    for (label, model) in models {
        let m = eval_model_iso(model, &trials);
        println!(
            "{:<14} | {:>5.1}% / {:>5.3}  | {:>5.1}% / {:>5.3}  | {:>5.1}% / {:>5.3}  | {:>8.4}",
            label,
            m.h2_t1 * 100.0_f64, m.h2_mrr,
            m.gram_t1 * 100.0_f64, m.gram_mrr,
            m.joint_t1 * 100.0_f64, m.joint_mrr,
            m.h2_norm
        );
        iso_res.push(m);
    }

    println!("\n[2/2] RÉSISTANCE AUX PERTURBATIONS & TEST ANTI-GEL H2 (PTGRS OOD d=4, 14 causes)");
    println!(
        "{:<14} | {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} | {:>5} {:>5} | {:>11}",
        "Modèle", "Rec0%", "Rec10%", "Rec20%", "Rec30%", "DropL1", "CutL2", "S_ref", "S_abs", "ΔcosH2(r-b)"
    );
    println!("{}", "-".repeat(102));

    let mut pert_res = Vec::with_capacity(models.len());
    for (label, model) in models {
        let p = eval_perturb(model, seed ^ 0x7777_8888);
        println!(
            "{:<14} | {:>+7.4} {:>+7.4} {:>+7.4} {:>+7.4} | {:>+7.4} {:>+7.4} | {:>5.3} {:>5.3} | {:>+11.4}",
            label,
            p.rec_noise[0], p.rec_noise[1], p.rec_noise[2], p.rec_noise[3],
            p.rec_drop_l1, p.rec_cut_l2, p.s_ref, p.s_abs, p.h2_rot - p.h2_blk
        );
        pert_res.push(p);
    }

    println!("  Détail anti-gel H2 :");
    for (i, (label, _)) in models.iter().enumerate() {
        let p = &pert_res[i];
        println!(
            "    {:<14} : Rec(rot)={:+.4} vs Rec(blk4)={:+.4} | cosH2(rot)={:+.3} cosH2(blk4)={:+.3} | stab20%={:+.3}",
            label, p.rec_rot, p.rec_blk, p.h2_rot, p.h2_blk, p.h2_stab20
        );
    }

    if models.len() == 4 {
        let mdl_rob = 0.5_f64 * (pert_res[2].rec_noise[2] + pert_res[2].rec_noise[3]);
        let phi_rob = 0.5_f64 * (pert_res[3].rec_noise[2] + pert_res[3].rec_noise[3]);
        println!("\n--- VERDICT PRÉ-ENREGISTRÉ V4 : MDL vs PHISHARP V4 ---");
        println!(
            "  Isomorphisme pur H2 (Top1)   : Surface={:4.1}% | Step0={:4.1}% | LL={:4.1}% | MDL={:4.1}% | V4 φ={:4.1}%",
            surf_t1 * 100.0_f64,
            iso_res[0].h2_t1 * 100.0_f64,
            iso_res[1].h2_t1 * 100.0_f64,
            iso_res[2].h2_t1 * 100.0_f64,
            iso_res[3].h2_t1 * 100.0_f64
        );
        println!(
            "  Robustesse bruit 20-30% (Rec): MDL={:+.4} vs V4 φ={:+.4} | DropL1 : MDL={:+.4} vs V4 φ={:+.4}",
            mdl_rob, phi_rob, pert_res[2].rec_drop_l1, pert_res[3].rec_drop_l1
        );
        println!(
            "  Anti-gel H2 Δcos(rot-blk4)   : MDL={:+.4} vs V4 φ={:+.4} (cible >= +0.1500)",
            pert_res[2].h2_rot - pert_res[2].h2_blk,
            pert_res[3].h2_rot - pert_res[3].h2_blk
        );
    }
}
