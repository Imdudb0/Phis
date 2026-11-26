use super::*;

// ============================================================================
// Utilitaires : bijection disjointe, similarités invariantes, rangs sans biais
// ============================================================================

/// Renomme bijectivement tous les octets uniques de `bytes` dans [low, high_incl].
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

/// Vérifie automatiquement que deux flux n'ont strictement aucun octet en commun.
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

/// Similarité cosinus de H2 invariante aux 3! = 6 permutations des nœuds méta-relationnels.
/// Retourne 0.0 si l'un des deux états H2 est éteint (norme quasi nulle).
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

/// Signature structurelle invariante à l'alphabet ET aux permutations de nœuds de (H1, G).
fn operator_signature(state: &HierState, unfold: &TopDownUnfold) -> Vec<f64> {
    let k1 = state.h1.len();
    let g1 = &unfold.g_op;
    let g2 = mat_mul(g1, g1, k1, k1, k1);
    let g3 = mat_mul(&g2, g1, k1, k1, k1);

    let mut tr1 = 0.0_f64;
    let mut tr2 = 0.0_f64;
    let mut tr3 = 0.0_f64;
    for i in 0..k1 {
        tr1 += g1[i * k1 + i];
        tr2 += g2[i * k1 + i];
        tr3 += g3[i * k1 + i];
    }

    // Asymétrie non-réciproque de G : ||G - G^T||_F
    let mut asym_sq = 0.0_f64;
    for i in 0..k1 {
        for j in 0..k1 {
            let d = g1[i * k1 + j] - g1[j * k1 + i];
            asym_sq += d * d;
        }
    }

    // Entropies triées des lignes de G et masses triées des colonnes de G
    let mut row_h = vec![0.0_f64; k1];
    let mut col_m = vec![0.0_f64; k1];
    let ln_k = (k1.max(2) as f64).ln();
    for i in 0..k1 {
        let mut h = 0.0_f64;
        for j in 0..k1 {
            let p = g1[i * k1 + j];
            col_m[j] += p;
            if p > 1e-12_f64 {
                h -= p * p.ln();
            }
        }
        row_h[i] = h / ln_k;
    }
    row_h.sort_by(|a, b| a.total_cmp(b));
    col_m.sort_by(|a, b| a.total_cmp(b));

    // Spectre de grammaire interne de H1 (produits scalaires triés, invariant aux permutations)
    let mut h1_pairs = Vec::with_capacity((k1 * (k1 - 1)) / 2);
    for i in 0..k1 {
        for j in (i + 1)..k1 {
            h1_pairs.push(dot(&state.h1[i], &state.h1[j]));
        }
    }
    h1_pairs.sort_by(|a, b| a.total_cmp(b));

    let mut sig = Vec::with_capacity(4 + 2 * k1 + h1_pairs.len());
    sig.push(tr1 / (k1 as f64));
    sig.push(tr2 / (k1 as f64));
    sig.push(tr3 / (k1 as f64));
    sig.push(asym_sq.sqrt() / (k1 as f64));
    sig.extend(row_h);
    sig.extend(col_m.into_iter().map(|c| c / (k1 as f64)));
    sig.extend(h1_pairs);
    sig
}

/// Baseline de surface (sans modèle) : fréquences marginales triées + entropies locales triées.
fn surface_signature(view: &RelationalView) -> Vec<f64> {
    let v = view.v_size;
    let mut pm = view.p_marg.clone();
    pm.sort_by(|a, b| a.total_cmp(b));

    let ln_v = (v.max(2) as f64).ln();
    let mut h1 = vec![0.0_f64; v];
    let mut h2 = vec![0.0_f64; v];
    for u in 0..v {
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        for w in 0..v {
            let p1 = view.graph.r[0][u * v + w];
            let p2 = view.graph.r[1][u * v + w];
            if p1 > 1e-12_f64 {
                s1 -= p1 * p1.ln();
            }
            if p2 > 1e-12_f64 {
                s2 -= p2 * p2.ln();
            }
        }
        h1[u] = s1 / ln_v;
        h2[u] = s2 / ln_v;
    }
    h1.sort_by(|a, b| a.total_cmp(b));
    h2.sort_by(|a, b| a.total_cmp(b));

    let mut sig = Vec::with_capacity(3 * v);
    sig.extend(pm);
    sig.extend(h1);
    sig.extend(h2);
    sig
}

/// Calcule (top1_espéré, mrr_espéré) avec gestion exacte des égalités :
/// si tous les scores sont identiques (ex. H2=0), renvoie exactement (0.25, 0.52083).
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
        let rank = (better + 1 + offset) as f64;
        mrr_sum += 1.0_f64 / rank;
    }
    let mrr = mrr_sum / ((tied + 1) as f64);
    (top1, mrr)
}

// ============================================================================
// Générateurs de moules topologiques à vocabulaire STRICTEMENT V = 8
// ============================================================================

fn gen_cycle_v8(rng: &mut Rng, len: usize) -> Vec<u8> {
    let mut cur = (rng.next_u64() % 8) as u8;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(cur);
        // Cycle périodique pur avec 8 % de saut de phase +2
        if rng.next_f64() < 0.08_f64 {
            cur = (cur + 2) % 8;
        } else {
            cur = (cur + 1) % 8;
        }
    }
    out
}

/// Moule topologique de l'ambiguïté : bifurcation 1 -> 2 reconvergeant sur un pivot.
fn gen_bifurcation_v8(rng: &mut Rng, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len + 8);
    let mut step = 0usize;
    while out.len() < len {
        out.push(0); // Pivot d'entrée
        // Force les deux branches au début pour garantir V = 8, puis tirage 50/50
        let branch_a = if step == 0 {
            true
        } else if step == 1 {
            false
        } else {
            rng.next_f64() < 0.5_f64
        };
        if branch_a {
            out.extend_from_slice(&[1, 2, 3]);
        } else {
            out.extend_from_slice(&[4, 5, 6]);
        }
        out.push(7); // Pivot de reconvergence
        step += 1;
    }
    out.truncate(len);
    out
}

/// Moule topologique de symétrie : blocs miroirs palindromiques a b c d d c b a.
fn gen_mirror_v8(rng: &mut Rng, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len + 8);
    let mut phase = (rng.next_u64() % 8) as u8;
    while out.len() < len {
        let a = phase % 8;
        let b = (phase + 1) % 8;
        let c = (phase + 2) % 8;
        let d = (phase + 3) % 8;
        out.extend_from_slice(&[a, b, c, d, d, c, b, a]);
        phase = (phase + if rng.next_f64() < 0.5_f64 { 4 } else { 2 }) % 8;
    }
    out.truncate(len);
    out
}

/// Moule arithmétique : Collatz mod 8 garanti V = 8.
fn gen_collatz_v8(rng: &mut Rng, len: usize) -> Vec<u8> {
    for _ in 0..20 {
        let seq = gen_collatz_trace(rng.next_u64(), len);
        let uniq: HashSet<u8> = seq.iter().copied().collect();
        if uniq.len() == 8 {
            return seq;
        }
    }
    gen_collatz_trace(27, len)
}

/// Moule protocolaire : trames [délimiteur=7, champ in {4,5,6}, val in {0..3}, checksum in {0..3}].
fn gen_frames_v8(rng: &mut Rng, len: usize) -> Vec<u8> {
    let mut counter = (rng.next_u64() % 12) as u8;
    let mut out = Vec::with_capacity(len + 4);
    while out.len() < len {
        let field = counter % 3;
        let val = (counter + field) % 4;
        let chk = (val + field + 1) % 4;
        out.push(7);
        out.push(4 + field);
        out.push(val);
        out.push(chk);
        counter = (counter + 1) % 12;
    }
    out.truncate(len);
    out
}

/// Moule bi-stable : deux cycles de taille 4 ({0..3} et {4..7}) couplés par ponts rares.
fn gen_two_basin_v8(rng: &mut Rng, len: usize) -> Vec<u8> {
    let mut basin = (rng.next_u64() & 1) as u8;
    let mut pos = (rng.next_u64() % 4) as u8;
    let mut out = Vec::with_capacity(len);
    for t in 0..len {
        out.push(basin * 4 + pos);
        if pos == 3 && (t == 8 || rng.next_f64() < 0.30_f64) {
            basin ^= 1;
            pos = 0;
        } else {
            pos = (pos + 1) % 4;
        }
    }
    out
}

/// Échantillonne une cause PTGRS OOD dont les flux réalisent exactement V = 8 symboles.
fn sample_ptgrs_ood_v8(rng: &mut Rng, depth: usize) -> PtgrsCause {
    let n_nonterms = 5usize;
    let n_terms = 8usize;
    let nonterms: Vec<u8> = (200u8..200u8 + (n_nonterms as u8)).collect();
    let terminals: Vec<u8> = (0u8..(n_terms as u8)).collect();

    let mut rules = Vec::with_capacity(n_nonterms);
    for (idx, &nt) in nonterms.iter().enumerate() {
        let n1 = nonterms[(idx + 1) % n_nonterms];
        let n2 = nonterms[(idx + 2) % n_nonterms];
        let t1 = terminals[(idx + rng.range_usize(n_terms)) % n_terms];
        let t2 = terminals[(idx + 2 + rng.range_usize(n_terms)) % n_terms];
        let t3 = terminals[(idx + 4 + rng.range_usize(n_terms)) % n_terms];
        let t4 = terminals[(idx + 6) % n_terms];

        let rhs_primary = match rng.range_usize(3) {
            0 => vec![t1, n1, t2, n2, t1],
            1 => vec![t1, t2, n1, t3, n2],
            _ => vec![n1, t1, t2, n2, t3],
        };
        let rhs_alt = vec![t3, t4, t2];
        let prob_primary = 0.70_f64 + 0.18_f64 * rng.next_f64();
        rules.push(RewriteRule {
            lhs: nt,
            rhs_primary,
            rhs_alt,
            prob_primary,
        });
    }

    let _ = depth;
    PtgrsCause {
        start: nonterms[0],
        rules,
        terminals,
    }
}

// ============================================================================
// Épreuve 1 : Isomorphisme méta-relationnel (alphabets strictement disjoints)
// ============================================================================

struct IsoTrial {
    query: Vec<u8>,
    cands: Vec<Vec<u8>>, // 4 candidats sur [128..=255] (0 octet commun avec query)
    pos_idx: usize,
}

#[derive(Default, Clone, Copy)]
struct IsoMetrics {
    h2_top1: f64,
    h2_mrr: f64,
    op_top1: f64,
    op_mrr: f64,
    joint_top1: f64,
    joint_mrr: f64,
    h2_norm: f64,
}

fn build_trials_topological(seed: u64, trials_per_family: usize) -> Vec<IsoTrial> {
    let mut rng = Rng::new(seed);
    let fams: [fn(&mut Rng, usize) -> Vec<u8>; 6] = [
        gen_cycle_v8,
        gen_bifurcation_v8,
        gen_mirror_v8,
        gen_collatz_v8,
        gen_frames_v8,
        gen_two_basin_v8,
    ];

    let mut trials = Vec::with_capacity(fams.len() * trials_per_family);
    for (f_idx, &gen_pos) in fams.iter().enumerate() {
        for _ in 0..trials_per_family {
            let q_raw = gen_pos(&mut rng, 180);
            let p_raw = gen_pos(&mut rng, 180); // Nouvelle instance stochastique

            let query = remap_into_range(&q_raw, 0, 127, &mut rng);
            let pos = remap_into_range(&p_raw, 128, 255, &mut rng);
            assert_disjoint_bytes(&query, &pos);

            // 3 distracteurs issus de 3 autres familles
            let mut others: Vec<usize> = (0..fams.len()).filter(|&i| i != f_idx).collect();
            rng.shuffle(&mut others);

            let mut tagged: Vec<(bool, Vec<u8>)> = Vec::with_capacity(4);
            tagged.push((true, pos));
            for &d_idx in others.iter().take(3) {
                let d_raw = fams[d_idx](&mut rng, 180);
                let d_bytes = remap_into_range(&d_raw, 128, 255, &mut rng);
                assert_disjoint_bytes(&query, &d_bytes);
                tagged.push((false, d_bytes));
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

fn build_trials_ptgrs_ood(seed: u64, n_trials: usize) -> Vec<IsoTrial> {
    let mut rng = Rng::new(seed);
    let mut trials = Vec::with_capacity(n_trials);

    for _ in 0..n_trials {
        let causes: Vec<PtgrsCause> = (0..4).map(|_| sample_ptgrs_ood_v8(&mut rng, 4)).collect();

        let q_raw = causes[0].emit_effect(&mut rng, 4, 180);
        let p_raw = causes[0].emit_effect(&mut rng, 4, 180);
        let query = remap_into_range(&q_raw, 0, 127, &mut rng);
        let pos = remap_into_range(&p_raw, 128, 255, &mut rng);
        assert_disjoint_bytes(&query, &pos);

        let mut tagged: Vec<(bool, Vec<u8>)> = Vec::with_capacity(4);
        tagged.push((true, pos));
        for c in &causes[1..4] {
            let d_raw = c.emit_effect(&mut rng, 4, 180);
            let d_bytes = remap_into_range(&d_raw, 128, 255, &mut rng);
            assert_disjoint_bytes(&query, &d_bytes);
            tagged.push((false, d_bytes));
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
    trials
}

fn eval_surface_baseline(trials: &[IsoTrial]) -> (f64, f64) {
    let mut top1_sum = 0.0_f64;
    let mut mrr_sum = 0.0_f64;

    for tr in trials {
        let vq = RelationalView::from_bytes(&tr.query);
        let sq = surface_signature(&vq);
        let mut scores = Vec::with_capacity(tr.cands.len());
        for c in &tr.cands {
            let vc = RelationalView::from_bytes(c);
            let sc = surface_signature(&vc);
            scores.push(cosine_slice(&sq, &sc));
        }
        let (t1, mrr) = tie_aware_rank_metrics(&scores, tr.pos_idx);
        top1_sum += t1;
        mrr_sum += mrr;
    }

    let n = trials.len().max(1) as f64;
    (top1_sum / n, mrr_sum / n)
}

fn eval_model_on_iso(model: &PhisharpModel, trials: &[IsoTrial]) -> IsoMetrics {
    let mut m = IsoMetrics::default();
    let n = trials.len().max(1) as f64;

    for tr in trials {
        let vq = RelationalView::from_bytes(&tr.query);
        let tq = model.run_abduction(&vq, DEFAULT_BUDGET, true, &[], false);
        let uq = model.unfold_top_down(&vq.node_init, &tq.state, None, false);
        let gq = multi_scale_gain(&vq.graph, &uq.r_hat, vq.v_size);
        let op_q = operator_signature(&tq.state, &uq);

        let h2_n: f64 = tq
            .state
            .h2
            .iter()
            .map(|x| vec_norm(x))
            .sum::<f64>()
            / (K2 as f64);
        m.h2_norm += h2_n;

        let mut scores_h2 = Vec::with_capacity(4);
        let mut scores_op = Vec::with_capacity(4);
        let mut scores_joint = Vec::with_capacity(4);

        for cand in &tr.cands {
            let vc = RelationalView::from_bytes(cand);
            let tc = model.run_abduction(&vc, DEFAULT_BUDGET, true, &[], false);
            let uc = model.unfold_top_down(&vc.node_init, &tc.state, None, false);
            let gc = multi_scale_gain(&vc.graph, &uc.r_hat, vc.v_size);
            let op_c = operator_signature(&tc.state, &uc);

            scores_h2.push(h2_perm_cosine(&tq.state.h2, &tc.state.h2));
            scores_op.push(cosine_slice(&op_q, &op_c));

            // Co-abduction jointe boîte noire sur E_q ⊕ E_cand (alphabets disjoints)
            let mut joint_bytes = Vec::with_capacity(tr.query.len() + cand.len());
            joint_bytes.extend_from_slice(&tr.query);
            joint_bytes.extend_from_slice(cand);
            let vj = RelationalView::from_bytes(&joint_bytes);
            let tj = model.run_abduction(&vj, DEFAULT_BUDGET, true, &[], false);
            let uj = model.unfold_top_down(&vj.node_init, &tj.state, None, false);
            let gj = multi_scale_gain(&vj.graph, &uj.r_hat, vj.v_size);
            // Synergie nette (corrige la facilité intrinsèque de chaque candidat)
            scores_joint.push(gj - 0.5_f64 * (gq + gc));
        }

        let (t1_h2, mrr_h2) = tie_aware_rank_metrics(&scores_h2, tr.pos_idx);
        let (t1_op, mrr_op) = tie_aware_rank_metrics(&scores_op, tr.pos_idx);
        let (t1_jt, mrr_jt) = tie_aware_rank_metrics(&scores_joint, tr.pos_idx);

        m.h2_top1 += t1_h2;
        m.h2_mrr += mrr_h2;
        m.op_top1 += t1_op;
        m.op_mrr += mrr_op;
        m.joint_top1 += t1_jt;
        m.joint_mrr += mrr_jt;
    }

    m.h2_top1 /= n;
    m.h2_mrr /= n;
    m.op_top1 /= n;
    m.op_mrr /= n;
    m.joint_top1 /= n;
    m.joint_mrr /= n;
    m.h2_norm /= n;
    m
}

// ============================================================================
// Épreuve 2 : Résistance aux perturbations structurelles & sensibilité
// ============================================================================

#[derive(Default, Clone, Copy)]
struct PerturbMetrics {
    rec_noise: [f64; 4], // 0%, 10%, 20%, 30%
    h2_stab_20: f64,     // Stabilité cos(H2(E), H2(E_20%))
    rec_drop_l1: f64,    // Rec_exc(C) après ablation d'un nœud L1 à t=3 -> t=6
    rec_cut_l2: f64,     // Rec_exc(C) après coupure de L2 à t=3 -> t=6
    rec_rot: f64,        // Cause préservée (rotation + bijection)
    rec_blk: f64,        // Cause brisée (permutation par blocs de 4)
    phi_rot: f64,
    phi_blk: f64,
    h2_cos_rot: f64,
    h2_cos_blk: f64,
}

fn corrupt_bytes_same_vocab(bytes: &[u8], rate: f64, rng: &mut Rng) -> Vec<u8> {
    let mut vocab: Vec<u8> = bytes.iter().copied().collect::<HashSet<u8>>().into_iter().collect();
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

fn eval_model_perturbations(model: &PhisharpModel, seed: u64) -> PerturbMetrics {
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
        let unf_clean = model.unfold_top_down(&v_clean.node_init, &tr_clean.state, None, false);
        m.rec_noise[0] += observational_equivalence_excess(&v_clean, &unf_clean.r_hat, &truth);

        // 1. Résilience au bruit d'observation (10%, 20%, 30%) mesurée face à C PROPRE
        for (idx, &rate) in noise_rates.iter().enumerate().skip(1) {
            let noisy = corrupt_bytes_same_vocab(&clean, rate, &mut rng);
            let vn = RelationalView::from_bytes(&noisy);
            let tr_n = model.run_abduction(&vn, DEFAULT_BUDGET, true, &[], false);
            let unf_n = model.unfold_top_down(&vn.node_init, &tr_n.state, None, false);
            m.rec_noise[idx] += observational_equivalence_excess(&vn, &unf_n.r_hat, &truth);
            if idx == 2 {
                m.h2_stab_20 += h2_perm_cosine(&tr_clean.state.h2, &tr_n.state.h2);
            }
        }

        // 2. Résilience interne à mi-trajectoire (t=3 -> t=6)
        let mid_state = &tr_clean.snapshots[0].1;

        // (a) Ablation du nœud L1 le plus actif à t=3, puis reprise de la recherche jusqu'à t=6
        let mut st_drop1 = mid_state.clone();
        let peak1 = st_drop1
            .h1
            .iter()
            .enumerate()
            .max_by(|a, b| dot(a.1, a.1).total_cmp(&dot(b.1, b.1)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        st_drop1.h1[peak1] = [0.0_f64; DIM];
        for _ in 3..DEFAULT_BUDGET {
            st_drop1 = model.abductive_step(&v_clean, &st_drop1, true, None, None, false);
        }
        let unf_d1 = model.unfold_top_down(&v_clean.node_init, &st_drop1, None, false);
        m.rec_drop_l1 += observational_equivalence_excess(&v_clean, &unf_d1.r_hat, &truth);

        // (b) Coupure complète de L2 à t=3, puis poursuite jusqu'à t=6 sans L2
        let mut st_cut2 = mid_state.clone();
        for h in &mut st_cut2.h2 {
            *h = [0.0_f64; DIM];
        }
        for _ in 3..DEFAULT_BUDGET {
            st_cut2 = model.abductive_step(&v_clean, &st_cut2, true, None, None, true);
        }
        let unf_c2 = model.unfold_top_down(&v_clean.node_init, &st_cut2, None, true);
        m.rec_cut_l2 += observational_equivalence_excess(&v_clean, &unf_c2.r_hat, &truth);

        // 3. Sensibilité différentielle : rotation + bijection (cause préservée) vs blocs mélangés (brisée)
        let rot_shift = 45usize;
        let mut rot = Vec::with_capacity(clean.len());
        rot.extend_from_slice(&clean[rot_shift..]);
        rot.extend_from_slice(&clean[..rot_shift]);

        // Bijection aléatoire sur la version tournée (0 dépendance aux valeurs d'octets)
        let mut bij: Vec<u8> = (0u8..=255).collect();
        rng.shuffle(&mut bij);
        let rot_bij: Vec<u8> = rot.iter().map(|&b| bij[b as usize]).collect();
        let v_rot_plain = RelationalView::from_bytes(&rot);
        let v_rot_bij = RelationalView::from_bytes(&rot_bij);

        let tr_rot_plain = model.run_abduction(&v_rot_plain, DEFAULT_BUDGET, true, &[], false);
        let unf_rot_plain =
            model.unfold_top_down(&v_rot_plain.node_init, &tr_rot_plain.state, None, false);
        m.rec_rot += observational_equivalence_excess(&v_rot_plain, &unf_rot_plain.r_hat, &truth);

        let tr_rot_bij = model.run_abduction(&v_rot_bij, DEFAULT_BUDGET, true, &[], false);
        let rep_rot = model.evaluate_phi_state(&v_rot_bij, &tr_rot_bij.state, DEFAULT_BUDGET, true);
        m.phi_rot += rep_rot.phi_sharp;
        m.h2_cos_rot += h2_perm_cosine(&tr_clean.state.h2, &tr_rot_bij.state.h2);

        // Permutation par blocs de 4 octets (conserve les marginales + 75 % des bigrammes, brise la cause)
        let blk = block_shuffle(&clean, 4, &mut rng);
        let v_blk = RelationalView::from_bytes(&blk);
        let tr_blk = model.run_abduction(&v_blk, DEFAULT_BUDGET, true, &[], false);
        let unf_blk = model.unfold_top_down(&v_blk.node_init, &tr_blk.state, None, false);
        let rep_blk = model.evaluate_phi_state(&v_blk, &tr_blk.state, DEFAULT_BUDGET, true);

        m.rec_blk += observational_equivalence_excess(&v_blk, &unf_blk.r_hat, &truth);
        m.phi_blk += rep_blk.phi_sharp;
        m.h2_cos_blk += h2_perm_cosine(&tr_clean.state.h2, &tr_blk.state.h2);
    }

    let n = trials as f64;
    for r in &mut m.rec_noise {
        *r /= n;
    }
    m.h2_stab_20 /= n;
    m.rec_drop_l1 /= n;
    m.rec_cut_l2 /= n;
    m.rec_rot /= n;
    m.rec_blk /= n;
    m.phi_rot /= n;
    m.phi_blk /= n;
    m.h2_cos_rot /= n;
    m.h2_cos_blk /= n;
    m
}

// ============================================================================
// Point d'entrée principal : exécution et tableau comparatif des 4 modèles
// ============================================================================

pub fn run_eval_meta(models: &[(&str, &PhisharpModel)], seed: u64) {
    println!();
    println!("================================================================================================");
    println!("=== BANC HORS-ENTRAÎNEMENT : ISOMORPHISME MÉTA-RELATIONNEL & RÉSISTANCE AUX PERTURBATIONS    ===");
    println!("=== (Modèles figés, 0 modification d'entraînement, 0 octet commun vérifié, seed={})         ===", seed);
    println!("================================================================================================");

    // --- ÉPREUVE 1A : Moules topologiques hors-PTGRS (V=8, 0 octet commun) ---
    let trials_topo = build_trials_topological(seed, 6); // 36 essais (6 par famille)
    let (surf_t1_a, surf_mrr_a) = eval_surface_baseline(&trials_topo);

    println!();
    println!(
        "[1A] ISOMORPHISME DE MOULES TOPOLOGIQUES HORS-PTGRS (36 essais, 4 choix, V=8, 0 octet commun)"
    );
    println!(
        "     Hasard pur : Top1=25.0% MRR=0.521 | Baseline Surface (sans modèle) : Top1={:5.1}% MRR={:.3}",
        surf_t1_a * 100.0_f64,
        surf_mrr_a
    );
    println!(
        "{:<16} | {:>14} | {:>14} | {:>14} | {:>8}",
        "Modèle", "H2 seul (T1/MRR)", "Op(H1,G) (T1/MRR)", "Joint (T1/MRR)", "||H2||"
    );
    println!("{}", "-".repeat(78));

    let mut res_a = Vec::with_capacity(models.len());
    for (label, model) in models {
        let m = eval_model_on_iso(model, &trials_topo);
        println!(
            "{:<16} | {:>5.1}% / {:>5.3} | {:>5.1}% / {:>5.3}   | {:>5.1}% / {:>5.3} | {:>8.4}",
            label,
            m.h2_top1 * 100.0_f64,
            m.h2_mrr,
            m.op_top1 * 100.0_f64,
            m.op_mrr,
            m.joint_top1 * 100.0_f64,
            m.joint_mrr,
            m.h2_norm
        );
        res_a.push(m);
    }

    // --- ÉPREUVE 1B : Isomorphisme de règles PTGRS OOD (V=8, 0 octet commun) ---
    let trials_ood = build_trials_ptgrs_ood(seed ^ 0xBEEF_1234, 24);
    let (surf_t1_b, surf_mrr_b) = eval_surface_baseline(&trials_ood);

    println!();
    println!(
        "[1B] ISOMORPHISME DE CAUSES PTGRS OOD (24 essais, 4 choix, V=8, 0 octet commun)"
    );
    println!(
        "     Hasard pur : Top1=25.0% MRR=0.521 | Baseline Surface (sans modèle) : Top1={:5.1}% MRR={:.3}",
        surf_t1_b * 100.0_f64,
        surf_mrr_b
    );
    println!(
        "{:<16} | {:>14} | {:>14} | {:>14} | {:>8}",
        "Modèle", "H2 seul (T1/MRR)", "Op(H1,G) (T1/MRR)", "Joint (T1/MRR)", "||H2||"
    );
    println!("{}", "-".repeat(78));

    let mut res_b = Vec::with_capacity(models.len());
    for (label, model) in models {
        let m = eval_model_on_iso(model, &trials_ood);
        println!(
            "{:<16} | {:>5.1}% / {:>5.3} | {:>5.1}% / {:>5.3}   | {:>5.1}% / {:>5.3} | {:>8.4}",
            label,
            m.h2_top1 * 100.0_f64,
            m.h2_mrr,
            m.op_top1 * 100.0_f64,
            m.op_mrr,
            m.joint_top1 * 100.0_f64,
            m.joint_mrr,
            m.h2_norm
        );
        res_b.push(m);
    }

    // --- ÉPREUVE 2 : Résistance aux perturbations & sensibilité différentielle ---
    println!();
    println!("[2/2] RÉSISTANCE AUX PERTURBATIONS & SENSIBILITÉ STRUCTURELLE (PTGRS OOD d=4, 16 causes)");
    println!(
        "{:<16} | {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} | {:>7} {:>7} | {:>11}",
        "Modèle", "Rec0%", "Rec10%", "Rec20%", "Rec30%", "DropL1", "CutL2", "φ_rot", "φ_blk", "ΔcosH2(r-b)"
    );
    println!("{}", "-".repeat(104));

    let mut res_p = Vec::with_capacity(models.len());
    for (label, model) in models {
        let p = eval_model_perturbations(model, seed ^ 0x7777_8888);
        let d_cos = p.h2_cos_rot - p.h2_cos_blk;
        println!(
            "{:<16} | {:>+7.4} {:>+7.4} {:>+7.4} {:>+7.4} | {:>+7.4} {:>+7.4} | {:>7.4} {:>7.4} | {:>+11.4}",
            label,
            p.rec_noise[0],
            p.rec_noise[1],
            p.rec_noise[2],
            p.rec_noise[3],
            p.rec_drop_l1,
            p.rec_cut_l2,
            p.phi_rot,
            p.phi_blk,
            d_cos
        );
        res_p.push(p);
    }

    println!("  Détail sensibilité externe (Rec_exc(C) préservé vs brisé | stabilité H2 sous 20% bruit) :");
    for (i, (label, _)) in models.iter().enumerate() {
        let p = &res_p[i];
        println!(
            "    {:<16} : Rec(rot)={:+.4} vs Rec(blk4)={:+.4} (Δ={:+.4}) | cosH2(rot)={:+.3} cosH2(blk4)={:+.3} | stabH2@20%={:+.3}",
            label,
            p.rec_rot,
            p.rec_blk,
            p.rec_rot - p.rec_blk,
            p.h2_cos_rot,
            p.h2_cos_blk,
            p.h2_stab_20
        );
    }

    // --- SYNTHÈSE MDL vs φ_sharp ---
    if models.len() == 4 {
        let mdl_iso = 0.5_f64 * (res_a[2].h2_mrr + res_b[2].h2_mrr);
        let phi_iso = 0.5_f64 * (res_a[3].h2_mrr + res_b[3].h2_mrr);
        let mdl_op = 0.5_f64 * (res_a[2].op_mrr + res_b[2].op_mrr);
        let phi_op = 0.5_f64 * (res_a[3].op_mrr + res_b[3].op_mrr);
        let mdl_jt = 0.5_f64 * (res_a[2].joint_mrr + res_b[2].joint_mrr);
        let phi_jt = 0.5_f64 * (res_a[3].joint_mrr + res_b[3].joint_mrr);

        let mdl_rob = 0.5_f64 * (res_p[2].rec_noise[2] + res_p[2].rec_noise[3]);
        let phi_rob = 0.5_f64 * (res_p[3].rec_noise[2] + res_p[3].rec_noise[3]);

        let mdl_l2_dep = res_p[2].rec_noise[0] - res_p[2].rec_cut_l2;
        let phi_l2_dep = res_p[3].rec_noise[0] - res_p[3].rec_cut_l2;

        println!();
        println!("--- SYNTHÈSE DILEMME MDL (MIP=0.018) vs PHISHARP φ (MIP=0.664) ---");
        println!(
            "  Isomorphisme MRR moy (1A+1B) : H2 [MDL={:.3} vs φ={:.3}] | Op(H1,G) [MDL={:.3} vs φ={:.3}] | Joint [MDL={:.3} vs φ={:.3}]",
            mdl_iso, phi_iso, mdl_op, phi_op, mdl_jt, phi_jt
        );
        println!(
            "  Robustesse Rec(C) bruit 20-30%: MDL={:+.4} vs φ={:+.4}",
            mdl_rob, phi_rob
        );
        println!(
            "  Dépendance causale réelle L2  : chute Rec(C) sous CutL2 [MDL={:+.4} vs φ={:+.4}]",
            mdl_l2_dep, phi_l2_dep
        );
    }
}
