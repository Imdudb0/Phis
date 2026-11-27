use std::collections::HashMap;

pub(crate) const DIM: usize = 12; // Repère orthonormé exact jusqu'à V = 12
pub(crate) const K1: usize = 6;   // Nœuds causaux concrets (Niveau 1)
pub(crate) const K2: usize = 3;   // Moules méta-relationnels (Niveau 2)
pub(crate) const DEFAULT_BUDGET: usize = 6;
pub(crate) const MAX_CTX: usize = 48;
pub(crate) const TASK_W: [f64; 3] = [0.45_f64, 0.35_f64, 0.20_f64]; // [next, mid, skip]

// ============================================================================
// 1. PRNG déterministe (SplitMix64 + Box-Muller, zéro dépendance)
// ============================================================================

#[derive(Clone)]
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0_f64 / ((1u64 << 53) as f64))
    }

    pub(crate) fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12_f64);
        let u2 = self.next_f64();
        (-2.0_f64 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    pub(crate) fn range_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % upper
    }

    pub(crate) fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.range_usize(i + 1);
            slice.swap(i, j);
        }
    }
}

// ============================================================================
// 2. Repère fixe orthonormé à L0 (strictement invariant par bijection, zéro fuite)
// ============================================================================

/// Construit V vecteurs fixes dans R^DIM dépendant UNIQUEMENT de l'indice d'ordre
/// d'apparition u in 0..V (strictement invariant par bijection d'alphabet,
/// strictement orthonormé pour V <= DIM, et immuable pendant le déploiement).
pub(crate) fn build_fixed_basis(v_size: usize) -> Vec<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(v_size);
    for u in 0..v_size {
        if u < DIM {
            let mut v = vec![0.0_f64; DIM];
            v[u] = 1.0_f64;
            basis.push(v);
        } else {
            // Vecteur pseudo-orthogonal déterministe sur la sphère unité de R^DIM
            let mut rng = Rng::new(0x517C_C1B7_2722_0A95u64 ^ ((u as u64).wrapping_mul(0x9E37_79B9)));
            let mut v = vec![0.0_f64; DIM];
            let mut sq = 0.0_f64;
            for x in &mut v {
                *x = rng.next_normal();
                sq += (*x) * (*x);
            }
            let inv = 1.0_f64 / sq.sqrt().max(1e-12_f64);
            for x in &mut v {
                *x *= inv;
            }
            basis.push(v);
        }
    }
    basis
}

// ============================================================================
// 3. PTGRS générique & OOD (C réservée exclusivement à l'étalonnage post-hoc)
// ============================================================================

#[derive(Clone)]
pub(crate) struct RewriteRule {
    pub(crate) lhs: u8,
    pub(crate) rhs_primary: Vec<u8>,
    pub(crate) rhs_alt: Vec<u8>,
    pub(crate) prob_primary: f64,
}

#[derive(Clone)]
pub(crate) struct PtgrsCause {
    pub(crate) start: u8,
    pub(crate) rules: Vec<RewriteRule>,
    pub(crate) terminals: Vec<u8>,
}

pub(crate) struct NonMarkovGroundTruth {
    pub(crate) next: HashMap<(u8, u8, u8), f64>, // P(w | u, v)
    pub(crate) mid: HashMap<(u8, u8, u8), f64>,  // P(v | u, _, w)
    pub(crate) skip: HashMap<(u8, u8, u8), f64>, // P(z | u, v, _)
}

impl PtgrsCause {
    /// Générateur d'entraînement strictement générique (aucun format NL ni moule ciblé).
    pub(crate) fn sample_generic(rng: &mut Rng) -> Self {
        let mut pool: Vec<u8> = (0u8..=255).collect();
        rng.shuffle(&mut pool);

        let n_nonterms = 3 + rng.range_usize(3); // 3..5 non-terminaux
        let n_terms = 5 + rng.range_usize(5);    // 5..9 terminaux
        let nonterms = &pool[0..n_nonterms];
        let terminals = pool[n_nonterms..n_nonterms + n_terms].to_vec();

        let mut rules = Vec::with_capacity(n_nonterms);
        for (idx, &nt) in nonterms.iter().enumerate() {
            let primary_len = 2 + rng.range_usize(3);
            let mut rhs_primary = Vec::with_capacity(primary_len);
            let nt_pos = rng.range_usize(primary_len);
            let target_nt = nonterms[(idx + 1 + rng.range_usize(n_nonterms - 1)) % n_nonterms];

            for pos in 0..primary_len {
                if pos == nt_pos {
                    rhs_primary.push(target_nt);
                } else {
                    rhs_primary.push(terminals[rng.range_usize(n_terms)]);
                }
            }

            let alt_len = 2 + rng.range_usize(2);
            let mut rhs_alt = Vec::with_capacity(alt_len);
            for _ in 0..alt_len {
                rhs_alt.push(terminals[rng.range_usize(n_terms)]);
            }

            let prob_primary = 0.65_f64 + 0.25_f64 * rng.next_f64();
            rules.push(RewriteRule {
                lhs: nt,
                rhs_primary,
                rhs_alt,
                prob_primary,
            });
        }

        Self {
            start: nonterms[0],
            rules,
            terminals,
        }
    }

    /// Famille OOD (barreau 1) : double branchement récursif emboîté.
    pub(crate) fn sample_ood(rng: &mut Rng, depth_family: usize) -> Self {
        let mut pool: Vec<u8> = (0u8..=255).collect();
        rng.shuffle(&mut pool);

        let n_nonterms = 4 + depth_family.min(4);
        let n_terms = 7 + rng.range_usize(4); // 7..10 terminaux
        let nonterms = &pool[0..n_nonterms];
        let terminals = pool[n_nonterms..n_nonterms + n_terms].to_vec();

        let mut rules = Vec::with_capacity(n_nonterms);
        for (idx, &nt) in nonterms.iter().enumerate() {
            let n1 = nonterms[(idx + 1) % n_nonterms];
            let n2 = nonterms[(idx + 2) % n_nonterms];
            let t1 = terminals[rng.range_usize(n_terms)];
            let t2 = terminals[rng.range_usize(n_terms)];
            let t3 = terminals[rng.range_usize(n_terms)];

            let rhs_primary = vec![t1, n1, t2, n2, t1];
            let rhs_alt = vec![t3, t2, t1];
            let prob_primary = 0.72_f64 + 0.18_f64 * rng.next_f64();

            rules.push(RewriteRule {
                lhs: nt,
                rhs_primary,
                rhs_alt,
                prob_primary,
            });
        }

        Self {
            start: nonterms[0],
            rules,
            terminals,
        }
    }

    pub(crate) fn emit_effect(&self, rng: &mut Rng, max_depth: usize, target_len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(target_len + 32);
        let mut rule_map: HashMap<u8, &RewriteRule> = HashMap::new();
        for r in &self.rules {
            rule_map.insert(r.lhs, r);
        }

        while out.len() < target_len {
            let mut frontier: Vec<(u8, usize)> = vec![(self.start, 0)];
            while let Some((sym, depth)) = frontier.pop() {
                if out.len() >= target_len {
                    break;
                }
                if let Some(rule) = rule_map.get(&sym) {
                    let chosen = if depth < max_depth && rng.next_f64() < rule.prob_primary {
                        &rule.rhs_primary
                    } else {
                        &rule.rhs_alt
                    };
                    for &child in chosen.iter().rev() {
                        frontier.push((child, depth + 1));
                    }
                } else {
                    out.push(sym);
                }
            }
            if out.is_empty() {
                out.push(self.terminals[0]);
            }
        }
        out.truncate(target_len);
        out
    }

    /// Distribution observationnelle non-markovienne de référence de C (jamais vue en entraînement).
    pub(crate) fn ground_truth_non_markov(
        &self,
        rng: &mut Rng,
        max_depth: usize,
    ) -> NonMarkovGroundTruth {
        let mut c_next: HashMap<(u8, u8, u8), usize> = HashMap::new();
        let mut s_next: HashMap<(u8, u8), usize> = HashMap::new();
        let mut c_mid: HashMap<(u8, u8, u8), usize> = HashMap::new();
        let mut s_mid: HashMap<(u8, u8), usize> = HashMap::new();
        let mut c_skip: HashMap<(u8, u8, u8), usize> = HashMap::new();
        let mut s_skip: HashMap<(u8, u8), usize> = HashMap::new();

        for _ in 0..32 {
            let sample = self.emit_effect(rng, max_depth, 180);
            for w in sample.windows(3) {
                *c_next.entry((w[0], w[1], w[2])).or_insert(0) += 1;
                *s_next.entry((w[0], w[1])).or_insert(0) += 1;

                *c_mid.entry((w[0], w[2], w[1])).or_insert(0) += 1;
                *s_mid.entry((w[0], w[2])).or_insert(0) += 1;
            }
            for w in sample.windows(4) {
                *c_skip.entry((w[0], w[1], w[3])).or_insert(0) += 1;
                *s_skip.entry((w[0], w[1])).or_insert(0) += 1;
            }
        }

        let norm = |c: HashMap<(u8, u8, u8), usize>,
                    s: HashMap<(u8, u8), usize>|
         -> HashMap<(u8, u8, u8), f64> {
            let mut out = HashMap::with_capacity(c.len());
            for ((a, b, t), cnt) in c {
                let tot = *s.get(&(a, b)).unwrap_or(&1) as f64;
                out.insert((a, b, t), (cnt as f64) / tot);
            }
            out
        };

        NonMarkovGroundTruth {
            next: norm(c_next, s_next),
            mid: norm(c_mid, s_mid),
            skip: norm(c_skip, s_skip),
        }
    }
}

// ============================================================================
// 4. Vue relationnelle NON-MARKOVIENNE (relations ternaires, 100 % bijective)
// ============================================================================

#[derive(Clone)]
pub(crate) struct CtxEntry {
    pub(crate) u: usize,
    pub(crate) v: usize,
    pub(crate) weight: f64,
    pub(crate) target: Vec<f64>, // Distribution conditionnelle empirique de taille v_size
}

#[derive(Clone)]
pub(crate) struct NonMarkovTables {
    pub(crate) tables: [Vec<CtxEntry>; 3], // 0: (u,v)->w, 1: (u,w)->mid, 2: (u,v)->skip3
    pub(crate) h_marg: [f64; 3],           // Entropie croisée de référence unigramme P_marg
    pub(crate) h_markov1: [f64; 3],        // Entropie croisée du meilleur modèle markovien d'ordre 1
}

#[derive(Clone)]
pub(crate) struct RelationalView {
    pub(crate) vocab: Vec<u8>,
    pub(crate) seq: Vec<usize>,
    pub(crate) v_size: usize,
    pub(crate) p_marg: Vec<f64>,
    pub(crate) basis: Vec<Vec<f64>>, // Repère fixe immuable L0 (v_size x DIM)
    pub(crate) obs: NonMarkovTables,
    pub(crate) shuf: NonMarkovTables, // Contre-exemple E- à marginales P_marg exactes
}

fn build_ctx_tables(seq: &[usize], v_size: usize, p_marg: &[f64]) -> NonMarkovTables {
    // Modèles markoviens d'ordre 1 empiriques (pour mesurer le gain strictement non-markovien)
    let mut m1_lag1 = vec![vec![1e-6_f64; v_size]; v_size];
    let mut m1_lag2 = vec![vec![1e-6_f64; v_size]; v_size];
    for w in seq.windows(2) {
        if w[0] < v_size && w[1] < v_size {
            m1_lag1[w[0]][w[1]] += 1.0_f64;
        }
    }
    for w in seq.windows(3) {
        if w[0] < v_size && w[2] < v_size {
            m1_lag2[w[0]][w[2]] += 1.0_f64;
        }
    }
    for row in m1_lag1.iter_mut().chain(m1_lag2.iter_mut()) {
        let s: f64 = row.iter().sum();
        for x in row.iter_mut() {
            *x /= s.max(1e-12_f64);
        }
    }

    let mut counts: [HashMap<(usize, usize), (f64, Vec<f64>)>; 3] =
        [HashMap::new(), HashMap::new(), HashMap::new()];

    for w in seq.windows(3) {
        let (a, b, c) = (w[0], w[1], w[2]);
        if a < v_size && b < v_size && c < v_size {
            let e0 = counts[0]
                .entry((a, b))
                .or_insert_with(|| (0.0_f64, vec![0.0_f64; v_size]));
            e0.0 += 1.0_f64;
            e0.1[c] += 1.0_f64;

            let e1 = counts[1]
                .entry((a, c))
                .or_insert_with(|| (0.0_f64, vec![0.0_f64; v_size]));
            e1.0 += 1.0_f64;
            e1.1[b] += 1.0_f64;
        }
    }
    for w in seq.windows(4) {
        let (a, b, d) = (w[0], w[1], w[3]);
        if a < v_size && b < v_size && d < v_size {
            let e2 = counts[2]
                .entry((a, b))
                .or_insert_with(|| (0.0_f64, vec![0.0_f64; v_size]));
            e2.0 += 1.0_f64;
            e2.1[d] += 1.0_f64;
        }
    }

    let mut tables: [Vec<CtxEntry>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut h_marg = [1.0_f64; 3];
    let mut h_markov1 = [1.0_f64; 3];

    for t_idx in 0..3 {
        // Tri déterministe : fréquence décroissante, puis (u, v) croissant (100 % invariant bijection)
        let mut items: Vec<((usize, usize), f64, Vec<f64>)> = counts[t_idx]
            .drain()
            .map(|(k, (cnt, vec))| (k, cnt, vec))
            .collect();
        items.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.0.cmp(&b.0.0))
                .then_with(|| a.0.1.cmp(&b.0.1))
        });
        items.truncate(MAX_CTX);

        let tot: f64 = items.iter().map(|x| x.1).sum();
        if tot <= 0.0_f64 {
            continue;
        }

        let mut hm = 0.0_f64;
        let mut hmk = 0.0_f64;
        for ((u, v), cnt, mut vec) in items {
            let weight = cnt / tot;
            for x in &mut vec {
                *x /= cnt;
            }
            for w in 0..v_size {
                let p = vec[w];
                if p > 1e-12_f64 {
                    hm -= weight * p * p_marg[w].max(1e-9_f64).ln();
                    let p_mk = match t_idx {
                        0 => m1_lag1[v][w],
                        1 => m1_lag1[u][w],
                        _ => m1_lag2[v][w],
                    };
                    hmk -= weight * p * p_mk.max(1e-9_f64).ln();
                }
            }
            tables[t_idx].push(CtxEntry {
                u,
                v,
                weight,
                target: vec,
            });
        }
        h_marg[t_idx] = hm.max(0.05_f64);
        h_markov1[t_idx] = hmk.max(0.05_f64);
    }

    NonMarkovTables {
        tables,
        h_marg,
        h_markov1,
    }
}

impl RelationalView {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Self {
        let mut vocab = Vec::new();
        let mut map = [usize::MAX; 256];
        let mut seq = Vec::with_capacity(bytes.len());

        for &b in bytes {
            let idx = b as usize;
            if map[idx] == usize::MAX {
                map[idx] = vocab.len();
                vocab.push(b);
            }
            seq.push(map[idx]);
        }
        Self::from_indexed(vocab, seq)
    }

    pub(crate) fn from_indexed(vocab: Vec<u8>, seq: Vec<usize>) -> Self {
        let v_size = vocab.len().max(1);
        let mut freq = vec![0.0_f64; v_size];
        for &s in &seq {
            if s < v_size {
                freq[s] += 1.0_f64;
            }
        }
        let tot = seq.len().max(1) as f64;
        let mut p_marg = vec![0.0_f64; v_size];
        for u in 0..v_size {
            p_marg[u] = (freq[u] + 1e-6_f64) / (tot + 1e-6_f64 * (v_size as f64));
        }

        let basis = build_fixed_basis(v_size);
        let obs = build_ctx_tables(&seq, v_size, &p_marg);

        // Permutation déterministe 100 % invariante par bijection préservant exactement P_marg
        let mut shuf_seq = seq.clone();
        let mut shuf_rng = Rng::new(
            0xD1B5_4A32_D192_ED03u64
                ^ ((seq.len() as u64).wrapping_mul(0x9E37_79B9))
                ^ (v_size as u64),
        );
        shuf_rng.shuffle(&mut shuf_seq);
        let shuf = build_ctx_tables(&shuf_seq, v_size, &p_marg);

        Self {
            vocab,
            seq,
            v_size,
            p_marg,
            basis,
            obs,
            shuf,
        }
    }

    pub(crate) fn subslice_same_vocab(&self, start: usize, end: usize) -> Self {
        let s = start.min(self.seq.len());
        let e = end.min(self.seq.len()).max(s);
        Self::from_indexed(self.vocab.clone(), self.seq[s..e].to_vec())
    }
}

// ============================================================================
// 5. Équivalence observationnelle non-markovienne avec C et corrélations
// ============================================================================

/// Récupération excédentaire normalisée de la structure NON-MARKOVIENNE de C
/// au-dessus de P_marg (sur les 3 tâches : next, bracket-mid, skip-3).
pub(crate) fn observational_equivalence_nm(
    view: &RelationalView,
    preds: &[Vec<Vec<f64>>; 3],
    truth: &NonMarkovGroundTruth,
) -> f64 {
    let maps = [&truth.next, &truth.mid, &truth.skip];
    let mut total = 0.0_f64;

    for t_idx in 0..3 {
        let entries = &view.obs.tables[t_idx];
        let mut rec_sum = 0.0_f64;
        let mut w_sum = 0.0_f64;

        for (i, entry) in entries.iter().enumerate() {
            if i >= preds[t_idx].len() {
                break;
            }
            let bu = view.vocab[entry.u];
            let bv = view.vocab[entry.v];
            let pred = &preds[t_idx][i];

            let mut c_mass = 0.0_f64;
            let mut tvd_pred = 0.0_f64;
            let mut tvd_marg = 0.0_f64;

            for w in 0..view.v_size {
                let bw = view.vocab[w];
                let pt = *maps[t_idx].get(&(bu, bv, bw)).unwrap_or(&0.0_f64);
                c_mass += pt;
                tvd_pred += (pt - pred[w]).abs();
                tvd_marg += (pt - view.p_marg[w]).abs();
            }
            if c_mass > 0.5_f64 {
                let tvd_p = 0.5_f64 * tvd_pred;
                let tvd_m = 0.5_f64 * tvd_marg;
                let excess = ((tvd_m - tvd_p) / (tvd_m + 0.05_f64)).clamp(-1.0_f64, 1.0_f64);
                rec_sum += entry.weight * excess;
                w_sum += entry.weight;
            }
        }
        if w_sum > 0.0_f64 {
            total += TASK_W[t_idx] * (rec_sum / w_sum);
        }
    }
    total
}

pub(crate) fn pearson_corr(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0_f64;
    }
    let mx = xs[..n].iter().sum::<f64>() / (n as f64);
    let my = ys[..n].iter().sum::<f64>() / (n as f64);
    let mut num = 0.0_f64;
    let mut dx = 0.0_f64;
    let mut dy = 0.0_f64;
    for i in 0..n {
        let a = xs[i] - mx;
        let b = ys[i] - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    let den = (dx * dy).sqrt();
    if den < 1e-12_f64 { 0.0_f64 } else { num / den }
}

fn ranks(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    let mut order: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
    order.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut r = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (order[j].1 - order[i].1).abs() < 1e-12_f64 {
            j += 1;
        }
        let avg = 0.5_f64 * ((i + j - 1) as f64);
        for k in i..j {
            r[order[k].0] = avg;
        }
        i = j;
    }
    r
}

pub(crate) fn spearman_corr(xs: &[f64], ys: &[f64]) -> f64 {
    let rx = ranks(xs);
    let ry = ranks(ys);
    pearson_corr(&rx, &ry)
}
