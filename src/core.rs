use std::collections::HashMap;

pub(crate) const DIM: usize = 8;
pub(crate) const K1: usize = 6; // Nœuds causaux réifiés (Niveau 1)
pub(crate) const K2: usize = 3; // Nœuds méta-relationnels (Niveau 2)
pub(crate) const DEFAULT_BUDGET: usize = 6;
pub(crate) const SCALE_W: [f64; 3] = [0.50_f64, 0.30_f64, 0.20_f64];

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
// 2. Primitives algébriques
// ============================================================================

#[inline]
pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0_f64 / (1.0_f64 + (-x.clamp(-20.0_f64, 20.0_f64)).exp())
}

#[inline]
pub(crate) fn mat_vec(mat: &[f64], x: &[f64; DIM]) -> [f64; DIM] {
    let mut y = [0.0_f64; DIM];
    for r in 0..DIM {
        let mut acc = 0.0_f64;
        let base = r * DIM;
        for c in 0..DIM {
            acc += mat[base + c] * x[c];
        }
        y[r] = acc;
    }
    y
}

#[inline]
pub(crate) fn dot(a: &[f64; DIM], b: &[f64; DIM]) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..DIM {
        s += a[i] * b[i];
    }
    s
}

#[inline]
pub(crate) fn vec_norm(v: &[f64; DIM]) -> f64 {
    dot(v, v).sqrt()
}

#[inline]
pub(crate) fn vec_dist(a: &[f64; DIM], b: &[f64; DIM]) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..DIM {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

#[inline]
pub(crate) fn rms_norm(v: &mut [f64; DIM]) {
    let mean_sq = dot(v, v) / (DIM as f64);
    if mean_sq < 1e-18_f64 {
        return;
    }
    let denom = (mean_sq + 0.25_f64).sqrt();
    for x in v.iter_mut() {
        *x /= denom;
    }
}

pub(crate) fn softmax_slice(logits: &[f64], out: &mut [f64]) {
    let mut max_l = f64::NEG_INFINITY;
    for &l in logits {
        if l > max_l {
            max_l = l;
        }
    }
    let mut sum_e = 0.0_f64;
    for (o, &l) in out.iter_mut().zip(logits.iter()) {
        let e = if l < -1e8_f64 {
            0.0_f64
        } else {
            (l - max_l).exp()
        };
        *o = e;
        sum_e += e;
    }
    if sum_e <= 1e-15_f64 {
        let u = 1.0_f64 / (out.len().max(1) as f64);
        for o in out.iter_mut() {
            *o = u;
        }
    } else {
        let inv = 1.0_f64 / sum_e;
        for o in out.iter_mut() {
            *o *= inv;
        }
    }
}

// ============================================================================
// 3. PTGRS purement générique (entraînement fermé ; C réservée à l'étalonnage)
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

pub(crate) struct MultiScaleGroundTruth {
    pub(crate) r1: HashMap<(u8, u8), f64>,
    pub(crate) r2: HashMap<(u8, u8), f64>,
    pub(crate) r3: HashMap<(u8, u8), f64>,
}

impl PtgrsCause {
    pub(crate) fn sample_generic(rng: &mut Rng) -> Self {
        let mut pool: Vec<u8> = (0u8..=255).collect();
        rng.shuffle(&mut pool);

        let n_nonterms = 3 + rng.range_usize(3);
        let n_terms = 5 + rng.range_usize(6);
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

    pub(crate) fn sample_ood(rng: &mut Rng, depth_family: usize) -> Self {
        let mut pool: Vec<u8> = (0u8..=255).collect();
        rng.shuffle(&mut pool);

        let n_nonterms = 4 + depth_family.min(4);
        let n_terms = 7 + rng.range_usize(6);
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

    pub(crate) fn ground_truth_multi_scale(
        &self,
        rng: &mut Rng,
        max_depth: usize,
    ) -> MultiScaleGroundTruth {
        let mut c1: HashMap<(u8, u8), usize> = HashMap::new();
        let mut s1: HashMap<u8, usize> = HashMap::new();
        let mut c2: HashMap<(u8, u8), usize> = HashMap::new();
        let mut s2: HashMap<u8, usize> = HashMap::new();
        let mut c3: HashMap<(u8, u8), usize> = HashMap::new();
        let mut s3: HashMap<u8, usize> = HashMap::new();

        for _ in 0..24 {
            let sample = self.emit_effect(rng, max_depth, 180);
            for w in sample.windows(2) {
                *c1.entry((w[0], w[1])).or_insert(0) += 1;
                *s1.entry(w[0]).or_insert(0) += 1;
            }
            for w in sample.windows(3) {
                *c2.entry((w[0], w[2])).or_insert(0) += 1;
                *s2.entry(w[0]).or_insert(0) += 1;
            }
            for w in sample.windows(4) {
                *c3.entry((w[0], w[3])).or_insert(0) += 1;
                *s3.entry(w[0]).or_insert(0) += 1;
            }
        }

        let norm = |c: HashMap<(u8, u8), usize>, s: HashMap<u8, usize>| -> HashMap<(u8, u8), f64> {
            let mut out = HashMap::with_capacity(c.len());
            for ((u, v), cnt) in c {
                let tot = *s.get(&u).unwrap_or(&1) as f64;
                out.insert((u, v), (cnt as f64) / tot);
            }
            out
        };

        MultiScaleGroundTruth {
            r1: norm(c1, s1),
            r2: norm(c2, s2),
            r3: norm(c3, s3),
        }
    }
}

// ============================================================================
// 4. Vue relationnelle multi-échelle (100 % invariante par bijection)
// ============================================================================

#[derive(Clone)]
pub(crate) struct ScaleGraph {
    pub(crate) r: [Vec<f64>; 3],
    pub(crate) w: [Vec<f64>; 3],
    pub(crate) h_marg: [f64; 3],
}

#[derive(Clone)]
pub(crate) struct RelationalView {
    pub(crate) vocab: Vec<u8>,
    pub(crate) seq: Vec<usize>,
    pub(crate) v_size: usize,
    pub(crate) p_marg: Vec<f64>,
    pub(crate) graph: ScaleGraph,
    pub(crate) shuf_graph: ScaleGraph,
    pub(crate) node_init: Vec<[f64; DIM]>,
}

pub(crate) fn build_scale_graph(seq: &[usize], v_size: usize, p_marg: &[f64]) -> ScaleGraph {
    let mut r = [
        vec![0.0_f64; v_size * v_size],
        vec![0.0_f64; v_size * v_size],
        vec![0.0_f64; v_size * v_size],
    ];
    let mut w = [
        vec![0.0_f64; v_size],
        vec![0.0_f64; v_size],
        vec![0.0_f64; v_size],
    ];
    let mut h_marg = [1.0_f64; 3];

    for (s_idx, lag) in [1usize, 2, 3].into_iter().enumerate() {
        if seq.len() <= lag || v_size == 0 {
            continue;
        }
        let mut counts = vec![0.0_f64; v_size * v_size];
        let mut row_sums = vec![0.0_f64; v_size];
        let mut total = 0.0_f64;

        for t in 0..(seq.len() - lag) {
            let u = seq[t];
            let v = seq[t + lag];
            if u < v_size && v < v_size {
                counts[u * v_size + v] += 1.0_f64;
                row_sums[u] += 1.0_f64;
                total += 1.0_f64;
            }
        }
        if total <= 0.0_f64 {
            continue;
        }

        let mut hm = 0.0_f64;
        for u in 0..v_size {
            if row_sums[u] > 0.0_f64 {
                let wu = row_sums[u] / total;
                w[s_idx][u] = wu;
                for v in 0..v_size {
                    let p = counts[u * v_size + v] / row_sums[u];
                    r[s_idx][u * v_size + v] = p;
                    if p > 1e-12_f64 {
                        hm -= wu * p * p_marg[v].max(1e-9_f64).ln();
                    }
                }
            }
        }
        h_marg[s_idx] = hm.max(0.05_f64);
    }

    ScaleGraph { r, w, h_marg }
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

        let graph = build_scale_graph(&seq, v_size, &p_marg);

        let mut bwd_c = vec![0.0_f64; v_size * v_size];
        let mut bwd_s = vec![0.0_f64; v_size];
        for win in seq.windows(2) {
            if win[0] < v_size && win[1] < v_size {
                bwd_c[win[1] * v_size + win[0]] += 1.0_f64;
                bwd_s[win[1]] += 1.0_f64;
            }
        }

        let mut last_pos = vec![None::<usize>; v_size];
        let mut gap_sum = vec![0.0_f64; v_size];
        let mut gap_cnt = vec![0.0_f64; v_size];
        for (pos, &s) in seq.iter().enumerate() {
            if s < v_size {
                if let Some(prev) = last_pos[s] {
                    gap_sum[s] += (pos - prev) as f64;
                    gap_cnt[s] += 1.0_f64;
                }
                last_pos[s] = Some(pos);
            }
        }

        let ln_v = (v_size.max(2) as f64).ln();
        let mut node_init = vec![[0.0_f64; DIM]; v_size];

        for u in 0..v_size {
            if freq[u] <= 0.0_f64 {
                continue;
            }
            let mut h1 = 0.0_f64;
            let mut hb = 0.0_f64;
            let mut h2 = 0.0_f64;
            let mut h3 = 0.0_f64;
            let mut deg_fwd = 0.0_f64;
            let mut deg_bwd = 0.0_f64;
            let mut align_12 = 0.0_f64;

            for v in 0..v_size {
                let p1 = graph.r[0][u * v_size + v];
                let p2 = graph.r[1][u * v_size + v];
                let p3 = graph.r[2][u * v_size + v];
                let pb = if bwd_s[u] > 0.0_f64 {
                    bwd_c[u * v_size + v] / bwd_s[u]
                } else {
                    0.0_f64
                };

                if p1 > 1e-12_f64 {
                    h1 -= p1 * p1.ln();
                    deg_fwd += 1.0_f64;
                }
                if pb > 1e-12_f64 {
                    hb -= pb * pb.ln();
                    deg_bwd += 1.0_f64;
                }
                if p2 > 1e-12_f64 {
                    h2 -= p2 * p2.ln();
                }
                if p3 > 1e-12_f64 {
                    h3 -= p3 * p3.ln();
                }
                align_12 += p1 * p2;
            }

            let asym = (deg_fwd - deg_bwd) / (v_size as f64);
            let mean_gap = if gap_cnt[u] > 0.0_f64 {
                (gap_sum[u] / gap_cnt[u]) / (v_size as f64 + 1.0_f64)
            } else {
                1.0_f64
            };

            node_init[u] = [
                (p_marg[u] * (v_size as f64) - 1.0_f64).tanh(),
                h1 / ln_v,
                hb / ln_v,
                h2 / ln_v,
                h3 / ln_v,
                asym.tanh(),
                (align_12 * 3.0_f64).tanh(),
                (1.0_f64 / (1.0_f64 + mean_gap)).tanh(),
            ];
        }

        let mut shuf_seq = seq.clone();
        let mut shuf_rng = Rng::new(
            0xD1B5_4A32_D192_ED03u64
                ^ ((seq.len() as u64).wrapping_mul(0x9E37_79B9))
                ^ (v_size as u64),
        );
        shuf_rng.shuffle(&mut shuf_seq);
        let shuf_graph = build_scale_graph(&shuf_seq, v_size, &p_marg);

        Self {
            vocab,
            seq,
            v_size,
            p_marg,
            graph,
            shuf_graph,
            node_init,
        }
    }

    pub(crate) fn subslice_same_vocab(&self, start: usize, end: usize) -> Self {
        let s = start.min(self.seq.len());
        let e = end.min(self.seq.len()).max(s);
        Self::from_indexed(self.vocab.clone(), self.seq[s..e].to_vec())
    }
}

// ============================================================================
// 5. Métriques d'information, équivalence externe C et corrélations
// ============================================================================

pub(crate) fn scale_gain(
    target: &[f64],
    pred: &[f64],
    row_w: &[f64],
    h_marg: f64,
    v: usize,
) -> f64 {
    let mut nll = 0.0_f64;
    let mut w_tot = 0.0_f64;
    for u in 0..v {
        let wu = row_w[u];
        if wu <= 0.0_f64 {
            continue;
        }
        w_tot += wu;
        let base = u * v;
        for w in 0..v {
            let p = target[base + w];
            if p > 1e-12_f64 {
                let q = pred[base + w].max(1e-9_f64);
                nll -= wu * p * q.ln();
            }
        }
    }
    if w_tot <= 0.0_f64 {
        return 0.0_f64;
    }
    nll /= w_tot;
    ((h_marg - nll) / h_marg.max(0.05_f64)).clamp(-1.0_f64, 1.0_f64)
}

pub(crate) fn multi_scale_gain(graph: &ScaleGraph, r_hat: &[Vec<f64>; 3], v: usize) -> f64 {
    let mut acc = 0.0_f64;
    for s in 0..3 {
        acc += SCALE_W[s] * scale_gain(&graph.r[s], &r_hat[s], &graph.w[s], graph.h_marg[s], v);
    }
    acc.clamp(-1.0_f64, 1.0_f64)
}

pub(crate) fn multi_scale_l1(graph: &ScaleGraph, r_hat: &[Vec<f64>; 3], v: usize) -> f64 {
    let mut total = 0.0_f64;
    for s in 0..3 {
        let mut sc = 0.0_f64;
        let mut wt = 0.0_f64;
        for u in 0..v {
            let wu = graph.w[s][u];
            if wu <= 0.0_f64 {
                continue;
            }
            wt += wu;
            let mut l1 = 0.0_f64;
            let base = u * v;
            for w in 0..v {
                l1 += (graph.r[s][base + w] - r_hat[s][base + w]).abs();
            }
            sc += wu * 0.5_f64 * l1;
        }
        if wt > 0.0_f64 {
            total += SCALE_W[s] * (sc / wt);
        }
    }
    total
}

pub(crate) fn multi_scale_pred_tvd(
    a: &[Vec<f64>; 3],
    b: &[Vec<f64>; 3],
    row_w: &[Vec<f64>; 3],
    v: usize,
) -> f64 {
    let mut total = 0.0_f64;
    for s in 0..3 {
        let mut sc = 0.0_f64;
        let mut wt = 0.0_f64;
        for u in 0..v {
            let wu = row_w[s][u];
            if wu <= 0.0_f64 {
                continue;
            }
            wt += wu;
            let mut l1 = 0.0_f64;
            let base = u * v;
            for w in 0..v {
                l1 += (a[s][base + w] - b[s][base + w]).abs();
            }
            sc += wu * 0.5_f64 * l1;
        }
        if wt > 0.0_f64 {
            total += SCALE_W[s] * (sc / wt);
        }
    }
    total
}

pub(crate) fn log_likelihood_graph(graph: &ScaleGraph, r_hat: &[Vec<f64>; 3], v: usize) -> f64 {
    let mut ll = 0.0_f64;
    for s in 0..3 {
        let mut sc = 0.0_f64;
        let mut wt = 0.0_f64;
        for u in 0..v {
            let wu = graph.w[s][u];
            if wu <= 0.0_f64 {
                continue;
            }
            wt += wu;
            let base = u * v;
            for w in 0..v {
                let p = graph.r[s][base + w];
                if p > 1e-12_f64 {
                    sc += wu * p * r_hat[s][base + w].max(1e-9_f64).ln();
                }
            }
        }
        if wt > 0.0_f64 {
            ll += SCALE_W[s] * (sc / wt);
        }
    }
    ll
}

/// Récupération excédentaire normalisée de la structure conditionnelle de C au-dessus de P_marg.
pub(crate) fn observational_equivalence_excess(
    view: &RelationalView,
    r_hat: &[Vec<f64>; 3],
    truth: &MultiScaleGroundTruth,
) -> f64 {
    let v = view.v_size;
    let maps = [&truth.r1, &truth.r2, &truth.r3];
    let mut total = 0.0_f64;

    for s in 0..3 {
        let mut rec_sum = 0.0_f64;
        let mut w_sum = 0.0_f64;
        for u in 0..v {
            let wu = view.graph.w[s][u];
            if wu <= 0.0_f64 {
                continue;
            }
            let bu = view.vocab[u];
            let mut c_mass = 0.0_f64;
            let mut tvd_pred = 0.0_f64;
            let mut tvd_marg = 0.0_f64;

            for w in 0..v {
                let bw = view.vocab[w];
                let pt = *maps[s].get(&(bu, bw)).unwrap_or(&0.0_f64);
                c_mass += pt;
                tvd_pred += (pt - r_hat[s][u * v + w]).abs();
                tvd_marg += (pt - view.p_marg[w]).abs();
            }
            if c_mass > 0.5_f64 {
                let tvd_p = 0.5_f64 * tvd_pred;
                let tvd_m = 0.5_f64 * tvd_marg;
                let excess = ((tvd_m - tvd_p) / (tvd_m + 0.05_f64)).clamp(-1.0_f64, 1.0_f64);
                rec_sum += wu * excess;
                w_sum += wu;
            }
        }
        if w_sum > 0.0_f64 {
            total += SCALE_W[s] * (rec_sum / w_sum);
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
