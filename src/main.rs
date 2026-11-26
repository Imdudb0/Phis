use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::thread;

const DIM: usize = 8;
const K1: usize = 6; // Nœuds causaux réifiés (Niveau 1)
const K2: usize = 3; // Nœuds méta-relationnels (Niveau 2)
const DEFAULT_BUDGET: usize = 6;
const MAGIC: &[u8; 8] = b"PHIS2421";

// ============================================================================
// 1. PRNG déterministe (SplitMix64 + Box-Muller, zéro dépendance)
// ============================================================================

#[derive(Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0_f64 / ((1u64 << 53) as f64))
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12_f64);
        let u2 = self.next_f64();
        (-2.0_f64 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn range_usize(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % upper
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.range_usize(i + 1);
            slice.swap(i, j);
        }
    }
}

// ============================================================================
// 2. PTGRS purement générique (entraînement fermé ; C réservée à l'étalonnage)
// ============================================================================

#[derive(Clone)]
struct RewriteRule {
    lhs: u8,
    rhs_primary: Vec<u8>,
    rhs_alt: Vec<u8>,
    prob_primary: f64,
}

#[derive(Clone)]
struct PtgrsCause {
    start: u8,
    rules: Vec<RewriteRule>,
    terminals: Vec<u8>,
}

struct MultiScaleGroundTruth {
    r1: HashMap<(u8, u8), f64>,
    r2: HashMap<(u8, u8), f64>,
    r3: HashMap<(u8, u8), f64>,
}

impl PtgrsCause {
    /// Générateur d'entraînement strictement générique (aucun format NL ni moule ciblé).
    fn sample_generic(rng: &mut Rng) -> Self {
        let mut pool: Vec<u8> = (0u8..=255).collect();
        rng.shuffle(&mut pool);

        let n_nonterms = 3 + rng.range_usize(3); // 3..5 non-terminaux
        let n_terms = 5 + rng.range_usize(6); // 5..10 terminaux
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

    /// Famille PTGRS hors-distribution (barreau 1) : double branchement récursif.
    fn sample_ood(rng: &mut Rng, depth_family: usize) -> Self {
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

    fn emit_effect(&self, rng: &mut Rng, max_depth: usize, target_len: usize) -> Vec<u8> {
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

    /// Distribution observationnelle multi-échelle de référence de C (jamais vue à l'entraînement).
    fn ground_truth_multi_scale(&self, rng: &mut Rng, max_depth: usize) -> MultiScaleGroundTruth {
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
// 3. Représentation relationnelle multi-échelle (100 % invariante par bijection)
// ============================================================================

#[derive(Clone)]
struct ScaleGraph {
    r: [Vec<f64>; 3], // R^(1), R^(2), R^(3) : v_size * v_size
    w: [Vec<f64>; 3], // Poids de lignes normalisés (somme = 1 par échelle)
    h_marg: [f64; 3], // Entropie croisée de référence unigramme P_marg par échelle
}

#[derive(Clone)]
struct RelationalView {
    vocab: Vec<u8>,
    seq: Vec<usize>,
    v_size: usize,
    p_marg: Vec<f64>,
    graph: ScaleGraph,
    shuf_graph: ScaleGraph, // Contre-exemple E- à marginales P_marg exactement identiques
    node_init: Vec<[f64; DIM]>,
}

fn build_scale_graph(seq: &[usize], v_size: usize, p_marg: &[f64]) -> ScaleGraph {
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
    fn from_bytes(bytes: &[u8]) -> Self {
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

    fn from_indexed(vocab: Vec<u8>, seq: Vec<usize>) -> Self {
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

    fn subslice_same_vocab(&self, start: usize, end: usize) -> Self {
        let s = start.min(self.seq.len());
        let e = end.min(self.seq.len()).max(s);
        Self::from_indexed(self.vocab.clone(), self.seq[s..e].to_vec())
    }
}

// ============================================================================
// 4. Architecture connexionniste hiérarchique top-down Phisharp V2.1
//    Liaisons dynamiques A(t), E_emit(t), G(t) co-évoluant avec H1(t) et H2(t)
// ============================================================================

const OFF_W_LAT1: usize = 0;
const OFF_W_LAT2: usize = OFF_W_LAT1 + K1 * K1;
const OFF_W_UP_12: usize = OFF_W_LAT2 + K2 * K2;
const OFF_W_DOWN_21: usize = OFF_W_UP_12 + K2 * K1;

const OFF_U_IN: usize = OFF_W_DOWN_21 + K1 * K2;
const OFF_U_OBS_1: usize = OFF_U_IN + DIM * DIM;
const OFF_U_G_OUT: usize = OFF_U_OBS_1 + DIM * DIM;
const OFF_U_G_IN: usize = OFF_U_G_OUT + DIM * DIM;
const OFF_U_LAT1: usize = OFF_U_G_IN + DIM * DIM;
const OFF_U_UP_12: usize = OFF_U_LAT1 + DIM * DIM;
const OFF_U_LAT2: usize = OFF_U_UP_12 + DIM * DIM;
const OFF_U_DOWN_21: usize = OFF_U_LAT2 + DIM * DIM;
const OFF_U_META: usize = OFF_U_DOWN_21 + DIM * DIM;
const OFF_U_G: usize = OFF_U_META + DIM * DIM;
const OFF_U_DOWN_10: usize = OFF_U_G + DIM * DIM;
const OFF_U_EMIT: usize = OFF_U_DOWN_10 + DIM * DIM;
const OFF_U_SYM: usize = OFF_U_EMIT + DIM * DIM;
const OFF_U_REF1: usize = OFF_U_SYM + DIM * DIM;
const OFF_U_REF2: usize = OFF_U_REF1 + DIM * DIM;

const OFF_B1: usize = OFF_U_REF2 + DIM * DIM;
const OFF_B2: usize = OFF_B1 + DIM;
const OFF_GATES: usize = OFF_B2 + DIM;
const PARAM_COUNT: usize = OFF_GATES + 8;

const SCALE_W: [f64; 3] = [0.50_f64, 0.30_f64, 0.20_f64];

#[derive(Clone)]
struct HierState {
    h1: Vec<[f64; DIM]>,
    h2: Vec<[f64; DIM]>,
    l_a: Vec<f64>, // v_size * k1 : potentiels dynamiques d'ancrage L0 -> L1
    l_e: Vec<f64>, // k1 * v_size : potentiels dynamiques d'émission L1 -> L0
    l_g: Vec<f64>, // k1 * k1     : potentiels dynamiques de réécriture latente
}

impl HierState {
    fn zeros(v_size: usize, k1: usize, k2: usize) -> Self {
        Self {
            h1: vec![[0.0_f64; DIM]; k1],
            h2: vec![[0.0_f64; DIM]; k2],
            l_a: vec![0.0_f64; v_size * k1],
            l_e: vec![0.0_f64; k1 * v_size],
            l_g: vec![0.0_f64; k1 * k1],
        }
    }

    /// Initialisation douce strictement invariante au renommage bijectif d'alphabet
    /// (dépend uniquement des indices d'ordre d'apparition u, k).
    fn seeded(v_size: usize) -> Self {
        let mut h1 = vec![[0.0_f64; DIM]; K1];
        for (k, v) in h1.iter_mut().enumerate() {
            for d in 0..DIM {
                v[d] = 0.12_f64 * (((k + 1) * (d + 2)) as f64 * 1.13_f64).sin();
            }
            rms_norm(v);
        }
        let mut h2 = vec![[0.0_f64; DIM]; K2];
        for (m, v) in h2.iter_mut().enumerate() {
            for d in 0..DIM {
                v[d] = 0.12_f64 * (((m + 3) * (d + 1)) as f64 * 1.29_f64).cos();
            }
            rms_norm(v);
        }

        // Brisure de symétrie légère entre symboles et causes latentes (permet de résoudre
        // les graphes réguliers/cycles tout en restant 100 % invariant par bijection).
        let mut l_a = vec![0.0_f64; v_size * K1];
        for u in 0..v_size {
            for k in 0..K1 {
                l_a[u * K1 + k] = 0.18_f64 * (((u + 1) * (k + 2)) as f64 * 1.37_f64).sin();
            }
            center_row(&mut l_a[u * K1..(u + 1) * K1]);
        }

        let mut l_e = vec![0.0_f64; K1 * v_size];
        for k in 0..K1 {
            for u in 0..v_size {
                l_e[k * v_size + u] = 0.18_f64 * (((k + 1) * (u + 3)) as f64 * 1.19_f64).cos();
            }
            center_row(&mut l_e[k * v_size..(k + 1) * v_size]);
        }

        let l_g = vec![0.0_f64; K1 * K1];

        Self {
            h1,
            h2,
            l_a,
            l_e,
            l_g,
        }
    }

    fn with_dead_l1_node(&self, v_size: usize) -> Self {
        let old_k1 = self.h1.len();
        let new_k1 = old_k1 + 1;
        let mut h1 = self.h1.clone();
        h1.push([0.0_f64; DIM]);

        let mut l_a = vec![0.0_f64; v_size * new_k1];
        for u in 0..v_size {
            for k in 0..old_k1 {
                l_a[u * new_k1 + k] = self.l_a[u * old_k1 + k];
            }
        }
        let mut l_e = vec![0.0_f64; new_k1 * v_size];
        for k in 0..old_k1 {
            for u in 0..v_size {
                l_e[k * v_size + u] = self.l_e[k * v_size + u];
            }
        }
        let mut l_g = vec![0.0_f64; new_k1 * new_k1];
        for a in 0..old_k1 {
            for b in 0..old_k1 {
                l_g[a * new_k1 + b] = self.l_g[a * old_k1 + b];
            }
        }

        Self {
            h1,
            h2: self.h2.clone(),
            l_a,
            l_e,
            l_g,
        }
    }

    fn with_cloned_l1_node(&self, v_size: usize, src_k: usize) -> Self {
        let old_k1 = self.h1.len();
        let new_k1 = old_k1 + 1;
        let mut h1 = self.h1.clone();
        h1.push(self.h1[src_k]);

        let mut l_a = vec![0.0_f64; v_size * new_k1];
        for u in 0..v_size {
            for k in 0..old_k1 {
                l_a[u * new_k1 + k] = self.l_a[u * old_k1 + k];
            }
            l_a[u * new_k1 + old_k1] = self.l_a[u * old_k1 + src_k];
        }
        let mut l_e = vec![0.0_f64; new_k1 * v_size];
        for k in 0..old_k1 {
            for u in 0..v_size {
                l_e[k * v_size + u] = self.l_e[k * v_size + u];
            }
        }
        for u in 0..v_size {
            l_e[old_k1 * v_size + u] = self.l_e[src_k * v_size + u];
        }
        let mut l_g = vec![0.0_f64; new_k1 * new_k1];
        for a in 0..old_k1 {
            for b in 0..old_k1 {
                l_g[a * new_k1 + b] = self.l_g[a * old_k1 + b];
            }
            l_g[a * new_k1 + old_k1] = self.l_g[a * old_k1 + src_k];
            l_g[old_k1 * new_k1 + a] = self.l_g[src_k * old_k1 + a];
        }
        l_g[old_k1 * new_k1 + old_k1] = self.l_g[src_k * old_k1 + src_k];

        Self {
            h1,
            h2: self.h2.clone(),
            l_a,
            l_e,
            l_g,
        }
    }

    fn with_cloned_l2_node(&self, src_m: usize) -> Self {
        let mut out = self.clone();
        out.h2.push(self.h2[src_m]);
        out
    }
}

struct TopDownUnfold {
    a_ground: Vec<f64>,   // V * k1
    g_op: Vec<f64>,       // k1 * k1
    emit: Vec<f64>,       // k1 * V
    ge: [Vec<f64>; 3],    // G^s * E_emit (k1 * V)
    ag: [Vec<f64>; 3],    // A * G^s      (V * k1)
    r_hat: [Vec<f64>; 3], // R_hat^(1..3) = A * G^s * E_emit (V * V)
}

#[derive(Clone)]
struct AbductionTrace {
    state: HierState,
    snapshots: Vec<(usize, HierState)>,
    phi_history: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
struct PhiReport {
    generativity: f64,
    discrimination: f64,
    projection: f64,
    mip_integration: f64,
    diff_stability: f64,
    residual_norm: f64,
    phi_sharp: f64,
    shaping_score: f64,
}

#[derive(Clone)]
struct PhisharpModel {
    params: Vec<f64>,
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0_f64 / (1.0_f64 + (-x.clamp(-20.0_f64, 20.0_f64)).exp())
}

#[inline]
fn mat_vec(mat: &[f64], x: &[f64; DIM]) -> [f64; DIM] {
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
fn dot(a: &[f64; DIM], b: &[f64; DIM]) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..DIM {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn vec_norm(v: &[f64; DIM]) -> f64 {
    dot(v, v).sqrt()
}

#[inline]
fn vec_dist(a: &[f64; DIM], b: &[f64; DIM]) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..DIM {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

#[inline]
fn rms_norm(v: &mut [f64; DIM]) {
    let mean_sq = dot(v, v) / (DIM as f64);
    if mean_sq < 1e-18_f64 {
        return;
    }
    let denom = (mean_sq + 0.25_f64).sqrt();
    for x in v.iter_mut() {
        *x /= denom;
    }
}

#[inline]
fn center_row(row: &mut [f64]) {
    if row.is_empty() {
        return;
    }
    let mean = row.iter().sum::<f64>() / (row.len() as f64);
    for x in row.iter_mut() {
        *x = (*x - mean).clamp(-6.0_f64, 6.0_f64);
    }
}

fn softmax_slice(logits: &[f64], out: &mut [f64]) {
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

fn mat_mul(a: &[f64], b: &[f64], m: usize, p: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        let ai = i * p;
        let ci = i * n;
        for k in 0..p {
            let aik = a[ai + k];
            if aik == 0.0_f64 {
                continue;
            }
            let bk = k * n;
            for j in 0..n {
                c[ci + j] += aik * b[bk + j];
            }
        }
    }
    c
}

impl PhisharpModel {
    fn new_random(rng: &mut Rng) -> Self {
        let mut params = vec![0.0_f64; PARAM_COUNT];
        for (i, p) in params.iter_mut().enumerate() {
            let scale = if i < OFF_U_IN {
                0.45_f64
            } else if i < OFF_B1 {
                0.35_f64
            } else {
                0.15_f64
            };
            *p = rng.next_normal() * scale;
        }
        Self { params }
    }

    fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut buf = Vec::with_capacity(16 + self.params.len() * 8);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(self.params.len() as u64).to_le_bytes());
        for &p in &self.params {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        fs::write(path, buf)
    }

    fn load(path: &Path) -> std::io::Result<Self> {
        let data = fs::read(path)?;
        if data.len() < 16 || &data[0..8] != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Format de modèle .phis V2.1 invalide",
            ));
        }
        let count = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        if count != PARAM_COUNT || data.len() != 16 + count * 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Nombre de paramètres V2.1 incompatible",
            ));
        }
        let mut params = Vec::with_capacity(count);
        for chunk in data[16..].chunks_exact(8) {
            params.push(f64::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(Self { params })
    }

    #[inline]
    fn w_lat1(&self, dst: usize, src: usize) -> f64 {
        if dst < K1 && src < K1 {
            self.params[OFF_W_LAT1 + dst * K1 + src]
        } else {
            0.0_f64
        }
    }
    #[inline]
    fn w_lat2(&self, dst: usize, src: usize) -> f64 {
        if dst < K2 && src < K2 {
            self.params[OFF_W_LAT2 + dst * K2 + src]
        } else {
            0.0_f64
        }
    }
    #[inline]
    fn w_up_12(&self, dst2: usize, src1: usize) -> f64 {
        if dst2 < K2 && src1 < K1 {
            self.params[OFF_W_UP_12 + dst2 * K1 + src1]
        } else {
            0.0_f64
        }
    }
    #[inline]
    fn w_down_21(&self, dst1: usize, src2: usize) -> f64 {
        if dst1 < K1 && src2 < K2 {
            self.params[OFF_W_DOWN_21 + dst1 * K2 + src2]
        } else {
            0.0_f64
        }
    }

    fn u_in(&self) -> &[f64] { &self.params[OFF_U_IN..OFF_U_OBS_1] }
    fn u_obs_1(&self) -> &[f64] { &self.params[OFF_U_OBS_1..OFF_U_G_OUT] }
    fn u_g_out(&self) -> &[f64] { &self.params[OFF_U_G_OUT..OFF_U_G_IN] }
    fn u_g_in(&self) -> &[f64] { &self.params[OFF_U_G_IN..OFF_U_LAT1] }
    fn u_lat1(&self) -> &[f64] { &self.params[OFF_U_LAT1..OFF_U_UP_12] }
    fn u_up_12(&self) -> &[f64] { &self.params[OFF_U_UP_12..OFF_U_LAT2] }
    fn u_lat2(&self) -> &[f64] { &self.params[OFF_U_LAT2..OFF_U_DOWN_21] }
    fn u_down_21(&self) -> &[f64] { &self.params[OFF_U_DOWN_21..OFF_U_META] }
    fn u_meta(&self) -> &[f64] { &self.params[OFF_U_META..OFF_U_G] }
    fn u_g(&self) -> &[f64] { &self.params[OFF_U_G..OFF_U_DOWN_10] }
    fn u_down_10(&self) -> &[f64] { &self.params[OFF_U_DOWN_10..OFF_U_EMIT] }
    fn u_emit(&self) -> &[f64] { &self.params[OFF_U_EMIT..OFF_U_SYM] }
    fn u_sym(&self) -> &[f64] { &self.params[OFF_U_SYM..OFF_U_REF1] }
    fn u_ref1(&self) -> &[f64] { &self.params[OFF_U_REF1..OFF_U_REF2] }
    fn u_ref2(&self) -> &[f64] { &self.params[OFF_U_REF2..OFF_B1] }
    fn b1(&self) -> &[f64] { &self.params[OFF_B1..OFF_B2] }
    fn b2(&self) -> &[f64] { &self.params[OFF_B2..OFF_GATES] }
    fn gates(&self) -> &[f64] { &self.params[OFF_GATES..PARAM_COUNT] }

    /// Déploiement vertical pur H = (H1, H2, L_A, L_E, L_G) ==> R_hat^(1..3) = A * G^s * E_emit.
    /// Zéro lecture gauche-droite de E. Respecte les coupes MIP (cut_l1 et cut_vertical).
    fn unfold_top_down(
        &self,
        node_init: &[[f64; DIM]],
        state: &HierState,
        cut_l1: Option<u32>,
        cut_vertical: bool,
    ) -> TopDownUnfold {
        let v = node_init.len();
        let k1 = state.h1.len();
        let k2 = state.h2.len();

        let mut h1_norms = vec![0.0_f64; k1];
        let mut any_active = false;
        for k in 0..k1 {
            let n = vec_norm(&state.h1[k]);
            h1_norms[k] = n;
            if n > 1e-9_f64 {
                any_active = true;
            }
        }

        if v == 0 || k1 == 0 || !any_active {
            let unif = if v > 0 { 1.0_f64 / (v as f64) } else { 0.0_f64 };
            let r = vec![unif; v * v];
            return TopDownUnfold {
                a_ground: vec![if k1 > 0 { 1.0_f64 / (k1 as f64) } else { 0.0_f64 }; v * k1],
                g_op: vec![if k1 > 0 { 1.0_f64 / (k1 as f64) } else { 0.0_f64 }; k1 * k1],
                emit: vec![unif; k1 * v],
                ge: [
                    vec![unif; k1 * v],
                    vec![unif; k1 * v],
                    vec![unif; k1 * v],
                ],
                ag: [
                    vec![if k1 > 0 { 1.0_f64 / (k1 as f64) } else { 0.0_f64 }; v * k1],
                    vec![if k1 > 0 { 1.0_f64 / (k1 as f64) } else { 0.0_f64 }; v * k1],
                    vec![if k1 > 0 { 1.0_f64 / (k1 as f64) } else { 0.0_f64 }; v * k1],
                ],
                r_hat: [r.clone(), r.clone(), r],
            };
        }

        // 1. Modulation verticale L2 -> L1 (non-réciproque)
        let mut h1_eff = state.h1.clone();
        let mut g2_ctx = [0.0_f64; DIM];

        if !cut_vertical && k2 > 0 {
            let mut h2_down = vec![[0.0_f64; DIM]; k2];
            let inv_k2 = 1.0_f64 / (k2 as f64);
            for m in 0..k2 {
                h2_down[m] = mat_vec(self.u_down_21(), &state.h2[m]);
                let meta_m = mat_vec(self.u_meta(), &state.h2[m]);
                for d in 0..DIM {
                    g2_ctx[d] += meta_m[d] * inv_k2;
                }
            }
            for d in 0..DIM {
                g2_ctx[d] = g2_ctx[d].tanh();
            }

            for k in 0..k1 {
                if h1_norms[k] <= 1e-9_f64 {
                    continue;
                }
                let mut td = [0.0_f64; DIM];
                for m in 0..k2 {
                    let w = self.w_down_21(k, m);
                    if w != 0.0_f64 {
                        for d in 0..DIM {
                            td[d] += w * h2_down[m][d];
                        }
                    }
                }
                for d in 0..DIM {
                    h1_eff[k][d] += 0.65_f64 * td[d].tanh();
                }
                rms_norm(&mut h1_eff[k]);
            }
        }

        // 2. Opérateur latent non-réciproque G (k1 x k1) gouverné par (H1, H2, L_G)
        let mut g_op = vec![0.0_f64; k1 * k1];
        let mut row_logits = vec![0.0_f64; k1];

        for a in 0..k1 {
            if h1_norms[a] <= 1e-9_f64 {
                g_op[a * k1 + a] = 1.0_f64;
                continue;
            }
            let ug_a = mat_vec(self.u_g(), &h1_eff[a]);
            let mut qa = [0.0_f64; DIM];
            for d in 0..DIM {
                qa[d] = ug_a[d] + 0.55_f64 * h1_eff[a][d] * g2_ctx[d];
            }
            let side_a = cut_l1.map(|m| ((m >> a.min(30)) & 1) != 0);

            for b in 0..k1 {
                if h1_norms[b] <= 1e-9_f64 {
                    row_logits[b] = -1e9_f64;
                    continue;
                }
                if let Some(sa) = side_a {
                    let sb = ((cut_l1.unwrap() >> b.min(30)) & 1) != 0;
                    if sa != sb {
                        row_logits[b] = -1e9_f64;
                        continue;
                    }
                }
                let active = (h1_norms[a] * h1_norms[b]).tanh();
                let meta_gate = if cut_vertical {
                    0.50_f64
                } else {
                    0.35_f64 + 1.25_f64 * sigmoid(dot(&g2_ctx, &h1_eff[a]) + dot(&g2_ctx, &h1_eff[b]))
                };
                row_logits[b] = dot(&qa, &h1_eff[b])
                    + self.w_lat1(b, a) * active
                    + meta_gate * state.l_g[a * k1 + b];
            }
            softmax_slice(&row_logits, &mut g_op[a * k1..(a + 1) * k1]);
        }

        let g2 = mat_mul(&g_op, &g_op, k1, k1, k1);
        let g3 = mat_mul(&g2, &g_op, k1, k1, k1);

        // 3. Ancrage relationnel dynamique A (V x k1) : L_A + couplage (node_init, H1)
        let mut a_ground = vec![0.0_f64; v * k1];
        let mut sym_in = vec![[0.0_f64; DIM]; v];
        for u in 0..v {
            sym_in[u] = mat_vec(self.u_in(), &node_init[u]);
            for k in 0..k1 {
                if h1_norms[k] <= 1e-9_f64 {
                    row_logits[k] = -1e9_f64;
                } else {
                    row_logits[k] = state.l_a[u * k1 + k] + 0.45_f64 * dot(&sym_in[u], &h1_eff[k]);
                }
            }
            softmax_slice(&row_logits, &mut a_ground[u * k1..(u + 1) * k1]);
        }

        // 4. Émission top-down dynamique E_emit (k1 x V) : L_E + couplage (H1, node_init)
        let mut z_sym = vec![[0.0_f64; DIM]; v];
        let mut u_sym_x = vec![[0.0_f64; DIM]; v];
        for u in 0..v {
            let mut h_mix = [0.0_f64; DIM];
            for k in 0..k1 {
                let w = a_ground[u * k1 + k];
                for d in 0..DIM {
                    h_mix[d] += w * h1_eff[k][d];
                }
            }
            let proj = mat_vec(self.u_down_10(), &h_mix);
            for d in 0..DIM {
                z_sym[u][d] = (proj[d] + 0.50_f64 * proj[d] * node_init[u][d]).tanh();
            }
            u_sym_x[u] = mat_vec(self.u_sym(), &node_init[u]);
        }

        let mut emit = vec![0.0_f64; k1 * v];
        let mut emit_logits = vec![0.0_f64; v];
        for k in 0..k1 {
            if h1_norms[k] <= 1e-9_f64 {
                let unif = 1.0_f64 / (v as f64);
                for u in 0..v {
                    emit[k * v + u] = unif;
                }
                continue;
            }
            let ek = mat_vec(self.u_emit(), &h1_eff[k]);
            let act_k = h1_norms[k].tanh();
            for u in 0..v {
                let mut target_u = [0.0_f64; DIM];
                for d in 0..DIM {
                    target_u[d] = z_sym[u][d] + act_k * u_sym_x[u][d];
                }
                emit_logits[u] = state.l_e[k * v + u] + 0.45_f64 * dot(&ek, &target_u);
            }
            softmax_slice(&emit_logits, &mut emit[k * v..(k + 1) * v]);
        }

        // 5. Déploiement multi-échelle vertical : R_hat^(s) = A * G^s * E_emit
        let ge1 = mat_mul(&g_op, &emit, k1, k1, v);
        let ge2 = mat_mul(&g2, &emit, k1, k1, v);
        let ge3 = mat_mul(&g3, &emit, k1, k1, v);

        let ag1 = mat_mul(&a_ground, &g_op, v, k1, k1);
        let ag2 = mat_mul(&a_ground, &g2, v, k1, k1);
        let ag3 = mat_mul(&a_ground, &g3, v, k1, k1);

        let r_hat1 = mat_mul(&a_ground, &ge1, v, k1, v);
        let r_hat2 = mat_mul(&a_ground, &ge2, v, k1, v);
        let r_hat3 = mat_mul(&a_ground, &ge3, v, k1, v);

        TopDownUnfold {
            a_ground,
            g_op,
            emit,
            ge: [ge1, ge2, ge3],
            ag: [ag1, ag2, ag3],
            r_hat: [r_hat1, r_hat2, r_hat3],
        }
    }

    /// Une étape abductive hiérarchique : co-évolution de (L_A, L_E, L_G, H1, H2)
    /// à partir du résidu matriciel multi-échelle Δ^(s) = R^(s) - R_hat^(s).
    fn abductive_step(
        &self,
        view: &RelationalView,
        state: &HierState,
        use_refutation: bool,
        cut_l1: Option<u32>,
        cut_l2: Option<u32>,
        cut_vertical: bool,
    ) -> HierState {
        let v = view.v_size;
        let k1 = state.h1.len();
        let k2 = state.h2.len();

        let unfold = self.unfold_top_down(&view.node_init, state, cut_l1, cut_vertical);
        let res_norm = multi_scale_l1(&view.graph, &unfold.r_hat, v);

        // 1. Résidu matriciel multi-échelle Δ^(s) = R^(s) - R_hat^(s)
        let mut delta = [
            vec![0.0_f64; v * v],
            vec![0.0_f64; v * v],
            vec![0.0_f64; v * v],
        ];
        let mut delta_obs = vec![[0.0_f64; DIM]; v];
        for s in 0..3 {
            let sw = SCALE_W[s];
            for u in 0..v {
                if view.graph.w[s][u] <= 0.0_f64 {
                    continue;
                }
                let base = u * v;
                for w in 0..v {
                    let diff = view.graph.r[s][base + w] - unfold.r_hat[s][base + w];
                    delta[s][base + w] = diff;
                    if diff.abs() > 1e-12_f64 {
                        let c = sw * diff;
                        for d in 0..DIM {
                            delta_obs[u][d] += c * view.node_init[w][d];
                        }
                    }
                }
            }
        }

        // 2. Projections relationnelles du résidu vers L_A, L_E et L_G (100 % non-réciproques)
        let mut grad_a = vec![0.0_f64; v * k1];
        let mut grad_e = vec![0.0_f64; k1 * v];
        let mut grad_g = vec![0.0_f64; k1 * k1];

        // Masses d'usage par cause latente pour invariance d'échelle par rapport à V et K1
        let mut mass_a = [vec![0.0_f64; k1], vec![0.0_f64; k1], vec![0.0_f64; k1]];
        let mut mass_ag = [vec![0.0_f64; k1], vec![0.0_f64; k1], vec![0.0_f64; k1]];
        for s in 0..3 {
            for u in 0..v {
                let wu = view.graph.w[s][u];
                if wu <= 0.0_f64 {
                    continue;
                }
                for k in 0..k1 {
                    mass_a[s][k] += wu * unfold.a_ground[u * k1 + k];
                    mass_ag[s][k] += wu * unfold.ag[s][u * k1 + k];
                }
            }
        }

        // (a) Flux résiduel sur L_A : Δ^(s) * (G^s E_emit)^T
        for s in 0..3 {
            let sw = SCALE_W[s];
            for u in 0..v {
                if view.graph.w[s][u] <= 0.0_f64 {
                    continue;
                }
                let u_base = u * v;
                for k in 0..k1 {
                    let k_base = k * v;
                    let mut acc = 0.0_f64;
                    for w in 0..v {
                        acc += delta[s][u_base + w] * unfold.ge[s][k_base + w];
                    }
                    grad_a[u * k1 + k] += sw * acc;
                }
            }
        }

        // (b) Flux résiduel sur L_E : (A G^s)^T * diag(w^(s)) * Δ^(s) (normalisé par masse)
        for s in 0..3 {
            let sw = SCALE_W[s];
            for k in 0..k1 {
                let inv_m = sw / (mass_ag[s][k] + 0.02_f64);
                let k_base = k * v;
                for u in 0..v {
                    let wu = view.graph.w[s][u];
                    if wu <= 0.0_f64 {
                        continue;
                    }
                    let coeff = inv_m * wu * unfold.ag[s][u * k1 + k];
                    if coeff.abs() <= 1e-12_f64 {
                        continue;
                    }
                    let u_base = u * v;
                    for w in 0..v {
                        grad_e[k_base + w] += coeff * delta[s][u_base + w];
                    }
                }
            }
        }

        // (c) Flux résiduel dans l'espace latent G : projets gauche/droite aux échelles 1..3
        for s in 0..3 {
            let sw = SCALE_W[s];
            // M_delta[u, b] = sum_w Δ^(s)[u, w] * E_emit[b, w]
            let mut d_emit = vec![0.0_f64; v * k1];
            for u in 0..v {
                if view.graph.w[s][u] <= 0.0_f64 {
                    continue;
                }
                let u_base = u * v;
                for b in 0..k1 {
                    let b_base = b * v;
                    let mut acc = 0.0_f64;
                    for w in 0..v {
                        acc += delta[s][u_base + w] * unfold.emit[b_base + w];
                    }
                    d_emit[u * k1 + b] = acc;
                }
            }

            if s == 0 {
                for a in 0..k1 {
                    let inv_m = sw / (mass_a[0][a] + 0.02_f64);
                    for u in 0..v {
                        let wu = view.graph.w[0][u];
                        if wu <= 0.0_f64 {
                            continue;
                        }
                        let c = inv_m * wu * unfold.a_ground[u * k1 + a];
                        for b in 0..k1 {
                            grad_g[a * k1 + b] += c * d_emit[u * k1 + b];
                        }
                    }
                }
            } else {
                // Pour s = 1 (lag 2) et s = 2 (lag 3) : contributions amont (A G^{s-1}) et aval (G^{s-1} E)
                let left_mat = &unfold.ag[s - 1];
                let left_mass = &mass_ag[s - 1];
                let right_mat = &unfold.ge[s - 1];

                let mut d_right = vec![0.0_f64; v * k1];
                for u in 0..v {
                    if view.graph.w[s][u] <= 0.0_f64 {
                        continue;
                    }
                    let u_base = u * v;
                    for b in 0..k1 {
                        let b_base = b * v;
                        let mut acc = 0.0_f64;
                        for w in 0..v {
                            acc += delta[s][u_base + w] * right_mat[b_base + w];
                        }
                        d_right[u * k1 + b] = acc;
                    }
                }

                for a in 0..k1 {
                    let inv_ml = 0.5_f64 * sw / (left_mass[a] + 0.02_f64);
                    let inv_ma = 0.5_f64 * sw / (mass_a[s][a] + 0.02_f64);
                    for u in 0..v {
                        let wu = view.graph.w[s][u];
                        if wu <= 0.0_f64 {
                            continue;
                        }
                        let cl = inv_ml * wu * left_mat[u * k1 + a];
                        let ca = inv_ma * wu * unfold.a_ground[u * k1 + a];
                        for b in 0..k1 {
                            grad_g[a * k1 + b] += cl * d_emit[u * k1 + b] + ca * d_right[u * k1 + b];
                        }
                    }
                }
            }
        }

        if let Some(mask) = cut_l1 {
            for a in 0..k1 {
                let sa = ((mask >> a.min(30)) & 1) != 0;
                for b in 0..k1 {
                    let sb = ((mask >> b.min(30)) & 1) != 0;
                    if sa != sb {
                        grad_g[a * k1 + b] = 0.0_f64;
                    }
                }
            }
        }

        // 3. Portes d'intégration pilotées par le résidu (contractent en attracteur stable)
        let g = self.gates();
        let alpha1 = 0.14_f64 + 0.46_f64 * sigmoid(g[0] * res_norm + g[1]);
        let alpha2 = 0.14_f64 + 0.46_f64 * sigmoid(g[2] * res_norm + g[3]);
        let eta_ae = 0.60_f64 + 3.40_f64 * sigmoid(g[4] * res_norm + g[5]);
        let eta_g = 0.50_f64 + 3.00_f64 * sigmoid(g[6] * res_norm + g[7]);

        // Mise à jour des liaisons dynamiques L_A, L_E, L_G avec amortissement doux + centrage
        let mut next_l_a = state.l_a.clone();
        for u in 0..v {
            let active = view.graph.w[0][u] > 0.0_f64
                || view.graph.w[1][u] > 0.0_f64
                || view.graph.w[2][u] > 0.0_f64;
            for k in 0..k1 {
                let idx = u * k1 + k;
                if active {
                    next_l_a[idx] = 0.94_f64 * state.l_a[idx] + eta_ae * grad_a[idx];
                } else {
                    next_l_a[idx] = 0.94_f64 * state.l_a[idx];
                }
            }
            center_row(&mut next_l_a[u * k1..(u + 1) * k1]);
        }

        let mut next_l_e = state.l_e.clone();
        for k in 0..k1 {
            for w in 0..v {
                let idx = k * v + w;
                next_l_e[idx] = 0.94_f64 * state.l_e[idx] + eta_ae * grad_e[idx];
            }
            center_row(&mut next_l_e[k * v..(k + 1) * v]);
        }

        let mut next_l_g = state.l_g.clone();
        for a in 0..k1 {
            let sa = cut_l1.map(|m| ((m >> a.min(30)) & 1) != 0);
            for b in 0..k1 {
                let idx = a * k1 + b;
                if let Some(side_a) = sa {
                    let side_b = ((cut_l1.unwrap() >> b.min(30)) & 1) != 0;
                    if side_a != side_b {
                        next_l_g[idx] = 0.0_f64;
                        continue;
                    }
                }
                next_l_g[idx] = 0.94_f64 * state.l_g[idx] + eta_g * grad_g[idx];
            }
            center_row(&mut next_l_g[a * k1..(a + 1) * k1]);
        }

        // 4. Mise à jour Niveau 1 (résidu latent Δ_G non-réciproque + Δ_obs + W_lat1 + L2->L1)
        let mut next_h1 = vec![[0.0_f64; DIM]; k1];
        let b1 = self.b1();

        let mut lat1_msg = vec![[0.0_f64; DIM]; k1];
        let mut g_out_msg = vec![[0.0_f64; DIM]; k1];
        let mut g_in_msg = vec![[0.0_f64; DIM]; k1];
        for src in 0..k1 {
            lat1_msg[src] = mat_vec(self.u_lat1(), &state.h1[src]);
            g_out_msg[src] = mat_vec(self.u_g_out(), &state.h1[src]);
            g_in_msg[src] = mat_vec(self.u_g_in(), &state.h1[src]);
        }
        let mut down2_msg = vec![[0.0_f64; DIM]; k2];
        for m in 0..k2 {
            down2_msg[m] = mat_vec(self.u_down_21(), &state.h2[m]);
        }
        let mut obs1_msg = vec![[0.0_f64; DIM]; v];
        for u in 0..v {
            obs1_msg[u] = mat_vec(self.u_obs_1(), &delta_obs[u]);
        }

        for dst in 0..k1 {
            if vec_norm(&state.h1[dst]) <= 1e-9_f64 {
                continue;
            }
            let mut acc = [0.0_f64; DIM];

            for u in 0..v {
                let wu = view.graph.w[0][u];
                if wu <= 0.0_f64 {
                    continue;
                }
                let coeff = wu * (0.25_f64 + 0.75_f64 * unfold.a_ground[u * k1 + dst]);
                for d in 0..DIM {
                    acc[d] += coeff * obs1_msg[u][d];
                }
            }

            for src in 0..k1 {
                if let Some(mask) = cut_l1 {
                    let s_dst = ((mask >> dst.min(30)) & 1) != 0;
                    let s_src = ((mask >> src.min(30)) & 1) != 0;
                    if s_dst != s_src {
                        continue;
                    }
                }
                let w = self.w_lat1(dst, src);
                let dg_out = grad_g[dst * k1 + src];
                let dg_in = grad_g[src * k1 + dst];
                for d in 0..DIM {
                    acc[d] += w * lat1_msg[src][d]
                        + 1.20_f64 * dg_out * g_out_msg[src][d]
                        + 1.20_f64 * dg_in * g_in_msg[src][d];
                }
            }

            if !cut_vertical {
                for m in 0..k2 {
                    let w = self.w_down_21(dst, m);
                    if w != 0.0_f64 {
                        for d in 0..DIM {
                            acc[d] += 0.60_f64 * w * down2_msg[m][d];
                        }
                    }
                }
            }

            for d in 0..DIM {
                next_h1[dst][d] =
                    (1.0_f64 - alpha1) * state.h1[dst][d] + alpha1 * (acc[d] + b1[d]).tanh();
            }
            rms_norm(&mut next_h1[dst]);
        }

        // 5. Mise à jour Niveau 2 (réification pure de L1 et de son graphe G, zéro accès à L0)
        let mut next_h2 = vec![[0.0_f64; DIM]; k2];
        let b2 = self.b2();

        let mut e1_reified = vec![[0.0_f64; DIM]; k1];
        for k in 0..k1 {
            let mut flow = [0.0_f64; DIM];
            for j in 0..k1 {
                let gkj = unfold.g_op[k * k1 + j];
                for d in 0..DIM {
                    flow[d] += gkj * next_h1[j][d];
                }
            }
            let mut combined = [0.0_f64; DIM];
            for d in 0..DIM {
                combined[d] = next_h1[k][d]
                    + 0.50_f64 * flow[d]
                    + 0.45_f64 * (next_h1[k][d] - state.h1[k][d]);
            }
            e1_reified[k] = mat_vec(self.u_up_12(), &combined);
        }

        let mut lat2_msg = vec![[0.0_f64; DIM]; k2];
        for src in 0..k2 {
            lat2_msg[src] = mat_vec(self.u_lat2(), &state.h2[src]);
        }

        for dst in 0..k2 {
            if vec_norm(&state.h2[dst]) <= 1e-9_f64 {
                continue;
            }
            let mut acc = [0.0_f64; DIM];

            if !cut_vertical {
                for src1 in 0..k1 {
                    let w = self.w_up_12(dst, src1);
                    if w != 0.0_f64 {
                        for d in 0..DIM {
                            acc[d] += w * e1_reified[src1][d];
                        }
                    }
                }
            }

            for src in 0..k2 {
                if let Some(mask) = cut_l2 {
                    let s_dst = ((mask >> dst.min(30)) & 1) != 0;
                    let s_src = ((mask >> src.min(30)) & 1) != 0;
                    if s_dst != s_src {
                        continue;
                    }
                }
                let w = self.w_lat2(dst, src);
                if w != 0.0_f64 {
                    for d in 0..DIM {
                        acc[d] += w * lat2_msg[src][d];
                    }
                }
            }

            for d in 0..DIM {
                next_h2[dst][d] =
                    (1.0_f64 - alpha2) * state.h2[dst][d] + alpha2 * (acc[d] + b2[d]).tanh();
            }
            rms_norm(&mut next_h2[dst]);
        }

        let mut next_state = HierState {
            h1: next_h1,
            h2: next_h2,
            l_a: next_l_a,
            l_e: next_l_e,
            l_g: next_l_g,
        };

        // 6. Réfutation active conservatrice (teste une contre-hypothèse sur L1, L2 et G)
        if use_refutation
            && cut_l1.is_none()
            && cut_l2.is_none()
            && !cut_vertical
            && k1 > 0
            && k2 > 0
        {
            let mut peak1 = 0usize;
            let mut best_n1 = -1.0_f64;
            for (k, h) in next_state.h1.iter().enumerate() {
                let n = dot(h, h);
                if n > best_n1 {
                    best_n1 = n;
                    peak1 = k;
                }
            }
            let mut peak2 = 0usize;
            let mut best_n2 = -1.0_f64;
            for (m, h) in next_state.h2.iter().enumerate() {
                let n = dot(h, h);
                if n > best_n2 {
                    best_n2 = n;
                    peak2 = m;
                }
            }

            let mut refuted = next_state.clone();
            let p1 = mat_vec(self.u_ref1(), &refuted.h1[peak1]);
            for d in 0..DIM {
                refuted.h1[peak1][d] -= 0.35_f64 * p1[d].tanh();
            }
            rms_norm(&mut refuted.h1[peak1]);

            let p2 = mat_vec(self.u_ref2(), &refuted.h2[peak2]);
            for d in 0..DIM {
                refuted.h2[peak2][d] -= 0.35_f64 * p2[d].tanh();
            }
            rms_norm(&mut refuted.h2[peak2]);

            // Teste aussi la réfutation de l'arête dominante sortant de peak1 dans L_G
            let mut dom_b = 0usize;
            let mut dom_val = f64::NEG_INFINITY;
            for b in 0..k1 {
                let val = refuted.l_g[peak1 * k1 + b];
                if val > dom_val {
                    dom_val = val;
                    dom_b = b;
                }
            }
            refuted.l_g[peak1 * k1 + dom_b] -= 0.65_f64;
            center_row(&mut refuted.l_g[peak1 * k1..(peak1 + 1) * k1]);

            let gain_curr = multi_scale_gain(&view.graph, &unfold.r_hat, v);
            let unfold_base = self.unfold_top_down(&view.node_init, &next_state, None, false);
            let gain_base = multi_scale_gain(&view.graph, &unfold_base.r_hat, v);
            let unfold_ref = self.unfold_top_down(&view.node_init, &refuted, None, false);
            let gain_ref = multi_scale_gain(&view.graph, &unfold_ref.r_hat, v);

            if gain_ref > gain_base {
                next_state = refuted;
            } else if gain_base >= gain_curr {
                // Consolide la composante qui a survécu à la tentative de réfutation
                for d in 0..DIM {
                    let d1 = next_state.h1[peak1][d] - refuted.h1[peak1][d];
                    next_state.h1[peak1][d] += 0.14_f64 * d1;
                    let d2 = next_state.h2[peak2][d] - refuted.h2[peak2][d];
                    next_state.h2[peak2][d] += 0.14_f64 * d2;
                }
                rms_norm(&mut next_state.h1[peak1]);
                rms_norm(&mut next_state.h2[peak2]);
            } else {
                // Amortit tout dépassement pour stabiliser l'attracteur aux grands budgets
                for k in 0..k1 {
                    for d in 0..DIM {
                        next_state.h1[k][d] = 0.5_f64 * (state.h1[k][d] + next_state.h1[k][d]);
                    }
                    rms_norm(&mut next_state.h1[k]);
                }
                for m in 0..k2 {
                    for d in 0..DIM {
                        next_state.h2[m][d] = 0.5_f64 * (state.h2[m][d] + next_state.h2[m][d]);
                    }
                    rms_norm(&mut next_state.h2[m]);
                }
                for i in 0..next_state.l_a.len() {
                    next_state.l_a[i] = 0.5_f64 * (state.l_a[i] + next_state.l_a[i]);
                }
                for i in 0..next_state.l_e.len() {
                    next_state.l_e[i] = 0.5_f64 * (state.l_e[i] + next_state.l_e[i]);
                }
                for i in 0..next_state.l_g.len() {
                    next_state.l_g[i] = 0.5_f64 * (state.l_g[i] + next_state.l_g[i]);
                }
            }
        }

        next_state
    }

    fn run_abduction(
        &self,
        view: &RelationalView,
        budget: usize,
        use_refutation: bool,
        capture_steps: &[usize],
        record_phi: bool,
    ) -> AbductionTrace {
        let mut state = HierState::zeros(view.v_size, K1, K2);
        let mut snapshots = Vec::new();
        let mut phi_history = Vec::new();

        if capture_steps.contains(&0) {
            snapshots.push((0, state.clone()));
        }
        if record_phi {
            phi_history.push(
                self.evaluate_phi_state(view, &state, 0, use_refutation)
                    .phi_sharp,
            );
        }
        if budget == 0 {
            return AbductionTrace {
                state,
                snapshots,
                phi_history,
            };
        }

        state = HierState::seeded(view.v_size);
        for step in 1..=budget {
            state = self.abductive_step(view, &state, use_refutation, None, None, false);
            if capture_steps.contains(&step) {
                snapshots.push((step, state.clone()));
            }
            if record_phi {
                phi_history.push(
                    self.evaluate_phi_state(view, &state, step, use_refutation)
                        .phi_sharp,
                );
            }
        }

        AbductionTrace {
            state,
            snapshots,
            phi_history,
        }
    }

    fn evaluate_phi(
        &self,
        view: &RelationalView,
        budget: usize,
        use_refutation: bool,
    ) -> PhiReport {
        let trace = self.run_abduction(view, budget, use_refutation, &[], false);
        self.evaluate_phi_state(view, &trace.state, budget, use_refutation)
    }

    /// Substitut V2.1 de φ_sharp :
    /// - G, D, P mesurés strictement au-dessus des marginales P_marg
    /// - Combinaison sans compensation min(G, D, P)
    /// - MIP exacte hiérarchique + facteur d'unicité (annule nœuds morts et clones)
    /// - VERROU ANTI-GOODHART : poids STRICTEMENT NUL sur MIP tant que G, D ou P <= 0
    fn evaluate_phi_state(
        &self,
        view: &RelationalView,
        state: &HierState,
        budget: usize,
        use_refutation: bool,
    ) -> PhiReport {
        let v = view.v_size;
        if v <= 1 || view.seq.len() < 16 {
            return PhiReport {
                generativity: 0.0,
                discrimination: 0.0,
                projection: 0.0,
                mip_integration: 0.0,
                diff_stability: 0.0,
                residual_norm: 1.0,
                phi_sharp: 0.0,
                shaping_score: -1.0,
            };
        }

        let unfold = self.unfold_top_down(&view.node_init, state, None, false);

        // 1. Générativité multi-échelle au-dessus de H(P_marg)
        let gen_raw = multi_scale_gain(&view.graph, &unfold.r_hat, v);

        // 2. Discrimination : contraste face à E- (permuté à marginales P_marg identiques)
        let neg_gain = multi_scale_gain(&view.shuf_graph, &unfold.r_hat, v);
        let disc_raw = (gen_raw - neg_gain).clamp(-1.0_f64, 1.0_f64);

        // 3. Projection stricte : abduit sur la 1re moitié seule, déplie sur la 2e sans la lire
        let mid = view.seq.len() / 2;
        let first_half = view.subslice_same_vocab(0, mid);
        let second_half = view.subslice_same_vocab(mid, view.seq.len());
        let proj_raw = if budget == 0 || state.h1.is_empty() {
            0.0_f64
        } else {
            let tr_half = self.run_abduction(&first_half, budget, use_refutation, &[], false);
            let unfold_half = self.unfold_top_down(&first_half.node_init, &tr_half.state, None, false);
            multi_scale_gain(&second_half.graph, &unfold_half.r_hat, v)
        };

        // 4. Intégration causale exacte (MIP hiérarchique + unicité des unités causales)
        let k1 = state.h1.len();
        let k2 = state.h2.len();
        let mip_integration = if budget == 0 || k1 <= 1 || k2 <= 1 {
            0.0_f64
        } else {
            // Facteur d'unicité causale : une copie clonée sur L1 ou L2 donne distance = 0 -> MIP = 0
            let mut min_dist = f64::INFINITY;
            for a in 0..k1 {
                for b in (a + 1)..k1 {
                    let d = vec_dist(&state.h1[a], &state.h1[b]);
                    if d < min_dist {
                        min_dist = d;
                    }
                }
            }
            for a in 0..k2 {
                for b in (a + 1)..k2 {
                    let d = vec_dist(&state.h2[a], &state.h2[b]);
                    if d < min_dist {
                        min_dist = d;
                    }
                }
            }
            let uniq_factor = (min_dist * 3.0_f64).tanh().clamp(0.0_f64, 1.0_f64);

            if uniq_factor <= 1e-9_f64 {
                0.0_f64
            } else {
                let full_next = self.abductive_step(view, state, false, None, None, false);
                let pred_full = self.unfold_top_down(&view.node_init, &full_next, None, false);

                let mut min_div = f64::INFINITY;

                // (a) Toutes les bipartitions internes de Niveau 1
                let masks1 = 1u32 << (k1.min(10) - 1);
                for mask in 1..masks1 {
                    let cut_next =
                        self.abductive_step(view, state, false, Some(mask), None, false);
                    let pred_cut =
                        self.unfold_top_down(&view.node_init, &cut_next, Some(mask), false);
                    let div =
                        multi_scale_pred_tvd(&pred_full.r_hat, &pred_cut.r_hat, &view.graph.w, v);
                    if div < min_div {
                        min_div = div;
                    }
                }

                // (b) Toutes les bipartitions internes de Niveau 2
                let masks2 = 1u32 << (k2.min(10) - 1);
                for mask in 1..masks2 {
                    let cut_next =
                        self.abductive_step(view, state, false, None, Some(mask), false);
                    let pred_cut = self.unfold_top_down(&view.node_init, &cut_next, None, false);
                    let div =
                        multi_scale_pred_tvd(&pred_full.r_hat, &pred_cut.r_hat, &view.graph.w, v);
                    if div < min_div {
                        min_div = div;
                    }
                }

                // (c) Coupe verticale L1 <-> L2 (interdit un L2 décoratif)
                let vcut_next = self.abductive_step(view, state, false, None, None, true);
                let pred_vcut = self.unfold_top_down(&view.node_init, &vcut_next, None, true);
                let vdiv =
                    multi_scale_pred_tvd(&pred_full.r_hat, &pred_vcut.r_hat, &view.graph.w, v);
                if vdiv < min_div {
                    min_div = vdiv;
                }

                // (d) Ablation de chaque nœud individuel (annule les branches mortes/épiphénoménales)
                for drop1 in 0..k1 {
                    let mut abl = state.clone();
                    abl.h1[drop1] = [0.0_f64; DIM];
                    let pred_abl = self.unfold_top_down(&view.node_init, &abl, None, false);
                    let eff =
                        multi_scale_pred_tvd(&unfold.r_hat, &pred_abl.r_hat, &view.graph.w, v);
                    if eff < min_div {
                        min_div = eff;
                    }
                }
                for drop2 in 0..k2 {
                    let mut abl = state.clone();
                    abl.h2[drop2] = [0.0_f64; DIM];
                    let pred_abl = self.unfold_top_down(&view.node_init, &abl, None, false);
                    let eff =
                        multi_scale_pred_tvd(&unfold.r_hat, &pred_abl.r_hat, &view.graph.w, v);
                    if eff < min_div {
                        min_div = eff;
                    }
                }

                ((min_div * 10.0_f64).tanh().clamp(0.0_f64, 1.0_f64)) * uniq_factor
            }
        };

        // 5. Stabilité différentielle (réfutation partielle L1+L2 + déplacement central)
        let diff_stability = if budget == 0 || gen_raw <= 1e-6_f64 || k1 == 0 || k2 == 0 {
            0.0_f64
        } else {
            let mut ref_state = state.clone();
            for d in 0..DIM {
                ref_state.h1[0][d] *= 0.72_f64;
                ref_state.h2[0][d] *= 0.72_f64;
            }
            let pred_ref = self.unfold_top_down(&view.node_init, &ref_state, None, false);
            let g_ref = multi_scale_gain(&view.graph, &pred_ref.r_hat, v);
            let ref_stab = (g_ref / gen_raw).clamp(0.0_f64, 1.0_f64);

            let q1 = view.seq.len() / 4;
            let q3 = (3 * view.seq.len()) / 4;
            let shifted = view.subslice_same_vocab(q1, q3);
            let g_shift = multi_scale_gain(&shifted.graph, &unfold.r_hat, v);
            let shift_stab = (g_shift / gen_raw).clamp(0.0_f64, 1.0_f64);

            0.5_f64 * (ref_stab + shift_stab)
        };

        // 6. Résidu non expliqué
        let residual_norm = multi_scale_l1(&view.graph, &unfold.r_hat, v).clamp(0.0_f64, 1.0_f64);

        let g_c = gen_raw.clamp(0.0_f64, 1.0_f64);
        let d_c = disc_raw.clamp(0.0_f64, 1.0_f64);
        let p_c = proj_raw.clamp(0.0_f64, 1.0_f64);

        // Combinaison sans compensation : min(G, D, P)
        let base_min = g_c.min(d_c).min(p_c);
        let phi_sharp =
            base_min * (mip_integration * diff_stability).sqrt() * (1.0_f64 - 0.5_f64 * residual_norm);

        // VERROU ANTI-GOODHART STRICT :
        // Zéro récompense sur MIP ou diff_stability tant que G, D et P ne battent pas tous P_marg.
        let shaping_score = if gen_raw > 0.0_f64 && disc_raw > 0.0_f64 && proj_raw > 0.0_f64 {
            phi_sharp + 0.05_f64 * base_min
        } else {
            let min_raw = gen_raw.min(disc_raw).min(proj_raw);
            0.55_f64 * min_raw + 0.25_f64 * (gen_raw + disc_raw + proj_raw) / 3.0_f64
                - 0.10_f64 * residual_norm
        };

        PhiReport {
            generativity: g_c,
            discrimination: d_c,
            projection: p_c,
            mip_integration,
            diff_stability,
            residual_norm,
            phi_sharp,
            shaping_score,
        }
    }

    /// Canal `ask` pur : déploiement vertical depuis l'opérateur latent G (sans autorégression),
    /// sélectionné par clôture de résidu auto-référentielle : max [φ_sharp(E + ΔE) - φ_sharp(E)] - λ·Écho.
    fn ask_emit(&self, view: &RelationalView, budget: usize, out_len: usize) -> Vec<u8> {
        if view.vocab.is_empty() || view.seq.is_empty() {
            return Vec::new();
        }
        let v = view.v_size;
        let trace = self.run_abduction(view, budget, true, &[], false);
        let unfold = self.unfold_top_down(&view.node_init, &trace.state, None, false);
        let phi_base = self
            .evaluate_phi_state(view, &trace.state, budget, true)
            .phi_sharp;

        let mut node_res: Vec<(usize, f64)> = (0..v)
            .filter(|&u| view.graph.w[0][u] > 0.0_f64)
            .map(|u| {
                let mut err = 0.0_f64;
                for s in 0..3 {
                    let mut l1 = 0.0_f64;
                    for w in 0..v {
                        l1 += (view.graph.r[s][u * v + w] - unfold.r_hat[s][u * v + w]).abs();
                    }
                    err += SCALE_W[s] * 0.5_f64 * l1;
                }
                (u, err)
            })
            .collect();
        node_res.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut e_4grams: HashSet<[usize; 4]> = HashSet::new();
        for w in view.seq.windows(4) {
            e_4grams.insert([w[0], w[1], w[2], w[3]]);
        }

        let n_candidates = node_res.len().min(4).max(1);
        let k1 = trace.state.h1.len();

        let mut best_cand_idx: Vec<usize> = Vec::new();
        let mut best_closure = f64::NEG_INFINITY;

        for c_i in 0..n_candidates {
            let anchor = if c_i < node_res.len() {
                node_res[c_i].0
            } else {
                0
            };

            let mut lat_dist = vec![0.0_f64; k1];
            for k in 0..k1 {
                lat_dist[k] = unfold.a_ground[anchor * k1 + k];
            }

            let mut cand_idx = Vec::with_capacity(out_len);
            cand_idx.push(anchor);

            let mut edge_visits: HashMap<(usize, usize), usize> = HashMap::new();
            let mut sym_visits = vec![0usize; v];
            sym_visits[anchor] = 1;

            for _ in 1..out_len {
                let mut next_lat = vec![0.0_f64; k1];
                for src in 0..k1 {
                    let ps = lat_dist[src];
                    if ps <= 1e-12_f64 {
                        continue;
                    }
                    for dst in 0..k1 {
                        let vis = *edge_visits.get(&(src, dst)).unwrap_or(&0) as f64;
                        next_lat[dst] +=
                            ps * unfold.g_op[src * k1 + dst] / (1.0_f64 + 0.6_f64 * vis);
                    }
                }
                let sum_l: f64 = next_lat.iter().sum();
                if sum_l > 1e-12_f64 {
                    for x in &mut next_lat {
                        *x /= sum_l;
                    }
                }

                let src_peak = lat_dist
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let dst_peak = next_lat
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                *edge_visits.entry((src_peak, dst_peak)).or_insert(0) += 1;

                lat_dist = next_lat;

                let mut best_sym = 0usize;
                let mut best_p = f64::NEG_INFINITY;
                for u in 0..v {
                    let mut p = 0.0_f64;
                    for k in 0..k1 {
                        p += lat_dist[k] * unfold.emit[k * v + u];
                    }
                    let penalized = p / (1.0_f64 + 0.35_f64 * (sym_visits[u] as f64));
                    if penalized > best_p {
                        best_p = penalized;
                        best_sym = u;
                    }
                }
                sym_visits[best_sym] += 1;
                cand_idx.push(best_sym);
            }

            let mut extended_seq = view.seq.clone();
            extended_seq.extend_from_slice(&cand_idx);
            let ext_view = RelationalView::from_indexed(view.vocab.clone(), extended_seq);
            let phi_ext = self.evaluate_phi(&ext_view, budget, true).phi_sharp;

            let echo_ratio = if cand_idx.len() >= 4 {
                let mut hits = 0usize;
                let total_w = cand_idx.len() - 3;
                for w in cand_idx.windows(4) {
                    if e_4grams.contains(&[w[0], w[1], w[2], w[3]]) {
                        hits += 1;
                    }
                }
                (hits as f64) / (total_w as f64)
            } else {
                0.0_f64
            };

            let closure_score = (phi_ext - phi_base) - 0.15_f64 * echo_ratio;
            if closure_score > best_closure {
                best_closure = closure_score;
                best_cand_idx = cand_idx;
            }
        }

        best_cand_idx.into_iter().map(|i| view.vocab[i]).collect()
    }
}

// ============================================================================
// 5. Métriques d'information, baselines (LL, MDL), récupération excédentaire C
// ============================================================================

fn scale_gain(target: &[f64], pred: &[f64], row_w: &[f64], h_marg: f64, v: usize) -> f64 {
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

fn multi_scale_gain(graph: &ScaleGraph, r_hat: &[Vec<f64>; 3], v: usize) -> f64 {
    let mut acc = 0.0_f64;
    for s in 0..3 {
        acc += SCALE_W[s] * scale_gain(&graph.r[s], &r_hat[s], &graph.w[s], graph.h_marg[s], v);
    }
    acc.clamp(-1.0_f64, 1.0_f64)
}

fn multi_scale_l1(graph: &ScaleGraph, r_hat: &[Vec<f64>; 3], v: usize) -> f64 {
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

fn multi_scale_pred_tvd(
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

fn log_likelihood_graph(graph: &ScaleGraph, r_hat: &[Vec<f64>; 3], v: usize) -> f64 {
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

fn mdl_graph(
    graph: &ScaleGraph,
    r_hat: &[Vec<f64>; 3],
    state: &HierState,
    seq_len: usize,
    v: usize,
) -> f64 {
    let ll = log_likelihood_graph(graph, r_hat, v);
    let mut energy = 0.0_f64;
    for h in state.h1.iter().chain(state.h2.iter()) {
        for &val in h {
            if val.abs() > 0.05_f64 {
                energy += val.abs();
            }
        }
    }
    let n = seq_len.max(2) as f64;
    ll - 0.02_f64 * energy * n.ln() / n
}

/// Étalonnage corrigé : mesure EXCLUSIVEMENT la récupération excédentaire normalisée
/// de la structure conditionnelle de C au-dessus des marginales P_marg.
/// Un prédicteur marginal obtient 0.0 et un prédicteur uniforme <= 0.0 quelle que soit V.
fn observational_equivalence_excess(
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

fn pearson_corr(xs: &[f64], ys: &[f64]) -> f64 {
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

fn spearman_corr(xs: &[f64], ys: &[f64]) -> f64 {
    let rx = ranks(xs);
    let ry = ranks(ys);
    pearson_corr(&rx, &ry)
}

// ============================================================================
// 6. Entraînement fermé multi-budget (PTGRS générique + φ_sharp uniquement)
// ============================================================================

fn train_closed(out_path: &Path, steps: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let mut model = PhisharpModel::new_random(&mut rng);

    let pop_pairs = 16usize;
    let sigma: f64 = 0.075;
    let lr: f64 = 0.055;
    let batch_size = 6usize;

    let mut m_adam = vec![0.0_f64; PARAM_COUNT];
    let mut v_adam = vec![0.0_f64; PARAM_COUNT];
    let beta1: f64 = 0.9;
    let beta2: f64 = 0.999;

    println!(
        "=== ENTRAÎNEMENT FERMÉ PHISHARP V2.1 (PTGRS générique + A/E/G dynamiques, {} params) ===",
        PARAM_COUNT
    );

    for step in 1..=steps {
        let t_budget = 3 + rng.range_usize(5); // T dans 3..=7
        let t_next = t_budget + 2;

        let mut batch_views = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let cause = PtgrsCause::sample_generic(&mut rng);
            let depth = 3 + rng.range_usize(3);
            let bytes = cause.emit_effect(&mut rng, depth, 150);
            batch_views.push(RelationalView::from_bytes(&bytes));
        }

        let mut epsilons: Vec<Vec<f64>> = Vec::with_capacity(pop_pairs);
        for _ in 0..pop_pairs {
            let mut e = vec![0.0_f64; PARAM_COUNT];
            for val in &mut e {
                *val = rng.next_normal();
            }
            epsilons.push(e);
        }

        let base_params = &model.params;
        let views_ref = &batch_views;
        let mut diffs = vec![0.0_f64; pop_pairs];

        thread::scope(|s| {
            let mut handles = Vec::with_capacity(pop_pairs);
            for (idx, eps) in epsilons.iter().enumerate() {
                handles.push(s.spawn(move || {
                    let mut p_plus = base_params.clone();
                    let mut p_minus = base_params.clone();
                    for i in 0..PARAM_COUNT {
                        p_plus[i] += sigma * eps[i];
                        p_minus[i] -= sigma * eps[i];
                    }
                    let m_plus = PhisharpModel { params: p_plus };
                    let m_minus = PhisharpModel { params: p_minus };

                    let eval_cand = |m: &PhisharpModel| -> f64 {
                        let mut acc = 0.0_f64;
                        let snaps = [t_budget, t_next];
                        for view in views_ref {
                            let tr = m.run_abduction(view, t_next, true, &snaps, false);
                            let r1 = m.evaluate_phi_state(view, &tr.snapshots[0].1, t_budget, true);
                            let r2 = m.evaluate_phi_state(view, &tr.snapshots[1].1, t_next, true);
                            let drop_pen = (r1.phi_sharp - r2.phi_sharp).max(0.0_f64);
                            acc += 0.45_f64 * r1.shaping_score
                                + 0.45_f64 * r2.shaping_score
                                + 0.10_f64 * r1.shaping_score.min(r2.shaping_score)
                                - 0.20_f64 * drop_pen;
                        }
                        acc / (views_ref.len() as f64)
                    };

                    (idx, eval_cand(&m_plus) - eval_cand(&m_minus))
                }));
            }
            for h in handles {
                let (idx, d) = h.join().unwrap();
                diffs[idx] = d;
            }
        });

        let std_d = (diffs.iter().map(|d| d * d).sum::<f64>() / (pop_pairs as f64))
            .sqrt()
            .max(1e-6_f64);

        let mut grad = vec![0.0_f64; PARAM_COUNT];
        for (idx, eps) in epsilons.iter().enumerate() {
            let coeff = (diffs[idx] / std_d) / ((2 * pop_pairs) as f64 * sigma);
            for i in 0..PARAM_COUNT {
                grad[i] += coeff * eps[i];
            }
        }

        let b1_corr = 1.0_f64 - beta1.powi(step as i32);
        let b2_corr = 1.0_f64 - beta2.powi(step as i32);
        for i in 0..PARAM_COUNT {
            m_adam[i] = beta1 * m_adam[i] + (1.0_f64 - beta1) * grad[i];
            v_adam[i] = beta2 * v_adam[i] + (1.0_f64 - beta2) * grad[i] * grad[i];
            let m_hat = m_adam[i] / b1_corr;
            let v_hat = v_adam[i] / b2_corr;
            model.params[i] += lr * m_hat / (v_hat.sqrt() + 1e-8_f64);
            model.params[i] = model.params[i].clamp(-4.0_f64, 4.0_f64);
        }

        if step == 1 || step % 25 == 0 || step == steps {
            let mut g_sum = 0.0_f64;
            let mut d_sum = 0.0_f64;
            let mut p_sum = 0.0_f64;
            let mut mip_sum = 0.0_f64;
            let mut phi_t = 0.0_f64;
            let mut phi_t2 = 0.0_f64;
            let snaps = [t_budget, t_next];
            for view in &batch_views {
                let tr = model.run_abduction(view, t_next, true, &snaps, false);
                let r1 = model.evaluate_phi_state(view, &tr.snapshots[0].1, t_budget, true);
                let r2 = model.evaluate_phi_state(view, &tr.snapshots[1].1, t_next, true);
                g_sum += r1.generativity;
                d_sum += r1.discrimination;
                p_sum += r1.projection;
                mip_sum += r1.mip_integration;
                phi_t += r1.phi_sharp;
                phi_t2 += r2.phi_sharp;
            }
            let nb = batch_views.len() as f64;
            println!(
                "step {:4}/{} (T={},{}) | φ(T)={:.4} φ(T+2)={:.4} | G={:.3} D={:.3} P={:.3} MIP={:.3}",
                step,
                steps,
                t_budget,
                t_next,
                phi_t / nb,
                phi_t2 / nb,
                g_sum / nb,
                d_sum / nb,
                p_sum / nb,
                mip_sum / nb
            );
        }
    }

    model.save(out_path).expect("Échec écriture modèle V2.1");
    println!("Modèle V2.1 figé sauvegardé : {}", out_path.display());
}

// ============================================================================
// 7. Porte d'étalonnage V2.1 (récupération excédentaire C + 6 témoins négatifs)
// ============================================================================

fn run_calibration(model: &PhisharpModel, seed: u64) {
    let mut rng = Rng::new(seed);
    let mut eval_rng = Rng::new(seed ^ 0x55AA_CC33);
    println!("=== PORTE D'ÉTALONNAGE V2.1 (PTGRS générique, récupération excédentaire C) ===");

    let mut eq_scores = Vec::new();
    let mut phi_scores = Vec::new();
    let mut ll_scores = Vec::new();
    let mut mdl_scores = Vec::new();

    // 1. Inter-causes (36 essais variant V, profondeur, longueur et budget)
    for trial in 0..36 {
        let cause = PtgrsCause::sample_generic(&mut rng);
        let depth = 3 + (trial % 3);
        let len = match trial % 3 {
            0 => 96,
            1 => 144,
            _ => 192,
        };
        let bytes = cause.emit_effect(&mut rng, depth, len);
        let view = RelationalView::from_bytes(&bytes);
        let truth = cause.ground_truth_multi_scale(&mut eval_rng, depth);

        let budget = match trial % 6 {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 4,
            4 => 6,
            _ => 8,
        };
        let use_ref = (trial % 4) != 1;
        let trace = model.run_abduction(&view, budget, use_ref, &[], false);
        let rep = model.evaluate_phi_state(&view, &trace.state, budget, use_ref);
        let unfold = model.unfold_top_down(&view.node_init, &trace.state, None, false);

        let eq = observational_equivalence_excess(&view, &unfold.r_hat, &truth);
        let ll = log_likelihood_graph(&view.graph, &unfold.r_hat, view.v_size);
        let mdl = mdl_graph(
            &view.graph,
            &unfold.r_hat,
            &trace.state,
            view.seq.len(),
            view.v_size,
        );

        eq_scores.push(eq);
        phi_scores.push(rep.phi_sharp);
        ll_scores.push(ll);
        mdl_scores.push(mdl);
    }

    let r_phi = pearson_corr(&phi_scores, &eq_scores);
    let r_ll = pearson_corr(&ll_scores, &eq_scores);
    let r_mdl = pearson_corr(&mdl_scores, &eq_scores);

    let s_phi = spearman_corr(&phi_scores, &eq_scores);
    let s_ll = spearman_corr(&ll_scores, &eq_scores);
    let s_mdl = spearman_corr(&mdl_scores, &eq_scores);

    let mean_phi = phi_scores.iter().sum::<f64>() / (phi_scores.len() as f64);
    let mean_eq = eq_scores.iter().sum::<f64>() / (eq_scores.len() as f64);

    // 2. Intra-cause (suivi pur de la trajectoire t = 0,1,2,4,6,8 sur 12 causes fixes)
    let traj_budgets = [0usize, 1, 2, 4, 6, 8];
    let mut within_r_phi = 0.0_f64;
    let mut within_r_ll = 0.0_f64;
    let mut within_r_mdl = 0.0_f64;
    for _ in 0..12 {
        let cause = PtgrsCause::sample_generic(&mut rng);
        let bytes = cause.emit_effect(&mut rng, 4, 160);
        let view = RelationalView::from_bytes(&bytes);
        let truth = cause.ground_truth_multi_scale(&mut eval_rng, 4);

        let tr = model.run_abduction(&view, 8, true, &traj_budgets, false);
        let mut eqs = Vec::with_capacity(traj_budgets.len());
        let mut phis = Vec::with_capacity(traj_budgets.len());
        let mut lls = Vec::with_capacity(traj_budgets.len());
        let mut mdls = Vec::with_capacity(traj_budgets.len());

        for (b, st) in &tr.snapshots {
            let rep = model.evaluate_phi_state(&view, st, *b, true);
            let unf = model.unfold_top_down(&view.node_init, st, None, false);
            eqs.push(observational_equivalence_excess(&view, &unf.r_hat, &truth));
            phis.push(rep.phi_sharp);
            lls.push(log_likelihood_graph(&view.graph, &unf.r_hat, view.v_size));
            mdls.push(mdl_graph(&view.graph, &unf.r_hat, st, view.seq.len(), view.v_size));
        }
        within_r_phi += pearson_corr(&phis, &eqs);
        within_r_ll += pearson_corr(&lls, &eqs);
        within_r_mdl += pearson_corr(&mdls, &eqs);
    }
    within_r_phi /= 12.0_f64;
    within_r_ll /= 12.0_f64;
    within_r_mdl /= 12.0_f64;

    println!(
        "Inter-causes Pearson  vs Rec_exc(C) : φ_sharp={:+.4} | Vraisemblance={:+.4} | MDL={:+.4}",
        r_phi, r_ll, r_mdl
    );
    println!(
        "Inter-causes Spearman vs Rec_exc(C) : φ_sharp={:+.4} | Vraisemblance={:+.4} | MDL={:+.4}",
        s_phi, s_ll, s_mdl
    );
    println!(
        "Intra-cause  Pearson  (t=0..8, 12x) : φ_sharp={:+.4} | Vraisemblance={:+.4} | MDL={:+.4}",
        within_r_phi, within_r_ll, within_r_mdl
    );
    println!(
        "Moyennes PTGRS (t=0..8)             : φ_sharp={:.4} | Rec_exc(C)={:+.4}",
        mean_phi, mean_eq
    );

    // Témoin 1 : octets aléatoires uniformes (10x)
    let mut rand_phi = 0.0_f64;
    for _ in 0..10 {
        let bytes: Vec<u8> = (0..160).map(|_| (rng.next_u64() % 10) as u8).collect();
        let view = RelationalView::from_bytes(&bytes);
        rand_phi += model.evaluate_phi(&view, DEFAULT_BUDGET, true).phi_sharp;
    }
    rand_phi /= 10.0_f64;

    // Témoin 2 : H := E (aucune intégration latente : k1 = 0, k2 = 0 => MIP = 0)
    let cause_ref = PtgrsCause::sample_generic(&mut rng);
    let bytes_ref = cause_ref.emit_effect(&mut rng, 4, 160);
    let view_ref = RelationalView::from_bytes(&bytes_ref);
    let empty_state = HierState::zeros(view_ref.v_size, 0, 0);
    let h_eq_e_phi = model
        .evaluate_phi_state(&view_ref, &empty_state, DEFAULT_BUDGET, true)
        .phi_sharp;

    // Témoin 3 : cause sur-générale (état latent nul => prédiction uniforme, D = 0)
    let zero_state = HierState::zeros(view_ref.v_size, K1, K2);
    let overgen_phi = model
        .evaluate_phi_state(&view_ref, &zero_state, DEFAULT_BUDGET, true)
        .phi_sharp;

    // Témoin 4 : E permuté à marginales P_marg identiques (10x)
    let mut perm_sum = 0.0_f64;
    let mut perm_max = 0.0_f64;
    for _ in 0..10 {
        let c = PtgrsCause::sample_generic(&mut rng);
        let mut b = c.emit_effect(&mut rng, 4, 160);
        rng.shuffle(&mut b);
        let vp = RelationalView::from_bytes(&b);
        let p = model.evaluate_phi(&vp, DEFAULT_BUDGET, true).phi_sharp;
        perm_sum += p;
        if p > perm_max {
            perm_max = p;
        }
    }
    let perm_mean = perm_sum / 10.0_f64;

    // Témoin 5 : branches épiphénoménales (nœud mort L1, clone L1, clone L2)
    let trace_ref = model.run_abduction(&view_ref, DEFAULT_BUDGET, true, &[], false);
    let phi_clean = model
        .evaluate_phi_state(&view_ref, &trace_ref.state, DEFAULT_BUDGET, true)
        .phi_sharp;

    let l1_dead = trace_ref.state.with_dead_l1_node(view_ref.v_size);
    let phi_l1_dead = model
        .evaluate_phi_state(&view_ref, &l1_dead, DEFAULT_BUDGET, true)
        .phi_sharp;

    let l1_clone = trace_ref.state.with_cloned_l1_node(view_ref.v_size, 0);
    let phi_l1_clone = model
        .evaluate_phi_state(&view_ref, &l1_clone, DEFAULT_BUDGET, true)
        .phi_sharp;

    let l2_clone = trace_ref.state.with_cloned_l2_node(0);
    let phi_l2_clone = model
        .evaluate_phi_state(&view_ref, &l2_clone, DEFAULT_BUDGET, true)
        .phi_sharp;

    // Témoin 6 : invariance exacte par renommage bijectif d'alphabet
    let mut bij: Vec<u8> = (0u8..=255).collect();
    rng.shuffle(&mut bij);
    let mut inv_bij = [0u8; 256];
    for (orig, &mapped) in bij.iter().enumerate() {
        inv_bij[mapped as usize] = orig as u8;
    }
    let renamed_bytes: Vec<u8> = bytes_ref.iter().map(|&b| bij[b as usize]).collect();
    let view_ren = RelationalView::from_bytes(&renamed_bytes);

    let tr_orig = model.run_abduction(&view_ref, DEFAULT_BUDGET, true, &[], true);
    let tr_ren = model.run_abduction(&view_ren, DEFAULT_BUDGET, true, &[], true);
    let max_phi_diff = tr_orig
        .phi_history
        .iter()
        .zip(tr_ren.phi_history.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let out_orig = model.ask_emit(&view_ref, DEFAULT_BUDGET, 24);
    let out_ren = model.ask_emit(&view_ren, DEFAULT_BUDGET, 24);
    let out_ren_inv: Vec<u8> = out_ren.iter().map(|&b| inv_bij[b as usize]).collect();
    let ask_exact = out_orig == out_ren_inv;

    println!("--- Témoins négatifs ---");
    println!("  Octets aléatoires (10x)       : φ_sharp = {:.4}", rand_phi);
    println!("  H := E (sans intégration)     : φ_sharp = {:.4}", h_eq_e_phi);
    println!("  Cause sur-générale            : φ_sharp = {:.4}", overgen_phi);
    println!(
        "  E permuté à marginales égales : moy={:.4} | max={:.4}",
        perm_mean, perm_max
    );
    println!(
        "  Branches épiphénoménales      : propre={:.4} -> L1_morte={:.4} | L1_clone={:.4} | L2_clone={:.4}",
        phi_clean, phi_l1_dead, phi_l1_clone, phi_l2_clone
    );
    println!(
        "  Invariance bijective          : max|Δφ(t)|={:.2e} | ask(π(E))==π(ask(E)) : {}",
        max_phi_diff, ask_exact
    );

    let signal_alive = mean_phi > 0.03_f64 && phi_clean > 0.03_f64;
    let pass_corr = signal_alive
        && ((r_phi > r_ll && r_phi > r_mdl && r_phi > 0.40_f64)
            || (s_phi > s_ll && s_phi > s_mdl && s_phi > 0.45_f64 && r_phi > 0.35_f64));
    let pass_neg = signal_alive
        && rand_phi < 0.03_f64
        && h_eq_e_phi < 1e-9_f64
        && overgen_phi < 0.02_f64
        && perm_mean < 0.03_f64
        && perm_max < 0.08_f64
        && phi_l1_dead < 1e-9_f64
        && phi_l1_clone < 1e-9_f64
        && phi_l2_clone < 1e-9_f64;
    let pass_inv = max_phi_diff < 1e-9_f64 && ask_exact;

    println!("--- VERDICT ÉTALONNAGE V2.1 ---");
    if pass_corr && pass_neg && pass_inv {
        println!(
            "PASS : A est lisible. Passage au barreau 1 (OOD) et à la rupture de générateur autorisé."
        );
    } else {
        println!(
            "ARRÊT : Étalonnage échoué (alive={}, corr_ok={}, neg_ok={}, inv_ok={}). A non lisible.",
            signal_alive, pass_corr, pass_neg, pass_inv
        );
    }
}

// ============================================================================
// 8. Barreau 1 : PTGRS OOD, courbe anytime t=0..10, ablations, triple résidu
// ============================================================================

fn run_eval_ood(model: &PhisharpModel, seed: u64) {
    let mut rng = Rng::new(seed);
    println!("=== ÉVALUATION BARREAU 1 V2.1 (PTGRS OOD : profondeurs d=2, 4, 8) ===");

    let max_t = 10usize;
    let all_snaps: Vec<usize> = (0..=max_t).collect();

    for &depth in &[2usize, 4, 8] {
        let mut curves = vec![0.0_f64; max_t + 1];
        let mut no_ref_sum = 0.0_f64;
        let trials = 12usize;

        for _ in 0..trials {
            let cause = PtgrsCause::sample_ood(&mut rng, depth);
            let bytes = cause.emit_effect(&mut rng, depth, 180);
            let view = RelationalView::from_bytes(&bytes);

            let tr = model.run_abduction(&view, max_t, true, &all_snaps, false);
            for (step, st) in &tr.snapshots {
                curves[*step] += model.evaluate_phi_state(&view, st, *step, true).phi_sharp;
            }
            no_ref_sum += model.evaluate_phi(&view, DEFAULT_BUDGET, false).phi_sharp;
        }

        for val in &mut curves {
            *val /= trials as f64;
        }
        no_ref_sum /= trials as f64;

        print!("Profondeur d={} | Anytime φ(t=0..10): ", depth);
        for (b, val) in curves.iter().enumerate() {
            print!("t{}={:.3} ", b, val);
        }
        println!();
        let retention = if curves[DEFAULT_BUDGET] > 1e-6_f64 {
            curves[max_t] / curves[DEFAULT_BUDGET]
        } else {
            0.0_f64
        };
        println!(
            "  Ablations (d={}) : budget=0 -> {:.4} | sans réfutation (t={}) -> {:.4} vs avec -> {:.4} | maintien t10/t6 = {:.2}x",
            depth, curves[0], DEFAULT_BUDGET, no_ref_sum, curves[DEFAULT_BUDGET], retention
        );
    }

    println!("--- Test du résidu (vrai complément vs étranger C' vs écho) ---");
    let mut gain_true = 0.0_f64;
    let mut gain_foreign = 0.0_f64;
    let mut gain_echo = 0.0_f64;
    let trials_r = 15usize;

    for _ in 0..trials_r {
        let c_true = PtgrsCause::sample_ood(&mut rng, 4);
        let c_other = PtgrsCause::sample_ood(&mut rng, 4);

        let full_true = c_true.emit_effect(&mut rng, 4, 180);
        let foreign_chunk = c_other.emit_effect(&mut rng, 4, 60);

        let partial = &full_true[0..120];
        let mut with_true = partial.to_vec();
        with_true.extend_from_slice(&full_true[120..180]);

        let mut with_foreign = partial.to_vec();
        with_foreign.extend_from_slice(&foreign_chunk);

        let mut with_echo = partial.to_vec();
        with_echo.extend_from_slice(&partial[0..60]);

        let phi_partial = model
            .evaluate_phi(&RelationalView::from_bytes(partial), DEFAULT_BUDGET, true)
            .phi_sharp;
        let phi_true = model
            .evaluate_phi(&RelationalView::from_bytes(&with_true), DEFAULT_BUDGET, true)
            .phi_sharp;
        let phi_foreign = model
            .evaluate_phi(&RelationalView::from_bytes(&with_foreign), DEFAULT_BUDGET, true)
            .phi_sharp;
        let phi_echo = model
            .evaluate_phi(&RelationalView::from_bytes(&with_echo), DEFAULT_BUDGET, true)
            .phi_sharp;

        gain_true += phi_true - phi_partial;
        gain_foreign += phi_foreign - phi_partial;
        gain_echo += phi_echo - phi_partial;
    }

    println!(
        "Δφ_sharp moyen : vrai complément={:+.4} | étranger C'={:+.4} | écho={:+.4}",
        gain_true / (trials_r as f64),
        gain_foreign / (trials_r as f64),
        gain_echo / (trials_r as f64)
    );
}

// ============================================================================
// 9. Test de RUPTURE DE GÉNÉRATEUR (hors-PTGRS : CA 1D & séquences algorithmiques)
// ============================================================================

fn gen_elementary_ca(rule: u8, seed: u64, target_len: usize) -> Vec<u8> {
    let width = 12usize;
    let mut cells = vec![0u8; width];
    let mut rng = Rng::new(seed);
    for c in &mut cells {
        *c = (rng.next_u64() & 1) as u8;
    }
    if cells.iter().all(|&x| x == 0) {
        cells[width / 2] = 1;
    }

    for _ in 0..6 {
        let mut next = vec![0u8; width];
        for i in 0..width {
            let l = cells[(i + width - 1) % width];
            let m = cells[i];
            let r = cells[(i + 1) % width];
            let pat = (l << 2) | (m << 1) | r;
            next[i] = (rule >> pat) & 1;
        }
        cells = next;
    }

    let mut out = Vec::with_capacity(target_len + 4);
    while out.len() < target_len {
        for blk in 0..(width / 3) {
            let sym = (cells[3 * blk] << 2) | (cells[3 * blk + 1] << 1) | cells[3 * blk + 2];
            out.push(sym);
        }
        let mut next = vec![0u8; width];
        for i in 0..width {
            let l = cells[(i + width - 1) % width];
            let m = cells[i];
            let r = cells[(i + 1) % width];
            let pat = (l << 2) | (m << 1) | r;
            next[i] = (rule >> pat) & 1;
        }
        cells = next;
    }
    out.truncate(target_len);
    out
}

fn gen_fibonacci_mod(modulus: u8, seed: u64, target_len: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut a = (1 + (rng.next_u64() as u8) % (modulus - 1)) % modulus;
    let mut b = (1 + (rng.next_u64() as u8) % (modulus - 1)) % modulus;
    let mut out = Vec::with_capacity(target_len);
    for _ in 0..target_len {
        out.push(a);
        let c = (a.wrapping_add(b)) % modulus;
        a = b;
        b = c;
    }
    out
}

fn gen_collatz_trace(seed: u64, target_len: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut n = 27u64 + (rng.next_u64() % 500);
    let mut out = Vec::with_capacity(target_len);
    while out.len() < target_len {
        out.push((n % 8) as u8);
        if n <= 1 {
            n = 27u64 + (rng.next_u64() % 500);
        } else if n % 2 == 0 {
            n /= 2;
        } else {
            n = 3 * n + 1;
        }
    }
    out
}

fn gen_thue_morse_blocks(offset: u64, target_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(target_len);
    for i in 0..target_len {
        let idx = offset + (i as u64);
        let b0 = (idx.count_ones() & 1) as u8;
        let b1 = ((idx + 1).count_ones() & 1) as u8;
        let b2 = ((idx + 2).count_ones() & 1) as u8;
        out.push((b0 << 2) | (b1 << 1) | b2);
    }
    out
}

fn gen_structured_frames(seed: u64, target_len: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut counter = (rng.next_u64() % 5) as u8;
    let mut out = Vec::with_capacity(target_len + 4);
    while out.len() < target_len {
        let field = (counter % 3) + 1;
        let val = (counter + field) % 5;
        let chk = (field + val + 2) % 5;
        out.push(9u8);
        out.push(5u8 + field);
        out.push(val);
        out.push(chk);
        counter = (counter + 1) % 5;
    }
    out.truncate(target_len);
    out
}

fn run_eval_transfer(model: &PhisharpModel, seed: u64) {
    let mut rng = Rng::new(seed);
    println!("=== TEST DE RUPTURE DE GÉNÉRATEUR V2.1 (HORS-PTGRS : CA 1D & ALGORITHMIQUE) ===");
    println!(
        "{:<24} | {:>7} | {:>7} | {:>5} {:>5} {:>5} {:>5} | {:>7} | {:>6}",
        "Générateur hors-PTGRS", "φ_réel", "φ_perm", "G", "D", "P", "MIP", "no_ref", "t10/t6"
    );
    println!("{}", "-".repeat(92));

    let families: Vec<(&str, fn(&mut Rng) -> Vec<u8>, bool)> = vec![
        ("CA 1D Rule 110 (Turing)", |r| gen_elementary_ca(110, r.next_u64(), 180), true),
        ("CA 1D Rule 54 (Gliders)", |r| gen_elementary_ca(54, r.next_u64(), 180), true),
        ("CA 1D Rule 90 (Fractal)", |r| gen_elementary_ca(90, r.next_u64(), 180), true),
        ("CA 1D Rule 30 (Chaos)",   |r| gen_elementary_ca(30, r.next_u64(), 180), false),
        ("Fibonacci mod 11",        |r| gen_fibonacci_mod(11, r.next_u64(), 180), true),
        ("Collatz mod 8",           |r| gen_collatz_trace(r.next_u64(), 180), true),
        ("Thue-Morse (blocs 3b)",   |r| gen_thue_morse_blocks(r.next_u64() % 1024, 180), true),
        ("Trames protocole/log",    |r| gen_structured_frames(r.next_u64(), 180), true),
    ];

    let mut structured_passed = 0usize;
    let mut structured_total = 0usize;
    let snaps = [DEFAULT_BUDGET, 10];

    for (name, sampler, expect_structure) in families {
        let trials = 8usize;
        let mut phi_real = 0.0_f64;
        let mut phi_perm = 0.0_f64;
        let mut g_sum = 0.0_f64;
        let mut d_sum = 0.0_f64;
        let mut p_sum = 0.0_f64;
        let mut mip_sum = 0.0_f64;
        let mut phi_noref = 0.0_f64;
        let mut phi_t10 = 0.0_f64;

        for _ in 0..trials {
            let bytes = sampler(&mut rng);
            let view = RelationalView::from_bytes(&bytes);
            let tr = model.run_abduction(&view, 10, true, &snaps, false);
            let r6 = model.evaluate_phi_state(&view, &tr.snapshots[0].1, DEFAULT_BUDGET, true);
            let r10 = model.evaluate_phi_state(&view, &tr.snapshots[1].1, 10, true);
            let rn = model.evaluate_phi(&view, DEFAULT_BUDGET, false);

            let mut perm_bytes = bytes.clone();
            rng.shuffle(&mut perm_bytes);
            let vp = RelationalView::from_bytes(&perm_bytes);
            let rp = model.evaluate_phi(&vp, DEFAULT_BUDGET, true);

            phi_real += r6.phi_sharp;
            phi_perm += rp.phi_sharp;
            g_sum += r6.generativity;
            d_sum += r6.discrimination;
            p_sum += r6.projection;
            mip_sum += r6.mip_integration;
            phi_noref += rn.phi_sharp;
            phi_t10 += r10.phi_sharp;
        }

        let n = trials as f64;
        phi_real /= n;
        phi_perm /= n;
        g_sum /= n;
        d_sum /= n;
        p_sum /= n;
        mip_sum /= n;
        phi_noref /= n;
        phi_t10 /= n;

        let ret = if phi_real > 1e-6_f64 { phi_t10 / phi_real } else { 0.0_f64 };
        println!(
            "{:<24} | {:>7.4} | {:>7.4} | {:>5.3} {:>5.3} {:>5.3} {:>5.3} | {:>7.4} | {:>5.2}x",
            name, phi_real, phi_perm, g_sum, d_sum, p_sum, mip_sum, phi_noref, ret
        );

        if expect_structure {
            structured_total += 1;
            if phi_real > 0.04_f64 && phi_real > phi_perm + 0.03_f64 && ret >= 0.75_f64 {
                structured_passed += 1;
            }
        }
    }

    println!("{}", "-".repeat(92));
    println!(
        "VERDICT RUPTURE DE GÉNÉRATEUR : {}/{} familles structurées hors-PTGRS transférées (Rule 30 = témoin chaotique).",
        structured_passed, structured_total
    );
}

// ============================================================================
// 10. Interface CLI stricte (`ask --model out/sharp_v21.phis --file perceived.eff`)
// ============================================================================

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  phisharp train         --out out/sharp_v21.phis [--steps 300] [--seed 42]");
        eprintln!("  phisharp calibrate     --model out/sharp_v21.phis [--seed 101]");
        eprintln!("  phisharp eval-ood      --model out/sharp_v21.phis [--seed 202]");
        eprintln!("  phisharp eval-transfer --model out/sharp_v21.phis [--seed 303]");
        eprintln!("  phisharp ask           --model out/sharp_v21.phis --file perceived.eff");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "train" => {
            let out =
                get_arg(&args, "--out").unwrap_or_else(|| "out/sharp_v21.phis".to_string());
            let steps: usize = get_arg(&args, "--steps")
                .and_then(|s| s.parse().ok())
                .unwrap_or(300);
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(42);
            train_closed(Path::new(&out), steps, seed);
        }
        "calibrate" => {
            let model_path =
                get_arg(&args, "--model").unwrap_or_else(|| "out/sharp_v21.phis".to_string());
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(101);
            let model = PhisharpModel::load(Path::new(&model_path)).expect("Chargement modèle V2.1");
            run_calibration(&model, seed);
        }
        "eval-ood" => {
            let model_path =
                get_arg(&args, "--model").unwrap_or_else(|| "out/sharp_v21.phis".to_string());
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(202);
            let model = PhisharpModel::load(Path::new(&model_path)).expect("Chargement modèle V2.1");
            run_eval_ood(&model, seed);
        }
        "eval-transfer" => {
            let model_path =
                get_arg(&args, "--model").unwrap_or_else(|| "out/sharp_v21.phis".to_string());
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(303);
            let model = PhisharpModel::load(Path::new(&model_path)).expect("Chargement modèle V2.1");
            run_eval_transfer(&model, seed);
        }
        "ask" => {
            let model_path = get_arg(&args, "--model").expect("--model requis");
            let file_path = get_arg(&args, "--file").expect("--file requis");
            let model = PhisharpModel::load(Path::new(&model_path)).expect("Chargement modèle V2.1");
            let raw = fs::read(&file_path).expect("Lecture perceived.eff");
            let view = RelationalView::from_bytes(&raw);
            let out_bytes = model.ask_emit(&view, DEFAULT_BUDGET + 2, 48);
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(&out_bytes).expect("Écriture stdout");
        }
        other => {
            eprintln!("Commande inconnue : {}", other);
            std::process::exit(1);
        }
    }
}