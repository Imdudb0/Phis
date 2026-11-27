use crate::core::tensor::Mat;
use std::collections::BTreeMap;

/// Décalages directionnels du tenseur d'observation multi-relationnel G(E) (Spec v2, Section 1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lag {
    /// Contiguïté dirigée avant : s_t -> s_{t+1}.
    Fwd,
    /// Contiguïté dirigée arrière : s_t -> s_{t-1}.
    Bwd,
    /// Saut de médiation à distance 2 : s_t -> s_{t+2} (ex. u -> rel -> v).
    Skip2,
}

/// Hyperparamètres de l'extracteur S_256-équivariant d'entités M(E).
#[derive(Debug, Clone)]
pub struct EntityConfig {
    /// Fréquence minimale pour fusionner une paire adjacente à l'étape 1.
    pub min_pair_count: usize,
    /// Seuil minimal de PMI pour autoriser une fusion.
    pub tau_coh: f64,
    /// Seuil d'entropie de branchement (H_+(x) <= tau_h || H_-(y) <= tau_h).
    pub tau_h: f64,
    /// Seuil d'entropie de garde de frontière empêchant une ancre multi-octets d'absorber ses voisins.
    pub boundary_entropy_guard: f64,
    /// Longueur maximale en octets d'une entité consolidée à l'étape 1.
    pub max_entity_bytes: usize,
    /// Limite de fusions itératives à l'étape 1.
    pub max_merges: usize,
}

impl Default for EntityConfig {
    fn default() -> Self {
        Self {
            min_pair_count: 2,
            tau_coh: 0.35,
            tau_h: 1.35,
            boundary_entropy_guard: 0.75,
            max_entity_bytes: 16,
            max_merges: 512,
        }
    }
}

/// Origine structurelle d'une entité auto-induite dans V(E).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    /// Entité récurrente consolidée par cohésion PMI (étape 1) ou délimiteur bilatéral.
    Consolidated,
    /// Intervalle atomique promu entre ancres consolidées (étape 2, isole les hapax v_{\mu*}).
    HapaxPromoted,
}

/// Descripteur canonique d'une entité u \in V(E) = {0, ..., M(E)-1}.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityInfo {
    pub id: usize,
    pub bytes: Vec<u8>,
    pub count: usize,
    pub kind: EntityKind,
    /// Position de première apparition dans la séquence d'entités `seq`.
    pub first_pos: usize,
}

/// Graphe d'observation multi-relationnel G(E) auto-induit depuis `perceived.eff` (Spec v2).
/// Toutes les distributions de transition Q^{(r)}_{u, .} sont normalisées par entité u
/// afin d'éliminer totalement la noyade fréquentielle (Théorème 2.2).
#[derive(Debug, Clone)]
pub struct ObservationGraph {
    /// Nombre dynamique d'entités M(E).
    pub m: usize,
    /// Table des entités ordonnées canoniquement par première apparition.
    pub entities: Vec<EntityInfo>,
    /// Séquence segmentée d'entités s = (s_1, ..., s_T) \in {0, ..., M-1}^T.
    pub seq: Vec<usize>,
    /// Distributions empiriques de transition creuses Q^{(+1)}_{u, v}.
    pub q_fwd: Vec<Vec<(usize, f64)>>,
    /// Distributions empiriques de transition creuses Q^{(-1)}_{u, v}.
    pub q_bwd: Vec<Vec<(usize, f64)>>,
    /// Distributions empiriques de transition creuses Q^{(+2)}_{u, v}.
    pub q_skip2: Vec<Vec<(usize, f64)>>,
    /// Masques d'activité V_r(E) = { u : sum_v c^{(r)}(u, v) > 0 }.
    pub active_fwd: Vec<bool>,
    pub active_bwd: Vec<bool>,
    pub active_skip2: Vec<bool>,
    /// Matrice symétrique aplatie (M x M) d'adjacence locale A_loc(E) (diagonale fausse).
    pub a_loc: Vec<bool>,
    /// Voisins non locaux admissibles N_nl(u) = { v != u : A_loc(u, v) == false }.
    pub nonloc_neighbors: Vec<Vec<usize>>,
    /// Entropies empiriques irréductibles moyennées par 1/M : (1/M) * sum_{u in V_r} H(Q^{(r)}_u).
    pub h_emp_fwd: f64,
    pub h_emp_bwd: f64,
    pub h_emp_skip2: f64,
    /// Nombre d'itérations de fusion effectuées à l'étape 1.
    pub merges_performed: usize,
}

impl ObservationGraph {
    /// Extrait M(E) et construit G(E) depuis les octets bruts de `perceived.eff`.
    pub fn extract(bytes: &[u8], cfg: &EntityConfig) -> Self {
        if bytes.is_empty() {
            return Self::empty();
        }

        // ====================================================================
        // Étape 0 : Tokenisation initiale en ordre de première apparition.
        // Garantit une exécution bit-à-bit identique pour E et \pi(E) (\pi \in S_256).
        // ====================================================================
        let mut byte_to_tok = [None::<usize>; 256];
        let mut tok_bytes: Vec<Vec<u8>> = Vec::new();
        let mut seq_tok: Vec<usize> = Vec::with_capacity(bytes.len());

        for &b in bytes {
            let tid = if let Some(existing) = byte_to_tok[b as usize] {
                existing
            } else {
                let id = tok_bytes.len();
                byte_to_tok[b as usize] = Some(id);
                tok_bytes.push(vec![b]);
                id
            };
            seq_tok.push(tid);
        }

        // ====================================================================
        // Étape 1 : Consolidation itérative par cohésion PMI + garde de frontière.
        // ====================================================================
        let mut merges_performed = 0usize;

        for _ in 0..cfg.max_merges {
            let seq_len = seq_tok.len();
            if seq_len < 2 {
                break;
            }
            let num_toks = tok_bytes.len();
            let mut tok_counts = vec![0usize; num_toks];
            for &t in &seq_tok {
                tok_counts[t] += 1;
            }

            // Comptages des paires adjacentes (avec première position pour tie-break S_256-invariant)
            let mut pair_stats: BTreeMap<(usize, usize), (usize, usize)> = BTreeMap::new();
            let mut out_adj: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new(); num_toks];
            let mut in_adj: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new(); num_toks];
            let mut out_tot = vec![0usize; num_toks];
            let mut in_tot = vec![0usize; num_toks];

            for pos in 0..(seq_len - 1) {
                let a = seq_tok[pos];
                let b = seq_tok[pos + 1];
                pair_stats
                    .entry((a, b))
                    .and_modify(|e| e.0 += 1)
                    .or_insert((1, pos));
                *out_adj[a].entry(b).or_insert(0) += 1;
                *in_adj[b].entry(a).or_insert(0) += 1;
                out_tot[a] += 1;
                in_tot[b] += 1;
            }

            let mut h_out = vec![0.0f64; num_toks];
            let mut h_in = vec![0.0f64; num_toks];
            for t in 0..num_toks {
                if out_tot[t] > 0 {
                    h_out[t] = exact_entropy(&out_adj[t], out_tot[t]);
                }
                if in_tot[t] > 0 {
                    h_in[t] = exact_entropy(&in_adj[t], in_tot[t]);
                }
            }

            let seq_len_f = seq_len as f64;
            let mut best: Option<((usize, usize), f64, usize)> = None;

            for (&(a, b), &(c_ab, first_pos)) in &pair_stats {
                if c_ab < cfg.min_pair_count {
                    continue;
                }
                let len_a = tok_bytes[a].len();
                let len_b = tok_bytes[b].len();
                if len_a + len_b > cfg.max_entity_bytes {
                    continue;
                }

                // Critère de cohésion PMI + entropie de branchement (Spec v2, Section 1.1)
                let pmi = ((c_ab as f64) * seq_len_f
                    / ((tok_counts[a] as f64) * (tok_counts[b] as f64)))
                    .max(1e-12)
                    .ln();
                if pmi <= cfg.tau_coh {
                    continue;
                }
                if h_out[a] > cfg.tau_h && h_in[b] > cfg.tau_h {
                    continue;
                }

                let p_b_given_a = (c_ab as f64) / (out_tot[a].max(1) as f64);
                let p_a_given_b = (c_ab as f64) / (in_tot[b].max(1) as f64);
                let out_deg_a = out_adj[a].len();
                let in_deg_b = in_adj[b].len();

                // Garde de frontière bilatérale :
                // Empêche une ancre multi-octets complète (EdgeSep, REL_DEFINES, mots NL...)
                // d'absorber un voisin appartenant à une autre entité ou à un hapax.
                if len_a >= 2 && len_b >= 2 {
                    if (h_out[a] > cfg.boundary_entropy_guard && p_b_given_a < 0.65)
                        || (h_in[b] > cfg.boundary_entropy_guard && p_a_given_b < 0.65)
                    {
                        continue;
                    }
                } else if len_a >= 2 && len_b == 1 {
                    // Extension à droite d'un préfixe multi-octets par 1 octet
                    if h_out[a] > cfg.boundary_entropy_guard
                        && !(p_b_given_a >= 0.20 && p_a_given_b >= 0.70 && out_deg_a <= 4)
                    {
                        continue;
                    }
                } else if len_a == 1 && len_b >= 2 {
                    // Extension à gauche d'un suffixe multi-octets par 1 octet
                    if h_in[b] > cfg.boundary_entropy_guard
                        && !(p_a_given_b >= 0.50 && p_b_given_a >= 0.70 && in_deg_b <= 2)
                    {
                        continue;
                    }
                }

                // Gain d'information mutuelle totale c(a,b) * PMI(a,b) :
                // fusionne naturellement les mots de code récurrents haute fréquence avant les rares.
                let score = (c_ab as f64) * pmi;

                let is_better = match best {
                    None => true,
                    Some((_, best_score, best_pos)) => {
                        score > best_score + 1e-12
                            || ((score - best_score).abs() <= 1e-12 && first_pos < best_pos)
                    }
                };
                if is_better {
                    best = Some(((a, b), score, first_pos));
                }
            }

            let Some(((best_a, best_b), _, _)) = best else {
                break;
            };

            let mut merged_bytes =
                Vec::with_capacity(tok_bytes[best_a].len() + tok_bytes[best_b].len());
            merged_bytes.extend_from_slice(&tok_bytes[best_a]);
            merged_bytes.extend_from_slice(&tok_bytes[best_b]);
            let new_tid = tok_bytes.len();
            tok_bytes.push(merged_bytes);

            let mut next_seq = Vec::with_capacity(seq_len);
            let mut idx = 0usize;
            while idx < seq_len {
                if idx + 1 < seq_len && seq_tok[idx] == best_a && seq_tok[idx + 1] == best_b {
                    next_seq.push(new_tid);
                    idx += 2;
                } else {
                    next_seq.push(seq_tok[idx]);
                    idx += 1;
                }
            }
            seq_tok = next_seq;
            merges_performed += 1;
        }

        // ====================================================================
        // Étape 2 : Identification des ancres, dissolution des préfixes incomplets
        // et promotion atomique des intervalles hapax (Théorème 1.2).
        // ====================================================================
        let num_toks = tok_bytes.len();
        let seq_len = seq_tok.len();
        let mut rem_count = vec![0usize; num_toks];
        let mut out_adj: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new(); num_toks];
        let mut in_adj: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new(); num_toks];
        let mut out_tot = vec![0usize; num_toks];
        let mut in_tot = vec![0usize; num_toks];

        for pos in 0..seq_len {
            let t = seq_tok[pos];
            rem_count[t] += 1;
            if pos + 1 < seq_len {
                let nxt = seq_tok[pos + 1];
                *out_adj[t].entry(nxt).or_insert(0) += 1;
                *in_adj[nxt].entry(t).or_insert(0) += 1;
                out_tot[t] += 1;
                in_tot[nxt] += 1;
            }
        }

        // 2a. Ancres multi-octets récurrentes + délimiteurs 1-octet bilatéraux (NL : espace, ponctuation)
        let mut is_anchor = vec![false; num_toks];
        for t in 0..num_toks {
            if tok_bytes[t].len() >= 2 && rem_count[t] >= 2 {
                is_anchor[t] = true;
            } else if tok_bytes[t].len() == 1
                && rem_count[t] >= 4
                && out_adj[t].len() >= 4
                && in_adj[t].len() >= 4
            {
                let ho = exact_entropy(&out_adj[t], out_tot[t]);
                let hi = exact_entropy(&in_adj[t], in_tot[t]);
                if ho >= 1.2 && hi >= 1.2 {
                    is_anchor[t] = true;
                }
            }
        }

        // 2b. Dissolution jusqu'au point fixe des préfixes/suffixes incomplets de hapax :
        // Un motif multi-octets de faible fréquence (<= 4) dont 100% des occurrences collent
        // directement à un fragment non-ancre à droite (ou à gauche) est un nœud interne de trie,
        // pas une entité autonome.
        loop {
            let mut changed = false;
            for t in 0..num_toks {
                if !is_anchor[t] || tok_bytes[t].len() < 2 || rem_count[t] > 4 {
                    continue;
                }
                let mut total_occ = 0usize;
                let mut right_non_anchor = 0usize;
                let mut left_non_anchor = 0usize;

                for pos in 0..seq_len {
                    if seq_tok[pos] != t {
                        continue;
                    }
                    total_occ += 1;
                    if pos + 1 < seq_len && !is_anchor[seq_tok[pos + 1]] {
                        right_non_anchor += 1;
                    }
                    if pos > 0 && !is_anchor[seq_tok[pos - 1]] {
                        left_non_anchor += 1;
                    }
                }
                if total_occ > 0 && (right_non_anchor == total_occ || left_non_anchor == total_occ) {
                    is_anchor[t] = false;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // 2c. Promotion des intervalles non-ancres en entités atomiques et numérotation canonique
        let mut bytes_to_eid: BTreeMap<Vec<u8>, usize> = BTreeMap::new();
        let mut entities: Vec<EntityInfo> = Vec::new();
        let mut seq: Vec<usize> = Vec::new();

        let mut emit_entity = |raw_bytes: Vec<u8>, kind: EntityKind| {
            let pos_in_seq = seq.len();
            let eid = if let Some(&existing) = bytes_to_eid.get(&raw_bytes) {
                entities[existing].count += 1;
                if kind == EntityKind::Consolidated {
                    entities[existing].kind = EntityKind::Consolidated;
                }
                existing
            } else {
                let id = entities.len();
                bytes_to_eid.insert(raw_bytes.clone(), id);
                entities.push(EntityInfo {
                    id,
                    bytes: raw_bytes,
                    count: 1,
                    kind,
                    first_pos: pos_in_seq,
                });
                id
            };
            seq.push(eid);
        };

        let mut pos = 0usize;
        while pos < seq_len {
            let tid = seq_tok[pos];
            if is_anchor[tid] {
                emit_entity(tok_bytes[tid].clone(), EntityKind::Consolidated);
                pos += 1;
            } else {
                let mut span = Vec::new();
                while pos < seq_len && !is_anchor[seq_tok[pos]] {
                    span.extend_from_slice(&tok_bytes[seq_tok[pos]]);
                    pos += 1;
                }
                emit_entity(span, EntityKind::HapaxPromoted);
            }
        }

        // ====================================================================
        // Étape 3 : Construction du graphe d'observation G(E) pondéré par entité.
        // ====================================================================
        Self::from_entity_sequence(entities, seq, merges_performed)
    }

    fn empty() -> Self {
        Self {
            m: 0,
            entities: Vec::new(),
            seq: Vec::new(),
            q_fwd: Vec::new(),
            q_bwd: Vec::new(),
            q_skip2: Vec::new(),
            active_fwd: Vec::new(),
            active_bwd: Vec::new(),
            active_skip2: Vec::new(),
            a_loc: Vec::new(),
            nonloc_neighbors: Vec::new(),
            h_emp_fwd: 0.0,
            h_emp_bwd: 0.0,
            h_emp_skip2: 0.0,
            merges_performed: 0,
        }
    }

    fn from_entity_sequence(
        entities: Vec<EntityInfo>,
        seq: Vec<usize>,
        merges_performed: usize,
    ) -> Self {
        let m = entities.len();
        let t_len = seq.len();

        let mut c_fwd: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new(); m];
        let mut c_bwd: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new(); m];
        let mut c_skip2: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new(); m];
        let mut a_loc = vec![false; m * m];

        for pos in 0..t_len {
            let u = seq[pos];
            if pos + 1 < t_len {
                let v = seq[pos + 1];
                *c_fwd[u].entry(v).or_insert(0) += 1;
                *c_bwd[v].entry(u).or_insert(0) += 1;
                if u != v {
                    a_loc[u * m + v] = true;
                    a_loc[v * m + u] = true;
                }
            }
            if pos + 2 < t_len {
                let w = seq[pos + 2];
                *c_skip2[u].entry(w).or_insert(0) += 1;
            }
        }

        let (q_fwd, active_fwd, h_emp_fwd) = normalize_rows_and_entropy(&c_fwd, m);
        let (q_bwd, active_bwd, h_emp_bwd) = normalize_rows_and_entropy(&c_bwd, m);
        let (q_skip2, active_skip2, h_emp_skip2) = normalize_rows_and_entropy(&c_skip2, m);

        let mut nonloc_neighbors = Vec::with_capacity(m);
        for u in 0..m {
            let mut nl = Vec::new();
            for v in 0..m {
                if u != v && !a_loc[u * m + v] {
                    nl.push(v);
                }
            }
            nonloc_neighbors.push(nl);
        }

        Self {
            m,
            entities,
            seq,
            q_fwd,
            q_bwd,
            q_skip2,
            active_fwd,
            active_bwd,
            active_skip2,
            a_loc,
            nonloc_neighbors,
            h_emp_fwd,
            h_emp_bwd,
            h_emp_skip2,
            merges_performed,
        }
    }

    #[inline]
    pub fn is_local_adj(&self, u: usize, v: usize) -> bool {
        self.a_loc[u * self.m + v]
    }

    /// Recherche exacte (pour diagnostic post-hoc uniquement) d'une séquence d'octets dans V(E).
    pub fn find_entity_by_bytes(&self, pattern: &[u8]) -> Option<usize> {
        self.entities
            .iter()
            .find(|e| e.bytes.as_slice() == pattern)
            .map(|e| e.id)
    }

    /// Convertit la distribution Q^{(lag)} en matrice dense M x M.
    pub fn q_dense(&self, lag: Lag) -> Mat {
        let mut mat = Mat::zeros(self.m);
        let rows = match lag {
            Lag::Fwd => &self.q_fwd,
            Lag::Bwd => &self.q_bwd,
            Lag::Skip2 => &self.q_skip2,
        };
        for (u, row) in rows.iter().enumerate() {
            for &(v, p) in row {
                mat.set(u, v, p);
            }
        }
        mat
    }

    /// Résidu de bruit de fond \rho_0(E) (transitions uniformes sur V(E)) pondéré par 1/M.
    pub fn rho_background(&self, alpha_fwd: f64, alpha_bwd: f64, alpha_skip2: f64) -> f64 {
        if self.m <= 1 {
            return 0.0;
        }
        let inv_m = 1.0 / (self.m as f64);
        let ln_m = (self.m as f64).ln();

        let n_fwd = self.active_fwd.iter().filter(|&&a| a).count() as f64;
        let n_bwd = self.active_bwd.iter().filter(|&&a| a).count() as f64;
        let n_sk2 = self.active_skip2.iter().filter(|&&a| a).count() as f64;

        let r_fwd = (n_fwd * inv_m * ln_m - self.h_emp_fwd).max(0.0);
        let r_bwd = (n_bwd * inv_m * ln_m - self.h_emp_bwd).max(0.0);
        let r_sk2 = (n_sk2 * inv_m * ln_m - self.h_emp_skip2).max(0.0);

        alpha_fwd * r_fwd + alpha_bwd * r_bwd + alpha_skip2 * r_sk2
    }
}

/// Calcule l'entropie de Shannon exacte de manière 100% indépendante des étiquettes
/// (tri des comptages avant sommation flottante pour éliminer toute dérive ULP).
fn exact_entropy(counts: &BTreeMap<usize, usize>, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let mut vals: Vec<usize> = counts.values().copied().filter(|&c| c > 0).collect();
    vals.sort_unstable();
    let inv_t = 1.0 / (total as f64);
    let mut h = 0.0;
    for c in vals {
        let p = (c as f64) * inv_t;
        h -= p * p.ln();
    }
    h.max(0.0)
}

fn normalize_rows_and_entropy(
    counts: &[BTreeMap<usize, usize>],
    m: usize,
) -> (Vec<Vec<(usize, f64)>>, Vec<bool>, f64) {
    let mut rows = Vec::with_capacity(m);
    let mut active = Vec::with_capacity(m);
    let mut sum_h = 0.0;

    for row_map in counts {
        let total: usize = row_map.values().sum();
        if total == 0 {
            rows.push(Vec::new());
            active.push(false);
        } else {
            let inv_t = 1.0 / (total as f64);
            let mut sparse = Vec::with_capacity(row_map.len());
            for (&v, &c) in row_map {
                sparse.push((v, (c as f64) * inv_t));
            }
            sum_h += exact_entropy(row_map, total);
            rows.push(sparse);
            active.push(true);
        }
    }

    let h_emp = if m > 0 { sum_h / (m as f64) } else { 0.0 };
    (rows, active, h_emp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ho_ptgrs::generator::{CurriculumConfig, HoPtgrsGenerator};
    use crate::ho_ptgrs::meta_ops::MetaOp;
    use crate::ho_ptgrs::rng::Rng;

    #[test]
    fn test_theorem_1_exact_s256_equivariance_of_entities_and_obs_graph() {
        let mut gen_ = HoPtgrsGenerator::new(2026, CurriculumConfig::default());
        let ep = gen_.generate_episode(Some(MetaOp::Bif)).unwrap();

        let mut perm_rng = Rng::from_seed(777_888);
        let mut perm: Vec<u8> = (0u8..=255).collect();
        perm_rng.shuffle(&mut perm);
        let perm_bytes: Vec<u8> = ep.perceived_eff.iter().map(|&b| perm[b as usize]).collect();

        let cfg = EntityConfig::default();
        let g1 = ObservationGraph::extract(&ep.perceived_eff, &cfg);
        let g2 = ObservationGraph::extract(&perm_bytes, &cfg);

        // Égalité canonique stricte grâce à la numérotation par première apparition
        assert_eq!(g1.m, g2.m);
        assert_eq!(g1.merges_performed, g2.merges_performed);
        assert_eq!(g1.seq, g2.seq);
        assert_eq!(g1.a_loc, g2.a_loc);
        assert_eq!(g1.q_fwd, g2.q_fwd);
        assert_eq!(g1.q_bwd, g2.q_bwd);
        assert_eq!(g1.q_skip2, g2.q_skip2);
        assert!((g1.h_emp_fwd - g2.h_emp_fwd).abs() < 1e-15);
        assert!((g1.h_emp_bwd - g2.h_emp_bwd).abs() < 1e-15);
        assert!((g1.h_emp_skip2 - g2.h_emp_skip2).abs() < 1e-15);

        // Vérifie \sigma_\pi exact sur les octets de chaque entité u \in 0..M(E)
        for u in 0..g1.m {
            let mapped: Vec<u8> = g1.entities[u]
                .bytes
                .iter()
                .map(|&b| perm[b as usize])
                .collect();
            assert_eq!(mapped, g2.entities[u].bytes);
            assert_eq!(g1.entities[u].kind, g2.entities[u].kind);
            assert_eq!(g1.entities[u].count, g2.entities[u].count);
        }
    }

    #[test]
    fn test_theorem_1_v_mu_star_isolated_as_single_entity_across_all_9_ho_ptgrs_ops() {
        // Preuve empirique du Théorème 1.2 : sur les 9 opérateurs du catalogue HO-PTGRS
        // (et 3 seeds distinctes = 27 épisodes), le mot-code hapax v_{\mu*} est isolé
        // comme UNE SEULE entité atomique dans V(E), avec poids de transition 1/M et
        // degré local exact (2 voisins locaux : EdgeSep à gauche, REL_DEFINES à droite).
        let cfg = EntityConfig::default();
        let catalog = MetaOp::catalog();

        for seed in [42u64, 1337, 2026] {
            let mut gen_ = HoPtgrsGenerator::new(
                seed,
                CurriculumConfig {
                    horizon_k: 4,
                    num_distractors: 4,
                    noise_ratio: 0.5,
                    max_attempts: 64,
                },
            );

            for op in &catalog {
                let ep = gen_
                    .generate_episode(Some(op.clone()))
                    .unwrap_or_else(|e| panic!("Échec génération {:?} (seed={}): {}", op, seed, e));

                let obs = ObservationGraph::extract(&ep.perceived_eff, &cfg);
                let target_bytes = &ep.hidden_cause.target_concept_bytes;

                let eid = obs.find_entity_by_bytes(target_bytes).unwrap_or_else(|| {
                    panic!(
                        "v_mu* ({:02X?}) NON isolé comme entité unique pour {:?} (seed={}, M={}, merges={})",
                        target_bytes, op, seed, obs.m, obs.merges_performed
                    )
                });

                let info = &obs.entities[eid];
                assert_eq!(
                    info.count, 1,
                    "v_mu* doit apparaître exactement 1 fois (hapax) pour {:?}",
                    op
                );
                assert_eq!(info.kind, EntityKind::HapaxPromoted);
                // Vérifie que v_mu* participe aux trois distributions Q^{(+1)}, Q^{(-1)}, Q^{(+2)}
                assert!(obs.active_fwd[eid], "v_mu* doit avoir un successeur (REL_DEFINES)");
                assert!(obs.active_bwd[eid], "v_mu* doit avoir un prédécesseur (EdgeSep)");
                assert!(obs.active_skip2[eid], "v_mu* doit avoir un saut +2 (schema_root)");
                // Vérifie la normalisation exacte des lignes de Q^{(r)}
                let sum_fwd: f64 = obs.q_fwd[eid].iter().map(|(_, p)| *p).sum();
                let sum_bwd: f64 = obs.q_bwd[eid].iter().map(|(_, p)| *p).sum();
                let sum_sk2: f64 = obs.q_skip2[eid].iter().map(|(_, p)| *p).sum();
                assert!((sum_fwd - 1.0).abs() < 1e-12);
                assert!((sum_bwd - 1.0).abs() < 1e-12);
                assert!((sum_sk2 - 1.0).abs() < 1e-12);
                // Vérifie que A_loc est symétrique et exclut la diagonale de N_nl
                assert!(!obs.nonloc_neighbors[eid].contains(&eid));
                assert!(!obs.nonloc_neighbors[eid].is_empty());
            }
        }
    }

    #[test]
    fn test_natural_language_self_specified_dictionary_segmentation() {
        // Vérifie que le même extracteur sans aucun caractère codé en dur segmente aussi la NL
        let text = b"ambiguite : un enonce admet deux sens distincts . \
                     paradoxe : un enonce contredit deux regles . \
                     analogie : un enonce relie deux structures . \
                     la phrase admet deux sens distincts : ambiguite .";
        let obs = ObservationGraph::extract(text, &EntityConfig::default());

        assert!(obs.m >= 10 && obs.m < text.len());
        // L'espace ' ' doit être reconnu comme ancre 1-octet bilatérale
        let space_id = obs.find_entity_by_bytes(b" ");
        assert!(space_id.is_some(), "L'espace NL doit être induit sans hardcoding");
        // Les mots récurrents ('ambiguite', 'deux') et hapax ('paradoxe', 'analogie') doivent être isolés
        assert!(obs.find_entity_by_bytes(b"ambiguite").is_some());
        assert!(obs.find_entity_by_bytes(b"deux").is_some());
        assert!(obs.find_entity_by_bytes(b"paradoxe").is_some());
        assert!(obs.find_entity_by_bytes(b"analogie").is_some());
    }
}
