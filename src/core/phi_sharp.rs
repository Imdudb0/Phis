use crate::core::spectral::{CausalLaplacian, FiedlerResult};
use crate::core::tensor::{Mat, RelTensor};
use crate::ho_ptgrs::rng::Rng;
use std::collections::{BTreeMap, BTreeSet};

pub const BYTE_VOCAB: usize = 256;
/// Dimension des signatures relationnelles unigrammes \phi_rel(b; E).
pub const REL_DIM: usize = 10;
/// Dimension des signatures relationnelles de transitions \phi_bi(u, v; E).
pub const BI_DIM: usize = 10;
/// Plafond de classes de transitions conservées individuellement (le reste est replié de manière S_256-invariante).
pub const MAX_BI_CLASSES: usize = 256;

/// Hyperparamètres analytiques de \phi_\sharp(R, E) = \beta_iit * \lambda_2 - \alpha * \rho_KL - \beta * V_stab.
#[derive(Debug, Clone)]
pub struct PhiSharpConfig {
    /// Facteur d'échelle \beta_iit devant \lambda_2(L_{\bar{w}}) pour équilibrer \lambda_2 et \rho_KL.
    pub beta_iit: f64,
    /// Poids \alpha du résidu abductif d'excès KL -\alpha * \rho_KL(R, E).
    pub alpha: f64,
    /// Poids \beta de la fragilité différentielle -\beta * V_stab(R, E).
    pub beta: f64,
    /// Poids relatif de la composante transitions dans \rho_KL.
    pub lambda_bi: f64,
    /// Rayon de lissage C^1 de la charnière \psi_\tau sur \Delta_{ij} et V_stab.
    pub hinge_tau: f64,
    /// Rayon de lissage du plafond smooth_cap(\rho_{ij}, \rho_0(E), \eps).
    pub cap_eps: f64,
    /// Dimension maximale de Krylov pour Lanczos.
    pub max_krylov: usize,
    /// Itérations maximales de raffinement par puissance inverse CG.
    pub max_inv_power: usize,
}

impl Default for PhiSharpConfig {
    fn default() -> Self {
        Self {
            beta_iit: 10.0,
            alpha: 0.5,
            beta: 0.25,
            lambda_bi: 0.5,
            hinge_tau: 1e-4,
            cap_eps: 1e-3,
            max_krylov: 32,
            max_inv_power: 4,
        }
    }
}

/// Résumé relationnel auto-spécifié de `perceived.eff`, strictement équivariant sous S_256.
/// Aucune caractéristique ne dépend de la valeur numérique d'un octet.
#[derive(Debug, Clone)]
pub struct EffectSummary {
    pub n: usize,
    pub len: usize,
    /// Alphabet observé A(E) (taille k_uni <= 256).
    pub alphabet: Vec<u8>,
    pub k_uni: usize,
    /// Table d'indexation octet -> indice dans 0..k_uni (jamais utilisée comme feature).
    pub byte_to_alpha: [Option<usize>; BYTE_VOCAB],
    /// Signatures relationnelles \phi_rel(a; E) : row-major (k_uni x REL_DIM).
    pub rel_feats: Vec<f64>,
    /// Distribution empirique spatialisée Q_{i, a} : row-major (N x k_uni), somme totale = 1.
    pub uni: Vec<f64>,
    pub uni_mass: Vec<f64>,
    /// Entropie empirique irréductible unigramme H_emp^uni(E) >= 0.
    pub h_emp_uni: f64,

    /// Nombre de classes de transitions observées k_bi <= MAX_BI_CLASSES.
    pub k_bi: usize,
    /// Signatures relationnelles de transitions \phi_bi(g; E) : row-major (k_bi x BI_DIM).
    pub bi_feats: Vec<f64>,
    /// Distribution empirique spatialisée des transitions T_{i, g} : row-major (N x k_bi).
    pub bi: Vec<f64>,
    pub bi_mass: Vec<f64>,
    /// Entropie empirique irréductible des transitions H_emp^bi(E) >= 0.
    pub h_emp_bi: f64,
    pub has_bigrams: bool,

    /// Moyennes nodales des signatures relationnelles (N x REL_DIM et N x BI_DIM) pour \Psi_0(E).
    pub node_uni_sig: Vec<f64>,
    pub node_bi_sig: Vec<f64>,
    /// Matrice symétrique S_256-invariante de recouvrement cosinus inter-nœuds O_{ij}(E) \in [0, 1].
    pub node_overlap: Mat,
}

impl EffectSummary {
    /// Construit le résumé relationnel strictement S_256-équivariant depuis les octets bruts de `perceived.eff`.
    pub fn from_bytes(bytes: &[u8], n: usize) -> Self {
        let len = bytes.len();
        if len == 0 || n == 0 {
            return Self::empty(n);
        }

        // 1. Extraction de l'alphabet observé A(E) et comptages de base
        let mut counts = [0usize; BYTE_VOCAB];
        let mut positions: Vec<Vec<usize>> = vec![Vec::new(); BYTE_VOCAB];
        for (pos, &b) in bytes.iter().enumerate() {
            let bi = b as usize;
            counts[bi] += 1;
            positions[bi].push(pos);
        }

        let mut alphabet = Vec::new();
        let mut byte_to_alpha = [None; BYTE_VOCAB];
        for b in 0..BYTE_VOCAB {
            if counts[b] > 0 {
                let idx = alphabet.len();
                alphabet.push(b as u8);
                byte_to_alpha[b] = Some(idx);
            }
        }
        let k_uni = alphabet.len();
        let max_count = counts.iter().copied().max().unwrap_or(1) as f64;
        let ln_len1 = ((len as f64) + 1.0).ln().max(1e-12);
        let ln_kmax = (k_uni.max(2) as f64).ln();

        // 2. Statistiques de voisinage et de transitions (strictement par égalité/fréquence)
        let mut out_counts: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new(); k_uni];
        let mut in_counts: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new(); k_uni];
        let mut out_totals = vec![0usize; k_uni];
        let mut in_totals = vec![0usize; k_uni];
        let mut bigram_counts: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        let mut bigram_pos_sum: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut bigram_pos_sq_sum: BTreeMap<(usize, usize), f64> = BTreeMap::new();

        for pos in 1..len {
            let u = byte_to_alpha[bytes[pos - 1] as usize].unwrap();
            let v = byte_to_alpha[bytes[pos] as usize].unwrap();
            *out_counts[u].entry(v).or_insert(0) += 1;
            *in_counts[v].entry(u).or_insert(0) += 1;
            out_totals[u] += 1;
            in_totals[v] += 1;

            *bigram_counts.entry((u, v)).or_insert(0) += 1;
            let x = ((pos as f64) + 0.5) / (len as f64);
            *bigram_pos_sum.entry((u, v)).or_insert(0.0) += x;
            *bigram_pos_sq_sum.entry((u, v)).or_insert(0.0) += x * x;
        }

        // Voisins distincts dans la fenêtre +/- 2
        let mut win_neighbors: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); k_uni];
        for pos in 0..len {
            let u = byte_to_alpha[bytes[pos] as usize].unwrap();
            let lo = pos.saturating_sub(2);
            let hi = (pos + 2).min(len - 1);
            for p2 in lo..=hi {
                if p2 != pos {
                    let v = byte_to_alpha[bytes[p2] as usize].unwrap();
                    win_neighbors[u].insert(v);
                }
            }
        }

        // 3. Calcul des signatures relationnelles \phi_rel(a; E) \in R^{REL_DIM}
        let mut rel_feats = vec![0.0; k_uni * REL_DIM];
        for a in 0..k_uni {
            let b = alphabet[a] as usize;
            let c = counts[b] as f64;
            let pos_list = &positions[b];

            // f0: log-fréquence normalisée
            let f0 = (c + 1.0).ln() / ln_len1;
            // f1: fréquence relative au mode
            let f1 = c / max_count;
            // f2: entropie normalisée des prédécesseurs
            let f2 = normalized_entropy(&in_counts[a], in_totals[a], ln_kmax);
            // f3: entropie normalisée des successeurs (détecte bifurcations vs chaînes)
            let f3 = normalized_entropy(&out_counts[a], out_totals[a], ln_kmax);

            // f4, f5: taux d'auto-retour à lag court [1..2] et moyen [3..8]
            let mut ret_short = 0usize;
            let mut ret_med = 0usize;
            for &p in pos_list {
                if (p + 1 < len && bytes[p + 1] as usize == b)
                    || (p + 2 < len && bytes[p + 2] as usize == b)
                {
                    ret_short += 1;
                }
                let max_d = (p + 8).min(len - 1);
                if p + 3 <= max_d && ((p + 3)..=max_d).any(|q| bytes[q] as usize == b) {
                    ret_med += 1;
                }
            }
            let f4 = (ret_short as f64) / c;
            let f5 = (ret_med as f64) / c;

            // f6, f7, f8: position moyenne, écart-type spatial et étendue normalisée
            let mut mean_x = 0.0;
            for &p in pos_list {
                mean_x += ((p as f64) + 0.5) / (len as f64);
            }
            mean_x /= c;
            let mut var_x = 0.0;
            for &p in pos_list {
                let dx = ((p as f64) + 0.5) / (len as f64) - mean_x;
                var_x += dx * dx;
            }
            let std_x = (var_x / c).sqrt();
            let span_x = ((*pos_list.last().unwrap() - *pos_list.first().unwrap()) as f64)
                / (len as f64);

            // f9: degré de co-occurrence dans la fenêtre +/- 2
            let f9 = (win_neighbors[a].len() as f64) / (k_uni as f64);

            let off = a * REL_DIM;
            rel_feats[off] = f0;
            rel_feats[off + 1] = f1;
            rel_feats[off + 2] = f2;
            rel_feats[off + 3] = f3;
            rel_feats[off + 4] = f4;
            rel_feats[off + 5] = f5;
            rel_feats[off + 6] = mean_x;
            rel_feats[off + 7] = std_x;
            rel_feats[off + 8] = span_x;
            rel_feats[off + 9] = f9;
        }

        // 4. Distribution spatiale unigramme Q_{i, a} et entropie irréductible H_emp^uni(E)
        let mut uni = vec![0.0; n * k_uni];
        let mut uni_mass = vec![0.0; n];
        let inv_l = 1.0 / (len as f64);
        let mut assign = vec![0.0; n];

        for (pos, &b) in bytes.iter().enumerate() {
            spatial_assignment(pos, len, n, &mut assign);
            let a = byte_to_alpha[b as usize].unwrap();
            for i in 0..n {
                let w = assign[i] * inv_l;
                uni[i * k_uni + a] += w;
                uni_mass[i] += w;
            }
        }
        let h_emp_uni = compute_empirical_entropy(&uni, &uni_mass, n, k_uni);

        // 5. Transitions observées et signatures \phi_bi(g; E) (repliement strictement S_256-invariant)
        let has_bigrams = len >= 2 && !bigram_counts.is_empty();
        let (k_bi, bi_feats, bi, bi_mass, h_emp_bi) = if has_bigrams {
            // Déterminer un seuil de fréquence strictement invariant par permutation si > MAX_BI_CLASSES
            let threshold = if bigram_counts.len() <= MAX_BI_CLASSES {
                0usize
            } else {
                let mut freqs: Vec<usize> = bigram_counts.values().copied().collect();
                freqs.sort_unstable_by(|a, b| b.cmp(a));
                // Tous les bigrammes avec count > threshold sont <= MAX_BI_CLASSES - 1
                freqs[MAX_BI_CLASSES - 1]
            };

            let mut bi_to_class: BTreeMap<(usize, usize), usize> = BTreeMap::new();
            let mut kept_pairs: Vec<(usize, usize)> = Vec::new();
            let mut tail_pairs: Vec<(usize, usize)> = Vec::new();

            for (&pair, &cnt) in &bigram_counts {
                if cnt > threshold {
                    let cid = kept_pairs.len();
                    kept_pairs.push(pair);
                    bi_to_class.insert(pair, cid);
                } else {
                    tail_pairs.push(pair);
                }
            }

            let has_tail = !tail_pairs.is_empty();
            let tail_cid = kept_pairs.len();
            if has_tail {
                for &pair in &tail_pairs {
                    bi_to_class.insert(pair, tail_cid);
                }
            }
            let k_bi = kept_pairs.len() + if has_tail { 1 } else { 0 };
            let mut bi_feats = vec![0.0; k_bi * BI_DIM];

            let compute_pair_feat = |u: usize, v: usize| -> [f64; BI_DIM] {
                let c_uv = bigram_counts[&(u, v)] as f64;
                let c_vu = bigram_counts.get(&(v, u)).copied().unwrap_or(0) as f64;
                let c_u_tot = (counts[alphabet[u] as usize] as f64).max(1.0);
                let c_v_tot = (counts[alphabet[v] as usize] as f64).max(1.0);
                let c_u_out = (out_totals[u] as f64).max(1.0);
                let c_v_in = (in_totals[v] as f64).max(1.0);

                let g0 = (c_uv + 1.0).ln() / ln_len1;
                let pmi = ((c_uv * (len as f64)) / (c_u_tot * c_v_tot)).max(1e-12).ln();
                let g1 = (pmi / 3.0).tanh();
                let g2 = c_uv / c_u_out;
                let g3 = c_uv / c_v_in;
                let g4 = c_vu / (c_uv + c_vu);

                let mean_p = bigram_pos_sum[&(u, v)] / c_uv;
                let mean_sq = bigram_pos_sq_sum[&(u, v)] / c_uv;
                let std_p = (mean_sq - mean_p * mean_p).max(0.0).sqrt();
                let g7 = if u == v { 1.0 } else { 0.0 };
                let g8 = rel_feats[v * REL_DIM + 3] - rel_feats[u * REL_DIM + 3];
                let g9 = rel_feats[v * REL_DIM] - rel_feats[u * REL_DIM];

                [g0, g1, g2, g3, g4, mean_p, std_p, g7, g8, g9]
            };

            for (cid, &(u, v)) in kept_pairs.iter().enumerate() {
                let f = compute_pair_feat(u, v);
                bi_feats[cid * BI_DIM..(cid + 1) * BI_DIM].copy_from_slice(&f);
            }
            if has_tail {
                let mut acc = [0.0f64; BI_DIM];
                let mut w_sum = 0.0;
                for &(u, v) in &tail_pairs {
                    let w = bigram_counts[&(u, v)] as f64;
                    let f = compute_pair_feat(u, v);
                    for m in 0..BI_DIM {
                        acc[m] += w * f[m];
                    }
                    w_sum += w;
                }
                let inv_w = 1.0 / w_sum.max(1.0);
                for m in 0..BI_DIM {
                    bi_feats[tail_cid * BI_DIM + m] = acc[m] * inv_w;
                }
            }

            let mut bi = vec![0.0; n * k_bi];
            let mut bi_mass = vec![0.0; n];
            let inv_bm = 1.0 / ((len - 1) as f64);
            for pos in 1..len {
                spatial_assignment(pos, len, n, &mut assign);
                let u = byte_to_alpha[bytes[pos - 1] as usize].unwrap();
                let v = byte_to_alpha[bytes[pos] as usize].unwrap();
                let cid = bi_to_class[&(u, v)];
                for i in 0..n {
                    let w = assign[i] * inv_bm;
                    bi[i * k_bi + cid] += w;
                    bi_mass[i] += w;
                }
            }
            let h_emp_bi = compute_empirical_entropy(&bi, &bi_mass, n, k_bi);
            (k_bi, bi_feats, bi, bi_mass, h_emp_bi)
        } else {
            (0, Vec::new(), Vec::new(), vec![0.0; n], 0.0)
        };

        // 6. Moyennes nodales des signatures relationnelles et recouvrement cosinus inter-nœuds
        let mut node_uni_sig = vec![0.0; n * REL_DIM];
        let mut node_bi_sig = vec![0.0; n * BI_DIM];
        let mut node_norms = vec![0.0; n];

        for i in 0..n {
            let um = uni_mass[i];
            if um > 1e-12 {
                let inv_um = 1.0 / um;
                let q_row = &uni[i * k_uni..(i + 1) * k_uni];
                let mut sq = 0.0;
                for a in 0..k_uni {
                    let p = q_row[a] * inv_um;
                    sq += p * p;
                    if p > 0.0 {
                        for m in 0..REL_DIM {
                            node_uni_sig[i * REL_DIM + m] += p * rel_feats[a * REL_DIM + m];
                        }
                    }
                }
                node_norms[i] = sq.sqrt();
            }
            if has_bigrams && bi_mass[i] > 1e-12 {
                let inv_bm = 1.0 / bi_mass[i];
                let t_row = &bi[i * k_bi..(i + 1) * k_bi];
                for g in 0..k_bi {
                    let p = t_row[g] * inv_bm;
                    if p > 0.0 {
                        for m in 0..BI_DIM {
                            node_bi_sig[i * BI_DIM + m] += p * bi_feats[g * BI_DIM + m];
                        }
                    }
                }
            }
        }

        let mut node_overlap = Mat::zeros(n);
        for i in 0..n {
            for j in i..n {
                let denom = node_norms[i] * node_norms[j];
                let cos = if denom > 1e-12 && uni_mass[i] > 1e-12 && uni_mass[j] > 1e-12 {
                    let inv_i = 1.0 / uni_mass[i];
                    let inv_j = 1.0 / uni_mass[j];
                    let qi = &uni[i * k_uni..(i + 1) * k_uni];
                    let qj = &uni[j * k_uni..(j + 1) * k_uni];
                    let mut dot = 0.0;
                    for a in 0..k_uni {
                        dot += (qi[a] * inv_i) * (qj[a] * inv_j);
                    }
                    (dot / denom).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                node_overlap.set(i, j, cos);
                node_overlap.set(j, i, cos);
            }
        }

        Self {
            n,
            len,
            alphabet,
            k_uni,
            byte_to_alpha,
            rel_feats,
            uni,
            uni_mass,
            h_emp_uni,
            k_bi,
            bi_feats,
            bi,
            bi_mass,
            h_emp_bi,
            has_bigrams,
            node_uni_sig,
            node_bi_sig,
            node_overlap,
        }
    }

    fn empty(n: usize) -> Self {
        Self {
            n,
            len: 0,
            alphabet: Vec::new(),
            k_uni: 0,
            byte_to_alpha: [None; BYTE_VOCAB],
            rel_feats: Vec::new(),
            uni: Vec::new(),
            uni_mass: vec![0.0; n],
            h_emp_uni: 0.0,
            k_bi: 0,
            bi_feats: Vec::new(),
            bi: Vec::new(),
            bi_mass: vec![0.0; n],
            h_emp_bi: 0.0,
            has_bigrams: false,
            node_uni_sig: vec![0.0; n * REL_DIM],
            node_bi_sig: vec![0.0; n * BI_DIM],
            node_overlap: Mat::zeros(n),
        }
    }

    /// Résidu de bruit de fond \rho_0(E) = D_KL(Q_E || Uniforme(A(E))) >= 0.
    pub fn rho_background(&self, lambda_bi: f64) -> f64 {
        let (w_uni, w_bi) = residual_weights(self.has_bigrams, lambda_bi);
        let rho0_u = if self.k_uni > 0 {
            ((self.k_uni as f64).ln() - self.h_emp_uni).max(0.0)
        } else {
            0.0
        };
        let rho0_b = if w_bi > 0.0 && self.k_bi > 0 {
            ((self.k_bi as f64).ln() - self.h_emp_bi).max(0.0)
        } else {
            0.0
        };
        w_uni * rho0_u + w_bi * rho0_b
    }
}

fn normalized_entropy(counts: &BTreeMap<usize, usize>, total: usize, ln_kmax: f64) -> f64 {
    if total == 0 || ln_kmax <= 0.0 {
        return 0.0;
    }
    let inv_t = 1.0 / (total as f64);
    let mut h = 0.0;
    for &c in counts.values() {
        if c > 0 {
            let p = (c as f64) * inv_t;
            h -= p * p.ln();
        }
    }
    (h / ln_kmax).clamp(0.0, 1.0)
}

fn compute_empirical_entropy(dist: &[f64], mass: &[f64], n: usize, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let mut h_total = 0.0;
    for i in 0..n {
        let m = mass[i];
        if m <= 1e-15 {
            continue;
        }
        let inv_m = 1.0 / m;
        let row = &dist[i * k..(i + 1) * k];
        for &q in row {
            if q > 1e-15 {
                h_total -= q * (q * inv_m).ln();
            }
        }
    }
    h_total.max(0.0)
}

/// Partition de l'unité gaussienne lisse alignant la position `pos` sur les N nœuds.
pub fn spatial_assignment(pos: usize, len: usize, n: usize, out: &mut [f64]) {
    let x = ((pos as f64) + 0.5) / (len as f64);
    let sigma = (0.65 / (n as f64)).max(1e-4);
    let inv_2sig2 = 0.5 / (sigma * sigma);

    let mut max_logit = f64::NEG_INFINITY;
    for i in 0..n {
        let center = ((i as f64) + 0.5) / (n as f64);
        let d = x - center;
        let v = -d * d * inv_2sig2;
        out[i] = v;
        if v > max_logit {
            max_logit = v;
        }
    }
    let mut sum = 0.0;
    for v in out.iter_mut() {
        *v = (*v - max_logit).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in out.iter_mut() {
        *v *= inv_sum;
    }
}

/// Plongements relationnels précalculés pour un épisode E (indépendants de R).
#[derive(Debug, Clone)]
pub struct PrecomputedHeads {
    pub k_uni: usize,
    /// U_uni[a * h + r] = (W_rel_uni * \phi_rel(a))_r
    pub u_uni: Vec<f64>,
    /// u0_uni[a] = <b_rel_uni, \phi_rel(a)>
    pub u0_uni: Vec<f64>,
    pub k_bi: usize,
    /// U_bi[g * h + r] = (W_rel_bi * \phi_bi(g))_r
    pub u_bi: Vec<f64>,
    /// u0_bi[g] = <b_rel_bi, \phi_bi(g)>
    pub u0_bi: Vec<f64>,
}

/// Paramètres appris \theta strictement S_256-équivariants (aucune table d'octets 0..255).
#[derive(Debug, Clone)]
pub struct ModelParams {
    pub n: usize,
    pub d: usize,
    pub h: usize,
    /// Poids directs W_dir (h x d) : R_{ij, c} -> pré-activation d'arête.
    pub w_dir: Vec<f64>,
    /// Poids de composition 2-FWL W_comp (h x d) : (R_c R_c)_{ij} -> pré-activation d'arête.
    pub w_comp: Vec<f64>,
    /// Biais d'arête b_edge (h).
    pub b_edge: Vec<f64>,
    /// Appariement bilinéaire unigramme W_rel_uni (h x REL_DIM) et biais b_rel_uni (REL_DIM).
    pub w_rel_uni: Vec<f64>,
    pub b_rel_uni: Vec<f64>,
    /// Appariement bilinéaire transitions W_rel_bi (h x BI_DIM) et biais b_rel_bi (BI_DIM).
    pub w_rel_bi: Vec<f64>,
    pub b_rel_bi: Vec<f64>,
    /// Encodeur du pôle régional \Psi_0(E) sur signatures relationnelles :
    /// w_enc_uni (d x REL_DIM), w_enc_bi (d x BI_DIM), couplage asymétrique (d x d), poids recouvrement (d).
    pub w_enc_uni: Vec<f64>,
    pub w_enc_bi: Vec<f64>,
    pub w_init_asym: Vec<f64>,
    pub w_init_ov: Vec<f64>,
    /// Matrice de poids non-réciproques W_sol (N x N) générant le champ solénoïdal A_\perp.
    pub w_sol: Mat,
    /// Couplage de l'état R dans le champ solénoïdal (d).
    pub nu_sol: Vec<f64>,
}

impl ModelParams {
    pub fn random_init(n: usize, d: usize, h: usize, rng: &mut Rng) -> Self {
        let mut init_vec = |len: usize, scale: f64| -> Vec<f64> {
            (0..len)
                .map(|_| (rng.gen_f64() * 2.0 - 1.0) * scale)
                .collect()
        };

        let s_edge = (1.0 / (2.0 * d as f64)).sqrt();
        let s_rel = (1.0 / ((h * REL_DIM) as f64).sqrt()).max(0.35);
        let s_enc = (1.0 / (REL_DIM as f64)).sqrt();

        let mut w_sol = Mat::zeros(n);
        let s_sol = 0.2 / (n as f64).sqrt();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    w_sol.set(i, j, (rng.gen_f64() * 2.0 - 1.0) * s_sol);
                }
            }
        }

        Self {
            n,
            d,
            h,
            w_dir: init_vec(h * d, s_edge),
            w_comp: init_vec(h * d, s_edge * 0.5),
            b_edge: vec![0.0; h],
            w_rel_uni: init_vec(h * REL_DIM, s_rel),
            b_rel_uni: vec![0.0; REL_DIM],
            w_rel_bi: init_vec(h * BI_DIM, s_rel),
            b_rel_bi: vec![0.0; BI_DIM],
            w_enc_uni: init_vec(d * REL_DIM, s_enc),
            w_enc_bi: init_vec(d * BI_DIM, s_enc),
            w_init_asym: init_vec(d * d, 0.5 / (d as f64).sqrt()),
            w_init_ov: init_vec(d, 0.3),
            w_sol,
            nu_sol: init_vec(d, 0.15),
        }
    }

    /// Précalcule les matrices d'appariement relationnel U = W_rel * \phi_rel pour l'épisode E.
    pub fn precompute_heads(&self, effect: &EffectSummary) -> PrecomputedHeads {
        let h = self.h;

        let mut u_uni = vec![0.0; effect.k_uni * h];
        let mut u0_uni = vec![0.0; effect.k_uni];
        for a in 0..effect.k_uni {
            let feat = &effect.rel_feats[a * REL_DIM..(a + 1) * REL_DIM];
            let mut b0 = 0.0;
            for m in 0..REL_DIM {
                b0 += self.b_rel_uni[m] * feat[m];
            }
            u0_uni[a] = b0;
            for r in 0..h {
                let w_row = &self.w_rel_uni[r * REL_DIM..(r + 1) * REL_DIM];
                let mut acc = 0.0;
                for m in 0..REL_DIM {
                    acc += w_row[m] * feat[m];
                }
                u_uni[a * h + r] = acc;
            }
        }

        let mut u_bi = vec![0.0; effect.k_bi * h];
        let mut u0_bi = vec![0.0; effect.k_bi];
        for g in 0..effect.k_bi {
            let feat = &effect.bi_feats[g * BI_DIM..(g + 1) * BI_DIM];
            let mut b0 = 0.0;
            for m in 0..BI_DIM {
                b0 += self.b_rel_bi[m] * feat[m];
            }
            u0_bi[g] = b0;
            for r in 0..h {
                let w_row = &self.w_rel_bi[r * BI_DIM..(r + 1) * BI_DIM];
                let mut acc = 0.0;
                for m in 0..BI_DIM {
                    acc += w_row[m] * feat[m];
                }
                u_bi[g * h + r] = acc;
            }
        }

        PrecomputedHeads {
            k_uni: effect.k_uni,
            u_uni,
            u0_uni,
            k_bi: effect.k_bi,
            u_bi,
            u0_bi,
        }
    }

    /// Encode le pôle régional initial R(0) = \Psi_0(E) \in B_K (non-réciproque et S_256-invariant).
    pub fn encode_regional_pole(&self, effect: &EffectSummary, radius_k: f64) -> RelTensor {
        debug_assert_eq!(effect.n, self.n);
        let n = self.n;
        let d = self.d;

        let mut sigs = vec![0.0; n * d];
        for i in 0..n {
            let u_sig = &effect.node_uni_sig[i * REL_DIM..(i + 1) * REL_DIM];
            let b_sig = &effect.node_bi_sig[i * BI_DIM..(i + 1) * BI_DIM];
            for c in 0..d {
                let mut acc = 0.0;
                let w_u = &self.w_enc_uni[c * REL_DIM..(c + 1) * REL_DIM];
                let w_b = &self.w_enc_bi[c * BI_DIM..(c + 1) * BI_DIM];
                for m in 0..REL_DIM {
                    acc += w_u[m] * u_sig[m];
                }
                if effect.has_bigrams {
                    for m in 0..BI_DIM {
                        acc += 0.5 * w_b[m] * b_sig[m];
                    }
                }
                sigs[i * d + c] = acc.tanh();
            }
        }

        // Transformation asymétrique t_j = W_init_asym * s_j pour briser la symétrie dès t = 0
        let mut target_sigs = vec![0.0; n * d];
        for j in 0..n {
            for c in 0..d {
                let mut acc = 0.0;
                for a in 0..d {
                    acc += self.w_init_asym[c * d + a] * sigs[j * d + a];
                }
                target_sigs[j * d + c] = acc;
            }
        }

        let mut r0 = RelTensor::zeros(n, d);
        for c in 0..d {
            let w_ov = self.w_init_ov[c];
            for i in 0..n {
                for j in 0..n {
                    let ov = effect.node_overlap.get(i, j);
                    let raw = sigs[i * d + c] * target_sigs[j * d + c]
                        + 0.25 * (sigs[i * d + c] - sigs[j * d + c])
                        + w_ov * ov;
                    r0.set(i, j, c, raw.tanh() * 0.35);
                }
            }
        }
        r0.clamp_to_ball(radius_k);
        r0
    }

    /// Calcule le résidu d'excès KL \rho_KL(R, E) >= 0 (passe avant seule).
    pub fn residual_only(&self, r: &RelTensor, effect: &EffectSummary, lambda_bi: f64) -> f64 {
        let heads = self.precompute_heads(effect);
        self.residual_only_precomputed(r, effect, &heads, lambda_bi)
    }

    pub fn residual_only_precomputed(
        &self,
        r: &RelTensor,
        effect: &EffectSummary,
        heads: &PrecomputedHeads,
        lambda_bi: f64,
    ) -> f64 {
        if effect.k_uni == 0 {
            return 0.0;
        }
        let (w_uni, w_bi) = residual_weights(effect.has_bigrams, lambda_bi);
        let n = self.n;
        let d = self.d;
        let h = self.h;
        let inv_sqrt_n = 1.0 / (n as f64).sqrt();

        let comp: Vec<Mat> = (0..d)
            .map(|c| r.channels[c].matmul(&r.channels[c]))
            .collect();

        let mut h_node = vec![0.0; h];
        let mut logits = vec![0.0; heads.k_uni.max(heads.k_bi).max(1)];
        let mut ce_uni = 0.0;
        let mut ce_bi = 0.0;

        for i in 0..n {
            h_node.fill(0.0);
            for j in 0..n {
                for r_idx in 0..h {
                    let mut z = self.b_edge[r_idx];
                    let w_d = &self.w_dir[r_idx * d..(r_idx + 1) * d];
                    let w_c = &self.w_comp[r_idx * d..(r_idx + 1) * d];
                    for c in 0..d {
                        z += w_d[c] * r.get(i, j, c) + w_c[c] * comp[c].get(i, j);
                    }
                    h_node[r_idx] += z.tanh();
                }
            }
            for val in &mut h_node {
                *val *= inv_sqrt_n;
            }

            if effect.uni_mass[i] > 0.0 {
                ce_uni += bilinear_cross_entropy(
                    &h_node,
                    &heads.u_uni,
                    &heads.u0_uni,
                    &effect.uni[i * heads.k_uni..(i + 1) * heads.k_uni],
                    effect.uni_mass[i],
                    &mut logits[..heads.k_uni],
                );
            }
            if w_bi > 0.0 && heads.k_bi > 0 && effect.bi_mass[i] > 0.0 {
                ce_bi += bilinear_cross_entropy(
                    &h_node,
                    &heads.u_bi,
                    &heads.u0_bi,
                    &effect.bi[i * heads.k_bi..(i + 1) * heads.k_bi],
                    effect.bi_mass[i],
                    &mut logits[..heads.k_bi],
                );
            }
        }

        let kl_uni = (ce_uni - effect.h_emp_uni).max(0.0);
        let kl_bi = if w_bi > 0.0 {
            (ce_bi - effect.h_emp_bi).max(0.0)
        } else {
            0.0
        };
        w_uni * kl_uni + w_bi * kl_bi
    }

    /// Calcule exactement (\rho_KL(R, E), \nabla_R \rho_KL(R, E)).
    pub fn residual_and_grad_r(
        &self,
        r: &RelTensor,
        effect: &EffectSummary,
        lambda_bi: f64,
    ) -> (f64, RelTensor) {
        let heads = self.precompute_heads(effect);
        self.residual_and_grad_r_precomputed(r, effect, &heads, lambda_bi)
    }

    pub fn residual_and_grad_r_precomputed(
        &self,
        r: &RelTensor,
        effect: &EffectSummary,
        heads: &PrecomputedHeads,
        lambda_bi: f64,
    ) -> (f64, RelTensor) {
        if effect.k_uni == 0 {
            return (0.0, RelTensor::zeros(self.n, self.d));
        }
        let (w_uni, w_bi) = residual_weights(effect.has_bigrams, lambda_bi);
        let n = self.n;
        let d = self.d;
        let h = self.h;
        let inv_sqrt_n = 1.0 / (n as f64).sqrt();

        let comp: Vec<Mat> = (0..d)
            .map(|c| r.channels[c].matmul(&r.channels[c]))
            .collect();

        let mut m_edge = vec![0.0; n * n * h];
        let mut h_nodes = vec![0.0; n * h];

        for i in 0..n {
            for j in 0..n {
                let base = (i * n + j) * h;
                for r_idx in 0..h {
                    let mut z = self.b_edge[r_idx];
                    let w_d = &self.w_dir[r_idx * d..(r_idx + 1) * d];
                    let w_c = &self.w_comp[r_idx * d..(r_idx + 1) * d];
                    for c in 0..d {
                        z += w_d[c] * r.get(i, j, c) + w_c[c] * comp[c].get(i, j);
                    }
                    let m = z.tanh();
                    m_edge[base + r_idx] = m;
                    h_nodes[i * h + r_idx] += m;
                }
            }
            for r_idx in 0..h {
                h_nodes[i * h + r_idx] *= inv_sqrt_n;
            }
        }

        let mut ce_uni = 0.0;
        let mut ce_bi = 0.0;
        let mut grad_h = vec![0.0; n * h];
        let mut probs = vec![0.0; heads.k_uni.max(heads.k_bi).max(1)];

        for i in 0..n {
            let hi = &h_nodes[i * h..(i + 1) * h];
            let ghi = &mut grad_h[i * h..(i + 1) * h];

            if effect.uni_mass[i] > 0.0 {
                ce_uni += bilinear_cross_entropy_and_grad_h(
                    hi,
                    &heads.u_uni,
                    &heads.u0_uni,
                    &effect.uni[i * heads.k_uni..(i + 1) * heads.k_uni],
                    effect.uni_mass[i],
                    w_uni,
                    &mut probs[..heads.k_uni],
                    ghi,
                );
            }
            if w_bi > 0.0 && heads.k_bi > 0 && effect.bi_mass[i] > 0.0 {
                ce_bi += bilinear_cross_entropy_and_grad_h(
                    hi,
                    &heads.u_bi,
                    &heads.u0_bi,
                    &effect.bi[i * heads.k_bi..(i + 1) * heads.k_bi],
                    effect.bi_mass[i],
                    w_bi,
                    &mut probs[..heads.k_bi],
                    ghi,
                );
            }
        }

        let mut grad_r = RelTensor::zeros(n, d);
        let mut grad_comp: Vec<Mat> = (0..d).map(|_| Mat::zeros(n)).collect();

        for i in 0..n {
            let ghi = &grad_h[i * h..(i + 1) * h];
            for j in 0..n {
                let base = (i * n + j) * h;
                for r_idx in 0..h {
                    let m = m_edge[base + r_idx];
                    let dz = inv_sqrt_n * ghi[r_idx] * (1.0 - m * m);
                    if dz == 0.0 {
                        continue;
                    }
                    let w_d = &self.w_dir[r_idx * d..(r_idx + 1) * d];
                    let w_c = &self.w_comp[r_idx * d..(r_idx + 1) * d];
                    for c in 0..d {
                        grad_r.channels[c].add_assign(i, j, dz * w_d[c]);
                        grad_comp[c].add_assign(i, j, dz * w_c[c]);
                    }
                }
            }
        }

        // Adjoint exact de C_c = R_c * R_c : dR_c += G_c * R_c^T + R_c^T * G_c
        for c in 0..d {
            let term1 = grad_comp[c].matmul_t_right(&r.channels[c]);
            let term2 = r.channels[c].matmul_t_left(&grad_comp[c]);
            grad_r.channels[c].axpy(1.0, &term1);
            grad_r.channels[c].axpy(1.0, &term2);
        }

        let kl_uni = (ce_uni - effect.h_emp_uni).max(0.0);
        let kl_bi = if w_bi > 0.0 {
            (ce_bi - effect.h_emp_bi).max(0.0)
        } else {
            0.0
        };
        (w_uni * kl_uni + w_bi * kl_bi, grad_r)
    }
}

/// Sondes déterministes de stabilité différentielle \mathcal{T}_{inv} = \mathcal{T}_{dep} \cup \mathcal{T}_{abs}.
#[derive(Debug, Clone)]
pub struct StabilityProbes {
    pub rotations: Vec<Mat>,
    pub p_avg: Mat,
    pub abs_kappa: f64,
}

impl StabilityProbes {
    pub fn build(
        n: usize,
        num_rotations: usize,
        max_angle: f64,
        abs_kappa: f64,
        seed: u64,
    ) -> Self {
        let mut rng = Rng::from_seed(seed);
        let mut rotations = Vec::with_capacity(num_rotations);

        for _ in 0..num_rotations {
            let mut q = Mat::identity(n);
            let num_planes = n.max(2);
            for _ in 0..num_planes {
                if n < 2 {
                    break;
                }
                let p = rng.gen_range(n);
                let mut r = rng.gen_range(n - 1);
                if r >= p {
                    r += 1;
                }
                let theta = (rng.gen_f64() * 2.0 - 1.0) * max_angle;
                let c = theta.cos();
                let s = theta.sin();
                for col in 0..n {
                    let qp = q.get(p, col);
                    let qr = q.get(r, col);
                    q.set(p, col, c * qp - s * qr);
                    q.set(r, col, s * qp + c * qr);
                }
            }
            rotations.push(q);
        }

        let mu = 0.35;
        let off = mu / (n.max(1) as f64);
        let mut p_avg = Mat::zeros(n);
        for i in 0..n {
            for j in 0..n {
                p_avg.set(i, j, if i == j { (1.0 - mu) + off } else { off });
            }
        }

        Self {
            rotations,
            p_avg,
            abs_kappa,
        }
    }

    pub fn probe_count(&self) -> usize {
        self.rotations.len() + if self.abs_kappa > 0.0 { 1 } else { 0 }
    }

    pub fn apply_rotation(q: &Mat, r: &RelTensor) -> RelTensor {
        let mut out = RelTensor::zeros(r.n, r.d);
        for c in 0..r.d {
            out.channels[c] = q.matmul(&r.channels[c]).matmul_t_right(q);
        }
        out
    }

    pub fn adjoint_rotation(q: &Mat, grad_rot: &RelTensor) -> RelTensor {
        let mut out = RelTensor::zeros(grad_rot.n, grad_rot.d);
        for c in 0..grad_rot.d {
            out.channels[c] = q.matmul_t_left(&grad_rot.channels[c]).matmul(q);
        }
        out
    }

    pub fn apply_abstraction(&self, r: &RelTensor) -> RelTensor {
        let k = self.abs_kappa;
        let one_minus_k = 1.0 - k;
        let mut out = RelTensor::zeros(r.n, r.d);
        for c in 0..r.d {
            let prp = self.p_avg.matmul(&r.channels[c]).matmul(&self.p_avg);
            for idx in 0..(r.n * r.n) {
                out.channels[c].data[idx] =
                    one_minus_k * r.channels[c].data[idx] + k * prp.data[idx];
            }
        }
        out
    }

    pub fn adjoint_abstraction(&self, grad_abs: &RelTensor) -> RelTensor {
        self.apply_abstraction(grad_abs)
    }
}

/// Bilan complet d'une évaluation de \phi_\sharp(R, E).
#[derive(Debug, Clone)]
pub struct PhiSharpEval {
    /// Scalaire global \phi_\sharp(R, E) = \beta_iit * \lambda_2 - \alpha * \rho_KL - \beta * V_stab.
    pub phi_sharp: f64,
    /// Connectivité algébrique brute \lambda_2(L_{\bar{w}}).
    pub lambda2: f64,
    /// Résidu d'excès KL \rho_KL(R, E) >= 0.
    pub residual: f64,
    /// Plafond de bruit de fond \rho_0(E) >= 0.
    pub rho_background: f64,
    /// Fragilité différentielle V_stab(R, E).
    pub v_stab: f64,
    /// Matrice des poids causaux dirigés w_{ij}(R, E).
    pub causal_weights: Mat,
    pub active_causal_edges: usize,
    pub fiedler: FiedlerResult,
}

/// Évalue \phi_\sharp(R, E) et (si `compute_grad = true`) son gradient analytique EXACT \nabla_R \phi_\sharp.
pub fn evaluate_phi_sharp(
    params: &ModelParams,
    r: &RelTensor,
    effect: &EffectSummary,
    probes: &StabilityProbes,
    cfg: &PhiSharpConfig,
    compute_grad: bool,
) -> (PhiSharpEval, Option<RelTensor>) {
    let n = r.n;
    let d = r.d;
    let heads = params.precompute_heads(effect);
    let rho_bg = effect.rho_background(cfg.lambda_bi);

    // 1. Résidu d'excès KL \rho_KL(R, E) et son gradient \nabla_R \rho_KL
    let (rho, grad_rho_opt) = if compute_grad {
        let (val, g) = params.residual_and_grad_r_precomputed(r, effect, &heads, cfg.lambda_bi);
        (val, Some(g))
    } else {
        (
            params.residual_only_precomputed(r, effect, &heads, cfg.lambda_bi),
            None,
        )
    };

    let p_bar = (-rho).exp();

    // 2. Poids causaux dirigés par ablation exacte w_{ij}(R, E)
    let mut causal_weights = Mat::zeros(n);
    let mut abl_deriv_rho = Mat::zeros(n);
    let mut abl_deriv_rho_ij = Mat::zeros(n);
    let mut active_causal_edges = 0usize;

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if (0..d).all(|c| r.get(i, j, c) == 0.0) {
                continue;
            }

            let r_abl = r.with_ablated_edge(i, j);
            let rho_ij = params.residual_only_precomputed(&r_abl, effect, &heads, cfg.lambda_bi);
            let (s_ij, ds_drho_ij) = smooth_cap(rho_ij, rho_bg, cfg.cap_eps);
            let delta = s_ij - rho;
            let (h_val, h_prime) = smooth_hinge(delta, cfg.hinge_tau);

            if h_val > 0.0 {
                let wij = p_bar * h_val;
                causal_weights.set(i, j, wij);
                active_causal_edges += 1;

                if compute_grad {
                    abl_deriv_rho.set(i, j, -p_bar * (h_val + h_prime));
                    abl_deriv_rho_ij.set(i, j, p_bar * h_prime * ds_drho_ij);
                }
            }
        }
    }

    // 3. Connectivité algébrique \lambda_2(L_{\bar{w}})
    let lap = CausalLaplacian::from_directed_weights(&causal_weights);
    let fiedler = lap.compute_fiedler(cfg.max_krylov, cfg.max_inv_power);
    let lambda2 = fiedler.lambda2;

    // 4. Stabilité différentielle V_stab(R, E) et assemblage du gradient exact
    let total_probes = probes.probe_count();
    let mut v_stab = 0.0;
    let mut grad_r = if compute_grad {
        Some(RelTensor::zeros(n, d))
    } else {
        None
    };

    let mut coeff_on_grad_rho = -cfg.alpha;

    if compute_grad {
        let g_w = CausalLaplacian::grad_directed_weights(&fiedler.v2);
        let grad_acc = grad_r.as_mut().unwrap();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let gw = cfg.beta_iit * g_w.get(i, j);
                if gw <= 0.0 {
                    continue;
                }
                let dw_drho = abl_deriv_rho.get(i, j);
                let dw_drho_ij = abl_deriv_rho_ij.get(i, j);
                if dw_drho == 0.0 && dw_drho_ij == 0.0 {
                    continue;
                }

                coeff_on_grad_rho += gw * dw_drho;

                let mult = gw * dw_drho_ij;
                if mult.abs() > 0.0 {
                    let r_abl = r.with_ablated_edge(i, j);
                    let (_, mut g_abl) =
                        params.residual_and_grad_r_precomputed(&r_abl, effect, &heads, cfg.lambda_bi);
                    for c in 0..d {
                        g_abl.set(i, j, c, 0.0);
                    }
                    grad_acc.axpy(mult, &g_abl);
                }
            }
        }
    }

    if total_probes > 0 && cfg.beta > 0.0 {
        let inv_m = 1.0 / (total_probes as f64);

        for q in &probes.rotations {
            let r_rot = StabilityProbes::apply_rotation(q, r);
            let rho_rot =
                params.residual_only_precomputed(&r_rot, effect, &heads, cfg.lambda_bi);
            let (hv, hp) = smooth_hinge(rho_rot - rho, cfg.hinge_tau);
            v_stab += inv_m * hv;

            if compute_grad && hp > 0.0 {
                let scale = -cfg.beta * inv_m * hp;
                coeff_on_grad_rho -= scale;
                let (_, g_rot) = params.residual_and_grad_r_precomputed(
                    &r_rot,
                    effect,
                    &heads,
                    cfg.lambda_bi,
                );
                let g_back = StabilityProbes::adjoint_rotation(q, &g_rot);
                grad_r.as_mut().unwrap().axpy(scale, &g_back);
            }
        }

        if probes.abs_kappa > 0.0 {
            let r_abs = probes.apply_abstraction(r);
            let rho_abs =
                params.residual_only_precomputed(&r_abs, effect, &heads, cfg.lambda_bi);
            let (hv, hp) = smooth_hinge(rho_abs - rho, cfg.hinge_tau);
            v_stab += inv_m * hv;

            if compute_grad && hp > 0.0 {
                let scale = -cfg.beta * inv_m * hp;
                coeff_on_grad_rho -= scale;
                let (_, g_abs) = params.residual_and_grad_r_precomputed(
                    &r_abs,
                    effect,
                    &heads,
                    cfg.lambda_bi,
                );
                let g_back = probes.adjoint_abstraction(&g_abs);
                grad_r.as_mut().unwrap().axpy(scale, &g_back);
            }
        }
    }

    if let (Some(g_acc), Some(g_rho)) = (grad_r.as_mut(), grad_rho_opt.as_ref()) {
        g_acc.axpy(coeff_on_grad_rho, g_rho);
    }

    let phi_sharp = cfg.beta_iit * lambda2 - cfg.alpha * rho - cfg.beta * v_stab;

    let eval = PhiSharpEval {
        phi_sharp,
        lambda2,
        residual: rho,
        rho_background: rho_bg,
        v_stab,
        causal_weights,
        active_causal_edges,
        fiedler,
    };
    (eval, grad_r)
}

#[inline]
pub fn residual_weights(has_bigrams: bool, lambda_bi: f64) -> (f64, f64) {
    if !has_bigrams || lambda_bi <= 0.0 {
        (1.0, 0.0)
    } else {
        let denom = 1.0 + lambda_bi;
        (1.0 / denom, lambda_bi / denom)
    }
}

/// Charnière quadratique C^1 : 0 si x <= 0, x^2/(2\tau) sur (0, \tau), x - \tau/2 si x >= \tau.
#[inline]
pub fn smooth_hinge(x: f64, tau: f64) -> (f64, f64) {
    if x <= 0.0 {
        (0.0, 0.0)
    } else if x < tau {
        (0.5 * x * x / tau, x / tau)
    } else {
        (x - 0.5 * tau, 1.0)
    }
}

/// Plafond lisse C^1 sans biais sous le seuil b :
/// - pour a <= b : vaut exactement `a` (dérivée = 1, zéro zone morte sous \rho_0(E))
/// - pour a >= b + eps : plafonné à `b + 0.5*eps` (dérivée = 0)
#[inline]
pub fn smooth_cap(a: f64, b: f64, eps: f64) -> (f64, f64) {
    let (h, hp) = smooth_hinge(a - b, eps);
    (a - h, 1.0 - hp)
}

fn bilinear_cross_entropy(
    h_node: &[f64],
    u_mat: &[f64],
    u0: &[f64],
    target: &[f64],
    mass: f64,
    logits: &mut [f64],
) -> f64 {
    let k = u0.len();
    let h = h_node.len();
    let mut max_l = f64::NEG_INFINITY;
    for a in 0..k {
        let row = &u_mat[a * h..(a + 1) * h];
        let mut acc = u0[a];
        for r in 0..h {
            acc += row[r] * h_node[r];
        }
        logits[a] = acc;
        if acc > max_l {
            max_l = acc;
        }
    }
    let mut sum_exp = 0.0;
    for &l in logits.iter().take(k) {
        sum_exp += (l - max_l).exp();
    }
    let lse = max_l + sum_exp.ln();
    let mut dot_qt = 0.0;
    for a in 0..k {
        if target[a] > 0.0 {
            dot_qt += target[a] * logits[a];
        }
    }
    mass * lse - dot_qt
}

#[allow(clippy::too_many_arguments)]
fn bilinear_cross_entropy_and_grad_h(
    h_node: &[f64],
    u_mat: &[f64],
    u0: &[f64],
    target: &[f64],
    mass: f64,
    loss_weight: f64,
    probs: &mut [f64],
    grad_h_out: &mut [f64],
) -> f64 {
    let k = u0.len();
    let h = h_node.len();
    let mut max_l = f64::NEG_INFINITY;
    for a in 0..k {
        let row = &u_mat[a * h..(a + 1) * h];
        let mut acc = u0[a];
        for r in 0..h {
            acc += row[r] * h_node[r];
        }
        probs[a] = acc;
        if acc > max_l {
            max_l = acc;
        }
    }
    let mut sum_exp = 0.0;
    let mut dot_qt = 0.0;
    for a in 0..k {
        let l = probs[a];
        if target[a] > 0.0 {
            dot_qt += target[a] * l;
        }
        let ex = (l - max_l).exp();
        probs[a] = ex;
        sum_exp += ex;
    }
    let lse = max_l + sum_exp.ln();
    let inv_sum = 1.0 / sum_exp;

    for a in 0..k {
        let p_a = probs[a] * inv_sum;
        let dl = loss_weight * (mass * p_a - target[a]);
        if dl == 0.0 {
            continue;
        }
        let row = &u_mat[a * h..(a + 1) * h];
        for r in 0..h {
            grad_h_out[r] += dl * row[r];
        }
    }

    mass * lse - dot_qt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ho_ptgrs::generator::{CurriculumConfig, HoPtgrsGenerator};
    use crate::ho_ptgrs::meta_ops::MetaOp;

    #[test]
    fn test_exact_s256_permutation_invariance() {
        // Génère un vrai fichier `perceived.eff` HO-PTGRS, puis applique une permutation aléatoire
        // bijective \pi \in S_256 sur tous les octets : R(0), \rho_KL, \lambda_2, V_stab, \phi_\sharp
        // et le tenseur complet \nabla_R \phi_\sharp doivent être IDENTIQUES à 1e-11 près.
        let mut gen_ = HoPtgrsGenerator::new(2026, CurriculumConfig::default());
        let ep = gen_.generate_episode(Some(MetaOp::Bif)).unwrap();

        let mut perm_rng = Rng::from_seed(99991);
        let mut perm: Vec<u8> = (0u8..=255).collect();
        perm_rng.shuffle(&mut perm);
        let permuted_eff: Vec<u8> = ep.perceived_eff.iter().map(|&b| perm[b as usize]).collect();

        let n = 6;
        let d = 2;
        let h = 8;
        let mut p_rng = Rng::from_seed(1234);
        let mut params = ModelParams::random_init(n, d, h, &mut p_rng);
        for v in &mut params.w_rel_uni {
            *v *= 2.0;
        }
        for v in &mut params.w_rel_bi {
            *v *= 2.0;
        }

        let eff1 = EffectSummary::from_bytes(&ep.perceived_eff, n);
        let eff2 = EffectSummary::from_bytes(&permuted_eff, n);
        assert_eq!(eff1.k_uni, eff2.k_uni);
        assert_eq!(eff1.k_bi, eff2.k_bi);
        assert!((eff1.h_emp_uni - eff2.h_emp_uni).abs() < 1e-12);
        assert!((eff1.h_emp_bi - eff2.h_emp_bi).abs() < 1e-12);
        assert!((eff1.rho_background(0.5) - eff2.rho_background(0.5)).abs() < 1e-12);

        let r0_1 = params.encode_regional_pole(&eff1, 4.0);
        let r0_2 = params.encode_regional_pole(&eff2, 4.0);
        let mut diff_r0 = r0_1.clone();
        diff_r0.axpy(-1.0, &r0_2);
        assert!(
            diff_r0.frob_norm() < 1e-12,
            "R(0) n'est pas S_256-invariant : ||diff|| = {}",
            diff_r0.frob_norm()
        );

        let probes = StabilityProbes::build(n, 2, 0.06, 0.12, 4242);
        let cfg = PhiSharpConfig::default();

        // Quelques pas sur le résidu pour activer \lambda_2 > 0
        let mut r1 = r0_1;
        let mut r2 = r0_2;
        for _ in 0..8 {
            let (_, g1) = params.residual_and_grad_r(&r1, &eff1, cfg.lambda_bi);
            let (_, g2) = params.residual_and_grad_r(&r2, &eff2, cfg.lambda_bi);
            r1.axpy(-1.0, &g1);
            r2.axpy(-1.0, &g2);
        }

        let (e1, g1_opt) = evaluate_phi_sharp(&params, &r1, &eff1, &probes, &cfg, true);
        let (e2, g2_opt) = evaluate_phi_sharp(&params, &r2, &eff2, &probes, &cfg, true);

        assert!((e1.residual - e2.residual).abs() < 1e-11);
        assert!((e1.lambda2 - e2.lambda2).abs() < 1e-11);
        assert!((e1.v_stab - e2.v_stab).abs() < 1e-11);
        assert!((e1.phi_sharp - e2.phi_sharp).abs() < 1e-11);

        let mut diff_g = g1_opt.unwrap();
        diff_g.axpy(-1.0, &g2_opt.unwrap());
        assert!(
            diff_g.frob_norm() < 1e-10,
            "grad_R phi_sharp n'est pas S_256-invariant : ||diff|| = {}",
            diff_g.frob_norm()
        );
    }

    #[test]
    fn test_theorem_2_diagonal_memorization_has_zero_lambda2() {
        let mut rng = Rng::from_seed(77);
        let n = 6;
        let d = 2;
        let h = 6;
        let params = ModelParams::random_init(n, d, h, &mut rng);
        let bytes: Vec<u8> = (0..64).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        let eff = EffectSummary::from_bytes(&bytes, n);
        let probes = StabilityProbes::build(n, 2, 0.05, 0.1, 99);
        let cfg = PhiSharpConfig::default();

        let mut r_diag = RelTensor::zeros(n, d);
        for c in 0..d {
            for i in 0..n {
                r_diag.set(i, i, c, 0.8);
            }
        }

        let (eval, _) = evaluate_phi_sharp(&params, &r_diag, &eff, &probes, &cfg, false);
        assert_eq!(eval.active_causal_edges, 0);
        assert_eq!(eval.lambda2, 0.0);
        assert!(eval.phi_sharp <= 0.0);
    }

    #[test]
    fn test_residual_gradient_exact_finite_diff() {
        let mut rng = Rng::from_seed(101);
        let n = 4;
        let d = 2;
        let h = 4;
        let params = ModelParams::random_init(n, d, h, &mut rng);
        let bytes: Vec<u8> = (0..48).map(|i| ((i * 53 + 7) % 19) as u8).collect();
        let eff = EffectSummary::from_bytes(&bytes, n);

        let mut r = RelTensor::zeros(n, d);
        for c in 0..d {
            for i in 0..n {
                for j in 0..n {
                    r.set(i, j, c, (rng.gen_f64() * 2.0 - 1.0) * 0.6);
                }
            }
        }

        let (_, analytic) = params.residual_and_grad_r(&r, &eff, 0.5);
        let eps = 1e-6;
        for c in 0..d {
            for i in 0..n {
                for j in 0..n {
                    let mut rp = r.clone();
                    let mut rm = r.clone();
                    rp.set(i, j, c, r.get(i, j, c) + eps);
                    rm.set(i, j, c, r.get(i, j, c) - eps);
                    let vp = params.residual_only(&rp, &eff, 0.5);
                    let vm = params.residual_only(&rm, &eff, 0.5);
                    let fd = (vp - vm) / (2.0 * eps);
                    let an = analytic.get(i, j, c);
                    assert!(
                        (fd - an).abs() < 1e-6,
                        "Residual grad mismatch ({i},{j},{c}): fd={fd}, an={an}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_full_phi_sharp_gradient_exact_finite_diff() {
        let mut rng = Rng::from_seed(2026);
        let n = 5;
        let d = 2;
        let h = 8;
        let mut params = ModelParams::random_init(n, d, h, &mut rng);
        for v in &mut params.w_rel_uni {
            *v *= 2.5;
        }
        for v in &mut params.w_rel_bi {
            *v *= 2.5;
        }

        // Deux régimes structurels distincts pour obtenir un gap spectral net sur \lambda_2
        let mut bytes = Vec::with_capacity(80);
        for _ in 0..8 {
            bytes.extend_from_slice(&[12u8, 45, 78, 12, 45]);
        }
        for _ in 0..8 {
            bytes.extend_from_slice(&[190u8, 210, 230, 210, 230]);
        }
        let eff = EffectSummary::from_bytes(&bytes, n);
        let probes = StabilityProbes::build(n, 2, 0.08, 0.15, 31415);
        let cfg = PhiSharpConfig {
            beta_iit: 10.0,
            alpha: 0.4,
            beta: 0.3,
            lambda_bi: 0.5,
            hinge_tau: 1e-4,
            cap_eps: 1e-3,
            max_krylov: 16,
            max_inv_power: 6,
        };

        let mut r = params.encode_regional_pole(&eff, 4.0);
        for _ in 0..20 {
            let (_, g_rho) = params.residual_and_grad_r(&r, &eff, cfg.lambda_bi);
            r.axpy(-1.2, &g_rho);
            r.clamp_to_ball(4.0);
        }
        for c in 0..d {
            for i in 0..n {
                for j in 0..n {
                    if (i < 2) != (j < 2) {
                        r.set(i, j, c, r.get(i, j, c) * 0.45);
                    }
                }
            }
        }
        for _ in 0..3 {
            let (_, g_rho) = params.residual_and_grad_r(&r, &eff, cfg.lambda_bi);
            r.axpy(-0.4, &g_rho);
            r.clamp_to_ball(4.0);
        }

        let (eval, grad_opt) = evaluate_phi_sharp(&params, &r, &eff, &probes, &cfg, true);
        let analytic = grad_opt.unwrap();
        assert!(eval.active_causal_edges > 0);
        assert!(
            eval.lambda2 > 1e-4 && (eval.fiedler.lambda3 - eval.lambda2).abs() > 1e-4,
            "lambda2={} doit être simple et franchement > 0 (lambda3={})",
            eval.lambda2,
            eval.fiedler.lambda3
        );

        let eps = 1e-6;
        for c in 0..d {
            for i in 0..n {
                for j in 0..n {
                    let mut rp = r.clone();
                    let mut rm = r.clone();
                    rp.set(i, j, c, r.get(i, j, c) + eps);
                    rm.set(i, j, c, r.get(i, j, c) - eps);
                    let (ep, _) = evaluate_phi_sharp(&params, &rp, &eff, &probes, &cfg, false);
                    let (em, _) = evaluate_phi_sharp(&params, &rm, &eff, &probes, &cfg, false);
                    let fd = (ep.phi_sharp - em.phi_sharp) / (2.0 * eps);
                    let an = analytic.get(i, j, c);
                    assert!(
                        (fd - an).abs() < 1e-4,
                        "Phi_sharp grad mismatch ({i},{j},{c}): fd={fd}, an={an}"
                    );
                }
            }
        }
    }
}
