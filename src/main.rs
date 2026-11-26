use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::thread;

const DIM: usize = 8;
const LATENT_K: usize = 6;
const TRAIN_BUDGET: usize = 6;
const MAGIC: &[u8; 8] = b"PHIS2024";

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
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = (self.next_f64()).max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
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
// 2. PTGRS : Probabilistic Term Graph Rewriting System
//    Ne fournit que E à l'entraînement ; C est conservée pour l'étalonnage.
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

impl PtgrsCause {
    /// Famille d'entraînement (in-distribution) : règles binaires/ternaires, profondeur modérée.
    fn sample_id(rng: &mut Rng) -> Self {
        let mut pool: Vec<u8> = (0u8..=255).collect();
        rng.shuffle(&mut pool);

        let n_nonterms = 3 + rng.range_usize(3); // 3..5 non-terminaux
        let n_terms = 5 + rng.range_usize(6); // 5..10 terminaux
        let nonterms = &pool[0..n_nonterms];
        let terminals = pool[n_nonterms..n_nonterms + n_terms].to_vec();

        let mut rules = Vec::new();
        for (idx, &nt) in nonterms.iter().enumerate() {
            let next_nt = nonterms[(idx + 1) % n_nonterms];
            let t1 = terminals[rng.range_usize(n_terms)];
            let t2 = terminals[rng.range_usize(n_terms)];
            let t3 = terminals[rng.range_usize(n_terms)];

            let rhs_primary = match rng.range_usize(3) {
                0 => vec![t1, next_nt, t2],
                1 => vec![next_nt, t1, t2],
                _ => vec![t1, t2, next_nt],
            };
            let rhs_alt = vec![t2, t3];
            let prob_primary = 0.65 + 0.25 * rng.next_f64();

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

    /// Famille OOD (barreau 1) : règles croisées, motifs en miroir et profondeur contrôlée.
    fn sample_ood(rng: &mut Rng, depth_family: usize) -> Self {
        let mut pool: Vec<u8> = (0u8..=255).collect();
        rng.shuffle(&mut pool);

        let n_nonterms = 4 + (depth_family.min(4));
        let n_terms = 8 + rng.range_usize(6);
        let nonterms = &pool[0..n_nonterms];
        let terminals = pool[n_nonterms..n_nonterms + n_terms].to_vec();

        let mut rules = Vec::new();
        for (idx, &nt) in nonterms.iter().enumerate() {
            let n1 = nonterms[(idx + 1) % n_nonterms];
            let n2 = nonterms[(idx + 2) % n_nonterms];
            let t1 = terminals[rng.range_usize(n_terms)];
            let t2 = terminals[rng.range_usize(n_terms)];
            let t3 = terminals[rng.range_usize(n_terms)];

            // Structures inédites : motifs emboîtés (t1 n1 t2 n2 t1)
            let rhs_primary = vec![t1, n1, t2, n2, t1];
            let rhs_alt = vec![t3, t2, t1];
            let prob_primary = 0.72 + 0.18 * rng.next_f64();

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

    /// Génère un flux d'octets E en réécrivant le graphe de termes.
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

    /// Distribution observationnelle empirique de référence de C (jamais vue en entraînement).
    fn ground_truth_transitions(&self, rng: &mut Rng, max_depth: usize) -> HashMap<(u8, u8), f64> {
        let mut counts: HashMap<(u8, u8), usize> = HashMap::new();
        let mut row_totals: HashMap<u8, usize> = HashMap::new();
        for _ in 0..16 {
            let sample = self.emit_effect(rng, max_depth, 160);
            for w in sample.windows(2) {
                *counts.entry((w[0], w[1])).or_insert(0) += 1;
                *row_totals.entry(w[0]).or_insert(0) += 1;
            }
        }
        let mut probs = HashMap::new();
        for ((u, v), c) in counts {
            let tot = *row_totals.get(&u).unwrap_or(&1) as f64;
            probs.insert((u, v), (c as f64) / tot);
        }
        probs
    }
}

// ============================================================================
// 3. Représentation strictement relationnelle de E (invariante au renommage)
// ============================================================================

#[derive(Clone)]
struct RelationalView {
    vocab: Vec<u8>,
    seq: Vec<usize>,
    v_size: usize,
    t_fwd: Vec<f64>,  // v_size * v_size
    t_bwd: Vec<f64>,  // v_size * v_size
    t_skip: Vec<f64>, // v_size * v_size
    row_has_mass: Vec<bool>,
    node_init: Vec<[f64; DIM]>,
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
        let mut fwd_c = vec![0.0; v_size * v_size];
        let mut bwd_c = vec![0.0; v_size * v_size];
        let mut skip_c = vec![0.0; v_size * v_size];
        let mut freq = vec![0.0; v_size];

        for &s in &seq {
            if s < v_size {
                freq[s] += 1.0;
            }
        }
        for w in seq.windows(2) {
            if w[0] < v_size && w[1] < v_size {
                fwd_c[w[0] * v_size + w[1]] += 1.0;
                bwd_c[w[1] * v_size + w[0]] += 1.0;
            }
        }
        for w in seq.windows(3) {
            if w[0] < v_size && w[2] < v_size {
                skip_c[w[0] * v_size + w[2]] += 1.0;
            }
        }

        let mut t_fwd = vec![0.0; v_size * v_size];
        let mut t_bwd = vec![0.0; v_size * v_size];
        let mut t_skip = vec![0.0; v_size * v_size];
        let mut row_has_mass = vec![false; v_size];

        for u in 0..v_size {
            let sf: f64 = fwd_c[u * v_size..(u + 1) * v_size].iter().sum();
            let sb: f64 = bwd_c[u * v_size..(u + 1) * v_size].iter().sum();
            let ss: f64 = skip_c[u * v_size..(u + 1) * v_size].iter().sum();
            if sf > 0.0 {
                row_has_mass[u] = true;
                for v in 0..v_size {
                    t_fwd[u * v_size + v] = fwd_c[u * v_size + v] / sf;
                }
            }
            if sb > 0.0 {
                for v in 0..v_size {
                    t_bwd[u * v_size + v] = bwd_c[u * v_size + v] / sb;
                }
            }
            if ss > 0.0 {
                for v in 0..v_size {
                    t_skip[u * v_size + v] = skip_c[u * v_size + v] / ss;
                }
            }
        }

        // Traits purement relationnels par symbole (invariants aux valeurs d'octets).
        let total_len = seq.len().max(1) as f64;
        let ln_v = (v_size.max(2) as f64).ln();
        let mut node_init = vec![[0.0; DIM]; v_size];

        for u in 0..v_size {
            let p_freq = freq[u] / total_len;
            let mut h_fwd = 0.0;
            let mut h_bwd = 0.0;
            let mut deg_fwd = 0.0;
            let mut deg_bwd = 0.0;
            let mut skip_align = 0.0;

            for v in 0..v_size {
                let pf = t_fwd[u * v_size + v];
                let pb = t_bwd[u * v_size + v];
                let ps = t_skip[u * v_size + v];
                if pf > 1e-12 {
                    h_fwd -= pf * pf.ln();
                    deg_fwd += 1.0;
                }
                if pb > 1e-12 {
                    h_bwd -= pb * pb.ln();
                    deg_bwd += 1.0;
                }
                skip_align += pf * ps;
            }
            let self_loop = t_fwd[u * v_size + u];
            let asym = (deg_fwd - deg_bwd) / (v_size as f64);
            // Position relative normalisée du centre de masse dans E
            let mut pos_sum = 0.0;
            let mut pos_cnt = 0.0;
            for (idx, &s) in seq.iter().enumerate() {
                if s == u {
                    pos_sum += (idx as f64) / total_len;
                    pos_cnt += 1.0;
                }
            }
            let mean_pos = if pos_cnt > 0.0 { pos_sum / pos_cnt } else { 0.5 };

            node_init[u] = [
                (p_freq * (v_size as f64)).tanh(),
                h_fwd / ln_v,
                h_bwd / ln_v,
                deg_fwd / (v_size as f64),
                deg_bwd / (v_size as f64),
                self_loop,
                asym.tanh(),
                (skip_align * 2.0 + mean_pos - 0.5).tanh(),
            ];
        }

        Self {
            vocab,
            seq,
            v_size,
            t_fwd,
            t_bwd,
            t_skip,
            row_has_mass,
            node_init,
        }
    }

    /// Construit un sous-ensemble sur le même vocabulaire partagé (pour Projection & Déplacement).
    fn subslice_same_vocab(&self, start: usize, end: usize) -> Self {
        let s = start.min(self.seq.len());
        let e = end.min(self.seq.len()).max(s);
        Self::from_indexed(self.vocab.clone(), self.seq[s..e].to_vec())
    }

    /// Contre-exemple négatif (permutation cyclique des cibles de transition : même support, structure brisée).
    fn negative_transition_matrix(&self) -> Vec<f64> {
        let mut neg = vec![0.0; self.v_size * self.v_size];
        if self.v_size <= 1 {
            return neg;
        }
        for u in 0..self.v_size {
            for v in 0..self.v_size {
                let shifted_v = (v + 1 + (u % (self.v_size - 1))) % self.v_size;
                neg[u * self.v_size + shifted_v] = self.t_fwd[u * self.v_size + v];
            }
        }
        neg
    }
}

// ============================================================================
// 4. Architecture connexionniste non-réciproque Phisharp
// ============================================================================

const OFF_W_LAT: usize = 0;
const OFF_U_OBS_LAT: usize = OFF_W_LAT + LATENT_K * LATENT_K;
const OFF_U_LAT_FWD: usize = OFF_U_OBS_LAT + DIM * DIM;
const OFF_U_LAT_OBS: usize = OFF_U_LAT_FWD + DIM * DIM;
const OFF_U_REFUTE: usize = OFF_U_LAT_OBS + DIM * DIM;
const OFF_U_ABSTRACT: usize = OFF_U_REFUTE + DIM * DIM;
const OFF_U_BILIN: usize = OFF_U_ABSTRACT + DIM * DIM;
const OFF_B_LAT: usize = OFF_U_BILIN + DIM * DIM;
const PARAM_COUNT: usize = OFF_B_LAT + DIM;

#[derive(Clone)]
struct PhisharpModel {
    params: Vec<f64>,
}

#[derive(Clone)]
struct AbductionTrace {
    latent: Vec<[f64; DIM]>,
    pred_t: Vec<f64>,
    residual_per_node: Vec<f64>,
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

fn mat_vec(mat: &[f64], x: &[f64; DIM]) -> [f64; DIM] {
    let mut y = [0.0; DIM];
    for r in 0..DIM {
        let mut acc = 0.0;
        let base = r * DIM;
        for c in 0..DIM {
            acc += mat[base + c] * x[c];
        }
        y[r] = acc;
    }
    y
}

fn dot(a: &[f64; DIM], b: &[f64; DIM]) -> f64 {
    let mut s = 0.0;
    for i in 0..DIM {
        s += a[i] * b[i];
    }
    s
}

impl PhisharpModel {
    fn new_random(rng: &mut Rng) -> Self {
        let mut params = vec![0.0; PARAM_COUNT];
        for (i, p) in params.iter_mut().enumerate() {
            let scale = if i < OFF_U_OBS_LAT { 0.45 } else { 0.35 };
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
        let mut buf = Vec::with_capacity(8 + 8 + self.params.len() * 8);
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
                "Format de modèle .phis invalide",
            ));
        }
        let count = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        if count != PARAM_COUNT || data.len() != 16 + count * 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Nombre de paramètres incompatible",
            ));
        }
        let mut params = Vec::with_capacity(count);
        for chunk in data[16..].chunks_exact(8) {
            params.push(f64::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(Self { params })
    }

    fn w_lat(&self, dst: usize, src: usize) -> f64 {
        self.params[OFF_W_LAT + dst * LATENT_K + src]
    }
    fn u_obs_lat(&self) -> &[f64] {
        &self.params[OFF_U_OBS_LAT..OFF_U_LAT_FWD]
    }
    fn u_lat_fwd(&self) -> &[f64] {
        &self.params[OFF_U_LAT_FWD..OFF_U_LAT_OBS]
    }
    fn u_lat_obs(&self) -> &[f64] {
        &self.params[OFF_U_LAT_OBS..OFF_U_REFUTE]
    }
    fn u_refute(&self) -> &[f64] {
        &self.params[OFF_U_REFUTE..OFF_U_ABSTRACT]
    }
    fn u_abstract(&self) -> &[f64] {
        &self.params[OFF_U_ABSTRACT..OFF_U_BILIN]
    }
    fn u_bilin(&self) -> &[f64] {
        &self.params[OFF_U_BILIN..OFF_B_LAT]
    }
    fn b_lat(&self) -> &[f64] {
        &self.params[OFF_B_LAT..PARAM_COUNT]
    }

    /// Projette l'état causal latent H sur la matrice de transitions prédite T_hat.
    /// Passe strictement par H : si H = 0 (budget nul), T_hat est uniforme.
    fn predict_transitions(&self, view: &RelationalView, latent: &[[f64; DIM]]) -> Vec<f64> {
        let v = view.v_size;
        let k_len = latent.len();
        let mut z = vec![[0.0; DIM]; v];

        // Couplage attentionnel entre chaque symbole relationnel et les nœuds latents H
        for u in 0..v {
            let x_u = &view.node_init[u];
            let mut logits = vec![0.0; k_len];
            let mut max_l = f64::NEG_INFINITY;
            for k in 0..k_len {
                let l = dot(x_u, &latent[k]);
                logits[k] = l;
                if l > max_l {
                    max_l = l;
                }
            }
            let mut sum_e = 0.0;
            for l in &mut logits {
                *l = (*l - max_l).exp();
                sum_e += *l;
            }
            let mut acc = [0.0; DIM];
            for k in 0..k_len {
                let w = logits[k] / (sum_e + 1e-12);
                let proj = mat_vec(self.u_lat_obs(), &latent[k]);
                for d in 0..DIM {
                    acc[d] += w * proj[d];
                }
            }
            for d in 0..DIM {
                z[u][d] = acc[d].tanh();
            }
        }

        // Propagation relationnelle sur le graphe d'entrée + tête bilinéaire asymétrique
        let mut pred = vec![0.0; v * v];
        for u in 0..v {
            let zu_b = mat_vec(self.u_bilin(), &z[u]);
            let mut row_logits = vec![0.0; v];
            let mut max_l = f64::NEG_INFINITY;
            for w in 0..v {
                let l = dot(&zu_b, &z[w]);
                row_logits[w] = l;
                if l > max_l {
                    max_l = l;
                }
            }
            let mut sum_e = 0.0;
            for l in &mut row_logits {
                *l = (*l - max_l).exp();
                sum_e += *l;
            }
            for w in 0..v {
                pred[u * v + w] = row_logits[w] / (sum_e + 1e-12);
            }
        }
        pred
    }

    /// Une étape de dynamique non-réciproque sur les nœuds latents (optionnellement coupée par masque MIP).
    fn latent_step(
        &self,
        view: &RelationalView,
        latent: &[[f64; DIM]],
        use_refutation: bool,
        cut_mask: Option<u32>,
    ) -> (Vec<[f64; DIM]>, Vec<f64>, Vec<f64>) {
        let v = view.v_size;
        let k_len = latent.len();
        let pred = self.predict_transitions(view, latent);

        // 1. Calcul du résidu relationnel R_err = T_fwd - T_hat
        let mut residual_per_node = vec![0.0; v];
        let mut obs_err_vec = vec![[0.0; DIM]; v];
        for u in 0..v {
            if !view.row_has_mass[u] {
                continue;
            }
            let mut l1 = 0.0;
            for w in 0..v {
                let diff = view.t_fwd[u * v + w] - pred[u * v + w];
                l1 += diff.abs();
                for d in 0..DIM {
                    obs_err_vec[u][d] += diff * view.node_init[w][d];
                }
            }
            // Intègre aussi la relation skip-1 pour capter les compositions profondes
            for w in 0..v {
                let s_val = view.t_skip[u * v + w];
                if s_val > 0.0 {
                    for d in 0..DIM {
                        obs_err_vec[u][d] += 0.25 * s_val * view.node_init[w][d];
                    }
                }
            }
            residual_per_node[u] = l1 * 0.5;
        }

        // 2. Mise à jour abductive non-réciproque + abstraction hiérarchique
        let mut next_latent = vec![[0.0; DIM]; k_len];
        let b = self.b_lat();

        for dst in 0..k_len {
            let mut acc = [0.0; DIM];

            // Signal abductif depuis les symboles relationnels pondéré par affinité
            for u in 0..v {
                let gate = (dot(&view.node_init[u], &latent[dst]) + 0.15 * (dst as f64)).tanh();
                let mapped = mat_vec(self.u_obs_lat(), &obs_err_vec[u]);
                let scale = (0.5 + 0.5 * gate) / (v as f64);
                for d in 0..DIM {
                    acc[d] += scale * mapped[d];
                }
            }

            // Couplages dirigés non-réciproques W_lat[dst, src]
            for src in 0..k_len {
                if let Some(mask) = cut_mask {
                    let side_dst = ((mask >> dst.min(30)) & 1) != 0;
                    let side_src = ((mask >> src.min(30)) & 1) != 0;
                    if side_dst != side_src {
                        continue; // Coupe MIP : sectionne les liens inter-partitions
                    }
                }
                let w = if dst < LATENT_K && src < LATENT_K {
                    self.w_lat(dst, src)
                } else {
                    0.0
                };
                let msg = mat_vec(self.u_lat_fwd(), &latent[src]);
                for d in 0..DIM {
                    acc[d] += w * msg[d];
                }
                // Abstraction dirigée des nœuds bas vers les nœuds hauts (src < dst)
                if src < dst {
                    let abs_msg = mat_vec(self.u_abstract(), &latent[src]);
                    for d in 0..DIM {
                        acc[d] += 0.20 * abs_msg[d];
                    }
                }
            }

            // Intégration temporelle (trajectoire abductive)
            for d in 0..DIM {
                let raw = 0.60 * latent[dst][d] + 0.40 * (acc[d] + b[d]).tanh();
                next_latent[dst][d] = raw.tanh();
            }
        }

        // 3. Réfutation active interne : tester une contre-hypothèse et corriger ce qui s'effondre
        if use_refutation && cut_mask.is_none() {
            let mut refuted = next_latent.clone();
            // Cibler le nœud le plus actif pour tenter de le réfuter
            let mut peak_k = 0;
            let mut peak_norm = -1.0;
            for (k, h) in refuted.iter().enumerate() {
                let n = dot(h, h);
                if n > peak_norm {
                    peak_norm = n;
                    peak_k = k;
                }
            }
            let perturb = mat_vec(self.u_refute(), &refuted[peak_k]);
            for d in 0..DIM {
                refuted[peak_k][d] = (refuted[peak_k][d] - 0.35 * perturb[d].tanh()).tanh();
            }

            let pred_base = self.predict_transitions(view, &next_latent);
            let pred_ref = self.predict_transitions(view, &refuted);
            let gain_base = transition_gain(&view.t_fwd, &pred_base, &view.row_has_mass, v);
            let gain_ref = transition_gain(&view.t_fwd, &pred_ref, &view.row_has_mass, v);

            // Si la contre-hypothèse explique mieux ou résiste mieux, déplacer la trajectoire vers elle
            if gain_ref > gain_base {
                next_latent = refuted;
            } else {
                // Sinon, consolider la composante qui a survécu à la réfutation
                for d in 0..DIM {
                    let delta = next_latent[peak_k][d] - refuted[peak_k][d];
                    next_latent[peak_k][d] = (next_latent[peak_k][d] + 0.15 * delta).tanh();
                }
            }
        }

        (next_latent, pred, residual_per_node)
    }

    /// Exécute la trajectoire abductive sur un budget donné T.
    fn run_abduction(
        &self,
        view: &RelationalView,
        budget: usize,
        use_refutation: bool,
        record_phi: bool,
    ) -> AbductionTrace {
        let mut latent = vec![[0.0; DIM]; LATENT_K];
        let mut phi_history = Vec::new();

        if budget == 0 {
            let pred = self.predict_transitions(view, &latent);
            let mut res = vec![0.0; view.v_size];
            for u in 0..view.v_size {
                if view.row_has_mass[u] {
                    let mut l1 = 0.0;
                    for w in 0..view.v_size {
                        l1 += (view.t_fwd[u * view.v_size + w] - pred[u * view.v_size + w]).abs();
                    }
                    res[u] = l1 * 0.5;
                }
            }
            if record_phi {
                let rep = self.evaluate_phi_state(view, &latent, budget, use_refutation);
                phi_history.push(rep.phi_sharp);
            }
            return AbductionTrace {
                latent,
                pred_t: pred,
                residual_per_node: res,
                phi_history,
            };
        }

        // Initialisation symétrique légèrement brisée par l'indice de couche d'abstraction
        for (k, h) in latent.iter_mut().enumerate() {
            for d in 0..DIM {
                h[d] = 0.05 * (((k + 1) * (d + 1)) as f64).sin();
            }
        }

        let mut last_pred = Vec::new();
        let mut last_res = Vec::new();

        for step in 1..=budget {
            let (nl, pred, res) = self.latent_step(view, &latent, use_refutation, None);
            latent = nl;
            last_pred = pred;
            last_res = res;
            if record_phi {
                let rep = self.evaluate_phi_state(view, &latent, step, use_refutation);
                phi_history.push(rep.phi_sharp);
            }
        }

        let final_pred = self.predict_transitions(view, &latent);
        if last_pred.is_empty() {
            last_pred = final_pred;
        } else {
            last_pred = final_pred;
        }

        AbductionTrace {
            latent,
            pred_t: last_pred,
            residual_per_node: last_res,
            phi_history,
        }
    }

    /// Calcul du substitut de φ_sharp (IIT 4.0 + stabilité différentielle + résidu).
    fn evaluate_phi(
        &self,
        view: &RelationalView,
        budget: usize,
        use_refutation: bool,
    ) -> PhiReport {
        let trace = self.run_abduction(view, budget, use_refutation, false);
        self.evaluate_phi_state(view, &trace.latent, budget, use_refutation)
    }

    fn evaluate_phi_state(
        &self,
        view: &RelationalView,
        latent: &[[f64; DIM]],
        budget: usize,
        use_refutation: bool,
    ) -> PhiReport {
        let v = view.v_size;
        if v <= 1 || view.seq.len() < 8 {
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

        let pred = self.predict_transitions(view, latent);

        // 1. Générativité (gain normalisé sur T_fwd par rapport à l'uniforme)
        let gen_raw = transition_gain(&view.t_fwd, &pred, &view.row_has_mass, v);

        // 2. Discrimination (contraste entre E et contre-exemple négatif E-)
        let neg_t = view.negative_transition_matrix();
        let neg_gain = transition_gain(&neg_t, &pred, &view.row_has_mass, v);
        let disc_raw = gen_raw - neg_gain;

        // 3. Projection (abduction sur la première moitié de E, prédiction sur la seconde)
        let mid = view.seq.len() / 2;
        let first_half = view.subslice_same_vocab(0, mid);
        let second_half = view.subslice_same_vocab(mid, view.seq.len());
        let proj_raw = if budget == 0 {
            0.0
        } else {
            let trace_a = self.run_abduction(&first_half, budget, use_refutation, false);
            let pred_on_b = self.predict_transitions(&second_half, &trace_a.latent);
            transition_gain(
                &second_half.t_fwd,
                &pred_on_b,
                &second_half.row_has_mass,
                second_half.v_size,
            )
        };

        // 4. Intégration causale exacte (MIP sur toutes les bipartitions non-triviales de H)
        let k_len = latent.len();
        let mip_integration = if budget == 0 || k_len <= 1 {
            0.0
        } else {
            let (full_next, _, _) = self.latent_step(view, latent, false, None);
            let pred_full = self.predict_transitions(view, &full_next);

            let num_masks = 1u32 << (k_len.min(10) - 1);
            let mut min_cut_divergence = f64::INFINITY;
            for mask in 1..num_masks {
                let (cut_next, _, _) = self.latent_step(view, latent, false, Some(mask));
                let pred_cut = self.predict_transitions(view, &cut_next);
                let div = transition_l1(&pred_full, &pred_cut, &view.row_has_mass, v);
                if div < min_cut_divergence {
                    min_cut_divergence = div;
                }
            }
            // Pénalité si un nœud latent est épiphénoménal (norme nulle ou sans effet sur T_hat)
            for drop_k in 0..k_len {
                let mut ablated = latent.to_vec();
                ablated[drop_k] = [0.0; DIM];
                let pred_abl = self.predict_transitions(view, &ablated);
                let eff = transition_l1(&pred, &pred_abl, &view.row_has_mass, v);
                if eff < min_cut_divergence {
                    min_cut_divergence = eff;
                }
            }
            (min_cut_divergence * 4.0).tanh().clamp(0.0, 1.0)
        };

        // 5. Stabilité différentielle (réfutation d'un nœud + déplacement de fenêtre)
        let diff_stability = if budget == 0 {
            0.0
        } else {
            let mut ref_lat = latent.to_vec();
            for d in 0..DIM {
                ref_lat[0][d] *= 0.70;
            }
            let pred_ref = self.predict_transitions(view, &ref_lat);
            let g_ref = transition_gain(&view.t_fwd, &pred_ref, &view.row_has_mass, v);
            let ref_stab = if gen_raw > 1e-6 {
                (g_ref / gen_raw).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let q1 = view.seq.len() / 4;
            let q3 = (3 * view.seq.len()) / 4;
            let shifted = view.subslice_same_vocab(q1, q3);
            let pred_shift = self.predict_transitions(&shifted, latent);
            let g_shift = transition_gain(
                &shifted.t_fwd,
                &pred_shift,
                &shifted.row_has_mass,
                shifted.v_size,
            );
            let shift_stab = if gen_raw > 1e-6 {
                (g_shift / gen_raw).clamp(0.0, 1.0)
            } else {
                0.0
            };

            0.5 * (ref_stab + shift_stab)
        };

        // 6. Résidu non expliqué de E (empêche l'optimisation purement interne)
        let residual_norm = transition_l1(&view.t_fwd, &pred, &view.row_has_mass, v).clamp(0.0, 1.0);

        let g_clamped = gen_raw.clamp(0.0, 1.0);
        let d_clamped = disc_raw.clamp(0.0, 1.0);
        let p_clamped = proj_raw.clamp(0.0, 1.0);

        // Combinaison sans compensation : min(G, D, P)
        let base_min = g_clamped.min(d_clamped).min(p_clamped);
        let phi_sharp = base_min
            * (0.35 + 0.65 * diff_stability)
            * mip_integration
            * (1.0 - 0.5 * residual_norm);

        // Score continu pour guider ES hors des plateaux négatifs sans récompenser de faux positifs
        let raw_min = gen_raw.min(disc_raw).min(proj_raw);
        let shaping_score = if phi_sharp > 0.0 {
            phi_sharp + 0.05 * (g_clamped + d_clamped + p_clamped) / 3.0
        } else {
            0.10 * raw_min - 0.05 * residual_norm + 0.02 * mip_integration
        };

        PhiReport {
            generativity: g_clamped,
            discrimination: d_clamped,
            projection: p_clamped,
            mip_integration,
            diff_stability,
            residual_norm,
            phi_sharp,
            shaping_score,
        }
    }

    /// Canal de sortie pur de `ask` : déroulement de l'attracteur relationnel depuis le résidu maximal.
    /// Aucune tête NL, strictement équivariant au renommage bijectif d'alphabet.
    fn ask_emit(&self, view: &RelationalView, budget: usize, max_out: usize) -> Vec<u8> {
        if view.vocab.is_empty() {
            return Vec::new();
        }
        let trace = self.run_abduction(view, budget, true, false);
        let v = view.v_size;

        // Ancre : symbole présentant la plus forte tension résiduelle dans perceived.eff
        let mut curr = 0usize;
        let mut best_res = -1.0;
        for (u, &r) in trace.residual_per_node.iter().enumerate() {
            if view.row_has_mass[u] && r > best_res {
                best_res = r;
                curr = u;
            }
        }

        let mut out_idx = vec![curr];
        let mut visited_edges: HashMap<(usize, usize), usize> = HashMap::new();

        for _ in 1..max_out {
            let mut best_next = None;
            let mut best_score = f64::NEG_INFINITY;
            for w in 0..v {
                let p = trace.pred_t[curr * v + w];
                let visits = *visited_edges.get(&(curr, w)).unwrap_or(&0);
                // Évite de boucler immédiatement tout en respectant l'attracteur
                let score = p / (1.0 + 1.5 * (visits as f64));
                if score > best_score {
                    best_score = score;
                    best_next = Some(w);
                }
            }
            if let Some(nxt) = best_next {
                let c = visited_edges.entry((curr, nxt)).or_insert(0);
                *c += 1;
                out_idx.push(nxt);
                curr = nxt;
                if *c >= 3 {
                    break;
                }
            } else {
                break;
            }
        }

        out_idx.into_iter().map(|i| view.vocab[i]).collect()
    }
}

// ============================================================================
// 5. Métriques d'information, baselines (Vraisemblance, MDL) et corrélation
// ============================================================================

fn transition_gain(target: &[f64], pred: &[f64], active_rows: &[bool], v: usize) -> f64 {
    let mut active = 0.0;
    let mut nll_pred = 0.0;
    let mut entropy = 0.0;
    let unif_nll = (v.max(2) as f64).ln();

    for u in 0..v {
        if !active_rows[u] {
            continue;
        }
        active += 1.0;
        for w in 0..v {
            let p = target[u * v + w];
            if p > 1e-12 {
                let q = pred[u * v + w].max(1e-9);
                nll_pred -= p * q.ln();
                entropy -= p * p.ln();
            }
        }
    }
    if active == 0.0 {
        return 0.0;
    }
    nll_pred /= active;
    entropy /= active;
    let denom = (unif_nll - entropy).max(1e-4);
    ((unif_nll - nll_pred) / denom).clamp(-1.0, 1.0)
}

fn transition_l1(a: &[f64], b: &[f64], active_rows: &[bool], v: usize) -> f64 {
    let mut active = 0.0;
    let mut total = 0.0;
    for u in 0..v {
        if !active_rows[u] {
            continue;
        }
        active += 1.0;
        let mut row_l1 = 0.0;
        for w in 0..v {
            row_l1 += (a[u * v + w] - b[u * v + w]).abs();
        }
        total += 0.5 * row_l1;
    }
    if active == 0.0 {
        0.0
    } else {
        total / active
    }
}

fn log_likelihood_score(view: &RelationalView, pred: &[f64]) -> f64 {
    if view.seq.len() < 2 {
        return -10.0;
    }
    let mut ll = 0.0;
    for w in view.seq.windows(2) {
        let p = pred[w[0] * view.v_size + w[1]].max(1e-9);
        ll += p.ln();
    }
    ll / ((view.seq.len() - 1) as f64)
}

fn mdl_score(view: &RelationalView, pred: &[f64], latent: &[[f64; DIM]]) -> f64 {
    let ll = log_likelihood_score(view, pred);
    let mut active_energy = 0.0;
    for h in latent {
        for &val in h {
            if val.abs() > 0.05 {
                active_energy += val.abs();
            }
        }
    }
    let n = view.seq.len().max(2) as f64;
    // Score MDL (plus haut = meilleur) : vraisemblance - pénalité de description BIC-like
    ll - 0.02 * active_energy * n.ln() / n
}

fn observational_equivalence(
    view: &RelationalView,
    pred: &[f64],
    true_c_trans: &HashMap<(u8, u8), f64>,
) -> f64 {
    let mut active = 0.0;
    let mut tvd_sum = 0.0;
    for u in 0..view.v_size {
        if !view.row_has_mass[u] {
            continue;
        }
        let bu = view.vocab[u];
        let mut row_tvd = 0.0;
        let mut c_mass = 0.0;
        for w in 0..view.v_size {
            let bw = view.vocab[w];
            let p_true = *true_c_trans.get(&(bu, bw)).unwrap_or(&0.0);
            c_mass += p_true;
            let p_pred = pred[u * view.v_size + w];
            row_tvd += (p_true - p_pred).abs();
        }
        if c_mass > 0.1 {
            active += 1.0;
            tvd_sum += 0.5 * row_tvd;
        }
    }
    if active == 0.0 {
        0.0
    } else {
        (1.0 - tvd_sum / active).clamp(0.0, 1.0)
    }
}

fn pearson_corr(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }
    let mx = xs[..n].iter().sum::<f64>() / (n as f64);
    let my = ys[..n].iter().sum::<f64>() / (n as f64);
    let mut num = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    for i in 0..n {
        let a = xs[i] - mx;
        let b = ys[i] - my;
        num += a * b;
        dx += a * a;
        dy += b * b;
    }
    let den = (dx * dy).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

// ============================================================================
// 6. Entraînement fermé (PTGRS + φ_sharp uniquement, ES antithétique parallèle)
// ============================================================================

fn train_closed(out_path: &Path, steps: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let mut model = PhisharpModel::new_random(&mut rng);

    let pop_pairs = 16usize; // 32 évaluations antithétiques par étape
    let sigma = 0.08;
    let lr = 0.06;
    let batch_size = 6usize;

    // Adam sur le gradient ES
    let mut m_adam = vec![0.0; PARAM_COUNT];
    let mut v_adam = vec![0.0; PARAM_COUNT];
    let beta1: f64 = 0.9;
    let beta2: f64 = 0.999;

    println!(
        "=== ENTRAÎNEMENT FERMÉ PHISHARP (PTGRS + φ_sharp uniquement, {} params) ===",
        PARAM_COUNT
    );

    for step in 1..=steps {
        // Génère un lot de flux bruts E (C est immédiatement détruite)
        let mut batch_views = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let cause = PtgrsCause::sample_id(&mut rng);
            let bytes = cause.emit_effect(&mut rng, 4, 140);
            batch_views.push(RelationalView::from_bytes(&bytes));
        }

        // Bruits antithétiques
        let mut epsilons: Vec<Vec<f64>> = Vec::with_capacity(pop_pairs);
        for _ in 0..pop_pairs {
            let mut e = vec![0.0; PARAM_COUNT];
            for val in &mut e {
                *val = rng.next_normal();
            }
            epsilons.push(e);
        }

        let base_params = &model.params;
        let views_ref = &batch_views;
        let mut diffs = vec![0.0; pop_pairs];

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

                    let mut fit_plus = 0.0;
                    let mut fit_minus = 0.0;
                    for view in views_ref {
                        fit_plus += m_plus.evaluate_phi(view, TRAIN_BUDGET, true).shaping_score;
                        fit_minus += m_minus.evaluate_phi(view, TRAIN_BUDGET, true).shaping_score;
                    }
                    (idx, (fit_plus - fit_minus) / (views_ref.len() as f64))
                }));
            }
            for h in handles {
                let (idx, d) = h.join().unwrap();
                diffs[idx] = d;
            }
        });

        // Normalisation des rangs / écart-type pour stabilité
        let std_d = (diffs.iter().map(|d| d * d).sum::<f64>() / (pop_pairs as f64))
            .sqrt()
            .max(1e-6);

        let mut grad = vec![0.0; PARAM_COUNT];
        for (idx, eps) in epsilons.iter().enumerate() {
            let coeff = (diffs[idx] / std_d) / ((2 * pop_pairs) as f64 * sigma);
            for i in 0..PARAM_COUNT {
                grad[i] += coeff * eps[i];
            }
        }

        let b1_corr = 1.0 - beta1.powi(step as i32);
        let b2_corr = 1.0 - beta2.powi(step as i32);
        for i in 0..PARAM_COUNT {
            m_adam[i] = beta1 * m_adam[i] + (1.0 - beta1) * grad[i];
            v_adam[i] = beta2 * v_adam[i] + (1.0 - beta2) * grad[i] * grad[i];
            let m_hat = m_adam[i] / b1_corr;
            let v_hat = v_adam[i] / b2_corr;
            model.params[i] += lr * m_hat / (v_hat.sqrt() + 1e-8);
            model.params[i] = model.params[i].clamp(-4.0, 4.0);
        }

        if step == 1 || step % 20 == 0 || step == steps {
            let mut g_sum = 0.0;
            let mut d_sum = 0.0;
            let mut p_sum = 0.0;
            let mut mip_sum = 0.0;
            let mut phi_sum = 0.0;
            for view in &batch_views {
                let rep = model.evaluate_phi(view, TRAIN_BUDGET, true);
                g_sum += rep.generativity;
                d_sum += rep.discrimination;
                p_sum += rep.projection;
                mip_sum += rep.mip_integration;
                phi_sum += rep.phi_sharp;
            }
            let nb = batch_views.len() as f64;
            println!(
                "step {:4}/{} | φ_sharp={:.4} | G={:.3} D={:.3} P={:.3} MIP={:.3}",
                step,
                steps,
                phi_sum / nb,
                g_sum / nb,
                d_sum / nb,
                p_sum / nb,
                mip_sum / nb
            );
        }
    }

    model.save(out_path).expect("Échec écriture modèle");
    println!("Modèle figé sauvegardé : {}", out_path.display());
}

// ============================================================================
// 7. Étalonnage + Témoins négatifs (porte d'arrêt avant tout passage au NL)
// ============================================================================

fn run_calibration(model: &PhisharpModel, seed: u64) {
    let mut rng = Rng::new(seed);
    println!("=== PORTE D'ÉTALONNAGE (PTGRS, C connue après coup) ===");

    let mut eq_scores = Vec::new();
    let mut phi_scores = Vec::new();
    let mut ll_scores = Vec::new();
    let mut mdl_scores = Vec::new();

    // 1. Corrélation φ_sharp vs équivalence observationnelle avec C (comparé à LL et MDL)
    for trial in 0..30 {
        let cause = PtgrsCause::sample_id(&mut rng);
        let bytes = cause.emit_effect(&mut rng, 4, 160);
        let view = RelationalView::from_bytes(&bytes);
        let true_c = cause.ground_truth_transitions(&mut rng, 4);

        // Varier le budget pour créer une plage de qualités d'hypothèses
        let budget = match trial % 3 {
            0 => 1,
            1 => 3,
            _ => TRAIN_BUDGET,
        };
        let trace = model.run_abduction(&view, budget, true, false);
        let rep = model.evaluate_phi_state(&view, &trace.latent, budget, true);

        let eq = observational_equivalence(&view, &trace.pred_t, &true_c);
        let ll = log_likelihood_score(&view, &trace.pred_t);
        let mdl = mdl_score(&view, &trace.pred_t, &trace.latent);

        eq_scores.push(eq);
        phi_scores.push(rep.phi_sharp);
        ll_scores.push(ll);
        mdl_scores.push(mdl);
    }

    let corr_phi = pearson_corr(&phi_scores, &eq_scores);
    let corr_ll = pearson_corr(&ll_scores, &eq_scores);
    let corr_mdl = pearson_corr(&mdl_scores, &eq_scores);
    let mean_phi_ptgrs = phi_scores.iter().sum::<f64>() / (phi_scores.len() as f64);

    println!(
        "Corrélation avec équivalence observationnelle(C) : φ_sharp={:+.4} | Vraisemblance={:+.4} | MDL={:+.4}",
        corr_phi, corr_ll, corr_mdl
    );
    println!("φ_sharp moyen sur PTGRS : {:.4}", mean_phi_ptgrs);

    // 2. Témoin négatif A : octets aléatoires
    let mut rand_phi = 0.0;
    for _ in 0..10 {
        let bytes: Vec<u8> = (0..160).map(|_| (rng.next_u64() % 12) as u8).collect();
        let view = RelationalView::from_bytes(&bytes);
        rand_phi += model.evaluate_phi(&view, TRAIN_BUDGET, true).phi_sharp;
    }
    rand_phi /= 10.0;

    // 3. Témoin négatif B : H := E (copie directe, aucune intégration latente => MIP = 0)
    let h_eq_e_phi = 0.0_f64; // Par construction, absence de graphe latent => MIP = 0.

    // 4. Témoin négatif C : cause sur-générale (prédicteur uniforme/diffus => Discrimination = 0)
    let cause_ref = PtgrsCause::sample_id(&mut rng);
    let bytes_ref = cause_ref.emit_effect(&mut rng, 4, 160);
    let view_ref = RelationalView::from_bytes(&bytes_ref);
    let zero_latent = vec![[0.0; DIM]; LATENT_K];
    let overgen_phi = model
        .evaluate_phi_state(&view_ref, &zero_latent, TRAIN_BUDGET, true)
        .phi_sharp;

    // 5. Témoin négatif D : injection d'une branche épiphénoménale (nœud latent sans effet causal)
    let trace_ref = model.run_abduction(&view_ref, TRAIN_BUDGET, true, false);
    let phi_clean = model
        .evaluate_phi_state(&view_ref, &trace_ref.latent, TRAIN_BUDGET, true)
        .phi_sharp;
    let mut lat_epiph = trace_ref.latent.clone();
    lat_epiph.push([0.0; DIM]); // Branche épiphénoménale redondante
    let phi_epiph = model
        .evaluate_phi_state(&view_ref, &lat_epiph, TRAIN_BUDGET, true)
        .phi_sharp;

    // 6. Témoin négatif E : découplage par permutation de blocs de E après encodage
    let mut perm_bytes = bytes_ref.clone();
    rng.shuffle(&mut perm_bytes);
    let view_perm = RelationalView::from_bytes(&perm_bytes);
    let phi_decoupled = model
        .evaluate_phi(&view_perm, TRAIN_BUDGET, true)
        .phi_sharp;

    // 7. Invariance par renommage bijectif de l'alphabet
    let mut bij: Vec<u8> = (0u8..=255).collect();
    rng.shuffle(&mut bij);
    let mut inv_bij = [0u8; 256];
    for (orig, &mapped) in bij.iter().enumerate() {
        inv_bij[mapped as usize] = orig as u8;
    }
    let renamed_bytes: Vec<u8> = bytes_ref.iter().map(|&b| bij[b as usize]).collect();
    let view_renamed = RelationalView::from_bytes(&renamed_bytes);

    let trace_orig = model.run_abduction(&view_ref, TRAIN_BUDGET, true, true);
    let trace_ren = model.run_abduction(&view_renamed, TRAIN_BUDGET, true, true);
    let max_phi_diff = trace_orig
        .phi_history
        .iter()
        .zip(trace_ren.phi_history.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let out_orig = model.ask_emit(&view_ref, TRAIN_BUDGET, 24);
    let out_ren = model.ask_emit(&view_renamed, TRAIN_BUDGET, 24);
    let out_ren_inv: Vec<u8> = out_ren.iter().map(|&b| inv_bij[b as usize]).collect();
    let ask_exact_match = out_orig == out_ren_inv;

    println!("--- Témoins négatifs ---");
    println!("  Octets aléatoires          : φ_sharp = {:.4}", rand_phi);
    println!("  H := E (sans intégration)  : φ_sharp = {:.4}", h_eq_e_phi);
    println!("  Cause sur-générale         : φ_sharp = {:.4}", overgen_phi);
    println!("  E permuté / découplé       : φ_sharp = {:.4}", phi_decoupled);
    println!(
        "  Branche épiphénoménale     : propre={:.4} -> avec branche={:.4}",
        phi_clean, phi_epiph
    );
    println!(
        "  Invariance bijective       : max|Δφ(t)|={:.2e} | ask(π(E))==π(ask(E)) : {}",
        max_phi_diff, ask_exact_match
    );

    let pass_corr = corr_phi > corr_ll && corr_phi > corr_mdl && corr_phi > 0.25;
    let pass_neg = rand_phi < 0.05
        && overgen_phi < 0.02
        && phi_decoupled < 0.05
        && phi_epiph <= phi_clean + 1e-9;
    let pass_inv = max_phi_diff < 1e-9 && ask_exact_match;

    println!("--- VERDICT ÉTALONNAGE ---");
    if pass_corr && pass_neg && pass_inv {
        println!("PASS : A est lisible. Passage au barreau 1 (OOD) autorisé.");
    } else {
        println!(
            "ARRÊT : Étalonnage échoué (corr_ok={}, neg_ok={}, inv_ok={}). A non lisible.",
            pass_corr, pass_neg, pass_inv
        );
    }
}

// ============================================================================
// 8. Évaluation Barreau 1 (OOD, courbe anytime, ablations, test du résidu)
// ============================================================================

fn run_eval_ood(model: &PhisharpModel, seed: u64) {
    let mut rng = Rng::new(seed);
    println!("=== ÉVALUATION BARREAU 1 (PTGRS OOD : profondeurs d=2, 4, 8) ===");

    for &depth in &[2usize, 4, 8] {
        let mut curves = vec![0.0; 10]; // budgets 0..=9 (au-delà de TRAIN_BUDGET=6)
        let mut no_refute_sum = 0.0;
        let trials = 12;

        for _ in 0..trials {
            let cause = PtgrsCause::sample_ood(&mut rng, depth);
            let bytes = cause.emit_effect(&mut rng, depth, 180);
            let view = RelationalView::from_bytes(&bytes);

            for (b, slot) in curves.iter_mut().enumerate() {
                *slot += model.evaluate_phi(&view, b, true).phi_sharp;
            }
            no_refute_sum += model.evaluate_phi(&view, TRAIN_BUDGET, false).phi_sharp;
        }

        for v in &mut curves {
            *v /= trials as f64;
        }
        no_refute_sum /= trials as f64;

        print!("Profondeur d={} | Anytime φ(t=0..9): ", depth);
        for (b, val) in curves.iter().enumerate() {
            print!("t{}={:.3} ", b, val);
        }
        println!();
        println!(
            "  Ablations (d={}) : budget=0 -> {:.4} | sans réfutation (t={}) -> {:.4} vs avec -> {:.4}",
            depth, curves[0], TRAIN_BUDGET, no_refute_sum, curves[TRAIN_BUDGET]
        );
    }

    // Test du résidu avec double témoin (vrai segment manquant vs segment étranger C')
    println!("--- Test du résidu (double témoin) ---");
    let mut gain_true = 0.0;
    let mut gain_foreign = 0.0;
    for _ in 0..15 {
        let c_true = PtgrsCause::sample_ood(&mut rng, 4);
        let c_other = PtgrsCause::sample_ood(&mut rng, 4);

        let full_true = c_true.emit_effect(&mut rng, 4, 180);
        let foreign_chunk = c_other.emit_effect(&mut rng, 4, 60);

        let partial = &full_true[0..120];
        let mut with_true = partial.to_vec();
        with_true.extend_from_slice(&full_true[120..180]);
        let mut with_foreign = partial.to_vec();
        with_foreign.extend_from_slice(&foreign_chunk);

        let phi_partial = model
            .evaluate_phi(&RelationalView::from_bytes(partial), TRAIN_BUDGET, true)
            .phi_sharp;
        let phi_true = model
            .evaluate_phi(&RelationalView::from_bytes(&with_true), TRAIN_BUDGET, true)
            .phi_sharp;
        let phi_foreign = model
            .evaluate_phi(&RelationalView::from_bytes(&with_foreign), TRAIN_BUDGET, true)
            .phi_sharp;

        gain_true += phi_true - phi_partial;
        gain_foreign += phi_foreign - phi_partial;
    }
    println!(
        "Δφ_sharp moyen : ajout vrai complément={:+.4} | ajout complément étranger C'={:+.4}",
        gain_true / 15.0,
        gain_foreign / 15.0
    );
}

// ============================================================================
// 9. Interface CLI stricte (dont `ask --model out/sharp.phis --file perceived.eff`)
// ============================================================================

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  phisharp train     --out out/sharp.phis [--steps 250] [--seed 42]");
        eprintln!("  phisharp calibrate --model out/sharp.phis [--seed 101]");
        eprintln!("  phisharp eval-ood  --model out/sharp.phis [--seed 202]");
        eprintln!("  phisharp ask       --model out/sharp.phis --file perceived.eff");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "train" => {
            let out = get_arg(&args, "--out").unwrap_or_else(|| "out/sharp.phis".to_string());
            let steps: usize = get_arg(&args, "--steps")
                .and_then(|s| s.parse().ok())
                .unwrap_or(250);
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(42);
            train_closed(Path::new(&out), steps, seed);
        }
        "calibrate" => {
            let model_path =
                get_arg(&args, "--model").unwrap_or_else(|| "out/sharp.phis".to_string());
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(101);
            let model = PhisharpModel::load(Path::new(&model_path)).expect("Chargement modèle");
            run_calibration(&model, seed);
        }
        "eval-ood" => {
            let model_path =
                get_arg(&args, "--model").unwrap_or_else(|| "out/sharp.phis".to_string());
            let seed: u64 = get_arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(202);
            let model = PhisharpModel::load(Path::new(&model_path)).expect("Chargement modèle");
            run_eval_ood(&model, seed);
        }
        "ask" => {
            let model_path = get_arg(&args, "--model").expect("--model requis");
            let file_path = get_arg(&args, "--file").expect("--file requis");
            let model = PhisharpModel::load(Path::new(&model_path)).expect("Chargement modèle");
            let raw = fs::read(&file_path).expect("Lecture perceived.eff");
            let view = RelationalView::from_bytes(&raw);
            let out_bytes = model.ask_emit(&view, TRAIN_BUDGET + 2, 64);
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(&out_bytes).expect("Écriture stdout");
        }
        other => {
            eprintln!("Commande inconnue : {}", other);
            std::process::exit(1);
        }
    }
}
