//! Probabilistic Term Graph Rewriting System (PTGRS).
//!
//! Séparation stricte des types :
//! - `PtgrsSample` : contient UNIQUEMENT `observed_e: Vec<u8>`. C'est la seule structure vue à l'entraînement.
//! - `PtgrsEvalCase` : contient `observed_e` + la cause cachée `HiddenCause` (pour l'évaluation post-hoc uniquement).
//!
//! Tout échantillon PTGRS utilise une bijection aléatoire d'octets pour que l'alphabet soit
//! strictement auto-spécifié et exempt de tout prior lexical ou symbolique fixe.

use crate::math::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleFamily {
    /// Familles vues à l'entraînement (PTGRS fermé)
    Compose,
    PermuteBind,
    ShareFold,
    ContextualShift,
    ChainStep,
    /// Familles hors-distribution (jamais vues à l'entraînement)
    OodDeepNested,
    OodCyclicEquiv,
    OodRelationalTransfer,
}

impl RuleFamily {
    pub fn in_dist_all() -> &'static [RuleFamily] {
        &[
            RuleFamily::Compose,
            RuleFamily::PermuteBind,
            RuleFamily::ShareFold,
            RuleFamily::ContextualShift,
            RuleFamily::ChainStep,
        ]
    }

    pub fn ood_all() -> &'static [RuleFamily] {
        &[
            RuleFamily::OodDeepNested,
            RuleFamily::OodCyclicEquiv,
            RuleFamily::OodRelationalTransfer,
        ]
    }
}

#[derive(Clone, Debug)]
pub struct TermNode {
    pub id: usize,
    pub op: u8,          // identifiant abstrait d'opérateur (0..op_vocab)
    pub children: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct TermGraph {
    pub nodes: Vec<TermNode>,
    pub root: usize,
}

impl TermGraph {
    pub fn canonical_signature(&self, node_id: usize, depth: usize, visited: &mut Vec<usize>) -> u64 {
        if depth > 12 || node_id >= self.nodes.len() {
            return 0x517C_C1B7_2722_0A95;
        }
        if let Some(pos) = visited.iter().position(|&x| x == node_id) {
            // Partage / cycle dans le graphe de termes
            return 0x9E37_79B9_0000_0000u64 ^ (pos as u64);
        }
        visited.push(node_id);
        let node = &self.nodes[node_id];
        let mut h = (node.op as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ (node.children.len() as u64);
        for (idx, &ch) in node.children.iter().enumerate() {
            let sub = self.canonical_signature(ch, depth + 1, visited);
            h = h.rotate_left(11) ^ sub.wrapping_add((idx as u64 + 1).wrapping_mul(0x94D0_49BB_1331_11EB));
        }
        visited.pop();
        h
    }

    /// Sérialise le graphe sous forme d'expression relationnelle dans l'alphabet fourni.
    pub fn render_in_alphabet(&self, node_id: usize, alphabet: &AlphabetMap, depth: usize, out: &mut Vec<u8>, visited: &mut Vec<usize>) {
        if depth > 7 || node_id >= self.nodes.len() {
            out.push(alphabet.wildcard_byte());
            return;
        }
        if let Some(pos) = visited.iter().position(|&x| x == node_id) {
            out.push(alphabet.ref_marker());
            out.push(alphabet.op_byte((pos % alphabet.op_count()) as u8));
            return;
        }
        visited.push(node_id);
        let node = &self.nodes[node_id];
        out.push(alphabet.op_byte(node.op));
        if !node.children.is_empty() {
            out.push(alphabet.open_delim());
            for (i, &ch) in node.children.iter().enumerate() {
                if i > 0 {
                    out.push(alphabet.sep_byte());
                }
                self.render_in_alphabet(ch, alphabet, depth + 1, out, visited);
            }
            out.push(alphabet.close_delim());
        }
        visited.pop();
    }
}

#[derive(Clone, Debug)]
pub enum RhsTemplate {
    /// f(x, y) -> g(h(x), y)
    Compose { g: u8, h: u8 },
    /// f(x, y) -> k(y, x)
    Permute { k: u8 },
    /// f(x) -> d(x, x) (partage de sous-terme dans le graphe)
    Share { d: u8 },
    /// c(a, x) -> r(x, a) avec permutation d'arité variable
    Shift { r: u8 },
    /// x -> s(x) ou s(f(x)) en chaîne
    Chain { s: u8 },
    /// OOD : f(g(x, y), z) -> h(z, m(x, y, x))
    OodDeep { h: u8, m: u8 },
    /// OOD : transformation cyclique à parité alternée
    OodCyclic { p: u8, q: u8 },
    /// OOD : composition transitive relationnelle (a~b, b~c -> a~c)
    OodTransitive { rel: u8 },
}

#[derive(Clone, Debug)]
pub struct RewriteRule {
    pub family: RuleFamily,
    pub lhs_op: u8,
    pub prob: f64,
    pub template: RhsTemplate,
}

#[derive(Clone, Debug)]
pub struct HiddenCause {
    pub rules: Vec<RewriteRule>,
    pub seed_graph: TermGraph,
    pub alphabet: AlphabetMap,
    pub num_steps: usize,
}

/// Bijection aléatoire d'octets propre à chaque instance PTGRS.
#[derive(Clone, Debug)]
pub struct AlphabetMap {
    pub bytes: Vec<u8>, // longueur >= 24 : [ops..., open, close, sep, arrow, clause_end, ref_mark, wildcard, query_mark]
    pub num_ops: usize,
}

impl AlphabetMap {
    pub fn random(rng: &mut Rng, num_ops: usize) -> Self {
        // On permute des octets imprimables ASCII (33..=126) ou octets quelconques (1..=254)
        // L'usage d'octets quelconques garantit l'absence de tout biais ASCII.
        let mut pool: Vec<u8> = (1u8..=254u8).collect();
        rng.shuffle(&mut pool);
        let total = num_ops + 8;
        Self {
            bytes: pool[..total].to_vec(),
            num_ops,
        }
    }

    pub fn op_count(&self) -> usize {
        self.num_ops
    }

    #[inline]
    pub fn op_byte(&self, op: u8) -> u8 {
        self.bytes[(op as usize) % self.num_ops]
    }
    #[inline]
    pub fn open_delim(&self) -> u8 { self.bytes[self.num_ops] }
    #[inline]
    pub fn close_delim(&self) -> u8 { self.bytes[self.num_ops + 1] }
    #[inline]
    pub fn sep_byte(&self) -> u8 { self.bytes[self.num_ops + 2] }
    #[inline]
    pub fn arrow_byte(&self) -> u8 { self.bytes[self.num_ops + 3] }
    #[inline]
    pub fn clause_end(&self) -> u8 { self.bytes[self.num_ops + 4] }
    #[inline]
    pub fn ref_marker(&self) -> u8 { self.bytes[self.num_ops + 5] }
    #[inline]
    pub fn wildcard_byte(&self) -> u8 { self.bytes[self.num_ops + 6] }
    #[inline]
    pub fn query_marker(&self) -> u8 { self.bytes[self.num_ops + 7] }
}

/// Donnée d'entraînement : UNIQUEMENT l'effet observé E en octets.
#[derive(Clone, Debug)]
pub struct PtgrsSample {
    pub observed_e: Vec<u8>,
}

/// Donnée d'évaluation : inclut la cause cachée C (jamais exposée à l'entraînement).
#[derive(Clone, Debug)]
pub struct PtgrsEvalCase {
    pub observed_e: Vec<u8>,
    pub cause: HiddenCause,
    pub target_explanation: Vec<u8>,
    pub ood: bool,
}

fn build_random_term_graph(rng: &mut Rng, num_ops: usize, num_nodes: usize) -> TermGraph {
    let mut nodes = Vec::with_capacity(num_nodes);
    for id in 0..num_nodes {
        let op = rng.usize_below(num_ops) as u8;
        let max_arity = if id == 0 { 0 } else { 2.min(id) };
        let arity = if id == 0 { 0 } else { rng.usize_below(max_arity + 1) };
        let mut children = Vec::with_capacity(arity);
        for _ in 0..arity {
            children.push(rng.usize_below(id));
        }
        nodes.push(TermNode { id, op, children });
    }
    TermGraph {
        root: num_nodes - 1,
        nodes,
    }
}

fn apply_rule_step(graph: &mut TermGraph, rules: &[RewriteRule], rng: &mut Rng, step_idx: usize) -> bool {
    let mut candidates = Vec::new();
    for (nid, node) in graph.nodes.iter().enumerate() {
        for (rid, rule) in rules.iter().enumerate() {
            if node.op == rule.lhs_op {
                candidates.push((nid, rid, rule.prob));
            }
        }
    }
    if candidates.is_empty() {
        return false;
    }
    let probs: Vec<f64> = candidates.iter().map(|c| c.2).collect();
    let pick = rng.sample_categorical(&probs);
    let (nid, rid, _) = candidates[pick];
    let rule = &rules[rid];

    let existing_children = graph.nodes[nid].children.clone();
    let c0 = *existing_children.first().unwrap_or(&0);
    let c1 = *existing_children.get(1).unwrap_or(&c0);

    match &rule.template {
        RhsTemplate::Compose { g, h } => {
            let h_id = graph.nodes.len();
            graph.nodes.push(TermNode { id: h_id, op: *h, children: vec![c0] });
            graph.nodes[nid].op = *g;
            graph.nodes[nid].children = vec![h_id, c1];
        }
        RhsTemplate::Permute { k } => {
            graph.nodes[nid].op = *k;
            graph.nodes[nid].children = if existing_children.len() >= 2 {
                vec![c1, c0]
            } else {
                vec![c0]
            };
        }
        RhsTemplate::Share { d } => {
            graph.nodes[nid].op = *d;
            graph.nodes[nid].children = vec![c0, c0];
        }
        RhsTemplate::Shift { r } => {
            graph.nodes[nid].op = *r;
            if existing_children.len() >= 2 {
                let mut rot = existing_children;
                rot.rotate_left(1);
                graph.nodes[nid].children = rot;
            }
        }
        RhsTemplate::Chain { s } => {
            let inner = graph.nodes.len();
            graph.nodes.push(TermNode {
                id: inner,
                op: graph.nodes[nid].op,
                children: existing_children,
            });
            graph.nodes[nid].op = *s;
            graph.nodes[nid].children = vec![inner];
        }
        RhsTemplate::OodDeep { h, m } => {
            let c2 = *existing_children.get(2).unwrap_or(&c1);
            let m_id = graph.nodes.len();
            graph.nodes.push(TermNode {
                id: m_id,
                op: *m,
                children: vec![c0, c1, c0],
            });
            graph.nodes[nid].op = *h;
            graph.nodes[nid].children = vec![c2, m_id];
        }
        RhsTemplate::OodCyclic { p, q } => {
            let next_op = if step_idx % 2 == 0 { *p } else { *q };
            let prev_id = graph.nodes.len();
            graph.nodes.push(TermNode {
                id: prev_id,
                op: next_op,
                children: vec![c0],
            });
            graph.nodes[nid].op = if step_idx % 2 == 0 { *q } else { *p };
            // Partage mutuel dans le graphe
            graph.nodes[nid].children = vec![prev_id, c1];
        }
        RhsTemplate::OodTransitive { rel } => {
            let bridge = graph.nodes.len();
            graph.nodes.push(TermNode {
                id: bridge,
                op: *rel,
                children: vec![c1, c0],
            });
            graph.nodes[nid].op = *rel;
            graph.nodes[nid].children = vec![c0, bridge];
        }
    }
    true
}

fn generate_rules(rng: &mut Rng, num_ops: usize, num_rules: usize, ood: bool) -> Vec<RewriteRule> {
    let families = if ood { RuleFamily::ood_all() } else { RuleFamily::in_dist_all() };
    let mut rules = Vec::with_capacity(num_rules);
    for i in 0..num_rules {
        let fam = families[(i + rng.usize_below(families.len())) % families.len()];
        let lhs_op = (i % num_ops) as u8;
        let prob = 0.25 + 0.75 * rng.next_f64();
        let op_rand = |r: &mut Rng| (r.usize_below(num_ops)) as u8;
        let template = match fam {
            RuleFamily::Compose => RhsTemplate::Compose { g: op_rand(rng), h: op_rand(rng) },
            RuleFamily::PermuteBind => RhsTemplate::Permute { k: op_rand(rng) },
            RuleFamily::ShareFold => RhsTemplate::Share { d: op_rand(rng) },
            RuleFamily::ContextualShift => RhsTemplate::Shift { r: op_rand(rng) },
            RuleFamily::ChainStep => RhsTemplate::Chain { s: op_rand(rng) },
            RuleFamily::OodDeepNested => RhsTemplate::OodDeep { h: op_rand(rng), m: op_rand(rng) },
            RuleFamily::OodCyclicEquiv => RhsTemplate::OodCyclic { p: op_rand(rng), q: op_rand(rng) },
            RuleFamily::OodRelationalTransfer => RhsTemplate::OodTransitive { rel: op_rand(rng) },
        };
        rules.push(RewriteRule { family: fam, lhs_op, prob, template });
    }
    rules
}

fn build_eval_case_internal(rng: &mut Rng, ood: bool, num_ops: usize, num_steps: usize) -> PtgrsEvalCase {
    let alphabet = AlphabetMap::random(rng, num_ops);
    let num_rules = if ood { 5 + rng.usize_below(4) } else { 3 + rng.usize_below(4) };
    let rules = generate_rules(rng, num_ops, num_rules, ood);
    let seed_graph = build_random_term_graph(rng, num_ops, if ood { 10 } else { 7 });

    let mut current = seed_graph.clone();
    let mut observed_e = Vec::with_capacity(256);
    let mut visited = Vec::new();

    // Chaque pas de réécriture produit des paires relationnelles (avant -> après) qui composent E.
    // E contient donc la trace des effets observables dans l'alphabet auto-spécifié.
    for step in 0..num_steps {
        let mut before_bytes = Vec::new();
        current.render_in_alphabet(current.root, &alphabet, 0, &mut before_bytes, &mut visited);
        visited.clear();

        let applied = apply_rule_step(&mut current, &rules, rng, step);

        let mut after_bytes = Vec::new();
        current.render_in_alphabet(current.root, &alphabet, 0, &mut after_bytes, &mut visited);
        visited.clear();

        if applied {
            observed_e.extend_from_slice(&before_bytes);
            observed_e.push(alphabet.arrow_byte());
            observed_e.extend_from_slice(&after_bytes);
            observed_e.push(alphabet.clause_end());
        } else {
            // Sonde d'un sous-terme actif pour entretenir la couverture relationnelle
            let sub = rng.usize_below(current.nodes.len());
            current.render_in_alphabet(sub, &alphabet, 0, &mut observed_e, &mut visited);
            visited.clear();
            observed_e.push(alphabet.clause_end());
        }
    }

    // Dernière observation (état résultant / effet terminal)
    observed_e.push(alphabet.query_marker());
    let mut target_explanation = Vec::new();
    current.render_in_alphabet(current.root, &alphabet, 0, &mut target_explanation, &mut visited);
    observed_e.extend_from_slice(&target_explanation);
    observed_e.push(alphabet.clause_end());

    // Garantir une longueur exploitable et bornée
    if observed_e.len() > 768 {
        observed_e.truncate(768);
    }

    PtgrsEvalCase {
        observed_e,
        cause: HiddenCause {
            rules,
            seed_graph,
            alphabet,
            num_steps,
        },
        target_explanation,
        ood,
    }
}

/// Génère un échantillon d'entraînement FERMÉ : uniquement E (jamais C).
pub fn sample_training_effect(rng: &mut Rng) -> PtgrsSample {
    let num_ops = 6 + rng.usize_below(7);
    let num_steps = 5 + rng.usize_below(6);
    let case = build_eval_case_internal(rng, false, num_ops, num_steps);
    PtgrsSample {
        observed_e: case.observed_e,
    }
}

/// Génère un cas d'évaluation PTGRS complet (in-dist ou OOD) avec cause cachée post-hoc.
pub fn sample_eval_case(rng: &mut Rng, ood: bool) -> PtgrsEvalCase {
    let num_ops = if ood { 9 + rng.usize_below(6) } else { 6 + rng.usize_below(7) };
    let num_steps = if ood { 8 + rng.usize_below(6) } else { 5 + rng.usize_below(6) };
    build_eval_case_internal(rng, ood, num_ops, num_steps)
}

/// Échelon 2 de l'échelle d'évaluation : Mini-dictionnaire généré par PTGRS (cause C connue post-hoc).
/// Les règles de réécriture forment des "entrées de dictionnaire" relationnelles suivies d'une requête implicite.
pub fn sample_mini_dictionary(rng: &mut Rng, ood: bool) -> PtgrsEvalCase {
    let num_ops = 10 + rng.usize_below(6);
    let alphabet = AlphabetMap::random(rng, num_ops);
    let rules = generate_rules(rng, num_ops, 6, ood);
    let seed_graph = build_random_term_graph(rng, num_ops, 8);

    let mut observed_e = Vec::with_capacity(512);
    let mut visited = Vec::new();

    // Chaque règle génère plusieurs exemples définitionnels dans le mini-dictionnaire
    for r in &rules {
        let mut g = TermGraph {
            nodes: vec![
                TermNode { id: 0, op: (r.lhs_op + 1) % (num_ops as u8), children: vec![] },
                TermNode { id: 1, op: (r.lhs_op + 2) % (num_ops as u8), children: vec![] },
                TermNode { id: 2, op: r.lhs_op, children: vec![0, 1] },
            ],
            root: 2,
        };
        g.render_in_alphabet(g.root, &alphabet, 0, &mut observed_e, &mut visited);
        visited.clear();
        observed_e.push(alphabet.arrow_byte());
        apply_rule_step(&mut g, std::slice::from_ref(r), rng, 0);
        g.render_in_alphabet(g.root, &alphabet, 0, &mut observed_e, &mut visited);
        visited.clear();
        observed_e.push(alphabet.clause_end());
    }

    // Requête composite finale dérivée par la composition des règles cachées
    let mut query_g = seed_graph.clone();
    observed_e.push(alphabet.query_marker());
    query_g.render_in_alphabet(query_g.root, &alphabet, 0, &mut observed_e, &mut visited);
    visited.clear();
    observed_e.push(alphabet.clause_end());

    for s in 0..4 {
        apply_rule_step(&mut query_g, &rules, rng, s);
    }
    let mut target_explanation = Vec::new();
    query_g.render_in_alphabet(query_g.root, &alphabet, 0, &mut target_explanation, &mut visited);

    PtgrsEvalCase {
        observed_e,
        cause: HiddenCause {
            rules,
            seed_graph,
            alphabet,
            num_steps: 4,
        },
        target_explanation,
        ood,
    }
}

/// Équivalence observationnelle entre deux causes cachées (pas égalité de graphes !) :
/// Compare les signatures canoniques de réduction sur un ensemble de graphes-sondes déterministes.
pub fn observational_equivalence(c1: &HiddenCause, c2: &HiddenCause) -> f64 {
    let mut probe_rng1 = Rng::new(0xA1B2_C3D4_E5F6_0001);
    let mut probe_rng2 = Rng::new(0xA1B2_C3D4_E5F6_0001);
    let probes = 12;
    let mut matches = 0.0;
    let mut visited = Vec::new();

    for p in 0..probes {
        let ops = c1.alphabet.op_count().min(c2.alphabet.op_count()).max(2);
        let mut g1 = build_random_term_graph(&mut probe_rng1, ops, 5 + (p % 4));
        let mut g2 = g1.clone();
        for s in 0..3 {
            apply_rule_step(&mut g1, &c1.rules, &mut probe_rng1, s);
            apply_rule_step(&mut g2, &c2.rules, &mut probe_rng2, s);
        }
        let s1 = g1.canonical_signature(g1.root, 0, &mut visited);
        visited.clear();
        let s2 = g2.canonical_signature(g2.root, 0, &mut visited);
        visited.clear();
        if s1 == s2 {
            matches += 1.0;
        } else {
            // Proximité structurelle (opérateurs racines et arité)
            let n1 = &g1.nodes[g1.root];
            let n2 = &g2.nodes[g2.root];
            if n1.op == n2.op && n1.children.len() == n2.children.len() {
                matches += 0.35;
            }
        }
    }
    matches / (probes as f64)
}

/// Mesure post-hoc de qualité de cause retrouvée à partir d'une distribution relationnelle
/// estimée sur E, confrontée aux transitions réelles produites par la cause cachée C.
pub fn cause_recovery_score(
    eval_case: &PtgrsEvalCase,
    predicted_transition_matrix: &[f64],
    unique_symbols: &[u8],
) -> f64 {
    let k = unique_symbols.len();
    if k < 2 || predicted_transition_matrix.len() != k * k {
        return 0.0;
    }

    // On exécute C sur une sonde fraîche pour obtenir les transitions causales vraies dans l'alphabet de C
    let mut probe_rng = Rng::new(0xDEAD_BEEF_1234_5678);
    let mut g = eval_case.cause.seed_graph.clone();
    let mut true_seq = Vec::new();
    let mut visited = Vec::new();
    for s in 0..6 {
        apply_rule_step(&mut g, &eval_case.cause.rules, &mut probe_rng, s);
        g.render_in_alphabet(g.root, &eval_case.cause.alphabet, 0, &mut true_seq, &mut visited);
        visited.clear();
        true_seq.push(eval_case.cause.alphabet.clause_end());
    }

    let mut byte_to_idx = [usize::MAX; 256];
    for (idx, &b) in unique_symbols.iter().enumerate() {
        byte_to_idx[b as usize] = idx;
    }

    let mut hits = 0.0;
    let mut count = 0.0;
    for w in true_seq.windows(2) {
        let i = byte_to_idx[w[0] as usize];
        let j = byte_to_idx[w[1] as usize];
        if i < k && j < k {
            let p = predicted_transition_matrix[i * k + j].clamp(0.0, 1.0);
            // Comparé au hasard uniforme 1/k
            let lift = (p * (k as f64)).clamp(0.0, 4.0) / 4.0;
            hits += lift;
            count += 1.0;
        }
    }
    if count < 1.0 { 0.0 } else { (hits / count).clamp(0.0, 1.0) }
}