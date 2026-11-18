use crate::rng::Rng;
use crate::synth::codec::{BlockKind, RandomCodec, SpanMeta};
use crate::synth::ptgrs::{
    DerivationResult, PatNodeKind, PatSymbol, PatternGraph, RewriteRule, Signature, SymbolId,
    TermGraph, run_ptgrs, structural_similarity,
};
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct EpisodeConfig {
    pub num_symbols: usize,
    pub num_rules: usize,
    pub max_rewrite_steps: usize,
    pub max_graph_nodes: usize,
    pub max_target_bytes: usize,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            num_symbols: 12,
            num_rules: 4,
            max_rewrite_steps: 6,
            max_graph_nodes: 48,
            max_target_bytes: 48,
        }
    }
}

#[derive(Clone, Debug)]
enum LogicalBlock {
    Rule { id: usize, rule: RewriteRule },
    State { id: usize, graph: TermGraph },
    Query { id: usize, probe: TermGraph },
}

impl LogicalBlock {
    fn id(&self) -> usize {
        match self {
            LogicalBlock::Rule { id, .. }
            | LogicalBlock::State { id, .. }
            | LogicalBlock::Query { id, .. } => *id,
        }
    }

    fn kind(&self) -> BlockKind {
        match self {
            LogicalBlock::Rule { .. } => BlockKind::RuleSpec,
            LogicalBlock::State { .. } => BlockKind::StateSnapshot,
            LogicalBlock::Query { .. } => BlockKind::Query,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Episode {
    pub bytes_v1: Vec<u8>,
    pub bytes_v2: Vec<u8>,
    pub bytes_do: Vec<u8>,
    pub spans_v1: Vec<SpanMeta>,
    pub spans_v2: Vec<SpanMeta>,
    pub num_blocks: usize,
    /// Matrice B x B d'homologie structurelle / causale issue du PTGRS (Pilier 3)
    pub homology_matrix: Vec<f32>,
    /// Longueur B : 1.0 si le bloc est causalement affecté par l'intervention do (Pilier 2)
    pub do_affected_blocks: Vec<f32>,
    /// Segment causal à fermer / re-dériver sans fuite (Pilier 4 - test de fermeture)
    pub closure_span_v1: (usize, usize),
    pub closure_target_bytes: Vec<u8>,
    /// Effet intentionnel / réponse à la requête, exprimé strictement dans le codec de E1
    pub target_bytes: Vec<u8>,
    /// Masque booléen [256] indiquant l'alphabet d'octets présent dans E1
    pub alphabet_mask_v1: Vec<f32>,
}

fn pattern_as_representative_graph(pat: &PatternGraph, sig: &Signature, pool: &[SymbolId]) -> TermGraph {
    let leaves: Vec<SymbolId> = pool.iter().copied().filter(|&s| sig.arity(s) == 0).collect();
    let mut nodes = Vec::new();
    let mut var_memo: std::collections::HashMap<usize, crate::synth::ptgrs::NodeId> =
        std::collections::HashMap::new();

    fn build(
        pat: &PatternGraph,
        idx: usize,
        sig: &Signature,
        pool: &[SymbolId],
        leaves: &[SymbolId],
        nodes: &mut Vec<crate::synth::ptgrs::TermNode>,
        var_memo: &mut std::collections::HashMap<usize, crate::synth::ptgrs::NodeId>,
    ) -> crate::synth::ptgrs::NodeId {
        match &pat.nodes[idx] {
            PatNodeKind::Var(v) => {
                if let Some(&existing) = var_memo.get(v) {
                    return existing;
                }
                let sym = leaves[v % leaves.len()];
                let id = crate::synth::ptgrs::NodeId(nodes.len());
                nodes.push(crate::synth::ptgrs::TermNode {
                    sym,
                    args: Vec::new(),
                });
                var_memo.insert(*v, id);
                id
            }
            PatNodeKind::App { sym, args } => {
                let csym = match sym {
                    PatSymbol::Concrete(s) => *s,
                    PatSymbol::SymVar(sv) => {
                        let matching: Vec<SymbolId> = pool
                            .iter()
                            .copied()
                            .filter(|&s| sig.arity(s) == args.len())
                            .collect();
                        if matching.is_empty() {
                            leaves[0]
                        } else {
                            matching[sv % matching.len()]
                        }
                    }
                };
                let mut cargs = Vec::with_capacity(args.len());
                for &a in args {
                    cargs.push(build(pat, a, sig, pool, leaves, nodes, var_memo));
                }
                let id = crate::synth::ptgrs::NodeId(nodes.len());
                nodes.push(crate::synth::ptgrs::TermNode { sym: csym, args: cargs });
                id
            }
        }
    }

    let root = build(pat, pat.root, sig, pool, &leaves, &mut nodes, &mut var_memo);
    TermGraph { nodes, root }
}

fn serialize_blocks(
    codec: &RandomCodec,
    blocks: &[LogicalBlock],
) -> (Vec<u8>, Vec<SpanMeta>) {
    let mut bytes = Vec::new();
    let mut spans = Vec::with_capacity(blocks.len());
    for blk in blocks {
        let start = bytes.len();
        match blk {
            LogicalBlock::Rule { rule, .. } => codec.encode_rule(rule, &mut bytes),
            LogicalBlock::State { graph, .. } => codec.encode_state_block(graph, &mut bytes),
            LogicalBlock::Query { probe, .. } => codec.encode_query_block(probe, &mut bytes),
        }
        let end = bytes.len();
        spans.push(SpanMeta {
            block_id: blk.id(),
            kind: blk.kind(),
            start,
            end,
        });
    }
    (bytes, spans)
}

pub fn sample_episode(rng: &mut Rng, cfg: &EpisodeConfig) -> Episode {
    // Boucle de rejet légère garantissant que la requête produit une dérivation non triviale
    // et que tous les octets cibles existent dans l'alphabet de E1.
    for _attempt in 0..32 {
        let sig = Signature::sample(rng, cfg.num_symbols);
        let half = cfg.num_symbols / 2;
        let pool_a: Vec<SymbolId> = (0..half).map(SymbolId).collect();
        let pool_b: Vec<SymbolId> = (half..cfg.num_symbols).map(SymbolId).collect();

        // Générer règles sur les deux sous-alphabets (y compris règles polymorphes structurelles)
        let mut rules = Vec::with_capacity(cfg.num_rules);
        for r_id in 0..cfg.num_rules {
            let pool = if r_id % 2 == 0 { &pool_a } else { &pool_b };
            let polymorphic = rng.bernoulli(0.5);
            let (lhs, rhs) = PatternGraph::sample_rule_pair(rng, &sig, pool, polymorphic);
            rules.push(RewriteRule {
                id: r_id,
                lhs,
                rhs,
                weight: rng.uniform_f32(0.5, 2.0),
                enabled: true,
            });
        }

        // Graphe observé principal (pool_a ou mixte) et graphe sonde pour la requête
        let g_obs = TermGraph::sample_random(rng, &sig, &pool_a, 2, true);
        let g_probe = TermGraph::sample_random(rng, &sig, &pool_b, 2, true);

        let ptgrs_seed = rng.next_u64();
        let mut r_obs = Rng::from_seed(ptgrs_seed);
        let deriv_obs: DerivationResult = run_ptgrs(
            &mut r_obs,
            &sig,
            &g_obs,
            &rules,
            cfg.max_rewrite_steps,
            cfg.max_graph_nodes,
        );

        let mut r_probe = Rng::from_seed(ptgrs_seed ^ 0x9e37_79b9_7f4a_7c15);
        let deriv_probe: DerivationResult = run_ptgrs(
            &mut r_probe,
            &sig,
            &g_probe,
            &rules,
            cfg.max_rewrite_steps,
            cfg.max_graph_nodes,
        );

        // Construire les blocs logiques de l'épisode (monde fermé)
        let mut blocks: Vec<LogicalBlock> = Vec::new();
        for rule in &rules {
            let id = blocks.len();
            blocks.push(LogicalBlock::Rule {
                id,
                rule: rule.clone(),
            });
        }
        let obs_init_id = blocks.len();
        blocks.push(LogicalBlock::State {
            id: obs_init_id,
            graph: deriv_obs.initial_graph.clone(),
        });
        let obs_final_id = blocks.len();
        blocks.push(LogicalBlock::State {
            id: obs_final_id,
            graph: deriv_obs.final_graph.clone(),
        });
        // Blocs distracteurs structurels indépendants (apprend à filtrer un contexte riche)
        let num_distractors = rng.range_usize(0, 4);
        for d_idx in 0..num_distractors {
            let d_pool = if d_idx % 2 == 0 { &pool_a } else { &pool_b };
            let g_dist = TermGraph::sample_random(rng, &sig, d_pool, 2, true);
            let did = blocks.len();
            blocks.push(LogicalBlock::State {
                id: did,
                graph: g_dist,
            });
        }
        let query_id = blocks.len();
        blocks.push(LogicalBlock::Query {
            id: query_id,
            probe: deriv_probe.initial_graph.clone(),
        });
        let num_blocks = blocks.len();

        // Intervention contrefactuelle do() : désactiver ou perturber une règle ayant tiré (ou règle 0)
        let intervened_rule_id = if let Some(first_step) = deriv_obs.steps.first() {
            first_step.rule_id
        } else if let Some(first_p) = deriv_probe.steps.first() {
            first_p.rule_id
        } else {
            0
        };

        let mut rules_do = rules.clone();
        rules_do[intervened_rule_id].enabled = false;
        let mut r_obs_do = Rng::from_seed(ptgrs_seed);
        let deriv_obs_do = run_ptgrs(
            &mut r_obs_do,
            &sig,
            &g_obs,
            &rules_do,
            cfg.max_rewrite_steps,
            cfg.max_graph_nodes,
        );
        let mut r_probe_do = Rng::from_seed(ptgrs_seed ^ 0x9e37_79b9_7f4a_7c15);
        let deriv_probe_do = run_ptgrs(
            &mut r_probe_do,
            &sig,
            &g_probe,
            &rules_do,
            cfg.max_rewrite_steps,
            cfg.max_graph_nodes,
        );

        let mut blocks_do = blocks.clone();
        for blk in &mut blocks_do {
            match blk {
                LogicalBlock::Rule { rule, .. } if rule.id == intervened_rule_id => {
                    rule.enabled = false;
                    // Pour rendre le changement visible dans E_do de l'axiome perturbé :
                    rule.rhs = rule.lhs.clone();
                }
                LogicalBlock::State { id, graph } if *id == obs_final_id => {
                    *graph = deriv_obs_do.final_graph.clone();
                }
                _ => {}
            }
        }

        // Vecteur causalement affecté par do() : l'axiome perturbé + tout état dérivé qui change
        let mut do_affected = vec![0.0f32; num_blocks];
        do_affected[intervened_rule_id] = 1.0;
        let obs_changed = deriv_obs_do.final_graph.nodes.len() != deriv_obs.final_graph.nodes.len()
            || deriv_obs_do
                .final_graph
                .nodes
                .iter()
                .zip(deriv_obs.final_graph.nodes.iter())
                .any(|(a, b)| a.sym != b.sym || a.args != b.args);
        if obs_changed {
            do_affected[obs_final_id] = 1.0;
        }
        let probe_changed = deriv_probe_do.final_graph.nodes.len() != deriv_probe.final_graph.nodes.len()
            || deriv_probe_do
                .final_graph
                .nodes
                .iter()
                .zip(deriv_probe.final_graph.nodes.iter())
                .any(|(a, b)| a.sym != b.sym || a.args != b.args);
        if probe_changed {
            do_affected[query_id] = 1.0;
        }

        // Matrice d'homologie vraie M_hom (B x B) purement issue du PTGRS :
        // (1) liens causaux dans les dérivations tau, (2) homomorphisme structurel pur alpha-invariant
        let full_pool: Vec<SymbolId> = (0..cfg.num_symbols).map(SymbolId).collect();
        let rep_graphs: Vec<TermGraph> = blocks
            .iter()
            .map(|b| match b {
                LogicalBlock::Rule { rule, .. } => {
                    pattern_as_representative_graph(&rule.lhs, &sig, &full_pool)
                }
                LogicalBlock::State { graph, .. } => graph.clone(),
                LogicalBlock::Query { probe, .. } => probe.clone(),
            })
            .collect();

        let mut fired_obs: HashSet<usize> = HashSet::new();
        for st in &deriv_obs.steps {
            fired_obs.insert(st.rule_id);
        }
        let mut fired_probe: HashSet<usize> = HashSet::new();
        for st in &deriv_probe.steps {
            fired_probe.insert(st.rule_id);
        }

        let mut homology = vec![0.0f32; num_blocks * num_blocks];
        for i in 0..num_blocks {
            homology[i * num_blocks + i] = 1.0;
            for j in (i + 1)..num_blocks {
                let mut score = structural_similarity(&rep_graphs[i], &rep_graphs[j]);
                // Liens causaux avérés par la trace tau
                let is_causal_pair = (i == obs_init_id && j == obs_final_id)
                    || (i < rules.len() && fired_obs.contains(&i) && (j == obs_init_id || j == obs_final_id))
                    || (i < rules.len() && fired_probe.contains(&i) && j == query_id);
                if is_causal_pair {
                    score = score.max(1.0);
                }
                homology[i * num_blocks + j] = score;
                homology[j * num_blocks + i] = score;
            }
        }

        let mut max_var_plus1 = 16usize;
        let mut max_sym_var_plus1 = 16usize;
        for r in &rules {
            for pat in [&r.lhs, &r.rhs] {
                for node in &pat.nodes {
                    match node {
                        PatNodeKind::Var(v) => {
                            max_var_plus1 = max_var_plus1.max(*v + 1);
                        }
                        PatNodeKind::App {
                            sym: PatSymbol::SymVar(sv),
                            ..
                        } => {
                            max_sym_var_plus1 = max_sym_var_plus1.max(*sv + 1);
                        }
                        _ => {}
                    }
                }
            }
        }

        let c1 = RandomCodec::sample(rng, cfg.num_symbols, max_var_plus1, max_sym_var_plus1, true);
        let c2 = RandomCodec::sample(rng, cfg.num_symbols, max_var_plus1, max_sym_var_plus1, true);

        let mut blocks_v1 = blocks.clone();
        // Permuter l'ordre des blocs non-requête dans V1 pour éviter tout biais positionnel fixe
        let split_q = blocks_v1.len() - 1;
        rng.shuffle(&mut blocks_v1[..split_q]);

        let mut blocks_v2 = blocks.clone();
        rng.shuffle(&mut blocks_v2);

        let (bytes_v1, spans_v1) = serialize_blocks(&c1, &blocks_v1);
        let (bytes_v2, spans_v2) = serialize_blocks(&c2, &blocks_v2);
        let (bytes_do, _) = serialize_blocks(&c1, &blocks_do);

        // Cible téléologique Y* dans le codec C1
        let mut target_bytes = Vec::new();
        c1.encode_term_graph(&deriv_probe.final_graph, &mut target_bytes);
        if target_bytes.len() > cfg.max_target_bytes {
            target_bytes.truncate(cfg.max_target_bytes);
        }

        // Vérifier que tous les octets de target_bytes appartiennent bien à Alphabet(E1)
        let mut present = [false; 256];
        for &b in &bytes_v1 {
            present[b as usize] = true;
        }
        let target_in_alphabet = !target_bytes.is_empty() && target_bytes.iter().all(|&b| present[b as usize]);
        if !target_in_alphabet {
            continue;
        }

        let alphabet_mask_v1: Vec<f32> = present.iter().map(|&p| if p { 1.0 } else { 0.0 }).collect();

        // Span de fermeture causale : l'état final dérivé obs_final_id dans E1
        let closure_span = spans_v1
            .iter()
            .find(|s| s.block_id == obs_final_id)
            .map(|s| (s.start, s.end))
            .unwrap_or((0, bytes_v1.len().min(8)));
        let closure_target_bytes = bytes_v1[closure_span.0..closure_span.1].to_vec();

        return Episode {
            bytes_v1,
            bytes_v2,
            bytes_do,
            spans_v1,
            spans_v2,
            num_blocks,
            homology_matrix: homology,
            do_affected_blocks: do_affected,
            closure_span_v1: closure_span,
            closure_target_bytes,
            target_bytes,
            alphabet_mask_v1,
        };
    }

    // Fallback déterministe garanti valide si 32 tentatives n'avaient pas clos l'alphabet
    let sig_ = Signature {
        arities: vec![0, 0, 1, 1],
    };
    let rule = RewriteRule {
        id: 0,
        lhs: PatternGraph {
            nodes: vec![
                PatNodeKind::Var(0),
                PatNodeKind::App {
                    sym: PatSymbol::Concrete(SymbolId(2)),
                    args: vec![0],
                },
            ],
            root: 1,
        },
        rhs: PatternGraph {
            nodes: vec![PatNodeKind::Var(0)],
            root: 0,
        },
        weight: 1.0,
        enabled: true,
    };
    let mut g_obs = TermGraph::new_empty(SymbolId(0));
    let n0 = NodeId0_leaf(&mut g_obs, SymbolId(0));
    g_obs.root = g_obs.add_node(SymbolId(2), vec![n0]);
    let g_probe = g_obs.clone();

    let blocks = vec![
        LogicalBlock::Rule {
            id: 0,
            rule: rule.clone(),
        },
        LogicalBlock::State {
            id: 1,
            graph: g_obs.clone(),
        },
        LogicalBlock::State {
            id: 2,
            graph: TermGraph::new_empty(SymbolId(0)),
        },
        LogicalBlock::Query {
            id: 3,
            probe: g_probe,
        },
    ];
    let c1 = RandomCodec::sample(rng, 4, 4, 4, false);
    let c2 = RandomCodec::sample(rng, 4, 4, 4, false);
    let (bytes_v1, spans_v1) = serialize_blocks(&c1, &blocks);
    let (bytes_v2, spans_v2) = serialize_blocks(&c2, &blocks);

    let mut target_bytes = Vec::new();
    c1.encode_term_graph(&TermGraph::new_empty(SymbolId(0)), &mut target_bytes);

    let mut present = [false; 256];
    for &b in &bytes_v1 {
        present[b as usize] = true;
    }
    let alphabet_mask_v1: Vec<f32> = present.iter().map(|&p| if p { 1.0 } else { 0.0 }).collect();
    let closure_span_v1 = (spans_v1[2].start, spans_v1[2].end);
    let closure_target_bytes = bytes_v1[closure_span_v1.0..closure_span_v1.1].to_vec();

    Episode {
        bytes_v1: bytes_v1.clone(),
        bytes_v2,
        bytes_do: bytes_v1,
        spans_v1,
        spans_v2,
        num_blocks: 4,
        homology_matrix: vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        do_affected_blocks: vec![1.0, 0.0, 1.0, 1.0],
        closure_span_v1,
        closure_target_bytes,
        target_bytes,
        alphabet_mask_v1,
    }
}

#[allow(non_snake_case)]
fn NodeId0_leaf(g: &mut TermGraph, sym: SymbolId) -> crate::synth::ptgrs::NodeId {
    g.add_node(sym, Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn episode_satisfies_closed_alphabet_and_four_pillars() {
        let mut rng = Rng::from_seed(314159);
        let cfg = EpisodeConfig::default();
        for _ in 0..20 {
            let ep = sample_episode(&mut rng, &cfg);
            assert!(!ep.bytes_v1.is_empty());
            assert!(!ep.bytes_v2.is_empty());
            assert!(!ep.bytes_do.is_empty());
            assert_eq!(ep.homology_matrix.len(), ep.num_blocks * ep.num_blocks);
            assert_eq!(ep.do_affected_blocks.len(), ep.num_blocks);
            for &b in &ep.target_bytes {
                assert_eq!(
                    ep.alphabet_mask_v1[b as usize], 1.0,
                    "Target byte {} absent de Alphabet(E1)!",
                    b
                );
            }
        }
    }
}