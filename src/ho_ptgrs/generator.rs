use codec::EpisodicCodec;
use meta_ops::{build_def_graph, evaluate_meta_op, Instance, MetaOp};
use rewriting::{PatternGraph, Rule};
use rigidity::{verify_rigidity_2fwl, RigidityReport};
use rng::Rng;
use term_graph::{SymbolId, TermGraph};

use std::collections::BTreeMap;

/// Paramètres de curriculum du générateur HO-PTGRS : (k, D, r_bruit).
#[derive(Debug, Clone)]
pub struct CurriculumConfig {
    /// Horizon de réification k (minimum 3 pour supporter cycles, bifurcations et chaînes).
    pub horizon_k: usize,
    /// Nombre de distracteurs D >= 3 dans G_def.
    pub num_distractors: usize,
    /// Ratio de règles de bruit structurel injectées dans \Sigma_inst.
    pub noise_ratio: f64,
    /// Limite d'essais de rejection sampling.
    pub max_attempts: usize,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            horizon_k: 4,
            num_distractors: 4,
            noise_ratio: 0.5,
            max_attempts: 64,
        }
    }
}

/// Cause cachée H* (utilisée UNIQUEMENT pour l'évaluation post-hoc, jamais fournie à Phisharp).
#[derive(Debug, Clone)]
pub struct HiddenCause {
    pub target_op: MetaOp,
    pub distractors: Vec<MetaOp>,
    /// Mot de code d'octets du concept cible v_{\mu^*} dans perceived.eff.
    pub target_concept_bytes: Vec<u8>,
}

/// Statistiques de diagnostic de l'épisode généré.
#[derive(Debug, Clone)]
pub struct EpisodeStats {
    pub total_bytes: usize,
    pub def_nodes: usize,
    pub def_edges: usize,
    pub inst_roots: usize,
    pub inst_rules_total: usize,
    pub noise_rules_added: usize,
    pub codec_invariants_ok: bool,
}

/// Épisode complet produit par le HO-PTGRS.
#[derive(Debug, Clone)]
pub struct Episode {
    /// Contenu exact de `perceived.eff` (seule entrée visible par Phisharp).
    pub perceived_eff: Vec<u8>,
    pub hidden_cause: HiddenCause,
    pub rigidity: RigidityReport,
    pub stats: EpisodeStats,
}

/// Allocateur de symboles strictement dans \Sigma_inst (>= 10_000, disjoint de \Sigma_def).
struct InstSymAlloc {
    next_id: u32,
    arities: BTreeMap<SymbolId, usize>,
    next_rule_id: u32,
}

impl InstSymAlloc {
    fn new() -> Self {
        Self {
            next_id: 10_000,
            arities: BTreeMap::new(),
            next_rule_id: 1,
        }
    }

    fn fresh(&mut self, arity: usize) -> SymbolId {
        let s = SymbolId(self.next_id);
        self.next_id += 1;
        self.arities.insert(s, arity);
        s
    }

    fn rule_id(&mut self) -> u32 {
        let id = self.next_rule_id;
        self.next_rule_id += 1;
        id
    }
}

/// Générateur principal d'épisodes HO-PTGRS.
pub struct HoPtgrsGenerator {
    pub rng: Rng,
    pub config: CurriculumConfig,
}

impl HoPtgrsGenerator {
    pub fn new(seed: u64, config: CurriculumConfig) -> Self {
        Self {
            rng: Rng::from_seed(seed),
            config,
        }
    }

    /// Génère un épisode vérifiant les Théorèmes A et B (identifiabilité + rigidité + jauge).
    pub fn generate_episode(&mut self, forced_target: Option<MetaOp>) -> Result<Episode, String> {
        let k = self.config.horizon_k.max(3);
        let catalog = MetaOp::catalog();
        let d_count = self.config.num_distractors.clamp(1, catalog.len() - 1);

        for _attempt in 0..self.config.max_attempts {
            // 1. Tirer \mu* et D distracteurs distincts
            let target = forced_target.clone().unwrap_or_else(|| {
                let idx = self.rng.gen_range(catalog.len());
                catalog[idx].clone()
            });
            let mut others: Vec<MetaOp> = catalog
                .iter()
                .filter(|op| **op != target)
                .cloned()
                .collect();
            self.rng.shuffle(&mut others);
            let distractors: Vec<MetaOp> = others.into_iter().take(d_count).collect();

            // 2. Construire le dictionnaire G_def et vérifier la rigidité 2-FWL (Théorème B.1)
            let mut all_ops = vec![target.clone()];
            all_ops.extend(distractors.iter().cloned());
            self.rng.shuffle(&mut all_ops);

            let def_graph = build_def_graph(&all_ops);
            let rigidity = verify_rigidity_2fwl(&def_graph);
            if !rigidity.is_rigid {
                continue;
            }

            // 3. Synthétiser (G_0, R_0) sur \Sigma_inst pour \mu*
            let mut alloc = InstSymAlloc::new();
            let mut inst = synthesize_instance(&mut self.rng, &mut alloc, &target, k);

            // Vérifier séparabilité exacte à horizon k (Théorème B.2)
            if !evaluate_meta_op(&target, &inst, k) {
                continue;
            }
            if distractors.iter().any(|d| evaluate_meta_op(d, &inst, k)) {
                continue;
            }

            // 4. Injecter le bruit structurel en préservant la valeur de vérité de tous les opérateurs
            let noise_added =
                inject_verified_noise(&mut self.rng, &mut alloc, &mut inst, &target, &distractors, k, self.config.noise_ratio);

            // 5. Échantillonner le codec épisodique aléatoire invariant sous S_256 (Théorème A)
            let def_syms = def_graph.all_symbols();
            let inst_syms = inst.all_symbols();
            assert!(
                def_syms.is_disjoint(&inst_syms),
                "Violation de partition stricte Sigma_def / Sigma_inst"
            );

            let max_inst_idx = compute_max_inst_index(&inst);
            let codec = EpisodicCodec::sample(
                &mut self.rng,
                &def_syms,
                &inst_syms,
                def_graph.node_symbols.len(),
                max_inst_idx,
            );
            let codec_ok = codec.verify_invariants();
            if !codec_ok {
                continue;
            }

            // 6. Sérialiser en octets `perceived.eff`
            let perceived_eff = codec.serialize(&mut self.rng, &def_graph, &inst);

            let target_node = def_graph.concept_nodes[&target];
            let target_sym = def_graph.node_symbols[target_node];
            let target_concept_bytes = codec.def_code(target_sym).unwrap().to_vec();

            let stats = EpisodeStats {
                total_bytes: perceived_eff.len(),
                def_nodes: def_graph.node_symbols.len(),
                def_edges: def_graph.edges.len(),
                inst_roots: inst.roots.len(),
                inst_rules_total: inst.rules.len(),
                noise_rules_added: noise_added,
                codec_invariants_ok: codec_ok,
            };

            return Ok(Episode {
                perceived_eff,
                hidden_cause: HiddenCause {
                    target_op: target,
                    distractors,
                    target_concept_bytes,
                },
                rigidity,
                stats,
            });
        }

        Err("Échec du rejection sampling dans max_attempts".to_string())
    }
}

fn compute_max_inst_index(inst: &Instance) -> usize {
    let mut m = 4usize;
    for g in &inst.roots {
        m = m.max(g.nodes.len() + 1);
    }
    for r in &inst.rules {
        m = m.max(r.lhs.nodes.len() + 1).max(r.rhs.nodes.len() + 1);
        for n in r.lhs.nodes.iter().chain(r.rhs.nodes.iter()) {
            if let crate::rewriting::PatternNode::Var(vid) = n {
                m = m.max(*vid as usize + 1);
            }
        }
    }
    m
}

// ============================================================================
// Synthétiseurs stochastiques pour les 6 opérateurs de la base canonique
// ============================================================================

fn unary_rule(alloc: &mut InstSymAlloc, from: SymbolId, to: SymbolId, prob: f64) -> Rule {
    let mut lb = PatternGraph::builder();
    let v = lb.var(0);
    let lr = lb.sym(from, vec![v]);
    let mut rb = PatternGraph::builder();
    let v2 = rb.var(0);
    let rr = rb.sym(to, vec![v2]);
    Rule {
        id: alloc.rule_id(),
        lhs: lb.finish(lr),
        rhs: rb.finish(rr),
        prob,
    }
}

fn unary_term(head: SymbolId, leaf: SymbolId) -> TermGraph {
    let mut b = TermGraph::builder();
    let l = b.add(leaf, vec![]);
    let r = b.add(head, vec![l]);
    b.finish(r)
}

fn synthesize_instance(
    rng: &mut Rng,
    alloc: &mut InstSymAlloc,
    op: &MetaOp,
    horizon_k: usize,
) -> Instance {
    match op {
        MetaOp::Bif => synth_bif(rng, alloc, horizon_k),
        MetaOp::Fix => synth_fix(rng, alloc, horizon_k),
        MetaOp::Bot => synth_bot(rng, alloc, horizon_k),
        MetaOp::Iso => synth_iso(rng, alloc, horizon_k),
        MetaOp::Quot => synth_quot(rng, alloc, horizon_k),
        MetaOp::Comp(a, b) => synth_comp(rng, alloc, a, b, horizon_k),
    }
}

/// \mu_bif : chaîne préfixe aléatoire suivie d'une bifurcation vers >= 2 formes normales distinctes.
fn synth_bif(rng: &mut Rng, alloc: &mut InstSymAlloc, k: usize) -> Instance {
    let leaf = alloc.fresh(0);
    let pre_len = rng.gen_range_inclusive(0, k.saturating_sub(2).min(2));
    let mut rules = Vec::new();

    let mut curr = alloc.fresh(1);
    let start = curr;
    for _ in 0..pre_len {
        let next = alloc.fresh(1);
        rules.push(unary_rule(alloc, curr, next, 1.0));
        curr = next;
    }

    let branches = rng.gen_range_inclusive(2, 3);
    for _ in 0..branches {
        let b_mid = alloc.fresh(1);
        rules.push(unary_rule(alloc, curr, b_mid, 1.0 / branches as f64));
        if pre_len + 2 <= k && rng.gen_bool(0.5) {
            let b_end = alloc.fresh(1);
            rules.push(unary_rule(alloc, b_mid, b_end, 1.0));
        }
    }

    Instance {
        roots: vec![unary_term(start, leaf)],
        rules,
        arities: alloc.arities.clone(),
    }
}

/// \mu_fix : entrée aléatoire dans un cycle dirigé de longueur c in [2, k] sans forme normale.
fn synth_fix(rng: &mut Rng, alloc: &mut InstSymAlloc, k: usize) -> Instance {
    let leaf = alloc.fresh(0);
    let cycle_len = rng.gen_range_inclusive(2, k.min(4));
    let pre_len = rng.gen_range_inclusive(0, k.saturating_sub(cycle_len).min(1));
    let mut rules = Vec::new();

    let start = alloc.fresh(1);
    let mut curr = start;
    for _ in 0..pre_len {
        let next = alloc.fresh(1);
        rules.push(unary_rule(alloc, curr, next, 1.0));
        curr = next;
    }

    let cycle_entry = curr;
    for _ in 1..cycle_len {
        let next = alloc.fresh(1);
        rules.push(unary_rule(alloc, curr, next, 1.0));
        curr = next;
    }
    rules.push(unary_rule(alloc, curr, cycle_entry, 1.0));

    Instance {
        roots: vec![unary_term(start, leaf)],
        rules,
        arities: alloc.arities.clone(),
    }
}

/// \mu_bot : deux sous-termes en conflit produisant un absorbeur structurel \bot spécifié par R_0.
fn synth_bot(rng: &mut Rng, alloc: &mut InstSymAlloc, k: usize) -> Instance {
    let z = alloc.fresh(0); // Absorbeur \bot
    let f = alloc.fresh(2); // Constructeur binaire
    let a = alloc.fresh(0);
    let b = alloc.fresh(0);
    let mut rules = Vec::new();

    // Règles d'absorption auto-spécifiées : f(z, x) -> z et f(x, z) -> z
    {
        let mut lb = PatternGraph::builder();
        let pz = lb.sym(z, vec![]);
        let px = lb.var(0);
        let lr = lb.sym(f, vec![pz, px]);
        let mut rb = PatternGraph::builder();
        let rr = rb.sym(z, vec![]);
        rules.push(Rule {
            id: alloc.rule_id(),
            lhs: lb.finish(lr),
            rhs: rb.finish(rr),
            prob: 1.0,
        });
    }
    {
        let mut lb = PatternGraph::builder();
        let px = lb.var(0);
        let pz = lb.sym(z, vec![]);
        let lr = lb.sym(f, vec![px, pz]);
        let mut rb = PatternGraph::builder();
        let rr = rb.sym(z, vec![]);
        rules.push(Rule {
            id: alloc.rule_id(),
            lhs: lb.finish(lr),
            rhs: rb.finish(rr),
            prob: 1.0,
        });
    }

    // Règle de contradiction : f(a, b) -> z
    {
        let mut lb = PatternGraph::builder();
        let pa = lb.sym(a, vec![]);
        let pb = lb.sym(b, vec![]);
        let lr = lb.sym(f, vec![pa, pb]);
        let mut rb = PatternGraph::builder();
        let rr = rb.sym(z, vec![]);
        rules.push(Rule {
            id: alloc.rule_id(),
            lhs: lb.finish(lr),
            rhs: rb.finish(rr),
            prob: 1.0,
        });
    }

    // Dérivation préliminaire optionnelle sur les opérandes dans le budget k
    let mut left_sym = a;
    if k >= 3 && rng.gen_bool(0.7) {
        let a0 = alloc.fresh(0);
        let mut lb = PatternGraph::builder();
        let lr = lb.sym(a0, vec![]);
        let mut rb = PatternGraph::builder();
        let rr = rb.sym(a, vec![]);
        rules.push(Rule {
            id: alloc.rule_id(),
            lhs: lb.finish(lr),
            rhs: rb.finish(rr),
            prob: 1.0,
        });
        left_sym = a0;
    }

    let mut tb = TermGraph::builder();
    let nl = tb.add(left_sym, vec![]);
    let nr = tb.add(b, vec![]);
    let root = tb.add(f, vec![nl, nr]);

    Instance {
        roots: vec![tb.finish(root)],
        rules,
        arities: alloc.arities.clone(),
    }
}

/// \mu_iso : deux domaines strictement disjoints instanciant le même squelette confluent (diamant ou chaîne).
fn synth_iso(rng: &mut Rng, alloc: &mut InstSymAlloc, k: usize) -> Instance {
    let use_diamond = rng.gen_bool(0.6);
    let chain_len = rng.gen_range_inclusive(2, k.min(4));
    let mut roots = Vec::new();
    let mut rules = Vec::new();

    for _domain in 0..2 {
        let leaf = alloc.fresh(0);
        if use_diamond {
            let s0 = alloc.fresh(1);
            let s1 = alloc.fresh(1);
            let s2 = alloc.fresh(1);
            let s3 = alloc.fresh(1);
            rules.push(unary_rule(alloc, s0, s1, 0.5));
            rules.push(unary_rule(alloc, s0, s2, 0.5));
            rules.push(unary_rule(alloc, s1, s3, 1.0));
            rules.push(unary_rule(alloc, s2, s3, 1.0));
            roots.push(unary_term(s0, leaf));
        } else {
            let start = alloc.fresh(1);
            let mut curr = start;
            for _ in 0..chain_len {
                let next = alloc.fresh(1);
                rules.push(unary_rule(alloc, curr, next, 1.0));
                curr = next;
            }
            roots.push(unary_term(start, leaf));
        }
    }

    Instance {
        roots,
        rules,
        arities: alloc.arities.clone(),
    }
}

/// \mu_quot : >= 3 termes à dérivations non isomorphes convergeant vers des instances d'un moule commun M.
fn synth_quot(rng: &mut Rng, alloc: &mut InstSymAlloc, k: usize) -> Instance {
    // Moule commun M = Outer(Inner(Hole, Cst), Cst) -> 2 constructeurs internes + 1 trou
    let outer = alloc.fresh(2);
    let inner = alloc.fresh(2);
    let cst = alloc.fresh(0);

    let m = rng.gen_range_inclusive(3, 4);
    let mut roots = Vec::new();
    let mut rules = Vec::new();

    for i in 0..m {
        let hole_val = alloc.fresh(0);
        // Forme normale cible : Outer(Inner(hole_val, cst), cst)
        let mut nf_rhs = PatternGraph::builder();
        let hv = nf_rhs.sym(hole_val, vec![]);
        let c1 = nf_rhs.sym(cst, vec![]);
        let c2 = nf_rhs.sym(cst, vec![]);
        let inn = nf_rhs.sym(inner, vec![hv, c1]);
        let out_r = nf_rhs.sym(outer, vec![inn, c2]);
        let target_rhs = nf_rhs.finish(out_r);

        // Longueur de chaîne distincte pour garantir que les D_k ne soient pas tous isomorphes
        let steps = (i % k.min(3)) + 1;
        let start = alloc.fresh(0);
        let mut curr = start;
        for step_idx in 0..steps {
            let mut lb = PatternGraph::builder();
            let lr = lb.sym(curr, vec![]);
            if step_idx + 1 == steps {
                rules.push(Rule {
                    id: alloc.rule_id(),
                    lhs: lb.finish(lr),
                    rhs: target_rhs.clone(),
                    prob: 1.0,
                });
            } else {
                let next = alloc.fresh(0);
                let mut rb = PatternGraph::builder();
                let rr = rb.sym(next, vec![]);
                rules.push(Rule {
                    id: alloc.rule_id(),
                    lhs: lb.finish(lr),
                    rhs: rb.finish(rr),
                    prob: 1.0,
                });
                curr = next;
            }
        }

        let mut tb = TermGraph::builder();
        let r = tb.add(start, vec![]);
        roots.push(tb.finish(r));
    }

    Instance {
        roots,
        rules,
        arities: alloc.arities.clone(),
    }
}

/// \mu_comp : synthèse des compositions canoniques d'ordre supérieur.
fn synth_comp(
    rng: &mut Rng,
    alloc: &mut InstSymAlloc,
    a: &MetaOp,
    b: &MetaOp,
    k: usize,
) -> Instance {
    match (a, b) {
        (MetaOp::Bif, MetaOp::Fix) | (MetaOp::Fix, MetaOp::Bif) => {
            // Bifurcation hybride : une branche normalise, l'autre boucle en cycle dirigé
            let leaf = alloc.fresh(0);
            let start = alloc.fresh(1);
            let nf_sym = alloc.fresh(1);
            let c1 = alloc.fresh(1);
            let c2 = alloc.fresh(1);
            let mut rules = vec![
                unary_rule(alloc, start, nf_sym, 0.5),
                unary_rule(alloc, start, c1, 0.5),
                unary_rule(alloc, c1, c2, 1.0),
                unary_rule(alloc, c2, c1, 1.0),
            ];
            if k >= 4 && rng.gen_bool(0.5) {
                let nf2 = alloc.fresh(1);
                rules.push(unary_rule(alloc, nf_sym, nf2, 1.0));
            }
            Instance {
                roots: vec![unary_term(start, leaf)],
                rules,
                arities: alloc.arities.clone(),
            }
        }
        (MetaOp::Iso, MetaOp::Bif) | (MetaOp::Bif, MetaOp::Iso) => {
            let mut roots = Vec::new();
            let mut rules = Vec::new();
            for _ in 0..2 {
                let leaf = alloc.fresh(0);
                let s0 = alloc.fresh(1);
                let b1 = alloc.fresh(1);
                let b2 = alloc.fresh(1);
                rules.push(unary_rule(alloc, s0, b1, 0.5));
                rules.push(unary_rule(alloc, s0, b2, 0.5));
                roots.push(unary_term(s0, leaf));
            }
            Instance {
                roots,
                rules,
                arities: alloc.arities.clone(),
            }
        }
        (MetaOp::Iso, MetaOp::Fix) | (MetaOp::Fix, MetaOp::Iso) => {
            let cycle_len = rng.gen_range_inclusive(2, k.min(3));
            let mut roots = Vec::new();
            let mut rules = Vec::new();
            for _ in 0..2 {
                let leaf = alloc.fresh(0);
                let start = alloc.fresh(1);
                let mut curr = start;
                for _ in 1..cycle_len {
                    let next = alloc.fresh(1);
                    rules.push(unary_rule(alloc, curr, next, 1.0));
                    curr = next;
                }
                rules.push(unary_rule(alloc, curr, start, 1.0));
                roots.push(unary_term(start, leaf));
            }
            Instance {
                roots,
                rules,
                arities: alloc.arities.clone(),
            }
        }
        (MetaOp::Quot, MetaOp::Bif) | (MetaOp::Bif, MetaOp::Quot) => {
            let outer = alloc.fresh(2);
            let inner = alloc.fresh(2);
            let cst = alloc.fresh(0);
            let mut roots = Vec::new();
            let mut rules = Vec::new();

            for _ in 0..3 {
                let hv = alloc.fresh(0);
                let nf1 = alloc.fresh(0);
                let nf2 = alloc.fresh(0);
                // Règles de bifurcation sur le trou spécifique de chaque instance
                for target_nf in [nf1, nf2] {
                    let mut lb = PatternGraph::builder();
                    let lr = lb.sym(hv, vec![]);
                    let mut rb = PatternGraph::builder();
                    let rr = rb.sym(target_nf, vec![]);
                    rules.push(Rule {
                        id: alloc.rule_id(),
                        lhs: lb.finish(lr),
                        rhs: rb.finish(rr),
                        prob: 0.5,
                    });
                }
                // Racine instanciant le moule commun Outer(Inner(hv, cst), cst)
                let mut tb = TermGraph::builder();
                let nh = tb.add(hv, vec![]);
                let nc1 = tb.add(cst, vec![]);
                let nc2 = tb.add(cst, vec![]);
                let ni = tb.add(inner, vec![nh, nc1]);
                let nr = tb.add(outer, vec![ni, nc2]);
                roots.push(tb.finish(nr));
            }

            Instance {
                roots,
                rules,
                arities: alloc.arities.clone(),
            }
        }
        _ => synth_bif(rng, alloc, k),
    }
}

/// Injecte des règles distractrices dans \Sigma_inst en garantissant la conservation stricte de \mu* et des distracteurs.
fn inject_verified_noise(
    rng: &mut Rng,
    alloc: &mut InstSymAlloc,
    inst: &mut Instance,
    target: &MetaOp,
    distractors: &[MetaOp],
    k: usize,
    noise_ratio: f64,
) -> usize {
    let desired = ((inst.rules.len() as f64) * noise_ratio).round() as usize;
    let mut added = 0usize;

    for _ in 0..desired {
        let decoy_a = alloc.fresh(1);
        let decoy_b = alloc.fresh(1);
        let noise_rule = unary_rule(alloc, decoy_a, decoy_b, rng.gen_f64().clamp(0.1, 0.9));

        inst.rules.push(noise_rule);
        inst.arities = alloc.arities.clone();

        let still_valid = evaluate_meta_op(target, inst, k)
            && distractors.iter().all(|d| !evaluate_meta_op(d, inst, k));

        if still_valid {
            added += 1;
        } else {
            inst.rules.pop();
        }
    }
    added
}
