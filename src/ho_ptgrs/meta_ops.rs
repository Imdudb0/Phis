use super::rewriting::{detect_absorber_symbols, DerivationGraph, Rule};
use super::term_graph::{anti_unify, SymbolId, TermGraph};

use std::collections::{BTreeMap, BTreeSet};

/// Base canonique des 6 opérateurs méta-relationnels M et leur clôture par composition.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MetaOp {
    /// Ambiguïté / Non-confluence (>= 2 formes normales incomparables, acyclique).
    Bif,
    /// Analogie / Isomorphisme de D_k sur deux sous-signatures disjointes.
    Iso,
    /// Contradiction / Obstruction (atteinte d'un absorbeur structurel \bot).
    Bot,
    /// Auto-référence / Paradoxe (cycle dirigé pur, 0 forme normale).
    Fix,
    /// Abstraction / Moule commun (anti-unification non triviale sur >= 3 dérivations non isomorphes).
    Quot,
    /// Composition d'ordre supérieur (clôture récursive \mu_comp).
    Comp(Box<MetaOp>, Box<MetaOp>),
}

impl MetaOp {
    pub fn comp(outer: MetaOp, inner: MetaOp) -> Self {
        MetaOp::Comp(Box::new(outer), Box::new(inner))
    }

    /// Pool complet d'opérateurs (5 opérateurs atomiques + compositions canoniques d'ordre supérieur).
    pub fn catalog() -> Vec<MetaOp> {
        vec![
            MetaOp::Bif,
            MetaOp::Iso,
            MetaOp::Bot,
            MetaOp::Fix,
            MetaOp::Quot,
            MetaOp::comp(MetaOp::Bif, MetaOp::Fix),
            MetaOp::comp(MetaOp::Iso, MetaOp::Bif),
            MetaOp::comp(MetaOp::Iso, MetaOp::Fix),
            MetaOp::comp(MetaOp::Quot, MetaOp::Bif),
        ]
    }
}

/// Instance auto-spécifiée sur \Sigma_inst : termes racines + règles R_0 + arités.
#[derive(Debug, Clone)]
pub struct Instance {
    pub roots: Vec<TermGraph>,
    pub rules: Vec<Rule>,
    pub arities: BTreeMap<SymbolId, usize>,
}

impl Instance {
    pub fn all_symbols(&self) -> BTreeSet<SymbolId> {
        let mut set = BTreeSet::new();
        for g in &self.roots {
            set.extend(g.symbols());
        }
        for r in &self.rules {
            set.extend(r.symbols());
        }
        set
    }
}

fn reaches_absorber(dg: &DerivationGraph, absorbers: &BTreeSet<SymbolId>) -> bool {
    if absorbers.is_empty() {
        return false;
    }
    dg.normal_forms().into_iter().any(|nf_id| {
        let g = &dg.states[nf_id];
        g.nodes.len() == 1 && absorbers.contains(&g.nodes[g.root].label)
    })
}

/// Évalue exactement un prédicat topologique \mu \in \overline{M} sur D_k(G_inst, R_0).
pub fn evaluate_meta_op(op: &MetaOp, inst: &Instance, horizon_k: usize) -> bool {
    const MAX_STATES: usize = 64;
    if inst.roots.is_empty() {
        return false;
    }
    let dgs: Vec<DerivationGraph> = inst
        .roots
        .iter()
        .map(|g| DerivationGraph::build(g, &inst.rules, horizon_k, MAX_STATES))
        .collect();
    let absorbers = detect_absorber_symbols(&inst.rules, &inst.arities);

    match op {
        MetaOp::Bif => {
            dgs.len() == 1
                && !dgs[0].has_directed_cycle()
                && !reaches_absorber(&dgs[0], &absorbers)
                && dgs[0].normal_forms().len() >= 2
        }
        MetaOp::Fix => {
            dgs.len() == 1
                && dgs[0].has_directed_cycle()
                && dgs[0].normal_forms().is_empty()
                && !reaches_absorber(&dgs[0], &absorbers)
        }
        MetaOp::Bot => {
            !absorbers.is_empty()
                && dgs.iter().all(|dg| !dg.has_directed_cycle())
                && dgs.iter().any(|dg| reaches_absorber(dg, &absorbers))
        }
        MetaOp::Iso => {
            if dgs.len() != 2 {
                return false;
            }
            let s0 = dgs[0].reachable_symbols();
            let s1 = dgs[1].reachable_symbols();
            s0.is_disjoint(&s1)
                && dgs[0].states.len() >= 3
                && !dgs[0].has_directed_cycle()
                && !dgs[1].has_directed_cycle()
                && dgs[0].normal_forms().len() == 1
                && dgs[1].normal_forms().len() == 1
                && !reaches_absorber(&dgs[0], &absorbers)
                && !reaches_absorber(&dgs[1], &absorbers)
                && dgs[0].is_unlabeled_isomorphic(&dgs[1])
        }
        MetaOp::Quot => {
            if dgs.len() < 3 {
                return false;
            }
            let all_clean = dgs.iter().all(|dg| {
                !dg.has_directed_cycle()
                    && dg.normal_forms().len() == 1
                    && !reaches_absorber(dg, &absorbers)
            });
            if !all_clean {
                return false;
            }
            let all_iso = (1..dgs.len()).all(|i| dgs[0].is_unlabeled_isomorphic(&dgs[i]));
            if all_iso {
                return false;
            }
            let nfs: Vec<TermGraph> = dgs
                .iter()
                .map(|dg| dg.states[dg.normal_forms()[0]].clone())
                .collect();
            anti_unify(&nfs).map_or(false, |t| t.is_nontrivial())
        }
        MetaOp::Comp(a, b) => match (a.as_ref(), b.as_ref()) {
            (MetaOp::Bif, MetaOp::Fix) | (MetaOp::Fix, MetaOp::Bif) => {
                dgs.len() == 1
                    && dgs[0].has_directed_cycle()
                    && !dgs[0].normal_forms().is_empty()
                    && !reaches_absorber(&dgs[0], &absorbers)
            }
            (MetaOp::Iso, MetaOp::Bif) | (MetaOp::Bif, MetaOp::Iso) => {
                if dgs.len() != 2 {
                    return false;
                }
                let s0 = dgs[0].reachable_symbols();
                let s1 = dgs[1].reachable_symbols();
                s0.is_disjoint(&s1)
                    && !dgs[0].has_directed_cycle()
                    && !dgs[1].has_directed_cycle()
                    && dgs[0].normal_forms().len() >= 2
                    && dgs[1].normal_forms().len() >= 2
                    && !reaches_absorber(&dgs[0], &absorbers)
                    && !reaches_absorber(&dgs[1], &absorbers)
                    && dgs[0].is_unlabeled_isomorphic(&dgs[1])
            }
            (MetaOp::Iso, MetaOp::Fix) | (MetaOp::Fix, MetaOp::Iso) => {
                if dgs.len() != 2 {
                    return false;
                }
                let s0 = dgs[0].reachable_symbols();
                let s1 = dgs[1].reachable_symbols();
                s0.is_disjoint(&s1)
                    && dgs[0].states.len() >= 2
                    && dgs[0].has_directed_cycle()
                    && dgs[1].has_directed_cycle()
                    && dgs[0].normal_forms().is_empty()
                    && dgs[1].normal_forms().is_empty()
                    && dgs[0].is_unlabeled_isomorphic(&dgs[1])
            }
            (MetaOp::Quot, MetaOp::Bif) | (MetaOp::Bif, MetaOp::Quot) => {
                if dgs.len() < 3 {
                    return false;
                }
                let all_bif = dgs.iter().all(|dg| {
                    !dg.has_directed_cycle()
                        && dg.normal_forms().len() >= 2
                        && !reaches_absorber(dg, &absorbers)
                });
                all_bif && anti_unify(&inst.roots).map_or(false, |t| t.is_nontrivial())
            }
            _ => false,
        },
    }
}

/// Graphe relationnel du dictionnaire auto-spécifié G_def sur \Sigma_def.
#[derive(Debug, Clone)]
pub struct DefGraph {
    pub node_symbols: Vec<SymbolId>,
    /// Triplets relationnels (src, relation_symbol, dst).
    pub edges: Vec<(usize, SymbolId, usize)>,
    /// Nœuds cibles v_{\mu} pour chaque opérateur du dictionnaire.
    pub concept_nodes: BTreeMap<MetaOp, usize>,
    /// Rôle structurel pour le test de rigidité (permet de masquer les identités des concepts).
    pub is_concept_node: Vec<bool>,
}

impl DefGraph {
    pub fn all_symbols(&self) -> BTreeSet<SymbolId> {
        let mut set: BTreeSet<SymbolId> = self.node_symbols.iter().copied().collect();
        for &(_, rel, _) in &self.edges {
            set.insert(rel);
        }
        set
    }
}

// Plage réservée aux symboles de \Sigma_def : [0, 9999]
// Relations ontologiques partagées :
const REL_DEFINES: SymbolId = SymbolId(1);
const REL_STEP: SymbolId = SymbolId(2);
const REL_NORMAL_FORM: SymbolId = SymbolId(3);
const REL_DISTINCT: SymbolId = SymbolId(4);
const REL_CYCLE_EDGE: SymbolId = SymbolId(5);
const REL_NO_NF: SymbolId = SymbolId(6);
const REL_ABSORBS: SymbolId = SymbolId(7);
const REL_DISJOINT: SymbolId = SymbolId(8);
const REL_BIJECTION: SymbolId = SymbolId(9);
const REL_TEMPLATE_OF: SymbolId = SymbolId(10);
const REL_NON_ISO: SymbolId = SymbolId(11);
const REL_COMP_OUTER: SymbolId = SymbolId(12);
const REL_COMP_INNER: SymbolId = SymbolId(13);
const REL_ACYCLIC: SymbolId = SymbolId(14);

// Hubs ontologiques partagés (connectent le dictionnaire en un graphe connexe) :
const HUB_ACYCLIC: SymbolId = SymbolId(100);
const HUB_CYCLE_PROP: SymbolId = SymbolId(101);
const HUB_ABSORBER_PROP: SymbolId = SymbolId(102);
const HUB_NO_NF_PROP: SymbolId = SymbolId(103);

/// Construit le graphe définitionnel G_def pour les opérateurs `ops` (cible + distracteurs).
pub fn build_def_graph(ops: &[MetaOp]) -> DefGraph {
    let mut node_symbols = Vec::new();
    let mut is_concept_node = Vec::new();
    let mut edges = Vec::new();
    let mut concept_nodes = BTreeMap::new();

    let mut next_sym_id: u32 = 200;
    let mut alloc_node = |sym: Option<SymbolId>, is_concept: bool| -> usize {
        let s = sym.unwrap_or_else(|| {
            let id = SymbolId(next_sym_id);
            next_sym_id += 1;
            id
        });
        let idx = node_symbols.len();
        node_symbols.push(s);
        is_concept_node.push(is_concept);
        idx
    };

    // Hubs ontologiques partagés
    let h_acyclic = alloc_node(Some(HUB_ACYCLIC), false);
    let h_cycle = alloc_node(Some(HUB_CYCLE_PROP), false);
    let h_absorber = alloc_node(Some(HUB_ABSORBER_PROP), false);
    let h_no_nf = alloc_node(Some(HUB_NO_NF_PROP), false);

    // Construit le schéma topologique d'un opérateur et renvoie son nœud racine de motif.
    #[allow(clippy::too_many_arguments)]
    fn build_schema(
        op: &MetaOp,
        h_acyclic: usize,
        h_cycle: usize,
        h_absorber: usize,
        h_no_nf: usize,
        alloc: &mut dyn FnMut(Option<SymbolId>, bool) -> usize,
        edges: &mut Vec<(usize, SymbolId, usize)>,
    ) -> usize {
        match op {
            MetaOp::Bif => {
                // Racine -> 2 branches vers 2 formes normales distinctes, acyclique
                let r = alloc(None, false);
                let nf1 = alloc(None, false);
                let nf2 = alloc(None, false);
                edges.push((r, REL_STEP, nf1));
                edges.push((r, REL_STEP, nf2));
                edges.push((r, REL_NORMAL_FORM, nf1));
                edges.push((r, REL_NORMAL_FORM, nf2));
                edges.push((nf1, REL_DISTINCT, nf2));
                edges.push((nf2, REL_DISTINCT, nf1));
                edges.push((r, REL_ACYCLIC, h_acyclic));
                r
            }
            MetaOp::Fix => {
                // Cycle dirigé s1 -> s2 -> s1 sans forme normale
                let s1 = alloc(None, false);
                let s2 = alloc(None, false);
                edges.push((s1, REL_STEP, s2));
                edges.push((s2, REL_STEP, s1));
                edges.push((s1, REL_CYCLE_EDGE, h_cycle));
                edges.push((s1, REL_NO_NF, h_no_nf));
                s1
            }
            MetaOp::Bot => {
                // Dérivation vers un absorbeur qui absorbe un constructeur
                let r = alloc(None, false);
                let z = alloc(None, false);
                let ctor = alloc(None, false);
                edges.push((r, REL_STEP, z));
                edges.push((r, REL_NORMAL_FORM, z));
                edges.push((z, REL_ABSORBS, ctor));
                edges.push((z, REL_ABSORBS, h_absorber));
                edges.push((r, REL_ACYCLIC, h_acyclic));
                r
            }
            MetaOp::Iso => {
                // Deux domaines disjoints reliés par bijection, chacun confluent (1 NF)
                let hub = alloc(None, false);
                let d1 = alloc(None, false);
                let d2 = alloc(None, false);
                let nf1 = alloc(None, false);
                let nf2 = alloc(None, false);
                edges.push((hub, REL_STEP, d1));
                edges.push((hub, REL_STEP, d2));
                edges.push((d1, REL_DISJOINT, d2));
                edges.push((d2, REL_DISJOINT, d1));
                edges.push((d1, REL_BIJECTION, d2));
                edges.push((d2, REL_BIJECTION, d1));
                edges.push((d1, REL_NORMAL_FORM, nf1));
                edges.push((d2, REL_NORMAL_FORM, nf2));
                edges.push((hub, REL_ACYCLIC, h_acyclic));
                hub
            }
            MetaOp::Quot => {
                // Un moule commun projeté vers 3 instances non isomorphes
                let tmpl = alloc(None, false);
                let i1 = alloc(None, false);
                let i2 = alloc(None, false);
                let i3 = alloc(None, false);
                edges.push((tmpl, REL_TEMPLATE_OF, i1));
                edges.push((tmpl, REL_TEMPLATE_OF, i2));
                edges.push((tmpl, REL_TEMPLATE_OF, i3));
                edges.push((i1, REL_NON_ISO, i2));
                edges.push((i2, REL_NON_ISO, i3));
                edges.push((tmpl, REL_ACYCLIC, h_acyclic));
                tmpl
            }
            MetaOp::Comp(a, b) => {
                let comp_root = alloc(None, false);
                let sa = build_schema(a, h_acyclic, h_cycle, h_absorber, h_no_nf, alloc, edges);
                let sb = build_schema(b, h_acyclic, h_cycle, h_absorber, h_no_nf, alloc, edges);
                edges.push((comp_root, REL_COMP_OUTER, sa));
                edges.push((comp_root, REL_COMP_INNER, sb));
                comp_root
            }
        }
    }

    for op in ops {
        let v_concept = alloc_node(None, true);
        let schema_root = build_schema(
            op,
            h_acyclic,
            h_cycle,
            h_absorber,
            h_no_nf,
            &mut alloc_node,
            &mut edges,
        );
        edges.push((v_concept, REL_DEFINES, schema_root));
        concept_nodes.insert(op.clone(), v_concept);
    }

    DefGraph {
        node_symbols,
        edges,
        concept_nodes,
        is_concept_node,
    }
}
