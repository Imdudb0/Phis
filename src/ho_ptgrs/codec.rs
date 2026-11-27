use super::meta_ops::{DefGraph, Instance};
use super::rewriting::{PatternNode, Rule, PatternGraph};
use super::rng::Rng;
use super::term_graph::{SymbolId, TermGraph};

use std::collections::{BTreeMap, BTreeSet};

/// Marqueurs syntaxiques internes propres à chaque bloc (disjoints entre Def et Inst).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DefTag {
    BlockStart,
    EdgeSep,
    BlockEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum InstTag {
    BlockStart,
    TermStart,
    NodeSep,
    ChildStart,
    ChildEnd,
    RuleStart,
    RuleArrow,
    VarPrefix,
    BlockEnd,
}

/// Codec épisodique aléatoire préfixe-libre sur l'alphabet des octets B = {0..255}.
/// Garanti invariant en loi sous S_256 et strictement disjoint entre \Sigma_def et \Sigma_inst.
#[derive(Debug, Clone)]
pub struct EpisodicCodec {
    def_sym_map: BTreeMap<SymbolId, Vec<u8>>,
    inst_sym_map: BTreeMap<SymbolId, Vec<u8>>,
    def_tags: BTreeMap<DefTag, Vec<u8>>,
    inst_tags: BTreeMap<InstTag, Vec<u8>>,
    /// Encodage des indices de nœuds / variables locaux par bloc (également préfixe-libres et disjoints).
    def_idx_codes: Vec<Vec<u8>>,
    inst_idx_codes: Vec<Vec<u8>>,
}

impl EpisodicCodec {
    /// Génère un arbre préfixe-libre aléatoire de longueur variable [2..4] octets invariant sous S_256.
    pub fn sample(
        rng: &mut Rng,
        def_syms: &BTreeSet<SymbolId>,
        inst_syms: &BTreeSet<SymbolId>,
        max_def_idx: usize,
        max_inst_idx: usize,
    ) -> Self {
        let total_needed = def_syms.len()
            + inst_syms.len()
            + 3 // DefTag
            + 9 // InstTag
            + max_def_idx.max(1)
            + max_inst_idx.max(1);

        let mut pool = generate_prefix_free_pool(rng, total_needed);
        rng.shuffle(&mut pool);
        let mut iter = pool.into_iter();

        let mut def_sym_map = BTreeMap::new();
        for &s in def_syms {
            def_sym_map.insert(s, iter.next().unwrap());
        }
        let mut inst_sym_map = BTreeMap::new();
        for &s in inst_syms {
            inst_sym_map.insert(s, iter.next().unwrap());
        }

        let mut def_tags = BTreeMap::new();
        for tag in [DefTag::BlockStart, DefTag::EdgeSep, DefTag::BlockEnd] {
            def_tags.insert(tag, iter.next().unwrap());
        }

        let mut inst_tags = BTreeMap::new();
        for tag in [
            InstTag::BlockStart,
            InstTag::TermStart,
            InstTag::NodeSep,
            InstTag::ChildStart,
            InstTag::ChildEnd,
            InstTag::RuleStart,
            InstTag::RuleArrow,
            InstTag::VarPrefix,
            InstTag::BlockEnd,
        ] {
            inst_tags.insert(tag, iter.next().unwrap());
        }

        let def_idx_codes: Vec<Vec<u8>> = (0..max_def_idx.max(1))
            .map(|_| iter.next().unwrap())
            .collect();
        let inst_idx_codes: Vec<Vec<u8>> = (0..max_inst_idx.max(1))
            .map(|_| iter.next().unwrap())
            .collect();

        Self {
            def_sym_map,
            inst_sym_map,
            def_tags,
            inst_tags,
            def_idx_codes,
            inst_idx_codes,
        }
    }

    /// Renvoie le mot de code d'un symbole de \Sigma_def (pour évaluation post-hoc uniquement).
    pub fn def_code(&self, sym: SymbolId) -> Option<&[u8]> {
        self.def_sym_map.get(&sym).map(|v| v.as_slice())
    }

    /// Vérifie analytiquement l'invariant de jauge :
    /// 1) Im(\pi_1) \cap Im(\pi_2) = \emptyset (zéro recouvrement lexical entre blocs)
    /// 2) Propriété préfixe-libre stricte sur l'ensemble de tous les mots de code.
    pub fn verify_invariants(&self) -> bool {
        let mut def_words: BTreeSet<&[u8]> = BTreeSet::new();
        for v in self.def_sym_map.values() {
            def_words.insert(v.as_slice());
        }
        for v in self.def_tags.values() {
            def_words.insert(v.as_slice());
        }
        for v in &self.def_idx_codes {
            def_words.insert(v.as_slice());
        }

        let mut inst_words: BTreeSet<&[u8]> = BTreeSet::new();
        for v in self.inst_sym_map.values() {
            inst_words.insert(v.as_slice());
        }
        for v in self.inst_tags.values() {
            inst_words.insert(v.as_slice());
        }
        for v in &self.inst_idx_codes {
            inst_words.insert(v.as_slice());
        }

        if !def_words.is_disjoint(&inst_words) {
            return false;
        }

        let mut all: Vec<&[u8]> = def_words.into_iter().chain(inst_words).collect();
        all.sort_unstable();
        for pair in all.windows(2) {
            if pair[1].starts_with(pair[0]) {
                return false;
            }
        }
        true
    }

    /// Sérialise (G_def, G_inst) en flux d'octets brut `perceived.eff` avec permutation d'ordre aléatoire.
    pub fn serialize(&self, rng: &mut Rng, def: &DefGraph, inst: &Instance) -> Vec<u8> {
        let mut def_bytes = self.serialize_def(rng, def);
        let mut inst_bytes = self.serialize_inst(rng, inst);

        let mut out = Vec::with_capacity(def_bytes.len() + inst_bytes.len());
        if rng.gen_bool(0.5) {
            out.append(&mut def_bytes);
            out.append(&mut inst_bytes);
        } else {
            out.append(&mut inst_bytes);
            out.append(&mut def_bytes);
        }
        out
    }

    fn serialize_def(&self, rng: &mut Rng, def: &DefGraph) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.def_tags[&DefTag::BlockStart]);

        let mut shuffled_edges = def.edges.clone();
        rng.shuffle(&mut shuffled_edges);

        for (src, rel, dst) in shuffled_edges {
            buf.extend_from_slice(&self.def_tags[&DefTag::EdgeSep]);
            buf.extend_from_slice(&self.def_sym_map[&def.node_symbols[src]]);
            buf.extend_from_slice(&self.def_sym_map[&rel]);
            buf.extend_from_slice(&self.def_sym_map[&def.node_symbols[dst]]);
        }

        buf.extend_from_slice(&self.def_tags[&DefTag::BlockEnd]);
        buf
    }

    fn serialize_inst(&self, rng: &mut Rng, inst: &Instance) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.inst_tags[&InstTag::BlockStart]);

        // Mélanger les termes racines et les règles indépendamment
        let mut roots: Vec<&TermGraph> = inst.roots.iter().collect();
        rng.shuffle(&mut roots);
        let mut rules: Vec<&Rule> = inst.rules.iter().collect();
        rng.shuffle(&mut rules);

        // Ordre aléatoire entre les termes et les règles à l'intérieur du bloc
        let terms_first = rng.gen_bool(0.5);
        if terms_first {
            for g in &roots {
                self.encode_term_graph(&mut buf, g);
            }
            for r in &rules {
                self.encode_rule(&mut buf, r);
            }
        } else {
            for r in &rules {
                self.encode_rule(&mut buf, r);
            }
            for g in &roots {
                self.encode_term_graph(&mut buf, g);
            }
        }

        buf.extend_from_slice(&self.inst_tags[&InstTag::BlockEnd]);
        buf
    }

    fn encode_term_graph(&self, buf: &mut Vec<u8>, g: &TermGraph) {
        buf.extend_from_slice(&self.inst_tags[&InstTag::TermStart]);
        buf.extend_from_slice(&self.inst_idx_codes[g.root]);
        for (idx, node) in g.nodes.iter().enumerate() {
            buf.extend_from_slice(&self.inst_tags[&InstTag::NodeSep]);
            buf.extend_from_slice(&self.inst_idx_codes[idx]);
            buf.extend_from_slice(&self.inst_sym_map[&node.label]);
            buf.extend_from_slice(&self.inst_tags[&InstTag::ChildStart]);
            for &c in &node.succ {
                buf.extend_from_slice(&self.inst_idx_codes[c]);
            }
            buf.extend_from_slice(&self.inst_tags[&InstTag::ChildEnd]);
        }
    }

    fn encode_rule(&self, buf: &mut Vec<u8>, r: &Rule) {
        buf.extend_from_slice(&self.inst_tags[&InstTag::RuleStart]);
        self.encode_pattern(buf, &r.lhs);
        buf.extend_from_slice(&self.inst_tags[&InstTag::RuleArrow]);
        self.encode_pattern(buf, &r.rhs);
    }

    fn encode_pattern(&self, buf: &mut Vec<u8>, pat: &PatternGraph) {
        buf.extend_from_slice(&self.inst_idx_codes[pat.root]);
        for (idx, node) in pat.nodes.iter().enumerate() {
            buf.extend_from_slice(&self.inst_tags[&InstTag::NodeSep]);
            buf.extend_from_slice(&self.inst_idx_codes[idx]);
            match node {
                PatternNode::Var(vid) => {
                    buf.extend_from_slice(&self.inst_tags[&InstTag::VarPrefix]);
                    buf.extend_from_slice(&self.inst_idx_codes[*vid as usize]);
                }
                PatternNode::Sym(sym, succ) => {
                    buf.extend_from_slice(&self.inst_sym_map[sym]);
                    buf.extend_from_slice(&self.inst_tags[&InstTag::ChildStart]);
                    for &c in succ {
                        buf.extend_from_slice(&self.inst_idx_codes[c]);
                    }
                    buf.extend_from_slice(&self.inst_tags[&InstTag::ChildEnd]);
                }
            }
        }
    }
}

/// Construit au moins `needed` mots de code binaires préfixe-libres de longueur variable [2..4],
/// symétriques sous l'action du groupe S_256.
fn generate_prefix_free_pool(rng: &mut Rng, needed: usize) -> Vec<Vec<u8>> {
    let mut single_bytes: Vec<u8> = (0u8..=255).collect();
    rng.shuffle(&mut single_bytes);

    let mut ready: Vec<Vec<u8>> = Vec::with_capacity(needed * 2);
    let mut alphabet_buf: Vec<u8> = (0u8..=255).collect();

    // Étape 1 : expansion vers des préfixes de longueur 2
    while ready.len() < needed + 32 {
        if let Some(b0) = single_bytes.pop() {
            rng.shuffle(&mut alphabet_buf);
            let branch = rng.gen_range_inclusive(8, 24);
            for &b1 in alphabet_buf.iter().take(branch) {
                ready.push(vec![b0, b1]);
            }
        } else {
            let pick = rng.gen_range(ready.len());
            let base = ready.swap_remove(pick);
            rng.shuffle(&mut alphabet_buf);
            let branch = rng.gen_range_inclusive(4, 12);
            for &b in alphabet_buf.iter().take(branch) {
                let mut ext = base.clone();
                ext.push(b);
                ready.push(ext);
            }
        }
    }

    // Étape 2 : approfondissement aléatoire d'une fraction des feuilles vers longueurs 3 et 4
    let deepen_steps = (ready.len() / 5).max(4);
    for _ in 0..deepen_steps {
        let pick = rng.gen_range(ready.len());
        if ready[pick].len() >= 4 {
            continue;
        }
        let base = ready.swap_remove(pick);
        rng.shuffle(&mut alphabet_buf);
        let branch = rng.gen_range_inclusive(2, 4);
        for &b in alphabet_buf.iter().take(branch) {
            let mut ext = base.clone();
            ext.push(b);
            ready.push(ext);
        }
    }

    ready
}
