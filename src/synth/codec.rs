use crate::rng::Rng;
use crate::synth::ptgrs::{PatNodeKind, PatSymbol, PatternGraph, RewriteRule, SymbolId, TermGraph};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotationStyle {
    PrefixPolish,
    PrefixDelimited,
    PostfixRpn,
    InfixBinary,
    TaggedFrame,
    ProseDelimited,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    RuleSpec,
    StateSnapshot,
    Query,
    Target,
}

#[derive(Clone, Debug)]
pub struct SpanMeta {
    pub block_id: usize,
    pub kind: BlockKind,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug)]
pub struct RandomCodec {
    pub style: NotationStyle,
    pub sym_tokens: HashMap<usize, Vec<u8>>,
    pub var_tokens: HashMap<usize, Vec<u8>>,
    pub sym_var_tokens: HashMap<usize, Vec<u8>>,
    pub tok_open: Vec<u8>,
    pub tok_close: Vec<u8>,
    pub tok_sep: Vec<u8>,
    pub tok_arrow: Vec<u8>,
    pub tok_block_open: Vec<u8>,
    pub tok_block_close: Vec<u8>,
    pub tok_tag_rule: Vec<u8>,
    pub tok_tag_state: Vec<u8>,
    pub tok_tag_query: Vec<u8>,
    pub tok_ref: Vec<u8>,
}

impl RandomCodec {
    pub fn sample(
        rng: &mut Rng,
        num_symbols: usize,
        max_vars: usize,
        max_sym_vars: usize,
        disjoint_halves: bool,
    ) -> Self {
        let styles = [
            NotationStyle::PrefixPolish,
            NotationStyle::PrefixDelimited,
            NotationStyle::PostfixRpn,
            NotationStyle::InfixBinary,
            NotationStyle::TaggedFrame,
            NotationStyle::ProseDelimited,
        ];
        let style = *rng.choose(&styles);

        let mut all_bytes: Vec<u8> = (1u8..=255u8).collect();
        rng.shuffle(&mut all_bytes);

        let (pool_low, pool_high, pool_syntax) = if disjoint_halves {
            let s_split = (all_bytes.len() * 4) / 5;
            let h_split = s_split / 2;
            (&all_bytes[..h_split], &all_bytes[h_split..s_split], &all_bytes[s_split..])
        } else {
            let s_split = (all_bytes.len() * 4) / 5;
            (&all_bytes[..s_split], &all_bytes[..s_split], &all_bytes[s_split..])
        };

        let mut used_tokens: HashSet<Vec<u8>> = HashSet::new();

        let mut draw_unique = |rng: &mut Rng, byte_pool: &[u8], min_len: usize, max_len: usize| -> Vec<u8> {
            for _ in 0..256 {
                let len = rng.range_usize(min_len, max_len + 1);
                let mut tok = Vec::with_capacity(len);
                for _ in 0..len {
                    tok.push(*rng.choose(byte_pool));
                }
                if used_tokens.insert(tok.clone()) {
                    return tok;
                }
            }
            let mut tok = vec![*rng.choose(byte_pool), *rng.choose(byte_pool), *rng.choose(byte_pool)];
            while !used_tokens.insert(tok.clone()) {
                tok.push(*rng.choose(byte_pool));
            }
            tok
        };

        let (sym_min, sym_max) = if style == NotationStyle::ProseDelimited {
            (2, 4)
        } else {
            (1, 2)
        };
        let mut sym_tokens = HashMap::new();
        let half_sym = num_symbols / 2;
        for s in 0..num_symbols {
            let pool = if s < half_sym { pool_low } else { pool_high };
            sym_tokens.insert(s, draw_unique(rng, pool, sym_min, sym_max));
        }

        let mut var_tokens = HashMap::new();
        for v in 0..max_vars {
            var_tokens.insert(v, draw_unique(rng, pool_syntax, 1, 2));
        }

        let mut sym_var_tokens = HashMap::new();
        for sv in 0..max_sym_vars {
            sym_var_tokens.insert(sv, draw_unique(rng, pool_syntax, 1, 2));
        }

        let tok_open = draw_unique(rng, pool_syntax, 1, 1);
        let tok_close = draw_unique(rng, pool_syntax, 1, 1);
        let tok_sep = draw_unique(rng, pool_syntax, 1, 1);
        let tok_arrow = draw_unique(rng, pool_syntax, 1, 2);
        let tok_block_open = draw_unique(rng, pool_syntax, 1, 1);
        let tok_block_close = draw_unique(rng, pool_syntax, 1, 1);
        let tok_tag_rule = draw_unique(rng, pool_syntax, 1, 2);
        let tok_tag_state = draw_unique(rng, pool_syntax, 1, 2);
        let tok_tag_query = draw_unique(rng, pool_syntax, 1, 2);
        let tok_ref = draw_unique(rng, pool_syntax, 1, 1);

        Self {
            style,
            sym_tokens,
            var_tokens,
            sym_var_tokens,
            tok_open,
            tok_close,
            tok_sep,
            tok_arrow,
            tok_block_open,
            tok_block_close,
            tok_tag_rule,
            tok_tag_state,
            tok_tag_query,
            tok_ref,
        }
    }

    pub fn encode_term_graph(&self, graph: &TermGraph, out: &mut Vec<u8>) {
        let mut visited: HashSet<usize> = HashSet::new();

        fn emit(
            codec: &RandomCodec,
            graph: &TermGraph,
            nid: usize,
            visited: &mut HashSet<usize>,
            out: &mut Vec<u8>,
            depth: usize,
        ) {
            if depth > 24 {
                return;
            }
            let is_revisited = !visited.insert(nid);
            let node = &graph.nodes[nid];
            let sym_tok = codec
                .sym_tokens
                .get(&node.sym.0)
                .expect("missing sym token");

            if is_revisited {
                out.extend_from_slice(&codec.tok_ref);
                out.extend_from_slice(sym_tok);
                return;
            }

            if node.args.is_empty() {
                out.extend_from_slice(sym_tok);
                return;
            }

            match codec.style {
                NotationStyle::PrefixPolish | NotationStyle::ProseDelimited => {
                    out.extend_from_slice(sym_tok);
                    for &arg in &node.args {
                        out.extend_from_slice(&codec.tok_sep);
                        emit(codec, graph, arg.0, visited, out, depth + 1);
                    }
                }
                NotationStyle::PrefixDelimited => {
                    out.extend_from_slice(sym_tok);
                    out.extend_from_slice(&codec.tok_open);
                    for (i, &arg) in node.args.iter().enumerate() {
                        if i > 0 {
                            out.extend_from_slice(&codec.tok_sep);
                        }
                        emit(codec, graph, arg.0, visited, out, depth + 1);
                    }
                    out.extend_from_slice(&codec.tok_close);
                }
                NotationStyle::PostfixRpn => {
                    for &arg in &node.args {
                        emit(codec, graph, arg.0, visited, out, depth + 1);
                        out.extend_from_slice(&codec.tok_sep);
                    }
                    out.extend_from_slice(sym_tok);
                }
                NotationStyle::InfixBinary => {
                    if node.args.len() == 2 {
                        out.extend_from_slice(&codec.tok_open);
                        emit(codec, graph, node.args[0].0, visited, out, depth + 1);
                        out.extend_from_slice(&codec.tok_sep);
                        out.extend_from_slice(sym_tok);
                        out.extend_from_slice(&codec.tok_sep);
                        emit(codec, graph, node.args[1].0, visited, out, depth + 1);
                        out.extend_from_slice(&codec.tok_close);
                    } else {
                        out.extend_from_slice(sym_tok);
                        out.extend_from_slice(&codec.tok_open);
                        for (i, &arg) in node.args.iter().enumerate() {
                            if i > 0 {
                                out.extend_from_slice(&codec.tok_sep);
                            }
                            emit(codec, graph, arg.0, visited, out, depth + 1);
                        }
                        out.extend_from_slice(&codec.tok_close);
                    }
                }
                NotationStyle::TaggedFrame => {
                    out.extend_from_slice(&codec.tok_open);
                    out.extend_from_slice(sym_tok);
                    for &arg in &node.args {
                        out.extend_from_slice(&codec.tok_sep);
                        emit(codec, graph, arg.0, visited, out, depth + 1);
                    }
                    out.extend_from_slice(&codec.tok_close);
                }
            }
        }

        emit(self, graph, graph.root.0, &mut visited, out, 0);
    }

    pub fn encode_pattern(&self, pat: &PatternGraph, out: &mut Vec<u8>) {
        fn emit(codec: &RandomCodec, pat: &PatternGraph, idx: usize, out: &mut Vec<u8>, depth: usize) {
            if depth > 24 {
                return;
            }
            match &pat.nodes[idx] {
                PatNodeKind::Var(v) => {
                    let tok = codec.var_tokens.get(v).expect("missing var token");
                    out.extend_from_slice(tok);
                }
                PatNodeKind::App { sym, args } => {
                    let sym_tok = match sym {
                        PatSymbol::Concrete(SymbolId(s)) => {
                            codec.sym_tokens.get(s).expect("missing sym token")
                        }
                        PatSymbol::SymVar(sv) => {
                            codec.sym_var_tokens.get(sv).expect("missing sym_var token")
                        }
                    };
                    if args.is_empty() {
                        out.extend_from_slice(sym_tok);
                        return;
                    }
                    match codec.style {
                        NotationStyle::PrefixPolish | NotationStyle::ProseDelimited => {
                            out.extend_from_slice(sym_tok);
                            for &a in args {
                                out.extend_from_slice(&codec.tok_sep);
                                emit(codec, pat, a, out, depth + 1);
                            }
                        }
                        NotationStyle::PrefixDelimited => {
                            out.extend_from_slice(sym_tok);
                            out.extend_from_slice(&codec.tok_open);
                            for (i, &a) in args.iter().enumerate() {
                                if i > 0 {
                                    out.extend_from_slice(&codec.tok_sep);
                                }
                                emit(codec, pat, a, out, depth + 1);
                            }
                            out.extend_from_slice(&codec.tok_close);
                        }
                        NotationStyle::PostfixRpn => {
                            for &a in args {
                                emit(codec, pat, a, out, depth + 1);
                                out.extend_from_slice(&codec.tok_sep);
                            }
                            out.extend_from_slice(sym_tok);
                        }
                        NotationStyle::InfixBinary => {
                            if args.len() == 2 {
                                out.extend_from_slice(&codec.tok_open);
                                emit(codec, pat, args[0], out, depth + 1);
                                out.extend_from_slice(&codec.tok_sep);
                                out.extend_from_slice(sym_tok);
                                out.extend_from_slice(&codec.tok_sep);
                                emit(codec, pat, args[1], out, depth + 1);
                                out.extend_from_slice(&codec.tok_close);
                            } else {
                                out.extend_from_slice(sym_tok);
                                out.extend_from_slice(&codec.tok_open);
                                for (i, &a) in args.iter().enumerate() {
                                    if i > 0 {
                                        out.extend_from_slice(&codec.tok_sep);
                                    }
                                    emit(codec, pat, a, out, depth + 1);
                                }
                                out.extend_from_slice(&codec.tok_close);
                            }
                        }
                        NotationStyle::TaggedFrame => {
                            out.extend_from_slice(&codec.tok_open);
                            out.extend_from_slice(sym_tok);
                            for &a in args {
                                out.extend_from_slice(&codec.tok_sep);
                                emit(codec, pat, a, out, depth + 1);
                            }
                            out.extend_from_slice(&codec.tok_close);
                        }
                    }
                }
            }
        }

        emit(self, pat, pat.root, out, 0);
    }

    pub fn encode_rule(&self, rule: &RewriteRule, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.tok_block_open);
        out.extend_from_slice(&self.tok_tag_rule);
        out.extend_from_slice(&self.tok_sep);
        self.encode_pattern(&rule.lhs, out);
        out.extend_from_slice(&self.tok_arrow);
        self.encode_pattern(&rule.rhs, out);
        out.extend_from_slice(&self.tok_block_close);
    }

    pub fn encode_state_block(&self, graph: &TermGraph, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.tok_block_open);
        out.extend_from_slice(&self.tok_tag_state);
        out.extend_from_slice(&self.tok_sep);
        self.encode_term_graph(graph, out);
        out.extend_from_slice(&self.tok_block_close);
    }

    pub fn encode_query_block(&self, probe: &TermGraph, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.tok_block_open);
        out.extend_from_slice(&self.tok_tag_query);
        out.extend_from_slice(&self.tok_sep);
        self.encode_term_graph(probe, out);
        out.extend_from_slice(&self.tok_arrow);
        out.extend_from_slice(&self.tok_block_close);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::ptgrs::{Signature, TermGraph};

    #[test]
    fn random_codecs_produce_disjoint_surface_bytes() {
        let mut rng = Rng::from_seed(77);
        let sig = Signature {
            arities: vec![0, 0, 2, 0, 0, 2],
        };
        let mut g = TermGraph::new_empty(SymbolId(0));
        let l1 = g.add_node(SymbolId(1), vec![]);
        g.root = g.add_node(SymbolId(2), vec![l1, l1]);

        let c1 = RandomCodec::sample(&mut rng, 6, 4, 4, true);
        let c2 = RandomCodec::sample(&mut rng, 6, 4, 4, true);

        let mut b1 = Vec::new();
        let mut b2 = Vec::new();
        c1.encode_state_block(&g, &mut b1);
        c2.encode_state_block(&g, &mut b2);
        assert!(!b1.is_empty() && !b2.is_empty());
        assert_ne!(b1, b2);
        let _ = sig;
    }
}