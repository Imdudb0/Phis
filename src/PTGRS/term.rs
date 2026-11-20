//! Hash-consed term graph: one arena holds the symbol table and the nodes.
//! Identical (symbol, children) pairs always yield the same `NodeId`, so
//! structural equality is id equality and sharing is maximal.

use std::collections::HashMap;

pub type NodeId = u32;
pub type SymId = u32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Pred,
    Func, // constants are Func with arity 0
    Var,
}

#[derive(Clone, Debug)]
pub struct SymInfo {
    pub name: String,
    pub kind: Kind,
    pub arity: usize,
    pub aux: usize, // variable index for Kind::Var
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Node {
    pub sym: SymId,
    pub args: Box<[NodeId]>,
}

#[derive(Default)]
pub struct Graph {
    syms: Vec<SymInfo>,
    by_name: HashMap<String, SymId>,
    nodes: Vec<Node>,
    table: HashMap<Node, NodeId>,
    vars: Vec<NodeId>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph::default()
    }

    fn new_sym(&mut self, name: String, kind: Kind, arity: usize, aux: usize) -> SymId {
        let id = self.syms.len() as SymId;
        self.by_name.insert(name.clone(), id);
        self.syms.push(SymInfo { name, kind, arity, aux });
        id
    }

    /// Get or create a symbol by name.
    pub fn sym(&mut self, name: &str, kind: Kind, arity: usize) -> SymId {
        if let Some(&s) = self.by_name.get(name) {
            return s;
        }
        self.new_sym(name.to_string(), kind, arity, 0)
    }

    pub fn rename_sym(&mut self, s: SymId, new_name: &str) {
        let old = self.syms[s as usize].name.clone();
        self.by_name.remove(&old);
        self.syms[s as usize].name = new_name.to_string();
        self.by_name.insert(new_name.to_string(), s);
    }

    /// Hash-consing constructor.
    pub fn mk(&mut self, sym: SymId, args: &[NodeId]) -> NodeId {
        let node = Node { sym, args: args.to_vec().into_boxed_slice() };
        if let Some(&id) = self.table.get(&node) {
            return id;
        }
        let id = self.nodes.len() as NodeId;
        self.nodes.push(node.clone());
        self.table.insert(node, id);
        id
    }

    /// Pattern variable number `i` (`?0`, `?1`, ...).
    pub fn var(&mut self, i: usize) -> NodeId {
        while self.vars.len() <= i {
            let k = self.vars.len();
            let s = self.new_sym(format!("?{k}"), Kind::Var, 0, k);
            let n = self.mk(s, &[]);
            self.vars.push(n);
        }
        self.vars[i]
    }

    pub fn head(&self, id: NodeId) -> SymId {
        self.nodes[id as usize].sym
    }

    pub fn args(&self, id: NodeId) -> &[NodeId] {
        &self.nodes[id as usize].args
    }

    pub fn info(&self, s: SymId) -> &SymInfo {
        &self.syms[s as usize]
    }

    pub fn name(&self, s: SymId) -> &str {
        &self.syms[s as usize].name
    }

    pub fn var_index(&self, id: NodeId) -> Option<usize> {
        let s = &self.syms[self.nodes[id as usize].sym as usize];
        if s.kind == Kind::Var { Some(s.aux) } else { None }
    }

    pub fn max_var(&self, id: NodeId) -> Option<usize> {
        if let Some(i) = self.var_index(id) {
            return Some(i);
        }
        let mut best: Option<usize> = None;
        for &a in self.args(id).iter() {
            if let Some(m) = self.max_var(a) {
                best = Some(best.map_or(m, |b| b.max(m)));
            }
        }
        best
    }

    pub fn depth(&self, id: NodeId) -> usize {
        1 + self.args(id).iter().map(|&a| self.depth(a)).max().unwrap_or(0)
    }

    /// One-way matching of `pat` against ground `tgt`, extending `sub`.
    /// On failure `sub` may be partially extended; callers clone before trying.
    pub fn match_term(&self, pat: NodeId, tgt: NodeId, sub: &mut Vec<Option<NodeId>>) -> bool {
        if pat == tgt {
            return true;
        }
        let pn = &self.nodes[pat as usize];
        let ps = &self.syms[pn.sym as usize];
        if ps.kind == Kind::Var {
            let cur = sub[ps.aux];
            return match cur {
                Some(b) => b == tgt,
                None => {
                    sub[ps.aux] = Some(tgt);
                    true
                }
            };
        }
        let tn = &self.nodes[tgt as usize];
        if pn.sym != tn.sym || pn.args.len() != tn.args.len() {
            return false;
        }
        for k in 0..pn.args.len() {
            if !self.match_term(pn.args[k], tn.args[k], sub) {
                return false;
            }
        }
        true
    }

    /// Instantiate a pattern under a total substitution.
    pub fn subst(&mut self, pat: NodeId, sub: &[Option<NodeId>]) -> NodeId {
        let sym = self.nodes[pat as usize].sym;
        let info = &self.syms[sym as usize];
        if info.kind == Kind::Var {
            return sub[info.aux].expect("unbound variable in right-hand side");
        }
        let args: Vec<NodeId> = self.nodes[pat as usize].args.to_vec();
        if args.is_empty() {
            return pat;
        }
        let mut na: Vec<NodeId> = Vec::with_capacity(args.len());
        for a in args {
            na.push(self.subst(a, sub));
        }
        self.mk(sym, &na)
    }

    pub fn show(&self, id: NodeId) -> String {
        let mut s = String::new();
        self.show_into(id, &mut s);
        s
    }

    fn show_into(&self, id: NodeId, out: &mut String) {
        let n = &self.nodes[id as usize];
        out.push_str(self.name(n.sym));
        if !n.args.is_empty() {
            out.push('(');
            for (i, &a) in n.args.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                self.show_into(a, out);
            }
            out.push(')');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_consing_and_matching() {
        let mut g = Graph::new();
        let a = g.sym("a", Kind::Func, 0);
        let p = g.sym("p", Kind::Pred, 1);
        let na = g.mk(a, &[]);
        let f1 = g.mk(p, &[na]);
        let f2 = g.mk(p, &[na]);
        assert_eq!(f1, f2);
        let x = g.var(0);
        let pat = g.mk(p, &[x]);
        let mut sub = vec![None; 1];
        assert!(g.match_term(pat, f1, &mut sub));
        assert_eq!(sub[0], Some(na));
        assert_eq!(g.show(f1), "p(a)");
    }
}