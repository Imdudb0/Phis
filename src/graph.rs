//! Substrat : nœuds + arêtes portées par matrices D×D fondamentalement
//! non-réciproques (W_{i→j} != W_{j→i}^T par construction).

use crate::rng::Rng;

pub const D: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeKind {
    Term,
    Var,
    Byte,
    Concept,
}

impl NodeKind {
    pub fn to_u8(self) -> u8 { self as u8 }
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => NodeKind::Term,
            1 => NodeKind::Var,
            2 => NodeKind::Byte,
            3 => NodeKind::Concept,
            _ => NodeKind::Byte,
        }
    }
}

#[derive(Clone)]
pub struct Node {
    pub emb: [f32; D],
    pub kind: NodeKind,
    pub label: u32,
}

#[derive(Clone)]
pub struct Edge {
    pub src: usize,
    pub dst: usize,
    pub w: Box<[[f32; D]; D]>,
}

pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl Graph {
    pub fn new() -> Self { Graph { nodes: vec![], edges: vec![] } }

    pub fn add_node(&mut self, emb: [f32; D], kind: NodeKind, label: u32) -> usize {
        self.nodes.push(Node { emb, kind, label });
        self.nodes.len() - 1
    }

    pub fn add_edge(&mut self, src: usize, dst: usize, w: Box<[[f32; D]; D]>) {
        self.edges.push(Edge { src, dst, w });
    }

    pub fn random_matrix(rng: &mut Rng) -> Box<[[f32; D]; D]> {
        let mut m = Box::new([[0.0f32; D]; D]);
        for i in 0..D {
            for j in 0..D {
                m[i][j] = rng.gauss() * 0.1;
            }
        }
        m
    }

    pub fn clone_graph(&self) -> Graph {
        Graph { nodes: self.nodes.clone(), edges: self.edges.clone() }
    }

    /// Activation propageant W_{src→dst} sur la source vers le dest, avec
    /// non-linéarité tanh et retour sur l'emb du nœud dst.
    pub fn propagate(&mut self, steps: usize) {
        for _ in 0..steps {
            let mut delta = vec![[0.0f32; D]; self.nodes.len()];
            for e in &self.edges {
                let src = &self.nodes[e.src].emb;
                for i in 0..D {
                    let mut acc = 0.0f32;
                    for j in 0..D { acc += e.w[i][j] * src[j]; }
                    delta[e.dst][i] += acc;
                }
            }
            for (n, d) in self.nodes.iter_mut().zip(delta.iter()) {
                for i in 0..D {
                    n.emb[i] = (n.emb[i] + 0.3 * d[i]).tanh();
                }
            }
        }
    }
}