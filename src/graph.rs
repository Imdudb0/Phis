use std::f32::consts::LN_2;

pub const D: usize = 4;
pub const N_BYTE: usize = 256; // alphabet = octets, toujours présent
pub const N_WS: usize = 32;    // régional R
pub const N_C: usize = 8;      // cause
pub const N_G: usize = 16;     // régénératif
pub const N_A: usize = 8;      // abstraction
pub const N: usize = N_BYTE + N_WS + N_C + N_G + N_A;
pub const WS0: usize = N_BYTE;
pub const C0: usize = WS0 + N_WS;
pub const G0: usize = C0 + N_C;
pub const A0: usize = G0 + N_G;

pub type S = [f32; D];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EdgeKind { ByteWs, WsWs, WsC, CC, CG, GByte, WsA, AWs, CWs }

impl EdgeKind {
    pub fn is_fast(self) -> bool { matches!(self, EdgeKind::AWs | EdgeKind::CWs) }
    pub fn from_code(c: u8) -> Option<EdgeKind> {
        Some(match c {
            0 => EdgeKind::ByteWs, 1 => EdgeKind::WsWs, 2 => EdgeKind::WsC,
            3 => EdgeKind::CC, 4 => EdgeKind::CG, 5 => EdgeKind::GByte,
            6 => EdgeKind::WsA, 7 => EdgeKind::AWs, 8 => EdgeKind::CWs,
            _ => return None,
        })
    }
}

#[derive(Clone)]
pub struct Edge { pub src: u32, pub dst: u32, pub kind: EdgeKind, pub w: [f32; D * D] }

#[derive(Clone, Copy)]
pub struct Meta { pub eta_fast: f32, pub explore_gain: f32, pub phi_ema: f32 }

pub struct Graph {
    pub edges: Vec<Edge>,
    pub incoming: Vec<Vec<usize>>,
    pub bias: Vec<S>,
    pub meta: Meta,
}

pub struct Rng(pub u64);
impl Rng {
    pub fn new(seed: u64) -> Self { Rng(if seed == 0 { 1 } else { seed }) }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn next_f32(&mut self) -> f32 { (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32) }
    pub fn below(&mut self, n: usize) -> usize { if n == 0 { 0 } else { (self.next_u64() % n as u64) as usize } }
    pub fn norm(&mut self) -> f32 {
        (self.next_f32() + self.next_f32() + self.next_f32() + self.next_f32() - 2.0) * 0.7
    }
}

pub struct Budget { pub ops: u64, pub max: u64 }

impl Graph {
    pub fn build(rng: &mut Rng) -> Graph {
        let mut g = Graph {
            edges: Vec::new(), incoming: vec![Vec::new(); N], bias: vec![[0.0; D]; N],
            meta: Meta { eta_fast: 0.02, explore_gain: 1.0, phi_ema: 0.0 },
        };
        for b in 0..N_BYTE { for k in 0..2 {
            let w = WS0 + (b * 7 + k * 13 + 3) % N_WS;
            g.add_edge(b, w, EdgeKind::ByteWs, rng, 0.15);
        }}
        for i in 0..N_WS { for off in [1usize, 5, 11] {
            let j = (i + off) % N_WS;
            g.add_edge(WS0 + i, WS0 + j, EdgeKind::WsWs, rng, 0.12);
        }}
        for i in 0..N_WS {
            let c1 = i % N_C; let c2 = (i + 3) % N_C;
            g.add_edge(WS0 + i, C0 + c1, EdgeKind::WsC, rng, 0.12);
            g.add_edge(WS0 + i, C0 + c2, EdgeKind::WsC, rng, 0.12);
        }
        for i in 0..N_C { g.add_edge(C0 + i, C0 + (i + 1) % N_C, EdgeKind::CC, rng, 0.2); }
        for i in 0..N_C { for k in 0..4 {
            let j = (i * 4 + k) % N_G;
            g.add_edge(C0 + i, G0 + j, EdgeKind::CG, rng, 0.15);
        }}
        for i in 0..N_G { for k in 0..16 {
            let b = i * 16 + k; // couverture complète de l'alphabet
            g.add_edge(G0 + i, b, EdgeKind::GByte, rng, 0.1);
        }}
        for i in 0..N_WS { g.add_edge(WS0 + i, A0 + i % N_A, EdgeKind::WsA, rng, 0.1); }
        for k in 0..N_A { for t in 0..4 {
            let i = ((k + 1) % N_A) + t * N_A;
            g.add_edge(A0 + k, WS0 + i, EdgeKind::AWs, rng, 0.08);
        }}
        for k in 0..N_C { for t in 0..2 {
            let res = (k + 1 + t) % N_A; let i = res + 8 * (k % 4);
            g.add_edge(C0 + k, WS0 + i, EdgeKind::CWs, rng, 0.08);
        }}
        for n in 0..N { for d in 0..D { g.bias[n][d] = rng.norm() * 0.01; } }
        g.assert_non_reciprocal();
        g
    }

    // Non-réciprocité fondamentale: jamais de paire i->j et j->i.
    pub fn add_edge(&mut self, src: usize, dst: usize, kind: EdgeKind, rng: &mut Rng, scale: f32) {
        if src == dst { return; }
        for &ei in &self.incoming[src] { if self.edges[ei].src as usize == dst { return; } }
        for &ei in &self.incoming[dst] { if self.edges[ei].src as usize == src { return; } }
        let mut w = [0.0; D * D];
        for v in w.iter_mut() { *v = rng.norm() * scale; }
        let idx = self.edges.len();
        self.edges.push(Edge { src: src as u32, dst: dst as u32, kind, w });
        self.incoming[dst].push(idx);
    }

    pub fn rebuild_incoming(&mut self) {
        self.incoming = vec![Vec::new(); N];
        for (i, e) in self.edges.iter().enumerate() { self.incoming[e.dst as usize].push(i); }
    }

    pub fn assert_non_reciprocal(&self) {
        for e in &self.edges {
            for &ei in &self.incoming[e.src as usize] {
                assert!(self.edges[ei].src as usize != e.dst as usize, "réciprocité interdite");
            }
        }
    }
}

pub fn zero_state() -> Vec<S> { vec![[0.0; D]; N] }

pub fn step_full(g: &Graph, x: &[S], input: &[S], gain: &[f32], noise: f32,
                 rng: &mut Rng, budget: &mut Budget, cut: Option<EdgeKind>) -> Vec<S> {
    let mut nx = x.to_vec();
    for j in 0..N {
        let mut acc = g.bias[j];
        for &ei in &g.incoming[j] {
            let e = &g.edges[ei];
            if cut == Some(e.kind) { continue; }
            let s = e.src as usize;
            for d in 0..D {
                let mut v = 0.0;
                for k in 0..D { v += e.w[d * D + k] * x[s][k]; }
                acc[d] += v;
            }
            budget.ops += (D * D) as u64;
        }
        for d in 0..D {
            let mut v = (acc[d] + input[j][d]) * gain[j];
            if noise > 0.0 { v += rng.norm() * noise; }
            nx[j][d] = v.tanh();
        }
    }
    nx
}

// Boucle générative: ws (régional R) et a gelés; c, g, octets actifs.
pub fn gen_step(g: &Graph, x: &[S], gain: &[f32], noise: f32,
                rng: &mut Rng, budget: &mut Budget, cut: Option<EdgeKind>) -> Vec<S> {
    let mut nx = x.to_vec();
    for j in 0..N {
        if (j >= WS0 && j < C0) || j >= A0 { continue; }
        let mut acc = g.bias[j];
        for &ei in &g.incoming[j] {
            let e = &g.edges[ei];
            if cut == Some(e.kind) { continue; }
            let s = e.src as usize;
            for d in 0..D {
                let mut v = 0.0;
                for k in 0..D { v += e.w[d * D + k] * x[s][k]; }
                acc[d] += v;
            }
            budget.ops += (D * D) as u64;
        }
        for d in 0..D {
            let mut v = acc[d] * gain[j];
            if noise > 0.0 { v += rng.norm() * noise; }
            nx[j][d] = v.tanh();
        }
    }
    nx
}

pub fn norm(v: &S) -> f32 { v.iter().map(|a| a * a).sum::<f32>().sqrt() }
pub fn outer(a: &S, b: &S) -> [f32; D * D] {
    let mut o = [0.0; D * D];
    for i in 0..D { for j in 0..D { o[i * D + j] = a[i] * b[j]; } }
    o
}
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut d = 0.0; let mut na = 0.0; let mut nb = 0.0;
    for i in 0..a.len().min(b.len()) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    d / (na.sqrt() * nb.sqrt() + 1e-6)
}
pub fn entropy_bits(p: &[f32]) -> f32 {
    let mut h = 0.0;
    for &v in p { if v > 1e-12 { h -= v * (v.ln() / LN_2); } }
    h
}
pub fn tv(p: &[f32], q: &[f32]) -> f32 {
    let mut s = 0.0;
    for i in 0..p.len().min(q.len()) { s += (p[i] - q[i]).abs(); }
    0.5 * s
}
pub fn softmax_bytes(x: &[S]) -> [f32; 256] {
    let mut m = f32::NEG_INFINITY;
    for b in 0..N_BYTE { m = m.max(x[b][0]); }
    let mut out = [0.0; 256]; let mut s = 0.0;
    for b in 0..N_BYTE { let v = (x[b][0] - m).exp(); out[b] = v; s += v; }
    if s > 0.0 { for b in 0..N_BYTE { out[b] /= s; } }
    out
}
pub fn argmax_byte(p: &[f32; 256]) -> u8 {
    let mut bi = 0; let mut bv = f32::NEG_INFINITY;
    for b in 0..256 { if p[b] > bv { bv = p[b]; bi = b; } }
    bi as u8
}