//! Algèbre de causes : le méta-a-priori abductif. Zéro donnée de domaine.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    Lit,
    Seq,
    Rep,
    Alt,
    Rec,
    Var,
}

pub const IDENT: [u8; 256] = {
    let mut arr = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        arr[i] = i as u8;
        i += 1;
    }
    arr
};

const BLAKE2B_IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const BLAKE2B_SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

#[inline]
fn blake2b_g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

fn blake2b_compress(h: &mut [u64; 8], block: &[u8; 128], t: [u64; 2], last: bool) {
    let mut m = [0u64; 16];
    for i in 0..16 {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&block[i * 8..(i + 1) * 8]);
        m[i] = u64::from_le_bytes(bytes);
    }

    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&BLAKE2B_IV);
    v[12] ^= t[0];
    v[13] ^= t[1];
    if last {
        v[14] = !v[14];
    }

    for r in 0..12 {
        let s = &BLAKE2B_SIGMA[r % 10];
        blake2b_g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        blake2b_g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        blake2b_g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        blake2b_g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        blake2b_g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        blake2b_g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        blake2b_g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        blake2b_g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }

    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// BLAKE2b with `digest_size = 8`.
pub fn hash8(data: &[u8]) -> [u8; 8] {
    let mut h = BLAKE2B_IV;
    h[0] ^= 0x0101_0008;

    if data.is_empty() {
        let block = [0u8; 128];
        blake2b_compress(&mut h, &block, [0, 0], true);
    } else {
        let mut offset = 0usize;
        let mut t0 = 0u64;
        let mut t1 = 0u64;
        while data.len() - offset > 128 {
            let mut block = [0u8; 128];
            block.copy_from_slice(&data[offset..offset + 128]);
            let (new_t0, carry) = t0.overflowing_add(128);
            t0 = new_t0;
            if carry {
                t1 = t1.wrapping_add(1);
            }
            blake2b_compress(&mut h, &block, [t0, t1], false);
            offset += 128;
        }
        let rem = data.len() - offset;
        let mut block = [0u8; 128];
        block[..rem].copy_from_slice(&data[offset..]);
        let (new_t0, carry) = t0.overflowing_add(rem as u64);
        t0 = new_t0;
        if carry {
            t1 = t1.wrapping_add(1);
        }
        blake2b_compress(&mut h, &block, [t0, t1], true);
    }

    h[0].to_le_bytes()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub op: Op,
    pub lit: Vec<u8>,
    pub kids: Vec<usize>,
    pub k: usize,
    pub key: [u8; 256],
    pub sel: usize,
    pub var: isize,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            op: Op::Lit,
            lit: Vec::new(),
            kids: Vec::new(),
            k: 1,
            key: IDENT,
            sel: 0,
            var: -1,
        }
    }
}

impl Node {
    pub fn lit(lit: Vec<u8>) -> Self {
        Self {
            op: Op::Lit,
            lit,
            ..Default::default()
        }
    }

    pub fn seq(kids: Vec<usize>) -> Self {
        Self {
            op: Op::Seq,
            kids,
            ..Default::default()
        }
    }

    pub fn rep(kid: usize, k: usize) -> Self {
        Self {
            op: Op::Rep,
            kids: vec![kid],
            k,
            ..Default::default()
        }
    }

    pub fn alt(kids: Vec<usize>, sel: usize) -> Self {
        Self {
            op: Op::Alt,
            kids,
            sel,
            ..Default::default()
        }
    }

    pub fn rec(kid: usize, key: [u8; 256]) -> Self {
        Self {
            op: Op::Rec,
            kids: vec![kid],
            key,
            ..Default::default()
        }
    }

    pub fn var(var: isize) -> Self {
        Self {
            op: Op::Var,
            var,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cause {
    pub nodes: Vec<Node>,
    pub root: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Intervention {
    Lit(Vec<u8>),
    Order(Vec<usize>),
    Count(usize),
    Sel(usize),
    Key([u8; 256]),
}

fn eval_node(
    i: usize,
    nodes: &[Node],
    do_map: Option<&HashMap<usize, Intervention>>,
    memo: &mut [Option<Vec<u8>>],
) -> Vec<u8> {
    if let Some(ref v) = memo[i] {
        return v.clone();
    }
    let n = &nodes[i];
    let iv = do_map.and_then(|m| m.get(&i));
    let v = match n.op {
        Op::Lit => match iv {
            Some(Intervention::Lit(bs)) => bs.clone(),
            _ => n.lit.clone(),
        },
        Op::Var => Vec::new(),
        Op::Seq => {
            let mut out = Vec::new();
            match iv {
                Some(Intervention::Order(order)) => {
                    for &t in order {
                        let k = n.kids[t];
                        out.extend_from_slice(&eval_node(k, nodes, do_map, memo));
                    }
                }
                _ => {
                    for &k in &n.kids {
                        out.extend_from_slice(&eval_node(k, nodes, do_map, memo));
                    }
                }
            }
            out
        }
        Op::Rep => {
            let kk = match iv {
                Some(Intervention::Count(cnt)) => *cnt,
                _ => n.k,
            };
            let child = eval_node(n.kids[0], nodes, do_map, memo);
            let mut out = Vec::with_capacity(child.len() * kk);
            for _ in 0..kk {
                out.extend_from_slice(&child);
            }
            out
        }
        Op::Alt => {
            if n.kids.is_empty() {
                Vec::new()
            } else {
                let s = match iv {
                    Some(Intervention::Sel(sel)) => *sel,
                    _ => n.sel,
                } % n.kids.len();
                eval_node(n.kids[s], nodes, do_map, memo)
            }
        }
        Op::Rec => {
            let key = match iv {
                Some(Intervention::Key(k)) => k,
                _ => &n.key,
            };
            let child = eval_node(n.kids[0], nodes, do_map, memo);
            child.iter().map(|&b| key[b as usize]).collect()
        }
    };
    memo[i] = Some(v.clone());
    v
}

/// Évaluation pure de TOUS les nœuds (nécessaire aux répertoires).
pub fn values(c: &Cause, do_map: Option<&HashMap<usize, Intervention>>) -> Vec<Vec<u8>> {
    let mut memo = vec![None; c.nodes.len()];
    if !c.nodes.is_empty() {
        eval_node(c.root, &c.nodes, do_map, &mut memo);
        for i in 0..c.nodes.len() {
            if memo[i].is_none() {
                eval_node(i, &c.nodes, do_map, &mut memo);
            }
        }
    }
    memo.into_iter().map(|x| x.unwrap_or_default()).collect()
}

fn walk_emit(
    i: usize,
    start: usize,
    nodes: &[Node],
    v: &[Vec<u8>],
    do_map: Option<&HashMap<usize, Intervention>>,
    occ: &mut [Vec<(usize, usize)>],
) {
    occ[i].push((start, start + v[i].len()));
    let n = &nodes[i];
    let iv = do_map.and_then(|m| m.get(&i));
    match n.op {
        Op::Seq => {
            let mut cur = start;
            match iv {
                Some(Intervention::Order(order)) => {
                    for &t in order {
                        let k = n.kids[t];
                        walk_emit(k, cur, nodes, v, do_map, occ);
                        cur += v[k].len();
                    }
                }
                _ => {
                    for &k in &n.kids {
                        walk_emit(k, cur, nodes, v, do_map, occ);
                        cur += v[k].len();
                    }
                }
            }
        }
        Op::Rep => {
            let kk = match iv {
                Some(Intervention::Count(cnt)) => *cnt,
                _ => n.k,
            };
            let mut cur = start;
            let k = n.kids[0];
            for _ in 0..kk {
                walk_emit(k, cur, nodes, v, do_map, occ);
                cur += v[k].len();
            }
        }
        Op::Alt => {
            if !n.kids.is_empty() {
                let s = match iv {
                    Some(Intervention::Sel(sel)) => *sel,
                    _ => n.sel,
                } % n.kids.len();
                walk_emit(n.kids[s], start, nodes, v, do_map, occ);
            }
        }
        Op::Rec => {
            walk_emit(n.kids[0], start, nodes, v, do_map, occ);
        }
        Op::Lit | Op::Var => {}
    }
}

/// Sortie + occurrences positionnelles de chaque nœud (purviews).
pub fn emit(
    c: &Cause,
    do_map: Option<&HashMap<usize, Intervention>>,
) -> (Vec<u8>, Vec<Vec<(usize, usize)>>) {
    let v = values(c, do_map);
    let mut occ = vec![Vec::new(); c.nodes.len()];
    if !c.nodes.is_empty() {
        walk_emit(c.root, 0, &c.nodes, &v, do_map, &mut occ);
    }
    let root_val = if c.nodes.is_empty() {
        Vec::new()
    } else {
        v[c.root].clone()
    };
    (root_val, occ)
}

/// Déduplication structurelle => le partage (term graph) est gratuit.
pub struct Builder {
    pub nodes: Vec<Node>,
    pub h: Vec<[u8; 8]>,
    pub idx: HashMap<[u8; 8], usize>,
    pub dedup: bool,
}

impl Builder {
    pub fn new(dedup: bool) -> Self {
        Self {
            nodes: Vec::new(),
            h: Vec::new(),
            idx: HashMap::new(),
            dedup,
        }
    }

    pub fn add(&mut self, n: Node) -> usize {
        let mut buf = Vec::new();
        match n.op {
            Op::Lit => {
                buf.push(b'L');
                buf.extend_from_slice(&n.lit);
            }
            Op::Var => {
                buf.push(b'V');
                buf.push(n.var.rem_euclid(256) as u8);
            }
            Op::Seq => {
                buf.push(b'Q');
                for (pos, &k) in n.kids.iter().enumerate() {
                    if pos > 0 {
                        buf.push(b'|');
                    }
                    buf.extend_from_slice(&self.h[k]);
                }
            }
            Op::Rep => {
                buf.push(b'R');
                buf.extend_from_slice(n.k.to_string().as_bytes());
                buf.push(b'|');
                buf.extend_from_slice(&self.h[n.kids[0]]);
            }
            Op::Alt => {
                buf.push(b'A');
                buf.push(n.sel as u8);
                for (pos, &k) in n.kids.iter().enumerate() {
                    if pos > 0 {
                        buf.push(b'|');
                    }
                    buf.extend_from_slice(&self.h[k]);
                }
            }
            Op::Rec => {
                buf.push(b'C');
                buf.extend_from_slice(&n.key);
                buf.extend_from_slice(&self.h[n.kids[0]]);
            }
        }
        let key = hash8(&buf);
        if self.dedup {
            if let Some(&existing) = self.idx.get(&key) {
                return existing;
            }
        }
        let i = self.nodes.len();
        self.nodes.push(n);
        self.h.push(key);
        self.idx.insert(key, i);
        i
    }

    pub fn cause(&self, root: usize) -> Cause {
        Cause {
            nodes: self.nodes.clone(),
            root,
        }
    }
}

pub fn indegrees(c: &Cause) -> Vec<usize> {
    let mut deg = vec![0usize; c.nodes.len()];
    for n in &c.nodes {
        for &k in &n.kids {
            deg[k] += 1;
        }
    }
    deg
}

/// Signature MÉTA-RELATIONNELLE : constantes effacées, topologie conservée.
pub fn signature(c: &Cause, rounds: usize) -> Vec<[u8; 8]> {
    let deg = indegrees(c);
    let mut lab: Vec<[u8; 8]> = c
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let b = match n.op {
                Op::Lit | Op::Var => "a",
                Op::Seq => "q",
                Op::Rep => "r",
                Op::Alt => "t",
                Op::Rec => "c",
            };
            let s = format!("{}{}:{}", b, deg[i].min(3), n.kids.len());
            hash8(s.as_bytes())
        })
        .collect();

    for _ in 0..rounds {
        let mut nxt = Vec::with_capacity(c.nodes.len());
        for (i, n) in c.nodes.iter().enumerate() {
            let mut buf = Vec::with_capacity(10 + n.kids.len() * 9);
            buf.extend_from_slice(&lab[i]);
            buf.push(b'(');
            for (pos, &k) in n.kids.iter().enumerate() {
                if pos > 0 {
                    buf.push(b',');
                }
                buf.extend_from_slice(&lab[k]);
            }
            buf.push(b')');
            nxt.push(hash8(&buf));
        }
        lab = nxt;
    }

    lab.sort();
    lab
}

/// Subsomption relationnelle : VAR lie un nœud, topologie de partage préservée.
pub fn unify(sc: &Cause, si: usize, tc: &Cause, ti: usize, abstract_lit: bool) -> bool {
    fn rec(
        a: usize,
        b: usize,
        sc: &Cause,
        tc: &Cause,
        abstract_lit: bool,
        binds: &mut HashMap<isize, usize>,
        pairs: &mut HashMap<(usize, usize), bool>,
    ) -> bool {
        if let Some(&cached) = pairs.get(&(a, b)) {
            return cached;
        }
        let na = &sc.nodes[a];
        let nb = &tc.nodes[b];
        let ok = if na.op == Op::Var {
            if let Some(&bound) = binds.get(&na.var) {
                bound == b
            } else {
                binds.insert(na.var, b);
                true
            }
        } else if na.op != nb.op || na.kids.len() != nb.kids.len() {
            false
        } else if na.op == Op::Lit {
            if abstract_lit {
                true
            } else {
                na.lit == nb.lit
            }
        } else {
            let mut sub_ok = na
                .kids
                .iter()
                .zip(nb.kids.iter())
                .all(|(&x, &y)| rec(x, y, sc, tc, abstract_lit, binds, pairs));
            if sub_ok && !abstract_lit {
                if na.op == Op::Rep {
                    sub_ok = na.k == nb.k;
                }
                if na.op == Op::Rec {
                    sub_ok = na.key == nb.key;
                }
            }
            sub_ok
        };
        pairs.insert((a, b), ok);
        ok
    }

    let mut binds = HashMap::new();
    let mut pairs = HashMap::new();
    rec(si, ti, sc, tc, abstract_lit, &mut binds, &mut pairs)
}

/// Partition IIT : coupe le partage de u en copies indépendantes.
pub fn sever(c: &Cause, u: usize) -> Cause {
    fn rec(
        i: usize,
        u: usize,
        c: &Cause,
        b: &mut Builder,
        memo: &mut HashMap<usize, usize>,
    ) -> usize {
        if i != u {
            if let Some(&cached) = memo.get(&i) {
                return cached;
            }
        }
        let n = &c.nodes[i];
        let kids: Vec<usize> = n.kids.iter().map(|&k| rec(k, u, c, b, memo)).collect();
        let j = b.add(Node {
            op: n.op,
            lit: n.lit.clone(),
            kids,
            k: n.k,
            key: n.key,
            sel: n.sel,
            var: n.var,
        });
        if i != u {
            memo.insert(i, j);
        }
        j
    }

    let mut b = Builder::new(false);
    let mut memo = HashMap::new();
    let root = rec(c.root, u, c, &mut b, &mut memo);
    b.cause(root)
}