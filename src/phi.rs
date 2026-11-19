//! Φ-lite : puissance cause-effet intégrée, interventionnelle, partitionnée.

use std::collections::{BTreeSet, HashMap};
use std::hash::Hash;

use crate::algebra::{emit, values, Cause, Intervention, Op, IDENT};
use crate::rng::PyRandom;

#[derive(Clone, Debug)]
pub struct SourceDetail {
    pub node: usize,
    pub occ: usize,
    pub eps: f64,
    pub iota: f64,
}

#[derive(Clone, Debug)]
pub struct PhiInfo {
    pub faithful: bool,
    pub phi: f64,
    pub sources: Vec<SourceDetail>,
    pub occ: Vec<Vec<(usize, usize)>>,
}

fn next_permutation(a: &mut [usize]) -> bool {
    if a.len() < 2 {
        return false;
    }
    let mut i = a.len() - 1;
    while i > 0 && a[i - 1] >= a[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = a.len() - 1;
    while a[j] <= a[i - 1] {
        j -= 1;
    }
    a.swap(i - 1, j);
    a[i..].reverse();
    true
}

fn first_k_permutations(n: usize, limit: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut p: Vec<usize> = (0..n).collect();
    loop {
        out.push(p.clone());
        if out.len() >= limit {
            break;
        }
        if !next_permutation(&mut p) {
            break;
        }
    }
    out
}

pub fn state_spaces(c: &Cause, _extra_keys: &[[u8; 256]]) -> Vec<(usize, Vec<Intervention>)> {
    let mut out = Vec::new();
    for (i, n) in c.nodes.iter().enumerate() {
        let vs: Vec<Intervention> = match n.op {
            Op::Lit => {
                let mut lits = vec![n.lit.clone()];
                if !n.lit.is_empty() {
                    let mut rev = n.lit.clone();
                    rev.reverse();
                    let mut rot = n.lit[1..].to_vec();
                    rot.push(n.lit[0]);
                    for cand in [rev, rot, Vec::new()] {
                        if !lits.contains(&cand) {
                            lits.push(cand);
                        }
                    }
                }
                lits.into_iter().map(Intervention::Lit).collect()
            }
            Op::Seq if n.kids.len() >= 2 => first_k_permutations(n.kids.len(), 6)
                .into_iter()
                .map(Intervention::Order)
                .collect(),
            Op::Rep => {
                let mut s = BTreeSet::from([1usize, 2, 3, n.k]);
                let mut res = Vec::new();
                while let Some(v) = s.pop_first() {
                    res.push(Intervention::Count(v));
                    if res.len() == 4 {
                        break;
                    }
                }
                res
            }
            Op::Alt => (0..n.kids.len()).map(Intervention::Sel).collect(),
                    Op::Rec => {
            // P21 : {clé du mécanisme, identité} = « ce recodage porte-t-il un pouvoir causal ? »
            // Aucune clé étrangère venue de E ⇒ Φ(c,E) ne dépend plus de l'alphabet global.
            // Cardinalité ≤ 2, commensurable avec les alternatives LIT (rev/rot/vide).
            let mut keys = vec![n.key];
            if !keys.contains(&IDENT) { keys.push(IDENT); }
            keys.into_iter().map(Intervention::Key).collect()
        }
            _ => Vec::new(),
        };
        if vs.len() > 1 {
            out.push((i, vs));
        }
    }
    out
}

fn entropy<T: Eq + Hash>(counts: &HashMap<T, usize>, n: usize) -> f64 {
    let nf = n as f64;
    let mut h = 0.0;
    for &c in counts.values() {
        if c > 0 {
            let p = c as f64 / nf;
            h -= p * p.log2();
        }
    }
    h
}

/// Information multi (KL joint || produit des marginales), en bits.
fn mi(cols: &[Vec<Option<Vec<u8>>>], n: usize) -> f64 {
    if cols.len() < 2 || n == 0 {
        return 0.0;
    }
    let mut joint: HashMap<Vec<&Option<Vec<u8>>>, usize> = HashMap::new();
    for r in 0..n {
        let tup: Vec<&Option<Vec<u8>>> = cols.iter().map(|col| &col[r]).collect();
        *joint.entry(tup).or_insert(0) += 1;
    }
    let hj = entropy(&joint, n);

    let mut hs = 0.0;
    for col in cols {
        let mut cnt: HashMap<&Option<Vec<u8>>, usize> = HashMap::new();
        for x in col {
            *cnt.entry(x).or_insert(0) += 1;
        }
        hs += entropy(&cnt, n);
    }
    (hs - hj).max(0.0)
}

pub fn phi(
    c: &Cause,
    e: &[u8],
    rng: &mut PyRandom,
    n: usize,
    extra_keys: &[[u8; 256]],
    nnull: usize,
) -> (f64, PhiInfo) {
    let (out0, occ0) = emit(c, None);
    if out0.as_slice() != e {
        return (
            0.0,
            PhiInfo {
                faithful: false,
                phi: 0.0,
                sources: Vec::new(),
                occ: occ0,
            },
        );
    }

    let sp = state_spaces(c, extra_keys);
    let sp_map: HashMap<usize, &[Intervention]> =
        sp.iter().map(|(i, vs)| (*i, vs.as_slice())).collect();

    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let mut do_map = HashMap::with_capacity(sp.len());
        for (i, vs) in &sp {
            let pick = rng.randrange(vs.len());
            do_map.insert(*i, vs[pick].clone());
        }
        samples.push(emit(c, Some(&do_map)));
    }

    let mut total = 0.0;
    let mut details = Vec::new();
    for (u, spans) in occ0.iter().enumerate() {
        if spans.len() < 2 {
            continue;
        }
        let eps = if let Some(&vs) = sp_map.get(&u) {
            let mut changed = 0usize;
            for s in vs {
                let mut do_one = HashMap::with_capacity(1);
                do_one.insert(u, s.clone());
                if emit(c, Some(&do_one)).0.as_slice() != e {
                    changed += 1;
                }
            }
            changed as f64 / vs.len() as f64
        } else {
            0.0
        };

        if eps == 0.0 {
            details.push(SourceDetail {
                node: u,
                occ: spans.len(),
                eps: 0.0,
                iota: 0.0,
            });
            continue;
        }

        let mut cols: Vec<Vec<Option<Vec<u8>>>> = vec![Vec::with_capacity(n); spans.len()];
        for (out, occ_s) in &samples {
            let ss = &occ_s[u];
            if ss.len() != spans.len() {
                for col in &mut cols {
                    col.push(None);
                }
            } else {
                for (t, &(a, b)) in ss.iter().enumerate() {
                    cols[t].push(Some(out[a..b].to_vec()));
                }
            }
        }

        let raw = mi(&cols, n);
        let mut null = 0.0;
        for _ in 0..nnull {
            let mut sh = cols.clone();
            for col in &mut sh {
                rng.shuffle(col);
            }
            null += mi(&sh, n);
        }
        let iota = (raw - null / (nnull.max(1) as f64)).max(0.0);
        total += eps * iota;
        details.push(SourceDetail {
            node: u,
            occ: spans.len(),
            eps,
            iota,
        });
    }

    (
        total,
        PhiInfo {
            faithful: true,
            phi: total,
            sources: details,
            occ: occ0,
        },
    )
}

/// Paramètres sans pouvoir causal, et alternatives observationnellement équivalentes.
pub fn underdetermined(
    c: &Cause,
    e: &[u8],
) -> (Vec<(usize, Vec<usize>)>, Vec<(usize, Vec<usize>)>) {
    let mut undet = Vec::new();
    let mut open_alt = Vec::new();
    for (i, n) in c.nodes.iter().enumerate() {
        if n.op == Op::Alt && n.kids.len() >= 2 {
            let eq: Vec<usize> = (0..n.kids.len())
                .filter(|&s| {
                    let mut do_map = HashMap::with_capacity(1);
                    do_map.insert(i, Intervention::Sel(s));
                    values(c, Some(&do_map))[c.root].as_slice() == e
                })
                .collect();
            if eq.len() >= 2 {
                open_alt.push((i, eq));
            }
        }
        if n.op == Op::Rep {
            let eq: Vec<usize> = (1..=4usize)
                .filter(|&k| {
                    let mut do_map = HashMap::with_capacity(1);
                    do_map.insert(i, Intervention::Count(k));
                    values(c, Some(&do_map))[c.root].as_slice() == e
                })
                .collect();
            if eq.len() >= 2 {
                undet.push((i, eq));
            }
        }
    }
    (undet, open_alt)
}