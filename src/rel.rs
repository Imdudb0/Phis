use crate::phi::Dist;
#[derive(Clone, Debug)]
pub struct Relation {
    pub members: Vec<usize>,
    pub overlap: Vec<usize>,
    pub phi: f64,
    pub cross: bool,
}
struct Relatum {
    d: usize,
    z: Vec<usize>,
    lat: Vec<bool>,
    states: Vec<Vec<u16>>,
    unit: f64,
}
fn first_common(a: &Relatum, b: &Relatum) -> Option<usize> {
    for (i, &x) in a.z.iter().enumerate() {
        if a.lat[i] {
            if let Ok(k) = b.z.binary_search(&x) {
                if b.lat[k] {
                    return Some(x);
                }
            }
        }
    }
    None
}
fn congruent(a: &Relatum, b: &Relatum) -> Vec<usize> {
    let mut best: Vec<usize> = Vec::new();
    for sa in &a.states {
        for sb in &b.states {
            let mut o: Vec<usize> = Vec::new();
            for (i, &x) in a.z.iter().enumerate() {
                if !a.lat[i] {
                    continue;
                }
                if let Ok(k) = b.z.binary_search(&x) {
                    if b.lat[k] && sa[i] == sb[k] {
                        o.push(x);
                    }
                }
            }
            if o.len() > best.len() {
                best = o;
            }
        }
    }
    best
}
fn common_overlap(rel: &[Relatum], pick: &[(usize, usize)], x: usize, v: u16) -> Option<(Vec<usize>, f64)> {
    let mut sel: Vec<(&Relatum, &Vec<u16>)> = Vec::with_capacity(pick.len());
    for &(_, ri) in pick {
        let r = &rel[ri];
        let i = r.z.binary_search(&x).ok()?;
        let s = r.states.iter().find(|s| s[i] == v)?;
        sel.push((r, s));
    }
    let (r0, s0) = sel[0];
    let mut o: Vec<usize> = Vec::new();
    for (i, &y) in r0.z.iter().enumerate() {
        if !r0.lat[i] {
            continue;
        }
        let ok = sel[1..].iter().all(|(r, s)| match r.z.binary_search(&y) {
            Ok(k) => r.lat[k] && s[k] == s0[i],
            Err(_) => false,
        });
        if ok {
            o.push(y);
        }
    }
    if o.is_empty() {
        return None;
    }
    let u = sel.iter().map(|(r, _)| r.unit).fold(f64::INFINITY, f64::min);
    let phi = o.len() as f64 * u;
    Some((o, phi))
}
fn disjoint(foot: &[Vec<usize>], mem: &[usize]) -> bool {
    for p in 0..mem.len() {
        for q in (p + 1)..mem.len() {
            let (a, b) = (&foot[mem[p]], &foot[mem[q]]);
            if a.iter().any(|x| b.binary_search(x).is_ok()) {
                return false;
            }
        }
    }
    true
}
pub fn relations(dists: &[Dist], latent: &[bool], foot: &[Vec<usize>]) -> Vec<Relation> {
    let n = latent.len();
    let mut rel: Vec<Relatum> = Vec::new();
    for (di, d) in dists.iter().enumerate() {
        if !d.is_distinction() {
            continue;
        }
        for f in d.cause.iter().chain(d.effect.iter()) {
            let lat: Vec<bool> = f.nodes.iter().map(|&j| latent[j]).collect();
            if f.states.is_empty() || !lat.iter().any(|&b| b) {
                continue;
            }
            rel.push(Relatum {
                d: di,
                z: f.nodes.clone(),
                lat,
                states: f.states.clone(),
                unit: d.phi_d / f.nodes.len() as f64,
            });
        }
    }
    let mut by_node: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (ri, r) in rel.iter().enumerate() {
        for (i, &x) in r.z.iter().enumerate() {
            if r.lat[i] {
                by_node[x].push(ri);
            }
        }
    }
    let mut pairs: Vec<(usize, usize, f64, Vec<usize>)> = Vec::new();
    for x in 0..n {
        let lst = &by_node[x];
        for p in 0..lst.len() {
            for q in (p + 1)..lst.len() {
                let (a, b) = (&rel[lst[p]], &rel[lst[q]]);
                if a.d == b.d || first_common(a, b) != Some(x) {
                    continue;
                }
                let o = congruent(a, b);
                if o.is_empty() {
                    continue;
                }
                let phi = o.len() as f64 * a.unit.min(b.unit);
                pairs.push((a.d.min(b.d), a.d.max(b.d), phi, o));
            }
        }
    }
    pairs.sort_by(|u, v| {
        u.0.cmp(&v.0)
            .then(u.1.cmp(&v.1))
            .then(v.2.total_cmp(&u.2))
            .then_with(|| u.3.cmp(&v.3))
    });
    pairs.dedup_by(|later, first| later.0 == first.0 && later.1 == first.1);
    let mut cl: Vec<(Vec<usize>, f64, Vec<usize>)> = Vec::new();
    let mut trip: Vec<(u16, usize, usize)> = Vec::new();
    for x in 0..n {
        if by_node[x].len() < 3 {
            continue;
        }
        trip.clear();
        for &ri in &by_node[x] {
            let r = &rel[ri];
            if let Ok(i) = r.z.binary_search(&x) {
                for s in &r.states {
                    trip.push((s[i], r.d, ri));
                }
            }
        }
        trip.sort_unstable();
        let mut g = 0usize;
        while g < trip.len() {
            let v = trip[g].0;
            let mut h = g;
            while h < trip.len() && trip[h].0 == v {
                h += 1;
            }
            let mut pick: Vec<(usize, usize)> = Vec::new();
            for &(_, d, ri) in &trip[g..h] {
                let same = matches!(pick.last(), Some(c) if c.0 == d);
                if same {
                    if let Some(c) = pick.last_mut() {
                        if rel[ri].unit > rel[c.1].unit {
                            c.1 = ri;
                        }
                    }
                } else {
                    pick.push((d, ri));
                }
            }
            if pick.len() >= 3 {
                if let Some((o, phi)) = common_overlap(&rel, &pick, x, v) {
                    cl.push((pick.iter().map(|p| p.0).collect(), phi, o));
                }
            }
            g = h;
        }
    }
    cl.sort_by(|u, v| u.0.cmp(&v.0).then(v.1.total_cmp(&u.1)).then_with(|| u.2.cmp(&v.2)));
    cl.dedup_by(|later, first| later.0 == first.0);
    let mut out: Vec<Relation> = Vec::with_capacity(pairs.len() + cl.len());
    for (a, b, phi, o) in pairs {
        out.push(Relation { members: vec![a, b], overlap: o, phi, cross: disjoint(foot, &[a, b]) });
    }
    for (mem, phi, o) in cl {
        let cross = disjoint(foot, &mem);
        out.push(Relation { members: mem, overlap: o, phi, cross });
    }
    out
}