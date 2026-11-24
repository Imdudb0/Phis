const NONE: usize = usize::MAX;
#[derive(Clone, Debug)]
pub struct Null {
    pub n: usize,
    pub count: [usize; 256],
    pub bits: [f64; 256],
}
impl Null {
    pub fn of(eff: &[u8]) -> Null {
        let n = eff.len();
        let mut count = [0usize; 256];
        for &b in eff {
            count[b as usize] += 1;
        }
        let mut bits = [f64::INFINITY; 256];
        for b in 0..256 {
            if count[b] > 0 {
                bits[b] = (n as f64 / count[b] as f64).log2();
            }
        }
        Null { n, count, bits }
    }
}
#[derive(Clone, Debug)]
pub struct Repeat {
    pub len: usize,
    pub lb: usize,
    pub rb: usize,
    pub parent: Option<usize>,
    pub bits: f64,
}
impl Repeat {
    pub fn count(&self) -> usize {
        self.rb - self.lb + 1
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    pub pos: usize,
    pub len: usize,
    pub class: usize,
}
impl Anchor {
    pub fn end(&self) -> usize {
        self.pos + self.len
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chan {
    Identity,
    Forward,
    Backward,
    Iso,
    IsoBytes,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub w: f32,
}
#[derive(Clone, Debug, Default)]
pub struct Relations {
    pub identity: Vec<Edge>,
    pub forward: Vec<Edge>,
    pub backward: Vec<Edge>,
    /// isomorphie structurelle : cycles d'ancres au même motif de répétition de classes
    pub iso: Vec<Edge>,
    /// isomorphie sur les octets : répétitions égales à un renommage bijectif d'octets près
    pub iso_bytes: Vec<Edge>,
}
impl Relations {
    pub fn channel(&self, c: Chan) -> &[Edge] {
        match c {
            Chan::Identity => self.identity.as_slice(),
            Chan::Forward => self.forward.as_slice(),
            Chan::Backward => self.backward.as_slice(),
            Chan::Iso => self.iso.as_slice(),
            Chan::IsoBytes => self.iso_bytes.as_slice(),
        }
    }
}
#[derive(Clone, Debug)]
pub struct Percept {
    pub eff: Vec<u8>,
    pub null: Null,
    pub surprise: Vec<f32>,
    pub cum: Vec<f64>,
    pub sa: Vec<usize>,
    pub lcp: Vec<usize>,
    pub repeats: Vec<Repeat>,
    pub anchors: Vec<Anchor>,
    pub rel: Relations,
}
impl Percept {
    pub fn new(eff: &[u8]) -> Percept {
        let n = eff.len();
        let null = Null::of(eff);
        let surprise: Vec<f32> = eff.iter().map(|&b| null.bits[b as usize] as f32).collect();
        let mut cum: Vec<f64> = Vec::with_capacity(n + 1);
        let mut acc = 0.0f64;
        cum.push(acc);
        for &b in eff {
            acc += null.bits[b as usize];
            cum.push(acc);
        }
        let sa = suffix_array(eff);
        let lcp = lcp_array(eff, &sa);
        let (repeats, owner) = maximal_repeats(eff, &sa, &lcp, &cum);
        let thr = (repeats.len().max(2) as f64).log2(); // Chernoff : ≲ 1 fausse classe attendue
        let keep: Vec<bool> = repeats.iter().map(|r| chance_surprise(n, r) >= thr).collect();
        let mut anchors: Vec<Anchor> = Vec::new();
        for (p, &c) in owner.iter().enumerate() {
            if c != NONE && keep[c] {
                anchors.push(Anchor { pos: p, len: repeats[c].len, class: c });
            }
        }
        let rel = relate(eff, &sa, &repeats, &anchors);
        Percept { eff: eff.to_vec(), null, surprise, cum, sa, lcp, repeats, anchors, rel }
    }
    pub fn mass(&self, a: usize, b: usize) -> f64 {
        self.cum[b] - self.cum[a]
    }
    pub fn occurrences(&self, class: usize) -> &[usize] {
        let r = &self.repeats[class];
        &self.sa[r.lb..=r.rb]
    }
    pub fn content(&self, class: usize) -> &[u8] {
        window(&self.eff, &self.sa, &self.repeats[class])
    }
}
fn suffix_array(s: &[u8]) -> Vec<usize> {
    let n = s.len();
    let mut sa: Vec<usize> = (0..n).collect();
    if n < 2 {
        return sa;
    }
    sa.sort_unstable_by_key(|&i| s[i]);
    let mut rank = vec![0usize; n];
    for j in 1..n {
        rank[sa[j]] = rank[sa[j - 1]] + usize::from(s[sa[j]] != s[sa[j - 1]]);
    }
    let mut next = vec![0usize; n];
    let mut by_second = vec![0usize; n];
    let mut k = 1usize;
    while rank[sa[n - 1]] + 1 < n {
        let mut p = 0;
        for i in n.saturating_sub(k)..n {
            by_second[p] = i;
            p += 1;
        }
        for j in 0..n {
            if sa[j] >= k {
                by_second[p] = sa[j] - k;
                p += 1;
            }
        }
        let classes = rank[sa[n - 1]] + 1;
        let mut bucket = vec![0usize; classes + 1];
        for &r in &rank {
            bucket[r + 1] += 1;
        }
        for r in 0..classes {
            bucket[r + 1] += bucket[r];
        }
        for &i in &by_second {
            sa[bucket[rank[i]]] = i;
            bucket[rank[i]] += 1;
        }
        next[sa[0]] = 0;
        for j in 1..n {
            let (a, b) = (sa[j - 1], sa[j]);
            let ka = (rank[a], if a + k < n { rank[a + k] as isize } else { -1 });
            let kb = (rank[b], if b + k < n { rank[b + k] as isize } else { -1 });
            next[b] = next[a] + usize::from(ka != kb);
        }
        std::mem::swap(&mut rank, &mut next);
        k <<= 1;
    }
    sa
}
fn lcp_array(s: &[u8], sa: &[usize]) -> Vec<usize> {
    let n = s.len();
    let mut rank = vec![0usize; n];
    for (r, &i) in sa.iter().enumerate() {
        rank[i] = r;
    }
    let mut lcp = vec![0usize; n];
    let mut h = 0usize;
    for i in 0..n {
        let r = rank[i];
        if r == 0 {
            h = 0;
            continue;
        }
        let j = sa[r - 1];
        while i + h < n && j + h < n && s[i + h] == s[j + h] {
            h += 1;
        }
        lcp[r] = h;
        h = h.saturating_sub(1);
    }
    lcp
}
fn maximal_repeats(
    eff: &[u8],
    sa: &[usize],
    lcp: &[usize],
    cum: &[f64],
) -> (Vec<Repeat>, Vec<usize>) {
    let n = eff.len();
    let mut repeats: Vec<Repeat> = Vec::new();
    let mut owner = vec![NONE; n];
    if n < 2 {
        return (repeats, owner);
    }
    let left: Vec<u16> = sa
        .iter()
        .map(|&p| if p == 0 { 256 } else { u16::from(eff[p - 1]) })
        .collect();
    let mut diff = vec![n; n];
    for k in (0..n - 1).rev() {
        diff[k] = if left[k + 1] != left[k] { k + 1 } else { diff[k + 1] };
    }
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    let mut pending: Vec<usize> = Vec::new();
    for i in 1..=n {
        let cur = if i < n { lcp[i] } else { 0 };
        let mut lb = i - 1;
        loop {
            let (len, b) = stack[stack.len() - 1];
            if cur >= len {
                break;
            }
            stack.pop();
            lb = b;
            let rb = i - 1;
            if diff[b] > rb {
                continue;
            }
            let id = repeats.len();
            let split = pending.partition_point(|&c| repeats[c].lb < b);
            let mut cursor = b;
            for &c in &pending[split..] {
                for r in cursor..repeats[c].lb {
                    owner[sa[r]] = id;
                }
                cursor = repeats[c].rb + 1;
            }
            for r in cursor..=rb {
                owner[sa[r]] = id;
            }
            for &c in &pending[split..] {
                repeats[c].parent = Some(id);
            }
            pending.truncate(split);
            pending.push(id);
            let p = sa[b];
            repeats.push(Repeat { len, lb: b, rb, parent: None, bits: cum[p + len] - cum[p] });
        }
        if cur > stack[stack.len() - 1].0 {
            stack.push((cur, lb));
        }
    }
    (repeats, owner)
}
fn window<'a>(eff: &'a [u8], sa: &[usize], r: &Repeat) -> &'a [u8] {
    let p = sa[r.lb];
    &eff[p..p + r.len]
}
fn chance_surprise(n: usize, r: &Repeat) -> f64 {
    let lam = (n + 1 - r.len) as f64 * (-r.bits).exp2(); // occurrences attendues sous le nul
    let obs = r.count() as f64;
    if lam <= 0.0 {
        f64::INFINITY
    } else if obs <= lam {
        0.0
    } else {
        obs * (obs / lam).log2() - (obs - lam) * std::f64::consts::LOG2_E
    }
}
fn code(prev: &[usize], a: usize, t: usize) -> usize {
    let p = prev[t];
    if p != NONE && p >= a { t - p } else { 0 }
}
fn same_cycle(prev: &[usize], a1: usize, a2: usize, len: usize) -> bool {
    (0..len).all(|d| code(prev, a1, a1 + d) == code(prev, a2, a2 + d))
}
fn repeated_in(prev: &[usize], a: usize, b: usize) -> usize {
    (a..b).filter(|&t| code(prev, a, t) != 0).count()
}
fn repeated(w: &[u8]) -> usize {
    let mut seen = [false; 256];
    let mut r = 0;
    for &b in w {
        if seen[b as usize] {
            r += 1;
        } else {
            seen[b as usize] = true;
        }
    }
    r
}
fn pattern_hash(w: &[u8]) -> Option<u64> {
    let mut id = [u16::MAX; 256];
    let mut next = 0u16;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in w {
        let slot = &mut id[b as usize];
        if *slot == u16::MAX {
            *slot = next;
            next += 1;
        }
        h = (h ^ u64::from(*slot)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    if usize::from(next) < w.len() { Some(h) } else { None }
}
fn same_pattern(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut ia = [u16::MAX; 256];
    let mut ib = [u16::MAX; 256];
    let mut next = 0u16;
    for t in 0..a.len() {
        let (x, y) = (a[t] as usize, b[t] as usize);
        let (new_a, new_b) = (ia[x] == u16::MAX, ib[y] == u16::MAX);
        if new_a != new_b {
            return false;
        }
        if new_a {
            ia[x] = next;
            ib[y] = next;
            next += 1;
        } else if ia[x] != ib[y] {
            return false;
        }
    }
    true
}
fn relate(eff: &[u8], sa: &[usize], repeats: &[Repeat], anchors: &[Anchor]) -> Relations {
    let m = anchors.len();
    let mut rel = Relations::default();
    let mut prev = vec![NONE; m];
    let mut next = vec![NONE; m];
    let mut last = vec![NONE; repeats.len()];
    for (k, a) in anchors.iter().enumerate() {
        let j = last[a.class];
        if j != NONE {
            prev[k] = j;
            next[j] = k;
            let w = repeats[a.class].bits as f32;
            rel.identity.push(Edge { from: j, to: k, w });
            rel.identity.push(Edge { from: k, to: j, w });
        }
        last[a.class] = k;
    }
    for (k, a) in anchors.iter().enumerate() {
        let e = a.end();
        let j = anchors.partition_point(|x| x.pos < e);
        if j < m {
            rel.forward.push(Edge { from: k, to: j, w: (anchors[j].pos - e) as f32 });
        }
    }
    let mut by_end: Vec<usize> = (0..m).collect();
    by_end.sort_unstable_by_key(|&k| (anchors[k].end(), anchors[k].len));
    let mut j = 0;
    for (k, a) in anchors.iter().enumerate() {
        while j < m && anchors[by_end[j]].end() <= a.pos {
            j += 1;
        }
        if j > 0 {
            let b = by_end[j - 1];
            rel.backward.push(Edge { from: k, to: b, w: (a.pos - anchors[b].end()) as f32 });
        }
    }
    rel.iso = iso_edges(&prev, &next);
    rel.iso_bytes = iso_bytes_edges(eff, sa, repeats, anchors);
    rel
}
fn iso_edges(prev: &[usize], next: &[usize]) -> Vec<Edge> {
    const BASE: u64 = 0x9E37_79B9_7F4A_7C15;
    let m = prev.len();
    let mut out: Vec<Edge> = Vec::new();
    let mut q = vec![NONE; m + 1];
    for k in (0..m).rev() {
        q[k] = q[k + 1].min(next[k]);
    }
    let mut inv = BASE;
    for _ in 0..6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(BASE.wrapping_mul(inv)));
    }
    let mut pw = vec![1u64; m + 1];
    let mut ipw = vec![1u64; m + 1];
    for t in 1..=m {
        pw[t] = pw[t - 1].wrapping_mul(BASE);
        ipw[t] = ipw[t - 1].wrapping_mul(inv);
    }
    let mut fen = vec![0u64; m + 1];
    let mut keyed: Vec<(usize, u64, usize)> = Vec::new();
    for a in (0..m).rev() {
        let b = next[a];
        if b == NONE {
            continue;
        }
        let v = ((b - a) as u64).wrapping_mul(ipw[b]);
        let mut i = b + 1;
        while i <= m {
            fen[i] = fen[i].wrapping_add(v);
            i += i & i.wrapping_neg();
        }
        if q[a] < b {
            let mut s = 0u64;
            let mut j = b;
            while j > 0 {
                s = s.wrapping_add(fen[j]);
                j -= j & j.wrapping_neg();
            }
            keyed.push((b - a, pw[b - 1].wrapping_mul(s), a));
        }
    }
    keyed.sort_unstable();
    let mut i = 0;
    while i < keyed.len() {
        let (len, hash, _) = keyed[i];
        let mut e = i + 1;
        while e < keyed.len() && keyed[e].0 == len && keyed[e].1 == hash {
            e += 1;
        }
        if e - i >= 2 {
            let mut groups: Vec<Vec<usize>> = Vec::new();
            for &(_, _, a) in &keyed[i..e] {
                let home = groups.iter().position(|g| same_cycle(prev, g[0], a, len));
                match home {
                    Some(g) => groups[g].push(a),
                    None => groups.push(vec![a]),
                }
            }
            for g in &groups {
                if g.len() < 2 {
                    continue;
                }
                let w = repeated_in(prev, g[0], g[0] + len) as f32;
                for p in g.windows(2) {
                    out.push(Edge { from: p[0], to: p[1], w });
                    out.push(Edge { from: p[1], to: p[0], w });
                }
            }
        }
        i = e;
    }
    out
}
fn iso_bytes_edges(eff: &[u8], sa: &[usize], repeats: &[Repeat], anchors: &[Anchor]) -> Vec<Edge> {
    let m = anchors.len();
    let mut out: Vec<Edge> = Vec::new();
    let mut first = vec![0usize; repeats.len() + 1];
    for a in anchors {
        first[a.class + 1] += 1;
    }
    for c in 0..repeats.len() {
        first[c + 1] += first[c];
    }
    let mut fill = first.clone();
    let mut by_class = vec![0usize; m];
    for (k, a) in anchors.iter().enumerate() {
        by_class[fill[a.class]] = k;
        fill[a.class] += 1;
    }
    let mut keyed: Vec<(usize, u64, usize)> = Vec::new();
    for c in 0..repeats.len() {
        if first[c] == first[c + 1] {
            continue;
        }
        if let Some(h) = pattern_hash(window(eff, sa, &repeats[c])) {
            keyed.push((repeats[c].len, h, c));
        }
    }
    keyed.sort_unstable();
    let mut i = 0;
    while i < keyed.len() {
        let (len, hash, _) = keyed[i];
        let mut e = i + 1;
        while e < keyed.len() && keyed[e].0 == len && keyed[e].1 == hash {
            e += 1;
        }
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for &(_, _, c) in &keyed[i..e] {
            let wc = window(eff, sa, &repeats[c]);
            let home = groups
                .iter()
                .position(|g| same_pattern(window(eff, sa, &repeats[g[0]]), wc));
            match home {
                Some(g) => groups[g].push(c),
                None => groups.push(vec![c]),
            }
        }
        for g in groups {
            if g.len() < 2 {
                continue;
            }
            let w = repeated(window(eff, sa, &repeats[g[0]])) as f32;
            let mut members: Vec<usize> = Vec::new();
            for &c in &g {
                members.extend_from_slice(&by_class[first[c]..first[c + 1]]);
            }
            members.sort_unstable();
            for p in members.windows(2) {
                out.push(Edge { from: p[0], to: p[1], w });
                out.push(Edge { from: p[1], to: p[0], w });
            }
        }
        i = e;
    }
    out
}