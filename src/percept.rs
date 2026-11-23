//! Pôle régional R : ce que la perception de E déclenche dans le modèle.
//!
//! - champ de surprise octet-niveau sous le nul « octets i.i.d. à la fréquence de E » ;
//! - ancres : débuts de répétitions maximales (tableau de suffixes + LCP) ;
//! - canaux de relation entre ancres : identité, proximité avant, proximité arrière,
//!   isomorphie des motifs d'égalité.
//!
//! Tout est dérivé de E seul : aucun paramètre, aucun séparateur supposé.
//! Coûts : tableau de suffixes O(n log n) ; LCP et répétitions O(n) ;
//! relations O(n log n + somme des longueurs d'ancres).

const NONE: usize = usize::MAX;

/// Le nul : octets i.i.d. à la fréquence empirique de E.
#[derive(Clone, Debug)]
pub struct Null {
    pub n: usize,
    pub count: [usize; 256],
    /// Surprise d'un octet en bits, -log2(count/n) ; +inf si l'octet est absent de E.
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

/// Répétition maximale : sous-chaîne présente au moins deux fois, non extensible à gauche
/// ni à droite (gauche-diverse et droite-diverse ; début et fin de E comptent comme bornes
/// distinctes). Ses occurrences sont `sa[lb..=rb]`.
#[derive(Clone, Debug)]
pub struct Repeat {
    pub len: usize,
    pub lb: usize,
    pub rb: usize,
    /// Répétition maximale englobante la plus proche (préfixe strict, même début).
    pub parent: Option<usize>,
    /// Surprise du contenu sous le nul (somme du champ sur la répétition).
    pub bits: f64,
}

impl Repeat {
    pub fn count(&self) -> usize {
        self.rb - self.lb + 1
    }
}

/// Ancre : position où débute au moins une répétition maximale. `class` est la plus longue ;
/// les plus courtes débutant au même point sont `repeats[class].parent`, puis son parent, etc.
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
}

/// Arête orientée entre deux ancres (indices dans `Percept::anchors`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub w: f32,
}

/// Canaux de relation. Une relation symétrique figure dans les deux sens.
#[derive(Clone, Debug, Default)]
pub struct Relations {
    /// Occurrences successives (ordre du texte) d'une même répétition maximale ;
    /// w = surprise du contenu partagé, en bits.
    pub identity: Vec<Edge>,
    /// De l'ancre vers la première ancre qui débute à sa fin ou après ; w = écart en octets.
    pub forward: Vec<Edge>,
    /// De l'ancre vers l'ancre qui se termine au plus près avant son début (la plus longue
    /// à égalité) ; w = écart en octets.
    pub backward: Vec<Edge>,
    /// Ancres de contenus distincts, de même longueur, dont les motifs d'égalité sont
    /// isomorphes (même motif à renommage bijectif des octets près), chaînées dans l'ordre
    /// du texte ; w = nombre de positions répétées du motif.
    pub iso: Vec<Edge>,
}

impl Relations {
    pub fn channel(&self, c: Chan) -> &[Edge] {
        match c {
            Chan::Identity => self.identity.as_slice(),
            Chan::Forward => self.forward.as_slice(),
            Chan::Backward => self.backward.as_slice(),
            Chan::Iso => self.iso.as_slice(),
        }
    }
}

/// Perception de E : champ de surprise, tableau de suffixes, répétitions maximales,
/// ancres et relations.
#[derive(Clone, Debug)]
pub struct Percept {
    pub eff: Vec<u8>,
    pub null: Null,
    /// Champ de surprise : surprise[i] = null.bits[eff[i]].
    pub surprise: Vec<f32>,
    /// Sommes cumulées du champ : cum[i] = somme de surprise[0..i] ; longueur n + 1.
    pub cum: Vec<f64>,
    pub sa: Vec<usize>,
    /// lcp[r] = plus long préfixe commun de sa[r-1] et sa[r] ; lcp[0] = 0.
    pub lcp: Vec<usize>,
    /// Toutes les répétitions maximales.
    pub repeats: Vec<Repeat>,
    /// Ancres, par position strictement croissante.
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
        let mut anchors: Vec<Anchor> = Vec::new();
        for (p, &c) in owner.iter().enumerate() {
            if c != NONE {
                anchors.push(Anchor { pos: p, len: repeats[c].len, class: c });
            }
        }
        let rel = relate(eff, &sa, &repeats, &anchors);
        Percept { eff: eff.to_vec(), null, surprise, cum, sa, lcp, repeats, anchors, rel }
    }

    /// Surprise totale du champ sur [a, b).
    pub fn mass(&self, a: usize, b: usize) -> f64 {
        self.cum[b] - self.cum[a]
    }

    /// Positions des occurrences d'une répétition (ordre du tableau de suffixes).
    pub fn occurrences(&self, class: usize) -> &[usize] {
        let r = &self.repeats[class];
        &self.sa[r.lb..=r.rb]
    }

    pub fn content(&self, class: usize) -> &[u8] {
        window(&self.eff, &self.sa, &self.repeats[class])
    }
}

/// Tableau de suffixes par doublement de préfixes (tri par comptage), O(n log n).
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

/// LCP de Kasai, O(n).
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

/// Répétitions maximales par parcours ascendant des lcp-intervalles (droite-maximalité) et
/// test de diversité du caractère précédent (gauche-maximalité).
/// Renvoie les répétitions et, par position, la plus longue qui y débute (NONE sinon).
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
    // left[k] : octet qui précède le suffixe sa[k] ; 256 = début de E.
    let left: Vec<u16> = sa
        .iter()
        .map(|&p| if p == 0 { 256 } else { u16::from(eff[p - 1]) })
        .collect();
    // diff[k] : plus petit k' > k tel que left[k'] != left[k] (n si aucun).
    let mut diff = vec![n; n];
    for k in (0..n - 1).rev() {
        diff[k] = if left[k + 1] != left[k] { k + 1 } else { diff[k + 1] };
    }
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    // Répétitions déjà fermées, en attente de leur parent ; triées par lb croissant.
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
                continue; // non gauche-diverse : extensible à gauche, pas maximale
            }
            let id = repeats.len();
            let split = pending.partition_point(|&c| repeats[c].lb < b);
            // Les suffixes non couverts par une répétition fille débutent la plus longue
            // répétition maximale qui les contient : c'est celle-ci.
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

/// Nombre de positions dont l'octet a déjà paru plus tôt dans la fenêtre.
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

/// Empreinte du motif d'égalité (indices de première apparition) ; None s'il n'y a
/// aucune égalité interne, c'est-à-dire pas de motif.
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

/// Isomorphie exacte des motifs d'égalité : il existe une bijection d'octets de a vers b.
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

    // Identité : chaînage des occurrences successives de la même classe, dans les deux sens.
    let mut last = vec![NONE; repeats.len()];
    for (k, a) in anchors.iter().enumerate() {
        let j = last[a.class];
        if j != NONE {
            let w = repeats[a.class].bits as f32;
            rel.identity.push(Edge { from: j, to: k, w });
            rel.identity.push(Edge { from: k, to: j, w });
        }
        last[a.class] = k;
    }

    // Proximité avant : première ancre qui débute à la fin de l'ancre ou après.
    for (k, a) in anchors.iter().enumerate() {
        let e = a.end();
        let j = anchors.partition_point(|x| x.pos < e);
        if j < m {
            rel.forward.push(Edge { from: k, to: j, w: (anchors[j].pos - e) as f32 });
        }
    }

    // Proximité arrière : ancre qui se termine au plus près avant le début de l'ancre
    // (la plus longue à fin égale), par balayage sur les fins croissantes.
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

    // Isomorphie des motifs d'égalité. Ancres de chaque classe (CSR, indices croissants).
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
    // Classes portant des ancres et un motif, triées par (longueur, empreinte).
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
        // Vérification exacte des empreintes égales (aucune collision ne fusionne).
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
                rel.iso.push(Edge { from: p[0], to: p[1], w });
                rel.iso.push(Edge { from: p[1], to: p[0], w });
            }
        }
        i = e;
    }
    rel
}