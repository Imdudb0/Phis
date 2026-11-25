//! E = `perceived.eff`, octets bruts. Rien d'autre n'entre.
//!
//! CLAUSE D'ARCHITECTURE (elle porte l'invariance et l'auto-spécification) :
//! aucune valeur d'octet n'est jamais représentée comme identité apprise. La vue
//! ne contient que des RELATIONS : classes d'égalité, ordre, adjacence,
//! containment multi-échelle, écho de forme canonique. Toute feature est
//! invariante par renommage bijectif de l'alphabet, et l'émission se fait par
//! POINTEURS vers des positions de E (lot 4) : l'alphabet d'expression de la cause
//! est donc toujours fourni par l'entrée, jamais par un vocabulaire appris.
//!
//! Conséquence de test : `Perm` applied to E leaves the view bit-identical.
//! Le test d'invariance du harness est un détecteur de fuite de value-coding.
//!
//! Déterminisme : aucun HashMap n'est itéré. Toutes les structures ordonnées
//! sont des BTreeMap/BTreeSet ou des Vec triés par (poids desc, index asc).

use crate::rng::{fnv1a_u16, Rng};

/// Nombre de features relationnelles par nœud. Figé, pré-enregistré.
pub const F: usize = 16;

/// Types de relation. Chaque type recevra DEUX matrices apprises indépendantes
/// (avant / arrière), jamais transposées l'une de l'autre : non-réciprocité
/// structurelle du substrat (lot 2).
pub const REL_EQ: u8 = 0; // partage de classes d'octets
pub const REL_ADJ: u8 = 1; // contiguïté au même niveau
pub const REL_CONTAIN: u8 = 2; // parent / enfant multi-échelle
pub const REL_SIM: u8 = 3; // même forme canonique (écho structurel)
pub const REL_ORD: u8 = 4; // ordre à distance bucketisée
pub const REL_SELF: u8 = 5; // boucle propre
pub const REL_KINDS: usize = 6;

#[derive(Clone, Copy, Debug)]
pub struct ViewCfg {
    pub max_atoms: usize,
    pub fanout: [usize; 3],
    pub topk: usize,
    pub eq_cap: usize,
    pub ord_dists: [usize; 3],
}

impl Default for ViewCfg {
    fn default() -> Self {
        ViewCfg {
            max_atoms: 384,
            fanout: [4, 4, 4],
            topk: 6,
            eq_cap: 24,
            ord_dists: [2, 8, 32],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VNode {
    pub start: u32,
    pub len: u32,
    pub scale: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct REdge {
    pub src: u32,
    pub dst: u32,
    pub kind: u8,
    /// 0 : sens canonique (src < dst, ou boucle). 1 : sens inverse.
    /// Le substrat route vers W_fwd[kind] ou W_bwd[kind] selon ce bit.
    pub rev: u8,
    pub w: f32,
}

pub struct EffView {
    pub nb: usize,
    pub atom_len: usize,
    pub n_atoms: usize,
    pub nodes: Vec<VNode>,
    pub feat: Vec<f32>, // n * F
    pub edges: Vec<REdge>,
    pub out: Vec<Vec<u32>>, // nœud -> identifiants d'arêtes entrantes/sortantes locales
    pub level_start: Vec<usize>,
    pub level_count: Vec<usize>,
}

impl EffView {
    #[inline]
    pub fn n(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn span(&self, v: usize) -> (usize, usize) {
        let nd = self.nodes[v];
        (nd.start as usize, nd.start as usize + nd.len as usize)
    }

    #[inline]
    pub fn atom_of_pos(&self, pos: usize) -> usize {
        (pos / self.atom_len.max(1)).min(self.n_atoms.saturating_sub(1))
    }

    #[inline]
    pub fn features(&self, v: usize) -> &[f32] {
        &self.feat[v * F..(v + 1) * F]
    }
}

/// Renommage bijectif de l'alphabet. Outil de TEST uniquement : il n'apparaît
/// ni dans l'entraînement, ni dans le choix d'un checkpoint.
#[derive(Clone)]
pub struct Perm {
    pub fwd: [u8; 256],
    pub inv: [u8; 256],
}

impl Perm {
    pub fn identity() -> Self {
        let mut fwd = [0u8; 256];
        for i in 0..256 {
            fwd[i] = i as u8;
        }
        Perm { fwd, inv: fwd }
    }

    pub fn random(rng: &mut Rng) -> Self {
        let mut fwd = [0u8; 256];
        for i in 0..256 {
            fwd[i] = i as u8;
        }
        rng.shuffle(&mut fwd);
        let mut inv = [0u8; 256];
        for i in 0..256 {
            inv[fwd[i] as usize] = i as u8;
        }
        Perm { fwd, inv }
    }

    pub fn apply(&self, b: &[u8]) -> Vec<u8> {
        b.iter().map(|&x| self.fwd[x as usize]).collect()
    }

    pub fn unapply(&self, b: &[u8]) -> Vec<u8> {
        b.iter().map(|&x| self.inv[x as usize]).collect()
    }
}

/// Forme canonique locale : chaque octet devient son rang de première occurrence
/// DANS le span. Invariante par toute bijection d'alphabet.
pub fn canonical_pattern(b: &[u8]) -> Vec<u16> {
    let mut seen = [u16::MAX; 256];
    let mut next: u16 = 0;
    let mut out = Vec::with_capacity(b.len());
    for &x in b {
        let s = &mut seen[x as usize];
        if *s == u16::MAX {
            *s = next;
            next = next.saturating_add(1);
        }
        out.push(*s);
    }
    out
}

struct Aux {
    is_first: Vec<bool>,
    class_size: Vec<u32>,
    next_same: Vec<u32>,
    run: Vec<u32>,
}

fn build_aux(b: &[u8]) -> Aux {
    let n = b.len();
    let mut first_pos = [usize::MAX; 256];
    let mut counts = [0u32; 256];
    for (i, &x) in b.iter().enumerate() {
        counts[x as usize] += 1;
        if first_pos[x as usize] == usize::MAX {
            first_pos[x as usize] = i;
        }
    }
    let mut is_first = vec![false; n];
    let mut class_size = vec![0u32; n];
    for (i, &x) in b.iter().enumerate() {
        is_first[i] = first_pos[x as usize] == i;
        class_size[i] = counts[x as usize];
    }
    let mut next_same = vec![n as u32; n];
    let mut last = [n; 256];
    for i in (0..n).rev() {
        let x = b[i] as usize;
        next_same[i] = last[x] as u32;
        last[x] = i;
    }
    let mut run = vec![1u32; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && b[j] == b[i] {
            j += 1;
        }
        let l = (j - i) as u32;
        for k in i..j {
            run[k] = l;
        }
        i = j;
    }
    Aux { is_first, class_size, next_same, run }
}

/// Score de périodicité interne : meilleure fraction d'accords sur un décalage p.
fn period_score(b: &[u8], s: usize, e: usize) -> f32 {
    let len = e.saturating_sub(s);
    if len < 8 {
        return 0.0;
    }
    let pmax = (len / 2).min(32);
    let mut best = 0.0f32;
    for p in 1..=pmax {
        let cmp = len - p;
        if cmp < 4 {
            continue;
        }
        let mut hit = 0usize;
        for i in 0..cmp {
            if b[s + i] == b[s + i + p] {
                hit += 1;
            }
        }
        let sc = hit as f32 / cmp as f32;
        if sc > best {
            best = sc;
        }
    }
    best
}

pub fn build(bytes: &[u8], cfg: &ViewCfg) -> EffView {
    let nb = bytes.len();
    let atom_len = if nb == 0 {
        1
    } else {
        ((nb + cfg.max_atoms - 1) / cfg.max_atoms).max(1)
    };
    let n_atoms = if nb == 0 { 1 } else { (nb + atom_len - 1) / atom_len };

    // ---- hiérarchie ----
    let mut nodes: Vec<VNode> = Vec::new();
    let mut level_start: Vec<usize> = vec![0];
    let mut level_count: Vec<usize> = Vec::new();
    let mut contain: Vec<(u32, u32)> = Vec::new();

    for a in 0..n_atoms {
        let s = a * atom_len;
        let e = ((a + 1) * atom_len).min(nb);
        nodes.push(VNode { start: s as u32, len: e.saturating_sub(s) as u32, scale: 0 });
    }
    level_count.push(n_atoms);

    let mut prev_start = 0usize;
    let mut prev_count = n_atoms;
    for (li, &f) in cfg.fanout.iter().enumerate() {
        if prev_count <= 1 || f == 0 {
            break;
        }
        let start = nodes.len();
        let cnt = (prev_count + f - 1) / f;
        for k in 0..cnt {
            let a = prev_start + k * f;
            let b = (a + f).min(prev_start + prev_count);
            let s = nodes[a].start;
            let e = nodes[b - 1].start + nodes[b - 1].len;
            let parent = (start + k) as u32;
            nodes.push(VNode { start: s, len: e - s, scale: (li + 1) as u8 });
            for c in a..b {
                contain.push((parent, c as u32));
            }
        }
        level_start.push(start);
        level_count.push(cnt);
        prev_start = start;
        prev_count = cnt;
    }

    let n = nodes.len();
    let aux = build_aux(bytes);

    // ---- formes canoniques (passe 1) ----
    let mut shape_key: Vec<(u8, u32, u64)> = Vec::with_capacity(n);
    let mut shape_groups: std::collections::BTreeMap<(u8, u32, u64), Vec<u32>> =
        std::collections::BTreeMap::new();
    let mut classes: Vec<Vec<u8>> = Vec::with_capacity(n);
    for (vi, nd) in nodes.iter().enumerate() {
        let s = nd.start as usize;
        let e = (s + nd.len as usize).min(nb);
        let span = if e > s { &bytes[s..e] } else { &[][..] };
        let h = fnv1a_u16(&canonical_pattern(span));
        let key = (nd.scale, nd.len, h);
        shape_key.push(key);
        shape_groups.entry(key).or_default().push(vi as u32);
        let mut seen = [false; 256];
        let mut cl: Vec<u8> = Vec::new();
        for &x in span {
            if !seen[x as usize] {
                seen[x as usize] = true;
                cl.push(x);
            }
        }
        cl.sort_unstable();
        classes.push(cl);
    }

    // ---- features (passe 2) ----
    let mut feat = vec![0.0f32; n * F];
    let lnn = ((1 + nb) as f32).ln().max(1e-6);
    for (vi, nd) in nodes.iter().enumerate() {
        let s = nd.start as usize;
        let e = (s + nd.len as usize).min(nb);
        let len = e.saturating_sub(s);
        let lf = len.max(1) as f32;
        let base = vi * F;

        let distinct = classes[vi].len();
        let mut novelty = 0usize;
        let mut adj_eq = 0usize;
        let mut eqdist = 0.0f32;
        let mut maxrun = 0u32;
        let mut cohesion = 0usize;
        let mut clsum = 0.0f32;
        for i in s..e {
            if aux.is_first[i] {
                novelty += 1;
            }
            if i > s && bytes[i] == bytes[i - 1] {
                adj_eq += 1;
            }
            let ns = aux.next_same[i] as usize;
            let d = if ns < nb { ns - i } else { nb };
            eqdist += (d as f32) / (nb.max(1) as f32);
            if ns < e {
                cohesion += 1;
            }
            if aux.run[i] > maxrun {
                maxrun = aux.run[i];
            }
            clsum += ((1 + aux.class_size[i]) as f32).ln();
        }
        let reps = shape_groups.get(&shape_key[vi]).map(|v| v.len()).unwrap_or(1);
        let left_eq = if s > 0 && s < nb && bytes[s - 1] == bytes[s] { 1.0 } else { 0.0 };
        let right_eq =
            if e > 0 && e < nb && e > s && bytes[e - 1] == bytes[e] { 1.0 } else { 0.0 };

        feat[base + 0] = 1.0;
        feat[base + 1] = (s as f32) / (nb.max(1) as f32);
        feat[base + 2] = (len as f32) / (nb.max(1) as f32);
        feat[base + 3] = ((1 + len) as f32).ln() / 12.0;
        feat[base + 4] = (nd.scale as f32) / 3.0;
        feat[base + 5] = (distinct as f32) / lf;
        feat[base + 6] = (novelty as f32) / lf;
        feat[base + 7] = (adj_eq as f32) / (len.saturating_sub(1).max(1) as f32);
        feat[base + 8] = eqdist / lf;
        feat[base + 9] = (maxrun as f32) / lf;
        feat[base + 10] = period_score(bytes, s, e);
        feat[base + 11] = clsum / (lf * lnn);
        feat[base + 12] = ((reps as f32).ln() / ((n as f32).ln().max(1e-6))).min(1.0);
        feat[base + 13] = left_eq;
        feat[base + 14] = right_eq;
        feat[base + 15] = (cohesion as f32) / lf;
    }

    // ---- candidats de relation ----
    // (kind, autre, poids) ; tri déterministe, top-k par type.
    let mut cand: Vec<Vec<(u8, u32, f32)>> = vec![Vec::new(); n];
    let mut push_pair = |cand: &mut Vec<Vec<(u8, u32, f32)>>, a: u32, b: u32, k: u8, w: f32| {
        if a == b {
            return;
        }
        cand[a as usize].push((k, b, w));
        cand[b as usize].push((k, a, w));
    };

    // ADJ + ORD
    for l in 0..level_start.len() {
        let st = level_start[l];
        let ct = level_count[l];
        for i in 0..ct {
            let a = (st + i) as u32;
            if i + 1 < ct {
                push_pair(&mut cand, a, (st + i + 1) as u32, REL_ADJ, 1.0);
            }
            for &d in cfg.ord_dists.iter() {
                if i + d < ct {
                    let w = 1.0 / (1.0 + (d as f32).ln());
                    push_pair(&mut cand, a, (st + i + d) as u32, REL_ORD, w);
                }
            }
        }
    }

    // CONTAIN
    for &(p, c) in contain.iter() {
        push_pair(&mut cand, p, c, REL_CONTAIN, 1.0);
    }

    // SIM : chaîne déterministe à l'intérieur de chaque groupe de forme.
    for (_, group) in shape_groups.iter() {
        if group.len() < 2 {
            continue;
        }
        let cap = group.len().min(64);
        for i in 0..cap - 1 {
            push_pair(&mut cand, group[i], group[i + 1], REL_SIM, 1.0);
        }
    }

    // EQ : index inversé classe -> nœuds, échantillonnage régulier borné.
    let mut index: std::collections::BTreeMap<u8, Vec<u32>> = std::collections::BTreeMap::new();
    for (vi, cl) in classes.iter().enumerate() {
        for &x in cl.iter() {
            index.entry(x).or_default().push(vi as u32);
        }
    }
    let mut shared: Vec<std::collections::BTreeMap<u32, u32>> =
        (0..n).map(|_| std::collections::BTreeMap::new()).collect();
    for (_, list) in index.iter() {
        let sel: Vec<u32> = if list.len() <= cfg.eq_cap {
            list.clone()
        } else {
            let stride = list.len() / cfg.eq_cap;
            (0..cfg.eq_cap).map(|k| list[(k * stride).min(list.len() - 1)]).collect()
        };
        for i in 0..sel.len() {
            for j in (i + 1)..sel.len() {
                let a = sel[i].min(sel[j]);
                let b = sel[i].max(sel[j]);
                if a == b {
                    continue;
                }
                *shared[a as usize].entry(b).or_insert(0) += 1;
            }
        }
    }
    for a in 0..n {
        for (&b, &sh) in shared[a].iter() {
            let la = classes[a].len();
            let lb = classes[b as usize].len();
            let union = (la + lb).saturating_sub(sh as usize).max(1);
            let w = sh as f32 / union as f32;
            if w > 0.0 {
                push_pair(&mut cand, a as u32, b, REL_EQ, w);
            }
        }
    }

    // ---- sélection top-k, dédup, matérialisation dirigée ----
    let mut chosen: std::collections::BTreeSet<(u32, u32, u8, u32)> =
        std::collections::BTreeSet::new(); // (min, max, kind, bits poids)
    let mut weights: std::collections::BTreeMap<(u32, u32, u8), f32> =
        std::collections::BTreeMap::new();
    for a in 0..n {
        let v = &mut cand[a];
        v.sort_unstable_by(|x, y| {
            x.0.cmp(&y.0)
                .then(y.2.partial_cmp(&x.2).unwrap_or(std::cmp::Ordering::Equal))
                .then(x.1.cmp(&y.1))
        });
        let mut cur_kind = 255u8;
        let mut taken = 0usize;
        for &(k, b, w) in v.iter() {
            if k != cur_kind {
                cur_kind = k;
                taken = 0;
            }
            if taken >= cfg.topk {
                continue;
            }
            taken += 1;
            let lo = (a as u32).min(b);
            let hi = (a as u32).max(b);
            let key = (lo, hi, k);
            let e = weights.entry(key).or_insert(0.0);
            if w > *e {
                *e = w;
            }
            chosen.insert((lo, hi, k, (w.to_bits()) as u32));
        }
    }

    let mut edges: Vec<REdge> = Vec::new();
    for (&(lo, hi, k), &w) in weights.iter() {
        edges.push(REdge { src: lo, dst: hi, kind: k, rev: 0, w });
        edges.push(REdge { src: hi, dst: lo, kind: k, rev: 1, w });
    }
    for a in 0..n {
        edges.push(REdge { src: a as u32, dst: a as u32, kind: REL_SELF, rev: 0, w: 1.0 });
    }
    edges.sort_unstable_by(|x, y| {
        x.src.cmp(&y.src).then(x.kind.cmp(&y.kind)).then(x.dst.cmp(&y.dst)).then(x.rev.cmp(&y.rev))
    });

    let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (ei, e) in edges.iter().enumerate() {
        out[e.src as usize].push(ei as u32);
    }

    EffView {
        nb,
        atom_len,
        n_atoms,
        nodes,
        feat,
        edges,
        out,
        level_start,
        level_count,
    }
}

/// Rendu par pointeurs : l'alphabet vient de E, jamais d'un vocabulaire appris.
/// Utilisé par l'émission (lot 4) et par la générativité (lot 5).
pub fn render_atoms(bytes: &[u8], view: &EffView, atoms: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &a in atoms {
        let ai = (a as usize).min(view.n_atoms.saturating_sub(1));
        let s = ai * view.atom_len;
        let e = ((ai + 1) * view.atom_len).min(bytes.len());
        if e > s {
            out.extend_from_slice(&bytes[s..e]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<u8> {
        let mut v = Vec::new();
        for i in 0..600 {
            v.push(((i * 7) % 13) as u8);
            if i % 37 == 0 {
                v.extend_from_slice(&[9, 9, 9, 4, 4]);
            }
        }
        v
    }

    /// Témoin d'invariance EXACTE : la vue ne doit pas changer d'un bit sous
    /// renommage bijectif de l'alphabet. Toute fuite de value-coding casse ce test.
    #[test]
    fn invariance_renommage() {
        let b = sample();
        let cfg = ViewCfg::default();
        let v1 = build(&b, &cfg);
        let mut rng = Rng::new(20261125);
        for _ in 0..8 {
            let p = Perm::random(&mut rng);
            let b2 = p.apply(&b);
            let v2 = build(&b2, &cfg);
            assert_eq!(v1.nodes.len(), v2.nodes.len());
            assert_eq!(v1.edges.len(), v2.edges.len());
            assert_eq!(v1.feat, v2.feat, "fuite de valeur d'octet dans les features");
            for (e1, e2) in v1.edges.iter().zip(v2.edges.iter()) {
                assert_eq!((e1.src, e1.dst, e1.kind, e1.rev), (e2.src, e2.dst, e2.kind, e2.rev));
                assert_eq!(e1.w.to_bits(), e2.w.to_bits());
            }
        }
    }

    #[test]
    fn determinisme_construction() {
        let b = sample();
        let cfg = ViewCfg::default();
        let v1 = build(&b, &cfg);
        let v2 = build(&b, &cfg);
        assert_eq!(v1.feat, v2.feat);
        assert_eq!(v1.edges.len(), v2.edges.len());
    }

    #[test]
    fn entree_vide_survit() {
        let v = build(&[], &ViewCfg::default());
        assert!(v.n() >= 1);
        assert!(v.feat.iter().all(|x| x.is_finite()));
    }
}