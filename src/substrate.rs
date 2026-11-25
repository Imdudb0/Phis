//! Substrat connexioniste : graphe de nœuds, matrices FONDAMENTALEMENT
//! NON-RÉCIPROQUES. Pour chaque type de relation k, deux blocs indépendants
//! W[k][0] et W[k][1] ; aucun n'est la transposée de l'autre, aucun n'est lié.
//!
//! « Au lieu de calculer brutalement la structure vue, on calcule par relation » :
//! on agrège les sources pondérées PAR TYPE de relation, puis on transforme.
//! Linéarité : Σ_e w_e·W_g·h_src = W_g·(Σ_e w_e·h_src) — une matmul par type,
//! pas une par arête. Le budget se compte en `rel` sans truquer le calcul.
//!
//! ÉQUIVARIANCE : aucun paramètre indexé par une valeur d'octet, aucun embedding
//! de position appris. Les features viennent de `eff` (invariantes), les slots
//! d'agrégation sont ordonnés par des relations. Le renommage bijectif de
//! l'alphabet laisse donc la dynamique IDENTIQUE, bit pour bit.
//!
//! LECTURE : la tête générative score des ARÊTES (candidats relationnels), pas un
//! vocabulaire. L'alphabet d'expression vient de E. Le programme de pointeurs
//! n'est qu'une MESURE (argmax) de l'état ; il n'est jamais manipulé par la
//! recherche, qui n'agit que sur des activations.

use std::collections::BTreeMap;

use crate::ad::{Params, Tape};
use crate::eff::{canonical_pattern, EffView, REL_KINDS};
use crate::rng::{fnv1a_u16, Rng};

pub const H: usize = 24;
pub const Q: usize = 4;
pub const N_OPS: usize = 8;
pub const P: usize = 24;
pub const GSTATS: usize = 8;
pub const NW: usize = REL_KINDS * 2;

// blocs de paramètres — ordre FIGÉ, consigné dans le manifeste de version
pub const B_WIN: usize = 0;
pub const B_BIN: usize = 1;
pub const B_W: usize = 2; // 2 .. 2+NW-1 : W[kind][rev]
pub const B_WU: usize = 14;
pub const B_BU: usize = 15;
pub const B_WG: usize = 16;
pub const B_BG: usize = 17;
pub const B_WE: usize = 18;
pub const B_WEP: usize = 19;
pub const B_BE: usize = 20;
pub const B_WC: usize = 21;
pub const B_WCP: usize = 22;
pub const B_BC: usize = 23;
pub const B_UP: usize = 24;
pub const B_VP: usize = 25;
pub const B_PB: usize = 26;
pub const B_WPI: usize = 27;
pub const B_BPI: usize = 28;
pub const B_PROP: usize = 29;
pub const B_MECH: usize = 30;
pub const B_RES: usize = 31;
pub const NBLOCKS: usize = 32;

pub fn init(rng: &mut Rng) -> Params {
    let mut p = Params::default();
    p.glorot(crate::eff::F, H, rng); // 0
    p.zeros(1, H); // 1
    for _ in 0..NW {
        p.glorot(H, H, rng); // 2..13, indépendants par construction
    }
    p.glorot(H, H, rng); // 14
    p.zeros(1, H); // 15
    p.glorot(2 * H, H, rng); // 16
    p.zeros(1, H); // 17
    p.glorot(H, Q, rng); // 18
    p.glorot(H, Q, rng); // 19
    p.zeros(1, Q); // 20
    p.glorot(H, Q, rng); // 21
    p.glorot(H, Q, rng); // 22
    p.zeros(1, Q); // 23
    p.glorot(H, H, rng); // 24
    p.glorot(H, H, rng); // 25
    p.zeros(1, P + 1); // 26
    p.glorot(H + GSTATS, N_OPS, rng); // 27
    p.zeros(1, N_OPS); // 28
    p.glorot(H, 1, rng); // 29
    p.glorot(H, 1, rng); // 30
    p.glorot(H, 1, rng); // 31
    assert_eq!(p.nblocks(), NBLOCKS);
    p
}

// ---------------------------------------------------------------- câblage

#[derive(Clone)]
pub struct Slot {
    pub src: Vec<usize>,
    pub w: Vec<f32>,
    pub eid: Vec<usize>,
}

#[derive(Clone)]
pub struct Group {
    pub slots: Vec<Slot>,
}

pub struct Wiring {
    pub n: usize,
    pub na: usize,
    pub groups: Vec<Group>,
    pub inc: Vec<Vec<usize>>,
    pub ptr_src: Vec<usize>, // na * P
    pub ptr_w: Vec<f32>,
    pub ptr_eid: Vec<usize>,
    pub rel_count: usize,
}

#[inline]
fn edge_lt(v: &EffView, a: usize, b: usize) -> std::cmp::Ordering {
    let ea = v.edges[a];
    let eb = v.edges[b];
    eb.w
        .partial_cmp(&ea.w)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(ea.kind.cmp(&eb.kind))
        .then(ea.src.cmp(&eb.src))
        .then(a.cmp(&b))
}

pub fn wire(v: &EffView) -> Wiring {
    let n = v.n();
    let na = v.n_atoms;
    let mut inc: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (ei, e) in v.edges.iter().enumerate() {
        inc[e.dst as usize].push(ei);
    }
    for l in inc.iter_mut() {
        l.sort_unstable_by(|&a, &b| edge_lt(v, a, b));
    }

    let mut groups: Vec<Group> = Vec::with_capacity(NW);
    let mut rel_count = 0usize;
    for gid in 0..NW {
        let kind = (gid / 2) as u8;
        let rev = (gid % 2) as u8;
        let mut per: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut dmax = 0usize;
        for d in 0..n {
            for &ei in inc[d].iter() {
                let e = v.edges[ei];
                if e.kind == kind && e.rev == rev {
                    per[d].push(ei);
                }
            }
            dmax = dmax.max(per[d].len());
        }
        let mut slots = Vec::with_capacity(dmax);
        for s in 0..dmax {
            let mut src = vec![0usize; n];
            let mut w = vec![0.0f32; n];
            let mut eid = vec![usize::MAX; n];
            for d in 0..n {
                if s < per[d].len() {
                    let ei = per[d][s];
                    src[d] = v.edges[ei].src as usize;
                    w[d] = v.edges[ei].w;
                    eid[d] = ei;
                }
            }
            rel_count += 1;
            slots.push(Slot { src, w, eid });
        }
        groups.push(Group { slots });
    }

    let mut ptr_src = vec![0usize; na * P];
    let mut ptr_w = vec![0.0f32; na * P];
    let mut ptr_eid = vec![usize::MAX; na * P];
    for a in 0..na {
        for k in 0..P.min(inc[a].len()) {
            let ei = inc[a][k];
            ptr_src[a * P + k] = v.edges[ei].src as usize;
            ptr_w[a * P + k] = v.edges[ei].w;
            ptr_eid[a * P + k] = ei;
        }
    }

    Wiring { n, na, groups, inc, ptr_src, ptr_w, ptr_eid, rel_count }
}

pub fn gstats(v: &EffView) -> [f32; GSTATS] {
    let mut s = [0.0f32; GSTATS];
    let tot = v.edges.len().max(1) as f32;
    for e in v.edges.iter() {
        s[e.kind as usize] += 1.0;
    }
    for k in 0..REL_KINDS {
        s[k] /= tot;
    }
    s[6] = (tot / v.n().max(1) as f32) / 32.0;
    s[7] = (v.n() as f32) / 512.0;
    s
}

// ---------------------------------------------------------------- intervention

/// Toute intervention de φ (partition, réfutation, déplacement, abstraction)
/// s'exprime ici : gain par arête, gain par groupe, nombre de tours.
/// Elle ne touche JAMAIS aux paramètres ni à l'entrée.
#[derive(Clone)]
pub struct Itv {
    pub edge_gain: Option<Vec<f32>>,
    pub group_gain: [f32; NW],
    pub rounds: usize,
}

impl Default for Itv {
    fn default() -> Self {
        Itv { edge_gain: None, group_gain: [1.0; NW], rounds: 2 }
    }
}

impl Itv {
    pub fn cut_edges(&self, v: &EffView, keep: impl Fn(usize, usize) -> bool) -> Itv {
        let mut g = vec![1.0f32; v.edges.len()];
        for (ei, e) in v.edges.iter().enumerate() {
            let base = self.edge_gain.as_ref().map(|x| x[ei]).unwrap_or(1.0);
            g[ei] = if keep(e.src as usize, e.dst as usize) { base } else { 0.0 };
        }
        Itv { edge_gain: Some(g), group_gain: self.group_gain, rounds: self.rounds }
    }
}

// ---------------------------------------------------------------- dynamique

/// Agrégation relationnelle restreinte à des cibles. C'est le cœur de φ :
/// un mécanisme est un petit sous-ensemble, son purview se calcule ici sans
/// relancer toute la dynamique.
pub fn aggregate(t: &mut Tape, p: &Params, h: usize, w: &Wiring, targets: &[usize], itv: &Itv) -> usize {
    let k = targets.len();
    if k == 0 {
        return t.input(1, H, &vec![0.0; H]);
    }
    let ones_h = t.input(1, H, &vec![1.0; H]);
    let mut total: Option<usize> = None;
    for gid in 0..NW {
        let g = &w.groups[gid];
        if g.slots.is_empty() || itv.group_gain[gid] == 0.0 {
            continue;
        }
        let mut acc = t.input(k, H, &vec![0.0; k * H]);
        let mut any = false;
        for slot in g.slots.iter() {
            let mut idx = vec![0usize; k];
            let mut wv = vec![0.0f32; k];
            for (i, &d) in targets.iter().enumerate() {
                let eid = slot.eid[d];
                if eid == usize::MAX {
                    continue;
                }
                let eg = itv.edge_gain.as_ref().map(|x| x[eid]).unwrap_or(1.0);
                let gain = slot.w[d] * eg * itv.group_gain[gid];
                if gain != 0.0 {
                    idx[i] = slot.src[d];
                    wv[i] = gain;
                }
            }
            if wv.iter().all(|&x| x == 0.0) {
                continue;
            }
            any = true;
            let hs = t.gather_rows(h, &idx);
            let wc = t.input(k, 1, &wv);
            let outer = t.matmul(wc, ones_h);
            let sc = t.mul(hs, outer);
            acc = t.add(acc, sc);
        }
        if !any {
            continue;
        }
        let wg = t.param(p, B_W + gid);
        let mg = t.matmul(acc, wg);
        total = Some(match total {
            None => mg,
            Some(a) => t.add(a, mg),
        });
    }
    total.unwrap_or_else(|| t.input(k, H, &vec![0.0; k * H]))
}

pub fn step(t: &mut Tape, p: &Params, h: usize, w: &Wiring, itv: &Itv) -> usize {
    let targets: Vec<usize> = (0..w.n).collect();
    let m = aggregate(t, p, h, w, &targets, itv);
    let wu = t.param(p, B_WU);
    let bu = t.param(p, B_BU);
    let mm = t.matmul(m, wu);
    let mb = t.add_row(mm, bu);
    let u = t.tanh(mb);

    let wg = t.param(p, B_WG);
    let rows_h: Vec<usize> = (0..H).collect();
    let rows_m: Vec<usize> = (H..2 * H).collect();
    let wg_h = t.gather_rows(wg, &rows_h);
    let wg_m = t.gather_rows(wg, &rows_m);
    let g1 = t.matmul(h, wg_h);
    let g2 = t.matmul(m, wg_m);
    let gm = t.add(g1, g2);
    let bg = t.param(p, B_BG);
    let gb = t.add_row(gm, bg);
    let gate = t.sigmoid(gb);

    let one = t.input(w.n, H, &vec![1.0; w.n * H]);
    let inv = t.sub(one, gate);
    let a = t.mul(gate, h);
    let b = t.mul(inv, u);
    let s = t.add(a, b);
    t.layer_norm_rows(s)
}

pub struct Fwd {
    pub h0: usize,
    pub hprev: usize,
    pub h: usize,
    pub hnext: usize,
    pub rep_e: usize, // (n,Q) répertoire d'effet
    pub rep_c: usize, // (n,Q) répertoire de cause
    pub mech: usize,  // (n,1) appartenance au mécanisme
    pub prop: usize,  // (n,1) logits de proposition de site
    pub res: usize,   // (n,1) saillance de résidu
    pub ptr_logits: usize, // (na,P+1) slot 0 = NULL
    pub ptr: usize,   // (na,P+1) distribution
    pub policy: usize, // (1,N_OPS)
}

pub fn forward(t: &mut Tape, p: &Params, v: &EffView, w: &Wiring, itv: &Itv) -> Fwd {
    assert!(itv.rounds >= 1, "il faut au moins un tour pour avoir un passé");
    let n = w.n;
    let x = t.input(n, crate::eff::F, &v.feat);
    let win = t.param(p, B_WIN);
    let bin = t.param(p, B_BIN);
    let xm = t.matmul(x, win);
    let xb = t.add_row(xm, bin);
    let xt = t.tanh(xb);
    let h0 = t.layer_norm_rows(xt);

    let mut states = vec![h0];
    let mut h = h0;
    for _ in 0..(itv.rounds + 1) {
        h = step(t, p, h, w, itv);
        states.push(h);
    }
    let hprev = states[itv.rounds - 1];
    let hcur = states[itv.rounds];
    let hnext = states[itv.rounds + 1];

    let we = t.param(p, B_WE);
    let wep = t.param(p, B_WEP);
    let be = t.param(p, B_BE);
    let e1 = t.matmul(hcur, we);
    let e2 = t.matmul(hnext, wep);
    let es = t.add(e1, e2);
    let eb = t.add_row(es, be);
    let rep_e = t.softmax_rows(eb);

    let wc = t.param(p, B_WC);
    let wcp = t.param(p, B_WCP);
    let bc = t.param(p, B_BC);
    let c1 = t.matmul(hcur, wc);
    let c2 = t.matmul(hprev, wcp);
    let cs = t.add(c1, c2);
    let cb = t.add_row(cs, bc);
    let rep_c = t.softmax_rows(cb);

    let wm = t.param(p, B_MECH);
    let ml = t.matmul(hcur, wm);
    let mech = t.sigmoid(ml);
    let wp = t.param(p, B_PROP);
    let prop = t.matmul(hcur, wp);
    let wr = t.param(p, B_RES);
    let rl = t.matmul(hcur, wr);
    let res = t.sigmoid(rl);

    // tête générative : scores d'ARÊTES, jamais de vocabulaire
    let na = w.na;
    let atoms: Vec<usize> = (0..na).collect();
    let ha = t.gather_rows(hcur, &atoms);
    let up = t.param(p, B_UP);
    let vp = t.param(p, B_VP);
    let uh = t.matmul(ha, up);
    let vh = t.matmul(hcur, vp);
    let mut logits = t.input(na, P + 1, &vec![0.0; na * (P + 1)]);
    let inv_sqrt = 1.0 / (H as f32).sqrt();
    for k in 0..P {
        let idx: Vec<usize> = (0..na).map(|a| w.ptr_src[a * P + k]).collect();
        let wv: Vec<f32> = (0..na).map(|a| w.ptr_w[a * P + k]).collect();
        let vs = t.gather_rows(vh, &idx);
        let pr = t.mul(uh, vs);
        let d = t.sum_rows(pr);
        let wc2 = t.input(na, 1, &wv);
        let dw = t.mul(d, wc2);
        let ds = t.scale(dw, inv_sqrt);
        let mut oh = vec![0.0f32; P + 1];
        oh[k + 1] = 1.0;
        let ohv = t.input(1, P + 1, &oh);
        let col = t.matmul(ds, ohv);
        logits = t.add(logits, col);
    }
    let pb = t.param(p, B_PB);
    let ptr_logits = t.add_row(logits, pb);
    let ptr = t.softmax_rows(ptr_logits);

    let poolw = t.input(1, n, &vec![1.0 / n.max(1) as f32; n]);
    let pooled = t.matmul(poolw, hcur);
    let gsv = t.input(1, GSTATS, &gstats(v));
    let wpi = t.param(p, B_WPI);
    let rows_h: Vec<usize> = (0..H).collect();
    let rows_g: Vec<usize> = (H..H + GSTATS).collect();
    let wpi_h = t.gather_rows(wpi, &rows_h);
    let wpi_g = t.gather_rows(wpi, &rows_g);
    let p1 = t.matmul(pooled, wpi_h);
    let p2 = t.matmul(gsv, wpi_g);
    let pl = t.add(p1, p2);
    let bpi = t.param(p, B_BPI);
    let policy = t.add_row(pl, bpi);

    Fwd { h0, hprev, h: hcur, hnext, rep_e, rep_c, mech, prop, res, ptr_logits, ptr, policy }
}

/// Répertoire local d'un purview, sous intervention. Utilisé par φ pour le
/// whole et pour chaque partition, à mécanisme fixé.
pub fn local_rep(
    t: &mut Tape,
    p: &Params,
    h: usize,
    w: &Wiring,
    targets: &[usize],
    itv: &Itv,
    effect: bool,
) -> usize {
    let m = aggregate(t, p, h, w, targets, itv);
    let (wa, bb) = if effect { (B_WE, B_BE) } else { (B_WC, B_BC) };
    let wv = t.param(p, wa);
    let bv = t.param(p, bb);
    let mm = t.matmul(m, wv);
    let mb = t.add_row(mm, bv);
    t.softmax_rows(mb)
}

// ---------------------------------------------------------------- programme

pub type ShapeKey = (u8, u32, u64);

pub fn shape_keys(bytes: &[u8], v: &EffView) -> Vec<ShapeKey> {
    v.nodes
        .iter()
        .map(|nd| {
            let s = nd.start as usize;
            let e = (s + nd.len as usize).min(bytes.len());
            let span = if e > s { &bytes[s..e] } else { &[][..] };
            (nd.scale, nd.len, fnv1a_u16(&canonical_pattern(span)))
        })
        .collect()
}

/// Règle relationnelle : « pour un atome de cette forme canonique, la source est
/// la rank-ième arête entrante de type (kind,rev) ». Transférable à un E′ inédit :
/// c'est ce qui rend la Projection mesurable.
#[derive(Clone, Copy, Debug)]
pub struct PtrRule {
    pub tgt: ShapeKey,
    pub kind: u8,
    pub rev: u8,
    pub rank: u8,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub rules: Vec<PtrRule>,
    pub order: Vec<usize>,
}

fn rank_of(v: &EffView, w: &Wiring, a: usize, eid: usize) -> u8 {
    let e = v.edges[eid];
    let mut l: Vec<usize> = w.inc[a]
        .iter()
        .cloned()
        .filter(|&x| v.edges[x].kind == e.kind && v.edges[x].rev == e.rev)
        .collect();
    l.sort_unstable_by(|&x, &y| edge_lt(v, x, y));
    l.iter().position(|&x| x == eid).unwrap_or(0).min(255) as u8
}

/// MESURE de l'état, pas une manipulation symbolique. argmax, aucun paramètre
/// de sortie appris sur du NL, aucun rendu de H supervisé.
pub fn read_program(t: &Tape, f: &Fwd, bytes: &[u8], v: &EffView, w: &Wiring) -> Program {
    let keys = shape_keys(bytes, v);
    let row = P + 1;
    let mut rules = Vec::new();
    for a in 0..w.na {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for k in 0..row {
            let x = t.at(f.ptr, a * row + k);
            if x > bv {
                bv = x;
                best = k;
            }
        }
        if best == 0 {
            continue; // NULL : cet atome n'est pas régénéré
        }
        let eid = w.ptr_eid[a * P + (best - 1)];
        if eid == usize::MAX {
            continue;
        }
        let e = v.edges[eid];
        rules.push(PtrRule { tgt: keys[a], kind: e.kind, rev: e.rev, rank: rank_of(v, w, a, eid) });
    }
    Program { rules, order: (0..w.na).collect() }
}

/// Exécution du programme sur un effet quelconque. Sert à la Générativité (E),
/// à la Projection (E′ inédi) et à la Discrimination (E″ d'une autre cause).
/// Retourne (octets régénérés, couverture).
pub fn execute(bytes: &[u8], v: &EffView, w: &Wiring, prog: &Program) -> (Vec<u8>, f32) {
    let keys = shape_keys(bytes, v);
    let mut by_shape: BTreeMap<ShapeKey, PtrRule> = BTreeMap::new();
    for r in prog.rules.iter() {
        by_shape.entry(r.tgt).or_insert(*r);
    }
    let mut out = Vec::new();
    let mut covered = 0usize;
    for &a in prog.order.iter() {
        if a >= w.na {
            continue;
        }
        let r = match by_shape.get(&keys[a]) {
            Some(r) => *r,
            None => continue,
        };
        let mut l: Vec<usize> = w.inc[a]
            .iter()
            .cloned()
            .filter(|&x| v.edges[x].kind == r.kind && v.edges[x].rev == r.rev)
            .collect();
        l.sort_unstable_by(|&x, &y| edge_lt(v, x, y));
        let ri = r.rank as usize;
        if ri >= l.len() {
            continue;
        }
        let s = v.edges[l[ri]].src as usize;
        let (bs, be) = v.span(s);
        let be = be.min(bytes.len());
        if be > bs {
            out.extend_from_slice(&bytes[bs..be]);
            covered += 1;
        }
    }
    (out, covered as f32 / w.na.max(1) as f32)
}

/// Le témoin « couverture 1 / intégration 0 » : H := E. Il doit être EXPRESSIBLE
/// (sinon le témoin est truqué) et φ doit le classer au plancher. Sa
/// discrimination est nulle : il régénère aussi bien l'effet d'une autre cause.
pub fn identity_program(bytes: &[u8], v: &EffView) -> Program {
    let keys = shape_keys(bytes, v);
    let mut rules = Vec::new();
    for a in 0..v.n_atoms {
        rules.push(PtrRule { tgt: keys[a], kind: crate::eff::REL_SELF, rev: 0, rank: 0 });
    }
    Program { rules, order: (0..v.n_atoms).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eff::{build, Perm, ViewCfg};
    use crate::rng::Rng;

    fn sample() -> Vec<u8> {
        let mut v = Vec::new();
        for i in 0..400 {
            v.push(((i * 5) % 11) as u8);
            if i % 29 == 0 {
                v.extend_from_slice(&[3, 3, 8, 8, 1]);
            }
        }
        v
    }

    #[test]
    fn non_reciprocite_structurelle() {
        let mut rng = Rng::new(5);
        let p = init(&mut rng);
        for k in 0..REL_KINDS {
            let a = &p.data[B_W + 2 * k];
            let b = &p.data[B_W + 2 * k + 1];
            assert_eq!(a.len(), b.len());
            assert_ne!(a, b, "les deux sens doivent être des blocs indépendants");
        }
    }

    /// Invariance EXACTE de la dynamique : si ce test casse, une valeur d'octet
    /// a fuité quelque part dans le chemin appris.
    #[test]
    fn invariance_dynamique() {
        let b = sample();
        let cfg = ViewCfg::default();
        let mut rng = Rng::new(9);
        let p = init(&mut rng);
        let v1 = build(&b, &cfg);
        let w1 = wire(&v1);
        let mut t1 = Tape::new();
        let f1 = forward(&mut t1, &p, &v1, &w1, &Itv::default());
        let pr1 = read_program(&t1, &f1, &b, &v1, &w1);

        let perm = Perm::random(&mut rng);
        let b2 = perm.apply(&b);
        let v2 = build(&b2, &cfg);
        let w2 = wire(&v2);
        let mut t2 = Tape::new();
        let f2 = forward(&mut t2, &p, &v2, &w2, &Itv::default());
        let pr2 = read_program(&t2, &f2, &b2, &v2, &w2);

        for i in 0..v1.n_atoms * (P + 1) {
            assert!((t1.at(f1.ptr, i) - t2.at(f2.ptr, i)).abs() < 1e-6);
        }
        assert_eq!(pr1.rules.len(), pr2.rules.len());
        for (a, c) in pr1.rules.iter().zip(pr2.rules.iter()) {
            assert_eq!((a.kind, a.rev, a.rank), (c.kind, c.rev, c.rank));
            assert_eq!(a.tgt, c.tgt);
        }
        let (o1, _) = execute(&b, &v1, &w1, &pr1);
        let (o2, _) = execute(&b2, &v2, &w2, &pr2);
        assert_eq!(perm.unapply(&o2), o1, "sortie équivalente après permutation inverse");
    }

    #[test]
    fn partition_coupe_tout() {
        let b = sample();
        let v = build(&b, &ViewCfg::default());
        let w = wire(&v);
        let mut rng = Rng::new(4);
        let p = init(&mut rng);
        let mut t = Tape::new();
        let f = forward(&mut t, &p, &v, &w, &Itv::default());
        let itv = Itv::default().cut_edges(&v, |_, _| false);
        let targets: Vec<usize> = (0..8.min(w.n)).collect();
        let m = aggregate(&mut t, &p, f.h, &w, &targets, &itv);
        assert!(t.val(m).iter().all(|&x| x == 0.0));
    }

    /// Le témoin identité doit être expressible et de couverture 1 : sinon le
    /// test qui doit tuer φ_sharp est truqué d'avance.
    #[test]
    fn temoin_identite_couverture_un() {
        let b = sample();
        let v = build(&b, &ViewCfg::default());
        let w = wire(&v);
        let id = identity_program(&b, &v);
        let (out, cov) = execute(&b, &v, &w, &id);
        assert!(cov > 0.99, "couverture {cov}");
        assert_eq!(out, b);
    }

    #[test]
    fn formes_du_graphe() {
        use crate::ad::Shape;
        let b = sample();
        let v = build(&b, &ViewCfg::default());
        let w = wire(&v);
        let mut rng = Rng::new(13);
        let p = init(&mut rng);
        let mut t = Tape::new();
        let f = forward(&mut t, &p, &v, &w, &Itv::default());
        assert_eq!(t.shape(f.h), Shape { r: w.n, c: H });
        assert_eq!(t.shape(f.rep_e), Shape { r: w.n, c: Q });
        assert_eq!(t.shape(f.rep_c), Shape { r: w.n, c: Q });
        assert_eq!(t.shape(f.ptr), Shape { r: w.na, c: P + 1 });
        assert_eq!(t.shape(f.policy), Shape { r: 1, c: N_OPS });
        assert_eq!(t.shape(f.prop), Shape { r: w.n, c: 1 });
        assert_eq!(t.shape(f.mech), Shape { r: w.n, c: 1 });
    }
}