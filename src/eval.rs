//! Harness. HORS ENTRAÎNEMENT : rien ici ne sert à entraîner, régler un
//! hyperparamètre ni choisir un checkpoint. Chaque jeu n'est vu qu'une fois
//! par empreinte de modèle (registre). Toute entrée du modèle passe par
//! `ask --model ... --file perceived.eff`, sans gabarit ni consigne.
//!
//! A se lit dans la dynamique (diagnostics). B se lit sur stdout, sans accès à A.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::ad::Tape;
use crate::eff::{self, EffView, Perm, ViewCfg};
use crate::emit;
use crate::model::{Ledger, Model};
use crate::phi;
use crate::ptgrs::{self, Cause, FamilySpec, MiniDict};
use crate::rng::{fnv1a, tag, Rng};
use crate::substrate::{self, Itv, Program, PtrRule, ShapeKey, Wiring};

// ---------------------------------------------------------------- pré-enregistrement

#[derive(Clone, Debug)]
pub struct Prereg {
    pub date: String,
    pub n_calib: usize,
    pub n_ood: usize,
    pub n_dict: usize,
    pub probes: usize,
    pub copy_win_max: f32,
    pub spearman_min: f32,
    pub random_phi_ratio_max: f32,
    pub mip_sweep: Vec<usize>,
    pub budget0_a_max: f32,
    pub ablation_drop_min: f32,
    pub residue_gain_min: f32,
    pub b_mrr_min: f32,
    pub b_margin_min: f32,
    pub shuffle_collapse_min: f32,
    pub a_min: f32,
    pub raw: String,
}

impl Default for Prereg {
    fn default() -> Self {
        Prereg {
            date: "2026-11-25".into(),
            n_calib: 24,
            n_ood: 16,
            n_dict: 12,
            probes: 8,
            copy_win_max: 0.05,
            spearman_min: 0.30,
            random_phi_ratio_max: 0.60,
            mip_sweep: vec![2, 6, 12],
            budget0_a_max: 0.15,
            ablation_drop_min: 0.10,
            residue_gain_min: 0.0,
            b_mrr_min: 0.45,
            b_margin_min: 0.10,
            shuffle_collapse_min: 0.15,
            a_min: 0.35,
            raw: String::new(),
        }
    }
}

impl Prereg {
    pub fn load(path: &str) -> io::Result<Prereg> {
        let mut p = Prereg::default();
        let raw = fs::read_to_string(path)?;
        p.raw = raw.clone();
        for line in raw.lines() {
            let l = line.split('#').next().unwrap_or("").trim();
            if l.is_empty() || l.starts_with('[') {
                continue;
            }
            let (k, v) = match l.split_once('=') {
                Some(x) => x,
                None => continue,
            };
            let k = k.trim();
            let v = v.trim().trim_matches('"');
            let us = |s: &str| s.parse::<usize>().unwrap_or(0);
            let fl = |s: &str| s.parse::<f32>().unwrap_or(0.0);
            match k {
                "date" => p.date = v.to_string(),
                "n_calib" => p.n_calib = us(v),
                "n_ood" => p.n_ood = us(v),
                "n_dict" => p.n_dict = us(v),
                "probes" => p.probes = us(v),
                "copy_win_max" => p.copy_win_max = fl(v),
                "spearman_min" => p.spearman_min = fl(v),
                "random_phi_ratio_max" => p.random_phi_ratio_max = fl(v),
                "mip_sweep" => p.mip_sweep = v.split(',').map(|x| us(x.trim())).collect(),
                "budget0_a_max" => p.budget0_a_max = fl(v),
                "ablation_drop_min" => p.ablation_drop_min = fl(v),
                "residue_gain_min" => p.residue_gain_min = fl(v),
                "b_mrr_min" => p.b_mrr_min = fl(v),
                "b_margin_min" => p.b_margin_min = fl(v),
                "shuffle_collapse_min" => p.shuffle_collapse_min = fl(v),
                "a_min" => p.a_min = fl(v),
                _ => {}
            }
        }
        Ok(p)
    }

    pub fn fingerprint(&self) -> u64 {
        if self.raw.is_empty() {
            fnv1a(format!("{:?}", self).as_bytes())
        } else {
            fnv1a(self.raw.as_bytes())
        }
    }

    pub fn write_default(path: &str) -> io::Result<()> {
        let p = Prereg::default();
        let s = format!(
            "date = \"{}\"\nn_calib = {}\nn_ood = {}\nn_dict = {}\nprobes = {}\n\
             copy_win_max = {}\nspearman_min = {}\nrandom_phi_ratio_max = {}\n\
             mip_sweep = \"2,6,12\"\nbudget0_a_max = {}\nablation_drop_min = {}\n\
             residue_gain_min = {}\nb_mrr_min = {}\nb_margin_min = {}\n\
             shuffle_collapse_min = {}\na_min = {}\n",
            p.date, p.n_calib, p.n_ood, p.n_dict, p.probes, p.copy_win_max, p.spearman_min,
            p.random_phi_ratio_max, p.budget0_a_max, p.ablation_drop_min, p.residue_gain_min,
            p.b_mrr_min, p.b_margin_min, p.shuffle_collapse_min, p.a_min
        );
        fs::write(path, s)
    }
}

// ---------------------------------------------------------------- ask + diagnostics

#[derive(Clone, Debug)]
pub struct TraceRow {
    pub step: usize,
    pub ops: usize,
    pub rel: u64,
    pub sharp: f32,
    pub stab: f32,
    pub cov: f32,
    pub op: u8,
}

#[derive(Clone, Debug, Default)]
pub struct Diag {
    pub fp: u64,
    pub ops: usize,
    pub rel: u64,
    pub phi: f32,
    pub stab: f32,
    pub cov: f32,
    pub admissible: bool,
    pub nspans: usize,
    pub trace: Vec<TraceRow>,
    pub residue: Vec<(usize, f32)>,
    pub rules: Vec<PtrRule>,
    pub ablate: String,
}

fn parse_diag(s: &str) -> Diag {
    let mut d = Diag::default();
    for line in s.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let g = |i: usize| f.get(i).copied().unwrap_or("");
        let uf = |i: usize| g(i).parse::<f32>().unwrap_or(0.0);
        let uu = |i: usize| g(i).parse::<usize>().unwrap_or(0);
        match g(0) {
            "meta" => match g(1) {
                "fp" => d.fp = g(2).parse().unwrap_or(0),
                "ops" => d.ops = uu(2),
                "rel" => d.rel = g(2).parse().unwrap_or(0),
                "phi" => d.phi = uf(2),
                "stab" => d.stab = uf(2),
                "cov" => d.cov = uf(2),
                "admissible" => d.admissible = uu(2) != 0,
                "nspans" => d.nspans = uu(2),
                _ => {}
            },
            "ablate" => d.ablate = g(1).to_string(),
            "trace" => d.trace.push(TraceRow {
                step: uu(1),
                ops: uu(2),
                rel: g(3).parse().unwrap_or(0),
                sharp: uf(4),
                stab: uf(6),
                cov: uf(7),
                op: g(8).parse().unwrap_or(255),
            }),
            "residue" => d.residue.push((uu(1), uf(2))),
            "rule" => d.rules.push(PtrRule {
                tgt: (g(1).parse().unwrap_or(0), g(2).parse().unwrap_or(0), g(3).parse().unwrap_or(0)),
                kind: g(4).parse().unwrap_or(0),
                rev: g(5).parse().unwrap_or(0),
                rank: g(6).parse().unwrap_or(0),
            }),
            _ => {}
        }
    }
    d
}

pub struct AskRun {
    pub out: Vec<u8>,
    pub diag: Diag,
    pub status: i32,
}

pub fn ask(model: &str, e: &[u8], ablate: &str, work: &Path) -> io::Result<AskRun> {
    fs::create_dir_all(work)?;
    let fpath = work.join("perceived.eff");
    fs::write(&fpath, e)?;
    let dpath = work.join("trace.tsv");
    let mp = Path::new(model).canonicalize()?;
    let mut c = Command::new(std::env::current_exe()?);
    c.arg("ask").arg("--model").arg(&mp).arg("--file").arg(&fpath);
    c.env("PHISHARP_TRACE", &dpath);
    if ablate.is_empty() {
        c.env_remove("PHISHARP_ABLATE");
    } else {
        c.env("PHISHARP_ABLATE", ablate);
    }
    let o = c.output()?;
    let d = if dpath.exists() { parse_diag(&fs::read_to_string(&dpath)?) } else { Diag::default() };
    Ok(AskRun { out: o.stdout, diag: d, status: o.status.code().unwrap_or(-1) })
}

fn prog_of(d: &Diag, na: usize) -> Program {
    Program { rules: d.rules.clone(), order: (0..na).collect() }
}

// ---------------------------------------------------------------- outils

fn edge_cmp(v: &EffView, a: usize, b: usize) -> Ordering {
    let ea = v.edges[a];
    let eb = v.edges[b];
    eb.w.partial_cmp(&ea.w)
        .unwrap_or(Ordering::Equal)
        .then(ea.kind.cmp(&eb.kind))
        .then(ea.src.cmp(&eb.src))
        .then(a.cmp(&b))
}

fn cand_list(v: &EffView, w: &Wiring, a: usize) -> Vec<(usize, u8, u8, u8)> {
    let mut l = w.inc[a].clone();
    l.sort_unstable_by(|&x, &y| edge_cmp(v, x, y));
    let mut seen: BTreeMap<(u8, u8), u8> = BTreeMap::new();
    let mut out = Vec::new();
    for eid in l {
        let e = v.edges[eid];
        let r = seen.entry((e.kind, e.rev)).or_insert(0);
        out.push((eid, e.kind, e.rev, *r));
        *r = r.saturating_add(1);
    }
    out
}

fn topk(s: &[f32], k: usize) -> Vec<usize> {
    let mut i: Vec<usize> = (0..s.len()).filter(|&j| s[j] > 0.0).collect();
    i.sort_unstable_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap_or(Ordering::Equal).then(a.cmp(&b)));
    i.truncate(k);
    i
}

fn mean(x: &[f32]) -> f32 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f32>() / x.len() as f32
    }
}

fn spearman(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 3 {
        return 0.0;
    }
    let rank = |x: &[f32]| -> Vec<f32> {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_unstable_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(Ordering::Equal).then(i.cmp(&j)));
        let mut r = vec![0.0f32; n];
        for (pos, &i) in idx.iter().enumerate() {
            r[i] = pos as f32;
        }
        r
    };
    let ra = rank(a);
    let rb = rank(b);
    let ma = mean(&ra);
    let mb = mean(&rb);
    let mut num = 0.0f32;
    let mut da = 0.0f32;
    let mut db = 0.0f32;
    for i in 0..n {
        let x = ra[i] - ma;
        let y = rb[i] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 0.0 || db <= 0.0 {
        0.0
    } else {
        num / (da.sqrt() * db.sqrt())
    }
}

/// φ du substitut évalué sur une cause DONNÉE. Sert à l'étalonnage : on injecte
/// le programme comme mécanisme et on lit la puissance intégrée. Observateur
/// seul : cette fonction n'est jamais appelée par l'entraînement.
pub fn phi_of_program(m: &Model, bytes: &[u8], v: &EffView, w: &Wiring, prog: &Program) -> f32 {
    let keys = substrate::shape_keys(bytes, v);
    let mut by: BTreeMap<ShapeKey, PtrRule> = BTreeMap::new();
    for r in prog.rules.iter() {
        by.entry(r.tgt).or_insert(*r);
    }
    let mut g = vec![0.0f32; v.edges.len()];
    for a in 0..w.na {
        let r = match by.get(&keys[a]) {
            Some(r) => *r,
            None => continue,
        };
        let mut l: Vec<usize> = w.inc[a]
            .iter()
            .cloned()
            .filter(|&x| v.edges[x].kind == r.kind && v.edges[x].rev == r.rev)
            .collect();
        l.sort_unstable_by(|&x, &y| edge_cmp(v, x, y));
        if (r.rank as usize) < l.len() {
            g[l[r.rank as usize]] = 1.0;
        }
    }
    let mut t = Tape::new();
    let f = substrate::forward(&mut t, &m.params, v, w, &Itv::default());
    let mut mech = phi::build_mech(&t, &f, v, w, &m.search.phi);
    let n = v.n();
    let mut score = vec![0.0f32; w.na];
    for (ei, e) in v.edges.iter().enumerate() {
        if g[ei] > 0.0 && (e.dst as usize) < w.na {
            score[e.dst as usize] += g[ei];
        }
    }
    let mm = topk(&score, m.search.phi.mech);
    let mut in_m = vec![false; n];
    for &i in mm.iter() {
        in_m[i] = true;
    }
    let mut we = vec![0.0f32; n];
    let mut wc = vec![0.0f32; n];
    for (ei, e) in v.edges.iter().enumerate() {
        if g[ei] <= 0.0 {
            continue;
        }
        if in_m[e.src as usize] {
            we[e.dst as usize] += g[ei];
        }
        if in_m[e.dst as usize] {
            wc[e.src as usize] += g[ei];
        }
    }
    mech.m = mm;
    mech.in_m = in_m;
    mech.gain = g;
    mech.pe = topk(&we, m.search.phi.purview);
    mech.pc = topk(&wc, m.search.phi.purview);
    phi::evaluate(&mut t, &m.params, v, w, &f, &mech, &m.search.phi).sharp_f
}

// ---------------------------------------------------------------- A

#[derive(Clone, Copy, Debug, Default)]
pub struct AScores {
    pub generativity: f32,
    pub disc: f32,
    pub proj: f32,
    pub has_proj: bool,
    pub a_known: f32,
    pub a_free: f32,
}

pub fn a_scores(
    cfg: &ViewCfg,
    bytes: &[u8],
    v: &EffView,
    w: &Wiring,
    prog: &Program,
    cause: Option<&Cause>,
    foreign: &[Vec<u8>],
    probes: &[Vec<u8>],
    phi: f32,
    stab: f32,
) -> AScores {
    let (reg, _) = substrate::execute(bytes, v, w, prog);
    let generativity = 1.0 - ptgrs::behavior_distance(bytes, &reg);
    let mut ds = Vec::new();
    for f in foreign.iter() {
        let fv = eff::build(f, cfg);
        let fw = substrate::wire(&fv);
        let (fr, _) = substrate::execute(f, &fv, &fw, prog);
        ds.push(ptgrs::behavior_distance(f, &fr));
    }
    let disc = mean(&ds);
    let mut proj = 0.0f32;
    let mut has = false;
    if let Some(c) = cause {
        if !probes.is_empty() {
            let mut ps = Vec::new();
            for x in probes.iter() {
                let xv = eff::build(x, cfg);
                let xw = substrate::wire(&xv);
                let (xr, _) = substrate::execute(x, &xv, &xw, prog);
                ps.push(1.0 - ptgrs::behavior_distance(&c.observe(x), &xr));
            }
            proj = mean(&ps);
            has = true;
        }
    }
    let a_known = if has {
        generativity.min(disc).min(proj)
    } else {
        generativity.min(disc)
    };
    let a_free = generativity.min(disc).min(phi).min(stab);
    AScores { generativity, disc, proj, has_proj: has, a_known, a_free }
}

// ---------------------------------------------------------------- baselines

pub struct BytePredictor {
    pub order: usize,
    cnt: Vec<BTreeMap<u64, BTreeMap<u8, u32>>>,
    tot: Vec<BTreeMap<u64, u32>>,
}

impl BytePredictor {
    /// Mêmes données PTGRS, même nombre d'épisodes, aucune cause cachée conservée.
    pub fn train(episodes: usize, seed: u64, order: usize) -> Self {
        let mut p = BytePredictor {
            order,
            cnt: (0..=order).map(|_| BTreeMap::new()).collect(),
            tot: (0..=order).map(|_| BTreeMap::new()).collect(),
        };
        let mut rng = Rng::new(seed);
        let spec = FamilySpec::train();
        for ep in 0..episodes {
            let mut er = rng.fork(tag("baseline").wrapping_add(ep as u64));
            let d = ptgrs::episode(&spec, &mut er);
            let e = d.e;
            drop(d.cause);
            for i in 0..e.len() {
                for k in 0..=order {
                    if i >= k {
                        let key = fnv1a(&e[i - k..i]);
                        *p.cnt[k].entry(key).or_default().entry(e[i]).or_insert(0) += 1;
                        *p.tot[k].entry(key).or_insert(0) += 1;
                    }
                }
            }
        }
        p
    }

    pub fn prob(&self, ctx: &[u8], next: u8) -> f32 {
        let alpha = 0.5f32;
        for k in (0..=self.order).rev() {
            if ctx.len() >= k {
                let key = fnv1a(&ctx[ctx.len() - k..]);
                if let Some(&t) = self.tot[k].get(&key) {
                    let c = self.cnt[k].get(&key).and_then(|m| m.get(&next)).copied().unwrap_or(0);
                    return (c as f32 + alpha) / (t as f32 + alpha * 256.0);
                }
            }
        }
        1.0 / 256.0
    }

    pub fn seq_surprisal(&self, ctx0: &[u8], seq: &[u8]) -> f32 {
        if seq.is_empty() {
            return 8.0;
        }
        let mut ctx: Vec<u8> = ctx0.to_vec();
        let mut s = 0.0f32;
        for &x in seq {
            s += -self.prob(&ctx, x).log2();
            ctx.push(x);
            if ctx.len() > self.order {
                ctx.remove(0);
            }
        }
        s / seq.len() as f32
    }

    pub fn atom_surprisal(&self, bytes: &[u8], s: usize, e: usize) -> f32 {
        let e = e.min(bytes.len());
        if e <= s {
            return 8.0;
        }
        let c0 = s.saturating_sub(self.order);
        self.seq_surprisal(&bytes[c0..s], &bytes[s..e])
    }
}

/// Baseline 1 : prédicteur d'octets. Même espace de candidats relationnels,
/// sélection par vraisemblance conditionnelle, aucun φ.
pub fn predictor_program(pred: &BytePredictor, bytes: &[u8], v: &EffView, w: &Wiring) -> Program {
    let keys = substrate::shape_keys(bytes, v);
    let mut by: BTreeMap<ShapeKey, PtrRule> = BTreeMap::new();
    for a in 0..w.na {
        let (s, e) = v.span(a);
        let e = e.min(bytes.len());
        if e <= s {
            continue;
        }
        let target = &bytes[s..e];
        let mut best = 8.0f32;
        let mut br: Option<PtrRule> = None;
        for (_eid, kind, rev, rank) in cand_list(v, w, a) {
            let src = v.edges[_eid].src as usize;
            let (ss, se) = v.span(src);
            let se = se.min(bytes.len());
            let ctx = if se > ss { &bytes[se.saturating_sub(pred.order)..se] } else { &[][..] };
            let sc = pred.seq_surprisal(ctx, target);
            if sc < best {
                best = sc;
                br = Some(PtrRule { tgt: keys[a], kind, rev, rank });
            }
        }
        if let Some(r) = br {
            by.entry(r.tgt).or_insert(r);
        }
    }
    Program { rules: by.values().copied().collect(), order: (0..w.na).collect() }
}

/// Baseline 2 : recherche guidée par la vraisemblance, sur le MÊME espace
/// d'hypothèses (programmes de pointeurs relationnels), même budget d'essais.
pub fn lik_search(
    pred: &BytePredictor,
    bytes: &[u8],
    v: &EffView,
    w: &Wiring,
    budget: usize,
) -> (Program, u64) {
    let keys = substrate::shape_keys(bytes, v);
    let mut order: Vec<usize> = (0..w.na).collect();
    let sur: Vec<f32> = (0..w.na).map(|a| { let (s, e) = v.span(a); pred.atom_surprisal(bytes, s, e) }).collect();
    order.sort_unstable_by(|&a, &b| sur[b].partial_cmp(&sur[a]).unwrap_or(Ordering::Equal).then(a.cmp(&b)));
    let mut rules: BTreeMap<ShapeKey, PtrRule> = BTreeMap::new();
    let mut best = f32::INFINITY;
    let mut evals = 0u64;
    let mk = |r: &BTreeMap<ShapeKey, PtrRule>| Program {
        rules: r.values().copied().collect(),
        order: (0..w.na).collect(),
    };
    for _pass in 0..2 {
        for &a in order.iter() {
            if evals as usize >= budget {
                break;
            }
            let cur = rules.get(&keys[a]).copied();
            let mut local = cur;
            let cands = cand_list(v, w, a);
            for (_eid, kind, rev, rank) in cands.into_iter().chain(std::iter::once((usize::MAX, 0, 0, 0))) {
                if evals as usize >= budget {
                    break;
                }
                if kind == 0 && rev == 0 && rank == 0 && _eid == usize::MAX {
                    rules.remove(&keys[a]);
                } else {
                    rules.insert(keys[a], PtrRule { tgt: keys[a], kind, rev, rank });
                }
                let (out, _) = substrate::execute(bytes, v, w, &mk(&rules));
                let sc = pred.seq_surprisal(&[], &out);
                evals += 1;
                if sc < best {
                    best = sc;
                    local = if _eid == usize::MAX { None } else { Some(PtrRule { tgt: keys[a], kind, rev, rank }) };
                }
            }
            match local {
                Some(r) => {
                    rules.insert(keys[a], r);
                }
                None => {
                    rules.remove(&keys[a]);
                }
            }
        }
    }
    (mk(&rules), evals * (w.rel_count.max(1) as u64))
}

/// Émission des baselines : atomes couverts, classés par surprisal, même
/// plancher relatif et même délimiteur que Phisharp. Armes égales.
pub fn baseline_emit(pred: &BytePredictor, bytes: &[u8], v: &EffView, w: &Wiring, prog: &Program, ec: &emit::EmitCfg) -> Vec<u8> {
    let keys = substrate::shape_keys(bytes, v);
    let by: BTreeMap<ShapeKey, PtrRule> = prog.rules.iter().map(|r| (r.tgt, *r)).collect();
    let mut sc = vec![0.0f32; w.na];
    for a in 0..w.na {
        if by.contains_key(&keys[a]) {
            let (s, e) = v.span(a);
            sc[a] = pred.atom_surprisal(bytes, s, e);
        }
    }
    let mx = sc.iter().cloned().fold(0.0f32, f32::max);
    let mut atoms: Vec<usize> = if mx > 0.0 {
        (0..w.na).filter(|&a| sc[a] >= ec.flow_rho * mx && sc[a] > 0.0).collect()
    } else {
        Vec::new()
    };
    atoms.sort_unstable_by(|&a, &b| sc[b].partial_cmp(&sc[a]).unwrap_or(Ordering::Equal).then(a.cmp(&b)));
    emit::render(bytes, v, &atoms, ec.delim)
}

// ---------------------------------------------------------------- B

fn chunks(out: &[u8], delim: u8) -> Vec<Vec<u8>> {
    out.split(|&x| x == delim).filter(|c| !c.is_empty()).map(|c| c.to_vec()).collect()
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > hay.len() {
        return false;
    }
    (0..=hay.len() - needle.len()).any(|i| &hay[i..i + needle.len()] == needle)
}

/// Rang fractionnaire de la bonne entrée. Continu, sans accès à A.
pub fn dict_rank(out: &[u8], bodies: &[(usize, usize)], target: usize, e: &[u8], delim: u8) -> (usize, f32) {
    let cs = chunks(out, delim);
    let k = bodies.len();
    let mut sc = vec![0.0f32; k];
    for (i, &(s, t)) in bodies.iter().enumerate() {
        let body = &e[s..t.min(e.len())];
        for c in cs.iter() {
            if contains(body, c) {
                sc[i] += 1.0;
            }
        }
    }
    let tv = sc[target];
    let greater = sc.iter().filter(|&&x| x > tv).count();
    let equal = sc.iter().enumerate().filter(|&(i, &x)| i != target && x == tv).count();
    let rank = 1.0 + greater as f32 + equal as f32 / 2.0;
    (greater + 1, 1.0 / rank)
}

fn wordset(b: &[u8]) -> BTreeSet<Vec<u8>> {
    let mut s = BTreeSet::new();
    let mut cur = Vec::new();
    for &x in b {
        if x.is_ascii_alphanumeric() {
            cur.push(x.to_ascii_lowercase());
        } else if !cur.is_empty() {
            s.insert(cur.clone());
            cur.clear();
        }
    }
    if !cur.is_empty() {
        s.insert(cur);
    }
    s
}

pub fn no_common_words(a: &[u8], b: &[u8]) -> bool {
    wordset(a).intersection(&wordset(b)).next().is_none()
}

/// Témoin : permutation des octets À L'INTÉRIEUR de chaque corps. Fréquences et
/// longueurs préservées, forme relationnelle détruite. Un système qui lit du
/// sens doit s'effondrer au hasard ; un système de surface non.
pub fn shuffle_bodies(md: &MiniDict, seed: u64) -> MiniDict {
    let mut rng = Rng::new(seed);
    let mut e = md.e.clone();
    for &(s, t) in md.entries.iter() {
        let t = t.min(e.len());
        if t > s {
            rng.shuffle(&mut e[s..t]);
        }
    }
    MiniDict { e, entries: md.entries.clone(), target: md.target, query: md.query, causes: md.causes.clone() }
}

/// Barreau 3 : mini-langue à grammaire connue. Rendu lettre-paire injectif du
/// barreau 2 ; la disjointure d'alphabet est préservée, donc zéro mot commun.
pub fn minilang(md: &MiniDict) -> (Vec<u8>, Vec<(usize, usize)>, (usize, usize)) {
    let mut e = Vec::new();
    let mut entries = Vec::new();
    for (i, &(s, t)) in md.entries.iter().enumerate() {
        if i > 0 {
            e.push(b'\n');
        }
        let st = e.len();
        for &x in &md.e[s..t.min(md.e.len())] {
            e.push(b'a' + (x >> 4));
            e.push(b'a' + (x & 15));
            e.push(b' ');
        }
        entries.push((st, e.len()));
    }
    e.push(b'\n');
    let qs = e.len();
    for &x in &md.e[md.query.0..md.query.1.min(md.e.len())] {
        e.push(b'a' + (x >> 4));
        e.push(b'a' + (x & 15));
        e.push(b' ');
    }
    let qe = e.len();
    (e, entries, (qs, qe))
}

pub struct DictFile {
    pub e: Vec<u8>,
    pub bodies: Vec<(usize, usize)>,
    pub heads: Vec<String>,
    pub target: usize,
    pub query: (usize, usize),
}

/// Format : `#TARGET<TAB>tête`, `#QUERY<TAB>phrase ambiguë`, puis `tête<TAB>définition`.
pub fn load_dict(path: &str) -> io::Result<DictFile> {
    let s = fs::read_to_string(path)?;
    let mut target_head = String::new();
    let mut query = String::new();
    let mut rows: Vec<(String, String)> = Vec::new();
    for line in s.lines() {
        if let Some(r) = line.strip_prefix("#TARGET\t") {
            target_head = r.trim().to_string();
        } else if let Some(r) = line.strip_prefix("#QUERY\t") {
            query = r.trim().to_string();
        } else if let Some((h, d)) = line.split_once('\t') {
            if !h.starts_with('#') && !d.trim().is_empty() {
                rows.push((h.trim().to_string(), d.trim().to_string()));
            }
        }
    }
    if rows.len() < 3 || query.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "dictionnaire: >=3 entrees et #QUERY requis"));
    }
    let mut e = Vec::new();
    let mut bodies = Vec::new();
    let mut heads = Vec::new();
    for (h, d) in rows.iter() {
        e.extend_from_slice(h.as_bytes());
        e.push(b':');
        e.push(b' ');
        let st = e.len();
        e.extend_from_slice(d.as_bytes());
        bodies.push((st, e.len()));
        heads.push(h.clone());
        e.push(b'\n');
    }
    e.push(b'\n');
    let qs = e.len();
    e.extend_from_slice(query.as_bytes());
    let q = (qs, e.len());
    let target = heads.iter().position(|h| *h == target_head).unwrap_or(0);
    Ok(DictFile { e, bodies, heads, target, query: q })
}

// ---------------------------------------------------------------- rapport

pub struct Report {
    pub lines: Vec<String>,
    pub a_reached: bool,
    pub b_reached: bool,
    pub calib_ok: bool,
}

impl Report {
    fn put(&mut self, k: &str, v: String) {
        self.lines.push(format!("{}\t{}", k, v));
    }
    pub fn verdict(&self) -> &'static str {
        match (self.a_reached, self.b_reached) {
            (true, true) => "these_soutenue",
            (true, false) => "moitie_abductive_soutenue;_explication_vient_du_sens_refutee",
            (false, true) => "pattern_matching_ou_heuristique",
            (false, false) => "sens_non_trouve;_seconde_moitie_non_testee",
        }
    }
    pub fn write(&self, path: &str) -> io::Result<()> {
        let mut s = String::new();
        for l in self.lines.iter() {
            s.push_str(l);
            s.push('\n');
        }
        s.push_str(&format!("verdict\t{}\n", self.verdict()));
        fs::write(path, s)
    }
}

pub fn run_all(
    model_path: &str,
    prereg_path: &str,
    work: &str,
    dict: Option<&str>,
    n_override: Option<usize>,
    force: bool,
) -> io::Result<Report> {
    ptgrs::enter_eval();
    let pre = Prereg::load(prereg_path)?;
    let m = Model::load(model_path)?;
    let work = PathBuf::from(work);
    let mut led = Ledger::load(&work.join("ledger.tsv").to_string_lossy());
    let sets = ["calib", "ood", "minidict", "minilang", "ablation", "invariance", "residue", "dict"];
    for s in sets.iter() {
        if led.seen(m.fingerprint, s) && !force {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("jeu {} deja vu par cette version figee ; --force enregistre une deviation", s),
            ));
        }
    }
    let nc = n_override.unwrap_or(pre.n_calib);
    let no = n_override.unwrap_or(pre.n_ood);
    let nd = n_override.unwrap_or(pre.n_dict);

    let mut rep = Report { lines: Vec::new(), a_reached: false, b_reached: false, calib_ok: false };
    rep.put("prereg_date", pre.date.clone());
    rep.put("prereg_fp", format!("{:016x}", pre.fingerprint()));
    rep.put("model_fp", format!("{:016x}", m.fingerprint));
    rep.put("n_override", format!("{}", n_override.is_some() as u8));
    rep.put("forced", format!("{}", force as u8));

    let pred = BytePredictor::train(m.episodes as usize, m.seed, 4);
    let mut rng = Rng::new(m.seed ^ tag("eval"));
    let mut foreign: Vec<Vec<u8>> = Vec::new();
    for i in 0..4 {
        foreign.push(ptgrs::episode(&FamilySpec::ood(i), &mut rng).e);
    }

    // ---- étalonnage ----
    let mut q: Vec<f32> = Vec::new();
    let mut ph: Vec<f32> = Vec::new();
    let mut lk: Vec<f32> = Vec::new();
    let mut md: Vec<f32> = Vec::new();
    let mut copy_wins = 0usize;
    let mut phi_real: Vec<f32> = Vec::new();
    let mut phi_rand: Vec<f32> = Vec::new();
    let mut sweep_flip = 0usize;
    for i in 0..nc {
        let ep = ptgrs::episode(&FamilySpec::train(), &mut rng);
        let v = eff::build(&ep.e, &m.view);
        let w = substrate::wire(&v);
        let r = ask(model_path, &ep.e, "", &work.join(format!("c{}", i)))?;
        let prog = prog_of(&r.diag, w.na);
        let ps = ptgrs::probes(&ep.cause, &mut rng, pre.probes);
        let qual = 1.0
            - ptgrs::obs_distance(
                &ep.cause,
                |x| {
                    let xv = eff::build(x, &m.view);
                    let xw = substrate::wire(&xv);
                    substrate::execute(x, &xv, &xw, &prog).0
                },
                &ps,
            );
        let id = substrate::identity_program(&ep.e, &v);
        let pid = phi_of_program(&m, &ep.e, &v, &w, &id);
        let prec = phi_of_program(&m, &ep.e, &v, &w, &prog);
        if pid >= prec {
            copy_wins += 1;
        }
        let (reg, _) = substrate::execute(&ep.e, &v, &w, &prog);
        q.push(qual);
        ph.push(prec);
        lk.push(-pred.seq_surprisal(&[], &reg));
        md.push(-ptgrs::mdl_bits(&reg));
        phi_real.push(r.diag.phi);
        let rb: Vec<u8> = (0..ep.e.len()).map(|_| rng.below(256) as u8).collect();
        let rr = ask(model_path, &rb, "", &work.join(format!("cr{}", i)))?;
        phi_rand.push(rr.diag.phi);
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for k in pre.mip_sweep.iter() {
            let tagv = format!("mip{}", k);
            let rk = ask(model_path, &ep.e, &tagv, &work.join(format!("cm{}{}", k, i)))?;
            lo = lo.min(rk.diag.phi);
            hi = hi.max(rk.diag.phi);
        }
        if hi - lo > 0.25 {
            sweep_flip += 1;
        }
    }
    let sp_phi = spearman(&q, &ph);
    let sp_lik = spearman(&q, &lk);
    let sp_mdl = spearman(&q, &md);
    let cwr = copy_wins as f32 / nc.max(1) as f32;
    let rratio = mean(&phi_rand) / mean(&phi_real).max(1e-6);
    rep.put("calib_spearman_phi", format!("{:.4}", sp_phi));
    rep.put("calib_spearman_likelihood", format!("{:.4}", sp_lik));
    rep.put("calib_spearman_mdl", format!("{:.4}", sp_mdl));
    rep.put("calib_copy_win_rate", format!("{:.4}", cwr));
    rep.put("calib_random_phi_ratio", format!("{:.4}", rratio));
    rep.put("calib_mip_unstable", format!("{}", sweep_flip));
    rep.calib_ok = cwr <= pre.copy_win_max
        && sp_phi >= pre.spearman_min
        && sp_phi > sp_lik
        && sp_phi > sp_mdl
        && rratio <= pre.random_phi_ratio_max;
    rep.put("calib_ok", format!("{}", rep.calib_ok as u8));
    led.record(m.fingerprint, "calib")?;
    if !rep.calib_ok {
        rep.put("stop", "etalonnage_echoue: A n'est pas lisible".into());
        rep.write(&work.join("report.tsv").to_string_lossy())?;
        return Ok(rep);
    }

    // ---- barreau 1 : PTGRS hors-distribution ----
    let mut a1: Vec<f32> = Vec::new();
    let mut ak: Vec<f32> = Vec::new();
    let mut b1_pred: Vec<f32> = Vec::new();
    let mut b1_lik: Vec<f32> = Vec::new();
    for i in 0..no {
        let spec = FamilySpec::ood(i % 5);
        let ep = ptgrs::episode(&spec, &mut rng);
        let v = eff::build(&ep.e, &m.view);
        let w = substrate::wire(&v);
        let r = ask(model_path, &ep.e, "", &work.join(format!("o{}", i)))?;
        let prog = prog_of(&r.diag, w.na);
        let ps = ptgrs::probes(&ep.cause, &mut rng, pre.probes);
        let sc = a_scores(&m.view, &ep.e, &v, &w, &prog, Some(&ep.cause), &foreign, &ps, r.diag.phi, r.diag.stab);
        a1.push(sc.a_free);
        ak.push(sc.a_known);
        let pp = predictor_program(&pred, &ep.e, &v, &w);
        let (lp, _) = lik_search(&pred, &ep.e, &v, &w, m.search.max_ops * 24);
        b1_pred.push(a_scores(&m.view, &ep.e, &v, &w, &pp, Some(&ep.cause), &foreign, &ps, 0.0, 0.0).a_known);
        b1_lik.push(a_scores(&m.view, &ep.e, &v, &w, &lp, Some(&ep.cause), &foreign, &ps, 0.0, 0.0).a_known);
    }
    rep.put("ood_a_free", format!("{:.4}", mean(&a1)));
    rep.put("ood_a_known", format!("{:.4}", mean(&ak)));
    rep.put("ood_baseline_predictor", format!("{:.4}", mean(&b1_pred)));
    rep.put("ood_baseline_likelihood", format!("{:.4}", mean(&b1_lik)));
    led.record(m.fingerprint, "ood")?;

    // ---- barreaux 2 et 3 ----
    let mut mrr2: Vec<f32> = Vec::new();
    let mut mrr3: Vec<f32> = Vec::new();
    let mut mrr_sh: Vec<f32> = Vec::new();
    let mut mrr_bp: Vec<f32> = Vec::new();
    let mut mrr_bl: Vec<f32> = Vec::new();
    let mut disjoint_ok = 0usize;
    for i in 0..nd {
        let mdic = ptgrs::minidict(&FamilySpec::ood(2), &mut rng, 5);
        let v = eff::build(&mdic.e, &m.view);
        let w = substrate::wire(&v);
        let r = ask(model_path, &mdic.e, "", &work.join(format!("d{}", i)))?;
        let (_, rr) = dict_rank(&r.out, &mdic.entries, mdic.target, &mdic.e, m.emit.delim);
        mrr2.push(rr);
        let q = &mdic.e[mdic.query.0..mdic.query.1];
        let mut db: Vec<u8> = Vec::new();
        for &(s, t) in mdic.entries.iter() {
            db.extend_from_slice(&mdic.e[s..t]);
        }
        if q.iter().all(|x| !db.contains(x)) {
            disjoint_ok += 1;
        }
        let sh = shuffle_bodies(&mdic, m.seed ^ (i as u64));
        let rs = ask(model_path, &sh.e, "", &work.join(format!("ds{}", i)))?;
        mrr_sh.push(dict_rank(&rs.out, &sh.entries, sh.target, &sh.e, m.emit.delim).1);
        let (le, lb, lq) = minilang(&mdic);
        let rl = ask(model_path, &le, "", &work.join(format!("l{}", i)))?;
        mrr3.push(dict_rank(&rl.out, &lb, mdic.target, &le, m.emit.delim).1);
        let _ = lq;
        let pp = predictor_program(&pred, &mdic.e, &v, &w);
        let (lp, _) = lik_search(&pred, &mdic.e, &v, &w, m.search.max_ops * 24);
        mrr_bp.push(dict_rank(&baseline_emit(&pred, &mdic.e, &v, &w, &pp, &m.emit), &mdic.entries, mdic.target, &mdic.e, m.emit.delim).1);
        mrr_bl.push(dict_rank(&baseline_emit(&pred, &mdic.e, &v, &w, &lp, &m.emit), &mdic.entries, mdic.target, &mdic.e, m.emit.delim).1);
    }
    rep.put("minidict_mrr", format!("{:.4}", mean(&mrr2)));
    rep.put("minidict_shuffled_mrr", format!("{:.4}", mean(&mrr_sh)));
    rep.put("minidict_disjoint", format!("{}/{}", disjoint_ok, nd));
    rep.put("minilang_mrr", format!("{:.4}", mean(&mrr3)));
    rep.put("minidict_baseline_predictor", format!("{:.4}", mean(&mrr_bp)));
    rep.put("minidict_baseline_likelihood", format!("{:.4}", mean(&mrr_bl)));
    led.record(m.fingerprint, "minidict")?;
    led.record(m.fingerprint, "minilang")?;

    // ---- barreau 4 : vrai dictionnaire ----
    let mut mrr4 = 0.0f32;
    let mut ncw = false;
    if let Some(dp) = dict {
        let df = load_dict(dp)?;
        let r = ask(model_path, &df.e, "", &work.join("dict"))?;
        mrr4 = dict_rank(&r.out, &df.bodies, df.target, &df.e, m.emit.delim).1;
        let qb = &df.e[df.query.0..df.query.1];
        let tb = &df.e[df.bodies[df.target].0..df.bodies[df.target].1];
        ncw = no_common_words(qb, tb);
        rep.put("dict_mrr", format!("{:.4}", mrr4));
        rep.put("dict_no_common_words", format!("{}", ncw as u8));
        rep.put("dict_target", df.heads[df.target].clone());
        led.record(m.fingerprint, "dict")?;
    }

    // ---- ablations, invariance, résidu ----
    let na = no.min(4);
    let mut drop_b0 = 0.0f32;
    let mut drop_nr = 0.0f32;
    let mut inv_ok = 0usize;
    let mut res_ok = 0usize;
    let mut a0: Vec<f32> = Vec::new();
    for i in 0..na {
        let ep = ptgrs::episode(&FamilySpec::ood(1), &mut rng);
        let v = eff::build(&ep.e, &m.view);
        let w = substrate::wire(&v);
        let base = ask(model_path, &ep.e, "", &work.join(format!("a{}", i)))?;
        let pb = prog_of(&base.diag, w.na);
        let sb = a_scores(&m.view, &ep.e, &v, &w, &pb, Some(&ep.cause), &foreign, &[], base.diag.phi, base.diag.stab);
        let b0 = ask(model_path, &ep.e, "budget0", &work.join(format!("a0{}", i)))?;
        let p0 = prog_of(&b0.diag, w.na);
        let s0 = a_scores(&m.view, &ep.e, &v, &w, &p0, Some(&ep.cause), &foreign, &[], b0.diag.phi, b0.diag.stab);
        a0.push(s0.a_free);
        let nr = ask(model_path, &ep.e, "norefute", &work.join(format!("an{}", i)))?;
        let pn = prog_of(&nr.diag, w.na);
        let sn = a_scores(&m.view, &ep.e, &v, &w, &pn, Some(&ep.cause), &foreign, &[], nr.diag.phi, nr.diag.stab);
        drop_b0 += sb.a_free - s0.a_free;
        drop_nr += sb.a_free - sn.a_free;

        let perm = Perm::random(&mut rng);
        let pe = perm.apply(&ep.e);
        let pr = ask(model_path, &pe, "", &work.join(format!("iv{}", i)))?;
        if perm.unapply(&pr.out) == base.out && (pr.diag.phi - base.diag.phi).abs() < 1e-6 {
            inv_ok += 1;
        }

        if let Some(&(ra, _)) = base.diag.residue.first() {
            let s = ra * v.atom_len;
            let ctx_s = s.saturating_sub(2 * v.atom_len);
            let ctx_e = (s + 3 * v.atom_len).min(ep.e.len());
            let mut e2 = ep.e.clone();
            e2.extend_from_slice(&ep.e[ctx_s..ctx_e]);
            let r2 = ask(model_path, &e2, "", &work.join(format!("rs{}", i)))?;
            let mut e3 = ep.e.clone();
            let rs = rng.below(ep.e.len().saturating_sub(ctx_e - ctx_s).max(1));
            e3.extend_from_slice(&ep.e[rs..(rs + (ctx_e - ctx_s)).min(ep.e.len())]);
            let r3 = ask(model_path, &e3, "", &work.join(format!("rc{}", i)))?;
            if r2.diag.phi - base.diag.phi > pre.residue_gain_min
                && r2.diag.phi > r3.diag.phi
            {
                res_ok += 1;
            }
        }
    }
    let nn = na.max(1) as f32;
    rep.put("ablation_budget0_a", format!("{:.4}", mean(&a0)));
    rep.put("ablation_drop_budget", format!("{:.4}", drop_b0 / nn));
    rep.put("ablation_drop_refutation", format!("{:.4}", drop_nr / nn));
    rep.put("invariance_exact", format!("{}/{}", inv_ok, na));
    rep.put("residue_gain_ok", format!("{}/{}", res_ok, na));
    led.record(m.fingerprint, "ablation")?;
    led.record(m.fingerprint, "invariance")?;
    led.record(m.fingerprint, "residue")?;

    // ---- lecture croisée ----
    let collapse = mean(&mrr2) - mean(&mrr_sh);
    let margin = mean(&mrr2) - mean(&mrr_bp).max(mean(&mrr_bl));
    rep.put("b_collapse", format!("{:.4}", collapse));
    rep.put("b_margin", format!("{:.4}", margin));
    let b_best = mean(&mrr2).max(mrr4);
    rep.a_reached = mean(&a1) >= pre.a_min
        && mean(&a0) <= pre.budget0_a_max
        && drop_b0 / nn >= pre.ablation_drop_min
        && drop_nr / nn >= pre.ablation_drop_min
        && inv_ok == na
        && res_ok * 2 >= na;
    rep.b_reached = b_best >= pre.b_mrr_min
        && margin >= pre.b_margin_min
        && collapse >= pre.shuffle_collapse_min;
    rep.write(&work.join("report.tsv").to_string_lossy())?;
    Ok(rep)
}

/// Évaluateur aveugle : sorties anonymisées, grille fixée d'avance. La clé est
/// scellée dans un fichier séparé que l'évaluateur ne voit pas.
pub fn blind(model_path: &str, dict_path: &str, work: &str) -> io::Result<()> {
    ptgrs::enter_eval();
    let m = Model::load(model_path)?;
    let df = load_dict(dict_path)?;
    let work = PathBuf::from(work);
    let v = eff::build(&df.e, &m.view);
    let w = substrate::wire(&v);
    let pred = BytePredictor::train(m.episodes as usize, m.seed, 4);
    let r = ask(model_path, &df.e, "", &work.join("blind"))?;
    let pp = predictor_program(&pred, &df.e, &v, &w);
    let (lp, _) = lik_search(&pred, &df.e, &v, &w, m.search.max_ops * 24);
    let outs = [
        r.out.clone(),
        baseline_emit(&pred, &df.e, &v, &w, &pp, &m.emit),
        baseline_emit(&pred, &df.e, &v, &w, &lp, &m.emit),
    ];
    let mut rng = Rng::new(m.seed ^ tag("blind"));
    let mut idx = [0usize, 1, 2];
    rng.shuffle(&mut idx);
    let mut items = String::from("item\tsystem\tjustesse(0-3)\tclarte(0-3)\toutput\n");
    let mut key = String::from("system\tcode\n");
    let names = ["phisharp", "byte_predictor", "likelihood_search"];
    for (j, &i) in idx.iter().enumerate() {
        let code = format!("S{}", j + 1);
        let txt: String = outs[i].iter().map(|&b| if b >= 32 && b < 127 { b as char } else { ' ' }).collect();
        items.push_str(&format!("1\t{}\t\t\t{}\n", code, txt));
        key.push_str(&format!("{}\t{}\n", names[i], code));
    }
    fs::create_dir_all(&work)?;
    let mut f = fs::File::create(work.join("blind.tsv"))?;
    f.write_all(items.as_bytes())?;
    fs::write(work.join("blind.key.sealed"), key)?;
    Ok(())
}