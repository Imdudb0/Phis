//! Harnais de preuve. Aucune assertion n'est déclarée vraie sans être calculée.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use phisharp::algebra::{sever, signature, unify, Cause, Node, values, IDENT};
use phisharp::core::{abduce, residual, Cfg, Perception, Router};
use phisharp::ptgrs::episode;
use phisharp::phi::{phi, phi_pi};
use phisharp::rng::PyRandom;
use phisharp::model::{load, save, train_manifest, train_manifest_stats, understand};

#[path = "common/fixture.rs"] mod fix;
use fix::{fixture, WORDS, NONCE};

fn check(results: &mut Vec<(String, bool, String)>, name: &str, cond: bool, detail: &str) {
    results.push((name.to_string(), cond, detail.to_string()));
    println!(
        "[{}] {}  {}",
        if cond { "PASS" } else { "FAIL" },
        name,
        detail
    );
}

/// Intervalle de confiance de Wilson (score) à 95 %.
/// n = 0 => (0,1) : indéterminé, jamais NaN (A20 doit alors échouer, pas paniquer).
fn wilson(k: usize, n: usize) -> (f64, f64) {
    const Z: f64 = 1.959_963_984_540_054;
    if n == 0 {
        return (0.0, 1.0);
    }
    let nf = n as f64;
    let p = k.min(n) as f64 / nf;
    let z2 = Z * Z;
    let den = 1.0 + z2 / nf;
    let center = (p + z2 / (2.0 * nf)) / den;
    let half = Z * (p * (1.0 - p) / nf + z2 / (4.0 * nf * nf)).sqrt() / den;
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn motif(kind: usize, u: &[u8], v1: &[u8], v2: &[u8], w: &[u8], sep: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    match kind {
        0 => {
            out.extend_from_slice(u);
            out.extend_from_slice(v1);
            out.extend_from_slice(sep);
            out.extend_from_slice(u);
            out.extend_from_slice(v2);
        }
        1 => {
            out.extend_from_slice(u);
            out.extend_from_slice(v1);
            out.extend_from_slice(sep);
            out.extend_from_slice(u);
            out.extend_from_slice(v1);
        }
        _ => {
            out.extend_from_slice(u);
            out.extend_from_slice(v1);
            out.extend_from_slice(sep);
            out.extend_from_slice(w);
            out.extend_from_slice(v1);
        }
    }
    out
}

fn pools(hi: bool) -> Vec<Vec<u8>> {
    if hi {
        (0..4)
            .map(|t| {
                let lo = (0x80 + t * 0x10) as u8;
                (lo..lo + 12).collect()
            })
            .collect()
    } else {
        let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; // 36 = 4 x 9
        (0..4).map(|t| chars[t * 9..(t + 1) * 9].to_vec()).collect()
    }
}

fn tok(rng: &mut PyRandom, pool: &[u8]) -> Vec<u8> {
    vec![*rng.choice(pool), *rng.choice(pool)]
}

fn atom_pattern(atoms: &[&[u8]]) -> Vec<usize> {
    let mut seen: HashMap<&[u8], usize> = HashMap::new();
    let mut out = Vec::with_capacity(atoms.len());
    for &a in atoms {
        let len = seen.len();
        let id = *seen.entry(a).or_insert(len);
        out.push(id);
    }
    out
}

fn skeleton(x: &[u8], cfg: &Cfg) -> Vec<usize> {
    let p = Perception::new(x.to_vec(), cfg);
    let slices: Vec<&[u8]> = p.atoms.iter().map(|&(a, b)| &x[a..b]).collect();
    atom_pattern(&slices)
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

fn main() -> ExitCode {
    let mut results: Vec<(String, bool, String)> = Vec::new();
    let cfg = Cfg::default();
    fs::create_dir_all("out").expect("failed to create out/");

    // A0 : audit du test lui-même (invariant réellement instantié)
    let mut a0_ok = true;
    let mut a0_detail = Vec::new();
    for hi in [true, false] {
        let (_, bb) = fixture(WORDS, hi, 7, false, false);
        let pat: Vec<Vec<usize>> = bb.iter().map(|x| skeleton(x, &cfg)).collect();
        let ok_hi = pat[0] == pat[3] && pat[1] != pat[0] && pat[2] != pat[0];
        a0_ok &= ok_hi;
        a0_detail.push(format!("hi={hi} {pat:?}"));
    }
    check(
        &mut results,
        "A0 audit-fixture (invariant réellement instantié)",
        a0_ok,
        &a0_detail.join(" | "),
    );

    // A12 : audit de source — aucun mot du domaine dans le paquet
    let mut src = Vec::new();
    if let Ok(entries) = fs::read_dir(Path::new("src")) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("rs") {
                if let Ok(bytes) = fs::read(&p) {
                    src.extend_from_slice(&bytes);
                }
            }
        }
    }
    let mut leak = Vec::new();
    for &w in WORDS.iter().chain(NONCE.iter()).chain(["the cause of"].iter()) {
        if contains_subslice(&src, w.as_bytes()) {
            leak.push(w);
        }
    }
    check(
        &mut results,
        "A12 audit-source (aucun contenu de domaine dans le modèle)",
        !src.is_empty() && leak.is_empty(),
        &format!("leaks={leak:?}"),
    );

    eprintln!("entraînement : une seule séance, une passe…");
    let (router, manifest, tstats) = train_manifest_stats(&cfg, 200, 1, None);
    save(&router, &cfg, "out/sharp.phis").expect("save failed");
    let (router, cfg) = load("out/sharp.phis").expect("load failed");

    for (tag, names) in [("WORDS", WORDS), ("NONCE", NONCE)] {
        let (e, _) = fixture(names, false, 7, false, false);
        let mut p = Perception::new(e.clone(), &cfg);
        p.ensure_frames();
        let (s, _) = router.score(&p);
        for &(a, b, c, r) in &p.slots {
            if c != 0 { continue; }
            let ab = abduce(&p, a, b, &s, &cfg, &mut PyRandom::new(0), None);
            let ops: Vec<String> = ab.cause.as_ref()
                .map(|cc| cc.nodes.iter().map(|n| format!("{:?}", n.op)).collect())
                .unwrap_or_default();
            let arite = ab.cause.as_ref().map(|cc| cc.nodes[cc.root].kids.len()).unwrap_or(0);
            eprintln!("{tag} rec{r} ({a},{b}) cover={:?} arite={arite} ops={ops:?} phi={:.3}", ab.cover.iter().map(|&i| (p.items[i].i, p.items[i].j)).collect::<Vec<_>>(), ab.phi);
        }
    }

    // A9 : non-réciprocité (Jacobien M du routage graphe + F != B^T)
    let per0 = Perception::new(fixture(WORDS, false, 7, false, false).0, &cfg);
    let m = router.influence(&per0);
    let mut norm_m2 = 0.0f64;
    let mut norm_asym2 = 0.0f64;
    for i in 0..m.len() {
        for j in 0..m.len() {
            norm_m2 += m[i][j] * m[i][j];
            let d = m[i][j] - m[j][i];
            norm_asym2 += d * d;
        }
    }
    let asym_ratio = if norm_m2 > 0.0 {
        norm_asym2.sqrt() / norm_m2.sqrt()
    } else {
        0.0
    };
    let nonrecip_fb = per0
        .f
        .iter()
        .take(50)
        .any(|(&(a, b), &fv)| (fv - per0.b.get(&(b, a)).copied().unwrap_or(0.0)).abs() > 1e-9);
    check(
        &mut results,
        "A9 non-réciprocité (asymétrie Jacobien M et F != B^T)",
        router.nonreciprocal() && asym_ratio > 0.1 && nonrecip_fb,
        &format!("asym={asym_ratio:.4}"),
    );

    // A10 : une seule passe (assert dans train) + A1 zéro donnée cible
    let mut train_set: HashSet<Vec<u8>> = HashSet::new();
    let mut rng = PyRandom::new(1);
    for _ in 0..200 {
        let (e, _) = episode(&mut rng, 60);
        train_set.insert(e);
    }
    check(&mut results, "A10 une-seule-passe (unicité construite, pas subie)",
      manifest.len() == 200, &format!("uniques={}", manifest.len()));

    // ---- Preuve principale : réponse NL, zéro-shot ---------------------
    for (tag, names, hi) in [
        ("A3a octets disjoints", WORDS, true),
        ("A3b ASCII, tokens disjoints", WORDS, false),
        ("A11 termes NONCE (anti-mémorisation)", NONCE, false),
    ] {
        let (e, bodies) = fixture(names, hi, 7, false, false);
        check(&mut results, "A1 fixture absente de l'entraînement", !manifest.contains(&e), "");
        let (ans, rep) = understand(&e, &router, &cfg, 0);
        let mut expect = Vec::new();
        expect.extend_from_slice(b"the cause of ");
        expect.extend_from_slice(&bodies[3]);
        expect.extend_from_slice(b" is ");
        expect.extend_from_slice(names[0].as_bytes());
        expect.push(b'.');

        let rung_str = rep.rung.as_deref().unwrap_or("None");
        check(
            &mut results,
            &format!("{tag} : réponse exacte"),
            ans == expect,
            &format!("rung={rung_str} got={:?}", String::from_utf8_lossy(&ans)),
        );
        check(
            &mut results,
            &format!("{tag} : réponse en NL (trame close)"),
            matches!(rung_str, "unify" | "signature"),
            "",
        );

        let set_ans: HashSet<u8> = ans.iter().copied().collect();
        let set_e: HashSet<u8> = e.iter().copied().collect();
        check(
            &mut results,
            "A2 clôture d'alphabet réponse ⊆ A(E)",
            set_ans.is_subset(&set_e),
            "",
        );

        let set_b3: HashSet<u8> = bodies[3].iter().copied().collect();
        let set_b0: HashSet<u8> = bodies[0].iter().copied().collect();
        let mut overlap: Vec<u8> = set_b3.intersection(&set_b0).copied().collect();
        overlap.sort();
        check(
            &mut results,
            "A11bis zéro octet commun requête/définition",
            overlap.is_empty(),
            &format!("overlap={overlap:?}"),
        );
    }

    // ---- A4 : Φ copie = 0 < Φ abduite ----------------------------------
    let (e, _bodies) = fixture(WORDS, false, 7, false, false);
    let mut per = Perception::new(e.clone(), &cfg);
    per.ensure_frames();
    let (scores, _) = router.score(&per);
    let mut rng2 = PyRandom::new(0);
    let (a, b) = (per.slots[0].0, per.slots[0].1);
    let ab = abduce(&per, a, b, &scores, &cfg, &mut rng2, None);
    let copy = Cause {
        nodes: vec![Node::lit(e[a..b].to_vec())],
        root: 0,
    };
    let (ph_copy, _) = phi(&copy, &e[a..b], &mut PyRandom::new(0), 48, &[], 3);
    check(
        &mut results,
        "A4 Φ(copie)=0 < Φ(abduction)",
        ph_copy == 0.0 && ab.phi > 0.0,
        &format!("copy={ph_copy:.3} abduced={:.3}", ab.phi),
    );

    // ---- A5 : certificat de partition ----------------------------------
    let srcs: Vec<usize> = ab
        .info
        .as_ref()
        .map(|inf| {
            inf.sources
                .iter()
                .filter(|d| d.iota > 0.0 && d.eps > 0.0)
                .map(|d| d.node)
                .collect()
        })
        .unwrap_or_default();
    let mut drops = Vec::new();
    if let Some(ref cause) = ab.cause {
        for u in srcs {
            let cut = sever(cause, u);
            let (pcut, _) = phi(&cut, &e[a..b], &mut PyRandom::new(0), 48, &[], 3);
            drops.push(pcut);
        }
    }
    let max_drop = drops.iter().copied().reduce(f64::max).unwrap_or(f64::INFINITY);
    check(
        &mut results,
        "A5 partition : couper le partage effondre Φ",
        !drops.is_empty() && max_drop < ab.phi,
        &format!("whole={:.3} cuts={drops:?}", ab.phi),
    );

    // ---- A6 : résidu causal sur le MÊME slot requête intact vs mutilé --
    let (e2, _) = fixture(WORDS, false, 7, true, false);
    let mut per2 = Perception::new(e2, &cfg);
    per2.ensure_frames();
    let (s2, _) = router.score(&per2);
    let q_intact = per.slots.iter().find(|s| s.3 == 3 && s.2 == 0).copied();
    let q_mutil = per2.slots.iter().find(|s| s.3 == 3 && s.2 == 0).copied();
    if let (Some((ix, iy, _, _)), Some((mx, my, _, _))) = (q_intact, q_mutil) {
        let ab_q = abduce(&per, ix, iy, &scores, &cfg, &mut PyRandom::new(0), None);
        let ab2 = abduce(&per2, mx, my, &s2, &cfg, &mut PyRandom::new(0), None);
        let r2 = residual(&per2, mx, my, &ab2);
        let missing = !r2.gap.is_empty() || !r2.inert.is_empty();
        check(&mut results,
            "A6 résidu : même slot requête mutilé => Φ chute et ρ non vide",
            ab2.phi < ab_q.phi && missing,
            &format!("intact={:.3} mutilated={:.3} gap={:?} inert={:?}", ab_q.phi, ab2.phi, r2.gap, r2.inert));
        eprintln!("A6 diag slot=({mx},{my}) bytes={:?} atoms_chevauchants={:?} complete={} cause={} nsrc={:?} gap={:?} inert={:?}",
        &per2.e[mx..my],
        per2.atoms.iter().filter(|&&(x,y)| x < my && y > mx).collect::<Vec<_>>(),
        ab2.complete, ab2.cause.is_some(),
        ab2.info.as_ref().map(|i| i.sources.len()), r2.gap, r2.inert);
    } else {
        check(&mut results, "A6 résidu", false, "slot requête introuvable");
    }

    // ---- A7 : complétude (contrôleur aléatoire, gros budget) ------------
    let (er, _) = episode(&mut PyRandom::new(999), 60);
    let perr = Perception::new(er.clone(), &cfg);
    let (sr, _) = router.score(&perr);
    let ab_t = abduce(&perr, 0, er.len(), &sr, &cfg, &mut PyRandom::new(3), None);
    let rand_router = Router::new(cfg.d, cfg.h, 42);
    let (srnd, _) = rand_router.score(&perr);
    let ab_r = abduce(
        &perr,
        0,
        er.len(),
        &srnd,
        &cfg,
        &mut PyRandom::new(3),
        Some(cfg.pops * 20),
    );
    check(
        &mut results,
        "A7 complétude : contrôleur aléatoire + budget x20 atteint l'optimum",
        ab_r.phi >= ab_t.phi - 1e-9,
        &format!("trained={:.3} random={:.3}", ab_t.phi, ab_r.phi),
    );

    // bruit intrinsèque : même routeur, deux graines Monte-Carlo
    let mut floor = 0.0f64;
    for s in 0..32u64 {
        let (ee, _) = episode(&mut PyRandom::new(1000 + s), 60);
        let pe = Perception::new(ee.clone(), &cfg);
        let (st, _) = router.score(&pe);
        let a1 = abduce(&pe, 0, ee.len(), &st, &cfg, &mut PyRandom::new(5), None);
        let a2 = abduce(&pe, 0, ee.len(), &st, &cfg, &mut PyRandom::new(6), None);
        floor = floor.max((a1.phi - a2.phi).abs());
    }
    let mut gains = Vec::new();
    for s in 0..12u64 {
        let (ee, _) = episode(&mut PyRandom::new(1000 + s), 60);
        let pe = Perception::new(ee.clone(), &cfg);
        let (st, _) = router.score(&pe);
        let (su, _) = rand_router.score(&pe);
        let at = abduce(&pe, 0, ee.len(), &st, &cfg, &mut PyRandom::new(5), None);
        let au = abduce(&pe, 0, ee.len(), &su, &cfg, &mut PyRandom::new(5), None);
        gains.push((at.pops, au.pops, at.phi, au.phi));
    }
    let mt = gains.iter().map(|g| g.0 as f64).sum::<f64>() / gains.len() as f64;
    let mu = gains.iter().map(|g| g.1 as f64).sum::<f64>() / gains.len() as f64;
    let pt = gains.iter().map(|g| g.2).sum::<f64>() / gains.len() as f64;
    let pu = gains.iter().map(|g| g.3).sum::<f64>() / gains.len() as f64;
    check(&mut results, "A8a efficacité : pops réduits", mt <= mu, &format!("pops {mt:.1} vs {mu:.1}"));
    check(&mut results, "A8b non-infériorité Φ (tolérance = bruit mesuré)", pt >= pu - floor, &format!("phi {pt:.3} vs {pu:.3} | plancher_bruit={floor:.3}"));

    check(&mut results, "A1bis sondes A7/A8 hors entraînement", !manifest.contains(&er), "");

    let cfg_ref = Cfg { completes: cfg.completes * 4, ..cfg.clone() };  
    let (mut hit_t, mut hit_u) = (0usize, 0usize);
    for s in 0..16u64 {
        let (ee, _) = episode(&mut PyRandom::new(1000 + s), 60);
        let pe = Perception::new(ee.clone(), &cfg);
        let (st, _) = router.score(&pe);
        let (su, _) = rand_router.score(&pe);
        let at = abduce(&pe, 0, ee.len(), &st, &cfg,     &mut PyRandom::new(5), None);
        let au = abduce(&pe, 0, ee.len(), &su, &cfg,     &mut PyRandom::new(5), None);
        let ar = abduce(&pe, 0, ee.len(), &st, &cfg_ref, &mut PyRandom::new(5), Some(cfg.pops * 4));
        hit_t += (at.phi >= ar.phi - 1e-9) as usize;
        hit_u += (au.phi >= ar.phi - 1e-9) as usize;
    }
    check(&mut results, "A8c taux d'atteinte de l'optimum top-32", hit_t >= hit_u, &format!("{hit_t}/16 vs {hit_u}/16"));


// D1 : la sonde doit être non dégénérée, sinon A8b ne veut rien dire
check(&mut results, "A8b-0 sonde de bruit non dégénérée", floor > 0.0, &format!("plancher={floor:.6}"));

// P19 : le motif, pas le contexte — égalité AU BIT
fn dump(cfg: &Cfg, router: &Router, names: [&str; 3])
    -> Vec<(usize, Vec<u8>, Vec<String>, usize, u64)> {
    let (e, _) = fixture(names, false, 7, false, false);
    let mut p = Perception::new(e, cfg); p.ensure_frames();
    let (s, _) = router.score(&p);
    let mut rows = Vec::new();
    for &(a, b, c, r) in &p.slots {
        if c != 0 { continue; }
        let ab = abduce(&p, a, b, &s, cfg, &mut PyRandom::new(0), None);
        let ops: Vec<String> = ab.cause.as_ref()
            .map(|cc| cc.nodes.iter().map(|n| format!("{:?}", n.op)).collect()).unwrap_or_default();
        let ar = ab.cause.as_ref().map(|cc| cc.nodes[cc.root].kids.len()).unwrap_or(0);
        rows.push((r, p.e[a..b].to_vec(), ops, ar, ab.phi.to_bits()));
    }
    rows
}
let dw = dump(&cfg, &router, WORDS);
let dn = dump(&cfg, &router, NONCE);
check(&mut results, "A17 P19 WORDS ≡ NONCE (corps, ops, arité, Φ au bit)", dw == dn,
      &format!("{:?}", dw.iter().zip(&dn).map(|(x, y)| (x.0, x.3, y.3, x.4 == y.4)).collect::<Vec<_>>()));

// A18 : invariance par bijection d'octets (anti-mémorisation forte)
let (e0, _) = fixture(WORDS, false, 7, false, false);
let sg = |b: u8| b ^ 0x20;
let e1: Vec<u8> = e0.iter().map(|&b| sg(b)).collect();
let (a0, r0) = understand(&e0, &router, &cfg, 0);
let (a1, r1) = understand(&e1, &router, &cfg, 0);
check(&mut results, "A18 invariance σ (réponse=σ(réponse), Φ identique)",
      a1 == a0.iter().map(|&b| sg(b)).collect::<Vec<u8>>() && r0.phi == r1.phi && r0.rung == r1.rung, "");

    // ---- A13 : ablation — sans abduction, pas de réponse ----------------
    let (e3, _) = fixture(WORDS, false, 7, false, false);
    let mut per3 = Perception::new(e3, &cfg);
    per3.ensure_frames();
    let lit_only = Cfg {
        completes: 1,
        pops: 1,
        ..cfg.clone()
    };
    let (s3, _) = router.score(&per3);
    let mut ok_abl = true;
    for &(x, y, _c, _r) in &per3.slots {
        let ab3 = abduce(&per3, x, y, &s3, &lit_only, &mut PyRandom::new(0), None);
        if ab3.cause.is_some() && ab3.phi > 0.0 {
            ok_abl = false;
        }
    }
    check(
        &mut results,
        "A13 ablation : budget dégénéré => aucune structure intégrée",
        ok_abl,
        "",
    );

    // ---- A14 : soundness de l'unification (bijection) -------------------
    let ca = Cause {
        nodes: vec![
            Node::lit(b"CB".to_vec()),
            Node::lit(b"DFF".to_vec()),
            Node::lit(b"AA".to_vec()),
            Node::seq(vec![0, 1, 0, 2]),
        ],
        root: 3,
    };
    let cb = Cause {
        nodes: vec![
            Node::lit(b"PM".to_vec()),
            Node::lit(b"NM".to_vec()),
            Node::lit(b"RMQ".to_vec()),
            Node::lit(b"ZZ".to_vec()),
            Node::seq(vec![0, 1, 2, 3]),
        ],
        root: 4,
    };
    check(
        &mut results,
        "A14 soundness unification (bijection sur partage)",
        unify(&ca, 3, &ca, 3, true)
            && !unify(&ca, 3, &cb, 4, true)
            && !unify(&cb, 4, &ca, 3, true),
        "",
    );

    // ---- A15 : invariance associative (normal_form) ---------------------
    let c1 = Cause {
        nodes: vec![
            Node::lit(b"CB".to_vec()),
            Node::lit(b"D".to_vec()),
            Node::lit(b"FF".to_vec()),
            Node::lit(b"AA".to_vec()),
            Node::seq(vec![0, 1, 2, 0, 3]),  // ÉTAIT [0,1,0,2,3]
        ],
        root: 4,
    };
    let c2 = Cause {
        nodes: vec![
            Node::lit(b"CB".to_vec()),
            Node::lit(b"DFF".to_vec()),
            Node::lit(b"AA".to_vec()),
            Node::seq(vec![0, 1, 0, 2]),
        ],
        root: 3,
    };
    // GARDE-FOU : deux causes ne sont comparables que si elles régénèrent le même effet.   
    let same_effect = values(&c1, None)[c1.root] == values(&c2, None)[c2.root];
    check(
        &mut results,
        "A15 invariance associative (même effet + signature & unify après normal_form)",
        same_effect && signature(&c1, 3) == signature(&c2, 3) && unify(&c1,     4, &c2, 3, true),
        &format!("same_effect={same_effect}"),
    );

// A21 : isole D4 — Φ doit être une FONCTION, au bit
{
    let (e, _) = fixture(WORDS, false, 7, false, false);
    let mut p = Perception::new(e, &cfg); p.ensure_frames();
    let (s, _) = router.score(&p);
    let ab = abduce(&p, p.slots[0].0, p.slots[0].1, &s, &cfg, &mut PyRandom::new(0), None);
    let c = ab.cause.as_ref().unwrap();
    let t = &p.e[p.slots[0].0..p.slots[0].1];
    let (p1, _) = phi(c, t, &mut PyRandom::new(11), 64, &[], 3);
    let (p2, _) = phi(c, t, &mut PyRandom::new(11), 64, &[], 3);
    check(&mut results, "A21 Φ déterministe au bit (même entrée, deux appels)",
          p1.to_bits() == p2.to_bits(), &format!("{:016x} vs {:016x}", p1.to_bits(), p2.to_bits()));
}

// A19 : monotonie du pipeline (vrai falsificateur de commensurabilité)
check(&mut results, "A19 P21 reach monotone en budget", tstats.q_gt_1 == 0,
      &format!("violations={}/{}", tstats.q_gt_1, tstats.solvable));

// A22 : MESURE — l'espace d'hypothèses manque-t-il ? (pas un gate)
println!("[MEAS] A22 espace : Φ(vérité)>reach sur {}/{} épisodes ; rejetés={}",
         tstats.space_gap, tstats.solvable, tstats.rejected);

// A8c : tolérance = résolution mesurée, n=128, McNemar exact
fn mcnemar_exact(b: usize, c: usize) -> f64 {
    let n = b + c; if n == 0 { return 1.0; }
    let k = b.min(c); let mut p = 0.5f64.powi(n as i32); let mut s = p;
    for i in 0..k { p *= (n - i) as f64 / (i + 1) as f64; s += p; }
    (2.0 * s).min(1.0)
}
let (mut hit_t, mut hit_u, mut bt, mut bu) = (0usize, 0usize, 0usize, 0usize);
for s in 0..128u64 {
    let (ee, _) = episode(&mut PyRandom::new(1000 + s), 60);
    let pe = Perception::new(ee.clone(), &cfg);
    let (st_, _) = router.score(&pe);
    let (su, _) = rand_router.score(&pe);
    let at = abduce(&pe, 0, ee.len(), &st_, &cfg,     &mut PyRandom::new(5), None);
    let au = abduce(&pe, 0, ee.len(), &su,  &cfg,     &mut PyRandom::new(5), None);
    let ar = abduce(&pe, 0, ee.len(), &st_, &cfg_ref, &mut PyRandom::new(5), Some(cfg.pops*4));
    let tol = at.eps.max(au.eps).max(ar.eps);
    let ht = at.reach >= ar.reach - tol;
    let hu = au.reach >= ar.reach - tol;
    hit_t += ht as usize; hit_u += hu as usize;
    if ht && !hu { bt += 1; } if hu && !ht { bu += 1; }
}
println!("[MEAS] A8c {hit_t}/128 vs {hit_u}/128  discordants=({bt},{bu})  p={:.4}",
         mcnemar_exact(bt, bu));
check(&mut results, "A8c non-infériorité du taux d'atteinte (n=128, tol=eps)",
      mcnemar_exact(bt, bu) >= 0.05 || bt >= bu, &format!("{hit_t} vs {hit_u}"));

// A23 : le drapeau --seed doit être opérant
let (e0, _) = fixture(WORDS, false, 7, false, false);
let (_, r0) = understand(&e0, &router, &cfg, 0);
let (_, r1) = understand(&e0, &router, &cfg, 1);
check(&mut results, "A23 --seed consommé (Φ varie avec la graine)", r0.phi != r1.phi, "");

    // A25 : LE falsificateur de π. Une couche de codec ne doit rien changer.
{
    let mut k = IDENT; k[b'C' as usize] = b'A'; k[b'D' as usize] = b'B';
    let c  = Cause { nodes: vec![Node::lit(b"AB".to_vec()), Node::rep(0, 2)], root: 1 };
    let cp = Cause { nodes: vec![Node::lit(b"CD".to_vec()), Node::rec(0, k), Node::rep(1, 2)], root: 2 };
    let e = b"ABAB";
    let f  = phi(&c,  e, &mut PyRandom::new(5), 48, &[], 3).0;
    let fp = phi(&cp, e, &mut PyRandom::new(5), 48, &[], 3).0;
    let g  = phi_pi(&c,  e, &mut PyRandom::new(5), 48, 3);
    let gp = phi_pi(&cp, e, &mut PyRandom::new(5), 48, 3);
    println!("[MEAS] A25 codec : Φ {f:.4}->{fp:.4} (×{:.2})   Φ_pi {g:.4}->{gp:.4} (×{:.2})", fp/f, gp/g);
    check(&mut results, "A25 π : Φ invariante par couche de codec", (gp - g).abs() <= 1e-9, "");
}

// A26 : MESURE — Φ récompense-t-elle le découpage fin ?
{
    let e = b"ABAB";
    let a = Cause { nodes: vec![Node::lit(e.to_vec())], root: 0 };
    let b = Cause { nodes: vec![Node::lit(b"AB".to_vec()), Node::rep(0,2)], root: 1 };
    let c = Cause { nodes: vec![Node::lit(b"A".to_vec()), Node::lit(b"B".to_vec()),
                                Node::seq(vec![0,1,0,1])], root: 2 };
    let v: Vec<f64> = [&a,&b,&c].iter()
        .map(|x| phi(x, e, &mut PyRandom::new(5), 48, &[], 3).0).collect();
    println!("[MEAS] A26 raffinement Lit/Rep/Seq : {v:?}");
}

// A20 : UNE seule entrée, statistique, sur le plafond-vérité
let (lo, hi) = wilson(tstats.q_ge_999, tstats.solvable);
check(&mut results, "A20 P20 régime atteint (q≈1 minoritaire, IC95 exclut 0.5)",
      tstats.solvable == tstats.episodes && hi < 0.5,
      &format!("q≈1={}/{} IC=[{lo:.3},{hi:.3}] gap={} rejetés={}",
               tstats.q_ge_999, tstats.solvable, tstats.space_gap, tstats.rejected));

// A8b : apparié (CRN), donc AUCUN plancher de bruit n'est requis — un IC sur n=256
let mut d: Vec<f64> = Vec::new();
for s in 0..256u64 {
    let (ee, _) = episode(&mut PyRandom::new(1000 + s), 60);
    let pe = Perception::new(ee.clone(), &cfg);
    let (st_, _) = router.score(&pe); let (su, _) = rand_router.score(&pe);
    let at = abduce(&pe, 0, ee.len(), &st_, &cfg, &mut PyRandom::new(5), None);
    let au = abduce(&pe, 0, ee.len(), &su,  &cfg, &mut PyRandom::new(5), None);
    d.push(at.reach - au.reach);
}
let m = d.iter().sum::<f64>() / d.len() as f64;
let sd = (d.iter().map(|x| (x-m).powi(2)).sum::<f64>() / (d.len()-1) as f64).sqrt();
let h = 1.96 * sd / (d.len() as f64).sqrt();
println!("[MEAS] A8b ΔΦ apparié = {m:+.4}  IC95=[{:+.4},{:+.4}]  (plancher MC inutile en CRN)", m-h, m+h);

    match fs::read("perceived.eff") {
        Ok(demo) => {
            let (e, _) = fixture(WORDS, false, 7, false, false);
            check(&mut results, "A16 perceived.eff == fixture valide (anti-drift)", demo == e, &format!("demo={} octets, attendu={}", demo.len(), e.len()));
        }
        Err(_) => check(&mut results, "A16 perceived.eff présent", false,
                    "absent — lancer `cargo run --release --bin gen`"),
    }

    let n_ok = results.iter().filter(|(_, c, _)| *c).count();
    println!("\n=== {n_ok}/{} PASS ===", results.len());
    if n_ok == results.len() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}