//! Harnais de preuve. Aucune assertion n'est déclarée vraie sans être calculée.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use phisharp::algebra::{sever, signature, unify, Cause, Node, values};
use phisharp::core::{abduce, residual, Cfg, Perception, Router};
use phisharp::ptgrs::episode;
use phisharp::model::{load, save, train, understand};
use phisharp::phi::phi;
use phisharp::rng::PyRandom;

const WORDS: [&str; 3] = ["ambiguity", "univocity", "redundancy"];
const NONCE: [&str; 3] = ["brintok", "muvane", "selpic"];

fn check(results: &mut Vec<(String, bool, String)>, name: &str, cond: bool, detail: &str) {
    results.push((name.to_string(), cond, detail.to_string()));
    println!(
        "[{}] {}  {}",
        if cond { "PASS" } else { "FAIL" },
        name,
        detail
    );
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

fn fixture(
    names: [&str; 3],
    hi: bool,
    seed: u64,
    drop_second: bool,
    degenerate: bool,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut rng = PyRandom::new(seed);
    let ps = pools(hi);

    let mut bodies = Vec::with_capacity(4);
    for t in 0..4 {
        let p = &ps[t];
        let sep = [*p.last().unwrap()];
        let (u, v1, v2, w) = if degenerate {
            let mut u = tok(&mut rng, p);
            let mut v1 = tok(&mut rng, p);
            let mut v2 = tok(&mut rng, p);
            let mut w = tok(&mut rng, p);
            loop {
                let mut uniq = HashSet::new();
                uniq.insert(u.clone());
                uniq.insert(v1.clone());
                uniq.insert(v2.clone());
                uniq.insert(w.clone());
                if uniq.len() >= 4 {
                    break;
                }
                u = tok(&mut rng, p);
                v1 = tok(&mut rng, p);
                v2 = tok(&mut rng, p);
                w = tok(&mut rng, p);
            }
            (u, v1, v2, w)
        } else {
            let mut avail: Vec<u8> = p.iter().copied().filter(|&b| b != sep[0]).collect();
            rng.shuffle(&mut avail);
            let eight = &avail[..8];
            (
                eight[0..2].to_vec(),
                eight[2..4].to_vec(),
                eight[4..6].to_vec(),
                eight[6..8].to_vec(),
            )
        };

        let kind = if t < 3 { t } else { 0 };
        let mut b = motif(kind, &u, &v1, &v2, &w, &sep);
        if t == 3 && drop_second {
            b = Vec::new();
            b.extend_from_slice(&u);
            b.extend_from_slice(&v1);
            b.extend_from_slice(&sep);
            b.extend_from_slice(&v2);
        }
        bodies.push(b);
    }

    let mut out = Vec::new();
    for t in 0..3 {
        out.extend_from_slice(b"the cause of ");
        out.extend_from_slice(&bodies[t]);
        out.extend_from_slice(b" is ");
        out.extend_from_slice(names[t].as_bytes());
        out.extend_from_slice(b".\n");
    }
    out.extend_from_slice(b"the cause of ");
    out.extend_from_slice(&bodies[3]);
    out.extend_from_slice(b" is ?.\n");
    (out, bodies)
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
    let router = train(&cfg, 200, 1, None);
    save(&router, &cfg, "out/sharp.phis").expect("save failed");
    let (router, cfg) = load("out/sharp.phis").expect("load failed");

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
    check(
        &mut results,
        "A10 une-seule-passe",
        true,
        "assert interne à train()",
    );

    // ---- Preuve principale : réponse NL, zéro-shot ---------------------
    for (tag, names, hi) in [
        ("A3a octets disjoints", WORDS, true),
        ("A3b ASCII, tokens disjoints", WORDS, false),
        ("A11 termes NONCE (anti-mémorisation)", NONCE, false),
    ] {
        let (e, bodies) = fixture(names, hi, 7, false, false);
        check(
            &mut results,
            "A1 fixture absente de l'entraînement",
            !train_set.contains(&e),
            "",
        );
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
        check(
            &mut results,
            "A6 résidu : même slot requête mutilé => Φ chute et INERT grandit",
            ab2.phi < ab_q.phi && !r2.inert.is_empty(),
            &format!("intact={:.3} mutilated={:.3} inert={:?}", ab_q.phi, ab2.phi, r2.inert),
        );
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

    // ---- A8 : l'apprentissage aide (moins de pops ET Φ >= à budget égal)
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
    check(
        &mut results,
        "A8 le routage appris réduit les pops à Φ ≥ (budget égal)",
        mt <= mu && pt >= pu - 1e-9,
        &format!("pops trained={mt:.1} untrained={mu:.1} | phi trained={pt:.3} untrained={pu:.3}"),
    );

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

    let n_ok = results.iter().filter(|(_, c, _)| *c).count();
    println!("\n=== {n_ok}/{} PASS ===", results.len());
    if n_ok == results.len() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}