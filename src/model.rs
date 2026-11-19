//! understand() : abduction -> Φ -> méta-relation -> réponse ⊆ A(E).

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::Path;

use crate::algebra::{signature, unify};
use crate::core::{abduce, residual, Abduction, Cfg, Perception, Residual, Router};
use crate::ptgrs::episode;
use crate::json::Json;
use crate::rng::PyRandom;
use crate::phi::phi;

#[derive(Clone, Debug)]
pub struct Matched {
    pub def_record: usize,
    pub body: (usize, usize),
    pub name: (usize, usize),
}

#[derive(Clone, Debug)]
pub struct Report {
    pub phi: BTreeMap<(usize, usize), f64>,
    pub pops: usize,
    pub records: Vec<(usize, usize)>,
    pub slots: Vec<(usize, usize, usize, usize)>,
    pub frame: Vec<Vec<u8>>,
    pub residual: BTreeMap<(usize, usize), Residual>,
    pub rung: Option<String>,
    pub matched: Option<Matched>,
}

fn bytes_to_latin1(bs: &[u8]) -> String {
    bs.iter().map(|&b| b as char).collect()
}

fn round4(x: f64) -> f64 {
    let r = (x * 10000.0).round() / 10000.0;
    if r == 0.0 {
        0.0
    } else {
        r
    }
}

impl Report {
    pub fn to_json(&self, answer: Option<&[u8]>) -> Json {
        let phi_obj = Json::Obj(
            self.phi
                .iter()
                .map(|(&(r, c), &v)| (format!("{r}:{c}"), Json::Float(round4(v))))
                .collect(),
        );
        let records_arr = Json::Arr(
            self.records
                .iter()
                .map(|&(a, b)| Json::Arr(vec![Json::Int(a as i64), Json::Int(b as i64)]))
                .collect(),
        );
        let slots_arr = Json::Arr(
            self.slots
                .iter()
                .map(|&(a, b, c, r)| {
                    Json::Arr(vec![
                        Json::Int(a as i64),
                        Json::Int(b as i64),
                        Json::Int(c as i64),
                        Json::Int(r as i64),
                    ])
                })
                .collect(),
        );
        let frame_arr = Json::Arr(
            self.frame
                .iter()
                .map(|p| Json::Str(bytes_to_latin1(p)))
                .collect(),
        );
        let res_obj = Json::Obj(
            self.residual
                .iter()
                .map(|(&(r, c), v)| {
                    let pairs = |lst: &[(usize, usize)]| -> Json {
                        Json::Arr(
                            lst.iter()
                                .map(|&(x, y)| {
                                    Json::Arr(vec![Json::Int(x as i64), Json::Int(y as i64)])
                                })
                                .collect(),
                        )
                    };
                    let triples = |lst: &[(usize, usize, Vec<usize>)]| -> Json {
                        Json::Arr(
                            lst.iter()
                                .map(|(a, n, vs)| {
                                    Json::Arr(vec![
                                        Json::Int(*a as i64),
                                        Json::Int(*n as i64),
                                        Json::Arr(
                                            vs.iter().map(|&x| Json::Int(x as i64)).collect(),
                                        ),
                                    ])
                                })
                                .collect(),
                        )
                    };
                    let obj = Json::Obj(vec![
                        ("GAP".to_string(), pairs(&v.gap)),
                        ("INERT".to_string(), pairs(&v.inert)),
                        ("UNDET".to_string(), triples(&v.undet)),
                        ("OPEN".to_string(), triples(&v.open)),
                    ]);
                    (format!("{r}:{c}"), obj)
                })
                .collect(),
        );
        let rung_val = match &self.rung {
            Some(s) => Json::Str(s.clone()),
            None => Json::Null,
        };
        let matched_val = match &self.matched {
            Some(m) => Json::Obj(vec![
                ("def_record".to_string(), Json::Int(m.def_record as i64)),
                (
                    "body".to_string(),
                    Json::Arr(vec![
                        Json::Int(m.body.0 as i64),
                        Json::Int(m.body.1 as i64),
                    ]),
                ),
                (
                    "name".to_string(),
                    Json::Arr(vec![
                        Json::Int(m.name.0 as i64),
                        Json::Int(m.name.1 as i64),
                    ]),
                ),
            ]),
            None => Json::Null,
        };

        let mut fields = vec![
            ("phi".to_string(), phi_obj),
            ("pops".to_string(), Json::Int(self.pops as i64)),
            ("records".to_string(), records_arr),
            ("slots".to_string(), slots_arr),
            ("frame".to_string(), frame_arr),
            ("residual".to_string(), res_obj),
            ("rung".to_string(), rung_val),
            ("matched".to_string(), matched_val),
        ];
        if let Some(ans) = answer {
            fields.push(("answer".to_string(), Json::Str(bytes_to_latin1(ans))));
        }
        Json::Obj(fields)
    }
}

fn open_slot(per: &Perception) -> Option<(usize, (usize, usize, usize))> {
    let mut cols: BTreeMap<usize, Vec<(usize, usize, usize)>> = BTreeMap::new();
    for &(a, b, c, r) in &per.slots {
        cols.entry(c).or_default().push((a, b, r));
    }

    let mut fallback: Option<(usize, (usize, usize, usize))> = None;
    for (c, lst) in cols {
        let sets: Vec<(HashSet<u8>, (usize, usize, usize))> = lst
            .into_iter()
            .map(|(a, b, r)| (per.e[a..b].iter().copied().collect(), (a, b, r)))
            .collect();
        let coherent = (0..sets.len()).any(|i| {
            ((i + 1)..sets.len()).any(|j| !sets[i].0.is_disjoint(&sets[j].0))
        });
        for (t, (s1, abr)) in sets.iter().enumerate() {
            if s1.is_empty() {
                continue;
            }
            let others: Vec<&HashSet<u8>> = sets
                .iter()
                .enumerate()
                .filter(|&(u, (s2, _))| u != t && !s2.is_empty())
                .map(|(_, (s2, _))| s2)
                .collect();
            if !others.is_empty() && others.iter().all(|s2| s1.is_disjoint(s2)) {
                if coherent {
                    return Some((c, *abr));
                }
                if fallback.is_none() {
                    fallback = Some((c, *abr));
                }
            }
        }
    }
    fallback
}

fn col_partner_score(per: &Perception, col: usize) -> usize {
    let mut tot = 0usize;
    for &(a, b, c, _r) in &per.slots {
        if c != col {
            continue;
        }
        if let Some(&idx) = per.item_at.get(&(a, b)) {
            tot += per.partner.get(&idx).map_or(0, |v| v.len());
        }
    }
    tot
}

pub fn understand(e: &[u8], router: &Router, cfg: &Cfg, seed: u64) -> (Vec<u8>, Report) {
    let mut rng = PyRandom::new(seed);
    let mut per = Perception::new(e.to_vec(), cfg);
    per.ensure_frames();
    let (scores, _hm) = router.score(&per);

    let mut causes: BTreeMap<(usize, usize), Abduction> = BTreeMap::new();
    let mut resids: BTreeMap<(usize, usize), Residual> = BTreeMap::new();
    let mut pops_total = 0usize;

    for &(a, b, col, ri) in &per.slots {
        let ab = abduce(&per, a, b, &scores, cfg, &mut rng, None);
        pops_total += ab.pops;
        let res = residual(&per, a, b, &ab);
        causes.insert((ri, col), ab);
        resids.insert((ri, col), res);
    }

    let mut report = Report {
        phi: causes.iter().map(|(&k, v)| (k, round4(v.phi))).collect(),
        pops: pops_total,
        records: per.records.clone(),
        slots: per.slots.clone(),
        frame: per.frame.clone(),
        residual: resids,
        rung: None,
        matched: None,
    };

    let oslot = open_slot(&per);
    let Some((ocol, (qa, qb, qrec))) = oslot else {
        report.rung = Some("self-explain".to_string());
        return (e.to_vec(), report);
    };

    let candidates: Vec<usize> = per
        .slots
        .iter()
        .map(|&(_, _, c, _)| c)
        .filter(|&c| c != ocol)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if candidates.is_empty() {
        return (e.to_vec(), report);
    }

    let mut bcol = candidates[0];
    let mut best_col_score = col_partner_score(&per, bcol);
    for &c in &candidates[1..] {
        let s = col_partner_score(&per, c);
        if s > best_col_score {
            best_col_score = s;
            bcol = c;
        }
    }

    let Some(qab) = causes.get(&(qrec, bcol)) else {
        report.rung = Some("no-cause".to_string());
        return (e.to_vec(), report);
    };
    let Some(ref qcause) = qab.cause else {
        report.rung = Some("no-cause".to_string());
        return (e.to_vec(), report);
    };
    let qsig = signature(qcause, 3);

    let mut best_name: Option<Vec<u8>> = None;
    let mut best_rung: Option<&'static str> = None;
    let mut best_score = (-1.0f64, -1.0f64);

    for &(a, b, c, r) in &per.slots {
        if c != bcol || r == qrec {
            continue;
        }
        let Some(dab) = causes.get(&(r, bcol)) else {
            continue;
        };
        let Some(ref dcause) = dab.cause else {
            continue;
        };

        let rung = if unify(dcause, dcause.root, qcause, qcause.root, true) {
            "unify"
        } else if signature(dcause, 3) == qsig {
            "signature"
        } else {
            continue;
        };

        let sc = (dab.phi + qab.phi, dab.phi);
        let gt = sc
            .0
            .total_cmp(&best_score.0)
            .then_with(|| sc.1.total_cmp(&best_score.1))
            == Ordering::Greater;
        if gt {
            if let Some(&(na, nb, _, _)) = per
                .slots
                .iter()
                .find(|&&(_, _, cc, rr)| rr == r && cc == ocol)
            {
                best_score = sc;
                best_rung = Some(rung);
                best_name = Some(per.e[na..nb].to_vec());
                report.matched = Some(Matched {
                    def_record: r,
                    body: (a, b),
                    name: (na, nb),
                });
            }
        }
    }

    let Some(name_bytes) = best_name else {
        report.rung = Some("no-match".to_string());
        return (e.to_vec(), report);
    };

    let (ra, rb) = per.records[qrec];
    let mut answer = Vec::with_capacity((qa - ra) + name_bytes.len() + (rb - qb));
    answer.extend_from_slice(&per.e[ra..qa]);
    answer.extend_from_slice(&name_bytes);
    answer.extend_from_slice(&per.e[qb..rb]);
    report.rung = best_rung.map(str::to_string);

    let alpha_e: HashSet<u8> = e.iter().copied().collect();
    assert!(
        answer.iter().all(|b| alpha_e.contains(b)),
        "violation de clôture d'alphabet"
    );
    (answer, report)
}

/// UNE SEULE séance : une passe, chaque épisode vu exactement une fois.
pub fn train(
    cfg: &Cfg,
    episodes: usize,
    seed: u64,
    log: Option<&dyn Fn(usize, f64, f64, usize)>,
) -> Router {
    let mut router = Router::new(cfg.d, cfg.h, seed);
    let mut rng = PyRandom::new(seed);
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut phi_ref: f64 = 1.0;  // référence glissante, non saturante
    
    for ep in 0..episodes {
        let (e, truth) = episode(&mut rng, 60);
        assert!(!seen.contains(&e), "violation une-seule-passe");
        seen.insert(e.clone());

        let per = Perception::new(e.clone(), cfg);
        let (scores, hm) = router.score(&per);
        let ab = abduce(&per, 0, e.len(), &scores, cfg, &mut rng, None);
        
        let (phi_oracle, _) = phi(&truth, &e, &mut PyRandom::new(seed ^ 0x5eed), 32, &[], 3);
        let q    = ab.phi / (phi_oracle + 1.0);          // LINÉAIRE en Φ, aucun plateau
        let cost = ab.pops as f64 / cfg.pops.max(1) as f64;
        let r    = q - 0.05 * cost;
        
        phi_ref  = 0.95 * phi_ref + 0.05 * ab.phi;
        router.update(&per, &hm, &ab.used, r, 0.05);
        if ep % 20 == 0 { eprintln!("r range check: q={q:.3} cost={cost:.3} r={r:.3}"); }
        if let Some(logger) = log {
            if ep % 20 == 0 {
                logger(ep, r, ab.phi, ab.pops);
            }
        }
    }
    router
}

fn mat_to_json(m: &[Vec<f64>]) -> Json {
    Json::Arr(
        m.iter()
            .map(|row| Json::Arr(row.iter().map(|&v| Json::Float(v)).collect()))
            .collect(),
    )
}

fn json_to_mat(v: &Json, rows: usize, cols: usize, name: &str) -> Result<Vec<Vec<f64>>, String> {
    let arr = v
        .as_arr()
        .ok_or_else(|| format!("{name} must be an array"))?;
    if arr.len() != rows {
        return Err(format!("{name} row count mismatch"));
    }
    let mut out = Vec::with_capacity(rows);
    for row_j in arr {
        let r = row_j
            .as_arr()
            .ok_or_else(|| format!("{name} row must be an array"))?;
        if r.len() != cols {
            return Err(format!("{name} col count mismatch"));
        }
        let mut vals = Vec::with_capacity(cols);
        for item in r {
            vals.push(
                item.as_f64()
                    .ok_or_else(|| format!("{name} element must be number"))?,
            );
        }
        out.push(vals);
    }
    Ok(out)
}

pub fn save(router: &Router, cfg: &Cfg, path: &str) -> Result<(), String> {
    let p = Path::new(path);
    if let Some(parent) = p.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let cfg_json = Json::Obj(vec![
        ("Lmax".to_string(), Json::Int(cfg.lmax as i64)),
        (
            "max_atoms_per_item".to_string(),
            Json::Int(cfg.max_atoms_per_item as i64),
        ),
        ("pops".to_string(), Json::Int(cfg.pops as i64)),
        ("completes".to_string(), Json::Int(cfg.completes as i64)),
        ("phi_N".to_string(), Json::Int(cfg.phi_n as i64)),
        ("phi_N_final".to_string(), Json::Int(cfg.phi_n_final as i64)),
        ("lift_cap".to_string(), Json::Int(cfg.lift_cap as i64)),
        ("d".to_string(), Json::Int(cfg.d as i64)),
        ("h".to_string(), Json::Int(cfg.h as i64)),
    ]);
    let root = Json::Obj(vec![
        ("cfg".to_string(), cfg_json),
        ("Wf".to_string(), mat_to_json(&router.wf)),
        ("Wp".to_string(), mat_to_json(&router.wp)),
        ("Wb".to_string(), mat_to_json(&router.wb)),
        (
            "u".to_string(),
            Json::Arr(router.u.iter().map(|&v| Json::Float(v)).collect()),
        ),
        ("baseline".to_string(), Json::Float(router.baseline)),
        ("m2".to_string(), Json::Float(router.m2)),
    ]);
    fs::write(p, root.to_pretty_string()).map_err(|e| e.to_string())
}

pub fn load(path: &str) -> Result<(Router, Cfg), String> {
    let raw = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let root = Json::parse(&raw)?;
    let cfg_j = root.get("cfg").ok_or("missing cfg")?;
    let get_u = |k: &str| -> Result<usize, String> {
        cfg_j
            .get(k)
            .and_then(Json::as_usize)
            .ok_or_else(|| format!("missing or invalid cfg.{k}"))
    };
    let cfg = Cfg {
        lmax: get_u("Lmax")?,
        max_atoms_per_item: get_u("max_atoms_per_item")?,
        pops: get_u("pops")?,
        completes: get_u("completes")?,
        phi_n: get_u("phi_N")?,
        phi_n_final: get_u("phi_N_final")?,
        lift_cap: get_u("lift_cap")?,
        d: get_u("d")?,
        h: get_u("h")?,
    };
    let wf = json_to_mat(root.get("Wf").ok_or("missing Wf")?, cfg.d, cfg.h, "Wf")?;
    let wp = json_to_mat(root.get("Wp").ok_or("missing Wp")?, cfg.d, cfg.h, "Wp")?;
    let wb = json_to_mat(root.get("Wb").ok_or("missing Wb")?, cfg.d, cfg.h, "Wb")?;
    let u_arr = root
        .get("u")
        .and_then(Json::as_arr)
        .ok_or("missing u")?;
    if u_arr.len() != cfg.h {
        return Err("u length mismatch".to_string());
    }
    let mut u = Vec::with_capacity(cfg.h);
    for item in u_arr {
        u.push(item.as_f64().ok_or("u element must be number")?);
    }
    let baseline = root
        .get("baseline")
        .and_then(Json::as_f64)
        .ok_or("missing baseline")?;    
    let m2 = root.get("m2").and_then(Json::as_f64).unwrap_or(0.0);
    Ok((
        Router {
            d: cfg.d,
            h: cfg.h,
            wf,
            wp,
            wb,
            u,
            baseline,
            m2,
        },
        cfg,
    ))
}