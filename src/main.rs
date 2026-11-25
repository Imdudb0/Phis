#![forbid(unsafe_code)]

mod ablation_placeholder {}

mod abduction;
mod ad;
mod eff;
mod emit;
mod eval;
mod model;
mod phi;
mod ptgrs;
mod rng;
mod substrate;
mod train;

use std::io::{self, Write};

use crate::model::{Model, CLOSED_EXPERIMENT};
use crate::rng::fnv1a;

fn argval(a: &[String], k: &str) -> Option<String> {
    for i in 0..a.len() {
        if a[i] == k && i + 1 < a.len() {
            return Some(a[i + 1].clone());
        }
    }
    None
}

fn argnum(a: &[String], k: &str, d: usize) -> usize {
    argval(a, k).and_then(|s| s.parse().ok()).unwrap_or(d)
}

/// Interface de requête, forme unique. Aucun gabarit, aucune consigne ajoutée.
/// stdout ne contient QUE les octets de l'explication.
fn cmd_ask(a: &[String]) -> io::Result<()> {
    let mp = argval(a, "--model").unwrap_or_else(|| "out/sharp.phis".into());
    let fp = argval(a, "--file").ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "--file requis"))?;
    let m = Model::load(&mp)?;
    let e = std::fs::read(&fp)?;
    let v = eff::build(&e, &m.view);
    let w = substrate::wire(&v);
    let mut sc = m.search;
    let mut ab = String::new();
    if let Ok(x) = std::env::var("PHISHARP_ABLATE") {
        ab = x.clone();
        match x.as_str() {
            "budget0" => sc.max_ops = 0,
            "norefute" => sc.allow_refutation = false,
            "mip2" => sc.phi.mip = 2,
            "mip6" => sc.phi.mip = 6,
            "mip12" => sc.phi.mip = 12,
            _ => ab.clear(),
        }
    }
    // Graine CONSTANTE : une graine dérivée des octets casserait l'invariance.
    let (ex, out) = emit::explain(&m.params, &e, &v, &w, &sc, &m.emit, m.seed);
    if let Ok(p) = std::env::var("PHISHARP_TRACE") {
        emit::write_diag(&p, m.fingerprint, &out, &ex)?;
        if !ab.is_empty() {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new().append(true).open(&p)?;
            writeln!(f, "ablate\t{}", ab)?;
        }
    }
    io::stdout().write_all(&ex.bytes)?;
    io::stdout().flush()?;
    Ok(())
}

fn cmd_train(a: &[String]) -> io::Result<()> {
    let out = argval(a, "--out").unwrap_or_else(|| "out/sharp.phis".into());
    let mut cfg = train::TrainCfg::default();
    cfg.episodes = argnum(a, "--episodes", cfg.episodes);
    cfg.batch = argnum(a, "--batch", cfg.batch);
    cfg.search.max_ops = argnum(a, "--max-ops", cfg.search.max_ops);
    cfg.infer.max_ops = argnum(a, "--infer-ops", cfg.infer.max_ops);
    if let Some(s) = argval(a, "--seed") {
        cfg.seed = s.parse().unwrap_or(cfg.seed);
    }
    if let Some(s) = argval(a, "--lr") {
        cfg.lr = s.parse().unwrap_or(cfg.lr);
    }
    let (m, rep) = train::train(&cfg, &out)?;
    let man = format!("{}.manifest.tsv", out);
    train::write_manifest(&man, &cfg, &rep)?;
    eprintln!(
        "modele {}  empreinte {:016x}  episodes {}  pas {}  phi_final {:.6}",
        CLOSED_EXPERIMENT, rep.fingerprint, rep.episodes, rep.steps, rep.final_phi_mean
    );
    eprintln!("manifeste {}", man);
    let _ = m;
    Ok(())
}

fn cmd_eval(a: &[String], dict: Option<&str>) -> io::Result<()> {
    let mp = argval(a, "--model").unwrap_or_else(|| "out/sharp.phis".into());
    let pp = argval(a, "--prereg").unwrap_or_else(|| "prereg.toml".into());
    let wk = argval(a, "--work").unwrap_or_else(|| "out/eval".into());
    let n = argval(a, "--n").and_then(|s| s.parse::<usize>().ok());
    let force = a.iter().any(|x| x == "--force");
    let rep = eval::run_all(&mp, &pp, &wk, dict, n, force)?;
    for l in rep.lines.iter() {
        println!("{}", l);
    }
    println!("verdict\t{}", rep.verdict());
    Ok(())
}

fn cmd_smoke(a: &[String]) -> io::Result<()> {
    let wk = argval(a, "--work").unwrap_or_else(|| "out/smoke".into());
    std::fs::create_dir_all(&wk)?;
    let mp = format!("{}/sharp.phis", wk);
    let mut cfg = train::TrainCfg::default();
    cfg.episodes = 4;
    cfg.batch = 2;
    cfg.search.max_ops = 2;
    cfg.infer.max_ops = 2;
    cfg.search.phi = phi::PhiCfg { mech: 3, purview: 3, mip: 2, perturb: true };
    cfg.infer.phi = cfg.search.phi;
    cfg.log_every = 0;
    let (_, rep) = train::train(&cfg, &mp)?;
    let pp = format!("{}/prereg.toml", wk);
    eval::Prereg::write_default(&pp)?;
    ptgrs::enter_eval();
    let mut rng = rng::Rng::new(1);
    let ep = ptgrs::episode(&ptgrs::FamilySpec::train(), &mut rng);
    let r = eval::ask(&mp, &ep.e, "", std::path::Path::new(&format!("{}/ask", wk)))?;
    println!("smoke\tfingerprint\t{:016x}", rep.fingerprint);
    println!("smoke\tstatus\t{}", r.status);
    println!("smoke\tops\t{}", r.diag.ops);
    println!("smoke\tphi\t{:.6}", r.diag.phi);
    println!("smoke\tspans\t{}", r.diag.nspans);
    println!("smoke\tout_bytes\t{}", r.out.len());
    println!("smoke\tinput_fingerprint\t{:016x}", fnv1a(&ep.e));
    Ok(())
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let cmd = a.first().cloned().unwrap_or_else(|| "help".into());
    let r = match cmd.as_str() {
        "ask" => cmd_ask(&a),
        "train" => cmd_train(&a),
        "prereg" => eval::Prereg::write_default(&argval(&a, "--out").unwrap_or_else(|| "prereg.toml".into())),
        "eval" => cmd_eval(&a, argval(&a, "--dict").as_deref()),
        "calib" => cmd_eval(&a, None),
        "blind" => eval::blind(
            &argval(&a, "--model").unwrap_or_else(|| "out/sharp.phis".into()),
            &argval(&a, "--dict").unwrap_or_else(|| "dict.tsv".into()),
            &argval(&a, "--work").unwrap_or_else(|| "out/blind".into()),
        ),
        "smoke" => cmd_smoke(&a),
        _ => {
            eprintln!(
                "phisharp\n\n  ask --model out/sharp.phis --file perceived.eff\n  \
                 train [--out P] [--episodes N] [--batch B] [--seed S] [--lr X] [--max-ops N] [--infer-ops N]\n  \
                 prereg [--out prereg.toml]\n  \
                 eval --model P [--prereg P] [--work D] [--dict F] [--n N] [--force]\n  \
                 blind --model P --dict F [--work D]\n  smoke [--work D]\n"
            );
            Ok(())
        }
    };
    if let Err(e) = r {
        eprintln!("erreur: {}", e);
        std::process::exit(1);
    }
}