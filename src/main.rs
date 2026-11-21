mod abduction;
mod explain;
mod graph;
mod lang;
mod model;
mod perception;
mod phi;
mod ptgrs;
mod rng;
mod train;

use std::fs;
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  phisharp train --out <model.phis> [--lang <file.lang>] [--iters N] [--seed S]");
    eprintln!("  phisharp gen   --out <perceived.eff> [--seed S]");
    eprintln!("  phisharp ask   --model <model.phis> --file <perceived.eff>");
    std::process::exit(2);
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { usage(); }
    let cmd = args[1].as_str();
    let mut opts: std::collections::HashMap<String, String> = Default::default();
    let mut i = 2;
    while i < args.len() {
        let a = &args[i];
        if a.starts_with("--") {
            let key = a.trim_start_matches("--").to_string();
            let val = if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                let v = args[i + 1].clone(); i += 2; v
            } else { i += 1; String::new() };
            opts.insert(key, val);
        } else { i += 1; }
    }

    match cmd {
        "train" => {
            let out = opts.get("out").cloned().unwrap_or_else(|| { usage() });
            let seed: u64 = opts.get("seed").and_then(|s| s.parse().ok()).unwrap_or(42);
            let iters: usize = opts.get("iters").and_then(|s| s.parse().ok()).unwrap_or(64);
            let mut rng = rng::Rng::new(seed);
            let mut m = model::Model::new(&mut rng);
            let mut dict = lang::bootstrap();
            if let Some(path) = opts.get("lang") {
                match lang::load_lang_file(path) {
                    Ok(extra) => dict.extend(extra),
                    Err(e) => { eprintln!("lang file: {e}"); return ExitCode::FAILURE; }
                }
            }
            m.load_lang(&dict, &mut rng);
            train::train_one_session(&mut m, iters, seed);
            if let Err(e) = m.save(&out) { eprintln!("save: {e}"); return ExitCode::FAILURE; }
            eprintln!("trained: terms={} concepts={} c2t={}", m.term_dict.len(), m.concept_emb.len(), m.c2t.len());
            ExitCode::SUCCESS
        }
        "gen" => {
            let out = opts.get("out").cloned().unwrap_or_else(|| { usage() });
            let seed: u64 = opts.get("seed").and_then(|s| s.parse().ok()).unwrap_or(7);
            let bytes = ptgrs::Ptgrs::default_gen().generate(seed);
            if let Err(e) = fs::write(&out, &bytes) { eprintln!("write: {e}"); return ExitCode::FAILURE; }
            ExitCode::SUCCESS
        }
        "ask" => {
            let mpath = opts.get("model").cloned().unwrap_or_else(|| { usage() });
            let fpath = opts.get("file").cloned().unwrap_or_else(|| { usage() });
            let m = match model::Model::load(&mpath) {
                Ok(m) => m,
                Err(e) => { eprintln!("load model: {e}"); return ExitCode::FAILURE; }
            };
            if !m.trained {
                eprintln!("warning: model not marked trained (proceeding anyway)");
            }
            let bytes = match fs::read(&fpath) {
                Ok(b) => b,
                Err(e) => { eprintln!("read E: {e}"); return ExitCode::FAILURE; }
            };
            let perc = perception::perceive(&bytes, &m);
            let abd = abduction::Abduction::new();
            let (hyp, phi) = abd.run(&m, &perc);
            println!("{}", explain::verbalize(&m, &hyp, 32));
            println!("[phi_sharp={:.6}]", phi);
            println!("[residue] {}", explain::residue(&m, &hyp, phi));
            ExitCode::SUCCESS
        }
        _ => usage(),
    }
}