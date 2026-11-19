use std::fs;
use std::io::{self, Write};

use crate::core::Cfg;
use crate::model::{load, save, train, understand};

fn take_opt_val(args: &[String], i: &mut usize, name: &str) -> Result<String, String> {
    if let Some(rest) = args[*i].strip_prefix(&format!("{name}=")) {
        return Ok(rest.to_string());
    }
    if *i + 1 >= args.len() {
        return Err(format!("missing value for {name}"));
    }
    *i += 1;
    Ok(args[*i].clone())
}

pub fn run(args: &[String]) -> i32 {
    if args.is_empty() {
        eprintln!("usage: phisharp {{ask,train}} ...");
        return 2;
    }

    match args[0].as_str() {
        "train" => {
            let mut out = String::from("out/sharp.phis");
            let mut episodes = 240usize;
            let mut seed = 1u64;
            let mut i = 1usize;
            while i < args.len() {
                let arg = &args[i];
                if arg == "--out" || arg.starts_with("--out=") {
                    match take_opt_val(args, &mut i, "--out") {
                        Ok(v) => out = v,
                        Err(e) => {
                            eprintln!("{e}");
                            return 2;
                        }
                    }
                } else if arg == "--episodes" || arg.starts_with("--episodes=") {
                    match take_opt_val(args, &mut i, "--episodes")
                        .and_then(|v| v.parse::<usize>().map_err(|e| e.to_string()))
                    {
                        Ok(v) => episodes = v,
                        Err(e) => {
                            eprintln!("invalid --episodes: {e}");
                            return 2;
                        }
                    }
                } else if arg == "--seed" || arg.starts_with("--seed=") {
                    match take_opt_val(args, &mut i, "--seed")
                        .and_then(|v| v.parse::<u64>().map_err(|e| e.to_string()))
                    {
                        Ok(v) => seed = v,
                        Err(e) => {
                            eprintln!("invalid --seed: {e}");
                            return 2;
                        }
                    }
                } else {
                    eprintln!("unknown argument: {arg}");
                    return 2;
                }
                i += 1;
            }

            let cfg = Cfg::default();
            let log_fn = |e: usize, rw: f64, ph: f64, po: usize| {
                eprintln!("ep {e:4}  r={rw:+.3}  phi={ph:.3}  pops={po}");
            };
            let r = train(&cfg, episodes, seed, Some(&log_fn));
            if let Err(e) = save(&r, &cfg, &out) {
                eprintln!("save error: {e}");
                return 1;
            }
            eprintln!("saved {out}");
            0
        }
        "ask" => {
            let mut model: Option<String> = None;
            let mut file: Option<String> = None;
            let mut json_flag = false;
            let mut seed = 0u64;
            let mut i = 1usize;
            while i < args.len() {
                let arg = &args[i];
                if arg == "--model" || arg.starts_with("--model=") {
                    match take_opt_val(args, &mut i, "--model") {
                        Ok(v) => model = Some(v),
                        Err(e) => {
                            eprintln!("{e}");
                            return 2;
                        }
                    }
                } else if arg == "--file" || arg.starts_with("--file=") {
                    match take_opt_val(args, &mut i, "--file") {
                        Ok(v) => file = Some(v),
                        Err(e) => {
                            eprintln!("{e}");
                            return 2;
                        }
                    }
                } else if arg == "--json" {
                    json_flag = true;
                } else if arg == "--seed" || arg.starts_with("--seed=") {
                    match take_opt_val(args, &mut i, "--seed")
                        .and_then(|v| v.parse::<u64>().map_err(|e| e.to_string()))
                    {
                        Ok(v) => seed = v,
                        Err(e) => {
                            eprintln!("invalid --seed: {e}");
                            return 2;
                        }
                    }
                } else {
                    eprintln!("unknown argument: {arg}");
                    return 2;
                }
                i += 1;
            }

            let Some(model_path) = model else {
                eprintln!("--model is required");
                return 2;
            };
            let Some(file_path) = file else {
                eprintln!("--file is required");
                return 2;
            };

            let (router, cfg) = match load(&model_path) {
                Ok(pair) => pair,
                Err(e) => {
                    eprintln!("load error: {e}");
                    return 1;
                }
            };
            let e_bytes = match fs::read(&file_path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("read error: {e}");
                    return 1;
                }
            };

            let (answer, report) = understand(&e_bytes, &router, &cfg, seed);
            if json_flag {
                let s = report.to_json(Some(&answer)).to_pretty_string();
                let mut stdout = io::stdout().lock();
                if let Err(e) = writeln!(stdout, "{s}") {
                    eprintln!("write error: {e}");
                    return 1;
                }
            } else {
                let mut stdout = io::stdout().lock();
                if stdout.write_all(&answer).and_then(|_| stdout.write_all(b"\n")).is_err() {
                    return 1;
                }
                let rung = report.rung.as_deref().unwrap_or("None");
                eprintln!("[rung={rung}  pops={}]", report.pops);
            }
            0
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            2
        }
    }
}