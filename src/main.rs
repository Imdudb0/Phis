#![allow(dead_code)]
mod abstraction; mod ask; mod dynamics; mod graph; mod model; mod phi; mod ptgrs; mod train;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}
fn usage() {
    eprintln!("phisharp — compréhension par abduction");
    eprintln!("  ask   --model out/sharp.phis --file perceived.eff");
    eprintln!("  train --out out/sharp.phis [--samples 48] [--seed 7]");
    eprintln!("  emit  --out perceived.eff [--seed 3] [--depth 6]");
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = match a.get(1).map(|s| s.as_str()) {
        Some("ask") => {
            let m = flag(&a[2..], "--model"); let f = flag(&a[2..], "--file");
            match (m, f) {
                (Some(m), Some(f)) => ask::run(&m, &f),
                _ => { usage(); Err("ask requiert --model et --file".into()) }
            }
        }
        Some("train") => {
            let out = flag(&a[2..], "--out").unwrap_or_else(|| "out/sharp.phis".into());
            let samples: usize = flag(&a[2..], "--samples").and_then(|s| s.parse().ok()).unwrap_or(48);
            let seed: u64 = flag(&a[2..], "--seed").and_then(|s| s.parse().ok()).unwrap_or(7);
            train::run(&out, samples, seed)
        }
        Some("emit") => {
            let out = flag(&a[2..], "--out").unwrap_or_else(|| "perceived.eff".into());
            let seed: u64 = flag(&a[2..], "--seed").and_then(|s| s.parse().ok()).unwrap_or(3);
            let depth: usize = flag(&a[2..], "--depth").and_then(|s| s.parse().ok()).unwrap_or(6);
            let mut rng = graph::Rng::new(seed);
            let (e, _cause) = ptgrs::generate(&mut rng, depth);
            match std::fs::write(&out, &e) {
                Ok(()) => { eprintln!("E écrit: {out} ({} octets, cause cachée non divulguée)", e.len()); Ok(()) }
                Err(x) => Err(format!("write {out}: {x}")),
            }
        }
        _ => { usage(); Ok(()) }
    };
    if let Err(e) = r { eprintln!("erreur: {e}"); std::process::exit(1); }
}