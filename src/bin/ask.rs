use phisharp::model::controller::readout_infer;
use phisharp::model::phis::load_phis;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;

fn print_usage_and_exit() -> ! {
    let _ = writeln!(
        io::stderr(),
        "Usage: ask --model <out/sharp.phis> --file <perceived.eff>"
    );
    process::exit(2);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut model_path: Option<String> = None;
    let mut file_path: Option<String> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                if i >= args.len() {
                    print_usage_and_exit();
                }
                model_path = Some(args[i].clone());
            }
            "--file" => {
                i += 1;
                if i >= args.len() {
                    print_usage_and_exit();
                }
                file_path = Some(args[i].clone());
            }
            _ => print_usage_and_exit(),
        }
        i += 1;
    }

    let model_path = match model_path {
        Some(p) => p,
        None => print_usage_and_exit(),
    };
    let file_path = match file_path {
        Some(p) => p,
        None => print_usage_and_exit(),
    };

    let (_cfg, params, store) = match load_phis(&model_path) {
        Ok(loaded) => loaded,
        Err(err) => {
            let _ = writeln!(
                io::stderr(),
                "Erreur chargement modèle '{}': {}",
                model_path,
                err
            );
            process::exit(1);
        }
    };

    let perceived_bytes = match fs::read(&file_path) {
        Ok(b) => b,
        Err(err) => {
            let _ = writeln!(
                io::stderr(),
                "Erreur lecture fichier '{}': {}",
                file_path,
                err
            );
            process::exit(1);
        }
    };

    let output = readout_infer(&store, &params, &perceived_bytes);
    if let Err(err) = io::stdout().write_all(&output) {
        let _ = writeln!(io::stderr(), "Erreur écriture sortie: {}", err);
        process::exit(1);
    }
    let _ = io::stdout().flush();
}