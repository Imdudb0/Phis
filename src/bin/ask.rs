use phisharp::explain;
use phisharp::model::Model;
use phisharp::search::Step;
use std::io::Write;

const USAGE: &str = "usage : ask --model <fichier.phis> --file <perceived.eff>";

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2)
}

fn main() {
    let mut model_path: Option<String> = None;
    let mut file_path: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let slot = match a.as_str() {
            "--model" => &mut model_path,
            "--file" => &mut file_path,
            _ => fail(USAGE),
        };
        let v = args.next().unwrap_or_else(|| fail(USAGE));
        if slot.replace(v).is_some() {
            fail(USAGE);
        }
    }
    let (Some(model_path), Some(file_path)) = (model_path, file_path) else {
        fail(USAGE)
    };
    let model = Model::load(&model_path).unwrap_or_else(|e| fail(&e));
    let answer = model
        .ask_file(&file_path, &mut |s: &Step| eprintln!("{}", explain::progress(s)))
        .unwrap_or_else(|e| fail(&e));
    let report = explain::report(&answer.percept, &answer.outcome);
    let mut out = std::io::stdout().lock();
    let done = out.write_all(&report).and_then(|()| out.flush());
    if done.is_err() {
        std::process::exit(1);
    }
}