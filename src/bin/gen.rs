#[path = "common/fixture.rs"] mod fix;
use std::process::ExitCode;

fn main() -> ExitCode {
    let (e, bodies) = fix::fixture(fix::WORDS, false, 7, false, false);
    if let Err(err) = std::fs::write("perceived.eff", &e) {
        eprintln!("write perceived.eff: {err}");
        return ExitCode::FAILURE;
    }
    eprintln!("perceived.eff : {} octets, corps requête = {:?}",
              e.len(), String::from_utf8_lossy(&bodies[3]));
    ExitCode::SUCCESS
}