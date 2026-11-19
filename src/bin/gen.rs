#[path = "common/fixture.rs"] mod fixture;
fn main() -> std::process::ExitCode {
    let (e, bodies) = fixture::fixture(fixture::WORDS, false, 7, false, false);
    std::fs::write("perceived.eff", &e).expect("write perceived.eff");
    eprintln!("perceived.eff écrit, corps requête = {:?}", String::from_utf8_lossy(&bodies[3]));
    std::process::ExitCode::SUCCESS
}