use phisharp::net::{K_DEFAULT, K_MAX, K_MIN};
use phisharp::train::{self, Epoch, Plan};

const USAGE: &str = "usage : train [--out <fichier.phis>] [--k <unités>] [--seed <entier>] \
                     [--epochs <n>] [--windows <n>] [--block <n>]";
const SEED: u64 = 0x5048_4953;

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2)
}

fn main() {
    let mut path = String::from("out/sharp.phis");
    let mut k = K_DEFAULT;
    let mut seed = SEED;
    let (mut epochs, mut windows, mut block) = (None, None, None);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let v = args.next().unwrap_or_else(|| fail(USAGE));
        match a.as_str() {
            "--out" => path = v,
            "--k" => k = v.parse().unwrap_or_else(|_| fail(USAGE)),
            "--seed" => seed = v.parse().unwrap_or_else(|_| fail(USAGE)),
            "--epochs" => epochs = Some(v.parse().unwrap_or_else(|_| fail(USAGE))),
            "--windows" => windows = Some(v.parse().unwrap_or_else(|_| fail(USAGE))),
            "--block" => block = Some(v.parse().unwrap_or_else(|_| fail(USAGE))),
            _ => fail(USAGE),
        }
    }
    if !(K_MIN..=K_MAX).contains(&k) {
        fail(&format!("k doit être compris entre {K_MIN} et {K_MAX}"));
    }
    if k == 3 {
        eprintln!(
            "avertissement : k=3 ⇒ orient(3) est un 3-cycle unique ; seul le système entier \
             peut avoir φ>0, le postulat d'exclusion est vide."
        );
    }
    let mut plan = Plan::measured(k, seed);
    if let Some(v) = epochs { plan.epochs = v; }
    if let Some(v) = windows { plan.windows = v; }
    if let Some(v) = block { plan.block = v; }
    if plan.epochs == 0 || plan.windows == 0 || plan.block == 0 {
        fail("epochs, windows et block doivent être ≥ 1");
    }
    eprintln!("{}", train::header(&plan));
    let session = train::session(&plan, &mut |e: &Epoch| eprintln!("{}", train::progress(e)));
    if let Err(e) = session.model.save(&path) {
        fail(&e);
    }
    println!("{}", train::report(&session));
    println!("Modèle écrit : {path}");
}