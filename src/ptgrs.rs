//! PTGRS : Probabilistic Term Graph Rewriting System. Ne fabrique QUE E
//! (bytes). La cause cachée n'existe pas côté générateur : elle est
//! seulement évaluée après coup par φ_sharp.

use crate::rng::Rng;

#[derive(Clone)]
pub enum Pat {
    Byte(u8),
    Seq(Vec<Pat>),
    Choice(Vec<(Pat, f32)>),
    Repeat(Box<Pat>, usize),
}

impl Pat {
    fn expand(&self, rng: &mut Rng, out: &mut Vec<u8>) {
        match self {
            Pat::Byte(b) => out.push(*b),
            Pat::Seq(ps) => for p in ps { p.expand(rng, out); },
            Pat::Choice(cs) => {
                let total: f32 = cs.iter().map(|(_, p)| *p).sum();
                let mut x = rng.next_f32() * total;
                for (p, w) in cs {
                    x -= *w;
                    if x <= 0.0 { p.expand(rng, out); return; }
                }
                cs.last().unwrap().0.expand(rng, out);
            }
            Pat::Repeat(p, n) => for _ in 0..*n { p.expand(rng, out); },
        }
    }
}

pub struct Ptgrs {
    pub start: Pat,
    pub seed: u64,
}

impl Ptgrs {
    pub fn default_gen() -> Self {
        // Générateur minimal mais non trivial : produit des "phrases"
        // composées de mots de longueurs variables séparés par des espaces
        // et ponctuées. Volontairement varié pour tester φ_sharp.
        let word = || Pat::Repeat(
            Box::new(Pat::Choice(vec![
                (Pat::Byte(b'a'), 1.0), (Pat::Byte(b'b'), 1.0),
                (Pat::Byte(b'c'), 1.0), (Pat::Byte(b'd'), 1.0),
                (Pat::Byte(b'e'), 1.0), (Pat::Byte(b'f'), 1.0),
            ])),
            3,
        );
        let phrase = Pat::Seq(vec![
            word(),
            Pat::Byte(b' '),
            word(),
            Pat::Choice(vec![
                (Pat::Byte(b'.'), 1.0),
                (Pat::Byte(b'?'), 0.5),
                (Pat::Byte(b'!'), 0.5),
            ]),
        ]);
        Ptgrs { start: phrase, seed: 0 }
    }

    pub fn generate(&self, seed: u64) -> Vec<u8> {
        let mut rng = Rng::new(seed ^ self.seed);
        let mut out = vec![];
        self.start.expand(&mut rng, &mut out);
        out
    }
}