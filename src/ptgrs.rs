//! PTGRS probabiliste + sérialisation multi-codec aléatoire. Entraînement.

use crate::algebra::{values, Builder, Cause, Node, IDENT, Op};
use crate::rng::PyRandom;

pub fn random_key(rng: &mut PyRandom, alpha: &[u8]) -> [u8; 256] {
    let mut tgt = alpha.to_vec();
    rng.shuffle(&mut tgt);
    let mut key = IDENT;
    for (&a, &b) in alpha.iter().zip(tgt.iter()) {
        key[a as usize] = b;
    }
    key
}

pub fn sample_cause(rng: &mut PyRandom, alpha: &[u8], depth: usize) -> Cause {
    fn grow(d: usize, rng: &mut PyRandom, alpha: &[u8], b: &mut Builder) -> usize {
        if d == 0 || rng.random() < 0.35 {
            let l = rng.randint(1, 3);
            let mut lit = Vec::with_capacity(l);
            for _ in 0..l {
                lit.push(*rng.choice(alpha));
            }
            return b.add(Node::lit(lit));
        }
        let ops = [Op::Seq, Op::Rep, Op::Alt, Op::Rec];
        let weights = [0.45, 0.20, 0.15, 0.20];
        let op = *rng.choices_one(&ops, &weights);
        match op {
            Op::Seq => {
                let n_kids = rng.randint(2, 3);
                let mut kids = Vec::with_capacity(n_kids);
                for _ in 0..n_kids {
                    kids.push(grow(d - 1, rng, alpha, b));
                }
                b.add(Node::seq(kids))
            }
            Op::Rep => {
                let child = grow(d - 1, rng, alpha, b);
                let k = rng.randint(2, 3);
                b.add(Node::rep(child, k))
            }
            Op::Alt => {
                let k0 = grow(d - 1, rng, alpha, b);
                let k1 = grow(d - 1, rng, alpha, b);
                let sel = rng.randrange(2);
                b.add(Node::alt(vec![k0, k1], sel))
            }
            Op::Rec => {
                let child = grow(d - 1, rng, alpha, b);
                let key = random_key(rng, alpha);
                b.add(Node::rec(child, key))
            }
            _ => unreachable!(),
        }
    }

    let mut b = Builder::new(true);
    let mut root = grow(depth, rng, alpha, &mut b);
    if rng.random() < 0.5 {
        let key = random_key(rng, alpha);
        root = b.add(Node::rec(root, key));
    }
    b.cause(root)
}

pub fn episode_harder(rng: &mut PyRandom, tries: usize, level: usize) -> (Vec<u8>, Cause) {
    let lv = level.min(3);
    for _ in 0..tries {
        let base = rng.randrange_range(32, 160);
        let alpha: Vec<u8> = (base..base + rng.randint(10, 18)).map(|x| x as u8).collect();
        let c = sample_cause(rng, &alpha, 4 + lv);
        let e = values(&c, None)[c.root].clone();
        if (60..=140 + 60 * lv).contains(&e.len()) {
            let mut seen = [false; 256]; let mut uniq = 0;
            for &b in &e { if !seen[b as usize] { seen[b as usize] = true; uniq += 1; } }
            if uniq >= 8 { return (e, c); }
        }
    }
    episode_hard(rng, 60)
}

pub fn episode_hard(rng: &mut PyRandom, tries: usize) -> (Vec<u8>, Cause) {
    for _ in 0..tries {
        let base = rng.randrange_range(32, 180);
        let alpha_len = rng.randint(8, 16);
        let alpha: Vec<u8> = (base..base + alpha_len).map(|x| x as u8).collect();
        let depth = rng.randint(3, 4);
        let c = sample_cause(rng, &alpha, depth);
        let e = values(&c, None)[c.root].clone();
        if (40..=140).contains(&e.len()) {
            let mut seen = [false; 256]; let mut uniq = 0usize;
            for &b in &e { if !seen[b as usize] { seen[b as usize] = true; uniq += 1; } }
            if uniq >= 6 { return (e, c); }
        }
    }
    episode(rng, 60)
}

pub fn episode(rng: &mut PyRandom, tries: usize) -> (Vec<u8>, Cause) {
    let base = rng.randrange_range(32, 200);
    for _ in 0..tries {
        let alpha_len = rng.randint(6, 12);
        let alpha: Vec<u8> = (base..base + alpha_len).map(|x| x as u8).collect();
        let depth = rng.randint(2, 3);
        let c = sample_cause(rng, &alpha, depth);
        let e = values(&c, None)[c.root].clone();
        if (6..=48).contains(&e.len()) {
            let mut seen = [false; 256];
            let mut uniq = 0usize;
            for &byte in &e {
                if !seen[byte as usize] {
                    seen[byte as usize] = true;
                    uniq += 1;
                }
            }
            if uniq >= 3 {
                return (e, c);
            }
        }
    }
    let alpha: Vec<u8> = (base..base + 8).map(|x| x as u8).collect();
    let depth = rng.randint(2, 3);
    let c = sample_cause(rng, &alpha, depth);
    let e = values(&c, None)[c.root].clone();
    (e, c)
}