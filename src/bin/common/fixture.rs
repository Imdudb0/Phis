const WORDS: [&str; 3] = ["ambiguity", "univocity", "redundancy"];

fn pools(hi: bool) -> Vec<Vec<u8>> {
    if hi {
        (0..4)
            .map(|t| {
                let lo = (0x80 + t * 0x10) as u8;
                (lo..lo + 12).collect()
            })
            .collect()
    } else {
        let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; // 36 = 4 x 9
        (0..4).map(|t| chars[t * 9..(t + 1) * 9].to_vec()).collect()
    }
}

fn motif(kind: usize, u: &[u8], v1: &[u8], v2: &[u8], w: &[u8], sep: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    match kind {
        0 => {
            out.extend_from_slice(u);
            out.extend_from_slice(v1);
            out.extend_from_slice(sep);
            out.extend_from_slice(u);
            out.extend_from_slice(v2);
        }
        1 => {
            out.extend_from_slice(u);
            out.extend_from_slice(v1);
            out.extend_from_slice(sep);
            out.extend_from_slice(u);
            out.extend_from_slice(v1);
        }
        _ => {
            out.extend_from_slice(u);
            out.extend_from_slice(v1);
            out.extend_from_slice(sep);
            out.extend_from_slice(w);
            out.extend_from_slice(v1);
        }
    }
    out
}

fn fixture(
    names: [&str; 3],
    hi: bool,
    seed: u64,
    drop_second: bool,
    degenerate: bool,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut rng = PyRandom::new(seed);
    let ps = pools(hi);

    let mut bodies = Vec::with_capacity(4);
    for t in 0..4 {
        let p = &ps[t];
        let sep = [*p.last().unwrap()];
        let (u, v1, v2, w) = if degenerate {
            let mut u = tok(&mut rng, p);
            let mut v1 = tok(&mut rng, p);
            let mut v2 = tok(&mut rng, p);
            let mut w = tok(&mut rng, p);
            loop {
                let mut uniq = HashSet::new();
                uniq.insert(u.clone());
                uniq.insert(v1.clone());
                uniq.insert(v2.clone());
                uniq.insert(w.clone());
                if uniq.len() >= 4 {
                    break;
                }
                u = tok(&mut rng, p);
                v1 = tok(&mut rng, p);
                v2 = tok(&mut rng, p);
                w = tok(&mut rng, p);
            }
            (u, v1, v2, w)
        } else {
            let mut avail: Vec<u8> = p.iter().copied().filter(|&b| b != sep[0]).collect();
            rng.shuffle(&mut avail);
            let eight = &avail[..8];
            (
                eight[0..2].to_vec(),
                eight[2..4].to_vec(),
                eight[4..6].to_vec(),
                eight[6..8].to_vec(),
            )
        };

        let kind = if t < 3 { t } else { 0 };
        let mut b = motif(kind, &u, &v1, &v2, &w, &sep);
        if t == 3 && drop_second {
            b = Vec::new();
            b.extend_from_slice(&u);
            b.extend_from_slice(&v1);
            b.extend_from_slice(&sep);
            b.extend_from_slice(&v2);
        }
        bodies.push(b);
    }

    let mut out = Vec::new();
    for t in 0..3 {
        out.extend_from_slice(b"the cause of ");
        out.extend_from_slice(&bodies[t]);
        out.extend_from_slice(b" is ");
        out.extend_from_slice(names[t].as_bytes());
        out.extend_from_slice(b".\n");
    }
    out.extend_from_slice(b"the cause of ");
    out.extend_from_slice(&bodies[3]);
    out.extend_from_slice(b" is ?.\n");
    (out, bodies)
}