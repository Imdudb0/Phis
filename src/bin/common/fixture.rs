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