use crate::graph::Rng;

pub struct Cause {
    pub depth: usize, pub rules: Vec<usize>,
    pub skeleton: Vec<(usize, usize)>, pub spans: Vec<(usize, usize)>,
}

struct Tg { labels: Vec<u8>, children: Vec<Vec<usize>> }

impl Tg {
    fn roots(&self) -> Vec<usize> {
        let mut is_child = vec![false; self.labels.len()];
        for ch in &self.children { for &c in ch { is_child[c] = true; } }
        (0..self.labels.len()).filter(|&i| !is_child[i]).collect()
    }
    fn leaves(&self) -> Vec<usize> {
        (0..self.labels.len()).filter(|&i| self.children[i].is_empty()).collect()
    }
}

fn pick_rule(rng: &mut Rng) -> usize {
    let r = rng.next_f32();
    if r < 0.40 { 0 } else if r < 0.65 { 1 } else if r < 0.85 { 2 } else { 3 }
}

fn apply(tg: &mut Tg, rule: usize, rng: &mut Rng) {
    match rule {
        0 => {
            let lv = tg.leaves(); if lv.is_empty() { return; }
            let n = lv[rng.below(lv.len())]; let l = tg.labels[n];
            let c1 = tg.labels.len(); tg.labels.push((l + 1) % 16); tg.children.push(Vec::new());
            let c2 = tg.labels.len(); tg.labels.push((l + 2) % 16); tg.children.push(Vec::new());
            tg.children[n] = vec![c1, c2];
        }
        1 => {
            let rt = tg.roots(); if rt.is_empty() { return; }
            let r = rt[rng.below(rt.len())]; let l = tg.labels[r];
            tg.labels.push((l + 3) % 16); tg.children.push(vec![r]);
        }
        2 => {
            let lv = tg.leaves(); if lv.is_empty() { return; }
            let l = lv[rng.below(lv.len())];
            let cand: Vec<usize> = (0..tg.labels.len())
                .filter(|&i| i != l && tg.children[i].len() < 2 && !tg.children[i].contains(&l)).collect();
            if cand.is_empty() { return; }
            let n = cand[rng.below(cand.len())]; tg.children[n].push(l);
        }
        _ => { let n = rng.below(tg.labels.len()); tg.labels[n] = (tg.labels[n] + 5) % 16; }
    }
}

// token = préfixe + lettre: "xxA".."xxP" / "yyA".."yyP".
// Disjoints MÊME minuscules (xxa != yya), longueur 3 -> survivent au filtre mots.
fn render(tg: &Tg, n: usize, prefix: &[u8], out: &mut Vec<u8>, spans: &mut Vec<(usize, usize)>, depth: usize) {
    if depth > 24 { return; }
    let start = out.len();
    out.extend_from_slice(prefix);
    out.push(b'A' + tg.labels[n]);
    if !tg.children[n].is_empty() {
        out.push(b' '); out.push(b'(');
        for &c in &tg.children[n] { out.push(b' '); render(tg, c, prefix, out, spans, depth + 1); }
        out.push(b' '); out.push(b')');
    }
    spans[n] = (start, out.len());
}

pub fn generate(rng: &mut Rng, depth: usize) -> (Vec<u8>, Cause) {
    let mut tg = Tg { labels: vec![0], children: vec![Vec::new()] };
    let mut rules = Vec::new();
    for _ in 0..depth { let r = pick_rule(rng); apply(&mut tg, r, rng); rules.push(r); }
    let mut skeleton = Vec::new();
    for (i, ch) in tg.children.iter().enumerate() { for &c in ch { skeleton.push((i, c)); } }

    let roots = tg.roots();
    let mut a = Vec::new(); let mut spans_a = vec![(0usize, 0usize); tg.labels.len()];
    for &r in &roots { render(&tg, r, b"xx", &mut a, &mut spans_a, 0); }
    let mut b = Vec::new(); let mut spans_b = vec![(0usize, 0usize); tg.labels.len()];
    for &r in &roots { render(&tg, r, b"yy", &mut b, &mut spans_b, 0); }

    // question = bloc séparé sondant la racine (vocabulaire instance)
    let root_label = tg.labels[roots[0]];
    let question = format!("xx{} ?", (b'A' + root_label) as char);

    let off_b = a.len() + 3;
    let mut e = a;
    e.extend_from_slice(b" | ");
    e.extend_from_slice(&b);
    e.extend_from_slice(b"\n\n");
    let off_q = e.len();
    e.extend_from_slice(question.as_bytes());

    let mut spans = spans_a;
    for (s, t) in spans_b { spans.push((s + off_b, t + off_b)); }
    spans.push((off_q, off_q + question.len()));
    (e, Cause { depth, rules, skeleton, spans })
}