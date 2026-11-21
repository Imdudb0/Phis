use crate::graph::*;

#[derive(Clone)]
pub struct Assembly {
    pub span: (usize, usize),
    pub words: Vec<Vec<u8>>,
    pub kind: u8,                 // 0 def, 1 head, 2 question
    pub head: Option<usize>,
    pub ws: Vec<S>,
}

pub struct Elig(pub Vec<[f32; D * D]>);

pub struct SearchResult {
    pub assemblies: Vec<Assembly>,
    pub b: Vec<Vec<f32>>,         // cause: liaisons dirigées non-réciproques sur assemblées
    pub drive: Vec<f32>,
    pub probe: usize,
    pub fidelity: f32, pub phi: f32, pub integration: f32, pub specificity: f32,
    pub residual: f32, pub steps: usize, pub ops: u64,
}

pub fn words_of(e: &[u8], s: usize, t: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new(); let mut cur = Vec::new();
    for &b in &e[s..t] {
        if b.is_ascii_alphabetic() { cur.push(b.to_ascii_lowercase()); }
        else if cur.len() > 2 { out.push(std::mem::take(&mut cur)); }
        else { cur.clear(); }
    }
    if cur.len() > 2 { out.push(cur); }
    out.sort(); out.dedup(); out
}

fn push_asm(asms: &mut Vec<Assembly>, e: &[u8], s: usize, t: usize, kind: u8, head: Option<usize>) {
    if t <= s { return; }
    asms.push(Assembly { span: (s, t), words: words_of(e, s, t), kind, head, ws: Vec::new() });
}

// Formation générique par la ponctuation (octets, aucune donnée de domaine).
pub fn form_assemblies(e: &[u8]) -> Vec<Assembly> {
    let mut asms = Vec::new();
    let mut blocks = Vec::new();
    let mut s = 0usize; let mut i = 0usize;
    while i < e.len() {
        let dbl = i + 1 < e.len() && e[i] == b'\n' && e[i + 1] == b'\n';
        let pipe = i + 2 < e.len() && e[i] == b' ' && e[i + 1] == b'|' && e[i + 2] == b' ';
        if dbl || pipe {
            blocks.push((s, i)); i += if dbl { 2 } else { 3 }; s = i;
        } else { i += 1; }
    }
    blocks.push((s, e.len()));
    for (bs, bt) in blocks {
        let mut a = bs; let mut b = bt;
        while a < b && (e[a] as char).is_whitespace() { a += 1; }
        while b > a && (e[b - 1] as char).is_whitespace() { b -= 1; }
        if b <= a { continue; }
        if e[b - 1] == b'?' { push_asm(&mut asms, e, a, b, 2, None); continue; }
        let cut = (a..b).find(|&k| e[k] == b',' || e[k] == b'(');
        match cut {
            Some(c) => {
                let head_idx = asms.len();
                push_asm(&mut asms, e, a, c, 1, None);
                let mut cs = c + 1;
                for k in c + 1..=b {
                    if k == b || e[k] == b'.' {
                        push_asm(&mut asms, e, cs, k, 0, Some(head_idx));
                        cs = k + 1;
                    }
                }
            }
            None => push_asm(&mut asms, e, a, b, 0, None),
        }
    }
    if asms.is_empty() { push_asm(&mut asms, e, 0, e.len(), 0, None); }
    asms
}

fn jaccard(a: &[Vec<u8>], b: &[Vec<u8>]) -> f32 {
    if a.is_empty() || b.is_empty() { return 0.0; }
    let mut inter = 0;
    for w in a { if b.contains(w) { inter += 1; } }
    let union = a.len() + b.len() - inter;
    inter as f32 / union as f32
}

// Structure de E (l'effet à régénérer): overlap + adjacence définitionnelle + séquence.
pub fn effect_structure(asms: &[Assembly]) -> Vec<Vec<f32>> {
    let n = asms.len();
    let mut t = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            t[i][j] = jaccard(&asms[i].words, &asms[j].words);
        }
        if i + 1 < n { t[i][i + 1] = t[i][i + 1].max(0.15); }
        if asms[i].kind == 1 {
            for j in 0..n {
                if asms[j].head == Some(i) {
                    t[i][j] = t[i][j].max(0.5);
                    t[j][i] = t[j][i].max(0.2);
                }
            }
        }
    }
    t
}

pub fn drive(b: &[Vec<f32>], probe: usize, n: usize) -> Vec<f32> {
    let mut a = vec![0.0; n];
    if probe < n { a[probe] = 1.0; }
    for _ in 0..3 {
        let mut na = vec![0.0; n];
        for j in 0..n { for i in 0..n { na[j] += b[i][j] * a[i]; } }
        if probe < n { na[probe] = 1.0; }
        let mx = na.iter().cloned().fold(0.0f32, f32::max).max(1e-9);
        for v in na.iter_mut() { *v /= mx; }
        a = na;
    }
    a
}

fn fill_ws(g: &Graph, e: &[u8], asms: &mut [Assembly], rng: &mut Rng, budget: &mut Budget) {
    let gain = vec![1.0f32; N];
    for a in asms.iter_mut() {
        let mut x = zero_state();
        let mut input = zero_state();
        let end = a.span.1.min(a.span.0 + 24);
        for p in a.span.0..end {
            let b = e[p] as usize;
            input[b][0] = 1.0;
            x = step_full(g, &x, &input, &gain, 0.0, rng, budget, None);
            input[b][0] = 0.0;
        }
        a.ws = x[WS0..C0].to_vec();
    }
}

// Recherche abductive: la cause CROÎT pour expliquer E (meilleure = fidélité max).
// phi MESURE la cause découverte; il n'est pas le critère de croissance.
pub fn search(g: &Graph, e: &[u8], max_steps: usize, budget: &mut Budget, rng: &mut Rng,
              explore_gain: f32, mut elig: Option<&mut Elig>) -> SearchResult {
    let mut assemblies = form_assemblies(e);
    fill_ws(g, e, &mut assemblies, rng, budget);
    let n = assemblies.len();
    let t = effect_structure(&assemblies);
    let probe = assemblies.iter().position(|a| a.kind == 2).unwrap_or(0);

    let mut b = vec![vec![0.0; n]; n];
    for j in 0..n { if t[probe][j] > 0.0 { b[probe][j] = t[probe][j]; } }
    let mut drv = drive(&b, probe, n);
    let mut best_fid = -1.0f32;
    let mut best_b = b.clone(); let mut best_drv = drv.clone();
    let mut best = (0.0f32, 0.0f32, 0.0f32, 0.0f32); // fid, integ, spec, phi au meilleur fid
    let mut phi_ema = 0.0f32; let mut stall = 0usize; let mut steps = 0usize;
    let mut threshold = 0.005f32;
    
    let effective_steps = max_steps.max(8 * n).min(2048);
    for step in 0..effective_steps { 
        if budget.ops >= budget.max || n == 0 { break; }
        steps = step + 1;
        let (fid, integ, spec, phi) = crate::phi::compute(&b, &t, &drv, probe, n, budget);
        if fid > best_fid {
            best_fid = fid; best_b = b.clone(); best_drv = drv.clone();
            best = (fid, integ, spec, phi);
        }
        let reward = phi - phi_ema;
        phi_ema = 0.8 * phi_ema + 0.2 * phi;

        if let Some(el) = elig.as_deref_mut() {
            let r = reward.max(0.0);
            for (ei, ed) in g.edges.iter().enumerate() {
                if ed.kind.is_fast() { continue; }
                let o = outer(&g.bias[ed.src as usize], &g.bias[ed.dst as usize]);
                for k in 0..D * D { el.0[ei][k] += o[k] * r; }
            }
        }
        for i in 0..n { for j in 0..n {
            if b[i][j] > 0.0 {
                b[i][j] = (b[i][j] + 0.05 * reward * drv[i] * drv[j]).clamp(0.0, 1.0);
            }
        }}
        // abduction créative: lier la structure manquante la plus atteignable (résidu)
        let mut reach = vec![0.0f32; n];
        for i in 0..n { if drv[i] > 0.03 {
            for j in 0..n { reach[j] = reach[j].max(t[i][j] * drv[i]); }
        }}
        let mut pick = None; let mut bv = threshold;
        for i in 0..n { for j in 0..n {
            if t[i][j] > 0.0 && b[i][j] == 0.0 && reach[j] > 0.0 {
                let v = t[i][j] * reach[j];
                if v > bv { bv = v; pick = Some((i, j)); }
            }
        }}
        match pick {
            Some((i, j)) => { b[i][j] = t[i][j]; stall = 0; threshold = 0.005; }
            None => { stall += 1; threshold *= 0.6 / explore_gain.max(0.3); }
        }
        drv = drive(&b, probe, n);
        if stall > 6 { break; }
    }
    let residual = crate::phi::residual(&best_b, &t, &best_drv);
    SearchResult {
        assemblies, b: best_b, drive: best_drv, probe,
        fidelity: best.0, phi: best.3, integration: best.1, specificity: best.2,
        residual, steps, ops: budget.ops,
    }
}